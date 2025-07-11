#pragma once
#include <raft/core/operators.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/neighbors/detail/faiss_select/Select.cuh>
#include <raft/util/cuda_utils.cuh>

#include <cub/cub.cuh>

#include <limits>

#include "ffanns/distance/distance.hpp"
#include "../../distance/detail/distance.cuh"
#include "../../distance/detail/distance_ops/l2_exp.cuh"
// #include "../../distance/detail/distance_ops/l2_unexp.cuh"
#include "../../distance/detail/pairwise_distance_base.cuh"
#include "../../distance/distance.cuh"

namespace ffanns::neighbors::detail {

template <typename Policy, typename Pair, typename myWarpSelect, typename IdxT>
DI void loadAllWarpQShmem(myWarpSelect** heapArr,
                          Pair* shDumpKV,
                          const IdxT m,
                          const unsigned int numOfNN)
{
  const int lid = raft::laneId();
#pragma unroll
  for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
    const auto rowId = (threadIdx.x / Policy::AccThCols) + i * Policy::AccThRows;
    if (rowId < m) {
#pragma unroll
      for (int j = 0; j < myWarpSelect::kNumWarpQRegisters; ++j) {
        const int idx = j * warpSize + lid;
        if (idx < numOfNN) {
          Pair KVPair          = shDumpKV[rowId * numOfNN + idx];
          heapArr[i]->warpV[j] = KVPair.key;
          heapArr[i]->warpK[j] = KVPair.value;
        }
      }
    }
  }
}

template <typename Policy, typename Pair, typename myWarpSelect>
DI void loadWarpQShmem(myWarpSelect* heapArr,
                       Pair* shDumpKV,
                       const int rowId,
                       const unsigned int numOfNN)
{
  const int lid = raft::laneId();
#pragma unroll
  for (int j = 0; j < myWarpSelect::kNumWarpQRegisters; ++j) {
    const int idx = j * warpSize + lid;
    if (idx < numOfNN) {
      Pair KVPair       = shDumpKV[rowId * numOfNN + idx];
      heapArr->warpV[j] = KVPair.key;
      heapArr->warpK[j] = KVPair.value;
    }
  }
}

template <typename Policy, typename Pair, typename myWarpSelect, typename IdxT>
DI void storeWarpQShmem(myWarpSelect* heapArr,
                        Pair* shDumpKV,
                        const IdxT rowId,
                        const unsigned int numOfNN)
{
  const int lid = raft::laneId();

#pragma unroll
  for (int j = 0; j < myWarpSelect::kNumWarpQRegisters; ++j) {
    const int idx = j * warpSize + lid;
    if (idx < numOfNN) {
      Pair otherKV                    = Pair(heapArr->warpV[j], heapArr->warpK[j]);
      shDumpKV[rowId * numOfNN + idx] = otherKV;
    }
  }
}

template <typename Policy, typename Pair, typename myWarpSelect, typename IdxT, typename OutT>
DI void storeWarpQGmem(myWarpSelect** heapArr,
                       volatile OutT* out_dists,
                       volatile IdxT* out_inds,
                       const IdxT m,
                       const unsigned int numOfNN,
                       const IdxT starty)
{
  const int lid = raft::laneId();
#pragma unroll
  for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
    const auto gmemRowId = starty + i * Policy::AccThRows;
    if (gmemRowId < m) {
#pragma unroll
      for (int j = 0; j < myWarpSelect::kNumWarpQRegisters; ++j) {
        const auto idx = j * warpSize + lid;
        if (idx < numOfNN) {
          out_dists[std::size_t(gmemRowId) * numOfNN + idx] = heapArr[i]->warpK[j];
          out_inds[std::size_t(gmemRowId) * numOfNN + idx]  = (IdxT)heapArr[i]->warpV[j];
        }
      }
    }
  }
}

template <typename Policy, typename Pair, typename myWarpSelect, typename IdxT, typename OutT>
DI void loadPrevTopKsGmemWarpQ(myWarpSelect** heapArr,
                               volatile OutT* out_dists,
                               volatile IdxT* out_inds,
                               const IdxT m,
                               const unsigned int numOfNN,
                               const IdxT starty)
{
  const int lid = raft::laneId();
#pragma unroll
  for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
    const auto gmemRowId = starty + i * Policy::AccThRows;
    if (gmemRowId < m) {
#pragma unroll
      for (int j = 0; j < myWarpSelect::kNumWarpQRegisters; ++j) {
        const auto idx = j * warpSize + lid;
        if (idx < numOfNN) {
          heapArr[i]->warpK[j] = out_dists[std::size_t(gmemRowId) * numOfNN + idx];
          heapArr[i]->warpV[j] = (uint32_t)out_inds[std::size_t(gmemRowId) * numOfNN + idx];
        }
      }
      static constexpr auto kLaneWarpKTop = myWarpSelect::kNumWarpQRegisters - 1;
      heapArr[i]->warpKTop = raft::shfl(heapArr[i]->warpK[kLaneWarpKTop], heapArr[i]->kLane);
    }
  }
}

template <typename Pair, int NumWarpQRegs, typename myWarpSelect>
DI void updateSortedWarpQ(
  myWarpSelect& heapArr, Pair* allWarpTopKs, int rowId, int finalNumVals, int startId = 0)
{
  constexpr uint32_t mask = 0xffffffffu;
  const int lid           = raft::laneId();
  // calculate srcLane such that tid 0 -> 31, 1 -> 0,... 31 -> 30.
  // warp around 0 to 31 required for NN > 32
  const auto srcLane = (warpSize + (lid - 1)) & (warpSize - 1);

  for (int k = startId; k < finalNumVals; k++) {
    Pair KVPair = allWarpTopKs[rowId * (256) + k];
#pragma unroll
    for (int i = 0; i < NumWarpQRegs; i++) {
      unsigned activeLanes = __ballot_sync(mask, KVPair.value < heapArr->warpK[i]);
      if (activeLanes) {
        Pair tempKV;
        tempKV.value               = raft::shfl(heapArr->warpK[i], srcLane);
        tempKV.key                 = raft::shfl(heapArr->warpV[i], srcLane);
        const auto firstActiveLane = __ffs(activeLanes) - 1;
        if (firstActiveLane == lid) {
          heapArr->warpK[i] = KVPair.value;
          heapArr->warpV[i] = KVPair.key;
        } else if (lid > firstActiveLane) {
          heapArr->warpK[i] = tempKV.value;
          heapArr->warpV[i] = tempKV.key;
        }
        if (i == 0 && NumWarpQRegs > 1) {
          heapArr->warpK[1] = __shfl_up_sync(mask, heapArr->warpK[1], 1);
          heapArr->warpV[1] = __shfl_up_sync(mask, heapArr->warpV[1], 1);
          if (lid == 0) {
            heapArr->warpK[1] = tempKV.value;
            heapArr->warpV[1] = tempKV.key;
          }
          break;
        }
      }
    }
  }
}
  
template <typename DataT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename OpT,
          typename FinalLambda,
          int NumWarpQ,
          int NumThreadQ,
          bool usePrevTopKs = false,
          bool isRowMajor   = true>
__launch_bounds__(Policy::Nthreads, 2) RAFT_KERNEL fusedL2kNN(const DataT* x,
                                                              const DataT* y,
                                                              const OutT* _xn,
                                                              const OutT* _yn,
                                                              const IdxT m,
                                                              const IdxT n,
                                                              const IdxT k,
                                                              const IdxT lda,
                                                              const IdxT ldb,
                                                              const IdxT ldd,
                                                              OpT distance_op,
                                                              FinalLambda fin_op,
                                                              unsigned int numOfNN,
                                                              volatile int* mutexes,
                                                              volatile OutT* out_dists,
                                                              volatile IdxT* out_inds)
{
  using AccT = typename OpT::AccT;
  extern __shared__ char smem[];

  typedef cub::KeyValuePair<uint32_t, AccT> Pair;
  constexpr auto identity = std::numeric_limits<AccT>::max();
  constexpr auto keyMax   = std::numeric_limits<uint32_t>::max();
  constexpr auto Dir      = false;
  using namespace raft::neighbors::detail::faiss_select;
  typedef WarpSelect<AccT, uint32_t, Dir, Comparator<AccT>, NumWarpQ, NumThreadQ, 32> myWarpSelect;

  auto rowEpilog_lambda =
    [m, n, &distance_op, numOfNN, out_dists, out_inds, mutexes] __device__(IdxT gridStrideY) {
      if (gridDim.x == 1) { return; }

      // Use ::template to disambiguate (See:
      // https://en.cppreference.com/w/cpp/language/dependent_name)
      int smem_offset = OpT::template shared_mem_size<Policy>();
      Pair* shDumpKV  = (Pair*)(&smem[smem_offset]);

      const int lid     = threadIdx.x % warpSize;
      const IdxT starty = gridStrideY + (threadIdx.x / Policy::AccThCols);

      //  0 -> consumer done consuming the buffer.
      // -1 -> consumer started consuming the buffer
      // -2 -> producer done filling the buffer
      //  1 -> prod acquired to fill the buffer
      if (blockIdx.x == 0) {
        auto cta_processed = 0;
        myWarpSelect heapArr1(identity, keyMax, numOfNN);
        myWarpSelect heapArr2(identity, keyMax, numOfNN);
        myWarpSelect* heapArr[] = {&heapArr1, &heapArr2};
        __syncwarp();

        loadAllWarpQShmem<Policy, Pair>(heapArr, &shDumpKV[0], m, numOfNN);

        while (cta_processed < gridDim.x - 1) {
          if (threadIdx.x == 0) {
            while (atomicCAS((int*)&mutexes[gridStrideY / Policy::Mblk], -2, -1) != -2)
              ;
          }
          __threadfence();
          __syncthreads();

#pragma unroll
          for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
            const auto rowId = starty + i * Policy::AccThRows;
            if (rowId < m) {
#pragma unroll
              for (int j = 0; j < myWarpSelect::kNumWarpQRegisters; ++j) {
                Pair otherKV;
                otherKV.value  = identity;
                otherKV.key    = keyMax;
                const auto idx = j * warpSize + lid;
                if (idx < numOfNN) {
                  otherKV.value         = out_dists[rowId * numOfNN + idx];
                  otherKV.key           = (uint32_t)out_inds[rowId * numOfNN + idx];
                  const auto shMemRowId = (threadIdx.x / Policy::AccThCols) + i * Policy::AccThRows;
                  shDumpKV[shMemRowId * numOfNN + idx] = otherKV;
                }
              }
            }
          }
          __threadfence();
          __syncthreads();

          if (threadIdx.x == 0) { atomicExch((int*)&mutexes[gridStrideY / Policy::Mblk], 0); }
          __threadfence();

        // Perform merging of otherKV with topk's across warp.
#pragma unroll
          for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
            const auto rowId = starty + i * Policy::AccThRows;
            if (rowId < m) {
#pragma unroll
              for (int j = 0; j < myWarpSelect::kNumWarpQRegisters; ++j) {
                Pair otherKV;
                otherKV.value  = identity;
                otherKV.key    = keyMax;
                const auto idx = j * warpSize + lid;
                if (idx < numOfNN) {
                  const auto shMemRowId = (threadIdx.x / Policy::AccThCols) + i * Policy::AccThRows;
                  otherKV               = shDumpKV[shMemRowId * numOfNN + idx];
                }
                heapArr[i]->add(otherKV.value, otherKV.key);
              }
            }
          }
          cta_processed++;
        }
#pragma unroll
        for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
          const auto rowId = starty + i * Policy::AccThRows;
          if (rowId < m) {
            bool needSort = (heapArr[i]->numVals > 0);
            needSort      = __any_sync(0xffffffff, needSort);
            if (needSort) { heapArr[i]->reduce(); }
          }
        }
        storeWarpQGmem<Policy, Pair>(heapArr, out_dists, out_inds, m, numOfNN, starty);
      } else {
        if (threadIdx.x == 0) {
          while (atomicCAS((int*)&mutexes[gridStrideY / Policy::Mblk], 0, 1) != 0)
            ;
        }
        __threadfence();
        __syncthreads();

#pragma unroll
        for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
          const auto rowId = starty + i * Policy::AccThRows;
          if (rowId < m) {
            for (int idx = lid; idx < numOfNN; idx += warpSize) {
              const auto shMemRowId = (threadIdx.x / Policy::AccThCols) + i * Policy::AccThRows;
              Pair KVPair           = shDumpKV[shMemRowId * numOfNN + idx];
              out_dists[rowId * numOfNN + idx] = KVPair.value;
              out_inds[rowId * numOfNN + idx]  = (IdxT)KVPair.key;
            }
          }
        }
        __threadfence();
        __syncthreads();

        if (threadIdx.x == 0) { atomicExch((int*)&mutexes[gridStrideY / Policy::Mblk], -2); }
        __threadfence();
      }
    };

  // epilogue operation lambda for final value calculation
  auto epilog_lambda =
    [&distance_op, numOfNN, m, n, ldd, out_dists, out_inds, keyMax, identity] __device__(
      AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
      OutT * regxn,
      OutT * regyn,
      IdxT gridStrideX,
      IdxT gridStrideY) {
      // Use ::template to disambiguate (See:
      // https://en.cppreference.com/w/cpp/language/dependent_name)
      int smem_offset = OpT::template shared_mem_size<Policy>();
      Pair* shDumpKV  = (Pair*)(&smem[smem_offset]);

      constexpr uint32_t mask = 0xffffffffu;
      const IdxT starty       = gridStrideY + (threadIdx.x / Policy::AccThCols);
      const IdxT startx       = gridStrideX + (threadIdx.x % Policy::AccThCols);
      const int lid           = raft::laneId();

      myWarpSelect heapArr1(identity, keyMax, numOfNN);
      myWarpSelect heapArr2(identity, keyMax, numOfNN);
      myWarpSelect* heapArr[] = {&heapArr1, &heapArr2};
      if (usePrevTopKs) {
        if (gridStrideX == blockIdx.x * Policy::Nblk) {
          loadPrevTopKsGmemWarpQ<Policy, Pair>(heapArr, out_dists, out_inds, m, numOfNN, starty);
        }
      }

      if (gridStrideX > blockIdx.x * Policy::Nblk) {
#pragma unroll
        for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
          const auto rowId     = (threadIdx.x / Policy::AccThCols) + i * Policy::AccThRows;
          Pair tempKV          = shDumpKV[(rowId * numOfNN) + numOfNN - 1];
          heapArr[i]->warpKTop = tempKV.value;
        }

        // total vals can atmost be 256, (32*8)
        int numValsWarpTopK[Policy::AccRowsPerTh];
        int anyWarpTopKs = 0;
#pragma unroll
        for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
          const auto rowId   = starty + i * Policy::AccThRows;
          numValsWarpTopK[i] = 0;
          if (rowId < m) {
#pragma unroll
            for (int j = 0; j < Policy::AccColsPerTh; ++j) {
              const auto colId = startx + j * Policy::AccThCols;
              if (colId < ldd) {
                if (acc[i][j] < heapArr[i]->warpKTop) { numValsWarpTopK[i]++; }
              }
            }
            anyWarpTopKs += numValsWarpTopK[i];
          }
        }
        anyWarpTopKs = __syncthreads_or(anyWarpTopKs > 0);
        if (anyWarpTopKs) {
          Pair* allWarpTopKs = (Pair*)(&smem[0]);
          uint32_t needScanSort[Policy::AccRowsPerTh];

#pragma unroll
          for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
            const auto gmemRowId = starty + i * Policy::AccThRows;
            needScanSort[i]      = 0;
            if (gmemRowId < m) {
              int myVals      = numValsWarpTopK[i];
              needScanSort[i] = __ballot_sync(mask, myVals > 0);
              if (needScanSort[i]) {
#pragma unroll
                for (unsigned int k = 1; k <= 16; k *= 2) {
                  const unsigned int n = __shfl_up_sync(mask, numValsWarpTopK[i], k);
                  if (lid >= k) { numValsWarpTopK[i] += n; }
                }
              }
              // As each thread will know its total vals to write.
              // we only store its starting location.
              numValsWarpTopK[i] -= myVals;
            }

            if (needScanSort[i]) {
              const auto rowId = (threadIdx.x / Policy::AccThCols) + i * Policy::AccThRows;
              if (gmemRowId < m) {
                if (needScanSort[i] & ((uint32_t)1 << lid)) {
#pragma unroll
                  for (int j = 0; j < Policy::AccColsPerTh; ++j) {
                    const auto colId = startx + j * Policy::AccThCols;
                    if (colId < ldd) {
                      if (acc[i][j] < heapArr[i]->warpKTop) {
                        Pair otherKV                                     = {colId, acc[i][j]};
                        allWarpTopKs[rowId * (256) + numValsWarpTopK[i]] = otherKV;
                        numValsWarpTopK[i]++;
                      }
                    }
                  }
                }
                __syncwarp();
                const int finalNumVals = raft::shfl(numValsWarpTopK[i], 31);
                loadWarpQShmem<Policy, Pair>(heapArr[i], &shDumpKV[0], rowId, numOfNN);
                updateSortedWarpQ<Pair, myWarpSelect::kNumWarpQRegisters>(
                  heapArr[i], &allWarpTopKs[0], rowId, finalNumVals);
              }
            }
          }
          __syncthreads();
#pragma unroll
          for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
            if (needScanSort[i]) {
              const auto rowId     = (threadIdx.x / Policy::AccThCols) + i * Policy::AccThRows;
              const auto gmemRowId = starty + i * Policy::AccThRows;
              if (gmemRowId < m) {
                storeWarpQShmem<Policy, Pair>(heapArr[i], shDumpKV, rowId, numOfNN);
              }
            }
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
          const auto gmemRowId  = starty + i * Policy::AccThRows;
          const auto shMemRowId = (threadIdx.x / Policy::AccThCols) + i * Policy::AccThRows;
          if (gmemRowId < m) {
#pragma unroll
            for (int j = 0; j < Policy::AccColsPerTh; ++j) {
              const auto colId = startx + j * Policy::AccThCols;
              Pair otherKV     = {keyMax, identity};
              if (colId < ldd) {
                otherKV.value = acc[i][j];
                otherKV.key   = colId;
              }
              heapArr[i]->add(otherKV.value, otherKV.key);
            }

            bool needSort = (heapArr[i]->numVals > 0);
            needSort      = __any_sync(mask, needSort);
            if (needSort) { heapArr[i]->reduce(); }
            storeWarpQShmem<Policy, Pair>(heapArr[i], shDumpKV, shMemRowId, numOfNN);
          }
        }
      }

      if (((gridStrideX + Policy::Nblk * gridDim.x) >= n) && gridDim.x == 1) {
        // This is last iteration of grid stride X
        loadAllWarpQShmem<Policy, Pair>(heapArr, &shDumpKV[0], m, numOfNN);
        storeWarpQGmem<Policy, Pair>(heapArr, out_dists, out_inds, m, numOfNN, starty);
      }
    };

  constexpr bool write_out = false;
  ffanns::distance::detail::PairwiseDistances<DataT,
                                            OutT,
                                            IdxT,
                                            Policy,
                                            OpT,
                                            decltype(epilog_lambda),
                                            FinalLambda,
                                            decltype(rowEpilog_lambda),
                                            isRowMajor,
                                            write_out>
    obj(x,
        y,
        m,
        n,
        k,
        lda,
        ldb,
        ldd,
        _xn,
        _yn,
        nullptr,  // output ptr, can be null as write_out == false.
        smem,
        distance_op,
        epilog_lambda,
        fin_op,
        rowEpilog_lambda);
  obj.run();
}

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          int VecLen,
          bool usePrevTopKs,
          bool isRowMajor>
void fusedL2ExpKnnImpl(const DataT* x,
                       const DataT* y,
                       const AccT* xn,
                       const AccT* yn,
                       IdxT m,
                       IdxT n,
                       IdxT k,
                       IdxT lda,
                       IdxT ldb,
                       IdxT ldd,
                       bool sqrt,
                       OutT* out_dists,
                       IdxT* out_inds,
                       IdxT numOfNN,
                       cudaStream_t stream,
                       void* workspace,
                       size_t& worksize)
{
  typedef typename raft::linalg::Policy2x8<AccT, 1>::Policy RowPolicy;
  typedef typename raft::linalg::Policy4x4<AccT, VecLen>::ColPolicy ColPolicy;

  typedef typename std::conditional<true, RowPolicy, ColPolicy>::type KPolicy;

  ASSERT(isRowMajor, "Only Row major inputs are allowed");

  ASSERT(!(((x != y) && (worksize < (m + n) * sizeof(AccT))) || (worksize < m * sizeof(AccT))),
         "workspace size error");
  ASSERT(workspace != nullptr, "workspace is null");

  dim3 blk(KPolicy::Nthreads);

  typedef cub::KeyValuePair<uint32_t, AccT> Pair;

  ffanns::distance::detail::ops::l2_exp_distance_op<DataT, AccT, IdxT> distance_op{sqrt};
  raft::identity_op fin_op{};

  if constexpr (isRowMajor) {
    constexpr auto fusedL2ExpKnn32RowMajor = fusedL2kNN<DataT,
                                                        OutT,
                                                        IdxT,
                                                        KPolicy,
                                                        decltype(distance_op),
                                                        decltype(fin_op),
                                                        32,
                                                        2,
                                                        usePrevTopKs,
                                                        isRowMajor>;
    constexpr auto fusedL2ExpKnn64RowMajor = fusedL2kNN<DataT,
                                                        OutT,
                                                        IdxT,
                                                        KPolicy,
                                                        decltype(distance_op),
                                                        decltype(fin_op),
                                                        64,
                                                        3,
                                                        usePrevTopKs,
                                                        isRowMajor>;

    auto fusedL2ExpKnnRowMajor = fusedL2ExpKnn32RowMajor;
    if (numOfNN <= 32) {
      fusedL2ExpKnnRowMajor = fusedL2ExpKnn32RowMajor;
    } else if (numOfNN <= 64) {
      fusedL2ExpKnnRowMajor = fusedL2ExpKnn64RowMajor;
    } else {
      ASSERT(numOfNN <= 64, "fusedL2kNN: num of nearest neighbors must be <= 64");
    }

    const auto sharedMemSize =
      distance_op.template shared_mem_size<KPolicy>() + (KPolicy::Mblk * numOfNN * sizeof(Pair));
    dim3 grid = ffanns::distance::detail::launchConfigGenerator<KPolicy>(
      m, n, sharedMemSize, fusedL2ExpKnnRowMajor);
    int32_t* mutexes = nullptr;
    if (grid.x > 1) {
      const auto numMutexes   = raft::ceildiv<int>(m, KPolicy::Mblk);
      const auto normsSize    = (x != y) ? (m + n) * sizeof(AccT) : n * sizeof(AccT);
      const auto requiredSize = sizeof(int32_t) * numMutexes + normsSize;
      if (worksize < requiredSize) {
        worksize = requiredSize;
        return;
      } else {
        mutexes = (int32_t*)((char*)workspace + normsSize);
        RAFT_CUDA_TRY(cudaMemsetAsync(mutexes, 0, sizeof(int32_t) * numMutexes, stream));
      }
    }

    // calculate norms if they haven't been passed in
    if (!xn) {
      AccT* xn_ = (AccT*)workspace;
      workspace = xn_ + m;
      raft::linalg::rowNorm(
        xn_, x, k, m, raft::linalg::L2Norm, isRowMajor, stream, raft::identity_op{});
      xn = xn_;
    }
    if (!yn) {
      if (x == y) {
        yn = xn;
      } else {
        AccT* yn_ = (AccT*)(workspace);
        raft::linalg::rowNorm(
          yn_, y, k, n, raft::linalg::L2Norm, isRowMajor, stream, raft::identity_op{});
        yn = yn_;
      }
    }

    fusedL2ExpKnnRowMajor<<<grid, blk, sharedMemSize, stream>>>(x,
                                                                y,
                                                                xn,
                                                                yn,
                                                                m,
                                                                n,
                                                                k,
                                                                lda,
                                                                ldb,
                                                                ldd,
                                                                distance_op,
                                                                fin_op,
                                                                (uint32_t)numOfNN,
                                                                mutexes,
                                                                out_dists,
                                                                out_inds);
  } else {
  }

  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          bool usePrevTopKs,
          bool isRowMajor>
void fusedL2ExpKnn(IdxT m,
                   IdxT n,
                   IdxT k,
                   IdxT lda,
                   IdxT ldb,
                   IdxT ldd,
                   const DataT* x,
                   const DataT* y,
                   const AccT* xn,
                   const AccT* yn,
                   bool sqrt,
                   OutT* out_dists,
                   IdxT* out_inds,
                   IdxT numOfNN,
                   cudaStream_t stream,
                   void* workspace,
                   size_t& worksize)
{
  size_t bytesA = sizeof(DataT) * lda;
  size_t bytesB = sizeof(DataT) * ldb;
  if (16 % sizeof(DataT) == 0 && bytesA % 16 == 0 && bytesB % 16 == 0) {
    fusedL2ExpKnnImpl<DataT, AccT, OutT, IdxT, 16 / sizeof(DataT), usePrevTopKs, isRowMajor>(
      x,
      y,
      xn,
      yn,
      m,
      n,
      k,
      lda,
      ldb,
      ldd,
      sqrt,
      out_dists,
      out_inds,
      numOfNN,
      stream,
      workspace,
      worksize);
  } else if (8 % sizeof(DataT) == 0 && bytesA % 8 == 0 && bytesB % 8 == 0) {
    fusedL2ExpKnnImpl<DataT, AccT, OutT, IdxT, 8 / sizeof(DataT), usePrevTopKs, isRowMajor>(
      x,
      y,
      xn,
      yn,
      m,
      n,
      k,
      lda,
      ldb,
      ldd,
      sqrt,
      out_dists,
      out_inds,
      numOfNN,
      stream,
      workspace,
      worksize);
  } else {
    fusedL2ExpKnnImpl<DataT, AccT, OutT, IdxT, 1, usePrevTopKs, isRowMajor>(x,
                                                                            y,
                                                                            xn,
                                                                            yn,
                                                                            m,
                                                                            n,
                                                                            k,
                                                                            lda,
                                                                            ldb,
                                                                            ldd,
                                                                            sqrt,
                                                                            out_dists,
                                                                            out_inds,
                                                                            numOfNN,
                                                                            stream,
                                                                            workspace,
                                                                            worksize);
  }
}

template <typename value_idx,
          typename value_t,
          bool usePrevTopKs   = false,
          typename distance_t = float>
void fusedL2Knn(size_t D,
                value_idx* out_inds,
                distance_t* out_dists,
                const value_t* index,
                const value_t* query,
                size_t n_index_rows,
                size_t n_query_rows,
                int k,
                bool rowMajorIndex,
                bool rowMajorQuery,
                cudaStream_t stream,
                ffanns::distance::DistanceType metric,
                const distance_t* index_norms = NULL,
                const distance_t* query_norms = NULL)
{
  // Validate the input data
  ASSERT(k > 0, "l2Knn: k must be > 0");
  ASSERT(D > 0, "l2Knn: D must be > 0");
  ASSERT(n_index_rows > 0, "l2Knn: n_index_rows must be > 0");
  ASSERT(index, "l2Knn: index must be provided (passed null)");
  ASSERT(n_query_rows > 0, "l2Knn: n_query_rows must be > 0");
  ASSERT(query, "l2Knn: query must be provided (passed null)");
  ASSERT(out_dists, "l2Knn: out_dists must be provided (passed null)");
  ASSERT(out_inds, "l2Knn: out_inds must be provided (passed null)");
  // Currently we only support same layout for x & y inputs.
  ASSERT(rowMajorIndex == rowMajorQuery,
         "l2Knn: rowMajorIndex and rowMajorQuery should have same layout");
  // TODO: Add support for column major layout
  ASSERT(rowMajorIndex == true, "l2Knn: only rowMajor inputs are supported for now.");

  // Even for L2 Sqrt distance case we use non-sqrt version as FAISS bfKNN only support
  // non-sqrt metric & some tests in RAFT/cuML (like Linkage) fails if we use L2 sqrt.
  constexpr bool sqrt = false;

  size_t worksize = 0, tempWorksize = 0;
  rmm::device_uvector<char> workspace(worksize, stream);
  value_idx lda = D, ldb = D, ldd = n_index_rows;
  // <ffanns::distance::DistanceType::L2Expanded, float, float, float, value_idx>
  switch (metric) {
    case ffanns::distance::DistanceType::L2SqrtExpanded:
    case ffanns::distance::DistanceType::L2Expanded:
      tempWorksize =
        ffanns::distance::getWorkspaceSize<ffanns::distance::DistanceType::L2Expanded,
                                         value_t,
                                         distance_t,
                                         distance_t,
                                         value_idx>(query, index, n_query_rows, n_index_rows, D);
      worksize = tempWorksize;
      workspace.resize(worksize, stream);
      fusedL2ExpKnn<value_t, distance_t, distance_t, value_idx, usePrevTopKs, true>(
        n_query_rows,
        n_index_rows,
        D,
        lda,
        ldb,
        ldd,
        query,
        index,
        query_norms,
        index_norms,
        sqrt,
        out_dists,
        out_inds,
        k,
        stream,
        workspace.data(),
        worksize);
      if (worksize > tempWorksize) {
        workspace.resize(worksize, stream);
        fusedL2ExpKnn<value_t, distance_t, distance_t, value_idx, usePrevTopKs, true>(
          n_query_rows,
          n_index_rows,
          D,
          lda,
          ldb,
          ldd,
          query,
          index,
          query_norms,
          index_norms,
          sqrt,
          out_dists,
          out_inds,
          k,
          stream,
          workspace.data(),
          worksize);
      }
      break;
    default: printf("[ffanns] only L2SqrtExpanded and L2Expanded distance metric is supported\n"); break;
  };
}

} // namespace ffanns::neighbors::detail