#pragma once

#include "distance_ops/all_ops.cuh"
#include "pairwise_matrix/dispatch.cuh"
// #include "pairwise_matrix/dispatch_sm60.cuh"
#include "pairwise_matrix/dispatch_sm80.cuh"
#include "ffanns/distance/distance.hpp"
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_dev_essentials.cuh>  // to_float

#include <type_traits>

namespace ffanns {
namespace distance {
namespace detail {

/**
 * @brief: A tag type for overload resolution based on DistanceType
 *
 * It is not possible to partially specialize function templates on a single
 * parameter. Instead, it is often easier to use a combination of conventional
 * method overloading and a parameter with a specific tag type. The following
 * type is used to help method overloading based on the DistanceType enum.
 */
template <DistanceType d>
using distance_tag = std::integral_constant<DistanceType, d>;

template <typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void distance_impl_l2_expanded(  // NOTE: different name
  bool perform_sqrt,             // dispatch on sqrt
  const DataT* x,
  const DataT* y,
  OutT* out,
  IdxT m,
  IdxT n,
  IdxT k,
  AccT* workspace,
  size_t worksize,
  FinOpT fin_op,
  cudaStream_t stream,
  bool is_row_major)
{
  // raft distance support inputs as float/double and output as uint8_t/float/double.
  static_assert(!((sizeof(OutT) > 1) && (sizeof(AccT) != sizeof(OutT))),
                "OutT can be uint8_t, float, double,"
                "if sizeof(OutT) > 1 then sizeof(AccT) == sizeof(OutT).");

  ASSERT(!(worksize < (m + n) * sizeof(AccT)), "workspace size error");
  ASSERT(workspace != nullptr, "workspace is null");

  // TODO: May we have a better method to avoid misalignment?
  uintptr_t offset = alignof(OutT) - (reinterpret_cast<uintptr_t>(workspace) % alignof(OutT));
  if (offset == alignof(OutT)) { offset = 0; }
  OutT* x_norm = reinterpret_cast<OutT*>(reinterpret_cast<char*>(workspace) + offset);

  offset       = (reinterpret_cast<uintptr_t>(x_norm) % alignof(OutT));
  OutT* y_norm = x_norm;
  // TODO: Column major case looks to have lower accuracy for X == Y,
  // perhaps the use of stridedSummationKernel could be causing this,
  // need to investigate and fix.
  if ((x == y) && is_row_major) {
    raft::linalg::rowNorm(x_norm,
                          x,
                          k,
                          std::max(m, n),
                          raft::linalg::L2Norm,
                          is_row_major,
                          stream,
                          raft::identity_op{});
  } else {
    y_norm += m;
    raft::linalg::rowNorm(
      x_norm, x, k, m, raft::linalg::L2Norm, is_row_major, stream, raft::identity_op{});
    raft::linalg::rowNorm(
      y_norm, y, k, n, raft::linalg::L2Norm, is_row_major, stream, raft::identity_op{});
  }

  ops::l2_exp_distance_op<DataT, AccT, IdxT> distance_op{perform_sqrt};
  pairwise_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::InnerProduct> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  raft::linalg::gemm(handle,
                     out,
                     const_cast<DataT*>(x),
                     const_cast<DataT*>(y),
                     m,
                     n,
                     k,
                     !is_row_major,
                     !is_row_major,
                     is_row_major,
                     stream);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::L2Expanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT* workspace,
                   size_t worksize,
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  bool perform_sqrt   = false;
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  distance_impl_l2_expanded(
    perform_sqrt, x, y, out, m, n, k, workspace, worksize, fin_op, stream, is_row_major);
}

template <ffanns::distance::DistanceType distanceType,
          typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_ = int>
void distance(raft::resources const& handle,
              const InType* x,
              const InType* y,
              OutType* out,
              Index_ m,
              Index_ n,
              Index_ k,
              void* workspace,
              size_t worksize,
              FinalLambda fin_op,
              bool isRowMajor    = true,
              OutType metric_arg = 2.0f)
{
  // raft distance support inputs as float/double and output as uint8_t/float/double.
  static_assert(!((sizeof(OutType) > 1) && (sizeof(AccType) != sizeof(OutType))),
                "OutType can be uint8_t, float, double,"
                "if sizeof(OutType) > 1 then sizeof(AccType) == sizeof(OutType).");

  distance_impl<InType, AccType, OutType, FinalLambda, Index_>(
    handle,
    distance_tag<distanceType>{},
    x,
    y,
    out,
    m,
    n,
    k,
    reinterpret_cast<AccType*>(workspace),
    worksize,
    fin_op,
    isRowMajor,
    metric_arg);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <ffanns::distance::DistanceType distanceType,
          typename InType,
          typename AccType,
          typename OutType,
          typename Index_ = int>
void distance(raft::resources const& handle,
              const InType* x,
              const InType* y,
              OutType* out,
              Index_ m,
              Index_ n,
              Index_ k,
              void* workspace,
              size_t worksize,
              bool isRowMajor    = true,
              OutType metric_arg = 2.0f)
{
  auto fin_op = raft::identity_op();

  distance<distanceType, InType, AccType, OutType, decltype(fin_op), Index_>(
    handle, x, y, out, m, n, k, workspace, worksize, fin_op, isRowMajor, metric_arg);
}

template <ffanns::distance::DistanceType distanceType,
          typename InType,
          typename AccType,
          typename OutType,
          typename Index_ = int>
size_t getWorkspaceSize(const InType* x, const InType* y, Index_ m, Index_ n, Index_ k)
{
  size_t worksize             = 0;
  constexpr bool is_allocated = (distanceType <= ffanns::distance::DistanceType::CosineExpanded) ||
                                (distanceType == ffanns::distance::DistanceType::CorrelationExpanded);
  constexpr int numOfBuffers =
    (distanceType == ffanns::distance::DistanceType::CorrelationExpanded) ? 2 : 1;

  if (is_allocated) {
    // TODO : when X == Y allocate std::max(m, n) instead of m + n when column major input
    // accuracy issue is resolved until then we allocate as m + n.
    worksize += numOfBuffers * m * sizeof(AccType);
    worksize += numOfBuffers * n * sizeof(AccType);
  }

  return worksize;
}

};  // namespace detail
};  // namespace distance
};  // namespace ffanns