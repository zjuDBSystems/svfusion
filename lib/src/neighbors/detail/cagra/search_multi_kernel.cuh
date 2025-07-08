#pragma once

#include "compute_distance-ext.cuh"
#include "device_common.hpp"
#include "hashmap.hpp"
#include "search_plan.cuh"
#include "utils.hpp"
// TODO: This shouldn't be invoking anything from spatial/knn
#include "../ann_utils.cuh"
#include "../../../core/nvtx.hpp"

#include "ffanns/distance/distance.hpp"
#include "ffanns/neighbors/common.hpp"
#include "ffanns/neighbors/hd_mapper.hpp"
#include "ffanns/selection/select_k.hpp"
#include "topk_for_cagra/topk.h"  //todo replace with raft kernel
#include "ffanns/core/host_distance.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/logger-ext.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>

// RMM & Thrust 依赖
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/gather.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace ffanns::neighbors::cagra::detail {

namespace multi_kernel_search {

struct TimingAccumulator {
  float total_kernel1_ratio = 0.0f;
  float total_graph_transfer_ratio = 0.0f;
  float total_kernel2_ratio = 0.0f;
  float total_data_transfer_ratio = 0.0f;
  float total_kernel3_ratio = 0.0f;
  float total_time = 0.0f;
  int num_iters = 0;
  
  void add(const float kernel1_ratio, const float graph_transfer_ratio, const float kernel2_ratio, const float data_transfer_ratio, const float kernel3_ratio, const float compute_time) {
      total_kernel1_ratio += kernel1_ratio;
      total_graph_transfer_ratio += graph_transfer_ratio;
      total_kernel2_ratio += kernel2_ratio;
      total_data_transfer_ratio += data_transfer_ratio;
      total_kernel3_ratio += kernel3_ratio;
      total_time += compute_time;
      num_iters++;
  }
  
  void print() const {
    printf("[search_multi_kernel] 本轮search各阶段耗时占比：\n");
    printf("  Kernel1: %f%%\n", total_kernel1_ratio / num_iters * 100);
    printf("  Graph Transfer: %f%%\n", total_graph_transfer_ratio / num_iters * 100);
    printf("  Kernel2: %f%%\n", total_kernel2_ratio / num_iters * 100);
    printf("  Data Transfer: %f%%\n", total_data_transfer_ratio / num_iters * 100);
    printf("  Kernel3: %f%%\n", total_kernel3_ratio / num_iters  * 100);
  }

  void print2() const {
    printf("[search_multi_kernel] 本轮search各阶段耗时：\n");
    printf("  Kernel1: %fms\n", total_kernel1_ratio);
    printf("  Graph Transfer: %fms\n", total_graph_transfer_ratio);
    printf("  Kernel2: %fms\n", total_kernel2_ratio);
    printf("  Data Transfer: %fms\n", total_data_transfer_ratio);
    printf("  Kernel3: %fms\n", total_kernel3_ratio);
  }

  std::tuple<float, float, float, float, float, float> get_ratios() const {
    return std::make_tuple(
      total_kernel1_ratio / num_iters,
      total_graph_transfer_ratio / num_iters,
      total_kernel2_ratio / num_iters,
      total_data_transfer_ratio / num_iters,
      total_kernel3_ratio / num_iters,
      total_time
    );
  }
};

template <typename DataT, typename IndexT, typename DistanceT>
struct ComputeDistanceContext {
  host_device_mapper* hd_mapper;       // CPU-GPU 数据映射器
  graph_hd_mapper* graph_mapper;       // 图数据映射器
  size_t* hd_status;                   // CPU-GPU 状态统计
  int* in_edges;                       // 输入边数据
  // 存储 pinned memory 的底层指针
  DistanceT* cpu_distances_ptr;        // Pinned memory for distances
  IndexT* host_indices_ptr;            // Pinned memory for indices
  unsigned int* query_miss_counter_ptr; // Pinned memory for miss counter
  IndexT* host_graph_buffer_ptr;
  unsigned iter = 0;
  ffanns::distance::DistanceType metric;

  ComputeDistanceContext(host_device_mapper* hd_mapper_ptr,
                         graph_hd_mapper* graph_mapper_ptr,
                         size_t* hd_status_ptr,
                         int* in_edges_ptr,
                         DistanceT* cpu_distances_ptr_,
                         IndexT* host_indices_ptr_,
                         unsigned int* query_miss_counter_ptr_,
                         IndexT* host_graph_buffer_ptr_,
                         unsigned iter_,
                         ffanns::distance::DistanceType metric_)
    : hd_mapper(hd_mapper_ptr),
      graph_mapper(graph_mapper_ptr),
      hd_status(hd_status_ptr),
      in_edges(in_edges_ptr),
      cpu_distances_ptr(cpu_distances_ptr_),
      host_indices_ptr(host_indices_ptr_),
      query_miss_counter_ptr(query_miss_counter_ptr_),
      host_graph_buffer_ptr(host_graph_buffer_ptr_),
      iter(iter_),
      metric(metric_) 
  {}
};

template<typename T>
using HostDistFunc = float (*)(const T*, const T*, uint32_t);

template<typename DataT>
inline HostDistFunc<DataT>
select_host_dist(ffanns::distance::DistanceType dt)
{
    switch (dt) {
        case ffanns::distance::DistanceType::InnerProduct:
            // RAFT_LOG_INFO("Using InnerProduct");
            return static_cast<HostDistFunc<DataT>>(
                       ffanns::core::neg_inner_product_avx2);
        /* L2 和 L2Expanded 都走这条 */
        default:
            // RAFT_LOG_INFO("Using l2_distance");
            return static_cast<HostDistFunc<DataT>>(
                       ffanns::core::l2_distance_avx2);
    }
}

__device__ inline uint32_t splitmix32(uint32_t x) {
  x += 0x9e3779b9u;               // SplitMix32 一步即可
  x = (x ^ (x >> 16)) * 0x85ebca6bu;
  x = (x ^ (x >> 13)) * 0xc2b2ae35u;
  return x ^ (x >> 16);
}

template <class T>
RAFT_KERNEL set_value_kernel(T* const dev_ptr, const T val)
{
  *dev_ptr = val;
}

template <class T>
RAFT_KERNEL set_value_kernel(T* const dev_ptr, const T val, const std::size_t count)
{
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= count) { return; }
  dev_ptr[tid] = val;
}

template <class T>
void set_value(T* const dev_ptr, const T val, cudaStream_t cuda_stream)
{
  set_value_kernel<T><<<1, 1, 0, cuda_stream>>>(dev_ptr, val);
}

template <class T>
void set_value(T* const dev_ptr, const T val, const std::size_t count, cudaStream_t cuda_stream)
{
  constexpr std::uint32_t block_size = 256;
  const auto grid_size               = (count + block_size - 1) / block_size;
  set_value_kernel<T><<<grid_size, block_size, 0, cuda_stream>>>(dev_ptr, val, count);
}

template <class T>
RAFT_KERNEL set_value_batch_kernel(T* const dev_ptr,
                                   const std::size_t ld,
                                   const T val,
                                   const std::size_t count,
                                   const std::size_t batch_size)
{
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= count * batch_size) { return; }
  const auto batch_id              = tid / count;
  const auto elem_id               = tid % count;
  dev_ptr[elem_id + ld * batch_id] = val;
}

template <class T>
void set_value_batch(T* const dev_ptr,
                     const std::size_t ld,
                     const T val,
                     const std::size_t count,
                     const std::size_t batch_size,
                     cudaStream_t cuda_stream)
{
  constexpr std::uint32_t block_size = 256;
  const auto grid_size               = (count * batch_size + block_size - 1) / block_size;
  set_value_batch_kernel<T>
    <<<grid_size, block_size, 0, cuda_stream>>>(dev_ptr, ld, val, count, batch_size);
}

template <class T>
RAFT_KERNEL get_value_kernel(T* const host_ptr, const T* const dev_ptr)
{
  *host_ptr = *dev_ptr;
}

template <class T>
void get_value(T* const host_ptr, const T* const dev_ptr, cudaStream_t cuda_stream)
{
  get_value_kernel<T><<<1, 1, 0, cuda_stream>>>(host_ptr, dev_ptr);
}

inline void check_pointer(const void* ptr, const char* name) {
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
        RAFT_LOG_INFO("Error getting attributes for %s: %s\n", 
               name, cudaGetErrorString(err));
        return;
    }
    
    RAFT_LOG_INFO("%s is on %s, value=%p\n", 
           name,
           attrs.type == cudaMemoryTypeDevice ? "device" : "host",
           ptr);
}

template <typename DistanceT, typename IndexT>
RAFT_KERNEL scatter_kernel(const DistanceT* __restrict__ src,
                           DistanceT* __restrict__ dst,
                           const IndexT* __restrict__ offsets,
                           const int num)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < num) {
        // printf("tid=%d, src[tid]=%f, dst[tid]=%f, offsets[tid]=%d\n", tid, src[tid], dst[offsets[tid]], offsets[tid]);
        if (offsets[tid] != utils::get_max_value<IndexT>()) {
            dst[offsets[tid]] = src[tid];
        }
        // dst[offsets[tid]] = src[tid];
    }
}

template <typename IndexT>
RAFT_KERNEL graph_scatter_kernel(const IndexT* __restrict__ src_buffer,
                           IndexT* __restrict__ dst_graph,
                           const IndexT* __restrict__ device_indices,
                           const uint32_t graph_degree,
                           const uint32_t num_entries)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_entries) return;

    const IndexT* src_row = src_buffer + tid * graph_degree;
    IndexT* dst_row = dst_graph + device_indices[tid] * graph_degree;
    for (uint32_t i = 0; i < graph_degree; i++) {
      dst_row[i] = src_row[i];
    }
}

template <class DATASET_DESCRIPTOR_T, class SAMPLE_FILTER_T>
RAFT_KERNEL random_pickup_kernel1(
  const DATASET_DESCRIPTOR_T* dataset_desc,
  const typename DATASET_DESCRIPTOR_T::DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const std::size_t num_pickup,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const typename DATASET_DESCRIPTOR_T::INDEX_T* seed_ptr,  // [num_queries, num_seeds]
  const std::uint32_t num_seeds,                 
  typename DATASET_DESCRIPTOR_T::INDEX_T* const result_indices_ptr,       // [num_queries, ldr]
  typename DATASET_DESCRIPTOR_T::INDEX_T* const d_result_indices_ptr,
  const std::uint32_t ldr,                                                 // (*) ldr >= num_pickup
  SAMPLE_FILTER_T sample_filter,
  host_device_mapper* hd_mapper
)
{
  assert(num_distilation == 1);
  using INDEX_T    = typename DATASET_DESCRIPTOR_T::INDEX_T;

  const auto global_team_index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t query_id      = blockIdx.y;
  const unsigned int max_attempts = 1000;
  if (global_team_index >= num_pickup) { return; }

  INDEX_T seed_index, final_seed_index, final_device_index;
  unsigned int attempts = 0;
  bool final_hit = false;
  if (seed_ptr && (global_team_index < num_seeds)) {
    seed_index = seed_ptr[global_team_index + (num_seeds * query_id)];
  } else {
    while(true) {
      if (attempts >= max_attempts)
        break;  
      seed_index = device::xorshift64(global_team_index ^ (attempts << 8) ^ rand_xor_mask) % dataset_desc->size;
      attempts ++;
      if constexpr (!std::is_same<SAMPLE_FILTER_T,
                  ffanns::neighbors::filtering::none_sample_filter>::value) {
          if (!sample_filter(seed_index))
          continue;   
      }
      auto [is_hit, dev_idx] = hd_mapper->get_wo_replace_safe(seed_index, false);
      if (is_hit) {
        final_hit = true;
        final_seed_index = seed_index;
        final_device_index = dev_idx;
        break;  
      }
    }
  }

  const auto store_gmem_index = global_team_index + (ldr * query_id);
  if (final_hit) {
    result_indices_ptr[store_gmem_index] = final_seed_index;
    d_result_indices_ptr[store_gmem_index] = final_device_index;
  } else {
    result_indices_ptr[store_gmem_index] = utils::get_max_value<INDEX_T>();;
    d_result_indices_ptr[store_gmem_index] = utils::get_max_value<INDEX_T>();;
  }
}

// MAX_DATASET_DIM : must equal to or greater than dataset_dim
template <class DATASET_DESCRIPTOR_T>
RAFT_KERNEL random_pickup_kernel2(
  const DATASET_DESCRIPTOR_T* dataset_desc,
  const typename DATASET_DESCRIPTOR_T::DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const std::size_t num_pickup,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const result_indices_ptr,       // [num_queries, ldr]
  const typename DATASET_DESCRIPTOR_T::INDEX_T* const tmp_device_indices_ptr,  // [num_queries, ldr]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries, ldr]
  const std::uint32_t ldr,                                                // (*) ldr >= num_pickup
  typename DATASET_DESCRIPTOR_T::INDEX_T* const visited_hashmap_ptr,  // [num_queries, 1 << bitlen]
  const std::uint32_t hash_bitlen)
{
  using DATA_T     = typename DATASET_DESCRIPTOR_T::DATA_T;
  using INDEX_T    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using DISTANCE_T = typename DATASET_DESCRIPTOR_T::DISTANCE_T;

  const auto team_size_bits    = dataset_desc->team_size_bitshift();
  const auto ldb               = hashmap::get_size(hash_bitlen);
  const auto global_team_index = (blockIdx.x * blockDim.x + threadIdx.x) >> team_size_bits;
  const std::uint32_t query_id      = blockIdx.y;
  if (global_team_index >= num_pickup) { return; }
  extern __shared__ uint8_t smem[];
  dataset_desc = dataset_desc->setup_workspace(smem, queries_ptr, query_id);
  __syncthreads();

  const auto store_gmem_index = global_team_index + (ldr * query_id);
  INDEX_T best_index_team_local = result_indices_ptr[store_gmem_index];
  INDEX_T best_index_team_tmp = tmp_device_indices_ptr[store_gmem_index];
  DISTANCE_T norm2 = dataset_desc->compute_distance(best_index_team_tmp, true);

  if ((threadIdx.x & ((1u << team_size_bits) - 1u)) == 0) {
    if (hashmap::insert(
          visited_hashmap_ptr + (ldb * query_id), hash_bitlen, best_index_team_local)) {
      result_distances_ptr[store_gmem_index] = norm2;
    } else {
      result_distances_ptr[store_gmem_index] = utils::get_max_value<DISTANCE_T>();
      result_indices_ptr[store_gmem_index]   = utils::get_max_value<INDEX_T>();
    }
  }
}

// MAX_DATASET_DIM : must be equal to or greater than dataset_dim
template <typename DataT, typename IndexT, typename DistanceT, typename SAMPLE_FILTER_T>
void random_pickup(const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
                   const DataT* queries_ptr,  // [num_queries, dataset_dim]
                   std::size_t num_queries,
                   std::size_t num_pickup,
                   unsigned num_distilation,
                   uint64_t rand_xor_mask,
                   const IndexT* seed_ptr,  // [num_queries, num_seeds]
                   std::uint32_t num_seeds,
                   IndexT* result_indices_ptr,       // [num_queries, ldr]
                   DistanceT* result_distances_ptr,  // [num_queries, ldr]
                   std::size_t ldr,                  // (*) ldr >= num_pickup
                   IndexT* visited_hashmap_ptr,      // [num_queries, 1 << bitlen]
                   std::uint32_t hash_bitlen,
                   SAMPLE_FILTER_T sample_filter,
                   host_device_mapper* hd_mapper,
                   std::size_t *hd_status,
                   IndexT* d_result_indices_ptr,
                   cudaStream_t cuda_stream)
{
#ifdef FFANNS_DEBUG_LOG
  cudaEvent_t start, stop, kernel1_end, data_transfer;
  cudaEventCreate(&start);
  cudaEventCreate(&stop); 
  cudaEventCreate(&kernel1_end);
  cudaEventCreate(&data_transfer);
  cudaEventRecord(start, cuda_stream);
#endif

  const auto block_size                = 256u;
  const auto num_teams_per_threadblock = block_size / dataset_desc.team_size;
  const dim3 grid_size1((num_pickup + block_size - 1) / block_size, num_queries);
  const dim3 grid_size((num_pickup + num_teams_per_threadblock - 1) / num_teams_per_threadblock,
                       num_queries);
  // RAFT_LOG_INFO("[random_pickup] block_size: %u, num_teams_per_threadblock: %u, num_pickup: %u,  grid_size: (%u, %u)", block_size, num_teams_per_threadblock, num_pickup, grid_size1.x, grid_size1.y);
  // check_pointer(hd_mapper, "hd_mapper");
  
  random_pickup_kernel1<<<grid_size1, block_size, dataset_desc.smem_ws_size_in_bytes, cuda_stream>>>(
    dataset_desc.dev_ptr(cuda_stream),
    queries_ptr,
    num_pickup,
    num_distilation,
    rand_xor_mask,
    seed_ptr,
    num_seeds,
    result_indices_ptr,
    d_result_indices_ptr,
    ldr,
    sample_filter,
    hd_mapper
  );
  
#ifdef FFANNS_DEBUG_LOG
  cudaEventRecord(kernel1_end, cuda_stream);
#endif
  RAFT_CUDA_TRY(cudaStreamSynchronize(cuda_stream));

#ifdef FFANNS_DEBUG_LOG
  cudaEventRecord(data_transfer, cuda_stream);
  cudaEventSynchronize(data_transfer);
#endif 
  RAFT_CUDA_TRY(cudaStreamSynchronize(cuda_stream));
  
  random_pickup_kernel2<<<grid_size, block_size, dataset_desc.smem_ws_size_in_bytes, cuda_stream>>>(
    dataset_desc.dev_ptr(cuda_stream),
    queries_ptr,
    num_pickup,
    result_indices_ptr,
    d_result_indices_ptr,
    result_distances_ptr,
    ldr,
    visited_hashmap_ptr,
    hash_bitlen);
  
#ifdef FFANNS_DEBUG_LOG
  cudaEventRecord(stop, cuda_stream);
  cudaEventSynchronize(stop);
  float kernel1_time, data_transfer_time, kernel2_time, total_time;
  cudaEventElapsedTime(&kernel1_time, start, kernel1_end);
  cudaEventElapsedTime(&data_transfer_time, kernel1_end, data_transfer);
  cudaEventElapsedTime(&kernel2_time, data_transfer, stop);
  cudaEventElapsedTime(&total_time, start, stop);
  // RAFT_LOG_INFO("[random_pickup] Kernel1 time taken: %f us", kernel1_time * 1000);
  // RAFT_LOG_INFO("[random_pickup] Data transfer time taken: %f us", data_transfer_time * 1000);
  // RAFT_LOG_INFO("[random_pickup] Kernel2 time taken: %f us", kernel2_time * 1000);
  // RAFT_LOG_INFO("[random_pickup] Total time taken: %f ms", total_time);
#endif

}

template <class INDEX_T>
RAFT_KERNEL pickup_next_parents_kernel(
  INDEX_T* const parent_candidates_ptr,        // [num_queries, lds]
  const std::size_t lds,                       // (*) lds >= parent_candidates_size
  const std::uint32_t parent_candidates_size,  //
  INDEX_T* const visited_hashmap_ptr,          // [num_queries, 1 << hash_bitlen]
  const std::size_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  INDEX_T* const parent_list_ptr,      // [num_queries, ldd]
  const std::size_t ldd,               // (*) ldd >= parent_list_size
  const std::size_t parent_list_size,  //
  std::uint32_t* const terminate_flag)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;

  const std::size_t ldb   = hashmap::get_size(hash_bitlen);
  const uint32_t query_id = blockIdx.x;
  if (threadIdx.x < 32) {
    // pickup next parents with single warp
    for (std::uint32_t i = threadIdx.x; i < parent_list_size; i += 32) {
      parent_list_ptr[i + (ldd * query_id)] = utils::get_max_value<INDEX_T>();
    }
    std::uint32_t parent_candidates_size_max = parent_candidates_size;
    if (parent_candidates_size % 32) {
      parent_candidates_size_max += 32 - (parent_candidates_size % 32);
    }
    std::uint32_t num_new_parents = 0;
    for (std::uint32_t j = threadIdx.x; j < parent_candidates_size_max; j += 32) {
      INDEX_T index;
      int new_parent = 0;
      if (j < parent_candidates_size) {
        index = parent_candidates_ptr[j + (lds * query_id)];
        if ((index & index_msb_1_mask) == 0) {  // check most significant bit
          new_parent = 1;
        }
      }
      const std::uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
      if (new_parent) {
        const auto i = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_new_parents;
        if (i < parent_list_size) {
          parent_list_ptr[i + (ldd * query_id)] = j;
          parent_candidates_ptr[j + (lds * query_id)] |=
            index_msb_1_mask;  // set most significant bit as used node
        }
      }
      num_new_parents += __popc(ballot_mask);
      if (num_new_parents >= parent_list_size) { break; }
    }
    if ((num_new_parents > 0) && (threadIdx.x == 0)) { *terminate_flag = 0; }
  } else if (small_hash_bitlen) {
    // reset small-hash
    hashmap::init(visited_hashmap_ptr + (ldb * query_id), hash_bitlen, 32);
  }

  if (small_hash_bitlen) {
    __syncthreads();
    // insert internal-topk indices into small-hash
    for (unsigned i = threadIdx.x; i < parent_candidates_size; i += blockDim.x) {
      auto key = parent_candidates_ptr[i + (lds * query_id)] &
                 ~index_msb_1_mask;  // clear most significant bit
      hashmap::insert(visited_hashmap_ptr + (ldb * query_id), hash_bitlen, key);
    }
  }
}

template <class INDEX_T>
void pickup_next_parents(INDEX_T* const parent_candidates_ptr,  // [num_queries, lds]
                         const std::size_t lds,                 // (*) lds >= parent_candidates_size
                         const std::size_t parent_candidates_size,  //
                         const std::size_t num_queries,
                         INDEX_T* const visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
                         const std::size_t hash_bitlen,
                         const std::size_t small_hash_bitlen,
                         INDEX_T* const parent_list_ptr,      // [num_queries, ldd]
                         const std::size_t ldd,               // (*) ldd >= parent_list_size
                         const std::size_t parent_list_size,  //
                         std::uint32_t* const terminate_flag,
                         cudaStream_t cuda_stream = 0)
{
  std::uint32_t block_size = 32;
  if (small_hash_bitlen) {
    block_size = 128;
    while (parent_candidates_size > block_size) {
      block_size *= 2;
    }
    block_size = min(block_size, (uint32_t)512);
  }
  pickup_next_parents_kernel<INDEX_T>
    <<<num_queries, block_size, 0, cuda_stream>>>(parent_candidates_ptr,
                                                  lds,
                                                  parent_candidates_size,
                                                  visited_hashmap_ptr,
                                                  hash_bitlen,
                                                  small_hash_bitlen,
                                                  parent_list_ptr,
                                                  ldd,
                                                  parent_list_size,
                                                  terminate_flag);
}

template <class DATASET_DESCRIPTOR_T>
RAFT_KERNEL compute_distance_to_child_nodes_kernel1(
  const typename DATASET_DESCRIPTOR_T::INDEX_T* const
    parent_node_list,  // [num_queries, search_width]
  typename DATASET_DESCRIPTOR_T::INDEX_T* const
    parent_candidates_ptr,  // [num_queries, search_width]
  const std::size_t lds,
  const std::uint32_t search_width,
  const DATASET_DESCRIPTOR_T* dataset_desc,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const 
    neighbor_graph_ptr, // [device graph_size, graph_degree]
  const std::uint32_t graph_degree,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const result_indices_ptr,       // [num_queries, ldd]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries, ldd]
  const std::uint32_t ldd,  // (*) ldd >= search_width * graph_degree
  uint8_t* compute_distance_flags_ptr,
  graph_hd_mapper* graph_mapper,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const miss_host_graphids,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const miss_device_graphids,
  unsigned int* const miss_counter,
  const std::uint32_t num_queries)
{
  using INDEX_T    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using DISTANCE_T = typename DATASET_DESCRIPTOR_T::DISTANCE_T;
  
  const auto global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const auto total_search_pairs = num_queries * search_width; 
  if (global_thread_id >= total_search_pairs) return;

  const std::uint32_t query_id = global_thread_id / search_width;
  const std::uint32_t local_search_id = global_thread_id % search_width;

  const std::size_t parent_list_index =
    parent_node_list[local_search_id + (search_width * query_id)];

  if (parent_list_index == utils::get_max_value<INDEX_T>()) { 
    for (uint32_t idx = 0; idx < graph_degree; idx++) {
      const size_t result_idx = query_id * ldd + local_search_id * graph_degree + idx;
      result_distances_ptr[result_idx] = utils::get_max_value<DISTANCE_T>();
      compute_distance_flags_ptr[result_idx] = false;
      result_indices_ptr[result_idx] = utils::get_max_value<INDEX_T>();   
    }
    return; 
  }

  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  const auto raw_parent_index        = parent_candidates_ptr[parent_list_index + (lds * query_id)];

  if (raw_parent_index == utils::get_max_value<INDEX_T>()) {
    for (uint32_t idx = 0; idx < graph_degree; idx++) {
      const size_t result_idx = query_id * ldd + local_search_id * graph_degree + idx;
      result_distances_ptr[result_idx] = utils::get_max_value<DISTANCE_T>();
      compute_distance_flags_ptr[result_idx] = false;
      result_indices_ptr[result_idx] = utils::get_max_value<INDEX_T>();
    }
    return;
  }

  const auto parent_index = raw_parent_index & ~index_msb_1_mask;
  auto [is_hit, device_parent_index] = graph_mapper->get(parent_index);
  if (!is_hit) {
    unsigned int pos = atomicAdd(miss_counter, 1);
    miss_host_graphids[pos] = parent_index;
    miss_device_graphids[pos] = device_parent_index;
  }

  // if (is_hit) {
  for (uint32_t idx = 0; idx < graph_degree; idx++) {
    const size_t result_idx = query_id * ldd + local_search_id * graph_degree + idx;
    const size_t child_id_offset = device_parent_index * graph_degree + idx;
    result_indices_ptr[result_idx] = child_id_offset;
  }
  // }  
}

template <class DATASET_DESCRIPTOR_T, class SAMPLE_FILTER_T>
RAFT_KERNEL compute_distance_to_child_nodes_kernel2(
  const std::uint32_t search_width,
  const DATASET_DESCRIPTOR_T* dataset_desc,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const 
    neighbor_graph_ptr, // [device graph_size, graph_degree]
  const std::uint32_t graph_degree,
  const typename DATASET_DESCRIPTOR_T::DATA_T* query_ptr,  // [num_queries, data_dim]
  typename DATASET_DESCRIPTOR_T::INDEX_T* const
    visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  const std::uint32_t hash_bitlen,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const result_indices_ptr,       // [num_queries, ldd]
  typename DATASET_DESCRIPTOR_T::INDEX_T* const d_result_indices_ptr,
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries, ldd]
  const std::uint32_t ldd,  // (*) ldd >= search_width * graph_degree
  SAMPLE_FILTER_T sample_filter,
  uint8_t* compute_distance_flags_ptr,
  host_device_mapper* hd_mapper,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const miss_host_indices1,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const miss_result_idx_offsets,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const miss_host_indices2,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const miss_device_indices,
  unsigned int* const miss_counter1_vec,
  unsigned int* const miss_counter1,
  unsigned int* const miss_counter2,
  int* const in_edges,
  unsigned int iter)
{
  using INDEX_T    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using DISTANCE_T = typename DATASET_DESCRIPTOR_T::DISTANCE_T;

  const auto team_size_bits = dataset_desc->team_size_bitshift();
  const auto team_size      = 1u << team_size_bits;
  const uint32_t ldb        = hashmap::get_size(hash_bitlen);

  const auto global_team_id = threadIdx.x + blockDim.x * blockIdx.x;
  const auto query_id       = blockIdx.y;

  if (global_team_id >= search_width * graph_degree) { return; }
  const std::size_t result_idx = ldd * query_id + global_team_id;
  const std::size_t child_id_offset = result_indices_ptr[result_idx];
  if (child_id_offset == utils::get_max_value<INDEX_T>()) {
    return;
  }
  const std::size_t child_id = neighbor_graph_ptr[child_id_offset];
  result_indices_ptr[result_idx] = child_id;
  // const std::size_t child_id = result_indices_ptr[result_idx];
  const auto miss_counter_offset = query_id * search_width * graph_degree;

  // if (child_id == utils::get_max_value<INDEX_T>()) {
  //   return;
  // }

  if constexpr (!std::is_same<SAMPLE_FILTER_T,
                              ffanns::neighbors::filtering::none_sample_filter>::value) {
    if (!sample_filter(child_id)) {
      // printf("[compute_distance_to_child_nodes_kernel2] child_id filtered!");
      result_distances_ptr[result_idx] = utils::get_max_value<DISTANCE_T>();
      compute_distance_flags_ptr[result_idx] = 0;
      return;
    }
  }

  const auto compute_distance_flag = hashmap::insert<INDEX_T>(
    visited_hashmap_ptr + (ldb * blockIdx.y), hash_bitlen, child_id);

  if (compute_distance_flag) {
    compute_distance_flags_ptr[result_idx] = 1;
    // auto [is_hit, device_index] = hd_mapper->get(child_id);
    auto [is_hit, device_index] = hd_mapper->get_wo_replace(child_id, true);

    if (!is_hit) {
      // α 再调大（例如 7~8）,β 控“中点位置”：>0.5 → 后移，<0.5 → 前移
      // if (iter < 40) {
      //   float ratio = (float)iter / 40.0f;
      //   float x = 6.0f * (ratio - 0.5);
      //   float skip_prob = 0.9 / (1.f + __expf(x));
      //   uint32_t rand_seed = child_id ^ (query_id << 16) ^ (iter << 8);
      //   float rand_val = splitmix32(rand_seed) * 2.3283064e-10f; 
      //   if (rand_val < skip_prob) {
      //     hashmap::remove<INDEX_T>(visited_hashmap_ptr + (ldb * blockIdx.y), hash_bitlen, child_id);
      //     result_distances_ptr[result_idx] = utils::get_max_value<DISTANCE_T>();
      //     compute_distance_flags_ptr[result_idx] = 0;
      //     return;
      //   }
      // }
      unsigned int pos = atomicAdd(&miss_counter1_vec[query_id], 1);
      atomicAdd(miss_counter1, 1);
      miss_host_indices1[miss_counter_offset + pos] = child_id;
      miss_result_idx_offsets[miss_counter_offset + pos] = result_idx;
      compute_distance_flags_ptr[result_idx] = 0;
      // auto result = hd_mapper->replace(child_id, in_edges);
      // is_hit = result.first;
      // device_index = result.second;
      // // is_hit = false;
      // // better to not replace
      // if (!is_hit) {
      //   unsigned int pos = atomicAdd(&miss_counter1_vec[query_id], 1);
      //   atomicAdd(miss_counter1, 1);
      //   miss_host_indices1[miss_counter_offset + pos] = child_id;
      //   miss_result_idx_offsets[miss_counter_offset + pos] = result_idx;
      //   // Set distance flgas as 0 to avoid computing on device
      //   compute_distance_flags_ptr[result_idx] = 0;
      // } else { // sucessfully replaced
      //   unsigned int pos = atomicAdd(miss_counter2, 1);
      //   miss_host_indices2[pos] = child_id;
      //   miss_device_indices[pos] = device_index;
      // }
    }
    // !!! debugging
    d_result_indices_ptr[result_idx] = device_index;
    // d_result_indices_ptr[result_idx] = child_id;
    // !!! end debugging
  } else {
    result_distances_ptr[result_idx] = utils::get_max_value<DISTANCE_T>();
    compute_distance_flags_ptr[result_idx] = 0;
  }
}

template <class DATASET_DESCRIPTOR_T>
RAFT_KERNEL compute_distance_to_child_nodes_kernel3(
  const std::uint32_t search_width,
  const DATASET_DESCRIPTOR_T* dataset_desc,
  const std::uint32_t graph_degree,
  const typename DATASET_DESCRIPTOR_T::DATA_T* query_ptr,  // [num_queries, data_dim]
  typename DATASET_DESCRIPTOR_T::INDEX_T* const tmp_result_indices_ptr,
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries, ldd]
  const std::uint32_t ldd,  // (*) ldd >= search_width * graph_degree
  uint8_t* compute_distance_flags_ptr)
{
  using INDEX_T    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using DISTANCE_T = typename DATASET_DESCRIPTOR_T::DISTANCE_T;

  const auto team_size_bits = dataset_desc->team_size_bitshift();
  const auto team_size      = 1u << team_size_bits;
  const auto tid            = threadIdx.x + blockDim.x * blockIdx.x;
  const auto global_team_id = tid >> team_size_bits;
  const auto query_id       = blockIdx.y;

  extern __shared__ uint8_t smem[];
  // Load a query
  dataset_desc = dataset_desc->setup_workspace(smem, query_ptr, query_id);

  __syncthreads();
  if (global_team_id >= search_width * graph_degree) { return; }

  const auto compute_distance_flag = compute_distance_flags_ptr[ldd * blockIdx.y + global_team_id];

  if (compute_distance_flag) {
    const std::size_t tmp_child_id = tmp_result_indices_ptr[ldd * blockIdx.y + global_team_id];
    DISTANCE_T norm2 = dataset_desc->compute_distance(tmp_child_id, compute_distance_flag);
    if ((threadIdx.x & (team_size - 1)) == 0) {
      result_distances_ptr[ldd * blockIdx.y + global_team_id] = norm2;
    }
  } else {
    return;
  }
}

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          class SAMPLE_FILTER_T>
void compute_distance_to_child_nodes(
  const IndexT* parent_node_list,        // [num_queries, search_width]
  IndexT* const parent_candidates_ptr,   // [num_queries, search_width]
  DistanceT* const parent_distance_ptr,  // [num_queries, search_width]
  std::size_t lds,
  std::uint32_t search_width,
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  const IndexT* neighbor_graph_ptr,  // [dataset_size, graph_degree]
  IndexT* d_neighbor_graph_ptr,
  std::uint32_t graph_degree,
  const DataT* query_ptr,  // [num_queries, data_dim]
  const DataT* host_query_ptr,
  std::uint32_t num_queries,
  IndexT* visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  std::uint32_t hash_bitlen,
  IndexT* result_indices_ptr,       // [num_queries, ldd]
  DistanceT* result_distances_ptr,  // [num_queries, ldd]
  std::uint32_t ldd,                // (*) ldd >= search_width * graph_degree
  SAMPLE_FILTER_T sample_filter,
  size_t itopk_size,
  IndexT* d_result_indices_ptr,
  ComputeDistanceContext<DataT, IndexT, DistanceT>* compute_ctx,
  raft::resources const& res,
  TimingAccumulator * time_accumulator)
{
  auto hd_mapper = compute_ctx->hd_mapper;
  auto graph_mapper = compute_ctx->graph_mapper;
  cudaStream_t cuda_stream      = raft::resource::get_cuda_stream(res);
  HostDistFunc<DataT> host_dist_fn = select_host_dist<DataT>(compute_ctx->metric);

#ifdef FFANNS_DEBUG_LOG
  cudaEvent_t start, stop, kernel1_end, kernel2_start, kernel2_end, data_transfer;
  cudaEventCreate(&start);
  cudaEventCreate(&stop); 
  cudaEventCreate(&kernel1_end);
  cudaEventCreate(&kernel2_start);
  cudaEventCreate(&kernel2_end);
  cudaEventCreate(&data_transfer);
  cudaEventRecord(start, cuda_stream);
#endif

  const auto block_size      = 128;
  const auto teams_per_block = block_size / dataset_desc.team_size;
  
  const dim3 grid_size((search_width * graph_degree + teams_per_block - 1) / teams_per_block,
                       num_queries);  
  const auto grid_size1 = (num_queries * search_width + block_size - 1) / block_size;
  const dim3 grid_size2((search_width * graph_degree + block_size - 1) / block_size, num_queries);
  // const int grid_size1  = (search_width * num_queries + block_size - 1) / block_size;

  std::vector<uint8_t> host_compute_distance_flags(num_queries * ldd);
  rmm::device_uvector<uint8_t> compute_distance_flags(num_queries * ldd, cuda_stream);

  rmm::device_scalar<unsigned int> graph_miss_counter(0, cuda_stream);
  rmm::device_uvector<IndexT> miss_host_graphids(num_queries * search_width, cuda_stream);
  rmm::device_uvector<IndexT> miss_device_graphids(num_queries * search_width, cuda_stream);
  compute_distance_to_child_nodes_kernel1<<<grid_size1,
                                           block_size,
                                           dataset_desc.smem_ws_size_in_bytes,
                                           cuda_stream>>>(parent_node_list,
                                                          parent_candidates_ptr,
                                                          lds,
                                                          search_width,
                                                          dataset_desc.dev_ptr(cuda_stream),
                                                          d_neighbor_graph_ptr,
                                                          graph_degree,
                                                          result_indices_ptr,
                                                          result_distances_ptr,
                                                          ldd,
                                                          compute_distance_flags.data(),
                                                          graph_mapper,
                                                          miss_host_graphids.data(),
                                                          miss_device_graphids.data(),
                                                          graph_miss_counter.data(),
                                                          num_queries);

#ifdef FFANNS_DEBUG_LOG
  cudaEventRecord(kernel1_end, cuda_stream);
#endif
  RAFT_CUDA_TRY(cudaStreamSynchronize(cuda_stream));
  
  unsigned int num_graph_miss = graph_miss_counter.value(cuda_stream);
  if (num_graph_miss > 0) {
    std::vector<IndexT> host_graphids(num_graph_miss);
    auto host_graph_buffer =  compute_ctx->host_graph_buffer_ptr;
    rmm::device_uvector<IndexT> device_graph_buffer(num_graph_miss * graph_degree, cuda_stream);
    raft::copy(host_graphids.data(), miss_host_graphids.data(), num_graph_miss, cuda_stream);
    // if (num_graph_miss > 0.2 * num_queries * search_width) {
    //   RAFT_LOG_INFO("[compute_distance_to_child_nodes] !!!num_graph_miss: %u", num_graph_miss);
    // }
    for (size_t i = 0; i < num_graph_miss; i++) {
        const IndexT* neighbor_list_head = neighbor_graph_ptr + host_graphids[i] * graph_degree;
        memcpy(host_graph_buffer + i * graph_degree, neighbor_list_head, graph_degree * sizeof(IndexT));
    }
    raft::copy(device_graph_buffer.data(), host_graph_buffer, num_graph_miss * graph_degree, cuda_stream);
    int threadsPerBlock = 128;
    int blocks = (num_graph_miss + threadsPerBlock - 1) / threadsPerBlock;
    graph_scatter_kernel<<<blocks, threadsPerBlock, 0, cuda_stream>>>(device_graph_buffer.data(),
                                                                  d_neighbor_graph_ptr,
                                                                  miss_device_graphids.data(),
                                                                  graph_degree,
                                                                  num_graph_miss);
    compute_ctx->hd_status[2] += num_graph_miss;
  }
  compute_ctx->hd_status[3] += num_queries * search_width;

#ifdef FFANNS_DEBUG_LOG
  cudaEventRecord(kernel2_start, cuda_stream);
#endif
  RAFT_CUDA_TRY(cudaStreamSynchronize(cuda_stream));
  rmm::device_uvector<unsigned int> miss_counter1_vec(num_queries, cuda_stream);
  thrust::fill(thrust::cuda::par.on(cuda_stream), miss_counter1_vec.begin(), miss_counter1_vec.end(), 0u);
  rmm::device_scalar<unsigned int> miss_counter1(0, cuda_stream);
  rmm::device_scalar<unsigned int> miss_counter2(0, cuda_stream);
  rmm::device_uvector<IndexT> miss_host_indices1(num_queries * search_width * graph_degree, cuda_stream);
  rmm::device_uvector<IndexT> result_idx_offsets(num_queries * search_width * graph_degree, cuda_stream);
  thrust::fill(thrust::cuda::par.on(cuda_stream), result_idx_offsets.begin(), result_idx_offsets.end(), utils::get_max_value<IndexT>());
  rmm::device_uvector<IndexT> miss_host_indices2(num_queries * search_width * graph_degree, cuda_stream);
  rmm::device_uvector<IndexT> miss_device_indices(num_queries * search_width * graph_degree, cuda_stream);
  compute_distance_to_child_nodes_kernel2<<<grid_size2,
                                           block_size,
                                           dataset_desc.smem_ws_size_in_bytes,
                                           cuda_stream>>>(search_width,
                                                          dataset_desc.dev_ptr(cuda_stream),
                                                          d_neighbor_graph_ptr,
                                                          graph_degree,
                                                          query_ptr,
                                                          visited_hashmap_ptr,
                                                          hash_bitlen,
                                                          result_indices_ptr,
                                                          d_result_indices_ptr,
                                                          result_distances_ptr,
                                                          ldd, 
                                                          sample_filter,
                                                          compute_distance_flags.data(),
                                                          hd_mapper,
                                                          miss_host_indices1.data(),
                                                          result_idx_offsets.data(),
                                                          miss_host_indices2.data(),
                                                          miss_device_indices.data(),
                                                          miss_counter1_vec.data(),
                                                          miss_counter1.data(),
                                                          miss_counter2.data(),
                                                          compute_ctx->in_edges,
                                                          compute_ctx->iter);
  
#ifdef FFANNS_DEBUG_LOG
  cudaEventRecord(kernel2_end, cuda_stream);
#endif
  RAFT_CUDA_TRY(cudaStreamSynchronize(cuda_stream));

  // !!! debugging
  unsigned int num_miss1 = miss_counter1.value(cuda_stream);
  // cudaStream_t stream_host_comp = nullptr;
  // cudaStream_t stream_device_comp = nullptr;
  // if (res.has_resource_factory(raft::resource::resource_type::CUDA_STREAM_POOL)) {
  //   size_t pool_size = raft::resource::get_stream_pool_size(res);
  //   if (pool_size >= 2) {  
  //       stream_host_comp = raft::resource::get_stream_from_stream_pool(res, 0);
  //       stream_device_comp = raft::resource::get_stream_from_stream_pool(res, 1);
  //   } else {
  //       RAFT_LOG_INFO("CUDA stream pool size (%zu) is insufficient. Requires at least 2 streams.", pool_size);
  //   }
  // } else {
  //   RAFT_LOG_INFO("CUDA stream pool resource not found.");
  // }
  // rmm::device_uvector<DistanceT> tmp_device_distances(num_miss1, cuda_stream);
  rmm::device_uvector<DistanceT> tmp_device_distances(num_queries * search_width * graph_degree, cuda_stream);
  if (num_miss1 > 0) {
    // std::vector<DistanceT> cpu_distances(num_miss1);
    // std::vector<IndexT> host_indices(num_miss1);
    // std::vector<IndexT> host_result_idx_offsets(num_miss1);
    // raft::copy(host_indices.data(), miss_host_indices1.data(), num_miss1, cuda_stream);
    // raft::copy(host_result_idx_offsets.data(), result_idx_offsets.data(), num_miss1, cuda_stream);
    auto cpu_distances =  compute_ctx->cpu_distances_ptr;
    auto host_indices = compute_ctx->host_indices_ptr;
    auto query_miss_counter = compute_ctx->query_miss_counter_ptr;
    raft::copy(host_indices, miss_host_indices1.data(), num_queries * search_width * graph_degree, cuda_stream);
    raft::copy(query_miss_counter, miss_counter1_vec.data(), num_queries, cuda_stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(cuda_stream));

#pragma omp parallel for num_threads(32) schedule(dynamic, 1)
    for (size_t qid = 0; qid < num_queries; qid++) {
      size_t num_miss = query_miss_counter[qid];
      if (num_miss == 0) continue;
      const auto query_offset = qid * search_width * graph_degree;
      const DataT* query_data = host_query_ptr + qid * dataset_desc.stride;
      const DataT* child_data;
      for (size_t j = 0; j < num_miss; j++) {
        size_t idx = query_offset + j;
        child_data = dataset_desc.ptr + host_indices[idx] * dataset_desc.stride;
        // cpu_distances[idx] = ffanns::core::l2_distance_avx2(query_data, child_data, dataset_desc.stride); 
        cpu_distances[idx] = host_dist_fn(query_data, child_data, dataset_desc.stride);
      }
    }
    raft::copy(tmp_device_distances.data(), cpu_distances, num_queries * search_width * graph_degree, cuda_stream);   
    
// #pragma omp parallel for num_threads(32) schedule(static)
//     for (size_t i = 0; i < num_miss1; i++) {
//       const DataT* child_data = dataset_desc.ptr + host_indices[i] * dataset_desc.stride;
//       size_t query_id = host_result_idx_offsets[i] / ldd;
//       const DataT* query_data = host_query_ptr + query_id * dataset_desc.stride;
//       cpu_distances[i] = ffanns::core::l2_distance_avx512(query_data, child_data, dataset_desc.stride);
//       //cpu_distances[i] = 0.02;
//     }
//     raft::copy(tmp_device_distances.data(), cpu_distances.data(), num_miss1, cuda_stream);                                            
    compute_ctx->hd_status[0] += num_miss1;
  }

  unsigned int num_miss2 = miss_counter2.value(cuda_stream);
  if (num_miss2 > 0) {
      // if (num_miss2 > 1000) {
      //   RAFT_LOG_INFO("[compute_distance_to_child_nodes] !!!num_miss2: %u", num_miss2);
      // }
      std::vector<IndexT> host_indices(num_miss2);
      std::vector<IndexT> device_indices(num_miss2);
      
      // 复制miss信息到host
      raft::copy(host_indices.data(), miss_host_indices2.data(), num_miss2, cuda_stream);
      raft::copy(device_indices.data(), miss_device_indices.data(), num_miss2, cuda_stream);
      
      // // 传输数据
      for (size_t i = 0; i < num_miss2; i++) {
          raft::copy(
              const_cast<DataT*>(dataset_desc.dd_ptr) + device_indices[i] * dataset_desc.stride,
              dataset_desc.ptr + host_indices[i] * dataset_desc.stride,
              dataset_desc.stride,
              cuda_stream
          );
      }
      compute_ctx->hd_status[0] += num_miss2;
  }
  compute_ctx->hd_status[1] += num_queries * search_width * graph_degree;
  /* */
  // !!! end debugging
#ifdef FFANNS_DEBUG_LOG
  cudaEventRecord(data_transfer, cuda_stream);
#endif
  RAFT_CUDA_TRY(cudaStreamSynchronize(cuda_stream));

  compute_distance_to_child_nodes_kernel3<<<grid_size,
                                           block_size,
                                           dataset_desc.smem_ws_size_in_bytes,
                                           cuda_stream>>>(search_width,
                                                          dataset_desc.dev_ptr(cuda_stream),
                                                          graph_degree,
                                                          query_ptr,
                                                          d_result_indices_ptr,
                                                          result_distances_ptr,
                                                          ldd,
                                                          compute_distance_flags.data());

  // raft::resource::sync_stream_pool(res);                                                  
  if (num_miss1 > 0) {
    RAFT_CUDA_TRY(cudaStreamSynchronize(cuda_stream));
    int threadsPerBlock = 128;
    int blocks = (num_queries * search_width * graph_degree + threadsPerBlock - 1) / threadsPerBlock;
    scatter_kernel<<<blocks, threadsPerBlock, 0, cuda_stream>>>(tmp_device_distances.data(),
                                                                  result_distances_ptr,
                                                                  result_idx_offsets.data(),
                                                                  num_queries * search_width * graph_degree);
  }
#ifdef FFANNS_DEBUG_LOG                                 
  cudaEventRecord(stop, cuda_stream);
  cudaEventSynchronize(stop);
  float kernel1_time, graph_transfer_time, kernel2_time, data_transfer_time, kernel3_time, total_time;
  cudaEventElapsedTime(&kernel1_time, start, kernel1_end);
  cudaEventElapsedTime(&graph_transfer_time, kernel1_end, kernel2_start);
  cudaEventElapsedTime(&kernel2_time, kernel2_start, kernel2_end);
  cudaEventElapsedTime(&data_transfer_time, kernel2_end, data_transfer);
  cudaEventElapsedTime(&kernel3_time, data_transfer, stop);
  cudaEventElapsedTime(&total_time, start, stop);
  time_accumulator->add(kernel1_time, graph_transfer_time, kernel2_time, data_transfer_time, kernel3_time, total_time);
  // time_accumulator->add(kernel1_time / total_time, graph_transfer_time / total_time, kernel2_time / total_time, data_transfer_time / total_time, kernel3_time / total_time, total_time);
#endif
  // RAFT_LOG_INFO("[compute_distance_to_child_nodes] Kernel1 time taken: %f us", kernel1_time * 1000);
  // RAFT_LOG_INFO("[compute_distance_to_child_nodes] Graph transfer time taken: %f us", graph_transfer_time * 1000);
  // RAFT_LOG_INFO("[compute_distance_to_child_nodes] Kernel2 time taken: %f us", kernel2_time * 1000);
  // RAFT_LOG_INFO("[compute_distance_to_child_nodes] Data transfer time taken: %f us", data_transfer_time * 1000);
  // RAFT_LOG_INFO("[compute_distance_to_child_nodes] Kernel3 time taken: %f us", kernel3_time * 1000);
  // RAFT_LOG_INFO("[compute_distance_to_child_nodes] Total time taken: %f us", total_time * 1000);
}

template <class INDEX_T>
RAFT_KERNEL remove_parent_bit_kernel(const std::uint32_t num_queries,
                                     const std::uint32_t num_topk,
                                     INDEX_T* const topk_indices_ptr,  // [ld, num_queries]
                                     const std::uint32_t ld)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;

  uint32_t i_query = blockIdx.x;
  if (i_query >= num_queries) return;

  for (unsigned i = threadIdx.x; i < num_topk; i += blockDim.x) {
    topk_indices_ptr[i + (ld * i_query)] &= ~index_msb_1_mask;  // clear most significant bit
  }
}

template <class INDEX_T>
void remove_parent_bit(const std::uint32_t num_queries,
                       const std::uint32_t num_topk,
                       INDEX_T* const topk_indices_ptr,  // [ld, num_queries]
                       const std::uint32_t ld,
                       cudaStream_t cuda_stream = 0)
{
  const std::size_t grid_size  = num_queries;
  const std::size_t block_size = 256;
  remove_parent_bit_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
    num_queries, num_topk, topk_indices_ptr, ld);
}

// This function called after the `remove_parent_bit` function
template <class INDEX_T, class DISTANCE_T, class SAMPLE_FILTER_T>
RAFT_KERNEL apply_filter_kernel(INDEX_T* const result_indices_ptr,
                                DISTANCE_T* const result_distances_ptr,
                                const std::size_t lds,
                                const std::uint32_t result_buffer_size,
                                const std::uint32_t num_queries,
                                SAMPLE_FILTER_T sample_filter)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  const auto tid                     = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= result_buffer_size * num_queries) { return; }
  const auto i     = tid % result_buffer_size;
  const auto j     = tid / result_buffer_size;
  const auto index = i + j * lds;

  if (result_indices_ptr[index] != ~index_msb_1_mask &&
      !sample_filter(result_indices_ptr[index])) {
    result_indices_ptr[index]   = utils::get_max_value<INDEX_T>();
    result_distances_ptr[index] = utils::get_max_value<DISTANCE_T>();
  }
}

template <class INDEX_T, class DISTANCE_T, class SAMPLE_FILTER_T>
void apply_filter(INDEX_T* const result_indices_ptr,
                  DISTANCE_T* const result_distances_ptr,
                  const std::size_t lds,
                  const std::uint32_t result_buffer_size,
                  const std::uint32_t num_queries,
                  SAMPLE_FILTER_T sample_filter,
                  cudaStream_t cuda_stream)
{
  const std::uint32_t block_size = 256;
  const std::uint32_t grid_size  = raft::ceildiv(num_queries * result_buffer_size, block_size);

  apply_filter_kernel<<<grid_size, block_size, 0, cuda_stream>>>(result_indices_ptr,
                                                                 result_distances_ptr,
                                                                 lds,
                                                                 result_buffer_size,
                                                                 num_queries,
                                                                 sample_filter);
}

template <class T>
RAFT_KERNEL batched_memcpy_kernel(T* const dst,  // [batch_size, ld_dst]
                                  const uint64_t ld_dst,
                                  const T* const src,  // [batch_size, ld_src]
                                  const uint64_t ld_src,
                                  const uint64_t count,
                                  const uint64_t batch_size)
{
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= count * batch_size) { return; }
  const auto i          = tid % count;
  const auto j          = tid / count;
  dst[i + (ld_dst * j)] = src[i + (ld_src * j)];
}

template <class T>
void batched_memcpy(T* const dst,  // [batch_size, ld_dst]
                    const uint64_t ld_dst,
                    const T* const src,  // [batch_size, ld_src]
                    const uint64_t ld_src,
                    const uint64_t count,
                    const uint64_t batch_size,
                    cudaStream_t cuda_stream)
{
  assert(ld_dst >= count);
  assert(ld_src >= count);
  constexpr uint32_t block_size = 256;
  const auto grid_size          = (batch_size * count + block_size - 1) / block_size;
  batched_memcpy_kernel<T>
    <<<grid_size, block_size, 0, cuda_stream>>>(dst, ld_dst, src, ld_src, count, batch_size);
}

// result_buffer (work buffer) for "multi-kernel"
// +--------------------+------------------------------+-------------------+
// | internal_top_k (A) | neighbors of internal_top_k  | internal_topk (B) |
// | <itopk_size>       | <search_width * graph_degree> | <itopk_size>      |
// +--------------------+------------------------------+-------------------+
// |<---                 result_buffer_allocation_size                 --->|
// |<---                       result_buffer_size  --->|                     // Double buffer (A)
//                      |<---  result_buffer_size                      --->| // Double buffer (B)
template <typename DataT, typename IndexT, typename DistanceT, typename SAMPLE_FILTER_T>
struct search : search_plan_impl<DataT, IndexT, DistanceT, SAMPLE_FILTER_T> {
  using base_type  = search_plan_impl<DataT, IndexT, DistanceT, SAMPLE_FILTER_T>;
  using DATA_T     = typename base_type::DATA_T;
  using INDEX_T    = typename base_type::INDEX_T;
  using DISTANCE_T = typename base_type::DISTANCE_T;

  static_assert(std::is_same_v<DISTANCE_T, float>, "Only float is supported as resulting distance");

  using base_type::algo;
  using base_type::hashmap_max_fill_rate;
  using base_type::hashmap_min_bitlen;
  using base_type::hashmap_mode;
  using base_type::itopk_size;
  using base_type::max_iterations;
  using base_type::max_queries;
  using base_type::min_iterations;
  using base_type::num_random_samplings;
  using base_type::rand_xor_mask;
  using base_type::search_width;
  using base_type::team_size;
  using base_type::thread_block_size;

  using base_type::dim;
  using base_type::graph_degree;
  using base_type::topk;

  using base_type::hash_bitlen;

  using base_type::dataset_size;
  using base_type::hashmap_size;
  using base_type::result_buffer_size;
  using base_type::small_hash_bitlen;
  using base_type::small_hash_reset_interval;

  using base_type::smem_size;

  using base_type::dataset_desc;
  using base_type::dev_seed;
  using base_type::hashmap;
  using base_type::num_executed_iterations;
  using base_type::num_seeds;
  using base_type::metric;

  size_t result_buffer_allocation_size;
  rmm::device_uvector<INDEX_T> result_indices;       // results_indices_buffer
  rmm::device_uvector<DISTANCE_T> result_distances;  // result_distances_buffer
  rmm::device_uvector<INDEX_T> parent_node_list;
  rmm::device_uvector<uint32_t> topk_hint;
  rmm::device_scalar<uint32_t> terminate_flag;  // dev_terminate_flag, host_terminate_flag.;
  rmm::device_uvector<uint32_t> topk_workspace;

  // temporary storage for _find_topk
  rmm::device_uvector<float> input_keys_storage;
  rmm::device_uvector<float> output_keys_storage;
  rmm::device_uvector<INDEX_T> input_values_storage;
  rmm::device_uvector<INDEX_T> output_values_storage;

  // add extra data structure for cpu-gpu coprocessing
  rmm::device_uvector<IndexT> d_result_indices;  
  rmm::device_uvector<uint32_t> miss_positions;  // 用于存储miss的位置

  search(raft::resources const& res,
         search_params params,
         const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
         int64_t dim,
         int64_t graph_degree,
         uint32_t topk)
    : base_type(res, params, dataset_desc, dim, graph_degree, topk),
      result_indices(0, raft::resource::get_cuda_stream(res)),
      result_distances(0, raft::resource::get_cuda_stream(res)),
      parent_node_list(0, raft::resource::get_cuda_stream(res)),
      topk_hint(0, raft::resource::get_cuda_stream(res)),
      topk_workspace(0, raft::resource::get_cuda_stream(res)),
      terminate_flag(raft::resource::get_cuda_stream(res)),
      input_keys_storage(0, raft::resource::get_cuda_stream(res)),
      output_keys_storage(0, raft::resource::get_cuda_stream(res)),
      input_values_storage(0, raft::resource::get_cuda_stream(res)),
      output_values_storage(0, raft::resource::get_cuda_stream(res)),
      d_result_indices(0, raft::resource::get_cuda_stream(res)),
      miss_positions(0, raft::resource::get_cuda_stream(res))
  {
    set_params(res);
  }

  void set_params(raft::resources const& res)
  {
    //
    // Allocate memory for intermediate buffer and workspace.
    //
    // RAFT_LOG_INFO("itopk_size: %u, num_seeds: %u, search_width: %u, graph_degree: %u", itopk_size, num_seeds, search_width, graph_degree);
    result_buffer_size            = itopk_size + (search_width * graph_degree);
    result_buffer_allocation_size = result_buffer_size + itopk_size;
    result_indices.resize(result_buffer_allocation_size * max_queries,
                          raft::resource::get_cuda_stream(res));
    result_distances.resize(result_buffer_allocation_size * max_queries,
                            raft::resource::get_cuda_stream(res));

    parent_node_list.resize(max_queries * search_width, raft::resource::get_cuda_stream(res));
    topk_hint.resize(max_queries, raft::resource::get_cuda_stream(res));

    size_t topk_workspace_size = _cuann_find_topk_bufferSize(
      itopk_size, max_queries, result_buffer_size, utils::get_cuda_data_type<DATA_T>());
    RAFT_LOG_DEBUG("# topk_workspace_size: %lu", topk_workspace_size);
    topk_workspace.resize(topk_workspace_size, raft::resource::get_cuda_stream(res));

    hashmap.resize(hashmap_size, raft::resource::get_cuda_stream(res));
    d_result_indices.resize(result_buffer_allocation_size * max_queries, 
                            raft::resource::get_cuda_stream(res));
    miss_positions.resize(result_buffer_allocation_size * max_queries,
                            raft::resource::get_cuda_stream(res));
  }

  ~search() {}

  inline void _find_topk(raft::resources const& handle,
                         uint32_t topK,
                         uint32_t sizeBatch,
                         uint32_t numElements,
                         const float* inputKeys,    // [sizeBatch, ldIK,]
                         uint32_t ldIK,             // (*) ldIK >= numElements
                         const INDEX_T* inputVals,  // [sizeBatch, ldIV,]
                         uint32_t ldIV,             // (*) ldIV >= numElements
                         float* outputKeys,         // [sizeBatch, ldOK,]
                         uint32_t ldOK,             // (*) ldOK >= topK
                         INDEX_T* outputVals,       // [sizeBatch, ldOV,]
                         uint32_t ldOV,             // (*) ldOV >= topK
                         void* workspace,
                         bool sort,
                         uint32_t* hints)
  {
    auto stream = raft::resource::get_cuda_stream(handle);

    // _cuann_find_topk right now is limited to a max-k of 1024.
    // RAFT has a matrix::select_k function - which handles arbitrary sized values of k,
    // but doesn't accept strided inputs unlike _cuann_find_topk
    // The multi-kernel search path requires strided access - since its cleverly allocating memory
    // (layout described in the search_plan_impl function below), such that both the
    // neighbors and the internal_topk are adjacent - in a double buffered format.
    // Since this layout doesn't work with the matrix::select_k code - we have to copy
    // over to a contiguous (non-strided) access to handle topk larger than 1024, and
    // potentially also copy back to a strided layout afterwards
    if (topK <= 1024) {
      return _cuann_find_topk(topK,
                              sizeBatch,
                              numElements,
                              inputKeys,
                              ldIK,
                              inputVals,
                              ldIV,
                              outputKeys,
                              ldOK,
                              outputVals,
                              ldOV,
                              workspace,
                              sort,
                              hints,
                              stream);
    }

    if (ldIK > numElements) {
      if (input_keys_storage.size() != sizeBatch * numElements) {
        input_keys_storage.resize(sizeBatch * numElements, stream);
      }
      batched_memcpy(
        input_keys_storage.data(), numElements, inputKeys, ldIK, numElements, sizeBatch, stream);
      inputKeys = input_keys_storage.data();
    }

    if (ldIV > numElements) {
      if (input_values_storage.size() != sizeBatch * numElements) {
        input_values_storage.resize(sizeBatch * numElements, stream);
      }

      batched_memcpy(
        input_values_storage.data(), numElements, inputVals, ldIV, numElements, sizeBatch, stream);
      inputVals = input_values_storage.data();
    }

    if ((ldOK > topK) && (output_keys_storage.size() != sizeBatch * topK)) {
      output_keys_storage.resize(sizeBatch * topK, stream);
    }

    if ((ldOV > topK) && (output_values_storage.size() != sizeBatch * topK)) {
      output_values_storage.resize(sizeBatch * topK, stream);
    }

    ffanns::selection::select_k(
      handle,
      raft::make_device_matrix_view<const float, int64_t>(inputKeys, sizeBatch, numElements),
      raft::make_device_matrix_view<const INDEX_T, int64_t>(inputVals, sizeBatch, numElements),
      raft::make_device_matrix_view<float, int64_t>(
        ldOK > topK ? output_keys_storage.data() : outputKeys, sizeBatch, topK),
      raft::make_device_matrix_view<INDEX_T, int64_t>(
        ldOV > topK ? output_values_storage.data() : outputVals, sizeBatch, topK),
      true,  // select_min
      sort);

    if (ldOK > topK) {
      batched_memcpy(outputKeys, ldOK, output_keys_storage.data(), topK, topK, sizeBatch, stream);
    }

    if (ldOV > topK) {
      batched_memcpy(outputVals, ldOV, output_values_storage.data(), topK, topK, sizeBatch, stream);
    }
  }

  void operator()(raft::resources const& res,
                  raft::host_matrix_view<const INDEX_T, int64_t> graph,
                  raft::device_matrix_view<INDEX_T, int64_t, raft::row_major> d_graph,
                  INDEX_T* const topk_indices_ptr,       // [num_queries, topk]
                  DISTANCE_T* const topk_distances_ptr,  // [num_queries, topk]
                  const DATA_T* const queries_ptr,       // [num_queries, dataset_dim]
                  const DATA_T* const host_queries_ptr, 
                  const uint32_t num_queries,
                  const INDEX_T* dev_seed_ptr,              // [num_queries, num_seeds]
                  uint32_t* const num_executed_iterations,  // [num_queries,]
                  uint32_t topk,
                  SAMPLE_FILTER_T sample_filter,
                  host_device_mapper* host_hd_mapper,
                  graph_hd_mapper* host_graph_mapper,
                  int* in_edges)
  {
    #ifdef FFANNS_MISS_LOG
    static std::ofstream miss_log;
    static bool first_call = true;
    if (first_call) {
        miss_log.open(bench_config::instance().get_miss_log_path(), std::ios::trunc);
        miss_log << "chunk_id,miss_count,total_count,miss_rate\n";
        // miss_log << "chunk_id,miss_count,total_count,miss_rate,ratio1,ratio2,ratio3,ratio4,ratio5,sum_time,search_time\n";
        first_call = false;
    }
    #endif
    size_t hd_status[4] = {0, 0, 0, 0};
    // cudaEvent_t start, stop, random_pickup_end;
    // cudaEvent_t find_topk_start, find_topk_end, pickup_parents_end, compute_dist_end, test_graph_end;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop); 
    // cudaEventCreate(&random_pickup_end); 
    // cudaEventCreate(&find_topk_start);
    // cudaEventCreate(&find_topk_end);
    // cudaEventCreate(&pickup_parents_end);
    // cudaEventCreate(&compute_dist_end);
    // cudaEventCreate(&test_graph_end);
    // cudaEventRecord(start, raft::resource::get_cuda_stream(res));
    TimingAccumulator accumulator;
    // Init hashmap
    cudaStream_t stream      = raft::resource::get_cuda_stream(res);
    const uint32_t hash_size = hashmap::get_size(hash_bitlen);
    set_value_batch(
      hashmap.data(), hash_size, utils::get_max_value<INDEX_T>(), hash_size, num_queries, stream);

    // Topk hint can not be used when applying a filter
    uint32_t* const top_hint_ptr =
      std::is_same<SAMPLE_FILTER_T, ffanns::neighbors::filtering::none_sample_filter>::value
        ? topk_hint.data()
        : nullptr;
    // Init topk_hint
    if (top_hint_ptr != nullptr && topk_hint.size() > 0) {
      set_value(top_hint_ptr, 0xffffffffu, num_queries, stream);
    }

    // RAFT_LOG_INFO("[search_multi_kernel] host_hd_mapper, size of host_device_mapping: %llu", host_hd_mapper->host_device_mapping_size());
    auto hd_mapper = host_hd_mapper->dev_ptr(res);
    auto graph_mapper = host_graph_mapper->dev_ptr(res);

    // RAFT_LOG_INFO("[search_multi_kernel] num_queries: %u, topk: %u, num_random_samplings: %u", num_queries, topk, num_random_samplings);
    // Choose initial entry point candidates at random
    random_pickup<DataT, IndexT, DistanceT>(dataset_desc,
                                            queries_ptr,
                                            num_queries,
                                            result_buffer_size,
                                            num_random_samplings,
                                            rand_xor_mask,
                                            dev_seed_ptr,
                                            num_seeds,
                                            result_indices.data(),
                                            result_distances.data(),
                                            result_buffer_allocation_size,
                                            hashmap.data(),
                                            hash_bitlen,
                                            sample_filter,
                                            hd_mapper,
                                            hd_status,
                                            d_result_indices.data(),
                                            stream);
    // cudaEventRecord(random_pickup_end, stream);
    // float total_find_topk_time = 0.0;
    // float total_pickup_time = 0.0;
    // float total_compute_dist_time = 0.0;
    unsigned iter = 0;
    const unsigned int MAX_QUERIES = 10000;
    assert(num_queries <= MAX_QUERIES);

    static auto cpu_distances = raft::make_pinned_vector<DistanceT>(res, MAX_QUERIES * search_width * graph_degree);
    static auto host_indices = raft::make_pinned_vector<IndexT>(res, MAX_QUERIES * search_width * graph_degree);
    static auto query_miss_counter = raft::make_pinned_vector<unsigned int>(res, MAX_QUERIES);
    static auto host_graph_buffer = raft::make_pinned_vector<IndexT>(res, MAX_QUERIES * search_width * graph_degree);
    ComputeDistanceContext<DataT, IndexT, DistanceT> compute_ctx(
      hd_mapper,
      graph_mapper,
      hd_status,
      in_edges,
      cpu_distances.data_handle(),
      host_indices.data_handle(),
      query_miss_counter.data_handle(),
      host_graph_buffer.data_handle(),
      0,
      metric
    );
    while (1) {
      // cudaEventRecord(find_topk_start, stream);
      // Make an index list of internal top-k nodes
      _find_topk(res,
                 itopk_size,
                 num_queries,
                 result_buffer_size,
                 result_distances.data() + (iter & 0x1) * itopk_size,
                 result_buffer_allocation_size,
                 result_indices.data() + (iter & 0x1) * itopk_size,
                 result_buffer_allocation_size,
                 result_distances.data() + (1 - (iter & 0x1)) * result_buffer_size,
                 result_buffer_allocation_size,
                 result_indices.data() + (1 - (iter & 0x1)) * result_buffer_size,
                 result_buffer_allocation_size,
                 topk_workspace.data(),
                 true,
                 top_hint_ptr);
      
      // cudaEventRecord(find_topk_end, stream);

      // termination (1)
      if ((iter + 1 == max_iterations)) {
        iter++;
        break;
      }

      if (iter + 1 >= min_iterations) { set_value<uint32_t>(terminate_flag.data(), 1, stream); }

      // pickup parent nodes
      uint32_t _small_hash_bitlen = 0;
      if ((iter + 1) % small_hash_reset_interval == 0) { _small_hash_bitlen = small_hash_bitlen; }
      pickup_next_parents(result_indices.data() + (1 - (iter & 0x1)) * result_buffer_size,
                          result_buffer_allocation_size,
                          itopk_size,
                          num_queries,
                          hashmap.data(),
                          hash_bitlen,
                          _small_hash_bitlen,
                          parent_node_list.data(),
                          search_width,
                          search_width,
                          terminate_flag.data(),
                          stream);

      // cudaEventRecord(pickup_parents_end, stream);

      // termination (2)
      if (iter + 1 >= min_iterations && terminate_flag.value(stream)) {
        iter++;
        break;
      }

      // Compute distance to child nodes that are adjacent to the parent node
      compute_ctx.iter = iter;
      // ffanns::common::nvtx::range<ffanns::common::nvtx::domain::ffanns> compute_range("compute_distance_iter");
      compute_distance_to_child_nodes(
        parent_node_list.data(),
        result_indices.data() + (1 - (iter & 0x1)) * result_buffer_size,
        result_distances.data() + (1 - (iter & 0x1)) * result_buffer_size,
        result_buffer_allocation_size,
        search_width,
        dataset_desc,
        graph.data_handle(),
        d_graph.data_handle(),
        graph.extent(1),
        queries_ptr,
        host_queries_ptr,
        num_queries,
        hashmap.data(),
        hash_bitlen,
        result_indices.data() + itopk_size,
        result_distances.data() + itopk_size,
        result_buffer_allocation_size,
        sample_filter,
        itopk_size,
        d_result_indices.data() + itopk_size,
        &compute_ctx,
        res,
        &accumulator);
      
      // cudaEventRecord(compute_dist_end, stream);
      // cudaEventSynchronize(compute_dist_end);

      // float iter_find_topk_time, iter_pickup_time, iter_compute_time;
      // cudaEventElapsedTime(&iter_find_topk_time, find_topk_start, find_topk_end);
      // cudaEventElapsedTime(&iter_pickup_time, find_topk_end, pickup_parents_end);
      // cudaEventElapsedTime(&iter_compute_time, pickup_parents_end, compute_dist_end);
      // total_find_topk_time += iter_find_topk_time;
      // total_pickup_time += iter_pickup_time;
      // total_compute_dist_time += iter_compute_time;

      iter++;
    }  // while ( 1 )
    // accumulator.print2();
    // RAFT_LOG_INFO("[search_multi_kernel] Average find_topk time: %f ms", total_find_topk_time);
    // RAFT_LOG_INFO("[search_multi_kernel] Average pickup time: %f ms", total_pickup_time);
    // RAFT_LOG_INFO("[search_multi_kernel] Average compute distance time: %f ms", total_compute_dist_time);
    // RAFT_LOG_INFO("[search_multi_kernel] Total iterations: %d", iter);


    auto result_indices_ptr   = result_indices.data() + (iter & 0x1) * result_buffer_size;
    auto result_distances_ptr = result_distances.data() + (iter & 0x1) * result_buffer_size;

    if constexpr (!std::is_same<SAMPLE_FILTER_T,
                                ffanns::neighbors::filtering::none_sample_filter>::value) {
      
      // Remove parent bit in search results
      remove_parent_bit(num_queries,
                        result_buffer_size,
                        result_indices.data() + (iter & 0x1) * itopk_size,
                        result_buffer_allocation_size,
                        stream);

      apply_filter<INDEX_T, DISTANCE_T, SAMPLE_FILTER_T>(
        result_indices.data() + (iter & 0x1) * itopk_size,
        result_distances.data() + (iter & 0x1) * itopk_size,
        result_buffer_allocation_size,
        result_buffer_size,
        num_queries,
        sample_filter,
        stream);     

      result_indices_ptr   = result_indices.data() + (1 - (iter & 0x1)) * result_buffer_size;
      result_distances_ptr = result_distances.data() + (1 - (iter & 0x1)) * result_buffer_size;
      _find_topk(res,
                 itopk_size,
                 num_queries,
                 result_buffer_size,
                 result_distances.data() + (iter & 0x1) * itopk_size,
                 result_buffer_allocation_size,
                 result_indices.data() + (iter & 0x1) * itopk_size,
                 result_buffer_allocation_size,
                 result_distances_ptr,
                 result_buffer_allocation_size,
                 result_indices_ptr,
                 result_buffer_allocation_size,
                 topk_workspace.data(),
                 true,
                 top_hint_ptr);                 
    } else {
      remove_parent_bit(
        num_queries, itopk_size, result_indices_ptr, result_buffer_allocation_size, stream);
    }

    // Copy results from working buffer to final buffer
    batched_memcpy(topk_indices_ptr,
                   topk,
                   result_indices_ptr,
                   result_buffer_allocation_size,
                   topk,
                   num_queries,
                   stream);
    if (topk_distances_ptr) {
      batched_memcpy(topk_distances_ptr,
                     topk,
                     result_distances_ptr,
                     result_buffer_allocation_size,
                     topk,
                     num_queries,
                     stream);
    }

    if (num_executed_iterations) {
      for (std::uint32_t i = 0; i < num_queries; i++) {
        num_executed_iterations[i] = iter;
      }
    }
    host_device_mapper::free_dev_ptr(hd_mapper, raft::resource::get_cuda_stream(res));
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    // cudaEventRecord(stop, raft::resource::get_cuda_stream(res));
    // cudaEventSynchronize(stop);
    // float total_time, random_pickup_time;
    // cudaEventElapsedTime(&random_pickup_time, start, random_pickup_end);
    // cudaEventElapsedTime(&total_time, start, stop);
    // // RAFT_LOG_INFO("[compute_distance_to_child_nodes] Total time taken: %f ms", accumulator.total_time);
    // // RAFT_LOG_INFO("[search_multi_kernel] Random pickup time: %f ms", random_pickup_time);
    // RAFT_LOG_INFO("[search_multi_kernel] Total time taken: %f ms", total_time);
    float miss_rate = hd_status[0] * 1.0 / hd_status[1];
    float graph_miss_rate = hd_status[2] * 1.0 / hd_status[3];
    RAFT_LOG_INFO("[search_multi_kernel] HD status, miss: %lu, total: %lu, miss_rate: %f", hd_status[0], hd_status[1], miss_rate);
    RAFT_LOG_INFO("[search_multi_kernel] Graph HD status, miss: %lu, total: %lu, miss_rate: %f", hd_status[2], hd_status[3], graph_miss_rate);

    #ifdef FFANNS_MISS_LOG
    miss_rate = hd_status[0] * 1.0 / hd_status[1];
    // auto compute_res = accumulator.get_ratios();
    // RAFT_LOG_INFO("[search_multi_kernel] HD status, miss: %lu, total: %lu, miss_rate: %f", hd_status[0], hd_status[1], miss_rate);
    static size_t chunk_id = 0;
    miss_log << chunk_id << "," << hd_status[0] << "," << hd_status[1] << "," << miss_rate << "\n";
    // miss_log << chunk_id << "," << hd_status[0] << "," << hd_status[1] << "," 
    //           << miss_rate << "," << std::get<0>(compute_res) << "," << std::get<1>(compute_res) << ","
    //           << std::get<2>(compute_res) << "," << std::get<3>(compute_res) << "," << std::get<4>(compute_res) << "," 
    //           << std::get<5>(compute_res) << "," << total_time << "\n";
    chunk_id++;
    if (chunk_id % 500 == 0) {
      miss_log.flush();
    }
    #endif
  }

};

}  // namespace multi_kernel_search

} // namespace ffanns::neighbors::cagra::detail

/*
std::vector<IndexT> host_result_indices(num_queries * ldd);
  raft::copy(host_result_indices.data(), result_indices_ptr, num_queries * ldd, cuda_stream);
  raft::copy(host_compute_distance_flags.data(), compute_distance_flags.data(), num_queries * ldd, cuda_stream);

  const size_t vectors_per_query = search_width * graph_degree;
  std::vector<IndexT> tmp_host_result_indices(num_queries * ldd);
  rmm::device_uvector<IndexT> tmp_device_indices(num_queries * ldd, cuda_stream);

  // 为每个query准备一个连续的缓冲区
  std::vector<DataT> query_batch_buffer(vectors_per_query * dataset_desc.stride);
  size_t count_idx = 0;

  for (size_t query_id = 0; query_id < num_queries; query_id++) {
      // 统计这个query需要传输的向量数量
      size_t query_transfer_count = 0;
      const size_t query_base_idx = query_id * ldd;
      
      // 第一遍：收集需要传输的向量
      for (size_t tid = 0; tid < vectors_per_query; tid++) {
          const size_t result_idx = query_base_idx + tid;
          if (host_compute_distance_flags[result_idx]) {
              const IndexT old_index = host_result_indices[result_idx];
              // 复制到连续的缓冲区
              std::memcpy(
                  query_batch_buffer.data() + query_transfer_count * dataset_desc.stride,
                  dataset_desc.ptr + old_index * dataset_desc.stride,
                  dataset_desc.stride * sizeof(DataT)
              );
              tmp_host_result_indices[result_idx] = count_idx + query_transfer_count;
              query_transfer_count++;
          }
      }
      
      // 批量传输这个query的所有向量
      if (query_transfer_count > 0) {
          raft::copy(
              const_cast<DataT*>(dataset_desc.dd_ptr) + count_idx * dataset_desc.stride,
              query_batch_buffer.data(),
              query_transfer_count * dataset_desc.stride,
              cuda_stream
          );
          count_idx += query_transfer_count;
      }
  }

  // 最后传输索引映射
  raft::copy(tmp_device_indices.data(), tmp_host_result_indices.data(), num_queries * ldd, cuda_stream);
  */
 