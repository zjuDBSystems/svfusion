/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "utils.hpp"
#include "hashmap.hpp"
#include "ffanns/neighbors/hd_mapper.hpp"
#include "ffanns/core/host_distance.hpp"
#include "ffanns/distance/distance.hpp"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <cstdint>
#include <tuple>
#include <cassert>
#include <cstdio>

namespace ffanns::neighbors::cagra::detail {

// ========================================================================
// Common structures
// ========================================================================

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
  DataT* host_vector_buffer_ptr;
  unsigned int* num_graph_miss_ptr;
  unsigned int* num_miss1_ptr;
  unsigned int* num_miss2_ptr;
  unsigned iter = 0;
  ffanns::distance::DistanceType metric;
  bool is_internal_search = true;

  uint8_t*       compute_distance_flags_ptr = nullptr;
  unsigned int*  graph_miss_counter_ptr    = nullptr;
  IndexT*        miss_host_graphids_ptr    = nullptr;
  IndexT*        miss_device_graphids_ptr  = nullptr;
  unsigned int*  miss_counter1_vec_ptr      = nullptr;
  unsigned int*  miss_counter1_ptr          = nullptr;
  unsigned int*  miss_counter2_ptr          = nullptr;
  unsigned int*  miss_counter2_write_ptr    = nullptr;
  IndexT*        miss_host_indices1_ptr     = nullptr;
  IndexT*        result_idx_offsets_ptr     = nullptr;
  IndexT*        miss_host_indices2_ptr     = nullptr;
  IndexT*        miss_device_indices_ptr    = nullptr;
  DistanceT*     tmp_device_distances_ptr  = nullptr;

  ComputeDistanceContext(host_device_mapper* hd_mapper_ptr,
                         graph_hd_mapper* graph_mapper_ptr,
                         size_t* hd_status_ptr,
                         int* in_edges_ptr,
                         DistanceT* cpu_distances_ptr_,
                         IndexT* host_indices_ptr_,
                         unsigned int* query_miss_counter_ptr_,
                         IndexT* host_graph_buffer_ptr_,
                         DataT* host_vector_buffer_ptr_,
                         unsigned int* num_graph_miss_ptr_,
                         unsigned int* num_miss1_ptr_,
                         unsigned int* num_miss2_ptr_,
                         unsigned iter_,
                         ffanns::distance::DistanceType metric_,
                         bool is_internal_search_,
                         uint8_t* compute_distance_flags_ptr_,
                         unsigned int* graph_miss_counter_ptr_,
                         IndexT* miss_host_graphids_ptr_,
                         IndexT* miss_device_graphids_ptr_,
                         unsigned int* miss_counter1_vec_ptr_,
                         unsigned int* miss_counter1_ptr_,
                         unsigned int* miss_counter2_ptr_,
                         unsigned int* miss_counter2_write_ptr_,
                         IndexT* miss_host_indices1_ptr_,
                         IndexT* result_idx_offsets_ptr_,
                         IndexT* miss_host_indices2_ptr_,
                         IndexT* miss_device_indices_ptr_,
                         DistanceT* tmp_device_distances_ptr_)
    : hd_mapper(hd_mapper_ptr),
      graph_mapper(graph_mapper_ptr),
      hd_status(hd_status_ptr),
      in_edges(in_edges_ptr),
      cpu_distances_ptr(cpu_distances_ptr_),
      host_indices_ptr(host_indices_ptr_),
      query_miss_counter_ptr(query_miss_counter_ptr_),
      host_graph_buffer_ptr(host_graph_buffer_ptr_),
      host_vector_buffer_ptr(host_vector_buffer_ptr_),
      num_graph_miss_ptr(num_graph_miss_ptr_),
      num_miss1_ptr(num_miss1_ptr_),
      num_miss2_ptr(num_miss2_ptr_),
      iter(iter_),
      metric(metric_),
      is_internal_search(is_internal_search_),
      compute_distance_flags_ptr(compute_distance_flags_ptr_),
      graph_miss_counter_ptr(graph_miss_counter_ptr_),
      miss_host_graphids_ptr(miss_host_graphids_ptr_),
      miss_device_graphids_ptr(miss_device_graphids_ptr_),
      miss_counter1_vec_ptr(miss_counter1_vec_ptr_),
      miss_counter1_ptr(miss_counter1_ptr_),
      miss_counter2_ptr(miss_counter2_ptr_),
      miss_counter2_write_ptr(miss_counter2_write_ptr_),
      miss_host_indices1_ptr(miss_host_indices1_ptr_),
      result_idx_offsets_ptr(result_idx_offsets_ptr_),
      miss_host_indices2_ptr(miss_host_indices2_ptr_),
      miss_device_indices_ptr(miss_device_indices_ptr_),
      tmp_device_distances_ptr(tmp_device_distances_ptr_)
  {}

  ComputeDistanceContext(host_device_mapper* hd_mapper_ptr,
    graph_hd_mapper* graph_mapper_ptr,
    size_t* hd_status_ptr,
    int* in_edges_ptr,
    DistanceT* cpu_distances_ptr_,
    IndexT* host_indices_ptr_,
    unsigned int* query_miss_counter_ptr_,
    IndexT* host_graph_buffer_ptr_,
    DataT* host_vector_buffer_ptr_,
    unsigned int* num_graph_miss_ptr_,
    unsigned int* num_miss1_ptr_,
    unsigned int* num_miss2_ptr_,
    unsigned iter_,
    ffanns::distance::DistanceType metric_,
    bool is_internal_search_)
      : hd_mapper(hd_mapper_ptr),
      graph_mapper(graph_mapper_ptr),
      hd_status(hd_status_ptr),
      in_edges(in_edges_ptr),
      cpu_distances_ptr(cpu_distances_ptr_),
      host_indices_ptr(host_indices_ptr_),
      query_miss_counter_ptr(query_miss_counter_ptr_),
      host_graph_buffer_ptr(host_graph_buffer_ptr_),
      host_vector_buffer_ptr(host_vector_buffer_ptr_),
      num_graph_miss_ptr(num_graph_miss_ptr_),
      num_miss1_ptr(num_miss1_ptr_),
      num_miss2_ptr(num_miss2_ptr_),
      iter(iter_),
      metric(metric_),
      is_internal_search(is_internal_search_)
      {}
};

// ========================================================================
// Host distance function selection
// ========================================================================

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

// ========================================================================
// Random number generation
// ========================================================================

__device__ inline uint32_t splitmix32(uint32_t x) {
  x += 0x9e3779b9u;               // SplitMix32 一步即可
  x = (x ^ (x >> 16)) * 0x85ebca6bu;
  x = (x ^ (x >> 13)) * 0xc2b2ae35u;
  return x ^ (x >> 16);
}

// ========================================================================
// Memory operation utilities
// ========================================================================

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
RAFT_KERNEL get_value_kernel(T* const host_ptr, const T* const dev_ptr)
{
  *host_ptr = *dev_ptr;
}

template <class T>
void get_value(T* const host_ptr, const T* const dev_ptr, cudaStream_t cuda_stream)
{
  get_value_kernel<T><<<1, 1, 0, cuda_stream>>>(host_ptr, dev_ptr);
}

template <class T>
auto get_value(const T* const dev_ptr, cudaStream_t stream) -> T
{
  T value;
  RAFT_CUDA_TRY(cudaMemcpyAsync(&value, dev_ptr, sizeof(value), cudaMemcpyDefault, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  return value;
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
  const auto grid_size = (count * batch_size + block_size - 1) / block_size;
  set_value_batch_kernel<T><<<grid_size, block_size, 0, cuda_stream>>>(
    dev_ptr, ld, val, count, batch_size);
}

// ========================================================================
// Data transfer kernels
// ========================================================================

template <typename DistanceT, typename IndexT>
RAFT_KERNEL scatter_kernel(const DistanceT* __restrict__ src,
                           DistanceT* __restrict__ dst,
                           IndexT* __restrict__ offsets,
                           const int num)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < num) {
        IndexT off = offsets[tid];
        if (off != utils::get_max_value<IndexT>()) {
            dst[off] = src[tid];
            // Reset the offset to sentinel to avoid per-iteration full clears
            offsets[tid] = utils::get_max_value<IndexT>();
        }
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

template <typename DataT, typename IndexT>
RAFT_KERNEL data_scatter_kernel(const DataT* __restrict__ src_buffer,
                           DataT* __restrict__ dst_dataset,
                           const IndexT* __restrict__ device_indices,
                           const uint32_t dim,
                           const uint32_t num_entries)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_entries) return;

    const DataT* src_row = src_buffer + tid * dim;
    DataT* dst_row = dst_dataset + device_indices[tid] * dim;
    for (uint32_t i = 0; i < dim; i++) {
      dst_row[i] = src_row[i];
    }
}

// ========================================================================
// Parent bit manipulation
// ========================================================================

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
    // TODO: 暂时先不清空删除点的id,防止返回invalid id
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


// only for multi_kernel_search
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

template<typename IndexT, typename DistT>
RAFT_KERNEL naive_post_unique(IndexT* __restrict__ idx_base,
                                DistT*  __restrict__ dst_base,
                                uint32_t ld,               // result_buffer_allocation_size
                                uint32_t buf_size,         // result_buffer_size
                                uint32_t topk,
                                uint32_t num_queries)
{
  const uint32_t qid = blockIdx.x * blockDim.x + threadIdx.x;
  if (qid >= num_queries) return;

  idx_base += qid * ld;          // 本 query 的首地址
  dst_base += qid * ld;

  // --------------- 一次线性扫描去重 -----------------
  uint32_t write = 0;            // 写指针（压缩后的位置）
  for (uint32_t read = 0; read < buf_size; ++read) {
      IndexT cur_id = idx_base[read];
      if (cur_id == utils::get_max_value<IndexT>())
          continue;

      bool seen = false;
      for (uint32_t j = 0; j < write; ++j) {
          if (idx_base[j] == cur_id) { seen = true; break; }
      }
      if (seen) continue;

      idx_base[write] = cur_id;
      dst_base[write] = dst_base[read];
      ++write;
      if (write == topk) break;     // 已够 k 个，提前结束
  }
  assert(write == topk);
  // --------------- 不足 k 用哨兵补齐 -----------------
  for (uint32_t i = write; i < topk; ++i) {
      idx_base[i] = utils::get_max_value<IndexT>();
      dst_base[i] = utils::get_max_value<DistT>();
  }
}

}  // namespace ffanns::neighbors::cagra::detail
