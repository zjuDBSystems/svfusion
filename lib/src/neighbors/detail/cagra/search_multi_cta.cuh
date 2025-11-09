#pragma once

#include "compute_distance-ext.cuh"
#include "device_common.hpp"
#include "search_plan.cuh"

#include "../ann_utils.cuh"
#include "ffanns/neighbors/common.hpp"
#include "ffanns/neighbors/cagra.hpp"
#include "topk_for_cagra/topk.h"  //todo replace with raft kernel
#include "search_utils.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/logger-ext.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/core/pinned_mdarray.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace ffanns::neighbors::cagra::detail {
namespace multi_cta_search {

template <class DATASET_DESCRIPTOR_T, class SAMPLE_FILTER_T>
RAFT_KERNEL random_pickup_kernel(
  const DATASET_DESCRIPTOR_T* dataset_desc,
  const typename DATASET_DESCRIPTOR_T::DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const std::size_t num_pickup,                                    // per CTA
  const std::size_t num_cta_per_query,
  const std::size_t num_distilation,
  const uint64_t rand_xor_mask,
  const typename DATASET_DESCRIPTOR_T::INDEX_T* seed_ptr,  // [num_queries, num_seeds]
  const std::uint32_t num_seeds,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const result_indices_ptr,       // [num_queries * num_cta_per_query, ldr]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries * num_cta_per_query, ldr]
  const std::uint32_t ldr,                                                // result_buffer_allocation_size
  typename DATASET_DESCRIPTOR_T::INDEX_T* const visited_hashmap_ptr,      // [num_queries, 1 << visited_hash_bitlen]
  const std::uint32_t visited_hash_bitlen,
  SAMPLE_FILTER_T sample_filter,
  host_device_mapper* hd_mapper,
  const uint32_t smem_ws_size)  // Pass workspace size from host
{
  assert(num_distilation == 1); 

  using DATA_T     = typename DATASET_DESCRIPTOR_T::DATA_T;
  using INDEX_T = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using DISTANCE_T = typename DATASET_DESCRIPTOR_T::DISTANCE_T;
  
  const auto team_size_bits    = dataset_desc->team_size_bitshift();
  const auto team_size          = 1u << team_size_bits;
  const auto visited_ldb        = hashmap::get_size(visited_hash_bitlen);
  const auto global_team_index  = (blockIdx.x * blockDim.x + threadIdx.x) >> team_size_bits;
  const auto query_id = blockIdx.y;

  const auto cta_id = global_team_index / num_pickup;
  const auto pickup_id_in_cta = global_team_index % num_pickup;
  if (global_team_index >= num_pickup * num_cta_per_query) { return; }
  extern __shared__ uint8_t smem[];
  dataset_desc = dataset_desc->setup_workspace(smem, queries_ptr, query_id);
  __syncthreads();  // Ensure all threads have the query data ready

  const uint32_t max_teams_per_block = blockDim.x >> team_size_bits;
  auto* __restrict__ shared_seed_indices = reinterpret_cast<INDEX_T*>(smem + smem_ws_size);
  auto* __restrict__ shared_device_indices = shared_seed_indices + max_teams_per_block;
  const auto team_id_in_block = (threadIdx.x >> team_size_bits);
  const auto lane_id = threadIdx.x & ((1u << team_size_bits) - 1u);

  INDEX_T selected_seed_index = utils::get_max_value<INDEX_T>();
  INDEX_T selected_device_index = utils::get_max_value<INDEX_T>();

  if (lane_id == 0) {
    const unsigned int max_attempts = 1000;
    INDEX_T seed_index;
    // Random seed selection - exactly like kernel1's while loop
    unsigned int attempts = 0;
    while (true) {
      if (attempts >= max_attempts)
        break;
      seed_index = device::xorshift64(global_team_index ^ (attempts << 8) ^ rand_xor_mask) 
                    % dataset_desc->size;
      attempts++;
      
      if constexpr (!std::is_same<SAMPLE_FILTER_T, 
                                  ffanns::neighbors::filtering::none_sample_filter>::value) {
        if (!sample_filter(seed_index))
          continue;
      }
      auto [is_hit, dev_idx] = hd_mapper->get_wo_replace_safe(seed_index, false);
      if (is_hit) {
        selected_seed_index = seed_index;
        selected_device_index = dev_idx;
        break;
      }
    }
    // Store in shared memory (if not found, will store MAX_VALUE)
    shared_seed_indices[team_id_in_block] = selected_seed_index;
    shared_device_indices[team_id_in_block] = selected_device_index;
  }
  __syncwarp();

  selected_seed_index = shared_seed_indices[team_id_in_block];
  selected_device_index = shared_device_indices[team_id_in_block];
  const auto store_gmem_index = ldr * (query_id * num_cta_per_query + cta_id) +  pickup_id_in_cta;

  if (selected_seed_index != utils::get_max_value<INDEX_T>()) {
    DISTANCE_T computed_distance = dataset_desc->compute_distance(selected_device_index, true);
    if (lane_id == 0) {
      // Per-CTA visited hashmap to avoid cross-CTA interference for the same query
      const uint32_t global_cta_id = static_cast<uint32_t>(query_id) * static_cast<uint32_t>(num_cta_per_query) + static_cast<uint32_t>(cta_id);
      if (hashmap::insert(visited_hashmap_ptr + (visited_ldb * global_cta_id), visited_hash_bitlen, selected_seed_index)) {
        result_indices_ptr[store_gmem_index] = selected_seed_index;
        result_distances_ptr[store_gmem_index] = computed_distance;
      } else {
        // Already visited - mark as invalid (like kernel2)
        result_indices_ptr[store_gmem_index] = utils::get_max_value<INDEX_T>();
        result_distances_ptr[store_gmem_index] = utils::get_max_value<DISTANCE_T>();
      }
    }
  } else {
    // No valid seed found in kernel1 phase - store MAX_VALUE
    if (lane_id == 0) {
      result_indices_ptr[store_gmem_index] = utils::get_max_value<INDEX_T>();
      result_distances_ptr[store_gmem_index] = utils::get_max_value<DISTANCE_T>();
    }
  }
  
}

template <typename DataT, typename IndexT, typename DistanceT, typename SAMPLE_FILTER_T>
void random_pickup(const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
                         const DataT* queries_ptr,
                         std::size_t num_queries,
                         std::size_t num_pickup,
                         std::size_t num_cta_per_query,
                         unsigned num_distilation,
                         uint64_t rand_xor_mask,
                         const IndexT* seed_ptr,
                         std::uint32_t num_seeds,
                         IndexT* result_indices_ptr,
                         DistanceT* result_distances_ptr,
                         std::size_t ldr,
                         IndexT* visited_hashmap_ptr,
                         std::uint32_t visited_hash_bitlen,
                         SAMPLE_FILTER_T sample_filter,
                         host_device_mapper* hd_mapper,
                         cudaStream_t cuda_stream)
{
  const auto block_size = 256u;
  const auto num_teams_per_threadblock = block_size / dataset_desc.team_size;
  const dim3 grid_size((num_pickup * num_cta_per_query + num_teams_per_threadblock - 1) / num_teams_per_threadblock,
                       num_queries);
  
  // Calculate shared memory size
  // Need space for: 2 * max_teams_per_block * sizeof(IndexT) + dataset workspace
  const size_t shared_indices_size = 2 * num_teams_per_threadblock * sizeof(IndexT);
  const size_t total_smem_size = shared_indices_size + dataset_desc.smem_ws_size_in_bytes;
  
  random_pickup_kernel<<<grid_size, block_size, total_smem_size, cuda_stream>>>(
    dataset_desc.dev_ptr(cuda_stream),
    queries_ptr,
    num_pickup,
    num_cta_per_query,
    num_distilation,
    rand_xor_mask,
    seed_ptr,
    num_seeds,
    result_indices_ptr,
    result_distances_ptr,
    ldr,
    visited_hashmap_ptr,
    visited_hash_bitlen,
    sample_filter,
    hd_mapper,
    dataset_desc.smem_ws_size_in_bytes);  // Pass the workspace size
    
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <class INDEX_T>
RAFT_KERNEL pickup_next_parents_kernel(
  INDEX_T* const result_indices_ptr,           // row base: [num_queries * num_cta_per_query, lds]
  float*   const result_distances_ptr,         // row base distances: same layout as indices
  const std::size_t lds,                       // stride per CTA (>= itopk_size)
  const std::uint32_t parent_candidates_size,  // itopk_size per CTA
  INDEX_T* const traversed_hashmap_ptr,        // [num_queries, 1 << hash_bitlen]
  const std::size_t hash_bitlen,
  INDEX_T* const parent_list_ptr,              // [num_queries * num_cta_per_query, ldd]
  const std::size_t ldd,                       // search_width (stride per global CTA)
  const std::size_t parent_list_size,          // search_width (parents per CTA)
  std::uint32_t* const terminate_flag,
  const std::uint32_t num_cta_per_query,       // For Multi-CTA coordination
  INDEX_T* const visited_hashmap_ptr,          // [num_queries * num_cta_per_query, 1 << small_hash_bitlen]
  const std::size_t visited_hash_bitlen,       // small-hash bitlen
  const std::uint32_t old_half_offset,         // absolute old-half offset in row
  const std::uint32_t new_half_offset,         // absolute new-half offset in row
  unsigned int* graph_miss_counter, unsigned int* miss_counter1_vec, unsigned int* miss_counter1,
  unsigned int* miss_counter2, unsigned int* miss_counter2_write)         
{
  const std::uint32_t global_cta_id = blockIdx.x;
  const std::uint32_t query_id = global_cta_id / num_cta_per_query;
  
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  const std::size_t ldb   = hashmap::get_size(hash_bitlen);
  const std::size_t visited_ldb = hashmap::get_size(visited_hash_bitlen);
  INDEX_T* query_hashmap = traversed_hashmap_ptr + (ldb * query_id);
  
  INDEX_T* cta_parent_list = parent_list_ptr + (ldd * global_cta_id);
  INDEX_T* row_idx_base    = result_indices_ptr + (lds * global_cta_id);
  INDEX_T* cta_candidates  = row_idx_base + new_half_offset;
  if (threadIdx.x == 0) {
    miss_counter1_vec[global_cta_id] = 0;
    if (global_cta_id == 0) {
      *graph_miss_counter = 0;
      *miss_counter1 = 0;
      *miss_counter2 = 0;
      *miss_counter2_write = 0;
    }
  }
  if (threadIdx.x < 32) {
    for (std::uint32_t i = threadIdx.x; i < parent_list_size; i += 32) {
      cta_parent_list[i] = utils::get_max_value<INDEX_T>();
    }
    std::uint32_t parent_candidates_size_max = parent_candidates_size;
    if (parent_candidates_size % 32) {
      parent_candidates_size_max += 32 - (parent_candidates_size % 32);
    }
    
    std::uint32_t num_new_parents = 0;
    for (std::uint32_t j = threadIdx.x; j < parent_candidates_size_max; j += 32) {
      // Step 1: Each thread checks if its candidate is valid (not used and not MAX)
      INDEX_T index;
      int is_candidate = 0;
      if (j < parent_candidates_size) {
        index = cta_candidates[j];
        // Check if not already used (MSB clear) and valid
        if ((index != utils::get_max_value<INDEX_T>()) && 
            ((index & index_msb_1_mask) == 0)) {
          is_candidate = 1;  // This is a valid candidate
        }
      }
      // Step 2: Collect all valid candidates using ballot
      const auto ballot_mask = __ballot_sync(0xffffffff, is_candidate);
      const auto candidate_id = __popc(ballot_mask & ((1u << threadIdx.x) - 1u));
      
      // Step 3: Process candidates in order, trying to insert into hashmap
      for (int k = 0; k < __popc(ballot_mask); k++) {
        int parent_added = 0;
        // When k matches this thread's candidate_id, this thread tries to insert
        if (is_candidate && (k == candidate_id)) {
          if (hashmap::insert(query_hashmap, hash_bitlen, index)) {
            if (num_new_parents < parent_list_size) {
              cta_parent_list[num_new_parents] = j;
              cta_candidates[j] |= index_msb_1_mask;  // Mark as used
              parent_added = 1;
            }
          }
          // If insert failed, this node was already selected by another CTA
          // Continue to next candidate
        }
        // Check if any thread added a parent
        if (__any_sync(0xffffffff, parent_added)) {
          num_new_parents++;
        }
        if (num_new_parents >= parent_list_size) { break; }
      }
      
      if (num_new_parents >= parent_list_size) { break; }
    }
    if ((num_new_parents > 0) && (threadIdx.x == 0)) { *terminate_flag = 0; }
  } else {
    hashmap::init(visited_hashmap_ptr + (visited_ldb * global_cta_id), visited_hash_bitlen, 32);
  }

  __syncthreads();
  // insert internal-topk indices into visited_hashmap
  for (unsigned i = threadIdx.x; i < parent_candidates_size; i += blockDim.x) {
    auto key = cta_candidates[i] & ~index_msb_1_mask;  // clear most significant bit
    if (key != utils::get_max_value<INDEX_T>()) {
      hashmap::insert(visited_hashmap_ptr + (visited_ldb * global_cta_id), visited_hash_bitlen, key);
    }
  }

  __syncthreads();
  // Optional: prune kicked parents here using visited membership (O(1))
  {
    INDEX_T* row_idx = row_idx_base;
    float*   row_dst = result_distances_ptr + (static_cast<std::size_t>(global_cta_id) * lds);
    INDEX_T* cta_visited = visited_hashmap_ptr + (visited_ldb * global_cta_id);
    for (std::uint32_t i = threadIdx.x; i < parent_candidates_size; i += blockDim.x) {
      const std::size_t off_old = static_cast<std::size_t>(old_half_offset) + i;
      INDEX_T v = row_idx[off_old];
      if (v == utils::get_max_value<INDEX_T>()) continue;
      if ((v & index_msb_1_mask) == 0) continue;  // only prune used parents
      const INDEX_T key = (v & ~index_msb_1_mask);
      bool in_new = hashmap::search<INDEX_T, 0>(cta_visited, visited_hash_bitlen, key);
      if (!in_new) {
        hashmap::remove<INDEX_T>(query_hashmap, hash_bitlen, key);
        row_idx[off_old] = utils::get_max_value<INDEX_T>();
        row_dst[off_old] = utils::get_max_value<float>();
      }
    }
  }
}

template <class INDEX_T>
void pickup_next_parents(INDEX_T* const result_indices_ptr,  // row base: [num_queries * num_cta_per_query, lds]
                         float*   const result_distances_ptr, // row base distances
                         const std::size_t lds,                 // stride per CTA (>= itopk_size)
                         const std::size_t parent_candidates_size,  // itopk_size per CTA
                         const std::size_t num_queries,
                         INDEX_T* const hashmap_ptr,           // traversed_hashmap [num_queries, 1 << hash_bitlen]
                         const std::size_t hash_bitlen,
                         INDEX_T* const parent_list_ptr,       // [num_queries * num_cta_per_query, ldd]
                         const std::size_t ldd,                // search_width (stride per global CTA)
                         const std::size_t parent_list_size,   // search_width per CTA
                         std::uint32_t* const terminate_flag,
                         const std::uint32_t num_cta_per_query, // Multi-CTA parameter
                         INDEX_T* const visited_hashmap_ptr,
                         const std::size_t visited_hash_bitlen,
                         const std::uint32_t old_half_offset,
                         const std::uint32_t new_half_offset,
                         unsigned int* graph_miss_counter, unsigned int* miss_counter1_vec, unsigned int* miss_counter1,
                         unsigned int* miss_counter2, unsigned int* miss_counter2_write,
                         cudaStream_t cuda_stream = 0)
{
  // Grid configuration for Multi-CTA:
  // - Each block processes one CTA (not one query as in multi-kernel)
  const std::uint32_t grid_size = num_queries * num_cta_per_query;
  const std::uint32_t block_size = 128; //original 32
  pickup_next_parents_kernel<INDEX_T>
    <<<grid_size, block_size, 0, cuda_stream>>>(result_indices_ptr,
                                                result_distances_ptr,
                                                lds,
                                                parent_candidates_size,
                                                hashmap_ptr,
                                                hash_bitlen,
                                                parent_list_ptr,
                                                ldd,
                                                parent_list_size,
                                                terminate_flag,
                                                num_cta_per_query,
                                                visited_hashmap_ptr, visited_hash_bitlen,
                                                old_half_offset, new_half_offset,
                                                graph_miss_counter, miss_counter1_vec, miss_counter1, miss_counter2, miss_counter2_write);
}

template <class DATASET_DESCRIPTOR_T>
RAFT_KERNEL compute_distance_to_child_nodes_kernel1(
  const typename DATASET_DESCRIPTOR_T::INDEX_T* const parent_node_list,  // [num_queries * num_cta_per_query, search_width]
  typename DATASET_DESCRIPTOR_T::INDEX_T* const parent_candidates_ptr,  // [num_queries * num_cta_per_query, search_width]
  const std::size_t lds,
  const std::uint32_t search_width,
  const DATASET_DESCRIPTOR_T* dataset_desc,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const neighbor_graph_ptr, // [device graph_size, graph_degree]
  const std::uint32_t graph_degree,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const result_indices_ptr,       // [num_queries * num_cta_per_query, ldd]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries * num_cta_per_query, ldd]
  const std::uint32_t ldd,  // (*) ldd >= search_width * graph_degree
  uint8_t* compute_distance_flags_ptr,
  graph_hd_mapper* graph_mapper,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const miss_host_graphids,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const miss_device_graphids,
  unsigned int* const miss_counter,
  const std::uint32_t num_queries,
  const std::uint32_t num_cta_per_query)
{
  using INDEX_T    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using DISTANCE_T = typename DATASET_DESCRIPTOR_T::DISTANCE_T;
  
  const auto global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const auto total_ctas = num_queries * num_cta_per_query;
  const auto total_search_pairs = total_ctas * search_width; 
  if (global_thread_id >= total_search_pairs) return;

  // Each global CTA processes its own parent list row
  const std::uint32_t global_cta_id = global_thread_id / search_width;
  const std::uint32_t local_search_id = global_thread_id % search_width;

  const std::size_t parent_list_index =
    parent_node_list[local_search_id + (search_width * global_cta_id)];

  if (parent_list_index == utils::get_max_value<INDEX_T>()) { 
    for (uint32_t idx = 0; idx < graph_degree; idx++) {
      const size_t result_idx = global_cta_id * ldd + local_search_id * graph_degree + idx;
      result_distances_ptr[result_idx] = utils::get_max_value<DISTANCE_T>();
      compute_distance_flags_ptr[result_idx] = false;
      result_indices_ptr[result_idx] = utils::get_max_value<INDEX_T>();   
    }
    return; 
  }

  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  const auto raw_parent_index        = parent_candidates_ptr[parent_list_index + (lds * global_cta_id)];

  if (raw_parent_index == utils::get_max_value<INDEX_T>()) {
    for (uint32_t idx = 0; idx < graph_degree; idx++) {
      const size_t result_idx = global_cta_id * ldd + local_search_id * graph_degree + idx;
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

  // if (is_hit) 
  for (uint32_t idx = 0; idx < graph_degree; idx++) {
    const size_t result_idx = global_cta_id * ldd + local_search_id * graph_degree + idx;
    const size_t child_id_offset = device_parent_index * graph_degree + idx;
    result_indices_ptr[result_idx] = child_id_offset;
  }  
}

template <class DATASET_DESCRIPTOR_T, class SAMPLE_FILTER_T>
RAFT_KERNEL compute_distance_to_child_nodes_kernel2(
  const std::uint32_t search_width,
  const DATASET_DESCRIPTOR_T* dataset_desc,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const 
    neighbor_graph_ptr, // [device graph_size, graph_degree]
  const std::uint32_t graph_degree,
  const typename DATASET_DESCRIPTOR_T::DATA_T* query_ptr,  // [num_queries, data_dim]
  typename DATASET_DESCRIPTOR_T::INDEX_T* const visited_hashmap_ptr,  // [num_queries, 1 << small_hash_bitlen]
  const std::uint32_t visited_hash_bitlen,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const traversed_hashmap_ptr,
  std::uint32_t traversed_hash_bitlen,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const result_indices_ptr,       // [num_queries, ldd]
  typename DATASET_DESCRIPTOR_T::INDEX_T* const d_result_indices_ptr,
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries, ldd]
  const std::uint32_t ldd,  // (*) ldd >= search_width * graph_degree
  SAMPLE_FILTER_T sample_filter,
  const std::uint32_t num_cta_per_query,
  uint8_t* compute_distance_flags_ptr,
  host_device_mapper* hd_mapper,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const miss_host_indices1,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const miss_result_idx_offsets,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const miss_host_indices2,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const miss_device_indices,
  unsigned int* const miss_counter1_vec,
  unsigned int* const miss_counter1,
  unsigned int* const miss_counter2,
  unsigned int* const miss_counter2_write,
  int* const in_edges,
  unsigned int iter,
  bool is_internal_search)
{
  using INDEX_T    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using DISTANCE_T = typename DATASET_DESCRIPTOR_T::DISTANCE_T;

  const auto team_size_bits = dataset_desc->team_size_bitshift();
  const auto team_size      = 1u << team_size_bits;
  
  // Calculate hashmap sizes
  const uint32_t visited_ldb = hashmap::get_size(visited_hash_bitlen);
  const uint32_t traversed_ldb = hashmap::get_size(traversed_hash_bitlen);

  const auto global_team_id = threadIdx.x + blockDim.x * blockIdx.x;
  
  // Multi-CTA: blockIdx.y is global_cta_id, not query_id
  const auto global_cta_id = blockIdx.y;
  const auto query_id = global_cta_id / num_cta_per_query;
  // const auto cta_id_in_query = global_cta_id % num_cta_per_query;

  if (global_team_id >= search_width * graph_degree) { return; }
  
  // Result index is based on global_cta_id, not query_id
  const std::size_t result_idx = ldd * global_cta_id + global_team_id;
  const std::size_t child_id_offset = result_indices_ptr[result_idx];
  if (child_id_offset == utils::get_max_value<INDEX_T>()) {
    compute_distance_flags_ptr[result_idx] = 0;
    return;
  }
  const std::size_t child_id = neighbor_graph_ptr[child_id_offset];
  result_indices_ptr[result_idx] = child_id;
   
  // !!! remove filter in internal iteration
  if constexpr (!std::is_same<SAMPLE_FILTER_T,
                              ffanns::neighbors::filtering::none_sample_filter>::value) {
    if (is_internal_search && (!sample_filter(child_id))) {
      result_distances_ptr[result_idx] = utils::get_max_value<DISTANCE_T>();
      compute_distance_flags_ptr[result_idx] = 0;
      return;
    }
  }

  const auto miss_counter_offset = global_cta_id * search_width * graph_degree;
  auto max_transfer_size = 4096;

  // Double hashmap check for Multi-CTA
  // Step 1: Check visited_hashmap (per-iteration dedup within query)
  const auto visited_check = hashmap::insert<INDEX_T>(
    visited_hashmap_ptr + (visited_ldb * global_cta_id), visited_hash_bitlen, child_id);
  if (!visited_check) {
    // Already in visited hashmap (processed by this or another CTA in this iteration)
    result_distances_ptr[result_idx] = utils::get_max_value<DISTANCE_T>();
    compute_distance_flags_ptr[result_idx] = 0;
    return;
  }
  
  // Step 2: Check traversed_hashmap (global dedup across all iterations)
  const auto traversed_check = hashmap::search<INDEX_T, 0>(
    traversed_hashmap_ptr + (traversed_ldb * query_id), traversed_hash_bitlen, child_id); 
  if (traversed_check) {
    // Already in traversed hashmap (processed in previous iterations)
    result_distances_ptr[result_idx] = utils::get_max_value<DISTANCE_T>();
    compute_distance_flags_ptr[result_idx] = 0;
    return;
  }
  compute_distance_flags_ptr[result_idx] = 1;
  auto [is_hit, device_index] = hd_mapper->get_wo_replace(child_id, true);
  if (!is_hit) {
    unsigned int tmp_pos2 = atomicAdd(miss_counter2, 1);
    if (tmp_pos2 >= max_transfer_size-1) {
      // directly reduce to cpu computation
      atomicSub(miss_counter2, 1);
      unsigned int pos1 = atomicAdd(&miss_counter1_vec[global_cta_id], 1);
      atomicAdd(miss_counter1, 1);
      miss_host_indices1[miss_counter_offset + pos1] = child_id;
      miss_result_idx_offsets[miss_counter_offset + pos1] = result_idx;
      compute_distance_flags_ptr[result_idx] = 0;
    } else {
      auto result = hd_mapper->replace(child_id, in_edges, is_internal_search);
      is_hit = result.first;
      device_index = result.second;
      if (!is_hit) {
        atomicSub(miss_counter2, 1);
        unsigned int pos1 = atomicAdd(&miss_counter1_vec[global_cta_id], 1);
        atomicAdd(miss_counter1, 1);
        miss_host_indices1[miss_counter_offset + pos1] = child_id;
        miss_result_idx_offsets[miss_counter_offset + pos1] = result_idx;
        compute_distance_flags_ptr[result_idx] = 0;
      } else { // sucessfully replaced
        unsigned int pos2 = atomicAdd(miss_counter2_write, 1);
        miss_host_indices2[pos2] = child_id;
        miss_device_indices[pos2] = device_index;
      }
    }
  }
  d_result_indices_ptr[result_idx] = device_index;
}

template <class DATASET_DESCRIPTOR_T>
RAFT_KERNEL compute_distance_to_child_nodes_kernel3(
  const std::uint32_t search_width,
  const DATASET_DESCRIPTOR_T* dataset_desc,
  const std::uint32_t graph_degree,
  const typename DATASET_DESCRIPTOR_T::DATA_T* query_ptr,  // [num_queries, data_dim]
  typename DATASET_DESCRIPTOR_T::INDEX_T* const tmp_result_indices_ptr,  // [num_queries * num_cta_per_query, ldd]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries * num_cta_per_query, ldd]
  const std::uint32_t ldd,  // (*) ldd >= search_width * graph_degree
  uint8_t* compute_distance_flags_ptr,
  const std::uint32_t num_cta_per_query)  // Multi-CTA parameter
{
  using INDEX_T    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using DISTANCE_T = typename DATASET_DESCRIPTOR_T::DISTANCE_T;

  const auto team_size_bits = dataset_desc->team_size_bitshift();
  const auto team_size      = 1u << team_size_bits;
  const auto tid            = threadIdx.x + blockDim.x * blockIdx.x;
  const auto global_team_id = tid >> team_size_bits;
  const auto global_cta_id = blockIdx.y;
  const auto query_id = global_cta_id / num_cta_per_query;

  extern __shared__ uint8_t smem[];
  // Load the query for this CTA's query_id
  dataset_desc = dataset_desc->setup_workspace(smem, query_ptr, query_id);

  __syncthreads();
  if (global_team_id >= search_width * graph_degree) { return; }

  // Use global_cta_id for memory indexing (not query_id)
  const auto result_idx = ldd * global_cta_id + global_team_id;
  const auto compute_distance_flag = compute_distance_flags_ptr[result_idx];

  if (!compute_distance_flag) { return; }
  const std::size_t tmp_child_id = tmp_result_indices_ptr[result_idx];
  DISTANCE_T norm2               = dataset_desc->compute_distance(tmp_child_id, compute_distance_flag);
  if ((threadIdx.x & (team_size - 1)) == 0) {
    result_distances_ptr[result_idx] = norm2;
  }
}

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          class SAMPLE_FILTER_T>
void compute_distance_to_child_nodes(
  const IndexT* parent_node_list,        // [num_queries * num_cta_per_query, search_width]
  IndexT* const parent_candidates_ptr,   // [num_queries * num_cta_per_query, search_width]
  DistanceT* const parent_distance_ptr,  // [num_queries * num_cta_per_query, search_width]
  std::size_t lds,
  std::uint32_t search_width,
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  const IndexT* neighbor_graph_ptr,  // [dataset_size, graph_degree]
  IndexT* d_neighbor_graph_ptr,
  std::uint32_t graph_degree,
  const DataT* query_ptr,  // [num_queries, data_dim]
  const DataT* host_query_ptr,
  std::uint32_t num_queries,
  std::uint32_t num_cta_per_query,
  IndexT* visited_hashmap_ptr,  // [num_queries, 1 << small_hash_bitlen]
  std::uint32_t visited_hash_bitlen,
  IndexT* traversed_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  std::uint32_t traversed_hash_bitlen,
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

  // Stage timing for profiling: k1, graph, k2, data, k3
  // cudaEvent_t ev_start, ev_k1_end, ev_graph_end, ev_k2_end, ev_data_end, ev_k3_end;
  // cudaEventCreate(&ev_start);
  // cudaEventCreate(&ev_k1_end);
  // cudaEventCreate(&ev_graph_end);
  // cudaEventCreate(&ev_k2_end);
  // cudaEventCreate(&ev_data_end);
  // cudaEventCreate(&ev_k3_end);
  // cudaEventRecord(ev_start, cuda_stream);

  const auto block_size      = 128;
  const auto teams_per_block = block_size / dataset_desc.team_size;
  const auto total_ctas = num_queries * num_cta_per_query;
  
  const auto grid_size1 = (total_ctas * search_width + block_size - 1) / block_size;
  const dim3 grid_size2((search_width * graph_degree + block_size - 1) / block_size, total_ctas);
  const dim3 grid_size3((search_width * graph_degree + teams_per_block - 1) / teams_per_block, total_ctas);  

  auto compute_distance_flags = compute_ctx->compute_distance_flags_ptr;
  auto graph_miss_counter = compute_ctx->graph_miss_counter_ptr;
  auto miss_host_graphids = compute_ctx->miss_host_graphids_ptr;
  auto miss_device_graphids = compute_ctx->miss_device_graphids_ptr;
  
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
                                                        compute_distance_flags,
                                                        graph_mapper,
                                                        miss_host_graphids,
                                                        miss_device_graphids,
                                                        graph_miss_counter,
                                                        num_queries,
                                                        num_cta_per_query);
                                               
  // cudaEventRecord(ev_k1_end, cuda_stream);
  auto num_graph_miss_ptr = compute_ctx->num_graph_miss_ptr;
  RAFT_CUDA_TRY(cudaMemcpyAsync(num_graph_miss_ptr, graph_miss_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost, cuda_stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(cuda_stream));
  unsigned int num_graph_miss = *num_graph_miss_ptr;
  if (num_graph_miss > 0) {
    std::vector<IndexT> host_graphids(num_graph_miss);
    auto host_graph_buffer =  compute_ctx->host_graph_buffer_ptr;
    lightweight_uvector<IndexT> device_graph_buffer(res, num_graph_miss * graph_degree, cuda_stream);
    raft::copy(host_graphids.data(), miss_host_graphids, num_graph_miss, cuda_stream);
    for (size_t i = 0; i < num_graph_miss; i++) {
        size_t miss_graph_id = host_graphids[i];
        // RAFT_LOG_INFO("[compute_distance_to_child_nodes] miss_graph_id: %u", miss_graph_id);
        const IndexT* neighbor_list_head = neighbor_graph_ptr + miss_graph_id * graph_degree;
        memcpy(host_graph_buffer + i * graph_degree, neighbor_list_head, graph_degree * sizeof(IndexT));
    }
    raft::copy(device_graph_buffer.data(), host_graph_buffer, num_graph_miss * graph_degree, cuda_stream);
    int threadsPerBlock = 128;
    int blocks = (num_graph_miss + threadsPerBlock - 1) / threadsPerBlock;
    graph_scatter_kernel<<<blocks, threadsPerBlock, 0, cuda_stream>>>(device_graph_buffer.data(),
                                                                  d_neighbor_graph_ptr,
                                                                  miss_device_graphids,
                                                                  graph_degree,
                                                                  num_graph_miss);
    compute_ctx->hd_status[2] += num_graph_miss;
  }
  compute_ctx->hd_status[3] += total_ctas * search_width;

  auto miss_counter1_vec = compute_ctx->miss_counter1_vec_ptr;
  auto miss_counter1 = compute_ctx->miss_counter1_ptr;
  auto miss_counter2 = compute_ctx->miss_counter2_ptr;
  auto miss_counter2_write = compute_ctx->miss_counter2_write_ptr;
  auto miss_host_indices1 = compute_ctx->miss_host_indices1_ptr;
  auto result_idx_offsets = compute_ctx->result_idx_offsets_ptr;
  auto miss_host_indices2 = compute_ctx->miss_host_indices2_ptr;
  auto miss_device_indices = compute_ctx->miss_device_indices_ptr;
 
  // cudaEventRecord(ev_graph_end, cuda_stream);
  compute_distance_to_child_nodes_kernel2<<<grid_size2,
                                           block_size,
                                           dataset_desc.smem_ws_size_in_bytes,
                                           cuda_stream>>>(search_width,
                                                      dataset_desc.dev_ptr(cuda_stream),
                                                      d_neighbor_graph_ptr,
                                                      graph_degree,
                                                      query_ptr,
                                                      visited_hashmap_ptr,
                                                      visited_hash_bitlen,
                                                      traversed_hashmap_ptr,
                                                      traversed_hash_bitlen,
                                                      result_indices_ptr,
                                                      d_result_indices_ptr,
                                                      result_distances_ptr,
                                                      ldd, 
                                                      sample_filter,
                                                      num_cta_per_query,
                                                      compute_distance_flags,
                                                      hd_mapper,
                                                      miss_host_indices1,
                                                      result_idx_offsets,
                                                      miss_host_indices2,
                                                      miss_device_indices,
                                                      miss_counter1_vec,
                                                      miss_counter1,
                                                      miss_counter2,
                                                      miss_counter2_write,
                                                      compute_ctx->in_edges,
                                                      compute_ctx->iter,
                                                      compute_ctx->is_internal_search);

// cudaEventRecord(ev_k2_end, cuda_stream);
  auto num_miss1_ptr = compute_ctx->num_miss1_ptr;
  auto num_miss2_ptr = compute_ctx->num_miss2_ptr;
  RAFT_CUDA_TRY(cudaMemcpyAsync(num_miss1_ptr, miss_counter1, sizeof(unsigned int), cudaMemcpyDeviceToHost, cuda_stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(num_miss2_ptr, miss_counter2, sizeof(unsigned int), cudaMemcpyDeviceToHost, cuda_stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(cuda_stream));
  unsigned int num_miss1 = *num_miss1_ptr;
  unsigned int num_miss2 = *num_miss2_ptr;
  auto tmp_device_distances = compute_ctx->tmp_device_distances_ptr;
  if (num_miss1 > 0) {
    auto cpu_distances =  compute_ctx->cpu_distances_ptr;
    auto host_indices = compute_ctx->host_indices_ptr;
    auto query_miss_counter = compute_ctx->query_miss_counter_ptr;
    raft::copy(host_indices, miss_host_indices1, total_ctas * search_width * graph_degree, cuda_stream);
    raft::copy(query_miss_counter, miss_counter1_vec, total_ctas, cuda_stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(cuda_stream));

#pragma omp parallel for num_threads(32) schedule(dynamic, 1)
    for (size_t cta_id = 0; cta_id < total_ctas ; cta_id++) {
      size_t qid = cta_id / num_cta_per_query;
      size_t num_miss = query_miss_counter[cta_id];
      if (num_miss == 0) continue;
      const auto query_offset = cta_id * search_width * graph_degree;
      const DataT* query_data = host_query_ptr + qid * dataset_desc.stride;
      const DataT* child_data;
      for (size_t j = 0; j < num_miss; j++) {
        size_t idx = query_offset + j;
        size_t miss_child_id = host_indices[idx];
        child_data = dataset_desc.ptr + miss_child_id * dataset_desc.stride;
        cpu_distances[idx] = host_dist_fn(query_data, child_data, dataset_desc.stride);
      }
    }
    raft::copy(tmp_device_distances, cpu_distances, total_ctas * search_width * graph_degree, cuda_stream);   
    compute_ctx->hd_status[0] += num_miss1;
  }
  auto test_num_miss2 = get_value<unsigned int>(miss_counter2_write, cuda_stream);
  assert(num_miss2 == test_num_miss2);
  if (num_miss2 > 0) {
    std::vector<IndexT> host_indices(num_miss2);
    raft::copy(host_indices.data(), miss_host_indices2, num_miss2, cuda_stream);
    auto host_vector_buffer = compute_ctx->host_vector_buffer_ptr;
    auto device_vector_buffer = 
        raft::make_device_matrix<DataT, int64_t, raft::row_major>(res,
                                                                  num_miss2,
                                                                  dataset_desc.stride);
    for (size_t i = 0; i < num_miss2; i++) {
      size_t miss_child_id = host_indices[i];
      const DataT* child_data = dataset_desc.ptr + miss_child_id * dataset_desc.stride;
      memcpy(host_vector_buffer + i * dataset_desc.stride, child_data, dataset_desc.stride * sizeof(DataT));
    }
    raft::copy(device_vector_buffer.data_handle(), host_vector_buffer, num_miss2 * dataset_desc.stride, cuda_stream);
    int threadsPerBlock = 128;
    int blocks = (num_miss2 + threadsPerBlock - 1) / threadsPerBlock;
    data_scatter_kernel<<<blocks, threadsPerBlock, 0, cuda_stream>>>(device_vector_buffer.data_handle(),
                                                              const_cast<DataT*>(dataset_desc.dd_ptr),
                                                              miss_device_indices,
                                                              dataset_desc.stride,
                                                              num_miss2);
    compute_ctx->hd_status[0] += num_miss2;
  }
  compute_ctx->hd_status[1] += total_ctas * search_width * graph_degree;
  // cudaEventRecord(ev_data_end, cuda_stream);

  compute_distance_to_child_nodes_kernel3<<<grid_size3,
                                           block_size,
                                           dataset_desc.smem_ws_size_in_bytes,
                                           cuda_stream>>>(search_width,
                                                          dataset_desc.dev_ptr(cuda_stream),
                                                          graph_degree,
                                                          query_ptr,
                                                          d_result_indices_ptr,
                                                          result_distances_ptr,
                                                          ldd,
                                                          compute_distance_flags,
                                                          num_cta_per_query);
  // RAFT_CUDA_TRY(cudaStreamSynchronize(cuda_stream));
  // cudaEventRecord(ev_k3_end, cuda_stream);
  
  if (num_miss1 > 0) {
    // RAFT_CUDA_TRY(cudaStreamSynchronize(cuda_stream));
    int threadsPerBlock = 128;
    int blocks = (total_ctas * search_width * graph_degree + threadsPerBlock - 1) / threadsPerBlock;
    scatter_kernel<<<blocks, threadsPerBlock, 0, cuda_stream>>>(tmp_device_distances,
                                                                  result_distances_ptr,
                                                                  result_idx_offsets,
                                                                  total_ctas * search_width * graph_degree);
  }

  // Accumulate timings into the provided accumulator
  // if (time_accumulator) {
  //   float t_k1=0.f, t_graph=0.f, t_k2=0.f, t_data=0.f, t_k3=0.f, t_total=0.f;
  //   cudaEventElapsedTime(&t_k1,   ev_start,    ev_k1_end);
  //   cudaEventElapsedTime(&t_graph,ev_k1_end,   ev_graph_end);
  //   cudaEventElapsedTime(&t_k2,   ev_graph_end,ev_k2_end);
  //   cudaEventElapsedTime(&t_data, ev_k2_end,   ev_data_end);
  //   cudaEventElapsedTime(&t_k3,   ev_data_end, ev_k3_end);
  //   t_total = t_k1 + t_graph + t_k2 + t_data + t_k3;
  //   time_accumulator->add(t_k1, t_graph, t_k2, 
  //     t_data, t_k3, t_total);
  // }

  // cudaEventDestroy(ev_start);
  // cudaEventDestroy(ev_k1_end);
  // cudaEventDestroy(ev_graph_end);
  // cudaEventDestroy(ev_k2_end);
  // cudaEventDestroy(ev_data_end);
  // cudaEventDestroy(ev_k3_end);

}

template <typename INDEX_T, typename DISTANCE_T>
RAFT_KERNEL gather_cta_results_kernel(const INDEX_T* result_indices_ptr,
                                      const DISTANCE_T* result_distances_ptr,
                                      INDEX_T* gathered_indices_ptr,
                                      DISTANCE_T* gathered_distances_ptr,
                                      const uint32_t itopk_size,
                                      const uint32_t num_cta_per_query,
                                      const uint32_t result_buffer_allocation_size,
                                      const uint32_t visited_hash_bitlen,
                                      const uint32_t /*total_elements_unused*/)
{
  // One block per query
  const uint32_t query_id = blockIdx.x;
  const uint32_t candidates_per_query = num_cta_per_query * itopk_size;
  const size_t dst_base = static_cast<size_t>(query_id) * candidates_per_query;

  extern __shared__ unsigned char smem[];
  auto* __restrict__ hash_tbl = reinterpret_cast<INDEX_T*>(smem);
  const INDEX_T kInvalid = utils::get_max_value<INDEX_T>();
  const DISTANCE_T kInf = utils::get_max_value<DISTANCE_T>();

  // Initialize per-block hashmap in shared memory
  hashmap::init(hash_tbl, visited_hash_bitlen, 0);
  __syncthreads();

  // Stream through candidates; keep-first via shared hash
  for (uint32_t i = threadIdx.x; i < candidates_per_query; i += blockDim.x) {
    const uint32_t cta_id = i / itopk_size;
    const uint32_t local_idx = i % itopk_size;
    const size_t src_row = static_cast<size_t>(query_id) * num_cta_per_query + cta_id;
    const size_t src_offset = src_row * result_buffer_allocation_size + local_idx;

    const INDEX_T idx = result_indices_ptr[src_offset];
    const DISTANCE_T dst = result_distances_ptr[src_offset];

    // write index unconditionally to keep fixed per-query length
    gathered_indices_ptr[dst_base + i] = idx;

    if (idx == kInvalid) {
      gathered_distances_ptr[dst_base + i] = kInf;
      continue;
    }

    bool inserted = hashmap::insert<INDEX_T>(hash_tbl, visited_hash_bitlen, idx);
    gathered_distances_ptr[dst_base + i] = inserted ? dst : kInf;
  }
}

template <typename INDEX_T, typename DISTANCE_T>
void gather_cta_results(const INDEX_T* result_indices_ptr,
                       const DISTANCE_T* result_distances_ptr,
                       INDEX_T* gathered_indices_ptr,
                       DISTANCE_T* gathered_distances_ptr,
                       const uint32_t itopk_size,
                       const uint32_t num_cta_per_query,
                       const uint32_t result_buffer_allocation_size,
                       const uint32_t num_queries,
                       cudaStream_t stream)
{
  const uint32_t candidates_per_query = num_cta_per_query * itopk_size;
  // Choose small-hash bitlen so that table size >= 2x candidates (fill rate <= 0.5)
  uint32_t bitlen = 8;
  while (hashmap::get_size(bitlen) < (candidates_per_query << 1)) { ++bitlen; }
  const size_t shared_bytes = hashmap::get_size(bitlen) * sizeof(INDEX_T);
  // 常见 multi-CTA 配置（itopk=32, cta=8 → Nc=256）下共享内存充足
  constexpr size_t kSharedLimit = 48 * 1024;
  RAFT_EXPECTS(shared_bytes <= kSharedLimit,
               "gather_cta_results: candidates exceed shared memory threshold");

  const dim3 grid(num_queries);
  const dim3 block(128);
  gather_cta_results_kernel<<<grid, block, shared_bytes, stream>>>(
    result_indices_ptr,
    result_distances_ptr,
    gathered_indices_ptr,
    gathered_distances_ptr,
    itopk_size,
    num_cta_per_query,
    result_buffer_allocation_size,
    /*visited_hash_bitlen*/ bitlen,
    /*total_elements_unused*/ 0);
}

// remove_parent_bit and its kernel are provided by search_utils.hpp

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SAMPLE_FILTER_T>
struct search : public search_plan_impl<DataT, IndexT, DistanceT, SAMPLE_FILTER_T> {
  using base_type  = search_plan_impl<DataT, IndexT, DistanceT, SAMPLE_FILTER_T>;
  using DATA_T     = typename base_type::DATA_T;
  using INDEX_T    = typename base_type::INDEX_T;
  using DISTANCE_T = typename base_type::DISTANCE_T;

  // Inherit all base class members
  using base_type::algo;
  using base_type::dim;
  using base_type::graph_degree;
  using base_type::dataset_size;
  using base_type::topk;
  using base_type::itopk_size;
  using base_type::search_width;
  using base_type::max_queries;
  using base_type::max_iterations;
  using base_type::min_iterations;
  using base_type::thread_block_size;
  using base_type::team_size;
  using base_type::hashmap_max_fill_rate;
  using base_type::hashmap_min_bitlen;
  using base_type::hashmap_mode;
  using base_type::num_random_samplings;
  using base_type::rand_xor_mask;
  
  using base_type::hash_bitlen;
  using base_type::small_hash_bitlen;
  using base_type::small_hash_reset_interval;
  using base_type::hashmap_size;
  using base_type::result_buffer_size;
  using base_type::smem_size;
  
  using base_type::dataset_desc;
  using base_type::dev_seed;
  using base_type::hashmap;
  using base_type::num_executed_iterations;
  using base_type::num_seeds;
  using base_type::metric;

  // Multi-CTA specific members
  uint32_t num_cta_per_query;
  size_t result_buffer_allocation_size;  // Double buffer size per CTA
  lightweight_uvector<INDEX_T> result_indices;      // Multi-CTA result buffer
  lightweight_uvector<DISTANCE_T> result_distances; // Multi-CTA result buffer
  lightweight_uvector<INDEX_T> parent_node_list;  
  lightweight_uvector<INDEX_T> intermediate_indices;
  lightweight_uvector<DISTANCE_T> intermediate_distances;
  lightweight_uvector<std::uint32_t> topk_workspace;  // Workspace for topk operations
  lightweight_uvector<uint32_t> terminate_flag;

  // Additional buffers for CPU-GPU co-processing
  lightweight_uvector<INDEX_T> d_result_indices;    // Device copy for miss handling
  lightweight_uvector<INDEX_T> miss_positions;      // Track data miss positions
  // used for compute_distance_to_child_nodes_kernels
  lightweight_uvector<uint8_t> compute_distance_flags;
  lightweight_uvector<unsigned int> graph_miss_counter;
  lightweight_uvector<IndexT> miss_host_graphids;
  lightweight_uvector<IndexT> miss_device_graphids;
  lightweight_uvector<unsigned int> miss_counter1_vec;
  lightweight_uvector<unsigned int> miss_counter1;
  lightweight_uvector<unsigned int> miss_counter2;
  lightweight_uvector<unsigned int> miss_counter2_write;
  lightweight_uvector<IndexT> miss_host_indices1;
  lightweight_uvector<IndexT> result_idx_offsets;
  lightweight_uvector<IndexT> miss_host_indices2;
  lightweight_uvector<IndexT> miss_device_indices;
  lightweight_uvector<DistanceT> tmp_device_distances;

  // Visited hashmap for per-iteration CTA deduplication
  lightweight_uvector<INDEX_T> visited_hashmap;     // Per-iteration dedup between CTAs
  
  // Constructor
  search(raft::resources const& res,
         search_params params,
         const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
         int64_t dim,
         int64_t dataset_size,
         int64_t graph_degree,
         uint32_t topk)
    : base_type(res, params, dataset_desc, dim, dataset_size, graph_degree, topk),
      result_indices(res), result_distances(res),
      intermediate_indices(res), intermediate_distances(res),
      parent_node_list(res), d_result_indices(res),
      miss_positions(res), topk_workspace(res),
      terminate_flag(res), visited_hashmap(res),
      compute_distance_flags(res), graph_miss_counter(res), miss_host_graphids(res), miss_device_graphids(res),
      miss_counter1_vec(res), miss_counter1(res), miss_counter2(res), miss_counter2_write(res),
      miss_host_indices1(res), result_idx_offsets(res), miss_host_indices2(res), miss_device_indices(res), 
      tmp_device_distances(res)
  {
    set_params(res, params);
  }

  void set_params(raft::resources const& res, const search_params& params)
  {
    size_t global_itopk_size                = itopk_size;
    constexpr unsigned multi_cta_itopk_size = 32;
    this->itopk_size = multi_cta_itopk_size;
    search_width = 1;
    RAFT_LOG_DEBUG("params.itopk_size: %lu", (uint64_t)params.itopk_size);
    RAFT_LOG_DEBUG("global_itopk_size: %lu", (uint64_t)global_itopk_size);
    num_cta_per_query = std::max(params.search_width, 
                                 raft::ceildiv(global_itopk_size, (size_t)multi_cta_itopk_size));
    
    // Each CTA needs: itopk_size + (search_width * graph_degree)
    result_buffer_size = itopk_size + (search_width * graph_degree);
    // Double buffer layout for CPU-GPU co-processing:
    result_buffer_allocation_size = result_buffer_size + itopk_size;
    result_indices.resize(result_buffer_allocation_size * num_cta_per_query * max_queries,
                          raft::resource::get_cuda_stream(res));
    result_distances.resize(result_buffer_allocation_size * num_cta_per_query * max_queries,
                            raft::resource::get_cuda_stream(res));
    parent_node_list.resize(max_queries * num_cta_per_query * search_width, raft::resource::get_cuda_stream(res));
    
    // RAFT_EXPECTS(result_buffer_size <= 256, "Result buffer size per CTA cannot exceed 256");
    smem_size = dataset_desc.smem_ws_size_in_bytes +
                (sizeof(INDEX_T) + sizeof(DISTANCE_T)) * result_buffer_size +
                sizeof(INDEX_T) * hashmap::get_size(small_hash_bitlen) +
                sizeof(INDEX_T) * search_width +
                sizeof(int);
    
    RAFT_LOG_DEBUG("# smem_size: %u", smem_size);
    
    // Allocate intermediate buffers for merging results from multiple CTAs
    uint32_t num_intermediate_results = num_cta_per_query * itopk_size;
    intermediate_indices.resize(num_intermediate_results * max_queries,
                               raft::resource::get_cuda_stream(res));
    intermediate_distances.resize(num_intermediate_results * max_queries,
                                 raft::resource::get_cuda_stream(res));
    
    // Allocate hashmap
    hashmap.resize(max_queries * hashmap::get_size(hash_bitlen), raft::resource::get_cuda_stream(res));
    // Per-CTA visited hashmap: allocate a table per CTA (not per query)
    visited_hashmap.resize(max_queries * num_cta_per_query * hashmap::get_size(small_hash_bitlen),
                           raft::resource::get_cuda_stream(res));
    // Allocate topk workspace
    size_t topk_workspace_size = _cuann_find_topk_bufferSize(
      itopk_size, max_queries * num_cta_per_query, result_buffer_size, 
      utils::get_cuda_data_type<DATA_T>());
    topk_workspace.resize(topk_workspace_size, raft::resource::get_cuda_stream(res));
    terminate_flag.resize(1, raft::resource::get_cuda_stream(res));

  // Allocate additional buffers for CPU-GPU co-processing
    cudaStream_t cuda_stream      = raft::resource::get_cuda_stream(res);
    const size_t total_result_buffer_size = num_cta_per_query * max_queries * result_buffer_allocation_size;
    d_result_indices.resize(total_result_buffer_size, cuda_stream);
    miss_positions.resize(total_result_buffer_size, cuda_stream);
  }

  void check(const uint32_t topk) override
  {
    RAFT_EXPECTS(num_cta_per_query * 32 >= topk,
                 "`num_cta_per_query` (%u) * 32 must be >= `topk` (%u)",
                 num_cta_per_query, topk);
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
                         bool sort)
  {
    auto stream = raft::resource::get_cuda_stream(handle);
    assert (topK <= 1024);
    _cuann_find_topk(topK,
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
                      nullptr,
                      stream);
  }

  void operator()(raft::resources const& res,
                  raft::host_matrix_view<const INDEX_T, int64_t> graph,
                  raft::device_matrix_view<INDEX_T, int64_t, raft::row_major> d_graph,
                  INDEX_T* const topk_indices_ptr,
                  DISTANCE_T* const topk_distances_ptr,
                  const DATA_T* const queries_ptr,
                  const DATA_T* const host_queries_ptr,
                  const uint32_t num_queries,
                  const INDEX_T* dev_seed_ptr,
                  uint32_t* const num_executed_iterations,
                  uint32_t topk,
                  SAMPLE_FILTER_T sample_filter,
                  host_device_mapper* host_hd_mapper,
                  graph_hd_mapper* host_graph_mapper,
                  int* in_edges,
                  float* miss_rate,
                  ffanns::neighbors::cagra::search_context<DATA_T, INDEX_T>* search_ctx = nullptr) override
  {
    // RAFT_LOG_INFO("[search_multi_cta] itopk_size = %u, num_cta_per_query = %u", itopk_size, num_cta_per_query);
    TimingAccumulator accumulator;
    cudaStream_t stream = raft::resource::get_cuda_stream(res);

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, stream);

    // Define maximum CTA per query for static allocation
    constexpr uint32_t MAX_CTA_PER_QUERY = 8;
    RAFT_EXPECTS(num_cta_per_query <= MAX_CTA_PER_QUERY,
                 "num_cta_per_query (%u) exceeds MAX_CTA_PER_QUERY (%u)",
                 num_cta_per_query, MAX_CTA_PER_QUERY);

    IndexT* traversed_hashmap_ptr = hashmap.data();
    IndexT* visited_hashmap_ptr = visited_hashmap.data();
    const uint32_t traversed_hash_size = hashmap::get_size(hash_bitlen);
    const uint32_t visited_hash_size = hashmap::get_size(small_hash_bitlen);
    set_value_batch(traversed_hashmap_ptr, traversed_hash_size, utils::get_max_value<INDEX_T>(),
                    traversed_hash_size, num_queries, stream);
    // Reset per-CTA visited hashmaps (num_queries * num_cta_per_query tables)
    set_value_batch(visited_hashmap_ptr, visited_hash_size, utils::get_max_value<INDEX_T>(),
                    visited_hash_size, num_queries * num_cta_per_query, stream);
    
    const auto total_ctas = num_queries * num_cta_per_query;
    compute_distance_flags.resize(total_ctas * result_buffer_allocation_size);
    graph_miss_counter.resize(1);
    miss_host_graphids.resize(total_ctas * search_width);
    miss_device_graphids.resize(total_ctas * search_width);
    miss_counter1_vec.resize(total_ctas);
    miss_counter1.resize(1);
    miss_counter2.resize(1);
    miss_counter2_write.resize(1);
    miss_host_indices1.resize(total_ctas * search_width * graph_degree);
    result_idx_offsets.resize(total_ctas * search_width * graph_degree);
    miss_host_indices2.resize(total_ctas * search_width * graph_degree);
    miss_device_indices.resize(total_ctas * search_width * graph_degree);
    tmp_device_distances.resize(total_ctas * search_width * graph_degree);
    set_value<IndexT>(result_idx_offsets.data(), utils::get_max_value<IndexT>(), total_ctas * search_width * graph_degree, stream);
    
    // Initialize iteration counter
    unsigned iter = 0;
    // Disable per-call pinned allocation defaults (prefer external search_ctx when provided)
    const unsigned int MAX_QUERIES = 0;   // legacy default disabled
    const unsigned int MAX_TRANS_NUM = 0; // legacy default disabled

    std::array<size_t, 4> hd_status = {0, 0, 0, 0};
    auto hd_mapper = host_hd_mapper->dev_ptr(res);
    auto graph_mapper = host_graph_mapper->dev_ptr(res);

    random_pickup(dataset_desc,
                  queries_ptr,
                  num_queries,
                  result_buffer_size,
                  num_cta_per_query,
                  num_random_samplings,
                  rand_xor_mask,
                  dev_seed_ptr,
                  num_seeds,
                  result_indices.data(),
                  result_distances.data(),
                  result_buffer_allocation_size,
                  visited_hashmap.data(),
                  small_hash_bitlen,
                  sample_filter,
                  hd_mapper,
                  stream);
    
    // Always create per-call pinned buffers (baseline), but prefer external context if provided
    const size_t host_elems = static_cast<size_t>(MAX_QUERIES) *
                              static_cast<size_t>(num_cta_per_query) *
                              static_cast<size_t>(search_width) *
                              static_cast<size_t>(graph_degree);
    auto cpu_distances      = raft::make_pinned_vector<DistanceT>(res, host_elems);
    auto host_indices       = raft::make_pinned_vector<IndexT>(res, host_elems);
    auto query_miss_counter = raft::make_pinned_vector<unsigned int>(res, static_cast<size_t>(MAX_QUERIES) * static_cast<size_t>(num_cta_per_query));
    auto host_graph_buffer  = raft::make_pinned_vector<IndexT>(res, host_elems);
    auto host_vector_buffer = raft::make_pinned_matrix<DataT, int64_t, raft::row_major>(res,  MAX_TRANS_NUM, dataset_desc.stride);
    auto is_internal_search = true;
    std::optional<raft::pinned_scalar<unsigned int>> num_graph_miss_owner, num_miss1_owner, num_miss2_owner;

    bool use_search_ctx = (search_ctx && search_ctx->has_buffers());
    DistanceT*    cpu_distances_ptr      = use_search_ctx ? reinterpret_cast<DistanceT*>(search_ctx->cpu_distances)   : cpu_distances.data_handle();
    IndexT*       host_indices_ptr       = use_search_ctx ? reinterpret_cast<IndexT*>(search_ctx->host_indices)       : host_indices.data_handle();
    unsigned int* query_miss_counter_ptr = use_search_ctx ? search_ctx->query_miss_counter                             : query_miss_counter.data_handle();
    IndexT*       host_graph_buffer_ptr  = use_search_ctx ? reinterpret_cast<IndexT*>(search_ctx->host_graph_buffer)   : host_graph_buffer.data_handle();
    DataT*        host_vector_buffer_ptr = use_search_ctx ? reinterpret_cast<DataT*>(search_ctx->host_vector_buffer)   : host_vector_buffer.data_handle();
    if (!use_search_ctx) {
      num_graph_miss_owner.emplace(raft::make_pinned_scalar<unsigned int>(res, 0));
      num_miss1_owner.emplace(raft::make_pinned_scalar<unsigned int>(res, 0));
      num_miss2_owner.emplace(raft::make_pinned_scalar<unsigned int>(res, 0));
    }
    unsigned int* num_graph_miss_ptr = use_search_ctx ? search_ctx->num_graph_miss : num_graph_miss_owner->data_handle();
    unsigned int* num_miss1_ptr = use_search_ctx ? search_ctx->num_miss1 : num_miss1_owner->data_handle();
    unsigned int* num_miss2_ptr = use_search_ctx ? search_ctx->num_miss2 : num_miss2_owner->data_handle();

    ComputeDistanceContext<DataT, IndexT, DistanceT> compute_ctx(
      hd_mapper, graph_mapper, hd_status.data(), in_edges,
      cpu_distances_ptr, host_indices_ptr,
      query_miss_counter_ptr, host_graph_buffer_ptr,
      host_vector_buffer_ptr, num_graph_miss_ptr, num_miss1_ptr, num_miss2_ptr,
      0, metric, is_internal_search,
      compute_distance_flags.data(), graph_miss_counter.data(), 
      miss_host_graphids.data(), miss_device_graphids.data(),
      miss_counter1_vec.data(), miss_counter1.data(), 
      miss_counter2.data(), miss_counter2_write.data(),
      miss_host_indices1.data(), result_idx_offsets.data(), 
      miss_host_indices2.data(), miss_device_indices.data(), tmp_device_distances.data());

    // // Iteration-stage timing (per-iteration breakdown)
    // float acc_find_ms = 0.f, acc_pickup_ms = 0.f, acc_compute_ms = 0.f;
    // int   acc_iters = 0;
    // cudaEvent_t ev_it_start, ev_it_find, ev_it_pickup, ev_it_compute;
    // cudaEventCreate(&ev_it_start);
    // cudaEventCreate(&ev_it_find);
    // cudaEventCreate(&ev_it_pickup);
    // cudaEventCreate(&ev_it_compute);

    // Main search loop
    while (1) {
      // cudaEventRecord(ev_it_start, stream);
      _find_topk(res,
        itopk_size,
        num_queries * num_cta_per_query,
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
        true);
      // cudaEventRecord(ev_it_find, stream);

      if ((iter + 1) == max_iterations) {
        iter++;
        break;
      }

      if (iter + 1 >= min_iterations) { set_value<uint32_t>(terminate_flag.data(), 1, stream); }
      
      // Multi-CTA parent selection
      // Memory layout:
      // - parent_candidates: [num_queries * num_cta_per_query, result_buffer_allocation_size]
      // - parent_list: [num_queries * num_cta_per_query, search_width]
      pickup_next_parents(result_indices.data(),
                          result_distances.data(),
                          result_buffer_allocation_size,  // Stride per CTA (lds)
                          itopk_size,                      // Candidates per CTA
                          num_queries,
                          hashmap.data(),                  // traversed_hashmap for coordination
                          hash_bitlen,
                          parent_node_list.data(),         // Parent list [num_queries * num_cta_per_query, search_width]
                          search_width,                    // ldd: stride per global CTA
                          search_width,                    // Parents to select per CTA
                          terminate_flag.data(),
                          num_cta_per_query,               // Multi-CTA parameter
                          visited_hashmap.data(),          // visited hashmap (per-CTA)
                          small_hash_bitlen,               // visited hash bitlen
                          (iter & 0x1) ? result_buffer_size : 0,  // old-half offset (absolute)
                          (1 - (iter & 0x1)) ? result_buffer_size : 0, // new-half offset (absolute)
                          graph_miss_counter.data(), miss_counter1_vec.data(), miss_counter1.data(), 
                          miss_counter2.data(), miss_counter2_write.data(),
                          stream);
      // cudaEventRecord(ev_it_pickup, stream);

      if (iter + 1 >= min_iterations && get_value(terminate_flag.data(), stream)) {
        iter++;
        break;
      }

      compute_ctx.iter = iter;
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
        num_cta_per_query,
        visited_hashmap.data(),
        small_hash_bitlen,
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

      // cudaEventRecord(ev_it_compute, stream);
      // // Synchronize to read per-iteration timings (profiling-only; introduces measurement overhead)
      // cudaEventSynchronize(ev_it_compute);
      // float t_find=0.f, t_pick=0.f, t_comp=0.f;
      // cudaEventElapsedTime(&t_find,  ev_it_start,  ev_it_find);
      // cudaEventElapsedTime(&t_pick,  ev_it_find,   ev_it_pickup);
      // cudaEventElapsedTime(&t_comp,  ev_it_pickup,ev_it_compute);
      // acc_find_ms   += t_find;
      // acc_pickup_ms += t_pick;
      // acc_compute_ms+= t_comp;
      // acc_iters++;

     iter++;
    }

    // RAFT_LOG_INFO("[search_multi_cta] Total iterations: %d, MAX_ITERATIONS: %d", iter, max_iterations);
    // accumulator.print2();
    
    auto result_indices_ptr   = result_indices.data() + (iter & 0x1) * result_buffer_size;
    auto result_distances_ptr = result_distances.data() + (iter & 0x1) * result_buffer_size;

    // if constexpr (!std::is_same<SAMPLE_FILTER_T,
    //   ffanns::neighbors::filtering::none_sample_filter>::value) {
    //     // TODO: suppose no filter
    // } else {
    remove_parent_bit(
      num_queries * num_cta_per_query, itopk_size, result_indices_ptr, result_buffer_allocation_size, stream);
    // }
    
    // Final aggregation: gather scattered CTA results and select final topk
    const size_t candidates_per_query = num_cta_per_query * itopk_size;
    const size_t total_elements = num_queries * candidates_per_query;
    
    // Allocate contiguous buffer for gathered results
    lightweight_uvector<INDEX_T> gathered_indices(res, total_elements);
    lightweight_uvector<DISTANCE_T> gathered_distances(res, total_elements);
    
    // Gather scattered CTA results into contiguous buffer
    gather_cta_results(result_indices_ptr,
                      result_distances_ptr,
                      gathered_indices.data(),
                      gathered_distances.data(),
                      itopk_size,
                      num_cta_per_query,
                      result_buffer_allocation_size,
                      num_queries,
                      stream);
    
    // Allocate workspace for final topk selection
    size_t final_topk_workspace_size = _cuann_find_topk_bufferSize(
      topk, max_queries, candidates_per_query, 
      utils::get_cuda_data_type<DATA_T>());
    lightweight_uvector<std::uint8_t> final_topk_workspace(res, final_topk_workspace_size);
    
    // Final topk selection from gathered candidates using the same _find_topk pattern
    _find_topk(res,
              topk,                              // output size per query
              num_queries,                       // number of queries
              candidates_per_query,              // input size per query
              gathered_distances.data(),         // input distances
              candidates_per_query,              // input distances stride
              gathered_indices.data(),           // input indices  
              candidates_per_query,              // input indices stride
              topk_distances_ptr,                // output distances
              topk,                              // output distances stride
              topk_indices_ptr,                  // output indices
              topk,                              // output indices stride
              final_topk_workspace.data(),      // workspace
              true);                             // sort results

    if (num_executed_iterations) {
      for (std::uint32_t i = 0; i < num_queries; i++) {
        num_executed_iterations[i] = iter;
      }
    }
    host_device_mapper::free_dev_ptr(hd_mapper, raft::resource::get_cuda_stream(res));
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // cudaEventRecord(stop, raft::resource::get_cuda_stream(res));
    // cudaEventSynchronize(stop);
    // float total_time;
    // cudaEventElapsedTime(&total_time, start, stop);
    // // Per-iteration breakdown averages
    // if (acc_iters > 0) {
    //   const float avg_find   = acc_find_ms   / acc_iters;
    //   const float avg_pickup = acc_pickup_ms / acc_iters;
    //   const float avg_compute= acc_compute_ms/ acc_iters;
    //   const float stage_sum  = avg_find + avg_pickup + avg_compute;
    //   RAFT_LOG_INFO("[search_multi_cta] [total-time(ms)] %f, stage_sum = %.3f, find=%.3f, pickup=%.3f, compute=%.3f", total_time, stage_sum * acc_iters, acc_find_ms, acc_pickup_ms, acc_compute_ms);
    //   // RAFT_LOG_INFO("[search_multi_cta][iter-avg(ms)] find=%.3f, pickup=%.3f, hash=%.3f, compute=%.3f",
    //   //               avg_find, avg_pickup, avg_hash, avg_compute);
    //   if (stage_sum > 0.f) {
    //     RAFT_LOG_INFO("[search_multi_cta][iter-share(%%)] find=%.1f, pickup=%.1f, compute=%.1f",
    //                   100.f*avg_find/stage_sum, 100.f*avg_pickup/stage_sum, 100.f*avg_compute/stage_sum);
    //   }
    // }

    // cudaEventDestroy(ev_it_start);
    // cudaEventDestroy(ev_it_find);
    // cudaEventDestroy(ev_it_pickup);
    // cudaEventDestroy(ev_it_compute);
    // RAFT_LOG_INFO("[search_multi_cta] Total time taken: %f ms", total_time);
  }
};

}  // namespace multi_cta_search
}  // namespace ffanns::neighbors::cagra::detail
