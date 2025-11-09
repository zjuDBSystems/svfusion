#include "../ann_utils.cuh"
#include "ffanns/neighbors/cagra.hpp"
#include <raft/core/device_resources.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/spatial/knn/detail/ann_utils.cuh>
#include <raft/stats/histogram.cuh>
#include <raft/core/pinned_mdarray.hpp>

#include <rmm/device_buffer.hpp>

#include <omp.h>

#include <cstdint>

namespace ffanns::neighbors::cagra {

static const std::string RAFT_NAME = "raft";

template <class IdxT>
inline RAFT_KERNEL update_d_graph_kernel(IdxT* __restrict__ device_graph,
  IdxT* __restrict__ update_buffer,
  const uint32_t* __restrict__ positions,
  const size_t row_count,
  const size_t degree)
{
  auto row_idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (row_idx < row_count) {
    uint32_t target_row = positions[row_idx];
    if (target_row != UINT32_MAX) {
      auto update_buffer_ptr = update_buffer + (size_t)row_idx * degree;
      auto device_graph_ptr = device_graph + (size_t)target_row * degree;
      for (size_t col_idx = 0; col_idx < degree; col_idx += 1) {
        device_graph_ptr[col_idx] = update_buffer_ptr[col_idx];
      }
    }
  }
}

template <class T, class IdxT>
void add_node_core(
  raft::resources const& handle,
  ffanns::neighbors::cagra::index<T, IdxT>& idx,
  raft::host_matrix_view<const T, int64_t, raft::layout_stride> additional_dataset_view, 
  // raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::layout_stride, Accessor>
  //   additional_dataset_view,
  raft::host_matrix_view<IdxT, std::int64_t> updated_graph,
  raft::device_matrix_view<IdxT, std::int64_t> updated_d_graph,
  raft::host_vector_view<int, std::int64_t> host_num_incoming_edges,
  std::uint32_t chunk_id,
  const std::size_t insert_position,
  const cagra::extend_params& extend_params,
  ffanns::neighbors::cagra::search_context<T, IdxT>* search_ctx = nullptr)
{
  // cudaEvent_t start, stop, tag1, search_done, step2_done, step3_done;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop); 
  // cudaEventCreate(&tag1);
  // cudaEventCreate(&search_done);
  // cudaEventCreate(&step2_done);
  // cudaEventCreate(&step3_done);
  // cudaEventRecord(start, raft::resource::get_cuda_stream(handle));
  // auto delete_bitset_ptr = idx.get_delete_bitset_ptr();
  // auto count_scalar = raft::make_device_scalar<int64_t>(handle, 0);
  // delete_bitset_ptr->count(handle, count_scalar.view());
  // int64_t host_count = 0;
  // raft::copy(&host_count, count_scalar.data_handle(), 1, raft::resource::get_cuda_stream(handle));
  // raft::resource::sync_stream(handle);
  // RAFT_LOG_INFO("[add_graph_nodes] delete bitset count = %ld", delete_bitset_ptr->size() - host_count);

  using DistanceT                 = float;
  const std::size_t degree        = idx.graph_degree();
  const std::size_t dim           = idx.dim();
  const std::size_t old_size      = idx.dataset().extent(0);
  const std::size_t num_add       = additional_dataset_view.extent(0);
  const std::size_t new_size      = (insert_position != old_size) ? old_size : (old_size + num_add);
  const std::uint32_t base_degree = degree * 2;
  auto hd_mapper_ptr = idx.hd_mapper_ptr();
  auto edge_log = idx.get_edge_log_ptr();

  #ifdef FFANNS_TIME_LOG
  static std::ofstream time_log;
  if (chunk_id == 0) {
    time_log.open(bench_config::instance().get_insert_log_path());
    time_log << "chunk_id,search_time_ms,total_time_ms\n";
  }
  #endif
  // RAFT_LOG_INFO("New size: %ld", new_size);
  // auto host_num_incoming_edges = raft::make_host_vector<int, std::uint64_t>(new_size);
  // RAFT_LOG_INFO("[host_num_incoming_edges] [first element] = %d", *(static_cast<const int*>(host_num_incoming_edges.data_handle())));

  // const std::size_t max_chunk_size = 10000;
  const std::size_t max_chunk_size = extend_params.max_chunk_size == 0 ? 10000 : extend_params.max_chunk_size;
  ffanns::neighbors::cagra::search_params params;
  // params.algo = ffanns::neighbors::cagra::search_algo::MULTI_KERNEL;
  params.max_iterations = extend_params.max_chunk_size == 0 ? 120 : 30;
  params.itopk_size = std::max(base_degree * 2lu, 256lu);
  params.metric = idx.metric();

  // Memory space for rank-based neighbor list
  auto mr = raft::resource::get_workspace_resource(handle);

  auto neighbor_indices = raft::make_device_mdarray<IdxT, std::int64_t>(
    handle, mr, raft::make_extents<std::int64_t>(max_chunk_size, base_degree));

  auto neighbor_distances = raft::make_device_mdarray<DistanceT, std::int64_t>(
    handle, mr, raft::make_extents<std::int64_t>(max_chunk_size, base_degree));

  auto queries = raft::make_device_mdarray<T, std::int64_t>(
    handle, mr, raft::make_extents<std::int64_t>(max_chunk_size, dim));
  std::vector<T> host_queries(max_chunk_size * dim);

  auto host_neighbor_indices =
    raft::make_host_matrix<IdxT, std::int64_t>(max_chunk_size, base_degree);
  // TODO: Check additional dataset on host or device
  ffanns::spatial::knn::detail::utils::batch_load_iterator<T> additional_dataset_batch(
    additional_dataset_view.data_handle(),
    num_add,
    additional_dataset_view.stride(0),
    max_chunk_size,
    raft::resource::get_cuda_stream(handle),
    mr);
  
  for (const auto& batch : additional_dataset_batch) {
    // Step 1: Obtain K (=base_degree) nearest neighbors of the new vectors by CAGRA search
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(queries.data_handle(),
                                    sizeof(T) * dim,
                                    batch.data(),
                                    sizeof(T) * additional_dataset_view.stride(0),
                                    sizeof(T) * dim,
                                    batch.size(),
                                    cudaMemcpyDefault,
                                    raft::resource::get_cuda_stream(handle)));
    
    const auto queries_view = raft::make_device_matrix_view<const T, std::int64_t>(
      queries.data_handle(), batch.size(), dim);
    
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(host_queries.data(),
      sizeof(T) * dim,
      batch.data(),
      sizeof(T) * dim,
      sizeof(T) * dim,
      batch.size(),
      cudaMemcpyDefault,
      raft::resource::get_cuda_stream(handle)));
    const auto host_queries_view = raft::make_host_matrix_view<const T, std::int64_t>(
      host_queries.data(), batch.size(), dim);

    auto neighbor_indices_view = raft::make_device_matrix_view<IdxT, std::int64_t>(
      neighbor_indices.data_handle(), batch.size(), base_degree);
    auto neighbor_distances_view = raft::make_device_matrix_view<float, std::int64_t>(
      neighbor_distances.data_handle(), batch.size(), base_degree);
    
    auto delete_bitset_ptr = idx.get_delete_bitset_ptr();
    auto delete_filter = ffanns::neighbors::filtering::bitset_filter(delete_bitset_ptr->view());
    // cudaEventRecord(tag1, raft::resource::get_cuda_stream(handle));
    neighbors::cagra::search(
      handle, params, idx, queries_view, host_queries_view, neighbor_indices_view, neighbor_distances_view, delete_filter, false, search_ctx);
    
    raft::copy(host_neighbor_indices.data_handle(),
            neighbor_indices.data_handle(),
            batch.size() * base_degree,
            raft::resource::get_cuda_stream(handle));
    raft::resource::sync_stream(handle);
    // cudaEventRecord(search_done, raft::resource::get_cuda_stream(handle));
    // RAFT_LOG_INFO("[add_node_core] Search done");
    // for (std::size_t vec_i = 0; vec_i < batch.size(); vec_i++) {
    //   for (std::uint32_t i = 0; i < base_degree; i++) {
    //     const auto a_id = host_neighbor_indices(vec_i, i);
    //     if (a_id >= new_size) {
    //       std::fprintf(stderr, "!!!!!Invalid neighbor index (%u)\n", a_id);
    //     }
    //   }
    // }

    auto host_delete_bitset_ptr = idx.get_host_delete_bitset_ptr();
    // Step 2: rank-based reordering
// #pragma omp parallel
//     {
#pragma omp parallel for schedule(dynamic)
    // 
    // for (std::size_t vec_i = omp_get_thread_num(); vec_i < batch.size();
    //       vec_i += omp_get_num_threads()) {
    for (std::size_t vec_i = 0; vec_i < batch.size(); vec_i++) {
      std::vector<std::pair<IdxT, std::size_t>> detourable_node_count_list(base_degree);
      // Count detourable edges
      for (std::uint32_t i = 0; i < base_degree; i++) {
        std::uint32_t detourable_node_count = 0;
        const auto a_id                     = host_neighbor_indices(vec_i, i);
        for (std::uint32_t j = 0; j < i; j++) {
          const auto b0_id = host_neighbor_indices(vec_i, j);
          assert(b0_id < idx.size());
          for (std::uint32_t k = 0; k < degree; k++) {
            const auto b1_id = updated_graph(b0_id, k);
            if (a_id == b1_id) {
              detourable_node_count++;
              break;
            }
          }
        }
        detourable_node_count_list[i] = std::make_pair(a_id, detourable_node_count);
      }
      std::sort(detourable_node_count_list.begin(),
                detourable_node_count_list.end(),
                [&](const std::pair<IdxT, std::size_t> a, const std::pair<IdxT, std::size_t> b) {
                  return a.second < b.second;
                });
      for (std::size_t i = 0; i < degree; i++) {
          updated_graph(insert_position + batch.offset() + vec_i, i) = detourable_node_count_list[i].first;
      }
    }
    // }
    // cudaEventRecord(step2_done, raft::resource::get_cuda_stream(handle));

    // Step 3: Add reverse edges
    std::unordered_set<uint32_t> updated_rows;
    std::vector<std::unordered_set<uint32_t>> thread_updated_rows(omp_get_max_threads());
    const std::uint32_t rev_edge_search_range = degree / 2;
    const std::uint32_t num_rev_edges         = degree / 2;
    
#pragma omp parallel 
    {
      int tid = omp_get_thread_num();
      auto& local_updated_rows = thread_updated_rows[tid];
      std::vector<IdxT> rev_edges(num_rev_edges), temp(degree);
      // for (std::size_t vec_i = 0; vec_i < batch.size(); vec_i++) {
      for (std::size_t vec_i = omp_get_thread_num(); vec_i < batch.size();
        vec_i += omp_get_num_threads()) {
        // Create a reverse edge list
        std::size_t valid_rev_cnt = 0; 
        const auto target_new_node_id = insert_position + batch.offset() + vec_i;
        for (std::size_t i = 0; i < num_rev_edges; i++) {
          const auto target_node_id = updated_graph(insert_position + batch.offset() + vec_i, i);
          if (target_node_id >= new_size) {
            RAFT_LOG_INFO("Invalid target_node_id (%u) at i=%d for target_new_node_id=%u", target_node_id, i, target_new_node_id);
          }
          IdxT replace_id                        = new_size;
          IdxT replace_id_j                      = 0;
          std::size_t replace_num_incoming_edges = 0;
          for (std::int32_t j = degree - 1; j >= static_cast<std::int32_t>(rev_edge_search_range);
              j--) {
            const auto neighbor_id               = updated_graph(target_node_id, j);
            if (neighbor_id >= new_size) {
              RAFT_LOG_INFO("Invalid neighbor_id (%u) at j=%d for target_node_id=%u", neighbor_id, j, target_node_id);
            }
            if (!host_delete_bitset_ptr->test(neighbor_id)) {
              continue;
            }

            std::size_t num_incoming_edges;
            // #pragma omp atomic read
            num_incoming_edges = host_num_incoming_edges(neighbor_id);
            if (num_incoming_edges > replace_num_incoming_edges) {
              // Check duplication
              bool dup = false;
              for (std::uint32_t k = 0; k < i; k++) {
                if (rev_edges[k] == neighbor_id) {
                  dup = true;
                  break;
                }
              }
              if (dup) { continue; }

              // Update rev edge candidate
              replace_num_incoming_edges = num_incoming_edges;
              replace_id                 = neighbor_id;
              replace_id_j               = j;
            }
          }
          if (replace_id >= new_size) {
            std::fprintf(stderr, "!!!!!Invalid rev edge index (%u)\n", replace_id);
            continue;
          }
          
          // 使用局部锁处理图更新 - 可以使用细粒度锁进一步优化
          // #pragma omp critical(graph_update)
          // {
          //     updated_graph(target_node_id, replace_id_j) = target_new_node_id;
          //     // edge_log->record_update(target_node_id, target_new_node_id);
          // }
          updated_graph(target_node_id, replace_id_j) = target_new_node_id;
          // edge_log->record_update(target_node_id, target_new_node_id);
          // #pragma omp atomic update 
          host_num_incoming_edges(target_new_node_id)++;

          // #pragma omp atomic update
          host_num_incoming_edges(replace_id)--; 
          // updated_rows.insert(target_node_id);
          local_updated_rows.insert(target_node_id);
          // rev_edges[i] = replace_id;
          rev_edges[valid_rev_cnt++] = replace_id; 
        }

        // Create a neighbor list of a new node by interleaving the kNN neighbor list and reverse edge
        // list
        std::uint32_t interleave_switch = 0, rank_base_i = 0, rev_edges_return_i = 0, num_add = 0;
        const auto rank_based_list_ptr =
          updated_graph.data_handle() + (insert_position + batch.offset() + vec_i) * degree;
        const auto rev_edges_return_list_ptr = rev_edges.data();
        while ((num_add < degree) &&
        ((rank_base_i < degree) || (rev_edges_return_i < num_rev_edges))) {
          const auto node_list_ptr =
            interleave_switch == 0 ? rank_based_list_ptr : rev_edges_return_list_ptr;
          auto& node_list_index          = interleave_switch == 0 ? rank_base_i : rev_edges_return_i;
          // const auto max_node_list_index = interleave_switch == 0 ? degree : num_rev_edges;
          const std::uint32_t max_node_list_index = interleave_switch == 0 ? degree : valid_rev_cnt;
          for (; node_list_index < max_node_list_index; node_list_index++) {
            const auto candidate = node_list_ptr[node_list_index];
            // Check duplication
            bool dup = false;
            for (std::uint32_t j = 0; j < num_add; j++) {
              if (temp[j] == candidate) {
                dup = true;
                break;
              }
            }
            if (!dup) {
              temp[num_add] = candidate;
              // if (interleave_switch == 0) {
              //   host_num_incoming_edges(candidate)++;
              // } 
              num_add++;
              break;
            }
          }
          interleave_switch = 1 - interleave_switch;
        }
        if (num_add < degree) {
          RAFT_FAIL("Number of edges is not enough (target_new_node_id:%lu, num_add:%lu, degree:%lu)",
                    (uint64_t)target_new_node_id, (uint64_t)num_add, (uint64_t)degree);
        }
        for (std::uint32_t i = 0; i < degree; i++) {
          const auto new_neighbor = temp[i];
          if (!host_delete_bitset_ptr->test(new_neighbor)) {
            RAFT_LOG_INFO("[add_node_core] Invalid new_neighbor (%u) at i=%d for target_new_node_id=%u", new_neighbor, i, target_new_node_id);
          }
          updated_graph(target_new_node_id, i) = new_neighbor;
          // #pragma omp atomic update
          host_num_incoming_edges(new_neighbor)++;
        }
      }
    // updated_rows.insert(local_updated_rows.begin(), local_updated_rows.end());
  }
    // cudaEventRecord(step3_done, raft::resource::get_cuda_stream(handle));
    for (const auto& local_set : thread_updated_rows) {
      updated_rows.insert(local_set.begin(), local_set.end());
    }

    std::vector<uint32_t> sorted_rows(updated_rows.begin(), updated_rows.end());
    size_t updated_size = sorted_rows.size();

    static auto host_buffer = raft::make_pinned_matrix<IdxT, int64_t, raft::row_major>(handle, batch.size() * num_rev_edges, degree);
    static auto h_device_positions = raft::make_pinned_vector<IdxT>(handle, batch.size() * num_rev_edges);
    auto& graph_mapper = idx.get_graph_hd_mapper();
    graph_mapper.map_host_to_device_rows(handle, sorted_rows, h_device_positions.data_handle());
// #pragma omp parallel for
    for (size_t i = 0; i < updated_size; i++) {
        uint32_t host_row = sorted_rows[i];
        uint32_t device_row = h_device_positions.data_handle()[i];
        if (device_row != UINT32_MAX) {
            memcpy(host_buffer.data_handle() + i * degree,
                  updated_graph.data_handle() + host_row * degree,
                  degree * sizeof(IdxT));
        }
    }

    auto device_buffer = raft::make_device_matrix<IdxT, int64_t>(handle, updated_size, degree);
    rmm::device_uvector<IdxT> device_positions(updated_size, raft::resource::get_cuda_stream(handle));
    raft::copy(device_buffer.data_handle(),
           host_buffer.data_handle(),
           updated_size * degree,
           raft::resource::get_cuda_stream(handle));
    raft::copy(device_positions.data(),
           h_device_positions.data_handle(),
           updated_size,
           raft::resource::get_cuda_stream(handle));
    int threadsPerBlock = 256;
    int blocks = (updated_size + threadsPerBlock - 1) / threadsPerBlock;
    update_d_graph_kernel<<<blocks, threadsPerBlock, 0, raft::resource::get_cuda_stream(handle)>>>(
          updated_d_graph.data_handle(),
          device_buffer.data_handle(),
          device_positions.data(),
          updated_size,
          degree
      );

    // cudaEventRecord(stop, raft::resource::get_cuda_stream(handle));
    // cudaEventSynchronize(stop);
    // float tag1_time, search_time, step2_time, step3_time, update_time, total_time;
    // cudaEventElapsedTime(&tag1_time, start, tag1);
    // cudaEventElapsedTime(&search_time, tag1, search_done);
    // cudaEventElapsedTime(&step2_time, search_done, step2_done);
    // cudaEventElapsedTime(&step3_time, step2_done, step3_done);
    // cudaEventElapsedTime(&update_time, step3_done, stop);
    // cudaEventElapsedTime(&total_time, start, stop);
    // RAFT_LOG_INFO("[add_node_core] Tag1 time taken: %f ms", tag1_time);
    // RAFT_LOG_INFO("[add_node_core] Search time taken: %f ms", search_time);
    // RAFT_LOG_INFO("[add_node_core] Step 2 time taken: %f ms", step2_time);
    // RAFT_LOG_INFO("[add_node_core] Step 3 time taken: %f ms", step3_time);
    // RAFT_LOG_INFO("[add_node_core] Update time taken: %f ms", update_time);
    // RAFT_LOG_INFO("[add_node_core] Total time taken: %f ms", total_time);

    #ifdef FFANNS_TIME_LOG
    time_log << chunk_id << ","
             << search_time << ","
             << total_time << "\n";
    if (chunk_id % 20 == 0) {
      time_log.flush();
    }
    RAFT_LOG_INFO("Search time taken: %f ms", search_time);
    RAFT_LOG_INFO("Total time taken: %f ms", total_time);
    if (chunk_id % 1000 == 0) {
      std::string count_file_name = bench_config::instance().get_count_log_path(chunk_id);
      hd_mapper_ptr->snapshot_access_counts(count_file_name);
      // mapper.snapshot_access_counts(count_file_name);
    }
    #endif
  }
}

template <class T, class IdxT>
void add_graph_nodes(
  raft::resources const& handle,
  raft::host_matrix_view<const T, int64_t, raft::layout_stride> input_updated_dataset_view,
  raft::device_matrix_view<T, int64_t, raft::layout_stride> input_updated_d_dataset_view,
  neighbors::cagra::index<T, IdxT>& index,
  raft::host_matrix_view<IdxT, std::int64_t> updated_graph_view,
  raft::host_vector_view<int, std::int64_t> updated_in_edges_view,
  raft::device_matrix_view<IdxT, int64_t> updated_d_graph_view,
  const cagra::extend_params& params,
  const std::size_t insert_position,
  const std::size_t num_new_nodes,
  ffanns::neighbors::cagra::search_context<T, IdxT>* search_ctx = nullptr)
{
  auto delete_bitset_ptr = index.get_delete_bitset_ptr();
  auto count_scalar = raft::make_device_scalar<int64_t>(handle, 0);
  delete_bitset_ptr->count(handle, count_scalar.view());
  int64_t host_count = 0;
  raft::copy(&host_count, count_scalar.data_handle(), 1, raft::resource::get_cuda_stream(handle));
  raft::resource::sync_stream(handle);
  // RAFT_LOG_INFO("[add_graph_nodes] delete bitset count = %ld", delete_bitset_ptr->size() - host_count);

  assert(input_updated_dataset_view.extent(0) >= index.size());
  
  const std::size_t initial_dataset_size = index.size();
  const std::size_t new_dataset_size     = input_updated_dataset_view.extent(0);
  const std::size_t initial_d_dataset_size = index.d_dataset().extent(0);
  const std::size_t initial_d_graph_size = index.d_graph().extent(0);
  // const std::size_t num_new_nodes        = new_dataset_size - initial_dataset_size;
  const std::size_t degree               = index.graph_degree();
  const std::size_t dim                  = index.dim();
  const std::size_t stride               = input_updated_dataset_view.stride(0);
  const std::size_t max_chunk_size_      = params.max_chunk_size == 0 ? 10000 : params.max_chunk_size;
  // const std::size_t max_chunk_size_      = 10000;
  // TODO: In-place update for graph data
  // memcpy(updated_graph_view.data_handle(),
  //      index.graph().data_handle(),
  //      index.graph().size() * sizeof(IdxT));
  memset(updated_in_edges_view.data_handle() + insert_position,  
       0,                                                           
       num_new_nodes * sizeof(int));    
  
  neighbors::cagra::index<T, IdxT> internal_index(
    handle,
    index.metric(),
    raft::make_host_matrix_view<const T, int64_t>(nullptr, 0, dim),
    raft::make_host_matrix_view<const IdxT, int64_t>(nullptr, 0, degree),
    index.d_in_edges(),
    index.hd_mapper_ptr(),
    index.get_graph_hd_mapper_ptr(),
    index.get_delete_bitset_ptr(),
    index.get_host_delete_bitset_ptr(),
    index.get_edge_log_ptr(),
    index.get_delete_slots_ptr(),
    index.get_free_slots_ptr());

  // RAFT_LOG_INFO("Finish to load additional data");

  auto& mapper = internal_index.hd_mapper();
  auto& graph_mapper = internal_index.get_graph_hd_mapper();
  internal_index.update_d_graph(handle, index.d_graph());
  
  bool using_free_slots = insert_position != initial_dataset_size;
  assert (num_new_nodes != 0);
  // RAFT_LOG_INFO("[add_graph_nodes] num_new_nodes = %d", num_new_nodes);
  uint32_t chunk_id = 0;
  float sum_miss_rates = 0.0f;
  for (std::size_t additional_dataset_offset = 0; additional_dataset_offset < num_new_nodes;
       additional_dataset_offset += max_chunk_size_) {
    const auto actual_chunk_size =
      std::min(num_new_nodes - additional_dataset_offset, max_chunk_size_);
    // TODO: make_host_strided_matrix_view
    const auto batch_base_size = using_free_slots ? initial_dataset_size : (initial_dataset_size + additional_dataset_offset);
    const auto batch_new_size = using_free_slots ? initial_dataset_size : (initial_dataset_size + additional_dataset_offset + actual_chunk_size);
    auto dataset_view = raft::make_host_matrix_view<const T, std::int64_t>(
      input_updated_dataset_view.data_handle(),
      batch_base_size, dim);
    auto graph_view = raft::make_host_matrix_view<const IdxT, std::int64_t>(
      updated_graph_view.data_handle(), 
      batch_base_size, degree);

    raft::device_matrix_view<const T, std::int64_t> d_dataset_view;
    if (mapper.is_full()) {
      d_dataset_view = input_updated_d_dataset_view;
    } else {
      d_dataset_view = raft::make_device_strided_matrix_view<const T, std::int64_t>(
        input_updated_d_dataset_view.data_handle(),
        initial_d_dataset_size + additional_dataset_offset,
        dim,
        stride);
    }

    // update dataset and graph, both on host
    internal_index.update_dataset(handle, dataset_view, d_dataset_view);
    internal_index.update_graph(handle, graph_view);
    raft::resource::sync_stream(handle);
    
    auto updated_graph = raft::make_host_matrix_view<IdxT, std::int64_t>(
      updated_graph_view.data_handle(),
      batch_new_size, degree);
    auto updated_in_edges = raft::make_host_vector_view<int, std::int64_t>(
      updated_in_edges_view.data_handle(),
      batch_new_size);
    auto updated_d_in_edges = raft::make_device_vector_view<int, std::int64_t>(
      index.d_in_edges().data_handle(),
      batch_new_size);
    internal_index.update_in_edges(updated_in_edges, updated_d_in_edges);

    // TODO: addtional data on device
    std::size_t current_insert_position = insert_position + additional_dataset_offset;
    auto additional_dataset_view = raft::make_host_matrix_view<const T, std::int64_t>(
      input_updated_dataset_view.data_handle() +
        current_insert_position * stride,
      actual_chunk_size,
      dim);
    neighbors::cagra::add_node_core<T, IdxT>(
      handle, internal_index, additional_dataset_view, updated_graph, 
      updated_d_graph_view, updated_in_edges, chunk_id, current_insert_position, params, search_ctx);
    // RAFT_LOG_INFO("sizeof chunk = %lu, chunk_id = %u", actual_chunk_size, chunk_id);
    sum_miss_rates += mapper.miss_rate;
    
    auto d_in_edges = internal_index.d_in_edges();
    raft::copy(d_in_edges.data_handle() + current_insert_position,
                updated_in_edges.data_handle() + current_insert_position,
                actual_chunk_size,
                raft::resource::get_cuda_stream(handle));
    raft::resource::sync_stream(handle);
    
    bool mapper_full = mapper.is_full();
    if (using_free_slots) {
        mapper.batch_insert_free_slots(current_insert_position, actual_chunk_size, handle, 
          mapper_full, initial_d_dataset_size + additional_dataset_offset);
    } else {
      mapper.batch_insert(initial_dataset_size + additional_dataset_offset, 
                          initial_d_dataset_size + additional_dataset_offset, actual_chunk_size, handle, mapper_full);  
    }

    bool graph_mapper_full = graph_mapper.is_full();
    if (using_free_slots) {
      graph_mapper.batch_insert_free_slots(current_insert_position, actual_chunk_size, handle,
                          graph_mapper_full, initial_d_graph_size + additional_dataset_offset);
    } else {
      graph_mapper.batch_insert(initial_dataset_size + additional_dataset_offset, 
                                initial_d_graph_size + additional_dataset_offset, actual_chunk_size, handle, graph_mapper_full);
    }
    if (!graph_mapper_full) {
      size_t current_size = initial_d_graph_size + additional_dataset_offset + actual_chunk_size;
      auto new_d_graph_view = raft::make_device_matrix_view<IdxT, std::int64_t>(
          updated_d_graph_view.data_handle(),
          current_size,
          degree);
      assert(internal_index.d_graph().size() == current_size - actual_chunk_size);
      raft::copy(new_d_graph_view.data_handle() + (current_size - actual_chunk_size) * degree,
                updated_graph_view.data_handle() + current_insert_position * degree,
                actual_chunk_size * degree,
                raft::resource::get_cuda_stream(handle));
      internal_index.update_d_graph(handle, new_d_graph_view);
    }          

    if (using_free_slots) {
      auto delete_bitset = index.get_delete_bitset_ptr();
      auto host_delete_bitset = index.get_host_delete_bitset_ptr();
      // set delete_tag for batch data as true
      host_delete_bitset->mark_range_valid(current_insert_position, current_insert_position + actual_chunk_size);
      auto flip_indices = 
        raft::make_device_vector<int64_t, int64_t>(handle, actual_chunk_size);
    
      thrust::sequence(thrust::cuda::par.on(raft::resource::get_cuda_stream(handle)),
                          thrust::device_pointer_cast(flip_indices.data_handle()),
                          thrust::device_pointer_cast(flip_indices.data_handle() + actual_chunk_size),
                          current_insert_position);
      delete_bitset->set(handle, flip_indices.view(), true);
    }

    // if ( (chunk_id + 1) % 10 == 0) {
    //   // TODO: temp: 只在测试纯插入过程中需要定期更新old d_in_edges
    //   auto d_in_edges = internal_index.d_in_edges();
    //   raft::copy(d_in_edges.data_handle() + initial_dataset_size,
    //              updated_in_edges.data_handle() + initial_dataset_size,
    //              additional_dataset_offset + actual_chunk_size,
    //              raft::resource::get_cuda_stream(handle));
    //   // mapper.decay_recent_access(0.5, handle);
    // }

  //   if ( (chunk_id + 1) % 50 == 0 || (chunk_id + 1) % 50 == 1) {
  //     int* in_edges = updated_in_edges.data_handle();
  // //     size_t graph_size = updated_in_edges.size();
  // //     std::memset(in_edges, 0, graph_size * sizeof(int));
      
  // // #pragma omp parallel for
  // //     for (size_t n = 0; n < graph_size; n++) {
  // //         for (size_t i = 0; i < degree; i++) {
  // //             auto neighbor = updated_graph(n, i);
  // // #pragma omp atomic
  // //             in_edges[neighbor]++;
  // //         }
  // //     }

  //     auto d_in_edges = internal_index.d_in_edges();
  //     std::string count_file_name = bench_config::instance().get_count_log_path(chunk_id+1);
  //     // mapper.snapshot_access_counts(count_file_name, updated_in_edges.data_handle(), raft::resource::get_cuda_stream(handle));
  //     mapper.snapshot_access_counts(count_file_name, updated_in_edges.data_handle(), d_in_edges.data_handle(), raft::resource::get_cuda_stream(handle));
  //   }
    mapper.decay_recent_access(0.667, handle);
    chunk_id++;
  }

  index.hd_mapper_ptr()->miss_rate = sum_miss_rates / chunk_id;
  index.update_d_graph(handle, internal_index.d_graph());
}

// Note: remove class Accessor in template
template <class T, class IdxT>
void extend_core(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::host_matrix_view<const T, int64_t, raft::row_major> additional_dataset,
  ffanns::neighbors::cagra::index<T, IdxT>& index,
  std::optional<raft::host_matrix_view<T, int64_t, raft::layout_stride>> new_dataset_buffer_view,
  std::optional<raft::device_matrix_view<T, int64_t, raft::layout_stride>> new_d_dataset_buffer_view,
  std::optional<raft::host_matrix_view<IdxT, int64_t, raft::layout_stride>> new_graph_buffer_view,
  std::optional<raft::device_matrix_view<IdxT, int64_t>> new_d_graph_buffer_view,
  IdxT start_id, IdxT end_id,
  ffanns::neighbors::cagra::search_context<T, IdxT>* search_ctx = nullptr)
{
  const std::size_t num_new_nodes        = additional_dataset.extent(0);
  const std::size_t initial_dataset_size = index.size();
  const std::size_t degree               = index.graph_degree();
  const std::size_t dim                  = index.dim();
  const std::size_t initial_d_dataset_size = index.d_dataset().extent(0);
  // const std::size_t initial_d_graph_size = index.d_graph().extent(0);

  std::size_t insert_position = initial_dataset_size;
  bool using_free_slots = false;
  auto free_slots = index.get_free_slots_ptr();
  for (auto it = free_slots->begin(); it != free_slots->end(); ++it) {
    auto& range = *it;
    auto range_size = range.second - range.first;
  
    if (range_size >= num_new_nodes) {
      insert_position = range.first;
      using_free_slots = true;
      
      range.first += num_new_nodes;
      // slot is use up and delete it
      if (range.first >= range.second) {
        free_slots->erase(it);
      }
      RAFT_LOG_INFO("[extend_core] using free_slots with range [%d, %d) to insert %ld new nodes", 
                    insert_position, insert_position + num_new_nodes, num_new_nodes);
      break;
    }
  }

  // const std::size_t new_dataset_size     = initial_dataset_size + num_new_nodes;
  const std::size_t new_dataset_size = using_free_slots ? 
    initial_dataset_size : 
    initial_dataset_size + num_new_nodes;

  if (new_dataset_buffer_view.has_value() && (new_dataset_size != initial_dataset_size) &&
      static_cast<std::size_t>(new_dataset_buffer_view.value().extent(0)) != new_dataset_size) {
    RAFT_LOG_ERROR(
      "The extended dataset size (%lu) must be the initial dataset size (%lu) + additional dataset "
      "size (%lu). "
      "Please fix the memory buffer size for the extended dataset.",
      new_dataset_buffer_view.value().extent(0),
      initial_dataset_size,
      num_new_nodes);
  }
  // TODO: new_d_dataset_buffer is full, need replacement

  using ds_idx_type = decltype(index.data().n_rows());
  
  const auto* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index.data());
  assert(strided_dset != nullptr && "index.data() must be of type strided_dataset.");
  const auto stride    = strided_dset->stride();
  
  auto updated_dataset_view =
      raft::make_host_matrix_view<T, std::int64_t>(nullptr, 0, stride);
  assert(new_dataset_buffer_view.has_value() && "new_dataset_buffer_view must be provided");
  updated_dataset_view = new_dataset_buffer_view.value();

  // auto updated_graph = raft::make_host_matrix<IdxT, std::int64_t>(new_dataset_size, degree);// 
  auto updated_graph_view = raft::make_host_matrix_view<IdxT, std::int64_t>(nullptr, 0, degree);
  assert(new_graph_buffer_view.has_value() && "new_graph_buffer_view must be provided");
  updated_graph_view = new_graph_buffer_view.value();
  // 使用 assert 进行动态类型检查
  // Ensure the old dataset area is filled with the original dataset
  if (using_free_slots) {
    // copy only when free slots are used, as addtinitoal_datasize is included in the data_space, 
    // implemented by workload_manager.cu
    updated_dataset_view = raft::make_host_matrix_view<T, std::int64_t>(
      updated_dataset_view.data_handle(), 
      initial_dataset_size, 
      stride);
    memcpy(updated_dataset_view.data_handle() + insert_position * stride,
           additional_dataset.data_handle(),
           sizeof(T) * num_new_nodes * stride);
    updated_graph_view = raft::make_host_matrix_view<IdxT, std::int64_t>(
      updated_graph_view.data_handle(), 
      initial_dataset_size, 
      degree);
  } 
  else {
    std::memset(updated_graph_view.data_handle() + initial_dataset_size * degree, 
              0,  // 或者其他表示"无连接"的值，如0xFF
              num_new_nodes * degree * sizeof(IdxT));
  }

  assert(new_d_dataset_buffer_view.has_value() && "new_device_dataset_buffer_view must be provided");
  auto updated_d_dataset_view = new_d_dataset_buffer_view.value();
  // not full
  if (static_cast<std::size_t>(updated_d_dataset_view.extent(0)) > initial_d_dataset_size) {
    raft::copy(updated_d_dataset_view.data_handle() + initial_d_dataset_size * stride,
               additional_dataset.data_handle(),
               num_new_nodes * stride,
               raft::resource::get_cuda_stream(handle));
  }
  assert(new_d_graph_buffer_view.has_value() && "new_d_graph_buffer_view must be provided");
  auto updated_d_graph_view = new_d_graph_buffer_view.value();

  auto host_in_edges = index.host_in_edges();
  auto updated_in_edges = raft::make_host_vector_view<int, std::int64_t>(host_in_edges.data_handle(), new_dataset_size);
  auto d_in_edges = index.d_in_edges();
  auto updated_d_in_edges = raft::make_device_vector_view<int, std::int64_t>(d_in_edges.data_handle(), new_dataset_size);
  thrust::fill(thrust::cuda::par.on(raft::resource::get_cuda_stream(handle)),
               updated_d_in_edges.data_handle() + insert_position,
               updated_d_in_edges.data_handle() + insert_position + num_new_nodes,
               0);

  // Add graph nodes
  ffanns::neighbors::cagra::add_graph_nodes<T, IdxT>(
    handle, raft::make_const_mdspan(updated_dataset_view), 
    updated_d_dataset_view, index, updated_graph_view, updated_in_edges, 
    updated_d_graph_view, params, insert_position, num_new_nodes, search_ctx);

  // Update index dataset
  // Note that we delete raft::make_const_mdspan for in-place update reservation
  index.update_dataset(handle, updated_dataset_view, updated_d_dataset_view);
  // Update index graph
  index.update_graph(handle, updated_graph_view);
  // index.own_graph(handle, updated_graph.view());
  // index.own_graph(std::move(updated_graph));
  index.update_in_edges(updated_in_edges, updated_d_in_edges);

  auto tag_to_id = index.get_tag_to_id_ptr();
  assert(tag_to_id != nullptr && "tag_to_id must be provided");
  assert(start_id != 0 && end_id != 0 && start_id < end_id && "start_id and end_id must be provided");
  // RAFT_LOG_INFO("[extend_core] start_id = %u, end_id = %u", start_id, end_id);

  if (tag_to_id->size() < new_dataset_size) {
    tag_to_id->resize(new_dataset_size, raft::resource::get_cuda_stream(handle));
  }
  thrust::sequence(thrust::cuda::par.on(raft::resource::get_cuda_stream(handle)),
                  tag_to_id->begin() + insert_position,
                  tag_to_id->begin() + insert_position + num_new_nodes,
                  start_id);
}

} // namespace ffanns::neighbors::cagra
