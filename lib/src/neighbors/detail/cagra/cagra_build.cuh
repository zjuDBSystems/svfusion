#pragma once

// #include "../../../core/nvtx.hpp"
// #include "../../vpq_dataset.cuh"
#include "graph_core.cuh"
#include "ffanns/neighbors/cagra.hpp"
#include <chrono>
#include <cstdio>
#include <vector>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger-ext.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include "ffanns/distance/distance.hpp"
// #include <cuvs/neighbors/ivf_pq.hpp>
// #include <cuvs/neighbors/refine.hpp>

// TODO: Fixme- this needs to be migrated
#include "../../nn_descent.cuh"

// TODO: This shouldn't be calling spatial/knn APIs
#include "../ann_utils.cuh"

#include <rmm/resource_ref.hpp>
#include <rmm/mr/device/tracking_resource_adaptor.hpp>
#include <thrust/sequence.h>

namespace ffanns::neighbors::cagra::detail {

static const std::string RAFT_NAME = "raft";

template <typename IdxT>
void host_build_in_edges(
    int* output,
    size_t output_size,
    const IdxT* graph,
    size_t num_edges
) {
    // 初始化输出数组
    std::memset(output, 0, output_size * sizeof(int));
    #pragma omp parallel for
    for (size_t i = 0; i < num_edges; ++i) {
        IdxT id = graph[i];
        if (id < output_size) {
            #pragma omp atomic
            output[id]++;
        }
    }
}

 // 添加内存监控代码
 inline void print_memory_info() {
  // 在您设置的内存池中获取信息
  auto current_mr = rmm::mr::get_current_device_resource();
  auto tracking_mr_ptr = dynamic_cast<rmm::mr::tracking_resource_adaptor<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>>*>(current_mr);
  
  if (tracking_mr_ptr != nullptr) {
      // 成功转换为 tracking_resource_adaptor
      size_t allocated_bytes = tracking_mr_ptr->get_allocated_bytes();
      double usage_percentage = (static_cast<double>(allocated_bytes) / (39.0 * 1024 * 1024 * 1024)) * 100.0;
                   
      RAFT_LOG_INFO("内存池已分配: %.2f GB (%.2f%% 的最大容量)",
                   allocated_bytes / (1024.0 * 1024.0 * 1024.0),
                   usage_percentage);
                   
      // // 如果需要查看未释放的分配，可以打印详细信息
      // if (allocated_bytes > 30 * 1024 * 1024 * 1024ull) { // 如果已分配超过30GB
      //     RAFT_LOG_WARN("内存使用超过30GB，打印未释放分配");
      //     RAFT_LOG_WARN("%s", tracking_mr_ptr->get_outstanding_allocations_str().c_str());
      // }
  } else {
    RAFT_LOG_WARN("无法获取内存池信息，当前资源不是tracking_resource_adaptor");
  }

}

template <typename DataT, typename IdxT, typename accessor>
void build_knn_graph(
  raft::resources const& res,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  ffanns::neighbors::nn_descent::index_params build_params)
{
  auto nn_descent_idx = ffanns::neighbors::nn_descent::index<IdxT>(res, knn_graph, build_params.metric);
  RAFT_LOG_INFO("[build_knn_graph] nn_descent start to build");
  print_memory_info();
  ffanns::neighbors::nn_descent::build<DataT, IdxT>(res, build_params, dataset, nn_descent_idx);
  RAFT_LOG_INFO("[build_knn_graph] nn_desenct finish building");
  print_memory_info();

  using internal_IdxT = typename std::make_unsigned<IdxT>::type;
  using g_accessor    = typename decltype(nn_descent_idx.graph())::accessor_type;
  using g_accessor_internal =
    raft::host_device_accessor<std::experimental::default_accessor<internal_IdxT>,
                               g_accessor::mem_type>;

  auto knn_graph_internal =
    raft::mdspan<internal_IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor_internal>(
      reinterpret_cast<internal_IdxT*>(nn_descent_idx.graph().data_handle()),
      nn_descent_idx.graph().extent(0),
      nn_descent_idx.graph().extent(1));
  
  detail::graph::sort_knn_graph(res, dataset, knn_graph_internal);
  // RAFT_LOG_INFO("[build_knn_graph::knn_graph] [first element] = %d", *static_cast<const IdxT*>(knn_graph.data_handle()));
  // raft::neighbors::cagra::detail::graph::sort_knn_graph(res, dataset, knn_graph_internal);
}

template <
  typename IdxT = uint32_t,
  typename g_accessor =
    raft::host_device_accessor<std::experimental::default_accessor<IdxT>, raft::memory_type::host>>
void optimize(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor> knn_graph,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> new_graph,
  const bool guarantee_connectivity = false)
{
  using internal_IdxT = typename std::make_unsigned<IdxT>::type;

  auto new_graph_internal = raft::make_host_matrix_view<internal_IdxT, int64_t>(
    reinterpret_cast<internal_IdxT*>(new_graph.data_handle()),
    new_graph.extent(0),
    new_graph.extent(1));

  using g_accessor_internal =
    raft::host_device_accessor<std::experimental::default_accessor<internal_IdxT>,
                               raft::memory_type::host>;
  auto knn_graph_internal =
    raft::mdspan<internal_IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor_internal>(
      reinterpret_cast<internal_IdxT*>(knn_graph.data_handle()),
      knn_graph.extent(0),
      knn_graph.extent(1));

  detail::graph::optimize(res, knn_graph_internal, new_graph_internal, guarantee_connectivity);
  // raft::neighbors::cagra::detail::graph::optimize(
  //   res, knn_graph_internal, new_graph_internal, guarantee_connectivity);
}

template <typename T,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
index<T, IdxT> build(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> index_graph, 
  // raft::device_matrix_view<T, int64_t, raft::layout_stride> device_dataset_view,
  raft::device_matrix<T, int64_t>& device_dataset_ref,
  raft::device_matrix<IdxT, int64_t>& device_graph_ref,
  std::shared_ptr<ffanns::core::bitset<std::uint32_t, int64_t>> delete_bitset,
  std::shared_ptr<rmm::device_uvector<uint32_t>> tag_to_id,
  raft::host_vector_view<int, int64_t> host_in_edges_view,
  raft::device_vector_view<int, int64_t> d_in_edges_view,
  uint32_t start_id, uint32_t end_id)
{
  RAFT_LOG_INFO("Memory status before build process:");
  print_memory_info();

  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;
  size_t dataset_rows = dataset.extent(0);
  // size_t d_dataset_rows = device_dataset_view.extent(0);
  size_t max_device_rows = index<T, uint32_t>::get_max_device_rows();
  size_t d_dataset_rows = std::min(dataset_rows, max_device_rows);
  // size_t d_graph_rows = device_graph_view.extent(0);
  size_t max_device_graph_rows = index<T, uint32_t>::get_max_graph_device_rows();
  size_t d_graph_rows = std::min(dataset_rows, max_device_graph_rows);
  RAFT_LOG_INFO("[build::cagra_graph] [intermediate_degree] = %lu, [graph_degree] = %lu", intermediate_degree, graph_degree);
  if (intermediate_degree >= static_cast<size_t>(dataset.extent(0))) {
    RAFT_LOG_WARN(
      "Intermediate graph degree cannot be larger than dataset size, reducing it to %lu",
      dataset.extent(0));
    intermediate_degree = dataset.extent(0) - 1;
  }
  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }

  std::optional<raft::host_matrix<IdxT, int64_t>> knn_graph(
    raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), intermediate_degree));
  
  // Currently use nn-descent
  // auto knn_build_params = graph_build_params::nn_descent_params(intermediate_degree);
  auto nn_descent_params = graph_build_params::nn_descent_params(intermediate_degree, params.metric);
  // exceed device capacity, need to batch_build
  if (dataset_rows > d_dataset_rows) {
    size_t num_batches = (dataset_rows + d_dataset_rows - 1) / d_dataset_rows;
    RAFT_LOG_INFO("Dataset size (%lu) exceeds device capacity (%lu). Using %lu clusters for batch processing.", 
      dataset_rows, d_dataset_rows, num_batches * 2);
    nn_descent_params.n_clusters = num_batches * 2;
  } else {
    nn_descent_params.n_clusters = 1;
  }
  // Use nn-descent to build CAGRA knn graph
  build_knn_graph<T, IdxT>(res, dataset, knn_graph->view(), nn_descent_params);

  RAFT_LOG_INFO("Memory status after build knn:");
  print_memory_info();

  // auto cagra_graph = raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), graph_degree);
  // optimize<IdxT>(res, knn_graph->view(), cagra_graph.view(), params.guarantee_connectivity);
  optimize<IdxT>(res, knn_graph->view(), index_graph, params.guarantee_connectivity);
  RAFT_LOG_INFO("optimizing graph");
  

  // free intermediate graph before trying to create the index
  knn_graph.reset();
  RAFT_LOG_INFO("Graph optimized, creating index");

  index<T, IdxT> idx(res, params.metric);
  // RAFT_LOG_INFO("[build::cagra_graph] [first element] = %d", *static_cast<const IdxT*>(cagra_graph.data_handle()));
  
  idx.update_graph(res, index_graph);
  // idx.own_graph(res, index_graph);
   
  // update d_graph
  device_graph_ref = raft::make_device_matrix<IdxT, int64_t>(
    res, max_device_graph_rows, graph_degree);
  auto device_graph_view = raft::make_device_matrix_view<IdxT, int64_t>(
    device_graph_ref.data_handle(), d_graph_rows, graph_degree);
  raft::copy(device_graph_view.data_handle(), index_graph.data_handle(), 
              d_graph_rows * graph_degree, raft::resource::get_cuda_stream(res));
  idx.update_d_graph(res, device_graph_view);

  auto &mapper = idx.hd_mapper();
  auto stream = raft::resource::get_cuda_stream(res);
  mapper.host_device_mapping.resize(dataset.extent(0), stream); 
  mapper.d_host_device_mapping = mapper.host_device_mapping.data();
  mapper.access_counts.resize(dataset.extent(0), stream);
  mapper.recent_access.resize(dataset.extent(0), stream);
  mapper.d_access_counts = mapper.access_counts.data();
  mapper.d_recent_access = mapper.recent_access.data();
  auto &graph_mapper = idx.get_graph_hd_mapper();
  device_dataset_ref = raft::make_device_matrix<T, int64_t>(
    res, max_device_rows, dataset.extent(1));
  auto device_dataset_view = raft::make_device_matrix_view<T, int64_t>(
    device_dataset_ref.data_handle(), d_dataset_rows, dataset.extent(1));
  if (dataset_rows > d_dataset_rows) {
    size_t num_partitions = 4; 
    assert (dataset_rows % num_partitions == 0);
    assert (d_dataset_rows % num_partitions == 0);
    size_t datarow_per_partition = dataset_rows / num_partitions;
    size_t d_datarow_per_partition = d_dataset_rows / num_partitions;

    thrust::fill(thrust::cuda::par.on(stream),
                  mapper.host_device_mapping.begin(),
                  mapper.host_device_mapping.end(),
                  mapper.INVALID_ID);

    for (size_t p = 0; p < num_partitions; p++) {
        RAFT_LOG_INFO("[build::cagra_graph] device start: %lu, host start: %lu", 
                      p * d_datarow_per_partition, p * datarow_per_partition);
        thrust::sequence(thrust::cuda::par.on(stream),
                  mapper.host_device_mapping.begin() + p * datarow_per_partition,
                  mapper.host_device_mapping.begin() + p * datarow_per_partition + d_datarow_per_partition,
                  p * d_datarow_per_partition);
        thrust::sequence(thrust::cuda::par.on(stream),
                  mapper.device_host_mapping.begin() + p * d_datarow_per_partition,
                  mapper.device_host_mapping.begin() + (p+1) * d_datarow_per_partition,
                  p * datarow_per_partition);
        raft::copy(device_dataset_view.data_handle() + p * d_datarow_per_partition * dataset.extent(1), 
                   dataset.data_handle() + p * datarow_per_partition * dataset.extent(1), 
                   d_datarow_per_partition * dataset.extent(1), raft::resource::get_cuda_stream(res));
    }
  } else {
    thrust::sequence(thrust::cuda::par.on(stream),
                    mapper.host_device_mapping.begin(),
                    mapper.host_device_mapping.begin() + d_dataset_rows);
    thrust::fill(thrust::cuda::par.on(stream),
                    mapper.host_device_mapping.begin() + d_dataset_rows,
                    mapper.host_device_mapping.end(),
                    mapper.INVALID_ID);
    thrust::sequence(thrust::cuda::par.on(stream),
                    mapper.device_host_mapping.begin(),
                    mapper.device_host_mapping.begin() + d_dataset_rows);
    raft::copy(device_dataset_view.data_handle(), dataset.data_handle(), 
            d_dataset_rows * dataset.extent(1), raft::resource::get_cuda_stream(res));
  }
  thrust::fill(thrust::cuda::par.on(stream),
                  mapper.access_counts.begin(),
                  mapper.access_counts.end(),
                  0);
  thrust::fill(thrust::cuda::par.on(stream),
                  mapper.recent_access.begin(),
                  mapper.recent_access.end(),
                  0.0f);
  mapper.current_size = d_dataset_rows;
  idx.update_dataset(res, dataset, device_dataset_view);
  
  graph_mapper.host_device_mapping.resize(dataset.extent(0), stream); 
  graph_mapper.d_host_device_mapping = graph_mapper.host_device_mapping.data();
  graph_mapper.access_counts.resize(dataset.extent(0), stream);
  graph_mapper.d_access_counts = graph_mapper.access_counts.data();

  thrust::sequence(thrust::cuda::par.on(stream),
                  graph_mapper.host_device_mapping.begin(),
                  graph_mapper.host_device_mapping.begin() + d_graph_rows);
  thrust::fill(thrust::cuda::par.on(stream),
                  graph_mapper.host_device_mapping.begin() + d_graph_rows,
                  graph_mapper.host_device_mapping.end(),
                  graph_mapper.INVALID_ID);
  thrust::sequence(thrust::cuda::par.on(stream),
                  graph_mapper.device_host_mapping.begin(),
                  graph_mapper.device_host_mapping.begin() + d_graph_rows);
  thrust::fill(thrust::cuda::par.on(stream),
                  graph_mapper.access_counts.begin(),
                  graph_mapper.access_counts.end(),
                  0);
  graph_mapper.current_size = d_graph_rows;

  idx.update_delete_bitset(delete_bitset);

  // update tag_to_id
  auto num = end_id - start_id;
  if (tag_to_id->size() < num) {
    tag_to_id->resize(num, stream);
  }
  thrust::sequence(thrust::cuda::par.on(stream),
                  tag_to_id->begin(),
                  tag_to_id->begin() + num);
  idx.update_tag_to_id(tag_to_id);

  // Initialize in_edges using external host view
  host_build_in_edges(host_in_edges_view.data_handle(),
                      dataset.extent(0),
                      index_graph.data_handle(),
                      dataset.extent(0) * graph_degree);
  RAFT_LOG_INFO("[build] in_edges initialized via external host view with size %lu", dataset.extent(0));
  raft::copy(d_in_edges_view.data_handle(), host_in_edges_view.data_handle(), dataset.extent(0), raft::resource::get_cuda_stream(res));
  idx.update_in_edges(host_in_edges_view, d_in_edges_view);

  RAFT_LOG_INFO("Memory status at the end of build process:");
  print_memory_info();

  return idx;
}

} // namespace ffanns::neighbors::cagra::detail
