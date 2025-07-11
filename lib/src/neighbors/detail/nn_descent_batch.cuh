#pragma once

#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <sys/types.h>
#undef RAFT_EXPLICIT_INSTANTIATE_ONLY

#include "nn_descent.cuh"
#include "ffanns/neighbors/brute_force.hpp"
#include "ffanns/neighbors/nn_descent.hpp"

#include "ffanns/cluster/kmeans.hpp"
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/managed_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/detail/gather_inplace.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/sample_rows.cuh>

#include <rmm/resource_ref.hpp>
#include <rmm/mr/device/tracking_resource_adaptor.hpp>

#include <thrust/copy.h>

#include <vector_types.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <type_traits>
namespace ffanns::neighbors::nn_descent::detail::experimental {

static const std::string RAFT_NAME = "raft";

template <typename T,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<std::experimental::default_accessor<T>, raft::memory_type::host>>
void get_balanced_kmeans_centroids(
  raft::resources const& res,
  ffanns::distance::DistanceType metric,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  raft::device_matrix_view<T, IdxT> centroids)
{
  size_t num_rows   = static_cast<size_t>(dataset.extent(0));
  size_t num_cols   = static_cast<size_t>(dataset.extent(1));
  size_t n_clusters = centroids.extent(0);
  size_t num_subsamples =
    std::min(static_cast<size_t>(num_rows / n_clusters), static_cast<size_t>(num_rows * 0.1));

  auto d_subsample_dataset =
    raft::make_device_matrix<T, int64_t, raft::row_major>(res, num_subsamples, num_cols);
  raft::matrix::sample_rows<T, int64_t, Accessor>(
    res, raft::random::RngState{0}, dataset, d_subsample_dataset.view());

  ffanns::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.metric = metric;

  auto d_subsample_dataset_const_view =
    raft::make_device_matrix_view<const T, int, raft::row_major>(
      d_subsample_dataset.data_handle(), num_subsamples, num_cols);
  auto centroids_view = raft::make_device_matrix_view<T, int, raft::row_major>(
    centroids.data_handle(), n_clusters, num_cols);
  ffanns::cluster::kmeans::fit(res, kmeans_params, d_subsample_dataset_const_view, centroids_view);
}

//
// Get the top k closest centroid indices for each data point
// Loads the data in batches onto device if data is on host for memory efficiency
//
template <typename T, typename IdxT = uint32_t>
void get_global_nearest_k(
  raft::resources const& res,
  size_t k,
  size_t num_rows,
  size_t n_clusters,
  const T* dataset,
  raft::host_matrix_view<IdxT, IdxT, raft::row_major> global_nearest_cluster,
  raft::device_matrix_view<T, IdxT, raft::row_major> centroids,
  ffanns::distance::DistanceType metric)
{
  size_t num_cols     = centroids.extent(1);
  auto centroids_view = raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
    centroids.data_handle(), n_clusters, num_cols);

  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, dataset));
  float* ptr = reinterpret_cast<float*>(attr.devicePointer);

  size_t num_batches = n_clusters;
  size_t batch_size  = (num_rows + n_clusters) / n_clusters;
  if (ptr == nullptr) {  // data on host

    auto d_dataset_batch =
      raft::make_device_matrix<T, int64_t, raft::row_major>(res, batch_size, num_cols);

    auto nearest_clusters_idx =
      raft::make_device_matrix<int64_t, int64_t, raft::row_major>(res, batch_size, k);
    auto nearest_clusters_idxt =
      raft::make_device_matrix<IdxT, int64_t, raft::row_major>(res, batch_size, k);
    auto nearest_clusters_dist =
      raft::make_device_matrix<T, int64_t, raft::row_major>(res, batch_size, k);

    for (size_t i = 0; i < num_batches; i++) {
      size_t batch_size_ = batch_size;

      if (i == num_batches - 1) { batch_size_ = num_rows - batch_size * i; }
      raft::copy(d_dataset_batch.data_handle(),
                 dataset + i * batch_size * num_cols,
                 batch_size_ * num_cols,
                 raft::resource::get_cuda_stream(res));

      std::optional<raft::device_vector_view<const T, int64_t>> norms_view;
      ffanns::neighbors::brute_force::index<T> brute_force_index(
        res, centroids_view, norms_view, metric);
      ffanns::neighbors::brute_force::search(res,
                                           brute_force_index,
                                           raft::make_const_mdspan(d_dataset_batch.view()),
                                           nearest_clusters_idx.view(),
                                           nearest_clusters_dist.view());

      thrust::copy(raft::resource::get_thrust_policy(res),
                   nearest_clusters_idx.data_handle(),
                   nearest_clusters_idx.data_handle() + nearest_clusters_idx.size(),
                   nearest_clusters_idxt.data_handle());
      raft::copy(global_nearest_cluster.data_handle() + i * batch_size * k,
                 nearest_clusters_idxt.data_handle(),
                 batch_size_ * k,
                 raft::resource::get_cuda_stream(res));
    }
  } else {  // data on device
    auto nearest_clusters_idx =
      raft::make_device_matrix<int64_t, int64_t, raft::row_major>(res, num_rows, k);
    auto nearest_clusters_dist =
      raft::make_device_matrix<T, int64_t, raft::row_major>(res, num_rows, k);

    std::optional<raft::device_vector_view<const T, int64_t>> norms_view;
    ffanns::neighbors::brute_force::index<T> brute_force_index(
      res, centroids_view, norms_view, metric);
    auto dataset_view =
      raft::make_device_matrix_view<const T, int64_t, raft::row_major>(dataset, num_rows, num_cols);
    ffanns::neighbors::brute_force::search(res,
                                         brute_force_index,
                                         dataset_view,
                                         nearest_clusters_idx.view(),
                                         nearest_clusters_dist.view());

    auto nearest_clusters_idxt =
      raft::make_device_matrix<IdxT, int64_t, raft::row_major>(res, batch_size, k);
    for (size_t i = 0; i < num_batches; i++) {
      size_t batch_size_ = batch_size;

      if (i == num_batches - 1) { batch_size_ = num_rows - batch_size * i; }
      thrust::copy(raft::resource::get_thrust_policy(res),
                   nearest_clusters_idx.data_handle() + i * batch_size_ * k,
                   nearest_clusters_idx.data_handle() + (i + 1) * batch_size_ * k,
                   nearest_clusters_idxt.data_handle());
      raft::copy(global_nearest_cluster.data_handle() + i * batch_size_ * k,
                 nearest_clusters_idxt.data_handle(),
                 batch_size_ * k,
                 raft::resource::get_cuda_stream(res));
    }
  }
}

//
// global_nearest_cluster [num_rows X k=2] : top 2 closest clusters for each data point
// inverted_indices [num_rows x k vector] : sparse vector for data indices for each cluster
// cluster_size [n_cluster] : cluster size for each cluster
// offset [n_cluster] : offset in inverted_indices for each cluster
// Loads the data in batches onto device if data is on host for memory efficiency
//
template <typename IdxT = uint32_t>
void get_inverted_indices(raft::resources const& res,
                          size_t n_clusters,
                          size_t& max_cluster_size,
                          size_t& min_cluster_size,
                          raft::host_matrix_view<IdxT, IdxT> global_nearest_cluster,
                          raft::host_vector_view<IdxT, IdxT> inverted_indices,
                          raft::host_vector_view<IdxT, IdxT> cluster_size,
                          raft::host_vector_view<IdxT, IdxT> offset)
{
  // build sparse inverted indices and get number of data points for each cluster
  size_t num_rows = global_nearest_cluster.extent(0);
  size_t k        = global_nearest_cluster.extent(1);

  auto local_offset = raft::make_host_vector<IdxT>(n_clusters);

  max_cluster_size = 0;
  min_cluster_size = std::numeric_limits<size_t>::max();

  std::fill(cluster_size.data_handle(), cluster_size.data_handle() + n_clusters, 0);
  std::fill(local_offset.data_handle(), local_offset.data_handle() + n_clusters, 0);

  // TODO: this part isn't really a bottleneck but maybe worth trying omp parallel
  // for with atomic add
  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < k; j++) {
      IdxT cluster_id = global_nearest_cluster(i, j);
      cluster_size(cluster_id) += 1;
    }
  }

  offset(0) = 0;
  for (size_t i = 1; i < n_clusters; i++) {
    offset(i) = offset(i - 1) + cluster_size(i - 1);
  }
  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < k; j++) {
      IdxT cluster_id = global_nearest_cluster(i, j);
      inverted_indices(offset(cluster_id) + local_offset(cluster_id)) = i;
      local_offset(cluster_id) += 1;
    }
  }

  max_cluster_size = static_cast<size_t>(
    *std::max_element(cluster_size.data_handle(), cluster_size.data_handle() + n_clusters));
  min_cluster_size = static_cast<size_t>(
    *std::min_element(cluster_size.data_handle(), cluster_size.data_handle() + n_clusters));
}

template <typename KeyType, typename ValueType>
struct KeyValuePair {
  KeyType key;
  ValueType value;
};

template <typename KeyType, typename ValueType>
struct CustomKeyComparator {
  __device__ bool operator()(const KeyValuePair<KeyType, ValueType>& a,
                             const KeyValuePair<KeyType, ValueType>& b) const
  {
    if (a.key == b.key) { return a.value < b.value; }
    return a.key < b.key;
  }
};

template <typename IdxT, int BLOCK_SIZE, int ITEMS_PER_THREAD>
RAFT_KERNEL merge_subgraphs(IdxT* cluster_data_indices,
                            size_t graph_degree,
                            size_t num_cluster_in_batch,
                            float* global_distances,
                            float* batch_distances,
                            IdxT* global_indices,
                            IdxT* batch_indices)
{
  size_t batch_row = blockIdx.x;
  typedef cub::BlockMergeSort<KeyValuePair<float, IdxT>, BLOCK_SIZE, ITEMS_PER_THREAD>
    BlockMergeSortType;
  __shared__ typename cub::BlockMergeSort<KeyValuePair<float, IdxT>, BLOCK_SIZE, ITEMS_PER_THREAD>::
    TempStorage tmpSmem;

  extern __shared__ char sharedMem[];
  float* blockKeys  = reinterpret_cast<float*>(sharedMem);
  IdxT* blockValues = reinterpret_cast<IdxT*>(&sharedMem[graph_degree * 2 * sizeof(float)]);
  int16_t* uniqueMask =
    reinterpret_cast<int16_t*>(&sharedMem[graph_degree * 2 * (sizeof(float) + sizeof(IdxT))]);

  if (batch_row < num_cluster_in_batch) {
    // load batch or global depending on threadIdx
    size_t global_row = cluster_data_indices[batch_row];

    KeyValuePair<float, IdxT> threadKeyValuePair[ITEMS_PER_THREAD];

    size_t halfway   = BLOCK_SIZE / 2;
    size_t do_global = threadIdx.x < halfway;

    float* distances;
    IdxT* indices;

    if (do_global) {
      distances = global_distances;
      indices   = global_indices;
    } else {
      distances = batch_distances;
      indices   = batch_indices;
    }

    size_t idxBase = (threadIdx.x * do_global + (threadIdx.x - halfway) * (1lu - do_global)) *
                     static_cast<size_t>(ITEMS_PER_THREAD);
    size_t arrIdxBase = (global_row * do_global + batch_row * (1lu - do_global)) * graph_degree;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + i;
      if (colId < graph_degree) {
        threadKeyValuePair[i].key   = distances[arrIdxBase + colId];
        threadKeyValuePair[i].value = indices[arrIdxBase + colId];
      } else {
        threadKeyValuePair[i].key   = std::numeric_limits<float>::max();
        threadKeyValuePair[i].value = std::numeric_limits<IdxT>::max();
      }
    }

    __syncthreads();

    BlockMergeSortType(tmpSmem).Sort(threadKeyValuePair, CustomKeyComparator<float, IdxT>{});

    // load sorted result into shared memory to get unique values
    idxBase = threadIdx.x * ITEMS_PER_THREAD;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + i;
      if (colId < 2 * graph_degree) {
        blockKeys[colId]   = threadKeyValuePair[i].key;
        blockValues[colId] = threadKeyValuePair[i].value;
      }
    }

    __syncthreads();

    // get unique mask
    if (threadIdx.x == 0) { uniqueMask[0] = 1; }
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + i;
      if (colId > 0 && colId < 2 * graph_degree) {
        uniqueMask[colId] = static_cast<int16_t>(blockValues[colId] != blockValues[colId - 1]);
      }
    }

    __syncthreads();

    // prefix sum
    if (threadIdx.x == 0) {
      for (int i = 1; i < 2 * graph_degree; i++) {
        uniqueMask[i] += uniqueMask[i - 1];
      }
    }

    __syncthreads();
    // load unique values to global memory
    if (threadIdx.x == 0) {
      global_distances[global_row * graph_degree] = blockKeys[0];
      global_indices[global_row * graph_degree]   = blockValues[0];
    }

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + i;
      if (colId > 0 && colId < 2 * graph_degree) {
        bool is_unique       = uniqueMask[colId] != uniqueMask[colId - 1];
        int16_t global_colId = uniqueMask[colId] - 1;
        if (is_unique && static_cast<size_t>(global_colId) < graph_degree) {
          global_distances[global_row * graph_degree + global_colId] = blockKeys[colId];
          global_indices[global_row * graph_degree + global_colId]   = blockValues[colId];
        }
      }
    }
  }
}

//
// builds knn graph using NN Descent and merge with global graph
//
template <typename T,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<std::experimental::default_accessor<T>, raft::memory_type::host>>
void build_and_merge(raft::resources const& res,
                     const index_params& params,
                     size_t num_data_in_cluster,
                     size_t graph_degree,
                     size_t int_graph_node_degree,
                     T* cluster_data,
                     IdxT* cluster_data_indices,
                     int* int_graph,
                     IdxT* inverted_indices,
                     IdxT* global_indices_d,
                     float* global_distances_d,
                     IdxT* batch_indices_h,
                     IdxT* batch_indices_d,
                     float* batch_distances_d,
                     GNND<const T, int>& nnd)
{
  // nnd.build(cluster_data, num_data_in_cluster, int_graph, true, batch_distances_d);
  nnd.build(cluster_data, num_data_in_cluster, int_graph);

  // remap indices
#pragma omp parallel for
  for (size_t i = 0; i < num_data_in_cluster; i++) {
    for (size_t j = 0; j < graph_degree; j++) {
      size_t local_idx                      = int_graph[i * int_graph_node_degree + j];
      batch_indices_h[i * graph_degree + j] = inverted_indices[local_idx];
    }
  }

  raft::copy(batch_indices_d,
             batch_indices_h,
             num_data_in_cluster * graph_degree,
             raft::resource::get_cuda_stream(res));

  size_t num_elems     = graph_degree * 2;
  size_t sharedMemSize = num_elems * (sizeof(float) + sizeof(IdxT) + sizeof(int16_t));

  if (num_elems <= 128) {
    merge_subgraphs<IdxT, 32, 4>
      <<<num_data_in_cluster, 32, sharedMemSize, raft::resource::get_cuda_stream(res)>>>(
        cluster_data_indices,
        graph_degree,
        num_data_in_cluster,
        global_distances_d,
        batch_distances_d,
        global_indices_d,
        batch_indices_d);
  } else if (num_elems <= 512) {
    merge_subgraphs<IdxT, 128, 4>
      <<<num_data_in_cluster, 128, sharedMemSize, raft::resource::get_cuda_stream(res)>>>(
        cluster_data_indices,
        graph_degree,
        num_data_in_cluster,
        global_distances_d,
        batch_distances_d,
        global_indices_d,
        batch_indices_d);
  } else if (num_elems <= 1024) {
    merge_subgraphs<IdxT, 128, 8>
      <<<num_data_in_cluster, 128, sharedMemSize, raft::resource::get_cuda_stream(res)>>>(
        cluster_data_indices,
        graph_degree,
        num_data_in_cluster,
        global_distances_d,
        batch_distances_d,
        global_indices_d,
        batch_indices_d);
  } else if (num_elems <= 2048) {
    merge_subgraphs<IdxT, 256, 8>
      <<<num_data_in_cluster, 256, sharedMemSize, raft::resource::get_cuda_stream(res)>>>(
        cluster_data_indices,
        graph_degree,
        num_data_in_cluster,
        global_distances_d,
        batch_distances_d,
        global_indices_d,
        batch_indices_d);
  } else {
    // this is as far as we can get due to the shared mem usage of cub::BlockMergeSort
    RAFT_FAIL("The degree of knn is too large (%lu). It must be smaller than 1024", graph_degree);
  }
  raft::resource::sync_stream(res);
}

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
//
// For each cluster, gather the data samples that belong to that cluster, and
// call build_and_merge
//
template <typename T, typename IdxT = uint32_t>
void cluster_nnd(raft::resources const& res,
                 const index_params& params,
                 size_t graph_degree,
                 size_t extended_graph_degree,
                 size_t max_cluster_size,
                 raft::host_matrix_view<const T, int64_t> dataset,
                 IdxT* offsets,
                 IdxT* cluster_size,
                 IdxT* cluster_data_indices,
                 int* int_graph,
                 IdxT* inverted_indices,
                 IdxT* global_indices_h,
                 float* global_distances_h,
                 IdxT* batch_indices_h,
                 IdxT* batch_indices_d,
                 float* batch_distances_d,
                 const BuildConfig& build_config)
{
  size_t num_rows = dataset.extent(0);
  size_t num_cols = dataset.extent(1);

  GNND<const T, int> nnd(res, build_config);

  auto cluster_data_matrix =
    raft::make_host_matrix<T, int64_t, raft::row_major>(max_cluster_size, num_cols);

  for (size_t cluster_id = 0; cluster_id < params.n_clusters; cluster_id++) {
    RAFT_LOG_DEBUG(
      "# Data on host. Running clusters: %lu / %lu", cluster_id + 1, params.n_clusters);
    size_t num_data_in_cluster = cluster_size[cluster_id];
    size_t offset              = offsets[cluster_id];
    RAFT_LOG_INFO("# Data on host. Running clusters: %lu / %lu, size of num_data= %lu", cluster_id + 1, params.n_clusters, num_data_in_cluster);

#pragma omp parallel for
    for (size_t i = 0; i < num_data_in_cluster; i++) {
      for (size_t j = 0; j < num_cols; j++) {
        size_t global_row         = (inverted_indices + offset)[i];
        cluster_data_matrix(i, j) = dataset(global_row, j);
      }
    }

    build_and_merge<T, IdxT>(res,
                             params,
                             num_data_in_cluster,
                             graph_degree,
                             extended_graph_degree,
                             cluster_data_matrix.data_handle(),
                             cluster_data_indices + offset,
                             int_graph,
                             inverted_indices + offset,
                             global_indices_h,
                             global_distances_h,
                             batch_indices_h,
                             batch_indices_d,
                             batch_distances_d,
                             nnd);
    nnd.reset(res);
    RAFT_LOG_INFO("Finished cluster %lu", cluster_id);
    print_memory_info();
  }
  RAFT_LOG_INFO("Finished all clusters");
  print_memory_info();
}

template <typename T, typename IdxT = uint32_t>
void cluster_nnd(raft::resources const& res,
                 const index_params& params,
                 size_t graph_degree,
                 size_t extended_graph_degree,
                 size_t max_cluster_size,
                 raft::device_matrix_view<const T, int64_t> dataset,
                 IdxT* offsets,
                 IdxT* cluster_size,
                 IdxT* cluster_data_indices,
                 int* int_graph,
                 IdxT* inverted_indices,
                 IdxT* global_indices_h,
                 float* global_distances_h,
                 IdxT* batch_indices_h,
                 IdxT* batch_indices_d,
                 float* batch_distances_d,
                 const BuildConfig& build_config)
{
  size_t num_rows = dataset.extent(0);
  size_t num_cols = dataset.extent(1);

  GNND<const T, int> nnd(res, build_config);

  auto cluster_data_matrix =
    raft::make_device_matrix<T, int64_t, raft::row_major>(res, max_cluster_size, num_cols);

  for (size_t cluster_id = 0; cluster_id < params.n_clusters; cluster_id++) {
    RAFT_LOG_DEBUG(
      "# Data on device. Running clusters: %lu / %lu", cluster_id + 1, params.n_clusters);
    size_t num_data_in_cluster = cluster_size[cluster_id];
    size_t offset              = offsets[cluster_id];

    auto cluster_data_view = raft::make_device_matrix_view<T, IdxT>(
      cluster_data_matrix.data_handle(), num_data_in_cluster, num_cols);
    auto cluster_data_indices_view = raft::make_device_vector_view<const IdxT, IdxT>(
      cluster_data_indices + offset, num_data_in_cluster);

    auto dataset_IdxT =
      raft::make_device_matrix_view<const T, IdxT>(dataset.data_handle(), num_rows, num_cols);
    raft::matrix::gather(res, dataset_IdxT, cluster_data_indices_view, cluster_data_view);

    build_and_merge<T, IdxT>(res,
                             params,
                             num_data_in_cluster,
                             graph_degree,
                             extended_graph_degree,
                             cluster_data_view.data_handle(),
                             cluster_data_indices + offset,
                             int_graph,
                             inverted_indices + offset,
                             global_indices_h,
                             global_distances_h,
                             batch_indices_h,
                             batch_indices_d,
                             batch_distances_d,
                             nnd);
    nnd.reset(res);
  }
}

template <typename T,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<std::experimental::default_accessor<float>, raft::memory_type::host>>
void batch_build(raft::resources const& res,
                 const index_params& params,
                 raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
                 index<IdxT>& global_idx)
{
  size_t graph_degree        = params.graph_degree;
  size_t intermediate_degree = params.intermediate_graph_degree;

  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));

  auto centroids =
    raft::make_device_matrix<T, IdxT, raft::row_major>(res, params.n_clusters, num_cols);
  get_balanced_kmeans_centroids<T, IdxT>(res, params.metric, dataset, centroids.view());

  size_t k                    = 2;
  auto global_nearest_cluster = raft::make_host_matrix<IdxT, IdxT, raft::row_major>(num_rows, k);
  get_global_nearest_k<T, IdxT>(res,
    k,
    num_rows,
    params.n_clusters,
    dataset.data_handle(),
    global_nearest_cluster.view(),
    centroids.view(),
    params.metric);

  auto inverted_indices = raft::make_host_vector<IdxT, IdxT, raft::row_major>(num_rows * k);
  auto cluster_size     = raft::make_host_vector<IdxT, IdxT, raft::row_major>(params.n_clusters);
  auto offset           = raft::make_host_vector<IdxT, IdxT, raft::row_major>(params.n_clusters);

  size_t max_cluster_size, min_cluster_size;
  get_inverted_indices(res,
                       params.n_clusters,
                       max_cluster_size,
                       min_cluster_size,
                       global_nearest_cluster.view(),
                       inverted_indices.view(),
                       cluster_size.view(),
                       offset.view());

  if (intermediate_degree >= min_cluster_size) {
    RAFT_LOG_WARN(
      "Intermediate graph degree cannot be larger than minimum cluster size, reducing it to %lu",
      dataset.extent(0));
    intermediate_degree = min_cluster_size - 1;
  }
  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }

  size_t extended_graph_degree =
    align32::roundUp(static_cast<size_t>(graph_degree * (graph_degree <= 32 ? 1.0 : 1.3)));
  size_t extended_intermediate_degree = align32::roundUp(
    static_cast<size_t>(intermediate_degree * (intermediate_degree <= 32 ? 1.0 : 1.3)));

  auto int_graph = raft::make_host_matrix<int, int64_t, raft::row_major>(
    max_cluster_size, static_cast<int64_t>(extended_graph_degree));

  BuildConfig build_config{.max_dataset_size      = max_cluster_size,
                          .dataset_dim           = num_cols,
                          .node_degree           = extended_graph_degree,
                          .internal_node_degree  = extended_intermediate_degree,
                          .max_iterations        = params.max_iterations,
                          .termination_threshold = params.termination_threshold};
                          // .output_graph_degree   = graph_degree};
  RAFT_LOG_INFO("[batch_build] max_cluster_size: %lu", max_cluster_size);
  RAFT_LOG_INFO("[batch_build] min_cluster_size: %lu", min_cluster_size);
  
  auto global_indices_h   = raft::make_managed_matrix<IdxT, int64_t>(res, num_rows, graph_degree);
  auto global_distances_h = raft::make_managed_matrix<float, int64_t>(res, num_rows, graph_degree);

  std::fill(global_indices_h.data_handle(),
            global_indices_h.data_handle() + num_rows * graph_degree,
            std::numeric_limits<IdxT>::max());
  std::fill(global_distances_h.data_handle(),
            global_distances_h.data_handle() + num_rows * graph_degree,
            std::numeric_limits<float>::max());
  
  auto batch_indices_h =
    raft::make_host_matrix<IdxT, int64_t, raft::row_major>(max_cluster_size, graph_degree);
  auto batch_indices_d =
    raft::make_device_matrix<IdxT, int64_t, raft::row_major>(res, max_cluster_size, graph_degree);
  auto batch_distances_d =
    raft::make_device_matrix<float, int64_t, raft::row_major>(res, max_cluster_size, graph_degree);

  auto cluster_data_indices = raft::make_device_vector<IdxT, IdxT>(res, num_rows * k);
  raft::copy(cluster_data_indices.data_handle(),
             inverted_indices.data_handle(),
             num_rows * k,
             raft::resource::get_cuda_stream(res));
  
  RAFT_LOG_INFO("Building and merging clusters");
  print_memory_info();
  cluster_nnd<T, IdxT>(res,
                      params,
                      graph_degree,
                      extended_graph_degree,
                      max_cluster_size,
                      dataset,
                      offset.data_handle(),
                      cluster_size.data_handle(),
                      cluster_data_indices.data_handle(),
                      int_graph.data_handle(),
                      inverted_indices.data_handle(),
                      global_indices_h.data_handle(),
                      global_distances_h.data_handle(),
                      batch_indices_h.data_handle(),
                      batch_indices_d.data_handle(),
                      batch_distances_d.data_handle(),
                      build_config);
  
  cudaStreamSynchronize(raft::resource::get_cuda_stream(res));
  RAFT_LOG_INFO("Finished building and merging clusters");
  print_memory_info();
  
  raft::copy(global_idx.graph().data_handle(),
             global_indices_h.data_handle(),
             num_rows * graph_degree,
             raft::resource::get_cuda_stream(res));
  // if (params.return_distances && global_idx.distances().has_value()) {
  //   raft::copy(global_idx.distances().value().data_handle(),
  //              global_distances_h.data_handle(),
  //              num_rows * graph_degree,
  //              raft::resource::get_cuda_stream(res));
  // }

}

}  // namespace ffanns::neighbors::nn_descent::detail::experimental