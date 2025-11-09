#pragma once

// #include "../../../core/nvtx.hpp"
#include <raft/common/nvtx.hpp>

#include "factory.cuh"
#include "sample_filter_utils.cuh"
#include "search_plan.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include "ffanns/distance/distance.hpp"

#include "ffanns/neighbors/cagra.hpp"

// TODO: Fix these when ivf methods are moved over
// #include "../../ivf_common.cuh"
// #include "../../ivf_pq/ivf_pq_search.cuh"
#include <raft/neighbors/detail/ivf_common.cuh>
#include "ffanns/neighbors/common.hpp"

// TODO: This shouldn't be calling spatial/knn apis
#include "../ann_utils.cuh"

#include <rmm/cuda_stream_view.hpp>

namespace ffanns::neighbors::cagra::detail {

template <typename IndexT>
__global__ void map_neighbors_kernel(IndexT* neighbors, 
                                       const IndexT* d_mapping,
                                       uint32_t num_queries,
                                       uint32_t num_neighbors)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_queries)
    {
        uint32_t offset = row * num_neighbors;
        // neighbors[idx] holds a tag, map it to external id using d_mapping.
        for (uint32_t j = 0; j < num_neighbors; j++)
        {
            uint32_t idx = offset + j;
            IndexT tag = neighbors[idx];
            neighbors[idx] = d_mapping[tag];
        }
    }
}

template <typename DataT, typename IndexT, typename DistanceT, typename CagraSampleFilterT>
void search_main_core(raft::resources const& res,
                      search_params params,
                      const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
                      raft::host_matrix_view<const IndexT, int64_t> graph,
                      raft::device_matrix_view<IndexT, int64_t, raft::row_major> d_graph,
                      raft::device_matrix_view<const DataT, int64_t, raft::row_major> queries,
                      raft::host_matrix_view<const DataT, int64_t, raft::row_major> host_queries,
                      raft::device_matrix_view<IndexT, int64_t, raft::row_major> neighbors,
                      raft::device_matrix_view<DistanceT, int64_t, raft::row_major> distances,
                      host_device_mapper* hd_mapper,
                      graph_hd_mapper* graph_hd_mapper,
                      CagraSampleFilterT sample_filter = CagraSampleFilterT(),
                      int* in_edges = nullptr,
                      float* miss_rate = nullptr,
                      ffanns::neighbors::cagra::search_context<DataT, IndexT>* search_ctx = nullptr)
{
  // RAFT_LOG_INFO("[search_main_core] Start search_main_core");
  RAFT_LOG_DEBUG("# dataset size = %lu, dim = %lu\n",
                 static_cast<size_t>(graph.extent(0)),
                 static_cast<size_t>(queries.extent(1)));
  RAFT_LOG_DEBUG("# query size = %lu, dim = %lu\n",
                 static_cast<size_t>(queries.extent(0)),
                 static_cast<size_t>(queries.extent(1)));
  const uint32_t topk = neighbors.extent(1);

  cudaDeviceProp deviceProp = raft::resource::get_device_properties(res);
  if (params.max_queries == 0) {
    params.max_queries = std::min<size_t>(queries.extent(0), deviceProp.maxGridSize[1]);
  }

  // TODO: add NVTX
  using CagraSampleFilterT_s = typename CagraSampleFilterT_Selector<CagraSampleFilterT>::type; 
  std::unique_ptr<search_plan_impl<DataT, IndexT, DistanceT, CagraSampleFilterT_s>> plan =
    factory<DataT, IndexT, DistanceT, CagraSampleFilterT_s>::create(
      res, params, dataset_desc, queries.extent(1), graph.extent(0), graph.extent(1), topk);

  plan->check(topk);

  const uint32_t max_queries = plan->max_queries;
  const uint32_t query_dim   = queries.extent(1);

  for (unsigned qid = 0; qid < queries.extent(0); qid += max_queries) {
    const uint32_t n_queries = std::min<std::size_t>(max_queries, queries.extent(0) - qid);
    auto _topk_indices_ptr   = reinterpret_cast<IndexT*>(neighbors.data_handle()) + (topk * qid);
    auto _topk_distances_ptr = distances.data_handle() + (topk * qid);
    // todo(tfeher): one could keep distances optional and pass nullptr
    const auto* _query_ptr = queries.data_handle() + (query_dim * qid);
    const auto* _host_query_ptr = host_queries.data_handle() + (query_dim * qid);
    const auto* _seed_ptr =
      plan->num_seeds > 0
        ? reinterpret_cast<const IndexT*>(plan->dev_seed.data()) + (plan->num_seeds * qid)
        : nullptr;
    uint32_t* _num_executed_iterations = nullptr;

    (*plan)(res,
            graph,
            d_graph,
            _topk_indices_ptr,
            _topk_distances_ptr,
            _query_ptr,
            _host_query_ptr,
            n_queries,
            _seed_ptr,
            _num_executed_iterations,
            topk,
            set_offset(sample_filter, qid),
            hd_mapper,
            graph_hd_mapper,
            in_edges,
            miss_rate,
            search_ctx);
  }
}

template <typename T,
          typename InternalIdxT,
          typename CagraSampleFilterT,
          typename IdxT      = uint32_t,
          typename DistanceT = float>
void search_main(raft::resources const& res,
                 search_params params,
                 index<T, IdxT>& index,
                 raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
                 raft::host_matrix_view<const T, int64_t, raft::row_major> host_queries,
                 raft::device_matrix_view<InternalIdxT, int64_t, raft::row_major> neighbors,
                 raft::device_matrix_view<DistanceT, int64_t, raft::row_major> distances,
                 CagraSampleFilterT sample_filter = CagraSampleFilterT(),
                 bool external_flag = false,
                 ffanns::neighbors::cagra::search_context<T, InternalIdxT>* search_ctx = nullptr)
{
  auto stream         = raft::resource::get_cuda_stream(res);

  // TODO： Should have two copies, this graph needs to be on device
  const auto& graph   = index.graph();
  
  auto graph_internal = raft::make_host_matrix_view<const InternalIdxT, int64_t, raft::row_major>(
    reinterpret_cast<const InternalIdxT*>(graph.data_handle()), graph.extent(0), graph.extent(1));
  auto d_graph_internal = index.d_graph();

  // n_rows has the same type as the dataset index (the array extents type)
  using ds_idx_type = decltype(index.data().n_rows());

  auto* strided_dset = static_cast<const strided_dataset<T, ds_idx_type>*>(&index.data());
  assert(strided_dset != nullptr && "index.data() must be of type strided_dataset.");
  auto* strided_device_dset = static_cast<const strided_dataset<T, ds_idx_type>*>(&index.d_data());
  assert(strided_device_dset != nullptr && "index.d_data() must be of type strided_dataset.");
  auto& desc = dataset_descriptor_init_with_cache<T, InternalIdxT, DistanceT>(
    res, params, *strided_dset, *strided_device_dset, index.metric());
  
  auto hd_mapper_ptr = index.hd_mapper_ptr();
  auto graph_hd_mapper_ptr = index.get_graph_hd_mapper_ptr();
  auto d_in_edges = index.d_in_edges();
  search_main_core<T, InternalIdxT, DistanceT, CagraSampleFilterT>(
    res, params, desc, graph_internal, d_graph_internal, queries, host_queries, neighbors, distances,
    hd_mapper_ptr.get(), graph_hd_mapper_ptr.get(), sample_filter, d_in_edges.data_handle(), &hd_mapper_ptr->miss_rate, search_ctx);

  // tag_to_id
  if (external_flag) {
    // RAFT_LOG_INFO("[search_main] external_flag is set, mapping neighbors from internal to external ids!!!");
    // map neighbors from internal to external ids
    auto num_queries = neighbors.extent(0);
    auto num_neighbors = neighbors.extent(1);
    const auto block_size                = 256u;
    auto numBlocks = (num_queries + block_size - 1) / block_size;
    const InternalIdxT* d_mapping = thrust::raw_pointer_cast(index.get_tag_to_id().data());
    map_neighbors_kernel<<<numBlocks, block_size, 0, stream>>>(neighbors.data_handle(), d_mapping, num_queries, num_neighbors);
    cudaStreamSynchronize(stream);
  }

  hd_mapper_ptr->decay_recent_access(0.2668, res);
  
  // static_assert(std::is_same_v<DistanceT, float>,
  //               "only float distances are supported at the moment");
  // float* dist_out          = distances.data_handle();
  // const DistanceT* dist_in = distances.data_handle();
  // // We're converting the data from T to DistanceT during distance computation
  // // and divide the values by kDivisor. Here we restore the original scale.
  // constexpr float kScale = ffanns::spatial::knn::detail::utils::config<T>::kDivisor /
  //                          ffanns::spatial::knn::detail::utils::config<DistanceT>::kDivisor;
  // // Todo: integrate ivf::detail::postprocess_distances
  // auto raft_metric = static_cast<raft::distance::DistanceType>(index.metric());
  // raft::neighbors::ivf::detail::postprocess_distances(dist_out,
  //                                                     dist_in,
  //                                                     raft_metric,
  //                                                     distances.extent(0),
  //                                                     distances.extent(1),
  //                                                     kScale,
  //                                                     true,
  //                                                     raft::resource::get_cuda_stream(res));
}

} // namespace ffanns::neighbors::cagra::detail