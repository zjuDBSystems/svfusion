#pragma once

#include "detail/cagra/add_nodes.cuh"
#include "detail/cagra/cagra_build.cuh"
 
#include "detail/cagra/cagra_search.cuh"
#include "detail/cagra/graph_core.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resources.hpp>

#include "ffanns/distance/distance.hpp"
#include "ffanns/neighbors/cagra.hpp"
#include "ffanns/neighbors/common.hpp"

#include <rmm/cuda_stream_view.hpp>

namespace ffanns::neighbors::cagra {

template <typename T,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
index<T, IdxT> build(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> index_graph, 
  raft::device_matrix<T, int64_t>& device_dataset_ref,  
  raft::device_matrix<IdxT, int64_t>& device_graph_ref,  
  std::shared_ptr<ffanns::core::bitset<std::uint32_t, int64_t>> delete_bitset,
  std::shared_ptr<rmm::device_uvector<IdxT>> tag_to_id,
  IdxT start_id, IdxT end_id)
{
  return ffanns::neighbors::cagra::detail::build<T, IdxT, Accessor>(res, params, dataset, index_graph,
    device_dataset_ref, device_graph_ref, delete_bitset, tag_to_id, start_id, end_id);
}

template <typename T, typename IdxT, typename CagraSampleFilterT>
void search_with_filtering(raft::resources const& res,
                           const search_params& params,
                           index<T, IdxT>& idx,
                           raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
                           raft::host_matrix_view<const T, int64_t, raft::row_major> host_queries,
                           raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
                           raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                           CagraSampleFilterT sample_filter = CagraSampleFilterT(),
                           bool external_flag = false)
{
  RAFT_EXPECTS(
    queries.extent(0) == neighbors.extent(0) && queries.extent(0) == distances.extent(0) && queries.extent(0) == host_queries.extent(0),
    "Number of rows in output neighbors and distances matrices must equal the number of queries.");

  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1),
               "Number of columns in output neighbors and distances matrices must equal k");
  RAFT_EXPECTS(queries.extent(1) == idx.dim(),
               "Number of query dimensions should equal number of dimensions in the index.");
  
  using internal_IdxT   = typename std::make_unsigned<IdxT>::type;
  auto queries_internal = raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
    queries.data_handle(), queries.extent(0), queries.extent(1));
  auto host_queries_internal = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
    host_queries.data_handle(), host_queries.extent(0), host_queries.extent(1));
  auto neighbors_internal = raft::make_device_matrix_view<internal_IdxT, int64_t, raft::row_major>(
    reinterpret_cast<internal_IdxT*>(neighbors.data_handle()),
    neighbors.extent(0),
    neighbors.extent(1));
  auto distances_internal = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
    distances.data_handle(), distances.extent(0), distances.extent(1));

  return cagra::detail::search_main<T, internal_IdxT, CagraSampleFilterT, IdxT>(
    res, params, idx, queries_internal, host_queries_internal, neighbors_internal, distances_internal, sample_filter, external_flag);
}

template <typename T, typename IdxT>
void search(raft::resources const& res,
            const search_params& params,
            index<T, IdxT>& idx,
            raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
            raft::host_matrix_view<const T, int64_t, raft::row_major> host_queries,
            raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const ffanns::neighbors::filtering::base_filter& sample_filter_ref,
            bool external_flag = false)
{
  try {
    using none_filter_type  = ffanns::neighbors::filtering::none_sample_filter;
    auto& sample_filter     = dynamic_cast<const none_filter_type&>(sample_filter_ref);
    auto sample_filter_copy = sample_filter;
    return search_with_filtering<T, IdxT, none_filter_type>(
      res, params, idx, queries, host_queries, neighbors, distances, sample_filter_copy, external_flag);
    return;
  } catch (const std::bad_cast&) {
    // RAFT_FAIL("Now only suupport none-filter");
  }

  try {
    auto& sample_filter =
      dynamic_cast<const ffanns::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(
        sample_filter_ref);
    auto sample_filter_copy = sample_filter;
    return search_with_filtering<T, IdxT, decltype(sample_filter_copy)>(
      res, params, idx, queries, host_queries, neighbors, distances, sample_filter_copy, external_flag);
  } catch (const std::bad_cast&) {
    RAFT_FAIL("Unsupported sample filter type");
  }
}

// Note: remove class Accessor in template
template <typename T, typename IdxT>
void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::host_matrix_view<const T, int64_t, raft::row_major> additional_dataset,
  ffanns::neighbors::cagra::index<T, IdxT>& index,
  std::optional<raft::host_matrix_view<T, int64_t, raft::layout_stride>> ndv,
  std::optional<raft::device_matrix_view<T, int64_t, raft::layout_stride>> nddv,
  std::optional<raft::host_matrix_view<IdxT, int64_t, raft::layout_stride>> ngv,
  std::optional<raft::device_matrix_view<IdxT, int64_t>> ndgv,
  IdxT start_id, IdxT end_id)
{
  cagra::extend_core<T, IdxT>(handle, params, additional_dataset, index, ndv, nddv, ngv, ndgv, start_id, end_id);
}

}