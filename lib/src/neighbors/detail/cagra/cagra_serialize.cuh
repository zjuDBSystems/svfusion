#pragma once

#include "ffanns/neighbors/cagra.hpp"
#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger-ext.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/serialize.hpp>

// #include "../../../core/nvtx.hpp"
#include "../dataset_serialize.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <type_traits>

namespace ffanns::neighbors::cagra::detail {

static const std::string RAFT_NAME = "raft";

constexpr int serialization_version = 4;

template <typename T, typename IdxT>
void serialize(raft::resources const& res,
               std::ostream& os,
               const index<T, IdxT>& index_,
               bool include_dataset)
{
  // raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("cagra::serialize");

  RAFT_LOG_INFO(
    "Saving CAGRA index, size %zu, dim %u", static_cast<size_t>(index_.size()), index_.dim());

  std::string dtype_string = raft::detail::numpy_serializer::get_numpy_dtype<T>().to_string();
  dtype_string.resize(4);
  os << dtype_string;

  raft::serialize_scalar(res, os, serialization_version);
  raft::serialize_scalar(res, os, index_.size());
  raft::serialize_scalar(res, os, index_.dim());
  raft::serialize_scalar(res, os, index_.graph_degree());
  raft::serialize_scalar(res, os, index_.metric());

  raft::serialize_mdspan(res, os, index_.graph());

  auto hd_mapper_ptr = index_.hd_mapper_ptr();
  // hd_mapper_ptr->serialize(res, os);

  // TODO: host_in_edges() is not a const method
  // assert (index_.size() == index_.host_in_edges().size());
  // raft::serialize_mdspan(res, os, index_.host_in_edges());
  
  include_dataset &= (index_.data().n_rows() > 0);

  raft::serialize_scalar(res, os, include_dataset);
  if (include_dataset) {
    RAFT_LOG_INFO("Saving CAGRA index with dataset on device");
    // neighbors::detail::serialize(res, os, index_.data());
    neighbors::detail::serialize(res, os, index_.d_data());
  } else {
    RAFT_LOG_INFO("Saving CAGRA index WITHOUT dataset");
  }
}

template <typename T, typename IdxT>
void serialize(raft::resources const& res,
               const std::string& filename,
               const index<T, IdxT>& index_,
               bool include_dataset)
{
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  detail::serialize(res, of, index_, include_dataset);

  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
}

template <typename T, typename IdxT>
void deserialize(raft::resources const& res, std::istream& is, index<T, IdxT>* index_)
{
  // raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("cagra::deserialize");

  char dtype_string[4];
  is.read(dtype_string, 4);

  auto ver = raft::deserialize_scalar<int>(res, is);
  if (ver != serialization_version) {
    RAFT_FAIL("serialization version mismatch, expected %d, got %d ", serialization_version, ver);
  }
  auto n_rows       = raft::deserialize_scalar<IdxT>(res, is);
  auto dim          = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto graph_degree = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto metric       = raft::deserialize_scalar<ffanns::distance::DistanceType>(res, is);

  auto graph = raft::make_host_matrix<IdxT, int64_t>(n_rows, graph_degree);
  deserialize_mdspan(res, is, graph.view());

  *index_ = index<T, IdxT>(res, metric);
  index_->update_graph(res, raft::make_const_mdspan(graph.view()));

  auto in_edges = raft::make_host_vector<int, int64_t>(n_rows);
  deserialize_mdspan(res, is, in_edges.view());
  index_->own_in_edges(in_edges);

  auto hd_mapper_ptr = index_->hd_mapper_ptr();
  // hd_mapper_ptr->deserialize_from(res, is);
  bool has_dataset = raft::deserialize_scalar<bool>(res, is);
  // TODO: Dataset deserialization
  // if (has_dataset) {
  //   index_->update_dataset(res, ffanns::neighbors::detail::deserialize_dataset<int64_t>(res, is));
  // }
}

template <typename T, typename IdxT>
void deserialize(raft::resources const& res, const std::string& filename, index<T, IdxT>* index_)
{
  std::ifstream is(filename, std::ios::in | std::ios::binary);

  if (!is) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  detail::deserialize<T, IdxT>(res, is, index_);

  is.close();
}

} // namespace ffanns::neighbors::cagra::detail