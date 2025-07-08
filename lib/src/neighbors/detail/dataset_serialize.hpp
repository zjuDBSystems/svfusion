#pragma once

#include "ffanns/neighbors/common.hpp"

#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <raft/core/logger-ext.hpp>

#include <cuda_fp16.h>

#include <fstream>
#include <memory>

namespace ffanns::neighbors::detail {

using dataset_instance_tag                              = uint32_t;
constexpr dataset_instance_tag kSerializeEmptyDataset   = 1;
constexpr dataset_instance_tag kSerializeStridedDataset = 2;
// constexpr dataset_instance_tag kSerializeVPQDataset     = 3;

template <typename IdxT>
void serialize(const raft::resources& res, std::ostream& os, const empty_dataset<IdxT>& dataset)
{
  raft::serialize_scalar(res, os, dataset.suggested_dim);
}

template <typename DataT, typename IdxT>
void serialize(const raft::resources& res,
               std::ostream& os,
               const strided_dataset<DataT, IdxT>& dataset)
{
  auto n_rows = dataset.n_rows();
  auto dim    = dataset.dim();
  auto stride = dataset.stride();
  raft::serialize_scalar(res, os, n_rows);
  raft::serialize_scalar(res, os, dim);
  raft::serialize_scalar(res, os, stride);
  // Remove padding before saving the dataset
  // Assume we only need to serialize dataset on device
  auto src = dataset.d_view();
  auto dst = raft::make_host_matrix<DataT, IdxT>(n_rows, dim);
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(dst.data_handle(),
                                  sizeof(DataT) * dim,
                                  src.data_handle(),
                                  sizeof(DataT) * stride,
                                  sizeof(DataT) * dim,
                                  n_rows,
                                  cudaMemcpyDefault,
                                  raft::resource::get_cuda_stream(res)));
  raft::resource::sync_stream(res);
  raft::serialize_mdspan(res, os, dst.view());
}

template <typename IdxT>
void serialize(const raft::resources& res, std::ostream& os, const dataset<IdxT>& dataset)
{
  if (auto x = dynamic_cast<const empty_dataset<IdxT>*>(&dataset); x != nullptr) {
    raft::serialize_scalar(res, os, kSerializeEmptyDataset);
    return serialize(res, os, *x);
  }
  if (auto x = dynamic_cast<const strided_dataset<float, IdxT>*>(&dataset); x != nullptr) {
    raft::serialize_scalar(res, os, kSerializeStridedDataset);
    raft::serialize_scalar(res, os, CUDA_R_32F);
    return serialize(res, os, *x);
  }
}

template <typename IdxT>
auto deserialize_empty(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<empty_dataset<IdxT>>
{
  auto suggested_dim = raft::deserialize_scalar<uint32_t>(res, is);
  return std::make_unique<empty_dataset<IdxT>>(suggested_dim);
}

template <typename DataT, typename IdxT>
auto deserialize_strided(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<strided_dataset<DataT, IdxT>>
{
  auto n_rows     = raft::deserialize_scalar<IdxT>(res, is);
  auto dim        = raft::deserialize_scalar<uint32_t>(res, is);
  auto stride     = raft::deserialize_scalar<uint32_t>(res, is);
  auto host_array = raft::make_host_matrix<DataT, IdxT>(n_rows, dim);
  raft::deserialize_mdspan(res, is, host_array.view());
  return make_strided_dataset(res, std::move(host_array), stride);
}

template <typename IdxT>
auto deserialize_dataset(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<dataset<IdxT>>
{
  switch (raft::deserialize_scalar<dataset_instance_tag>(res, is)) {
    case kSerializeEmptyDataset: return deserialize_empty<IdxT>(res, is);
    case kSerializeStridedDataset:
      switch (raft::deserialize_scalar<cudaDataType_t>(res, is)) {
        case CUDA_R_32F: return deserialize_strided<float, IdxT>(res, is);
        default: break;
      }
    default: break;
  }
  RAFT_FAIL("Failed to deserialize dataset: unsupported combination of instance tags.");
}

} // namespace ffanns::neighbors::detail