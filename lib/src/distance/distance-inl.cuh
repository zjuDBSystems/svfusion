#pragma once

#include "detail/distance.cuh"
#include "ffanns/distance/distance.hpp"
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/device_uvector.hpp>

#include <type_traits>

namespace ffanns {
namespace distance {

template <ffanns::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT = int>
size_t getWorkspaceSize(const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k)
{
  return detail::getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT>(x, y, m, n, k);
}

template <ffanns::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT = int,
          typename layout>
size_t getWorkspaceSize(raft::device_matrix_view<DataT, IdxT, layout> const& x,
                        raft::device_matrix_view<DataT, IdxT, layout> const& y)
{
  RAFT_EXPECTS(x.extent(1) == y.extent(1), "Number of columns must be equal.");

  return getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT>(
    x.data_handle(), y.data_handle(), x.extent(0), y.extent(0), x.extent(1));
}

template <typename Type, typename IdxT = int, typename DistT = Type>
void pairwise_distance(raft::resources const& handle,
                       const Type* x,
                       const Type* y,
                       DistT* dist,
                       IdxT m,
                       IdxT n,
                       IdxT k,
                       rmm::device_uvector<char>& workspace,
                       ffanns::distance::DistanceType metric,
                       bool isRowMajor  = true,
                       DistT metric_arg = 2.0f)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  auto dispatch = [&](auto distance_type) {
    auto worksize = getWorkspaceSize<distance_type(), Type, DistT, DistT, IdxT>(x, y, m, n, k);
    workspace.resize(worksize, stream);
    detail::distance<distance_type(), Type, DistT, DistT, IdxT>(
      handle, x, y, dist, m, n, k, workspace.data(), worksize, isRowMajor, metric_arg);
  };

  switch (metric) {
    case DistanceType::L2Expanded:
      dispatch(std::integral_constant<DistanceType, DistanceType::L2Expanded>{});
      break;
    case ffanns::distance::DistanceType::InnerProduct:
      dispatch(std::integral_constant<DistanceType, DistanceType::InnerProduct>{});
      break;
    default: THROW("Unknown or unsupported distance metric '%d'!", (int)metric);
  };
}

template <typename Type, typename IdxT = int, typename DistT = Type>
void pairwise_distance(raft::resources const& handle,
                       const Type* x,
                       const Type* y,
                       DistT* dist,
                       IdxT m,
                       IdxT n,
                       IdxT k,
                       ffanns::distance::DistanceType metric,
                       bool isRowMajor  = true,
                       DistT metric_arg = 2.0f)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<char> workspace(0, stream);
  pairwise_distance<Type, IdxT, DistT>(
    handle, x, y, dist, m, n, k, workspace, metric, isRowMajor, metric_arg);
}

template <typename Type,
          typename layout = raft::layout_c_contiguous,
          typename IdxT   = int,
          typename DistT  = Type>
void pairwise_distance(raft::resources const& handle,
                       raft::device_matrix_view<const Type, IdxT, layout> const x,
                       raft::device_matrix_view<const Type, IdxT, layout> const y,
                       raft::device_matrix_view<DistT, IdxT, layout> dist,
                       ffanns::distance::DistanceType metric,
                       DistT metric_arg = DistT(2.0f))
{
  RAFT_EXPECTS(x.extent(1) == y.extent(1), "Number of columns must be equal.");
  RAFT_EXPECTS(dist.extent(0) == x.extent(0),
               "Number of rows in output must be equal to "
               "number of rows in X");
  RAFT_EXPECTS(dist.extent(1) == y.extent(0),
               "Number of columns in output must be equal to "
               "number of rows in Y");

  RAFT_EXPECTS(x.is_exhaustive(), "Input x must be contiguous.");
  RAFT_EXPECTS(y.is_exhaustive(), "Input y must be contiguous.");
  RAFT_EXPECTS(dist.is_exhaustive(), "Output must be contiguous.");

  constexpr auto rowmajor = std::is_same_v<layout, raft::layout_c_contiguous>;

  auto stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<char> workspace(0, stream);

  pairwise_distance(handle,
                    x.data_handle(),
                    y.data_handle(),
                    dist.data_handle(),
                    x.extent(0),
                    y.extent(0),
                    x.extent(1),
                    metric,
                    rowmajor,
                    metric_arg);
}

};  // namespace distance
};  // namespace ffanns
