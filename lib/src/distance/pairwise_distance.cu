#include "distance.cuh"
#include <cstdint>
#include "ffanns/distance/distance.hpp"
#include <raft/core/device_mdspan.hpp>

namespace ffanns::distance {

/**
 * @defgroup pairwise_distance_runtime Pairwise Distances Runtime API
 * @{
 */
void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous> const x,
  raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous> const y,
  raft::device_matrix_view<float, std::int64_t, raft::layout_c_contiguous> dist,
  ffanns::distance::DistanceType metric,
  float metric_arg)
{
  auto x_v = raft::make_device_matrix_view<const float, int, raft::layout_c_contiguous>(
    x.data_handle(), x.extent(0), x.extent(1));
  auto y_v = raft::make_device_matrix_view<const float, int, raft::layout_c_contiguous>(
    y.data_handle(), y.extent(0), y.extent(1));
  auto d_v = raft::make_device_matrix_view<float, int, raft::layout_c_contiguous>(
    dist.data_handle(), dist.extent(0), dist.extent(1));
  pairwise_distance<float, raft::layout_c_contiguous, int>(
    handle, x_v, y_v, d_v, metric, metric_arg);
}

}  // namespace ffanns::distance