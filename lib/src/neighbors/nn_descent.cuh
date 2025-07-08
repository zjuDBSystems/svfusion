#pragma once

#include "detail/nn_descent.cuh"
#include "detail/nn_descent_batch.cuh"
#include "ffanns/neighbors/nn_descent.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>

namespace ffanns::neighbors::nn_descent {

template <typename T, typename IdxT = uint32_t>
void build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
           index<IdxT>& idx)
{
  if (params.n_clusters > 1) {
    if constexpr (std::is_same_v<T, float>) {
      RAFT_LOG_INFO("[nn_descent::batch_buid] start batch build!!!");
      detail::experimental::batch_build<T, IdxT>(res, params, dataset, idx);
    } else {
      RAFT_FAIL("Batched nn-descent is only supported for float precision");
    }
  } else {
    ffanns::neighbors::nn_descent::detail::build<T, IdxT>(res, params, dataset, idx);
  }
}

}