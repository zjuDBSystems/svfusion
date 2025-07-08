#pragma once

// #include "../../distance/distance.cuh"
// #include "../../distance/fused_distance_nn.cuh"
#include "ffanns/cluster/kmeans.hpp"
#include "ffanns/distance/distance.hpp"

#include <raft/core/cudart_utils.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/random/permute.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <cuda.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <optional>
#include <random>

namespace ffanns::cluster::kmeans::detail {

// Computes the intensity histogram from a sequence of labels
template <typename SampleIteratorT, typename CounterT, typename IndexT>
void countLabels(raft::resources const& handle,
                 SampleIteratorT labels,
                 CounterT* count,
                 IndexT n_samples,
                 IndexT n_clusters,
                 rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  // CUB::DeviceHistogram requires a signed index type
  typedef typename std::make_signed_t<IndexT> CubIndexT;

  CubIndexT num_levels  = n_clusters + 1;
  CubIndexT lower_level = 0;
  CubIndexT upper_level = n_clusters;

  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(nullptr,
                                                    temp_storage_bytes,
                                                    labels,
                                                    count,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    static_cast<CubIndexT>(n_samples),
                                                    stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(workspace.data(),
                                                    temp_storage_bytes,
                                                    labels,
                                                    count,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    static_cast<CubIndexT>(n_samples),
                                                    stream));
}

} // namespace ffanns::cluster::kmeans::detail