/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 // #include "detail/kernels/rbf_fin_op.cuh"  // rbf_fin_op
#include "distance-inl.cuh"

#define instantiate_ffanns_distance_getWorkspaceSize(DistT, DataT, AccT, OutT, IdxT)             \
  template size_t ffanns::distance::getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT>(            \
    const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k);                                   \
                                                                                               \
  template size_t                                                                              \
  ffanns::distance::getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT, raft::layout_c_contiguous>( \
    raft::device_matrix_view<DataT, IdxT, raft::layout_c_contiguous> const& x,                 \
    raft::device_matrix_view<DataT, IdxT, raft::layout_c_contiguous> const& y)

#define instantiate_ffanns_distance_getWorkspaceSize_by_algo(DistT)                     \
  instantiate_ffanns_distance_getWorkspaceSize(DistT, float, float, float, int);        \
  instantiate_ffanns_distance_getWorkspaceSize(DistT, float, float, float, int64_t)

instantiate_ffanns_distance_getWorkspaceSize_by_algo(ffanns::distance::DistanceType::L2Expanded);
instantiate_ffanns_distance_getWorkspaceSize_by_algo(ffanns::distance::DistanceType::InnerProduct);

#undef instantiate_ffanns_distance_getWorkspaceSize_by_algo
#undef instantiate_ffanns_distance_getWorkspaceSize

#define instantiate_ffanns_distance_pairwise_distance(DataT, IdxT, DistT)                 \
  template void ffanns::distance::pairwise_distance(raft::resources const& handle,        \
                                                  const DataT* x,                       \
                                                  const DataT* y,                       \
                                                  DistT* dist,                          \
                                                  IdxT m,                               \
                                                  IdxT n,                               \
                                                  IdxT k,                               \
                                                  rmm::device_uvector<char>& workspace, \
                                                  ffanns::distance::DistanceType metric,  \
                                                  bool isRowMajor,                      \
                                                  DistT metric_arg)

instantiate_ffanns_distance_pairwise_distance(float, int, float);

#undef instantiate_ffanns_distance_pairwise_distance

// Same, but without workspace
#define instantiate_ffanns_distance_pairwise_distance(DataT, IdxT, DistT)                \
  template void ffanns::distance::pairwise_distance(raft::resources const& handle,       \
                                                  const DataT* x,                      \
                                                  const DataT* y,                      \
                                                  DistT* dist,                         \
                                                  IdxT m,                              \
                                                  IdxT n,                              \
                                                  IdxT k,                              \
                                                  ffanns::distance::DistanceType metric, \
                                                  bool isRowMajor,                     \
                                                  DistT metric_arg)

instantiate_ffanns_distance_pairwise_distance(float, int, float);

#undef instantiate_ffanns_distance_pairwise_distance

#define instantiate_ffanns_distance_pairwise_distance(DataT, layout, IdxT, DistT) \
  template void ffanns::distance::pairwise_distance(                              \
    raft::resources const& handle,                                              \
    raft::device_matrix_view<const DataT, IdxT, layout> const x,                \
    raft::device_matrix_view<const DataT, IdxT, layout> const y,                \
    raft::device_matrix_view<DistT, IdxT, layout> dist,                         \
    ffanns::distance::DistanceType metric,                                        \
    DistT metric_arg)

instantiate_ffanns_distance_pairwise_distance(float, raft::layout_c_contiguous, int, float);

#undef instantiate_ffanns_distance_pairwise_distance