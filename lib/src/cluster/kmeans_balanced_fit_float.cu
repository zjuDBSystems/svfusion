/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

// clang-format off
#include "kmeans_balanced.cuh"
#include "../neighbors/detail/ann_utils.cuh"
#include <raft/core/resources.hpp>
// clang-format on

namespace ffanns::cluster::kmeans {

void fit(const raft::resources& handle,
         ffanns::cluster::kmeans::balanced_params const& params,
         raft::device_matrix_view<const float, int> X,
         raft::device_matrix_view<float, int> centroids)
{
  ffanns::cluster::kmeans_balanced::fit(
    handle, params, X, centroids, ffanns::spatial::knn::detail::utils::mapping<float>{});
}
}  // ffanns::cluster::kmeans
