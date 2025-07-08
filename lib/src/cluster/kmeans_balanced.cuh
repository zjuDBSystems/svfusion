/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

 #pragma once

 #include "detail/kmeans_balanced.cuh"
 #include <raft/core/mdarray.hpp>
 #include <raft/core/resource/device_memory_resource.hpp>
 #include <raft/util/cuda_utils.cuh>
 
 #include <utility>
 
 namespace ffanns::cluster::kmeans_balanced {
 
 /**
  * @brief Find clusters of balanced sizes with a hierarchical k-means algorithm.
  *
  * This variant of the k-means algorithm first clusters the dataset in mesoclusters, then clusters
  * the subsets associated to each mesocluster into fine clusters, and finally runs a few k-means
  * iterations over the whole dataset and with all the centroids to obtain the final clusters.
  *
  * Each k-means iteration applies expectation-maximization-balancing:
  *  - Balancing: adjust centers for clusters that have a small number of entries. If the size of a
  *    cluster is below a threshold, the center is moved towards a bigger cluster.
  *  - Expectation: predict the labels (i.e find closest cluster centroid to each point)
  *  - Maximization: calculate optimal centroids (i.e find the center of gravity of each cluster)
  *
  * The number of mesoclusters is chosen by rounding the square root of the number of clusters. E.g
  * for 512 clusters, we would have 23 mesoclusters. The number of fine clusters per mesocluster is
  * chosen proportionally to the number of points in each mesocluster.
  *
  * This variant of k-means uses random initialization and a fixed number of iterations, though
  * iterations can be repeated if the balancing step moved the centroids.
  *
  * Additionally, this algorithm supports quantized datasets in arbitrary types but the core part of
  * the algorithm will work with a floating-point type, hence a conversion function can be provided
  * to map the data type to the math type.
  *
  * @tparam DataT Type of the input data.
  * @tparam MathT Type of the centroids and mapped data.
  * @tparam IndexT Type used for indexing.
  * @tparam MappingOpT Type of the mapping function.
  * @param[in]  handle     The raft resources
  * @param[in]  params     Structure containing the hyper-parameters
  * @param[in]  X          Training instances to cluster. The data must be in row-major format.
  *                        [dim = n_samples x n_features]
  * @param[out] centroids  The generated centroids [dim = n_clusters x n_features]
  * @param[in]  mapping_op (optional) Functor to convert from the input datatype to the arithmetic
  *                        datatype. If DataT == MathT, this must be the identity.
  * @param[in]  X_norm        (optional) Dataset's row norms [dim = n_samples]
  */
 template <typename DataT, typename MathT, typename IndexT, typename MappingOpT = raft::identity_op>
 void fit(const raft::resources& handle,
          ffanns::cluster::kmeans::balanced_params const& params,
          raft::device_matrix_view<const DataT, IndexT> X,
          raft::device_matrix_view<MathT, IndexT> centroids,
          MappingOpT mapping_op                                               = raft::identity_op(),
          std::optional<raft::device_vector_view<const MathT, IndexT>> X_norm = std::nullopt)
 {
   RAFT_EXPECTS(X.extent(1) == centroids.extent(1),
                "Number of features in dataset and centroids are different");
   RAFT_EXPECTS(static_cast<uint64_t>(X.extent(0)) * static_cast<uint64_t>(X.extent(1)) <=
                  static_cast<uint64_t>(std::numeric_limits<IndexT>::max()),
                "The chosen index type cannot represent all indices for the given dataset");
   RAFT_EXPECTS(centroids.extent(0) > IndexT{0} && centroids.extent(0) <= X.extent(0),
                "The number of centroids must be strictly positive and cannot exceed the number of "
                "points in the training dataset.");
 
   ffanns::cluster::kmeans::detail::build_hierarchical(
     handle,
     params,
     X.extent(1),
     X.data_handle(),
     X.extent(0),
     centroids.data_handle(),
     centroids.extent(0),
     mapping_op,
     X_norm.has_value() ? X_norm.value().data_handle() : nullptr);
 }
 
 }  // namespace  ffanns::cluster::kmeans_balanced
 