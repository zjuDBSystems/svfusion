#pragma once
#include "ffanns/distance/distance.hpp"
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng_state.hpp>

namespace ffanns::cluster::kmeans {

/** Base structure for parameters that are common to all k-means algorithms */
struct base_params {
    /**
     * Metric to use for distance computation. The supported metrics can vary per algorithm.
     */
    ffanns::distance::DistanceType metric = ffanns::distance::DistanceType::L2Expanded;
};

/**
 * @defgroup kmeans_params k-means hyperparameters
 * @{
 */

/**
 * Simple object to specify hyper-parameters to the kmeans algorithm.
 */
struct params : base_params {
    enum InitMethod {

        /**
         * Sample the centroids using the kmeans++ strategy
         */
        KMeansPlusPlus,

        /**
         * Sample the centroids uniformly at random
         */
        Random,

        /**
         * User provides the array of initial centroids
         */
        Array
    };

    /**
     * The number of clusters to form as well as the number of centroids to generate (default:8).
     */
    int n_clusters = 8;

    /**
     * Method for initialization, defaults to k-means++:
     *  - InitMethod::KMeansPlusPlus (k-means++): Use scalable k-means++ algorithm
     * to select the initial cluster centers.
     *  - InitMethod::Random (random): Choose 'n_clusters' observations (rows) at
     * random from the input data for the initial centroids.
     *  - InitMethod::Array (ndarray): Use 'centroids' as initial cluster centers.
     */
    InitMethod init = KMeansPlusPlus;

    /**
     * Maximum number of iterations of the k-means algorithm for a single run.
     */
    int max_iter = 300;

    /**
     * Relative tolerance with regards to inertia to declare convergence.
     */
    double tol = 1e-4;

    // /**
    //  * verbosity level.
    //  */
    // raft::level_enum verbosity = raft::level_enum::info;

    /**
     * Seed to the random number generator.
     */
    raft::random::RngState rng_state{0};

    /**
     * Number of instance k-means algorithm will be run with different seeds.
     */
    int n_init = 1;

    /**
     * Oversampling factor for use in the k-means|| algorithm
     */
    double oversampling_factor = 2.0;

    // batch_samples and batch_centroids are used to tile 1NN computation which is
    // useful to optimize/control the memory footprint
    // Default tile is [batch_samples x n_clusters] i.e. when batch_centroids is 0
    // then don't tile the centroids
    int batch_samples = 1 << 15;

    /**
     * if 0 then batch_centroids = n_clusters
     */
    int batch_centroids = 0;  //

    bool inertia_check = false;
};

/**
 * Simple object to specify hyper-parameters to the balanced k-means algorithm.
 *
 * The following metrics are currently supported in k-means balanced:
 *  - CosineExpanded
 *  - InnerProduct
 *  - L2Expanded
 *  - L2SqrtExpanded
 */
struct balanced_params : base_params {
    /**
     * Number of training iterations
     */
    uint32_t n_iters = 20;
};

/**
 * @brief Find balanced clusters with k-means algorithm.
 *
 * @code{.cpp}
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int n_features = 15;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, params.n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               centroids);
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[out]  centroids       [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 */
void fit(const raft::resources& handle,
    ffanns::cluster::kmeans::balanced_params const& params,
    raft::device_matrix_view<const float, int> X,
    raft::device_matrix_view<float, int> centroids);
  
}