#pragma once

#include "ffanns/neighbors/common.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include "ffanns/distance/distance.hpp"

#include <cuda_fp16.h>
namespace ffanns::neighbors::nn_descent {
struct index_params : ffanns::neighbors::index_params {
  size_t graph_degree              = 64;      // Degree of output graph.
  size_t intermediate_graph_degree = 128;     // Degree of input graph for pruning.
  size_t max_iterations            = 20;      // Number of nn-descent iterations.
  float termination_threshold      = 0.0001;  // Termination threshold of nn-descent.
  size_t n_clusters                = 1;       // defaults to not using any batching

  /** @brief Construct NN descent parameters for a specific kNN graph degree
   *
   * @param graph_degree output graph degree
   */
  index_params(size_t graph_degree = 64, 
              ffanns::distance::DistanceType metric = ffanns::distance::DistanceType::L2Expanded)
      : graph_degree(graph_degree), 
        intermediate_graph_degree(static_cast<size_t>(1.5 * graph_degree))
  {
      this->metric = metric; // 显式设置基类的 metric 成员变量
  }
};

  /**
 * @defgroup nn_descent_cpp_index nn-descent index
 * @{
 */
/**
 * @brief nn-descent Build an nn-descent index
 * The index contains an all-neighbors graph of the input dataset
 * stored in host memory of dimensions (n_rows, n_cols)
 *
 * Reference:
 * Hui Wang, Wan-Lei Zhao, Xiangxiang Zeng, and Jianye Yang. 2021.
 * Fast k-NN Graph Construction by GPU based NN-Descent. In Proceedings of the 30th ACM
 * International Conference on Information & Knowledge Management (CIKM '21). Association for
 * Computing Machinery, New York, NY, USA, 1929–1938. https://doi.org/10.1145/3459637.3482344
 *
 * @tparam IdxT dtype to be used for constructing knn-graph
 */
template <typename IdxT>
struct index : ffanns::neighbors::index {
 public:
  index(raft::resources const& res, int64_t n_rows, int64_t n_cols, 
        ffanns::distance::DistanceType metric = ffanns::distance::DistanceType::L2Expanded)
    : ffanns::neighbors::index(),
      res_{res},
      metric_{metric},
      graph_{raft::make_host_matrix<IdxT, int64_t, raft::row_major>(n_rows, n_cols)},
      graph_view_{graph_.view()}
  {
  }

  index(raft::resources const& res,
        raft::host_matrix_view<IdxT, int64_t, raft::row_major> graph_view,
        ffanns::distance::DistanceType metric = ffanns::distance::DistanceType::L2Expanded)
    : ffanns::neighbors::index(),
      res_{res},
      metric_{metric},
      graph_{raft::make_host_matrix<IdxT, int64_t, raft::row_major>(0, 0)},
      graph_view_{graph_view}
  {
  }

  /** Distance metric used for clustering. */
  [[nodiscard]] constexpr inline auto metric() const noexcept -> ffanns::distance::DistanceType
  {
    return metric_;
  }

  // /** Total length of the index (number of vectors). */
  [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT
  {
    return graph_view_.extent(0);
  }

  /** Graph degree */
  [[nodiscard]] constexpr inline auto graph_degree() const noexcept -> uint32_t
  {
    return graph_view_.extent(1);
  }

  /** neighborhood graph [size, graph-degree] */
  [[nodiscard]] inline auto graph() noexcept
    -> raft::host_matrix_view<IdxT, int64_t, raft::row_major>
  {
    return graph_view_;
  }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

 private:
  raft::resources const& res_;
  ffanns::distance::DistanceType metric_;
  raft::host_matrix<IdxT, int64_t, raft::row_major> graph_;  // graph to return for non-int IdxT
  raft::host_matrix_view<IdxT, int64_t, raft::row_major>
    graph_view_;  // view of graph for user provided matrix
};

} // ffanns::neighbors::nn_descent
