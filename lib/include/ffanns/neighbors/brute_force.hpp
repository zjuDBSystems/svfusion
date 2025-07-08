#pragma once

#include "ffanns/neighbors/common.hpp"
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>

#include <cuda_fp16.h>

namespace ffanns::neighbors::brute_force {

struct index_params : ffanns::neighbors::index_params {};

struct search_params : ffanns::neighbors::search_params {};

/**
 * @defgroup bruteforce_cpp_index Bruteforce index
 * @{
 */
/**
 * @brief Brute Force index.
 *
 * The index stores the dataset and norms for the dataset in device memory.
 *
 * @tparam T data element type
 */
template <typename T, typename DistT = T>
struct index : ffanns::neighbors::index {
  using index_params_type  = brute_force::index_params;
  using search_params_type = brute_force::search_params;
  using index_type         = int64_t;
  using value_type         = T;

 public:
  index(const index&)            = delete;
  index(index&&)                 = default;
  index& operator=(const index&) = delete;
  index& operator=(index&&)      = default;
  ~index()                       = default;

  /**
   * @brief Construct an empty index.
   *
   * Constructs an empty index. This index will either need to be trained with `build`
   * or loaded from a saved copy with `deserialize`
   */
  index(raft::resources const& handle);

  /** Construct a brute force index from dataset
   *
   * Constructs a brute force index from a dataset. This lets us precompute norms for
   * the dataset, providing a speed benefit over doing this at query time.
   * This index will copy the host dataset onto the device, and take ownership of any
   * precaculated norms.
   */
  index(raft::resources const& res,
        raft::host_matrix_view<const T, int64_t, raft::row_major> dataset_view,
        std::optional<raft::device_vector<DistT, int64_t>>&& norms,
        ffanns::distance::DistanceType metric,
        DistT metric_arg = 0.0);

  /** Construct a brute force index from dataset
   *
   * Constructs a brute force index from a dataset. This lets us precompute norms for
   * the dataset, providing a speed benefit over doing this at query time.
   * This index will store a non-owning reference to the dataset, but will move
   * any norms supplied.
   */
  index(raft::resources const& res,
        raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view,
        std::optional<raft::device_vector<DistT, int64_t>>&& norms,
        ffanns::distance::DistanceType metric,
        DistT metric_arg = 0.0);

  /** Construct a brute force index from dataset
   *
   * This class stores a non-owning reference to the dataset and norms.
   * Having precomputed norms gives us a performance advantage at query time.
   */
  index(raft::resources const& res,
        raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view,
        std::optional<raft::device_vector_view<const DistT, int64_t>> norms_view,
        ffanns::distance::DistanceType metric,
        DistT metric_arg = 0.0);

  /**
   * Replace the dataset with a new dataset.
   */
  void update_dataset(raft::resources const& res,
                      raft::device_matrix_view<const T, int64_t, raft::row_major> dataset);

  /**
   * Replace the dataset with a new dataset.
   *
   * We create a copy of the dataset on the device. The index manages the lifetime of this copy.
   */
  void update_dataset(raft::resources const& res,
                      raft::host_matrix_view<const T, int64_t, raft::row_major> dataset);

  /** Distance metric used for retrieval */
  ffanns::distance::DistanceType metric() const noexcept { return metric_; }

  /** Metric argument */
  DistT metric_arg() const noexcept { return metric_arg_; }

  /** Total length of the index (number of vectors). */
  size_t size() const noexcept { return dataset_view_.extent(0); }

  /** Dimensionality of the data. */
  size_t dim() const noexcept { return dataset_view_.extent(1); }

  /** Dataset [size, dim] */
  raft::device_matrix_view<const T, int64_t, raft::row_major> dataset() const noexcept
  {
    return dataset_view_;
  }

  /** Dataset norms */
  raft::device_vector_view<const DistT, int64_t, raft::row_major> norms() const
  {
    return norms_view_.value();
  }

  /** Whether ot not this index has dataset norms */
  inline bool has_norms() const noexcept { return norms_view_.has_value(); }

 private:
  ffanns::distance::DistanceType metric_;
  raft::device_matrix<T, int64_t, raft::row_major> dataset_;
  std::optional<raft::device_vector<DistT, int64_t>> norms_;
  std::optional<raft::device_vector_view<const DistT, int64_t>> norms_view_;
  raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view_;
  DistT metric_arg_;
};

auto build(raft::resources const& handle,
        const ffanns::neighbors::brute_force::index_params& index_params,
        raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
    -> ffanns::neighbors::brute_force::index<float, float>;

auto build(raft::resources const& handle,
        const ffanns::neighbors::brute_force::index_params& index_params,
        raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
    -> ffanns::neighbors::brute_force::index<float, float>;

void search(raft::resources const& handle,
        const ffanns::neighbors::brute_force::search_params& params,
        const ffanns::neighbors::brute_force::index<float, float>& index,
        raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
        raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
        raft::device_matrix_view<float, int64_t, raft::row_major> distances,
        const ffanns::neighbors::filtering::base_filter& sample_filter =
            ffanns::neighbors::filtering::none_sample_filter{});

void search(raft::resources const& handle,
            const ffanns::neighbors::brute_force::index<float, float>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const ffanns::neighbors::filtering::base_filter& sample_filter =
              ffanns::neighbors::filtering::none_sample_filter{});

} // namespace ffanns::neighbors::brute_force