#include "./detail/knn_brute_force.cuh"

#include "ffanns/neighbors/brute_force.hpp"

#include <raft/core/copy.hpp>

namespace ffanns::neighbors::brute_force {

template <typename T, typename DistT>
index<T, DistT>::index(raft::resources const& res)
    // this constructor is just for a temporary index, for use in the deserialization
    // api. all the parameters here will get replaced with loaded values - that aren't
    // necessarily known ahead of time before deserialization.
    // TODO: do we even need a handle here - could just construct one?
    : ffanns::neighbors::index(),
    metric_(ffanns::distance::DistanceType::L2Expanded),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    norms_(std::nullopt),
    metric_arg_(0)
{
}

template <typename T, typename DistT>
index<T, DistT>::index(raft::resources const& res,
                       raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
                       std::optional<raft::device_vector<DistT, int64_t>>&& norms,
                       ffanns::distance::DistanceType metric,
                       DistT metric_arg)
  : ffanns::neighbors::index(),
    metric_(metric),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    norms_(std::move(norms)),
    metric_arg_(metric_arg)
{
  if (norms_) { norms_view_ = raft::make_const_mdspan(norms_.value().view()); }
  update_dataset(res, dataset);
  raft::resource::sync_stream(res);
}

template <typename T, typename DistT>
index<T, DistT>::index(raft::resources const& res,
                       raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
                       std::optional<raft::device_vector<DistT, int64_t>>&& norms,
                       ffanns::distance::DistanceType metric,
                       DistT metric_arg)
  : ffanns::neighbors::index(),
    metric_(metric),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    norms_(std::move(norms)),
    metric_arg_(metric_arg)
{
  if (norms_) { norms_view_ = raft::make_const_mdspan(norms_.value().view()); }
  update_dataset(res, dataset);
}

template <typename T, typename DistT>
index<T, DistT>::index(raft::resources const& res,
                       raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view,
                       std::optional<raft::device_vector_view<const DistT, int64_t>> norms_view,
                       ffanns::distance::DistanceType metric,
                       DistT metric_arg)
  : ffanns::neighbors::index(),
    metric_(metric),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    dataset_view_(dataset_view),
    norms_view_(norms_view),
    metric_arg_(metric_arg)
{
}

template <typename T, typename DistT>
void index<T, DistT>::update_dataset(
  raft::resources const& res, raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)
{
  dataset_view_ = dataset;
}

template <typename T, typename DistT>
void index<T, DistT>::update_dataset(
  raft::resources const& res, raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
{
  dataset_ = raft::make_device_matrix<T, int64_t>(res, dataset.extent(0), dataset.extent(1));
  raft::copy(res, dataset_.view(), dataset);
  dataset_view_ = raft::make_const_mdspan(dataset_.view());
}

#define FFANNS_INST_BFKNN(T, DistT)                                                               \
  auto build(raft::resources const& res,                                                        \
             const ffanns::neighbors::brute_force::index_params& index_params,                    \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)               \
    ->ffanns::neighbors::brute_force::index<T, DistT>                                             \
  {                                                                                             \
    return detail::build<T, DistT>(res, dataset, index_params.metric, index_params.metric_arg); \
  }                                                                                             \
  auto build(raft::resources const& res,                                                        \
             const ffanns::neighbors::brute_force::index_params& index_params,                    \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)                 \
    ->ffanns::neighbors::brute_force::index<T, DistT>                                             \
  {                                                                                             \
    return detail::build<T, DistT>(res, dataset, index_params.metric, index_params.metric_arg); \
  }                                                                                             \
                                                                                                \
  void search(raft::resources const& res,                                                       \
              const ffanns::neighbors::brute_force::search_params& params,                      \
              const ffanns::neighbors::brute_force::index<T, DistT>& idx,                       \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,              \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,            \
              raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,              \
              const ffanns::neighbors::filtering::base_filter& sample_filter)                   \
  {                                                                                             \
    detail::search<T, int64_t, DistT, raft::row_major>(                                         \
      res, idx, queries, neighbors, distances, sample_filter);                                  \
  }                                                                                             \
  void search(raft::resources const& res,                                                       \
              const ffanns::neighbors::brute_force::index<T, DistT>& idx,                       \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,              \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,            \
              raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,              \
              const ffanns::neighbors::filtering::base_filter& sample_filter)                   \
  {                                                                                             \
    detail::search<T, int64_t, DistT, raft::row_major>(                                         \
      res, idx, queries, neighbors, distances, sample_filter);                                  \
  }                                                                                             \
                                                                                                \
  template struct ffanns::neighbors::brute_force::index<T, DistT>;

FFANNS_INST_BFKNN(float, float);

#undef FFANNS_INST_BFKNN

} // namespace ffanns::neighbors::brute_force