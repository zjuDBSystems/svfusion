#include "cagra.cuh"
#include "ffanns/neighbors/cagra.hpp"

namespace ffanns::neighbors::cagra {

#define FFANNS_INST_CAGRA_SEARCH(T, IdxT)                                             \
  void search(raft::resources const& handle,                                          \
              ffanns::neighbors::cagra::search_params const& params,                  \
              ffanns::neighbors::cagra::index<T, IdxT>& index,                        \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,    \
              raft::host_matrix_view<const T, int64_t, raft::row_major> host_queries, \
              raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,     \
              raft::device_matrix_view<float, int64_t, raft::row_major> distances,    \
              const ffanns::neighbors::filtering::base_filter& sample_filter,         \
              bool external_flag)                                                     \
  {                                                                                   \
    ffanns::neighbors::cagra::search<T, IdxT>(                                        \
      handle, params, index, queries, host_queries, neighbors, distances, sample_filter, external_flag);           \
  }

FFANNS_INST_CAGRA_SEARCH(float, uint32_t);

#undef FFANNS_INST_CAGRA_SEARCH

}  // namespace ffanns::neighbors::cagra
