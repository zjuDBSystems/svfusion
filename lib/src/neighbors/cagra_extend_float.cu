#include "cagra.cuh"
#include "ffanns/neighbors/cagra.hpp"

namespace ffanns::neighbors::cagra {

#define FFANNS_INST_CAGRA_EXTEND(T, IdxT)                                                       \
  void extend(raft::resources const& handle,                                                    \
              const cagra::extend_params& params,                                               \
              raft::host_matrix_view<const T, int64_t, raft::row_major> additional_dataset,     \
              ffanns::neighbors::cagra::index<T, IdxT>& idx,                                    \
              std::optional<raft::host_matrix_view<T, int64_t, raft::layout_stride>> ndv,       \
              std::optional<raft::device_matrix_view<T, int64_t, raft::layout_stride>> nddv,    \
              std::optional<raft::host_matrix_view<IdxT, int64_t, raft::layout_stride>> ngv,    \
              std::optional<raft::device_matrix_view<IdxT, int64_t>> ndgv,                      \
              IdxT start_id, IdxT end_id)                                                       \
  {                                                                                             \
    ffanns::neighbors::cagra::extend<T, IdxT>(handle, params, additional_dataset, idx, ndv, nddv, ngv, ndgv, start_id, end_id); \
  }
  
FFANNS_INST_CAGRA_EXTEND(float, uint32_t);

#undef FFANNS_INST_CAGRA_EXTEND

}  // namespace ffanns::neighbors::cagra
