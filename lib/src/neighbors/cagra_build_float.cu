#include "cagra.cuh"
#include "ffanns/neighbors/cagra.hpp"

namespace ffanns::neighbors::cagra {

#define FFANNS_INST_CAGRA_BUILD(T, IdxT)                                            \
  auto build(raft::resources const& handle,                                         \
             const ffanns::neighbors::cagra::index_params& params,              \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,     \
             raft::host_matrix_view<IdxT, int64_t, raft::row_major> index_graph,    \
             raft::device_matrix<T, int64_t>& device_dataset_ref,  \
             raft::device_matrix<IdxT, int64_t>& device_graph_ref,   \
             std::shared_ptr<ffanns::core::bitset<std::uint32_t, int64_t>> delete_bitset,   \
             std::shared_ptr<rmm::device_uvector<IdxT>> tag_to_id,                                  \
             IdxT start_id, IdxT end_id)                                                    \
    -> ffanns::neighbors::cagra::index<T, IdxT>                                 \
  {                                                                                 \
    return ffanns::neighbors::cagra::build<T, IdxT>(handle, params, dataset, index_graph, device_dataset_ref, device_graph_ref, delete_bitset, tag_to_id, start_id, end_id);   \
  }

FFANNS_INST_CAGRA_BUILD(float, uint32_t);

#undef FFANNS_INST_CAGRA_BUILD

}  // namespace ffanns::neighbors::cagra
