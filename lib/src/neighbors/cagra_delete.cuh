#pragma once

#include "detail/cagra/cagra_delete.cuh"

namespace ffanns::neighbors::cagra {

#define FFANNS_INST_CAGRA_DELETE(DTYPE)                                       \
  void lazy_delete(raft::resources const& handle,                             \
                ffanns::neighbors::cagra::index<DTYPE, uint32_t>& index,      \
                int64_t start_id,                                             \
                int64_t end_id)                                               \
   {                                                                          \
    ffanns::neighbors::cagra::detail::lazy_delete<DTYPE, uint32_t>(           \
      handle, index, start_id, end_id);                                       \
   };                                                                         \
                                                                              \
  void consolidate_delete(raft::resources const& handle,                      \
                ffanns::neighbors::cagra::index<DTYPE, uint32_t>& index,      \
                raft::host_matrix_view<DTYPE, int64_t> consolidate_dataset)     \
   {                                                                          \
    ffanns::neighbors::cagra::detail::consolidate_delete<DTYPE, uint32_t>(    \
      handle, index, consolidate_dataset);                                    \
   };                                                                         \

}  // namespace ffanns::neighbors::cagra