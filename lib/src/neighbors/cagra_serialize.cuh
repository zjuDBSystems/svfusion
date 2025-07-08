#pragma once

#include "detail/cagra/cagra_serialize.cuh"

namespace ffanns::neighbors::cagra {

#define FFANNS_INST_CAGRA_SERIALIZE(DTYPE)                                                    \
  void serialize(raft::resources const& handle,                                               \
                 const std::string& filename,                                                 \
                 const ffanns::neighbors::cagra::index<DTYPE, uint32_t>& index,               \
                 bool include_dataset)                                                        \
  {                                                                                           \
    ffanns::neighbors::cagra::detail::serialize<DTYPE, uint32_t>(                             \
      handle, filename, index, include_dataset);                                              \
  };                                                                                          \
                                                                                              \
  void deserialize(raft::resources const& handle,                                             \
                   const std::string& filename,                                               \
                   ffanns::neighbors::cagra::index<DTYPE, uint32_t>* index)                   \
  {                                                                                           \
    ffanns::neighbors::cagra::detail::deserialize<DTYPE, uint32_t>(handle, filename, index);  \
  };                                                                                          \
  void serialize(raft::resources const& handle,                                               \
                 std::ostream& os,                                                            \
                 const ffanns::neighbors::cagra::index<DTYPE, uint32_t>& index,               \
                 bool include_dataset)                                                        \
  {                                                                                           \
    ffanns::neighbors::cagra::detail::serialize<DTYPE, uint32_t>(                             \
      handle, os, index, include_dataset);                                                    \
  }                                                                                           \
                                                                                              \
  void deserialize(raft::resources const& handle,                                             \
                   std::istream& is,                                                          \
                   ffanns::neighbors::cagra::index<DTYPE, uint32_t>* index)                   \
  {                                                                                           \
    ffanns::neighbors::cagra::detail::deserialize<DTYPE, uint32_t>(handle, is, index);        \
  }                                                                                           

}  // namespace ffanns::neighbors::cagra
