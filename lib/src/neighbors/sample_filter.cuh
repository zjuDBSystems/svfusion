#pragma once

#include "ffanns/neighbors/common.hpp"
// #include "ffanns/core/bitset.hpp"
#include <raft/core/bitset.cuh>
#include <raft/core/detail/macros.hpp>
// #include <raft/sparse/convert/csr.cuh>

#include <cstddef>
#include <cstdint>

namespace ffanns::neighbors::filtering {

/* A filter that filters nothing. This is the default behavior. */
inline _RAFT_HOST_DEVICE bool none_sample_filter::operator()(
  // the index of the current sample inside the current inverted list
  const uint32_t sample_ix) const
{
  return true;
}

template <typename bitset_t, typename index_t>
bitset_filter<bitset_t, index_t>::bitset_filter(
  const ffanns::core::bitset_view<bitset_t, index_t> bitset_for_filtering)
  : bitset_view_{bitset_for_filtering}
{
}

template <typename bitset_t, typename index_t>
inline _RAFT_HOST_DEVICE bool bitset_filter<bitset_t, index_t>::operator()(
  // the index of the current sample
  const uint32_t sample_ix) const
{
  return bitset_view_.test(sample_ix);
}

// template <typename bitset_t, typename index_t>
// template <typename csr_matrix_t>
// void bitset_filter<bitset_t, index_t>::to_csr(raft::resources const& handle, csr_matrix_t& csr)
// {
//   raft::sparse::convert::bitset_to_csr(handle, bitset_view_, csr);
// }

} // namespace ffanns::neighbors::filtering