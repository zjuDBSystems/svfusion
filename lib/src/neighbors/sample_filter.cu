#include "sample_filter.cuh"

namespace ffanns::neighbors::filtering {

template struct bitset_filter<uint8_t, uint32_t>;
template struct bitset_filter<uint16_t, uint32_t>;
template struct bitset_filter<uint32_t, uint32_t>;
template struct bitset_filter<uint32_t, int64_t>;
template struct bitset_filter<uint64_t, int64_t>;

}  // namespace ffanns::neighbors::filtering
