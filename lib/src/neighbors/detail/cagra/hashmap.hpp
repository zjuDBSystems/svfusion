#pragma once

#include "utils.hpp"

// TODO: This shouldn't be invoking anything from detail outside of neighbors/
#include <raft/core/detail/macros.hpp>
#include <raft/util/device_atomics.cuh>

#include <cstdint>

// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored
// #pragma GCC diagnostic pop
namespace ffanns::neighbors::cagra::detail {
namespace hashmap {

RAFT_INLINE_FUNCTION uint32_t get_size(const uint32_t bitlen) { return 1U << bitlen; }

template <class IdxT>
RAFT_DEVICE_INLINE_FUNCTION void init(IdxT* const table,
                                      const unsigned bitlen,
                                      unsigned FIRST_TID = 0)
{
  if (threadIdx.x < FIRST_TID) return;
  for (unsigned i = threadIdx.x - FIRST_TID; i < get_size(bitlen); i += blockDim.x - FIRST_TID) {
    table[i] = utils::get_max_value<IdxT>();
  }
}

template <class IdxT>
RAFT_DEVICE_INLINE_FUNCTION uint32_t insert(IdxT* const table,
                                            const uint32_t bitlen,
                                            const IdxT key)
{
  // Open addressing is used for collision resolution
  const uint32_t size     = get_size(bitlen);
  const uint32_t bit_mask = size - 1;
#if 1
  // Linear probing
  IdxT index                = (key ^ (key >> bitlen)) & bit_mask;
  constexpr uint32_t stride = 1;
#else
  // Double hashing
  uint32_t index        = key & bit_mask;
  const uint32_t stride = (key >> bitlen) * 2 + 1;
#endif
  for (unsigned i = 0; i < size; i++) {
    const IdxT old = atomicCAS(&table[index], ~static_cast<IdxT>(0), key);
    if (old == ~static_cast<IdxT>(0)) {
      return 1;
    } else if (old == key) {
      return 0;
    }
    index = (index + stride) & bit_mask;
  }
  return 0;
}

template <unsigned TEAM_SIZE, class IdxT>
RAFT_DEVICE_INLINE_FUNCTION uint32_t insert(IdxT* const table,
                                            const uint32_t bitlen,
                                            const IdxT key)
{
  IdxT ret = 0;
  if (threadIdx.x % TEAM_SIZE == 0) { ret = insert(table, bitlen, key); }
  for (unsigned offset = 1; offset < TEAM_SIZE; offset *= 2) {
    ret |= __shfl_xor_sync(0xffffffff, ret, offset);
  }
  return ret;
}

template <class IdxT>
RAFT_DEVICE_INLINE_FUNCTION uint32_t
insert(unsigned team_size, IdxT* const table, const uint32_t bitlen, const IdxT key)
{
  IdxT ret = 0;
  if (threadIdx.x % team_size == 0) { ret = insert(table, bitlen, key); }
  for (unsigned offset = 1; offset < team_size; offset *= 2) {
    ret |= __shfl_xor_sync(0xffffffff, ret, offset);
  }
  return ret;
}

// Search for a key in the hashmap
// Returns 1 if found, 0 if not found
template <class IdxT, unsigned SUPPORT_REMOVE = 0>
RAFT_DEVICE_INLINE_FUNCTION uint32_t search(IdxT* const table,
                                           const uint32_t bitlen, 
                                           const IdxT key)
{
  const uint32_t size     = get_size(bitlen);
  const uint32_t bit_mask = size - 1;
  
  // Use same hash strategy as insert for consistency
#if 1
  // Linear probing
  IdxT index                = (key ^ (key >> bitlen)) & bit_mask;
  constexpr uint32_t stride = 1;
#else
  // Double hashing
  IdxT index            = key & bit_mask;
  const uint32_t stride = (key >> bitlen) * 2 + 1;
#endif
  
  constexpr IdxT hashval_empty = ~static_cast<IdxT>(0);
  const IdxT removed_key       = key | utils::gen_index_msb_1_mask<IdxT>::value;
  
  for (unsigned i = 0; i < size; i++) {
    const IdxT val = table[index];
    if (val == key) {
      return 1;  // Found
    } else if (val == hashval_empty) {
      return 0;  // Empty slot means key not present
    } else if (SUPPORT_REMOVE) {
      // Check if this key has been removed (for compatibility with CUVS)
      if (val == removed_key) { return 0; }
    }
    index = (index + stride) & bit_mask;
  }
  return 0;  // Not found after full table scan
}

template <class IdxT>
RAFT_DEVICE_INLINE_FUNCTION bool remove(IdxT* const table,
                                       const uint32_t bitlen,
                                       const IdxT key)
{
  // 使用与insert相同的哈希策略确保一致性
  const uint32_t size = get_size(bitlen);
  const uint32_t bit_mask = size - 1;
  
  // 线性探测 - 与insert使用相同的哈希函数
  IdxT index = (key ^ (key >> bitlen)) & bit_mask;
  constexpr uint32_t stride = 1;
  
  // 在哈希表中搜索key
  for (unsigned i = 0; i < size; i++) {
    IdxT current = table[index];
    
    // 找到key，将其替换为空值标记
    if (current == key) {
      atomicExch(&table[index], utils::get_max_value<IdxT>());
      return true;
    }
    
    // 如果遇到空槽，说明key不存在
    if (current == utils::get_max_value<IdxT>()) {
      return false;
    }
    
    // 继续线性探测
    index = (index + stride) & bit_mask;
  }
  
  // 遍历整个表都没找到
  return false;
}

}  // namespace hashmap
}  // namespace ffanns::neighbors::cagra::detail
