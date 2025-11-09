#pragma once

#include "hashmap.hpp"
#include "utils.hpp"

#include "ffanns/distance/distance.hpp"

// TODO: This shouldn't be invoking anything in detail APIs outside of ffanns/neighbors
#include <raft/core/detail/macros.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/warp_primitives.cuh>

#include <cuda_fp16.h>

#include <cfloat>
#include <cstdint>

namespace ffanns::neighbors::cagra::detail {
namespace device {

// warpSize for compile time calculation
constexpr unsigned warp_size = 32;

// using LOAD_256BIT_T = ulonglong4;
using LOAD_128BIT_T = uint4;
using LOAD_64BIT_T  = uint64_t;

template <class LOAD_T, class DATA_T>
RAFT_DEVICE_INLINE_FUNCTION constexpr unsigned get_vlen()
{
  return utils::size_of<LOAD_T>() / utils::size_of<DATA_T>();
}

/** Xorshift rondem number generator.
 *
 * See https://en.wikipedia.org/wiki/Xorshift#xorshift for reference.
 */
_RAFT_HOST_DEVICE inline uint64_t xorshift64(uint64_t u)
{
  u ^= u >> 12;
  u ^= u << 25;
  u ^= u >> 27;
  return u * 0x2545F4914F6CDD1DULL;
}

template <uint32_t Dim = 1024, uint32_t Stride = 128, typename T>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto swizzling(T x) -> T
{
  // Address swizzling reduces bank conflicts in shared memory, but increases
  // the amount of operation instead.
  // return x;
  if constexpr (Stride <= 32) {
    return x;
  } else if constexpr (Dim <= 1024) {
    return x ^ (x >> 5);
  } else {
    return x ^ ((x >> 5) & 0x1f);
  }
}

template <uint32_t TeamSize, typename T>
RAFT_DEVICE_INLINE_FUNCTION auto team_sum(T x) -> T
{
#pragma unroll
  for (uint32_t stride = TeamSize >> 1; stride > 0; stride >>= 1) {
    x += raft::shfl_xor(x, stride, TeamSize);
  }
  return x;
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION auto team_sum(T x, uint32_t team_size_bitshift) -> T
{
  switch (team_size_bitshift) {
    case 5: x += raft::shfl_xor(x, 16);
    case 4: x += raft::shfl_xor(x, 8);
    case 3: x += raft::shfl_xor(x, 4);
    case 2: x += raft::shfl_xor(x, 2);
    case 1: x += raft::shfl_xor(x, 1);
    default: return x;
  }
}

RAFT_DEVICE_INLINE_FUNCTION void lds(float& x, uint32_t addr)
{
  asm volatile("ld.shared.f32 {%0}, [%1];" : "=f"(x) : "r"(addr));
}
RAFT_DEVICE_INLINE_FUNCTION void lds(half& x, uint32_t addr)
{
  asm volatile("ld.shared.u16 {%0}, [%1];" : "=h"(reinterpret_cast<uint16_t&>(x)) : "r"(addr));
}
RAFT_DEVICE_INLINE_FUNCTION void lds(half2& x, uint32_t addr)
{
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(reinterpret_cast<uint32_t&>(x)) : "r"(addr));
}
RAFT_DEVICE_INLINE_FUNCTION void lds(half (&x)[1], uint32_t addr)
{
  asm volatile("ld.shared.u16 {%0}, [%1];" : "=h"(*reinterpret_cast<uint16_t*>(x)) : "r"(addr));
}
RAFT_DEVICE_INLINE_FUNCTION void lds(half (&x)[2], uint32_t addr)
{
  asm volatile("ld.shared.v2.u16 {%0, %1}, [%2];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)), "=h"(*reinterpret_cast<uint16_t*>(x + 1))
               : "r"(addr));
}
RAFT_DEVICE_INLINE_FUNCTION void lds(half (&x)[4], uint32_t addr)
{
  asm volatile("ld.shared.v4.u16 {%0, %1, %2, %3}, [%4];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 1)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 2)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 3))
               : "r"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void lds(uint32_t& x, uint32_t addr)
{
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(x) : "r"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void lds(uint32_t& x, const uint32_t* addr)
{
  lds(x, uint32_t(__cvta_generic_to_shared(addr)));
}

RAFT_DEVICE_INLINE_FUNCTION void lds(uint4& x, uint32_t addr)
{
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
               : "r"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void lds(uint4& x, const uint4* addr)
{
  lds(x, uint32_t(__cvta_generic_to_shared(addr)));
}

RAFT_DEVICE_INLINE_FUNCTION void sts(uint32_t addr, const half2& x)
{
  asm volatile("st.shared.v2.u16 [%0], {%1, %2};"
               :
               : "r"(addr),
                 "h"(reinterpret_cast<const uint16_t&>(x.x)),
                 "h"(reinterpret_cast<const uint16_t&>(x.y)));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_cg(uint4& x, const uint4* addr)
{
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
               : "l"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(uint4& x, const uint4* addr)
{
  asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
               : "l"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(uint32_t& x, const uint32_t* addr)
{
  asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_cg(uint32_t& x, const uint32_t* addr)
{
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half& x, const half* addr)
{
  asm volatile("ld.global.ca.u16 {%0}, [%1];"
               : "=h"(reinterpret_cast<uint16_t&>(x))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}
RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half (&x)[1], const half* addr)
{
  asm volatile("ld.global.ca.u16 {%0}, [%1];"
               : "=h"(*reinterpret_cast<uint16_t*>(x))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}
RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half (&x)[2], const half* addr)
{
  asm volatile("ld.global.ca.v2.u16 {%0, %1}, [%2];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)), "=h"(*reinterpret_cast<uint16_t*>(x + 1))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}
RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half (&x)[4], const half* addr)
{
  asm volatile("ld.global.ca.v4.u16 {%0, %1, %2, %3}, [%4];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 1)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 2)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 3))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half2& x, const half* addr)
{
  asm volatile("ld.global.ca.u32 %0, [%1];"
               : "=r"(reinterpret_cast<uint32_t&>(x))
               : "l"(reinterpret_cast<const uint32_t*>(addr)));
}
RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half2 (&x)[1], const half* addr)
{
  asm volatile("ld.global.ca.u32 %0, [%1];"
               : "=r"(*reinterpret_cast<uint32_t*>(x))
               : "l"(reinterpret_cast<const uint32_t*>(addr)));
}
RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half2 (&x)[2], const half* addr)
{
  asm volatile("ld.global.ca.v2.u32 {%0, %1}, [%2];"
               : "=r"(*reinterpret_cast<uint32_t*>(x)), "=r"(*reinterpret_cast<uint32_t*>(x + 1))
               : "l"(reinterpret_cast<const uint32_t*>(addr)));
}

}  // namespace device
}  // namespace ffanns::neighbors::cagra::detail
