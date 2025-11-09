// Prevent multiple inclusions across and within translation units
#pragma once

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <raft/core/logger.hpp>

namespace ffanns::core {

static const std::string RAFT_NAME = "raft";
// AVX2 下水平求和：将 __m256 中的8个 float 累加为一个 float
inline float reduce_add_ps(__m256 v) {
    // 分别取低128位和高128位
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    // 相加
    __m128 sum128 = _mm_add_ps(vlow, vhigh);
    // 采用 SSE 的水平加法
    __m128 shuf = _mm_movehdup_ps(sum128);  // (sum128[1], sum128[1], sum128[3], sum128[3])
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline float _mm256_reduce_add_ps(__m256 x)
{
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

inline float l2_distance_avx2_new(const float* a, const float* b, uint32_t size) {
#ifdef USE_AVX2
    // a = (const float *)__builtin_assume_aligned(a, 32);
    // b = (const float *)__builtin_assume_aligned(b, 32);
    
    float result = 0.0f;

    uint16_t niters = (uint16_t)(size / 8);
    __m256 sum = _mm256_setzero_ps();
    
    for (uint16_t j = 0; j < niters; j++) {
        // 预取下一次迭代的数据
        if (j < (niters - 1)) {
            _mm_prefetch((char *)(a + 8 * (j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(b + 8 * (j + 1)), _MM_HINT_T0);
        }
        
        // 加载向量
        // __m256 a_vec = _mm256_load_ps(a + 8 * j);  // 使用load_ps而不是loadu_ps，假设已对齐
        // __m256 b_vec = _mm256_load_ps(b + 8 * j);
        __m256 a_vec = _mm256_loadu_ps(a + 8 * j);
        __m256 b_vec = _mm256_loadu_ps(b + 8 * j);
        // a_vec - b_vec
        __m256 diff_vec = _mm256_sub_ps(a_vec, b_vec);
        
        sum = _mm256_fmadd_ps(diff_vec, diff_vec, sum);
    }
    
    // 水平求和
    result = _mm256_reduce_add_ps(sum);
    
    for (uint32_t i = niters * 8; i < size; i++) {
        float d = a[i] - b[i];
        result += d * d;
    }
    
    return result;
#else
    // 没有AVX2时回退到标量实现
    RAFT_LOG_INFO("[l2_distance_avx2] AVX2 not supported, falling back to scalar implementation");
    float result = 0.0f;
    #ifdef _OPENMP
    #pragma omp simd reduction(+ : result)
    #endif
    for (uint32_t i = 0; i < size; i++) {
        float d = a[i] - b[i];
        result += d * d;
    }
    return result;
#endif
}

inline float l2_distance_avx512(const float* a, const float* b, uint32_t size) {
#ifdef USE_AVX2
    // RAFT_LOG_INFO("[l2_distance_avx512] Using AVX512 implementation");
    uint32_t n_iters = size / 16;  // 一次处理16个float元素
    __m512 sum = _mm512_setzero_ps();
    
    // 主循环：每次处理16个float
    for (uint32_t i = 0; i < n_iters; i++) {
        // 预取代码（如需启用）
        if (i < (n_iters - 1)) {
            _mm_prefetch((char *)(a + 16 * (i + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(b + 16 * (i + 1)), _MM_HINT_T0);
        }

        __m512 va = _mm512_loadu_ps(a + i * 16);
        __m512 vb = _mm512_loadu_ps(b + i * 16);
        __m512 diff = _mm512_sub_ps(va, vb);
        __m512 sq = _mm512_mul_ps(diff, diff);
        sum = _mm512_add_ps(sum, sq);
    }
    
    // 水平求和，使用AVX512专用指令
    float result = _mm512_reduce_add_ps(sum);
    
    // 处理不足16个元素的尾部部分
    for (uint32_t i = n_iters * 16; i < size; i++) {
        float d = a[i] - b[i];
        result += d * d;
    }
    
    return result;
#else
    // 如果不支持AVX512，尝试使用AVX2
    #ifdef USE_AVX2
        return l2_distance_avx2(a, b, size);
    #else
        // 没有AVX512或AVX2时回退到标量实现
        RAFT_LOG_INFO("[l2_distance_avx512] AVX512/AVX2 not supported, falling back to scalar implementation");
        float result = 0.0f;
        for (uint32_t i = 0; i < size; i++) {
            float d = a[i] - b[i];
            result += d * d;
        }
        return result;
    #endif
#endif
}

inline float l2_distance_avx512(const uint8_t* a, const uint8_t* b, uint32_t size) {
    return 0.0f;
}

// 使用 AVX2 计算 L2 距离的函数，size 表示向量长度
inline float l2_distance_avx2(const float* a, const float* b, uint32_t size) {
#ifdef USE_AVX2
    // RAFT_LOG_INFO("[l2_distance_avx2] Using AVX2 implementation");
    uint32_t n_iters = size / 8;
    __m256 sum = _mm256_setzero_ps();
    // 主循环：每次处理8个 float
    for (uint32_t i = 0; i < n_iters; i++) {
        // 预取下一次迭代的数据
        if (i < (n_iters - 1)) {
            _mm_prefetch((char *)(a + 8 * (i + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(b + 8 * (i + 1)), _MM_HINT_T0);
        }

        __m256 va = _mm256_loadu_ps(a + i * 8);
        __m256 vb = _mm256_loadu_ps(b + i * 8);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
    }
    // float result = reduce_add_ps(sum);
    float result = _mm256_reduce_add_ps(sum);
    // 处理不足8个元素的尾部部分
    for (uint32_t i = n_iters * 8; i < size; i++) {
        float d = a[i] - b[i];
        result += d * d;
    }
    return result;
#else
    // 没有 AVX2 时回退到标量实现
    RAFT_LOG_INFO("[l2_distance_avx2] AVX2 not supported, falling back to scalar implementation");
    float result = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        float d = a[i] - b[i];
        result += d * d;
    }
    return result;
#endif
}

// 使用 AVX2 计算 uint8_t 类型向量的 L2 距离
inline float l2_distance_avx2(const uint8_t* a, const uint8_t* b, uint32_t size) {
#ifdef USE_AVX2
    uint32_t n_iters = size / 32;
    __m256 sum = _mm256_setzero_ps();
    
    for (uint32_t i = 0; i < n_iters; i++) {
        // 加载32个uint8_t元素
        __m256i va = _mm256_loadu_si256((__m256i*)(a + i * 32));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b + i * 32));
        
        // 处理前16个元素 - 先转换为16位再相减
        __m128i va_low = _mm256_extracti128_si256(va, 0);
        __m128i vb_low = _mm256_extracti128_si256(vb, 0);
        __m256i va_low_16 = _mm256_cvtepu8_epi16(va_low);
        __m256i vb_low_16 = _mm256_cvtepu8_epi16(vb_low);
        __m256i diff_low = _mm256_sub_epi16(va_low_16, vb_low_16);
        
        // 计算平方
        __m256i sq_low = _mm256_mullo_epi16(diff_low, diff_low);
        
        // 处理后16个元素 - 同样先转换为16位再相减
        __m128i va_high = _mm256_extracti128_si256(va, 1);
        __m128i vb_high = _mm256_extracti128_si256(vb, 1);
        __m256i va_high_16 = _mm256_cvtepu8_epi16(va_high);
        __m256i vb_high_16 = _mm256_cvtepu8_epi16(vb_high);
        __m256i diff_high = _mm256_sub_epi16(va_high_16, vb_high_16);
        
        // 计算平方
        __m256i sq_high = _mm256_mullo_epi16(diff_high, diff_high);
        
        // 将16位平方结果转换为32位并累加到sum
        // 处理前8个元素
        __m128i sq_low_low = _mm256_extracti128_si256(sq_low, 0);
        __m256i sq_low_low_32 = _mm256_cvtepi16_epi32(sq_low_low);
        __m256 sq_low_low_f = _mm256_cvtepi32_ps(sq_low_low_32);
        sum = _mm256_add_ps(sum, sq_low_low_f);
        
        // 处理中间8个元素
        __m128i sq_low_high = _mm256_extracti128_si256(sq_low, 1);
        __m256i sq_low_high_32 = _mm256_cvtepi16_epi32(sq_low_high);
        __m256 sq_low_high_f = _mm256_cvtepi32_ps(sq_low_high_32);
        sum = _mm256_add_ps(sum, sq_low_high_f);
        
        // 处理接下来8个元素
        __m128i sq_high_low = _mm256_extracti128_si256(sq_high, 0);
        __m256i sq_high_low_32 = _mm256_cvtepi16_epi32(sq_high_low);
        __m256 sq_high_low_f = _mm256_cvtepi32_ps(sq_high_low_32);
        sum = _mm256_add_ps(sum, sq_high_low_f);
        
        // 处理最后8个元素
        __m128i sq_high_high = _mm256_extracti128_si256(sq_high, 1);
        __m256i sq_high_high_32 = _mm256_cvtepi16_epi32(sq_high_high);
        __m256 sq_high_high_f = _mm256_cvtepi32_ps(sq_high_high_32);
        sum = _mm256_add_ps(sum, sq_high_high_f);
    }
    
    float result = reduce_add_ps(sum);
    
    // 处理剩余元素
    for (uint32_t i = n_iters * 32; i < size; i++) {
        int16_t d = static_cast<int16_t>(a[i]) - static_cast<int16_t>(b[i]);
        result += d * d;
    }
    
    return result;
#else
    // 没有 AVX2 时回退到标量实现
    float result = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        int16_t d = static_cast<int16_t>(a[i]) - static_cast<int16_t>(b[i]);
        result += d * d;
    }
    return result;
#endif
}

// 使用 AVX2 计算 L2 距离的函数，size 表示向量长度
__host__ inline void batch_l2_distance_avx2(
    const float* a, const float* const* batch_vectors,
    float* distances,  uint32_t batch_size, uint32_t size) 
{
    #ifdef USE_AVX2
        // RAFT_LOG_INFO("[l2_distance_avx2] Using AVX2 implementation");
        uint32_t n_iters = size / 8;
        // std::vector<__m256> sums(batch_size, _mm256_setzero_ps());
        __m256* sums = new __m256[batch_size];
        for (uint32_t b = 0; b < batch_size; b++) {
            sums[b] = _mm256_setzero_ps();
        }
        // 主循环：每次处理8个 float
        for (uint32_t i = 0; i < n_iters; i++) {
            if (i < (n_iters - 1)) {
                _mm_prefetch((char *)(a + 8 * (i + 1)), _MM_HINT_T0);
            }
            __m256 va = _mm256_loadu_ps(a + i * 8);

            for (uint32_t b = 0; b < batch_size; b++) {
                if (b + 1 < batch_size) {
                    _mm_prefetch((char*)(batch_vectors[b + 1] + i * 8), _MM_HINT_T0);
                }
                __m256 vb = _mm256_loadu_ps(batch_vectors[b] + i * 8);
                __m256 diff = _mm256_sub_ps(va, vb);
                sums[b] = _mm256_fmadd_ps(diff, diff, sums[b]);
                // __m256 sq = _mm256_mul_ps(diff, diff);
                // sums[b] = _mm256_add_ps(sums[b], sq);
            }
        }

        for (uint32_t b = 0; b < batch_size; b++) {
            // 水平求和
            float result = _mm256_reduce_add_ps(sums[b]);
            
            // 处理尾部元素
            for (uint32_t i = n_iters * 8; i < size; i++) {
                float d = a[i] - batch_vectors[b][i];
                result += d * d;
            }
            
            distances[b] = result;
        }
        delete[] sums;
    #else
        // 没有 AVX2 时回退到标量实现
        RAFT_LOG_INFO("[l2_distance_avx2] AVX2 not supported !!!");
    #endif
}


// // 分块大小（可调整）
// const uint32_t BLOCK_SIZE = 16; // 处理16个8浮点块

// for (uint32_t block = 0; block < n_iters; block += BLOCK_SIZE) {
//     uint32_t end_block = std::min(block + BLOCK_SIZE, n_iters);
    
//     for (uint32_t i = block; i < end_block; i++) {
//         __m256 va = _mm256_loadu_ps(a + i * 8);
        
//         for (uint32_t b = 0; b < batch_size; b++) {
//             // 计算过程...
//         }
//     }
// }

__host__ inline void batch_l2_distance_avx2(
    const uint8_t* a, const uint8_t* const* batch_vectors,
    float* distances,  uint32_t batch_size, uint32_t size) 
{
}

inline float inner_product_avx2(const float* a,
                                const float* b,
                                uint32_t      size) {
#ifdef USE_AVX2
    /* -------- AVX2 处理 8 个 float/循环 -------- */
    uint32_t n_iters = size >> 3;          // size / 8
    __m256   sum     = _mm256_setzero_ps();

    for (uint32_t i = 0; i < n_iters; ++i) {
        /* 预取下一批数据，隐藏 L1 miss 延迟 */
        if (i + 1 < n_iters) {
            _mm_prefetch((char*)(a + 8 * (i + 1)), _MM_HINT_T0);
            _mm_prefetch((char*)(b + 8 * (i + 1)), _MM_HINT_T0);
        }

        __m256 va = _mm256_loadu_ps(a + (i << 3));
        __m256 vb = _mm256_loadu_ps(b + (i << 3));
        sum       = _mm256_fmadd_ps(va, vb, sum);  // sum += va*vb
    }

    /* 水平求和 8&times;float 向量 */
    float result = _mm256_reduce_add_ps(sum);

    /* 处理 size 不是 8 的倍数的尾巴 */
    for (uint32_t i = n_iters << 3; i < size; ++i)
        result += a[i] * b[i];

    return result;
#else
    RAFT_LOG_INFO("[inner_product_avx2] AVX2 not supported !!!");
    /* -------- 标量回退 -------- */
    float result = 0.f;
        for (uint32_t i = 0; i < size; ++i) result += a[i] * b[i];
    return result;
#endif
}

inline float inner_product_avx2(const uint8_t* a,
    const uint8_t* b,
    uint32_t      size) {
        RAFT_LOG_INFO("[inner_product_avx2] uint8_t not supported !!!");
        return 0.0f;
    }

inline float neg_inner_product_avx2(const float* a,
                                    const float* b,
                                    uint32_t     size) {
    return -inner_product_avx2(a, b, size);
}

inline float neg_inner_product_avx2(const uint8_t* a,
    const uint8_t* b,
    uint32_t     size) {
return -inner_product_avx2(a, b, size);
}

} // namespace ffanns::core
