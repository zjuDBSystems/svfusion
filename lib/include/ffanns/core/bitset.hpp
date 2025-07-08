#pragma once

#include <raft/core/bitset.hpp>
#include <vector>
#include <cstdint>
#include <algorithm>

extern template struct raft::core::bitset<uint8_t, uint32_t>;
extern template struct raft::core::bitset<uint16_t, uint32_t>;
extern template struct raft::core::bitset<uint32_t, uint32_t>;
extern template struct raft::core::bitset<uint32_t, int64_t>;
extern template struct raft::core::bitset<uint64_t, int64_t>;

namespace ffanns::core {
/* To use bitset functions containing CUDA code, include <raft/core/bitset.cuh> */

template <typename bitset_t, typename index_t>
using bitset_view = raft::core::bitset_view<bitset_t, index_t>;

template <typename bitset_t, typename index_t>
using bitset = raft::core::bitset<bitset_t, index_t>;

class HostBitSet {
    private:
        std::vector<uint32_t> bits;
        size_t num_bits;
        
        static constexpr size_t BITS_PER_WORD = 32;
        
        static size_t word_index(size_t bit_pos) {
            return bit_pos / BITS_PER_WORD;
        }
        
        static uint32_t bit_mask(size_t bit_pos) {
            return 1u << (bit_pos % BITS_PER_WORD);
        }
    public:
        // 默认构造函数：所有位设为1（未删除）
        explicit HostBitSet(size_t size) : num_bits(size) {
            bits.resize((size + BITS_PER_WORD - 1) / BITS_PER_WORD, ~0u);
            if (num_bits % BITS_PER_WORD != 0) {
                bits.back() &= (1u << (num_bits % BITS_PER_WORD)) - 1;
            }
        }
        
        // 带初始值的构造函数
        HostBitSet(size_t size, bool all_valid) : num_bits(size) {
            bits.resize((size + BITS_PER_WORD - 1) / BITS_PER_WORD, all_valid ? ~0u : 0);  
            // 处理最后一个字的未使用位
            if (all_valid && num_bits % BITS_PER_WORD != 0) {
                bits.back() &= (1u << (num_bits % BITS_PER_WORD)) - 1;
            }
        }
        
        void mark_deleted(size_t pos) {
            if (pos < num_bits) {
                bits[word_index(pos)] &= ~bit_mask(pos);
            }
        }
        
        void mark_valid(size_t pos) {
            if (pos < num_bits) {
                bits[word_index(pos)] |= bit_mask(pos);
            }
        }
        
        // 检查位是否已删除（为0）
        bool is_deleted(size_t pos) const {
            if (pos < num_bits) {
                return (bits[word_index(pos)] & bit_mask(pos)) == 0;
            }
            return false;
        }
        
        // 检查位是否有效（为1）
        bool test(size_t pos) const {
            if (pos < num_bits) {
                return (bits[word_index(pos)] & bit_mask(pos)) != 0;
            }
            return false;
        }
        
        // 标记范围为删除状态
        void mark_range_deleted(size_t start_pos, size_t end_pos) {
            if (start_pos >= end_pos || start_pos >= num_bits) return;
            
            end_pos = std::min(end_pos, num_bits);
            
            size_t start_word = word_index(start_pos);
            size_t start_offset = start_pos % BITS_PER_WORD;
            size_t end_word = word_index(end_pos - 1);
            size_t end_offset = (end_pos - 1) % BITS_PER_WORD;
            
            // 创建第一个和最后一个字的掩码
            uint32_t first_mask = ~0u << start_offset;
            uint32_t last_mask = (end_offset + 1 == BITS_PER_WORD) 
                         ? ~0u 
                         : ~(~0u << (end_offset + 1));
            
            if (start_word == end_word) {
                bits[start_word] &= ~(first_mask & last_mask);
            } else {
                bits[start_word] &= ~first_mask;
                
                for (size_t i = start_word + 1; i < end_word; ++i) {
                    bits[i] = 0;
                }
                
                bits[end_word] &= ~last_mask;
            }
        }

        void mark_range_valid(size_t start_pos, size_t end_pos) {
            if (start_pos >= end_pos || start_pos >= num_bits) return;
            
            end_pos = std::min(end_pos, num_bits);
            
            size_t start_word = word_index(start_pos);
            size_t start_offset = start_pos % BITS_PER_WORD;
            size_t end_word = word_index(end_pos - 1);
            size_t end_offset = (end_pos - 1) % BITS_PER_WORD;
            
            // 创建第一个和最后一个字的掩码
            uint32_t first_mask = ~0u << start_offset;
            uint32_t last_mask = (end_offset + 1 == BITS_PER_WORD) 
                                ? ~0u 
                                : ~(~0u << (end_offset + 1));
            
            if (start_word == end_word) {
                bits[start_word] |= first_mask & last_mask;  // 使用 OR 操作将对应位标记为 1
            } else {
                bits[start_word] |= first_mask;  // 标记第一个字
                for (size_t i = start_word + 1; i < end_word; ++i) {
                    bits[i] = ~0u;  // 其他字全部设为 1
                }
                bits[end_word] |= last_mask;  // 标记最后一个字
            }
        }
        
        // 标记所有位为删除状态
        void mark_all_deleted() {
            std::fill(bits.begin(), bits.end(), 0);
        }
        
        // 标记所有位为有效状态
        void mark_all_valid() {
            std::fill(bits.begin(), bits.end(), ~0u);
            
            // 处理最后一个字的未使用位
            if (num_bits % BITS_PER_WORD != 0) {
                bits.back() &= (1u << (num_bits % BITS_PER_WORD)) - 1;
            }
        }
        
        // 计算未删除的位数
        size_t count_valid() const {
            size_t count = 0;
            for (uint32_t word : bits) {
                for (uint32_t temp = word; temp; temp &= temp - 1) {
                    count++;
                }
            }
            return count;
        }
        
        size_t count_deleted() const {
            return num_bits - count_valid();
        }
        
        size_t size() const {
            return num_bits;
        }
        
        const uint32_t* data() const {
            return bits.data();
        }
        
        uint32_t* data() {
            return bits.data();
        }
        
        size_t data_size() const {
            return bits.size();
        }
        
        // 为了兼容性，保留原始接口但加上警告
        [[deprecated("Use mark_deleted instead")]]
        void reset(size_t pos) {
            mark_deleted(pos);
        }
        
        [[deprecated("Use mark_valid instead")]]
        void set(size_t pos) {
            mark_valid(pos);
        }
        
        // [[deprecated("Use is_valid instead")]]
        // bool test(size_t pos) const {
        //     return is_valid(pos);
        // }
        
        [[deprecated("Use mark_all_deleted instead")]]
        void reset_all() {
            mark_all_deleted();
        }
        
        [[deprecated("Use mark_all_valid instead")]]
        void set_all() {
            mark_all_valid();
        }
        
        [[deprecated("Use mark_range_deleted instead")]]
        void set(size_t start_pos, size_t end_pos) {
            mark_range_deleted(start_pos, end_pos);
        }
    };

}  // end namespace ffanns::core
