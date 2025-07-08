#pragma once

#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>   // get_device_for_address
#include <raft/core/error.hpp>
#include <raft/core/serialize.hpp>

#include <rmm/device_buffer.hpp>
// for serialize
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <thrust/uninitialized_fill.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>  
#include <thrust/sequence.h>

#include <fstream>
#include <vector>
#include <cuda/atomic> 
#include <random>
#include <unordered_set>
#include <algorithm>


namespace ffanns::neighbors {

using raft::RAFT_NAME;
#define HD_MAPPER_INVALID_ID UINT32_MAX

struct device_partition {
    uint32_t start_slot;     // 分区起始位置
    uint32_t size;           // 分区大小
    uint32_t clock_hand;     // 分区私有时钟
};

inline RAFT_KERNEL decay_kernel(float* data, float alpha, size_t n) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= alpha;
    }
}

 // TODO: batch_insert_kernel add atomicity
inline RAFT_KERNEL batch_insert_kernel(
    uint32_t* host_device_mapping,
    uint32_t* device_host_mapping,
    uint32_t* ref_bits,
    device_partition* partitions,
    uint32_t start_partition_num,
    size_t initial_dataset_size,
    size_t num_new_nodes,
    uint32_t* start_slot,
    size_t device_capacity)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_new_nodes) {
        uint32_t new_host_id = initial_dataset_size + tid;
        // uint32_t current_slot = (*start_slot + tid) % device_capacity;
        uint32_t current_slot = *start_slot + tid;
        device_partition* partition = &partitions[start_partition_num];
        if (current_slot >= (partition->start_slot + partition->size)) {
            current_slot = partition->start_slot + ((current_slot - partition->start_slot) % partition->size);
        }
        
        // 直接更新映射关系
        uint32_t old_host_id = device_host_mapping[current_slot];
        host_device_mapping[old_host_id] = HD_MAPPER_INVALID_ID;
        device_host_mapping[current_slot] = new_host_id;
        host_device_mapping[new_host_id] = current_slot;
        ref_bits[current_slot] = 1;
    }
}

inline RAFT_KERNEL map_host_to_device_kernel(
    uint32_t* host_device_mapping,
    uint32_t* host_rows,
    uint32_t* device_rows,
    size_t num_rows)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    
    uint32_t host_row = host_rows[idx];
    
    // 使用mapper获取设备行索引
    uint32_t mapping_result = host_device_mapping[host_row];
    device_rows[idx] = mapping_result;
}

inline RAFT_KERNEL find_victim_in_partition_kernel(
    uint32_t* ref_bits,
    device_partition* partition,
    uint32_t* result) 
{
    while (true) {
        // printf("[find_victim_in_partition] clock_hand: %u\n", partition->clock_hand);
        uint32_t local_slot = atomicAdd(&partition->clock_hand, 1) % partition->size;
        uint32_t global_slot = partition->start_slot + local_slot;

        uint32_t current_ref = ref_bits[global_slot];
        if (current_ref == 0U) {
            if (atomicCAS(&ref_bits[global_slot], 0U, 1U) == 0U) {
                // 成功获取该槽位
                *result = global_slot;
                return;
            }
        } else {
            // 如果当前槽位引用位为 1，则清零以给予“第二次机会”
            atomicExch(&ref_bits[global_slot], 0U);
        }
    }
}

// GPU kernel：针对每个被删除的 host id，更新 host_device_mapping 与 device_host_mapping
inline RAFT_KERNEL reset_mapper_kernel(uint32_t* d_host_device_mapping,
                                        uint32_t* d_device_host_mapping,
                                        const uint32_t* delete_list,
                                        size_t delete_count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < delete_count) {
        uint32_t host_id = delete_list[idx];
        uint32_t slot = d_host_device_mapping[host_id];
        if (slot != HD_MAPPER_INVALID_ID) {
            // printf("[reset_mapper_kernel] Resetting host_id: %u, slot: %u\n", host_id, slot);
            d_device_host_mapping[slot] = HD_MAPPER_INVALID_ID;
            d_host_device_mapping[host_id] = HD_MAPPER_INVALID_ID;
        }
    }
}

inline __global__ void test_kernel(const uint32_t* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 20 && idx < size) {
        printf("data[%zu] = %u\n", idx, data[idx]);
    }
}

struct host_device_mapper {
    static constexpr uint32_t INVALID_ID = UINT32_MAX;
    static constexpr uint32_t LOADING_ID = UINT32_MAX - 1;
    static constexpr uint32_t NUM_PARTITIONS = 32;
    static constexpr uint32_t NUM_PARTITION_BITS = 5;
    static constexpr uint32_t HASH_CONSTANT = 2654435761U;
    static constexpr float alpha = 0.0037;
    static constexpr float beta = 0.1151;
    static_assert((NUM_PARTITIONS & (NUM_PARTITIONS - 1)) == 0, 
                  "NUM_PARTITIONS must be power of 2");
    // data on device
    rmm::device_uvector<uint32_t> host_device_mapping;  
    rmm::device_uvector<uint32_t> device_host_mapping;   
    rmm::device_uvector<uint32_t> ref_bits;                 
    rmm::device_uvector<device_partition> partitions;   
    rmm::device_uvector<uint32_t> access_counts; 
    rmm::device_uvector<float> recent_access; 

    // device pointers
    uint32_t* d_host_device_mapping;
    uint32_t* d_device_host_mapping;
    uint32_t* d_ref_bits;
    device_partition* d_partitions;
    uint32_t* d_access_counts;
    float* d_recent_access;

    size_t device_capacity;
    size_t current_size;

    host_device_mapper(raft::resources const& res)
        : host_device_mapping(0, raft::resource::get_cuda_stream(res))
        , device_host_mapping(0, raft::resource::get_cuda_stream(res))
        , ref_bits(0, raft::resource::get_cuda_stream(res))
        , partitions(0, raft::resource::get_cuda_stream(res))
        , access_counts(0, raft::resource::get_cuda_stream(res))
        , recent_access(0, raft::resource::get_cuda_stream(res))
        , device_capacity(0)
        , current_size(0)
    {

    }

    host_device_mapper(raft::resources const& res,
                      size_t host_max_vectors,
                      size_t device_max_vectors) 
        : host_device_mapping(host_max_vectors, raft::resource::get_cuda_stream(res))
        , device_host_mapping(device_max_vectors, raft::resource::get_cuda_stream(res))
        , ref_bits(device_max_vectors, raft::resource::get_cuda_stream(res))
        , partitions(NUM_PARTITIONS, raft::resource::get_cuda_stream(res))
        , access_counts(host_max_vectors, raft::resource::get_cuda_stream(res))
        , recent_access(host_max_vectors, raft::resource::get_cuda_stream(res))
        , device_capacity(device_max_vectors)
        , current_size(0)
    {
        auto stream = raft::resource::get_cuda_stream(res);

        thrust::fill(thrust::cuda::par.on(stream),
                                 host_device_mapping.begin(),
                                 host_device_mapping.end(),
                                 INVALID_ID);
                    
        thrust::fill(thrust::cuda::par.on(stream),
                                 device_host_mapping.begin(),
                                 device_host_mapping.end(),
                                 INVALID_ID);
                    
        thrust::fill(thrust::cuda::par.on(stream),
                                 ref_bits.begin(),
                                 ref_bits.end(),
                                 1);
                    
        thrust::fill(thrust::cuda::par.on(stream),
                                 access_counts.begin(),
                                 access_counts.end(),
                                 0);
                            
        thrust::fill(thrust::cuda::par.on(stream),
                                 recent_access.begin(),
                                 recent_access.end(),
                                 0.0f);

        initialize_partitions(res);

        d_host_device_mapping = host_device_mapping.data();
        d_device_host_mapping = device_host_mapping.data();
        d_ref_bits = ref_bits.data();
        d_partitions = partitions.data();
        d_access_counts = access_counts.data();
        d_recent_access = recent_access.data();
        
        RAFT_LOG_INFO("Created GPU mapper with host_size: %zu, device_size: %zu, "
                      "num_partitions: %u", 
                      host_max_vectors, device_max_vectors, NUM_PARTITIONS);
    }

    host_device_mapper(const host_device_mapper &) = delete;
    auto operator=(const host_device_mapper &) -> host_device_mapper & = delete;
    
private:
    void initialize_partitions(raft::resources const& res) {
        auto stream = raft::resource::get_cuda_stream(res);
        std::vector<device_partition> h_partitions(NUM_PARTITIONS);
        const uint32_t base_size = device_capacity / NUM_PARTITIONS;
        const uint32_t remainder = device_capacity % NUM_PARTITIONS;
        
        uint32_t current_start = 0;
        for (uint32_t i = 0; i < NUM_PARTITIONS; i++) {
            auto& p = h_partitions[i];
            p.start_slot = current_start;
            p.size = base_size + (i < remainder ? 1 : 0);
            p.clock_hand = 0;
            current_start += p.size;
        }
        
        assert(current_start == device_capacity && 
               "Partition allocation doesn't match device capacity");
        
        RAFT_CUDA_TRY(cudaMemcpyAsync(
            partitions.data(),
            h_partitions.data(),
            NUM_PARTITIONS * sizeof(device_partition),
            cudaMemcpyHostToDevice,
            stream));
    }

public:
    [[nodiscard]] auto is_full() noexcept -> bool {
        return current_size == device_capacity;
    }
    
    [[nodiscard]] auto host_device_mapping_size() noexcept -> size_t {
        return host_device_mapping.size();
    }
    
    [[nodiscard]] auto device_host_mapping_size() noexcept -> size_t {
        return device_host_mapping.size();
    }

    [[nodiscard]] host_device_mapper* dev_ptr(raft::resources const& res) const {
        host_device_mapper* d_mapper;
        // 在栈上创建临时对象
        host_device_mapper temp(res);
        auto stream = raft::resource::get_cuda_stream(res);
        
        // 只设置设备指针和必要数据
        temp.d_host_device_mapping = d_host_device_mapping;
        temp.d_device_host_mapping = d_device_host_mapping;
        temp.d_ref_bits = d_ref_bits;
        temp.d_partitions = d_partitions;
        temp.d_access_counts = d_access_counts;
        temp.d_recent_access = d_recent_access;
        temp.device_capacity = device_capacity;
        temp.current_size = current_size;
        
        // 分配设备内存并复制
        RAFT_CUDA_TRY(cudaMallocAsync(&d_mapper, sizeof(host_device_mapper), stream));
        RAFT_CUDA_TRY(cudaMemcpyAsync(d_mapper, &temp, sizeof(host_device_mapper),
                                    cudaMemcpyHostToDevice, stream));
        cudaStreamSynchronize(stream);
        
        // host_device_mapper verify(res);
        // RAFT_CUDA_TRY(cudaMemcpy(&verify, d_mapper, sizeof(host_device_mapper),
        //                     cudaMemcpyDeviceToHost));
        // RAFT_LOG_INFO("verify.d_access_counts: %p", verify.d_access_counts);
        return d_mapper;
    }

    static void free_dev_ptr(host_device_mapper* d_ptr, cudaStream_t stream) {
        if (d_ptr) {
            RAFT_CUDA_TRY(cudaFreeAsync(d_ptr, stream));
        }
    }

    __device__ uint32_t get_partition_id(uint32_t host_id) {
        return (host_id * HASH_CONSTANT) & (NUM_PARTITIONS - 1);
    }

    __device__ uint32_t find_victim_in_partition(device_partition* partition) {
        while (true) {
            // 原子获取当前clock_hand
            uint32_t local_slot = atomicAdd(&partition->clock_hand, 1) % partition->size;
            uint32_t global_slot = partition->start_slot + local_slot;

            uint32_t current_ref = d_ref_bits[global_slot];
            if (current_ref == 0U) {
                if (atomicCAS(&d_ref_bits[global_slot], 0U, 1U) == 0U) {
                    return global_slot;
                }
            } else {
                atomicExch(&d_ref_bits[global_slot], 0U);
            }
        }
    }

    __device__ uint32_t find_victim_withscore_in_partition(device_partition* partition, int* const in_edges) {
        uint32_t local_slot, global_slot, current_ref, candidate_host;
        
        // const uint32_t MAX_ATTEMPTS = min(partition->size * 2, 100u); // 设置合理上限
        const uint32_t MAX_ATTEMPTS = 100u;
        uint32_t attempts = 0;

        while (attempts++ < MAX_ATTEMPTS) {
            // 原子获取当前clock_hand
            local_slot = atomicAdd(&partition->clock_hand, 1) % partition->size;
            global_slot = partition->start_slot + local_slot;

            current_ref = d_ref_bits[global_slot];
            candidate_host = d_device_host_mapping[global_slot];

            //TODO: 有可能delete_lists转成free_lists的时候被device_host_mapping被释放了
            if (current_ref == 0U) {
                float recent = d_recent_access[candidate_host];
                int in_degree = in_edges[candidate_host];
                float score   = alpha * recent + beta * __logf(1.0f + (float)in_degree);
                // printf("[find_victim_withscore_in_partition] host_id: %u, recent: %f, in_degree: %d, score: %f\n", candidate_host, recent, in_degree, score);
                
                if ((score <= 0.8) && (atomicCAS(&d_ref_bits[global_slot], 0U, 1U) == 0U)) {
                    return global_slot;
                } 
            } else {
                atomicExch(&d_ref_bits[global_slot], 0U);
            }
        }

        return INVALID_ID;  
    }

    __device__ auto get(uint32_t host_id) -> thrust::pair<uint8_t, uint32_t> {
        // printf("[hd_mapper] Accessing host_id: %u\n", host_id);
        atomicAdd(d_access_counts + host_id, 1);
        
        while (true) {
            uint32_t device_slot = atomicAdd(&d_host_device_mapping[host_id], 0);
            if (device_slot == INVALID_ID) {
                // 尝试将INVALID_ID更新为LOADING_ID
                if (atomicCAS(&d_host_device_mapping[host_id], INVALID_ID, LOADING_ID) == INVALID_ID) {
                    // 当前线程负责加载数据
                    break;
                }
            } else if (device_slot == LOADING_ID) {
                __nanosleep(100);
                continue;
            } else {
                atomicExch(d_ref_bits + device_slot, 1);
                return {1, device_slot};
            }
        }

        uint32_t partition_id = get_partition_id(host_id);
        device_partition* partition = &d_partitions[partition_id];
        // find_victim_in_partition ensure the victim slot is atomic
        uint32_t victim_slot = find_victim_in_partition(partition);

        // 更新映射
        uint32_t old_host_id = d_device_host_mapping[victim_slot];
        atomicExch(&d_host_device_mapping[old_host_id], INVALID_ID);
        atomicExch(&d_device_host_mapping[victim_slot], host_id);
        atomicExch(&d_host_device_mapping[host_id], victim_slot); 

        return {0, victim_slot};
    }

    __device__ auto get_wo_replace_safe(uint32_t host_id, bool count_tag=false) -> thrust::pair<uint8_t, uint32_t> {
        if (count_tag)
            atomicAdd(d_access_counts + host_id, 1);
            atomicAdd(d_recent_access + host_id, 1.0f);
        uint32_t device_slot = d_host_device_mapping[host_id];
        if (device_slot == INVALID_ID) {
            return {0, INVALID_ID};
        } else {
            return {1, device_slot}; 
        }
    }
    
    __device__ auto get_wo_replace(uint32_t host_id, bool count_tag=false) -> thrust::pair<uint8_t, uint32_t> {
        // printf("[hd_mapper] Accessing host_id: %u\n", host_id);
        if (count_tag)
            atomicAdd(d_access_counts + host_id, 1);
            atomicAdd(d_recent_access + host_id, 1.0f);
        
        uint32_t device_slot = atomicAdd(&d_host_device_mapping[host_id], 0);
        if (device_slot == INVALID_ID) {
            return {0, INVALID_ID};
        } else if (device_slot == LOADING_ID) {
            return {0, INVALID_ID};
        } else {
            atomicExch(d_ref_bits + device_slot, 1);
            return {1, device_slot}; 
        }
    }

    __device__ auto replace(uint32_t host_id, int* const in_edges) -> thrust::pair<uint8_t, uint32_t> {
        float recent = d_recent_access[host_id];
        int in_degree = in_edges[host_id];
        float score   = alpha * recent + beta * __logf(1.0f + (float)in_degree);
        if (score <= 0.85) {
            return {0, INVALID_ID};
        }

        uint32_t device_slot = INVALID_ID;
        while (true) {
            device_slot = atomicAdd(&d_host_device_mapping[host_id], 0);
            if (device_slot == INVALID_ID) {
                // 尝试将INVALID_ID更新为LOADING_ID
                if (atomicCAS(&d_host_device_mapping[host_id], INVALID_ID, LOADING_ID) == INVALID_ID) {
                    break;
                }
            } else if (device_slot == LOADING_ID) {
                __nanosleep(100);
                continue;
            } else {
                return {1, device_slot};
            }
        } 
        
        // printf("[get_wo_replace] host_id: %u, recent: %f, in_degree: %d, score: %f\n", host_id, recent, in_degree, score);
        uint32_t partition_id = get_partition_id(host_id);
        device_partition* partition = &d_partitions[partition_id];
        // find_victim_in_partition ensure the victim slot is atomic
        uint32_t victim_slot = find_victim_withscore_in_partition(partition, in_edges);

        if (victim_slot == INVALID_ID) {
            atomicExch(&d_host_device_mapping[host_id], victim_slot); 
            return {0, INVALID_ID};
        }

        // 更新映射
        uint32_t old_host_id = d_device_host_mapping[victim_slot];
        atomicExch(&d_host_device_mapping[old_host_id], INVALID_ID);
        atomicExch(&d_device_host_mapping[victim_slot], host_id);
        atomicExch(&d_host_device_mapping[host_id], victim_slot); 
        return {1, victim_slot};
        
    }

    void batch_insert(
        size_t initial_dataset_size, 
        size_t initial_d_dataset_size,
        size_t num_new_nodes,
        raft::resources const& res, bool is_full) noexcept  
    {
        if (!is_full)
            assert(initial_d_dataset_size == current_size);
        auto stream = raft::resource::get_cuda_stream(res);
        host_device_mapping.resize(initial_dataset_size + num_new_nodes, stream);
        access_counts.resize(initial_dataset_size + num_new_nodes, stream);
        recent_access.resize(initial_dataset_size + num_new_nodes, stream);
        d_host_device_mapping = host_device_mapping.data();
        d_access_counts = access_counts.data();
        d_recent_access = recent_access.data();
                    
        thrust::fill(thrust::cuda::par.on(stream),
                                access_counts.begin() + initial_dataset_size,
                                access_counts.end(),
                                0);

        thrust::fill(thrust::cuda::par.on(stream),
                                recent_access.begin() + initial_dataset_size,
                                recent_access.end(),
                                0.0f);

        if (is_full) {
            thrust::fill(thrust::cuda::par.on(stream),
                host_device_mapping.begin() + initial_dataset_size,
                host_device_mapping.begin() + initial_dataset_size + num_new_nodes,
                INVALID_ID);
        } else {
            thrust::sequence(thrust::cuda::par.on(stream),
                        host_device_mapping.begin() + initial_dataset_size,
                        host_device_mapping.begin() + initial_dataset_size + num_new_nodes,
                        initial_d_dataset_size);
            thrust::sequence(thrust::cuda::par.on(stream),
                        device_host_mapping.begin() + initial_d_dataset_size,
                        device_host_mapping.begin() + initial_d_dataset_size + num_new_nodes,
                        initial_dataset_size);
        // uint32_t insert_offset = current_size;
            current_size += num_new_nodes;
        }
    }

    void batch_insert_free_slots(
        size_t free_slots_pos, 
        size_t num_new_nodes,
        raft::resources const& res,
        bool is_full, size_t initial_dataset_size) noexcept  
    {
        assert((free_slots_pos + num_new_nodes) <= current_size);
        auto stream = raft::resource::get_cuda_stream(res);
                    
        thrust::fill(thrust::cuda::par.on(stream),
                                access_counts.begin() + free_slots_pos,
                                access_counts.begin() + free_slots_pos + num_new_nodes,
                                0);

        thrust::fill(thrust::cuda::par.on(stream),
                                recent_access.begin() + free_slots_pos,
                                recent_access.begin() + free_slots_pos + num_new_nodes,
                                0.0f);
        if (!is_full) {
            thrust::sequence(thrust::cuda::par.on(stream),
                host_device_mapping.begin() + free_slots_pos,
                host_device_mapping.begin() + free_slots_pos + num_new_nodes,
                initial_dataset_size);
            
            // device->host映射：从device物理位置映射回free_slots位置
            thrust::sequence(thrust::cuda::par.on(stream),
                device_host_mapping.begin() + initial_dataset_size,
                device_host_mapping.begin() + initial_dataset_size + num_new_nodes,
                free_slots_pos);
            current_size += num_new_nodes;
        }
    }

    void reset_mapper(raft::resources const& res, const uint32_t* delete_list, size_t delete_count) {
        RAFT_LOG_INFO("[vector reset_mapper] delete_list.size: %zu", delete_count);
        auto stream = raft::resource::get_cuda_stream(res);

        constexpr uint32_t BLOCK_SIZE = 512;
        uint32_t gridSize = (delete_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reset_mapper_kernel<<<gridSize, BLOCK_SIZE, 0, stream>>>(
            d_host_device_mapping,
            d_device_host_mapping,
            delete_list,
            delete_count);
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    }

    void decay_recent_access(float alpha, raft::resources const& res) {
        auto stream = raft::resource::get_cuda_stream(res);
        const int blockSize = 256;
        const int gridSize = (host_device_mapping_size() + blockSize - 1) / blockSize;
        
        decay_kernel<<<gridSize, blockSize, 0, stream>>>(d_recent_access, alpha, host_device_mapping_size());
    }

    void snapshot_access_counts(const std::string& filename, const int* in_edges, const int* d_in_edges, cudaStream_t cuda_stream) const {
            std::vector<uint32_t> h_access_counts(access_counts.size());
            std::vector<float> h_recent_access(recent_access.size());
            raft::copy(h_access_counts.data(), access_counts.data(), access_counts.size(), cuda_stream);
            raft::copy(h_recent_access.data(), recent_access.data(), recent_access.size(), cuda_stream);

            std::vector<int> h_in_edges(access_counts.size());
            raft::copy(h_in_edges.data(), d_in_edges, access_counts.size(), cuda_stream);

            std::vector<float> scores(access_counts.size());
            for (size_t host_id = 0; host_id < h_access_counts.size(); ++host_id) {
                float recent = h_recent_access[host_id];
                int in_degree = h_in_edges[host_id];
                scores[host_id] = alpha * recent + beta * logf(1.0f + (float)in_degree);
            }

            std::ofstream ofs(filename);
            if (!ofs.is_open()) {
                std::cerr << "Failed to open file: " << filename << std::endl;
                return;
            }
            if (in_edges != nullptr) {
                ofs << "host_id,access_count,in_edges,recent_access,d_in_edges,score\n";
                for (size_t host_id = 0; host_id < h_access_counts.size(); ++host_id) {
                    ofs << host_id << "," << h_access_counts[host_id] << "," << in_edges[host_id] << "," 
                    << h_recent_access[host_id] << "," << h_in_edges[host_id] << ","
                    << scores[host_id] << "\n";
                }
            } else {
                ofs << "host_id,access_count,recent_access,d_in_edges,score\n";
                for (size_t host_id = 0; host_id < h_access_counts.size(); ++host_id) {
                    ofs << host_id << "," << h_access_counts[host_id] << ","  
                    << h_recent_access[host_id] << "," << h_in_edges[host_id] << ","
                    << scores[host_id] << "\n";
                }
            }
            
            ofs.close();
            std::cout << "Snapshot of access_counts written to " << filename << std::endl;
    }
};

struct graph_hd_mapper {
    static constexpr uint32_t INVALID_ID = UINT32_MAX;
    static constexpr uint32_t LOADING_ID = UINT32_MAX - 1;
    static constexpr uint32_t NUM_PARTITIONS = 8;
    static constexpr uint32_t NUM_PARTITION_BITS = 3;
    static constexpr uint32_t HASH_CONSTANT = 2654435761U;
    static_assert((NUM_PARTITIONS & (NUM_PARTITIONS - 1)) == 0, 
                  "NUM_PARTITIONS must be power of 2");
    // data on device
    rmm::device_uvector<uint32_t> host_device_mapping;  
    rmm::device_uvector<uint32_t> device_host_mapping;   
    rmm::device_uvector<uint32_t> ref_bits;                 
    rmm::device_uvector<device_partition> partitions;   
    rmm::device_uvector<uint32_t> access_counts; 

    // device pointers
    uint32_t* d_host_device_mapping;
    uint32_t* d_device_host_mapping;
    uint32_t* d_ref_bits;
    device_partition* d_partitions;
    uint32_t* d_access_counts;

    size_t device_capacity;
    size_t current_size;

    graph_hd_mapper(raft::resources const& res)
        : host_device_mapping(0, raft::resource::get_cuda_stream(res))
        , device_host_mapping(0, raft::resource::get_cuda_stream(res))
        , ref_bits(0, raft::resource::get_cuda_stream(res))
        , partitions(0, raft::resource::get_cuda_stream(res))
        , access_counts(0, raft::resource::get_cuda_stream(res))
        , device_capacity(0)
        , current_size(0)
    {

    }

    graph_hd_mapper(raft::resources const& res,
                      size_t host_max_nodes,
                      size_t device_max_nodes) 
        : host_device_mapping(host_max_nodes, raft::resource::get_cuda_stream(res))
        , device_host_mapping(device_max_nodes, raft::resource::get_cuda_stream(res))
        , ref_bits(device_max_nodes, raft::resource::get_cuda_stream(res))
        , partitions(NUM_PARTITIONS, raft::resource::get_cuda_stream(res))
        , access_counts(host_max_nodes, raft::resource::get_cuda_stream(res))
        , device_capacity(device_max_nodes)
        , current_size(0)
    {
        auto stream = raft::resource::get_cuda_stream(res);

        thrust::fill(thrust::cuda::par.on(stream),
                                 host_device_mapping.begin(),
                                 host_device_mapping.end(),
                                 INVALID_ID);
                    
        thrust::fill(thrust::cuda::par.on(stream),
                                 device_host_mapping.begin(),
                                 device_host_mapping.end(),
                                 INVALID_ID);
                    
        thrust::fill(thrust::cuda::par.on(stream),
                                 ref_bits.begin(),
                                 ref_bits.end(),
                                 1);
                    
        thrust::fill(thrust::cuda::par.on(stream),
                                 access_counts.begin(),
                                 access_counts.end(),
                                 0);

        initialize_partitions(res);

        d_host_device_mapping = host_device_mapping.data();
        d_device_host_mapping = device_host_mapping.data();
        d_ref_bits = ref_bits.data();
        d_partitions = partitions.data();
        d_access_counts = access_counts.data();
        
        RAFT_LOG_INFO("Created graph mapper with host_size: %zu, device_size: %zu, "
                      "num_partitions: %u", 
                      host_max_nodes, device_max_nodes, NUM_PARTITIONS);
    }

    graph_hd_mapper(const graph_hd_mapper &) = delete;
    auto operator=(const graph_hd_mapper &) -> graph_hd_mapper & = delete;
    
private:
    void initialize_partitions(raft::resources const& res) {
        auto stream = raft::resource::get_cuda_stream(res);
        std::vector<device_partition> h_partitions(NUM_PARTITIONS);
        const uint32_t base_size = device_capacity / NUM_PARTITIONS;
        const uint32_t remainder = device_capacity % NUM_PARTITIONS;
        
        uint32_t current_start = 0;
        for (uint32_t i = 0; i < NUM_PARTITIONS; i++) {
            auto& p = h_partitions[i];
            p.start_slot = current_start;
            p.size = base_size + (i < remainder ? 1 : 0);
            p.clock_hand = 0;
            current_start += p.size;
        }
        
        assert(current_start == device_capacity && 
               "Partition allocation doesn't match device capacity");
        
        RAFT_CUDA_TRY(cudaMemcpyAsync(
            partitions.data(),
            h_partitions.data(),
            NUM_PARTITIONS * sizeof(device_partition),
            cudaMemcpyHostToDevice,
            stream));
    }

public:
    [[nodiscard]] auto is_full() noexcept -> bool {
        return current_size == device_capacity;
    }
    
    [[nodiscard]] auto host_device_mapping_size() noexcept -> size_t {
        return host_device_mapping.size();
    }
    
    [[nodiscard]] auto device_host_mapping_size() noexcept -> size_t {
        return device_host_mapping.size();
    }

    [[nodiscard]] graph_hd_mapper* dev_ptr(raft::resources const& res) const {
        graph_hd_mapper* d_mapper;
        // 在栈上创建临时对象
        graph_hd_mapper temp(res);
        auto stream = raft::resource::get_cuda_stream(res);
        
        // 只设置设备指针和必要数据
        temp.d_host_device_mapping = d_host_device_mapping;
        temp.d_device_host_mapping = d_device_host_mapping;
        temp.d_ref_bits = d_ref_bits;
        temp.d_partitions = d_partitions;
        temp.d_access_counts = d_access_counts;
        temp.device_capacity = device_capacity;
        temp.current_size = current_size;
        
        // 分配设备内存并复制
        RAFT_CUDA_TRY(cudaMallocAsync(&d_mapper, sizeof(graph_hd_mapper), stream));
        RAFT_CUDA_TRY(cudaMemcpyAsync(d_mapper, &temp, sizeof(graph_hd_mapper),
                                    cudaMemcpyHostToDevice, stream));
        cudaStreamSynchronize(stream);
        return d_mapper;
    }

    static void free_dev_ptr(graph_hd_mapper* d_ptr, cudaStream_t stream) {
        if (d_ptr) {
            RAFT_CUDA_TRY(cudaFreeAsync(d_ptr, stream));
        }
    }

    __device__ uint32_t get_partition_id(uint32_t host_id) {
        return (host_id * HASH_CONSTANT) & (NUM_PARTITIONS - 1);
    }

    __device__ uint32_t find_victim_in_partition(device_partition* partition) {
        while (true) {
            // 原子获取当前clock_hand
            uint32_t local_slot = atomicAdd(&partition->clock_hand, 1) % partition->size;
            uint32_t global_slot = partition->start_slot + local_slot;

            uint32_t current_ref = d_ref_bits[global_slot];
            if (current_ref == 0U) {
                if (atomicCAS(&d_ref_bits[global_slot], 0U, 1U) == 0U) {
                    return global_slot;
                }
            } else {
                atomicExch(&d_ref_bits[global_slot], 0U);
            }
        }
    }

    __device__ auto get(uint32_t host_id) -> thrust::pair<uint8_t, uint32_t> {
        // printf("[hd_mapper] Accessing host_id: %u\n", host_id);
        atomicAdd(d_access_counts + host_id, 1);
        
        while (true) {
            uint32_t device_slot = atomicAdd(&d_host_device_mapping[host_id], 0);
            if (device_slot == INVALID_ID) {
                // 尝试将INVALID_ID更新为LOADING_ID
                if (atomicCAS(&d_host_device_mapping[host_id], INVALID_ID, LOADING_ID) == INVALID_ID) {
                    // 当前线程负责加载数据
                    break;
                }
            } else if (device_slot == LOADING_ID) {
                __nanosleep(100);
                continue;
            } else {
                atomicExch(d_ref_bits + device_slot, 1);
                return {1, device_slot};
            }
        }

        uint32_t partition_id = get_partition_id(host_id);
        device_partition* partition = &d_partitions[partition_id];
        // find_victim_in_partition ensure the victim slot is atomic
        uint32_t victim_slot = find_victim_in_partition(partition);

        // 更新映射
        uint32_t old_host_id = d_device_host_mapping[victim_slot];
        atomicExch(&d_host_device_mapping[old_host_id], INVALID_ID);
        atomicExch(&d_device_host_mapping[victim_slot], host_id);
        atomicExch(&d_host_device_mapping[host_id], victim_slot); 

        return {0, victim_slot};
    }

    void batch_insert(
        size_t initial_dataset_size, 
        size_t initial_d_graph_size,
        size_t num_new_nodes,
        raft::resources const& res, bool is_full) noexcept  
    {
        if (!is_full)
            assert(initial_d_graph_size == current_size);
        auto stream = raft::resource::get_cuda_stream(res);
        host_device_mapping.resize(initial_dataset_size + num_new_nodes, stream);
        access_counts.resize(initial_dataset_size + num_new_nodes, stream);
        // recent_access.resize(initial_dataset_size + num_new_nodes, stream);
        d_host_device_mapping = host_device_mapping.data();
        d_access_counts = access_counts.data();
        // d_recent_access = recent_access.data();
                    
        thrust::fill(thrust::cuda::par.on(stream),
                                access_counts.begin() + initial_dataset_size,
                                access_counts.end(),
                                0);

        // thrust::fill(thrust::cuda::par.on(stream),
        //                         recent_access.begin() + initial_dataset_size,
        //                         recent_access.end(),
        //                         0.0f);

        if (is_full) {
            thrust::fill(thrust::cuda::par.on(stream),
                host_device_mapping.begin() + initial_dataset_size,
                host_device_mapping.begin() + initial_dataset_size + num_new_nodes,
                INVALID_ID);
        } else {
            thrust::sequence(thrust::cuda::par.on(stream),
                        host_device_mapping.begin() + initial_dataset_size,
                        host_device_mapping.begin() + initial_dataset_size + num_new_nodes,
                        initial_d_graph_size);
            thrust::sequence(thrust::cuda::par.on(stream),
                        device_host_mapping.begin() + initial_d_graph_size,
                        device_host_mapping.begin() + initial_d_graph_size + num_new_nodes,
                        initial_dataset_size);
        // uint32_t insert_offset = current_size;
            current_size += num_new_nodes;
        }
    }

    void batch_insert_free_slots(
        size_t free_slots_pos, 
        size_t num_new_nodes,
        raft::resources const& res,
        bool is_full, size_t initial_dataset_size) noexcept  
    {
        assert((free_slots_pos + num_new_nodes) <= current_size);
        auto stream = raft::resource::get_cuda_stream(res);
                    
        thrust::fill(thrust::cuda::par.on(stream),
                                access_counts.begin() + free_slots_pos,
                                access_counts.begin() + free_slots_pos + num_new_nodes,
                                0);

        if (!is_full) {
            thrust::sequence(thrust::cuda::par.on(stream),
                host_device_mapping.begin() + free_slots_pos,
                host_device_mapping.begin() + free_slots_pos + num_new_nodes,
                initial_dataset_size);
            
            // device->host映射：从device物理位置映射回free_slots位置
            thrust::sequence(thrust::cuda::par.on(stream),
                device_host_mapping.begin() + initial_dataset_size,
                device_host_mapping.begin() + initial_dataset_size + num_new_nodes,
                free_slots_pos);
            current_size += num_new_nodes;
        }
    }

    [[nodiscard]] auto batch_replace_insert(
        size_t initial_dataset_size, 
        size_t num_new_nodes,
        raft::resources const& res) noexcept -> uint32_t 
    {
        // RAFT_LOG_INFO("[batch_insert] d_access_counts: %p", d_access_counts);
        auto stream = raft::resource::get_cuda_stream(res);
        host_device_mapping.resize(initial_dataset_size + num_new_nodes, stream);
        access_counts.resize(initial_dataset_size + num_new_nodes, stream);
        d_host_device_mapping = host_device_mapping.data();
        d_access_counts = access_counts.data();

        thrust::fill(thrust::cuda::par.on(stream),
                             host_device_mapping.begin() + initial_dataset_size,
                             host_device_mapping.end(),
                             INVALID_ID);
                    
        thrust::fill(thrust::cuda::par.on(stream),
                                access_counts.begin() + initial_dataset_size,
                                access_counts.end(),
                                0);
        
        constexpr uint32_t BLOCK_SIZE = 128;  // 或 128，具体值可以根据您的GPU调整
        const uint32_t num_blocks = (num_new_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 grid(num_blocks);

        uint32_t start_partition = (initial_dataset_size + 1) & (NUM_PARTITIONS - 1);
        device_partition* start_p = &d_partitions[start_partition];

        rmm::device_scalar<uint32_t> start_slot(stream);
        find_victim_in_partition_kernel<<<1, 1, 0, stream>>>(
            d_ref_bits,
            start_p,
            start_slot.data()
        );

        batch_insert_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            d_host_device_mapping,
            d_device_host_mapping,
            d_ref_bits,
            d_partitions,
            start_partition,
            initial_dataset_size,
            num_new_nodes,
            start_slot.data(),
            device_capacity
        );
        
        return start_slot.value(stream);
    }

    void map_host_to_device_rows(raft::resources const& res, 
                                const std::vector<uint32_t>& host_rows,
                                uint32_t* device_rows_out)
    {
        auto stream = raft::resource::get_cuda_stream(res);
        size_t num_rows = host_rows.size();
        rmm::device_uvector<uint32_t> d_host_rows(num_rows, stream);
        rmm::device_uvector<uint32_t> d_device_rows(num_rows, stream);
        raft::copy(d_host_rows.data(), host_rows.data(), num_rows, stream);

        constexpr uint32_t BLOCK_SIZE = 128;
        uint32_t gridSize = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        map_host_to_device_kernel<<<gridSize, BLOCK_SIZE, 0, stream>>>(
            d_host_device_mapping,
            d_host_rows.data(),
            d_device_rows.data(),
            num_rows);
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

        // std::vector<uint32_t> result_device_rows(num_rows);
        // raft::copy(result_device_rows.data(), d_device_rows.data(), num_rows, stream);
        raft::copy(device_rows_out, d_device_rows.data(), num_rows, stream);
        raft::resource::sync_stream(res);
        // return result_device_rows;
    }

    void reset_mapper(raft::resources const& res, const uint32_t* delete_list, size_t delete_count) {
        RAFT_LOG_INFO("[graph reset_mapper] delete_list.size: %zu", delete_count);
        auto stream = raft::resource::get_cuda_stream(res);

        constexpr uint32_t BLOCK_SIZE = 512;
        uint32_t gridSize = (delete_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reset_mapper_kernel<<<gridSize, BLOCK_SIZE, 0, stream>>>(
            d_host_device_mapping,
            d_device_host_mapping,
            delete_list,
            delete_count);
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    }

    void reset_for_consolidate(raft::resources const& res, size_t graph_size, size_t d_graph_size) {
        auto reset_size = std::min(graph_size, d_graph_size);
        RAFT_LOG_INFO("[graph eset_for_consolidate] reset_size: %zu", reset_size);
        auto stream = raft::resource::get_cuda_stream(res);

        // host side
        // acces counts no need to reset, as host does not change
        thrust::sequence(thrust::cuda::par.on(stream),
                    host_device_mapping.begin(),
                    host_device_mapping.begin() + reset_size,
                    0);
        auto host_device_mapping_size = host_device_mapping.size();
        if (graph_size > d_graph_size) {
            thrust::fill(thrust::cuda::par.on(stream),
                             host_device_mapping.begin() + reset_size,
                             host_device_mapping.begin() + graph_size,
                             INVALID_ID);
        }

        // device side
        thrust::fill(thrust::cuda::par.on(stream),
                    ref_bits.begin(),
                    ref_bits.begin() + d_graph_size,
                    1);
        thrust::sequence(thrust::cuda::par.on(stream),
                    device_host_mapping.begin(),
                    device_host_mapping.begin() + reset_size,
                    0);
        if (graph_size < d_graph_size) {
            thrust::fill(thrust::cuda::par.on(stream),
                             device_host_mapping.begin() + reset_size,
                             device_host_mapping.begin() + d_graph_size,
                             INVALID_ID);
        }
        
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    }
};

// void verify_data_consistency(
//     host_device_mapper &mapper,
//     raft::resources const &res,
//     raft::host_matrix_view<const float, int64_t> host_data_view, // CPU上所有的向量数据
//     raft::device_matrix_view<const float, int64_t> device_data_view,   // GPU上所有的向量数据
//     size_t num_vectors_to_check)                        // 检查的向量数量
// {
//     auto stream = raft::resource::get_cuda_stream(res);

//     // 获取主机上的向量总数
//     size_t total_vectors = host_data_view.extent(0);
//     size_t dim = host_data_view.extent(1);

//     // 创建一个包含所有向量ID的列表
//     std::vector<uint32_t> all_vector_ids(total_vectors);
//     std::iota(all_vector_ids.begin(), all_vector_ids.end(), 0);

//     std::unordered_set<uint32_t> vector_ids_set;

//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<uint32_t> dis(0, static_cast<uint32_t>(total_vectors - 1));

//     while (vector_ids_set.size() < num_vectors_to_check) {
//         uint32_t random_id = dis(gen);
//         vector_ids_set.insert(random_id);
//     }

//     std::vector<uint32_t> vector_ids_to_check(vector_ids_set.begin(), vector_ids_set.end());
    
//     rmm::device_uvector<float> d_temp_vector(dim, stream);
//     std::vector<float> h_temp_vector(dim);
//     std::vector<float> h_cpu_vector(dim);

//     // int inconsistencies = 0;

//     // for (uint32_t vector_id : vector_ids_to_check) {
//     // }
// }

} // namespace ffanns::neighbors