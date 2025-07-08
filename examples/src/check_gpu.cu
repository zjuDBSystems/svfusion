#include <raft/core/device_resources.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <thread>

void check_gpu_memory(int device_id) {
    // 设置当前设备
    RAFT_CUDA_TRY(cudaSetDevice(device_id));
    
    // 创建 CUDA stream
    cudaStream_t stream;
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));
    
    // 初始化设备资源
    std::shared_ptr<raft::device_resources> dev_resources = 
        std::make_shared<raft::device_resources>(
            rmm::cuda_stream_view(stream),  // 使用创建的 stream
            std::make_shared<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>>(
                rmm::mr::get_current_device_resource(),
                1024 * 1024 * 1024ull  // 1GB pool
            )
        );
    
    // 获取内存信息
    size_t free_memory, total_memory;
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
    
    std::cout << "GPU " << device_id << ":" << std::endl;
    std::cout << "  Total memory: " << (total_memory / (1024*1024*1024.0)) << " GB" << std::endl;
    std::cout << "  Free memory:  " << (free_memory / (1024*1024*1024.0)) << " GB" << std::endl;
    std::cout << "  Used memory:  " << ((total_memory - free_memory) / (1024*1024*1024.0)) << " GB" << std::endl;
    std::cout << std::endl;
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    // 清理
    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
}

int main() {
    try {
        // 获取系统中的 GPU 数量
        int device_count;
        RAFT_CUDA_TRY(cudaGetDeviceCount(&device_count));
        std::cout << "Found " << device_count << " CUDA devices\n\n";

        // 遍历每个 GPU
        for (int i = 0; i < device_count; i++) {
            // 获取 GPU 属性
            cudaDeviceProp prop;    
            RAFT_CUDA_TRY(cudaGetDeviceProperties(&prop, i));
            std::cout << "Device " << i << ": " << prop.name << std::endl;
            
            // 检查内存
            check_gpu_memory(i);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
