#pragma once

#include "workload_manager.hpp"
#include <ffanns/neighbors/common.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/core/logger.hpp>
#include <raft/core/host_mdspan.hpp>

#include <iostream>
#include <chrono>
#include <thread>
#include <filesystem>

void generate_host_dataset(raft::device_resources const& dev_resources,
                         raft::host_matrix_view<float, int64_t> host_dataset) {
    auto device_dataset = raft::make_device_matrix<float, int64_t>(
        dev_resources, host_dataset.extent(0), host_dataset.extent(1));

    raft::random::RngState r(1234ULL);
    auto labels = raft::make_device_vector<int64_t, int64_t>(
        dev_resources, host_dataset.extent(0));
    raft::random::make_blobs(dev_resources, device_dataset.view(), labels.view());

    raft::copy(host_dataset.data_handle(),
              device_dataset.data_handle(),
              device_dataset.size(),
              dev_resources.get_stream());
}

template<typename IdxT>
void print_single_node(const raft::host_matrix_view<const IdxT, int64_t, raft::row_major>& graph_view,
                      size_t node_id,
                      size_t max_neighbors = 10) 
{
    // 检查 graph_view 是否有效
    RAFT_LOG_INFO("Graph dimensions: %ld x %ld", graph_view.extent(0), graph_view.extent(1));
    RAFT_LOG_INFO("Checking node %ld", node_id);

    // 检查输入参数
    if (graph_view.data_handle() == nullptr) {
        RAFT_LOG_ERROR("Invalid graph view: null data handle");
        return;
    }

    if (node_id >= graph_view.extent(0)) {
        RAFT_LOG_WARN("Node ID %ld exceeds graph size %ld", node_id, graph_view.extent(0));
        return;
    }

    // 直接访问 CPU 上的数据
    const IdxT* node_neighbors = graph_view.data_handle() + node_id * graph_view.extent(1);

    size_t neighbors_to_print = std::min(max_neighbors, static_cast<size_t>(graph_view.extent(1)));
    // 构建邻居字符串
    std::string neighbors_str = "";
    for(size_t i = 0; i < neighbors_to_print; i++) {
        neighbors_str += std::to_string(node_neighbors[i]);
        if (i < neighbors_to_print - 1) {
            neighbors_str += ", ";
        }
    }
    if(graph_view.extent(1) > max_neighbors) {
        neighbors_str += "...";
    }

    RAFT_LOG_INFO("Node %ld information:", node_id);
    RAFT_LOG_INFO("- Total neighbors: %ld", graph_view.extent(1));
    RAFT_LOG_INFO("- Neighbors: [%s]", neighbors_str.c_str());
}

template<typename IdxT>
void print_device_single_node(raft::resources const& res,
                            raft::device_matrix_view<IdxT, int64_t, raft::row_major>& graph_view,
                            size_t node_id,
                            size_t max_neighbors = 10) 
{
    // 检查 graph_view 是否有效
    RAFT_LOG_INFO("Device Graph dimensions: %ld x %ld", graph_view.extent(0), graph_view.extent(1));
    RAFT_LOG_INFO("Checking device node %ld", node_id);

    // 检查输入参数
    if (graph_view.data_handle() == nullptr) {
        RAFT_LOG_ERROR("Invalid device graph view: null data handle");
        return;
    }

    if (node_id >= graph_view.extent(0)) {
        RAFT_LOG_WARN("Node ID %ld exceeds graph size %ld", node_id, graph_view.extent(0));
        return;
    }

    // 确定要复制的数据大小
    size_t row_size = graph_view.extent(1);
    
    // 分配host内存
    rmm::device_uvector<IdxT> d_row(row_size, raft::resource::get_cuda_stream(res));
    std::vector<IdxT> h_row(row_size);

    // 复制指定行到临时device buffer
    RAFT_CUDA_TRY(cudaMemcpyAsync(
        d_row.data(),
        graph_view.data_handle() + node_id * row_size,
        row_size * sizeof(IdxT),
        cudaMemcpyDeviceToDevice,
        raft::resource::get_cuda_stream(res)));

    // 从device复制到host
    RAFT_CUDA_TRY(cudaMemcpyAsync(
        h_row.data(),
        d_row.data(),
        row_size * sizeof(IdxT),
        cudaMemcpyDeviceToHost,
        raft::resource::get_cuda_stream(res)));
    
    // 确保复制完成
    RAFT_CUDA_TRY(cudaStreamSynchronize(raft::resource::get_cuda_stream(res)));

    // 构建邻居字符串
    size_t neighbors_to_print = std::min(max_neighbors, static_cast<size_t>(row_size));
    std::string neighbors_str = "";
    for(size_t i = 0; i < neighbors_to_print; i++) {
        neighbors_str += std::to_string(h_row[i]);
        if (i < neighbors_to_print - 1) {
            neighbors_str += ", ";
        }
    }
    if(row_size > max_neighbors) {
        neighbors_str += "...";
    }

    RAFT_LOG_INFO("Device Node %ld information:", node_id);
    RAFT_LOG_INFO("- Total neighbors: %ld", row_size);
    RAFT_LOG_INFO("- Neighbors: [%s]", neighbors_str.c_str());
}

template <typename DataT>
std::pair<size_t, size_t> read_fbin_file(const std::string& filename, DataT* &query_data) {
    std::ifstream reader;
    reader.exceptions(std::ios::failbit | std::ios::badbit);
    std::pair<size_t, size_t> result{0, 0};  // 默认返回值
    
    try {
        reader.open(filename, std::ios::binary);
        RAFT_LOG_INFO("Reading bin file %s", filename.c_str());
        
        // 读取点数和维度
        int32_t npts_i32, ndims_i32;
        reader.read(reinterpret_cast<char*>(&npts_i32), sizeof(int32_t));
        reader.read(reinterpret_cast<char*>(&ndims_i32), sizeof(int32_t));
        
        size_t total_elements = static_cast<size_t>(npts_i32) * ndims_i32;
        RAFT_LOG_INFO("Total points: %d, dimensions: %d, total elements: %zu", 
            npts_i32, ndims_i32, total_elements);

        // 分配内存
        query_data = new DataT[total_elements];
        
        // 读取所有数据
        reader.read(reinterpret_cast<char*>(query_data), 
                   total_elements * sizeof(DataT));

        // 验证文件大小
        reader.seekg(0, std::ios::end);
        size_t file_size = reader.tellg();
        size_t expected_size = 2 * sizeof(int32_t) + total_elements * sizeof(DataT);
        
        if (file_size != expected_size) {
            RAFT_LOG_WARN("File size mismatch! Expected: %zu, Actual: %zu", 
                expected_size, file_size);
        }

        result = {static_cast<size_t>(npts_i32), static_cast<size_t>(ndims_i32)};

    } catch (const std::exception& e) {
        RAFT_LOG_ERROR("Error reading file: %s", e.what());
        throw;
    }

    if (reader.is_open()) {
        reader.close();
    }
    
    return result;  // 确保所有路径都有返回值
}

void print_search_results(raft::resources const& dev_resources,
                         const raft::device_matrix<uint32_t, int64_t>& neighbor_indices,
                         const raft::device_matrix<float, int64_t>& neighbor_distances,
                         int num_queries_to_print = 5,    // 打印前几个查询结果
                         int num_neighbors_to_print = 10)  // 每个查询打印前几个邻居
{
    // 分配host内存
    std::vector<uint32_t> h_indices(neighbor_indices.extent(0) * neighbor_indices.extent(1));
    std::vector<float> h_distances(neighbor_distances.extent(0) * neighbor_distances.extent(1));
    
    // 拷贝结果到host
    raft::copy(h_indices.data(), 
               neighbor_indices.data_handle(),
               neighbor_indices.size(),
               raft::resource::get_cuda_stream(dev_resources));
    raft::copy(h_distances.data(), 
               neighbor_distances.data_handle(),
               neighbor_distances.size(),
               raft::resource::get_cuda_stream(dev_resources));
    
    // 等待拷贝完成
    RAFT_CUDA_TRY(cudaStreamSynchronize(raft::resource::get_cuda_stream(dev_resources)));
    
    // 打印结果
    // num_queries_to_print = std::min(static_cast<int>(neighbor_indices.extent(0)), num_queries_to_print);
    // num_neighbors_to_print = std::min(static_cast<int>(neighbor_indices.extent(1)), num_neighbors_to_print);
    
    // for (int i = 0; i < num_queries_to_print; ++i) {
    //     RAFT_LOG_INFO("Query %d:", i);
    //     for (int j = 0; j < num_neighbors_to_print; ++j) {
    //         size_t idx = i * neighbor_indices.extent(1) + j;
    //         RAFT_LOG_INFO("    Neighbor %d: index = %u, distance = %.6f", 
    //                      j, h_indices[idx], h_distances[idx]);
    //     }
    //     RAFT_LOG_INFO("");  // 空行分隔不同查询的结果
    // }
}


void save_neighbors_to_binary(
    const std::vector<std::vector<uint32_t>>& step_neighbors,
    const ffanns::neighbors::bench_config& config)
{
    try {
        std::string filepath = config.get_search_path();
        filepath = filepath.substr(0, filepath.length() - 4) + ".bin";
        
        std::ofstream out(filepath, std::ios::binary);
        if (!out) {
            RAFT_LOG_ERROR("Failed to open file: %s", filepath.c_str());
            return;
        }

        // 写入文件头部信息
        uint32_t num_steps = step_neighbors.size();
        uint32_t topk = 10;
        uint32_t num_queries = step_neighbors[0].size() / topk;  // 10000
        
        out.write(reinterpret_cast<const char*>(&num_steps), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&num_queries), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&topk), sizeof(uint32_t));

        // 直接写入每个step的neighbors数组
        for (const auto& neighbors : step_neighbors) {
            out.write(reinterpret_cast<const char*>(neighbors.data()), 
                     neighbors.size() * sizeof(uint32_t));
        }

        RAFT_LOG_INFO("Saved %zu steps of neighbors to %s (shape: [%u, %u])", 
                     num_steps, filepath.c_str(), num_queries, topk);
        
        // 以下为新增功能：为每个 step 生成单独的 CSV 文件
        // 以 config.get_search_path() 为基准，去掉扩展名后加上 "_step<step编号>.csv"
        // std::string base_filepath = config.get_search_path();
        // size_t pos = base_filepath.find_last_of('.');
        // if (pos != std::string::npos) {
        //     base_filepath = base_filepath.substr(0, pos);
        // }
        
        // for (size_t step = 0; step < step_neighbors.size(); ++step) {
        //     std::string csv_filename = base_filepath + "_step" + std::to_string(step) + ".csv";
        //     std::ofstream csv_out(csv_filename);
        //     if (!csv_out) {
        //         RAFT_LOG_ERROR("Failed to open CSV file: %s", csv_filename.c_str());
        //         continue;
        //     }
            
        //     const std::vector<uint32_t>& neighbors = step_neighbors[step];
        //     // 检查 neighbors 的大小是否为 topk 的整数倍
        //     if (neighbors.size() % topk != 0) {
        //         RAFT_LOG_ERROR("Step %zu: neighbor count %zu is not a multiple of topk %u", 
        //                        step, neighbors.size(), topk);
        //         continue;
        //     }
        //     uint32_t num_queries_csv = neighbors.size() / topk;
        //     // 每行写入一个 query 对应的 topk 结果
        //     for (uint32_t q = 0; q < num_queries_csv; ++q) {
        //         for (uint32_t k = 0; k < topk; ++k) {
        //             csv_out << neighbors[q * topk + k];
        //             if (k < topk - 1) {
        //                 csv_out << ",";
        //             }
        //         }
        //         csv_out << "\n";
        //     }
        //     RAFT_LOG_INFO("Saved CSV for step %zu to %s (shape: [%u, %u])", 
        //                   step, csv_filename.c_str(), num_queries_csv, topk);
        // }
                     
    } catch (const std::exception& e) {
        RAFT_LOG_ERROR("Error saving to binary: %s", e.what());
        throw;
    }
}

void save_step_neighbors_to_binary(
    const std::vector<std::vector<uint32_t>>& step_neighbors,
    const int step_num,
    const ffanns::neighbors::bench_config& config)
{
    try {
        std::string filepath = config.get_search_path();
        filepath = filepath.substr(0, filepath.length() - 4) + "_step" + std::to_string(step_num) + ".bin";
        
        std::ofstream out(filepath, std::ios::binary);
        if (!out) {
            RAFT_LOG_ERROR("Failed to open file: %s", filepath.c_str());
            return;
        }

        // 写入文件头部信息
        uint32_t num_steps = step_neighbors.size();
        uint32_t topk = 10;
        uint32_t num_queries = step_neighbors[0].size() / topk;  // 10000
        
        out.write(reinterpret_cast<const char*>(&num_steps), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&num_queries), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&topk), sizeof(uint32_t));

        // 直接写入每个step的neighbors数组
        for (const auto& neighbors : step_neighbors) {
            out.write(reinterpret_cast<const char*>(neighbors.data()), 
                     neighbors.size() * sizeof(uint32_t));
        }

        RAFT_LOG_INFO("Saved %zu steps of neighbors to %s (shape: [%u, %u])", 
                     num_steps, filepath.c_str(), num_queries, topk);
                     
    } catch (const std::exception& e) {
        RAFT_LOG_ERROR("Error saving to binary: %s", e.what());
        throw;
    }
}

void log_step_time_csv(const ffanns::neighbors::bench_config& config,
                              int                       step,
                              const ffanns::test::OperationType    op,
                              double                    seconds,
                              bool                first_write = false,
                              float                  miss_rate = 0.0f)
{
    std::filesystem::path csv_path = config.get_time_log_path();
    std::ios_base::openmode mode =
        first_write ? std::ios::out               // 覆盖
                    : std::ios::app;              // 追加

    std::ofstream out(csv_path, mode);
    if (!out) {
        RAFT_LOG_ERROR("Cannot open time-log file: %s", csv_path.c_str());
        return;
    }

    if (first_write)                              // 只在第一次写表头
        out << "step,operation,time,miss_rate\n";

    static const char* op_name[] = { "insert", "search", "delete" };

    out << step << ','
        << op_name[static_cast<int>(op)] << ','
        << std::fixed << std::setprecision(12) << seconds << ','
        << std::fixed << std::setprecision(12) << miss_rate <<'\n';
}

void print_device_properties(const cudaDeviceProp& prop) {
    printf("\n===== 设备属性 =====\n");
    
    // 基本信息
    printf("设备名称: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    
    // 内存相关
    printf("全局内存大小: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("L2缓存大小: %d KB\n", prop.l2CacheSize / 1024);
    printf("共享内存大小/块: %lu KB\n", prop.sharedMemPerBlock / 1024);
    printf("常量内存大小: %lu KB\n", prop.totalConstMem / 1024);
    
    // 多处理器相关
    printf("SM数量: %d\n", prop.multiProcessorCount);
    printf("每个SM的最大线程数: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("每个块的最大线程数: %d\n", prop.maxThreadsPerBlock);
    
    // 线程维度限制
    printf("线程块维度限制: (%d, %d, %d)\n", 
        prop.maxThreadsDim[0], 
        prop.maxThreadsDim[1], 
        prop.maxThreadsDim[2]);
    
    printf("网格维度限制: (%d, %d, %d)\n", 
        prop.maxGridSize[0], 
        prop.maxGridSize[1], 
        prop.maxGridSize[2]);
    
    // 内存和缓存
    printf("每个块的寄存器数: %d\n", prop.regsPerBlock);
    printf("纹理对齐要求: %lu bytes\n", prop.textureAlignment);
    
    // 时钟频率
    printf("GPU时钟频率: %d MHz\n", prop.clockRate / 1000);
    printf("内存时钟频率: %d MHz\n", prop.memoryClockRate / 1000);
    
    // 内存总线
    printf("内存总线宽度: %d bits\n", prop.memoryBusWidth);
    
    // 其他特性
    printf("是否支持统一寻址: %s\n", prop.unifiedAddressing ? "是" : "否");
    printf("是否支持并发kernel: %s\n", prop.concurrentKernels ? "是" : "否");
    printf("异步引擎数量: %d\n", prop.asyncEngineCount);
    printf("ECC支持: %s\n", prop.ECCEnabled ? "开启" : "关闭");
    
    // Warp相关
    printf("Warp大小: %d\n", prop.warpSize);
    
    // 可以计算每个SM的最大常驻warp数
    int warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    printf("每个SM的最大常驻warp数: %d\n", warps_per_sm);
}
