// yaml_benchmark.cu 或 修改现有的 benchmark.cu
#include "workload_manager.hpp"
#include <raft/core/device_resources.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/tracking_resource_adaptor.hpp>
#include <iostream>

std::string getArgValue(int argc, char** argv, const std::string& option, const std::string& defaultValue) {
    std::string prefix = option + "=";
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.substr(0, prefix.length()) == prefix) {
            return arg.substr(prefix.length());
        }
    }
    return defaultValue;
}

int main(int argc, char** argv) {
    std::string config_file = "/data/workspace/streaming_anns/runbooks/streaming_test.yaml";
    int gpu_id = 0;
    std::string data_type = "float";   

    if (argc > 1) {
        // 处理帮助选项
        if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
            std::cout << "用法: " << argv[0] << " [-config_file=<文件路径>] [-gpu_id=<设备ID>]" << std::endl;
            std::cout << "  -config_file  YAML配置文件的路径 (默认: default_config.yaml)" << std::endl;
            std::cout << "  -gpu_id       要使用的GPU设备ID (默认: 0)" << std::endl;
            std::cout << "  -data_type    数据类型，支持 'float' 或 'uint8' (默认: float)" << std::endl;
            return 0;
        }
        
        // 获取配置文件和GPU ID
        config_file = getArgValue(argc, argv, "-config_file", config_file);
        
        std::string gpu_id_str = getArgValue(argc, argv, "-gpu_id", std::to_string(gpu_id));
        try {
            gpu_id = std::stoi(gpu_id_str);
        } catch (const std::exception& e) {
            std::cerr << "无效的GPU ID: " << gpu_id_str << std::endl;
            return 1;
        }
        data_type = getArgValue(argc, argv, "-data_type", "float");
    }
    
    std::cout << "使用配置文件: " << config_file << std::endl;
    std::cout << "使用GPU ID: " << gpu_id << std::endl;
    std::cout << "使用数据类型: " << data_type << std::endl;
    
    // 初始化CUDA资源
    raft::device_resources dev_resources;
    RAFT_CUDA_TRY(cudaSetDevice(gpu_id));  // 这个也可以设置为可配置的
    
    // 创建CUDA流
    cudaStream_t stream;
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    // 设置内存池资源
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
        rmm::mr::get_current_device_resource(), 50 * 1024 * 1024 * 1024ull, 55 * 1024 * 1024 * 1024ull);
    rmm::mr::tracking_resource_adaptor<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>> 
        tracking_mr(&pool_mr, false);
    rmm::mr::set_current_device_resource(&tracking_mr);
    
    size_t n_streams = 4;
    raft::resource::set_cuda_stream_pool(dev_resources, std::make_shared<rmm::cuda_stream_pool>(n_streams));
    std::filesystem::path yaml_path(config_file);
    std::string workload_tag = yaml_path.filename().string();
    ffanns::neighbors::bench_config::instance().set_workload_tag(workload_tag);

    try {
        if (data_type == "float" || data_type == "float32") {
            ffanns::test::WorkloadManager<float> workload_manager(config_file);
            workload_manager.execute(dev_resources);
        } else if (data_type == "uint8" || data_type == "uint8_t") {
            ffanns::test::WorkloadManager<uint8_t> workload_manager(config_file);
            workload_manager.execute(dev_resources);
        } else {
            std::cerr << "不支持的数据类型: " << data_type << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
    return 0;
}