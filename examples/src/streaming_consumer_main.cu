// streaming_consumer_main.cu
#include <iostream>
#include <csignal>
#include <yaml-cpp/yaml.h>
#include <raft/core/device_resources.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/tracking_resource_adaptor.hpp>
#include "streaming_consumer_manager.hpp"

namespace ffanns {
namespace test {

// 全局指针用于信号处理器访问consumer（单机测试简化方案）
void* g_consumer_ptr = nullptr;

void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cout << "Received SIGINT, shutting down..." << std::endl;
        if (g_consumer_ptr != nullptr) {
            // 根据数据类型调用对应的stop方法
            static_cast<StreamingConsumerManager<float>*>(g_consumer_ptr)->stop();
        }
    }
}

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

} // namespace test
} // namespace ffanns

int main(int argc, char** argv) {
    using namespace ffanns::test;
    std::signal(SIGINT, signal_handler);
    
    std::string config_file = getArgValue(argc, argv, "--config", 
        "/data/workspace/svfusion/examples/runbooks/streaming_test.yaml");
    std::string data_type = getArgValue(argc, argv, "--data-type", "float");
    std::string brokers = getArgValue(argc, argv, "--brokers", "localhost:9092");
    std::string topic = getArgValue(argc, argv, "--topic", "vector-queries");
    
    std::cout << "Streaming Consumer Configuration:" << std::endl;
    std::cout << "  Config: " << config_file << std::endl;
    std::cout << "  Data type: " << data_type << std::endl;
    std::string control_topic = getArgValue(argc, argv, "--control-topic", "svfusion-control");
    std::cout << "  Kafka: " << brokers << "/" << topic << " (control: " << control_topic << ")" << std::endl;
    
    // 初始化CUDA资源（参考benchmark.cu）
    raft::device_resources dev_resources;
    RAFT_CUDA_TRY(cudaSetDevice(0));  // 单机测试，使用GPU 0
    
    // 设置内存池资源（consumer需要较少内存）
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
        rmm::mr::get_current_device_resource(), 20 * 1024 * 1024 * 1024ull, 25 * 1024 * 1024 * 1024ull);
    rmm::mr::tracking_resource_adaptor<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>> 
        tracking_mr(&pool_mr, false);
    rmm::mr::set_current_device_resource(&tracking_mr);
    
    // 设置流池
    size_t n_streams = 5;
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(n_streams);
    raft::resource::set_cuda_stream_pool(dev_resources, stream_pool);
    
    try {
        if (data_type == "float") {
            StreamingConsumerManager<float> consumer(config_file, brokers, topic, control_topic);
            g_consumer_ptr = &consumer;  // 设置全局指针供信号处理器使用

            // initialize会自动检测stream pool配置
            consumer.initialize(dev_resources, stream_pool);
            // consumer.initialize(dev_resources, nullptr);

            consumer.run_streaming_workload(dev_resources);
            consumer.stop();
            g_consumer_ptr = nullptr;  // 清理全局指针
        } else {
            std::cerr << "Unsupported data type: " << data_type << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Streaming consumer finished." << std::endl;
    return 0;
}
