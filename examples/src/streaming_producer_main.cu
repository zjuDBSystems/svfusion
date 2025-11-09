// streaming_producer_main.cu
#include <iostream>
#include <chrono>
#include <thread>
#include <csignal>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include <librdkafka/rdkafka.h>

#include "streaming_producer.hpp"
#include "workload_manager.hpp"
#include <raft/core/logger.hpp>

namespace ffanns {
namespace test {

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    if (signal == SIGINT) {
        RAFT_LOG_INFO("[signal_handler] Received SIGINT, shutting down...");
        g_running = false;
    }
}

static inline void cleanup_kafka_control(rd_kafka_topic_t* topic,
                                         rd_kafka_t* consumer,
                                         bool stop_consume) {
    if (topic) {
        if (stop_consume) {
            rd_kafka_consume_stop(topic, 0);
        }
        rd_kafka_topic_destroy(topic);
    }
    if (consumer) {
        rd_kafka_destroy(consumer);
    }
}

// 从YAML配置创建WorkloadPatternGenerator
template<typename DataT>
ffanns::test::WorkloadPatternGenerator<DataT> create_pattern_from_yaml(
    const std::string& config_file, const std::string& dataset_name) {
    
    ffanns::test::WorkloadPatternGenerator<DataT> generator(dataset_name);

    // 解析YAML配置
    YAML::Node config = YAML::LoadFile(config_file);
    auto dataset_node = config.begin();
    auto dataset_config = dataset_node->second;

    // 读取全局QPS设置（所有操作共享）
    double global_qps = dataset_config["qps"] ? dataset_config["qps"].as<double>() : 100.0;
    generator.set_global_qps(global_qps);
    RAFT_LOG_INFO("[create_pattern_from_yaml] Global QPS set to: %.1f", global_qps);
    // 读取 batch_size（仅用于 SEARCH 批消息封装）；默认 1
    size_t batch_size = dataset_config["batch_size"] ? dataset_config["batch_size"].as<size_t>() : 1;
    generator.set_batch_size(batch_size);
    RAFT_LOG_INFO("[create_pattern_from_yaml] Batch size set to: %zu", batch_size);
    
    // 遍历操作步骤 (skip first build operation since consumer auto-builds)
    bool skip_first = true;
    for (const auto& op_pair : dataset_config) {
        auto key = op_pair.first.as<std::string>();

        // 跳过非数字键（这些是数据集属性）
        if (!std::isdigit(key[0])) {
            continue;
        }

        auto op_config = op_pair.second;
        std::string op_type = op_config["operation"].as<std::string>();

        // Skip first operation (build) - handled by consumer
        if (skip_first) {
            RAFT_LOG_INFO("[create_pattern_from_yaml] Skipping step %s (handled by consumer build phase)", key.c_str());
            skip_first = false;
            continue;
        }
        
        if (op_type == "insert") {
            size_t start = op_config["start"].as<size_t>();
            size_t end = op_config["end"].as<size_t>();
            size_t count = end - start;

            generator.add_insert_step(start, end);
            RAFT_LOG_INFO("[create_pattern_from_yaml] Step %s: INSERT range=[%zu, %zu), count=%zu",
                           key.c_str(), start, end, count);

        } else if (op_type == "search") {
            int k = op_config["k"] ? op_config["k"].as<int>() : 10;
            size_t count = op_config["count"] ? op_config["count"].as<size_t>() : 5000;

            generator.add_search_step(count, k);
            RAFT_LOG_INFO("[create_pattern_from_yaml] Step %s: SEARCH count=%zu, k=%d",
                           key.c_str(), count, k);

        } else if (op_type == "delete") {
            size_t start = op_config["start"].as<size_t>();
            size_t end = op_config["end"].as<size_t>();
            size_t count = end - start;

            generator.add_delete_step(start, end);
            RAFT_LOG_INFO("[create_pattern_from_yaml] Step %s: DELETE range=[%zu, %zu), count=%zu",
                           key.c_str(), start, end, count);
        }
    }
    
    return generator;
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

int streaming_producer_main(int argc, char** argv) {
    // 设置信号处理
    std::signal(SIGINT, signal_handler);
    
    // 解析命令行参数
    std::string config_file = getArgValue(argc, argv, "--config", 
        "/data/workspace/svfusion/examples/runbooks/streaming_test.yaml");
    std::string data_type = getArgValue(argc, argv, "--data-type", "float");
    std::string brokers = getArgValue(argc, argv, "--brokers", "localhost:9092");
    std::string topic = getArgValue(argc, argv, "--topic", "vector-queries");
    std::string control_topic_name = getArgValue(argc, argv, "--control-topic", "svfusion-control");
    RAFT_LOG_INFO("[streaming_producer_main] Streaming Producer Configuration:");
    RAFT_LOG_INFO("[streaming_producer_main]   Config file: %s", config_file.c_str());
    RAFT_LOG_INFO("[streaming_producer_main]   Data type: %s", data_type.c_str());
    RAFT_LOG_INFO("[streaming_producer_main]   Kafka brokers: %s", brokers.c_str());
    RAFT_LOG_INFO("[streaming_producer_main]   Kafka topic: %s", topic.c_str());
    RAFT_LOG_INFO("[streaming_producer_main]   Control topic: %s", control_topic_name.c_str());
    RAFT_LOG_INFO("");
    
    try {
        if (data_type == "float") {
            // 解析配置，获取数据集名称
            YAML::Node config = YAML::LoadFile(config_file);
            auto dataset_node = config.begin();
            std::string dataset_name = dataset_node->first.as<std::string>();
            RAFT_LOG_INFO("[streaming_producer_main] Using dataset: %s", dataset_name.c_str());
            
            // 创建工作负载生成器
            auto generator = create_pattern_from_yaml<float>(config_file, dataset_name);
            
            // 设置running状态以支持信号中断
            generator.set_running_state(&g_running);
            
            // Phase 1: Wait for consumer BUILD_COMPLETE
            RAFT_LOG_INFO("[streaming_producer_main] Waiting for consumer BUILD_COMPLETE...");
            {
                char errstr[512];
                rd_kafka_conf_t* conf = rd_kafka_conf_new();
                rd_kafka_conf_set(conf, "bootstrap.servers", brokers.c_str(), errstr, sizeof(errstr));
                auto* consumer = rd_kafka_new(RD_KAFKA_CONSUMER, conf, errstr, sizeof(errstr));
                auto* control_topic = rd_kafka_topic_new(consumer, control_topic_name.c_str(), nullptr);
                // 可能在producer先启动时，控制topic尚不存在。循环重试直到topic可用或超时。
                bool consume_started = false;
                int timeout_sec = 60;
                auto start_wait = std::chrono::steady_clock::now();
                while (!consume_started) {
                    if (!g_running.load()) {
                        cleanup_kafka_control(control_topic, consumer, /*stop_consume=*/false);
                        RAFT_LOG_INFO("[streaming_producer_main] Interrupted while waiting control topic");
                        return 0;
                    }
                    // 从BEGINNING开始，确保在topic创建后不会错过首条BUILD_COMPLETE
                    if (rd_kafka_consume_start(control_topic, 0, RD_KAFKA_OFFSET_BEGINNING) == 0) {
                        consume_started = true;
                        break;
                    }
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now() - start_wait).count();
                    if (elapsed >= timeout_sec) {
                        RAFT_LOG_ERROR("[streaming_producer_main] Failed to start consuming control topic within %d seconds: %s",
                                       timeout_sec, rd_kafka_err2str(rd_kafka_last_error()));
                        cleanup_kafka_control(control_topic, consumer, /*stop_consume=*/false);
                        return 1;
                    }
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }

                bool ready = false;
                auto start = std::chrono::steady_clock::now();

                while (!ready) {
                    if (!g_running.load()) {
                        cleanup_kafka_control(control_topic, consumer, /*stop_consume=*/true);
                        RAFT_LOG_INFO("[streaming_producer_main] Interrupted while waiting BUILD_COMPLETE");
                        return 0;
                    }
                    auto* msg = rd_kafka_consume(control_topic, 0, 1000);
                    if (msg && msg->err == RD_KAFKA_RESP_ERR_NO_ERROR) {
                        std::string data(static_cast<char*>(msg->payload), msg->len);
                        if (data.find("BUILD_COMPLETE") != std::string::npos) {
                            ready = true;
                        }
                    }
                    if (msg) rd_kafka_message_destroy(msg);

                    if (std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now() - start).count() > timeout_sec) {
                        RAFT_LOG_ERROR("[streaming_producer_main] Timeout waiting for BUILD_COMPLETE");
                        cleanup_kafka_control(control_topic, consumer, /*stop_consume=*/true);
                        return 1;
                    }
                }

                cleanup_kafka_control(control_topic, consumer, /*stop_consume=*/true);
                RAFT_LOG_INFO("[streaming_producer_main] BUILD_COMPLETE received, starting search phase");
            }

            // Phase 2: Send search queries
            ffanns::test::ProducerConfig kafka_config;
            kafka_config.brokers = brokers;
            kafka_config.topic = topic;
            ffanns::test::KafkaProducer producer(kafka_config);

            // Use global QPS from generator for all operations
            double global_qps = generator.get_global_qps();
            ffanns::test::RateController rate_limiter(global_qps);
            RAFT_LOG_INFO("[streaming_producer_main] Using global QPS: %.1f for all operations", global_qps);
            
            uint64_t total_sent = 0;
            bool started = false;
            std::chrono::steady_clock::time_point start_time;
            
            while (g_running.load()) {
                try {
                    // 获取当前步骤
                    const auto* current_step = generator.get_current_step();
                    if (!current_step) {
                        RAFT_LOG_INFO("[streaming_producer_main] All workload steps completed!");
                        break;
                    }

                    // 等待速率控制（使用全局QPS）
                    rate_limiter.wait_for_next();
                    
                    // 生成消息并在“发送时刻”打入起始时间（避免过早打点）
                    auto query = generator.generate_next_message();
                    auto send_ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
                    query.set_timestamp_us(static_cast<uint64_t>(send_ts_us));
                    bool success = producer.send_query(query);
                    
                    if (success) {
                        auto now = std::chrono::steady_clock::now();
                        if (!started) {
                            start_time = now;
                            started = true;
                        }
                        total_sent++;
                        
                        // 每1000条消息打印一次统计
                        if (total_sent % 1000 == 0 && started) {
                            auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
                                                now - start_time);
                            double actual_qps = elapsed.count() > 0
                                                  ? static_cast<double>(total_sent) / elapsed.count()
                                                  : 0.0;
                            
                            auto progress = generator.get_step_progress();
                            // 在统计中按 batch_size 放大消息与QPS，便于论文表达（查询级统计）
                            const size_t bs = generator.get_batch_size();
                            const unsigned long long effective_total = (unsigned long long)(total_sent * bs);
                            const double effective_qps = actual_qps * static_cast<double>(bs);
                            RAFT_LOG_INFO("[streaming_producer_main] Progress: Messages: %zu/%zu, Total sent(msg): %llu, Eff sent(q): %llu, Actual QPS(msg): %.1f, Eff QPS(q): %.1f, Target QPS(msg): %.1f",
                                           progress.first, progress.second,
                                           (unsigned long long)total_sent, effective_total,
                                           actual_qps, effective_qps,
                                           global_qps);
                        }
                    } else {
                        RAFT_LOG_INFO("[streaming_producer_main] Failed to send message");
                    }
                    
                } catch (const std::exception& e) {
                    std::string error_msg = e.what();
                    if (error_msg.find("All workload steps completed") != std::string::npos) {
                        RAFT_LOG_INFO("[streaming_producer_main] All workload steps completed!");
                        break;
                    } else if (error_msg.find("Interrupted by signal") != std::string::npos) {
                        RAFT_LOG_INFO("[streaming_producer_main] Interrupted by signal");
                        break;
                    } else {
                        RAFT_LOG_INFO("[streaming_producer_main] Error generating message: %s", error_msg.c_str());
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                }
            }
            
            // 最终统计
            auto end_time = std::chrono::steady_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
                end_time - start_time).count();
            double final_qps = total_duration > 0 ? static_cast<double>(total_sent) / total_duration : 0.0;
            RAFT_LOG_INFO("[streaming_producer_main] Final Statistics:");
            RAFT_LOG_INFO("[streaming_producer_main]   Total messages sent: %llu", (unsigned long long)total_sent);
            RAFT_LOG_INFO("[streaming_producer_main]   Total duration: %ld seconds", total_duration);
            const size_t bs = generator.get_batch_size();
            const double eff_final_qps = final_qps * static_cast<double>(bs);
            RAFT_LOG_INFO("[streaming_producer_main]   Average QPS: %.2f (eff=%.2f with batch=%zu)",
                            final_qps, eff_final_qps, bs);
            RAFT_LOG_INFO("[streaming_producer_main]   Producer errors: %llu", (unsigned long long)producer.get_error_count());
            
        } else {
            RAFT_LOG_INFO("[streaming_producer_main] Unsupported data type: %s", data_type.c_str());
            return 1;
        }
        
    } catch (const std::exception& e) {
        RAFT_LOG_INFO("[streaming_producer_main] Error: %s", e.what());
        return 1;
    }
    
    RAFT_LOG_INFO("[streaming_producer_main] Streaming producer finished.");
    return 0;
}

} // namespace test
} // namespace ffanns

// main函数必须在全局命名空间
int main(int argc, char** argv) {
    return ffanns::test::streaming_producer_main(argc, argv);
}
