#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <random>
#include <cmath>
#include <cassert>

#include <librdkafka/rdkafka.h>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/logger.hpp>

#include "datasets.hpp"
#include "utils.hpp"  // for read_fbin_file
#include "streaming_messages.pb.h"  // 生成的protobuf头文件

namespace ffanns {
namespace test {

/**
 * 操作类型枚举（映射到protobuf）
 */
enum class StreamingOperation {
    SEARCH = 1,
    INSERT = 2,
    DELETE = 3
};

/**
 * 模板化的查询消息构建器
 */
template<typename DataT>
class QueryMessageBuilder {
private:
    ffanns::streaming::VectorQuery pb_message_;
    
public:
    QueryMessageBuilder() {
        pb_message_.set_query_id(generate_next_id());
        // 延迟到发送时刻再设置真实时间戳，这里用0作为未设置的标记
        pb_message_.set_timestamp_us(0);
        pb_message_.set_k(10);  // 默认值
    }
    
    // 也可以手动设置query_id
    QueryMessageBuilder& set_query_id(uint32_t id) {
        pb_message_.set_query_id(id);
        return *this;
    }
    
    // 设置操作类型
    QueryMessageBuilder& set_operation(StreamingOperation op) {
        pb_message_.set_operation(static_cast<ffanns::streaming::OperationType>(op));
        return *this;
    }
    
    // 设置向量数据
    QueryMessageBuilder& set_vector(const std::vector<DataT>& vector) {
        pb_message_.set_dimension(vector.size());
        
        if constexpr (std::is_same_v<DataT, float>) {
            pb_message_.set_data_type(ffanns::streaming::DataType::FLOAT32);
            pb_message_.clear_vector_float();
            for (float val : vector) {
                pb_message_.add_vector_float(val);
            }
        } else if constexpr (std::is_same_v<DataT, uint8_t>) {
            pb_message_.set_data_type(ffanns::streaming::DataType::UINT8);
            pb_message_.set_vector_uint8(
                std::string(reinterpret_cast<const char*>(vector.data()), 
                           vector.size()));
        } else {
            throw std::runtime_error("Unsupported data type for vector");
        }
        return *this;
    }
    
    // 设置top-k
    QueryMessageBuilder& set_k(int k) {
        pb_message_.set_k(k);
        return *this;
    }
    
    // 设置向量ID（用于INSERT/DELETE操作）
    QueryMessageBuilder& set_vector_id(uint64_t id) {
        pb_message_.set_vector_id(id);
        return *this;
    }
    
    // 设置距离类型
    QueryMessageBuilder& set_distance_type(ffanns::DistanceType dist_type) {
        switch(dist_type) {
            case ffanns::DistanceType::EUCLIDEAN:
                pb_message_.set_distance_type(ffanns::streaming::DistanceType::L2_EXPANDED);
                break;
            case ffanns::DistanceType::INNER_PRODUCT:
                pb_message_.set_distance_type(ffanns::streaming::DistanceType::INNER_PRODUCT);
                break;
            default:
                pb_message_.set_distance_type(ffanns::streaming::DistanceType::L2_EXPANDED);
                break;
        }
        return *this;
    }
    
    // 构建消息
    [[nodiscard]] ffanns::streaming::VectorQuery build() const {
        return pb_message_;
    }
    
    // 序列化为字节数组
    [[nodiscard]] std::string serialize() const {
        std::string serialized;
        if (!pb_message_.SerializeToString(&serialized)) {
            throw std::runtime_error("Failed to serialize protobuf message");
        }
        return serialized;
    }
    
    // 从字节数组反序列化
    static ffanns::streaming::VectorQuery deserialize(const std::string& data) {
        ffanns::streaming::VectorQuery message;
        if (!message.ParseFromString(data)) {
            throw std::runtime_error("Failed to parse protobuf message");
        }
        return message;
    }
    
    // 生成递增的query_id（线程安全）
    static uint32_t generate_next_id() {
        static std::atomic<uint32_t> counter{1};
        return counter.fetch_add(1);
    }
    
    // 重置ID计数器（测试时使用）
    static void reset_id_counter() {
        static std::atomic<uint32_t> counter{1};
        counter.store(1);
    }
};

/**
 * 速率控制器 - 精确控制QPS（研究论文版本）
 * 使用固定间隔发送，确保精确的QPS
 */
class RateController {
private:
    double target_qps_;
    std::chrono::nanoseconds interval_{};
    std::chrono::steady_clock::time_point next_{};
    bool started_{false};
    mutable std::mutex mutex_;

public:
    explicit RateController(double qps) { set_qps(qps, /*rephase=*/true); }

    // 设置目标QPS；当 rephase=true 时，下一次 wait_for_next 将立即返回（首条立即发送）
    void set_qps(double qps, bool rephase = true) {
        std::lock_guard<std::mutex> lock(mutex_);
        target_qps_ = qps;
        interval_ = std::chrono::nanoseconds(
            static_cast<int64_t>(std::llround(1e9 / target_qps_)));
        if (rephase) {
            started_ = false;
        }
    }

    double get_qps() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return target_qps_;
    }

    // 等待直到下一个发送时间（固定节拍推进）
    void wait_for_next() {
        auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!started_) {
                // 首条消息：立即发送，并对齐下一个节拍
                started_ = true;
                next_ = now + interval_;
                return;
            }
            // 如果处理耗时导致落后节拍，则跳过过期节拍，直接对齐下一将来节拍
            while (next_ <= now) next_ += interval_;
        }
        std::this_thread::sleep_until(next_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            next_ += interval_;
        }
    }
};

/**
 * 查询向量加载器 - 用于SEARCH操作
 * 从dataset的query文件一次性加载所有查询向量到内存并持久化
 * 每个search operation从头按顺序发送
 */
template<typename DataT>
class QueryVectorLoader {
private:
    raft::host_matrix<DataT, int64_t> query_buffer_;  // 持久化的查询向量缓冲区
    size_t current_index_;
    size_t query_count_;
    size_t query_dim_;
    ffanns::DistanceType distance_type_;
    
public:
    QueryVectorLoader(const std::string& dataset_name) 
        : current_index_(0), query_count_(0), query_dim_(0),
          query_buffer_(raft::make_host_matrix<DataT, int64_t>(0, 0)) {
        
        // 创建数据集以获取query文件路径和distance type
        auto dataset = ffanns::create_dataset<DataT>(dataset_name);
        distance_type_ = dataset->distance_type();
        std::string query_file = dataset->query_filename();
        
        // 使用read_fbin_file一次性加载所有查询向量（参考workload_manager）
        DataT* query_vectors_raw;
        auto [count, dim] = read_fbin_file(query_file, query_vectors_raw);
        query_count_ = count;
        query_dim_ = dim;
        
        // 创建host矩阵并复制数据
        query_buffer_ = raft::make_host_matrix<DataT, int64_t>(query_count_, query_dim_);
        std::memcpy(query_buffer_.data_handle(), query_vectors_raw, 
                    query_count_ * query_dim_ * sizeof(DataT));
        
        // 释放原始数据
        delete[] query_vectors_raw;
        
        current_index_ = 0;
        RAFT_LOG_INFO("[QueryVectorLoader] Loaded %zu query vectors (dim=%zu) from %s - persisted in memory", 
                      query_count_, query_dim_, query_file.c_str());
    }
    
    // 重置索引，用于新的search operation
    void reset() {
        current_index_ = 0;
        RAFT_LOG_DEBUG("[QueryVectorLoader] Reset query index for new search operation");
    }
    
    // 获取下一个查询向量（严格顺序访问）；当 batch_size>1 时返回扁平化的 batch（支持环回）
    std::vector<DataT> get_next_query(size_t batch_size = 1) {
        if (query_count_ == 0) {
            throw std::runtime_error("No query vectors loaded");
        }
        if (batch_size == 0) batch_size = 1;

        const size_t dim = query_dim_;
        std::vector<DataT> out(batch_size * dim);

        assert(batch_size <= query_count_ - current_index_);
        const DataT* src = query_buffer_.data_handle() + current_index_ * dim;
        std::memcpy(out.data(), src, out.size() * sizeof(DataT));
        current_index_ += batch_size;
        if (current_index_ >= query_count_) current_index_ = 0;
        return out;
    }
    
    // 获取当前查询索引（用于groundtruth评估）
    size_t get_current_query_index() const {
        return current_index_ > 0 ? current_index_ - 1 : 0;
    }
    
    size_t get_dimension() const {
        return query_dim_;
    }
    
    size_t get_query_count() const {
        return query_count_;
    }
    
    ffanns::DistanceType get_distance_type() const {
        return distance_type_;
    }
};

/**
 * 插入向量加载器 - 用于INSERT操作
 * 从数据集的指定范围加载向量
 */
template<typename DataT>
class InsertVectorLoader {
private:
    std::unique_ptr<ffanns::Dataset<DataT>> dataset_;
    raft::host_matrix<DataT, int64_t> insert_buffer_;
    size_t current_index_;
    size_t loaded_count_;
    size_t start_idx_;  // 记录起始索引
    
public:
    InsertVectorLoader(const std::string& dataset_name) 
        : current_index_(0), loaded_count_(0), start_idx_(0),
          insert_buffer_(raft::make_host_matrix<DataT, int64_t>(0, 0)) {
        
        dataset_ = ffanns::create_dataset<DataT>(dataset_name);
        dataset_->init_data_stream();
    }
    
    // 加载指定范围的插入向量
    void load_range(size_t start, size_t end) {
        size_t dim = dataset_->num_dimensions();
        size_t count = end - start;
        
        insert_buffer_ = raft::make_host_matrix<DataT, int64_t>(count, dim);
        auto buffer_view = insert_buffer_.view();
        
        loaded_count_ = dataset_->read_batch_pos(buffer_view, 0, start, count);
        current_index_ = 0;
        start_idx_ = start;
        
        RAFT_LOG_INFO("[InsertVectorLoader] Loaded %zu vectors from [%zu, %zu) for INSERT",
                       loaded_count_, start, end);
    }
    
    // 获取下一个插入向量（严格顺序）
    std::vector<DataT> get_next_vector(size_t batch_size = 1) {
        if (current_index_ >= loaded_count_) {
            throw std::runtime_error("All insert vectors consumed");
        }

        if (batch_size == 0) batch_size = 1;
        size_t dim = dataset_->num_dimensions();
        std::vector<DataT> vector(batch_size * dim);
        
        // const DataT* row_ptr = insert_buffer_.data_handle() + current_index_ * dim;
        // std::memcpy(vector.data(), row_ptr, dim * sizeof(DataT));
        current_index_ += batch_size;
        return vector;
    }
    
    // 获取当前向量的实际ID
    size_t get_current_vector_id() const {
        return start_idx_ + current_index_ - 1;
    }
    
    bool has_more() const {
        return current_index_ < loaded_count_;
    }
    
    size_t get_dimension() const {
        return dataset_ ? dataset_->num_dimensions() : 0;
    }

    size_t get_loaded_count() const {
        return loaded_count_;
    }
    
    ffanns::DistanceType get_distance_type() const {
        return dataset_ ? dataset_->distance_type() : ffanns::DistanceType::EUCLIDEAN;
    }
};

/**
 * 工作负载模式生成器 - 支持不同操作类型的模式
 */
template<typename DataT>
class WorkloadPatternGenerator {
public:
    struct WorkloadStep {
        StreamingOperation operation;
        size_t count;              // 操作数量
        int k = 10;                // top-k（仅搜索操作使用）
        size_t start = 0;          // INSERT/DELETE操作的起始索引
        size_t end = 0;            // INSERT/DELETE操作的结束索引
    };
    
private:
    std::vector<WorkloadStep> pattern_;
    size_t current_step_;
    std::unique_ptr<QueryVectorLoader<DataT>> query_loader_;  // 持久化的查询向量加载器
    std::unique_ptr<InsertVectorLoader<DataT>> insert_loader_;  // 插入向量加载器
    size_t current_step_message_count_;  // 当前步骤已发送的消息数
    const std::atomic<bool>* running_ptr_ = nullptr;  // 用于检查是否应该继续运行
    bool insert_loaded_ = false;  // 标记INSERT数据是否已加载
    double global_qps_ = 100.0;  // 全局QPS设置（所有操作共享）
    size_t batch_size_ = 1;      // SEARCH 批大小（message-level 批发送，每条消息携带 batch_size 个向量）
    size_t insert_batch_size_ = 1;  // INSERT 批大小（message-level 批发送，每条消息携带 insert_batch_size 个向量）
    
public:
    WorkloadPatternGenerator(const std::string& dataset_name) 
        : current_step_(0), current_step_message_count_(0), insert_loaded_(false) {
        query_loader_ = std::make_unique<QueryVectorLoader<DataT>>(dataset_name);
        insert_loader_ = std::make_unique<InsertVectorLoader<DataT>>(dataset_name);
    }
    
    // 设置运行状态指针（用于信号处理）
    void set_running_state(const std::atomic<bool>* running) {
        running_ptr_ = running;
    }

    WorkloadPatternGenerator& set_global_qps(double qps) {
        global_qps_ = qps;
        return *this;
    }

    WorkloadPatternGenerator& set_batch_size(size_t b) {
        batch_size_ = b > 0 ? b : 1;
        return *this;
    }

    size_t get_batch_size() const { return batch_size_; }

    WorkloadPatternGenerator& set_insert_batch_size(size_t b) {
        insert_batch_size_ = b > 0 ? b : 1;
        return *this;
    }

    size_t get_insert_batch_size() const { return insert_batch_size_; }

    double get_global_qps() const {
        return global_qps_;
    }
        
    // 添加INSERT步骤（指定范围）
    WorkloadPatternGenerator& add_insert_step(size_t start, size_t end) {
        pattern_.push_back({StreamingOperation::INSERT, end - start, 10, start, end});
        return *this;
    }
    
    // 添加SEARCH步骤
    WorkloadPatternGenerator& add_search_step(size_t count = 5000, int k = 10) {
        pattern_.push_back({StreamingOperation::SEARCH, count, k, 0, 0});
        return *this;
    }

    // 添加DELETE步骤（指定范围）
    WorkloadPatternGenerator& add_delete_step(size_t start, size_t end) {
        pattern_.push_back({StreamingOperation::DELETE, end - start, 10, start, end});
        return *this;
    }
    
    // 生成下一个查询消息
    ffanns::streaming::VectorQuery generate_next_message() {
        if (current_step_ >= pattern_.size()) {
            throw std::runtime_error("No more workload steps");
        }
        
        auto& step = pattern_[current_step_];
        
        // 如果是BUILD占位步骤（count=0），直接跳到下一步
        if (step.count == 0) {
            if (!next_step()) {
                throw std::runtime_error("All workload steps completed");
            }
            // 检查是否被中断
            if (running_ptr_ && !running_ptr_->load()) {
                throw std::runtime_error("Interrupted by signal");
            }
            // 递归调用以处理真正需要执行的步骤
            return generate_next_message();
        }
        
        // 检查当前步骤是否已完成
        if (current_step_message_count_ >= step.count) {
            if (!next_step()) {
                throw std::runtime_error("All workload steps completed");
            }
            // 递归调用处理下一步（可能又是BUILD占位或新的步骤）
            return generate_next_message();
        }
        
        QueryMessageBuilder<DataT> builder;
        
        if (step.operation == StreamingOperation::INSERT) {
            // 加载INSERT数据（如果需要）
            if (!insert_loaded_ && step.start < step.end) {
                insert_loader_->load_range(step.start, step.end);
                assert (insert_loader_->get_loaded_count() % insert_batch_size_ == 0);
                insert_loaded_ = true;
            }
            auto vector = insert_loader_->get_next_vector(insert_batch_size_);
            size_t last_vector_id = insert_loader_->get_current_vector_id();
            
            builder.set_operation(StreamingOperation::INSERT)
                 // .set_vector(vector)
                   .set_k(insert_batch_size_)
                   .set_vector_id(last_vector_id - insert_batch_size_ + 1)
                   .set_distance_type(insert_loader_->get_distance_type());
            current_step_message_count_ += insert_batch_size_;
            return builder.build();

        } else if (step.operation == StreamingOperation::SEARCH) {
            if (current_step_message_count_ == 0) {
                query_loader_->reset();
            }
            if (batch_size_ > 1) {
                auto flat_batch = query_loader_->get_next_query(batch_size_);
                builder.set_operation(StreamingOperation::SEARCH)
                       .set_vector(flat_batch)
                       .set_k(step.k)
                       .set_distance_type(query_loader_->get_distance_type())
                       .set_query_id(static_cast<uint32_t>(current_step_message_count_ + 1));
                current_step_message_count_ += batch_size_;
                return builder.build();
            } else {
                auto vector = query_loader_->get_next_query(1);
                builder.set_operation(StreamingOperation::SEARCH)
                       .set_vector(vector)
                       .set_k(step.k)
                       .set_distance_type(query_loader_->get_distance_type())
                       .set_query_id(static_cast<uint32_t>(current_step_message_count_ + 1));
            }

        } else if (step.operation == StreamingOperation::DELETE) {
            // DELETE: send single batch message (vector_id=start, k=len), then finish this step
            if (current_step_message_count_ == 0) {
                size_t start = step.start;
                size_t len   = step.end - step.start;
                builder.set_operation(StreamingOperation::DELETE)
                       .set_vector_id(static_cast<uint32_t>(start))
                       .set_k(static_cast<int>(len));
                // Mark this step as completed after this message
                if (step.count > 0) {
                    current_step_message_count_ = step.count - 1;
                }
            } else {
                // Already sent the batch message; advance to next step
                if (!next_step()) {
                    throw std::runtime_error("All workload steps completed");
                }
                return generate_next_message();
            }
        }
        
        // 增加当前步骤的消息计数（SEARCH 在 batch_size_==1 时递增 1；INSERT/DELETE 为 1）
        current_step_message_count_++;
        
        return builder.build();
    }
    
    // 获取当前步骤
    const WorkloadStep* get_current_step() const {
        if (current_step_ >= pattern_.size()) return nullptr;
        return &pattern_[current_step_];
    }
    
    // 进入下一步骤
    bool next_step() {
        if (current_step_ < pattern_.size() - 1) {
            current_step_++;
            current_step_message_count_ = 0;  // 重置消息计数
            insert_loaded_ = false;  // 重置INSERT加载标记
            return true;
        }
        return false;
    }
    // 重置到第一步
    void reset() {
        current_step_ = 0;
        current_step_message_count_ = 0;
        insert_loaded_ = false;
        query_loader_->reset();  // 重置查询索引
    }
    
    // 获取当前步骤的进度信息
    std::pair<size_t, size_t> get_step_progress() const {
        if (current_step_ >= pattern_.size()) return {0, 0};
        return {current_step_message_count_, pattern_[current_step_].count};
    }
    
    size_t total_steps() const {
        return pattern_.size();
    }
};

/**
 * Kafka生产者配置
 */
struct ProducerConfig {
    std::string brokers = "localhost:9092";
    std::string topic = "vector-queries";
    int batch_size = 1;            // 立即发送每条消息（研究论文版本）
    int linger_ms = 0;             // 无延迟（研究论文版本）
    std::string compression = "lz4"; // 压缩算法
    
    static ProducerConfig from_args(int argc, char** argv) {
        ProducerConfig config;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg.find("--brokers=") == 0) {
                config.brokers = arg.substr(10);
            } else if (arg.find("--topic=") == 0) {
                config.topic = arg.substr(8);
            } else if (arg.find("--batch-size=") == 0) {
                config.batch_size = std::stoi(arg.substr(13));
            }
        }
        return config;
    }
};

/**
 * 简单的Kafka生产者包装器
 */
class KafkaProducer {
private:
    rd_kafka_t* producer_;
    rd_kafka_topic_t* topic_;
    std::string topic_name_;
    std::atomic<uint64_t> sent_count_{0};
    std::atomic<uint64_t> error_count_{0};
    
public:
    KafkaProducer(const ProducerConfig& config) : topic_name_(config.topic) {
        char errstr[512];
        
        // 创建Kafka配置
        rd_kafka_conf_t* conf = rd_kafka_conf_new();
        rd_kafka_conf_set(conf, "bootstrap.servers", config.brokers.c_str(), errstr, sizeof(errstr));
        rd_kafka_conf_set(conf, "batch.size", std::to_string(config.batch_size).c_str(), errstr, sizeof(errstr));
        rd_kafka_conf_set(conf, "linger.ms", std::to_string(config.linger_ms).c_str(), errstr, sizeof(errstr));
        rd_kafka_conf_set(conf, "compression.type", config.compression.c_str(), errstr, sizeof(errstr));
        
        // 创建生产者
        producer_ = rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));
        if (!producer_) {
            throw std::runtime_error("Failed to create producer: " + std::string(errstr));
        }
        
        // 创建主题
        topic_ = rd_kafka_topic_new(producer_, topic_name_.c_str(), nullptr);
        if (!topic_) {
            rd_kafka_destroy(producer_);
            throw std::runtime_error("Failed to create topic: " + topic_name_);
        }
        
        RAFT_LOG_INFO("[KafkaProducer::KafkaProducer] Kafka producer initialized for topic: %s, brokers: %s",
                       topic_name_.c_str(), config.brokers.c_str());
    }
    
    ~KafkaProducer() {
        if (topic_) rd_kafka_topic_destroy(topic_);
        if (producer_) {
            RAFT_LOG_INFO("[KafkaProducer::~KafkaProducer] Flushing producer... sent=%lu, errors=%lu",
                           sent_count_.load(), error_count_.load());
            rd_kafka_flush(producer_, 10000); // 等待10秒刷新
            rd_kafka_destroy(producer_);
        }
    }
    
    // 发送protobuf消息
    bool send_query(const ffanns::streaming::VectorQuery& query) {
        std::string serialized;
        if (!query.SerializeToString(&serialized)) {
            error_count_++;
            RAFT_LOG_INFO("[KafkaProducer::send_query] Failed to serialize protobuf message");
            return false;
        }
        return send_raw(serialized, std::to_string(query.query_id()));
    }
    
    // 发送原始字符串消息
    bool send_raw(const std::string& message, const std::string& key = "") {
        int result = rd_kafka_produce(
            topic_,
            0,  // 固定到分区0，匹配consumer只消费分区0的实现
            RD_KAFKA_MSG_F_COPY,    // 拷贝消息
            const_cast<char*>(message.c_str()),
            message.length(),
            key.empty() ? nullptr : key.c_str(),
            key.length(),
            nullptr  // 回调函数
        );
        
        if (result == -1) {
            error_count_++;
            RAFT_LOG_INFO("[KafkaProducer::send_raw] Failed to produce message: %s",
                           rd_kafka_err2str(rd_kafka_last_error()));
            return false;
        }
        
        sent_count_++;
        // 处理事件（非阻塞）
        rd_kafka_poll(producer_, 0);
        return true;
    }
    
    // 强制刷新缓冲区
    void flush(int timeout_ms = 1000) {
        rd_kafka_flush(producer_, timeout_ms);
    }
    
    uint64_t get_sent_count() const { return sent_count_.load(); }
    uint64_t get_error_count() const { return error_count_.load(); }
};

} // namespace test
} // namespace ffanns
