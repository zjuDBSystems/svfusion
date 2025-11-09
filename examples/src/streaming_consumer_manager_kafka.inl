// Implementation of Kafka-related member functions for StreamingConsumerManager<DataT>
namespace ffanns { namespace test {

template <typename DataT>
void StreamingConsumerManager<DataT>::setup_kafka_consumer() {
    char errstr[512];
    
    kafka_conf_ = rd_kafka_conf_new();
    rd_kafka_conf_set(kafka_conf_, "bootstrap.servers", kafka_brokers_.c_str(), errstr, sizeof(errstr));
    
    kafka_consumer_ = rd_kafka_new(RD_KAFKA_CONSUMER, kafka_conf_, errstr, sizeof(errstr));
    if (!kafka_consumer_) {
        throw std::runtime_error("Failed to create Kafka consumer");
    }
    
    kafka_topic_handle_ = rd_kafka_topic_new(kafka_consumer_, kafka_topic_.c_str(), nullptr);
    if (!kafka_topic_handle_) {
        cleanup_kafka_consumer();
        throw std::runtime_error("Failed to create topic handle");
    }
    
    if (rd_kafka_consume_start(kafka_topic_handle_, 0, RD_KAFKA_OFFSET_BEGINNING) == -1) {
        if (rd_kafka_consume_start(kafka_topic_handle_, 0, RD_KAFKA_OFFSET_END) == -1) {
            throw std::runtime_error("Failed to start consuming: " + std::string(rd_kafka_err2str(rd_kafka_last_error())));
        }
        RAFT_LOG_INFO("[setup_kafka_consumer] Starting from END (partition might be empty)");
    } else {
        RAFT_LOG_INFO("[setup_kafka_consumer] Starting from BEGINNING to receive all messages");
    }
    
    RAFT_LOG_INFO("[StreamingConsumerManager::setup_kafka_consumer] Simple Kafka consumer ready (single-machine testing)");
}

template <typename DataT>
void StreamingConsumerManager<DataT>::cleanup_kafka_consumer() {
    if (kafka_topic_handle_) {
        rd_kafka_consume_stop(kafka_topic_handle_, 0);
        rd_kafka_topic_destroy(kafka_topic_handle_);
    }
    if (kafka_consumer_) {
        rd_kafka_destroy(kafka_consumer_);
    }
}

template <typename DataT>
void StreamingConsumerManager<DataT>::send_build_complete_signal() {
    char errstr[512];
    rd_kafka_conf_t* conf = rd_kafka_conf_new();
    rd_kafka_conf_set(conf, "bootstrap.servers", kafka_brokers_.c_str(), errstr, sizeof(errstr));
    
    rd_kafka_t* producer = rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));
    if (!producer) {
        RAFT_LOG_INFO("[StreamingConsumerManager::send_build_complete_signal] Failed to create producer for control signal: %s", errstr);
        return;
    }
    rd_kafka_topic_t* topic = rd_kafka_topic_new(producer, control_topic_.c_str(), nullptr);
    if (!topic) {
        rd_kafka_destroy(producer);
        RAFT_LOG_INFO("[StreamingConsumerManager::send_build_complete_signal] Failed to create control topic");
        return;
    }
    
    auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    std::string message = "BUILD_COMPLETE_" + std::to_string(timestamp);
    if (rd_kafka_produce(topic, /*partition*/ 0, RD_KAFKA_MSG_F_COPY,
                         (void*)message.c_str(), message.length(), nullptr, 0, nullptr) == -1) {
        RAFT_LOG_INFO("[send_build_complete_signal] Failed to send BUILD_COMPLETE signal");
    } else {
        RAFT_LOG_INFO("[send_build_complete_signal] ✓ Sent BUILD_COMPLETE signal: %s", message.c_str());
    }
    
    rd_kafka_flush(producer, 5000);
    rd_kafka_topic_destroy(topic);
    rd_kafka_destroy(producer);
}

}} // namespace ffanns::test
