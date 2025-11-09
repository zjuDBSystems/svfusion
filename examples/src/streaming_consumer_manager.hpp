#pragma once

#include <string>
#include <atomic>
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <librdkafka/rdkafka.h>
#include <raft/core/device_resources.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include "datasets.hpp"
#include "streaming_messages.pb.h"
#include "streaming_utils.hpp"
#include <ffanns/neighbors/cagra.hpp>
#include <algorithm>
#include <iomanip>
#include <raft/core/logger.hpp>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>
#include "utils.hpp"  // For save_step_neighbors_to_binary / save_neighbors_to_binary
#include <filesystem>
#include <optional>
#include <rmm/device_uvector.hpp>

namespace ffanns {
namespace test {

template <typename DataT>
class StreamingConsumerManager {
public:
    StreamingConsumerManager(const std::string& config_file, 
                           const std::string& kafka_brokers = "localhost:9092",
                           const std::string& kafka_topic = "vector-queries",
                           const std::string& control_topic = "svfusion-control");
    
    ~StreamingConsumerManager();
    
    void initialize(raft::device_resources& dev_resources,
                   std::shared_ptr<rmm::cuda_stream_pool> stream_pool = nullptr);
    void run_streaming_workload(raft::device_resources& dev_resources);
    void stop();

    // Producer-consumer pattern methods
    void ingestion_thread_func();  // Kafka ingestion thread
    void search_thread_func(raft::device_resources& dev_resources);  // Single-stream search thread
    void search_stream_func(size_t worker_id);                        // Per-stream search thread

private:
    void setup_kafka_consumer();
    void cleanup_kafka_consumer();
    void initialize_index(raft::device_resources& dev_resources);
    void send_build_complete_signal();
    // Unified search core used by single-stream and multi-stream workers
    void run_search_core(raft::device_resources& res,
                         raft::device_matrix<DataT, int64_t>& query_device_buffer,
                         const DataT* host_query_ptr,
                         size_t batch_size,
                         size_t dim,
                         int k,
                         size_t qid,
                         ffanns::neighbors::cagra::search_context<DataT, uint32_t>* search_ctx);
    void parse_config();
    void print_stats();
    void accumulate_operation_results(raft::device_resources& dev_resources);  // Accumulate current operation results
    void save_all_results();  // Save all accumulated results to file
    
    // Configuration
    std::string config_file_;
    std::string dataset_name_;
    size_t max_pts_;
    std::string res_path_;
    size_t first_op_start_;   
    size_t first_op_end_;    
    static constexpr int SEARCH_K = 10;  // Maximum k value to support (can be overridden by yaml)
    size_t batch_size_{1};   
    size_t insert_batch_size_{1};
    double qps_{0.0};        
    // Track dataset/graph row counts
    size_t build_rows_{0};
    size_t offset_{0};
    size_t insert_staging_offset_{0};
    size_t d_dataset_offset_{0};
    size_t d_graph_offset_{0};
    // Insert operations from YAML (excluding the first build op)
    std::vector<std::pair<size_t, size_t>> insert_ops_;
    struct InsertBatch { size_t start; size_t end; size_t host_offset; };
    std::vector<InsertBatch> insert_batches_;
    size_t next_insert_batch_idx_{0};
    size_t next_insert_expected_id_{0};
    // legacy placeholders removed: using strict ordered id per batch
    
    // Kafka
    std::string kafka_brokers_;
    std::string kafka_topic_;
    std::string control_topic_;
    rd_kafka_t* kafka_consumer_;
    rd_kafka_topic_t* kafka_topic_handle_;
    rd_kafka_conf_t* kafka_conf_;
    
    // Index and dataset
    std::unique_ptr<ffanns::Dataset<DataT>> dataset_;
    std::unique_ptr<ffanns::neighbors::cagra::index<DataT, uint32_t>> index_;
    raft::host_matrix<DataT, int64_t> host_space_;
    
    // Critical data structures for index lifecycle management
    raft::host_matrix<uint32_t, int64_t> graph_space_;
    std::shared_ptr<ffanns::core::bitset<std::uint32_t, int64_t>> delete_bitset_ptr_;
    std::shared_ptr<ffanns::core::HostBitSet> host_delete_bitset_ptr_;
    std::shared_ptr<rmm::device_uvector<uint32_t>> tag_to_id_;
    // External owners for in_edges (host/device) used by build and lifetime of index
    raft::host_vector<int, int64_t> in_edges_host_;
    raft::device_vector<int, int64_t> in_edges_device_;
    // IMPORTANT: Even though these are size 0, they must be kept alive!
    raft::device_matrix<DataT, int64_t> device_data_owner_;
    raft::device_matrix<uint32_t, int64_t> device_graph_space_;
    
    std::atomic<bool> running_{false};
    std::chrono::steady_clock::time_point start_time_;
    std::atomic<bool> ingestion_ready_{false};
    std::atomic<bool> worker_ready_{false};
    
    struct MessageQueue {
        std::queue<ffanns::streaming::VectorQuery> messages;
        mutable std::mutex mtx;
        std::condition_variable cv;

        void push(ffanns::streaming::VectorQuery&& msg) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                messages.push(std::move(msg));
            }
            cv.notify_one();
        }

        bool pop(ffanns::streaming::VectorQuery& msg, std::chrono::milliseconds timeout) {
            std::unique_lock<std::mutex> lock(mtx);
            if (cv.wait_for(lock, timeout, [this] { return !messages.empty(); })) {
                msg = std::move(messages.front());
                messages.pop();
                return true;
            }
            return false;
        }

        size_t size() const {
            std::lock_guard<std::mutex> lock(mtx);
            return messages.size();
        }
    };

    // Queues: separate search/insert/delete paths to reduce contention
    MessageQueue search_queue_;   // SEARCH operations
    MessageQueue insert_queue_;   // INSERT operations
    MessageQueue delete_queue_;   // DELETE operations
    std::unique_ptr<std::thread> ingestion_thread_;
    std::unique_ptr<std::thread> search_thread_;
    std::vector<std::unique_ptr<std::thread>> search_threads_;
    std::unique_ptr<std::thread> insert_thread_;
    std::unique_ptr<std::thread> delete_thread_;

    // Stream pool & per-stream contexts
    std::shared_ptr<rmm::cuda_stream_pool> stream_pool_;
    size_t num_search_streams_{0};
    size_t num_insert_streams_{1};
    // Device resources pointer for background insert thread
    raft::device_resources* dev_res_ptr_{nullptr};
    struct WorkerCtx {
        // Configuration constants for buffer sizing (matching search operator patterns)
        static constexpr size_t MAX_QUERIES = 4;        // Queries per search call (1 for streaming)
        static constexpr size_t MAX_CTA_PER_QUERY = 8; // Default CTAs per query for multi-CTA mode
        static constexpr size_t MAX_TRANS_NUM = 4;      // Max vectors for miss handling (0 = no miss handling)

        raft::device_resources worker_res;
        raft::device_matrix<DataT, int64_t> query_device_buffer;
        DataT* pinned_host_staging{nullptr};
        size_t dim{0};
        size_t batch_size{1};
        // Pinned owners for host-side miss staging
        raft::pinned_mdarray<float,    raft::vector_extent<size_t>,  raft::layout_c_contiguous> cpu_distances_owner;
        raft::pinned_mdarray<uint32_t, raft::vector_extent<size_t>,  raft::layout_c_contiguous> host_indices_owner;
        raft::pinned_mdarray<unsigned int, raft::vector_extent<size_t>, raft::layout_c_contiguous> query_miss_counter_owner;
        raft::pinned_mdarray<uint32_t, raft::vector_extent<size_t>,  raft::layout_c_contiguous> host_graph_buffer_owner;
        raft::pinned_mdarray<DataT,   raft::matrix_extent<int64_t>,  raft::row_major>           host_vector_buffer_owner;
        raft::pinned_scalar<unsigned int> num_graph_miss_owner;
        raft::pinned_scalar<unsigned int> num_miss1_owner;
        raft::pinned_scalar<unsigned int> num_miss2_owner;
        ffanns::neighbors::cagra::search_context<DataT, uint32_t> search_ctx{};
        WorkerCtx(raft::device_resources const& main_res,
                  rmm::cuda_stream_view stream_view,
                  size_t n_cols,
                  size_t graph_degree,
                  size_t search_width = 1,
                  size_t batch_size = 1,
                  size_t max_queries = MAX_QUERIES,
                  size_t max_cta_per_query = MAX_CTA_PER_QUERY,
                  size_t max_trans_num = MAX_TRANS_NUM)
            : worker_res(main_res),
              query_device_buffer(raft::make_device_matrix<DataT, int64_t>(main_res, static_cast<int64_t>(batch_size), n_cols)),
              dim(n_cols),
              batch_size(batch_size),
              cpu_distances_owner(raft::make_pinned_vector<float>(main_res,
                  max_queries * max_cta_per_query * search_width * graph_degree)),
              host_indices_owner(raft::make_pinned_vector<uint32_t>(main_res,
                  max_queries * max_cta_per_query * search_width * graph_degree)),
              query_miss_counter_owner(raft::make_pinned_vector<unsigned int>(main_res,
                  max_queries * max_cta_per_query)),
              host_graph_buffer_owner(raft::make_pinned_vector<uint32_t>(main_res,
                  max_queries * max_cta_per_query * search_width * graph_degree)),
              host_vector_buffer_owner(raft::make_pinned_matrix<DataT, int64_t, raft::row_major>(main_res,
                  max_trans_num, int64_t(n_cols))),
              num_graph_miss_owner(raft::make_pinned_scalar<unsigned int>(main_res, 0u)),
              num_miss1_owner(raft::make_pinned_scalar<unsigned int>(main_res, 0u)),
              num_miss2_owner(raft::make_pinned_scalar<unsigned int>(main_res, 0u)) {
            raft::resource::set_cuda_stream(worker_res, stream_view);
            cudaMallocHost(reinterpret_cast<void**>(&pinned_host_staging), batch_size * dim * sizeof(DataT));

            // Only set pointers if buffers are actually allocated (non-zero size)
            search_ctx.cpu_distances      = (max_queries > 0) ? cpu_distances_owner.data_handle() : nullptr;
            search_ctx.host_indices       = (max_queries > 0) ? host_indices_owner.data_handle() : nullptr;
            search_ctx.query_miss_counter = (max_queries > 0) ? query_miss_counter_owner.data_handle() : nullptr;
            search_ctx.host_graph_buffer  = (max_queries > 0) ? host_graph_buffer_owner.data_handle() : nullptr;
            search_ctx.host_vector_buffer = (max_trans_num > 0) ? host_vector_buffer_owner.data_handle() : nullptr;
            search_ctx.num_graph_miss     = num_graph_miss_owner.data_handle();
            search_ctx.num_miss1          = num_miss1_owner.data_handle();
            search_ctx.num_miss2          = num_miss2_owner.data_handle();
        }
        ~WorkerCtx() {
            if (pinned_host_staging) cudaFreeHost(pinned_host_staging);
        }
    };
    std::vector<std::unique_ptr<WorkerCtx>> workers_;
    // Dedicated WorkerCtx for insert path (may share default stream if no pool)
    std::unique_ptr<WorkerCtx> insert_worker_;

    static constexpr size_t MAX_RESULT_QUERIES = 5000;  // Renamed to avoid confusion with WorkerCtx::MAX_QUERIES
    raft::device_matrix<uint32_t, int64_t> all_neighbors_;        // [MAX_RESULT_QUERIES, k]
    raft::device_matrix<float, int64_t> all_distances_;           // [MAX_RESULT_QUERIES, k]
    // Single query buffer for immediate copy (single stream mode)
    raft::device_matrix<DataT, int64_t> query_device_buffer_;

    // Multi-stream mode flag (thread-per-stream, not pool-based)
    bool use_multi_stream_{false};

    // Per-operation statistics (reset after each operation)
    std::atomic<uint64_t> total_searches_{0};        // Also serves as operation progress counter
    std::atomic<uint64_t> total_messages_{0};
    std::atomic<uint64_t> total_latency_us_{0};      // Sum of all search latencies in current operation
    // Insert latency statistics (lightweight, analogous to search)
    std::atomic<uint64_t> total_inserts_{0};
    std::atomic<uint64_t> total_insert_latency_us_{0};
    std::atomic<size_t> operation_number_{2};        // Current operation number (starts from 2 since op 1 is build)
    std::mutex save_mutex_;                          // Mutex for saving results and resetting stats
    std::vector<std::vector<uint32_t>> step_neighbors_;  // Accumulate all operation results

    // Insert path: single-record handler and thread loop (no barrier, no write lock)
    void insert_thread_func();
    // Delete path: single-record lazy_delete (no barrier)
    void delete_thread_func();
};

template <typename DataT>
StreamingConsumerManager<DataT>::StreamingConsumerManager(
    const std::string& config_file,
    const std::string& kafka_brokers,
    const std::string& kafka_topic,
    const std::string& control_topic)
    : config_file_(config_file), 
      kafka_brokers_(kafka_brokers),
      kafka_topic_(kafka_topic),
      control_topic_(control_topic),
      kafka_consumer_(nullptr),
      kafka_topic_handle_(nullptr),
      kafka_conf_(nullptr),
      first_op_start_(0),
      first_op_end_(0),
      host_space_(raft::make_host_matrix<DataT, int64_t>(0, 0)),
      graph_space_(raft::make_host_matrix<uint32_t, int64_t>(0, 0)),
      device_data_owner_(raft::make_device_matrix<DataT, int64_t>(raft::device_resources{}, 0, 0)),
      device_graph_space_(raft::make_device_matrix<uint32_t, int64_t>(raft::device_resources{}, 0, 0)),
      all_neighbors_(raft::make_device_matrix<uint32_t, int64_t>(raft::device_resources{}, 0, 0)),
      all_distances_(raft::make_device_matrix<float, int64_t>(raft::device_resources{}, 0, 0)),
      query_device_buffer_(raft::make_device_matrix<DataT, int64_t>(raft::device_resources{}, 0, 0)),
      in_edges_host_(raft::make_host_vector<int, int64_t>(0)),
      in_edges_device_(raft::make_device_vector<int, int64_t>(raft::device_resources{}, 0)){
    
    parse_config();
}

template <typename DataT>
StreamingConsumerManager<DataT>::~StreamingConsumerManager() {
    cleanup_kafka_consumer();
}

template <typename DataT>
void StreamingConsumerManager<DataT>::parse_config() {
    YAML::Node config = YAML::LoadFile(config_file_);
    auto dataset_node = config.begin();
    dataset_name_ = dataset_node->first.as<std::string>();
    auto dataset_config = dataset_node->second;
    
    max_pts_ = dataset_config["max_pts"].as<size_t>();
    res_path_ = dataset_config["res_path"].as<std::string>();
    if (dataset_config["batch_size"]) {
        batch_size_ = std::max<size_t>(1, dataset_config["batch_size"].as<size_t>());
    }
    if (dataset_config["insert_batch_size"]) {
        insert_batch_size_ = std::max<size_t>(1, dataset_config["insert_batch_size"].as<size_t>());
    }
    if (dataset_config["qps"]) {
        qps_ = dataset_config["qps"].as<double>();
    }

    // Initialize bench_config: align to workload_manager style output
    // dataset_name: top-level key; base_log_dir: parent of res_path (e.g., /results)
    auto& save_cfg = ffanns::neighbors::bench_config::instance();
    save_cfg.dataset_name = dataset_name_;
    {
        std::filesystem::path p(res_path_);
        // Make base_log_dir the grandparent of res_path (strip trailing experiment folder like streaming_test)
        auto base = p;
        if (base.has_parent_path()) base = base.parent_path();       // .../results/streaming_test
        if (base.has_parent_path()) base = base.parent_path();       // .../results
        save_cfg.base_log_dir = base.string();                       // e.g., /data/workspace/svfusion/results
    }
    save_cfg.chunk_size = 1024;                // match workload_manager default
    
    // 保存第一个 INSERT 操作的范围，并收集后续 INSERT 操作
    bool first_insert_found = false;
    for (const auto& op_pair : dataset_config) {
        auto key = op_pair.first.as<std::string>();
        if (!std::isdigit(key[0])) continue;
        auto op_config = op_pair.second;
        auto op_type = op_config["operation"].as<std::string>();
        if (op_type == "insert") {
            size_t s = op_config["start"].as<size_t>();
            size_t e = op_config["end"].as<size_t>();
            if (!first_insert_found) {
                first_op_start_ = s;
                first_op_end_ = e;
                first_insert_found = true;
                RAFT_LOG_INFO("[StreamingConsumerManager::parse_config] First operation for build: INSERT %zu-%zu", s, e);
            } else {
                insert_ops_.emplace_back(s, e);
            }
        }
    }
    RAFT_LOG_INFO("[StreamingConsumerManager::parse_config] Consumer Config - Dataset: %s, Kafka: %s/%s, batch_size=%zu, insert_batch_size=%zu, qps=%.3f",
                  dataset_name_.c_str(), kafka_brokers_.c_str(), kafka_topic_.c_str(), batch_size_, insert_batch_size_, qps_);
}

// moved to streaming_consumer_manager_kafka.inl

template <typename DataT>
void StreamingConsumerManager<DataT>::initialize_index(raft::device_resources& dev_resources) {
    // 基础设置（参考workload_manager.cu）
    ffanns::neighbors::cagra::index<DataT, uint32_t>::set_max_device_rows(2000000);
    ffanns::neighbors::cagra::index<DataT, uint32_t>::set_max_graph_device_rows(2000000);
    
    dataset_ = ffanns::create_dataset<DataT>(dataset_name_);
    dataset_->init_data_stream();
    
    size_t n_rows = first_op_end_ - first_op_start_;
    size_t n_cols = dataset_->num_dimensions();
    
    // host_space_必须使用max_pts_大小，与workload_manager保持一致
    host_space_ = raft::make_host_matrix<DataT, int64_t>(max_pts_, n_cols);
    auto host_space_view = host_space_.view();
    
    size_t n_samples = dataset_->read_batch_pos(host_space_view, 0, first_op_start_, n_rows);
    build_rows_ = n_samples;
    insert_staging_offset_ = build_rows_;
    offset_ = build_rows_;
    
    ffanns::neighbors::cagra::index_params build_params;
    if (dataset_->distance_type() == ffanns::DistanceType::EUCLIDEAN) {
        build_params.metric = ffanns::distance::DistanceType::L2Expanded;
    } else if (dataset_->distance_type() == ffanns::DistanceType::INNER_PRODUCT) {
        build_params.metric = ffanns::distance::DistanceType::InnerProduct;
    }
    build_params.graph_degree = 64;
    build_params.intermediate_graph_degree = 128;
    
    graph_space_ = raft::make_host_matrix<uint32_t, int64_t>(max_pts_, build_params.graph_degree);
    delete_bitset_ptr_ = std::make_shared<ffanns::core::bitset<std::uint32_t, int64_t>>(dev_resources, max_pts_);
    host_delete_bitset_ptr_ = std::make_shared<ffanns::core::HostBitSet>(max_pts_, true);
    tag_to_id_ = std::make_shared<rmm::device_uvector<uint32_t>>(max_pts_, raft::resource::get_cuda_stream(dev_resources));
    
    auto dataset_view = raft::make_host_matrix_view<DataT, int64_t>(host_space_view.data_handle(), n_samples, n_cols);
    auto graph_view = raft::make_host_matrix_view<uint32_t, int64_t>(graph_space_.data_handle(), n_samples, build_params.graph_degree);
    device_data_owner_ = raft::make_device_matrix<DataT, int64_t>(dev_resources, 0, n_cols);
    device_graph_space_ = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, 0, build_params.graph_degree);
    
    // Allocate external in_edges owners and pass views to build
    in_edges_host_ = raft::make_host_vector<int, int64_t>(max_pts_);
    in_edges_device_ = raft::make_device_vector<int, int64_t>(dev_resources, max_pts_);
    auto host_in_edges_view = raft::make_host_vector_view<int, int64_t>(in_edges_host_.data_handle(), n_samples);
    auto dev_in_edges_view = raft::make_device_vector_view<int, int64_t>(in_edges_device_.data_handle(), n_samples);
    auto index = ffanns::neighbors::cagra::build(
        dev_resources, build_params, dataset_view, graph_view, device_data_owner_, 
        device_graph_space_, delete_bitset_ptr_, tag_to_id_, host_in_edges_view, dev_in_edges_view, 0, n_samples);
    index.update_host_delete_bitset(host_delete_bitset_ptr_);
    index_ = std::make_unique<ffanns::neighbors::cagra::index<DataT, uint32_t>>(std::move(index));
    
    RAFT_LOG_INFO("[StreamingConsumerManager::initialize_index] Index built: %zu vectors x %zu dims", n_samples, n_cols);

    // Initialize device offsets from owners allocated by build
    size_t build_d_datasize = std::min(n_samples, 
        static_cast<size_t>(ffanns::neighbors::cagra::index<DataT, uint32_t>::get_max_device_rows()));
    size_t build_d_graphsize = std::min(n_samples, 
        static_cast<size_t>(ffanns::neighbors::cagra::index<DataT, uint32_t>::get_max_graph_device_rows()));
    d_dataset_offset_ = build_d_datasize;
    d_graph_offset_   = build_d_graphsize;
    RAFT_LOG_INFO("[StreamingConsumerManager::initialize_index] d_dataset_offset: %zu, d_graph_offset: %zu", d_dataset_offset_, d_graph_offset_);

    // Preload all subsequent insert batches into host_space_ contiguous region
    size_t offset = insert_staging_offset_;
    for (auto& p : insert_ops_) {
        size_t s = p.first, e = p.second;
        if (e <= s) continue;
        size_t n = e - s;
        size_t wrote = dataset_->read_batch_pos(host_space_.view(), offset, s, n);
        if (wrote != n) {
            RAFT_LOG_WARN("[initialize_index] Preload mismatch for batch [%zu,%zu): expect %zu, got %zu", s, e, n, wrote);
        }
        insert_batches_.push_back(InsertBatch{ s, e, offset });
        RAFT_LOG_INFO("[initialize_index] Preloaded batch [%zu,%zu) at host_offset=%zu", s, e, offset);
        offset += n;
    }
}

// moved to streaming_consumer_manager_kafka.inl

template <typename DataT>
void StreamingConsumerManager<DataT>::initialize(raft::device_resources& dev_resources,
                                                 std::shared_ptr<rmm::cuda_stream_pool> stream_pool) {
    // Save device resource pointer for background insert thread use
    dev_res_ptr_ = &dev_resources;
    setup_kafka_consumer();
    initialize_index(dev_resources);

    size_t n_cols = dataset_->num_dimensions();

    // Pre-allocate memory for all queries
    all_neighbors_ = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, MAX_RESULT_QUERIES, SEARCH_K);
    all_distances_ = raft::make_device_matrix<float, int64_t>(dev_resources, MAX_RESULT_QUERIES, SEARCH_K);
    RAFT_LOG_INFO("[initialize] Pre-allocated buffers for %zu queries (dim=%zu, k=%d)",
                  MAX_RESULT_QUERIES, n_cols, SEARCH_K);

    size_t stream_pool_size = stream_pool ? stream_pool->get_pool_size()
                                          : raft::resource::get_stream_pool_size(dev_resources);
    if (stream_pool_size >= 2) {
        if (!stream_pool) {
            throw std::runtime_error(
                "Stream pool shared_ptr must be provided when multi-stream mode is enabled");
        }
        use_multi_stream_ = true;
        stream_pool_ = stream_pool;
        // Reserve last stream for insert; others for search workers
        num_search_streams_ = stream_pool_size - 1;
        workers_.reserve(num_search_streams_);
        for (size_t i = 0; i < num_search_streams_; ++i) {
            auto stream_view = stream_pool_->get_stream(i);
            auto gd = static_cast<size_t>(index_->graph().extent(1));
            // Constructor signature: (res, stream, n_cols, graph_degree, search_width=1, batch_size=1, ...)
            workers_.push_back(std::make_unique<WorkerCtx>(
                dev_resources, stream_view, n_cols, gd, 1, batch_size_));
        }
        // Create dedicated insert worker on the last stream
        auto insert_stream_view = stream_pool_->get_stream(num_search_streams_);
        auto gd = static_cast<size_t>(index_->graph().extent(1));
        insert_worker_ = std::make_unique<WorkerCtx>(dev_resources, insert_stream_view, n_cols, gd, 1, insert_batch_size_);
        RAFT_LOG_INFO("[initialize] Multi-stream mode: %zu search streams + 1 insert stream (pool=%zu)",
                     num_search_streams_, stream_pool_size);
    } else {
        use_multi_stream_ = false;
        query_device_buffer_ = raft::make_device_matrix<DataT, int64_t>(dev_resources, batch_size_, n_cols);
        // Create insert worker on default stream to enable search_ctx reuse
        auto gd = static_cast<size_t>(index_->graph().extent(1));
        auto default_stream_view = raft::resource::get_cuda_stream(dev_resources);
        insert_worker_ = std::make_unique<WorkerCtx>(dev_resources, default_stream_view, n_cols, gd, 1, 1);
        RAFT_LOG_INFO("[initialize] Single-stream mode (no additional stream pool configured)");
    }

    start_time_ = std::chrono::steady_clock::now();
}

template <typename DataT>
void StreamingConsumerManager<DataT>::run_streaming_workload(raft::device_resources& dev_resources) {
    running_ = true;
    std::cout << "Starting streaming workload with producer-consumer architecture..." << std::endl;

    try {
        const size_t n_cols = dataset_->num_dimensions();
        if (use_multi_stream_) {
            for (size_t i = 0; i < workers_.size(); ++i) {
                auto& ctx = *workers_[i];
                ffanns::test::warmup_worker<DataT>(ctx.worker_res, *index_, n_cols, all_neighbors_, all_distances_, 4);
            }
        } else {
            ffanns::test::warmup_worker<DataT>(dev_resources, *index_, n_cols, all_neighbors_, all_distances_, 8);
        }
        worker_ready_.store(true, std::memory_order_release);
        RAFT_LOG_INFO("[run_streaming_workload] Central warm-up completed");
    } catch (...) {
        RAFT_LOG_INFO("[run_streaming_workload] Central warm-up skipped due to exception");
        worker_ready_.store(true, std::memory_order_release);
    }

    ingestion_thread_ = std::make_unique<std::thread>(
        &StreamingConsumerManager<DataT>::ingestion_thread_func, this);

    // Start insert/delete threads (single-writer each, no barrier)
    insert_thread_ = std::make_unique<std::thread>(&StreamingConsumerManager<DataT>::insert_thread_func, this);
    delete_thread_ = std::make_unique<std::thread>(&StreamingConsumerManager<DataT>::delete_thread_func, this);

    if (use_multi_stream_) {
        for (size_t i = 0; i < workers_.size(); ++i) {
            search_threads_.push_back(std::make_unique<std::thread>(
                &StreamingConsumerManager<DataT>::search_stream_func, this, i));
        }
    } else {
        search_thread_ = std::make_unique<std::thread>(
            &StreamingConsumerManager<DataT>::search_thread_func, this, std::ref(dev_resources));
    }

    // Wait until both threads are ready (bounded wait), then prime data topic and signal producer
    {
        auto t0 = std::chrono::steady_clock::now();
        while ((!ingestion_ready_.load(std::memory_order_acquire) ||
                !worker_ready_.load(std::memory_order_acquire))) {
            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0).count() > 5) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // App-level handshake: warm Kafka metadata/fetch path for data topic
        ffanns::test::prime_kafka_data_topic(kafka_brokers_, kafka_topic_, static_cast<uint32_t>(dataset_->num_dimensions()));
        send_build_complete_signal();
    }

    // Wait for threads to complete
    if (ingestion_thread_ && ingestion_thread_->joinable()) {
        ingestion_thread_->join();
    }

    // Signal worker to stop after ingestion completes
    running_ = false;
    search_queue_.cv.notify_all();
    insert_queue_.cv.notify_all();
    delete_queue_.cv.notify_all();
    if (use_multi_stream_) {
        for (auto& th : search_threads_) {
            if (th && th->joinable()) th->join();
        }
    } else {
        if (search_thread_ && search_thread_->joinable()) {
            search_thread_->join();
        }
    }

    if (insert_thread_ && insert_thread_->joinable()) insert_thread_->join();
    if (delete_thread_ && delete_thread_->joinable()) delete_thread_->join();

    print_stats();
    save_all_results();
}

// Ingestion thread: pulls messages from Kafka as fast as possible
template <typename DataT>
void StreamingConsumerManager<DataT>::ingestion_thread_func() {
    RAFT_LOG_INFO("[Ingestion] Thread started");
    ingestion_ready_.store(true, std::memory_order_release);

    int no_message_count = 0;
    const int MAX_NO_MESSAGE_COUNT = 10;

    while (running_.load()) {
        // Use short timeout for responsive consumption
        rd_kafka_message_t* msg = rd_kafka_consume(kafka_topic_handle_, 0, 100);

        if (msg && msg->err == RD_KAFKA_RESP_ERR_NO_ERROR) {
            ffanns::streaming::VectorQuery query;
            if (query.ParseFromArray(msg->payload, msg->len)) {
                // Skip priming message (query_id == 0) from stats and processing
                if (query.query_id() == 0) { rd_kafka_message_destroy(msg); continue; }

                // Route by operation type: SEARCH -> search queue; INSERT/DELETE -> write queue
                switch (query.operation()) {
                    case ffanns::streaming::OperationType::SEARCH:
                        search_queue_.push(std::move(query));
                        break;
                    case ffanns::streaming::OperationType::INSERT:
                        insert_queue_.push(std::move(query));
                        break;
                    case ffanns::streaming::OperationType::DELETE:
                        delete_queue_.push(std::move(query));
                        break;
                    default:
                        // Ignore unknown operations for now
                        break;
                }

                total_messages_++;
                no_message_count = 0;

                if (total_messages_ % 1000 == 0) {
                    RAFT_LOG_INFO("[Ingestion] Messages: %lu, SearchQ: %zu, InsertQ: %zu, DeleteQ: %zu",
                                  total_messages_.load(), search_queue_.size(), insert_queue_.size(), delete_queue_.size());
                }
            }
            rd_kafka_message_destroy(msg);
        } else {
            if (msg) rd_kafka_message_destroy(msg);
            no_message_count++;

            if (no_message_count >= MAX_NO_MESSAGE_COUNT && total_messages_ > 0) {
                RAFT_LOG_INFO("[Ingestion] No messages for %d seconds, stopping", MAX_NO_MESSAGE_COUNT);
                break;
            }
        }
    }

    running_ = false;
    search_queue_.cv.notify_all();
    insert_queue_.cv.notify_all();
    delete_queue_.cv.notify_all();
    RAFT_LOG_INFO("[Ingestion] Thread finished, total messages: %lu", total_messages_.load());
}

// Insert thread: processes INSERT messages one-by-one (no barrier, single-writer)
template <typename DataT>
void StreamingConsumerManager<DataT>::insert_thread_func() {
    RAFT_LOG_INFO("[Insert] Thread started");
    // All batches preloaded during initialize_index; now extend per incoming id (1xD view)
    if (next_insert_batch_idx_ < insert_batches_.size()) {
        next_insert_expected_id_ = insert_batches_[next_insert_batch_idx_].start;
    }
    while (running_.load() || insert_queue_.size() > 0) {
        ffanns::streaming::VectorQuery q;
        if (!insert_queue_.pop(q, std::chrono::milliseconds(50))) {
            if (!running_.load()) break;
            continue;
        }
        if (q.operation() != ffanns::streaming::OperationType::INSERT) continue;
        
        size_t insert_chunk_size = insert_worker_ ? insert_worker_->batch_size : 1;
        size_t message_chunk_size   = static_cast<size_t>(q.k());
        assert(insert_chunk_size == message_chunk_size);
        // RAFT_LOG_INFO("[Insert] Insert chunk size: %zu", insert_chunk_size);
      
        if (next_insert_batch_idx_ >= insert_batches_.size()) {
            RAFT_LOG_WARN("[Insert] No pending batches, ignoring vector_id=%lu", (unsigned long)q.vector_id());
            continue;
        }
        const auto& batch = insert_batches_[next_insert_batch_idx_];
        const size_t s = batch.start, e = batch.end;
        const size_t id = static_cast<size_t>(q.vector_id());
        if (id != next_insert_expected_id_) {
            // In paper code, we assume ordered inserts; log and ignore if out-of-order
            RAFT_LOG_WARN("[Insert] Out-of-order id=%zu, expect %zu (batch [%zu,%zu))", id, next_insert_expected_id_, s, e);
            continue;
        }
        // Build 1xD additional_dataset at host_space_[batch.host_offset + (id - s)]
        const int64_t dim = host_space_.extent(1);
        auto additional_dataset = raft::make_host_matrix_view<const DataT, int64_t>(
            host_space_.data_handle() +
                (static_cast<int64_t>(batch.host_offset) + static_cast<int64_t>(id - s)) * dim,
            insert_chunk_size, dim);

        // Compute new extents first (align to workload_manager: updated views reflect the new size)
        size_t add_dev_rows = 0;
        size_t add_graph_rows = 0;
        auto& mapper = index_->hd_mapper();
        auto& gmapper = index_->get_graph_hd_mapper();
        add_dev_rows   = (mapper.current_size   < mapper.device_capacity)   ? insert_chunk_size : 0;
        add_graph_rows = (gmapper.current_size < gmapper.device_capacity) ? insert_chunk_size : 0;
        // Advance counters to represent the post-insert sizes
        offset_ += insert_chunk_size;
        d_dataset_offset_ += add_dev_rows;
        d_graph_offset_ += add_graph_rows;

        // Build updated host dataset/graph views (0..new_host_rows]
        auto updated_dataset = raft::make_host_matrix_view<DataT, int64_t>(
            host_space_.data_handle(), offset_, dim);
        auto updated_graph = raft::make_host_matrix_view<uint32_t, int64_t>(
            graph_space_.data_handle(), offset_, static_cast<int64_t>(index_->graph().extent(1)));

        // Build updated device dataset/graph views with new row counts
        auto updated_device_dataset = raft::make_device_matrix_view<DataT, int64_t>(
            device_data_owner_.data_handle(), d_dataset_offset_, dim);
        auto updated_device_graph = raft::make_device_matrix_view<uint32_t, int64_t>(
            device_graph_space_.data_handle(), d_graph_offset_, static_cast<int64_t>(index_->graph().extent(1)));

        const auto t_start_us_now = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        // Perform extend with full updated views
        ffanns::neighbors::cagra::extend_params extend_params;
        extend_params.max_chunk_size = insert_chunk_size;
        // Use dedicated insert worker resources if available
        auto& insert_res = insert_worker_ ? insert_worker_->worker_res : *dev_res_ptr_;
        ffanns::neighbors::cagra::extend(insert_res, extend_params, additional_dataset, *index_,
                                         updated_dataset,
                                         updated_device_dataset,
                                         updated_graph,
                                         updated_device_graph,
                                         static_cast<uint32_t>(id), static_cast<uint32_t>(id + insert_chunk_size),
                                         insert_worker_ ? &insert_worker_->search_ctx : nullptr);

        // Record latency
        const auto t_now_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        // uint64_t e2e_us = t_now_us - static_cast<uint64_t>(t_start_us_now);
        uint64_t e2e_us = t_now_us - q.timestamp_us();
        total_inserts_.fetch_add(insert_chunk_size, std::memory_order_acq_rel);
        total_insert_latency_us_.fetch_add(e2e_us * insert_chunk_size, std::memory_order_acq_rel);
        if ((total_inserts_.load(std::memory_order_acquire) % 1000) == 0) {
            auto cnt = total_inserts_.load(std::memory_order_acquire);
            auto sum = total_insert_latency_us_.load(std::memory_order_acquire);
            double avg_ms = (cnt > 0) ? (static_cast<double>(sum) / 1000.0) / static_cast<double>(cnt) : 0.0;
            RAFT_LOG_INFO("[Operation %zu] Progress: %lu inserts, avg latency so far: %.2fms",
                          operation_number_.load(), cnt, avg_ms);
        }

        // Advance expected id; rotate to next batch if finished
        next_insert_expected_id_ += insert_chunk_size;
        if (next_insert_expected_id_ >= e) {
            next_insert_batch_idx_++;
            if (next_insert_batch_idx_ < insert_batches_.size()) {
                next_insert_expected_id_ = insert_batches_[next_insert_batch_idx_].start;
            }
        }
    }
    // Final insert stats
    {
        auto cnt = total_inserts_.load(std::memory_order_acquire);
        auto sum = total_insert_latency_us_.load(std::memory_order_acquire);
        double avg_ms = (cnt > 0) ? (static_cast<double>(sum) / 1000.0) / static_cast<double>(cnt) : 0.0;
        RAFT_LOG_INFO("[Operation %zu] Completed: %lu inserts, avg latency: %.2fms",
                      operation_number_.load(), cnt, avg_ms);
    }
}

// Delete thread: processes DELETE messages one-by-one (no barrier)
template <typename DataT>
void StreamingConsumerManager<DataT>::delete_thread_func() {
    RAFT_LOG_INFO("[Delete] Thread started");
    while (running_.load() || delete_queue_.size() > 0) {
        ffanns::streaming::VectorQuery q;
        if (!delete_queue_.pop(q, std::chrono::milliseconds(50))) {
            if (!running_.load()) break;
            continue;
        }
        if (q.operation() != ffanns::streaming::OperationType::DELETE) continue;
        const auto start = static_cast<int64_t>(q.vector_id());
        const auto len   = static_cast<int64_t>(q.k());
        if (len > 0) {
            const auto end = start + len;
            ffanns::neighbors::cagra::lazy_delete(*dev_res_ptr_, *index_, start, end);
            RAFT_LOG_INFO("[Delete] Lazy deleted batch [%ld, %ld)", (long)start, (long)end);
        } else {
            RAFT_LOG_WARN("[Delete] Invalid delete batch length: %ld (start=%ld)", (long)len, (long)start);
        }
    }
    RAFT_LOG_INFO("[Delete] Thread finished");
}


// Worker thread: processes searches from local queue
template <typename DataT>
void StreamingConsumerManager<DataT>::search_thread_func(raft::device_resources& dev_resources) {
    RAFT_LOG_INFO("[Search] Thread started");

    while (running_.load() || search_queue_.size() > 0) {
        ffanns::streaming::VectorQuery query;
        if (search_queue_.pop(query, std::chrono::milliseconds(100))) {
            // Only handle SEARCH for now (float32)
            if (query.operation() == ffanns::streaming::OperationType::SEARCH &&
                query.data_type() == ffanns::streaming::DataType::FLOAT32) {
                const size_t dim = static_cast<size_t>(query.dimension());
                if (dim == 0 || dim != dataset_->num_dimensions()) { continue; }
                const size_t qid = static_cast<size_t>(query.query_id());
                if (qid < 1 || qid > MAX_RESULT_QUERIES) { continue; }
                const int k = query.k() > 0 ? std::min<int>(int(query.k()), SEARCH_K) : 10;
                // DEBUG
                auto start_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();

                run_search_core(dev_resources,
                                query_device_buffer_, query.vector_float().data(),
                                1, dim, k, qid, nullptr);

                auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                uint64_t e2e = (query.timestamp_us() > 0) ? (now_us - query.timestamp_us()) : (now_us - start_us);
                auto searches = total_searches_.fetch_add(1, std::memory_order_acq_rel) + 1;
                total_latency_us_ += e2e;
                if (searches % 1000 == 0) { print_stats(); }
            }
        }
    }

    RAFT_LOG_INFO("[Search] Thread finished, total searches: %lu", total_searches_.load());
}

// Per-stream worker: each thread owns a CUDA stream + thread-local staging
template <typename DataT>
void StreamingConsumerManager<DataT>::search_stream_func(size_t worker_id) {
    RAFT_LOG_INFO("[Search-%zu] Thread started", worker_id);
    auto& ctx = *workers_[worker_id];

    while (running_.load() || search_queue_.size() > 0) {
        if (!running_.load() && search_queue_.size() == 0) break;
        ffanns::streaming::VectorQuery query;
        if (!search_queue_.pop(query, std::chrono::milliseconds(50))) {
            if (!running_.load()) break;
            continue;
        }
        assert(query.operation() == ffanns::streaming::OperationType::SEARCH);
        assert(query.data_type() == ffanns::streaming::DataType::FLOAT32);
        // Expect producer to send a full batch: vector size == batch_size * dim
        assert(static_cast<size_t>(query.vector_float_size()) == ctx.batch_size * ctx.dim);

        size_t qid = static_cast<size_t>(query.query_id());
        if (qid < 1 || qid > MAX_RESULT_QUERIES) continue;
        int k = query.k() > 0 ? std::min<int>(int(query.k()), SEARCH_K) : 10;
        // Copy protobuf data to pinned staging then run unified search core
        const DataT* src = query.vector_float().data();
        // Copy a full batch (ctx.batch_size rows)
        std::memcpy(ctx.pinned_host_staging, src, ctx.batch_size * ctx.dim * sizeof(DataT));

        auto search_start_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        run_search_core(ctx.worker_res,
                        ctx.query_device_buffer, ctx.pinned_host_staging,
                        ctx.batch_size, ctx.dim, k, qid, &ctx.search_ctx);
        
        // E2E latency
        auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        // uint64_t e2e = now_us - query.timestamp_us();
        // DEBUG to test pure search latency`
        uint64_t e2e = now_us - search_start_us;
        
        total_searches_.fetch_add(ctx.batch_size, std::memory_order_acq_rel);
        total_latency_us_ += e2e * ctx.batch_size;
        // Add theoretical intra-batch offsets based on configured qps
        // if (qps_ > 0.0 && ctx.batch_size > 1) {
        //     const double per_query_interval_us = 1e6 / (qps_ * static_cast<double>(ctx.batch_size));
        //     const double extra = per_query_interval_us * (static_cast<double>(ctx.batch_size) * (ctx.batch_size - 1) / 2.0);
        //     total_latency_us_.fetch_add(static_cast<uint64_t>(extra), std::memory_order_acq_rel);
        // }
        auto searches = total_searches_.load(std::memory_order_acquire);
        // Check if this operation is complete (reached MAX_RESULT_QUERIES)
        if (searches >= MAX_RESULT_QUERIES) {
            // Try to acquire save lock - only one thread should save and reset
            if (save_mutex_.try_lock()) {
                if (searches >= MAX_RESULT_QUERIES) {
                    // Print final stats for this operation
                    RAFT_LOG_INFO("[Operation %zu] Completed: %zu searches, avg latency: %.2fms",
                                  operation_number_.load(),
                                  total_searches_.load(),
                                  total_latency_us_.load() / 1000.0 / total_searches_.load());

                    // Accumulate results for this operation
                    accumulate_operation_results(ctx.worker_res);
                    total_searches_.store(0, std::memory_order_release);
                    total_latency_us_.store(0, std::memory_order_release);
                    operation_number_.fetch_add(1, std::memory_order_acq_rel);
                }
                save_mutex_.unlock();
            }
        } else if (searches % 1000 == 0) {
            // Print intermediate progress
            RAFT_LOG_INFO("[Operation %zu] Progress: %zu/%zu searches, avg latency so far: %.2fms",
                          operation_number_.load(),
                          searches,
                          MAX_RESULT_QUERIES,
                          total_latency_us_.load() / 1000.0 / searches);
        }
    }

    RAFT_LOG_INFO("[Search-%zu] Thread finished", worker_id);
}

// Single-stream message processing (ONLY called by search_thread_func in single-stream mode)
// unified search core implementation
template <typename DataT>
void StreamingConsumerManager<DataT>::run_search_core(
    raft::device_resources& res,
    raft::device_matrix<DataT, int64_t>& query_device_buffer,
    const DataT* host_query_ptr,
    size_t batch_size,
    size_t dim,
    int k,
    size_t qid,
    ffanns::neighbors::cagra::search_context<DataT, uint32_t>* search_ctx) {

    auto host_view = raft::make_host_matrix_view<const DataT, int64_t>(host_query_ptr, static_cast<int64_t>(batch_size), static_cast<int64_t>(dim));
    auto dev_view = query_device_buffer.view();
    raft::copy(dev_view.data_handle(), host_view.data_handle(), host_view.size(),
               raft::resource::get_cuda_stream(res));

    auto row_offset = static_cast<int64_t>((qid - 1) * SEARCH_K);
    auto neighbor_indices_view = raft::make_device_matrix_view<uint32_t, int64_t>(
        all_neighbors_.data_handle() + row_offset, static_cast<int64_t>(batch_size), k);
    auto neighbor_distances_view = raft::make_device_matrix_view<float, int64_t>(
        all_distances_.data_handle() + row_offset, static_cast<int64_t>(batch_size), k);

    ffanns::neighbors::cagra::search_params params;
    params.itopk_size = 256;
    params.max_iterations = 50;
    params.metric = index_->metric();
    auto no_filter = ffanns::neighbors::filtering::none_sample_filter();

    ffanns::neighbors::cagra::search(res, params, *index_,
                                     dev_view, host_view,
                                     neighbor_indices_view, neighbor_distances_view,
                                     no_filter, true,
                                     search_ctx);

    raft::resource::sync_stream(res);
}

// moved to streaming_consumer_manager_stats.inl

template <typename DataT>
void StreamingConsumerManager<DataT>::stop() {
    RAFT_LOG_INFO("[StreamingConsumerManager::stop] Shutting down...");
    running_ = false;
    search_queue_.cv.notify_all();
    // Give threads a moment to exit cleanly
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Save any accumulated results before shutting down
    // save_all_results();
}

template <typename DataT>
void StreamingConsumerManager<DataT>::accumulate_operation_results(raft::device_resources& dev_resources) {
    try {
        // Copy neighbors from device to host
        std::vector<uint32_t> host_neighbors(MAX_RESULT_QUERIES * SEARCH_K);
        raft::copy(host_neighbors.data(),
                   all_neighbors_.data_handle(),
                   MAX_RESULT_QUERIES * SEARCH_K,
                   raft::resource::get_cuda_stream(dev_resources));
        raft::resource::sync_stream(dev_resources);

        // Accumulate this operation's results
        step_neighbors_.push_back(std::move(host_neighbors));

        RAFT_LOG_INFO("[accumulate_operation_results] Accumulated operation %zu results (%zu queries x %d neighbors, total ops: %zu)",
                      operation_number_.load(), MAX_RESULT_QUERIES, SEARCH_K, step_neighbors_.size());
    } catch (const std::exception& e) {
        RAFT_LOG_ERROR("[accumulate_operation_results] Failed to accumulate operation %zu: %s",
                       operation_number_.load(), e.what());
    }
}

template <typename DataT>
void StreamingConsumerManager<DataT>::save_all_results() {
    if (step_neighbors_.empty()) {
        RAFT_LOG_INFO("[save_all_results] No results to save");
        return;
    }

    try {
        // Align with workload_manager: save a combined binary without step suffix
        save_neighbors_to_binary(step_neighbors_, ffanns::neighbors::bench_config::instance());

        RAFT_LOG_INFO("[save_all_results] Saved results to %s",
                      ffanns::neighbors::bench_config::instance().get_search_path().c_str());
    } catch (const std::exception& e) {
        RAFT_LOG_ERROR("[save_all_results] Failed to save results: %s", e.what());
    }
}

} // namespace test
} // namespace ffanns

// Include out-of-line template member implementations to keep this header focused
#include "streaming_consumer_manager_kafka.inl"
#include "streaming_consumer_manager_stats.inl"
