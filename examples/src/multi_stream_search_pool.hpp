#pragma once

#include <raft/core/device_resources.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <ffanns/neighbors/cagra.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

namespace ffanns {
namespace test {

/**
 * @brief Multi-stream search pool for concurrent CAGRA searches
 *
 * This class manages a pool of search instances, each with its own stream
 * and resources. It provides a simple interface to get an idle instance,
 * use it for search, and return it to the pool.
 */
template <typename DataT, typename IndexT = uint32_t>
class MultiStreamSearchPool {
public:
    // 单个搜索实例（使用RAFT stream pool中的stream）
    struct SearchInstance {
        size_t id;
        size_t stream_idx;  // stream pool中的索引
        raft::device_resources worker_res;  // 轻量级resources拷贝，绑定到特定stream

        // 仅保留临时query buffer用于H2D拷贝
        raft::device_matrix<DataT, int64_t> query_staging_buffer;

        // 状态
        std::atomic<bool> is_busy{false};

        SearchInstance(size_t id_, size_t stream_idx_,
                      raft::device_resources const& main_res,
                      rmm::cuda_stream_view stream_view,
                      size_t dim)
            : id(id_),
              stream_idx(stream_idx_),
              worker_res(main_res),
              query_staging_buffer(raft::make_device_matrix<DataT, int64_t>(main_res, 1, dim)) {
            // 重定向worker_res到指定stream（共享stream pool已由拷贝继承）
            raft::resource::set_cuda_stream(worker_res, stream_view);
        }
    };

    /**
     * @brief Constructor
     * @param main_res 主RAFT resources（必须已配置stream pool和memory pool）
     * @param dim Dimension of vectors
     */
    MultiStreamSearchPool(raft::device_resources& main_res,
                          std::shared_ptr<rmm::cuda_stream_pool> stream_pool,
                          size_t dim)
        : main_resources_(main_res), stream_pool_(std::move(stream_pool)), dim_(dim) {
        if (stream_pool_ == nullptr || stream_pool_->get_pool_size() == 0) {
            throw std::runtime_error("Stream pool not configured in main resources");
        }
        num_streams_ = stream_pool_->get_pool_size();
        initialize();
    }

    ~MultiStreamSearchPool() {
        // 等待所有streams完成
        for (size_t i = 0; i < num_streams_; i++) {
            cudaStreamSynchronize(stream_pool_->get_stream(i).value());
        }
        // 不需要destroy streams，由stream pool管理
    }

    /**
     * @brief 获取一个空闲的搜索实例（阻塞直到有可用）
     */
    SearchInstance* acquire() {
        std::unique_lock<std::mutex> lock(pool_mutex_);

        // 等待直到有空闲实例
        pool_cv_.wait(lock, [this] { return !idle_instances_.empty(); });

        auto* instance = idle_instances_.front();
        idle_instances_.pop();
        instance->is_busy = true;

        return instance;
    }

    /**
     * @brief 尝试获取一个空闲的搜索实例（非阻塞）
     * @return nullptr if no idle instance available
     */
    SearchInstance* try_acquire() {
        std::lock_guard<std::mutex> lock(pool_mutex_);

        if (idle_instances_.empty()) {
            return nullptr;
        }

        auto* instance = idle_instances_.front();
        idle_instances_.pop();
        instance->is_busy = true;

        return instance;
    }

    /**
     * @brief 归还搜索实例到池中（内部使用，不同步）
     */
    void release_instance(SearchInstance* instance) {
        if (!instance) return;

        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            instance->is_busy = false;
            idle_instances_.push(instance);
        }

        pool_cv_.notify_one();
    }

    /**
     * @brief 异步搜索句柄（带端到端时间戳）
     */
    struct AsyncSearchHandle {
        SearchInstance* instance;
        cudaEvent_t event;
        int k;
        uint64_t producer_timestamp_us;  // Producer端的原始时间戳（微秒）
        uint32_t query_id;  // 查询ID用于追踪
        std::vector<DataT> host_query_storage;  // 保存原始查询，确保异步拷贝期间内存有效

        // 保存结果views用于后续读取
        raft::device_matrix_view<IndexT, int64_t> neighbors_view;
        raft::device_matrix_view<float, int64_t> distances_view;

        AsyncSearchHandle(SearchInstance* inst,
                          int k_val,
                          uint64_t timestamp_us,
                          uint32_t id,
                          std::vector<DataT>&& host_query,
                          raft::device_matrix_view<IndexT, int64_t> neighbors,
                          raft::device_matrix_view<float, int64_t> distances)
            : instance(inst),
              k(k_val),
              producer_timestamp_us(timestamp_us),
              query_id(id),
              host_query_storage(std::move(host_query)),
              neighbors_view(neighbors),
              distances_view(distances),
              event(nullptr) {
            cudaEventCreate(&event);
        }

        ~AsyncSearchHandle() {
            if (event) cudaEventDestroy(event);
        }
    };

    /**
     * @brief 提交异步搜索请求（使用预分配的结果views）
     */
    std::unique_ptr<AsyncSearchHandle> submit_search(
                                ffanns::neighbors::cagra::index<DataT, IndexT>& index,
                                const ffanns::neighbors::cagra::search_params& params,
                                std::vector<DataT> query_host,
                                int k,
                                uint64_t producer_timestamp_us,
                                uint32_t query_id,
                                raft::device_matrix_view<IndexT, int64_t> neighbors_view,
                                raft::device_matrix_view<float, int64_t> distances_view) {
        auto* instance = acquire();
        if (!instance) return nullptr;

        auto handle = std::make_unique<AsyncSearchHandle>(
            instance, k, producer_timestamp_us, query_id, std::move(query_host),
            neighbors_view, distances_view);

        // 创建views
        auto query_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
            handle->host_query_storage.data(), 1, dim_);

        auto query_device_view = raft::make_device_matrix_view<DataT, int64_t>(
            instance->query_staging_buffer.data_handle(), 1, dim_);

        // 使用worker_res的stream进行异步拷贝
        auto stream = raft::resource::get_cuda_stream(instance->worker_res);
        raft::copy(query_device_view.data_handle(),
                  query_host_view.data_handle(),
                  query_host_view.size(),
                  stream);

        auto filter = ffanns::neighbors::filtering::none_sample_filter();
        // 传入worker_res，search会自动使用其绑定的stream
        // 结果直接写入传入的views（最终位置）
        ffanns::neighbors::cagra::search(instance->worker_res, params, index,
                                        query_device_view, query_host_view,
                                        neighbors_view, distances_view,
                                        filter, true);

        // 记录event用于后续检查
        cudaEventRecord(handle->event, stream.value());

        return handle;
    }

    /**
     * @brief 检查搜索是否完成（非阻塞）
     */
    bool is_complete(const AsyncSearchHandle* handle) {
        if (!handle) return true;
        return cudaEventQuery(handle->event) == cudaSuccess;
    }

    uint64_t finalize_search(AsyncSearchHandle* handle) {
        if (!handle || !handle->instance) return 0;
        // Wait for completion
        cudaEventSynchronize(handle->event);
        auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        uint64_t e2e_latency_us = now_us - handle->producer_timestamp_us;
        // Release instance back to pool
        release_instance(handle->instance);
        handle->instance = nullptr;
        return e2e_latency_us;
    }

    size_t get_num_streams() const { return num_streams_; }
    size_t get_num_idle() const {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        return idle_instances_.size();
    }
    size_t get_num_busy() const {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        return num_streams_ - idle_instances_.size();
    }

private:
    void initialize() {
        for (size_t i = 0; i < num_streams_; i++) {
            // 从stream pool获取第i个stream
            auto stream_view = stream_pool_->get_stream(i);

            // 创建搜索实例（传入main resources、共享stream pool和stream view）
            auto instance = std::make_unique<SearchInstance>(
                i, i, main_resources_, stream_view, dim_);

            // 加入空闲队列
            idle_instances_.push(instance.get());
            instances_.push_back(std::move(instance));
        }
    }

private:
    raft::device_resources& main_resources_;  // 主resources引用（接受任何raft::resources类型）
    std::shared_ptr<rmm::cuda_stream_pool> stream_pool_;
    size_t num_streams_;
    size_t dim_;

    std::vector<std::unique_ptr<SearchInstance>> instances_;
    mutable std::mutex pool_mutex_;
    std::condition_variable pool_cv_;  // 用于通知有穿闲实例可用
    std::queue<SearchInstance*> idle_instances_;
};

} // namespace test
} // namespace ffanns
