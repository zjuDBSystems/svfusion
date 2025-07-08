#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <memory>

namespace ffanns::neighbors::cagra::detail {

template <typename T, typename IdxT>
struct AsyncConsolidateState {
    std::thread worker_thread;
    std::atomic<bool> is_running{false};
    std::atomic<bool> has_result{false};
    std::atomic<bool> shutdown{false};
    
    // 互斥锁保护共享数据
    std::mutex mutex;

    bool sync_result(raft::resources const& res, ffanns::neighbors::cagra::index<T, IdxT>& index);
    
    // 这里只声明基本接口，不包含CUDA特定代码
    void submit_task(
        raft::resources const& res,
        ffanns::neighbors::cagra::index<T, IdxT>& index,
        raft::host_matrix_view<T, IdxT> consolidate_dataset);
    
    bool is_task_running() const {
        return is_running.load();
    }
    
    bool has_task_result() const {
        return has_result.load();
    }
    
    // 构造函数和析构函数
    AsyncConsolidateState() = default;
    ~AsyncConsolidateState();
    
    // 禁用拷贝构造和赋值
    AsyncConsolidateState(const AsyncConsolidateState&) = delete;
    AsyncConsolidateState& operator=(const AsyncConsolidateState&) = delete;
};

// 全局单例函数
template <typename T, typename IdxT>
AsyncConsolidateState<T, IdxT>& get_async_consolidate_state();

}  // namespace ffanns::neighbors::cagra::detail

// // 公共API函数，在cagra.hpp中只需声明此函数
// namespace ffanns::neighbors::cagra {
//     template <typename T, typename IdxT>
//     detail::AsyncConsolidateState<T, IdxT>& get_async_consolidate_state() {
//         return detail::get_async_consolidate_state<T, IdxT>();
//     }
// }