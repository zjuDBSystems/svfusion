#pragma once

#include "ffanns/neighbors/cagra.hpp"
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <bitset>
#include <queue>
#include <omp.h>
#include "ffanns/tsl/robin_set.h"
#include "ffanns/core/host_distance.hpp"
// #include <cuvs/distance/distance.hpp>

#include <cuda_runtime.h>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp> 
#include <rmm/device_uvector.hpp>
#include <vector>
#include <cmath>
#include <algorithm> // for std::max
#include <numeric> 

namespace ffanns::neighbors::cagra::detail {

static const std::string RAFT_NAME = "raft";

template <typename T, typename IdxT>
void lazy_delete(raft::resources const& res,
                index<T, IdxT>& index_,
                int64_t start_id,
                int64_t end_id)
{
    int64_t len = end_id - start_id;
    if (len <= 0) {
        RAFT_LOG_INFO("[lazy_delete] invalid range: start_id (%d) >= end_id (%d)", start_id, end_id);
        return;
    }
    // auto start_time = std::chrono::steady_clock::now();
    auto removed_indices = raft::make_device_vector<int64_t, int64_t>(res, len);

    // 利用 thrust::sequence 填充 [start_id, end_id) 的索引
    thrust::sequence(
        raft::resource::get_thrust_policy(res),
        thrust::device_pointer_cast(removed_indices.data_handle()),
        thrust::device_pointer_cast(removed_indices.data_handle() + len),
        start_id,   
        static_cast<int64_t>(1)           
    );
    raft::resource::sync_stream(res);

    auto delete_bitset_ptr = index_.get_delete_bitset_ptr();
    delete_bitset_ptr->set(res, removed_indices.view(), false);
    auto host_delete_bitset_ptr = index_.get_host_delete_bitset_ptr();
    RAFT_LOG_INFO("[lazy_delete] start_id: %d, end_id: %d", start_id, end_id);
    host_delete_bitset_ptr->mark_range_deleted(start_id, end_id);
    RAFT_LOG_INFO("[lazy_delete] host_delete_bitset_ptr->count_deleted(): %d", host_delete_bitset_ptr->count_deleted());

//     const IdxT* graph_data = index_.graph().data_handle();
//     int* in_edges = index_.host_in_edges().data_handle();
//     auto degree = index_.graph_degree();
// #pragma omp parallel for schedule(dynamic)
//     for (size_t d = start_id; d < end_id; d++) {
//         const IdxT* graph_data_ptr = graph_data + d * degree;
//         for (size_t j = 0; j < degree; j++) {
//             IdxT neighbor = graph_data_ptr[j];   
//             if (host_delete_bitset_ptr->test(neighbor)) {
// #pragma omp atomic
//                 in_edges[neighbor]--;
//             }         
//         }
//     }

//     auto end_time = std::chrono::steady_clock::now();
//     std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
//     RAFT_LOG_INFO("[lazy_delete] elapsed time: %f ms", elapsed.count());
}

template <typename T, typename IdxT, typename SetT>
bool robust_prune(cudaStream_t stream,
                                   uint32_t p,
                                   const SetT& candidates,
                                   const size_t degree,
                                   const size_t dim,
                                   float alpha,
                                //    T* d_data,
                                   const T* h_data,
                                   IdxT* subgraph_row
                                    ) 
{
    // 候选集合，初始包括原邻居和新候选
    bool test_flag = false;
    tsl::robin_set<uint32_t> V = candidates;  // 假设 candidates 已经包括 N_out(p) 并排除了 p 本身

    std::vector<uint32_t> new_neighbors;
    std::priority_queue<std::pair<T, uint32_t>,
                        std::vector<std::pair<T, uint32_t>>,
                        std::greater<>> min_heap;
    std::priority_queue<std::pair<T, uint32_t>,
                        std::vector<std::pair<T, uint32_t>>,
                        std::greater<>> backup_heap;

    // RAFT_LOG_INFO("[robust_prune] p: %d, V.size(): %d, dim: %ld", p, V.size(), dim);
    const T* p_data = h_data + p * dim;
    std::unordered_map<uint32_t, float> dist_cache;
    dist_cache.reserve(V.size());
    // 初次将候选节点加入堆（仅初始时计算距离）
    for (auto v : V) {
        const T* v_data = h_data + v * dim;
        // float dist = l2_distance_two_vectors(stream, p_data, v_data, dim);
        // float dist = cpu_l2_distance(p_data, v_data, dim);
        float dist = ffanns::core::l2_distance_avx2(p_data, v_data, dim);
        // RAFT_LOG_INFO("[robust_prune] p: %d, v: %d, dist: %f", p, v, dist);
        min_heap.emplace(dist, v);
        dist_cache[v] = dist;
    }

    while (!min_heap.empty() && new_neighbors.size() < degree) {
        auto [dist_p_pstar, p_star] = min_heap.top();
        min_heap.pop();

        // 如果p_star已经被删除，则跳过（因为可能前面删除操作里已被去除）
        if (V.find(p_star) == V.end()) continue;

        // 添加p_star到邻居表
        new_neighbors.push_back(p_star);
        V.erase(p_star);

        // 对剩余候选节点进行筛选，不符合alpha阈值的节点移除
        std::vector<uint32_t> to_remove;
        for (auto candidate : V) {
            const T* pstar_data = h_data + p_star * dim;
            float dist_p_candidate = dist_cache[candidate];
            const T* candidate_data = h_data + candidate * dim;
            // float dist_p_candidate = cpu_l2_distance(p_data, candidate_data, dim);
            float dist_pstar_candidate = ffanns::core::l2_distance_avx2(pstar_data, candidate_data, dim);
            // float dist_p_candidate = ffanns::core::l2_distance_avx2(p_data, candidate_data, dim);
            if (alpha * dist_pstar_candidate <= dist_p_candidate) {
                to_remove.push_back(candidate);
                backup_heap.emplace(dist_p_candidate, candidate);
            }
        }
        // 移除不符合的候选
        for (auto r : to_remove) {
            V.erase(r);
        }
    }

    if (new_neighbors.size() < degree) {
        // RAFT_LOG_INFO("[robust_prune] p: %d, new_neighbors.size(): %d, degree: %d", p, new_neighbors.size(), degree);
        test_flag = true;
    }

    while (new_neighbors.size() < degree && !backup_heap.empty()) {
        auto [dist, candidate] = backup_heap.top();
        backup_heap.pop();
        new_neighbors.push_back(candidate);
    }
    
    if (new_neighbors.size() < degree) {
        RAFT_LOG_INFO("[robust_prune] p: %d, new_neighbors.size(): %d, degree: %d", p, new_neighbors.size(), degree);
    }
    for (size_t i = 0; i < degree; i++) {
        subgraph_row[i] = new_neighbors[i];
    }
    // return new_neighbors;
    return test_flag;
}

template <typename T, typename IdxT>
void consolidate_delete(raft::resources const& res,
                index<T, IdxT>& index_,
                raft::host_matrix_view<T, IdxT> consolidate_dataset)
                // raft::device_matrix_view<T, IdxT, raft::layout_stride> consolidate_dataset)
{
    using bitset_t = uint32_t;
    // auto delete_bitset_ptr = index_.get_delete_bitset_ptr();
    auto host_bitset = index_.get_host_delete_bitset_ptr();
     

    const std::size_t graph_size      = index_.dataset().extent(0);
    const std::size_t degree        = index_.graph_degree();
    const std::size_t dim        = index_.dim();
    // RAFT_LOG_INFO("[consolidate_delete] graph_size: %ld %ld", graph_size, index_.graph().size());
    auto updated_graph = raft::make_host_matrix<IdxT, std::int64_t>(graph_size, degree);
    memcpy(updated_graph.data_handle(),
       index_.graph().data_handle(),
       index_.graph().size() * sizeof(IdxT));

    int influenced_nodes = 0;
    // int deleted_nodes = 0;
    int num_over_pruned_nodes = 0;
    RAFT_LOG_INFO("[consolidate_delete] graph_size: %d", graph_size);

    // const int num_streams = 2;
    // std::vector<cudaStream_t> stream_pool(num_streams);
    // for (int i = 0; i < num_streams; i++) {
    //     RAFT_CUDA_TRY(cudaStreamCreate(&stream_pool[i]));
    // }
    #pragma omp parallel
    {   
        const std::size_t num_threads = omp_get_num_threads();
        const std::size_t thread_id   = omp_get_thread_num();
        // cudaStream_t local_stream = stream_pool[thread_id % num_streams];
        cudaStream_t local_stream = raft::resource::get_cuda_stream(res);

        for (std::size_t loc = thread_id; loc < graph_size; loc += num_threads) {
        // for (std::size_t loc = 0; loc < graph_size; loc += 1) {
            if (!host_bitset->test(loc)) { //loc is marked as deleted
                // #pragma omp atomic
                // deleted_nodes += 1;
                continue;
            }
            tsl::robin_set<uint32_t> expanded_nodes_set;
            std::vector<uint32_t> expanded_nodes_lists;
            expanded_nodes_lists.reserve(degree);
            bool change_tag = false;
            for  (std::size_t i = 0; i < degree; i++) {
                const auto neighbor = updated_graph(loc, i);
                if (host_bitset->test(neighbor)) {
                    expanded_nodes_set.insert(neighbor);
                    expanded_nodes_lists.push_back(neighbor);
                } 
                else //neighbor is marked as deleted
                {
                    change_tag = true;
                    for (std::size_t j = 0; j < degree; j++) {
                        const auto d_neighbor = updated_graph(neighbor, j);
                        if (d_neighbor != loc && host_bitset->test(d_neighbor)) {
                            expanded_nodes_set.insert(d_neighbor);
                        }
                    }
                }
            }

            if (expanded_nodes_set.size() <= degree) {
                if (expanded_nodes_set.size() < degree) {
                    RAFT_LOG_INFO("[consolidate_delete!!!!!] loc: %d, expanded_nodes_set.size(): %d", loc, expanded_nodes_set.size());
                    RAFT_LOG_INFO("[consolidate_delete!!!!!] expanded_nodes_lists.size(): %d", expanded_nodes_lists.size());
                    continue;
                }
                if (change_tag) {
                    RAFT_LOG_INFO("[consolidate_delete!!!!!!!!!] just the same size but deleted: loc: %d, expanded_nodes_set.size(): %d", loc, expanded_nodes_set.size());
                }
                continue;
            } else {
                // RAFT_LOG_INFO("[consolidate_delete] loc: %d", loc);
                IdxT* subgraph_row = updated_graph.data_handle() + loc * degree;
                bool test_flag = robust_prune(local_stream, static_cast<uint32_t>(loc), expanded_nodes_set, degree, dim, 1.1f, consolidate_dataset.data_handle(), subgraph_row);
                if (test_flag) {
#pragma omp atomic
                    num_over_pruned_nodes += 1;
                }
#pragma omp atomic
                influenced_nodes += 1;
            }
        }
    }
    RAFT_LOG_INFO("[consolidate_delete] num_over_pruned_nodes: %d", num_over_pruned_nodes);
    RAFT_LOG_INFO("[consolidate_delete] influenced_nodes: %d", influenced_nodes);

    index_.own_graph(res, updated_graph.view());
    auto d_graph = index_.d_graph();

    raft::copy(d_graph.data_handle(),
                updated_graph.data_handle(),
                graph_size * degree,
                raft::resource::get_cuda_stream(res));

    const IdxT* graph_data = updated_graph.data_handle();
    int* in_edges = index_.host_in_edges().data_handle();
    std::memset(in_edges, 0, graph_size * sizeof(int));
    
#pragma omp parallel for
    for (size_t n = 0; n < graph_size; n++) {
        if (!host_bitset->test(n)) {
            continue;
        }
    
        for (size_t i = 0; i < degree; i++) {
            auto neighbor = updated_graph(n, i);
            in_edges[neighbor]++;
        }
    }

    // for (int i = 0; i < num_streams; i++) {
    //     RAFT_CUDA_TRY(cudaStreamDestroy(stream_pool[i]));
    // }
    // RAFT_LOG_INFO("[consolidate_delete] deleted_nodes: %d", deleted_nodes);
    // RAFT_LOG_INFO("[consolidate_delete] influenced_nodes: %d", influenced_nodes);

    // RAFT_LOG_INFO("[consolidate] host_bitset test value of %ld: %d", 999, static_cast<int>(host_bitset.test(999)));
     
}

template <typename T, typename IdxT>
struct AsyncConsolidateState {
    std::thread worker_thread;
    std::atomic<bool> is_running{false};
    std::atomic<bool> has_result{false};
    std::atomic<bool> shutdown{false};
    
    std::unique_ptr<raft::host_matrix<IdxT, int64_t>> result_graph;
    int influenced_nodes = 0;
    int num_over_pruned_nodes = 0;
    
    // 互斥锁保护共享数据
    std::mutex mutex;
    
    // 构造函数
    AsyncConsolidateState() = default;
    
    // 析构函数，确保线程安全退出
    ~AsyncConsolidateState() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            shutdown = true;
        }
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }
    
    // 禁用拷贝构造和赋值
    AsyncConsolidateState(const AsyncConsolidateState&) = delete;
    AsyncConsolidateState& operator=(const AsyncConsolidateState&) = delete;
    
    // 检查是否有任务正在运行
    bool is_task_running() const {
        return is_running.load();
    }
    
    // 检查是否有结果可用
    bool has_task_result() const {
        return has_result.load();
    }

    bool sync_result(raft::resources const& res, ffanns::neighbors::cagra::index<T, IdxT>& index) {
        // Check if result is available
        if (!has_result.load()) {
            RAFT_LOG_INFO("[AsyncConsolidate] 没有可同步的结果");
            return false;  // No result to sync
        }

        std::lock_guard<std::mutex> lock(mutex);
        
        // Get the current graph size from the index (may have grown since consolidation started)
        const std::size_t new_graph_size = index.size();
        const std::size_t old_graph_size = result_graph->extent(0);
        const std::size_t degree = index.graph_degree();
        
        RAFT_LOG_INFO("[AsyncConsolidate] 开始同步结果, 当前图大小: %d, 旧图大小: %d", 
                    new_graph_size, old_graph_size);
        
        // Create a new graph combining consolidated old graph and newly inserted vectors
        auto new_graph = raft::make_host_matrix<IdxT, int64_t>(new_graph_size, degree);
        
        // Copy consolidated graph for old vectors
        memcpy(new_graph.data_handle(),
            result_graph->data_handle(),
            old_graph_size * degree * sizeof(IdxT));
        
        // Copy graph entries for newly inserted vectors
        if (new_graph_size > old_graph_size) {
            RAFT_LOG_INFO("[AsyncConsolidate] 新增节点数: %d", new_graph_size - old_graph_size);
            memcpy(new_graph.data_handle() + old_graph_size * degree,
                index.graph().data_handle() + old_graph_size * degree,
                (new_graph_size - old_graph_size) * degree * sizeof(IdxT));
        }

        auto edge_log = index.get_edge_log_ptr();
        RAFT_LOG_INFO("[AsyncConsolidate] current edge log size: %d", edge_log->size());
        auto update_edges = edge_log->get_aggregated_updates();
        std::vector<IdxT> sources;
        sources.reserve(update_edges.size());
        for (const auto& pair : update_edges) {
            sources.push_back(pair.first);
        }
        
        auto new_graph_ptr = new_graph.data_handle();

#pragma omp parallel
    {
        std::vector<IdxT> backup_buffer(degree/2);
        for (size_t i = omp_get_thread_num(); i < sources.size(); i += omp_get_num_threads()) {
            const IdxT source = sources[i];
            const auto& targets = update_edges.at(source);
            // RAFT_LOG_INFO("[AsyncConsolidate] source: %d, targets.size(): %d", source, targets.size());
            std::uint32_t rev_edge_idx = degree / 2;
            std::uint32_t num_edges = static_cast<std::uint32_t>(targets.size());
            if (num_edges == 0 || source >= new_graph_size) continue;
            std::uint32_t shift_count = degree - rev_edge_idx - num_edges; 
            auto local_graph_ptr = new_graph_ptr + source * degree;

            if (shift_count > 0) {
                std::memcpy(backup_buffer.data(), 
                          local_graph_ptr + rev_edge_idx,
                          shift_count * sizeof(IdxT));
            }

            std::memcpy(local_graph_ptr + rev_edge_idx,
                   targets.data(),
                   num_edges * sizeof(IdxT));

            if (shift_count > 0) {
                std::memcpy(local_graph_ptr + rev_edge_idx + num_edges,
                            backup_buffer.data(),
                            shift_count * sizeof(IdxT));
            }
        }
    }
        

        auto d_graph = index.d_graph();
        raft::copy(d_graph.data_handle(),
                    new_graph.data_handle(),
                    new_graph_size * degree,
                    raft::resource::get_cuda_stream(res));

        auto host_bitset = index.get_host_delete_bitset_ptr();
        const IdxT* graph_data = new_graph.data_handle();
        int* in_edges = index.host_in_edges().data_handle();
        std::memset(in_edges, 0, new_graph_size * sizeof(int));
        
    #pragma omp parallel for
        for (size_t n = 0; n < new_graph_size; n++) {
            if (!host_bitset->test(n)) {
                continue;
            }
        
            for (size_t i = 0; i < degree; i++) {
                auto neighbor = new_graph(n, i);
                in_edges[neighbor]++;
            }
        }

        index.own_graph(std::move(new_graph));
        
        RAFT_LOG_INFO("[AsyncConsolidate] 图同步成功，影响节点: %d, 过度剪枝节点: %d", 
                    influenced_nodes, num_over_pruned_nodes);

        edge_log->set_consolidating(false);
        edge_log->clear();
        
        // 重置结果标志
        has_result.store(false);
        return true;
    }
    
    // 提交异步consolidate任务
    void submit_task(
        raft::resources const& res,
        ffanns::neighbors::cagra::index<T, IdxT>& index,
        raft::host_matrix_view<T, IdxT> consolidate_dataset
    ) {
        if (is_running.load()) {
            RAFT_LOG_INFO("[AsyncConsolidate] 已有任务正在运行, 跳过此次consolidate");
            return;
        }
        
        // 获取必要的索引信息
        const std::size_t graph_size = index.dataset().extent(0);
        const std::size_t degree = index.graph_degree();
        const std::size_t dim = index.dim();
        
        // 创建graph的副本
        auto updated_graph = raft::make_host_matrix<IdxT, int64_t>(graph_size, degree);
        memcpy(updated_graph.data_handle(),
              index.graph().data_handle(),
              index.graph().size() * sizeof(IdxT));
        
        auto original_host_bitset = index.get_host_delete_bitset_ptr();
        auto host_bitset = std::make_shared<ffanns::core::HostBitSet>(*original_host_bitset);
        
        // 设置状态标志
        is_running.store(true);
        has_result.store(false);
        
        // 确保之前的线程已经结束
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
        
        // 启动工作线程执行consolidate
        worker_thread = std::thread([this, 
                                    host_bitset,
                                    updated_graph = std::move(updated_graph),
                                    consolidate_dataset, 
                                    graph_size, 
                                    degree, 
                                    dim,
                                    &res]() mutable {
            
            // 本地影响统计
            int local_influenced_nodes = 0;
            int local_over_pruned_nodes = 0;
            
            // consolidate主体代码
            RAFT_LOG_INFO("[AsyncConsolidate] 开始后台consolidate，图大小: %d", graph_size);
            
            #pragma omp parallel
            {   
                const std::size_t num_threads = omp_get_num_threads();
                const std::size_t thread_id = omp_get_thread_num();
                cudaStream_t local_stream = raft::resource::get_cuda_stream(res);
                
                for (std::size_t loc = thread_id; loc < graph_size; loc += num_threads) {
                    // 跳过已删除的节点
                    if (!host_bitset->test(loc)) {
                        continue;
                    }
                    
                    // 构建扩展节点集合
                    tsl::robin_set<uint32_t> expanded_nodes_set;
                    std::vector<uint32_t> expanded_nodes_lists;
                    expanded_nodes_lists.reserve(degree);
                    bool change_tag = false;
                    
                    for (std::size_t i = 0; i < degree; i++) {
                        const auto neighbor = updated_graph(loc, i);
                        if (host_bitset->test(neighbor)) {
                            expanded_nodes_set.insert(neighbor);
                            expanded_nodes_lists.push_back(neighbor);
                        } 
                        else // neighbor is marked as deleted
                        {
                            change_tag = true;
                            for (std::size_t j = 0; j < degree; j++) {
                                const auto d_neighbor = updated_graph(neighbor, j);
                                if (d_neighbor != loc && host_bitset->test(d_neighbor)) {
                                    expanded_nodes_set.insert(d_neighbor);
                                }
                            }
                        }
                    }
                    
                    // 检查是否需要执行pruning
                    if (expanded_nodes_set.size() <= degree) {
                        if (expanded_nodes_set.size() < degree) {
                            RAFT_LOG_INFO("[AsyncConsolidate] loc: %d, expanded_nodes_set.size(): %d", 
                                         loc, expanded_nodes_set.size());
                            continue;
                        }
                        if (change_tag) {
                            RAFT_LOG_INFO("[AsyncConsolidate] 相同大小但已删除: loc: %d, size: %d", 
                                         loc, expanded_nodes_set.size());
                        }
                        continue;
                    } else {
                        // 执行pruning
                        IdxT* subgraph_row = updated_graph.data_handle() + loc * degree;
                        bool test_flag = robust_prune(
                            local_stream, 
                            static_cast<uint32_t>(loc), 
                            expanded_nodes_set, 
                            degree, 
                            dim, 
                            1.1f, 
                            consolidate_dataset.data_handle(), 
                            subgraph_row);
                            
                        if (test_flag) {
                            #pragma omp atomic
                            local_over_pruned_nodes += 1;
                        }
                        
                        #pragma omp atomic
                        local_influenced_nodes += 1;
                    }
                }
            }
            
            RAFT_LOG_INFO("[AsyncConsolidate!!!!!!!!] 完成，影响节点: %d, 过度剪枝节点: %d", 
                         local_influenced_nodes, local_over_pruned_nodes);

            
            // 存储结果
            {
                std::lock_guard<std::mutex> lock(mutex);
                result_graph = std::make_unique<raft::host_matrix<IdxT, int64_t>>(
                    std::move(updated_graph));
                influenced_nodes = local_influenced_nodes;
                num_over_pruned_nodes = local_over_pruned_nodes;
            }
            
            // 更新状态
            is_running.store(false);
            has_result.store(true);
        });
    }
};

// 全局单例
template <typename T, typename IdxT>
AsyncConsolidateState<T, IdxT>& get_async_consolidate_state() {
    static AsyncConsolidateState<T, IdxT> instance;
    return instance;
}

template struct AsyncConsolidateState<float, uint32_t>;

template struct AsyncConsolidateState<uint8_t, uint32_t>;

template AsyncConsolidateState<float, uint32_t>& 
    get_async_consolidate_state<float, uint32_t>();

template AsyncConsolidateState<uint8_t, uint32_t>& 
    get_async_consolidate_state<uint8_t, uint32_t>();

}  // namespace ffanns::neighbors::cagra::detail