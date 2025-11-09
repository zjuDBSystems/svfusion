#include "workload_manager.hpp"
#include "utils.hpp"
#include <ffanns/core/bitset.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <chrono>
#include <rmm/device_uvector.hpp>

namespace ffanns {
namespace test {

template <typename DataT>
WorkloadManager<DataT>::WorkloadManager(const std::string& config_file)
    : config_file_(config_file) {
    parse_config();
}

template <typename DataT>
void WorkloadManager<DataT>::parse_config() {
    YAML::Node config = YAML::LoadFile(config_file_);
    
    // 应该只有一个数据集节点在顶层
    if (config.size() != 1) {
        throw std::runtime_error("配置文件必须只包含一个数据集");
    }
    
    // 获取数据集节点
    auto dataset_node = config.begin();
    dataset_name_ = dataset_node->first.as<std::string>();
    auto dataset_config = dataset_node->second;
    
    // 解析数据集属性
    max_pts_ = dataset_config["max_pts"].as<size_t>();
    res_path_ = dataset_config["res_path"].as<std::string>();
    
    // 解析操作
    operations_.clear();
    for (const auto& op_pair : dataset_config) {
        auto key = op_pair.first.as<std::string>();
        
        // 跳过非数字键（这些是数据集属性）
        if (!std::isdigit(key[0])) {
            continue;
        }
        
        auto op_config = op_pair.second;
        Operation op;
        
        std::string op_type = op_config["operation"].as<std::string>();
        if (op_type == "insert") {
            op.type = OperationType::INSERT;
            op.start = op_config["start"].as<size_t>();
            op.end = op_config["end"].as<size_t>();
        } else if (op_type == "search") {
            op.type = OperationType::SEARCH;
        } else if (op_type == "delete") {
            op.type = OperationType::DELETE;
            op.start = op_config["start"].as<size_t>();
            op.end = op_config["end"].as<size_t>();
        } else if (op_type == "consolidate") {
            op.type = OperationType::CONSOLIDATE;
        } else {
            throw std::runtime_error("未知操作类型: " + op_type);
        }
        
        // 解析QPS和k值（可选字段）
        if (op_config["qps"]) {
            op.qps = op_config["qps"].as<double>();
        }
        if (op_config["k"]) {
            op.k = op_config["k"].as<int>();
        }
        
        operations_.push_back(op);
    }
    
    RAFT_LOG_INFO("已加载数据集 %s 的配置，共有 %zu 个操作", 
                 dataset_name_.c_str(), operations_.size());
}

template <typename DataT>
std::unique_ptr<ffanns::Dataset<DataT>> prepare_dataset(const std::string& dataset_name) {
    if (dataset_name == "msturing-30M") {
        if constexpr (std::is_same_v<DataT, float>) {
            return std::make_unique<ffanns::MSTuringANNS30M>();
        } else {
            throw std::runtime_error("msturing-30M 仅支持 float 数据类型");
        }
    } else if (dataset_name == "sift-1M") {
        if constexpr (std::is_same_v<DataT, uint8_t>) {
            return std::make_unique<ffanns::Sift1M>();
        } else {
            throw std::runtime_error("Sift1M 仅支持 uint8_t 数据类型");
        }
    } else if (dataset_name == "sift-1B") {
        if constexpr (std::is_same_v<DataT, uint8_t>) {
            return std::make_unique<ffanns::Sift1B>();
        } else {
            throw std::runtime_error("Sift1B 仅支持 uint8_t 数据类型");
        }
    } else if (dataset_name == "wikipedia") {
        if constexpr (std::is_same_v<DataT, float>) {
            return std::make_unique<ffanns::WikipediaDataset>();
        } else {
            throw std::runtime_error("wikipedia35M 仅支持 float 数据类型");
        }
    } else if (dataset_name == "msmacro") {
        if constexpr (std::is_same_v<DataT, float>) {
            return std::make_unique<ffanns::MSMacroDataset>();
        } else {
            throw std::runtime_error("msmacro 仅支持 float 数据类型");
        }
    } 
    throw std::runtime_error("未知数据集: " + dataset_name);
}

template <typename DataT>
raft::device_matrix<DataT, int64_t> load_query_vectors(
    raft::resources const& handle,
    const std::string& file_name) 
{
    DataT* query_vectors;
    auto [query_count, query_dim] = read_fbin_file(file_name, query_vectors);
    
    auto d_query_vectors = raft::make_device_matrix<DataT, int64_t>(
        handle, query_count, query_dim);
    
    RAFT_CUDA_TRY(cudaMemcpyAsync(
        d_query_vectors.data_handle(),
        query_vectors,
        query_count * query_dim * sizeof(DataT),
        cudaMemcpyHostToDevice,
        raft::resource::get_cuda_stream(handle)
    ));

    delete[] query_vectors;
    return d_query_vectors;
}

template <typename DataT>
raft::host_matrix<DataT, int64_t> load_host_query_vectors(
    const std::string& file_name) 
{
    DataT* query_vectors;
    auto [query_count, query_dim] = read_fbin_file(file_name, query_vectors);
    
    auto h_query_vectors = raft::make_host_matrix<DataT, int64_t>(
        query_count, query_dim);
    std::memcpy(
        h_query_vectors.data_handle(),  
        query_vectors,                   
        query_count * query_dim * sizeof(DataT)   
    );
    return h_query_vectors;
}

template <typename DataT, class SAMPLE_FILTER_T>
std::vector<uint32_t> perform_search(
    raft::resources const& dev_resources,
    ffanns::neighbors::cagra::index<DataT, uint32_t>& index,
    raft::device_matrix_view<const DataT, int64_t, raft::row_major> query_vectors,
    raft::host_matrix_view<const DataT, int64_t, raft::row_major> host_query_vectors,
    SAMPLE_FILTER_T delete_filter) {
    
    using namespace ffanns::neighbors;
    cagra::search_params search_params;
    search_params.itopk_size = 256;
    search_params.max_iterations = 225;
    search_params.metric = index.metric();
    int topk = 10;
    auto neighbor_indices = raft::make_device_matrix<uint32_t, int64_t>(
        dev_resources, query_vectors.extent(0), topk);
    auto neighbor_distances = raft::make_device_matrix<float, int64_t>(
        dev_resources, query_vectors.extent(0), topk);
    auto neighbor_indices_view = raft::make_device_matrix_view<uint32_t, int64_t>(
        neighbor_indices.data_handle(), query_vectors.extent(0), topk);
    auto neighbor_distances_view = raft::make_device_matrix_view<float, int64_t>(
        neighbor_distances.data_handle(), query_vectors.extent(0), topk);
    
    cagra::search(dev_resources, search_params, index, 
        query_vectors, host_query_vectors,
        neighbor_indices_view, neighbor_distances_view,
        delete_filter,
        true);
    
    std::vector<uint32_t> current_neighbors(neighbor_indices.size());
    raft::copy(current_neighbors.data(), 
               neighbor_indices.data_handle(),
               neighbor_indices.size(), 
               raft::resource::get_cuda_stream(dev_resources));
    
    RAFT_CUDA_TRY(cudaStreamSynchronize(raft::resource::get_cuda_stream(dev_resources)));
    return current_neighbors;
}

template <typename DataT>
void WorkloadManager<DataT>::run_workload(
    raft::resources const& dev_resources,
    raft::host_matrix_view<DataT, int64_t> host_space_view,
    // raft::device_matrix_view<DataT, int64_t> device_space_view,
    ffanns::Dataset<DataT>& ext_dataset,
    const ffanns::neighbors::bench_config& config)
{
    using namespace ffanns::neighbors;
    ffanns::neighbors::cagra::index<float,uint32_t>::set_max_device_rows(1000000);
    ffanns::neighbors::cagra::index<float,uint32_t>::set_max_graph_device_rows(1000000);
    
    // 加载查询向量
    auto d_query_vectors = load_query_vectors<DataT>(dev_resources, ext_dataset.query_filename());
    auto h_query_vectors = load_host_query_vectors<DataT>(ext_dataset.query_filename());
    // 初始化
    ext_dataset.init_data_stream();
    int64_t max_rows = host_space_view.extent(0);
    size_t n_dim = ext_dataset.num_dimensions();
    size_t offset = 0, d_dataset_offset = 0, d_graph_offset = 0;
    
    // 初始化索引参数
    cagra::index_params index_params;
    if (ext_dataset.distance_type() == DistanceType::EUCLIDEAN) {
        index_params.metric = ffanns::distance::DistanceType::L2Expanded;
    } else if (ext_dataset.distance_type() == DistanceType::INNER_PRODUCT) {
        index_params.metric = ffanns::distance::DistanceType::InnerProduct;
    } else {
        RAFT_LOG_INFO("[ERROR]Unsupported distance type: %d", static_cast<int>(ext_dataset.distance_type()));
        return;
    }
    RAFT_LOG_INFO("dataset metric = %d", static_cast<int>(ext_dataset.distance_type()));
    RAFT_LOG_INFO("index_params.metric = %d", static_cast<int>(index_params.metric));
    index_params.graph_degree = 64;
    index_params.intermediate_graph_degree = 128;
    
    // 为图分配空间
    auto graph_space = raft::make_host_matrix<uint32_t, int64_t>(max_rows, index_params.graph_degree);
    
    // 初始化删除位集
    auto delete_bitset_ptr = std::make_shared<ffanns::core::bitset<std::uint32_t, int64_t>>(dev_resources, max_rows);
    auto host_delete_bitset_ptr = std::make_shared<ffanns::core::HostBitSet>(max_rows, true);
    auto delete_filter = ffanns::neighbors::filtering::bitset_filter(delete_bitset_ptr->view());
    
    // 初始化tag_to_id映射
    auto tag_to_id = std::make_shared<rmm::device_uvector<uint32_t>>(max_rows, raft::resource::get_cuda_stream(dev_resources));
    
    // First Operation:: BUILD
    assert(!operations_.empty());
    if (operations_[0].type != OperationType::INSERT) {
        RAFT_LOG_INFO("[ERROR]First operation must be INSERT!!!");
        return;
    }
    const auto& first_op = operations_[0];
    size_t n_samples = ext_dataset.read_batch_pos(host_space_view, offset, first_op.start, first_op.end - first_op.start);
    size_t build_d_datasize = std::min(n_samples, 
        static_cast<size_t>(ffanns::neighbors::cagra::index<DataT, uint32_t>::get_max_device_rows()));
    size_t build_d_graphsize = std::min(n_samples, 
        static_cast<size_t>(ffanns::neighbors::cagra::index<DataT, uint32_t>::get_max_graph_device_rows()));
    RAFT_LOG_INFO("Initial batch size = %ld, build_d_datasize = %ld, build_d_graphsize = %ld", n_samples, 
        build_d_datasize, build_d_graphsize);
    offset += n_samples;
    d_dataset_offset += build_d_datasize;
    d_graph_offset += build_d_graphsize;
    
    auto dataset_view = raft::make_host_matrix_view<DataT, int64_t>(
        host_space_view.data_handle(), n_samples, n_dim);
    // delete the allocation of device_space_view
    auto device_data_owner = raft::make_device_matrix<DataT, int64_t>(dev_resources, 0, n_dim);
    auto graph_view = raft::make_host_matrix_view<uint32_t, int64_t>(
        graph_space.data_handle(), n_samples, index_params.graph_degree);
    auto device_graph_space = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, 0, index_params.graph_degree);
    // auto device_graph_space = raft::make_device_matrix<uint32_t, int64_t>
    //     (dev_resources, ffanns::neighbors::cagra::index<DataT, uint32_t>::get_max_graph_device_rows(), index_params.graph_degree);
    // auto device_graph_view = raft::make_device_matrix_view<uint32_t, int64_t>(
    //     device_graph_space.data_handle(), d_graph_offset, index_params.graph_degree);
    
    auto start = std::chrono::high_resolution_clock::now();
    // Prepare external in_edges owners and views
    auto in_edges_host = raft::make_host_vector<int, int64_t>(max_rows);
    auto in_edges_device = raft::make_device_vector<int, int64_t>(dev_resources, max_rows);
    auto host_in_edges_view = raft::make_host_vector_view<int, int64_t>(in_edges_host.data_handle(), n_samples);
    auto dev_in_edges_view = raft::make_device_vector_view<int, int64_t>(in_edges_device.data_handle(), n_samples);
    auto index = cagra::build(
        dev_resources, index_params, dataset_view, graph_view, device_data_owner, 
        device_graph_space, delete_bitset_ptr, tag_to_id, host_in_edges_view, dev_in_edges_view, 0, n_samples);
    index.update_host_delete_bitset(host_delete_bitset_ptr);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    auto device_space_view = device_data_owner.view();
    RAFT_LOG_INFO("Total time: %.2f seconds", duration);
    RAFT_LOG_INFO("- Graph memory: %.2f MB", 
        (index.graph().size() * sizeof(uint32_t)) / (1024.0 * 1024.0));
    RAFT_LOG_INFO("- Dataset memory: %.2f MB",  (index.dataset().size() * sizeof(DataT)) / (1024.0 * 1024.0));
    RAFT_LOG_INFO("- Dataset size: %ld x %ld", index.dataset().extent(0), index.dataset().extent(1));
    
    log_step_time_csv(config, 0, first_op.type, duration, true, index.hd_mapper_ptr()->miss_rate);
    auto history_delete_count = 0;
    auto& async_consolidate_state = ffanns::neighbors::cagra::detail::get_async_consolidate_state<DataT, uint32_t>();

  
    // for (size_t j = 0; j < 50; j++) {
    //     auto result = perform_search(dev_resources, index, 
    //                     raft::make_const_mdspan(d_query_vectors.view()),
    //                     raft::make_const_mdspan(h_query_vectors.view()),
    //                     delete_filter);
    // }
    auto mapper =  index.hd_mapper_ptr();
    mapper->decay_recent_access(0, dev_resources);
    
    // 执行操作
    for (size_t i = 1; i < operations_.size(); i++) {
        if (async_consolidate_state.has_task_result()) {
            RAFT_LOG_INFO("[run_workload] find result of aync consolidate, start to sync!!!");
            if (async_consolidate_state.sync_result(dev_resources, index)) {
                RAFT_LOG_INFO("[run_workload] sync success!!!");
            } else {
                RAFT_LOG_INFO("[run_workload] sync failed!!!");
            }
        }

        const auto& op = operations_[i];
        RAFT_LOG_INFO("[run_workload] Execute Operation %zu: %d", i + 1, static_cast<int>(op.type));
        auto start_op = std::chrono::high_resolution_clock::now();
        auto end_op = std::chrono::high_resolution_clock::now();

        switch (op.type) {
            case OperationType::INSERT: {
                // 后续插入，扩展索引
                cagra::extend_params extend_params;
                
                size_t n_samples = ext_dataset.read_batch_pos(host_space_view, offset, op.start, op.end - op.start);
                auto additional_dataset = raft::make_host_matrix_view<DataT, int64_t>(
                    host_space_view.data_handle() + offset * n_dim, n_samples, n_dim);
                if constexpr (std::is_same_v<DataT, float>) {
                    RAFT_LOG_INFO("[cagra_build_insert] !!! offset=%ld, first_new_data = %f",
                                    offset, *static_cast<const float*>(additional_dataset.data_handle()));
                } else if constexpr (std::is_same_v<DataT, uint8_t>) {
                    RAFT_LOG_INFO("[cagra_build_insert] !!! offset=%ld, first_new_data = %u",
                                    offset, static_cast<unsigned int>(*static_cast<const uint8_t*>(additional_dataset.data_handle())));
                }
                offset += n_samples;

                auto updated_dataset = raft::make_host_matrix_view<DataT, int64_t>(
                    host_space_view.data_handle(), offset, n_dim);
                auto updated_graph = raft::make_host_matrix_view<uint32_t, int64_t>(
                    graph_space.data_handle(), offset, index_params.graph_degree);
                
                auto& mapper = index.hd_mapper();
                auto& graph_mapper = index.get_graph_hd_mapper();
                
                raft::device_matrix_view<DataT, int64_t> updated_device_dataset;
                raft::device_matrix_view<uint32_t, int64_t> updated_device_graph;
                
                if (mapper.current_size == mapper.device_capacity) {
                    RAFT_LOG_INFO("[run_workload_insert] Device Dataset is full, need to replace");
                    // d_dataset_offset has been set to mapper.device_capacity
                    // d_dataset_offset = mapper.device_capacity;
                } else {
                    RAFT_LOG_INFO("[run_workload_insert] Device Dataset is not full, extend");
                    d_dataset_offset += n_samples;
                }
                updated_device_dataset = raft::make_device_matrix_view<DataT, int64_t>(
                    device_space_view.data_handle(), d_dataset_offset, n_dim);
                
                if (graph_mapper.current_size == graph_mapper.device_capacity) {
                    RAFT_LOG_INFO("[run_workload_insert] Device Graph is full, need to replace");
                    // d_graph_offset = graph_mapper.device_capacity;
                } else {
                    RAFT_LOG_INFO("[run_workload_insert] Device Graph is not full, extend");
                    d_graph_offset += n_samples;
                }
                updated_device_graph = raft::make_device_matrix_view<uint32_t, int64_t>(
                    device_graph_space.data_handle(), d_graph_offset, index_params.graph_degree);
                
                start_op = std::chrono::high_resolution_clock::now();
                ffanns::neighbors::cagra::extend(dev_resources, extend_params, additional_dataset, index,
                    updated_dataset, updated_device_dataset, updated_graph, updated_device_graph, op.start, op.end);
                end_op = std::chrono::high_resolution_clock::now();
                
                // use free_slots, restore offset length
                if (index.size() < offset) {
                    RAFT_LOG_INFO("[workload_manager] using free slots");
                    offset -= n_samples;
                }
                
                break;
            }
            
            case OperationType::SEARCH: {
                start_op = std::chrono::high_resolution_clock::now();
                if (offset > 0) {  // 只有当我们有数据时才进行搜索
                    // Original: Full query batch
                    auto result = perform_search(dev_resources, index, 
                        raft::make_const_mdspan(d_query_vectors.view()),
                        raft::make_const_mdspan(h_query_vectors.view()),
                        delete_filter);
                    
                    // TEST: Only use first query for Multi-CTA testing
                    // auto d_first_query = raft::make_device_matrix_view<const DataT, int64_t, raft::row_major>(
                    //     d_query_vectors.data_handle(), 1, d_query_vectors.extent(1));
                    // auto h_first_query = raft::make_host_matrix_view<const DataT, int64_t, raft::row_major>(
                    //     h_query_vectors.data_handle(), 1, h_query_vectors.extent(1));
                    
                    // printf("[TEST] Running Multi-CTA search with single query (dim=%ld)\n", d_query_vectors.extent(1));
                    
                    // auto result = perform_search(dev_resources, index, 
                    //     d_first_query,
                    //     h_first_query,
                    //     delete_filter);
                    
                    end_op = std::chrono::high_resolution_clock::now();
                    step_neighbors_.push_back(result);

                    save_step_neighbors_to_binary(step_neighbors_, i+1, config);
                }
                break;
            }
            
            case OperationType::DELETE: {
                start_op = std::chrono::high_resolution_clock::now();
                if (offset > 0) {  // 只有当我们有数据时才进行删除
                    ffanns::neighbors::cagra::lazy_delete(dev_resources, index, op.start, op.end);
                    end_op = std::chrono::high_resolution_clock::now();
                    
                    auto count_scalar = raft::make_device_scalar<int64_t>(dev_resources, 0);
                    delete_bitset_ptr->count(dev_resources, count_scalar.view());
                    int64_t host_count = 0;
                    raft::copy(&host_count, count_scalar.data_handle(), 1, 
                              raft::resource::get_cuda_stream(dev_resources));
                    raft::resource::sync_stream(dev_resources);
                    RAFT_LOG_INFO("[run_workload] Delete Count = %ld", delete_bitset_ptr->size() - host_count);
                    history_delete_count += (op.end - op.start);

                    RAFT_LOG_INFO("[run_workload] Activate Count = %ld", offset - (delete_bitset_ptr->size() - host_count));
                    int64_t activate_count = offset - (delete_bitset_ptr->size() - host_count);
                    float threshold = 0.2;
                    // if (i < 100) {threshold = 0.2;}    // try to dynamic adjust threshold
                    // if ((history_delete_count * 1.0f / activate_count) >= threshold) {
                    //     RAFT_LOG_INFO("[run_workload] Consolidated Delete Count = %ld", history_delete_count);
                    // // if (history_delete_count >= 0.1 * 30000000) {
                    //     auto consolidate_host_dataset = raft::make_host_matrix_view<DataT, int64_t>(host_space_view.data_handle(), offset, n_dim);
                    //     auto consolidate_graph = raft::make_host_matrix_view<uint32_t, int64_t>(
                    //         graph_space.data_handle(), offset, index_params.graph_degree);
                    //     ffanns::neighbors::cagra::consolidate_delete(dev_resources, index, consolidate_host_dataset, consolidate_graph);
                    //     end_op = std::chrono::high_resolution_clock::now();
                    //     history_delete_count = 0;
                    //     // async_consolidate_state.submit_task(dev_resources, index, consolidate_host_dataset);
                    //     // auto edge_log = index.get_edge_log_ptr();
                    //     // edge_log->set_consolidating(true);
                    //     auto mapper =  index.hd_mapper_ptr();
                    //     mapper->decay_recent_access(0, dev_resources);
                    // }
                }
                break;
            }
        }

        // if (i == 1) {
        //     for (size_t j = 0; j < 50; j++) {
        //         auto result = perform_search(dev_resources, index, 
        //                         raft::make_const_mdspan(d_query_vectors.view()),
        //                         raft::make_const_mdspan(h_query_vectors.view()),
        //                         delete_filter);
        //     }
        //     auto mapper =  index.hd_mapper_ptr();
        //     mapper->decay_recent_access(0, dev_resources);
        // }

        // auto end_op = std::chrono::high_resolution_clock::now();
        auto duration_op = std::chrono::duration_cast<std::chrono::milliseconds>(end_op - start_op);
        double duration = std::chrono::duration<double>(end_op - start_op).count();
        RAFT_LOG_INFO("Operation %zu (%d) Time: %.2f seconds", i + 1, static_cast<int>(op.type), duration);
        log_step_time_csv(config, i, op.type, duration, false, index.hd_mapper_ptr()->miss_rate);
        index.hd_mapper_ptr()->miss_rate = 0.0f;
        auto mapper =  index.hd_mapper_ptr();
        auto d_in_edges = index.d_in_edges();
        std::string count_file_name = bench_config::instance().get_count_log_path(i);
        // mapper->snapshot_access_counts(count_file_name, nullptr, d_in_edges.data_handle(), raft::resource::get_cuda_stream(dev_resources));
        // mapper->decay_recent_access(0, dev_resources);
    }
    
    // 保存搜索结果
    if (!step_neighbors_.empty()) {
        save_neighbors_to_binary(step_neighbors_, config);
    }
}

template <typename DataT>
void WorkloadManager<DataT>::execute(raft::resources const& dev_resources) {
    // 准备数据集
    auto& config = ffanns::neighbors::bench_config::instance();
    config.dataset_name = dataset_name_;
    config.mode = "hd";
    config.chunk_size = 1024;
    
    auto ext_dataset = prepare_dataset<DataT>(config.dataset_name);
    const size_t max_rows = std::min(max_pts_, ext_dataset->num_samples());
    const size_t dim = ext_dataset->num_dimensions();
    
    // 为数据分配内存
    auto data_space = raft::make_host_matrix<DataT, int64_t>(max_rows, dim);
    // auto device_data_space = raft::make_device_matrix<DataT, int64_t>
    //     (dev_resources, ffanns::neighbors::cagra::index<DataT, uint32_t>::get_max_device_rows(), dim);
    
    // 运行工作负载
    // run_workload(dev_resources, data_space.view(), device_data_space.view(), *ext_dataset, config);
    run_workload(dev_resources, data_space.view(), *ext_dataset, config);
}

template class ffanns::test::WorkloadManager<float>;
template class ffanns::test::WorkloadManager<uint8_t>;

} // namespace test
} // namespace ffanns
