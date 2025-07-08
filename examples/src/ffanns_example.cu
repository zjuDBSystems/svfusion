#include <cstdint>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/core/logger.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <ffanns/neighbors/cagra.hpp>

#include <iostream>
#include <chrono>
#include <thread>

using raft::RAFT_NAME; 

void generate_host_dataset(raft::device_resources const& dev_resources,
                         raft::host_matrix_view<float, int64_t> host_dataset) {
    auto device_dataset = raft::make_device_matrix<float, int64_t>(
        dev_resources, host_dataset.extent(0), host_dataset.extent(1));

    raft::random::RngState r(1234ULL);
    auto labels = raft::make_device_vector<int64_t, int64_t>(
        dev_resources, host_dataset.extent(0));
    raft::random::make_blobs(dev_resources, device_dataset.view(), labels.view());

    raft::copy(host_dataset.data_handle(),
              device_dataset.data_handle(),
              device_dataset.size(),
              dev_resources.get_stream());
}

template<typename IdxT>
void print_single_node(const raft::host_matrix_view<const IdxT, int64_t, raft::row_major>& graph_view,
                      size_t node_id,
                      size_t max_neighbors = 10) 
{
    // 检查 graph_view 是否有效
    RAFT_LOG_INFO("Graph dimensions: %ld x %ld", graph_view.extent(0), graph_view.extent(1));
    RAFT_LOG_INFO("Checking node %ld", node_id);

    // 检查输入参数
    if (graph_view.data_handle() == nullptr) {
        RAFT_LOG_ERROR("Invalid graph view: null data handle");
        return;
    }

    if (node_id >= graph_view.extent(0)) {
        RAFT_LOG_WARN("Node ID %ld exceeds graph size %ld", node_id, graph_view.extent(0));
        return;
    }

    // 直接访问 CPU 上的数据
    const IdxT* node_neighbors = graph_view.data_handle() + node_id * graph_view.extent(1);

    size_t neighbors_to_print = std::min(max_neighbors, static_cast<size_t>(graph_view.extent(1)));
    // 构建邻居字符串
    std::string neighbors_str = "";
    for(size_t i = 0; i < neighbors_to_print; i++) {
        neighbors_str += std::to_string(node_neighbors[i]);
        if (i < neighbors_to_print - 1) {
            neighbors_str += ", ";
        }
    }
    if(graph_view.extent(1) > max_neighbors) {
        neighbors_str += "...";
    }

    RAFT_LOG_INFO("Node %ld information:", node_id);
    RAFT_LOG_INFO("- Total neighbors: %ld", graph_view.extent(1));
    RAFT_LOG_INFO("- Neighbors: [%s]", neighbors_str.c_str());
}

void cagra_build_simple(raft::device_resources const& dev_resources, 
                        raft::host_matrix_view<float, int64_t> data_space,
                        raft::device_matrix_view<float, int64_t> device_data_space)
{
    using namespace ffanns::neighbors;
    auto start = std::chrono::high_resolution_clock::now();
     // Create input arrays.
    int64_t n_samples = 100000;
    int64_t n_dim     = 100;
    
    auto dataset_view = raft::make_host_matrix_view<float, int64_t>(data_space.data_handle(), n_samples, n_dim);
    auto device_dataset_view = raft::make_device_matrix_view<float, int64_t>(device_data_space.data_handle(), n_samples, n_dim);

    generate_host_dataset(dev_resources, dataset_view);
    RAFT_LOG_INFO("[dataset] [first element] = %f", *static_cast<const float*>(dataset_view.data_handle()));

    cagra::index_params index_params;
    auto index = cagra::build(dev_resources, index_params, dataset_view, device_dataset_view);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    RAFT_LOG_INFO("Total time: %.2f seconds", duration.count() / 1000.0);
    RAFT_LOG_INFO("- Graph degree: %d", index.graph_degree());
    RAFT_LOG_INFO("- Graph size: %ld", index.graph().size());
    RAFT_LOG_INFO("- Graph memory: %.2f MB", 
        (index.graph().size() * sizeof(uint32_t)) / (1024.0 * 1024.0));
    RAFT_LOG_INFO("- Dataset size: %ld x %ld", index.dataset().extent(0), index.dataset().extent(1));

    const auto* graph_data = index.graph().data_handle();
    RAFT_LOG_INFO("First neighbor value: %d", graph_data[0]);
    print_single_node(index.graph(), 0);
    // RAFT_LOG_INFO("First 10 values of host_device_mapping:");
    // RAFT_LOG_INFO("size of host_device_mapping = %lu", mapper.host_device_mapping.size());
    // for (size_t i = 0; i < std::min(size_t(10), mapper.host_device_mapping.size()); ++i) {
    //     RAFT_LOG_INFO("[%zu] = %u", i, mapper.host_device_mapping[i]);
    // }
    /***************** extend *****************/
    auto additional_dataset      = raft::make_host_matrix_view<float, int64_t>(data_space.data_handle(), n_samples, n_dim);
    generate_host_dataset(dev_resources, additional_dataset);
    cagra::extend_params extend_params;

    auto updated_dataset = raft::make_host_matrix_view<float, int64_t>(data_space.data_handle(), 
        n_samples + additional_dataset.extent(0), n_dim);
    
    auto& mapper = index.hd_mapper();
    raft::device_matrix_view<float, int64_t> updated_device_dataset;
    if (mapper.current_size == mapper.device_capacity) {
        // already full, need to replace
        updated_device_dataset = raft::make_device_matrix_view<float, int64_t>(device_data_space.data_handle(), 
            n_samples, n_dim);
    } else {
        updated_device_dataset = raft::make_device_matrix_view<float, int64_t>(device_data_space.data_handle(), 
            n_samples + additional_dataset.extent(0), n_dim);
    }
    
    ffanns::neighbors::cagra::extend(dev_resources, extend_params, additional_dataset, index, updated_dataset, updated_device_dataset, std::nullopt);
    // RAFT_LOG_INFO("After extend - Dataset size: %ld x %ld", index.dataset().extent(0), index.dataset().extent(1));
    // RAFT_LOG_INFO("extended first element] = %f", *static_cast<const float*>(index.dataset().data_handle()));
    // RAFT_LOG_INFO("extended first element] = %f", *(static_cast<const float*>(index.dataset().data_handle()))+n_samples);
}


int main() {
    raft::device_resources dev_resources;
    RAFT_CUDA_TRY(cudaSetDevice(6));
    // 创建 CUDA stream
    cudaStream_t stream;
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
        rmm::mr::get_current_device_resource(), 10 * 1024 * 1024 * 1024ull, 12 * 1024 * 1024 * 1024ull);
    rmm::mr::set_current_device_resource(&pool_mr);

    const size_t max_rows = 1000000; // 10M
    const size_t dim = 100;
    auto data_space = raft::make_host_matrix<float, int64_t>(max_rows, dim);
    memset(data_space.data_handle(), 0, sizeof(float) * data_space.size());

    auto dataset_buffer = raft::make_device_matrix<float, int64_t>(dev_resources, 200000, dim);

    cagra_build_simple(dev_resources, data_space.view(), dataset_buffer.view());

    std::this_thread::sleep_for(std::chrono::seconds(2));

    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
}
