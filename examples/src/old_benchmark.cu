// Project headers
#include "datasets.hpp"
#include "utils.hpp"
#include <ffanns/neighbors/cagra.hpp>
#include <ffanns/core/bitset.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/random/make_blobs.cuh>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <chrono>
#include <iostream>
#include <thread>
#include <bitset>

using raft::RAFT_NAME;

std::unique_ptr<ffanns::Dataset> prepare_dataset(const std::string& dataset_name) {
    if (dataset_name == "MSTuring-30M") {
        return std::make_unique<ffanns::MSTuringANNS30M>();
    } else if (dataset_name == "Sift1M") {
        return std::make_unique<ffanns::Sift1M>();
    }
    throw std::runtime_error("Unknown dataset: " + dataset_name);
}

raft::device_matrix<float, int64_t> load_query_vectors(
    raft::resources const& handle,
    const std::string& file_name) 
{
    float* query_vectors;
    auto [query_count, query_dim] = read_fbin_file(file_name, query_vectors);
    
    auto d_query_vectors = raft::make_device_matrix<float, int64_t>(
        handle, query_count, query_dim);
    
    RAFT_CUDA_TRY(cudaMemcpyAsync(
        d_query_vectors.data_handle(),
        query_vectors,
        query_count * query_dim * sizeof(float),
        cudaMemcpyHostToDevice,
        raft::resource::get_cuda_stream(handle)
    ));

    // cudaDeviceProp deviceProp = raft::resource::get_device_properties(handle);
    // print_device_properties(deviceProp);

    delete[] query_vectors;
    return d_query_vectors;
}

raft::host_matrix<float, int64_t> load_host_query_vectors(
    const std::string& file_name) 
{
    float* query_vectors;
    auto [query_count, query_dim] = read_fbin_file(file_name, query_vectors);
    
    auto h_query_vectors = raft::make_host_matrix<float, int64_t>(
        query_count, query_dim);
    std::memcpy(
        h_query_vectors.data_handle(),  
        query_vectors,                   
        query_count * query_dim * sizeof(float)   
    );
    return h_query_vectors;
}

template <class SAMPLE_FILTER_T>
std::vector<uint32_t> perform_search(
    raft::resources const& dev_resources,
    ffanns::neighbors::cagra::index<float, uint32_t>& index,  // 移除const
    raft::device_matrix_view<const float, int64_t, raft::row_major> query_vectors,
    raft::host_matrix_view<const float, int64_t, raft::row_major> host_query_vectors,
    SAMPLE_FILTER_T delete_filter) {
    
    using namespace ffanns::neighbors;
    cagra::search_params search_params;
    search_params.itopk_size = 256;
    auto neighbor_indices = raft::make_device_matrix<uint32_t, int64_t>(
        dev_resources, query_vectors.extent(0), 100);
    auto neighbor_distances = raft::make_device_matrix<float, int64_t>(
        dev_resources, query_vectors.extent(0), 100);
    auto neighbor_indices_view = raft::make_device_matrix_view<uint32_t, int64_t>(
        neighbor_indices.data_handle(), query_vectors.extent(0), 100);
    auto neighbor_distances_view = raft::make_device_matrix_view<float, int64_t>(
        neighbor_distances.data_handle(), query_vectors.extent(0), 100);
    
    auto delete_bitset_ptr = index.get_delete_bitset_ptr();
    auto delete_filter2 = ffanns::neighbors::filtering::bitset_filter(delete_bitset_ptr->view());
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
    // for (size_t i = 0; i < 10; i++) {
    //     RAFT_LOG_INFO("neighbor[%ld] = %d", i, current_neighbors[i]);
    // }
    RAFT_CUDA_TRY(cudaStreamSynchronize(raft::resource::get_cuda_stream(dev_resources)));
    // const int num_queries = query_vectors.extent(0);
    // const int topk = 100;
    // for (int i = 0; i < num_queries; i++) {
    //     auto begin_iter = current_neighbors.begin() + i * topk;
    //     auto end_iter = begin_iter + topk;
    //     std::sort(begin_iter, end_iter);
    // }
    return current_neighbors;
}

void run_svfusion(raft::resources const& dev_resources, 
                    raft::host_matrix_view<float, int64_t> host_space_view,
                    raft::device_matrix_view<float, int64_t> device_space_view,
                    ffanns::Dataset& ext_dataset,
                    const ffanns::neighbors::bench_config& config)
{
    using namespace ffanns::neighbors;
    std::vector<std::vector<uint32_t>> step_neighbors;

    auto d_query_vectors = load_query_vectors(dev_resources, ext_dataset.query_filename());
    auto h_query_vectors = load_host_query_vectors(ext_dataset.query_filename());
    ext_dataset.init_data_stream();
    size_t n_dim = ext_dataset.num_dimensions();
    size_t offset = 0;
    
    // 读取初始批次数据
    size_t n_samples = ext_dataset.read_batch(host_space_view, offset, 1000000);
    // size_t n_samples = ext_dataset.read_batch(host_space_view, offset, 100000);
    RAFT_LOG_INFO("Initial batch size = %ld", n_samples);
    offset += n_samples;

    auto dataset_view = raft::make_host_matrix_view<float, int64_t>(
        host_space_view.data_handle(), n_samples, n_dim);
    auto device_dataset_view = raft::make_device_matrix_view<float, int64_t>(
        device_space_view.data_handle(), n_samples, n_dim);
    RAFT_LOG_INFO("[dataset] [first element] = %f", *static_cast<const float*>(dataset_view.data_handle()));

    cagra::index_params index_params;
    index_params.graph_degree = 64;
    auto device_graph_space = raft::make_device_matrix<uint32_t, int64_t>
            (dev_resources, ffanns::neighbors::cagra::index<float, uint32_t>::get_max_device_rows() * 2, index_params.graph_degree); 
    auto device_graph_view = raft::make_device_matrix_view<uint32_t, int64_t>(device_graph_space.data_handle(), n_samples, index_params.graph_degree);
    int64_t max_rows = host_space_view.extent(0);
    auto delete_bitset_ptr = std::make_shared<ffanns::core::bitset<std::uint32_t, int64_t>>(dev_resources, max_rows);
    RAFT_LOG_INFO("Initial delete bitset size = %ld", delete_bitset_ptr->size());
    auto delete_filter = ffanns::neighbors::filtering::bitset_filter(delete_bitset_ptr->view());
    
    auto host_delete_bitset_ptr = std::make_shared<ffanns::core::HostBitSet>(max_rows, true);
    RAFT_LOG_INFO("Initial host delete bitset size = %ld", host_delete_bitset_ptr->count_deleted());
    
    // auto delete_filter = ffanns::neighbors::filtering::none_sample_filter();
    
    // auto removed_indices =
    //       raft::make_device_vector<int64_t, int64_t>(dev_resources, 1000);
    // // std::vector<int64_t> host_value(1, 857);
    // // raft::copy(removed_indices.data_handle(), host_value.data(), 1, raft::resource::get_cuda_stream(dev_resources));
    // thrust::sequence(
    //     raft::resource::get_thrust_policy(dev_resources),
    //     thrust::device_pointer_cast(removed_indices.data_handle()),
    //     thrust::device_pointer_cast(removed_indices.data_handle() + 1000),
    //     1000,  // 初始值为 1000
    //     1);    // 递增步长为 1
    // raft::resource::sync_stream(dev_resources);
    // delete_bitset_ptr->set(dev_resources, removed_indices.view(), 0);
    // auto count_scalar = raft::make_device_scalar<int64_t>(dev_resources, 0);
    // delete_bitset_ptr->count(dev_resources, count_scalar.view());
    // int64_t host_count = 0;
    // raft::copy(&host_count, count_scalar.data_handle(), 1, raft::resource::get_cuda_stream(dev_resources));
    // raft::resource::sync_stream(dev_resources);
    // RAFT_LOG_INFO("Initial delete bitset count = %ld", delete_bitset_ptr->size() - host_count);

    auto tag_to_id = std::make_shared<rmm::device_uvector<uint32_t>>(max_rows, raft::resource::get_cuda_stream(dev_resources));
    auto host_tag_to_id = std::vector<uint32_t>(n_samples);
    auto start = std::chrono::high_resolution_clock::now();
    auto index = cagra::build(dev_resources, index_params, dataset_view, device_dataset_view, device_graph_view, delete_bitset_ptr, tag_to_id, 0, n_samples);
    index.update_host_delete_bitset(host_delete_bitset_ptr);
    raft::copy(host_tag_to_id.data(), 
               tag_to_id->data(),
               n_samples, 
               raft::resource::get_cuda_stream(dev_resources));
    for (size_t i = 0; i < 10; i++) {
        RAFT_LOG_INFO("tag_to_id[%ld] = %d", i, host_tag_to_id[i]);
    }


    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    RAFT_LOG_INFO("Total time: %.2f seconds", duration.count() / 1000.0);
    RAFT_LOG_INFO("- Graph memory: %.2f MB", 
        (index.graph().size() * sizeof(uint32_t)) / (1024.0 * 1024.0));
    RAFT_LOG_INFO("- Dataset memory: %.2f MB",  (index.dataset().size() * sizeof(float)) / (1024.0 * 1024.0));
    RAFT_LOG_INFO("- Dataset size: %ld x %ld", index.dataset().extent(0), index.dataset().extent(1));
    // print_single_node(index.graph(), 0);
    // raft::device_matrix_view<uint32_t, int64_t, raft::row_major> tmp_d_graph_view = index.d_graph();
    // print_device_single_node(dev_resources, tmp_d_graph_view, 0);

    /******* search  */
    // auto step_neighbor = perform_search(dev_resources, index, raft::make_const_mdspan(d_query_vectors.view()), raft::make_const_mdspan(h_query_vectors.view()), delete_filter);
    // step_neighbors.push_back(step_neighbor);
    
    /****************extend */
    cagra::extend_params extend_params;

    for (int i = 0; i < 20; i++) {
        /*****delete */
        // ffanns::neighbors::cagra::lazy_delete(dev_resources, index, 1000 + 8000 * i, 1000 + 8000 * i + 4000);
        // ffanns::neighbors::cagra::lazy_delete(dev_resources, index, 8000 + 16000*i, 16000 + 16000*i);
        /****** */
        RAFT_LOG_INFO("[cagra_build_insert] extend iteration!!!: %d", i);

        auto host_delete_bitset_ptr = index.get_host_delete_bitset_ptr();

        // size_t num_batch_insert = 8000;
        size_t num_batch_insert = 900000;
        n_samples = ext_dataset.read_batch(host_space_view, offset, num_batch_insert);
        auto additional_dataset = raft::make_host_matrix_view<float, int64_t>(host_space_view.data_handle() + offset * n_dim, n_samples, n_dim);
        RAFT_LOG_INFO("[cagra_build_insert] !!! offset=%ld, first_new_data = %f", offset, *static_cast<const float*>(additional_dataset.data_handle()));
        offset += n_samples;
        
        auto updated_dataset = raft::make_host_matrix_view<float, int64_t>(host_space_view.data_handle(), 
            offset, n_dim);
        auto& mapper = index.hd_mapper();
        auto& graph_mapper = index.get_graph_hd_mapper();
        raft::device_matrix_view<float, int64_t> updated_device_dataset;
        raft::device_matrix_view<uint32_t, int64_t> updated_device_graph;
        if (mapper.current_size == mapper.device_capacity) {
            // already full, need to replace
            // TODO: n_samples is not correct
            RAFT_LOG_INFO("[cagra_build_insert] Device Dataset is full, need to replace");
            updated_device_dataset = raft::make_device_matrix_view<float, int64_t>(device_space_view.data_handle(), 
                mapper.device_capacity, n_dim);
        } else {
            RAFT_LOG_INFO("[cagra_build_insert] Device Dataset is not full, extend");
            updated_device_dataset = raft::make_device_matrix_view<float, int64_t>(device_space_view.data_handle(), 
                offset, n_dim);
        }
        if (graph_mapper.current_size == graph_mapper.device_capacity) {
            RAFT_LOG_INFO("[cagra_build_insert] Device Graph is full, need to replace");
            updated_device_graph = raft::make_device_matrix_view<uint32_t, int64_t>(device_graph_space.data_handle(), 
                mapper.device_capacity, index_params.graph_degree);
        }  else {
            RAFT_LOG_INFO("[cagra_build_insert] Device Graph is not full, extend");
            updated_device_graph = raft::make_device_matrix_view<uint32_t, int64_t>(device_graph_space.data_handle(), 
                offset, index_params.graph_degree);
        }
        ffanns::neighbors::cagra::extend(dev_resources, extend_params, additional_dataset, index, 
            updated_dataset, updated_device_dataset, updated_device_graph, offset - n_samples, offset);
        
        // index = cagra::build(dev_resources, index_params, updated_dataset, updated_device_dataset, updated_device_graph, delete_bitset_ptr, tag_to_id, 0, offset);
        
        auto step_neighbor = perform_search(dev_resources, index, raft::make_const_mdspan(d_query_vectors.view()),raft::make_const_mdspan(h_query_vectors.view()),  delete_filter);
        step_neighbors.push_back(step_neighbor);
        
        auto count_scalar = raft::make_device_scalar<int64_t>(dev_resources, 0);
        delete_bitset_ptr->count(dev_resources, count_scalar.view());
        int64_t host_count = 0;
        raft::copy(&host_count, count_scalar.data_handle(), 1, raft::resource::get_cuda_stream(dev_resources));
        raft::resource::sync_stream(dev_resources);
        RAFT_LOG_INFO("[main] step %d: delete bitset count = %ld", i, delete_bitset_ptr->size() - host_count);

        // if ( (i+1) % 10 == 0 && i != 99) {
        //     auto consolidate_host_dataset = raft::make_host_matrix_view<float, int64_t>(host_space_view.data_handle(), offset, n_dim);
        //     ffanns::neighbors::cagra::consolidate_delete(dev_resources, index, consolidate_host_dataset);
        //     // step_neighbor = perform_search(dev_resources, index, raft::make_const_mdspan(d_query_vectors.view()), delete_filter);
        //     // step_neighbors.push_back(step_neighbor);
        // }
    }

    // auto consolidate_device_dataset = raft::make_device_matrix_view<float, int64_t>(device_space_view.data_handle(), offset, n_dim);
    // auto consolidate_host_dataset = raft::make_host_matrix_view<float, int64_t>(host_space_view.data_handle(), offset, n_dim);
    // ffanns::neighbors::cagra::consolidate_delete(dev_resources, index, consolidate_host_dataset);
    save_neighbors_to_binary(step_neighbors, config);

}

int main() {
    raft::device_resources dev_resources;
    RAFT_CUDA_TRY(cudaSetDevice(7));
    // 创建 CUDA stream
    cudaStream_t stream;
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    // Set pool memory resource with 10 GiB initial pool size. All allocations use the same pool.
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
        rmm::mr::get_current_device_resource(), 35 * 1024 * 1024 * 1024ull, 38 * 1024 * 1024 * 1024ull);
    rmm::mr::set_current_device_resource(&pool_mr);
    size_t n_streams = 2;
    raft::resource::set_cuda_stream_pool(dev_resources, std::make_shared<rmm::cuda_stream_pool>(n_streams));

    auto& config = ffanns::neighbors::bench_config::instance();
    config.dataset_name = "MSTuring-30M";
    config.mode = "hd";
    config.chunk_size = 64;

    auto ext_dataset = prepare_dataset(config.dataset_name);
    const size_t max_rows = ext_dataset->num_samples();
    const size_t dim = ext_dataset->num_dimensions();

    auto data_space = raft::make_host_matrix<float, int64_t>(max_rows, dim);
    auto device_data_space = raft::make_device_matrix<float, int64_t>
        (dev_resources, ffanns::neighbors::cagra::index<float, uint32_t>::get_max_device_rows(), dim);

    run_svfusion(dev_resources, data_space.view(), device_data_space.view(), *ext_dataset, config);

    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
}

//  cagra::search(dev_resources, search_params, index, 
//         raft::make_const_mdspan(d_query_vectors.view()), neighbor_indices_view, neighbor_distances_view);