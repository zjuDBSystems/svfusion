#pragma once

#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <raft/core/resources.hpp>
#include "datasets.hpp"
#include <ffanns/neighbors/cagra.hpp>
#include <ffanns/neighbors/cagra_async.hpp>

namespace ffanns {
namespace test {

enum class OperationType {
    INSERT,
    SEARCH,
    DELETE,
    CONSOLIDATE
};

struct Operation {
    OperationType type;
    size_t start = 0;
    size_t end = 0;
    double qps = 1000.0;  // QPS控制，默认1000
    int k = 10;           // top-k结果数量（仅搜索操作使用）
};

template <typename DataT>
class WorkloadManager {
public:
    WorkloadManager(const std::string& config_file);
    
    void execute(raft::resources const& dev_resources);

private:
    void parse_config();
    void run_workload(
        raft::resources const& dev_resources,
        raft::host_matrix_view<DataT, int64_t> host_space_view,
        ffanns::Dataset<DataT>& ext_dataset,
        const ffanns::neighbors::bench_config& config);
    
    std::string config_file_;
    std::string dataset_name_;
    size_t max_pts_;
    std::string res_path_;
    std::vector<Operation> operations_;

    // 结果跟踪
    std::vector<std::vector<uint32_t>> step_neighbors_;
};

} // namespace test
} // namespace ffanns