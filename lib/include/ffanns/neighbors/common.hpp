#pragma once

#include <cstdint>

#include "ffanns/distance/distance.hpp"
// #include "ffanns/core/bitset.hpp"
#include <ffanns/core/bitset.hpp>

#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>   // get_device_for_address
#include <raft/util/integer_utils.hpp>  // rounding up

#include <raft/core/detail/macros.hpp>

#include <memory>
#include <numeric>
#include <type_traits>
#include <filesystem>
#include <fstream>


#ifdef __cpp_lib_bitops
#include <bit>
#endif

namespace ffanns::neighbors {

using raft::RAFT_NAME;

/** The base for approximate KNN index structures. */
struct index {};

/** The base for KNN index parameters. */
struct index_params {
  /** Distance type. */
  ffanns::distance::DistanceType metric = ffanns::distance::DistanceType::L2Expanded;
  /** The argument used by some distance metrics. */
  float metric_arg = 2.0f;
};

struct search_params {};

/** Two-dimensional dataset; maybe owning, maybe compressed, maybe strided. */
template <typename IdxT>
struct dataset {
  using index_type = IdxT;
  /**  Size of the dataset. */
  [[nodiscard]] virtual auto n_rows() const noexcept -> index_type = 0;
  /** Dimensionality of the dataset. */
  [[nodiscard]] virtual auto dim() const noexcept -> uint32_t = 0;
  // TODO: Ensure the dataset are seprate with the index. 
  [[nodiscard]] virtual auto is_owning() const noexcept -> bool = 0;
  virtual ~dataset() noexcept                                   = default;
};

template <typename IdxT>
struct empty_dataset : public dataset<IdxT> {
  using index_type = IdxT;
  uint32_t suggested_dim;
  explicit empty_dataset(uint32_t dim) noexcept : suggested_dim(dim) {}
  [[nodiscard]] auto n_rows() const noexcept -> index_type final { return 0; }
  [[nodiscard]] auto dim() const noexcept -> uint32_t final { return suggested_dim; }
  [[nodiscard]] auto is_owning() const noexcept -> bool final { return true; }
};

template <typename DataT, typename IdxT>
struct strided_dataset : public dataset<IdxT> {
  using index_type = IdxT;
  using value_type = DataT;
  using host_view_type  = raft::host_matrix_view<const value_type, index_type, raft::layout_stride>;
  using device_view_type  = raft::device_matrix_view<const value_type, index_type, raft::layout_stride>;
  
  [[nodiscard]] auto n_rows() const noexcept -> index_type final { return view().extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t final
  {
    return static_cast<uint32_t>(view().extent(1));
  }
  /** Leading dimension of the dataset. */
  [[nodiscard]] constexpr auto stride() const noexcept -> uint32_t
  {
    auto v = view();
    return static_cast<uint32_t>(v.stride(0) > 0 ? v.stride(0) : v.extent(1));
  }
  /** Get the view of the data. */
  // TODO: Temporarily using host_view_type
  // [[nodiscard]] virtual auto view() const noexcept -> view_type = 0;
  [[nodiscard]] virtual auto view() const noexcept -> host_view_type= 0;
  [[nodiscard]] virtual auto d_view() const noexcept -> device_view_type = 0;
};

// suppose host_strides_dataset is non_owning_dataset
template <typename DataT, typename IdxT>
struct host_strided_dataset : public strided_dataset<DataT, IdxT> {
  using index_type = IdxT;
  using value_type = DataT;
  using view_type  = typename strided_dataset<value_type, index_type>::host_view_type;
  using d_view_type = typename strided_dataset<value_type, index_type>::device_view_type;
  view_type data;
  explicit host_strided_dataset(view_type v) noexcept : data(v) {};
  [[nodiscard]] auto is_owning() const noexcept -> bool final { return false; }
  [[nodiscard]] auto view() const noexcept -> view_type final { return data; };
  [[nodiscard]] auto d_view() const noexcept -> d_view_type final { 
    return raft::make_device_matrix_view<const value_type, index_type>(nullptr, 0, 0);
  };
};

template <typename DataT, typename IdxT>
struct device_strided_dataset : public strided_dataset<DataT, IdxT> {
  using index_type = IdxT;
  using value_type = DataT;
  using h_view_type  = typename strided_dataset<value_type, index_type>::host_view_type;
  using view_type = typename strided_dataset<value_type, index_type>::device_view_type;
  view_type data;
  explicit device_strided_dataset(view_type v) noexcept : data(v) {}
  [[nodiscard]] auto is_owning() const noexcept -> bool final { return false; }
  [[nodiscard]] auto view() const noexcept -> h_view_type final {
    return raft::make_host_matrix_view<const value_type, index_type>(nullptr, 0, 0);
  }
  [[nodiscard]] auto d_view() const noexcept -> view_type final { return data; }
};

template <typename DatasetT>
struct is_strided_dataset : std::false_type {};

template <typename DataT, typename IdxT>
struct is_strided_dataset<strided_dataset<DataT, IdxT>> : std::true_type {};

template <typename DataT, typename IdxT>
struct is_strided_dataset<host_strided_dataset<DataT, IdxT>> : std::true_type {};

template <typename DataT, typename IdxT>
struct is_strided_dataset<device_strided_dataset<DataT, IdxT>> : std::true_type {};

template <typename DatasetT>
inline constexpr bool is_strided_dataset_v = is_strided_dataset<DatasetT>::value;

template <typename SrcT>
auto make_host_strided_dataset(const raft::resources& res, const SrcT& src, uint32_t required_stride)
  -> std::unique_ptr<strided_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using extents_type = typename SrcT::extents_type;
  using value_type   = typename SrcT::value_type;
  using index_type   = typename SrcT::index_type;
  using layout_type  = typename SrcT::layout_type;
  static_assert(extents_type::rank() == 2, "The input must be a matrix.");
  static_assert(std::is_same_v<layout_type, raft::layout_right> ||
                  std::is_same_v<layout_type, raft::layout_right_padded<value_type>> ||
                  std::is_same_v<layout_type, raft::layout_stride>,
                "The input must be row-major");
  RAFT_EXPECTS(src.extent(1) <= required_stride,
               "The input row length must be not larger than the desired stride.");
  const uint32_t src_stride = src.stride(0) > 0 ? src.stride(0) : src.extent(1);
  // TODO: 暂时强制host_strided_matrix_view为视图
  RAFT_EXPECTS(src.stride(1) <= 1, "Data must be row-major");
  RAFT_EXPECTS(required_stride == src_stride, "Stride must match the required stride");
  // TODO: there is no make_host_strided_matrix_view api
  return std::make_unique<host_strided_dataset<value_type, index_type>>(
    raft::make_host_matrix_view<const value_type, index_type>(
      src.data_handle(), src.extent(0), src.extent(1)));
}

template <typename SrcT>
auto make_host_aligned_dataset(const raft::resources& res, const SrcT& src, uint32_t align_bytes = 16)
  -> std::unique_ptr<strided_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type       = typename SrcT::value_type;
  constexpr size_t kSize = sizeof(value_type);
  uint32_t required_stride =
    raft::round_up_safe<size_t>(src.extent(1) * kSize, std::lcm(align_bytes, kSize)) / kSize;
  return make_host_strided_dataset(res, src, required_stride);
}

template <typename SrcT>
auto make_device_strided_dataset(const raft::resources& res, const SrcT& src, uint32_t required_stride)
  -> std::unique_ptr<strided_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using extents_type = typename SrcT::extents_type;
  using value_type   = typename SrcT::value_type;
  using index_type   = typename SrcT::index_type;
  using layout_type  = typename SrcT::layout_type;
  static_assert(extents_type::rank() == 2, "The input must be a matrix.");
  static_assert(std::is_same_v<layout_type, raft::layout_right> ||
                  std::is_same_v<layout_type, raft::layout_right_padded<value_type>> ||
                  std::is_same_v<layout_type, raft::layout_stride>,
                "The input must be row-major");
  RAFT_EXPECTS(src.extent(1) <= required_stride,
               "The input row length must be not larger than the desired stride.");
  cudaPointerAttributes ptr_attrs;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&ptr_attrs, src.data_handle()));
  auto* device_ptr             = reinterpret_cast<value_type*>(ptr_attrs.devicePointer);
  const uint32_t src_stride    = src.stride(0) > 0 ? src.stride(0) : src.extent(1);
  const bool device_accessible = device_ptr != nullptr;
  const bool row_major         = src.stride(1) <= 1;
  const bool stride_matches    = required_stride == src_stride;

  if (!device_accessible || !row_major || !stride_matches) {
    throw std::runtime_error("Invalid dataset properties");
  }
  return std::make_unique<device_strided_dataset<value_type, index_type>>(
    raft::make_device_strided_matrix_view<const value_type, index_type>(
      device_ptr, src.extent(0), src.extent(1), required_stride));

  // if (device_accessible && row_major && stride_matches) {
  //   // Everything matches: make a non-owning dataset
  //   return std::make_unique<device_strided_dataset<value_type, index_type>>(
  //     raft::make_device_strided_matrix_view<const value_type, index_type>(
  //       device_ptr, src.extent(0), src.extent(1), required_stride));
  // }
  // Something is wrong: have to make a copy and produce an owning dataset
  // auto out_layout =
  //   raft::make_strided_layout(src.extents(), std::array<index_type, 2>{required_stride, 1});
  // auto out_array =
  //   raft::make_device_matrix<value_type, index_type>(res, src.extent(0), required_stride);

  // using out_mdarray_type          = decltype(out_array);
  // using out_layout_type           = typename out_mdarray_type::layout_type;
  // using out_container_policy_type = typename out_mdarray_type::container_policy_type;
  // using out_owning_type =
  //   owning_dataset<value_type, index_type, out_layout_type, out_container_policy_type>;

  // RAFT_CUDA_TRY(cudaMemsetAsync(out_array.data_handle(),
  //                               0,
  //                               out_array.size() * sizeof(value_type),
  //                               raft::resource::get_cuda_stream(res)));
  // RAFT_CUDA_TRY(cudaMemcpy2DAsync(out_array.data_handle(),
  //                                 sizeof(value_type) * required_stride,
  //                                 src.data_handle(),
  //                                 sizeof(value_type) * src_stride,
  //                                 sizeof(value_type) * src.extent(1),
  //                                 src.extent(0),
  //                                 cudaMemcpyDefault,
  //                                 raft::resource::get_cuda_stream(res)));

  // return std::make_unique<out_owning_type>(std::move(out_array), out_layout);
}

template <typename SrcT>
auto make_device_aligned_dataset(const raft::resources& res, const SrcT& src, uint32_t align_bytes = 16)
  -> std::unique_ptr<strided_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type       = typename SrcT::value_type;
  constexpr size_t kSize = sizeof(value_type);
  uint32_t required_stride =
    raft::round_up_safe<size_t>(src.extent(1) * kSize, std::lcm(align_bytes, kSize)) / kSize;
  return make_device_strided_dataset(res, src, required_stride);
}

namespace filtering {

enum class FilterType { None, Bitset };

struct base_filter {
  virtual ~base_filter()                     = default;
  virtual FilterType get_filter_type() const = 0;
};

/* A filter that filters nothing. This is the default behavior. */
struct none_sample_filter : public base_filter {
  inline _RAFT_HOST_DEVICE bool operator()(
    // the index of the current sample
    const uint32_t sample_ix) const;
  
  FilterType get_filter_type() const override { return FilterType::None; }
};

/**
 * @brief Filter an index with a bitset
 *
 * @tparam bitset_t Data type of the bitset
 * @tparam index_t Indexing type
 */
template <typename bitset_t, typename index_t>
struct bitset_filter : public base_filter {
  using view_t = ffanns::core::bitset_view<bitset_t, index_t>;

  // View of the bitset to use as a filter
  const view_t bitset_view_;

  bitset_filter(const view_t bitset_for_filtering);
  inline _RAFT_HOST_DEVICE bool operator()(
    // the index of the current sample
    const uint32_t sample_ix) const;

  FilterType get_filter_type() const override { return FilterType::Bitset; }

  view_t view() const { return bitset_view_; }

  // TODO: implement to_csr interface
  // template <typename csr_matrix_t>
  // void to_csr(raft::resources const& handle, csr_matrix_t& csr);
};

} // namespace filtering

struct bench_config {
    std::string dataset_name{"default_dataset_name"};
    std::string base_log_dir{"/data/workspace/svfusion/results"};
    std::string mode{"device"};
    size_t chunk_size{16};
    
    bool enable_logging{true};
    std::string log_level{"INFO"};
    
    void set_workload_tag(std::string tag) { workload_tag_ = tag; }
    
    std::string get_miss_log_path() const {
        std::filesystem::path log_path = base_log_dir;
        log_path /= dataset_name;
        log_path /= "search_miss.csv";
        std::filesystem::create_directories(log_path.parent_path());
        return log_path.string();
    }

    std::string get_insert_log_path() const {
        std::filesystem::path log_path = base_log_dir;
        log_path /= dataset_name;
        log_path /= (mode + "_insert_" + std::to_string(chunk_size) + ".csv");
        std::filesystem::create_directories(log_path.parent_path());
        return log_path.string();
    }

    std::string get_count_log_path(const std::uint32_t chunk_id) const {
        std::filesystem::path log_path = base_log_dir;
        log_path /= dataset_name;
        log_path /= ("access_counts_snapshot_step_" + std::to_string(chunk_id) + ".csv");
        std::filesystem::create_directories(log_path.parent_path());
        return log_path.string();
    }

    std::string get_search_path() const {
        std::filesystem::path log_path = base_log_dir;
        log_path /= dataset_name;
        log_path /= "search_results";
        log_path /= ("search_" + std::to_string(chunk_size) + ".bin");
        std::filesystem::create_directories(log_path.parent_path());
        return log_path.string();
    }

    std::string get_result_path() const {
        std::filesystem::path log_path = base_log_dir;
        log_path /= dataset_name;
        log_path /= "search_results";
        log_path /= workload_tag_;
        log_path /= ("search_" + std::to_string(chunk_size) + ".bin");
        std::filesystem::create_directories(log_path.parent_path());
        return log_path.string();
    }

    std::string get_time_log_path() const {
      std::filesystem::path log_path = base_log_dir;
      log_path /= dataset_name;
      log_path /= "search_results";
      log_path /= workload_tag_;
      log_path /= ("time_log.csv");
      std::filesystem::create_directories(log_path.parent_path());
      return log_path.string();
    }

    static bench_config& instance() {
        static bench_config config;
        return config;
    }

private:
    // private constructor to avoid direct instantiation
    std::string workload_tag_;
    bench_config() = default;
};

}    