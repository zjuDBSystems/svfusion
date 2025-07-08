#pragma once

#include "ffanns/neighbors/common.hpp"
#include "ffanns/neighbors/hd_mapper.hpp"
#include "ffanns/neighbors/nn_descent.hpp"
#include "ffanns/distance/distance.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/stream_view.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <raft/core/logger.hpp>

#include <optional>
#include <variant>
#include <bitset>

namespace ffanns::neighbors::cagra
{ 

  namespace graph_build_params
  {
    // TODO: using nn_descent_params = ...
    using nn_descent_params = ffanns::neighbors::nn_descent::index_params;
  }

  struct index_params : ffanns::neighbors::index_params
  {
    /** Degree of input graph for pruning. */
    size_t intermediate_graph_degree = 128;
    /** Degree of output graph. */
    size_t graph_degree = 64;

    /** Use nn_descent here */
    graph_build_params::nn_descent_params graph_build_params;

    /**
     * Whether to use MST optimization to guarantee graph connectivity.
     */
    bool guarantee_connectivity = false;
    // TODO: dataset configuration, we override this as FALSE
    bool attach_dataset_on_build = false;
  };

  // TODO: Support other modes
  enum class search_algo
  {
    /** For large batch sizes. */
    SINGLE_CTA,
    /** For small batch sizes. */
    MULTI_CTA,
    MULTI_KERNEL,
    AUTO
  };
   

  enum class hash_mode { HASH, SMALL, AUTO };

  struct search_params : ffanns::neighbors::search_params
  {
    ffanns::distance::DistanceType metric = ffanns::distance::DistanceType::L2Expanded;
    /** Maximum number of queries to search at the same time (batch size). Auto select when 0.*/
    size_t max_queries = 0;

    /** Number of intermediate search results retained during the search.
     *
     *  This is the main knob to adjust trade off between accuracy and search speed.
     *  Higher values improve the search accuracy.
     */
    size_t itopk_size = 64;

    /** Upper limit of search iterations. Auto select when 0.*/
    size_t max_iterations = 0;

    // In the following we list additional search parameters for fine tuning.
    // Reasonable default values are automatically chosen.

    // TODO: serach_algo
    /** Which search implementation to use. */
    search_algo algo = search_algo::SINGLE_CTA;

    /** Number of threads used to calculate a single distance. 4, 8, 16, or 32. */
    size_t team_size = 0;

    /** Number of graph nodes to select as the starting point for the search in each iteration. aka
     * search width?*/
    size_t search_width = 1;
    /** Lower limit of search iterations. */
    size_t min_iterations = 0;

    /** Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0. */
    size_t thread_block_size = 0;
    /** Hashmap type. Auto selection when AUTO. */
    hash_mode hashmap_mode = hash_mode::AUTO;
    /** Lower limit of hashmap bit length. More than 8. */
    size_t hashmap_min_bitlen = 0;
    /** Upper limit of hashmap fill rate. More than 0.1, less than 0.9.*/
    float hashmap_max_fill_rate = 0.5;

    /** Number of iterations of initial random seed node selection. 1 or more. */
    uint32_t num_random_samplings = 1;
    /** Bit mask used for initial random seed node selection. */
    uint64_t rand_xor_mask = 0x128394;

    /** Whether to use the persistent version of the kernel (only SINGLE_CTA is supported a.t.m.) */
    bool persistent = false;
    /** Persistent kernel: time in seconds before the kernel stops if no requests received. */
    float persistent_lifetime = 2;
    float persistent_device_usage = 1.0;
  };

  struct extend_params
  {
    /** The additional dataset is divided into chunks and added to the graph. This is the knob to
     * adjust the tradeoff between the recall and operation throughput. Large chunk sizes can result
     * in high throughput, but use more working memory (O(max_chunk_size*degree^2)). This can also
     * degrade recall because no edges are added between the nodes in the same chunk. Auto select when
     * 0. */
    uint32_t max_chunk_size = 0;
  };

  static_assert(std::is_aggregate_v<index_params>);
  static_assert(std::is_aggregate_v<search_params>);

  template <typename IdxT>
  struct ReverseEdgeLog {
      struct EdgeUpdate {
          IdxT source;  // 源顶点
          IdxT target;  // 目标顶点（新插入的节点）
      };
      
      std::vector<EdgeUpdate> updates;
      bool is_consolidating{false};
      ReverseEdgeLog() = default;
      
      void record_update(IdxT source, IdxT target) {
          if (is_consolidating) {
              updates.push_back({source, target});
          }
      }
      
      void set_consolidating(bool state) {
          is_consolidating = state;
      }
      
      void clear() { updates.clear();}
      
      [[nodiscard]] size_t size() const {
          return updates.size();
      }

      [[nodiscard]] std::unordered_map<IdxT, std::vector<IdxT>> get_aggregated_updates() const {
          std::unordered_map<IdxT, std::vector<IdxT>> source_to_targets;
          for (const auto& update : updates) {
              source_to_targets[update.source].push_back(update.target);
          }
          return source_to_targets;
      }
  };

  template <typename T, typename IdxT>
  struct index : ffanns::neighbors::index
  {
    using index_params_type  = cagra::index_params;
    using search_params_type = cagra::search_params;
    using index_type         = IdxT;
    using value_type         = T;
    using dataset_index_type = int64_t;
    static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                  "IdxT must be able to represent all values of uint32_t");
    inline static size_t MAX_DEVICE_ROWS = 0;
    inline static size_t MAX_GRAPH_DEVICE_ROWS = 0;
  public:
    [[nodiscard]] static constexpr inline auto get_max_device_rows() noexcept -> size_t { return MAX_DEVICE_ROWS; }
    [[nodiscard]] static constexpr inline auto get_max_graph_device_rows() noexcept -> size_t { return MAX_GRAPH_DEVICE_ROWS; }
    static void set_max_device_rows(std::size_t v) noexcept {
      MAX_DEVICE_ROWS = v;
    }
    static void set_max_graph_device_rows(std::size_t v) noexcept {
      MAX_GRAPH_DEVICE_ROWS = v;
    }
    [[nodiscard]] constexpr inline auto metric() const noexcept -> ffanns::distance::DistanceType
    {
      return metric_;
    }

    /** Total length of the index (number of vectors). */
    [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT
    {
      auto data_rows = dataset_->n_rows();
      return data_rows > 0 ? data_rows : graph_view_.extent(0);
    }

    /** Dimensionality of the data. */
    [[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t { return dataset_->dim(); }

    /** Graph degree */
    [[nodiscard]] constexpr inline auto graph_degree() const noexcept -> uint32_t
    {
      return graph_view_.extent(1);
    }

    // TODO: need to modify the original implementation (device_matrix_view)
    [[nodiscard]] inline auto dataset() const noexcept
    -> raft::host_matrix_view<const T, int64_t, raft::layout_stride>
    {
      auto p = dynamic_cast<strided_dataset<T, int64_t>*>(dataset_.get());
      if (p != nullptr) { return p->view(); }
      auto d = dataset_->dim();
      return raft::make_host_matrix_view<const T, int64_t>(nullptr, 0, d);
    }

    [[nodiscard]] inline auto d_dataset() const noexcept
    -> raft::device_matrix_view<const T, int64_t, raft::layout_stride>
    {
      auto p = dynamic_cast<strided_dataset<T, int64_t>*>(d_dataset_.get());
      if (p != nullptr) { return p->d_view(); }
      auto d = dataset_->dim();
      return raft::make_device_strided_matrix_view<const T, int64_t>(nullptr, 0, d, d);
    }

    /** Dataset [size, dim] */
    [[nodiscard]] inline auto data() const noexcept -> const ffanns::neighbors::dataset<int64_t>&
    {
      return *dataset_;
    }

    [[nodiscard]] inline auto d_data() const noexcept -> const ffanns::neighbors::dataset<int64_t>&
    {
      return *d_dataset_;
    }

    /** neighborhood graph [size, graph-degree] */
    [[nodiscard]] inline auto graph() const noexcept
        -> raft::host_matrix_view<const IdxT, int64_t, raft::row_major>
    {
      return graph_view_;
    }

    [[nodiscard]] inline auto d_graph() noexcept
      -> raft::device_matrix_view<IdxT, int64_t, raft::row_major>
    {
      return d_graph_view_;
    }

    [[nodiscard]] inline auto host_in_edges() noexcept
        -> raft::host_vector_view<int, int64_t>
    {
      return host_in_edges_.view();
    }

    [[nodiscard]] auto d_in_edges() noexcept -> std::shared_ptr<rmm::device_uvector<int>> {
      return d_in_edges_;
    }

    // Don't allow copying the index for performance reasons (try avoiding copying data)
    index(const index &) = delete;
    index(index &&) = default;
    auto operator=(const index &) -> index & = delete;
    auto operator=(index &&) -> index & = default;
    ~index() = default;

    /** Construct an empty index. */
    // TODO: ADD dataset param
    index(raft::resources const &res,
          ffanns::distance::DistanceType metric = ffanns::distance::DistanceType::L2Expanded)
        : ffanns::neighbors::index(),
          metric_(metric),
          graph_(raft::make_host_matrix<IdxT, int64_t>(0, 0)),
          host_in_edges_(raft::make_host_vector<int, int64_t>(0)),
          d_in_edges_(std::make_shared<rmm::device_uvector<int>>(0, raft::resource::get_cuda_stream(res))),
          dataset_(new ffanns::neighbors::empty_dataset<int64_t>(0)),
          d_dataset_(new ffanns::neighbors::empty_dataset<int64_t>(0)),
          hd_mapper_(std::make_shared<ffanns::neighbors::host_device_mapper>(res, 0, MAX_DEVICE_ROWS)),
          graph_hd_mapper_(std::make_shared<ffanns::neighbors::graph_hd_mapper>(res, 0, MAX_GRAPH_DEVICE_ROWS)),
          delete_bitset_(std::make_shared<ffanns::core::bitset<std::uint32_t, int64_t>>(res, 0)),
          tag_to_id_(std::make_shared<rmm::device_uvector<uint32_t>>(0, raft::resource::get_cuda_stream(res))),
          edge_log_(std::make_shared<ReverseEdgeLog<IdxT>>()),
          delete_slots_(std::make_shared<std::vector<std::pair<IdxT, IdxT>>>()),
          free_slots_(std::make_shared<std::vector<std::pair<IdxT, IdxT>>>())
    {
      
    }

    /** Construct an index from dataset and knn_graph arrays. */
    template <typename data_accessor, typename graph_accessor>
    index(raft::resources const &res,
          ffanns::distance::DistanceType metric,
          raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, data_accessor> dataset,
          raft::mdspan<const IdxT, raft::matrix_extent<int64_t>, raft::row_major, graph_accessor> knn_graph,
          std::shared_ptr<rmm::device_uvector<int>> d_in_edges, 
          std::shared_ptr<ffanns::neighbors::host_device_mapper> hd_mapper,
          std::shared_ptr<ffanns::neighbors::graph_hd_mapper> graph_hd_mapper,
          std::shared_ptr<ffanns::core::bitset<std::uint32_t, int64_t>> delete_bitset,
          std::shared_ptr<ffanns::core::HostBitSet> host_delete_bitset,
          std::shared_ptr<ReverseEdgeLog<IdxT>> edge_log,
          std::shared_ptr<std::vector<std::pair<IdxT, IdxT>>> delete_slots,
          std::shared_ptr<std::vector<std::pair<IdxT, IdxT>>> free_slots)
        : ffanns::neighbors::index(),
          metric_(metric),
          graph_(raft::make_host_matrix<IdxT, int64_t>(0, 0)),
          host_in_edges_(raft::make_host_vector<int, int64_t>(0)),
          d_in_edges_(d_in_edges),
          dataset_(make_host_aligned_dataset(res, dataset, 16)),
          hd_mapper_(hd_mapper),
          graph_hd_mapper_(graph_hd_mapper),
          delete_bitset_(delete_bitset),
          host_delete_bitset_(host_delete_bitset),
          edge_log_(edge_log),
          delete_slots_(delete_slots),
          free_slots_(free_slots),
          tag_to_id_(std::make_shared<rmm::device_uvector<uint32_t>>(0, raft::resource::get_cuda_stream(res)))
    {
      RAFT_EXPECTS(dataset.extent(0) == knn_graph.extent(0),
                 "Dataset and knn_graph must have equal number of rows");
      update_graph(res, knn_graph);
      raft::resource::sync_stream(res);
    }

  void update_dataset(raft::resources const& res,
                      raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
  {
    dataset_ = make_host_aligned_dataset(res, dataset, 16);
  }

  void update_dataset(raft::resources const& res,
                      raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
                      raft::device_matrix_view<const T, int64_t, raft::row_major> device_dataset)
  {
    dataset_ = make_host_aligned_dataset(res, dataset, 16);
    d_dataset_ = make_device_aligned_dataset(res, device_dataset, 16);
  }

  /**
   * Replace the graph with a new graph.
   *
   * We create a copy of the graph on the device. The index manages the lifetime of this copy.
   */
  void update_graph(raft::resources const& res,
                    raft::host_matrix_view<const IdxT, int64_t, raft::row_major> knn_graph)
  {
    graph_view_ = knn_graph;
  }

  void own_graph(raft::resources const& res,
                    raft::host_matrix_view<const IdxT, int64_t, raft::row_major> knn_graph) {
    graph_ = raft::make_host_matrix<IdxT, int64_t>(knn_graph.extent(0), knn_graph.extent(1));
    std::memcpy(graph_.data_handle(), 
                knn_graph.data_handle(), 
                knn_graph.size() * sizeof(IdxT));
    graph_view_ = graph_.view();
    RAFT_LOG_INFO("Graph owned - size: %ld x %ld", graph_.extent(0), graph_.extent(1));
  }

  void own_graph(raft::host_matrix<IdxT, int64_t>&& moved_graph) {
    // 直接移动所有权，避免复制
    graph_ = std::move(moved_graph);
    graph_view_ = graph_.view();
    RAFT_LOG_INFO("Graph owned by move - size: %ld x %ld", graph_.extent(0), graph_.extent(1));
  }
  
  void update_d_graph(raft::resources const& res,
                    raft::device_matrix_view<IdxT, int64_t, raft::row_major> knn_graph)
  {
    d_graph_view_ = knn_graph;
  }

  void own_in_edges(raft::host_vector<int, int64_t> in_edges) {
    host_in_edges_ = raft::make_host_vector<int, int64_t>(in_edges.extent(0));
    std::memcpy(host_in_edges_.data_handle(), 
                in_edges.data_handle(), 
                in_edges.size() * sizeof(int));
  }

  void own_in_edges(raft::resources const& res, raft::host_vector<int, int64_t> in_edges) {
    host_in_edges_ = raft::make_host_vector<int, int64_t>(in_edges.extent(0));
    d_in_edges_->resize(in_edges.extent(0), raft::resource::get_cuda_stream(res));
    std::memcpy(host_in_edges_.data_handle(), 
                in_edges.data_handle(), 
                in_edges.size() * sizeof(int));
    raft::copy(d_in_edges_->data(), host_in_edges_.data_handle(), in_edges.size(), raft::resource::get_cuda_stream(res));
  }

  void update_delete_bitset(std::shared_ptr<ffanns::core::bitset<std::uint32_t, int64_t>> delete_bitset) {
    delete_bitset_ = delete_bitset;
  }

  void update_host_delete_bitset(std::shared_ptr<ffanns::core::HostBitSet> delete_bitset) {
    host_delete_bitset_ = delete_bitset;
  }

  void update_tag_to_id(std::shared_ptr<rmm::device_uvector<IdxT>> tag_to_id) {
    tag_to_id_ = tag_to_id;
  }

  [[nodiscard]] constexpr inline auto max_device_rows() const noexcept -> int64_t {
    return max_device_rows_;
  }

  [[nodiscard]] constexpr inline auto max_graph_device_rows() const noexcept -> int64_t {
    return max_graph_device_rows_;
  }

  [[nodiscard]] auto hd_mapper() noexcept -> ffanns::neighbors::host_device_mapper& {
    return *hd_mapper_;
  }

  [[nodiscard]] auto hd_mapper_ptr() const noexcept -> std::shared_ptr<ffanns::neighbors::host_device_mapper> {
    return hd_mapper_;
  }

  [[nodiscard]] auto get_graph_hd_mapper() noexcept -> ffanns::neighbors::graph_hd_mapper& {
    return *graph_hd_mapper_;
  }

  [[nodiscard]] auto get_graph_hd_mapper_ptr() const noexcept -> std::shared_ptr<ffanns::neighbors::graph_hd_mapper> {
    return graph_hd_mapper_;
  }

  [[nodiscard]] auto get_delete_bitset() noexcept -> ffanns::core::bitset<std::uint32_t, int64_t>& {
    return *delete_bitset_;
  }

  [[nodiscard]] auto get_delete_bitset_ptr() noexcept -> std::shared_ptr<ffanns::core::bitset<std::uint32_t, int64_t>> {
    return delete_bitset_;
  }

  [[nodiscard]] auto get_host_delete_bitset() noexcept -> ffanns::core::HostBitSet& {
    return *host_delete_bitset_;
  }

  [[nodiscard]] auto get_host_delete_bitset_ptr() noexcept -> std::shared_ptr<ffanns::core::HostBitSet> {
    return host_delete_bitset_;
  }

  [[nodiscard]] auto get_tag_to_id() noexcept -> rmm::device_uvector<IdxT>& {
    return *tag_to_id_;
  }

  [[nodiscard]] auto get_tag_to_id_ptr() noexcept -> std::shared_ptr<rmm::device_uvector<IdxT>> {
    return tag_to_id_;
  }

  [[nodiscard]] auto get_edge_log_ptr() noexcept -> std::shared_ptr<ReverseEdgeLog<IdxT>> {
    return edge_log_;
  }

  [[nodiscard]] auto get_delete_slots_ptr() noexcept -> std::shared_ptr<std::vector<std::pair<IdxT, IdxT>>> {
    return delete_slots_;
  }

  [[nodiscard]] auto get_free_slots_ptr() noexcept -> std::shared_ptr<std::vector<std::pair<IdxT, IdxT>>> {
    return free_slots_;
  }

  private:
    ffanns::distance::DistanceType metric_;
    raft::host_matrix<IdxT, int64_t, raft::row_major> graph_;
    raft::host_matrix_view<const IdxT, int64_t, raft::row_major> graph_view_;
    raft::device_matrix_view<IdxT, int64_t, raft::row_major> d_graph_view_;
    raft::host_vector<int, int64_t> host_in_edges_;  
    std::shared_ptr<rmm::device_uvector<int>> d_in_edges_; 
    std::shared_ptr<neighbors::host_device_mapper> hd_mapper_;
    std::shared_ptr<neighbors::graph_hd_mapper> graph_hd_mapper_;
    std::unique_ptr<neighbors::dataset<int64_t>> dataset_;
    std::unique_ptr<neighbors::dataset<int64_t>> d_dataset_;
    std::shared_ptr<ffanns::core::bitset<std::uint32_t, int64_t>> delete_bitset_;
    std::shared_ptr<ffanns::core::HostBitSet> host_delete_bitset_;
    std::shared_ptr<rmm::device_uvector<IdxT>> tag_to_id_; 
    std::shared_ptr<ReverseEdgeLog<IdxT>> edge_log_;
    std::shared_ptr<std::vector<std::pair<IdxT, IdxT>>> delete_slots_;
    std::shared_ptr<std::vector<std::pair<IdxT, IdxT>>> free_slots_;
    size_t max_device_rows_ = MAX_DEVICE_ROWS;
    size_t max_graph_device_rows_ = MAX_GRAPH_DEVICE_ROWS;
  };

auto build(raft::resources const& res,
           const cagra::index_params& params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
           raft::host_matrix_view<uint32_t, int64_t, raft::row_major> index_graph,
           // raft::device_matrix_view<float, int64_t, raft::layout_stride> device_dataset_view,
           raft::device_matrix<float, int64_t>& device_dataset_ref,
           // raft::device_matrix_view<uint32_t, int64_t, raft::row_major> device_graph_view,
           raft::device_matrix<uint32_t, int64_t>& device_graph_ref,
           std::shared_ptr<ffanns::core::bitset<std::uint32_t, int64_t>> delete_bitset,
           std::shared_ptr<rmm::device_uvector<uint32_t>> tag_to_id,
           uint32_t start_id, uint32_t end_id)
  -> ffanns::neighbors::cagra::index<float, uint32_t>;

auto build(raft::resources const& res,
          const cagra::index_params& params,
          raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
          raft::host_matrix_view<uint32_t, int64_t, raft::row_major> index_graph,
          // raft::device_matrix_view<uint8_t, int64_t, raft::layout_stride> device_dataset_view,
          raft::device_matrix<uint8_t, int64_t>& device_dataset_ref,
          raft::device_matrix<uint32_t, int64_t>& device_graph_ref,
          std::shared_ptr<ffanns::core::bitset<std::uint32_t, int64_t>> delete_bitset,
          std::shared_ptr<rmm::device_uvector<uint32_t>> tag_to_id,
          uint32_t start_id, uint32_t end_id)
-> ffanns::neighbors::cagra::index<uint8_t, uint32_t>;

void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::host_matrix_view<const float, int64_t, raft::row_major> additional_dataset,
  ffanns::neighbors::cagra::index<float, uint32_t>& idx,
  std::optional<raft::host_matrix_view<float, int64_t, raft::layout_stride>>
    new_dataset_buffer_view                                                         = std::nullopt,
  std::optional<raft::device_matrix_view<float, int64_t, raft::layout_stride>>
    new_d_dataset_buffer_view                                                       = std::nullopt,
  std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::layout_stride>>
    new_graph_buffer_view                                                           = std::nullopt,
  std::optional<raft::device_matrix_view<uint32_t, int64_t>> 
    new_d_graph_buffer_view                                                         = std::nullopt,
  uint32_t start_id = 0, uint32_t end_id = 0);

void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> additional_dataset,
  ffanns::neighbors::cagra::index<uint8_t, uint32_t>& idx,
  std::optional<raft::host_matrix_view<uint8_t, int64_t, raft::layout_stride>>
    new_dataset_buffer_view                                                           = std::nullopt,
  std::optional<raft::device_matrix_view<uint8_t, int64_t, raft::layout_stride>>
    new_d_dataset_buffer_view                                                         = std::nullopt,
  std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::layout_stride>>
    new_graph_buffer_view                                                             = std::nullopt,
  std::optional<raft::device_matrix_view<uint32_t, int64_t>> 
    new_d_graph_buffer_view                                                           = std::nullopt,
  uint32_t start_id = 0, uint32_t end_id = 0);

void search(raft::resources const& res,
            ffanns::neighbors::cagra::search_params const& params,
            ffanns::neighbors::cagra::index<float, uint32_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::host_matrix_view<const float, int64_t, raft::row_major> host_queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const ffanns::neighbors::filtering::base_filter& sample_filter =
              ffanns::neighbors::filtering::none_sample_filter{},
              bool external_flag = false);

void search(raft::resources const& res,
            ffanns::neighbors::cagra::search_params const& params,
            ffanns::neighbors::cagra::index<uint8_t, uint32_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> host_queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const ffanns::neighbors::filtering::base_filter& sample_filter =
              ffanns::neighbors::filtering::none_sample_filter{},
              bool external_flag = false);

void serialize(raft::resources const& handle,
               const std::string& filename,
               const ffanns::neighbors::cagra::index<float, uint32_t>& index,
               bool include_dataset = true);

void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 ffanns::neighbors::cagra::index<float, uint32_t>* index);

void serialize(raft::resources const& handle,
               std::ostream& os,
               const ffanns::neighbors::cagra::index<float, uint32_t>& index,
               bool include_dataset = true);

void deserialize(raft::resources const& handle,
                 std::istream& is,
                 ffanns::neighbors::cagra::index<float, uint32_t>* index);

void lazy_delete(raft::resources const& handle,
                 ffanns::neighbors::cagra::index<float, uint32_t>& index,
                 int64_t start_id,
                 int64_t end_id);

void lazy_delete(raft::resources const& handle,
                  ffanns::neighbors::cagra::index<uint8_t, uint32_t>& index,
                  int64_t start_id,
                  int64_t end_id);


void consolidate_delete(raft::resources const& handle,
                        ffanns::neighbors::cagra::index<float, uint32_t>& index,
                        raft::host_matrix_view<float, int64_t> consolidate_dataset);

void consolidate_delete(raft::resources const& handle,
                        ffanns::neighbors::cagra::index<uint8_t, uint32_t>& index,
                        raft::host_matrix_view<uint8_t, int64_t> consolidate_dataset);

} // ffanns::neighbors::cagra