#pragma once

#include <string>
#include <chrono>
#include <librdkafka/rdkafka.h>
#include "streaming_messages.pb.h"
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <ffanns/neighbors/cagra.hpp>

namespace ffanns {
namespace test {

inline void prime_kafka_data_topic(const std::string& brokers,
                                   const std::string& topic,
                                   uint32_t dim) {
  char errstr[512];
  rd_kafka_conf_t* conf = rd_kafka_conf_new();
  rd_kafka_conf_set(conf, "bootstrap.servers", brokers.c_str(), errstr, sizeof(errstr));
  rd_kafka_t* producer = rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));
  if (!producer) return;
  rd_kafka_topic_t* t = rd_kafka_topic_new(producer, topic.c_str(), nullptr);
  if (!t) { rd_kafka_destroy(producer); return; }

  ffanns::streaming::VectorQuery q;
  q.set_query_id(0);
  q.set_operation(ffanns::streaming::OperationType::SEARCH);
  q.set_data_type(ffanns::streaming::DataType::FLOAT32);
  q.set_dimension(dim);
  q.set_k(1);
  q.set_timestamp_us(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
  for (uint32_t i = 0; i < dim; ++i) q.add_vector_float(0.0f);

  std::string payload;
  if (q.SerializeToString(&payload)) {
    rd_kafka_produce(t, /*partition*/ 0, RD_KAFKA_MSG_F_COPY,
                     (void*)payload.data(), payload.size(), nullptr, 0, nullptr);
    rd_kafka_flush(producer, 2000);
  }

  rd_kafka_topic_destroy(t);
  rd_kafka_destroy(producer);
}

template <typename DataT>
inline void warmup_worker(raft::device_resources& dev_resources,
                          ffanns::neighbors::cagra::index<DataT, uint32_t>& index,
                          size_t n_cols,
                          raft::device_matrix<uint32_t, int64_t>& all_neighbors,
                          raft::device_matrix<float, int64_t>& all_distances,
                          int iters = 4) {
  auto warmup_dev = raft::make_device_matrix<DataT, int64_t>(dev_resources, 1, n_cols);
  std::vector<DataT> warmup_host(n_cols, DataT{0});
  auto host_view = raft::make_host_matrix_view<const DataT, int64_t>(warmup_host.data(), 1, n_cols);
  raft::copy(warmup_dev.data_handle(), host_view.data_handle(), n_cols,
             raft::resource::get_cuda_stream(dev_resources));

  ffanns::neighbors::cagra::search_params sp;
  sp.itopk_size = 256;
  sp.max_iterations = 100;
  sp.metric = index.metric();

  auto neighbors_view = raft::make_device_matrix_view<uint32_t, int64_t>(all_neighbors.data_handle(), 1, 1);
  auto distances_view = raft::make_device_matrix_view<float, int64_t>(all_distances.data_handle(), 1, 1);

  auto filter = ffanns::neighbors::filtering::none_sample_filter();
  for (int i = 0; i < iters; ++i) {
    ffanns::neighbors::cagra::search(dev_resources, sp, index, warmup_dev.view(), host_view,
                                     neighbors_view, distances_view, filter, true);
  }
  raft::resource::sync_stream(dev_resources);
}

}  // namespace test
}  // namespace ffanns

