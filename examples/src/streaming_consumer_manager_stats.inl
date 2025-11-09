// Implementation of lightweight stats printer for StreamingConsumerManager<DataT>
namespace ffanns { namespace test {

template <typename DataT>
void StreamingConsumerManager<DataT>::print_stats() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
    
    uint64_t searches = total_searches_.load();
    uint64_t messages = total_messages_.load();
    uint64_t total_latency = total_latency_us_.load();
    
    double throughput = elapsed > 0 ? static_cast<double>(messages) / elapsed : 0.0;

    if (use_multi_stream_) {
        double avg_latency = searches > 0 ? static_cast<double>(total_latency) / searches : 0.0;
        std::cout << "Messages: " << messages
                  << ", Searches: " << searches
                  << ", Throughput: " << std::fixed << std::setprecision(1) << throughput << " msg/s"
                  << ", Avg Latency: " << std::fixed << std::setprecision(0) << avg_latency << " μs"
                  << ", Streams: " << num_search_streams_ << std::endl;
    } else {
        double avg_latency = searches > 0 ? static_cast<double>(total_latency) / searches : 0.0;
        std::cout << "Messages: " << messages
                  << ", Searches: " << searches
                  << ", Throughput: " << std::fixed << std::setprecision(1) << throughput << " msg/s"
                  << ", Avg Latency: " << std::fixed << std::setprecision(0) << avg_latency << " μs" << std::endl;
    }
}

}} // namespace ffanns::test
