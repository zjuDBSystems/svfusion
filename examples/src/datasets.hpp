#pragma once

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/core/logger.hpp>

#include <cstddef>  
#include <string>
#include <iostream>
#include <fstream>  

using raft::RAFT_NAME; 

namespace ffanns
{

    #define MILLION 1000000UL

    enum class DistanceType
    {
        EUCLIDEAN,
        INNER_PRODUCT,
        COSINE
    };

    template <typename DataT>
    class Dataset
    {
    public:
        using view_type = raft::host_matrix_view<DataT, int64_t, raft::layout_stride>;

        Dataset() = default;
        virtual ~Dataset() = default;

        size_t num_samples() const { return n_samples_; }
        size_t num_dimensions() const { return n_dimensions_; }
        size_t num_queries() const { return n_queries_; }

        std::string dataset_filename() const { return ds_fn_; }
        std::string query_filename() const { return qs_fn_; }
        std::string groundtruth_filename() const { return gt_fn_; }

        // TODO: other attributes
        // self.private_nq = 10000

        virtual DistanceType distance_type() const = 0;

        void init_data_stream() {
            data_file_.close(); 
            data_file_.open(ds_fn_, std::ios::binary);
            if (!data_file_) {
                throw std::runtime_error("Cannot open dataset file: " + ds_fn_);
            }
            
            int32_t nvecs, dim;
            data_file_.read(reinterpret_cast<char*>(&nvecs), sizeof(int32_t));
            data_file_.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
            RAFT_LOG_INFO("nvecs = %d, dim = %d", nvecs, dim);
            if (dim != static_cast<int32_t>(n_dimensions_)) {
                throw std::runtime_error("Dimension mismatch in data file");
            }
            
            current_pos_ = 0;
        }

        size_t read_batch(view_type data_space, size_t offset, size_t batch_size) {
            if (!data_file_.is_open()) {
                throw std::runtime_error("Data file not initialized. Call init_data_stream() first.");
            }

            if (offset >= data_space.extent(0)) {
                return 0;
            }

            batch_size = std::min(batch_size, data_space.extent(0) - offset);
            DataT* target = data_space.data_handle() + offset * n_dimensions_;
            
            data_file_.read(reinterpret_cast<char*>(target),
                        batch_size * n_dimensions_ * sizeof(DataT));

            if (!data_file_) {
                throw std::runtime_error("Error reading data file");
            }

            current_pos_ += batch_size;
            return batch_size;  
        }

        size_t read_batch_pos(view_type data_space, size_t offset, size_t start_idx, size_t batch_size) {
            if (!data_file_.is_open()) {
                throw std::runtime_error("Data file not initialized. Call init_data_stream() first.");
            }

            if ((offset + batch_size) > data_space.extent(0)) {
                return 0;
            }

            std::streampos file_offset = sizeof(int32_t)*2 + start_idx * n_dimensions_ * sizeof(DataT);
            data_file_.seekg(file_offset, std::ios::beg);

            batch_size = std::min(batch_size, data_space.extent(0) - offset);
            DataT* target = data_space.data_handle() + offset * n_dimensions_;
            
            data_file_.read(reinterpret_cast<char*>(target),
                        batch_size * n_dimensions_ * sizeof(DataT));

            if (!data_file_) {
                throw std::runtime_error("Error reading data file");
            }

            return batch_size;  
        }
        
        void reset_stream() {
            if (data_file_.is_open()) {
                data_file_.clear();  
                data_file_.seekg(sizeof(int32_t)*2, std::ios::beg);  
                current_pos_ = 0;
            }
        }

        void get_groundtruth(size_t k=0) {
            assert(!gt_fn_.empty());
            std::ifstream gt(gt_fn_, std::ios::binary);
            if (!gt) {
                throw std::runtime_error("Cannot open groundtruth file: " + gt_fn_);
            }

            uint32_t n, d;
            gt.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
            gt.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
            RAFT_LOG_INFO("[get_groundtruth] n = %d, d = %d", n, d);
        }
        
        size_t current_pos() const { return current_pos_; }
        bool eof() const { return current_pos_ >= n_samples_; }

    protected:
        size_t n_samples_M_{0};
        size_t n_samples_{0};
        size_t n_dimensions_{0};
        size_t n_queries_{0};

        std::string ds_fn_;
        std::string qs_fn_;
        std::string gt_fn_;
    
    private:
        std::ifstream data_file_;
        size_t current_pos_{0};

    };

    /* MSTuringANNS10M */
    class MSTuringANNS30M : public Dataset<float>
    {
    public:
        MSTuringANNS30M()
        {
            n_samples_M_ = 12;
            n_samples_ = n_samples_M_ * MILLION; // 10M samples
            n_dimensions_ = 100;   
            n_queries_ = 10000;    // 10K queries

            ds_fn_ = "/data2/pyc/data/msturing30M/msturing30M.fbin";
            qs_fn_ = "/data2/pyc/data/msturing30M/testQuery10K.fbin";
            gt_fn_ = "/data2/pyc/data/msturing30M/msturing30M_gt.ivecs";
        }

        DistanceType distance_type() const override
        {
            return DistanceType::EUCLIDEAN;
        }
    };

    class Sift1M : public Dataset<uint8_t>
    {
    public:
        Sift1M()
        {
            n_samples_M_ = 1;
            n_samples_ = n_samples_M_ * MILLION; // 10M samples
            n_dimensions_ = 128;   
            n_queries_ = 10000;    // 10K queries

            ds_fn_ = "/data2/pyc/data/sift1M/sift_base.fbin";
            qs_fn_ = "/data2/pyc/data/sift1M/sift_query.fbin";
            gt_fn_ = "/data2/pyc/data/sift1M/sift_groundtruth.ivecs";
        }

        DistanceType distance_type() const override
        {
            return DistanceType::EUCLIDEAN;
        }
    };

    class Sift1B : public Dataset<uint8_t>
    {
    public:
        Sift1B()
        {
            n_samples_M_ = 100;
            n_samples_ = n_samples_M_ * MILLION; // 10M samples
            n_dimensions_ = 128;   
            n_queries_ = 10000;    // 10K queries

            ds_fn_ = "/data2/pyc/data/sift1B/base.1B.u8bin";
            qs_fn_ = "/data2/pyc/data/sift1B/query.public.10K.u8bin";
            gt_fn_ = "/data2/pyc/data/sift1B/sift_groundtruth.ivecs";
        }

        DistanceType distance_type() const override
        {
            return DistanceType::EUCLIDEAN;
        }
    };

    class WikipediaDataset : public Dataset<float>
    {
    public:
        WikipediaDataset()
        {
            n_samples_M_ = 3;
            n_samples_ = n_samples_M_ * MILLION; // 10M samples
            n_dimensions_ = 768;   
            n_queries_ = 5000;    // 10K queries

            ds_fn_ = "/data2/pyc/data/wikipedia/wikipedia_base.bin";
            qs_fn_ = "/data2/pyc/data/wikipedia/wikipedia_query.bin";
            gt_fn_ = "/data2/pyc/data/msturing30M/msturing30M_gt.ivecs";
        }

        DistanceType distance_type() const override
        {
            return DistanceType::INNER_PRODUCT;
        }
    };

} // namespace ffanns
