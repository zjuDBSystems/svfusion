/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "../distance_ops/all_ops.cuh"    // ops::*
#include "../distance_ops/cutlass.cuh"    // ops::has_cutlass_op
// #include "../kernels/rbf_fin_op.cuh"      // rbf_fin_op
#include "../pairwise_matrix/params.cuh"  // pairwise_matrix_params
#include <raft/core/operators.hpp>        // raft::identity_op
#include <raft/util/raft_explicit.hpp>    // RAFT_EXPLICIT

namespace ffanns::distance::detail {

template <typename OpT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void pairwise_matrix_dispatch(OpT distance_op,
                              IdxT m,
                              IdxT n,
                              IdxT k,
                              const DataT* x,
                              const DataT* y,
                              const OutT* x_norm,
                              const OutT* y_norm,
                              OutT* out,
                              FinOpT fin_op,
                              cudaStream_t stream,
                              bool is_row_major) RAFT_EXPLICIT;

};  // namespace ffanns::distance::detail

#define instantiate_ffanns_distance_detail_pairwise_matrix_dispatch(                     \
  OpT, DataT, AccT, OutT, FinOpT, IdxT)                                                \
  extern template void ffanns::distance::detail::                                        \
    pairwise_matrix_dispatch<OpT<DataT, AccT, IdxT>, DataT, AccT, OutT, FinOpT, IdxT>( \
      OpT<DataT, AccT, IdxT> distance_op,                                              \
      IdxT m,                                                                          \
      IdxT n,                                                                          \
      IdxT k,                                                                          \
      const DataT* x,                                                                  \
      const DataT* y,                                                                  \
      const OutT* x_norm,                                                              \
      const OutT* y_norm,                                                              \
      OutT* out,                                                                       \
      FinOpT fin_op,                                                                   \
      cudaStream_t stream,                                                             \
      bool is_row_major)

#define instantiate_ffanns_distance_detail_pairwise_matrix_dispatch_by_algo_default(OpT, IdxT) \
  instantiate_ffanns_distance_detail_pairwise_matrix_dispatch(                                 \
    OpT, float, float, float, raft::identity_op, IdxT);                                      

#define instantiate_ffanns_distance_detail_pairwise_matrix_dispatch_by_algo(OpT, IdxT, FinOpT) \
  instantiate_ffanns_distance_detail_pairwise_matrix_dispatch(                                 \
    OpT, float, float, float, FinOpT<float>, IdxT);        
    
// The following two instances are used in the RBF kernel object. Note the use of int64_t for the
// index type.
instantiate_ffanns_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  ffanns::distance::detail::ops::l2_exp_distance_op, int);
instantiate_ffanns_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  ffanns::distance::detail::ops::l2_exp_distance_op, int64_t);
    
    
#undef instantiate_ffanns_distance_detail_pairwise_matrix_dispatch_by_algo
#undef instantiate_ffanns_distance_detail_pairwise_matrix_dispatch
