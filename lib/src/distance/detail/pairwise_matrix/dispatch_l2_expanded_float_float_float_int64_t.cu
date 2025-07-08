#include "../distance_ops/all_ops.cuh"  // ops::*
#include "dispatch-inl.cuh"             // dispatch
// #include "dispatch_sm60.cuh"
#include "dispatch_sm80.cuh"
#include <raft/core/operators.hpp>  // raft::identity_op
#define instantiate_raft_distance_detail_pairwise_matrix_dispatch(                     \
  OpT, DataT, AccT, OutT, FinOpT, IdxT)                                                \
  template void ffanns::distance::detail::                                             \
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

instantiate_raft_distance_detail_pairwise_matrix_dispatch(
  ffanns::distance::detail::ops::l2_exp_distance_op, float, float, float, raft::identity_op, int64_t);

#undef instantiate_raft_distance_detail_pairwise_matrix_dispatch
