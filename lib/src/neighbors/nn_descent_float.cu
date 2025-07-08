#include "nn_descent.cuh"
#include "ffanns/neighbors/nn_descent.hpp"

namespace ffanns::neighbors::nn_descent {

#define FFANNS_INST_NN_DESCENT_BUILD(T, IdxT)                                     \
                                                                                  \
  auto build(raft::resources const& handle,                                       \
             const ffanns::neighbors::nn_descent::index_params& params,             \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)   \
    ->ffanns::neighbors::nn_descent::index<IdxT>                                    \
  {                                                                               \
    return ffanns::neighbors::nn_descent::build<T, IdxT>(handle, params, dataset);  \
  };

FFANNS_INST_NN_DESCENT_BUILD(float, uint32_t);

#undef FFANNS_INST_NN_DESCENT_BUILD

}  // namespace ffanns::neighbors::nn_descent
