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

/* This file has two responsibilities:
 *
 * 1. Dispatch to the correct implementation of a kernel based on the
 *    architecture of the device on which the kernel will be launched. For
 *    instance, the cosine distance has a CUTLASS-based implementation that can
 *    be used on SM80+ and the normal implementation that is used on older
 *    architectures.
 *
 * 2. Provide concise function templates that can be instantiated in
 *    src/distance/detail/pairwise_matrix/. Previously,
 *    ffanns::distance::detail::distance was instantiated. The function
 *    necessarily required a large set of include files, which slowed down the
 *    build. The ffanns::distance::detail::pairwise_matrix_arch_dispatch functions
 *    do not require as large an include files set, which speeds up the build.
 */

#include "../distance_ops/cutlass.cuh"           // ops::has_cutlass_op
#include "../pairwise_matrix/params.cuh"         // pairwise_matrix_params
#include <raft/util/arch.cuh>                    // raft::util::arch::SM_*

// NOTE: to minimize compile times, we do not include dispatch_sm80.cuh.
// Including dispatch_sm80.cuh can slow down compile times (due to CUTLASS).
// Therefore, it is the including file's responsibility to include the correct
// dispatch_smXX.cuh headers, as is done in ffanns/distance/detail/distance.cuh
// and src/distance/detail/pairwise_matrix/dispatch_*.cu.

namespace ffanns::distance::detail {

// This forward-declaration ensures that we do not need to include
// dispatch_sm80.cuh if we are not calling it in practice. This makes compiling
// all the non-CUTLASS based distance instantiations faster. For CUTLASS-based
// distances, dispatch_sm80.cuh has to be included by the file including this
// file.
template <typename OpT,
          typename IdxT,
          typename DataT,
          typename OutT,
          typename FinOpT,
          typename SM_compat_t>
void pairwise_matrix_sm80_dispatch(OpT,
                                   pairwise_matrix_params<IdxT, DataT, OutT, FinOpT>,
                                   SM_compat_t,
                                   cudaStream_t);

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
                              bool is_row_major)
{
  // Create kernel parameter struct. Flip x and y if column major.
  IdxT ldx    = is_row_major ? k : m;
  IdxT ldy    = is_row_major ? k : n;
  IdxT ld_out = is_row_major ? n : m;

  pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params{
    m, n, k, ldx, ldy, ld_out, x, y, x_norm, y_norm, out, fin_op, is_row_major};

  if (!params.is_row_major) { params.flip_x_and_y(); }

  // Dispatch rule:
  // - execute CUTLASS-based kernel on SM_80 and above
  // - execute normal kernel below SM_80
  namespace arch = raft::util::arch;

  constexpr bool cutlass_op_unavailable = !ops::has_cutlass_op<OpT>();

  if constexpr (cutlass_op_unavailable) {
    // Always execute legacy kernels when no cutlass op is available
    // auto any_range = arch::SM_range(arch::SM_min(), arch::SM_future());
    // pairwise_matrix_sm60_dispatch(distance_op, params, any_range, stream);
    // TODO: not implemented
  } else {
    auto cutlass_range = arch::SM_range(arch::SM_80(), arch::SM_future());
    // TODO: the cutlass doesn't support the odd `k` on half DataT.
    bool if_unsupported_on_half = (sizeof(DataT) == 2) && ((k % 2) != 0);

    if (if_unsupported_on_half) {
        // TODO: not implemented
    //   auto any_range = arch::SM_range(arch::SM_min(), arch::SM_future());
    //   pairwise_matrix_sm60_dispatch(distance_op, params, any_range, stream);
    }  
    else {
      // TODO
      pairwise_matrix_sm80_dispatch(distance_op, params, cutlass_range, stream);
    }
  }
}

};  // namespace ffanns::distance::detail
