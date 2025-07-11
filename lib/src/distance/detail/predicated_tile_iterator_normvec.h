/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

/*! \file
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

This file contains a customized version of PredicatedTileIterator from CUTLASS 2.9.0
(https://github.com/NVIDIA/cutlass/blob/v2.9.0/include/cutlass/epilogue/threadblock/predicated_tile_iterator.h#L75)

Changes:
- added `Layout_` template param
- Only the row index is used to load the data in load_with_byte_offset().
  This way the same normalization data is used across all columns in a row.

*/

#pragma once

#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/threadblock/output_tile_thread_map.h>
#include <cutlass/epilogue/threadblock/predicated_tile_iterator_params.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/transform/pitch_linear_thread_map.h>

////////////////////////////////////////////////////////////////////////////////

namespace ffanns {

////////////////////////////////////////////////////////////////////////////////

namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load and store output tile from global memory in epilogue.
///
/// Satisfies: ReadableTileIterator | PredicatedTileIterator | ForwardTileIterator
///
template <typename ThreadMap_,  ///< Thread map (conept: OutputTileThreadMap)
          typename Element_,    ///< Element data type
          typename Layout_,
          bool ScatterD     = false,  ///< Scatter D operand or not
          bool UseCUDAStore = false>
class PredicatedTileIteratorNormVec {
 public:
  using ThreadMap = ThreadMap_;
  using Shape     = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout         = Layout_;
  using TensorRef      = cutlass::TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index       = typename Layout::Index;
  using LongIndex   = typename Layout::LongIndex;
  using TensorCoord = cutlass::MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads           = ThreadMap::kThreads;
  static int const kIterations        = ThreadMap::Count::kTile;

  static_assert(ThreadMap::Iterations::kRow > 0, "ThreadMap::Iterations::kRow must be > 0");
  static_assert(ThreadMap::Iterations::kGroup > 0, "ThreadMap::Iterations::kGroup must be > 0");
  static_assert(ThreadMap::Iterations::kCluster > 0, "ThreadMap::Iterations::kCluster must be > 0");
  static_assert(ThreadMap::Iterations::kColumn > 0, "ThreadMap::Iterations::kColumn must be > 0");

  /// Fragment object
  using Fragment =
    cutlass::Array<Element,
                   ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow *
                     ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster *
                     ThreadMap::kElementsPerAccess>;

  /// Memory access size
  using AccessType = cutlass::AlignedArray<Element, ThreadMap::kElementsPerAccess>;

  //
  // Parameters struct
  //

  /// Uses a non-template class
  struct Params : cutlass::epilogue::threadblock::PredicatedTileIteratorParams {
    using Base = cutlass::epilogue::threadblock::PredicatedTileIteratorParams;

    CUTLASS_HOST_DEVICE
    Params() {}

    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
      : PredicatedTileIteratorParams(
          layout.stride(0) * int(sizeof(AccessType)) / kElementsPerAccess,
          cutlass::epilogue::threadblock::make_OutputTileThreadMapDesc<ThreadMap>())
    {
    }

    CUTLASS_HOST_DEVICE
    Params(Base const& base) : Base(base) {}
  };

  /// Mask object
  struct Mask {
    static int const kCount = ThreadMap::Iterations::kColumn;

    /// Predicate state
    bool predicates[kCount];

    //
    // Mask
    //
    CUTLASS_HOST_DEVICE
    Mask() { enable(); }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_HOST_DEVICE void clear()
    {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = false;
      }
    }

    ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
    CUTLASS_DEVICE void enable()
    {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = true;
      }
    }
  };

 private:
  //
  // Data members
  //

  /// Parameters structure containing reference and precomputed state.
  cutlass::epilogue::threadblock::PredicatedTileIteratorParams params_;

  /// Byte-level pointer
  uint8_t* byte_pointer_;

  /// cutlass::Array of boolean values to contain steady-state predicates
  Mask mask_;

  /// Extent of the matrix tile in rows
  Index extent_row_;

  /// Extent of the matrix tile in rows
  Index extent_column_;

  /// A thread's starting row position (assuming steady-state predicates have been computed)
  Index thread_start_row_;

  /// A thread's starting column
  Index thread_start_column_;

  /// Internal state counter
  int state_[3];

  /// Scatter indices
  int const* indices_;

  //
  // Static asserts about internal strides
  //

  static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(cutlass::epilogue::threadblock::PredicatedTileIteratorParams::stride) == 8,
                "Expected 64b strides");

 private:
  //
  // Methods
  //

 public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  PredicatedTileIteratorNormVec(
    cutlass::epilogue::threadblock::PredicatedTileIteratorParams const& params,
    Element* pointer,
    TensorCoord extent,
    int thread_idx,
    TensorCoord threadblock_offset = TensorCoord(),
    int const* indices             = nullptr)
    : params_(params), indices_(indices)
  {
    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    extent_row_    = extent.row();
    extent_column_ = extent.column();

    thread_start_row_    = thread_offset.row();
    thread_start_column_ = thread_offset.column();

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
      mask_.predicates[c] =
        ((thread_offset.column() + ThreadMap::Delta::kColumn * c) < extent.column());
    }

    // Null pointer performs no accesses
    if (!pointer) { mask_.clear(); }

    if (ScatterD && !indices) { mask_.clear(); }

    // Initialize pointer
    byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                    LongIndex(thread_offset.row()) * LongIndex(params_.stride);

    if (ScatterD) {
      byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                      LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;
    }

    // Initialize internal state counter
    state_[0] = state_[1] = state_[2] = 0;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset)
  {
    byte_pointer_ += pointer_offset * cutlass::sizeof_bits<Element>::value / 8;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment& frag, int64_t byte_offset) const
  {
    uint8_t* byte_pointer = byte_pointer_;
    AccessType* frag_ptr  = reinterpret_cast<AccessType*>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int frag_row_idx =
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup +
                           cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          AccessType* memory_pointer = reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

          if (ScatterD && row_guard) {
            assert(indices_);

            memory_pointer = reinterpret_cast<AccessType*>(
              byte_pointer + byte_offset +
              LongIndex(indices_[row_offset + thread_start_row_]) * LongIndex(params_.stride));
          }

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            bool guard = row_guard && mask_.predicates[column];
            if (column == 0) {
              cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                (void*)&memory_pointer[0],
                guard);
            } else {
              frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column] =
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn];
            }
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            if (!ScatterD) { byte_pointer += params_.increment_row; }
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) { byte_pointer += params_.increment_group; }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment& frag) const { load_with_byte_offset(frag, 0); }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) const
  {
    uint8_t* byte_pointer      = byte_pointer_;
    AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int frag_row_idx =
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup +
                           cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          AccessType* memory_pointer = reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

          if (ScatterD && row_guard) {
            assert(indices_);

            memory_pointer = reinterpret_cast<AccessType*>(
              byte_pointer + byte_offset +
              LongIndex(indices_[row_offset + thread_start_row_]) * LongIndex(params_.stride));
          }

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            bool guard = row_guard && mask_.predicates[column];

            if (UseCUDAStore) {
              if (guard) {
                memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess] =
                  frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column];
              }
            } else {
              cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                (void*)&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess],
                guard);
            }
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            if (!ScatterD) { byte_pointer += params_.increment_row; }
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) { byte_pointer += params_.increment_group; }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const& frag) const { store_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void downsample_load_with_byte_offset(Fragment& frag,
                                        int64_t byte_offset,
                                        int convolution_P,
                                        int convolution_Q,
                                        int add_P,
                                        int add_Q,
                                        int problem_N) const
  {
    uint8_t* byte_pointer = byte_pointer_;
    AccessType* frag_ptr  = reinterpret_cast<AccessType*>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int frag_row_idx =
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup +
                           cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          int output_row = row_offset + thread_start_row_;
          int output_N   = output_row / (convolution_P * convolution_Q);
          int output_PQ  = output_row % (convolution_P * convolution_Q);
          int output_P   = output_PQ / convolution_Q;
          int output_Q   = output_PQ % convolution_Q;

          int input_row = output_N * 2 * convolution_P * 2 * convolution_Q +
                          (2 * output_P + add_P) * 2 * convolution_Q + 2 * output_Q + add_Q;

          int64_t byte_offset = (input_row - output_row) * problem_N * sizeof(float);

          AccessType* memory_pointer = reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            bool guard = row_guard && mask_.predicates[column];

            cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
              frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
              (void*)&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess],
              guard);
          }

          if (row + 1 < ThreadMap::Iterations::kRow) { byte_pointer += params_.increment_row; }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) { byte_pointer += params_.increment_group; }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void upsample_load_with_byte_offset(Fragment& frag,
                                      int64_t byte_offset,
                                      int convolution_P,
                                      int convolution_Q,
                                      int add_P,
                                      int add_Q,
                                      int problem_N) const
  {
    uint8_t* byte_pointer = byte_pointer_;
    AccessType* frag_ptr  = reinterpret_cast<AccessType*>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int frag_row_idx =
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup +
                           cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          int output_row = row_offset + thread_start_row_;
          int output_N   = output_row / (convolution_P * convolution_Q);
          int output_PQ  = output_row % (convolution_P * convolution_Q);
          int output_P   = output_PQ / convolution_Q;
          int output_Q   = output_PQ % convolution_Q;
          int row_add_P  = add_P;
          int row_add_Q  = add_Q;
          if (output_P > convolution_P - 2) row_add_P = 0;
          if (output_Q > convolution_Q - 2) row_add_Q = 0;

          int input_row = output_N * (convolution_P / 2) * (convolution_Q / 2) +
                          ((output_P + row_add_P) / 2) * (convolution_Q / 2) +
                          (output_Q + row_add_Q) / 2;

          int64_t byte_offset = (input_row - output_row) * problem_N * sizeof(float);

          AccessType* memory_pointer = reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            bool guard = row_guard && mask_.predicates[column];

            cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
              frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
              (void*)&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess],
              guard);
          }

          if (row + 1 < ThreadMap::Iterations::kRow) { byte_pointer += params_.increment_row; }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) { byte_pointer += params_.increment_group; }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  CUTLASS_DEVICE
  cutlass::MatrixCoord thread_start() const
  {
    return MatrixCoord(thread_start_row_, thread_start_column_);
  }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_row() const { return thread_start_row_; }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_column() const { return thread_start_column_; }

  /// Extent of the matrix in rows
  CUTLASS_DEVICE
  Index extent_row() const { return extent_row_; }

  /// Extent of the matrix in columns
  CUTLASS_DEVICE
  Index extent_column() const { return extent_column_; }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorNormVec& operator++()
  {
    ++state_[0];

    if (!ScatterD) { byte_pointer_ += params_.advance_row; }

    thread_start_row_ += ThreadMap::Shape::kRow;

    if (state_[0] == ThreadMap::Count::kRow) {
      state_[0] = 0;
      ++state_[1];
      byte_pointer_ += params_.advance_group;

      thread_start_row_ +=
        (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {
        state_[1] = 0;
        ++state_[2];
        byte_pointer_ += params_.advance_cluster;

        thread_start_row_ += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
                             ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;
          byte_pointer_ += params_.advance_tile;
        }
      }
    }

    return *this;
  }

  ///< Efficiently disables all accesses guarded by mask
  CUTLASS_DEVICE void clear_mask() { mask_.clear(); }

  ///< Efficiently enables all accesses guarded by mask
  CUTLASS_DEVICE void enable_mask() { mask_.enable(); }

  ///< Sets the mask
  CUTLASS_DEVICE void get_mask(Mask& mask) const { mask = mask_; }

  ///< Sets the mask
  CUTLASS_DEVICE void set_mask(Mask const& mask) { mask_ = mask; }
};

///////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace ffanns

////////////////////////////////////////////////////////////////////////////////
