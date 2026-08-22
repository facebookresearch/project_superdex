/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <mochi_core/linear_algebra/any_matrix.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/overload_visitor.h>

#include <tuple>
#include <variant>

namespace mochi {

// Variant implementation of Jacobian matrices, considering two cases:
// - As a 3x3 block diagonal matrix, and only the diagonal blocks are stored. This case is used for
//   soft FOMs.
// - As a dense matrix. This case is used for soft ROMs.
// In both cases, the Jacobian is stored as row-major to accelerate the dresidual projection (both
// FOM and ROM) and the composition of the ROM Jacobian with the skinning Jacobian (ROMs only).
class VariantJacobian {
 public:
  using MatrixBlockDiag = RowMatrix<real, krylov::kDynamic, 3>;
  using MatrixBlockDiagView = RowMatrixView<real, krylov::kDynamic, 3>;
  using MatrixBlockDiagConstView = RowMatrixView<real const, krylov::kDynamic, 3>;

  VariantJacobian(bool isBlockDiag, int numRows, int numCols) {
    if (isBlockDiag) {
      MOCHI_ASSERT(numCols == 3, "Block diagonal Jacobian must have 3 columns");
      _value = MatrixBlockDiag::Zero(numRows, numCols);
    } else {
      _value = RowMatrix<real>::Zero(numRows, numCols);
    }
  }

  int Rows() const {
    return std::visit([](auto const& mat) { return mat.Rows(); }, _value);
  }

  int Cols() const {
    return std::visit(
        OverloadVisitor{
            [](RowMatrix<real> const& mat) { return mat.Cols(); },
            [](MatrixBlockDiag const& mat) { return mat.Rows(); }},
        _value);
  }

  bool IsBlockDiagonal() const {
    return std::holds_alternative<MatrixBlockDiag>(_value);
  }

  MatrixBlockDiagView GetBlockDiagView() {
    MOCHI_ASSERT(std::holds_alternative<MatrixBlockDiag>(_value), "Jacobian must be block-diag");
    return AsView(std::get<MatrixBlockDiag>(_value));
  }

  MatrixBlockDiagConstView GetBlockDiagView() const {
    MOCHI_ASSERT(std::holds_alternative<MatrixBlockDiag>(_value), "Jacobian must be block-diag");
    return AsConstView(std::get<MatrixBlockDiag>(_value));
  }

  RowMatrixView<real> GetDenseView() {
    MOCHI_ASSERT(std::holds_alternative<RowMatrix<real>>(_value), "Jacobian must be dense");
    return AsView(std::get<RowMatrix<real>>(_value));
  }

  RowMatrixView<real const> GetDenseView() const {
    MOCHI_ASSERT(std::holds_alternative<RowMatrix<real>>(_value), "Jacobian must be dense");
    return AsConstView(std::get<RowMatrix<real>>(_value));
  }

  // output = Transpose(_value) * input, where input and output are dense matrix views.
  // If kAdd = true, it computes output += Transpose(_value) * input.
  template <
      bool kAdd = false,
      int kCols,
      krylov::Direction kDirIn,
      krylov::Direction kDirOut,
      int kLeadDimIn,
      int kLeadDimOut>
  void TransposeMultiply(
      MatrixView<real const, krylov::kDynamic, kCols, kDirIn, kLeadDimIn> input,
      MatrixView<real, krylov::kDynamic, kCols, kDirOut, kLeadDimOut> output) const {
    MOCHI_PROFILE_SCOPE();
    MOCHI_ASSERT(input.Rows() == Rows(), "Sizes don't match");
    MOCHI_ASSERT(output.Rows() == Cols(), "Sizes don't match");
    MOCHI_ASSERT(output.Cols() == input.Cols(), "Sizes don't match");

    auto transposeMultiplyVisitor = OverloadVisitor{
        [&input, &output](RowMatrix<real> const& J) {
          // ROM codepath. Parallelize the dense matrix-matrix product across the contraction
          // direction to exploit that the number of FOM DoFs is much greater than the number of ROM
          // DoFs.
          ParallelMatMatAlongK<kAdd>(J.Transpose(), input, output);
        },
        [&input, &output](MatrixBlockDiag const& J) {
          int numBlockRows = J.Rows() / 3;
          int constexpr kMinBlockRowsPerTask = 128; // First guess. Not tuned.
          ParallelForN("TransposeMultiply_Range", numBlockRows, kMinBlockRowsPerTask, [&](int br) {
            int r = br * 3;
            if constexpr (kAdd) {
              output.template MiddleRows<3>(r, 3) +=
                  J.template MiddleRows<3>(r, 3).Transpose() * input.template MiddleRows<3>(r, 3);
            } else {
              output.template MiddleRows<3>(r, 3) =
                  J.template MiddleRows<3>(r, 3).Transpose() * input.template MiddleRows<3>(r, 3);
            }
          });
        }};
    std::visit(transposeMultiplyVisitor, _value);
  }

  // output = Transpose(_value) * input * _value, where input is a block sparse view and output is a
  // dense view. If kAdd = true, it computes output += Transpose(_value) * input * _value. Notes:
  // - input_J is a temporary to store input * _value.
  // - This overload is only supported for dense Jacobian.
  // - For performance reasons, input_J is enforced to be row-major and output to be col-major.
  template <bool kAdd = false>
  void FrontTransposeAndBackMultiply(
      BlockSparseMatrixView<real const, 3> input,
      RowMatrixView<real> input_J,
      MatrixView<real> output) const {
    MOCHI_PROFILE_SCOPE();
    MOCHI_ASSERT(input.Rows() == input.Cols() && input.Rows() == Rows(), "Inconsistent sizes.");
    MOCHI_ASSERT(input_J.Rows() == input.Rows() && input_J.Cols() == Cols(), "Inconsistent sizes.");
    MOCHI_ASSERT(output.Rows() == Cols() && output.Cols() == Cols(), "Inconsistent sizes.");
    MOCHI_ASSERT(
        std::holds_alternative<RowMatrix<real>>(_value),
        "This overload can only be used for dense jacobians");
    auto const& J = std::get<RowMatrix<real>>(_value);
    input_J = input * J; // RowMatrix = BlockSparseMatrix x RowMatrix
    TransposeMultiply<kAdd>(AsConstView(input_J), output);
  }

  // output = Transpose(_value) * input * _value, where input and output are block sparse matrix
  // views. If kAdd = true, it computes output += Transpose(_value) * input * _value. Notes:
  // - output must have the same size and sparsity pattern as input.
  // - This overload is only supported for block diagonal Jacobian.
  template <bool kAdd = false>
  void FrontTransposeAndBackMultiply(
      BlockSparseMatrixView<real const, 3> input,
      BlockSparseMatrixView<real, 3> output) const {
    MOCHI_PROFILE_SCOPE();
    MOCHI_ASSERT(input.Rows() == input.Cols() && input.Rows() == Rows(), "Inconsistent sizes.");
    MOCHI_ASSERT(output.Rows() == Cols() && output.Cols() == Cols(), "Inconsistent sizes.");
    MOCHI_ASSERT(
        std::holds_alternative<MatrixBlockDiag>(_value),
        "This overload can only be used for block diagonal jacobians");
    if (input.NumNonZeros() == 0)
      MOCHI_UNLIKELY {
        return;
      }
    auto const& J = std::get<MatrixBlockDiag>(_value);
    constexpr int kMinFlopsPerTask = 500000; // 50 μs @ 10 GFLOPs
    auto const numFlopsPerBlockRow =
        Max(1, (2 * (2 * 3 - 1) * input.NumNonZeros()) / input.BlockRows());
    int const minBlockRowsPerTask = Max(1, kMinFlopsPerTask / numFlopsPerBlockRow);
    ParallelForRange(
        "FrontTransposeAndBackMultiply",
        0,
        input.BlockRows(),
        minBlockRowsPerTask,
        INT_MAX,
        [&](int brBegin, int brEnd) {
          // Use stack memory if there are <= 100 non-zero blocks per row.
          MOCHI_FILO_STACK_ALLOCATOR(tmpAlloc, 3 * 3 * 100 * sizeof(real));
          DynamicArray<real> buffer(3 * input.MaxNnzPerRow(), &tmpAlloc);
          for (int br = brBegin; br < brEnd; ++br) {
            MOCHI_ASSERT_VERBOSE(output.Values(br).NumBlocks() == input.Values(br).NumBlocks());
            auto indices = input.Indices(br);
            auto inputBlockRow = input.Values(br);
            auto numBlockCols = inputBlockRow.NumBlocks();
            BlockRowView<real, 3> bufferView{
                buffer.data(), /*leadDim*/ 3 * numBlockCols, numBlockCols};
            for (int bc = 0; bc < numBlockCols; bc++) {
              // Row-Major = Row-Major x Row-Major. May be computed as the transposed product.
              bufferView[bc] = inputBlockRow[bc] * J.template MiddleRows<3>(3 * indices[bc], 3);
            }
            // Row-Major (+)= Col-Major x Row-Major. May be computed as the transposed product.
            if constexpr (kAdd) {
              output.Values(br).Underlying() +=
                  J.template MiddleRows<3>(3 * br, 3).Transpose() * bufferView.Underlying();
            } else {
              output.Values(br).Underlying() =
                  J.template MiddleRows<3>(3 * br, 3).Transpose() * bufferView.Underlying();
            }
          }
        });
  }

 protected:
  std::variant<RowMatrix<real>, MatrixBlockDiag> _value;
};

} // namespace mochi
