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

#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/variant_jacobian.h>

#include "matrix_expression_test.h"

#include <gtest/gtest.h>

#include <optional>

using namespace mochi;

class VariantJacobianTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Create a dense matrix for the variant Jacobian
    int numRows = 3 * kBlockSize;
    int numColsDense = 6;
    _denseMatrix.Reset(numRows, numColsDense);
    _denseMatrix.SetRandom(123, -10_r, 10_r);

    // Create a block diagonal matrix for the variant Jacobian
    auto blockDiagMatrix = VariantJacobian::MatrixBlockDiag{numRows, kBlockSize};
    blockDiagMatrix.SetRandom(456, -10_r, 10_r);

    // Create the full dense copy of the block diagonal matrix
    _blockDiagMatrixFull.Reset(numRows, numRows);
    _blockDiagMatrixFull.SetZero();
    for (int i = 0; i < numRows; i += kBlockSize) {
      for (int j = 0; j < kBlockSize; j++) {
        for (int k = 0; k < kBlockSize; k++) {
          _blockDiagMatrixFull(i + j, i + k) = blockDiagMatrix(i + j, k);
        }
      }
    }

    // Create the variant Jacobians
    auto denseVar = VariantJacobian(false, _denseMatrix.Rows(), _denseMatrix.Cols());
    denseVar.GetDenseView() = _denseMatrix;
    _denseVar = denseVar;

    auto blockVar = VariantJacobian(true, blockDiagMatrix.Rows(), kBlockSize);
    blockVar.GetBlockDiagView() = blockDiagMatrix;
    _blockVar = blockVar;

    // Create a block sparse matrix for testing
    EXPECT_EQ(_blockDiagMatrixFull.Rows() % kBlockSize, 0);
    int nBlockCols = _blockDiagMatrixFull.Rows() / kBlockSize;
    DynamicArray<int> ptr = {0, 1, 2, 4};
    MOCHI_ASSERT(ptr.size() == nBlockCols + 1);
    int nnzs = kBlockSize * kBlockSize * ptr.back();
    DynamicArray<int> idx = {0, 1, 0, 2};
    MOCHI_ASSERT(kBlockSize * kBlockSize * idx.size() == nnzs);
    DynamicArray<real> val(nnzs);
    auto generator = RandomGenerator(42);
    SetRandom(generator, -1_r, 1_r, MakeSpan(val));
    _inputBlockSparse.Reset(nBlockCols, ptr, idx, val);

    // Create the dense copy of the block sparse matrix
    _inputDense.Reset(ToMatrix(_inputBlockSparse));
  }

  static constexpr int kBlockSize = 3;
  static constexpr real kFrontTol = 1e-6_r;
  static constexpr real kFrontAndBackTol = 1e-4_r;

  RowMatrix<real> _denseMatrix;
  RowMatrix<real> _blockDiagMatrixFull;
  std::optional<VariantJacobian> _denseVar;
  std::optional<VariantJacobian> _blockVar;
  BlockSparseMatrix<real, kBlockSize> _inputBlockSparse;
  RowMatrix<real> _inputDense;
};

TEST_F(VariantJacobianTest, RowsAndCols) {
  EXPECT_EQ(_denseVar->Rows(), _denseMatrix.Rows());
  EXPECT_EQ(_denseVar->Cols(), _denseMatrix.Cols());
  EXPECT_EQ(_blockVar->Rows(), _blockDiagMatrixFull.Rows());
  EXPECT_EQ(_blockVar->Cols(), _blockDiagMatrixFull.Cols());
}

TEST_F(VariantJacobianTest, TransposeMultiply_Dense) {
  Matrix<real> res1(_denseMatrix.Cols(), _inputDense.Cols());
  _denseVar->TransposeMultiply(AsConstView(_inputDense), AsView(res1));
  Matrix<real> res2 = _denseMatrix.Transpose() * _inputDense;
  EXPECT_TRUE(test::IsNear(res1, res2, kFrontTol));
  _denseVar->TransposeMultiply<true>(AsConstView(_inputDense), AsView(res1));
  EXPECT_TRUE(test::IsNear(res1, Matrix<real>(2_r * res2), 2 * kFrontTol));
}

TEST_F(VariantJacobianTest, TransposeMultiply_BlockDiag) {
  Matrix<real> res1(_blockDiagMatrixFull.Cols(), _inputDense.Cols());
  _blockVar->TransposeMultiply(AsConstView(_inputDense), AsView(res1));
  Matrix<real> res2 = _blockDiagMatrixFull.Transpose() * _inputDense;
  EXPECT_TRUE(test::IsNear(res1, res2, kFrontTol));
  _blockVar->TransposeMultiply<true>(AsConstView(_inputDense), AsView(res1));
  EXPECT_TRUE(test::IsNear(res1, Matrix<real>(2_r * res2), 2 * kFrontTol));
}

TEST_F(VariantJacobianTest, FrontTransposeAndBackMultiply_Dense) {
  // Dense Output
  RowMatrix<real> tmp(_inputDense.Rows(), _denseMatrix.Cols());
  Matrix<real> res1(_denseMatrix.Cols(), _denseMatrix.Cols());
  _denseVar->FrontTransposeAndBackMultiply(_inputBlockSparse, tmp, res1);
  Matrix<real> res2 = _denseMatrix.Transpose() * _inputDense * _denseMatrix;
  EXPECT_TRUE(test::IsNear(res1, res2, kFrontAndBackTol));
  _denseVar->FrontTransposeAndBackMultiply<true>(_inputBlockSparse, tmp, res1);
  EXPECT_TRUE(test::IsNear(res1, Matrix<real>(2_r * res2), 2 * kFrontAndBackTol));

  // NOTE: BlockSparse output not supported for a VariantJacobian with dense storage.
}

TEST_F(VariantJacobianTest, FrontTransposeAndBackMultiply_BlockSparse) {
  // BlockSparse Output
  BlockSparseMatrix<real, kBlockSize> res1 = _inputBlockSparse; // Copy sparsity
  _blockVar->FrontTransposeAndBackMultiply(_inputBlockSparse, res1);
  Matrix<real> res2 = _blockDiagMatrixFull.Transpose() * _inputDense * _blockDiagMatrixFull;
  EXPECT_TRUE(test::IsNear(ToMatrix(res1), res2, kFrontAndBackTol));
  _blockVar->FrontTransposeAndBackMultiply<true>(_inputBlockSparse, res1);
  EXPECT_TRUE(test::IsNear(ToMatrix(res1), Matrix<real>(2_r * res2), 2 * kFrontAndBackTol));

  // NOTE: Dense output not supported for a VariantJacobian with block diagonal storage.
}
