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

#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/block_sparse_matrix_utils.h>
#include <mochi_core/linear_algebra/matrix_fwd.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/memory/monotonic_allocator.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

using namespace mochi;

/**************************************************************************************
 * BlockSparseMatrix Examples
 */

struct TestCase_Block3_6x6 {
  // |   123|
  // |   456|
  // |   789|
  // |135   |
  // |246   |
  // |357   |

  static constexpr int kBlockSize = 3;
  static constexpr int kNumBlockCols = 2;
  static constexpr int kBlockPointers[] = {0, 1, 2};
  static constexpr int kBlockIndices[] = {1, 0};
  static constexpr real kValues[] =
      {1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, 9_r, 1_r, 3_r, 5_r, 2_r, 4_r, 6_r, 3_r, 5_r, 7_r};
  static constexpr real kValuesDense[6][6] = {
      {0_r, 0_r, 0_r, 1_r, 2_r, 3_r},
      {0_r, 0_r, 0_r, 4_r, 5_r, 6_r},
      {0_r, 0_r, 0_r, 7_r, 8_r, 9_r},
      {1_r, 3_r, 5_r, 0_r, 0_r, 0_r},
      {2_r, 4_r, 6_r, 0_r, 0_r, 0_r},
      {3_r, 5_r, 7_r, 0_r, 0_r, 0_r}};
};

// Check that a BlockSparseMatrix has the above data
template <class TC, class Mat>
static void CheckMat(Mat const& mat) {
  EXPECT_EQ(isize(TC::kValues) / (TC::kBlockSize * TC::kBlockSize), mat.NumNonZeroBlocks());
  EXPECT_EQ(isize(TC::kValues), mat.NumNonZeros());
  EXPECT_EQ((isize(TC::kBlockPointers) - 1) * TC::kBlockSize, mat.Rows());
  EXPECT_EQ(TC::kNumBlockCols * TC::kBlockSize, mat.Cols());
  EXPECT_EQ(mat.Rows(), mat.CERows().iVal());
  EXPECT_EQ(mat.Cols(), mat.CECols().iVal());
  EXPECT_EQ(TC::kNumBlockCols, mat.BlockCols());
  EXPECT_EQ(isize(TC::kBlockPointers) - 1, mat.BlockRows());
  EXPECT_SPAN_EQ(MakeSpan(TC::kBlockPointers), mat.Pointers());
  EXPECT_SPAN_EQ(MakeSpan(TC::kBlockIndices), mat.Indices());
  EXPECT_SPAN_EQ(MakeSpan(TC::kValues), mat.Values());
  for (int r = 0; r < mat.Rows(); ++r) {
    for (int c = 0; c < mat.Cols(); ++c) {
      EXPECT_NEAR_EQ(TC::kValuesDense[r][c], mat(r, c));
    }
  }
}

struct TestCase_Block3_3x6 {
  // |   123|
  // |   456|
  // |   789|

  static constexpr int kBlockSize = 3;
  static constexpr int kNumBlockCols = 2;
  static constexpr int kBlockPointers[] = {0, 1};
  static constexpr int kBlockIndices[] = {1};
  static constexpr real kValues[] = {1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, 9_r};
  static constexpr real kValuesDense[3][6] = {
      {0_r, 0_r, 0_r, 1_r, 2_r, 3_r},
      {0_r, 0_r, 0_r, 4_r, 5_r, 6_r},
      {0_r, 0_r, 0_r, 7_r, 8_r, 9_r}};
};

struct TestCase_Block3_6x3 {
  // |   |
  // |   |
  // |   |
  // |123|
  // |456|
  // |789|

  static constexpr int kBlockSize = 3;
  static constexpr int kNumBlockCols = 1;
  static constexpr int kBlockPointers[] = {0, 0, 1};
  static constexpr int kBlockIndices[] = {0};
  static constexpr real kValues[] = {1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, 9_r};
  static constexpr real kValuesDense[6][3] = {
      {0_r, 0_r, 0_r},
      {0_r, 0_r, 0_r},
      {0_r, 0_r, 0_r},
      {1_r, 2_r, 3_r},
      {4_r, 5_r, 6_r},
      {7_r, 8_r, 9_r}};
};

struct TestCase_Block1_3x3 {
  // |1 2|
  // | 3 |
  // |4 5|

  static constexpr int kBlockSize = 1;
  static constexpr int kNumBlockCols = 3;
  static constexpr int kBlockPointers[] = {0, 2, 3, 5};
  static constexpr int kBlockIndices[] = {0, 2, 1, 0, 2};
  static constexpr real kValues[] = {1_r, 2_r, 3_r, 4_r, 5_r};
  static constexpr real kValuesDense[3][3] = {{1_r, 0_r, 2_r}, {0_r, 3_r, 0_r}, {4_r, 0_r, 5_r}};
};

/**************************************************************************************
 * Helper Functions
 */

// Return a new BlockSparseMatrix owning a copy of the data
template <class TC>
static auto MakeMat() {
  return BlockSparseMatrix<real, TC::kBlockSize>{
      TC::kNumBlockCols,
      DynamicArray<int>(std::begin(TC::kBlockPointers), std::end(TC::kBlockPointers)),
      DynamicArray<int>(std::begin(TC::kBlockIndices), std::end(TC::kBlockIndices)),
      DynamicArray<real>(std::begin(TC::kValues), std::end(TC::kValues))};
}

// Return a new BlockSparseMatrixView pointing to the data
template <class TC>
static auto MakeMatConstView() {
  return BlockSparseMatrixView<real const, TC::kBlockSize>{
      TC::kNumBlockCols,
      MakeSpan(TC::kBlockPointers),
      MakeSpan(TC::kBlockIndices),
      MakeSpan(TC::kValues)};
}

template <class Mat>
static void ExpectEmpty(Mat const& mat) {
  EXPECT_TRUE(mat.empty());
  EXPECT_FALSE(mat); // operator bool
  EXPECT_EQ(0, mat.NumNonZeroBlocks());
  EXPECT_EQ(0, mat.NumNonZeros());
  EXPECT_EQ(0, mat.Rows());
  EXPECT_EQ(0, mat.Cols());
  EXPECT_EQ(0, mat.CERows().iVal());
  EXPECT_EQ(0, mat.CECols().iVal());
  EXPECT_EQ(0, mat.BlockCols());
  EXPECT_EQ(0, mat.BlockRows());
  EXPECT_EQ(0, isize(mat.Pointers()));
  EXPECT_EQ(0, isize(mat.Indices()));
  EXPECT_EQ(0, isize(mat.Values()));
}

template <class MatA, class MatB>
static bool HasSameAddresses(MatA const& a, MatB const& b) {
  return (a.Pointers().data() == b.Pointers().data()) &&
      (a.Indices().data() == b.Indices().data()) && (a.Values().data() == b.Values().data());
}

template <class TC, class Mat>
static void CheckPerBlockRowAccessors(Mat const& mat) {
  constexpr int kBS = TC::kBlockSize;
  for (int br = 0; br < mat.BlockRows(); ++br) {
    auto const indices = mat.Indices(br);
    int const nnzBlocks = TC::kBlockPointers[br + 1] - TC::kBlockPointers[br];
    EXPECT_SPAN_EQ(Span(TC::kBlockIndices + TC::kBlockPointers[br], nnzBlocks), indices);
    EXPECT_EQ(mat.Indices().data() + TC::kBlockPointers[br], indices.data()); // Same address
    auto const values = mat.Values(br);
    EXPECT_EQ(nnzBlocks, values.NumBlocks());
    EXPECT_EQ(mat.Values().data() + kBS * kBS * TC::kBlockPointers[br], values.data());
  }
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    CRIdx kBlockSize,
    krylov::Direction kInputMajorDir,
    krylov::Direction kOutputMajorDir>
static void TestBlockSparseMatrixVectorProduct() {
  using MatIn = Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, kInputMajorDir>;
  using MatOut = Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, kOutputMajorDir>;
  using MatInView = MatrixView<Scalar, krylov::kDynamic, krylov::kDynamic, kInputMajorDir>;
  using MatOutView = MatrixView<Scalar, krylov::kDynamic, krylov::kDynamic, kOutputMajorDir>;

  // Block sparse matrix for testing.
  CRIdx const numBlockRows = 11;
  CRIdx const numBlockCols = numBlockRows + 5; // Rectangular to increase test coverage
  DynamicArray<Ptr> rowPtr;
  DynamicArray<CRIdx> colIdx;
  rowPtr.reserve(numBlockRows + 1);
  colIdx.reserve(numBlockRows * (numBlockRows + 1) / 2);
  rowPtr.push_back(0);
  CRIdx maxNumNonZeroBlocksPerRow = 0;
  for (CRIdx ii = 0; ii < numBlockRows; ++ii) {
    CRIdx const numNonZeroBlocksInRow = ii; // Different in each row to increase test coverage
    maxNumNonZeroBlocksPerRow = std::max(maxNumNonZeroBlocksPerRow, numNonZeroBlocksInRow);
    std::set<CRIdx> tmpColIdx;
    for (CRIdx jj = 0; jj < numNonZeroBlocksInRow; ++jj) {
      tmpColIdx.insert(
          static_cast<CRIdx>(ii + static_cast<int64_t>(jj) * numBlockCols / numNonZeroBlocksInRow) %
          numBlockCols);
    }
    for (auto idx : tmpColIdx) {
      colIdx.push_back(idx);
    }
    rowPtr.push_back(rowPtr.back() + numNonZeroBlocksInRow);
  }
  EXPECT_LE(maxNumNonZeroBlocksPerRow, numBlockCols);

  DynamicArray<Scalar> values(rowPtr[numBlockRows] * kBlockSize * kBlockSize);
  Scalar maxValue = 0;
  for (CRIdx ir = 0; ir < numBlockRows; ++ir) {
    Ptr const start = rowPtr[ir] * kBlockSize * kBlockSize;
    Ptr const end = rowPtr[ir + 1] * kBlockSize * kBlockSize;
    for (Ptr ii = start; ii < end; ++ii) {
      values[ii] = Scalar((ii - start + ir) * kPI);
      maxValue = std::max(maxValue, std::abs(values[ii]));
    }
  }

  BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr> B(numBlockCols, rowPtr, colIdx, values);
  auto BAsView = AsView(B);
  auto BAsConstView = AsConstView(B);

  // Sparse matrix analog of B. Used to generate the expected results.
  DynamicArray<Ptr> cPtr(B.Rows() + 1);
  cPtr[0] = 0;
  for (int i = 0; i < B.Rows(); ++i) {
    cPtr[i + 1] = cPtr[i] + (rowPtr[i / kBlockSize + 1] - rowPtr[i / kBlockSize]) * kBlockSize;
  }

  DynamicArray<CRIdx> cIdx;
  cIdx.reserve(values.size());
  for (CRIdx irb = 0; irb < numBlockRows; ++irb) {
    for (CRIdx rb = 0; rb < kBlockSize; ++rb) {
      for (Ptr pos = rowPtr[irb]; pos < rowPtr[irb + 1]; ++pos) {
        for (CRIdx cb = 0; cb < kBlockSize; ++cb) {
          cIdx.push_back(cb + colIdx[pos] * kBlockSize);
        }
      }
    }
  }

  DynamicArray<Scalar> cValues(values);
  SparseMatrix<Scalar, CRIdx, Ptr> C(B.Cols(), cPtr, cIdx, cValues);

  // Apply() and expressions tests.
  for (auto numColsDense : {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 24, 25}) {
    // Dense matrix for testing.
    MatIn X(B.Cols(), numColsDense);
    MatInView Xview(X.data(), X.Rows(), X.Cols(), X.LeadDim());
    X.SetRandom(1);
    Scalar const xNorm = X.Norm();

    // Allocate inputs and outputs.
    MatOut BX1(B.Rows(), numColsDense), BX2(B.Rows(), numColsDense), BX3(B.Rows(), numColsDense);
    MatOut CX(B.Rows(), numColsDense);
    MatOut DX1(B.Rows(), numColsDense), DX2(B.Rows(), numColsDense), DX3(B.Rows(), numColsDense);
    MatOutView BX1view(BX1.data(), BX1.Rows(), BX1.Cols(), BX1.LeadDim());
    MatOutView BX2view(BX2.data(), BX2.Rows(), BX2.Cols(), BX2.LeadDim());
    MatOutView BX3view(BX3.data(), BX3.Rows(), BX3.Cols(), BX3.LeadDim());

    // Relative tolerance: (2 operations) x (Max number of nnz per row) x
    //                     (Max Entry in Matrix) x (Machine Precision)
    Scalar const relTol = Scalar(2 * (maxNumNonZeroBlocksPerRow * kBlockSize)) * maxValue *
        std::numeric_limits<Scalar>::epsilon();

    // Apply() tests.
    BX1.SetRandom(1);
    BX2.SetRandom(2);
    BX3.SetRandom(3);
    B.Apply(X, BX1);
    BAsView.Apply(X, BX2view); // View for output
    BAsConstView.Apply(Xview, BX3view); // View for input and output
    C.Apply(X, CX);
    DX1 = BX1 - CX;
    DX2 = BX2 - CX;
    DX3 = BX3 - CX;
    EXPECT_LT(DX1.Norm(), xNorm * relTol);
    EXPECT_LT(DX2.Norm(), xNorm * relTol);
    EXPECT_LT(DX3.Norm(), xNorm * relTol);

    // ApplyToRange() tests.
    CRIdx const numBlocksApply = numBlockRows / 2;
    CRIdx const numRowsApply = numBlocksApply * kBlockSize;
    CRIdx const blockBeginApply = 3;
    CRIdx const blockEndApply = blockBeginApply + numBlocksApply;
    CRIdx const rowBeginApply = blockBeginApply * kBlockSize;
    CRIdx const rowEndApply = blockEndApply * kBlockSize;
    EXPECT_LT(blockEndApply, numBlockRows); // Not out of bounds.
    EXPECT_GT(blockBeginApply, 0); // To increase test coverage.
    EXPECT_GT(numBlocksApply, 0); // Not a dummy test.
    BX1.SetRandom(4);
    BX2.SetRandom(5);
    BX3.SetRandom(6);
    MatOut BX10 = BX1;
    MatOut BX20 = BX2;
    MatOut BX30 = BX3;
    B.ApplyToRange(X, BX1, rowBeginApply, rowEndApply);
    BAsView.ApplyToRange(X, BX2view, rowBeginApply, rowEndApply);
    BAsConstView.ApplyToRange(Xview, BX3view, rowBeginApply, rowEndApply);
    DX1 = BX1 - BX10;
    DX1.MiddleRows(rowBeginApply, numRowsApply) =
        BX1.MiddleRows(rowBeginApply, numRowsApply) - CX.MiddleRows(rowBeginApply, numRowsApply);
    DX2 = BX2 - BX20;
    DX2.MiddleRows(rowBeginApply, numRowsApply) =
        BX2.MiddleRows(rowBeginApply, numRowsApply) - CX.MiddleRows(rowBeginApply, numRowsApply);
    DX3 = BX3 - BX30;
    DX3.MiddleRows(rowBeginApply, numRowsApply) =
        BX3.MiddleRows(rowBeginApply, numRowsApply) - CX.MiddleRows(rowBeginApply, numRowsApply);
    EXPECT_LT(DX1.Norm(), xNorm * relTol);
    EXPECT_LT(DX2.Norm(), xNorm * relTol);
    EXPECT_LT(DX3.Norm(), xNorm * relTol);

    // Expressions tests.
    BX1view.SetRandom(7);
    BX2view.SetRandom(8);
    BX3view.SetRandom(9);
    BX1view = B * Xview;
    BX2view = BAsView * Xview;
    BX3view = BAsConstView * Xview;
    DX1 = BX1 - CX;
    DX2 = BX2 - CX;
    DX3 = BX3 - CX;
    EXPECT_LT(DX1.Norm(), xNorm * relTol);
    EXPECT_LT(DX2.Norm(), xNorm * relTol);
    EXPECT_LT(DX3.Norm(), xNorm * relTol);

    BX1view += B * Xview;
    BX2view += BAsView * Xview;
    BX3view += BAsConstView * Xview;
    DX1 = BX1 - Scalar(2) * CX;
    DX2 = BX2 - Scalar(2) * CX;
    DX3 = BX3 - Scalar(2) * CX;
    EXPECT_LT(DX1.Norm(), xNorm * relTol);
    EXPECT_LT(DX2.Norm(), xNorm * relTol);
    EXPECT_LT(DX3.Norm(), xNorm * relTol);

    Xview *= Scalar(2);
    BX1view -= B * Xview;
    BX2view -= BAsView * Xview;
    BX3view -= BAsConstView * Xview;
    DX1 = BX1;
    DX2 = BX2;
    DX3 = BX3;
    EXPECT_LT(DX1.Norm(), xNorm * relTol);
    EXPECT_LT(DX2.Norm(), xNorm * relTol);
    EXPECT_LT(DX3.Norm(), xNorm * relTol);

    // TransposeApply() tests.
    {
      MatIn Y(B.Rows(), numColsDense);
      Y.SetRandom(5);
      MatOut BtY1(B.Cols(), numColsDense);
      BtY1.SetRandom(1);
      MatOut BtY2(B.Cols(), numColsDense);
      BtY2.SetRandom(2);
      MatOut BtY3(B.Cols(), numColsDense);
      BtY3.SetRandom(3);
      MatOut CtY(B.Cols(), numColsDense);
      CtY.SetRandom(4);
      B.TransposeApply(Y, BtY1);
      BAsView.TransposeApply(Y, BtY2);
      BAsConstView.TransposeApply(Y, BtY3);
      C.TransposeApply(Y, CtY);
      //
      auto const tol = CtY.Norm() * std::numeric_limits<Scalar>::epsilon();
      EXPECT_TRUE(test::NearEqualMatrices(BtY1, BtY2, tol));
      EXPECT_TRUE(test::NearEqualMatrices(BtY1, BtY3, tol));
      EXPECT_TRUE(test::NearEqualMatrices(BtY1, CtY, tol));
    }

  } // for (auto numColsDense : {...})
}

template <typename Scalar, typename CRIdx, typename Ptr, int kBlockSize>
void TestBlockSparseMatrixVectorProductHelper() {
  constexpr auto kColMajor = krylov::Direction::ColMajor;
  constexpr auto kRowMajor = krylov::Direction::RowMajor;
  TestBlockSparseMatrixVectorProduct<Scalar, CRIdx, Ptr, kBlockSize, kColMajor, kColMajor>();
  TestBlockSparseMatrixVectorProduct<Scalar, CRIdx, Ptr, kBlockSize, kColMajor, kRowMajor>();
  TestBlockSparseMatrixVectorProduct<Scalar, CRIdx, Ptr, kBlockSize, kRowMajor, kColMajor>();
  TestBlockSparseMatrixVectorProduct<Scalar, CRIdx, Ptr, kBlockSize, kRowMajor, kRowMajor>();
}

/**************************************************************************************
 * namespace mochi::details Helper functions
 */

namespace mochi::details {

template <typename Scalar, typename CRIdx, typename Ptr, CRIdx kBlockSize>
static auto BuildBlockSparseMatrix(int numBlockRows) {
  DynamicArray<Ptr> rowPtr;
  rowPtr.reserve(numBlockRows + 1);
  DynamicArray<CRIdx> colIdx;
  int offsets[] = {-9, -8, -7, -3, -1, 0, 1, 3, 7, 11, 12};
  colIdx.reserve(numBlockRows * sizeof(offsets) / sizeof(int));
  rowPtr.push_back(0);
  for (CRIdx ir = 0; ir < numBlockRows; ++ir) {
    for (auto o : offsets) {
      if (ir + o >= 0 && ir + o < numBlockRows) {
        colIdx.push_back(ir + o);
      }
    }
    rowPtr.push_back((Ptr)colIdx.size());
  }
  DynamicArray<Scalar> values;
  values.reserve(rowPtr[numBlockRows] * kBlockSize * kBlockSize);
  for (CRIdx ir = 0; ir < numBlockRows; ++ir) {
    Ptr const start = rowPtr[ir] * kBlockSize * kBlockSize;
    Ptr const end = rowPtr[ir + 1] * kBlockSize * kBlockSize;
    for (Ptr ii = start; ii < end; ++ii) {
      values.push_back(Scalar(ii - start) * Scalar(kPI) + ir);
    }
  }
  return BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr>(
      numBlockRows, std::move(rowPtr), std::move(colIdx), std::move(values));
}

template <typename Scalar, typename CRIdx, typename Ptr, CRIdx kBlockSize>
static void BlockSparseMatrixMove() {
  int nBlocks = 16;
  auto A = BuildBlockSparseMatrix<Scalar, CRIdx, Ptr, kBlockSize>(nBlocks);
  BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr> B;
  B = std::move(A);
  EXPECT_EQ(B.BlockRows(), nBlocks);
  EXPECT_EQ(B.BlockCols(), nBlocks);

  auto C = BuildBlockSparseMatrix<Scalar, CRIdx, Ptr, kBlockSize>(nBlocks);
  EXPECT_EQ(B.Values(), C.Values());

  std::swap(A, B);
  EXPECT_EQ(A.Values(), C.Values());

  BlockSparseMatrixView<Scalar, kBlockSize, CRIdx, Ptr> AV;
  AV.Reset(A);
  EXPECT_EQ(AV.BlockRows(), nBlocks);
  EXPECT_EQ(AV.BlockCols(), nBlocks);

  auto const& v0 = A.Values();
  auto const& v1 = AV.Values();
  EXPECT_EQ(v0.size(), v1.size());
  bool equal = true;
  for (int i = 0; i < v0.size(); ++i) {
    equal &= v0[i] == v1[i];
  }
  EXPECT_TRUE(equal);
}

template <typename Scalar, typename CRIdx, typename Ptr>
static void BlockedStructure() {
  /// Tests for SparseMatrix specialization.
  using SMatrix = SparseMatrix<Scalar, CRIdx, Ptr>;
  CRIdx const numCols = 6;
  {
    // Successful case
    DynamicArray<Ptr> rptr{0, 4, 8, 10, 12};
    DynamicArray<CRIdx> cIdx{0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 4, 5};
    DynamicArray<Scalar> val(cIdx.size(), Scalar(1));
    SMatrix K(numCols, rptr, cIdx, val);
    EXPECT_TRUE(IsBlockable<1>(K));
    EXPECT_TRUE(IsBlockable<2>(K));
  }
  {
    // Successful case because dense square matrices are blockable with "block size = matrix size"
    DynamicArray<Ptr> rptr{0, 6, 12, 18, 24, 30, 36};
    DynamicArray<CRIdx> cIdx{0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                             0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
    DynamicArray<Scalar> val(cIdx.size(), Scalar(1));
    SMatrix K(numCols, rptr, cIdx, val);
    EXPECT_TRUE(IsBlockable<6>(K));
  }
  {
    // Failing case because the first column index in row #1 is not a multiple of the block size
    DynamicArray<Ptr> rptr{0, 4, 8, 10, 12};
    DynamicArray<CRIdx> cIdx{1, 2, 3, 4, 1, 2, 3, 4, 4, 5, 4, 5};
    DynamicArray<Scalar> val(cIdx.size(), Scalar(1));
    SMatrix K(numCols, rptr, cIdx, val);
    EXPECT_FALSE(IsBlockable<2>(K));
    EXPECT_TRUE(IsBlockable<1>(K)); // But blockable with block size = 1
  }
  {
    // Failing case because column indices in row #0 are not sorted
    DynamicArray<Ptr> rptr{0, 4, 8, 10, 12};
    DynamicArray<CRIdx> cIdx{0, 2, 1, 3, 0, 1, 2, 3, 4, 5, 4, 5};
    DynamicArray<Scalar> val(cIdx.size(), Scalar(1));
    SMatrix K(numCols, rptr, cIdx, val);
    EXPECT_FALSE(IsBlockable<2>(K));
    EXPECT_TRUE(IsBlockable<1>(K)); // But blockable with block size = 1
  }
  {
    // Failing case because row #1 differs from row #0
    DynamicArray<Ptr> rptr{0, 4, 8, 10, 12};
    DynamicArray<CRIdx> cIdx{0, 1, 2, 3, 0, 2, 3, 5, 4, 5, 4, 5};
    DynamicArray<Scalar> val(cIdx.size(), Scalar(1));
    SMatrix K(numCols, rptr, cIdx, val);
    EXPECT_FALSE(IsBlockable<2>(K));
    EXPECT_TRUE(IsBlockable<1>(K)); // But blockable with block size = 1
  }
  {
    // Failing case because row #3 has only 1 entry
    DynamicArray<Ptr> rptr{0, 4, 8, 10, 11};
    DynamicArray<CRIdx> cIdx{0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 4};
    DynamicArray<Scalar> val(cIdx.size(), Scalar(1));
    SMatrix K(numCols, rptr, cIdx, val);
    EXPECT_FALSE(IsBlockable<2>(K));
    EXPECT_TRUE(IsBlockable<1>(K)); // But blockable with block size = 1
  }
  {
    // Failing case because the number of rows is not a multiple of the block size
    DynamicArray<Ptr> rptr{0, 4, 8, 10, 12, 12};
    DynamicArray<CRIdx> cIdx{0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 4, 5};
    DynamicArray<Scalar> val(cIdx.size(), Scalar(1));
    SMatrix K(numCols, rptr, cIdx, val);
    EXPECT_FALSE(IsBlockable<2>(K));
    EXPECT_TRUE(IsBlockable<1>(K)); // But blockable with block size = 1
  }
  {
    // Failing case because the number of columns is not a multiple of the block size
    DynamicArray<Ptr> rptr{0, 4, 8, 10, 12};
    DynamicArray<CRIdx> cIdx{0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 4, 5};
    DynamicArray<Scalar> val(cIdx.size(), Scalar(1));
    SMatrix K(numCols + 1, rptr, cIdx, val);
    EXPECT_FALSE(IsBlockable<2>(K));
    EXPECT_TRUE(IsBlockable<1>(K)); // But blockable with block size = 1
  }

  /// Tests for Matrix specialization.
  {
    Matrix<Scalar> mat1;
    RowMatrix<Scalar> mat2;
    EXPECT_TRUE(IsBlockable<1>(mat1));
    EXPECT_TRUE(IsBlockable<1>(mat2));
    EXPECT_TRUE(IsBlockable<2>(AsView(mat1)));
    EXPECT_TRUE(IsBlockable<2>(AsView(mat2)));
    EXPECT_TRUE(IsBlockable<3>(AsConstView(mat1)));
    EXPECT_TRUE(IsBlockable<3>(AsConstView(mat2)));
  }
  {
    Matrix<Scalar, 3, 6> mat1;
    RowMatrix<Scalar, 3, 6> mat2;
    EXPECT_TRUE(IsBlockable<1>(mat1));
    EXPECT_TRUE(IsBlockable<1>(AsView(mat1)));
    EXPECT_TRUE(IsBlockable<1>(AsConstView(mat1)));
    EXPECT_TRUE(IsBlockable<1>(mat2));
    EXPECT_FALSE(IsBlockable<2>(mat1));
    EXPECT_FALSE(IsBlockable<2>(mat2));
    EXPECT_FALSE(IsBlockable<2>(AsView(mat2)));
    EXPECT_FALSE(IsBlockable<2>(AsConstView(mat2)));
    EXPECT_TRUE(IsBlockable<3>(mat1));
    EXPECT_TRUE(IsBlockable<3>(mat2));
  }
  {
    Matrix<Scalar> mat1(8, 6);
    RowMatrix<Scalar> mat2(8, 6);
    EXPECT_TRUE(IsBlockable<1>(mat1));
    EXPECT_TRUE(IsBlockable<1>(mat2));
    EXPECT_TRUE(IsBlockable<2>(mat1));
    EXPECT_TRUE(IsBlockable<2>(AsView(mat1)));
    EXPECT_TRUE(IsBlockable<2>(AsConstView(mat1)));
    EXPECT_TRUE(IsBlockable<2>(mat2));
    EXPECT_FALSE(IsBlockable<3>(mat1));
    EXPECT_FALSE(IsBlockable<3>(mat2));
    EXPECT_FALSE(IsBlockable<3>(AsView(mat2)));
    EXPECT_FALSE(IsBlockable<3>(AsConstView(mat2)));
  }
}

} // namespace mochi::details

/**************************************************************************************
 * RowMultiplier Test Cases
 */

namespace mochi::details::row_multiplier {

template <int kBlockSize>
void Example01(int nCol) {
  std::vector<int> col{1, 2, 3};
  RowMatrix<real> x(4 * kBlockSize, nCol);
  x.SetRandom(123);
  RowMatrix<real, kBlockSize, krylov::kDynamic> val(kBlockSize, col.size() * kBlockSize);
  val.SetRandom(321);
  auto ref = RowMatrix<real>::Zero(kBlockSize, nCol);
  for (int ii = 0; ii < col.size(); ++ii) {
    ref += val.Block(0, ii * kBlockSize, kBlockSize, kBlockSize) *
        x.MiddleRows(col[ii] * kBlockSize, kBlockSize);
  }
  RowMatrix<real> Ax(kBlockSize, nCol);
  details::RowMultiplier<real, kBlockSize>::ApplyToRowMajor(
      col, val, details::GetAccessor(x), details::GetAccessor(Ax), 0, nCol);
  EXPECT_TRUE(test::NearEqualMatrices(ref, Ax, kDefaultNearEqualEpsilon<real> * nCol * kBlockSize));
}

template <int kBlockSize>
void Example02() {
  std::vector<int> col{1, 2, 4, 6, 7};
  ColumnVector<real> x(8 * kBlockSize);
  x.SetRandom(123);
  RowMatrix<real, kBlockSize, krylov::kDynamic> val(kBlockSize, col.size() * kBlockSize);
  val.SetRandom(321);
  auto ref = ColumnVector<real>::Zero(kBlockSize);
  for (int ii = 0; ii < col.size(); ++ii) {
    ref += val.Block(0, ii * kBlockSize, kBlockSize, kBlockSize) *
        x.MiddleRows(col[ii] * kBlockSize, kBlockSize);
  }
  ColumnVector<real> Ax(kBlockSize);
  auto const rowLeadDim = kBlockSize * col.size();
  ColumnVector<real> xTmp;
  //--- Use safe value for xlen
  auto xlen = details::RowMultiplier<real, kBlockSize>::GetWorkspaceSize(col.size() * kBlockSize);
  details::RowMultiplier<real, kBlockSize>::ApplyToColVector(
      col,
      Span{val.Data(), val.Rows() * val.Cols()},
      col.size(),
      rowLeadDim,
      details::GetAccessor(x),
      details::GetAccessor(Ax),
      0,
      0,
      xTmp,
      xlen);
  EXPECT_TRUE(test::NearEqualMatrices(ref, Ax, kDefaultNearEqualEpsilon<real> * kBlockSize));
}

} // namespace mochi::details::row_multiplier

TEST(RowMultiplier, ApplyToRowMajor) {
  for (auto k : {1, 2, 4, 7, 8, 15, 16, 17}) {
    details::row_multiplier::Example01<1>(k);
    details::row_multiplier::Example01<3>(k);
    details::row_multiplier::Example01<4>(k);
    details::row_multiplier::Example01<6>(k);
    details::row_multiplier::Example01<8>(k);
    details::row_multiplier::Example01<9>(k);
    details::row_multiplier::Example01<12>(k);
  }
}

TEST(RowMultiplier, ApplyToColVector) {
  details::row_multiplier::Example02<1>();
  details::row_multiplier::Example02<3>();
  details::row_multiplier::Example02<4>();
  details::row_multiplier::Example02<6>();
  details::row_multiplier::Example02<8>();
  details::row_multiplier::Example02<9>();
  details::row_multiplier::Example02<12>();
  details::row_multiplier::Example02<14>();
  details::row_multiplier::Example02<15>();
  details::row_multiplier::Example02<17>();
}
/**************************************************************************************
 * BlockSparseMatrix Test Cases
 */

TEST(BlockSparseMatrix, DefaultConstructor) {
  BlockSparseMatrix<real, 3> mat;
  ExpectEmpty(mat);
  BlockSparseMatrixView<real, 3> view;
  ExpectEmpty(view);
}

TEST(BlockSparseMatrix, ConstructFromArrays) {
  using TC = TestCase_Block3_6x6;
  auto mat = MakeMat<TC>();
  CheckMat<TC>(mat);
  auto view = MakeMatConstView<TC>();
  CheckMat<TC>(view);
}

TEST(BlockSparseMatrix, MoveConstructor) {
  using TC = TestCase_Block3_6x6;
  auto mat = MakeMat<TC>();
  auto mat2 = std::move(mat); // move construct from same type
  CheckMat<TC>(mat2);
  EXPECT_FALSE(HasSameAddresses(mat, mat2)); // NOLINT(bugprone-use-after-move)
  BlockSparseMatrix<real, 3, int, int, std::vector> mat3(std::move(mat2));
  CheckMat<TC>(mat3);
  CheckMat<TC>(mat2); // NOLINT(bugprone-use-after-move)
}

TEST(BlockSparseMatrix, CopyConstructor) {
  using TC = TestCase_Block3_6x6;
  auto mat = MakeMat<TC>();
  auto mat2 = mat; // copy construct from same type
  CheckMat<TC>(mat2);
  EXPECT_FALSE(HasSameAddresses(mat, mat2));
  auto view = MakeMatConstView<TC>();
  auto view2 = view; // copy construct from same type
  CheckMat<TC>(view2);
  EXPECT_TRUE(HasSameAddresses(view, view2));
  //
  BlockSparseMatrix<real, 3, int, int, std::vector> mat3(
      std::move(mat2)); // Move constructor from default to non-default storage type.
  CheckMat<TC>(mat3);
  // Move not possible from a different storage type. mat2 must have been copied.
  CheckMat<TC>(mat2); // NOLINT(bugprone-use-after-move)
  EXPECT_FALSE(HasSameAddresses(mat, mat3));
  //
  BlockSparseMatrix<real, 3, int, int> mat5(
      std::move(mat3)); // Move constructor from non-default to default storage type.
  CheckMat<TC>(mat5);
  // Move not possible from a different storage type. mat3 must have been copied.
  CheckMat<TC>(mat3); // NOLINT(bugprone-use-after-move)
  EXPECT_FALSE(HasSameAddresses(mat3, mat5));
}

TEST(BlockSparseMatrix, ConstructFromOther) {
  using TC = TestCase_Block3_6x6;
  using BMat = BlockSparseMatrix<real, TC::kBlockSize>;
  using BMatVector = BlockSparseMatrix<real, TC::kBlockSize, int, int, std::vector>;
  using BMatView = BlockSparseMatrixView<real, TC::kBlockSize>;
  using BMatConstView = BlockSparseMatrixView<real const, TC::kBlockSize>;
  BMat mat = MakeMat<TC>();
  BMatVector matvector = mat;
  BMatView view = AsView(mat);
  BMatConstView cview = AsConstView(mat);

  // Owning from "std::vector"-matrix
  {
    BMat mat2 = matvector;
    CheckMat<TC>(mat2);
    EXPECT_FALSE(HasSameAddresses(mat2, matvector));
  }

  // Owning from view
  {
    BMat mat2 = view;
    CheckMat<TC>(mat2);
    EXPECT_FALSE(HasSameAddresses(mat2, view));
  }

  // Owning from const view
  {
    BMat mat2 = cview;
    CheckMat<TC>(mat2);
    EXPECT_FALSE(HasSameAddresses(mat2, cview));
  }

  // View from owning
  {
    BMatView view2 = mat;
    CheckMat<TC>(view2);
    EXPECT_TRUE(HasSameAddresses(mat, view2));
  }

  // Const view from owning
  {
    BMatConstView view2 = mat;
    CheckMat<TC>(view2);
    EXPECT_TRUE(HasSameAddresses(mat, view2));
  }

  // View from owning & "std::vector" storage
  {
    BMatView view2 = matvector;
    CheckMat<TC>(view2);
    EXPECT_TRUE(HasSameAddresses(matvector, view2));
  }

  // Const view from owning & "std::vector" storage
  {
    BMatConstView view2 = matvector;
    CheckMat<TC>(view2);
    EXPECT_TRUE(HasSameAddresses(matvector, view2));
  }

  // Const view from view
  {
    BMatConstView view2 = view;
    CheckMat<TC>(view2);
    EXPECT_TRUE(HasSameAddresses(view, view2));
  }
}

TEST(BlockSparseMatrix, ConstructFromGraph) {
  // From empty graph
  {
    BlockSparseMatrix<real, 3> mat0(0, Graph<int, int>{{}, {}});
    EXPECT_EQ(0, mat0.Rows());
    EXPECT_EQ(0, mat0.Cols());
    BlockSparseMatrix<real, 3> mat1(Graph<int, int>{{}, {}});
    EXPECT_EQ(0, mat1.Rows());
    EXPECT_EQ(0, mat1.Cols());
  }

  // From 6x6 graph
  {
    using TC = TestCase_Block3_6x6;
    DynamicArray<int> pointers(std::begin(TC::kBlockPointers), std::end(TC::kBlockPointers));
    DynamicArray<int> indices(std::begin(TC::kBlockIndices), std::end(TC::kBlockIndices));
    Graph<int, int> graph(pointers, indices);
    BlockSparseMatrix<real, 3> mat(2, graph);
    EXPECT_EQ(graph.NumTargets(), mat.NumNonZeroBlocks());
    EXPECT_EQ(graph.NumTargets() * 3 * 3, isize(mat.Values()));
    std::copy(std::begin(TC::kValues), std::end(TC::kValues), mat.Values().begin());
    CheckMat<TC>(mat);

    // Repeat, but this time use std::move
    BlockSparseMatrix<real, 3> mat2(2, std::move(graph));
    EXPECT_EQ(0, isize(graph.GetPointers())); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(0, isize(graph.GetTargets())); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(isize(mat.Values()), isize(mat2.Values()));
    std::copy(std::begin(TC::kValues), std::end(TC::kValues), mat2.Values().begin());
    CheckMat<TC>(mat2);

    // If the number of block columns if not specified, it must default to a square matrix whose
    // number of block rows and block columns is derived from the graph.
    BlockSparseMatrix<real, 3> mat3(Graph<int, int>(pointers, indices));
    std::copy(std::begin(TC::kValues), std::end(TC::kValues), mat3.Values().begin());
    CheckMat<TC>(mat3);
  }

  // From 6x6 graph
  {
    using TC = TestCase_Block3_6x6;
    int constexpr N = 2048;
    std::array<int, N> buffer1{};
    MonotonicAllocator mem1(buffer1.data(), N * sizeof(int));
    DynamicArray<int> pointers(std::begin(TC::kBlockPointers), std::end(TC::kBlockPointers), &mem1);
    DynamicArray<int> indices(std::begin(TC::kBlockIndices), std::end(TC::kBlockIndices), &mem1);

    //--- Make a copy of pointers and indices to preserve the original values
    DynamicArray<int> pg(pointers, &mem1), ig(indices, &mem1);

    //--- Move the copies into the graph
    //--- The graph should storage its data on "mem1"-pace
    Graph<int, int> graph(std::move(pg), std::move(ig));

    std::array<char, N * sizeof(real)> buffer2{};
    MonotonicAllocator mem2(buffer2.data(), N * sizeof(real));
    //--- The block sparse matrix will store its integral containers on "new-delete"-space
    //--- and the scalar values on "mem2"-space
    BlockSparseMatrix<real, 3> mat(2, graph, &mem2);
    EXPECT_EQ(graph.NumTargets(), mat.NumNonZeroBlocks());
    EXPECT_EQ(graph.NumTargets() * 3 * 3, isize(mat.Values()));
    std::copy(std::begin(TC::kValues), std::end(TC::kValues), mat.Values().begin());
    CheckMat<TC>(mat);
    EXPECT_GE(int(mat.Values().begin() - (real*)(buffer2.data())), 0);
    EXPECT_LT(int(mat.Values().begin() - (real*)(buffer2.data())), N);

    // Repeat, but this time use std::move
    //--- The block sparse matrix will store its integral containers on "mem1"-space
    //--- and the scalar values on "mem2"-space
    BlockSparseMatrix<real, 3> mat2(2, std::move(graph), &mem2);
    EXPECT_EQ(0, isize(graph.GetPointers())); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(0, isize(graph.GetTargets())); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(isize(mat.Values()), isize(mat2.Values()));
    std::copy(std::begin(TC::kValues), std::end(TC::kValues), mat2.Values().begin());
    CheckMat<TC>(mat2);
    EXPECT_GE(int(mat2.Pointers().data() - buffer1.data()), 0);
    EXPECT_LT(int(mat2.Pointers().data() - buffer1.data()), N);
    EXPECT_GE(int(mat2.Indices().data() - buffer1.data()), 0);
    EXPECT_LT(int(mat2.Indices().data() - buffer1.data()), N);
    EXPECT_GE(int(mat2.Values().begin() - (real*)(buffer2.data())), 0);
    EXPECT_LT(int(mat2.Values().begin() - (real*)(buffer2.data())), N);

    // If the number of block columns if not specified, it must default to a square matrix whose
    // number of block rows and block columns is derived from the graph.
    BlockSparseMatrix<real, 3> mat3(Graph<int, int>(pointers, indices), &mem2);
    std::copy(std::begin(TC::kValues), std::end(TC::kValues), mat3.Values().begin());
    EXPECT_GE(int(mat3.Values().begin() - (real*)(buffer2.data())), 0);
    EXPECT_LT(int(mat3.Values().begin() - (real*)(buffer2.data())), N);
    CheckMat<TC>(mat3);
  }
}

TEST(BlockSparseMatrix, BlockedStructure) {
  // Specializations tested: Scalar={float,double}, CRIdx={short,int,long}, Ptr=int. Note
  // IsBlockable is agnostic to Scalar and Ptr.
  mochi::details::BlockedStructure<real, short, int>();
  mochi::details::BlockedStructure<real, int, int>();
  mochi::details::BlockedStructure<real, long, int>();
}

TEST(BlockSparseMatrix, BlockedMove) {
  mochi::details::BlockSparseMatrixMove<real, int, int, 3>();
}

TEST(BlockSparseMatrix, AsView) {
  using TC = TestCase_Block3_6x6;
  auto mat = MakeMat<TC>();
  auto view = AsView(mat);
  static_assert(std::is_same_v<decltype(view), BlockSparseMatrixView<real, TC::kBlockSize>>);
  CheckMat<TC>(view);
  auto cview = AsConstView(mat);
  static_assert(std::is_same_v<
                decltype(cview),
                BlockSparseMatrixView<real const, TC::kBlockSize, int const, int const>>);
  CheckMat<TC>(cview);
}

TEST(BlockSparseMatrix, OtherSizes) {
  CheckMat<TestCase_Block1_3x3>(MakeMat<TestCase_Block1_3x3>());
  CheckMat<TestCase_Block3_3x6>(MakeMat<TestCase_Block3_3x6>());
  CheckMat<TestCase_Block3_6x3>(MakeMat<TestCase_Block3_6x3>());
}

TEST(BlockSparseMatrix, PlusEquals) {
  // |   123|
  // |   456|
  // |   789|
  // |135   |
  // |246   |
  // |357   |
  using TC = TestCase_Block3_6x6;

  // BlockSparseMatrix += self
  {
    auto a = MakeMat<TC>();
    a += a;
    for (int r = 0; r < a.Rows(); ++r) {
      for (int c = 0; c < a.Cols(); ++c) {
        EXPECT_NEAR_EQ(TC::kValuesDense[r][c] * 2_r, a(r, c));
      }
    }
  }

  // BlockSparseMatrix += BlockSparseMatrixView with same sparsity
  {
    auto a = MakeMat<TC>();
    auto const b = MakeMat<TC>();
    a += AsConstView(b);
    for (int r = 0; r < a.Rows(); ++r) {
      for (int c = 0; c < a.Cols(); ++c) {
        EXPECT_NEAR_EQ(TC::kValuesDense[r][c] * 2_r, a(r, c));
      }
    }
  }

  // BlockSparseMatrix += BlockSparseMatrix with no non-zero values
  {
    auto a = MakeMat<TC>();
    auto const b = BlockSparseMatrix<real, 3>(2, {0, 0, 0}, {}, {});
    CheckMat<TC>(a);
    a += b;
  }

  // BlockSparseMatrix += BlockSparseMatrix with same dimensions, but fewer non-zero values
  {
    // |   123|      |   111|
    // |   456|      |   111|
    // |   789|  +=  |   111|
    // |135   |      |      |
    // |246   |      |      |
    // |357   |      |      |
    auto a = MakeMat<TC>();
    auto const b = BlockSparseMatrix<real, 3>(
        2,
        {0, 1, 1}, // pointers
        {1}, // indices
        {1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r}); // values
    a += b;
    for (int r = 0; r < a.Rows(); ++r) {
      for (int c = 0; c < a.Cols(); ++c) {
        real expectedValue = TC::kValuesDense[r][c];
        if ((r < 3) && (c >= 3)) {
          expectedValue += 1_r;
        }
        EXPECT_NEAR_EQ(expectedValue, a(r, c));
      }
    }
  }

  // BlockSparseMatrix += BlockSparseMatrix with fewer rows.
  {
    // |   123|      |   111|
    // |   456|      |   111|
    // |   789|  +=  |   111|
    // |135   |
    // |246   |
    // |357   |
    auto a = MakeMat<TC>();
    auto const b = BlockSparseMatrix<real, 3>(
        2,
        {0, 1}, // pointers
        {1}, // indices
        {1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r}); // values
    a += b;
    for (int r = 0; r < a.Rows(); ++r) {
      for (int c = 0; c < a.Cols(); ++c) {
        real expectedValue = TC::kValuesDense[r][c];
        if ((r < 3) && (c >= 3)) {
          expectedValue += 1_r;
        }
        EXPECT_NEAR_EQ(expectedValue, a(r, c));
      }
    }
  }

  // BlockSparseMatrix += BlockSparseMatrix with fewer columns.
  {
    // |   123|      |   |
    // |   456|      |   |
    // |   789|  +=  |   |
    // |135   |      |111|
    // |246   |      |111|
    // |357   |      |111|
    auto a = MakeMat<TC>();
    auto const b = BlockSparseMatrix<real, 3>(
        1,
        {0, 0, 1}, // pointers
        {0}, // indices
        {1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r}); // values
    a += b;
    for (int r = 0; r < a.Rows(); ++r) {
      for (int c = 0; c < a.Cols(); ++c) {
        real expectedValue = TC::kValuesDense[r][c];
        if ((r >= 3) && (c < 3)) {
          expectedValue += 1_r;
        }
        EXPECT_NEAR_EQ(expectedValue, a(r, c));
      }
    }
  }
}

TEST(BlockSparseMatrix, MatrixVectorProduct) {
  TestBlockSparseMatrixVectorProductHelper<real, int, int, 1>();
  TestBlockSparseMatrixVectorProductHelper<real, int, int, 2>();
  TestBlockSparseMatrixVectorProductHelper<real, int, int, 3>();
  TestBlockSparseMatrixVectorProductHelper<real, int, int, 4>();
  //
  TestBlockSparseMatrixVectorProductHelper<real, int, int64_t, 3>();
  TestBlockSparseMatrixVectorProductHelper<real, int, int64_t, 4>();

// Compilation is slow. Skip it in non-debug builds. Edit this line to locally run it in non-debug
// builds.
#if MOCHI_DEBUG
  TestBlockSparseMatrixVectorProductHelper<real, int, int, 5>();
  TestBlockSparseMatrixVectorProductHelper<real, int, int, 6>();
#endif
}

TEST(BlockSparseMatrix, MatrixVectorProductWithLargeBlockSize) {
// Compilation is slow. Skip it in non-debug builds. Edit this line to locally run it in non-debug
// builds.
#if MOCHI_DEBUG
  TestBlockSparseMatrixVectorProductHelper<real, int, int, 7>();
  TestBlockSparseMatrixVectorProductHelper<real, int, int, 8>();
  TestBlockSparseMatrixVectorProductHelper<real, int, int, 9>();
#endif
}

TEST(BlockSparseMatrix, Empty) {
  // Default constructed
  {
    BlockSparseMatrix<real, 3> mat;
    EXPECT_TRUE(mat.empty());
    ExpectEmpty(mat);
  }

  // Zero rows
  {
    BlockSparseMatrix<real, 3> mat(1, {}, {}, {});
    EXPECT_TRUE(mat.empty());
    EXPECT_FALSE(mat); // operator bool
    EXPECT_EQ(0, mat.NumNonZeroBlocks());
    EXPECT_EQ(0, mat.NumNonZeros());
  }

  // Zero non-zeros
  {
    BlockSparseMatrix<real, 3> mat(1, {0, 0}, {}, {});
    EXPECT_FALSE(mat.empty());
    EXPECT_TRUE(mat); // operator bool
    EXPECT_EQ(0, mat.NumNonZeroBlocks());
    EXPECT_EQ(0, mat.NumNonZeros());
  }

  // Non-Zero non-zeros
  {
    BlockSparseMatrix<real, 3> mat(1, {0, 1}, {0}, {0_r, 1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r});
    EXPECT_FALSE(mat.empty());
    EXPECT_TRUE(mat); // operator bool
    EXPECT_EQ(1, mat.NumNonZeroBlocks());
    EXPECT_EQ(9, mat.NumNonZeros());
  }
}

TEST(BlockSparseMatrix, SetZero) {
  auto A = MakeMat<TestCase_Block3_6x6>();
  EXPECT_NE(0_r, A(0, 4));
  A.SetZero();
  for (int r = 0; r < A.Rows(); ++r) {
    for (int c = 0; c < A.Cols(); ++c) {
      EXPECT_EQ(0_r, A(r, c));
    }
  }
}

TEST(BlockSparseMatrix, Duplicate) {
  using TC = TestCase_Block3_6x6;
  BlockSparseMatrix<real, 3> mat = MakeMat<TC>();
  BlockSparseMatrix<real, 3> mat2 = mat.Duplicate();
  BlockSparseMatrix<real, 3> mat3 = AsConstView(mat).Duplicate();
  CheckMat<TC>(mat);
  CheckMat<TC>(mat2);
  CheckMat<TC>(mat3);
  EXPECT_NE(mat.Values().data(), mat2.Values().data()); // Not the same address
  EXPECT_NE(mat.Values().data(), mat3.Values().data()); // Not the same address
  EXPECT_NE(mat2.Values().data(), mat3.Values().data()); // Not the same address
}

TEST(BlockSparseMatrix, IsBlockRowEmpty) {
  constexpr int kBlockSize = 2;
  int const numBlockCols = 3;
  DynamicArray<int> rowPtr = {0, 2, 2, 3, 3};
  DynamicArray<int> colIdx = {0, 1, 2};
  DynamicArray<real> values = {1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, 9_r, 10_r, 11_r, 12_r};
  BlockSparseMatrix<real, kBlockSize> A(numBlockCols, rowPtr, colIdx, values);

  EXPECT_FALSE(A.IsBlockRowEmpty(0));
  EXPECT_TRUE(A.IsBlockRowEmpty(1));
  EXPECT_FALSE(A.IsBlockRowEmpty(2));
  EXPECT_TRUE(A.IsBlockRowEmpty(3));

  EXPECT_FALSE(AsView(A).IsBlockRowEmpty(0));
  EXPECT_TRUE(AsView(A).IsBlockRowEmpty(1));
  EXPECT_FALSE(AsConstView(A).IsBlockRowEmpty(2));
  EXPECT_TRUE(AsConstView(A).IsBlockRowEmpty(3));
}

TEST(BlockSparseMatrix, Norm) {
  using TC = TestCase_Block3_6x6;
  auto mat = MakeMat<TC>();
  real expectedNormSqr = 0_r;
  for (int r = 0; r < mat.Rows(); ++r) {
    for (int c = 0; c < mat.Cols(); ++c) {
      expectedNormSqr += mat(r, c) * mat(r, c);
    }
  }
  EXPECT_NEAR_EQ(expectedNormSqr, mat.NormSqr());
  EXPECT_NEAR_EQ(Sqrt(expectedNormSqr), mat.Norm());
}

TEST(BlockSparseMatrix, SparseMatProduct1) {
  auto A = MakeMat<TestCase_Block3_6x3>();
  auto B = MakeMat<TestCase_Block3_3x6>();
  DynamicArray<int> cRowPtr({0, 0, 1});
  DynamicArray<int> cColIdx({1});
  DynamicArray<real> cValues(9, 0);
  BlockSparseMatrix<real, 3, int, int> C(2, cRowPtr, cColIdx, cValues);
  mochi::details::SparseMatProduct(A, B, C);
  //
  auto crow = C.Values(1);
  EXPECT_EQ(1, crow.NumBlocks());
  auto cmat = crow[0];
  EXPECT_EQ(real(30), cmat(0, 0));
  EXPECT_EQ(real(36), cmat(0, 1));
  EXPECT_EQ(real(42), cmat(0, 2));
  EXPECT_EQ(real(66), cmat(1, 0));
  EXPECT_EQ(real(81), cmat(1, 1));
  EXPECT_EQ(real(96), cmat(1, 2));
  EXPECT_EQ(real(102), cmat(2, 0));
  EXPECT_EQ(real(126), cmat(2, 1));
  EXPECT_EQ(real(150), cmat(2, 2));
}

namespace mochi::details {
template <int kBlockSize, typename Scalar, typename CRIdx, typename Ptr>
void SparseMatProduct2(int nBlocks) {
  auto A = details::BuildBlockSparseMatrix<Scalar, CRIdx, Ptr, kBlockSize>(nBlocks);
  auto Adense = ToMatrix(A);
  Matrix<Scalar> AAdense = Adense * Adense;
  auto AAref = ToBlockSparseMatrix<kBlockSize>(AAdense, true);
  DynamicArray<Ptr> aPtr(AAref.Pointers().begin(), AAref.Pointers().end());
  DynamicArray<CRIdx> aIdx(AAref.Indices().begin(), AAref.Indices().end());
  DynamicArray<Scalar> values(AAref.Values().size(), Scalar(0));
  BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr> AA(AAref.BlockCols(), aPtr, aIdx, values);
  mochi::details::SparseMatProduct(A, A, AA);
  auto refValues = AAref.Values();
  auto newValues = AA.Values();
  for (size_t i = 0; i < newValues.size(); ++i) {
    EXPECT_NEAR(
        newValues[i],
        refValues[i],
        Scalar(AAref.MaxNnzPerRow()) * Abs(refValues[i] * std::numeric_limits<Scalar>::epsilon()));
  }
}
} // namespace mochi::details

TEST(BlockSparseMatrix, SparseMatProduct2) {
  mochi::details::SparseMatProduct2<2, real, int, int>(23);
  mochi::details::SparseMatProduct2<3, real, int, int>(19);
  mochi::details::SparseMatProduct2<4, real, int, int>(21);
}

TEST(BlockSparseMatrix, TransposeEx1) {
  auto A = Matrix<real>::Zero(2, 3);
  A(0, 0) = real(1.0);
  A(0, 1) = real(2.0);
  A(1, 2) = real(3.0);
  auto AB = ToBlockSparseMatrix<1>(A, true);
  auto ABt = Transpose(AB);
  EXPECT_EQ(3, ABt.Rows());
  EXPECT_EQ(2, ABt.Cols());
  EXPECT_EQ(AB.NumNonZeroBlocks(), ABt.NumNonZeroBlocks());
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Transpose(A), ABt));
  EXPECT_TRUE(mochi::test::NearEqualMatrices(ABt, Transpose(AsView(AB))));
  EXPECT_TRUE(mochi::test::NearEqualMatrices(ABt, Transpose(AsConstView(AB))));
}

TEST(BlockSparseMatrix, TransposeEx2) {
  DynamicArray<int> pointers{0, 2, 3, 5};
  DynamicArray<int> cidx{0, 1, 3, 1, 2};
  constexpr int kBlockSize = 3;
  DynamicArray<real> values(kBlockSize * kBlockSize * isize(cidx));
  for (int i = 0; i < isize(values); ++i) {
    values[i] = real(i + 0.3);
  }
  BlockSparseMatrix<real, kBlockSize, int, int> A(4, pointers, cidx, values);
  auto At = Transpose(A);
  //
  EXPECT_EQ(A.BlockCols(), At.BlockRows());
  EXPECT_EQ(A.BlockRows(), At.BlockCols());
  EXPECT_EQ(A.Cols(), At.Rows());
  EXPECT_EQ(A.Rows(), At.Cols());
  EXPECT_EQ(A.NumNonZeroBlocks(), At.NumNonZeroBlocks());
  //
  auto Ad = ToMatrix(A);
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Transpose(Ad), At));
  EXPECT_TRUE(mochi::test::NearEqualMatrices(At, Transpose(AsView(A))));
  EXPECT_TRUE(mochi::test::NearEqualMatrices(At, Transpose(AsConstView(A))));
}

TEST(BlockSparseMatrix, Reset) {
  // Reset an owning BlockSparseMatrix
  {
    BlockSparseMatrix<real, 3> mat;
    ExpectEmpty(mat);

    // Reset from default to populated
    mat.Reset(MakeMat<TestCase_Block3_6x6>());
    CheckMat<TestCase_Block3_6x6>(mat);

    // Change dimensions and values
    mat.Reset(MakeMat<TestCase_Block3_3x6>());
    CheckMat<TestCase_Block3_3x6>(mat);

    // Change dimensions and values using a view
    mat.Reset(MakeMatConstView<TestCase_Block3_6x3>());
    CheckMat<TestCase_Block3_6x3>(mat);

    // Change dimensions using a different constructor
    DynamicArray<int> pointers = {0, 0, 0};
    mat.Reset(1, pointers, DynamicArray<int>{}, DynamicArray<real>{});
    EXPECT_EQ(6, mat.Rows());
    EXPECT_EQ(3, mat.Cols());
    EXPECT_EQ(0, mat.NumNonZeros());
  }

  // Reset a BlockSparseMatrixView
  {
    BlockSparseMatrixView<real const, 3> view;
    ExpectEmpty(view);

    // Reset from owning matrix
    auto mat = MakeMat<TestCase_Block3_6x6>();
    view.Reset(mat);
    EXPECT_EQ(mat.Values().data(), view.Values().data()); // same address
    CheckMat<TestCase_Block3_6x6>(view);

    // Reset from const view
    auto mat2 = MakeMat<TestCase_Block3_3x6>();
    view.Reset(AsConstView(mat2));
    EXPECT_EQ(mat2.Values().data(), view.Values().data()); // same address
    CheckMat<TestCase_Block3_3x6>(view);

    // Reset using a different constructor
    mat = MakeMat<TestCase_Block3_6x6>();
    view.Reset(mat.BlockCols(), mat.Pointers(), mat.Indices(), mat.Values());
    EXPECT_EQ(mat.Values().data(), view.Values().data()); // same address
    CheckMat<TestCase_Block3_6x6>(view);
  }
}

TEST(BlockSparseMatrix, IndicesAndValuesPerBlockRow) {
  using TC = TestCase_Block3_6x6;
  auto mat = MakeMat<TC>();
  CheckPerBlockRowAccessors<TC>(mat);
  CheckPerBlockRowAccessors<TC>(AsView(mat));
  CheckPerBlockRowAccessors<TC>(AsConstView(mat));

  // Mutability: write through mutable view's Values(r_c)
  AsView(mat).Values(0)(0, 0) = 99_r;
  EXPECT_EQ(99_r, mat(0, 3));

  // Empty block row (TestCase_Block3_6x3: block row 0 has no entries)
  auto mat2 = MakeMat<TestCase_Block3_6x3>();
  EXPECT_TRUE(mat2.Indices(0).empty());
  EXPECT_EQ(0, mat2.Values(0).NumBlocks());
}
