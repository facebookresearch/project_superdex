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

#include <mochi_core/linear_algebra/actor_pseudo_matrix.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/solvers/interaction_matrix_info.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>

#include <algorithm>
#include <iterator>
#include <vector>

using namespace mochi;
using namespace mochi::test;

TEST(ActorPseudoMatrix, Empty) {
  ActorPseudoMatrix<real> ops{0, {}, {}};
  EXPECT_EQ(0, ops.Rows());
  EXPECT_EQ(0, ops.Cols());
}

TEST(ActorPseudoMatrix, DenseActor) {
  // |123|
  // |245|
  // |356|
  real constexpr kValues[9] = {1, 2, 3, 2, 4, 5, 3, 5, 6};
  auto actorMat = Matrix<real>{3, 3};
  std::copy(std::begin(kValues), std::end(kValues), actorMat.data());

  ActorPseudoMatrix<real> ops{0, actorMat, {}};
  EXPECT_EQ(3, ops.Rows());
  EXPECT_EQ(3, ops.Cols());
  EXPECT_TRUE(NearEqualMatrices(actorMat, ToMatrix(ops)));
}

TEST(ActorPseudoMatrix, SparseActor) {
  // |1 3|
  // | 45|
  // |356|
  auto actorMat = SparseMatrix<real>{
      3,
      DynamicArray<int>{0, 2, 4, 7},
      DynamicArray<int>{0, 2, 1, 2, 0, 1, 2},
      DynamicArray<real>{1, 3, 4, 5, 3, 5, 6}};
  auto actorMatAsBlockable = SparseMatrix<real>{
      3,
      DynamicArray<int>{0, 3, 6, 9},
      DynamicArray<int>{0, 1, 2, 0, 1, 2, 0, 1, 2},
      DynamicArray<real>{1, 0, 3, 0, 4, 5, 3, 5, 6}};

  ActorPseudoMatrix<real> ops1{0, actorMat, {}};
  EXPECT_EQ(3, ops1.Rows());
  EXPECT_EQ(3, ops1.Cols());
  EXPECT_TRUE(NearEqualMatrices(actorMat, ToMatrix(ops1)));
}

TEST(ActorPseudoMatrix, BlockSparseActor) {
  // |123   |
  // |245   |
  // |356   |
  // |   234|
  // |   356|
  // |   467|
  // clang-format off
    auto actorMat = BlockSparseMatrix<real, 3>{
            2,
            DynamicArray<int>{0, 1, 2},
            DynamicArray<int>{0, 1},
            DynamicArray<real>{1, 2, 3, 2, 4, 5, 3, 5, 6, 2, 3, 4, 3, 5, 6, 4, 6, 7}};
  // clang-format on

  ActorPseudoMatrix<real> ops{0, actorMat, {}};
  EXPECT_EQ(6, ops.Rows());
  EXPECT_EQ(6, ops.Cols());
  EXPECT_TRUE(NearEqualMatrices(actorMat, ToMatrix(ops)));
}

TEST(ActorPseudoMatrix, BlockSparseActorToBlockSparseMatrix) {
  // |123   |
  // |245   |
  // |356   |
  // |   234|
  // |   356|
  // |   467|
  // clang-format off
    auto actorMat = BlockSparseMatrix<real, 3>{
            2,
            DynamicArray<int>{0, 1, 2},
            DynamicArray<int>{0, 1},
            DynamicArray<real>{1, 2, 3, 2, 4, 5, 3, 5, 6, 2, 3, 4, 3, 5, 6, 4, 6, 7}};
  // clang-format on

  ActorPseudoMatrix<real> ops{0, actorMat, {}};
  EXPECT_EQ(6, ops.Rows());
  EXPECT_EQ(6, ops.Cols());
  auto bsrMatrix = ToBlockSparseMatrix<3>(ops);
  EXPECT_EQ(actorMat.Rows(), bsrMatrix.Rows());
  EXPECT_EQ(actorMat.BlockRows(), bsrMatrix.BlockRows());
  EXPECT_EQ(actorMat.Cols(), bsrMatrix.Cols());
  EXPECT_EQ(actorMat.NumNonZeroBlocks(), bsrMatrix.NumNonZeroBlocks());
  //
  EXPECT_NEAR_EQ(actorMat.NormSqr(), bsrMatrix.NormSqr());
  //
  auto const nnz = actorMat.NumNonZeroBlocks() * 9;
  EXPECT_TRUE(NearEqualMatrices(
      ColumnVectorView<real const>(actorMat.Values().data(), nnz),
      ColumnVectorView<real const>(bsrMatrix.Values().data(), nnz)));
}

namespace {

void TestBlockDiagonal(ActorPseudoMatrix<real> const& ops, RowMatrix<real, 24, 24> const& ref) {
  EXPECT_EQ(12, ops.Rows());
  EXPECT_EQ(12, ops.Cols());
  EXPECT_TRUE(
      NearEqualMatrices(ToMatrix(ops), ref.Block(ops.offset, ops.offset, ops.Rows(), ops.Cols())));

  for (int i = 0; i < ops.Rows(); ++i) {
    for (int len = 1; len + i <= ops.Rows(); ++len) {
      auto Dc =
          krylov::GetBlockDiagonal<krylov::kDynamic, krylov::Direction::ColMajor>(ops, i, len);
      auto refD = krylov::GetBlockDiagonal<krylov::kDynamic, krylov::Direction::ColMajor>(
          ref, i + ops.offset, len);
      EXPECT_TRUE(NearEqualMatrices(Dc, refD));
      auto Dr =
          krylov::GetBlockDiagonal<krylov::kDynamic, krylov::Direction::RowMajor>(ops, i, len);
      EXPECT_TRUE(NearEqualMatrices(Dr, refD));
      if (len == 3) {
        auto D3c = krylov::GetBlockDiagonal<3, krylov::Direction::ColMajor>(ops, i, len);
        EXPECT_TRUE(NearEqualMatrices(D3c, refD));
        auto D3r = krylov::GetBlockDiagonal<3, krylov::Direction::RowMajor>(ops, i, len);
        EXPECT_TRUE(NearEqualMatrices(D3r, refD));
      }
    }
  }

  std::vector<real> dref(ref.Rows());
  auto dspan = MakeSpan(dref);
  krylov::ExtractDiagonal(ref, dspan);

  std::vector<real> odiag(Min(ops.Rows(), ops.Cols()));
  krylov::ExtractDiagonal(ops, MakeSpan(odiag));

  EXPECT_TRUE(NearEqualSpan(MakeSpan(odiag), dspan.subspan(ops.offset, odiag.size())));
}
} // namespace

TEST(ActorPseudoMatrix, ActorWithContact) {
  // Actor C: Sparse 12x12
  // |1 4         |
  // | 2          |
  // |4 3         |
  // |   589123456|
  // |   861246824|
  // |   917135791|
  // |   12182    |
  // |   243293   |
  // |   365 31   |
  // |   487   256|
  // |   529   537|
  // |   641   674|
  auto actorC_sparse = SparseMatrix<real>{
      12,
      DynamicArray<int>{0, 2, 3, 5, 14, 23, 32, 37, 43, 48, 54, 60, 66},
      DynamicArray<int>{0, 2, //
                        1, //
                        0, 2, //
                        3, 4, 5, 6, 7,  8,  9, 10, 11, //
                        3, 4, 5, 6, 7,  8,  9, 10, 11, //
                        3, 4, 5, 6, 7,  8,  9, 10, 11, //
                        3, 4, 5, 6, 7, //
                        3, 4, 5, 6, 7,  8, //
                        3, 4, 5, 7, 8, //
                        3, 4, 5, 9, 10, 11, //
                        3, 4, 5, 9, 10, 11, //
                        3, 4, 5, 9, 10, 11},
      DynamicArray<real>{1, 4, //
                         2, //
                         4, 3, //
                         5, 8, 9, 1, 2, 3, 4, 5, 6, //
                         8, 6, 1, 2, 4, 6, 8, 2, 4, //
                         9, 1, 7, 1, 3, 5, 7, 9, 1, //
                         1, 2, 1, 8, 2, //
                         2, 4, 3, 2, 9, 3, //
                         3, 6, 5, 3, 1, //
                         4, 8, 7, 2, 5, 6, //
                         5, 2, 9, 5, 3, 7, //
                         6, 4, 1, 6, 7, 4}};

  // Global system - 3 actors: A (3 x 3), B (9 x 9) and C (12 x 12)
  // Contact: Sparse 21 x 21 with 3 row/col offset
  // |2        987369      |
  // | 3       852471      |
  // |  4      741582      |
  // |                     |
  // |                     |
  // |                     |
  // |      5        147   |
  // |       6       258   |
  // |        7      369   |
  // |987                  |
  // |854                  |
  // |721                  |
  // |345                  |
  // |678                  |
  // |912                  |
  // |      123            |
  // |      456            |
  // |      789            |
  // |                  1 3|
  // |                   5 |
  // |                  2 8|
  auto contact0_sparseBlockable = SparseMatrix<real>{
      21,
      DynamicArray<int>{0,  9,  18, 27, 27, 27, 27, 33, 39, 45, 48,
                        51, 54, 57, 60, 63, 66, 69, 72, 74, 75, 77},
      DynamicArray<int>{0, 1, 2, 9,  10, 11, 12, 13, 14, //
                        0, 1, 2, 9,  10, 11, 12, 13, 14, //
                        0, 1, 2, 9,  10, 11, 12, 13, 14, //
                        6, 7, 8, 15, 16, 17, //
                        6, 7, 8, 15, 16, 17, //
                        6, 7, 8, 15, 16, 17, //
                        0, 1, 2, //
                        0, 1, 2, //
                        0, 1, 2, //
                        0, 1, 2, //
                        0, 1, 2, //
                        0, 1, 2, //
                        6, 7, 8, //
                        6, 7, 8, //
                        6, 7, 8, 18, 20, 19, 18, 20},
      DynamicArray<real>{2, 0, 0, 9, 8, 7, 3, 6, 9, //
                         0, 3, 0, 8, 5, 2, 4, 7, 1, //
                         0, 0, 4, 7, 4, 1, 5, 8, 2, //
                         5, 0, 0, 1, 4, 7, //
                         0, 6, 0, 2, 5, 8, //
                         0, 0, 7, 3, 6, 9, //
                         9, 8, 7, //
                         8, 5, 4, //
                         7, 2, 1, //
                         3, 4, 5, //
                         6, 7, 8, //
                         9, 1, 2, //
                         1, 2, 3, //
                         4, 5, 6, //
                         7, 8, 9, 1, 3, 5, 2, 8}};

  // The first 3x3 diagonal block of 'contact0' will not be updated by 'contact1'.
  // The remaining entries are scaled by 'kAlpha0' in 'contact0' and by '1 - kAlpha0' in 'contact1'.
  real constexpr kAlpha0 = 0.123456789_r;
  auto values = contact0_sparseBlockable.Values();
  for (int i = 0; i < isize(values); ++i) {
    if ((i != 0) && (i != 10) && (i != 20)) {
      values[i] *= kAlpha0;
    }
  }

  auto contact1_sparseBlockable = SparseMatrix<real>{
      21,
      DynamicArray<int>{0,  6,  12, 18, 18, 18, 18, 24, 30, 36, 39,
                        42, 45, 48, 51, 54, 57, 60, 63, 65, 66, 68},
      DynamicArray<int>{9, 10, 11, 12, 13, 14, //
                        9, 10, 11, 12, 13, 14, //
                        9, 10, 11, 12, 13, 14, //
                        6, 7,  8,  15, 16, 17, //
                        6, 7,  8,  15, 16, 17, //
                        6, 7,  8,  15, 16, 17, //
                        0, 1,  2, //
                        0, 1,  2, //
                        0, 1,  2, //
                        0, 1,  2, //
                        0, 1,  2, //
                        0, 1,  2, //
                        6, 7,  8, //
                        6, 7,  8, //
                        6, 7,  8,  18, 20, 19, 18, 20},
      DynamicArray<real>{9, 8, 7, 3, 6, 9, //
                         8, 5, 2, 4, 7, 1, //
                         7, 4, 1, 5, 8, 2, //
                         5, 0, 0, 1, 4, 7, //
                         0, 6, 0, 2, 5, 8, //
                         0, 0, 7, 3, 6, 9, //
                         9, 8, 7, //
                         8, 5, 4, //
                         7, 2, 1, //
                         3, 4, 5, //
                         6, 7, 8, //
                         9, 1, 2, //
                         1, 2, 3, //
                         4, 5, 6, //
                         7, 8, 9, 1, 3, 5, 2, 8}};

  // Scale contact values with the "complement"
  auto constexpr kAlpha1 = 1_r - kAlpha0;
  for (auto& val : contact1_sparseBlockable.Values()) {
    val *= kAlpha1;
  }

  RowMatrix<real, 24, 24> ref{
      {1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 9, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 4, 5, 6, 7, 8, 9, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 5, 6, 7, 8, 9, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 6, 7, 8, 9, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 7, 8, 9, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 8, 9, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 8, 9, 1, 2, 3, 4, 5, 6},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 6, 1, 2, 4, 6, 8, 2, 4},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 1, 7, 1, 3, 5, 7, 9, 1},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 8, 2, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 3, 2, 9, 3, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 5, 0, 3, 1, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 7, 0, 0, 0, 3, 5, 9},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 2, 9, 0, 0, 0, 5, 8, 7},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 4, 1, 0, 0, 0, 8, 7, 12}};

  auto actorC_blockSparse = ToBlockSparseMatrix<3>(actorC_sparse);
  auto actorC_matrix = ToMatrix(actorC_sparse);
  AnyMatrixView<real const> actor[3] = {
      AsConstView(actorC_blockSparse), AsConstView(actorC_sparse), AsConstView(actorC_matrix)};

  auto contact0_blockSparse = ToBlockSparseMatrix<3>(contact0_sparseBlockable);
  auto contact1_blockSparse = ToBlockSparseMatrix<3>(contact1_sparseBlockable);

  auto contact0_matrix = ToMatrix(contact0_sparseBlockable);
  auto contact1_matrix = ToMatrix(contact1_sparseBlockable);

  AnyInteractionMatrixViewInfo<real const> contactData0[3] = {
      {3, 3, contact0_blockSparse, /*symmetricPair*/ std::nullopt},
      {3, 3, contact0_sparseBlockable, /*symmetricPair*/ std::nullopt},
      {3, 3, contact0_matrix, /*symmetricPair*/ std::nullopt}};
  AnyInteractionMatrixViewInfo<real const> contactData1[3] = {
      {3, 3, contact1_blockSparse, /*symmetricPair*/ std::nullopt},
      {3, 3, contact1_sparseBlockable, /*symmetricPair*/ std::nullopt},
      {3, 3, contact1_matrix, /*symmetricPair*/ std::nullopt}};

  for (auto const& actorC : actor) {
    for (auto const& contact0 : contactData0) {
      for (auto const& contact1 : contactData1) {
        ActorPseudoMatrix<real> ops{12, actorC, {contact0, contact1}};
        TestBlockDiagonal(ops, ref);
      }
    }
  }
}

TEST(ActorPseudoMatrix, ActorBlockSparseWithContactBlockSparseToBlockSparseMatrix) {
  auto actorBsr = BlockSparseMatrix<real, 3>{
      4,
      DynamicArray<int>{0, 2, 5, 8, 10},
      // clang-format off
      DynamicArray<int>{
          0, 1, //
          0, 1, 2, //
          1, 2, 3, //
          0, 3},
      // clang-format on
      DynamicArray<real>(90, 1_r)};

  auto contact0 = BlockSparseMatrix<real, 3>{
      1, DynamicArray<int>{0, 1}, DynamicArray<int>{0}, DynamicArray<real>(9, 2_r)};

  auto contact1 = BlockSparseMatrix<real, 3>{
      2, DynamicArray<int>{0, 1, 3}, DynamicArray<int>{0, 0, 1}, DynamicArray<real>(27, 3_r)};

  int const actorSize = actorBsr.Rows();
  int const actorNNZB = actorBsr.NumNonZeroBlocks();

  {
    ActorPseudoMatrix<real> ops{
        0,
        AsConstView(actorBsr),
        {{20, 20, contact0, std::nullopt}, {13, 13, contact1, std::nullopt}}};
    auto bsrMatrix = ToBlockSparseMatrix<3>(ops);
    EXPECT_EQ(actorSize, bsrMatrix.Rows());
    EXPECT_EQ(actorNNZB, bsrMatrix.NumNonZeroBlocks());
    auto mat1 = ToMatrix(bsrMatrix);
    auto mat2 = ToMatrix(actorBsr);
    EXPECT_TRUE(NearEqualMatrices(mat1, mat2));
  }

  {
    int const offsetC0 = 3;
    int const offsetC1 = 9;
    ActorPseudoMatrix<real> ops{
        0,
        AsConstView(actorBsr),
        {{offsetC0, offsetC0, contact0, /*symmetricPair*/ std::nullopt},
         {offsetC1, offsetC1, contact1, /*symmetricPair*/ std::nullopt}}};
    auto bsrMatrix = ToBlockSparseMatrix<3>(ops);
    EXPECT_EQ(actorSize, bsrMatrix.Rows());
    EXPECT_EQ(actorNNZB, bsrMatrix.NumNonZeroBlocks());
    auto mat1 = ToMatrix(bsrMatrix);
    auto mat2 = ToMatrix(actorBsr);
    for (int ii = 0; ii < 3; ++ii) {
      for (int jj = 0; jj < 3; ++jj) {
        mat2(offsetC0 + ii, offsetC0 + jj) += 2_r;
      }
    }
    //
    for (int ii = 0; ii < 3; ++ii) {
      for (int jj = 0; jj < 3; ++jj) {
        mat2(offsetC1 + ii, offsetC1 + jj) += 3_r;
      }
    }
    EXPECT_TRUE(NearEqualMatrices(mat1, mat2));
  }

  {
    int const offsetA = 5;
    int const offsetC0 = 9 + offsetA;
    int const offsetC1 = 0 + offsetA;
    ActorPseudoMatrix<real> ops{
        offsetA,
        AsConstView(actorBsr),
        {{offsetC0, offsetC0, contact0, /*symmetricPair*/ std::nullopt},
         {offsetC1, offsetC1, contact1, /*symmetricPair*/ std::nullopt}}};
    auto bsrMatrix = ToBlockSparseMatrix<3>(ops);
    EXPECT_EQ(actorSize, bsrMatrix.Rows());
    EXPECT_EQ(actorNNZB, bsrMatrix.NumNonZeroBlocks());
    auto mat1 = ToMatrix(bsrMatrix);
    auto mat2 = ToMatrix(actorBsr);
    for (int ii = 0; ii < 3; ++ii) {
      for (int jj = 0; jj < 3; ++jj) {
        mat2(offsetC0 - offsetA + ii, offsetC0 - offsetA + jj) += 2_r;
      }
    }
    //
    for (int ii = 0; ii < 3; ++ii) {
      for (int jj = 0; jj < 3; ++jj) {
        mat2(offsetC1 - offsetA + ii, offsetC1 - offsetA + jj) += 3_r;
        mat2(3 + offsetC1 - offsetA + ii, offsetC1 - offsetA + jj) += 3_r;
        mat2(3 + offsetC1 - offsetA + ii, 3 + offsetC1 - offsetA + jj) += 3_r;
      }
    }
    EXPECT_TRUE(NearEqualMatrices(mat1, mat2));
  }
}

template <int kBlockSize, typename InteractionMatrixT>
static void ExpectOffDiagonalInteractionOutsideActorDoesNotChangeBlockSparseActor(
    InteractionMatrixT const& interactionMatrix) {
  int constexpr kNumBlocks = 3;
  int constexpr kActorDofs = kBlockSize * kNumBlocks;
  int constexpr kNumOffsetBlocks = 1;
  int constexpr kActorOffset = kBlockSize * kNumOffsetBlocks;
  int constexpr kOtherActorOffset = kActorOffset + kActorDofs;

  DynamicArray<int> pointers;
  DynamicArray<int> indices;
  pointers.reserve(kNumBlocks + 1);
  indices.reserve(kNumBlocks * kNumBlocks);
  pointers.push_back(0);
  for (int r = 0; r < kNumBlocks; ++r) {
    for (int c = 0; c < kNumBlocks; ++c) {
      indices.push_back(c);
    }
    pointers.push_back(isize(indices));
  }
  DynamicArray<real> values(kNumBlocks * kNumBlocks * kBlockSize * kBlockSize);
  for (int i = 0; i < isize(values); ++i) {
    values[i] = static_cast<real>(i + 1);
  }
  BlockSparseMatrix<real, kBlockSize> actorBsr{
      kNumBlocks, std::move(pointers), std::move(indices), std::move(values)};

  auto const expected = ToMatrix(actorBsr);
  ActorPseudoMatrix<real> rowsOverlapOnly{
      kActorOffset,
      AsConstView(actorBsr),
      {{kActorOffset, kOtherActorOffset, interactionMatrix, /*symmetricPair*/ std::nullopt}}};
  ActorPseudoMatrix<real> colsOverlapOnly{
      kActorOffset,
      AsConstView(actorBsr),
      {{kOtherActorOffset, kActorOffset, interactionMatrix, /*symmetricPair*/ std::nullopt}}};

  real constexpr kExactTolerance = 0_r;
  EXPECT_TRUE(NearEqualMatrices(
      expected, ToMatrix(ToBlockSparseMatrix<kBlockSize>(rowsOverlapOnly)), kExactTolerance));
  EXPECT_TRUE(NearEqualMatrices(
      expected, ToMatrix(ToBlockSparseMatrix<kBlockSize>(colsOverlapOnly)), kExactTolerance));
}

template <int kBlockSize>
static void ExpectOffDiagonalInteractionsOutsideActorDoNotChangeBlockSparseActor() {
  RowMatrix<real> denseInteraction(2 * kBlockSize, 2 * kBlockSize);
  denseInteraction.SetRandom(10 * kBlockSize);
  auto sparseInteraction = ToSparseMatrix(denseInteraction, true);
  auto blockSparseInteraction = ToBlockSparseMatrix<kBlockSize>(sparseInteraction);
  ExpectOffDiagonalInteractionOutsideActorDoesNotChangeBlockSparseActor<kBlockSize>(
      sparseInteraction);
  ExpectOffDiagonalInteractionOutsideActorDoesNotChangeBlockSparseActor<kBlockSize>(
      blockSparseInteraction);
}

TEST(ActorPseudoMatrix, OffDiagonalInteractionOutsideActorDoesNotChangeBlockSparseActor) {
  ExpectOffDiagonalInteractionsOutsideActorDoNotChangeBlockSparseActor<3>();
  ExpectOffDiagonalInteractionsOutsideActorDoNotChangeBlockSparseActor<4>();
}

TEST(ActorPseudoMatrix, ActorBlockSparseWithContactSparseToBlockSparseMatrix) {
  auto actorBsr = BlockSparseMatrix<real, 3>{
      3,
      DynamicArray<int>{0, 2, 5, 7},
      // clang-format off
      DynamicArray<int>{
          0, 1, //
          0, 1, 2, //
          1, 2},
      // clang-format on
      DynamicArray<real>(63, 1_r)};

  // Blockable sparse matrix: column indices are in consecutive groups of 3.
  // clang-format off
  auto contactSparse = SparseMatrix<real>{
      3,
      DynamicArray<int>{0, 3, 6, 9},
      DynamicArray<int>{
          0, 1, 2, //
          0, 1, 2, //
          0, 1, 2},
      DynamicArray<real>{
          2_r, 2_r, 0_r, //
          2_r, 2_r, 2_r, //
          0_r, 2_r, 2_r}};
  // clang-format on

  int const actorSize = 9;
  int const actorNNZB = 7;
  for (auto aOffset : {0, 11, 23, 25}) {
    ActorPseudoMatrix<real> ops{
        aOffset, AsConstView(actorBsr), {{20, 20, contactSparse, std::nullopt}}};
    auto bsrMatrix = ToBlockSparseMatrix<3>(ops);
    EXPECT_EQ(actorSize, bsrMatrix.Rows());
    EXPECT_EQ(actorNNZB, bsrMatrix.NumNonZeroBlocks());
    auto mat1 = ToMatrix(bsrMatrix);
    auto mat2 = ToMatrix(actorBsr);
    EXPECT_TRUE(NearEqualMatrices(mat1, mat2));
  }

  {
    auto const aOffset = 2;
    for (auto cOffset : {2, 5}) {
      auto const diffOffset = cOffset - aOffset;
      ActorPseudoMatrix<real> ops{
          aOffset, AsConstView(actorBsr), {{cOffset, cOffset, contactSparse, std::nullopt}}};
      auto bsrMatrix = ToBlockSparseMatrix<3>(ops);
      EXPECT_EQ(actorSize, bsrMatrix.Rows());
      EXPECT_EQ(actorNNZB, bsrMatrix.NumNonZeroBlocks());
      auto mat1 = ToMatrix(bsrMatrix);
      auto mat2 = ToMatrix(actorBsr);
      mat2(0 + diffOffset, 0 + diffOffset) += 2;
      mat2(0 + diffOffset, 1 + diffOffset) += 2;
      mat2(1 + diffOffset, 0 + diffOffset) += 2;
      mat2(1 + diffOffset, 1 + diffOffset) += 2;
      if (2 + diffOffset < bsrMatrix.Rows()) {
        mat2(1 + diffOffset, 2 + diffOffset) += 2;
        mat2(2 + diffOffset, 1 + diffOffset) += 2;
        mat2(2 + diffOffset, 2 + diffOffset) += 2;
      }
      EXPECT_TRUE(NearEqualMatrices(mat1, mat2));
    }
  }

  // Non-blockable sparse matrix: column indices are NOT in consecutive groups of 3.
  // clang-format off
  auto contactSparseNonblockable = SparseMatrix<real>{
      6,
      DynamicArray<int>{0, 3, 5, 8, 11, 13, 17},
      DynamicArray<int>{
          0, 2, 5, // row 0: blocks 0, 0, 1
          1, 4, // row 1: blocks 0, 1
          0, 3, 5, // row 2: blocks 0, 1, 1
          2, 3, 4, // row 3: blocks 0, 1, 1 (consecutive in block 1)
          0, 5, // row 4: blocks 0, 1 (skipping columns)
          1, 2, 3, 4, // row 5: blocks 0, 0, 1, 1 (transition)
      },
      DynamicArray<real>{
          2_r, 3_r, 4_r, // row 0 values
          5_r, 6_r, // row 1 values
          7_r, 8_r, 9_r, // row 2 values
          10_r, 11_r, 12_r, // row 3 values
          13_r, 14_r, // row 4 values
          15_r, 16_r, 17_r, 18_r // row 5 values
      }};
  // clang-format on

  // Use a larger actor matrix that can accommodate the 6x6 non-blockable contact matrix.
  auto actorBsrLarge = BlockSparseMatrix<real, 3>{
      4,
      DynamicArray<int>{0, 4, 8, 12, 16},
      // clang-format off
      DynamicArray<int>{
          0, 1, 2, 3,
          0, 1, 2, 3,
          0, 1, 2, 3,
          0, 1, 2, 3,
      },
      // clang-format on
      DynamicArray<real>(16 * 9, 1_r)};

  // Test with various offsets to exercise all codepaths.
  for (auto aOffset : {0, 3, 4, 5, 6}) {
    for (auto cOffset :
         {aOffset - 1, aOffset - 2, aOffset, aOffset + 1, aOffset + 2, aOffset + 3}) {
      if (cOffset < 0) {
        continue;
      }
      ActorPseudoMatrix<real> actorPseudoMat{
          aOffset,
          AsConstView(actorBsrLarge),
          {{cOffset, cOffset, contactSparseNonblockable, std::nullopt}}};
      auto bsrMatrix = ToBlockSparseMatrix<3>(actorPseudoMat);
      EXPECT_EQ(actorBsrLarge.Rows(), bsrMatrix.Rows());
      EXPECT_EQ(actorBsrLarge.Cols(), bsrMatrix.Cols());
      EXPECT_EQ(actorBsrLarge.NumNonZeroBlocks(), bsrMatrix.NumNonZeroBlocks());

      // Verify against ToMatrix(ActorPseudoMatrix) which uses a different codepath.
      EXPECT_TRUE(NearEqualMatrices(ToMatrix(bsrMatrix), ToMatrix(actorPseudoMat)));
    }
  }
}

TEST(ActorPseudoMatrix, MultipleInteractionMatrices) {
#define MOCHI_MULT_INTER_MAT_TEST_HELPER(j)                                       \
  auto D2c = krylov::GetBlockDiagonal<j, krylov::Direction::ColMajor>(apm, i, j); \
  auto D2r = krylov::GetBlockDiagonal<j, krylov::Direction::RowMajor>(apm, i, j); \
  EXPECT_TRUE(NearEqualMatrices(D2c, refD));                                      \
  EXPECT_TRUE(NearEqualMatrices(D2r, refD));

  auto runTests = [](int aOffset, int bOffset, int cOffset, int actorOffset, int actorDofs) {
    std::vector<AnyInteractionMatrixViewInfo<real const>> interactionMatrices;

    Matrix<real> actorMatrix(actorDofs, actorDofs);
    actorMatrix.SetRandom(aOffset + bOffset + cOffset + actorOffset + actorDofs);

    Matrix<real> ref = actorMatrix;
    auto addToRef = [&](auto const& I, int iOffset) {
      int rowBegin = Max(iOffset, actorOffset);
      int rowEnd = Min(iOffset + I.Rows(), actorOffset + actorDofs);
      int len = rowEnd - rowBegin;
      if (len > 0) {
        ref.Block(rowBegin - actorOffset, rowBegin - actorOffset, len, len) +=
            I.Block(rowBegin - iOffset, rowBegin - iOffset, len, len);
      }
    };

    // Matrix A: 4x4 sparse
    // |    |
    // | 456|
    // | 789|
    // |1 2 |
    auto aSparse =
        SparseMatrix<real>{4, {0, 0, 3, 6, 8}, {1, 2, 3, 1, 2, 3, 0, 2}, {4, 5, 6, 7, 8, 9, 1, 2}};
    AnyMatrix<real> aInteraction = aSparse;
    interactionMatrices.emplace_back(
        aOffset, aOffset, AsConstView(aInteraction), /*symmetricPair*/ std::nullopt);
    addToRef(ToMatrix(aSparse), aOffset);

    // Matrix B: 3x3 dense
    Matrix<real> bDense = {{3_r, 4_r, 6_r}, {5_r, 0_r, -1_r}, {2_r, 3_r, 4_r}};
    AnyMatrix<real> bInteraction = bDense;
    interactionMatrices.emplace_back(
        bOffset, bOffset, AsConstView(bInteraction), /*symmetricPair*/ std::nullopt);
    addToRef(bDense, bOffset);

    // Matrix C: 3x3 block sparse
    auto cDense = RowMatrix<real>{{1, 2, 8}, {2, 4, 7}, {3, 5, 6}};
    AnyMatrix<real> cInteraction = ToBlockSparseMatrix<3>(cDense);
    interactionMatrices.emplace_back(
        cOffset, cOffset, AsConstView(cInteraction), /*symmetricPair*/ std::nullopt);
    addToRef(cDense, cOffset);

    ActorPseudoMatrix<real> apm{actorOffset, AsConstView(actorMatrix), interactionMatrices};

    EXPECT_EQ(ref.Rows(), apm.Rows());
    EXPECT_EQ(ref.Cols(), apm.Cols());
    EXPECT_TRUE(NearEqualMatrices(ref, ToMatrix(apm)));

    for (int i = 0; i < apm.Rows(); ++i) {
      for (int j = 1; j + i <= apm.Rows(); ++j) {
        auto refD =
            krylov::GetBlockDiagonal<krylov::kDynamic, krylov::Direction::ColMajor>(ref, i, j);
        auto Dc =
            krylov::GetBlockDiagonal<krylov::kDynamic, krylov::Direction::ColMajor>(apm, i, j);
        auto Dr =
            krylov::GetBlockDiagonal<krylov::kDynamic, krylov::Direction::RowMajor>(apm, i, j);
        EXPECT_TRUE(NearEqualMatrices(Dc, refD));
        EXPECT_TRUE(NearEqualMatrices(Dr, refD));

        if (j == 7) {
          MOCHI_MULT_INTER_MAT_TEST_HELPER(7);
        } else if (j == 6) {
          MOCHI_MULT_INTER_MAT_TEST_HELPER(6);
        } else if (j == 5) {
          MOCHI_MULT_INTER_MAT_TEST_HELPER(5);
        } else if (j == 4) {
          MOCHI_MULT_INTER_MAT_TEST_HELPER(4);
        } else if (j == 3) {
          MOCHI_MULT_INTER_MAT_TEST_HELPER(3);
        } else if (j == 2) {
          MOCHI_MULT_INTER_MAT_TEST_HELPER(2);
        } else if (j == 1) {
          MOCHI_MULT_INTER_MAT_TEST_HELPER(1);
        }
      }
    }

    std::vector<real> refDiag(ref.Rows());
    krylov::ExtractDiagonal(ref, MakeSpan(refDiag));

    std::vector<real> apmDiag(Min(apm.Rows(), apm.Cols()));
    krylov::ExtractDiagonal(apm, MakeSpan(apmDiag));

    EXPECT_TRUE(NearEqualSpan(apmDiag, refDiag));
  };

#undef MOCHI_MULT_INTER_MAT_TEST_HELPER

  for (auto aOffset : {0, 1, 2, 3, 5, 10}) {
    for (auto bOffset : {0, 1, 2, 3, 5, 10}) {
      for (auto cOffset : {0, 1, 2, 3, 5, 10}) {
        for (auto actorOffset : {0, 1, 2, 3, 4}) {
          for (auto actorDofs : {1, 3, 7}) {
            runTests(aOffset, bOffset, cOffset, actorOffset, actorDofs);
          }
        }
      }
    }
  }
}

// TODO: Test all other methods
