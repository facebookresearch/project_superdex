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
#include <mochi_core/linear_algebra/low_rank_augmented_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/solvers/island_operators.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <utility>
#include <variant>
#include <vector>

using namespace mochi;
using namespace mochi::test;

TEST(IslandOperators, kIsIslandOperators) {
  static_assert(!IsIslandOperators<real>);
  static_assert(!IsIslandOperators<Matrix<real>>);
  static_assert(!IsIslandOperators<SparseMatrix<real>>);
  static_assert(!IsIslandOperators<BlockSparseMatrix<real, 3>>);
  static_assert(!IsIslandOperators<LowRankAugmentedMatrix<Matrix<real>>>);
  static_assert(IsIslandOperators<IslandOperators<real>>);
}

TEST(IslandOperators, kIsLinearOperator) {
  static_assert(IsLinearOperator<IslandOperators<real>>);
}

TEST(IslandOperators, Empty) {
  IslandOperators<real> ops({}, {}, {});
  EXPECT_EQ(0, ops.Rows());
  EXPECT_EQ(0, ops.Cols());
  EXPECT_TRUE(ops.IsBlockable<3>());

  auto spmat = ops.FullSparseMatrix();
  EXPECT_EQ(0, spmat.Rows());
  EXPECT_EQ(0, spmat.Cols());

  auto bspmat = ops.FullBlockSparseMatrix<3>();
  EXPECT_EQ(0, bspmat.Rows());
  EXPECT_EQ(0, bspmat.Cols());

  auto mat = ops.CondenseFullMatrix();
  EXPECT_EQ(0, GetNumRows(mat));
  EXPECT_EQ(0, GetNumCols(mat));
}

TEST(IslandOperators, SingleDenseActor) {
  // |123|
  // |245|
  // |356|
  real constexpr kValues[9] = {1, 2, 3, 2, 4, 5, 3, 5, 6};
  auto actorMat = Matrix<real>{3, 3};
  std::copy(std::begin(kValues), std::end(kValues), actorMat.data());
  std::vector<std::pair<int, AnyMatrixView<real const>>> actorMatrices;
  actorMatrices.emplace_back(0, AsConstView(actorMat));

  IslandOperators ops(actorMatrices, {}, {});
  EXPECT_EQ(3, ops.Rows());
  EXPECT_EQ(3, ops.Cols());
  EXPECT_TRUE(ops.IsBlockable<3>());
  EXPECT_TRUE(NearEqualMatrices(actorMat, ToMatrix(ops)));
  EXPECT_TRUE(NearEqualMatrices(actorMat, ops.FullSparseMatrix()));
  EXPECT_TRUE(NearEqualMatrices(actorMat, ops.FullBlockSparseMatrix<3>()));
  auto global = ops.CondenseFullMatrix();
  std::visit([&](auto const& A) { EXPECT_TRUE(NearEqualMatrices(actorMat, A)); }, global);

  // 'Apply' and 'ApplyToRange' methods.
  SparseMatrix<real> fullSp = ops.FullSparseMatrix();
  ColumnVector<real> x(ops.Rows()), y0(ops.Cols()), y1(ops.Cols());
  x.SetRandom(1);
  y0.SetRandom(2);
  y1.SetRandom(3);
  fullSp.Apply(x, y0);
  ops.Apply(x, y1);
  EXPECT_TRUE(NearEqualMatrices(y0, y1));

  x.SetRandom(4);
  int rowBegin = 1, rowEnd = 2;
  fullSp.ApplyToRange(x, y0, rowBegin, rowEnd);
  EXPECT_FALSE(NearEqualMatrices(y0, y1)); // Not same as before
  ops.ApplyToRange(x, y1, rowBegin, rowEnd);
  EXPECT_TRUE(NearEqualMatrices(y0, y1));
}

TEST(IslandOperators, SingleSparseActor) {
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
  std::vector<std::pair<int, AnyMatrixView<real const>>> actorMatrices;
  actorMatrices.emplace_back(0, AsConstView(actorMat));
  std::vector<std::pair<int, AnyMatrixView<real const>>> actorMatricesAsBlockable;
  actorMatricesAsBlockable.emplace_back(0, AsConstView(actorMatAsBlockable));

  IslandOperators ops1(actorMatrices, {}, {});
  EXPECT_EQ(3, ops1.Rows());
  EXPECT_EQ(3, ops1.Cols());
  EXPECT_FALSE(ops1.IsBlockable<3>());
  EXPECT_TRUE(NearEqualMatrices(actorMat, ToMatrix(ops1)));
  EXPECT_TRUE(NearEqualMatrices(actorMat, ops1.FullSparseMatrix()));
  EXPECT_TRUE(ops1.FullBlockSparseMatrix<3>().empty());
  auto global1 = ops1.CondenseFullMatrix();
  std::visit([&](auto const& A) { EXPECT_TRUE(NearEqualMatrices(actorMat, A)); }, global1);

  IslandOperators ops2(actorMatricesAsBlockable, {}, {});
  EXPECT_TRUE(ops2.IsBlockable<3>());
  EXPECT_TRUE(NearEqualMatrices(actorMat, ToMatrix(ops2)));
  EXPECT_TRUE(NearEqualMatrices(actorMat, ops2.FullSparseMatrix()));
  EXPECT_TRUE(NearEqualMatrices(actorMat, ops2.FullBlockSparseMatrix<3>()));
  auto global2 = ops2.CondenseFullMatrix();
  std::visit([&](auto const& A) { EXPECT_TRUE(NearEqualMatrices(actorMat, A)); }, global2);

  // 'Apply' and 'ApplyToRange' methods.
  SparseMatrix<real> fullSp = ops1.FullSparseMatrix();
  ColumnVector<real> x(ops1.Rows()), y0(ops1.Cols()), y1(ops1.Cols()), y2(ops1.Cols());
  x.SetRandom(1);
  y0.SetRandom(2);
  y1.SetRandom(3);
  y2.SetRandom(4);
  fullSp.Apply(x, y0);
  ops1.Apply(x, y1);
  ops2.Apply(x, y2);
  EXPECT_TRUE(NearEqualMatrices(y0, y1));
  EXPECT_TRUE(NearEqualMatrices(y0, y2));

  x.SetRandom(5);
  int rowBegin = 1, rowEnd = 2;
  fullSp.ApplyToRange(x, y0, rowBegin, rowEnd);
  EXPECT_FALSE(NearEqualMatrices(y0, y1)); // Not same as before
  ops1.ApplyToRange(x, y1, rowBegin, rowEnd);
  ops2.ApplyToRange(x, y2, rowBegin, rowEnd);
  EXPECT_TRUE(NearEqualMatrices(y0, y1));
  EXPECT_TRUE(NearEqualMatrices(y0, y2));
}

TEST(IslandOperators, SingleBlockSparseActor) {
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

  std::vector<std::pair<int, AnyMatrixView<real const>>> actorMatrices;
  actorMatrices.emplace_back(0, AsConstView(actorMat));

  IslandOperators ops(actorMatrices, {}, {});
  EXPECT_EQ(6, ops.Rows());
  EXPECT_EQ(6, ops.Cols());
  EXPECT_TRUE(ops.IsBlockable<3>());
  EXPECT_TRUE(NearEqualMatrices(actorMat, ToMatrix(ops)));
  EXPECT_TRUE(NearEqualMatrices(actorMat, ops.FullSparseMatrix()));
  EXPECT_TRUE(NearEqualMatrices(actorMat, ops.FullBlockSparseMatrix<3>()));
  auto global = ops.CondenseFullMatrix();
  std::visit([&](auto const& A) { EXPECT_TRUE(NearEqualMatrices(actorMat, A)); }, global);

  // 'Apply' and 'ApplyToRange' methods.
  SparseMatrix<real> fullSp = ops.FullSparseMatrix();
  ColumnVector<real> x(ops.Rows()), y0(ops.Cols()), y1(ops.Cols());
  x.SetRandom(1);
  y0.SetRandom(2);
  y1.SetRandom(3);
  fullSp.Apply(x, y0);
  ops.Apply(x, y1);
  EXPECT_TRUE(NearEqualMatrices(y0, y1));

  x.SetRandom(4);
  int rowBegin = 3, rowEnd = 6;
  fullSp.ApplyToRange(x, y0, rowBegin, rowEnd);
  EXPECT_FALSE(NearEqualMatrices(y0, y1)); // Not same as before
  ops.ApplyToRange(x, y1, rowBegin, rowEnd);
  EXPECT_TRUE(NearEqualMatrices(y0, y1));
}

TEST(IslandOperators, MixedActors) {
  // |123|
  // |245|
  // |356|
  real constexpr kValues[9] = {1, 2, 3, 2, 4, 5, 3, 5, 6};
  auto denseActorMat = Matrix<real>{3, 3};
  std::copy(std::begin(kValues), std::end(kValues), denseActorMat.data());

  // |1 3|
  // | 45|
  // |356|
  auto sparseActorMat = SparseMatrix<real>{
      3,
      DynamicArray<int>{0, 2, 4, 7},
      DynamicArray<int>{0, 2, 1, 2, 0, 1, 2},
      DynamicArray<real>{1, 3, 4, 5, 3, 5, 6}};
  auto sparseActorMatBlockable = SparseMatrix<real>{
      3,
      DynamicArray<int>{0, 3, 6, 9},
      DynamicArray<int>{0, 1, 2, 0, 1, 2, 0, 1, 2},
      DynamicArray<real>{1, 0, 3, 0, 4, 5, 3, 5, 6}};

  // |123   |
  // |245   |
  // |356   |
  // |   234|
  // |   356|
  // |   467|
  // clang-format off
  auto blockSparseActorMat = BlockSparseMatrix<real, 3>{
      2,
      DynamicArray<int>{0, 1, 2},
      DynamicArray<int>{0, 1},
      DynamicArray<real>{1, 2, 3, 2, 4, 5, 3, 5, 6, 2, 3, 4, 3, 5, 6, 4, 6, 7}};
  // clang-format on

  std::vector<std::pair<int, AnyMatrixView<real const>>> actorMatrices;
  int offset = 0;
  actorMatrices.emplace_back(offset, AsConstView(denseActorMat));
  offset += denseActorMat.Rows();
  actorMatrices.emplace_back(offset, AsConstView(sparseActorMat));
  offset += sparseActorMat.Rows();
  actorMatrices.emplace_back(offset, AsConstView(blockSparseActorMat));
  offset += blockSparseActorMat.Rows();
  IslandOperators ops1(actorMatrices, {}, {});

  std::vector<std::pair<int, AnyMatrixView<real const>>> actorMatricesBlockable;
  offset = 0;
  actorMatricesBlockable.emplace_back(offset, AsConstView(denseActorMat));
  offset += denseActorMat.Rows();
  actorMatricesBlockable.emplace_back(offset, AsConstView(sparseActorMatBlockable));
  offset += sparseActorMatBlockable.Rows();
  actorMatricesBlockable.emplace_back(offset, AsConstView(blockSparseActorMat));
  offset += blockSparseActorMat.Rows();
  IslandOperators ops2(actorMatricesBlockable, {}, {});

  // Full Matrix:
  // |123         |
  // |245         |
  // |356         |
  // |   1 3      |
  // |    45      |
  // |   356      |
  // |      123   |
  // |      245   |
  // |      356   |
  // |         234|
  // |         356|
  // |         467|

  EXPECT_EQ(12, ops1.Rows());
  EXPECT_EQ(12, ops1.Cols());
  EXPECT_EQ(12, ops2.Rows());
  EXPECT_EQ(12, ops2.Cols());
  EXPECT_FALSE(ops1.IsBlockable<3>());
  EXPECT_TRUE(ops2.IsBlockable<3>());

  real constexpr kFullValuesDense[12 * 12] = {1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
                                              2, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
                                              3, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
                                              0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, //
                                              0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 0, 0, //
                                              0, 0, 0, 3, 5, 6, 0, 0, 0, 0, 0, 0, //
                                              0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, //
                                              0, 0, 0, 0, 0, 0, 2, 4, 5, 0, 0, 0, //
                                              0, 0, 0, 0, 0, 0, 3, 5, 6, 0, 0, 0, //
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 4, //
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 5, 6, //
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 6, 7}; //
  MatrixView<real const, 12, 12, krylov::Direction::RowMajor> expectedFullValues(kFullValuesDense);

  EXPECT_TRUE(NearEqualMatrices(expectedFullValues, ToMatrix(ops1)));
  EXPECT_TRUE(NearEqualMatrices(expectedFullValues, ToMatrix(ops2)));

  EXPECT_TRUE(NearEqualMatrices(expectedFullValues, ops1.FullSparseMatrix()));
  EXPECT_TRUE(NearEqualMatrices(expectedFullValues, ops2.FullSparseMatrix()));

  EXPECT_TRUE(ops1.FullBlockSparseMatrix<3>().empty());
  EXPECT_TRUE(NearEqualMatrices(expectedFullValues, ops2.FullBlockSparseMatrix<3>()));

  auto global1 = ops1.CondenseFullMatrix();
  auto global2 = ops2.CondenseFullMatrix();
  std::visit(
      [&](auto const& A) { EXPECT_TRUE(NearEqualMatrices(expectedFullValues, A)); }, global1);
  std::visit(
      [&](auto const& A) { EXPECT_TRUE(NearEqualMatrices(expectedFullValues, A)); }, global2);

  // 'Apply' and 'ApplyToRange' methods.
  SparseMatrix<real> fullSp = ops1.FullSparseMatrix();
  ColumnVector<real> x(ops1.Rows()), y0(ops1.Cols()), y1(ops1.Cols()), y2(ops1.Cols());
  x.SetRandom(1);
  y0.SetRandom(2);
  y1.SetRandom(3);
  y2.SetRandom(4);
  fullSp.Apply(x, y0);
  ops1.Apply(x, y1);
  ops2.Apply(x, y2);
  EXPECT_TRUE(NearEqualMatrices(y0, y1));
  EXPECT_TRUE(NearEqualMatrices(y0, y2));

  x.SetRandom(5);
  int rowBegin = 0, rowEnd = 9;
  fullSp.ApplyToRange(x, y0, rowBegin, rowEnd);
  EXPECT_FALSE(NearEqualMatrices(y0, y1)); // Not same as before
  ops1.ApplyToRange(x, y1, rowBegin, rowEnd);
  ops2.ApplyToRange(x, y2, rowBegin, rowEnd);
  EXPECT_TRUE(NearEqualMatrices(y0, y1));
  EXPECT_TRUE(NearEqualMatrices(y0, y2));
}

TEST(IslandOperators, ActorsInContact) {
  // Actor A: 3x3
  // |1 3|
  // | 45|
  // |356|
  auto actorA_sparse = SparseMatrix<real>{
      3,
      DynamicArray<int>{0, 2, 4, 7},
      DynamicArray<int>{0, 2, 1, 2, 0, 1, 2},
      DynamicArray<real>{1, 3, 4, 5, 3, 5, 6}};
  auto actorA_sparseBlockable = SparseMatrix<real>{
      3,
      DynamicArray<int>{0, 3, 6, 9},
      DynamicArray<int>{0, 1, 2, 0, 1, 2, 0, 1, 2},
      DynamicArray<real>{1, 0, 3, 0, 4, 5, 3, 5, 6}};
  auto actorA_blockSparse = ToBlockSparseMatrix<3>(actorA_sparseBlockable);
  auto actorA_dense = ToMatrix(actorA_sparse);
  EXPECT_TRUE(NearEqualMatrices(actorA_sparse, actorA_sparseBlockable));
  EXPECT_TRUE(NearEqualMatrices(actorA_sparse, actorA_blockSparse));
  EXPECT_TRUE(NearEqualMatrices(actorA_sparse, actorA_dense));
  EXPECT_TRUE(NearEqualMatrices(actorA_sparse, Transpose(actorA_dense))); // Should be symmetrical

  // Actor B: 9x9
  // |912345678|
  // |123456789|
  // |234567891|
  // |345678912|
  // |456789123|
  // |567891234|
  // |678912345|
  // |789123456|
  // |891234567|
  auto actorB_sparse = SparseMatrix<real>{
      9,
      DynamicArray<int>{0, 9, 18, 27, 36, 45, 54, 63, 72, 81},
      DynamicArray<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, //
                        0, 1, 2, 3, 4, 5, 6, 7, 8, //
                        0, 1, 2, 3, 4, 5, 6, 7, 8, //
                        0, 1, 2, 3, 4, 5, 6, 7, 8, //
                        0, 1, 2, 3, 4, 5, 6, 7, 8, //
                        0, 1, 2, 3, 4, 5, 6, 7, 8, //
                        0, 1, 2, 3, 4, 5, 6, 7, 8, //
                        0, 1, 2, 3, 4, 5, 6, 7, 8, //
                        0, 1, 2, 3, 4, 5, 6, 7, 8},
      DynamicArray<real>{9, 1, 2, 3, 4, 5, 6, 7, 8, //
                         1, 2, 3, 4, 5, 6, 7, 8, 9, //
                         2, 3, 4, 5, 6, 7, 8, 9, 1, //
                         3, 4, 5, 6, 7, 8, 9, 1, 2, //
                         4, 5, 6, 7, 8, 9, 1, 2, 3, //
                         5, 6, 7, 8, 9, 1, 2, 3, 4, //
                         6, 7, 8, 9, 1, 2, 3, 4, 5, //
                         7, 8, 9, 1, 2, 3, 4, 5, 6, //
                         8, 9, 1, 2, 3, 4, 5, 6, 7}};
  SparseMatrix<real> actorB_sparseBlockable = actorB_sparse;
  auto actorB_blockSparse = ToBlockSparseMatrix<3>(actorB_sparseBlockable);
  auto actorB_dense = ToMatrix(actorB_sparse);
  EXPECT_TRUE(NearEqualMatrices(actorB_sparse, actorB_sparseBlockable));
  EXPECT_TRUE(NearEqualMatrices(actorB_sparse, actorB_blockSparse));
  EXPECT_TRUE(NearEqualMatrices(actorB_sparse, actorB_dense));
  EXPECT_TRUE(NearEqualMatrices(actorB_sparse, Transpose(actorB_dense))); // Should be symmetrical

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

  // Same as above, but filled in with zeros so that it's blockable.
  auto actorC_sparseBlockable = SparseMatrix<real>{
      12,
      DynamicArray<int>{0, 3, 6, 9, 18, 27, 36, 42, 48, 54, 60, 66, 72},
      DynamicArray<int>{0, 1, 2, //
                        0, 1, 2, //
                        0, 1, 2, //
                        3, 4, 5, 6, 7,  8,  9, 10, 11, //
                        3, 4, 5, 6, 7,  8,  9, 10, 11, //
                        3, 4, 5, 6, 7,  8,  9, 10, 11, //
                        3, 4, 5, 6, 7,  8, //
                        3, 4, 5, 6, 7,  8, //
                        3, 4, 5, 6, 7,  8, //
                        3, 4, 5, 9, 10, 11, //
                        3, 4, 5, 9, 10, 11, //
                        3, 4, 5, 9, 10, 11},
      DynamicArray<real>{1, 0, 4, //
                         0, 2, 0, //
                         4, 0, 3, //
                         5, 8, 9, 1, 2, 3, 4, 5, 6, //
                         8, 6, 1, 2, 4, 6, 8, 2, 4, //
                         9, 1, 7, 1, 3, 5, 7, 9, 1, //
                         1, 2, 1, 8, 2, 0, //
                         2, 4, 3, 2, 9, 3, //
                         3, 6, 5, 0, 3, 1, //
                         4, 8, 7, 2, 5, 6, //
                         5, 2, 9, 5, 3, 7, //
                         6, 4, 1, 6, 7, 4}};

  // Same as above, but in other matrix formats.
  auto actorC_blockSparse = ToBlockSparseMatrix<3>(actorC_sparseBlockable);
  auto actorC_dense = ToMatrix(actorC_sparse);
  EXPECT_TRUE(NearEqualMatrices(actorC_sparse, actorC_sparseBlockable));
  EXPECT_TRUE(NearEqualMatrices(actorC_sparse, actorC_blockSparse));
  EXPECT_TRUE(NearEqualMatrices(actorC_sparse, actorC_dense));
  EXPECT_TRUE(NearEqualMatrices(actorC_sparse, Transpose(actorC_dense))); // Should be symmetrical

  // Contact: Sparse 18x18 with 3 row/col offset
  // This is intentionally smaller than the 24x24 global system.
  // There is no contact with actor A, nor with the last 3 rows/cols of actor C.
  // |2        987369   |
  // | 3       852471   |
  // |  4      741582   |
  // |                  |
  // |                  |
  // |                  |
  // |      5        147|
  // |       6       258|
  // |        7      369|
  // |987               |
  // |854               |
  // |721               |
  // |345               |
  // |678               |
  // |912               |
  // |      123         |
  // |      456         |
  // |      789         |
  auto contact_sparse = SparseMatrix<real>{
      18,
      DynamicArray<int>{0, 7, 14, 21, 21, 21, 21, 25, 29, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60},
      DynamicArray<int>{0, 9,  10, 11, 12, 13, 14, //
                        1, 9,  10, 11, 12, 13, 14, //
                        2, 9,  10, 11, 12, 13, 14, //
                        6, 15, 16, 17, //
                        7, 15, 16, 17, //
                        8, 15, 16, 17, //
                        0, 1,  2, //
                        0, 1,  2, //
                        0, 1,  2, //
                        0, 1,  2, //
                        0, 1,  2, //
                        0, 1,  2, //
                        6, 7,  8, //
                        6, 7,  8, //
                        6, 7,  8},
      DynamicArray<real>{2, 9, 8, 7, 3, 6, 9, //
                         3, 8, 5, 2, 4, 7, 1, //
                         4, 7, 4, 1, 5, 8, 2, //
                         5, 1, 4, 7, //
                         6, 2, 5, 8, //
                         7, 3, 6, 9, //
                         9, 8, 7, //
                         8, 5, 4, //
                         7, 2, 1, //
                         3, 4, 5, //
                         6, 7, 8, //
                         9, 1, 2, //
                         1, 2, 3, //
                         4, 5, 6, //
                         7, 8, 9}};

  // Same as above, but filled in with zeros so that it's blockable.
  auto contact_sparseBlockable = SparseMatrix<real>{
      18,
      DynamicArray<int>{0, 9, 18, 27, 27, 27, 27, 33, 39, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72},
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
                        6, 7, 8},
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
                         7, 8, 9}};

  // Same as above, but in other matrix formats.
  auto contact_blockSparse = ToBlockSparseMatrix<3>(contact_sparseBlockable);
  auto contact_dense = ToMatrix(contact_sparse);
  EXPECT_TRUE(NearEqualMatrices(contact_sparse, contact_sparseBlockable));
  EXPECT_TRUE(NearEqualMatrices(contact_sparse, contact_blockSparse));
  EXPECT_TRUE(NearEqualMatrices(contact_sparse, contact_dense));
  EXPECT_TRUE(NearEqualMatrices(contact_sparse, Transpose(contact_dense))); // should be symmetrical

  // Helper to check the full matrix
  auto CheckGlobalMatrix = [&](auto const& fullMat) {
    EXPECT_EQ(24, fullMat.Rows());
    EXPECT_EQ(24, fullMat.Cols());

    // Combined contribution from actor matrices
    real constexpr kFullActorMatrix[24][24] = {
        {1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, //
        {0, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, //
        {3, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, //
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
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 7, 0, 0, 0, 2, 5, 6},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 2, 9, 0, 0, 0, 5, 3, 7},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 4, 1, 0, 0, 0, 6, 7, 4}};

    // Check rows of actor A
    for (int r = 0; r < 24; ++r) {
      for (int c = 0; c < 24; ++c) {
        real actorContribution = kFullActorMatrix[r][c];
        real contactContribution = ((r >= 3) && (c >= 3) && (r - 3 < contact_dense.Rows()) &&
                                    (c - 3 < contact_dense.Cols()))
            ? contact_dense(r - 3, c - 3)
            : 0_r;
        real expectedValue = actorContribution + contactContribution;
        real actualValue = fullMat(r, c);
        EXPECT_NEAR_EQ(expectedValue, actualValue);
      }
    }
  };

  // Various combinations of matrix types
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        for (int d = 0; d < 4; ++d) {
          std::vector<std::pair<int, AnyMatrixView<real const>>> actorMatrices;
          std::vector<AnyInteractionMatrixViewInfo<real const>> contactMatrices;
          // clang-format off
          switch (a) {
            case 0: actorMatrices.emplace_back(0, actorA_sparse); break;
            case 1: actorMatrices.emplace_back(0, actorA_sparseBlockable); break;
            case 2: actorMatrices.emplace_back(0, actorA_blockSparse); break;
            case 3: actorMatrices.emplace_back(0, actorA_dense); break;
          }
          switch (b) {
            case 0: actorMatrices.emplace_back(3, actorB_sparse); break;
            case 1: actorMatrices.emplace_back(3, actorB_sparseBlockable); break;
            case 2: actorMatrices.emplace_back(3, actorB_blockSparse); break;
            case 3: actorMatrices.emplace_back(3, actorB_dense); break;
          }
          switch (c) {
            case 0: actorMatrices.emplace_back(12, actorC_sparse); break;
            case 1: actorMatrices.emplace_back(12, actorC_sparseBlockable); break;
            case 2: actorMatrices.emplace_back(12, actorC_blockSparse); break;
            case 3: actorMatrices.emplace_back(12, actorC_dense); break;
          }
          switch (d) {
            case 0: contactMatrices.emplace_back(3, 3, contact_sparse, /*symmetricPair*/ std::nullopt); break;
            case 1: contactMatrices.emplace_back(3, 3, contact_sparseBlockable, /*symmetricPair*/ std::nullopt); break;
            case 2: contactMatrices.emplace_back(3, 3, contact_blockSparse, /*symmetricPair*/ std::nullopt); break;
            case 3: contactMatrices.emplace_back(3, 3, contact_dense, /*symmetricPair*/ std::nullopt); break;
          }
          // clang-format on

          // Put it all together
          IslandOperators<real> ops{actorMatrices, contactMatrices, {}};
          EXPECT_EQ(24, ops.Rows());
          EXPECT_EQ(24, ops.Cols());

          // Blockable if and only if actor A, actor C and contact are not in sparse format.
          EXPECT_EQ(ops.IsBlockable<3>(), (a != 0) && (c != 0) && (d != 0));

          CheckGlobalMatrix(ToMatrix(ops));

          SparseMatrix<real> fullSp = ops.FullSparseMatrix();
          CheckGlobalMatrix(fullSp);

          if (ops.IsBlockable<3>()) {
            CheckGlobalMatrix(ops.FullBlockSparseMatrix<3>());
          } else {
            EXPECT_TRUE(ops.FullBlockSparseMatrix<3>().empty());
          }

          auto mat = ops.CondenseFullMatrix();
          std::visit([&](auto const& A) { CheckGlobalMatrix(A); }, mat);

          // 'Apply' and 'ApplyToRange' methods.
          real const absTol = fullSp.Norm() * std::numeric_limits<real>::epsilon();
          ColumnVector<real> x(ops.Rows()), y0(ops.Cols()), y1(ops.Cols());
          x.SetRandom(1);
          y0.SetRandom(2);
          y1.SetRandom(3);
          fullSp.Apply(x, y0);
          ops.Apply(x, y1);
          EXPECT_TRUE(NearEqualMatrices(y0, y1, absTol));

          x.SetRandom(4);
          int rowBegin = 3, rowEnd = 15;
          fullSp.ApplyToRange(x, y0, rowBegin, rowEnd);
          EXPECT_FALSE(NearEqualMatrices(y0, y1, absTol)); // Not same as before
          ops.ApplyToRange(x, y1, rowBegin, rowEnd);
          EXPECT_TRUE(NearEqualMatrices(y0, y1, absTol));
        }
      }
    }
  }
}

TEST(IslandOperators, MultipleInteractionMatrices) {
  auto runTests = [](int aRowOffset,
                     int aColOffset,
                     int bRowOffset,
                     int bColOffset,
                     int cRowOffset,
                     int cColOffset,
                     int dRowOffset,
                     int dColOffset,
                     int actor0Dofs,
                     int actor1Dofs) {
    if (actor0Dofs + actor1Dofs < Max(aRowOffset + 3,
                                      aColOffset + 3,
                                      bRowOffset + 6,
                                      bColOffset + 6,
                                      cRowOffset + 3,
                                      cColOffset + 3,
                                      dRowOffset + 3,
                                      dColOffset + 3)) {
      return; // Illegal case.
    }

    bool const blockableOffsets = (aRowOffset % 3 == 0) && (aColOffset % 3 == 0) &&
        (bRowOffset % 3 == 0) && (bColOffset % 3 == 0) && (cRowOffset % 3 == 0) &&
        (cColOffset % 3 == 0) && (dRowOffset % 3 == 0) && (dColOffset % 3 == 0) &&
        (actor0Dofs % 3 == 0) && (actor1Dofs % 3 == 0);

    auto actorMatrix0 = Matrix<real>::Zero(actor0Dofs, actor0Dofs);
    Matrix<real> actor1Asym(actor1Dofs, actor1Dofs);
    actor1Asym.SetRandom(actor0Dofs + actor1Dofs, -0.5_r, 0.5_r);
    Matrix<real> actorMatrix1 = actor1Asym + actor1Asym.Transpose(); // Must be symmetric.

    auto ref = Matrix<real>::Zero(actor0Dofs + actor1Dofs, actor0Dofs + actor1Dofs);
    ref.Block(0, 0, actor0Dofs, actor0Dofs) += actorMatrix0;
    ref.Block(actor0Dofs, actor0Dofs, actor1Dofs, actor1Dofs) += actorMatrix1;
    auto addToRef = [&](auto const& I, int iRowOffset, int iColOffset) {
      ref.Block(iRowOffset, iColOffset, I.Rows(), I.Cols()) += I;
    };

    std::vector<AnyInteractionMatrixViewInfo<real const>> interactionMatrices;
    std::vector<AnyInteractionMatrixViewInfo<real const>> interactionMatricesBlockable;

    // Matrix A: 3x3 sparse
    auto aSparse = SparseMatrix<real>{3, {0, 2, 2, 3}, {0, 2, 2}, {0.1_r, -0.2_r, -0.3_r}};
    AnyMatrix<real> aInteraction = aSparse;
    AnyMatrix<real> aInteractionBlockable = SparseMatrix<real>{
        3,
        {0, 3, 6, 9},
        {0, 1, 2, 0, 1, 2, 0, 1, 2},
        {0.1_r, 0_r, -0.2_r, 0_r, 0_r, 0_r, 0_r, 0_r, -0.3_r}};
    interactionMatrices.emplace_back(
        aRowOffset, aColOffset, AsConstView(aInteraction), /*symmetricPair*/ std::nullopt);
    interactionMatricesBlockable.emplace_back(
        aRowOffset, aColOffset, AsConstView(aInteractionBlockable), /*symmetricPair*/ std::nullopt);
    addToRef(ToMatrix(aSparse), aRowOffset, aColOffset);

    // Matrix B: 4x4 sparse
    auto bSparse = SparseMatrix<real>{
        4,
        {0, 0, 3, 6, 8},
        {1, 2, 3, 1, 2, 3, 0, 2},
        {-0.4_r, 0.5_r, 0.6_r, 0.7_r, 0.8_r, 0.9_r, 0.1_r, -0.2_r}};
    AnyMatrix<real> bInteraction = bSparse;
    AnyMatrix<real> bInteractionBlockable = SparseMatrix<real>{
        6,
        {0, 6, 12, 18, 21, 24, 27},
        {0, 1, 2, 3, 4, 5, //
         0, 1, 2, 3, 4, 5, //
         0, 1, 2, 3, 4, 5, //
         0, 1, 2, //
         0, 1, 2, //
         0, 1, 2},
        {0_r,   0_r,    0_r,    0_r,   0_r, 0_r, //
         0_r,   -0.4_r, 0.5_r,  0.6_r, 0_r, 0_r, //
         0_r,   0.7_r,  0.8_r,  0.9_r, 0_r, 0_r, //
         0.1_r, 0_r,    -0.2_r, //
         0_r,   0_r,    0_r, //
         0_r,   0_r,    0_r}};
    interactionMatrices.emplace_back(
        bRowOffset, bColOffset, AsConstView(bInteraction), /*symmetricPair*/ std::nullopt);
    interactionMatricesBlockable.emplace_back(
        bRowOffset, bColOffset, AsConstView(bInteractionBlockable), /*symmetricPair*/ std::nullopt);
    addToRef(ToMatrix(bSparse), bRowOffset, bColOffset);

    // Matrix C: 2x2 sparse
    auto cSparse = SparseMatrix<real>{2, {0, 2, 3}, {0, 1, 0}, {-0.3_r, 0.4_r, 0.5_r}};
    AnyMatrix<real> cInteraction = cSparse;
    AnyMatrix<real> cInteractionBlockable = SparseMatrix<real>{
        3,
        {0, 3, 6, 9},
        {0, 1, 2, 0, 1, 2, 0, 1, 2},
        {-0.3_r, 0.4_r, 0_r, 0.5_r, 0_r, 0_r, 0_r, 0_r, 0_r}};
    interactionMatrices.emplace_back(
        cRowOffset, cColOffset, AsConstView(cInteraction), /*symmetricPair*/ std::nullopt);
    interactionMatricesBlockable.emplace_back(
        cRowOffset, cColOffset, AsConstView(cInteractionBlockable), /*symmetricPair*/ std::nullopt);
    addToRef(ToMatrix(cSparse), cRowOffset, cColOffset);

    // Matrix D: 3x3 dense
    Matrix<real> dDense =
        RowMatrix<real>{{1_r, 0.2_r, -0.9_r}, {-0.2_r, 0.4_r, 1_r}, {-0.3_r, 0.5_r, 0.6_r}};
    AnyMatrix<real> dInteraction = dDense;
    interactionMatrices.emplace_back(
        dRowOffset, dColOffset, AsConstView(dInteraction), /*symmetricPair*/ std::nullopt);
    interactionMatricesBlockable.emplace_back(
        dRowOffset, dColOffset, AsConstView(dInteraction), /*symmetricPair*/ std::nullopt);
    addToRef(dDense, dRowOffset, dColOffset);

    std::vector<std::pair<int, AnyMatrixView<real const>>> actorMatrices;
    if (actor0Dofs > 0) {
      actorMatrices.emplace_back(0, AsConstView(actorMatrix0));
    }
    actorMatrices.emplace_back(actor0Dofs, AsConstView(actorMatrix1));

    IslandOperators<real> ops1(actorMatrices, interactionMatrices, {});
    IslandOperators<real> ops2(actorMatrices, interactionMatricesBlockable, {});

    EXPECT_FALSE(ops1.IsBlockable<3>());
    EXPECT_EQ(blockableOffsets, ops2.IsBlockable<3>());

    SparseMatrix<real> fullSp1 = ops1.FullSparseMatrix();
    SparseMatrix<real> fullSp2 = ops2.FullSparseMatrix();

    EXPECT_TRUE(NearEqualMatrices(ref, fullSp1));
    EXPECT_TRUE(NearEqualMatrices(fullSp1, fullSp2));

    EXPECT_TRUE(NearEqualMatrices(fullSp1, ToMatrix(ops1)));
    EXPECT_TRUE(NearEqualMatrices(fullSp1, ToMatrix(ops2)));

    EXPECT_TRUE(ops1.FullBlockSparseMatrix<3>().empty());
    if (blockableOffsets) {
      EXPECT_TRUE(NearEqualMatrices(fullSp1, ops2.FullBlockSparseMatrix<3>()));
    } else {
      EXPECT_TRUE(ops2.FullBlockSparseMatrix<3>().empty());
    }

    auto global1 = ops1.CondenseFullMatrix();
    auto global2 = ops2.CondenseFullMatrix();
    std::visit([&](auto const& A) { EXPECT_TRUE(NearEqualMatrices(fullSp1, A)); }, global1);
    std::visit([&](auto const& A) { EXPECT_TRUE(NearEqualMatrices(fullSp1, A)); }, global2);

    // 'Apply' and 'ApplyToRange' methods.
    real const absTol = 2_r * ref.Cols() * std::numeric_limits<real>::epsilon();
    ColumnVector<real> x(ops1.Rows()), y0(ops1.Cols()), y1(ops1.Cols()), y2(ops1.Cols());
    x.SetRandom(1);
    y0.SetRandom(2);
    y1.SetRandom(3);
    y2.SetRandom(4);
    fullSp1.Apply(x, y0);
    ops1.Apply(x, y1);
    ops2.Apply(x, y2);
    EXPECT_TRUE(NearEqualMatrices(y0, y1, absTol));
    EXPECT_TRUE(NearEqualMatrices(y0, y2, absTol));

    x.SetRandom(5);
    int rowBegin = 3, rowEnd = Max(3, ops1.Rows() - 3);
    fullSp1.ApplyToRange(x, y0, rowBegin, rowEnd);
    EXPECT_EQ(rowEnd > rowBegin, !NearEqualMatrices(y0, y1)); // Not same as before
    ops1.ApplyToRange(x, y1, rowBegin, rowEnd);
    ops2.ApplyToRange(x, y2, rowBegin, rowEnd);
    EXPECT_TRUE(NearEqualMatrices(y0, y1, absTol));
    EXPECT_TRUE(NearEqualMatrices(y0, y2, absTol));
  };

  for (auto aOffset : {0, 3, 5, 9, 13}) {
    for (auto bOffset : {0, 3, 5, 9, 13}) {
      for (auto cOffset : {0, 3, 5, 9, 13}) {
        for (auto dOffset : {0, 3, 5, 9, 13}) {
          for (auto actor0Dofs : {0, 3, 8}) {
            for (auto actor1Dofs : {1, 6, 15}) {
              for (auto colVsRowOffset : {-6, -4, 0, 4, 6}) {
                runTests(
                    /*aRowOffset*/ aOffset,
                    /*aColOffset*/ Max(0, aOffset + colVsRowOffset),
                    /*bRowOffset*/ bOffset,
                    /*bColOffset*/ Max(0, bOffset - colVsRowOffset),
                    /*cRowOffset*/ cOffset,
                    /*cColOffset*/ Max(0, cOffset + colVsRowOffset),
                    /*dRowOffset*/ dOffset,
                    /*dColOffset*/ Max(0, dOffset - colVsRowOffset),
                    actor0Dofs,
                    actor1Dofs);
              }
            }
          }
        }
      }
    }
  }
}

// TODO: Test all other methods
