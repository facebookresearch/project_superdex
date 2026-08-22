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
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/solvers/actor_preconditioner.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>

#include <algorithm>
#include <type_traits>
#include <utility>
#include <variant>

using namespace mochi;
using namespace mochi::test;

namespace {

/**
 * @brief Extract sparse matrix C from actorMat
 *
 * @param[in] rowC Number of rows in the contact matrix
 * @param[in] localCStart Local start index of the contact matrix
 * @param[in,out] actorMat Actor matrix
 * @return Contact matrix C in SparseMatrix format
 *
 * @note We assume that the column indices on each row are sorted.
 */
auto ExtractPseudoContactMatrix(int rowC, int localCstart, SparseMatrix<real>& actorMat) {
  // Prepare a contact matrix
  DynamicArray<int> ptr1(rowC + 1);
  ptr1[0] = 0;
  //
  DynamicArray<int> col1;
  DynamicArray<real> val1;
  // Transfer the contact part
  for (int i = 0; i < rowC; ++i) {
    int rowActor = i + localCstart;
    if ((rowActor < 0) || (rowActor >= actorMat.Rows())) {
      ptr1[i + 1] = ptr1[i];
      continue;
    }
    auto const myCol = actorMat.Indices(rowActor);
    auto myVal = actorMat.Values(rowActor);
    for (int p = 0; p < isize(myCol); ++p) {
      if (myCol[p] < localCstart) {
        continue;
      }
      // Assume that the column indices are sorted
      if (myCol[p] >= localCstart + rowC) {
        break;
      }
      col1.push_back(myCol[p] - localCstart);
      val1.push_back(myVal[p]);
      myVal[p] = 0_r;
    }
    ptr1[i + 1] = isize(col1);
  }
  return SparseMatrix<real>{rowC, std::move(ptr1), std::move(col1), std::move(val1)};
}

/**
 * @brief Extract block sparse matrix C from actorMat
 *
 * @param[in] numBlocks Number of blocks in the contact matrix
 * @param[in] cLocalStart Local start block index of the contact matrix
 * @param[in,out] actorMat Actor matrix
 * @return Contact matrix C in BlockSparseMatrix format
 * @note We assume that the block column indices on each block row are sorted.
 */
template <int kBlockSize>
auto ExtractPseudoContactMatrix(
    int numBlocks,
    int cLocalStart,
    BlockSparseMatrix<real, kBlockSize>& actorMat) {
  // Prepare a contact matrix
  DynamicArray<int> ptr1(numBlocks + 1);
  ptr1[0] = 0;
  DynamicArray<int> col1;
  // Transfer the contact part in two passes
  // First pass for the "graph"
  for (int i = 0; i < numBlocks; ++i) {
    int actorBlockRow = i + cLocalStart;
    if ((actorBlockRow < 0) || (actorBlockRow >= actorMat.BlockRows())) {
      ptr1[i + 1] = ptr1[i];
      continue;
    }
    auto const myCol = actorMat.Indices(actorBlockRow);
    for (int p = 0; p < isize(myCol); ++p) {
      if (myCol[p] < cLocalStart) {
        continue;
      }
      // Assume that the column indices are sorted
      if (myCol[p] >= cLocalStart + numBlocks) {
        break;
      }
      col1.push_back(myCol[p] - cLocalStart);
    }
    ptr1[i + 1] = isize(col1);
  }
  DynamicArray<real> val1(kBlockSize * kBlockSize * ptr1[numBlocks], 0_r);
  BlockSparseMatrix<real, kBlockSize> C{
      numBlocks, std::move(ptr1), std::move(col1), std::move(val1)};
  // Second pass for the numerical values
  for (int i = 0; i < numBlocks; ++i) {
    int actorBlockRow = i + cLocalStart;
    if ((actorBlockRow < 0) || (actorBlockRow >= actorMat.BlockRows())) {
      continue;
    }
    auto const myCol = actorMat.Indices(actorBlockRow);
    auto myVal = actorMat.Values(actorBlockRow);
    auto const cCol = C.Indices(i);
    auto cVal = C.Values(i);
    for (int k = 0; k < isize(cCol); ++k) {
      auto const* myPtr = std::lower_bound(myCol.begin(), myCol.end(), cCol[k] + cLocalStart);
      auto const p = int(myPtr - myCol.begin());
      // Copy matrix kBlockSize x kBlockSize
      cVal[k] = myVal[p];
      myVal[p].SetZero();
    }
  }
  return std::move(C);
}

template <int kPrecBlockSize, typename MatType>
void TestActorPrecSolve(
    MatType const& A,
    ActorPseudoMatrix<real> const& actorPM,
    Span<PreconditionerType const> precList) {
  int const n = GetNumRows(A);
  ColumnVector<real> x(n), Px(n), Nx(n);
  x.SetRandom(123);
  for (auto pType : precList) {
    switch (pType) {
      default: {
        MOCHI_ASSERT(
            false, "Per-actor preconditioner type (%i) not supported.", static_cast<int>(pType));
        break;
      }
      case PreconditionerType::AMG: {
        krylov::AMGPrec<real, kPrecBlockSize> P(A);
        AMGActorPrec<real, kPrecBlockSize> N(actorPM);
        EXPECT_EQ(pType, N.GetType());
        P.Solve(x, Px);
        N.Solve(x, Nx);
        break;
      }
      case PreconditionerType::BlockJacobi: {
        krylov::BlockJacobiPrec<real, kPrecBlockSize> P(A);
        BlockJacobiActorPrec<real, kPrecBlockSize> N(actorPM);
        EXPECT_EQ(pType, N.GetType());
        P.Solve(x, Px);
        N.Solve(x, Nx);
        break;
      }
      case PreconditionerType::ColoredSSOR: {
        krylov::ColoredSSORPrec<MatType> P(A);
        ColoredSSORActorPrec<real, kPrecBlockSize> N(actorPM);
        EXPECT_EQ(pType, N.GetType());
        P.Solve(x, Px);
        N.Solve(x, Nx);
        break;
      }
      case PreconditionerType::SymInverse: {
        krylov::SymInversePrec<real> P(A);
        SymInverseActorPrec<real> N(actorPM);
        EXPECT_EQ(pType, N.GetType());
        P.Solve(x, Px);
        N.Solve(x, Nx);
        break;
      }
      case PreconditionerType::Jacobi: {
        krylov::JacobiPrec<real> P(A);
        BlockJacobiActorPrec<real, 1> N(actorPM);
        EXPECT_EQ(pType, N.GetType());
        P.Solve(x, Px);
        N.Solve(x, Nx);
        break;
      }
      case PreconditionerType::ILU0: {
        krylov::RelaxedILUPrec<MatType> P(A, /*fillInLevel*/ 0, /*alphaRelax*/ 0_r);
        ILU0ActorPrec<real, kPrecBlockSize> N(actorPM);
        EXPECT_EQ(pType, N.GetType());
        P.Solve(x, Px);
        N.Solve(x, Nx);
        break;
      }
    };
    EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, Nx));
  }
}

template <int kActorBlockSize>
void TestActorPrecFromBlockSparseActorMatrix(Span<PreconditionerType const> precList) {
  int const n = 3;
  auto Ab = mochi::test::MakeBlockSparseMatrix<real, kActorBlockSize>(n, n + 1, n + 2);
  BlockSparseMatrix<real, kActorBlockSize> actorMat(Ab);
  int const offsetA = Ab.Rows();
  //--- First "pseudo-contact" matrix
  int const localCStart = -kActorBlockSize;
  int const offsetC = offsetA + localCStart;
  auto C = ExtractPseudoContactMatrix(3, localCStart / kActorBlockSize, actorMat);
  //--- Second "pseudo-contact" matrix
  int const localDStart = Ab.Rows() - 3 * kActorBlockSize;
  int const offsetD = offsetA + localDStart;
  auto D = ExtractPseudoContactMatrix(5, localDStart / kActorBlockSize, actorMat);
  auto Dsp = ToSparseMatrix(D);
  ActorPseudoMatrix<real> actorPM{
      offsetA,
      actorMat,
      {{offsetC, offsetC, C, std::nullopt}, {offsetD, offsetD, Dsp, std::nullopt}}};
  {
    auto const matAb = ToMatrix(Ab);
    auto const matActor = ToMatrix(actorPM.actorMatrix);
    EXPECT_FALSE(mochi::test::NearEqualMatrices(matAb, matActor));
    EXPECT_TRUE(mochi::test::NearEqualMatrices(matAb, ToMatrix(actorPM)));
  }
  TestActorPrecSolve</*kPrecBlockSize*/ kActorBlockSize>(Ab, actorPM, precList);
}

void TestActorPrecFromDenseActorMatrix(Span<PreconditionerType const> precList) {
  int constexpr kBlockSize = 3;
  int const n = 3;
  auto Ab = mochi::test::MakeBlockSparseMatrix<real, kBlockSize>(n, n + 1, n + 2);
  BlockSparseMatrix<real, kBlockSize> actorMat(Ab);
  int const offsetA = Ab.Rows();
  //--- First "pseudo-contact" matrix
  int const localCStart = -kBlockSize;
  int const offsetC = offsetA + localCStart;
  auto C = ExtractPseudoContactMatrix(3, localCStart / kBlockSize, actorMat);
  //--- Second "pseudo-contact" matrix
  int const localDStart = Ab.Rows() - 3 * kBlockSize;
  int const offsetD = offsetA + localDStart;
  auto D = ExtractPseudoContactMatrix(5, localDStart / kBlockSize, actorMat);
  auto Dsp = ToSparseMatrix(D);
  //--- Third "pseudo-contact" matrix
  int const localEStart = Ab.Rows() / 2;
  MOCHI_ASSERT(localEStart % kBlockSize == 0);
  int const offsetE = offsetA + localEStart;
  auto E = ExtractPseudoContactMatrix(kBlockSize, localEStart / kBlockSize, actorMat);
  auto Emat = ToMatrix(E);
  //
  auto const matActor = ToMatrix(actorMat);
  //
  ActorPseudoMatrix<real> actorPM{
      offsetA,
      matActor,
      {{offsetC, offsetC, C, std::nullopt},
       {offsetD, offsetD, Dsp, std::nullopt},
       {offsetE, offsetE, Emat, std::nullopt}}};
  {
    auto const matAb = ToMatrix(Ab);
    EXPECT_FALSE(mochi::test::NearEqualMatrices(matAb, matActor));
    EXPECT_TRUE(mochi::test::NearEqualMatrices(matAb, ToMatrix(actorPM)));
  }
  TestActorPrecSolve</*kPrecBlockSize*/ kBlockSize>(Ab, actorPM, precList);
}

void TestActorPrecFromSparseActorMatrix(Span<PreconditionerType const> precList) {
  int constexpr kPrecBlockSize = 3;
  int const n = 8 * kPrecBlockSize;
  auto As = mochi::test::Make3ptLaplacianMatrix(n);
  SparseMatrix<real> actorMat(As);
  int const offsetA = As.Rows();
  //--- Use a 3x3 matrix for the contact C
  int const localCstart = -1;
  int const offsetC = offsetA + localCstart;
  auto C = ExtractPseudoContactMatrix(3, localCstart, actorMat);
  //--- Use a single diagonal entry for the contact D
  int const localDstart = n / 2;
  int const offsetD = offsetA + localDstart;
  auto D = ExtractPseudoContactMatrix(1, localDstart, actorMat);
  //
  ActorPseudoMatrix<real> actorPM{
      offsetA,
      actorMat,
      {{offsetC, offsetC, C, std::nullopt}, {offsetD, offsetD, D, std::nullopt}}};
  {
    auto const matAs = ToMatrix(As);
    auto const matActor = ToMatrix(actorPM.actorMatrix);
    EXPECT_FALSE(mochi::test::NearEqualMatrices(matAs, matActor));
    EXPECT_TRUE(mochi::test::NearEqualMatrices(matAs, ToMatrix(actorPM)));
  }
  TestActorPrecSolve<kPrecBlockSize>(As, actorPM, precList);
}

} // namespace

TEST(ActorPreconditionerTest, ActorPreconditioner) {
  static_assert(
      std::variant_size_v<std::decay_t<decltype(ActorPseudoMatrix<real>::actorMatrix)>> == 4,
      "Please update the unit test if the actor matrix types change");

  // Preconditioners supported for all actor matrix types.
  DynamicArray<PreconditionerType> precAll(
      {PreconditionerType::BlockJacobi,
       PreconditionerType::SymInverse,
       PreconditionerType::Jacobi});
  TestActorPrecFromDenseActorMatrix(precAll);
  TestActorPrecFromSparseActorMatrix(precAll);
  TestActorPrecFromBlockSparseActorMatrix<3>(precAll);
  TestActorPrecFromBlockSparseActorMatrix<4>(precAll);

  // Preconditioners only supported for block sparse actor matrices.
  DynamicArray<PreconditionerType> precBSp(
      {PreconditionerType::AMG, PreconditionerType::ColoredSSOR, PreconditionerType::ILU0});
  TestActorPrecFromBlockSparseActorMatrix<3>(precBSp);
  TestActorPrecFromBlockSparseActorMatrix<4>(precBSp);
}
