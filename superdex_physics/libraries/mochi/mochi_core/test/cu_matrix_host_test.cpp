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

#include <mochi_core/linear_algebra/base_tools.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/strided_matrix.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

using namespace mochi;

namespace {

/// @brief Make a const reference to any object.
auto const& Const(auto const& v) {
  return v;
}

void ExpectEqualMatrices(auto&& A, auto&& B) {
  for (int r = 0; r < A.Rows(); ++r) {
    for (int c = 0; c < A.Cols(); ++c) {
      EXPECT_EQ(A(r, c), B(r, c));
    }
  }
}

void ExpectCloseMatrices(auto&& A, auto&& B, auto eps) {
  double diff = 0;
  double sum = 0;
  EXPECT_TRUE(A.Rows() <= B.Rows() && A.Cols() <= B.Cols());
  for (int r = 0; r < A.Rows(); ++r) {
    for (int c = 0; c < A.Cols(); ++c) {
      diff += std::abs(A(r, c) - B(r, c));
      sum += std::abs(A(r, c) + B(r, c));
    }
  }
  bool isClose = diff < eps * sum;
  EXPECT_TRUE(isClose);
}

template <typename Scalar, int kStride>
void TestStridedView() {
  StridedMatrix<Scalar, 3, 3> A1{{10, 3, 5}, {2, 12, 4}, {-1, -2, 16}};

  StridedMatrix<Scalar, 3, 3, kStride> A;
  A = A1;

  Matrix<Scalar, 3, 3> Am{{10, 3, 5}, {2, 12, 4}, {-1, -2, 16}};
  StridedMatrix<Scalar, 3, 3, kStride> B = A * A;
  Matrix<Scalar, 3, 3> Bm = Am * Am;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      EXPECT_EQ(B(row, col), Bm(row, col));
    }
  }

  ColumnVector<Scalar> Astorage(3 * 3 * kStride * 5);
  StridedView<Scalar, 3, 3, kStride> AV{3, 3, Astorage.GetSpan()};
  for (int block = 0; block < 5 * kStride; ++block) {
    auto Astrided = AV[block];
    Astrided = static_cast<Scalar>(block % 7 + 1) * A1;
  }
  ColumnVector<Scalar> Bstorage(3 * 3 * kStride * 5);
  StridedView<Scalar, 3, 3, kStride> BV{3, 3, Bstorage.GetSpan()};
  for (int block = 0; block < 5 * kStride; ++block) {
    auto Astrided = AV[block];
    auto Bstrided = BV[block];
    Bstrided = Astrided * A1;
  }

  for (int block = 0; block < 5 * kStride; ++block) {
    auto Bstrided = BV[block];
    Matrix<Scalar, 3, 3> Res = static_cast<Scalar>(block % 7 + 1) * Am * Am;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        EXPECT_EQ(Bstrided(row, col), Res(row, col));
      }
    }
  }
}

template <typename Scalar, int kStride>
void TestStridedBlocks() {
  constexpr int kRowsA = 4;
  constexpr int kColsA = 7;
  constexpr int kColsB = 5;
  Matrix<Scalar, kRowsA, kColsA> Ah;
  Ah.SetRandom(13);
  StridedMatrix<Scalar, kRowsA, kColsA, kStride> Acu;
  Acu = Ah;
  Matrix<Scalar, kColsA, kColsB> Ch;
  Ch.SetRandom(66);
  StridedMatrix<Scalar, kColsA, kColsB, kStride> Ccu;
  Ccu = Ch;
  constexpr int kMax = std::max({kRowsA, kColsA, kColsB});
  Matrix<Scalar, 2 * kMax + 6, 2 * kMax + 11> Bh;
  Bh.SetZero();
  StridedMatrix<Scalar, kMax + 6, kMax + 11, kStride> B;
  B.SetZero();

  B.template Block<kRowsA, kColsB>(1, 2) = Acu * Ccu;
  B.Block(1, 2, kRowsA, kColsB) += Scalar{1.2} * Const(Acu) * Ccu;

  Bh.template Block<kRowsA, kColsB>(1, 2, kRowsA, kColsB) = Ah * Ch;
  Bh.Block(1, 2, kRowsA, kColsB) += Scalar{1.2} * Const(Ah) * Ccu;

  B.template Block<kColsB, kRowsA>(kRowsA + 3, 1) = Ccu.Transpose() * Acu.Transpose();
  Bh.template Block<kColsB, kRowsA>(kRowsA + 3, 1, kColsB, kRowsA) =
      Ch.Transpose() * Ah.Transpose();

  B.template Block<2, 2>(3, kMax + 3) =
      Acu.template Block<2, 3>(1, 1) * Ccu.template Block<3, 2>(1, 2);
  Bh.template Block<2, 2>(3, kMax + 3, 2, 2) =
      Ah.template Block<2, 3>(1, 1, 2, 3) * Ch.template Block<3, 2>(1, 2, 3, 2);

  B.template Block<2, 2>(3, kMax + 3) -=
      Const(Acu).template Block<2, 3>(1, 1) * Const(Ccu.template Block<3, 2>(1, 2));
  Bh.template Block<2, 2>(3, kMax + 3, 2, 2) -=
      Const(Ah).template Block<2, 3>(1, 1, 2, 3) * Const(Ch.template Block<3, 2>(1, 2, 3, 2));

  ExpectCloseMatrices(B, Bh, Scalar{1e-5});
}

} // namespace

template <typename Scalar, int kStride>
static void TestStridedMatrixHost() {
  TestStridedView<Scalar, kStride>();
  TestStridedBlocks<Scalar, kStride>();
}

TEST(StridedAlgebra, StridedMatrixHost) {
  TestStridedMatrixHost<float, 1>();
  TestStridedMatrixHost<float, 32>();
}
