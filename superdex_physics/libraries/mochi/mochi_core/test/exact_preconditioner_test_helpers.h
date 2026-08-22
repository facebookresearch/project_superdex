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

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>

#include <gtest/gtest.h>

#include <limits>

namespace mochi::test {

template <typename Scalar, typename PrecType, typename MatrixType>
static void TestExactPreconditionerOnMatrix(MatrixType& A) {
  EXPECT_EQ(A.Rows(), A.Cols());
  int const sizeA = A.Rows();
  int const kNumColsB = 4;
  Matrix<Scalar> B(sizeA, kNumColsB);
  B.SetRandom(1);

  PrecType P(A);
  Matrix<Scalar, mochi::krylov::kDynamic, kNumColsB> AB = A * B;
  Matrix<Scalar> PAB(sizeA, kNumColsB);
  P(AB, PAB); // Compile-time cols input.
  Matrix<Scalar, mochi::krylov::kDynamic, kNumColsB> PB(sizeA, kNumColsB);
  P(B, PB); // Compile-time cols output.
  Matrix<Scalar> APB = A * PB;

  Scalar const absTol = 8 * sizeA * sizeA * sizeA * std::numeric_limits<Scalar>::epsilon();
  for (int i = 0; i < sizeA; ++i) {
    for (int j = 0; j < kNumColsB; ++j) {
      EXPECT_LE(mochi::Abs(PAB(i, j) - B(i, j)), absTol);
      EXPECT_LE(mochi::Abs(APB(i, j) - B(i, j)), absTol);
    }
  }
}

template <typename Scalar, typename PrecType, bool kTestSymmetricMatrix>
static void TestExactPreconditioner() {
  // Tests an exact preconditioner on various matrix types.

  auto symmetrizeDenseMatrix = [](auto&& mat) {
    ASSERT_EQ(mat.Rows(), mat.Cols());
    for (int ii = 0; ii < mat.Rows(); ++ii) {
      for (int jj = 0; jj < ii; ++jj) {
        mat(ii, jj) = mat(jj, ii);
      }
    }
  };

  // Compile-time, col-major matrix.
  {
    constexpr int m = 8;
    Matrix<Scalar, m, m> C;
    C.SetRandom(1);
    if (kTestSymmetricMatrix) {
      symmetrizeDenseMatrix(C);
    }
    TestExactPreconditionerOnMatrix<Scalar, PrecType>(C);
  }

  // Dynamic, row-major matrix.
  {
    constexpr int m = 12;
    RowMatrix<Scalar> C(m, m);
    C.SetRandom(2);
    if (kTestSymmetricMatrix) {
      symmetrizeDenseMatrix(C);
    }
    TestExactPreconditionerOnMatrix<Scalar, PrecType>(C);
  }

  // Sparse matrix.
  {
    constexpr int m = 6;
    DynamicArray<int> rp{0, 2, 5, 8, 11, 14, 16};
    DynamicArray<int> ci{0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5};
    DynamicArray<Scalar> va{2, -1, -1, 3, -1, -1, 4, -1, -1, 2, -1, -1, 2, -1, -1, 2};
    if (!kTestSymmetricMatrix) {
      for (int i = 0; i < isize(va); ++i) {
        va[i] += static_cast<Scalar>(i);
      }
    }
    SparseMatrix<Scalar, int, int> C(m, rp, ci, va);
    TestExactPreconditionerOnMatrix<Scalar, PrecType>(C);
    // Block sparse matrix with block size = 2.
    {
      auto bSpC = ToBlockSparseMatrix<2>(C);
      TestExactPreconditionerOnMatrix<Scalar, PrecType>(bSpC);
    }

    // Block sparse matrix with block size = 3.
    {
      auto bSpC = ToBlockSparseMatrix<3>(C);
      TestExactPreconditionerOnMatrix<Scalar, PrecType>(bSpC);
    }
  }
}

} // namespace mochi::test
