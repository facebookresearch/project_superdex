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
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/utils/sparsity_utils.h>
#include <mochi_core/utils/task_scheduler.h>

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <vector>

#include "data/krylov_solver_test_data.h"

namespace mochi {

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
// Shared constants
// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

template <typename Scalar>
struct KrylovTestConstants {
  static constexpr Scalar kAbsTol = Scalar(1e-17);
  static constexpr Scalar kRelDivTol = Scalar(1e10);
  static constexpr Scalar kCondMat[2] = {Scalar(1712), Scalar(2.4104e05)};
};

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
// Test data
// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

template <typename Scalar>
struct TestProblem {
  SparseMatrix<Scalar> matrix;
  ColumnVector<Scalar> rhs;
  ColumnVector<Scalar> solution;
};

template <typename Scalar>
static TestProblem<Scalar> GetTestProblem(int testId) {
  // Test 0: Tri-diagonal matrix [-1, 2, -1] of size 64 x 64
  //         Condition number ~ 1712
  // Test 1: Sparse matrix of size 100 x 100
  //         Condition number ~ 2.4e+05
  MOCHI_ASSERT(testId == 0 || testId == 1, "Invalid test ID.");

  DynamicArray<int> rowPtr;
  DynamicArray<int> colIdx;
  DynamicArray<Scalar> values;
  DynamicArray<Scalar> bValues;
  DynamicArray<Scalar> solValues;
  int matrixSize = 0;

  if (testId == 1) {
    matrixSize = kPsdMatrixSize[0];

    rowPtr.resize(matrixSize + 1, 0);
    rowPtr[matrixSize] = kPsdMatrixNnz;
    int ir = kPsdMatrixRows[0];
    for (int ipos = 0; ipos < kPsdMatrixNnz; ++ipos) {
      if (kPsdMatrixRows[ipos] > ir) {
        rowPtr[ir + 1] = ipos;
        ir = kPsdMatrixRows[ipos];
      }
    }

    colIdx.assign(kPsdMatrixCols, kPsdMatrixCols + std::size(kPsdMatrixCols));
    values.assign(kPsdMatrixVals, kPsdMatrixVals + std::size(kPsdMatrixVals));
    bValues.assign(kPsdMatrixRhs, kPsdMatrixRhs + std::size(kPsdMatrixRhs));
    solValues.assign(kPsdMatrixOut, kPsdMatrixOut + std::size(kPsdMatrixOut));

  } else {
    matrixSize = 64;
    rowPtr.resize(matrixSize + 1, 0);
    colIdx.reserve(3 * matrixSize);
    values.reserve(3 * matrixSize);

    rowPtr[1] = 2;
    colIdx.push_back(0);
    values.push_back(Scalar(2.0));
    colIdx.push_back(1);
    values.push_back(Scalar(-1.0));

    for (int ii = 1; ii < matrixSize - 1; ++ii) {
      colIdx.push_back(ii - 1);
      values.push_back(Scalar(-1.0));
      colIdx.push_back(ii);
      values.push_back(2.0);
      colIdx.push_back(ii + 1);
      values.push_back(Scalar(-1.0));
      rowPtr[ii + 1] = rowPtr[ii] + 3;
    }

    colIdx.push_back(matrixSize - 2);
    values.push_back(Scalar(-1.0));
    colIdx.push_back(matrixSize - 1);
    values.push_back(Scalar(2.0));
    rowPtr[matrixSize] = rowPtr[matrixSize - 1] + 2;

    bValues.resize(matrixSize, 1.0);

    solValues.resize(matrixSize);
    solValues[0] = Scalar(32.0);
    auto shift = Scalar(31);
    for (int ii = 1; ii < matrixSize; ++ii) {
      solValues[ii] = solValues[ii - 1] + shift;
      shift -= 1;
    }
  }

  return {
      .matrix =
          SparseMatrix<Scalar>(matrixSize, std::move(rowPtr), std::move(colIdx), std::move(values)),
      .rhs = ColumnVector<Scalar>(ColumnVectorView<Scalar const>(bValues.data(), matrixSize)),
      .solution =
          ColumnVector<Scalar>(ColumnVectorView<Scalar const>(solValues.data(), matrixSize))};
}

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
// Matrix / vector helpers
// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

template <typename MatrixType>
auto MakeMatrixOperator(MatrixType const& A) {
  return [&A](auto const& x, auto& Ax) { Ax = A * x; };
}

template <typename Scalar>
static Matrix<Scalar> BuildUpperTriangularOnesMatrix(int n) {
  auto A = Matrix<Scalar>::Zero(n, n);
  for (int ii = 0; ii < n; ++ii) {
    for (int jj = ii; jj < n; ++jj) {
      A(ii, jj) = Scalar(1);
    }
  }
  return A;
}

template <typename Scalar>
static SparseMatrix<Scalar> MakeLaplacianSparseMatrix(int nx, int ny) {
  int n = nx * ny;
  DynamicArray<int> Ar(5 * n), Ac(5 * n);
  DynamicArray<Scalar> Av(5 * n);
  int nnz = 0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      int row = i + j * nx;
      if (j > 0) {
        Ar[nnz] = row;
        Ac[nnz] = row - nx;
        Av[nnz++] = Scalar(-1);
      }
      if (i > 0) {
        Ar[nnz] = row;
        Ac[nnz] = row - 1;
        Av[nnz++] = Scalar(-1);
      }
      Ar[nnz] = row;
      Ac[nnz] = row;
      Av[nnz++] = Scalar(4);
      if (i + 1 < nx) {
        Ar[nnz] = row;
        Ac[nnz] = row + 1;
        Av[nnz++] = Scalar(-1);
      }
      if (j + 1 < ny) {
        Ar[nnz] = row;
        Ac[nnz] = row + nx;
        Av[nnz++] = Scalar(-1);
      }
    }
  }

  std::vector<NdArray<int, 2>> pattern;
  pattern.reserve(nnz);
  for (int i = 0; i < nnz; ++i) {
    pattern.emplace_back(Ar[i], Ac[i]);
  }

  SparseMatrix<Scalar> mat(n, MakeSparsityGraph(std::move(pattern)));
  std::copy(Av.data(), Av.data() + nnz, mat.Values().data());
  return mat;
}

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
// Preconditioner helpers
// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

inline auto IdentityPreconditioner() {
  return [](auto const& x, auto& Px) { Px = x; };
}

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
// Scheduler helpers
// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

inline std::unique_ptr<TaskScheduler> SetupScheduler(bool singleThreadedMode) {
  auto const numThreads = TaskScheduler::GetNumSupportedLogicalProcessors();
  auto scheduler = std::make_unique<TaskScheduler>(numThreads);
  EXPECT_EQ(numThreads, scheduler->GetNumThreads());
  if (singleThreadedMode) {
    scheduler->SetGlobalSingleThreadedMode(true);
  }
  return scheduler;
}

} // namespace mochi
