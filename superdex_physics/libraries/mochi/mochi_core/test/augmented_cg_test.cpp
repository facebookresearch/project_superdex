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

#include <mochi_core/solvers/linear_solver.h>
#include <mochi_core/test/mochi_test_helpers.h>

#include <gtest/gtest.h>

#include <limits>

#include "krylov_solver_test_helpers.h"

using namespace mochi;

// AugmentedCG implementation requires MOCHI_USE_EIGEN
TEST_IF(MOCHI_USE_EIGEN, KrylovSolver, AugmentedCG) {
  using Scalar = real;

  int nx = 20, ny = 20, n = nx * ny;
  auto inputMat = MakeLaplacianSparseMatrix<Scalar>(nx, ny);

  // Build expected output vector.
  ColumnVector<Scalar> expected(n);
  expected.SetRandom(123);

  // Build computed output vector.
  ColumnVector<Scalar> outputVec(n);

  // Build RHS vector.
  ColumnVector<Scalar> inputVec(n);
  inputMat.Apply(expected, inputVec);

  // Build params and solve.
  KrylovSolverParams params;
  params.preconditionerType = PreconditionerType::SSOR;
  params.absTol = 1.e-20;
  params.relTol = 10.0 * std::numeric_limits<Scalar>::epsilon();
  params.relDivTol = 1.e17;

  auto constexpr kErrorRelTol = Scalar(5e-06);

  {
    //--- Test Augmented PCG without compression
    KrylovSolverParams dcgParams(params);
    dcgParams.solverType = LinearSolverType::AugmentedCG;
    dcgParams.preconditionerType = params.preconditionerType;
    dcgParams.maxSubspaceSize = 32;
    dcgParams.numRecyclingDir = 8;
    dcgParams.normType = LinearSolverConvergenceNorm::ResidualL2;
    LinearSolver<Scalar> dcgSolver(dcgParams);
    EXPECT_EQ(dcgSolver.GetParams().solverType, LinearSolverType::AugmentedCG);
    int initNumIter = -1;
    for (int ii = 0; ii < 3; ++ii) {
      outputVec.SetZero();
      auto dcgStatus =
          dcgSolver.Solve(inputMat, inputVec, outputVec, /* hasOperatorChanged */ ii % 2 == 0);
      ColumnVector<Scalar> delta = expected - outputVec;
      EXPECT_NEAR_TOL(delta.Norm(), Scalar(0), kErrorRelTol * expected.Norm());
      if (ii == 0) {
        initNumIter = dcgStatus.numIterDone;
      } else {
        EXPECT_LE(dcgStatus.numIterDone, Max(0, initNumIter - ii * dcgParams.numRecyclingDir));
      }
    }
  }

  {
    //--- Test Augmented PCG with compression
    KrylovSolverParams dcgParams(params);
    dcgParams.solverType = LinearSolverType::AugmentedCG;
    dcgParams.preconditionerType = params.preconditionerType;
    dcgParams.maxSubspaceSize = 16;
    dcgParams.numRecyclingDir = 8;
    dcgParams.normType = LinearSolverConvergenceNorm::ResidualL2;
    LinearSolver<Scalar> dcgSolver(dcgParams);
    EXPECT_EQ(dcgSolver.GetParams().solverType, LinearSolverType::AugmentedCG);
    int initNumIter = -1;
    for (int ii = 0; ii < 3; ++ii) {
      outputVec.SetZero();
      auto dcgStatus =
          dcgSolver.Solve(inputMat, inputVec, outputVec, /* hasOperatorChanged */ ii % 2 == 0);
      ColumnVector<Scalar> delta = expected - outputVec;
      EXPECT_NEAR_TOL(delta.Norm(), Scalar(0), kErrorRelTol * expected.Norm());
      if (ii == 0) {
        initNumIter = dcgStatus.numIterDone;
      } else {
        EXPECT_LE(dcgStatus.numIterDone, Max(0, initNumIter - dcgParams.numRecyclingDir));
      }
    }
  }
}
