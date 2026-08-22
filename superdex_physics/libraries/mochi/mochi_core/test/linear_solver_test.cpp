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

#include <mochi_core/linear_algebra/krylov/iteration_status.h>
#include <mochi_core/test/mochi_test_helpers.h>

#include <gtest/gtest.h>

#include <limits>

#include "krylov_solver_test_helpers.h"

using namespace mochi;

TEST(KrylovSolver, IsCudaSolver) {
  EXPECT_FALSE(details::IsCudaSolver(LinearSolverType::CG));
  EXPECT_FALSE(details::IsCudaSolver(LinearSolverType::GMRES));
  EXPECT_FALSE(details::IsCudaSolver(LinearSolverType::AugmentedCG));
  EXPECT_FALSE(details::IsCudaSolver(LinearSolverType::LDLT));
  EXPECT_FALSE(details::IsCudaSolver(LinearSolverType::LU));
  EXPECT_FALSE(details::IsCudaSolver(LinearSolverType::AsyncCG));
  EXPECT_FALSE(details::IsCudaSolver(LinearSolverType::ParallelCG));
  EXPECT_FALSE(details::IsCudaSolver(LinearSolverType::MINRES));
  EXPECT_TRUE(details::IsCudaSolver(LinearSolverType::CudaCG));
  EXPECT_TRUE(details::IsCudaSolver(LinearSolverType::CudaGMRES));
  EXPECT_TRUE(details::IsCudaSolver(LinearSolverType::ExperimentalCudaSparseCholesky));
  EXPECT_TRUE(details::IsCudaSolver(LinearSolverType::ExperimentalCudaSparseLDLT));
  EXPECT_TRUE(details::IsCudaSolver(LinearSolverType::ExperimentalCudaSparseLU));
  static_assert(
      static_cast<int>(LinearSolverType::Count) == 14,
      "Please update this unit test if LinearSolverType enumerator changes");
}

TEST(KrylovSolver, IsIterativeSolver) {
  EXPECT_TRUE(details::IsIterativeSolver(LinearSolverType::CG));
  EXPECT_TRUE(details::IsIterativeSolver(LinearSolverType::GMRES));
  EXPECT_TRUE(details::IsIterativeSolver(LinearSolverType::AugmentedCG));
  EXPECT_FALSE(details::IsIterativeSolver(LinearSolverType::LDLT));
  EXPECT_FALSE(details::IsIterativeSolver(LinearSolverType::LU));
  EXPECT_TRUE(details::IsIterativeSolver(LinearSolverType::AsyncCG));
  EXPECT_TRUE(details::IsIterativeSolver(LinearSolverType::ParallelCG));
  EXPECT_TRUE(details::IsIterativeSolver(LinearSolverType::MINRES));
  EXPECT_TRUE(details::IsIterativeSolver(LinearSolverType::CudaCG));
  EXPECT_TRUE(details::IsIterativeSolver(LinearSolverType::CudaGMRES));
  EXPECT_FALSE(details::IsIterativeSolver(LinearSolverType::ExperimentalCudaSparseCholesky));
  EXPECT_FALSE(details::IsIterativeSolver(LinearSolverType::ExperimentalCudaSparseLDLT));
  EXPECT_FALSE(details::IsIterativeSolver(LinearSolverType::ExperimentalCudaSparseLU));
  static_assert(
      static_cast<int>(LinearSolverType::Count) == 14,
      "Please update this unit test if LinearSolverType enumerator changes");
}

TEST(KrylovSolver, IsConverged) {
  static_assert(!krylov::IsConverged(krylov::IterationStatus::Active));
  static_assert(krylov::IsConverged(krylov::IterationStatus::ConvergedAtol));
  static_assert(krylov::IsConverged(krylov::IterationStatus::ConvergedRtol));
  static_assert(!krylov::IsConverged(krylov::IterationStatus::DivergedRes));
  static_assert(
      static_cast<int>(krylov::IterationStatus::Count) == 4,
      "Please update unit tests if IterationStatus enumerator changes");
}

static void LinearSolve_PsdMatrix(PreconditionerType precType) {
  static_assert(
      static_cast<int>(LinearSolverType::Count) == 14,
      "Please update this unit test if LinearSolverType enumerator changes");

  auto problem = GetTestProblem<real>(1);
  int size = problem.matrix.Rows();
  auto const& inputMat = problem.matrix;

  // Build RHS and expected output vectors.
  auto const& inputVec = problem.rhs;
  auto const& expected = problem.solution;

  // Build computed output vector.
  ColumnVector<real> outputVec(size);

  // Build params and solve.
  KrylovSolverParams params;
  params.preconditionerType = precType;
  params.absTol = 1.e-20;
  params.relTol =
      10.0 * std::numeric_limits<float>::epsilon(); // Use float because the input data are in float
  params.relDivTol = 1.e17;
  params.maxIter = 10000;

  real constexpr kErrorRelTol = 2e-2_r;

  {
    LinearSolver<real> pcgSolver(params);
    KrylovSolverParams cgParams(params);
    cgParams.solverType = LinearSolverType::CG;
    pcgSolver.SetParams(cgParams); // Set params through 'SetParams'.
    EXPECT_EQ(pcgSolver.GetParams().solverType, LinearSolverType::CG);
    outputVec.SetZero();
    auto pcgStatus = pcgSolver.Solve(inputMat, inputVec, outputVec);
    EXPECT_TRUE(pcgStatus.converged);
    ColumnVector<real> delta = expected - outputVec;
    EXPECT_NEAR_TOL(delta.Norm(), 0_r, kErrorRelTol * expected.Norm());
  }

  {
    KrylovSolverParams cgPrecInnerParams(params);
    cgPrecInnerParams.solverType = LinearSolverType::CG;
    cgPrecInnerParams.normType = LinearSolverConvergenceNorm::ResidualPreconditionerInduced;
    LinearSolver<real> pcgPrecInnerSolver(cgPrecInnerParams);
    outputVec.SetZero();
    auto pcgPrecInnerStatus = pcgPrecInnerSolver.Solve(inputMat, inputVec, outputVec);
    EXPECT_TRUE(pcgPrecInnerStatus.converged);
    ColumnVector<real> delta = expected - outputVec;
    EXPECT_NEAR_TOL(delta.Norm(), 0_r, kErrorRelTol * expected.Norm());
  }

  {
    KrylovSolverParams gmresParams(params);
    gmresParams.solverType = LinearSolverType::GMRES;
    LinearSolver<real> gmresSolver(gmresParams); // Set params through the constructor.
    EXPECT_EQ(gmresSolver.GetParams().solverType, LinearSolverType::GMRES);
    outputVec.SetZero();
    auto gmresStatus = gmresSolver.Solve(inputMat, inputVec, outputVec);
    EXPECT_TRUE(gmresStatus.converged);
    ColumnVector<real> delta = expected - outputVec;
    EXPECT_NEAR_TOL(delta.Norm(), 0_r, kErrorRelTol * expected.Norm());
  }

#if MOCHI_USE_EIGEN // AugmentedCG implementation requires MOCHI_USE_EIGEN
  {
    //--- Test Augmented PCG without compression
    LinearSolver<real> dcgSolver(params);
    KrylovSolverParams dcgParams(params);
    dcgParams.solverType = LinearSolverType::AugmentedCG;
    dcgParams.maxSubspaceSize = 40;
    dcgParams.numRecyclingDir = 8;
    dcgParams.normType = LinearSolverConvergenceNorm::ResidualL2;
    dcgSolver.SetParams(dcgParams); // Set params through 'SetParams'.
    EXPECT_EQ(dcgSolver.GetParams().solverType, LinearSolverType::AugmentedCG);
    int initNumIter = -1;
    for (int ii = 0; ii < 4; ++ii) {
      outputVec.SetZero();
      auto dcgStatus =
          dcgSolver.Solve(inputMat, inputVec, outputVec, /* hasOperatorChanged */ ii % 2 == 0);
      ColumnVector<real> delta = expected - outputVec;
      EXPECT_NEAR_TOL(delta.Norm(), 0_r, kErrorRelTol * expected.Norm());
      if (ii == 0) {
        initNumIter = dcgStatus.numIterDone;
      } else {
        EXPECT_LE(dcgStatus.numIterDone, Max(0, initNumIter - ii * dcgParams.numRecyclingDir));
      }
    }
  }
#endif // MOCHI_USE_EIGEN

  {
    //--- Test the dense LDLT solver
    KrylovSolverParams ldltParams(params);
    ldltParams.solverType = LinearSolverType::LDLT;
    ldltParams.preconditionerType = PreconditionerType::None;
    LinearSolver<real> ldltSolver(ldltParams); // Set params through the constructor.
    EXPECT_EQ(ldltSolver.GetParams().solverType, LinearSolverType::LDLT);
    auto denseMat = ToMatrix(inputMat);
    for (int ii = 0; ii < 3; ++ii) {
      outputVec.SetZero();
      auto const ldltStatus =
          ldltSolver.Solve(denseMat, inputVec, outputVec, /* hasOperatorChanged */ ii % 2 == 0);
      ColumnVector<real> delta = expected - outputVec;
      EXPECT_TRUE(ldltStatus.converged);
      EXPECT_NEAR_TOL(delta.Norm(), 0_r, kErrorRelTol * expected.Norm());
    }
  }

  {
    //--- Test the sparse LDLT solver
    KrylovSolverParams ldltParams(params);
    ldltParams.solverType = LinearSolverType::LDLT;
    ldltParams.preconditionerType = PreconditionerType::None;
    LinearSolver<real> ldltSolver(ldltParams); // Set params through the constructor.
    EXPECT_EQ(ldltSolver.GetParams().solverType, LinearSolverType::LDLT);
    for (int ii = 0; ii < 3; ++ii) {
      outputVec.SetZero();
      auto const ldltStatus =
          ldltSolver.Solve(inputMat, inputVec, outputVec, /* hasOperatorChanged */ ii % 2 == 0);
      ColumnVector<real> delta = expected - outputVec;
      EXPECT_TRUE(ldltStatus.converged);
      EXPECT_NEAR_TOL(delta.Norm(), 0_r, kErrorRelTol * expected.Norm());
    }
  }

  {
    //--- Test the dense LU solver
    KrylovSolverParams luParams(params);
    luParams.solverType = LinearSolverType::LU;
    luParams.preconditionerType = PreconditionerType::None;
    LinearSolver<real> luSolver(luParams); // Set params through the constructor.
    EXPECT_EQ(luSolver.GetParams().solverType, LinearSolverType::LU);
    for (int ii = 0; ii < 3; ++ii) {
      outputVec.SetZero();
      auto const luStatus =
          luSolver.Solve(inputMat, inputVec, outputVec, /* hasOperatorChanged */ ii % 2 == 0);
      ColumnVector<real> delta = expected - outputVec;
      EXPECT_TRUE(luStatus.converged);
      EXPECT_NEAR_TOL(delta.Norm(), 0_r, kErrorRelTol * expected.Norm());
    }
  }
}

TEST(KrylovSolver, LinearSolve_Preconditioner_None) {
  LinearSolve_PsdMatrix(PreconditionerType::None);
}

TEST(KrylovSolver, LinearSolve_Preconditioner_Jacobi) {
  LinearSolve_PsdMatrix(PreconditionerType::Jacobi);
}

TEST(KrylovSolver, LinearSolve_Preconditioner_SSOR) {
  LinearSolve_PsdMatrix(PreconditionerType::SSOR);
}
