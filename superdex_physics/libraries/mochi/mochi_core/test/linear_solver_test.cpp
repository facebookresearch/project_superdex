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

#include <array>
#include <limits>
#include <utility>

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

TEST(KrylovSolver, EffectiveConvergenceNorm) {
  using Norm = LinearSolverConvergenceNorm;
  using Type = LinearSolverType;
  Norm constexpr kL2 = Norm::ResidualL2, kPrec = Norm::PreconditionedResidualL2,
                 kInduced = Norm::ResidualPreconditionerInduced,
                 kWeighted = Norm::ResidualPerActorWeighted;
  // Effective norm of each requested norm, in enumerator order.
  std::array<Norm, 4> constexpr kRequested = {kL2, kPrec, kInduced, kWeighted};
  std::pair<Type, std::array<Norm, 4>> const kTable[] = {
      {Type::CG, kRequested},
      {Type::ParallelCG, kRequested},
      {Type::AsyncCG, kRequested},
      {Type::LDLT, kRequested}, // Direct solvers don't use residual weights.
      {Type::CudaCG, {kL2, kPrec, kInduced, kPrec}},
      {Type::GMRES, {kL2, kL2, kL2, kL2}},
      {Type::CudaGMRES, {kL2, kL2, kL2, kL2}},
      {Type::AugmentedCG, {kL2, kL2, kL2, kL2}},
      {Type::MINRES, {kInduced, kInduced, kInduced, kInduced}}};
  for (auto const& [solverType, expectedNorms] : kTable) {
    for (int i = 0; i < isize(expectedNorms); ++i) {
      KrylovSolverParams params;
      params.solverType = solverType;
      params.normType = kRequested[i];
      EXPECT_EQ(expectedNorms[i], details::EffectiveConvergenceNorm(params));
      EXPECT_EQ(
          details::IsIterativeSolver(solverType) && expectedNorms[i] == kWeighted,
          details::UsesResidualWeights(params));
    }
  }
  static_assert(
      static_cast<int>(Type::Count) == 14 && static_cast<int>(Norm::Count) == 4,
      "Please update this unit test if either enumerator changes");
}

TEST(KrylovSolver, IsIterationStatusConverged) {
  static_assert(!krylov::IsConverged(krylov::IterationStatus::Active));
  static_assert(krylov::IsConverged(krylov::IterationStatus::ConvergedAtol));
  static_assert(krylov::IsConverged(krylov::IterationStatus::ConvergedRtol));
  static_assert(!krylov::IsConverged(krylov::IterationStatus::DivergedRes));
  static_assert(
      static_cast<int>(krylov::IterationStatus::Count) == 4,
      "Please update unit tests if IterationStatus enumerator changes");
}

TEST(KrylovSolver, IsLinearSolverConverged) {
  static_assert(LinearSolverStatus{}.convergence == LinearSolverConvergenceStatus::None);
  static_assert(!IsConverged(LinearSolverConvergenceStatus::None));
  static_assert(IsConverged(LinearSolverConvergenceStatus::Converged));
  static_assert(!IsConverged(LinearSolverConvergenceStatus::Stopped));
  static_assert(!IsConverged(LinearSolverConvergenceStatus::Diverged));
  static_assert(
      static_cast<int>(LinearSolverConvergenceStatus::Count) == 4,
      "Please update unit tests if LinearSolverConvergenceStatus enumerator changes");
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

  for (auto const normType :
       {LinearSolverConvergenceNorm::ResidualL2,
        LinearSolverConvergenceNorm::PreconditionedResidualL2,
        LinearSolverConvergenceNorm::ResidualPreconditionerInduced,
        LinearSolverConvergenceNorm::ResidualPerActorWeighted}) {
    LinearSolver<real> pcgSolver(params);
    KrylovSolverParams cgParams(params);
    cgParams.solverType = LinearSolverType::CG;
    cgParams.normType = normType;
    pcgSolver.SetParams(cgParams); // Set params through 'SetParams'.
    EXPECT_EQ(pcgSolver.GetParams().solverType, LinearSolverType::CG);
    outputVec = 0.3_r * expected;
    auto pcgStatus = pcgSolver.Solve(inputMat, inputVec, outputVec);
    EXPECT_EQ(pcgStatus.convergence, LinearSolverConvergenceStatus::Converged);
    ColumnVector<real> delta = expected - outputVec;
    EXPECT_NEAR_TOL(delta.Norm(), 0_r, kErrorRelTol * expected.Norm());
  }

  {
    // A heavily weighted row makes the relative residuals in the weighted norm and the L2-norm
    // differ by more than 10x.
    ColumnVector<real> weights(size);
    weights.SetConstant(1_r);
    weights[90] = 1e6_r;
    auto relativeResidual = [&](Span<real const> w) {
      ColumnVector<real> const residual = inputVec - inputMat * outputVec;
      krylov::WeightedDot<real> const dot{w};
      return Sqrt(dot.NormSqr(residual) / dot.NormSqr(inputVec));
    };
    KrylovSolverParams weightedParams(params);
    weightedParams.normType = LinearSolverConvergenceNorm::ResidualPerActorWeighted;
    weightedParams.relTol = 1e-3;
    AnyMatrix<real> const anyMat = inputMat;
    for (auto const solverType :
         {LinearSolverType::CG, LinearSolverType::ParallelCG, LinearSolverType::AsyncCG}) {
      weightedParams.solverType = solverType;
      LinearSolver<real> weightedSolver(weightedParams);
      outputVec.SetZero();
      auto const status = weightedSolver.Solve(
          anyMat,
          inputVec,
          outputVec,
          /*hasOperatorChanged*/ true,
          InitialGuessHint::Zero,
          MakeConstSpan(weights));
      EXPECT_EQ(status.convergence, LinearSolverConvergenceStatus::Converged);
      real const rel = relativeResidual(MakeConstSpan(weights));
      EXPECT_NEAR(rel, status.relativeResidualNorm, 5e-2 * rel);
      EXPECT_GE(relativeResidual({}), 10 * rel);
    }
  }

  {
    KrylovSolverParams gmresParams(params);
    gmresParams.solverType = LinearSolverType::GMRES;
    LinearSolver<real> gmresSolver(gmresParams); // Set params through the constructor.
    EXPECT_EQ(gmresSolver.GetParams().solverType, LinearSolverType::GMRES);
    outputVec.SetZero();
    auto gmresStatus = gmresSolver.Solve(
        inputMat,
        inputVec,
        outputVec,
        /*hasOperatorChanged*/ true,
        InitialGuessHint::Zero);
    EXPECT_EQ(gmresStatus.convergence, LinearSolverConvergenceStatus::Converged);
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
    dcgSolver.SetParams(dcgParams); // Set params through 'SetParams'.
    EXPECT_EQ(dcgSolver.GetParams().solverType, LinearSolverType::AugmentedCG);
    int initNumIter = -1;
    for (int ii = 0; ii < 4; ++ii) {
      outputVec.SetZero();
      auto dcgStatus = dcgSolver.Solve(
          inputMat,
          inputVec,
          outputVec,
          /*hasOperatorChanged*/ ii % 2 == 0,
          InitialGuessHint::Zero);
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
      EXPECT_EQ(ldltStatus.convergence, LinearSolverConvergenceStatus::Converged);
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
    auto const A = MakeLaplacianSparseMatrix<real>(LinearSolver<real>::kSparseLdltDofThreshold, 1);
    ColumnVector<real> xRef(A.Rows()), b(A.Rows()), x(A.Rows());
    xRef.SetRandom(123);
    A.Apply(xRef, b);
    for (int ii = 0; ii < 3; ++ii) {
      x.SetZero();
      auto const ldltStatus = ldltSolver.Solve(A, b, x, /* hasOperatorChanged */ ii % 2 == 0);
      ColumnVector<real> delta = xRef - x;
      EXPECT_EQ(ldltStatus.convergence, LinearSolverConvergenceStatus::Converged);
      EXPECT_NEAR_TOL(delta.Norm(), 0_r, kErrorRelTol * xRef.Norm());
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
      EXPECT_EQ(luStatus.convergence, LinearSolverConvergenceStatus::Converged);
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
