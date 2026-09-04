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

#include <mochi_core/linear_algebra/krylov/gmres.h>

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/task_scheduler.h>

#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "krylov_solver_test_helpers.h"

using namespace mochi;

template <typename Scalar, typename StopCriterion, typename Dot>
static void TestGmres(bool singleThreadedMode) {
  using Matrix = Matrix<Scalar>;
  using Vector = ColumnVector<Scalar>;

  Scalar constexpr kAbsTol = KrylovTestConstants<Scalar>::kAbsTol;
  Scalar constexpr kRelDivTol = KrylovTestConstants<Scalar>::kRelDivTol;

  auto scheduler = SetupScheduler(singleThreadedMode);

  auto opP = IdentityPreconditioner();
  Dot dot{};

  for (int itest = 0; itest < 2; ++itest) {
    auto problem = GetTestProblem<Scalar>(itest);
    int const matrixSize = problem.matrix.Rows();
    auto A = ToMatrix(problem.matrix);

    auto b = problem.rhs;
    auto ref = problem.solution;

    auto sol = Vector::Zero(matrixSize);
    auto solView = AsView(sol);

    //--- Input data is specified with 7 digits of accuracy (hence -> float)
    auto relativeTol = Scalar(std::numeric_limits<float>::epsilon() * Scalar(matrixSize));
    Scalar solTol = relativeTol * KrylovTestConstants<Scalar>::kCondMat[itest];

    //--- GMRes does not include a restart
    int maxIter = matrixSize;

    auto opA = MakeMatrixOperator(A);
    auto info = krylov::GMRes(
        opA,
        b,
        sol,
        opP,
        maxIter,
        StopCriterion{relativeTol, kAbsTol, kRelDivTol},
        maxIter,
        VerbosityLevel::Warning,
        InitialGuessHint::Zero,
        dot);

    int fullIter = info.numIterDone;
    EXPECT_LT(info.numIterDone, maxIter);
    EXPECT_LT(info.relativeResidualNorm, relativeTol);
    Vector gmres_error(matrixSize);
    gmres_error = ref - sol;
    EXPECT_LT(dot.Norm(gmres_error), solTol * dot.Norm(ref));

    //
    // Use the reference as initial guess
    // GMRes should not do any iteration
    //

    sol = ref;
    info = krylov::GMRes(
        A,
        AsView(b),
        solView,
        opP,
        maxIter,
        StopCriterion{relativeTol, kAbsTol, kRelDivTol},
        maxIter,
        VerbosityLevel::Warning,
        InitialGuessHint::Unknown,
        dot);

    EXPECT_LT(info.numIterDone, 1);
    EXPECT_LT(info.relativeResidualNorm, relativeTol);

    //
    // Use a non-zero initial guess
    //
    sol = Scalar(0.4) * ref;
    info = krylov::GMRes(
        A,
        b,
        sol,
        opP,
        maxIter,
        StopCriterion{relativeTol, kAbsTol, kRelDivTol},
        maxIter,
        VerbosityLevel::Warning,
        InitialGuessHint::Unknown,
        dot);

    EXPECT_LT(info.numIterDone, maxIter);
    EXPECT_LT(info.relativeResidualNorm, relativeTol);
    gmres_error = ref - sol;
    EXPECT_LT(dot.Norm(gmres_error), solTol * dot.Norm(ref));

    // Activate a restart after original convergence
    krylov::SetZero(sol);
    info = krylov::GMRes(
        opA,
        b,
        sol,
        opP,
        10 * matrixSize,
        StopCriterion{relativeTol, kAbsTol, kRelDivTol},
        fullIter,
        VerbosityLevel::Warning,
        InitialGuessHint::Zero,
        dot);
    EXPECT_EQ(info.numIterDone, fullIter);
    EXPECT_LT(info.relativeResidualNorm, relativeTol);
    gmres_error = ref - sol;
    EXPECT_LT(dot.Norm(gmres_error), solTol * dot.Norm(ref));

    // Activate a restart before original convergence
    if (fullIter > 1) {
      krylov::SetZero(sol);
      info = krylov::GMRes(
          opA,
          b,
          sol,
          opP,
          10 * matrixSize,
          StopCriterion{relativeTol, kAbsTol, kRelDivTol},
          (3 * fullIter) / 4,
          VerbosityLevel::Warning,
          InitialGuessHint::Zero,
          dot);
      EXPECT_GE(info.numIterDone, fullIter);
      EXPECT_LT(info.relativeResidualNorm, relativeTol);
      gmres_error = ref - sol;
      EXPECT_LT(dot.Norm(gmres_error), solTol * dot.Norm(ref));
    }

  } // for (int itest = 0; itest < 2; ++itest)

  {
    //
    // Use the exact inverse as preconditioner to test convergence in 1 iteration
    //
    int const n = 10;

    Matrix A(n, n);
    Vector b(n);
    auto x = Vector::Zero(n);
    A.SetRandom(1);
    b.SetRandom(2);

    Matrix B = A * Transpose(A);
    Matrix invB = Inverse(B);
    auto opInvB = [&invB](auto const& x, auto& Px) { Px = invB * x; };
    auto opB = [&B](auto const& x, auto& Bx) { Bx = B * x; };

    Scalar const relTol = Scalar(10 * n * n) * std::numeric_limits<Scalar>::epsilon();
    auto const info = krylov::GMRes(
        opB,
        b,
        x,
        opInvB,
        n,
        StopCriterion{relTol, kAbsTol, kRelDivTol},
        {},
        VerbosityLevel::Warning,
        InitialGuessHint::Zero,
        dot);

    EXPECT_EQ(info.convergence, LinearSolverConvergenceStatus::Converged);
    EXPECT_EQ(info.numIterDone, 1);
    EXPECT_LE(info.relativeResidualNorm, relTol);
  }

  {
    //
    // Use a triangular matrix to test the convergence for a non-symmetric matrix
    //
    constexpr int n = 5;
    auto const relTol = Scalar(1.0e-06);
    Matrix A = BuildUpperTriangularOnesMatrix<Scalar>(n);
    Vector b(n);
    auto x = Vector::Zero(n);
    for (int ii = 0; ii < n; ++ii) {
      b[ii] = Scalar(1);
    }
    auto opA = MakeMatrixOperator(A);
    auto info = krylov::GMRes(
        opA,
        b,
        x,
        opP,
        n,
        StopCriterion{relTol, kAbsTol, kRelDivTol},
        {},
        VerbosityLevel::Warning,
        InitialGuessHint::Zero,
        dot);
    EXPECT_LE(info.numIterDone, n);
    EXPECT_EQ(info.convergence, LinearSolverConvergenceStatus::Converged);
    EXPECT_NEAR_EQ(x[0], Scalar(0));
    EXPECT_NEAR_EQ(x[1], Scalar(0));
    EXPECT_NEAR_EQ(x[2], Scalar(0));
    EXPECT_NEAR_EQ(x[3], Scalar(0));
    EXPECT_NEAR_EQ(x[4], Scalar(1));
  }
}

TEST(KrylovSolver, GMRes) {
  // Use "real" instead of float or double to improve compiler performance. Both are checked by CI.
  TestGmres<real, krylov::StatusImplicitResidualNorm<real>, krylov::UsualDot>(
      /*singleThreadedMode*/ true);
  TestGmres<real, krylov::StatusImplicitResidualNorm<real>, krylov::UsualDot>(
      /*singleThreadedMode*/ false);
}

TEST(KrylovSolver, GMRes_InitialGuessHint) {
  constexpr int kSize = 4;
  using Vector = ColumnVector<real>;

  Vector b(kSize);
  b.SetRandom(1);
  auto x = Vector::Zero(kSize);
  int operatorApplications = 0;
  auto opA = [&](auto const& input, auto& output) {
    ++operatorApplications;
    output = input;
  };
  auto opP = IdentityPreconditioner();

  auto runSolve = [&](InitialGuessHint initialGuessHint) {
    x.SetZero();
    operatorApplications = 0;
    auto const status = krylov::GMRes(
        opA,
        b,
        x,
        opP,
        kSize,
        krylov::StatusImplicitResidualNorm<real>{
            10_r * std::numeric_limits<real>::epsilon(), 0_r, 1e10_r},
        0,
        VerbosityLevel::Silent,
        initialGuessHint);

    EXPECT_EQ(status.convergence, LinearSolverConvergenceStatus::Converged);
    ColumnVector<real> error = b - x;
    EXPECT_NEAR_TOL(error.Norm(), 0_r, 10_r * std::numeric_limits<real>::epsilon() * b.Norm());
  };

  runSolve(InitialGuessHint::Unknown);
  int const generalOperatorApplications = operatorApplications;
  runSolve(InitialGuessHint::Zero);

  EXPECT_EQ(generalOperatorApplications, operatorApplications + 1);
}
