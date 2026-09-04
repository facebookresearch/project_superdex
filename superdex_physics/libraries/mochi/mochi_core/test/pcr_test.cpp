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

#include <mochi_core/linear_algebra/krylov/pcr.h>

#include <mochi_core/linear_algebra/krylov/identity_prec.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/solvers/linear_solver.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/task_scheduler.h>

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <vector>

#include "krylov_solver_test_helpers.h"

using namespace mochi;

template <typename Scalar, typename StopCriterion, typename Dot>
static void TestPcr(bool singleThreadedMode) {
  using Matrix = Matrix<Scalar>;
  using Vector = ColumnVector<Scalar>;

  Scalar constexpr kAbsTol = KrylovTestConstants<Scalar>::kAbsTol;
  Scalar constexpr kRelDivTol = KrylovTestConstants<Scalar>::kRelDivTol;

  auto scheduler = SetupScheduler(singleThreadedMode);

  Dot dot{};
  auto opP = IdentityPreconditioner();

  LinearSolverStatus info;
  auto factory = krylov::MatrixFactoryType<Vector>{};

  for (int itest = 0; itest < 2; ++itest) {
    auto problem = GetTestProblem<Scalar>(itest);
    int const matrixSize = problem.matrix.Rows();
    int const maxIter = 1000;

    auto b = problem.rhs;
    auto ref = problem.solution;

    Vector sol(matrixSize);
    auto solView = AsView(sol);

    auto A = ToMatrix(problem.matrix);
    auto opA = MakeMatrixOperator(A);

    auto prec = std::make_unique<krylov::IdentityPrec<Scalar>>(A);
    auto P = details::PrecApplyer<Scalar>{*prec};

    Vector pcrError(matrixSize);

    auto runCommonChecks = [&](Scalar const& resRelTol, Scalar const& solRelTol) {
      pcrError = ref - sol;
      EXPECT_EQ(info.convergence, LinearSolverConvergenceStatus::Converged);
      EXPECT_LE(info.numIterDone, maxIter);
      EXPECT_LT(info.relativeResidualNorm, resRelTol);
      EXPECT_LT(dot.Norm(pcrError), solRelTol * dot.Norm(ref));
    };

    //
    // PCR
    //

    Scalar const resRelTol = Scalar(100) * std::numeric_limits<Scalar>::epsilon();
    Scalar solRelTol[2] = {resRelTol, static_cast<Scalar>(2.0e-02)};
    StopCriterion stopper{resRelTol, kAbsTol, kRelDivTol};

    // With A and opP.
    sol.SetZero();
    info = krylov::PCR(
        A,
        AsView(b),
        solView,
        opP,
        maxIter,
        stopper,
        VerbosityLevel::Warning,
        InitialGuessHint::Zero,
        dot,
        factory);
    runCommonChecks(resRelTol, solRelTol[itest]);
    EXPECT_LE(info.numIterDone, matrixSize);

    // With the solution as initial guess, opA and P.
    info = krylov::PCR(
        opA,
        b,
        sol,
        P,
        maxIter,
        stopper,
        VerbosityLevel::Warning,
        InitialGuessHint::Unknown,
        dot,
        factory);
    runCommonChecks(resRelTol, solRelTol[itest]);
    EXPECT_LT(info.numIterDone, 1);

    // With arbitrary non-zero initial guess, A and P.
    sol = Scalar(0.3) * ref;
    info = krylov::PCR(
        A,
        b,
        sol,
        P,
        maxIter,
        stopper,
        VerbosityLevel::Warning,
        InitialGuessHint::Unknown,
        dot,
        factory);
    runCommonChecks(resRelTol, solRelTol[itest]);
    EXPECT_LE(info.numIterDone, matrixSize);
  }

  {
    //
    // Use the exact inverse as preconditioner to test convergence in 1 iteration
    //
    int const n = 10;

    Matrix A(n, n);
    Vector b(n), x(n);
    A.SetRandom(1);
    b.SetRandom(2);
    x.SetZero();

    Matrix B = A * Transpose(A);
    Matrix invB = SymInverse(B);
    auto opInvB = [&invB](auto const& x, auto& Px) { Px = invB * x; };
    auto opB = [&B](auto const& x, auto& Bx) { Bx = B * x; };

    Scalar const relTol = Scalar(10 * n * n) * std::numeric_limits<Scalar>::epsilon();
    StopCriterion stopper{relTol, kAbsTol, kRelDivTol};
    info = krylov::PCR(
        opB,
        b,
        x,
        opInvB,
        n,
        stopper,
        VerbosityLevel::Warning,
        InitialGuessHint::Unknown,
        dot,
        factory);

    EXPECT_EQ(info.convergence, LinearSolverConvergenceStatus::Converged);
    EXPECT_EQ(info.numIterDone, 1);
    EXPECT_LE(info.relativeResidualNorm, relTol);
  }
}

TEST(KrylovSolver, Pcr) {
  TestPcr<real, krylov::StatusResidualL2<krylov::UsualDot, real>, krylov::UsualDot>(
      /*singleThreadedMode*/ true);
  TestPcr<real, krylov::StatusResidualL2<krylov::UsualDot, real>, krylov::UsualDot>(
      /*singleThreadedMode*/ false);
  TestPcr<real, krylov::StatusPreconditionedResidualL2<krylov::UsualDot, real>, krylov::UsualDot>(
      /*singleThreadedMode*/ true);
  TestPcr<real, krylov::StatusPreconditionedResidualL2<krylov::UsualDot, real>, krylov::UsualDot>(
      /*singleThreadedMode*/ false);
}

TEST(KrylovSolver, Pcr_InitialGuessHint) {
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
  int preconditionerApplications = 0;
  auto opP = [&](auto const& input, auto& output) {
    ++preconditionerApplications;
    output = input;
  };

  auto runSolve = [&](InitialGuessHint initialGuessHint) {
    x.SetZero();
    operatorApplications = 0;
    preconditionerApplications = 0;
    krylov::StatusPreconditionedResidualL2<krylov::UsualDot, real> stopCriterion{
        10_r * std::numeric_limits<real>::epsilon(), 0_r, 1e10_r};
    auto const status =
        krylov::PCR(opA, b, x, opP, kSize, stopCriterion, VerbosityLevel::Silent, initialGuessHint);

    EXPECT_EQ(status.convergence, LinearSolverConvergenceStatus::Converged);
    ColumnVector<real> error = b - x;
    EXPECT_NEAR_TOL(error.Norm(), 0_r, 10_r * std::numeric_limits<real>::epsilon() * b.Norm());
  };

  runSolve(InitialGuessHint::Unknown);
  int const generalOperatorApplications = operatorApplications;
  int const generalPreconditionerApplications = preconditionerApplications;
  runSolve(InitialGuessHint::Zero);

  EXPECT_EQ(generalOperatorApplications, operatorApplications + 1);
  EXPECT_EQ(generalPreconditionerApplications, preconditionerApplications + 1);
}
