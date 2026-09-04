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

#include <mochi_core/linear_algebra/krylov/async_pcg.h>
#include <mochi_core/linear_algebra/krylov/identity_prec.h>
#include <mochi_core/linear_algebra/krylov/parallel_pcg.h>
#include <mochi_core/linear_algebra/krylov/pcg.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/solvers/krylov_solver.h>
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
static void TestPcg(bool singleThreadedMode) {
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

    auto A = ToMatrix(problem.matrix);
    auto b = problem.rhs;
    auto ref = problem.solution;

    Vector sol(matrixSize);
    auto solView = AsView(sol);

    auto opA = MakeMatrixOperator(A);

    auto prec = std::make_unique<krylov::IdentityPrec<Scalar>>(A);
    auto P = details::PrecApplyer<Scalar>{*prec};

    Vector pcgError(matrixSize);

    auto runCommonChecks = [&](Scalar const& resRelTol, Scalar const& solRelTol) {
      pcgError = ref - sol;
      EXPECT_EQ(info.convergence, LinearSolverConvergenceStatus::Converged);
      EXPECT_LE(info.numIterDone, maxIter);
      EXPECT_LT(info.relativeResidualNorm, resRelTol);
      EXPECT_LT(dot.Norm(pcgError), solRelTol * dot.Norm(ref));
    };

    //
    // PCG
    //

    Scalar const pcgResRelTol = Scalar(100) * std::numeric_limits<Scalar>::epsilon();
    Scalar pcgSolRelTol[2] = {pcgResRelTol, static_cast<Scalar>(2.0e-02)};
    StopCriterion pcgStopper{pcgResRelTol, kAbsTol, kRelDivTol};

    // With Polak-Ribiere formula, opA and opP.
    sol.SetZero();
    info = krylov::PCG(
        opA,
        AsView(b),
        solView,
        opP,
        maxIter,
        pcgStopper,
        false,
        VerbosityLevel::Warning,
        true,
        InitialGuessHint::Zero,
        dot,
        factory);
    runCommonChecks(pcgResRelTol, pcgSolRelTol[itest]);
    EXPECT_LE(info.numIterDone, matrixSize);

    // With Fletcher-Reeves formula, A and opP.
    sol.SetZero();
    info = krylov::PCG(
        A,
        b,
        sol,
        opP,
        maxIter,
        pcgStopper,
        /*abortIfNotSpd*/ false,
        VerbosityLevel::Warning,
        /*usePolakRibiere*/ false,
        InitialGuessHint::Zero,
        dot,
        factory);
    runCommonChecks(pcgResRelTol, pcgSolRelTol[itest]);
    EXPECT_LE(info.numIterDone, matrixSize);

    // With the solution as initial guess, opA and P.
    info = krylov::PCG(
        opA,
        b,
        sol,
        P,
        maxIter,
        pcgStopper,
        /*abortIfNotSpd*/ false,
        VerbosityLevel::Warning,
        /*usePolakRibiere*/ true,
        InitialGuessHint::Unknown,
        dot,
        factory);
    runCommonChecks(pcgResRelTol, pcgSolRelTol[itest]);
    EXPECT_LT(info.numIterDone, 1);

    // With arbitrary non-zero initial guess, A and P.
    sol = Scalar(0.3) * ref;
    info = krylov::PCG(
        A,
        b,
        sol,
        P,
        maxIter,
        pcgStopper,
        /*abortIfNotSpd*/ false,
        VerbosityLevel::Warning,
        /*usePolakRibiere*/ true,
        InitialGuessHint::Unknown,
        dot,
        factory);
    runCommonChecks(pcgResRelTol, pcgSolRelTol[itest]);
    EXPECT_LE(info.numIterDone, matrixSize);

    //
    // Parallel PCG
    //

    // With Polak-Ribiere formula.
    sol.SetZero();
    info = krylov::ParallelPCG(
        A,
        AsView(b),
        solView,
        P,
        maxIter,
        pcgStopper,
        false,
        VerbosityLevel::Warning,
        true,
        InitialGuessHint::Zero,
        dot,
        factory);
    runCommonChecks(pcgResRelTol, pcgSolRelTol[itest]);
    EXPECT_LE(info.numIterDone, matrixSize);

    // With Fletcher-Reeves formula.
    sol.SetZero();
    info = krylov::ParallelPCG(
        A,
        b,
        sol,
        P,
        maxIter,
        pcgStopper,
        /*abortIfNotSpd*/ false,
        VerbosityLevel::Warning,
        /*usePolakRibiere*/ false,
        InitialGuessHint::Zero,
        dot,
        factory);
    runCommonChecks(pcgResRelTol, pcgSolRelTol[itest]);
    EXPECT_LE(info.numIterDone, matrixSize);

    // With the solution as initial guess.
    info = krylov::ParallelPCG(
        A,
        b,
        sol,
        P,
        maxIter,
        pcgStopper,
        /*abortIfNotSpd*/ false,
        VerbosityLevel::Warning,
        /*usePolakRibiere*/ true,
        InitialGuessHint::Unknown,
        dot,
        factory);
    runCommonChecks(pcgResRelTol, pcgSolRelTol[itest]);
    EXPECT_LT(info.numIterDone, 1);

    // With an arbitrary non-zero initial guess.
    sol = Scalar(0.3) * ref;
    info = krylov::ParallelPCG(
        A,
        b,
        sol,
        P,
        maxIter,
        pcgStopper,
        /*abortIfNotSpd*/ false,
        VerbosityLevel::Warning,
        /*usePolakRibiere*/ true,
        InitialGuessHint::Unknown,
        dot,
        factory);
    runCommonChecks(pcgResRelTol, pcgSolRelTol[itest]);
    EXPECT_LE(info.numIterDone, matrixSize);

    //
    // Asynchronous PCG
    //

    int const restartPeriod = 15; // Small enough so that restart codepaths are tested
    Scalar const asyncPcgResRelTol =
        Scalar(100) * std::numeric_limits<float>::epsilon(); // Restarting prevents from converging
                                                             // to double-precision epsilon.
    Scalar asyncPcgSolRelTol[2] = {asyncPcgResRelTol, static_cast<Scalar>(2.0e-02)};
    StopCriterion asyncPcgStopper{asyncPcgResRelTol, kAbsTol, kRelDivTol};

    // With Polak-Ribiere formula and opP.
    sol.SetZero();
    info = krylov::AsyncPCG(
        A,
        AsView(b),
        solView,
        opP,
        maxIter,
        asyncPcgStopper,
        false,
        VerbosityLevel::Warning,
        true,
        InitialGuessHint::Zero,
        dot,
        factory,
        restartPeriod);
    runCommonChecks(asyncPcgResRelTol, asyncPcgSolRelTol[itest]);
    EXPECT_GT(info.numIterDone, restartPeriod); // Restart codepaths are tested

    // With Fletcher-Reeves formula, P, and an arbitrary non-zero initial guess.
    sol = Scalar(0.3) * ref;
    info = krylov::AsyncPCG(
        A,
        b,
        sol,
        P,
        maxIter,
        asyncPcgStopper,
        false,
        VerbosityLevel::Warning,
        false,
        InitialGuessHint::Unknown,
        dot,
        factory,
        restartPeriod);
    runCommonChecks(asyncPcgResRelTol, asyncPcgSolRelTol[itest]);
    EXPECT_GT(info.numIterDone, restartPeriod); // Restart codepaths are tested
  } // for (int itest = 0; itest < 2; ++itest)

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
    info = krylov::PCG(
        opB,
        b,
        x,
        opInvB,
        n,
        stopper,
        /*abortIfNotSpd*/ false,
        VerbosityLevel::Warning,
        /*usePolakRibiere*/ true,
        InitialGuessHint::Unknown,
        dot,
        factory);

    EXPECT_EQ(info.convergence, LinearSolverConvergenceStatus::Converged);
    EXPECT_EQ(info.numIterDone, 1);
    EXPECT_LE(info.relativeResidualNorm, relTol);
  }

  {
    //
    // Use a triangular matrix to test the non-convergence
    //
    constexpr int n = 5;
    Matrix A = BuildUpperTriangularOnesMatrix<Scalar>(n);
    auto x = Vector::Zero(n);
    Vector b(n);
    for (int ii = 0; ii < n; ++ii) {
      b[ii] = Scalar(1);
    }
    auto opA = MakeMatrixOperator(A);
    auto localOpP = IdentityPreconditioner();
    StopCriterion defaultStopper{Scalar(1e-6), kAbsTol, kRelDivTol};
    info = krylov::PCG(
        opA,
        b,
        x,
        localOpP,
        n,
        defaultStopper,
        /*abortIfNotSpd*/ false,
        VerbosityLevel::Warning,
        /*usePolakRibiere*/ true,
        InitialGuessHint::Unknown,
        dot,
        factory);
    EXPECT_EQ(info.numIterDone, n);
    EXPECT_EQ(info.convergence, LinearSolverConvergenceStatus::Stopped);
  }
}

TEST(KrylovSolver, Pcg) {
  // Use "real" instead of float or double to improve compiler performance. Both are checked by CI.
  // clang-format off
  TestPcg<real, krylov::StatusResidualL2<krylov::UsualDot, real>, krylov::UsualDot>( /*singleThreadedMode*/ true);
  TestPcg<real, krylov::StatusPreconditionedResidualL2<krylov::UsualDot, real>, krylov::UsualDot>( /*singleThreadedMode*/ true);
  TestPcg<real, krylov::StatusResidualPreconditionerInduced<krylov::UsualDot, real>, krylov::UsualDot>( /*singleThreadedMode*/ true);
  TestPcg<real, krylov::StatusResidualL2<krylov::UsualDot, real>, krylov::UsualDot>( /*singleThreadedMode*/ false);
  TestPcg<real, krylov::StatusPreconditionedResidualL2<krylov::UsualDot, real>, krylov::UsualDot>( /*singleThreadedMode*/ false);
  TestPcg<real, krylov::StatusResidualPreconditionerInduced<krylov::UsualDot, real>, krylov::UsualDot>( /*singleThreadedMode*/ false);
  // clang-format on
}

TEST(KrylovSolver, Pcg_InitialGuessHint) {
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
    auto const status = krylov::PCG(
        opA,
        b,
        x,
        opP,
        kSize,
        stopCriterion,
        /*abortIfNotSpd*/ false,
        VerbosityLevel::Silent,
        /*usePolakRibiere*/ true,
        initialGuessHint);

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

TEST(KrylovSolver, Pcg_ZeroRhsWithKnownZeroHint) {
  constexpr int kSize = 4;
  using Vector = ColumnVector<real>;

  auto const b = Vector::Zero(kSize);
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
  krylov::StatusPreconditionedResidualL2<krylov::UsualDot, real> stopCriterion{1e-6_r, 0_r, 1e10_r};

  auto const status = krylov::PCG(
      opA,
      b,
      x,
      opP,
      kSize,
      stopCriterion,
      /*abortIfNotSpd*/ false,
      VerbosityLevel::Silent,
      /*usePolakRibiere*/ true,
      InitialGuessHint::Zero);

  EXPECT_EQ(status.convergence, LinearSolverConvergenceStatus::Converged);
  EXPECT_EQ(status.numIterDone, 0);
  EXPECT_EQ(operatorApplications, 0);
  EXPECT_EQ(preconditionerApplications, 1);
  EXPECT_EQ(x.Norm(), 0_r);
}
