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

#include <mochi_core/linear_algebra/krylov/minres.h>

#include <mochi_core/linear_algebra/krylov/block_jacobi_prec.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/solvers/linear_solver.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/task_scheduler.h>

#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "krylov_solver_test_helpers.h"

using namespace mochi;

template <typename Scalar, typename StopCriterion, typename Dot>
static void TestMinRes(bool singleThreadedMode) {
  using Vector = ColumnVector<Scalar>;

  Scalar constexpr kAbsTol = KrylovTestConstants<Scalar>::kAbsTol;
  Scalar constexpr kRelDivTol = KrylovTestConstants<Scalar>::kRelDivTol;

  auto scheduler = SetupScheduler(singleThreadedMode);

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

    krylov::JacobiPrec<Scalar> P(A);
    auto opP = [&P](auto const& x, auto& Px) { P.Solve(x, Px); };

    int maxIter = matrixSize;
    auto opA = MakeMatrixOperator(A);
    auto info = krylov::MinRes(
        opA,
        b,
        sol,
        opP,
        maxIter,
        StopCriterion{relativeTol, kAbsTol, kRelDivTol},
        VerbosityLevel::Warning,
        dot);

    EXPECT_LT(info.numIterDone, maxIter);
    EXPECT_LT(info.relativeResidualNorm, relativeTol);
    Vector minres_error(matrixSize);
    minres_error = ref - sol;
    EXPECT_LT(dot.Norm(minres_error), solTol * dot.Norm(ref));

    //
    // Use the reference as initial guess
    // MinRes should not do any iteration
    //

    sol = ref;
    info = krylov::MinRes(
        A,
        AsView(b),
        solView,
        opP,
        maxIter,
        StopCriterion{relativeTol, kAbsTol, kRelDivTol},
        VerbosityLevel::Warning,
        dot);

    EXPECT_LT(info.numIterDone, 1);
    EXPECT_LT(info.relativeResidualNorm, relativeTol);

    //
    // Use a non-zero initial guess
    //
    sol = Scalar(0.4) * ref;
    info = krylov::MinRes(
        A,
        b,
        sol,
        opP,
        maxIter,
        StopCriterion{relativeTol, kAbsTol, kRelDivTol},
        VerbosityLevel::Warning,
        dot);

    EXPECT_LT(info.numIterDone, maxIter);
    EXPECT_LT(info.relativeResidualNorm, relativeTol);
    minres_error = ref - sol;
    EXPECT_LT(dot.Norm(minres_error), solTol * dot.Norm(ref));

  } // for (int itest = 0; itest < 2; ++itest)
}

TEST(KrylovSolver, MinRes) {
  // Use "real" instead of float or double to improve compiler performance. Both are checked by CI.
  TestMinRes<real, krylov::StatusImplicitResidualNorm<real>, krylov::UsualDot>(
      /*singleThreadedMode*/ true);
  TestMinRes<real, krylov::StatusImplicitResidualNorm<real>, krylov::UsualDot>(
      /*singleThreadedMode*/ false);
}

TEST(KrylovSolver, MinRes_LuckyBreakdown) {
  // With A = I and identity preconditioner, the Krylov subspace is 1-dimensional for any RHS.
  // At iteration 1: v_new = Az - delta*v - gamma*v_old = z - 1*v - 0 = 0 → lucky breakdown.
  // The solver must perform the final Givens rotation to produce x = b exactly.
  // We use b with unit norm to avoid normalization rounding errors.
  constexpr int kSize = 4;
  auto A = Matrix<real>::Zero(kSize, kSize);
  A.SetIdentity();
  auto opA = MakeMatrixOperator(A);
  auto prec = IdentityPreconditioner();

  auto b = ColumnVector<real>::Zero(kSize);
  b(0) = real(1);

  auto x = ColumnVector<real>::Zero(kSize);

  // Use a very tight tolerance so the solver enters the loop rather than converging at the
  // pre-loop status check.
  auto info = krylov::MinRes(
      opA,
      b,
      x,
      prec,
      kSize,
      krylov::StatusImplicitResidualNorm<real>{real(1e-30), real(1e-30), real(1e10)});

  EXPECT_TRUE(info.converged);
  EXPECT_EQ(info.numIterDone, 1);
  EXPECT_NEAR(info.residualNorm, 0.0, 1e-12);

  ColumnVector<real> error(kSize);
  error = b - x;
  krylov::UsualDot dot{};
  EXPECT_LT(dot.Norm(error), real(1e-12));
}

TEST(KrylovSolver, MinRes_NonSpdPreconditioner) {
  // A preconditioner that negates the second component makes dot(b, P⁻¹b) < 0 for b = [0,1,0,0].
  // MINRES must detect this and return converged = false with numIterDone = 0.
  constexpr int kSize = 4;
  auto A = Matrix<real>::Zero(kSize, kSize);
  A.SetIdentity();
  auto opA = MakeMatrixOperator(A);

  auto negatingPrec = [](auto const& in, auto& out) {
    out = in;
    out(1) = -in(1);
  };

  ColumnVector<real> b(kSize);
  b(0) = real(0);
  b(1) = real(1);
  b(2) = real(0);
  b(3) = real(0);

  auto x = ColumnVector<real>::Zero(kSize);

  auto info = krylov::MinRes(
      opA,
      b,
      x,
      negatingPrec,
      kSize,
      krylov::StatusImplicitResidualNorm<real>{real(1e-12), real(1e-12), real(1e10)},
      VerbosityLevel::Silent);

  EXPECT_FALSE(info.converged);
  EXPECT_EQ(info.numIterDone, 0);
}

// Parameterized over the three non-finite values that must all route to the failure path.
class MinResNonFinite : public ::testing::TestWithParam<real> {};

INSTANTIATE_TEST_SUITE_P(
    NonFiniteValues,
    MinResNonFinite,
    ::testing::Values(
        std::numeric_limits<real>::infinity(),
        -std::numeric_limits<real>::infinity(),
        std::numeric_limits<real>::quiet_NaN()));

TEST_P(MinResNonFinite, PreLoopNonFinitePreconditioner) {
  // A preconditioner that injects a non-finite value into the first output component makes the
  // pre-loop gammaSqr = dot(b, P⁻¹b) non-finite. MINRES must detect this via the IsFinite guard
  // and fail without poisoning x with NaN/Inf.
  real const nonFinite = GetParam();
  constexpr int kSize = 4;
  auto A = Matrix<real>::Zero(kSize, kSize);
  A.SetIdentity();
  auto opA = MakeMatrixOperator(A);

  auto nonFinitePrec = [nonFinite](auto const& in, auto& out) {
    out = in;
    out(0) = nonFinite;
  };

  auto b = ColumnVector<real>::Zero(kSize);
  b(0) = real(1);

  auto x = ColumnVector<real>::Zero(kSize);

  auto info = krylov::MinRes(
      opA,
      b,
      x,
      nonFinitePrec,
      kSize,
      krylov::StatusImplicitResidualNorm<real>{real(1e-12), real(1e-12), real(1e10)},
      VerbosityLevel::Silent);

  EXPECT_FALSE(info.converged);
  EXPECT_EQ(info.numIterDone, 0);
  EXPECT_TRUE(IsFinite(x.GetConstSpan()));
}

TEST_P(MinResNonFinite, InLoopNonFiniteOperator) {
  // A is identity except one diagonal entry is non-finite. With an identity preconditioner and a
  // non-zero RHS, the pre-loop checks pass (finite, positive gammaSqr) and the non-finite value is
  // first encountered inside the iteration via Apply(A, z, Az), making the in-loop gammaSqr
  // non-finite. MINRES must detect this and fail without poisoning x.
  real const nonFinite = GetParam();
  constexpr int kSize = 4;
  auto A = Matrix<real>::Zero(kSize, kSize);
  A.SetIdentity();
  A(2, 2) = nonFinite;
  auto opA = MakeMatrixOperator(A);
  auto prec = IdentityPreconditioner();

  auto b = ColumnVector<real>::Zero(kSize);
  for (int i = 0; i < kSize; ++i) {
    b(i) = real(1);
  }

  auto x = ColumnVector<real>::Zero(kSize);

  // Tight tolerance so the solver enters the loop rather than converging at the pre-loop check.
  auto info = krylov::MinRes(
      opA,
      b,
      x,
      prec,
      kSize,
      krylov::StatusImplicitResidualNorm<real>{real(1e-30), real(1e-30), real(1e10)},
      VerbosityLevel::Silent);

  EXPECT_FALSE(info.converged);
  EXPECT_TRUE(IsFinite(x.GetConstSpan()));
}
