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

#include <mochi_core/solvers/qp_solver.h>

#include <mochi_core/test/mochi_test_helpers.h>

using namespace mochi;

namespace {
template <int d, typename T>
std::vector<NdArray<T, d>> GenerateTargetList() {
  // Use the same QPSolver to solve a series of QP with different targets
  std::vector<NdArray<T, d>> targets;
  // Corner points with 1 active constraint
  for (int i = 0; i < d; i++) {
    for (int j : {-2, 2}) {
      NdArray<T, d> x;
      AsView(x).SetZero();
      x[i] = T(j);
      targets.emplace_back(x);
    }
  }
  // Corner points with d active constraint
  for (int i = 0; i < std::pow(2, d); i++) {
    NdArray<T, d> x;
    for (int j = 0; j < d; j++) {
      x[j] = (i & (1 << j)) != 0 ? -2 : 2;
    }
    targets.emplace_back(x);
  }
  return targets;
}

template <int d, typename T, bool IsSparse>
DynamicArray<ColumnVector<T>>
SolveSeriesOfQP(QPSolver<T>& solver, bool output, QPSolverConvergenceStatus expectedOutcome) {
  if (output) {
    auto params = solver.GetQPSolverParams();
    params.verbosity = VerbosityLevel::Verbose;
    params.absTolComplementarity = 1e-4_r;
    params.absTolLagrangianGradient = 1e-4_r;
    solver.SetQPSolverParams(params);
  }

  // Solve QP with series of targets
  DynamicArray<ColumnVector<T>> results;
  for (auto target : GenerateTargetList<d, T>()) {
    Matrix<T> h(d, d);
    ColumnVector<T> g(d);
    h.SetIdentity();
    g = -AsConstView(target);
    if (IsSparse) {
      SparseMatrix<T> hSparse = ToSparseMatrix(h);
      solver.SetObjective(hSparse, g, test::ExpectOK{});
    } else {
      solver.SetObjective(h, g, test::ExpectOK{});
    }

    // Solve for closest point in the box to target
    ColumnVector<T> x(d);
    for (int i = 0; i < d; i++) {
      x[i] = T(0);
    }
    auto res = solver.Solve(x);
    results.emplace_back(x);
    EXPECT_EQ(res, expectedOutcome);
  }
  return results;
}

template <int d, typename T, bool IsSparse>
DynamicArray<ColumnVector<T>> SolveNoConstraint(bool output) {
  QPSolver<T> solver(d, 0);
  return SolveSeriesOfQP<d, T, IsSparse>(
      solver, output, mochi::QPSolverConvergenceStatus::FeasibleSolutionFound);
}

template <int d, typename T>
void TestNoConstraint(bool output) {
  // Sparse
  auto results1 = SolveNoConstraint<d, T, true>(output);
  // Dense
  auto results2 = SolveNoConstraint<d, T, false>(output);
  // Compare
  for (int i = 0; i < isize(results1); i++) {
    ColumnVector<T> diff;
    diff = results1[i] - results2[i];
    EXPECT_NEAR(diff.Norm(), 0_r, 1e-3_r);
  }
}

template <int d, typename T, bool UseGeneralLinearConstraints, bool IsSparse>
DynamicArray<ColumnVector<T>> SolveClosestInBox(bool output) {
  QPSolver<T> solver(d, UseGeneralLinearConstraints ? d : 0);

  // Set constraints
  if (UseGeneralLinearConstraints) {
    Matrix<T> A(d, d);
    A.SetIdentity();
    if (IsSparse) {
      SparseMatrix<T> ASparse = ToSparseMatrix(A);
      solver.SetA(ASparse, test::ExpectOK{});
    } else {
      solver.SetA(A, test::ExpectOK{});
    }
    for (int i = 0; i < d; i++) {
      solver.SetABounds(i, -1_r, 1_r, test::ExpectOK{});
    }
  } else {
    for (int i = 0; i < d; i++) {
      solver.SetBounds(i, -1_r, 1_r, test::ExpectOK{});
    }
  }

  // Use the same QPSolver to solve a series of QP with different targets
  return SolveSeriesOfQP<d, T, IsSparse>(
      solver, output, QPSolverConvergenceStatus::FeasibleSolutionFound);
}

template <int d, typename T>
void TestClosestInBox(bool output) {
  // Sparse
  auto results1 = SolveClosestInBox<d, T, true, true>(output);
  auto results2 = SolveClosestInBox<d, T, false, true>(output);
  // Dense
  auto results3 = SolveClosestInBox<d, T, true, false>(output);
  auto results4 = SolveClosestInBox<d, T, false, false>(output);
  // Compare
  for (int i = 0; i < isize(results1); i++) {
    ColumnVector<T> diff;
    diff = results1[i] - results2[i];
    EXPECT_NEAR(diff.Norm(), 0_r, 1e-3_r);
    diff = results1[i] - results3[i];
    EXPECT_NEAR(diff.Norm(), 0_r, 1e-3_r);
    diff = results1[i] - results4[i];
    EXPECT_NEAR(diff.Norm(), 0_r, 1e-3_r);
  }
}

template <int d, typename T, bool IsSparse>
DynamicArray<ColumnVector<T>> SolveClosestInOctagon(bool output) {
  QPSolver<T> solver(d, std::pow(2, d));

  // Set constraints
  for (int i = 0; i < d; i++) {
    solver.SetBounds(i, -1_r, 1_r, test::ExpectOK{});
  }
  Matrix<T> A(std::pow(2, d), d);
  for (int i = 0; i < std::pow(2, d); i++) {
    for (int j = 0; j < d; j++) {
      A(i, j) = (i & (1 << j)) != 0 ? -1 : 1;
    }
    solver.SetABounds(i, -1_r, 1_r, test::ExpectOK{});
  }
  if (IsSparse) {
    SparseMatrix<T> ASparse = ToSparseMatrix(A);
    solver.SetA(ASparse, test::ExpectOK{});
  } else {
    solver.SetA(A, test::ExpectOK{});
  }

  // Use the same QPSolver to solve a series of QP with different targets
  return SolveSeriesOfQP<d, T, IsSparse>(
      solver, output, QPSolverConvergenceStatus::FeasibleSolutionFound);
}

template <int d, typename T>
void TestClosestInOctagon(bool output) {
  // Sparse
  auto results1 = SolveClosestInOctagon<d, T, true>(output);
  // Dense
  auto results2 = SolveClosestInOctagon<d, T, false>(output);
  // Compare
  for (int i = 0; i < isize(results1); i++) {
    ColumnVector<T> diff;
    diff = results1[i] - results2[i];
    EXPECT_NEAR(diff.Norm(), 0_r, 1e-3_r);
  }
}

template <int d, typename T, bool IsSparse>
DynamicArray<ColumnVector<T>> SolveInfeasible(bool output) {
  QPSolver<T> solver(d, 1);

  Matrix<T> A(1, d);
  for (int i = 0; i < d; i++) {
    A(0, i) = 1_r;
  }
  if (IsSparse) {
    SparseMatrix<T> ASparse = ToSparseMatrix(A);
    solver.SetA(ASparse, test::ExpectOK{});
  } else {
    solver.SetA(A, test::ExpectOK{});
  }
  solver.SetABounds(0, -std::numeric_limits<T>::infinity(), -1_r, test::ExpectOK{});
  for (int i = 0; i < d; i++) {
    solver.SetBounds(i, 0_r, std::numeric_limits<T>::infinity(), test::ExpectOK{});
  }

  if (output) {
    auto params = solver.GetQPSolverParams();
    params.verbosity = VerbosityLevel::Verbose;
    solver.SetQPSolverParams(params);
  }

  // Solve QP with series of targets
  return SolveSeriesOfQP<d, T, IsSparse>(
      solver, output, QPSolverConvergenceStatus::InfeasibilityDetected);
}

template <int d, typename T>
void TestInfeasible(bool output) {
  // Sparse
  SolveInfeasible<2, real, true>(output);
  // Dense
  SolveInfeasible<2, real, false>(output);
}
} // namespace

TEST_IF(MOCHI_USE_DOUBLE_PRECISION, QP, Infeasible) {
  bool output = false;
  TestInfeasible<2, real>(output);
  TestInfeasible<3, real>(output);
}

TEST_IF(MOCHI_USE_DOUBLE_PRECISION, QP, NoConstraint) {
  bool output = false;
  TestNoConstraint<2, real>(output);
  TestNoConstraint<3, real>(output);
}

TEST_IF(MOCHI_USE_DOUBLE_PRECISION, QP, ClosestInBox) {
  bool output = false;
  TestClosestInBox<2, real>(output);
  TestClosestInBox<3, real>(output);
}

TEST_IF(MOCHI_USE_DOUBLE_PRECISION, QP, ClosestInOctagon) {
  bool output = false;
  TestClosestInOctagon<2, real>(output);
  TestClosestInOctagon<3, real>(output);
}
