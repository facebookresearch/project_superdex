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

#include <mochi_core/solvers/line_search.h>
#include <mochi_core/solvers/newton_solver.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/sparsity_utils.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

using namespace mochi;

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

/**
 * One-DoF problem with objective f, gradient df and Hessian ddf.
 */
static SnleProblem<real> GetTestProblem(real (*f)(real), real (*df)(real), real (*ddf)(real)) {
  // Some SNLE data
  auto residual = ColumnVector<real>{1};
  auto dresidual = AnyMatrix<real>{Matrix<real>{1, 1}};
  SetZero(residual);
  SetZero(dresidual);

  // Capture the SNLE data in an assembly function
  auto assembleFn = [=](SnleProblem<real>& problem, AssemblyParams const& params) mutable {
    auto solution = problem.GetSolution();
    real x = solution[0];

    double objective = 0.0;
    if (params.assemObj) {
      objective = f(x);
    }
    if (params.assemRes) {
      residual[0] = df(x);
    }
    if (params.assemDRes) {
      auto& dresMat = std::get<Matrix<real>>(dresidual);
      dresMat(0, 0) = ddf(x);
    }

    // Add SNLE data to the problem, as if it came from a single actor
    if (params.assemObj) {
      problem.objective = objective;
    }
    if (params.assemRes) {
      problem.actorResiduals.resize(1);
      problem.actorResiduals[0] = std::make_pair(0, &residual);
    }
    if (params.assemDRes) {
      problem.actorMatrices.resize(1);
      problem.actorMatrices[0] = std::make_pair(0, &dresidual);
    }
  };

  // Trivial increment function
  auto incrementFn = [](SnleProblem<real>& problem) { problem.solution += problem.increment; };

  // Create an SnleProblem
  SnleProblemFunctions<real> functions;
  functions.assemble = std::move(assembleFn);
  functions.onPostNewIncrement = std::move(incrementFn);
  int constexpr kNumDofs = 1;
  return SnleProblem<real>{kNumDofs, kNumDofs, std::move(functions)};
}

/**
 * Broyden-like problem
 */
static SnleProblem<real> GetBroydenProblem(int n) {
  // Sparsity
  std::vector<NdArray<int, 2>> spPattern;
  for (int i = 0; i < n; ++i) {
    if (i > 0) {
      spPattern.emplace_back(i, i - 1);
    }
    spPattern.emplace_back(i, i);
    if (i + 1 < n) {
      spPattern.emplace_back(i, i + 1);
    }
  }

  // Some SNLE data
  auto residual = ColumnVector<real>{n};
  auto dresidual = AnyMatrix<real>{SparseMatrix<real>{MakeSparsityGraph(spPattern)}};
  SetZero(residual);
  SetZero(dresidual);

  // Capture the SNLE data in an assembly function
  auto assembleFn = [=](SnleProblem<real>& problem, AssemblyParams const& params) mutable {
    auto solution = problem.GetSolution();
    if (params.assemObj) {
      double objective = 0.0;
      for (int i = 0; i < n; ++i) {
        objective += solution[i] *
            (1_r + (3_r / 2_r) * solution[i] - (1_r / 6_r) * solution[i] * solution[i]);
      }
      for (int i = 0; i < n - 1; ++i) {
        objective -= 2_r * solution[i] * solution[i + 1];
      }
      problem.objective = objective;
    }

    if (params.assemRes) {
      MOCHI_ASSERT(residual.size() == n);
      residual[0] = 1_r + (3_r - 0.5_r * solution[0]) * solution[0] - 2_r * solution[1];
      for (int i = 1; i < n - 1; ++i) {
        residual[i] = 1_r + (3_r - 0.5_r * solution[i]) * solution[i] - 2_r * solution[i - 1] -
            2_r * solution[i + 1];
      }
      residual[n - 1] =
          1_r + (3_r - 0.5_r * solution[n - 1]) * solution[n - 1] - 2_r * solution[n - 2];
      problem.actorResiduals.resize(1);
      problem.actorResiduals[0] = std::make_pair(0, &residual);
    }

    if (params.assemDRes) {
      auto& dresMat = std::get<SparseMatrix<real>>(dresidual);
      MOCHI_ASSERT(dresMat.Rows() == n);
      for (int i = 0; i < n; ++i) {
        if (i > 0) {
          dresMat.SetValue(i, i - 1, -2_r);
        }
        dresMat.SetValue(i, i, 3_r - solution[i]);
        if (i + 1 < n) {
          dresMat.SetValue(i, i + 1, -2_r);
        }
      }
      problem.actorMatrices.resize(1);
      problem.actorMatrices[0] = std::make_pair(0, &dresidual);
    }
  };

  // Trivial increment function
  auto incrementFn = [](SnleProblem<real>& problem) { problem.solution += problem.increment; };

  // Create an SnleProblem
  SnleProblemFunctions<real> functions;
  functions.assemble = std::move(assembleFn);
  functions.onPostNewIncrement = std::move(incrementFn);
  return SnleProblem<real>{n, n, std::move(functions)};
}

/**
 * Broyden-like problem
 */
static SnleProblem<real> GetBroydenProblem2(int n) {
  // Sparsity
  std::vector<NdArray<int, 2>> spPattern;
  for (int i = 0; i < n; ++i) {
    if (i > 0) {
      spPattern.emplace_back(i, i - 1);
    }
    spPattern.emplace_back(i, i);
    if (i + 1 < n) {
      spPattern.emplace_back(i, i + 1);
    }
  }

  // Some SNLE data
  auto residual = ColumnVector<real>{n};
  auto dresidual = AnyMatrix<real>{SparseMatrix<real>{MakeSparsityGraph(spPattern)}};
  SetZero(residual);
  SetZero(dresidual);

  // Capture the SNLE data in an assembly function
  auto assembleFn = [=](SnleProblem<real>& problem, AssemblyParams const& params) mutable {
    auto solution = problem.GetSolution();
    if (params.assemObj) {
      double objective = 0.0;
      objective += solution[0] + (3_r / 4_r) * solution[0] * solution[0] -
          (1_r / 6_r) * solution[0] * solution[0] * solution[0];
      for (int i = 1; i + 1 < n; ++i) {
        objective += solution[i] + (3_r / 2_r) * solution[i] * solution[i] -
            (1_r / 6_r) * solution[i] * solution[i] * solution[i];
      }
      objective += solution[n - 1] + (3_r / 4_r) * solution[n - 1] * solution[n - 1] -
          (1_r / 6_r) * solution[n - 1] * solution[n - 1] * solution[n - 1];
      for (int i = 0; i < n - 1; ++i) {
        objective -= 1.5_r * solution[i] * solution[i + 1];
      }
      problem.objective = objective;
    }

    if (params.assemRes) {
      MOCHI_ASSERT(residual.size() == n);
      residual[0] =
          1_r + 1.5_r * solution[0] - 0.5_r * solution[0] * solution[0] - 1.5_r * solution[1];
      for (int i = 1; i < n - 1; ++i) {
        residual[i] = 1_r + 3_r * solution[i] - 0.5_r * solution[i] * solution[i] -
            1.5_r * solution[i - 1] - 1.5_r * solution[i + 1];
      }
      residual[n - 1] = 1_r + 1.5_r * solution[n - 1] - 0.5_r * solution[n - 1] * solution[n - 1] -
          1.5_r * solution[n - 2];
      problem.actorResiduals.resize(1);
      problem.actorResiduals[0] = std::make_pair(0, &residual);
    }

    if (params.assemDRes) {
      auto& dresMat = std::get<SparseMatrix<real>>(dresidual);
      MOCHI_ASSERT(dresMat.Rows() == n);
      for (int i = 0; i < n; ++i) {
        if (i > 0) {
          dresMat.SetValue(i, i - 1, -1.5_r);
        }
        if ((i == 0) || (i == n - 1)) {
          dresMat.SetValue(i, i, 1.5_r - solution[i]);
        } else {
          dresMat.SetValue(i, i, 3_r - solution[i]);
        }
        if (i + 1 < n) {
          dresMat.SetValue(i, i + 1, -1.5_r);
        }
      }
      problem.actorMatrices.resize(1);
      problem.actorMatrices[0] = std::make_pair(0, &dresidual);
    }
  };

  // Trivial increment function
  auto incrementFn = [](SnleProblem<real>& problem) { problem.solution += problem.increment; };

  // Create an SnleProblem
  SnleProblemFunctions<real> functions;
  functions.assemble = std::move(assembleFn);
  functions.onPostNewIncrement = std::move(incrementFn);
  return SnleProblem<real>{n, n, std::move(functions)};
}

TEST(NewtonSolver, SmoothLocalMinimum) {
  constexpr auto kEpsilon = std::numeric_limits<real>::epsilon();
  constexpr real kAbsTol = 1e3_r * kEpsilon;

  SnleProblem<real> problem = GetTestProblem(
      [](real x) { return -(x * x * x) / 30.0_r - x * Sin(x) - Cos(x); },
      [](real x) { return -(x * x) / 10.0_r - x * Cos(x); },
      [](real x) { return -x / 5.0_r - Cos(x) + x * Sin(x); });
  for (auto solverType :
       {NonLinearSolverType::Newton, NonLinearSolverType::BFGS, NonLinearSolverType::SR1}) {
    for (auto dResidualAssemblyPeriod : {1, 2, 4, 8}) {
      bool const wasWarningEnabled = IsLogChannelEnabled(LogChannel::Warning);
      if (solverType != NonLinearSolverType::Newton && dResidualAssemblyPeriod == 1) {
        // Disable warnings about quasi-Newton methods with dResidualAssemblyPeriod == 1.
        EnableLogChannel(LogChannel::Warning, false);
      }
      if (solverType != NonLinearSolverType::Newton) {
        // Disable warnings about using the dense LDLT with a low-rank augmented matrix.
        EnableLogChannel(LogChannel::Warning, false);
      }
      MOCHI_DEFER(EnableLogChannel(LogChannel::Warning, wasWarningEnabled));

      for (auto lineSearchType :
           {LineSearchType::None,
            LineSearchType::Simple,
            LineSearchType::Armijo,
            LineSearchType::WolfeWeak,
            LineSearchType::WolfeStrong,
            LineSearchType::ResidualNorm}) {
        ColumnVector<real, 1> x0(-3_r); // Hessian is positive definite starting from -3.
        problem.SetSolution(x0);

        // Set linear parameters
        NewtonSolverParams params;
        params.lParams.solverType = LinearSolverType::LDLT;
        params.lParams.preconditionerType = PreconditionerType::None;
        params.lParams.absTol = kAbsTol;
        params.lParams.relTol = kEpsilon;

        // Set nonlinear parameters
        params.solverType = solverType;
        params.dResidualAssemblyPeriod = dResidualAssemblyPeriod;
        params.maxIter = 200;
        params.lineSearch.type = lineSearchType;
        params.psdProjMode = PsdProjectionMode::Never;
        params.convergenceMode = NonLinearSolverConvergenceMode::Global;
        params.absTolRes = kAbsTol;
        params.relTolRes = kEpsilon;
        params.solRelTol = 0_r;

        // Create solver
        NewtonSolver<real> solver(params);

        // Solver problem
        auto result = solver.Solve(problem);

        // Expected solution
        double const ref = -1.4275517787645941208;
        double const obj = -ref * ref * ref / 30.0 - ref * Sin(ref) - Cos(ref);

        EXPECT_TRUE(result.convergence == ConvergenceStatus::Converged);
        if (params.lineSearch.type == LineSearchType::None ||
            params.lineSearch.type == LineSearchType::ResidualNorm) {
          // None and ResidualNorm do not update the objective
          EXPECT_NEAR(result.objective, 0_r, kAbsTol);
        } else {
          EXPECT_NEAR(result.objective, obj, kAbsTol);
        }

        EXPECT_NEAR(problem.GetSolution()[0], real(ref), kAbsTol);

        problem.UpdateObjective();
        EXPECT_NEAR(problem.GetObjective(), real(obj), kAbsTol);

        problem.SetSolution(x0);
        EXPECT_TRUE(test::NearEqualMatrices(problem.GetSolution(), x0));

        {
          problem.UpdateObjective();
          double obj0 = -x0[0] * x0[0] * x0[0] / 30.0 - x0[0] * Sin(x0[0]) - Cos(x0[0]);
          EXPECT_NEAR(problem.GetObjective(), real(obj0), kAbsTol);
        }

        problem.UpdateResidual();
        {
          double res0 = -x0[0] * x0[0] / 10.0 - x0[0] * Cos(x0[0]);
          EXPECT_NEAR(problem.GetResidual()[0], real(res0), kAbsTol);
        }

        auto resultBis = solver.Solve(problem);
        EXPECT_NEAR(problem.GetSolution()[0], real(ref), kAbsTol);

        problem.UpdateObjective();
        EXPECT_NEAR(problem.GetObjective(), real(obj), kAbsTol);
        EXPECT_EQ(result.convergence, resultBis.convergence);
        EXPECT_EQ(result.numIterDone, resultBis.numIterDone);
        EXPECT_EQ(result.numLastLSIterDone, resultBis.numLastLSIterDone);
        EXPECT_EQ(result.totalNumLinearIterDone, resultBis.totalNumLinearIterDone);

        // The stored increment must be the step taken, also when the line search fails and the
        // solve keeps its last trial. With these parameters, Armijo and Wolfe reject both trials.
        params.maxIter = 1;
        params.lineSearch.maxIter = 2;
        params.lineSearch.alpha = 0.5_r;
        params.lineSearch.wolfe1 = 0.75_r;
        solver.SetParams(params);
        problem.SetSolution(x0);
        solver.Solve(problem);
        EXPECT_NEAR(problem.GetIncrement()[0], problem.GetSolution()[0] - x0[0], kAbsTol);
      }
    }
  }
}

TEST(NewtonSolver, QuasiNewton) {
  // Verify quasi-Newton updates help convergence on multi-dimensional Broyden problems, and that
  // BFGS skips the updates of steps without positive curvature.
  constexpr real kEpsilon = std::numeric_limits<real>::epsilon();
  constexpr real kAbsTol = 1e3_r * kEpsilon;

  constexpr int n = 50;
  constexpr int period = 8;

  SnleProblem<real> problem = GetBroydenProblem(n);
  ColumnVector<real> x0(n);
  x0.SetConstant(-3.0_r);

  NewtonSolverParams params;
  params.lParams.solverType = LinearSolverType::CG;
  params.lParams.preconditionerType = PreconditionerType::None;
  params.lParams.normType = LinearSolverConvergenceNorm::ResidualL2;
  params.lParams.absTol = 0.1_r * kAbsTol;
  params.lParams.relTol = 0.1_r * kEpsilon;
  params.linearToleranceStrategy = LinearToleranceStrategy::Constant;
  params.dResidualAssemblyPeriod = period;
  params.maxIter = 200;
  params.lineSearch.type = LineSearchType::ResidualNorm;
  params.psdProjMode = PsdProjectionMode::Never;
  params.convergenceMode = NonLinearSolverConvergenceMode::Global;
  params.absTolRes = kAbsTol;
  params.relTolRes = kEpsilon;
  params.solRelTol = 0_r;

  // Baseline: frozen Newton (no Hessian updates between reassemblies)
  problem.SetSolution(x0);
  params.solverType = NonLinearSolverType::Newton;
  NewtonSolver<real> solver(params);
  auto resultNewton = solver.Solve(problem);
  EXPECT_EQ(resultNewton.convergence, ConvergenceStatus::Converged);

  // BFGS should converge in fewer iterations than frozen Newton
  for (auto solverType : {NonLinearSolverType::BFGS, NonLinearSolverType::SR1}) {
    problem.SetSolution(x0);
    params.solverType = solverType;
    solver.SetParams(params);
    auto resultQuasiNewton = solver.Solve(problem);
    EXPECT_EQ(ConvergenceStatus::Converged, resultQuasiNewton.convergence);
    EXPECT_LT(resultQuasiNewton.numIterDone, resultNewton.numIterDone);
  }

  // The BFGS update of a step without positive curvature can make the operator indefinite. From
  // x = 0, Armijo accepts the Newton step of E(x) = x⁴/2 - 3x³/2 + x²/2 - x to x = 1, where the
  // gradient drops from -1 to -5/2.
  SnleProblem<real> quartic = GetTestProblem(
      [](real x) { return 0.5_r * x * x * x * x - 1.5_r * x * x * x + 0.5_r * x * x - x; },
      [](real x) { return 2_r * x * x * x - 4.5_r * x * x + x - 1_r; },
      [](real x) { return 6_r * x * x - 9_r * x + 1_r; });
  quartic.SetSolution(ColumnVector<real, 1>(0_r));
  params.solverType = NonLinearSolverType::BFGS;
  params.lineSearch.type = LineSearchType::Armijo;
  solver.SetParams(params);
  EXPECT_EQ(solver.Solve(quartic).convergence, ConvergenceStatus::Converged);
}

TEST(NewtonSolver, LinearToleranceStrategy) {
  constexpr real kEpsilon = std::numeric_limits<real>::epsilon();
  constexpr real kRelTol = kEpsilon;
  constexpr real kAbsTol = 1e3_r * kEpsilon;

  // Set up the solver parameters.
  NewtonSolverParams params;
  params.lParams.solverType = LinearSolverType::CG;
  params.lParams.preconditionerType = PreconditionerType::None;
  params.lParams.normType = LinearSolverConvergenceNorm::ResidualL2;
  params.lParams.absTol = 0.1_r * kAbsTol;
  params.lParams.relTol = 0.1_r * kRelTol;
  params.maxIter = 200;
  params.lineSearch.type = LineSearchType::ResidualNorm;
  params.psdProjMode = PsdProjectionMode::Never;
  params.convergenceMode = NonLinearSolverConvergenceMode::Global;
  params.absTolRes = kAbsTol;
  params.relTolRes = kRelTol;
  params.solRelTol = 0_r;

  // Set up the solver.
  NewtonSolver<real> solver(params);

  // Loop over Broyden problems of multiple sizes. Consider only large sizes so that the problem is
  // sufficiently hard for Eisenstat-Walker to converge in strictly fewer iterations than constant.
  for (int n : {100, 500, 1000}) {
    SnleProblem<real> problem = GetBroydenProblem(n);
    ColumnVector<real> x0(n);
    x0.SetConstant(-3.0_r);

    // Solve the problem with constant linear tolerance strategy.
    problem.SetSolution(x0);

    params.linearToleranceStrategy = LinearToleranceStrategy::Constant;
    solver.SetParams(params);
    auto const resultConstant = solver.Solve(problem);

    // Solve the problem with Eisenstat-Walker choice #1 linear tolerance strategy.
    problem.SetSolution(x0);

    params.linearToleranceStrategy = LinearToleranceStrategy::EisenstatWalker1;
    solver.SetParams(params);
    auto const resultEW1 = solver.Solve(problem);

    // Solve the problem with Eisenstat-Walker choice #2 linear tolerance strategy.
    problem.SetSolution(x0);

    params.linearToleranceStrategy = LinearToleranceStrategy::EisenstatWalker2;
    solver.SetParams(params);
    auto const resultEW2 = solver.Solve(problem);

    // Check the solver converged
    EXPECT_EQ(resultConstant.convergence, ConvergenceStatus::Converged);
    EXPECT_EQ(resultEW1.convergence, ConvergenceStatus::Converged);
    EXPECT_EQ(resultEW2.convergence, ConvergenceStatus::Converged);

    if (params.lineSearch.type == LineSearchType::ResidualNorm) {
      // The objective function is not updated
      EXPECT_NEAR(resultConstant.objective, 0_r, kAbsTol);
      EXPECT_NEAR(resultEW1.objective, 0_r, kAbsTol);
      EXPECT_NEAR(resultEW2.objective, 0_r, kAbsTol);
    }

    // Check the total number of linear iterations is smaller with Eisenstat-Walker strategies.
    EXPECT_LT(resultEW1.totalNumLinearIterDone, resultConstant.totalNumLinearIterDone);
    EXPECT_LT(resultEW2.totalNumLinearIterDone, resultConstant.totalNumLinearIterDone);
  }
}

TEST(NewtonSolver, LinearToleranceStrategy2) {
  constexpr real kEpsilon = std::numeric_limits<real>::epsilon();
  constexpr real kRelTol = kEpsilon;
  constexpr real kAbsTol = 1e3_r * kEpsilon;

  // Set up the solver parameters.
  NewtonSolverParams params;
  params.lParams.solverType = LinearSolverType::CG;
  params.lParams.preconditionerType = PreconditionerType::None;
  params.lParams.normType = LinearSolverConvergenceNorm::ResidualL2;
  params.lParams.absTol = 0.1_r * kAbsTol;
  params.lParams.relTol = 0.1_r * kRelTol;
  params.maxIter = 200;
  params.lineSearch.type = LineSearchType::ResidualNorm;
  params.psdProjMode = PsdProjectionMode::Never;
  params.convergenceMode = NonLinearSolverConvergenceMode::Global;
  params.absTolRes = kAbsTol;
  params.relTolRes = kRelTol;
  params.solRelTol = 0_r;

  // Set up the solver.
  NewtonSolver<real> solver(params);

  int n = 1234;

  SnleProblem<real> problem = GetBroydenProblem2(n);
  ColumnVector<real> x0(n);
  x0.SetConstant(-3.0_r);

  // Solve the problem with constant linear tolerance strategy.
  problem.SetSolution(x0);

  params.linearToleranceStrategy = LinearToleranceStrategy::Constant;
  solver.SetParams(params);
  auto const resultConstant = solver.Solve(problem);

  {
    problem.SetSolution(x0);
    problem.UpdateObjective();
    problem.UpdateResidual();
    auto resultConstantBis = solver.Solve(problem);
    EXPECT_EQ(resultConstant.convergence, resultConstantBis.convergence);
    EXPECT_EQ(resultConstant.numIterDone, resultConstantBis.numIterDone);
    EXPECT_EQ(resultConstant.numLastLSIterDone, resultConstantBis.numLastLSIterDone);
    EXPECT_EQ(resultConstant.totalNumLinearIterDone, resultConstantBis.totalNumLinearIterDone);
  }

  // Solve the problem with Eisenstat-Walker choice #1 linear tolerance strategy.
  problem.SetSolution(x0);
  params.linearToleranceStrategy = LinearToleranceStrategy::EisenstatWalker1;
  solver.SetParams(params);
  auto const resultEW1 = solver.Solve(problem);
  {
    problem.SetSolution(x0);
    problem.UpdateObjective();
    problem.UpdateResidual();
    auto resultEW1Bis = solver.Solve(problem);
    EXPECT_EQ(resultEW1.convergence, resultEW1Bis.convergence);
    EXPECT_EQ(resultEW1.numIterDone, resultEW1Bis.numIterDone);
    EXPECT_EQ(resultEW1.numLastLSIterDone, resultEW1Bis.numLastLSIterDone);
    EXPECT_EQ(resultEW1.totalNumLinearIterDone, resultEW1Bis.totalNumLinearIterDone);
  }

  // Solve the problem with Eisenstat-Walker choice #2 linear tolerance strategy.
  problem.SetSolution(x0);
  params.linearToleranceStrategy = LinearToleranceStrategy::EisenstatWalker2;
  solver.SetParams(params);
  auto const resultEW2 = solver.Solve(problem);
  {
    problem.SetSolution(x0);
    problem.UpdateObjective();
    problem.UpdateResidual();
    auto resultEW2Bis = solver.Solve(problem);
    EXPECT_EQ(resultEW2.convergence, resultEW2Bis.convergence);
    EXPECT_EQ(resultEW2.numIterDone, resultEW2Bis.numIterDone);
    EXPECT_EQ(resultEW2.numLastLSIterDone, resultEW2Bis.numLastLSIterDone);
    EXPECT_EQ(resultEW2.totalNumLinearIterDone, resultEW2Bis.totalNumLinearIterDone);
  }

  // Check the solver converged
  EXPECT_EQ(resultConstant.convergence, ConvergenceStatus::Converged);
  EXPECT_EQ(resultEW1.convergence, ConvergenceStatus::Converged);
  EXPECT_EQ(resultEW2.convergence, ConvergenceStatus::Converged);

  if (params.lineSearch.type == LineSearchType::ResidualNorm) {
    // The objective function is not updated
    EXPECT_NEAR(resultConstant.objective, 0_r, kAbsTol);
    EXPECT_NEAR(resultEW1.objective, 0_r, kAbsTol);
    EXPECT_NEAR(resultEW2.objective, 0_r, kAbsTol);
  }

  // This structured problem can converge in one CG iteration per Newton step for every strategy.
  // Eisenstat-Walker must not require more linear iterations, but may tie the constant strategy.
  EXPECT_LE(resultEW1.totalNumLinearIterDone, resultConstant.totalNumLinearIterDone);
  EXPECT_LE(resultEW2.totalNumLinearIterDone, resultConstant.totalNumLinearIterDone);

  ColumnVector<real> xref(n);
  xref.SetConstant(-Sqrt(2_r));
  EXPECT_TRUE(test::NearEqualMatrices(problem.GetSolution(), xref));
}

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

/**
 * Two-actor problem with mixed linear/nonlinear actors for testing PerActorWeighted convergence.
 * Actor 0 owns DoF 0: r₀ = x₀ - b + c·x₁    (linear, converges in 1 Newton iteration if c = 0)
 * Actor 1 owns DoF 1: r₁ = x₁² - 2 + c·x₀   (nonlinear, solution x₁ = √2 if c = 0)
 * By default, b = 2 and c = 0, which decouples the actors (diagonal dresidual).
 *
 * If populateConvergenceWeights is false, actorConvergenceWeights is left empty so that
 * PerActorWeighted mode triggers the size-mismatch fallback-to-Global path.
 */
static SnleProblem<real> GetTwoActorMixedProblem(
    ColumnVector<real> const& weight0,
    ColumnVector<real> const& weight1,
    bool populateConvergenceWeights = true,
    bool reverseActorOrder = false,
    real coupling = 0_r,
    real load0 = 2_r) {
  auto residual0 = ColumnVector<real>::Zero(1);
  auto residual1 = ColumnVector<real>::Zero(1);
  auto dresidual = AnyMatrix<real>{Matrix<real>::Zero(2, 2)};

  auto assembleFn = [=, &weight0, &weight1](
                        SnleProblem<real>& problem, AssemblyParams const& params) mutable {
    auto const x = problem.GetSolution();
    if (params.assemObj) {
      problem.objective = 0.5 * x[0] * x[0] - load0 * x[0] + x[1] * x[1] * x[1] / 3.0 - 2_r * x[1] +
          coupling * x[0] * x[1];
    }
    if (params.assemRes) {
      residual0[0] = x[0] - load0 + coupling * x[1];
      residual1[0] = x[1] * x[1] - 2_r + coupling * x[0];
      problem.actorResiduals.clear();
      problem.actorConvergenceWeights.clear();
      auto addActor =
          [&](int offset, ColumnVector<real>& residual, ColumnVector<real> const& weight) {
            problem.actorResiduals.emplace_back(offset, &residual);
            if (populateConvergenceWeights) {
              problem.actorConvergenceWeights.emplace_back(offset, &weight);
            }
          };
      if (reverseActorOrder) {
        addActor(1, residual1, weight1);
        addActor(0, residual0, weight0);
      } else {
        addActor(0, residual0, weight0);
        addActor(1, residual1, weight1);
      }
    }
    if (params.assemDRes) {
      auto& d = std::get<Matrix<real>>(dresidual);
      d(0, 0) = 1_r;
      d(1, 1) = 2_r * x[1];
      d(0, 1) = coupling;
      d(1, 0) = coupling;
      problem.actorMatrices.clear();
      problem.actorMatrices.emplace_back(0, &dresidual);
    }
  };

  SnleProblemFunctions<real> functions;
  functions.assemble = std::move(assembleFn);
  functions.onPostNewIncrement = [](SnleProblem<real>& p) { p.solution += p.increment; };
  return SnleProblem<real>{2, 2, std::move(functions)};
}

/**
 * Actor 0 converges via absTol in 1 iteration, actor 1 converges via relTol in 1 iteration.
 * Actor 0: linear r₀ = x₀ - 2, from x₀=0: after 1 step r₀ = 0 ≤ absTol.
 * Actor 1: nonlinear r₁ = x₁² - 2, from x₁=2: after 1 step x₁=1.5, r₁=0.25.
 *   Initial weighted norm = 2, post-step weighted norm = 0.25. Ratio = 0.125 ≤ relTol = 0.2.
 *   But 0.25 > absTol = 0.1, so actor 1 does NOT converge via absTol.
 */
TEST(NewtonSolver, PerActorWeighted_AbsTolAndRelTol) {
  auto w0 = ColumnVector<real>::Ones(1);
  auto w1 = ColumnVector<real>::Ones(1);

  auto problem = GetTwoActorMixedProblem(w0, w1);
  problem.SetSolution(ColumnVector<real>({{0_r, 2_r}}));

  NewtonSolverParams params;
  params.convergenceMode = NonLinearSolverConvergenceMode::PerActorWeighted;
  params.absTolRes = 0.1_r;
  params.relTolRes = 0.2_r;
  params.maxIter = 10;

  NewtonSolver<real> solver(params);
  auto const status = solver.Solve(problem);

  EXPECT_EQ(status.convergence, ConvergenceStatus::Converged);
  ASSERT_EQ(isize(status.actorConvergence), 2);
  EXPECT_EQ(status.actorConvergence[0], ConvergenceStatus::Converged);
  EXPECT_EQ(status.actorConvergence[1], ConvergenceStatus::Converged);
  EXPECT_EQ(status.numIterDone, 1);
}

TEST(NewtonSolver, PerActorWeighted_SortsWeightsWithResiduals) {
  for (bool const reverseActorOrder : {false, true}) {
    auto w0 = ColumnVector<real>::Ones(1);
    auto w1 = ColumnVector<real>::Ones(1);
    w1[0] = 100_r;

    auto problem =
        GetTwoActorMixedProblem(w0, w1, /*populateConvergenceWeights*/ true, reverseActorOrder);
    problem.SetSolution(ColumnVector<real>({{0_r, 2_r}}));

    NewtonSolverParams params;
    params.convergenceMode = NonLinearSolverConvergenceMode::PerActorWeighted;
    params.absTolRes = 3_r;
    params.relTolRes = 0_r;
    params.maxIter = 0;

    NewtonSolver<real> solver(params);
    auto const status = solver.Solve(problem);

    EXPECT_EQ(status.convergence, ConvergenceStatus::Stopped);
    ASSERT_EQ(isize(status.actorConvergence), 2);
    EXPECT_EQ(status.actorConvergence[0], ConvergenceStatus::Converged);
    EXPECT_EQ(status.actorConvergence[1], ConvergenceStatus::Stopped);
  }
}

/**
 * One actor converges, the other does not.
 * Actor 0: linear r₀ = x₀ - 2, from x₀=0: after 1 step r₀ = 0 (converges).
 * Actor 1: nonlinear r₁ = x₁² - 2, from x₁=2: after 1 step r₁ = 0.25.
 *   absTol = 0.1: 0.25 > 0.1 → fails.
 *   relTol = 0.05: 0.25/2.0 = 0.125 > 0.05 → fails.
 */
TEST(NewtonSolver, PerActorWeighted_OneActorFails) {
  auto w0 = ColumnVector<real>::Ones(1);
  auto w1 = ColumnVector<real>::Ones(1);

  auto problem = GetTwoActorMixedProblem(w0, w1);
  problem.SetSolution(ColumnVector<real>({{0_r, 2_r}}));

  NewtonSolverParams params;
  params.convergenceMode = NonLinearSolverConvergenceMode::PerActorWeighted;
  params.absTolRes = 0.1_r;
  params.relTolRes = 0.05_r;
  params.maxIter = 1;

  NewtonSolver<real> solver(params);
  auto const status = solver.Solve(problem);

  EXPECT_EQ(status.convergence, ConvergenceStatus::Stopped);
  ASSERT_EQ(isize(status.actorConvergence), 2);
  EXPECT_EQ(status.actorConvergence[0], ConvergenceStatus::Converged);
  EXPECT_EQ(status.actorConvergence[1], ConvergenceStatus::Stopped);
}

/**
 * PerActorWeighted mode with unpopulated actorConvergenceWeights should fall back to Global
 * convergence checking and converge.
 */
TEST(NewtonSolver, PerActorWeighted_FallbackToGlobal) {
  auto w0 = ColumnVector<real>::Ones(1);
  auto w1 = ColumnVector<real>::Ones(1);

  auto problem = GetTwoActorMixedProblem(w0, w1, /*populateConvergenceWeights*/ false);
  problem.SetSolution(ColumnVector<real>({{0_r, 2_r}}));

  NewtonSolverParams params;
  params.convergenceMode = NonLinearSolverConvergenceMode::PerActorWeighted;
  params.absTolRes = 1e-4_r;
  params.maxIter = 10;

  // Suppress the expected "falling back to Global" warning emitted by the solver.
  bool const wasWarningEnabled = IsLogChannelEnabled(LogChannel::Warning);
  EnableLogChannel(LogChannel::Warning, false);
  MOCHI_DEFER(EnableLogChannel(LogChannel::Warning, wasWarningEnabled));

  NewtonSolver<real> solver(params);
  auto const status = solver.Solve(problem);

  EXPECT_EQ(ConvergenceStatus::Converged, status.convergence); // Solver converged.
  EXPECT_TRUE(status.actorConvergence.empty()); // Empty after fallback.
}

/**
 * Parameters and residual weights of the last linear solve in @p numNewtonIters Newton iterations.
 * Each linear solve stops after one CG iteration and each step skips the line search, so every
 * linear solver norm follows the same Newton iterates.
 */
[[nodiscard]] static std::pair<KrylovSolverParams, ColumnVector<real>> GetLastLinearSolve(
    SnleProblem<real>& problem,
    ColumnVector<real> const& x0,
    NewtonSolverParams params,
    int numNewtonIters) {
  struct Solver : NewtonSolver<real> {
    using NewtonSolver<real>::NewtonSolver;

    [[nodiscard]] std::pair<KrylovSolverParams, ColumnVector<real>> GetLinearSolve() const {
      return {_linearSolver->GetParams(), _workspace.residualWeights};
    }
  };

  params.maxIter = numNewtonIters;
  params.lineSearch.type = LineSearchType::None;
  params.lParams.solverType = LinearSolverType::CG;
  params.lParams.preconditionerType = PreconditionerType::Jacobi;
  params.lParams.maxIter = 1;
  params.verbosity = VerbosityLevel::Error; // The linear solves stop without converging.

  problem.SetSolution(x0);
  Solver solver(params);
  solver.Solve(problem);
  return solver.GetLinearSolve();
}

/**
 * Forcing terms, absolute tolerances and residual weights against an oracle in double precision for
 * the Jacobi-preconditioned CG iteration. Actor 0 starts with a residual below absTol, which then
 * scales it, and the coupling leaves it a linear residual. Global mode solves on the L2-norm.
 */
TEST(NewtonSolver, ForcingTermNorms) {
  double constexpr kCoupling = 1e-4;
  double constexpr kLoad0 = 9.3e-5;
  double constexpr kAbsTol = 1e-4;
  double constexpr kRelTol = 1e-5;
  double constexpr kLinearAbsTol = 1e-6;
  std::array<double, 2> constexpr kWeights = {1, 4};
  std::array<double, 2> constexpr kX0 = {0, 1.43};
  double const kTol = 1e5 * std::numeric_limits<real>::epsilon();

  auto residual = [](std::array<double, 2> const& x) {
    return std::array<double, 2>{
        x[0] - kLoad0 + kCoupling * x[1], x[1] * x[1] - 2 + kCoupling * x[0]};
  };
  std::array<double, 2> const f0 = residual(kX0);
  std::array<double, 2> const z = {f0[0], f0[1] / (2 * kX0[1])};
  std::array<double, 2> const jz = {z[0] + kCoupling * z[1], kCoupling * z[0] + 2 * kX0[1] * z[1]};
  double const alpha = (f0[0] * z[0] + f0[1] * z[1]) / (z[0] * jz[0] + z[1] * jz[1]);
  std::array<double, 2> const r0 = {f0[0] - alpha * jz[0], f0[1] - alpha * jz[1]};
  std::array<double, 2> const f1 = residual({kX0[0] - alpha * z[0], kX0[1] - alpha * z[1]});

  auto scale = [&](std::array<double, 2> const& f, int i) {
    return Max(Sqrt(kWeights[i]) * Abs(f[i]), kAbsTol);
  };
  auto norm = [&](std::array<double, 2> const& v, bool perActor) {
    double normSqr = 0;
    for (int i = 0; i < 2; ++i) {
      normSqr += perActor ? kWeights[i] * Sqr(v[i] / scale(f0, i)) : Sqr(v[i]);
    }
    return Sqrt(normSqr);
  };
  auto expectedForcingTerm = [&](LinearToleranceStrategy strategy, bool perActor) {
    if (strategy == LinearToleranceStrategy::EisenstatWalker1) {
      return Abs(norm(f1, perActor) - norm(r0, perActor)) / norm(f0, perActor);
    } else if (strategy == LinearToleranceStrategy::EisenstatWalker2) {
      return 0.9 * Pow(norm(f1, perActor) / norm(f0, perActor), 0.5 + 0.5 * Sqrt(5.0));
    } else {
      return kRelTol;
    }
  };

  auto const weight0 = ColumnVector<real>({{real(kWeights[0])}});
  auto const weight1 = ColumnVector<real>({{real(kWeights[1])}});
  auto twoActors = GetTwoActorMixedProblem(
      weight0,
      weight1,
      /*populateConvergenceWeights*/ true,
      /*reverseActorOrder*/ false,
      real(kCoupling),
      real(kLoad0));
  ColumnVector<real> const x0({{real(kX0[0]), real(kX0[1])}});

  NewtonSolverParams params;
  params.absTolRes = real(kAbsTol);
  params.lParams.relTol = kRelTol;
  params.lParams.absTol = kLinearAbsTol;

  using Norm = LinearSolverConvergenceNorm;
  using Mode = NonLinearSolverConvergenceMode;
  for (auto const strategy :
       {LinearToleranceStrategy::EisenstatWalker1,
        LinearToleranceStrategy::EisenstatWalker2,
        LinearToleranceStrategy::Constant}) {
    params.linearToleranceStrategy = strategy;
    double const expectedL2 = expectedForcingTerm(strategy, false);
    double const expectedPerActor = expectedForcingTerm(strategy, true);

    for (auto const& [normType, mode] :
         {std::pair{Norm::ResidualL2, Mode::PerActorWeighted},
          std::pair{Norm::ResidualPerActorWeighted, Mode::PerActorWeighted},
          std::pair{Norm::ResidualPerActorWeighted, Mode::Global}}) {
      params.lParams.normType = normType;
      params.convergenceMode = mode;
      auto const [linearParams, weights] =
          GetLastLinearSolve(twoActors, x0, params, /*numNewtonIters*/ 2);
      bool const perActor =
          (normType == Norm::ResidualPerActorWeighted && mode == Mode::PerActorWeighted);

      double const expected = perActor ? expectedPerActor : expectedL2;
      // Constant keeps relTol exactly.
      double const tol = (strategy == LinearToleranceStrategy::Constant) ? 0 : kTol * expected;
      EXPECT_NEAR(expected, linearParams.relTol, tol);

      // The per-actor norm takes absTol in Newton's units, which the largest scale converts.
      double const expectedAbsTol =
          perActor ? kLinearAbsTol / Max(scale(f1, 0), scale(f1, 1)) : kLinearAbsTol;
      EXPECT_NEAR(expectedAbsTol, linearParams.absTol, kTol * expectedAbsTol);

      // The second solve's weights use the scales of the second iterate.
      ASSERT_EQ(perActor ? 2 : 0, weights.Rows());
      for (int i = 0; i < weights.Rows(); ++i) {
        double const expectedWeight = kWeights[i] / Sqr(scale(f1, i));
        EXPECT_NEAR(expectedWeight, weights[i], kTol * expectedWeight);
      }
    }

    if (strategy != LinearToleranceStrategy::Constant) {
      EXPECT_GT(Max(expectedL2, expectedPerActor), 1.4 * Min(expectedL2, expectedPerActor));
    }
  }
}

/**
 * With absTol = 0, an actor without residual takes the largest actor norm as its scale, in the
 * first iteration and after a step. A residual weight beyond the largest finite value is clamped to
 * it. With a tiny scale, a weight that fits stays exact even where 1/scale² overflows in FP32, and
 * a zero weight stays 0.
 */
TEST(NewtonSolver, PerActorWeighted_ZeroAbsTol) {
  NewtonSolverParams params;
  params.convergenceMode = NonLinearSolverConvergenceMode::PerActorWeighted;
  params.lParams.normType = LinearSolverConvergenceNorm::ResidualPerActorWeighted;
  params.absTolRes = 0_r;
  params.lParams.absTol = 1e-9;

  // At x = (2, 2), the residual is (0, 2). A step leaves actor 0 at exactly 0, since its
  // preconditioned residual is 0, and actor 1 at 1/4. Actor 1's norm scales both actors.
  ColumnVector<real> const x0({{2_r, 2_r}});
  auto const weight1 = ColumnVector<real>::Ones(1);
  real constexpr kMaxReal = std::numeric_limits<real>::max();
  for (real const w0 : {100_r, kMaxReal}) {
    auto const weight0 = ColumnVector<real>({{w0}});
    auto problem = GetTwoActorMixedProblem(weight0, weight1);
    for (auto const& [numNewtonIters, actor1Norm] : {std::pair{1, 2_r}, std::pair{2, 0.25_r}}) {
      auto const [linearParams, weights] = GetLastLinearSolve(problem, x0, params, numNewtonIters);
      ASSERT_EQ(2, weights.Rows());
      EXPECT_EQ(Min(w0 / Sqr(actor1Norm), kMaxReal), weights[0]);
      EXPECT_EQ(1_r / Sqr(actor1Norm), weights[1]);
      EXPECT_EQ(static_cast<double>(1e-9_r / actor1Norm), linearParams.absTol);
    }
  }

  // A tiny absTol scales actor 0, so that 1/scale² overflows in FP32.
  struct Case {
    real absTol;
    real w0;
    real expected;
  };
  for (auto const& [absTol, w0, expected] :
       {Case{
            .absTol = std::ldexp(1_r, -68),
            .w0 = std::ldexp(1_r, -16),
            .expected = std::ldexp(1_r, 120)},
        Case{.absTol = std::ldexp(1_r, -149), .w0 = 0_r, .expected = 0_r}}) {
    params.absTolRes = absTol;
    auto const weight0 = ColumnVector<real>({{w0}});
    auto problem = GetTwoActorMixedProblem(weight0, weight1);
    auto const [linearParams, weights] = GetLastLinearSolve(problem, x0, params, 1);
    ASSERT_EQ(2, weights.Rows());
    EXPECT_EQ(expected, weights[0]);
    EXPECT_EQ(0.25_r, weights[1]);
  }
}

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

/**
 * Where x[0] < −1, the residual and the objective are NaN or ±∞. From (1, 1), the full step lands
 * there and must be rejected, and every line search accepts the half step, unless it tests the
 * merit and its start merit is that value too. From (−2, −2), every line search fails both trials.
 * Accepted or not, the search ends at the half step.
 */
TEST(NewtonSolver, LineSearchRejectsNonFiniteValues) {
  real invalidValue = 0_r;
  auto residual = ColumnVector<real>::Zero(2);
  SnleProblemFunctions<real> functions;
  functions.assemble = [&invalidValue, &residual](
                           SnleProblem<real>& problem, AssemblyParams const& params) {
    auto const x = problem.GetSolution();
    bool const isInvalid = x[0] < -1_r;
    if (params.assemObj) {
      problem.objective = isInvalid ? invalidValue : 0.5 * x.Dot(x);
    }
    if (params.assemRes) {
      residual = x;
      if (isInvalid) {
        residual.SetConstant(invalidValue);
      }
      problem.actorResiduals = {{0, &residual}};
    }
  };
  functions.onPostNewIncrement = [](SnleProblem<real>& p) { p.solution += p.increment; };
  SnleProblem<real> problem{2, 2, std::move(functions)};

  using Ls = LineSearch<real>;
  for (real const value : {std::numeric_limits<real>::quiet_NaN(), -kInf, kInf}) {
    invalidValue = value;
    for (auto const& [type, search] :
         {std::pair{LineSearchType::Simple, &Ls::Simple},
          std::pair{LineSearchType::Armijo, &Ls::Armijo},
          std::pair{LineSearchType::WolfeWeak, &Ls::WolfeWeak},
          std::pair{LineSearchType::WolfeStrong, &Ls::WolfeStrong},
          std::pair{LineSearchType::ResidualNorm, &Ls::ResidualNorm}}) {
      LineSearchParams const params{
          .type = type,
          .maxIter = 2,
          .alpha = 0.5_r,
          .wolfe1 = 1e-4_r,
          .wolfe2 = 0.9_r,
          .maxRelIncrease = 0_r};
      for (auto const& [start, step, invalidStartMerit] :
           {std::tuple{1_r, -3_r, false},
            std::tuple{1_r, -3_r, true},
            std::tuple{-2_r, 3_r, true}}) {
        problem.SetSolution(ColumnVector<real>({{start, start}}));
        problem.increment.SetConstant(step);
        problem.UpdateObjResDRes({.assemObj = true, .assemRes = true});
        NewtonSolverStatus<real> status;
        status.objective = invalidStartMerit ? value : problem.GetObjective();

        bool const accepts =
            (start == 1_r) && (!invalidStartMerit || type == LineSearchType::ResidualNorm);
        EXPECT_EQ(search(problem, status, params), accepts);
        EXPECT_EQ(status.numLastLSIterDone, accepts ? 1 : params.maxIter);
        EXPECT_EQ(problem.GetSolution()[0], -0.5_r);
      }
    }
  }
}

/**
 * Two-DoF problem for the step dx = (1, 0) from the zero solution, where the objective is merit0
 * and the residual (startSlope, 0). Elsewhere, the residual is (slope, 1), and the objective is
 * merit0 + rise up to the step maxScale dx and merit0 + |merit0| beyond. They include a
 * non-potential term, whose energy rises by nonPotentialRise from the start and whose residual is
 * (2 nonPotentialResidual, 0) at the start and (nonPotentialResidual, 0) elsewhere.
 */
static SnleProblem<real> GetLineSearchTrialProblem(
    double merit0,
    real startSlope,
    double rise,
    real slope,
    real maxScale,
    double nonPotentialRise,
    real nonPotentialResidual) {
  auto residual = ColumnVector<real>{2};
  auto nonPotential = ColumnVector<real>::Zero(2);
  auto assembleFn = [=](SnleProblem<real>& problem, AssemblyParams const& params) mutable {
    real const x = problem.GetSolution()[0];
    if (params.assemObj) {
      problem.objective = (x == 0_r) ? merit0
          : (x <= maxScale)          ? merit0 + rise
                                     : merit0 + Abs(merit0);
      problem.nonPotentialEnergy = (x == 0_r ? -0.5 : 0.5) * nonPotentialRise;
      nonPotential[0] = (x == 0_r ? 2_r : 1_r) * nonPotentialResidual;
      problem.nonPotentialResiduals = {{0, &nonPotential}};
    }
    if (params.assemRes) {
      residual[0] = (x == 0_r) ? startSlope : slope;
      residual[1] = (x == 0_r) ? 0_r : 1_r;
      problem.actorResiduals.resize(1);
      problem.actorResiduals[0] = std::make_pair(0, &residual);
    }
  };

  SnleProblemFunctions<real> functions;
  functions.assemble = std::move(assembleFn);
  functions.onPostNewIncrement = [](SnleProblem<real>& problem) {
    problem.solution += problem.increment;
  };
  int constexpr kNumDofs = 2;
  return SnleProblem<real>{kNumDofs, kNumDofs, std::move(functions)};
}

// A merit-based line search treats a trial that fails its merit test, but changes the merit by no
// more than the band, as passing that test if its slope passes the merit test's Armijo condition;
// the Wolfe searches still apply their curvature test. Out of the band, and for trials that pass
// the merit test, the slope test doesn't apply. Steps longer than maxScale dx rise by |merit0|,
// which every merit test and the band reject: by default, the first two trials do, so the third, at
// step scale 1/4, decides. The Wolfe searches never step beyond dx. A trial still descending too
// steeply for their curvature test is accepted at dx and bounds the step from below otherwise, and
// if no trial passes both tests, they take the longest such trial, if any.
TEST(NewtonSolver, MeritLineSearches) {
  using Search = bool (*)(SnleProblem<real>&, NewtonSolverStatus<real>&, LineSearchParams const&);
  std::array<std::pair<LineSearchType, Search>, 4> constexpr kSearches = {
      {{LineSearchType::Simple, &LineSearch<real>::Simple},
       {LineSearchType::Armijo, &LineSearch<real>::Armijo},
       {LineSearchType::WolfeWeak, &LineSearch<real>::WolfeWeak},
       {LineSearchType::WolfeStrong, &LineSearch<real>::WolfeStrong}}};
  real constexpr kEps = std::numeric_limits<real>::epsilon();
  // Rises and slopes are in units of |merit0|, so the band is 2000 kEps. The start slope -kG is
  // small enough that a merit drop of kEps passes every merit test, in both precisions.
  real constexpr kG = 1000_r * kEps;
  real constexpr kSteep = -0.95_r * kG;
  real constexpr kNaN = std::numeric_limits<real>::quiet_NaN();
  std::array<bool, 4> constexpr kAllAccept = {true, true, true, true};
  struct Case {
    double rise{};
    real startSlope{};
    real slope{};
    std::array<bool, 4> accepted{}; // In the order of kSearches
    double nonPotentialRise{};
    real nonPotentialResidual{};
    real maxScale = 0.25_r;
    int maxIter = 3;
    std::array<real, 4> scale = {0.25_r, 0.25_r, 0.25_r, 0.25_r}; // Final step scales
  };
  for (double const merit0 : {4.0, -0.25}) {
    real const unit = Abs(real(merit0));
    // clang-format off
    for (Case const& c :
         {Case{1500 * kEps, -kG, -0.5_r * kG, {true, true, true, true}},
          Case{1500 * kEps, -kG, 1.5_r * kG, {false, false, false, false}},
          Case{1500 * kEps, -kG, 0.9999_r * kG, {true, false, false, false}}, // c1 = 0
          Case{1500 * kEps, -kG, kSteep, {true, true, true, true}}, // Wolfe falls back on it
          Case{1500 * kEps, -kG, -kInf, {false, false, false, false}},
          Case{1500 * kEps, -kInf, -0.5_r * kG, {false, false, false, false}},
          // Uphill, NaN or +∞ start slope: Armijo and Wolfe fail every trial and keep the last one
          Case{4000 * kEps, kG, -0.5_r * kG, {false, false, false, false}},
          Case{-10 * kEps, kNaN, -0.5_r * kG, {true, false, false, false}},
          Case{-10 * kEps, kInf, -0.5_r * kG, {true, false, false, false}},
          Case{4000 * kEps, -kG, -0.5_r * kG, {false, false, false, false}}, // Out of band
          // Out-of-band drop that fails the Armijo test
          Case{-3000 * kEps, -2e8_r * kEps, -4e8_r * kEps, {true, false, false, false}},
          // In-band drop that fails the Armijo test
          Case{-10 * kEps, -1e6_r * kEps, -0.5e6_r * kEps, {true, true, true, true}},
          Case{-10 * kEps, -kG, 2_r * kG, {true, true, true, false}}, // Merit test passes
          Case{kNaN, -kG, -0.5_r * kG, {false, false, false, false}}, // NaN merit
          // The merit replaces the energy of a non-potential term by its work, here zero: it falls
          // while the objective rises, rises while it falls, or changes within the band
          Case{4000 * kEps, -kG, -0.5_r * kG, {true, true, true, true}, 8000 * kEps},
          Case{-10 * kEps, -kG, -0.5_r * kG, {false, false, false, false}, -8000 * kEps},
          Case{4000 * kEps, -kG, -0.5_r * kG, {true, true, true, true}, 3000 * kEps},
          // Its work is the trapezoid rule on its residual: it turns the rise into a drop, or into
          // an in-band rise that the uphill slope fails
          Case{4000 * kEps, -kG, 1.5_r * kG, {true, true, true, false}, {}, -12e3_r * kEps},
          Case{4000 * kEps, -kG, 1.5_r * kG, {false, false, false, false}, {}, -8e3_r * kEps},
          // Steep drops up to maxScale dx: Wolfe accepts dx, ...
          Case{-10 * kEps, -kG, kSteep, kAllAccept, {}, {}, 2_r, 3, {1_r, 1_r, 1_r, 1_r}},
          // ... bisects towards dx and takes the last trial, ...
          Case{-10 * kEps, -kG, kSteep, kAllAccept, {}, {}, 0.8_r, 3, {0.5_r, 0.5_r, 0.75_r, 0.75_r}},
          // ... or an earlier one, assembled again, ...
          Case{-10 * kEps, -kG, kSteep, kAllAccept, {}, {}, 0.6_r, 3, {0.5_r, 0.5_r, 0.5_r, 0.5_r}},
          // ... also once the bisection has closed in on maxScale
          Case{-10 * kEps, -kG, kSteep, kAllAccept, {}, {}, 0.6_r, 1000, {0.5_r, 0.5_r, 0.6_r, 0.6_r}}}) {
      // clang-format on
      std::array<bool, 4> accepted{};
      for (int i = 0; i < isize(kSearches); ++i) {
        auto const [type, search] = kSearches[i];
        SnleProblem<real> problem = GetLineSearchTrialProblem(
            merit0,
            unit * c.startSlope,
            unit * c.rise,
            unit * c.slope,
            c.maxScale,
            unit * c.nonPotentialRise,
            unit * c.nonPotentialResidual);
        problem.SetSolution(ColumnVector<real>::Zero(2));
        problem.UpdateObjResDRes({.assemObj = true, .assemRes = true});
        problem.increment[0] = 1_r;
        problem.increment[1] = 0_r;
        NewtonSolverStatus<real> status;
        status.objective = problem.GetObjective();
        accepted[i] = search(
            problem,
            status,
            {.type = type,
             .maxIter = c.maxIter,
             .alpha = 0.5_r,
             .wolfe1 = 1e-4_r,
             .wolfe2 = 0.9_r,
             .maxRelIncrease = 0_r});

        EXPECT_NEAR(problem.GetSolution()[0], c.scale[i], 1e-5_r);
        EXPECT_EQ(problem.GetIncrement()[0], problem.GetSolution()[0]); // The step taken
        if (accepted[i]) { // Assembled, with its objective in status.objective
          EXPECT_FALSE(problem.HasSolutionChangedSinceLastAssembly());
          EXPECT_EQ(status.objective, problem.GetObjective());
        }
        EXPECT_LT(status.numLastLSIterDone, 100); // Wolfe stops once the bisection closes in
      }
      EXPECT_EQ(accepted, c.accepted);
    }
  }
}

TEST(NewtonSolver, LinearSolveTimeIsPartOfSolveTime) {
  auto w0 = ColumnVector<real>::Ones(1);
  auto w1 = ColumnVector<real>::Ones(1);
  auto problem = GetTwoActorMixedProblem(w0, w1);

  NewtonSolverParams params;
  params.absTolRes = 1e-4_r;
  params.maxIter = 10;
  NewtonSolver<real> solver(params);
  problem.SetSolution(ColumnVector<real>({{0_r, 2_r}}));
  auto const status = solver.Solve(problem);

  ASSERT_GT(status.numIterDone, 0);
  EXPECT_GT(status.linearSolveSec, 0.0);
  EXPECT_LE(status.linearSolveSec, ToSeconds(status.timer.GetElapsed()));
}
