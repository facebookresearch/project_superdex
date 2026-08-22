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

#include <mochi_core/solvers/newton_solver_params.h>
#include <mochi_core/solvers/newton_solver_status.h>
#include <mochi_core/solvers/snle_problem.h>

#include <cmath>
#include <limits>

namespace mochi {

/**
 * Utility struct implementing line search algorithms. It has no internal state.
 */
template <typename T>
struct LineSearch {
  using Problem = SnleProblem<T>;
  using Status = NewtonSolverStatus<T>;
  using LsParams = LineSearchParams;

  /**
   * No line search. Just take the step.
   */
  static bool None(Problem& problem, Status& status, LsParams const& params);

  /**
   * Simple backtracking line search with improvement in the merit function.
   */
  static bool Simple(Problem& problem, Status& status, LsParams const& params);

  /**
   * Backtracking line search with Armijo condition for improvement.
   */
  static bool Armijo(Problem& problem, Status& status, LsParams const& params);

  /**
   * Backtracking line search with Wolfe weak conditions for improvement.
   */
  static bool WolfeWeak(Problem& problem, Status& status, LsParams const& params);

  /**
   * Backtracking line search with Wolfe strong conditions for improvement.
   */
  static bool WolfeStrong(Problem& problem, Status& status, LsParams const& params);

  /**
   * Backtracking line search with residual norm condition for improvement.
   */
  static bool ResidualNorm(Problem& problem, Status& status, LsParams const& params);

  /**
   * Backtracking line search that accepts either the Armijo condition or the residual norm
   * condition for improvement.
   */
  static bool ArmijoOrResidualNorm(Problem& problem, Status& status, LsParams const& params);
};

template <typename T>
bool LineSearch<T>::None(
    Problem& problem,
    Status& /*status*/,
    [[maybe_unused]] LsParams const& params) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(params.type == LineSearchType::None, "Inconsistent line search type.");

  // Update the current solution without performing line search
  problem.UpdateSolution();

  return true;
}

template <typename T>
bool LineSearch<T>::Simple(Problem& problem, Status& status, LsParams const& params) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(params.type == LineSearchType::Simple, "Inconsistent line search type.");
  MOCHI_ASSERT(
      params.maxIter > 0,
      "At least 1 line search iteration is required. To run without line search, 'None' line search type must be used.");

  // Save current solution
  ColumnVector<T> const sol0(problem.GetSolution());

  bool improved = false;
  int& it = status.numLastLSIterDone;
  for (it = 0; it < params.maxIter; ++it) {
    // Compute new candidate solution
    if (it > 0) {
      problem.SetSolution(sol0, /*invokePost*/ false);
      problem.ScaleIncrement(params.alpha);
    }
    problem.UpdateSolution();

    // Compute objective at x + dx position
    problem.UpdateObjective();
    double const merit = problem.GetObjective();

    // Check enough improvement
    if (merit <= (status.merit + std::abs(status.merit * params.maxRelIncrease))) {
      status.merit = merit;
      improved = true;
      break;
    }
  }

  return improved;
}

template <typename T>
bool LineSearch<T>::Armijo(Problem& problem, Status& status, LsParams const& params) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(params.type == LineSearchType::Armijo, "Inconsistent line search type.");
  MOCHI_ASSERT(
      params.maxIter > 0,
      "At least 1 line search iteration is required. To run without line search, 'None' line search type must be used.");

  // Save current solution
  ColumnVector<T> const sol0(problem.GetSolution());

  // Compute Armijo factor
  double const dotDx0Res0 = problem.GetResidual().Dot(problem.GetIncrement());
  MOCHI_ASSERT_VERBOSE(dotDx0Res0 <= 0, "Armijo line search requires a descent direction.");
  double const factor0 = dotDx0Res0 * params.wolfe1;

  bool improved = false;
  auto scale = T(1);
  int& it = status.numLastLSIterDone;
  for (it = 0; it < params.maxIter; ++it) {
    // Compute new candidate solution
    if (it > 0) {
      problem.SetSolution(sol0, /*invokePost*/ false);
      scale *= params.alpha;
      problem.ScaleIncrement(params.alpha);
    }
    problem.UpdateSolution();

    // Compute merit at x + dx position
    problem.UpdateObjective();
    double const merit = problem.GetObjective();

    // Check enough improvement
    if (merit <= status.merit + scale * factor0) {
      status.merit = merit;
      improved = true;
      break;
    }
  }

  return improved;
}

template <typename T>
bool LineSearch<T>::WolfeWeak(Problem& problem, Status& status, LsParams const& params) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(params.type == LineSearchType::WolfeWeak, "Inconsistent line search type.");
  MOCHI_ASSERT(
      params.maxIter > 0,
      "At least 1 line search iteration is required. To run without line search, 'None' line search type must be used.");

  // Save current solution
  ColumnVector<T> const sol0(problem.GetSolution());

  // Compute Wolfe factors
  double const dotDx0Res0 = problem.GetResidual().Dot(problem.GetIncrement());
  MOCHI_ASSERT_VERBOSE(dotDx0Res0 <= 0, "Wolfe line search requires a descent direction.");
  double const t1 = params.wolfe1 * dotDx0Res0;
  double const t2 = params.wolfe2 * dotDx0Res0;

  bool improved = false;
  auto scale = T(1);
  auto scaleMin = T(0);
  auto scaleMax = std::numeric_limits<T>::infinity();
  int& it = status.numLastLSIterDone;
  for (it = 0; it < params.maxIter; ++it) {
    MOCHI_ASSERT(
        scaleMin <= scaleMax, "Inconsistent scale bounds: [%.5e, %.5e].\n", scaleMin, scaleMax);

    // Compute new candidate solution
    if (it > 0) {
      problem.SetSolution(sol0, /*invokePost*/ false);
    }
    problem.UpdateSolution();

    // Compute merit and gradient at x + dx position
    problem.UpdateObjResDRes(
        {.assemObj = true, .assemRes = true, .assemDRes = false, .psdDRes = false});

    // Compute values for Wolfe conditions
    double const dotDxRes = problem.GetResidual().Dot(problem.GetIncrement());
    double const merit = problem.GetObjective();

    // Check current iterate
    if (!std::isfinite(dotDxRes) || !std::isfinite(merit) || (merit > status.merit + scale * t1)) {
      // If not finite or 1st Wolfe condition is not met, reduce the scale.
      scaleMax = scale;
      T const scaleNew = scaleMin + params.alpha * (scale - scaleMin);
      problem.ScaleIncrement(scaleNew / scale);
      scale = scaleNew;
    } else if (dotDxRes < t2 * scale) {
      // If 1st Wolfe condition is met but 2nd is not, increase the scale.
      scaleMin = scale;
      T const scaleNew =
          std::isfinite(scaleMax) ? T(0.5) * (scale + scaleMax) : scale / params.alpha;
      problem.ScaleIncrement(scaleNew / scale);
      scale = scaleNew;
    } else {
      // If both conditions are met, update merit and exit line search.
      status.merit = merit;
      improved = true;
      break;
    }
  }

  return improved;
}

template <typename T>
bool LineSearch<T>::WolfeStrong(Problem& problem, Status& status, LsParams const& params) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(
      params.type == LineSearchType::WolfeStrong, "Inconsistent line search type.");
  MOCHI_ASSERT(
      params.maxIter > 0,
      "At least 1 line search iteration is required. To run without line search, 'None' line search type must be used.");

  // Save current solution
  ColumnVector<T> const sol0(problem.GetSolution());

  // Compute Wolfe factors
  double const dotDx0Res0 = problem.GetResidual().Dot(problem.GetIncrement());
  MOCHI_ASSERT_VERBOSE(dotDx0Res0 <= 0, "Wolfe line search requires a descent direction.");
  double const t1 = params.wolfe1 * dotDx0Res0;
  double const t2 = params.wolfe2 * dotDx0Res0;

  bool improved = false;
  auto scale = T(1);
  auto scaleMin = T(0);
  auto scaleMax = std::numeric_limits<T>::infinity();
  int& it = status.numLastLSIterDone;
  for (it = 0; it < params.maxIter; ++it) {
    MOCHI_ASSERT(
        scaleMin <= scaleMax, "Inconsistent scale bounds: [%.5e, %.5e].\n", scaleMin, scaleMax);

    // Compute new candidate solution
    if (it > 0) {
      problem.SetSolution(sol0, /*invokePost*/ false);
    }
    problem.UpdateSolution();

    // Compute merit and gradient at x + dx position
    problem.UpdateObjResDRes(
        {.assemObj = true, .assemRes = true, .assemDRes = false, .psdDRes = false});

    // Compute values for Wolfe conditions
    double const dotDxRes = problem.GetResidual().Dot(problem.GetIncrement());
    double const merit = problem.GetObjective();

    if (!std::isfinite(dotDxRes) || !std::isfinite(merit) || (merit > status.merit + scale * t1) ||
        (dotDxRes > std::abs(t2 * scale))) {
      // If not finite, 1st Wolfe condition is not met, or 2nd Wolfe condition is not met by above,
      // reduce the scale.
      scaleMax = scale;
      T const scaleNew = scaleMin + params.alpha * (scale - scaleMin);
      problem.ScaleIncrement(scaleNew / scale);
      scale = scaleNew;
    } else if (dotDxRes < -std::abs(t2 * scale)) {
      // If 2nd Wolfe condition is not met by below, increase the scale.
      scaleMin = scale;
      T const scaleNew =
          std::isfinite(scaleMax) ? T(0.5) * (scale + scaleMax) : scale / params.alpha;
      problem.ScaleIncrement(scaleNew / scale);
      scale = scaleNew;
    } else {
      // If both conditions are met, update merit and exit line search.
      status.merit = merit;
      improved = true;
      break;
    }
  }

  return improved;
}

template <typename T>
bool LineSearch<T>::ResidualNorm(Problem& problem, Status& status, LsParams const& params) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(
      params.type == LineSearchType::ResidualNorm, "Inconsistent line search type.");
  MOCHI_ASSERT(
      params.maxIter > 0,
      "At least 1 line search iteration is required. To run without line search, 'None' line search type must be used.");

  // Save current solution
  ColumnVector<T> const sol0(problem.GetSolution());

  // Compute current residual norm
  auto const resNorm0 = problem.GetResidual().Norm();

  bool improved = false;
  int& it = status.numLastLSIterDone;
  for (it = 0; it < params.maxIter; ++it) {
    // Compute new candidate solution
    if (it > 0) {
      problem.SetSolution(sol0, /*invokePost*/ false);
      problem.ScaleIncrement(params.alpha);
    }
    problem.UpdateSolution();

    // Compute the residual at x + dx
    problem.UpdateResidual();
    auto const resNorm = problem.GetResidual().Norm();

    if (resNorm <= resNorm0) {
      improved = true;
      break;
    }
  }

  return improved;
}

template <typename T>
bool LineSearch<T>::ArmijoOrResidualNorm(Problem& problem, Status& status, LsParams const& params) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(
      params.type == LineSearchType::ArmijoOrResidualNorm, "Inconsistent line search type.");
  MOCHI_ASSERT(
      params.maxIter > 0,
      "At least 1 line search iteration is required. To run without line search, 'None' line search type must be used.");

  // Compute Armijo factor. Armijo requires a descent direction. If it's not a descent direction,
  // fall back to residual norm.
  double const dotDx0Res0 = problem.GetResidual().Dot(problem.GetIncrement());
  if (dotDx0Res0 > 0) {
    auto localParams = params;
    localParams.type = LineSearchType::ResidualNorm;
    return ResidualNorm(problem, status, localParams);
  }
  double const factor0 = dotDx0Res0 * params.wolfe1;

  // Save current solution
  ColumnVector<T> const sol0(problem.GetSolution());

  // Compute current residual norm
  auto const resNorm0 = problem.GetResidual().Norm();

  bool improved = false;
  auto scale = T(1);
  int& it = status.numLastLSIterDone;
  for (it = 0; it < params.maxIter; ++it) {
    // Compute new candidate solution
    if (it > 0) {
      problem.SetSolution(sol0, /*invokePost*/ false);
      scale *= params.alpha;
      problem.ScaleIncrement(params.alpha);
    }
    problem.UpdateSolution();

    // Compute merit and residual at x + dx position, both in the same pass. Even if the residual is
    // not needed here, it is likely needed by the caller.
    problem.UpdateObjResDRes(
        AssemblyParams{.assemObj = true, .assemRes = true, .assemDRes = false});

    // Check improvement of the residual norm or enough improvement of the merit
    double const merit = problem.GetObjective();
    auto const resNorm = problem.GetResidual().Norm();
    if ((merit <= status.merit + scale * factor0) || (resNorm < resNorm0)) {
      status.merit = merit;
      improved = true;
      break;
    }
  }

  return improved;
}

} // namespace mochi
