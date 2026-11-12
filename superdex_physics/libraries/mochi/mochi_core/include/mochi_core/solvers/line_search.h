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
#include <mochi_core/utils/basic_utils.h>

#include <limits>

namespace mochi {

/**
 * Whether the line search type tests the merit.
 */
[[nodiscard]] constexpr bool LineSearchUsesMerit(LineSearchType type) {
  switch (type) {
    case LineSearchType::Simple:
    case LineSearchType::Armijo:
    case LineSearchType::WolfeWeak:
    case LineSearchType::WolfeStrong:
      return true;
    case LineSearchType::None:
    case LineSearchType::ResidualNorm:
      return false;
    case LineSearchType::Count:
      break;
  }
  static_assert(
      static_cast<int>(LineSearchType::Count) == 6,
      "Please update the switch statement above if LineSearchType enum changes");
  MOCHI_ASSERT(false, "Invalid line search type.");
  return false;
}

/**
 * Utility struct implementing line search algorithms. It has no internal state. Simple, Armijo and
 * the Wolfe searches test the merit of @ref SnleProblem::LineSearchMerit, and keep the objective in
 * @ref NewtonSolverStatus::objective.
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
   * Line search with Wolfe weak conditions for improvement. See @ref Wolfe.
   */
  static bool WolfeWeak(Problem& problem, Status& status, LsParams const& params);

  /**
   * Line search with Wolfe strong conditions for improvement. See @ref Wolfe.
   */
  static bool WolfeStrong(Problem& problem, Status& status, LsParams const& params);

  /**
   * Backtracking line search with residual norm condition for improvement.
   */
  static bool ResidualNorm(Problem& problem, Status& status, LsParams const& params);

 private:
  /**
   * Line search with Wolfe conditions, the 2nd one strong if @p strong, and steps no longer than
   * the full step. A trial that meets the 1st condition or the slope test but fails the 2nd by
   * below is accepted if it is the full step, and otherwise bounds the step from below. If no trial
   * meets both conditions, the line search takes the longest such trial, if any, assembled again
   * unless it was the last one.
   */
  static bool Wolfe(Problem& problem, Status& status, LsParams const& params, bool strong);

  /**
   * Slope test of the trials whose merit change is within an estimate of the merit's rounding
   * error.
   *
   * Near convergence, typically in single precision, the merit change of a good step can be
   * smaller than its rounding error, so the merit test can reject the step on rounding error. With
   * M0 and M1 the merits at x and at the trial x + s dx, ε the machine epsilon of T and
   * k = @ref kRoundingErrorFactor, a trial that fails the merit test with
   * |M1 - M0| <= 2 k ε |M0| is accepted if
   * F(x + s dx)·dx <= (2 c1 - 1) F(x)·dx, with F the residual and c1 the Armijo constant of the
   * merit test. This is the Armijo condition on (s / 2) (F(x)·dx + F(x + s dx)·dx), the
   * trapezoidal estimate of M1 - M0, which is exact on quadratic merits and free of the
   * cancellation in M1 - M0: Hager and Zhang's approximate Armijo condition (2005). It applies
   * only to trials that have a finite merit and fail the merit test, and never accepts one whose
   * slope or slope bound is not finite.
   */
  // TODO: Explore applying the slope test also to trials that pass the merit test within the band.
  class RoundingErrorSlopeTest {
   public:
    /**
     * @param[in] merit0 Merit M0 at the start of the line search.
     * @param[in] slope0 F(x)·dx at the start of the line search.
     * @param[in] c1 Armijo constant c1 of the merit test.
     */
    RoundingErrorSlopeTest(double merit0, double slope0, double c1)
        : _merit0(merit0),
          _band(2.0 * kRoundingErrorFactor * std::numeric_limits<T>::epsilon() * Abs(merit0)),
          _slopeBound((2.0 * c1 - 1.0) * slope0) {}

    /**
     * Whether the trial, with finite merit @p merit and the increment holding @p scale times dx,
     * changes the merit within the band and passes the slope test. Assembles the residual if the
     * slope decides.
     */
    [[nodiscard]] bool Passes(Problem& problem, double merit, T scale) const {
      MOCHI_ASSERT_VERBOSE(IsFinite(merit), "The slope test requires a finite merit.");
      if (!IsFinite(_slopeBound) || !NearEqual(merit, _merit0, _band)) {
        return false;
      }
      problem.UpdateResidual(); // No-op if already assembled
      double const slope = problem.GetResidual().Dot(problem.GetIncrement());
      return IsFinite(slope) && slope <= scale * _slopeBound;
    }

   private:
    // The band allows each of the two merits a rounding error of k ε |M0|. |M0| stands in for the
    // rounding magnitudes of the merit's terms, which cancellation within the terms can make
    // orders of magnitude larger, hence the large k.
    // TODO: Explore per-term rounding magnitudes instead of |M0| for the band. See D125958065.
    static constexpr double kRoundingErrorFactor = 1000.0;

    double const _merit0;
    double const _band; // 2 k ε |M0|
    double const _slopeBound; // (2 c1 - 1) F(x)·dx
  };
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

  // Simple requires no sufficient decrease, hence c1 = 0
  RoundingErrorSlopeTest const slopeTest(
      status.objective, problem.GetResidual().Dot(problem.GetIncrement()), /*c1*/ 0.0);
  typename Problem::LineSearchMerit const searchMerit(problem);
  bool const finiteStart = IsFinite(status.objective);

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
    double const merit = searchMerit(scale);

    // Check enough improvement
    if (finiteStart && IsFinite(merit) &&
        (merit <= (status.objective + Abs(status.objective * params.maxRelIncrease)) ||
         slopeTest.Passes(problem, merit, scale))) {
      status.objective = problem.GetObjective();
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
  double const factor0 = dotDx0Res0 * params.wolfe1;
  RoundingErrorSlopeTest const slopeTest(status.objective, dotDx0Res0, params.wolfe1);
  typename Problem::LineSearchMerit const searchMerit(problem);
  bool const finiteStart = IsFinite(status.objective) && IsFinite(dotDx0Res0);

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
    double const merit = searchMerit(scale);

    // Check enough improvement
    if (finiteStart && IsFinite(merit) &&
        (merit <= status.objective + scale * factor0 || slopeTest.Passes(problem, merit, scale))) {
      status.objective = problem.GetObjective();
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
  return Wolfe(problem, status, params, /*strong*/ false);
}

template <typename T>
bool LineSearch<T>::WolfeStrong(Problem& problem, Status& status, LsParams const& params) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(
      params.type == LineSearchType::WolfeStrong, "Inconsistent line search type.");
  return Wolfe(problem, status, params, /*strong*/ true);
}

template <typename T>
bool LineSearch<T>::Wolfe(Problem& problem, Status& status, LsParams const& params, bool strong) {
  MOCHI_ASSERT(
      params.maxIter > 0,
      "At least 1 line search iteration is required. To run without line search, 'None' line search type must be used.");

  // Save current solution
  ColumnVector<T> const sol0(problem.GetSolution());

  // Compute Wolfe factors
  double const dotDx0Res0 = problem.GetResidual().Dot(problem.GetIncrement());
  double const t1 = params.wolfe1 * dotDx0Res0;
  double const t2 = params.wolfe2 * dotDx0Res0;
  RoundingErrorSlopeTest const slopeTest(status.objective, dotDx0Res0, params.wolfe1);
  typename Problem::LineSearchMerit const searchMerit(problem);
  bool const finiteStart = IsFinite(status.objective) && IsFinite(dotDx0Res0);

  bool improved = false;
  auto scale = T(1);
  auto scaleNew = scale;
  auto scaleMin = T(0);
  auto scaleMax = T(1);
  int& it = status.numLastLSIterDone;
  for (it = 0; it < params.maxIter; ++it) {
    MOCHI_ASSERT(
        scaleMin <= scaleMax, "Inconsistent scale bounds: [%.5e, %.5e].\n", scaleMin, scaleMax);

    // Compute new candidate solution, if the scale bounds leave room for it
    if (it > 0) {
      if (!(scaleMin < scaleNew && scaleNew < scaleMax)) {
        break;
      }
      problem.SetSolution(sol0, /*invokePost*/ false);
      problem.ScaleIncrement(scaleNew / scale);
      scale = scaleNew;
    }
    problem.UpdateSolution();

    // Compute merit and gradient at x + dx position
    problem.UpdateObjResDRes(
        {.assemObj = true, .assemRes = true, .assemDRes = false, .psdDRes = false});

    // Compute values for Wolfe conditions
    double const dotDxRes = problem.GetResidual().Dot(problem.GetIncrement());
    double const merit = searchMerit(scale);

    // TODO: Explore choosing the next trial by safeguarded interpolation of the trials' merits and
    // slopes.
    if (!finiteStart || !IsFinite(dotDxRes) || !IsFinite(merit) ||
        !(merit <= status.objective + scale * t1 || slopeTest.Passes(problem, merit, scale)) ||
        (strong && dotDxRes > Abs(t2 * scale))) {
      // If not finite, or neither the 1st Wolfe condition nor the slope test is met, or the strong
      // 2nd Wolfe condition is not met by above, reduce the scale.
      scaleMax = scale;
      scaleNew = scaleMin + params.alpha * (scale - scaleMin);
    } else if ((strong ? dotDxRes < -Abs(t2 * scale) : dotDxRes < t2 * scale) && scale < T(1)) {
      // If the 2nd Wolfe condition is not met by below, short of the full step, increase the scale.
      scaleMin = scale;
      scaleNew = T(0.5) * (scale + scaleMax);
    } else {
      // Otherwise, update the objective and exit line search.
      status.objective = problem.GetObjective();
      improved = true;
      break;
    }
  }

  // Failing that, take the longest trial that met the 1st Wolfe condition or the slope test and
  // failed the 2nd by below, if any.
  if (!improved && scaleMin > T(0)) {
    if (scale != scaleMin) {
      problem.SetSolution(sol0, /*invokePost*/ false);
      problem.ScaleIncrement(scaleMin / scale);
      problem.UpdateSolution();
      problem.UpdateObjResDRes(
          {.assemObj = true, .assemRes = true, .assemDRes = false, .psdDRes = false});
    }
    status.objective = problem.GetObjective();
    improved = true;
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
  bool const finiteStart = IsFinite(resNorm0);

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

    if (finiteStart && IsFinite(resNorm) && resNorm <= resNorm0) {
      improved = true;
      break;
    }
  }

  return improved;
}

} // namespace mochi
