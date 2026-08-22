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

#include <mochi_core/solvers/newton_solver.h>

#include <mochi_core/linear_algebra/krylov/preconditioner_utils.h>
#include <mochi_core/linear_algebra/low_rank_augmented_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/solvers/line_search.h>
#include <mochi_core/solvers/linear_solver.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/sparsity_utils.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_core/utils/time.h>

#include <limits>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

namespace mochi {

namespace {

/** @brief Result of an actor's convergence check. */
template <typename T>
struct ActorConvergenceResult {
  bool absConverged; // True if absolute criterion satisfied: |r|_W <= absTol
  bool relConverged; // True if relative criterion satisfied: |r|_W <= relTol * |r0|_W
  T weightedResNorm; // Current weighted residual norm |r|_W
  T resNormSqr; // Current unweighted squared residual norm |r|^2
};

/** @brief Weighted L2 convergence check. */
template <typename T>
ActorConvergenceResult<T> CheckWeightedConvergence(
    ColumnVectorView<T const> residual,
    ColumnVectorView<T const> weights,
    T initWeightedNorm,
    T absTol,
    T relTol) {
  MOCHI_ASSERT_VERBOSE(
      residual.Rows() == weights.Rows(), "Inconsistent actor residual and weight sizes.");
  T normSqr = {};
  T weightedNormSqr = {};
  for (int i = 0; i < residual.Rows(); ++i) {
    T const r2 = Sqr(residual(i));
    normSqr += r2;
    weightedNormSqr += weights(i) * r2;
  }
  T const weightedNorm = Sqrt(weightedNormSqr);
  return {
      .absConverged = (weightedNorm <= absTol),
      .relConverged = (initWeightedNorm > T{0}) && (weightedNorm <= relTol * initWeightedNorm),
      .weightedResNorm = weightedNorm,
      .resNormSqr = normSqr,
  };
}

} // namespace

template <typename T>
NewtonSolver<T>::NewtonSolver(
    NewtonSolverParams const& params,
    std::shared_ptr<PreconditionerRecyclingManager<T>> preconditionerRecyclingMgr)
    : _preconditionerRecyclingMgr(preconditionerRecyclingMgr) {
  SetParams(params);
}

template <typename T>
bool NewtonSolver<T>::StopCriterionType::StopCriterion_ResNorm(
    Problem const& problem,
    Status& status,
    Params const& params) {
  MOCHI_PROFILE_SCOPE();

  static_assert(
      static_cast<int>(NonLinearSolverConvergenceMode::Count) == 2,
      "Please update this function if NonLinearSolverConvergenceMode enumerator changes");
  bool usePerActor = (params.convergenceMode == NonLinearSolverConvergenceMode::PerActorWeighted);

  if (usePerActor && isize(problem.actorConvergenceWeights) != isize(problem.actorResiduals)) {
    MOCHI_LOG_WARNING_ONCE(
        "PerActorWeighted convergence mode is active but actorConvergenceWeights size "
        "does not match actorResiduals size. Falling back to Global mode. Either populate "
        "actorConvergenceWeights in the assembly callback, or set "
        "convergenceMode = NonLinearSolverConvergenceMode::Global.");
    usePerActor = false;
  }

  bool isConverged = false;
  if (usePerActor) {
    auto const& actorResiduals = problem.actorResiduals;

    if (status.numIterDone == 0) {
      status.actorResidualWeightedNorm0.resize_noinit(actorResiduals.size());
      status.actorConvergence.resize_noinit(actorResiduals.size());
    }

    // Compute per-actor convergence and accumulate global residual norm.
    status.resNorm = {};
    isConverged = true;
    auto const& globalResidual = problem.GetResidual();
    for (int i = 0; i < isize(actorResiduals); ++i) {
      auto const& [dofOffset, residualPtr] = actorResiduals[i];
      MOCHI_ASSERT_VERBOSE(residualPtr, "Invalid actor residual pointer.");
      auto const& actorResidual = globalResidual.MiddleRows(dofOffset, residualPtr->Rows());

      // Get the initial weighted norm (0 on first iteration to skip relative check).
      T const initWeightedNorm =
          (status.numIterDone == 0) ? T{0} : status.actorResidualWeightedNorm0[i];

      auto const& [weightOffset, weightsPtr] = problem.actorConvergenceWeights[i];
      MOCHI_ASSERT_VERBOSE(
          weightOffset == dofOffset,
          "Actor convergence weight offset does not match actor residual offset.");
      MOCHI_ASSERT_VERBOSE(weightsPtr, "Invalid actor weight pointer.");
      auto const result = CheckWeightedConvergence(
          actorResidual,
          AsConstView(*weightsPtr),
          initWeightedNorm,
          params.absTolRes,
          params.relTolRes);
      status.resNorm += result.resNormSqr;

      // Store the actor weighted norm on first iteration.
      if (status.numIterDone == 0) {
        status.actorResidualWeightedNorm0[i] = result.weightedResNorm;
      }

      // Check convergence.
      bool const isActorConverged = (result.absConverged || result.relConverged);

      status.actorConvergence[i] =
          isActorConverged ? ConvergenceStatus::Converged : ConvergenceStatus::Stopped;
      isConverged &= isActorConverged;
    }
    status.resNorm = Sqrt(status.resNorm);

  } else {
    status.resNorm = problem.GetResidual().Norm();
    bool const absConverged = (status.resNorm <= params.absTolRes);
    bool const relConverged =
        (status.numIterDone > 0) && (status.resNorm <= params.relTolRes * status.resNorm0);
    isConverged = (absConverged || relConverged);
  }

  if (isConverged) {
    status.convergence = ConvergenceStatus::Converged;
    status.stopReasonStr = "Converged";
  }

  return isConverged;
}

template <typename T>
bool NewtonSolver<T>::StopCriterionType::StopCriterion_RelStepNorm(
    Problem const& problem,
    Status& status,
    Params const& params) {
  MOCHI_PROFILE_SCOPE();
  if (status.numIterDone == 0) {
    // No step taken yet.
    return false;
  }

  // Compute the relative size of the last computed solution increment
  // TODO: Assess using a per-actor (possibly weighted) norm when convergenceMode =
  // NonLinearSolverConvergenceMode::PerActorWeighted.
  T const xNorm = problem.GetSolution().Norm();
  T const dxNorm = status.dxSolve.Norm();

  // Check if relative step norm is below the solution relative tolerance
  if (dxNorm <= params.solRelTol * xNorm) {
    status.convergence = ConvergenceStatus::Stopped;
    status.stopReasonStr = "|dx| <= solRelTol*|x|";
    return true;
  }

  return false;
}

template <typename T>
bool NewtonSolver<T>::StopCriterionType::StopCriterion_MaxIters(
    Problem const& /*problem*/,
    Status& status,
    Params const& params) {
  MOCHI_PROFILE_SCOPE();

  // Check the number if the number of iterations done is above the maximum
  if (status.numIterDone >= params.maxIter) {
    status.convergence = ConvergenceStatus::Stopped;
    status.stopReasonStr = "Maximum iterations";
    return true;
  }

  return false;
};

template <typename T>
bool NewtonSolver<T>::StopCriterionType::StopCriterion_MaxElapsedTime(
    Problem const& /*problem*/,
    Status& status,
    Params const& params) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(params.maxElapsedTime >= TimeSpan{0});
  if ((status.numIterDone > 0) && (params.maxElapsedTime > TimeSpan{0}) &&
      (status.timer.GetElapsed() >= params.maxElapsedTime)) {
    status.convergence = ConvergenceStatus::Stopped;
    status.stopReasonStr = "Maximum elapsed time";
    return true;
  } else {
    return false;
  }
};

template <typename T>
bool NewtonSolver<T>::StopCriterionType::StopCriterion_NoImprovement(
    Problem const& /*problem*/,
    Status& status,
    Params const& params) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(params.stopIfNoImprovement, "Inconsistent solver parameters.");
  if ((status.numIterDone > 0) && !status.improvedInLastIter) {
    if (params.verbosity >= VerbosityLevel::Verbose) {
      MOCHI_LOG("\n\tNo improvement in Newton iteration %i.\n", status.numIterDone);
    }
    status.convergence = ConvergenceStatus::Stopped;
    status.stopReasonStr = "No improvement";
    return true;
  } else {
    return false;
  }
};

// -----------------------------------------------------------------------------------------

template <typename T>
void NewtonSolver<T>::SetParams(NewtonSolverParams const& params) {
  MOCHI_PROFILE_SCOPE();

  if (((params.linearToleranceStrategy == LinearToleranceStrategy::EisenstatWalker1) ||
       (params.linearToleranceStrategy == LinearToleranceStrategy::EisenstatWalker3)) &&
      (params.lParams.normType != LinearSolverConvergenceNorm::ResidualL2)) {
    MOCHI_LOG_WARNING_ONCE(
        "Inconsistent solver settings: Eisenstat-Walker Choices #1 and #3 require using the L2-norm of "
        "the residual to monitor convergence of the linear solver."); // Could be fixed if needed.
  }

  if ((params.solverType == NonLinearSolverType::BFGS ||
       params.solverType == NonLinearSolverType::SR1) &&
      (params.dResidualAssemblyPeriod == 1)) {
    MOCHI_LOG_WARNING_ONCE(
        "Inconsistent solver settings: Quasi-Newton methods require dResidualAssemblyPeriod > 1. Using dResidualAssemblyPeriod = 1 is equivalent to using Newton's method.");
  }

  if ((params.solverType != NonLinearSolverType::Newton) &&
      (params.linearToleranceStrategy != LinearToleranceStrategy::Constant)) {
    MOCHI_LOG_WARNING_ONCE(
        "Suboptimal solver settings: When using a Quasi-Newton method, the linear tolerance strategy (linearToleranceStrategy) "
        "should be set to 'Constant' with a relatively loose linear relative tolerance. Adaptive linear tolerance strategies are "
        "designed for Newton's method.");
  }

  if (params.lineSearch.alpha <= 0_r || params.lineSearch.alpha >= 1_r) {
    MOCHI_LOG_WARNING_ONCE(
        "Line search alpha (%g) is invalid. Valid range is (0, 1).", params.lineSearch.alpha);
  }
  if (params.lineSearch.wolfe1 <= 0_r || params.lineSearch.wolfe1 >= 1_r) {
    MOCHI_LOG_WARNING_ONCE(
        "First Wolfe coefficient (%g) is invalid. Valid range is (0, 1).",
        params.lineSearch.wolfe1);
  }
  if (params.lineSearch.wolfe2 <= params.lineSearch.wolfe1 || params.lineSearch.wolfe2 >= 1_r) {
    MOCHI_LOG_WARNING_ONCE(
        "Second Wolfe coefficient (%g) is invalid. Valid range is (wolfe1, 1).",
        params.lineSearch.wolfe2);
  }

  _params = params;

  // Set stop criteria. Convergence criteria are checked first so that the reported convergence
  // status is always correct.
  _stopCriteria.clear();
  _stopCriteria.reserve(5);
  // ConvergenceStatus::Converged checks.
  _stopCriteria.push_back(StopCriterionType::StopCriterion_ResNorm);
  // ConvergenceStatus::Stopped checks.
  _stopCriteria.push_back(StopCriterionType::StopCriterion_MaxIters);
  _stopCriteria.push_back(StopCriterionType::StopCriterion_MaxElapsedTime);
  _stopCriteria.push_back(StopCriterionType::StopCriterion_RelStepNorm);
  if (params.stopIfNoImprovement) {
    _stopCriteria.push_back(StopCriterionType::StopCriterion_NoImprovement);
  }

  // Create the linear solver
  _linearSolver = std::make_unique<LinearSolver<T>>(params.lParams, _preconditionerRecyclingMgr);

  // Set line-search type.
  switch (params.lineSearch.type) {
    case LineSearchType::None:
      _lineSearch = LineSearch<T>::None;
      break;
    case LineSearchType::Simple:
      _lineSearch = LineSearch<T>::Simple;
      break;
    case LineSearchType::Armijo:
      _lineSearch = LineSearch<T>::Armijo;
      break;
    case LineSearchType::WolfeWeak:
      _lineSearch = LineSearch<T>::WolfeWeak;
      break;
    case LineSearchType::WolfeStrong:
      _lineSearch = LineSearch<T>::WolfeStrong;
      break;
    case LineSearchType::ResidualNorm:
      _lineSearch = LineSearch<T>::ResidualNorm;
      break;
    case LineSearchType::ArmijoOrResidualNorm:
      _lineSearch = LineSearch<T>::ArmijoOrResidualNorm;
      break;
    default:
      MOCHI_ASSERT(false, "Invalid line search type.");
      _lineSearch = LineSearch<T>::None; // Fall back to no line search.
  }
  static_assert(
      static_cast<int>(LineSearchType::Count) == 7,
      "Please update the switch statement above if LineSearchType enum changes");
}

template <typename T>
NewtonSolverParams const& NewtonSolver<T>::GetParams() const {
  return _params;
}

template <typename T>
bool NewtonSolver<T>::TakeStep(
    Problem& problem,
    Status& status,
    bool const takeFinalLsStepIfNoImprovement) const {
  MOCHI_PROFILE_SCOPE();

  // Save current solution
  auto sol = problem.GetSolution();
  ColumnVector<T> const sol0 = takeFinalLsStepIfNoImprovement
      ? ColumnVector<T>{}
      : sol.Duplicate(); // For performance, perform copy only if potentially needed

  // Line-search in dx direction
  bool const improved = _lineSearch(problem, status, _params.lineSearch);
  status.totalNumLSIterDone += status.numLastLSIterDone;

  // Restore previous solution if there is no improvement
  if (!improved && !takeFinalLsStepIfNoImprovement) {
    problem.SetSolution(sol0);
    problem.UpdateObjResDRes({.assemObj = true, .assemRes = true, .assemDRes = false});
    status.merit = problem.GetObjective();
  }

  if (_params.verbosity >= VerbosityLevel::Verbose) {
    T const stepNorm = problem.GetIncrement().Norm();
    if (!improved) {
      MOCHI_LOG(
          "\n\tFinished line-search without improvement after %i iter, |dx|=%.5e\n",
          status.numLastLSIterDone,
          stepNorm);
    } else {
      MOCHI_LOG(
          "\n\tSuccessfully took a step after %i line-search iter, |dx|=%.5e\n",
          status.numLastLSIterDone,
          stepNorm);
    }
  }

  return improved;
}

template <typename T>
bool NewtonSolver<T>::IsSolverDone(
    std::vector<StopCriterion> const& stopCriteria,
    ColumnVectorView<T const> initialSolution,
    Problem const& problem,
    Status& status) const {
  for (size_t i = 0; i < stopCriteria.size(); ++i) {
    if (stopCriteria[i](problem, status, _params)) {
      if (_params.verbosity >= VerbosityLevel::Verbose) {
        auto step = problem.GetSolution().Duplicate();
        step -= initialSolution;
        auto const stepSize = step.Norm();
        MOCHI_LOG(
            "\nFinished Newton-Raphson solve with |res|=%.5e, obj=%.5e |step|=%.5e\n"
            "Solver finished after iter=%i, with stopping reason: %s\n",
            status.resNorm,
            status.merit,
            stepSize,
            status.numIterDone,
            status.stopReasonStr.c_str());
      }
      return true;
    }
  }
  return false;
}

template <typename T>
void NewtonSolver<T>::SetLinearSolverRelativeTolerance(Status const& status, T linearResidualNorm) {
  // Maximum eta. Default in Eisenstat-Walker is 0.9. Using 0.01 since the assembly cost dominates
  // and solving the linear system to a lower tolerance pays off.
  // TODO:
  // - Assess making kMaxEta an increasing function of the number of DoFs (deterministic proxy for
  //   the relative cost of assembly vs. linear solve), e.g. blending between 1e-4 and 1e-2.
  // - Assess reducing kMaxEta if the previous non-linear solves exhibited near-linear behavior,
  //   e.g. when free falling.
  T constexpr kMaxEta = T(0.01);

  // Safeguard eta. Used only for EW1 and EW2. Default in Eisenstat-Walker is 0.1. Using 0.01 to
  // match maximum eta since that's already sufficiently low.
  T constexpr kSafeguardEtaTh = T(0.01);

  // Minimum eta. Used only for EW1 and EW2. Prevents attempting to solve beyond the floating-point
  // noise floor.
  // TODO: Assess max-clamping kMinEta to a fraction of the non-linear residual norm that is left to
  // achieve non-linear convergence.
  T constexpr kMinEta = T(100) * std::numeric_limits<T>::epsilon();
  static_assert(
      kMaxEta >= kSafeguardEtaTh && kSafeguardEtaTh >= kMinEta, "Inconsistent parameters");

  auto linearParams(_linearSolver->GetParams());
  T const prevEta = static_cast<T>(linearParams.relTol);
  T eta = {};
  switch (_params.linearToleranceStrategy) {
    case LinearToleranceStrategy::Constant: {
      return;
    }
    case LinearToleranceStrategy::EisenstatWalker1: {
      // 'Choice 1' in https://softlib.rice.edu/pub/CRPC-TRs/reports/CRPC-TR94463.pdf
      if (status.numIterDone == 0) {
        eta = kMaxEta;
      } else {
        T const safeguardEta = Pow(prevEta, T(0.5) + T(0.5) * Sqrt(T(5)));
        eta = Abs(status.resNorm - linearResidualNorm) / status.prevResNorm;
        if (safeguardEta > kSafeguardEtaTh) {
          eta = Max(eta, safeguardEta);
        }
        eta = Clamp(eta, kMinEta, kMaxEta);
      }
      break;
    }
    case LinearToleranceStrategy::EisenstatWalker2: {
      // 'Choice 2' in https://softlib.rice.edu/pub/CRPC-TRs/reports/CRPC-TR94463.pdf
      if (status.numIterDone == 0) {
        eta = kMaxEta;
      } else {
        T const alpha = T(0.5) + T(0.5) * Sqrt(T(5));
        T const gamma = T(0.9);
        T const safeguardEta = gamma * Pow(prevEta, alpha);
        eta = gamma * Pow(status.resNorm / status.prevResNorm, alpha);
        if (safeguardEta > kSafeguardEtaTh) {
          eta = Max(eta, safeguardEta);
        }
        eta = Clamp(eta, kMinEta, kMaxEta);
      }
      break;
    }
    case LinearToleranceStrategy::EisenstatWalker3: {
      auto const minEta = static_cast<T>(_params.lParams.relTol);
      eta = Max(Min(T(1) / (T(2) + status.numIterDone), Sqrt(status.resNorm)), minEta);
      break;
    }
    default: {
      MOCHI_ASSERT(false, "Invalid linear tolerance strategy.");
      eta = kMaxEta;
    }
  }
  static_assert(
      static_cast<int>(LinearToleranceStrategy::Count) == 4,
      "Please update the switch statement above if LinearToleranceStrategy enum changes");

  linearParams.relTol = static_cast<real>(eta);
  _linearSolver->SetParams(linearParams);
}

template <typename T>
void NewtonSolver<T>::ResetLinearOperator(
    Problem const& problem,
    bool useIslandOperators,
    AnyLinearOperator& outLinOp) {
  auto resetFn = [](AnyLinearOperator& linOp, auto&& newOp) {
    // Potential optimization (T189246854): When resetting a LowRankAugmentedMatrix, the memory for
    // the augmentation vectors could be reused.
    linOp.~AnyLinearOperator();
    new (&linOp) AnyLinearOperator(std::forward<decltype(newOp)>(newOp));
  };

  if (_params.solverType == NonLinearSolverType::Newton) {
    if (useIslandOperators) {
      resetFn(outLinOp, problem.GetOperators());
    } else {
      std::visit([&](auto&& dres) { resetFn(outLinOp, std::move(dres)); }, problem.GetDResidual());
    }
  } else {
    int capacity = 0;
    if (_params.solverType == NonLinearSolverType::BFGS) {
      capacity = 2 * (_params.dResidualAssemblyPeriod - 1);
    } else if (_params.solverType == NonLinearSolverType::SR1) {
      capacity = _params.dResidualAssemblyPeriod - 1;
    } else {
      MOCHI_ASSERT(false, "Unsupported solver type.");
    }
    if (useIslandOperators) {
      resetFn(outLinOp, LowRankAugmentedMatrix(problem.GetOperators(), capacity));
    } else {
      std::visit(
          [&](auto&& dres) {
            resetFn(outLinOp, LowRankAugmentedMatrix(std::move(dres), capacity));
          },
          problem.GetDResidual());
    }
  }
}

template <typename T>
void NewtonSolver<T>::AddLowRankUpdate(bool projectPsd, AnyLinearOperator& outLinOp) {
  std::visit(
      [&](auto& linOp) {
        if constexpr (IsLowRankAugmentedMatrix<decltype(linOp)>) {
          // Reference: Numerical Optimization, J. Nocedal and S.J. Wright, Chapter 6
          // (https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf)
          // Potential optimization (T189246854): 'y' could be computed without performing a full
          // matrix-vector product.
          _workspace.deltaRes = _workspace.residual - _workspace.prevIterResidual;
          linOp.Apply(_workspace.deltaSol, _workspace.y);

          constexpr T kEpsilon = kDefaultNearEqualEpsilon<T>;
          if (_params.solverType == NonLinearSolverType::BFGS) {
            T const c1 = _workspace.deltaSol.Dot(_workspace.deltaRes);
            T const c2 = _workspace.deltaSol.Dot(_workspace.y);
            T const deltaSolNorm = _workspace.deltaSol.Norm();
            if ((Abs(c1) > kEpsilon * deltaSolNorm * _workspace.deltaRes.Norm()) &&
                (Abs(c2) > kEpsilon * deltaSolNorm * _workspace.y.Norm()) &&
                (!projectPsd || (c1 > 0 && c2 > 0))) {
              _workspace.U.template LeftCols<1>(1) = _workspace.deltaRes;
              _workspace.U.template RightCols<1>(1) = _workspace.y;
              _workspace.V.template LeftCols<1>(1) = _workspace.deltaRes * (1 / c1);
              _workspace.V.template RightCols<1>(1) = _workspace.y * (-1 / c2);
              linOp.AddRankOneUpdates(_workspace.U, _workspace.V);
            }
          } else if (_params.solverType == NonLinearSolverType::SR1) {
            _workspace.U = _workspace.deltaRes - _workspace.y;
            T const c = _workspace.U.Dot(_workspace.deltaSol);
            if ((Abs(c) > kEpsilon * _workspace.deltaSol.Norm() * _workspace.U.Norm()) &&
                (!projectPsd || c > 0)) {
              _workspace.V = _workspace.U * (1 / c);
              linOp.AddRankOneUpdates(_workspace.U, _workspace.V);
            }
          } else {
            MOCHI_ASSERT(false, "Unexpected solver type.");
          }
        } else {
          // Allow Newton with dResidualAssemblyPeriod > 1, but don't update the linear operator
          MOCHI_ASSERT(
              _params.solverType == NonLinearSolverType::Newton, "Unexpected solver type.");
        }
      },
      outLinOp);
}

template <typename T>
void NewtonSolver<T>::PrepareLinearOperator(
    Problem const& problem,
    Status const& status,
    bool useIslandOperators,
    bool projectPsd,
    AnyLinearOperator& outLinOp) {
  if (status.itersSinceDResidualAssembly == 0) {
    ResetLinearOperator(problem, useIslandOperators, outLinOp);
  } else {
    AddLowRankUpdate(projectPsd, outLinOp);
  }
}

template <typename T>
NewtonSolverStatus<T> NewtonSolver<T>::Solve(Problem& problem) {
  MOCHI_PROFILE_SCOPE();

  // Set all dirty flags in case the same Problem structure is re-used, e.g. with multi-stage time
  // integration schemes and in unit tests.
  problem.InvalidateCachedData();

  // Prepare workspace.
  _workspace.Resize(problem.GetDofsSize(), _params);

  // Create solver result
  Status status;
  status.dxSolve.Resize(problem.GetDofsSize());
  status.dxSolve.SetConstant(std::numeric_limits<real>::infinity());

  // Get active projection mode
  PsdProjectionMode psdMode = _params.psdProjMode;

  // Prepare vector views
  auto dxSolve = AsView(status.dxSolve);

  // TODO Explore whether we can avoid accessing `SnleProblem<T>::increment`
  auto dxStep = AsView(problem.increment);

  // Owning copy of the initial solution.
  ColumnVector<T> const initialSolution = problem.GetSolution();

  // Linear operator with the approximate DResidual for the linear solve.
  AnyLinearOperator linOp = {};

  T linearResidualNorm = {}; // Expressed in the norm used to monitor linear solver convergence
  while (true) {
    MOCHI_PROFILE_SCOPE_N("SolverIteration");
    bool const isFirstIter = (status.numIterDone == 0);
    ++status.itersSinceDResidualAssembly;

    // Save the step from the previous iteration. Used with quasi-Newton methods in
    // PrepareLinearOperator. Saved early since consistency checks may modify it.
    if (!isFirstIter) {
      _workspace.SaveDeltaSol(problem);
    }

    // Full finite difference consistency test. For debugging purposes.
    if (_params.consistencyResDRes) {
      problem.ConsistencyCheckResDRes(
          _params.consistencyResDResStep, _params.consistencyResDResLogEntries);
    }

    // Finite difference consistency test of the residual norm. For debugging purposes.
    if (_params.consistencyResNorm) {
      real const error =
          problem.ConsistencyCheckResNorm(_params.consistencyResNormStep, _params.verbosity);
      status.resNormError = Max(status.resNormError, error);
    }

    // Only start the first trial projecting if the PSD mode is Always.
    bool projectPsd = (psdMode == PsdProjectionMode::Always);

    // Only start the first trial with fitted Hessian if requested externally.
    auto fittedSaturationHessian = _params.fittedSaturationHessian;

    // Compute residual (always) and merit (only if using a merit-based line search).
    // On the first step we compute dresidual at the same time, assuming we will need it.
    {
      AssemblyParams assemblyParams;
      static_assert(
          static_cast<int>(LineSearchType::Count) == 7,
          "Please make sure the logic below is correct if LineSearchType enum changes");
      assemblyParams.assemObj = (_params.lineSearch.type != LineSearchType::None) &&
          (_params.lineSearch.type != LineSearchType::ResidualNorm);
      assemblyParams.assemRes = true;
      assemblyParams.assemDRes = isFirstIter;
      assemblyParams.psdDRes = projectPsd;
      assemblyParams.fittedSaturationHessian = fittedSaturationHessian;
      problem.UpdateObjResDRes(assemblyParams);
      status.resNorm = problem.GetResidual().Norm();
      if (assemblyParams.assemDRes) {
        status.itersSinceDResidualAssembly = 0;
      }
      if (assemblyParams.assemObj) {
        status.merit = problem.GetObjective();
      }
      _workspace.AdvanceResidual(problem, /*savePrevious*/ !isFirstIter);
    }

    // Set initial residual and merit.
    if (isFirstIter) {
      status.resNorm0 = status.resNorm;
      status.merit0 = status.merit; // Dummy value if not using a merit-based line search
    }

    // Log iteration status.
    if (_params.verbosity >= VerbosityLevel::Verbose) {
      MOCHI_LOG(
          "\nNewton-Raphson iteration %i with |res|=%.5e%s, obj=%.5e\n"
          "Parameters: absTol=%.5e, relTol=%.5e, solRelTol=%.5e, maxIter=%i, convergenceMode=%s\n",
          status.numIterDone,
          status.resNorm,
          (_params.convergenceMode != NonLinearSolverConvergenceMode::Global) ? " (global)" : "",
          status.merit,
          _params.absTolRes,
          _params.relTolRes,
          _params.solRelTol,
          _params.maxIter,
          SReflect::EnumToString(_params.convergenceMode));
    }

    // Check stop criteria.
    if (IsSolverDone(_stopCriteria, initialSolution, problem, status)) {
      return status;
    }

    // Set the relative tolerance of the linear solver and advance the iteration state for the next
    // iteration. Order matters: SetLinearSolverRelativeTolerance reads prevResNorm and numIterDone.
    SetLinearSolverRelativeTolerance(status, linearResidualNorm);
    status.prevResNorm = status.resNorm;
    status.numIterDone++;

    // Try to take step.
    bool retry = false;
    do {
      // Assemble the dresidual, if needed. Skip it in the first iteration (unless it's a retry)
      // because it was assembled above together with the residual and (possibly) the merit.
      if ((!isFirstIter &&
           (status.itersSinceDResidualAssembly >= _params.dResidualAssemblyPeriod ||
            !status.improvedInLastIter)) ||
          retry) {
        problem.UpdateDResidual(projectPsd, fittedSaturationHessian);
        status.itersSinceDResidualAssembly = 0;
      }

      // Set up the linear operator used by the linear solver. Use IslandOperators if the solver is
      // iterative (direct solvers require the global matrix), not CUDA (solving with
      // IslandOperators is not supported on CUDA) and the preconditioner can be directly computed
      // from the IslandOperators (if the preconditioner requires condensing the global matrix, it's
      // more efficient to use the global matrix also for the matrix-vector products).
      bool const useIslandOperators = details::IsIterativeSolver(_params.lParams.solverType) &&
          !details::IsCudaSolver(_params.lParams.solverType) &&
          ((_params.lParams.preconditionerType == PreconditionerType::None) ||
           (_params.lParams.preconditionerType == PreconditionerType::PerActor));
      PrepareLinearOperator(problem, status, useIslandOperators, projectPsd, linOp);

      // Solve the linear system.
      dxSolve.SetZero();
      auto linearResult = _linearSolver->Solve(linOp, problem.GetResidual(), dxSolve);

      if (linearResult.converged && _params.verbosity >= VerbosityLevel::Verbose) {
        MOCHI_LOG(
            "\n\tLinear solver converged after %d iterations: |r|=%.5e, |r|/|r0|=%.5e\n",
            linearResult.numIterDone,
            linearResult.residualNorm,
            linearResult.relativeResidualNorm);
      } else if (!linearResult.converged) {
        bool const abortedDueToNonPsd = !projectPsd && _params.lParams.abortIfNotSpd &&
            (linearResult.numIterDone <= _params.lParams.maxIter);
        if (abortedDueToNonPsd && _params.verbosity >= VerbosityLevel::Verbose) {
          MOCHI_LOG(
              "\n\tLinear solver aborted after %d iterations due to non-PSD matrix.",
              linearResult.numIterDone);
        } else if (!abortedDueToNonPsd && _params.verbosity >= VerbosityLevel::Warning) {
          MOCHI_LOG_WARNING(
              "\n\tLinear solver didn't converge after %d iterations: |r|=%.5e, |r|/|r0|=%.5e\n",
              linearResult.numIterDone,
              linearResult.residualNorm,
              linearResult.relativeResidualNorm);
        }
      }

      linearResidualNorm = static_cast<T>(linearResult.residualNorm);
      status.totalNumLinearIterDone += linearResult.numIterDone;
      retry = !linearResult.converged;

      // If the Newton-Raphson step is not successful, the iteration will be retried if (a) we did
      // not use a fitted Hessian, (b) we are not doing PSD projection but we could, and/or (c) the
      // DResidual is not up-to-date (itersSinceDResidualAssembly > 0). Otherwise, this is the last
      // try.
      bool const isLastTry = fittedSaturationHessian.AllTrue() &&
          (projectPsd || psdMode == PsdProjectionMode::Never) &&
          (status.itersSinceDResidualAssembly == 0);

      // Try to take the step only if the linear solver converged or if it's the last try.
      if (isLastTry || linearResult.converged) {
        if (MOCHI_ASSERT_VERBOSE_ENABLED && projectPsd) {
          [[maybe_unused]] auto resDot = problem.GetResidual().Dot(dxSolve);
          MOCHI_ASSERT_VERBOSE(resDot >= 0, "NOT PSD: residual.Dot(dxSolve) = %g", resDot);
        }

        // Linear solve is solving H*dx = g. We need to advance along -dx. If there is no
        // improvement in the line search (LS), the logic is as follows:
        // - If this is the last try in this Newton-Raphson iteration: Take the final LS iterate.
        // - If this is NOT the last try: Discard the final LS iterate, restore the previous
        //   Newton-Raphson iterate, and try Newton-Raphson iteration again using other tricks such
        //   as SPD projection.
        dxStep = -dxSolve;
        status.improvedInLastIter = TakeStep(
            problem,
            status,
            /*takeFinalLsStepIfNoImprovement*/ isLastTry && !_params.gradientDescentFallback);
        retry = !status.improvedInLastIter;
      }

      if (retry) { // The Newton-Raphson step was not successful.
        if (isLastTry) {
          // If 'retry' and 'isLastTry' are true, this is the last try and there was no improvement
          // in the line search. Try a gradient descent step if the gradient descent fallback is
          // enabled.
          if (_params.gradientDescentFallback) {
            dxStep = -_workspace.residual;
            status.improvedInLastIter = TakeStep(
                problem,
                status,
                /*takeFinalLsStepIfNoImprovement*/ true);
          }

          if (!status.improvedInLastIter && _params.verbosity >= VerbosityLevel::Verbose) {
            MOCHI_LOG(
                "\n\t\tUnable to take Newton-Raphson step with improvement. This was the last try in this Newton-Raphson iteration. Taking step without improvement (|dx|=%.5e).\n",
                problem.GetIncrement().Norm());
          }

          // Continue to the next Newton-Raphson iteration.
          break;

        } else {
          MOCHI_ASSERT_VERBOSE(
              !fittedSaturationHessian.AllTrue() || (status.itersSinceDResidualAssembly > 0) ||
              (psdMode == PsdProjectionMode::IfFailAlways) ||
              (psdMode == PsdProjectionMode::IfFailRetry));

          fittedSaturationHessian = SaturationHessianParams::All(true);

          if (psdMode == PsdProjectionMode::IfFailAlways) {
            // Change PSD mode to always project.
            psdMode = PsdProjectionMode::Always;
          }

          projectPsd = true;
          if (_params.verbosity >= VerbosityLevel::Verbose) {
            MOCHI_LOG(
                "\n\t\tUnable to take a valid step. DResidual update, globally fitted Hessian, and/or PSD projection possible. Retrying step.\n");
          }
        } // end if last try
      } // end if retry
    } while (retry);

    // Explosion control
    if (_params.explosionControl) {
      problem.UpdateResidual();
      status.resNorm = problem.GetResidual().Norm();
      if ((status.resNorm > _params.relDivTol * status.resNorm0) ||
          status.resNorm > _params.absDivTol || !IsFinite(status.resNorm)) {
        if (_params.verbosity >= VerbosityLevel::Warning) {
          MOCHI_LOG_WARNING(
              "\n\tSolution explosion detected (%f,%f) rel=%f abs=%f.\n",
              status.resNorm,
              status.resNorm0,
              _params.relDivTol,
              _params.absDivTol);
        }
        status.convergence = ConvergenceStatus::Diverged;
        status.stopReasonStr = "Diverged";
        return status;
      }
    }
  }
}

#if MOCHI_USE_EXTERN_TEMPLATE
template class NewtonSolver<real>;
#endif // MOCHI_USE_EXTERN_TEMPLATE

} // namespace mochi
