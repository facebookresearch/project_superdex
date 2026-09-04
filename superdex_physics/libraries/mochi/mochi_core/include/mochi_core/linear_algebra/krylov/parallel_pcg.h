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

#include <mochi_core/linear_algebra/any_matrix.h>
#include <mochi_core/linear_algebra/krylov/block_jacobi_prec.h>
#include <mochi_core/linear_algebra/krylov/stopping_criterion.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_traits.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/solvers/island_operators.h>
#include <mochi_core/solvers/krylov_solver.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_core/utils/time.h>

#include <atomic>
#include <type_traits>
#include <utility>

namespace mochi::krylov {

namespace parallel_pcg {

template <typename MatType>
int GetNumParallelWorkers(MatType const& A) {
  // Empirically chosen values: ~100k FLOPs per worker for dense matrices, ~25k otherwise.
  int const numTargetFlopsPerWorker = IsMatrix<MatType> ? 100000 : 25000;
  int const numTargetWorkers = Min(FlopsPerApply(A) / numTargetFlopsPerWorker, GetNumRows(A));
  return Min(numTargetWorkers, TaskScheduler::StaticGetNumOtherThreads() + 1);
}

} // namespace parallel_pcg

/** @brief Solve a linear system of equations using a parallel preconditioned CG method. All
 * operations are performed in parallel, including matrix-vector products, preconditioner solves,
 * orthogonalization, and convergence checks.
 *
 * @tparam MatType Type of the matrix.
 * @tparam RhsType Type of the right-hand side vector.
 * @tparam SolType Type of the solution vector.
 * @tparam PrecType Type of the preconditioner.
 * @tparam Dot Type of the dot operation object/functor.
 * @tparam StopCriterion Type of the stop criteria checker.
 * @tparam VectorFactory Type of the vector factory.
 *
 * @param[in] A The matrix of the linear system.
 * @param[in] b The right-hand side vector of \f$ A x = b\f$.
 * @param[in,out] x Vector containing the initial guess at input and the solution at output.
 * @param[in] prec The preconditioner.
 * @param[in] maxIter Maximum number of iterations.
 * @param[in,out] statusCheck A functor called every iteration to check the stop criteria. The norm
 * used in the stop criteria is determined by this object.
 * @param[in] abortIfNotSpd Boolean to abort the solve if the matrix is detected not to be symmetric
 * positive definite. Default is false.
 * @param[in] verbosity Verbosity level for logging output.
 * @param[in] usePolakRibiere Boolean to use the Polak-Ribiere formula for beta (if true) or the
 * Fletcher-Reeves formula (if false). Default is true.
 * @param[in] initialGuessHint Indicates whether @p x is known to be zero. The zero hint skips the
 * initial matrix-vector product and requires @p x to be exactly zero.
 * @param[in] dot The dot operator.
 * @param[in] vectorFactory Factory to create vectors of a given type.
 *
 * @return Linear solver status. Contains the convergence status, number of iterations, and achieved
 * absolute and relative residuals.
 *
 * @note The preconditioner must implement a 'ConcurrentSolve' method.
 * @note The input matrix must be a supported matrix or linear operator type. Matrix application
 * functors are NOT supported.
 * @note CUDA matrices are not supported.
 * @note It falls back to regular PCG if the creation of the pool of parallel workers fails.
 * @note It uses left preconditioning.
 * @note The norm used in the stop criteria is specified by the object 'statusCheck'.
 * @note Complex arithmetic is not supported.
 *
 * TODO:
 * - Assess variations of PCG that require fewer synchronization points
 *   (https://www.sciencedirect.com/science/article/abs/pii/S0167819113000719).
 * - Assess using only a subset of the workers to perform BLAS1 operations.
 */
template <
    typename MatType,
    typename RhsType,
    typename SolType,
    typename PrecType,
    typename Dot = UsualDot,
    typename StopCriterion = StatusResidualL2<Dot, real>,
    typename VectorFactory = MatrixFactoryType<RhsType>>
LinearSolverStatus ParallelPCG(
    MatType const& A,
    RhsType const& b,
    SolType& x,
    PrecType const& prec,
    int maxIter,
    StopCriterion& statusCheck,
    bool abortIfNotSpd = false,
    VerbosityLevel verbosity = VerbosityLevel::Warning,
    bool usePolakRibiere = true,
    InitialGuessHint initialGuessHint = InitialGuessHint::Unknown,
    Dot dot = {},
    VectorFactory vectorFactory = {}) {
  MOCHI_PROFILE_SCOPE();
  static_assert(IsLinearOperator<MatType>, "Unsupported matrix type");
  static_assert(!mochi::IsCuda<MatType>, "CUDA matrices not supported with parallel PCG");
  using NonConstScalar = typename MatType::NonConstScalar;
  static_assert(
      std::is_same_v<StopCriterion, StatusResidualL2<Dot, NonConstScalar>> ||
          std::is_same_v<StopCriterion, StatusPreconditionedResidualL2<Dot, NonConstScalar>> ||
          std::is_same_v<StopCriterion, StatusResidualPreconditionerInduced<Dot, NonConstScalar>>,
      "Unsupported stop criterion");
  constexpr bool kNeedPrecResidual =
      std::is_same_v<StopCriterion, StatusPreconditionedResidualL2<Dot, NonConstScalar>> ||
      std::is_same_v<StopCriterion, StatusResidualPreconditionerInduced<Dot, NonConstScalar>>;
  constexpr bool kCheckStatusComputesRTz =
      std::is_same_v<StopCriterion, StatusResidualPreconditionerInduced<Dot, NonConstScalar>>;
  MOCHI_ASSERT_VERBOSE(
      initialGuessHint != InitialGuessHint::Zero || dot(x, x) == 0,
      "InitialGuessHint::Zero requires an exactly zero initial guess.");

  auto* scheduler = TaskScheduler::TryGet();
  auto const numTargetWorkers = parallel_pcg::GetNumParallelWorkers(A);
  std::atomic<bool> success = (scheduler && numTargetWorkers > 1); // At least 2 workers
  LinearSolverStatus solverStatus = {};
  if (success) {
    auto r = vectorFactory.GetCopy(b);
    auto Ap = vectorFactory.GetSameAs(b);
    auto p = vectorFactory.GetSameAs(x);
    auto z = vectorFactory.GetSameAs(x);
    statusCheck.SetScaling(r, prec, z);

    ParallelBarrier barrier(numTargetWorkers);
    ParallelDot<NonConstScalar> parDot(numTargetWorkers);

    // Utility to check if all workers in the pool of parallel workers are ready before a timeout is
    // reached. It returns true if all the workers are ready and false otherwise. The return is the
    // same for all the workers.
    // WARNING: Each worker must own a COPY (not a reference) of the lambda.
    TaskSemaphore ready1(numTargetWorkers), ready2(numTargetWorkers);
    TimePoint const timeoutTime = Timer::Now() +
        /* Empirically chosen wait time */ TimeSpanFromSeconds(50e-6 + numTargetWorkers * 5e-6);
    auto areAllWorkersReady = [/* By VALUE */ ready1,
                               /* By VALUE */ ready2,
                               numTargetWorkers](
                                  TimePoint timeoutTime, int numWorkers, bool isMaster) {
      // The master is responsible for removing missing workers w.r.t. numTargetWorkers.
      MOCHI_ASSERT_VERBOSE(numWorkers <= numTargetWorkers);
      if (isMaster) {
        for (int i = numWorkers; i < numTargetWorkers; ++i) {
          ready1.Done();
          ready2.Done();
        }
      }
      MOCHI_ASSERT_VERBOSE(!ready1.IsDone() && !ready2.IsDone());

      // Indicate worker is ready.
      ready1.Done();

      // Wait for all workers to be ready (or timeout).
      while (!ready1.IsDone() && (Timer::Now() < timeoutTime)) {
      }

      if (!ready1.IsDone()) {
        // Not all workers are ready. Indicate other workers that they must give up.
        ready1.Add(1);
      } else {
        // All workers seem ready. Indicate this worker's intent to go ahead.
        ready2.Done();
      }

      // Wait for final decision. Go ahead only if all workers intend to go ahead.
      while (ready1.IsDone() && !ready2.IsDone()) {
      }
      MOCHI_ASSERT_VERBOSE(ready1.IsDone() == ready2.IsDone()); // Either both or none must be done.

      return ready1.IsDone();
    };

    auto pcgWorkerTask = [&, /* Must be captured by VALUE */ areAllWorkersReady](
                             int workerIdx, int numWorkers) {
      // Notes:
      // - The task is executed by either none or all of the workers. If it's executed, the workers
      //   do NOT yield until the task is completed.
      // - Workers write exclusively to their range of rows. Writing outside their range is illegal.
      // - Workers may read from rows outside their range, e.g. in matrix-vector products and
      //   preconditioner solves. Barriers are used in those cases to avoid race conditions.
      // - The master worker is defined as the worker with workerIdx = 0.
      // - The master worker is responsible for updating 'solverStatus'. It's illegal for other
      //   workers to modify it.
      MOCHI_PROFILE_SCOPE_N("PcgWorkerTask");
      bool const isMaster = (workerIdx == 0);
      if (!areAllWorkersReady(timeoutTime, numWorkers, isMaster)) {
        success = false;
        return;
      }

      auto const workerRowRanges = GetRowRangesPerWorker(A, numWorkers);
      auto const rowBegin = workerRowRanges[workerIdx];
      auto const rowEnd = workerRowRanges[workerIdx + 1];
      auto const numRows = rowEnd - rowBegin;
      MOCHI_ASSERT_VERBOSE(
          numRows >= 0 && rowBegin >= 0 && rowEnd <= A.Rows(), "Invalid row ranges.");

      auto workerStatusCheck = statusCheck; // Worker copy to prevent race conditions.
      auto workerBarrier = barrier; // Worker copy. The copy is mandatory for 'ParallelBarrier'.
      auto workerParDot = parDot; // Worker copy. The copy is mandatory for 'ParallelDot'.
      workerBarrier.ReduceNumWorkers(numWorkers, isMaster);
      workerParDot.ReduceNumWorkers(numWorkers, isMaster);

      ParallelWorkerInfo workerInfo{workerIdx, numWorkers, rowBegin, rowEnd, workerBarrier};

      auto xWorker = x.MiddleRows(rowBegin, numRows);
      auto rWorker = r.MiddleRows(rowBegin, numRows);
      auto zWorker = z.MiddleRows(rowBegin, numRows);
      auto pWorker = p.MiddleRows(rowBegin, numRows);
      auto ApWorker = Ap.MiddleRows(rowBegin, numRows);

      IterationStatus iterStatus = {};
      NonConstScalar beta = 0;
      int iter = 0;

      auto computeBetaAndPrecResidual = [&]() {
        if (usePolakRibiere && (iter > 1)) {
          beta = workerParDot.Dot(dot, r, z, rowBegin, rowEnd, workerIdx); // r_i^T z_{i-1}
        } else {
          beta = 0;
          workerBarrier.Wait(); // TODO: Not needed for some preconditioners.
        }
        prec.ConcurrentSolve(r, z, workerInfo); // z_{i} = Prec^{-1} r_{i}
        // Post-solve barrier not needed. The next 'Dot' serves as implicit barrier and prevents
        // 'r' from being modified before the solve is complete.
      };

      if (initialGuessHint != InitialGuessHint::Zero) {
        // Pre- and post-ApplyToRange barriers not needed: 'x' is up-to-date and the next 'Dot'
        // prevents 'x' from being modified before the product is complete.
        ApplyToRange(A, x, Ap, rowBegin, rowEnd);
        rWorker -= ApWorker;
      }

      if constexpr (kNeedPrecResidual) {
        // Even with x_0 = 0, do not reuse z from SetScaling(): it was computed with Solve(),
        // whereas ParallelPCG uses ConcurrentSolve(). Reuse could therefore give iteration 0 a
        // different effective preconditioner from subsequent iterations.
        computeBetaAndPrecResidual();
        iterStatus = workerStatusCheck.ParallelCheckStatus(
            iter, r, z, {}, {}, rowBegin, rowEnd, workerIdx, workerParDot);
      } else {
        iterStatus = workerStatusCheck.ParallelCheckStatus(
            iter, r, {}, {}, {}, rowBegin, rowEnd, workerIdx, workerParDot);
        if (iterStatus == IterationStatus::Active) {
          computeBetaAndPrecResidual();
        }
      }
      if (iterStatus != IterationStatus::Active) {
        if (isMaster) {
          solverStatus = {
              .numIterDone = iter,
              .residualNorm = workerStatusCheck.GetLatestResidualNorm(),
              .relativeResidualNorm = workerStatusCheck.GetLatestRelativeResidualNorm(),
              .convergence = IsConverged(iterStatus) ? LinearSolverConvergenceStatus::Converged
                                                     : LinearSolverConvergenceStatus::Diverged};
        }
        return;
      }

      pWorker = zWorker;
      NonConstScalar rTz{}; // r_0^T z_0
      if constexpr (kCheckStatusComputesRTz) {
        rTz = workerStatusCheck.GetLatestResidualNormSqr();
      } else {
        rTz = workerParDot.Dot(dot, r, z, rowBegin, rowEnd, workerIdx);
      }
      for (iter = 1; iter <= maxIter; ++iter) {
        // Wait for 'p' components from other workers to be up-to-date before starting the
        // matrix-vector product. An explicit post-product barrier is not needed: The next 'Dot'
        // serves as implicit barrier and prevents 'p' from being modified before the product is
        // complete.
        workerBarrier.Wait();
        ApplyToRange(A, p, Ap, rowBegin, rowEnd);

        auto const pTAp = workerParDot.Dot(dot, p, Ap, rowBegin, rowEnd, workerIdx);
        if (pTAp <= 0)
          MOCHI_UNLIKELY {
            if (!abortIfNotSpd) {
              if (isMaster && verbosity >= VerbosityLevel::Warning) {
                MOCHI_LOG_WARNING(
                    "Matrix does not seem to be SPD at iteration %d. A-dot product: %e.",
                    iter,
                    static_cast<double>(pTAp));
              }
            } else {
              if (isMaster) {
                solverStatus = {
                    .numIterDone = iter,
                    .residualNorm = static_cast<double>(workerStatusCheck.GetLatestResidualNorm()),
                    .relativeResidualNorm =
                        static_cast<double>(workerStatusCheck.GetLatestRelativeResidualNorm()),
                    .convergence = LinearSolverConvergenceStatus::Diverged};
              }
              return;
            }
          }

        auto const alpha = rTz / pTAp;
        xWorker += alpha * pWorker; // x_i = x_{i-1} + alpha_i p_i
        rWorker -= alpha * ApWorker; // r_i = r_{i-1} - alpha_i A * p_i

        if constexpr (kNeedPrecResidual) {
          computeBetaAndPrecResidual();
          iterStatus = workerStatusCheck.ParallelCheckStatus(
              iter, r, z, {}, {}, rowBegin, rowEnd, workerIdx, workerParDot);
        } else {
          iterStatus = workerStatusCheck.ParallelCheckStatus(
              iter, r, {}, {}, {}, rowBegin, rowEnd, workerIdx, workerParDot);
          if (iterStatus == IterationStatus::Active) {
            computeBetaAndPrecResidual();
          }
        }

        if (iterStatus != IterationStatus::Active) {
          if (isMaster) {
            solverStatus = {
                .numIterDone = iter,
                .residualNorm = static_cast<double>(workerStatusCheck.GetLatestResidualNorm()),
                .relativeResidualNorm =
                    static_cast<double>(workerStatusCheck.GetLatestRelativeResidualNorm()),
                .convergence = IsConverged(iterStatus) ? LinearSolverConvergenceStatus::Converged
                                                       : LinearSolverConvergenceStatus::Diverged};
          }
          return;
        }

        auto const rTzPrev = rTz;
        if constexpr (kCheckStatusComputesRTz) {
          rTz = workerStatusCheck.GetLatestResidualNormSqr();
        } else {
          rTz = workerParDot.Dot(dot, r, z, rowBegin, rowEnd, workerIdx);
        }

        if (rTz == 0)
          MOCHI_UNLIKELY {
            if (isMaster) {
              if (verbosity >= VerbosityLevel::Error) {
                // The residual is not zero at this point. The preconditioner may be singular.
                MOCHI_LOG_ERROR("Zero Preconditioner-dot product at iteration %d.", iter);
              }
              solverStatus = {
                  .numIterDone = iter,
                  .residualNorm = static_cast<double>(workerStatusCheck.GetLatestResidualNorm()),
                  .relativeResidualNorm =
                      static_cast<double>(workerStatusCheck.GetLatestRelativeResidualNorm()),
                  .convergence = LinearSolverConvergenceStatus::Diverged};
            }
            return;
          }

        beta = (rTz - beta) / rTzPrev;
        pWorker = zWorker + beta * pWorker;
      } // for (iter = 1; iter <= maxIter; ++iter)

      if (isMaster) {
        solverStatus = {
            .numIterDone = maxIter,
            .residualNorm = static_cast<double>(workerStatusCheck.GetLatestResidualNorm()),
            .relativeResidualNorm =
                static_cast<double>(workerStatusCheck.GetLatestRelativeResidualNorm()),
            .convergence = LinearSolverConvergenceStatus::Stopped};
      }
    };

    TaskSemaphore sem;
    TaskScheduler::BatchTaskFn task = [sem, &pcgWorkerTask](int workerIdx, int numWorkers) {
      // Disable nested parallelization in the worker threads during the execution of the
      // concurrent algorithm.
      TaskScheduler::PushLocalSingleThreadedMode();
      pcgWorkerTask(workerIdx, numWorkers);
      TaskScheduler::PopLocalSingleThreadedMode();
      sem.Done();
    };

    int const numMinWorkers = (numTargetWorkers + 1) / 2; // At least half the target (rounded up).
    if (scheduler->BatchEnqueueOnAvailableWorkers(
            sem,
            std::move(task),
            numMinWorkers,
            numTargetWorkers,
            /*includeSelf*/ true) >= numMinWorkers) {
      // Wait for all workers to finish to ensure sufficient lifespan of 'pcgWorkerTask' (among
      // others) and thread-safe read of 'success' below.
      sem.Wait();
    } else {
      success = false;
    }
  }

  if (!success) {
    // Fallback to regular PCG.
    solverStatus =
        PCG(A,
            b,
            x,
            prec,
            maxIter,
            statusCheck,
            abortIfNotSpd,
            verbosity,
            usePolakRibiere,
            initialGuessHint,
            dot,
            vectorFactory);
  }

  return solverStatus;
}

} // namespace mochi::krylov
