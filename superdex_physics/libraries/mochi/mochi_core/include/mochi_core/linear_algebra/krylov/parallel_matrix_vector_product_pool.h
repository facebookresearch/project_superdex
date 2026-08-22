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
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/solvers/island_operators.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_core/utils/time.h>

#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mochi::krylov {

/* Class to perform a series of matrix-vector products in parallel in the context of an iterative
 * linear solver. It creates a pool of parallel workers at construction, and then ensures each
 * worker performs the same subset of the product in every iteration. This reduces parallelization
 * overhead and increases per-core cache efficiency. Neither the master thread nor the worker
 * threads yield to other tasks until the pool is shut down, except perhaps as part of the shutdown
 * process.
 *
 * WARNINGS:
 * - It's the responsibility of the caller to release the workers after using the pool. This is
 *   achieved by destructing the pool instance or by calling 'ReleaseWorkers()' method.
 * - If the pool is used for multiple solves, it's the responsibility of the caller to reset it
 *   before the 2nd, 3rd, ... solve via 'Reset()' method. If there is a non-negligible amount of
 *   work to be done between subsequent solves, it's recommended to release the workers after each
 *   solve so that they're available for other purposes.
 *
 * Notes:
 * - The pool requires that all parallel tasks are allocated to different threads, none of them
 *   being the master thread. The task scheduler doesn't provide this guarantee. If the task
 *   scheduler fails to allocate the tasks to different threads, the matrix-vector products fall
 *   back to their regular implementation.
 * - The pool also falls back to the regular matrix-vector product under any of the following
 *   conditions:
 *   a. There are not enough available workers.
 *   b. The matrix is sufficiently small so that there is no benefit in parallelization.
 *
 * Example usage:
 * ```
 * BlockSparseMatrix<real, 3> A = ...; // Create matrix
 * ColumnVector<real> b1(n), b2(n), x(n); // Create right-hand side(s) and solution vector
 * auto opA = ParallelMatrixVectorProductPool(A); // Create parallel pool
 * status = PCG(opA, b1, x, prec, maxIter, statusCheck); // Perform solve
 * opA.ReleaseWorkers(); // Release the workers so that they can be used for other purposes
 * DoOtherWork(); // Arbitrary work, possibly unrelated to the solver
 * opA.Reset(); // Reset the pool before the new solve
 * status = PCG(opA, b2, x, prec, maxIter, statusCheck); // Solve with a new right-hand side
 * opA.ReleaseWorkers(); // Release the workers so that they can be used for other purposes
 * ```
 *
 * TODO:
 * [P2] A range of rows may be sub-optimal to characterize the division of work for some operators.
 */
template <typename MatrixType>
class ParallelMatrixVectorProductPool {
  static_assert(IsLinearOperator<MatrixType>, "Unsupported matrix type");
  static_assert(
      !mochi::IsCuda<MatrixType>,
      "ParallelMatrixVectorProductPool not supported for CUDA matrices");

  // Delete operator=. The copy could have inconsistent state.
  MOCHI_DECLARE_NO_ASSIGN(ParallelMatrixVectorProductPool)

 public:
  using Scalar = typename MatrixType::Scalar;
  using NonConstScalar = typename MatrixType::NonConstScalar;

  static constexpr int kShutdownCommand = -1;
  static_assert(kShutdownCommand < 0, "Shutdown command must be a negative number");

  // Time the master thread waits for the workers to initialize. If the max time is reached, the
  // regular matrix-vector product is used. Small matrices directly fall back to the regular
  // matrix-vector product without attempting to create the pool, i.e. there is no waiting overhead.
  static constexpr double kInitializePoolWaitTimeSeconds = 100 * 1e-6; // Empirically chosen value.

  struct ProblemData {
    explicit ProblemData(MatrixType const& A) : A(A) {}

    // Delete operator= due to matrix views.
    MOCHI_DECLARE_NO_ASSIGN(ProblemData)

    // Matrix of the system of equations.
    MatrixType const& A;
    // View of the input vector for the matrix-vector product. The view is reset by the master
    // thread at the beginning of each iteration.
    ColumnVectorView<Scalar const> xView;
    // View of the output vector for the matrix-vector product. The view is reset by the master
    // thread at the beginning of each iteration.
    ColumnVectorView<NonConstScalar> AxView;
  };

  struct PoolData {
    explicit PoolData(int numWorkers) : numWorkers(numWorkers) {
      MOCHI_ASSERT(numWorkers >= 0, "Invalid number of workers.");
      workerThreadIds.resize(numWorkers);
    }

    // Master thread ID.
    std::thread::id const masterThreadId = std::this_thread::get_id();
    // Number of parallel workers. It does NOT include the master thread. It MUST only be read and
    // written by the master thread.
    int numWorkers = 0;
    // Vector with the thread ID of each worker. It does NOT include the master thread. The size may
    // be larger than the number of workers.
    std::vector<std::thread::id> workerThreadIds = {};
    // Command from the master thread to the workers.
    std::atomic<int> command = 0;
    // Number of workers whose matrix-vector product contribution is pending. It does NOT include
    // the master thread.
    std::atomic<int> numPendingWorkers = 0;
    // Number of workers that have successfully registered in the pool. It does NOT include the
    // master thread.
    std::atomic<int> numRegisteredWorkers = 0;
    // Task semaphore for pool shutdown.
    TaskSemaphore shutdownSem;
  };

  explicit ParallelMatrixVectorProductPool(MatrixType const& A, bool masterPerformsProduct = true)
      : _masterPerformsProduct(masterPerformsProduct), _problemData(A) {}

  ~ParallelMatrixVectorProductPool() {
    ReleaseWorkers();
  }

 public:
  bool IsPoolEnabled() const {
    return _poolData != nullptr;
  }

  void ReleaseWorkers() const {
    MOCHI_PROFILE_SCOPE();
    if (_poolData) {
      MOCHI_ASSERT(
          std::this_thread::get_id() == _poolData->masterThreadId,
          "Workers must be released from the master thread.");

      // Command shutdown.
      if (_usePool) {
        CommandShutdown();
      } else {
        MOCHI_ASSERT(_poolData->command == kShutdownCommand);
      }

      // Wait for workers to shut down.
      _poolData->shutdownSem.Wait();
      _poolData.reset();
    }
  }

  void Reset() const {
    ReleaseWorkers();
    _numIter = 0;
  }

  template <typename VectorIn, typename VectorOut>
  void operator()(VectorIn const& x, VectorOut& Ax) const {
    MOCHI_PROFILE_SCOPE();
    if (_numIter == 0) {
      InitializePool();
    }

    if (_usePool) {
      TaskScheduler::PushLocalSingleThreadedMode();
      MOCHI_DEFER(TaskScheduler::PopLocalSingleThreadedMode());
      MOCHI_ASSERT(
          std::this_thread::get_id() == _poolData->masterThreadId,
          "operator() must be called from the master thread.");
      _problemData.xView.Reset(x);
      _problemData.AxView.Reset(Ax);
      CommandProduct();
      if (_masterPerformsProduct) {
        MOCHI_ASSERT_VERBOSE(
            _masterRowRange.first >= 0 && _masterRowRange.second >= _masterRowRange.first);
        ApplyToRange(_problemData.A, x, Ax, _masterRowRange.first, _masterRowRange.second);
      }
      WaitForProductComplete();
    } else {
      Apply(_problemData.A, x, Ax);
    }
    _numIter++;
  }

  /* Compute the matrix-vector product while the master thread performs other asynchronous work.
   *
   * @param[in] x Input vector.
   * @param[out] Ax Output vector, that is, Ax = A * x.
   * @param[in] beforeProduct Work to perform by the master thread before the matrix-vector product
   * starts.
   * @param[in] duringProduct Work to perform by the master thread simultaneously with the
   * matrix-vector product.
   * @param[in] afterProduct Work to perform by the master thread after the matrix-vector product is
   * complete.
   *
   * @note Only implemented if the pool is initialized with masterPerformsProduct = false.
   * TODO: Extend support to masterPerformsProduct = true.
   */
  template <typename VectorIn, typename VectorOut>
  void AsyncApply(
      VectorIn const& x,
      VectorOut& Ax,
      std::function<void()> const& beforeProduct,
      std::function<void()> const& duringProduct,
      std::function<void()> const& afterProduct) const {
    MOCHI_PROFILE_SCOPE();
    MOCHI_ASSERT(
        !_masterPerformsProduct,
        "AsyncApply requires the pool to be set up with the master thread not participating in the product.");
    if (_numIter == 0) {
      InitializePool();
    }

    bool const forceLocalSingleThreadedMode = _usePool;
    if (forceLocalSingleThreadedMode) {
      TaskScheduler::PushLocalSingleThreadedMode();
    }

    beforeProduct();
    if (_usePool) {
      MOCHI_ASSERT(
          std::this_thread::get_id() == _poolData->masterThreadId,
          "AsyncApply must be called from the master thread.");
      _problemData.xView.Reset(x);
      _problemData.AxView.Reset(Ax);
      CommandProduct();
      duringProduct();
      WaitForProductComplete();
    } else {
      TaskSemaphore sem;
      Schedule(sem, "Apply", [&]() { Apply(_problemData.A, x, Ax); });
      duringProduct();
      sem.Wait();
    }
    afterProduct();
    _numIter++;

    if (forceLocalSingleThreadedMode) {
      TaskScheduler::PopLocalSingleThreadedMode();
    }
  }

 protected:
  int GetNumTargetWorkers() const {
    // Empirically chosen values: ~100k FLOPs per worker for dense matrices, ~25k FLOPS otherwise.
    int const numTargetFlopsPerWorker = IsMatrix<MatrixType> ? 100000 : 25000;
    int const numTargetWorkers =
        Min(FlopsPerApply(_problemData.A) / numTargetFlopsPerWorker, GetNumRows(_problemData.A));
    return Min(numTargetWorkers, TaskScheduler::StaticGetNumOtherThreads());
  }

  /*
   * Methods for the master thread.
   */
 protected:
  void CommandShutdown() const {
    if (_poolData) {
      MOCHI_ASSERT(
          std::this_thread::get_id() == _poolData->masterThreadId,
          "Pool must be shut down from the master thread.");
      MOCHI_ASSERT(_poolData->command != kShutdownCommand, "Pool has already been shut down.");
      _poolData->shutdownSem.Add(_poolData->numWorkers);
      _poolData->command = kShutdownCommand;
    }
    _usePool = false;
  }

  void CommandProduct() const {
    MOCHI_ASSERT_VERBOSE(std::this_thread::get_id() == _poolData->masterThreadId);
    MOCHI_ASSERT_VERBOSE(_poolData->numRegisteredWorkers == _poolData->numWorkers);
    MOCHI_ASSERT_VERBOSE(_poolData->numPendingWorkers == 0);
    _poolData->numPendingWorkers = _poolData->numWorkers;
    ++_poolData->command;
  }

  void InitializePool() const {
    MOCHI_PROFILE_SCOPE();
    MOCHI_ASSERT(!_poolData, "Pool must be shut down before it's initialized again.");
    int const numTargetWorkers = GetNumTargetWorkers();

    // Use the parallel pool only if there are at least 2 workers. Otherwise, fall back to the
    // regular matrix-vector product implementation.
    _usePool = (numTargetWorkers > 1);
    if (_usePool) {
      auto* scheduler = TaskScheduler::TryGet();
      MOCHI_ASSERT(scheduler); // Must succeed if numTargetWorkers > 1
      int const targetNonMasterWorkers =
          _masterPerformsProduct ? (numTargetWorkers - 1) : numTargetWorkers;
      _poolData = std::make_shared<PoolData>(targetNonMasterWorkers);

      TaskScheduler::BatchTaskFn task = [&, poolData = _poolData](int workerIdx, int numWorkers) {
        // Disable nested parallelization in the worker threads during the execution of the
        // concurrent algorithm.
        TaskScheduler::PushLocalSingleThreadedMode();
        MatrixVectorProductTask(
            _problemData,
            *poolData,
            workerIdx,
            numWorkers + static_cast<int>(_masterPerformsProduct));
        TaskScheduler::PopLocalSingleThreadedMode();
      };

      TaskSemaphore dummySem;
      int const minNonMasterWorkers =
          (targetNonMasterWorkers + 1) / 2; // At least half the target (rounded up).
      int const nonMasterWorkers = scheduler->BatchEnqueueOnAvailableWorkers(
          dummySem, // Dummy semaphore. Shutdown is handled by
                    // ParallelMatrixVectorProductPool's logic.
          std::move(task),
          minNonMasterWorkers,
          targetNonMasterWorkers,
          /*includeSelf*/ false);
      bool success = (nonMasterWorkers >= minNonMasterWorkers);
      if (success) {
        _poolData->numWorkers = nonMasterWorkers;
        if (_masterPerformsProduct) {
          auto const workerRowRanges = GetRowRangesPerWorker(_problemData.A, nonMasterWorkers + 1);
          _masterRowRange = {
              workerRowRanges[nonMasterWorkers], workerRowRanges[nonMasterWorkers + 1]};
        }
        success = WaitForWorkersStartup(TimeSpanFromSeconds(kInitializePoolWaitTimeSeconds));
      } else {
        _poolData.reset(); // Batch enqueue failed. There is no pool of workers.
      }
      if (!success) {
        CommandShutdown();
      }
    }
  }

  [[nodiscard]] bool WaitForWorkersStartup(TimeSpan waitTime) const {
    MOCHI_PROFILE_SCOPE();
    MOCHI_ASSERT_VERBOSE(std::this_thread::get_id() == _poolData->masterThreadId);
    Timer timer = {};
    while ((_poolData->numRegisteredWorkers < _poolData->numWorkers) &&
           (ToMicroseconds(timer.GetElapsed()) <= ToMicroseconds(waitTime))) {
    }

    // Check that all workers are ready.
    if (_poolData->numRegisteredWorkers < _poolData->numWorkers) {
      return false;
    }
    MOCHI_ASSERT_VERBOSE(_poolData->numRegisteredWorkers == _poolData->numWorkers);

    // Check that all tasks are allocated to different threads and none of them are the master
    // thread.
    std::unordered_set<std::thread::id> uniqueWorkerThreadIds(
        _poolData->workerThreadIds.begin(),
        _poolData->workerThreadIds.begin() + _poolData->numWorkers);
    return (uniqueWorkerThreadIds.size() == _poolData->numWorkers) &&
        (uniqueWorkerThreadIds.count(_poolData->masterThreadId) == 0);
  }

  void WaitForProductComplete() const {
    MOCHI_PROFILE_SCOPE();
    MOCHI_ASSERT_VERBOSE(std::this_thread::get_id() == _poolData->masterThreadId);
    while (_poolData->numPendingWorkers > 0) {
    }
  }

  /*
   * Methods for the worker threads.
   */
 protected:
  static void RegisterWorker(PoolData& poolData, int taskId) {
    poolData.workerThreadIds[taskId] = std::this_thread::get_id();
    ++poolData.numRegisteredWorkers;
  }

  static void ShutdownWorker(PoolData& poolData) {
    poolData.shutdownSem.Done();
  }

  static void WorkerProductComplete(PoolData& poolData) {
    MOCHI_ASSERT_VERBOSE(std::this_thread::get_id() != poolData.masterThreadId);
    MOCHI_ASSERT_VERBOSE(poolData.numPendingWorkers > 0);
    --poolData.numPendingWorkers;
  }

  [[nodiscard]] static int WaitForCommand(PoolData const& poolData, int targetIter) {
    MOCHI_PROFILE_SCOPE();
    // Busy wait to avoid yielding between iterations.
    while ((poolData.command != targetIter) && (poolData.command != kShutdownCommand)) {
    }
    return static_cast<int>(poolData.command);
  }

  static void
  MatrixVectorProductTask(ProblemData& problemData, PoolData& poolData, int taskId, int numTasks) {
    MOCHI_PROFILE_SCOPE();
    RegisterWorker(poolData, taskId);
    auto const taskRowRanges = GetRowRangesPerWorker(problemData.A, numTasks);
    auto const rowBegin = taskRowRanges[taskId];
    auto const rowEnd = taskRowRanges[taskId + 1];
    MOCHI_ASSERT_VERBOSE(rowBegin >= 0 && rowEnd >= rowBegin, "Invalid row range.");

    int numItersDone = 0;
    while (true) {
      int const command = WaitForCommand(poolData, numItersDone + 1);
      if (command == numItersDone + 1) {
        ApplyToRange(problemData.A, problemData.xView, problemData.AxView, rowBegin, rowEnd);
        WorkerProductComplete(poolData);
        ++numItersDone;
      } else {
        MOCHI_ASSERT(command == kShutdownCommand, "Unexpected command.");
        ShutdownWorker(poolData);
        return;
      }
    }
  }

 protected: // Iteration variables are mutable so that operator() is const (required by the solver)
  // Whether the master thread performs part of the product.
  bool const _masterPerformsProduct = true;
  // Range of rows for the part of the product performed by the master. Only used if
  // _masterPerformsProduct = true.
  mutable std::pair<int, int> _masterRowRange = {-1, -1};
  // Whether to use the pool. If false, the regular matrix-vector product implementation is used.
  mutable bool _usePool = false;
  // Iteration number.
  mutable int _numIter = 0;
  // Problem data.
  mutable ProblemData _problemData;
  // Pointer to the pool data. Ownership is shared by the master thread and the tasks to ensure safe
  // access to the task semaphore during shutdown.
  mutable std::shared_ptr<PoolData> _poolData = nullptr;
};

} // namespace mochi::krylov
