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

#include <mochi_core/linear_algebra/krylov/preconditioner.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/solvers/actor_preconditioner.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace mochi {

template <typename T>
struct IslandOperators;

/** @brief Applies one actor preconditioner to its owned global row interval.
 *
 * @note The referenced preconditioner must outlive this entry. Concurrent row ranges are
 * actor-local.
 */
template <typename T>
struct ActorPreconditionerEntry {
  int offset;
  int size;
  std::reference_wrapper<ActorPreconditioner<T>> preconditioner;

  void Solve(ColumnVectorView<T const> x, ColumnVectorView<T> y) const {
    preconditioner.get().Solve(x.MiddleRows(offset, size), y.MiddleRows(offset, size));
  }

  void ConcurrentSolve(
      ColumnVectorView<T const> x,
      ColumnVectorView<T> y,
      ParallelWorkerInfo const& data) const {
    MOCHI_ASSERT_VERBOSE(
        data.rBegin >= 0 && data.rBegin <= data.rEnd && data.rEnd <= size,
        "Invalid actor row range.");
    preconditioner.get().ConcurrentSolve(
        x.MiddleRows(offset, size), y.MiddleRows(offset, size), data);
  }
};

/** @brief Applies one actor preconditioner per actor.
 *
 * @warning An instance must not participate in multiple linear solves concurrently.
 */
template <typename T>
struct PerActorPrec final : Preconditioner<T> {
 private:
  static_assert(
      static_cast<int>(ActorPreconditionerParallelMode::Count) == 3,
      "Please update PerActorPrec if ActorPreconditionerParallelMode changes.");

  struct ActorInfo {
    int actorIndex = 0;
    ActorPreconditionerParallelMode mode = ActorPreconditionerParallelMode::Count;
    int rowBlockSize = 0;
    int numBlockRows = 0;
    int maxConcurrentWorkers = 0;
    ActorPreconditionerCost cost;
    // Estimated duration on one worker.
    double serialCost = 0.0;
  };

  struct ConcurrentSolveTask {
    int actorIndex = 0;
    ActorPreconditionerParallelMode mode = ActorPreconditionerParallelMode::Count;
    // Half-open row range relative to the actor.
    int actorRowBegin = 0;
    int actorRowEnd = 0;
    // Team fields are used only by SynchronizedTeam tasks.
    int teamWorkerId = 0;
    int teamSize = 0;
    std::optional<ParallelBarrier> teamBarrier = std::nullopt;
  };

  struct ConcurrentSolvePlan {
    DynamicArray<DynamicArray<ConcurrentSolveTask>> workerTasks;
    bool requiresFinalBarrier = false;
  };

  struct RefillCandidate {
    double duration = 0.0;
    int infoIndex = 0;
  };

  struct RefillCandidateLess {
    bool operator()(RefillCandidate const& lhs, RefillCandidate const& rhs) const {
      if (lhs.duration != rhs.duration) {
        return lhs.duration < rhs.duration;
      }
      return lhs.infoIndex > rhs.infoIndex;
    }
  };

  static constexpr int kInactiveHeapPosition = -1;

  // Min-heap of worker indices keyed by predicted load. Construction zeroes every load and inserts
  // every worker. A worker's row range is the rows it owns in the matrix-vector product.
  class WorkerLoadHeap {
   public:
    WorkerLoadHeap(
        DynamicArray<double>& loads,
        DynamicArray<int>& heapWorkerIds,
        DynamicArray<int>& heapPositions,
        Span<int const> workerRowRanges)
        : _loads(loads),
          _heapWorkerIds(heapWorkerIds),
          _heapPositions(heapPositions),
          _workerRowRanges(workerRowRanges) {
      MOCHI_ASSERT_VERBOSE(!loads.empty(), "Worker heap must not be empty.");
      MOCHI_ASSERT_VERBOSE(
          isize(heapPositions) == isize(loads) && isize(workerRowRanges) == isize(loads) + 1);
      std::fill(_loads.begin(), _loads.end(), 0.0);
      // Equal loads order workers by index, so index order is a valid heap.
      _heapWorkerIds.resize_noinit(_loads.size());
      std::iota(_heapWorkerIds.begin(), _heapWorkerIds.end(), 0);
      std::iota(_heapPositions.begin(), _heapPositions.end(), 0);
    }

    [[nodiscard]] double Load(int workerId) const {
      return _loads[workerId];
    }

    [[nodiscard]] double MaxLoad() const {
      MOCHI_ASSERT_VERBOSE(!_heapWorkerIds.empty(), "Worker heap must not be empty.");
      return _maximumLoad;
    }

    // Returns NearestActiveWorker's choice if its load equals the minimum, and otherwise the
    // lowest-index worker with the minimum load. Nearby workers are more likely to own the rows
    // they write, which can avoid the final barrier.
    [[nodiscard]] int FindLeast(int preferredRowBegin, int preferredRowEnd) {
      MOCHI_ASSERT_VERBOSE(!_heapWorkerIds.empty(), "Worker heap must not be empty.");
      double const minimumLoad = _loads[_heapWorkerIds.front()];
      int const localWorker =
          NearestActiveWorker(static_cast<int64_t>(preferredRowBegin) + preferredRowEnd);
      return localWorker >= 0 && _loads[localWorker] == minimumLoad ? localWorker
                                                                    : _heapWorkerIds.front();
    }

    [[nodiscard]] int PopLeast(int preferredRowBegin, int preferredRowEnd) {
      int const workerId = FindLeast(preferredRowBegin, preferredRowEnd);
      RemoveAt(_heapPositions[workerId]);
      return workerId;
    }

    void Push(int workerId, double load) {
      MOCHI_ASSERT_VERBOSE(
          _heapPositions[workerId] == kInactiveHeapPosition, "Worker is already in the heap.");
      MOCHI_ASSERT_VERBOSE(load >= _loads[workerId], "Worker load must not decrease.");
      _loads[workerId] = load;
      _maximumLoad = Max(_maximumLoad, load);
      _heapPositions[workerId] = Size();
      _heapWorkerIds.push_back(workerId);
      SiftUp(Size() - 1);
      _localityValid = false;
    }

    void SetLoad(int workerId, double load) {
      int const position = _heapPositions[workerId];
      MOCHI_ASSERT_VERBOSE(position >= 0, "Worker is not in the heap.");
      MOCHI_ASSERT_VERBOSE(load >= _loads[workerId], "Worker load must not decrease.");
      _loads[workerId] = load;
      _maximumLoad = Max(_maximumLoad, load);
      SiftDown(position);
      _localityValid = false;
    }

   private:
    [[nodiscard]] int Size() const {
      return isize(_heapWorkerIds);
    }

    // Twice the midpoint of the worker's row range, which keeps midpoint comparisons in integers.
    [[nodiscard]] int64_t RowMidpointTwice(int workerId) const {
      return static_cast<int64_t>(_workerRowRanges[workerId]) + _workerRowRanges[workerId + 1];
    }

    [[nodiscard]] static int64_t Distance(int64_t lhs, int64_t rhs) {
      return Abs(lhs - rhs);
    }

    [[nodiscard]] bool IsBefore(int lhsWorkerId, int rhsWorkerId) const {
      double const lhsLoad = _loads[lhsWorkerId];
      double const rhsLoad = _loads[rhsWorkerId];
      return lhsLoad != rhsLoad ? lhsLoad < rhsLoad : lhsWorkerId < rhsWorkerId;
    }

    void SwapHeapEntries(int lhs, int rhs) {
      std::swap(_heapWorkerIds[lhs], _heapWorkerIds[rhs]);
      _heapPositions[_heapWorkerIds[lhs]] = lhs;
      _heapPositions[_heapWorkerIds[rhs]] = rhs;
    }

    void SiftUp(int position) {
      while (position > 0) {
        int const parent = (position - 1) / 2;
        if (!IsBefore(_heapWorkerIds[position], _heapWorkerIds[parent])) {
          break;
        }
        SwapHeapEntries(position, parent);
        position = parent;
      }
    }

    void SiftDown(int position) {
      while (2 * position + 1 < Size()) {
        int bestChild = 2 * position + 1;
        int const rightChild = bestChild + 1;
        if (rightChild < Size() &&
            IsBefore(_heapWorkerIds[rightChild], _heapWorkerIds[bestChild])) {
          bestChild = rightChild;
        }
        if (!IsBefore(_heapWorkerIds[bestChild], _heapWorkerIds[position])) {
          break;
        }
        SwapHeapEntries(position, bestChild);
        position = bestChild;
      }
    }

    void RemoveAt(int position) {
      MOCHI_ASSERT_VERBOSE(position >= 0 && position < Size(), "Worker is not in the heap.");
      int const removedWorker = _heapWorkerIds[position];
      int const lastWorker = _heapWorkerIds.back();
      _heapWorkerIds.pop_back();
      _heapPositions[removedWorker] = kInactiveHeapPosition;
      if (position == Size()) {
        return;
      }
      _heapWorkerIds[position] = lastWorker;
      _heapPositions[lastWorker] = position;
      if (position > 0 && IsBefore(_heapWorkerIds[position], _heapWorkerIds[(position - 1) / 2])) {
        SiftUp(position);
      } else {
        SiftDown(position);
      }
    }

    // Starts a new NearestActiveWorker scan at the worker whose row range contains the preferred
    // midpoint.
    void ResetLocality(int64_t preferredMidpointTwice) {
      auto const upper = std::upper_bound(
          _workerRowRanges.begin(),
          _workerRowRanges.end(),
          preferredMidpointTwice,
          [](int64_t midpointTwice, int boundary) {
            return midpointTwice < 2 * static_cast<int64_t>(boundary);
          });
      int const owner = static_cast<int>(upper - _workerRowRanges.begin()) - 1;
      MOCHI_ASSERT_VERBOSE(
          owner >= 0 && owner < isize(_loads), "Preferred rows must lie within the worker rows.");
      _localityLeft = owner;
      _localityRight = owner;
      _localityMidpointTwice = preferredMidpointTwice;
      _localityValid = true;
    }

    // Returns the worker whose row range contains the preferred midpoint if it is in the heap.
    // Otherwise returns whichever nearest heap worker on either side has the closer row-range
    // midpoint, preferring the lower index on ties, or -1 if the heap is empty. Until the next Push
    // or SetLoad, calls with the same midpoint resume the previous scan, so the scan work across
    // consecutive PopLeast calls for the same rows is linear in the number of workers.
    [[nodiscard]] int NearestActiveWorker(int64_t preferredMidpointTwice) {
      if (!_localityValid || _localityMidpointTwice != preferredMidpointTwice) {
        ResetLocality(preferredMidpointTwice);
      }
      int const numWorkers = isize(_loads);
      while (_localityLeft >= 0 && _heapPositions[_localityLeft] == kInactiveHeapPosition) {
        --_localityLeft;
      }
      while (_localityRight < numWorkers &&
             _heapPositions[_localityRight] == kInactiveHeapPosition) {
        ++_localityRight;
      }
      if (_localityLeft < 0) {
        return _localityRight < numWorkers ? _localityRight : -1;
      }
      if (_localityRight >= numWorkers) {
        return _localityLeft;
      }
      int64_t const leftDistance =
          Distance(RowMidpointTwice(_localityLeft), preferredMidpointTwice);
      int64_t const rightDistance =
          Distance(RowMidpointTwice(_localityRight), preferredMidpointTwice);
      return leftDistance <= rightDistance ? _localityLeft : _localityRight;
    }

    DynamicArray<double>& _loads;
    DynamicArray<int>& _heapWorkerIds;
    DynamicArray<int>& _heapPositions;
    Span<int const> _workerRowRanges;
    double _maximumLoad{0.0};
    int64_t _localityMidpointTwice{0};
    int _localityLeft{0};
    int _localityRight{0};
    bool _localityValid{false};
  };

  struct RemainderCandidate {
    double finish = 0.0;
    int workerId = 0;
  };

  struct RemainderCandidateGreater {
    bool operator()(RemainderCandidate const& lhs, RemainderCandidate const& rhs) const {
      return lhs.finish != rhs.finish ? lhs.finish > rhs.finish : lhs.workerId > rhs.workerId;
    }
  };

  // Scratch arrays reused by the SimulateBroad and TrySimulateReuseMatVecRanges calls of one plan.
  struct SimulationScratch {
    SimulationScratch(int numWorkers, Allocator* allocator)
        : loads(numWorkers, allocator),
          heapPositions(numWorkers, allocator),
          selectedWorkers(allocator),
          blockCounts(allocator),
          roundingHeap(allocator),
          heapWorkerIds(allocator) {
      selectedWorkers.reserve(numWorkers);
      blockCounts.reserve(numWorkers);
      roundingHeap.reserve(numWorkers);
      heapWorkerIds.reserve(numWorkers);
    }

    // Member destruction reverses this allocation order, as required by FiloAllocator.
    DynamicArray<double> loads;
    DynamicArray<int> heapPositions;
    DynamicArray<int> selectedWorkers;
    DynamicArray<int> blockCounts;
    DynamicArray<RemainderCandidate> roundingHeap;
    DynamicArray<int> heapWorkerIds;
  };

 public:
  static constexpr auto kType = PreconditionerType::PerActor;

  /** @brief Construct a per-actor preconditioner from an ordered row partition.
   *
   * @param[in] actorPrecs Entries must have positive sizes, be contiguous in increasing offset
   * order starting at zero, and reference preconditioners that outlive this object.
   */
  explicit PerActorPrec(std::vector<ActorPreconditionerEntry<T>>&& actorPrecs)
      : _actorPrecs(std::move(actorPrecs)) {}

  ~PerActorPrec() override = default;
  MOCHI_DECLARE_NO_COPY_NO_MOVE(PerActorPrec);

  void Solve(ColumnVectorView<T const> x, ColumnVectorView<T> y) const override {
    // TODO[T175051452]: Introduce efficient parallelization.
    for (auto const& actor : _actorPrecs) {
      actor.Solve(x, y);
    }
  }

  void ConcurrentSolve(
      ColumnVectorView<T const> x,
      ColumnVectorView<T> y,
      ParallelWorkerInfo const& data) const override {
    MOCHI_ASSERT_VERBOSE(_concurrentSolvePlan, "Concurrent solve was not prepared.");
    auto const& plan = *_concurrentSolvePlan;
    [[maybe_unused]] int const planNumWorkers = isize(plan.workerTasks);
    MOCHI_ASSERT_VERBOSE(
        data.numWorkers == planNumWorkers && data.workerId >= 0 && data.workerId < planNumWorkers,
        "Invalid parallel worker information.");

    for (auto const& task : plan.workerTasks[data.workerId]) {
      auto const& actor = _actorPrecs[task.actorIndex];
      switch (task.mode) {
        case ActorPreconditionerParallelMode::SingleWorker:
          actor.Solve(x, y);
          break;
        case ActorPreconditionerParallelMode::IndependentRows:
          actor.ConcurrentSolve(
              x,
              y,
              ParallelWorkerInfo{
                  data.workerId,
                  data.numWorkers,
                  task.actorRowBegin,
                  task.actorRowEnd,
                  data.barrier});
          break;
        case ActorPreconditionerParallelMode::SynchronizedTeam:
          MOCHI_ASSERT_VERBOSE(task.teamBarrier.has_value(), "Team barrier is not set.");
          actor.ConcurrentSolve(
              x,
              y,
              ParallelWorkerInfo{
                  task.teamWorkerId,
                  task.teamSize,
                  task.actorRowBegin,
                  task.actorRowEnd,
                  *task.teamBarrier});
          break;
        default:
          MOCHI_ASSERT(false, "Invalid actor preconditioner parallel mode.");
          break;
      }
    }

    // Nonlocal writes must finish before another worker reads the output.
    if (plan.requiresFinalBarrier) {
      data.BarrierWait();
    }
  }

  /** @brief Plans which workers apply which actor rows in @ref ConcurrentSolve to heuristically
   * minimize its predicted duration, based on each actor's @ref ActorPreconditionerCost.
   *
   * @note Each @p workerRowRanges boundary inside an independent-row actor must be a multiple of
   * that actor's row block size, measured from the actor's first row.
   */
  void PrepareConcurrentSolve(Span<int const> workerRowRanges) const override {
    _concurrentSolvePlan.emplace(MakeConcurrentSolvePlan(workerRowRanges));
  }

  /// @note Invalidates any state created by @ref PrepareConcurrentSolve.
  void Update(IslandOperators<T> const& A);

  constexpr PreconditionerType GetType() const override {
    return kType;
  }

 private:
  // Empirical affine barrier-cost model. Actual costs may differ substantially by architecture.
  [[nodiscard]] static double EstimatedBarrierCost(int numWorkers) {
    MOCHI_ASSERT_VERBOSE(numWorkers > 0, "The number of workers must be positive.");
    return numWorkers == 1 ? 0.0 : 1500.0 + 1160.0 * numWorkers;
  }

  // On p workers, estimated duration is fixed work plus parallel work scaled by the largest
  // aligned-block share, plus synchronization within the actor.
  [[nodiscard]] static double EstimatedDuration(ActorInfo const& actor, int numWorkers) {
    MOCHI_ASSERT_VERBOSE(
        actor.numBlockRows > 0 && numWorkers > 0 && numWorkers <= actor.maxConcurrentWorkers);
    int const blocksPerWorker = 1 + (actor.numBlockRows - 1) / numWorkers;
    return actor.cost.fixedCost +
        actor.cost.parallelCost * (static_cast<double>(blocksPerWorker) / actor.numBlockRows) +
        actor.cost.numTeamBarriers * EstimatedBarrierCost(numWorkers);
  }

  // Returns the team width in [1, widthLimit] with the earliest predicted overall finish,
  // max(minimumFinish, teamStart(width) + duration), preferring the narrowest on ties.
  template <typename TeamStart>
  [[nodiscard]] static int BestSynchronizedWidth(
      ActorInfo const& actor,
      int widthLimit,
      double minimumFinish,
      TeamStart const& teamStart) {
    MOCHI_ASSERT_VERBOSE(
        actor.mode == ActorPreconditionerParallelMode::SynchronizedTeam && widthLimit > 0 &&
        widthLimit <= actor.maxConcurrentWorkers);
    int bestWidth = 1;
    double bestFinish = Max(minimumFinish, teamStart(1) + EstimatedDuration(actor, 1));
    for (int width = 2; width <= widthLimit; ++width) {
      double const finish = Max(minimumFinish, teamStart(width) + EstimatedDuration(actor, width));
      if (finish < bestFinish) {
        bestWidth = width;
        bestFinish = finish;
      }
    }
    return bestWidth;
  }

  // Tie-breaker in MakeActorOrder for actors with equal serial cost.
  [[nodiscard]] static int ModePriority(ActorPreconditionerParallelMode mode) {
    switch (mode) {
      case ActorPreconditionerParallelMode::SynchronizedTeam:
        return 0;
      case ActorPreconditionerParallelMode::SingleWorker:
        return 1;
      case ActorPreconditionerParallelMode::IndependentRows:
        return 2;
      default:
        MOCHI_ASSERT(false, "Invalid actor preconditioner parallel mode.");
        return 3;
    }
  }

  [[nodiscard]] static bool IsLocalWrite(
      int workerId,
      int globalRowBegin,
      int globalRowEnd,
      Span<int const> workerRowRanges) {
    return globalRowBegin >= workerRowRanges[workerId] &&
        globalRowEnd <= workerRowRanges[workerId + 1];
  }

  // A null plan estimates cost only; a non-null plan records each task.
  static void AddTask(ConcurrentSolvePlan* plan, int workerId, ConcurrentSolveTask&& task) {
    if (plan != nullptr) {
      plan->workerTasks[workerId].push_back(std::move(task));
    }
  }

  void ScheduleSingleWorker(
      ActorInfo const& info,
      WorkerLoadHeap& workers,
      Span<int const> workerRowRanges,
      bool& requiresFinalBarrier,
      ConcurrentSolvePlan* plan) const {
    auto const& actor = _actorPrecs[info.actorIndex];
    int const workerId = workers.FindLeast(actor.offset, actor.offset + actor.size);
    workers.SetLoad(workerId, workers.Load(workerId) + info.serialCost);
    requiresFinalBarrier = requiresFinalBarrier ||
        !IsLocalWrite(workerId, actor.offset, actor.offset + actor.size, workerRowRanges);
    AddTask(
        plan,
        workerId,
        ConcurrentSolveTask{
            .actorIndex = info.actorIndex,
            .mode = info.mode,
            .actorRowBegin = 0,
            .actorRowEnd = actor.size});
  }

  void ScheduleIndependentRows(
      ActorInfo const& info,
      int numWorkers,
      WorkerLoadHeap& workers,
      SimulationScratch& scratch,
      Span<int const> workerRowRanges,
      bool& requiresFinalBarrier,
      ConcurrentSolvePlan* plan) const {
    MOCHI_ASSERT_VERBOSE(numWorkers > 0 && numWorkers <= info.maxConcurrentWorkers);
    auto const& actor = _actorPrecs[info.actorIndex];
    double const blockCost = info.cost.parallelCost / info.numBlockRows;
    MOCHI_ASSERT_VERBOSE(
        blockCost > 0.0 && IsFinite(blockCost),
        "Independent-row block cost must be positive and finite.");
    double const totalCost = info.cost.parallelCost;

    scratch.selectedWorkers.resize_noinit(numWorkers);
    for (int i = 0; i < numWorkers; ++i) {
      scratch.selectedWorkers[i] = workers.PopLeast(actor.offset, actor.offset + actor.size);
    }

    // Equalize predicted finish times as far as whole blocks allow, then place each remaining block
    // on the worker that would finish earliest.
    int activeWorkers = 1;
    double loadSum = workers.Load(scratch.selectedWorkers[0]);
    while (activeWorkers < numWorkers) {
      double const target = (loadSum + totalCost) / activeWorkers;
      if (target <= workers.Load(scratch.selectedWorkers[activeWorkers])) {
        break;
      }
      loadSum += workers.Load(scratch.selectedWorkers[activeWorkers]);
      ++activeWorkers;
    }
    double const target = (loadSum + totalCost) / activeWorkers;
    MOCHI_ASSERT_VERBOSE(IsFinite(target), "Independent-row target cost must be finite.");

    scratch.blockCounts.resize_noinit(scratch.loads.size());
    for (int workerId : scratch.selectedWorkers) {
      scratch.blockCounts[workerId] = 0;
    }
    int assignedBlocks = 0;
    for (int i = 0; i < activeWorkers; ++i) {
      int const count = Clamp(
          static_cast<int>((target - workers.Load(scratch.selectedWorkers[i])) / blockCost),
          0,
          info.numBlockRows - assignedBlocks);
      scratch.blockCounts[scratch.selectedWorkers[i]] = count;
      assignedBlocks += count;
    }

    scratch.roundingHeap.clear();
    for (int i = 0; i < numWorkers; ++i) {
      int const workerId = scratch.selectedWorkers[i];
      scratch.roundingHeap.push_back(
          RemainderCandidate{
              .finish = workers.Load(workerId) + scratch.blockCounts[workerId] * blockCost,
              .workerId = workerId});
    }
    std::make_heap(
        scratch.roundingHeap.begin(), scratch.roundingHeap.end(), RemainderCandidateGreater{});
    while (assignedBlocks < info.numBlockRows) {
      std::pop_heap(
          scratch.roundingHeap.begin(), scratch.roundingHeap.end(), RemainderCandidateGreater{});
      auto candidate = scratch.roundingHeap.back();
      scratch.roundingHeap.pop_back();
      ++scratch.blockCounts[candidate.workerId];
      ++assignedBlocks;
      candidate.finish += blockCost;
      scratch.roundingHeap.push_back(candidate);
      std::push_heap(
          scratch.roundingHeap.begin(), scratch.roundingHeap.end(), RemainderCandidateGreater{});
    }

    // Assign contiguous actor blocks in worker-row order to preserve write locality.
    std::sort(scratch.selectedWorkers.begin(), scratch.selectedWorkers.end());
    int blockBegin = 0;
    for (int workerId : scratch.selectedWorkers) {
      int const blockCount = scratch.blockCounts[workerId];
      int const blockEnd = blockBegin + blockCount;
      if (blockBegin != blockEnd) {
        int const actorRowBegin = blockBegin * info.rowBlockSize;
        int const actorRowEnd = blockEnd * info.rowBlockSize;
        requiresFinalBarrier = requiresFinalBarrier ||
            !IsLocalWrite(workerId,
                          actor.offset + actorRowBegin,
                          actor.offset + actorRowEnd,
                          workerRowRanges);
        AddTask(
            plan,
            workerId,
            ConcurrentSolveTask{
                .actorIndex = info.actorIndex,
                .mode = info.mode,
                .actorRowBegin = actorRowBegin,
                .actorRowEnd = actorRowEnd});
      }
      blockBegin = blockEnd;
      workers.Push(workerId, workers.Load(workerId) + blockCount * blockCost);
    }
    MOCHI_ASSERT_VERBOSE(
        blockBegin == info.numBlockRows, "Actor block rows were not fully assigned.");
  }

  void ScheduleSynchronizedTeam(
      ActorInfo const& info,
      Span<int const> workers,
      WorkerLoadHeap& workerHeap,
      Span<int const> workerRowRanges,
      bool& requiresFinalBarrier,
      ConcurrentSolvePlan* plan) const {
    int const teamSize = isize(workers);
    // The team starts when every member is free; all members remain busy until the common finish.
    double start = 0.0;
    for (int workerId : workers) {
      start = Max(start, workerHeap.Load(workerId));
    }
    double const finish = start + EstimatedDuration(info, teamSize);
    for (int workerId : workers) {
      workerHeap.SetLoad(workerId, finish);
    }

    auto const& actor = _actorPrecs[info.actorIndex];
    std::optional<ParallelBarrier> teamBarrier = std::nullopt;
    if (plan != nullptr) {
      // Each task gets a copy of the same barrier, so the copies share one counter.
      teamBarrier.emplace(teamSize);
    }
    for (int teamWorkerId = 0; teamWorkerId < teamSize; ++teamWorkerId) {
      int const blockBegin =
          static_cast<int>(static_cast<int64_t>(teamWorkerId) * info.numBlockRows / teamSize);
      int const blockEnd =
          static_cast<int>((static_cast<int64_t>(teamWorkerId) + 1) * info.numBlockRows / teamSize);
      int const actorRowBegin = blockBegin * info.rowBlockSize;
      int const actorRowEnd = blockEnd * info.rowBlockSize;
      int const workerId = workers[teamWorkerId];
      requiresFinalBarrier =
          requiresFinalBarrier ||
          !IsLocalWrite(
              workerId, actor.offset + actorRowBegin, actor.offset + actorRowEnd, workerRowRanges);
      AddTask(
          plan,
          workerId,
          ConcurrentSolveTask{
              .actorIndex = info.actorIndex,
              .mode = info.mode,
              .actorRowBegin = actorRowBegin,
              .actorRowEnd = actorRowEnd,
              .teamWorkerId = teamWorkerId,
              .teamSize = teamSize,
              .teamBarrier = teamBarrier});
    }
  }

  // Validate actor scheduling metadata and derive the worker limits used by the planner.
  [[nodiscard]] DynamicArray<ActorInfo> MakeActorInfos(int numWorkers, Allocator* allocator) const {
    DynamicArray<ActorInfo> infos(allocator);
    infos.reserve(_actorPrecs.size());
    for (int actorIndex = 0; actorIndex < isize(_actorPrecs); ++actorIndex) {
      auto const& actor = _actorPrecs[actorIndex];
      auto const parallelism = actor.preconditioner.get().GetConcurrentSolveRequirements();
      auto const cost = actor.preconditioner.get().GetConcurrentSolveCost();
      MOCHI_ASSERT_VERBOSE(
          IsFinite(cost.fixedCost) && IsFinite(cost.parallelCost) && cost.fixedCost >= 0.0 &&
              cost.parallelCost >= 0.0 && IsFinite(cost.fixedCost + cost.parallelCost) &&
              cost.fixedCost + cost.parallelCost > 0.0 && cost.maxUsefulWorkers > 0 &&
              cost.numTeamBarriers >= 0,
          "Invalid actor preconditioner cost.");
      MOCHI_ASSERT(actor.size > 0, "Actor size must be positive.");
      int const rowBlockSize = parallelism.mode == ActorPreconditionerParallelMode::SingleWorker
          ? actor.size
          : parallelism.rowBlockSize;
      MOCHI_ASSERT_VERBOSE(
          rowBlockSize > 0 && actor.size % rowBlockSize == 0,
          "Actor size must be divisible by a positive row block size.");
      int const numBlockRows = actor.size / rowBlockSize;
      if (parallelism.mode == ActorPreconditionerParallelMode::SingleWorker) {
        MOCHI_ASSERT(
            cost.maxUsefulWorkers == 1, "Single-worker actors must have maxUsefulWorkers == 1.");
        MOCHI_ASSERT(cost.numTeamBarriers == 0, "Single-worker actors cannot have team barriers.");
      }
      if (parallelism.mode == ActorPreconditionerParallelMode::IndependentRows) {
        MOCHI_ASSERT(cost.fixedCost == 0.0, "Independent-row actors cannot have fixed cost.");
        MOCHI_ASSERT(
            cost.parallelCost > 0.0, "Independent-row actors must have positive parallel cost.");
        MOCHI_ASSERT(
            cost.numTeamBarriers == 0, "Independent-row actors cannot have team barriers.");
      }
      ActorInfo info{
          .actorIndex = actorIndex,
          .mode = parallelism.mode,
          .rowBlockSize = rowBlockSize,
          .numBlockRows = numBlockRows,
          .maxConcurrentWorkers = Min(cost.maxUsefulWorkers, numBlockRows, numWorkers),
          .cost = cost,
          .serialCost = cost.fixedCost + cost.parallelCost};
      if (parallelism.mode == ActorPreconditionerParallelMode::SynchronizedTeam) {
        // Exclude synchronized-team widths slower than the actor's best standalone width.
        info.maxConcurrentWorkers =
            BestSynchronizedWidth(info, info.maxConcurrentWorkers, 0.0, [](int) { return 0.0; });
      }
      infos.push_back(info);
    }
    return infos;
  }

  // A shared actor order prevents barrier cycles between overlapping synchronized teams.
  [[nodiscard]] static DynamicArray<int> MakeActorOrder(
      Span<ActorInfo const> infos,
      Allocator* allocator) {
    DynamicArray<int> order(allocator);
    order.resize_noinit(infos.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
      if (infos[lhs].serialCost != infos[rhs].serialCost) {
        return infos[lhs].serialCost > infos[rhs].serialCost;
      }
      int const lhsPriority = ModePriority(infos[lhs].mode);
      int const rhsPriority = ModePriority(infos[rhs].mode);
      return lhsPriority != rhsPriority ? lhsPriority < rhsPriority
                                        : infos[lhs].actorIndex < infos[rhs].actorIndex;
    });
    return order;
  }

  // Initial width caps for SimulateBroad: about the number of workers each actor needs to finish
  // within a lower bound on the predicted plan duration. The short actors' combined share is
  // rounded up to whole workers. If that leaves at least half a worker idle, spareInfoIndex
  // receives the actor that would get one more worker, and otherwise -1.
  [[nodiscard]] static DynamicArray<int> MakeBaseCaps(
      Span<ActorInfo const> infos,
      Span<int const> actorOrder,
      int numWorkers,
      int& spareInfoIndex,
      Allocator* allocator) {
    DynamicArray<int> caps(infos.size(), 1, allocator);
    spareInfoIndex = -1;
    if (actorOrder.empty()) {
      return caps;
    }
    double totalCost = 0.0;
    double globalH = 0.0;
    for (int infoIndex : actorOrder) {
      auto const& info = infos[infoIndex];
      MOCHI_ASSERT_VERBOSE(
          IsFinite(totalCost + info.serialCost),
          "Aggregate actor preconditioner cost must be finite.");
      totalCost += info.serialCost;
      globalH = Max(globalH, EstimatedDuration(info, info.maxConcurrentWorkers));
    }
    // Derive initial width caps from the larger of total work per worker and the slowest actor at
    // its worker limit.
    globalH = Max(globalH, totalCost / numWorkers);
    MOCHI_ASSERT_VERBOSE(
        globalH > 0.0 && IsFinite(globalH),
        "Global actor preconditioner cost must be positive and finite.");

    double smallUse = 0.0;
    int usedLargeWorkers = 0;
    for (int infoIndex : actorOrder) {
      auto const& info = infos[infoIndex];
      double const share = info.serialCost / globalH;
      if (share < 1.0) {
        smallUse += share;
      } else {
        caps[infoIndex] = Min(info.maxConcurrentWorkers, Max(1, static_cast<int>(Floor(share))));
        usedLargeWorkers += caps[infoIndex];
      }
    }

    // Short actors reserve fractional capacity; remaining whole-worker slots go to the currently
    // slowest large actor.
    int const largeWorkerBudget =
        Clamp(static_cast<int>(Floor(numWorkers - smallUse)), usedLargeWorkers, numWorkers);
    DynamicArray<RefillCandidate> refillHeap(allocator);
    refillHeap.reserve(actorOrder.size());
    auto addRefillCandidate = [&](int infoIndex) {
      auto const& info = infos[infoIndex];
      double const share = info.serialCost / globalH;
      if (share >= 1.0 && caps[infoIndex] < info.maxConcurrentWorkers) {
        refillHeap.push_back(
            RefillCandidate{
                .duration = EstimatedDuration(info, caps[infoIndex]), .infoIndex = infoIndex});
        std::push_heap(refillHeap.begin(), refillHeap.end(), RefillCandidateLess{});
      }
    };
    for (int infoIndex : actorOrder) {
      addRefillCandidate(infoIndex);
    }
    while (usedLargeWorkers < largeWorkerBudget && !refillHeap.empty()) {
      std::pop_heap(refillHeap.begin(), refillHeap.end(), RefillCandidateLess{});
      auto const best = refillHeap.back();
      refillHeap.pop_back();
      ++caps[best.infoIndex];
      ++usedLargeWorkers;
      addRefillCandidate(best.infoIndex);
    }

    if (usedLargeWorkers < static_cast<int>(Round(numWorkers - smallUse)) && !refillHeap.empty()) {
      spareInfoIndex = refillHeap.front().infoIndex;
    }
    return caps;
  }

  // Broad schedules every actor from one shared worker heap.
  double SimulateBroad(
      Span<int const> workerRowRanges,
      Span<ActorInfo const> infos,
      Span<int const> order,
      Span<int const> baseCaps,
      double finalBarrierCost,
      SimulationScratch& scratch,
      ConcurrentSolvePlan* plan) const {
    int const numWorkers = isize(workerRowRanges) - 1;
    WorkerLoadHeap workers(
        scratch.loads, scratch.heapWorkerIds, scratch.heapPositions, workerRowRanges);
    bool requiresFinalBarrier = false;
    int remainingActors = isize(order);

    for (int infoIndex : order) {
      auto const& info = infos[infoIndex];
      // Relax proportional caps near the tail so remaining actors can consume idle workers.
      int const tailCap = Max(1, numWorkers / remainingActors);
      int const widthLimit = Min(info.maxConcurrentWorkers, Max(baseCaps[infoIndex], tailCap));
      switch (info.mode) {
        case ActorPreconditionerParallelMode::SingleWorker:
          ScheduleSingleWorker(info, workers, workerRowRanges, requiresFinalBarrier, plan);
          break;
        case ActorPreconditionerParallelMode::IndependentRows:
          ScheduleIndependentRows(
              info, widthLimit, workers, scratch, workerRowRanges, requiresFinalBarrier, plan);
          break;
        case ActorPreconditionerParallelMode::SynchronizedTeam: {
          auto const& actor = _actorPrecs[info.actorIndex];
          double const currentMax = workers.MaxLoad();
          scratch.selectedWorkers.resize_noinit(widthLimit);
          for (int i = 0; i < widthLimit; ++i) {
            scratch.selectedWorkers[i] = workers.PopLeast(actor.offset, actor.offset + actor.size);
          }

          // Workers pop in nondecreasing load order, so each team starts when its last member is
          // free.
          int const bestWidth = BestSynchronizedWidth(info, widthLimit, currentMax, [&](int width) {
            return workers.Load(scratch.selectedWorkers[width - 1]);
          });

          for (int workerId : scratch.selectedWorkers) {
            workers.Push(workerId, workers.Load(workerId));
          }
          scratch.selectedWorkers.resize_noinit(bestWidth);
          std::sort(scratch.selectedWorkers.begin(), scratch.selectedWorkers.end());
          ScheduleSynchronizedTeam(
              info,
              MakeConstSpan(scratch.selectedWorkers),
              workers,
              workerRowRanges,
              requiresFinalBarrier,
              plan);
          break;
        }
        default:
          MOCHI_ASSERT(false, "Invalid actor preconditioner parallel mode.");
          break;
      }

      --remainingActors;
    }
    if (plan != nullptr) {
      plan->requiresFinalBarrier = requiresFinalBarrier;
    }
    return workers.MaxLoad() + (requiresFinalBarrier && numWorkers > 1 ? finalBarrierCost : 0.0);
  }

  // Reuse matvec row ownership exactly when it respects every actor's worker limit.
  std::optional<double> TrySimulateReuseMatVecRanges(
      Span<int const> workerRowRanges,
      Span<ActorInfo const> infos,
      SimulationScratch& scratch,
      ConcurrentSolvePlan* plan) const {
    int const numWorkers = isize(workerRowRanges) - 1;
    std::fill(scratch.loads.begin(), scratch.loads.end(), 0.0);
    int infoIndex = 0;
    int workerId = 0;
    int actorWorkerCount = 0;
    while (infoIndex < isize(infos) && workerId < numWorkers) {
      auto const& info = infos[infoIndex];
      auto const& actor = _actorPrecs[info.actorIndex];
      int const actorRowEnd = actor.offset + actor.size;
      int const workerRowEnd = workerRowRanges[workerId + 1];
      int const globalRowBegin = Max(actor.offset, workerRowRanges[workerId]);
      int const globalRowEnd = Min(actorRowEnd, workerRowEnd);
      if (globalRowBegin < globalRowEnd) {
        if (++actorWorkerCount > info.maxConcurrentWorkers) {
          return std::nullopt;
        }
        int const localRowBegin = globalRowBegin - actor.offset;
        int const localRowEnd = globalRowEnd - actor.offset;
        MOCHI_ASSERT_VERBOSE(
            localRowBegin % info.rowBlockSize == 0 && localRowEnd % info.rowBlockSize == 0,
            "Matvec row ranges must begin and end on actor row-block boundaries.");
        MOCHI_ASSERT_VERBOSE(
            IsLocalWrite(workerId, globalRowBegin, globalRowEnd, workerRowRanges),
            "Matvec-range task must write only rows owned by its worker.");
        int const numBlocks = (localRowEnd - localRowBegin) / info.rowBlockSize;
        scratch.loads[workerId] +=
            info.cost.parallelCost * static_cast<double>(numBlocks) / info.numBlockRows;
        AddTask(
            plan,
            workerId,
            ConcurrentSolveTask{
                .actorIndex = info.actorIndex,
                .mode = info.mode,
                .actorRowBegin = localRowBegin,
                .actorRowEnd = localRowEnd});
      }
      if (actorRowEnd <= workerRowEnd) {
        ++infoIndex;
        actorWorkerCount = 0;
      }
      if (workerRowEnd <= actorRowEnd) {
        ++workerId;
      }
    }
    MOCHI_ASSERT_VERBOSE(
        infoIndex == isize(infos), "Matvec row ranges do not cover all actor rows.");
    if (plan != nullptr) {
      plan->requiresFinalBarrier = false;
    }
    return *std::ranges::max_element(scratch.loads);
  }

  // A ConcurrentSolve output is complete only when its most loaded worker finishes. Letting each
  // worker apply the rows it owns for the matrix-vector product can leave other workers idle,
  // because actors differ in cost per row, parallel mode, and useful worker count. The planner
  // therefore balances worker loads predicted from actor costs. Inaccurate predictions leave
  // workers idle in every application that reuses the plan but do not change the result.
  //
  // Broad visits actors in MakeActorOrder's order and gives each the least-loaded workers: one for
  // a single-worker actor, aligned row ranges for an independent-row actor, and, for a synchronized
  // actor, the team width with the lowest predicted overall finish. When every actor supports
  // independent rows, reusing the matrix-vector row ranges is also estimated and is kept if it
  // respects every actor's worker limit and is at least as fast as Broad. Otherwise, Broad gives
  // MakeBaseCaps's spare worker, if any, to its actor only if that is predicted faster. Predictions
  // include waiting for teammates, team barriers, and, if any worker writes rows that another
  // worker owns for the matrix-vector product, the final all-worker barrier.
  //
  // TODO(T290274539): Known planner gaps:
  // - MakeActorOrder orders by serial cost, ignoring how widely each actor can spread. An actor
  //   with a long shortest duration, such as a single-worker actor, that is slightly cheaper than
  //   spreadable actors is scheduled after them on top of the loads they balanced, approaching
  //   twice the optimal duration. Also estimating Broad in decreasing
  //   EstimatedDuration(info, info.maxConcurrentWorkers) order fixes such cases, but preliminary
  //   analysis on synthetic islands found the measured gain too rare and small to justify
  //   doubling the planning cost. See D121469076.
  // - TeamReuse, which runs synchronized actors on nested teams drawn from a shared worker prefix,
  //   measured slower than Broad overall in preliminary analysis on synthetic islands. See
  //   D120605491.
  // - ScheduleSingleWorker: When no final barrier is required yet, consider comparing the
  //   least-loaded worker with the local owner. The first nonlocal assignment adds one full-worker
  //   barrier every time the preconditioner is applied and can cost more than the load imbalance
  //   it removes.
  // - Tasks writing rows another worker owns for the matrix-vector product are charged only the
  //   final all-worker barrier, not the cache-line transfers: reading the owner's input rows,
  //   taking ownership of the output rows, and the owner reading them back in the next dot product.
  //   Add this cost to predictions. See D121889725.
  // - Skip simulating Broad when reusing the matvec ranges reaches a lower bound on any plan's
  //   duration.
  // - For an independent-row actor, adding a worker never increases predicted cost: fixedCost is
  //   zero and parallelCost is split among the workers. A small actor may therefore be split across
  //   more workers than pays off. If profiling shows this, add a per-worker overhead to the model.
  // - Short actors holding more than half a worker still reserve a whole one, so at 2 workers a
  //   dominant actor stays on one worker while the other worker finishes the short actors early
  //   and waits.
  // - PrepareConcurrentSolve replans on every call, adding a few microseconds per solve, which is
  //   significant for small systems. Caching would amortize this and could justify a costlier
  //   planner, but ParallelPCG's worker count is nondeterministic, so it needs one plan per worker
  //   count, each invalidated when the actor partition or costs change. If caching is impractical,
  //   consider scoring independent candidates concurrently across workers.
  [[nodiscard]] ConcurrentSolvePlan MakeConcurrentSolvePlan(Span<int const> workerRowRanges) const {
    int const numWorkers = isize(workerRowRanges) - 1;
    MOCHI_ASSERT(numWorkers > 0, "At least one worker is required.");
    int numRows = 0;
    for (auto const& actor : _actorPrecs) {
      MOCHI_ASSERT(
          actor.offset == numRows,
          "Actor preconditioners must have contiguous row ranges in offset order.");
      numRows += actor.size;
    }
    MOCHI_ASSERT(
        workerRowRanges[0] == 0 && workerRowRanges[numWorkers] == numRows,
        "Worker row ranges do not cover the preconditioner.");
    MOCHI_ASSERT_VERBOSE(
        std::is_sorted(workerRowRanges.begin(), workerRowRanges.end()),
        "Worker row ranges must be nondecreasing.");

    // Keep transient planner arrays off the heap for typical islands.
    MOCHI_FILO_STACK_ALLOCATOR(planningAllocator, 16 * 1024);

    // Validate actor metadata and establish the relative order shared by every candidate.
    auto const infos = MakeActorInfos(numWorkers, &planningAllocator);
    auto const order = MakeActorOrder(MakeConstSpan(infos), &planningAllocator);
    int spareInfoIndex = -1;
    auto baseCaps = MakeBaseCaps(
        MakeConstSpan(infos), MakeConstSpan(order), numWorkers, spareInfoIndex, &planningAllocator);
    bool allIndependent = true;
    for (auto const& info : infos) {
      allIndependent =
          allIndependent && info.mode == ActorPreconditionerParallelMode::IndependentRows;
    }
    double const finalBarrierCost = EstimatedBarrierCost(numWorkers);

    SimulationScratch scratch(numWorkers, &planningAllocator);
    auto const makeEmptyPlan = [numWorkers] {
      return ConcurrentSolvePlan{
          .workerTasks = DynamicArray<DynamicArray<ConcurrentSolveTask>>(numWorkers),
          .requiresFinalBarrier = false};
    };
    ConcurrentSolvePlan plan = makeEmptyPlan();
    auto const simulateBroad = [&](ConcurrentSolvePlan* target) {
      return SimulateBroad(
          workerRowRanges,
          MakeConstSpan(infos),
          MakeConstSpan(order),
          MakeConstSpan(baseCaps),
          finalBarrierCost,
          scratch,
          target);
    };

    // The candidates are:
    // - Broad, which gives each actor at most its width cap in workers: its entry in baseCaps,
    //   raised near the end of the order to the remaining actors' even share of the workers.
    // - Broad with the spare worker: when rounding in MakeBaseCaps leaves at least half a worker
    //   idle, the cap of the actor at spareInfoIndex is raised by one.
    // - Reusing the matrix-vector row ranges. Reuse is possible when every actor supports
    //   independent rows and the ranges respect every actor's worker limit.
    // Simulating a candidate predicts its duration and, if given a plan, also builds it; building
    // adds less time than a second simulation. If reuse is possible, Broad candidates are only
    // predicted and the fastest is simulated again to build it, so no Broad plan is built and then
    // discarded for reuse. Otherwise, each Broad candidate is built as it is simulated.
    auto const reuseDuration = allIndependent
        ? TrySimulateReuseMatVecRanges(workerRowRanges, MakeConstSpan(infos), scratch, nullptr)
        : std::nullopt;
    bool const buildEachBroadCandidate = !reuseDuration.has_value();
    double bestBroadDuration = simulateBroad(buildEachBroadCandidate ? &plan : nullptr);

    // Reuse is compared with Broad before the spare worker is tried: in benchmarks, when only Broad
    // with the spare worker was predicted faster than reuse, reuse was measured faster.
    if (reuseDuration.has_value() && *reuseDuration <= bestBroadDuration) {
      [[maybe_unused]] auto const duration =
          TrySimulateReuseMatVecRanges(workerRowRanges, MakeConstSpan(infos), scratch, &plan);
      MOCHI_ASSERT(duration.has_value(), "Selected matvec row ranges must remain valid.");
      return plan;
    }

    // Simulates Broad with the current width caps and keeps the result only if it is predicted
    // strictly faster than the best Broad candidate so far.
    auto const keepBroadIfFaster = [&] {
      ConcurrentSolvePlan candidate =
          buildEachBroadCandidate ? makeEmptyPlan() : ConcurrentSolvePlan{};
      double const duration = simulateBroad(buildEachBroadCandidate ? &candidate : nullptr);
      if (duration >= bestBroadDuration) {
        return false;
      }
      bestBroadDuration = duration;
      if (buildEachBroadCandidate) {
        plan = std::move(candidate);
      }
      return true;
    };

    // The spare worker can still lose, e.g., by pushing short actors onto loaded workers or by
    // adding the final barrier.
    if (spareInfoIndex >= 0) {
      ++baseCaps[spareInfoIndex];
      if (!keepBroadIfFaster()) {
        --baseCaps[spareInfoIndex];
      }
    }
    if (!buildEachBroadCandidate) {
      simulateBroad(&plan);
    }
    return plan;
  }

  std::vector<ActorPreconditionerEntry<T>> _actorPrecs;
  mutable std::optional<ConcurrentSolvePlan> _concurrentSolvePlan;
};

} // namespace mochi
