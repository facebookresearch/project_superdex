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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mochi {
class Context;
} // namespace mochi

namespace superdex::studio {

// Handed to each @ref AsyncTask's work so a long-running task can cooperatively bail out when the
// batch is cancelled. Tasks whose work blocks in a helper subprocess can ignore it (they are
// aborted by killing the subprocess via @ref AsyncTaskRunner::Config::onCancel); CPU-bound tasks
// should poll @ref IsCancelRequested at safe checkpoints and return false (leaving their outputs
// unused) once it is set. Cheap to copy; wraps the runner's cancel flag, which outlives the batch.
class AsyncCancelToken {
 public:
  explicit AsyncCancelToken(std::atomic<bool> const& cancelRequested)
      : _cancelRequested(&cancelRequested) {}

  [[nodiscard]] bool IsCancelRequested() const {
    return _cancelRequested->load();
  }

 private:
  std::atomic<bool> const* _cancelRequested;
};

// A single unit of background work for @ref AsyncTaskRunner.
struct AsyncTask {
  std::string label; // shown in the modal, e.g. "Convert link3.obj -> link3.glb"
  // Runs on a worker thread; returns success. Receives a cancel token to poll (see
  // @ref AsyncCancelToken). A task that returns false because it was cancelled is reported as
  // Cancelled rather than Failed when the batch was cancelled.
  std::function<bool(AsyncCancelToken const&)> work;
};

// Lifecycle of an AsyncTask, surfaced in the progress modal.
enum class AsyncTaskStatus : uint8_t {
  Pending, // not yet claimed by a worker
  Running, // currently executing on a worker
  Succeeded, // finished, work() returned true
  Failed, // finished, work() returned false
  Cancelled, // skipped or aborted because the batch was cancelled
};

// Runs a batch of @ref AsyncTask on one or more async threads.
class AsyncTaskRunner {
 public:
  struct Config {
    std::string title; // modal title, e.g. "Importing URDF"
    std::vector<AsyncTask> tasks; // background work to run
    std::function<void(bool allSucceeded)> onComplete; // runs on the main thread when all done
    mochi::Context* context = nullptr; // scheduler bound on the coordinator; sizes the worker pool
    // Optional: invoked on the main thread when the user clicks Cancel, in addition to the per-task
    // cancel flag that skips not-yet-started tasks. Use it to abort work that is already running
    // but blocked outside the task's control (e.g. terminate a helper subprocess) so its worker
    // thread unblocks promptly. Must be safe to call while a task runs on a worker thread.
    std::function<void()> onCancel;
    // When true the tasks run strictly in order on a single worker instead of in parallel, so a
    // later task may consume an earlier task's result (e.g. a build pipeline). Default parallel.
    bool serial = false;
  };

  AsyncTaskRunner() = default;
  ~AsyncTaskRunner();

  AsyncTaskRunner(AsyncTaskRunner const&) = delete;
  AsyncTaskRunner& operator=(AsyncTaskRunner const&) = delete;
  AsyncTaskRunner(AsyncTaskRunner&&) = delete;
  AsyncTaskRunner& operator=(AsyncTaskRunner&&) = delete;

  // Start running `config.tasks` on worker threads and open the progress modal. With
  // an empty task list, onComplete(true) runs immediately and no modal is shown (so
  // trivial work never flickers a dialog). Must not be called while a batch is running.
  void Begin(Config config);

  // True while a batch is active (workers running or modal still showing).
  bool IsRunning() const;

  // Draw the progress modal for one frame; call once per frame on the main thread.
  // Returns true while the modal is still active, false once the batch has finished.
  bool ShowModalWindow();

 private:
  void JoinWorkers();

  Config _config;
  std::thread _coordinator; // binds the Context scheduler and drives the batch

  std::atomic<int> _completed{0}; // tasks finished (success, failure, or cancelled)
  std::atomic<int> _succeeded{0}; // tasks that returned true
  std::atomic<bool> _workDone{false}; // set by the coordinator once the batch finishes
  std::atomic<bool> _cancelRequested{false}; // set on the main thread when the user clicks Cancel
  int _total = 0; // _config.tasks.size()

  std::mutex _statusMutex; // guards _statuses, _startSec, _endSec
  std::vector<AsyncTaskStatus> _statuses; // per-task lifecycle, indexed like _config.tasks
  // Per-task wall-clock seconds relative to _batchStart; -1 until set. A task's
  // elapsed time is end (or now, while running) minus start. Overlapping
  // [start, end) windows across tasks are direct evidence of parallel execution.
  std::vector<double> _startSec;
  std::vector<double> _endSec;
  std::chrono::steady_clock::time_point _batchStart; // reference for the times above

  bool _running = false; // main-thread flag: batch active
  bool _openModal = false; // request to OpenPopup next frame
};

// A hidden, serial background task queue: runs enqueued jobs one at a time on a single worker
// thread, with no modal and no batch (unlike @ref AsyncTaskRunner). Jobs are enqueued over time and
// identified by an integer key so they can be cancelled individually (keying by, e.g., an owner id
// lets one owner's jobs be dropped without touching another's). Cancellation removes
// not-yet-started jobs and sets a running job's cancel flag; it does NOT interrupt a job already
// running (a blocking helper call runs to completion), so a job's work should poll the flag and
// discard its own result once set. Delivering results back to the main thread is the caller's
// responsibility (the work captures its own thread-safe sink); this queue owns only scheduling and
// cancellation. Thread-safe.
class BackgroundTaskQueue {
 public:
  // Runs on the worker thread. Poll @p cancelled and skip delivering a result once it is set.
  using Work = std::function<void(std::atomic<bool> const& cancelled)>;

  BackgroundTaskQueue();
  ~BackgroundTaskQueue(); // stops the worker and joins (waits for the in-flight job to finish)

  BackgroundTaskQueue(BackgroundTaskQueue const&) = delete;
  BackgroundTaskQueue& operator=(BackgroundTaskQueue const&) = delete;
  BackgroundTaskQueue(BackgroundTaskQueue&&) = delete;
  BackgroundTaskQueue& operator=(BackgroundTaskQueue&&) = delete;

  // Enqueue @p work under @p key. Any not-yet-started job with the same key is dropped first, and a
  // running job with the same key has its cancel flag set (supersede-latest). Runs after the jobs
  // already queued (serial, FIFO).
  void Enqueue(uint64_t key, Work work);

  // Cancel jobs for @p key: remove queued ones and flag a running one to discard its result.
  // @p key == 0 cancels every job.
  void Cancel(uint64_t key);

 private:
  struct Job {
    uint64_t key = 0;
    Work work;
    std::shared_ptr<std::atomic<bool>> cancelled;
  };
  void WorkerLoop();

  std::mutex _mutex;
  std::condition_variable _cv;
  std::deque<Job> _queue;
  std::shared_ptr<std::atomic<bool>> _runningCancel; // cancel flag of the job currently running
  uint64_t _runningKey = 0;
  bool _stop = false;
  std::thread _worker;
};

} // namespace superdex::studio
