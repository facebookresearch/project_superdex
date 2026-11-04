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

#include "core/async_task.h"

#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_physics/cpp_api/mochi_context.h>

#include <imgui.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <memory>
#include <numeric>
#include <optional>
#include <string_view>
#include <thread>
#include <utility>

namespace superdex::studio {

// Wall-clock seconds elapsed since `start`.
static double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

AsyncTaskRunner::~AsyncTaskRunner() {
  JoinWorkers();
}

void AsyncTaskRunner::Begin(Config config) {
  // A previous batch must be fully finished before a new one starts.
  MOCHI_ASSERT(!_running);

  _config = std::move(config);
  _total = static_cast<int>(_config.tasks.size());
  _completed.store(0);
  _succeeded.store(0);
  _workDone.store(false);
  _cancelRequested.store(false);

  // Nothing to do: finalize immediately and show no modal.
  if (_total == 0) {
    if (_config.onComplete) {
      _config.onComplete(true);
    }
    _config = {};
    return;
  }

  MOCHI_ASSERT(_config.context != nullptr);
  for (int index = 0; index < _total; ++index) {
    for (int const dependency : _config.tasks[static_cast<size_t>(index)].dependsOn) {
      // Tasks run in batch order, so a task can only wait on earlier ones.
      MOCHI_ASSERT(dependency >= 0 && dependency < index);
    }
  }

  {
    std::lock_guard<std::mutex> lock(_statusMutex);
    auto const n = static_cast<size_t>(_total);
    _statuses.assign(n, AsyncTaskStatus::Pending);
    _skipRequested = std::make_unique<std::atomic<bool>[]>(n);
    _skipReasons.assign(n, AsyncTaskStatus::Skipped);
    _blockedBy.assign(n, -1);
    _workers.assign(n, std::thread::id{});
    _startSec.assign(n, -1.0);
    _endSec.assign(n, -1.0);
  }
  _batchStart = std::chrono::steady_clock::now();

  MOCHI_LOG(
      "Async batch '%s' started (%d task%s)",
      _config.title.c_str(),
      _total,
      _total == 1 ? "" : "s");

  _running = true;
  _openModal = true;

  // A single coordinator thread drives the batch, which may execute serially or in parallel.
  _coordinator = std::thread([this]() {
    {
      // BindThisThread() ensures that the mesh processing utilities will be able to take advantage
      // of concurrency within the mochi_physics library.
      _config.context->BindThisThread();
      MOCHI_DEFER(_config.context->UnbindThisThread());

      // TODO: Schedule these jobs in parallel if (and only if) !_config.serial, starting each task
      // once its dependencies finish.
      for (int index = 0; index < _total; ++index) {
        RunTask(index);
      }
    }

    _workDone.store(true);
  });
}

void AsyncTaskRunner::RunTask(int index) {
  auto const i = static_cast<size_t>(index);
  AsyncTask const& task = _config.tasks[i];
  {
    std::lock_guard<std::mutex> lock(_statusMutex);
    if (_statuses[i] != AsyncTaskStatus::Pending) {
      return; // skipped before it started
    }
    // Skip tasks that have not started once the batch is cancelled, so a cancel drains the queue
    // without running further work.
    if (_cancelRequested.load()) {
      _statuses[i] = AsyncTaskStatus::Cancelled;
      MOCHI_LOG("Async task cancelled (not started): %s", task.label.c_str());
      _completed.fetch_add(1);
      return;
    }
    // A task whose input is missing would fail, or quietly build from stale files.
    for (int const dependency : task.dependsOn) {
      if (_statuses[static_cast<size_t>(dependency)] != AsyncTaskStatus::Succeeded) {
        _statuses[i] = AsyncTaskStatus::Skipped;
        _blockedBy[i] = dependency;
        MOCHI_LOG_WARNING(
            "Async task skipped: %s (needs %s)",
            task.label.c_str(),
            _config.tasks[static_cast<size_t>(dependency)].label.c_str());
        _completed.fetch_add(1);
        return;
      }
    }
    _statuses[i] = AsyncTaskStatus::Running;
    _workers[i] = std::this_thread::get_id();
    _startSec[i] = SecondsSince(_batchStart);
  }
  MOCHI_LOG("Async task started: %s", task.label.c_str());
  AsyncCancelToken const token(_cancelRequested, _skipRequested[i]);
  bool const ok = task.work ? task.work(token) : false;
  // A task that returned success completed its work and is reported as such even if it was
  // cancelled or skipped while it ran: atomic tasks (e.g. an SDF bake) cannot be interrupted
  // mid-flight, so a stop landing during them does not undo the finished result. Only a task that
  // did not complete (returned false) while a stop was pending is reported as stopped -- e.g. its
  // helper was killed via Config::abortTask; any other incomplete task is a genuine failure.
  AsyncTaskStatus status = AsyncTaskStatus::Succeeded;
  double elapsed = 0.0;
  {
    std::lock_guard<std::mutex> lock(_statusMutex);
    if (!ok) {
      status = _cancelRequested.load() ? AsyncTaskStatus::Cancelled
          : _skipRequested[i].load()   ? _skipReasons[i]
                                       : AsyncTaskStatus::Failed;
    }
    _statuses[i] = status;
    _endSec[i] = SecondsSince(_batchStart);
    elapsed = _endSec[i] - _startSec[i];
  }
  if (status == AsyncTaskStatus::Succeeded) {
    MOCHI_LOG("Async task succeeded (%.2fs): %s", elapsed, task.label.c_str());
    _succeeded.fetch_add(1);
  } else if (status == AsyncTaskStatus::Cancelled) {
    MOCHI_LOG("Async task cancelled (%.2fs): %s", elapsed, task.label.c_str());
  } else if (status == AsyncTaskStatus::Skipped) {
    MOCHI_LOG_WARNING("Async task skipped (%.2fs): %s", elapsed, task.label.c_str());
  } else if (status == AsyncTaskStatus::TimedOut) {
    MOCHI_LOG_WARNING("Async task timed out (%.2fs): %s", elapsed, task.label.c_str());
  } else {
    MOCHI_LOG_WARNING("Async task FAILED (%.2fs): %s", elapsed, task.label.c_str());
  }
  _completed.fetch_add(1);
}

bool AsyncTaskRunner::SkipTaskLocked(int index, AsyncTaskStatus reason) {
  auto const i = static_cast<size_t>(index);
  AsyncTask const& task = _config.tasks[i];
  if (_statuses[i] == AsyncTaskStatus::Pending) {
    _statuses[i] = reason;
    MOCHI_LOG_WARNING("Async task skipped (not started): %s", task.label.c_str());
    _completed.fetch_add(1);
    return true;
  }
  if (_statuses[i] != AsyncTaskStatus::Running || _skipRequested[i].load()) {
    return false;
  }
  _skipReasons[i] = reason;
  _skipRequested[i].store(true);
  if (reason == AsyncTaskStatus::TimedOut) {
    MOCHI_LOG_WARNING(
        "Async task ran longer than %g s; aborting it: %s",
        _config.timeoutSeconds,
        task.label.c_str());
  } else {
    MOCHI_LOG("Async task skip requested by user: %s", task.label.c_str());
  }
  // Under the lock the task is still running on its worker, so the abort cannot reach the next one.
  if (_config.abortTask) {
    _config.abortTask(_workers[i]);
  }
  return true;
}

bool AsyncTaskRunner::SkipTask(std::string_view label) {
  if (!_running) {
    return false;
  }
  std::lock_guard<std::mutex> lock(_statusMutex);
  for (int index = 0; index < _total; ++index) {
    if (_config.tasks[static_cast<size_t>(index)].label == label &&
        SkipTaskLocked(index, AsyncTaskStatus::Skipped)) {
      return true;
    }
  }
  return false;
}

void AsyncTaskRunner::TimeOutTasks() {
  if (!(_config.timeoutSeconds > 0.0)) {
    return;
  }
  double const nowSec = SecondsSince(_batchStart);
  std::lock_guard<std::mutex> lock(_statusMutex);
  for (int index = 0; index < _total; ++index) {
    auto const i = static_cast<size_t>(index);
    if (_statuses[i] == AsyncTaskStatus::Running &&
        nowSec - _startSec[i] > _config.timeoutSeconds) {
      SkipTaskLocked(index, AsyncTaskStatus::TimedOut);
    }
  }
}

std::vector<AsyncTaskRunner::TaskInfo> AsyncTaskRunner::GetTaskInfos() {
  std::vector<TaskInfo> infos;
  if (!_running) {
    return infos;
  }
  infos.reserve(static_cast<size_t>(_total));
  double const nowSec = SecondsSince(_batchStart);
  std::lock_guard<std::mutex> lock(_statusMutex);
  for (int index = 0; index < _total; ++index) {
    auto const i = static_cast<size_t>(index);
    double const startSec = _startSec[i];
    double const endSec = _endSec[i];
    infos.push_back(
        TaskInfo{
            _config.tasks[i].label,
            _statuses[i],
            startSec < 0.0 ? 0.0 : (endSec >= 0.0 ? endSec : nowSec) - startSec,
            _statuses[i] == AsyncTaskStatus::Running && _skipRequested[i].load(),
            _blockedBy[i]});
  }
  return infos;
}

bool AsyncTaskRunner::IsRunning() const {
  return _running;
}

bool AsyncTaskRunner::ShowModalWindow() {
  if (!_running) {
    return false;
  }
  TimeOutTasks();

  std::string const& title = _config.title;
  // Keep the modal asserted for the whole batch. The mouse click that starts a batch (e.g. the
  // double-click or drag-release that opens an editor and kicks off tessellation) is processed by
  // ImGui on the same frame and dismisses a just-opened popup, so re-open it whenever it is not
  // currently shown rather than relying on a single OpenPopup.
  if (_openModal || !ImGui::IsPopupOpen(title.c_str())) {
    ImGui::OpenPopup(title.c_str());
    _openModal = false;
  }
  ImVec2 const center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  ImGuiWindowFlags const windowFlags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize;
  // Match the app's window background instead of the (darker) popup background.
  ImGui::PushStyleColor(ImGuiCol_PopupBg, ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
  bool const modalVisible = ImGui::BeginPopupModal(title.c_str(), nullptr, windowFlags);
  ImGui::PopStyleColor();
  if (modalVisible) {
    int const completed = _completed.load();
    float const fraction =
        _total > 0 ? static_cast<float>(completed) / static_cast<float>(_total) : 1.0f;
    std::array<char, 32> overlay{};
    std::snprintf(overlay.data(), overlay.size(), "%d / %d", completed, _total);
    // Sized in font heights so the table keeps its proportions at any display scale.
    float const width = 36.0f * ImGui::GetFontSize();
    ImGui::ProgressBar(fraction, ImVec2(width, 0.0f), overlay.data());
    if (_config.timeoutSeconds > 0.0) {
      ImGui::TextDisabled(
          "Processing %d items (tasks time out after %g s)", _total, _config.timeoutSeconds);
    } else {
      ImGui::TextDisabled("Processing %d items", _total);
    }
    ImGui::Spacing();
    ShowTaskTable(width, 15.0f * ImGui::GetFontSize());

    // Cancel sits below the task table, pinned bottom-left like the importer dialog's footer
    // Cancel. The table is a fixed height and scrolls internally, so this button stays visible
    // without scrolling no matter how long the batch is. It aborts the running task(s) via
    // Config::abortTask and skips any not-yet-started tasks; already-finished tasks keep their
    // results. The batch then completes through the normal path (the aborted worker unblocks),
    // closing the modal and returning control.
    ImGui::Spacing();
    ImGui::Separator();
    bool const cancelling = _cancelRequested.load();
    ImGui::BeginDisabled(cancelling || _workDone.load());
    if (ImGui::Button(cancelling ? "Cancelling..." : "Cancel", ImVec2(120.0f, 0.0f))) {
      MOCHI_LOG("Async batch '%s' cancellation requested by user", _config.title.c_str());
      std::lock_guard<std::mutex> lock(_statusMutex);
      _cancelRequested.store(true);
      for (int index = 0; index < _total; ++index) {
        if (_statuses[static_cast<size_t>(index)] == AsyncTaskStatus::Running &&
            _config.abortTask) {
          _config.abortTask(_workers[static_cast<size_t>(index)]);
        }
      }
    }
    ImGui::EndDisabled();

    if (_workDone.load()) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  // Finalize regardless of whether the modal is visible. A popup that never opened (or that was
  // dismissed by the same click that started the batch) must not leave the batch stuck "running"
  // with its onComplete unfired -- the completion logic does not depend on the modal being drawn.
  if (_workDone.load()) {
    JoinWorkers();
    bool const allSucceeded = _succeeded.load() == _total;
    MOCHI_LOG(
        "Async batch '%s' finished: %d/%d succeeded",
        _config.title.c_str(),
        _succeeded.load(),
        _total);
    // Move onComplete out before clearing state so it can safely start a new batch (e.g. a chained
    // import) without clobbering live members.
    auto onComplete = std::move(_config.onComplete);
    _running = false;
    _config = {};
    if (onComplete) {
      onComplete(allSucceeded);
    }
    return false;
  }
  return true;
}

void AsyncTaskRunner::ShowTaskTable(float width, float height) {
  // Sticky table of every task: name, color-coded status (green = done, red = failed or timed out,
  // yellow = running, dimmed = pending, grey = skipped or cancelled), elapsed time, and a Skip
  // button while it can still be skipped. Sortable; defaults to Status ascending, which groups rows
  // in AsyncTaskStatus order with the longest-running task first within each group.
  ImVec4 const green(0.40f, 0.80f, 0.45f, 1.0f);
  ImVec4 const red(0.90f, 0.45f, 0.45f, 1.0f);
  ImVec4 const yellow(0.95f, 0.85f, 0.45f, 1.0f);
  ImVec4 const grey(0.60f, 0.60f, 0.60f, 1.0f);
  std::vector<TaskInfo> const tasks = GetTaskInfos();
  std::vector<int> rows(tasks.size());
  std::iota(rows.begin(), rows.end(), 0);

  auto const tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
      ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_Sortable;
  if (!ImGui::BeginTable("##AsyncTaskTable", 4, tableFlags, ImVec2(width, height))) {
    return;
  }
  ImGui::TableSetupColumn("Task", ImGuiTableColumnFlags_WidthStretch, 5.0f);
  ImGui::TableSetupColumn(
      "Status",
      ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_DefaultSort |
          ImGuiTableColumnFlags_PreferSortAscending,
      1.6f);
  ImGui::TableSetupColumn(
      "Time",
      ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_PreferSortDescending,
      1.2f);
  ImGui::TableSetupColumn(
      "##Skip",
      ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoSort,
      ImGui::CalcTextSize("Skip").x + 2.0f * ImGui::GetStyle().FramePadding.x);
  ImGui::TableSetupScrollFreeze(0, 1);
  ImGui::TableHeadersRow();

  // Re-sort every frame: running rows' elapsed grows continuously, so the
  // order is never stable (ImGui's SpecsDirty flag alone would not suffice).
  if (ImGuiTableSortSpecs const* sortSpecs = ImGui::TableGetSortSpecs();
      sortSpecs != nullptr && sortSpecs->SpecsCount > 0) {
    ImGuiTableColumnSortSpecs const& spec = sortSpecs->Specs[0];
    bool const ascending = spec.SortDirection == ImGuiSortDirection_Ascending;
    std::ranges::sort(rows, [&](int ia, int ib) {
      TaskInfo const& a = tasks[static_cast<size_t>(ia)];
      TaskInfo const& b = tasks[static_cast<size_t>(ib)];
      int primary = 0;
      switch (spec.ColumnIndex) {
        case 0:
          primary = a.label.compare(b.label);
          break;
        case 1:
          primary = static_cast<int>(a.status) - static_cast<int>(b.status);
          break;
        default:
          primary = a.elapsedSeconds < b.elapsedSeconds
              ? -1
              : (a.elapsedSeconds > b.elapsedSeconds ? 1 : 0);
          break;
      }
      if (!ascending) {
        primary = -primary;
      }
      if (primary != 0) {
        return primary < 0;
      }
      // Within a group (e.g. equal status) show the longest-running first;
      // break remaining ties by index so the order is stable frame-to-frame.
      if (a.elapsedSeconds != b.elapsedSeconds) {
        return a.elapsedSeconds > b.elapsedSeconds;
      }
      return ia < ib;
    });
  }

  std::optional<int> skip;
  for (int const index : rows) {
    TaskInfo const& row = tasks[static_cast<size_t>(index)];
    ImGui::TableNextRow();
    ImGui::PushID(index);
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(row.label.c_str());
    ImGui::TableNextColumn();
    switch (row.status) {
      case AsyncTaskStatus::Pending:
        ImGui::TextDisabled("%s", "Pending");
        break;
      case AsyncTaskStatus::Running:
        ImGui::TextColored(yellow, "%s", row.skipping ? "Stopping" : "Running");
        break;
      case AsyncTaskStatus::Succeeded:
        ImGui::TextColored(green, "%s", "Done");
        break;
      case AsyncTaskStatus::Failed:
        ImGui::TextColored(red, "%s", "Failed");
        break;
      case AsyncTaskStatus::TimedOut:
        ImGui::TextColored(red, "%s", "Timed Out");
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip(
              "Ran longer than the %g s task timeout (Settings > Tasks).", _config.timeoutSeconds);
        }
        break;
      case AsyncTaskStatus::Skipped:
        ImGui::TextColored(grey, "%s", "Skipped");
        if (row.blockedBy >= 0 && ImGui::IsItemHovered()) {
          ImGui::SetTooltip(
              "Not run: it needs \"%s\", which did not succeed.",
              _config.tasks[static_cast<size_t>(row.blockedBy)].label.c_str());
        }
        break;
      case AsyncTaskStatus::Cancelled:
        ImGui::TextColored(grey, "%s", "Cancelled");
        break;
    }
    ImGui::TableNextColumn();
    // Tasks that never started leave their time cell blank.
    if (row.elapsedSeconds > 0.0) {
      ImGui::Text("%.2f s", row.elapsedSeconds);
    }
    ImGui::TableNextColumn();
    bool const skippable =
        (row.status == AsyncTaskStatus::Pending || row.status == AsyncTaskStatus::Running) &&
        !row.skipping && !_cancelRequested.load();
    if (skippable && ImGui::SmallButton("Skip")) {
      skip = index;
    }
    if (skippable && ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Skip this task, and the tasks that need its result.");
    }
    ImGui::PopID();
  }
  ImGui::EndTable();

  if (skip.has_value()) {
    std::lock_guard<std::mutex> lock(_statusMutex);
    SkipTaskLocked(*skip, AsyncTaskStatus::Skipped);
  }
}

void AsyncTaskRunner::JoinWorkers() {
  if (_coordinator.joinable()) {
    _coordinator.join();
  }
}

BackgroundTaskQueue::BackgroundTaskQueue() {
  _worker = std::thread([this] { WorkerLoop(); });
}

BackgroundTaskQueue::~BackgroundTaskQueue() {
  {
    std::lock_guard<std::mutex> const lock(_mutex);
    _stop = true;
  }
  _cv.notify_all();
  // Waits for the in-flight job's blocking work (e.g. the Hausdorff CLI) to finish so it is not
  // left orphaned when the owner is torn down. Queued jobs are dropped by the loop's stop check.
  // (Killing the running subprocess outright to avoid this wait is future work; it needs a process
  // handle the mesh CLI does not currently expose.)
  if (_worker.joinable()) {
    _worker.join();
  }
}

void BackgroundTaskQueue::Enqueue(uint64_t key, Work work) {
  auto cancelled = std::make_shared<std::atomic<bool>>(false);
  {
    std::lock_guard<std::mutex> const lock(_mutex);
    // Supersede an earlier request for the same key: drop it if still queued, or flag it to discard
    // its result if it is the one currently running.
    for (auto it = _queue.begin(); it != _queue.end();) {
      if (it->key == key) {
        it = _queue.erase(it);
      } else {
        ++it;
      }
    }
    if (_runningKey == key && _runningCancel) {
      _runningCancel->store(true);
    }
    _queue.push_back(Job{key, std::move(work), std::move(cancelled)});
  }
  _cv.notify_one();
}

void BackgroundTaskQueue::Cancel(uint64_t key) {
  std::lock_guard<std::mutex> const lock(_mutex);
  for (auto it = _queue.begin(); it != _queue.end();) {
    if (key == 0 || it->key == key) {
      it = _queue.erase(it);
    } else {
      ++it;
    }
  }
  if (_runningCancel && (key == 0 || _runningKey == key)) {
    _runningCancel->store(true);
  }
}

void BackgroundTaskQueue::WorkerLoop() {
  for (;;) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(_mutex);
      _cv.wait(lock, [this] { return _stop || !_queue.empty(); });
      if (_stop) {
        return; // teardown: drop any still-queued jobs (the in-flight one, if any, already ran)
      }
      job = std::move(_queue.front());
      _queue.pop_front();
      _runningCancel = job.cancelled;
      _runningKey = job.key;
    }
    if (!job.cancelled->load() && job.work) {
      job.work(*job.cancelled);
    }
    {
      std::lock_guard<std::mutex> const lock(_mutex);
      _runningCancel.reset();
      _runningKey = 0;
    }
  }
}

} // namespace superdex::studio
