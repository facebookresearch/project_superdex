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

  {
    std::lock_guard<std::mutex> lock(_statusMutex);
    _statuses.assign(static_cast<size_t>(_total), AsyncTaskStatus::Pending);
    _startSec.assign(static_cast<size_t>(_total), -1.0);
    _endSec.assign(static_cast<size_t>(_total), -1.0);
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
    auto const runTask = [this](int index) {
      AsyncTask const& task = _config.tasks[static_cast<size_t>(index)];
      // Skip tasks that have not started once the batch is cancelled, so a cancel drains the queue
      // without running further work. (Tasks already running are aborted via Config::onCancel.)
      if (_cancelRequested.load()) {
        {
          std::lock_guard<std::mutex> lock(_statusMutex);
          _statuses[static_cast<size_t>(index)] = AsyncTaskStatus::Cancelled;
        }
        MOCHI_LOG("Async task cancelled (not started): %s", task.label.c_str());
        _completed.fetch_add(1);
        return;
      }
      {
        std::lock_guard<std::mutex> lock(_statusMutex);
        _statuses[static_cast<size_t>(index)] = AsyncTaskStatus::Running;
        _startSec[static_cast<size_t>(index)] = SecondsSince(_batchStart);
      }
      MOCHI_LOG("Async task started: %s", task.label.c_str());
      AsyncCancelToken const token(_cancelRequested);
      bool const ok = task.work ? task.work(token) : false;
      // A task that returned success completed its work and is reported as such even if a cancel
      // was requested while it ran: atomic tasks (e.g. an SDF bake) cannot be interrupted
      // mid-flight, so a cancel landing during them does not undo the finished result. Only a task
      // that did not complete (returned false) while a cancel was pending is reported as cancelled
      // -- e.g. its helper was killed via Config::onCancel; any other incomplete task is a genuine
      // failure.
      bool const cancelled = !ok && _cancelRequested.load();
      double elapsed = 0.0;
      {
        std::lock_guard<std::mutex> lock(_statusMutex);
        _statuses[static_cast<size_t>(index)] = ok
            ? AsyncTaskStatus::Succeeded
            : (cancelled ? AsyncTaskStatus::Cancelled : AsyncTaskStatus::Failed);
        _endSec[static_cast<size_t>(index)] = SecondsSince(_batchStart);
        elapsed = _endSec[static_cast<size_t>(index)] - _startSec[static_cast<size_t>(index)];
      }
      if (ok) {
        MOCHI_LOG("Async task succeeded (%.2fs): %s", elapsed, task.label.c_str());
        _succeeded.fetch_add(1);
      } else if (cancelled) {
        MOCHI_LOG("Async task cancelled (%.2fs): %s", elapsed, task.label.c_str());
      } else {
        MOCHI_LOG_WARNING("Async task FAILED (%.2fs): %s", elapsed, task.label.c_str());
      }
      _completed.fetch_add(1);
    };
    {
      // BindThisThread() ensures that the mesh processing utilities will be able to take advantage
      // of concurrency within the mochi_physics library.
      _config.context->BindThisThread();
      MOCHI_DEFER(_config.context->UnbindThisThread());

      // TODO: Schedule these jobs in parallel if (and only if) !_config.serial.
      for (int index = 0; index < _total; ++index) {
        runTask(index);
      }
    }

    _workDone.store(true);
  });
}

bool AsyncTaskRunner::IsRunning() const {
  return _running;
}

bool AsyncTaskRunner::ShowModalWindow() {
  if (!_running) {
    return false;
  }

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
    float const width = 480.0f;
    ImGui::ProgressBar(fraction, ImVec2(width, 0.0f), overlay.data());
    ImGui::TextDisabled("Processing %d items", _total);
    ImGui::Spacing();

    // Sticky table of every task: name, color-coded status (green = done, red = failed,
    // yellow = running, dimmed = pending, grey = cancelled), and elapsed time. Sortable; defaults
    // to Status ascending, which groups rows as {Pending, Running, Succeeded, Failed, Cancelled}
    // with the longest-running task first within each group.
    ImVec4 const green(0.40f, 0.80f, 0.45f, 1.0f);
    ImVec4 const red(0.90f, 0.45f, 0.45f, 1.0f);
    ImVec4 const yellow(0.95f, 0.85f, 0.45f, 1.0f);
    ImVec4 const grey(0.60f, 0.60f, 0.60f, 1.0f);
    double const nowSec = SecondsSince(_batchStart);

    // Snapshot task state under the lock; draw/sort afterwards. _config.tasks is
    // immutable during a batch, so labels are read without the lock below. Pending
    // tasks have not started, so their elapsed time is 0.
    struct Row {
      int index;
      double elapsed;
      AsyncTaskStatus status;
    };
    std::vector<Row> rows;
    rows.reserve(static_cast<size_t>(_total));
    {
      std::lock_guard<std::mutex> lock(_statusMutex);
      for (int i = 0; i < _total; ++i) {
        double const startSec = _startSec[static_cast<size_t>(i)];
        double const endSec = _endSec[static_cast<size_t>(i)];
        double const elapsed = startSec < 0.0 ? 0.0 : (endSec >= 0.0 ? endSec : nowSec) - startSec;
        rows.push_back(Row{i, elapsed, _statuses[static_cast<size_t>(i)]});
      }
    }

    auto const tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
        ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_Sortable;
    if (ImGui::BeginTable("##AsyncTaskTable", 3, tableFlags, ImVec2(width, 240.0f))) {
      ImGui::TableSetupColumn("Task", ImGuiTableColumnFlags_WidthStretch, 5.0f);
      ImGui::TableSetupColumn(
          "Status",
          ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_DefaultSort |
              ImGuiTableColumnFlags_PreferSortAscending,
          1.5f);
      ImGui::TableSetupColumn(
          "Time",
          ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_PreferSortDescending,
          1.0f);
      ImGui::TableSetupScrollFreeze(0, 1);
      ImGui::TableHeadersRow();

      // Re-sort every frame: running rows' elapsed grows continuously, so the
      // order is never stable (ImGui's SpecsDirty flag alone would not suffice).
      if (ImGuiTableSortSpecs const* sortSpecs = ImGui::TableGetSortSpecs();
          sortSpecs != nullptr && sortSpecs->SpecsCount > 0) {
        ImGuiTableColumnSortSpecs const& spec = sortSpecs->Specs[0];
        bool const ascending = spec.SortDirection == ImGuiSortDirection_Ascending;
        std::sort(rows.begin(), rows.end(), [&](Row const& a, Row const& b) {
          int primary = 0;
          switch (spec.ColumnIndex) {
            case 0:
              primary = _config.tasks[static_cast<size_t>(a.index)].label.compare(
                  _config.tasks[static_cast<size_t>(b.index)].label);
              break;
            case 1:
              // Status order follows the enum: Pending, Running, Succeeded, Failed.
              primary = static_cast<int>(a.status) - static_cast<int>(b.status);
              break;
            default:
              primary = a.elapsed < b.elapsed ? -1 : (a.elapsed > b.elapsed ? 1 : 0);
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
          if (a.elapsed != b.elapsed) {
            return a.elapsed > b.elapsed;
          }
          return a.index < b.index;
        });
      }

      for (Row const& row : rows) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(_config.tasks[static_cast<size_t>(row.index)].label.c_str());
        ImGui::TableNextColumn();
        switch (row.status) {
          case AsyncTaskStatus::Pending:
            ImGui::TextDisabled("%s", "Pending");
            break;
          case AsyncTaskStatus::Running:
            ImGui::TextColored(yellow, "%s", "Running");
            break;
          case AsyncTaskStatus::Succeeded:
            ImGui::TextColored(green, "%s", "Done");
            break;
          case AsyncTaskStatus::Failed:
            ImGui::TextColored(red, "%s", "Failed");
            break;
          case AsyncTaskStatus::Cancelled:
            ImGui::TextColored(grey, "%s", "Cancelled");
            break;
        }
        ImGui::TableNextColumn();
        // Pending tasks have not started, so leave their time cell blank.
        if (row.status != AsyncTaskStatus::Pending) {
          ImGui::Text("%.2f s", row.elapsed);
        }
      }
      ImGui::EndTable();
    }

    // Cancel sits below the task table, pinned bottom-left like the importer dialog's footer
    // Cancel. The table is a fixed height (240 px) and scrolls internally, so this button stays
    // visible without scrolling no matter how long the batch is. It aborts the running task(s) via
    // Config::onCancel and skips any not-yet-started tasks; already-finished tasks keep their
    // results. The batch then completes through the normal path (the aborted worker unblocks),
    // closing the modal and returning control.
    ImGui::Spacing();
    ImGui::Separator();
    bool const cancelling = _cancelRequested.load();
    ImGui::BeginDisabled(cancelling || _workDone.load());
    if (ImGui::Button(cancelling ? "Cancelling..." : "Cancel", ImVec2(120.0f, 0.0f))) {
      _cancelRequested.store(true);
      MOCHI_LOG("Async batch '%s' cancellation requested by user", _config.title.c_str());
      if (_config.onCancel) {
        _config.onCancel();
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
