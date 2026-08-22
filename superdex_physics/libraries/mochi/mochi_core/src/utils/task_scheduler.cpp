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

#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/string_utils.h>
#include <mochi_core/utils/task_scheduler.h>

#include <marl/conditionvariable.h>
#include <marl/scheduler.h>
#include <marl/waitgroup.h>

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if MOCHI_PLATFORM_WINDOWS
#include <winnt.h>
#elif MOCHI_PLATFORM_LINUX
#include <sys/sysinfo.h>
#include <cstdio>
#include <unordered_set>
#endif

namespace mochi {

namespace {

// Should match marl::Scheduler::MaxWorkerThreads, which is not public.
constexpr int kMaxThreadsSupportedByMarl = 256;

int GetNumLogicalProcessors() {
  return std::thread::hardware_concurrency();
}

} // namespace

/**************************************************************************************************
  TaskScheduler
*/

thread_local TaskScheduler* TaskScheduler::s_currentScheduler = nullptr;
thread_local int TaskScheduler::s_currentSchedulerRefCount = 0;
thread_local bool TaskScheduler::s_marlAlreadyBound = false;
thread_local bool TaskScheduler::s_isCurrentThreadAWorker = false;
thread_local int TaskScheduler::s_localSingleThreadedCount = 0;

TaskScheduler::Config::Config() : numThreads(TaskScheduler::GetNumSupportedLogicalProcessors()) {}

TaskScheduler::TaskScheduler(int numThreads) : TaskScheduler(Config{numThreads}) {}

TaskScheduler::TaskScheduler(Config const& config) {
  if (config.numThreads > kMaxThreadsSupportedByMarl) {
    MOCHI_LOG_WARNING(
        "The requested number of threads (%d) is larger than the maximum supported (%d) and will be clamped to %d.",
        config.numThreads,
        kMaxThreadsSupportedByMarl,
        kMaxThreadsSupportedByMarl);
  }
  int const numLogicalProcessors = GetNumLogicalProcessors();
  if ((numLogicalProcessors > 0) && (config.numThreads > numLogicalProcessors)) {
    MOCHI_LOG_WARNING(
        "The requested number of threads (%d) is larger than the maximum number of threads that can run concurrently on the machine (%d). This may degrade simulation performance.",
        config.numThreads,
        numLogicalProcessors);
  }
  marl::Scheduler::Config marlConfig;
  marlConfig.fiberStackSize = config.fiberStackSize;
  marlConfig.workerThread.count = Clamp(config.numThreads, 0, kMaxThreadsSupportedByMarl);
  marlConfig.workerThread.initializer = [this](int workerId) {
    // Bind our TaskScheduler so that it can be accessed from within each worker thread
    MOCHI_ASSERT_VERBOSE(s_currentScheduler == nullptr);
    MOCHI_ASSERT_VERBOSE(s_currentSchedulerRefCount == 0);
    s_currentScheduler = this;
    s_isCurrentThreadAWorker = true;
    ++s_currentSchedulerRefCount;

    char threadName[32];
    snprintf(threadName, sizeof(threadName), "Mochi Worker %d", workerId);
    marl::Thread::setName(threadName);
  };

  _marl = new marl::Scheduler(marlConfig);
  BindThisThread();
}

TaskScheduler::~TaskScheduler() {
  UnbindThisThread();
  delete _marl;
}

// Static function
int TaskScheduler::GetNumSupportedLogicalProcessors() {
  return Min(kMaxThreadsSupportedByMarl, GetNumLogicalProcessors());
}

#if MOCHI_PLATFORM_LINUX

// Linux system files like "/proc/cpuinfo" are not normal files. We can't use mochi::ReadFileString
// because it expects to be able to seek to the end of the file to determine the file size up front.
// That doesn't work in this case. We have to read the pipe sequentially until we hit the end.
static std::string Linux_ReadSystemFile(std::string const& filename, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  FILE* file = fopen(filename.c_str(), "r");
  if (!file) {
    MOCHI_ERROR_SET(error, "Failed to open file for reading");
    return {};
  }
  MOCHI_DEFER(fclose(file));

  std::string content;
  char buffer[1024];
  while (fgets(buffer, sizeof(buffer), file) != nullptr) {
    content += buffer;
  }
  return content;
}

// Parse physical core count from "/proc/cpuinfo" using physical id + core id pairs
static int Linux_GetPhysicalCoresFromCpuinfo(Error& error) {
  MOCHI_ERROR_RETURN(error, -1);

  // Read the entire file into a string
  std::string contents = Linux_ReadSystemFile("/proc/cpuinfo", error);
  MOCHI_ERROR_IF(contents.empty(), error, "Empty cpuinfo");
  MOCHI_ERROR_RETURN(error, -1);

  // Split lines and trim whitespace
  auto lines = Split(contents, "\n");
  for (auto& line : lines) {
    line = Trim(line);
  }

  std::unordered_set<std::string> uniqueCores;
  std::string currentPhysicalId;
  std::string currentCoreId;

  // Parse line by line
  for (auto const& line : lines) {
    if (line.find("physical id") == 0) {
      size_t colonPos = line.find(':');
      if (colonPos != std::string::npos) {
        currentPhysicalId = Trim(line.substr(colonPos + 1));
      }
    } else if (line.find("core id") == 0) {
      size_t colonPos = line.find(':');
      if (colonPos != std::string::npos) {
        currentCoreId = Trim(line.substr(colonPos + 1));

        // When we have both physical id and core id, add to unique set
        if (!currentPhysicalId.empty() && !currentCoreId.empty()) {
          uniqueCores.insert(Format("%s:%s", currentPhysicalId.c_str(), currentCoreId.c_str()));
          currentPhysicalId = "";
          currentCoreId = "";
        }
      }
    }
  }

  return uniqueCores.empty() ? -1 : isize(uniqueCores);
}

// Try to get physical cores from CPU topology
static int Linux_GetPhysicalCoresFromCpuTopology(Error& error) {
  MOCHI_ERROR_RETURN(error, -1);
  std::unordered_set<std::string> uniqueCores;

  for (int cpu = 0; cpu < 1024; ++cpu) { // reasonable upper limit
    Error readError;
    std::string physicalId = Linux_ReadSystemFile(
        Format("/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu), readError);
    std::string coreId = Linux_ReadSystemFile(
        Format("/sys/devices/system/cpu/cpu%d/topology/core_id", cpu), readError);
    if (!readError.IsOK()) {
      break;
    }

    physicalId = Trim(physicalId);
    coreId = Trim(coreId);

    if (!physicalId.empty() && !coreId.empty()) {
      uniqueCores.insert(Format("%s:%s", physicalId.c_str(), coreId.c_str()));
    } else {
      // No more CPUs found
      break;
    }
  }

  MOCHI_ERROR_IF(uniqueCores.empty(), error, "Failed to get physical core count from sysfs");
  return uniqueCores.empty() ? -1 : isize(uniqueCores);
}
#endif // MOCHI_PLATFORM_LINUX

// Static function
int TaskScheduler::GetNumSupportedPhysicalProcessors() {
  static int const numCores = []() {
#if MOCHI_PLATFORM_WINDOWS
    // Call WIN32 API to get the buffer size IN BYTES required to receive processor info.
    DWORD bufferLen = 0;
    ::GetLogicalProcessorInformation(nullptr, &bufferLen);
    MOCHI_ASSERT(
        bufferLen % sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) == 0,
        "Buffer size should be a multiple of sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)");

    // Call WIN32 API again to get the processor information
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> processorInfo(
        bufferLen / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (::GetLogicalProcessorInformation(processorInfo.data(), &bufferLen)) {
      int reportedCoreCount = 0;
      for (auto const& proc : processorInfo) {
        if (proc.Relationship == RelationProcessorCore) {
          reportedCoreCount++;
        }
      }
      MOCHI_ASSERT(reportedCoreCount <= static_cast<int>(std::thread::hardware_concurrency()));
      return reportedCoreCount;
    } else {
      MOCHI_ASSERT(false, "GetLogicalProcessorInformation failed");
    }
#elif MOCHI_PLATFORM_LINUX
    // Linux environments vary, so we have two ways of possibly determining the physical processor
    // count. First, try to parse the system file, "/proc/cpuinfo".
    Error error;
    int physicalCores = Linux_GetPhysicalCoresFromCpuinfo(error);
    if (error.IsOK()) {
      MOCHI_ASSERT(physicalCores > 0);
      return physicalCores;
    }

    // If that fails, then try this fallback method
    error = Error{}; // clear error
    physicalCores = Linux_GetPhysicalCoresFromCpuTopology(error);
    if (error.IsOK()) {
      MOCHI_ASSERT(physicalCores > 0);
      return physicalCores;
    }

    // If all methods fail, log a warning and fall back to logical processor count
    MOCHI_LOG_WARNING(
        "Failed to determine number of physical vs logical processors. "
        "The same value (%d processors) will be returned for both.",
        GetNumLogicalProcessors());
#else
  // APIs may exist to differentiate between logical and physical processors on other platforms.
  // Feel free to add them here. At the time of writing, this was a non-issue for ARM-based Mac
  // and Android devices, which only run one logical thread per physical core.
#endif
    return GetNumLogicalProcessors();
  }();
  return Min(numCores, kMaxThreadsSupportedByMarl);
}

// Static function
TaskScheduler* TaskScheduler::TryGet() {
  return s_currentScheduler;
}

// Static function
TaskScheduler& TaskScheduler::Get() {
  MOCHI_ASSERT_VERBOSE(
      s_currentScheduler != nullptr, "You must first call TaskScheduler::BindThisThread");
  return *s_currentScheduler;
}

// Static function
[[nodiscard]] bool TaskScheduler::IsCurrentThreadAWorker() {
  return s_isCurrentThreadAWorker;
}

[[nodiscard]] int TaskScheduler::GetNumThreads() const {
  return (_isGlobalSingleThreaded.load() || s_localSingleThreadedCount > 0)
      ? 0
      : _marl->config().workerThread.count;
}

[[nodiscard]] int TaskScheduler::GetNumOtherThreads() const {
  int numThreads = GetNumThreads();
  return Max(0, s_isCurrentThreadAWorker ? (numThreads - 1) : numThreads);
}

void TaskScheduler::BindThisThread() {
  // This is a reference counted mechanism because the TaskScheduler may be used by different
  // systems which do not know if the thread has already been bound. Furthermore, the thread-local
  // static variables will not be shared between Windows DLLs, but they will be shared on other
  // platforms like macos/ios/android/linux. Therefore, we allow the caller to call BindThisThread
  // even if it is already bound as long as they call UnbindThisThread an equal number of times
  // during shutdown.
  MOCHI_ASSERT_VERBOSE(
      (s_currentScheduler == nullptr) || (s_currentScheduler == this),
      "Attempting to bind more than one TaskScheduler to the same thread");
  MOCHI_ASSERT_VERBOSE(
      (marl::Scheduler::get() == nullptr) || (marl::Scheduler::get() == _marl),
      "Attempting to bind a TaskScheduler to a thread that is already bound to a marl::Scheduler");
  if (s_currentScheduler == this) {
    // This call was redundant, but that's OK. Just count the references.
    ++s_currentSchedulerRefCount;
  } else {
    // This was the first from this thread, or the first call from this DLL.
    // Set up the static thread-local state.
    s_currentScheduler = this;
    s_currentSchedulerRefCount = 1;
    s_localSingleThreadedCount = 0;
    s_marlAlreadyBound = (_marl->get() != nullptr);
    if (s_marlAlreadyBound) {
      // Marl was already bound even though s_currentScheduler was not set. This can happen when
      // TaskScheduler is statically linked into multiple Windows DLLs. Note that each DLL will
      // have its own copy of s_currentScheduler, but there is only one copy of the pointer stored
      // by Marl because marl is compiled into its own DLL.
    } else {
      _marl->bind();
    }
  }
}

void TaskScheduler::UnbindThisThread() {
  if (s_currentScheduler == this) {
    --s_currentSchedulerRefCount;
    if (s_currentSchedulerRefCount == 0) {
      s_localSingleThreadedCount = 0;
      s_currentScheduler = nullptr;
      if (!s_marlAlreadyBound) {
        _marl->unbind();
      }
    }
  }
}

void TaskScheduler::SetGlobalSingleThreadedMode(bool globalSingleThreaded) {
  _isGlobalSingleThreaded.store(globalSingleThreaded);
}

// Static function
void TaskScheduler::PushLocalSingleThreadedMode() {
  ++s_localSingleThreadedCount;
}

// Static function
void TaskScheduler::PopLocalSingleThreadedMode() {
  MOCHI_ASSERT(
      s_localSingleThreadedCount > 0,
      "PopLocalSingleThreadedMode must be preceded by PushLocalSingleThreadedMode.");
  --s_localSingleThreadedCount;
}

// Static function
[[nodiscard]] bool TaskScheduler::IsLocalSingleThreaded() {
  return s_localSingleThreadedCount > 0;
}

void TaskScheduler::AddTaskNoProfile(TaskFn&& fn, bool isSingleThreaded) {
  auto flags = isSingleThreaded ? marl::Task::Flags::SameThread : marl::Task::Flags::None;
  _marl->enqueue(marl::Task{std::move(fn), flags});
}

void TaskScheduler::AddTaskVerboseProfile(
    [[maybe_unused]] std::string_view debugNameStringLiteral,
    TaskFn&& fn) {
  auto flags = (GetNumThreads() == 0) ? marl::Task::Flags::SameThread : marl::Task::Flags::None;
  if (ProfilerIsConnected()) {
    MOCHI_PROFILE_SCOPE();
    uint64_t id = ++_profileTaskCounter; // Give each task a unique identifier
    // Warning: std::string_view is not necessarily null-terminated, but it is supposed to come from
    // a string literal in which case this is OK.
    std::string zoneName = Format("AddTask %s (%llx)", debugNameStringLiteral.data(), id);
    MOCHI_PROFILE_LABEL(zoneName);
    // "Replace "Add" with "Run"
    memcpy(zoneName.data(), "Run", 3);
    _marl->enqueue(
        marl::Task{
            [fn = std::move(fn), zoneName = std::move(zoneName)]() {
              MOCHI_PROFILE_SCOPE_N("RunTask");
              MOCHI_PROFILE_LABEL(zoneName);
              fn();
            },
            flags});
  } else {
    _marl->enqueue(marl::Task{std::move(fn), flags});
  }
}

void TaskScheduler::TryToWakeUpMoreWorkers(int numWorkers, TimeSpan dummyTaskDuration) {
  for (int i = 0; i < numWorkers; ++i) {
    AddTaskNoProfile(
        [dummyTaskDuration]() {
          Timer timer = {};
          while (ToNanoseconds(timer.GetElapsed()) < ToNanoseconds(dummyTaskDuration)) {
          }
        },
        false);
  }
}

int TaskScheduler::BatchEnqueueOnAvailableWorkers(
    TaskSemaphore sem,
    BatchTaskFn&& task,
    int minWorkers,
    int targetWorkers,
    bool includeSelf) {
  MOCHI_ASSERT_VERBOSE(
      minWorkers > 0 && targetWorkers >= minWorkers, "Inconsistent number of workers.");
  int const minOtherWorkers = includeSelf ? (minWorkers - 1) : minWorkers;
  if (minOtherWorkers > GetNumOtherThreads()) {
    // Early return. It also covers the single-threaded case.
    return 0;
  }

  sem.Add(targetWorkers);
  int const scheduledTasks = _marl->batchEnqueueOnAvailableWorkers(
      task,
      minWorkers,
      /* Handle single-threaded case */ GetNumThreads() == 0 ? 1 : targetWorkers,
      includeSelf);
  sem.Done(targetWorkers - scheduledTasks);
  return scheduledTasks;
}

/**************************************************************************************************
  TaskSemaphore
*/

namespace {
struct TaskSemaphoreDataImpl : public TaskSemaphore::Data {
  std::atomic<int> count = {0};
  marl::ConditionVariable cv{marl::Allocator::Default};
  marl::mutex mutex;
};
} // namespace

TaskSemaphore::TaskSemaphore(int initialCount) : _data(std::make_shared<TaskSemaphoreDataImpl>()) {
  auto* data = static_cast<TaskSemaphoreDataImpl*>(_data.get());
  data->count = initialCount;
}

void TaskSemaphore::Add(int count) const {
  MOCHI_ASSERT_VERBOSE(
      count >= 0, "TaskSemaphore::Add can only be used to increase the semaphore count.");
  auto* data = static_cast<TaskSemaphoreDataImpl*>(_data.get());
  data->count += count;
}

bool TaskSemaphore::Done(int numTasksDone) const {
  MOCHI_ASSERT_VERBOSE(
      numTasksDone >= 0, "TaskSemaphore::Done can only be used to decrease the semaphore count.");
  auto* data = static_cast<TaskSemaphoreDataImpl*>(_data.get());
  auto remaining = (numTasksDone == 1) ? --data->count : (data->count -= numTasksDone);
  if (remaining == 0) {
    marl::lock lock(data->mutex);
    data->cv.notify_all();
    return true;
  } else {
    MOCHI_ASSERT_VERBOSE(remaining > 0, "TaskSemaphore::Done() called too many times");
    return false;
  }
}

bool TaskSemaphore::IsDone() const {
  auto* data = static_cast<TaskSemaphoreDataImpl*>(_data.get());
  return (data->count.load() == 0);
}

void TaskSemaphore::Wait() const {
  auto* data = static_cast<TaskSemaphoreDataImpl*>(_data.get());
  if (data->count.load() == 0) {
    return; // Already done
  }

  // This profile scope is always enabled (if profiling in general is enabled). We need to be able
  // to see the "Wait" label in the profiler, or it will be very confusing when we see another task
  // start to run on the same thread.
  MOCHI_PROFILE_SCOPE();
  marl::lock lock(data->mutex);
  data->cv.wait(lock, [data] { return data->count == 0; });
}

bool TaskSemaphore::WaitFor(TimeSpan duration) const {
  auto* data = static_cast<TaskSemaphoreDataImpl*>(_data.get());
  if (data->count.load() == 0) {
    return true; // Already done
  }

  // This profile scope is always enabled (if profiling in general is enabled). We need to be able
  // to see the "Wait" label in the profiler, or it will be very confusing when we see another task
  // start to run on the same thread.
  MOCHI_PROFILE_SCOPE();
  marl::lock lock(data->mutex);
  return data->cv.wait_for(lock, duration, [data] { return data->count == 0; });
}

} // namespace mochi
