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

#include <gtest/gtest.h>

#include <mochi_core/async/executor.h>
#include <mochi_core/async/generator.h>
#include <mochi_core/async/root_task.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/task_scheduler.h>

#include <coroutine>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace mochi;

namespace {

Task<int> Triple(int n) {
  co_return 3 * n;
}

int MultiplyAdd(int factor, int n, int increment) {
  return factor * n + increment;
}

RootTask DestroyOwner(std::unique_ptr<RootTask>& owner, bool& resumedAfterReset) {
  owner.reset();
  resumedAfterReset = true;
  co_return;
}

struct MemberCoroutineObject {
  bool completed = false;

  RootTask Complete() {
    completed = true;
    co_return;
  }
};

RootTask VectorTest(int n) {
  Executor executor;
  std::vector<int> v;
  v.reserve(n);
  for (int i = 0; i < n; ++i) {
    v.push_back(n - 2 * i);
  }
  auto result = co_await executor.Execute(Triple, async::ForEach{v});
  DynamicArray<int> expectedResult;
  expectedResult.reserve(n);
  for (auto x : v) {
    expectedResult.push_back(3 * x);
  }
  EXPECT_EQ(result, expectedResult);

  auto result2 = co_await executor.Execute(MultiplyAdd, 4, async::ForEach{v}, -1);
  expectedResult.clear();
  for (auto x : v) {
    expectedResult.push_back(MultiplyAdd(4, x, -1));
  }
  EXPECT_EQ(result2, expectedResult);
}

void TestExecutor(bool localSingleThreaded) {
  TaskScheduler ts(Min(TaskScheduler::GetNumSupportedLogicalProcessors(), 8));
  ScopedLocalSingleThreadedMode const singleThreaded(localSingleThreaded);
  for (int n = 0; n < 300; n += 25) {
    VectorTest(n).RunAndWait(ts, "MainCoroutine");
  }
}

// The four functors below cover all four WorkGenerator::NextTask specializations
// (Task<T>, Task<void>, plain T, plain void). Each throws on input == 2 with a
// distinct message so the test can verify the original exception is what propagates,
// rather than e.g. a spurious std::bad_optional_access from the harvest loop.
Task<int> throwingTaskInt(int n) {
  if (n == 2) {
    throw std::runtime_error("task-int");
  }
  co_return n * 2;
}

Task<void> throwingTaskVoid(int n) {
  if (n == 2) {
    throw std::runtime_error("task-void");
  }
  co_return;
}

int throwingPlainInt(int n) {
  if (n == 2) {
    throw std::runtime_error("plain-int");
  }
  return n * 2;
}

void throwingPlainVoid(int n) {
  if (n == 2) {
    throw std::runtime_error("plain-void");
  }
}

RootTask exceptionTest(int which) {
  Executor executor;
  std::vector<int> v = {1, 2, 3, 4}; // Exactly one element triggers the throw.
  char const* expected = nullptr;
  std::string caughtMsg;
  try {
    switch (which) {
      case 0: {
        expected = "task-int";
        auto r = co_await executor.Execute(throwingTaskInt, async::ForEach{v});
        (void)r;
        break;
      }
      case 1:
        expected = "task-void";
        co_await executor.Execute(throwingTaskVoid, async::ForEach{v});
        break;
      case 2: {
        expected = "plain-int";
        auto r = co_await executor.Execute(throwingPlainInt, async::ForEach{v});
        (void)r;
        break;
      }
      case 3:
        expected = "plain-void";
        co_await executor.Execute(throwingPlainVoid, async::ForEach{v});
        break;
    }
    ADD_FAILURE() << "Expected exception for case " << which;
  } catch (std::runtime_error const& e) {
    caughtMsg = e.what();
  }
  EXPECT_STREQ(caughtMsg.c_str(), expected);
}

void TestExecutorExceptions() {
  TaskScheduler ts(Min(TaskScheduler::GetNumSupportedLogicalProcessors(), 8));
  for (int i = 0; i < 4; ++i) {
    exceptionTest(i).RunAndWait(ts, "MainCoroutine");
  }
}

// Every worker throws. This is the case the final_suspend-based death-count decrement
// is meant to make safe: even when no worker completes normally, each still reaches
// final_suspend(), so the count reaches zero and Execute() resumes instead of hanging.
RootTask allThrowTest(bool& finished) {
  Executor executor;
  std::vector<int> v(8, 2); // 2 triggers the throw in throwingTaskInt for every element.
  std::string caughtMsg;
  try {
    auto r = co_await executor.Execute(throwingTaskInt, async::ForEach{v});
    (void)r;
    ADD_FAILURE() << "Expected exception when all workers throw";
  } catch (std::runtime_error const& e) {
    caughtMsg = e.what();
  }
  EXPECT_EQ(caughtMsg, "task-int");
  finished = true;
}

void TestExecutorAllThrow() {
  TaskScheduler ts(Min(TaskScheduler::GetNumSupportedLogicalProcessors(), 8));
  bool finished = false;
  allThrowTest(finished).RunAndWait(ts, "MainCoroutine");
  EXPECT_TRUE(finished);
}

Generator<int> EvenOdd(auto&& range) {
  for (auto x : range) {
    if (x % 2 == 0) {
      co_yield x;
    }
  }
  for (auto x : range) {
    if (x % 2 != 0) {
      co_yield x;
    }
  }
}

Generator<MatrixView<float>> MatrixGenerator(float* space) {
  for (int i = 0; i < 4; ++i) {
    co_yield MatrixView<float>(space + 3 * i, 2 + i, 1);
  }
}

Generator<int> IntGenerator(int n) {
  for (int i = 0; i < n; ++i) {
    co_yield i;
  }
}

struct TestException : public std::exception {
  char const* what() const noexcept override {
    return "TestException";
  }
};

void DoThrow() {
  throw TestException{};
}

Task<int> ThrowingTask() {
  DoThrow();
  co_return 42;
}

Task<void> ThrowingVoidTask() {
  DoThrow();
  co_return;
}

RootTask HoldResource(std::shared_ptr<int>, bool shouldThrow, std::thread::id& bodyThread) {
  bodyThread = std::this_thread::get_id();
  if (shouldThrow) {
    DoThrow();
  }
  co_return;
}

RootTask awaitThrowingTask(bool& caughtException) {
  try {
    [[maybe_unused]] int result = co_await ThrowingTask();
  } catch (TestException const&) {
    caughtException = true;
  }
}

void TestTaskExceptionHandling() {
  TaskScheduler ts(Min<int>(std::thread::hardware_concurrency(), 8));

  // Test exception via GetValue() for Task<int>
  {
    auto task = ThrowingTask();
    task.Handle().resume();
    EXPECT_TRUE(task.Handle().done());
    EXPECT_THROW(task.GetValue(), TestException);
  }

  // Test exception via GetValue() for Task<void>
  {
    auto task = ThrowingVoidTask();
    task.Handle().resume();
    EXPECT_TRUE(task.Handle().done());
    EXPECT_THROW(task.GetValue(), TestException);
  }

  // Test exception propagation via co_await
  {
    bool caughtException = false;
    awaitThrowingTask(caughtException).RunAndWait(ts, "AwaitThrowingTask");
    EXPECT_TRUE(caughtException);
  }
}

Task<int> SimpleTask(int value) {
  co_return value * 2;
}

struct CallbackContext {
  bool called = false;
  int result = 0;
};

std::coroutine_handle<> callbackContinuation(void* arg) {
  auto* ctx = static_cast<CallbackContext*>(arg);
  ctx->called = true;
  return std::noop_coroutine();
}

void TestTaskCallbackContinuation() {
  // Test callback continuation for Task<int>
  {
    CallbackContext ctx;
    auto task = SimpleTask(21);
    task.SetContinuation(callbackContinuation, &ctx);
    task.Handle().resume();
    EXPECT_TRUE(task.Handle().done());
    EXPECT_TRUE(ctx.called);
    ctx.result = task.GetValue();
    EXPECT_EQ(ctx.result, 42);
  }

  // Test callback continuation for Task<void>
  {
    CallbackContext ctx;
    auto task = ThrowingVoidTask();
    task.SetContinuation(callbackContinuation, &ctx);
    task.Handle().resume();
    EXPECT_TRUE(task.Handle().done());
    EXPECT_TRUE(ctx.called);
  }
}

void TestGenerator() {
  std::vector<int> v;
  int N = 100;
  v.reserve(N);
  for (int i = 0; i < N; ++i) {
    v.push_back(i);
  }
  std::vector<int> result;
  result.reserve(N);
  for (auto i : EvenOdd(v)) {
    result.push_back(i);
  }
  std::vector<int> expectedResult;
  expectedResult.reserve(N);
  for (int i = 0; i < 100; ++i) {
    expectedResult.push_back(i < 50 ? 2 * i : 2 * (i - 50) + 1);
  }
  EXPECT_EQ(result, expectedResult);

  result.clear();
  {
    auto g = EvenOdd(v);
    while (g.Next()) {
      result.push_back(g.Value());
    }
  }
  EXPECT_EQ(result, expectedResult);

  // Test that the generator does not try to use operator= of its return type.
  std::vector<float> space(100);
  for (int i = 0; auto block : MatrixGenerator(space.data())) {
    EXPECT_EQ(&block(0, 0), space.data() + 3 * i);
    EXPECT_EQ(block.Rows(), 2 + i);
    EXPECT_EQ(block.Cols(), 1);
    ++i;
  }

  int count = 0;
  // Test that a non yielding generator does not do any iteration.
  for ([[maybe_unused]] auto _ : IntGenerator(0)) {
    ++count;
  }
  EXPECT_EQ(count, 0);
}

} // namespace

// In local single-threaded mode, the coroutine runs before RunAndWait's dispatch returns.
TEST(Coroutines, ScheduledCoroutineCanDestroyItsRootTask) {
  bool resumedAfterReset = false;
  std::unique_ptr<RootTask> owner;
  owner = std::make_unique<RootTask>(DestroyOwner(owner, resumedAfterReset));
  TaskScheduler scheduler(1);

  {
    ScopedLocalSingleThreadedMode const singleThreaded;
    owner->RunAndWait(scheduler);
  }

  EXPECT_EQ(owner, nullptr);
  EXPECT_TRUE(resumedAfterReset);
}

// RunAndWait destroys the frame on the thread that completes the coroutine, before it returns or
// rethrows, in every scheduler mode.
TEST(Coroutines, RootTaskRunAndWaitDestroysFrameOnCompletingThread) {
  for (int mode = 0; mode < 3; ++mode) {
    for (bool const shouldThrow : {false, true}) {
      TaskScheduler scheduler(1);
      scheduler.SetGlobalSingleThreadedMode(mode == 1);
      ScopedLocalSingleThreadedMode const singleThreaded(mode == 2);
      std::thread::id bodyThread;
      std::thread::id destroyThread;
      auto resource = std::shared_ptr<int>(new int(0), [&destroyThread](int const* value) {
        destroyThread = std::this_thread::get_id();
        delete value;
      });
      auto task = HoldResource(std::move(resource), shouldThrow, bodyThread);

      if (shouldThrow) {
        EXPECT_THROW(task.RunAndWait(scheduler), TestException);
      } else {
        task.RunAndWait(scheduler);
      }

      EXPECT_NE(destroyThread, std::thread::id{});
      EXPECT_EQ(destroyThread, bodyThread);
    }
  }
}

TEST(Coroutines, RootTaskRunAndWaitSupportsMemberAndCapturelessLambdaCoroutines) {
  TaskScheduler scheduler(1);
  MemberCoroutineObject object;
  bool lambdaCompleted = false;
  auto const completeLambda = [](bool& completed) -> RootTask {
    completed = true;
    co_return;
  };

  object.Complete().RunAndWait(scheduler);
  completeLambda(lambdaCompleted).RunAndWait(scheduler);

  EXPECT_TRUE(object.completed);
  EXPECT_TRUE(lambdaCompleted);
}

TEST(Coroutines, Execute) {
  TestExecutor(false);
  TestExecutor(true);
}

TEST(Coroutines, ExecuteExceptions) {
  TestExecutorExceptions();
}

TEST(Coroutines, ExecuteAllThrow) {
  TestExecutorAllThrow();
}

TEST(Coroutines, Generator) {
  TestGenerator();
}

TEST(Coroutines, TaskExceptionHandling) {
  TestTaskExceptionHandling();
}

TEST(Coroutines, TaskCallbackContinuation) {
  TestTaskCallbackContinuation();
}
