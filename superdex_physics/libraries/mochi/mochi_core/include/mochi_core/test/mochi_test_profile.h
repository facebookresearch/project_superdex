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

#include <mochi_core/utils/profile.h>

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string_view>
#include <thread>
#include <vector>

namespace mochi::test {

#if MOCHI_PROFILE_ENABLE

/**
  MochiProfileListener receives calls from the Google Test framework so that it can automatically
  profile every test. The test names will appear in the tool as if they were functions. The source
  location displayed in the tool will be the location of the TEST macro, not the location of this
  listener.

  To use enable this feature, call InitializeUnitTestProfiler (argc, argv) from your main function,
  before you call RUN_ALL_TESTS().

  Use the command line argument "--profile" to make sure you don't miss anything. It will wait for
  the profiler to connect before continuing with the tests.
*/
class MochiProfileListener : public testing::EmptyTestEventListener {
 public:
  explicit MochiProfileListener(bool waitForConnect) : _shouldWaitForConnect(waitForConnect) {}

  // TestEventListener API:
  void OnTestProgramStart(testing::UnitTest const& unitTest) override;
  void OnTestCaseStart(testing::TestCase const& testCase) override;
  void OnTestStart(testing::TestInfo const& testInfo) override;
  void OnTestEnd(testing::TestInfo const& testInfo) override;
  void OnTestCaseEnd(testing::TestCase const& testCase) override;
  void OnTestProgramEnd(testing::UnitTest const& /*unitTest*/) override;

 private:
  void ZoneBegin(char const* name, char const* file, int line);
  void ZoneEnd();

  std::vector<std::unique_ptr<tracy::SourceLocationData>> _srcLocations;
  bool _shouldWaitForConnect = false;
};

/***********************************************************************************************
  Inlines
*/

inline void MochiProfileListener::OnTestProgramStart(testing::UnitTest const&) {
  // This function is called once before any tests run. Initialize the profiler now. This will also
  // prevent each instance of MochiPhysics from attempting to initialize it redundantly.
  ProfilerInitialize();

  // Optionally wait for the profiler to connect.
  if (_shouldWaitForConnect) {
    printf("Waiting for profiler to connect...\n");
    while (!ProfilerIsConnected()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  // Create a zone to span all of the tests. This makes the tool easier to use because you can
  // select it and list all the child zones instead of strubbing through the timeline to find them.
  ZoneBegin("RUN_ALL_TESTS", "main.cpp", 0);
}

inline void MochiProfileListener::OnTestCaseStart(testing::TestCase const& testCase) {
  // This funciton is called before each "test case". The test case is named by the first parameter
  // to the TEST or TEST_F macro. The test functions that belong to a single test case will be
  // executed back-to-back. Thus, this serves as an overall label for the collection of tests. In
  // the tool, it should appear as a single zone and the individual test functions should appear as
  // child zones (as if they were functions called by this one parent function).
  auto const* firstTest =
      testCase.GetTestInfo(0); // Use the first TEST macro to report source location.
  ZoneBegin(testCase.name(), firstTest->file(), firstTest->line());
}

inline void MochiProfileListener::OnTestStart(testing::TestInfo const& testInfo) {
  // This function is called before each test. The test is named by the second parameter to the TEST
  // or TEST_F macro. In the tool, this test should appear as a zone nested under the "test case".
  ZoneBegin(testInfo.name(), testInfo.file(), testInfo.line());
}

inline void MochiProfileListener::OnTestEnd(testing::TestInfo const&) {
  // End the test zone
  ZoneEnd();
}

inline void MochiProfileListener::OnTestCaseEnd(testing::TestCase const&) {
  // End the test case zone
  ZoneEnd();
}

inline void MochiProfileListener::OnTestProgramEnd(testing::UnitTest const&) {
  // End the RUN_ALL_TESTS zone
  ZoneEnd();

  // This function is called after the last test has completed. We will now shut down the profiler
  // to ensure that it flushes all data to the tool before we destroy any of our strings which were
  // originally passed by char*.
  ProfilerShutdown();
}

inline void MochiProfileListener::ZoneBegin(char const* name, char const* file, int line) {
#if MOCHI_USE_TRACY
  // Make it look like the zone was located in the named test function.
  tracy::SourceLocationData srcLoc{name, name, file, static_cast<uint32_t>(line), 0};

  // Make a copy of the SourceLocationData so it has a stable address. This is required
  // by Tracy because they simply store the address (location data is normally a static constexpr).
  _srcLocations.emplace_back(std::make_unique<tracy::SourceLocationData>(srcLoc));

  // Now, begin a zone just like tracy::ScopedZone (see TracyScoped.hpp).
  // Unlike tracy::ScopedZone, this zone will continue until we end it manually.
  using namespace tracy;
  TracyQueuePrepare(QueueType::ZoneBegin);
  MemWrite(&item->zoneBegin.time, tracy::Profiler::GetTime());
  MemWrite(&item->zoneBegin.srcloc, reinterpret_cast<uint64_t>(_srcLocations.back().get()));
  TracyQueueCommit(zoneBeginThread);
#else
#error TODO: Implement this feature for your profiler, or disable it.
#endif
}

inline void MochiProfileListener::ZoneEnd() {
#if MOCHI_USE_TRACY
  using namespace tracy;
  TracyQueuePrepare(QueueType::ZoneEnd);
  MemWrite(&item->zoneEnd.time, tracy::Profiler::GetTime());
  TracyQueueCommit(zoneEndThread);
#else
#error TODO: Implement this feature for your profiler, or disable it.
#endif
}

#endif // MOCHI_PROFILE_ENABLE

/***********************************************************************************************
  InitializeUnitTestProfiler
*/

inline bool HasArgument(int argc, char** argv, std::string_view search) {
  for (int i = 0; i < argc; ++i) {
    if (0 == strncmp(argv[i], search.data(), search.size())) {
      return true;
    }
  }
  return false;
}

inline void InitializeUnitTestProfiler(int argc, char** argv) {
  bool waitForConnect = HasArgument(argc, argv, "--profile");
#if MOCHI_PROFILE_ENABLE
  // Create an instance of MochiProfileListener and register it with gtest. They will take ownership
  // of the pointer.
  testing::UnitTest::GetInstance()->listeners().Append(
      new mochi::test::MochiProfileListener(waitForConnect));
#else
  if (waitForConnect) {
    printf("Profiler not supported in this build. Ignoring argument --profile.\n");
  }
#endif
}

} // namespace mochi::test
