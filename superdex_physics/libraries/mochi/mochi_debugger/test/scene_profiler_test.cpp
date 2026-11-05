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

#include "mochi_debugger_test.h"

#include <mochi_core/test/wait_until.h>
#include <mochi_physics/src/mochi_scene.h>
#include <mochi_physics/src/mochi_scene_debugger.h>
#include <mochi_physics/src/mochi_scene_profiler.h>

#include <gtest/gtest.h>

using namespace mochi;
using namespace mochi::dbg;

namespace {

class SceneProfilerTest : public MochiDebuggerTest {
 protected:
  static bool IsProfilerEnabled(Scene* scene) {
    return profiler::IsEnabled(assert_cast<SceneImpl*>(scene)->GetRegistry());
  }

  static std::shared_ptr<SceneDebugger> GetDebugger(Scene* scene) {
    return assert_cast<SceneImpl*>(scene)->GetDebugger();
  }
};

} // namespace

TEST_F(SceneProfilerTest, EnabledWhileConnected) {
  Scene* scene = _context->CreateScene("SceneProfilerTest");
  StartServer();
  ConnectClient();
  ASSERT_NE(nullptr, GetDebugger(scene));
  EXPECT_FALSE(IsProfilerEnabled(scene)); // Not until the debugger runs on the scene's thread.

  scene->UpdateDebugger();
  EXPECT_TRUE(IsProfilerEnabled(scene));

  DisconnectClient();
  test::WaitUntil([&] { return GetDebugger(scene) == nullptr; });
  scene->UpdateDebugger();
  EXPECT_FALSE(IsProfilerEnabled(scene));
}

TEST_F(SceneProfilerTest, StaysEnabledAfterQuickReconnect) {
  Scene* scene = _context->CreateScene("SceneProfilerTest");
  StartServer();
  ConnectClient();
  auto const oldDebugger = GetDebugger(scene);
  ASSERT_NE(nullptr, oldDebugger);
  scene->UpdateDebugger();
  ASSERT_TRUE(IsProfilerEnabled(scene));

  // Both debuggers run on the scene's thread only after the new one has replaced the old one.
  DisconnectClient();
  ConnectClient();
  test::WaitUntil([&] {
    auto const debugger = GetDebugger(scene);
    return debugger != nullptr && debugger != oldDebugger;
  });
  scene->UpdateDebugger();
  EXPECT_TRUE(IsProfilerEnabled(scene));
}
