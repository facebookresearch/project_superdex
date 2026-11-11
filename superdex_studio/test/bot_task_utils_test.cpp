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

#include "assets/bot_task_utils.h"

#include <superdex_robotics/scene/bot_task.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/reflection.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

using namespace superdex::studio;
using namespace mochi::test;
using superdex::robotics::BotTaskPrefab;

namespace {

class BotTaskUtilsTest : public testing::Test {
 protected:
  mochi::TempDirCleanup _tempDirCleanup =
      mochi::CreateTempDirectory("bot_task_utils_test", ExpectOK{});
  std::string _path = (_tempDirCleanup.Path() / "task.superdex_task").generic_string();

  [[nodiscard]] BotTaskPrefab ReadTask() const {
    return superdex::robotics::LoadBotTaskPrefabFromFile(_path, ExpectOK{});
  }
};

TEST_F(BotTaskUtilsTest, CreateWritesAnEmptyNamedTask) {
  std::string error;
  ASSERT_TRUE(CreateBotTaskFile(_path, "Task", error)) << error;
  auto const task = ReadTask();
  EXPECT_EQ(std::string(task.metadata.name), "Task");
  EXPECT_TRUE(task.resetState.empty());
}

TEST_F(BotTaskUtilsTest, SetNameKeepsTheRestOfTheTask) {
  BotTaskPrefab task;
  task.metadata.name = "Old";
  task.metadata.author = "Someone";
  task.resetState.resize(1);
  task.resetState[0].name = "board";
  task.resetState[0].prefabName = "nist_board";
  superdex::robotics::SaveParamsToFile(task, _path, ExpectOK{});

  std::string error;
  ASSERT_TRUE(SetBotTaskName(_path, "New", error)) << error;

  BotTaskPrefab expected = task;
  expected.metadata.name = "New";
  EXPECT_EQ(SReflect::ToJsonString(ReadTask(), true), SReflect::ToJsonString(expected, true));
}

TEST_F(BotTaskUtilsTest, SetNameOnMissingFileFailsWithoutCreatingIt) {
  std::string error;
  EXPECT_FALSE(SetBotTaskName(_path, "New", error));
  EXPECT_FALSE(error.empty());
  EXPECT_FALSE(std::filesystem::exists(_path));
}

} // namespace
