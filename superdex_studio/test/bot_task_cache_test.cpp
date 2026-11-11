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

#include "assets/bot_task_cache.h"

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

class BotTaskCacheTest : public testing::Test {
 protected:
  mochi::TempDirCleanup _tempDirCleanup =
      mochi::CreateTempDirectory("bot_task_cache_test", ExpectOK{});
  std::string _path = (_tempDirCleanup.Path() / "task.superdex_task").generic_string();

  static BotTaskPrefab TaskWithSpawn(std::string const& name) {
    BotTaskPrefab task;
    task.resetState.resize(1);
    task.resetState[0].name = name;
    task.resetState[0].prefabName = "board";
    return task;
  }

  void WriteTask(BotTaskPrefab const& task) const {
    superdex::robotics::SaveParamsToFile(task, _path, ExpectOK{});
  }

  [[nodiscard]] BotTaskPrefab ReadTask() const {
    return superdex::robotics::LoadBotTaskPrefabFromFile(_path, ExpectOK{});
  }
};

TEST_F(BotTaskCacheTest, LoadedEntryIsUnmodifiedUntilEdited) {
  WriteTask(TaskWithSpawn("a"));
  BotTaskCache cache;
  std::string error;
  auto* entry = cache.GetOrLoad(_path, error);
  ASSERT_NE(entry, nullptr) << error;
  EXPECT_FALSE(BotTaskCache::IsModified(*entry));
  entry->json = SReflect::ToJsonString(TaskWithSpawn("b"), true);
  EXPECT_TRUE(BotTaskCache::IsModified(*entry));
}

TEST_F(BotTaskCacheTest, LoadMissingFileFails) {
  BotTaskCache cache;
  std::string error;
  EXPECT_EQ(cache.GetOrLoad(_path, error), nullptr);
  EXPECT_FALSE(error.empty());
}

TEST_F(BotTaskCacheTest, SaveWritesEditsAndClearsModified) {
  WriteTask(TaskWithSpawn("a"));
  BotTaskCache cache;
  std::string error;
  auto* entry = cache.GetOrLoad(_path, error);
  ASSERT_NE(entry, nullptr) << error;
  entry->json = SReflect::ToJsonString(TaskWithSpawn("b"), true);
  ASSERT_TRUE(cache.Save(_path, error)) << error;
  EXPECT_FALSE(BotTaskCache::IsModified(*entry));
  EXPECT_EQ(std::string(ReadTask().resetState[0].name), "b");
}

TEST_F(BotTaskCacheTest, SaveRefusesInvalidTask) {
  WriteTask(TaskWithSpawn("a"));
  BotTaskCache cache;
  std::string error;
  auto* entry = cache.GetOrLoad(_path, error);
  ASSERT_NE(entry, nullptr) << error;
  BotTaskPrefab duplicate = TaskWithSpawn("a");
  duplicate.resetState.push_back(TaskWithSpawn("a").resetState[0]);
  entry->json = SReflect::ToJsonString(duplicate, true);
  EXPECT_FALSE(cache.SaveModified(error));
  EXPECT_TRUE(BotTaskCache::IsModified(*entry));
  EXPECT_EQ(ReadTask().resetState.size(), 1);
}

TEST_F(BotTaskCacheTest, RestoreRevertsEntriesMissingFromSnapshot) {
  WriteTask(TaskWithSpawn("a"));
  BotTaskCache cache;
  auto const before = cache.CaptureValues();
  std::string error;
  auto* entry = cache.GetOrLoad(_path, error);
  ASSERT_NE(entry, nullptr) << error;
  std::string const original = entry->json;
  entry->json = SReflect::ToJsonString(TaskWithSpawn("b"), true);
  cache.RestoreValues(before);
  EXPECT_EQ(entry->json, original);
}

TEST_F(BotTaskCacheTest, RestoreAfterSaveRevertsToLoadedValue) {
  WriteTask(TaskWithSpawn("a"));
  BotTaskCache cache;
  auto const before = cache.CaptureValues();
  std::string error;
  auto* entry = cache.GetOrLoad(_path, error);
  ASSERT_NE(entry, nullptr) << error;
  std::string const original = entry->json;
  entry->json = SReflect::ToJsonString(TaskWithSpawn("b"), true);
  ASSERT_TRUE(cache.Save(_path, error)) << error;
  cache.RestoreValues(before);
  EXPECT_EQ(entry->json, original);
  EXPECT_TRUE(BotTaskCache::IsModified(*entry));
  EXPECT_TRUE(cache.AnyModified());
}

TEST_F(BotTaskCacheTest, SaveAsMovesEditsToTheCopy) {
  WriteTask(TaskWithSpawn("a"));
  BotTaskCache cache;
  std::string error;
  auto* original = cache.GetOrLoad(_path, error);
  ASSERT_NE(original, nullptr) << error;
  original->json = SReflect::ToJsonString(TaskWithSpawn("b"), true);
  std::string const copyPath = (_tempDirCleanup.Path() / "copy.superdex_task").generic_string();
  ASSERT_TRUE(cache.SaveAs(_path, copyPath, error)) << error;

  EXPECT_FALSE(BotTaskCache::IsModified(*original));
  EXPECT_EQ(std::string(ReadTask().resetState[0].name), "a");
  auto const* copy = cache.Find(copyPath);
  ASSERT_NE(copy, nullptr);
  EXPECT_FALSE(BotTaskCache::IsModified(*copy));
  EXPECT_EQ(
      std::string(
          superdex::robotics::LoadBotTaskPrefabFromFile(copyPath, ExpectOK{}).resetState[0].name),
      "b");
}

TEST_F(BotTaskCacheTest, SaveAsOverLoadedTaskKeepsItsLoadedValue) {
  std::string const copyPath = (_tempDirCleanup.Path() / "copy.superdex_task").generic_string();
  WriteTask(TaskWithSpawn("a"));
  superdex::robotics::SaveParamsToFile(TaskWithSpawn("c"), copyPath, ExpectOK{});
  BotTaskCache cache;
  auto const before = cache.CaptureValues();
  std::string error;
  auto const* loadedCopy = cache.GetOrLoad(copyPath, error);
  ASSERT_NE(loadedCopy, nullptr) << error;
  std::string const copyAsLoaded = loadedCopy->json;
  auto* original = cache.GetOrLoad(_path, error);
  ASSERT_NE(original, nullptr) << error;
  original->json = SReflect::ToJsonString(TaskWithSpawn("b"), true);
  ASSERT_TRUE(cache.SaveAs(_path, copyPath, error)) << error;
  cache.RestoreValues(before);
  EXPECT_EQ(cache.Find(copyPath)->json, copyAsLoaded);
}

} // namespace
