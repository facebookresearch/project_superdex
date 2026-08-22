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

#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/file_utils.h>
#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/file_utils.h>

#include <filesystem>
#include <string>
#include <string_view>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

class ModBotTest : public testing::Test {
 protected:
  std::string WriteBotFile(std::string_view name, std::string_view content) {
    auto path = _tempDir / name;
    mochi::WriteFile(path, content, ExpectOK{});
    return path.generic_string();
  }

  mochi::TempDirCleanup _tempDirCleanup = mochi::CreateTempDirectory("mod_bot_test", ExpectOK{});
  std::filesystem::path _tempDir = _tempDirCleanup.Path();
};

} // namespace

// A mod bot is identified by the presence of the 'base' key, not by it being non-empty: a mod bot
// created in Studio has no base assigned yet.
TEST_F(ModBotTest, GetBotFileType_EmptyBaseIsModBot) {
  auto path = WriteBotFile("empty_mod.superdex_bot", R"({ "name": "Mod Bot", "base": "" })");

  EXPECT_EQ(FileBotLoader{}.GetBotFileType(path, ExpectOK{}), BotFileType::ModBotPrefab);
}

TEST_F(ModBotTest, GetBotFileType_NoBaseKeyIsBotPrefab) {
  auto path = WriteBotFile("plain.superdex_bot", R"({ "name": "Bot" })");

  EXPECT_EQ(FileBotLoader{}.GetBotFileType(path, ExpectOK{}), BotFileType::BotPrefab);
}

TEST_F(ModBotTest, GetBotFileType_MalformedJsonFails) {
  auto path = WriteBotFile("malformed.superdex_bot", R"({ "name": )");

  auto const suppressWarning = SuppressLogWarning();
  FileBotLoader{}.GetBotFileType(path, ExpectNotOK{});
}

TEST_F(ModBotTest, GetBotFileType_NonObjectRootFails) {
  auto path = WriteBotFile("array_root.superdex_bot", "[]");

  FileBotLoader{}.GetBotFileType(path, ExpectNotOK{});
}

// Deserializing a plain bot into a ModBotPrefab would otherwise succeed while silently dropping
// every field ModBotPrefab does not declare.
TEST_F(ModBotTest, LoadModBotPrefab_PlainBotFileFails) {
  auto path = WriteBotFile("plain.superdex_bot", R"({ "name": "Bot", "links": [] })");

  FileBotLoader{}.LoadModBotPrefab(path, ExpectNotOK{});
}

TEST_F(ModBotTest, LoadBotPrefabFromFile_EmptyBaseBuildsEmptyBot) {
  auto path = WriteBotFile("empty_mod.superdex_bot", R"({ "name": "Mod Bot", "base": "" })");

  auto botPrefab = LoadBotPrefabFromFile(path, ExpectOK{});
  EXPECT_EQ(std::string(botPrefab.name), "Mod Bot");
  EXPECT_TRUE(botPrefab.links.empty());
  EXPECT_TRUE(botPrefab.joints.empty());
}

// Mirrors the Studio "Create > Mod Bot..." flow: a saved base-less mod bot must round-trip as a
// mod bot, not degrade into a plain bot.
TEST_F(ModBotTest, SaveAndLoad_EmptyModBotRoundTrips) {
  ModBotPrefab modBotPrefab;
  modBotPrefab.name = "Mod Bot";
  auto path = (_tempDir / "roundtrip.superdex_bot").generic_string();
  SaveToFile(modBotPrefab, path, ExpectOK{});

  EXPECT_EQ(FileBotLoader{}.GetBotFileType(path, ExpectOK{}), BotFileType::ModBotPrefab);
  auto loaded = FileBotLoader{}.LoadModBotPrefab(path, ExpectOK{});
  EXPECT_EQ(std::string(loaded.name), "Mod Bot");
  EXPECT_TRUE(loaded.base.empty());
}
