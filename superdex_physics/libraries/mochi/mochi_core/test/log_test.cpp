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

#include <gtest/gtest.h>

#include <iterator>
#include <string>
#include <vector>

using namespace mochi;

namespace {
struct MessageInfo {
  LogChannel channel = {};
  std::string text;
  std::string file;
  int line = 0;
};
} // namespace

TEST(Log, SetLogCallback) {
  // Custom logging function:
  std::vector<MessageInfo> messages;
  LogFn fn = [&](LogChannel chan, char const* msg, char const* file, int line) {
    messages.push_back(MessageInfo{chan, msg, file, line});
  };

  // Backup & replace
  auto prevFn = GetLogCallback();
  SetLogCallback(fn);

  MOCHI_LOG("");
  MOCHI_LOG("Hello World");
  MOCHI_LOG("Multi\nLine");
  MOCHI_LOG("Message %s %d %s", "formatted", 4, "you");

  char const* const kExpectedMessages[] = {
      "\n",
      "Hello World\n",
      "Multi\nLine\n",
      "Message formatted 4 you\n",
  };

  EXPECT_EQ(std::size(kExpectedMessages), messages.size());
  for (size_t i = 0; i < messages.size(); ++i) {
    EXPECT_EQ(LogChannel::Info, messages[i].channel);
    EXPECT_STREQ(kExpectedMessages[i], messages[i].text.c_str());
    EXPECT_STRNE("", messages[i].file.c_str()); // Expect non-empty file name
    EXPECT_STREQ(messages[0].file.c_str(), messages[i].file.c_str()); // File names should match
    if (i > 0) {
      EXPECT_EQ(messages[i - 1].line + 1, messages[i].line); // Line numbers increasing by 1
    }
  }

  // Restore
  SetLogCallback(prevFn);
}

TEST(Log, EnableLogChannel) {
  // Custom logging function:
  std::vector<MessageInfo> messages;
  LogFn fn = [&](LogChannel chan, char const* msg, char const* file, int line) {
    messages.push_back(MessageInfo{chan, msg, file, line});
  };

  // Backup & replace
  auto prevFn = GetLogCallback();
  SetLogCallback(fn);
  bool wasEnabled[(size_t)LogChannel::Count] = {};
  for (size_t i = 0; i < (size_t)LogChannel::Count; ++i) {
    wasEnabled[i] = IsLogChannelEnabled((LogChannel)i);
  }

  // Disable all
  for (size_t i = 0; i < (size_t)LogChannel::Count; ++i) {
    EnableLogChannel((LogChannel)i, false);
  }

  // Log to multiple channels
  auto spam = [] {
    MOCHI_LOG("info");
    MOCHI_LOG_WARNING("warn");
    MOCHI_LOG_ERROR("oof");
  };

  spam();
  EXPECT_EQ(0, messages.size()); // crickets
  messages.clear();

  // Enable just warnings
  EnableLogChannel(LogChannel::Warning, true);
  spam();
  EXPECT_EQ(1, messages.size());
  EXPECT_EQ(LogChannel::Warning, messages.front().channel);
  EXPECT_STREQ("warn\n", messages.front().text.c_str());
  messages.clear();
  EnableLogChannel(LogChannel::Warning, false);

  // Enable just errors
  EnableLogChannel(LogChannel::Error, true);
  spam();
  EXPECT_EQ(1, messages.size());
  EXPECT_EQ(LogChannel::Error, messages.front().channel);
  EXPECT_STREQ("oof\n", messages.front().text.c_str());
  messages.clear();
  EnableLogChannel(LogChannel::Error, false);

  // Restore
  for (size_t i = 0; i < (size_t)LogChannel::Count; ++i) {
    EnableLogChannel((LogChannel)i, wasEnabled[i]);
  }
  SetLogCallback(prevFn);
}
