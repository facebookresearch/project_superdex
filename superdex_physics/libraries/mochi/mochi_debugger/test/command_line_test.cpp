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

#include <mochi_debugger/lib/command_line.h>

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_physics/dbg/protocol.h>

#include <gtest/gtest.h>

#include <string>

using namespace mochi;
using namespace mochi::dbg;

static auto Parse(DynamicArray<std::string> args) {
  return CommandLine::Parse(args);
}

TEST(CommandLine, NoArgs) {
  auto [cli, error] = Parse({});
  EXPECT_EQ(CommandLine{}, cli);
  EXPECT_STREQ("", error.c_str());
}

TEST(CommandLine, Connect) {
  // --connect <empty>
  {
    auto [cli, error] = Parse({"--connect"});
    EXPECT_STREQ("127.0.0.1", cli.address.c_str());
    EXPECT_EQ(kDefaultDebugServerPort, cli.port);
    EXPECT_STREQ("", error.c_str());
  }

  // --connect <address>
  {
    auto [cli, error] = Parse({"--connect", "127.0.0.1"});
    EXPECT_STREQ("127.0.0.1", cli.address.c_str());
    EXPECT_EQ(kDefaultDebugServerPort, cli.port);
    EXPECT_STREQ("", error.c_str());
  }

  // --connect <address:port>
  {
    auto [cli, error] = Parse({"--connect", "127.0.0.1:1234"});
    EXPECT_STREQ("127.0.0.1", cli.address.c_str());
    EXPECT_EQ(1234, cli.port);
    EXPECT_STREQ("", error.c_str());
  }

  // --connect <port>
  {
    auto [cli, error] = Parse({"--connect", "1234"});
    EXPECT_STREQ("127.0.0.1", cli.address.c_str());
    EXPECT_EQ(1234, cli.port);
    EXPECT_STREQ("", error.c_str());
  }

  // --connect --connect
  {
    auto [cli, error] = Parse({"--connect", "--connect"});
    EXPECT_STREQ("127.0.0.1", cli.address.c_str());
    EXPECT_EQ(kDefaultDebugServerPort, cli.port);
    EXPECT_STREQ("", error.c_str());
  }
}

TEST(CommandLine, Singleton) {
  {
    auto [cli, error] = Parse({});
    EXPECT_FALSE(cli.singleton);
    EXPECT_STREQ("", error.c_str());
  }

  {
    auto [cli, error] = Parse({"--singleton"});
    EXPECT_TRUE(cli.singleton);
    EXPECT_STREQ("", error.c_str());
  }
}

TEST(CommandLine, RedundantArgs) {
  // --connect --connect <address>
  {
    auto [cli, error] = Parse({"--connect", "--connect", "127.0.0.1"});
    EXPECT_STREQ("127.0.0.1", cli.address.c_str());
    EXPECT_EQ(kDefaultDebugServerPort, cli.port);
    EXPECT_STREQ("", error.c_str());
  }

  // --connect <address> --connect
  {
    auto [cli, error] = Parse({"--connect", "127.0.0.1", "--connect"});
    EXPECT_STREQ("127.0.0.1", cli.address.c_str());
    EXPECT_EQ(kDefaultDebugServerPort, cli.port);
    EXPECT_STREQ("", error.c_str());
  }
}

TEST(CommandLine, InvalidConnectAddress) {
  auto [cli, error] = Parse({"--connect", "localhost"});
  EXPECT_EQ(CommandLine{}, cli);
  EXPECT_STREQ("Invalid address", error.c_str());
}

TEST(CommandLine, UnknownArg) {
  auto [cli, error] = Parse({"--connect", "127.0.0.1", "--monkey"});
  EXPECT_EQ(CommandLine{}, cli);
  EXPECT_STREQ("Unknown argument: --monkey", error.c_str());
}

TEST(CommandLine, ArgsFromMain) {
  // Build argv with non-const points (like main)
  std::string args[] = {"skip_me", "--connect", "127.0.0.1"};
  char* argv[] = {args[0].data(), args[1].data(), args[2].data()};
  int argc = isize(args);
  auto list = CommandLine::ArgsFromMain(argc, argv);
  ASSERT_EQ(2, list.size());
  EXPECT_STREQ("--connect", list[0].c_str());
  EXPECT_STREQ("127.0.0.1", list[1].c_str());
}
