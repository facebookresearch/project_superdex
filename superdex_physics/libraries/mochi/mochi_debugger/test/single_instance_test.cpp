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

#include <mochi_debugger/lib/single_instance.h>

#include <mochi_core/test/wait_until.h>
#include <mochi_core/utils/dynamic_array.h>

#include <gtest/gtest.h>
#include <optional>

#include <string>

using namespace mochi;
using namespace mochi::dbg;

// Disabled because some CI machines restrict loopback networking privileges.
TEST(SingleInstanceHelper, DISABLED_ForwardsConnectCommandLineToPrimary) {
  CommandLine primaryCommandLine;
  primaryCommandLine.singleton = true;
  SingleInstanceHelper primary(primaryCommandLine, {});
  ASSERT_FALSE(primary.ShouldExit());

  CommandLine secondaryCommandLine;
  secondaryCommandLine.singleton = true;
  secondaryCommandLine.address = "127.0.0.1";
  secondaryCommandLine.port = 1234;
  DynamicArray<std::string> secondaryArgs = {"--singleton", "--connect", "127.0.0.1:1234"};
  SingleInstanceHelper secondary(secondaryCommandLine, secondaryArgs);
  EXPECT_TRUE(secondary.ShouldExit());

  std::optional<CommandLine> received;
  test::WaitUntil([&] {
    received = primary.PollCommandLine();
    return received.has_value();
  });

  ASSERT_TRUE(received.has_value());
  EXPECT_FALSE(received->singleton);
  EXPECT_STREQ("127.0.0.1", received->address.c_str());
  EXPECT_EQ(1234, received->port);
}
