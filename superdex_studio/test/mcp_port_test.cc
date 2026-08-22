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

#include "app/mcp_port.h"

#include <gtest/gtest.h>

namespace superdex::studio {
namespace {

TEST(SelectMcpPort, UnsetUsesDefault) {
  EXPECT_EQ(SelectMcpPort(nullptr), (McpPortSelection{18086, false}));
}

TEST(SelectMcpPort, ValidOverrideUsesRequestedPort) {
  EXPECT_EQ(SelectMcpPort("1"), (McpPortSelection{1, true}));
  EXPECT_EQ(SelectMcpPort("18087"), (McpPortSelection{18087, true}));
  EXPECT_EQ(SelectMcpPort("65535"), (McpPortSelection{65535, true}));
}

TEST(SelectMcpPort, MalformedOverrideUsesDefault) {
  EXPECT_EQ(SelectMcpPort(""), (McpPortSelection{18086, false}));
  EXPECT_EQ(SelectMcpPort("18086x"), (McpPortSelection{18086, false}));
  EXPECT_EQ(SelectMcpPort(" 18086"), (McpPortSelection{18086, false}));
  EXPECT_EQ(SelectMcpPort("99999999999999999999"), (McpPortSelection{18086, false}));
}

TEST(SelectMcpPort, OutOfRangeOverrideUsesDefault) {
  EXPECT_EQ(SelectMcpPort("0"), (McpPortSelection{18086, false}));
  EXPECT_EQ(SelectMcpPort("-1"), (McpPortSelection{18086, false}));
  EXPECT_EQ(SelectMcpPort("65536"), (McpPortSelection{18086, false}));
}

} // namespace
} // namespace superdex::studio
