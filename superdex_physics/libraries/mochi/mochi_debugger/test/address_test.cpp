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

#include <mochi_debugger/lib/address.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/dbg/protocol.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

using namespace mochi;
using namespace mochi::dbg;

static std::pair<std::string, uint16_t> Parse(std::string_view arg, Error& error) {
  return ParseAddressAndPort(arg, error);
}

TEST(Address, ParsePortValidatesNumericRange) {
  EXPECT_EQ(0, ParsePort("0", test::ExpectOK{}));
  EXPECT_EQ(65535, ParsePort("65535", test::ExpectOK{}));
  for (std::string_view port : {"", "abc", "12abc", "65536"}) {
    Error error;
    ParsePort(port, error);
    EXPECT_FALSE(error.IsOK());
    EXPECT_STREQ("Invalid port number", error.GetDescription());
  }
}

TEST(Address, ValidateAddressAcceptsIpv4Address) {
  ValidateAddress("192.168.1.20", test::ExpectOK{});
}

TEST(Address, ValidateAddressRejectsInvalidIpv4Address) {
  for (std::string_view address :
       {"", "localhost", "127.0.0", "127.0.0.1.2", "127.0.0.256", "127.0.0.01"}) {
    Error error;
    ValidateAddress(address, error);
    EXPECT_FALSE(error.IsOK());
    EXPECT_STREQ("Invalid address", error.GetDescription());
  }
}

TEST(Address, EmptyUsesLocalhostDefaultPort) {
  auto [address, port] = Parse("", test::ExpectOK{});
  EXPECT_STREQ("127.0.0.1", address.c_str());
  EXPECT_EQ(kDefaultDebugServerPort, port);
}

TEST(Address, AddressUsesDefaultPort) {
  auto [address, port] = Parse("192.168.1.20", test::ExpectOK{});
  EXPECT_STREQ("192.168.1.20", address.c_str());
  EXPECT_EQ(kDefaultDebugServerPort, port);
}

TEST(Address, AddressAndPort) {
  auto [address, port] = Parse("192.168.1.20:1234", test::ExpectOK{});
  EXPECT_STREQ("192.168.1.20", address.c_str());
  EXPECT_EQ(1234, port);
}

TEST(Address, PortUsesLocalhost) {
  {
    auto [address, port] = Parse("1234", test::ExpectOK{});
    EXPECT_STREQ("127.0.0.1", address.c_str());
    EXPECT_EQ(1234, port);
  }
  {
    auto [address, port] = Parse(":2345", test::ExpectOK{});
    EXPECT_STREQ("127.0.0.1", address.c_str());
    EXPECT_EQ(2345, port);
  }
}

TEST(Address, PortRange) {
  {
    auto [address, port] = Parse("127.0.0.1:0", test::ExpectOK{});
    EXPECT_STREQ("127.0.0.1", address.c_str());
    EXPECT_EQ(0, port);
  }
  {
    auto [address, port] = Parse("127.0.0.1:65535", test::ExpectOK{});
    EXPECT_STREQ("127.0.0.1", address.c_str());
    EXPECT_EQ(65535, port);
  }
}

TEST(Address, InvalidAddress) {
  for (std::string arg : {"localhost", "127.0.0", "127.0.0.1.2", "127.0.0.256", "127.0.0.01"}) {
    Error error;
    Parse(arg, error);
    EXPECT_FALSE(error.IsOK());
    EXPECT_STREQ("Invalid address", error.GetDescription());
  }
}

TEST(Address, MoreThanOneColonIsInvalidAddress) {
  for (std::string arg : {"127.0.0.1:1234:5", "127.0.0.1::1234"}) {
    Error error;
    Parse(arg, error);
    EXPECT_FALSE(error.IsOK());
    EXPECT_STREQ("Invalid address", error.GetDescription());
  }
}

TEST(Address, InvalidPort) {
  for (std::string arg : {"127.0.0.1:", "127.0.0.1:abc", "127.0.0.1:65536", "65536"}) {
    Error error;
    Parse(arg, error);
    EXPECT_FALSE(error.IsOK());
    EXPECT_STREQ("Invalid port number", error.GetDescription());
  }
}
