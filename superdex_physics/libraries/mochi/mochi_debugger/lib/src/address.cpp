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

#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/string_utils.h>
#include <mochi_physics/dbg/protocol.h>

#include <charconv>
#include <utility>

using namespace mochi;
using namespace mochi::dbg;

uint16_t dbg::ParsePort(std::string_view str, Error& error) {
  MOCHI_ERROR_IF(str.empty(), error, "Invalid port number");
  MOCHI_ERROR_RETURN(error, 0);

  uint16_t port = 0;
  char const* const begin = str.data();
  char const* const end = str.data() + str.size();
  auto const [ptr, ec] = std::from_chars(begin, end, port);
  MOCHI_ERROR_IF((ec != std::errc{}) || (ptr != end), error, "Invalid port number");
  MOCHI_ERROR_RETURN(error, 0);
  return port;
}

static bool ContainsOnlyDigits(std::string_view s) {
  if (s.empty()) {
    return false;
  }
  for (char const c : s) {
    if (c < '0' || c > '9') {
      return false;
    }
  }
  return true;
}

static bool IsValidIpv4(std::string_view s) {
  int octetCount = 0;
  size_t pos = 0;
  for (;;) {
    size_t const dot = s.find('.', pos);
    std::string_view const octet =
        (dot == std::string_view::npos) ? s.substr(pos) : s.substr(pos, dot - pos);
    if (octet.empty() || octet.size() > 3) {
      return false;
    }
    // Reject leading zeros (e.g. "001"); some resolvers interpret these as octal.
    if (octet.size() > 1 && octet[0] == '0') {
      return false;
    }
    int value = 0;
    for (char const c : octet) {
      if (c < '0' || c > '9') {
        return false;
      }
      value = value * 10 + (c - '0');
    }
    if (value > 255) {
      return false;
    }
    ++octetCount;
    if (dot == std::string_view::npos) {
      break;
    }
    pos = dot + 1;
  }
  return octetCount == 4;
}

void dbg::ValidateAddress(std::string_view address, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(IsValidIpv4(address), error, "Invalid address");
}

std::pair<std::string, uint16_t> dbg::ParseAddressAndPort(std::string_view arg, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  if (arg.empty()) {
    return {std::string{kLocalHost}, kDefaultDebugServerPort};
  }

  auto const tokens = mochi::Split(arg, ":", true);
  if (tokens.size() > 2) {
    MOCHI_ERROR_SET(error, "Invalid address");
    return {};
  }

  if (tokens.size() == 2) {
    std::string_view address = tokens[0].empty() ? kLocalHost : tokens[0];
    ValidateAddress(address, error);
    uint16_t const port = ParsePort(tokens[1], error);
    MOCHI_ERROR_RETURN(error, {});
    return {std::string{address}, port};
  }

  std::string_view const token = tokens[0];
  if (ContainsOnlyDigits(token)) {
    uint16_t const port = ParsePort(token, error);
    MOCHI_ERROR_RETURN(error, {});
    return {std::string{kLocalHost}, port};
  }

  ValidateAddress(token, error);
  MOCHI_ERROR_RETURN(error, {});
  return {std::string{arg}, kDefaultDebugServerPort};
}
