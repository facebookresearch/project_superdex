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

#pragma once

#include <mochi_core/utils/error.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace mochi::dbg {

constexpr std::string_view kLocalHost = "127.0.0.1";

/**
 * @brief Parse a network port number from a string.
 *
 * @param str String to parse.
 * @param error Set an error if the port is invalid.
 * @return Parsed port.
 */
uint16_t ParsePort(std::string_view str, Error& error);

/**
 * @brief Validate an IPv4 address (no port suffix).
 *
 * @param address Address string to check
 * @param error Set an error if the address is invalid.
 */
void ValidateAddress(std::string_view address, Error& error);

/**
 * @brief Parse an IPv4 address and port string.
 *
 * @details If argument is empty, "127.0.0.1" and @ref kDefaultDebugServerPort are returned.
 * If argument is a bare IPv4 address, then @ref kDefaultDebugServerPort is used.
 * If argument is a bare port, then address "127.0.0.1" is used.
 * The argument may also be formatted as "address:port".
 *
 * @param[in] arg Input string in the form "", "address", "port", or "address:port".
 * @param[out] error Error set when the address or port is invalid.
 * @return Parsed address and port.
 */
std::pair<std::string, uint16_t> ParseAddressAndPort(std::string_view arg, Error& error);

} // namespace mochi::dbg
