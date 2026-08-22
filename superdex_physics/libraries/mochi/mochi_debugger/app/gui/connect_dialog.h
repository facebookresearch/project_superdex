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

#include <mochi_core/net/server_list.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_physics/dbg/protocol.h>

#include <array>
#include <cstdint>
#include <string_view>

namespace mochi::dbg {

struct UiState;

// Modal "Connect" dialog state
struct ConnectDialogState {
  bool shouldOpen = false; // Setting this to true triggers the dialog to open
  bool shouldCancel = false; // Setting this to true triggers the dialog to cancel
  bool startPaused = true; // Option to automatically pause the scene(s)
  std::array<char, 64> address = {}; // filled with defaults when the dialog opens
  std::array<char, 6> port = {}; // room for 5 digits + null terminator
  net::ServerList serverList{kDiscoveryPort};
  DynamicArray<net::ServerInfo> servers; // discovered servers (polled while open)
};

// Clear the log, store the address and port, and initiate a connection.
void ConnectTo(UiState& state, std::string_view address, uint16_t port);

void BuildConnectDialog(UiState& state);

} // namespace mochi::dbg
