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

#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/reflection.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>

namespace mochi::net {

// -------------------------------------------------------------------------------------------------
// Types
// -------------------------------------------------------------------------------------------------

struct ServerInfo {
  DynamicString label; // Human-readable label for identification
  DynamicString address; // IP address (e.g. "127.0.0.1")
  uint16_t port{0}; // TCP port
  uint16_t numClients{0}; // Number of clients already connected
  uint16_t maxClients{0}; // Maximum number of clients that can connect
  uint64_t version{0}; // User defined version number

  MOCHI_STRUCT_BEGIN(mochi::net::ServerInfo)
  MOCHI_FIELD(label)
  MOCHI_FIELD(address)
  MOCHI_FIELD(port)
  MOCHI_FIELD(numClients)
  MOCHI_FIELD(maxClients)
  MOCHI_FIELD(version)
  MOCHI_STRUCT_END()
};

// -------------------------------------------------------------------------------------------------
// ServerList (UDP Discovery)
// -------------------------------------------------------------------------------------------------

/**
 * @brief Discovers servers on the local network via UDP broadcast.
 *
 * @ref Refresh sends a UDP broadcast and starts collecting responses asynchronously.
 * @ref GetServers returns whatever has been collected so far — repeated calls may return
 * different results as more responses arrive.
 *
 * @note On Android, receiving broadcast/multicast UDP responses requires the app to hold a
 * @c WifiManager.MulticastLock (permission @c CHANGE_WIFI_MULTICAST_STATE). Without it the OS
 * filters the inbound datagrams and @ref GetServers stays empty. This is an app-side requirement;
 * the library cannot acquire the lock itself.
 */
class ServerList {
  MOCHI_DECLARE_MOVE_ONLY(ServerList);

 public:
  /** @brief Create a server list that probes the given UDP discovery port. */
  explicit ServerList(uint16_t discoveryPort);
  ~ServerList();

  /**
   * @brief Send a UDP broadcast and begin collecting server responses asynchronously.
   *
   * @details Clears previous results immediately. Responses are collected on a background thread.
   * @see GetServers
   */
  void Refresh();

  /**
   * @brief Get the servers discovered since the last @ref Refresh call.
   *
   * Thread-safe. May return different results on repeated calls as responses arrive.
   */
  void GetServers(DynamicArray<ServerInfo>& outList) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

} // namespace mochi::net
