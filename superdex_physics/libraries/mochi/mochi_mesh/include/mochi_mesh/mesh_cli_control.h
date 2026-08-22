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

#include <string>

namespace mochi::mesh {

/// @brief Absolute path to the superdex_mesh_cli helper, or empty if it is not installed.
///
/// The helper is optional. It is GPL-licensed and ships as its own distribution, so a deployment
/// may leave it out on purpose; mesh operations then fail with a clean error and everything else
/// works. Nothing here treats its absence as fatal.
///
/// Exposed so a front end can say so once at startup rather than letting the user discover it at
/// the first mesh operation. Resolution is relative to the running executable, not the working
/// directory, so it behaves the same whether the program was launched from a shell, from a
/// console script, or from a pinned shortcut.
[[nodiscard]] std::string ResolveMeshCliPath();

/// @brief Terminate any superdex_mesh_cli helper subprocesses currently running for this process.
///
/// Each in-flight mesh op blocks on its helper subprocess (spawn-per-request over pipes). Killing
/// the subprocess makes that blocking read return, so the owning worker thread unblocks promptly
/// and the op reports a clean error. This lets a UI Cancel button stop a long-running op (e.g. a
/// mistakenly huge tessellation) and hand control back to the user.
///
/// Safe to call from any thread, including while a helper op is running on another thread, and a
/// no-op when nothing is running.
void CancelInFlightMeshCli();

} // namespace mochi::mesh
