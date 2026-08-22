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

#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/file_utils.h>

#include <optional>
#include <string>
#include <string_view>

// Internal URDF import/export interface for superdex_robotics. The implementation is a
// hand-rolled tinyxml2 parser/exporter (urdf_import_utils.cpp / urdf_export_utils.cpp) and is a
// private implementation detail — the public entry points live in
// superdex_robotics/utils/file_utils.h (LoadBotPrefabFromUrdf*, SaveToUrdf*).

namespace superdex::robotics {
using namespace mochi;

// ---------------------------------------------------------------------------
// Mesh path resolution helpers
// ---------------------------------------------------------------------------

/**
 * @brief Walk up from a .urdf file's directory to find the nearest ROS package root.
 * @return Directory containing a @c package.xml marker, or empty if none is found.
 */
std::string FindUrdfPackageRoot(std::string_view urdfPath);

/**
 * @brief Resolve a URDF mesh reference to an on-disk path.
 *
 * Handles @c package:// URIs (with and without a valid @c package.xml), absolute paths, and
 * paths relative to the .urdf file. Returns the first candidate that exists on disk.
 */
std::optional<std::string> ResolveMeshPath(std::string_view meshPath, std::string_view urdfPath);

/**
 * @brief Natural (alphanumeric) string comparison. Orders embedded numbers by value.
 * @return Negative if @p l < @p r, zero if equal, positive if @p l > @p r.
 */
int AlphaNumCompare(char const* l, char const* r);

// ---------------------------------------------------------------------------
// Import (tinyxml2 backend)
// ---------------------------------------------------------------------------

void LoadBotPrefabFromUrdfFile(
    BotPrefab& outData,
    std::string_view path,
    UrdfMeshReferences* meshRefs,
    Error& error);

void LoadBotPrefabFromUrdfXml(BotPrefab& outData, std::string_view xmlString, Error& error);

// ---------------------------------------------------------------------------
// Export (tinyxml2 backend)
// ---------------------------------------------------------------------------

std::string ExportBotPrefabToUrdfXml(BotPrefab const& botPrefab, Error& error);

void ExportBotPrefabToUrdfFile(BotPrefab const& botPrefab, std::string_view path, Error& error);

} // namespace superdex::robotics
