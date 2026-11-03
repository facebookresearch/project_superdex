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

#include <tinyxml2.h>

#include <functional>
#include <optional>
#include <string>
#include <string_view>

// Internal URDF import/export interface for superdex_robotics. The implementation is a
// hand-rolled tinyxml2 parser/exporter (urdf_import_utils.cpp / urdf_export_utils.cpp) and is a
// private implementation detail — the public entry points live in
// superdex_robotics/utils/file_utils.h (LoadBotPrefabFromUrdf*, ExpandUrdfGeomLinks, SaveToUrdf*).

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

/* @brief Parse a URDF file into links and joints, with no geometry assigned. */
void LoadBotPrefabFromUrdfFile(
    BotPrefab& outData,
    std::string_view path,
    UrdfImportInfo& outInfo,
    Error& error);

void LoadBotPrefabFromUrdfXml(BotPrefab& outData, std::string_view xmlString, Error& error);

// ---------------------------------------------------------------------------
// Export (tinyxml2 backend)
// ---------------------------------------------------------------------------

std::string ExportBotPrefabToUrdfXml(BotPrefab const& botPrefab, Error& error);

void ExportBotPrefabToUrdfFile(BotPrefab const& botPrefab, std::string_view path, Error& error);

/* @brief Mesh written for one link `<visual>` or `<collision>`. */
struct UrdfMeshReference {
  /* @brief URDF mesh `filename`, or empty to omit the element. */
  std::string filename;
  /* @brief Visual material color (RGBA), or nullopt for no `<material>`. */
  std::optional<Real4> rgba;
};

/* @brief How @ref AppendBotToUrdf names a bot's links and joints and writes their meshes and
 * velocity limits. A null callback keeps the single-bot exporter's behavior. */
struct UrdfBotWriter {
  /* @brief URDF name for a link or joint name. Null keeps names unchanged. */
  std::function<std::string(std::string_view name)> rename;
  /* @brief Mesh for a link's render model (@p visual) or collision shape. Null omits all meshes. */
  std::function<UrdfMeshReference(BotLinkPrefab const& link, bool visual, Error& error)> mesh;
  /* @brief Velocity limit of a non-fixed joint with position limits. Null writes a placeholder 1,
   * since @ref BotJointPrefab has no velocity limit. */
  std::function<double(BotJointPrefab const& joint, Error& error)> velocity;
};

/* @brief Append a bot's links and joints (all but the root joint) under @p robotXml. */
void AppendBotToUrdf(
    tinyxml2::XMLElement* robotXml,
    BotPrefab const& bot,
    UrdfBotWriter const& writer,
    Error& error);

/* @brief Append an empty `<link>` under @p robotXml. */
void AppendUrdfLink(tinyxml2::XMLElement* robotXml, std::string_view name);

/* @brief Append a `fixed` `<joint>` from @p parent to @p child at @p parentFromChild. */
void AppendUrdfFixedJoint(
    tinyxml2::XMLElement* robotXml,
    std::string_view name,
    std::string_view parent,
    std::string_view child,
    TransformRT const& parentFromChild);

/* @brief Print a URDF document to a string. */
std::string PrintUrdf(tinyxml2::XMLDocument const& doc);

} // namespace superdex::robotics
