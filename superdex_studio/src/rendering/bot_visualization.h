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

#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/dynamic_array.h>

#include <mochi_renderer/debug.h>

#include <filament/Engine.h>
#include <math/mat4.h>
#include <math/vec4.h>

#include "assets/mochi_prefab_asset.h"
#include "core/settings.h"

namespace superdex::studio {

struct SceneStage;

// Per-editor visibility toggles, driven by the viewport Show menu. Appearance (colors, sizes) is
// app-wide and lives in AppSettings::botVisualization.
struct BotVisualizationFlags {
  bool showInertiaBox = false;
  bool showCenterOfMass = false;
  bool showLocalTransform = false;
  bool showJointLimits = false;
  bool showRevoluteLimits = true;
  bool showPrismaticLimits = true;
  bool showSphericalLimits = true;
  // Draw a sphere marker at each cycle joint's pivot.
  bool showCycles = true;
  bool showLinearTransmissions = true;
  bool showSpatialTendons = true;
};

/* @brief Draws link property visualizations for a specific link.
 * Includes: inertia equivalent box, center of mass sphere, local transform axes. */
void DrawLinkVisualization(
    mochi_renderer::DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    int linkIndex,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    LinkVisualizationSettings const& settings);

/* @brief Draws link visualizations for all links in the bot. */
void DrawAllLinkVisualizations(
    mochi_renderer::DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    LinkVisualizationSettings const& settings);

/* @brief Draws joint limit visualization for a specific joint using debug draw primitives.
 * For revolute joints: cylinder axis + arc fan showing rotation limits.
 * For prismatic joints: box showing translation axis and extents.
 * For spherical joints: cone showing rotational freedom. */
void DrawJointLimitVisualization(
    mochi_renderer::DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    int linkIndex,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    JointVisualizationSettings const& settings);

/* @brief Draws joint limits for all joints in the bot. */
void DrawAllJointLimits(
    mochi_renderer::DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    JointVisualizationSettings const& settings);

/* @brief Draws a sphere marker at a single cycle joint's pivot (the loop-closure point,
 * located at the joint origin expressed in the child link's frame). Uses the shared joint
 * color/scale and is gated by showJointLimits + showCycles. */
void DrawCycleJointVisualization(
    mochi_renderer::DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    int cycleIndex,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    JointVisualizationSettings const& settings);

/* @brief Draws sphere markers for all cycle joints in the bot. */
void DrawAllCycleJointVisualizations(
    mochi_renderer::DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    JointVisualizationSettings const& settings);

/* @brief Draws linear transmission visualization for a specific transmission.
 * Draws straight line segments connecting joint frames that the transmission traverses.
 * Color is binary based on transmission displacement: base orange when stretched
 * (displacement >= 0), dark brown when compressed or slack (displacement < 0). */
void DrawLinearTransmissionVisualization(
    mochi_renderer::DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    int transmissionIndex,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    TransmissionVisualizationSettings const& settings,
    float displacement = 0.0f);

/* @brief Draws linear transmission visualizations for all transmissions in the bot.
 * If transmissionDisplacements is provided, colors are adjusted based on stretch/compression. */
void DrawAllLinearTransmissionVisualizations(
    mochi_renderer::DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    TransmissionVisualizationSettings const& settings,
    mochi::DynamicArray<float> const* transmissionDisplacements = nullptr);

/* @brief Draws spatial tendon visualization for a specific tendon.
 * Draws polyline segments connecting adjacent waypoints.
 * LinearJoint elements break the polyline (no segment across them).
 * Color is binary based on tendon displacement: base purple when stretched (displacement >= 0),
 * dark purple when compressed or slack (displacement < 0). */
void DrawSpatialTendonVisualization(
    mochi_renderer::DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    int tendonIndex,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    TransmissionVisualizationSettings const& settings,
    float displacement = 0.0f);

/* @brief Draws spatial tendon visualizations for all tendons in the bot.
 * If transmissionDisplacements is provided, colors are adjusted based on stretch/compression.
 * transmissionOffset indicates where SpatialTendon data starts in the combined array. */
void DrawAllSpatialTendonVisualizations(
    mochi_renderer::DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    TransmissionVisualizationSettings const& settings,
    mochi::DynamicArray<float> const* transmissionDisplacements = nullptr,
    size_t transmissionOffset = 0);

} // namespace superdex::studio
