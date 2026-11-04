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

namespace mochi {
class Context;
} // namespace mochi

namespace superdex::robotics {
struct BotPrefab;
class RoboticsContext;
struct IBotLoader;
} // namespace superdex::robotics

namespace superdex::studio {

// Which configuration the exported skin binds to. In both cases the GLB opens in the bot's current
// pose (@ref robotics::BotPrefab::defaultPose).
enum class GlbBindPose {
  Rest, ///< Bind to the bot's zero pose. The current pose is captured
        ///< by the joint node transforms.
  Current, ///< Freeze the current pose as the bind pose.
};

// Which geometry @ref ExportSkeletalGlb writes, and what the skin binds to.
struct GlbExportOptions {
  // Include each link's render mesh (@ref robotics::BotLinkPrefab::renderModelFile) as a
  // "BotRenderMesh" node, preserving its material factors and base color texture. A deformable skin
  // (@ref robotics::BotPrefab::skin), if present, contributes its own textured render mesh here,
  // and the links it surfaces are represented by it instead of their own render mesh.
  bool includeRenderMeshes = true;
  // Include each link's collision surface mesh (@ref robotics::BotLinkPrefab::shapeFile) as an
  // untextured "BotCollisionMesh" node. A deformable skin's collision mesh, if present, is added
  // here too.
  bool includeCollisionMeshes = true;
  GlbBindPose bindPose = GlbBindPose::Rest;
};

// Writes @p botPrefab to @p outputPath as a skinned GLB: the meshes @p options selects are bound to
// a joint hierarchy mirroring the bot's articulation. Instantiates the bot in a throwaway scene,
// which poses it at @ref robotics::BotPrefab::defaultPose and resolves its link transforms and
// meshes.
//
// Render and collision geometry become two separate meshes sharing one skeleton, so an importer can
// show or hide them independently. Each is split into one primitive per distinct material.
//
// Selecting neither source writes the joint hierarchy alone, with no mesh and no skin. Note that
// without a skin there is nothing for an importer to build an armature from, so Blender reads such
// a file as plain empties rather than bones.
//
// A deformable skin (@ref robotics::BotPrefab::skin), if present, is exported as additional skinned
// geometry with its real per-vertex weights and its bone indices remapped onto the composed
// skeleton (so an attached hand's skin lands correctly on the arm), and its full PBR material
// (base color, metallic-roughness, normal, occlusion and emissive textures and factors) is
// re-embedded. The links the skin stands in for carry no separate rigid mesh; other links (e.g.
// hard couplings and nails) keep their own: render meshes are skipped for the links the skin's
// render model hides (see robotics::ComputeSkinHiddenLinks), collision meshes for the links whose
// collision the skin's collision mesh covers (see robotics::ComputeSkinCoveredLinks).
//
// Geometry and joints are converted from Mochi's Z-up basis (@ref mochi::CoordinateSpace::Default)
// into the Y-up one the render meshes use (@ref mochi_renderer::RenderSpace).
void ExportSkeletalGlb(
    char const* outputPath,
    robotics::BotPrefab const& botPrefab,
    mochi::Context* context,
    robotics::RoboticsContext* roboticsContext,
    robotics::IBotLoader const& loader,
    GlbExportOptions const& options,
    mochi::Error& error);

} // namespace superdex::studio
