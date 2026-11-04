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

#include "simulation/skin_sim_sync.h"

#include "simulation/soft_surface_queries.h"

#include <mochi_core/utils/basic_utils.h>

#include <tuple>
#include <utility>

namespace superdex::studio {

mochi::DynamicArray<int> ComputeSkinBoneLinkRemap(superdex::robotics::BotPrefab const& prefab) {
  if (prefab._skinBoneLinks.empty()) {
    return {};
  }
  mochi::DynamicArray<mochi::DynamicString> linkNames;
  linkNames.reserve(prefab.links.size());
  for (auto const& link : prefab.links) {
    linkNames.push_back(link.name);
  }
  mochi::DynamicArray<int> boneLinks =
      superdex::robotics::ComputeSkinBoneLinkIndices(prefab, mochi::MakeConstSpan(linkNames));
  for (int b = 0; b < mochi::isize(boneLinks); ++b) {
    if (boneLinks[b] == superdex::robotics::kIndexNone) {
      // Mirrors the SceneStage path, which reports the same mismatch. Posing falls back to identity
      // for this bone rather than mis-addressing another link.
      MOCHI_LOG_ERROR_ONCE(
          "Skin bone references link '%s', which is not in the built bot; that bone will not "
          "deform.",
          prefab._skinBoneLinks[b].c_str());
    }
  }
  return boneLinks;
}

void SkinSimSync::Track(
    mochi::Actor& actor,
    mochi::DynamicArray<int> boneLinkRemap,
    int stagedIndex) {
  char const* const actorName = actor.GetName();
  _actor = actor.GetHandle();
  // Must match the staged name in SceneStage so ApplySoftMeshUpdates and ApplySkinnedPose find it.
  _name = SkinStagedName(actorName ? actorName : "");
  _stagedIndex = stagedIndex;
  _boneLinkRemap = std::move(boneLinkRemap);
  _queriesRegistered = false;
  if (actor.GetSurfaceMesh().IsEmpty()) {
    return;
  }
  mochi::ErrorLog e;
  std::tie(_positions, _normals) = RegisterSoftSurfaceQueries(actor, SoftSurface::Surface, e);
  if (!e.IsOK()) {
    MOCHI_LOG_ERROR("Failed to register skin surface queries for '%s'", _name.c_str());
    // Release whichever one did register.
    actor.CancelQuery(_positions);
    actor.CancelQuery(_normals);
    return;
  }
  _queriesRegistered = true;
}

void SkinSimSync::Untrack(mochi::Scene* scene) {
  if (!_queriesRegistered) {
    return;
  }
  if (auto* actor = scene->GetActor(_actor)) {
    actor->CancelQuery(_positions);
    actor->CancelQuery(_normals);
  }
  _queriesRegistered = false;
}

void SkinSimSync::Produce(
    mochi::Actor const& actor,
    mochi::Span<mochi::TransformRT const> linkTransforms,
    mochi::CoordinateSpaceConverter const& editorToRenderer,
    SkinnedPoseUpdate& pose,
    SoftMeshUpdate* surface) const {
  MOCHI_ASSERT_VERBOSE(actor.GetHandle() == _actor);
  // The skin's bones follow its links: in nested-link order for a natively skinned actor, else
  // reordered into skin-bone (GLB joint) order so the skin's bone indices address the right links.
  pose.name = _name;
  pose.stagedIndex = _stagedIndex;
  if (_boneLinkRemap.empty()) {
    pose.linkWorldTransforms.assign(linkTransforms.begin(), linkTransforms.end());
  } else {
    pose.linkWorldTransforms.clear();
    pose.linkWorldTransforms.reserve(_boneLinkRemap.size());
    for (int const linkIndex : _boneLinkRemap) {
      pose.linkWorldTransforms.push_back(
          linkIndex >= 0 && linkIndex < mochi::isize(linkTransforms) ? linkTransforms[linkIndex]
                                                                     : mochi::TransformRT{});
    }
  }
  if (surface != nullptr) {
    MOCHI_ASSERT_VERBOSE(_queriesRegistered);
    surface->name = _name;
    surface->stagedIndex = _stagedIndex;
    surface->worldTransform = actor.GetRootTransform();
    mochi::ErrorLog e;
    ReadSoftSurfaceNodes(
        actor, SoftSurface::Surface, editorToRenderer, surface->positions, surface->normals, e);
    // The skin's collision wireframe shows the same surface.
    surface->collisionPositions.clear();
    surface->collisionNormals.clear();
  }
}

} // namespace superdex::studio
