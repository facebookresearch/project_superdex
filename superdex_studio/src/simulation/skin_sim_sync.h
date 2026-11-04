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

#include "rendering/scene_stage.h"

#include <superdex_robotics/superdex_robotics.h>

#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_physics/mochi_physics.h>

#include <string>

namespace superdex::studio {

// For each bone of `prefab`'s skin, its link index in nested-link order. Empty when bone i is link
// i (a natively skinned bot); a composed bot that adopted a sub-bot's skin gets the remap.
[[nodiscard]] mochi::DynamicArray<int> ComputeSkinBoneLinkRemap(
    superdex::robotics::BotPrefab const& prefab);

// Mirrors the skin of one simulated articulated actor into the skin SceneStage staged for it: poses
// it from the link transforms each step and, when the actor carries the skin's collision surface,
// reads Mochi's deformed surface. Track, Untrack and Produce run on the simulation thread.
class SkinSimSync {
 public:
  // Starts tracking `actor`'s skin. `boneLinkRemap` is ComputeSkinBoneLinkRemap's result for a bot
  // (empty otherwise). The updates target the staged actor at `stagedIndex`, or, when it is
  // negative, the one named SkinStagedName(actor name). Registers the surface queries when the
  // actor has a skin surface; if they cannot be registered (logged), the skin is still posed.
  void Track(mochi::Actor& actor, mochi::DynamicArray<int> boneLinkRemap, int stagedIndex = -1);
  // Cancels the surface queries. Call while the actor is still alive.
  void Untrack(mochi::Scene* scene);

  // Fills this step's skin pose from the tracked actor's nested link world transforms and, when
  // HasSurface(), `surface` with its deformed surface (null otherwise). Reuses the update buffers.
  void Produce(
      mochi::Actor const& actor,
      mochi::Span<mochi::TransformRT const> linkTransforms,
      mochi::CoordinateSpaceConverter const& editorToRenderer,
      SkinnedPoseUpdate& pose,
      SoftMeshUpdate* surface) const;

  [[nodiscard]] mochi::ActorHandle GetActorHandle() const {
    return _actor;
  }
  // The staged skin's name.
  [[nodiscard]] std::string const& GetName() const {
    return _name;
  }
  // Whether the actor carries the skin's collision surface, read into Produce's `surface`.
  [[nodiscard]] bool HasSurface() const {
    return _queriesRegistered;
  }

 private:
  mochi::ActorHandle _actor;
  std::string _name;
  int _stagedIndex = -1;
  mochi::DynamicArray<int> _boneLinkRemap;
  mochi::QueryHandle _positions;
  mochi::QueryHandle _normals;
  bool _queriesRegistered = false;
};

} // namespace superdex::studio
