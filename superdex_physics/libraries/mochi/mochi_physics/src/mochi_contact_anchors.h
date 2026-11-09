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

#include "mochi_active_collisions.h"
#include "mochi_common_components.h"
#include "mochi_ecs.h"

#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/span.h>

#include <cstdint>
#include <type_traits>

namespace mochi {

// Registry access of the friction anchor systems: whether a collider is rigid.
using FrictionAnchorReg = ecs::PartialRegistry<TagRigidActor const, TagStaticActor const>;

// Friction anchor of a colliding contact sample on a rigid collider: a material point of the
// collider.
struct FrictionAnchor {
  static_assert(
      std::is_same_v<std::underlying_type_t<entt::entity>, uint32_t>,
      "Colliders are stored as the integral value of entt::entity.");

  int sample = 0;
  uint32_t collider = 0; // entt::to_integral, which reflection can serialize
  Real3 point = {}; // In the collider's local space

  MOCHI_STRUCT_BEGIN(mochi::FrictionAnchor);
  MOCHI_FIELD(sample);
  MOCHI_FIELD(collider);
  MOCHI_FIELD(point);
  MOCHI_STRUCT_END();
};
static_assert(
    std::is_trivially_copyable_v<FrictionAnchor> &&
        sizeof(FrictionAnchor) == sizeof(int) + sizeof(uint32_t) + sizeof(Real3),
    "Anchors are copied in bulk, without padding.");

/**
 * @brief Friction anchors of a colliding actor, grouped by colliding contact sample: the anchors of
 * each sample are contiguous. CFrictionAnchorIndex locates the group of each sample.
 *
 * @details An anchor is a point in the collider's local space. Anchors apply only to rigid
 * colliders, static or dynamic, so a sample has at most one anchor per collider. Each collision
 * detection pass at TimeStep::Current sets ContactDetectionResult::anchorOffset to the stage-start
 * position of each contact relative to its anchor, or zero without one. The anchors stay fixed
 * during the stages of a step: UpdateFrictionAnchors rebuilds them once, after the last stage,
 * from the nonzero offsets and the contact positions.
 *
 * @note Anchors are points rather than offsets, and are committed only at the end of a step, the
 * only state of the trajectory within it: the stage-start positions of multi-step and multi-stage
 * integrators are not the ends of the previous step or stage, so an offset from one of them does
 * not locate the anchor in the next one.
 *
 * @note Only the anchors are captured; the index is derived from them after a restore.
 * @note The collider includes the entity version: anchors on a destroyed collider never match a
 * recycled entity, and the next rebuild drops them.
 */
struct CFrictionAnchors : public NoCopy {
  DynamicArray<FrictionAnchor> anchors;

  MOCHI_STRUCT_BEGIN(mochi::CFrictionAnchors);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_FIELD(anchors);
  MOCHI_STRUCT_END();
};

/**
 * @brief Index of the friction anchors of a colliding actor by colliding contact sample. Derived
 * from CFrictionAnchors, so it is not captured.
 *
 * @details For a sample s with anchors, index[s] is the start of its group. Other entries may be
 * stale, so an entry i is valid only if anchors[i].sample == s: a rebuild or a restore then only
 * touches the samples of the new anchors.
 */
struct CFrictionAnchorIndex : private DynamicArray<int>, public NoCopy {
  CFrictionAnchorIndex() = default;
  explicit CFrictionAnchorIndex(int numSamples) {
    resize(numSamples, kSentinelIndex);
  }

  using DynamicArray<int>::operator[];
  using DynamicArray<int>::size;
};

// Sets the anchor offsets of the contacts with rigid colliders to their stage-start positions
// relative to their anchors; contacts without an anchor get zero.
void FillFrictionAnchorOffsets(
    FrictionAnchorReg const& reg,
    CFrictionAnchorIndex const& index,
    Span<FrictionAnchor const> anchors,
    Span<ActiveCollision> outCollisions);

// Sets the anchor offsets of the current contact results of a colliding actor from its stored
// friction anchors, after a collision detection pass at TimeStep::Current.
template <bool kIsSync>
void FillFrictionAnchorOffsets(
    FrictionAnchorReg reg,
    CFrictionAnchorIndex const& index,
    CFrictionAnchors const& anchorsComp,
    CActiveCollisions<kIsSync, TimeStep::Current>& activeCollisions) {
  FillFrictionAnchorOffsets(reg, index, anchorsComp.anchors, MakeSpan(activeCollisions));
}

// Rebuilds the anchors of a colliding actor and their index from the nonzero anchor offsets of its
// contacts with rigid colliders, at their positions at the end of the step, after the last stage.
// A diverged last stage keeps the previous anchors. Reuses the capacity of the anchors.
void UpdateFrictionAnchors(
    FrictionAnchorReg reg,
    CConvergenceStatus const* convergence,
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current> const& asyncCollisions,
    CActiveCollisions</*kIsSync*/ true, TimeStep::Current> const& syncCollisions,
    CFrictionAnchors& outAnchorsComp,
    CFrictionAnchorIndex& outIndex);

// Emplaces empty friction anchor components on an actor with numSamples contact samples. Call
// only if ExperimentalEvalParams::elastoPlasticFriction is enabled.
void EmplaceFrictionAnchors(entt::registry& reg, entt::entity e, int numSamples);

// Post-restore system: the index is not captured, so point it to the group of each sample with
// restored anchors.
void ReindexFrictionAnchors(CFrictionAnchors const& anchorsComp, CFrictionAnchorIndex& outIndex);

} // namespace mochi
