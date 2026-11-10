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

// Whether contacts with the collider keep friction anchors, and so can use elasto-plastic friction:
// only rigid colliders, static or dynamic, whose local frame follows the material. All async
// colliders are static, so it only varies for sync contact.
[[nodiscard]] inline bool HasFrictionAnchors(FrictionAnchorReg reg, entt::entity collider) {
  return reg.any_of<TagRigidActor, TagStaticActor>(collider);
}

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

enum class AnchorSet { Active, Pending };

} // namespace mochi
MOCHI_ENUM_BEGIN(mochi::AnchorSet);
MOCHI_ENUM_ITEM(Active);
MOCHI_ENUM_ITEM(Pending);
MOCHI_ENUM_END();
namespace mochi {

/**
 * @brief Friction anchors of a colliding actor, grouped by colliding contact sample: the anchors of
 * each sample are contiguous. CFrictionAnchorIndex locates the group of each sample in the Active
 * set.
 *
 * @details An anchor is a point in the collider's local space. Anchors apply only to colliders
 * with friction anchors (see HasFrictionAnchors), which are rigid, so a sample has at most one
 * anchor per collider. AnchorSet::Active holds the anchors the step uses, and AnchorSet::Pending
 * those for the next step. At the start of each step, PromotePendingFrictionAnchors copies Pending
 * into Active without consuming it, so Pending stays valid while an island sleeps. Each collision
 * detection pass at TimeStep::Current sets ContactDetectionResult::anchorOffset to the stage-start
 * position of each contact relative to its Active anchor, or zero without one. The anchors stay
 * fixed during the stages of a step. After the last stage, AdvanceFrictionAnchorOffsets advances
 * the offsets to the end of the step, and UpdateFrictionAnchors rebuilds Pending from the nonzero
 * ones. A diverged last stage keeps Pending, which equals Active: the promotion copies it, and
 * resets drop the same anchors from both sets.
 *
 * @note Queued preload resets drop the anchors from both sets after promotion, so that no later
 * promotion brings them back, even if a sleeping island skips the solve that rebuilds Pending.
 * @note Anchors are points rather than offsets so that every stage measures the slip from the same
 * anchor, whatever its stage-start position: the stage start of multi-step and multi-stage
 * integrators is not the end of the previous step or stage. The end of the step must be the end of
 * its last stage (see EndsAtLastStage).
 * @note Both sets are captured: Active reproduces the completed step's forces, while Pending
 * resumes forward simulation. The index is not: IndexFrictionAnchors derives it from Active at the
 * start of each step.
 * @note The collider includes the entity version: anchors on a destroyed collider never match a
 * recycled entity, and the next rebuild drops them.
 */
template <AnchorSet kSet>
struct CFrictionAnchors : NoCopy {
  DynamicArray<FrictionAnchor> anchors;

  MOCHI_TEMPLATE_BEGIN(mochi::CFrictionAnchors, kSet);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_FIELD(anchors);
  MOCHI_TEMPLATE_END();
};

/**
 * @brief Index of the friction anchors of a colliding actor by colliding contact sample. Derived
 * from CFrictionAnchors<AnchorSet::Active> at the start of each step (see IndexFrictionAnchors),
 * for the collision detection passes that follow. Not captured.
 *
 * @details For a sample s with anchors, index[s] is the start of its group. Other entries may be
 * stale, so an entry i is valid only if anchors[i].sample == s: indexing then only touches the
 * samples with anchors.
 *
 * @note Valid only from IndexFrictionAnchors to the end of the step: changes to the anchors since
 * then, e.g., the rebuild of the pending anchors, which uses it as scratch, or a restore, leave it
 * stale.
 */
struct CFrictionAnchorIndex : private DynamicArray<int>, public NoCopy {
  CFrictionAnchorIndex() = default;
  explicit CFrictionAnchorIndex(int numSamples) {
    resize(numSamples, kSentinelIndex);
  }

  using DynamicArray<int>::operator[];
  using DynamicArray<int>::size;
};

/**
 * @brief Requests to clear friction preload since the last step. At the start of the next step,
 * ApplyFrictionAnchorClearRequests drops each actor's anchors from both sets, both as colliding
 * and collider.
 * Global context, present only with elasto-plastic friction.
 *
 * @note Captured, so that restoring a state also restores its pending requests, which apply to the
 * anchors of that state.
 */
struct CFrictionAnchorClearRequests : public NoCopy {
  DynamicArray<uint32_t> entities; // entt::to_integral, including the version

  MOCHI_STRUCT_BEGIN(mochi::CFrictionAnchorClearRequests);
  MOCHI_ATTRIBUTE(CaptureStateCtx);
  MOCHI_FIELD(entities);
  MOCHI_STRUCT_END();
};

// Sets the anchor offsets of the contacts that use elasto-plastic friction to their stage-start
// positions relative to their anchors; contacts without an anchor get zero.
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
    CFrictionAnchors<AnchorSet::Active> const& active,
    CActiveCollisions<kIsSync, TimeStep::Current>& activeCollisions) {
  FillFrictionAnchorOffsets(reg, index, MakeConstSpan(active.anchors), MakeSpan(activeCollisions));
}

// Rebuilds the next-step anchors from nonzero contact offsets after AdvanceFrictionAnchorOffsets.
// Uses the index as scratch and reuses Pending's capacity. A diverged last stage keeps Pending,
// which equals the anchors the step used.
void UpdateFrictionAnchors(
    FrictionAnchorReg reg,
    CConvergenceStatus const* convergence,
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current> const& asyncCollisions,
    CActiveCollisions</*kIsSync*/ true, TimeStep::Current> const& syncCollisions,
    CFrictionAnchors<AnchorSet::Pending>& outNext,
    CFrictionAnchorIndex& outIndex);

// Requests dropping, at the start of the next step, the friction anchors of an actor, both as
// colliding and collider (see CFrictionAnchorClearRequests). Does nothing without elasto-plastic
// friction.
void RequestFrictionAnchorClear(entt::registry& reg, entt::entity e);

// Emplaces empty friction anchor components on an actor with numSamples contact samples. Call
// only if ExperimentalEvalParams::elastoPlasticFriction is enabled.
void EmplaceFrictionAnchors(entt::registry& reg, entt::entity e, int numSamples);

// Prepares the friction anchors at the start of a step, before collision detection fills the
// anchor offsets: promotes Pending into Active if promotePending (false for a solve-free assembly,
// which keeps the Active anchors the last step used), applies the queued clear requests to both
// sets, and indexes Active.
void PrepareFrictionAnchors(entt::registry& reg, bool promotePending);

// Building blocks of PrepareFrictionAnchors, declared for tests that drive single actors.

// Copies the Pending anchors of a colliding actor into Active, without consuming them.
void PromotePendingFrictionAnchors(
    CFrictionAnchors<AnchorSet::Pending> const& pending,
    CFrictionAnchors<AnchorSet::Active>& outActive);

// Applies and clears the queued requests to reset friction preload, in both anchor sets.
void ApplyFrictionAnchorClearRequests(entt::registry& reg);

// Points the index to the group of each sample with anchors.
void IndexFrictionAnchors(
    CFrictionAnchors<AnchorSet::Active> const& active,
    CFrictionAnchorIndex& outIndex);

} // namespace mochi
