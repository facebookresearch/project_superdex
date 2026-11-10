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

#include "mochi_contact_anchors.h"

#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/profile.h>

#include <algorithm>
#include <array>

using namespace mochi;

// Whether entry i of the index points to an anchor of the sample. Entries of samples without
// anchors may be stale, so the sample is checked.
[[nodiscard]] MOCHI_FORCE_INLINE static bool
IsAnchorOfSample(Span<FrictionAnchor const> anchors, int i, int sample) {
  return i != kSentinelIndex && i < isize(anchors) && anchors[i].sample == sample;
}

void mochi::FillFrictionAnchorOffsets(
    FrictionAnchorReg const& reg,
    CFrictionAnchorIndex const& index,
    Span<FrictionAnchor const> anchors,
    Span<ActiveCollision> outCollisions) {
  // Rebuild lays the anchors out in the order of the contacts it visited, which detection mostly
  // repeats: try the anchor after the last match before looking the sample up.
  int const numAnchors = isize(anchors);
  int hint = 0;
  for (auto& collision : outCollisions) {
    if (!HasFrictionAnchors(reg, collision.colliderEntity)) {
      continue;
    }
    auto& result = collision.collisionResult;
    auto const colliderId = entt::to_integral(collision.colliderEntity);
    int const numContacts = isize(result.sampleIndices);
    MOCHI_ASSERT_VERBOSE(
        isize(result.posCollidingStageStart) == numContacts,
        "Collision detection must evaluate the stage-start contact positions.");
    result.anchorOffset.resize_noinit(numContacts);
    for (int c = 0; c < numContacts; ++c) {
      int const s = result.sampleIndices[c];
      if (hint < numAnchors && anchors[hint].sample == s && anchors[hint].collider == colliderId) {
        result.anchorOffset[c] = result.posCollidingStageStart[c] - anchors[hint++].point;
        continue;
      }
      Real3 offset{};
      for (int i = index[s]; IsAnchorOfSample(anchors, i, s); ++i) {
        if (anchors[i].collider == colliderId) {
          offset = result.posCollidingStageStart[c] - anchors[i].point;
          hint = i + 1;
          break;
        }
      }
      result.anchorOffset[c] = offset;
    }
  }
}

void mochi::UpdateFrictionAnchors(
    FrictionAnchorReg reg,
    CConvergenceStatus const* convergence,
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current> const& asyncCollisions,
    CActiveCollisions</*kIsSync*/ true, TimeStep::Current> const& syncCollisions,
    CFrictionAnchors<AnchorSet::Pending>& outNext,
    CFrictionAnchorIndex& outIndex) {
  // A stopped solve is accepted. A diverged last stage keeps Pending, which equals the anchors the
  // step used: the promotion copied it, and resets drop the same anchors from both sets.
  if (convergence && convergence->stageStatus == ConvergenceStatus::Diverged) {
    return;
  }

  auto& pending = outNext.anchors;
  std::array<Span<ActiveCollision const>, 2> const allCollisions{
      MakeConstSpan(asyncCollisions), MakeConstSpan(syncCollisions)};

  // 1. Append the anchors in visiting order, using the index as scratch to locate the first anchor
  // of each sample. Sized to the contacts that use elasto-plastic friction, which bounds the number
  // of anchors.
  int maxAnchors = 0;
  for (auto const collisions : allCollisions) {
    for (auto const& collision : collisions) {
      if (HasFrictionAnchors(reg, collision.colliderEntity)) {
        maxAnchors += isize(collision.collisionResult.sampleIndices);
      }
    }
  }
  pending.resize_noinit(maxAnchors);
  int numAnchors = 0;
  bool isGrouped = true;
  for (auto const collisions : allCollisions) {
    for (auto const& collision : collisions) {
      if (!HasFrictionAnchors(reg, collision.colliderEntity)) {
        continue;
      }
      auto const& result = collision.collisionResult;
      auto const colliderId = entt::to_integral(collision.colliderEntity);
      int const numContacts = isize(result.sampleIndices);
      MOCHI_ASSERT_VERBOSE(
          isize(result.anchorOffset) == numContacts && isize(result.posColliding) == numContacts,
          "Collision detection must fill the friction anchor offsets and contact positions.");
      for (int i = 0; i < numContacts; ++i) {
        // Contacts with zero offset lose their anchor.
        Real3 const& offset = result.anchorOffset[i];
        if (offset == Real3{}) {
          continue;
        }
        int const sampleIdx = result.sampleIndices[i];
        MOCHI_ASSERT_VERBOSE(
            sampleIdx >= 0 && sampleIdx < isize(outIndex), "Invalid sample index.");
        // The first anchor of a sample starts its group; later ones must directly follow it.
        Span<FrictionAnchor const> const appended(pending.data(), static_cast<size_t>(numAnchors));
        if (!IsAnchorOfSample(appended, outIndex[sampleIdx], sampleIdx)) {
          outIndex[sampleIdx] = numAnchors;
        } else if (pending[numAnchors - 1].sample != sampleIdx) {
          isGrouped = false; // The sample's anchors are not contiguous.
        }
        pending[numAnchors++] = FrictionAnchor{
            .sample = sampleIdx, .collider = colliderId, .point = result.posColliding[i] - offset};
      }
    }
  }
  pending.resize(numAnchors);

  // 2. If the anchors of a sample are not contiguous, which requires anchors on several rigid
  // colliders, group them by sample. index[s] is the position of the first anchor of s, unique per
  // sample, so sorting by it groups the anchors of each sample, in order of first appearance. The
  // order within a sample does not matter: it has at most one anchor per collider.
  if (!isGrouped) {
    std::ranges::sort(pending, [&](FrictionAnchor const& a, FrictionAnchor const& b) {
      return outIndex[a.sample] < outIndex[b.sample];
    });
  }
}

void mochi::PromotePendingFrictionAnchors(
    CFrictionAnchors<AnchorSet::Pending> const& pending,
    CFrictionAnchors<AnchorSet::Active>& outActive) {
  outActive.anchors.clear();
  outActive.anchors.append(pending.anchors);
}

void mochi::IndexFrictionAnchors(
    CFrictionAnchors<AnchorSet::Active> const& active,
    CFrictionAnchorIndex& outIndex) {
  // Only the samples with anchors are set; other entries may be stale, and lookups reject them.
  auto const& anchors = active.anchors;
  for (int i = 0; i < isize(anchors); ++i) {
    if (i == 0 || anchors[i - 1].sample != anchors[i].sample) {
      outIndex[anchors[i].sample] = i;
    }
  }
}

void mochi::RequestFrictionAnchorClear(entt::registry& reg, entt::entity e) {
  if (auto* requests = reg.try_ctx<CFrictionAnchorClearRequests>()) {
    requests->entities.push_back(entt::to_integral(e));
  }
}

void mochi::ApplyFrictionAnchorClearRequests(entt::registry& reg) {
  auto* requests = reg.try_ctx<CFrictionAnchorClearRequests>();
  if (!requests || requests->entities.empty()) {
    return;
  }

  MOCHI_PROFILE_SCOPE();

  // Drop all the anchors of requested actors as colliding, from both sets: a promotion before the
  // next solve rebuilds Pending, e.g., while the island sleeps, must not bring them back. Keep the
  // requested colliders with friction anchors for the second pass.
  auto& entities = requests->entities;
  int numColliders = 0;
  for (int i = 0; i < isize(entities); ++i) {
    auto const e = static_cast<entt::entity>(entities[i]);
    // A requested entity may have been destroyed since its request.
    if (!reg.valid(e)) {
      continue;
    }
    if (auto* active = reg.try_get<CFrictionAnchors<AnchorSet::Active>>(e)) {
      active->anchors.clear();
    }
    if (auto* pending = reg.try_get<CFrictionAnchors<AnchorSet::Pending>>(e)) {
      pending->anchors.clear();
    }
    if (HasFrictionAnchors(reg, e)) {
      entities[numColliders++] = entities[i];
    }
  }
  entities.resize(numColliders);
  if (numColliders == 0) {
    return;
  }
  SortAndRemoveDuplicates(entities);

  // Drop the anchors of all actors on the requested colliders. The stable compaction keeps the
  // anchors of each sample contiguous.
  auto const isRequested = [&](uint32_t collider) {
    return std::ranges::binary_search(entities, collider);
  };
  auto const dropRequested = [&](DynamicArray<FrictionAnchor>& anchors) {
    if (anchors.empty()) {
      return;
    }
    // Consecutive anchors often share a collider: search only when it changes.
    uint32_t lastCollider = anchors[0].collider;
    bool removeLast = isRequested(lastCollider);
    int const numAnchors = isize(anchors);
    int numKept = 0;
    for (int i = 0; i < numAnchors; ++i) {
      uint32_t const collider = anchors[i].collider;
      if (collider != lastCollider) {
        lastCollider = collider;
        removeLast = isRequested(collider);
      }
      if (!removeLast) {
        anchors[numKept++] = anchors[i];
      }
    }
    anchors.resize(numKept);
  };
  auto view = reg.view<CFrictionAnchors<AnchorSet::Active>, CFrictionAnchors<AnchorSet::Pending>>();
  for (auto&& [e, active, pending] : view.each()) {
    dropRequested(active.anchors);
    dropRequested(pending.anchors);
  }
  entities.clear();
}

void mochi::PrepareFrictionAnchors(entt::registry& reg, bool promotePending) {
  if (promotePending) {
    ecs::InvokeForEachGlobal(&PromotePendingFrictionAnchors, reg);
  }
  ApplyFrictionAnchorClearRequests(reg);
  ecs::InvokeForEachGlobal(&IndexFrictionAnchors, reg);
}

void mochi::EmplaceFrictionAnchors(entt::registry& reg, entt::entity e, int numSamples) {
  reg.emplace<CFrictionAnchors<AnchorSet::Active>>(e);
  reg.emplace<CFrictionAnchors<AnchorSet::Pending>>(e);
  reg.emplace<CFrictionAnchorIndex>(e, numSamples);
}
