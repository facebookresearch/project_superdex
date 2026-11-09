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

#include <algorithm>
#include <array>

using namespace mochi;

// Whether entry i of the index points to an anchor of the sample. Entries of samples without
// anchors may be stale, so the sample is checked.
[[nodiscard]] MOCHI_FORCE_INLINE static bool
IsAnchorOfSample(Span<FrictionAnchor const> anchors, int i, int sample) {
  return i != kSentinelIndex && i < isize(anchors) && anchors[i].sample == sample;
}

// Points the index to the start of the group of each sample in anchors grouped by sample. Only the
// samples with anchors are set; other entries may be stale, and lookups reject them.
static void IndexAnchorGroups(Span<FrictionAnchor const> anchors, CFrictionAnchorIndex& outIndex) {
  for (int i = 0; i < isize(anchors); ++i) {
    if (i == 0 || anchors[i - 1].sample != anchors[i].sample) {
      outIndex[anchors[i].sample] = i;
    }
  }
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
    CFrictionAnchors& outAnchorsComp,
    CFrictionAnchorIndex& outIndex) {
  // A stopped solve is still the accepted state; only a diverged last stage, which resets the
  // actor, keeps the previous anchors.
  if (convergence && convergence->stageStatus == ConvergenceStatus::Diverged) {
    return;
  }

  auto& anchors = outAnchorsComp.anchors;
  std::array<Span<ActiveCollision const>, 2> const allCollisions{
      MakeConstSpan(asyncCollisions), MakeConstSpan(syncCollisions)};

  // 1. Append the anchors in visiting order, setting index[s] at the first anchor of each sample.
  // Sized to the contacts that use elasto-plastic friction, which bounds the number of anchors.
  int maxAnchors = 0;
  for (auto const collisions : allCollisions) {
    for (auto const& collision : collisions) {
      if (HasFrictionAnchors(reg, collision.colliderEntity)) {
        maxAnchors += isize(collision.collisionResult.sampleIndices);
      }
    }
  }
  anchors.resize_noinit(maxAnchors);
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
        Span<FrictionAnchor const> const appended(anchors.data(), static_cast<size_t>(numAnchors));
        if (!IsAnchorOfSample(appended, outIndex[sampleIdx], sampleIdx)) {
          outIndex[sampleIdx] = numAnchors;
        } else if (anchors[numAnchors - 1].sample != sampleIdx) {
          isGrouped = false; // The sample's anchors are not contiguous.
        }
        anchors[numAnchors++] = FrictionAnchor{
            .sample = sampleIdx, .collider = colliderId, .point = result.posColliding[i] - offset};
      }
    }
  }
  anchors.resize(numAnchors);

  // 2. If the anchors of each sample are contiguous, they are grouped and indexed already. That is
  // always the case when no sample has anchors on several rigid colliders.
  if (isGrouped) {
    return;
  }

  // 3. Otherwise, group them by sample. index[s] is the position of the first anchor of s, unique
  // per sample, so sorting by it groups the anchors of each sample, in order of first appearance.
  // The order within a sample does not matter: it has at most one anchor per collider. Then point
  // the index to the new group starts.
  std::ranges::sort(anchors, [&](FrictionAnchor const& a, FrictionAnchor const& b) {
    return outIndex[a.sample] < outIndex[b.sample];
  });
  IndexAnchorGroups(anchors, outIndex);
}

void mochi::ReindexFrictionAnchors(
    CFrictionAnchors const& anchorsComp,
    CFrictionAnchorIndex& outIndex) {
  IndexAnchorGroups(anchorsComp.anchors, outIndex);
}

void mochi::EmplaceFrictionAnchors(entt::registry& reg, entt::entity e, int numSamples) {
  reg.emplace<CFrictionAnchors>(e);
  reg.emplace<CFrictionAnchorIndex>(e, numSamples);
}
