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

#include "mochi_sleep.h"

#include "mochi_common_components.h"
#include "mochi_constraint.h"
#include "mochi_contact.h"
#include "mochi_differentiable.h"
#include "mochi_group.h"
#include "mochi_island.h"
#include "mochi_scene_recorder.h"

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/span.h>

using namespace mochi;

void sleep::ValidateParams(
    entt::registry const& reg,
    experimental::SleepParams const& params,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      params.sleepThreshold > 0.0 && params.sleepThreshold <= 1.0,
      error,
      "Sleep threshold (SleepParams::sleepThreshold) must be in (0, 1].");
  MOCHI_ERROR_IF_NOT(
      params.minStepsBeforeSleep >= 2,
      error,
      "Minimum steps before sleep (SleepParams::minStepsBeforeSleep) must be at least 2.");
  MOCHI_ERROR_IF(
      params.canSleep && reg.try_ctx<TagDifferentiableScene const>() != nullptr,
      error,
      "Island sleeping (SleepParams::canSleep) is not supported in differentiable scenes.");
  MOCHI_ERROR_IF(
      params.canSleep && reg.try_ctx<TagIKScene const>() != nullptr,
      error,
      "Island sleeping (SleepParams::canSleep) is not supported in scenes owned by an IKSolver.");
}

void sleep::InitIsland(entt::registry& reg, entt::entity island) {
  reg.emplace<TagIslandIsAwake>(island);
  reg.emplace<CIslandSleepState>(island);
}

real sleep::StageRestValue(NewtonSolverStatus<real> const& status) {
  if (status.convergence != ConvergenceStatus::Converged) {
    return 0_r;
  }
  if (status.numIterDone == 0) {
    return 1_r;
  }
  MOCHI_ASSERT_VERBOSE(status.resNorm0 > 0_r, "Iterations imply a positive initial residual.");
  return status.resNorm / status.resNorm0;
}

void sleep::RecordStep(entt::registry& reg, entt::entity island, real restValue) {
  auto const& params = reg.ctx<CSleepParams const>();
  auto& state = reg.get<CIslandSleepState>(island);
  if (!params.canSleep || (restValue < params.sleepThreshold)) {
    state.numConsecutiveRestSteps = 0;
  } else if (state.numConsecutiveRestSteps < params.minStepsBeforeSleep) {
    ++state.numConsecutiveRestSteps;
  }
}

static void AddWakeUpRegionForStaticCollider(entt::registry& reg, entt::entity e) {
  // If the actor has a collider, then wake up dynamic actors that overlap its Aabb.
  // NOTE: This is a conservative heuristic. It may wake actors that would not actually interact
  // through contact.
  auto const* colliderInfo = reg.try_get<CColliderInfo const>(e);
  auto const* root = reg.try_get<CRootTransform const>(e);
  auto const* bounds = reg.try_get<CBoundingVolume const>(e);
  auto const* contactParams = reg.try_get<CContactParams const>(e);
  if (!colliderInfo || colliderInfo->type == ColliderType::None || !root || !bounds ||
      !contactParams) {
    return; // Cannot touch anything
  }
  // Same swept bounds as the broadphase uses for static colliders.
  reg.ctx<CWakeUpRequests>().regions.push_back(
      GetStaticColliderStepBounds(*root, *bounds, *contactParams));
}

static void AddWakeUpRegionsForStaticDescendants(entt::registry& reg, entt::entity e) {
  auto const* members = reg.try_get<CGroupMembers const>(e);
  if (!members) {
    return;
  }
  ForEachDescendant(reg, *members, [&](entt::entity child) {
    if (reg.all_of<TagStaticActor>(child)) {
      AddWakeUpRegionForStaticCollider(reg, child);
    }
  });
}

void sleep::WakeUp(entt::registry& reg, entt::entity e) {
  if (!reg.ctx<CSleepParams const>().canSleep || reg.ctx<CWakeUpRequests const>().wakeAll) {
    return;
  }
  if (reg.all_of<TagIsland>(e)) {
    reg.emplace_or_replace<TagWakeUp>(e);
    return;
  }
  if (reg.all_of<TagArticulatedActor>(e)) {
    AddWakeUpRegionsForStaticDescendants(reg, e);
  }
  if (reg.all_of<TagStaticActor>(e)) {
    AddWakeUpRegionForStaticCollider(reg, e);
    // Static articulated links also belong to their articulation's island.
    if (auto const* memberInfo = reg.try_get<CIslandMemberInfo const>(e)) {
      reg.emplace_or_replace<TagWakeUp>(memberInfo->island);
    }
    return;
  }
  if (auto const* memberInfo = reg.try_get<CIslandMemberInfo const>(e)) {
    reg.emplace_or_replace<TagWakeUp>(memberInfo->island);
    return;
  }
  if (auto const* constraintInfo = reg.try_get<CConstraintInfo const>(e)) {
    for (entt::entity actor : constraintInfo->actors) {
      // Wake the actor's island. Do not add wake regions for nested static links.
      if (auto const* memberInfo = reg.try_get<CIslandMemberInfo const>(actor)) {
        reg.emplace_or_replace<TagWakeUp>(memberInfo->island);
      }
    }
  }
}

void sleep::WakeAll(entt::registry& reg) {
  reg.ctx<CWakeUpRequests>().wakeAll = true;
}

static void EmplaceTagWakeUpOnOverlappingIslands(entt::registry& reg, Span<Aabb const> regions) {
  for (auto&& [e, stepBounds, memberInfo] :
       reg.view<CConservativeStepBounds const, CIslandMemberInfo const>().each()) {
    if (reg.all_of<TagWakeUp>(memberInfo.island)) {
      continue;
    }
    for (Aabb const& region : regions) {
      if (HasOverlap(stepBounds.worldAabb, region)) {
        reg.emplace<TagWakeUp>(memberInfo.island);
        break;
      }
    }
  }
}

// Tag every island that must wake up.
static void GatherWakeRequests(entt::registry& reg) {
  auto const& requests = reg.ctx<CWakeUpRequests const>();
  if (requests.wakeAll) {
    for (entt::entity island : reg.view<TagIsland>()) {
      reg.emplace_or_replace<TagWakeUp>(island);
    }
  } else if (!requests.regions.empty()) {
    EmplaceTagWakeUpOnOverlappingIslands(reg, MakeConstSpan(requests.regions));
  }
}

// Wake the tagged islands. They must then stay awake for at least minStepsBeforeSleep solves.
static void ApplyWakeRequests(entt::registry& reg) {
  for (auto&& [island, state] : reg.view<TagWakeUp, CIslandSleepState>().each()) {
    state.numConsecutiveRestSteps = 0;
    reg.emplace_or_replace<TagIslandIsAwake>(island);
  }
  reg.clear<TagWakeUp>();
  auto& requests = reg.ctx<CWakeUpRequests>();
  requests.wakeAll = false;
  requests.regions.clear();
}

static void PutActorToSleep(entt::registry& reg, entt::entity e) {
  if (auto* convergence = reg.try_get<CConvergenceStatus>(e)) {
    convergence->stageStatus = ConvergenceStatus::None;
    convergence->stepStatus = ConvergenceStatus::None;
  }
}

static void PutIslandToSleep(entt::registry& reg, entt::entity island) {
  MOCHI_ASSERT_VERBOSE(
      reg.all_of<TagIslandIsAwake>(island), "Expected an island that was awake to begin with");

  // Remove TagIslandIsAwake which marked this island as awake. Note that it is safe to do this
  // while iterating over a view that is filtered by TagIslandIsAwake.
  reg.remove<TagIslandIsAwake>(island);

  // Prepare island members for sleep
  for (entt::entity e : reg.get<CIslandDescendants const>(island).actors) {
    PutActorToSleep(reg, e);
  }
}

// Put islands to sleep if they have been at rest long enough.
static void SleepRestingIslands(entt::registry& reg) {
  // Sleeping islands are not recorded, so nothing sleeps while recording.
  auto const& params = reg.ctx<CSleepParams const>();
  if (!params.canSleep || reg.try_ctx<TagSceneRecordingEnabled>()) {
    return;
  }
  MOCHI_ASSERT_VERBOSE(
      !reg.try_ctx<TagDifferentiableScene>(), "Sleep requires non-differentiable.");

  for (auto&& [island, state] :
       reg.view<CIslandSleepState const, TagIslandIsAwake const>().each()) {
    if (state.numConsecutiveRestSteps >= params.minStepsBeforeSleep) {
      PutIslandToSleep(reg, island);
    }
  }
}

void sleep::PreStep(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();
  GatherWakeRequests(reg);
  ApplyWakeRequests(reg);
  SleepRestingIslands(reg);
}

bool sleep::IsIslandAwake(entt::registry const& reg, entt::entity island) {
  MOCHI_ASSERT_VERBOSE(reg.all_of<TagIsland>(island), "Not an island.");
  return reg.all_of<TagIslandIsAwake>(island);
}

void sleep::ForceSleep(entt::registry& reg, entt::entity island) {
  MOCHI_ASSERT_VERBOSE(reg.all_of<TagIsland>(island), "Not an island.");
  reg.remove<TagIslandIsAwake>(island);
  reg.remove<TagWakeUp>(island);
}

void sleep::InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CIslandSleepState>(reg);
  ecs::RegisterComponent<CSleepParams>(reg);
  ecs::RegisterComponent<CWakeUpRequests>(reg);
  ecs::RegisterComponent<TagIslandIsAwake>(reg);
  ecs::RegisterComponent<TagWakeUp>(reg);
  reg.set<CSleepParams>();
  reg.set<CWakeUpRequests>();
}
