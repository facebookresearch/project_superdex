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

#include "mochi_island.h"

#include "mochi_common_components.h"
#include "mochi_contact.h"
#include "mochi_contact_filter.h"
#include "mochi_differentiable.h"
#include "mochi_ecs.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"

#include <mochi_core/utils/array_utils.h>

#include <algorithm>
#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace mochi;

namespace {

// Global context component used in this cpp file to reduce re-allocation of temporary data storage.
struct CTempIslandPartitioningData {
  std::vector<entt::entity> islands;
  std::vector<int> actorPartitions;
};

// Global context component with special support for unit tests
struct CIslandTestConfig {
  std::function<void()> preIslandUpdateCallback;
  std::function<void()> postIslandUpdateCallback;
};

// If everything is forced to simulate using a single island, then this is it.
struct CGlobalIsland {
  entt::entity island = entt::null;
  bool forceSingleIslandByUserRequest = false;
};

} // namespace

namespace mochi::island {
void InitializeOnce(entt::registry& reg) {
  // ECS Component Types:
  ecs::RegisterComponent<CGlobalIsland>(reg);
  ecs::RegisterComponent<CIslandDescendants>(reg);
  ecs::RegisterComponent<CIslandDofInfo>(reg);
  ecs::RegisterComponent<CIslandMemberInfo>(reg);
  ecs::RegisterComponent<CIslandMembers>(reg);
  ecs::RegisterComponent<CIslandSolverStats>(reg);
  ecs::RegisterComponent<CIslandPreconditioner>(reg);
  ecs::RegisterComponent<CIslandTestConfig>(reg);
  ecs::RegisterComponent<CTempIslandPartitioningData>(reg);
  ecs::RegisterComponent<TagIsland>(reg);
  ecs::RegisterComponent<TagIslandCompositionChanged>(reg);
  ecs::RegisterComponent<TagForceSingleIsland>(reg);

  // Init global context component
  auto& data = reg.set<CTempIslandPartitioningData>();
  reg.set<CGlobalIsland>();

  // Reserve memory for a modest number of islands up front
  data.islands.reserve(64);
  data.actorPartitions.reserve(64);
}

[[nodiscard]] bool ShouldRunSingleThreaded(
    entt::registry const& reg,
    entt::entity island,
    CIslandDescendants const& descendants) {
  MOCHI_ASSERT(reg.all_of<TagIsland>(island), "Expected an island.");

  // Covers typical rigid and articulated islands while excluding pathological many-actor islands.
  constexpr int kMaxNumDofs = 128;

  // Conservatively excludes contact-heavy islands. Some contact-heavy islands remain faster
  // single-threaded even above this cutoff.
  constexpr int kMaxNumContactSamples = 6000;

  // Check the DoF threshold.
  if (reg.get<CIslandDofInfo const>(island).dofsSize >= kMaxNumDofs) {
    return false;
  }

  // Rigid, static, and compound actors only (compounds include articulated bodies).
  // Compound members are flattened into descendants.actors, so each is checked individually.
  for (entt::entity e : descendants.actors) {
    MOCHI_ASSERT_VERBOSE(
        !reg.all_of<TagArticulatedActor>(e) || reg.all_of<TagCompoundActor>(e),
        "Expected articulated actors to be compounds.");
    if (!reg.any_of<TagRigidActor, TagStaticActor, TagCompoundActor>(e)) {
      return false;
    }
  }

  // Check the contact-sample threshold.
  int numSamples = 0;
  for (entt::entity e : descendants.actors) {
    if (!reg.all_of<TagUseContact, CContactSamples<TimeStep::Current>>(e)) {
      continue;
    }
    auto const& samples = reg.get<CContactSamples<TimeStep::Current> const>(e);
    // CContactSamples::activePositions is only populated during the step, so derive the active
    // sample count from CActiveBoundaryFaces, which is up-to-date even before the first step.
    auto const* activeBoundaryFaces = reg.try_get<CActiveBoundaryFaces const>(e);
    int const numActorSamples = activeBoundaryFaces
        ? isize(activeBoundaryFaces->ViewIndices()) * activeBoundaryFaces->NumQuadPerFace()
        : isize(samples.positions);
    numSamples += numActorSamples;
    if (numSamples >= kMaxNumContactSamples) {
      return false;
    }
  }
  return true;
}

} // namespace mochi::island

entt::entity island::CreateEmpty(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();
  auto island = reg.create();

  // Island-specific components:
  reg.emplace<TagIsland>(island);
  reg.emplace<CIslandMembers>(island);
  reg.emplace<CIslandDescendants>(island);
  reg.emplace<CIslandDofInfo>(island);
  reg.emplace<CIslandContactSnle>(island);
  reg.emplace<CIslandPreconditioner>(
      island, std::make_shared<PreconditionerRecyclingManager<real>>());
  reg.emplace<CIslandSolverStats>(island);
  if (reg.try_ctx<TagDifferentiableScene>()) {
    InitDifferentiableIsland(reg, island);
  }

  // Let other systems (e.g. debug draw) know that this island is ready for inspection.
  reg.emplace<TagFullyInitialized>(island);

  return island;
}

static void
AddIslandMemberInfo(entt::registry& reg, entt::entity island, entt::entity actor, bool isNested) {
  // Add CIslandMemberInfo to actor
  auto& memberInfo = reg.emplace<CIslandMemberInfo>(actor);
  memberInfo.island = island;
  memberInfo.isNested = isNested;

  // If actor is a compound, then also add the component to every nested member
  auto const* children = reg.try_get<CGroupMembers const>(actor);
  if (children) {
    ForEachDescendant(reg, *children, [&](entt::entity e) {
      auto& nestedInfo = reg.emplace<CIslandMemberInfo>(e);
      nestedInfo.island = island;
      nestedInfo.isNested = true;
    });
  }
}

void island::AddActor(entt::registry& reg, entt::entity island, entt::entity actor) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(reg.all_of<TagIsland>(island), "Not an island");
  MOCHI_ASSERT(reg.all_of<CActorInfo>(actor), "Not an actor");
  MOCHI_ASSERT(
      !reg.any_of<CIslandMemberInfo>(actor),
      "Actor must first be removed from its previous island");
  MOCHI_ASSERT(
      !reg.any_of<CGroupMemberInfo>(actor),
      "An actor in a group cannot be added to an island directly.");

  // Add actor to CIslandMembers
  auto& members = reg.get<CIslandMembers>(island);
  MOCHI_ASSERT(!Contains(members.actors, actor));
  members.actors.push_back(actor);

  // Add CIslandMemberInfo to all actors that are joining the island, including nested ones.
  AddIslandMemberInfo(reg, island, actor, false);

  // Update derived data on the next step
  reg.emplace_or_replace<TagIslandCompositionChanged>(island);
}

void island::AddNestedActor(
    entt::registry& reg,
    entt::entity island,
    entt::entity parentCompound,
    entt::entity nestedChild) {
  MOCHI_ASSERT(reg.all_of<TagIsland>(island), "Not an island");
  MOCHI_ASSERT(reg.all_of<TagCompoundActor>(parentCompound), "Not a compound");
  MOCHI_ASSERT(
      reg.get<CIslandMemberInfo const>(parentCompound).island == island, "Not an island member");
  MOCHI_ASSERT(
      reg.get<CGroupMemberInfo const>(nestedChild).group == parentCompound,
      "Not a compound member");

  // Add CIslandMemberInfo to the nested child and any nested grandchildren.
  AddIslandMemberInfo(reg, island, nestedChild, true);

  // Update derived data on the next step
  reg.emplace_or_replace<TagIslandCompositionChanged>(island);
}

entt::entity island::CreateForActor(entt::registry& reg, entt::entity actor) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(reg.try_get<CActorInfo const>(actor), "Not an actor");
  entt::entity island = island::CreateEmpty(reg);
  island::AddActor(reg, island, actor);
  return island;
}

void island::RemoveActor(entt::registry& reg, entt::entity actor) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(reg.try_get<CActorInfo const>(actor), "Not an actor");
  auto const* info = reg.try_get<CIslandMemberInfo const>(actor);
  if (info == nullptr) {
    return; // Not in an island
  }

  entt::entity island = info->island;
  MOCHI_ASSERT(reg.all_of<TagIsland>(island), "Not an island");
  MOCHI_ASSERT(!reg.try_get<CActorInfo const>(island), "Island cannot be an actor");

  // Remove the actor from CIslandMembers unless it was nested within a compound (in which case it
  // shouldn't be listed there).
  auto& members = reg.get<CIslandMembers>(island);
  auto foundIt = std::find(members.actors.begin(), members.actors.end(), actor);
  if (info->isNested) {
    MOCHI_ASSERT(foundIt == members.actors.end(), "CIslandMemberInfo::isNested was incorrect");
  } else {
    MOCHI_ASSERT(foundIt != members.actors.end(), "Actor not in island");
    EraseIndexUnordered(members.actors, foundIt - members.actors.begin());
  }

  // Remove CIslandMemberInfo from all actors leaving the island, including nested onces.
  ForEntityAndEachDescendant(reg, actor, [&](entt::entity e) { reg.erase<CIslandMemberInfo>(e); });

  if (members.actors.empty() && members.constraints.empty()) {
    // Destroy the empty island
    reg.destroy(island);
  } else {
    // Update derived data on the next step
    reg.emplace_or_replace<TagIslandCompositionChanged>(island);
  }
}

void island::InitDifferentiableIsland(entt::registry& reg, entt::entity island) {
  reg.emplace_or_replace<CIslandDerivedStateInfo>(island);
  reg.emplace_or_replace<CIslandDiffInputInfo>(island);
}

// Remove an actor from srcIsland and add it to dstIsland.
// Faster than AddActor + RemoveActor because components are not destroyed.
static void TransferActor(
    entt::registry& reg,
    entt::entity actor,
    int srcIndex,
    entt::entity srcIsland,
    entt::entity dstIsland) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(srcIsland != dstIsland);
  MOCHI_ASSERT_VERBOSE(reg.all_of<CActorInfo>(actor), "Not an actor");
  MOCHI_ASSERT_VERBOSE(reg.all_of<TagIsland>(srcIsland), "Not an island");
  MOCHI_ASSERT_VERBOSE(reg.all_of<TagIsland>(dstIsland), "Not an island");

  // Update CIslandMembers on both islands
  auto& srcMembers = reg.get<CIslandMembers>(srcIsland);
  auto& dstMembers = reg.get<CIslandMembers>(dstIsland);
  MOCHI_ASSERT_VERBOSE(srcIndex < isize(srcMembers.actors));
  MOCHI_ASSERT_VERBOSE(srcMembers.actors[srcIndex] == actor);
  EraseIndexUnordered(srcMembers.actors, srcIndex);
  MOCHI_ASSERT_VERBOSE(!Contains(dstMembers.actors, actor));
  dstMembers.actors.push_back(actor);

  // Update CIslandMemberInfo on all actors being transferred, including nested ones.
  ForEntityAndEachDescendant(reg, actor, [&](entt::entity e) {
    auto& memberInfo = reg.get<CIslandMemberInfo>(e);
    MOCHI_ASSERT_VERBOSE(memberInfo.island == srcIsland);
    MOCHI_ASSERT_VERBOSE(memberInfo.isNested == (e != actor));
    memberInfo.island = dstIsland;
  });

  // Update derived data on the next step
  reg.emplace_or_replace<TagIslandCompositionChanged>(srcIsland);
  reg.emplace_or_replace<TagIslandCompositionChanged>(dstIsland);
}

// Merge the contents of srcIsland into dstIsland. Then destroy srcIsland.
static void MergeIslandPair(entt::registry& reg, entt::entity srcIsland, entt::entity dstIsland) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(srcIsland != dstIsland);

  // Move actors and constraints to dstIsland
  auto const& srcMembers = reg.get<CIslandMembers const>(srcIsland);
  auto& dstMembers = reg.get<CIslandMembers>(dstIsland);
  Append(dstMembers.actors, srcMembers.actors);
  Append(dstMembers.constraints, srcMembers.constraints);

  // Update CIslandMemberInfo for each actor that moved, including nested ones.
  ForEachDescendant(reg, srcMembers, [&](entt::entity e) {
    auto& memberInfo = reg.get<CIslandMemberInfo>(e);
    MOCHI_ASSERT_VERBOSE(memberInfo.island == srcIsland);
    memberInfo.island = dstIsland;
  });

  reg.destroy(srcIsland);

  // Update derived data on the next step
  reg.emplace_or_replace<TagIslandCompositionChanged>(dstIsland);
}

static void TryMergeIslands(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();
  // If any dynamic actor has a potential collider
  for (auto&& [e, syncColls, membership] :
       reg.view<CConservativePotentialColliders<ContactType::Sync> const, CIslandMemberInfo const>()
           .each()) {
    entt::entity dstIsland = membership.island;
    for (auto const& col : syncColls) {
      auto const* memberInfo = reg.try_get<CIslandMemberInfo const>(col.entity);
      if (memberInfo) {
        entt::entity srcIsland = memberInfo->island;
        if (srcIsland != dstIsland) {
          MergeIslandPair(reg, srcIsland, dstIsland);
        }
      }
    }
  }
}

static void TrySplitIslands(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();

  // Recycle memory stored on CTempIslandPartitioningData
  auto& ctx = reg.ctx<CTempIslandPartitioningData>();
  auto& islands = ctx.islands;
  islands.clear();

  // Find all islands that might need to split
  for (auto&& [e, members] : reg.view<CIslandMembers>().each()) {
    if (members.actors.size() > 1) {
      islands.push_back(e);
    }
  }

  // For each island
  int const numIslandsBeforeAnySplits = isize(islands);
  for (int islandIdx = 0; islandIdx < numIslandsBeforeAnySplits; ++islandIdx) {
    entt::entity island = islands[islandIdx];
    auto const& members = reg.get<CIslandMembers>(island).actors;
    auto& actorPartitions = ctx.actorPartitions;

    // actorPartitions[i] is the partition index for actor i
    actorPartitions.clear();
    actorPartitions.resize(members.size(), -1);

    // We will partition the actors by walking the graph of potential colliders.
    // Actors in the same partition must remain in the same island.
    int nextPartitionId = 0;
    int numUnusedPartitionIds = 0;

    // For each member
    for (int i = 0; i < isize(members); ++i) {
      entt::entity iActor = members[i];
      if (actorPartitions[i] == -1) {
        // Start a new partition
        MOCHI_ASSERT_VERBOSE(!Contains(actorPartitions, nextPartitionId));
        actorPartitions[i] = nextPartitionId++;
      }

      // Find all entities that might collide with iActor, or with any of iActor's descendants.
      ForEntityAndEachDescendant(reg, iActor, [&](entt::entity e) {
        if (auto const* potentialColliders =
                reg.try_get<CConservativePotentialColliders<ContactType::Sync>>(e)) {
          for (auto const& jCollider : *potentialColliders) {
            entt::entity jActor = jCollider.entity;

            auto const* jInfo = &reg.get<CIslandMemberInfo const>(jActor);
            MOCHI_ASSERT_VERBOSE(
                jInfo->island == island, "These islands should have already merged");

            // If the other actor is nested in a group, then find the parent group instead.
            while (jInfo->isNested) {
              jActor = reg.get<CGroupMemberInfo const>(jActor).group;
              jInfo = &reg.get<CIslandMemberInfo const>(jActor);
              MOCHI_ASSERT_VERBOSE(
                  jInfo->island == island, "These islands should have already merged");
            }

            if (jActor == iActor) {
              continue; // Self contact
            }

            // Find the index of jActor in this island. This will be fast for small islands.
            // If islands become large, then it might be worth caching this index with
            // CIslandMemberInfo.
            auto itr = std::find(members.begin(), members.end(), jActor);
            MOCHI_ASSERT_VERBOSE(itr != members.end(), "CIslandMemberInfo must be out-of-sync");
            int j = static_cast<int>(itr - members.begin());

            if (actorPartitions[j] != actorPartitions[i]) {
              if (actorPartitions[j] == -1) {
                // Add it to the same partition
                actorPartitions[j] = actorPartitions[i];
              } else {
                // Merge the greater partition index into the lesser one
                auto replaceThis = Max(actorPartitions[i], actorPartitions[j]);
                auto withThis = Min(actorPartitions[i], actorPartitions[j]);
                std::replace(actorPartitions.begin(), actorPartitions.end(), replaceThis, withThis);
                ++numUnusedPartitionIds;
              }
            }
          }
        }
      });
    }

    // Partition IDs are allocated sequentially, so nextPartitionId should be the
    // number of unique IDs unless some were merged in the loop above.
    int const numUniquePartitions = nextPartitionId - numUnusedPartitionIds;
#if MOCHI_ASSERT_VERBOSE_ENABLED
    {
      // Check numUniquePartitions in Debug builds
      std::unordered_set<int> uniqueSet;
      for (int id : actorPartitions) {
        uniqueSet.insert(id);
      }
      MOCHI_ASSERT_VERBOSE(
          numUniquePartitions == isize(uniqueSet), "numUniquePartitions is incorrect");
    }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

    // If we found more than one partition, then the island should split because actors in one
    // partition have no synchronous contact dependency with islands in other groups.
    if (numUniquePartitions > 1) {
      MOCHI_ASSERT_VERBOSE(
          actorPartitions[0] == 0, "The first actor always goes in the first partition");
      // Resize the islands array based on the number of partition IDs allocated. The number
      // actually used (after mergers) may be less. If so, we'll prune them at the end.
      int const maxNumNewIslands = nextPartitionId - 1;
      int const prevNumIslands = isize(islands);
      islands.resize(prevNumIslands + maxNumNewIslands, entt::null);
      // Transfer the members to new islands according to their partition IDs
      for (int i = isize(members) - 1; i > 0; --i) {
        MOCHI_ASSERT(actorPartitions[i] >= 0);
        if (actorPartitions[i] > 0) {
          int newIslandIndex = prevNumIslands + actorPartitions[i] - 1;
          MOCHI_ASSERT(newIslandIndex < isize(islands));
          auto& newIsland = islands[newIslandIndex];
          if (newIsland == entt::null) {
            newIsland = island::CreateEmpty(reg);
          }
          TransferActor(reg, reg.get<CIslandMembers>(island).actors[i], i, island, newIsland);
        }
      }
      // Prune any entt::null values from the islands array. These correspond to the unused (merged)
      // partition IDs.
      islands.erase(
          std::remove(islands.begin() + prevNumIslands, islands.end(), entt::null), islands.end());
      [[maybe_unused]] int numNewIslands = isize(islands) - prevNumIslands;
      MOCHI_ASSERT(numNewIslands == numUniquePartitions - 1);
    }
  }
}

static void UpdateIslandDofInfo(
    entt::registry& reg,
    CIslandMembers const& members,
    CIslandDofInfo& outDofInfo) {
  // Assign each member a DOF offset
  int dofsSizeFound = 0;
  int poseSizeFound = 0;
  for (auto actor : members.actors) {
    auto& actorDof = reg.get<CDofOffset>(actor);
    if (actorDof.poseOffset != poseSizeFound || actorDof.dofsOffset != dofsSizeFound) {
      actorDof.dofsOffset = dofsSizeFound;
      actorDof.poseOffset = poseSizeFound;
      reg.emplace_or_replace<TagGlobalDofsChanged>(actor); // Notify actor
    }
    dofsSizeFound += reg.get<CActorDofInfo const>(actor).dofsSize;
    poseSizeFound += reg.get<CActorDofInfo const>(actor).poseSize;
  }
  // Store the total
  outDofInfo.dofsSize = dofsSizeFound;
  outDofInfo.poseSize = poseSizeFound;
}

static void UpdateIslandDifferentiabilityInfo(
    entt::registry& reg,
    CIslandMembers const& members,
    CIslandDerivedStateInfo& outDofInfo,
    CIslandDiffInputInfo& outInputInfo) {
  // Assign each member a DOF and input offset
  int dofsSizeFound = 0;
  int inputSizeFound = 0;
  for (auto actor : members.actors) {
    auto& actorDof = reg.get<CDerivedStateOffset>(actor);
    if (actorDof.dofsOffset != dofsSizeFound) {
      actorDof.dofsOffset = dofsSizeFound;
    }
    dofsSizeFound += reg.get<CActorDerivedStateInfo const>(actor).dofsSize;
    auto& actorInput = reg.get<CDiffInputOffset>(actor);
    if (actorInput.dofsOffset != inputSizeFound) {
      actorInput.dofsOffset = inputSizeFound;
    }
    inputSizeFound += reg.get<CActorDiffInputInfo const>(actor).dofsSize;
  }
  // Store the total
  outDofInfo.dofsSize = dofsSizeFound;
  outInputInfo.size = inputSizeFound;
}

// If we are required to force all entities into a single island, then this function will make it so
// and return true.
static bool UpdateForceGlobalIsland(entt::registry& reg) {
  auto& global = reg.ctx<CGlobalIsland>();

  // We need one global island if the user requested it, or TagForceSingleIsland is present.
  bool needsGlobalIsland =
      global.forceSingleIslandByUserRequest || !reg.view<TagForceSingleIsland>().empty();

  if (needsGlobalIsland) {
    // Ensure that the global island exists
    if (!reg.valid(global.island)) {
      global.island = island::CreateEmpty(reg);
    }

    // Merge any other islands into it
    for (auto&& [srcIsland] : reg.view<TagIsland>().each()) {
      if (srcIsland != global.island) {
        MergeIslandPair(reg, srcIsland, global.island);
      }
    }
  } else {
    // If there was a global island, then forget it. It may get split like any other island. It will
    // be cleaned up automatically when it becomes empty (again like any other island).
    global.island = entt::null;
  }

  return needsGlobalIsland;
}

// Sort CIslandMembers
static void SortIslandMembers(CIslandMembers& outMembers) {
  // Sort actors and constraints by entity value for deterministic ordering
  auto sortMembers = [](std::vector<entt::entity>& members) {
    std::sort(members.begin(), members.end(), [&](entt::entity a, entt::entity b) {
      return static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
    });
  };
  sortMembers(outMembers.actors);
  sortMembers(outMembers.constraints);
}

// Update CIslandDescendants based on CIslandMembers
static void UpdateIslandDescendants(
    entt::registry const& reg,
    CIslandMembers const& members,
    CIslandDescendants& outDescendants) {
  outDescendants.Clear();
  ForEachDescendant(reg, members, [&](auto e) {
    // Add to list of all entities
    outDescendants.actors.push_back(e);

    // Add entity to type-specific lists
    if (reg.any_of<TagCompoundActor>(e)) {
      outDescendants.compoundActors.push_back(e);
    }
    if (reg.any_of<TagRigidActor>(e)) {
      outDescendants.rigidActors.push_back(e);
    }
    if (reg.any_of<TagSoftActor>(e)) {
      outDescendants.softActors.push_back(e);
    }
    if (reg.any_of<TagShellActor>(e)) {
      outDescendants.shellActors.push_back(e);
    }
    if (reg.any_of<TagRodActor>(e)) {
      outDescendants.rodActors.push_back(e);
    }
    if (reg.any_of<TagNestedSoftActor>(e)) {
      outDescendants.nestedSoftActors.push_back(e);
    }
    if (reg.any_of<TagBlendedActor>(e)) {
      outDescendants.blendedActors.push_back(e);
    }
  });
}

void island::PreStep(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();

  // Unit test support
  auto const* testConfig = reg.try_ctx<CIslandTestConfig const>();
  if (testConfig && testConfig->preIslandUpdateCallback) {
    testConfig->preIslandUpdateCallback();
  }

  // In some cases, we may be forced to merge all entities into a single island.
  bool hasGlobalIsland = UpdateForceGlobalIsland(reg);

  if (!hasGlobalIsland) {
    // Merge islands if they contain actors that might interact with ContactType::Sync
    TryMergeIslands(reg);

    // Split islands if we can find a subset of the actors that do not interact with the others via
    // ContactType::Sync
    TrySplitIslands(reg);
  }

  // Update derived data for islands that have change
  for (auto&& [island, members, descendants] :
       reg.view<TagIslandCompositionChanged, CIslandMembers, CIslandDescendants>().each()) {
    // First sort members
    SortIslandMembers(members);
    // Update CIslandDescendants
    UpdateIslandDescendants(reg, members, descendants);
    // Reset the island preconditioner.
    reg.get<CIslandPreconditioner>(island).Reset();
  }

  // Update island's DoFs. This is always executed since there are scenarios where the island's
  // composition does not change, but its member actors change their DoF count so the island offsets
  // must be recomputed.
  // Note: This function is relatively cheap. Just a scan-type op over actors to compute the
  // corresponding offsets.
  for (auto&& [island, members, dofs] : reg.view<CIslandMembers const, CIslandDofInfo>().each()) {
    UpdateIslandDofInfo(reg, members, dofs);
  }

  // Similarly to the above, for differentiable scenes update island's derived-state and
  // differentiable-input sizes.
  for (auto&& [island, members, dofs, input] :
       reg.view<CIslandMembers const, CIslandDerivedStateInfo, CIslandDiffInputInfo>().each()) {
    UpdateIslandDifferentiabilityInfo(reg, members, dofs, input);
  }

  // All islands are now up-to-date
  reg.clear<TagIslandCompositionChanged>();

  // Unit test support
  if (testConfig && testConfig->postIslandUpdateCallback) {
    testConfig->postIslandUpdateCallback();
  }
}

bool mochi::island::GetForceSingleIsland(entt::registry const& reg) {
  return reg.ctx<CGlobalIsland const>().forceSingleIslandByUserRequest;
}

void mochi::island::SetForceSingleIsland(entt::registry& reg, bool forceSingleIsland) {
  reg.ctx<CGlobalIsland>().forceSingleIslandByUserRequest = forceSingleIsland;
}

void mochi::island::SetTestCallback_PreIslandUpdate(entt::registry& reg, std::function<void()> fn) {
  reg.ctx_or_set<CIslandTestConfig>().preIslandUpdateCallback = std::move(fn);
}

void mochi::island::SetTestCallback_PostIslandUpdate(
    entt::registry& reg,
    std::function<void()> fn) {
  reg.ctx_or_set<CIslandTestConfig>().postIslandUpdateCallback = std::move(fn);
}
