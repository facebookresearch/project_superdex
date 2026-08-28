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

#include "mochi_ecs.h"
#include "mochi_group.h"

#include <mochi_core/linear_algebra/krylov/preconditioner_utils.h>
#include <mochi_core/solvers/newton_solver_status.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>
#include <mochi_physics/mochi_physics.h>

#include <functional>
#include <memory>
#include <vector>

namespace mochi {

// Forwards
struct CActorDofInfo;

/**************************************************************************
  ECS Component
*/

// Stores the list of actors that are direct members of an island.
// Derives from CGroupMembers to support utilities like ForEachDescendant.
struct CIslandMembers : CGroupMembers {};

// Stores the list of actors that are within the island, including those nested within compounds.
// WARNING: Updated in island::PreStep. May not be up-to-date in between steps.
struct CIslandDescendants : NoCopy {
  // All entities in the island
  std::vector<entt::entity> actors;

  // Entities of specific types
  std::vector<entt::entity> compoundActors; // TagCompoundActor (includes articulated compounds)
  std::vector<entt::entity> rigidActors; // TagRigidActor (includes articulated rigids)
  std::vector<entt::entity> softActors; // TagSoftActor (includes roms)
  std::vector<entt::entity> shellActors; // TagShellActor
  std::vector<entt::entity> rodActors; // TagRodActor
  std::vector<entt::entity> nestedSoftActors; // TagNestedSoftActor
  std::vector<entt::entity> blendedActors; // TagBlendedActor

  void Clear() {
    actors.clear();
    compoundActors.clear();
    rigidActors.clear();
    softActors.clear();
    shellActors.clear();
    rodActors.clear();
    nestedSoftActors.clear();
    blendedActors.clear();
  }
};

// Each entity within an island has CIslandMemberInfo, including those nested within compounds.
struct CIslandMemberInfo : NoCopy {
  entt::entity island = {};

  // True if the entity is nested within a compound.
  // False if the entity is directly listed by CIslandMembers.
  bool isNested = false;
};

// Stores the total degrees of freedom simulated by the island. Updated during island::PreStep.
// This information is used for determining the size of full-island vectors and matrices. Step,
// residual and dresidual are of dofsSize, while solution is of poseSize.
struct CIslandDofInfo : NoCopy {
  int dofsSize = 0; // Number of DoFs
  int poseSize = 0; // Size of the pose representation (aka position state)
};

// Struct storing information on the status of the solver during an integration stage.
// This is essentially a reduced version of NewtonSolverStatus with only the information that is
// relevant to report performance through the scene stats.
struct StageSolverStats {
  real resNorm = 0_r; // Residual norm with the current solution
  int numIterDone = 0; // Number of iterations done
  real resNormError = 0_r; // Maximum error in the residual norm across Newton iterations
  int numLSIterDone = 0; // Number of line-search iterations

  inline static StageSolverStats FromNewtonSolverStatus(NewtonSolverStatus<real> const& status) {
    return StageSolverStats{
        .resNorm = status.resNorm,
        .numIterDone = status.numIterDone,
        .resNormError = status.resNormError,
        .numLSIterDone = status.totalNumLSIterDone,
    };
  }
};

// Stores information on the statistics of an island solve execution.
struct CIslandSolverStats : NoCopy {
  DynamicArray<StageSolverStats> stages; // Stats per integration stage
};

/*
  Preconditioner of the island. Uses shared_ptr for compatibility with the linear solver.
*/
struct CIslandPreconditioner : public std::shared_ptr<PreconditionerRecyclingManager<real>> {
  void Reset() {
    *this = {std::make_shared<PreconditionerRecyclingManager<real>>()};
  }
};

/**************************************************************************
  ECS Tags
*/

// Identifies an entity as a simulation island.
struct TagIsland {};

// If even one entity has this tag, then all islands will be merged into one, even if the user
// didn't explicitly request it via Scene::ForceSingleIsland. This could be useful work-around in
// some cases, but it will hurt performance.
struct TagForceSingleIsland {};

// Indicates that the members of an island have change, or that their nested descendants have
// changed. Derived data will need to be updated on the next step.
struct TagIslandCompositionChanged {};

} // namespace mochi

namespace mochi::island {

/**************************************************************************
  Island Modifications
*/

// Create an island with no members.
entt::entity CreateEmpty(entt::registry& reg);

// Create an island, then call AddActor (see below).
MOCHI_API entt::entity CreateForActor(entt::registry& reg, entt::entity actor);

// Add an actor to an island. It must NOT be a member of another island nor group.
void AddActor(entt::registry& reg, entt::entity island, entt::entity actor);

// Call this when a an actor is added to a compound and the compound is already in an island. It
// will add CIslandMemberInfo to the nested child actor and any nested descendants of it.
void AddNestedActor(
    entt::registry& reg,
    entt::entity island,
    entt::entity parentCompound,
    entt::entity nestedChild);

// Remove an actor from its current island (if any). Destroy the island if empty.
void RemoveActor(entt::registry& reg, entt::entity actor);

// Add differentiability components to the island.
void InitDifferentiableIsland(entt::registry& reg, entt::entity island);

/**************************************************************************
  Island Simulation
*/

// Call this at the beginning of a simulation step to update all islands.
// WARNING: Must come AFTER updating CConservativeStepBounds and CPotentialColliders for all actors.
MOCHI_API void PreStep(entt::registry& reg);

// Return whether this island should be stepped in single-threaded mode.
// WARNING: Must come AFTER island::PreStep, which updates CIslandDofInfo and CIslandDescendants.
[[nodiscard]] bool ShouldRunSingleThreaded(
    entt::registry const& reg,
    entt::entity island,
    CIslandDescendants const& descendants);

// Debugging feature exposed to the public API as Scene::GetForceSingleIsland.
bool GetForceSingleIsland(entt::registry const& reg);

// Debugging feature exposed to the public API as Scene::SetForceSingleIsland.
void SetForceSingleIsland(entt::registry& reg, bool forceSingleIsland);

/**************************************************************************
  Unit Testing Support
*/

// FOR UNIT TESTS ONLY: Set a function to call at the start of island::PreStep.
MOCHI_API void SetTestCallback_PreIslandUpdate(entt::registry& reg, std::function<void()> fn);

// FOR UNIT TESTS ONLY: Set a function to call at the end of island::PreStep.
MOCHI_API void SetTestCallback_PostIslandUpdate(entt::registry& reg, std::function<void()> fn);

void InitializeOnce(entt::registry& reg);

} // namespace mochi::island
