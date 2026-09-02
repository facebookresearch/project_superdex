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

#include "mochi_differentiable.h"
#include "mochi_ecs.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"

#include <mochi_core/utils/constraints.h>
#include <mochi_core/utils/spmat_utils.h>

namespace mochi {

// Forwards
struct CGroupMembers;

/*
  This tag is added to every entity that needs to be in a compound (i.e., a constraint or a soft
  skinned actor). On the following simulation step, tagged entities will be added to compounds
  automatically if the user didn't do it manually.
*/
struct TagEnsureEntityInCompound {};

/**************************************************************************
Common ECS systems for compound actors.
*/

// Initializes a compound actor for some active actors and constraint definitions
void InitCompoundActor(entt::registry& reg, entt::entity compoundEntity, Error& error);

// Add an existing actor to the compound
void AddActorToCompound(
    entt::registry& reg,
    entt::entity compoundEntity,
    entt::entity actorEntity,
    Error& error);

void RemoveActorFromCompound(
    entt::registry& reg,
    entt::entity compoundEntity,
    entt::entity actorEntity,
    Error& error);

// Add an existing contraint to the compound
void AddConstraintToCompound(
    entt::registry& reg,
    entt::entity compoundEntity,
    entt::entity constraintEntity,
    Error& error);

void RemoveConstraintFromCompound(
    entt::registry& reg,
    entt::entity compoundEntity,
    entt::entity constraintEntity,
    Error& error);

namespace compound {

// Assemble all constraints within a compound to CCompoundConstraintSnle.
// Used for regular compound actors and for articulated compounds.
void AssembleConstraints(
    AssemblyParams const& params,
    ecs::Included<TagCompoundActor>,
    CGroupMembers const& members,
    entt::registry& reg,
    CCompoundConstraintSnle& outConstraintSnl);

// Update CDofOffset for each actor. If there are constraints, then also update
// CCompoundConstraintSnle because it depends on the global dofs of each constrained actor.
void UpdateDofInfo(
    entt::registry& reg,
    entt::entity compound,
    CGroupMembers const& members,
    int compoundDofsOffset,
    int compoundPoseOffset,
    int compoundDerivedStateOffset,
    int compoundDiffInputOffset);

// ECS system to call UpdateDofInfo only for compounds that have changed.
void OnGlobalDofsChanged(
    ecs::Included<TagGlobalDofsChanged, TagCompoundActor>,
    ecs::OptionalTag<TagArticulatedActor> isArticulated,
    entt::registry& reg,
    entt::entity compound,
    CGroupMembers const& groupMembers,
    CDofOffset const& dofOffset,
    CDerivedStateOffset const* derivedStateOffset,
    CDiffInputOffset const* diffInputOffset);

// Updates the size and sparsity of the CCompoundConstraintSnle residual and dresidual.
// Must be called again if the DOF offset changes for any actor in the compound.
void UpdateConstraintGlobalSparsity(
    entt::registry& reg,
    CGroupMembers const& members,
    entt::entity compound);

// Updates the sparsity of the CCompoundConstraintSnle residual for input gradient assembly.
// Must be called again if the DOF offset changes for any actor in the compound.
// Note that we reuse the CCompoundConstraintSnle of regular assembly for input gradient assembly,
// because its size is always larger. However, we update the sparsity indices stored by each
// constraint in its CConstraintGlobalInputSparsityCache
void UpdateConstraintGlobalInputSparsity(
    entt::registry& reg,
    CGroupMembers const& members,
    entt::entity compound);

// Call this once at the start of each simulation step, to ensure that necessary compounds are up to
// date. There are two reasons for actors to live in the same compound:
// 1) They share a constraint.
// 2) They are part of a soft skinned actor.
// The function ensures that newly created constraints get grouped into compounds. It will also
// handle splitting of automatic compounds after the constraints are removed.
void UpdateAutoCompounds(entt::registry& reg);

void InitializeOnce(entt::registry& reg);

} // namespace compound

} // namespace mochi
