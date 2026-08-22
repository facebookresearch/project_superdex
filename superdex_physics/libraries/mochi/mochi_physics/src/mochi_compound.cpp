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

#include "mochi_compound.h"

#include "mochi_articulated_body.h"
#include "mochi_blended.h"
#include "mochi_constraint.h"
#include "mochi_contact_filter.h"
#include "mochi_discretization_components.h"
#include "mochi_ecs_utils.h"
#include "mochi_group.h"
#include "mochi_island.h"
#include "mochi_rigid.h"
#include "mochi_rom_jacobian.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"
#include "mochi_soft.h"
#include "mochi_soft_skinned.h"
#include "mochi_solve.h"
#include "mochi_step.h"

#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/solvers/newton_solver.h>
#include <mochi_core/solvers/snle_problem.h>
#include <mochi_core/utils/activations.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/profile.h>

#include <algorithm>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

using namespace mochi;
using namespace mochi::compound;

// Get the actor to use for compound formation. Returns parent articulated actor if:
// 1. The actor is a link in an articulation, AND
// 2. The constraint has hasMixedLinks set (indicating it involves link actors from different
//    articulations or a mix of link and non-link actors)
// Otherwise, returns the actor itself.
static entt::entity GetActorForCompoundFormation(
    entt::registry const& reg,
    entt::entity actor,
    entt::entity constraint) {
  // Only return parent articulated actor if the constraint has link actors that aren't all from the
  // same articulation.
  auto const& info = reg.get<CConstraintInfo const>(constraint);
  if (info.hasMixedLinks) {
    entt::entity parentArticulated = TryGetParentArticulatedActor(reg, actor);
    if (parentArticulated != entt::null) {
      return parentArticulated;
    }
  }
  return actor;
}

namespace mochi::compound {

// ECS Tag (this file only):
// Indicates that a compound was created automatically by UpdateAutoCompounds. As such, it can also
// be destroyed or modified automatically. The user will never see it.
struct TagAutoCompound {};

// ECS Tag (this file only):
// Indicates that an automatically created compound was modified (e.g. constraint removed). We might
// be able to split it. We will check in the next call to UpdateAutoCompounds.
struct TagAutoCompoundChanged {};

} // namespace mochi::compound

void compound::AssembleConstraints(
    AssemblyParams const& params,
    ecs::Included<TagCompoundActor>,
    CGroupMembers const& members,
    entt::registry& reg,
    CCompoundConstraintSnle& outConstraintSnle) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(
      !members.constraints.empty(),
      "Only compounds with constraints should have CCompoundConstraintSnle");
  MOCHI_ASSERT_VERBOSE(
      outConstraintSnle.residuals.size() == 1 && outConstraintSnle.dresiduals.size() == 1,
      "CCompoundConstraintSnle must have 1 residual and 1 dresidual.");

  // Resize the residual based on the gradient target
  auto getMaxDofIndex = [](Span<int const> resIndices) {
    if (resIndices.empty()) {
      return -1;
    }
    return *std::max_element(resIndices.begin(), resIndices.end());
  };

  int maxDofIndex = -1;
  auto isInputTarget = params.gradTarget == GradTarget::CurrentInput ||
      params.gradTarget == GradTarget::PreviousInput;
  for (entt::entity c : members.constraints) {
    if (isInputTarget) {
      if (auto const* sparsity = reg.try_get<CConstraintGlobalInputSparsityCache const>(c)) {
        maxDofIndex = Max(maxDofIndex, getMaxDofIndex(sparsity->resIndices));
      }
    } else {
      auto const& sparsity = reg.get<CConstraintGlobalSparsityCache const>(c);
      maxDofIndex = Max(maxDofIndex, getMaxDofIndex(sparsity.resIndices));
    }
  }
  outConstraintSnle.residuals[0].first = 0;
  outConstraintSnle.residuals[0].second.Resize(maxDofIndex + 1);

  outConstraintSnle.SetZero(params);

  static_assert(
      static_cast<int>(GradTarget::Count) == 5,
      "Please update the switch statement below if GradTarget enum changes");

  // Traverse constraints. Dispatch to the appropriate templatized function.
  auto assembleConstraintFn = AssembleConstraint<GradTarget::Current>;
  switch (params.gradTarget) {
    case GradTarget::Current:
      assembleConstraintFn = AssembleConstraint<GradTarget::Current>;
      break;
    case GradTarget::Previous:
      assembleConstraintFn = AssembleConstraint<GradTarget::Previous>;
      break;
    case GradTarget::CurrentInput:
      assembleConstraintFn = AssembleConstraint<GradTarget::CurrentInput>;
      break;
    case GradTarget::PreviousInput:
      assembleConstraintFn = AssembleConstraint<GradTarget::PreviousInput>;
      break;
    default:
      MOCHI_ASSERT_VERBOSE(false, "Unexpected grad target");
      break;
  }
  for (entt::entity e : members.constraints) {
    assembleConstraintFn(reg, e, params, outConstraintSnle);
  }
}

/******************************************************************************
InitCompoundActor
*/

void mochi::InitCompoundActor(entt::registry& reg, entt::entity compoundEntity, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_PROFILE_SCOPE();

  // Compounds are considered "actors" in most respects. CActorInfo is often expected as a way
  // to identify actors. However, it won't have a user-defined name, nor a valid ActorType (enum
  // from the public API). To identify actors which are compounds, use TagCompoundActor.
  reg.emplace<CActorInfo>(compoundEntity, "Compound", ActorType::None);
  reg.emplace<TagCompoundActor>(compoundEntity);
  reg.emplace<CGroupMembers>(compoundEntity);
  reg.emplace<CActorDofInfo>(compoundEntity);
  reg.emplace<CDofOffset>(compoundEntity);

  // Similarly, emplace differentiability components if needed
  if (reg.try_ctx<TagDifferentiableScene>()) {
    reg.emplace<CActorDerivedStateInfo>(compoundEntity);
    reg.emplace<CDerivedStateOffset>(compoundEntity);
    reg.emplace<CActorDiffInputInfo>(compoundEntity);
    reg.emplace<CDiffInputOffset>(compoundEntity);
  }
}

// Recount poseSize and dofsSize. Used by MOCHI_ASSERT_VERBOSE (see below)
#if MOCHI_ASSERT_VERBOSE_ENABLED
static int CountTotalPoseSize(entt::registry const& reg, CGroupMembers const& members) {
  int poseSize = 0;
  for (entt::entity e : members.actors) {
    poseSize += reg.get<CActorDofInfo const>(e).poseSize;
  }
  return poseSize;
}

static int CountTotalDofsSize(entt::registry const& reg, CGroupMembers const& members) {
  int dofsSize = 0;
  for (entt::entity e : members.actors) {
    dofsSize += reg.get<CActorDofInfo const>(e).dofsSize;
  }
  return dofsSize;
}

static int CountTotalDerivedStateSize(entt::registry const& reg, CGroupMembers const& members) {
  int dofsSize = 0;
  for (entt::entity e : members.actors) {
    dofsSize += reg.get<CActorDerivedStateInfo const>(e).dofsSize;
  }
  return dofsSize;
}

static int CountTotalDiffInputSize(entt::registry const& reg, CGroupMembers const& members) {
  int size = 0;
  for (entt::entity e : members.actors) {
    size += reg.get<CActorDiffInputInfo const>(e).dofsSize;
  }
  return size;
}
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

void mochi::AddActorToCompound(
    entt::registry& reg,
    entt::entity compound,
    entt::entity actor,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(!reg.all_of<TagCompoundActor>(compound), error, "Not a compound");
  MOCHI_ERROR_IF(!reg.try_get<CActorInfo const>(actor), error, "Not an actor");
  MOCHI_ERROR_IF(
      reg.try_get<CGroupMemberInfo const>(actor), error, "Actor is already in a compound");
  MOCHI_ERROR_IF(
      reg.all_of<TagStaticActor>(actor), error, "A static actor cannot be added to a compound");
  MOCHI_ERROR_IF(
      reg.all_of<TagCompoundActor>(actor) && !reg.any_of<TagArticulatedActor>(actor),
      error,
      "A compound actor cannot be nested within another compound, unless it is an articulated actor");
  MOCHI_ERROR_RETURN(error);

  MOCHI_ASSERT((reg.all_of<CDofOffset, CActorDofInfo>(actor)), "Actor missing required components");

  // Immediately remove the actor from its previous island (if any)
  island::RemoveActor(reg, actor);

  // Find the compound's island. Create one if necessary.
  entt::entity island = {};
  if (auto const* islandMembership = reg.try_get<CIslandMemberInfo const>(compound)) {
    island = islandMembership->island;
  } else {
    island = island::CreateForActor(reg, compound);
  }
  MOCHI_ASSERT(reg.valid(island));

  // Add actor to the compound
  auto& members = reg.get<CGroupMembers>(compound);
  members.actors.push_back(actor);

  // CGroupMemberInfo points back to the compound entity
  reg.emplace<CGroupMemberInfo>(actor, compound);

  // Add actor (and any nested descendants) to the island
  island::AddNestedActor(reg, island, compound, actor);

  // Update the compound's state and derivative sizes
  auto const& actorDofInfo = reg.get<CActorDofInfo>(actor);
  auto& compoundDofInfo = reg.get<CActorDofInfo>(compound);
  compoundDofInfo.poseSize += actorDofInfo.poseSize;
  compoundDofInfo.dofsSize += actorDofInfo.dofsSize;
  MOCHI_ASSERT_VERBOSE(compoundDofInfo.poseSize == CountTotalPoseSize(reg, members));
  MOCHI_ASSERT_VERBOSE(compoundDofInfo.dofsSize == CountTotalDofsSize(reg, members));
  auto* compoundDerivedStateInfo = reg.try_get<CActorDerivedStateInfo>(compound);
  auto const* actorDerivedStateInfo = reg.try_get<CActorDerivedStateInfo>(actor);
  if (compoundDerivedStateInfo && actorDerivedStateInfo) {
    compoundDerivedStateInfo->dofsSize += actorDerivedStateInfo->dofsSize;
    MOCHI_ASSERT_VERBOSE(
        compoundDerivedStateInfo->dofsSize == CountTotalDerivedStateSize(reg, members));
  }
  auto* compoundDiffInputInfo = reg.try_get<CActorDiffInputInfo>(compound);
  auto const* actorDiffInputInfo = reg.try_get<CActorDiffInputInfo>(actor);
  if (compoundDiffInputInfo && actorDiffInputInfo) {
    compoundDiffInputInfo->dofsSize += actorDiffInputInfo->dofsSize;
    MOCHI_ASSERT_VERBOSE(compoundDiffInputInfo->dofsSize == CountTotalDiffInputSize(reg, members));
  }

  // compound::UpdateDofInfo should be called on the next step
  reg.emplace_or_replace<TagGlobalDofsChanged>(compound);
}

void mochi::RemoveActorFromCompound(
    entt::registry& reg,
    entt::entity compound,
    entt::entity actor,
    Error& error) {
  MOCHI_ERROR_IF(!reg.all_of<TagCompoundActor>(compound), error, "Not a compound actor");
  MOCHI_ERROR_IF(
      reg.all_of<TagArticulatedActor>(compound),
      error,
      "It is illegal to remove links from articulated actors");
  MOCHI_ERROR_RETURN(error);

  // Remove actor from CGroupMembers
  auto& groupMembers = reg.get<CGroupMembers>(compound);
  auto it = std::find(groupMembers.actors.begin(), groupMembers.actors.end(), actor);
  MOCHI_ERROR_IF(it == groupMembers.actors.end(), error, "Actor is not a member of the compound.");
  MOCHI_ERROR_RETURN(error);

  // Remove the actor (and any nested descendants) from the compound's island. We may be in the
  // process of destroying the actor, so don't add it to a new island yet.
  island::RemoveActor(reg, actor);

  groupMembers.actors.erase(it);

  // Remove the reference to the group from actor
  [[maybe_unused]] auto* groupInfo = reg.try_get<CGroupMemberInfo>(actor);
  MOCHI_ASSERT(
      groupInfo && (groupInfo->group == compound),
      "CGroupMemberInfo out-of-sync with CGroupMembers!");
  reg.remove<CGroupMemberInfo>(actor);

  // Update the compound's state and derivative sizes
  auto const& actorDofInfo = reg.get<CActorDofInfo>(actor);
  auto& compoundDofInfo = reg.get<CActorDofInfo>(compound);
  compoundDofInfo.poseSize -= actorDofInfo.poseSize;
  compoundDofInfo.dofsSize -= actorDofInfo.dofsSize;
  MOCHI_ASSERT_VERBOSE(compoundDofInfo.poseSize == CountTotalPoseSize(reg, groupMembers));
  MOCHI_ASSERT_VERBOSE(compoundDofInfo.dofsSize == CountTotalDofsSize(reg, groupMembers));
  auto* compoundDerivedStateInfo = reg.try_get<CActorDerivedStateInfo>(compound);
  auto const* actorDerivedStateInfo = reg.try_get<CActorDerivedStateInfo>(actor);
  if (compoundDerivedStateInfo && actorDerivedStateInfo) {
    compoundDerivedStateInfo->dofsSize -= actorDerivedStateInfo->dofsSize;
    MOCHI_ASSERT_VERBOSE(
        compoundDerivedStateInfo->dofsSize == CountTotalDerivedStateSize(reg, groupMembers));
  }
  auto* compoundDiffInputInfo = reg.try_get<CActorDiffInputInfo>(compound);
  auto const* actorDiffInputInfo = reg.try_get<CActorDiffInputInfo>(actor);
  if (compoundDiffInputInfo && actorDiffInputInfo) {
    compoundDiffInputInfo->dofsSize -= actorDiffInputInfo->dofsSize;
    MOCHI_ASSERT_VERBOSE(
        compoundDiffInputInfo->dofsSize == CountTotalDiffInputSize(reg, groupMembers));
  }

  if (groupMembers.actors.empty()) {
    // If the compound no longer has any actors, then it should not simulate in any island.
    island::RemoveActor(reg, compound);
  } else {
    // compound::UpdateDofInfo should be called on the next step
    reg.emplace_or_replace<TagGlobalDofsChanged>(compound);
  }

  if (reg.all_of<TagAutoCompound>(compound)) {
    // This compound was created automatically. Now that an actor has been removed, we might be able
    // to split it up or destroy it. We defer these checks until the next step in case the user is
    // in the middle of removing multiple actors.
    reg.emplace_or_replace<TagAutoCompoundChanged>(compound);
  }
}

void mochi::AddConstraintToCompound(
    entt::registry& reg,
    entt::entity compound,
    entt::entity constraint,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  MOCHI_ERROR_IF(!reg.all_of<TagCompoundActor>(compound), error, "Not a compound");
  MOCHI_ERROR_IF(
      reg.any_of<CGroupMemberInfo>(constraint), error, "Constraint already in a compound");
  MOCHI_ERROR_RETURN(error);

  auto& members = reg.get<CGroupMembers>(compound);
  auto const& info = reg.get<CConstraintInfo const>(constraint);

  // Ensure actors involved in constraint belong to the compound.
  // Do this first to avoid partial failure.
  // For articulation links constrained to external actors, we check that the parent articulated
  // actor is a member.
  for (auto constraintActor : info.actors) {
    entt::entity actorToCheck = GetActorForCompoundFormation(reg, constraintActor, constraint);
    auto const& actors = members.actors;
    auto itr = std::find(actors.begin(), actors.end(), actorToCheck);
    MOCHI_ERROR_IF(itr == actors.end(), error, "Actor in constraint is not compound member");
    MOCHI_ERROR_RETURN(error);
  }

  // Add constraint to the group
  members.constraints.push_back(constraint);

  // Add components to the constraint
  reg.emplace<CGroupMemberInfo>(constraint, compound);

  // Add CCompoundConstraintSnle to the compound if this was the first constraint
  if (members.constraints.size() == 1) {
    // Articulated compounds can use CCompoundConstraintSnle, but in that case the constraints apply
    // to the full DOF bodies, which are not used directly by the solver.
    auto& constraintSnle = reg.emplace<CCompoundConstraintSnle>(compound);
    bool isArticulated = reg.any_of<TagArticulatedActor>(compound);
    constraintSnle.useInSolver = !isArticulated;
  }

  // compound::UpdateDofInfo should be called on the next step
  reg.emplace_or_replace<TagGlobalDofsChanged>(compound);
}

void mochi::RemoveConstraintFromCompound(
    entt::registry& reg,
    entt::entity compound,
    entt::entity constraint,
    Error& error) {
  // This function reverts the changes from AddConstraintToCompound (see above).
  MOCHI_ERROR_RETURN(error);

  auto* memberInfo = reg.try_get<CGroupMemberInfo>(constraint);
  if (!memberInfo || memberInfo->group != compound) {
    MOCHI_ERROR_SET(error, "Constraint is not a member of the compound.");
    return;
  }

  // Remove constraint from the group
  auto& members = reg.get<CGroupMembers>(compound);
  {
    auto itr = std::find(members.constraints.begin(), members.constraints.end(), constraint);
    MOCHI_ASSERT(itr != members.constraints.end(), "CGroupMembers out-of-sync");
    members.constraints.erase(itr);
  }

  // Remove components from the constraint
  reg.erase<CGroupMemberInfo>(constraint);

  // Remove CCompoundConstraintSnle if we removed the last constraint
  if (members.constraints.empty()) {
    reg.erase<CCompoundConstraintSnle>(compound);
  }

  // compound::UpdateDofInfo should be called on the next step
  reg.emplace_or_replace<TagGlobalDofsChanged>(compound);

  if (reg.all_of<TagAutoCompound>(compound)) {
    // This compound was created automatically. Now that a constraint has been removed, we might be
    // able to split it up or destroy it. We defer these checks until the next step in case the user
    // is in the middle of removing multiple constraints.
    reg.emplace_or_replace<TagAutoCompoundChanged>(compound);
  }
}

void compound::UpdateConstraintGlobalSparsity(
    entt::registry& reg,
    CGroupMembers const& members,
    entt::entity compound) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(reg.all_of<TagCompoundActor>(compound), "Not a compound");

  // Gather all the pairs of global DOFs affected by constraints in this compound.
  // Update CConstraintGlobalSparsityCache::resIndices on each constraint, while we're at it.
  // For constraints involving link actors, we need to map to the parent articulated actor's DoFs.
  std::vector<NdArray<int, 2>> entries;
  for (entt::entity c : members.constraints) {
    auto const& info = reg.get<CConstraintInfo const>(c);
    auto& constraintGlobalResIndices = reg.get<CConstraintGlobalSparsityCache>(c).resIndices;
    constraintGlobalResIndices.clear();

    // Check if this constraint involves articulated link actors (set at constraint initialization)
    bool const hasMixedLinks = info.hasMixedLinks;

    if (!hasMixedLinks) {
      // Simple case: no link actors, use direct DoF mapping (identity Jacobian)
      for (int o = 0; o < isize(info.actors); ++o) {
        auto actorOffset = reg.get<CDofOffset>(info.actors[o]).dofsOffset;
        auto const& actorDofs = info.actorDofs[o];
        AppendSum(constraintGlobalResIndices, actorDofs, actorOffset);
      }
    } else {
      // Complex case: some actors are links of articulations.
      // We need to map through the link-to-articulated Jacobian.

      // Note: This branch should only be accessible if there are exactly two actors in the
      // constraint, at least one is a link of an articulation, and they are not links of the same
      // articulation. Therefore, we do not need to handle duplicate reduced DoFs from different
      // actors. These constraints are enforced elsewhere, but verbose debug asserts are included
      // here for redundancy.
      MOCHI_ASSERT_VERBOSE(
          isize(info.actors) == 2,
          "Mixed link constraint handling assumes constraints with exactly two actors.");
      MOCHI_ASSERT_VERBOSE(
          TryGetParentArticulatedActor(reg, info.actors[0]) !=
              TryGetParentArticulatedActor(reg, info.actors[1]),
          "Constraint without mixed links tagged as having mixed-links.");

      // Collect all compound DoF indices that are affected
      for (int o = 0; o < isize(info.actors); ++o) {
        entt::entity actor = info.actors[o];
        entt::entity parentArticulated = TryGetParentArticulatedActor(reg, actor);

        if (parentArticulated != entt::null) {
          // This actor is a link. The compound DoFs are from the parent articulated actor,
          // specifically the subset that affects this link.
          auto const& linkJacobian = reg.get<CArticulatedRigidJacobian const>(actor);
          auto const& articulatedDofOffset =
              reg.get<CDofOffset const>(parentArticulated).dofsOffset;
          for (int dof : linkJacobian.dofs) {
            int globalDof = articulatedDofOffset + dof;
            constraintGlobalResIndices.push_back(globalDof);
          }
        } else {
          // Regular actor, use direct DoF mapping
          auto actorOffset = reg.get<CDofOffset>(actor).dofsOffset;
          auto const& actorDofs = info.actorDofs[o];
          for (int localDof : actorDofs) {
            int globalDof = actorOffset + localDof;
            constraintGlobalResIndices.push_back(globalDof);
          }
        }
      }
    }

    // Add sparsity entries
    entries.reserve(entries.size() + Sqr(constraintGlobalResIndices.size()));
    for (int d0 : constraintGlobalResIndices) {
      for (int d1 : constraintGlobalResIndices) {
        entries.emplace_back(d0, d1);
      }
    }
  }

  // Initialize the SparseMatrix for the interaction dresidual.
  SparseMatrix<real> dresidual{MakeSparsityGraph(std::move(entries))};
  int const numRows = dresidual.Rows();

  // Update CConstraintGlobalSparsityCache::dresIndices (i.e. the indices of the
  // CCompoundConstraintSnle's dresidual that each entry in the local constraint dresidual
  // corresponds to).
  for (entt::entity c : members.constraints) {
    auto const& constraintGlobalResIndices = reg.get<CConstraintGlobalSparsityCache>(c).resIndices;
    auto& constraintGlobalDresIndices = reg.get<CConstraintGlobalSparsityCache>(c).dresIndices;
    constraintGlobalDresIndices.clear();
    constraintGlobalDresIndices.reserve(Sqr(constraintGlobalResIndices.size()));
    for (int row : constraintGlobalResIndices) {
      for (int col : constraintGlobalResIndices) {
        constraintGlobalDresIndices.push_back(dresidual.FindEntry(row, col));
        MOCHI_ASSERT_VERBOSE(constraintGlobalDresIndices.back() < dresidual.NumNonZeros());
      }
    }
  }

  // Reset the interaction dresidual.
  auto& constraintSnle = reg.get<CCompoundConstraintSnle>(compound);
  constraintSnle.dresiduals.clear();
  constraintSnle.dresiduals.emplace_back(
      /*rowOffset*/ 0, /*colOffset*/ 0, std::move(dresidual), /*symmetricPair*/ std::nullopt);

  // Resize the interaction residual to the same size (may be smaller than the size of the island's
  // SNLE problem because the constrained DOF indices may not span the full range of actor DOFs).
  constraintSnle.residuals.resize(1);
  constraintSnle.residuals[0].second.Resize(numRows);
}

void compound::UpdateConstraintGlobalInputSparsity(
    entt::registry& reg,
    CGroupMembers const& members,
    entt::entity compound) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(reg.all_of<TagCompoundActor>(compound), "Not a compound");

  // Update CConstraintGlobalInputSparsityCache::resIndices on constraints with differentiable input
  for (entt::entity c : members.constraints) {
    if (auto* sparsity = reg.try_get<CConstraintGlobalInputSparsityCache>(c)) {
      sparsity->resIndices.clear();
      auto const& info = reg.get<CConstraintInfo const>(c);
      MOCHI_ASSERT_VERBOSE(
          info.GetNumTargets() == GetNumConstrainedTargets(info.type),
          "This constraint does not support differentiability of targets");
      for (int o = 0; o < isize(info.actors); ++o) {
        auto actorOffset = reg.get<CDiffInputOffset>(info.actors[o]).dofsOffset;
        auto const& actorDofs = info.actorTargets[o];
        AppendSum(sparsity->resIndices, actorDofs, actorOffset);
      }
    }
  }
}

void mochi::compound::UpdateDofInfo(
    entt::registry& reg,
    entt::entity compound,
    CGroupMembers const& members,
    int compoundDofsOffset,
    int compoundPoseOffset,
    int compoundDerivedStateOffset,
    int compoundDiffInputOffset) {
  MOCHI_PROFILE_SCOPE();

  // Assign a new DOF and pose offset to each actor (if any)
  int nextActorDofsOffset = compoundDofsOffset;
  int nextActorPoseOffset = compoundPoseOffset;
  int nextDerivedStateOffset = compoundDerivedStateOffset;
  int nextDiffInputOffset = compoundDiffInputOffset;
  for (entt::entity actor : members.actors) {
    auto& dofOffset = reg.get<CDofOffset>(actor);
    auto const& dofInfo = reg.get<CActorDofInfo const>(actor);
    dofOffset.dofsOffset = nextActorDofsOffset;
    dofOffset.poseOffset = nextActorPoseOffset;
    nextActorDofsOffset += dofInfo.dofsSize;
    nextActorPoseOffset += dofInfo.poseSize;
    if (auto* derivedStateOffset = reg.try_get<CDerivedStateOffset>(actor)) {
      auto const& derivedStateInfo = reg.get<CActorDerivedStateInfo const>(actor);
      derivedStateOffset->dofsOffset = nextDerivedStateOffset;
      nextDerivedStateOffset += derivedStateInfo.dofsSize;
    }
    if (auto* diffInputOffset = reg.try_get<CDiffInputOffset>(actor)) {
      auto const& diffInputInfo = reg.get<CActorDiffInputInfo const>(actor);
      diffInputOffset->dofsOffset = nextDiffInputOffset;
      nextDiffInputOffset += diffInputInfo.dofsSize;
    }
  }

  // Constraints reference actor global DOFs, so they have to be updated too.
  if (!members.constraints.empty()) {
    UpdateConstraintGlobalSparsity(reg, members, compound);
    if (reg.try_ctx<TagDifferentiableScene>()) {
      UpdateConstraintGlobalInputSparsity(reg, members, compound);
    }
  }
}

void mochi::compound::OnGlobalDofsChanged(
    ecs::Included<TagGlobalDofsChanged, TagCompoundActor>,
    ecs::OptionalTag<TagArticulatedActor> isArticulated,
    entt::registry& reg,
    entt::entity compound,
    CGroupMembers const& groupMembers,
    CDofOffset const& dofOffset,
    CDerivedStateOffset const* derivedStateOffset,
    CDiffInputOffset const* diffInputOffset) {
  if (isArticulated.hasTag) {
    articulated::compound::InitFullDofProblem(reg, compound);
  } else {
    UpdateDofInfo(
        reg,
        compound,
        groupMembers,
        dofOffset.dofsOffset,
        dofOffset.poseOffset,
        derivedStateOffset ? derivedStateOffset->dofsOffset : 0,
        diffInputOffset ? diffInputOffset->dofsOffset : 0);
  }
}

// Given a set of actors, merge all of them into one compound. If they are already in compounds,
// merge those compounds. Otherwise, create a new compound. If some actor is in a user-created
// compound, then all actors must already belong to it. Return the merged compound.
static entt::entity MergeActorsInOneCompound(entt::registry& reg, Span<entt::entity const> actors) {
  // Collect all the compounds of the actors (if any). Also check if there is some user-created
  // compound.
  entt::entity userCreatedCompound = entt::null;
  std::vector<entt::entity> uniqueCompounds;
  uniqueCompounds.reserve(actors.size());
  for (auto const& actor : actors) {
    auto const* membership = reg.try_get<CGroupMemberInfo const>(actor);
    if (membership) {
      entt::entity compound = membership->group;
      if (!Contains(uniqueCompounds, compound)) {
        uniqueCompounds.emplace_back(compound);
      }
      if (!reg.all_of<TagAutoCompound>(compound)) {
        userCreatedCompound = compound;
      }
    }
  }

  // If there is a user-created compound, trivially return this compound.
  if (userCreatedCompound != entt::null) {
    // Validate that all actors belong to this user-created compound.
    for (auto const& actor : actors) {
      auto const* membership = reg.try_get<CGroupMemberInfo const>(actor);
      MOCHI_ASSERT(
          (membership != nullptr) && (membership->group == userCreatedCompound),
          "All actors must belong to the same user-created compound");
    }

    return userCreatedCompound;
  }

  // All of the compounds must have been created automatically. Merge them down to one.
  while (isize(uniqueCompounds) > 1) {
    // Merge the members from the last compound to the first compound.
    entt::entity dstCompound = uniqueCompounds.front();
    entt::entity srcCompound = uniqueCompounds.back();
    auto const& srcMembers = reg.get<CGroupMembers const>(srcCompound);

    // Transfer actors
    while (!srcMembers.actors.empty()) {
      entt::entity actor = srcMembers.actors.back();
      RemoveActorFromCompound(reg, srcCompound, actor, ErrorAssert{});
      AddActorToCompound(reg, dstCompound, actor, ErrorAssert{});
    }

    // Transfer constraints
    while (!srcMembers.constraints.empty()) {
      entt::entity constraint = srcMembers.constraints.back();
      RemoveConstraintFromCompound(reg, srcCompound, constraint, ErrorAssert{});
      AddConstraintToCompound(reg, dstCompound, constraint, ErrorAssert{});
    }

    // Destroy the last compound.
    reg.destroy(srcCompound);
    uniqueCompounds.pop_back();
  }

  // There should be at most one compound left. Get that compound, or create one.
  entt::entity outCompound = {};
  if (uniqueCompounds.empty()) {
    // Create a new compound. It will be managed internally, so the user will never see it.
    outCompound = reg.create();
    InitCompoundActor(reg, outCompound, ErrorAssert{});
    reg.emplace<TagAutoCompound>(outCompound);
  } else {
    // Everything will end up in this compound, which already existed.
    outCompound = uniqueCompounds[0];
  }

  // Add all actors to dstCompound unless they are already there
  for (entt::entity a : actors) {
    auto const* membership = reg.try_get<CGroupMemberInfo const>(a);
    if (membership) {
      MOCHI_ASSERT(
          membership->group == outCompound,
          "All other compounds should have been merged in the loop above");
    } else {
      AddActorToCompound(reg, outCompound, a, ErrorAssert{});
    }
  }

  return outCompound;
}

// If the constraint is not already in a compound, then add it to one automatically.
static void EnsureConstraintInCompound(entt::registry& reg, entt::entity constraintEntity) {
  if (reg.all_of<CGroupMemberInfo>(constraintEntity)) {
    return; // Already in a compound.
  }

  auto const& info = reg.get<CConstraintInfo const>(constraintEntity);
  int numActors = isize(info.actors);
  MOCHI_ASSERT(numActors > 0, "Invalid constraint data");

  // For constraints mixing articulated links and external actors, or articulated links from
  // different articulated actors, we need to use the parent articulated actor for compound
  // formation purposes.
  std::vector<entt::entity> actorsForCompound;
  actorsForCompound.reserve(numActors);
  for (entt::entity actor : info.actors) {
    entt::entity actorForCompound = GetActorForCompoundFormation(reg, actor, constraintEntity);
    actorsForCompound.push_back(actorForCompound);
  }

  // Merge all actors into one compound.
  entt::entity dstCompound = MergeActorsInOneCompound(reg, actorsForCompound);

  // Finally, add the constraint to dstCompound
  AddConstraintToCompound(reg, dstCompound, constraintEntity, ErrorAssert{});
}

// If the actors in a blended actor are not already in a compound, then add them to one
// automatically.
static void EnsureBlendedActorInCompound(entt::registry& reg, entt::entity blended) {
  // Collect all the actors
  auto const& composition = reg.get<CBlendedComposition const>(blended);
  std::vector<entt::entity> actors;
  actors.reserve(1 + composition.soft.size());
  actors.emplace_back(blended);
  for (auto soft : composition.soft) {
    actors.emplace_back(soft);
  }

  // Merge all actors into one compound
  MergeActorsInOneCompound(reg, actors);
}

void mochi::compound::UpdateAutoCompounds(entt::registry& reg) {
  // If a constraint was removed from an auto-compound, then we may be able to destroy the compound
  // or split it into multiple compounds. For simplicity, we accomplish this by destroying the
  // auto-compound, relying on auto-compound creation to regroup the actors as necessary.
  std::vector<entt::entity> compoundsToRemove;
  for (auto&& [compound] : reg.view<TagAutoCompoundChanged>().each()) {
    compoundsToRemove.emplace_back(compound);
  }
  for (entt::entity compound : compoundsToRemove) {
    auto& members = reg.get<CGroupMembers>(compound);
    // Make a copy of the actors for later
    std::vector<entt::entity> actors(members.actors);

    // Remove constraints
    while (!members.constraints.empty()) {
      entt::entity constraint = members.constraints.back();
      RemoveConstraintFromCompound(reg, compound, constraint, ErrorAssert{});
      reg.emplace_or_replace<TagEnsureEntityInCompound>(constraint);
    }

    // Remove actors.
    while (!members.actors.empty()) {
      entt::entity actor = members.actors.back();
      RemoveActorFromCompound(reg, compound, actor, ErrorAssert{});
      if (reg.any_of<TagBlendedActor>(actor)) {
        reg.emplace_or_replace<TagEnsureEntityInCompound>(actor);
      }
    }

    // Make sure the actors are in some island
    for (auto actor : actors) {
      island::CreateForActor(reg, actor);
    }

    // Remove the compound from the registry
    reg.destroy(compound);
  }
  reg.clear<TagAutoCompoundChanged>();

  // Create necessary auto-compounds
  for (auto&& [e] : reg.view<TagEnsureEntityInCompound>().each()) {
    if (reg.all_of<CConstraintInfo>(e)) {
      EnsureConstraintInCompound(reg, e);
    } else if (reg.all_of<TagBlendedActor>(e)) {
      EnsureBlendedActorInCompound(reg, e);
    } else {
      MOCHI_ASSERT(false, "This entity does not require an auto compound");
    }
  }
  reg.clear<TagEnsureEntityInCompound>();
}

namespace mochi::compound {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<TagAutoCompound>(reg);
  ecs::RegisterComponent<TagAutoCompoundChanged>(reg);
  ecs::RegisterComponent<TagEnsureEntityInCompound>(reg);
}
} // namespace mochi::compound
