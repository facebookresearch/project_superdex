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

#include <mochi_physics/utils/mochi_prefab.h>

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

// Prefab constraints (mochi::prefab::ConstraintLists) as one list: identity, labels, the actors and
// joints they reference, and the bookkeeping that keeps those references valid while actors are
// renamed, deleted, or reordered. No UI or rendering.

namespace superdex::studio {

/// One constraint in a prefab: its type (which ConstraintLists array) and its index in that array.
struct ConstraintRef {
  mochi::ConstraintType type = mochi::ConstraintType::None;
  int index = -1;

  bool operator==(ConstraintRef const&) const = default;
};

/// How a constraint type attaches to the prefab, which decides how it is created and listed.
enum class ConstraintCategory {
  ArticulatedJoint, // One joint of an articulated actor.
  RigidPair, // Two rigid actors.
  RigidPivot, // A point or frame on one rigid actor, tied to the world.
  Deformable, // Nodes of deformable actors.
};

/// The constraint types a prefab can declare, in menu order.
std::span<mochi::ConstraintType const> PrefabConstraintTypes();

/// Short display name, e.g. "Joint Target".
char const* ConstraintTypeLabel(mochi::ConstraintType type);

/// One-sentence description of what the constraint does.
char const* ConstraintTypeDescription(mochi::ConstraintType type);

ConstraintCategory GetConstraintCategory(mochi::ConstraintType type);

/// Calls `f(type, array)` for every ConstraintLists array, in PrefabConstraintTypes order. `Lists`
/// is ConstraintLists or ConstraintLists const.
template <class Lists, class F>
void ForEachConstraintArray(Lists& lists, F&& f) {
  // Rod constraints are the only ConstraintTypes prefabs do not declare (see mochi_prefab.cpp).
  static_assert(
      static_cast<int>(mochi::ConstraintType::Count) == 17,
      "ConstraintType changed: review the prefab constraint arrays below.");
  using mochi::ConstraintType;
  f(ConstraintType::ArticulatedSingleDofTarget, lists.articulatedSingleDofTarget);
  f(ConstraintType::ArticulatedSingleDofRange, lists.articulatedSingleDofRange);
  f(ConstraintType::Articulated3dRotationTarget, lists.articulated3dRotationTarget);
  f(ConstraintType::Articulated3dRotationRange, lists.articulated3dRotationRange);
  f(ConstraintType::RigidSphericalJoint, lists.rigidSphericalJoint);
  f(ConstraintType::RigidPrismaticJoint, lists.rigidPrismaticJoint);
  f(ConstraintType::JointRotationTracking, lists.jointRotationTracking);
  f(ConstraintType::JointRotationRange, lists.jointRotationRange);
  f(ConstraintType::RigidPivotPosition, lists.rigidPivotPosition);
  f(ConstraintType::RigidPivotToRigidTarget, lists.rigidPivotToRigidTarget);
  f(ConstraintType::RigidPivotRotation, lists.rigidPivotRotation);
  f(ConstraintType::DeformableNodePosition, lists.deformableNodePosition);
  f(ConstraintType::DeformableNodeToRigid, lists.deformableNodeToRigid);
  f(ConstraintType::DeformableNodeToDeformableNode, lists.deformableNodeToDeformableNode);
}

/// Calls `f(constraint)` with the typed constraint `ref` names. Returns false (and does not call
/// `f`) when `ref` names no constraint.
template <class Lists, class F>
bool VisitConstraint(Lists& lists, ConstraintRef ref, F&& f) {
  bool found = false;
  ForEachConstraintArray(lists, [&](mochi::ConstraintType type, auto& array) {
    if (type == ref.type && ref.index >= 0 && ref.index < static_cast<int>(array.size())) {
      f(array[static_cast<size_t>(ref.index)]);
      found = true;
    }
  });
  return found;
}

/// Whether constraint prefab type `C` acts on one joint of an articulated actor (`actorName`,
/// `jointIndex`).
template <class C>
constexpr bool kIsJointConstraint =
    std::is_same_v<C, mochi::prefab::ArticulatedSingleDofTargetConstraintPrefab> ||
    std::is_same_v<C, mochi::prefab::ArticulatedSingleDofRangeConstraintPrefab> ||
    std::is_same_v<C, mochi::prefab::Articulated3dRotationTargetConstraintPrefab> ||
    std::is_same_v<C, mochi::prefab::Articulated3dRotationRangeConstraintPrefab>;

/// Whether constraint prefab type `C` names two actors (`actorNameA`, `actorNameB`).
template <class C>
constexpr bool kIsPairConstraint =
    std::is_same_v<C, mochi::prefab::RigidSphericalJointConstraintPrefab> ||
    std::is_same_v<C, mochi::prefab::RigidPrismaticJointConstraintPrefab> ||
    std::is_same_v<C, mochi::prefab::JointRotationTrackingConstraintPrefab> ||
    std::is_same_v<C, mochi::prefab::JointRotationRangeConstraintPrefab> ||
    std::is_same_v<C, mochi::prefab::DeformableNodeToDeformableNodeConstraintPrefab>;

/// Calls `f(name)` for every actor name field of `constraint` (a DynamicString, const if
/// `constraint` is). Mirrors the actor-name resolution in mochi_prefab.cpp's AddConstraintsImpl.
template <class C, class F>
void ForEachConstraintActorName(C& constraint, F&& f) {
  using T = std::remove_const_t<C>;
  if constexpr (kIsPairConstraint<T>) {
    f(constraint.actorNameA);
    f(constraint.actorNameB);
  } else if constexpr (std::is_same_v<T, mochi::prefab::DeformableNodeToRigidConstraintPrefab>) {
    f(constraint.rigidActorName);
    f(constraint.deformableActorName);
  } else if constexpr (
      kIsJointConstraint<T> ||
      std::is_same_v<T, mochi::prefab::RigidPivotPositionConstraintPrefab> ||
      std::is_same_v<T, mochi::prefab::RigidPivotToRigidTargetConstraintPrefab> ||
      std::is_same_v<T, mochi::prefab::RigidPivotRotationConstraintPrefab> ||
      std::is_same_v<T, mochi::prefab::DeformableNodePositionConstraintPrefab>) {
    f(constraint.actorName);
  } else {
    static_assert(
        kIsPairConstraint<T>,
        "New constraint type: add its actor names to this if constexpr chain.");
  }
}

/// Every constraint in `lists`, in PrefabConstraintTypes order.
std::vector<ConstraintRef> CollectConstraints(mochi::prefab::ConstraintLists const& lists);

/// Appends a default constraint of `type` and returns it. Joint constraints get stiffness and
/// damping suited to a joint spring or limit rather than ConstraintParams' rigid-weld defaults.
ConstraintRef AppendConstraint(mochi::prefab::ConstraintLists& lists, mochi::ConstraintType type);

/// Appends a copy of `ref` and returns it, or nullopt when `ref` names no constraint.
std::optional<ConstraintRef> DuplicateConstraint(
    mochi::prefab::ConstraintLists& lists,
    ConstraintRef ref);

bool RemoveConstraint(mochi::prefab::ConstraintLists& lists, ConstraintRef ref);

/// The constraints on joint `jointIndex` of articulated actor `actorName`.
std::vector<ConstraintRef> FindJointConstraints(
    mochi::prefab::ConstraintLists const& lists,
    std::string_view actorName,
    int jointIndex);

/// The constraints that name actor `actorName` or an actor inside it ("actorName/..."), other than
/// those on the joints of `actorName` itself (listed by joint instead; see FindJointConstraints).
std::vector<ConstraintRef> FindActorConstraints(
    mochi::prefab::ConstraintLists const& lists,
    std::string_view actorName);

/// Whether `name` is `actor` or a path inside it ("actor/link", "nested/actor").
bool ActorNameRefersTo(std::string_view name, std::string_view actor);

/// Renames every reference to actor `oldName`, and to the actors inside it ("oldName/..."), to
/// `newName`. Returns the number of names changed.
int RenameConstraintActor(
    mochi::prefab::ConstraintLists& lists,
    std::string_view oldName,
    std::string_view newName);

/// Removes every constraint that references actor `name` or an actor inside it. Returns the number
/// removed.
int RemoveConstraintsReferencingActor(mochi::prefab::ConstraintLists& lists, std::string_view name);

/// Moves the joint constraints of articulated actor `actorName` to new joint indices after its
/// joints were removed or reordered: `oldToNew[i]` is joint i's new index, or a negative value when
/// joint i is gone (its constraints are removed). Joint indices outside `oldToNew` are left as is.
/// Returns the number of constraints removed.
int RemapJointConstraints(
    mochi::prefab::ConstraintLists& lists,
    std::string_view actorName,
    std::span<int const> oldToNew);

/// Appends a copy of every joint constraint of articulated actor `from`, retargeted to actor `to`
/// (a copy of `from` with the same joints). Returns the number copied.
int CopyJointConstraints(
    mochi::prefab::ConstraintLists& lists,
    std::string_view from,
    std::string_view to);

/// The top-level articulated actor named `name`, or null.
mochi::prefab::ArticulatedActorPrefab const* FindArticulatedActor(
    mochi::prefab::ScenePrefab const& prefab,
    std::string_view name);

/// Joint `jointIndex`'s name, or "joint_<index>" when it has none or `actor` is null.
std::string JointDisplayName(mochi::prefab::ArticulatedActorPrefab const* actor, int jointIndex);

/// The scale Mochi applies to the local distances of constraints on actor `name` (its "prefab
/// metric scale"): the product of the nested prefab scales along its path, times the articulated
/// actor's scale for a link ("arm/link").
mochi::real GetConstraintActorMetricScale(
    mochi::prefab::ScenePrefab const& prefab,
    std::string_view name);

/// What a constraint is attached to, for lists: "arm/elbow_joint", "lid & base", "cloth #12".
std::string DescribeConstraint(mochi::prefab::ScenePrefab const& prefab, ConstraintRef ref);

/// The actor names a constraint field can take, by actor kind.
struct ConstraintActorIndex {
  // Top-level rigid actors and the links of top-level articulated actors ("arm/link").
  std::vector<std::string> rigid;
  std::vector<std::string> deformable;
  std::vector<std::string> articulated;
  // Actors inside nested prefabs ("nested/actor", "nested/arm/link") and the articulated actors
  // that hold such links ("nested/arm"). Their kind is not tracked, so any field accepts them and
  // their joints are not checked.
  std::vector<std::string> nested;
};

/// Indexes the prefab's top-level actors by kind; `nestedActorNames` lists the actors of its nested
/// prefabs (as staged, "nested/actor").
ConstraintActorIndex BuildConstraintActorIndex(
    mochi::prefab::ScenePrefab const& prefab,
    std::span<std::string const> nestedActorNames);

/// Why `ref` cannot be created as authored (an unknown actor, a joint that does not exist, a DoF
/// its joint does not have, an invalid value), or an empty string when Studio finds no problem.
/// Best effort: actors inside nested prefabs are not checked by kind, and node indices are not
/// checked against the mesh.
std::string ValidateConstraint(
    mochi::prefab::ScenePrefab const& prefab,
    ConstraintActorIndex const& actors,
    ConstraintRef ref);

/// Removes every constraint ValidateConstraint rejects. Returns the reasons, one per removed
/// constraint.
std::vector<std::string> RemoveInvalidConstraints(
    mochi::prefab::ScenePrefab& prefab,
    ConstraintActorIndex const& actors);

/// Whether single-DoF constraints (target / range) can address DoF `dofIndex` of a `type` joint:
/// the translations of prismatic and free joints and the angle of revolute joints. Spherical and
/// free rotations take the 3D rotation constraints instead.
bool IsSingleDofConstraintDof(mochi::ArticulatedJointType type, int dofIndex);

/// Whether 3D rotation constraints (target / range) apply to a `type` joint (spherical or free).
bool Is3dRotationConstraintJoint(mochi::ArticulatedJointType type);

/// Whether a joint constraint fits a `type` joint: single-DoF constraints need a DoF they can
/// address, 3D rotation constraints a spherical or free joint.
bool JointFitsConstraint(mochi::ArticulatedJointType type, bool singleDof);

/// Whether DoF `dofIndex` of a `type` joint is a rotation (an angle) rather than a translation.
bool IsRotationDof(mochi::ArticulatedJointType type, int dofIndex);

} // namespace superdex::studio
