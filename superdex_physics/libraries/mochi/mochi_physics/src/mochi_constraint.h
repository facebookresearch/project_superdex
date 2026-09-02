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

#include "mochi_attributes.h"
#include "mochi_common_components.h"
#include "mochi_ecs.h"
#include "mochi_snle.h"

#include <mochi_core/utils/assembly_params.h>
#include <mochi_core/utils/constraints.h>
#include <mochi_core/utils/differentiability.h>
#include <mochi_physics/mochi_physics.h>

#include <memory>
#include <string>
#include <vector>

namespace mochi {

// Forward declarations
struct CArticulatedEntity;

/*
  Helper functions to obtain constraint sizes for each constraint type
*/
inline int constexpr kNumConstrainedActors[] = {
    /* ConstraintType::None                           */ 0,
    /* ConstraintType::ArticulatedSingleDofRange      */ 1,
    /* ConstraintType::Articulated3dRotationRange     */ 1,
    /* ConstraintType::ArticulatedSingleDofTarget     */ 1,
    /* ConstraintType::Articulated3dRotationTarget    */ 1,
    /* ConstraintType::JointRotationRange             */ 2,
    /* ConstraintType::JointRotationTracking          */ 2,
    /* ConstraintType::RigidPivotPosition             */ 1,
    /* ConstraintType::RigidPivotToRigidTarget        */ 1,
    /* ConstraintType::RigidPivotRotation             */ 1,
    /* ConstraintType::RigidPrismaticJoint            */ 2,
    /* ConstraintType::RigidSphericalJoint            */ 2,
    /* ConstraintType::DeformableNodePosition         */ 1,
    /* ConstraintType::DeformableNodeToRigid          */ 2,
    /* ConstraintType::DeformableNodeToDeformableNode */ 2,
    /* ConstraintType::RodElementRotationToRigid      */ 2};
static_assert(
    std::size(kNumConstrainedActors) == static_cast<size_t>(ConstraintType::Count),
    "Please update this code if the ConstraintType enum changes");
// This assumption is used in logic to handle constraints between articulated-body links and
// external actors, and between links of different articulated bodies.
static_assert(
    std::ranges::all_of(kNumConstrainedActors, [](int n) { return n <= 2; }),
    "All constraint types must have at most 2 actors");

MOCHI_FORCE_INLINE int constexpr GetNumConstrainedActors(ConstraintType type) {
  return kNumConstrainedActors[static_cast<int>(type)];
}

inline int constexpr kConstraintSize[] = {
    /* ConstraintType::None                           */ 0,
    /* ConstraintType::ArticulatedSingleDofRange      */ 1,
    /* ConstraintType::Articulated3dRotationRange     */ 3,
    /* ConstraintType::ArticulatedSingleDofTarget     */ 1,
    /* ConstraintType::Articulated3dRotationTarget    */ 3,
    /* ConstraintType::JointRotationRange             */ 3,
    /* ConstraintType::JointRotationTracking          */ 3,
    /* ConstraintType::RigidPivotPosition             */ 3,
    /* ConstraintType::RigidPivotToRigidTarget        */ 3,
    /* ConstraintType::RigidPivotRotation             */ 3,
    /* ConstraintType::RigidPrismaticJoint            */ 3,
    /* ConstraintType::RigidSphericalJoint            */ 3,
    /* ConstraintType::DeformableNodePosition         */ 3,
    /* ConstraintType::DeformableNodeToRigid          */ 3,
    /* ConstraintType::DeformableNodeToDeformableNode */ 3,
    /* ConstraintType::RodElementRotationToRigid      */ 3};
static_assert(
    std::size(kConstraintSize) == static_cast<size_t>(ConstraintType::Count),
    "Please update this code if the ConstraintType enum changes");

MOCHI_FORCE_INLINE int constexpr GetConstraintSize(ConstraintType type) {
  return kConstraintSize[static_cast<int>(type)];
}

inline int constexpr kNumConstrainedDofs[] = {
    /* ConstraintType::None                           */ 0,
    /* ConstraintType::ArticulatedSingleDofRange      */ 1,
    /* ConstraintType::Articulated3dRotationRange     */ 3,
    /* ConstraintType::ArticulatedSingleDofTarget     */ 1,
    /* ConstraintType::Articulated3dRotationTarget    */ 3,
    /* ConstraintType::JointRotationRange             */ 6,
    /* ConstraintType::JointRotationTracking          */ 6,
    /* ConstraintType::RigidPivotPosition             */ 6,
    /* ConstraintType::RigidPivotToRigidTarget        */ 6,
    /* ConstraintType::RigidPivotRotation             */ 3,
    /* ConstraintType::RigidPrismaticJoint            */ 9,
    /* ConstraintType::RigidSphericalJoint            */ 12,
    /* ConstraintType::DeformableNodePosition         */ 3,
    /* ConstraintType::DeformableNodeToRigid          */ 9,
    /* ConstraintType::DeformableNodeToDeformableNode */ 6,
    /* ConstraintType::RodElementRotationToRigid      */ 11};
static_assert(
    std::size(kNumConstrainedDofs) == static_cast<size_t>(ConstraintType::Count),
    "Please update this code if the ConstraintType enum changes");

MOCHI_FORCE_INLINE int constexpr GetNumConstrainedDofs(ConstraintType type) {
  return kNumConstrainedDofs[static_cast<int>(type)];
}

inline int constexpr kNumConstrainedTargets[] = {
    /* ConstraintType::None                           */ 0,
    /* ConstraintType::ArticulatedSingleDofRange      */ 0,
    /* ConstraintType::Articulated3dRotationRange     */ 0,
    /* ConstraintType::ArticulatedSingleDofTarget     */ 1,
    /* ConstraintType::Articulated3dRotationTarget    */ 3,
    /* ConstraintType::JointRotationRange             */ 0,
    /* ConstraintType::JointRotationTracking          */ 3,
    /* ConstraintType::RigidPivotPosition             */ 3,
    /* ConstraintType::RigidPivotToRigidTarget        */ 6,
    /* ConstraintType::RigidPivotRotation             */ 3,
    /* ConstraintType::RigidPrismaticJoint            */ 0,
    /* ConstraintType::RigidSphericalJoint            */ 0,
    /* ConstraintType::DeformableNodePosition         */ 3,
    /* ConstraintType::DeformableNodeToRigid          */ 0,
    /* ConstraintType::DeformableNodeToDeformableNode */ 0,
    /* ConstraintType::RodElementRotationToRigid.     */ 3};
static_assert(
    std::size(kNumConstrainedTargets) == static_cast<size_t>(ConstraintType::Count),
    "Please update this code if the ConstraintType enum changes");

MOCHI_FORCE_INLINE int constexpr GetNumConstrainedTargets(ConstraintType type) {
  return kNumConstrainedTargets[static_cast<int>(type)];
}

// Get the parent articulated actor for a link actor, or entt::null if not a link
entt::entity TryGetParentArticulatedActor(
    ecs::PartialRegistry<CArticulatedEntity const> reg,
    entt::entity actor);

/*
  Component storing basic information of the constraint: name, type, etc.
*/
struct CConstraintInfo : NoCopy {
  CConstraintInfo(
      ecs::PartialRegistry<CArticulatedEntity const> reg,
      ConstraintType typeIn,
      std::vector<entt::entity>&& actorsIn,
      std::vector<std::vector<int>>&& actorDofsIn,
      std::vector<std::vector<int>>&& actorTargetsIn,
      ConstraintParams const& params)
      : type(typeIn),
        actors(std::move(actorsIn)),
        actorDofs(std::move(actorDofsIn)),
        actorTargets(std::move(actorTargetsIn)),
        stiffness(params.stiffness),
        damping(params.damping),
        saturation(params.saturation),
        hasMixedLinks(ComputeHasMixedLinks(reg, actors)) {
    MOCHI_ASSERT(isize(actors) == GetNumConstrainedActors(type), "Inconsistent constraint info");
    MOCHI_ASSERT(isize(actorDofs) == GetNumConstrainedActors(type), "Inconsistent constraint info");
    MOCHI_ASSERT(
        isize(actorTargets) == GetNumConstrainedActors(type), "Inconsistent constraint info");
#if MOCHI_ASSERT_VERBOSE_ENABLED
    MOCHI_ASSERT_VERBOSE(
        stiffness >= 0_r && IsFinite(stiffness), "Stiffness must be non-negative and finite.");
    MOCHI_ASSERT_VERBOSE(
        damping >= 0_r && IsFinite(damping), "Damping must be non-negative and finite.");
    MOCHI_ASSERT_VERBOSE(
        saturation != 0_r && IsFinite(saturation), "Saturation must be non-zero and finite.");
    int numDofs = 0;
    for (auto const& perActorDofs : actorDofs) {
      numDofs += isize(perActorDofs);
    }
    MOCHI_ASSERT_VERBOSE(numDofs == GetNumConstrainedDofs(type), "Inconsistent constraint info");
    // Warning: if numTargets != GetNumConstrainedTargets(type), this constraint cannot be used for
    // differentiability of targets.
    int numTargets = GetNumTargets();
    MOCHI_ASSERT_VERBOSE(
        numTargets == GetNumConstrainedTargets(type) || numTargets == 0,
        "Inconsistent constraint info");
#endif
  }

  int GetNumTargets() const {
    int numTargets = 0;
    for (auto const& perActorTargets : actorTargets) {
      numTargets += isize(perActorTargets);
    }
    return numTargets;
  }

  ConstraintType type = {};
  std::string name;
  std::vector<entt::entity> actors; // actors affected by this constraint
  std::vector<std::vector<int>> actorDofs; // per-actor dofs affected by this constraint
  std::vector<std::vector<int>>
      actorTargets; // per-actor target indices affected by this constraint
  real stiffness = 1_r;
  real damping = 0_r;
  real saturation = -1_r; // Negative means no saturation
  // True if the constraint involves link actors from different articulations or a mix of link
  // actors and non-link actors.
  bool hasMixedLinks = false;

 private:
  static bool ComputeHasMixedLinks(
      ecs::PartialRegistry<CArticulatedEntity const> reg,
      std::vector<entt::entity> const& actors) {
    MOCHI_ASSERT(isize(actors) <= 2, "Constraints cannot act on more than two actors");
    if (isize(actors) != 2) {
      return false;
    }
    entt::entity const actorAParent = TryGetParentArticulatedActor(reg, actors[0]);
    entt::entity const actorBParent = TryGetParentArticulatedActor(reg, actors[1]);
    // At least one of the actors is a link actor (or both would be entt::null, and therefore
    // equal), and they are not link actors of the same articulation.
    return actorAParent != actorBParent;
  }
};

/*
  This component goes an an actor that is affected one or more constraints.
  It points back to the constraint entities.
*/
struct CConstraintMemberInfo : NoCopy {
  std::vector<entt::entity> constraints;
};

/*
  Struct to cache information about the entries in the CCompoundConstraintSnle that are affected
  by the constraint.
*/
struct ConstraintGlobalSparsityCache {
  // Indices of the CCompoundConstraintSnle's residual that each entry in the local constraint
  // residual corresponds to.
  std::vector<int> resIndices;
  // Indices of the CCompoundConstraintSnle's dresidual that each entry in the local constraint
  // dresidual corresponds to.
  std::vector<int> dresIndices;
};

/*
  Component to cache information about the entries in the CCompoundConstraintSnle that are affected
  by the constraint. Must be recomputed if CDofOffset changes for any actor in the compound.
*/
struct CConstraintGlobalSparsityCache : public ConstraintGlobalSparsityCache, public NoCopy {};

/*
  Component to cache information about the entries in the input-data CCompoundConstraintSnle that
  are affected by the constraint.
*/
struct CConstraintGlobalInputSparsityCache : public ConstraintGlobalSparsityCache, public NoCopy {};

/*
  Component with data specific to each constraint type.
  This is a templated struct that is specialized for each ConstraintType.
*/
template <ConstraintType kType>
struct CConstraintData : public NoCopy {};
template <>
struct CConstraintData<ConstraintType::RigidSphericalJoint> : public NoCopy {
  Real3 posLocalA = {}; // Joint position for rigid actor A in its local frame.
  Real3 posLocalB = {}; // Joint position for rigid actor B in its local frame.
};
template <>
struct CConstraintData<ConstraintType::RigidPrismaticJoint> : public NoCopy {
  Quaternion localFrame{}; // Local frame of rigid actor A in which the translation is measured
  Real3 tRef{}; // Reference translation
  std::optional<real> max = {}; // Max translation along the free axis (optional)
  std::optional<real> min = {}; // Min translation along the free axis (optional)
};
template <>
struct CConstraintData<ConstraintType::DeformableNodeToDeformableNode> : public NoCopy {
  Real3 restA{}; // Position of node A at rest
  Real3 restB{}; // Position of node B at rest
};
template <>
struct CConstraintData<ConstraintType::DeformableNodeToRigid> : public NoCopy {
  Real3 posLocalRigid{}; // Position of a point of the rigid actor in its local frame
  Real3 restDeformable{}; // Position of the deformable node at rest
};
template <>
struct CConstraintData<ConstraintType::DeformableNodePosition> : public NoCopy {
  Real3 rest = {}; // Position of the deformable node at rest
};
template <>
struct CConstraintData<ConstraintType::JointRotationRange> : public NoCopy {
  Quaternion q0; // Joint frame in rest configuration, local to actor A
  Quaternion qr; // Reference rotation, in the joint frame
  Real3 minRotVec = {}; // Minimum values of the rotation vector components
  Real3 maxRotVec = {}; // Maximum values of the rotation vector components
};
template <>
struct CConstraintData<ConstraintType::JointRotationTracking> : public NoCopy {
  Quaternion q0; // Joint frame in rest configuration, local to actor A
};
template <>
struct CConstraintData<ConstraintType::RodElementRotationToRigid> : public NoCopy {
  int elementIndex = kSentinelIndex; // Index of the rod element
  Quaternion q0; // Joint frame in rest configuration, local to rigid actor
};
template <>
struct CConstraintData<ConstraintType::RigidPivotPosition> : public NoCopy {
  Real3 posLocal = {}; // Position of the constrained point of the rigid actor in local coords (wrt
                       // root transform)
};
template <>
struct CConstraintData<ConstraintType::RigidPivotToRigidTarget> : public NoCopy {
  Real3 posLocal = {}; // Position of the constrained point of the rigid actor in local coords (wrt
                       // root transform)
};
template <>
struct CConstraintData<ConstraintType::RigidPivotRotation> : public NoCopy {
  Quaternion rotLocal; // Rotation of the constrained frame of the rigid actor in local coords
};
template <>
struct CConstraintData<ConstraintType::ArticulatedSingleDofTarget> : public NoCopy {
  int jointIdx = kSentinelIndex; // Index of the constrained joint in the articulated body
  int dofIdx = kSentinelIndex; // Index of the dof within the joint
};
template <>
struct CConstraintData<ConstraintType::Articulated3dRotationTarget> : public NoCopy {
  int jointIdx = kSentinelIndex; // Index of the constrained joint in the articulated body
};
template <>
struct CConstraintData<ConstraintType::ArticulatedSingleDofRange> : public NoCopy {
  int jointIdx = kSentinelIndex; // Index of the constrained joint in the articulated body
  int dofIdx = kSentinelIndex; // Index of the dof within the joint
  real minValue; // Minimum value for the dof
  real maxValue; // Maximum value for the dof
};
template <>
struct CConstraintData<ConstraintType::Articulated3dRotationRange> : public NoCopy {
  int jointIdx = kSentinelIndex; // Index of the constrained joint in the articulated body
  Real3 minValues; // Minimum values for the dofs
  Real3 maxValues; // Maximum values for the dofs
};

/*
  Component for constraint target data. Templatized for type and time step.
  Targets are captured as part of the state.
*/
template <typename T, TimeStep kTimeStep>
struct CConstraintTarget : public NoCopy {
  T value = {};

  MOCHI_TEMPLATE_BEGIN(mochi::CConstraintTarget, T, kTimeStep);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_ATTRIBUTE_IF(kTimeStep == TimeStep::StageStart, HasOldTarget);
  MOCHI_FIELD(value);
  MOCHI_TEMPLATE_END();
};

// Initialize entity as a constraint of type RigidSphericalJointConstraint
void InitConstraint_RigidSphericalJoint(
    entt::registry& reg,
    entt::entity e,
    RigidSphericalJointConstraintParams const& params,
    Error& error);

// Initialize entity as a constraint of type RigidPrismaticJointConstraint
void InitConstraint_RigidPrismaticJoint(
    entt::registry& reg,
    entt::entity e,
    RigidPrismaticJointConstraintParams const& params,
    Error& error);

// Initialize entity as a constraint of type DeformableNodeToDeformableNodeConstraint
void InitConstraint_DeformableNodeToDeformableNode(
    entt::registry& reg,
    entt::entity e,
    DeformableNodeToDeformableNodeConstraintParams const& params,
    Error& error);

// Initialize entity as a constraint of type DeformableNodeToRigidConstraint
void InitConstraint_DeformableNodeToRigid(
    entt::registry& reg,
    entt::entity e,
    DeformableNodeToRigidConstraintParams const& params,
    Error& error);

// Initialize entity as a constraint of type JointRotationRangeConstraint
void InitConstraint_JointRotationRange(
    entt::registry& reg,
    entt::entity e,
    JointRotationRangeConstraintParams const& params,
    Error& error);

// Initialize entity as a constraint of type JointRotationTrackingConstraint
void InitConstraint_JointRotationTracking(
    entt::registry& reg,
    entt::entity e,
    JointRotationTrackingConstraintParams const& params,
    Error& error);

// Initialize entity as a constraint of type RodElementRotationToRigidConstraint
void InitConstraint_RodElementRotationToRigid(
    entt::registry& reg,
    entt::entity e,
    RodElementRotationToRigidConstraintParams const& params,
    Error& error);

// Initialize entity as a constraint of type RigidPivotPositionConstraint
void InitConstraint_RigidPivotPosition(
    entt::registry& reg,
    entt::entity e,
    RigidPivotPositionConstraintParams const& params,
    Error& error);

// Initialize entity as a constraint of type RigidPivotToRigidTargetConstraint
void InitConstraint_RigidPivotToRigidTarget(
    entt::registry& reg,
    entt::entity e,
    RigidPivotToRigidTargetConstraintParams const& params,
    Error& error);

// Initialize entity as a constraint of type RigidPivotRotationConstraint
void InitConstraint_RigidPivotRotation(
    entt::registry& reg,
    entt::entity e,
    RigidPivotRotationConstraintParams const& params,
    Error& error);

// Initialize entity as a constraint of type DeformableNodePositionConstraint
void InitConstraint_DeformableNodePosition(
    entt::registry& reg,
    entt::entity e,
    DeformableNodePositionConstraintParams const& params,
    Error& error);

// Initialize entity as a constraint of type ArticulatedSingleDofTargetConstraint
void InitConstraint_ArticulatedSingleDofTarget(
    entt::registry& reg,
    entt::entity e,
    ArticulatedSingleDofTargetConstraintParams const& params,
    Error& error);

// Initialize entity as a constraint of type Articulated3dRotationTargetConstraint
void InitConstraint_Articulated3dRotationTarget(
    entt::registry& reg,
    entt::entity e,
    Articulated3dRotationTargetConstraintParams const& params,
    Error& error);

// Initialize entity as a constraint of type ArticulatedSingleDofRangeConstraint
void InitConstraint_ArticulatedSingleDofRange(
    entt::registry& reg,
    entt::entity e,
    ArticulatedSingleDofRangeConstraintParams const& params,
    Error& error);

// Initialize entity as a constraint of type Articulated3dRotationRangeConstraint
void InitConstraint_Articulated3dRotationRange(
    entt::registry& reg,
    entt::entity e,
    Articulated3dRotationRangeConstraintParams const& params,
    Error& error);

// Systems for managing constraint targets
template <typename T>
void SetConstraintTargetPosition(
    Real3 const& pos,
    CConstraintTarget<T, TimeStep::Current>& outTarget);
template <>
inline void SetConstraintTargetPosition<Real3>(
    Real3 const& pos,
    CConstraintTarget<Real3, TimeStep::Current>& outTarget) {
  outTarget.value = pos;
}
template <>
inline void SetConstraintTargetPosition<TransformRT>(
    Real3 const& pos,
    CConstraintTarget<TransformRT, TimeStep::Current>& outTarget) {
  outTarget.value.SetTranslation(pos);
}

template <typename T>
void SetConstraintTargetRotation(
    Quaternion const& rot,
    CConstraintTarget<T, TimeStep::Current>& outTarget);
template <>
inline void SetConstraintTargetRotation<Quaternion>(
    Quaternion const& rot,
    CConstraintTarget<Quaternion, TimeStep::Current>& outTarget) {
  outTarget.value = rot;
}
template <>
inline void SetConstraintTargetRotation<TransformRT>(
    Quaternion const& rot,
    CConstraintTarget<TransformRT, TimeStep::Current>& outTarget) {
  outTarget.value.SetRotation(rot);
}

template <typename T>
void UpdateConstraintOldTarget(
    CConstraintTarget<T, TimeStep::Current> const& target,
    CConstraintTarget<T, TimeStep::StageStart>& outTargetOld) {
  outTargetOld.value = target.value;
}

// Function for the evaluation of constraint value, constraint Jacobian wrt dofs, and/or constraint
// Jacobian wrt target. If either result is requested, the corresponding container must be
// pre-allocated with the correct size for the constraint type. Jacobians are expected in row-major
// format.
template <TimeStep kTimeStep>
void EvalConstraint(
    entt::registry& reg,
    entt::entity e,
    Span<real> outC,
    Span<real> outJac,
    Span<real> outJacTarget,
    bool& outActive);

// Function for constraint assembly
template <GradTarget kGradTarget>
void AssembleConstraint(
    entt::registry& reg,
    entt::entity e,
    AssemblyParams const& params,
    CCompoundConstraintSnle& outConstraintSnle);

namespace constraint {
void InitializeOnce(entt::registry& reg);
}

} // namespace mochi
