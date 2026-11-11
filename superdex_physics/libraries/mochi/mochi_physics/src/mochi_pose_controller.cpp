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

#include "mochi_pose_controller.h"
#include "mochi_articulated_body.h"

using namespace mochi;

Constraint* LinkPosConstraint::Create(
    Scene* scene,
    ActorHandle /* articulated */,
    ActorHandle link,
    ArticulatedJointType /* type */,
    ArticulatedDofInfo const& /* dofInfo */,
    ArticulatedPoseInfo const& /* poseInfo */,
    Span<real const> /* pose */,
    TransformRT const& linkTransform,
    int /* id */,
    PoseTrackingParams const& params) const {
  // Translation constraint. The pivot is zero.
  RigidPivotToRigidTargetConstraintParams transConParams;
  transConParams.stiffness = params.stiffness;
  transConParams.damping = params.damping;
  transConParams.saturation = params.saturation;
  transConParams.localPosition = {};
  transConParams.targetTransform = linkTransform;
  transConParams.actor = link;

  return scene->CreateRigidPivotToRigidTargetConstraint(transConParams, ErrorAssert{});
}

void LinkPosConstraint::SetTarget(
    Constraint* constraint,
    ArticulatedPoseInfo const& /* poseInfo */,
    Span<real const> /*pose*/,
    TransformRT const& linkTransformCom) const {
  constraint->SetTargetPosition(linkTransformCom.GetTranslation(), ErrorAssert{});
  constraint->SetTargetRotation(linkTransformCom.GetRotation(), ErrorAssert{});
}

void LinkPosConstraint::AddForce(
    ArticulatedJacobian const& jacobian,
    int link,
    Constraint const* constraint,
    ColumnVectorView<real> outForce,
    Error& error) const {
  MOCHI_ERROR_RETURN(error);
  auto forceSpan = constraint->GetForce(error);
  MOCHI_ERROR_RETURN(error);
  // The constraint force must act on all DoFs of the link
  MOCHI_ASSERT(forceSpan.size() == RigidSize::kDAll);
  ColumnVectorView<real const, RigidSize::kDAll> force(forceSpan);
  jacobian.AddLinkTransposeProduct(link, force, outForce);
}

Constraint* LinkRotConstraint::Create(
    Scene* scene,
    ActorHandle /* articulated */,
    ActorHandle link,
    ArticulatedJointType /* type */,
    ArticulatedDofInfo const& /* dofInfo */,
    ArticulatedPoseInfo const& /* poseInfo */,
    Span<real const> /* pose */,
    TransformRT const& linkTransform,
    int /* id */,
    PoseTrackingParams const& params) const {
  // Rotation constraint. The pivot is zero.
  RigidPivotRotationConstraintParams rotatConParams;
  rotatConParams.stiffness = params.stiffness;
  rotatConParams.damping = params.damping;
  rotatConParams.saturation = params.saturation;
  rotatConParams.localRotation = {};
  rotatConParams.targetRotation = linkTransform.GetRotation().ToRotationVector();
  rotatConParams.actor = link;

  return scene->CreateRigidPivotRotationConstraint(rotatConParams, ErrorAssert{});
}

void LinkRotConstraint::SetTarget(
    Constraint* constraint,
    ArticulatedPoseInfo const& /* poseInfo */,
    Span<real const> /*pose*/,
    TransformRT const& linkTransformCom) const {
  constraint->SetTargetRotation(linkTransformCom.GetRotation(), ErrorAssert{});
}

void LinkRotConstraint::AddForce(
    ArticulatedJacobian const& jacobian,
    int link,
    Constraint const* constraint,
    ColumnVectorView<real> outForce,
    Error& error) const {
  MOCHI_ERROR_RETURN(error);
  auto forceSpan = constraint->GetForce(error);
  MOCHI_ERROR_RETURN(error);
  // The constraint force must act only on rotation DoFs of the link
  MOCHI_ASSERT(forceSpan.size() == RigidSize::kDRot);
  ColumnVector<real, RigidSize::kDAll> force{};
  force.BottomRows<RigidSize::kDRot>(RigidSize::kDRot) =
      ColumnVectorView<real const, RigidSize::kDRot>(forceSpan);
  jacobian.AddLinkTransposeProduct(link, force, outForce);
}

Constraint* JointConstraint::Create(
    Scene* scene,
    ActorHandle articulated,
    ActorHandle /* link */,
    ArticulatedJointType type,
    ArticulatedDofInfo const& dofInfo,
    ArticulatedPoseInfo const& poseInfo,
    Span<real const> pose,
    TransformRT const& /* linkTransform */,
    int id,
    PoseTrackingParams const& params) const {
  // If the joint is hard, do not create a constraint
  if (type == ArticulatedJointType::Hard) {
    return nullptr;
  }

  // For prismatic or revolute joints, create a single-dof target constraint
  if (dofInfo.GetSize() == 1) {
    ArticulatedSingleDofTargetConstraintParams dofTargetParams;
    dofTargetParams.actor = articulated;
    dofTargetParams.jointIndex = id; // Joint index is the same as link index
    dofTargetParams.dofIndex = 0; // Only dof in the joint
    real target = pose[poseInfo.offset];
    dofTargetParams.targetValue = target;
    dofTargetParams.stiffness = params.stiffness;
    dofTargetParams.damping = params.damping;
    dofTargetParams.saturation = params.saturation;
    return scene->CreateArticulatedSingleDofTargetConstraint(dofTargetParams, ErrorAssert{});
  }
  // For spherical or free joints, create a 3D rotation target constraint.
  // Translation control on free joints is not supported.
  else {
    MOCHI_ASSERT_VERBOSE(dofInfo.rotSize == 3, "Unexpected joint type for joint tracking");
    Articulated3dRotationTargetConstraintParams rotTargetParams;
    rotTargetParams.actor = articulated;
    rotTargetParams.jointIndex = id; // Joint index is the same as link index
    auto const offset = poseInfo.GetRotOffset();
    Quaternion target{Real4{pose[offset], pose[offset + 1], pose[offset + 2], pose[offset + 3]}};
    rotTargetParams.target = target;
    rotTargetParams.stiffness = params.stiffness;
    rotTargetParams.damping = params.damping;
    rotTargetParams.saturation = params.saturation;
    return scene->CreateArticulated3dRotationTargetConstraint(rotTargetParams, ErrorAssert{});
  }
}

void JointConstraint::SetTarget(
    Constraint* constraint,
    ArticulatedPoseInfo const& poseInfo,
    Span<real const> pose,
    TransformRT const& /*linkTransformCom*/) const {
  if (poseInfo.rotSize == RigidSize::kRot) {
    auto const offset = poseInfo.GetRotOffset();
    Quaternion target{Real4{pose[offset], pose[offset + 1], pose[offset + 2], pose[offset + 3]}};
    constraint->SetTargetRotation(target, ErrorAssert{});
  } else {
    MOCHI_ASSERT_VERBOSE(poseInfo.GetSize() == 1, "Unexpected joint type");
    real target = pose[poseInfo.offset];
    constraint->SetTargetDof(target, ErrorAssert{});
  }
}

void JointConstraint::AddForce(
    ArticulatedJacobian const& /* jacobian */,
    int /* link */,
    Constraint const* constraint,
    ColumnVectorView<real> outForce,
    Error& error) const {
  MOCHI_ERROR_RETURN(error);
  auto const forceSpan = constraint->GetForce(error);
  MOCHI_ERROR_RETURN(error);
  auto const articulatedDofs = constraint->GetDofIndicesForActor(0);
  MOCHI_ASSERT(forceSpan.size() == articulatedDofs.size(), "Invalid force size");
  ArrayPlusEqualsIndexedDst(
      outForce.data(), forceSpan.data(), articulatedDofs.data(), isize(forceSpan));
}

void controller::SetCurrentTarget(
    Span<real const> pose,
    Span<TransformRT const> linkTransformsCom,
    bool& outChanged,
    CControllerTarget<TimeStep::Current>& outTarget,
    CControllerTarget<TimeStep::Previous>& outTargetOld,
    CControllerPrescribedMotion& outMotion) {
  if (!outMotion.newTargetApplied) {
    outTargetOld.CopyFrom(outTarget);
    outMotion.newTargetApplied = true;
  }
  outChanged = pose != MakeConstSpan(outTarget.JointPose()) ||
      linkTransformsCom != outTarget.LinkTransformsCom();
  outTarget.Set(AsConstView(pose), linkTransformsCom);
}

template <PoseConstraintT C>
void controller::SetTargets(
    Span<real const> pose,
    Span<TransformRT const> linkTransformsCom,
    Span<ArticulatedPoseInfo const> poseInfo,
    CPoseSliceController<C>& outSlice) {
  for (int i = 0; i < outSlice.info.size(); ++i) {
    auto link = outSlice.info[i].link; // link = joint
    outSlice.constraintType.SetTarget(
        outSlice.impl[i].constraint, poseInfo[link], pose, linkTransformsCom[link]);
  }
}

template <PoseConstraintT C>
void controller::AddForces(
    ColumnVectorView<real> outForce,
    Error& error,
    CArticulatedJacobian const& jacobian,
    CPoseSliceController<C> const& slice) {
  for (int i = 0; i < slice.info.size(); ++i) {
    slice.constraintType.AddForce(
        jacobian.value, slice.info[i].link, slice.impl[i].constraint, outForce, error);
  }
}

#define MOCHI_SPECIALIZE_CONTROLLER_FUNCS(C) \
  template void controller::SetTargets<C>(   \
      Span<real const>,                      \
      Span<TransformRT const>,               \
      Span<ArticulatedPoseInfo const>,       \
      CPoseSliceController<C>&);             \
  template void controller::AddForces<C>(    \
      ColumnVectorView<real>,                \
      Error&,                                \
      CArticulatedJacobian const&,           \
      CPoseSliceController<C> const&);
MOCHI_SPECIALIZE_CONTROLLER_FUNCS(LinkPosConstraint);
MOCHI_SPECIALIZE_CONTROLLER_FUNCS(LinkRotConstraint);
MOCHI_SPECIALIZE_CONTROLLER_FUNCS(JointConstraint);
#undef MOCHI_SPECIALIZE_CONTROLLER_FUNCS

namespace mochi::controller {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CLinkPosController>(reg);
  ecs::RegisterComponent<CLinkRotController>(reg);
  ecs::RegisterComponent<CJointController>(reg);
  ecs::RegisterComponent<CControllerTarget<TimeStep::Current>>(reg);
  ecs::RegisterComponent<CControllerTarget<TimeStep::Previous>>(reg);
  ecs::RegisterComponent<CControllerConstraints>(reg);
  ecs::RegisterComponent<CControllerPrescribedMotion>(reg);
  ecs::RegisterComponent<CTargetOwners>(reg);
}
} // namespace mochi::controller
