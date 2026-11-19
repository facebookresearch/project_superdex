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

// Back-propagation of the joint friction anchors (see mochi_joint_friction.h).

#include "mochi_differentiable.h"
#include "mochi_joint_friction.h"

#include <mochi_core/articulated_body/joint_friction.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/rigid_body_size.h>

using namespace mochi;

void articulated::compound::AccumulateJointFrictionAnchorSolveAdjoints(
    ecs::RequiredTag<TagArticulatedActor>,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedJointFrictionParams const& frictionParams,
    CTimeIntegratorState const& intState,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CArticulatedJointFrictionAnchors const& anchors,
    CDiffForceGrad const& forceGrad,
    CDiffJointFrictionAnchorGrad& outAnchorGrad) {
  // Actors without joint dry friction have no anchors.
  if (anchors.value.Rows() == 0) {
    return;
  }
  if (outAnchorGrad.grads.Rows() == 0) {
    outAnchorGrad.grads.Resize(forceGrad.Rows());
    outAnchorGrad.grads.SetZero();
  }
  MOCHI_ASSERT_VERBOSE(outAnchorGrad.grads.Rows() == forceGrad.Rows(), "Invalid anchor adjoints.");
  auto const* joints = bodyShape.shape->GetJointsData();
  real const dtFromAnchor = intState.dtFromStepStart;
  // aBar += (dr/da)^T z, with z = -forceGrad and r the gradient of the friction energy.
  for (int i = 0; i < isize(joints->jointTypes); ++i) {
    auto const& fp = frictionParams[i];
    if (!HasDryFriction(fp)) {
      continue;
    }
    auto const type = joints->jointTypes[i];
    auto const& dofInfo = joints->dofInfo[i];
    if (type == ArticulatedJointType::Revolute || type == ArticulatedJointType::Prismatic) {
      // dr/da = -(exact Hessian).
      int const offset = poseInfo[i].offset;
      real hessian = 0_r;
      AddSingleDofFriction(
          /*useFittedHessian*/ false,
          /*psdDRes*/ false,
          fp,
          dtFromAnchor,
          currPose.value[offset],
          anchors.value[offset],
          nullptr,
          nullptr,
          &hessian);
      outAnchorGrad.grads[dofInfo.offset] += hessian * forceGrad[dofInfo.offset];
    } else if (type == ArticulatedJointType::Spherical) {
      int const offset = poseInfo[i].GetRotOffset();
      int const dofOffset = dofInfo.GetRotOffset();
      Vec4r anchorAdjoint = Load<RigidSize::kDRot, Vec4r>(&outAnchorGrad.grads[dofOffset]);
      anchorAdjoint -= RotationFrictionDGradientDAnchorTransposeProduct(
          fp,
          dtFromAnchor,
          Quaternion(Load<RigidSize::kRot, Vec4r>(&currPose.value[offset])),
          Quaternion(Load<RigidSize::kRot, Vec4r>(&anchors.value[offset])),
          Load<RigidSize::kDRot, Vec4r>(&forceGrad[dofOffset]));
      Store<RigidSize::kDRot>(&outAnchorGrad.grads[dofOffset], anchorAdjoint);
    }
  }
}

void articulated::compound::HandOffJointFrictionAnchorAdjoints(
    ecs::RequiredTag<TagArticulatedActor>,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedJointFrictionParams const& frictionParams,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CArticulatedJointFrictionAnchors const& anchors,
    CDiffJointFrictionAnchorGrad& inOutAnchorGrad,
    CDiffContactGrad<GradTarget::Previous>& outGradPrev) {
  if (inOutAnchorGrad.grads.Rows() == 0) {
    return;
  }
  MOCHI_ASSERT_VERBOSE(
      inOutAnchorGrad.grads.Rows() == outGradPrev.Rows(), "Invalid anchor adjoints.");
  // Without the anchors of the old state, the step anchored its joints at its start pose, the old
  // pose.
  if (anchors.value.Rows() == 0) {
    AsView(outGradPrev) += inOutAnchorGrad.grads;
    inOutAnchorGrad.grads.Resize(0);
    return;
  }
  // Otherwise, back through the radial return to the old pose (see PrepareJointFrictionAnchors), in
  // place: the adjoint of the advanced anchors becomes that of the old ones. The anchors of joints
  // without dry friction are not advanced, so their adjoint stays.
  auto const* joints = bodyShape.shape->GetJointsData();
  auto& anchorGrad = inOutAnchorGrad.grads;
  for (int i = 0; i < isize(joints->jointTypes); ++i) {
    auto const type = joints->jointTypes[i];
    auto const& dofInfo = joints->dofInfo[i];
    auto const& fp = frictionParams[i];
    if (!HasDryFriction(fp)) {
      continue;
    }
    if (type == ArticulatedJointType::Revolute || type == ArticulatedJointType::Prismatic) {
      int const offset = poseInfo[i].offset;
      real const advancedAdjoint = anchorGrad[dofInfo.offset];
      anchorGrad[dofInfo.offset] = 0_r;
      AdvanceSingleDofFrictionAnchorVjp(
          currPose.value[offset],
          anchors.value[offset],
          fp.preslidingDisplacement,
          advancedAdjoint,
          outGradPrev[dofInfo.offset],
          anchorGrad[dofInfo.offset]);
    } else if (type == ArticulatedJointType::Spherical) {
      int const offset = poseInfo[i].GetRotOffset();
      int const dofOffset = dofInfo.GetRotOffset();
      Vec4r rotAdjoint = Load<RigidSize::kDRot, Vec4r>(&outGradPrev[dofOffset]);
      Vec4r anchorAdjoint{};
      AdvanceRotationFrictionAnchorVjp(
          Quaternion(Load<RigidSize::kRot, Vec4r>(&currPose.value[offset])),
          Quaternion(Load<RigidSize::kRot, Vec4r>(&anchors.value[offset])),
          fp.preslidingDisplacement,
          Load<RigidSize::kDRot, Vec4r>(&anchorGrad[dofOffset]),
          rotAdjoint,
          anchorAdjoint);
      Store<RigidSize::kDRot>(&outGradPrev[dofOffset], rotAdjoint);
      Store<RigidSize::kDRot>(&anchorGrad[dofOffset], anchorAdjoint);
    }
  }
}
