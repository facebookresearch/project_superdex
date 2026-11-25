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

#include "mochi_joint_friction.h"

#include <mochi_core/articulated_body/joint_friction.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/rigid_body_size.h>

using namespace mochi;

bool articulated::HasJointDryFriction(
    entt::registry const& reg,
    Span<entt::entity const> entities) {
  return std::ranges::any_of(entities, [&](entt::entity e) {
    auto const* friction = reg.try_get<CArticulatedJointFrictionParams>(e);
    return friction && HasDryFriction(*friction);
  });
}

bool articulated::compound::StoreJointFrictionParams(
    Span<ArticulatedJointFrictionParams const> friction,
    CArticulatedJointFrictionParams& outParams) {
  bool changed = isize(outParams) != isize(friction);
  outParams.resize(friction.size());
  for (int i = 0; i < isize(friction); ++i) {
    ArticulatedJointFrictionParams params = friction[i];
    params.falloffVel.reset(); // Deprecated and ignored
    changed |= params != outParams[i];
    outParams[i] = params;
  }
  return changed;
}

void articulated::compound::PrepareJointFrictionAnchors(
    ecs::OptionalCtxGlobal<TagAssemblyPreserveHistory const> preserveHistory,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedJointFrictionParams const& frictionParams,
    CArticulatedReducedPose<TimeStep::Previous> const& prevPose,
    CArticulatedJointFrictionAnchors& inOutAnchors) {
  auto& anchors = inOutAnchors.value;
  if (!HasDryFriction(frictionParams)) {
    anchors.Resize(0);
    return;
  }
  if (anchors.Rows() == 0) {
    anchors.Resize(prevPose.value.Rows());
    anchors = prevPose.value;
    return;
  }
  // A solve-free assembly reproduces the last step, so it keeps the anchors that step used.
  if (preserveHistory) {
    return;
  }
  // Radial return of the anchors of the joints with dry friction (revolute, prismatic and
  // spherical). The anchors of the other joints are not read: enabling their friction anchors them
  // at the pose (SettleJointFrictionAnchors).
  auto const& jointTypes = bodyShape.shape->GetJointsData()->jointTypes;
  for (int i = 0; i < isize(jointTypes); ++i) {
    auto const type = jointTypes[i];
    auto const& fp = frictionParams[i];
    if (!HasDryFriction(fp)) {
      continue;
    }
    if (type == ArticulatedJointType::Revolute || type == ArticulatedJointType::Prismatic) {
      int const offset = poseInfo[i].offset;
      anchors[offset] = AdvanceSingleDofFrictionAnchor(
          prevPose.value[offset], anchors[offset], fp.preslidingDisplacement);
    } else if (type == ArticulatedJointType::Spherical) {
      int const offset = poseInfo[i].GetRotOffset();
      Quaternion const anchor = AdvanceRotationFrictionAnchor(
          Quaternion(Load<RigidSize::kRot, Vec4r>(&prevPose.value[offset])),
          Quaternion(Load<RigidSize::kRot, Vec4r>(&anchors[offset])),
          fp.preslidingDisplacement);
      Store<RigidSize::kRot>(&anchors[offset], anchor.data);
    }
  }
}

void articulated::compound::SettleJointFrictionAnchors(
    Span<ArticulatedJointFrictionParams const> lastParams,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedReducedPose<TimeStep::Current> const& pose,
    CArticulatedJointFrictionAnchors& inOutAnchors) {
  auto& anchors = inOutAnchors.value;
  if (anchors.Rows() == 0) {
    return; // The next step anchors all joints at the pose.
  }
  auto const& jointTypes = bodyShape.shape->GetJointsData()->jointTypes;
  for (int i = 0; i < isize(jointTypes); ++i) {
    // A joint without dry friction returns to the pose: it starts without preload.
    real const delta = HasDryFriction(lastParams[i]) ? lastParams[i].preslidingDisplacement : 0_r;
    if (jointTypes[i] == ArticulatedJointType::Revolute ||
        jointTypes[i] == ArticulatedJointType::Prismatic) {
      int const offset = poseInfo[i].offset;
      anchors[offset] = AdvanceSingleDofFrictionAnchor(pose.value[offset], anchors[offset], delta);
    } else if (jointTypes[i] == ArticulatedJointType::Spherical) {
      int const offset = poseInfo[i].GetRotOffset();
      Quaternion const anchor = AdvanceRotationFrictionAnchor(
          Quaternion(Load<RigidSize::kRot, Vec4r>(&pose.value[offset])),
          Quaternion(Load<RigidSize::kRot, Vec4r>(&anchors[offset])),
          delta);
      Store<RigidSize::kRot>(&anchors[offset], anchor.data);
    }
  }
}

void articulated::compound::ClearJointFrictionAnchors(
    CArticulatedJointFrictionAnchors& outAnchors) {
  outAnchors.value.Resize(0);
}
