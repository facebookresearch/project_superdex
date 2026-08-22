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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/rigid_body_utils.h>

#include <mochi_physics/diffsim/mochi_diffsim.h>
#include <mochi_physics/mochi_physics.h>

#include <string_view>

namespace mochi::test {

Actor* FindActorByName(Scene* scene, std::string_view actorName);
void LoadScenePrefab(Scene* scene, std::string_view prefabName);
Actor* LoadScenePrefab(Scene* scene, std::string_view prefabName, std::string_view actorName);

using ActorAddEpsFn = void (*)(Actor*, int, real);

inline void RigidActorAddEps(Actor* actor, int dof, real eps) {
  ColumnVector<real, RigidSize::kAll> pose;
  auto transform = actor->GetCenterOfMassTransform(ExpectOK{});
  TransformToRawPose(transform, pose);
  pose[dof] += eps;
  // Normalize quaternion after perturbation
  if (dof >= RigidSize::kTrans) {
    auto quat = pose.BottomRows<RigidSize::kRot>(RigidSize::kRot);
    quat /= quat.Norm();
  }
  transform = TransformFromRawPose(pose);
  actor->SetCenterOfMassTransform(transform, test::ExpectOK{});
}

inline void ArticulatedActorAddEps(Actor* actor, int dof, real eps) {
  auto const numDofs = actor->GetNumDofs();
  DynamicArray<real> pose(numDofs);
  actor->GetArticulatedPose(pose, test::ExpectOK{});
  pose[dof] += eps;
  actor->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});
}

inline void RigidActorAddVelEps(Actor* actor, int dof, real eps) {
  auto linVel = actor->GetLinearVelocity(test::ExpectOK{});
  auto angVel = actor->GetAngularVelocity(test::ExpectOK{});
  Real3 delta{};
  delta[dof % 3] = eps;
  if (dof < RigidSize::kDTrans) {
    linVel += delta;
  } else {
    angVel += delta;
  }
  actor->SetVelocity(linVel, angVel, test::ExpectOK{});
}

inline void ArticulatedActorAddVelEps(Actor* actor, int dof, real eps) {
  auto const numDofs = actor->GetNumDofs();
  DynamicArray<real> vel(numDofs);
  actor->GetArticulatedJointVelocities(vel, test::ExpectOK{});
  vel[dof] += eps;
  actor->SetArticulatedJointVelocities(vel, test::ExpectOK{});
}

inline ActorAddEpsFn GetActorAddEpsFn(ActorType type) {
  switch (type) {
    case ActorType::Rigid:
      return RigidActorAddEps;
    case ActorType::Articulated:
      return ArticulatedActorAddEps;
    default:
      MOCHI_ASSERT_VERBOSE(false, "Unsupported actor type");
      return nullptr;
  }
}

inline ActorAddEpsFn GetActorAddVelEpsFn(ActorType type) {
  switch (type) {
    case ActorType::Rigid:
      return RigidActorAddVelEps;
    case ActorType::Articulated:
      return ArticulatedActorAddVelEps;
    default:
      MOCHI_ASSERT_VERBOSE(false, "Unsupported actor type");
      return nullptr;
  }
}

inline ColumnVector<real, RigidSize::kAll> ConvertRigidGradientInternalToExternal(
    Actor* actor,
    ColumnVectorView<real const, RigidSize::kDAll> inGrad) {
  ColumnVector<real, RigidSize::kAll> outGrad;
  auto transform = actor->GetCenterOfMassTransform(test::ExpectOK{});
  diffsim::ConvertRigidGradientLieToQuaternion(transform, inGrad, outGrad, test::ExpectOK{});
  return outGrad;
}

inline ColumnVector<real> ConvertArticulatedGradientInternalToExternal(
    Actor* actor,
    ColumnVectorView<real const> inGrad) {
  ColumnVector<real> outGrad(inGrad);
  ColumnVector<real> pose(actor->GetNumDofs());
  actor->GetArticulatedPose(pose, test::ExpectOK{});
  diffsim::ConvertArticulatedGradientLieToRotationVector(actor, pose, outGrad, test::ExpectOK{});
  return outGrad;
}

inline ColumnVector<real> ConvertActorGradientInternalToExternal(
    Actor* actor,
    ColumnVectorView<real const> inGrad) {
  switch (actor->GetType()) {
    case ActorType::Rigid:
      return ConvertRigidGradientInternalToExternal(
          actor, inGrad.TopRows<RigidSize::kDAll>(RigidSize::kDAll));
    case ActorType::Articulated:
      return ConvertArticulatedGradientInternalToExternal(actor, inGrad);
    default:
      MOCHI_ASSERT_VERBOSE(false, "Unsupported actor type");
      return {};
  }
}
} // namespace mochi::test
