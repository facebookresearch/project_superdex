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

#include "mochi_integration.h"

using namespace mochi;

void mochi::integration::ClearMultiStepIntegrationData(entt::registry& reg, entt::entity e) {
  if (auto* intRigidPoses = reg.try_get<CIntegrationRigidStates>(e)) {
    intRigidPoses->prevSteps.clear();
  }
  if (auto* intRigidVels = reg.try_get<CIntegrationRigidVels>(e)) {
    intRigidVels->prevSteps.clear();
  }
  if (auto* intDispls = reg.try_get<CIntegrationDisplacementSlices>(e)) {
    intDispls->prevSteps.clear();
  }
  if (auto* intVels = reg.try_get<CIntegrationVelocitySlices<DisplacementLayer::Default>>(e)) {
    intVels->prevSteps.clear();
  }
  if (auto* intSkinnedVels =
          reg.try_get<CIntegrationVelocitySlices<DisplacementLayer::Skinned>>(e)) {
    intSkinnedVels->prevSteps.clear();
  }
  if (auto* intReducedPoses = reg.try_get<CIntegrationArticulatedReducedPose>(e)) {
    intReducedPoses->prevSteps.clear();
  }
  if (auto* intJointVels = reg.try_get<CIntegrationArticulatedJointVels>(e)) {
    for (auto& jointVel : intJointVels->value) {
      jointVel.prevSteps.clear();
    }
  }
  if (auto* intRodPoses = reg.try_get<CIntegrationRodPoses>(e)) {
    intRodPoses->prevSteps.clear();
  }
}
