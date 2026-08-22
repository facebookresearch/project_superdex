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

#include "mochi_common_components.h"

namespace mochi::common_components {
void InitializeOnce(entt::registry& reg) {
  // ECS Component Types:

#define MOCHI_REGISTER_ALL_TIME_STEPS(component)              \
  ecs::RegisterComponent<component<TimeStep::Current>>(reg);  \
  ecs::RegisterComponent<component<TimeStep::Previous>>(reg); \
  ecs::RegisterComponent<component<TimeStep::StageStart>>(reg);
#define MOCHI_REGISTER_ALL_TIME_STEPS_PRE_POST(pre, post, component)     \
  ecs::RegisterComponent<component<pre, TimeStep::Current, post>>(reg);  \
  ecs::RegisterComponent<component<pre, TimeStep::Previous, post>>(reg); \
  ecs::RegisterComponent<component<pre, TimeStep::StageStart, post>>(reg);

  ecs::RegisterComponent<CActorInfo>(reg);
  MOCHI_REGISTER_ALL_TIME_STEPS(CBoundingVolume);
  ecs::RegisterComponent<CConservativeStepBounds>(reg);
  ecs::RegisterComponent<CContactParams>(reg);
  ecs::RegisterComponent<CConvergenceStatus>(reg);
  MOCHI_REGISTER_ALL_TIME_STEPS_PRE_POST(real, DisplacementLayer::Default, CDisplacementSlice);
  MOCHI_REGISTER_ALL_TIME_STEPS_PRE_POST(real, DisplacementLayer::Skinned, CDisplacementSlice);
  MOCHI_REGISTER_ALL_TIME_STEPS_PRE_POST(real, DisplacementLayer::Default, CVelocitySlice);
  MOCHI_REGISTER_ALL_TIME_STEPS_PRE_POST(real, DisplacementLayer::Skinned, CVelocitySlice);
  MOCHI_REGISTER_ALL_TIME_STEPS(CRigidState);
  ecs::RegisterComponent<CIntegrationDisplacementSlices>(reg);
  ecs::RegisterComponent<CIntegrationVelocitySlices<DisplacementLayer::Default>>(reg);
  ecs::RegisterComponent<CIntegrationVelocitySlices<DisplacementLayer::Skinned>>(reg);
  ecs::RegisterComponent<CIntegrationRigidStates>(reg);
  ecs::RegisterComponent<CRecenteringParams>(reg);
  ecs::RegisterComponent<CRootTransform>(reg);
  ecs::RegisterComponent<CSceneGravity>(reg);
  ecs::RegisterComponent<CSceneTime>(reg);
  ecs::RegisterComponent<CSceneStepCounter>(reg);
  ecs::RegisterComponent<CSceneHandle>(reg);
  ecs::RegisterComponent<CFinalDisplacementRef<TimeStep::Current>>(reg);
  ecs::RegisterComponent<CFinalDisplacementRef<TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CTimeIntegratorState>(reg);
  ecs::RegisterComponent<CMeshColor>(reg);
  ecs::RegisterComponent<TagArticulatedActor>(reg);
  ecs::RegisterComponent<TagArticulatedLinkActor>(reg);
  ecs::RegisterComponent<TagBlendedActor>(reg);
  ecs::RegisterComponent<TagCompoundActor>(reg);
  ecs::RegisterComponent<TagFullyInitialized>(reg);
  ecs::RegisterComponent<TagHasDeepFlowCollider>(reg);
  ecs::RegisterComponent<TagRigidActor>(reg);
  ecs::RegisterComponent<TagRomActor>(reg);
  ecs::RegisterComponent<TagSoftActor>(reg);
  ecs::RegisterComponent<TagSoftSkinnedActor>(reg);
  ecs::RegisterComponent<TagShellActor>(reg);
  ecs::RegisterComponent<TagRodActor>(reg);
  ecs::RegisterComponent<TagDeformableActor>(reg);
  ecs::RegisterComponent<TagStaticActor>(reg);
  ecs::RegisterComponent<TagExcludedFromDebugDraw>(reg);
  ecs::RegisterComponent<TagSkinnedContact>(reg);
  ecs::RegisterComponent<CNodalBasedStructure>(reg);
  ecs::RegisterComponent<CBoundaryNodalBasedStructure>(reg);
  ecs::RegisterComponent<CContactNodalBasedStructure>(reg);

#undef MOCHI_REGISTER_ALL_TIME_STEPS
#undef MOCHI_REGISTER_ALL_TIME_STEPS_PRE_POST
}
} // namespace mochi::common_components
