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

#include "mochi_soft_rom_components.h"
#include "mochi_ecs.h"
#include "mochi_hyper_reduction.h"

namespace mochi::rom {

void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CNeuralNetCromDecoderData>(reg);
  ecs::RegisterComponent<CRomCommonProperties>(reg);
  ecs::RegisterComponent<CRomProjectionStrategy>(reg);
  ecs::RegisterComponent<CRomLinearBasis>(reg);
  ecs::RegisterComponent<CRomShiftVector>(reg);
  ecs::RegisterComponent<CRomModeAmplitudes>(reg);
  ecs::RegisterComponent<CRomVelocity<real, TimeStep::Current>>(reg);
  ecs::RegisterComponent<CRomVelocity<real, TimeStep::Previous>>(reg);
  ecs::RegisterComponent<CAuxiliaryPositionsForRomRigidTransform>(reg);
  ecs::RegisterComponent<rom::CDynamicSampleMeshStrategy<experimental::DynamicSampleMeshBsh>>(reg);
  ecs::RegisterComponent<CRomAdaptiveBasisContactForceInformed>(reg);
  ecs::RegisterComponent<CNeuralAffineRomStrategy>(reg);
  ecs::RegisterComponent<CRomFomSwitchingParams>(reg);
  ecs::RegisterComponent<CSampleMeshCaching>(reg);
  ecs::RegisterComponent<TagRomActorFixRigidTransformInSolve>(reg);
}

} // namespace mochi::rom
