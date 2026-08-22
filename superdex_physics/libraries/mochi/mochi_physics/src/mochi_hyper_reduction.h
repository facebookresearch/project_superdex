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

#include "mochi_contact.h"
#include "mochi_ecs.h"

#include <mochi_physics/mochi_physics.h>

#include <mochi_core/contact/contact_samples_bsh.h>
#include <mochi_core/rom/dynamic_sample_mesh_bsh_manager.h>
#include <mochi_core/rom/rom_hyper_reduction.h>
#include <mochi_core/utils/no_copy.h>

#include <utility>

namespace mochi::rom::hyper {

// Component that manages dynamic sample mesh with BSH.
struct CDynamicSampleMeshBshManager : NoCopy {
  DynamicSampleMeshBshManager value;
  real maxColliderVelocity = 0.0;
  CDynamicSampleMeshBshManager(
      ContactSamplesBsh&& bsh,
      SdfLowerBoundAnchorSelection const& anchorSelectionMode,
      real maxColliderVelocity)
      : value(std::move(bsh), anchorSelectionMode), maxColliderVelocity(maxColliderVelocity) {}
};

void InitializeOnce(entt::registry& reg);

} // namespace mochi::rom::hyper
