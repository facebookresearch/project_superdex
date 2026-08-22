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

#include "mochi_common_components.h"
#include "mochi_contact.h"
#include "mochi_discretization_components.h"
#include "mochi_ecs.h"
#include "mochi_ecs_utils.h"
#include "mochi_shape.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"

#include <mochi_core/materials/batched_materials.h>
#include <mochi_core/utils/no_copy.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include <type_traits>
#include <variant>

namespace mochi {

/**************************************************************************
  ECS Components for materials
*/
namespace materials {
using AnyMaterialParams = std::variant<
    NeoHookeanMaterialParams,
    StVenantKirchhoffMaterialParams,
    LinearElasticMaterialParams,
    ActiveNeoHookeanMaterialParams,
    ActiveShapeTargetingArapMaterialParams,
    ArapMaterialParams>;

/// @brief Type-erased container for any per-element material constants. @c std::monostate indicates
/// no material is assigned.
using AnyPerElementMaterialParams = std::variant<
    std::monostate,
    PerElementLameParams,
    PerElementArapParams,
    PerElementActiveShapeTargetingArapParams,
    PerElementActiveNeoHookeanParams>;
} // namespace materials

inline SoftMaterialType GetSoftMaterialType(materials::AnyMaterialParams const& params) {
  return std::visit(
      [](auto const& p) {
        return materials::utils::MaterialTraits<std::decay_t<decltype(p)>>::kType;
      },
      params);
}

struct CSoftMaterialParams {
  materials::AnyMaterialParams params = {};
  materials::AnyPerElementMaterialParams perElementParams;
  materials::PerElementReferenceMaterialStiffness referenceMaterialStiffness;
  real density = 0_r;
  real massDampingCoefficient = 0_r;
  real stiffnessDampingCoefficient = 0_r;
  bool stiffnessDampingIncludeGeometricTerm = false;
};

struct CShellMaterialParams : experimental::ShellMaterialParams, NoCopy {
  using experimental::ShellMaterialParams::operator=;
};

namespace material {
// Call once on startup
inline void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CSoftMaterialParams>(reg);
  ecs::RegisterComponent<CShellMaterialParams>(reg);
}
} // namespace material

} // namespace mochi
