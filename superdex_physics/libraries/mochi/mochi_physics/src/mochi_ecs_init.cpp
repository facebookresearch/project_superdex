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

#include "mochi_ecs_init.h"
#include "mochi_ecs_registry.h"

#include "mochi_materials.h" // material::InitializeOnce is declared inline

namespace mochi {

// clang-format off
// Having function declarations here avoids header dependencies.
namespace actor { void InitializeOnce(entt::registry& reg); }
namespace articulated { void InitializeOnce(entt::registry& reg); }
namespace blended { void InitializeOnce(entt::registry& reg); }
namespace capture { void InitializeOnce(entt::registry& reg); }
namespace common_components { void InitializeOnce(entt::registry& reg); }
namespace compound { void InitializeOnce(entt::registry& reg); }
namespace constraint { void InitializeOnce(entt::registry& reg); }
namespace contact { void InitializeOnce(entt::registry& reg); }
namespace contact_filter { void InitializeOnce(entt::registry& reg); }
namespace controller { void InitializeOnce(entt::registry& reg); }
namespace debug_draw_systems { void InitializeOnce(entt::registry& reg); }
namespace differentiable { void InitializeOnce(entt::registry& reg); }
namespace discretization { void InitializeOnce(entt::registry& reg); }
namespace group { void InitializeOnce(entt::registry& reg); }
namespace island { void InitializeOnce(entt::registry& reg); }
namespace query { void InitializeOnce(entt::registry& reg); }
namespace rigid { void InitializeOnce(entt::registry& reg); }
namespace rom { void InitializeOnce(entt::registry& reg); }
namespace rom::hyper { void InitializeOnce(entt::registry& reg); }
namespace rom::jacobian { void InitializeOnce(entt::registry& reg); }
namespace scene_recorder { void InitializeOnce(entt::registry& reg); }
namespace shape { void InitializeOnce(entt::registry& reg); }
namespace simulation { void InitializeOnce(entt::registry& reg); }
namespace skinned { void InitializeOnce(entt::registry& reg); }
namespace snle { void InitializeOnce(entt::registry& reg); }
namespace soft { void InitializeOnce(entt::registry& reg); }
namespace shell { void InitializeOnce(entt::registry& reg); }
namespace point_cloud_contact { void InitializeOnce(entt::registry& reg); }
namespace rod { void InitializeOnce(entt::registry& reg); }
// clang-format on

void ecs::InitializeOnce(entt::registry& reg) {
  actor::InitializeOnce(reg);
  articulated::InitializeOnce(reg);
  blended::InitializeOnce(reg);
  capture::InitializeOnce(reg);
  common_components::InitializeOnce(reg);
  compound::InitializeOnce(reg);
  constraint::InitializeOnce(reg);
  contact::InitializeOnce(reg);
  contact_filter::InitializeOnce(reg);
  controller::InitializeOnce(reg);
  debug_draw_systems::InitializeOnce(reg);
  differentiable::InitializeOnce(reg);
  discretization::InitializeOnce(reg);
  group::InitializeOnce(reg);
  island::InitializeOnce(reg);
  material::InitializeOnce(reg);
  query::InitializeOnce(reg);
  rigid::InitializeOnce(reg);
  rom::hyper::InitializeOnce(reg);
  rom::InitializeOnce(reg);
  rom::jacobian::InitializeOnce(reg);
  scene_recorder::InitializeOnce(reg);
  shape::InitializeOnce(reg);
  simulation::InitializeOnce(reg);
  skinned::InitializeOnce(reg);
  snle::InitializeOnce(reg);
  soft::InitializeOnce(reg);
  shell::InitializeOnce(reg);
  point_cloud_contact::InitializeOnce(reg);
  rod::InitializeOnce(reg);
}

} // namespace mochi
