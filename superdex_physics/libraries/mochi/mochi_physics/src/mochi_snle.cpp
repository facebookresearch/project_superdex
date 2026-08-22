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

#include "mochi_snle.h"

namespace mochi::snle {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CActorSnle>(reg);
  ecs::RegisterComponent<CCompoundConstraintSnle>(reg);
  ecs::RegisterComponent<CIslandContactSnle>(reg);
  ecs::RegisterComponent<CSkinnedContactSnle>(reg);
  ecs::RegisterComponent<CSkinnedInteractionSnle>(reg);
  ecs::RegisterComponent<CSoftSkinnedUnposedSnle>(reg);
}

} // namespace mochi::snle
