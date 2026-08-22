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

#include "mochi_ecs_utils.h"

#include "mochi_common_components.h"

namespace mochi {

SceneHandle GetSceneHandle(entt::registry const& reg) {
  return reg.ctx<CSceneHandle const>().value;
}

entt::entity GetEntity(entt::registry const& reg, ActorHandle handle, Error& error) {
  MOCHI_ERROR_RETURN(error, entt::null);
  MOCHI_ERROR_IF_NOT(
      ActorHandleBelongsToScene(handle, GetSceneHandle(reg)), error, "Invalid ActorHandle.");
  MOCHI_ERROR_RETURN(error, entt::null);
  entt::entity e = GetEntityUnchecked(handle);
  MOCHI_ERROR_IF_NOT(reg.valid(e), error, "Invalid ActorHandle.");
  MOCHI_ERROR_RETURN(error, entt::null);
  return e;
}

entt::entity GetEntity(entt::registry const& reg, ConstraintHandle handle, Error& error) {
  MOCHI_ERROR_RETURN(error, entt::null);
  MOCHI_ERROR_IF_NOT(
      ConstraintHandleBelongsToScene(handle, GetSceneHandle(reg)),
      error,
      "Invalid ConstraintHandle.");
  MOCHI_ERROR_RETURN(error, entt::null);
  entt::entity e = GetEntityUnchecked(handle);
  MOCHI_ERROR_IF_NOT(reg.valid(e), error, "Invalid ConstraintHandle.");
  MOCHI_ERROR_RETURN(error, entt::null);
  return e;
}

} // namespace mochi
