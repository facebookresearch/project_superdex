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

#include <mochi_physics/mochi_physics.h>
#include "mochi_ecs.h"

#include <mochi_core/utils/reflection.h>

#include <memory>

namespace mochi {

class Scene;

// Unlike Actor, the ActorInterface class has a public destructor so std::unique_ptr can destroy it.
class ActorInterface : public Actor {
 public:
  // Overridden by MOCHI_STRUCT_BEGIN in the implementation. It describes the most-derived object,
  // dynamic_cast<void*>(this). simple_reflection's free functions that take an ActorInterface,
  // such as GetTypeInfo, TryGetTypeInfo, GetFinalTypeInfo and the typed ConsoleExecute and
  // ConsoleList, do not compile with it: this member makes simple_reflection treat ActorInterface
  // as a reflected struct, but it has no GetTypeInfo.
  [[nodiscard]] virtual SReflect::StructTypeInfo const& GetFinalTypeInfo() const = 0;
};
using ActorInterfacePtr = std::unique_ptr<ActorInterface>;

// Create an object that implements mochi::Actor to satisfy the public interface.
ActorInterfacePtr CreateActorInterface(entt::registry& reg, entt::entity e, Scene* scene);

namespace actor {
[[nodiscard]] bool CanOwnNestedActors(ActorType type);
void InitializeOnce(entt::registry& reg);
} // namespace actor

} // namespace mochi
