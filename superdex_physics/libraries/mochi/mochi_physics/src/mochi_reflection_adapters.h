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

#include <mochi_core/utils/reflection_adapters.h>
#include <mochi_physics/mochi_physics.h>

#if MOCHI_USE_REFLECTION

namespace mochi::reflection {

template <class T>
struct ReflectionObjectResolver {
  template <class Scope>
  static T* Resolve(ReflectionValue<T*> const& value, Scope const& scope) {
    if (!value.IsValid()) {
      return nullptr;
    }
    T* object = nullptr;
    if constexpr (std::is_base_of_v<Actor, T>) {
      object = dynamic_cast<T*>(const_cast<Actor*>(scope.GetActor(value)));
    } else if constexpr (std::is_base_of_v<Constraint, T>) {
      object = dynamic_cast<T*>(const_cast<Constraint*>(scope.GetConstraint(value)));
    } else if constexpr (std::is_same_v<std::remove_const_t<T>, Scene>) {
      if constexpr (std::is_base_of_v<Context, Scope>) {
        object = const_cast<Scene*>(scope.GetScene(value));
      } else {
        object = value == scope.GetHandle() ? const_cast<Scene*>(&scope) : nullptr;
      }
    }
    return object;
  }
};

} // namespace mochi::reflection

#endif // MOCHI_USE_REFLECTION
