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

#include "mochi_capture.h" // Reverse include for Intellisense

namespace mochi::capture {
namespace details {

void RegisterPostRestoreCallback(entt::registry& reg, std::function<void(entt::registry&)> fn);

} // namespace details

template <typename... Policies, typename SystemT>
inline void RegisterPostRestoreSystem(SystemT system, entt::registry& reg) {
  mochi::capture::details::RegisterPostRestoreCallback(
      reg, [system = std::move(system)](entt::registry& reg) {
        ecs::InvokeForEachGlobal<Policies...>(system, reg);
      });
}

} // namespace mochi::capture
