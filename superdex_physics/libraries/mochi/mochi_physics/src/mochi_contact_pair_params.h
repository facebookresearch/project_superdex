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

#include "mochi_ecs.h"

#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/error.h>
#include <mochi_physics/mochi_physics.h>

#include <optional>
#include <unordered_map>
#include <utility>

namespace mochi {

[[nodiscard]] inline ContactParams ApplyContactPairParamsOverride(
    ContactParams params,
    ContactPairParamsOverride const& paramsOverride) {
  auto const applyOverride = [](real& value, std::optional<real> const& optional) {
    if (optional) {
      value = *optional;
    }
  };
  applyOverride(params.penaltyCoefficient, paramsOverride.penaltyCoefficient);
  applyOverride(params.frictionFalloffVel, paramsOverride.frictionFalloffVel);
  applyOverride(params.viscousFrictionCoefficient, paramsOverride.viscousFrictionCoefficient);
  applyOverride(params.coulombFrictionCoefficient, paramsOverride.coulombFrictionCoefficient);
  applyOverride(
      params.normalViscousDampingCoefficient, paramsOverride.normalViscousDampingCoefficient);
  return params;
}

class CContactPairParamsOverrideTable final : public NoCopy {
 public:
  using EntityPair = std::pair<entt::entity, entt::entity>;
  using Records = std::
      unordered_map<EntityPair, ContactPairParamsOverride, PairHash<entt::entity, entt::entity>>;

  [[nodiscard]] static EntityPair CanonicalPair(entt::entity a, entt::entity b);

  [[nodiscard]] ContactPairParamsOverride const* Find(entt::entity a, entt::entity b) const;

  void Set(entt::entity a, entt::entity b, ContactPairParamsOverride const& paramsOverride);

  void Clear(entt::entity a, entt::entity b);

  void RemoveEntity(entt::entity entity);

  void DisableDissipation() {
    for (auto& record : _records) {
      record.second.coulombFrictionCoefficient = 0_r;
      record.second.viscousFrictionCoefficient = 0_r;
      record.second.normalViscousDampingCoefficient = 0_r;
    }
  }

  [[nodiscard]] bool Empty() const {
    return _records.empty();
  }

 private:
  Records _records;
};

void ValidateContactPairParamsOverride(
    ContactPairParamsOverride const& paramsOverride,
    Error& error);

} // namespace mochi
