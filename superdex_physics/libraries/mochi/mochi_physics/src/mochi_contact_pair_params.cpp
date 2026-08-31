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

#include "mochi_contact_pair_params.h"

#include "mochi_contact.h"

#include <algorithm>

namespace mochi {

CContactPairParamsOverrideTable::EntityPair CContactPairParamsOverrideTable::CanonicalPair(
    entt::entity a,
    entt::entity b) {
  return entt::to_integral(a) <= entt::to_integral(b) ? EntityPair{a, b} : EntityPair{b, a};
}

ContactPairParamsOverride const* CContactPairParamsOverrideTable::Find(
    entt::entity a,
    entt::entity b) const {
  auto const it = _records.find(CanonicalPair(a, b));
  return it == _records.end() ? nullptr : &it->second;
}

void CContactPairParamsOverrideTable::Set(
    entt::entity a,
    entt::entity b,
    ContactPairParamsOverride const& paramsOverride) {
  _records.insert_or_assign(CanonicalPair(a, b), paramsOverride);
}

void CContactPairParamsOverrideTable::Clear(entt::entity a, entt::entity b) {
  _records.erase(CanonicalPair(a, b));
}

void CContactPairParamsOverrideTable::RemoveEntity(entt::entity entity) {
  std::erase_if(_records, [entity](auto const& record) {
    return record.first.first == entity || record.first.second == entity;
  });
}

void ValidateContactPairParamsOverride(
    ContactPairParamsOverride const& paramsOverride,
    Error& error) {
  bool const hasAnyValue = paramsOverride.penaltyCoefficient.has_value() ||
      paramsOverride.frictionFalloffVel.has_value() ||
      paramsOverride.viscousFrictionCoefficient.has_value() ||
      paramsOverride.coulombFrictionCoefficient.has_value() ||
      paramsOverride.normalViscousDampingCoefficient.has_value();
  MOCHI_ERROR_IF_NOT(
      hasAnyValue, error, "Contact pair parameter override must contain at least one value.");
  MOCHI_ERROR_RETURN(error);

  ValidateContactParams(ApplyContactPairParamsOverride(ContactParams{}, paramsOverride), error);
}

} // namespace mochi
