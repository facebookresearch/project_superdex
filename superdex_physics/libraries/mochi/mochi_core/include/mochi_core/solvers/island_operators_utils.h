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

#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/solvers/island_operators.h>

#include <type_traits>

namespace mochi {

/// @brief Cast-copy IslandOperator with scalar entries to a copy with a different scalar type.
///
/// This function converts an IslandOperators object from one scalar type to another (e.g., from
/// float to double) by recursively applying StaticCast to all contained matrices. It processes both
/// actor matrices and interaction matrices, creating new owning copies with the target scalar type.
///
/// @tparam ToIslandOperatorsOwningLite The target IslandOperatorsOwningLite type (must be
/// non-const).
/// @tparam Scalar The source scalar type from the input IslandOperators.
/// @param anyOp The input IslandOperators to be converted.
/// @return ToIslandOperatorsOwningLite with owning copies of all matrices
///
/// @note The function returns an owning container that can be "exported" by const-view into an
/// IslandOperators object.
template <
    typename ToIslandOperatorsOwningLite,
    typename Scalar,
    MOCHI_CONCEPT(IsIslandOperatorsOwningLite<ToIslandOperatorsOwningLite>)>
[[nodiscard]] ToIslandOperatorsOwningLite StaticCast(IslandOperators<Scalar> const& anyOp) {
  static_assert(
      !std::is_const_v<ToIslandOperatorsOwningLite>, "Destination type must be non-const");
  using ToScalar = std::decay_t<typename ToIslandOperatorsOwningLite::NonConstScalar>;
  //
  // ToIslandOperatorsOwningLite should be identical to IslandOperatorsOwningLite<ToScalar>
  //
  ToIslandOperatorsOwningLite newOp;
  auto const& inputActor = anyOp.GetActorMatrices();
  newOp.actorMatrices.reserve(inputActor.size());
  for (auto const& actorData : inputActor) {
    newOp.actorMatrices.push_back(
        {std::get<0>(actorData), StaticCast<AnyMatrix<ToScalar>>(std::get<1>(actorData))});
  }
  auto const& inputInteraction = anyOp.GetInteractionMatrices();
  newOp.interactionMatrices.reserve(inputInteraction.size());
  // First pass to convert the interaction matrices
  for (auto const& interData : inputInteraction) {
    newOp.interactionMatrices.emplace_back(
        interData.rowOffset,
        interData.colOffset,
        StaticCast<AnyMatrix<ToScalar>>(interData.matrix),
        std::nullopt);
  }
  // Find matching symmetric pair
  for (size_t k = 0; k < inputInteraction.size(); ++k) {
    auto const& interData = inputInteraction[k];
    if (interData.symmetricPair.has_value()) {
      auto const* ptr = GetValues(interData.symmetricPair.value()).data();
      size_t j = 0;
      for (; j < inputInteraction.size(); ++j) {
        auto const& jData = inputInteraction[j];
        if (GetValues(jData.matrix).data() == ptr) {
          break;
        }
      }
      MOCHI_ASSERT_VERBOSE(
          j < inputInteraction.size(), "No match is found for the symmetric pair.");
      newOp.interactionMatrices[k].symmetricPair.emplace(
          AsConstView(newOp.interactionMatrices[j].matrix));
    }
  }
  return newOp;
}

} // namespace mochi
