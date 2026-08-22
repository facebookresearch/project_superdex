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

#include <mochi_core/materials/material_types.h>

namespace mochi::materials {

/**
 * @brief Query whether a PSD strategy is supported for a given material model. Each material model
 * defines its own specialization.
 */
template <typename ParamsType>
[[nodiscard]] constexpr bool IsPsdStrategySupported(MaterialPsdStrategy strategy);

template <typename ParamsType>
[[nodiscard]] constexpr bool IsPsdStrategySupported(ParamsType const& params) {
  return IsPsdStrategySupported<ParamsType>(params.psdStrategy);
}

/**
 * @brief Query whether a PSD strategy has been resolved and is supported by a material model.
 *
 * @return True iff @p strategy is not @ref MaterialPsdStrategy::MaterialDefault and is supported
 * for @p ParamsType.
 *
 * @note @ref MaterialPsdStrategy::MaterialDefault is a user-facing sentinel and must be resolved
 * before entering PSD-strategy-dependent batched material responses.
 */
template <typename ParamsType>
[[nodiscard]] constexpr bool IsResolvedPsdStrategySupported(MaterialPsdStrategy strategy) {
  return strategy != MaterialPsdStrategy::MaterialDefault &&
      IsPsdStrategySupported<ParamsType>(strategy);
}

} // namespace mochi::materials
