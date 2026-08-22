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

#include <cstdint>

namespace mochi::krylov {

enum class IterationStatus : std::uint8_t {
  /// @brief The iteration has not converged nor diverged and can continue.
  /// @note All iterative solvers explicitly use any status different from 'Active' as an exit flag.
  /// If this assumption becomes obsolete, please update all iterative solvers accordingly.
  Active = 0, /// @brief The iteration is active and has neither converged nor diverged yet.
  ConvergedAtol = 1, /// @brief The iteration has converged with the absolute tolerance.
  ConvergedRtol = 2, /// @brief The iteration has converged with the relative tolerance.
  DivergedRes = 3, /// @brief The iteration has diverged with a large residual.
  Count = 4
};

inline constexpr bool IsConverged(IterationStatus const& status) {
  return (status == IterationStatus::ConvergedAtol || status == IterationStatus::ConvergedRtol);
  static_assert(
      static_cast<int>(IterationStatus::Count) == 4,
      "Please update IsConverged if IterationStatus enumerator changes");
}

} // namespace mochi::krylov
