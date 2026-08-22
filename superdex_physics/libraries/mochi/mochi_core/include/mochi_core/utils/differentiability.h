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

namespace mochi {

// Enum to specify the target of a gradient, which could be the current state, the previous state,
// the increment of previous derived state, the current input, or the previous input. Let us define
// merit Psi, state q, and some other target variable x. When the assembly requests the gradient,
// GradTarget defines the variable for computing the gradient dPsi/dx. When the assembly requests
// the Hessian, GradTarget defines the variable for computing the mixed Hessian d2Psi/dqdx, i.e.,
// the gradient d(dPsi/dqT)/dx.
enum class GradTarget {
  Current = 0,
  Previous = 1,
  PreviousDelta = 2,
  CurrentInput = 3,
  PreviousInput = 4,
  Count = 5
};

// Enum to specify the state dependencies of a function or piece of code, which could be only
// position (zero-order), velocity and possibly position (first-order), or acceleration and possibly
// velocity and position (second-order).
enum class StateDependency { ZeroOrder = 0, FirstOrder = 1, SecondOrder = 2 };

// Function to condition the execution of a piece of code depending on the gradTarget, its
// StateDependency and its inputDependency.
bool constexpr IsAssemblyNeeded(
    StateDependency stateDependency,
    bool inputDependency,
    GradTarget gradTarget) {
  // This function makes strong assumptions on the values of StateDependency and GradTarget enums
  // clang-format off
  static_assert(static_cast<int>(GradTarget::Count) == 5, "Revisit the validation");
  // Validation of assumptions for neededDueToStateDependency
#define MOCHI_STATE_ORDER(kStateDependency) static_cast<int>(kStateDependency)
#define MOCHI_STATE_ORDER_FOR_GRAD_TARGET(kGradTarget) (static_cast<int>(kGradTarget) % 3)
  static_assert(MOCHI_STATE_ORDER(StateDependency::ZeroOrder) < MOCHI_STATE_ORDER(StateDependency::FirstOrder));
  static_assert(MOCHI_STATE_ORDER(StateDependency::FirstOrder) < MOCHI_STATE_ORDER(StateDependency::SecondOrder));
  static_assert(MOCHI_STATE_ORDER_FOR_GRAD_TARGET(GradTarget::Current) == MOCHI_STATE_ORDER(StateDependency::ZeroOrder));
  static_assert(MOCHI_STATE_ORDER_FOR_GRAD_TARGET(GradTarget::Previous) == MOCHI_STATE_ORDER(StateDependency::FirstOrder));
  static_assert(MOCHI_STATE_ORDER_FOR_GRAD_TARGET(GradTarget::PreviousDelta) == MOCHI_STATE_ORDER(StateDependency::SecondOrder));
  static_assert(MOCHI_STATE_ORDER_FOR_GRAD_TARGET(GradTarget::CurrentInput) == MOCHI_STATE_ORDER(StateDependency::ZeroOrder));
  static_assert(MOCHI_STATE_ORDER_FOR_GRAD_TARGET(GradTarget::PreviousInput) == MOCHI_STATE_ORDER(StateDependency::FirstOrder));
  bool const neededDueToStateDependency = MOCHI_STATE_ORDER_FOR_GRAD_TARGET(gradTarget) <= MOCHI_STATE_ORDER(stateDependency);
#undef MOCHI_STATE_ORDER
#undef MOCHI_STATE_ORDER_FOR_GRAD_TARGET
  // Validation of assumptions for neededDueToInputDependency
#define MOCHI_IS_GRAD_TARGET_STATE(kGradTarget) (static_cast<int>(kGradTarget) < 3)
  static_assert(MOCHI_IS_GRAD_TARGET_STATE(GradTarget::Current));
  static_assert(MOCHI_IS_GRAD_TARGET_STATE(GradTarget::Previous));
  static_assert(MOCHI_IS_GRAD_TARGET_STATE(GradTarget::PreviousDelta));
  static_assert(!MOCHI_IS_GRAD_TARGET_STATE(GradTarget::CurrentInput));
  static_assert(!MOCHI_IS_GRAD_TARGET_STATE(GradTarget::PreviousInput));
  bool const neededDueToInputDependency = MOCHI_IS_GRAD_TARGET_STATE(gradTarget) || inputDependency;
#undef MOCHI_IS_GRAD_TARGET_STATE
  // clang-format on
  return neededDueToStateDependency && neededDueToInputDependency;
}
} // namespace mochi
