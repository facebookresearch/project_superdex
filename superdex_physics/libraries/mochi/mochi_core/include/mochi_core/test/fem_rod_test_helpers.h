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

#include <mochi_core/element_operations/fem_rod.h>

namespace mochi::test {

// Computes the perturbed frame axis when element DoFs are perturbed from a base state.
// Parallel-transports the frame axis from the base element tangent to the perturbed tangent,
// then rotates about the perturbed tangent by any change in the twist angle.
inline Vec4r ComputePerturbedFrameAxis(
    Vec4r const& X0,
    Vec4r const& X1,
    Span<real const> baseDofs,
    Span<real const> perturbation,
    Vec4r const& frameAxis) {
  MOCHI_ASSERT_VERBOSE(isize(baseDofs) == 2 * fem::kNumRodFields);
  MOCHI_ASSERT_VERBOSE(isize(perturbation) == 2 * fem::kNumRodFields);

  auto const baseAll = Load<Vec8r>(&baseDofs[0]);
  Vec4r const x0Base = X0 + GetHalf<0>(baseAll);
  Vec4r const x1Base = X1 + GetHalf<1>(baseAll);
  Vec4r const eHatBase = Normalize<3>(x1Base - x0Base);

  auto const pertAll = Load<Vec8r>(&perturbation[0]);
  Vec4r const x0Pert = x0Base + GetHalf<0>(pertAll);
  Vec4r const x1Pert = x1Base + GetHalf<1>(pertAll);
  Vec4r const eHatPert = Normalize<3>(x1Pert - x0Pert);

  return fem::TransportFrameAxis(
      eHatBase, eHatPert, perturbation[fem::kRodThetaDofOffset], frameAxis);
}

} // namespace mochi::test
