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

#include "mochi_rod_pose.h"

#include <mochi_core/element_operations/fem_rod.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/simd.h>

namespace mochi::rod {

Real3 ComputeRodElementTangent(
    Span<Real3 const> meshNodes,
    ColumnVectorView<real const> displacement,
    int elementIndex) {
  int const i0 = elementIndex;
  int const i1 = (elementIndex + 1) % isize(meshNodes);
  int const offset0 = fem::kNumRodFields * i0;
  int const offset1 = fem::kNumRodFields * i1;
  Real3 const x0 = meshNodes[i0] +
      Real3{displacement[offset0], displacement[offset0 + 1], displacement[offset0 + 2]};
  Real3 const x1 = meshNodes[i1] +
      Real3{displacement[offset1], displacement[offset1 + 1], displacement[offset1 + 2]};
  return Normalize(x1 - x0);
}

void ApplyLieDeltaToPose(
    Span<Real3 const> meshNodes,
    ColumnVectorView<real const> refDisplacement,
    Span<Real3 const> refAxes,
    ColumnVectorView<real const> dofDelta,
    ColumnVectorView<real> outDisplacement,
    Span<Real3> outAxes) {
  MOCHI_ASSERT_VERBOSE(isize(meshNodes) >= 2, "Rod must have at least 2 nodes");
  MOCHI_ASSERT_VERBOSE(
      isize(refDisplacement) == isize(meshNodes) * fem::kNumRodFields,
      "refDisplacement size mismatch");
  int const numElements = isize(refAxes);
  MOCHI_ASSERT_VERBOSE(
      numElements == isize(meshNodes) - 1 || numElements == isize(meshNodes),
      "refAxes size mismatch");
  MOCHI_ASSERT_VERBOSE(isize(dofDelta) == isize(refDisplacement), "dofDelta size mismatch");
  MOCHI_ASSERT_VERBOSE(
      isize(outDisplacement) == isize(refDisplacement), "outDisplacement size mismatch");
  MOCHI_ASSERT_VERBOSE(isize(outAxes) == isize(refAxes), "outAxes size mismatch");

  outDisplacement = refDisplacement + dofDelta;

  for (int i = 0; i < numElements; ++i) {
    Vec4r const refTangent = ToSimd(ComputeRodElementTangent(meshNodes, refDisplacement, i));
    Vec4r const outTangent = ToSimd(ComputeRodElementTangent(meshNodes, outDisplacement, i));
    real const twist = dofDelta[i * fem::kNumRodFields + (fem::kNumRodFields - 1)];
    outAxes[i] =
        ToReal3(fem::TransportFrameAxis(refTangent, outTangent, twist, ToSimd(refAxes[i])));
  }
}

} // namespace mochi::rod
