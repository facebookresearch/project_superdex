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

#include <mochi_core/utils/matrix_utils.h>

namespace mochi {

// Get the deformation gradient. Non-SIMD reference implementation.
template <int kNumDofs, int kSpaceDim>
MOCHI_FORCE_INLINE NdArray<real, kSpaceDim, kSpaceDim> GetDeformationGrad(
    NdArray<real, kNumDofs, kSpaceDim> const& dBasisEvaluated,
    Span<real const> displacements) {
  NdArray<real, kSpaceDim, kSpaceDim> F = Eye<kSpaceDim>();

  // Reconstruct deformation gradient at quad point
  for (int f = 0; f < kNumDofs; ++f) {
    size_t const offset = f * kSpaceDim;
    Real3 const displacement3 = {
        displacements[offset + 0], displacements[offset + 1], displacements[offset + 2]};
    F += Outer(displacement3, dBasisEvaluated[f]);
  }

  return F;
}

// Get the deformation gradient. SIMD version specialized for 3D tetrahedrons.
MOCHI_FORCE_INLINE NdArray<Vec4r, 3> GetDeformationGrad(
    NdArray<Vec4r, 4> const& dBasisEvaluated,
    NdArray<Vec4r, 4> const& displacements) {
  NdArray<Vec4r, 3> F = {
      Vec4r{1_r, 0_r, 0_r, 0_r},
      Vec4r{0_r, 1_r, 0_r, 0_r},
      Vec4r{0_r, 0_r, 1_r, 0_r},
  };

  // Reconstruct deformation gradient at quad point
  for (int f = 0; f < 4; ++f) {
    F[0] += Broadcast<0>(displacements[f]) * dBasisEvaluated[f];
    F[1] += Broadcast<1>(displacements[f]) * dBasisEvaluated[f];
    F[2] += Broadcast<2>(displacements[f]) * dBasisEvaluated[f];
  }

  return F;
}

// This function is specialized for 3D 3-node triangles, and computes a 3x3 deformation gradient.
// It defaults to assuming plane-strain kinematics, where there is no transverse shear or
// out-of-plane stretch (but there may be a rotation of the normal). This is safe to use with
// general 3D material models, but may still require a change of parameters to approximate
// plane-stress if one wishes to model unconstrained shell structures, because this deformation
// gradient corresponds to plane-strain.
//
// Optionally, a cheaper deformation gradient can be computed with kAssumePlaneStrain = false, but
// this is only suitable for use in 2D materials that are insensitive to the transverse shear
// components (which will be spurious).  Note that the argument referencePositions is unused if
// kAssumePlaneStrain==false.
template <bool kAssumePlaneStrain = true>
MOCHI_FORCE_INLINE NdArray<Vec4r, 3> GetDeformationGrad(
    NdArray<Vec4r, 3> const& dBasisEvaluated,
    // Reference positions are passed as non-SIMD vectors, since that is the format in which
    // physical-space nodal coordinates are immediately-accessible for elements.
    NdArray<Vec4r, 3> const& referencePositions,
    NdArray<Vec4r, 3> const& displacements) {
  NdArray<Vec4r, 3> F = {
      Vec4r{1_r, 0_r, 0_r, 0_r},
      Vec4r{0_r, 1_r, 0_r, 0_r},
      Vec4r{0_r, 0_r, 1_r, 0_r},
  };
  for (int f = 0; f < 3; ++f) {
    F[0] += Broadcast<0>(displacements[f]) * dBasisEvaluated[f];
    F[1] += Broadcast<1>(displacements[f]) * dBasisEvaluated[f];
    F[2] += Broadcast<2>(displacements[f]) * dBasisEvaluated[f];
  }

  if constexpr (kAssumePlaneStrain) {
    // Get reference and current normals:
    NdArray<Vec4r, 3> currentPositions = referencePositions + displacements;
    Vec4r const referenceNormal = Normalize(Cross3(
        referencePositions[1] - referencePositions[0],
        referencePositions[2] - referencePositions[0]));
    Vec4r const currentNormal = Normalize(Cross3(
        currentPositions[1] - currentPositions[0], currentPositions[2] - currentPositions[0]));

    // Apply plane-strain assumption to the deformation gradient:
    F += Outer3(currentNormal - referenceNormal, referenceNormal);
  }
  return F;
}

} // namespace mochi
