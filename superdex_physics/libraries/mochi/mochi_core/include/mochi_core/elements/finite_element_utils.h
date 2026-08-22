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
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

namespace mochi {

// Evaluate the value of the field at a point given the value of the basis functions evaluated at
// that point
template <int kNumFields, int kNumDofs>
inline void EvaluateField(
    NdArray<real, kNumDofs> const& basisEvaluated,
    NdArray<real, kNumDofs, kNumFields> const& dofValues,
    NdArray<real, kNumFields>* outField) {
  MOCHI_ASSERT(outField);

  if constexpr (kNumFields == 3 && kNumDofs == 4) {
    NdArray<Vec4r, 4> dofValuesV;
    LoadMatrix<4, 3>(dofValuesV, dofValues);

    Vec4r fieldValue;
    auto const basisEvaluatedV = Load<Vec4r>(basisEvaluated.data());
    fieldValue = //
        (Broadcast<0>(basisEvaluatedV) * dofValuesV[0]) +
        (Broadcast<1>(basisEvaluatedV) * dofValuesV[1]) +
        (Broadcast<2>(basisEvaluatedV) * dofValuesV[2]) +
        (Broadcast<3>(basisEvaluatedV) * dofValuesV[3]);

    Store<3>(outField->data(), fieldValue);
  } else {
    *outField = {};
    // Linear combination of basis functions scaled by degrees of freedom
    for (int f = 0; f < kNumDofs; ++f) {
      *outField += basisEvaluated[f] * dofValues[f];
    }
  }
}

// Evaluate the value of the gradient of the field at a point given the value of the gradient of the
// basis functions evaluated at that point
template <int kNumFields, int kNumDofs, int kSpaceDim>
inline void EvaluateFieldGradient(
    NdArray<real, kNumDofs, kSpaceDim> const& dBasisEvaluated,
    NdArray<real, kNumDofs, kNumFields> const& dofValues,
    NdArray<real, kNumFields, kSpaceDim>* outFieldGradient) {
  MOCHI_ASSERT(outFieldGradient);

  if constexpr (kNumFields == 3 && kNumDofs == 4) {
    // Pad displacements for SIMD
    NdArray<Vec4r, 4> dofValuesV;
    LoadMatrix<4, 3>(dofValuesV, dofValues);

    NdArray<Vec4r, 4> dBasisEvaluatedV;
    LoadMatrix(dBasisEvaluatedV, dBasisEvaluated);

    NdArray<Vec4r, 3> fieldGradientV{};
    for (int f = 0; f < 4; ++f) {
      fieldGradientV[0] += Broadcast<0>(dofValuesV[f]) * dBasisEvaluatedV[f];
      fieldGradientV[1] += Broadcast<1>(dofValuesV[f]) * dBasisEvaluatedV[f];
      fieldGradientV[2] += Broadcast<2>(dofValuesV[f]) * dBasisEvaluatedV[f];
    }

    StoreMatrix<3, 3>(*outFieldGradient, fieldGradientV);

  } else {
    *outFieldGradient = {};
    for (int f = 0; f < kNumDofs; ++f) {
      // Get the map of the quadrature point
      *outFieldGradient += Outer(dofValues[f], dBasisEvaluated[f]);
    }
  }
}
} // namespace mochi
