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

#include <mochi_core/materials/batched_active_neo_hookean.h>

#include <mochi_core/materials/batched_active_aniso_arap.h>
#include <mochi_core/materials/batched_smith_neo_hookean.h>
#include <mochi_core/materials/material_utils.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/simd.h>

namespace mochi::materials {

template <int kBatchSize>
void BatchedActiveNeoHookeanConstitutiveResponse(
    BatchActiveNeoHookeanParams<kBatchSize> const& params,
    BatchReal3x3<kBatchSize> const& F,
    BatchDouble<kBatchSize>* outEnergy,
    BatchReal3x3<kBatchSize>* outPK1,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>* outTangent,
    bool projectPsd) {
  // TODO: Both sub-materials independently compute RotationVariantSvd3x3Batch. A fused
  // implementation that computes the SVD once and passes U, sigma, VT to both would eliminate one
  // redundant SVD per call.
  using Vd = BatchDouble<kBatchSize>;
  using V3x3 = BatchReal3x3<kBatchSize>;

  // Passive isotropic component (Smith Neo-Hookean).
  BatchedSmithNeoHookeanConstitutiveResponse<kBatchSize>(
      params.lame, F, outEnergy, outPK1, outTangent, projectPsd);

  // Active anisotropic component (Active Aniso ARAP).
  Vd anisoEnergy{0.0};
  V3x3 anisoPK1 MOCHI_NO_INIT;
  NdArray<V3x3, 3, 3> anisoTangent MOCHI_NO_INIT;
  BatchedActiveAnisoArapConstitutiveResponse<kBatchSize>(
      params.aniso,
      F,
      outEnergy ? &anisoEnergy : nullptr,
      outPK1 ? &anisoPK1 : nullptr,
      outTangent ? &anisoTangent : nullptr,
      projectPsd);

  if (outEnergy) {
    *outEnergy += anisoEnergy;
  }

  if (outPK1) {
    *outPK1 += anisoPK1;
  }

  if (outTangent) {
    *outTangent += anisoTangent;
  }
}

MOCHI_INSTANTIATE_BATCHED_MATERIAL(
    BatchedActiveNeoHookeanConstitutiveResponse,
    BatchActiveNeoHookeanParams);

} // namespace mochi::materials
