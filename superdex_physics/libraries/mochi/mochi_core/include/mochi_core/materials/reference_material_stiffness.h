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

#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array.h>

namespace mochi::materials {

/// @brief Per-element reference (zero-deformation) material stiffness `C₀ = ∂S/∂E|₀`, stored as a
/// symmetric 6×6 Voigt matrix per element.
///
/// @details Deformation-independent quantity consumed by stiffness (viscous) damping. Follows the
/// same size convention as @ref PerElementLameParams: `size()==1` → homogeneous (single tensor
/// broadcast to all elements), `size()==N` → heterogeneous (one tensor per element), empty → not
/// built (damping disabled). Voigt ordering is [00, 11, 22, 12, 02, 01].
struct PerElementReferenceMaterialStiffness {
  /// Symmetric 6×6 Voigt stiffness `C₀`, one per element.
  DynamicArray<NdArray<real, 6, 6>> data;

  /// True iff every element's `C₀` is isotropic by construction (2-parameter λ,μ Voigt form),
  /// enabling the faster isotropic contraction path in @ref mochi::fem::StressDampingWork. Set from
  /// @ref kIsotropicReferenceStiffness at material-set time. Default `false` uses the dense
  /// fallback (always correct).
  bool isIsotropic = false;

  [[nodiscard]] auto size() const {
    return data.size();
  }
  [[nodiscard]] bool empty() const {
    return data.empty();
  }
};

/// @brief Form the reference material stiffness `C₀` (symmetric 6×6 Voigt) for a batch of elements
/// from a batched constitutive response.
///
/// @details Evaluates @p constitutiveResponse at the identity deformation gradient, which returns
/// the two-point tangent `∂P/∂F|₀ = C₀ + δ⊗S₀` and the rest stress `S₀ = S(E=0) = P(F=I)`. The
/// geometric rest-stress part is stripped while packing into Voigt: `C₀_{iAkL} = (∂P/∂F)_{iAkL} −
/// δ_{ik}·S₀_{LA}`, so `c0v[a][b] = (∂P/∂F)_{i_a j_a i_b j_b} − [i_a==i_b]·S₀_{j_b j_a}`. This is
/// the single source of truth for forming the tensor; it is a no-op for passive materials
/// (`S₀ = 0`) and corrects active materials whose rest stress is nonzero.
///
/// @param[in] elementIndices  Indices into the material's per-element params for each batch lane.
/// @param[in] constitutiveResponse  Batched constitutive response callback.
/// @return The symmetric 6×6 Voigt stiffness for each lane.
template <int kBatchSize, class ConstitutiveResponseFn>
[[nodiscard]] NdArray<BatchReal<kBatchSize>, 6, 6> ComputeReferenceMaterialStiffnessVoigt(
    NdArray<int, kBatchSize> const& elementIndices,
    ConstitutiveResponseFn const& constitutiveResponse) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  using V3x3 = BatchReal3x3<kBatchSize>;

  // Voigt index map [00, 11, 22, 12, 02, 01].
  constexpr int kVi[6] = {0, 1, 2, 1, 0, 0};
  constexpr int kVj[6] = {0, 1, 2, 2, 2, 1};

  V3x3 S0 MOCHI_NO_INIT;
  NdArray<V3x3, 3, 3> dPdF MOCHI_NO_INIT;
  constitutiveResponse(
      elementIndices, Eye<3, V>(), static_cast<Vd*>(nullptr), &S0, &dPdF, /*projectPsd*/ false);

  NdArray<V, 6, 6> c0v MOCHI_NO_INIT;
  for (int a = 0; a < 6; ++a) {
    for (int b = 0; b < 6; ++b) {
      V c = dPdF[kVi[a]][kVj[a]][kVi[b]][kVj[b]];
      if (kVi[a] == kVi[b]) {
        c -= S0[kVj[b]][kVj[a]];
      }
      c0v[a][b] = c;
    }
  }
  return c0v;
}

/// @brief Build a per-element reference material stiffness store from a batched constitutive
/// response.
///
/// @details Evaluates @ref ComputeReferenceMaterialStiffnessVoigt once per entry (batch size 1) and
/// stores lane 0. @p numEntries is 1 for homogeneous materials and N for heterogeneous materials
/// (matching the per-element param convention). @p constitutiveResponse must gather its parameters
/// by the element index it is passed.
///
/// @param[in] isIsotropic  Whether the material's `C₀` is isotropic by construction (from @ref
///   kIsotropicReferenceStiffness); cached on the store to select the fast contraction path. Only
///   pass `true` when every element is isotropic.
template <class ConstitutiveResponseFn>
[[nodiscard]] PerElementReferenceMaterialStiffness BuildPerElementReferenceMaterialStiffness(
    ConstitutiveResponseFn const& constitutiveResponse,
    int numEntries,
    bool isIsotropic) {
  MOCHI_ASSERT_VERBOSE(numEntries > 0, "numEntries must be positive.");
  PerElementReferenceMaterialStiffness out;
  out.isIsotropic = isIsotropic;
  out.data.resize_noinit(numEntries);
  for (int i = 0; i < numEntries; ++i) {
    NdArray<int, 1> idx MOCHI_NO_INIT;
    idx[0] = i;
    auto const c0v = ComputeReferenceMaterialStiffnessVoigt<1>(idx, constitutiveResponse);
    for (int a = 0; a < 6; ++a) {
      for (int b = 0; b < 6; ++b) {
        out.data[i][a][b] = c0v[a][b][0]; // lane 0
      }
    }
  }
  return out;
}

/// @brief Gather the reference material stiffness for a batch of elements.
///
/// @details Broadcasts the single tensor to all lanes when @p store is homogeneous (`size()==1`),
/// otherwise gathers per lane by @p elementIndices.
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<BatchReal<kBatchSize>, 6, 6>
GatherReferenceMaterialStiffnessVoigt(
    PerElementReferenceMaterialStiffness const& store,
    NdArray<int, kBatchSize> const& elementIndices) {
  using V = BatchReal<kBatchSize>;
  MOCHI_ASSERT_VERBOSE(!store.empty(), "Reference material stiffness store is empty.");
  NdArray<V, 6, 6> out MOCHI_NO_INIT;
  if (store.size() == 1) {
    auto const& c0 = store.data[0];
    for (int a = 0; a < 6; ++a) {
      for (int b = 0; b < 6; ++b) {
        out[a][b] = V{c0[a][b]};
      }
    }
  } else {
    alignas(alignof(V)) real staging[V::kSize]{};
    for (int a = 0; a < 6; ++a) {
      for (int b = 0; b < 6; ++b) {
        for (int lane = 0; lane < kBatchSize; ++lane) {
          staging[lane] = store.data[elementIndices[lane]][a][b];
        }
        out[a][b] = Load<V>(staging);
      }
    }
  }
  return out;
}

} // namespace mochi::materials
