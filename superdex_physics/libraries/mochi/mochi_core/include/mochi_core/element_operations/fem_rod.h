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

#include <mochi_core/element_operations/batched_element_utils.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/local_to_global_map.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>

#include <limits>
#include <utility>

/*
 * This file implements element/stencil-level kernels for a Kirchhoff rod formulation with axial,
 * bending, and torsional deformation, but no transverse shear (as opposed to Cosserat rods, which
 * include transverse shear but require additional degrees of freedom).
 *
 * Some notes on different aspects of the formulation:
 *
 * Inertia and gravity: These assume lumped nodal masses.
 *
 * Axial deformation: This is equivalent to standard 1D linear finite elements, using a 1D reduced
 * St. Venant--Kirchhoff constitutive model that is linear in the axial Green--Lagrange strain.
 *
 * Bending: This uses the binormal curvature discretization from the discrete elastic rods
 * formulation from computer graphics (https://www.cs.columbia.edu/cg/rods/), which infintely
 * penalizes 180-degree folds between adjacent elements.
 *
 * Torsion: This leverages the Kirchhoff-rod zero transverse shear assumption to express element
 * rotations in terms of the centerline nodal displacement DoFs and a single additional scalar twist
 * angle per element (grouped with the displacement DoFs of the element's first node, for
 * bookkeeping). To avoid robustness issues for large deformations, the scalar twist is applied
 * relative to a "parallel transport in time" rotation of the element's material frame from the
 * previous-step to current centerline orientation (cf. the discrete viscous threads formulation,
 * https://www.cs.columbia.edu/cg/threads/). The twist energy's discrete rate of change of
 * cross-section axes w.r.t. the axial coordinate also uses the same formula as binormal curvature,
 * to robustly block adjacent elements from twisting by more than 180 degrees relative to each
 * other.
 */

namespace mochi::fem {

int constexpr kNumRodFields = 4;
int constexpr kRodThetaDofOffset = kNumRodFields - 1;
int constexpr kNumRodStencilNodes = 3;
int constexpr kNumRodStencilDofs = kNumRodStencilNodes * kNumRodFields;

// Minimal rotation transforming n0 to n
MOCHI_FORCE_INLINE VMatrix3x3r ParallelTransportOperator(Vec4r const& n0, Vec4r const& n) {
  Vec4r const n0CrossN = Cross3(n0, n);
  real const n0DotN = Dot<3>(n0, n);
  return VDiagonalMatrix<3>(n0DotN) + Skew3(n0CrossN) +
      1_r / Max(1_r + n0DotN, std::numeric_limits<real>::min()) * Outer3(n0CrossN, n0CrossN);
}

// This assumes (without checking) that the input vectors are normalized. This is equivalent to the
// formula in terms of un-normalized vectors, except the product of magnitudes in the denominator
// has been simplified to 1.
MOCHI_FORCE_INLINE Vec4r IntegratedCurvatureBinormal(Vec4r const& e0Hat, Vec4r const& e1Hat) {
  // The denominator should always be positive if the assumption of unit vector inputs is met,
  // except in the case of exactly-opposite orientations, where it goes singular. It is regularized
  // here to avoid NaN or sign flips.
  real const onePlusDot = Max(1_r + Dot<3>(e0Hat, e1Hat), std::numeric_limits<real>::min());
  return (2_r / onePlusDot) * Cross3(e0Hat, e1Hat);
}

// Alternate version of Rodrigues identity where the rotation axis is assumed to be a unit vector,
// and the angle is given separately (rather than as the magnitude), since the rotation angle
// naturally appears as a separate scalar DoF in this formulation.
MOCHI_FORCE_INLINE VMatrix3x3r RotationAboutAxis(Vec4r const& axis, real angle) {
  VMatrix3x3r const aa = Outer3(axis, axis);
  return aa + Cos(angle) * (VEye<3>() - aa) + Sin(angle) * Skew3(axis);
}

// Parallel-transports a frame axis from baseTangent to newTangent, then rotates by twist about
// newTangent. Both tangent vectors must be unit-length.
MOCHI_FORCE_INLINE Vec4r TransportFrameAxis(
    Vec4r const& baseTangent,
    Vec4r const& newTangent,
    real twist,
    Vec4r const& frameAxis) {
  // Uses v^T*M via DotVecMat3x3 with transposed operators: PT(a,b)^T = PT(b,a), R(n,θ)^T = R(n,-θ).
  Vec4r const parallelAxis =
      DotVecMat3x3(frameAxis, ParallelTransportOperator(newTangent, baseTangent));
  return DotVecMat3x3(parallelAxis, RotationAboutAxis(newTangent, -twist));
}

// Computes the rotation matrix for a rod element between nodes X0 and X1 in the reference
// configuration, in the actor-local coordiantes. The span elementDofs is assumed to be the
// displacement/twist DoFs for these two nodes in order, i.e., [ux_0, uy_0, uz_0, theta_0, ux_1,
// uy_1, uz_1, theta_1]. The frameAxis must be orthogonal to the deformed element tangent. The
// derivative of this rotation matrix w.r.t. these DoFs can be optionally computed if a non-null
// pointer is passed for outDRotation.
MOCHI_FORCE_INLINE void ComputeRodElementRotationLocal(
    Vec4r const& X0,
    Vec4r const& X1,
    Vec4r const& frameAxis,
    Span<real const> elementDofs,
    VMatrix3x3r& outRotation,
    NdArray<real, 3, 3, 8>* outDRotation) {
  MOCHI_ASSERT_VERBOSE(
      elementDofs.size() == 8,
      "Expected the displacement/twist DoFs for these two nodes in order, i.e., "
      "[ux_0, uy_0, uz_0, theta_0, ux_1, uy_1, uz_1, theta_1]");

  Vec4r const x0 = X0 + Load<Vec4r>(&(elementDofs[0]));
  Vec4r const x1 = X1 + Load<Vec4r>(&(elementDofs[kNumRodFields]));
  Vec4r const e = x1 - x0;
  Vec4r const eHat = Normalize<3>(e);
  Vec4r const b = Cross3(eHat, frameAxis);
  outRotation = Transpose3x3(VMatrix3x3r{eHat, frameAxis, b});
  if (outDRotation) {
    // Simplified using orthogonality of frameAxis and eHat.
    VMatrix3x3r const dframeAxis_deHat = Outer3(frameAxis, eHat) - Outer3(eHat, frameAxis);
    VMatrix3x3r const deHat_de = DNormalize3(e);
    VMatrix3x3r const dframeAxis_de = Dot3x3(dframeAxis_deHat, deHat_de);
    VMatrix3x3r const dCross_de = Dot3x3(Outer3(b, eHat) - Skew3(frameAxis), deHat_de);
    for (int i = 0; i < 3; i++) { // i = row of rotation being differentiated
      // Columns of rotation specified one-by-one, based on rotation's construction
      Store(&((*outDRotation)[i][0][0]), -deHat_de[i]);
      Store(&((*outDRotation)[i][1][0]), -dframeAxis_de[i]);
      Store(&((*outDRotation)[i][2][0]), -dCross_de[i]);
      Store(&((*outDRotation)[i][0][kNumRodFields]), deHat_de[i]);
      Store(&((*outDRotation)[i][1][kNumRodFields]), dframeAxis_de[i]);
      Store(&((*outDRotation)[i][2][kNumRodFields]), dCross_de[i]);
      // Derivatives w.r.t. angular DoFs (overwrites padding written by SIMD Store operations above)
      int const angleDof0 = kRodThetaDofOffset;
      (*outDRotation)[i][0][angleDof0] = 0_r;
      (*outDRotation)[i][1][angleDof0] = b[i];
      (*outDRotation)[i][2][angleDof0] = -frameAxis[i];
      int const angleDof1 = angleDof0 + kNumRodFields;
      for (int j = 0; j < 3; j++) {
        (*outDRotation)[i][j][angleDof1] = 0_r;
      }
    } // i
  } // if computing derivative
}

// Computes the Jacobian of an embedded point's position with respect to the contributing
// element's DoFs. An embedded point is defined by local coordinates (xi) relative to the element's
// deformed frame, expressed in a *unit-reference-tangent* basis (so xi[0] is in arc-length units).
//
// The current frame axis is taken as input, NOT computed from reference. The twist DoF is treated
// as zero (it's recentered each time step).
//
// Parameters:
//   X0, X1: Reference positions of element nodes
//   xi: Local coordinates [xi_t, xi_d, xi_b]. xi_t is a signed arc-length offset along the
//       (stretched) unit tangent; xi_d, xi_b are length-units offsets along frame axis & binormal.
//   invReferenceLength: 1 / |X1 - X0|. Used to scale xi_t into the correct multiplier of
//       (x1 - x0) for the deformed configuration.
//   currentFrameAxis: Current deformed frame axis (unit vector, orthogonal to deformed tangent)
//   elementDofs: [ux0, uy0, uz0, θ0, ux1, uy1, uz1, θ1] - twist DoF values are zero in practice
//   outJacobian: 3x8 Jacobian of position w.r.t. element DoFs
inline void ComputeEmbeddedPointElementJacobian(
    Real3 const& X0,
    Real3 const& X1,
    Real3 const& xi,
    real invReferenceLength,
    Real3 const& currentFrameAxis,
    Span<real const> elementDofs,
    NdArray<real, 3, 2 * kNumRodFields>& outJacobian) {
  static_assert(kNumRodFields == 4, "This code assumes 4 DoFs per element");
  MOCHI_ASSERT_VERBOSE(
      elementDofs.size() == 2 * kNumRodFields,
      "Expected the displacement/twist DoFs for two nodes: [ux0, uy0, uz0, θ0, ux1, uy1, uz1, θ1]");

  // Deformed positions and tangent
  Vec4r const x0 = ToSimd(X0) + Load<Vec4r>(&(elementDofs[0]));
  Vec4r const x1 = ToSimd(X1) + Load<Vec4r>(&(elementDofs[kNumRodFields]));
  Vec4r const e = x1 - x0;
  Vec4r const eHat = Normalize<3>(e);
  Vec4r const currentFrameAxis4 = ToSimd(currentFrameAxis);
  Vec4r const binormal = Cross3(eHat, currentFrameAxis4);

  // Frame axis derivatives:
  VMatrix3x3r const deHat_de = DNormalize3(e);
  // Simplified using orthogonality of currentFrameAxis and eHat.
  VMatrix3x3r const dframeAxis_deHat =
      Outer3(currentFrameAxis4, eHat) - Outer3(eHat, currentFrameAxis4);
  VMatrix3x3r const dframeAxis_de = Dot3x3(dframeAxis_deHat, deHat_de);
  // d(d)/dθ at θ=0 = eHat × d = binormal (same vector, different interpretation)
  Vec4r const& dframeAxis_dtheta = binormal;

  // Binormal derivatives: d(eHat × d)/d(e) = skew(-d)·d(eHat)/d(e) + skew(eHat)·d(d)/d(e)
  VMatrix3x3r const skewEHat = Skew3(eHat);
  VMatrix3x3r const db_de_fromEHat = Dot3x3(-Skew3(currentFrameAxis4), deHat_de);
  VMatrix3x3r const db_de_fromD = Dot3x3(skewEHat, dframeAxis_de);

  // Displacement derivative matrices: d(x_vis)/d(x0) and d(x_vis)/d(x1).
  // Forward map (in unit-reference-tangent parametrization):
  //   x_vis = mid + xi[0] · (e · invReferenceLength) + xi[1] · d + xi[2] · b
  // d(x_vis)/d(x) = d(mid)/d(x) + xi[0] · invReferenceLength · d(e)/d(x)
  //               + xi[1] · d(d)/d(e) · d(e)/d(x) + xi[2] · d(b)/d(e) · d(e)/d(x)
  real const xiT = xi[0] * invReferenceLength;
  VMatrix3x3r dFrameAndBinormal_de;
  for (int i = 0; i < 3; ++i) {
    dFrameAndBinormal_de[i] =
        xi[1] * dframeAxis_de[i] + xi[2] * (db_de_fromEHat[i] + db_de_fromD[i]);
  }
  VMatrix3x3r const dxvis_dx0 = VDiagonalMatrix<3>(0.5_r - xiT) - dFrameAndBinormal_de;
  VMatrix3x3r const dxvis_dx1 = VDiagonalMatrix<3>(0.5_r + xiT) + dFrameAndBinormal_de;

  // θ0 derivative: d(xi[1]·d + xi[2]·b)/dθ, where d(d)/dθ = binormal, d(b)/dθ = skew(eHat)·binormal
  Vec4r const theta0Deriv =
      xi[1] * dframeAxis_dtheta + xi[2] * DotMatVec3x3(skewEHat, dframeAxis_dtheta);

  // Store Jacobian
  for (int i = 0; i < 3; ++i) {
    Store(&(outJacobian[i][0]), dxvis_dx0[i]);
    Store(&(outJacobian[i][kNumRodFields]), dxvis_dx1[i]);
    outJacobian[i][kNumRodFields - 1] = theta0Deriv[i];
    outJacobian[i][2 * kNumRodFields - 1] = 0_r;
  }
}

/// @brief Compile-time element tag for the 3-node rod assembly stencil.
///
/// @details Used as the @p ElementT parameter for FEM assembly and batched rod vectors/matrices.
/// Each stencil node contributes @ref kNumRodFields fields through @ref ElOpFnType.
struct RodStencilElement {
  static constexpr int kSpaceDim = 3;
  static constexpr int kNumDofs = kNumRodStencilNodes; // 3
};

template <int kBatchSize>
using BatchRodVector = BatchElementVector<kBatchSize, RodStencilElement, kNumRodFields>;

template <int kBatchSize>
using BatchRodMatrix = BatchElementMatrix<kBatchSize, RodStencilElement, kNumRodFields>;

namespace details {

// ================================================================================================
// Rod kinematic helpers for batch-valued lanes
// ================================================================================================

/// @brief Minimal rotation transforming @p n0 to @p n.
template <class V>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<V, 3, 3> ParallelTransportOperator(
    NdArray<V, 3> const& n0,
    NdArray<V, 3> const& n) {
  NdArray<V, 3> const n0CrossN = Cross(n0, n);
  V const n0DotN = Dot(n0, n);
  V const invDenom = V{1} / Max(V{1} + n0DotN, V{std::numeric_limits<real>::min()});
  return DiagonalMatrix<3>(n0DotN) + Skew(n0CrossN) + invDenom * Outer(n0CrossN, n0CrossN);
}

/// @brief Derivatives of ParallelTransport(n0, n) * v w.r.t. both @p n0 and @p n, computed directly
/// as matrices (no rank-3 tensor). Returns {d/dn0, d/dn}.
template <class V>
[[nodiscard]]
// TODO[T247578555]: This is a workaround for a bug in VS2022, but it may hurt performance.
// A better solution is needed.
#if MOCHI_COMPILER_MSVC
inline
#else
MOCHI_FORCE_INLINE
#endif
    std::pair<NdArray<V, 3, 3>, NdArray<V, 3, 3>> DParallelTransportedVec(
        NdArray<V, 3> const& n0,
        NdArray<V, 3> const& n,
        NdArray<V, 3> const& v) {
  NdArray<V, 3> const cross = Cross(n0, n);
  V const c = V{1} / Max(V{1} + Dot(n0, n), V{std::numeric_limits<real>::min()});
  V const crossDotV = Dot(cross, v);
  NdArray<V, 3> const projV = v - (Sqr(c) * crossDotV) * cross;
  NdArray<V, 3, 3> const innerMat = -Skew(v) + c * (Outer(cross, v) + crossDotV * Eye<3, V>());
  return {Outer(projV, n) + Dot(innerMat, -Skew(n)), Outer(projV, n0) + Dot(innerMat, Skew(n0))};
}

/// @brief Global node index of local stencil node @p localNode for a given element, read from the
/// flattened local-to-global map (so it supports periodic wrap-around and collapsed boundaries).
[[nodiscard]] MOCHI_FORCE_INLINE int
RodStencilNodeIndex(Span<int const> l2gFlat, int elementIndex, int localNode) {
  constexpr int kStride = kNumRodStencilNodes * kNumRodFields;
  return l2gFlat[elementIndex * kStride + localNode * kNumRodFields] / kNumRodFields;
}

} // namespace details

// ================================================================================================
// Rod element operations
// ================================================================================================

/// @brief Compute gravity work for a batch of rod stencils.
///
/// Only the displacement DoFs (first 3 of 4) of node 0 contribute, via the incremental potential
/// -u·(m g). Node 0 always exists, so no lane masking is required. The caller discards padded-lane
/// contributions during scatter.
///
/// @tparam kBatchSize Batch size.
/// @param[in] nodalMasses  Per-node lumped mass array [kg].
/// @param[in] disp  Batched stencil displacement vector (12 DoFs = 3 nodes × 4 fields).
/// @param[in] elementIndices  Stencil indices for each batch lane.
/// @param[in] l2gFlat  Flattened local-to-global map; node indices are read from it.
/// @param[out] outEnergy  If non-null, accumulates per-element energy.
/// @param[out] outRes  If non-null, accumulates per-element residual.
/// @param[in] gravity  Gravity vector [m/s²].
/// @return true if outputs were written.
template <int kBatchSize>
bool RodGravity(
    Span<real const> nodalMasses,
    BatchRodVector<kBatchSize> const& disp,
    NdArray<int, kBatchSize> const& elementIndices,
    Span<int const> l2gFlat,
    BatchDouble<kBatchSize>* outEnergy,
    BatchRodVector<kBatchSize>* outRes,
    Real3 gravity) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;

  bool const evalObj = (outEnergy != nullptr);
  bool const evalRes = (outRes != nullptr);
  if (!evalObj && !evalRes) {
    return false;
  }

  // Gather per-lane nodal mass for node 0 (always valid).
  alignas(alignof(V)) real staging[V::kSize]{};
  for (int b = 0; b < kBatchSize; ++b) {
    int const node0 = details::RodStencilNodeIndex(l2gFlat, elementIndices[b], 0);
    MOCHI_ASSERT_VERBOSE(node0 >= 0 && node0 < isize(nodalMasses), "Node index out of range");
    staging[b] = nodalMasses[node0];
  }
  V const mass = Load<V>(staging);

  V const mgx = mass * V{gravity[0]};
  V const mgy = mass * V{gravity[1]};
  V const mgz = mass * V{gravity[2]};

  if (evalObj) {
    // Energy = -dot(u, m*g).
    // Note: This is a consistent incremental potential that generates the correct residual, but
    // it differs from the textbook gravitational potential energy, in that it is not computed
    // relative to a single zero location in space. This is cheaper, since it doesn't require
    // loading the reference geometry to compute absolute position.
    *outEnergy -= StaticCast<Vd>(disp[0] * mgx + disp[1] * mgy + disp[2] * mgz);
  }
  if (evalRes) {
    (*outRes)[0] -= mgx;
    (*outRes)[1] -= mgy;
    (*outRes)[2] -= mgz;
  }
  return true;
}

/// @brief Compute inertia work for a batch of rod stencils.
///
/// Applies lumped translational mass on the 3 displacement DoFs and rotational inertia on the twist
/// DoF of node 0. Node 0 always exists, so no lane masking is required; the caller discards
/// padded-lane contributions during scatter.
///
/// @tparam kBatchSize Batch size.
/// @param[in] nodalMasses  Per-node lumped mass array [kg].
/// @param[in] elementRotationalInertias  Per-element rotational inertia [kg·m²]. May be shorter
/// than the number of nodes (the last node of an open rod has no element); 0 is used for
/// out-of-range.
/// @param[in] disp  Batched stencil displacement vector.
/// @param[in] predTarget  Batched predicted target (xexp + Δt·vexp) for the 4 DoFs of node 0.
/// @param[in] elementIndices  Stencil indices for each batch lane.
/// @param[in] l2gFlat  Flattened local-to-global map; node indices are read from it.
/// @param[out] outEnergy  If non-null, accumulates per-element energy.
/// @param[out] outRes  If non-null, accumulates per-element residual.
/// @param[out] outDRes  If non-null, accumulates per-element stiffness.
/// @param[in] dtfi2  Inverse squared time step factor: 1/(Δt·α)².
/// @return true if outputs were written.
template <int kBatchSize>
bool RodInertia(
    Span<real const> nodalMasses,
    Span<real const> elementRotationalInertias,
    BatchRodVector<kBatchSize> const& disp,
    NdArray<BatchReal<kBatchSize>, kNumRodFields> const& predTarget,
    NdArray<int, kBatchSize> const& elementIndices,
    Span<int const> l2gFlat,
    BatchDouble<kBatchSize>* outEnergy,
    BatchRodVector<kBatchSize>* outRes,
    BatchRodMatrix<kBatchSize>* outDRes,
    real dtfi2) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;

  bool const evalObj = (outEnergy != nullptr);
  bool const evalRes = (outRes != nullptr);
  bool const evalDRes = (outDRes != nullptr);
  if (!evalObj && !evalRes && !evalDRes) {
    return false;
  }

  // Gather per-lane nodal mass and rotational inertia for node 0.
  int const numRotInertias = isize(elementRotationalInertias);
  alignas(alignof(V)) real massStaging[V::kSize]{};
  alignas(alignof(V)) real rotStaging[V::kSize]{};
  for (int b = 0; b < kBatchSize; ++b) {
    int const node0 = details::RodStencilNodeIndex(l2gFlat, elementIndices[b], 0);
    MOCHI_ASSERT_VERBOSE(node0 >= 0 && node0 < isize(nodalMasses), "Node index out of range");
    massStaging[b] = nodalMasses[node0];
    rotStaging[b] = (node0 < numRotInertias) ? elementRotationalInertias[node0] : 0_r;
  }
  V const dtfi2Mass = dtfi2 * Load<V>(massStaging);
  V const dtfi2Rot = dtfi2 * Load<V>(rotStaging);

  if (evalObj || evalRes) {
    // Acceleration at node 0: x - predTarget.
    NdArray<V, kNumRodFields> accel MOCHI_NO_INIT;
    for (int d = 0; d < kNumRodFields; ++d) {
      accel[d] = disp[d] - predTarget[d];
    }

    if (evalObj) {
      V const dispNormSqr = accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2];
      *outEnergy +=
          StaticCast<Vd>(0.5_r * (dtfi2Mass * dispNormSqr + dtfi2Rot * accel[3] * accel[3]));
    }
    if (evalRes) {
      (*outRes)[0] += dtfi2Mass * accel[0];
      (*outRes)[1] += dtfi2Mass * accel[1];
      (*outRes)[2] += dtfi2Mass * accel[2];
      (*outRes)[3] += dtfi2Rot * accel[3];
    }
  }

  if (evalDRes) {
    constexpr int kRowSize = kNumRodStencilDofs;
    (*outDRes)[0 * kRowSize + 0] += dtfi2Mass;
    (*outDRes)[1 * kRowSize + 1] += dtfi2Mass;
    (*outDRes)[2 * kRowSize + 2] += dtfi2Mass;
    (*outDRes)[3 * kRowSize + 3] += dtfi2Rot;
  }
  return true;
}

/// @brief Compute axial stress work for a batch of rod stencils.
///
/// Uses 1D Green–Lagrange strain with a reduced St. Venant–Kirchhoff model on the left element
/// (nodes 0 and 1) of each stencil. Node indices are read from @p l2gFlat, so node 1 may be
/// non-consecutive (periodic). A boundary stencil whose node 1 collapses onto node 0 is masked to
/// zero.
///
/// @tparam kBatchSize Batch size.
/// @param[in] meshNodes  Reference node positions [m].
/// @param[in] disp  Batched stencil displacement vector.
/// @param[in] elementIndices  Stencil indices for each batch lane.
/// @param[in] l2gFlat  Flattened local-to-global map; node indices and the boundary mask are read
///   from it.
/// @param[out] outEnergy  If non-null, accumulates per-element energy.
/// @param[out] outRes  If non-null, accumulates per-element residual.
/// @param[out] outDRes  If non-null, accumulates per-element stiffness.
/// @param[in] axialStiffness  EA [N].
/// @param[in] projectPsd  If true, project the geometric stiffness to PSD.
/// @param[in] stiffnessDampingFactor  β/dt factor for stiffness damping (dimensionless). Unified
///   into the elastic response by scaling the material stiffness to (1+factor)·EA and using the
///   modified strain ε − (factor/(1+factor))·ε₀ (ε₀ = stage-start strain).
/// @param[in] stageStartDisp  Stage-start displacement stencil (required when
///   @p stiffnessDampingFactor > 0; may be null otherwise).
/// @return true if outputs were written.
template <int kBatchSize>
bool RodAxialStress(
    Span<Real3 const> meshNodes,
    BatchRodVector<kBatchSize> const& disp,
    NdArray<int, kBatchSize> const& elementIndices,
    Span<int const> l2gFlat,
    BatchDouble<kBatchSize>* outEnergy,
    BatchRodVector<kBatchSize>* outRes,
    BatchRodMatrix<kBatchSize>* outDRes,
    real axialStiffness,
    bool projectPsd,
    real stiffnessDampingFactor = 0_r,
    NdArray<BatchReal<kBatchSize>, kNumRodStencilDofs> const* stageStartDisp = nullptr);

/// @brief Compute bending + torsion stress work for a batch of rod stencils.
///
/// Uses discrete-elastic-rod binormal bending plus torsion, using the current per-edge material
/// frame axes @p frameAxes and linearizing the twist about angle 0. The per-step
/// parallel-transport-in-time frame update is performed externally by the caller. Requires a full
/// 3-node stencil; a boundary stencil whose padded map collapses two of the three nodes is masked
/// to zero.
///
/// Precondition: for each lane, a0 must be orthogonal to the deformed tangent of element 0
/// (x[1]-x[0]) and a1 to that of element 1 (x[2]-x[1]).
///
/// @tparam kBatchSize Batch size.
/// @param[in] meshNodes  Reference node positions [m].
/// @param[in] frameAxes  Current per-edge material frame axes (axis of edge e stored at node e).
/// @param[in] referenceAxes  Reference per-edge material frame axes.
/// @param[in] disp  Batched stencil displacement vector.
/// @param[in] elementIndices  Stencil indices for each batch lane.
/// @param[in] l2gFlat  Flattened local-to-global map; node indices and the boundary mask are read
///   from it.
/// @param[out] outEnergy  If non-null, accumulates per-element energy.
/// @param[out] outRes  If non-null, accumulates per-element residual.
/// @param[out] outDRes  If non-null, accumulates per-element stiffness.
/// @param[in] flexuralStiffness  Bending stiffness {EI1, EI2} [N·m²].
/// @param[in] torsionalStiffness  Torsional stiffness GJ [N·m²].
/// @param[in] stiffnessDampingFactor  β/dt factor for stiffness damping (dimensionless). Unified
///   into the elastic response by scaling each material stiffness to (1+factor)·k and using the
///   modified strains ε − (factor/(1+factor))·ε₀ (ε₀ = stage-start strain).
/// @param[in] stageStartDisp  Stage-start displacement stencil (required when
///   @p stiffnessDampingFactor > 0; may be null otherwise).
/// @param[in] stageStartFrameAxes  Stage-start per-edge frame axes (required when
///   @p stiffnessDampingFactor > 0; may be empty otherwise).
/// @return true if outputs were written.
template <int kBatchSize>
bool RodBendTwistStress(
    Span<Real3 const> meshNodes,
    Span<Real3 const> frameAxes,
    Span<Real3 const> referenceAxes,
    BatchRodVector<kBatchSize> const& disp,
    NdArray<int, kBatchSize> const& elementIndices,
    Span<int const> l2gFlat,
    BatchDouble<kBatchSize>* outEnergy,
    BatchRodVector<kBatchSize>* outRes,
    BatchRodMatrix<kBatchSize>* outDRes,
    Real2 flexuralStiffness,
    real torsionalStiffness,
    real stiffnessDampingFactor = 0_r,
    NdArray<BatchReal<kBatchSize>, kNumRodStencilDofs> const* stageStartDisp = nullptr,
    Span<Real3 const> stageStartFrameAxes = {});

} // namespace mochi::fem
