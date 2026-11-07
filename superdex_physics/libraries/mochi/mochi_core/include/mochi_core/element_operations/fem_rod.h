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
#include <mochi_core/linear_algebra/krylov_interop.h>
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
 * St. Venant--Kirchhoff constitutive model that is linear in the axial Green--Lagrange strain, plus
 * an optional barrier against axial collapse.
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

// Computes the actor-local material-frame rotation of a rod element with reference nodes X0
// and X1. elementDofs is [ux_0, uy_0, uz_0, theta_0, ux_1, uy_1, uz_1, theta_1]. frameAxis must
// be unit length and orthogonal to the nonzero deformed element tangent, with the current twist
// already incorporated.
// If outJacobian is non-null, it receives the 3x8 Jacobian of the actor-local spatial rotation
// increment omega, with dR = skew(omega) R. This is evaluated at zero incremental twist and
// includes motion of frameAxis with the DoFs, as in TransportFrameAxis: parallel transport for
// translations and rotation about the tangent for theta_0. frameAxis is not held fixed, and
// theta_1 does not affect this element's frame.
MOCHI_FORCE_INLINE void ComputeRodElementRotationLocal(
    Vec4r const& X0,
    Vec4r const& X1,
    Vec4r const& frameAxis,
    Span<real const> elementDofs,
    VMatrix3x3r& outRotation,
    NdArray<real, 3, 2 * kNumRodFields>* outJacobian) {
  MOCHI_ASSERT_VERBOSE(
      elementDofs.size() == 2 * kNumRodFields,
      "Expected the displacement/twist DoFs for these two nodes in order, i.e., "
      "[ux_0, uy_0, uz_0, theta_0, ux_1, uy_1, uz_1, theta_1]");

  Vec4r const x0 = X0 + Load<Vec4r>(&(elementDofs[0]));
  Vec4r const x1 = X1 + Load<Vec4r>(&(elementDofs[kNumRodFields]));
  Vec4r const e = x1 - x0;
  // Match Normalize<3>'s divide-by-zero guard for collapsed elements.
  real const invLength = 1_r / (Sqrt(Dot<3>(e, e)) + std::numeric_limits<real>::min());
  Vec4r const eHat = invLength * e;
  Vec4r const b = Cross3(eHat, frameAxis);
  outRotation = Transpose3x3(VMatrix3x3r{eHat, frameAxis, b});
  if (outJacobian) {
    // Translations rotate the frame by parallel transport, omega = eHat x deHat = skew(eHat)
    // de/|e|; theta_0 rotates it about eHat; theta_1 does not affect it.
    VMatrix3x3r const jacTranslation = invLength * Skew3(eHat);
    auto jac = AsView<real, 3, 2 * kNumRodFields>(*outJacobian);
    jac.LeftCols<3>(3) = AsMatrixView(-jacTranslation);
    jac.Col(kRodThetaDofOffset) = AsColumnVectorView<3>(eHat);
    jac.MiddleCols<3>(kNumRodFields, 3) = AsMatrixView(jacTranslation);
    jac.Col(kNumRodFields + kRodThetaDofOffset).SetZero();
  }
}

// Differential geometry of a deformed rod element frame with respect to its edge vector.
// The frame is [tangent | frameAxis | binormal], where binormal = tangent x frameAxis.
// The derivative matrices map a change in the deformed edge to the corresponding change in each
// frame vector. Twist derivatives are not stored because they have the compact forms
// d(frameAxis)/dtheta = binormal and d(binormal)/dtheta = tangent x binormal.
struct RodElementFrameDifferential {
  Vec4r tangent;
  Vec4r binormal;
  VMatrix3x3r dFrameAxisDEdge;
  VMatrix3x3r dBinormalDEdge;
};

// Computes the frame and its edge derivatives. currentFrameAxis must be unit length and orthogonal
// to the deformed element tangent.
MOCHI_FORCE_INLINE RodElementFrameDifferential
ComputeRodElementFrameDifferential(Vec4r const& deformedEdge, Vec4r const& currentFrameAxis) {
  Vec4r const tangent = Normalize<3>(deformedEdge);
  Vec4r const binormal = Cross3(tangent, currentFrameAxis);
  VMatrix3x3r const dTangentDEdge = DNormalize3(deformedEdge);

  // The frame axis follows changes in the tangent by infinitesimal parallel transport.
  VMatrix3x3r const dFrameAxisDTangent =
      Outer3(currentFrameAxis, tangent) - Outer3(tangent, currentFrameAxis);
  VMatrix3x3r const dFrameAxisDEdge = Dot3x3(dFrameAxisDTangent, dTangentDEdge);

  // Differentiate binormal = tangent x frameAxis, including both tangent and frame-axis motion.
  VMatrix3x3r const dBinormalDEdge =
      Dot3x3(Outer3(binormal, tangent) - Skew3(currentFrameAxis), dTangentDEdge);
  return {tangent, binormal, dFrameAxisDEdge, dBinormalDEdge};
}

// Computes an embedded point's 3x8 position Jacobian from precomputed element differential
// geometry. The embedded point uses local coordinates xi = [xi_t, xi_d, xi_b] in the deformed
// element frame. xi_t is in arc-length units and is scaled by invReferenceLength; xi_d and xi_b
// are offsets along the frame axis and binormal. The element DoFs are ordered
// [ux0, uy0, uz0, theta0, ux1, uy1, uz1, theta1].
inline void ComputeEmbeddedPointElementJacobian(
    Real3 const& xi,
    real invReferenceLength,
    RodElementFrameDifferential const& differential,
    NdArray<real, 3, 2 * kNumRodFields>& outJacobian) {
  static_assert(kNumRodFields == 4, "This code assumes 4 DoFs per element");

  // Forward map in unit-reference-tangent coordinates:
  //   x_vis = mid + xi[0] * (edge * invReferenceLength) + xi[1] * frameAxis
  //                 + xi[2] * binormal.
  // Therefore d(x_vis)/d(edge) combines the scaled tangent term with changes in the frame axis
  // and binormal. Since edge = x1 - x0, this contribution has opposite signs for the two nodes.
  real const xiT = xi[0] * invReferenceLength;
  VMatrix3x3r dFrameAndBinormalDEdge;
  for (int i = 0; i < 3; ++i) {
    dFrameAndBinormalDEdge[i] =
        xi[1] * differential.dFrameAxisDEdge[i] + xi[2] * differential.dBinormalDEdge[i];
  }
  VMatrix3x3r const dxvis_dx0 = VDiagonalMatrix<3>(0.5_r - xiT) - dFrameAndBinormalDEdge;
  VMatrix3x3r const dxvis_dx1 = VDiagonalMatrix<3>(0.5_r + xiT) + dFrameAndBinormalDEdge;

  // Only the first node owns the element twist DoF. At zero incremental twist,
  // d(frameAxis)/dtheta = binormal and d(binormal)/dtheta = tangent x binormal.
  Vec4r const theta0Deriv =
      xi[1] * differential.binormal + xi[2] * Cross3(differential.tangent, differential.binormal);

  for (int i = 0; i < 3; ++i) {
    Store(&(outJacobian[i][0]), dxvis_dx0[i]);
    Store(&(outJacobian[i][kNumRodFields]), dxvis_dx1[i]);
    outJacobian[i][kNumRodFields - 1] = theta0Deriv[i];
    outJacobian[i][2 * kNumRodFields - 1] = 0_r;
  }
}

// Convenience overload that computes the element differential geometry before evaluating the
// embedded-point Jacobian.
//
// Parameters:
//   X0, X1: Reference positions of the element nodes.
//   xi: Local coordinates [xi_t, xi_d, xi_b]. xi_t is a signed arc-length offset along the
//       stretched tangent; xi_d and xi_b are length-unit offsets along frame axis and binormal.
//   invReferenceLength: 1 / |X1 - X0|, used to scale xi_t for the deformed configuration.
//   currentFrameAxis: Current unit frame axis, orthogonal to the deformed element tangent.
//   elementDofs: [ux0, uy0, uz0, theta0, ux1, uy1, uz1, theta1]. The twist DoFs are zero in the
//       stored pose because they are recentered each time step.
//   outJacobian: 3x8 Jacobian of embedded-point position with respect to the element DoFs.
inline void ComputeEmbeddedPointElementJacobian(
    Real3 const& X0,
    Real3 const& X1,
    Real3 const& xi,
    real invReferenceLength,
    Real3 const& currentFrameAxis,
    Span<real const> elementDofs,
    NdArray<real, 3, 2 * kNumRodFields>& outJacobian) {
  MOCHI_ASSERT_VERBOSE(
      elementDofs.size() == 2 * kNumRodFields,
      "Expected the displacement/twist DoFs for two nodes: [ux0, uy0, uz0, θ0, ux1, uy1, uz1, θ1]");

  Vec4r const x0 = ToSimd(X0) + Load<Vec4r>(&(elementDofs[0]));
  Vec4r const x1 = ToSimd(X1) + Load<Vec4r>(&(elementDofs[kNumRodFields]));
  ComputeEmbeddedPointElementJacobian(
      xi,
      invReferenceLength,
      ComputeRodElementFrameDifferential(x1 - x0, ToSimd(currentFrameAxis)),
      outJacobian);
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
/// Only the displacement DoFs (first 3 of 4) of node 0 contribute, via the potential
/// -(u - u₀)·(m g) relative to the stage-start displacement u₀. Node 0 always exists, so no lane
/// masking is required. The caller discards padded-lane contributions during scatter.
///
/// @tparam kBatchSize Batch size.
/// @param[in] nodalMasses  Per-node lumped mass array [kg].
/// @param[in] disp  Batched stencil displacement vector (12 DoFs = 3 nodes × 4 fields).
/// @param[in] stageStartDisp  Batched stencil stage-start displacement vector. Only node 0's
/// displacement DoFs are read, and only if @p outEnergy is non-null.
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
    BatchRodVector<kBatchSize> const& stageStartDisp,
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
    // Energy = -dot(u - u₀, m*g).
    *outEnergy -= StaticCast<Vd>(
        (disp[0] - stageStartDisp[0]) * mgx + (disp[1] - stageStartDisp[1]) * mgy +
        (disp[2] - stageStartDisp[2]) * mgz);
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
/// (nodes 0 and 1) of each stencil, plus an optional collapse barrier. Node indices are read from
/// @p l2gFlat, so node 1 may be non-consecutive (periodic). A boundary stencil whose node 1
/// collapses onto node 0 is masked to zero.
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
/// @param[in] axialCollapseBarrierStiffness  Stiffness k ≥ 0 [N] of the axial collapse barrier
///   k·⟨λ⁻² − 1⟩³ per unit reference length, with stretch λ; skipped when zero. Its stiffness uses
///   a PSD Gauss–Newton approximation, and it is excluded from stiffness damping. Below a stretch
///   of 2⁻⁸ in single precision (2⁻⁶⁴ in double), its energy continues linearly in λ² and its
///   stiffness saturates, to avoid overflow; the residual stays the exact derivative of the energy.
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
    real axialCollapseBarrierStiffness,
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
