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

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/rodrigues_utils.h>
#include <mochi_core/utils/transform_rt.h>

namespace mochi::articulated {

/************************************************************************************************/
// Exposed interface
/************************************************************************************************/

// Hessian tensor (H) is a 3D tensor of size:
// (|d|=RigidSize::kDAll*numLinks, |i|=#ReducedDofs, |j|=#ReducedDofs)
//  H(d,i,j) = ArticulatedHessian[j](d,i) = ∂²m_d/∂qᵢ∂qⱼ
// m_d: the component of full DoF (same meaning as the row index of the dense Jacobian)
// q_i, q_j: the component of reduced DoF (same meaning as the column index of the dense Jacobian)
using ArticulatedHessian = DynamicArray<Matrix<real>>;

// Compute Hessian tensor
void Hessian(
    Span<ArticulatedDofInfo const> dofInfo,
    Span<Real3 const> jointAxes,
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT const> jointTransforms,
    Span<TransformRT const> linkTransforms,
    RowMatrixView<real const> jacobian,
    ArticulatedHessian& outHessian);

// The articulated Hessian differentiates a locally transported Jacobian. For a Lie perturbation
// delta around q, define
//
//   J_flat(delta) = T_out(delta) J_world(q ⊞ delta) T_in(delta),
//   H_flat(d,i,j) = d J_flat(d,i) / d delta[j] at delta = 0,
//   D_j J_world = d J_world(q ⊞ delta) / d delta[j] at delta = 0.
//
// T_out expresses each perturbed link twist in the reference link tangent frame, and T_in maps
// reference reduced-coordinate perturbations into the perturbed joint tangent frames. Both are the
// identity at delta = 0, but their derivatives are generally nonzero for rotational coordinates.
// Therefore,
//
//   D_j J_world = H_flat[:,:,j]
//                 - T_out'[j] J_world
//                 - J_world T_in'[j].
//
// Here lambda = inContractedVector. Contract H_flat with this full-space covector:
//
//   outHessianContracted(i,j) += sum_d lambda[d] H_flat(d,i,j).
//
// This operation retains both Lie transport derivatives; it does not return the derivative of the
// ordinary world-space Jacobian.
void HessianContract(
    Span<ArticulatedDofInfo const> dofInfo,
    Span<Real3 const> jointAxes,
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT const> jointTransforms,
    Span<TransformRT const> linkTransforms,
    Span<real const> inContractedVector,
    RowMatrixView<real const> jacobian,
    RowMatrixView<real> outHessianContracted);

// Double-contract the transported articulated Hessian defined above, with
// lambda = inContractedVector and v = reducedVector:
//
//   out[j] += sum_d sum_i lambda[d] H_flat(d,i,j) v[i]
//           = lambda^T H_flat[:,:,j] v.
//
// As with HessianContract, this result retains the derivatives of T_out and T_in.
// outHessianDoubleContracted must not overlap reducedVector or jacobian.
void HessianDoubleContract(
    Span<ArticulatedDofInfo const> dofInfo,
    Span<Real3 const> jointAxes,
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT const> jointTransforms,
    Span<TransformRT const> linkTransforms,
    Span<real const> inContractedVector,
    RowMatrixView<real const> jacobian,
    Span<real const> reducedVector,
    ColumnVectorView<real> outHessianDoubleContracted);

// Double-contract the derivative of the ordinary world-space Jacobian:
//
//   out[j] += lambda^T (D_j J_world) v
//           = lambda^T H_flat[:,:,j] v
//             - lambda^T T_out'[j] J_world v
//             - lambda^T J_world T_in'[j] v.
//
// Unlike HessianDoubleContract, this removes both Lie transport derivatives.
// The implementation folds these input- and output-transport corrections into the articulated
// reverse traversal rather than materializing either correction matrix. Use this operation when
// differentiating a world-space quantity such as J_world(q) v.
// outJacobianDerivativeDoubleContracted must not overlap reducedVector or jacobian.
void JacobianDerivativeDoubleContract(
    Span<ArticulatedDofInfo const> dofInfo,
    Span<Real3 const> jointAxes,
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT const> jointTransforms,
    Span<TransformRT const> linkTransforms,
    Span<real const> inContractedVector,
    RowMatrixView<real const> jacobian,
    Span<real const> reducedVector,
    ColumnVectorView<real> outJacobianDerivativeDoubleContracted);

} // namespace mochi::articulated
