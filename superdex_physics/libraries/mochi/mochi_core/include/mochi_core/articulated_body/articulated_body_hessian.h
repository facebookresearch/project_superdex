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
// m_d: the component of full DOF (same meaning as the row index of the Jacobian)
// q_i, q_j: the component of reduced DOF (same meaning as the column index of the Jacobian)
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

// Compute Hessian tensor contracted with a vector
// During articulated body assembly, the contracted vector is the full gradient of each rigid link
// This full gradient is needed to compute the reduced gradient of the reduced DOFs
// But in this procedure, of Hessian contraction, the contracted vector will be modified
// Therefore, we propose to compute the contracted hessian and the reduced gradient in one go
// If the outReducedGradient is provided (size>0), then we set:
//      outReducedGradient = jacobian^T * inOutContractedVector.
// We note that the our method of computing outReducedGradient is O(|q|), which is faster than
// vanilla matrix-vector multiplication, which is O(|q|^2).
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
    RowMatrixView<real> outHessianContracted,
    ColumnVectorView<real> outReducedGradient);

} // namespace mochi::articulated
