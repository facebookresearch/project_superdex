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

#include "mochi_ecs.h"
#include "mochi_rom_jacobian.h"
#include "mochi_soft_rom_components.h"

#include <mochi_core/rom/rom_pivot.h>

#include <type_traits>

namespace mochi::rom::rigid_transform {

namespace details {

// outAuxPositions = meshCoords + displacements
MOCHI_API void ComputePositions(
    TetrahedralMesh const& mesh,
    Span<int const> activeNodes,
    ColumnVectorView<real const> displacements,
    ColumnVectorView<real> outAuxPositions);

// displacements -= meshCoords
MOCHI_API void SubtractMeshCoordinates(
    TetrahedralMesh const& mesh,
    Span<int const> activeNodes,
    ColumnVectorView<real> displacements);

template <typename InputJacobianType = void, typename OutputJacobianType = void>
void TransformDisplacementsAndJacobianImpl(
    TetrahedralMesh const& mesh,
    Span<int const> activeNodes,
    Real3 const& pivot,
    TransformRT const& rigidTransform,
    ColumnVectorView<real> auxiliaryPositions,
    ColumnVectorView<real> displ,
    InputJacobianType const* inJacobian = nullptr,
    OutputJacobianType* outJacobian = nullptr,
    bool computeJacWrtRigidTransform = true) {
  static_assert(std::is_void_v<InputJacobianType> == std::is_void_v<OutputJacobianType>);
  MOCHI_ASSERT_VERBOSE(auxiliaryPositions.Rows() == displ.Rows(), "Inconsistent sizes.");
  // auxiliaryPositions = meshCoords + displ
  ComputePositions(mesh, activeNodes, AsConstView(displ), auxiliaryPositions);

  TransformBatch(
      rigidTransform, AsConstView(auxiliaryPositions), AsView(displ), pivot, activeNodes);

  // displ -= mesh coordinates
  SubtractMeshCoordinates(mesh, activeNodes, displ);

  if constexpr (!std::is_void_v<InputJacobianType>) {
    MOCHI_ASSERT(inJacobian && outJacobian, "Missing Jacobians.")

    if (computeJacWrtRigidTransform) {
      pivoted::JacobianFromPositions(
          AsConstView(*inJacobian),
          rigidTransform,
          pivot,
          AsConstView(auxiliaryPositions),
          AsView(*outJacobian),
          activeNodes);
    } else {
      // If we need to neglect rigid transform params, we only need to transform the jacobian passed
      // which is the jacobian of the base ROM.
      DTransformBatch(
          rigidTransform,
          AsConstView(*inJacobian),
          AsView(*outJacobian),
          /*preTransform*/ {},
          /*postTransform*/ {},
          activeNodes);
    }

  } else {
    MOCHI_ASSERT_VERBOSE(!inJacobian && !outJacobian, "Unexpected Jacobians.");
  }
}
} // namespace details

/************************************
 * public functions
 */

inline void TransformDisplacements(
    TetrahedralMesh const& mesh,
    Span<int const> activeNodes,
    Real3 const& pivot,
    TransformRT const& rigidTransform,
    ColumnVectorView<real> auxiliaryPositions,
    ColumnVectorView<real> displ) {
  details::TransformDisplacementsAndJacobianImpl(
      mesh, activeNodes, pivot, rigidTransform, auxiliaryPositions, displ);
}

template <typename InputJacobianType, typename OutputJacobianType>
void TransformDisplacementsAndJacobian(
    TetrahedralMesh const& mesh,
    Span<int const> activeNodes,
    Real3 const& pivot,
    TransformRT const& rigidTransform,
    ColumnVectorView<real> auxiliaryPositions,
    ColumnVectorView<real> displacements,
    InputJacobianType const& inJacobian,
    OutputJacobianType& outJacobian,
    bool computeJacWrtRigidTransform) {
  details::TransformDisplacementsAndJacobianImpl(
      mesh,
      activeNodes,
      pivot,
      rigidTransform,
      auxiliaryPositions,
      displacements,
      &inJacobian,
      &outJacobian,
      computeJacWrtRigidTransform);
}

} // namespace mochi::rom::rigid_transform
