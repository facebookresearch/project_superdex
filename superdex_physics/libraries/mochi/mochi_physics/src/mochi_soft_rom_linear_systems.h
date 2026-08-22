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

#include "mochi_common_components.h"
#include "mochi_contact.h"
#include "mochi_ecs.h"
#include "mochi_rom_jacobian.h"
#include "mochi_soft_rom_affine_helpers.h"
#include "mochi_soft_rom_components.h"
#include "mochi_soft_rom_pivot.h"

#include <mochi_core/rom/rom_pivot.h>
#include <mochi_core/utils/array_utils.h>

namespace mochi::rom::linear {

/************************************
 * impl functions
 */

namespace impl {

template <typename BasisT>
void DisplacementFromBasis(
    Span<int const> activeNodes,
    BasisT const& basis,
    ColumnVectorView<real const> amplitudes,
    ColumnVectorView<real> outDisplacements) {
  if (activeNodes.empty()) {
    outDisplacements = basis * amplitudes;
  } else {
    constexpr int kDofsPerNode = 3;
    for (int nodeId : activeNodes) {
      outDisplacements.template MiddleRows<kDofsPerNode>(nodeId * kDofsPerNode, kDofsPerNode) =
          basis.template MiddleRows<kDofsPerNode>(nodeId * kDofsPerNode, kDofsPerNode) * amplitudes;
    }
  }
}

} // namespace impl

/************************************
 * public functions
 */

template <bool kForceUseAllNodes>
void ResolveDisplacement(
    ecs::RequiredTag<TagRomActor>,
    CRomLinearBasis const& basis,
    CRomModeAmplitudes const& currAmplitudes,
    CRomShiftVector const& shiftVector,
    CDisplacementSlice<real, TimeStep::Current>& displacements,
    CTetrahedralMesh const& mesh,
    CRigidState<TimeStep::Current> const& rigidTransform,
    CMeshPivot const& pivot,
    CAuxiliaryPositionsForRomRigidTransform& rigidTransformAuxPositions,
    CActiveUniqueNodes const* activeNodes = nullptr) {
  MOCHI_ASSERT_VERBOSE(!activeNodes || IsUnique(activeNodes->ViewIds()));
  MOCHI_ASSERT_VERBOSE(mesh.mesh, "Missing mesh.");

  bool const useAllMeshNodes = !activeNodes || kForceUseAllNodes;
  Span<int const> nodeSubset = useAllMeshNodes ? Span<int const>{} : activeNodes->ViewIds();

  displacements.value.SetZero();
  impl::DisplacementFromBasis(
      nodeSubset,
      basis.ViewMatrix(),
      AsConstView(currAmplitudes.value),
      AsView(displacements.value));
  AddAffineShiftVector(nodeSubset, AsConstView(shiftVector.value), AsView(displacements.value));

  rigid_transform::TransformDisplacements(
      *mesh.mesh,
      nodeSubset,
      pivot.position,
      rigidTransform.value,
      rigidTransformAuxPositions.data,
      displacements.value);
}

template <bool kForceUseAllNodes>
void ResolveDisplacementAndJacobian(
    ecs::RequiredTag<TagRomActor>,
    ecs::OptionalTag<TagRomActorFixRigidTransformInSolve> isRigidTransformFixedInSolve,
    CRomLinearBasis const& basis,
    CRomModeAmplitudes const& currAmplitudes,
    CRomShiftVector const& shiftVector,
    CDisplacementSlice<real, TimeStep::Current>& displacements,
    CRomJacobian& jacobian,
    CTetrahedralMesh const& mesh,
    CRigidState<TimeStep::Current> const& rigidTransform,
    CMeshPivot const& pivot,
    CAuxiliaryPositionsForRomRigidTransform& rigidTransformAuxPositions,
    CActiveUniqueNodes const* activeNodes = nullptr) {
  MOCHI_ASSERT_VERBOSE(!activeNodes || IsUnique(activeNodes->ViewIds()));
  MOCHI_ASSERT_VERBOSE(mesh.mesh, "Missing mesh.");

  bool const useAllMeshNodes = !activeNodes || kForceUseAllNodes;
  Span<int const> nodeSubset = useAllMeshNodes ? Span<int const>{} : activeNodes->ViewIds();

  displacements.value.SetZero();
  jacobian.Get<CRomJacobian::DenseT>().SetZero();

  impl::DisplacementFromBasis(
      nodeSubset, basis.ViewMatrix(), AsConstView(currAmplitudes.value), displacements.value);
  AddAffineShiftVector(nodeSubset, AsConstView(shiftVector.value), AsView(displacements.value));

  // for linear roms, the jacobian of the rom mapping and what needs to
  // be fed into the rigid transform is just the basis, and it never changes.
  auto const& preTransfJacobian = basis.ViewMatrix();
  // if the rigid transform is fixed during the solve, we do not compute the jacobian
  // with respect to the rigid transform parameters but only with respect to the amplitudes.
  bool const computeJacWrtRigidTransform = !isRigidTransformFixedInSolve;
  rigid_transform::TransformDisplacementsAndJacobian(
      *mesh.mesh,
      nodeSubset,
      pivot.position,
      rigidTransform.value,
      rigidTransformAuxPositions.data,
      displacements.value,
      preTransfJacobian,
      jacobian.Get<CRomJacobian::DenseT>(),
      computeJacWrtRigidTransform);
}

} // namespace mochi::rom::linear
