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

#include <mochi_core/element_operations/fem_rod.h>
#include <mochi_core/linear_algebra/matrix.h>

namespace mochi {

// A rod pose consisting of displacement-twist DOFs and element frame axes.
struct RodPose {
  ColumnVector<real> displacements; // 4 * numNodes (3 position + 1 twist per node)
  DynamicArray<Real3> frameAxes; // numElements

  MOCHI_STRUCT_BEGIN(mochi::RodPose);
  MOCHI_FIELD(displacements);
  MOCHI_FIELD(frameAxes);
  MOCHI_STRUCT_END();
};

// Container wrapping a RodPose value for use with IntegrationBundle.
struct RodPoseContainer {
  RodPoseContainer() = default;
  explicit RodPoseContainer(RodPose const& v) : value(v) {}
  explicit RodPoseContainer(RodPose&& v) : value(std::move(v)) {}

  RodPose value;

  MOCHI_STRUCT_BEGIN(mochi::RodPoseContainer);
  MOCHI_FIELD(value);
  MOCHI_STRUCT_END();
};

// Component for time integration of rod poses (displacement-twist + frame axes).
struct CIntegrationRodPoses : public IntegrationBundle<RodPoseContainer>, NoCopy {
  CIntegrationRodPoses() = default;
  explicit CIntegrationRodPoses(int numNodes, bool isClosedLoop) {
    int const numElements = isClosedLoop ? numNodes : numNodes - 1;
    stepStart.value.displacements = ColumnVector<real>::Zero(numNodes * fem::kNumRodFields);
    stepStart.value.frameAxes.resize(numElements, Real3{});
  }

  MOCHI_STRUCT_BEGIN(mochi::CIntegrationRodPoses);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_BASE_CLASS(IntegrationBundle<RodPoseContainer>);
  MOCHI_STRUCT_END();
};

// Component holding the rod pose at a given time level, following the CRigidState<kStep> pattern.
// CRodPose<Current> is the single source of truth for displacements + frame axes during the solve.
// CRodPose<StageStart> holds the stage-start pose computed from integration.
// CRodPose<Previous> holds the previous time step's pose for integration history.
template <TimeStep kStep>
struct CRodPose : public RodPoseContainer, NoCopy {
  using RodPoseContainer::RodPoseContainer;

  CRodPose() = default;
  explicit CRodPose(int numNodes, bool isClosedLoop)
      : RodPoseContainer(
            RodPose{
                ColumnVector<real>::Zero(numNodes * fem::kNumRodFields),
                DynamicArray<Real3>(isClosedLoop ? numNodes : numNodes - 1, Real3{})}) {}

  MOCHI_TEMPLATE_BEGIN(mochi::CRodPose, kStep);
  MOCHI_ATTRIBUTE_IF(kStep == TimeStep::Current, CaptureState);
  MOCHI_BASE_CLASS(RodPoseContainer);
  MOCHI_TEMPLATE_END();
};

namespace rod {

// Compute the unit tangent vector for a rod element.
[[nodiscard]] Real3 ComputeRodElementTangent(
    Span<Real3 const> meshNodes,
    ColumnVectorView<real const> displacement,
    int elementIndex);

// Apply a Lie-algebra increment to a rod pose: adds dofDelta to refDisplacement for all
// 4 DOFs per node (including twist), and updates frame axes via parallel transport followed
// by twist rotation.
void ApplyLieDeltaToPose(
    Span<Real3 const> meshNodes,
    ColumnVectorView<real const> refDisplacement,
    Span<Real3 const> refAxes,
    ColumnVectorView<real const> dofDelta,
    ColumnVectorView<real> outDisplacement,
    Span<Real3> outAxes);

} // namespace rod
} // namespace mochi
