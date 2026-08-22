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

#include "mochi_discretization_components.h"
#include "mochi_ecs.h"
#include "mochi_rom_jacobian.h"
#include "mochi_soft_rom_components.h"
#include "mochi_soft_rom_pivot.h"

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/rom/rom_pivot.h>

namespace mochi::rom::neural_net_crom {

namespace details {
inline void PrepareDecoderInputMatrix(
    Span<Real3 const> meshNodesPositions,
    Span<int const> activeNodes,
    ColumnVectorView<real const> latentState,
    MatrixView<real> X) {
  MOCHI_ASSERT(X.Rows() == latentState.Rows() + 3, "Inconsistent sizes.");
  int const qDim = latentState.Rows();
  int const nodeCount = activeNodes.empty() ? isize(meshNodesPositions) : isize(activeNodes);
  MOCHI_ASSERT(X.Cols() == nodeCount, "Inconsistent sizes.");
  for (int i = 0; i < nodeCount; ++i) {
    int const nodeId = activeNodes.empty() ? i : activeNodes[i];

    // The first qDim rows are latent state. The last 3 rows are node position.
    auto colX = X.Col(i);
    colX.TopRows(qDim) = latentState;

    auto const& currRefPos = meshNodesPositions[nodeId];
    colX(qDim + 0) = currRefPos[0];
    colX(qDim + 1) = currRefPos[1];
    colX(qDim + 2) = currRefPos[2];
  }
}
} // namespace details

inline void ResolveDisplacement(
    TetrahedralMesh const& mesh,
    Span<int const> activeNodes,
    CNeuralNetCromModel const& model,
    CNeuralNetCromDecoderData& decoderData,
    ColumnVectorView<real const> latentState,
    ColumnVectorView<real> outDisplacements,
    TransformRT const* rigidTransform = nullptr,
    Real3 const* pivot = nullptr,
    ColumnVectorView<real> displacementBuffer = {}) {
  MOCHI_ASSERT_VERBOSE(IsUnique(activeNodes));
  auto meshNodesPositions = mesh.GetNodeCoordinates();
  MOCHI_ASSERT(latentState.Rows() + 3 == model.decoder.InputDim(), "Inconsistent sizes.");
  MOCHI_ASSERT(latentState.Rows() + 3 == decoderData.inputMatrix.Rows(), "Inconsistent sizes.");
  MOCHI_ASSERT(outDisplacements.Rows() == 3 * isize(meshNodesPositions), "Inconsistent sizes.");
  MOCHI_ASSERT(
      (rigidTransform != nullptr) == (pivot != nullptr),
      "Either none or both of rigid transform and pivot must be provided.");

  int const nodeCount = activeNodes.empty() ? isize(meshNodesPositions) : isize(activeNodes);
  auto& mlpInput = decoderData.inputMatrix;
  auto& mlpOutput = decoderData.networkOutput;
  mlpInput.Resize(mlpInput.Rows(), nodeCount);
  mlpOutput.Resize(mlpOutput.Rows(), nodeCount);

  details::PrepareDecoderInputMatrix(
      meshNodesPositions, activeNodes, latentState, AsView(mlpInput));

  model.decoder.Forward(mlpInput, mlpOutput);

  auto mean = model.meanAndStdevForOutputInverseStandardize.template MiddleRows<3>(0, 3);
  auto stdev = model.meanAndStdevForOutputInverseStandardize.template MiddleRows<3>(3, 3);

  if (!activeNodes.empty()) {
    outDisplacements.SetZero();
  }
  for (int i = 0; i < nodeCount; ++i) {
    int const nodeId = activeNodes.empty() ? i : activeNodes[i];

    outDisplacements(nodeId * 3 + 0) = mlpOutput(0, i) * stdev[0] + mean[0]; // x
    outDisplacements(nodeId * 3 + 1) = mlpOutput(1, i) * stdev[1] + mean[1]; // y
    outDisplacements(nodeId * 3 + 2) = mlpOutput(2, i) * stdev[2] + mean[2]; // z
  }

  // Transform to local frame if rigidTransform is provided
  if (rigidTransform != nullptr && pivot != nullptr) {
    MOCHI_PROFILE_SCOPE_N("TransformDisplacements");
    MOCHI_ASSERT(
        !displacementBuffer.empty(), "displacementBuffer required for local frame transformation");
    MOCHI_ASSERT(displacementBuffer.Rows() == outDisplacements.Rows(), "Inconsistent sizes.");

    rigid_transform::TransformDisplacements(
        mesh, activeNodes, *pivot, *rigidTransform, displacementBuffer, outDisplacements);
  }
}

inline void ResolveDisplacementAndJacobian(
    TetrahedralMesh const& mesh,
    Span<int const> activeNodes,
    CNeuralNetCromModel const& model,
    CNeuralNetCromDecoderData& decoderData,
    ColumnVectorView<real const> latentState,
    ColumnVectorView<real> outDisplacements,
    RowMatrixView<real> outJacobian,
    TransformRT const* rigidTransform = nullptr,
    Real3 const* pivot = nullptr,
    ColumnVectorView<real> displacementBuffer = {},
    RowMatrixView<real> basisJacobian = {}) {
  MOCHI_ASSERT_VERBOSE(IsUnique(activeNodes));
  auto meshNodesPositions = mesh.GetNodeCoordinates();
  MOCHI_ASSERT(latentState.Rows() + 3 == model.decoder.InputDim(), "Inconsistent sizes.");
  MOCHI_ASSERT(latentState.Rows() + 3 == decoderData.inputMatrix.Rows(), "Inconsistent sizes.");
  MOCHI_ASSERT(outDisplacements.Rows() == 3 * isize(meshNodesPositions), "Inconsistent sizes.");
  MOCHI_ASSERT(outJacobian.Rows() == 3 * isize(meshNodesPositions), "Inconsistent sizes.");
  MOCHI_ASSERT(outJacobian.Cols() == latentState.Rows(), "Inconsistent sizes.");
  MOCHI_ASSERT(
      (rigidTransform != nullptr) == (pivot != nullptr),
      "Either none or both of rigid transform and pivot must be provided.");
  bool const requiresTransformation = (rigidTransform != nullptr);

  int const nodeCount = activeNodes.empty() ? isize(meshNodesPositions) : isize(activeNodes);
  auto& mlpInput = decoderData.inputMatrix;
  auto& mlpOutput = decoderData.networkOutput;
  auto& mlpJacobian = decoderData.networkOutputJacobian;
  mlpInput.Resize(mlpInput.Rows(), nodeCount);
  mlpOutput.Resize(mlpOutput.Rows(), nodeCount);
  mlpJacobian.Resize(nodeCount * 3, mlpInput.Rows());

  details::PrepareDecoderInputMatrix(
      meshNodesPositions, activeNodes, latentState, AsView(mlpInput));

  model.decoder.ForwardAndJacobian(mlpInput, mlpOutput, mlpJacobian);

  auto mean = model.meanAndStdevForOutputInverseStandardize.template MiddleRows<3>(0, 3);
  auto stdev = model.meanAndStdevForOutputInverseStandardize.template MiddleRows<3>(3, 3);

  // MLP jacobian block that pertains to the latent variables.
  auto mlpJacobianQ = mlpJacobian.LeftCols(latentState.Rows());

  RowMatrixView<real> destJacobian = requiresTransformation ? basisJacobian : outJacobian;
  if (!activeNodes.empty()) {
    outDisplacements.SetZero();
    destJacobian.SetZero();
  }
  for (int i = 0; i < nodeCount; ++i) {
    int const nodeId = activeNodes.empty() ? i : activeNodes[i];

    outDisplacements(nodeId * 3 + 0) = mlpOutput(0, i) * stdev[0] + mean[0]; // x
    outDisplacements(nodeId * 3 + 1) = mlpOutput(1, i) * stdev[1] + mean[1]; // y
    outDisplacements(nodeId * 3 + 2) = mlpOutput(2, i) * stdev[2] + mean[2]; // z

    for (int j = 0; j < mlpJacobianQ.Cols(); ++j) {
      destJacobian(nodeId * 3 + 0, j) = mlpJacobianQ(i * 3 + 0, j) * stdev[0]; // x
      destJacobian(nodeId * 3 + 1, j) = mlpJacobianQ(i * 3 + 1, j) * stdev[1]; // y
      destJacobian(nodeId * 3 + 2, j) = mlpJacobianQ(i * 3 + 2, j) * stdev[2]; // z
    }
  }

  // Transform to local frame if rigidTransform is provided
  if (requiresTransformation) {
    MOCHI_PROFILE_SCOPE_N("TransformDisplacementsAndJacobian");
    MOCHI_ASSERT(
        !displacementBuffer.empty(), "displacementBuffer required for local frame transformation");
    MOCHI_ASSERT(!basisJacobian.empty(), "basisJacobian required for local frame transformation");
    MOCHI_ASSERT(displacementBuffer.Rows() == outDisplacements.Rows(), "Inconsistent sizes.");
    MOCHI_ASSERT(
        outJacobian.Rows() == basisJacobian.Rows() && outJacobian.Cols() == basisJacobian.Cols(),
        "Inconsistent sizes.");

    if (!activeNodes.empty()) {
      outJacobian.SetZero();
    }
    rigid_transform::TransformDisplacementsAndJacobian(
        mesh,
        activeNodes,
        *pivot,
        *rigidTransform,
        displacementBuffer,
        outDisplacements,
        AsConstView(basisJacobian),
        outJacobian,
        /*computeJacWrtRigidTransform*/ false);
  }
}

} // namespace mochi::rom::neural_net_crom
