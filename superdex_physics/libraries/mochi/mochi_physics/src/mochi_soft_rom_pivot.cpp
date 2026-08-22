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

#include "mochi_soft_rom_pivot.h"

#include <mochi_core/rom/rom_pivot.h>

using namespace mochi;

void rom::rigid_transform::details::ComputePositions(
    TetrahedralMesh const& mesh,
    Span<int const> activeNodes,
    ColumnVectorView<real const> displacements,
    ColumnVectorView<real> outAuxPositions) {
  auto meshCoords = AsConstView(Flatten(mesh.GetNodeCoordinates()));
  if (activeNodes.empty()) {
    outAuxPositions = meshCoords + displacements;
  } else {
    for (int nodeId : activeNodes) {
      outAuxPositions[nodeId * 3 + 0] = meshCoords[nodeId * 3 + 0] + displacements[nodeId * 3 + 0];
      outAuxPositions[nodeId * 3 + 1] = meshCoords[nodeId * 3 + 1] + displacements[nodeId * 3 + 1];
      outAuxPositions[nodeId * 3 + 2] = meshCoords[nodeId * 3 + 2] + displacements[nodeId * 3 + 2];
    }
  }
}

void rom::rigid_transform::details::SubtractMeshCoordinates(
    TetrahedralMesh const& mesh,
    Span<int const> activeNodes,
    ColumnVectorView<real> displacements) {
  auto meshCoords = AsConstView(Flatten(mesh.GetNodeCoordinates()));
  if (activeNodes.empty()) {
    displacements -= meshCoords;
  } else {
    for (int nodeId : activeNodes) {
      displacements[nodeId * 3 + 0] -= meshCoords[nodeId * 3 + 0];
      displacements[nodeId * 3 + 1] -= meshCoords[nodeId * 3 + 1];
      displacements[nodeId * 3 + 2] -= meshCoords[nodeId * 3 + 2];
    }
  }
}
