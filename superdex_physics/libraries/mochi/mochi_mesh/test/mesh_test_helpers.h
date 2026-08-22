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

#include <mochi_core/geometry/mesh_data.h>

namespace mochi::mesh {

// Creates a unit cube triangle mesh (8 vertices, 12 triangles) for testing.
inline MeshData CreateCubeTriMesh() {
  MeshData mesh;
  mesh.nodesPerElement = 3;
  mesh.coordinates = {
      // clang-format off
      0_r, 0_r, 0_r,  // 0
      1_r, 0_r, 0_r,  // 1
      1_r, 1_r, 0_r,  // 2
      0_r, 1_r, 0_r,  // 3
      0_r, 0_r, 1_r,  // 4
      1_r, 0_r, 1_r,  // 5
      1_r, 1_r, 1_r,  // 6
      0_r, 1_r, 1_r,  // 7
      // clang-format on
  };
  mesh.connectivity = {
      // clang-format off
      0, 2, 1,  0, 3, 2,  // Bottom
      4, 5, 6,  4, 6, 7,  // Top
      0, 1, 5,  0, 5, 4,  // Front
      2, 7, 6,  2, 3, 7,  // Back
      0, 4, 7,  0, 7, 3,  // Left
      1, 6, 5,  1, 2, 6,  // Right
      // clang-format on
  };
  return mesh;
}

// Creates a unit cube triangle mesh with the top face removed (8 vertices, 10 triangles).
inline MeshData CreateCubeTriMeshWithHole() {
  MeshData mesh;
  mesh.nodesPerElement = 3;
  mesh.coordinates = {
      // clang-format off
      0_r, 0_r, 0_r,  // 0
      1_r, 0_r, 0_r,  // 1
      1_r, 1_r, 0_r,  // 2
      0_r, 1_r, 0_r,  // 3
      0_r, 0_r, 1_r,  // 4
      1_r, 0_r, 1_r,  // 5
      1_r, 1_r, 1_r,  // 6
      0_r, 1_r, 1_r,  // 7
      // clang-format on
  };
  mesh.connectivity = {
      // clang-format off
      0, 2, 1,  0, 3, 2,  // Bottom
      0, 1, 5,  0, 5, 4,  // Front
      2, 7, 6,  2, 3, 7,  // Back
      0, 4, 7,  0, 7, 3,  // Left
      1, 6, 5,  1, 2, 6,  // Right
      // clang-format on
  };
  return mesh;
}

// Creates a unit cube triangle mesh with one face's winding flipped, producing
// inconsistent orientations that cause the helper's mesh ingestion to fail.
inline MeshData CreateCubeTriMeshWithInconsistentWinding() {
  MeshData mesh;
  mesh.nodesPerElement = 3;
  mesh.coordinates = {
      // clang-format off
      0_r, 0_r, 0_r,  // 0
      1_r, 0_r, 0_r,  // 1
      1_r, 1_r, 0_r,  // 2
      0_r, 1_r, 0_r,  // 3
      0_r, 0_r, 1_r,  // 4
      1_r, 0_r, 1_r,  // 5
      1_r, 1_r, 1_r,  // 6
      0_r, 1_r, 1_r,  // 7
      // clang-format on
  };
  mesh.connectivity = {
      // clang-format off
      0, 1, 2,  0, 3, 2,  // Bottom (first face winding flipped: 0,2,1 -> 0,1,2)
      4, 5, 6,  4, 6, 7,  // Top
      0, 1, 5,  0, 5, 4,  // Front
      2, 7, 6,  2, 3, 7,  // Back
      0, 4, 7,  0, 7, 3,  // Left
      1, 6, 5,  1, 2, 6,  // Right
      // clang-format on
  };
  return mesh;
}

} // namespace mochi::mesh
