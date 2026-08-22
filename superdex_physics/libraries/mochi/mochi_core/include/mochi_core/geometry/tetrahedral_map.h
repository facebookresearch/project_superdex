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

#include <mochi_core/geometry/aabb.h>
#include <mochi_core/geometry/base_map.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

#include <memory>

namespace mochi {

// Forward declarations
template <typename Bv>
class BvhTree;
template <typename Bv>
class TetrahedralMeshBvhObject;

/*
Class that allows mapping data to and from deformed/reference space based on a tetrahedral mesh.

It follows this math:
- Point in reference space: p0
- Point in deformed space: p
- Tetrahedron in reference space: [A0, B0, C0, D0]
- Tetrahedron in deformed space: [A, B, C, D]
- Barycentric coordinates: bABCD = [bA, bB, bC, bD]
- From barycentric coords to ref space:
    p0 = [A0, B0, C0, D0] * bABCD = D0 + [A0-D0, B0-D0, C0-D0] * bABC = D0 + M0 * bABC
    with M0 = [A0-D0, B0-D0, C0-D0]
- From barycentric coords to deformed space:
    p = D + M * bABC, with M = [A-D, B-D, C-D]
- From deformed space to barycentric coords:
    bABC = inv(M) * (p - D)
- Full mapping from deformed space to reference space
    p0 = D0 + M0 * inv(M) * (p - D)
- Jacobian of the mapping, to transform contact normals
     dp0/dp = M0 * inv(M)

For convenience, the class caches D0, D, M0, inv(M) and dp0/dp, with matrices transposed
*/
class TetrahedralMap : public BaseMap {
 public:
  TetrahedralMap() = default;

  explicit TetrahedralMap(std::shared_ptr<TetrahedralMesh const> const& tetMesh);

  ~TetrahedralMap() override = default;

  // Functions of BaseMap
  void UpdateMap(Span<real const> dofs) override;
  void MapPoints(
      Span<Real3 const> originalPoints,
      Span<int const> originalInds,
      BvhTree<Aabb> const* pointBvh,
      DynamicArray<Real3>& outMappedPoints,
      DynamicArray<int>& outInds,
      DynamicArray<VMatrix3x3r>* outMapJac,
      DynamicArray<ColliderJacDofs>* outDofsJac) const override;

 protected:
  std::shared_ptr<TetrahedralMesh const> _tetMesh;
  std::unique_ptr<TetrahedralMeshBvhObject<Aabb>> _aabbTreeObject;
  std::unique_ptr<BvhTree<Aabb>> _aabbTree;
  DynamicArray<Real3> _coords0 = {};
  DynamicArray<Real3> _coords = {};
  DynamicArray<VMatrix3x3r> _M0T;
  DynamicArray<VMatrix3x3r> _MinvT;
  DynamicArray<VMatrix3x3r> _dp0dp;
  DynamicArray<Vec4r> _D0;
  DynamicArray<Vec4r> _D;
};

} // namespace mochi
