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

#include <mochi_core/geometry/bvh_tree.h>
#include <mochi_core/geometry/tetrahedral_map.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <algorithm>
#include <memory>
#include <tuple>

using namespace mochi;

TetrahedralMap::TetrahedralMap(std::shared_ptr<TetrahedralMesh const> const& tetMesh)
    : _tetMesh(tetMesh) {
  _numDoFs = 12;

  Span<Int4 const> tetConnectivity = tetMesh->GetElementConnectivity();
  Span<Real3 const> meshCoords = tetMesh->GetNodeCoordinates();
  _coords0.assign(meshCoords.begin(), meshCoords.end());
  _coords.assign(meshCoords.begin(), meshCoords.end());

  _aabbTreeObject = std::make_unique<TetrahedralMeshAabbObject>(tetMesh, _coords);
  _aabbTree = std::make_unique<AabbTree>(_aabbTreeObject.get(), BvhTreeParams{});

  _M0T.resize_noinit(tetConnectivity.size());
  _MinvT.resize_noinit(tetConnectivity.size());
  _dp0dp.resize_noinit(tetConnectivity.size());
  _D0.resize_noinit(tetConnectivity.size());
  _D.resize_noinit(tetConnectivity.size());
  for (int i = 0; i < tetConnectivity.size(); i++) {
    Int4 tetInds = tetConnectivity[i];
    Vec4r A0 = ToSimd(_coords[tetInds[0]]);
    Vec4r B0 = ToSimd(_coords[tetInds[1]]);
    Vec4r C0 = ToSimd(_coords[tetInds[2]]);
    _D0[i] = ToSimd(_coords[tetInds[3]]);
    _M0T[i] = VMatrix3x3r(A0 - _D0[i], B0 - _D0[i], C0 - _D0[i]);
    _D[i] = _D0[i];
    _MinvT[i] = Invert3x3(_M0T[i]);
    _dp0dp[i] = Transpose3x3(Dot3x3(_MinvT[i], _M0T[i]));
  }
}

void TetrahedralMap::UpdateMap(Span<real const> dofs) {
  auto disp = Unflatten<Real3 const>(dofs);
  MOCHI_ASSERT_VERBOSE(disp.size() == _coords.size());

  for (int i = 0; i < isize(_coords); i++) {
    _coords[i] = _coords0[i] + disp[i];
  }

  _aabbTree->Refit();

  Span<Int4 const> tetConnectivity = _tetMesh->GetElementConnectivity();
  for (int i = 0; i < isize(tetConnectivity); i++) {
    Int4 tetInds = tetConnectivity[i];
    Vec4r A = ToSimd(_coords[tetInds[0]]);
    Vec4r B = ToSimd(_coords[tetInds[1]]);
    Vec4r C = ToSimd(_coords[tetInds[2]]);
    _D[i] = ToSimd(_coords[tetInds[3]]);
    _MinvT[i] = Invert3x3(VMatrix3x3r(A - _D[i], B - _D[i], C - _D[i]));
    _dp0dp[i] = Transpose3x3(Dot3x3(_MinvT[i], _M0T[i]));
  }
}

void TetrahedralMap::MapPoints(
    Span<Real3 const> originalPoints,
    Span<int const> originalInds,
    BvhTree<Aabb> const* pointBvh,
    DynamicArray<Real3>& outMappedPoints,
    DynamicArray<int>& outInds,
    DynamicArray<VMatrix3x3r>* outMapJac,
    DynamicArray<ColliderJacDofs>* outDofsJac) const {
  static_assert(
      ColliderJacDofs::kMaxDoFs >= 12, "ColliderJacDofs::kMaxDoFs is not sufficiently large");

  // Points and indices to be mapped (in case we do local early culling)
  DynamicArray<Real3> pointsCulled;
  DynamicArray<int> indsCulled;
  Span<Real3 const> points;
  Span<int const> inds;

  // If no acceleration data structure is provided, build one
  std::unique_ptr<PointSetAabbObject> aabbTreeObjectPoints;
  std::unique_ptr<AabbTree> aabbTreePoints;
  AabbTree const* bvh = pointBvh;
  if (pointBvh == nullptr) {
    // First run early culling of points using the root of the AABB tree
    pointsCulled.reserve(originalPoints.size());
    indsCulled.reserve(originalPoints.size());
    auto const& rootBv = _aabbTree->GetRootBv();
    for (int i = 0; i < isize(originalPoints); i++) {
      if (ContainsPoint(rootBv, originalPoints[i])) {
        pointsCulled.push_back(originalPoints[i]);
        indsCulled.push_back(originalInds[i]);
      }
    }

    // Now build the BVH if there are points left
    if (!pointsCulled.empty()) {
      aabbTreeObjectPoints = std::make_unique<PointSetAabbObject>(pointsCulled);
      BvhTreeParams bvhParams;
      bvhParams.splittingAlgorithm = BvhSplittingAlgorithm::TopDown_Mean;
      aabbTreePoints = std::make_unique<AabbTree>(aabbTreeObjectPoints.get(), bvhParams);
      bvh = aabbTreePoints.get();
    }

    points = pointsCulled;
    inds = indsCulled;
  } else {
    // Use external acceleration data structure, so do not run early culling.
    points = originalPoints;
    inds = originalInds;
  }

  DynamicArray<std::tuple<int, int, int, Real3, Vec4r>> result;
  result.reserve(points.size());
  // Find points inside tets and map
  auto IntersectionCallback = [this, &points, &inds, &result](int tetIndex, int pointIndex) {
    // Compute barycentric coordinates
    Vec4r b = DotVecMat3x3(ToSimd(points[pointIndex]) - _D[tetIndex], _MinvT[tetIndex]);
    b = Blend<0, 0, 0, 1>(b, Vec4r{1_r} - VDot<3>(b, Vec4r{1_r}));

    // Test inclusion using barycentric coordinates
    if (AllTrue(b >= Vec4r{0_r})) {
      // Map position
      Real3 pos = ToReal3(_D0[tetIndex] + DotVecMat3x3(b, _M0T[tetIndex]));

      // Record mapped point
      result.emplace_back(pointIndex, inds[pointIndex], tetIndex, pos, b);
    }
  };
  if (bvh != nullptr) {
    _aabbTree->Intersect(*bvh, IntersectionCallback);
  }

  // Boundary points can be reported by multiple adjacent tetrahedra. Sort by point index so
  // std::unique keeps one mapping per input point.
  std::ranges::sort(
      result, [](auto const& a, auto const& b) { return std::get<0>(a) < std::get<0>(b); });
  result.erase(
      std::unique(
          result.begin(),
          result.end(),
          [](auto const& a, auto const& b) { return std::get<0>(a) == std::get<0>(b); }),
      result.end());

  // Reserve output vectors
  outInds.resize_noinit(result.size());
  outMappedPoints.resize_noinit(result.size());
  if (outMapJac) {
    outMapJac->resize_noinit(result.size());
  }
  if (outDofsJac) {
    outDofsJac->resize_noinit(result.size());
  }

  // Map points
  for (int i = 0; i < isize(result); i++) {
    auto [query, point, tet, pos, b] = result[i];
    outInds[i] = point;
    outMappedPoints[i] = pos;

    // Store mapping information if requested
    if (outMapJac) {
      (*outMapJac)[i] = _dp0dp[tet];
    }
    if (outDofsJac) {
      auto const& nodeInds = _tetMesh->GetElementConnectivity()[tet];
      for (int j = 0; j < 4; j++) {
        VMatrix3x3r dpddofj = Get(b, j) * VEye<3>();
        std::copy(dpddofj.begin(), dpddofj.end(), &(*outDofsJac)[i].jac[3 * j]);
        (*outDofsJac)[i].inds[3 * j] = 3 * nodeInds[j];
        (*outDofsJac)[i].inds[3 * j + 1] = 3 * nodeInds[j] + 1;
        (*outDofsJac)[i].inds[3 * j + 2] = 3 * nodeInds[j] + 2;
      }
    }
  }
}
