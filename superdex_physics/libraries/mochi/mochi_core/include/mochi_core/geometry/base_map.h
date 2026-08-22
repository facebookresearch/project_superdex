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

#include <mochi_core/contact/contact_types.h>
#include <mochi_core/geometry/aabb.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/vmatrix.h>

namespace mochi {

// Forward declarations
template <typename Bv>
class BvhTree;

/*
Base class for maps from deformed to reference space
*/
class BaseMap {
 protected:
  int _numDoFs = 0; // This needs to be set by each specific implementation of a map

 public:
  BaseMap() = default;

  virtual ~BaseMap() = default;

  [[nodiscard]] int GetNumDofs() const {
    return _numDoFs;
  }

  // This function receives a vector of dofs, and updates the mapping.
  virtual void UpdateMap(Span<real const> dofs) = 0;

  // This function receives a vector of query points in deformed space (originalPoints) and their
  // indices (originalInds) and maps them to reference space. Some points may lie outside the
  // definition of the map, and they may be discarded. The function returns the mapped positions
  // (outMappedPoints), the indices of the mapped points (outInds), and (optional) the mapping
  // Jacobians outMapJac and outDofsJac. The method may receive an acceleration data structure for
  // the points (pointBvh).
  virtual void MapPoints(
      Span<Real3 const> originalPoints,
      Span<int const> originalInds,
      BvhTree<Aabb> const* pointBvh,
      DynamicArray<Real3>& outMappedPoints,
      DynamicArray<int>& outInds,
      DynamicArray<VMatrix3x3r>* outMapJac,
      DynamicArray<ColliderJacDofs>* outDofsJac) const = 0;

  // This function takes 'outIndices' resulting from a collision detection query, and remaps them
  // using 'indsPoints' given by a previous call to 'MapPoints'. Optionally, it also remaps
  // accordingly the 'outMapJac' and 'outDofsJac' Jacobians output by 'MapPoints'. The function
  // assumes 'outIndices' are in growing order, hence reindexing is done in place.
  static void ReindexResult(
      Span<int const> indsPoints,
      Span<int> outIndices,
      DynamicArray<VMatrix3x3r>* outMapJac,
      DynamicArray<ColliderJacDofs>* outDofsJac);
};

} // namespace mochi
