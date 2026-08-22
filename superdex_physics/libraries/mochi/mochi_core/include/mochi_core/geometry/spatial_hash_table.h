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
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <cstdint>

namespace mochi {

class SpatialHashTable {
 public:
  // This is the only dynamic memory allocation for SpatialHashTable. minNumBins is rounded up to
  // the next power of two. As long as the maximum point capacity and rounded number of bins don't
  // need to change, the structure can be reset and repopulated with no new dynamic allocations.
  SpatialHashTable(real cellSize, int pointCapacity, int minNumBins);

  // This maintains the original memory allocation for the pointCapacity given at construction, but
  // resets the state to contain no points. The cost is bounded by the number of points, and is
  // independent of the number of bins (i.e., this operation remains efficient at arbitrarily-small
  // load factors).
  void Reset();

  // This adds a point, with the assumption that no other point with the given pointIndex is
  // contained. The pointIndex must be less than the point capacity. This only checks whether
  // capacity is exceeded in debug builds. This operation is always O(1), even if many points hash
  // to the same bin.
  void AddPoint(int pointIndex, Real3 const& point);

  // This iterates over a collection of points that contains AT LEAST all points within a distance
  // _cellSize (in the infinity norm) of the given position.  It is possible that some points it
  // iterates over are further away (and possibly much further in the case of hash collisions), but
  // it is left to the callback to decide what to do with those.
  template <typename CallbackType>
  void IteratePointsNearPosition(Real3 const& position, CallbackType&& callback) const {
    // Iterate the 3x3x3 grid-cell stencil centered at the cell containing the query position.
    Int3 const centerCellCoordinates = GetCellCoordinates(position);
    int binIndicesEncountered[3 * 3 * 3];
    int numBinsEncountered = 0;
    // Lossy prefilter for the exact duplicate-bin scan. A set bit means the bin may have been seen;
    // the exact scan below resolves false positives. Empty bins are not tracked.
    uint64_t binIndexLowBitMask = 0;
    unsigned int const xCenterHash = HashCellCoordinate(centerCellCoordinates[0], kHashX);
    unsigned int const yCenterHash = HashCellCoordinate(centerCellCoordinates[1], kHashY);
    unsigned int const zCenterHash = HashCellCoordinate(centerCellCoordinates[2], kHashZ);
    unsigned int const xHashes[3] = {xCenterHash - kHashX, xCenterHash, xCenterHash + kHashX};
    unsigned int const yHashes[3] = {yCenterHash - kHashY, yCenterHash, yCenterHash + kHashY};
    unsigned int const zHashes[3] = {zCenterHash - kHashZ, zCenterHash, zCenterHash + kHashZ};
    for (unsigned int const xHash : xHashes) {
      for (unsigned int const yHash : yHashes) {
        for (unsigned int const zHash : zHashes) {
          int const binIndex = GetBinIndex(xHash, yHash, zHash);
          IterateBinPointsIfNotEncountered(
              binIndex, binIndicesEncountered, numBinsEncountered, binIndexLowBitMask, callback);
        }
      }
    }
  }

  // Current number of points contained in the table.
  int GetNumPoints() const {
    return _numPoints;
  }

  // This is the number of points that can be added.
  int GetCapacity() const {
    return _pointCapacity;
  }

  real GetCellSize() const {
    return _cellSize;
  }

  // AABB for the points currently in the table. The table must be non-empty.
  Aabb GetOccupiedPointBounds() const {
    MOCHI_ASSERT(_numPoints > 0, "Spatial hash table is empty.");
    return {_occupiedPointMin, _occupiedPointMax};
  }

 private:
  void AddPointToBin(int pointIndex, int binIndex);

  MOCHI_FORCE_INLINE Int3 GetCellCoordinates(Real3 const& point) const {
    return StaticCast<Int3>(Floor(point / _cellSize));
  }

  MOCHI_FORCE_INLINE static unsigned int HashCellCoordinate(
      int cellCoordinate,
      unsigned int multiplier) {
    return static_cast<unsigned int>(cellCoordinate) * multiplier;
  }

  MOCHI_FORCE_INLINE int GetBinIndex(unsigned int xHash, unsigned int yHash, unsigned int zHash)
      const {
    return static_cast<int>((xHash ^ yHash ^ zHash) & _binIndexMask);
  }

  MOCHI_FORCE_INLINE int GetBinIndex(Int3 const& cellCoordinates) const {
    return GetBinIndex(
        HashCellCoordinate(cellCoordinates[0], kHashX),
        HashCellCoordinate(cellCoordinates[1], kHashY),
        HashCellCoordinate(cellCoordinates[2], kHashZ));
  }

  MOCHI_FORCE_INLINE void ExpandOccupiedPointBounds(Real3 const& point) {
    _occupiedPointMin = Min(_occupiedPointMin, point);
    _occupiedPointMax = Max(_occupiedPointMax, point);
  }

  template <typename CallbackType>
  MOCHI_FORCE_INLINE void IterateBinPointsIfNotEncountered(
      int binIndex,
      int* binIndicesEncountered,
      int& numBinsEncountered,
      uint64_t& binIndexLowBitMask,
      CallbackType& callback) const {
    int currentPointIndex = _binHeads[binIndex];
    if (currentPointIndex == kSentinelIndex) {
      return;
    }
    uint64_t const binIndexLowBit = uint64_t{1} << (static_cast<unsigned int>(binIndex) & 63u);
    if ((binIndexLowBitMask & binIndexLowBit) != 0) {
      for (int l = 0; l < numBinsEncountered; l++) {
        if (binIndicesEncountered[l] == binIndex) {
          return;
        }
      }
    }
    binIndexLowBitMask |= binIndexLowBit;
    while (currentPointIndex != kSentinelIndex) {
      callback(currentPointIndex);
      currentPointIndex = _binLists[currentPointIndex];
    }
    binIndicesEncountered[numBinsEncountered] = binIndex;
    numBinsEncountered++;
  }

  static constexpr unsigned int kHashX = 73856093u;
  static constexpr unsigned int kHashY = 19349663u;
  static constexpr unsigned int kHashZ = 83492791u;

  real _cellSize = {};
  int _pointCapacity = 0;
  int _numPoints = 0;
  int _numBins = 0;
  unsigned int _binIndexMask = 0;
  Real3 _occupiedPointMin = kInf3;
  Real3 _occupiedPointMax = -kInf3;
  DynamicArray<int> _binHeads;
  DynamicArray<int> _binLists;
  DynamicArray<int> _activeBins;
};

} // namespace mochi
