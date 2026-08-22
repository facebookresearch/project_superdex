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

#include <mochi_core/geometry/spatial_hash_table.h>

#include <mochi_core/utils/basic_utils.h>

namespace mochi {

[[nodiscard]] static int RoundNumBins(int minNumBins) {
  MOCHI_ASSERT_VERBOSE(minNumBins > 0, "Minimum number of bins must be positive.");
  return NextPowerOfTwo(minNumBins);
}

SpatialHashTable::SpatialHashTable(real cellSize, int pointCapacity, int minNumBins)
    : _cellSize(cellSize),
      _pointCapacity(pointCapacity),
      _numPoints(0),
      _numBins(RoundNumBins(minNumBins)),
      _binIndexMask(_numBins - 1),
      _binHeads(_numBins, kSentinelIndex),
      _binLists(_pointCapacity, kSentinelIndex) {
  _activeBins.reserve(_numBins);
}

void SpatialHashTable::Reset() {
  // Only clear active bins, which is bounded by the total number of points, to make this
  // efficient for small load factors.
  for (int const binIndex : _activeBins) {
    _binHeads[binIndex] = kSentinelIndex;
  }
  _activeBins.clear();
  _numPoints = 0;
  _occupiedPointMin = kInf3;
  _occupiedPointMax = -kInf3;
}

void SpatialHashTable::AddPoint(int pointIndex, Real3 const& point) {
  MOCHI_ASSERT_VERBOSE(_numPoints < _pointCapacity, "Exceeded capcity of spatial hash table");
  MOCHI_ASSERT_VERBOSE(pointIndex >= 0 && pointIndex < _pointCapacity, "Invalid point index");
  Int3 const cellCoordinates = GetCellCoordinates(point);
  int const binIndex = GetBinIndex(cellCoordinates);
  MOCHI_ASSERT_VERBOSE(binIndex >= 0 && binIndex < _numBins, "Invalid bin index");
  ExpandOccupiedPointBounds(point);
  AddPointToBin(pointIndex, binIndex);
  _numPoints++;
}

void SpatialHashTable::AddPointToBin(int pointIndex, int binIndex) {
  int const oldBinHead = _binHeads[binIndex];
  _binHeads[binIndex] = pointIndex;
  _binLists[pointIndex] = oldBinHead;
  // Track active bins for efficient reset operation.
  if (oldBinHead == kSentinelIndex) {
    _activeBins.push_back(binIndex);
    MOCHI_ASSERT_VERBOSE(_activeBins.size() <= _numBins, "More active bins than total bins");
  }
}

} // namespace mochi
