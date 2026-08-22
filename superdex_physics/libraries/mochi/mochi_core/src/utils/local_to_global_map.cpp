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

#include <mochi_core/utils/local_to_global_map.h>
#include <mochi_core/utils/profile.h>

#include <cstring>
#include <limits>

namespace mochi {

[[nodiscard]] Span<int const> Local2GlobalMap::GetElementSizes() const {
  return {_eleSizes.data(), _eleSizes.size()};
}

[[nodiscard]] Span<int const> Local2GlobalMap::GetElementOffsets() const {
  return {_eleOffsets.data(), _eleOffsets.size()};
}

[[nodiscard]] Span<int const> Local2GlobalMap::GetGlobalIndices() const {
  return {_indices.data(), _indices.size()};
}

[[nodiscard]] int Local2GlobalMap::GetNumElements() const {
  return isize(_eleSizes);
}

[[nodiscard]] int Local2GlobalMap::GetNumIndices() const {
  return isize(_indices);
}

[[nodiscard]] Interval<int> Local2GlobalMap::GetGlobalRange() const {
  return _globalRange;
}

[[nodiscard]] Interval<int> Local2GlobalMap::GetGlobalRange(int eleIdx) const {
  MOCHI_ASSERT_VERBOSE(eleIdx < isize(_eleSizes), "Invalid element index");

  Span<int const> eleIndices = this->GetGlobalIndices(eleIdx);
  int maxIdx = std::numeric_limits<int>::min();
  int minIdx = std::numeric_limits<int>::max();
  for (int eleIndex : eleIndices) {
    if (eleIndex > maxIdx) {
      maxIdx = eleIndex;
    }
    if (eleIndex < minIdx) {
      minIdx = eleIndex;
    }
  }
  return Interval<int>{minIdx, maxIdx + 1};
}

[[nodiscard]] int Local2GlobalMap::GetElementSize(int eleIdx) const {
  MOCHI_ASSERT_VERBOSE(eleIdx < isize(_eleSizes), "Invalid element index");
  return _eleSizes[eleIdx];
}

[[nodiscard]] int Local2GlobalMap::GetElementOffset(int eleIdx) const {
  MOCHI_ASSERT_VERBOSE(eleIdx < isize(_eleOffsets), "Invalid element index");
  return _eleOffsets[eleIdx];
}

[[nodiscard]] Span<int const> Local2GlobalMap::GetGlobalIndices(int eleIdx) const {
  MOCHI_ASSERT_VERBOSE(eleIdx < isize(_eleSizes), "Invalid element index");
  return {_indices.data() + _eleOffsets[eleIdx], static_cast<size_t>(_eleSizes[eleIdx])};
}

[[nodiscard]] Span<int const> Local2GlobalMap::GetStencilIndices(int eleIdx) const {
  MOCHI_ASSERT_VERBOSE(eleIdx >= 0 && eleIdx < isize(_eleSizes), "Invalid element index");
  MOCHI_ASSERT_VERBOSE(!_stencilIndices.empty(), "Stencil indices not initialized");
  return {_stencilIndices.data() + _eleOffsets[eleIdx], static_cast<size_t>(_eleSizes[eleIdx])};
}

void Local2GlobalMap::GetElementNodes(int eleIdx, Span<int> outNodes) const {
  MOCHI_ASSERT_VERBOSE(eleIdx >= 0 && eleIdx < isize(_eleSizes), "Invalid element index.");
  MOCHI_ASSERT_VERBOSE(_numFields > 0, "Number of fields not initialized.");
  MOCHI_ASSERT_VERBOSE(_eleSizes[eleIdx] == _numFields * isize(outNodes), "Inconsistent sizes.");
  int const numEleNode = isize(outNodes);
  int const* eleBegin = _indices.data() + _eleOffsets[eleIdx];
  for (int i = 0; i < numEleNode; ++i) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    for (int j = 0; j < _numFields; ++j) {
      MOCHI_ASSERT_VERBOSE(
          *(eleBegin + i * _numFields + j) % _numFields == j,
          "All fields of a node expected to be stored consecutively.");
    }
#endif
    outNodes[i] = *(eleBegin + i * _numFields) / _numFields;
  }
}

[[nodiscard]] int Local2GlobalMap::GetGlobalIndex(int eleIdx, int localIdx) const {
  MOCHI_ASSERT_VERBOSE(eleIdx < isize(_eleSizes), "Invalid element index");
  MOCHI_ASSERT_VERBOSE(localIdx < _eleSizes[eleIdx], "Invalid local index");
  return *(_indices.data() + _eleOffsets[eleIdx] + localIdx);
}

void Local2GlobalMap::InitializePaddedIndices(int stride) {
  MOCHI_ASSERT_VERBOSE(!_eleSizes.empty(), "Primary initialization must be called first.");
  MOCHI_ASSERT_VERBOSE(!HasPaddedIndices(), "Padded indices have already been initialized.");
  MOCHI_ASSERT_VERBOSE(_numFields > 0, "Number of fields not initialized.");
  MOCHI_ASSERT_VERBOSE(stride > 0, "Stride must be positive.");
  _paddedStride = stride;
  int const numElements = isize(_eleSizes);
  _paddedIndices.resize_noinit(numElements * stride);
  for (int e = 0; e < numElements; ++e) {
    auto indices = GetGlobalIndices(e);
    int const eleSize = isize(indices);
    MOCHI_ASSERT_VERBOSE(stride >= eleSize, "Stride must be >= element size.");
    int const base = e * stride;
    // Fill all positions with a repeating pattern of node 0's DoF indices. For multi-field nodes,
    // each padded node slot must contain the correct per-field DoF indices [g, g+1, ...,
    // g+numFields-1], not a single repeated value.
    for (int d = 0; d < stride; ++d) {
      _paddedIndices[base + d] = indices[d % _numFields];
    }
    if (_stencilIndices.empty()) {
      // No stencil: overwrite sequentially.
      for (int d = 0; d < eleSize; ++d) {
        _paddedIndices[base + d] = indices[d];
      }
    } else {
      // Stencil-aware: place each DoF at its stencil position.
      auto stencil = GetStencilIndices(e);
      for (int d = 0; d < eleSize; ++d) {
        MOCHI_ASSERT_VERBOSE(
            stencil[d] >= 0 && stencil[d] < stride, "Stencil index out of padded stride range.");
        _paddedIndices[base + stencil[d]] = indices[d];
      }
    }
  }
}

[[nodiscard]] Span<int const> Local2GlobalMap::GetPaddedGlobalIndices() const {
  MOCHI_ASSERT_VERBOSE(HasPaddedIndices(), "Padded indices not initialized.");
  return _paddedIndices;
}

[[nodiscard]] bool Local2GlobalMap::HasPaddedIndices() const {
  MOCHI_ASSERT_VERBOSE(
      (_paddedStride == 0) == _paddedIndices.empty(), "Inconsistent internal state.");
  return !_paddedIndices.empty();
}

[[nodiscard]] int Local2GlobalMap::GetPaddedStride() const {
  MOCHI_ASSERT_VERBOSE(HasPaddedIndices(), "Padded indices not initialized.");
  return _paddedStride;
}

void Local2GlobalMap::InitializeStencilIndices(Graph<int, int> const& nodalStencilIndices) {
  MOCHI_ASSERT_VERBOSE(_numFields > 0, "Global indices not yet initialized");
  MOCHI_ASSERT_VERBOSE(_stencilIndices.empty(), "Stencil indices have already been initialized.");
  int const numElements = isize(nodalStencilIndices);
  MOCHI_ASSERT_VERBOSE(
      numElements == isize(_eleSizes),
      "Global indices do not match number of elements for stencil indices");
  _stencilIndices.reserve(isize(_indices));
  for (int elementIndex = 0; elementIndex < numElements; elementIndex++) {
    auto const& nodeStencil = nodalStencilIndices[elementIndex];
    MOCHI_ASSERT_VERBOSE(
        _numFields * isize(nodeStencil) == _eleSizes[elementIndex],
        "Nodal stencil does not match global indices");
    // Convert each nodal stencil index into a group of consecutive DoF stencil indices.
    for (int const nodeStencilIndex : nodeStencil) {
      for (int j = 0; j < _numFields; j++) {
        _stencilIndices.push_back(nodeStencilIndex * _numFields + j);
      }
    }
  }
}

} // namespace mochi
