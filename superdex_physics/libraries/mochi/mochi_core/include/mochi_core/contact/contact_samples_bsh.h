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

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

#include <optional>

namespace mochi {

/*
Bounding Sphere Hierarchy (BSH) tree for the contact samples.
*/
class ContactSamplesBsh {
  class AncestorIterator {
   public:
    AncestorIterator(DynamicArray<int> const& parents, int nodeIdx, bool end)
        : _parents(parents), _currentNodeIdx(end ? -1 : parents[nodeIdx]) {}

    bool operator!=(AncestorIterator const& other) const {
      return _currentNodeIdx != other._currentNodeIdx;
    }

    int operator*() const {
      return _currentNodeIdx;
    }

    AncestorIterator& operator++() {
      if (_currentNodeIdx >= 0) {
        _currentNodeIdx = _parents[_currentNodeIdx];
      }
      return *this;
    }

   private:
    DynamicArray<int> const& _parents;
    int _currentNodeIdx;
  };

  class AncestorIterable {
   public:
    AncestorIterable(DynamicArray<int> const& parents, int nodeIdx)
        : _parents(parents), _nodeIdx(nodeIdx) {}

    AncestorIterator begin() const {
      return {_parents, _nodeIdx, false};
    }

    AncestorIterator end() const {
      return {_parents, _nodeIdx, true};
    }

   private:
    DynamicArray<int> const& _parents;
    int _nodeIdx;
  };

 public:
  struct NodeData {
    // Sample point index. "-1" if the node is an "internal" node to the BSH, that is, a node used
    // to generate the BSH but doesn't correspond to a sample point.
    int sampleIdx = -1;
    // Radius [m].
    real radius = 0_r;
    // Position [m].
    Real3 position = {};
  };

  explicit ContactSamplesBsh(NodeData const& rootData) {
    _nodesData.emplace_back(rootData);
    _children.emplace_back();
    _parents.emplace_back(-1); // Root has no parent
  }

  static constexpr int RootIdx() {
    return kRootIdx;
  }

  // Number of nodes in the BSH.
  size_t size() const {
    return _nodesData.size();
  }

  NodeData& GetNodeData(int nodeIdx) {
    return _nodesData[nodeIdx];
  }

  NodeData const& GetNodeData(int nodeIdx) const {
    return _nodesData[nodeIdx];
  }

  // Node index of the parent of a node. Empty if the node is the root.
  std::optional<int> Parent(int nodeIdx) const {
    return _parents[nodeIdx] >= 0 ? std::optional<int>(_parents[nodeIdx]) : std::nullopt;
  }

  // Node indices of the ancestors of a node.
  AncestorIterable Ancestors(int nodeIdx) const {
    MOCHI_ASSERT_VERBOSE(nodeIdx >= 0 && nodeIdx < _parents.size(), "Invalid node index.");
    return {_parents, nodeIdx};
  }

  // Node indices of the children of a node. Empty if the node is a leaf.
  Span<int const> Children(int nodeIdx) const {
    return _children[nodeIdx];
  }

  // Mapping from sample point index to BSH node index.
  int SampleIdxToNodeIdx(int sampleIdx) const {
    return _sampleIdxToNodeIdx[sampleIdx];
  }

  int NumSamples() const {
    return isize(_sampleIdxToNodeIdx);
  }

  // Add a node to the BSH.
  // NOTE: Not optimized. It may perform multiple reallocations.
  int AddNode(NodeData const& data, int parentIdx) {
    MOCHI_ASSERT(parentIdx >= 0 && parentIdx < isize(_nodesData), "Invalid parent index.");
    int const nodeIdx = isize(_nodesData);
    _nodesData.push_back(data);

    // Update children.
    _children.emplace_back();
    _children[parentIdx].push_back(nodeIdx);

    // Update parent.
    _parents.push_back(parentIdx);

    // Update sample-idx-to-bsh-node-idx mapping.
    if (data.sampleIdx >= 0) {
      if (_sampleIdxToNodeIdx.size() < data.sampleIdx + 1) {
        _sampleIdxToNodeIdx.resize(data.sampleIdx + 1, -1);
      }
      _sampleIdxToNodeIdx[data.sampleIdx] = nodeIdx;
    }

    return nodeIdx;
  }

  // Recursively update the data of a node and its children.
  void Update(Span<Real3 const> samplePositions, int nodeIdx = kRootIdx);

  static ContactSamplesBsh LoadFromNumpyArrays(
      Span<int const> children,
      Span<Int2 const> childrenRanges,
      Span<int> sampleIds,
      Span<int> roots);

 private:
  static constexpr int kRootIdx = 0;

  // Node data. Size is equal to the number of nodes.
  DynamicArray<NodeData> _nodesData{};
  // Indices of the children nodes. Size is equal to the number of nodes.
  // TODO: Vector of vectors may be suboptimal. Explore other storage types, e.g. CSR.
  DynamicArray<DynamicArray<int>> _children;
  // Index of the parent node. "-1" if the parent is the root. Size is equal to the number of nodes.
  DynamicArray<int> _parents;
  // Mapping from sample point index to BSH node index. Size is equal to the number of sample
  // points.
  DynamicArray<int> _sampleIdxToNodeIdx;
};

} // namespace mochi
