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

#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/interval.h>
#include <mochi_core/utils/span.h>

namespace mochi {

/**
 * Utility class mapping eles of local indices (elements) onto a unique global
 * indexing. Local indices are assumed to be in the interval [0, elementSize).
 * Several local indices may be mapped onto the same global index. No assumption
 * is made about the range, order, and/or sparsity of the global indices.
 */
class Local2GlobalMap {
 public:
  /**
   * Creates an empty map.
   */
  Local2GlobalMap() = default;

  /**
   * Initializes the map to represent the connectivity of the given mesh and vertex fields.
   */
  template <typename MeshT>
  void InitializeFromMesh(MeshT const* mesh, int numFields);

  /**
   * Initializes the map from an arbitrary connectivity of nodes for each element. The connectivity
   * type must produce an int when indexed into twice, e.g., connectivity[elementIndex][nodeIndex]
   * will be an integer. E.g., both Span<NdArray<int, kNumNodes>> and Graph<int,int> would satisfy
   * this requirement.
   */
  template <typename ConnectivityType>
  void InitializeFromElementNodeConnectivity(
      ConnectivityType const& nodalConnectivity,
      int numFields);

  /**
   * Initializes the map to represent the connectivity of the given mesh, assuming the
   * specified interpolation basis and number of store fields in the mesh. Mesh number
   * of nodes per element MUST match basis vertex nodes.
   *
   * TODO@Anybody: Basis with edge, face, or interior nodes are not implemented yet.
   */
  template <typename MeshT, typename BasisT>
  void InitializeFromMeshAndBasis(MeshT const* mesh, BasisT const& basis, int numFields);

  /**
   * @brief For problems requiring it, we can pass a mapping from local DoFs to positions in a
   * stencil, represented as a Graph.
   *
   * @warning Must be called after the primary initialization.
   */
  void InitializeStencilIndices(Graph<int, int> const& nodalStencilIndices);

  /** @brief Returns the total number of mapped indices. */
  [[nodiscard]] int GetNumIndices() const;

  /** @brief Returns the total number of mapped elements. */
  [[nodiscard]] int GetNumElements() const;

  /** @brief Returns the range [min(globalIndices), max(globalIndices)). */
  [[nodiscard]] Interval<int> GetGlobalRange() const;

  /** @brief Returns the range [min(eleGlobalIndices), max(eleGlobalIndices)) */
  [[nodiscard]] Interval<int> GetGlobalRange(int eleIdx) const;

  /** @brief Returns a span containing the sizes of the elements. */
  [[nodiscard]] Span<int const> GetElementSizes() const;

  /** @brief Returns a span containing the offsets of the elements. */
  [[nodiscard]] Span<int const> GetElementOffsets() const;

  /** @brief Returns a span containing all global indices. */
  [[nodiscard]] Span<int const> GetGlobalIndices() const;

  /** @brief Returns the number of indices of an element. */
  [[nodiscard]] int GetElementSize(int eleIdx) const;

  /** @brief Returns the starting offset of an element. */
  [[nodiscard]] int GetElementOffset(int eleIdx) const;

  /** @brief Returns a span with the global indices of an element. */
  [[nodiscard]] Span<int const> GetGlobalIndices(int eleIdx) const;

  /**
   * @brief Returns the stencil indices of each local DoF for an element. In standard FEM, this is
   * just [0, ..., (# of local DoFs) - 1] for each element, but it may be nontrivial for
   * macro-element stencils (e.g., bending stiffness in shell/shells).
   */
  [[nodiscard]] Span<int const> GetStencilIndices(int eleIdx) const;

  /** @brief Gets the global nodes of an element. */
  void GetElementNodes(int eleIdx, Span<int> outNodes) const;

  /** @brief Returns the global index of a specific local index. */
  [[nodiscard]] int GetGlobalIndex(int eleIdx, int localIdx) const;

  /**
   * @brief Builds a uniform-stride copy of the flat index array, padding short elements.
   *
   * @details For each element, the padded array reserves @p stride entries. The element's real
   * DoFs are written into those entries either sequentially (when no stencil is set) or at the
   * positions given by @ref GetStencilIndices (when @ref InitializeStencilIndices has been
   * called). All remaining (padded) entries are filled with a repeating copy of the per-field DoF
   * indices of the element's local node 0 — that is, if node 0's global DoFs are
   * `g0, g1, ..., g(numFields - 1)`, the padded entries cycle through that same sequence.
   *
   * Aliasing the padded entries onto node-0's DoFs avoids introducing spurious global connections:
   * a downstream scatter that writes zero into every padded entry leaves the global state
   * unchanged. It is the consumer's responsibility to treat padded entries as zero-contribution
   * slots (or, equivalently, to ensure that any value scattered into them is zero).
   *
   * @param[in] stride  Number of padded entries per element. Must be `>=` every element's size.
   *
   * @warning Must be called after the primary initialization. If a stencil is needed, @ref
   * InitializeStencilIndices must also be called before this method.
   *
   * @see GetPaddedGlobalIndices, HasPaddedIndices, GetPaddedStride, InitializeStencilIndices,
   * BuildPaddedConnectivity, BuildPaddedNodalBasedStructure
   */
  void InitializePaddedIndices(int stride);

  /** @brief Returns the padded flat index array. Requires @ref InitializePaddedIndices to have been
   * called.
   */
  [[nodiscard]] Span<int const> GetPaddedGlobalIndices() const;

  /** @brief Returns true if padded indices have been initialized. */
  [[nodiscard]] bool HasPaddedIndices() const;

  /** @brief Returns the padded stride (entries per element). Requires @ref InitializePaddedIndices
   * to have been called. */
  [[nodiscard]] int GetPaddedStride() const;

 private:
  Interval<int> _globalRange;
  DynamicArray<int> _eleSizes;
  DynamicArray<int> _eleOffsets;
  DynamicArray<int> _indices;
  DynamicArray<int> _stencilIndices;
  DynamicArray<int> _paddedIndices;
  int _paddedStride = 0;
  int _numFields = 0;
};

template <typename ConnectivityType>
void Local2GlobalMap::InitializeFromElementNodeConnectivity(
    ConnectivityType const& nodalConnectivity,
    int numFields) {
  MOCHI_ASSERT(
      _eleSizes.empty() && _eleOffsets.empty() && _indices.empty(),
      "Local2GlobalMap has already been initialized.");
  _numFields = numFields;
  // -1 means no node has been observed, so empty connectivity keeps _globalRange empty.
  int maxNodeIndex = -1;
  int const numElements = isize(nodalConnectivity);
  _eleSizes.reserve(numElements);
  _eleOffsets.reserve(numElements);
  int nodesSoFar = 0;
  for (int elementIndex = 0; elementIndex < numElements; elementIndex++) {
    auto const& elementConnectivity = nodalConnectivity[elementIndex];
    _eleSizes.push_back(numFields * isize(elementConnectivity));
    _eleOffsets.push_back(nodesSoFar * numFields);
    for (int globalNodeIndex : elementConnectivity) {
      nodesSoFar++;
      maxNodeIndex = std::max(maxNodeIndex, globalNodeIndex);
      for (int field = 0; field < numFields; field++) {
        int const indexValue = globalNodeIndex * numFields + field;
        _indices.push_back(indexValue);
      }
    }
  }
  _globalRange = Interval<int>{0, numFields * (maxNodeIndex + 1)};
}

template <typename MeshT>
void Local2GlobalMap::InitializeFromMesh(MeshT const* mesh, int numFields) {
  MOCHI_ASSERT(numFields > 0, "At least one mesh field required");
  InitializeFromElementNodeConnectivity(mesh->GetElementConnectivity(), numFields);
}

template <typename MeshT, typename BasisT>
void Local2GlobalMap::InitializeFromMeshAndBasis(
    MeshT const* mesh,
    BasisT const& basis,
    int numFields) {
  static_assert(BasisT::kPolyOrder == 1, "higher order polynomials not implemented");
  static_assert(MeshT::kNodesPerElement == BasisT::kVertexDof, "Element size mismatch");
  MOCHI_ASSERT(numFields > 0, "At least one mesh field required");
  MOCHI_ASSERT(
      _eleSizes.empty() && _eleOffsets.empty() && _indices.empty(),
      "Local2GlobalMap has already been initialized.");
  _numFields = numFields;

  int const numElems = mesh->GetNumElements();
  int const numTotalNodes = mesh->GetNumNodes();
  int const numEleNodes = basis.kNumDofs;
  int const numEleDofs = numEleNodes * numFields;
  int const numGloDofs = numTotalNodes * numFields;
  int const numLocDofs = numEleDofs * numElems;

  // Initialize element sizes and offsets
  _eleSizes.resize_noinit(numElems);
  _eleOffsets.resize_noinit(numElems);
  for (int i = 0; i < numElems; ++i) {
    _eleSizes[i] = numEleDofs;
    _eleOffsets[i] = i * numEleDofs;
  }

  // Initialize global indices vector
  Span<int const> topology = mesh->GetFlatConnectivity();
  _globalRange = Interval<int>{0, numGloDofs};
  _indices.reserve(numLocDofs);
  _stencilIndices.reserve(numLocDofs);
  int dofCount = 0;
  for (int n : topology) {
    for (int f = 0; f < numFields; ++f) {
      _indices.push_back(n * numFields + f);
      _stencilIndices.push_back(dofCount % numEleDofs);
      dofCount++;
    }
  }
}

} // namespace mochi
