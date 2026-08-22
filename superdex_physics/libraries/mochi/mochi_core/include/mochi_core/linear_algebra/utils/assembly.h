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

#include <mochi_core/linear_algebra/base_tools.h>
#include <mochi_core/linear_algebra/matrix_fwd.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/assembly_params.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph_utils.h>
#include <mochi_core/utils/nd_array.h>

#include <array>
#include <atomic>
#include <vector>

namespace mochi::details {
struct AtomicDetails {
  static constexpr int kMaskBits = 64;
  uint64_t mask;
  uint32_t index; // Must be uint32_t. Callers compare against UINT32_MAX.
};
} // namespace mochi::details

namespace mochi {

/** @brief Precomputed data structures to accelerate parallel FEM assembly. */
struct NodalBasedStructure {
  template <size_t kMaxNumNodes>
  NodalBasedStructure(Span<NdArray<int, kMaxNumNodes> const> const& connectivity)
      : NodalBasedStructure(GraphFromRangeOfRanges<int, int>(connectivity)) {}

  /** @brief Construct from element-to-node connectivity.
   *
   * @param[in] eToN Element-to-node connectivity graph.
   * @param[in] nToN Optional node-to-node connectivity. When empty (default), the node-to-node
   * pattern is derived from @p eToN, i.e., two nodes are connected iff they share an element.
   *
   * @details The node-to-node graph determines the layout of the target @ref BlockSparseMatrix that
   * assembly will scatter into. Provide @p nToN when the target matrix has a wider sparsity pattern
   * than what @p eToN alone implies — for example, when assembling boundary (surface) contributions
   * into a volumetric stiffness matrix whose sparsity was established from the full volumetric
   * connectivity.
   *
   * @note When @p nToN is provided, it must be a superset of the pattern implied by @p eToN (every
   * element's node pair must appear in it). Rows must be sorted in ascending order.
   * @note Element grouping, coloring, and inter-group dependencies are always derived from @p eToN
   * regardless of @p nToN.
   */
  NodalBasedStructure(Graph<int, int> eToN, Graph<int, int> const& nToN = {});

  MOCHI_DECLARE_NO_COPY(NodalBasedStructure);
  MOCHI_DECLARE_MOVE(NodalBasedStructure);

  void CreateSubTasks();

  void CreateDepMasks();

  static details::AtomicDetails AtomicIdx(int group) {
    auto ug = static_cast<uint32_t>(group);
    return {static_cast<uint64_t>(1) << (ug % kMaskBits), ug / kMaskBits};
  }

  static constexpr auto kMaskBits = details::AtomicDetails::kMaskBits;

  auto NumElements() const {
    return isize(_eToN);
  }

  auto NumGroups() const {
    MOCHI_ASSERT_VERBOSE(
        isize(_groupColor) == isize(_elemGroups) && isize(_groupColor) == isize(_dependentGroups) &&
            isize(_groupColor) == isize(_groupDependsOn),
        "Inconsistent number of groups.");
    return isize(_groupColor);
  }

  auto NumColors() const {
    return isize(_elemGroupColoring);
  }

  Span<int const> GetEleNodes(int eIdx) const {
    MOCHI_ASSERT_VERBOSE(eIdx >= 0 && eIdx < NumElements(), "Element out of range.");
    return _eToN[eIdx];
  }

  RowMatrixView<int const> GetNodeSparseIndices(int eIdx) const {
    MOCHI_ASSERT_VERBOSE(eIdx >= 0 && eIdx < NumElements(), "Element out of range.");
    int const numNodes = _eToN.EdgeCount(eIdx);
    return {&_nodeSparseIndices[eIdx * _maxNodesPerElementSquared], numNodes, numNodes};
  }

  Span<int const> GetElemsInGroup(int gIdx) const {
    MOCHI_ASSERT_VERBOSE(gIdx >= 0 && gIdx < NumGroups(), "Group out of range.");
    return _elemGroups[gIdx];
  }

  auto const& GetDependentGroups() const {
    return _dependentGroups;
  }

  auto const& GetDepMasks() const {
    return _depMasks;
  }

  auto IsInitialReady(int maskIdx) const {
    return _initialReady[maskIdx];
  }

  /// @brief Returns the node-to-node connectivity graph.
  [[nodiscard]] auto const& GetNToN() const {
    return _nToN;
  }

 private:
  /// @brief Element-to-node connectivity.
  Graph<int, int> _eToN;
  /// @brief Node-to-element connectivity.
  Graph<int, int> _nToE;
  /// @brief Node-to-node connectivity. (Pattern of the mesh matrix in BlockSparseMatrix format).
  Graph<int, int> _nToN;
  /// @brief Precomputed locations of the elemental blocks in the BlockSparseMatrix.
  /// @details Each element's indices are stored in a block of size '_maxNodesPerElementSquared',
  /// which is padded with trailing sentinel values if the element has fewer than the maximum number
  /// of nodes.
  DynamicArray<int> _nodeSparseIndices;
  /// @brief Maximum number of nodes per element squared.
  int _maxNodesPerElementSquared;

  /// @brief Decomposition of elements into small group, independent of the original coarser
  /// decompositions.
  Graph<int, int> _elemGroups;
  Graph<int, int> _elemGtoG;
  DynamicArray<int> _groupColor;
  /// @brief Coloring into not intra-connected group sets.
  Graph<int, int> _elemGroupColoring;
  /// @brief Dependencies of groups on lower-number-color groups.
  Graph<int, int> _groupDependsOn;
  /// @brief Reverse of _groupDependsOn.
  Graph<int, int> _dependentGroups;
  /// @brief Dependencies made into mask for atomic variables.
  Graph<details::AtomicDetails, int> _depMasks;
  /// @brief Initial ready mask for atomic variables.
  DynamicArray<int64_t> _initialReady;
};

/**
 * @brief Pads variable-width element connectivity to a fixed stencil width for FEM assembly.
 *
 * @details For each raw node `connectivity[e][i]`, `stencilGraph[e][i]` gives its slot in the
 * padded element. Slots not referenced by the stencil are filled with the element's first node,
 * matching the padding convention used by @ref Local2GlobalMap::InitializePaddedIndices.
 *
 * The returned connectivity must be paired with an L2G initialized from the same raw connectivity
 * and stencil graph. Element operations must produce zero contribution for padded slots, because
 * padded slots alias node-0 DoFs.
 *
 * @tparam kNumStencilNodes Fixed number of node slots per padded element.
 * @param[in] connectivity Variable-width element-to-node connectivity. Each element must have at
 * least one node and at most @p kNumStencilNodes nodes.
 * @param[in] stencilGraph Slot of each raw node within the padded element. Same shape as @p
 * connectivity.
 * @return Padded connectivity with exactly @p kNumStencilNodes nodes per element.
 *
 * @see BuildPaddedNodalBasedStructure, Local2GlobalMap::InitializeStencilIndices,
 * Local2GlobalMap::InitializePaddedIndices
 */
template <size_t kNumStencilNodes>
[[nodiscard]] DynamicArray<NdArray<int, kNumStencilNodes>> BuildPaddedConnectivity(
    Graph<int, int> const& connectivity,
    Graph<int, int> const& stencilGraph) {
  int const numElements = isize(connectivity);
  MOCHI_ASSERT_VERBOSE(isize(stencilGraph) == numElements);

  DynamicArray<NdArray<int, kNumStencilNodes>> paddedConnectivity;
  paddedConnectivity.resize_noinit(numElements);
  for (int eleIdx = 0; eleIdx < numElements; ++eleIdx) {
    auto const nodes = connectivity[eleIdx];
    auto const stencil = stencilGraph[eleIdx];
    MOCHI_ASSERT_VERBOSE(!nodes.empty());
    MOCHI_ASSERT_VERBOSE(isize(nodes) == isize(stencil));
    MOCHI_ASSERT_VERBOSE(isize(nodes) <= static_cast<int>(kNumStencilNodes));

    int const padNode = nodes[0];
    auto& padded = paddedConnectivity[eleIdx];
    for (int i = 0; i < static_cast<int>(kNumStencilNodes); ++i) {
      padded[i] = padNode;
    }
    for (int i = 0; i < isize(nodes); ++i) {
      MOCHI_ASSERT_VERBOSE(stencil[i] >= 0 && stencil[i] < static_cast<int>(kNumStencilNodes));
      padded[stencil[i]] = nodes[i];
    }
  }
  return paddedConnectivity;
}

/**
 * @brief Builds a @ref NodalBasedStructure for fixed-width FEM assembly from variable-width
 * connectivity.
 *
 * @details The element-to-node graph is padded with @ref BuildPaddedConnectivity so every element
 * exposes @p kNumStencilNodes nodes to the assembler. The node-to-node graph is preserved from the
 * original raw connectivity so the repeated pad node does not introduce duplicate sparsity entries.
 *
 * Use this with an L2G initialized from the same raw connectivity and stencil graph, then padded to
 * `kNumStencilNodes * numFields`.
 *
 * @tparam kNumStencilNodes Fixed number of node slots per padded element.
 * @param[in] connectivity Variable-width element-to-node connectivity.
 * @param[in] stencilGraph Slot of each raw node within the padded element. Same shape as @p
 * connectivity.
 * @return Nodal-based structure whose element-to-node graph is padded to @p kNumStencilNodes nodes
 * per element while retaining the original (raw) node-to-node sparsity.
 *
 * @see BuildPaddedConnectivity, Local2GlobalMap::InitializeStencilIndices,
 * Local2GlobalMap::InitializePaddedIndices
 */
template <size_t kNumStencilNodes>
[[nodiscard]] NodalBasedStructure BuildPaddedNodalBasedStructure(
    Graph<int, int> const& connectivity,
    Graph<int, int> const& stencilGraph) {
  NodalBasedStructure const originalNbs(connectivity);
  auto const paddedConnectivity =
      BuildPaddedConnectivity<kNumStencilNodes>(connectivity, stencilGraph);
  return {GraphFromRangeOfRanges<int, int>(paddedConnectivity), originalNbs.GetNToN()};
}

struct WorkState {
  using AtomicDetails = typename details::AtomicDetails;
  static constexpr auto kMaskBits = AtomicDetails::kMaskBits;

  WorkState(NodalBasedStructure const& nbs)
      : dependsOn(nbs.GetDepMasks()), dependents(nbs.GetDependentGroups()) {
    int nGroups = nbs.NumGroups();
    int nAtomic = (nGroups + kMaskBits - 1) / kMaskBits;
    finished = std::vector<std::atomic<uint64_t>>(nAtomic);
    acquired = std::vector<std::atomic<uint64_t>>(nAtomic);
    ready = std::vector<std::atomic<uint64_t>>(nAtomic);
    for (int i = 0; i < nAtomic; ++i) {
      ready[i] = nbs.IsInitialReady(i);
    }
    // The unused upper bits of the last `acquired` mask are set to 1 to ease checking that all work
    // has completed.
    if (nGroups % kMaskBits > 0) {
      acquired.back().store((~static_cast<uint64_t>(0)) << (nGroups % kMaskBits));
    }
  }

  /** @brief Mark a group as completed.
   *
   * @param group The group whose assembly is finished.
   * @param taskReadyMask Mask used to communicate to the calling task which group(s) may have been
   * readied by the completion.
   */
  void Complete(int group, Span<uint64_t> taskReadyMask = {});

  /** @note The calling thread must have been synchronized with all upstream threads of the group(s)
   * that are ready for execution using release-acquire semantics.
   */
  int Acquire(Span<uint64_t> taskReadyMask);

  int AcquireAny();

  static AtomicDetails AtomicIdx(int group) {
    auto ug = static_cast<uint32_t>(group);
    return {static_cast<uint64_t>(1) << (ug % kMaskBits), ug / kMaskBits};
  }

  std::vector<std::atomic<uint64_t>> finished;
  std::vector<std::atomic<uint64_t>> acquired;
  std::vector<std::atomic<uint64_t>> ready;
  Graph<AtomicDetails, int> const& dependsOn;
  Graph<int, int> const& dependents;
};

} // namespace mochi
