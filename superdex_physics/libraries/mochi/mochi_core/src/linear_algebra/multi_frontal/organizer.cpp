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

#include <mochi_core/linear_algebra/multi_frontal/front_matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/organizer.h>
#include <mochi_core/utils/graph_utils.h>

namespace mochi {

namespace {

struct FrontCostMultipliers {
  double triFactorCoef;
  double triSolveCoef;
  double rankUpdateCoef;
};

/// \brief Get a cost factor for factorizing and eliminating a super-node.
double SupernodeCost(int superWidth, int superHeight, FrontCostMultipliers const& multipliers) {
  superHeight -= superWidth;
  return multipliers.triFactorCoef * superWidth * superWidth * superWidth +
      multipliers.triSolveCoef * superWidth * superWidth * superHeight +
      multipliers.rankUpdateCoef * superWidth * superHeight * superHeight;
}

} // namespace

FrontalOrganizer::FrontalOrganizer(
    SymbolicEliminationTree const& tree,
    size_t blockSize,
    size_t dofsPerNode)
    : _blockSize(blockSize), _dofsPerNode(dofsPerNode) {
  MOCHI_ASSERT_VERBOSE(
      _blockSize % _dofsPerNode == 0, "Frontal block size must be a multiple of DoFs per node.");
  auto NSuper = tree.NumSuperNodes();
  _costs.resize(NSuper + 1, {0.0, 0.0, 0, 0});
  FrontCostMultipliers timeFactors{
      .triFactorCoef = 3.0, .triSolveCoef = 2.0, .rankUpdateCoef = 1.0};
  FrontCostMultipliers flopCountMultipliers{
      .triFactorCoef = 2.0 / 3.0, .triSolveCoef = 1.0, .rankUpdateCoef = 1.0};

  DynamicArray<size_t> firstChildFrontStack(NSuper + 1, 0);
  for (int i = 0; i < NSuper; ++i) {
    auto superSize = tree.SuperSize(i);
    auto superHeight = isize(tree.SuperIndices(i));
    auto frontNodeCount = superHeight - superSize;
    _costs[i].time += SupernodeCost(superSize, superHeight, timeFactors);
    _costs[i].flops += SupernodeCost(superSize, superHeight, flopCountMultipliers);
    _costs[i].frontSize = FrontStorageSize(frontNodeCount * _dofsPerNode, _blockSize);
    // Account for the fact that the first child's front will overlap its parent's.
    _costs[i].frontStackSize =
        std::max(firstChildFrontStack[i], _costs[i].frontSize + _costs[i].frontStackSize);
    auto parent = tree.SuperParent(i);
    parent = parent == -1 ? NSuper : parent;
    _costs[parent].time += _costs[i].time;
    _costs[parent].flops += _costs[i].flops;
    if (firstChildFrontStack[parent] == 0) {
      firstChildFrontStack[parent] = _costs[i].frontStackSize;
    } else {
      _costs[parent].frontStackSize =
          std::max(_costs[i].frontStackSize, _costs[parent].frontStackSize);
    }
  }

  GraphBuilder<int, int> idxBuilder{
      NSuper, static_cast<int>(tree.SuperIndices().NumTargets() - tree.NumNodes())};
  for (int i = 0; i < NSuper; ++i) {
    auto indices = tree.SuperIndices(i);
    idxBuilder.append(indices.subspan(tree.SuperSize(i)));
  }
  _indicesInParent = idxBuilder.Build(); // At this stage we have the actual node indices.
  // Loop over the supernodes and find the mapping in their parent's indices.
  DynamicArray<int> parentLocations(tree.NumNodes(), -1);
  for (int i = NSuper; --i >= 0;) {
    for (int j = 0; auto n : tree.SuperIndices(i)) {
      parentLocations[n] = j;
      ++j;
    }
    for (auto child : tree.TreeGraph()[i]) {
      for (auto& n : _indicesInParent[child]) {
        n = parentLocations[n];
        MOCHI_ASSERT_VERBOSE(n != -1);
      }
    }
#ifdef MOCHI_ASSERT_VERBOSE_ENABLED
    for (auto n : tree.SuperIndices(i)) {
      parentLocations[n] = -1;
    }
#endif
  }

  // There are at least NSuper ranges. Near the root of the tree, there may be
  // exactly one range per supernode while far away, there are likely many more. 2 * numSources is a
  // compromise.
  GraphBuilder<IndexRange, int> rangeBuilder{NSuper, 2 * NSuper};
  for (int i = 0; i < NSuper; ++i) {
    rangeBuilder.StartSet();
    auto relativeIndices = _indicesInParent[i];
    int nIndices = isize(relativeIndices);
    // nIndices == 0 at the root supernode (it has no parent); StartSet already
    // recorded the empty range set, so there is nothing to insert.
    if (nIndices > 0) {
      int startChild = 0;
      int startParent = relativeIndices[0];
      int currentLength = 1;
      for (int j = 1; j < nIndices; ++j) {
        if (relativeIndices[j] == relativeIndices[j - 1] + 1) {
          ++currentLength;
        } else {
          rangeBuilder.InsertTarget(IndexRange{startChild, startParent, currentLength});
          startChild = j;
          startParent = relativeIndices[j];
          currentLength = 1;
        }
      }
      rangeBuilder.InsertTarget(IndexRange{startChild, startParent, currentLength});
    }
  }
  _rangesInParent = rangeBuilder.Build();
}

std::pair<DynamicArray<int>, int> FrontalOrganizer::PickBranches(
    SymbolicEliminationTree const& tree,
    int rootNode,
    int nBranches) const {
  DynamicArray<int> pickedBranches;
  if (nBranches <= 0) {
    return {pickedBranches, 0};
  }
  pickedBranches.reserve(2 * nBranches);
  auto numSuperNodes = tree.NumSuperNodes();
  double totalTime = _costs[numSuperNodes].time;
  // Use half the per-branch budget as threshold so the algorithm produces
  // ~2x branches, giving the scheduler room to load-balance.
  double threshold = totalTime / (2.0 * nBranches);

  DynamicArray<int> stack;
  stack.reserve(nBranches);
  stack.push_back(rootNode);

  auto treeGraph = tree.TreeGraph();
  int trunkCount = 0; // Number of nodes that are within the truncated tree.
  while (!stack.empty()) {
    auto sn = stack.back();
    stack.pop_back();

    // Pick a node if its cost is at or below the threshold, or if it is a leaf
    // (a leaf above threshold is unlikely but must still be picked — it cannot be split further).
    if (_costs[sn].time <= threshold || tree.IsLeaf(sn)) {
      pickedBranches.push_back(sn);
    } else {
      ++trunkCount;
      for (auto child : treeGraph[sn]) {
        stack.push_back(child);
      }
    }
  }

  return {pickedBranches, trunkCount};
}

} // namespace mochi
