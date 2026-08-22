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

#include <mochi_core/contact/contact_samples_bsh.h>

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

using namespace mochi;

void mochi::ContactSamplesBsh::Update(Span<Real3 const> samplePositions, int nodeIdx) {
  MOCHI_ASSERT_VERBOSE(isize(samplePositions) == NumSamples(), "Inconsistent number of samples.");
  auto& nodeData = GetNodeData(nodeIdx);

  bool const isInternalNode = (nodeData.sampleIdx < 0);
  if (isInternalNode) {
    // Internal nodes obtain their position as the average of their children. There may be a better
    // way (i.e. optimize the sphere location for minimum radius), but this is fine for now.
    nodeData.position = Real3{0_r, 0_r, 0_r};
  } else {
    nodeData.position = samplePositions[nodeData.sampleIdx];
  }

  auto childrenIndices = Children(nodeIdx);
  for (int childIdx : childrenIndices) {
    // Recursively update children.
    Update(samplePositions, childIdx);
    if (isInternalNode) {
      nodeData.position += _nodesData[childIdx].position;
    }
  }

  if (isInternalNode) {
    MOCHI_ASSERT_VERBOSE(!childrenIndices.empty(), "Internal node must have children.");
    nodeData.position /= static_cast<real>(childrenIndices.size());
  }

  // Compute the radius as the smallest radius containing all of the child spheres.
  nodeData.radius = 0_r;
  for (int childIdx : childrenIndices) {
    auto const& childData = _nodesData[childIdx];
    real smallestRadiusContaining = Norm(childData.position - nodeData.position) + childData.radius;
    nodeData.radius = Max(smallestRadiusContaining, nodeData.radius);
  }
}

static void RecursiveLoadFromNumpyArrays(
    int childId, // Index of the child in 'childrenRanges' and 'sampleIds'
    int parentIdx,
    ContactSamplesBsh& bsh,
    Span<int const> children,
    Span<Int2 const> childrenRanges,
    Span<int> sampleIds) {
  // Add node to the BSH.
  int const sampleIdx = sampleIds[childId];
  int const childKey = bsh.AddNode(
      ContactSamplesBsh::NodeData{.sampleIdx = (sampleIdx >= 0) ? sampleIdx : -1}, parentIdx);

  // Recurse on the children.
  auto childrenRange = childrenRanges[childId];
  for (int i = childrenRange[0]; i < childrenRange[1]; ++i) {
    auto nextChildIdx = children[i];
    RecursiveLoadFromNumpyArrays(nextChildIdx, childKey, bsh, children, childrenRanges, sampleIds);
  }
}

ContactSamplesBsh mochi::ContactSamplesBsh::LoadFromNumpyArrays(
    Span<int const> children,
    Span<Int2 const> childrenRanges,
    Span<int> sampleIds,
    Span<int> roots) {
  NodeData rootData{.sampleIdx = -1}; // Root node is always "internal".
  ContactSamplesBsh bsh(rootData);
  for (int root : roots) {
    RecursiveLoadFromNumpyArrays(root, RootIdx(), bsh, children, childrenRanges, sampleIds);
  }
  return bsh;
}
