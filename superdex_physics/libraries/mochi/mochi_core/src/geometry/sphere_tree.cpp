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

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/sphere_tree.h>
#include <mochi_core/memory/aligned_allocator.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/math_utils.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>

using namespace mochi;

template <int kNodeSize>
SphereTree<kNodeSize>::SphereTree() : _spheres(GetCacheAlignedAllocator()) {}

// Split a range of indices log2(N) times to produce N child ranges. N must be a power of two.
template <int N>
static NdArray<int, N, 2>
PartitionPow2(Span<Real3 const> points, Span<int> indices, int begin, int end) {
  static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of two");

  int const count = end - begin;

  // When the number of points fits within a single node's children, assign one point per child.
  // Each child becomes a leaf (or empty), avoiding unnecessary tree levels below this node.
  if (count <= N) {
    NdArray<int, N, 2> outRanges;
    for (int c = 0; c < N; ++c) {
      outRanges[c][0] = begin + Min(c, count);
      outRanges[c][1] = begin + Min(c + 1, count);
    }
    return outRanges;
  }

  NdArray<int, N + 1> boundaries;
  boundaries[0] = begin;
  boundaries[N] = end;

  auto splitRange = [&](int rBegin, int rEnd) -> int {
    MOCHI_ASSERT_VERBOSE(rEnd >= rBegin);
    if (rEnd == rBegin) {
      return rBegin;
    }

    // Find the midpoint of the longest axis of the AABB containing these points. This heuristic
    // efficiently partitions the volume, which is more important for collision detection than
    // having a perfectly balanced tree. Early rejection of non-overlapping volumes is key.
    Aabb aabb = CalcAabbWithSortedIndices(points, indices.subspan(rBegin, rEnd - rBegin));
    auto const axis = ArgMax(aabb.GetSize());
    real const pivot = 0.5_r * (aabb.GetMin()[axis] + aabb.GetMax()[axis]);

    // Partition indices, keeping each half in sorted order. This ensures consistent node
    // ordering across different toolchains (unlike std::partition).
    auto mid =
        std::stable_partition(indices.begin() + rBegin, indices.begin() + rEnd, [&](int idx) {
          return points[idx][axis] < pivot;
        });

    int result = static_cast<int>(mid - indices.begin());

    // If the spatial partition failed to split (all points on one side due to coincident
    // coordinates), fall back to splitting at the index midpoint. This guarantees progress and
    // prevents infinite loops during tree construction.
    if (result == rBegin || result == rEnd) {
      result = rBegin + (rEnd - rBegin) / 2;
    }

    return result;
  };

  // Perform log2(N) levels of binary splitting.
  // Each level halves the stride, doubling the number of ranges:
  //   Level 0: 1 split  -> 2 ranges   (stride = N/2)
  //   Level 1: 2 splits -> 4 ranges   (stride = N/4)
  //   ...
  //   Level log2(N)-1: N/2 splits -> N ranges (stride = 1)
  for (int stride = N / 2; stride >= 1; stride /= 2) {
    for (int j = stride; j < N; j += 2 * stride) {
      boundaries[j] = splitRange(boundaries[j - stride], boundaries[j + stride]);
    }
  }

  // Output the N ranges, compacting non-empty ranges to the front.
  NdArray<int, N, 2> outRanges;
  int writeIdx = 0;
  for (int c = 0; c < N; ++c) {
    if (boundaries[c] != boundaries[c + 1]) {
      outRanges[writeIdx][0] = boundaries[c];
      outRanges[writeIdx][1] = boundaries[c + 1];
      MOCHI_ASSERT_VERBOSE(
          std::is_sorted(
              indices.begin() + outRanges[writeIdx][0], indices.begin() + outRanges[writeIdx][1]),
          "Indices were sorted at the start and they should stay sorted as we partition them.");
      ++writeIdx;
    }
  }

  // Any trailing ranges must be empty
  for (; writeIdx < N; ++writeIdx) {
    outRanges[writeIdx][0] = outRanges[writeIdx][1] = end;
  }

  return outRanges;
}

template <int kNodeSize>
SphereTree<kNodeSize> SphereTree<kNodeSize>::FromPoints(Span<Real3 const> points, int maxPerLeaf) {
  MOCHI_ASSERT(points.size() < std::numeric_limits<int>::max(), "Too many points");
  MOCHI_ASSERT(maxPerLeaf > 0, "maxPerLeaf must be positive");

#if MOCHI_ASSERT_VERBOSE_ENABLED
  real constexpr kMaxFloat = static_cast<real>(std::numeric_limits<float>::max());
  for (Real3 const& point : points) {
    for (int axis = 0; axis < 3; ++axis) {
      MOCHI_ASSERT_VERBOSE(
          point[axis] >= -kMaxFloat && point[axis] <= kMaxFloat,
          "SphereTree point coordinates must be finite and within the finite float range.");
    }
  }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

  SphereTree tree;
  tree._maxPerLeaf = maxPerLeaf;

  // Initialize indices
  tree._indices.reserve(points.size() + Simd<int>::kSize); // Padding for SIMD
  tree._indices.resize_noinit(points.size());
  std::iota(tree._indices.begin(), tree._indices.end(), 0);

  // Compute outer bounds from the full span of points.
  // This results in a smaller sphere than combining the child bounds.
  tree._bounds = CalcBoundingSphere(points, BoundingSphereAlgorithm::Best);

  // Approximate the amount of memory needed for the tree, assuming each leaf is half full.
  int const numPoints = isize(points);
  int64_t const estimatedNumNonEmptyLeaves =
      Min<int64_t>(numPoints, (2 * int64_t{numPoints} + tree._maxPerLeaf - 1) / tree._maxPerLeaf);
  int64_t const estimatedNumLeaves = estimatedNumNonEmptyLeaves + 1; // First leaf is always empty.
  size_t const estimatedNumNodes = static_cast<size_t>(
      Max<int64_t>(1, (estimatedNumNonEmptyLeaves + kNodeSize - 3) / (kNodeSize - 1)));
  tree._spheres.reserve(estimatedNumNodes);
  tree._children.reserve(estimatedNumNodes);
  tree._leaves.reserve(estimatedNumLeaves);

  // First node always exists.
  int numNodes = 1;
  tree._spheres.resize(numNodes);
  tree._children.resize(numNodes);

  // First leaf must always be empty.
  int numLeaves = 1;
  tree._leaves.resize(numLeaves);

  if (points.empty()) {
    // The tree currently has one node which points to empty children. Leave it as is.
    return tree;
  }

  struct WorkItem {
    int nodeIndex = 0;
    int begin = 0;
    int end = 0;
  };

  std::deque<WorkItem> workQueue;
  workQueue.push_back({0, 0, numPoints});

  // Construct the tree in BFS order. Query traversal uses the same order, so node and leaf
  // indices will be discovered in ascending order.
  while (!workQueue.empty()) {
    WorkItem item = workQueue.front();
    workQueue.pop_front();
    MOCHI_ASSERT_VERBOSE(item.end > item.begin);

    // Split into kNodeSize child ranges
    auto childRanges = PartitionPow2<kNodeSize>(points, tree._indices, item.begin, item.end);
    MOCHI_ASSERT_VERBOSE(
        childRanges[0][1] > childRanges[0][0],
        "Empty child ranges should be at the end of the list, so the first child range should never be empty.");

#if MOCHI_ASSERT_VERBOSE_ENABLED
    int countNextLevel = childRanges[0][1] - childRanges[0][0];
    for (int i = 1; i < kNodeSize; ++i) {
      MOCHI_ASSERT_VERBOSE(childRanges[i][0] >= childRanges[i - 1][1]);
      MOCHI_ASSERT_VERBOSE(childRanges[i][1] >= childRanges[i][0]);
      countNextLevel += childRanges[i][1] - childRanges[i][0];
    }
    MOCHI_ASSERT_VERBOSE(childRanges[kNodeSize - 1][1] == item.end);
    MOCHI_ASSERT_VERBOSE(countNextLevel == (item.end - item.begin));
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

    // Non-SIMD version of NodeSpheres for building the data one child at a time. Stored in the same
    // SoA layout as NodeSpheres (centers[0] = all X, centers[1] = all Y, centers[2] = all Z) so the
    // per-coordinate Load<VF> below reads contiguous SIMD lanes.
    struct TempNodeSpheres {
      NdArray<float, 3, kNodeSize> centers;
      NdArray<float, kNodeSize> radii;
    };

    TempNodeSpheres childSpheres{};
    NdArray<int, kNodeSize> childIndices;

    // Process the children
    for (int c = 0; c < kNodeSize; ++c) {
      int cBegin = childRanges[c][0];
      int cEnd = childRanges[c][1];
      int cCount = cEnd - cBegin;

      // Assemble a packet of child spheres for this node.
      if (cCount > 0) {
        Sphere const sphere = CalcBoundingSphereIndexed(
            points,
            MakeConstSpan(tree._indices).subspan(cBegin, cCount),
            BoundingSphereAlgorithm::Best);

        // Store the sphere data using single-precision floats even when mochi::real is double.
        Real3 const& center = sphere.GetCenter();
        Float3 const& centerf = StaticCast<Float3>(center);
        childSpheres.centers[0][c] = centerf[0]; // X
        childSpheres.centers[1][c] = centerf[1]; // Y
        childSpheres.centers[2][c] = centerf[2]; // Z
        real radius = sphere.GetRadius();

        // Conservatively expand the radius to account for double-to-float rounding error on both
        // the center position and the radius itself.
        if constexpr (MOCHI_USE_DOUBLE_PRECISION) {
          Real3 const centerError = Abs(StaticCast<Real3>(centerf) - center);
          for (int axis = 0; axis < 3; ++axis) {
            if (centerError[axis] > 0_r) {
              radius =
                  std::nextafter(radius + centerError[axis], std::numeric_limits<real>::infinity());
            }
          }
        }
        MOCHI_ASSERT_VERBOSE(
            radius >= 0_r && radius <= kMaxFloat,
            "SphereTree child-sphere radii must be finite and within the finite float range.");
        auto radiusf = static_cast<float>(radius);
        if constexpr (MOCHI_USE_DOUBLE_PRECISION) {
          if (static_cast<real>(radiusf) < radius) {
            radiusf = std::nextafter(radiusf, std::numeric_limits<float>::infinity());
          }
        }
        childSpheres.radii[c] = radiusf;
      } else {
        // This child is empty. In theory the sphere data could be anything, including all zeros.
        // In practice, however, it is better to keep all of the child spheres near each other
        // spatially. This reduces memory access in cases like GridSdf sampling.
        MOCHI_ASSERT(c > 0, "The first child should never be empty.");
        childSpheres.centers[0][c] = childSpheres.centers[0][0]; // X
        childSpheres.centers[1][c] = childSpheres.centers[1][0]; // Y
        childSpheres.centers[2][c] = childSpheres.centers[2][0]; // Z
        childSpheres.radii[c] = 0_r;
      }

      // Add child nodes or child leaves based on their sizes.
      if (cCount == 0) {
        // Empty leaf node
        childIndices[c] = 0;
      } else if (cCount <= maxPerLeaf) {
        // New leaf node
        childIndices[c] = -numLeaves;
        tree._leaves.push_back(Int2{cBegin, cEnd});
        ++numLeaves;
      } else {
        // New non-leaf node
        childIndices[c] = numNodes;
        tree._spheres.push_back();
        tree._children.push_back();
        workQueue.push_back({numNodes, cBegin, cEnd});
        ++numNodes;
      }
    }

    // Copy node data to the tree (in SIMD form)
    tree._spheres[item.nodeIndex].centers[0] = Load<VF>(&childSpheres.centers[0][0]); // X
    tree._spheres[item.nodeIndex].centers[1] = Load<VF>(&childSpheres.centers[1][0]); // Y
    tree._spheres[item.nodeIndex].centers[2] = Load<VF>(&childSpheres.centers[2][0]); // Z
    tree._spheres[item.nodeIndex].radii = Load<VF>(&childSpheres.radii[0]);
    tree._children[item.nodeIndex] = Load<VI>(&childIndices[0]);
  }

  // Initialize values past the end of tree._indices so that FindIntersectingSamplesFn can copy them
  // using full size SIMD operations without UB. Any such values will later be overwritten or
  // discarded. The actual values don't matter as long as they are not undefined.
  tree._indices.resize(tree._indices.size() + Simd<int>::kSize, 0);
  tree._indices.resize(tree._indices.size() - Simd<int>::kSize);

  return tree;
}

template <int kNodeSize>
void SphereTree<kNodeSize>::AssertNodeIsValid(
    [[maybe_unused]] Span<Real3 const> points,
    [[maybe_unused]] int iNode,
    [[maybe_unused]] DynamicArray<Sphere>& parentBounds) const {
#if MOCHI_ASSERT_ENABLED
  if (iNode >= 0) {
    VR3 const centers = {
        StaticCast<VR>(_spheres[iNode].centers[0]),
        StaticCast<VR>(_spheres[iNode].centers[1]),
        StaticCast<VR>(_spheres[iNode].centers[2])};
    VR const radii = StaticCast<VR>(_spheres[iNode].radii);

    // A positive child index maps to a node
    int numNonEmptyChildren = 0;
    for (int c = 0; c < kNodeSize; ++c) {
      int iChild = _children[iNode][c];
      Real3 const center = {centers[0][c], centers[1][c], centers[2][c]}; // X, Y, Z
      real const radius = radii[c];
      auto sphere = Sphere{center, radius};
      if (iChild != 0) { // If not the empty leaf
        ++numNonEmptyChildren;
        parentBounds.push_back(sphere);
        AssertNodeIsValid(points, iChild, parentBounds);
        parentBounds.pop_back();
      }
    }
    int const minNumNonEmptyChildren = iNode == 0 ? Min(2, isize(points)) : 2;
    MOCHI_ASSERT(
        numNonEmptyChildren >= minNumNonEmptyChildren,
        "A valid node should always have at least 2 non-empty children (except the root of a tree with less than 2 points)");
  } else {
    // A negative child index maps to a non-empty leaf
    int iLeaf = -iNode;
    int leafBegin = _leaves[iLeaf][0];
    int leafEnd = _leaves[iLeaf][1];
    MOCHI_ASSERT(leafEnd > leafBegin, "Expected a non-empty leaf node.");
    MOCHI_ASSERT(
        (leafEnd - leafBegin) <= _maxPerLeaf,
        "Leaf contains too many indices. It should have been split further.");
    MOCHI_ASSERT(
        std::is_sorted(_indices.begin() + leafBegin, _indices.begin() + leafEnd),
        "Indices within a leaf node should be sorted in ascending order.");
    for (int i = leafBegin; i < leafEnd; ++i) {
      int iPoint = _indices[i];
      Real3 const& pt = points[iPoint];
      for (auto const& bounds : parentBounds) {
        MOCHI_ASSERT(
            ContainsPoint(bounds, pt),
            "A point in a leaf node must be contained within bounds of all of its parents, up to the root");
      }
    }
  }
#endif // MOCHI_ASSERT_ENABLED
}

template <int kNodeSize>
void SphereTree<kNodeSize>::AssertTreeIsValid([[maybe_unused]] Span<Real3 const> points) const {
#if MOCHI_ASSERT_ENABLED
  int const numNodes = isize(_spheres);
  int const numLeaves = isize(_leaves);
  int const numIndices = isize(_indices);
  int const numPoints = isize(points);

  // Check array sizes
  MOCHI_ASSERT(numNodes >= 1, "Tree must have at least one node");
  MOCHI_ASSERT(numLeaves >= 1, "Tree must have at least one leaf node (the empty leaf)");
  MOCHI_ASSERT(_leaves[0][0] == _leaves[0][1], "The first leaf node should always be empty");
  MOCHI_ASSERT(numNodes == isize(_children), "Size mismatch");
  MOCHI_ASSERT(numIndices == numPoints, "Size mismatch");
  MOCHI_ASSERT(numIndices == GetNumSamples(), "Size mismatch");

  if (numPoints > 0) {
    // Check point indices
    {
      auto [min, max] = MinMax(MakeConstSpan(_indices));
      MOCHI_ASSERT((min == 0) && max == (numPoints - 1), "Invalid index range");
      DynamicArray<int> sortedIndices = _indices;
      std::ranges::sort(sortedIndices);
      for (int i = 0; i < numPoints; ++i) {
        MOCHI_ASSERT(sortedIndices[i] == i, "Indices should be unique");
      }
    }

    // Check child indices
    {
      VI vmin, vmax;
      vmin = vmax = _children[0];
      for (int iNode = 1; iNode < numNodes; ++iNode) {
        vmin = Min(vmin, _children[iNode]);
        vmax = Max(vmax, _children[iNode]);
      }
      int min = HMin(vmin);
      int max = HMax(vmax);
      MOCHI_ASSERT(max < numNodes, "Node index out-of-range");
      MOCHI_ASSERT(-min < numLeaves, "Leaf node index out-of-range");
    }
  } else {
    MOCHI_ASSERT(numNodes == 1, "An empty tree should still have one node");
    MOCHI_ASSERT(_children[0] == VI{}, "All children of an empty node should have index 0");
  }

  // Check partitioning recursively
  DynamicArray<Sphere> parentBounds;
  parentBounds.reserve(32);
  parentBounds.push_back(_bounds);
  AssertNodeIsValid(points, 0, parentBounds);
#endif // MOCHI_ASSERT_ENABLED
}

// Explicit instantiation
namespace mochi {
template class SphereTree<8>;
} // namespace mochi
