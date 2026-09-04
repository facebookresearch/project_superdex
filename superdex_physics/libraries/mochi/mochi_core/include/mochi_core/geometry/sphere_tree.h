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

#include <mochi_core/geometry/batch_sphere.h>
#include <mochi_core/geometry/sphere.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace mochi {

/**
 * @brief SIMD-accelerated bounding-sphere hierarchy
 *
 * @note Child-sphere data is stored in single precision even when @ref mochi::real is
 * double-precision, to save memory bandwidth.
 *
 * @tparam kNodeSize Number of children per node (the branching factor).
 */
template <int kNodeSize>
class SphereTree {
 public:
  static_assert(kNodeSize == 8, "SphereTree currently only supports oct-trees (kNodeSize == 8).");
  static_assert(Simd<real, kNodeSize>::kIsSupported);
  static_assert(Simd<int, kNodeSize>::kIsSupported);

  MOCHI_DECLARE_MOVE_ONLY(SphereTree);
  ~SphereTree() = default;

  /**
   * @brief Builds a tree where each leaf node references one or more 3D points by index.
   *
   * @note Other similar functions like FromSpheres or FromTriangles could be added in the future.
   * The tree will store indices for each leaf node. Those indices wouldn't have to refer to points.
   *
   * @param[in] points Points indexed by subsequent queries. Every coordinate must be finite and
   * within the finite range of a single-precision float.
   * @param[in] maxPerLeaf Maximum number of points per leaf. Must be positive.
   * @return The constructed tree.
   */
  static SphereTree FromPoints(Span<Real3 const> points, int maxPerLeaf);

  /**
   * @brief Finds candidate point indices using a custom sphere-overlap test.
   *
   * @param[in] hasOverlap Callable accepting a @ref BatchSphere and returning a per-lane mask.
   * @param[out] outIndices Replaced with indices from matching leaves.
   */
  template <typename HasOverlapFn>
  void FindIntersectingSamplesFn(HasOverlapFn const& hasOverlap, DynamicArray<int>& outIndices)
      const;

  /**
   * @brief Finds candidate point indices whose leaf bounds overlap @p bv.
   *
   * @note This function calls HasOverlap(bv, sphere) without qualification. Define custom overloads
   * in a namespace associated with Bv so argument-dependent lookup can find them, or use @ref
   * FindIntersectingSamplesFn to supply a callable directly.
   *
   * @param[in] bv Bounding volume to query.
   * @param[out] outIndices Replaced with indices from overlapping leaves.
   */
  template <class Bv>
  void FindIntersectingSamples(Bv const& bv, DynamicArray<int>& outIndices) const;

  /**
   * @brief Visits the root and each non-empty child sphere.
   *
   * @param[in] visit Callable invoked as visit(@ref Sphere const&, int depth, bool isLeaf).
   */
  template <typename VisitFn>
  void ForEachNodeSphere(VisitFn const& visit) const;

  /**
   * @brief Asserts that the tree is valid for @p points when assertions are enabled.
   *
   * @param[in] points Points used to construct the tree.
   */
  void AssertTreeIsValid(Span<Real3 const> points) const;

  /**
   * @brief Return the number of samples indexed by this tree.
   */
  int GetNumSamples() const {
    return isize(_indices);
  }

 private:
  SphereTree();
  void AssertNodeIsValid(Span<Real3 const> points, int iNode, DynamicArray<Sphere>& parentBounds)
      const;

  using VR = Simd<real, kNodeSize>;
  using VR3 = NdArray<VR, 3>;
  using VF = Simd<float, kNodeSize>;
  using VF3 = NdArray<VF, 3>;
  using VI = Simd<int, kNodeSize>;

  struct NodeSpheres {
    VF3 centers;
    VF radii;
  };

  // For each node, kNodeSize child spheres are stored in a contiguous packet.
  DynamicArray<NodeSpheres> _spheres;

  // For each node, kNodeSize indices are stored in a contiguous packet.
  // If a child index is positive, it refers to another node.
  // If a child index is zero or negative, then its absolute value is a leaf index.
  DynamicArray<VI> _children;

  // For each leaf node, we store a pair of integers indicating a span within the _indices array.
  DynamicArray<Int2> _leaves;

  // These are indices into the array of samples used to construct the tree (one index per sample).
  // Indices are partitioned such that each leaf node in the tree points to a span of indices. The
  // allocation includes a SIMD-width guard region. Move-only ownership preserves that capacity.
  DynamicArray<int> _indices;

  // Bounding sphere containing all samples
  Sphere _bounds;

  // Maximum number of sample indices for one leaf node. If the number of samples is greater, then
  // the node will be split into kNodeSize children.
  int _maxPerLeaf = 0;
};

/// @brief Eight-way @ref SphereTree specialization.
using SphereOctTree = SphereTree<8>;
extern template class SphereTree<8>;

template <int kNodeSize>
template <typename HasOverlapFn>
void SphereTree<kNodeSize>::FindIntersectingSamplesFn(
    HasOverlapFn const& hasOverlap,
    DynamicArray<int>& outIndices) const {
  outIndices.clear();
  if (_indices.empty()) {
    return;
  }

  // Integer with the same width as real.
  using IR = std::conditional_t<MOCHI_USE_DOUBLE_PRECISION, int64_t, int>;
  using VIR = Simd<IR, kNodeSize>;
  static_assert(sizeof(IR) == sizeof(real));

  // Reuse the output array as the BFS queue. It is resized for the final output after traversal.
  auto& queue = outIndices;
  queue.resize_noinit(_spheres.size() + VI::kSize); // Padding for SIMD writes
  queue[0] = 0;
  size_t queueBegin = 0;
  size_t queueEnd = 1;

  // Use a temporary buffer to store leaf node indices during traversal.
  MOCHI_FILO_STACK_ALLOCATOR(filoAlloc, sizeof(int) * 32 * 1024);
  DynamicArray<int> leavesHit(&filoAlloc);
  leavesHit.resize_noinit(_leaves.size() + VI::kSize); // Padding for SIMD writes
  size_t numLeavesHit = 0;

  while (queueEnd > queueBegin) {
    size_t const prevEnd = queueEnd;
    for (size_t i = queueBegin; i < prevEnd; ++i) {
      int const iNode = queue[i];
      NodeSpheres const& sphere = _spheres[iNode];
      auto const centers = StaticCast<VR3>(sphere.centers);
      auto const radii = StaticCast<VR>(sphere.radii);
      VI const& children = _children[iNode];
      auto const result = hasOverlap(BatchSphere<kNodeSize>{centers, radii});
      VI const hitMask = StaticCast<VI>(ReinterpretCast<VIR>(result));
      VI const nodeHits = hitMask & (children > 0);
      VI const leafHits = hitMask & (children < 0);
      queueEnd += StoreSelected(queue.data() + queueEnd, nodeHits, children);
      numLeavesHit += StoreSelected(leavesHit.data() + numLeavesHit, leafHits, -children);
    }
    queueBegin = prevEnd;
  }

  // Resize once to avoid a memmove and growth check for every leaf. Clearing first avoids copying
  // the now-unused queue if this resize grows the allocation.
  size_t const maxNumOutputIndices =
      Min(_indices.size(), numLeavesHit * static_cast<size_t>(_maxPerLeaf));
  outIndices.clear();
  outIndices.resize_noinit(maxNumOutputIndices + Simd<int>::kSize);

  // SIMD copies may load up to Simd<int>::kSize - 1 lanes past _indices.size(), but remain within
  // the capacity reserved during tree construction. Extra lanes are never counted. A later store
  // overwrites them, or the final resize discards them.
  MOCHI_ASSERT_VERBOSE(
      _indices.capacity() >= _indices.size() + Simd<int>::kSize,
      "SphereTree index storage must retain its SIMD guard region.");
  size_t outIndex = 0;
  if (_maxPerLeaf <= Simd<int>::kSize) {
    for (size_t i = 0; i < numLeavesHit; ++i) {
      Int2 const& leafRange = _leaves[leavesHit[i]];
      int const begin = leafRange[0];
      int const end = leafRange[1];
      auto inds = Load<Simd<int>>(&_indices[begin]);
      Store(&outIndices[outIndex], inds);
      outIndex += (end - begin);
    }
  } else {
    for (size_t i = 0; i < numLeavesHit; ++i) {
      Int2 const& leafRange = _leaves[leavesHit[i]];
      int const begin = leafRange[0];
      int const end = leafRange[1];
      int j = begin;
      for (; j < end; j += Simd<int>::kSize) {
        Store(&outIndices[outIndex], Load<Simd<int>>(&_indices[j]));
        outIndex += Min(Simd<int>::kSize, end - j);
      }
    }
  }

  outIndices.resize_noinit(outIndex);
}

template <int kNodeSize>
template <typename VisitFn>
void SphereTree<kNodeSize>::ForEachNodeSphere(VisitFn const& visit) const {
  int const numNodes = isize(_spheres);
  if (numNodes == 0) {
    return;
  }

  // The root bounding sphere is depth 0.
  // The child spheres stored in node i are at depth nodeDepth[i] + 1.
  visit(_bounds, /*depth*/ 0, /*isLeaf*/ false);

  // Compute the depth of each node. Nodes are created in BFS order, so a parent always has a
  // smaller index than its children. A single forward pass therefore visits each parent before
  // its children, and every node (except the root) is the positive child of exactly one parent.
  DynamicArray<int> nodeDepth;
  nodeDepth.resize_noinit(numNodes);
  nodeDepth[0] = 0;
  for (int iNode = 0; iNode < numNodes; ++iNode) {
    int const childDepth = nodeDepth[iNode] + 1;
    VI const& children = _children[iNode];
    for (int c = 0; c < kNodeSize; ++c) {
      int const child = children[c];
      if (child > 0) {
        // Mark the depth of this child.
        nodeDepth[child] = childDepth;
      }
    }
  }

  // Fire the callback for every non-zero child index (zero is the special empty-leaf slot).
  int numLeaves = 1; // 1 for the empty leaf at index 0
  for (int iNode = 0; iNode < numNodes; ++iNode) {
    int const childDepth = nodeDepth[iNode] + 1;
    NodeSpheres const& sphere = _spheres[iNode];
    auto const centers = StaticCast<VR3>(sphere.centers);
    auto const radii = StaticCast<VR>(sphere.radii);
    VI const& children = _children[iNode];
    for (int c = 0; c < kNodeSize; ++c) {
      int const child = children[c];
      if (child == 0) {
        // Empty child slot (carries a degenerate radius-0 sphere); skip it.
        continue;
      }
      Real3 const center = {centers[0][c], centers[1][c], centers[2][c]};
      real const radius = radii[c];
      bool const isLeaf = child < 0;
      visit(Sphere{center, radius}, childDepth, isLeaf);
      if (isLeaf) {
        ++numLeaves;
      }
    }
  }
  MOCHI_ASSERT(numLeaves == isize(_leaves));
}

template <int kNodeSize>
template <class Bv>
void SphereTree<kNodeSize>::FindIntersectingSamples(Bv const& bv, DynamicArray<int>& outIndices)
    const {
  FindIntersectingSamplesFn([&](auto const& sphere) { return HasOverlap(bv, sphere); }, outIndices);
}

} // namespace mochi
