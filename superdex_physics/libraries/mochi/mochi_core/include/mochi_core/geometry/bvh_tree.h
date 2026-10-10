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
#include <mochi_core/geometry/any_shape.h>
#include <mochi_core/geometry/obb.h>
#include <mochi_core/geometry/sphere.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/concepts.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/interval.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>

#include <functional>
#include <memory>
#include <variant>
#include <vector>

namespace mochi {

/*************************************************************************************************/

// Forward declarations.
class TriangularMesh;
class TetrahedralMesh;

/*************************************************************************************************/

/**
 * Provider structure feeding elemental information to a BVH tree.
 */
template <typename Bv>
struct BvhObject {
  virtual ~BvhObject() = default;

  /**
   * Retrieves the number of elements found in the adapter. At this point, this number is
   * expected to remain constant throughout the lifetime of a BvhTree.
   */
  virtual int GetNumElements() const = 0;

  /**
   * Retrieves the bounding volume of the [index]-th element.
   */
  virtual Bv GetBv(int index) const = 0;

  /**
   * Retrieves the distance to the [index]-th element.
   */
  virtual real GetDistanceSqr(Real3 const& point, int index) const = 0;
  virtual Vec4r VGetDistanceSqr(Vec4r point, int index) const = 0;
};

/**
 * Callback function invoked when a potential overlap with the input bounding volume is detected.
 * Provides the index of the element potentially overlapping.
 */
using BvhQueryFn = std::function<void(int index)>;

/**
 * Callback function invoked when a potential overlap with another element of the [other] tree is
 * detected. Provides the pair of indices of the elements in overlap.
 */
using BvhIntersectionQueryFn = std::function<void(int thisIndex, int otherIndex)>;

/**
 * Splitting algorithm to employ for the BVH Tree construction.
 */
enum struct BvhSplittingAlgorithm {
  // The tree is constructed top-down, splitting entries according to the area-weighted centroid.
  TopDown_AreaWeightedMean = 0,
  // The tree is constructed top-down, splitting entries according just to the centroid.
  // BVHs of points cannot use area weighting.
  TopDown_Mean = 1,
  // The number of unique enum values.
  Count = 2,
};

/**
 * BVH Tree construction parameters.
 */
struct BvhTreeParams {
  // Splitting algorithm for tree construction.
  BvhSplittingAlgorithm splittingAlgorithm = BvhSplittingAlgorithm::TopDown_AreaWeightedMean;
  // Maximum number of elements per leaf.
  size_t maxElementsPerLeaf = 8;
  // Maximum depth a branch can achieve.
  int maxDepthPerBranch = 20;
  // Refitting parallel granularity.
  int refitGranularity = 32;
};

/**
 * Bounding volume hierarchy tree structure, used as acceleration structure to determine potential
 * contacts between pairs of objects. Note that BvhTree assumes that the underlying object that was
 * used to create it is quasi-static (i.e. it does not suffer exceedingly large deformations) and
 * that its topology remains constant over time.
 */
template <typename Bv>
class BvhTree {
 public:
  static int constexpr kInvalidIndex = -1;
  static int constexpr kRootNode = 0;

  BvhTree(BvhObject<Bv> const* object, BvhTreeParams const& params);

  struct Node {
    // If this tree node a leaf?
    bool isLeafNode = false;
    // Index of the parent node
    int parentIndex = kInvalidIndex;
    // Index of the left child node. Valid only if isLeafNode == true.
    int leftChildIndex = kInvalidIndex;
    // Index of the right child node. Valid only if isLeafNode == true.
    int rightChildIndex = kInvalidIndex;
    // Element index range.
    Interval<int> elementIndexRange{kInvalidIndex, kInvalidIndex};
    // Bounding volume.
    Bv bv{};
  };

  /**
   * Returns true if the BVH tree has been constructed.
   */
  bool IsValid() const;

  /**
   * Returns the pointer to the object used to construct this BVH tree.
   */
  BvhObject<Bv> const* GetObject() const;

  /**
   * Retrieves the parameters used for constructing the BVH tree.
   */
  BvhTreeParams const& GetParams() const;

  /**
   * Returns the bounding volume of the root.
   */
  Bv const& GetRootBv() const {
    MOCHI_ASSERT_VERBOSE(IsValid(), "Invalid BVH Tree");
    return _nodes[kRootNode].bv;
  }

  /**
   * Refits the nodes of the tree to reflect the deformations the object underwent.
   */
  void Refit();

  /**
   * Queries all elements potentially intersecting the given bounding volume [bv]. [callback] will
   * be invoked for each potential overlap detected.
   */
  void Query(Bv const& bv, BvhQueryFn const& callback) const;

  /**
   * Queries all potential intersections with the [other] BVH tree. [callback] will be invoked for
   * each pair of elements potentially overlapping. [other] must be a different tree
   * (self-intersection queries are not supported).
   */
  void Intersect(BvhTree const& other, BvhIntersectionQueryFn const& callback) const;

  /**
   * Queries the index of the closest element to the given input point. Optionally outputs the
   * squared distance from the input point to the closest element in [outDistanceSqr].
   */
  int FindClosest(Real3 const& queryPoint, real* outDistanceSqr = nullptr) const;
  int VFindClosest(Vec4r queryPoint, real* outDistanceSqr = nullptr) const;

  /**
   * Get node by index.
   */
  Node const& GetNode(int index) const {
    return _nodes[index];
  }

  Node& GetNode(int index) {
    return _nodes[index];
  }

  /**
   * Get the number of nodes in the tree.
   */
  size_t GetNodeCount() const;

  /**
   * Get the elements associated with this leaf node.
   */
  RangeByIterators<std::vector<int>::const_iterator> GetElements(Node const& node) const {
    return _elements.begin() + node.elementIndexRange;
  }

  /**
   * Compute the Bv of a certain node.
   */
  Bv ComputeNodeBv(int index) const;

 private:
  int BuildNode(int parentIndex, Span<int> elementIndicesSpan, int depth);
  void RefitNode(int nodeIndex);
  void QueryNode(Bv const& bv, int nodeIndex, BvhQueryFn const& callback) const;
  void IntersectNodes(
      BvhTree const& other,
      int thisIndex,
      int otherIndex,
      BvhIntersectionQueryFn const& callback) const;
  void FindClosestOnNode(int nodeIndex, Vec4r point, int& bestIndex, real& bestDistanceSqr) const;
  void ShrinkBuffers();

  template <bool kSkipElementBvCheck, typename BvOther>
  MOCHI_FORCE_INLINE void
  EmitLeafElements(Node const& node, BvOther const& bv, DynamicArray<int>& outElements) const {
    auto elemRange = _elements.begin() + node.elementIndexRange;
    for (auto&& e : elemRange) {
      if constexpr (kSkipElementBvCheck) {
        outElements.push_back(e);
      } else {
        if (HasOverlap(bv, _object->GetBv(e))) {
          outElements.push_back(e);
        }
      }
    }
  }

 private:
  // Underlying object used to construct the BVH tree.
  BvhObject<Bv> const* _object = nullptr;
  // Parameters used for the tree's construction.
  BvhTreeParams _params = {};
  // Flat vector of tree nodes. The first element is the root.
  std::vector<Node> _nodes;
  // Flat vector of elements indices on each leaf. The ordering employed guarantees that nodes in
  // the same hierarchical level span contiguous regions in this vector.
  std::vector<int> _elements;
  // Indices of the leaf nodes.
  std::vector<int> _leaves;
  // Indices of the nodes (inner or leaf) per level.
  std::vector<std::vector<int>> _levels;
};

extern template class BvhTree<Aabb>;
extern template class BvhTree<Sphere>;

using AabbTree = BvhTree<Aabb>;

/*************************************************************************************************/

/**
 * Defines a basic BVH provider for triangular meshes.
 */
template <typename Bv>
class TriangularMeshBvhObject : public BvhObject<Bv> {
 public:
  TriangularMeshBvhObject() = default;
  explicit TriangularMeshBvhObject(
      std::shared_ptr<TriangularMesh const> const& mesh,
      Span<Real3 const> coordinates = {});

 public:
  void Initialize(
      std::shared_ptr<TriangularMesh const> const& mesh,
      Span<Real3 const> coordinates = {});
  bool IsInitialized() const;

 public:
  int GetNumElements() const override;
  Bv GetBv(int index) const override;
  real GetDistanceSqr(Real3 const& point, int index) const override;
  Vec4r VGetDistanceSqr(Vec4r point, int index) const override;

 private:
  std::shared_ptr<TriangularMesh const> _mesh;
  Span<Real3 const> _coordinates = {};
};

extern template class TriangularMeshBvhObject<Aabb>;

/**
 * Defines a basic BVH provider for tetrahedral meshes.
 */
template <typename Bv>
class TetrahedralMeshBvhObject : public BvhObject<Bv> {
 public:
  TetrahedralMeshBvhObject() = default;
  explicit TetrahedralMeshBvhObject(
      std::shared_ptr<TetrahedralMesh const> const& mesh,
      Span<Real3 const> coordinates = {});

 public:
  void Initialize(
      std::shared_ptr<TetrahedralMesh const> const& mesh,
      Span<Real3 const> coordinates = {});
  bool IsInitialized() const;

 public:
  int GetNumElements() const override;
  Bv GetBv(int index) const override;
  real GetDistanceSqr(Real3 const& point, int index) const override;
  Vec4r VGetDistanceSqr(Vec4r point, int index) const override;

 private:
  std::shared_ptr<TetrahedralMesh const> _mesh;
  Span<Real3 const> _coordinates = {};
};

extern template class TetrahedralMeshBvhObject<Aabb>;

using TetrahedralMeshAabbObject = TetrahedralMeshBvhObject<Aabb>;

/**
 * Defines a basic BVH provider for point sets.
 */
template <typename Bv>
class PointSetBvhObject : public BvhObject<Bv> {
 public:
  PointSetBvhObject() = default;
  PointSetBvhObject(Span<Real3 const> points) : _points(points) {}

 public:
  int GetNumElements() const override {
    return isize(_points);
  }
  Bv GetBv(int index) const override;
  real GetDistanceSqr(Real3 const& point, int index) const override;
  Vec4r VGetDistanceSqr(Vec4r point, int index) const override;

 private:
  Span<Real3 const> _points = {};
};

extern template class PointSetBvhObject<Aabb>;
extern template class PointSetBvhObject<Sphere>;

using PointSetAabbObject = PointSetBvhObject<Aabb>;

/*************************************************************************************************/

} // namespace mochi
