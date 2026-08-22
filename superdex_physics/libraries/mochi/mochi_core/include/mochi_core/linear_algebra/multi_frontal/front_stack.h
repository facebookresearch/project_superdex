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
#include <mochi_core/async/generator.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/elim_tree.h>
#include <mochi_core/linear_algebra/multi_frontal/front_matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/organizer.h>
#include <mochi_core/utils/dynamic_array.h>

namespace mochi {

/**
 * @brief Manages memory for frontal matrices during multifrontal factorization.
 *
 * @details Frontal matrices are typically arranged in a stack during the elimination tree
 * traversal. This class handles the allocation and management of that stack space, allowing for
 * efficient reuse of memory. It also supports providing an external buffer for the root node's
 * frontal matrix.
 *
 * @tparam Scalar The scalar type (e.g., float, double)
 * @tparam kColumnBlock The column block size for frontal matrices
 * @tparam kPackSmall If true, small matrices are packed differently
 */
template <typename Scalar, size_t kColumnBlock, bool kPackSmall>
class FrontStack {
 public:
  /**
   * @brief Construct a new FrontStack.
   *
   * @param tree The elimination tree
   * @param organizer The organizer for frontal matrix mapping
   * @param root The root of the subtree being factored
   * @param rootFrontSpace Optional pre-allocated space for the root's frontal matrix
   */
  FrontStack(
      EliminationTree const& tree,
      FrontalOrganizer const& organizer,
      int root,
      Span<Scalar> rootFrontSpace = {});

  /**
   * @brief Push a new supernode onto the stack and return its assigned frontal space.
   *
   * @param superNode The supernode to push
   * @return Span<Scalar> The pre-allocated space for the frontal matrix
   */
  Span<Scalar> PushFront(int superNode);

  /**
   * @brief Check if a supernode's frontal matrix is currently at the top of the stack.
   */
  [[nodiscard]] auto IsOnStack(int superNode) const {
    return superNode == _frontData.back().superNode;
  }

  /**
   * @brief Check if the parent of a supernode is currently at the top of the stack.
   */
  [[nodiscard]] bool IsParentOnStack(int superNode) const {
    return _parents[superNode] == _frontData.back().superNode;
  }

  /**
   * @brief Remove a supernode's frontal matrix from the stack.
   *
   * @param superNode The supernode to pop (must be at the top of the stack)
   */
  void PopFront([[maybe_unused]] int superNode) {
    MOCHI_ASSERT_VERBOSE(_frontData.back().superNode == superNode, "Invalid superNode");
    auto& top = _frontData.back();
    // Skip _topOfStack update when popping the root with external space, since its frontal matrix
    // lives in _rootFrontSpace, not the internal stack. Currently FactorSubtree breaks before
    // popping the root, so this branch is defensive — it exists for correctness if future callers
    // pop all nodes.
    if (top.superNode != _firstNode || _rootFrontSpace.empty()) {
      _topOfStack = top.frontSpan.end();
    }
    _frontData.pop_back();
  }

  /**
   * @brief Return the frontal space of the supernode at the top of the stack.
   */
  [[nodiscard]] auto GetTopFront() {
    return _frontData.back().frontSpan;
  }

  /**
   * @brief Return a span covering the entire internal stack space.
   */
  [[nodiscard]] Span<Scalar> FullSpace() {
    return Span(_stackStart.get(), _stackSize);
  }

 private:
  int _firstNode; ///< The root of the subtree
  Span<Scalar> _rootFrontSpace; ///< Pre-allocated space for the root
  size_t _stackSize; ///< Total size of the internal stack
  std::unique_ptr<Scalar[]> _stackStart; ///< Buffer for the internal stack
  Scalar* _topOfStack; ///< Pointer to the current top of the internal stack
  Span<int const> _parents; ///< Parent mapping from the elimination tree
  Span<FrontalOrganizer::Costs const> _costs; ///< Memory requirements for each supernode
  struct FrontData {
    int superNode; ///< The supernode ID
    Span<Scalar> frontSpan; ///< The allocated space for this supernode
  };
  DynamicArray<FrontData> _frontData; ///< Stack of allocated frontal spaces
};

template <typename Scalar, size_t kColumnBlock, bool kPackSmall>
Span<Scalar> FrontStack<Scalar, kColumnBlock, kPackSmall>::PushFront(int superNode) {
  auto frontSize = _costs[superNode].frontSize;

  Scalar* begin = nullptr;
  Scalar* end = nullptr;
  // If this is the root and we have pre-allocated space, use it.
  if (superNode == _firstNode && !_rootFrontSpace.empty()) {
    MOCHI_ASSERT_VERBOSE(
        _rootFrontSpace.size() >= frontSize, "Pre-allocated root space is too small");
    begin = _rootFrontSpace.begin();
    end = begin + frontSize;
  } else {
    // Otherwise, use the internal stack.
    end = _topOfStack;
    begin = end - frontSize;
    _topOfStack = begin;
  }
  _frontData.push_back({superNode, {begin, end}});

  // Verification of pointer ranges.
  MOCHI_ASSERT_VERBOSE(
      (superNode == _firstNode && !_rootFrontSpace.empty()) ||
          (FullSpace().begin() <= begin && begin <= FullSpace().end()),
      "Invalid front begin pointer");
  MOCHI_ASSERT_VERBOSE(
      (superNode == _firstNode && !_rootFrontSpace.empty()) ||
          (FullSpace().begin() <= end && end <= FullSpace().end()),
      "Invalid front end pointer");
  return {begin, end};
}

template <typename Scalar, size_t kColumnBlock, bool kPackSmall>
FrontStack<Scalar, kColumnBlock, kPackSmall>::FrontStack(
    EliminationTree const& tree,
    FrontalOrganizer const& organizer,
    int root,
    Span<Scalar> rootFrontSpace)
    : _firstNode(root),
      _rootFrontSpace(rootFrontSpace),
      _stackSize(organizer.GetStackSize(root)),
      // It is up to organizer.GetStackSize(root) to account for reduced stack
      // usage if the root space is provided. It is usually a minor difference
      // an may not be worth the added complexity.
      _stackStart(new Scalar[_stackSize]),
      _topOfStack(_stackStart.get() + _stackSize),
      _parents(tree.SuperParents()),
      _costs(organizer.GetCosts()) {
  _frontData.reserve(tree.SubtreeDepth(root) + 1);

  // Initial dummy element to simplify topOfStack logic.
  _frontData.emplace_back(FrontData{-1, {}});
}

} // namespace mochi
