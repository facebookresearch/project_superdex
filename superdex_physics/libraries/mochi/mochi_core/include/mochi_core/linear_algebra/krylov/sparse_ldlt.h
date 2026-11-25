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

#include <mochi_core/async/executor.h>
#include <mochi_core/async/root_task.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/assembly_helper.h>
#include <mochi_core/linear_algebra/multi_frontal/elim_tree.h>
#include <mochi_core/linear_algebra/multi_frontal/factor_branch.h>
#include <mochi_core/linear_algebra/multi_frontal/factor_subtree.h>
#include <mochi_core/linear_algebra/multi_frontal/l_matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/organizer.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/numbering/amd.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/graph_views.h>
#include <mochi_core/utils/task_scheduler.h>

#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>

namespace mochi::krylov {

namespace details {

template <typename Exec, int kDofsPerNode, typename Scalar, size_t kBlockSize>
RootTask FactorTree(Exec& dispatch, FactorData<kDofsPerNode, Scalar, kBlockSize>& data, int root) {
  co_await FactorBranch(dispatch, data, root);
}

} // namespace details

/// @brief LDL^T factorization of sparse and block sparse symmetric matrices using multi-frontal
/// algorithm.
///
/// @tparam Scalar_ The scalar type. May be const-qualified.
/// @tparam kBlockSize_ Block size of input matrix. Must be 1 for @ref SparseMatrix inputs.
/// @tparam Index The index type for sparse matrix indices (default: int).
///
/// @note Only supported for sparse and block sparse matrices.
/// @note The matrix must be symmetric, and so must its stored pattern, explicit zeros included.
/// Entries at the same position are summed.
/// @note If a @ref TaskScheduler with at least two workers is available and in multi-threaded mode
/// (both global and thread-local), both at construction and at each factorization, each
/// elimination-tree root with enough work is factorized in parallel, unless its tree is a single
/// supernode or a chain in which no supernode above the leaf gets at least two panel tasks.
/// @note Performance for kBlockSize_ == 1 (e.g. @ref SparseMatrix) has not been optimized.
template <typename Scalar_, int kBlockSize_, typename Index = int>
class SparseLDLt {
  // TODO:
  // - Tune kLDLtBlockSize.
 public:
  using Scalar = std::remove_const_t<Scalar_>;
  static constexpr int kBlockSize = kBlockSize_;

  /// @brief Block size for the multi-frontal L factor storage.
  /// @note This is an internal optimization parameter that affects memory layout, not the
  /// mathematical block size of the input matrix.
  static constexpr auto kLDLtBlockSize = static_cast<size_t>(RoundUp(96, std::lcm(kBlockSize, 24)));
  static_assert(
      kLDLtBlockSize % kBlockSize == 0,
      "kLDLtBlockSize must be a multiple of kBlockSize");

  /// @brief Constructor for block sparse matrices. It performs the factorization.
  ///
  /// @param[in] A Input symmetric block sparse matrix.
  /// @param[out] info Output status flag: 0 = success, otherwise the number of pivots whose inverse
  /// is zero or not finite, e.g. zero, infinite or NaN pivots.
  ///
  /// @warning The factorization does not pivot, so a nonsingular indefinite matrix can have a zero
  /// pivot. No tolerance flags small pivots.
  template <
      typename ScalarA,
      typename CRIdx,
      typename Ptr,
      template <typename, typename...> typename Storage>
  SparseLDLt(BlockSparseMatrix<ScalarA, kBlockSize, CRIdx, Ptr, Storage> const& A, int& info);

  /// @brief Constructor for sparse matrices. It performs the factorization.
  ///
  /// @param[in] A Input symmetric sparse matrix.
  /// @param[out] info Output status flag: 0 = success, otherwise the number of pivots whose inverse
  /// is zero or not finite, e.g. zero, infinite or NaN pivots.
  ///
  /// @warning The factorization does not pivot, so a nonsingular indefinite matrix can have a zero
  /// pivot. No tolerance flags small pivots.
  ///
  /// @note Only available when kBlockSize == 1.
  template <
      typename ScalarA,
      typename CRIdx,
      typename Ptr,
      template <typename, typename...> typename Storage>
    requires(kBlockSize_ == 1)
  SparseLDLt(SparseMatrix<ScalarA, CRIdx, Ptr, Storage> const& A, int& info);

  /// @brief Solve in place: x <- A^{-1} * x
  ///
  /// @param[in,out] x Input/output column vector. On input, the right-hand side. On output, the
  /// solution.
  ///
  /// @note Only supported for column vectors.
  template <
      typename ScalarX,
      int kRowsX,
      int kColsX,
      Direction kDirectionX,
      Ownership kOwnershipX,
      int kLeadDimX>
    requires(kColsX == 1 || kColsX == krylov::kDynamic)
  void LeftSolveInPlace(
      Matrix<ScalarX, kRowsX, kColsX, kDirectionX, kOwnershipX, kLeadDimX>& x) const;

  /// @brief Get the block size.
  [[nodiscard]] static constexpr int BlockSize() {
    return kBlockSize;
  }

  /// @brief Refactorize with new values but same sparsity pattern.
  ///
  /// @param[in] A Input block sparse matrix with same sparsity pattern as the original.
  /// @param[out] info Output status flag: 0 = success, otherwise the number of pivots whose inverse
  /// is zero or not finite, e.g. zero, infinite or NaN pivots.
  ///
  /// @warning The factorization does not pivot, so a nonsingular indefinite matrix can have a zero
  /// pivot. No tolerance flags small pivots.
  ///
  /// @note The row pointers and column indices, repeated indices included, must equal those of the
  /// matrix used in construction.
  template <
      typename ScalarA,
      typename CRIdx,
      typename Ptr,
      template <typename, typename...> typename Storage>
  void Refactorize(BlockSparseMatrix<ScalarA, kBlockSize, CRIdx, Ptr, Storage> const& A, int& info);

  /// @brief Refactorize with new values but same sparsity pattern.
  ///
  /// @param[in] A Input sparse matrix with same sparsity pattern as the original.
  /// @param[out] info Output status flag: 0 = success, otherwise the number of pivots whose inverse
  /// is zero or not finite, e.g. zero, infinite or NaN pivots.
  ///
  /// @warning The factorization does not pivot, so a nonsingular indefinite matrix can have a zero
  /// pivot. No tolerance flags small pivots.
  ///
  /// @note Only available when kBlockSize == 1.
  /// @note The row pointers and column indices, repeated indices included, must equal those of the
  /// matrix used in construction.
  template <
      typename ScalarA,
      typename CRIdx,
      typename Ptr,
      template <typename, typename...> typename Storage>
    requires(kBlockSize_ == 1)
  void Refactorize(SparseMatrix<ScalarA, CRIdx, Ptr, Storage> const& A, int& info) {
    Refactorize(AsBlockSparseMatrixConstView(A), info);
  }

 private:
  // Picks of {root} factorize the root directly. Otherwise, the picks run as branch tasks and trunk
  // supernodes split their panels into at most numThreads tasks (see
  // FrontalOrganizer::NumPanelTasks).
  struct RootFactorizationPlan {
    int root = 0;
    int numThreads = 0;
    DynamicArray<int> pickedBranches;
  };

  // Scalar-equivalent work below which task scheduling does not amortize its overhead.
  static constexpr double kMinParallelTaskWork = 1'000'000.0;

  // kMinParallelTaskWork in the organizer's time units.
  static constexpr double kMinTaskTime =
      kMinParallelTaskWork / (kBlockSize * kBlockSize * kBlockSize);

  /// @brief Initialize the multi-frontal structures.
  template <
      typename ScalarA,
      typename CRIdx,
      typename Ptr,
      template <typename, typename...> typename Storage>
  void Initialize(BlockSparseMatrix<ScalarA, kBlockSize, CRIdx, Ptr, Storage> const& A);

  /// @brief Factorize with new values but same sparsity pattern.
  ///
  /// @param[in] A Input block sparse matrix with same sparsity pattern as the original.
  /// @return The number of pivots whose inverse is zero or not finite.
  ///
  /// @note The row pointers and column indices, repeated indices included, must equal those of the
  /// matrix used in construction.
  template <
      typename ScalarA,
      typename CRIdx,
      typename Ptr,
      template <typename, typename...> typename Storage>
  [[nodiscard]] int Factorize(BlockSparseMatrix<ScalarA, kBlockSize, CRIdx, Ptr, Storage> const& A);

  /// @brief Matrix size (number of scalar rows/cols).
  Index _size = 0;

#if MOCHI_ASSERT_VERBOSE_ENABLED
  DynamicArray<int> _debugPointers;
  DynamicArray<int> _debugIndices;
#endif

  // Multi-frontal state.
  std::unique_ptr<SymbolicEliminationTree> _elimData;
  std::unique_ptr<FrontalOrganizer> _organizer;
  DynamicArray<RootFactorizationPlan> _rootFactorizationPlans;
  std::unique_ptr<LMatrix<Scalar, kLDLtBlockSize>> _lMatrix;
  std::unique_ptr<AssemblyHelper> _assemblyHelper;
  DynamicArray<int> _order;
  DynamicArray<int> _position;
};

// ============================================================================
// Template Implementation
// ============================================================================

template <typename Scalar_, int kBlockSize_, typename Index>
template <
    typename ScalarA,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
SparseLDLt<Scalar_, kBlockSize_, Index>::SparseLDLt(
    BlockSparseMatrix<ScalarA, kBlockSize, CRIdx, Ptr, Storage> const& A,
    int& info)
    : _size(A.Rows()) {
  static_assert(
      std::is_same_v<ScalarA const, Scalar const>,
      "Matrix scalar type must match SparseLDLt scalar type.");
  static_assert(
      std::is_same_v<CRIdx const, Index const>,
      "Matrix index type must match SparseLDLt index type.");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");

  Initialize(A);
  info = Factorize(A);
}

template <typename Scalar_, int kBlockSize_, typename Index>
template <
    typename ScalarA,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
  requires(kBlockSize_ == 1)
SparseLDLt<Scalar_, kBlockSize_, Index>::SparseLDLt(
    SparseMatrix<ScalarA, CRIdx, Ptr, Storage> const& A,
    int& info)
    : _size(A.Rows()) {
  static_assert(
      std::is_same_v<ScalarA const, Scalar const>,
      "Matrix scalar type must match SparseLDLt scalar type.");
  static_assert(
      std::is_same_v<CRIdx const, Index const>,
      "Matrix index type must match SparseLDLt index type.");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");

  auto blockView = AsBlockSparseMatrixConstView(A);
  Initialize(blockView);
  info = Factorize(blockView);
}

template <typename Scalar_, int kBlockSize_, typename Index>
template <
    typename ScalarA,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void SparseLDLt<Scalar_, kBlockSize_, Index>::Initialize(
    BlockSparseMatrix<ScalarA, kBlockSize, CRIdx, Ptr, Storage> const& A) {
  // Get the graph from the matrix.
  auto graphView = AsGraphView(A);

  // Use AMD reordering. AMD is faster end to end than nested dissection for problems with more than
  // ~500 DoFs. Nested dissection's factorization is occasionally faster but does not offset its
  // substantially higher ordering cost.
  _order = RenumberByAMD(graphView);
  _position = InversePermutation(_order);

  // Build elimination tree.
  _elimData = std::make_unique<SymbolicEliminationTree>(graphView, _order, _position);

  // Build frontal organizer.
  _organizer = std::make_unique<FrontalOrganizer>(*_elimData, kLDLtBlockSize, kBlockSize);

  // Precompute root plans using the thread budget available during initialization. Only workers
  // run the tasks of planned roots, so plans need at least two workers. A solver initialized
  // without them remains sequential if workers are enabled later. Plans computed with workers
  // remain valid if single-threaded mode is enabled later.
  auto* scheduler = TaskScheduler::TryGet();
  if (scheduler && scheduler->GetNumThreads() > 1) {
    auto const& eTree = *_elimData;
    auto const& organizer = *_organizer;
    int const numWorkers = Max(1, scheduler->GetNumThreads());
    double constexpr kMinRootTime = FrontalOrganizer::kBranchesPerThread * kMinTaskTime;
    auto const roots = eTree.Roots();
    _rootFactorizationPlans.reserve(roots.size());
    for (int root : roots) {
      double const rootTime = organizer.GetCosts(root).time;
      if (rootTime < kMinRootTime) {
        _rootFactorizationPlans.push_back(
            {.root = root, .numThreads = 1, .pickedBranches = {root}});
        continue;
      }

      auto const numThreads = static_cast<int>(
          Clamp(Floor(rootTime / kMinRootTime), 1.0, static_cast<double>(numWorkers)));
      _rootFactorizationPlans.push_back(
          {.root = root,
           .numThreads = numThreads,
           .pickedBranches = organizer.PickBranches(eTree, root, numThreads, kMinTaskTime).first});
    }
  }

  // Build L matrix storage.
  _lMatrix = std::make_unique<LMatrix<Scalar, kLDLtBlockSize>>(*_elimData, kBlockSize);

  // Build assembly helper.
  _assemblyHelper = std::make_unique<AssemblyHelper>(
      graphView, _elimData->SuperIndices(), _elimData->SuperBounds(), _order, _position);

#if MOCHI_ASSERT_VERBOSE_ENABLED
  _debugPointers.assign(A.Pointers().begin(), A.Pointers().end());
  _debugIndices.assign(A.Indices().begin(), A.Indices().end());
#endif
}

template <typename Scalar_, int kBlockSize_, typename Index>
template <
    typename ScalarA,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
int SparseLDLt<Scalar_, kBlockSize_, Index>::Factorize(
    BlockSparseMatrix<ScalarA, kBlockSize, CRIdx, Ptr, Storage> const& A) {
  // Zero out L and perform factorization.
  _lMatrix->SetZero();

  auto* scheduler = TaskScheduler::TryGet();
  if (_rootFactorizationPlans.empty() || !scheduler || scheduler->GetNumOtherThreads() == 0) {
    // Factor all roots. Handles disconnected sparsity graphs.
    for (int root : _elimData->Roots()) {
      FactorSubtree<kBlockSize, Scalar, kLDLtBlockSize>(
          *_elimData, *_organizer, *_lMatrix, A, *_assemblyHelper, root);
    }
  } else {
    MOCHI_ASSERT_VERBOSE(
        _rootFactorizationPlans.size() == _elimData->Roots().size(),
        "Factorization plans must match elimination-tree roots.");

    Executor executor(*scheduler);
    auto dispatch = [executor]<typename F>(F&& f) { executor.schedule(std::forward<F>(f)); };

    // Concurrent roots may improve utilization, but increase peak frontal-buffer memory and cache
    // contention. Process roots sequentially for now. TODO: benchmark this trade-off.
    for (auto const& plan : _rootFactorizationPlans) {
      // A single proper subtree still leaves a trunk to factor; only {root} covers the whole tree.
      if (plan.pickedBranches.size() == 1 && plan.pickedBranches[0] == plan.root) {
        FactorSubtree<kBlockSize, Scalar, kLDLtBlockSize>(
            *_elimData, *_organizer, *_lMatrix, A, *_assemblyHelper, plan.root);
        continue;
      }

      FactorData<kBlockSize, Scalar, kLDLtBlockSize> data(
          *_elimData,
          *_organizer,
          *_assemblyHelper,
          A,
          *_lMatrix,
          plan.pickedBranches,
          plan.numThreads,
          kMinTaskTime);
      details::FactorTree(dispatch, data, plan.root).RunAndWait(*scheduler, "Factor tree");
    }
  }

  // The diagonal of L stores inverse pivots: zero and NaN pivots leave non-finite values, and
  // infinite pivots leave zeros.
  int numFailedPivots = 0;
  for (int sn = 0; sn < _elimData->NumSuperNodes(); ++sn) {
    auto const lForSN = _lMatrix->LforSN(sn);
    for (size_t iBlock = 0; iBlock < lForSN.NumBlocks(); ++iBlock) {
      auto const block = lForSN.Block(iBlock);
      for (int c = 0; c < block.Cols(); ++c) {
        Scalar const inversePivot = block(c, c);
        numFailedPivots += (IsFinite(inversePivot) && inversePivot != Scalar{0}) ? 0 : 1;
      }
    }
  }
  return numFailedPivots;
}

template <typename Scalar_, int kBlockSize_, typename Index>
template <
    typename ScalarA,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void SparseLDLt<Scalar_, kBlockSize_, Index>::Refactorize(
    BlockSparseMatrix<ScalarA, kBlockSize, CRIdx, Ptr, Storage> const& A,
    int& info) {
  static_assert(
      std::is_same_v<ScalarA const, Scalar const>,
      "Matrix scalar type must match SparseLDLt scalar type.");
  static_assert(
      std::is_same_v<CRIdx const, Index const>,
      "Matrix index type must match SparseLDLt index type.");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Matrix must be square.");
  MOCHI_ASSERT_VERBOSE(A.Rows() == _size, "Matrix size must match original.");

#if MOCHI_ASSERT_VERBOSE_ENABLED
  MOCHI_ASSERT_VERBOSE(
      A.Pointers() == MakeConstSpan(_debugPointers),
      "Sparsity pattern pointers do not match the original.");
  MOCHI_ASSERT_VERBOSE(
      A.Indices() == MakeConstSpan(_debugIndices),
      "Sparsity pattern indices do not match the original.");
#endif

  info = Factorize(A);
}

template <typename Scalar_, int kBlockSize_, typename Index>
template <
    typename ScalarX,
    int kRowsX,
    int kColsX,
    Direction kDirectionX,
    Ownership kOwnershipX,
    int kLeadDimX>
  requires(kColsX == 1 || kColsX == krylov::kDynamic)
void SparseLDLt<Scalar_, kBlockSize_, Index>::LeftSolveInPlace(
    Matrix<ScalarX, kRowsX, kColsX, kDirectionX, kOwnershipX, kLeadDimX>& x) const {
  static_assert(
      std::is_same_v<std::remove_const_t<ScalarX>, Scalar>,
      "Right-hand side scalar type must match SparseLDLt scalar type.");
  static_assert(!krylov::IsCuda(kOwnershipX), "SparseLDLt not supported on CUDA");
  MOCHI_ASSERT_VERBOSE(x.Rows() == _size, "Vector size must match matrix size.");
  MOCHI_ASSERT_VERBOSE(
      x.Cols() == 1, "SparseLDLt::LeftSolveInPlace currently only supports column vectors.");

  MultiFrontalSolveInPlace<kBlockSize, Scalar, kLDLtBlockSize>(
      *_elimData, *_organizer, *_lMatrix, MakeConstSpan(_order), AsView(x));
}

// ============================================================================
// Deduction Guides
// ============================================================================

/// @brief Deduction guide for SparseMatrix input.
template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
SparseLDLt(SparseMatrix<Scalar, CRIdx, Ptr, Storage> const&, int&)
    -> SparseLDLt<std::remove_const_t<Scalar>, 1, std::remove_const_t<CRIdx>>;

/// @brief Deduction guide for BlockSparseMatrix input.
template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
SparseLDLt(BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage> const&, int&)
    -> SparseLDLt<std::remove_const_t<Scalar>, kBlockSize, std::remove_const_t<CRIdx>>;

} // namespace mochi::krylov
