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

#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/assembly_helper.h>
#include <mochi_core/linear_algebra/multi_frontal/elim_tree.h>
#include <mochi_core/linear_algebra/multi_frontal/factor_subtree.h>
#include <mochi_core/linear_algebra/multi_frontal/l_matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/organizer.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/graph_utils.h>
#include <mochi_core/utils/graph_views.h>

#include <memory>
#include <numeric>
#include <type_traits>

namespace mochi::krylov {

/// @brief LDL^T factorization of sparse and block sparse symmetric matrices using multi-frontal
/// algorithm.
///
/// @tparam Scalar_ The scalar type. May be const-qualified.
/// @tparam kBlockSize_ Block size of input matrix. Must be 1 for @ref SparseMatrix inputs.
/// @tparam Index The index type for sparse matrix indices (default: int).
///
/// @note Only supported for sparse and block sparse matrices.
/// @note The matrix is assumed to be symmetric. Only the lower triangular part is used.
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
  /// @param[in] A Input block sparse matrix (symmetric, only lower triangular part used).
  /// @param[out] info Output status flag: 0 = success, non-zero = failure.
  ///
  /// @warning The multi-frontal solver does not currently detect factorization failure (e.g., zero
  /// pivots). @p info is always set to 0. Do not rely on @p info to detect singular matrices.
  template <
      typename ScalarA,
      typename CRIdx,
      typename Ptr,
      template <typename, typename...> typename Storage>
  SparseLDLt(BlockSparseMatrix<ScalarA, kBlockSize, CRIdx, Ptr, Storage> const& A, int& info);

  /// @brief Constructor for sparse matrices. It performs the factorization.
  ///
  /// @param[in] A Input sparse matrix (symmetric, only lower triangular part used).
  /// @param[out] info Output status flag: 0 = success, non-zero = failure.
  ///
  /// @warning The multi-frontal solver does not currently detect factorization failure (e.g., zero
  /// pivots). @p info is always set to 0. Do not rely on @p info to detect singular matrices.
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
  /// @param[out] info Output status flag: 0 = success, non-zero = failure.
  ///
  /// @warning The multi-frontal solver does not currently detect factorization failure (e.g., zero
  /// pivots). @p info is always set to 0. Do not rely on @p info to detect singular matrices.
  ///
  /// @note The sparsity pattern must match the original matrix used in construction.
  template <
      typename ScalarA,
      typename CRIdx,
      typename Ptr,
      template <typename, typename...> typename Storage>
  void Refactorize(BlockSparseMatrix<ScalarA, kBlockSize, CRIdx, Ptr, Storage> const& A, int& info);

  /// @brief Refactorize with new values but same sparsity pattern.
  ///
  /// @param[in] A Input sparse matrix with same sparsity pattern as the original.
  /// @param[out] info Output status flag: 0 = success, non-zero = failure.
  ///
  /// @warning The multi-frontal solver does not currently detect factorization failure (e.g., zero
  /// pivots). @p info is always set to 0. Do not rely on @p info to detect singular matrices.
  ///
  /// @note Only available when kBlockSize == 1.
  /// @note The sparsity pattern must match the original matrix used in construction.
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
  ///
  /// @note The sparsity pattern must match the original matrix used in construction.
  template <
      typename ScalarA,
      typename CRIdx,
      typename Ptr,
      template <typename, typename...> typename Storage>
  void Factorize(BlockSparseMatrix<ScalarA, kBlockSize, CRIdx, Ptr, Storage> const& A);

  /// @brief Matrix size (number of scalar rows/cols).
  Index _size = 0;

#if MOCHI_ASSERT_VERBOSE_ENABLED
  DynamicArray<int> _debugPointers;
  DynamicArray<int> _debugIndices;
#endif

  // Multi-frontal state.
  std::unique_ptr<SymbolicEliminationTree> _elimData;
  std::unique_ptr<FrontalOrganizer> _organizer;
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
  Factorize(A);

  // TODO: Multi-frontal doesn't currently check for factorization failure.
  info = 0;
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
  Factorize(blockView);

  // TODO: Multi-frontal doesn't currently check for factorization failure.
  info = 0;
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

  // Create identity ordering for now.
  // TODO: Implement fill-reducing ordering.
  _order.resize_noinit(A.BlockRows());
  std::iota(_order.begin(), _order.end(), 0);
  _position = ReverseMap(_order);

  // Build elimination tree.
  _elimData = std::make_unique<SymbolicEliminationTree>(graphView, _order, _position);

  // Build frontal organizer.
  _organizer = std::make_unique<FrontalOrganizer>(*_elimData, kLDLtBlockSize, kBlockSize);

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
void SparseLDLt<Scalar_, kBlockSize_, Index>::Factorize(
    BlockSparseMatrix<ScalarA, kBlockSize, CRIdx, Ptr, Storage> const& A) {
  // Zero out L and perform factorization.
  _lMatrix->SetZero();

  // Factor all roots. Handles disconnected sparsity graphs.
  for (int root : _elimData->Roots()) {
    FactorSubtree<kBlockSize, Scalar, kLDLtBlockSize>(
        *_elimData, *_organizer, *_lMatrix, A, *_assemblyHelper, root);
  }
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

  Factorize(A);

  // TODO: Multi-frontal doesn't currently check for factorization failure.
  info = 0;
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
