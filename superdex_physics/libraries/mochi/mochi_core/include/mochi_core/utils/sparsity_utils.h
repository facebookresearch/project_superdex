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

#include <mochi_core/linear_algebra/any_matrix.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/graph_views.h>
#include <mochi_core/utils/local_to_global_map.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/span.h>

#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace mochi {

namespace details {

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void ConvertToFillLevel(
    int p,
    BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage> const& A,
    BlockSparseMatrix<
        std::remove_const_t<Scalar>,
        kBlockSize,
        std::remove_const_t<CRIdx>,
        std::remove_const_t<Ptr>>& R) {
  MOCHI_ASSERT_VERBOSE(p >= 0, "Fill-in level must not be negative.");
  if (p == 0) {
    R.Reset(A);
    return;
  }
  //
  // We are using the sparsity pattern of A^{p + 1}.
  //
  using NonConstIdx = std::remove_const_t<CRIdx>;
  using NonConstPtr = std::remove_const_t<Ptr>;
  auto const gA = AsGraphView(A);
  DynamicArray<NonConstPtr> pointers(gA.GetPointers().begin(), gA.GetPointers().end());
  DynamicArray<NonConstIdx> targets(gA.GetTargets().begin(), gA.GetTargets().end());
  Graph<NonConstIdx, NonConstPtr> gAk(std::move(pointers), std::move(targets));
  for (int k = 0; k < p; ++k) {
    gAk = Traverse(gAk, gA).SortTargets();
  }
  R.Reset(A.BlockCols(), gAk);
  R.SetZero();
  R += A;
}

template <
    typename Scalar,
    int kRows,
    int kCols,
    krylov::Direction kDir,
    krylov::Ownership kOwner,
    int kLead>
void ConvertToFillLevel(
    int p,
    Matrix<Scalar, kRows, kCols, kDir, kOwner, kLead> const& A,
    Matrix<std::remove_const_t<Scalar>, kRows, kCols, krylov::Direction::RowMajor>& R) {
  MOCHI_ASSERT_VERBOSE(p >= 0, "Fill-in level must not be negative.");
  MOCHI_ASSERT(p == 0, "Only zero fill-in level is supported for dense matrices.");
  R.Reset(A);
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void ConvertToFillLevel(
    int p,
    SparseMatrix<Scalar, CRIdx, Ptr, Storage> const& A,
    SparseMatrix<std::remove_const_t<Scalar>, std::remove_const_t<CRIdx>, std::remove_const_t<Ptr>>&
        R) {
  MOCHI_ASSERT_VERBOSE(p >= 0, "Fill-in level must not be negative.");
  if (p == 0) {
    R.Reset(A);
    return;
  }
  //
  // We are using the sparsity pattern of A^{p + 1}.
  //
  using NonConstIdx = std::remove_const_t<CRIdx>;
  using NonConstPtr = std::remove_const_t<Ptr>;
  auto const gA = AsGraphView(A);
  DynamicArray<NonConstPtr> pointers(gA.GetPointers().begin(), gA.GetPointers().end());
  DynamicArray<NonConstIdx> targets(gA.GetTargets().begin(), gA.GetTargets().end());
  Graph<NonConstIdx, NonConstPtr> gAk(std::move(pointers), std::move(targets));
  for (int k = 0; k < p; ++k) {
    gAk = Traverse(gAk, gA).SortTargets();
  }
  R.Reset(A.Cols(), gAk);
  R.SetZero();
  R += A;
}

} // namespace details

// Use a Local2GlobalMap to build a sparsity graph using M as auxiliary/scratch storage.
// The method can take an optional number of rows to support empty rows.
Graph<int, int> MakeSparsityGraph(
    Local2GlobalMap const& map,
    std::vector<std::vector<int>>& M,
    std::optional<int> numRowsOpt = {});

// Build a sparsity pattern from an associative collection of non-zero coordinates,
// represented as a vector of vectors: each entry in the vector contains all
// column indices of a given row.
// The method can take an optional number of rows to support empty rows.
Graph<int, int> MakeSparsityGraph(
    std::vector<std::vector<int>>&& M,
    std::optional<int> numRowsOpt = {});

// Use a Local2GlobalMap to build a sparsity graph, which can in turn be used to
// initialize a SparseMatrix. The method can take an optional number of rows to support empty rows.
Graph<int, int> MakeSparsityGraph(Local2GlobalMap const& map, std::optional<int> numRowsOpt = {});

// Build a sparsity pattern from a collection non-zero coordinates, represented as (row, col) pairs.
// The method can take an optional number of rows to support empty rows.
Graph<int, int> MakeSparsityGraph(
    std::vector<Int2> coordinates,
    std::optional<int> numRowsOpt = {});

// Build the sparsity pattern for a matrix where all values can be non-zero.
Graph<int, int> MakeDenseSparsityGraph(int numRows, int numCols);

// Use a Local2GlobalMap to build a square SparseMatrix.
template <typename T = real>
SparseMatrix<T, int, int> MakeSparseMatrix(Local2GlobalMap const& map) {
  Graph<int, int> sparsity = MakeSparsityGraph(map);
  int numCols = sparsity.size(); // Assumes a square matrix
  return SparseMatrix<T, int, int>{numCols, std::move(sparsity)};
}

// Enumerate the non-zero coordinates in an existing sparsity pattern, and append them to an output
// vector. Does NOT clear the output vector.
void AppendNonZeroCoordinates(
    std::vector<Int2>& outCoordinates,
    Graph<int, int> const& sparsity,
    int dofOffset = 0);

// Set the values on the specified rows of a matrix to zero, except for the diagonal, which will be
// set the specified value (usually 1). The rowOffset is added to each of the rowIndices and may be
// negative. The matrix may be smaller than 'rowIndices[i] + rowOffset', e.g. interaction matrices
// may not be full-size.
void SetZeroOnRows(
    SparseMatrixView<real> mat,
    Span<int const> rowIndices,
    int rowOffset,
    real diagonalValue);
void SetZeroOnRows(
    BlockSparseMatrixView<real, 3> mat,
    Span<int const> rowIndices,
    int rowOffset,
    real diagonalValue);
void SetZeroOnRows(
    BlockSparseMatrixView<real, 4> mat,
    Span<int const> rowIndices,
    int rowOffset,
    real diagonalValue);
void SetZeroOnRows(
    MatrixView<real> mat,
    Span<int const> rowIndices,
    int rowOffset,
    real diagonalValue);
void SetZeroOnRows(ColumnVectorView<real> col, Span<int const> rowIndices, int rowOffset);
void SetZeroOnRows(
    AnyMatrixView<real> mat,
    Span<int const> rowIndices,
    int rowOffset,
    real diagonalValue);

/**
 * @brief Set the values on the specified columns of a matrix to zero.
 *
 * @param[in,out] mat The matrix to modify.
 * @param[in] colIndices The column indices to zero out.
 * @param[in] colOffset An offset added to each column index (may be negative). The actual column
 * zeroed is `colIndices[i] + colOffset`. Out-of-bounds columns (negative or >= mat.Cols()) are
 * possible for interaction matrices that do not span the full system and therefore are ignored
 * without error.
 * @param[in] symmetricPair A matrix whose sparsity pattern is the transpose of `mat`. For a
 * symmetric sparsity pattern, pass `AsConstView(mat)`. For an off-diagonal interaction matrix from
 * actor A to actor B, the symmetric pair is the matrix from B to A. The symmetric pair's row c
 * contains the column indices of row c in the transpose, which are exactly the row indices that
 * have non-zeros in column c of `mat`.
 * @param[in] valueIndicesCache Optional precomputed cache of value array indices to zero. When
 * provided, the function simply zeros out all cached value indices, bypassing any row/column
 * lookups and significantly improving performance.
 *
 * @note For dense matrices, the symmetric pair and cache are ignored.
 */
void SetZeroOnCols(
    SparseMatrixView<real> mat,
    Span<int const> colIndices,
    int colOffset,
    AnyMatrixView<real const> symmetricPair,
    Span<int const> valueIndicesCache = {});
void SetZeroOnCols(
    BlockSparseMatrixView<real, 3> mat,
    Span<int const> colIndices,
    int colOffset,
    AnyMatrixView<real const> symmetricPair,
    Span<int const> valueIndicesCache = {});
void SetZeroOnCols(
    BlockSparseMatrixView<real, 4> mat,
    Span<int const> colIndices,
    int colOffset,
    AnyMatrixView<real const> symmetricPair,
    Span<int const> valueIndicesCache = {});
void SetZeroOnCols(
    MatrixView<real> mat,
    Span<int const> colIndices,
    int colOffset,
    AnyMatrixView<real const> symmetricPair,
    Span<int const> valueIndicesCache = {});
void SetZeroOnCols(
    AnyMatrixView<real> mat,
    Span<int const> colIndices,
    int colOffset,
    AnyMatrixView<real const> symmetricPair,
    Span<int const> valueIndicesCache = {});

/**
 * @brief For each column index in newColIndices, computes which indices in the sparse matrix's
 * value array correspond to non-zeros in that column, and appends them to valueIndicesCache.
 *
 * @param[in] sparsityGraph The sparsity graph (must be square with symmetric pattern).
 * @param[in] newColIndices The column indices whose value array indices should be appended to the
 * cache.
 * @param[in,out] valueIndicesCache The cache to append to (new indices are added at the end).
 *
 * @warning The sparsity graph must be symmetric (i.e., if (r, c) is in the graph, then (c, r) must
 * also be in the graph). Results are undefined if this requirement is not met.
 * @note The appended indices are sorted to improve memory efficiency when the cache is used.
 */
template <typename CRIdx, typename Ptr, template <typename, typename...> typename Storage>
void AppendColValueIndexCache(
    Graph<CRIdx, Ptr, Storage> const& sparsityGraph,
    Span<int const> newColIndices,
    DynamicArray<int>& valueIndicesCache) {
  MOCHI_PROFILE_SCOPE();
  if (newColIndices.empty()) {
    return;
  }

  [[maybe_unused]] int const numRows = isize(sparsityGraph.GetPointers()) - 1;
  MOCHI_ASSERT_VERBOSE(numRows > 0, "Sparsity graph must not be empty.");

  auto const& pointers = sparsityGraph.GetPointers();
  auto const& targets = sparsityGraph.GetTargets();

  // For a symmetric sparsity pattern, if column c has a non-zero in row r, then row r
  // must also have a non-zero in column c. We use this to find value indices:
  // 1. For each column c, get the rows with non-zeros from targets[pointers[c]:pointers[c+1]]
  // 2. For each such row r, search row r's targets to find the value index for (r, c)
  auto const initialSize = valueIndicesCache.size();
  for (int c : newColIndices) {
    MOCHI_ASSERT_VERBOSE(c >= 0 && c < numRows, "Column index out of range.");
    // Get all rows that have non-zeros in column c (using symmetry: these are the targets of c)
    for (int ptr = pointers[c]; ptr < pointers[c + 1]; ++ptr) {
      int r = targets[ptr];
      // Now find the value index for (r, c) by searching row r's targets for column c
      [[maybe_unused]] bool found = false;
      for (int rPtr = pointers[r]; rPtr < pointers[r + 1]; ++rPtr) {
        if (targets[rPtr] == c) {
          valueIndicesCache.push_back(rPtr); // This is the value index for (r, c)
          found = true;
          break;
        }
      }
      MOCHI_ASSERT_VERBOSE(found, "Sparsity pattern is not symmetric.");
    }
  }

  // Sort new indices for better memory access during lookup.
  ParallelSort(valueIndicesCache.begin() + initialSize, valueIndicesCache.end());
}

// Build a SparseMatrix which is a copy of the original, except that all zeros have been pruned from
// the sparsity pattern. If (blockSize > 1) then we will only blocks if all values within the block
// are zero.
SparseMatrix<real> DuplicateAndPrune(SparseMatrixView<real const> mat, int blockSize);

} // namespace mochi
