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

#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/sparsity_utils.h>
#include <mochi_core/utils/task_scheduler.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

using namespace mochi;

#if MOCHI_DEBUG
static bool EveryValidRowIsSortedAndWithoutDuplicatesUpToGivenBound(
    std::vector<std::vector<int>> const& M,
    std::vector<int> const& usableColumnsCountPerRow) {
  for (int i = 0; i < usableColumnsCountPerRow.size(); ++i) {
    auto it_begin = M[i].cbegin();
    auto it_end = it_begin + usableColumnsCountPerRow[i];

    bool const b1 = std::is_sorted(it_begin, it_end);
    if (!b1) {
      return false;
    }
    bool const b2 = std::adjacent_find(it_begin, it_end) == it_end;
    if (!b2) {
      return false;
    }
  }

  return true;
}
#endif

static Graph<int, int> CreateCSRGraphFromSortedAndUniquedMatrix(
    int const numRows,
    std::vector<std::vector<int>> const& M,
    int numUsableRows,
    std::vector<int> const& usableColumnsCountPerRow) {
  MOCHI_PROFILE_SCOPE();

  MOCHI_ASSERT(numUsableRows <= M.size());
  MOCHI_ASSERT(numUsableRows <= usableColumnsCountPerRow.size());
#if MOCHI_DEBUG
  MOCHI_ASSERT(
      EveryValidRowIsSortedAndWithoutDuplicatesUpToGivenBound(M, usableColumnsCountPerRow));
#endif

  DynamicArray<int> pointers;
  DynamicArray<int> indices;

  auto indicesSize = std::accumulate(
      usableColumnsCountPerRow.begin(),
      usableColumnsCountPerRow.begin() + numUsableRows,
      0); // It could also be precomputed and passed as argument to
          // CreateCSRGraphFromSortedAndUniquedMatrix
  indices.reserve(indicesSize);

  pointers.reserve(numRows + 1);
  pointers.push_back(0);
  for (int i = 0; i < numRows; ++i) {
    // this if is needed to cover the case when numRowsOps is used to modify the number of rows
    if (i < numUsableRows) {
      auto from_it_begin = M[i].cbegin();
      auto from_it_end = M[i].cbegin() + usableColumnsCountPerRow[i];
      auto destination_it = std::back_inserter(indices);
      std::copy(from_it_begin, from_it_end, destination_it);
    }
    pointers.push_back(isize(indices));
  }

  return Graph<int, int>{std::move(pointers), std::move(indices)};
}

static void SortAndUniqueEachValidRow(
    std::vector<std::vector<int>>& M,
    std::vector<int>& usableColumnsCountPerRow) {
  // on entry:
  // - M contains the values to sort and unique
  // - usableColumnsCountPerRow contains the number of usable entries per row in M.
  //
  // on exit:
  // - M[i] is sorted and without duplicates up to usableColumnsCountPerRow[i]
  // - usableColumnsCountPerRow contains the new count per row to use

  MOCHI_PROFILE_SCOPE();

  // the number of rows to use might not be all the matrix' rows
  // this can be the case when using a matrix that is larger but we only
  // care about a specific range of rows
  int const numUsableRows = isize(usableColumnsCountPerRow);

  MOCHI_ASSERT(numUsableRows <= M.size());

  // FRIZZI: this 200 should be tuned
  int const itemsPerThread = std::min(numUsableRows, 200);
  ParallelForN("SortAndFindDuplicates", numUsableRows, itemsPerThread, [&](int i) {
    MOCHI_ASSERT(usableColumnsCountPerRow[i] <= M[i].size());

    auto it_begin = M[i].begin();
    auto it_end = it_begin + usableColumnsCountPerRow[i];
    std::sort(it_begin, it_end);
    auto unique_end_it = std::unique(it_begin, it_end);

    // overwrite to reflect the sort and unique
    usableColumnsCountPerRow[i] = static_cast<int>(std::distance(it_begin, unique_end_it));
  });
}

//
// overloads accepting Local2GlobalMap
//
Graph<int, int> mochi::MakeSparsityGraph(
    Local2GlobalMap const& map,
    std::vector<std::vector<int>>& M,
    std::optional<int> numRowsOpt) {
  MOCHI_PROFILE_SCOPE();

  if (M.empty()) {
    return {};
  }

  int const currentColCount = isize(M[0]);
  auto const maxRangeValueFoundFromMapPlusOne = map.GetGlobalRange().Max() + 1;
  // if the matrix passed does not have enough rows, we resize it.
  if (M.size() < maxRangeValueFoundFromMapPlusOne) {
    M.resize(maxRangeValueFoundFromMapPlusOne, std::vector<int>(currentColCount));
  }

  std::vector<int> usableCountPerRow(maxRangeValueFoundFromMapPlusOne, 0);
  for (int e = 0; e < map.GetNumElements(); ++e) {
    auto indices = map.GetGlobalIndices(e);
    for (int ii = 0; ii < isize(indices); ++ii) {
      int const row = indices[ii];
      auto& vec = M[row];

      int col = usableCountPerRow[row];
      for (int j : indices) {
        vec[col++] = j;
      }
      usableCountPerRow[row] = col;
      MOCHI_ASSERT(col < currentColCount);
    }
  }

  int numRows = maxRangeValueFoundFromMapPlusOne;
  if (numRowsOpt) {
    MOCHI_ASSERT(numRowsOpt.value() >= numRows, "Input rows don't match coordinates");
    numRows = numRowsOpt.value();
  }

  SortAndUniqueEachValidRow(M, usableCountPerRow);
  return CreateCSRGraphFromSortedAndUniquedMatrix(
      numRows, M, maxRangeValueFoundFromMapPlusOne, usableCountPerRow);
}

Graph<int, int> mochi::MakeSparsityGraph(
    Local2GlobalMap const& map,
    std::optional<int> numRowsOpt) {
  MOCHI_PROFILE_SCOPE();

  auto const maxRangeValueFoundFromMapPlusOne = map.GetGlobalRange().Max() + 1;
  std::vector<std::vector<int>> M(maxRangeValueFoundFromMapPlusOne, std::vector<int>{});
  for (int e = 0; e < map.GetNumElements(); ++e) {
    auto indices = map.GetGlobalIndices(e);
    for (int ii = 0; ii < isize(indices); ++ii) {
      int const row = indices[ii];
      auto& vec = M[row];
      for (int j : indices) {
        vec.emplace_back(j);
      }
    }
  }

  // since here M was filled with only the values needed, so the number
  // of usable entries per row is just the size of each row
  std::vector<int> usableCountPerRow;
  usableCountPerRow.reserve(M.size());
  for (auto const& row : M) {
    usableCountPerRow.push_back(isize(row));
  }

  int numGraphRows = isize(M);
  if (numRowsOpt) {
    MOCHI_ASSERT(numRowsOpt.value() >= numGraphRows, "Input rows don't match coordinates");
    numGraphRows = numRowsOpt.value();
  }

  SortAndUniqueEachValidRow(M, usableCountPerRow);
  return CreateCSRGraphFromSortedAndUniquedMatrix(
      numGraphRows, M, maxRangeValueFoundFromMapPlusOne, usableCountPerRow);
}

//
// overloads accepting r-value ref of vector of vectors
//

Graph<int, int> mochi::MakeSparsityGraph(
    std::vector<std::vector<int>>&& M,
    std::optional<int> numRowsOpt) {
  MOCHI_PROFILE_SCOPE();
  if (numRowsOpt) {
    MOCHI_ASSERT(numRowsOpt.value() >= 0, "Number of rows must be non-negative.");
  }

  if (M.empty()) {
    int const numRows = numRowsOpt.value_or(0);
    return CreateCSRGraphFromSortedAndUniquedMatrix(numRows, {}, 0, {});
  }

  // since here M is given, the number
  // of usable entries per row is just the size of each row
  std::vector<int> usableCountPerRow;
  usableCountPerRow.reserve(M.size());
  for (auto const& row : M) {
    usableCountPerRow.push_back(isize(row));
  }

  int numGraphRows = isize(M);
  if (numRowsOpt) {
    MOCHI_ASSERT(numRowsOpt.value() >= numGraphRows, "Input rows don't match coordinates.");
    numGraphRows = numRowsOpt.value();
  }

  SortAndUniqueEachValidRow(M, usableCountPerRow);
  return CreateCSRGraphFromSortedAndUniquedMatrix(numGraphRows, M, isize(M), usableCountPerRow);
}

//
// overload accepting vector of (row, col) pairs
//

Graph<int, int> mochi::MakeSparsityGraph(
    std::vector<Int2> coordinates,
    std::optional<int> numRowsOpt) {
  // Coordinates is a list of (row, col) pairs.
  MOCHI_PROFILE_SCOPE();

  if (coordinates.empty()) {
    int const numRows = numRowsOpt.value_or(0);
    return CreateCSRGraphFromSortedAndUniquedMatrix(numRows, {}, 0, {});
  }

  auto const max_row_it =
      std::max_element(coordinates.cbegin(), coordinates.cend(), [](Int2 const& a, Int2 const& b) {
        return a[0] < b[0];
      });

  // since we have the max row, we need + 1 below to account for 0-based enumeration
  std::vector<std::vector<int>> M((*max_row_it)[0] + 1, std::vector<int>{});
  for (auto const& coord_it : coordinates) {
    auto const i = coord_it[0];
    auto const j = coord_it[1];
    M[i].emplace_back(j);
  }

  // since here M was filled with only the values needed, so the number
  // of usable entries per row is just the size of each row
  std::vector<int> usableCountPerRow;
  usableCountPerRow.reserve(M.size());
  for (auto const& row : M) {
    usableCountPerRow.push_back(isize(row));
  }

  int numGraphRows = isize(M);
  if (numRowsOpt) {
    MOCHI_ASSERT(numRowsOpt.value() >= numGraphRows, "Input rows don't match coordinates");
    numGraphRows = numRowsOpt.value();
  }

  SortAndUniqueEachValidRow(M, usableCountPerRow);
  return CreateCSRGraphFromSortedAndUniquedMatrix(numGraphRows, M, isize(M), usableCountPerRow);
}

//
// other functions
//

Graph<int, int> mochi::MakeDenseSparsityGraph(int rows, int cols) {
  MOCHI_ASSERT_VERBOSE(rows >= 0 && cols >= 0, "Matrix dimensions must be non-negative.");
  DynamicArray<int> pointers;
  DynamicArray<int> indices;
  if (rows > 0) {
    pointers.resize(rows + 1);
    indices.resize(rows * cols);
    for (int r = 0; r < rows; ++r) {
      pointers[r + 1] = (r + 1) * cols;
      std::iota(indices.begin() + r * cols, indices.begin() + r * cols + cols, 0);
    }
  }
  return Graph<int, int>{std::move(pointers), std::move(indices)};
}

void mochi::AppendNonZeroCoordinates(
    std::vector<Int2>& outCoordinates,
    Graph<int, int> const& sparsity,
    int dofOffset) {
  outCoordinates.reserve(outCoordinates.size() + sparsity.NumTargets());
  auto pointers = sparsity.GetPointers();
  auto indices = sparsity.GetTargets();
  for (int r = 0; r < sparsity.size(); ++r) {
    int rowLen = pointers[r + 1] - pointers[r];
    for (int i = 0; i < rowLen; ++i) {
      int c = indices[pointers[r] + i];
      outCoordinates.emplace_back(r + dofOffset, c + dofOffset);
    }
  }
}

void mochi::SetZeroOnRows(
    SparseMatrixView<real> mat,
    Span<int const> rowIndices,
    int rowOffset,
    real diagonalValue) {
  if (mat.NumNonZeros() == 0) {
    return; // Early exit (also avoids division by zero).
  }

  int const numRows = mat.Rows();
  bool const isDiagNonZero = (diagonalValue != 0_r);

  auto workerTask = [&](int iBegin, int iEnd) {
    for (int r : rowIndices.subspan(iBegin, iEnd - iBegin)) {
      r += rowOffset; // rowOffset may be negative
      if (r >= 0 && r < numRows) {
        auto values = mat.Values(r);
        AsView(values).SetZero();
        if (isDiagNonZero) {
          auto indices = mat.Indices(r);
          auto const* it = std::find(indices.begin(), indices.end(), r);
          if (it != indices.end()) {
            values[std::distance(indices.begin(), it)] = diagonalValue;
          }
        }
      }
    }
  };

  constexpr long long kMinValuesPerTask = 50000; // 50 μs @ 1G values per second.
  int const minRowsPerTask =
      Max(1, static_cast<int>(kMinValuesPerTask * mat.Rows() / mat.NumNonZeros()));
  ParallelForRange(
      "SetZeroOnRows_SparseMatrix", 0, isize(rowIndices), minRowsPerTask, INT_MAX, workerTask);
}

template <int kBlockSize>
static void SetZeroOnRowsBlockSparseImpl(
    BlockSparseMatrixView<real, kBlockSize> mat,
    Span<int const> rowIndices,
    int rowOffset,
    real diagonalValue) {
  if (mat.NumNonZeros() == 0) {
    return; // Early exit (also avoids division by zero).
  }

  int const numRows = mat.Rows();
  bool const isDiagNonZero = (diagonalValue != 0_r);
  using IdxType = typename BlockSparseMatrixView<real, kBlockSize>::NonConstIdx;

  auto workerTask = [&](int iBegin, int iEnd) {
    for (int i = iBegin; i < iEnd; ++i) {
      int r = rowIndices[i] + rowOffset; // rowOffset may be negative
      if (r >= 0 && r < numRows) {
        int br = r / kBlockSize;
        int lr = r % kBlockSize;
        auto blockIndices = mat.Indices(br);
        auto blockValues = mat.Values(br);
        MOCHI_ASSERT_VERBOSE(isize(blockIndices) == blockValues.NumBlocks());

        // Check if we have a full block of consecutive rows
        bool hasFullBlock = (lr == 0) && (i + kBlockSize - 1 < iEnd);
        if (hasFullBlock) {
          for (int k = 1; k < kBlockSize; ++k) {
            hasFullBlock &= (rowIndices[i + k] == rowIndices[i] + k);
          }
        }

        if (hasFullBlock) {
          // Zero the entire block
          blockValues.Underlying().SetZero();
          if (isDiagNonZero) {
            auto const* it = std::find(blockIndices.begin(), blockIndices.end(), br);
            if (it != blockIndices.end()) {
              auto diagBlock =
                  blockValues[static_cast<IdxType>(std::distance(blockIndices.begin(), it))];
              for (int k = 0; k < kBlockSize; ++k) {
                diagBlock(k, k) = diagonalValue;
              }
            }
          }
          i += kBlockSize - 1; // Skip the next (kBlockSize-1) rows. We already handled them.
        } else {
          // Since we were not given a set of consecutive rows, we have to set values individually.
          blockValues.Underlying().Row(lr).SetZero();
          if (isDiagNonZero) {
            auto const* it = std::find(blockIndices.begin(), blockIndices.end(), br);
            if (it != blockIndices.end()) {
              blockValues[static_cast<IdxType>(std::distance(blockIndices.begin(), it))](lr, lr) =
                  diagonalValue;
            }
          }
        }
      }
    }
  };

  constexpr long long kMinValuesPerTask = 50000; // 50 μs @ 1G values per second.
  int const minRowsPerTask =
      Max(1, static_cast<int>(kMinValuesPerTask * mat.Rows() / mat.NumNonZeros()));
  ParallelForRange(
      "SetZeroOnRows_BlockSparseMatrix", 0, isize(rowIndices), minRowsPerTask, INT_MAX, workerTask);
}

void mochi::SetZeroOnRows(
    BlockSparseMatrixView<real, 3> mat,
    Span<int const> rowIndices,
    int rowOffset,
    real diagonalValue) {
  SetZeroOnRowsBlockSparseImpl<3>(mat, rowIndices, rowOffset, diagonalValue);
}

void mochi::SetZeroOnRows(
    BlockSparseMatrixView<real, 4> mat,
    Span<int const> rowIndices,
    int rowOffset,
    real diagonalValue) {
  SetZeroOnRowsBlockSparseImpl<4>(mat, rowIndices, rowOffset, diagonalValue);
}

void mochi::SetZeroOnRows(
    MatrixView<real> mat,
    Span<int const> rowIndices,
    int rowOffset,
    real diagonalValue) {
  for (int r : rowIndices) {
    r += rowOffset; // rowOffset may be negative
    if (r >= 0 && r < mat.Rows()) {
      mat.Row(r).SetZero();
      if (r < mat.Cols()) {
        mat(r, r) = diagonalValue;
      }
    }
  }
}

void mochi::SetZeroOnRows(ColumnVectorView<real> col, Span<int const> rowIndices, int rowOffset) {
  int const numRows = col.Rows();
  for (int r : rowIndices) {
    r += rowOffset; // rowOffset may be negative
    if (r >= 0 && r < numRows) {
      col[r] = 0_r;
    }
  }
}

void mochi::SetZeroOnRows(
    AnyMatrixView<real> anyMat,
    Span<int const> rowIndices,
    int rowOffset,
    real diagonalValue) {
  std::visit([&](auto& mat) { SetZeroOnRows(mat, rowIndices, rowOffset, diagonalValue); }, anyMat);
}

#if MOCHI_ASSERT_VERBOSE_ENABLED
template <typename Mat, typename SymPair>
static void AssertSymmetricPairConsistency(Mat const& mat, SymPair const& symPair) {
  MOCHI_ASSERT_VERBOSE(
      symPair.Rows() == mat.Cols() && symPair.Cols() == mat.Rows(),
      "Inconsistent sizes between symmetric pairs.");
  auto const matGraph = AsGraphView(mat);
  auto const lookupGraph = AsGraphView(symPair);
  auto reversedLookupGraph = Reverse<int, int>(lookupGraph, GetNumBlockRows(mat)).SortTargets();
  MOCHI_ASSERT_VERBOSE(
      reversedLookupGraph.GetPointers() == matGraph.GetPointers() &&
          reversedLookupGraph.GetTargets() == matGraph.GetTargets(),
      "Inconsistent sparsity between symmetric pairs.");
}

template <typename MatType>
static void AssertValueIndicesCacheConsistency(
    MatType const& mat,
    Span<int const> colIndices,
    int colOffset,
    Span<int const> valueIndicesCache) {
  constexpr int kBlockSize = MatType::kBlockSize;
  auto const blockPointers = mat.Pointers();
  auto const blockIndices = mat.Indices();
  int const numCols = mat.Cols();

  DynamicArray<int> expectedCache;
  expectedCache.reserve(valueIndicesCache.size());
  for (int const c : colIndices) {
    int const adjustedCol = c + colOffset;
    if (adjustedCol < 0 || adjustedCol >= numCols) {
      continue;
    }
    int const bc = adjustedCol / kBlockSize;
    int const ci = adjustedCol % kBlockSize;
    for (int ptr = blockPointers[bc]; ptr < blockPointers[bc + 1]; ++ptr) {
      int const br = blockIndices[ptr];
      // Find block (br, bc) in block-row br's index list to recover its value offset.
      auto const brIndices = mat.Indices(br);
      int const dist =
          static_cast<int>(std::find(brIndices.begin(), brIndices.end(), bc) - brIndices.begin());
      MOCHI_ASSERT_VERBOSE(dist != isize(brIndices), "Sparsity pattern is not symmetric.");
      int const brValuesBegin = kBlockSize * kBlockSize * blockPointers[br];
      for (int ri = 0; ri < kBlockSize; ++ri) {
        expectedCache.push_back(brValuesBegin + kBlockSize * (ri * isize(brIndices) + dist) + ci);
      }
    }
  }

  std::unordered_set<int> expectedSet(expectedCache.begin(), expectedCache.end());
  std::unordered_set<int> providedSet(valueIndicesCache.begin(), valueIndicesCache.end());
  MOCHI_ASSERT_VERBOSE(expectedSet == providedSet, "Inconsistent value indices cache.");
}
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

static void SetZeroIndexed(Span<real> values, Span<int const> indices, std::string_view taskName) {
  constexpr int kMinValuesPerTask = 50000; // 50 μs @ 1G values per second.
  ParallelForEach(taskName, indices, kMinValuesPerTask, [&](int i) { values[i] = 0_r; });
}

void mochi::SetZeroOnCols(
    SparseMatrixView<real> mat,
    Span<int const> colIndices,
    int colOffset,
    AnyMatrixView<real const> symmetricPair,
    Span<int const> valueIndicesCache) {
  MOCHI_PROFILE_SCOPE();
  if (mat.NumNonZeros() == 0) {
    return; // Early exit (also avoids division by zero).
  }

  if (!valueIndicesCache.empty()) {
    // Fast path: use precomputed cache of value indices.
#if MOCHI_ASSERT_VERBOSE_ENABLED
    // Verify cache is correct.
    AssertValueIndicesCacheConsistency(
        AsBlockSparseMatrixConstView(mat), colIndices, colOffset, valueIndicesCache);
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

    SetZeroIndexed(mat.Values(), valueIndicesCache, "SetZeroOnCols_SparseMatrix");

  } else {
    // Use row indices from the symmetric pair's sparsity pattern.
    // The symmetric pair's row c contains the column indices of row c in the transpose,
    // which are exactly the row indices that have non-zeros in column c of this matrix.
    MOCHI_ASSERT(std::holds_alternative<SparseMatrixView<real const>>(symmetricPair));
    auto const& lookupMat = std::get<SparseMatrixView<real const>>(symmetricPair);
#if MOCHI_ASSERT_VERBOSE_ENABLED
    AssertSymmetricPairConsistency(mat, lookupMat);
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

    auto workerTask = [&](int iBegin, int iEnd) {
      for (auto c : colIndices.subspan(iBegin, iEnd - iBegin)) {
        c += colOffset; // colOffset may be negative
        if (c >= 0 && c < mat.Cols()) {
          auto const rowIndices = lookupMat.Indices(c);
          for (auto r : rowIndices) {
            mat.SetValue(r, c, 0_r);
          }
        }
      }
    };

    constexpr long long kMinValuesPerTask = 10000; // 50 μs @ 200M values per second.
    int const minColsPerTask =
        Max(1, static_cast<int>(kMinValuesPerTask * mat.Cols() / mat.NumNonZeros()));
    ParallelForRange(
        "SetZeroOnCols_SparseMatrix", 0, isize(colIndices), minColsPerTask, INT_MAX, workerTask);
  }
}

template <int kBlockSize>
static void SetZeroOnColsBlockSparseImpl(
    BlockSparseMatrixView<real, kBlockSize> mat,
    Span<int const> colIndices,
    int colOffset,
    AnyMatrixView<real const> symmetricPair,
    Span<int const> valueIndicesCache) {
  MOCHI_PROFILE_SCOPE();
  if (mat.NumNonZeros() == 0) {
    return; // Early exit (also avoids division by zero).
  }

  if (!valueIndicesCache.empty()) {
    // Fast path: use precomputed cache of value indices.
#if MOCHI_ASSERT_VERBOSE_ENABLED
    // Verify cache is correct.
    AssertValueIndicesCacheConsistency(mat, colIndices, colOffset, valueIndicesCache);
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

    SetZeroIndexed(mat.Values(), valueIndicesCache, "SetZeroOnCols_BlockSparseMatrix");

  } else {
    // Use block row indices from the symmetric pair's sparsity pattern.
    // The symmetric pair's block row bc contains the block column indices of block row bc in the
    // transpose, which are exactly the block row indices that have non-zeros in block column bc of
    // this matrix.
    MOCHI_ASSERT(
        (std::holds_alternative<BlockSparseMatrixView<real const, kBlockSize>>(symmetricPair)));
    auto const& lookupMat = std::get<BlockSparseMatrixView<real const, kBlockSize>>(symmetricPair);
#if MOCHI_ASSERT_VERBOSE_ENABLED
    AssertSymmetricPairConsistency(mat, lookupMat);
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

    auto workerTask = [&](int iBegin, int iEnd) {
      for (auto c : colIndices.subspan(iBegin, iEnd - iBegin)) {
        c += colOffset; // colOffset may be negative
        if (c >= 0 && c < mat.Cols()) {
          auto const bc = c / kBlockSize;
          auto const bRowIndices = lookupMat.Indices(bc);
          for (auto br : bRowIndices) {
            for (auto r = br * kBlockSize; r < (br + 1) * kBlockSize; ++r) {
              mat.SetValue(r, c, 0_r);
            }
          }
        }
      }
    };

    constexpr long long kMinValuesPerTask = 10000; // 50 μs @ 200M values per second.
    int const minColsPerTask =
        Max(1, static_cast<int>(kMinValuesPerTask * mat.Cols() / mat.NumNonZeros()));
    ParallelForRange(
        "SetZeroOnCols_BlockSparseMatrix",
        0,
        isize(colIndices),
        minColsPerTask,
        INT_MAX,
        workerTask);
  }
}

void mochi::SetZeroOnCols(
    BlockSparseMatrixView<real, 3> mat,
    Span<int const> colIndices,
    int colOffset,
    AnyMatrixView<real const> symmetricPair,
    Span<int const> valueIndicesCache) {
  SetZeroOnColsBlockSparseImpl<3>(mat, colIndices, colOffset, symmetricPair, valueIndicesCache);
}

void mochi::SetZeroOnCols(
    BlockSparseMatrixView<real, 4> mat,
    Span<int const> colIndices,
    int colOffset,
    AnyMatrixView<real const> symmetricPair,
    Span<int const> valueIndicesCache) {
  SetZeroOnColsBlockSparseImpl<4>(mat, colIndices, colOffset, symmetricPair, valueIndicesCache);
}

void mochi::SetZeroOnCols(
    MatrixView<real> mat,
    Span<int const> colIndices,
    int colOffset,
    AnyMatrixView<real const> /*symmetricPair*/,
    Span<int const> /*valueIndicesCache*/) {
  MOCHI_PROFILE_SCOPE();
  for (auto c : colIndices) {
    c += colOffset; // colOffset may be negative
    if (c >= 0 && c < mat.Cols()) {
      mat.Col(c).SetZero();
    }
  }
}

void mochi::SetZeroOnCols(
    AnyMatrixView<real> anyMat,
    Span<int const> colIndices,
    int colOffset,
    AnyMatrixView<real const> symmetricPair,
    Span<int const> valueIndicesCache) {
  std::visit(
      [&](auto& mat) {
        SetZeroOnCols(mat, colIndices, colOffset, symmetricPair, valueIndicesCache);
      },
      anyMat);
}

// Prune and copy a sparse matrix with (blockSize x blockSize) blocks. If a
// block contains a single non-zero values, then the entire block will be copied. Destination
// buffers must be large enough for the worst case (nothing pruned).
static int CopyAndPruneBlocks(
    int numRows,
    int srcNumNonZeros,
    int blockSize,
    int const* srcOffsets,
    int const* srcCols,
    real const* srcValues,
    int* dstOffsets,
    int* dstCols,
    real* dstValues) {
  MOCHI_PROFILE_SCOPE();
  int dstCount = 0;
  std::vector<bool> srcIsInNzBlock;
  if (blockSize > 1) {
    // This code will run if we ever have a block size other than 3x3. This implementation has
    // not been optimized. It is about 3.5 times slower than the version above, which is
    // optimized for 3x3 blocks.
    MOCHI_PROFILE_SCOPE_N("ComputeNonZeroBlocksForPruning");

    // All blocks are zero unless proven otherwise below.
    srcIsInNzBlock.resize(srcNumNonZeros, false);
    for (int r0 = 0; r0 < numRows; r0 += blockSize) { // Loop over row blocks.
      int const numColsInRow = srcOffsets[r0 + 1] - srcOffsets[r0];

      // Loop over blocks in the row block.
      for (int c0 = 0; c0 < numColsInRow; c0 += blockSize) {
        bool areNonZerosInBlock = false;
        for (int r = r0; !areNonZerosInBlock && (r < r0 + blockSize); ++r) {
          for (int c = c0; c < c0 + blockSize; ++c) {
            if (srcValues[srcOffsets[r] + c] != 0_r) {
              areNonZerosInBlock = true;
              break;
            }
          }
        }

        if (areNonZerosInBlock) {
          for (int r = r0; r < r0 + blockSize; ++r) {
            for (int c = c0; c < c0 + blockSize; ++c) {
              srcIsInNzBlock[srcOffsets[r] + c] = true;
            }
          }
        }
      } // for (int c0 = 0; c0 < numColsInRow; c0 += blockSize)
    } // for (int r0 = 0; r0 < numRows; r0 += blockSize)
  } // if (blockSize > 1)

  // Copy just the non-zero blocks from the source matrix to our new memory storage.
  // Initialize the new non-zero structure as we do. I tried a SIMD implementation but it
  // wasn't any faster. Presumably performance is bound by memory read/write not compute.
  {
    MOCHI_PROFILE_SCOPE_N("CopyAndPrune");
    for (int r = 0; r < numRows; ++r) {
      int const srcRowOffset = srcOffsets[r];
      int const srcRowWidth = srcOffsets[r + 1] - srcRowOffset;
      int const* srcRowCols = srcCols + srcRowOffset;
      real const* srcRowVals = srcValues + srcRowOffset;
      int dstRowOffset = dstCount;
      for (int c = 0; c < srcRowWidth; ++c) {
        real const val = srcRowVals[c];
        if (blockSize > 1 ? srcIsInNzBlock[srcRowOffset + c] : (val != 0_r)) {
          dstCols[dstCount] = srcRowCols[c];
          dstValues[dstCount] = val;
          ++dstCount;
        }
      }
      dstOffsets[r] = dstRowOffset;
    }
  }

  return dstCount;
}

// Special case for copying and pruning a sparse matrix with 3x3 blocks. If a block contains a
// single non-zero values, then the entire block will be copied. Destination buffers must be large
// enough for the worst case (nothing pruned).
static int CopyAndPruneBlocks3x3(
    int numRows,
    int const* srcOffsets,
    int const* srcCols,
    real const* srcValues,
    int* dstOffsets,
    int* dstCols,
    real* dstValues) {
  MOCHI_PROFILE_SCOPE();

  int dstCount = 0;
  Vec4i const zeroOneTwo = {0, 1, 2, 0};
  using Vec4rAsInts = Simd<int, sizeof(Vec4r) / sizeof(int)>;
  static_assert(sizeof(Vec4rAsInts) == sizeof(Vec4r)); // Same byte size as Vec4r
  Vec4rAsInts const zero = {};
  Vec4rAsInts const mask = ReinterpretCast<Vec4rAsInts>(SimdMask<Vec4r>(true, true, true, false));

  // Iterate 3 rows at a time
  for (int r0 = 0; r0 < numRows; r0 += 3) {
    int const numColsInRow = srcOffsets[r0 + 1] - srcOffsets[r0];
    int const prevDstCount = dstCount;

    // Iterate across the row 3 columns at a time
    for (int c0 = 0; c0 < numColsInRow; c0 += 3) {
      // Load the next 3x3 block.
      Vec4r block[3] = {
          Load<3, Vec4r>(&srcValues[srcOffsets[r0 + 0] + c0]),
          Load<3, Vec4r>(&srcValues[srcOffsets[r0 + 1] + c0]),
          Load<3, Vec4r>(&srcValues[srcOffsets[r0 + 2] + c0])};
      Vec4rAsInts blockBitwiseOr = mask &
          (ReinterpretCast<Vec4rAsInts>(block[0]) | ReinterpretCast<Vec4rAsInts>(block[1]) |
           ReinterpretCast<Vec4rAsInts>(block[2]));
      if (blockBitwiseOr != zero) {
        // This is a non-zero block.
        int globalCol = srcCols[srcOffsets[r0] + c0];

        // Copy the first row of this non-zero block to the destination.
        Vec4i dstColIndices = globalCol + zeroOneTwo;
        Store<3>(&dstCols[dstCount], dstColIndices);
        Store<3>(&dstValues[dstCount], block[0]);

        // We can't copy the other rows to their final destination yet because we don't know
        // how many other non-zero blocks there will be in this row. Therefore, we temporarily
        // copy the values to a safe location in the destination buffer.
        Store<3>(&dstValues[dstCount + 1 * numColsInRow], block[1]);
        Store<3>(&dstValues[dstCount + 2 * numColsInRow], block[2]);
        dstCount += 3;
      }
    }

    int numNonZeroColsInRow = dstCount - prevDstCount;
    if (numNonZeroColsInRow) {
      // Duplicate the non-zero column indices for the 2nd and 3rd rows of the block.
      int* colsBegin0 = &dstCols[prevDstCount];
      int* colsBegin1 = colsBegin0 + numNonZeroColsInRow;
      int* colsBegin2 = colsBegin1 + numNonZeroColsInRow;
      std::copy(colsBegin0, colsBegin1, colsBegin1); // Copy indices from 1st row to 2nd row
      std::copy(colsBegin0, colsBegin1, colsBegin2); // Copy indices from 1st row to 3rd row

      if (numNonZeroColsInRow != numColsInRow) {
        // We didn't know how many non-zero blocks there would be, so we had to copy the
        // values to a conservative location in the destination buffer. Now we can shift them
        // down into their final location.
        memmove(
            &dstValues[dstCount],
            &dstValues[prevDstCount + 1 * numColsInRow],
            numNonZeroColsInRow * sizeof(real));
        memmove(
            &dstValues[dstCount + numNonZeroColsInRow],
            &dstValues[prevDstCount + 2 * numColsInRow],
            numNonZeroColsInRow * sizeof(real));
      }

      // The 2nd and 3rd rows of the block are now complete
      dstCount += 2 * numNonZeroColsInRow;
    }

    dstOffsets[r0 + 0] = prevDstCount;
    dstOffsets[r0 + 1] = prevDstCount + numNonZeroColsInRow;
    dstOffsets[r0 + 2] = prevDstCount + numNonZeroColsInRow + numNonZeroColsInRow;
  }

  return dstCount;
}

SparseMatrix<real> mochi::DuplicateAndPrune(SparseMatrixView<real const> mat, int blockSize) {
  int const numRows = mat.Rows();
  int const numCols = mat.Cols();
  int const srcNumNonZeros = mat.NumNonZeros();
  int const* srcPointers = mat.Pointers().data();
  int const* srcIndices = mat.Indices().data();
  real const* srcValues = mat.Values().data();

  // Check the block structure
#if MOCHI_ASSERT_VERBOSE_ENABLED
  if (blockSize > 1) {
    MOCHI_ASSERT_VERBOSE(
        numRows % blockSize == 0,
        "Inferred block size is incorrect. Global numRows must be a multiple of the block size.");
    MOCHI_ASSERT_VERBOSE(
        numCols % blockSize == 0,
        "Inferred block size is incorrect. Global numCols must be a multiple of the block size.");
    for (int r0 = 0; r0 < numRows; r0 += blockSize) {
      int const numColsInRow = srcPointers[r0 + 1] - srcPointers[r0];
      MOCHI_ASSERT_VERBOSE(
          numColsInRow % blockSize == 0,
          "Inferred block size is incorrect. numColsInRow must be a multiple of the block size.");
      for (int r = r0 + 1; r < r0 + blockSize; ++r) {
        int numColsInNextRow = srcPointers[r + 1] - srcPointers[r];
        MOCHI_ASSERT_VERBOSE(
            numColsInNextRow == numColsInRow,
            "Inferred block size is incorrect. Each row within a block should have the same numer of columns.");
      }
    }
  }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

  // Allocate enough memory to store the non-zero structure + values.
  // We don't know how many non-zero values there will be.
  // We allocate for the worst case (favoring speed not size).
  DynamicArray<int> dstPointers(numRows + 1);
  DynamicArray<int> dstIndices(srcNumNonZeros);
  DynamicArray<real> dstValues(srcNumNonZeros);

  int dstNumNonZeros = 0;
  if (blockSize == 3) {
    dstNumNonZeros = CopyAndPruneBlocks3x3(
        numRows,
        srcPointers,
        srcIndices,
        srcValues,
        dstPointers.data(),
        dstIndices.data(),
        dstValues.data());
  } else {
    dstNumNonZeros = CopyAndPruneBlocks(
        numRows,
        srcNumNonZeros,
        blockSize,
        srcPointers,
        srcIndices,
        srcValues,
        dstPointers.data(),
        dstIndices.data(),
        dstValues.data());
  }

  // The number of non-zeros in the last row is (dstOffsets[numRows] - dstOffsets[numRows - 1]).
  // Thus, dstPointers[numRows] is also equal to the total number of non-zeros in the matrix.
  dstPointers[numRows] = dstNumNonZeros;

  // Resize vectors (without reallocating)
  dstIndices.resize(dstNumNonZeros);
  dstValues.resize(dstNumNonZeros);

  // Put it all together
  return SparseMatrix<real>{
      numCols, std::move(dstPointers), std::move(dstIndices), std::move(dstValues)};
}
