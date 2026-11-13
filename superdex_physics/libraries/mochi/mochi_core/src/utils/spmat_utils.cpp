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

#include <mochi_core/linear_algebra/base_tools.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/spmat_utils.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <variant>
#include <vector>

using namespace mochi;

IndexGroups
mochi::CreateIndexGroups(Span<int const> inds, bool forceTriplets, Allocator* allocator) {
  IndexGroups outGroups(allocator);
  if (inds.empty()) {
    return outGroups;
  }

  // Conservative memory allocation
  outGroups.reserve(inds.size());

  // Traverse the DoF indices creating consecutive groups
  int dst = inds[0];
  int src = 0;
  int count = 1;
  for (int j = 1; j < inds.size(); j++) {
    if ((inds[j] != inds[j - 1] + 1) || (forceTriplets && count == 3)) {
      outGroups.push_back(IndexGroup{src, dst, count});
      src = j;
      dst = inds[j];
      count = 1;
    } else {
      count++;
    }
  }
  outGroups.push_back(IndexGroup{src, dst, count});

  // Sort the index groups. This makes assembly into a sparse matrix a bit faster,
  // as we can assume that the index groups come in ascending order.
  std::ranges::sort(
      outGroups, [](IndexGroup const& a, IndexGroup const& b) { return a.dst < b.dst; });

  return outGroups;
}

#if MOCHI_ASSERT_VERBOSE_ENABLED
static void CheckRowsColsConsistency(
    Span<IndexGroup const> rows,
    Span<IndexGroup const> cols,
    RowMatrixView<real const> values) {
  int numRows = 0;
  for (auto row : rows) {
    MOCHI_ASSERT(row.src + row.count <= values.Rows(), "Indices of source rows out of bounds");
    numRows += row.count;
  }
  MOCHI_ASSERT(numRows == values.Rows(), "Row indices and values must have the same size");
  int numCols = 0;
  for (auto col : cols) {
    MOCHI_ASSERT(col.src + col.count <= values.Cols(), "Indices of source cols out of bounds");
    numCols += col.count;
  }
  MOCHI_ASSERT(numCols == values.Cols(), "Col indices and values must have the same size");
}
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

// Stores in offsets[j] the position of block column cols[j].dst / kBlockSize in the sorted block
// column indices of a block row (a sparse matrix row is a block row with kBlockSize = 1). The
// groups are sorted too, so each search starts where the previous one stopped, and a forward scan
// finds them all in one pass over the row.
template <int kBlockSize>
static MOCHI_FORCE_INLINE void
FindColGroupOffsets(Span<int const> indices, Span<IndexGroup const> cols, Span<int> offsets) {
  // Rows with more indices per group use binary searches. Sequential scan compares cost far less
  // than dependent binary-search probes, so this sits far above log2(row length).
  constexpr int64_t kMaxScanIndicesPerGroup = 32;
  bool const scan = isize(indices) <= kMaxScanIndicesPerGroup * isize(cols);
  int const* searchBegin = indices.begin();
  for (int j = 0; j < cols.size(); ++j) {
    int const blockCol = cols[j].dst / kBlockSize;
    int const* found = scan
        ? std::find_if(searchBegin, indices.end(), [blockCol](int c) { return c >= blockCol; })
        : std::lower_bound(searchBegin, indices.end(), blockCol);
    MOCHI_ASSERT_VERBOSE(
        (found != indices.end()) && (*found == blockCol),
        "Position [row][col] is not in the sparsity pattern.");
    offsets[j] = static_cast<int>(found - indices.begin());
    // Do NOT start the next search at (found + 1): IndexGroups may repeat the same destination
    // block column, i.e. cols[j + 1].dst / kBlockSize == cols[j].dst / kBlockSize is possible.
    searchBegin = found;
  }
}

void mochi::MatAddSubBlocks(
    SparseMatrixView<real> dstMatrix,
    Span<IndexGroup const> rows,
    Span<IndexGroup const> cols,
    RowMatrixView<real const> values) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
  CheckRowsColsConsistency(rows, cols, values);
#endif
  // Subsequent rows in dstMatrix often have the same sparsity. To avoid repeated searches, we
  // store the offset of the start of each column IndexGroup. We can then reuse this information
  // for any subsequent row that has the same sparsity (same column indices in dstMatrix).
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 512 * sizeof(int)); // Up to 512 index groups.
  DynamicArray<int> dstColOffsets(&allocator);
  dstColOffsets.resize_noinit(cols.size());
  int dstNumColsInRow = -1;

  // Update the cache for a specific row
  auto UpdateCache = [&](int globalRow) {
    Span<int const> dstIndices = dstMatrix.Indices(globalRow);
    FindColGroupOffsets<1>(dstIndices, cols, MakeSpan(dstColOffsets));
    dstNumColsInRow = isize(dstIndices);
  };

  // For each row group
  for (auto row : rows) {
    for (int k = 0; k < row.count; ++k) {
      int srcRow = row.src + k;
      int dstRow = row.dst + k;
      auto srcValues = values.Row(srcRow);
      auto dstValues = RowVectorView<real>{dstMatrix.Values(dstRow)};
      auto dstIndices = dstMatrix.Indices(dstRow);
      if (isize(dstIndices) != dstNumColsInRow) {
        UpdateCache(dstRow); // Sparsity has changed. Update the col offsets.
        MOCHI_ASSERT_VERBOSE(isize(dstIndices) == dstNumColsInRow); // Just checking
      }
      for (int j = 0; j < cols.size(); ++j) {
        if (dstIndices[dstColOffsets[j]] != cols[j].dst) {
          UpdateCache(dstRow); // Sparsity has changed. Update the col offsets.
          MOCHI_ASSERT_VERBOSE(dstIndices[dstColOffsets[j]] == cols[j].dst); // Just checking
        }
        auto dst = dstValues.MiddleCols(dstColOffsets[j], cols[j].count);
        auto src = srcValues.MiddleCols(cols[j].src, cols[j].count);
        dst += src;
      }
    }
  }
}

template <int kBlockSize>
void MatAddSubBlocksImpl(
    BlockSparseMatrixView<real, kBlockSize> dstMatrix,
    Span<IndexGroup const> rows,
    Span<IndexGroup const> cols,
    RowMatrixView<real const> values) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
  CheckRowsColsConsistency(rows, cols, values);
#endif

  // Subsequent block rows in dstMatrix often have the same sparsity. To avoid repeated searches, we
  // store the offset of the start of each column IndexGroup. We can then reuse this information
  // for any subsequent block row that has the same sparsity (same block column indices in
  // dstMatrix).
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 512 * sizeof(int)); // Up to 512 index groups.
  DynamicArray<int> dstBlockColOffsets(&allocator);
  dstBlockColOffsets.resize_noinit(cols.size());
  int dstNumBlockColsInRow = -1;

  // Update the cache for a specific block row
  auto UpdateCache = [&](int globalBr) {
    Span<int const> dstIndices = dstMatrix.Indices(globalBr);
    FindColGroupOffsets<kBlockSize>(dstIndices, cols, MakeSpan(dstBlockColOffsets));
    dstNumBlockColsInRow = isize(dstIndices);
  };

  for (auto row : rows) {
    int numRowsInBlock = {};
    int dstBr = row.dst / kBlockSize;
    int lr = row.dst - kBlockSize * dstBr; // Local row within the block row.
    for (int k = 0; k < row.count; k += numRowsInBlock, ++dstBr, lr = 0) {
      numRowsInBlock = Min(kBlockSize - lr, row.count - k);
      auto srcValues = values.MiddleRows(row.src + k, numRowsInBlock);
      auto dstValues = dstMatrix.Values(dstBr).Underlying();
      auto dstIndices = dstMatrix.Indices(dstBr);
      if (isize(dstIndices) != dstNumBlockColsInRow) {
        UpdateCache(dstBr); // Sparsity has changed. Update the block col offsets.
        MOCHI_ASSERT_VERBOSE(isize(dstIndices) == dstNumBlockColsInRow); // Just checking.
      }
      for (int j = 0; j < cols.size(); ++j) {
        if (dstIndices[dstBlockColOffsets[j]] != cols[j].dst / kBlockSize) {
          UpdateCache(dstBr); // Sparsity has changed. Update the block col offsets.
          MOCHI_ASSERT_VERBOSE(
              dstIndices[dstBlockColOffsets[j]] == cols[j].dst / kBlockSize); // Just checking
        }
        int lc = cols[j].dst - kBlockSize * dstIndices[dstBlockColOffsets[j]];
        auto dst = dstValues.Block(
            lr, kBlockSize * dstBlockColOffsets[j] + lc, numRowsInBlock, cols[j].count);
        auto src = srcValues.MiddleCols(cols[j].src, cols[j].count);
        dst += src;
      }
    }
  }
}

void mochi::MatAddSubBlocks(
    BlockSparseMatrixView<real, 3> dstMatrix,
    Span<IndexGroup const> rows,
    Span<IndexGroup const> cols,
    RowMatrixView<real const> values) {
  MatAddSubBlocksImpl<3>(dstMatrix, rows, cols, values);
}

void mochi::MatAddSubBlocks(
    BlockSparseMatrixView<real, 4> dstMatrix,
    Span<IndexGroup const> rows,
    Span<IndexGroup const> cols,
    RowMatrixView<real const> values) {
  MatAddSubBlocksImpl<4>(dstMatrix, rows, cols, values);
}

void mochi::MatAddSubBlocks(
    MatrixView<real> dstMatrix,
    Span<IndexGroup const> rows,
    Span<IndexGroup const> cols,
    RowMatrixView<real const> values) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
  CheckRowsColsConsistency(rows, cols, values);
#endif
  for (auto row : rows) {
    for (auto col : cols) {
      dstMatrix.Block(row.dst, col.dst, row.count, col.count) +=
          values.Block(row.src, col.src, row.count, col.count);
    }
  }
}

void mochi::MatAddSubBlocks(
    AnyMatrixView<real>& dstMatrix,
    Span<IndexGroup const> rows,
    Span<IndexGroup const> cols,
    RowMatrixView<real const> values) {
  std::visit([&](auto& dest) { MatAddSubBlocks(dest, rows, cols, values); }, dstMatrix);
}
