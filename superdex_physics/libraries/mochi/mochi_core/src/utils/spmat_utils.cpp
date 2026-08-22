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
  std::sort(outGroups.begin(), outGroups.end(), [](IndexGroup const& a, IndexGroup const& b) {
    return a.dst < b.dst;
  });

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
  DynamicArray<int> dstColOffsets(cols.size(), &allocator);
  int dstNumColsInRow = -1;

  // Update the cache for a specific row
  auto UpdateCache = [&](int globalRow) {
    Span<int const> dstIndices = dstMatrix.Indices(globalRow);
    int const* searchBegin = dstIndices.begin();
    for (int j = 0; j < cols.size(); ++j) {
      // Find column index cols[j].dst in the sorted half open range [searchBegin, dstIndices.end())
      int const* dstIdxFound = std::lower_bound(searchBegin, dstIndices.end(), cols[j].dst);
      MOCHI_ASSERT_VERBOSE(
          (dstIdxFound != dstIndices.end()) && (*dstIdxFound == cols[j].dst),
          "Position [globalRow][globalCol] is not in the sparse matrix non-zero structure!");
      dstColOffsets[j] = static_cast<int>(dstIdxFound - dstIndices.begin());
      // Advance searchBegin because we know the dstIndices are in ascending order.
      // Do NOT advance to (dstIdxFound + 1) because it is possible for IndexGroups to repeat the
      // same destination index. In other words, it is possible that (cols[j+1].dst == cols[j].dst).
      searchBegin = dstIdxFound;
    }
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
  DynamicArray<int> dstBlockColOffsets(cols.size(), &allocator);
  int dstNumBlockColsInRow = -1;

  // Update the cache for a specific block row
  auto UpdateCache = [&](int globalBr) {
    Span<int const> dstIndices = dstMatrix.Indices(globalBr);
    int const* searchBegin = dstIndices.begin();
    for (int j = 0; j < cols.size(); ++j) {
      // Find column index cols[j].dst / kBlockSize in the sorted range [searchBegin,
      // dstIndices.end())
      int const* dstIdxFound =
          std::lower_bound(searchBegin, dstIndices.end(), cols[j].dst / kBlockSize);
      MOCHI_ASSERT_VERBOSE(
          (dstIdxFound != dstIndices.end()) && (*dstIdxFound == cols[j].dst / kBlockSize),
          "Position [globalBr][globalBc] is not in the block sparsity pattern.");
      dstBlockColOffsets[j] = static_cast<int>(dstIdxFound - dstIndices.begin());
      // Advance searchBegin because we know the dstIndices are in ascending order.
      // Do NOT advance to (dstIdxFound + 1) because it is possible for IndexGroups to repeat the
      // same destination index. In other words, it is possible that (cols[j+1].dst == cols[j].dst).
      searchBegin = dstIdxFound;
    }
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
