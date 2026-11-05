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
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/task_scheduler.h>

#include <algorithm>
#include <limits>
#include <vector>

namespace mochi {

/*
  Struct to manage groups of consecutive DoF indices
*/
struct IndexGroup {
  int src = 0; // DoF index in the source data (the ContactJac data structure)
  int dst = 0; // DoF index in the destination data (res or dres)
  int count = 0; // Size of the DoF group

  bool operator==(IndexGroup const&) const = default;
};

struct IndexGroups : public DynamicArray<IndexGroup> {
  using DynamicArray::DynamicArray;
};

/**
  CreateIndexGroups: This function receives a span of DoF indices and produces a vector of index
  groups, i.e., groups of consecutive DoF indices. The resulting groups are sorted in ascending
  order. Possibly force the groups to be triplets (convenient for soft actors, where consecutive
  groups larger than 3 are coincidental but do not indicate shared sparsity structure).
*/
IndexGroups CreateIndexGroups(
    Span<int const> inds,
    bool forceTriplets,
    Allocator* allocator = GetDefaultAllocator());

/**
  MatAddSubBlocks: Add a collection of rectangular dense blocks to a sparse, block sparse, or dense
  matrix. The row and column indices are passed in groups of consecutive indices.

  Args:
    dstMatrix - The matrix into which values should be accumulated
    rows - The row indices to modify, in groups (total length == M)
    cols - The column indices to modify, in groups (total length == N)
    values - The values to add to the matrix (size == M * N)

  Requirements:
    The index groups of cols must come in ascending order.
*/
void MatAddSubBlocks(
    SparseMatrixView<real> dstMatrix,
    Span<IndexGroup const> rows,
    Span<IndexGroup const> cols,
    RowMatrixView<real const> values);
void MatAddSubBlocks(
    BlockSparseMatrixView<real, 3> dstMatrix,
    Span<IndexGroup const> rows,
    Span<IndexGroup const> cols,
    RowMatrixView<real const> values);
void MatAddSubBlocks(
    BlockSparseMatrixView<real, 4> dstMatrix,
    Span<IndexGroup const> rows,
    Span<IndexGroup const> cols,
    RowMatrixView<real const> values);
void MatAddSubBlocks(
    MatrixView<real> dstMatrix,
    Span<IndexGroup const> rows,
    Span<IndexGroup const> cols,
    RowMatrixView<real const> values);
void MatAddSubBlocks(
    AnyMatrixView<real>& dstMatrix,
    Span<IndexGroup const> rows,
    Span<IndexGroup const> cols,
    RowMatrixView<real const> values);

} // namespace mochi
