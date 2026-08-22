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
#include <mochi_core/linear_algebra/multi_frontal/assembly_helper.h>
#include <mochi_core/linear_algebra/multi_frontal/elim_tree.h>
#include <mochi_core/linear_algebra/multi_frontal/front_stack.h>
#include <mochi_core/linear_algebra/multi_frontal/l_matrix.h>

namespace mochi {
template <int kDofsPerNode, typename Scalar, size_t kBlockSize>
void FactorSubtree(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<Scalar, kBlockSize>& lMatrix,
    BlockSparseMatrixView<Scalar const, kDofsPerNode> const& A,
    AssemblyHelper& helper,
    int root,
    Span<Scalar> rootFrontSpace = {});

template <int kDofsPerNode, typename Scalar, size_t kBlockSize>
void MultiFrontalSolveInPlace(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<Scalar, kBlockSize>& lMatrix,
    Span<int const> order,
    ColumnVectorView<Scalar> x);

} // namespace mochi
