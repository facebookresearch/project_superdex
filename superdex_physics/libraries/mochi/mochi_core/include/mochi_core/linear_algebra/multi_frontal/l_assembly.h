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

#include <mochi_core/linear_algebra/block_one_d_view.h>
#include <mochi_core/linear_algebra/multi_frontal/assembly_helper.h>
#include <mochi_core/linear_algebra/multi_frontal/l_matrix.h>

namespace mochi {

template <typename Scalar, size_t kBlockSize, IsBlockSparseMatrix M>
void AssembleSupernodeL(
    LMatrix<Scalar, kBlockSize>& L,
    AssemblyHelper const& helper,
    M const& matrix,
    int sn,
    bool zeroFirst) {
  constexpr int kDofsPerNode = M::kBlockSize;
  auto nBlockRows = helper.SuperColNodeCount(sn);
  auto nd = helper.FirstNode(sn);
  for (auto l : L.LforSN(sn).template NodalColumns<kDofsPerNode>()) {
    if (zeroFirst) {
      l.SetZero();
    }
    auto rowBlocks = helper.InputRow(matrix, nd);
    BlockColView<Scalar, kDofsPerNode> lNodal(l.data(), l.LeadDim(), nBlockRows);
    auto colSD = helper.LPlacements(nd);
    for (auto [aCol, lRow] : colSD) {
      auto block = rowBlocks[aCol];
      lNodal[lRow] += block.Transpose();
    }
    --nBlockRows;
    ++nd;
  }
}

} // namespace mochi
