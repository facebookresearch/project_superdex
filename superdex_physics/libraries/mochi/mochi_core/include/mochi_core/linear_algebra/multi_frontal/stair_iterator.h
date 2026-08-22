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

#include <mochi_core/linear_algebra/matrix.h>

namespace mochi {

template <typename Scalar, size_t kDofsPerNode, size_t kBlockCols>
struct StairNodalIterator {
  struct TmpWrapper {
    MatrixView<
        Scalar,
        krylov::kDynamic,
        kDofsPerNode,
        krylov::Direction::ColMajor,
        krylov::kDynamic>
        view;
    auto* operator->() {
      return &view;
    }
  };
  Scalar* v;
  Scalar* blockEnd;
  Scalar* fullEnd;
  int fullNRows; //!< Number of rows in the block
  int currentNRows; //!< number of rows from the top-left diagonal corner of the current node.
  StairNodalIterator(
      Scalar* v,
      Scalar* block_end,
      Scalar* full_end,
      int full_n_rows,
      int current_n_rows)
      : v(v),
        blockEnd(block_end),
        fullEnd(full_end),
        fullNRows(full_n_rows),
        currentNRows(current_n_rows) {}

  StairNodalIterator& operator++() {
    v += kDofsPerNode * fullNRows + kDofsPerNode;
    currentNRows -= kDofsPerNode;
    if (v >= blockEnd) {
      v = blockEnd;
      auto blockSize = std::min<size_t>(kBlockCols, currentNRows) * currentNRows;
      blockEnd += blockSize;
      blockEnd = std::min(blockEnd, fullEnd);
      fullNRows = currentNRows;
    }
    return *this;
  }

  auto operator*() const {
    return MatrixView<
        Scalar,
        krylov::kDynamic,
        kDofsPerNode,
        krylov::Direction::ColMajor,
        krylov::kDynamic>(v, currentNRows, kDofsPerNode, fullNRows);
  };

  auto operator->() const {
    return TmpWrapper{operator*()};
  }

  bool operator==(StairNodalIterator const& other) const {
    return v == other.v;
  }

  auto operator<=>(StairNodalIterator const& other) const {
    return v <=> other.v;
  }
};

} // namespace mochi
