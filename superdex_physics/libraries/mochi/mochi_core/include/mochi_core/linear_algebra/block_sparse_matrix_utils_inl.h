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

#include "block_sparse_matrix_utils.h"

#include <mochi_core/linear_algebra/block_one_d_view.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/task_scheduler.h>

#include <limits>
#include <type_traits>
#include <utility>

namespace mochi::details {

template <
    int kBlockSize,
    typename ScalarA,
    typename CRIdxA,
    typename PtrA,
    template <typename, typename...> typename StorageA,
    typename ScalarB,
    typename CRIdxB,
    typename PtrB,
    template <typename, typename...> typename StorageB,
    typename ScalarAB,
    typename CRIdxAB,
    typename PtrAB,
    template <typename, typename...> typename StorageAB>
void SparseMatProduct(
    BlockSparseMatrix<ScalarA, kBlockSize, CRIdxA, PtrA, StorageA> const& A,
    BlockSparseMatrix<ScalarB, kBlockSize, CRIdxB, PtrB, StorageB> const& B,
    BlockSparseMatrix<ScalarAB, kBlockSize, CRIdxAB, PtrAB, StorageAB>& AB) {
  using Idx = std::remove_const_t<CRIdxA>;
  auto const nBlockRows = static_cast<Idx>(A.BlockRows());
  auto const nItems = AB.NumNonZeroBlocks();
  if (nItems == 0) {
    return;
  }
  [[maybe_unused]] Idx const dummyFlag = std::numeric_limits<Idx>::max();
  MOCHI_ASSERT_VERBOSE(AB.BlockCols() < dummyFlag, "Incompatible flag");
  // TODO Explore whether the 'minPerTask' formula remains appropriate
  ParallelForRange(
      "SparseMatProduct",
      /* rangeBegin */ 0,
      /* rangeEnd */ nBlockRows,
      // At least 250 non-zero blocks in AB per task.
      /* minPerTask */ Clamp<Idx>((250 * nBlockRows) / nItems, 1, nBlockRows),
      /* maxPerTask */ nBlockRows,
      [&](Idx brBegin, Idx brEnd) {
  // Offset of nodes in the current row of AB being formed.
#if MOCHI_ASSERT_VERBOSE_ENABLED
        DynamicArray<Idx> ndOffset(AB.BlockCols(), dummyFlag);
#else
        DynamicArray<Idx> ndOffset;
        ndOffset.resize_noinit(AB.BlockCols());
#endif
        for (Idx i = brBegin; i < brEnd; ++i) {
          auto ABblockColIdx = AB.Indices(i);
          auto ABrowValues = AB.Values(i);
          ABrowValues.SetZero();
          for (Idx k = 0; k < ABblockColIdx.size(); ++k) {
            ndOffset[ABblockColIdx[k]] = k;
          }
          auto Arow = A.Values(i);
          auto ArowIndices = A.Indices(i);
          for (Idx k = 0; k < ArowIndices.size(); ++k) {
            auto aBlockCol = ArowIndices[k];
            auto Brow = B.Values(aBlockCol);
            auto Bindices = B.Indices(aBlockCol);
            for (Idx jj = 0; jj < Bindices.size(); ++jj) {
              MOCHI_ASSERT_VERBOSE(
                  ndOffset[Bindices[jj]] != dummyFlag &&
                  ndOffset[Bindices[jj]] < ABblockColIdx.size());
              ABrowValues[ndOffset[Bindices[jj]]] += Arow[k] * Brow[jj];
            }
          }
#if MOCHI_ASSERT_VERBOSE_ENABLED
          for (Idx k = 0; k < ABblockColIdx.size(); ++k) {
            // Only for the above check.
            ndOffset[ABblockColIdx[k]] = dummyFlag;
          }
#endif
        }
      });
}

} // namespace mochi::details

namespace mochi {
template <
    typename Scalar_,
    int kBlockSize,
    typename CRIdx,
    typename Ptr_,
    template <typename, typename...> typename Storage>
auto Transpose(BlockSparseMatrix<Scalar_, kBlockSize, CRIdx, Ptr_, Storage> const& A) {
  auto const nBlockRows = A.BlockCols();
  auto const nBlockCols = A.BlockRows();
  using Idx = std::remove_const_t<CRIdx>;
  using Ptr = std::remove_const_t<Ptr_>;
  using Scalar = std::remove_const_t<Scalar_>;
  DynamicArray<Ptr> pointers(nBlockRows + 1, 0);
  DynamicArray<Idx> targets;
  targets.resize_noinit(A.NumNonZeroBlocks());
  for (Idx src = 0; src < A.BlockRows(); ++src) {
    for (auto t : A.Indices(src)) {
      ++pointers[t + 1];
    }
  }
  for (int i = 0; i < nBlockRows; ++i) {
    pointers[i + 1] += pointers[i];
  }
  MOCHI_ASSERT_VERBOSE(pointers.back() == Ptr(targets.size()), "Incompatible entries");
  DynamicArray<Scalar> values;
  values.resize_noinit(kBlockSize * kBlockSize * pointers.back());
  DynamicArray<Idx> count(nBlockRows, 0);
  for (Idx src = 0; src < A.BlockRows(); ++src) {
    auto idx = A.Indices(src);
    auto vA = A.Values(src);
    for (Idx j = 0; j < idx.size(); ++j) {
      auto tRow = idx[j];
      auto tRowOffset = pointers[tRow];
      targets[tRowOffset + count[tRow]] = src;
      // Block matrix
      auto ld = pointers[tRow + 1] - pointers[tRow];
      BlockRowView<Scalar, kBlockSize, Idx> r(
          values.data() + kBlockSize * kBlockSize * tRowOffset, ld * kBlockSize, ld);
      r[count[tRow]] = Transpose(vA[j]);
      ++count[tRow];
    }
  }
  return BlockSparseMatrix<Scalar, kBlockSize, Idx, Ptr>(
      nBlockCols, std::move(pointers), std::move(targets), std::move(values));
}

} // namespace mochi
