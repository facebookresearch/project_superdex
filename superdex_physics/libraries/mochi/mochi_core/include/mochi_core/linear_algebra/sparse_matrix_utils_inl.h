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

#include "sparse_matrix_utils.h"

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/graph_views.h>
#include <mochi_core/utils/simd.h>

#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>

namespace mochi::details {

/// @brief Routine to compute the numerical values of the product
///
/// @param[in] A Input sparse matrix
/// @param[in] B Input sparse matrix
/// @param[out] AB Output sparse matrix for the product A * B
///
/// @note The routine does not modify the sparsity pattern of AB.
/// It will ONLY compute entries for the allocated non-zero entries in AB.
///
template <
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
    SparseMatrix<ScalarA, CRIdxA, PtrA, StorageA> const& A,
    SparseMatrix<ScalarB, CRIdxB, PtrB, StorageB> const& B,
    SparseMatrix<ScalarAB, CRIdxAB, PtrAB, StorageAB>& AB) {
  // Performance notes:
  // - 'LoadIndexed' could be used to load 'resultIndices'. Using 'LoadIndexed' is substantially
  //   slower than the current implementation, both on x64 and ARM.
  // - If SIMD size of 8 is available, performance degrades by using it compared to SIMD size of 4.
  //   SIMD size of 4 is thus always used. The case in which SIMD size of 4 is not available (e.g.
  //   double precision on ARM) could perhaps be optimized further.
  static_assert(
      std::is_same_v<ScalarA const, ScalarB const>,
      "Inconsistent scalar types"); // The implementation allows AB to be of different type.
  static_assert(
      std::is_same_v<CRIdxA const, CRIdxB const> && std::is_same_v<CRIdxA const, CRIdxAB const>,
      "Inconsistent integer types");
  using Scalar = std::remove_const_t<ScalarA>;
  using Idx = std::remove_const_t<CRIdxA>;
  constexpr auto kVecSize = 4;
  using V4 = Simd<Scalar, kVecSize>;
  using I4 = Simd<Idx, kVecSize>;
  auto nRows = static_cast<Idx>(A.Rows());
  auto nItems = AB.NumNonZeros();
  if (nItems == 0) {
    return;
  }
  [[maybe_unused]] Idx const dummyFlag = std::numeric_limits<Idx>::max();
  MOCHI_ASSERT_VERBOSE(AB.Cols() < dummyFlag, "Incompatible flag");
  ParallelForRange(
      "SparseMatProduct",
      /* rangeBegin */ 0,
      /* rangeEnd */ nRows,
      // At least 250 non-zeros in AB per task.
      /* minPerTask */ Clamp<Idx>((250 * nRows) / nItems, 1, nRows),
      /* maxPerTask */ nRows,
      [&](Idx rowBegin, Idx rowEnd) {
        [[maybe_unused]] constexpr bool kUseSimd =
            V4::kIsSupported && I4::kIsSupported; // [[maybe_unused]] is a work-around for an
                                                  // erroneous compiler warning from VS2022 + CPP20
    // Offset of nodes in the current row of AB being formed.
#if MOCHI_ASSERT_VERBOSE_ENABLED
        DynamicArray<Idx> ndOffset(AB.Cols(), dummyFlag);
#else
        DynamicArray<Idx> ndOffset;
        ndOffset.resize_noinit(AB.Cols());
#endif
        for (Idx i = rowBegin; i < rowEnd; ++i) {
          auto ABrowIndices = AB.Indices(i);
          auto ABrowValues = AB.Values(i);
          for (Idx k = 0; k < ABrowIndices.size(); ++k) {
            ndOffset[ABrowIndices[k]] = k;
            ABrowValues[k] = 0;
          }
          auto ArowValues = A.Values(i);
          auto ArowIndices = A.Indices(i);
          for (Idx k = 0; k < ArowValues.size(); ++k) {
            auto AcolIdx = ArowIndices[k];
            auto BrowValues = B.Values(AcolIdx);
            auto BrowIndices = B.Indices(AcolIdx);
            Idx j = 0;
#if MOCHI_ASSERT_VERBOSE_ENABLED
            for (; j < BrowIndices.size(); ++j) {
              if (ndOffset[BrowIndices[j]] == dummyFlag ||
                  ndOffset[BrowIndices[j]] >= ABrowIndices.size()) {
                MOCHI_ASSERT_VERBOSE(false, "Out of bounds.");
              }
            }
            j = 0;
#endif
            if constexpr (kUseSimd) {
              V4 Aik = ArowValues[k];
              for (; j + kVecSize <= BrowIndices.size(); j += kVecSize) {
                auto resultIndices =
                    I4{ndOffset[BrowIndices[j + 0]],
                       ndOffset[BrowIndices[j + 1]],
                       ndOffset[BrowIndices[j + 2]],
                       ndOffset[BrowIndices[j + 3]]};
                auto result = Aik * Load<V4>(&BrowValues[j]);
                ABrowValues[Get<0>(resultIndices)] += Get<0>(result);
                ABrowValues[Get<1>(resultIndices)] += Get<1>(result);
                ABrowValues[Get<2>(resultIndices)] += Get<2>(result);
                ABrowValues[Get<3>(resultIndices)] += Get<3>(result);
              }
            }
            for (; j < BrowIndices.size(); ++j) {
              ABrowValues[ndOffset[BrowIndices[j]]] += ArowValues[k] * BrowValues[j];
            }
          }
#if MOCHI_ASSERT_VERBOSE_ENABLED
          for (Idx k = 0; k < ABrowIndices.size(); ++k) {
            // Only for the above check.
            ndOffset[ABrowIndices[k]] = dummyFlag;
          }
#endif
        }
      });
}
} // namespace mochi::details

namespace mochi {

template <
    typename Scalar_,
    typename CRIdx,
    typename Ptr_,
    template <typename, typename...> typename Storage>
auto Transpose(SparseMatrix<Scalar_, CRIdx, Ptr_, Storage> const& A) {
  auto nRows = A.Cols();
  auto nCols = A.Rows();
  using Idx = std::remove_const_t<CRIdx>;
  using Ptr = std::remove_const_t<Ptr_>;
  using Scalar = std::remove_const_t<Scalar_>;
  DynamicArray<Ptr> pointers(nRows + 1, 0);
  DynamicArray<Idx> targets;
  targets.resize_noinit(A.NumNonZeros());
  for (Idx src = 0; src < A.Rows(); ++src) {
    for (auto t : A.Indices(src)) {
      ++pointers[t + 1];
    }
  }
  std::exclusive_scan(
      pointers.begin() + 1, pointers.end(), pointers.begin() + 1, static_cast<Ptr>(0));
  DynamicArray<Scalar> values;
  values.resize_noinit(A.NumNonZeros());
  for (Idx src = 0; src < A.Rows(); ++src) {
    auto const idx = A.Indices(src);
    auto const vA = A.Values(src);
    for (Idx j = 0; j < idx.size(); ++j) {
      auto tRow = idx[j];
      auto tRowOffset = pointers[tRow + 1];
      targets[tRowOffset] = src;
      values[tRowOffset] = vA[j];
      ++pointers[tRow + 1];
    }
  }
  return SparseMatrix<Scalar, Idx, Ptr>(
      nCols, std::move(pointers), std::move(targets), std::move(values));
}

template <
    typename ScalarA,
    typename CRIdxA,
    typename PtrA,
    template <typename, typename...> typename StorageA,
    typename ScalarB,
    typename CRIdxB,
    typename PtrB,
    template <typename, typename...> typename StorageB>
auto operator*(
    SparseMatrix<ScalarA, CRIdxA, PtrA, StorageA> const& A,
    SparseMatrix<ScalarB, CRIdxB, PtrB, StorageB> const& B) {
  static_assert(std::is_same_v<ScalarA const, ScalarB const>, "Inconsistent scalar types");
  static_assert(std::is_same_v<CRIdxA const, CRIdxB const>, "Inconsistent row index types");
  static_assert(std::is_same_v<PtrA const, PtrB const>, "Inconsistent pointer index types");
  MOCHI_ASSERT(A.Cols() == B.Rows(), "Inconsistent matrix dimensions");

  using Scalar = std::remove_const_t<ScalarA>;
  using CRIdx = std::remove_const_t<CRIdxA>;
  using Ptr = std::remove_const_t<PtrA>;
  auto gA = AsGraphView(A);
  auto gB = AsGraphView(B);
  auto gAB = Traverse(gA, gB).SortTargets();
  SparseMatrix<Scalar, CRIdx, Ptr> AB(B.Cols(), std::move(gAB));
  details::SparseMatProduct(A, B, AB);
  return AB;
}

} // namespace mochi
