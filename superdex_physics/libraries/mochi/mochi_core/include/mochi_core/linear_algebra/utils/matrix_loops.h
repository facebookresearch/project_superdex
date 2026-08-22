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
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>

namespace mochi::blocking {

/** @brief Enum designating selections of part(s) of a matrix partitioned into 3 x 3 blocks.
 *
 *     ( A_00 A_01 A_02 )
 * A = ( A_10 A_11 A_12 )
 *     ( A_20 A_21 A_22 )
 */
enum class Part {
  kBefore = 0, //!< @brief Select subpart 0 (in row or column)
  kMiddle = 1, /// Select subpart 1
  kAfter = 2, /// Select subpart 2
  kBeforeAndMiddle = 3, /// Select subparts 0 and 1
  kAfterAndMiddle = 4, /// Select subparts 1 and 2
  kFull = -1, /// Select 0, 1, 2
  kLeft = kBefore, /// Part 0 when used as second index
  kAbove = kBefore, /// Part 0 when used as first index
  kRight = kAfter, /// Part 1 when used as second index
  kBelow = kAfter, /// Part 1 when used as first index
  kLeftWithDiag = kBeforeAndMiddle, /// Parts 0, 1 when used as second index
  kAboveWithDiag = kBeforeAndMiddle, /// Parts 0, 1 when used as first index
  kRightWithDiag = kAfterAndMiddle, /// Parts 1, 2 when used as second index
  kBelowWithDiag = kAfterAndMiddle, /// Parts 1, 2 when used as first index
  kDiag = kMiddle /// Part 1 alias
};

/**
 * @defgroup PartVariables Variables used as argument to PartitionedArg
 * to select sub-parts of matrices..
 * @{
 */

/// Type used to indicate which parts of a matrix partitioned as 3x3 are required.
template <Part rowPar, Part colPart>
struct PartTag {};

/// @brief Argument to request columns left of diagonal block
inline PartTag<Part::kFull, Part::kLeft> Left{};
/// @brief Argument to request columns right of diagonal block
inline PartTag<Part::kFull, Part::kRight> Right{};
/// @brief Argument to request rows above the diagonal block
inline PartTag<Part::kAbove, Part::kFull> Above{};
/// @brief Argument to request rows below the diagonal block
inline PartTag<Part::kBelow, Part::kFull> Below{};
/// @brief Argument to request rows from the diagonal blocks (A_10, A_11, A_12)
inline PartTag<Part::kDiag, Part::kFull> DiagRows{};
/// @brief Argument to request columns from the diagonal blocks (A_01, A_11, A_21)
inline PartTag<Part::kFull, Part::kDiag> DiagCols{};
/// @brief Argument to request the diagonal block A_11
inline PartTag<Part::kDiag, Part::kDiag> DiagBlock{};
/// @brief Argument to request the left columns including the diagonal block
inline PartTag<Part::kFull, Part::kLeftWithDiag> LeftWithDiag{};
/// @brief Argument to request the right columns including the diagonal block
inline PartTag<Part::kFull, Part::kRightWithDiag> RightWithDiag{};
/// @brief Argument to request the top rows including the diagonal block
inline PartTag<Part::kAboveWithDiag, Part::kFull> AboveWithDiag{};
/// @brief Argument to request the bottom rows including the diagonal block
inline PartTag<Part::kBelowWithDiag, Part::kFull> BelowWithDiag{};
/**@}*/

/** @brief Partitioned matrix argument passed to the loop functor. */
template <int kDiagSizeAtCTime, typename MatType>
struct PartitionedArg {
  MatType& A;
  int first;
  int second;
  int full;

  PartitionedArg(MatType& A, int first, int second, int n)
      : A(A), first(first), second(second), full(n) {}

  template <Part kSelect>
  MOCHI_FORCE_INLINE constexpr static int Offset(int first, int second) {
    switch (kSelect) {
      case Part::kBefore:
      case Part::kBeforeAndMiddle:
        return 0;
      case Part::kMiddle:
      case Part::kAfterAndMiddle:
        return first;
      case Part::kAfter:
        return first + second;
      default:
      case Part::kFull:
        return 0;
    }
  }

  template <Part kSelect, bool kForRow>
  MOCHI_FORCE_INLINE constexpr int Length() {
    switch (kSelect) {
      case Part::kBefore:
        return first;
      case Part::kBeforeAndMiddle:
        return first + second;
      case Part::kMiddle:
        return second;
      case Part::kAfter:
        return full - first - second;
      case Part::kAfterAndMiddle:
        return full - first;
      default:
      case Part::kFull:
        return kForRow ? A.Rows() : A.Cols();
    }
  }

  /**
   * @brief Calling the operator() with a single argument, for full columns, full rows or
   * the diagonal block.
   * @details Only the argument type is considered. The argument is otherwise unused.
   * Examples:
   *    A(Left) : returns the columns left of the diagonal
   *    A(BelowWithDiag) : returns the rows from the current first diagonal index to the bottom A
   * @tparam kRowSelect
   * @tparam kColSelect
   * @return
   */
  template <Part kRowSelect, Part kColSelect>
  MOCHI_FORCE_INLINE auto operator()(PartTag<kRowSelect, kColSelect>) {
    constexpr int kRows2AtCT = kRowSelect == Part::kMiddle
        ? kDiagSizeAtCTime
        : (kRowSelect == Part::kFull ? krylov::details::MatTraits<MatType>::kNumRows
                                     : krylov::kDynamic);
    constexpr int kCols2AtCT = kColSelect == Part::kMiddle
        ? kDiagSizeAtCTime
        : (kColSelect == Part::kFull ? krylov::details::MatTraits<MatType>::kNumCols
                                     : krylov::kDynamic);
    return A.template Block<kRows2AtCT, kCols2AtCT>(
        Offset<kRowSelect>(first, second),
        Offset<kColSelect>(first, second),
        Length<kRowSelect, true>(),
        Length<kColSelect, false>());
  }

  /**
   * @brief Calling the operator() with two arguments, the first for selecting sub-rows and the
   * second for selecting sub-columns.
   * @tparam kRowSelect
   * @tparam kColSelect
   * @return
   */
  template <Part kRowSelect, Part kColSelect>
  MOCHI_FORCE_INLINE auto operator()(
      PartTag<kRowSelect, Part::kFull>,
      PartTag<Part::kFull, kColSelect>) {
    return operator()(PartTag<kRowSelect, kColSelect>{});
  }
};

/** @brief Loop in a blocked manner, down the diagonal of a matrix or matrices,
 * passing partitioned matrices as argument to the functor.
 * @details The loop goes down diagonally, putting the top left corner of the 1,1 partition
 * at (i * kBlockSize, i * kBlockSize) for iteration i. The middle matrix is always square
 * with a size of kBlockSize or the remainder (n % kBlockSize). The argument matrices do not need
 * to be square. The loop ends after i for which (i+1) * kBlockSize >= n.
 *
 * @tparam kBlockSize Increment of diagonal indices at each step of the loop.
 * @param n Size of the diagonal, bounding the loop.
 * @param ftor The functor called with the partitioned matrices.
 * @param arg A list of matrices to partition and pass to the functor.
 */
template <int kBlockSize, int kNAtCompileTime = krylov::kDynamic, typename Ftor, typename... Arg>
MOCHI_FORCE_INLINE void PartDown(int n, Ftor&& ftor, Arg&&... arg) {
  int start = 0;
  for (; start + kBlockSize <= n; start += kBlockSize) {
    ftor(PartitionedArg<kBlockSize, Arg>(arg, start, kBlockSize, n)...);
  }
  if constexpr (kNAtCompileTime >= 0) {
    constexpr auto kRemain = kNAtCompileTime % kBlockSize;
    if constexpr (kRemain != 0) {
      ftor(PartitionedArg<kRemain, Arg>(arg, start, kRemain, n)...);
    }
  } else if (start != n) {
    ftor(PartitionedArg<krylov::kDynamic, Arg>(arg, start, n - start, n)...);
  }
}

/** @brief Loop in a blocked manner, up the diagonal of a matrix or matrices,
 * passing partitioned matrices as argument to the functor.
 * @details The loop goes up diagonally, putting the top left corner of the 1,1 partition
 * at (n-(i-1) * kBlockSize, n-(i-1) * kBlockSize) for iteration i, starting from the bottom right.
 * The middle matrix is always square with a size of kBlockSize or the remainder (n % kBlockSize).
 * The argument matrices do not need to be square. The loop ends after the iteration where the
 * top left corner reaches (0, 0).
 *
 * @tparam kBlockSize Increment of diagonal indices at each step of the loop.
 * @param n Size of the diagonal, bounding the loop.
 * @param ftor The functor called with the partitioned matrices.
 * @param arg A list of matrices to partition and pass to the functor.
 */
template <int kBlockSize, int kNAtCompileTime = krylov::kDynamic, typename Ftor, typename... Arg>
MOCHI_FORCE_INLINE void PartUp(int n, Ftor&& ftor, Arg&&... arg) {
  int start = n;
  for (; start - kBlockSize >= 0; start -= kBlockSize) {
    ftor(PartitionedArg<kBlockSize, Arg>(arg, start - kBlockSize, kBlockSize, n)...);
  }
  if constexpr (kNAtCompileTime >= 0) {
    constexpr auto kRemain = kNAtCompileTime % kBlockSize;
    if constexpr (kRemain != 0) {
      ftor(PartitionedArg<kRemain, Arg>(arg, 0, kRemain, n)...);
    }
  } else if (start != 0) {
    ftor(PartitionedArg<krylov::kDynamic, Arg>(arg, 0, start, n)...);
  }
}

} // namespace mochi::blocking
