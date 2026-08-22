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

#include <mochi_core/linear_algebra/factor_kernels.h>
#include <mochi_core/linear_algebra/ldlt.h>
#include <mochi_core/linear_algebra/multi_frontal/front_matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/stair_matrix.h>

namespace mochi {

/** @brief Object to work on fronts and L.
 *
 * The object allocates a temporary buffer to be used to speed up operations.
 *
 * @tparam Scalar
 * @tparam kBufferRows
 */
template <typename Scalar, size_t kBufferRows>
class FrontManipulator {
 public:
  explicit FrontManipulator(size_t bufferRows)
      : _bufferSize(kBufferRows * bufferRows), _uBuffer(new Scalar[_bufferSize]) {}

  /** @brief Compute L and D^-1 in place for a set of columns.
   * @param [inout] stairLD The columns for which L and D^-1 are computed.
   */
  template <size_t kStairColBlock>
  int ToLD(StairMatrixView<Scalar, kStairColBlock>& stairLD);

  /** @brief Compute the Schur-complement contribution from the elimination of a given super-node.
   *
   * @details In what follows, the front matrix $f F $F represents the assembled front from
   * the elimination of all the children of the given supernode.
   * It is decomposed into a diagonal block $f A $f
   * of size numEq x numEq, a block $f B $f under $f A $f and a diagonal block $f R $f where the
   * Schur complement update will be performed. Given this, a factorization $f A = L_A D_A L_A^T $f
   * is first performed. Then $fL_B = B L_A^{-T}$f is computed, to facilitate the computation of the
   * Schur complement. The factorization of $f A $f and computation of $f L_B $f is performed before
   * calling this method. The result these operation is contained in `stairLD`. `R` is
   * contained in `front` and its update is performed with the following formula:
   * $fS = R - B A^{-1} B = R - B L_A D^{-1} L_A^T = R - L_B D^{-1} L_B^T $f.
   *
   * @param stairLD The columns of a given supernode.
   * @param front Front object of a given supernode.
   * @param isLeaf
   */
  template <size_t kStairColBlock, size_t kColumnBlock, bool kPackSmall>
  void RankNUpdate(
      StairMatrixView<Scalar, kStairColBlock>& stairLD,
      Front<kColumnBlock, kPackSmall>& front,
      bool isLeaf);

 private:
  size_t _bufferSize;
  /// @brief A buffer that should be as large as the maximum number
  /// of columns in a block of L times kBufferRows.
  std::unique_ptr<Scalar[]> _uBuffer;
};

// TODO kSizeAtCT, used to save on one test and code size if it is fixed. Make use of it where
// possible
template <typename Scalar, int kBlockSize = kLDLtDefaultBlockSize, int kSizeAtCT = kDynamic>
void ApplyLmTOnRight(auto const& L, auto& X) {
  auto size = L.Rows();
  using namespace blocking;
  auto Xin = X.Transpose();
  PartDown<kBlockSize, kSizeAtCT>(
      size,
      [](auto&& X, auto&& L) {
        auto currentX = X(DiagRows);
        auto prevX = X(Above);
        if (prevX.Rows() > 0) {
          currentX -= L(DiagRows, Left) * prevX;
        }
        kernel::ApplyLm1OnLeft(L(DiagBlock), currentX);
      },
      Xin,
      L);
}

template <typename Scalar, size_t kBufferRows>
template <size_t kStairColBlock>
int FrontManipulator<Scalar, kBufferRows>::ToLD(StairMatrixView<Scalar, kStairColBlock>& stairLD) {
  size_t lBlockCount = stairLD.NumBlocks();
  for (size_t workIdx = 0; workIdx < lBlockCount; ++workIdx) {
    auto workBlock = stairLD.Block(workIdx);
    // Left loking update
    for (size_t ib = 0; ib < workIdx; ++ib) {
      auto leftPanel = stairLD.Block(ib);
      auto leftBlock = leftPanel.BottomRows(workBlock.Rows());
      auto leftD = leftPanel.TopRows(leftPanel.Cols());
      MOCHI_ASSERT_VERBOSE(
          kStairColBlock * workBlock.Cols() <= _bufferSize, "Buffer size too small.");
      auto U = MatrixView<Scalar, kStairColBlock, kDynamic>(
          this->_uBuffer.get(), kStairColBlock, workBlock.Cols());
      U = leftBlock.TopRows(workBlock.Cols()).Transpose();
      for (int r = 0; r < kStairColBlock; ++r) {
        U.Row(r) *= (Scalar{1} / leftD(r, r)); // U = D L^T
      }
      workBlock -= leftBlock * U;
    }
    auto Dblock = workBlock.TopRows(workBlock.Cols());
    // TODO: When singularity detection approach has been decided, add handling logic here.
    auto checkSingularities = [](int, Scalar /* pivot */, auto const& /* eqIndex */) {
      return false;
    };
    auto singularities = LDLtFactorize(Dblock, true, checkSingularities);
    if (singularities != 0) {
      return singularities;
    }
    auto Lpart = workBlock.BottomRows(workBlock.Rows() - workBlock.Cols());
    ApplyLmTOnRight<Scalar>(Dblock, Lpart);
    for (int c = 0; c < workBlock.Cols(); ++c) {
      Lpart.Col(c) *= Dblock(c, c);
    }
  }
  return 0;
}

template <typename Scalar, size_t kBufferRows>
template <size_t kStairColBlock, size_t kColumnBlock, bool kPackSmall>
void FrontManipulator<Scalar, kBufferRows>::RankNUpdate(
    StairMatrixView<Scalar, kStairColBlock>& stairLD,
    Front<kColumnBlock, kPackSmall>& front,
    bool isLeaf) {
  MOCHI_ASSERT_VERBOSE(
      stairLD.Rows() - stairLD.Cols() == front.Size(),
      "stairLD and front have incompatible sizes.");
  bool firstUpdate = isLeaf;
  for (auto lBlock : stairLD.Blocks()) {
    for (auto schurBlock : front.template Blocks<Scalar>()) {
      auto schurRows = schurBlock.Rows();
      auto schurCols = schurBlock.Cols();
      auto subL = lBlock.BottomRows(schurRows);
      auto lForU = subL.TopRows(schurCols);
      // Form U in temporary buffer
      MOCHI_ASSERT_VERBOSE(
          lBlock.Cols() * schurBlock.Cols() <= _bufferSize, "Buffer size too small.");
      RowMatrixView<Scalar> U(_uBuffer.get(), lBlock.Cols(), schurBlock.Cols());
      for (int c = 0; c < subL.Cols(); c++) {
        U.Row(c) = (Scalar(1) / lBlock(c, c)) * lForU.Col(c).Transpose();
      }
      // Performance note: The average dimensionality of the update is small, leading to inefficient
      // matrix-matrix products. Assess using different storage directions and matrix-matrix
      // kernels.
      if (firstUpdate) { // Leaves get uninitialized stack space.
        schurBlock = -subL * U;
      } else {
        schurBlock -= subL * U;
      }
    }
    firstUpdate = false;
  }
}

} // namespace mochi
