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

#include <mochi_core/linear_algebra/cuda/cuda_matrix_eval.h>
#include <mochi_core/linear_algebra/host_matrix_eval.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_accessors.h>

#include <type_traits>

namespace mochi {

/** @brief Implementation of the Assignment operator from expression to Matrix.
 *
 * @tparam Scalar
 * @tparam kRowsAtCompileTime
 * @tparam kColsAtCompileTime
 * @tparam kMajorDirection
 * @tparam kOwnership
 * @tparam kLeadingDim
 * @tparam RHS
 * @param rhs
 * @return
 *
 * @note When one matrix in the expression has its data on the device,
 * the assignment is only implemented when the storage orientation matches.
 */
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
template <IsMatrixLike RHS>
Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDirection, kOwnership, kLeadingDim>&
Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDirection, kOwnership, kLeadingDim>::
operator=(RHS const& rhs) {
  if constexpr (IsResizable()) {
    if (rhs.CERows().iVal() != this->CERows().iVal() ||
        rhs.CECols().iVal() != this->CECols().iVal()) {
      this->Resize(rhs.CERows().iVal(), rhs.CECols().iVal());
    }
  } else {
    MOCHI_ASSERT_VERBOSE(
        (this->CERows().iVal() == rhs.CERows().iVal()) &&
            (this->CECols().iVal() == rhs.CECols().iVal()),
        "Dimensions do not match");
  }
  if constexpr (IsMatrix<RHS>) {
    if constexpr ((IsCuda<RHS>) || (krylov::IsCuda(kOwnership))) {
      static_assert(
          kMajorDirection == krylov::details::MatTraits<RHS>::kMajorDir,
          "Different major assignments not implement in CUDA");
      size_t size = rhs.CERows().sVal() * rhs.CECols().sVal();
      // Check if all the data is contiguous on both sides
      auto checkContiguous = [size](auto const& M) {
        auto colStride = M.ColStride().sVal();
        auto rowStride = M.RowStride().sVal();
        return rowStride * (M.Rows() - 1) + colStride * (M.Cols() - 1) == size - 1;
      };
      if (checkContiguous(*this) && checkContiguous(rhs)) {
        constexpr bool scalarSupported =
            ((std::is_same_v<Scalar, double>) || (std::is_same_v<Scalar, float>));
        if constexpr ((IsCuda<RHS>) && (krylov::IsCuda(kOwnership)) && scalarSupported) {
          // Treat case where both pointers are on the device
          mochi::details::CudaDeviceCopy(this->Data(), rhs.Data(), size);
        } else {
          mochi::details::CudaCopy(this->Data(), rhs.Data(), size);
        }
      } else {
        int cudaWidth =
            (kMajorDirection == krylov::Direction::ColMajor) ? this->Rows() : this->Cols();
        int cudaHeight =
            (kMajorDirection == krylov::Direction::ColMajor) ? this->Cols() : this->Rows();
        mochi::details::CudaCopy2D(
            this->Data(), this->LeadDim(), rhs.Data(), rhs.LeadDim(), cudaWidth, cudaHeight);
      }
    } else {
      // The matrices on the left and on the right are not Cuda-based.
      auto dest = details::SetDest(details::GetAccessor(*this));
      auto res = details::Eval(rhs, dest, this->CERows(), this->CECols());
      details::Assign<Scalar>(dest, res, this->CERows(), this->CECols());
    }
  } else {
    // The RHS is an expression.
    auto dest = details::SetDest(details::GetAccessor(*this));
    if constexpr (krylov::IsCuda(kOwnership)) {
      auto res = details::CudaEval(rhs, dest, this->CERows(), this->CECols());
      details::CudaAssign<Scalar>(dest, res, this->CERows(), this->CECols());
    } else {
      auto res = details::Eval(rhs, dest, this->CERows(), this->CECols());
      details::Assign<Scalar>(dest, res, this->CERows(), this->CECols());
    }
  }
  return *this;
}

/** @brief Implementation of the += operator from expression to Matrix.
 *
 * @tparam Scalar
 * @tparam kRowsAtCompileTime
 * @tparam kColsAtCompileTime
 * @tparam kMajorDirection
 * @tparam kOwnership
 * @tparam kLeadingDim
 * @tparam RHSExpr
 * @param rhs
 * @return
 */
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
template <IsMatrixLike RHSExpr>
Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDirection, kOwnership, kLeadingDim>&
Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDirection, kOwnership, kLeadingDim>::
operator+=(RHSExpr const& rhs) {
  MOCHI_ASSERT_VERBOSE(
      (this->CERows().iVal() == rhs.CERows().iVal()) &&
          (this->CECols().iVal() == rhs.CECols().iVal()),
      "Dimensions do not match");
  auto dest = details::SumDest(details::GetAccessor(*this));
  if constexpr (krylov::IsCuda(kOwnership)) {
    auto res = details::CudaEval(rhs, dest, this->CERows(), this->CECols());
    details::CudaAssign<Scalar>(dest, res, this->CERows(), this->CECols());
  } else {
    auto res = details::Eval(rhs, dest, this->CERows(), this->CECols());
    details::Assign<Scalar>(dest, res, this->CERows(), this->CECols());
  }
  return *this;
}

/** @brief Implementation of the -= operator from expression to Matrix.
 *
 * @tparam Scalar
 * @tparam kRowsAtCompileTime
 * @tparam kColsAtCompileTime
 * @tparam kMajorDirection
 * @tparam kOwnership
 * @tparam kLeadingDim
 * @tparam RHSExpr
 * @param rhs
 * @return
 */
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
template <IsMatrixLike RHSExpr>
Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDirection, kOwnership, kLeadingDim>&
Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDirection, kOwnership, kLeadingDim>::
operator-=(RHSExpr const& rhs) {
  MOCHI_ASSERT_VERBOSE(
      (this->CERows().iVal() == rhs.CERows().iVal()) &&
          (this->CECols().iVal() == rhs.CECols().iVal()),
      "Dimensions do not match");
  auto dest = details::SubtractDest(details::GetAccessor(*this));
  if constexpr (krylov::IsCuda(kOwnership)) {
    auto res = details::CudaEval(rhs, dest, this->CERows(), this->CECols());
    details::CudaAssign<Scalar>(dest, res, this->CERows(), this->CECols());
  } else {
    auto res = details::Eval(rhs, dest, this->CERows(), this->CECols());
    details::Assign<Scalar>(dest, res, this->CERows(), this->CECols());
  }
  return *this;
}

/** @brief Implementation of the *= operator by a scalar
 *
 * @tparam Scalar
 * @tparam kRowsAtCompileTime
 * @tparam kColsAtCompileTime
 * @tparam kMajorDirection
 * @tparam kOwnership
 * @tparam kLeadingDim
 * @param[in] alpha
 * @return
 */
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDirection, kOwnership, kLeadingDim>&
Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDirection, kOwnership, kLeadingDim>::
operator*=(Scalar alpha) {
  *this = alpha * (*this);
  return *this;
}

/** @brief Implementation of the /= operator by a scalar
 *
 * @tparam Scalar
 * @tparam kRowsAtCompileTime
 * @tparam kColsAtCompileTime
 * @tparam kMajorDirection
 * @tparam kOwnership
 * @tparam kLeadingDim
 * @param[in] alpha
 * @return
 */
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDirection, kOwnership, kLeadingDim>&
Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDirection, kOwnership, kLeadingDim>::
operator/=(Scalar alpha) {
  MOCHI_ASSERT_VERBOSE(alpha != Scalar(0), "Division by zero")
  auto invAlpha = Scalar(1) / alpha;
  *this = invAlpha * (*this);
  return *this;
}

} // namespace mochi
