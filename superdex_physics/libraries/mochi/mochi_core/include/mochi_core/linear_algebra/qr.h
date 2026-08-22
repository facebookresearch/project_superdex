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

#include <limits>
#include <type_traits>

namespace mochi {

enum class QrAlgorithm {
  CGS, // Classical Gram-Schmidt. Unstable. Use only for small, well-conditioned matrices.
  MGS, // Modified Gram-Schmidt.
  ICGS, // Iterative classical Gram-Schmidt. EXPERIMENTAL: The convergence criterion has not been
        // tuned and may lead to more iterations than necessary.
  Count,
  Default = QrAlgorithm::MGS
};

/// @brief Class to compute the thin (aka reduced) QR factorization of a thin (or square) matrix A.
/// @remark Only Gram-Schmidt algorithms are implemented. More stable algorithms will be implemented
/// on an as-needed basis.
template <
    typename Scalar,
    int kRowsAtCT = krylov::kDynamic,
    int kColsAtCT = krylov::kDynamic,
    QrAlgorithm kAlgorithm = QrAlgorithm::Default>
class ThinQR {
 public:
  using NonConstScalar = std::remove_const_t<Scalar>;

  /// @brief Constructor for the factorization.
  /// @param[in] A Matrix to factorize.
  template <
      typename ScalarA,
      krylov::Direction kMajorDir,
      krylov::Ownership kOwnership,
      int kLeadDim>
  ThinQR(Matrix<ScalarA, kRowsAtCT, kColsAtCT, kMajorDir, kOwnership, kLeadDim> const& A)
      : _Q(A.Rows(), A.Cols()), _R(A.Cols(), A.Cols()) {
    MOCHI_ASSERT(A.Rows() >= A.Cols(), "Thin QR factorization requires a thin (or square) matrix.");

    _R.SetZero();
    for (int j = 0; j < A.Cols(); ++j) {
      auto Qcol = _Q.Col(j); // View
      Qcol = A.Col(j);
      if constexpr (kAlgorithm == QrAlgorithm::MGS) {
        for (int i = 0; i < j; ++i) {
          _R(i, j) = _Q.Col(i).Dot(Qcol);
          Qcol -= _R(i, j) * _Q.Col(i);
        }
      } else if constexpr (kAlgorithm == QrAlgorithm::CGS) {
        auto rCoeffs = _R.template Block<krylov::kDynamic, 1>(0, j, j, 1);
        rCoeffs = _Q.LeftCols(j).Transpose() * Qcol;
        Qcol -= _Q.LeftCols(j) * rCoeffs;
      } else {
        static_assert(kAlgorithm == QrAlgorithm::ICGS, "Unsupported QR algorithm");
        constexpr int kMaxIters = 3;
        constexpr auto kTolerance = std::numeric_limits<Scalar>::epsilon();
        auto rCoeffs = _R.template Block<krylov::kDynamic, 1>(0, j, j, 1);
        ColumnVector<NonConstScalar> rCoeffsCorrection(j);
        int iter = 0;
        do {
          rCoeffsCorrection = _Q.LeftCols(j).Transpose() * Qcol;
          rCoeffs += rCoeffsCorrection;
          Qcol -= _Q.LeftCols(j) * rCoeffsCorrection;
        } while (++iter < kMaxIters && rCoeffsCorrection.Norm() > kTolerance);
      }

      _R(j, j) = Qcol.Norm();
      Qcol /= (_R(j, j) + std::numeric_limits<Scalar>::min());
    }
  }

  /// @brief Const view of the thin (aka reduced) Q factor.
  /// @note Stored as column-major.
  [[nodiscard]] auto Q() const {
    return AsConstView(_Q);
  }

  /// @brief Const view of the reduced R factor.
  /// @note Stored as column-major.
  [[nodiscard]] auto R() const {
    return AsConstView(_R);
  }

 protected:
  Matrix<NonConstScalar, kRowsAtCT, kColsAtCT> _Q;
  Matrix<NonConstScalar, kColsAtCT, kColsAtCT> _R;
};

} // namespace mochi
