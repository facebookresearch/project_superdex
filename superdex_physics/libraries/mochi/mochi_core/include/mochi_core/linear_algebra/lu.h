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
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/linear_algebra/utils/matrix_loops.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/simd.h>

#include <cmath>
#include <type_traits>

namespace mochi {

using krylov::Direction;
using krylov::kDynamic;
using krylov::Ownership;

/// @brief Default block size for the LU factorization.
/// @note The current default of 24 is consistent with the default for the LDLt factorization, which
/// has been tuned on several architectures. See LDLt documentation for details.
constexpr int kLUDefaultBlockSize = 24;

/// @brief Batch of work size for some operations of the LU factorization. The block size must be a
/// multiple of this value.
/// @note The current default of 6 is consistent with the default for the LDLt factorization, which
/// has been tuned on several architectures. See LDLt documentation for details.
constexpr int kLUDefaultTrSize = 6;
static_assert(kLUDefaultBlockSize % kLUDefaultTrSize == 0);

/// @brief Recommended matrix size threshold to prefer @ref SmallLU over the default @ref LU for the
/// inverse operation.
///
/// @note Empirically consistent with @ref kSmallLDLtInverseSizeThreshold. See documentation there
/// for benchmark details and rationale.
constexpr int kSmallLUInverseSizeThreshold = 100;

enum class PermuteAlg { None, PartialRow, Rook, Count };

constexpr std::integral_constant<PermuteAlg, PermuteAlg::None> NoPermutation;
constexpr std::integral_constant<PermuteAlg, PermuteAlg::PartialRow> PartialPermutation;
constexpr std::integral_constant<PermuteAlg, PermuteAlg::Rook> RookPermutation;

/**
 * @brief Implementation of a permutation algorithm.
 * @details Permuter<kPermAlg> must implement:
 *
 * Obtain a permutation handling functor.
 *    auto GetPermuter(M& FullMat, int offset) -> bool(MatrixBlock, int offset)
 * It is given the full matrix so that the permuter can
 * apply the permutation to the full matrix. It is expected to find pivots only within
 * a diagonal block whose offset within the full matrix is passed along.
 * The functor returns false if the factorization should be aborted and true otherwise. This can
 * allow for more complex recovery if the block diagonal matrix is either singular
 * or very ill-conditioned and a pivot should be sought outside of the diagonal block.
 *
 * apply P on the left, before applying U^-1 L^-1:
 *    void ApplyPLeft(M&& x)
 * apply Q on the left, after applying U^-1 L^-1.
 *    void ApplyQLeft(M&& x)
 * apply P on the right, after applying U^-1 L^-1:
 *    void ApplyPright(M&& x)
 * apply Q on the right, before applying U^-1 L^-1:
 *    void ApplyQright(M&& x)
 *
 * Get the number of permutations
 *    uint32_t PermutationCount() const
 * @tparam kPermAlg Permutation algorithm
 */
template <PermuteAlg kPermAlg = PermuteAlg::None>
struct Permuter {
  explicit Permuter(int) {}

  template <typename FM, typename BT>
  auto GetPermuter(FM&&, BT&&) {
    return [](auto&&, auto&&) { return true; };
  }

  template <typename M>
  void ApplyPLeft(M&&) const {}

  template <typename M>
  void ApplyQLeft(M&&) const {}

  template <typename M>
  void ApplyQRight(M&&) const {}

  template <typename M>
  void ApplyPRight(M&&) const {}

  /// @brief Get the number of permutation (unsigned for bit-wise operations)
  [[nodiscard]] uint32_t PermutationCount() const {
    return 0;
  }
};

/// @brief Class to compute a (block) LU^T factorization of a matrix A.
///
/// @note It does not use pivoting by default. To enable Rook pivoting, use kPermAlg =
/// @ref PermuteAlg::Rook. To enable partial row pivoting, use kPermAlg = @ref
/// PermuteAlg::PartialRow.
/// @note For matrix sizes under 50-100, prefer the alias @ref SmallLU that uses a smaller block
/// size.
/// @note If the number of rows and/or columns is a compile-time value, prefer LU specializations
/// with compile-time rows and/or columns to improve performance, that is, prefer
///    Matrix<real, 3, 3> A;
///    LU<real, 3, 3> lu(A);
/// over
///    Matrix<real, 3, 3> A;
///    LU<real> lu(A);
template <
    typename InputScalar,
    int kRowsAtCT = kDynamic,
    int kColsAtCT = kDynamic,
    PermuteAlg kPermAlg = PermuteAlg::None,
    int kBlockSize = kLUDefaultBlockSize,
    int kTrSize = kLUDefaultTrSize>
class LU {
 public:
  using Scalar = std::remove_const_t<InputScalar>;
  static constexpr int kBlockSz = kBlockSize;
  static_assert(kBlockSize % kTrSize == 0, "Block size must be multiple of kTrSize");
  static_assert(kPermAlg < PermuteAlg::Count, "Unsupported permutation algorithm");
  static_assert(
      kRowsAtCT == kDynamic || kColsAtCT == kDynamic || kRowsAtCT == kColsAtCT,
      "LU requires a square matrix");
  static constexpr int kSizeAtCT = Max(kRowsAtCT, kColsAtCT);

  /// @brief Mantissa-exponent representation for determinant computation.
  struct MantissaExponent {
    Scalar mantissa;
    int exponent;
  };

  /// @brief Constructor for the factorization.
  ///
  /// @param[in] A     Matrix to factorize.
  template <typename ScalarA, Direction kMajorDirection, Ownership kOwnership, int kLeadDim>
  LU(Matrix<ScalarA, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadDim> const& A);

  /// @brief Constructor for use in CTAD expressions
  ///
  /// @param[in] A     Matrix to factorize.
  template <
      typename ScalarA,
      int kR,
      int kC,
      Direction kMajorDirection,
      Ownership kOwnership,
      int kLeadDim>
  LU(std::integral_constant<PermuteAlg, kPermAlg>,
     Matrix<ScalarA, kR, kC, kMajorDirection, kOwnership, kLeadDim> const& A)
      : LU(A) {}

  /// @brief Solve in place linear systems X <- X (LU)^{-1}
  /// @note If X has multiple columns, performance is better if it's stored row-major.
  template <
      typename ScalarX,
      int kRowsX,
      int kColsX,
      Direction kDirectionX,
      Ownership kOwnershipX,
      int kLeadDimX>
  void LeftSolveInPlace(
      Matrix<ScalarX, kRowsX, kColsX, kDirectionX, kOwnershipX, kLeadDimX>& Xin) const;

  /// @brief Solve in place linear systems X <- X (LU)^{-1}
  /// @note If X has multiple rows, performance is better if it's stored column-major.
  template <
      typename ScalarX,
      int kRowsX,
      int kColsX,
      Direction kDirectionX,
      Ownership kOwnershipX,
      int kLeadDimX>
  void RightSolveInPlace(
      Matrix<ScalarX, kRowsX, kColsX, kDirectionX, kOwnershipX, kLeadDimX>& Xin) const;

  /// @brief Compute the inverse and store it in the matrix passed as input.
  template <
      typename ScalarX,
      int kRowsX,
      int kColsX,
      Direction kDirectionX,
      Ownership kOwnershipX,
      int kLeadDimX>
  void Inverse(Matrix<ScalarX, kRowsX, kColsX, kDirectionX, kOwnershipX, kLeadDimX>& invA) const;

  /// @brief Determinant of the matrix in mantissa-exponent representation.
  /// @note Computed on demand from the diagonal of the factorization.
  MantissaExponent Determinant() const {
    // Note that it would be possible to use bitwise SIMD operations instead of std::frexp.
    Scalar mantissa{1};
    int exponent{0};
    for (int i = 0; i < _LU.Rows(); ++i) {
      int expn MOCHI_NO_INIT;
      mantissa = std::frexp(mantissa * _LU(i, i), &expn);
      exponent += expn;
    }
    if (mantissa == Scalar{0}) {
      // A zero mantissa means the factorization has a zero pivot on the diagonal (stored as 0
      // because the reciprocal is skipped when diag == 0), i.e., the matrix is singular.
      return {Scalar{0}, 0};
    }
    int expn MOCHI_NO_INIT;
    Scalar sign = (_permuter.PermutationCount() & 1) ? Scalar{-1} : Scalar{1};
    mantissa = std::frexp(sign / mantissa, &expn);
    return {mantissa, expn - exponent};
  }

  /// @brief Determinant of the matrix.
  /// @note Computed on demand. It may overflow for large matrices. Please use @ref Determinant to
  /// get a mantissa-exponent representation without overflow.
  Scalar ScalarDeterminant() const {
    auto determinant = Determinant();
    Scalar const det = std::ldexp(determinant.mantissa, determinant.exponent);
    if (!IsFinite(det)) {
      MOCHI_LOG_WARNING(
          "Determinant overflow. Please use LU::Determinant instead of LU::ScalarDeterminant "
          "to get a mantissa-exponent representation without overflow.");
    }
    return det;
  }

  auto const& GetLU() const {
    return _LU;
  }

 protected:
  void Factorize();

 protected:
  /// Matrix size.
  int _size = 0;
  /// LU factors.
  Matrix<Scalar, kSizeAtCT, kSizeAtCT, Direction::ColMajor> _LU;
  /// Permutation object. TODO: For C++20, add [[no_unique_address]]
  Permuter<kPermAlg> _permuter;
};

template <
    typename ScalarA,
    int kRowsAtCT,
    int kColsAtCT,
    PermuteAlg kPermAlg,
    Direction kMajorDirection,
    Ownership kOwnership,
    int kLeadDim>
LU(std::integral_constant<PermuteAlg, kPermAlg>,
   Matrix<ScalarA, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadDim> const& A)
    -> LU<ScalarA, kRowsAtCT, kColsAtCT, kPermAlg>;

template <
    typename InputScalar,
    int kRowsAtCT,
    int kColsAtCT,
    PermuteAlg kPermAlg,
    int kBlockSize,
    int kTrSize>
template <typename ScalarA, Direction kMajorDirection, Ownership kOwnership, int kLeadDim>
LU<InputScalar, kRowsAtCT, kColsAtCT, kPermAlg, kBlockSize, kTrSize>::LU(
    Matrix<ScalarA, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadDim> const& A)
    : _size(A.Rows()), _LU(A), _permuter(A.Rows()) {
  static_assert(std::is_same_v<InputScalar const, ScalarA const>, "Incompatible scalar types");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  Factorize();
}

template <
    typename InputScalar,
    int kRowsAtCT,
    int kColsAtCT,
    PermuteAlg kPermAlg,
    int kBlockSize,
    int kTrSize>
void LU<InputScalar, kRowsAtCT, kColsAtCT, kPermAlg, kBlockSize, kTrSize>::Factorize() {
  using namespace blocking;
  PartDown<kBlockSize, kSizeAtCT>(
      _LU.Rows(),
      [this, block = 0](auto&& LU) mutable {
        auto D = LU(DiagBlock);
        auto permuter = _permuter.GetPermuter(_LU, block);
        kernel::FlexibleUnblockedPivotFactor(D, permuter);
        block += D.Rows();
        if (block < _LU.Rows()) {
          auto b = LU(DiagRows, Right);
          auto c = LU(Below, DiagCols);
          auto r = LU(Below, Right);
          kernel::ApplyLm1OnLeft(D, b);
          kernel::ApplyUm1OnRight(D, c);
          r -= c * b;
        }
      },
      _LU);
}

template <
    typename InputScalar,
    int kRowsAtCT,
    int kColsAtCT,
    PermuteAlg kPermAlg,
    int kBlockSize,
    int kTrSize>
template <
    typename ScalarX,
    int kRowsX,
    int kColsX,
    Direction kDirectionX,
    Ownership kOwnershipX,
    int kLeadDimX>
void LU<InputScalar, kRowsAtCT, kColsAtCT, kPermAlg, kBlockSize, kTrSize>::LeftSolveInPlace(
    Matrix<ScalarX, kRowsX, kColsX, kDirectionX, kOwnershipX, kLeadDimX>& Xin) const {
  //
  // If B is the right-hand side, this routine computes X = A^{-1} B = Q U^{-1} L^{-1} P B
  //
  MOCHI_ASSERT_VERBOSE(Xin.Rows() == _size, "Dimensions do not match.");
  using namespace blocking;

  _permuter.ApplyPLeft(Xin);

  // Apply L^-1 in blocks.
  PartDown<kBlockSize, kSizeAtCT>(
      _LU.Rows(),
      [](auto&& X, auto&& L) {
        auto currentX = X(DiagRows);
        auto prevX = X(Above);
        if (prevX.Rows() > 0) {
          currentX -= L(DiagRows, Left) * prevX;
        }
        kernel::ApplyLm1OnLeft(L(DiagBlock), currentX);
      },
      Xin,
      _LU);

  // Apply U^-1 in blocks.
  PartUp<kBlockSize, kSizeAtCT>(
      _LU.Rows(),
      [](auto&& X, auto&& U) {
        auto currentX = X(DiagRows);
        auto prevX = X(Below);
        if (prevX.Rows() > 0) {
          currentX -= U(DiagRows, Right) * prevX;
        }
        kernel::ApplyUm1OnLeft(U(DiagBlock), currentX);
      },
      Xin,
      _LU);

  _permuter.ApplyQLeft(Xin);
}

template <
    typename InputScalar,
    int kRowsAtCT,
    int kColsAtCT,
    PermuteAlg kPermAlg,
    int kBlockSize,
    int kTrSize>
template <
    typename ScalarX,
    int kRowsX,
    int kColsX,
    Direction kDirectionX,
    Ownership kOwnershipX,
    int kLeadDimX>
void LU<InputScalar, kRowsAtCT, kColsAtCT, kPermAlg, kBlockSize, kTrSize>::RightSolveInPlace(
    Matrix<ScalarX, kRowsX, kColsX, kDirectionX, kOwnershipX, kLeadDimX>& Xin) const {
  // Assume A has been transformed as P A Q = LU i.e. A = P^-1 A Q^-1 = P^-1 L U Q^-1
  // X = B A^{-1} = B Q U^-1 L^-1 P
  MOCHI_ASSERT_VERBOSE(Xin.Cols() == _size, "Dimensions do not match.");
  using namespace blocking;

  _permuter.ApplyQRight(Xin);

  // Apply U^-1 in blocks.
  PartDown<kBlockSize, kSizeAtCT>(
      _LU.Rows(),
      [](auto&& X, auto&& U) {
        auto currentX = X(DiagCols);
        auto prevX = X(Left);
        if (prevX.Cols() > 0) {
          currentX -= prevX * U(Above, DiagCols);
        }
        kernel::ApplyUm1OnRight(U(DiagBlock), currentX);
      },
      Xin,
      _LU);

  // Apply L^-1 in blocks.
  PartUp<kBlockSize, kSizeAtCT>(
      _LU.Rows(),
      [](auto&& X, auto&& L) {
        auto currentX = X(DiagCols);
        auto prevX = X(Right);
        if (prevX.Cols() > 0) {
          currentX -= prevX * L(Below, DiagCols);
        }
        kernel::ApplyLm1OnRight<kTrSize, kBlockSize>(L(DiagBlock), currentX);
      },
      Xin,
      _LU);

  _permuter.ApplyPRight(Xin);
}

template <
    typename InputScalar,
    int kRowsAtCT,
    int kColsAtCT,
    PermuteAlg kPermAlg,
    int kBlockSize,
    int kTrSize>
template <
    typename ScalarX,
    int kRowsX,
    int kColsX,
    Direction kDirectionX,
    Ownership kOwnershipX,
    int kLeadDimX>
void LU<InputScalar, kRowsAtCT, kColsAtCT, kPermAlg, kBlockSize, kTrSize>::Inverse(
    Matrix<ScalarX, kRowsX, kColsX, kDirectionX, kOwnershipX, kLeadDimX>& invA) const {
  MOCHI_ASSERT_VERBOSE(
      (invA.Rows() == _size) && (invA.Cols() == _size), "Dimensions do not match.");
  using namespace blocking;

  // Set input matrix to the identity.
  invA.SetIdentity();

  if constexpr (kDirectionX == Direction::ColMajor) {
    // If input is col-major, use right solves to operate on blocks of columns.
    _permuter.ApplyQRight(invA);

    // Apply U^-1 in blocks (mostly) to the right part.
    PartDown<kBlockSize, kSizeAtCT>(
        _LU.Rows(),
        [](auto&& X, auto&& U) {
          auto currentX = X(AboveWithDiag, DiagCols);
          kernel::ApplyUm1OnRight(U(DiagBlock), currentX);
          // Propagate the values to the right.
          X(AboveWithDiag, Right) -= currentX * U(DiagRows, Right);
        },
        invA,
        _LU);

    // Apply L^-1 in blocks.
    PartUp<kBlockSize, kSizeAtCT>(
        _LU.Rows(),
        [](auto&& X, auto&& L) {
          auto currentX = X(DiagCols);
          auto prevX = X(Right);
          if (prevX.Cols() > 0) {
            currentX -= prevX * L(Below, DiagCols);
          }
          kernel::ApplyLm1OnRight<kTrSize, kBlockSize>(L(DiagBlock), currentX);
        },
        invA,
        _LU);

    _permuter.ApplyPRight(invA);
  } else {
    // If input is row-major, use left solves to operate on blocks of rows.
    static_assert(kDirectionX == Direction::RowMajor, "Unexpected storage direction");
    _permuter.ApplyPLeft(invA);

    // Apply L^-1 in blocks (mostly) to the lower part.
    PartDown<kBlockSize, kSizeAtCT>(
        _LU.Rows(),
        [](auto&& X, auto&& L) {
          auto currentX = X(DiagRows, LeftWithDiag);
          kernel::ApplyLm1OnLeft(L(DiagBlock), currentX);
          // Propagate the values below.
          X(Below, LeftWithDiag) -= L(Below, DiagCols) * currentX;
        },
        invA,
        _LU);

    // Apply U^-1 in blocks.
    PartUp<kBlockSize, kSizeAtCT>(
        _LU.Rows(),
        [](auto&& X, auto&& U) {
          auto currentX = X(DiagRows);
          auto prevX = X(Below);
          if (prevX.Rows() > 0) {
            currentX -= U(DiagRows, Right) * prevX;
          }
          kernel::ApplyUm1OnLeft(U(DiagBlock), currentX);
        },
        invA,
        _LU);

    _permuter.ApplyQLeft(invA);
  }
}

/// @brief LU specialization with smaller block sizes. Prefer it over the default LU for matrix
/// sizes up to 50-200.
/// @note The block size and batch of work size of 8 is optimal (or near optimal) for matrix sizes
/// up to ~100, both on x64 and ARM.
/// @note The optimal matrix size threshold to favor @ref SmallLU over the default LU depends on the
/// scalar type, operation, storage direction, and architecture. For the factorization, the optimal
/// threshold is often smaller than ~100. For the solve with a single vector, the optimal threshold
/// is often substantially larger than 100.
/// @note The notes above refer to single-precision arithmetic. Performance has not been tuned in
/// double precision yet.
template <
    typename InputScalar,
    int kRowsAtCT = kDynamic,
    int kColsAtCT = kDynamic,
    PermuteAlg kAlg = PermuteAlg::None>
using SmallLU = LU<InputScalar, kRowsAtCT, kColsAtCT, kAlg, 8, 8>;

void SwapRows(IsMatrix auto&& r1, IsMatrix auto&& r2) {
  using namespace std;
  for (int c = 0; c < r1.Cols(); ++c) {
    swap(r1[c], r2[c]);
  }
}

void SwapCols(IsMatrix auto&& r1, IsMatrix auto&& r2) {
  using namespace std;
  for (int c = 0; c < r1.Rows(); ++c) {
    swap(r1[c], r2[c]);
  }
}

template <>
struct Permuter<PermuteAlg::PartialRow> {
  explicit Permuter(int n) {
    Pvt.reserve(n);
  }

  template <typename M>
  auto GetPermuter(M& FA, int offset) {
    return [this, &FA, offset](auto&& A, int k) {
      auto v = Abs(A(k, k));
      int iMax = k;
      for (int i = k + 1; i < A.Rows(); ++i) {
        auto w = Abs(A(i, k));
        if (w > v) {
          v = w;
          iMax = i;
        }
      }
      int gIdx = iMax + offset;
      if (iMax != k) {
        ++permCount;
        SwapRows(FA.Row(k + offset), FA.Row(gIdx));
      }
      Pvt.push_back(gIdx);
      return true;
    };
  }

  template <typename M>
  void ApplyPLeft(M&& x) const {
    for (int i = 0; i < x.Rows(); ++i) {
      SwapRows(x.Row(i), x.Row(Pvt[i]));
    }
  }

  template <typename M>
  void ApplyQLeft(M&&) const {}

  template <typename M>
  void ApplyQRight(M&&) const {}

  template <typename M>
  void ApplyPRight(M&& x) const {
    // Applying P on the left starts with the last permutation
    for (int i = x.Cols(); i-- != 0;) {
      SwapCols(x.Col(i), x.Col(Pvt[i]));
    }
  }

  /// @brief Get the number of permutation (unsigned for bit-wise operations)
  [[nodiscard]] uint32_t PermutationCount() const {
    return permCount;
  }

  /// @brief Permutations of P. \f$ P = P_n * P_{n-1} * ... * P_1 \f$
  DynamicArray<int> Pvt;
  uint32_t permCount = 0;
};

template <>
struct Permuter<PermuteAlg::Rook> {
  explicit Permuter(int n) {
    Pvt.reserve(n);
    Qvt.reserve(n);
  }

  template <typename M>
  auto GetPermuter(M& FA, int offset) {
    return [this, &FA, offset](auto&& A, int k) {
      int br = k;
      int bc = k;
      if (std::isnan(A(k, k))) {
        Pvt.push_back(k + offset);
        Qvt.push_back(k + offset);
        return true;
      }
      auto bv = Abs(A(br, bc));
      int n = A.Rows();
      int nochanges = 0;
      while (nochanges != 2) {
        auto obv = bv;
        for (int r = k; r < n; ++r) {
          auto w = Abs(A(r, bc));
          if (w > bv) {
            bv = w;
            br = r;
          }
        }
        nochanges = (bv == obv) ? nochanges + 1 : 0;
        if (nochanges == 2) {
          break;
        }
        obv = bv;
        for (int c = k; c < n; ++c) {
          auto w = Abs(A(br, c));
          if (w > bv) {
            bv = w;
            bc = c;
          }
        }
        nochanges = (bv == obv) ? nochanges + 1 : 0;
        if (nochanges == 2) {
          break;
        }
      }
      if (br != k) {
        ++permCount;
        SwapRows(FA.Row(k + offset), FA.Row(br + offset));
      }
      if (bc != k) {
        ++permCount;
        SwapCols(FA.Col(k + offset), FA.Col(bc + offset));
      }
      Pvt.push_back(br + offset);
      Qvt.push_back(bc + offset);
      return true;
    };
  }

  template <typename M>
  void ApplyPLeft(M&& x) const {
    for (int i = 0; i < x.Rows(); ++i) {
      if (i != Pvt[i]) {
        SwapRows(x.Row(i), x.Row(Pvt[i]));
      }
    }
  }

  template <typename M>
  void ApplyQLeft(M&& x) const {
    for (int i = x.Rows(); i-- != 0;) {
      if (i != Qvt[i]) {
        SwapRows(x.Row(i), x.Row(Qvt[i]));
      }
    }
  }

  template <typename M>
  void ApplyQRight(M&& x) const {
    for (int i = 0; i < x.Cols(); ++i) {
      if (i != Qvt[i]) {
        SwapCols(x.Col(i), x.Col(Qvt[i]));
      }
    }
  }

  template <typename M>
  void ApplyPRight(M&& x) const {
    // Applying P on the left starts with the last permutation
    for (int i = x.Cols(); i-- != 0;) {
      if (i != Pvt[i]) {
        SwapCols(x.Col(i), x.Col(Pvt[i]));
      }
    }
  }
  /// @brief Get the number of permutation (unsigned for bit-wise operations)
  [[nodiscard]] uint32_t PermutationCount() const {
    return permCount;
  }

  /// @brief Permutations of P. \f$ P = P_n * P_{n-1} * ... * P_1 \f$
  DynamicArray<int> Pvt;
  /// @brief Permutations of Q. \f$ Q = Q_1 * Q_2 * ... * Q_{n} \f$
  DynamicArray<int> Qvt;

  uint32_t permCount = 0;
};

namespace details {

/// @brief Compute the inverse of a matrix via LU factorization, gating between @ref SmallLU and
/// @ref LU based on matrix size.
///
/// @param[in] A The matrix to invert.
/// @param[out] invA The inverse of @p A.
template <PermuteAlg kPermAlg, typename MatrixA, typename MatrixInv>
void LuInverse(MatrixA const& A, MatrixInv& invA) {
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix is not square.");
  using Traits = krylov::details::MatTraits<MatrixA>;
  using Scalar = std::remove_const_t<typename Traits::Scalar>;
  constexpr int kRows = Traits::kNumRows;
  constexpr int kCols = Traits::kNumCols;
  if (A.Rows() < kSmallLUInverseSizeThreshold) {
    SmallLU<Scalar, kRows, kCols, kPermAlg> lu(A);
    lu.Inverse(invA);
  } else {
    LU<Scalar, kRows, kCols, kPermAlg> lu(A);
    lu.Inverse(invA);
  }
}

} // namespace details

} // namespace mochi
