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
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/simd.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace mochi {

namespace details {

/// @brief Round a positive value to the closest power of two (in geometric/log sense).
template <typename T>
MOCHI_FORCE_INLINE T RoundToPowerOfTwo(T s) {
  static_assert(std::numeric_limits<T>::is_iec559, "Requires IEEE 754 floating-point");
  MOCHI_ASSERT_VERBOSE(s > T{0} && std::isnormal(s), "Invalid input.");
  if constexpr (std::is_same_v<T, float>) {
    uint32_t i = std::bit_cast<uint32_t>(s);
    i += 0x004AFB0D; // Magic offset for 23-bit mantissa
    i &= 0xFF800000; // Mask out the mantissa to exactly a power of 2
    return std::bit_cast<float>(i);

  } else {
    static_assert(std::is_same_v<T, double>, "Unsupported scalar type");
    uint64_t i = std::bit_cast<uint64_t>(s);
    i += 0x00095F619980C433ULL; // Magic offset for 52-bit mantissa
    i &= 0xFFF0000000000000ULL; // Mask out the mantissa to exactly a power of 2
    return std::bit_cast<double>(i);
  }
}

} // namespace details

using krylov::Direction;
using krylov::kDynamic;
using krylov::Ownership;

/// @brief Equilibration strategy for the LDLt factorization.
enum class LDLtEquilibration {
  None, ///< No equilibration.
  Diagonal, ///< Diagonal equilibration: 1/sqrt(|A_ii| + eps). Recommended for SPD matrices.
  Count ///< Number of LDLt equilibration enum values.
  // TODO: Implement max-row equilibration for indefinite matrices: 1/sqrt(max_j |A_ij| + eps).
};

constexpr std::integral_constant<LDLtEquilibration, LDLtEquilibration::None> NoEquilibration;
constexpr std::integral_constant<LDLtEquilibration, LDLtEquilibration::Diagonal>
    DiagonalEquilibration;

/// @brief Default block size for the LDLt factorization.
/// @note The current default of 24 is empirically optimal for matrix sizes over ~100, both on x64
/// and ARM.
/// @note For matrix sizes under ~100, smaller block sizes are faster and BasicLDLt should be used.
/// @note For matrix sizes over ~1000, larger block sizes may be faster in some architectures.
/// @note The notes above refer to single-precision arithmetic. Performance has not been tuned in
/// double precision yet.
constexpr int kLDLtDefaultBlockSize = 24;

/// @brief Batch of work size for some operations of the LDLt factorization. The block size must be
/// a multiple of this value.
/// @note For block size of 24 and single-precision arithmetic, the current default of 6 is
/// empirically optimal (or near optimal), both on x64 and ARM. In
/// some architectures, 4 or 8 may be sligthly faster.
/// @note Performance has not been tuned in double precision yet.
constexpr int kLDLtDefaultTrSize = 6;
static_assert(kLDLtDefaultBlockSize % kLDLtDefaultTrSize == 0);

/// @brief Recommended matrix size threshold to prefer @ref SmallLDLt over the default @ref LDLt
/// for the inverse operation.
///
/// @details Benchmarked on x64 and ARM in single precision. SmallLDLt is faster for inverse at
/// sizes ≤100 on all platforms. LDLt is faster at sizes ≥150. At size 100, the two variants are
/// within 3–6% across platforms and storage directions. Performance has not been tuned in double
/// precision yet.
///
/// @note For the factorization alone, the crossover is ~150 on all platforms. For the solve with
/// a single vector, SmallLDLt is faster up to ~500–1000.
constexpr int kSmallLDLtInverseSizeThreshold = 100;

/// @brief Class to compute a (block) LDL^T factorization of a matrix A.
///
/// @tparam InputScalar Scalar type.
/// @tparam kRowsAtCT Compile-time row size, or @ref kDynamic for runtime-sized.
/// @tparam kColsAtCT Compile-time column size, or @ref kDynamic for runtime-sized.
/// @tparam kEquilibration Equilibration strategy.
/// @tparam kBlockSize Block size for the blocked factorization algorithm.
/// @tparam kTrSize Batch of work size for triangular operations.
///
/// @note It does not use any pivoting.
/// @note It uses only the lower triangular part of A (if col-major) or the upper triangular part of
/// A (if row-major).
/// @note For matrix sizes under 50-100, prefer the alias @ref SmallLDLt that uses a smaller block
/// size.
template <
    typename InputScalar,
    int kRowsAtCT = kDynamic,
    int kColsAtCT = kDynamic,
    LDLtEquilibration kEquilibration = LDLtEquilibration::None,
    int kBlockSize = kLDLtDefaultBlockSize,
    int kTrSize = kLDLtDefaultTrSize>
class LDLt {
 public:
  using Scalar = std::remove_const_t<InputScalar>;
  static constexpr int kBlockSz = kBlockSize;
  static_assert(kBlockSize % kTrSize == 0, "Block size must be multiple of kTrSize");
  static_assert(kEquilibration < LDLtEquilibration::Count, "Unsupported equilibration strategy");
  static_assert(
      kRowsAtCT == kDynamic || kColsAtCT == kDynamic || kRowsAtCT == kColsAtCT,
      "LDLt requires a square matrix");
  static constexpr int kSizeAtCT = Max(kRowsAtCT, kColsAtCT);
  /// @brief Mantissa-exponent representation for determinant computation.
  struct MantissaExponent {
    Scalar mantissa;
    int exponent;
  };

  /// @brief Constructor for the factorization.
  ///
  /// @param[in] A     Symmetric matrix to factorize.
  /// @param[out] info Information status flag:
  ///                  - 0: Success.
  ///                  - i (> 0): Failure. The matrix has (at least) i singularities or needs
  ///                    pivoting.
  /// @details The singularity detection is based on the magnitude of the original entries in the
  /// diagonal vs. after partial elimination. Other methods can be used by switching to a different
  /// singularity detection functor, e.g. P897440614.
  template <typename ScalarA, Direction kMajorDirection, Ownership kOwnership, int kLeadDim>
  LDLt(
      Matrix<ScalarA, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadDim> const& A,
      int& info);

  /// @brief Constructor for use in CTAD expressions.
  ///
  /// @param[in] A     Symmetric matrix to factorize.
  /// @param[out] info Information status flag (see other constructor).
  template <
      typename ScalarA,
      int kR,
      int kC,
      Direction kMajorDirection,
      Ownership kOwnership,
      int kLeadDim>
  LDLt(
      std::integral_constant<LDLtEquilibration, kEquilibration>,
      Matrix<ScalarA, kR, kC, kMajorDirection, kOwnership, kLeadDim> const& A,
      int& info)
      : LDLt(A, info) {}

  /// @brief Solve in place linear systems X <- X (LDL^T)^{-1}
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

  /// @brief Solve in place linear systems X <- X (LDL^T)^{-1}
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
  /// @details The computed inverse is symmetric to machine precision.
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
    if (_singularities > 0) {
      return {Scalar{0}, 0};
    }

    Scalar mantissa{1};
    int exponent{0};
    for (int i = 0; i < _size; ++i) {
      int expn MOCHI_NO_INIT;
      if constexpr (kEquilibration == LDLtEquilibration::None) {
        mantissa = std::frexp(mantissa * _L(i, i), &expn);
      } else {
        // det(A) = det(S)^{-2} * det(A'), where A' is the equilibrated matrix. Correct the
        // determinant by det(S)^{-2} = 1 / (prod_i s_i)^2.
        mantissa = std::frexp(mantissa * _L(i, i) * Sqr(_scale(i)), &expn);
      }
      exponent += expn;
    }

    int expn MOCHI_NO_INIT;
    mantissa = std::frexp(Scalar{1} / mantissa, &expn);
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
          "Determinant overflow. Please use LDLt::Determinant instead of LDLt::ScalarDeterminant "
          "to get a mantissa-exponent representation without overflow.");
    }
    return det;
  }

  /// @brief Access for underlying matrix storage for testing only.
  auto const& GetStorage() const {
    return _L;
  }

 protected:
  /// @brief Return the symmetric input matrix as col-major for efficient copy into _L.
  template <typename MatA>
  MOCHI_FORCE_INLINE static decltype(auto) AsColMajor(MatA const& A) {
    if constexpr (details::MatTraits<MatA>::kMajorDir == Direction::ColMajor) {
      return A;
    } else {
      return A.Transpose();
    }
  }

  void Factorize(bool failOnSingularity);

  template <typename ScalarX, int kRowsX, int kColsX, Ownership kOwnershipX, int kLeadDimX>
  void InversePostTranspose(
      Matrix<ScalarX, kRowsX, kColsX, Direction::RowMajor, kOwnershipX, kLeadDimX>& invA) const;

  /// Matrix size.
  int _size = 0;
  /// LDLt factors.
  Matrix<Scalar, kSizeAtCT, kSizeAtCT, Direction::ColMajor> _L;
  /// Number of singularities detected.
  int _singularities = 0;
  /// Equilibration scale factors. The equilibrated matrix is A' = S A S where S = diag(_scale).
  /// Not populated for @ref LDLtEquilibration::None.
  ColumnVector<Scalar, kSizeAtCT> _scale;
};

template <
    typename InputScalar,
    int kRowsAtCT,
    int kColsAtCT,
    LDLtEquilibration kEquilibration,
    int kBlockSize,
    int kTrSize>
template <typename ScalarA, Direction kMajorDirection, Ownership kOwnership, int kLeadDim>
LDLt<InputScalar, kRowsAtCT, kColsAtCT, kEquilibration, kBlockSize, kTrSize>::LDLt(
    Matrix<ScalarA, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadDim> const& A,
    int& info)
    : _size(A.Rows()), _L(AsColMajor(A)) {
  static_assert(std::is_same_v<InputScalar const, ScalarA const>, "Incompatible scalar types");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  Factorize(/*failOnSingularity*/ true);
  info = _singularities;
}

template <
    typename ScalarA,
    int kRowsAtCT,
    int kColsAtCT,
    LDLtEquilibration kEquilibration,
    Direction kMajorDirection,
    Ownership kOwnership,
    int kLeadDim>
LDLt(
    std::integral_constant<LDLtEquilibration, kEquilibration>,
    Matrix<ScalarA, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadDim> const& A,
    int& info) -> LDLt<ScalarA, kRowsAtCT, kColsAtCT, kEquilibration>;

/// @brief Free-standing function to perform LDL^T factorization.
///
/// @param[in,out] L Matrix containing the input matrix on entry and the LDL^T factors on exit.
/// @param[in] failOnSingularity Whether to fail on singularity detection.
/// @param[in] singularityCheck Function to check for singularities and update determinant.
/// @return Number of singularities detected. It is a lower bound to the number of singularities if
/// failOnSingularity is true.
///
/// @note This is a low-level factorization kernel. It does not perform equilibration. For
/// equilibrated factorization, use the @ref LDLt class with equilibration.
template <
    int kBlockSize = kLDLtDefaultBlockSize,
    int kTrSize = kLDLtDefaultTrSize,
    IsHostMatrix MatrixType,
    typename SingularityCheckFn>
int LDLtFactorize(MatrixType& L, bool failOnSingularity, SingularityCheckFn&& singularityCheck) {
  using Scalar = typename krylov::details::MatTraits<MatrixType>::Scalar;
  int const size = L.Cols();
  int iCol = 0;
  int singularities = 0;
  auto check = [&singularityCheck, &iCol](int r, auto d, auto&& S) {
    return singularityCheck(iCol + r, d, S);
  };
  // TODO: Use matrix loops to simplify the implementation.
  for (; iCol + kBlockSize <= size; iCol += kBlockSize) {
    // Left looking update.
    auto panel = L.template Block<kDynamic, kBlockSize>(iCol, iCol, size - iCol, kBlockSize);
    panel -= L.Block(iCol, 0, size - iCol, iCol) *
        L.template Block<kDynamic, kBlockSize>(0, iCol, iCol, kBlockSize);
    auto D = L.template Block<kBlockSize, kBlockSize>(iCol, iCol, kBlockSize, kBlockSize);
    // Factor diagonal block into D = L D L^t.
    int blockSingularities = 0;
    kernel::FactorBlock(D, blockSingularities, check);
    if (blockSingularities > 0)
      MOCHI_UNLIKELY {
        singularities += blockSingularities;
        if (failOnSingularity) {
          MOCHI_LOG_ERROR("Matrix is singular or needs pivoting.");
          return singularities;
        }
      }
    auto firstBottomRow = iCol + kBlockSize;
    auto subPanel = panel.BottomRows(size - firstBottomRow);
    // Apply L^-T to lower part.
    kernel::ApplyLmtOnRight<Scalar, kTrSize, kBlockSize>(D, subPanel);
    // Copy panel to upper part and multiply panel by D^-1 on the right.
    for (int r = 0; r < subPanel.Rows(); ++r) {
      for (int c = 0; c < kBlockSize; ++c) {
        L(iCol + c, firstBottomRow + r) = subPanel(r, c);
        subPanel(r, c) *= D(c, c);
      }
    }
  }
  // Work on overflow block.
  // Left looking update.
  if (iCol < size) {
    auto D = L.Block(iCol, iCol, size - iCol, size - iCol);
    if (iCol > 0) {
      D -= L.Block(iCol, 0, size - iCol, iCol) * L.Block(0, iCol, iCol, size - iCol);
    }
    int const blockSingularities = kernel::Factor<Scalar>(D, check);

    if (blockSingularities > 0)
      MOCHI_UNLIKELY {
        singularities += blockSingularities;
        if (failOnSingularity) {
          MOCHI_LOG_ERROR("Matrix is singular or needs pivoting.");
          return singularities;
        }
      }
  }
  return singularities;
}

template <
    typename InputScalar,
    int kRowsAtCT,
    int kColsAtCT,
    LDLtEquilibration kEquilibration,
    int kBlockSize,
    int kTrSize>
void LDLt<InputScalar, kRowsAtCT, kColsAtCT, kEquilibration, kBlockSize, kTrSize>::Factorize(
    bool failOnSingularity) {
  Scalar constexpr kEpsilon = std::numeric_limits<Scalar>::epsilon();
  ColumnVector<Scalar, kSizeAtCT> absDiag(_size);
  for (int i = 0; i < _size; ++i) {
    absDiag(i) = Abs(_L(i, i));
  }

  if constexpr (kEquilibration == LDLtEquilibration::Diagonal) {
    // Apply diagonal equilibration: A' = S A S where S = diag(1/sqrt(|A_ii| + eps)). Scale factors
    // are rounded to the closest power of two so that the scaling operation only modifies the
    // floating-point exponents, leaving the mantissas (and thus the precision) unchanged.
    if constexpr (kSizeAtCT == kDynamic) {
      _scale.Resize(_size);
    }

    // Regularization threshold relative to the maximum diagonal entry.
    Scalar const diagEpsilon = kEpsilon * Max(MakeConstSpan(absDiag));

    // Compute scale factors: s_i = 1 / sqrt(|A_ii| + eps), rounded to the closest power of two.
    for (int i = 0; i < _size; ++i) {
      // TODO: Explicit vectorization may improve factorization performance for small matrices.
      _scale(i) = details::RoundToPowerOfTwo(Scalar{1} / Sqrt(absDiag(i) + diagEpsilon));
    }

    // Apply symmetric scaling: A'_ij = s_i * A_ij * s_j (lower triangle only).
    for (int j = 0; j < _size; ++j) {
      for (int i = j; i < _size; ++i) {
        _L(i, j) *= _scale(i) * _scale(j);
      }
    }
  }

  // Singularity check and determinant computation.
  auto singularityCheck = [&](int eq, auto d, auto const& /*S*/) {
    if constexpr (kEquilibration == LDLtEquilibration::None) {
      return (Abs(d) <= kEpsilon * absDiag(eq));
    } else {
      static_assert(kEquilibration == LDLtEquilibration::Diagonal);
      return (Abs(d) <= kEpsilon);
    }
  };

  _singularities =
      LDLtFactorize<kBlockSize, kTrSize, decltype(_L)>(_L, failOnSingularity, singularityCheck);
}

template <
    typename InputScalar,
    int kRowsAtCT,
    int kColsAtCT,
    LDLtEquilibration kEquilibration,
    int kBlockSize,
    int kTrSize>
template <
    typename ScalarX,
    int kRowsX,
    int kColsX,
    Direction kDirectionX,
    Ownership kOwnershipX,
    int kLeadDimX>
void LDLt<InputScalar, kRowsAtCT, kColsAtCT, kEquilibration, kBlockSize, kTrSize>::LeftSolveInPlace(
    Matrix<ScalarX, kRowsX, kColsX, kDirectionX, kOwnershipX, kLeadDimX>& Xin) const {
  // For equilibrated systems: A = S^{-1} A' S^{-1}, where A' = L D L^T.
  // Solving A X = B:  X = S (A')^{-1} S B = S L^{-T} D^{-1} L^{-1} S B.
  // Steps: (1) B <- S B, (2) solve A' Y = B, (3) X <- S Y.
  if (_singularities != 0) {
    MOCHI_LOG_ERROR("Factorization error prevents to compute the solution.");
    return;
  }
  MOCHI_ASSERT_VERBOSE(Xin.Rows() == _size, "Dimensions do not match.");

  // Pre-scale: B <- S B.
  if constexpr (kEquilibration != LDLtEquilibration::None) {
    for (int i = 0; i < _size; ++i) {
      Xin.Row(i) *= _scale(i);
    }
  }

  // Apply L^-1 in blocks.
  using namespace blocking;
  PartDown<kBlockSize, kSizeAtCT>(
      _size,
      [](auto&& X, auto&& L) {
        auto currentX = X(DiagRows);
        auto prevX = X(Above);
        if (prevX.Rows() > 0) {
          currentX -= L(DiagRows, Left) * prevX;
        }
        kernel::ApplyLm1OnLeft(L(DiagBlock), currentX);
      },
      Xin,
      _L);

  // Apply D^-1.
  for (int iCol = 0; iCol < _size; ++iCol) {
    Xin.Row(iCol) *= _L(iCol, iCol);
  }

  // Apply L^-T in blocks.
  PartUp<kBlockSize, kSizeAtCT>(
      _size,
      [](auto&& X, auto&& L) {
        auto currentX = X(DiagRows);
        auto prevX = X(Below);
        if (prevX.Rows() > 0) {
          currentX -= L(Below, DiagCols).Transpose() * prevX;
        }
        kernel::ApplyLmtOnLeft(L(DiagBlock), currentX);
      },
      Xin,
      _L);

  // Post-scale: X <- S Y.
  if constexpr (kEquilibration != LDLtEquilibration::None) {
    for (int i = 0; i < _size; ++i) {
      Xin.Row(i) *= _scale(i);
    }
  }
}

template <
    typename InputScalar,
    int kRowsAtCT,
    int kColsAtCT,
    LDLtEquilibration kEquilibration,
    int kBlockSize,
    int kTrSize>
template <
    typename ScalarX,
    int kRowsX,
    int kColsX,
    Direction kDirectionX,
    Ownership kOwnershipX,
    int kLeadDimX>
void LDLt<InputScalar, kRowsAtCT, kColsAtCT, kEquilibration, kBlockSize, kTrSize>::
    RightSolveInPlace(
        Matrix<ScalarX, kRowsX, kColsX, kDirectionX, kOwnershipX, kLeadDimX>& Xin) const {
  // If B is the left-hand side, this routine computes X = B * A^{-1} = B * L^{-T} D^{-1} L^{-1}.
  // Note that X = ( A^{-1} B^T )^T.
  // TODO: Introduce dedicated implementation for 'RightSolveInPlace'. Once it's available, update
  // 'Inverse' if using a right solve is faster than a left solve for the inverse.
  auto Xt = Xin.Transpose(); // View
  LeftSolveInPlace(Xt);
}

template <
    typename InputScalar,
    int kRowsAtCT,
    int kColsAtCT,
    LDLtEquilibration kEquilibration,
    int kBlockSize,
    int kTrSize>
template <typename ScalarX, int kRowsX, int kColsX, Ownership kOwnershipX, int kLeadDimX>
void LDLt<InputScalar, kRowsAtCT, kColsAtCT, kEquilibration, kBlockSize, kTrSize>::
    InversePostTranspose(
        Matrix<ScalarX, kRowsX, kColsX, Direction::RowMajor, kOwnershipX, kLeadDimX>& invA) const {
  // For equilibrated systems: A = S^{-1} A' S^{-1}, where A' = L D L^T.
  // Therefore: A^{-1} = S (A')^{-1} S, and A^{-1}[i,j] = s_i * (A')^{-1}[i,j] * s_j.
  // Set input matrix to the identity.
  invA.SetIdentity();

  // Check inputs and factorization flag.
  MOCHI_ASSERT_VERBOSE(
      (invA.Rows() == _size) && (invA.Cols() == _size), "Dimensions do not match.");
  if (_singularities != 0) {
    MOCHI_LOG_ERROR("Factorization error prevents to compute the inverse.");
    return;
  }

  // Apply L^-1 in blocks (mostly) to the lower part.
  using namespace blocking;
  Matrix<Scalar, kBlockSize, kSizeAtCT, Direction::ColMajor> workSpace(kBlockSize, _size);
  PartDown<kBlockSize, kSizeAtCT>(
      _size,
      [](auto&& X, auto&& L, auto&& workspace) {
        auto currentX = X(DiagRows, LeftWithDiag); // Columns on the right are zero.
        kernel::ApplyLm1OnLeft(L(DiagBlock), currentX);
        // Propagate the values below.
        auto nextX = X(Below, LeftWithDiag);
        if (nextX.Rows() > 0) {
          workspace(Right) = L(Below, DiagCols).Transpose();
          nextX -= workspace(Right).Transpose() * currentX;
        }
        /*
        Alternative implementation. Empirically, it has similar performance up to sizes ~100 and
        it's slower for large matrices.

        auto prevX = X(DiagRows, Left);
        if (prevX.Rows() > 0) {
          workspace(Above) = L(DiagRows, Left).Transpose();
          prevX -= workspace(Above).Transpose() * X(Above, Left);
        }
        kernel::ApplyLm1OnLeft(L(DiagBlock), X(DiagRows, LeftWithDiag));
        */
      },
      invA,
      _L,
      workSpace);

  // Apply D^-1.
  for (int iCol = 0; iCol < _size; ++iCol) {
    invA.Row(iCol) *= _L(iCol, iCol);
  }

  // Apply L^-T in blocks. Compute only the diagonal and lower triangular part. The upper triangular
  // part is symmetric.
  PartUp<kBlockSize, kSizeAtCT>(
      _size,
      [](auto&& X, auto&& L) {
        auto currentX = X(DiagRows, LeftWithDiag); // Columns on the right are symmetrized later.
        auto prevX = X(Below, LeftWithDiag);
        if (prevX.Rows() > 0) {
          currentX -= L(Below, DiagCols).Transpose() * prevX;
        }
        kernel::ApplyLmtOnLeft(L(DiagBlock), currentX);
      },
      invA,
      _L);

  // Copy the lower triangular part to the upper triangular part, applying equilibration scaling.
  // For equilibrated systems: A^{-1}[i,j] = s_i * (A')^{-1}[i,j] * s_j.
  if constexpr (kEquilibration != LDLtEquilibration::None) {
    for (int ii = 0; ii < _size; ++ii) {
      invA(ii, ii) *= Sqr(_scale(ii));
      for (int jj = 0; jj < ii; ++jj) {
        Scalar const scaled = invA(ii, jj) * _scale(ii) * _scale(jj);
        invA(ii, jj) = scaled;
        invA(jj, ii) = scaled;
      }
    }
  } else {
    // TODO: Explicit vectorization may improve LDLt::Inverse performance for small matrices.
    for (int ii = 0; ii < _size; ++ii) {
      for (int jj = 0; jj < ii; ++jj) {
        invA(jj, ii) = invA(ii, jj);
      }
    }
  }
}

template <
    typename InputScalar,
    int kRowsAtCT,
    int kColsAtCT,
    LDLtEquilibration kEquilibration,
    int kBlockSize,
    int kTrSize>
template <
    typename ScalarX,
    int kRowsX,
    int kColsX,
    Direction kDirectionX,
    Ownership kOwnershipX,
    int kLeadDimX>
void LDLt<InputScalar, kRowsAtCT, kColsAtCT, kEquilibration, kBlockSize, kTrSize>::Inverse(
    Matrix<ScalarX, kRowsX, kColsX, kDirectionX, kOwnershipX, kLeadDimX>& invA) const {
  // Inverse operates on block rows. It's most efficient if the input matrix is stored as row-major.
  if constexpr (kDirectionX == Direction::RowMajor) {
    InversePostTranspose(invA);
  } else {
    auto invAT = invA.Transpose(); // View
    InversePostTranspose(invAT);
  }
}

/// @brief LDLt specialization with smaller block sizes. Prefer it over the default LDLt for matrix
/// sizes up to 50-200.
/// @note The block size and batch of work size of 4 is optimal (or near optimal) for matrix sizes
/// up to ~100, both on x64 and ARM.
/// @note The optimal matrix size threshold to favor @ref SmallLDLt over the default LDLt depends on
/// the scalar type, operation, storage direction, and architecture. For the factorization, the
/// optimal threshold is often smaller than ~100. For the solve with a single vector, the optimal
/// threshold is often substantially larger than 100.
/// @note The notes above refer to single-precision arithmetic. Performance has not been tuned in
/// double precision yet.
template <
    typename InputScalar,
    int kRowsAtCT = kDynamic,
    int kColsAtCT = kDynamic,
    LDLtEquilibration kEquilibration = LDLtEquilibration::None>
using SmallLDLt = LDLt<InputScalar, kRowsAtCT, kColsAtCT, kEquilibration, 4, 4>;

namespace details {

/// @brief Compute the inverse of a symmetric matrix, gating between @ref SmallLDLt and @ref LDLt
/// based on matrix size.
///
/// @param[in] A The symmetric matrix to invert.
/// @param[out] invA The inverse of @p A.
///
/// @return The info flag from the LDLt factorization (0 = success).
template <LDLtEquilibration kEq = LDLtEquilibration::None, typename MatrixA, typename MatrixInv>
int LDLtInverse(MatrixA const& A, MatrixInv& invA) {
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix is not square.");
  using Traits = krylov::details::MatTraits<MatrixA>;
  using Scalar = std::remove_const_t<typename Traits::Scalar>;
  constexpr int kRows = Traits::kNumRows;
  constexpr int kCols = Traits::kNumCols;
  int info{};
  if (A.Rows() < kSmallLDLtInverseSizeThreshold) {
    SmallLDLt<Scalar, kRows, kCols, kEq> ldlt(A, info);
    ldlt.Inverse(invA);
  } else {
    LDLt<Scalar, kRows, kCols, kEq> ldlt(A, info);
    ldlt.Inverse(invA);
  }
  return info;
}

} // namespace details

} // namespace mochi
