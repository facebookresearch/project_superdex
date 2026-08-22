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

#include <mochi_core/mochi_platform.h>

#include <type_traits>

namespace mochi {
/// @brief Parent class of matrices handled with full expression templates (tag).
struct NewMatrix {};
/// @brief Parent class of matrix expressions handled with full expression templates (tag).
struct NewMatrixExpr {};

namespace details {
/** @brief Classification of matrices and expressions. */
enum class ExprDomain {
  Host = 1, /// @brief Data is on host.
  Device = 2, /// @brief Data is on device, with pointer on host.
  Strided = 4, /// @brief For use with StridedMatrix, whether on host or device.
  Unknown = 7 // TODO Remove. Only until all cases covered.
};

MOCHI_ANY constexpr ExprDomain operator&(ExprDomain a, ExprDomain b) {
  if (a == b) {
    return a;
  } else {
    return ExprDomain::Unknown;
  }
}

template <ExprDomain kDomain>
struct DomainType {
  constexpr static ExprDomain domain = kDomain;
};

template <ExprDomain kDomainLeft, ExprDomain kDomainRight>
MOCHI_ANY auto operator&(
    DomainType<kDomainLeft> const& /*l*/,
    DomainType<kDomainRight> const& /*r*/) {
  return DomainType<kDomainLeft & kDomainRight>{};
}

template <typename T>
MOCHI_ANY constexpr auto GetDomainFor(T const&);

} // namespace details

} // namespace mochi

namespace mochi::krylov::details {

template <typename T>
struct MatTraitsDef;

template <typename T>
using MatTraits = MatTraitsDef<std::decay_t<T>>;
} // namespace mochi::krylov::details

namespace mochi::details {
template <typename T>
constexpr bool IsAnyMatrixVariantDef = false;
template <typename T>
constexpr bool IsSparseMatrixDef = false;
template <typename T>
constexpr bool IsBlockSparseMatrixDef = false;
template <typename T>
constexpr bool IsIslandOperatorsDef = false;
template <typename T>
constexpr bool IsIslandOperatorsOwningLiteDef = false;
template <typename T>
constexpr bool IsLowRankAugmentedMatrixDef = false;
template <typename T>
constexpr bool IsStridedMatrixDef = false;
template <typename T, typename = void>
constexpr bool IsCudaDef = false;
template <typename T>
constexpr bool IsCudaDef<T, std::void_t<decltype(krylov::details::MatTraits<T>::kIsCuda)>> =
    krylov::details::MatTraits<T>::kIsCuda;
} // namespace mochi::details

namespace mochi {

/// @brief Concept of a variant-based AnyMatrix object.
///
/// @note It does not include AnyMatrixView objects.
template <typename T>
concept IsAnyMatrixVariant = details::IsAnyMatrixVariantDef<std::decay_t<T>>;

/// @brief Concept of an object whose memory is on the GPU.
template <typename T>
concept IsCuda = details::IsCudaDef<std::decay_t<T>>;

/// @brief Concept of a dense-matrix-like object adapted to the full expression templates.
/// @note It can be a dense matrix, a strided matrix or a matrix expression.
template <typename T>
concept IsMatrixLike = std::is_base_of_v<NewMatrix, std::decay_t<T>>;

/// @brief Concept of a matrix expression object adapted to the full expression templates.
template <typename T>
concept IsMatrixExpr = std::is_base_of_v<NewMatrixExpr, std::decay_t<T>>;

/// @brief Concept of strided matrix.
template <typename T>
concept IsStridedMatrix = details::IsStridedMatrixDef<std::decay_t<T>>;

/// @brief Concept of dense matrix.
/// @note False for strided matrices.
template <typename T>
concept IsMatrix = IsMatrixLike<T> && !IsMatrixExpr<T> && !IsStridedMatrix<T>;

/// @brief Concept of a dense matrix whose memory is on the GPU.
template <typename T>
concept IsCudaMatrix = IsMatrix<T> && IsCuda<T>;

/// @brief Concept of a dense matrix whose memory is on the host device.
template <typename T>
concept IsHostMatrix = IsMatrix<T> && !IsCuda<T>;

/// @brief Concept of a dense matrix that has the same Scalar const.
template <typename T, typename Mat>
concept IsCompatibleMatrixClass = IsMatrix<T> &&
    (std::is_same_v<
        typename krylov::details::MatTraits<T>::Scalar const,
        typename krylov::details::MatTraits<Mat>::Scalar const>);

/// @brief Concept of a sparse matrix.
template <typename T>
concept IsSparseMatrix = details::IsSparseMatrixDef<std::decay_t<T>>;

/// @brief Concept of a block sparse matrix.
template <typename T>
concept IsBlockSparseMatrix = details::IsBlockSparseMatrixDef<std::decay_t<T>>;

/// @brief Concept of a supported matrix type.
template <typename T>
concept IsAnyMatrix = IsMatrixLike<T> || IsSparseMatrix<T> || IsBlockSparseMatrix<T>;

/// @brief Concept of an island operators.
template <typename T>
concept IsIslandOperators = details::IsIslandOperatorsDef<std::decay_t<T>>;

/// @brief Concept of an island operators owning the operators.
template <typename T>
concept IsIslandOperatorsOwningLite = details::IsIslandOperatorsOwningLiteDef<std::decay_t<T>>;

/// @brief Concept of a low-rank augmented matrix.
template <typename T>
concept IsLowRankAugmentedMatrix = details::IsLowRankAugmentedMatrixDef<std::decay_t<T>>;

/// @brief Concept of a supported linear operator.
/// @note Supported linear operators must implement:
/// - T::Scalar
/// - T::NonConstScalar
/// - T::Rows() -> Idx
/// - T::Cols() -> Idx
/// - ToMatrix(T const& A) -> Matrix<T::NonConstScalar>
/// - FlopsPerApply(T const& A) -> Idx
/// - GetRowRangesPerWorker(T const& A, int numWorkers) -> std::vector<Idx>
/// - Apply(T const& A, Input const& in, Output&& out) -> void
/// - ApplyToRange(T const& A, Input const& in, Output&& out, Idx rowBegin, Idx rowEnd) -> void
/// where Idx is an integer type that may be different in each of the methods and functions above.
template <typename T>
concept IsLinearOperator = IsAnyMatrix<T> || IsIslandOperators<T> || IsLowRankAugmentedMatrix<T>;

} // namespace mochi
