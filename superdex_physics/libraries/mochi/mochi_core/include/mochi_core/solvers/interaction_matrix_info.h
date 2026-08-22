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

#include <mochi_core/linear_algebra/any_matrix.h>

#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mochi {

/// WARNING: If an interaction matrix is not square or has different row and col offsets, then it
/// must NOT overlap with the block diagonal corresponding to an actor, i.e. it must be an
/// off-diagonal submatrix for the interaction between two actors.
template <typename MatType, typename Scalar>
struct InteractionMatrixInfoImpl {
  int rowOffset = 0;
  int colOffset = 0;
  MatType matrix = {};

  /// @brief Optional view to the so-called "symmetric pair" of this interaction matrix. If this
  /// interaction matrix is symmetric, the symmetric pair is itself and 'symmetricPair' may be
  /// nullopt or a view of itself, i.e. a view of 'matrix'. If this interaction matrix is an
  /// off-diagonal block representing the interaction from actor A to actor B (rows=A, cols=B), then
  /// the symmetric pair is the matrix representing the interaction from B to A (rows=B, cols=A).
  ///
  /// @note If non-nullopt, the owning matrix must outlive this struct and remain at the same memory
  /// address.
  std::optional<AnyMatrixView<Scalar const>> symmetricPair = std::nullopt;

  InteractionMatrixInfoImpl() = default;
  InteractionMatrixInfoImpl(
      int rowOffset,
      int colOffset,
      MatType&& matrix,
      std::optional<AnyMatrixView<Scalar const>> const& symmetricPair)
      : rowOffset(rowOffset),
        colOffset(colOffset),
        matrix(std::move(matrix)),
        symmetricPair(symmetricPair) {}
};

template <typename T>
using AnyInteractionMatrixInfo = InteractionMatrixInfoImpl<AnyMatrix<T>, T>;

template <typename T>
using AnyInteractionMatrixViewInfo = InteractionMatrixInfoImpl<AnyMatrixView<T>, T>;

template <typename T>
using AnyInteractionMatrixPtrInfo = InteractionMatrixInfoImpl<AnyMatrix<T>*, T>;

// Overload get.
template <std::size_t N, typename MatType, typename Scalar>
auto& get(InteractionMatrixInfoImpl<MatType, Scalar>& info) {
  if constexpr (N == 0) {
    return info.rowOffset;
  } else if constexpr (N == 1) {
    return info.colOffset;
  } else if constexpr (N == 2) {
    return info.matrix;
  } else if constexpr (N == 3) {
    return info.symmetricPair;
  } else {
    static_assert(N < 4, "Index out of bounds");
  }
}
template <std::size_t N, typename MatType, typename Scalar>
auto const& get(InteractionMatrixInfoImpl<MatType, Scalar> const& info) {
  if constexpr (N == 0) {
    return info.rowOffset;
  } else if constexpr (N == 1) {
    return info.colOffset;
  } else if constexpr (N == 2) {
    return info.matrix;
  } else if constexpr (N == 3) {
    return info.symmetricPair;
  } else {
    static_assert(N < 4, "Index out of bounds");
  }
}

} // namespace mochi

namespace std {
// Specialize std::tuple_size.
template <typename MatType, typename Scalar>
struct tuple_size<mochi::InteractionMatrixInfoImpl<MatType, Scalar>>
    : std::integral_constant<std::size_t, 4> {};

// Specialize std::tuple_element.
template <std::size_t N, typename MatType, typename Scalar>
struct tuple_element<N, mochi::InteractionMatrixInfoImpl<MatType, Scalar>> {
  using type = std::decay_t<decltype(mochi::get<N>(
      declval<mochi::InteractionMatrixInfoImpl<MatType, Scalar>>()))>;
};
} // namespace std
