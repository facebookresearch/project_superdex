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

#include <type_traits>

namespace mochi::details {

enum class DestOp : unsigned int {
  Set = 1u, /// @brief A first update should set the values from uninitialized
  Add = 0u, /// @brief The destination contains partial result and should be added to
  Sub = 2u, /// @brief The destination contains partial result and should be update by subtraction
  NegSet =
      3u, /// @brief A first update should set the values from uninitialized with a negated value
};

/// @brief Negate the destination operation
constexpr DestOp operator-(DestOp op) {
  return static_cast<DestOp>(static_cast<unsigned int>(op) ^ 2u);
}

constexpr DestOp AfterSet(DestOp op, bool negate) {
  if (negate) {
    op = -op;
  }
  return static_cast<DestOp>(static_cast<unsigned int>(op) & ~1u);
}

static_assert(AfterSet(DestOp::Set, false) == DestOp::Add);
static_assert(AfterSet(DestOp::NegSet, false) == DestOp::Sub);
static_assert(AfterSet(DestOp::Add, false) == DestOp::Add);
static_assert(AfterSet(DestOp::Sub, false) == DestOp::Sub);
static_assert(AfterSet(DestOp::Set, true) == DestOp::Sub);
static_assert(AfterSet(DestOp::NegSet, true) == DestOp::Add);
static_assert(AfterSet(DestOp::Add, true) == DestOp::Sub);
static_assert(AfterSet(DestOp::Sub, true) == DestOp::Add);

/// @brief Data about matrices, including a scaling factor for linear combinations.
/// @tparam Scalar
///
/// @note Usage of this class assumes that either rowStride or colStride is equal to 1.
/// This assumption has to be maintained by the developer.
template <typename Scalar>
struct ScaledMatData {
  using NonConstScalar = std::remove_const_t<Scalar>;
  NonConstScalar scale = {};
  Scalar* v = nullptr;
  int rowStride =
      0; /// @brief Stride (in count of Scalar) between elements with a 1 increment in row index.
  int colStride =
      0; /// @brief Stride (in count of Scalar) between elements with a 1 increment in column index.
};

/// @brief Light description of a matrix
/// @tparam Scalar Scalar type for matrix entries
///
/// @note Usage of this class assumes that either rowStride or colStride is equal to 1.
/// This assumption has to be maintained by the developer.
template <typename Scalar>
struct MatDataDest {
  Scalar* v = nullptr;
  int rowStride =
      0; /// @brief Stride (in count of Scalar) between elements with a 1 increment in row index.
  int colStride =
      0; /// @brief Stride (in count of Scalar) between elements with a 1 increment in column index.
};

// Forward-declaration of accessor type
template <typename Scalar, int kRowStrideAtCompileTime, int kColStrideAtCompileTime>
struct Accessor;

} // namespace mochi::details
