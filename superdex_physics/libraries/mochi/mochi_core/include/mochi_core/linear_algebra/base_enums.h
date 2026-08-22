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
#include <mochi_core/utils/reflection.h>

namespace mochi::krylov {

enum class Ownership {
  Owner = 0, /// @brief The Matrix/Vector owns the data stored on the main (or host) memory.
  View = 1, /// @brief The Matrix/Vector does not own the data. Can be combined with others
  Cuda = (1u << 1), /// @brief The Matrix/Vector's data is on a CUDA GPU
  CudaView = Cuda | View, /// @brief The Matrix/Vector does not own the data, which is on a CUDA GPU
};

/// @brief Map an ownership to its owning version (useful when creating compatible matrices/vector
/// to a view)
template <Ownership o>
constexpr Ownership Owning = static_cast<Ownership>(static_cast<int>(o) & (~1));

template <Ownership o>
constexpr Ownership Viewed = static_cast<Ownership>(static_cast<int>(o) | 1);

static_assert(Ownership::Owner == Owning<Ownership::Owner>);
static_assert(Ownership::Owner == Owning<Ownership::View>);
static_assert(Ownership::Cuda == Owning<Ownership::Cuda>);
static_assert(Ownership::Cuda == Owning<Ownership::CudaView>);

static_assert(Ownership::View == Viewed<Ownership::Owner>);
static_assert(Ownership::View == Viewed<Ownership::View>);
static_assert(Ownership::CudaView == Viewed<Ownership::Cuda>);
static_assert(Ownership::CudaView == Viewed<Ownership::CudaView>);

/// @brief Utility to determine if an Ownership is Cuda.
MOCHI_ANY MOCHI_FORCE_INLINE constexpr bool IsCuda(Ownership ownership) {
  return ownership == Ownership::Cuda || ownership == Ownership::CudaView;
}

/// @brief Utility to determine if an Ownership owns the data.
MOCHI_ANY MOCHI_FORCE_INLINE constexpr bool IsOwner(Ownership ownership) {
  return ownership == Ownership::Owner || ownership == Ownership::Cuda;
}

/// @brief Utility to determine if an Ownership is a view.
MOCHI_ANY MOCHI_FORCE_INLINE constexpr bool IsView(Ownership ownership) {
  return ownership == Ownership::View || ownership == Ownership::CudaView;
}

MOCHI_ANY MOCHI_FORCE_INLINE constexpr Ownership operator|(Ownership a, Ownership b) {
  return static_cast<Ownership>(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

/// @brief Enum for setting a row / column-major storage
enum class Direction {
  RowMajor = 0,
  ColMajor = 1,
};

MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto operator~(Direction dir) {
  return dir == Direction::RowMajor ? Direction::ColMajor : Direction::RowMajor;
}

} // namespace mochi::krylov

MOCHI_ENUM_BEGIN(mochi::krylov::Ownership);
MOCHI_ENUM_ITEM(Owner);
MOCHI_ENUM_ITEM(View);
MOCHI_ENUM_ITEM(Cuda);
MOCHI_ENUM_ITEM(CudaView);
MOCHI_ENUM_END();

MOCHI_ENUM_BEGIN(mochi::krylov::Direction);
MOCHI_ENUM_ITEM(RowMajor);
MOCHI_ENUM_ITEM(ColMajor);
MOCHI_ENUM_END();
