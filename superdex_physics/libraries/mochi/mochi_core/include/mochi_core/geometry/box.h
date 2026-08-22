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

#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/reflection.h>

namespace mochi {

/**
 * @brief Data for an implicit box within a model file.
 *
 * @note Similar to @ref Obb (oriented bounding box), but with a serialization-friendly data
 * representation. Used in @ref ModelData.
 */
struct Box {
  Box() = default;
  Box(Real3 const& center, Real3 const& halfExtents, Quaternion const& rotation = {})
      : center(center), halfExtents(halfExtents), rotation(rotation) {}

  Real3 center{}; ///< Center of the box [m]
  Real3 halfExtents{1_r, 1_r, 1_r}; ///< Half extents of the box [m]
  Quaternion rotation; ///< Rotation about the center of the box

#if MOCHI_LANGUAGE_CPP20
  bool operator==(Box const& other) const = default;
  bool operator!=(Box const& other) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::Box)
  MOCHI_FIELD(center) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(halfExtents) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(rotation)
  MOCHI_STRUCT_END()
};

} // namespace mochi
