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

#include <mochi_core/geometry/aabb.h>
#include <mochi_core/geometry/obb.h>
#include <mochi_core/geometry/plane.h>
#include <mochi_core/geometry/sphere.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

#include <variant>

namespace mochi {

/**************************************************************************************************
  AnyShape - Variant class for a variety 3D geometric shapes.
             Note that the default constructor uses the first type alternative.
             In other words, AnyShape{} is equivalent to AnyShape{Sphere{}}.
*/

using AnyShape = std::variant<Sphere, Aabb, Obb, Plane>;

/*************************************************************************************************
 Utils for AnyShape
*/

int FindPointsInAnyShape(
    AnyShape const& anyShape,
    Span<Real3 const> inPoints,
    TransformRT const& shapeFromPoints,
    Span<Real3> outPointsInShape,
    Span<int> outIndices);

} // namespace mochi
