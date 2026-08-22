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

#include <mochi_core/geometry/any_shape.h>
#include <mochi_core/geometry/geometry_utils.h>

#include <variant>

using namespace mochi;

template <typename ShapeT>
static int FindPointsInShape(
    ShapeT shape,
    Span<Real3 const> inPoints,
    TransformRT const& shapeFromPoints,
    Span<Real3> outPointsInShape,
    Span<int> outIndices) {
  MOCHI_ASSERT(inPoints.size() == outPointsInShape.size(), "Incorrect output size");
  MOCHI_ASSERT(inPoints.size() == outIndices.size(), "Incorrect output size");
  int const numPoints = isize(inPoints);
  int count = 0;
  for (int i = 0; i < numPoints; ++i) {
    Real3 pointInShape = shapeFromPoints.TransformPoint(inPoints[i]);
    if (ContainsPoint(shape, pointInShape)) {
      outIndices[count] = i;
      outPointsInShape[count] = pointInShape;
      ++count;
    }
  }
  return count;
}

int mochi::FindPointsInAnyShape(
    AnyShape const& anyShape,
    Span<Real3 const> inPoints,
    TransformRT const& shapeFromPoints,
    Span<Real3> outPointsInShape,
    Span<int> outIndices) {
  return std::visit(
      [&](auto& shape) {
        return FindPointsInShape(shape, inPoints, shapeFromPoints, outPointsInShape, outIndices);
      },
      anyShape);
}
