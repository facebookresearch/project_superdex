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
#include <mochi_core/geometry/bvh_tree.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

#include <functional>
#include <optional>
#include <stack>
#include <utility>
#include <vector>

namespace mochi {
struct Ray {
  Vec4r origin;
  Vec4r direction;

  MOCHI_FORCE_INLINE Ray() : origin(SimdZero()), direction(SimdZero()) {}
  MOCHI_FORCE_INLINE Ray(Vec4r origin, Vec4r direction) : origin(origin), direction(direction) {}
  MOCHI_FORCE_INLINE Ray(Real3 origin, Real3 direction)
      : origin(ToSimd(origin)), direction(ToSimd(direction)) {}

  MOCHI_FORCE_INLINE Vec4r operator()(real t) const {
    return origin + t * direction;
  }
};

struct RayHit {
  // Where along the ray the intersection happened, or zero if the ray origin lies inside the target
  // volume.
  real t;
  // The coordinates of the intersection point, or the ray origin if it already lies inside the
  // target volume.
  Vec4r intersection;
  // The index of a primitive in a complex object that was hit
  int index;
};

// Selects the closer of two hits
std::optional<RayHit> const& CloserHit(
    std::optional<RayHit> const& h1,
    std::optional<RayHit> const& h2);

struct Triangle {
  Vec4r v1;
  Vec4r v2;
  Vec4r v3;

  inline Triangle(Vec4r v1, Vec4r v2, Vec4r v3) : v1(v1), v2(v2), v3(v3) {}
  inline Triangle(Real3 v1, Real3 v2, Real3 v3) : v1(ToSimd(v1)), v2(ToSimd(v2)), v3(ToSimd(v3)) {}

  Aabb GetBounds() const;
  Vec4r VGetDistanceSqr(Vec4r point, int index) const;
};

struct TriangleSoup : public BvhObject<Aabb> {
  std::vector<Triangle> triangles;

  TriangleSoup() = default;
  TriangleSoup(std::vector<Triangle>&& v) : triangles(std::move(v)) {}

  int GetNumElements() const override;
  Aabb GetBv(int index) const override;
  real GetDistanceSqr(Real3 const& point, int index) const override;
  Vec4r VGetDistanceSqr(Vec4r point, int index) const override;

  MOCHI_FORCE_INLINE auto begin() {
    return triangles.begin();
  }
  MOCHI_FORCE_INLINE auto end() {
    return triangles.end();
  }
  MOCHI_FORCE_INLINE auto begin() const {
    return triangles.begin();
  }
  MOCHI_FORCE_INLINE auto end() const {
    return triangles.end();
  }
  MOCHI_FORCE_INLINE Triangle& operator[](int i) {
    return triangles[i];
  }
  MOCHI_FORCE_INLINE Triangle const& operator[](int i) const {
    return triangles[i];
  }
};

/**
 * @brief Compute barycentric coordinates of @ref point with respect to @ref triangle.
 *
 * @return (λ₁, λ₂, λ₃) such that point ≈ λ₁·v1 + λ₂·v2 + λ₃·v3.
 *
 * @note The point is inside the triangle when all components are non-negative. For degenerate
 * triangles (zero area), returns all-negative coordinates.
 */
Vec4r BarycentricCoordinates(Triangle const& triangle, Vec4r const& point);

using BvhRayCastFn = std::function<std::optional<RayHit>(Ray const& ray, int index)>;

// Ray intersection functions
std::optional<RayHit> RayCast(Ray const& ray, Aabb const& box);
std::optional<RayHit> RayCast(Ray const& ray, Triangle const& triangle);
std::optional<RayHit>
RayCast(Ray const& ray, AabbTree const& bvhTree, BvhRayCastFn const& callback, int nodeIdx = 0);
std::optional<RayHit>
RayCast(Ray const& ray, AabbTree const& bvhTree, TriangleSoup const& bvhObject);

// NOTE: this function is O(n), prefer to build an acceleration structure if the triangle
// soup is ray cast against repeatedly.
std::optional<RayHit> RayCast(Ray const& ray, TriangleSoup const& triangles);
} // namespace mochi
