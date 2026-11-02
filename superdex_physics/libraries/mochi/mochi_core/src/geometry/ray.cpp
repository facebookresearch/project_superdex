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

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/ray.h>

#include <limits>
#include <optional>

using namespace mochi;

Aabb Triangle::GetBounds() const {
  auto min = Min(Min(v1, v2), v3);
  auto max = Max(Max(v1, v2), v3);

  return {min, max};
}

Vec4r Triangle::VGetDistanceSqr(Vec4r point, int /*index*/) const {
  VDistanceSignParams unused;
  return VDistancePointTriangleSqr(point, v1, v2, v3, unused);
}

std::optional<RayHit> mochi::RayCast(Ray const& ray, Aabb const& box) {
  Vec4r boxMin = box.VGetMin();
  Vec4r boxMax = box.VGetMax();

  Vec4r t1 = (boxMin - ray.origin) / ray.direction;
  Vec4r t2 = (boxMax - ray.origin) / ray.direction;

  real tmin = HMax<3>(Min(t1, t2));
  real tmax = HMin<3>(Max(t1, t2));

  if (tmax >= tmin && tmax >= 0_r) {
    RayHit hit{};
    hit.t = Max(0_r, tmin);
    hit.intersection = ray(hit.t);
    return hit;
  } else {
    return std::nullopt;
  }
}

Vec4r mochi::BarycentricCoordinates(Triangle const& triangle, Vec4r const& point) {
  Vec4r const e1 = triangle.v2 - triangle.v1;
  Vec4r const e2 = triangle.v3 - triangle.v1;
  Vec4r const w = point - triangle.v1;

  // Ratios of signed areas, accurate to about ε / sin(α) for the triangle's smallest angle α,
  // whatever the vertex order
  Vec4r const normal = Cross3(e1, e2);
  real const normalSqr = NormSqr<3>(normal);
  // Zero area up to rounding: for collinear edges, a fused cross product can be up to ε/2·|e1||e2|
  if (normalSqr <= Sqr(std::numeric_limits<real>::epsilon()) * NormSqr<3>(e1) * NormSqr<3>(e2))
      MOCHI_UNLIKELY {
    return {-1_r, -1_r, -1_r};
  }

  real const u = Dot<3>(Cross3(w, e2), normal) / normalSqr;
  real const v = Dot<3>(Cross3(e1, w), normal) / normalSqr;

  return {1_r - u - v, u, v};
}

std::optional<RayHit> mochi::RayCast(Ray const& ray, Triangle const& triangle) {
  // Calculate the edges of the triangle
  Vec4r e1 = triangle.v2 - triangle.v1;
  Vec4r e2 = triangle.v3 - triangle.v1;

  // Calculate the normal of the triangle
  Vec4r normal = Cross3(e1, e2);

  // Check if the ray is parallel to the triangle, up to rounding
  real dot = Dot<3>(ray.direction, normal);
  if (NearZero(
          dot, std::numeric_limits<real>::epsilon() * Norm<3>(ray.direction) * Norm<3>(normal))) {
    return std::nullopt; // Ray is parallel to the triangle
  }

  // Calculate the distance from the ray origin to the triangle
  real t = Dot<3>(normal, triangle.v1 - ray.origin) / dot;

  // Check if the intersection point is behind the ray origin
  if (t < 0) {
    return std::nullopt; // Intersection point is behind the ray origin
  }

  // Calculate the intersection point
  Vec4r intersection = ray(t);

  // Check if the intersection point is inside the triangle
  Vec4r barycentric = BarycentricCoordinates(triangle, intersection);
  if (HMin<3>(barycentric) < 0) {
    return std::nullopt; // Intersection point is outside the triangle
  }

  // Return the hit information
  RayHit hit{};
  hit.t = t;
  hit.intersection = intersection;
  return hit;
}

int TriangleSoup::GetNumElements() const {
  return isize(triangles);
}
Aabb TriangleSoup::GetBv(int index) const {
  return triangles[index].GetBounds();
}
real TriangleSoup::GetDistanceSqr(Real3 const& point, int index) const {
  return Get0(VGetDistanceSqr(ToSimd(point), index));
}
Vec4r TriangleSoup::VGetDistanceSqr(Vec4r point, int index) const {
  return triangles[index].VGetDistanceSqr(point, index);
}

std::optional<RayHit> const& mochi::CloserHit(
    std::optional<RayHit> const& h1,
    std::optional<RayHit> const& h2) {
  if (h1 && h2) {
    return (h1->t < h2->t) ? h1 : h2;
  }
  return h1 ? h1 : h2;
}

std::optional<RayHit>
mochi::RayCast(Ray const& ray, AabbTree const& bvhTree, BvhRayCastFn const& callback, int nodeIdx) {
  auto const& node = bvhTree.GetNode(nodeIdx);

  if (!RayCast(ray, node.bv)) {
    // Ray misses this node
    return std::nullopt;
  } else {
    if (node.isLeafNode) {
      // Intersect ray with all elements
      auto elements = bvhTree.GetElements(node);

      std::optional<RayHit> hit = std::nullopt;
      for (auto idx : elements) {
        hit = CloserHit(hit, callback(ray, idx));
      }
      return hit;
    } else {
      // Intersect ray with children
      auto hit1 = RayCast(ray, bvhTree, callback, node.leftChildIndex);
      auto hit2 = RayCast(ray, bvhTree, callback, node.rightChildIndex);
      return CloserHit(hit1, hit2);
    }
  }
}

std::optional<RayHit>
mochi::RayCast(Ray const& ray, AabbTree const& bvhTree, TriangleSoup const& bvhObject) {
  auto callback = [&](Ray const& ray, int index) {
    auto result = RayCast(ray, bvhObject.triangles[index]);
    if (result) {
      // Correctly set the index of the primitive that generated this hit
      result->index = index;
    }
    return result;
  };
  return RayCast(ray, bvhTree, callback);
}

std::optional<RayHit> mochi::RayCast(Ray const& ray, TriangleSoup const& triangles) {
  std::optional<RayHit> hit;
  for (int i = 0; i < triangles.triangles.size(); ++i) {
    auto const& triangle = triangles.triangles[i];
    auto newHit = RayCast(ray, triangle);
    if (newHit) {
      newHit->index = i;
    }
    hit = CloserHit(hit, newHit);
  }
  return hit;
}
