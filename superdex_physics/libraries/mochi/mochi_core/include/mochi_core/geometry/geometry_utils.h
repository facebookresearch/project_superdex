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
#include <mochi_core/geometry/any_shape.h>
#include <mochi_core/geometry/batch_sphere.h>
#include <mochi_core/geometry/capsule.h>
#include <mochi_core/geometry/obb.h>
#include <mochi_core/geometry/plane.h>
#include <mochi_core/geometry/sphere.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_core/utils/transform_srt.h>
#include <mochi_core/utils/vmatrix.h>

#include <limits>
#include <variant>

namespace mochi {

/**************************************************************************************************
  NearEqual: (abs(a-b) <= epsilon)
*/

[[nodiscard]] MOCHI_FORCE_INLINE bool NearEqual(Aabb const& a, Aabb const& b, Vec4r epsilon) {
  return AllTrue<3>(
      VNearEqual(a.VGetMin(), b.VGetMin(), epsilon) &
      VNearEqual(a.VGetMax(), b.VGetMax(), epsilon));
}

[[nodiscard]] MOCHI_FORCE_INLINE bool
NearEqual(Aabb const& a, Aabb const& b, real epsilon = kDefaultNearEqualEpsilon<real>) {
  return NearEqual(a, b, Vec4r{epsilon});
}

[[nodiscard]] MOCHI_FORCE_INLINE bool NearEqual(Capsule const& a, Capsule const& b, Vec4r epsilon) {
  return AllTrue<4>(
      VNearEqual(a.VGetPackedARadius(), b.VGetPackedARadius(), epsilon) &
      VNearEqual(a.VGetAB(), b.VGetAB(), epsilon));
}

[[nodiscard]] MOCHI_FORCE_INLINE bool
NearEqual(Capsule const& a, Capsule const& b, real epsilon = kDefaultNearEqualEpsilon<real>) {
  return NearEqual(a, b, Vec4r{epsilon});
}

[[nodiscard]] MOCHI_FORCE_INLINE bool NearEqual(Obb const& a, Obb const& b, Vec4r epsilon) {
  return NearEqual(a.GetTransform(), b.GetTransform(), epsilon) &&
      NearEqual(a.VGetHalfExtents(), b.VGetHalfExtents(), epsilon);
}

[[nodiscard]] MOCHI_FORCE_INLINE bool
NearEqual(Obb const& a, Obb const& b, real epsilon = kDefaultNearEqualEpsilon<real>) {
  return NearEqual(a, b, Vec4r{epsilon});
}

[[nodiscard]] MOCHI_FORCE_INLINE bool NearEqual(Plane const& a, Plane const& b, Vec4r epsilon) {
  return NearEqual(a.VGetPacked(), b.VGetPacked(), epsilon);
}

[[nodiscard]] MOCHI_FORCE_INLINE bool
NearEqual(Plane const& a, Plane const& b, real epsilon = kDefaultNearEqualEpsilon<real>) {
  return NearEqual(a, b, Vec4r{epsilon});
}

[[nodiscard]] MOCHI_FORCE_INLINE bool NearEqual(Sphere const& a, Sphere const& b, Vec4r epsilon) {
  return NearEqual(a.VGetPacked(), b.VGetPacked(), epsilon);
}

[[nodiscard]] MOCHI_FORCE_INLINE bool
NearEqual(Sphere const& a, Sphere const& b, real epsilon = kDefaultNearEqualEpsilon<real>) {
  return NearEqual(a, b, Vec4r{epsilon});
}

/**************************************************************************************************
  TransformShape
*/

[[nodiscard]] MOCHI_FORCE_INLINE Sphere
TransformShape(TransformRT const& transform, Sphere const& s) {
  return {transform.TransformPoint(s.VGetCenter()), s.GetRadius()};
}

[[nodiscard]] MOCHI_FORCE_INLINE Plane
TransformShape(TransformRT const& transform, Plane const& p) {
  Vec4r norm1 = p.VGetNormal();
  Vec4r norm2 = transform.GetRotation() * norm1;
  Vec4r pt1 = norm1 * p.GetDistanceFromOrigin(); // point on plane
  Vec4r pt2 = transform.TransformPoint(pt1);
  return {norm2, Dot<3>(pt2, norm2)};
}

// Takes a TRANSPOSED matrix
[[nodiscard]] MOCHI_FORCE_INLINE Aabb
TransformShape_Transposed(VMatrix4x4r const& transformT, Aabb const& aabb) {
  Vec4r diag = aabb.VGetMax() - aabb.VGetMin();
  Vec4r x = transformT[0] * Shuffle<0, 0, 0, 0>(diag);
  Vec4r y = transformT[1] * Shuffle<1, 1, 1, 1>(diag);
  Vec4r z = transformT[2] * Shuffle<2, 2, 2, 2>(diag);
  Vec4r p = DotVecMat4x4(aabb.VGetMin(), transformT);
  Vec4r verts[8];
  verts[0] = p;
  verts[1] = p + x;
  verts[2] = p + y;
  verts[3] = p + z;
  verts[4] = verts[1] + y; // p + x + y
  verts[5] = verts[2] + z; // p + y + z
  verts[6] = verts[3] + x; // p + z + x
  verts[7] = verts[4] + z; // p + x + y + z
  // clang-format off
  Vec4r min = Min(verts[0], Min(verts[1], Min(verts[2], Min(verts[3], Min(verts[4], Min(verts[5], Min(verts[6], verts[7])))))));
  Vec4r max = Max(verts[0], Max(verts[1], Max(verts[2], Max(verts[3], Max(verts[4], Max(verts[5], Max(verts[6], verts[7])))))));
  // clang-format on
  return {min, max};
}

// NOTE: The volume of an Aabb may increase when rotated.
[[nodiscard]] MOCHI_FORCE_INLINE Aabb
TransformShape(VMatrix4x4r const& transform, Aabb const& aabb) {
  return TransformShape_Transposed(Transpose4x4(transform), aabb);
}

// NOTE: The volume of an Aabb may increase when rotated.
[[nodiscard]] MOCHI_FORCE_INLINE Aabb
TransformShape(TransformRT const& transform, Aabb const& aabb) {
  return TransformShape_Transposed(ToVMatrix4x4Transpose(transform), aabb);
}

[[nodiscard]] MOCHI_FORCE_INLINE Capsule
TransformShape(TransformRT const& transform, Capsule const& capsule) {
  Vec4r a = transform.TransformPoint(capsule.VGetA()); // rotate and translate point A
  Vec4r atob = transform.TransformDirection(capsule.VGetAB()); // rotate the a-to-b vector
  return Capsule::FromPointAndVector(a, atob, capsule.GetRadius());
}

[[nodiscard]] MOCHI_FORCE_INLINE Capsule
TransformShape(TransformSRT const& transform, Capsule const& capsule) {
  Vec4r a = transform.TransformPoint(capsule.VGetA()); // scale rotate and translate point A
  Vec4r atob = transform.VGetScale() * capsule.VGetAB(); // scale the a-to-b vector
  atob = transform.TransformDirection(atob); // then rotate the a-to-b vector
  return Capsule::FromPointAndVector(a, atob, transform.GetScale() * capsule.GetRadius());
}

[[nodiscard]] MOCHI_FORCE_INLINE Obb
TransformShape(MatrixTransformRT const& transform, Obb const& oobb) {
  return {transform * oobb.GetTransform(), oobb.VGetHalfExtents()};
}

[[nodiscard]] MOCHI_FORCE_INLINE Obb TransformShape(TransformRT const& transform, Obb const& oobb) {
  return TransformShape(ToMatrixTransformRT(transform), oobb);
}

[[nodiscard]] MOCHI_FORCE_INLINE AnyShape
TransformShape(TransformRT const& transform, AnyShape const& any) {
  // Call TransformShape(transform, x) where x is the type stored in the variant.
  return std::visit([&transform](auto& x) { return AnyShape{TransformShape(transform, x)}; }, any);
}

/**************************************************************************************************
  Axis-Aligned Bounding Boxes
*/

// Compute an Aabb that contains all the coordinates
[[nodiscard]] Aabb CalcAabb(Span<Real3 const> coordinates);

// Compute an Aabb that contains the (coordinates + displacements) specified by index.
// WARNING: The indices must be strictly increasing.
[[nodiscard]] Aabb CalcAabbWithSortedIndices(
    Span<Real3 const> coordinates,
    Span<Real3 const> displacement,
    Span<int const> sortedIndices);

// Compute an Aabb that contains all the displaced coordinates.
[[nodiscard]] Aabb CalcAabbWithDisplacements(
    Span<Real3 const> coordinates,
    Span<Real3 const> displacements);

// Return the same Aabb (just so GetAabb works with any shape)
[[nodiscard]] MOCHI_FORCE_INLINE Aabb GetAabb(Aabb const& aabb) {
  return aabb;
}

// Get the Aabb that contains both
[[nodiscard]] MOCHI_FORCE_INLINE Aabb GetAabb(Aabb const& a, Aabb const& b) {
  return {Min(a.VGetMin(), b.VGetMin()), Max(a.VGetMax(), b.VGetMax())};
}

// Get the Aabb that contains a sphere
[[nodiscard]] MOCHI_FORCE_INLINE Aabb GetAabb(Sphere const& sphere) {
  Vec4r center = sphere.VGetPacked(); // (x, y, z, r)
  Vec4r diag = Broadcast<3>(center); // (r, r, r, r)
  return {center - diag, center + diag};
}

// Get the Aabb that contains the Obb
[[nodiscard]] MOCHI_FORCE_INLINE Aabb GetAabb(Obb const& oobb) {
  auto corners = oobb.VGetCorners();
  Vec4r min = corners[0];
  Vec4r max = corners[0];
  for (int i = 1; i < 8; ++i) {
    min = Min(min, corners[i]);
    max = Max(max, corners[i]);
  }
  return {min, max};
}

// For collision detection, we consider the volume of a plane to include everything below it
// (opposite direction from the plane's normal). This is only really useful for axis-aligned
// planes since an arbitrarily rotated plane would have infinite bounds in all directions.
[[nodiscard]] inline Aabb GetAabb(Plane const& plane) {
  real constexpr one = 0.99999_r; // close enough
  real constexpr inf = std::numeric_limits<real>::infinity();
  real const dist = plane.GetDistanceFromOrigin();
  Real3 const norm = plane.GetNormal();
  if (norm[0] >= one) { // +x
    return {Real3{-inf, -inf, -inf}, Real3{dist, inf, inf}};
  } else if (norm[1] >= one) { // +y
    return {Real3{-inf, -inf, -inf}, Real3{inf, dist, inf}};
  } else if (norm[2] >= one) { // +z
    return {Real3{-inf, -inf, -inf}, Real3{inf, inf, dist}};
  } else if (norm[0] <= -one) { // -x
    return {Real3{-dist, -inf, -inf}, Real3{inf, inf, inf}};
  } else if (norm[1] <= -one) { // -y
    return {Real3{-inf, -dist, -inf}, Real3{inf, inf, inf}};
  } else if (norm[2] <= -one) { // -z
    return {Real3{-inf, -inf, -dist}, Real3{inf, inf, inf}};
  } else {
    return {Real3{-inf, -inf, -inf}, Real3{inf, inf, inf}};
  }
}

[[nodiscard]] MOCHI_FORCE_INLINE Aabb GetAabb(AnyShape const& any) {
  // Call GetAabb(x) where x is the type stored in the variant.
  return std::visit([](auto& x) { return GetAabb(x); }, any);
}

/**************************************************************************************************
  Object-Oriented Bounding Boxes
*/

// Compute an Obb that contains all the coordinates
[[nodiscard]] Obb CalcObb(Span<Real3 const> coordinates);

// Compute an Obb that contains all the coordinates.
[[nodiscard]] Obb CalcObb(Span<Vec4r const> coordinates);

// Return the same Obb (just so GetObb works with any shape)
[[nodiscard]] MOCHI_FORCE_INLINE Obb GetObb(Obb const& oobb) {
  return oobb;
}

// Get the equivalent Obb to the given Aabb.
[[nodiscard]] MOCHI_FORCE_INLINE Obb GetObb(Aabb const& aabb) {
  return {MatrixTransformRT{VEye<3>(), aabb.VGetCenter()}, aabb.VGetSize() * 0.5_r};
}

// Get the Obb that contains the sphere.
[[nodiscard]] MOCHI_FORCE_INLINE Obb GetObb(Sphere const& sphere) {
  Vec4r center = sphere.VGetPacked(); // (x, y, z, r)
  Vec4r diag = Broadcast<3>(center); // (r, r, r, r)
  return {MatrixTransformRT{VEye<3>(), center}, diag};
}

[[nodiscard]] MOCHI_FORCE_INLINE Obb GetObb(Plane const& plane) {
  return GetObb(GetAabb(plane));
}

// Get the Obb that contains any shape
[[nodiscard]] MOCHI_FORCE_INLINE Obb GetObb(AnyShape const& any) {
  // Call GetObb(x) where x is the type stored in the variant.
  return std::visit([](auto& x) { return GetObb(x); }, any);
}

/**************************************************************************************************
  Bounding Spheres
*/

/** @brief Selects the tradeoff between bounding-sphere tightness and computation cost. */
enum class BoundingSphereAlgorithm {
  Fastest, ///< Sphere may be larger than necessary. Very fast to compute.
  Fast, ///< Sphere is typically 5-10% smaller than @ref Fastest. Still quite fast.
  Best, ///< Smallest sphere within floating-point precision; typically O(N) but slower.
  Count ///< Number of algorithms; not a valid selection.
};

/**
 * @brief Compute a bounding sphere containing all the coordinates.
 *
 * @param coordinates 3D coordinates
 * @param algorithm Algorithm to use for computing the bounding sphere (size vs speed tradeoff)
 * @return Sphere containing all the coordinates
 *
 * @see CalcBoundingSphereIndexed
 */
[[nodiscard]] Sphere CalcBoundingSphere(
    Span<Real3 const> coordinates,
    BoundingSphereAlgorithm algorithm);

/**
 * @brief Compute a bounding sphere containing all the coordinates, referenced by index.
 *
 * @param coordinates 3D coordinates
 * @param indices Indices into the @p coordinates array.
 * @param algorithm Algorithm to use for computing the bounding sphere (size vs speed tradeoff)
 * @return Bounding sphere
 *
 * @pre Every element of @p indices is a valid index into @p coordinates.
 *
 * @see CalcBoundingSphere
 */
[[nodiscard]] Sphere CalcBoundingSphereIndexed(
    Span<Real3 const> coordinates,
    Span<int const> indices,
    BoundingSphereAlgorithm algorithm);

// Return the same sphere (just so GetBoundingSphere works with any shape)
[[nodiscard]] MOCHI_FORCE_INLINE Sphere const& GetBoundingSphere(Sphere const& passthru) {
  return passthru;
}

// Compute a sphere that contains the given Plane (infinite radius)
[[nodiscard]] MOCHI_FORCE_INLINE Sphere GetBoundingSphere(Plane const& plane) {
  Vec4r pointOnPlane = plane.VGetPacked() * plane.GetDistanceFromOrigin();
  return {pointOnPlane, std::numeric_limits<real>::infinity()};
}

// Compute a sphere that contains the given Aabb
[[nodiscard]] MOCHI_FORCE_INLINE Sphere GetBoundingSphere(Aabb const& aabb) {
  real radius = 0.5_r * Norm<3>(aabb.VGetSize());
  return {aabb.GetCenter(), radius};
}

// Compute a sphere that contains the given Obb
[[nodiscard]] MOCHI_FORCE_INLINE Sphere GetBoundingSphere(Obb const& oobb) {
  real radius = 0.5_r * Norm<3>(oobb.VGetSize());
  return {oobb.GetCenter(), radius};
}

// Compute the bounding sphere for AnyShape
[[nodiscard]] MOCHI_FORCE_INLINE Sphere GetBoundingSphere(AnyShape const& any) {
  // Call GetBoundingSphere(x) where x is the type stored in the variant.
  return std::visit([](auto& x) { return GetBoundingSphere(x); }, any);
}

/**************************************************************************************************
  General Bounding Volume of different primitives, to allow templatized calls
*/

MOCHI_FORCE_INLINE void GetBoundingVolume(Sphere const& sphere, Aabb& outAabb) {
  outAabb = GetAabb(sphere);
}

MOCHI_FORCE_INLINE void GetBoundingVolume(Sphere const& sphere, Obb& outObb) {
  outObb = GetObb(sphere);
}

MOCHI_FORCE_INLINE void GetBoundingVolume(Sphere const& sphere, Sphere& outSphere) {
  outSphere = sphere;
}

MOCHI_FORCE_INLINE void GetBoundingVolume(Aabb const& aabb, Aabb& outAabb) {
  outAabb = aabb;
}

MOCHI_FORCE_INLINE void GetBoundingVolume(Aabb const& aabb, Obb& outObb) {
  outObb = GetObb(aabb);
}

MOCHI_FORCE_INLINE void GetBoundingVolume(Aabb const& aabb, Sphere& outSphere) {
  outSphere = GetBoundingSphere(aabb);
}

/**************************************************************************************************
  Surface area
*/

[[nodiscard]] MOCHI_FORCE_INLINE real CalcShapeSurfaceArea(Aabb const& aabb) {
  auto const size = aabb.VGetSize();
  auto const areas = size * Shuffle<1, 2, 0, 3>(size);
  return 2.0_r * HSum<3>(areas);
}

[[nodiscard]] MOCHI_FORCE_INLINE real CalcShapeSurfaceArea(Obb const& oobb) {
  auto const extents = oobb.VGetHalfExtents();
  auto const quarterAreas = extents * Shuffle<1, 2, 0, 3>(extents);
  return 8.0_r * HSum<3>(quarterAreas);
}

[[nodiscard]] MOCHI_FORCE_INLINE real CalcShapeSurfaceArea(Sphere const& sphere) {
  return 4.0_r * kPI * sphere.GetRadius() * sphere.GetRadius();
}

/**************************************************************************************************
  Volume
*/

[[nodiscard]] MOCHI_FORCE_INLINE real GetVolume(Aabb const& aabb) {
  return HProd<3>(aabb.VGetSize());
}

[[nodiscard]] MOCHI_FORCE_INLINE real GetVolume(Obb const& oobb) {
  return HProd<3>(oobb.VGetSize());
}

[[nodiscard]] MOCHI_FORCE_INLINE real GetVolume(Sphere const& sphere) {
  auto const radius = sphere.GetRadius();
  return (4.0_r / 3.0_r) * kPI * radius * radius * radius;
}

[[nodiscard]] MOCHI_FORCE_INLINE real GetVolume(Plane const& /*plane*/) {
  return std::numeric_limits<real>::infinity();
}

[[nodiscard]] MOCHI_FORCE_INLINE real GetVolume(AnyShape const& any) {
  return std::visit([](auto& x) { return GetVolume(x); }, any);
}

/**************************************************************************************************
  Bounding volume enclosing
*/

[[nodiscard]] inline Aabb EncloseShapes(Aabb const& a, Aabb const& b) {
  return GetAabb(a, b);
}

[[nodiscard]] inline Obb EncloseShapes(Obb const& a, Obb const& b) {
  // clang-format off
  NdArray<Vec4r, 16> points = {
    a.VGetCorner<0>(), a.VGetCorner<1>(), a.VGetCorner<2>(), a.VGetCorner<3>(),
    a.VGetCorner<4>(), a.VGetCorner<5>(), a.VGetCorner<6>(), a.VGetCorner<7>(),
    b.VGetCorner<0>(), b.VGetCorner<1>(), b.VGetCorner<2>(), b.VGetCorner<3>(),
    b.VGetCorner<4>(), b.VGetCorner<5>(), b.VGetCorner<6>(), b.VGetCorner<7>()
  };
  // clang-format on
  return CalcObb(Span<Vec4r const>(MakeSpan(points)));
}

[[nodiscard]] inline Sphere EncloseShapes(Sphere const& a, Sphere const& b) {
  Vec4r aVals = a.VGetPacked();
  Vec4r bVals = b.VGetPacked();
  Vec4r offset = aVals - bVals;
  Vec4r distance = VNorm<3>(offset);

  // Check if a encloses b
  if (Get<3>(offset - distance) >= 0_r) {
    return a;
  }

  // Check if b encloses a
  if (Get<3>(-offset - distance) >= 0_r) {
    return b;
  }

  // Otherwise place the center at the midpoint of the two most distant points.
  // If no sphere encloses the other one, the distance cannot be zero.
  Vec4r direction = offset / distance;
  Vec4r radiusDiff = Broadcast<3>(offset);
  Vec4r center = 0.5_r * (direction * radiusDiff + aVals + bVals);
  Vec4r radius = 0.5_r * (aVals + bVals + distance);
  return {Blend<0, 0, 0, 1>(center, radius)};
}

/**************************************************************************************************
  Bounding volume expansion
*/

// Expands the Aabb by [expand] in all directions.
[[nodiscard]] MOCHI_FORCE_INLINE Aabb ExpandShape(Aabb const& aabb, real expand) {
  return {aabb.VGetMin() - expand, aabb.VGetMax() + expand};
}

// Expands the Obb by [expand] in all directions.
[[nodiscard]] MOCHI_FORCE_INLINE Obb ExpandShape(Obb const& oobb, real expand) {
  return {oobb.GetTransform(), oobb.VGetHalfExtents() + expand};
}

// Expands the Sphere by [expand] in all directions.
[[nodiscard]] MOCHI_FORCE_INLINE Sphere ExpandShape(Sphere const& sphere, real expand) {
  return {sphere.GetCenter(), sphere.GetRadius() + expand};
}

// Expands the "volume" of the plane shifting the plane along the normal. Note that our collision
// code considers anything under the plane to be an overlap.
[[nodiscard]] MOCHI_FORCE_INLINE Plane ExpandShape(Plane const& plane, real expand) {
  return {plane.GetNormal(), plane.GetDistanceFromOrigin() + expand};
}

// Extends the shape by [expand] in all directions.
[[nodiscard]] MOCHI_FORCE_INLINE AnyShape ExpandShape(AnyShape const& any, real expand) {
  // Call ExpandShape(x, expand) where x is the type stored in the variant.
  return std::visit([expand](auto& x) { return AnyShape{ExpandShape(x, expand)}; }, any);
}

/**************************************************************************************************
  Direction of maximum spread.
*/

// Retrieves the direction of maximum spread of the Aabb.
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r GetMaxSpreadDirection(Aabb const& aabb) {
  auto const direction = ArgMax(aabb.GetSize());
  MOCHI_ASSERT(direction < 3, "Unable to determine direction of maximum spread.");
  return SimdBasisVector(static_cast<int>(direction));
}

// Retrieves the direction of maximum spread of the Obb.
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r GetMaxSpreadDirection(Obb const& oobb) {
  auto const direction = ArgMax(oobb.GetHalfExtents());
  auto const axes = Transpose3x3(oobb.VGetRotation());
  MOCHI_ASSERT(direction < 3, "Unable to determine direction of maximum spread.");
  return axes[direction];
}

/**************************************************************************************************
  ContainsPoint
*/

[[nodiscard]] MOCHI_FORCE_INLINE Vec4r VContainsPoint(Aabb const& aabb, Vec4r pt) {
  return (pt >= aabb.VGetMin()) & (pt <= aabb.VGetMax());
}

[[nodiscard]] MOCHI_FORCE_INLINE Vec4r VContainsPoint(Obb const& oobb, Vec4r pt) {
  Vec4r const q = Abs(oobb.GetTransform().InverseTransformPoint(pt));
  return q <= oobb.VGetHalfExtents();
}

[[nodiscard]] MOCHI_FORCE_INLINE Vec4r VContainsPoint(Sphere const& s, Vec4r pt) {
  Vec4r packed = s.VGetPacked(); // (x, y, z, r)
  Vec4r ray = packed - pt;
  Vec4r distSqr = VDot<3>(ray, ray);
  Vec4r radius = Broadcast<3>(packed);
  return distSqr <= radius * radius;
}

[[nodiscard]] MOCHI_FORCE_INLINE Vec4r VContainsPoint(Plane const& p, Vec4r pt) {
  // A plane contains all the points or or below the surface of the plane.
  // This definition is consistent with other utilities in this library.
  Vec4r ptDist = VDot<3>(pt, p.VGetPacked());
  return ptDist <= p.GetDistanceFromOrigin();
}

// Helper if you need a bool return
template <class ShapeT, MOCHI_CONCEPT(IsPrimitiveShape<ShapeT>)>
[[nodiscard]] MOCHI_FORCE_INLINE bool ContainsPoint(ShapeT const& s, Vec4r pt) {
  return AllTrue<3>(VContainsPoint(s, pt));
}
template <class ShapeT, MOCHI_CONCEPT(IsPrimitiveShape<ShapeT>)>
[[nodiscard]] MOCHI_FORCE_INLINE bool ContainsPoint(ShapeT const& s, Real3 pt) {
  return ContainsPoint<ShapeT>(s, ToSimd(pt));
}

/**************************************************************************************************
  HasOverlap
*/

[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(Aabb const& a, Aabb const& b) {
  return AllTrue<3>((a.VGetMin() <= b.VGetMax()) & (b.VGetMin() <= a.VGetMax()));
}

[[nodiscard]] inline bool HasOverlap(Obb const& a, Obb const& b) {
  // Overlap algorithm based on the Separating Axis Theorem. Implemented as described in Christer
  // Ericson's "Real-time Collision Detection" book.

  // The separating axis theorem, also known as hyperplane separation theorem, states that:
  // "Two convex objects do not overlap if there exists a line (called axis) onto which the two
  // objects' projections do not overlap". In the case of Obbs 15 separating axes must be tested at
  // most to determine the overlap: the 3 axes of [a], the 3 axes of [b], and the 9 axes
  // perpendicular to all of them.

  // To simplify the procedure, bring [b]'s reference frame to [a]:
  // t = Raᵀ (tb - ta) = (tb - ta)ᵀ Ra
  // R = Raᵀ Rb
  auto const& Ta = a.GetTransform();
  auto const& Tb = b.GetTransform();
  Vec4r t = DotVecMat3x3(Tb.VGetTranslation() - Ta.VGetTranslation(), Ta.VGetRotation());
  VMatrix3x3r R = Dot3x3(Transpose3x3(Ta.VGetRotation()), Tb.VGetRotation());

  // Add in an epsilon term to counteract arithmetic errors when two edges are parallel and their
  // cross product is (near) null.
  auto eps = std::numeric_limits<real>::epsilon();
  VMatrix3x3r Rabs = {Abs(R[0]) + eps, Abs(R[1]) + eps, Abs(R[2]) + eps};

  // Retrieve extents of both Obbs.
  Vec4r ea = a.VGetHalfExtents();
  Vec4r eb = b.VGetHalfExtents();

  // Test the 3 axes of [a].
  // for i in range(3):
  //    ra = e1[i]
  //    rb = np.dot(e2, Rabs[i, :])
  //    d = np.abs(t[i])
  //    if d > (ra + rb): return False
  Vec4r rb = DotMatVec3x3(Rabs, eb);
  Vec4r da = Abs(t);
  if (AnyTrue<3>(da > ea + rb)) {
    return false;
  }

  // Test the 3 axes of [b].
  // for i in range(3):
  //     ra = np.dot(e1, Rabs[:, i])
  //     rb = e2[i]
  //     d = np.abs(np.dot(t, R[:, i]))
  //     if d > (ra + rb): return False
  Vec4r ra = DotVecMat3x3(ea, Rabs);
  Vec4r db = Abs(DotVecMat3x3(t, R));
  if (AnyTrue<3>(db > ra + eb)) {
    return false;
  }

  // Test the other 9 axes corresponding to their cross products.
  // Compute helper terms terms.
  Vec4r eb110 = Shuffle<1, 0, 0>(eb);
  Vec4r eb221 = Shuffle<2, 2, 1>(eb);

  // ra00 = ea[1] * Rabs[2, 0] + ea[2] * Rabs[1, 0]
  // ra01 = ea[1] * Rabs[2, 1] + ea[2] * Rabs[1, 1]
  // ra02 = ea[1] * Rabs[2, 2] + ea[2] * Rabs[1, 2]
  // rb00 = eb[1] * Rabs[0, 2] + eb[2] * Rabs[0, 1]
  // rb01 = eb[0] * Rabs[0, 2] + eb[2] * Rabs[0, 0]
  // rb02 = eb[0] * Rabs[0, 1] + eb[1] * Rabs[0, 0]
  // d00 = np.abs(t[2] * R[1, 0] - t[1] * R[2, 0])
  // d01 = np.abs(t[2] * R[1, 1] - t[1] * R[2, 1])
  // d02 = np.abs(t[2] * R[1, 2] - t[1] * R[2, 2])
  // if d00 > ra00 + rb00: return False
  // if d01 > ra01 + rb01: return False
  // if d02 > ra02 + rb02: return False
  Vec4r ra0 = (Broadcast<1>(ea) * Rabs[2]) + (Broadcast<2>(ea) * Rabs[1]);
  Vec4r rb0 = (eb110 * Shuffle<2, 2, 1>(Rabs[0])) + (eb221 * Shuffle<1, 0, 0>(Rabs[0]));
  Vec4r d0 = Abs((Broadcast<2>(t) * R[1]) - (Broadcast<1>(t) * R[2]));
  if (AnyTrue<3>(d0 > ra0 + rb0)) {
    return false;
  }

  // ra10 = ea[0] * Rabs[2, 0] + ea[2] * Rabs[0, 0]
  // ra11 = ea[0] * Rabs[2, 1] + ea[2] * Rabs[0, 1]
  // ra12 = ea[0] * Rabs[2, 2] + ea[2] * Rabs[0, 2]
  // rb10 = eb[1] * Rabs[1, 2] + eb[2] * Rabs[1, 1]
  // rb11 = eb[0] * Rabs[1, 2] + eb[2] * Rabs[1, 0]
  // rb12 = eb[0] * Rabs[1, 1] + eb[1] * Rabs[1, 0]
  // d10 = np.abs(t[0] * R[2, 0] - t[2] * R[0, 0])
  // d11 = np.abs(t[0] * R[2, 1] - t[2] * R[0, 1])
  // d12 = np.abs(t[0] * R[2, 2] - t[2] * R[0, 2])
  // if d10 > ra10 + rb10: return False
  // if d11 > ra11 + rb11: return False
  // if d12 > ra12 + rb12: return False
  Vec4r ra1 = ((Broadcast<0>(ea) * Rabs[2]) + (Broadcast<2>(ea) * Rabs[0]));
  Vec4r rb1 = ((eb110 * Shuffle<2, 2, 1>(Rabs[1])) + (eb221 * Shuffle<1, 0, 0>(Rabs[1])));
  Vec4r d1 = Abs((Broadcast<0>(t) * R[2]) - (Broadcast<2>(t) * R[0]));
  if (AnyTrue<3>(d1 > ra1 + rb1)) {
    return false;
  }

  // ra20 = ea[0] * Rabs[1, 0] + ea[1] * Rabs[0, 0]
  // ra21 = ea[0] * Rabs[1, 1] + ea[1] * Rabs[0, 1]
  // ra22 = ea[0] * Rabs[1, 2] + ea[1] * Rabs[0, 2]
  // rb20 = eb[1] * Rabs[2, 2] + eb[2] * Rabs[2, 1]
  // rb21 = eb[0] * Rabs[2, 2] + eb[2] * Rabs[2, 0]
  // rb22 = eb[0] * Rabs[2, 1] + eb[1] * Rabs[2, 0]
  // d20 = np.abs(t[1] * R[0, 0] - t[0] * R[1, 0])
  // d21 = np.abs(t[1] * R[0, 1] - t[0] * R[1, 1])
  // d22 = np.abs(t[1] * R[0, 2] - t[0] * R[1, 2])
  // if d20 > ra20 + rb20: return False
  // if d21 > ra21 + rb21: return False
  // if d22 > ra22 + rb22: return False
  Vec4r ra2 = ((Broadcast<0>(ea) * Rabs[1]) + (Broadcast<1>(ea) * Rabs[0]));
  Vec4r rb2 = ((eb110 * Shuffle<2, 2, 1>(Rabs[2])) + (eb221 * Shuffle<1, 0, 0>(Rabs[2])));
  Vec4r d2 = Abs((Broadcast<1>(t) * R[0]) - (Broadcast<0>(t) * R[1]));
  if (AnyTrue<3>(d2 > ra2 + rb2)) {
    return false;
  }

  // They must be overlapping.
  return true;
}

[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(Sphere const& a, Sphere const& b) {
  return NormSqr(a.GetCenter() - b.GetCenter()) <= Sqr(a.GetRadius() + b.GetRadius());
}

/**
 * @brief Test overlap between a sphere and each of a batch of spheres.
 *
 * @param a The first sphere.
 * @param b The batch of spheres.
 * @return A per-lane SIMD mask; each sphere's lane is set if overlap was detected.
 */
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchReal<kBatchSize> HasOverlap(
    Sphere const& a,
    BatchSphere<kBatchSize> const& b) {
  auto const aCenter = BroadcastEach<BatchReal<kBatchSize>>(a.GetCenter());
  return NormSqr(aCenter - b.center) <= Sqr(a.GetRadius() + b.radius);
}

[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(Plane const& a, Plane const& b) {
  return (a.GetNormal() != -b.GetNormal()) ||
      (a.GetDistanceFromOrigin() >= -b.GetDistanceFromOrigin());
}

[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(Obb const& oobb, Aabb const& aabb) {
  return HasOverlap(
      oobb, Obb{MatrixTransformRT{VEye<3>(), aabb.VGetCenter()}, aabb.GetHalfExtents()});
}

[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(Aabb const& aabb, Obb const& oobb) {
  return HasOverlap(oobb, aabb);
}

[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(Sphere const& s, Aabb const& aabb) {
  Vec4r center = s.VGetPacked();

  // Closest point on the AABB to sphere center.
  Vec4r closestPoint = Clamp(center, aabb.VGetMin(), aabb.VGetMax());

  // Check if distance from sphere center to closest point is less than radius.
  real distSqr = NormSqr<3>(closestPoint - center);
  return distSqr <= Sqr(s.GetRadius());
}

[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(Aabb const& aabb, Sphere const& s) {
  return HasOverlap(s, aabb);
}

/**
 * @brief Test overlap between an @ref Aabb and each of a batch of spheres.
 *
 * @param aabb The axis-aligned bounding box.
 * @param sphere The batch of spheres.
 * @return A per-lane SIMD mask; each sphere's lane is set if overlap was detected.
 */
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchReal<kBatchSize> HasOverlap(
    Aabb const& aabb,
    BatchSphere<kBatchSize> const& sphere) {
  using V = BatchReal<kBatchSize>;

  // Closest point on the AABB to sphere center.
  auto const aabbMin = BroadcastEach<V>(aabb.GetMin());
  auto const aabbMax = BroadcastEach<V>(aabb.GetMax());
  auto const closestPoint = Clamp(sphere.center, aabbMin, aabbMax);

  // Check if distance from sphere center to closest point is less than radius.
  return NormSqr(closestPoint - sphere.center) <= Sqr(sphere.radius);
}

// For this purpose, everything under the plane counts as overlap
[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(Plane const& p, Aabb const& aabb) {
  Vec4r normalAndDist = p.VGetPacked(); // (normalX, normalY, normalZ, dist)
  Vec4r closestVertex = Select(normalAndDist < Vec4r{0_r}, aabb.VGetMax(), aabb.VGetMin());
  return Dot(normalAndDist, Set<3>(closestVertex, -1_r)) <= 0_r;
}

[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(Aabb const& aabb, Plane const& p) {
  return HasOverlap(p, aabb);
}

[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(Plane const& p, Obb const& oobb) {
  auto corners = oobb.VGetCorners();
  auto normalAndD = p.VGetPacked();
  auto distance = Broadcast<3>(normalAndD);
  for (int i = 0; i < 8; ++i) {
    Vec4r cornerDist = VDot<3>(corners[i], normalAndD);
    if (AllTrue<3>(cornerDist <= distance)) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(Obb const& oobb, Plane const& p) {
  return HasOverlap(p, oobb);
}

[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(Plane const& p, Sphere const& s) {
  Vec4r normalAndDist = p.VGetPacked();
  Vec4r centerAndRadius = s.VGetPacked();
  return Dot(normalAndDist, Set<3>(centerAndRadius, -1_r)) <= Get<3>(centerAndRadius);
}

/**
 * @brief Test overlap between an @ref Plane and each of a batch of spheres.
 *
 * @note All volume behind the plane counts as overlap.
 *
 * @param plane The plane.
 * @param sphere The batch of spheres.
 * @return A per-lane SIMD mask; each sphere's lane is set if overlap was detected.
 */
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchReal<kBatchSize> HasOverlap(
    Plane const& plane,
    BatchSphere<kBatchSize> const& sphere) {
  using V = BatchReal<kBatchSize>;
  auto const planeNormal = BroadcastEach<V>(plane.GetNormal());
  auto const planeDist = plane.GetDistanceFromOrigin();
  return Dot(planeNormal, sphere.center) - planeDist <= sphere.radius;
}

[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(Sphere const& s, Plane const& p) {
  return HasOverlap(p, s);
}

[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(Sphere const& s, Obb const& oobb) {
  Vec4r aCenter = s.VGetPacked();
  Vec4r bHalfExt = oobb.VGetHalfExtents();
  Vec4r aCenterInB = oobb.GetTransform().InverseTransformPoint(aCenter);
  Vec4r aCenterInBClamped = Clamp(aCenterInB, -bHalfExt, bHalfExt);
  real diffSqr = NormSqr<3>(aCenterInBClamped - aCenterInB);
  real aRadiusSqr = Sqr(s.GetRadius());
  return diffSqr <= aRadiusSqr;
}

[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(Obb const& oobb, Sphere const& s) {
  return HasOverlap(s, oobb);
}

/**
 * @brief Test overlap between an @ref Obb and each of a batch of spheres.
 *
 * @param obb The oriented bounding box.
 * @param sphere The batch of spheres.
 * @return A per-lane SIMD mask; each sphere's lane is set if overlap was detected.
 */
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchReal<kBatchSize> HasOverlap(
    Obb const& obb,
    BatchSphere<kBatchSize> const& sphere) {
  using V = BatchReal<kBatchSize>;
  auto const obbRot = Broadcast3x3<V>(obb.VGetRotation());
  auto const obbCenter = Broadcast3<V>(obb.VGetCenter());
  auto const obbHalfExt = Broadcast3<V>(obb.VGetHalfExtents());
  // Transform the sphere centers into the OBB's local frame: R^T * (center - obbCenter).
  auto const sphereCenterInObb = DotVecMat(sphere.center - obbCenter, obbRot);
  auto const sphereCenterInObbClamped = Clamp(sphereCenterInObb, -obbHalfExt, obbHalfExt);
  return NormSqr(sphereCenterInObbClamped - sphereCenterInObb) <= Sqr(sphere.radius);
}

template <typename ShapeT, MOCHI_CONCEPT(IsPrimitiveShape<ShapeT>)>
[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(AnyShape const& lhs, ShapeT const& rhs) {
  // Call HasOverlap(x, rhs) where x is the type stored in the variant.
  return std::visit([&rhs](auto& x) { return HasOverlap(x, rhs); }, lhs);
}

template <typename ShapeT, MOCHI_CONCEPT(IsPrimitiveShape<ShapeT>)>
[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(ShapeT const& lhs, AnyShape const& rhs) {
  return HasOverlap(rhs, lhs);
}

// WARNING: This overload must be declared AFTER the template overloads which involve
//          AnyShape. Otherwise, it will end up calling itself recursively on some compilers.
[[nodiscard]] MOCHI_FORCE_INLINE bool HasOverlap(AnyShape const& lhs, AnyShape const& rhs) {
  // Call HasOverlap(x, rhs) where x is the type stored in the lhs variant.
  return std::visit([&rhs](auto& x) { return HasOverlap(x, rhs); }, lhs);
}

/**
 * Checks for overlap between a single shape and multiple shapes in batch.
 *
 * @tparam kMaxBatchSize Maximum number of shapes to process in a batch.
 * @tparam ShapeL Type of the left-hand shape.
 * @tparam ShapeR Type of the right-hand shapes.
 * @param batchSize Number of shapes to process in this batch (must not exceed kMaxBatchSize).
 * @param lhs The single shape to check against all right-hand shapes.
 * @param rhs Span of shapes to check for overlap with the left-hand shape (must be at least
 *            batchSize in length).
 * @param outHasOverlap Output span to store overlap results (must be at least batchSize in length).
 *
 * @note Currently implemented as sequential calls to HasOverlap. It could be optimized in the
 *       future using vertical SIMD operations for better performance.
 */
template <
    int kMaxBatchSize,
    typename ShapeL,
    typename ShapeR,
    MOCHI_CONCEPT(IsPrimitiveShape<ShapeL>&& IsPrimitiveShape<ShapeR>)>
MOCHI_FORCE_INLINE void HasOverlapBatch(
    int batchSize,
    ShapeL const& lhs,
    Span<ShapeR const> rhs,
    Span<bool> outHasOverlap) {
  MOCHI_ASSERT_VERBOSE(
      (batchSize >= 0) && (batchSize <= Min(kMaxBatchSize, isize(rhs), isize(outHasOverlap))),
      "Invalid batch size.");
  for (int i = 0; i < batchSize; ++i) {
    outHasOverlap[i] = HasOverlap(lhs, rhs[i]);
  }
}

/**************************************************************************************************
  Tetrahedral Geometry Utils
*/

// Gets the barycentric coordinates of a query with respect to the physical node coordinates
// See https://en.wikipedia.org/wiki/Barycentric_coordinate_system
template <typename T>
[[nodiscard]] inline constexpr NdArray<T, 4> BarycentricCoords4(
    T const v1[3],
    T const v2[3],
    T const v3[3],
    T const v4[3],
    NdArray<T, 3> const& query) {
  NdArray<T, 4, 4> barycenter_matrix{
      NdArray<T, 4>{v1[0], v2[0], v3[0], v4[0]},
      NdArray<T, 4>{v1[1], v2[1], v3[1], v4[1]},
      NdArray<T, 4>{v1[2], v2[2], v3[2], v4[2]},
      NdArray<T, 4>{(T)1.0, (T)1.0, (T)1.0, (T)1.0}};

  auto inv = Invert(barycenter_matrix);
  NdArray<T, 4> query4{query[0], query[1], query[2], (T)1.0};

  return DotMatVec(inv, query4);
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 4> BarycentricCoords4(
    NdArray<T, 3> const& v1,
    NdArray<T, 3> const& v2,
    NdArray<T, 3> const& v3,
    NdArray<T, 3> const& v4,
    NdArray<T, 3> const& query) {
  return BarycentricCoords4(v1.data(), v2.data(), v3.data(), v4.data(), query);
}

// Queries whether or not a point is inside this element
template <typename T>
[[nodiscard]] inline constexpr bool IsInsideTetrahedron(
    T const v1[3],
    T const v2[3],
    T const v3[3],
    T const v4[3],
    NdArray<T, 3> const& query) {
  NdArray<T, 4> coords = BarycentricCoords4(v1, v2, v3, v4, query);

  return coords[0] >= 0 && coords[1] >= 0 && coords[2] >= 0 && coords[3] >= 0;
}

// Queries whether or not a point is inside this element
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr bool IsInsideTetrahedron(
    NdArray<T, 3> const& v1,
    NdArray<T, 3> const& v2,
    NdArray<T, 3> const& v3,
    NdArray<T, 3> const& v4,
    NdArray<T, 3> const& query) {
  return IsInsideTetrahedron(v1.data(), v2.data(), v3.data(), v4.data(), query);
}

/**************************************************************************************************
  Distance functions to geometric primitives
*/

// Computes the signed squared distance between a point and an Aabb. If outPar != nullptr
// it also outputs the parametric coordinate of the closest point.
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r
VDistancePointShapeSqr(Vec4r p, Aabb const& aabb, Vec4r* outPar = nullptr);

// Computes the signed squared distance between a point and an Obb. If outPar != nullptr
// it also outputs the parametric coordinate of the closest point.
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r
VDistancePointShapeSqr(Vec4r p, Obb const& oobb, Vec4r* outPar = nullptr);

// Computes the signed distance between a point and a Sphere.
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r VDistancePointShapeSqr(Vec4r p, Sphere const& sphere);
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r VDistancePointShape(Vec4r p, Sphere const& sphere);

// Computes the squared distance between a point and a segment. If outPar != nullptr
// it also outputs the parametric coordinate along the segment of the closest point.
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r
VDistancePointSegmentSqr(Vec4r p, Vec4r a, Vec4r b, Vec4r* outPar = nullptr);

/*
  Structure containing parameters required for signed distance computation.
*/
struct VDistanceSignParams {
  Vec4r normalA = {};
  Vec4r normalB = {};
  Vec4r normalC = {};
  Vec4r normalAB = {};
  Vec4r normalBC = {};
  Vec4r normalCA = {};
  bool computeSign = false;
  real outSign = 1_r;
};

// Computes the squared distance between a point and a triangle. The DistanceSignParams
// structure indicates wether or not the squared distance should be signed, in which
// case approximated vertex and edge normals should be provided. If outPar != nullptr
// it also outputs the barycentric coordinates of the closest point.
[[nodiscard]] Vec4r VDistancePointTriangleSqr(
    Vec4r p,
    Vec4r a,
    Vec4r b,
    Vec4r c,
    VDistanceSignParams& sign,
    Vec4r* outPar = nullptr);

// Computes the squared distance between a point and a tetrahedron.
[[nodiscard]] Vec4r VDistancePointTetrahedronSqr(
    Vec4r const& p,
    Vec4r const& a,
    Vec4r const& b,
    Vec4r const& c,
    Vec4r const& d);

// Computes the distance between a point and a triangle. The DistanceSignParams
// structure indicates wether or not the squared distance should be signed, in which
// case approximated vertex and edge normals should be provided. If outPar != nullptr
// it also outputs the barycentric coordinates of the closest point.
[[nodiscard]] Vec4r VDistancePointTriangle(
    Vec4r p,
    Vec4r a,
    Vec4r b,
    Vec4r c,
    VDistanceSignParams& sign,
    Vec4r* outPar = nullptr);

/*************************************************************************************************
  Closest point computation for geometric primitives
*/

/**
 * Computes the closest point on the Sphere.
 */
[[nodiscard]] MOCHI_FORCE_INLINE Real3 ClosestPtPointShape(Real3 const& p, Plane const& plane);
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r VClosestPtPointShape(Vec4r p, Plane const& plane);

/**
 * Computes the closest point on the Aabb.
 */
[[nodiscard]] MOCHI_FORCE_INLINE Real3
ClosestPtPointShape(Real3 const& p, Aabb const& aabb, Real3* outPar = nullptr);
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r
VClosestPtPointShape(Vec4r p, Aabb const& aabb, Vec4r* outPar = nullptr);

/**
 * Computes the closest point on the Obb.
 */
[[nodiscard]] MOCHI_FORCE_INLINE Real3
ClosestPtPointShape(Real3 const& p, Obb const& oobb, Real3* outPar = nullptr);
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r
VClosestPtPointShape(Vec4r p, Obb const& oobb, Vec4r* outPar = nullptr);

/**
 * Computes the closest point on the Sphere.
 */
[[nodiscard]] MOCHI_FORCE_INLINE Real3 ClosestPtPointShape(Real3 const& p, Sphere const& sphere);
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r VClosestPtPointShape(Vec4r p, Sphere const& sphere);

/**
 * Find the index of the closest triangle (face) in a triangular mesh.
 */
[[nodiscard]] int
ClosestFacePointTriangularMesh(Real3 const& p, Span<Real3 const> nodes, Span<Int3 const> faces);

} // namespace mochi

#include "geometry_utils_inl.h"
