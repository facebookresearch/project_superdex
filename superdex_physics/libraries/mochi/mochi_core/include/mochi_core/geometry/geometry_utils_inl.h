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
#include "geometry_utils.h" // reverse include for Intellisense

#include <cmath>
#include <limits>

namespace mochi {

/**************************************************************************************************
  Distance functions to geometric primitives
*/

MOCHI_FORCE_INLINE Vec4r VDistancePointShapeSqr(Vec4r p, Aabb const& aabb, Vec4r* outPar) {
  Vec4r c = aabb.VGetCenter();
  Vec4r e = aabb.VGetHalfExtents();
  Vec4r cp = p - c;

  if (outPar) {
    *outPar = (Clamp(cp, -e, e) / e) * 0.5_r + 0.5_r;
  }

  Vec4r q = Abs(cp) - e;
  Vec4r d = VNormSqr<3>(Max(q, SimdZero()));
  d -= Sqr(Min(HMax<3>(q), 0_r));
  return d;
}

MOCHI_FORCE_INLINE Vec4r VDistancePointShapeSqr(Vec4r p, Obb const& obb, Vec4r* outPar) {
  Vec4r x = obb.GetTransform().InverseTransformPoint(p);
  Vec4r e = obb.VGetHalfExtents();

  if (outPar) {
    *outPar = (Clamp(x, -e, e) / e) * 0.5_r + 0.5_r;
  }

  Vec4r q = Abs(x) - e;
  Vec4r d = VNormSqr<3>(Max(q, SimdZero()));
  d -= Sqr(Min(HMax<3>(q), 0_r));
  return d;
}

MOCHI_FORCE_INLINE Vec4r VDistancePointShapeSqr(Vec4r p, Sphere const& sphere) {
  Vec4r d = VDistancePointShape(p, sphere);
  return d * Abs(d);
}

MOCHI_FORCE_INLINE Vec4r VDistancePointShape(Vec4r p, Sphere const& sphere) {
  return VNorm<3>(p - sphere.VGetPacked()) - sphere.GetRadius();
}

MOCHI_FORCE_INLINE Vec4r VDistancePointSegmentSqr(Vec4r p, Vec4r a, Vec4r b, Vec4r* outPar) {
  Vec4r ab = b - a;

  Vec4r abMod2 = VNormSqr<3>(ab);
  Vec4r ones{1_r};
  Vec4r zero{0_r};

  // Find the parameter of the closest point in the segment
  Vec4r safeAbMod2 = Max(abMod2, Vec4r{std::numeric_limits<real>::min()});
  Vec4r wA = Clamp(VDot<3>(b - p, ab) / safeAbMod2, zero, ones);
  Vec4r wB = ones - wA;
  if (outPar != nullptr) {
    *outPar = wB;
  }

  return VNormSqr<3>(p - wA * a - wB * b);
}

inline Vec4r VDistancePointTriangleSqr(
    Vec4r p,
    Vec4r a,
    Vec4r b,
    Vec4r c,
    VDistanceSignParams& sign,
    Vec4r* outPar) {
  // Triangle sides
  Vec4r ab = b - a;
  Vec4r bc = c - b;
  Vec4r ca = a - c;

  // To point vectors
  Vec4r ap = p - a;
  Vec4r bp = p - b;
  Vec4r cp = p - c;

  // Plane normal
  Vec4r n = Cross3(ab, bc);

  // Side tests
  Vec4r abxn = Cross3(ab, n);
  Vec4r bcxn = Cross3(bc, n);
  Vec4r caxn = Cross3(ca, n);
  Vec4r zero = {};
  bool abSideOut = AllTrue<3>(VDot<3>(ap, abxn) > zero);
  bool bcSideOut = AllTrue<3>(VDot<3>(bp, bcxn) > zero);
  bool caSideOut = AllTrue<3>(VDot<3>(cp, caxn) > zero);

  // Outside of triangle. Test all edges for which the point is outside.
  if (abSideOut || bcSideOut || caSideOut) {
    real minDistSqr = std::numeric_limits<real>::infinity();
    Vec4r wA = {}, wB = {}, wC = {}; // Weights

    auto TestSide = [&p, &minDistSqr, &sign, &outPar](
                        Vec4r const& a,
                        Vec4r const& b,
                        Vec4r const& na,
                        Vec4r const& nb,
                        Vec4r const& nab,
                        Vec4r const& ap,
                        Vec4r const& bp,
                        Vec4r& wa,
                        Vec4r& wb,
                        Vec4r& wc) {
      auto Sign = [](real val) { return val >= 0 ? 1_r : -1_r; };

      Vec4r thiswb;
      real thisDistSqr = Get0(VDistancePointSegmentSqr(p, a, b, &thiswb));
      if (thisDistSqr < minDistSqr) {
        minDistSqr = thisDistSqr;
        if (sign.computeSign) {
          if (thiswb == Vec4r{1_r}) {
            sign.outSign = Sign(Dot<3>(bp, nb));
          } else if (thiswb == Vec4r{0_r}) {
            sign.outSign = Sign(Dot<3>(ap, na));
          } else {
            sign.outSign = Sign(Dot<3>(ap, nab));
          }
        }
        if (outPar != nullptr) {
          wb = thiswb;
          wa = 1_r - wb;
          wc = SimdZero();
        }
      }
    };

    // Distance to AB
    if (abSideOut) {
      TestSide(a, b, sign.normalA, sign.normalB, sign.normalAB, ap, bp, wA, wB, wC);
    }

    // Distance to BC
    if (bcSideOut) {
      TestSide(b, c, sign.normalB, sign.normalC, sign.normalBC, bp, cp, wB, wC, wA);
    }

    // Distance to CA
    if (caSideOut) {
      TestSide(c, a, sign.normalC, sign.normalA, sign.normalCA, cp, ap, wC, wA, wB);
    }

    if (outPar != nullptr) {
      *outPar = Blend<0, 1, 1, 1>(wA, Blend<0, 0, 1, 1>(wB, wC)); // (wA, wB, wC, wC)
    }
    return Vec4r{minDistSqr};
  }

  // Inside the triangle
  Vec4r abcArea = VNorm<3>(n);
  Vec4r nunit = n / abcArea;
  Vec4r dist = VDot<3>(ap, nunit);
  Vec4r dist2 = dist * dist;

  if (outPar != nullptr) {
    Vec4r q = p - nunit * dist;
    Vec4r bcqArea = VNorm<3>(Cross3(bc, q - c));
    Vec4r aqcArea = VNorm<3>(Cross3(q - a, ca));
    Vec4r parA = bcqArea / abcArea;
    Vec4r parB = aqcArea / abcArea;
    Vec4r parC = 1_r - (parA + parB);
    *outPar = Blend<0, 1, 1, 1>(parA, Blend<0, 0, 1, 1>(parB, parC));
  }

  if (sign.computeSign) {
    sign.outSign = Sign(Get0(dist));
  }

  return dist2;
}

inline Vec4r VDistancePointTriangle(
    Vec4r p,
    Vec4r a,
    Vec4r b,
    Vec4r c,
    VDistanceSignParams& sign,
    Vec4r* outPar) {
  Vec4r dist2 = VDistancePointTriangleSqr(p, a, b, c, sign, outPar);
  return Sqrt(dist2);
}

inline Vec4r VDistancePointTetrahedronSqr(
    Vec4r const& p,
    Vec4r const& a,
    Vec4r const& b,
    Vec4r const& c,
    Vec4r const& d) {
  Real4 coords = BarycentricCoords4(ToReal3(a), ToReal3(b), ToReal3(c), ToReal3(d), ToReal3(p));
  if (AllTrue(ToSimd(coords) >= Vec4r{0_r})) {
    return 0_r;
  }
  VDistanceSignParams unused;
  real minDistSqr = std::numeric_limits<real>::infinity();
  if (coords[0] < 0.0_r) { // Outside the plane of BDC
    minDistSqr = Get0(VDistancePointTriangleSqr(p, b, d, c, unused));
  }
  if (coords[1] < 0.0_r) { // Outside the plane of ACD
    real otherDistSqr = Get0(VDistancePointTriangleSqr(p, a, c, d, unused));
    if (otherDistSqr < minDistSqr) {
      minDistSqr = otherDistSqr;
    }
  }
  if (coords[2] < 0.0_r) { // Outside the plane of ADB
    real otherDistSqr = Get0(VDistancePointTriangleSqr(p, a, d, b, unused));
    if (otherDistSqr < minDistSqr) {
      minDistSqr = otherDistSqr;
    }
  }
  if (coords[3] < 0.0_r) { // Outside the plane of ABC
    real otherDistSqr = Get0(VDistancePointTriangleSqr(p, a, b, c, unused));
    if (otherDistSqr < minDistSqr) {
      minDistSqr = otherDistSqr;
    }
  }
  return Vec4r{minDistSqr};
}

/*************************************************************************************************
  Closest point computation for geometric primitives
*/

MOCHI_FORCE_INLINE Vec4r VClosestPtPointShape(Vec4r p, Plane const& plane) {
  Vec4r d = VDot<3>(p, plane.VGetNormal()) - plane.GetDistanceFromOrigin();
  return p - d * plane.VGetNormal();
}

MOCHI_FORCE_INLINE Real3 ClosestPtPointShape(Real3 const& p, Plane const& plane) {
  return ToReal3(VClosestPtPointShape(ToSimd(p), plane));
}

MOCHI_FORCE_INLINE Vec4r VClosestPtPointShape(Vec4r p, Aabb const& aabb, Vec4r* outPar) {
  Vec4r q = Clamp(p, aabb.VGetMin(), aabb.VGetMax());

  if (outPar) {
    *outPar = (q - aabb.VGetMin()) / aabb.VGetSize();
  }

  return q;
}

// Helper to call the SIMD overload for various shapes
namespace simdify {
template <typename ShapeT>
MOCHI_FORCE_INLINE Real3 ClosestPtPointShape(Real3 const& p, ShapeT const& shape, Real3* outPar) {
  if (outPar) {
    Vec4r par;
    Real3 pt = ToReal3(VClosestPtPointShape(ToSimd(p), shape, &par));
    *outPar = ToReal3(par);
    return pt;
  } else {
    return ToReal3(VClosestPtPointShape(ToSimd(p), shape, nullptr));
  }
}
} // namespace simdify

MOCHI_FORCE_INLINE Real3 ClosestPtPointShape(Real3 const& p, Aabb const& aabb, Real3* outPar) {
  return simdify::ClosestPtPointShape(p, aabb, outPar);
}

MOCHI_FORCE_INLINE Vec4r VClosestPtPointShape(Vec4r p, Obb const& obb, Vec4r* outPar) {
  auto const& T = obb.GetTransform();
  Vec4r q = Clamp(T.InverseTransformPoint(p), -obb.VGetHalfExtents(), obb.VGetHalfExtents());

  if (outPar) {
    *outPar = (q / obb.VGetHalfExtents()) * 0.5_r + 0.5_r;
  }

  return T.TransformPoint(q);
}

MOCHI_FORCE_INLINE Real3 ClosestPtPointShape(Real3 const& p, Obb const& obb, Real3* outPar) {
  return simdify::ClosestPtPointShape(p, obb, outPar);
}

MOCHI_FORCE_INLINE Vec4r VClosestPtPointShape(Vec4r p, Sphere const& sphere) {
  // Retrieve packed sphere, check distance to center.
  // If closer than [r], the closest point is [p]
  Vec4r center = sphere.VGetPacked();
  real radius = sphere.GetRadius();
  Vec4r q = p - center;
  real distSqr = NormSqr<3>(q);

  if (distSqr < Sqr(radius)) {
    return p;
  } else { // If not, project to the sphere's surface.
    return (radius * Normalize(q, distSqr)) + center;
  }
}

MOCHI_FORCE_INLINE Real3 ClosestPtPointShape(Real3 const& p, Sphere const& sphere) {
  return ToReal3(VClosestPtPointShape(ToSimd(p), sphere));
}

} // namespace mochi
