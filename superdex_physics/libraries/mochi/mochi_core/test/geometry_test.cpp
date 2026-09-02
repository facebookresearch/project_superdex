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
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/vmatrix.h>

#include <mochi_core/geometry/grid_sdf.h>
#include <mochi_core/geometry/sdf_bv.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/utils/rand_utils.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <variant>

#include "data/obb_test_data.h"

using namespace mochi;

/**********************************************************************************************
  Helpers
*/

static void ExpectAabb(Real3 const& expectedMin, Real3 const& expectedMax, Aabb const& actual) {
  // Scalar access
  EXPECT_NEAR_EQ(expectedMin, actual.GetMin());
  EXPECT_NEAR_EQ(expectedMax, actual.GetMax());

  // SIMD access (don't assume the value of the 4th component)
  EXPECT_NEAR_EQ(ToSimd(expectedMin, 1_r), actual.VGetMin());
  EXPECT_NEAR_EQ(ToSimd(expectedMax, 1_r), actual.VGetMax());

  // NearEqual function
  EXPECT_NEAR_EQ(Aabb(expectedMin, expectedMax), actual);
}

static void ExpectCapsule(
    Real3 const& a,
    Real3 const& b,
    real radius,
    Capsule const& actual,
    real tolerance = 1e-6_r) {
  // Scalar access
  EXPECT_NEAR_TOL(a, actual.GetA(), tolerance);
  EXPECT_NEAR_TOL(b, actual.GetB(), tolerance);
  EXPECT_NEAR_TOL(b - a, actual.GetAB(), tolerance);
  EXPECT_NEAR_TOL(radius, actual.GetRadius(), tolerance);
  EXPECT_NEAR_TOL(Norm(b - a), actual.GetLengthAB(), tolerance);

  // SIMD access (don't assume the value of the 4th component)
  EXPECT_NEAR_TOL(ToSimd(a, 1_r), actual.VGetA(), tolerance);
  EXPECT_NEAR_TOL(ToSimd(b, 1_r), actual.VGetB(), tolerance);
  EXPECT_NEAR_TOL(ToSimd(b - a, 0_r), actual.VGetAB(), tolerance);
  EXPECT_NEAR_TOL(Vec4r(Norm(b - a)), actual.VGetLengthAB(), tolerance);
  EXPECT_NEAR_TOL(ToSimd(a, radius), actual.VGetPackedARadius(), tolerance);

  // NearEqual function
  EXPECT_NEAR_TOL((Capsule::FromPoints(a, b, radius)), actual, tolerance);
}

static void ExpectObb(
    Real3 const& translation,
    Matrix3x3r const& rotation,
    Real3 const& extents,
    Obb const& actual) {
  // The test data is only accurate to single precision. The tolerance must therefore be
  // proportional to single precision's epsilon also in the double precision build.
  real constexpr kEpsilon = 1e3_r * static_cast<real>(std::numeric_limits<float>::epsilon());

  // NOTE: The rotation matrix computed by CalcObb may differ from the ground-truth data in the
  // sense that they might be rotated 180 degrees from one another. This is a consequence of
  // differences in the algorithms for the spectral decomposition. Instead of ensuring that both
  // output the same result, we are a bit more lax about this and only require that they point
  // towards the same directions (even if in opposite senses). This is ok because in the end, both
  // Obbs are equivalent.
  auto const VIsNearEqRotation = [](VMatrix3x3r const& a, VMatrix3x3r const& b, float eps) {
    Vec4r const epsilon = eps;

    auto I = Dot3x3(Transpose3x3(a), b);
    auto x = SimdBasisVector<0>();
    auto y = SimdBasisVector<1>();
    auto z = SimdBasisVector<2>();
    Vec4r eq = VNearEqual(Abs(I[0]), x, epsilon);
    eq &= VNearEqual(Abs(I[1]), y, epsilon);
    eq &= VNearEqual(Abs(I[2]), z, epsilon);
    EXPECT_TRUE(AllTrue<3>(eq));
  };

  auto const IsNearEqRotation = [&](Matrix3x3r const& a, Matrix3x3r const& b, float eps) {
    VIsNearEqRotation(ToSimdMatrix(a), ToSimdMatrix(b), eps);
  };

  // Scalar access
  EXPECT_NEAR_TOL(translation, actual.GetTransform().GetTranslation(), kEpsilon);
  EXPECT_NEAR_TOL(extents, actual.GetHalfExtents(), kEpsilon);
  IsNearEqRotation(rotation, actual.GetTransform().GetRotation(), kEpsilon);

  // SIMD access.
  EXPECT_NEAR_TOL(ToSimd(translation, 1_r), actual.GetTransform().VGetTranslation(), kEpsilon);
  EXPECT_NEAR_TOL(ToSimd(extents, 0_r), actual.VGetHalfExtents(), kEpsilon);
  VIsNearEqRotation(ToSimdMatrix(rotation), actual.GetTransform().VGetRotation(), kEpsilon);
}

static void ExpectSphere(Real3 const& expectedCenter, real expectedRadius, Sphere const& actual) {
  EXPECT_NEAR_EQ(expectedCenter, actual.GetCenter());
  EXPECT_NEAR_EQ(ToSimd(expectedCenter, 1_r), actual.VGetCenter());
  EXPECT_NEAR_EQ(expectedRadius, actual.GetRadius());
  EXPECT_NEAR_EQ(Sphere(expectedCenter, expectedRadius), actual);
}

static void ExpectPlane(Real3 const& expectedNorm, real expectedDist, Plane const& actual) {
  EXPECT_NEAR_EQ(expectedNorm, actual.GetNormal());
  EXPECT_NEAR_EQ(ToSimd(expectedNorm, 0_r), actual.VGetNormal());
  EXPECT_NEAR_EQ(expectedDist, actual.GetDistanceFromOrigin());
  EXPECT_NEAR_EQ(Plane(expectedNorm, expectedDist), actual);
}

/**********************************************************************************************
  Aabb
*/

TEST(Aabb, Class) {
  // Default
  {
    Aabb a;
    ExpectAabb(Real3{}, Real3{}, a);
  }

  // From Real3
  {
    Aabb a(Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r});
    ExpectAabb(Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, a);
  }

  // From Vec4r
  {
    Aabb a(Vec4r(1_r, 2_r, 3_r), Vec4r(4_r, 5_r, 6_r));
    ExpectAabb(Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, a);
  }

  // GetCenter
  {
    Aabb a(Real3(1_r, 2_r, 3_r), Real3(4_r, 5_r, 6_r));
    EXPECT_NEAR_EQ(Real3(2.5_r, 3.5_r, 4.5_r), a.GetCenter());
    EXPECT_NEAR_EQ(Vec4r(2.5_r, 3.5_r, 4.5_r, 1_r), a.VGetCenter());
  }
}

TEST(Aabb, Equal) {
  auto a = Aabb{Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}};
  EXPECT_TRUE(a == a);
  EXPECT_TRUE(Aabb{} == Aabb{});
  EXPECT_FALSE(a != a);
  EXPECT_FALSE(Aabb{} != Aabb{});
  constexpr real kNudge = kDefaultNearEqualEpsilon<real>; // Some small value
  for (int i = 0; i < 6; ++i) {
    // Any change makes them not equal
    auto min = a.GetMin();
    auto max = a.GetMax();
    if (i < 3) {
      min[i] += kNudge;
    } else {
      max[i - 3] += kNudge;
    }
    auto b = Aabb{min, max};
    EXPECT_FALSE(a == b);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(b != a);
  }
}

TEST(Aabb, NearEqual) {
  Aabb a{Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}};
  Aabb b{Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}};
  Aabb c{Real3{1.1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}};
  Aabb d{Real3{1.0_r, 2_r, 3_r}, Real3{4_r, 5_r, 6.1_r}};
  EXPECT_EQ(true, NearEqual(a, a, 0.2_r));
  EXPECT_EQ(true, NearEqual(a, b, 0.2_r));
  EXPECT_EQ(true, NearEqual(a, c, 0.2_r));
  EXPECT_EQ(true, NearEqual(a, d, 0.2_r));
  EXPECT_EQ(true, NearEqual(a, a, 0.02_r));
  EXPECT_EQ(true, NearEqual(a, b, 0.02_r));
  EXPECT_EQ(false, NearEqual(a, c, 0.02_r));
  EXPECT_EQ(false, NearEqual(a, d, 0.02_r));
}

TEST(Aabb, CalcAabb) {
  {
    Real3 const coords[] = {
        Real3{1.0_r, 2.0_r, 3.0_r},
        Real3{1.1_r, 1.9_r, 3.1_r},
        Real3{0.9_r, 2.1_r, 2.9_r},
        Real3{1.2_r, 1.8_r, 3.2_r},
    };

    // Empty span returns all zeros
    Aabb bounds = CalcAabb(Span(coords, 0_uz));
    EXPECT_NEAR_EQ(Real3(0.0_r, 0.0_r, 0.0_r), bounds.GetMin());
    EXPECT_NEAR_EQ(Real3(0.0_r, 0.0_r, 0.0_r), bounds.GetMax());

    // Length 1
    bounds = CalcAabb(Span(coords, 1_uz));
    EXPECT_NEAR_EQ(Real3(1.0_r, 2.0_r, 3.0_r), bounds.GetMin());
    EXPECT_NEAR_EQ(Real3(1.0_r, 2.0_r, 3.0_r), bounds.GetMax());

    // Length 2
    bounds = CalcAabb(Span(coords, 2_uz));
    EXPECT_NEAR_EQ(Real3(1.0_r, 1.9_r, 3.0_r), bounds.GetMin());
    EXPECT_NEAR_EQ(Real3(1.1_r, 2.0_r, 3.1_r), bounds.GetMax());

    // Length 3
    bounds = CalcAabb(Span(coords, 3_uz));
    EXPECT_NEAR_EQ(Real3(0.9_r, 1.9_r, 2.9_r), bounds.GetMin());
    EXPECT_NEAR_EQ(Real3(1.1_r, 2.1_r, 3.1_r), bounds.GetMax());

    // Length 4
    bounds = CalcAabb(Span(coords, 4_uz));
    EXPECT_NEAR_EQ(Real3(0.9_r, 1.8_r, 2.9_r), bounds.GetMin());
    EXPECT_NEAR_EQ(Real3(1.2_r, 2.1_r, 3.2_r), bounds.GetMax());
  }

  // Now test a larger number of points and vary the location of the min and max values.
  {
    std::vector<Real3> coords(50);
    auto coordsSpan = MakeConstSpan(coords);
    for (int sz = 2; sz < isize(coords); ++sz) {
      for (int i = 0; i < sz; ++i) {
        for (int j = 0; j < 3; ++j) {
          // Place the minimum value here
          coords[i][j] = -1;
          Aabb result = CalcAabb(coordsSpan.subspan(0, (size_t)sz));
          EXPECT_EQ(coords[i], result.GetMin());
          EXPECT_EQ(Real3{}, result.GetMax());

          // Place the maximum value here
          coords[i][j] = 1_r;
          result = CalcAabb(coordsSpan.subspan(0, (size_t)sz));
          EXPECT_EQ(Real3{}, result.GetMin());
          EXPECT_EQ(coords[i], result.GetMax());

          // Reset
          coords[i][j] = 0_r;
        }
      }
    }
  }
}

TEST(Aabb, CalcAabbWithDisplacements) {
  {
    Real3 const coords[] = {
        Real3{1.0_r, 2.0_r, 3.0_r},
        Real3{1.1_r, 1.9_r, 3.1_r},
        Real3{0.9_r, 2.1_r, 2.9_r},
        Real3{1.2_r, 1.8_r, 3.2_r},
    };
    Real3 const displacements[] = {
        Real3{10.0_r, 20.0_r, 30.0_r},
        Real3{40.0_r, 50.0_r, 60.0_r},
        Real3{70.0_r, 80.0_r, 90.0_r},
        Real3{100.0_r, 200.0_r, 300.0_r},
    };

    // Empty span returns all zeros
    Aabb bounds = CalcAabbWithDisplacements(Span(coords, 0_uz), Span(displacements, 0_uz));
    EXPECT_EQ(Real3(0.0_r, 0.0_r, 0.0_r), bounds.GetMin());
    EXPECT_EQ(Real3(0.0_r, 0.0_r, 0.0_r), bounds.GetMax());

    // Length 1
    bounds = CalcAabbWithDisplacements(Span(coords, 1_uz), Span(displacements, 1_uz));
    EXPECT_NEAR_EQ(Real3(11.0_r, 22.0_r, 33.0_r), bounds.GetMin());
    EXPECT_NEAR_EQ(Real3(11.0_r, 22.0_r, 33.0_r), bounds.GetMax());

    // Length 2
    bounds = CalcAabbWithDisplacements(Span(coords, 2_uz), Span(displacements, 2_uz));
    EXPECT_NEAR_EQ(Real3(11.0_r, 22.0_r, 33.0_r), bounds.GetMin());
    EXPECT_NEAR_EQ(Real3(41.1_r, 51.9_r, 63.1_r), bounds.GetMax());

    // Length 3
    bounds = CalcAabbWithDisplacements(Span(coords, 3_uz), Span(displacements, 3_uz));
    EXPECT_NEAR_EQ(Real3(11.0_r, 22.0_r, 33.0_r), bounds.GetMin());
    EXPECT_NEAR_EQ(Real3(70.9_r, 82.1_r, 92.9_r), bounds.GetMax());

    // Length 4
    bounds = CalcAabbWithDisplacements(Span(coords, 4_uz), Span(displacements, 4_uz));
    EXPECT_NEAR_EQ(Real3(11.0_r, 22.0_r, 33.0_r), bounds.GetMin());
    EXPECT_NEAR_EQ(Real3(101.2_r, 201.8_r, 303.2_r), bounds.GetMax());

    // CalcAabbWithSortedIndices (and displacements)
    {
      int indices[] = {0, 2};
      bounds =
          CalcAabbWithSortedIndices(MakeSpan(coords), MakeSpan(displacements), MakeSpan(indices));
      EXPECT_NEAR_EQ(Real3(11.0_r, 22.0_r, 33.0_r), bounds.GetMin());
      EXPECT_NEAR_EQ(Real3(70.9_r, 82.1_r, 92.9_r), bounds.GetMax());
    }
    {
      int indices[] = {1, 3};
      bounds =
          CalcAabbWithSortedIndices(MakeSpan(coords), MakeSpan(displacements), MakeSpan(indices));
      EXPECT_NEAR_EQ(Real3(41.1_r, 51.9_r, 63.1_r), bounds.GetMin());
      EXPECT_NEAR_EQ(Real3(101.2_r, 201.8_r, 303.2_r), bounds.GetMax());
    }
  }

  // Now test a larger number of points and vary the location of the min and max values.
  {
    std::vector<Real3> coords(50);
    std::vector<Real3> displacements(50);
    auto coordsSpan = MakeConstSpan(coords);
    auto dispSpan = MakeConstSpan(displacements);
    for (int sz = 2; sz < isize(coords); ++sz) {
      for (int i = 0; i < sz; ++i) {
        for (int j = 0; j < 3; ++j) {
          // Place the minimum value here
          coords[i][j] = -1;
          displacements[i][j] = -0.1_r;
          Aabb result = CalcAabbWithDisplacements(
              coordsSpan.subspan(0, (size_t)sz), dispSpan.subspan(0, (size_t)sz));
          EXPECT_EQ(coords[i] + displacements[i], result.GetMin());
          EXPECT_EQ(Real3{}, result.GetMax());

          // Place the maximum value here
          coords[i][j] = 1_r;
          result = CalcAabbWithDisplacements(
              coordsSpan.subspan(0, (size_t)sz), dispSpan.subspan(0, (size_t)sz));
          EXPECT_EQ(Real3{}, result.GetMin());
          EXPECT_EQ(coords[i] + displacements[i], result.GetMax());

          // Reset
          coords[i][j] = 0_r;
          displacements[i][j] = 0_r;
        }
      }
    }
  }
}

TEST(Aabb, GetAabb) {
  auto a = Aabb{Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}};
  auto b = Aabb{Real3{-2_r, 2_r, -4_r}, Real3{7_r, 4_r, 8_r}};
  auto c = Aabb{Real3{2_r, 3_r, 4_r}, Real3{3_r, 4_r, 5_r}}; // inside a

  // GetAabb(a) returns a (just so GetAabb(shape) works for any shape).
  EXPECT_EQ(a, GetAabb(a));

  // GetAabb(a, b) returns an Aabb that contains both a and b.
  EXPECT_EQ(a, GetAabb(a, a));
  EXPECT_EQ(b, GetAabb(b, b));
  EXPECT_EQ(a, GetAabb(a, c));
  EXPECT_EQ(a, GetAabb(c, a));
  EXPECT_EQ(Aabb(Real3{-2_r, 2_r, -4_r}, Real3{7_r, 5_r, 8_r}), GetAabb(a, b));
  EXPECT_EQ(Aabb(Real3{-2_r, 2_r, -4_r}, Real3{7_r, 5_r, 8_r}), GetAabb(b, a));
}

TEST(Aabb, TransformShape) {
  // 90 degree rotations
  Quaternion rotX = Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, 0.5_r * kPI);
  Quaternion rotY = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, 0.5_r * kPI);
  Quaternion rotZ = Quaternion::FromAxisAngle(Real3{0_r, 0_r, 1_r}, 0.5_r * kPI);
  Aabb bounds(Real3{-1_r, -2_r, -3_r}, Real3{0.1_r, 0.2_r, 0.3_r});
  Aabb bounds2 = TransformShape(TransformRT(rotX, Real3{}), bounds);
  EXPECT_NEAR_EQ(Aabb(Real3(-1_r, -0.3_r, -2_r), Real3(0.1_r, 3.0_r, 0.2_r)), bounds2);
  bounds2 = TransformShape(TransformRT(rotY, Real3{}), bounds);
  EXPECT_NEAR_EQ(Aabb(Real3(-3_r, -2_r, -0.1_r), Real3(0.3_r, 0.2_r, 1_r)), bounds2);
  bounds2 = TransformShape(TransformRT(rotZ, Real3{}), bounds);
  EXPECT_NEAR_EQ(Aabb(Real3(-0.2_r, -1_r, -3_r), Real3(2_r, 0.1_r, 0.3_r)), bounds2);

  // Rotation + translation
  auto rotXTrans = TransformRT(rotX, Real3{100_r, 200_r, 300_r});
  bounds2 = TransformShape(rotXTrans, bounds);
  EXPECT_NEAR_EQ(Aabb(Real3(99_r, 199.7_r, 298_r), Real3(100.1_r, 203_r, 300.2_r)), bounds2);

  // VMatrix4x4: Non-uniform scale
  auto scaleMat = VDiagonalMatrix<4>(Vec4r{2_r, 4_r, 8_r, 1_r});
  bounds2 = TransformShape(scaleMat, bounds);
  EXPECT_NEAR_EQ(Aabb(Real3{-2_r, -8_r, -24_r}, Real3{0.2_r, 0.8_r, 2.4_r}), bounds2);

  // VMatrix4x4: Negative scale (mirroring)
  auto negScaleMat = VDiagonalMatrix<4>(Vec4r{-2_r, -4_r, -8_r, 1_r});
  bounds2 = TransformShape(negScaleMat, bounds);
  EXPECT_NEAR_EQ(Aabb(Real3{-0.2_r, -0.8_r, -2.4_r}, Real3{2_r, 8_r, 24_r}), bounds2);

  // VMatrix4x4r: Rotation + translation
  bounds2 = TransformShape(ToVMatrix4x4(rotXTrans), bounds);
  EXPECT_NEAR_EQ(Aabb(Real3(99_r, 199.7_r, 298_r), Real3(100.1_r, 203_r, 300.2_r)), bounds2);

  // VMatrix4x4r: Scale + Rotation + translation
  bounds2 = TransformShape(Dot4x4(ToVMatrix4x4(rotXTrans), scaleMat), bounds);
  EXPECT_NEAR_EQ(Aabb(Real3(98_r, 197.6_r, 292_r), Real3(100.2_r, 224_r, 300.8_r)), bounds2);
}

TEST(Aabb, GetBoundingSphere) {
  Aabb aabb; // point
  EXPECT_NEAR_EQ(Sphere(), GetBoundingSphere(aabb));
  aabb = Aabb{Real3{1_r, 1_r, 1_r}, Real3{1_r, 1_r, 1_r}}; // point
  EXPECT_NEAR_EQ(Sphere(Real3(1_r, 1_r, 1_r), 0_r), GetBoundingSphere(aabb));
  aabb = Aabb{Real3{-1_r, -2_r, 0_r}, Real3{0_r, -1_r, 0_r}}; // 2D square
  EXPECT_NEAR_EQ(Sphere(Real3(-0.5_r, -1.5_r, 0_r), std::sqrt(2_r) / 2_r), GetBoundingSphere(aabb));
  aabb = Aabb{Real3{1_r, 2_r, 3_r}, Real3{2_r, 3_r, 4_r}}; // 3D volume
  EXPECT_NEAR_EQ(Sphere(Real3(1.5_r, 2.5_r, 3.5_r), std::sqrt(3_r) / 2_r), GetBoundingSphere(aabb));
}

TEST(Aabb, ContainsPoint) {
  // Degenerate
  {
    auto aabb = Aabb{};
    EXPECT_TRUE(ContainsPoint(aabb, Real3()));
    EXPECT_FALSE(ContainsPoint(aabb, Real3(1e-6_r, 0_r, 0_r)));
  }

  // Points exactly on the surface
  {
    auto aabb = Aabb{Real3{}, Real3{1_r, 1_r, 1_r}};
    EXPECT_TRUE(ContainsPoint(aabb, Real3(0_r, 0_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(1_r, 0_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(0_r, 1_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(0_r, 0_r, 1_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(1_r, 1_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(0_r, 1_r, 1_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(1_r, 0_r, 1_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(1_r, 1_r, 1_r)));
  }

  // Normal cases
  {
    auto aabb = Aabb{Real3{-1_r, -2_r, -3_r}, Real3{0.1_r, 0.2_r, 0.3_r}};
    EXPECT_TRUE(ContainsPoint(aabb, Real3(0_r, 0_r, 0_r)));

    // Barely inside
    EXPECT_TRUE(ContainsPoint(aabb, Real3(0.09999_r, 0_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(0_r, 0.19999_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(0_r, 0_r, 0.29999_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(-0.99999_r, 0_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(0_r, -1.99999_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(0_r, 0_r, -2.99999_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(0.09999_r, 0.19999_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(0.09999_r, 0_r, 0.29999_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(0_r, 0.19999_r, 0.29999_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(0.09999_r, 0.19999_r, 0.29999_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(-0.99999_r, -1.99999_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(-0.99999_r, 0_r, -2.99999_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(0_r, -1.99999_r, -2.99999_r)));
    EXPECT_TRUE(ContainsPoint(aabb, Real3(-0.99999_r, -1.99999_r, -2.99999_r)));

    // Barely outside
    EXPECT_FALSE(ContainsPoint(aabb, Real3(0.10001_r, 0_r, 0_r)));
    EXPECT_FALSE(ContainsPoint(aabb, Real3(0_r, 0.20001_r, 0_r)));
    EXPECT_FALSE(ContainsPoint(aabb, Real3(0_r, 0_r, 0.30001_r)));
    EXPECT_FALSE(ContainsPoint(aabb, Real3(-1.00001_r, 0_r, 0_r)));
    EXPECT_FALSE(ContainsPoint(aabb, Real3(0_r, -2.00001_r, 0_r)));
    EXPECT_FALSE(ContainsPoint(aabb, Real3(0_r, 0_r, -3.00001_r)));
    EXPECT_FALSE(ContainsPoint(aabb, Real3(0.10001_r, 0.20001_r, 0_r)));
    EXPECT_FALSE(ContainsPoint(aabb, Real3(0.10001_r, 0_r, 0.30001_r)));
    EXPECT_FALSE(ContainsPoint(aabb, Real3(0_r, 0.20001_r, 0.30001_r)));
    EXPECT_FALSE(ContainsPoint(aabb, Real3(0.10001_r, 0.20001_r, 0.30001_r)));
    EXPECT_FALSE(ContainsPoint(aabb, Real3(-1.00001_r, -2.00001_r, 0_r)));
    EXPECT_FALSE(ContainsPoint(aabb, Real3(-0.99999_r, 0_r, -3.00001_r)));
    EXPECT_FALSE(ContainsPoint(aabb, Real3(0_r, -2.00001_r, -3.00001_r)));
    EXPECT_FALSE(ContainsPoint(aabb, Real3(-1.00001_r, -2.00001_r, -3.00001_r)));
  }
}

TEST(Aabb, HasOverlap) {
  // degenerate aabb
  {
    auto a = Aabb{};
    EXPECT_TRUE(HasOverlap(a, Aabb()));
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-1_r, 0_r, 0_r), Real3(1_r, 0_r, 0_r))));
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0_r, -1_r, 0_r), Real3(0_r, 1_r, 0_r))));
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0_r, 0_r, -1_r), Real3(0_r, 0_r, 1_r))));
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0_r, 0_r, 0_r), Real3(1_r, 1_r, 1_r))));
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-1_r, -1_r, -1_r), Real3(0_r, 0_r, 0_r))));
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(0.00001_r, 0.00001_r, 0.00001_r), Real3(1_r, 1_r, 1_r))));
  }

  // exactly touching
  {
    auto a = Aabb{Real3{0_r, 0_r, 0_r}, Real3{1_r, 1_r, 1_r}};
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(1_r, 0_r, 0_r), Real3(2_r, 1_r, 1_r)))); // +x face
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0_r, 1_r, 0_r), Real3(1_r, 2_r, 1_r)))); // +y face
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0_r, 0_r, 1_r), Real3(1_r, 1_r, 2_r)))); // +z face
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-1_r, 0_r, 0_r), Real3(0_r, 1_r, 1_r)))); // -x face
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0_r, -1_r, 0_r), Real3(1_r, 0_r, 1_r)))); // -y face
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0_r, 0_r, -1_r), Real3(1_r, 1_r, 0_r)))); // -z face
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(1_r, 1_r, 0_r), Real3(2_r, 2_r, 1_r)))); // +x+y edge
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(1_r, 0_r, 1_r), Real3(2_r, 0_r, 2_r)))); // +x+z edge
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0_r, 1_r, 1_r), Real3(0_r, 2_r, 2_r)))); // +y+z edge
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-1_r, -1_r, 0_r), Real3(0_r, 0_r, 1_r)))); // +x+y edge
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-1_r, 0_r, -1_r), Real3(0_r, 1_r, 0_r)))); // +x+z edge
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0_r, -1_r, -1_r), Real3(1_r, 0_r, 1_r)))); // +y+z edge
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(1_r, 1_r, 1_r), Real3(2_r, 2_r, 2_r)))); // +x+y+z corner
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-1_r, 1_r, 1_r), Real3(0_r, 2_r, 2_r)))); // -x+y+z corner
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(1_r, -1_r, 1_r), Real3(2_r, 0_r, 2_r)))); // +x-y+z corner
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(1_r, 1_r, -1_r), Real3(2_r, 2_r, 0_r)))); // +x+y-z corner
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-1_r, -1_r, 1_r), Real3(0_r, 0_r, 2_r)))); // -x-y+z corner
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-1_r, 1_r, -1_r), Real3(0_r, 2_r, 0_r)))); // -x+y-z corner
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(1_r, -1_r, -1_r), Real3(2_r, 0_r, 0_r)))); // +x-y-z corner
    EXPECT_TRUE(
        HasOverlap(a, Aabb(Real3(-1_r, -1_r, -1_r), Real3(0_r, 0_r, 0_r)))); // -x-y-z corner
  }

  // normal cases
  {
    // clang-format off
    auto a = Aabb{Real3{-1_r, -2_r, -3_r}, Real3{0.1_r, 0.2_r, 0.3_r}};
    EXPECT_TRUE(HasOverlap(a, a));
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-10_r, -10_r, -10_r), Real3(10_r, 10_r, 10_r)))); // a inside b
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-0.5_r, -1_r, -1.5_r), Real3(0_r, 0_r, 0_r)))); // b inside a
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0.0999_r, -2_r, -3_r), Real3(1_r, 1_r, 1_r)))); // +x face overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0_r, 0.1999_r, 0_r), Real3(1_r, 1_r, 1_r)))); // +y face overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0_r, 0_r, 0.2999_r), Real3(1_r, 1_r, 1_r)))); // +z face overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-4_r, -4_r, -4_r), Real3(-0.9999_r, 1_r, 1_r)))); // -x face overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-4_r, -4_r, -4_r), Real3(1_r, -1.9999_r, 1_r)))); // -y face overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-4_r, -4_r, -4_r), Real3(1_r, 1_r, -2.9999_r)))); // -z face overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0.0999_r, 0.1999_r, 0_r), Real3(1_r, 1_r, 1_r)))); // +x+y edge overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0.0999_r, 0_r, 0.2999_r), Real3(1_r, 1_r, 1_r)))); // +x+z edge overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0_r, 0.1999_r, 0.2999_r), Real3(1_r, 1_r, 1_r)))); // +y+z edge overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0.0999_r, 0.1999_r, 0_r), Real3(1_r, 1_r, 1_r)))); // +x+y edge overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0.0999_r, 0_r, 0.2999_r), Real3(1_r, 1_r, 1_r)))); // +x+z edge overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0_r, 0.1999_r, 0.2999_r), Real3(1_r, 1_r, 1_r)))); // +y+z edge overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0.0999_r, 0.1999_r, 0.2999_r), Real3(1_r, 1_r, 1_r)))); // +x+y+z corner overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-4_r, 0.1999_r, 0.2999_r), Real3(0.9999_r, 1_r, 1_r)))); // -x+y+z corner overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0.0999_r, -4_r, 0.2999_r), Real3(1_r, -1.9999_r, 1_r)))); // +x-y+z corner overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0.0999_r, 0.1999_r, -4_r), Real3(1_r, 1_r, -2.9999_r)))); // +x+y-z corner overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-4_r, -4_r, 0.2999_r), Real3(-0.9999_r, -1.9999_r, 1_r)))); // -x-y+z corner overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-4_r, 0.1999_r, -4_r), Real3(-0.9999_r, 1_r, -2.9999_r)))); // -x+y-z corner overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(0.0999_r, -4_r, -4_r), Real3(1_r, -1.9999_r, -2.9999_r)))); // +x-y-z corner overlap
    EXPECT_TRUE(HasOverlap(a, Aabb(Real3(-4_r, -4_r, -4_r), Real3(-0.9999_r, -1.9999_r, -2.9999_r)))); // -x-y-z corner overlap
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(0.1001_r, -2_r, -3_r), Real3(1_r, 1_r, 1_r)))); // +x face outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(0_r, 0.2001_r, 0_r), Real3(1_r, 1_r, 1_r)))); // +y face outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(0_r, 0_r, 0.3001_r), Real3(1_r, 1_r, 1_r)))); // +z face outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(-4_r, -4_r, -4_r), Real3(-1.0001_r, 1_r, 1_r)))); // -x face outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(-4_r, -4_r, -4_r), Real3(1_r, -2.0001_r, 1_r)))); // -y face outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(-4_r, -4_r, -4_r), Real3(1_r, 1_r, -3.0001_r)))); // -z face outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(0.1001_r, 0.2001_r, 0_r), Real3(1_r, 1_r, 1_r)))); // +x+y edge outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(0.1001_r, 0_r, 0.3001_r), Real3(1_r, 1_r, 1_r)))); // +x+z edge outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(0_r, 0.2001_r, 0.3001_r), Real3(1_r, 1_r, 1_r)))); // +y+z edge outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(0.1001_r, 0.2001_r, 0_r), Real3(1_r, 1_r, 1_r)))); // +x+y edge outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(0.1001_r, 0_r, 0.3001_r), Real3(1_r, 1_r, 1_r)))); // +x+z edge outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(0_r, 0.2001_r, 0.3001_r), Real3(1_r, 1_r, 1_r)))); // +y+z edge outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(0.1001_r, 0.2001_r, 0.3001_r), Real3(1_r, 1_r, 1_r)))); // +x+y+z corner outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(-4_r, 0.2001_r, 0.3001_r), Real3(1.0001_r, 1_r, 1_r)))); // -x+y+z corner outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(0.1001_r, -4_r, 0.3001_r), Real3(1_r, -2.0001_r, 1_r)))); // +x-y+z corner outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(0.1001_r, 0.2001_r, -4_r), Real3(1_r, 1_r, -3.0001_r)))); // +x+y-z corner outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(-4_r, -4_r, 0.3001_r), Real3(-1.0001_r, -2.0001_r, 1_r)))); // -x-y+z corner outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(-4_r, 0.2001_r, -4_r), Real3(-1.0001_r, 1_r, -3.0001_r)))); // -x+y-z corner outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(0.1001_r, -4_r, -4_r), Real3(1_r, -2.0001_r, -3.0001_r)))); // +x-y-z corner outside
    EXPECT_FALSE(HasOverlap(a, Aabb(Real3(-4_r, -4_r, -4_r), Real3(-1.0001_r, -2.0001_r, -3.0001_r)))); // -x-y-z corner outside

    // clang-format on
  }
}

TEST(Aabb, ExpandShape) {
  // From empty
  {
    Aabb shape;
    ExpectAabb(Real3{}, Real3{}, shape); // zero volume
    shape = ExpandShape(shape, 1_r);
    ExpectAabb(Real3{-1_r, -1_r, -1_r}, Real3{1_r, 1_r, 1_r}, shape);
  }

  // From non-empty
  {
    Real3 min{1_r, 2_r, 3_r};
    Real3 max{4_r, 5_r, 6_r};
    Aabb shape(min, max);
    ExpectAabb(min, max, shape);
    shape = ExpandShape(shape, 0_r);
    ExpectAabb(min, max, shape); // no change
    shape = ExpandShape(shape, 1_r);
    ExpectAabb(min - 1_r, max + 1_r, shape);
    auto any = ExpandShape(AnyShape{shape}, -1_r); // via AnyShape
    ExpectAabb(min, max, std::get<Aabb>(any));
  }
}

TEST(Aabb, SetMinMax) {
  Aabb aabb;
  aabb.SetMin(Real3{1_r, 2_r, 3_r});
  aabb.SetMax(Real3{4_r, 5_r, 6_r});
  ExpectAabb(Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, aabb);
}

TEST(Aabb, Reflection) {
  auto aabb = Aabb{Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}};
  auto json = R"({"max":[4,5,6],"min":[1,2,3]})";
  auto const& ti = SReflect::GetTypeInfo<Aabb>();
  EXPECT_STREQ(json, SReflect::ToJsonString(aabb, false /*pretty*/).c_str());
  EXPECT_NEAR_EQ(aabb, SReflect::FromJsonString<Aabb>(json));
  EXPECT_EQ(SReflect::CoreType::CT_struct, ti._coreType);
  EXPECT_STREQ("mochi::Aabb", ti._nameWithNamespace);
  EXPECT_STREQ("Aabb", ti._name);
}

TEST(Aabb, GetVolume) {
  EXPECT_EQ(0_r, GetVolume(Aabb{Real3{}, Real3{}})); // empty volume
  EXPECT_EQ(0_r, GetVolume(Aabb{Real3{1_r, 2_r, 3_r}, Real3{1_r, 2_r, 3_r}})); // empty volume
  EXPECT_EQ(0_r, GetVolume(Aabb{Real3{-1_r, -2_r, 3_r}, Real3{1_r, 2_r, 3_r}})); // empty volume
  EXPECT_NEAR_EQ(48_r, GetVolume(Aabb{Real3{-1_r, -2_r, -3_r}, Real3{1_r, 2_r, 3_r}}));
}

/**********************************************************************************************
  Capsule
*/

TEST(Capsule, Class) {
  // Default
  {
    Capsule a;
    ExpectCapsule(Real3{}, Real3{}, 0_r, a);
  }

  // From Real3 points
  {
    Capsule a = Capsule::FromPoints(Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, 42_r);
    ExpectCapsule(Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, 42_r, a);
  }

  // From Vec4r points
  {
    Capsule a = Capsule::FromPoints(Vec4r(1_r, 2_r, 3_r), Vec4r(4_r, 5_r, 6_r), 42_r);
    ExpectCapsule(Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, 42_r, a);
  }

  // From Real3 point + vector
  {
    Capsule a = Capsule::FromPointAndVector(Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, 42_r);
    ExpectCapsule(Real3{1_r, 2_r, 3_r}, Real3{5_r, 7_r, 9_r}, 42_r, a);
  }

  // From Vec4r point + vector
  {
    Capsule a = Capsule::FromPointAndVector(Vec4r(1_r, 2_r, 3_r), Vec4r(4_r, 5_r, 6_r), 42_r);
    ExpectCapsule(Real3{1_r, 2_r, 3_r}, Real3{5_r, 7_r, 9_r}, 42_r, a);
  }
}

TEST(Capsule, NearEqual) {
  Capsule a = Capsule::FromPoints(Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, 42_r);
  Capsule b = Capsule::FromPoints(Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, 42_r); // same as a
  Capsule c =
      Capsule::FromPoints(Real3{1.1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, 42_r); // different posA
  Capsule d =
      Capsule::FromPoints(Real3{1.0_r, 2_r, 3_r}, Real3{4_r, 5_r, 6.1_r}, 42_r); // different posB
  Capsule e =
      Capsule::FromPoints(Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, 42.1_r); // different scale
  EXPECT_EQ(true, NearEqual(a, a, 0.2_r)); // equal
  EXPECT_EQ(true, NearEqual(a, b, 0.2_r)); // equal
  EXPECT_EQ(true, NearEqual(a, c, 0.2_r)); // close enough
  EXPECT_EQ(true, NearEqual(a, d, 0.2_r)); // close enough
  EXPECT_EQ(true, NearEqual(a, e, 0.2_r)); // close enough

  EXPECT_EQ(true, NearEqual(a, a, 0.02_r)); // equal
  EXPECT_EQ(true, NearEqual(a, b, 0.02_r)); // equal
  EXPECT_EQ(false, NearEqual(a, c, 0.02_r)); // not close enough
  EXPECT_EQ(false, NearEqual(a, d, 0.02_r)); // not close enough
  EXPECT_EQ(false, NearEqual(a, e, 0.02_r)); // not close enough
}

TEST(Capsule, TransformShape) {
  // 90 degree rotations
  Quaternion rotX = Quaternion::RotationX(0.5_r * kPI);
  Quaternion rotZ = Quaternion::RotationZ(0.5_r * kPI);
  Capsule cap1 =
      Capsule::FromPointAndVector(Real3{100_r, 200_r, 300_r}, Real3{1_r, 2_r, 3_r}, 42_r);
  Capsule cap2, cap3;

  // +90 about X, unit scale
  cap2 = TransformShape(TransformRT(rotX, Real3{}), cap1);
  cap3 = TransformShape(TransformSRT(1_r, rotX, Real3{}), cap1);
  real constexpr kTolerance = std::numeric_limits<real>::epsilon() *
      1000_r; // About 1e-4 for floats. Quaternion rotation is not exact because of sin and cos.
  ExpectCapsule(Real3{100_r, -300_r, 200_r}, Real3{101_r, -303_r, 202_r}, 42_r, cap2, kTolerance);
  EXPECT_NEAR_EQ(cap2, cap3);

  // Rotate +90 about Z, then translate
  cap2 = TransformShape(TransformRT{rotZ, Real3{0.1_r, 0.2_r, 0.3_r}}, cap1);
  cap3 = TransformShape(TransformSRT{1_r, rotZ, Real3{0.1_r, 0.2_r, 0.3_r}}, cap1);
  ExpectCapsule(
      Real3{-199.9_r, 100.2_r, 300.3_r}, Real3{-201.9_r, 101.2_r, 303.3_r}, 42_r, cap2, kTolerance);
  EXPECT_NEAR_EQ(cap2, cap3);

  // Scale, then rotate +90 about Z, then translate
  cap3 = TransformShape(TransformSRT{10_r, rotZ, Real3{0.1_r, 0.2_r, 0.3_r}}, cap1);
  ExpectCapsule(
      Real3{-1999.9_r, 1000.2_r, 3000.3_r},
      Real3{-2019.9_r, 1010.2_r, 3030.3_r},
      420_r,
      cap3,
      kTolerance * 10_r); // tolerace about 1e-3 for floats
}

/**********************************************************************************************
  Obb
*/

TEST(Obb, NearEqual) {
  Matrix3x3r rot = Eye<3>();
  Real3 center = {0.1_r, 0.2_r, 0.3_r};
  Real3 halfExtent = {1_r, 2_r, 3_r};
  Obb ref{MatrixTransformRT{rot, center}, halfExtent};
  EXPECT_EQ(true, NearEqual(ref, ref));

  // Different rotation
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      real prev = rot[i][j];
      rot[i][j] += 0.9e-5_r; // close enough
      EXPECT_EQ(true, NearEqual(ref, Obb(MatrixTransformRT(rot, center), halfExtent), 1e-5_r));
      rot[i][j] += 0.2e-5_r; // not close enough
      EXPECT_EQ(false, NearEqual(ref, Obb(MatrixTransformRT(rot, center), halfExtent), 1e-5_r));
      rot[i][j] = prev; // revert
    }
  }

  // Different translation
  for (int i = 0; i < 3; ++i) {
    real prev = center[i];
    center[i] += 0.9e-5_r; // close enough
    EXPECT_EQ(true, NearEqual(ref, Obb(MatrixTransformRT(rot, center), halfExtent), 1e-5_r));
    center[i] += 0.2e-5_r; // not close enough
    EXPECT_EQ(false, NearEqual(ref, Obb(MatrixTransformRT(rot, center), halfExtent), 1e-5_r));
    center[i] = prev; // revert
  }

  // Different extents
  for (int i = 0; i < 3; ++i) {
    real prev = halfExtent[i];
    halfExtent[i] += 0.9e-5_r; // close enough
    EXPECT_EQ(true, NearEqual(ref, Obb(MatrixTransformRT(rot, center), halfExtent), 1e-5_r));
    halfExtent[i] += 0.2e-5_r; // not close enough
    EXPECT_EQ(false, NearEqual(ref, Obb(MatrixTransformRT(rot, center), halfExtent), 1e-5_r));
    halfExtent[i] = prev; // revert
  }

  // 4th SIMD component ignored
  VMatrix3x3r vrot{ToSimd(rot[0], 911_r), ToSimd(rot[1], 911_r), ToSimd(rot[2], 911_r)};
  Vec4r vcenter = ToSimd(center, 911_r);
  Vec4r vhalfExtent = ToSimd(halfExtent, 911_r);
  EXPECT_EQ(
      true, NearEqual(ref, Obb(MatrixTransformRT(vrot, vcenter), vhalfExtent))); // still equal
}

TEST(Obb, CalcObb) {
  // Empty span returns all zeros
  Obb bounds = CalcObb({(Real3*)nullptr, 0_uz});
  EXPECT_NEAR_EQ(Real3(0_r, 0_r, 0_r), bounds.GetHalfExtents());

  // Test against known point clouds.
  for (int i = 0; i < kObbTest_NumFits; ++i) {
    bounds = CalcObb(MakeSpan(kObbTest_Points[i]));
    ExpectObb(kObbTest_Translations[i], kObbTest_Rotations[i], kObbTest_Extents[i], bounds);

    // Make sure the Obb contains the point cloud itself.
    for (int j = 0; j < kObbTest_NumPoints; ++j) {
      EXPECT_TRUE(ContainsPoint(bounds, kObbTest_Points[i][j]));
    }
  }
}

TEST(Obb, ContainsPoint) {
  // Degenerate
  {
    auto oobb = Obb{};
    EXPECT_TRUE(ContainsPoint(oobb, Real3()));
    EXPECT_FALSE(ContainsPoint(oobb, Real3(1e-6_r, 0_r, 0_r)));
  }

  // Points exactly on the surface
  {
    auto oobb = Obb{MatrixTransformRT{Eye<3>(), Real3{0.5_r, 0.5_r, 0.5_r}}, Real3{1_r, 1_r, 1_r}};
    EXPECT_TRUE(ContainsPoint(oobb, Real3(0_r, 0_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(1_r, 0_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(0_r, 1_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(0_r, 0_r, 1_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(1_r, 1_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(0_r, 1_r, 1_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(1_r, 0_r, 1_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(1_r, 1_r, 1_r)));
  }

  // Normal cases
  {
    Matrix3x3r rotX90 = {Real3{1_r, 0_r, 0_r}, Real3{0_r, 0_r, 1_r}, Real3{0_r, -1_r, 0_r}};
    Real3 halfExtents = {1_r, 3_r, 2_r}; // Becomes (1,2,3) when rotated
    Real3 center = {10_r, 20_r, 30_r};
    auto oobb = Obb{MatrixTransformRT{rotX90, center}, halfExtents};
    EXPECT_TRUE(ContainsPoint(oobb, center));

    // Barely inside
    EXPECT_TRUE(ContainsPoint(oobb, Real3(10.99999_r, 20_r, 30_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(10_r, 21.99999_r, 30_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(10_r, 20_r, 32.99999_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(9.00001_r, 20_r, 30_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(10_r, 18.00001_r, 30_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(10_r, 20_r, 27.00001_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(10.99999_r, 21.99999_r, 30_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(10.9999_r, 20_r, 32.99999_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(10_r, 21.99999_r, 32.99999_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(10.99999_r, 21.99999_r, 32.99999_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(9.00001_r, 18.00001_r, 30_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(9.00001_r, 20_r, 27.00001_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(10_r, 18.00001_r, 27.00001_r)));
    EXPECT_TRUE(ContainsPoint(oobb, Real3(9.00001_r, 18.00001_r, 27.00001_r)));

    // Barely outside
    EXPECT_FALSE(ContainsPoint(oobb, Real3(11.00001_r, 20_r, 30_r)));
    EXPECT_FALSE(ContainsPoint(oobb, Real3(10_r, 22.00001_r, 30_r)));
    EXPECT_FALSE(ContainsPoint(oobb, Real3(10_r, 20_r, 33.00001_r)));
    EXPECT_FALSE(ContainsPoint(oobb, Real3(8.99999_r, 20_r, 30_r)));
    EXPECT_FALSE(ContainsPoint(oobb, Real3(10_r, 17.99999_r, 30_r)));
    EXPECT_FALSE(ContainsPoint(oobb, Real3(10_r, 20_r, 26.99999_r)));
    EXPECT_FALSE(ContainsPoint(oobb, Real3(11.00001_r, 22.00001_r, 30_r)));
    EXPECT_FALSE(ContainsPoint(oobb, Real3(10.9999_r, 20_r, 33.00001_r)));
    EXPECT_FALSE(ContainsPoint(oobb, Real3(10_r, 22.00001_r, 33.00001_r)));
    EXPECT_FALSE(ContainsPoint(oobb, Real3(11.00001_r, 22.00001_r, 33.00001_r)));
    EXPECT_FALSE(ContainsPoint(oobb, Real3(8.99999_r, 17.99999_r, 30_r)));
    EXPECT_FALSE(ContainsPoint(oobb, Real3(8.99999_r, 20_r, 26.99999_r)));
    EXPECT_FALSE(ContainsPoint(oobb, Real3(10_r, 17.99999_r, 26.99999_r)));
    EXPECT_FALSE(ContainsPoint(oobb, Real3(8.99999_r, 17.99999_r, 26.99999_r)));
  }
}

TEST(Obb, EncloseShapes) {
  // Test all possible combinations.
  for (int i = 0; i < kObbTest_NumFits; ++i) {
    for (int j = 0; j < kObbTest_NumFits; ++j) {
      int k = j * kObbTest_NumFits + i;
      auto Ta = MatrixTransformRT{kObbTest_Rotations[i], kObbTest_Translations[i]};
      auto Tb = MatrixTransformRT{kObbTest_Rotations[j], kObbTest_Translations[j]};
      auto a = Obb{Ta, kObbTest_Extents[i]};
      auto b = Obb{Tb, kObbTest_Extents[j]};
      auto c = EncloseShapes(a, b);
      ExpectObb(
          kObbTest_MergeTranslations[k], kObbTest_MergeRotations[k], kObbTest_MergeExtents[k], c);
    }
  }
}

TEST(Obb, HasOverlap_Obb) {
  // Test all possible combinations.
  for (int i = 0; i < kObbTest_NumFits; ++i) {
    for (int j = 0; j < kObbTest_NumFits; ++j) {
      auto Ta = MatrixTransformRT{kObbTest_Rotations[i], kObbTest_Translations[i]};
      auto Tb = MatrixTransformRT{kObbTest_Rotations[j], kObbTest_Translations[j]};
      auto a = Obb{Ta, kObbTest_Extents[i]};
      auto b = Obb{Tb, kObbTest_Extents[j]};
      EXPECT_EQ(HasOverlap(a, b), kObbTest_Overlaps[i][j]);
    }
  }
}

TEST(Obb, GetAabb) {
  // Identity transform
  {
    Obb oobb(MatrixTransformRT{}, Real3{1_r, 2_r, 3_r});
    Aabb aabb = GetAabb(oobb);
    ExpectAabb(Real3{-1_r, -2_r, -3_r}, Real3{1_r, 2_r, 3_r}, aabb);
  }

  // Rotated & translated
  {
    Quaternion rotX = Quaternion::RotationX(0.5_r * kPI);
    Obb oobb(TransformRT{rotX, Real3{100_r, 200_r, 300_r}}, Real3{1_r, 2_r, 3_r});
    Aabb aabb = GetAabb(oobb);
    ExpectAabb(Real3{99_r, 197_r, 298_r}, Real3{101_r, 203_r, 302_r}, aabb);
  }
}

TEST(Obb, GetBoundingSphere) {
  // Identity transform
  {
    Obb oobb(MatrixTransformRT{}, Real3{1_r, 2_r, 3_r});
    Sphere s = GetBoundingSphere(oobb);
    ExpectSphere(Real3{0_r, 0_r, 0_r}, Norm(Real3{1_r, 2_r, 3_r}), s);
  }

  // Rotated & translated
  {
    Quaternion rotX = Quaternion::RotationX(0.5_r * kPI);
    Obb oobb(TransformRT{rotX, Real3{100_r, 200_r, 300_r}}, Real3{1_r, 2_r, 3_r});
    Sphere s = GetBoundingSphere(oobb);
    ExpectSphere(Real3{100_r, 200_r, 300_r}, Norm(Real3{1_r, 2_r, 3_r}), s);
  }
}

TEST(Obb, TransformShape) {
  // 90 degree rotations
  Quaternion rotX = Quaternion::RotationX(0.5_r * kPI);
  Quaternion rotY = Quaternion::RotationY(0.5_r * kPI);
  Quaternion rotZ = Quaternion::RotationZ(0.5_r * kPI);
  Obb box1(TransformRT{Real3{100_r, 200_r, 300_r}}, Real3{1_r, 2_r, 3_r});
  Obb box2;
  Matrix3x3r basis2;
  real const kTolerance = 1e-4_r;

  // +90 about X
  box2 = TransformShape(TransformRT(rotX, Real3{}), box1);
  basis2 = Transpose(box2.GetRotation());
  EXPECT_NEAR_TOL(Real3(1_r, 0_r, 0_r), basis2[0], kTolerance); // local X
  EXPECT_NEAR_TOL(Real3(0_r, 0_r, 1_r), basis2[1], kTolerance); // local Y
  EXPECT_NEAR_TOL(Real3(0_r, -1_r, 0_r), basis2[2], kTolerance); // local Z
  EXPECT_NEAR_TOL(Real3(100_r, -300_r, 200_r), box2.GetCenter(), kTolerance);
  EXPECT_NEAR_TOL(Real3(1_r, 2_r, 3_r), box2.GetHalfExtents(), kTolerance);

  // Then +90 about Y
  box2 = TransformShape(TransformRT(rotY, Real3{}), box2);
  basis2 = Transpose(box2.GetRotation());
  EXPECT_NEAR_TOL(Real3(0_r, 0_r, -1_r), basis2[0], kTolerance); // local X
  EXPECT_NEAR_TOL(Real3(1_r, 0_r, 0_r), basis2[1], kTolerance); // local Y
  EXPECT_NEAR_TOL(Real3(0_r, -1_r, 0_r), basis2[2], kTolerance); // local Z
  EXPECT_NEAR_TOL(Real3(200_r, -300_r, -100_r), box2.GetCenter(), kTolerance);
  EXPECT_NEAR_TOL(Real3(1_r, 2_r, 3_r), box2.GetHalfExtents(), kTolerance);

  // Then rotate +90 about Z and translate
  box2 = TransformShape(TransformRT{rotZ, Real3{0.1_r, 0.2_r, 0.3_r}}, box2);
  basis2 = Transpose(box2.GetRotation());
  EXPECT_NEAR_TOL(Real3(0_r, 0_r, -1_r), basis2[0], kTolerance); // local X
  EXPECT_NEAR_TOL(Real3(0_r, 1_r, 0_r), basis2[1], kTolerance); // local Y
  EXPECT_NEAR_TOL(Real3(1_r, 0_r, 0_r), basis2[2], kTolerance); // local Z
  EXPECT_NEAR_TOL(Real3(300.1_r, 200.2_r, -99.7_r), box2.GetCenter(), kTolerance);
  EXPECT_NEAR_TOL(Real3(1_r, 2_r, 3_r), box2.GetHalfExtents(), kTolerance);
}

TEST(Obb, ExpandShape) {
  // From empty
  {
    Obb shape;
    ExpectObb(Real3{}, Eye<3>(), Real3{}, shape);
    shape = ExpandShape(shape, 1_r);
    ExpectObb(Real3{}, Eye<3>(), Real3{1_r, 1_r, 1_r}, shape);
  }

  // From non-empty
  {
    MatrixTransformRT rt{
        ToVMatrix3x3(Quaternion::FromAxisAngle(Vec4r(1_r, 0_r, 0_r), 42_r)), Vec4r(1_r, 2_r, 3_r)};
    Obb shape{rt, Real3{0.1_r, 0.2_r, 0.3_r}};
    ExpectObb(rt.GetTranslation(), rt.GetRotation(), Real3{0.1_r, 0.2_r, 0.3_r}, shape);
    shape = ExpandShape(shape, 0_r);
    ExpectObb(rt.GetTranslation(), rt.GetRotation(), Real3{0.1_r, 0.2_r, 0.3_r}, shape);
    shape = ExpandShape(shape, 1_r);
    ExpectObb(rt.GetTranslation(), rt.GetRotation(), Real3{1.1_r, 1.2_r, 1.3_r}, shape);
    auto any = ExpandShape(AnyShape{shape}, -1_r); // via AnyShape
    ExpectObb(
        rt.GetTranslation(), rt.GetRotation(), Real3{0.1_r, 0.2_r, 0.3_r}, std::get<Obb>(any));
  }
}

TEST(Obb, GetVolume) {
  Obb obb;
  EXPECT_EQ(0_r, GetVolume(obb));
  obb = GetObb(Aabb{Real3{1_r, 2_r, 3_r}, Real3{1_r, 2_r, 3_r}}); // empty volume
  EXPECT_EQ(0_r, GetVolume(obb));
  obb = GetObb(Aabb{Real3{-1_r, -2_r, 3_r}, Real3{1_r, 2_r, 3_r}}); // empty volume
  EXPECT_EQ(0_r, GetVolume(obb));
  obb = GetObb(Aabb{Real3{-1_r, -2_r, -3_r}, Real3{1_r, 2_r, 3_r}});
  EXPECT_NEAR_EQ(48_r, GetVolume(obb));
  EXPECT_NEAR_EQ(48_r, GetVolume(TransformShape(TransformRT{}, obb)));
  EXPECT_NEAR_EQ(48_r, GetVolume(TransformShape(TransformRT{Real3{1_r, 2_r, 3_r}}, obb)));
  EXPECT_NEAR_EQ(
      48_r,
      GetVolume(
          TransformShape(TransformRT{Quaternion::FromRotationVector(Real3{1_r, 2_r, 3_r})}, obb)));
}

/**********************************************************************************************
  Plane
*/

TEST(Plane, Class) {
  // Default
  {
    Plane p;
    ExpectPlane(Real3{0_r, 1_r, 0_r}, 0_r, p); // we had to pick something valid
  }

  // From Real3
  {
    Plane p{Real3{1_r, 0_r, 0_r}, 123_r};
    ExpectPlane(Real3{1_r, 0_r, 0_r}, 123_r, p);
  }

  // From Vec4r
  {
    Plane p{Vec4r(1_r, 0_r, 0_r, 911_r), 123_r}; // discards w
    ExpectPlane(Real3{1_r, 0_r, 0_r}, 123_r, p);
  }
}

TEST(Plane, NearEqual) {
  Plane a{Real3{1_r, 2_r, 3_r}, 4_r};
  Plane b{Real3{1_r, 2_r, 3_r}, 4_r};
  Plane c{Real3{1.1_r, 2_r, 3_r}, 4_r};
  Plane d{Real3{1.0_r, 2_r, 3_r}, 4.1_r};
  EXPECT_EQ(true, NearEqual(a, a, 0.2_r));
  EXPECT_EQ(true, NearEqual(a, b, 0.2_r));
  EXPECT_EQ(true, NearEqual(a, c, 0.2_r));
  EXPECT_EQ(true, NearEqual(a, d, 0.2_r));
  EXPECT_EQ(true, NearEqual(a, a, 0.02_r));
  EXPECT_EQ(true, NearEqual(a, b, 0.02_r));
  EXPECT_EQ(false, NearEqual(a, c, 0.02_r));
  EXPECT_EQ(false, NearEqual(a, d, 0.02_r));
}

TEST(Plane, TransformShape) {
  // Plane of all points such that x == 5
  Plane p{Real3{1_r, 0_r, 0_r}, 5_r};
  Plane p2;

  auto rotX = Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, kPI / 2_r);
  auto rotY = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, kPI / 2_r);
  auto rotZ = Quaternion::FromAxisAngle(Real3{0_r, 0_r, 1_r}, kPI / 2_r);

  // Rotate about X
  p2 = TransformShape(TransformRT{rotX, Real3{}}, p);
  ExpectPlane(Real3{1_r, 0_r, 0_r}, 5_r, p2); // no change

  // Rotate about Y
  p2 = TransformShape(TransformRT{rotY, Real3{}}, p);
  ExpectPlane(Real3{0_r, 0_r, -1_r}, 5_r, p2); // now the plane (z == -5)

  // Rotate about Z
  p2 = TransformShape(TransformRT{rotZ, Real3{}}, p);
  ExpectPlane(Real3{0_r, 1_r, 0_r}, 5_r, p2); // now the plane (y == 5)

  // Translate in X
  p2 = TransformShape(TransformRT{Quaternion::Identity(), Real3{1_r, 0_r, 0_r}}, p);
  ExpectPlane(Real3{1_r, 0_r, 0_r}, 6_r, p2); // now the plane (x == 6)

  // Translate in Y
  p2 = TransformShape(TransformRT{Quaternion::Identity(), Real3{0_r, 1_r, 0_r}}, p);
  ExpectPlane(Real3{1_r, 0_r, 0_r}, 5_r, p2); // no change

  // Translate in Z
  p2 = TransformShape(TransformRT{Quaternion::Identity(), Real3{0_r, 0_r, 1_r}}, p);
  ExpectPlane(Real3{1_r, 0_r, 0_r}, 5_r, p2); // no change

  // Rotate about X and translate
  p2 = TransformShape(TransformRT{rotX, Real3{1_r, 2_r, 3_r}}, p);
  ExpectPlane(Real3{1_r, 0_r, 0_r}, 6_r, p2);

  // Rotate about Y and translate
  p2 = TransformShape(TransformRT{rotY, Real3{1_r, 2_r, 3_r}}, p);
  ExpectPlane(Real3{0_r, 0_r, -1_r}, 2_r, p2);

  // Rotate about Z and translate
  p2 = TransformShape(TransformRT{rotZ, Real3{1_r, 2_r, 3_r}}, p);
  ExpectPlane(Real3{0_r, 1_r, 0_r}, 7_r, p2);
}

TEST(Plane, HasOverlap_Aabb) {
  Plane p;
  Aabb aabb;
  EXPECT_TRUE(HasOverlap(p, aabb));

  // clang-format off
  aabb = {Real3{-1_r, -1_r, -1_r}, Real3{1_r, 1_r, 1_r}};

  // Test all six axis-aligned planes
  EXPECT_TRUE(HasOverlap(Plane(Real3(1_r, 0_r, 0_r), 1_r), aabb)); // +x face
  EXPECT_TRUE(HasOverlap(Plane(Real3(-1_r, 0_r, 0_r), 1_r), aabb)); // -x face
  EXPECT_TRUE(HasOverlap(Plane(Real3(0_r, 1_r, 0_r), 1_r), aabb)); // +y face
  EXPECT_TRUE(HasOverlap(Plane(Real3(0_r, -1_r, 0_r), 1_r), aabb)); // -y face
  EXPECT_TRUE(HasOverlap(Plane(Real3(0_r, 0_r, 1_r), 1_r), aabb)); // +z face
  EXPECT_TRUE(HasOverlap(Plane(Real3(0_r, 0_r, -1_r), 1_r), aabb)); // -z face

  // Test planes just outside each face
  EXPECT_FALSE(HasOverlap(Plane(Real3(1_r, 0_r, 0_r), -1.00001_r), aabb)); // outside +x face
  EXPECT_FALSE(HasOverlap(Plane(Real3(-1_r, 0_r, 0_r), -1.00001_r), aabb)); // outside -x face
  EXPECT_FALSE(HasOverlap(Plane(Real3(0_r, 1_r, 0_r), -1.00001_r), aabb)); // outside +y face
  EXPECT_FALSE(HasOverlap(Plane(Real3(0_r, -1_r, 0_r), -1.00001_r), aabb)); // outside -y face
  EXPECT_FALSE(HasOverlap(Plane(Real3(0_r, 0_r, 1_r), -1.00001_r), aabb)); // outside +z face
  EXPECT_FALSE(HasOverlap(Plane(Real3(0_r, 0_r, -1_r), -1.00001_r), aabb)); // outside -z face

  // Test planes just inside each face
  EXPECT_TRUE(HasOverlap(Plane(Real3(1_r, 0_r, 0_r), 1.00001_r), aabb)); // inside +x face
  EXPECT_TRUE(HasOverlap(Plane(Real3(-1_r, 0_r, 0_r), 1.00001_r), aabb)); // inside -x face
  EXPECT_TRUE(HasOverlap(Plane(Real3(0_r, 1_r, 0_r), 1.00001_r), aabb)); // inside +y face
  EXPECT_TRUE(HasOverlap(Plane(Real3(0_r, -1_r, 0_r), 1.00001_r), aabb)); // inside -y face
  EXPECT_TRUE(HasOverlap(Plane(Real3(0_r, 0_r, 1_r), 1.00001_r), aabb)); // inside +z face
  EXPECT_TRUE(HasOverlap(Plane(Real3(0_r, 0_r, -1_r), 1.00001_r), aabb)); // inside -z face

  // Diagonal plane
  Real3 norm = Normalize(Real3{1_r, -1_r, 1_r});
  real sqrt3 = Sqrt(3_r);
  EXPECT_TRUE(HasOverlap(Plane(norm, 0_r), aabb)); // plane bisects aabb
  EXPECT_TRUE(HasOverlap(Plane(norm, sqrt3), aabb)); // shared corner point, rest of aabb under plane
  EXPECT_TRUE(HasOverlap(Plane(norm, sqrt3 - 0.00001_r), aabb)); // aabb under plane
  EXPECT_TRUE(HasOverlap(Plane(norm, -sqrt3 + 0.00001_r), aabb)); // barely shares a corner, rest of aabb above plane
  EXPECT_FALSE(HasOverlap(Plane(norm, -sqrt3 - 0.00001_r), aabb)); // aabb above plane

  // Test all 8 corner diagonals
  Real3 normPPP = Normalize(Real3{1_r, 1_r, 1_r});
  Real3 normPPN = Normalize(Real3{1_r, 1_r, -1_r});
  Real3 normPNP = Normalize(Real3{1_r, -1_r, 1_r});
  Real3 normPNN = Normalize(Real3{1_r, -1_r, -1_r});
  Real3 normNPP = Normalize(Real3{-1_r, 1_r, 1_r});
  Real3 normNPN = Normalize(Real3{-1_r, 1_r, -1_r});
  Real3 normNNP = Normalize(Real3{-1_r, -1_r, 1_r});
  Real3 normNNN = Normalize(Real3{-1_r, -1_r, -1_r});

  EXPECT_TRUE(HasOverlap(Plane(normPPP, sqrt3), aabb)); // corner point
  EXPECT_TRUE(HasOverlap(Plane(normPPN, sqrt3), aabb)); // corner point
  EXPECT_TRUE(HasOverlap(Plane(normPNP, sqrt3), aabb)); // corner point
  EXPECT_TRUE(HasOverlap(Plane(normPNN, sqrt3), aabb)); // corner point
  EXPECT_TRUE(HasOverlap(Plane(normNPP, sqrt3), aabb)); // corner point
  EXPECT_TRUE(HasOverlap(Plane(normNPN, sqrt3), aabb)); // corner point
  EXPECT_TRUE(HasOverlap(Plane(normNNP, sqrt3), aabb)); // corner point
  EXPECT_TRUE(HasOverlap(Plane(normNNN, sqrt3), aabb)); // corner point

  // Test planes just outside each corner
  EXPECT_FALSE(HasOverlap(Plane(normPPP, -sqrt3 - 0.00001_r), aabb)); // outside corner
  EXPECT_FALSE(HasOverlap(Plane(normPPN, -sqrt3 - 0.00001_r), aabb)); // outside corner
  EXPECT_FALSE(HasOverlap(Plane(normPNP, -sqrt3 - 0.00001_r), aabb)); // outside corner
  EXPECT_FALSE(HasOverlap(Plane(normPNN, -sqrt3 - 0.00001_r), aabb)); // outside corner
  EXPECT_FALSE(HasOverlap(Plane(normNPP, -sqrt3 - 0.00001_r), aabb)); // outside corner
  EXPECT_FALSE(HasOverlap(Plane(normNPN, -sqrt3 - 0.00001_r), aabb)); // outside corner
  EXPECT_FALSE(HasOverlap(Plane(normNNP, -sqrt3 - 0.00001_r), aabb)); // outside corner
  EXPECT_FALSE(HasOverlap(Plane(normNNN, -sqrt3 - 0.00001_r), aabb)); // outside corner

  // Test with different sized AABBs
  aabb = {Real3{-2_r, -3_r, -4_r}, Real3{5_r, 6_r, 7_r}};
  EXPECT_TRUE(HasOverlap(Plane(Real3(1_r, 0_r, 0_r), 5_r), aabb)); // +x face
  EXPECT_TRUE(HasOverlap(Plane(Real3(-1_r, 0_r, 0_r), 2_r), aabb)); // -x face
  EXPECT_TRUE(HasOverlap(Plane(Real3(0_r, 1_r, 0_r), 6_r), aabb)); // +y face
  EXPECT_TRUE(HasOverlap(Plane(Real3(0_r, -1_r, 0_r), 3_r), aabb)); // -y face
  EXPECT_TRUE(HasOverlap(Plane(Real3(0_r, 0_r, 1_r), 7_r), aabb)); // +z face
  EXPECT_TRUE(HasOverlap(Plane(Real3(0_r, 0_r, -1_r), 4_r), aabb)); // -z face

  // Test with zero-volume AABB (point)
  aabb = {Real3{3_r, 4_r, 5_r}, Real3{3_r, 4_r, 5_r}};
  EXPECT_TRUE(HasOverlap(Plane(Real3(1_r, 0_r, 0_r), 3_r), aabb)); // plane contains point
  EXPECT_FALSE(HasOverlap(Plane(Real3(1_r, 0_r, 0_r), 2.99999_r), aabb)); // point above plane
  EXPECT_TRUE(HasOverlap(Plane(Real3(1_r, 0_r, 0_r), 3.00001_r), aabb)); // point below plane

  // Test with flat AABB (zero thickness in one dimension)
  aabb = {Real3{-1_r, 0_r, -1_r}, Real3{1_r, 0_r, 1_r}}; // flat in y
  EXPECT_TRUE(HasOverlap(Plane(Real3(0_r, 1_r, 0_r), 0_r), aabb)); // plane contains flat AABB
  EXPECT_FALSE(HasOverlap(Plane(Real3(0_r, 1_r, 0_r), -0.00001_r), aabb)); // AABB above plane
  EXPECT_TRUE(HasOverlap(Plane(Real3(0_r, 1_r, 0_r), 0.00001_r), aabb)); // AABB below plane
  EXPECT_TRUE(HasOverlap(Plane(Real3(1_r, 0_r, 0_r), 1_r), aabb)); // plane at edge of flat AABB
                                                               // clang-format on
}

TEST(Plane, ExpandShape) {
  // Everyting under the plane is considered to be part of its "volume".
  // Thus, expanding that value requires shifting the plane along its normal.
  Real3 normal{0_r, 1_r, 0_r};
  Plane p = ExpandShape(Plane{normal, 1_r}, 1_r);
  ExpectPlane(normal, 2_r, p);
  auto any = ExpandShape(AnyShape{p}, -1_r); // via AnyShape
  ExpectPlane(normal, 1_r, std::get<Plane>(any));
}

TEST(Plane, ContainsPoint) {
  Plane const kTestCases[] = {
      Plane{Real3{-1_r, 0_r, 0_r}, 0_r},
      Plane{Real3{0_r, 1_r, 0_r}, 0.5_r},
      Plane{Real3{0_r, 0_r, -1_r}, -0.25_r}};
  for (auto p : kTestCases) {
    // Point on plane
    EXPECT_TRUE(ContainsPoint(p, p.GetNormal() * p.GetDistanceFromOrigin()));

    // Point just below plane
    EXPECT_TRUE(ContainsPoint(p, p.GetNormal() * (p.GetDistanceFromOrigin() - 1e-5_r)));

    // Point just above plane
    EXPECT_FALSE(ContainsPoint(p, p.GetNormal() * (p.GetDistanceFromOrigin() + 1e-5_r)));
  }
}

TEST(Plane, HasOverlap) {
  // Same direction: always overlap
  EXPECT_TRUE(HasOverlap(Plane{Real3{0_r, 1_r, 0_r}, 5_r}, Plane{Real3{0_r, 1_r, 0_r}, 3_r}));

  // Anti-parallel, overlapping slab (half-spaces y<=5 and -y<=-3 → y>=3 → slab [3, 5])
  EXPECT_TRUE(HasOverlap(Plane{Real3{0_r, 1_r, 0_r}, 5_r}, Plane{Real3{0_r, -1_r, 0_r}, -3_r}));

  // Anti-parallel, touching (y<=0 and y>=0)
  EXPECT_TRUE(HasOverlap(Plane{Real3{0_r, 1_r, 0_r}, 0_r}, Plane{Real3{0_r, -1_r, 0_r}, 0_r}));

  // Anti-parallel, separated (y<=-1 and y>=2 → no overlap)
  EXPECT_FALSE(HasOverlap(Plane{Real3{0_r, 1_r, 0_r}, -1_r}, Plane{Real3{0_r, -1_r, 0_r}, -2_r}));

  // Oblique normals: always overlap
  EXPECT_TRUE(HasOverlap(Plane{Real3{1_r, 0_r, 0_r}, 5_r}, Plane{Real3{0_r, 1_r, 0_r}, 5_r}));
}

TEST(Plane, GetVolume) {
  EXPECT_EQ(std::numeric_limits<real>::infinity(), GetVolume(Plane{}));
  EXPECT_EQ(
      std::numeric_limits<real>::infinity(),
      GetVolume(Plane{Normalize(Real3{1_r, 2_r, 3_r}), 0.123_r}));
}

/**********************************************************************************************
  Sphere
*/

TEST(Sphere, Class) {
  // Default
  {
    Sphere s;
    ExpectSphere(Real3{}, 0_r, s);
  }

  // From Real3
  {
    Sphere s{Real3{1_r, 2_r, 3_r}, 4_r};
    ExpectSphere(Real3{1_r, 2_r, 3_r}, 4_r, s);
  }

  // From Vec4r
  {
    Sphere s{Vec4r(1_r, 2_r, 3_r, 911_r), 4_r}; // w component is discarded
    ExpectSphere(Real3{1_r, 2_r, 3_r}, 4_r, s);
  }
}

TEST(Sphere, NearEqual) {
  Sphere a{Real3{1_r, 2_r, 3_r}, 4_r};
  Sphere b{Real3{1_r, 2_r, 3_r}, 4_r};
  Sphere c{Real3{1.1_r, 2_r, 3_r}, 4_r};
  Sphere d{Real3{1.0_r, 2_r, 3_r}, 4.1_r};
  EXPECT_EQ(true, NearEqual(a, a, 0.2_r));
  EXPECT_EQ(true, NearEqual(a, b, 0.2_r));
  EXPECT_EQ(true, NearEqual(a, c, 0.2_r));
  EXPECT_EQ(true, NearEqual(a, d, 0.2_r));
  EXPECT_EQ(true, NearEqual(a, a, 0.02_r));
  EXPECT_EQ(true, NearEqual(a, b, 0.02_r));
  EXPECT_EQ(false, NearEqual(a, c, 0.02_r));
  EXPECT_EQ(false, NearEqual(a, d, 0.02_r));
}

TEST(Sphere, TransformShape) {
  Sphere s{Real3{1_r, 0_r, 0_r}, 5_r};
  Sphere s2;

  auto rotX = Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, kPI / 2_r);
  auto rotY = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, kPI / 2_r);
  auto rotZ = Quaternion::FromAxisAngle(Real3{0_r, 0_r, 1_r}, kPI / 2_r);

  // Rotate about X
  s2 = TransformShape(TransformRT{rotX, Real3{}}, s);
  ExpectSphere(Real3{1_r, 0_r, 0_r}, 5_r, s2); // no change

  // Rotate about Y
  s2 = TransformShape(TransformRT{rotY, Real3{}}, s);
  ExpectSphere(Real3{0_r, 0_r, -1_r}, 5_r, s2);

  // Rotate about Z
  s2 = TransformShape(TransformRT{rotZ, Real3{}}, s);
  ExpectSphere(Real3{0_r, 1_r, 0_r}, 5_r, s2);

  // Translate in X
  s2 = TransformShape(TransformRT{Quaternion::Identity(), Real3{1_r, 0_r, 0_r}}, s);
  ExpectSphere(Real3{2_r, 0_r, 0_r}, 5_r, s2);

  // Translate in Y
  s2 = TransformShape(TransformRT{Quaternion::Identity(), Real3{0_r, 1_r, 0_r}}, s);
  ExpectSphere(Real3{1_r, 1_r, 0_r}, 5_r, s2);

  // Translate in Z
  s2 = TransformShape(TransformRT{Quaternion::Identity(), Real3{0_r, 0_r, 1_r}}, s);
  ExpectSphere(Real3{1_r, 0_r, 1_r}, 5_r, s2);

  // Rotate about X and translate
  s2 = TransformShape(TransformRT{rotX, Real3{1_r, 2_r, 3_r}}, s);
  ExpectSphere(Real3{2_r, 2_r, 3_r}, 5_r, s2);

  // Rotate about Y and translate
  s2 = TransformShape(TransformRT{rotY, Real3{1_r, 2_r, 3_r}}, s);
  ExpectSphere(Real3{1_r, 2_r, 2_r}, 5_r, s2);

  // Rotate about Z and translate
  s2 = TransformShape(TransformRT{rotZ, Real3{1_r, 2_r, 3_r}}, s);
  ExpectSphere(Real3{1_r, 3_r, 3_r}, 5_r, s2);
}

TEST(Sphere, GetBoundingSphere) {
  // Sphere passthrough
  Sphere bounds = GetBoundingSphere(Sphere{Real3{1_r, 2_r, 3_r}, 123_r});
  ExpectSphere(Real3{1_r, 2_r, 3_r}, 123_r, bounds);

  // Planes are infinite
  bounds = GetBoundingSphere(Plane{Real3{0_r, 1_r, 0_r}, 123_r});
  EXPECT_NEAR_EQ(Real3(0_r, 123_r, 0_r), bounds.GetCenter());
  EXPECT_EQ(std::numeric_limits<real>::infinity(), bounds.GetRadius());
}

TEST(Sphere, GetAabb) {
  // Degenerate
  {
    auto s = Sphere{};
    EXPECT_NEAR_EQ(Aabb(), GetAabb(s));
  }

  // Nonzero radius
  {
    auto s = Sphere{Real3{}, 123_r};
    EXPECT_NEAR_EQ(Aabb(Real3(-123_r, -123_r, -123_r), Real3(123_r, 123_r, 123_r)), GetAabb(s));
  }

  // Nonzero center
  {
    auto c = Real3{1_r, -2_r, 3_r};
    auto s = Sphere{c, 0_r};
    EXPECT_NEAR_EQ(Aabb(c, c), GetAabb(s));
  }

  // Combined
  {
    auto c = Real3{1_r, -2_r, 3_r};
    auto s = Sphere{c, 4_r};
    EXPECT_NEAR_EQ(Aabb(Real3(-3_r, -6_r, -1_r), Real3(5_r, 2_r, 7_r)), GetAabb(s));
  }
}

TEST(Sphere, ContainsPoint) {
  // Degenerate
  {
    auto s = Sphere{};
    EXPECT_TRUE(ContainsPoint(s, Real3(0_r, 0_r, 0_r)));
    EXPECT_FALSE(ContainsPoint(s, Real3(1e-6_r, 0_r, 0_r)));
  }

  // Points exactly on the surface
  {
    auto s = Sphere{Real3{}, 1_r};
    EXPECT_TRUE(ContainsPoint(s, Real3(1_r, 0_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(s, Real3(-1_r, 0_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(s, Real3(0_r, 1_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(s, Real3(0_r, -1_r, 0_r)));
    EXPECT_TRUE(ContainsPoint(s, Real3(0_r, 0_r, 1_r)));
    EXPECT_TRUE(ContainsPoint(s, Real3(0_r, 0_r, -1_r)));
  }

  // Normal cases
  {
    auto s = Sphere{Real3{1_r, 2_r, 3_r}, 4_r};
    EXPECT_TRUE(ContainsPoint(s, Real3(1_r, 2_r, 3_r)));

    // barely inside
    EXPECT_TRUE(ContainsPoint(s, Real3(4.99999_r, 2_r, 3_r)));
    EXPECT_TRUE(ContainsPoint(s, Real3(-2.99999_r, 2_r, 3_r)));
    EXPECT_TRUE(ContainsPoint(s, Real3(1_r, 5.99999_r, 3_r)));
    EXPECT_TRUE(ContainsPoint(s, Real3(1_r, -1.99999_r, 3_r)));
    EXPECT_TRUE(ContainsPoint(s, Real3(1_r, 2_r, 6.99999_r)));
    EXPECT_TRUE(ContainsPoint(s, Real3(1_r, 2_r, -0.99999_r)));

    // barely outside
    EXPECT_FALSE(ContainsPoint(s, Real3(5.00001_r, 2_r, 3_r)));
    EXPECT_FALSE(ContainsPoint(s, Real3(-3.00001_r, 2_r, 3_r)));
    EXPECT_FALSE(ContainsPoint(s, Real3(1_r, 6.00001_r, 3_r)));
    EXPECT_FALSE(ContainsPoint(s, Real3(1_r, -2.00001_r, 3_r)));
    EXPECT_FALSE(ContainsPoint(s, Real3(1_r, 2_r, 7.00001_r)));
    EXPECT_FALSE(ContainsPoint(s, Real3(1_r, 2_r, -1.00001_r)));
  }
}

static void TestOverlapSphereAabb(
    std::function<bool(Sphere const&, Aabb const&)> const& hasOverlap,
    real eps = 0_r) {
  // degenerate sphere & aabb
  {
    auto s = Sphere{};
    EXPECT_TRUE(hasOverlap(s, Aabb()));
  }

  // degenerate sphere
  {
    auto aabb = Aabb{Real3{0_r, 0_r, 0_r}, Real3{1_r, 1_r, 1_r}};
    EXPECT_TRUE(hasOverlap(Sphere(Real3(eps, eps, eps), 0_r), aabb));
    EXPECT_TRUE(hasOverlap(Sphere(Real3(1_r - eps, eps, eps), 0_r), aabb));
    EXPECT_TRUE(hasOverlap(Sphere(Real3(eps, 1_r - eps, eps), 0_r), aabb));
    EXPECT_TRUE(hasOverlap(Sphere(Real3(1_r - eps, eps, 1_r - eps), 0_r), aabb));
    EXPECT_TRUE(hasOverlap(Sphere(Real3(1_r - eps, 1_r - eps, 1_r - eps), 0_r), aabb));
    EXPECT_FALSE(hasOverlap(Sphere(Real3(1.00001_r, 0_r, 0_r), 0_r), aabb));
    EXPECT_FALSE(hasOverlap(Sphere(Real3(0_r, 1.00001_r, 0_r), 0_r), aabb));
    EXPECT_FALSE(hasOverlap(Sphere(Real3(0_r, 0_r, 1.00001_r), 0_r), aabb));
  }

  // degenerate aabb
  {
    auto s = Sphere{Real3{}, 1_r + eps};
    EXPECT_TRUE(hasOverlap(s, Aabb()));
    EXPECT_TRUE(hasOverlap(s, Aabb(Real3(1_r, 0_r, 0_r), Real3(1_r, 0_r, 0_r)))); // point
    EXPECT_TRUE(hasOverlap(s, Aabb(Real3(0_r, 1_r, 0_r), Real3(0_r, 1_r, 0_r)))); // point
    EXPECT_TRUE(hasOverlap(s, Aabb(Real3(0_r, 0_r, 1_r), Real3(0_r, 0_r, 1_r)))); // point
    EXPECT_TRUE(hasOverlap(s, Aabb(Real3(0_r, -2_r, -2_r), Real3(0_r, 2_r, 2_r)))); // rectangle
    EXPECT_TRUE(hasOverlap(s, Aabb(Real3(1_r, -2_r, -2_r), Real3(1_r, 2_r, 2_r)))); // rectangle
    EXPECT_TRUE(hasOverlap(s, Aabb(Real3(-1_r, -2_r, -2_r), Real3(-1_r, 2_r, 2_r)))); // rectangle

    // clang-format off
    EXPECT_FALSE(hasOverlap(s, Aabb(Real3(1.00001_r, 0_r, 0_r), Real3(1.00001_r, 0_r, 0_r)))); // point
    EXPECT_FALSE(hasOverlap(s, Aabb(Real3(0_r, 1.00001_r, 0_r), Real3(0_r, 1.00001_r, 0_r)))); // point
    EXPECT_FALSE(hasOverlap(s, Aabb(Real3(0_r, 0_r, 1.00001_r), Real3(0_r, 0_r, 1.00001_r)))); // point
    EXPECT_FALSE(hasOverlap(s, Aabb(Real3(1.00001_r, -2_r, -2_r), Real3(1.00001_r, 2_r, 2_r)))); // rectangle
    EXPECT_FALSE(hasOverlap(s, Aabb(Real3(-1.00001_r, -2_r, -2_r), Real3(-1.00001_r, 2_r, 2_r)))); // rectangle
                                                                         // clang-format on
  }

  // normal cases
  {
    // clang-format off
    auto aabb = Aabb{Real3{-1_r, -2_r, 5_r}, Real3{1.1_r, 2.2_r, 10_r}};
    EXPECT_TRUE(hasOverlap(Sphere(Real3(0_r, 0_r, 7.5_r), 0.01_r), aabb)); // sphere inside near middle of aabb
    EXPECT_TRUE(hasOverlap(Sphere(Real3(1_r, 2_r, 9.9_r), 0.01_r), aabb)); // sphere inside near corner
    EXPECT_TRUE(hasOverlap(Sphere(Real3(1_r, 0_r, 7.5_r), 0.15_r), aabb)); // sphere overlapping face (more inside)
    EXPECT_TRUE(hasOverlap(Sphere(Real3(1.1_r, 0_r, 7.5_r), 0.15_r), aabb)); // sphere center on face (half inside)
    EXPECT_TRUE(hasOverlap(Sphere(Real3(1.2_r, 0_r, 7.5_r), 0.15_r), aabb)); // sphere overlapping face (more outside)
    // clang-format on

    // barely overlapping a face
    EXPECT_TRUE(hasOverlap(Sphere(Real3(1.19_r, 0_r, 7.5_r), 0.1_r), aabb)); // x face
    EXPECT_TRUE(hasOverlap(Sphere(Real3(0_r, 2.29_r, 7.5_r), 0.1_r), aabb)); // y face
    EXPECT_TRUE(hasOverlap(Sphere(Real3(0_r, 0_r, 10.09_r), 0.1_r), aabb)); // z face
    EXPECT_TRUE(hasOverlap(Sphere(Real3(-1.09_r, 0_r, 7.5_r), 0.1_r), aabb)); // -x face
    EXPECT_TRUE(hasOverlap(Sphere(Real3(0_r, -2.09_r, 7.5_r), 0.1_r), aabb)); // -y face
    EXPECT_TRUE(hasOverlap(Sphere(Real3(0_r, 0_r, 4.91_r), 0.1_r), aabb)); // -z face

    // almost overlapping a face
    EXPECT_FALSE(hasOverlap(Sphere(Real3(1.21_r, 0_r, 7.5_r), 0.1_r), aabb)); // x face
    EXPECT_FALSE(hasOverlap(Sphere(Real3(0_r, 2.31_r, 7.5_r), 0.1_r), aabb)); // y face
    EXPECT_FALSE(hasOverlap(Sphere(Real3(0_r, 0_r, 10.11_r), 0.1_r), aabb)); // z face
    EXPECT_FALSE(hasOverlap(Sphere(Real3(-1.11_r, 0_r, 7.5_r), 0.1_r), aabb)); // -x face
    EXPECT_FALSE(hasOverlap(Sphere(Real3(0_r, -2.11_r, 7.5_r), 0.1_r), aabb)); // -y face
    EXPECT_FALSE(hasOverlap(Sphere(Real3(0_r, 0_r, 4.89_r), 0.1_r), aabb)); // -z face

    // barely overlapping an edge
    aabb = Aabb{Real3{-1_r, -1_r, -1_r}, Real3{1_r, 1_r, 1_r}};
    EXPECT_TRUE(hasOverlap(Sphere(Real3(2.0_r, 2.0_r, 0.0_r), 1.42_r), aabb)); // +x+y edge
    EXPECT_TRUE(hasOverlap(Sphere(Real3(0.0_r, 2.0_r, 2.0_r), 1.42_r), aabb)); // +y+z edge
    EXPECT_TRUE(hasOverlap(Sphere(Real3(2.0_r, 0.0_r, 2.0_r), 1.42_r), aabb)); // +x+z edge
    EXPECT_TRUE(hasOverlap(Sphere(Real3(-2_r, -2_r, 0.0_r), 1.42_r), aabb)); // -x-y edge
    EXPECT_TRUE(hasOverlap(Sphere(Real3(0.0_r, -2_r, -2_r), 1.42_r), aabb)); // -y-z edge
    EXPECT_TRUE(hasOverlap(Sphere(Real3(-2_r, 0.0_r, -2_r), 1.42_r), aabb)); // -x-z edge

    // almost overlapping an edge
    EXPECT_FALSE(hasOverlap(Sphere(Real3(2.0_r, 2.0_r, 0.0_r), 1.41_r), aabb)); // +x+y edge
    EXPECT_FALSE(hasOverlap(Sphere(Real3(0.0_r, 2.0_r, 2.0_r), 1.41_r), aabb)); // +y+z edge
    EXPECT_FALSE(hasOverlap(Sphere(Real3(2.0_r, 0.0_r, 2.0_r), 1.41_r), aabb)); // +x+z edge
    EXPECT_FALSE(hasOverlap(Sphere(Real3(-2_r, -2_r, 0.0_r), 1.41_r), aabb)); // -x-y edge
    EXPECT_FALSE(hasOverlap(Sphere(Real3(0.0_r, -2_r, -2_r), 1.41_r), aabb)); // -y-z edge
    EXPECT_FALSE(hasOverlap(Sphere(Real3(-2_r, 0.0_r, -2_r), 1.41_r), aabb)); // -x-z edge
  }
}

TEST(Sphere, HasOverlap_Aabb) {
  TestOverlapSphereAabb([](Sphere const& s, Aabb const& aabb) { return HasOverlap(s, aabb); });
}

TEST(Sphere, HasOverlap_Obb) {
  // Test boxes with identiy rotation by reusing all the sphere-vs-aabb tests
  TestOverlapSphereAabb(
      [](Sphere const& s, Aabb const& aabb) { return HasOverlap(s, GetObb(aabb)); });

  // Repeat with rotate & translated shapes
  TestOverlapSphereAabb(
      [](Sphere const& s, Aabb const& aabb) {
        TransformRT transform{
            Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, kPI / 2_r), Real3{1_r, 2_r, 3_r}};
        auto box2 = TransformShape(transform, GetObb(aabb));
        auto sphere2 = TransformShape(transform, s);
        return HasOverlap(sphere2, box2);
      },
      1e1_r * std::numeric_limits<real>::epsilon());
}

// Pack one Sphere per SIMD lane into a BatchSphere<kBatchSize>.
template <int kBatchSize>
static BatchSphere<kBatchSize> MakeBatchSphere(std::array<Sphere, kBatchSize> const& spheres) {
  using V = BatchReal<kBatchSize>;
  real radii[V::kSize] = {};
  Real3 centers[V::kSize] = {};
  for (int i = 0; i < kBatchSize; ++i) {
    radii[i] = spheres[i].GetRadius();
    centers[i] = spheres[i].GetCenter();
  }
  BatchSphere<kBatchSize> batch{};
  batch.radius = Load<V>(radii);
  LoadTransposed<V::kSize>(&centers[0][0], batch.center);
  return batch;
}

// Verify that the batch HasOverlap overload matches its scalar counterpart for every sphere.
template <int kBatchSize, typename Shape>
static void TestOverlapShapeBatchSphere(Shape const& shape, Span<Sphere const> spheres) {
  using V = BatchSphere<kBatchSize>;
  using I = std::conditional_t<sizeof(real) == 4, int, int64_t>;
  ASSERT_FALSE(spheres.empty());
  bool sawOverlap = false;
  bool sawNoOverlap = false;
  for (size_t base = 0; base < spheres.size(); base += kBatchSize) {
    std::array<Sphere, kBatchSize> lanes = {};
    for (int i = 0; i < kBatchSize; ++i) {
      lanes[i] = spheres[Min(base + static_cast<size_t>(i), spheres.size() - 1)];
    }
    auto const batchSphere = MakeBatchSphere<kBatchSize>(lanes);
    auto const hasOverlap = ReinterpretCast<Simd<I, V::kSize>>(HasOverlap(shape, batchSphere));
    for (int i = 0; i < kBatchSize; ++i) {
      bool const expectedOverlap = HasOverlap(shape, lanes[i]);
      EXPECT_EQ(!!hasOverlap[i], expectedOverlap);
      sawOverlap |= expectedOverlap;
      sawNoOverlap |= !expectedOverlap;
    }
  }
  // The probe set must exercise both outcomes.
  EXPECT_TRUE(sawOverlap);
  EXPECT_TRUE(sawNoOverlap);
}

template <typename Shape>
static void TestOverlapShapeBatchSphere(Shape const& shape, Span<Sphere const> spheres) {
  TestOverlapShapeBatchSphere<4>(shape, spheres);
  TestOverlapShapeBatchSphere<8>(shape, spheres);
}

TEST(BatchSphere, HasOverlap_Sphere) {
  Sphere const a{Real3{0_r, 0_r, 0_r}, 1_r};
  real constexpr kProbeRadius = 0.5_r;
  real constexpr kTouch = 1_r + kProbeRadius; // center-to-center distance for exact contact
  Sphere const probes[] = {
      Sphere{Real3{0_r, 0_r, 0_r}, 0.1_r}, // fully inside a
      Sphere{Real3{0_r, 0_r, 0_r}, 5_r}, // enclosing a
      Sphere{Real3{1_r, 0_r, 0_r}, kProbeRadius}, // overlapping
      Sphere{Real3{kTouch, 0_r, 0_r}, kProbeRadius}, // exactly touching
      Sphere{Real3{std::nextafter(kTouch, 0_r), 0_r, 0_r}, kProbeRadius}, // just overlapping
      Sphere{Real3{std::nextafter(kTouch, kInf), 0_r, 0_r}, kProbeRadius}, // just separated
      Sphere{Real3{10_r, 0_r, 0_r}, kProbeRadius}, // far, separated
  };
  TestOverlapShapeBatchSphere(a, probes);
}

TEST(BatchSphere, HasOverlap_Aabb) {
  Aabb const aabb{Real3{-1_r, -2_r, -3_r}, Real3{1_r, 2_r, 3_r}};
  real constexpr kProbeRadius = 0.5_r;
  real constexpr kTouch = 1_r + kProbeRadius; // x at which a probe grazes the +x face
  Sphere const probes[] = {
      Sphere{Real3{0_r, 0_r, 0_r}, 0.5_r}, // fully inside
      Sphere{Real3{0_r, 0_r, 0_r}, 10_r}, // enclosing
      Sphere{Real3{1_r, 0_r, 0_r}, kProbeRadius}, // straddling the +x face
      Sphere{Real3{kTouch, 0_r, 0_r}, kProbeRadius}, // exactly touching the +x face
      Sphere{Real3{std::nextafter(kTouch, 0_r), 0_r, 0_r}, kProbeRadius}, // just overlapping
      Sphere{Real3{std::nextafter(kTouch, kInf), 0_r, 0_r}, kProbeRadius}, // just separated
      Sphere{Real3{10_r, 0_r, 0_r}, kProbeRadius}, // far, separated
  };
  TestOverlapShapeBatchSphere(aabb, probes);
}

TEST(BatchSphere, HasOverlap_Plane) {
  Plane const plane{Real3{0_r, 1_r, 0_r}, 2_r};
  real constexpr kProbeRadius = 0.5_r;
  real constexpr kTouch = 2_r + kProbeRadius; // center y at which the sphere just reaches the plane
  Sphere const probes[] = {
      Sphere{Real3{0_r, -5_r, 0_r}, 0.5_r}, // fully below the plane (deep in the half-space)
      Sphere{Real3{0_r, 2_r, 0_r}, 1_r}, // straddling the plane
      Sphere{Real3{0_r, kTouch, 0_r}, kProbeRadius}, // exactly touching from above
      Sphere{Real3{0_r, std::nextafter(kTouch, 0_r), 0_r}, kProbeRadius}, // just overlapping
      Sphere{Real3{0_r, std::nextafter(kTouch, kInf), 0_r}, kProbeRadius}, // just separated
      Sphere{Real3{0_r, 10_r, 0_r}, kProbeRadius}, // far above, separated
  };
  TestOverlapShapeBatchSphere(plane, probes);
}

TEST(BatchSphere, HasOverlap_Obb) {
  TransformRT const transform{
      Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, kPI / 5_r) *
          Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, kPI / 3_r),
      Real3{1_r, 2_r, 3_r}};
  Real3 const halfExtents{1_r, 2_r, 0.5_r};
  Obb const obb{transform, halfExtents};
  real constexpr kProbeRadius = 0.5_r;
  real const touchX = halfExtents[0] + kProbeRadius; // local x at which a probe grazes the +x face
  auto const world = [&](Real3 const& local) { return obb.GetTransform().TransformPoint(local); };

  Sphere const probes[] = {
      Sphere{obb.GetCenter(), 0.2_r}, // fully inside
      Sphere{obb.GetCenter(), 10_r}, // enclosing
      Sphere{world(Real3{halfExtents[0], 0_r, 0_r}), kProbeRadius}, // straddling the +x face
      Sphere{world(Real3{touchX - 0.01_r, 0_r, 0_r}), kProbeRadius}, // just overlapping
      Sphere{world(Real3{touchX + 0.01_r, 0_r, 0_r}), kProbeRadius}, // just separated
      Sphere{world(Real3{halfExtents[0] + 5_r, 0_r, 0_r}), kProbeRadius}, // far, separated
  };
  TestOverlapShapeBatchSphere(obb, probes);

  Obb const axisAlignedObb{TransformRT{Real3{1_r, 2_r, 3_r}}, halfExtents};
  Sphere const exactTouchProbes[] = {
      Sphere{Real3{2.5_r, 2_r, 3_r}, kProbeRadius}, // exactly touching the +x face
      Sphere{Real3{2.51_r, 2_r, 3_r}, kProbeRadius}, // separated
      Sphere{axisAlignedObb.GetCenter(), 0.2_r}, // fully inside
  };
  EXPECT_TRUE(HasOverlap(axisAlignedObb, exactTouchProbes[0]));
  TestOverlapShapeBatchSphere(axisAlignedObb, exactTouchProbes);
}

TEST(BatchSphere, HasOverlap_SdfBv) {
  // Build an SDF for a non-uniform box (AABB min=(0,0,0), max=(1,2,3)).
  auto const mesh =
      std::make_shared<TriangularMesh>(test::CreateMinimalTriMeshUnitCube(Real3{1_r, 2_r, 3_r}));
  GridSdfParams const params;
  GridSdf const sdf(mesh, params, test::ExpectOK{});

  // Use actor space as the points space, i.e. gridFromPoints == gridFromActor.
  SdfBv const sdfBv{
      .gridSdf = &sdf,
      .distanceThreshold = 0_r,
      .gridFromPointsT = sdf.GetGridFromActorTranspose()};

  Sphere const nearInside{Real3{0.9_r, 1_r, 1.5_r}, 0.05_r};
  Sphere const nearOutside{Real3{1.2_r, 1_r, 1.5_r}, 0.05_r};
  Sphere const probes[] = {
      Sphere{Real3{0.5_r, 1_r, 1.5_r}, 0.2_r}, // deep inside the box
      Sphere{Real3{0.5_r, 1_r, 1.5_r}, 5_r}, // enclosing the box
      Sphere{Real3{1.2_r, 1_r, 1.5_r}, 0.5_r}, // just outside the +x face, overlapping
      nearInside,
      nearOutside,
      Sphere{Real3{2_r, 1_r, 1.5_r}, 0.2_r}, // outside, separated
      Sphere{Real3{10_r, 1_r, 1.5_r}, 0.2_r}, // far, separated
  };

  EXPECT_TRUE(HasOverlap(sdfBv, nearInside));
  EXPECT_FALSE(HasOverlap(sdfBv, nearOutside));
  TestOverlapShapeBatchSphere(sdfBv, probes);

  SdfBv expandedSdfBv = sdfBv;
  expandedSdfBv.distanceThreshold = 0.25_r;
  EXPECT_TRUE(HasOverlap(expandedSdfBv, nearOutside));
  TestOverlapShapeBatchSphere(expandedSdfBv, probes);

  SdfBv contractedSdfBv = sdfBv;
  contractedSdfBv.distanceThreshold = -0.25_r;
  EXPECT_FALSE(HasOverlap(contractedSdfBv, nearInside));
  TestOverlapShapeBatchSphere(contractedSdfBv, probes);
}

TEST(Sphere, ExpandShape) {
  // From empty
  {
    Sphere shape;
    ExpectSphere(Real3{}, 0_r, shape);
    shape = ExpandShape(shape, 1_r);
    ExpectSphere(Real3{}, 1_r, shape);
  }

  // From non-empty
  {
    Real3 center{1_r, 2_r, 3_r};
    Sphere shape(center, 0.1_r);
    ExpectSphere(center, 0.1_r, shape);
    shape = ExpandShape(shape, 0_r);
    ExpectSphere(center, 0.1_r, shape);
    shape = ExpandShape(shape, 1_r);
    ExpectSphere(center, 1.1_r, shape);
    auto any = ExpandShape(AnyShape{shape}, -1_r); // via AnyShape
    ExpectSphere(center, 0.1_r, std::get<Sphere>(any));
  }
}

TEST(Sphere, EncloseShapes) {
  Real3 const kCenter{3_r, -2_r, 1_r};
  real constexpr kRadius = 2_r;
  Sphere sphere(kCenter, kRadius);

  auto test = [&](real shift, real scale) {
    Real3 resultCenter =
        kCenter + std::max(0_r, 0.5_r * (shift + scale - 1_r)) * kRadius * Real3{1_r, 0_r, 0_r};
    real resultRadius = std::max(1_r, 0.5_r * (shift + scale + 1_r)) * kRadius;
    Sphere other(kCenter + shift * kRadius * Real3{1_r, 0_r, 0_r}, scale * kRadius);
    Sphere result = EncloseShapes(sphere, other);
    EXPECT_NEAR_EQ(resultCenter, result.GetCenter());
    EXPECT_NEAR_EQ(resultRadius, result.GetRadius());
    result = EncloseShapes(other, sphere);
    EXPECT_NEAR_EQ(resultCenter, result.GetCenter());
    EXPECT_NEAR_EQ(resultRadius, result.GetRadius());
  };

  // Shift the test sphere by distances between 0 and 4 * radius.
  // Scale the test sphere by amounts between 0 and 1.
  auto lerp = [](real a, real b, real w) { return w * a + (1_r - w) * b; };
  int constexpr kNumShift = 41;
  real constexpr kMaxShift = 4_r;
  real constexpr kMinShift = 0_r;
  int constexpr kNumScale = 21;
  real constexpr kMaxScale = 1_r;
  real constexpr kMinScale = 0_r;
  for (int i = 0; i < kNumShift; i++) {
    real shiftLerp = static_cast<real>(i) / static_cast<real>(kNumShift - 1);
    real shift = lerp(kMaxShift, kMinShift, shiftLerp);
    for (int j = 0; j < kNumScale; j++) {
      real scaleLerp = static_cast<real>(j) / static_cast<real>(kNumScale - 1);
      real scale = lerp(kMaxScale, kMinScale, scaleLerp);
      test(shift, scale);
    }
  }
}

TEST(Sphere, GetVolume) {
  EXPECT_EQ(0_r, GetVolume(Sphere{})); // empty volume
  EXPECT_EQ(0_r, GetVolume(Sphere{Real3{1_r, 2_r, 3_r}, 0_r})); // empty volume
  EXPECT_NEAR_EQ(
      (4_r / 3_r) * kPI * 1.23_r * 1.23_r * 1.23_r,
      GetVolume(Sphere{Real3{1_r, 2_r, 3_r}, 1.23_r}));
}

/**********************************************************************************************
  AnyShape
*/

TEST(AnyShape, Class) {
  // Defaults to a degenerate sphere (AnyShape must always have a shape type)
  {
    AnyShape s;
    EXPECT_TRUE(!std::holds_alternative<Aabb>(s));
    EXPECT_TRUE(!std::holds_alternative<Obb>(s));
    EXPECT_TRUE(!std::holds_alternative<Plane>(s));
    EXPECT_TRUE(std::holds_alternative<Sphere>(s));
    EXPECT_TRUE(nullptr == std::get_if<Aabb>(&s));
    EXPECT_TRUE(nullptr == std::get_if<Obb>(&s));
    EXPECT_TRUE(nullptr == std::get_if<Plane>(&s));
    EXPECT_TRUE(nullptr != std::get_if<Sphere>(&s));
  }

  // From Plane
  {
    AnyShape s = Plane{Real3{1_r, 0_r, 0_r}, 123_r};
    EXPECT_TRUE(!std::holds_alternative<Aabb>(s));
    EXPECT_TRUE(!std::holds_alternative<Obb>(s));
    EXPECT_TRUE(std::holds_alternative<Plane>(s));
    EXPECT_TRUE(!std::holds_alternative<Sphere>(s));
    EXPECT_TRUE(nullptr == std::get_if<Aabb>(&s));
    EXPECT_TRUE(nullptr == std::get_if<Obb>(&s));
    EXPECT_TRUE(nullptr != std::get_if<Plane>(&s));
    EXPECT_TRUE(nullptr == std::get_if<Sphere>(&s));
    ExpectPlane(Real3{1_r, 0_r, 0_r}, 123_r, std::get<Plane>(s));
  }

  // From Sphere
  {
    AnyShape s = Sphere{Real3{1_r, 2_r, 3_r}, 123_r};
    EXPECT_TRUE(!std::holds_alternative<Aabb>(s));
    EXPECT_TRUE(!std::holds_alternative<Obb>(s));
    EXPECT_TRUE(!std::holds_alternative<Plane>(s));
    EXPECT_TRUE(std::holds_alternative<Sphere>(s));
    EXPECT_TRUE(nullptr == std::get_if<Aabb>(&s));
    EXPECT_TRUE(nullptr == std::get_if<Obb>(&s));
    EXPECT_TRUE(nullptr == std::get_if<Plane>(&s));
    EXPECT_TRUE(nullptr != std::get_if<Sphere>(&s));
    ExpectSphere(Real3{1_r, 2_r, 3_r}, 123_r, std::get<Sphere>(s));
  }

  // From Aabb
  Aabb aabb = Aabb{Real3{-1_r, -2_r, -3_r}, Real3{4_r, 5_r, 6_r}};
  {
    AnyShape s = aabb;
    EXPECT_TRUE(std::holds_alternative<Aabb>(s));
    EXPECT_TRUE(!std::holds_alternative<Obb>(s));
    EXPECT_TRUE(!std::holds_alternative<Plane>(s));
    EXPECT_TRUE(!std::holds_alternative<Sphere>(s));
    EXPECT_TRUE(nullptr != std::get_if<Aabb>(&s));
    EXPECT_TRUE(nullptr == std::get_if<Obb>(&s));
    EXPECT_TRUE(nullptr == std::get_if<Plane>(&s));
    EXPECT_TRUE(nullptr == std::get_if<Sphere>(&s));
    EXPECT_NEAR_EQ(aabb, std::get<Aabb>(s));
  }

  // From Obb
  Quaternion boxRot = Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, kPI / 2_r);
  Obb box = Obb{TransformRT{boxRot, Real3{1_r, 2_r, 3_r}}, Real3{4_r, 5_r, 6_r}};
  {
    AnyShape s = box;
    EXPECT_TRUE(!std::holds_alternative<Aabb>(s));
    EXPECT_TRUE(std::holds_alternative<Obb>(s));
    EXPECT_TRUE(!std::holds_alternative<Plane>(s));
    EXPECT_TRUE(!std::holds_alternative<Sphere>(s));
    EXPECT_TRUE(nullptr == std::get_if<Aabb>(&s));
    EXPECT_TRUE(nullptr != std::get_if<Obb>(&s));
    EXPECT_TRUE(nullptr == std::get_if<Plane>(&s));
    EXPECT_TRUE(nullptr == std::get_if<Sphere>(&s));
    EXPECT_NEAR_EQ(box, std::get<Obb>(s));
  }

  // Assignment
  {
    AnyShape s;
    s = Plane{Real3{1_r, 0_r, 0_r}, 123_r};
    ExpectPlane(Real3{1_r, 0_r, 0_r}, 123_r, std::get<Plane>(s));
    s = Sphere{Real3{1_r, 2_r, 3_r}, 123_r};
    ExpectSphere(Real3{1_r, 2_r, 3_r}, 123_r, std::get<Sphere>(s));
    s = box;
    EXPECT_NEAR_EQ(box, std::get<Obb>(s));
  }
}

TEST(AnyShape, GetVolume) {
  EXPECT_NEAR_EQ(48_r, GetVolume(AnyShape{Aabb{Real3{-1_r, -2_r, -3_r}, Real3{1_r, 2_r, 3_r}}}));
  EXPECT_NEAR_EQ(
      48_r, GetVolume(AnyShape{GetObb(Aabb{Real3{-1_r, -2_r, -3_r}, Real3{1_r, 2_r, 3_r}})}));
  EXPECT_EQ(std::numeric_limits<real>::infinity(), GetVolume(AnyShape{Plane{}}));
  EXPECT_NEAR_EQ(
      (4_r / 3_r) * kPI * 1.23_r * 1.23_r * 1.23_r,
      GetVolume(AnyShape{Sphere{Real3{1_r, 2_r, 3_r}, 1.23_r}}));
}

/**********************************************************************************************
  ClosestPt
*/

TEST(ClosestPt, PointSphere) {
  // clang-format off
  Real3 constexpr kTestPoints[] = {
      Real3{2_r, 2_r, 3_r}, Real3{4_r, 2_r, 3_r}, Real3{0_r, 2_r, 3_r}, Real3{-2_r, 2_r, 3_r},
      Real3{1_r, 3_r, 3_r}, Real3{1_r, 5_r, 3_r}, Real3{1_r, 1_r, 3_r}, Real3{1_r, -1_r, 3_r},
      Real3{1_r, 2_r, 4_r}, Real3{1_r, 2_r, 6_r}, Real3{1_r, 2_r, 2_r}, Real3{1_r, 2_r, 0_r}};
  Real3 constexpr kExpectedClosestPt[] = {
      Real3{2_r, 2_r, 3_r}, Real3{3_r, 2_r, 3_r}, Real3{0_r, 2_r, 3_r}, Real3{-1_r, 2_r, 3_r},
      Real3{1_r, 3_r, 3_r}, Real3{1_r, 4_r, 3_r}, Real3{1_r, 1_r, 3_r}, Real3{1_r, 0_r, 3_r},
      Real3{1_r, 2_r, 4_r}, Real3{1_r, 2_r, 5_r}, Real3{1_r, 2_r, 2_r}, Real3{1_r, 2_r, 1_r}};
  //clang-format on

  Sphere const sphere(Real3(1.0_r, 2.0_r, 3.0_r), 2.0_r);

  // Scalar version
  for (int i = 0; i < 12; ++i) {
    Real3 closestPt = ClosestPtPointShape(kTestPoints[i], sphere);
    EXPECT_NEAR_EQ(closestPt, kExpectedClosestPt[i]);
  }

  // SIMD version
  for (int i = 0; i < 12; ++i) {
    Vec4r closestPt = VClosestPtPointShape(ToSimd(kTestPoints[i]), sphere);
    EXPECT_NEAR_EQ(ToReal3(closestPt), kExpectedClosestPt[i]);
  }
}

TEST(ClosestPt, PointAabb) {
  // clang-format off
  Real3 constexpr kTestPoints[] = {
      Real3{2_r, 2_r, 3_r}, Real3{4_r, 2_r, 3_r}, Real3{0_r, 2_r, 3_r}, Real3{-2_r, 2_r, 3_r},
      Real3{1_r, 3_r, 3_r}, Real3{1_r, 5_r, 3_r}, Real3{1_r, 1_r, 3_r}, Real3{1_r, -1_r, 3_r},
      Real3{1_r, 2_r, 4_r}, Real3{1_r, 2_r, 6_r}, Real3{1_r, 2_r, 2_r}, Real3{1_r, 2_r, 0_r}};
  Real3 constexpr kExpectedClosestPt[] = {
      Real3{2_r, 2_r, 3_r}, Real3{3_r, 2_r, 3_r}, Real3{0_r, 2_r, 3_r}, Real3{-1_r, 2_r, 3_r},
      Real3{1_r, 3_r, 3_r}, Real3{1_r, 4_r, 3_r}, Real3{1_r, 1_r, 3_r}, Real3{1_r, 0_r, 3_r},
      Real3{1_r, 2_r, 4_r}, Real3{1_r, 2_r, 5_r}, Real3{1_r, 2_r, 2_r}, Real3{1_r, 2_r, 1_r}};
  Real3 constexpr kExpectedParam[] = {
      Real3{0.75_r, 0.50_r, 0.50_r}, Real3{1.00_r, 0.50_r, 0.50_r}, Real3{0.25_r, 0.50_r, 0.50_r},
      Real3{0.00_r, 0.50_r, 0.50_r}, Real3{0.50_r, 0.75_r, 0.50_r}, Real3{0.50_r, 1.00_r, 0.50_r},
      Real3{0.50_r, 0.25_r, 0.50_r}, Real3{0.50_r, 0.00_r, 0.50_r}, Real3{0.50_r, 0.50_r, 0.75_r},
      Real3{0.50_r, 0.50_r, 1.00_r}, Real3{0.50_r, 0.50_r, 0.25_r}, Real3{0.50_r, 0.50_r, 0.00_r}};
  // clang-format on

  Aabb const aabb{Real3(-1_r, 0_r, 1_r), Real3(3_r, 4_r, 5_r)};

  // Scalar version
  for (int i = 0; i < 12; ++i) {
    Real3 param;
    Real3 closestPt = ClosestPtPointShape(kTestPoints[i], aabb, &param);
    EXPECT_NEAR_EQ(closestPt, kExpectedClosestPt[i]);
    EXPECT_NEAR_EQ(param, kExpectedParam[i]);
  }

  // SIMD version
  for (int i = 0; i < 12; ++i) {
    Vec4r param;
    Vec4r closestPt = VClosestPtPointShape(ToSimd(kTestPoints[i]), aabb, &param);
    EXPECT_NEAR_EQ(ToReal3(closestPt), kExpectedClosestPt[i]);
    EXPECT_NEAR_EQ(ToReal3(param), kExpectedParam[i]);
  }
}

TEST(ClosestPt, PointObb) {
  // clang-format off
  Real3 constexpr kTestPoints[] = {
      Real3{1.836516261100769_r, 2.5245190262794495_r, 2.8415063619613647_r},
      Real3{3.509548783302307_r, 3.5735570788383484_r, 2.5245190858840942_r},
      Real3{0.16348373889923096_r, 1.4754809737205505_r, 3.1584936380386353_r},
      Real3{-1.5095487833023071_r, 0.4264429211616516_r, 3.4754809141159058_r},
      Real3{0.7758561372756958_r, 2.5915063619613647_r, 3.7745190262794495_r},
      Real3{0.3275684118270874_r, 3.7745190858840942_r, 5.323557078838348_r},
      Real3{1.2241438627243042_r, 1.4084936380386353_r, 2.2254809737205505_r},
      Real3{1.6724315881729126_r, 0.22548091411590576_r, 0.6764429211616516_r},
      Real3{1.5_r, 1.3876276016235352_r, 3.612372398376465_r},
      Real3{2.5_r, 0.16288280487060547_r, 4.8371171951293945_r},
      Real3{0.5_r, 2.612372398376465_r, 2.387627601623535_r},
      Real3{-0.5_r, 3.8371171951293945_r, 1.1628828048706055_r}};
  Real3 constexpr kExpectedClosestPt[] = {
      Real3{1.836516261100769_r, 2.5245190262794495_r, 2.8415063619613647_r},
      Real3{2.673032522201538_r, 3.049038052558899_r, 2.6830127239227295_r},
      Real3{0.16348373889923096_r, 1.4754809737205505_r, 3.1584936380386353_r},
      Real3{-0.6730325222015381_r, 0.9509619474411011_r, 3.3169872760772705_r},
      Real3{0.7758561372756958_r, 2.5915063619613647_r, 3.7745190262794495_r},
      Real3{0.5517122745513916_r, 3.1830127239227295_r, 4.549038052558899_r},
      Real3{1.2241438627243042_r, 1.4084936380386353_r, 2.2254809737205505_r},
      Real3{1.4482877254486084_r, 0.8169872760772705_r, 1.450961947441101_r},
      Real3{1.5_r, 1.3876276016235352_r, 3.612372398376465_r},
      Real3{2.0_r, 0.7752552032470703_r, 4.22474479675293_r},
      Real3{0.5_r, 2.612372398376465_r, 2.387627601623535_r},
      Real3{0.0_r, 3.2247447967529297_r, 1.7752552032470703_r}};
  Real3 constexpr kExpectedParam[] = {
      Real3{0.75_r, 0.50_r, 0.50_r},
      Real3{1.00_r, 0.50_r, 0.50_r},
      Real3{0.25_r, 0.50_r, 0.50_r},
      Real3{0.00_r, 0.50_r, 0.50_r},
      Real3{0.50_r, 0.75_r, 0.50_r},
      Real3{0.50_r, 1.00_r, 0.50_r},
      Real3{0.50_r, 0.25_r, 0.50_r},
      Real3{0.50_r, 0.00_r, 0.50_r},
      Real3{0.50_r, 0.50_r, 0.75_r},
      Real3{0.50_r, 0.50_r, 1.00_r},
      Real3{0.50_r, 0.50_r, 0.25_r},
      Real3{0.50_r, 0.50_r, 0.00_r}};
  Matrix3x3r constexpr R = {
      Real3{0.83651626_r, -0.22414386_r, 0.5_r},
      Real3{0.524519_r, 0.59150636_r, -0.6123724_r},
      Real3{-0.15849364_r, 0.774519_r, 0.6123724_r}};
  // clang-format on

  Obb const oobb{MatrixTransformRT{R, Real3{1_r, 2_r, 3_r}}, Real3{2_r, 2_r, 2_r}};

  // Scalar version
  for (int i = 0; i < 12; ++i) {
    Real3 param;
    Real3 closestPt = ClosestPtPointShape(kTestPoints[i], oobb, &param);
    EXPECT_NEAR_EQ(closestPt, kExpectedClosestPt[i]);
    EXPECT_NEAR_EQ(param, kExpectedParam[i]);
  }

  // SIMD version
  for (int i = 0; i < 12; ++i) {
    Vec4r param;
    Vec4r closestPt = VClosestPtPointShape(ToSimd(kTestPoints[i]), oobb, &param);
    EXPECT_NEAR_EQ(ToReal3(closestPt), kExpectedClosestPt[i]);
    EXPECT_NEAR_EQ(ToReal3(param), kExpectedParam[i]);
  }
}

/**********************************************************************************************
  Distance
*/

TEST(Distances, DistancePointSphere) {
  // clang-format off
  Real3 constexpr kTestPoints[] = {
      Real3{2_r, 2_r, 3_r}, Real3{4_r, 2_r, 3_r}, Real3{0_r, 2_r, 3_r}, Real3{-2_r, 2_r, 3_r},
      Real3{1_r, 3_r, 3_r}, Real3{1_r, 5_r, 3_r}, Real3{1_r, 1_r, 3_r}, Real3{1_r, -1_r, 3_r},
      Real3{1_r, 2_r, 4_r}, Real3{1_r, 2_r, 6_r}, Real3{1_r, 2_r, 2_r}, Real3{1_r, 2_r, 0_r}};
  real constexpr kExpectedDistances[] = {
      -1.0_r, 1.0_r, -1.0_r, 1.0_r, -1.0_r, 1.0_r,
      -1.0_r, 1.0_r, -1.0_r, 1.0_r, -1.0_r, 1.0_r
  };
  //clang-format on

  Sphere const sphere(Real3(1.0_r, 2.0_r, 3.0_r), 2.0_r);

  for (int i = 0; i < 12; ++i) {
    real distance = Get0(VDistancePointShape(ToSimd(kTestPoints[i]), sphere));
    EXPECT_NEAR_EQ(distance, kExpectedDistances[i]);

    real distanceSqr = Get0(VDistancePointShapeSqr(ToSimd(kTestPoints[i]), sphere));
    EXPECT_NEAR_EQ(SignedSqr(distance), distanceSqr);
  }
}

TEST(Distances, DistancePointAabb) {
  // clang-format off
  Real3 constexpr kTestPoints[] = {
      Real3{2_r, 2_r, 3_r}, Real3{4_r, 2_r, 3_r}, Real3{0_r, 2_r, 3_r}, Real3{-2_r, 2_r, 3_r},
      Real3{1_r, 3_r, 3_r}, Real3{1_r, 5_r, 3_r}, Real3{1_r, 1_r, 3_r}, Real3{1_r, -1_r, 3_r},
      Real3{1_r, 2_r, 4_r}, Real3{1_r, 2_r, 6_r}, Real3{1_r, 2_r, 2_r}, Real3{1_r, 2_r, 0_r}};
  real constexpr kExpectedDistances[] = {
      -1.0_r, 1.0_r, -1.0_r, 1.0_r, -1.0_r, 1.0_r,
      -1.0_r, 1.0_r, -1.0_r, 1.0_r, -1.0_r, 1.0_r
  };
  Real3 constexpr kExpectedParam[] = {
      Real3{0.75_r, 0.50_r, 0.50_r}, Real3{1.00_r, 0.50_r, 0.50_r}, Real3{0.25_r, 0.50_r, 0.50_r},
      Real3{0.00_r, 0.50_r, 0.50_r}, Real3{0.50_r, 0.75_r, 0.50_r}, Real3{0.50_r, 1.00_r, 0.50_r},
      Real3{0.50_r, 0.25_r, 0.50_r}, Real3{0.50_r, 0.00_r, 0.50_r}, Real3{0.50_r, 0.50_r, 0.75_r},
      Real3{0.50_r, 0.50_r, 1.00_r}, Real3{0.50_r, 0.50_r, 0.25_r}, Real3{0.50_r, 0.50_r, 0.00_r}};
  // clang-format on

  Aabb const aabb{Real3(-1_r, 0_r, 1_r), Real3(3_r, 4_r, 5_r)};

  for (int i = 0; i < 12; ++i) {
    Vec4r param;
    Vec4r distanceSqr = VDistancePointShapeSqr(ToSimd(kTestPoints[i]), aabb, &param);
    EXPECT_NEAR_EQ(SignedSqr(kExpectedDistances[i]), Get0(distanceSqr));
    EXPECT_NEAR_EQ(ToReal3(param), kExpectedParam[i]);
  }
}

TEST(Distances, DistancePointObb) {
  // clang-format off
  Real3 constexpr kTestPoints[] = {
      Real3{1.836516261100769_r, 2.5245190262794495_r, 2.8415063619613647_r},
      Real3{3.509548783302307_r, 3.5735570788383484_r, 2.5245190858840942_r},
      Real3{0.16348373889923096_r, 1.4754809737205505_r, 3.1584936380386353_r},
      Real3{-1.5095487833023071_r, 0.4264429211616516_r, 3.4754809141159058_r},
      Real3{0.7758561372756958_r, 2.5915063619613647_r, 3.7745190262794495_r},
      Real3{0.3275684118270874_r, 3.7745190858840942_r, 5.323557078838348_r},
      Real3{1.2241438627243042_r, 1.4084936380386353_r, 2.2254809737205505_r},
      Real3{1.6724315881729126_r, 0.22548091411590576_r, 0.6764429211616516_r},
      Real3{1.5_r, 1.3876276016235352_r, 3.612372398376465_r},
      Real3{2.5_r, 0.16288280487060547_r, 4.8371171951293945_r},
      Real3{0.5_r, 2.612372398376465_r, 2.387627601623535_r},
      Real3{-0.5_r, 3.8371171951293945_r, 1.1628828048706055_r}};
  real constexpr kExpectedDistances[] = {
      -1.0_r, 1.0_r, -1.0_r, 1.0_r, -1.0_r, 1.0_r,
      -1.0_r, 1.0_r, -1.0_r, 1.0_r, -1.0_r, 1.0_r
  };
  Real3 constexpr kExpectedParam[] = {
      Real3{0.75_r, 0.50_r, 0.50_r}, Real3{1.00_r, 0.50_r, 0.50_r}, Real3{0.25_r, 0.50_r, 0.50_r},
      Real3{0.00_r, 0.50_r, 0.50_r}, Real3{0.50_r, 0.75_r, 0.50_r}, Real3{0.50_r, 1.00_r, 0.50_r},
      Real3{0.50_r, 0.25_r, 0.50_r}, Real3{0.50_r, 0.00_r, 0.50_r}, Real3{0.50_r, 0.50_r, 0.75_r},
      Real3{0.50_r, 0.50_r, 1.00_r}, Real3{0.50_r, 0.50_r, 0.25_r}, Real3{0.50_r, 0.50_r, 0.00_r}};
  Matrix3x3r constexpr R = {
      Real3{0.83651626_r, -0.22414386_r, 0.5_r},
      Real3{0.524519_r, 0.59150636_r, -0.6123724_r},
      Real3{-0.15849364_r, 0.774519_r, 0.6123724_r}};
  // clang-format on

  Obb const oobb{MatrixTransformRT{R, Real3{1_r, 2_r, 3_r}}, Real3{2_r, 2_r, 2_r}};

  for (int i = 0; i < 12; ++i) {
    Vec4r param;
    Vec4r distanceSqr = VDistancePointShapeSqr(ToSimd(kTestPoints[i]), oobb, &param);
    EXPECT_NEAR_EQ(SignedSqr(kExpectedDistances[i]), Get0(distanceSqr));
    EXPECT_NEAR_EQ(ToReal3(param), kExpectedParam[i]);
  }
}

TEST(Distances, VDistancePointSegmentSqr) {
  real constexpr kTol = 1e-6_r;
  real constexpr kTolSqr = Sqr(kTol);
  real eps = 1e-4_r;
  Vec4r zero = 0_r;
  Vec4r ones = 1_r;
  Vec4r half = ones * 0.5_r;
  Vec4r outp{1_r, 2_r, 3_r};
  Vec4r ortho = Normalize<3>(Cross3(outp, ones));
  Vec4r epsv = ones * eps;
  Vec4r noutp = -outp;
  Vec4r nepsv = -epsv;
  Vec4r par;

  //---

  // Touching A
  EXPECT_NEAR_RTOL(Get0(VDistancePointSegmentSqr(zero, zero, ones, &par)), 0_r, kTolSqr);
  EXPECT_NEAR_RTOL(Get0(par), 0_r, kTol);

  // Touching B
  EXPECT_NEAR_RTOL(Get0(VDistancePointSegmentSqr(ones, zero, ones, &par)), 0_r, kTolSqr);
  EXPECT_NEAR_RTOL(Get0(par), 1_r, kTol);

  //---

  // Outside region A
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointSegmentSqr(noutp, zero, ones, &par)),
      Get0(VNormSqr<3>(noutp - zero)),
      kTol);
  EXPECT_NEAR_RTOL(Get0(par), 0_r, kTol);

  // Outside region B
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointSegmentSqr(outp, zero, ones, &par)), Get0(VNormSqr<3>(outp - ones)), kTol);
  EXPECT_NEAR_RTOL(Get0(par), 1_r, kTol);

  //---

  // Barely touching A (inside)
  EXPECT_NEAR_RTOL(Get0(VDistancePointSegmentSqr(epsv, zero, ones, &par)), 0_r, kTolSqr);
  EXPECT_NEAR_RTOL(Get0(par), Get0(VNorm<3>(epsv) / VNorm<3>(ones)), kTol);

  // Barely touching B (inside)
  EXPECT_NEAR_RTOL(Get0(VDistancePointSegmentSqr(ones - epsv, zero, ones, &par)), 0_r, kTolSqr);
  EXPECT_NEAR_RTOL(Get0(par), Get0(VNorm<3>(ones - epsv) / VNorm<3>(ones)), kTol);

  //---

  // Barely touching A (outside)
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointSegmentSqr(zero - epsv, zero, ones, &par)),
      Get0(VNormSqr<3>(nepsv)),
      kTol);
  EXPECT_NEAR_RTOL(Get0(par), 0_r, kTol);

  // Barely touching B (outside)
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointSegmentSqr(ones + epsv, zero, ones, &par)), Get0(VNormSqr<3>(epsv)), kTol);
  EXPECT_NEAR_RTOL(Get0(par), 1_r, kTol);

  //---

  // Far projection close to A
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointSegmentSqr(epsv + (ortho * 2_r), zero, ones, &par)), 4_r, kTol);
  EXPECT_NEAR_RTOL(Get0(par), Get0(VNorm<3>(epsv) / VNorm<3>(ones)), kTol);

  // Far projection close to B
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointSegmentSqr(ones - epsv + (ortho * 2_r), zero, ones, &par)), 4_r, kTol);
  EXPECT_NEAR_RTOL(Get0(par), Get0(VNorm<3>(ones - epsv) / VNorm<3>(ones)), kTol);

  // Far mid-edge projection
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointSegmentSqr(half + (ortho * 2_r), zero, ones, &par)), 4_r, kTol);
  EXPECT_NEAR_RTOL(Get0(par), Get0(VNorm<3>(half) / VNorm<3>(ones)), kTol);
}

TEST(Distances, VDistancePointTriangle) {
  real tol = 1e-6_r;
  real eps = 1e-4_r;
  Vec4r A = {1_r, 0_r, 0_r};
  Vec4r B = {0_r, 2_r, 0_r};
  Vec4r C = {0_r, 0_r, 3_r};
  Vec4r AB = B - A;
  Vec4r BC = C - B;
  Vec4r CA = A - C;
  Vec4r parA = {1.0_r, 0_r, 0_r};
  Vec4r parB = {0_r, 1.0_r, 0_r};
  Vec4r parC = {0_r, 0_r, 1.0_r};
  Vec4r parAB = {0.5_r, 0.5_r, 0_r};
  Vec4r parBC = {0_r, 0.5_r, 0.5_r};
  Vec4r parCA = {0.5_r, 0_r, 0.5_r};
  Vec4r parM = {0.25_r, 0.25_r, 0.5_r};
  Vec4r M = (A * Get<0>(parM)) + (B * Get<1>(parM)) + (C * Get<2>(parM));
  Vec4r normal = Normalize<3>(Cross3(B - A, C - A));
  Vec4r ABout = Normalize<3>(Cross3(B - A, normal));
  Vec4r BCout = Normalize<3>(Cross3(C - B, normal));
  Vec4r CAout = Normalize<3>(Cross3(A - C, normal));
  Vec4r P;

  VDistanceSignParams params;
  params.computeSign = false;

  Vec4r par;

  //---

  // Touching A
  EXPECT_NEAR_RTOL(Get0(VDistancePointTriangle(A, A, B, C, params, &par)), 0_r, tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parA)), 0_r, tol);

  // Touching B
  EXPECT_NEAR_RTOL(Get0(VDistancePointTriangle(B, A, B, C, params, &par)), 0_r, tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parB)), 0_r, tol);

  // Touching C
  EXPECT_NEAR_RTOL(Get0(VDistancePointTriangle(C, A, B, C, params, &par)), 0_r, tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parC)), 0_r, tol);

  //---

  // Outside region A
  P = A + (ABout * 2_r) + (CAout * 2_r) + (normal * 3_r);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)), Get0(VNorm<3>(P - A)), tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parA)), 0_r, tol);

  // Outside region B
  P = B + (ABout * 2_r) + (BCout * 2_r) + (normal * 3_r);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)), Get0(VNorm<3>(P - B)), tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parB)), 0_r, tol);

  // Outside region C
  P = C + (CAout * 2_r) + (BCout * 2_r) + (normal * 3_r);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)), Get0(VNorm<3>(P - C)), tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parC)), 0_r, tol);

  //---

  // Barely touching A (outside)
  P = A + (ABout * eps) + (CAout * eps);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)), Get0(VNorm<3>(P - A)), tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parA)), 0_r, tol);

  // Barely touching B (outside)
  P = B + (ABout * eps) + (BCout * eps);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)), Get0(VNorm<3>(P - B)), tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parB)), 0_r, tol);

  // Barely touching C (outside)
  P = C + (CAout * eps) + (BCout * eps);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)), Get0(VNorm<3>(P - C)), tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parC)), 0_r, tol);

  //---

  // Barely touching A (inside)
  P = A - (ABout * eps) - (CAout * eps);
  EXPECT_NEAR_RTOL(Get0(VDistancePointTriangle(P, A, B, C, params, &par)), 0_r, tol);

  // Barely touching B (inside)
  P = B - (ABout * eps) - (BCout * eps);
  EXPECT_NEAR_RTOL(Get0(VDistancePointTriangle(P, A, B, C, params, &par)), 0_r, tol);

  // Barely touching C (inside)
  P = C - (CAout * eps) - (BCout * eps);
  EXPECT_NEAR_RTOL(Get0(VDistancePointTriangle(P, A, B, C, params, &par)), 0_r, tol);

  //---

  // Outside region AB
  P = A + (AB * 0.5_r) + (ABout * 2_r) - (normal * 3_r);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)),
      Get0(VNorm<3>(P - A - (AB * 0.5_r))),
      tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parAB)), 0_r, tol);

  // Outside region BC
  P = B + (BC * 0.5_r) + (BCout * 2_r) - (normal * 3_r);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)),
      Get0(VNorm<3>(P - B - (BC * 0.5_r))),
      tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parBC)), 0_r, tol);

  // Outside region CA
  P = C + (CA * 0.5_r) + (CAout * 2_r) - (normal * 3_r);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)),
      Get0(VNorm<3>(P - C - (CA * 0.5_r))),
      tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parCA)), 0_r, tol);

  //---

  // Regular projection

  P = M + (normal * 2_r);
  EXPECT_NEAR_RTOL(Get0(VDistancePointTriangle(P, A, B, C, params, &par)), 2_r, tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parM)), 0_r, tol);
}

TEST(Distances, VDistancePointTriangle_Obtuse) {
  real tol = 1e-6_r;
  real eps = 1e-4_r;
  // Define a triangle with small aspect ratio
  Vec4r A = {1_r, 0_r, 0_r};
  Vec4r B = {0.5_r, 0.25_r, 1.5_r};
  Vec4r C = {0_r, 0_r, 3_r};
  Vec4r AB = B - A;
  Vec4r BC = C - B;
  Vec4r CA = A - C;
  Vec4r parA = {1.0_r, 0_r, 0_r};
  Vec4r parB = {0_r, 1.0_r, 0_r};
  Vec4r parC = {0_r, 0_r, 1.0_r};
  Vec4r parAB = {0.5_r, 0.5_r, 0_r};
  Vec4r parBC = {0_r, 0.5_r, 0.5_r};
  Vec4r parCA = {0.5_r, 0_r, 0.5_r};
  Vec4r parM = {0.25_r, 0.25_r, 0.5_r};
  Vec4r M = (A * Get<0>(parM)) + (B * Get<1>(parM)) + (C * Get<2>(parM));
  Vec4r normal = Normalize<3>(Cross3(B - A, C - A));
  Vec4r ABout = Normalize<3>(Cross3(B - A, normal));
  Vec4r BCout = Normalize<3>(Cross3(C - B, normal));
  Vec4r CAout = Normalize<3>(Cross3(A - C, normal));
  Vec4r P;

  VDistanceSignParams params;
  params.computeSign = false;

  Vec4r par;

  //---

  // Touching A
  EXPECT_NEAR_RTOL(Get0(VDistancePointTriangle(A, A, B, C, params, &par)), 0_r, tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parA)), 0_r, tol);

  // Touching B
  EXPECT_NEAR_RTOL(Get0(VDistancePointTriangle(B, A, B, C, params, &par)), 0_r, tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parB)), 0_r, tol);

  // Touching C
  EXPECT_NEAR_RTOL(Get0(VDistancePointTriangle(C, A, B, C, params, &par)), 0_r, tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parC)), 0_r, tol);

  //---

  // Outside region A
  P = A + (ABout * 2_r) + (CAout * 2_r) + (normal * 3_r);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)), Get0(VNorm<3>(P - A)), tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parA)), 0_r, tol);

  // Outside region B
  P = B + (ABout * 2_r) + (BCout * 2_r) + (normal * 3_r);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)), Get0(VNorm<3>(P - B)), tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parB)), 0_r, tol);

  // Outside region C
  P = C + (CAout * 2_r) + (BCout * 2_r) + (normal * 3_r);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)), Get0(VNorm<3>(P - C)), tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parC)), 0_r, tol);

  //---

  // Barely touching A (outside)
  P = A + (ABout * eps) + (CAout * eps);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)), Get0(VNorm<3>(P - A)), tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parA)), 0_r, tol);

  // Barely touching B (outside)
  P = B + (ABout * eps) + (BCout * eps);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)), Get0(VNorm<3>(P - B)), tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parB)), 0_r, tol);

  // Barely touching C (outside)
  P = C + (CAout * eps) + (BCout * eps);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)), Get0(VNorm<3>(P - C)), tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parC)), 0_r, tol);

  //---

  // Barely touching A (inside)
  P = A - (ABout * eps) - (CAout * eps);
  EXPECT_NEAR_RTOL(Get0(VDistancePointTriangle(P, A, B, C, params, &par)), 0_r, tol);

  // Barely touching B (inside)
  P = B - (ABout * eps) - (BCout * eps);
  EXPECT_NEAR_RTOL(Get0(VDistancePointTriangle(P, A, B, C, params, &par)), 0_r, tol);

  // Barely touching C (inside)
  P = C - (CAout * eps) - (BCout * eps);
  EXPECT_NEAR_RTOL(Get0(VDistancePointTriangle(P, A, B, C, params, &par)), 0_r, tol);

  //---

  // Outside region AB
  P = A + (AB * 0.5_r) + (ABout * 2_r) - (normal * 3_r);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)),
      Get0(VNorm<3>(P - A - (AB * 0.5_r))),
      tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parAB)), 0_r, tol);

  // Outside region BC
  P = B + (BC * 0.5_r) + (BCout * 2_r) - (normal * 3_r);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)),
      Get0(VNorm<3>(P - B - (BC * 0.5_r))),
      tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parBC)), 0_r, tol);

  // Outside region CA
  P = C + (CA * 0.5_r) + (CAout * 2_r) - (normal * 3_r);
  EXPECT_NEAR_RTOL(
      Get0(VDistancePointTriangle(P, A, B, C, params, &par)),
      Get0(VNorm<3>(P - C - (CA * 0.5_r))),
      tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parCA)), 0_r, tol);

  //---

  // Regular projection

  P = M + (normal * 2_r);
  EXPECT_NEAR_RTOL(Get0(VDistancePointTriangle(P, A, B, C, params, &par)), 2_r, tol);
  EXPECT_NEAR_RTOL(Get0(VNorm<3>(par - parM)), 0_r, tol);
}

TEST(Distances, VDistancePointTetrahedron) {
  static constexpr int kGrid = 25;
  static constexpr real kGridR = static_cast<real>(kGrid);

  // Define vertices of a tet
  Vec4r a{-0.5_r, 2.7_r, -1.7_r};
  Vec4r b{-2.3_r, 3.9_r, 3.4_r};
  Vec4r c{3.1_r, -2.1_r, -2.4_r};
  Vec4r d{2.5_r, -1.9_r, 3.1_r};

  auto TestPoint = [&a, &b, &c, &d](Vec4r const& pos) {
    // Get distance to tetrahedron
    real dist = Get0(VDistancePointTetrahedronSqr(pos, a, b, c, d));

    // If the point is inside the tetrahedron, test if distance is zero
    if (IsInsideTetrahedron(ToReal3(a), ToReal3(b), ToReal3(c), ToReal3(d), ToReal3(pos))) {
      EXPECT_NEAR_EQ(dist, 0_r);
      return;
    }

    // If the point is outside the tetrahedron, get the minimum distance to all boundary
    // primitives. It is sufficient to test the triangles, because it includes testing their
    // edges and vertices.
    VDistanceSignParams unused;
    real distMin = Get0(VDistancePointTriangleSqr(pos, a, b, c, unused));
    real distOther = Get0(VDistancePointTriangleSqr(pos, a, d, b, unused));
    if (distOther < distMin) {
      distMin = distOther;
    }
    distOther = Get0(VDistancePointTriangleSqr(pos, a, c, d, unused));
    if (distOther < distMin) {
      distMin = distOther;
    }
    distOther = Get0(VDistancePointTriangleSqr(pos, b, d, c, unused));
    if (distOther < distMin) {
      distMin = distOther;
    }

    // If the distance is practically zero, test absolute tolerance
    if (distMin < 1e-5_r) {
      EXPECT_NEAR_TOL(dist, 0_r, 1e-5_r);
    } else { // Otherwise test relative tolerance
      EXPECT_TRUE(NearEqualRel(dist, distMin, 1e-5_r));
    }
  };

  auto TestNearPoints = [&TestPoint](Vec4r const& pos) {
    TestPoint(pos);
    TestPoint(1e-4_r * SimdBasisVector<0>() + pos);
    TestPoint(-1e-4_r * SimdBasisVector<0>() + pos);
    TestPoint(1e-4_r * SimdBasisVector<1>() + pos);
    TestPoint(-1e-4_r * SimdBasisVector<1>() + pos);
    TestPoint(1e-4_r * SimdBasisVector<2>() + pos);
    TestPoint(-1e-4_r * SimdBasisVector<2>() + pos);
  };

  // Test corners, edge centers, face centers, and points nearby
  TestNearPoints(a);
  TestNearPoints(b);
  TestNearPoints(c);
  TestNearPoints(d);
  TestNearPoints(0.5_r * (a + b));
  TestNearPoints(0.5_r * (a + c));
  TestNearPoints(0.5_r * (a + d));
  TestNearPoints(0.5_r * (b + c));
  TestNearPoints(0.5_r * (b + d));
  TestNearPoints(0.5_r * (c + d));
  TestNearPoints((a + b + c) / 3_r);
  TestNearPoints((a + b + d) / 3_r);
  TestNearPoints((a + c + d) / 3_r);
  TestNearPoints((b + c + d) / 3_r);

  // Get bounding box and add some offset
  Vec4r minCoord = Min(a, b, c, d);
  minCoord -= 2_r;
  Vec4r maxCoord = Max(a, b, c, d);
  maxCoord += 2_r;

  // Test distances of a grid of points
  Vec4r delta = maxCoord - minCoord;
  for (int i = 0; i < kGrid; i++) {
    for (int j = 0; j < kGrid; j++) {
      for (int k = 0; k < kGrid; k++) {
        // Test point
        Vec4r pos =
            MulAdd(Vec4r((real)i / kGridR, (real)j / kGridR, (real)k / kGridR), delta, minCoord);
        TestPoint(pos);
      }
    }
  }
}
