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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/quaternion_utils.h>
#include <mochi_core/utils/rand_utils.h>

#include <gtest/gtest.h>
#include <picojson/picojson.h>

#include <cmath>
#include <limits>

using namespace mochi;

static_assert(alignof(Quaternion) == alignof(Vec4r), "Unexpected alignment");
static_assert(sizeof(Quaternion) == sizeof(real) * 4, "Unexpected padding");
static_assert(std::is_trivially_copyable_v<Quaternion>);

TEST(Quaternion, Constructors) {
  EXPECT_EQ(Quaternion(0_r, 0_r, 0_r, 1_r), Quaternion());
  EXPECT_EQ(Quaternion(1_r, 2_r, 3_r, 4_r), Quaternion(Vec4r(1_r, 2_r, 3_r, 4_r)));
  EXPECT_EQ(Quaternion(1_r, 2_r, 3_r, 4_r), Quaternion(Real4(1_r, 2_r, 3_r, 4_r)));
  EXPECT_EQ(Quaternion(1_r, 2_r, 3_r, 4_r), Quaternion(1_r, 2_r, 3_r, 4_r));
}

TEST(Quaternion, FactoryFunctions) {
  EXPECT_EQ(Quaternion(0_r, 0_r, 0_r, 1_r), Quaternion::Identity());
  EXPECT_EQ(Quaternion(0_r, 0_r, 0_r, 0_r), Quaternion::Zero());
  EXPECT_NEAR_EQ(
      Quaternion(0.382683456_r, 0_r, 0_r, 0.923879504_r), Quaternion::RotationX(kPI * 0.25_r));
  EXPECT_NEAR_EQ(
      Quaternion(0_r, 0.382683456_r, 0_r, 0.923879504_r), Quaternion::RotationY(kPI * 0.25_r));
  EXPECT_NEAR_EQ(
      Quaternion(0_r, 0_r, 0.382683456_r, 0.923879504_r), Quaternion::RotationZ(kPI * 0.25_r));
}

TEST(Quaternion, Negation) {
  EXPECT_EQ(Quaternion(1_r, 2_r, 3_r, 4_r), -Quaternion(-1_r, -2_r, -3_r, -4_r));
  EXPECT_EQ(Quaternion(-1_r, 2_r, -3_r, 4_r), -Quaternion(1_r, -2_r, 3_r, -4_r));
}

TEST(Quaternion, Multiplication) {
  // This test is not exhaustive, but it does cover a few cases of
  // (quaternion * quaternion) and (quaternion * Real3) and (Quaternion * Vec4r)
  Quaternion rotX90 = Quaternion::RotationX(90_r * kRadiansPerDegree);
  Quaternion rotX180 = Quaternion::RotationX(180_r * kRadiansPerDegree);
  Quaternion rotY90 = Quaternion::RotationY(90_r * kRadiansPerDegree);
  Real3 xAxis = Real3{1_r, 0_r, 0_r};
  Real3 yAxis = Real3{0_r, 1_r, 0_r};
  Real3 zAxis = Real3{0_r, 0_r, 1_r};

  EXPECT_NEAR_EQ(zAxis, rotX90 * yAxis); // (0,1,0) rotated 90 deg about x-axis
  EXPECT_NEAR_EQ(-yAxis, rotX180 * yAxis); // (0,1,0) rotated 180 deg about x-axis
  EXPECT_NEAR_EQ(-yAxis, (rotX90 * rotX90) * yAxis); // Same, but 90 deg rotations are concatenated
  EXPECT_NEAR_EQ(yAxis, rotY90 * yAxis); // (0,1,0) rotated 90 deg about itself

  // (0,1,0) rotated 90 deg about x-axis, then 90 deg about y-axis
  EXPECT_NEAR_EQ(xAxis, (rotY90 * rotX90) * yAxis);

  // (0,1,0) rotated 90 deg about y-axis, then 90 deg about x-axis
  EXPECT_NEAR_EQ(zAxis, (rotX90 * rotY90) * yAxis);

  // Quaternion * Vec4r where (w == 0)
  EXPECT_NEAR_EQ(ToSimd(zAxis, 0_r), rotX90 * ToSimd(yAxis, 0_r));
  EXPECT_NEAR_EQ(ToSimd(-yAxis, 0_r), rotX180 * ToSimd(yAxis, 0_r));
  EXPECT_NEAR_EQ(ToSimd(-yAxis, 0_r), (rotX90 * rotX90) * ToSimd(yAxis, 0_r));
  EXPECT_NEAR_EQ(ToSimd(yAxis, 0_r), rotY90 * ToSimd(yAxis, 0_r));

  // Quaternion * Vec4r where (w == 1)
  EXPECT_NEAR_EQ(ToSimd(zAxis, 1_r), rotX90 * ToSimd(yAxis, 1_r));
  EXPECT_NEAR_EQ(ToSimd(-yAxis, 1_r), rotX180 * ToSimd(yAxis, 1_r));
  EXPECT_NEAR_EQ(ToSimd(-yAxis, 1_r), (rotX90 * rotX90) * ToSimd(yAxis, 1_r));
  EXPECT_NEAR_EQ(ToSimd(yAxis, 1_r), rotY90 * ToSimd(yAxis, 1_r));

  // Quaternion algebra: i*j=k, j*k=i, k*i=j
  Quaternion const i = Quaternion(1_r, 0_r, 0_r, 0_r);
  Quaternion const j = Quaternion(0_r, 1_r, 0_r, 0_r);
  Quaternion const k = Quaternion(0_r, 0_r, 1_r, 0_r);
  EXPECT_EQ(k, i * j);
  EXPECT_EQ(i, j * k);
  EXPECT_EQ(j, k * i);

  // i^2 = j^2 = k^2 = -1 (w = -1)
  Quaternion const neg1(0_r, 0_r, 0_r, -1_r);
  EXPECT_EQ(neg1, i * i);
  EXPECT_EQ(neg1, j * j);
  EXPECT_EQ(neg1, k * k);
}

TEST(Quaternion, GetConjugate) {
  // Quaternion inverse flips the bits of XYZ, but not W
  EXPECT_EQ(Quaternion(1_r, -2_r, 3_r, -4_r), Quaternion(-1_r, 2_r, -3_r, -4_r).GetConjugate());

  // Rotation examples
  EXPECT_NEAR_EQ(Quaternion::RotationX(1_r), Quaternion::RotationX(-1_r).GetConjugate());
  EXPECT_NEAR_EQ(Quaternion::RotationY(kPI), Quaternion::RotationY(-kPI).GetConjugate());
  EXPECT_NEAR_EQ(
      Quaternion::RotationZ(kPI),
      Quaternion::FromAxisAngle(Real3{0_r, 0_r, -1_r}, kPI).GetConjugate());
  EXPECT_NEAR_EQ(
      Quaternion::RotationZ(kPI),
      Quaternion::FromAxisAngle(Real3{0_r, 0_r, 1_r}, -kPI).GetConjugate());
}

TEST(Quaternion, AxisAngle) {
  Real3 axis = {};
  real angle = 0_r;

  // Identity quaternion still has to return a unit length axis
  Quaternion q;
  q.ToAxisAngle(&axis, &angle);
  EXPECT_EQ(Real3(1_r, 0_r, 0_r), axis); // happens to choose x-axis
  EXPECT_EQ(0_r, angle);
  Quaternion q2 = Quaternion::FromAxisAngle(axis, angle);
  EXPECT_NEAR_EQ(q, q2);

  for (real a = 0.1_r; a < kPI; a += 0.1_r) {
    real constexpr kEps = 10_r * kDefaultNearEqualEpsilon<real>;

    // x-axis
    q = Quaternion::FromAxisAngle(Real3(1_r, 0_r, 0_r), a);
    EXPECT_NEAR_TOL(Quaternion::RotationX(a), q, kEps);
    q.ToAxisAngle(&axis, &angle);
    EXPECT_NEAR_TOL(Real3(1_r, 0_r, 0_r), axis, kEps);
    EXPECT_NEAR_TOL(a, angle, kEps);

    // y-axis
    q = Quaternion::FromAxisAngle(Real3(0_r, 1_r, 0_r), a);
    EXPECT_NEAR_TOL(Quaternion::RotationY(a), q, kEps);
    q.ToAxisAngle(&axis, &angle);
    EXPECT_NEAR_TOL(Real3(0_r, 1_r, 0_r), axis, kEps);
    EXPECT_NEAR_TOL(a, angle, kEps);

    // z-axis
    q = Quaternion::FromAxisAngle(Real3(0_r, 0_r, 1_r), a);
    EXPECT_NEAR_TOL(Quaternion::RotationZ(a), q, kEps);
    q.ToAxisAngle(&axis, &angle);
    EXPECT_NEAR_TOL(Real3(0_r, 0_r, 1_r), axis, kEps);
    EXPECT_NEAR_TOL(a, angle, kEps);
  }

  // Zero degree rotation (axis could be anything)
  q = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, 0_r);
  q.ToAxisAngle(&axis, &angle);
  // ToAxisAngle could reasonably return any unit vector for the axis.
  EXPECT_NEAR_EQ(1_r, Norm(axis));
  EXPECT_EQ(0_r, angle);

  // Test +/- 180 degree rotation on various axes
  for (int i = 0; i < 4; ++i) {
    Real3 expectedAxis = {};
    if (i < 3) {
      // x, y, or z axis
      expectedAxis[i] = 1_r;
    } else {
      // arbitrary vector
      expectedAxis = Normalize(Real3{1_r, 2_r, 3_r});
    }
    real constexpr kEps = std::numeric_limits<real>::epsilon();
    real constexpr kTestAngles[] = {-kPI - kEps, -kPI, -kPI + kEps, kPI - kEps, kPI, kPI + kEps};
    for (auto expectedAngle : kTestAngles) {
      q = Quaternion::FromAxisAngle(expectedAxis, expectedAngle);
      q.ToAxisAngle(&axis, &angle);
      // axis and -axis are equivalent.
      EXPECT_TRUE(NearEqual(expectedAxis, axis) || NearEqual(expectedAxis, -axis));
      // +kPI and -kPI are equivalent. ToAxisAngle always returns +kPi.
      EXPECT_NEAR(Abs(expectedAngle), Abs(angle), 1e2_r * kDefaultNearEqualEpsilon<real>);
    }
  }

  // Quaternion concatenation may lead to edge cases where conversion
  // to axis-angle returns non-unit-length axis.
  {
    real epsilon = 1.0e-10_r;
    for (int i = 0; i < 8; i++) { // Test problematic small numbers [1.0e-9,1.0e-1]
      epsilon *= 10_r;
      q = Quaternion(
          epsilon,
          epsilon,
          epsilon,
          std::sqrt(1_r - 3_r * epsilon * epsilon)); // Ensure quaternion has unit-length by setting
                                                     // w = sqrt(1 - x*x - y*y - z*z)
      q.ToAxisAngle(&axis, &angle);
      EXPECT_NEAR_EQ(Norm(axis), 1.0_r);
      EXPECT_TRUE(angle >= 0_r && angle <= 2_r * kPI);
    }
  }
}

TEST(Quaternion, ToAxisAngleVec4r) {
  // Sanity check for equivalence with the Real3 overload of ToAxisAngle.
  Quaternion const q = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, 1.5_r);
  Real3 axisR3{};
  real angleR3{};
  q.ToAxisAngle(&axisR3, &angleR3);
  Vec4r axisV4{};
  real angleV4{};
  q.ToAxisAngle(&axisV4, &angleV4);
  EXPECT_NEAR_EQ(ToSimd(axisR3, 0_r), axisV4);
  EXPECT_NEAR_EQ(angleR3, angleV4);
}

TEST(Quaternion, GetAngle) {
  // Identity quaternion should report angle 0.
  Quaternion q;
  EXPECT_NEAR_EQ(q.GetAngle(), 0_r);

  std::vector<Real3> const axes = {
      Normalize(Real3{1_r, 0_r, 0_r}),
      Normalize(Real3{0_r, 1_r, 0_r}),
      Normalize(Real3{0_r, 0_r, 1_r}),
      Normalize(Real3{1_r, 1_r, 0_r}),
      Normalize(Real3{1_r, 0_r, 1_r}),
      Normalize(Real3{0_r, 1_r, 1_r})};

  // Test consistency for a small number of rotations around the x/y/z axes.
  for (real a = 0.1_r; a < kPI; a += 0.1_r) {
    for (auto const& axis : axes) {
      q = Quaternion::FromAxisAngle(axis, a);
      real const angle = Normalize(q).GetAngle();
      EXPECT_TRUE(NearEqual(a, angle, 10_r * kDefaultNearEqualEpsilon<real>));
    }
  }
}

TEST(Quaternion, FromRotationVector) {
  // A rotation vector r = axis * angle
  // FromRotationVector should match FromAxisAngle(Normalize(r), Norm(r))

  // 90-degree rotation about X
  Real3 const rotVec{kPI * 0.5_r, 0_r, 0_r};
  Quaternion const qExpected = Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, kPI * 0.5_r);
  EXPECT_NEAR_EQ(qExpected, Quaternion::FromRotationVector(rotVec));

  // Vec4r overload should produce the same result
  EXPECT_NEAR_EQ(qExpected, Quaternion::FromRotationVector(ToSimd(rotVec, 0_r)));

  // Arbitrary rotation vector
  Real3 const arbitraryRot{0.5_r, -0.3_r, 0.7_r};
  real const angle = Norm(arbitraryRot);
  Real3 const axis = Normalize(arbitraryRot);
  EXPECT_NEAR_EQ(
      Quaternion::FromAxisAngle(axis, angle), Quaternion::FromRotationVector(arbitraryRot));

  // Very small rotation (exercises small-angle code path where angle <= 1e-9)
  Real3 const smallRot{1e-11_r, 0_r, 0_r};
  Quaternion const qSmall = Quaternion::FromRotationVector(smallRot);
  EXPECT_NEAR_EQ(1_r, Norm(qSmall));
}

TEST(Quaternion, ToRotationVector) {
  // RotationX(angle) should give rotation vector (angle, 0, 0)
  real const angle = 1.2_r;
  Quaternion const qx = Quaternion::RotationX(angle);
  EXPECT_NEAR_EQ(Real3(angle, 0_r, 0_r), qx.ToRotationVector());

  // Round-trip: axis/angle -> quaternion -> rotation vector -> quaternion
  Real3 const axis = Normalize(Real3{1_r, 2_r, 3_r});
  Quaternion const q = Quaternion::FromAxisAngle(axis, angle);
  Real3 const rotVec = q.ToRotationVector();
  EXPECT_NEAR_EQ(q, Quaternion::FromRotationVector(rotVec));

  // VToRotationVector should be consistent with ToRotationVector
  Vec4r const vRotVec = q.VToRotationVector();
  EXPECT_NEAR_EQ(ToSimd(rotVec, 0_r), vRotVec);

  // Identity -> zero rotation vector
  EXPECT_NEAR_EQ(Real3(0_r, 0_r, 0_r), Quaternion::Identity().ToRotationVector());
}

TEST(Quaternion, AdditionSubtraction) {
  Quaternion const a(1_r, 2_r, 3_r, 4_r);
  Quaternion const b(5_r, 6_r, 7_r, 8_r);

  EXPECT_EQ(Quaternion(6_r, 8_r, 10_r, 12_r), a + b);
  EXPECT_EQ(Quaternion(-4_r, -4_r, -4_r, -4_r), a - b);

  // a + (-a) = zero
  EXPECT_EQ(Quaternion::Zero(), a + (-a));

  // a - a = zero
  EXPECT_EQ(Quaternion::Zero(), a - a);

  // Commutativity of addition
  EXPECT_EQ(a + b, b + a);
}

TEST(Quaternion, EqualityOperators) {
  Quaternion const a(1_r, 2_r, 3_r, 4_r);
  Quaternion const b(1_r, 2_r, 3_r, 4_r);
  Quaternion const c(1_r, 2_r, 3_r, 5_r);

  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a != b);
  EXPECT_FALSE(a == c);
  EXPECT_TRUE(a != c);

  // Default constructor == Identity
  EXPECT_TRUE(Quaternion::Identity() == Quaternion());
}

TEST(Quaternion, Reflection) {
  Quaternion q(-1_r, 0_r, 0.5_r, 1.0_r);

  auto const& typeInfo = SReflect::GetTypeInfo<Quaternion>();

  // Serialization
  EXPECT_STREQ("[-1,0,0.5,1]", SReflect::ToJsonString(q, false).c_str());
  EXPECT_EQ(q, SReflect::FromJsonString<Quaternion>("[-1,0,0.5,1]"));

  // Type Introspection
  EXPECT_STREQ("Quaternion", typeInfo._name);
  EXPECT_STREQ("mochi::Quaternion", typeInfo._nameWithNamespace);
  EXPECT_EQ(SReflect::CoreType::CT_array, typeInfo._coreType);
  EXPECT_EQ(sizeof(Quaternion), typeInfo._sizeInBytes);
  EXPECT_EQ(alignof(Quaternion), typeInfo._alignment);
  EXPECT_EQ(&SReflect::GetTypeInfo<real>(), typeInfo._innerTypeInfo);

  // Factor Creation (does not require compile-time access to the Quaternion type)
  void* newObj = typeInfo.New();
  picojson::value json = picojson::object();
  typeInfo.Serialize(newObj, json);
  EXPECT_STREQ("[0,0,0,1]", json.serialize(false).c_str());
  typeInfo.Delete(newObj);
}
