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
#include <mochi_core/utils/rodrigues_utils.h>

#include <gtest/gtest.h>

#include <limits>

using namespace mochi;

TEST(QuaternionUtils, NearEqual) {
  // clang-format off

  real epsilon = 0_r; // exact equality
  EXPECT_TRUE(NearEqual(Real4(1_r, 2_r, 3_r, 4_r), Quaternion(1_r, 2_r, 3_r, 4_r).ToReal4(), epsilon));
  EXPECT_FALSE(NearEqual(Real4(1_r, 2_r, 3_r, 4_r), Quaternion(1.000001_r, 2_r, 3_r, 4_r).ToReal4(), epsilon));
  EXPECT_FALSE(NearEqual(Real4(1_r, 2_r, 3_r, 4_r), Quaternion(1_r, 2.000001_r, 3_r, 4_r).ToReal4(), epsilon));
  EXPECT_FALSE(NearEqual(Real4(1_r, 2_r, 3_r, 4_r), Quaternion(1_r, 2_r, 3.000001_r, 4_r).ToReal4(), epsilon));
  EXPECT_FALSE(NearEqual(Real4(1_r, 2_r, 3_r, 4_r), Quaternion(1_r, 2_r, 3_r, 4.000001_r).ToReal4(), epsilon));

  epsilon = 0.1_r; // close enough
  EXPECT_TRUE(NearEqual(Real4(1_r, 2_r, 3_r, 4_r), Quaternion(1_r, 2_r, 3_r, 4_r).ToReal4(), epsilon));
  EXPECT_TRUE(NearEqual(Real4(1_r, 2_r, 3_r, 4_r), Quaternion(1.09_r, 2_r, 3_r, 4_r).ToReal4(), epsilon));
  EXPECT_TRUE(NearEqual(Real4(1_r, 2_r, 3_r, 4_r), Quaternion(1_r, 2.09_r, 3_r, 4_r).ToReal4(), epsilon));
  EXPECT_TRUE(NearEqual(Real4(1_r, 2_r, 3_r, 4_r), Quaternion(1_r, 2_r, 3.09_r, 4_r).ToReal4(), epsilon));
  EXPECT_TRUE(NearEqual(Real4(1_r, 2_r, 3_r, 4_r), Quaternion(1_r, 2_r, 3_r, 4.09_r).ToReal4(), epsilon));
  EXPECT_FALSE(NearEqual(Real4(1_r, 2_r, 3_r, 4_r), Quaternion(1.11_r, 2_r, 3_r, 4_r).ToReal4(), epsilon));
  EXPECT_FALSE(NearEqual(Real4(1_r, 2_r, 3_r, 4_r), Quaternion(1_r, 2.11_r, 3_r, 4_r).ToReal4(), epsilon));
  EXPECT_FALSE(NearEqual(Real4(1_r, 2_r, 3_r, 4_r), Quaternion(1_r, 2_r, 3.11_r, 4_r).ToReal4(), epsilon));
  EXPECT_FALSE(NearEqual(Real4(1_r, 2_r, 3_r, 4_r), Quaternion(1_r, 2_r, 3_r, 4.11_r).ToReal4(), epsilon));

  // clang-format on
}

TEST(QuaternionUtils, EquivalentRotation) {
  // Q and -Q are equivalent
  EXPECT_TRUE(EquivalentRotation(Quaternion::Zero(), Quaternion::Zero()));
  EXPECT_TRUE(EquivalentRotation(Quaternion::Zero(), -Quaternion::Zero()));
  EXPECT_TRUE(EquivalentRotation(Quaternion::Identity(), Quaternion::Identity()));
  EXPECT_TRUE(EquivalentRotation(Quaternion::Identity(), -Quaternion::Identity()));
  EXPECT_TRUE(
      EquivalentRotation(Quaternion(1_r, -2_r, 3_r, -4_r), Quaternion(-1_r, 2_r, -3_r, 4_r)));

  // -Q is like negating the axis of rotation AND the angle
  Real3 axis{1_r, 0_r, 0_r};
  real angle = 0.1_r;
  EXPECT_FALSE(EquivalentRotation(
      Quaternion::FromAxisAngle(axis, angle),
      Quaternion::FromAxisAngle(axis, -angle))); // opposite angle only
  EXPECT_FALSE(EquivalentRotation(
      Quaternion::FromAxisAngle(axis, angle),
      Quaternion::FromAxisAngle(-axis, angle))); // opposite axis only
  EXPECT_TRUE(EquivalentRotation(
      Quaternion::FromAxisAngle(axis, angle),
      Quaternion::FromAxisAngle(-axis, -angle))); // opposite axis and angle is equivalent
}

TEST(QuaternionUtils, Lerp) {
  // Note: Lerp does not clamp the 't' parameter
  Quaternion const qa(10_r, 100_r, 1000_r, 10000_r);
  Quaternion const qb(20_r, 200_r, 2000_r, 20000_r);
  EXPECT_EQ(Quaternion(0_r, 0_r, 0_r, 0_r), Lerp(qa, qb, -1_r));
  EXPECT_EQ(Quaternion(5_r, 50_r, 500_r, 5000_r), Lerp(qa, qb, -0.5_r));
  EXPECT_EQ(Quaternion(10_r, 100_r, 1000_r, 10000_r), Lerp(qa, qb, 0_r));
  EXPECT_EQ(Quaternion(15_r, 150_r, 1500_r, 15000_r), Lerp(qa, qb, 0.5_r));
  EXPECT_EQ(Quaternion(20_r, 200_r, 2000_r, 20000_r), Lerp(qa, qb, 1_r));
  EXPECT_EQ(Quaternion(25_r, 250_r, 2500_r, 25000_r), Lerp(qa, qb, 1.5_r));
}

TEST(QuaternionUtils, Slerp) {
  // arbitrary rotation axis
  Real3 const axis = Normalize(Real3{1_r, 2_r, 3_r});

  // helper
  auto Rot = [&](real angleDeg) {
    return Quaternion::FromAxisAngle(axis, angleDeg * kRadiansPerDegree);
  };

  EXPECT_NEAR_EQ(Rot(0_r), Slerp(Rot(0_r), Rot(0_r), 0_r));
  EXPECT_NEAR_EQ(Rot(0_r), Slerp(Rot(0_r), Rot(0_r), 1_r));

  EXPECT_NEAR_EQ(Rot(0_r), Slerp(Rot(45_r), Rot(90_r), -1_r));
  EXPECT_NEAR_EQ(Rot(22.5_r), Slerp(Rot(45_r), Rot(90_r), -0.5_r));
  EXPECT_NEAR_EQ(Rot(45_r), Slerp(Rot(45_r), Rot(90_r), 0_r));
  EXPECT_NEAR_EQ(Rot(67.5_r), Slerp(Rot(45_r), Rot(90_r), 0.5_r));
  EXPECT_NEAR_EQ(Rot(90_r), Slerp(Rot(45_r), Rot(90_r), 1_r));
  EXPECT_NEAR_EQ(Rot(112.5_r), Slerp(Rot(45_r), Rot(90_r), 1.5_r));

  // Slerp from 0 to 180 deg should take the shorter 179 deg arc
  EXPECT_NEAR_EQ(Rot(0_r), Slerp(Rot(0_r), Rot(181_r), 0.0_r));
  EXPECT_NEAR_EQ(Rot(-44.75_r), Slerp(Rot(0_r), Rot(181_r), 0.25_r));
  EXPECT_NEAR_EQ(Rot(-89.5_r), Slerp(Rot(0_r), Rot(181_r), 0.5_r));
  EXPECT_NEAR_EQ(Rot(-134.25_r), Slerp(Rot(0_r), Rot(181_r), 0.75_r));
  EXPECT_NEAR_EQ(Rot(-179.0_r), Slerp(Rot(0_r), Rot(181_r), 1.0_r));

  // Small rotations take θ from ASin, where ACos(dot) would lose accuracy, and rotations close to
  // 180 degrees take it from ACos. Both stay accurate and unit; extrapolating to t = 1.5 amplifies
  // errors.
  real constexpr kTol = 8_r * std::numeric_limits<real>::epsilon();
  Quaternion const a = Rot(70_r);
  Real3 const axisB{3_r, -1_r, 2_r};
  for (real const angle : {1e-6_r, 2.7e-3_r, 3.5e-3_r, 0.05_r, kPI - 1e-3_r}) {
    Quaternion const b = Quaternion::FromAxisAngle(axisB, angle) * a;
    Quaternion const expected = Quaternion::FromAxisAngle(axisB, 1.5_r * angle) * a;
    EXPECT_TRUE(EquivalentRotation(Slerp(a, b, 1.5_r), expected, kTol));
  }

  // Equal and opposite inputs give the input itself at any t, although the rounded dot product of c
  // with itself isn't 1. So does a parallel input, for which b - a·dot is zero.
  Quaternion const c = Normalize(Quaternion{1_r, 1_r, 0_r, 0_r});
  EXPECT_EQ(c, Slerp(c, c, 65536_r));
  EXPECT_EQ(c, Slerp(c, -c, 65536_r));
  Quaternion const longIdentity{0_r, 0_r, 0_r, 1_r + std::numeric_limits<real>::epsilon()};
  EXPECT_EQ(Rot(0_r), Slerp(Rot(0_r), longIdentity, 0.5_r));

  // A rotation so small that the norm of b - a·dot underflows to zero still interpolates.
  Quaternion const tiny = Quaternion::RotationX(4_r * std::numeric_limits<real>::min());
  EXPECT_EQ(tiny, Slerp(Rot(0_r), tiny, 1_r));
}

TEST(QuaternionUtils, Norm) {
  EXPECT_EQ(1_r, Norm(Quaternion()));
  EXPECT_NEAR_EQ(
      Sqrt(1_r * 1_r + 2_r * 2_r + 3_r * 3_r + 4_r * 4_r), Norm(Quaternion(1_r, 2_r, 3_r, 4_r)));
  EXPECT_EQ(0_r, Norm(Quaternion::Zero()));
  EXPECT_EQ(1_r, Norm(Quaternion::Identity()));
}

TEST(QuaternionUtils, Normalize) {
  real const sro2 = Sqrt(2_r) / 2_r;
  EXPECT_NEAR_EQ(Quaternion(sro2, sro2, 0_r, 0_r), Normalize(Quaternion(1_r, 1_r, 0_r, 0_r)));

  Quaternion q{1_r, 2_r, 3_r, 4_r};
  EXPECT_NEAR_EQ(Quaternion(q / Norm(q)), Normalize(q));
}

TEST(QuaternionUtils, IsFinite) {
  // Test non-finite values in every position of the quaternion
  for (int i = 0; i < 4; ++i) {
    Quaternion q;
    EXPECT_TRUE(IsFinite(q));
    q.data = Set(q.data, i, std::numeric_limits<real>::infinity());
    EXPECT_FALSE(IsFinite(q));
    q.data = Set(q.data, i, -std::numeric_limits<real>::infinity());
    EXPECT_FALSE(IsFinite(q));
    q.data = Set(q.data, i, std::numeric_limits<real>::quiet_NaN());
    EXPECT_FALSE(IsFinite(q));
    q.data = Set(q.data, i, std::numeric_limits<real>::signaling_NaN());
    EXPECT_FALSE(IsFinite(q));
    q.data = Set(q.data, i, 123_r);
    EXPECT_TRUE(IsFinite(q));
  }
}

TEST(QuaternionUtils, ScalarMath) {
  Quaternion const q(1_r, 2_r, 3_r, 4_r);

  // real * Quaternion
  EXPECT_EQ(Quaternion(2_r, 4_r, 6_r, 8_r), 2_r * q);

  // Quaternion * real
  EXPECT_EQ(Quaternion(2_r, 4_r, 6_r, 8_r), q * 2_r);

  // Commutativity of scalar multiplication
  EXPECT_EQ(3_r * q, q * 3_r);

  // Multiply by zero
  EXPECT_EQ(Quaternion::Zero(), 0_r * q);

  // Multiply by one
  EXPECT_EQ(q, 1_r * q);

  // Divide by one
  EXPECT_EQ(q, q / 1_r);

  // Divide by two
  EXPECT_EQ(Quaternion(1_r / 2_r, 2_r / 2_r, 3_r / 2_r, 4_r / 2_r), q / 2_r);
}

TEST(QuaternionUtils, NormSqr) {
  // |q|^2 for (1,2,3,4) = 1+4+9+16 = 30
  EXPECT_EQ(30_r, NormSqr(Quaternion(1_r, 2_r, 3_r, 4_r)));

  // Identity: |q|^2 = 1
  EXPECT_EQ(1_r, NormSqr(Quaternion::Identity()));

  // Zero: |q|^2 = 0
  EXPECT_EQ(0_r, NormSqr(Quaternion::Zero()));

  // NormSqr == Norm^2
  Quaternion const q(3_r, 4_r, 0_r, 0_r);
  real const n = Norm(q);
  EXPECT_NEAR_EQ(n * n, NormSqr(q));
}

TEST(QuaternionUtils, Conjugate) {
  // Same as Quaternion::GetConjugate
  EXPECT_EQ(Quaternion(1_r, -2_r, 3_r, -4_r), Conjugate(Quaternion(-1_r, 2_r, -3_r, -4_r)));
  EXPECT_EQ(Quaternion(1_r, -2_r, 3_r, -4_r), Quaternion(-1_r, 2_r, -3_r, -4_r).GetConjugate());
}

TEST(QuaternionUtils, FromMatrix) {
  real constexpr kTol = 100_r * std::numeric_limits<real>::epsilon();

  // Identity quaternion
  {
    Quaternion q;
    auto rot = Rodrigues(q.VToRotationVector());
    Quaternion q_recovered = QuaternionFromMatrix(rot);
    EXPECT_NEAR_EQ(q, q_recovered);
  }

  // Test +/- 180 degree rotation on various axes
  for (int i = 0; i < 4; ++i) {
    Real3 axis = {};
    if (i < 3) {
      // x, y, or z axis
      axis[i] = 1_r;
    } else {
      // arbitrary vector
      axis = Normalize(Real3{1_r, 2_r, 3_r});
    }
    real constexpr kEps = 10_r * std::numeric_limits<real>::epsilon();
    real constexpr kTestAngles[] = {-kPI - kEps, -kPI, -kPI + kEps, kPI - kEps, kPI, kPI + kEps};
    for (auto angle : kTestAngles) {
      // Axis-angle to VMatrix3x3r
      auto q = Quaternion::FromAxisAngle(axis, angle);
      auto vmat = ToVMatrix3x3(q);
      // VMatrix3x3r to Quaternion
      auto q2 = QuaternionFromMatrix(vmat);
      EXPECT_TRUE(EquivalentRotation(q, q2, kEps));
      // Repeat with Matrix3x3r
      auto q3 = QuaternionFromMatrix(ToNdArray3x3(vmat));
      EXPECT_TRUE(EquivalentRotation(q, q3, kEps));
    }
  }

  // Rotations near 0 and 180 degrees are recovered to rounding error
  {
    // The negative largest component exercises the flip to w >= 0 near 180 degrees
    Real3 const axis = Normalize(Real3{1_r, 2_r, -3_r});
    for (real const angle : {1e-8_r, 1e-6_r, 1e-4_r, 1e-2_r, kPI - 1e-4_r, kPI - 1e-6_r}) {
      Quaternion const q =
          QuaternionFromMatrix(ToVMatrix3x3(Quaternion::FromAxisAngle(axis, angle)));
      EXPECT_TRUE(NearEqual(q.ToRotationVector(), angle * axis, kTol * angle));
      EXPECT_GE(q.data[3], 0_r);
    }
  }

  // Random rotations R are recovered, also from R U for an upper triangular U with a positive
  // diagonal, whose Gram-Schmidt orthonormalization is R
  {
    auto rng = RandomGenerator(42);
    for (int i = 0; i < 100; ++i) {
      Real3 axis{};
      SetRandom(rng, -1_r, 1_r, axis);
      axis = Normalize(axis);
      real angle = RandomUniformValue(rng, -1_r * kPI, 1_r * kPI);
      Quaternion q = Quaternion::FromAxisAngle(axis, angle);
      Quaternion q2 = QuaternionFromMatrix(ToVMatrix3x3(q));
      EXPECT_TRUE(EquivalentRotation(q, q2, kTol));
      EXPECT_GE(q2.data[3], 0_r);
      Real3 diagonal{};
      Real3 offDiagonal{};
      SetRandom(rng, 0.5_r, 2_r, diagonal);
      SetRandom(rng, -0.1_r, 0.1_r, offDiagonal);
      Matrix3x3r const upper{
          Real3{diagonal[0], offDiagonal[0], offDiagonal[1]},
          Real3{0_r, diagonal[1], offDiagonal[2]},
          Real3{0_r, 0_r, diagonal[2]}};
      EXPECT_TRUE(EquivalentRotation(q, QuaternionFromMatrix(Dot(ToMatrix3x3(q), upper)), kTol));
    }
  }
}

TEST(QuaternionUtils, ToVMatrix3x3Roundtrip) {
  Quaternion const q = Quaternion::FromAxisAngle(Normalize(Real3{1_r, 2_r, 3_r}), 1.0_r);

  VMatrix3x3r const mat = ToVMatrix3x3(q);
  VMatrix3x3r const matT = ToVMatrix3x3Transpose(q);

  // mat and matT should be transposes of each other
  EXPECT_NEAR_EQ(mat, Transpose3x3(matT));

  // ToVMatrix3x3_WithTranspose should return both
  auto [matBoth, matTBoth] = ToVMatrix3x3_WithTranspose(q);
  EXPECT_NEAR_EQ(mat, matBoth);
  EXPECT_NEAR_EQ(matT, matTBoth);

  // Round-trip: q -> matrix -> q should give equivalent rotation
  Quaternion const q2 = QuaternionFromMatrix(mat);
  EXPECT_TRUE(EquivalentRotation(q, q2));
}

TEST(QuaternionUtils, NearEqualQuaternionOverload) {
  Quaternion const a(1_r, 2_r, 3_r, 4_r);

  // Exact match
  EXPECT_TRUE(NearEqual(a, Quaternion(1_r, 2_r, 3_r, 4_r)));

  // Within epsilon
  EXPECT_TRUE(NearEqual(a, Quaternion(1.05_r, 2_r, 3_r, 4_r), 0.1_r));

  // Beyond epsilon
  EXPECT_FALSE(NearEqual(a, Quaternion(1.15_r, 2_r, 3_r, 4_r), 0.1_r));
}

TEST(QuaternionUtils, EquivalentRotationVec4rEpsilon) {
  Quaternion const a = Quaternion::RotationX(1_r);

  // q and -q should be equivalent rotations
  EXPECT_TRUE(EquivalentRotation(a, -a, Vec4r{0.001_r}));

  // Different rotations are not equivalent
  EXPECT_FALSE(EquivalentRotation(a, Quaternion::RotationX(2_r), Vec4r{0.001_r}));
}
