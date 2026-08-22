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
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/transform_rt.h>

#include <limits>

using namespace mochi;

static_assert(alignof(TransformRT) == alignof(Vec4r), "Unexpected alignment");
static_assert(sizeof(TransformRT) <= sizeof(real) * 8, "Unexpected padding");
static_assert(std::is_trivially_copyable_v<TransformRT>);

TEST(TransformRT, Constructors) {
  // Default = identity + no translation
  {
    TransformRT pose;
    EXPECT_EQ(Quaternion::Identity(), pose.GetRotation());
    EXPECT_EQ(Real3(0_r, 0_r, 0_r), pose.GetTranslation());
    EXPECT_EQ(Real4(0_r, 0_r, 0_r, 1_r), ToReal4(pose.VGetTranslation()));
    EXPECT_EQ(TransformRT::Identity(), pose);
  }

  // Quaternion, Vec4r
  {
    Quaternion rot = Quaternion::RotationX(kPI);
    Vec4r trans(1_r, 2_r, 3_r, 99999_r); // w will be forced to 1.0
    TransformRT pose(rot, trans);
    EXPECT_EQ(rot, pose.GetRotation());
    EXPECT_EQ(Real3(1_r, 2_r, 3_r), pose.GetTranslation());
    EXPECT_EQ(Real4(1_r, 2_r, 3_r, 1_r), ToReal4(pose.VGetTranslation()));
  }

  // Quaternion, Real3
  {
    Quaternion rot = Quaternion::RotationX(kPI);
    Real3 trans(1_r, 2_r, 3_r);
    TransformRT pose(rot, trans);
    EXPECT_EQ(rot, pose.GetRotation());
    EXPECT_EQ(trans, pose.GetTranslation());
    EXPECT_EQ(Real4(1_r, 2_r, 3_r, 1_r), ToReal4(pose.VGetTranslation()));
  }

  // Quaternion (only)
  {
    Quaternion rot = Quaternion::RotationX(kPI);
    TransformRT pose(rot);
    EXPECT_EQ(rot, pose.GetRotation());
    EXPECT_EQ(Real3(0_r, 0_r, 0_r), pose.GetTranslation());
    EXPECT_EQ(Real4(0_r, 0_r, 0_r, 1_r), ToReal4(pose.VGetTranslation()));
  }

  // Vec4r (only)
  {
    Vec4r trans = Vec4r(1_r, 2_r, 3_r, 99999_r); // w will be forced to 1.0
    TransformRT pose(trans);
    EXPECT_EQ(Quaternion::Identity(), pose.GetRotation());
    EXPECT_EQ(Real3(1_r, 2_r, 3_r), pose.GetTranslation());
    EXPECT_EQ(Real4(1_r, 2_r, 3_r, 1_r), ToReal4(pose.VGetTranslation()));
  }

  // Real3 (only)
  {
    Real3 trans(1_r, 2_r, 3_r);
    TransformRT pose(trans);
    EXPECT_EQ(Quaternion::Identity(), pose.GetRotation());
    EXPECT_EQ(Real3(1_r, 2_r, 3_r), pose.GetTranslation());
    EXPECT_EQ(Real4(1_r, 2_r, 3_r, 1_r), ToReal4(pose.VGetTranslation()));
  }
}

TEST(TransformRT, Accessors) {
  TransformRT pose;
  Quaternion rot = Quaternion::RotationX(kPI);

  // Get/Set Rotation
  EXPECT_EQ(Quaternion::Identity(), pose.GetRotation());
  pose.SetRotation(rot);
  EXPECT_EQ(rot, pose.GetRotation());

  // Get/Set Translation (Real3)
  EXPECT_EQ(Real3(0_r, 0_r, 0_r), pose.GetTranslation());
  pose.SetTranslation(Real3(1_r, 2_r, 3_r));
  EXPECT_EQ(Real3(1_r, 2_r, 3_r), pose.GetTranslation());

  // Get/Set Translation (Vec4r)
  EXPECT_EQ(Real4(1_r, 2_r, 3_r, 1_r), ToReal4(pose.VGetTranslation()));
  pose.SetTranslation(Vec4r(4_r, 5_r, 6_r, 12345_r));
  EXPECT_EQ(Real3(4_r, 5_r, 6_r), pose.GetTranslation());
  EXPECT_EQ(Real4(4_r, 5_r, 6_r, 1_r), ToReal4(pose.VGetTranslation()));
}

TEST(TransformRT, Copy) {
  TransformRT pose1(Quaternion::RotationX(kPI), Real3(1_r, 2_r, 3_r));
  TransformRT pose2;
  EXPECT_EQ(TransformRT::Identity(), pose2);
  pose2 = pose1; // assign
  EXPECT_EQ(pose1, pose2);
  TransformRT pose3(pose2); // copy
  EXPECT_EQ(pose1, pose3);
}

TEST(TransformRT, Equal) {
  Quaternion rot = Quaternion::RotationX(kPI);
  Quaternion rot2 = Quaternion::RotationY(0.00001f);
  TransformRT a(rot, Real3{});
  TransformRT b(rot, Real3{}); // same as a
  TransformRT c(rot * rot2, Real3{}); // different rotation
  TransformRT d(rot, Real3{0.00000001f, 0_r, 0_r}); // different translation
  TransformRT e(rot * rot2, Real3{0.00000001f, 0_r, 0_r}); // both

  EXPECT_TRUE(a == a);
  EXPECT_TRUE(a == b);
  EXPECT_TRUE(c == c);
  EXPECT_FALSE(a == c); // different rotation
  EXPECT_FALSE(a == d); // different translation
  EXPECT_FALSE(a == e); // different in both

  EXPECT_FALSE(a != a);
  EXPECT_FALSE(a != b);
  EXPECT_FALSE(c != c);
  EXPECT_TRUE(a != c); // different rotation
  EXPECT_TRUE(a != d); // different translation
  EXPECT_TRUE(a != e); // different in both
}

TEST(TransformRT, NearEqual) {
  Quaternion rot = Quaternion::RotationX(kPI);
  Quaternion rot2 = Quaternion::RotationY(0.00001f);
  TransformRT a(rot, Real3{});
  TransformRT b(rot, Real3{}); // same as a
  TransformRT c(rot * rot2, Real3{}); // different rotation
  TransformRT d(rot, Real3{0.0000001f, 0_r, 0_r}); // different translation
  TransformRT e(rot * rot2, Real3{0.001f, 0_r, 0_r}); // both

  // Zero tolerance
  real eps = 0_r;
  EXPECT_TRUE(NearEqual(a, a, eps));
  EXPECT_TRUE(NearEqual(a, b, eps));
  EXPECT_TRUE(NearEqual(c, c, eps));
  EXPECT_FALSE(NearEqual(a, c, eps)); // different rotation
  EXPECT_FALSE(NearEqual(a, d, eps)); // different translation
  EXPECT_FALSE(NearEqual(a, e, eps)); // different in both

  // Small tolerance
  eps = 1e-5_r;
  EXPECT_TRUE(NearEqual(a, a, eps));
  EXPECT_TRUE(NearEqual(a, b, eps));
  EXPECT_TRUE(NearEqual(c, c, eps));
  EXPECT_TRUE(NearEqual(a, c, eps)); // close enough
  EXPECT_TRUE(NearEqual(a, d, eps)); // close enough
  EXPECT_FALSE(NearEqual(a, e, eps)); // not close enough

  // Larger tolerance
  eps = 1.00001e-3_r;
  EXPECT_TRUE(NearEqual(a, a, eps));
  EXPECT_TRUE(NearEqual(a, b, eps));
  EXPECT_TRUE(NearEqual(c, c, eps));
  EXPECT_TRUE(NearEqual(a, c, eps)); // close enough
  EXPECT_TRUE(NearEqual(a, d, eps)); // close enough
  EXPECT_TRUE(NearEqual(a, e, eps)); // close enough
}

TEST(TransformRT, Invert) {
  // Identity
  EXPECT_EQ(TransformRT::Identity(), Invert(TransformRT::Identity()));

  Quaternion rotPosX = Quaternion::RotationX(1_r);
  Quaternion rotNegX = Quaternion::RotationX(-1_r);
  Real3 trans(1_r, 2_r, 3_r);

  // Invert rotations
  EXPECT_NEAR_EQ(rotNegX, Invert(TransformRT(rotPosX)).GetRotation());
  EXPECT_NEAR_EQ(rotPosX, Invert(TransformRT(rotNegX)).GetRotation());

  // Negate translation
  EXPECT_NEAR_EQ(-trans, Invert(TransformRT(trans)).GetTranslation());
  EXPECT_NEAR_EQ(ToSimd(-trans, 1_r), Invert(TransformRT(trans)).VGetTranslation());
  EXPECT_NEAR_EQ(trans, Invert(TransformRT(-trans)).GetTranslation());
  EXPECT_NEAR_EQ(ToSimd(trans, 1_r), Invert(TransformRT(-trans)).VGetTranslation());

  // Both at once
  TransformRT a(Quaternion::RotationX(kPI / 2_r), Real3(0_r, 1_r, 0_r));
  TransformRT aInv = Invert(a);
  EXPECT_NEAR_EQ(Quaternion::RotationX(-kPI / 2_r), aInv.GetRotation());
  EXPECT_NEAR_EQ(Real3(0_r, 0_r, 1_r), aInv.GetTranslation()); // rotated from (0,1,0)
  EXPECT_NEAR_EQ(Vec4r(0_r, 0_r, 1_r, 1_r), aInv.VGetTranslation());

  // Double invert
  EXPECT_NEAR_EQ(a, Invert(aInv));
}

TEST(TransformRT, NormalizeRotation) {
  // Suppress warnings about non-normalized quaternions
  auto prevLogFn = GetLogCallback();
  SetLogCallback(
      [](LogChannel /*channel*/, char const* /*message*/, char const* /*file*/, int /*line*/) {});
  MOCHI_DEFER(SetLogCallback(prevLogFn));

  auto r = Quaternion{1_r, 2_r, 3_r, 4_r};
  auto t = Real3{5_r, 6_r, 7_r};
  auto rt = TransformRT{r, t};
  EXPECT_EQ(Normalize(r), NormalizeRotation(rt).GetRotation());
  EXPECT_EQ(t, NormalizeRotation(rt).GetTranslation());

  // Zero-in-zero-out
  auto qZero = Quaternion(0_r, 0_r, 0_r, 0_r);
  EXPECT_EQ(qZero, NormalizeRotation(TransformRT{qZero, t}).GetRotation());
  EXPECT_EQ(t, NormalizeRotation(TransformRT{qZero, t}).GetTranslation());
}

TEST(TransformRT, MulTransform) {
  Quaternion aRot = Quaternion::RotationX(kPI / 2_r);
  Quaternion bRot = Quaternion::RotationX(-kPI);
  Real3 aTrans = Real3{0_r, 0_r, 1_r};
  Real3 bTrans = Real3{2_r, 0_r, 0_r};
  TransformRT a{aRot, aTrans};
  TransformRT b{bRot, bTrans};

  // Transform by a, then by b
  TransformRT c = b * a;
  EXPECT_NEAR_EQ(b.GetRotation() * a.GetRotation(), c.GetRotation());
  EXPECT_NEAR_EQ(Quaternion::RotationX(-kPI / 2_r), c.GetRotation()); // +45 deg, then -90 deg
  EXPECT_NEAR_EQ(Real3(2_r, 0_r, -1_r), c.GetTranslation());
  EXPECT_NEAR_EQ(Vec4r(2_r, 0_r, -1_r, 1_r), c.VGetTranslation());

  // Transform by b, then by a
  c = a * b;
  EXPECT_NEAR_EQ(a.GetRotation() * b.GetRotation(), c.GetRotation());
  EXPECT_NEAR_EQ(Quaternion::RotationX(-kPI / 2_r), c.GetRotation()); // -90 deg, then +45 deg
  EXPECT_NEAR_EQ(Real3(2_r, 0_r, 1_r), c.GetTranslation());
  EXPECT_NEAR_EQ(Vec4r(2_r, 0_r, 1_r, 1_r), c.VGetTranslation());

  // More examples
  EXPECT_NEAR_EQ(a, a * TransformRT::Identity());
  EXPECT_NEAR_EQ(a, TransformRT::Identity() * a);
  EXPECT_NEAR_EQ(a * b, a * Invert(a) * a * b * Invert(b) * b);
}

TEST(TransformRT, TransformPoint) {
  Real3 trans{1_r, 2_r, 3_r};
  TransformRT transformX90(Quaternion::RotationX(90_r * kRadiansPerDegree), trans);
  TransformRT transformY90(Quaternion::RotationY(90_r * kRadiansPerDegree), trans);
  TransformRT transformZ90(Quaternion::RotationZ(90_r * kRadiansPerDegree), trans);

  // Rotate point, then add translation
  EXPECT_NEAR_EQ(Real3(1_r, 2_r, 4_r), transformX90.TransformPoint(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(1_r, 3_r, 3_r), transformY90.TransformPoint(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(0_r, 2_r, 3_r), transformZ90.TransformPoint(Real3(0_r, 1_r, 0_r)));

  // Subtract translation, then rotate in the opposite direction
  EXPECT_NEAR_EQ(Real3(-1_r, -3_r, 1_r), Invert(transformX90).TransformPoint(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(3_r, -1_r, -1_r), Invert(transformY90).TransformPoint(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(-1_r, 1_r, -3_r), Invert(transformZ90).TransformPoint(Real3(0_r, 1_r, 0_r)));
}

TEST(TransformRT, TransformDirection) {
  Real3 trans{1_r, 2_r, 3_r}; // not used for transforming direction vectors
  TransformRT transformX90(Quaternion::RotationX(90_r * kRadiansPerDegree), trans);
  TransformRT transformY90(Quaternion::RotationY(90_r * kRadiansPerDegree), trans);
  TransformRT transformZ90(Quaternion::RotationZ(90_r * kRadiansPerDegree), trans);

  // Just rotate
  EXPECT_NEAR_EQ(Real3(0_r, 0_r, 1_r), transformX90.TransformDirection(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(0_r, 1_r, 0_r), transformY90.TransformDirection(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(-1_r, 0_r, 0_r), transformZ90.TransformDirection(Real3(0_r, 1_r, 0_r)));

  // Just rotate in opposite direction
  EXPECT_NEAR_EQ(
      Real3(0_r, 0_r, -1_r), Invert(transformX90).TransformDirection(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(
      Real3(0_r, 1_r, 0_r), Invert(transformY90).TransformDirection(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(
      Real3(1_r, 0_r, 0_r), Invert(transformZ90).TransformDirection(Real3(0_r, 1_r, 0_r)));
}

TEST(TransformRT, TransformPointDirectionVec4r) {
  Real3 const testPt{2_r, 3_r, 4_r};
  Vec4r const pt4 = ToSimd(testPt, 1_r);
  Vec4r const dir4 = ToSimd(testPt, 0_r);
  Real3 const translations[] = {{0_r, 0_r, 0_r}, {1_r, 2_r, 3_r}};
  Real3 const axes[] = {{1_r, 0_r, 0_r}, {0_r, 1_r, 0_r}, {0_r, 0_r, 1_r}, {1_r, 1_r, 1_r}};
  real const anglesDeg[] = {0_r, 20_r, 45_r, 90_r, 180_r, -135_r};
  for (auto const& t : translations) {
    for (auto const& axis : axes) {
      for (real angleDeg : anglesDeg) {
        TransformRT const rt(
            Quaternion::FromAxisAngle(Normalize(axis), angleDeg * kRadiansPerDegree), t);

        // Use Real3 overloads as ground truth for x,y,z. The Vec4r overloads must additionally set
        // w per the affine convention: point → w = 1, direction → w = 0.
        EXPECT_NEAR_EQ(ToSimd(rt.TransformPoint(testPt), 1_r), rt.TransformPoint(pt4));
        EXPECT_NEAR_EQ(ToSimd(rt.TransformDirection(testPt), 0_r), rt.TransformDirection(dir4));
        EXPECT_NEAR_EQ(
            ToSimd(rt.TransformPointInverse(testPt), 1_r), rt.TransformPointInverse(pt4));
        EXPECT_NEAR_EQ(
            ToSimd(rt.TransformDirectionInverse(testPt), 0_r), rt.TransformDirectionInverse(dir4));

        // Round-trip.
        real constexpr kRoundTripTol = 100_r * std::numeric_limits<real>::epsilon();
        EXPECT_NEAR_TOL(pt4, rt.TransformPointInverse(rt.TransformPoint(pt4)), kRoundTripTol);
        EXPECT_NEAR_TOL(pt4, rt.TransformPoint(rt.TransformPointInverse(pt4)), kRoundTripTol);
        EXPECT_NEAR_TOL(
            dir4, rt.TransformDirectionInverse(rt.TransformDirection(dir4)), kRoundTripTol);
        EXPECT_NEAR_TOL(
            dir4, rt.TransformDirection(rt.TransformDirectionInverse(dir4)), kRoundTripTol);
      }
    }
  }
}

TEST(TransformRT, ToFromVMatrix4x4) {
  Real3 const testAxes[] = {
      Real3{0.0_r, 0.0_r, 1.0_r},
      Real3{0.0_r, 1.0_r, 0.0_r},
      Real3{0.0_r, 1.0_r, 1.0_r},
      Real3{1.0_r, 0.0_r, 0.0_r},
      Real3{1.0_r, 0.0_r, 1.0_r},
      Real3{1.0_r, 1.0_r, 0.0_r},
      Real3{1.0_r, 1.0_r, 1.0_r}};
  Real3 const testTranslation = Real3{1.1f, 2.2f, 3.3f};
  real const testAnglesDeg[] = {0_r, 45_r, 90_r, 135_r, 180_r, -45_r, -90_r, -135_r, -180_r};
  for (Real3 const& axis : testAxes) {
    for (real deg : testAnglesDeg) {
      // Construct a TransformRT
      real angle = deg * kRadiansPerDegree;
      Quaternion rot = Quaternion::FromAxisAngle(Normalize(axis), angle);
      TransformRT rt = TransformRT{rot, testTranslation};

      // Convert to matrix
      VMatrix4x4r mat = ToVMatrix4x4(rt); // basis vectors as columns

      // The first 3 columns of the matrix should be the basis vectors
      VMatrix4x4r matTranspose = Transpose4x4(mat);
      EXPECT_NEAR_EQ(rot * Vec4r(1_r, 0_r, 0_r), matTranspose[0]);
      EXPECT_NEAR_EQ(rot * Vec4r(0_r, 1_r, 0_r), matTranspose[1]);
      EXPECT_NEAR_EQ(rot * Vec4r(0_r, 0_r, 1_r), matTranspose[2]);

      // Last column should be (x,y,z,1) translation
      EXPECT_NEAR_EQ(Vec4r(1.1_r, 2.2_r, 3.3_r, 1.0_r), matTranspose[3]);

      // Convert back to TransformRT and compare
      TransformRT rt2 = TransformRT::FromOrthoNormal(mat);
      EXPECT_TRUE(EquivalentRotation(rt.GetRotation(), rt2.GetRotation()));
      EXPECT_NEAR_EQ(rt.GetTranslation(), rt2.GetTranslation());

      Vec4r pt = Vec4r(2_r, 3_r, 4_r, 1_r);
      // DotMatVec4x4 should be equivalent to TransformRT.TransformPoint(Vec4r)
      EXPECT_NEAR_EQ(DotMatVec4x4(mat, pt), rt.TransformPoint(pt));
      // DotVecMat4x4 with a transposed matrix should be equivalent to
      // TransformRT.TransformPoint(Vec4r)
      EXPECT_NEAR_EQ(DotVecMat4x4(pt, matTranspose), rt.TransformPoint(pt));
    }
  }
}

TEST(TransformRT, IsFinite) {
  // Disable warnings about the invalid constructor arguments.
  bool wasWarningEnabled = IsLogChannelEnabled(LogChannel::Warning);
  EnableLogChannel(LogChannel::Warning, false);
  MOCHI_DEFER(EnableLogChannel(LogChannel::Warning, wasWarningEnabled));

  // Test non-finite values in every position of the TransformRT
  for (int i = 0; i < 4; ++i) {
    Quaternion q = {};
    Vec4r t = {};
    EXPECT_TRUE(IsFinite(TransformRT{q, t}));
    q.data = Set(q.data, i, std::numeric_limits<real>::infinity());
    EXPECT_FALSE(IsFinite(TransformRT{q, t}));
    q.data = Set(q.data, i, -std::numeric_limits<real>::infinity());
    EXPECT_FALSE(IsFinite(TransformRT{q, t}));
    q.data = Set(q.data, i, std::numeric_limits<real>::quiet_NaN());
    EXPECT_FALSE(IsFinite(TransformRT{q, t}));
    q.data = Set(q.data, i, std::numeric_limits<real>::signaling_NaN());
    EXPECT_FALSE(IsFinite(TransformRT{q, t}));
    q.data = Set(q.data, i, 123_r);
    EXPECT_TRUE(IsFinite(TransformRT{q, t}));

    // TransformRT accepts Vec4r or Real3 for translation, but it never keeps more than 3 values.
    bool expectFinite = (i == 3);
    t = Set(t, i, std::numeric_limits<real>::infinity());
    EXPECT_EQ(expectFinite, IsFinite(TransformRT{q, t}));
    t = Set(t, i, -std::numeric_limits<real>::infinity());
    EXPECT_EQ(expectFinite, IsFinite(TransformRT{q, t}));
    t = Set(t, i, std::numeric_limits<real>::quiet_NaN());
    EXPECT_EQ(expectFinite, IsFinite(TransformRT{q, t}));
    t = Set(t, i, std::numeric_limits<real>::signaling_NaN());
    EXPECT_EQ(expectFinite, IsFinite(TransformRT{q, t}));
    t = Set(t, i, 123_r);
    EXPECT_TRUE(IsFinite(TransformRT{q, t}));
  }
}

TEST(TransformRT, Reflection) {
  // Any non-default value will due. Values chosen for lossless floating-point representation.
  TransformRT rt(Quaternion{0.5_r, -0.5_r, 0.5_r, -0.5_r}, Real3{-1_r, 0.5_r, 1_r});

  // Serialization
  char const* json = R"({"rotation":[0.5,-0.5,0.5,-0.5],"translation":[-1,0.5,1]})";
  EXPECT_STREQ(json, SReflect::ToJsonString(rt, false).c_str());
  EXPECT_EQ(rt, SReflect::FromJsonString<TransformRT>(json));
}

TEST(TransformRT, DecomposeMatrixTransform) {
  Real3 constexpr kScales[] = {
      Real3{1_r, 1_r, 1_r},
      Real3{0.5_r, 0.5_r, 0.5_r},
      Real3{0.1_r, 0.2_r, 0.3_r},
      Real3{-2_r, 2_r, 2_r},
      Real3{2_r, -2_r, 2_r},
      Real3{2_r, 2_r, -2_r},
      Real3{-0.1_r, -0.2_r, 0.3_r},
      Real3{0.1_r, -0.2_r, -0.3_r},
      Real3{-0.1_r, -0.2_r, -0.3_r},
  };
  Real3 constexpr kAxes[] = {
      Real3{1_r, 0_r, 0_r},
      Real3{0_r, 1_r, 0_r},
      Real3{0_r, 0_r, 1_r},
      Real3{1_r, 1_r, 1_r},
      Real3{1_r, 2_r, 3_r}};
  real constexpr kAngles[] = {0_r, kPI * 0.25_r, kPI * 0.5_r, kPI * -1.1_r, kPI * 1.23_r};
  Real3 constexpr kTranslations[] = {Real3{}, Real3{1.1_r, 2.3_r, 1.3_r}};

  for (Real3 const& scale : kScales) {
    for (Real3 const& axis : kAxes) {
      for (real angle : kAngles) {
        auto rot = Quaternion::FromAxisAngle(Normalize(axis), angle);
        for (Real3 const& trans : kTranslations) {
          // Build a transformation matrix
          VMatrix4x4r expectedMat =
              Dot4x4(ToVMatrix4x4(TransformRT{rot, trans}), VDiagonalMatrix<4>(ToSimd(scale, 1_r)));

          // Decompose it into scale, rotation, and translation.
          auto [newScale, newRT] = DecomposeMatrixTransform(expectedMat);

          // The new values may not be equal to the originals, but we should be able to use them to
          // build the same matrix.
          VMatrix4x4r actualMat =
              Dot4x4(ToVMatrix4x4(newRT), VDiagonalMatrix<4>(ToSimd(newScale, 1_r)));
          EXPECT_NEAR_TOL(expectedMat, actualMat, Vec4r(1e-5_r));
        }
      }
    }
  }
}
