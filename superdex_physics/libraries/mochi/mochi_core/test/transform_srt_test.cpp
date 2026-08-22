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
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/transform_srt.h>

#include <limits>

using namespace mochi;

static_assert(alignof(TransformSRT) == alignof(Vec4r), "Unexpected alignment");
static_assert(sizeof(TransformSRT) == sizeof(real) * 8, "Unexpected padding");
static_assert(std::is_trivially_copyable_v<TransformSRT>);

TEST(TransformSRT, Constructors) {
  // Default = identity
  {
    TransformSRT srt;
    EXPECT_EQ(1_r, srt.GetScale());
    EXPECT_EQ(Quaternion::Identity(), srt.GetRotation());
    EXPECT_EQ(Real3(0_r, 0_r, 0_r), srt.GetTranslation());
    EXPECT_EQ(Real4(0_r, 0_r, 0_r, 1_r), ToReal4(srt.VGetTranslation()));
    EXPECT_EQ(TransformSRT::Identity(), srt);
  }

  // TransformRT (implicit conversion allowed)
  {
    TransformRT rt(Quaternion::RotationX(kPI), Real3{1_r, 2_r, 3_r});
    TransformSRT srt = rt;
    EXPECT_EQ(1_r, srt.GetScale());
    EXPECT_EQ(rt, srt.GetTransformRT());
  }

  // real, TransformRT
  {
    real scale = 42_r;
    TransformRT rt(Quaternion::RotationX(kPI), Real3{1_r, 2_r, 3_r});
    TransformSRT srt(scale, rt);
    EXPECT_EQ(scale, srt.GetScale());
    EXPECT_EQ(rt, srt.GetTransformRT());
  }

  // real, Quaternion, Vec4r
  {
    real scale = 42_r;
    Quaternion rot = Quaternion::RotationX(kPI);
    Vec4r trans = Vec4r(1_r, 2_r, 3_r, 99999_r); // w will be forced to 1.0
    TransformSRT srt(scale, rot, trans);
    EXPECT_EQ(scale, srt.GetScale());
    EXPECT_EQ(rot, srt.GetRotation());
    EXPECT_EQ(Real3(1_r, 2_r, 3_r), srt.GetTranslation());
    EXPECT_EQ(Real4(1_r, 2_r, 3_r, 1_r), ToReal4(srt.VGetTranslation()));
  }

  // real, Quaternion, Real3
  {
    real scale = 42_r;
    Quaternion rot = Quaternion::RotationX(kPI);
    Real3 trans(1_r, 2_r, 3_r);
    TransformSRT srt(scale, rot, trans);
    EXPECT_EQ(scale, srt.GetScale());
    EXPECT_EQ(rot, srt.GetRotation());
    EXPECT_EQ(trans, srt.GetTranslation());
    EXPECT_EQ(Real4(1_r, 2_r, 3_r, 1_r), ToReal4(srt.VGetTranslation()));
  }
}

TEST(TransformSRT, Accessors) {
  TransformSRT srt;
  Quaternion rot = Quaternion::RotationX(kPI);

  // Get/Set Scale
  EXPECT_EQ(1_r, srt.GetScale());
  srt.SetScale(42_r);
  EXPECT_EQ(42_r, srt.GetScale());

  // Get/Set Rotation
  EXPECT_EQ(Quaternion::Identity(), srt.GetRotation());
  srt.SetRotation(rot);
  EXPECT_EQ(rot, srt.GetRotation());

  // Get/Set Translation (Real3)
  EXPECT_EQ(Real3(0_r, 0_r, 0_r), srt.GetTranslation());
  srt.SetTranslation(Real3(1_r, 2_r, 3_r));
  EXPECT_EQ(Real3(1_r, 2_r, 3_r), srt.GetTranslation());

  // Get/Set Translation (Vec4r)
  EXPECT_EQ(Real4(1_r, 2_r, 3_r, 1_r), ToReal4(srt.VGetTranslation()));
  srt.SetTranslation(Vec4r(4_r, 5_r, 6_r, 12345_r));
  EXPECT_EQ(Real3(4_r, 5_r, 6_r), srt.GetTranslation());
  EXPECT_EQ(Real4(4_r, 5_r, 6_r, 1_r), ToReal4(srt.VGetTranslation()));

  // Get Packed
  EXPECT_EQ(Real4(4_r, 5_r, 6_r, 42_r), ToReal4(srt.VGetPackedTranslationAndScale()));

  // Get TransformRT
  EXPECT_EQ(TransformRT(srt.GetRotation(), srt.GetTranslation()), srt.GetTransformRT());
}

TEST(TransformSRT, Copy) {
  TransformSRT srt1(42_r, Quaternion::RotationX(kPI), Real3(1_r, 2_r, 3_r));
  TransformSRT srt2;
  EXPECT_EQ(TransformSRT::Identity(), srt2);
  srt2 = srt1; // assign
  EXPECT_EQ(srt1, srt2);
  TransformSRT srt3(srt2); // copy
  EXPECT_EQ(srt1, srt3);
}

TEST(TransformSRT, Equal) {
  Quaternion rot = Quaternion::RotationX(kPI);
  Quaternion rot2 = Quaternion::RotationY(0.00001f);
  TransformSRT a(42_r, rot, Real3{});
  TransformSRT b(42_r, rot, Real3{}); // same as a
  TransformSRT c(42_r, rot * rot2, Real3{}); // different rotation
  TransformSRT d(42_r, rot, Real3{0.00000001f, 0_r, 0_r}); // different translation
  TransformSRT e(42_r, rot * rot2, Real3{0.00000001f, 0_r, 0_r}); // different rot & trans
  TransformSRT f(99_r, rot, Real3{}); // different scale
  TransformSRT g(99_r, rot * rot2, Real3{0.00000001f, 0_r, 0_r}); // different everything

  EXPECT_TRUE(a == a);
  EXPECT_TRUE(a == b);
  EXPECT_TRUE(c == c);
  EXPECT_FALSE(a == c); // different rotation
  EXPECT_FALSE(a == d); // different translation
  EXPECT_FALSE(a == e); // different rotation & translation
  EXPECT_FALSE(a == f); // different scale
  EXPECT_FALSE(a == g); // different everything

  EXPECT_FALSE(a != a);
  EXPECT_FALSE(a != b);
  EXPECT_FALSE(c != c);
  EXPECT_TRUE(a != c); // different rotation
  EXPECT_TRUE(a != d); // different translation
  EXPECT_TRUE(a != e); // different rotation & translation
  EXPECT_TRUE(a != f); // different scale
  EXPECT_TRUE(a != g); // different everything
}

TEST(TransformSRT, NearEqual) {
  Quaternion rot = Quaternion::RotationX(kPI);
  Quaternion rot2 = Quaternion::RotationY(0.00001f);
  TransformSRT a(42_r, rot, Real3{});
  TransformSRT b(42_r, rot, Real3{}); // same as a
  TransformSRT c(42_r, rot * rot2, Real3{}); // different rotation
  TransformSRT d(42_r, rot, Real3{0.0000001f, 0_r, 0_r}); // different translation
  TransformSRT e(42_r, rot * rot2, Real3{0.001f, 0_r, 0_r}); // both
  TransformSRT f(42.0001_r, rot, Real3{}); // different scale

  // Zero tolerance
  real eps = 0_r;
  EXPECT_TRUE(NearEqual(a, a, eps));
  EXPECT_TRUE(NearEqual(a, b, eps));
  EXPECT_TRUE(NearEqual(c, c, eps));
  EXPECT_FALSE(NearEqual(a, c, eps)); // different rotation
  EXPECT_FALSE(NearEqual(a, d, eps)); // different translation
  EXPECT_FALSE(NearEqual(a, e, eps)); // different rotation & translation
  EXPECT_FALSE(NearEqual(a, f, eps)); // different scale

  // Small tolerance
  eps = 1e-5_r;
  EXPECT_TRUE(NearEqual(a, a, eps));
  EXPECT_TRUE(NearEqual(a, b, eps));
  EXPECT_TRUE(NearEqual(c, c, eps));
  EXPECT_TRUE(NearEqual(a, c, eps)); // close enough
  EXPECT_TRUE(NearEqual(a, d, eps)); // close enough
  EXPECT_FALSE(NearEqual(a, e, eps)); // not close enough
  EXPECT_FALSE(NearEqual(a, f, eps)); // not close enough

  // Larger tolerance
  eps = 1.00001e-3_r;
  EXPECT_TRUE(NearEqual(a, a, eps));
  EXPECT_TRUE(NearEqual(a, b, eps));
  EXPECT_TRUE(NearEqual(c, c, eps));
  EXPECT_TRUE(NearEqual(a, c, eps)); // close enough
  EXPECT_TRUE(NearEqual(a, d, eps)); // close enough
  EXPECT_TRUE(NearEqual(a, e, eps)); // close enough
  EXPECT_TRUE(NearEqual(a, f, eps)); // close enough
}

TEST(TransformSRT, Invert) {
  // Identity
  EXPECT_EQ(TransformSRT::Identity(), Invert(TransformSRT::Identity()));

  Quaternion rotPosX = Quaternion::RotationX(1_r);
  Quaternion rotNegX = Quaternion::RotationX(-1_r);
  Real3 trans(1_r, 2_r, 3_r);

  // Invert rotations
  EXPECT_NEAR_EQ(rotNegX, Invert(TransformSRT(1_r, rotPosX, Real3{})).GetRotation());
  EXPECT_NEAR_EQ(rotPosX, Invert(TransformSRT(1_r, rotNegX, Real3{})).GetRotation());

  // Negate translation
  EXPECT_NEAR_EQ(-trans, Invert(TransformSRT(1_r, Quaternion{}, trans)).GetTranslation());
  EXPECT_NEAR_EQ(
      ToSimd(-trans, 1_r), Invert(TransformSRT(1_r, Quaternion{}, trans)).VGetTranslation());
  EXPECT_NEAR_EQ(trans, Invert(TransformSRT(1_r, Quaternion{}, -trans)).GetTranslation());
  EXPECT_NEAR_EQ(
      ToSimd(trans, 1_r), Invert(TransformSRT(1_r, Quaternion{}, -trans)).VGetTranslation());

  // All at once
  TransformSRT a(42_r, Quaternion::RotationX(kPI / 2_r), Real3(0_r, 1_r, 0_r));
  TransformSRT aInv = Invert(a);
  EXPECT_NEAR_EQ(1_r / 42_r, aInv.GetScale());
  EXPECT_NEAR_EQ(Quaternion::RotationX(-kPI / 2_r), aInv.GetRotation());
  EXPECT_NEAR_EQ(Real3(0_r, 0_r, 1_r / 42_r), aInv.GetTranslation()); // rotated from (0,1,0)
  EXPECT_NEAR_EQ(Vec4r(0_r, 0_r, 1_r / 42_r, 1_r), aInv.VGetTranslation());

  // Double invert
  EXPECT_NEAR_EQ(a, Invert(aInv));
}

TEST(TransformSRT, MulTransform) {
  real aScale = 42_r;
  real bScale = 0.1_r;
  Quaternion aRot = Quaternion::RotationX(kPI / 2_r);
  Quaternion bRot = Quaternion::RotationX(-kPI);
  Real3 aTrans = Real3{0_r, 0_r, 1_r};
  Real3 bTrans = Real3{2_r, 0_r, 0_r};
  TransformSRT a{aScale, aRot, aTrans};
  TransformSRT b{bScale, bRot, bTrans};

  // Transform by a, then by b
  TransformSRT c = b * a;
  EXPECT_NEAR_EQ(aScale * bScale, c.GetScale());
  EXPECT_NEAR_EQ(b.GetRotation() * a.GetRotation(), c.GetRotation());
  EXPECT_NEAR_EQ(Quaternion::RotationX(-kPI / 2_r), c.GetRotation()); // +45 deg, then -90 deg
  EXPECT_NEAR_EQ(Real3(2_r, 0_r, -0.1_r), c.GetTranslation());
  EXPECT_NEAR_EQ(Vec4r(2_r, 0_r, -0.1_r, 1_r), c.VGetTranslation());

  // Transform by b, then by a
  c = a * b;
  EXPECT_NEAR_EQ(a.GetRotation() * b.GetRotation(), c.GetRotation());
  EXPECT_NEAR_EQ(Quaternion::RotationX(-kPI / 2_r), c.GetRotation()); // -90 deg, then +45 deg
  EXPECT_NEAR_EQ(Real3(84_r, 0_r, 1_r), c.GetTranslation());
  EXPECT_NEAR_EQ(Vec4r(84_r, 0_r, 1_r, 1_r), c.VGetTranslation());

  // More examples
  EXPECT_NEAR_EQ(a, a * TransformSRT::Identity());
  EXPECT_NEAR_EQ(a, TransformSRT::Identity() * a);
  EXPECT_NEAR_EQ(a * b, a * Invert(a) * a * b * Invert(b) * b);
}

TEST(TransformSRT, TransformPoint) {
  Real3 trans{1_r, 2_r, 3_r};
  TransformSRT transformX90(10_r, Quaternion::RotationX(90_r * kRadiansPerDegree), trans);
  TransformSRT transformY90(10_r, Quaternion::RotationY(90_r * kRadiansPerDegree), trans);
  TransformSRT transformZ90(10_r, Quaternion::RotationZ(90_r * kRadiansPerDegree), trans);

  // Rotate point, then add translation
  EXPECT_NEAR_EQ(Real3(1_r, 2_r, 13_r), transformX90.TransformPoint(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(1_r, 12_r, 3_r), transformY90.TransformPoint(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(-9_r, 2_r, 3_r), transformZ90.TransformPoint(Real3(0_r, 1_r, 0_r)));

  // Subtract translation, then rotate in the opposite direction
  EXPECT_NEAR_EQ(
      Real3(-0.1_r, -0.3_r, 0.1_r), Invert(transformX90).TransformPoint(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(
      Real3(0.3_r, -0.1_r, -0.1_r), Invert(transformY90).TransformPoint(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(
      Real3(-0.1_r, 0.1_r, -0.3_r), Invert(transformZ90).TransformPoint(Real3(0_r, 1_r, 0_r)));
}

TEST(TransformSRT, TransformDirection) {
  Real3 trans{1_r, 2_r, 3_r}; // not used for transforming direction vectors
  TransformSRT transformX90(10_r, Quaternion::RotationX(90_r * kRadiansPerDegree), trans);
  TransformSRT transformY90(10_r, Quaternion::RotationY(90_r * kRadiansPerDegree), trans);
  TransformSRT transformZ90(10_r, Quaternion::RotationZ(90_r * kRadiansPerDegree), trans);

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

TEST(TransformSRT, IsFinite) {
  // Disable warnings about the invalid constructor arguments.
  bool wasWarningEnabled = IsLogChannelEnabled(LogChannel::Warning);
  EnableLogChannel(LogChannel::Warning, false);
  MOCHI_DEFER(EnableLogChannel(LogChannel::Warning, wasWarningEnabled));

  // Test non-finite values in every position of the TransformSRT
  for (int i = 0; i < 4; ++i) {
    real s = 1_r;
    Quaternion q = {};
    Vec4r t = {};

    EXPECT_TRUE(IsFinite(TransformSRT{s, q, t}));
    s = std::numeric_limits<real>::infinity();
    EXPECT_FALSE(IsFinite(TransformSRT{s, q, t}));
    s = -std::numeric_limits<real>::infinity();
    EXPECT_FALSE(IsFinite(TransformSRT{s, q, t}));
    s = std::numeric_limits<real>::quiet_NaN();
    EXPECT_FALSE(IsFinite(TransformSRT{s, q, t}));
    s = std::numeric_limits<real>::signaling_NaN();
    EXPECT_FALSE(IsFinite(TransformSRT{s, q, t}));
    s = 123_r;
    EXPECT_TRUE(IsFinite(TransformSRT{s, q, t}));

    q.data = Set(q.data, i, std::numeric_limits<real>::infinity());
    EXPECT_FALSE(IsFinite(TransformSRT{s, q, t}));
    q.data = Set(q.data, i, -std::numeric_limits<real>::infinity());
    EXPECT_FALSE(IsFinite(TransformSRT{s, q, t}));
    q.data = Set(q.data, i, std::numeric_limits<real>::quiet_NaN());
    EXPECT_FALSE(IsFinite(TransformSRT{s, q, t}));
    q.data = Set(q.data, i, std::numeric_limits<real>::signaling_NaN());
    EXPECT_FALSE(IsFinite(TransformSRT{s, q, t}));
    q.data = Set(q.data, i, 123_r);
    EXPECT_TRUE(IsFinite(TransformSRT{s, q, t}));

    // TransformSRT accepts Vec4r or Real3 for translation, but it never keeps more than 3 values.
    bool expectFinite = (i == 3);
    EXPECT_TRUE(IsFinite(TransformSRT{s, q, t}));
    t = Set(t, i, std::numeric_limits<real>::infinity());
    EXPECT_EQ(expectFinite, IsFinite(TransformSRT{s, q, t}));
    t = Set(t, i, -std::numeric_limits<real>::infinity());
    EXPECT_EQ(expectFinite, IsFinite(TransformSRT{s, q, t}));
    t = Set(t, i, std::numeric_limits<real>::quiet_NaN());
    EXPECT_EQ(expectFinite, IsFinite(TransformSRT{s, q, t}));
    t = Set(t, i, std::numeric_limits<real>::signaling_NaN());
    EXPECT_EQ(expectFinite, IsFinite(TransformSRT{s, q, t}));
    t = Set(t, i, 123_r);
    EXPECT_TRUE(IsFinite(TransformSRT{s, q, t}));
  }
}

TEST(TransformSRT, Reflection) {
  // Any non-default value will due. Values chosen for lossless floating-point representation.
  TransformSRT srt(0.25_r, Quaternion{0.5_r, -0.5_r, 0.5_r, -0.5_r}, Real3{-1_r, 0.5_r, 1_r});

  // Serialization (json implementation writes keys in alphabetical order)
  char const* json = R"({"rotation":[0.5,-0.5,0.5,-0.5],"scale":0.25,"translation":[-1,0.5,1]})";
  EXPECT_STREQ(json, SReflect::ToJsonString(srt, false).c_str());
  EXPECT_EQ(srt, SReflect::FromJsonString<TransformSRT>(json));
}
