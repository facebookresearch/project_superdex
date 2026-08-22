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

#include <gtest/gtest.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/rodrigues_utils.h>

#include <array>
#include <limits>
#include <tuple>
#include <vector>

using namespace mochi;

static constexpr Real3 kTestVectors[]{
    Real3{0_r, 0_r, 0_r},
    Real3{1_r, 0_r, 0_r},
    Real3{0_r, 1_r, 0_r},
    Real3{0_r, 0_r, 1_r},
    Real3{0.149_r, 0.099_r, 0.941_r},
    Real3{0.164_r, 0.046_r, 0.393_r},
    Real3{0.543_r, 0.028_r, 0.595_r},
    Real3{0.267_r, 0.35_r, 0.042_r}};

TEST(Rodrigues, Rodrigues) {
  static constexpr NdArray<real, 3, 3> kTestRRotVector[]{
      NdArray<real, 3, 3>{
          Real3{1.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 1.0_r, 0.0_r}, Real3{0.0_r, 0.0_r, 1.0_r}},
      NdArray<real, 3, 3>{
          Real3{1.0_r, 0.0_r, 0.0_r},
          Real3{0.0_r, 0.5403023058681398_r, -0.8414709848078965_r},
          Real3{0.0_r, 0.8414709848078965_r, 0.5403023058681398_r}},
      NdArray<real, 3, 3>{
          Real3{0.5403023058681398_r, 0.0_r, 0.8414709848078965_r},
          Real3{0.0_r, 1.0_r, 0.0_r},
          Real3{-0.8414709848078965_r, 0.0_r, 0.5403023058681398_r}},
      NdArray<real, 3, 3>{
          Real3{0.5403023058681398_r, -0.8414709848078965_r, 0.0_r},
          Real3{0.8414709848078965_r, 0.5403023058681398_r, 0.0_r},
          Real3{0.0_r, 0.0_r, 1.0_r}},
      NdArray<real, 3, 3>{
          Real3{0.5855545229562815_r, -0.7967380702803358_r, 0.1494468066283393_r},
          Real3{0.810395186656789_r, 0.5798142937152801_r, -0.08411317522813422_r},
          Real3{-0.019635225716799223_r, 0.17036382294788238_r, 0.9851855793410869_r}},
      NdArray<real, 3, 3>{
          Real3{0.9229070204410048_r, -0.37737837037709615_r, 0.07634263024178534_r},
          Real3{0.3848077375765732_r, 0.9107052893167311_r, -0.15012954776368354_r},
          Real3{-0.012869993081036007_r, 0.16793284843072298_r, 0.985714421978312_r}},
      NdArray<real, 3, 3>{
          Real3{0.8319942650412291_r, -0.5254369426904977_r, 0.17804932517301938_r},
          Real3{0.5398354455149084_r, 0.6927514429858389_r, -0.47819779381209876_r},
          Real3{0.12791885984570614_r, 0.49397515878544_r, 0.8600147137105701_r}},
      NdArray<real, 3, 3>{
          Real3{0.9388739312793432_r, 0.005324060379021431_r, 0.3442199813275688_r},
          Real3{0.08661296276437308_r, 0.9640648723825875_r, -0.2511515807617436_r},
          Real3{-0.33318753855036143_r, 0.26561358440227756_r, 0.9046742440987947_r}}};
  for (int i = 0; i < isize(kTestVectors); ++i) {
    Vec4r p = ToSimd(kTestVectors[i]);
    Matrix3x3r R = ToNdArray3x3(Rodrigues(p));
    EXPECT_NEAR(Norm(R - kTestRRotVector[i]), 0_r, 1e-3_r);
  }
}

TEST(Rodrigues, InvRodrigues) {
  constexpr real kRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-14_r : 1e-5_r;

  for (auto testVector : kTestVectors) {
    Vec4r const p = ToSimd(testVector);
    EXPECT_NEAR_TOL(0_r, Norm(p - InvRodrigues(Rodrigues(p))), kRelTol * Norm(p));
  }

  constexpr Real3 kAxes[] = {
      Real3{1_r, 0_r, 0_r},
      Real3{0_r, 1_r, 0_r},
      Real3{0_r, 0_r, 1_r},
      Real3{1_r, 1_r, 1_r},
      Real3{0_r, 1_r, -1_r},
      Real3{1_r, 0_r, -1_r},
      Real3{-1_r, 1_r, 0_r},
      Real3{1_r, -1_r, 1_r},
  };

  // Test the near-zero branch.
  real const kNearZeroThetas[] = {
      0_r,
      1e-5_r,
      -1e-5_r,
      MOCHI_USE_DOUBLE_PRECISION ? 2e-4_r : 0.03_r, // Just below the thresholds
      MOCHI_USE_DOUBLE_PRECISION ? -2e-4_r : -0.03_r, // Just below the thresholds
  };
  for (real const theta : kNearZeroThetas) {
    for (auto const& a : kAxes) {
      Vec4r const rotVec = theta * ToSimd(Normalize(a));
      EXPECT_NEAR_TOL(0_r, Norm(rotVec - InvRodrigues(Rodrigues(rotVec))), kRelTol * Norm(rotVec));
    }
  }

  // Test the near-π branch.
  real const kNearPiThetas[] = {
      kPI,
      -kPI,
      kPI - 1e-5_r,
      -kPI + 1e-5_r,
      kPI - (MOCHI_USE_DOUBLE_PRECISION ? 2e-4_r : 0.03_r), // Just below the thresholds
      -kPI + (MOCHI_USE_DOUBLE_PRECISION ? 2e-4_r : 0.03_r), // Just below the thresholds
  };
  for (real const theta : kNearPiThetas) {
    for (auto const& a : kAxes) {
      Vec4r const rotVec = theta * ToSimd(Normalize(a));
      Vec4r const recovered = InvRodrigues(Rodrigues(rotVec));

      if (Abs(theta) == kPI) {
        // At exactly ±π, InvRodrigues could return -rotVec depending on the axis orientation
        // resolution.
        real const diff = Min(Norm(rotVec - recovered), Norm(rotVec + recovered));
        EXPECT_NEAR_TOL(0_r, diff, kRelTol * Norm(rotVec));
      } else {
        EXPECT_NEAR_TOL(0_r, Norm(rotVec - recovered), kRelTol * Norm(rotVec));
      }
    }
  }
}

// Test-local periodic delta for rotation vectors, accounting for the 2*pi wrapping discontinuity
// in VToRotationVector() near angle = pi. Remaps endRotVec so that the delta from startRotVec is
// the shortest path in rotation-vector space.
static Vec4r PeriodicDeltaRotVector(Vec4r startRotVec, Vec4r endRotVec) {
  if (auto const endAngleSqr = NormSqr<3>(endRotVec);
      endAngleSqr > Sqr(kDefaultNearEqualEpsilon<real>)) {
    real const endAngle = Sqrt(endAngleSqr);
    Vec4r const endDir = endRotVec / endAngle;
    if (Dot<3>(endRotVec - startRotVec, endDir) > kPI) {
      endRotVec -= (2_r * kPI) * endDir;
    }
  }
  return endRotVec - startRotVec;
}

static void TestDRotVectorDRotIncrement(Vec4r rotVec) {
  rotVec = RotVectorPiCap(rotVec);
  Quaternion q = Quaternion::FromRotationVector(rotVec);
  real constexpr kEps = 1e-2_r;

  VMatrix3x3r DaaDinc_FD = {};
  for (int i = 0; i < 3; ++i) {
    // Use one-sided finite differences to ensure we use rotVec as the reference
    Real3 delta = {};
    delta[i] += kEps;
    Quaternion qtestp = Quaternion::FromRotationVector(delta) * q;
    Vec4r rotVecPlus = qtestp.VToRotationVector();
    DaaDinc_FD[i] = PeriodicDeltaRotVector(rotVec, rotVecPlus) / kEps;
  }
  DaaDinc_FD = Transpose3x3(DaaDinc_FD);

  VMatrix3x3r DaaDinc_A = DRotVectorDRotIncrement(rotVec);
  real normFD = Norm3x3(DaaDinc_FD);
  VMatrix3x3r DaaDinc_Err = DaaDinc_A - DaaDinc_FD;
  EXPECT_NEAR(Norm3x3(DaaDinc_Err), 0_r, 1e-2_r * normFD);
}

static void TestDRotIncrementDRotVector(Vec4r rotVec) {
  rotVec = RotVectorPiCap(rotVec);
  Quaternion q = Quaternion::FromRotationVector(rotVec);
  real constexpr kEps = 1e-2_r;

  VMatrix3x3r DincDaa_FD = {};
  for (int i = 0; i < 3; ++i) {
    Real3 rotVect = ToReal3(rotVec);

    //+
    rotVect[i] += kEps;
    Quaternion qtestp = Quaternion::FromRotationVector(rotVect);
    Quaternion qincp = Normalize(qtestp * q.GetConjugate());
    Vec4r incp = qincp.VToRotationVector();

    //-
    rotVect[i] -= 2 * kEps;
    Quaternion qtestm = Quaternion::FromRotationVector(rotVect);
    Quaternion qincm = Normalize(qtestm * q.GetConjugate());
    Vec4r incm = qincm.VToRotationVector();

    DincDaa_FD[i] = PeriodicDeltaRotVector(incm, incp) / (2 * kEps);
  }
  DincDaa_FD = Transpose3x3(DincDaa_FD);

  VMatrix3x3r DincDaa_A = DRotIncrementDRotVector(rotVec);
  real normFD = Norm3x3(DincDaa_FD);
  VMatrix3x3r DincDaa_Err = DincDaa_A - DincDaa_FD;
  EXPECT_NEAR(Norm3x3(DincDaa_Err), 0_r, 1e-2_r * normFD);
}

TEST(Rodrigues, DRotVectorDRotIncrement) {
  real k90 = kPI / 2_r;
  real k180 = kPI;
  real k270 = 3 * kPI / 2_r;

  // No rotation
  TestDRotVectorDRotIncrement(Vec4r(0_r, 0_r, 0_r));

  // Small, which redirects to drotvector::DSmallRDTheta
  Vec4r r{0.5_r * Sqrt(drotvector::kThresholdDRotVectorDRotIncrement<real> / 3_r)};
  EXPECT_TRUE(NormSqr<3>(r) < drotvector::kThresholdDRotVectorDRotIncrement<real>);
  TestDRotVectorDRotIncrement(r);

  // Positive
  TestDRotVectorDRotIncrement(Vec4r(k90, 0_r, 0_r));
  TestDRotVectorDRotIncrement(Vec4r(0_r, k90, 0_r));
  TestDRotVectorDRotIncrement(Vec4r(0_r, 0_r, k90));
  TestDRotVectorDRotIncrement(Vec4r(k180, 0_r, 0_r));
  TestDRotVectorDRotIncrement(Vec4r(0_r, k180, 0_r));
  TestDRotVectorDRotIncrement(Vec4r(0_r, 0_r, k180));
  TestDRotVectorDRotIncrement(Vec4r(k270, 0_r, 0_r));
  TestDRotVectorDRotIncrement(Vec4r(0_r, k270, 0_r));
  TestDRotVectorDRotIncrement(Vec4r(0_r, 0_r, k270));

  // Negative
  TestDRotVectorDRotIncrement(-Vec4r(k90, 0_r, 0_r));
  TestDRotVectorDRotIncrement(-Vec4r(0_r, k90, 0_r));
  TestDRotVectorDRotIncrement(-Vec4r(0_r, 0_r, k90));
  TestDRotVectorDRotIncrement(-Vec4r(k180, 0_r, 0_r));
  TestDRotVectorDRotIncrement(-Vec4r(0_r, k180, 0_r));
  TestDRotVectorDRotIncrement(-Vec4r(0_r, 0_r, k180));
  TestDRotVectorDRotIncrement(-Vec4r(k270, 0_r, 0_r));
  TestDRotVectorDRotIncrement(-Vec4r(0_r, k270, 0_r));
  TestDRotVectorDRotIncrement(-Vec4r(0_r, 0_r, k270));

  // Combinations
  TestDRotVectorDRotIncrement(Vec4r(k90, -k90, 0_r));
  TestDRotVectorDRotIncrement(Vec4r(0_r, k90, -k90));
  TestDRotVectorDRotIncrement(Vec4r(k90, k90, k90));
  TestDRotVectorDRotIncrement(Vec4r(k180, -k180, 0_r));
  TestDRotVectorDRotIncrement(Vec4r(0_r, k180, -k180));
  TestDRotVectorDRotIncrement(Vec4r(k180, k180, k180));
  TestDRotVectorDRotIncrement(Vec4r(k270, -k270, 0_r));
  TestDRotVectorDRotIncrement(Vec4r(0_r, k270, -k270));
  TestDRotVectorDRotIncrement(Vec4r(k270, k270, k270));

  // Random
  real k170 = 170_r * kRadiansPerDegree;
  auto generator = RandomGenerator(42);
  std::array<Real3, 20> points{};
  SetRandom(generator, -k170, k170, MakeSpan(points));
  for (int i = 0; i < 20; ++i) {
    TestDRotVectorDRotIncrement(ToSimd(points[i]));
  }
}

TEST(Rodrigues, DRotIncrementDRotVector) {
  real k90 = kPI / 2_r;
  real k180 = kPI;
  real k270 = 3 * kPI / 2_r;

  // No rotation
  TestDRotIncrementDRotVector(Vec4r(0_r, 0_r, 0_r));

  // Small, which redirects to drotvector::DThetaDSmallR
  Vec4r r{0.5_r * Sqrt(drotvector::kThresholdDRotIncrementDRotVector<real> / 3_r)};
  EXPECT_TRUE(NormSqr<3>(r) < drotvector::kThresholdDRotIncrementDRotVector<real>);
  TestDRotVectorDRotIncrement(r);

  // Positive
  TestDRotIncrementDRotVector(Vec4r(k90, 0_r, 0_r));
  TestDRotIncrementDRotVector(Vec4r(0_r, k90, 0_r));
  TestDRotIncrementDRotVector(Vec4r(0_r, 0_r, k90));
  TestDRotIncrementDRotVector(Vec4r(k180, 0_r, 0_r));
  TestDRotIncrementDRotVector(Vec4r(0_r, k180, 0_r));
  TestDRotIncrementDRotVector(Vec4r(0_r, 0_r, k180));
  TestDRotIncrementDRotVector(Vec4r(k270, 0_r, 0_r));
  TestDRotIncrementDRotVector(Vec4r(0_r, k270, 0_r));
  TestDRotIncrementDRotVector(Vec4r(0_r, 0_r, k270));

  // Negative
  TestDRotIncrementDRotVector(-Vec4r(k90, 0_r, 0_r));
  TestDRotIncrementDRotVector(-Vec4r(0_r, k90, 0_r));
  TestDRotIncrementDRotVector(-Vec4r(0_r, 0_r, k90));
  TestDRotIncrementDRotVector(-Vec4r(k180, 0_r, 0_r));
  TestDRotIncrementDRotVector(-Vec4r(0_r, k180, 0_r));
  TestDRotIncrementDRotVector(-Vec4r(0_r, 0_r, k180));
  TestDRotIncrementDRotVector(-Vec4r(k270, 0_r, 0_r));
  TestDRotIncrementDRotVector(-Vec4r(0_r, k270, 0_r));
  TestDRotIncrementDRotVector(-Vec4r(0_r, 0_r, k270));

  // Combinations
  TestDRotIncrementDRotVector(Vec4r(k90, -k90, 0_r));
  TestDRotIncrementDRotVector(Vec4r(0_r, k90, -k90));
  TestDRotIncrementDRotVector(Vec4r(k90, k90, k90));
  TestDRotIncrementDRotVector(Vec4r(k180, -k180, 0_r));
  TestDRotIncrementDRotVector(Vec4r(0_r, k180, -k180));
  TestDRotIncrementDRotVector(Vec4r(k180, k180, k180));
  TestDRotIncrementDRotVector(Vec4r(k270, -k270, 0_r));
  TestDRotIncrementDRotVector(Vec4r(0_r, k270, -k270));
  TestDRotIncrementDRotVector(Vec4r(k270, k270, k270));

  // Random
  auto generator = RandomGenerator(42);
  std::array<Real3, 20> points{};
  SetRandom(generator, -kPI, kPI, MakeSpan(points));
  for (int i = 0; i < 20; ++i) {
    TestDRotIncrementDRotVector(ToSimd(points[i]));
  }
}

template <typename ApproxFunc, typename FullFunc, typename GoldFunc>
static bool IsApproximationMoreAccurate(
    double normSqr,
    ApproxFunc approxFunc,
    FullFunc fullFunc,
    GoldFunc goldFunc) {
  double val = Sqrt(normSqr / 3.0);
  Vec4d rotVecd(val);
  auto resultGold = goldFunc(rotVecd, Rodrigues(rotVecd));
  Vec4f rotVecf(static_cast<float>(val));
  auto resultApprox = StaticCast<VMatrix3x3d>(approxFunc(rotVecf));
  auto resultFull = StaticCast<VMatrix3x3d>(fullFunc(rotVecf, Rodrigues(rotVecf)));
  real errorApprox = Norm3x3(resultApprox - resultGold);
  real errorFull = Norm3x3(resultFull - resultGold);
  return errorApprox < errorFull;
}

template <typename ApproxFunc, typename FullFunc, typename GoldFunc>
static void
TestThreshold(ApproxFunc approxFunc, FullFunc fullFunc, GoldFunc goldFunc, real threshold) {
  // Geometric bisection search for the turning point
  double maxVal = 1e-3;
  double minVal = 1e-5;
  EXPECT_FALSE(IsApproximationMoreAccurate(maxVal, approxFunc, fullFunc, goldFunc));
  EXPECT_TRUE(IsApproximationMoreAccurate(minVal, approxFunc, fullFunc, goldFunc));
  double midVal = Sqrt(maxVal * minVal);
  for (int i = 0; i < 10; ++i) {
    if (IsApproximationMoreAccurate(midVal, approxFunc, fullFunc, goldFunc)) {
      minVal = midVal;
    } else {
      maxVal = midVal;
    }
    midVal = Sqrt(maxVal * minVal);
  }
  EXPECT_FALSE(IsApproximationMoreAccurate(maxVal, approxFunc, fullFunc, goldFunc));
  EXPECT_TRUE(IsApproximationMoreAccurate(minVal, approxFunc, fullFunc, goldFunc));

  // Validate that the threshold is within the interval
  EXPECT_LT(threshold, static_cast<real>(maxVal));
  EXPECT_GT(threshold, static_cast<real>(minVal));
}

TEST(Rodrigues, DRotVectorThresholds) {
  TestThreshold(
      drotvector::DSmallRDTheta<float>,
      drotvector::DLargeRDTheta<float>,
      drotvector::DLargeRDTheta<double>,
      drotvector::kThresholdDRotVectorDRotIncrement<real>);
  TestThreshold(
      drotvector::DThetaDSmallR<float>,
      drotvector::DThetaDLargeR<float>,
      drotvector::DThetaDLargeR<double>,
      drotvector::kThresholdDRotIncrementDRotVector<real>);
}

TEST(Rodrigues, SincNearZero) {
  // In float, x=0.01 is inside the Taylor branch (x⁴ = 1e-8 < float epsilon).
  // In double, x=0.01 uses the trig formula (x⁴ = 1e-8 > double epsilon).
  float constexpr kRelTol = std::numeric_limits<float>::epsilon();
  float constexpr x = 0.01f;
  float constexpr x4 = Sqr(Sqr(x));
  static_assert(x4 < std::numeric_limits<float>::epsilon());
  static_assert(x4 > std::numeric_limits<double>::epsilon());
  auto floatResult = Sinc<float>(x);
  auto doubleResult = static_cast<float>(Sinc<double>(x));
  EXPECT_NEAR_RTOL(floatResult, doubleResult, kRelTol);
}
