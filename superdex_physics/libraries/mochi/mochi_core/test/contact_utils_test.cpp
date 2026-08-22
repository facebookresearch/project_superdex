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

#include <mochi_core/contact/contact_utils.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/rand_utils.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace mochi;

// Helper to compare two ContactDetectionResult that should be equal
static void ExpectEqualResults(
    ContactDetectionResult const& a,
    ContactDetectionResult const& b,
    bool allowRotationAgnosticGsd = false) {
  // Equal size arrays
  EXPECT_EQ(a.sampleIndices.size(), b.sampleIndices.size());
  EXPECT_EQ(a.posColliding.size(), b.posColliding.size());
  EXPECT_EQ(a.sdfInfo.size(), b.sdfInfo.size());
  EXPECT_EQ(a.normalColliding.size(), b.normalColliding.size());

  // Equal array contents
  EXPECT_EQ(a.isSdfGradUnitary, b.isSdfGradUnitary);
  for (size_t i = 0; i < a.sampleIndices.size(); ++i) {
    EXPECT_EQ(a.sampleIndices[i], b.sampleIndices[i]);
    EXPECT_NEAR_EQ(a.sdfInfo.val[i], b.sdfInfo.val[i]);
    if (a.isSdfGradUnitary) {
      EXPECT_NEAR_EQ(1_r, Norm(a.sdfInfo.grad[i]));
    }
    if (b.isSdfGradUnitary) {
      EXPECT_NEAR_EQ(1_r, Norm(b.sdfInfo.grad[i]));
    }
    if (allowRotationAgnosticGsd) {
      // Allow for the possibility that sdfGrad might point in different directions
    } else {
      real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 1e-4_r;
      EXPECT_LT(1_r - kTol, Dot(a.sdfInfo.grad[i], b.sdfInfo.grad[i]));
    }
    EXPECT_NEAR_TOL(a.posColliding[i], b.posColliding[i], 5e-5_r);
  }
}

// Test N points using the ComputeBatchCollisionForceDForce specialization with batch size of N.
template <int N, GradTarget kGradTarget>
void TestPlaneContactAtPointN(
    ContactEvalConfig const& config,
    ContactParams const& params,
    Real3 pos,
    Real3 prevPos,
    Real3 ppoint,
    Real3 pnormal,
    real eps,
    real tolGrad,
    real tolHess,
    real dt) {
  ASSERT_TRUE(NearEqual(Norm(pnormal), 1_r)); // The logic below assumes pnormal is unitary.
  real d = Dot(pos - ppoint, pnormal);
  if (d + eps > params.GetPenaltyThresholdDist(false)) {
    return; // Point not colliding
  }

  // Introduce a small rotation to the plane normal between previous and current evaluations
  Real3 pnormal0 = Quaternion::FromRotationVector(Real3{0.1_r, -0.2_r, 0.1_r}) * pnormal;

  // Compute analytic
  DynamicArray<Real3> posColliding(N);
  DynamicArray<Real3> posColliding0(N);
  SdfInfo sdf;
  sdf.resize(N);
  SdfInfo sdf0;
  sdf0.resize(N);
  DynamicArray<Real3> normalColliding(N);
  Real3 forceAn[N] = {};
  [[maybe_unused]] VMatrix3x3r dforceAn[N] = {};
  for (int k = 0; k < N; ++k) {
    posColliding[k] = pos;
    sdf.grad[k] = pnormal;
    sdf.val[k] = d;

    posColliding0[k] = prevPos;
    sdf0.val[k] = Dot(prevPos - ppoint, pnormal0);
    sdf0.grad[k] = pnormal0;

    normalColliding[k] = -pnormal;
  }
  ComputeBatchCollisionForceDForce<N, kGradTarget>(
      {} /*outEnergy*/,
      MakeSpan(forceAn),
      MakeSpan(dforceAn),
      MakeConstSpan(sdf.val),
      MakeConstSpan(sdf.grad),
      config.explicitNormals ? MakeConstSpan(sdf0.val) : Span<real const>{},
      config.explicitNormals ? MakeConstSpan(sdf0.grad) : Span<Real3 const>{},
      MakeConstSpan(normalColliding),
      MakeConstSpan(posColliding),
      MakeConstSpan(posColliding0),
      params,
      config,
      dt,
      /*assemEnergy*/ false,
      /*assemForce*/ true,
      /*assemDForce*/ true,
      /*isSdfGradUnitary*/ true);

  [[maybe_unused]] VMatrix3x3r dforceFD[N];
  Real3 forceFD[N];
  for (int i = 0; i < 3; ++i) {
    auto evalEps = [&](real eps, Real3(&force)[N], double (&energy)[N]) {
      Real3 newPos = pos;
      Real3 newPrevPos = prevPos;
      Real3& varPos = kGradTarget == GradTarget::Current ? newPos : newPrevPos;
      varPos[i] += eps;
      for (int k = 0; k < N; ++k) {
        posColliding[k] = newPos;
        posColliding0[k] = newPrevPos;
        sdf.val[k] = Dot(newPos - ppoint, pnormal);
        sdf0.val[k] = Dot(newPrevPos - ppoint, pnormal0);
      }
      ComputeBatchCollisionForceDForce<N, kGradTarget>(
          MakeSpan(energy),
          {} /* outForce */,
          {} /* outDForce*/,
          MakeConstSpan(sdf.val),
          MakeConstSpan(sdf.grad),
          config.explicitNormals ? MakeConstSpan(sdf0.val) : Span<real const>{},
          config.explicitNormals ? MakeConstSpan(sdf0.grad) : Span<Real3 const>{},
          MakeConstSpan(normalColliding),
          MakeConstSpan(posColliding),
          MakeConstSpan(posColliding0),
          params,
          config,
          dt,
          /*assemEnergy*/ true,
          /*assemForce*/ false,
          /*assemDForce*/ false,
          /*isSdfGradUnitary*/ true);
      ComputeBatchCollisionForceDForce<N, GradTarget::Current>(
          {} /* outEnergy */,
          MakeSpan(force),
          {} /* outDForce*/,
          MakeConstSpan(sdf.val),
          MakeConstSpan(sdf.grad),
          config.explicitNormals ? MakeConstSpan(sdf0.val) : Span<real const>{},
          config.explicitNormals ? MakeConstSpan(sdf0.grad) : Span<Real3 const>{},
          MakeConstSpan(normalColliding),
          MakeConstSpan(posColliding),
          MakeConstSpan(posColliding0),
          params,
          config,
          dt,
          /*assemEnergy*/ false,
          /*assemForce*/ true,
          /*assemDForce*/ false,
          /*isSdfGradUnitary*/ true);
    };

    // +
    Real3 forcep[N] = {};
    double energyp[N] = {};
    evalEps(eps, forcep, energyp);

    //-
    Real3 forcem[N] = {};
    double energym[N] = {};
    evalEps(-eps, forcem, energym);

    // Estimate
    for (int k = 0; k < N; ++k) {
      dforceFD[k][i] = ToSimd((forcep[k] - forcem[k]) / (2 * eps));
      forceFD[k][i] = -(energyp[k] - energym[k]) / (2 * eps);
    }
  }

  // Check results.
  for (int k = 0; k < N; ++k) {
    dforceFD[k] = Transpose3x3(dforceFD[k]);
    real dfAnNorm = Norm3x3(dforceAn[k]);
    real dfFdNorm = Norm3x3(dforceFD[k]);
    real dfDiffNorm = Norm3x3(dforceFD[k] - dforceAn[k]);
    real constexpr kToleranceFactor = 1.25_r; // Some compilers need a larger tolerance
    EXPECT_NEAR(
        dfDiffNorm, 0.0_r, kToleranceFactor * Max(tolHess, tolHess * Max(dfAnNorm, dfFdNorm)));

    real fAnNorm = Norm(forceAn[k]);
    real fFdNorm = Norm(forceFD[k]);
    real fDiffNorm = Norm(forceFD[k] - forceAn[k]);
    if (fDiffNorm > 0_r) {
      EXPECT_NEAR(fDiffNorm, 0.0_r, Max(tolGrad, tolGrad * Max(fAnNorm, fFdNorm)));
    }
  }
}

template <GradTarget kGradTarget>
static void TestPlaneContactAtPoint(
    ContactEvalConfig const& config,
    ContactParams const& params,
    Real3 pos,
    Real3 prevPos,
    Real3 ppoint,
    Real3 pnormal,
    real eps,
    real tolGrad,
    real tolHess,
    real dt) {
  // Test all specializations of ComputeBatchCollisionForceDForce.
  static_assert(kCollResponseMaxBatchSize == 8, "Please update unit tests to cover all codepaths");
  TestPlaneContactAtPointN<1, kGradTarget>(
      config, params, pos, prevPos, ppoint, pnormal, eps, tolGrad, tolHess, dt);
  TestPlaneContactAtPointN<2, kGradTarget>(
      config, params, pos, prevPos, ppoint, pnormal, eps, tolGrad, tolHess, dt);
  TestPlaneContactAtPointN<3, kGradTarget>(
      config, params, pos, prevPos, ppoint, pnormal, eps, tolGrad, tolHess, dt);
  TestPlaneContactAtPointN<4, kGradTarget>(
      config, params, pos, prevPos, ppoint, pnormal, eps, tolGrad, tolHess, dt);
  TestPlaneContactAtPointN<5, kGradTarget>(
      config, params, pos, prevPos, ppoint, pnormal, eps, tolGrad, tolHess, dt);
  TestPlaneContactAtPointN<6, kGradTarget>(
      config, params, pos, prevPos, ppoint, pnormal, eps, tolGrad, tolHess, dt);
  TestPlaneContactAtPointN<7, kGradTarget>(
      config, params, pos, prevPos, ppoint, pnormal, eps, tolGrad, tolHess, dt);
  TestPlaneContactAtPointN<8, kGradTarget>(
      config, params, pos, prevPos, ppoint, pnormal, eps, tolGrad, tolHess, dt);
}

template <GradTarget kGradTarget>
void TestPlaneContactFromPoint(
    ContactEvalConfig const& config,
    ContactParams const& params,
    Real3 const& prevPos,
    Real3 const& ppoint,
    Real3 const& pnormal,
    int N) {
  // Relative error allowed in the gradient and Hessian: 1% in double precision, 5% in single. This
  // is relatively high for a finite differences test. Due to the approximations in the friction
  // force (for the explicit normal force and projection terms), and accumulation of numerical
  // error, it's hard to accurately estimate this Hessian using finite differences for some
  // displacements. We manually checked the results and analytic Hessian entries seem consistent
  // with FD ones, just slighly off sometimes.
  real constexpr kTolGrad = MOCHI_USE_DOUBLE_PRECISION ? 1e-2_r : 5e-2_r;
  real constexpr kTolHessDefault = MOCHI_USE_DOUBLE_PRECISION ? 1e-2_r : 5e-2_r;

  // Tests for GradTarget::Previous Hessian for normal dissipation are not robust in single
  // precision.
  real const tolHess = (kGradTarget == GradTarget::Previous && !MOCHI_USE_DOUBLE_PRECISION &&
                        params.normalViscousDampingCoefficient != 0)
      ? 3e-1_r
      : kTolHessDefault;

  real constexpr dt = 0.01_r;
  real range = params.coulombFrictionCoefficient == 0 ? params.GetPenaltyThresholdDist(false)
                                                      : params.frictionFalloffVel * dt;
  real kEps = Max(1e-9_r, 1e-3_r * range);

  // Test displacements orthogonal to normal
  Matrix2x3r basis;
  auto generator = RandomGenerator(42);
  Real3 rand;
  SetRandom(generator, -1_r, 1_r, rand);
  basis[0] = Normalize(Cross(pnormal, rand));
  basis[1] = Cross(pnormal, basis[0]);

  for (real s = -2 * range; s < 2 * range; s += 4 * range / N) {
    for (real t = -2 * range; t < 2 * range; t += 4 * range / N) {
      Real3 disp = s * basis[0] + t * basis[1];

      // Friction dforce cannot be robustly approximated with finite
      // differences for zero tangent displacements due to saddle point
      if (Norm(disp) < 1e-9) {
        continue;
      }

      Real3 pos = prevPos + disp;
      TestPlaneContactAtPoint<kGradTarget>(
          config, params, pos, prevPos, ppoint, pnormal, kEps, kTolGrad, tolHess, dt);
    }
  }

  // Test displacements in a sphere around
  Matrix3x3r P = DiagonalMatrix<3>(1_r) - Outer(pnormal, pnormal);

  for (real r = 2 * range / N; r < 2 * range; r += 2 * range / N) {
    for (real s = 0; s < kPI; s += kPI / 5) {
      for (real t = 0; t < 2 * kPI; t += 2 * kPI / N) {
        Real3 disp =
            Real3(r * std::cos(t) * std::sin(s), r * std::sin(t) * std::cos(s), r * std::cos(s));

        // Friction dforce cannot be robustly approximated with finite
        // differences for zero tangent displacements due to saddle point
        if (Norm(DotMatVec(P, disp)) < 1e-9) {
          continue;
        }

        Real3 pos = prevPos + disp;
        TestPlaneContactAtPoint<kGradTarget>(
            config, params, pos, prevPos, ppoint, pnormal, kEps, kTolGrad, tolHess, dt);
      }
    }
  }
}

template <GradTarget kGradTarget>
static void TestPlaneContact(
    ContactEvalConfig const& config,
    ContactParams const& paramsIn,
    Real3 const& pnormal) {
  int constexpr N = 3;
  Real3 const ppoint = {0_r, 0_r, 0_r};
  ContactParams params = paramsIn;
  params.penaltyCoefficient = 1e8_r;
  params.penaltyThresholdDefault = 0.001_r;
  params.penaltySmoothingHalfDistance = 0.001_r;

  // Test plane contact from a point lying on the plane
  TestPlaneContactFromPoint<kGradTarget>(config, params, ppoint, ppoint, pnormal, N);

  // Test plane contact from a point in a sphere around
  real range = params.GetPenaltyThresholdDist(false);

  for (real r = range / N; r < range; r += range / N) {
    for (real s = 0; s < kPI; s += kPI / N) {
      for (real t = 0; t < 2 * kPI; t += 2 * kPI / N) {
        Real3 disp =
            Real3{r * std::cos(t) * std::sin(s), r * std::sin(t) * std::cos(s), r * std::cos(s)};
        TestPlaneContactFromPoint<kGradTarget>(config, params, ppoint + disp, ppoint, pnormal, N);
      }
    }
  }
}

TEST(ContactPenalties, ConsistencyTests) {
  // Two different sets of contact evaluation config
  std::array<ContactEvalConfig, 2> config;
  // Implicit normals. Forbid friction fading. Do not use the fitted Hessian.
  config[0].explicitNormals = false;
  config[0].fadeFriction = false;
  config[0].useFittedHessian = false;
  // Explicit normals. Allow friction fading. Do not use the fitted Hessian.
  config[1].explicitNormals = true;
  config[1].fadeFriction = true;
  config[1].useFittedHessian = false;

  // Four different sets of contact parameters
  std::array<ContactParams, 4> params;
  // No friction
  params[0].coulombFrictionCoefficient = 0.0_r;
  params[0].viscousFrictionCoefficient = 0.0_r;
  // IPC friction
  params[1].frictionFalloffVel = 0.01_r;
  params[1].viscousFrictionCoefficient = 0.0_r;
  params[1].coulombFrictionCoefficient = 0.5_r;
  // Viscous friction
  params[2].viscousFrictionCoefficient = 1.0_r;
  params[2].coulombFrictionCoefficient = 0.0_r;
  // Normal viscous damping
  params[3].normalViscousDampingCoefficient = 1_r;
  params[3].viscousFrictionCoefficient = 0.0_r;
  params[3].coulombFrictionCoefficient = 0.0_r;

  std::array<Real3, 6> normals = {
      // Axis-aligned
      Real3(1_r, 0_r, 0_r),
      Real3(0_r, 1_r, 0_r),
      Real3(0_r, 0_r, 1_r),
      // Non-axis-aligned
      Normalize(Real3(1_r, 0_r, 1_r)),
      Normalize(Real3(-1_r, 0_r, 1_r)),
      Normalize(Real3(-1_r, 1_r, 1_r))};

  for (auto const& c : config) {
    for (auto const& p : params) {
      for (auto const& n : normals) {
        TestPlaneContact<GradTarget::Current>(c, p, n);
        if (c.explicitNormals) {
          TestPlaneContact<GradTarget::Previous>(c, p, n);
        }
      }
    }
  }
}

TEST(ContactPenalties, FadeFriction_DegenerateAlignmentUsesUnfadedFactor) {
  ContactEvalConfig fadedConfig;
  fadedConfig.fadeFriction = true;
  fadedConfig.validCollidingNormals = true;

  ContactEvalConfig unfadedConfig = fadedConfig;
  unfadedConfig.fadeFriction = false;

  ContactParams params;
  params.maxAlignmentNormals = -1_r;
  params.coulombFrictionCoefficient = 0.5_r;
  params.frictionFalloffVel = 0.01_r;

  real const distance[1] = {-0.001_r};
  Real3 const distanceGrad[1] = {Real3{0_r, 0_r, 1_r}};
  Real3 const normalColliding[1] = {Real3{0_r, 0_r, -1_r}};
  Real3 const posColliding[1] = {Real3{0_r, 0_r, 0_r}};
  Real3 const posCollidingStageStart[1] = {Real3{0.001_r, 0_r, 0_r}};

  auto const eval = [&](ContactEvalConfig const& config) {
    double energy[1] = {};
    Real3 force[1] = {};
    ComputeBatchCollisionForceDForce<1, GradTarget::Current>(
        MakeSpan(energy),
        MakeSpan(force),
        {} /* outDForce */,
        MakeConstSpan(distance),
        MakeConstSpan(distanceGrad),
        {} /* distanceStageStart */,
        {} /* distanceGradStageStart */,
        MakeConstSpan(normalColliding),
        MakeConstSpan(posColliding),
        MakeConstSpan(posCollidingStageStart),
        params,
        config,
        1_r /* dtFactor */,
        true /* assemEnergy */,
        true /* assemForce */,
        false /* assemDForce */,
        true /* isSdfGradUnitary */);
    return std::pair{energy[0], force[0]};
  };

  auto const [unfadedEnergy, unfadedForce] = eval(unfadedConfig);
  EXPECT_TRUE(IsFinite(unfadedEnergy) && IsFinite(unfadedForce));
  EXPECT_NE(unfadedForce[0], 0_r);

  auto const [fadedEnergy, fadedForce] = eval(fadedConfig);
  EXPECT_DOUBLE_EQ(unfadedEnergy, fadedEnergy);
  EXPECT_NEAR_EQ(unfadedForce, fadedForce);

  fadedConfig.validCollidingNormals = false;
  auto const [invalidNormalEnergy, invalidNormalForce] = eval(fadedConfig);
  EXPECT_DOUBLE_EQ(unfadedEnergy, invalidNormalEnergy);
  EXPECT_NEAR_EQ(unfadedForce, invalidNormalForce);
}

// Expect that two vectors point in the same direction +/- some angle
static void ExpectNearEqNormals(Real3 const& a, Real3 const& b, real angleTol) {
  real dot = Dot(Normalize(a), Normalize(b));
  EXPECT_LE(std::cos(angleTol), dot);
}

// Wraps each call FindPointContactsT to perform some common checks
template <typename ShapeT>
static void TestPointContactsImpl(
    Span<Real3 const> samplePoints,
    ShapeT const& shape,
    ContactDetectionParams const& params,
    TransformRT const& pointsFromCollider,
    ContactDetectionResult& outResult) {
  FindPointContactsT(
      samplePoints,
      &shape,
      params,
      pointsFromCollider,
      outResult.sampleIndices,
      outResult.posColliding,
      outResult.sdfInfo,
      outResult.isSdfGradUnitary);
  if (outResult.sampleIndices.empty()) {
    EXPECT_EQ(0, outResult.sampleIndices.size());
    EXPECT_EQ(0, outResult.posColliding.size());
    EXPECT_EQ(0, outResult.sdfInfo.size());
  } else {
    // Expect valid sample indices
    int const numHits = isize(outResult.sampleIndices);
    EXPECT_NE(0, numHits);
    for (int i : outResult.sampleIndices) {
      EXPECT_TRUE((i >= 0) && (i < isize(samplePoints)));
    }

    // Expect no duplicate indices
    auto indices = outResult.sampleIndices;
    std::sort(indices.begin(), indices.end());
    EXPECT_TRUE(std::unique(indices.begin(), indices.end()) == indices.end());

    // Expect an equal number of contact info structs with valid values
    EXPECT_EQ(outResult.sampleIndices.size(), outResult.sdfInfo.size());

    // Check unit SDF gradient
    for (int i = 0; i < numHits; ++i) {
      EXPECT_NEAR_EQ(1_r, Norm(outResult.sdfInfo.grad[i]));
    }
  }
}

// Wraps TestPointContactsImpl and tests with various output transforms
template <typename ShapeT>
static void TestPointContacts(
    Span<Real3 const> samplePoints,
    ShapeT const& shape,
    ContactDetectionParams const& params,
    ContactDetectionResult& outResult,
    bool allowRotationAgnosticGsd = false) {
  TestPointContactsImpl(samplePoints, shape, params, TransformRT{}, outResult);

  // Repeat with a non-identity transform
  auto angles = Real3{0.5_r * kPI, 0.25_r * kPI, -kPI};
  auto translation = Real3{0.1_r, -0.2_r, 0.3_r};
  auto resultTransform = TransformRT{Quaternion::FromRotationVector(angles), translation};
  DynamicArray<Real3> samplePoints2;
  samplePoints2.resize_noinit(isize(samplePoints));
  ArrayTransformPoints(MakeSpan(samplePoints2), samplePoints, resultTransform);
  ContactDetectionResult result2;
  TestPointContactsImpl(samplePoints2, shape, params, resultTransform, result2);
  EXPECT_EQ(outResult.sampleIndices.empty(), result2.sampleIndices.empty());

  // Compare to the first set of results.
  ExpectEqualResults(outResult, result2, allowRotationAgnosticGsd);
}

// Used for Obb and for box-shaped meshes.
template <typename ShapeT>
static void TestPointContactsBox(
    ShapeT const& shape,
    Quaternion rot,
    Real3 center,
    Real3 halfExt,
    ContactDetectionParams const& params,
    real angleEpsilon,
    bool testCenterAndCorners,
    bool expectUnitNormal = true) {
  TransformRT const shapeTransform{rot, center};

  // Test the exact center
  if (testCenterAndCorners) {
    ContactDetectionResult result;
    TestPointContacts({&center, 1}, shape, params, result, /*allowRotationAgnosticGsd*/ true);
    EXPECT_TRUE(!result.sampleIndices.empty());

    // When the sample point is in the exact center, there may be multiple equally valid results.
    // Therefore, this test simply expects that the point will be somewhere on the surface.
    auto const& sdf = result.sdfInfo;
    Obb implicitBox{TransformRT{rot, center}, halfExt};

    // Box colliders always give unit normals.
    EXPECT_TRUE(result.isSdfGradUnitary);
    EXPECT_NEAR_EQ(1_r, Norm(sdf.grad[0]));
    auto posCollider = result.posColliding[0] - sdf.grad[0] * sdf.val[0];

    EXPECT_TRUE(ContainsPoint(implicitBox,
                              posCollider - 1e-5_r * sdf.grad[0])); // just inside
    EXPECT_FALSE(ContainsPoint(implicitBox,
                               posCollider + 1e-5_r * sdf.grad[0])); // just outside
  }

  // Test the corners & edges
  if (testCenterAndCorners) {
    constexpr int kNumPoints = 8 + 12;
    // clang-format off
   constexpr Real3 kReferencePoints[kNumPoints] = {
       // Corners
       {-1_r, -1_r, -1_r},
       {-1_r, -1_r,  1_r},
       {-1_r,  1_r, -1_r},
       {-1_r,  1_r,  1_r},
       { 1_r, -1_r, -1_r},
       { 1_r, -1_r,  1_r},
       { 1_r,  1_r, -1_r},
       { 1_r,  1_r,  1_r},

       // Center of each edge
       { 0_r,  1_r,  1_r },
       { 0_r,  1_r, -1_r },
       { 0_r, -1_r,  1_r },
       { 0_r, -1_r, -1_r },
       { 1_r,  0_r,  1_r },
       { 1_r,  0_r, -1_r },
       {-1_r,  0_r,  1_r },
       {-1_r,  0_r, -1_r },
       { 1_r,  1_r,  0_r },
       { 1_r, -1_r,  0_r },
       {-1_r,  1_r,  0_r },
       {-1_r, -1_r,  0_r },
   };
    // clang-format on
    Real3 points[kNumPoints];
    real nudge = 0_r;
    bool expectHighPrecision = (rot == Quaternion{} && center == Real3{});
    if (expectHighPrecision) {
      // Test the EXACT corners. This case should be precise.
      for (int i = 0; i < kNumPoints; ++i) {
        points[i] = center + (kReferencePoints[i] * halfExt);
      }
    } else {
      // Test points just outside the corner, because we don't know what the normal will be inside.
      nudge = 1e-4_r;
      for (int i = 0; i < kNumPoints; ++i) {
        points[i] = shapeTransform.TransformPoint(
            (kReferencePoints[i] * halfExt) + (Normalize(kReferencePoints[i]) * nudge));
      }
    }

    ContactDetectionResult result;
    TestPointContacts(
        points, shape, params, result, /*allowRotationAgnosticGsd*/ expectHighPrecision);
    EXPECT_EQ(kNumPoints, result.sampleIndices.size());
    EXPECT_EQ(kNumPoints, result.sdfInfo.size());
    for (int i = 0; i < kNumPoints; ++i) {
      EXPECT_EQ(i, result.sampleIndices[i]);
      EXPECT_NEAR(nudge, result.sdfInfo.val[i], 1e-5_r);
      // The normal is not well defined at the exact corner. In practice, it will be one of the face
      // normals if the point is on the exact corner, or the outward diagonal if the point was
      // nudged far enough outside of the box. Therefore, this part of the test will accept any unit
      // vector pointing away from the box.
      EXPECT_NEAR_EQ(1_r, Norm(result.sdfInfo.grad[i]));
      EXPECT_LT(0_r, Dot(result.sdfInfo.grad[i], result.posColliding[i] - center));
    }

    // Move the points farther out so that they are barely within tolerance
    nudge = params.tolerance - 1e-5_r;
    for (int i = 0; i < kNumPoints; ++i) {
      points[i] = shapeTransform.TransformPoint(
          (kReferencePoints[i] * halfExt) + (Normalize(kReferencePoints[i]) * nudge));
    }
    result = {};
    TestPointContacts(points, shape, params, result);

    // Expect different distances, but the same normals
    EXPECT_EQ(kNumPoints, result.sampleIndices.size());
    EXPECT_EQ(kNumPoints, result.sdfInfo.size());
    for (int i = 0; i < kNumPoints; ++i) {
      EXPECT_EQ(i, result.sampleIndices[i]);
      EXPECT_NEAR(nudge, result.sdfInfo.val[i], 1e-5_r);
      // The normal is well defined in this case becaue the point was outside of the box (not
      // exactly on the corner of the box). The normal should point outward along the diagonal.
      Real3 expectedNormal = Normalize(shapeTransform.TransformDirection(kReferencePoints[i]));
      ExpectNearEqNormals(expectedNormal, result.sdfInfo.grad[i], angleEpsilon);
    }

    // Move the points just outside of tolerance and expect no hits
    nudge = params.tolerance + 1e-5_r;
    for (int i = 0; i < kNumPoints; ++i) {
      Real3 point = kReferencePoints[i] * halfExt;
      Real3 nudgeDir = kReferencePoints[i];
      points[i] = shapeTransform.TransformPoint(point + nudgeDir * nudge);
    }
    result = {};
    TestPointContacts(points, shape, params, result);
    EXPECT_TRUE(result.sampleIndices.empty());
  }

  // Test points on the flat faces
  {
    struct Sample {
      Real3 position;
      Real3 normal;
    };
    constexpr int kNumSamples = 12;
    constexpr Sample kSamples[kNumSamples] = {
        // Center of each face
        {{1_r, 0_r, 0_r}, {1_r, 0_r, 0_r}},
        {{0_r, 1_r, 0_r}, {0_r, 1_r, 0_r}},
        {{0_r, 0_r, 1_r}, {0_r, 0_r, 1_r}},
        {{-1_r, 0_r, 0_r}, {-1_r, 0_r, 0_r}},
        {{0_r, -1_r, 0_r}, {0_r, -1_r, 0_r}},
        {{0_r, 0_r, -1_r}, {0_r, 0_r, -1_r}},

        // Some off-center points on the faces
        {{1_r, 0.1_r, 0.1_r}, {1_r, 0_r, 0_r}},
        {{0.1_r, 1_r, 0.1_r}, {0_r, 1_r, 0_r}},
        {{0.1_r, 0.1_r, 1_r}, {0_r, 0_r, 1_r}},
        {{-1_r, 0.1_r, 0.1_r}, {-1_r, 0_r, 0_r}},
        {{0.1_r, -1_r, 0.1_r}, {0_r, -1_r, 0_r}},
        {{0.1_r, 0.1_r, -1_r}, {0_r, 0_r, -1_r}},
    };
    real const minExt = *std::min_element(halfExt.begin(), halfExt.end());
    real const kOffsets[] = {
        -minExt * 0.5_r, // well inside the OBB
        -1e-6_r, // barely inside
        0_r, // on surface
        1e-6_r, // barely outside
        params.tolerance - 1e-5_r, // barely within tolerance
        params.tolerance + 1e-5_r, // barely outside tolerance
    };
    for (real offset : kOffsets) {
      Real3 points[kNumSamples];
      for (int i = 0; i < kNumSamples; ++i) {
        points[i] = shapeTransform.TransformPoint(
            (kSamples[i].position * halfExt) + (kSamples[i].normal * offset));
      }
      ContactDetectionResult result;
      TestPointContacts(points, shape, params, result);
      if (offset > params.tolerance) {
        EXPECT_TRUE(result.sampleIndices.empty()); // All points should be too far away
      } else {
        EXPECT_EQ(expectUnitNormal, result.isSdfGradUnitary);
        EXPECT_EQ(kNumSamples, result.sampleIndices.size());
        EXPECT_EQ(kNumSamples, result.sdfInfo.size());
        for (int i = 0; i < kNumSamples; ++i) {
          EXPECT_EQ(i, result.sampleIndices[i]);
          EXPECT_NEAR(offset, result.sdfInfo.val[i], 1e-5_r);
          Real3 ptOnSurface = shapeTransform.TransformPoint(kSamples[i].position * halfExt);
          EXPECT_NEAR_EQ(1_r, Norm(result.sdfInfo.grad[i]));
          EXPECT_NEAR_TOL(ptOnSurface, ToReal3(result.GetApproxPosCollider(i)), 1e-5_r);
          ExpectNearEqNormals(
              shapeTransform.TransformDirection(kSamples[i].normal),
              result.sdfInfo.grad[i],
              1e-3_r);
        }
      }
    }
  }
}

// Generate a list of rotations to test
static std::vector<Quaternion> GetRotationsToTest() {
  std::vector<Quaternion> rotations;
  constexpr real kAnglesDeg[] = {30_r, 60_r};
  constexpr size_t kNumAngles = std::size(kAnglesDeg);
  rotations.reserve(kNumAngles * kNumAngles * kNumAngles);
  rotations.emplace_back();
  for (float i : kAnglesDeg) {
    for (float j : kAnglesDeg) {
      for (float k : kAnglesDeg) {
        rotations.push_back(
            Quaternion::FromAxisAngle(Vec4r(0_r, 0_r, 1_r), k * kRadiansPerDegree) *
            Quaternion::FromAxisAngle(Vec4r(0_r, 1_r, 0_r), j * kRadiansPerDegree) *
            Quaternion::FromAxisAngle(Vec4r(1_r, 0_r, 0_r), i * kRadiansPerDegree));
      }
    }
  }
  return rotations;
}

// Used for Obb and for box-shaped meshes
template <typename ShapeT>
static void TestPointContactsBoxVariations(
    std::function<ShapeT(Quaternion rotation, Real3 center, Real3 halfExt)> const& factory,
    real maxAngleEpsilon,
    bool testCenterAndCorners,
    std::vector<Quaternion> rotationsToTest) {
  // No points
  {
    ShapeT box = factory(Quaternion{}, Real3{}, Real3{1_r, 1_r, 1_r});
    ContactDetectionResult result;
    TestPointContacts({}, box, {}, result);
    EXPECT_TRUE(result.sampleIndices.empty());
  }

  // Repeat all of these tests for a variety of OBB positions, rotations, and dimensions.
  constexpr Real3 kTranslations[] = {{0_r, 0_r, 0_r}, {-0.44_r, 0.55_r, -0.66_r}};
  constexpr Real3 kHalfExts[] = {{1_r, 1_r, 1_r}, {0.2_r, 0.3_r, 0.4_r}};
  ContactDetectionParams params;
  params.tolerance = 0.01_r; // penalty falloff
  if (rotationsToTest.empty()) {
    rotationsToTest.push_back(Quaternion::Identity());
  }
  for (int i = 0; i < 2; ++i) {
    params.useAccelerationStructures = (i != 0);
    for (Real3 center : kTranslations) {
      for (Real3 halfExt : kHalfExts) {
        for (Quaternion rotation : rotationsToTest) {
          ShapeT shape = factory(rotation, center, halfExt);
          bool expectHighPrecision = (rotation == Quaternion{} && center == Real3{});
          real angleEps = expectHighPrecision ? 0.001_r : maxAngleEpsilon;

          // Run tests for this combination of inputs
          TestPointContactsBox(
              shape, rotation, center, halfExt, params, angleEps, testCenterAndCorners);
        }
      }
    }
  }
}

TEST(MochiContact, FindPointContacts_Obb) {
  // Run a series of tests that expect an oriented box. Use this factory function whenever the tests
  // need us to create a new shape.
  auto factory = [](Quaternion rotation, Real3 center, Real3 halfExt) {
    return Obb{TransformRT{rotation, center}, halfExt};
  };
  constexpr real kAngleTolForRotatedCases = 0.025_r;
  TestPointContactsBoxVariations<Obb>(
      factory, kAngleTolForRotatedCases, true, GetRotationsToTest());
}

static std::shared_ptr<TriangularMesh const>
CreateTriangularMeshBox(Quaternion rotation, Real3 center, Real3 halfExt) {
  /**
    Create a box with center at (0,0,0)

          2 ------- 3
        / |       / |
       /  |      /  |
      6 ------- 7   |
      |   0 ----|-- 1
      |  /      |  /
      | /       | /
      4 ------- 5
  */
  auto [coordinates, connectivity] = test::CreateMinimalTriMeshUnitCube();
  TransformRT rt(rotation, center);
  for (Real3& coord : coordinates) {
    coord -= Real3{0.5_r, 0.5_r, 0.5_r}; // center on origin
    coord *= 2.0_r * halfExt; // scale so that the AABB spans from -halfExt to +halfExt
    coord = rt.TransformPoint(coord); // rotate and translate
  }
  return std::make_unique<TriangularMesh>(coordinates, connectivity);
}

TEST(MochiContact, FindPointContacts_BoxMesh) {
  // This function creates a box-shaped MeshCollider with specificed parameters.
  auto factory = [](Quaternion rotation, Real3 center, Real3 halfExt) -> MeshCollider {
    MeshCollider result(CreateTriangularMeshBox(rotation, center, halfExt));
    result.Initialize();
    return result;
  };

  // Re-use the tests for oriented boxes. This is not sufficient to cover all MeshCollider behavior,
  // but it is a good start. TODO: Look into reducing floating-point round-off error in the
  // MeshCollider implementation (compared to the Obb).
  constexpr real kAngleTolForRotatedCases = 0.06_r; // about 3.5 deg
  TestPointContactsBoxVariations<MeshCollider>(
      factory, kAngleTolForRotatedCases, true, GetRotationsToTest());
}

// Helper function to load the corase duck mesh
static std::unique_ptr<TetrahedralMesh> LoadDuckTetMesh() {
  std::string filePath = test::GetAssetPath("duck/duck_coarse_mesh.mochi.json");
  return LoadTetrahedralMesh(filePath, test::ExpectOK{});
}

// Helper function to determine which tetrahedrons a range of points occupy
// Assumes outIndices is initially filled with -1
static void FindElementsContainingPoints(
    TetrahedralMesh const& mesh,
    Span<Real3 const> points,
    Span<int> outIndices) {
  EXPECT_EQ(points.size(), outIndices.size());
  constexpr int kMinPerTask = 32; // arbitrary guess
  Span<Real3 const> coordinates = mesh.GetNodeCoordinates();
  Span<Int4 const> connectivity = mesh.GetElementConnectivity();
  ParallelForN(
      "FindElementsContainingPoints", mesh.GetNumElements(), kMinPerTask, [&](int elemIdx) {
        Int4 tet = connectivity[elemIdx];
        Real3 v[4] = {
            coordinates[tet[0]], coordinates[tet[1]], coordinates[tet[2]], coordinates[tet[3]]};
        for (int i = 0; i < isize(points); ++i) {
          if (IsInsideTetrahedron(v[0], v[1], v[2], v[3], points[i])) {
            outIndices[i] = elemIdx;
          }
        }
      });
}

// The Duck mesh is not shipped externally.
TEST_IF(MOCHI_INTERNAL, MochiContact, FindPointContacts_DuckMesh) {
  // This test loads a tetrahedral mesh shaped like a rubber duck. It then calls FindPointContacts
  // to compute the signed distance at every point in a 3D grid so that it can check the results
  // using alternate calculations.
  TaskScheduler scheduler;
  auto tetMesh = LoadDuckTetMesh();
  auto triMesh = std::make_shared<TriangularMesh const>(CreateBoundaryMesh(*tetMesh));
  MeshCollider triMeshCollider{triMesh};
  triMeshCollider.Initialize();

  // Generate sample points that fill the volume of the Aabb with some padding.
  constexpr real kPadding = 0.1_r;
  constexpr int kGridSize = 10;
  constexpr int kNumPoints = kGridSize * kGridSize * kGridSize;
  Aabb const gridBounds = ExpandShape(tetMesh->GetAabb(), kPadding);
  Real3 const gridMin = gridBounds.GetMin();
  Real3 const delta = gridBounds.GetSize() / static_cast<real>(kGridSize - 1);
  std::vector<Real3> points(kNumPoints);
  Real3 pt = {};
  int idx = 0;
  for (int x = 0; x < kGridSize; ++x) {
    pt[0] = gridMin[0] + (static_cast<real>(x) * delta[0]);
    for (int y = 0; y < kGridSize; ++y) {
      pt[1] = gridMin[1] + (static_cast<real>(y) * delta[1]);
      for (int z = 0; z < kGridSize; ++z) {
        pt[2] = gridMin[2] + (static_cast<real>(z) * delta[2]);
        points[idx] = pt;
        ++idx;
      }
    }
  }

  // Test all the points using FindPointContactsParallel
  ContactDetectionParams cdParams;
  cdParams.tolerance = std::numeric_limits<real>::infinity(); // no culling
  cdParams.useAccelerationStructures = true;
  ContactDetectionResult result;
  result.sampleIndices.reserve(points.size());
  result.posColliding.reserve(points.size());
  result.sdfInfo.reserve(points.size());
  FindPointContactsParallel(
      points,
      &triMeshCollider,
      cdParams,
      TransformRT{},
      result.sampleIndices,
      result.posColliding,
      result.sdfInfo,
      result.isSdfGradUnitary);

  // Repeat, but this time set (useAccelerationStructures == false) so MeshCollider uses its brute
  // force approach rather than the BvTree.
  cdParams.useAccelerationStructures = false;
  ContactDetectionResult result2;
  result2.sampleIndices.reserve(points.size());
  result2.posColliding.reserve(points.size());
  result2.sdfInfo.reserve(points.size());
  FindPointContactsParallel(
      points,
      &triMeshCollider,
      cdParams,
      TransformRT{},
      result2.sampleIndices,
      result2.posColliding,
      result2.sdfInfo,
      result2.isSdfGradUnitary);

  // Verify that we received the same results either way. Ignore the parametric because the closest
  // point may be on a triangle edge or vertex, in which case the closest face index is arbitrary
  // and may differ between algorithms.
  ExpectEqualResults(result, result2);

  // Now, find the tetrahedron index that each sample point is in or -1 if outside.
  std::vector<int> tetIndices(kNumPoints, -1);
  FindElementsContainingPoints(*tetMesh, points, tetIndices);

  // Make sure the reported distance is always positive outside the mesh and negative inside
  for (int i = 0; i < kNumPoints; ++i) {
    EXPECT_EQ(i, result.sampleIndices[i]); // expect results for every point in order
    bool distanceIsPositive = (result.sdfInfo.val[i] > 0);
    bool pointIsActuallyOutside = (tetIndices[i] == -1);
    EXPECT_EQ(pointIsActuallyOutside, distanceIsPositive);
  }
}
TEST(MochiContact, FindPointContacts_GridSdf) {
  GridSdfParams params;
  params.boundaryPaddingDist = 0.11_r;
  params.resolutionDelta = {0.1_r, 0.05_r, 0.1_r};
  params.resolutionMode = GridSdfResolutionMode::Explicit;

  // Create an arbitrary box-shaped mesh
  Quaternion const rotation = {};
  Real3 const center = {0.1_r, 0.2_r, 0.3_r};
  Real3 const halfExt = {0.5_r, 0.25_r, 0.75_r};
  auto triMesh = CreateTriangularMeshBox(rotation, center, halfExt);

  // Compute a GridSdf for the mesh + padding
  GridSdf collider(triMesh, params, test::ExpectOK{});

  // Re-use the tests for oriented boxes, but ignore the exact center and corner points since we
  // don't expect high accuracy there. This proves that the distances are properly aligned with the
  // triangle mesh. This is a start, but more testing is warranted.
  constexpr real kAngleTol = 0.025_r;
  constexpr bool kTestCentersAndCorners = false;
  constexpr bool kExpectUnitNormal = false;
  TestPointContactsBox(
      collider,
      rotation,
      center,
      halfExt,
      ContactDetectionParams{},
      kAngleTol,
      kTestCentersAndCorners,
      kExpectUnitNormal);
}

TEST(MochiContact, FindPointContacts_Sphere) {
  constexpr Real3 kCenter = {1_r, 2_r, 3_r};
  constexpr real kRadius = 0.5_r;
  constexpr real kTol = 0.01_r;
  constexpr Real3 x = {1_r, 0_r, 0_r};
  constexpr Real3 y = {0_r, 1_r, 0_r};
  constexpr Real3 z = {0_r, 0_r, 1_r};
  Real3 const diag = Normalize(x + y + z);
  Sphere const kShape{kCenter, kRadius};

  ContactDetectionParams params;
  params.tolerance = kTol;

  // No points
  {
    ContactDetectionResult result;
    TestPointContacts({}, kShape, params, result);
    EXPECT_EQ(0, result.sampleIndices.size());
    EXPECT_EQ(0, result.sdfInfo.size());
  }

  // Test the exact center
  {
    ContactDetectionResult result;
    // The SDF gradient direction is allowed to ignore the output rotation in this case, since the
    // direction was arbitrary to begin with.
    TestPointContacts({&kCenter, 1}, kShape, params, result, /*allowRotationAgnosticGsd*/ true);
    EXPECT_EQ(true, result.isSdfGradUnitary);
    EXPECT_EQ(1, result.sampleIndices.size());
    EXPECT_EQ(0, result.sampleIndices[0]);
    EXPECT_EQ(1, result.sdfInfo.size());

    // Any point on the surface is legal
    EXPECT_NEAR_EQ(-kRadius, result.sdfInfo.val[0]);
    EXPECT_NEAR_EQ(1_r, Norm(result.sdfInfo.grad[0]));
    EXPECT_NEAR_EQ(kRadius, Norm(ToReal3(result.GetApproxPosCollider(0)) - kCenter));
  }

  // Test several points with known offset from the surface
  constexpr real kOffsets[] = {
      -kRadius / 2_r, // Well inside
      -1e-6_r, // Barely inside
      0_r, // Exact surface
      1e-6_r, // Barely outside
      kTol - 1e-6_r, // Barely within tolerance
      kTol + 1e-6_r}; // Barely outside tolerance
  for (real offset : kOffsets) {
    // Some points offset from the surface of the sphere
    Real3 const kPoints[] = {
        kCenter + (kRadius + offset) * x,
        kCenter + (kRadius + offset) * y,
        kCenter + (kRadius + offset) * z,
        kCenter + (kRadius + offset) * diag,
        kCenter - (kRadius + offset) * x,
        kCenter - (kRadius + offset) * y,
        kCenter - (kRadius + offset) * z,
        kCenter - (kRadius + offset) * diag};
    Real3 const kNormals[] = {x, y, z, diag, -x, -y, -z, -diag};
    static_assert(std::size(kNormals) == std::size(kPoints));
    ContactDetectionResult result;
    TestPointContacts(kPoints, kShape, params, result);
    if (offset > kTol) {
      EXPECT_TRUE(result.sampleIndices.empty()); // expect no contacts
    } else {
      EXPECT_EQ(true, result.isSdfGradUnitary);
      EXPECT_EQ(std::size(kPoints), result.sampleIndices.size());
      EXPECT_EQ(std::size(kPoints), result.sdfInfo.size());
      for (size_t i = 0; i < std::size(kPoints); ++i) {
        EXPECT_EQ(i, result.sampleIndices[i]);
        EXPECT_NEAR_EQ(offset, result.sdfInfo.val[i]);
        EXPECT_NEAR_EQ(1_r, Norm(result.sdfInfo.grad[i]));
        EXPECT_NEAR_EQ(kCenter + kNormals[i] * kRadius, ToReal3(result.GetApproxPosCollider(i)));
        EXPECT_NEAR_EQ(kNormals[i], result.sdfInfo.grad[i]);
      }
    }
  }
}

TEST(MochiContact, FindPointContactsParallel) {
  // FindPointContactsParallel is a template that works with all collider types.
  // We will need a TaskSchduler since MochiPhysics is not initialized.
  TaskScheduler scheduler;

  // Create a box mesh. We could use any shape, but meshes give us parametric coordinates too.
  Real3 const kCenter = {1_r, 2_r, 3_r};
  Real3 const kHalfExt = {0.5_r, 0.5_r, 0.5_r};
  auto mesh = CreateTriangularMeshBox(Quaternion{}, kCenter, kHalfExt);
  MeshCollider collider{mesh};
  collider.Initialize();

  // Generate several points to test
  auto rng = RandomGenerator(42);
  std::uniform_real_distribution<real> range(-1_r, 1_r);
  constexpr size_t kNumPoints = 999;
  std::vector<Real3> points;
  points.reserve(kNumPoints);
  for (size_t i = 0; i < kNumPoints; ++i) {
    points.push_back(kCenter + Real3{range(rng), range(rng), range(rng)});
  }

  // Include all points to keep this simple
  ContactDetectionParams params;
  params.tolerance = std::numeric_limits<real>::infinity();

  // Run the single threaded algorithm once for reference
  ContactDetectionResult result1;
  FindPointContactsT(
      points,
      &collider,
      params,
      TransformRT{},
      result1.sampleIndices,
      result1.posColliding,
      result1.sdfInfo,
      result1.isSdfGradUnitary);

  // Test the parallel algorithm for various thread counts
  for (int minPointsPerTask : {1, 10, 100, 1000, 10000}) {
    ContactDetectionResult result2;
    FindPointContactsParallel(
        points,
        &collider,
        params,
        TransformRT{},
        result2.sampleIndices,
        result2.posColliding,
        result2.sdfInfo,
        result2.isSdfGradUnitary,
        minPointsPerTask);

    // Expect results which are one-to-one
    ExpectEqualResults(result1, result2);
  }
}

TEST(MeshCollider, MeshCollider_Remesh) {
  auto testMeshCollider = [&](Span<Real3 const> coords,
                              Span<Int3 const> tris,
                              bool enableStitching,
                              bool expectTrue,
                              bool disableLogging) {
    auto logCallback = GetLogCallback();
    if (disableLogging) {
      SetLogCallback(nullptr);
    }
    auto mesh = std::make_shared<TriangularMesh const>(coords, tris);
    MeshCollider collider{mesh};
    bool allowRefitting = !enableStitching;
    collider.Initialize(allowRefitting);
    if (disableLogging) {
      SetLogCallback(logCallback);
    }
    EXPECT_TRUE(IsMeshClosed(collider.GetHalfEdge()) == expectTrue);
  };

  // Test a closed mesh
  auto&& [coords, tris] = test::CreateMinimalTriMeshUnitCube();
  testMeshCollider(coords, tris, false, true, false);

  // Replicate one of the vertices in the mesh
  coords.push_back(coords[0]);
  for (auto& tri : tris) {
    if (tri[0] == 0) {
      tri[0] = isize(coords) - 1;
      break;
    }
    if (tri[1] == 0) {
      tri[1] = isize(coords) - 1;
      break;
    }
    if (tri[2] == 0) {
      tri[2] = isize(coords) - 1;
      break;
    }
  }

  // Allow mesh stitching and validate that it's closed
  testMeshCollider(coords, tris, true, true, false);

  // Dot not allow mesh stitching and validate that it's open
  testMeshCollider(coords, tris, false, false, true);

  // Create a degenerate triangle; stitching won't help
  tris.push_back({0, 0, 1});
  testMeshCollider(coords, tris, true, false, true);

  // Confirm that the mesh still works after removing the degenerate triangle
  tris.pop_back();
  testMeshCollider(coords, tris, true, true, false);

  // Move the replicated vertex; stitching won't help
  coords.back()[0] += 1_r;
  testMeshCollider(coords, tris, true, false, true);
}

TEST(ContactJac, Move) {
  int nDoFsInternal = 3;
  int nDoFsState = 6;
  int nContacts = 10;
  for (bool sharedDoFs : {true, false}) {
    for (bool sharedJacs : {true, false}) {
      ContactJac jac0;
      jac0.Resize(sharedDoFs, sharedJacs, nDoFsInternal, nDoFsState, nContacts);
      auto const* jacsData = jac0.Jac(0).data();
      auto const* indsData = jac0.Inds(0).data();

      jac0.CompressIndices();
      auto const* indGroupsData = jac0.IndGroups(0).data();

      Matrix<real> jacAux(nDoFsInternal, nDoFsState);
      jac0.SetJacAuxView(jacAux);
      auto const* jacAuxData = jac0.JacAux().data();

      auto runChecks = [&](auto const& jac) {
        EXPECT_EQ(sharedDoFs, jac.hasSharedDoFs);
        EXPECT_EQ(sharedJacs, jac.hasSharedJacs);
        EXPECT_EQ(nDoFsInternal, jac.nDoFsInternal);
        EXPECT_EQ(nDoFsState, jac.nDoFsState);
        EXPECT_EQ(nContacts, jac.nContacts);
        EXPECT_TRUE(jac.groupsInitialized);
        EXPECT_EQ(jacsData, jac.Jac(0).data()); // Data must be moved
        EXPECT_EQ(indsData, jac.Inds(0).data()); // Data must be moved
        EXPECT_EQ(indGroupsData, jac.IndGroups(0).data()); // Data must be moved
        EXPECT_EQ(jacAuxData, jac.JacAux().data()); // Data must be moved
      };

      // Move constructor.
      ContactJac jac1 = std::move(jac0);
      runChecks(jac1);

      // Move assignment.
      ContactJac jac2;
      jac2 = std::move(jac1);
      runChecks(jac2);
    }
  }
}
