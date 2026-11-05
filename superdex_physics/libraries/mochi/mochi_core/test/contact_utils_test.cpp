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
#include <mochi_core/utils/spmat_utils.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace mochi;

namespace {

struct ExpectedCoalescedContactJac {
  std::vector<std::vector<int>> indices;
  std::vector<std::vector<Real3>> columns;
};

ExpectedCoalescedContactJac SortAndCoalesceContactJacForTest(ContactJac const& jac) {
  ExpectedCoalescedContactJac expected;
  expected.indices.resize(jac.nContacts);
  expected.columns.resize(jac.nContacts);
  int maxWidth = 0;
  for (int contact = 0; contact < jac.nContacts; ++contact) {
    auto const indices = jac.Inds(contact);
    auto const columns = jac.Jac(contact);
    std::map<int, Real3> summedColumns;
    for (int column = 0; column < jac.nDoFsInternal; ++column) {
      summedColumns[indices[column]] +=
          Real3{columns(0, column), columns(1, column), columns(2, column)};
    }
    for (auto const& [dofIndex, sum] : summedColumns) {
      expected.indices[contact].push_back(dofIndex);
      expected.columns[contact].push_back(sum);
    }
    maxWidth = Max(maxWidth, isize(summedColumns));
  }

  for (int contact = 0; contact < jac.nContacts; ++contact) {
    while (isize(expected.indices[contact]) < maxWidth) {
      expected.indices[contact].push_back(expected.indices[contact].front());
      expected.columns[contact].emplace_back();
    }
  }
  return expected;
}

} // namespace

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
      /*isSdfGradUnitary*/ true,
      /*isNormalCollidingUnitary*/ true);

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
          /*isSdfGradUnitary*/ true,
          /*isNormalCollidingUnitary*/ true);
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
          /*isSdfGradUnitary*/ true,
          /*isNormalCollidingUnitary*/ true);
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

TEST(ContactPenalties, Friction_DegenerateNormalsKeepFullStrength) {
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
  Real3 distanceGrad[1] = {Real3{0_r, 0_r, 1_r}};
  bool isSdfGradUnitary = true;
  Real3 normalColliding[1] = {Real3{0_r, 0_r, -1_r}};
  bool isNormalCollidingUnitary = true;
  Real3 const posColliding[1] = {Real3{0_r, 0_r, 0_r}};
  Real3 posCollidingStageStart[1] = {Real3{0.001_r, 0_r, 0_r}};

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
        isSdfGradUnitary,
        isNormalCollidingUnitary);
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

  // Without either normal, friction opposes the whole displacement, even along the former normal,
  // and no penalty force acts.
  distanceGrad[0] = Real3{};
  isSdfGradUnitary = false;
  posCollidingStageStart[0] = Real3{0_r, 0_r, 0.001_r};
  auto const [zeroNormalEnergy, zeroNormalForce] = eval(fadedConfig);
  EXPECT_DOUBLE_EQ(unfadedEnergy, zeroNormalEnergy);
  EXPECT_NEAR_EQ((Real3{0_r, 0_r, unfadedForce[0]}), zeroNormalForce);

  // Without the collider normal, friction uses the colliding normal, normalized if it isn't unit.
  params.maxAlignmentNormals = 0_r;
  normalColliding[0] = Real3{0_r, 0_r, -2_r};
  isNormalCollidingUnitary = false;
  posCollidingStageStart[0] = Real3{0.001_r, 0_r, 0.001_r};
  auto const [collidingNormalEnergy, collidingNormalForce] = eval(unfadedConfig);
  EXPECT_DOUBLE_EQ(unfadedEnergy, collidingNormalEnergy);
  EXPECT_NEAR_EQ((Real3{unfadedForce[0], 0_r, 0_r}), collidingNormalForce);
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

// Match the batch size used by production code.
// TODO[T289584846] - Adjust after AVX-512 turning.
constexpr int kPointContactBatchSize =
    Max(4, MOCHI_ARCH_ARM_NEON ? (2 * Simd<real>::kSize) : Simd<real>::kSize);

template <class ShapeT, class MakePoint, class Verify>
void TestEveryPointContactTail(ShapeT const& shape, MakePoint makePoint, Verify verify) {
  for (int count = 1; count <= kPointContactBatchSize; ++count) {
    DynamicArray<Real3> points;
    points.resize_noinit(count);
    for (int i = 0; i < count; ++i) {
      points[i] = makePoint(i);
    }

    ContactDetectionResult result;
    TestPointContactsImpl(points, shape, {}, TransformRT{}, result);
    ASSERT_EQ(static_cast<size_t>(count), result.sampleIndices.size());
    ASSERT_EQ(static_cast<size_t>(count), result.posColliding.size());
    ASSERT_EQ(static_cast<size_t>(count), result.sdfInfo.size());
    for (int i = 0; i < count; ++i) {
      EXPECT_EQ(i, result.sampleIndices[i]);
      EXPECT_NEAR_EQ(points[i], result.posColliding[i]);
      verify(points[i], result.sdfInfo.val[i], result.sdfInfo.grad[i]);
    }
  }
}

TEST(MochiContact, FindPointContacts_PlaneEveryTailSize) {
  Plane const plane{Real3{0_r, 1_r, 0_r}, 0_r};
  TestEveryPointContactTail(
      plane,
      [](int i) { return Real3{static_cast<real>(i) * 0.01_r, -0.5_r, 0.25_r}; },
      [](Real3 const&, real sdf, Real3 const& gradient) {
        EXPECT_NEAR_EQ(-0.5_r, sdf);
        EXPECT_NEAR_EQ((Real3{0_r, 1_r, 0_r}), gradient);
      });
}

TEST(MochiContact, FindPointContacts_SphereEveryTailSize) {
  Sphere const sphere{Real3{}, 2_r};
  TestEveryPointContactTail(
      sphere,
      [](int i) { return Real3{1_r + static_cast<real>(i) * 0.01_r, 0_r, 0_r}; },
      [](Real3 const& point, real sdf, Real3 const& gradient) {
        EXPECT_NEAR_EQ(point[0] - 2_r, sdf);
        EXPECT_NEAR_EQ((Real3{1_r, 0_r, 0_r}), gradient);
      });
}

TEST(MochiContact, FindPointContacts_ObbEveryTailSize) {
  Obb const box{TransformRT{}, Real3{2_r, 2_r, 2_r}};
  TestEveryPointContactTail(
      box,
      [](int i) { return Real3{1_r + static_cast<real>(i) * 0.01_r, 0.2_r, 0.3_r}; },
      [](Real3 const& point, real sdf, Real3 const& gradient) {
        EXPECT_NEAR_EQ(point[0] - 2_r, sdf);
        EXPECT_NEAR_EQ((Real3{1_r, 0_r, 0_r}), gradient);
      });
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

// A point's contact position must not depend on whether it lands in a full batch, the tail, or a
// buffer flushed early or at the end, because friction compares it bit for bit with the position
// from the stage-start pass, which repeats the query on a subset of the points. Distances and
// gradients only need to agree within rounding.
template <typename ShapeT>
static void TestPointContactsIndependentOfBatchPosition(
    ShapeT const& shape,
    Real3 const& sampleCenterInCollider,
    Real3 const& sampleHalfExtents) {
  // Far from the origin, where rounding differences are large.
  TransformRT const pointsFromCollider{
      Quaternion::FromRotationVector(Real3{0.31_r, -0.58_r, 0.24_r}), Real3{0.1_r, 0.2_r, -32.1_r}};
  ContactDetectionParams params;
  params.tolerance = kInf;

  // Enough points for several buffer and queue flushes, plus a tail.
  constexpr int kNumPoints = 1100;
  std::mt19937 rng(7);
  std::uniform_real_distribution<real> unit(-1_r, 1_r);
  DynamicArray<Real3> points(kNumPoints);
  for (auto& point : points) {
    Real3 const offset{unit(rng), unit(rng), unit(rng)};
    point = pointsFromCollider.TransformPoint(sampleCenterInCollider + offset * sampleHalfExtents);
  }

  auto query = [&](Span<Real3 const> queryPoints, ContactDetectionResult& outResult) {
    FindPointContactsT(
        queryPoints,
        &shape,
        params,
        pointsFromCollider,
        outResult.sampleIndices,
        outResult.posColliding,
        outResult.sdfInfo,
        outResult.isSdfGradUnitary);
  };

  ContactDetectionResult all;
  query(points, all);
  ASSERT_EQ(all.sampleIndices.size(), points.size());

  // A few ulp at the sample points' distance from the origin.
  constexpr real kTolerance = 16_r * 32_r * std::numeric_limits<real>::epsilon();
  int numMismatches = 0;
  for (int i = 0; i < kNumPoints; ++i) {
    ContactDetectionResult single;
    query(Span<Real3 const>{&points[i], 1}, single);
    ASSERT_EQ(single.sampleIndices.size(), size_t{1});
    numMismatches +=
        std::memcmp(&all.posColliding[i], &single.posColliding[0], sizeof(Real3)) != 0 ? 1 : 0;
    EXPECT_NEAR_TOL(all.sdfInfo.val[i], single.sdfInfo.val[0], kTolerance) << "point " << i;
    EXPECT_NEAR_TOL(all.sdfInfo.grad[i], single.sdfInfo.grad[0], kTolerance) << "point " << i;
  }
  EXPECT_EQ(numMismatches, 0) << numMismatches << " of " << kNumPoints << " positions differ";
}

// The flat colliders' top face is at y = 0, near z = 32 m.
static constexpr Real3 kFarTopFaceCenter{0_r, 0_r, 32_r};
static constexpr Real3 kFarTopFaceSampleHalfExtents{0.9_r, 0.1_r, 0.9_r};

static std::shared_ptr<TriangularMesh const> CreateFarGroundMesh() {
  return CreateTriangularMeshBox(Quaternion{}, Real3{0_r, -0.5_r, 32_r}, Real3{1_r, 0.5_r, 1_r});
}

TEST(MochiContact, FindPointContacts_PlaneIndependentOfBatchPosition) {
  TestPointContactsIndependentOfBatchPosition(
      Plane{Real3{0_r, 1_r, 0_r}, 0_r}, kFarTopFaceCenter, kFarTopFaceSampleHalfExtents);
}

TEST(MochiContact, FindPointContacts_SphereIndependentOfBatchPosition) {
  TestPointContactsIndependentOfBatchPosition(
      Sphere{Real3{0_r, -1_r, 32_r}, 1_r}, kFarTopFaceCenter, Real3{0.3_r, 0.1_r, 0.3_r});
}

TEST(MochiContact, FindPointContacts_ObbIndependentOfBatchPosition) {
  TestPointContactsIndependentOfBatchPosition(
      Obb{TransformRT{Quaternion::RotationY(0.37_r), Real3{0_r, -0.5_r, 32_r}},
          Real3{1_r, 0.5_r, 1_r}},
      kFarTopFaceCenter,
      kFarTopFaceSampleHalfExtents);
}

TEST(MochiContact, FindPointContacts_MeshIndependentOfBatchPosition) {
  MeshCollider collider{CreateFarGroundMesh()};
  collider.Initialize();
  TestPointContactsIndependentOfBatchPosition(
      collider, kFarTopFaceCenter, kFarTopFaceSampleHalfExtents);
}

TEST(MochiContact, FindPointContacts_GridSdfIndependentOfBatchPosition) {
  GridSdfParams params;
  params.resolutionMode = GridSdfResolutionMode::Explicit;
  params.resolutionDelta = {0.05_r, 0.05_r, 0.05_r};
  GridSdf const collider(CreateFarGroundMesh(), params, test::ExpectOK{});
  TestPointContactsIndependentOfBatchPosition(
      collider, kFarTopFaceCenter, kFarTopFaceSampleHalfExtents);
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
  DynamicArray<Real3> points;
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
  auto testMeshCollider =
      [&](Span<Real3 const> coords, Span<Int3 const> tris, bool expectTrue, bool disableLogging) {
        auto logCallback = GetLogCallback();
        if (disableLogging) {
          SetLogCallback(nullptr);
        }
        auto mesh = std::make_shared<TriangularMesh const>(coords, tris);
        MeshCollider collider{mesh};
        collider.Initialize();
        if (disableLogging) {
          SetLogCallback(logCallback);
        }
        EXPECT_TRUE(IsMeshClosed(collider.GetHalfEdge()) == expectTrue);
      };

  // Test a closed mesh
  auto&& [coords, tris] = test::CreateMinimalTriMeshUnitCube();
  testMeshCollider(coords, tris, true, false);

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

  // Stitching closes the mesh
  testMeshCollider(coords, tris, true, false);

  // Create a degenerate triangle; stitching won't help
  tris.push_back({0, 0, 1});
  testMeshCollider(coords, tris, false, true);

  // Confirm that the mesh still works after removing the degenerate triangle
  tris.pop_back();
  testMeshCollider(coords, tris, true, false);

  // Move the replicated vertex; stitching won't help
  coords.back()[0] += 1_r;
  testMeshCollider(coords, tris, false, true);
}

TEST(ContactJac, CoalesceColumnsByCachedMapping) {
  int constexpr kNumContacts = 3;
  int constexpr kSourceStride = 3;
  int constexpr kOldWidth = 3 * kSourceStride;
  int constexpr kNewWidth = 4;
  int constexpr kDofOffset = 11;
  std::array<int, kNumContacts> const sampleIndices = {1, 2, 0};

  ContactJacColumnCoalescingMap mapping;
  mapping.sourceBlockCount = 3;
  mapping.samplesPerElement = 2;
  mapping.elementDofOffsets = {0, 4, 7};
  mapping.elementDofIndices = {2, 3, 5, 7, 4, 8, 9};
  mapping.elementSourceContributionOffsets = {0, 2, 4, 6, 7, 10, 11};
  mapping.elementSourceContributionDstColumns = {0, 2, 1, 2, 0, 3, 0, 0, 1, 2, 1};

  auto initializeWideJacobian = [&](ContactJac& jac) {
    jac.Resize(false, false, kOldWidth, kNumContacts);
    for (int contact = 0; contact < kNumContacts; ++contact) {
      int const elementIndex = sampleIndices[contact] / mapping.samplesPerElement;
      auto const elementDofs = mapping.ElementDofIndices(elementIndex);
      auto indices = jac.Inds(contact);
      auto values = jac.Jac(contact);
      for (int sourceBlock = 0; sourceBlock < mapping.sourceBlockCount; ++sourceBlock) {
        auto const destinations = mapping.SourceContributionDstColumns(elementIndex, sourceBlock);
        ASSERT_FALSE(destinations.empty());
        for (int localEntry = 0; localEntry < kSourceStride; ++localEntry) {
          int const sourceColumn = sourceBlock * kSourceStride + localEntry;
          int const destination = destinations[Min(localEntry, isize(destinations) - 1)];
          indices[sourceColumn] = elementDofs[destination] + kDofOffset;
          if (localEntry < isize(destinations)) {
            real const value = 100_r * contact + 10_r * sourceBlock + localEntry + 1_r;
            for (int component = 0; component < ContactJac::kDofsPerNode; ++component) {
              values(component, sourceColumn) = value + component;
            }
          } else {
            values.Col(sourceColumn).SetZero();
          }
        }
      }
    }
  };

  ContactJac cached;
  ContactJac repeated;
  initializeWideJacobian(cached);
  initializeWideJacobian(repeated);
  auto const expected = SortAndCoalesceContactJacForTest(cached);
  auto const* const indicesData = cached.Inds(0).data();
  auto const* const jacData = cached.Jac(0).data();

  cached.CoalesceColumnsByCachedMapping(mapping, sampleIndices, kDofOffset);
  repeated.CoalesceColumnsByCachedMapping(mapping, sampleIndices, kDofOffset);

  EXPECT_EQ(kNewWidth, cached.nDoFsInternal);
  EXPECT_EQ(kNewWidth, cached.nDoFsState);
  EXPECT_EQ(indicesData, cached.Inds(0).data());
  EXPECT_EQ(jacData, cached.Jac(0).data());
  EXPECT_FALSE(cached.groupsInitialized);
  for (int contact = 0; contact < kNumContacts; ++contact) {
    EXPECT_SPAN_EQ(MakeConstSpan(expected.indices[contact]), cached.Inds(contact));
    EXPECT_SPAN_EQ(cached.Inds(contact), repeated.Inds(contact));
    for (int column = 0; column < kNewWidth; ++column) {
      Real3 const cachedColumn = {
          cached.Jac(contact)(0, column),
          cached.Jac(contact)(1, column),
          cached.Jac(contact)(2, column),
      };
      EXPECT_NEAR_EQ(expected.columns[contact][column], cachedColumn);
      for (int component = 0; component < ContactJac::kDofsPerNode; ++component) {
        EXPECT_EQ(cached.Jac(contact)(component, column), repeated.Jac(contact)(component, column));
      }
    }
  }

  std::array<int, kNewWidth> const paddedElementIndices = {15, 19, 20, 15};
  EXPECT_SPAN_EQ(MakeConstSpan(paddedElementIndices), cached.Inds(1));
  Real3 const paddedColumn = {
      cached.Jac(1)(0, kNewWidth - 1),
      cached.Jac(1)(1, kNewWidth - 1),
      cached.Jac(1)(2, kNewWidth - 1),
  };
  EXPECT_EQ(Real3{}, paddedColumn);

  // The same map must also work when the query omits the wider second element, reducing the source
  // stride from three to two.
  int constexpr kLowStride = 2;
  std::array<int, 2> const lowStrideSampleIndices = {0, 1};
  ContactJac lowStrideCached;
  auto initializeLowStrideJacobian = [&](ContactJac& jac) {
    jac.Resize(false, false, mapping.sourceBlockCount * kLowStride, isize(lowStrideSampleIndices));
    for (int contact = 0; contact < isize(lowStrideSampleIndices); ++contact) {
      auto indices = jac.Inds(contact);
      auto values = jac.Jac(contact);
      auto const elementDofs = mapping.ElementDofIndices(0);
      for (int sourceBlock = 0; sourceBlock < mapping.sourceBlockCount; ++sourceBlock) {
        auto const destinations = mapping.SourceContributionDstColumns(0, sourceBlock);
        ASSERT_EQ(kLowStride, isize(destinations));
        for (int localEntry = 0; localEntry < kLowStride; ++localEntry) {
          int const sourceColumn = sourceBlock * kLowStride + localEntry;
          indices[sourceColumn] = elementDofs[destinations[localEntry]] + kDofOffset;
          for (int component = 0; component < ContactJac::kDofsPerNode; ++component) {
            values(component, sourceColumn) =
                100_r * contact + 10_r * sourceBlock + localEntry + component;
          }
        }
      }
    }
  };
  initializeLowStrideJacobian(lowStrideCached);
  auto const lowStrideExpected = SortAndCoalesceContactJacForTest(lowStrideCached);
  lowStrideCached.CoalesceColumnsByCachedMapping(mapping, lowStrideSampleIndices, kDofOffset);
  for (int contact = 0; contact < isize(lowStrideSampleIndices); ++contact) {
    EXPECT_SPAN_EQ(
        MakeConstSpan(lowStrideExpected.indices[contact]), lowStrideCached.Inds(contact));
    for (int column = 0; column < lowStrideCached.nDoFsInternal; ++column) {
      Real3 const actualColumn = {
          lowStrideCached.Jac(contact)(0, column),
          lowStrideCached.Jac(contact)(1, column),
          lowStrideCached.Jac(contact)(2, column),
      };
      EXPECT_NEAR_EQ(lowStrideExpected.columns[contact][column], actualColumn);
    }
  }

  cached.CompressIndices();
  EXPECT_TRUE(cached.groupsInitialized);
  for (int contact = 0; contact < kNumContacts; ++contact) {
    int groupedColumns = 0;
    for (IndexGroup const& group : cached.IndGroups(contact)) {
      groupedColumns += group.count;
    }
    EXPECT_EQ(kNewWidth, groupedColumns);
  }
}

TEST(ContactJac, CompressIndicesOfRepeatedRows) {
  // Rows 0 and 1 are equal, as are rows 3 and 4, like the rows of contacts on one surface element.
  // Row 2 differs from row 1 only in its first triplet and from row 3 only in its last index. Each
  // row must get the groups of its own DoF indices, also when the group counts differ.
  constexpr int kWidth = 6;
  std::array<std::array<int, kWidth>, 5> const rows = {{
      {9, 10, 11, 3, 4, 5},
      {9, 10, 11, 3, 4, 5},
      {0, 1, 2, 3, 4, 5},
      {0, 1, 2, 3, 4, 3},
      {0, 1, 2, 3, 4, 3},
  }};
  ContactJac jac;
  // Per-contact DoFs, so CompressIndices forces triplets.
  jac.Resize(/*sharedDoFs*/ false, /*sharedJacs*/ false, kWidth, isize(rows));
  for (int contact = 0; contact < isize(rows); ++contact) {
    std::copy(rows[contact].begin(), rows[contact].end(), jac.Inds(contact).begin());
  }

  jac.CompressIndices();

  EXPECT_TRUE(jac.groupsInitialized);
  for (int contact = 0; contact < isize(rows); ++contact) {
    EXPECT_SPAN_EQ(
        CreateIndexGroups(rows[contact], /*forceTriplets*/ true), jac.IndGroups(contact));
  }
}

TEST(ContactJac, CoalesceColumnsByCachedMappingWithNoContacts) {
  ContactJacColumnCoalescingMap mapping;
  mapping.sourceBlockCount = 3;
  mapping.samplesPerElement = 1;
  mapping.elementDofOffsets = {0, 1};
  mapping.elementDofIndices = {2};
  mapping.elementSourceContributionOffsets = {0, 1, 2, 3};
  mapping.elementSourceContributionDstColumns = {0, 0, 0};

  ContactJac jac;
  jac.Resize(false, false, 3, 1);
  auto const* const indicesData = jac.Inds(0).data();
  auto const* const jacData = jac.Jac(0).data();
  jac.Resize(false, false, 3, 0);

  std::array<int, 0> const sampleIndices{};
  jac.CoalesceColumnsByCachedMapping(mapping, sampleIndices, 0);

  EXPECT_EQ(0, jac.nDoFsInternal);
  EXPECT_EQ(0, jac.nDoFsState);
  EXPECT_EQ(0, jac.nContacts);
  EXPECT_FALSE(jac.groupsInitialized);
  jac.Resize(false, false, 3, 1);
  EXPECT_EQ(indicesData, jac.Inds(0).data());
  EXPECT_EQ(jacData, jac.Jac(0).data());
}

TEST(ContactJac, Move) {
  constexpr int kNDoFsState = 8;
  int const nContacts = 10;
  RowMatrix<real, RigidSize::kDAll> const jacAux(RigidSize::kDAll, kNDoFsState);
  std::array<int, kNDoFsState> dofsAux{};
  std::iota(dofsAux.begin(), dofsAux.end(), 0);
  int const kInternalDofOffset = 12;
  for (bool sharedDoFs : {true, false}) {
    for (bool sharedJacs : {true, false}) {
      // Auxiliary Jacobians require shared DoFs.
      int const nDoFsInternal = sharedDoFs ? RigidSize::kDAll : kNDoFsState;
      int const internalDofOffset = sharedDoFs ? kInternalDofOffset : kSentinelIndex;
      ContactJac jac0;
      jac0.Resize(sharedDoFs, sharedJacs, nDoFsInternal, nContacts);
      if (sharedDoFs) {
        jac0.SetJacAux(jacAux, 0, dofsAux, internalDofOffset);
      }
      auto const* jacsData = jac0.Jac(0).data();
      auto const* indsData = jac0.Inds(0).data();

      jac0.CompressIndices();
      auto const* indGroupsData = jac0.IndGroups(0).data();
      auto const* jacAuxData = jac0.JacAux().data();

      auto runChecks = [&](auto const& jac) {
        EXPECT_EQ(sharedDoFs, jac.hasSharedDoFs);
        EXPECT_EQ(sharedJacs, jac.hasSharedJacs);
        EXPECT_EQ(nDoFsInternal, jac.nDoFsInternal);
        EXPECT_EQ(kNDoFsState, jac.nDoFsState);
        EXPECT_EQ(nContacts, jac.nContacts);
        EXPECT_EQ(internalDofOffset, jac.internalDofOffset);
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

TEST(ContactJac, SetJacAux) {
  constexpr int kNumContacts = 3;
  constexpr int kDofOffset = 10;
  constexpr int kInternalDofOffset = 12;
  std::array<int, 8> const dofsAux{0, 1, 2, 5, 6, 7, 8, 9};
  RowMatrix<real, RigidSize::kDAll> const jacAux(RigidSize::kDAll, isize(dofsAux));
  ContactJac jac;
  jac.Resize(true, false, RigidSize::kDAll, kNumContacts);
  auto const* const jacsData = jac.Jac(0).data();

  jac.SetJacAux(jacAux, kDofOffset, dofsAux, kInternalDofOffset);

  std::array<int, 8> const expectedInds{10, 11, 12, 15, 16, 17, 18, 19};
  EXPECT_SPAN_EQ(MakeConstSpan(expectedInds), jac.Inds(0));
  EXPECT_EQ(RigidSize::kDAll, jac.nDoFsInternal);
  EXPECT_EQ(isize(dofsAux), jac.nDoFsState);
  EXPECT_EQ(kInternalDofOffset, jac.internalDofOffset);
  EXPECT_EQ(jacAux.data(), jac.JacAux().data());
  EXPECT_EQ(jacsData, jac.Jac(0).data()); // The rigid Jacobian blocks are kept

  // Resizing removes the auxiliary Jacobian.
  jac.Resize(true, false, RigidSize::kDAll, kNumContacts);
  EXPECT_TRUE(jac.JacAux().empty());
  EXPECT_EQ(kSentinelIndex, jac.internalDofOffset);
  EXPECT_EQ(RigidSize::kDAll, jac.nDoFsState);
}
