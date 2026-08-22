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

#include "mochi_physics_test_fixture.h"

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include <gtest/gtest.h>

#include <limits>

using namespace mochi;

namespace {

// Testing accuracy of coefficient of restitution (CoR) approximation using normal viscous damping.
// Tests are parameterized by CoR and velocity magnitude. The normalViscousDampingCoefficient is
// computed from the relative impact velocity and target CoR via
// @ref experimental::CalibrateNormalViscousDampingCoefficient.

// Test fixture for colliding rigid boxes test, parameterized by CoR and velocity magnitude.
// It simulates two rigid boxes colliding with normal viscous damping calibrated to approximate
// the specified coefficient of restitution and verifies that the post-collision velocities
// match the analytical solution.
class CollidingRigidBoxesTest : public test::MochiSceneTestBase,
                                public ::testing::WithParamInterface<std::tuple<real, real>> {
 public:
  static real constexpr kBoxSideLength1 = 0.1_r;
  static real constexpr kBoxSideLength2 = 0.15_r;
  static real constexpr kDensity = 1000_r;
  // Separation of centers of mass; start just outside default penalty threshold
  static real constexpr kInitialSeparation = 0.5_r * (kBoxSideLength1 + kBoxSideLength2) + 0.003_r;
  // Interval should capture full impact event.
  static real constexpr kTotalSimulationTime = 0.02_r;
  static real constexpr kTimeStep = 1e-4_r;
  // @ref experimental::CalibrateNormalViscousDampingCoefficient relates the normal damping
  // coefficient to the CoR through a rational fit, so the effective CoR cannot be brought all the
  // way to the target via refinement. Check post-impact velocities for accuracy within a small
  // fraction of the relative impact velocity.
  static real constexpr kRelVelTol = 2e-2_r;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();

    real const cor = std::get<0>(GetParam());
    real const velocityMagnitude = std::get<1>(GetParam());

    // Compute normal viscous damping coefficient to approximate the target CoR at the relative
    // impact velocity.
    real const relativeImpactVelocity = 2_r * velocityMagnitude;
    real const normalViscousDamping = experimental::CalibrateNormalViscousDampingCoefficient(
        cor, relativeImpactVelocity, test::ExpectOK{});
    _velocityTolerance = kRelVelTol * relativeImpactVelocity;

    _scene->SetGravity(Real3{0_r, 0_r, 0_r});

    auto solverParams = _scene->GetSolverParams();
    // Higher-order time integration is needed to resolve accurate bouncing behavior, especially at
    // higher CoR values.
    solverParams.integrationMethod = IntegrationMethod::BDF2;
    // Using the implicit normal force to calculate viscosity improves quantitative accuracy, but is
    // not essential for stability and qualitative response.
    solverParams.experimentalEval.implicitNormalForceForDissipation = true;
    _scene->SetSolverParams(solverParams, test::ExpectOK{});

    auto [coordinates1, connectivity1] = test::CreateMinimalTetMeshUnitCube(
        Real3{kBoxSideLength1, kBoxSideLength1, kBoxSideLength1});
    auto shape1 = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates1)), Flatten(MakeSpan(connectivity1)), test::ExpectOK{});

    auto [coordinates2, connectivity2] = test::CreateMinimalTetMeshUnitCube(
        Real3{kBoxSideLength2, kBoxSideLength2, kBoxSideLength2});
    auto shape2 = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates2)), Flatten(MakeSpan(connectivity2)), test::ExpectOK{});

    // Position boxes so their centers of mass are at x = +/- kInitialSeparation/2, y = 0, z = 0.
    Real3 const com1Local{kBoxSideLength1 / 2_r, kBoxSideLength1 / 2_r, kBoxSideLength1 / 2_r};
    Real3 const com2Local{kBoxSideLength2 / 2_r, kBoxSideLength2 / 2_r, kBoxSideLength2 / 2_r};
    real const x1 = -kInitialSeparation / 2_r;
    real const x2 = kInitialSeparation / 2_r;

    RigidActorParams box1Params;
    box1Params.name = "Box1";
    box1Params.shape = shape1;
    box1Params.colliderType = ColliderType::Box;
    box1Params.density = kDensity;
    box1Params.contact.normalViscousDampingCoefficient = normalViscousDamping;
    box1Params.worldFromLocal.SetTranslation(Real3{x1, 0_r, 0_r} - com1Local);
    _box1 = _scene->CreateRigidActor(box1Params, test::ExpectOK{});

    RigidActorParams box2Params;
    box2Params.name = "Box2";
    box2Params.shape = shape2;
    box2Params.colliderType = ColliderType::Box;
    box2Params.density = kDensity;
    box2Params.contact.normalViscousDampingCoefficient = normalViscousDamping;
    box2Params.worldFromLocal.SetTranslation(Real3{x2, 0_r, 0_r} - com2Local);
    _box2 = _scene->CreateRigidActor(box2Params, test::ExpectOK{});

    _mass1 = _box1->GetMass(test::ExpectOK{});
    _mass2 = _box2->GetMass(test::ExpectOK{});

    _box1->SetVelocity(Real3{velocityMagnitude, 0_r, 0_r}, Real3{0_r, 0_r, 0_r}, test::ExpectOK{});
    _box2->SetVelocity(Real3{-velocityMagnitude, 0_r, 0_r}, Real3{0_r, 0_r, 0_r}, test::ExpectOK{});

    real const v1 = velocityMagnitude;
    real const v2 = -velocityMagnitude;
    _expectedV1 = ((_mass1 - cor * _mass2) * v1 + (1_r + cor) * _mass2 * v2) / (_mass1 + _mass2);
    _expectedV2 = ((_mass2 - cor * _mass1) * v2 + (1_r + cor) * _mass1 * v1) / (_mass1 + _mass2);
  }

 protected:
  Actor* _box1 = nullptr;
  Actor* _box2 = nullptr;
  real _mass1{0_r};
  real _mass2{0_r};
  real _expectedV1{0_r};
  real _expectedV2{0_r};
  real _velocityTolerance{0_r};
};

} // namespace

TEST_P(CollidingRigidBoxesTest, PostCollisionVelocitiesMatchAnalyticalSolution) {
  int const numSteps = static_cast<int>(kTotalSimulationTime / kTimeStep);

  for (int i = 0; i < numSteps; ++i) {
    _scene->Step(kTimeStep);
  }

  Real3 const finalV1 = _box1->GetLinearVelocity(test::ExpectOK{});
  Real3 const finalV2 = _box2->GetLinearVelocity(test::ExpectOK{});

  EXPECT_NEAR(finalV1[0], _expectedV1, _velocityTolerance)
      << "Box 1 x-velocity does not match analytical solution. "
      << "Expected: " << _expectedV1 << ", Got: " << finalV1[0];

  EXPECT_NEAR(finalV1[1], 0_r, _velocityTolerance)
      << "Box 1 y-velocity should be zero. Got: " << finalV1[1];

  EXPECT_NEAR(finalV1[2], 0_r, _velocityTolerance)
      << "Box 1 z-velocity should be zero. Got: " << finalV1[2];

  EXPECT_NEAR(finalV2[0], _expectedV2, _velocityTolerance)
      << "Box 2 x-velocity does not match analytical solution. "
      << "Expected: " << _expectedV2 << ", Got: " << finalV2[0];

  EXPECT_NEAR(finalV2[1], 0_r, _velocityTolerance)
      << "Box 2 y-velocity should be zero. Got: " << finalV2[1];

  EXPECT_NEAR(finalV2[2], 0_r, _velocityTolerance)
      << "Box 2 z-velocity should be zero. Got: " << finalV2[2];
}

INSTANTIATE_TEST_SUITE_P(
    CorAndVelocityValues,
    CollidingRigidBoxesTest,
    ::testing::Combine(
        ::testing::Values(0.1_r, 0.3_r, 0.6_r, 0.9_r, 1_r),
        ::testing::Values(1_r, 2_r, 3_r, 4_r, 5_r)),
    [](::testing::TestParamInfo<std::tuple<real, real>> const& info) {
      std::ostringstream name;
      name << "CoR_" << std::fixed << std::setprecision(1) << std::get<0>(info.param) << "_Vel_"
           << static_cast<int>(std::get<1>(info.param));
      std::string result = name.str();
      std::replace(result.begin(), result.end(), '.', '_');
      return result;
    });

// Calibration and its inverse are exact analytical inverses; verify they round-trip on a grid of
// CoR and impact velocity values.
TEST(NormalViscousDampingCalibration, CalibrateThenEffectiveRecoversCor) {
  for (real const cor : {0.1_r, 0.3_r, 0.6_r, 0.9_r, 1_r}) {
    for (real const velocity : {1_r, 2_r, 5_r}) {
      real const damping =
          experimental::CalibrateNormalViscousDampingCoefficient(cor, velocity, test::ExpectOK{});
      real const recoveredCor =
          experimental::EffectiveCoefficientOfRestitution(damping, velocity, test::ExpectOK{});
      EXPECT_NEAR_TOL(recoveredCor, cor, 1e-4_r) << "cor=" << cor << ", velocity=" << velocity;
    }
  }
}

// Reverse round-trip: mapping a damping coefficient to its effective CoR and back recovers it.
TEST(NormalViscousDampingCalibration, EffectiveThenCalibrateRecoversDamping) {
  for (real const damping : {0.5_r, 1_r, 3_r, 10_r}) {
    for (real const velocity : {1_r, 2_r, 5_r}) {
      real const cor =
          experimental::EffectiveCoefficientOfRestitution(damping, velocity, test::ExpectOK{});
      real const recoveredDamping =
          experimental::CalibrateNormalViscousDampingCoefficient(cor, velocity, test::ExpectOK{});
      EXPECT_NEAR_TOL(recoveredDamping, damping, 1e-4_r)
          << "damping=" << damping << ", velocity=" << velocity;
    }
  }
}

// Boundary values: a perfectly elastic collision needs no damping, and zero damping is elastic.
TEST(NormalViscousDampingCalibration, ElasticBoundary) {
  for (real const velocity : {1_r, 2_r, 5_r}) {
    EXPECT_NEAR_TOL(
        experimental::CalibrateNormalViscousDampingCoefficient(1_r, velocity, test::ExpectOK{}),
        0_r,
        1e-6_r);
    EXPECT_NEAR_TOL(
        experimental::EffectiveCoefficientOfRestitution(0_r, velocity, test::ExpectOK{}),
        1_r,
        1e-6_r);
  }
}

// Invalid inputs are rejected via the error argument.
TEST(NormalViscousDampingCalibration, RejectsInvalidInputs) {
  constexpr real kNan = std::numeric_limits<real>::quiet_NaN();

  // CoR must be finite and in (0, 1].
  (void)experimental::CalibrateNormalViscousDampingCoefficient(0_r, 1_r, test::ExpectNotOK{});
  (void)experimental::CalibrateNormalViscousDampingCoefficient(-0.1_r, 1_r, test::ExpectNotOK{});
  (void)experimental::CalibrateNormalViscousDampingCoefficient(1.1_r, 1_r, test::ExpectNotOK{});
  (void)experimental::CalibrateNormalViscousDampingCoefficient(kNan, 1_r, test::ExpectNotOK{});
  // Impact velocity must be finite and positive.
  (void)experimental::CalibrateNormalViscousDampingCoefficient(0.5_r, 0_r, test::ExpectNotOK{});
  (void)experimental::CalibrateNormalViscousDampingCoefficient(0.5_r, -1_r, test::ExpectNotOK{});
  (void)experimental::CalibrateNormalViscousDampingCoefficient(0.5_r, kNan, test::ExpectNotOK{});

  // Damping coefficient must be finite and non-negative.
  (void)experimental::EffectiveCoefficientOfRestitution(-1_r, 1_r, test::ExpectNotOK{});
  (void)experimental::EffectiveCoefficientOfRestitution(kNan, 1_r, test::ExpectNotOK{});
  // Impact velocity must be finite and positive.
  (void)experimental::EffectiveCoefficientOfRestitution(1_r, 0_r, test::ExpectNotOK{});
  (void)experimental::EffectiveCoefficientOfRestitution(1_r, -1_r, test::ExpectNotOK{});
  (void)experimental::EffectiveCoefficientOfRestitution(1_r, kNan, test::ExpectNotOK{});
}

// The calibration reduces to known analytical limits: e * v = 1 / c as CoR -> 0 (fully inelastic,
// where the damping diverges) and c * v = 1.5 * (1 - e) as CoR -> 1 (the classic Hunt & Crossley
// result). Both are checked as ratios to 1 so the comparison stays meaningful near the limits.
TEST(NormalViscousDampingCalibration, MatchesAnalyticalLimits) {
  constexpr real kNearZeroCor = 1e-4_r;
  constexpr real kNearOneCor = 1_r - 1e-4_r;
  for (real const velocity : {1_r, 2_r, 5_r}) {
    // e -> 0: c -> 1 / (e * v), equivalently e * v * c -> 1.
    real const inelasticDamping = experimental::CalibrateNormalViscousDampingCoefficient(
        kNearZeroCor, velocity, test::ExpectOK{});
    EXPECT_NEAR_TOL(kNearZeroCor * velocity * inelasticDamping, 1_r, 2e-3_r)
        << "velocity=" << velocity;

    // e -> 1: c * v -> 1.5 * (1 - e).
    real const elasticDamping = experimental::CalibrateNormalViscousDampingCoefficient(
        kNearOneCor, velocity, test::ExpectOK{});
    EXPECT_NEAR_TOL(velocity * elasticDamping / (1.5_r * (1_r - kNearOneCor)), 1_r, 2e-3_r)
        << "velocity=" << velocity;
  }
}
