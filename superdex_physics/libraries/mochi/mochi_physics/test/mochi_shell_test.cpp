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
#include <mochi_core/utils/defer.h>
#include <mochi_physics/src/mochi_linear_contact_skin.h>
#include <mochi_physics/src/mochi_point_cloud_contact.h>
#include <mochi_physics/src/mochi_rod.h>
#include <mochi_physics/src/mochi_shell.h>
#include <mochi_physics/src/mochi_step.h>
#include "mochi_core/test/mochi_test_helpers.h"
#include "mochi_physics_test_fixture.h"

#include <array>
#include <utility>

using namespace mochi;
using namespace mochi::shell;
using namespace mochi::experimental;

class MochiShellActorUnitCubeScene : public test::MochiSceneTestBase {
 protected:
  Actor* _actor = nullptr;

 public:
  void SetUp() override {
    test::MochiSceneTestBase::SetUp();

    // Create shell actor
    // This is a cube of size 1, centered at 0.5
    auto& reg = GetRegistry();
    auto&& [unitCubeCoordinates, unitCubeConnectivity] = test::CreateMinimalTriMeshUnitCube();
    ShellActorParams params;
    params.material.density = 1_r;
    params.colliderType = ColliderType::Auto;
    // Set reasonable membrane stiffness parameters (roughly corresponding to E=100, nu=0.25).
    params.material.membraneLambda = 40_r;
    params.material.membraneMu = 40_r;
    params.shape = _scene->GetContext()->CreateTriMeshShape(
        Flatten(MakeSpan(unitCubeCoordinates)),
        Flatten(MakeSpan(unitCubeConnectivity)),
        ErrorAssert{});
    _actor = CreateShellActor(_scene, params, ErrorAssert{});
    EXPECT_NE(entt::entity{}, mochi::GetEntity(reg, _actor->GetHandle(), test::ExpectOK{}));
  }
};

TEST_F(MochiShellActorUnitCubeScene, ShellActorFreefall) {
  // Simulate with a gravitational acceleration in all three directions.
  Real3 constexpr kGravity = {1_r, 2_r, 3_r};
  _scene->SetGravity(kGravity);

  // Need implicit midpoint for exact free-falling trajectory.
  auto solverParams = _scene->GetSolverParams();
  solverParams.integrationMethod = IntegrationMethod::SymplecticDIRK12;
  _scene->SetSolverParams(solverParams, ErrorAssert{});

  real constexpr kTimeInterval = 4_r;
  int constexpr kNumSteps = 50;
  real constexpr kTimeStep = kTimeInterval / (real)kNumSteps;
  for (int i = 0; i < kNumSteps; ++i) {
    _scene->Step(kTimeStep);
  }
  Real3 const kExpectedPosition = 0.5_r * kGravity * kTimeInterval * kTimeInterval;
  int const numDofs = _actor->GetNumDofs();
  DynamicArray<real> dofValues(numDofs);
  _actor->GetDofValues({}, dofValues, test::ExpectOK{});

  for (int i = 0; i < numDofs; i += 3) {
    // The result should be exact up to algebraic solver tolerances (and floating point rounding
    // errors). The truncation error from time integration should be zero in freefall with implicit
    // midpoint.
    real constexpr kTolerance = 2e-5_r;
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR_RTOL(kExpectedPosition[j], dofValues[i + j], kTolerance);
    }
  }
}

class MochiShellActorAxialDeformationScene : public test::MochiSceneTestBase {
 protected:
  Actor* _actor = nullptr;

 public:
  static constexpr real kScale = 2_r;
  static constexpr int kM = 16;
  static constexpr int kN = 16;
  static constexpr real kYoungsModulus3d = 1e5_r;
  static constexpr real kPoissonsRatio3d = 0_r;
  static constexpr real kDensity2d = 1_r;
  static constexpr real kThickness = 1e-1_r;
  static constexpr real kGravityMagnitude = 1_r;
  static constexpr real kBcEps = 1e-3_r;
  static constexpr real kTolerance = 1e-2;

  void PerTestSetUp() {
    auto& reg = GetRegistry();
    auto&& [coordinates, connectivity] =
        UniformSquareTriangularMeshData(Int2{kM, kN}, Real2{kScale, kScale});
    ShellActorParams params;

    params.material = mochi::experimental::ShellMaterialParamsFrom3dIsotropic(
        kYoungsModulus3d, kPoissonsRatio3d, kDensity2d / kThickness, kThickness, ErrorAssert{});

    params.shape = _scene->GetContext()->CreateTriMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), ErrorAssert{});
    _actor = CreateShellActor(_scene, params, ErrorAssert{});
    EXPECT_NE(entt::entity{}, mochi::GetEntity(reg, _actor->GetHandle(), test::ExpectOK{}));
  }
  void PerTestTearDown() {
    _scene->DestroyActor(_actor);
  }

  void RunTest() {
    PerTestSetUp();
    _scene->SetGravity({0_r, -kGravityMagnitude, 0_r});
    experimental::ConstrainNodesByPosition(
        _actor,
        [](int, Real3 const& x) -> bool { return x[1] > 0.5_r * kScale - kBcEps; },
        ErrorAssert{});

    // Take one large backward Euler step to get to static equilibrium quickly.
    test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
    real constexpr kTimeInterval = 1e8_r;
    int constexpr kNumSteps = 1;
    real constexpr kTimeStep = kTimeInterval / (real)kNumSteps;
    for (int i = 0; i < kNumSteps; ++i) {
      _scene->Step(kTimeStep);
    }

    // Check the maximum displacement at the end of the time stepping, and compare against the
    // expected solution from elementary strength of materials.
    int const numDofs = _actor->GetNumDofs();
    int const numNodes = numDofs / 3;
    DynamicArray<real> dofValues(numDofs);
    _actor->GetDofValues({}, dofValues, ErrorAssert{});
    real const expected =
        0.5_r * kDensity2d * kGravityMagnitude * kScale * kScale / kYoungsModulus3d / kThickness;
    real maxYDisplacement = 0_r;
    for (int nodeIndex = 0; nodeIndex < numNodes; nodeIndex++) {
      maxYDisplacement = Max(Abs(dofValues[3 * nodeIndex + 1]), maxYDisplacement);
    }
    EXPECT_NEAR(expected, maxYDisplacement, kTolerance * Abs(expected));
    PerTestTearDown();
  }
};
TEST_F(MochiShellActorAxialDeformationScene, ShellActorAxialDeformation) {
  RunTest();
}

class MochiShellActorSimplySupportedBeamScene : public test::MochiSceneTestBase {
 protected:
  Actor* _actor = nullptr;

 public:
  static constexpr real kScale = 2_r;
  static constexpr int kM = 1;
  static constexpr int kN = 8;
  static constexpr real kYoungsModulus3d = 1e9_r;
  static constexpr real kPoissonsRatio3d = 0_r;
  static constexpr real kDensity2d = 90_r;
  static constexpr real kThickness = 1e-1_r;
  static constexpr real kGravityMagnitude = 1_r;
  static constexpr real kBcEps = 1e-3_r;
  static constexpr real kTolerance = 5e-2;

  void PerTestSetup() {
    auto& reg = GetRegistry();
    auto&& [coordinates, connectivity] =
        UniformSquareTriangularMeshData(Int2{kM, kN}, Real2{kScale, kScale});
    ShellActorParams params;

    params.material = mochi::experimental::ShellMaterialParamsFrom3dIsotropic(
        kYoungsModulus3d, kPoissonsRatio3d, kDensity2d / kThickness, kThickness, ErrorAssert{});

    params.shape = _scene->GetContext()->CreateTriMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), ErrorAssert{});
    _actor = CreateShellActor(_scene, params, ErrorAssert{});
    EXPECT_NE(entt::entity{}, mochi::GetEntity(reg, _actor->GetHandle(), test::ExpectOK{}));
  }
  void PerTestTearDown() {
    _scene->DestroyActor(_actor);
  }

  void RunTest() {
    PerTestSetup();
    _scene->SetGravity({0_r, 0_r, -kGravityMagnitude});
    experimental::ConstrainNodesByPosition(
        _actor,
        [](int, Real3 const& x) -> bool { return Abs(x[1]) > 0.5_r * kScale - kBcEps; },
        ErrorAssert{});

    // Take one large backward Euler step to get to static equilibrium quickly.
    test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
    real constexpr kTimeInterval = 1e6_r;
    int constexpr kNumSteps = 1;

    real constexpr kTimeStep = kTimeInterval / (real)kNumSteps;
    for (int i = 0; i < kNumSteps; ++i) {
      _scene->Step(kTimeStep);
    }

    // Check the maximum deflection at the end of the time stepping, and compare against the
    // expected solution from Euler--Bernoulli beam theory.
    int const numDofs = _actor->GetNumDofs();
    int const numNodes = numDofs / 3;
    DynamicArray<real> dofValues(numDofs);
    _actor->GetDofValues({}, dofValues, ErrorAssert{});

    // This is an analytical exact solution to linear beam theory.  We do not expect to match this
    // exactly, and check it with a loose tolerance.
    real const I = Pow(kThickness, 3_r) * kScale / 12_r;
    real const w = kGravityMagnitude * kDensity2d * kScale;
    real const expectedExact = -5_r * w * Pow(kScale, 4_r) / (384_r * kYoungsModulus3d * I);
    real minZDisplacement = 0_r;
    for (int nodeIndex = 0; nodeIndex < numNodes; nodeIndex++) {
      minZDisplacement = Min(dofValues[3 * nodeIndex + 2], minZDisplacement);
    }
    EXPECT_NEAR(expectedExact, minZDisplacement, kTolerance * Abs(expectedExact));
    MOCHI_LOG_VERBOSE("Computed displacement = %.60lf", (double)minZDisplacement);
    PerTestTearDown();
  }
};

TEST_F(MochiShellActorSimplySupportedBeamScene, ShellActorSimplySupportedBeam) {
  RunTest();
}

class MochiShellActorSimplySupportedPlateScene : public test::MochiSceneTestBase {
 protected:
  Actor* _actor = nullptr;

 public:
  static constexpr real kScale = 1_r;
  static constexpr int kM = 32;
  static constexpr int kN = 32;
  static constexpr real kYoungsModulus3d = 4.8e5_r;
  static constexpr real kPoissonsRatio3d = 0.38_r;
  static constexpr real kDensity2d = 90_r;
  static constexpr real kThickness = 1e-1_r;
  static constexpr real kGravityMagnitude = 1_r;
  static constexpr real kBcEps = 1e-3_r;
  static constexpr real kTolerance = 5e-2;

  void PerTestSetUp() {
    auto& reg = GetRegistry();
    auto&& [coordinates, connectivity] =
        UniformSquareTriangularMeshData(Int2{kM, kN}, Real2{kScale, kScale});
    ShellActorParams params;

    params.material = mochi::experimental::ShellMaterialParamsFrom3dIsotropic(
        kYoungsModulus3d, kPoissonsRatio3d, kDensity2d / kThickness, kThickness, ErrorAssert{});

    params.shape = _scene->GetContext()->CreateTriMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), ErrorAssert{});
    _actor = CreateShellActor(_scene, params, ErrorAssert{});
    EXPECT_NE(entt::entity{}, mochi::GetEntity(reg, _actor->GetHandle(), test::ExpectOK{}));
  }
  void PerTestTearDown() {
    _scene->DestroyActor(_actor);
  }

  void RunTest() {
    PerTestSetUp();
    _scene->SetGravity({0_r, 0_r, -kGravityMagnitude});
    experimental::ConstrainNodesByPosition(
        _actor,
        [](int, Real3 const& x) -> bool {
          return (Abs(x[0]) > 0.5_r * kScale - kBcEps) || (Abs(x[1]) > 0.5_r * kScale - kBcEps);
        },
        ErrorAssert{});

    // Take one large backward Euler steps to get to static equilibrium quickly.
    test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
    real constexpr kTimeInterval = 1e6_r;
    int constexpr kNumSteps = 1;

    real constexpr kTimeStep = kTimeInterval / (real)kNumSteps;
    for (int i = 0; i < kNumSteps; ++i) {
      _scene->Step(kTimeStep);
    }

    // Check the maximum deflection at the end of the time stepping, and compare against a reference
    // value from the literature.
    int const numDofs = _actor->GetNumDofs();
    int const numNodes = numDofs / 3;
    DynamicArray<real> dofValues(numDofs);
    _actor->GetDofValues({}, dofValues, ErrorAssert{});
    // The "exact" expected value is taken from a high-resolution reference computation in the
    // literature, using a different code.  We only expect to match this up to discretization error,
    // and therefore test the corresponding assert with a loose tolerance.
    real constexpr kExpectedExact = -0.0078_r;
    // Gold values to detect smaller changes, with some potential for false positives.
    real constexpr kExpectedGold = -0.00754_r;
    real constexpr kGoldTol = 2e-5_r;
    real minZDisplacement = 0_r;
    for (int nodeIndex = 0; nodeIndex < numNodes; nodeIndex++) {
      minZDisplacement = Min(dofValues[3 * nodeIndex + 2], minZDisplacement);
    }

    EXPECT_NEAR(kExpectedExact, minZDisplacement, kTolerance * Abs(kExpectedExact));
    EXPECT_NEAR(kExpectedGold, minZDisplacement, kGoldTol);
    MOCHI_LOG_VERBOSE("Computed displacement = %.60lf", (double)minZDisplacement);
    PerTestTearDown();
  }
};

TEST_F(MochiShellActorSimplySupportedPlateScene, ShellActorSimplySupportedPlate) {
  RunTest();
}

namespace {
ShellMaterialParams MakeValidShellMaterialParams() {
  ShellMaterialParams p;
  p.membraneLambda = 30_r;
  p.membraneMu = 40_r;
  p.bendingAlpha = 1e-5_r;
  p.bendingBeta = 5e-5_r;
  p.density = 1_r;
  return p;
}
} // namespace

TEST(ShellMaterialValidation, ValidateShellMaterialParams) {
  // Valid baseline passes.
  ValidateShellMaterialParams(MakeValidShellMaterialParams(), test::ExpectOK{});

  // Each bound, one violation each.
  auto reject = [](auto mutate) {
    ShellMaterialParams p = MakeValidShellMaterialParams();
    mutate(p);
    ValidateShellMaterialParams(p, test::ExpectNotOK{});
  };
  reject([](ShellMaterialParams& p) { p.membraneMu = 0_r; }); // μ > 0
  reject([](ShellMaterialParams& p) { p.membraneLambda = -p.membraneMu; }); // λ > -μ (strict)
  reject([](ShellMaterialParams& p) { p.bendingBeta = 0_r; }); // β > 0
  reject(
      [](ShellMaterialParams& p) { p.bendingAlpha = -p.bendingBeta / 2_r; }); // α > -β/2 (strict)
  reject([](ShellMaterialParams& p) { p.density = 0_r; }); // ρ > 0
  reject([](ShellMaterialParams& p) {
    p.membraneMu = std::numeric_limits<real>::quiet_NaN();
  }); // finite
}

TEST(ShellMaterialValidation, ShellMaterialParamsFrom3dIsotropic) {
  real constexpr kE = 1e5_r;
  real constexpr kNu = 0.25_r;
  real constexpr kRho = 1e3_r;
  real constexpr kT = 1e-2_r;

  // Round-trip: valid 3D inputs produce valid ShellMaterialParams.
  ShellMaterialParams const params =
      ShellMaterialParamsFrom3dIsotropic(kE, kNu, kRho, kT, test::ExpectOK{});
  ValidateShellMaterialParams(params, test::ExpectOK{});

  // Each input bound, one violation each.
  ShellMaterialParamsFrom3dIsotropic(0_r, kNu, kRho, kT, test::ExpectNotOK{}); // E > 0
  ShellMaterialParamsFrom3dIsotropic(kE, -1_r, kRho, kT, test::ExpectNotOK{}); // ν > -1
  ShellMaterialParamsFrom3dIsotropic(kE, 0.5_r, kRho, kT, test::ExpectNotOK{}); // ν < 0.5
  ShellMaterialParamsFrom3dIsotropic(kE, kNu, 0_r, kT, test::ExpectNotOK{}); // ρ > 0
  ShellMaterialParamsFrom3dIsotropic(kE, kNu, kRho, 0_r, test::ExpectNotOK{}); // t > 0
}

class MochiShellCreateActorValidationScene : public test::MochiSceneTestBase {};

TEST_F(MochiShellCreateActorValidationScene, CreateShellActor_RejectsInvalidParams) {
  auto&& [coords, conn] = test::CreateMinimalTriMeshUnitCube();
  ShellActorParams params;
  params.shape = _scene->GetContext()->CreateTriMeshShape(
      Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), ErrorAssert{});
  EXPECT_NE(nullptr, CreateShellActor(_scene, params, test::ExpectOK{}));

  ShellActorParams invalidMaterial = params;
  invalidMaterial.material.membraneMu = -1_r;
  EXPECT_EQ(nullptr, CreateShellActor(_scene, invalidMaterial, test::ExpectNotOK{}));

  coords.emplace_back(2_r, 2_r, 2_r); // Not referenced by any triangle.
  params.shape = _scene->GetContext()->CreateTriMeshShape(
      Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), ErrorAssert{});
  EXPECT_EQ(nullptr, CreateShellActor(_scene, params, test::ExpectNotOK{}));
}

// ---------------------------------------------------------------------------
// Shell damping validation tests
// ---------------------------------------------------------------------------

TEST(ShellMaterialValidation, ValidateShellDampingCoefficients) {
  {
    ShellMaterialParams p = MakeValidShellMaterialParams();
    p.massDampingCoefficient = 5_r;
    p.stiffnessDampingCoefficient = 0.01_r;
    ValidateShellMaterialParams(p, test::ExpectOK{});
  }
  auto reject = [](auto mutate) {
    ShellMaterialParams p = MakeValidShellMaterialParams();
    mutate(p);
    ValidateShellMaterialParams(p, test::ExpectNotOK{});
  };
  reject([](ShellMaterialParams& p) { p.massDampingCoefficient = -1_r; });
  reject([](ShellMaterialParams& p) { p.stiffnessDampingCoefficient = -1_r; });
  reject([](ShellMaterialParams& p) {
    p.massDampingCoefficient = std::numeric_limits<real>::quiet_NaN();
  });
  reject([](ShellMaterialParams& p) {
    p.stiffnessDampingCoefficient = std::numeric_limits<real>::quiet_NaN();
  });
}

// ---------------------------------------------------------------------------
// Mass damping: backward Euler velocity decay
// ---------------------------------------------------------------------------

// Creates a flat shell with zero stiffness, no gravity, uniform initial velocity, and backward
// Euler integration. With mass damping α, the velocity should decay as v_n = v_0 / (1 + α·dt)^n.
class MochiShellMassDampingScene : public test::MochiSceneTestBase {
 protected:
  Actor* _actor = nullptr;
  int _numDofs = 0;
  static constexpr int kDofsPerNode = 3;
  static constexpr real kV0 = 1_r;

  Actor* CreateDampedShellActor(real massDampingCoefficient) {
    auto&& [coords, conn] = test::CreateMinimalTriMeshUnitCube();
    ShellActorParams params;
    params.material.membraneLambda = 0_r;
    params.material.membraneMu = 1e-10_r; // Near-zero but valid (must be > 0).
    params.material.bendingAlpha = 0_r;
    params.material.bendingBeta = 1e-10_r; // Near-zero but valid (must be > 0).
    params.material.density = 1_r;
    params.material.massDampingCoefficient = massDampingCoefficient;
    params.hasGravity = false;
    params.colliderType = ColliderType::None;
    params.shape = _scene->GetContext()->CreateTriMeshShape(
        Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), ErrorAssert{});
    return CreateShellActor(_scene, params, ErrorAssert{});
  }

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();
    auto solverParams = _scene->GetSolverParams();
    solverParams.integrationMethod = IntegrationMethod::BackwardEuler;
    _scene->SetSolverParams(solverParams, ErrorAssert{});
  }

  void InitWithDamping(real massDampingCoefficient) {
    _actor = CreateDampedShellActor(massDampingCoefficient);
    _numDofs = _actor->GetNumDofs();

    // Uniform initial velocity in x-direction.
    DynamicArray<real> vel(_numDofs, 0_r);
    for (int i = 0; i < _numDofs; i += kDofsPerNode) {
      vel[i] = kV0;
    }
    _actor->SetNodeVelocitiesLocal(MakeSpan(vel), ErrorAssert{});
  }
};

TEST_F(MochiShellMassDampingScene, VelocityDecay) {
  real constexpr kAlpha = 3_r;
  real constexpr kDt = 0.01_r;
  int constexpr kNumSteps = 10;
  real constexpr kVelRtol = 1e-4_r;
  InitWithDamping(kAlpha);

  // Step and track displacement to compute velocity via finite differences.
  DynamicArray<real> prevDispl(_numDofs, 0_r);
  DynamicArray<real> currDispl(_numDofs);
  for (int step = 0; step < kNumSteps; ++step) {
    _scene->Step(kDt);
    _actor->GetDofValues({}, MakeSpan(currDispl), ErrorAssert{});

    // Velocity = (d_curr - d_prev) / dt.
    real const expectedVel = kV0 / std::pow(1_r + kAlpha * kDt, step + 1);
    for (int i = 0; i < _numDofs; i += kDofsPerNode) {
      real const vel = (currDispl[i] - prevDispl[i]) / kDt;
      EXPECT_NEAR_RTOL(expectedVel, vel, kVelRtol);
    }
    std::copy(currDispl.begin(), currDispl.end(), prevDispl.begin());
  }
}

TEST_F(MochiShellMassDampingScene, UndampedVelocityConserved) {
  real constexpr kDt = 0.01_r;
  int constexpr kNumSteps = 5;
  real constexpr kVelRtol = 1e-4_r;
  InitWithDamping(0_r); // No damping.

  DynamicArray<real> prevDispl(_numDofs, 0_r);
  DynamicArray<real> currDispl(_numDofs);
  for (int step = 0; step < kNumSteps; ++step) {
    _scene->Step(kDt);
    _actor->GetDofValues({}, MakeSpan(currDispl), ErrorAssert{});

    for (int i = 0; i < _numDofs; i += kDofsPerNode) {
      real const vel = (currDispl[i] - prevDispl[i]) / kDt;
      EXPECT_NEAR_RTOL(kV0, vel, kVelRtol);
    }
    std::copy(currDispl.begin(), currDispl.end(), prevDispl.begin());
  }
}

// ---------------------------------------------------------------------------
// Stiffness damping: rigid translation invariance
// ---------------------------------------------------------------------------

class MochiShellStiffnessDampingScene : public test::MochiSceneTestBase {
 protected:
  Actor* _actor = nullptr;
  int _numDofs = 0;
  static constexpr real kV0 = 1_r;

  Actor* CreateStiffnessDampedShellActor(real stiffnessDampingCoefficient) {
    auto&& [coords, conn] = test::CreateMinimalTriMeshUnitCube();
    ShellActorParams params;
    params.material.membraneLambda = 40_r;
    params.material.membraneMu = 40_r;
    params.material.bendingAlpha = 1e-5_r;
    params.material.bendingBeta = 5e-5_r;
    params.material.density = 1_r;
    params.material.stiffnessDampingCoefficient = stiffnessDampingCoefficient;
    params.hasGravity = false;
    params.colliderType = ColliderType::None;
    params.shape = _scene->GetContext()->CreateTriMeshShape(
        Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), ErrorAssert{});
    return CreateShellActor(_scene, params, ErrorAssert{});
  }

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();
    auto solverParams = _scene->GetSolverParams();
    solverParams.integrationMethod = IntegrationMethod::BackwardEuler;
    _scene->SetSolverParams(solverParams, ErrorAssert{});
  }

  void InitWithDamping(real stiffnessDampingCoefficient) {
    _actor = CreateStiffnessDampedShellActor(stiffnessDampingCoefficient);
    _numDofs = _actor->GetNumDofs();
  }
};

TEST_F(MochiShellStiffnessDampingScene, StiffnessProportionalDecayRatio) {
  // Verify the *magnitude* of stiffness-proportional damping via a single backward Euler step on
  // a 1-element equilateral triangle in the stiff limit. In that limit, the ratio of damped to
  // undamped displacement of a single free node is dt/(dt+β), independent of stiffness and mass.
  static constexpr int kDofsPerNode = 3;
  static constexpr real kBeta = 1_r;
  static constexpr real kDt = 1_r;
  static constexpr int kFreeNode = 2;
  static constexpr int kFreeNodeYDof = kFreeNode * kDofsPerNode + 1;
  static constexpr real kRatioRtol = 1e-2_r;
  static constexpr real kSymmetryRtol = 1e-4_r;
  static constexpr real kStiffLimitBound = 1e-2_r;

  DynamicArray<Real3> coords = {
      Real3{0_r, 0_r, 0_r}, Real3{1_r, 0_r, 0_r}, Real3{0.5_r, kSqrt3Over2, 0_r}};
  DynamicArray<Int3> conn = {Int3{0, 1, 2}};
  auto shape = _scene->GetContext()->CreateTriMeshShape(
      Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), ErrorAssert{});

  auto createActor = [&](real stiffnessDampingCoeff) -> Actor* {
    ShellActorParams params;
    params.material.membraneLambda = 1e4_r;
    params.material.membraneMu = 1e4_r;
    params.material.density = 1_r;
    params.material.stiffnessDampingCoefficient = stiffnessDampingCoeff;
    params.hasGravity = false;
    params.colliderType = ColliderType::None;
    params.shape = shape;
    Actor* actor = CreateShellActor(_scene, params, ErrorAssert{});

    ConstrainNodesByPosition(actor, [](int i, Real3 const&) { return i <= 1; }, ErrorAssert{});

    int const numDofs = actor->GetNumDofs();
    DynamicArray<real> vel(numDofs, 0_r);
    vel[kFreeNodeYDof] = kV0;
    actor->SetNodeVelocitiesLocal(MakeSpan(vel), ErrorAssert{});

    return actor;
  };

  Actor* undampedActor = createActor(0_r);
  Actor* dampedActor = createActor(kBeta);

  _scene->Step(kDt);

  int const numDofs = undampedActor->GetNumDofs();
  DynamicArray<real> undampedDispl(numDofs);
  DynamicArray<real> dampedDispl(numDofs);
  undampedActor->GetDofValues({}, MakeSpan(undampedDispl), ErrorAssert{});
  dampedActor->GetDofValues({}, MakeSpan(dampedDispl), ErrorAssert{});

  real const dUndamped = undampedDispl[kFreeNodeYDof];
  real const dDamped = dampedDispl[kFreeNodeYDof];

  // Guard: both displacements must be positive and non-trivial, ruling out unconverged solves.
  EXPECT_GT(dUndamped, 0_r);
  EXPECT_GT(dDamped, 0_r);

  // Stiff-limit check: displacement << v0 * dt confirms ω²dt² >> 1.
  EXPECT_LT(dUndamped, kStiffLimitBound * kV0 * kDt);

  // 1D symmetry: x and z displacements of node 2 should be negligible vs. y.
  real const symmetryTol = kSymmetryRtol * dUndamped;
  EXPECT_NEAR_TOL(0_r, undampedDispl[kFreeNode * kDofsPerNode + 0], symmetryTol);
  EXPECT_NEAR_TOL(0_r, undampedDispl[kFreeNode * kDofsPerNode + 2], symmetryTol);
  EXPECT_NEAR_TOL(0_r, dampedDispl[kFreeNode * kDofsPerNode + 0], symmetryTol);
  EXPECT_NEAR_TOL(0_r, dampedDispl[kFreeNode * kDofsPerNode + 2], symmetryTol);

  // Ratio check: d_damped / d_undamped ≈ dt / (dt + β).
  real const expectedRatio = kDt / (kDt + kBeta);
  EXPECT_NEAR_RTOL(expectedRatio, dDamped / dUndamped, kRatioRtol);
}

TEST_F(MochiShellStiffnessDampingScene, BendingStiffnessProportionalDecayRatio) {
  // Same stiff-limit ratio technique as StiffnessProportionalDecayRatio, but exercising bending
  // rather than membrane stiffness. A 4-triangle equilateral patch with the central triangle fixed
  // and the three outer nodes given out-of-plane velocity. The 3-fold rotational symmetry reduces
  // the system to a single effective DOF.
  static constexpr int kDofsPerNode = 3;
  static constexpr real kBeta = 1_r;
  static constexpr real kDt = 1_r;
  static constexpr int kFirstOuterNode = 3;
  static constexpr int kNumOuterNodes = 3;
  static constexpr int kZComponent = 2;
  static constexpr real kRatioRtol = 1e-2_r;
  static constexpr real kSymmetryRtol = 1e-4_r;
  static constexpr real kStiffLimitBound = 1e-2_r;

  DynamicArray<Real3> coords = {
      Real3{0_r, 0_r, 0_r},
      Real3{1_r, 0_r, 0_r},
      Real3{0.5_r, kSqrt3Over2, 0_r},
      Real3{1.5_r, kSqrt3Over2, 0_r},
      Real3{-0.5_r, kSqrt3Over2, 0_r},
      Real3{0.5_r, -kSqrt3Over2, 0_r}};
  DynamicArray<Int3> conn = {Int3{0, 1, 2}, Int3{1, 3, 2}, Int3{2, 4, 0}, Int3{1, 0, 5}};
  auto shape = _scene->GetContext()->CreateTriMeshShape(
      Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), ErrorAssert{});

  auto createActor = [&](real stiffnessDampingCoeff) -> Actor* {
    ShellActorParams params;
    params.material.membraneLambda = 40_r;
    params.material.membraneMu = 40_r;
    params.material.bendingAlpha = 1e4_r;
    params.material.bendingBeta = 1e4_r;
    params.material.density = 1_r;
    params.material.stiffnessDampingCoefficient = stiffnessDampingCoeff;
    params.hasGravity = false;
    params.colliderType = ColliderType::None;
    params.shape = shape;
    Actor* actor = CreateShellActor(_scene, params, ErrorAssert{});

    ConstrainNodesByPosition(actor, [](int i, Real3 const&) { return i <= 2; }, ErrorAssert{});

    int const numDofs = actor->GetNumDofs();
    DynamicArray<real> vel(numDofs, 0_r);
    for (int n = 0; n < kNumOuterNodes; ++n) {
      vel[(kFirstOuterNode + n) * kDofsPerNode + kZComponent] = kV0;
    }
    actor->SetNodeVelocitiesLocal(MakeSpan(vel), ErrorAssert{});

    return actor;
  };

  Actor* undampedActor = createActor(0_r);
  Actor* dampedActor = createActor(kBeta);

  _scene->Step(kDt);

  int const numDofs = undampedActor->GetNumDofs();
  DynamicArray<real> undampedDispl(numDofs);
  DynamicArray<real> dampedDispl(numDofs);
  undampedActor->GetDofValues({}, MakeSpan(undampedDispl), ErrorAssert{});
  dampedActor->GetDofValues({}, MakeSpan(dampedDispl), ErrorAssert{});

  // Guard: all z-displacements of outer nodes must be positive.
  for (int n = 0; n < kNumOuterNodes; ++n) {
    int const zDof = (kFirstOuterNode + n) * kDofsPerNode + kZComponent;
    EXPECT_GT(undampedDispl[zDof], 0_r);
    EXPECT_GT(dampedDispl[zDof], 0_r);
  }

  // 3-fold symmetry: all three outer nodes should have the same z-displacement.
  real dUndampedSum = 0_r;
  real dDampedSum = 0_r;
  for (int n = 0; n < kNumOuterNodes; ++n) {
    int const zDof = (kFirstOuterNode + n) * kDofsPerNode + kZComponent;
    dUndampedSum += undampedDispl[zDof];
    dDampedSum += dampedDispl[zDof];
  }
  real const dUndamped = dUndampedSum / kNumOuterNodes;
  real const dDamped = dDampedSum / kNumOuterNodes;
  for (int n = 0; n < kNumOuterNodes; ++n) {
    int const zDof = (kFirstOuterNode + n) * kDofsPerNode + kZComponent;
    EXPECT_NEAR_RTOL(dUndamped, undampedDispl[zDof], kSymmetryRtol);
    EXPECT_NEAR_RTOL(dDamped, dampedDispl[zDof], kSymmetryRtol);
  }

  // In-plane displacements should be negligible relative to z.
  real const symmetryTol = kSymmetryRtol * dUndamped;
  for (int n = 0; n < kNumOuterNodes; ++n) {
    int const baseDof = (kFirstOuterNode + n) * kDofsPerNode;
    EXPECT_NEAR_TOL(0_r, undampedDispl[baseDof + 0], symmetryTol);
    EXPECT_NEAR_TOL(0_r, undampedDispl[baseDof + 1], symmetryTol);
    EXPECT_NEAR_TOL(0_r, dampedDispl[baseDof + 0], symmetryTol);
    EXPECT_NEAR_TOL(0_r, dampedDispl[baseDof + 1], symmetryTol);
  }

  // Stiff-limit check: displacement << v0 * dt confirms ω²dt² >> 1.
  EXPECT_LT(dUndamped, kStiffLimitBound * kV0 * kDt);

  // Ratio check: d_damped / d_undamped ≈ dt / (dt + β).
  real const expectedRatio = kDt / (kDt + kBeta);
  EXPECT_NEAR_RTOL(expectedRatio, dDamped / dUndamped, kRatioRtol);
}

TEST_F(MochiShellStiffnessDampingScene, RigidTranslationUndamped) {
  // Uniform velocity → zero strain rate → stiffness damping should have no effect.
  // Velocity should be conserved (same as undamped).
  static constexpr int kDofsPerNode = 3;
  real constexpr kBeta = 0.1_r;
  real constexpr kDt = 0.01_r;
  int constexpr kNumSteps = 5;
  real constexpr kVelRtol = 1e-3_r;
  InitWithDamping(kBeta);

  // Set uniform initial velocity.
  DynamicArray<real> vel(_numDofs, 0_r);
  for (int i = 0; i < _numDofs; i += kDofsPerNode) {
    vel[i] = kV0;
  }
  _actor->SetNodeVelocitiesLocal(MakeSpan(vel), ErrorAssert{});

  DynamicArray<real> prevDispl(_numDofs, 0_r);
  DynamicArray<real> currDispl(_numDofs);
  for (int step = 0; step < kNumSteps; ++step) {
    _scene->Step(kDt);
    _actor->GetDofValues({}, MakeSpan(currDispl), ErrorAssert{});
    for (int i = 0; i < _numDofs; i += kDofsPerNode) {
      real const velX = (currDispl[i] - prevDispl[i]) / kDt;
      EXPECT_NEAR_RTOL(kV0, velX, kVelRtol);
    }
    std::copy(currDispl.begin(), currDispl.end(), prevDispl.begin());
  }
}

// Cantilever shell strip in uniaxial tension under a tip end-load applied via
// SetExternalForcesOnDofs. Verifies (a) the API accepts shell actors, and (b) world-frame nodal
// forces are correctly rotated into the local frame at assembly time and produce the equilibrium
// stretch predicted by the exact St. Venant--Kirchhoff stretch-from-load relation.
//
// Parameterized on whether external forces are supplied as full per-node vectors (three
// consecutive in-order DoFs per node, exercising the fast path in
// `deformable::details::AddTranslationalExternalForceEntry`) or as scalar y-components only
// (one DoF per node with `component == 1`, exercising the slow / sparse path). The fixture's
// `worldFromLocal` rotation is about the Y axis so that local-y aligns with world-Y, making the
// two parameterizations physically equivalent.
class MochiShellActorExternalForcesScene : public test::MochiSceneTestBase,
                                           public ::testing::WithParamInterface<bool> {
 protected:
  Actor* _actor = nullptr;
  TransformRT _worldFromLocal;
  std::vector<Real3> _nodeCoordsLocal;

 public:
  static constexpr real kScale = 2_r;
  static constexpr int kM = 3;
  static constexpr int kN = 7;
  static constexpr real kYoungsModulus3d = 1e6_r;
  static constexpr real kPoissonsRatio3d = 0_r;
  static constexpr real kDensity3d = 1e3_r;
  static constexpr real kThickness = 1e-2_r;
  static constexpr real kBcEps = 1e-3_r;
  // Total end-load applied at the tip along the local strip axis (-y) to pull the free tip
  // away from the fixed top edge in tension.
  static constexpr real kTipForce = 1e1_r;
  // Loose tolerance: linear-elastic prediction is exact only in the small-strain limit.
  static constexpr real kLooseTolerance = 5e-2_r;
  // Tight tolerance: the St. Venant--Kirchhoff stretch-from-load relation is exact (and the
  // uniform-strain solution is exactly representable on this linear-FE mesh).
  static constexpr real kTightTolerance = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-3_r;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();
    // Rotation about Y only, so that the local y-axis (the strip's axial direction) maps to
    // world Y. This keeps both parameterizations of the end-load test physically equivalent
    // while still exercising the X-Z rotation entries of `worldFromLocalR` on node positions.
    _worldFromLocal = TransformRT{
        Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, kPI / 5_r), Real3{-0.5_r, 1.2_r, 2.1_r}};

    auto&& [coordinates, connectivity] =
        UniformSquareTriangularMeshData(Int2{kM, kN}, Real2{kScale, kScale});
    _nodeCoordsLocal = coordinates;
    ShellActorParams params;
    params.worldFromLocal = _worldFromLocal;
    params.material = mochi::experimental::ShellMaterialParamsFrom3dIsotropic(
        kYoungsModulus3d, kPoissonsRatio3d, kDensity3d, kThickness, ErrorAssert{});
    params.shape = _scene->GetContext()->CreateTriMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), ErrorAssert{});
    _actor = CreateShellActor(_scene, params, ErrorAssert{});
  }

  void TearDown() override {
    _scene->DestroyActor(_actor);
    test::MochiSceneTestBase::TearDown();
  }
};

TEST_P(MochiShellActorExternalForcesScene, ShellActorUniaxialEndLoad) {
  bool const useScalarYOnly = GetParam();

  _scene->SetGravity({0_r, 0_r, 0_r});

  // Identify the top-edge (fixed) and bottom-edge (free) nodes from the source mesh in local
  // coordinates, since classifying nodes in world space would be brittle under non-identity
  // `worldFromLocal`.
  int const numNodes = isize(_nodeCoordsLocal);
  DynamicArray<int> topNodes;
  DynamicArray<int> tipNodes;
  for (int n = 0; n < numNodes; ++n) {
    real const y = _nodeCoordsLocal[n][1];
    if (y > 0.5_r * kScale - kBcEps) {
      topNodes.push_back(n);
    }
    if (y < -0.5_r * kScale + kBcEps) {
      tipNodes.push_back(n);
    }
  }
  ASSERT_FALSE(topNodes.empty());
  ASSERT_FALSE(tipNodes.empty());

  // Fix the top edge: BCs are specified as world positions.
  DynamicArray<real> bcNodePositionsWorld;
  for (int const n : topNodes) {
    Real3 const worldPos = _worldFromLocal.TransformPoint(_nodeCoordsLocal[n]);
    for (int d = 0; d < 3; ++d) {
      bcNodePositionsWorld.push_back(worldPos[d]);
    }
  }
  _actor->AddBoundaryConditionNodesWorld(
      MakeConstSpan(topNodes), MakeConstSpan(bcNodePositionsWorld), ErrorAssert{});

  // Apply a uniformly distributed pull along the local -y axis at the bottom edge, expressed in
  // world coordinates. This stretches the strip in tension (pulling the free tip away from the
  // fixed top edge). Compute consistent nodal forces by splitting the total tip load evenly
  // between the edges along the tip, then giving each node half the load of each of its
  // incident edges, so that corner nodes receive half the load of interior nodes.
  Real3 const forceLocalDir{0_r, -1_r, 0_r};
  Real3 const forceWorldDir = _worldFromLocal.TransformDirection(forceLocalDir);
  std::sort(tipNodes.begin(), tipNodes.end(), [this](int a, int b) {
    return _nodeCoordsLocal[a][0] < _nodeCoordsLocal[b][0];
  });
  int const numTipEdges = isize(tipNodes) - 1;
  ASSERT_GT(numTipEdges, 0);
  real const halfForcePerEdge = 0.5_r * kTipForce / static_cast<real>(numTipEdges);
  DynamicArray<real> forceMagPerNode(tipNodes.size(), 0_r);
  for (int e = 0; e < numTipEdges; ++e) {
    forceMagPerNode[e] += halfForcePerEdge;
    forceMagPerNode[e + 1] += halfForcePerEdge;
  }
  DynamicArray<int> forceDofs;
  DynamicArray<real> forceValues;
  for (int i = 0; i < isize(tipNodes); ++i) {
    int const n = tipNodes[i];
    if (useScalarYOnly) {
      // Slow path: a single DoF per node with `component == 1`. Equivalent to the full-vector
      // case because the fixture's Y-axis rotation makes `forceWorldDir[0] == forceWorldDir[2]
      // == 0`.
      forceDofs.push_back(3 * n + 1);
      forceValues.push_back(forceMagPerNode[i] * forceWorldDir[1]);
    } else {
      // Fast path: three consecutive in-order DoFs per node.
      for (int d = 0; d < 3; ++d) {
        forceDofs.push_back(3 * n + d);
        forceValues.push_back(forceMagPerNode[i] * forceWorldDir[d]);
      }
    }
  }
  _actor->SetExternalForcesOnDofs(
      MakeConstSpan(forceDofs), MakeConstSpan(forceValues), ErrorAssert{});

  // Single large backward-Euler step to reach static equilibrium.
  test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
  _scene->Step(1e2_r);

  int const numDofs = _actor->GetNumDofs();
  DynamicArray<real> dofValues(numDofs);
  _actor->GetDofValues({}, MakeSpan(dofValues), ErrorAssert{});

  // Average the tip's local-y displacement and track the maximum off-axis (local x, z)
  // displacement at the tip.
  real avgTipDisplY = 0_r;
  real maxTipDisplOffAxis = 0_r;
  for (int const n : tipNodes) {
    avgTipDisplY += dofValues[3 * n + 1];
    maxTipDisplOffAxis =
        Max(maxTipDisplOffAxis, Sqrt(Sqr(dofValues[3 * n + 0]) + Sqr(dofValues[3 * n + 2])));
  }
  avgTipDisplY /= static_cast<real>(tipNodes.size());
  // The applied force is in local -y, so the tip moves in local -y and the strip's elongation
  // is the negative of the tip's local-y displacement.
  real const tipElongation = -avgTipDisplY;

  // Expected uniaxial-tension elongation: Δ = F · L / (E · A), with A = width × thickness. The
  // shell strip has length L = kScale (between the top BC and the free tip) and width kScale.
  // This linear formula is only exact in the small-strain limit, so it uses the loose tolerance.
  real constexpr kExpected = kTipForce * kScale / (kYoungsModulus3d * kScale * kThickness);
  EXPECT_NEAR(kExpected, tipElongation, kLooseTolerance * Abs(kExpected));
  EXPECT_NEAR(0_r, maxTipDisplOffAxis, kLooseTolerance * Abs(kExpected));

  // Stretch should satisfy this exactly for the Green--Lagrange-strain-based St. Venant--Kirchhoff
  // constitutive model with zero Poisson's ratio, so it uses the tight tolerance. The 2D
  // membrane analogue of the rod's E·A is E · width · thickness.
  real const actualLength = kScale + tipElongation;
  real const stretchRatio = actualLength / kScale;
  real const glStrain = 0.5_r * (Sqr(stretchRatio) - 1_r);
  real constexpr kAxialStiffness = kYoungsModulus3d * kScale * kThickness;
  EXPECT_NEAR(
      kTipForce, kAxialStiffness * glStrain * stretchRatio, kTightTolerance * Abs(kTipForce));
}

INSTANTIATE_TEST_SUITE_P(
    DofLayout,
    MochiShellActorExternalForcesScene,
    ::testing::Values(false, true),
    [](::testing::TestParamInfo<bool> const& info) {
      return info.param ? "ScalarYOnly" : "FullVector";
    });
class MochiShellContactSkinTest : public test::MochiSceneTestBase {
 protected:
  static constexpr std::array<Real3, 5> kPhysicsNodes = {
      Real3{0_r, 0_r, 0_r},
      Real3{1_r, 0_r, 0_r},
      Real3{1_r, 1_r, 0_r},
      Real3{0_r, 1_r, 0_r},
      Real3{0.5_r, 0.5_r, 0_r},
  };
  static constexpr std::array<Int3, 4> kPhysicsTriangles = {
      Int3{0, 1, 4},
      Int3{1, 2, 4},
      Int3{2, 3, 4},
      Int3{3, 0, 4}};
  static constexpr std::array<Real3, 3> kSkinNodes = {
      Real3{0.5_r, 0_r, 0_r},
      Real3{1_r, 0.25_r, 0_r},
      Real3{0.5_r, 1_r, 0_r},
  };
  static constexpr std::array<Int3, 1> kSkinTriangles = {Int3{0, 1, 2}};

  ShapeHandle CreateShape(bool includeContactSkin = true, bool extrapolateContactSkin = false) {
    ModelData model;
    model.mesh.emplace();
    model.mesh->nodesPerElement = 3;
    model.mesh->coordinates = DynamicArray<real>{Flatten(MakeConstSpan(kPhysicsNodes))};
    model.mesh->connectivity = DynamicArray<int>{Flatten(MakeConstSpan(kPhysicsTriangles))};

    if (includeContactSkin) {
      model.contactSkinMesh.emplace();
      model.contactSkinMesh->nodesPerElement = 3;
      auto skinNodes = kSkinNodes;
      if (extrapolateContactSkin) {
        skinNodes[0][0] = 2_r;
      }
      model.contactSkinMesh->coordinates = DynamicArray<real>{Flatten(MakeConstSpan(skinNodes))};
      model.contactSkinMesh->connectivity =
          DynamicArray<int>{Flatten(MakeConstSpan(kSkinTriangles))};
      model.contactSkinMesh->skinning.emplace();
      model.contactSkinMesh->skinning->weightsPerNode = 3;
      model.contactSkinMesh->skinning->indices = DynamicArray<int>{0, 0, 1, 1, 1, 2, 2, 3, 3};
      model.contactSkinMesh->skinning->weights = DynamicArray<real>{
          extrapolateContactSkin ? -0.5_r : 0.25_r,
          extrapolateContactSkin ? -0.5_r : 0.25_r,
          extrapolateContactSkin ? 2_r : 0.5_r,
          0.25_r,
          0.5_r,
          0.25_r,
          0.5_r,
          0.25_r,
          0.25_r};
    }
    return _mochiContext->CreateModelShape(model, test::ExpectOK{});
  }

  Actor* CreateActor(
      ShapeHandle shape,
      bool useContactSkin = false,
      ActorBoundaryElementType contactElementType = ActorBoundaryElementType::Default,
      ColliderType colliderType = ColliderType::None) {
    ShellActorParams params;
    params.shape = shape;
    params.material.density = 1_r;
    params.colliderType = colliderType;
    params.useContactSkin = useContactSkin;
    params.contactElementType = contactElementType;
    return experimental::CreateShellActor(_scene, params, test::ExpectOK{});
  }
};

TEST_F(MochiShellContactSkinTest, ContactSkinSelectionRequiresContactSkin) {
  ShellActorParams params;
  params.shape = CreateShape(/*includeContactSkin=*/false);
  params.useContactSkin = true;
  EXPECT_EQ(nullptr, experimental::CreateShellActor(_scene, params, test::ExpectNotOK{}));
}

TEST_F(MochiShellContactSkinTest, AuthoredSkinIsExposedIndependentlyOfContactSelection) {
  constexpr std::array kExpectedSkinningIndices = {0, 0, 1, 1, 1, 2, 2, 3, 3};
  for (bool const useContactSkin : {false, true}) {
    SCOPED_TRACE(useContactSkin);
    Actor* const actor = CreateActor(CreateShape(), useContactSkin);
    auto& reg = GetRegistry();
    entt::entity const entity = GetEntity(actor);

    EXPECT_EQ(isize(kPhysicsNodes), actor->GetMesh().GetNumNodes());
    EXPECT_EQ(isize(kPhysicsNodes), actor->GetSurfaceMesh().GetNumNodes());
    EXPECT_EQ(
        reg.get<CTriangularMesh const>(entity).mesh, reg.get<CSurfaceMesh const>(entity).mesh);
    EXPECT_EQ(
        useContactSkin, (reg.all_of<TagUseDeformableContactSkin, CContactSkinningData>(entity)));
    EXPECT_NE(
        useContactSkin, (reg.all_of<CContactLocal2GlobalMap, CContactNodalBasedStructure>(entity)));

    MeshDataView const contactSkin = actor->GetContactSkinMesh();
    EXPECT_SPAN_EQ(Flatten(MakeConstSpan(kSkinNodes)), contactSkin.coordinates);
    EXPECT_SPAN_EQ(Flatten(MakeConstSpan(kSkinTriangles)), contactSkin.connectivity);
    ASSERT_TRUE(contactSkin.skinning.has_value());
    EXPECT_EQ(3, contactSkin.skinning->weightsPerNode);
    EXPECT_SPAN_EQ(MakeConstSpan(kExpectedSkinningIndices), contactSkin.skinning->indices);
  }
}

TEST_F(MochiShellContactSkinTest, SurfaceQueriesFollowPhysicsMeshAndSkinQueriesFollowEmbedding) {
  std::array<Real3, 5> const displacements = {
      Real3{1_r, 0_r, 0_r},
      Real3{0_r, 2_r, 0_r},
      Real3{0_r, 0_r, 3_r},
      Real3{1_r, 1_r, 1_r},
      Real3{2_r, 0_r, 0_r},
  };
  std::array<Real3, 5> const expectedSurfacePositions = {
      Real3{1_r, 0_r, 0_r},
      Real3{1_r, 2_r, 0_r},
      Real3{1_r, 1_r, 3_r},
      Real3{1_r, 2_r, 1_r},
      Real3{2.5_r, 0.5_r, 0_r},
  };
  std::array<Real3, 3> const expectedSkinPositions = {
      Real3{1_r, 1_r, 0_r},
      Real3{1_r, 1.75_r, 0.75_r},
      Real3{1_r, 1.5_r, 2_r},
  };
  Real3 const expectedSkinNormal = Normalize(Cross(
      expectedSkinPositions[1] - expectedSkinPositions[0],
      expectedSkinPositions[2] - expectedSkinPositions[0]));

  for (bool const useContactSkin : {false, true}) {
    SCOPED_TRACE(useContactSkin);
    Actor* const actor = CreateActor(CreateShape(), useContactSkin);
    auto& reg = GetRegistry();
    entt::entity const entity = GetEntity(actor);
    actor->RegisterQueryAndCompute(QueryType::SurfaceNodePositions, test::ExpectOK{});
    actor->RegisterQueryAndCompute(QueryType::SurfaceNodeNormals, test::ExpectOK{});
    actor->RegisterQueryAndCompute(QueryType::ContactSkinNodePositions, test::ExpectOK{});
    actor->RegisterQueryAndCompute(QueryType::ContactSkinNodeNormals, test::ExpectOK{});

    auto& displacement = reg.get<CDisplacementSlice<real, TimeStep::Current>>(entity).value;
    auto const flatDisplacements = Flatten(MakeConstSpan(displacements));
    for (int i = 0; i < isize(flatDisplacements); ++i) {
      displacement(i) = flatDisplacements[i];
    }
    ecs::InvokeOnEntity(&UpdateQueryNodePositions, reg, entity);
    ecs::InvokeOnEntity(&UpdateQuerySurfaceNodePositions, reg, entity);
    ecs::InvokeOnEntity(&UpdateQuerySurfaceNodeNormals, reg, entity);
    ecs::InvokeOnEntity(&UpdateQueryContactSkinNodePositionsAndNormals, reg, entity);

    EXPECT_SPAN_EQ(
        MakeConstSpan(expectedSurfacePositions),
        Unflatten<Real3 const>(actor->GetSurfaceMeshNodePositionsLocal(test::ExpectOK{})));
    EXPECT_EQ(
        isize(Flatten(MakeConstSpan(expectedSurfacePositions))),
        isize(actor->GetSurfaceMeshNodeNormalsLocal(test::ExpectOK{})));
    EXPECT_SPAN_EQ(
        MakeConstSpan(expectedSkinPositions),
        Unflatten<Real3 const>(actor->GetContactSkinMeshNodePositionsLocal(test::ExpectOK{})));
    auto const skinNormals =
        Unflatten<Real3 const>(actor->GetContactSkinMeshNodeNormalsLocal(test::ExpectOK{}));
    ASSERT_EQ(expectedSkinPositions.size(), skinNormals.size());
    for (Real3 const& normal : skinNormals) {
      EXPECT_NEAR_EQ(expectedSkinNormal, normal);
    }
  }
}

TEST_F(MochiShellContactSkinTest, LinearSkinJacobianCombinesDuplicateInfluences) {
  Actor* const actor =
      CreateActor(CreateShape(), /*useContactSkin=*/true, ActorBoundaryElementType::P1Q3);
  auto& reg = GetRegistry();
  entt::entity const entity = GetEntity(actor);
  auto const& jacobian = reg.get<CContactSkinningData const>(entity).jacobian;

  std::array<std::array<int, 3>, 3> const expectedNodes = {
      std::array{0, 1, -1},
      std::array{1, 2, -1},
      std::array{2, 3, -1},
  };
  std::array<std::array<real, 3>, 3> const expectedWeights = {
      std::array{0.5_r, 0.5_r, 0_r},
      std::array{0.75_r, 0.25_r, 0_r},
      std::array{0.5_r, 0.5_r, 0_r},
  };

  EXPECT_EQ(3, jacobian.Rows());
  EXPECT_EQ(kSpaceDim3 * isize(kPhysicsNodes), jacobian.Cols());
  for (int row = 0; row < jacobian.Rows(); ++row) {
    DynamicArray<int> expectedColumns;
    DynamicArray<Real3> expectedValues;
    for (int influence = 0; influence < 3; ++influence) {
      int const node = expectedNodes[row][influence];
      if (node < 0) {
        continue;
      }
      for (int component = 0; component < kSpaceDim3; ++component) {
        expectedColumns.push_back(kSpaceDim3 * node + component);
        Real3 value{};
        value[component] = expectedWeights[row][influence];
        expectedValues.push_back(value);
      }
    }
    EXPECT_SPAN_EQ(MakeConstSpan(expectedColumns), jacobian.Indices(row));
    ASSERT_EQ(expectedValues.size(), jacobian.Values(row).size());
    for (int i = 0; i < isize(expectedValues); ++i) {
      EXPECT_NEAR_EQ(expectedValues[i], jacobian.Values(row)[i]);
    }
  }

  auto const& mapping = reg.get<CContactSkinningData const>(entity).columnCoalescingMap;
  ContactJacColumnCoalescingMap expectedMapping;
  expectedMapping.sourceBlockCount = 3;
  expectedMapping.samplesPerElement = 3;
  expectedMapping.elementDofOffsets = {0, 12};
  expectedMapping.elementDofIndices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  expectedMapping.elementSourceContributionOffsets = {0, 6, 12, 18};
  expectedMapping.elementSourceContributionDstColumns = {
      0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11};
  EXPECT_TRUE(mapping.IsInitialized());
  test::ExpectEqualCoalescingMaps(expectedMapping, mapping);

  CContactSkinningData rebuiltSkinningData;
  rebuiltSkinningData.jacobian = jacobian;
  InitializeContactSkinningColumnCoalescingMap(
      reg.get<CFemSurfaceDiscretization const>(entity), rebuiltSkinningData);
  test::ExpectEqualCoalescingMaps(expectedMapping, rebuiltSkinningData.columnCoalescingMap);
}

TEST_F(MochiShellContactSkinTest, ContactJacobianMatchesFiniteDifferences) {
  Actor* const actor = CreateActor(
      CreateShape(),
      /*useContactSkin=*/true,
      ActorBoundaryElementType::P1Q3);
  auto& reg = GetRegistry();
  entt::entity const entity = GetEntity(actor);

  ContactDetectionResult result;
  for (int sampleIndex = 0; sampleIndex < 3; ++sampleIndex) {
    result.sampleIndices.push_back(sampleIndex);
    result.jacColliderFromWorld.push_back(VEye<3>());
  }
  CCollJacs<CollRole::Colliding> collidingJacobians;
  collidingJacobians.emplace_back(
      /*isSync*/ false, &result, false, entt::null, /*collidingPartitionId=*/0);
  SetupContactSkinCollidingJacobians(
      {},
      reg.get<CFemSurfaceDiscretization const>(entity),
      reg.get<CRootTransform const>(entity),
      reg.get<CDofOffset const>(entity),
      reg.get<CContactSkinningData const>(entity),
      collidingJacobians);
  ContactJac const& contactJacobian = collidingJacobians[0].GetJacs()[0];
  EXPECT_EQ(12, contactJacobian.nDoFsInternal);
  EXPECT_EQ(12, contactJacobian.nDoFsState);
  EXPECT_EQ(3, contactJacobian.nContacts);

  test::ContactJacSnapshot const firstSetup = test::SnapshotContactJac(contactJacobian);
  SetupContactSkinCollidingJacobians(
      {},
      reg.get<CFemSurfaceDiscretization const>(entity),
      reg.get<CRootTransform const>(entity),
      reg.get<CDofOffset const>(entity),
      reg.get<CContactSkinningData const>(entity),
      collidingJacobians);
  test::ExpectEqualContactJacSnapshots(
      firstSetup, test::SnapshotContactJac(collidingJacobians[0].GetJacs()[0]));

  auto& displacement = reg.get<CDisplacementSlice<real, TimeStep::Current>>(entity).value;
  ColumnVector<real> const baseDisplacement = displacement;
  auto& samples = reg.get<CContactSamples<TimeStep::Current>>(entity);
  real constexpr kEpsilon = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-4_r;
  real constexpr kTolerance = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 5e-4_r;

  for (int dof = 0; dof < displacement.Rows(); ++dof) {
    displacement = baseDisplacement;
    displacement(dof) += kEpsilon;
    ecs::InvokeOnEntity(
        linear_contact_skin::UpdatePositionsAndSamples<TimeStep::Current>, reg, entity);
    auto const positionsPlus = samples.positions;

    displacement = baseDisplacement;
    displacement(dof) -= kEpsilon;
    ecs::InvokeOnEntity(
        linear_contact_skin::UpdatePositionsAndSamples<TimeStep::Current>, reg, entity);
    auto const& positionsMinus = samples.positions;

    for (int contact = 0; contact < contactJacobian.nContacts; ++contact) {
      int const sampleIndex = result.sampleIndices[contact];
      Real3 const finiteDifference =
          (positionsPlus[sampleIndex] - positionsMinus[sampleIndex]) / (2_r * kEpsilon);

      Real3 analytic{};
      auto const indices = contactJacobian.Inds(contact);
      for (int column = 0; column < isize(indices); ++column) {
        if (indices[column] == dof) {
          for (int component = 0; component < kSpaceDim3; ++component) {
            analytic[component] += contactJacobian.Jac(contact)(component, column);
          }
        }
      }
      for (int component = 0; component < kSpaceDim3; ++component) {
        EXPECT_NEAR(analytic[component], finiteDifference[component], kTolerance)
            << "Contact " << contact << ", DoF " << dof << ", component " << component;
      }
    }
  }
  displacement = baseDisplacement;
}

TEST_F(MochiShellContactSkinTest, ContactForceIsDistributedThroughSkinWeights) {
  // A half turn about x keeps the shell in the z = 0 plane but flips its local frame relative to
  // the collider, so query forces transformed with the wrong frame point into the plane.
  std::array<std::pair<char const*, Quaternion>, 2> const orientations = {
      std::pair{"Identity", Quaternion::Identity()},
      std::pair{"HalfTurnX", Quaternion(1_r, 0_r, 0_r, 0_r)}};
  for (auto const& [orientationName, rotation] : orientations) {
    SCOPED_TRACE(orientationName);
    ShellActorParams shellParams;
    shellParams.shape = CreateShape();
    shellParams.material.density = 1_r;
    shellParams.useContactSkin = true;
    shellParams.contactElementType = ActorBoundaryElementType::P1Q1;
    shellParams.worldFromLocal = TransformRT{rotation, Real3{0_r, 0_r, -0.01_r}};
    shellParams.contact.penaltyCoefficient = 1e6_r;
    Actor* const shellActor = experimental::CreateShellActor(_scene, shellParams, test::ExpectOK{});
    MOCHI_DEFER(_scene->DestroyActor(shellActor));
    shellActor->RegisterQuery(QueryType::ContactPoints, test::ExpectOK{});
    shellActor->RegisterQuery(QueryType::NodeContactForces, test::ExpectOK{});

    RigidActorParams planeParams;
    planeParams.shape =
        _mochiContext->CreatePlaneShape(Real3{0_r, 0_r, 1_r}, 0_r, test::ExpectOK{});
    planeParams.isStatic = true;
    planeParams.colliderType = ColliderType::Plane;
    planeParams.contact.penaltyCoefficient = shellParams.contact.penaltyCoefficient;
    Actor* const planeActor = _scene->CreateRigidActor(planeParams, test::ExpectOK{});
    MOCHI_DEFER(_scene->DestroyActor(planeActor));

    _scene->SetGravity({});
    _scene->Step(1e-4_r);

    auto& reg = GetRegistry();
    entt::entity const entity = GetEntity(shellActor);
    auto const& contactSnle = reg.get<CSkinnedContactSnle const>(entity);
    ASSERT_TRUE(contactSnle.useInSolver);
    ASSERT_EQ(1, isize(contactSnle.residuals));

    // Residuals are in the shell's local frame.
    std::array<Real3, 5> nodeForces{};
    int const shellDofOffset = reg.get<CDofOffset const>(entity).dofsOffset;
    for (auto const& [residualOffset, residual] : contactSnle.residuals) {
      for (int i = 0; i < residual.Rows(); ++i) {
        int const localDof = residualOffset + i - shellDofOffset;
        if (localDof >= 0 && localDof < kSpaceDim3 * isize(nodeForces)) {
          nodeForces[localDof / kSpaceDim3][localDof % kSpaceDim3] -= residual[i];
        }
      }
    }

    real totalNormalForce = 0_r;
    for (Real3& force : nodeForces) {
      force = rotation * force;
      totalNormalForce += force[2];
    }
    ASSERT_GT(totalNormalForce, 0_r);
    // The rotated actor frame adds round-off relative to the force magnitude.
    real const tangentialTolerance = 1e-6_r * totalNormalForce;
    for (Real3 const& force : nodeForces) {
      EXPECT_NEAR(0_r, force[0], tangentialTolerance);
      EXPECT_NEAR(0_r, force[1], tangentialTolerance);
    }

    std::array<real, 5> const expectedFractions = {
        1_r / 6_r, 5_r / 12_r, 1_r / 4_r, 1_r / 6_r, 0_r};
    for (int node = 0; node < isize(nodeForces); ++node) {
      EXPECT_NEAR(expectedFractions[node], nodeForces[node][2] / totalNormalForce, 1e-4_r)
          << "Physics node " << node;
    }

    auto const contactPoints = shellActor->GetContactPointsWorld(test::ExpectOK{});
    ASSERT_FALSE(contactPoints.empty());
    Real3 totalContactForce{};
    for (ContactPoint const& contactPoint : contactPoints) {
      EXPECT_EQ(0, contactPoint.elementIndex);
      EXPECT_NEAR_EQ((Real3{1_r / 3_r, 1_r / 3_r, 1_r / 3_r}), contactPoint.parametricCoords);
      totalContactForce += contactPoint.force;
    }
    real const forceTolerance = 1e-5_r * totalNormalForce;
    EXPECT_NEAR_TOL((Real3{0_r, 0_r, totalNormalForce}), totalContactForce, forceTolerance);

    auto const contactForces = shellActor->GetNodeContactForcesWorld(test::ExpectOK{});
    ASSERT_EQ(3, isize(contactForces));
    std::array<bool, 3> foundNodes{};
    Real3 totalNodeContactForce{};
    for (NodeContactForce const& contactForce : contactForces) {
      ASSERT_GE(contactForce.index, 0);
      ASSERT_LT(contactForce.index, isize(foundNodes));
      foundNodes[contactForce.index] = true;
      EXPECT_NEAR_TOL(totalContactForce / 3_r, contactForce.force, forceTolerance);
      totalNodeContactForce += contactForce.force;
    }
    EXPECT_EQ((std::array{true, true, true}), foundNodes);
    EXPECT_NEAR_TOL(totalContactForce, totalNodeContactForce, forceTolerance);
  }
}

TEST_F(MochiShellContactSkinTest, DirectContactQueriesUsePhysicsMeshIndices) {
  ShellActorParams shellParams;
  shellParams.shape = CreateShape();
  shellParams.material.density = 1_r;
  shellParams.contactElementType = ActorBoundaryElementType::P1Q1;
  shellParams.worldFromLocal.SetTranslation(Real3{0_r, 0_r, -0.01_r});
  shellParams.contact.penaltyCoefficient = 1e6_r;
  Actor* const shellActor = experimental::CreateShellActor(_scene, shellParams, test::ExpectOK{});
  shellActor->RegisterQuery(QueryType::ContactPoints, test::ExpectOK{});
  shellActor->RegisterQuery(QueryType::NodeContactForces, test::ExpectOK{});

  RigidActorParams planeParams;
  planeParams.shape = _mochiContext->CreatePlaneShape(Real3{0_r, 0_r, 1_r}, 0_r, test::ExpectOK{});
  planeParams.isStatic = true;
  planeParams.colliderType = ColliderType::Plane;
  planeParams.contact.penaltyCoefficient = shellParams.contact.penaltyCoefficient;
  _scene->CreateRigidActor(planeParams, test::ExpectOK{});

  _scene->SetGravity({});
  _scene->Step(1e-4_r);

  auto const contactPoints = shellActor->GetContactPointsWorld(test::ExpectOK{});
  ASSERT_EQ(isize(kPhysicsTriangles), isize(contactPoints));
  std::array<bool, 4> foundElements{};
  for (ContactPoint const& contactPoint : contactPoints) {
    ASSERT_GE(contactPoint.elementIndex, 0);
    ASSERT_LT(contactPoint.elementIndex, isize(foundElements));
    foundElements[contactPoint.elementIndex] = true;
  }
  EXPECT_EQ((std::array{true, true, true, true}), foundElements);

  auto const contactForces = shellActor->GetNodeContactForcesWorld(test::ExpectOK{});
  ASSERT_EQ(isize(kPhysicsNodes), isize(contactForces));
  std::array<bool, 5> foundNodes{};
  for (NodeContactForce const& contactForce : contactForces) {
    ASSERT_GE(contactForce.index, 0);
    ASSERT_LT(contactForce.index, isize(foundNodes));
    foundNodes[contactForce.index] = true;
  }
  EXPECT_EQ((std::array{true, true, true, true, true}), foundNodes);
}

TEST_F(MochiShellContactSkinTest, ContactSkinQuadratureDoesNotChangePointCloudCollider) {
  Actor* const actor = CreateActor(
      CreateShape(),
      /*useContactSkin=*/true,
      ActorBoundaryElementType::P1Q6,
      ColliderType::PointCloud);
  auto& reg = GetRegistry();
  entt::entity const entity = GetEntity(actor);

  EXPECT_EQ(6, reg.get<CFemSurfaceDiscretization const>(entity).GetNumQuadPoints());
  EXPECT_EQ(6, isize(reg.get<CContactSamples<TimeStep::Current> const>(entity).positions));
  EXPECT_EQ(
      isize(kPhysicsNodes),
      reg.get<CColliderPointCloudDiscretization const>(entity).GetNumColliderPoints());

  ecs::InvokeOnEntity(
      linear_contact_skin::UpdatePositionsAndSamples<TimeStep::Current>, reg, entity);
  for (Real3 const& sample : reg.get<CContactSamples<TimeStep::Current> const>(entity).positions) {
    EXPECT_GE(sample[0], 0.5_r);
  }
}

TEST_F(MochiShellContactSkinTest, ContactSkinBoundsUseSkinAndColliderGeometryForBothStates) {
  ShapeHandle const shape =
      CreateShape(/*includeContactSkin=*/true, /*extrapolateContactSkin=*/true);
  Actor* const skinOnlyActor = CreateActor(shape, /*useContactSkin=*/true);
  Actor* const pointCloudActor = CreateActor(
      shape,
      /*useContactSkin=*/true,
      ActorBoundaryElementType::Default,
      ColliderType::PointCloud);
  auto& reg = GetRegistry();
  entt::entity const skinOnlyEntity = GetEntity(skinOnlyActor);
  entt::entity const pointCloudEntity = GetEntity(pointCloudActor);

  auto const expectBounds = [&reg](entt::entity entity, Aabb const& expected) {
    Aabb const actual = GetAabb(reg.get<CBoundingVolume const>(entity).localShape);
    EXPECT_NEAR_EQ(expected.GetMin(), actual.GetMin());
    EXPECT_NEAR_EQ(expected.GetMax(), actual.GetMax());
  };

  Aabb const physicsBounds = CalcAabb(MakeConstSpan(kPhysicsNodes));
  real const radius = reg.get<CPointCloudColliderParams const>(pointCloudEntity).radius;
  Aabb const expandedPhysicsBounds = GetAabb(ExpandShape(GetObb(physicsBounds), radius));
  Aabb const skinBounds{Real3{0.5_r, 0_r, 0_r}, Real3{2_r, 1_r, 0_r}};
  auto const expectTranslatedBounds = [&](Real3 const& translation) {
    Aabb const translatedSkinBounds{
        skinBounds.GetMin() + translation, skinBounds.GetMax() + translation};
    Aabb const translatedPhysicsBounds{
        expandedPhysicsBounds.GetMin() + translation, expandedPhysicsBounds.GetMax() + translation};
    Aabb const expectedPointCloudBounds{
        Min(translatedSkinBounds.VGetMin(), translatedPhysicsBounds.VGetMin()),
        Max(translatedSkinBounds.VGetMax(), translatedPhysicsBounds.VGetMax())};
    expectBounds(skinOnlyEntity, translatedSkinBounds);
    expectBounds(pointCloudEntity, expectedPointCloudBounds);
  };

  expectTranslatedBounds({});

  Real3 const currentTranslation{1_r, 2_r, 3_r};
  Real3 const stageStartTranslation{-2_r, -3_r, -4_r};
  for (entt::entity const entity : {skinOnlyEntity, pointCloudEntity}) {
    auto& current = reg.get<CDisplacementSlice<real, TimeStep::Current>>(entity).value;
    auto& stageStart = reg.get<CDisplacementSlice<real, TimeStep::StageStart>>(entity).value;
    for (int node = 0; node < isize(kPhysicsNodes); ++node) {
      for (int component = 0; component < kSpaceDim3; ++component) {
        current[kSpaceDim3 * node + component] = currentTranslation[component];
        stageStart[kSpaceDim3 * node + component] = stageStartTranslation[component];
      }
    }
    ecs::InvokeOnEntity(linear_contact_skin::UpdateBounds<TimeStep::Current>, reg, entity);
  }
  expectTranslatedBounds(currentTranslation);

  for (entt::entity const entity : {skinOnlyEntity, pointCloudEntity}) {
    ecs::InvokeOnEntity(linear_contact_skin::UpdateBounds<TimeStep::StageStart>, reg, entity);
  }
  expectTranslatedBounds(stageStartTranslation);
}

TEST_F(MochiShellContactSkinTest, MaxGeometrySpeedUsesContactSkinThroughEcsDispatch) {
  Actor* const actor = CreateActor(
      CreateShape(/*includeContactSkin=*/true, /*extrapolateContactSkin=*/true),
      /*useContactSkin=*/true);
  auto& reg = GetRegistry();
  entt::entity const entity = GetEntity(actor);
  DynamicArray<real> velocity(actor->GetNumDofs(), 0_r);
  velocity[0] = 1_r;
  velocity[3] = -1_r;
  actor->SetNodeVelocitiesLocal(MakeConstSpan(velocity), test::ExpectOK{});

  UpdateMaxGeometrySpeeds(reg);

  EXPECT_NEAR_EQ(3_r, reg.get<CConservativeStepBounds const>(entity).maxGeometrySpeed);
}

TEST_F(MochiShellContactSkinTest, PhysicsNodeConsumersIgnoreAuthoredSkinOrdering) {
  ShapeHandle const shape = CreateShape();
  Actor* const actorA = CreateActor(shape);
  Actor* const actorB = CreateActor(shape);

  DynamicArray<int> selectedNodes;
  Real3 const physicsNode = kPhysicsNodes[4];
  Real3 constexpr kHalfExtent{0.01_r, 0.01_r, 0.01_r};
  actorA->RegisterQueryAndCompute(QueryType::SurfaceNodePositions, test::ExpectOK{});
  actorA->QueryNodesInVolumeLocal(
      Aabb{physicsNode - kHalfExtent, physicsNode + kHalfExtent},
      /*boundaryOnly=*/true,
      [&](int nodeIndex, Real3) { selectedNodes.push_back(nodeIndex); },
      test::ExpectOK{});
  std::array<int, 1> const expectedNodes = {4};
  EXPECT_SPAN_EQ(MakeConstSpan(expectedNodes), MakeConstSpan(selectedNodes));

  DeformableNodeToDeformableNodeConstraintParams constraintParams;
  constraintParams.actorA = actorA->GetHandle();
  constraintParams.actorB = actorB->GetHandle();
  constraintParams.nodeIndexA = 4;
  constraintParams.nodeIndexB = 4;
  EXPECT_NE(
      nullptr,
      _scene->CreateDeformableNodeToDeformableNodeConstraint(constraintParams, test::ExpectOK{}));
}

TEST_F(MochiShellContactSkinTest, ShellWithoutContactSkinKeepsDefaultRepresentation) {
  Actor* const actor = CreateActor(CreateShape(/*includeContactSkin=*/false));
  auto& reg = GetRegistry();
  entt::entity const entity = GetEntity(actor);

  EXPECT_EQ(isize(kPhysicsNodes), actor->GetMesh().GetNumNodes());
  EXPECT_EQ(isize(kPhysicsNodes), actor->GetSurfaceMesh().GetNumNodes());
  EXPECT_EQ(MeshDataView{}, actor->GetContactSkinMesh());
  EXPECT_FALSE(actor->IsQuerySupported(QueryType::ContactSkinNodePositions));
  EXPECT_FALSE(actor->IsQuerySupported(QueryType::ContactSkinNodeNormals));
  EXPECT_FALSE((reg.any_of<
                CContactSkinMesh,
                TagUseDeformableContactSkin,
                CContactSkinningData,
                CDeformedContactSkinNodes>(entity)));
  EXPECT_TRUE((reg.all_of<CContactLocal2GlobalMap, CContactNodalBasedStructure>(entity)));
}
