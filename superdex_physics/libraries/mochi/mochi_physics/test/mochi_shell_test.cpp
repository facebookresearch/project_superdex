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
#include <mochi_physics/src/mochi_shell.h>
#include "mochi_core/test/mochi_test_helpers.h"
#include "mochi_physics_test_fixture.h"

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

TEST_F(MochiShellCreateActorValidationScene, CreateShellActor_RejectsInvalidMaterial) {
  auto&& [coords, conn] = test::CreateMinimalTriMeshUnitCube();
  ShellActorParams params;
  params.shape = _scene->GetContext()->CreateTriMeshShape(
      Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), ErrorAssert{});
  params.material.membraneMu = -1_r; // Invalid

  Error error;
  Actor* actor = CreateShellActor(_scene, params, error);
  EXPECT_FALSE(error.IsOK());
  EXPECT_EQ(actor, nullptr);
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
