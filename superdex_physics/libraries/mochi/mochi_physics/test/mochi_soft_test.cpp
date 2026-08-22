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

#include <mochi_core/materials/material_params_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/src/mochi_deformable.h>
#include <mochi_physics/src/mochi_soft.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

using namespace mochi;

static SoftMaterialParams MakeNeoHookeanMaterial(real youngsModulus) {
  SoftMaterialParams params;
  params.type = SoftMaterialType::NeoHookean;
  params.neoHookean.youngsModulus = youngsModulus;
  params.neoHookean.psdStrategy = MaterialPsdStrategy::Projection;
  return params;
}

static void ExpectNeoHookeanMaterialEq(
    SoftMaterialParams const& expected,
    SoftMaterialParams const& actual) {
  EXPECT_EQ(expected.type, actual.type);
  EXPECT_EQ(expected.density, actual.density);
  EXPECT_NEAR_RTOL(expected.neoHookean.youngsModulus, actual.neoHookean.youngsModulus, 1e-4_r);
  EXPECT_NEAR_RTOL(expected.neoHookean.poissonRatio, actual.neoHookean.poissonRatio, 1e-4_r);
  EXPECT_EQ(expected.neoHookean.psdStrategy, actual.neoHookean.psdStrategy);
}

class MochiSoftActorScene : public test::MochiSceneTestBase {
 protected:
  static constexpr int kNumDofsSoft = 24; // a soft with 8 nodes
  static constexpr real kDt = 0.01_r;

  Actor* _actor = nullptr;
  ColumnVector<real> _pos;
  ColumnVector<real> _predPos;
  ColumnVector<real> _currVel;
  TransformRT _transform;
  std::vector<Real3> _coords;
  std::vector<Int4> _connect;

  ContactDetectionParams _collParams;
  ContactDetectionResult _collResult;

 public:
  void SetUp() override { // Called just before each test case

    test::MochiSceneTestBase::SetUp();

    // Initialize soft actor state
    _pos.Reset(kNumDofsSoft);
    _predPos.Reset(kNumDofsSoft);
    _currVel.Reset(kNumDofsSoft);
    _transform = {};

    // Create soft actor
    // This is a cube of size 1, centered at 0.5
    auto& reg = GetRegistry();
    auto&& [unitCubeCoordinates, unitCubeConnectivity] = test::CreateMinimalTetMeshUnitCube();
    _coords = unitCubeCoordinates;
    _connect = unitCubeConnectivity;
    SoftActorParams params;
    params.shape = _scene->GetContext()->CreateTetMeshShape(
        Flatten(MakeSpan(unitCubeCoordinates)),
        Flatten(MakeSpan(unitCubeConnectivity)),
        ErrorAssert{});
    params.worldFromLocal = _transform;
    _actor = _scene->CreateSoftActor(params, ErrorAssert{});
    auto entity = mochi::GetEntity(reg, _actor->GetHandle(), test::ExpectOK{});
    EXPECT_EQ(kNumDofsSoft, reg.get<CActorDofInfo>(entity).poseSize);
  }
};

TEST_F(MochiSoftActorScene, GetSetDisplacements) {
  ColumnVector<real> displ(kNumDofsSoft);
  displ.SetRandom(123);
  _actor->SetDisplacements(displ, ErrorAssert{});
  EXPECT_SPAN_EQ(_actor->GetDisplacements(ErrorAssert{}), MakeConstSpan(displ));
}

TEST_F(MochiSoftActorScene, GetSetSoftMaterialParamsField) {
  {
    auto const base = MakeNeoHookeanMaterial(1000_r);
    _actor->SetSoftMaterialParams(base, test::ExpectOK{});

    auto updated = MakeNeoHookeanMaterial(2000_r);
    updated.density = -1_r;
    experimental::SetSoftMaterialParamsField(_actor, updated, 1, test::ExpectOK{});

    // Per-element setters update only material model params. Density remains actor-wide.
    EXPECT_EQ(base.density, _actor->GetDensity(test::ExpectOK{}));
    updated.density = base.density;

    ExpectNeoHookeanMaterialEq(
        base, experimental::GetSoftMaterialParamsField(_actor, 0, test::ExpectOK{}));
    ExpectNeoHookeanMaterialEq(
        updated, experimental::GetSoftMaterialParamsField(_actor, 1, test::ExpectOK{}));
    ExpectNeoHookeanMaterialEq(
        base,
        experimental::GetSoftMaterialParamsField(_actor, isize(_connect) - 1, test::ExpectOK{}));
  }

  // Validation errors leave material unchanged.
  {
    auto const base = MakeNeoHookeanMaterial(1000_r);
    _actor->SetSoftMaterialParams(base, test::ExpectOK{});

    auto const expectUnchanged = [&] {
      ExpectNeoHookeanMaterialEq(
          base, experimental::GetSoftMaterialParamsField(_actor, 0, test::ExpectOK{}));
    };

    auto const updated = MakeNeoHookeanMaterial(2000_r);
    experimental::SetSoftMaterialParamsField(_actor, updated, -1, test::ExpectNotOK{});
    expectUnchanged();
    experimental::SetSoftMaterialParamsField(_actor, updated, isize(_connect), test::ExpectNotOK{});
    expectUnchanged();

    auto invalid = updated;
    invalid.neoHookean.youngsModulus = -1_r;
    experimental::SetSoftMaterialParamsField(_actor, invalid, 0, test::ExpectNotOK{});
    expectUnchanged();

    auto otherType = updated;
    otherType.type = SoftMaterialType::Arap;
    experimental::SetSoftMaterialParamsField(_actor, otherType, 0, test::ExpectNotOK{});
    expectUnchanged();

    auto incompatiblePsd = updated;
    incompatiblePsd.neoHookean.psdStrategy = MaterialPsdStrategy::Fast;
    experimental::SetSoftMaterialParamsField(_actor, incompatiblePsd, 0, test::ExpectNotOK{});
    expectUnchanged();
  }

  // GetSoftMaterialParamsField reports an error if the index is invalid.
  {
    (void)experimental::GetSoftMaterialParamsField(_actor, -1, test::ExpectNotOK{});
    (void)experimental::GetSoftMaterialParamsField(_actor, isize(_connect), test::ExpectNotOK{});
  }

  // SetSoftMaterialParamsField and GetSoftMaterialParamsField reject non-soft actors.
  {
    RigidActorParams params;
    params.shape = _mochiContext->CreatePlaneShape(Real3{0_r, 1_r, 0_r}, 0_r, test::ExpectOK{});
    params.isStatic = true;
    auto* rigidActor = _scene->CreateRigidActor(params, test::ExpectOK{});

    auto const material = MakeNeoHookeanMaterial(1000_r);
    experimental::SetSoftMaterialParamsField(rigidActor, material, 0, test::ExpectNotOK{});
    (void)experimental::GetSoftMaterialParamsField(rigidActor, 0, test::ExpectNotOK{});
  }
}

// ---------------------------------------------------------------------------
// Soft damping: validation and API round-trip
// ---------------------------------------------------------------------------

TEST(SoftMaterialValidation, ValidateDampingCoefficients) {
  auto base = [] {
    SoftMaterialParams p;
    p.type = SoftMaterialType::NeoHookean;
    p.neoHookean.youngsModulus = 1000_r;
    return p;
  };
  {
    auto p = base();
    p.massDampingCoefficient = 5_r;
    p.stiffnessDampingCoefficient = 0.01_r;
    ValidateSoftMaterialParams(p, test::ExpectOK{});
  }
  auto reject = [&](auto mutate) {
    auto p = base();
    mutate(p);
    ValidateSoftMaterialParams(p, test::ExpectNotOK{});
  };
  reject([](SoftMaterialParams& p) { p.massDampingCoefficient = -1_r; });
  reject([](SoftMaterialParams& p) { p.stiffnessDampingCoefficient = -1_r; });
  reject([](SoftMaterialParams& p) {
    p.massDampingCoefficient = std::numeric_limits<real>::quiet_NaN();
  });
  reject([](SoftMaterialParams& p) {
    p.stiffnessDampingCoefficient = std::numeric_limits<real>::quiet_NaN();
  });
}

TEST_F(MochiSoftActorScene, DampingCoefficientsRoundTrip) {
  auto params = MakeNeoHookeanMaterial(1000_r);
  params.massDampingCoefficient = 2.5_r;
  params.stiffnessDampingCoefficient = 0.02_r;
  _actor->SetSoftMaterialParams(params, test::ExpectOK{});

  auto const out = _actor->GetSoftMaterialParams(test::ExpectOK{});
  EXPECT_NEAR_RTOL(params.massDampingCoefficient, out.massDampingCoefficient, 1e-6_r);
  EXPECT_NEAR_RTOL(params.stiffnessDampingCoefficient, out.stiffnessDampingCoefficient, 1e-6_r);
}

// ---------------------------------------------------------------------------
// Soft mass damping: backward Euler velocity decay
// ---------------------------------------------------------------------------

// A free soft cube with no stress, no gravity, uniform initial velocity, and backward Euler. With
// mass damping α, the velocity decays as v_n = v_0 / (1 + α·dt)^n.
class MochiSoftMassDampingScene : public test::MochiSceneTestBase {
 protected:
  Actor* _actor = nullptr;
  int _numDofs = 0;
  static constexpr int kDofsPerNode = 3;
  static constexpr real kV0 = 1_r;
  static constexpr real kDt = 0.01_r;

  Actor* CreateDampedActor(real massDampingCoefficient) {
    auto&& [coords, conn] = test::CreateMinimalTetMeshUnitCube();
    SoftActorParams params;
    params.material.type = SoftMaterialType::NeoHookean;
    params.material.neoHookean.youngsModulus = 1000_r;
    params.material.density = 1_r;
    params.material.massDampingCoefficient = massDampingCoefficient;
    params.hasGravity = false;
    params.hasStress = false; // Isolate the mass term.
    params.shape = _scene->GetContext()->CreateTetMeshShape(
        Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), ErrorAssert{});
    Actor* actor = _scene->CreateSoftActor(params, ErrorAssert{});
    // Disable recentering so rigid motion stays in the local displacements we measure.
    actor->SetRecenteringParams(RecenteringParams{.useRecentering = false}, ErrorAssert{});
    return actor;
  }

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();
    auto solverParams = _scene->GetSolverParams();
    solverParams.integrationMethod = IntegrationMethod::BackwardEuler;
    _scene->SetSolverParams(solverParams, ErrorAssert{});
  }

  void InitWithDamping(real massDampingCoefficient) {
    _actor = CreateDampedActor(massDampingCoefficient);
    _numDofs = _actor->GetNumDofs();
    DynamicArray<real> vel(_numDofs, 0_r);
    for (int i = 0; i < _numDofs; i += kDofsPerNode) {
      vel[i] = kV0;
    }
    _actor->SetNodeVelocitiesLocal(MakeSpan(vel), ErrorAssert{});
  }
};

TEST_F(MochiSoftMassDampingScene, VelocityDecay) {
  real constexpr kAlpha = 3_r;
  int constexpr kNumSteps = 10;
  real constexpr kVelRtol = 1e-4_r;
  InitWithDamping(kAlpha);

  DynamicArray<real> prevDispl(_numDofs, 0_r);
  DynamicArray<real> currDispl(_numDofs);
  for (int step = 0; step < kNumSteps; ++step) {
    _scene->Step(kDt);
    _actor->GetDofValues({}, MakeSpan(currDispl), ErrorAssert{});
    real const expectedVel = kV0 / Pow(1_r + kAlpha * kDt, step + 1);
    for (int i = 0; i < _numDofs; i += kDofsPerNode) {
      real const vel = (currDispl[i] - prevDispl[i]) / kDt;
      EXPECT_NEAR_RTOL(expectedVel, vel, kVelRtol);
    }
    std::copy(currDispl.begin(), currDispl.end(), prevDispl.begin());
  }
}

TEST_F(MochiSoftMassDampingScene, UndampedVelocityConserved) {
  int constexpr kNumSteps = 5;
  real constexpr kVelRtol = 1e-4_r;
  InitWithDamping(0_r);

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
// Soft stiffness damping
// ---------------------------------------------------------------------------

// Stiffness-proportional (Rayleigh) damping for soft (FEM tetrahedral) actors, exercised over every
// soft material model on a single-tetrahedron actor. The magnitude test pins all but one node and
// checks the damped/undamped displacement ratio in the stiff limit against the analytic value
// dt/(dt+beta); the rigid-translation test checks that zero strain rate produces no damping.
class MochiSoftStiffnessDampingScene : public test::MochiSceneTestBase,
                                       public ::testing::WithParamInterface<SoftMaterialType> {
 protected:
  static constexpr real kStiffness = 1e4_r;

  Actor* CreateSingleTetActor(SoftMaterialType type, real stiffnessDampingCoefficient) {
    auto&& [coords, conn] = test::CreateMinimalTetMeshSingleTet();
    SoftActorParams params;
    params.material.type = type;
    // Only the sub-struct matching `type` is read, so set them all to the same value.
    params.material.arap.stiffness = kStiffness;
    params.material.activeShapeTargetingArap.stiffness = kStiffness;
    params.material.activeNeoHookean.passiveIsotropic.youngsModulus = kStiffness;
    params.material.neoHookean.youngsModulus = kStiffness;
    params.material.stVenantKirchhoff.youngsModulus = kStiffness;
    params.material.linearElastic.youngsModulus = kStiffness;
    params.material.density = 1_r;
    params.material.stiffnessDampingCoefficient = stiffnessDampingCoefficient;
    params.hasGravity = false;
    params.shape = _scene->GetContext()->CreateTetMeshShape(
        Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), ErrorAssert{});
    Actor* actor = _scene->CreateSoftActor(params, ErrorAssert{});
    // Disable recentering so the deformation we drive stays in the local displacements we measure.
    actor->SetRecenteringParams(RecenteringParams{.useRecentering = false}, ErrorAssert{});
    return actor;
  }

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();
    auto solverParams = _scene->GetSolverParams();
    solverParams.integrationMethod = IntegrationMethod::BackwardEuler;
    _scene->SetSolverParams(solverParams, ErrorAssert{});
  }
};

// Verify the magnitude of stiffness-proportional damping via a single backward Euler step on a
// single tetrahedron with three nodes pinned and one free. The free node is driven along its own
// axis (pure uniaxial stretch); in the stiff limit the ratio of damped to undamped displacement of
// that node is dt/(dt+beta), independent of stiffness and mass. This holds for every soft material
// model. A transverse (shear) velocity is deliberately avoided: for the corotational ARAP materials
// it couples into the near-rigid rotation mode and the simple ratio no longer applies, whereas the
// strain-based materials are direction-insensitive.
TEST_P(MochiSoftStiffnessDampingScene, StiffnessProportionalDecayRatio) {
  static constexpr int kDofsPerNode = 3;
  static constexpr int kFreeNode = 3;
  // Node 3 lies on the +z axis of the reference tet, so driving it along +z is a pure
  // uniaxial stretch (no shear, no rotation).
  static constexpr int kStretchAxis = 2;
  static constexpr real kBeta = 1_r;
  static constexpr real kDt = 1_r;
  static constexpr real kV0 = 1_r;
  static constexpr real kRatioRtol = 1e-2_r;
  static constexpr real kStiffLimitBound = 1e-2_r;
  static constexpr real kParallelRtol = 1e-3_r;

  Actor* actor = CreateSingleTetActor(GetParam(), 0_r);
  // Pin nodes 0, 1, 2; node 3 is free. Constraints persist across the reset, so pin only once.
  experimental::ConstrainNodesByPosition(
      actor, [](int i, Real3 const&) { return i <= 2; }, ErrorAssert{});

  // One backward Euler step from rest with the free node given velocity kV0 along its stretch axis;
  // returns the free node's displacement.
  auto singleStep = [&]() -> Real3 {
    actor->SetZeroDisplacementsAndVelocities(ErrorAssert{});
    int const numDofs = actor->GetNumDofs();
    DynamicArray<real> vel(numDofs, 0_r);
    vel[kFreeNode * kDofsPerNode + kStretchAxis] = kV0;
    actor->SetNodeVelocitiesLocal(MakeSpan(vel), ErrorAssert{});
    _scene->Step(kDt);
    DynamicArray<real> displ(numDofs);
    actor->GetDofValues({}, MakeSpan(displ), ErrorAssert{});
    return Real3{
        displ[kFreeNode * kDofsPerNode],
        displ[kFreeNode * kDofsPerNode + 1],
        displ[kFreeNode * kDofsPerNode + 2]};
  };

  Real3 const undamped = singleStep();
  auto params = actor->GetSoftMaterialParams(test::ExpectOK{});
  params.stiffnessDampingCoefficient = kBeta;
  actor->SetSoftMaterialParams(params, test::ExpectOK{});
  Real3 const damped = singleStep();

  // Guard: both displacements must be non-trivial, ruling out unconverged or zero solves.
  EXPECT_GT(Norm(undamped), 0_r);
  EXPECT_GT(Norm(damped), 0_r);

  // Stiff-limit check: displacement << v0 * dt confirms dt^2 K / m >> 1.
  EXPECT_LT(Norm(undamped), kStiffLimitBound * kV0 * kDt);

  // Parallelism: both solutions differ only by a scalar factor in the stiff limit.
  EXPECT_NEAR_RTOL(1_r, Dot(undamped, damped) / (Norm(undamped) * Norm(damped)), kParallelRtol);

  // Ratio check: ||damped|| / ||undamped|| ~ dt / (dt + beta).
  EXPECT_NEAR_RTOL(kDt / (kDt + kBeta), Norm(damped) / Norm(undamped), kRatioRtol);
}

// A rigidly translating body has zero strain and zero strain rate, so neither the elastic stress
// nor the stiffness damping exerts any force: it keeps translating at constant velocity, regardless
// of the damping coefficient or material model.
TEST_P(MochiSoftStiffnessDampingScene, RigidTranslationUndamped) {
  real constexpr kBeta = 0.05_r;
  real constexpr kV0 = 0.5_r;
  real constexpr kDt = 0.01_r;
  int constexpr kNumSteps = 5;
  real constexpr kRtol = 1e-4_r;
  real constexpr kAbsTol = 1e-6_r;

  Actor* actor = CreateSingleTetActor(GetParam(), kBeta);
  int const numDofs = actor->GetNumDofs();
  DynamicArray<real> vel(numDofs, 0_r);
  for (int i = 0; i < numDofs; i += 3) {
    vel[i] = kV0; // uniform x-velocity
  }
  actor->SetNodeVelocitiesLocal(MakeSpan(vel), ErrorAssert{});

  for (int s = 0; s < kNumSteps; ++s) {
    _scene->Step(kDt);
  }
  DynamicArray<real> displ(numDofs);
  actor->GetDofValues({}, MakeSpan(displ), ErrorAssert{});

  real const expectedX = kV0 * kNumSteps * kDt;
  for (int i = 0; i < numDofs; ++i) {
    if (i % 3 == 0) {
      EXPECT_NEAR_RTOL(expectedX, displ[i], kRtol);
    } else {
      EXPECT_NEAR_TOL(0_r, displ[i], kAbsTol);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllMaterials,
    MochiSoftStiffnessDampingScene,
    ::testing::Values(
        SoftMaterialType::NeoHookean,
        SoftMaterialType::StVenantKirchhoff,
        SoftMaterialType::LinearElastic,
        SoftMaterialType::Arap,
        SoftMaterialType::ActiveNeoHookean,
        SoftMaterialType::ActiveShapeTargetingArap),
    [](testing::TestParamInfo<SoftMaterialType> const& info) -> std::string {
      switch (info.param) {
        case SoftMaterialType::NeoHookean:
          return "NeoHookean";
        case SoftMaterialType::StVenantKirchhoff:
          return "StVenantKirchhoff";
        case SoftMaterialType::LinearElastic:
          return "LinearElastic";
        case SoftMaterialType::ActiveNeoHookean:
          return "ActiveNeoHookean";
        case SoftMaterialType::ActiveShapeTargetingArap:
          return "ActiveShapeTargetingArap";
        case SoftMaterialType::Arap:
          return "Arap";
        case SoftMaterialType::Count:
          return "Count";
      }
      return "Unknown";
    });
