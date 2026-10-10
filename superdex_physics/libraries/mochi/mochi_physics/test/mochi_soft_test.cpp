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
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/src/mochi_contact.h>
#include <mochi_physics/src/mochi_deformable.h>
#include <mochi_physics/src/mochi_discretization_components.h>
#include <mochi_physics/src/mochi_linear_contact_skin.h>
#include <mochi_physics/src/mochi_snle.h>
#include <mochi_physics/src/mochi_soft.h>
#include <mochi_physics/src/mochi_step.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <utility>
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

TEST_F(MochiSoftActorScene, RejectsUnreferencedNodes) {
  _coords.emplace_back(2_r, 2_r, 2_r); // Not referenced by any tetrahedron.
  SoftActorParams params;
  params.shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(_coords)), Flatten(MakeSpan(_connect)), test::ExpectOK{});
  EXPECT_EQ(nullptr, _scene->CreateSoftActor(params, test::ExpectNotOK{}));
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

class MochiSoftContactSkin : public test::MochiSceneTestBase {
 protected:
  enum class Embedding { Identity, Rotated, Shifted };

  static ModelData MakeModel(Embedding embedding = Embedding::Identity) {
    ModelData model = test::CreateUnitCubeContactSkinModel();
    auto& skinning = *model.contactSkinMesh->skinning;

    if (embedding == Embedding::Rotated) {
      // Quarter turn of the unit cube about z, (x, y, z) -> (1 - y, x, z), which preserves
      // orientation.
      std::array<int, 8> constexpr kRotatedNodes = {1, 3, 0, 2, 5, 7, 4, 6};
      for (int& index : skinning.indices) {
        index = kRotatedNodes[index];
      }
    } else if (embedding == Embedding::Shifted) {
      // Nodes 0 and 1 are (0, 0, 0) and (1, 0, 0), so adding them with weights 0.5 and -0.5 shifts
      // the skin by -0.5 along x.
      SkinningData shifted;
      shifted.weightsPerNode = 3;
      for (int i = 0; i < isize(skinning.indices); ++i) {
        shifted.indices.push_back(skinning.indices[i]);
        shifted.indices.push_back(0);
        shifted.indices.push_back(1);
        shifted.weights.push_back(skinning.weights[i]);
        shifted.weights.push_back(0.5_r);
        shifted.weights.push_back(-0.5_r);
      }
      skinning = std::move(shifted);
    }
    return model;
  }

  // Skin-node positions embedded from physics-node positions through the contact skin's skinning.
  static DynamicArray<Real3> EmbedSkinNodes(
      MeshDataView const& contactSkin,
      Span<Real3 const> physicsPositions) {
    SkinningDataView const& skinning = *contactSkin.skinning;
    DynamicArray<Real3> skinPositions(contactSkin.GetNumNodes(), Real3{});
    for (int node = 0; node < isize(skinPositions); ++node) {
      for (int k = 0; k < skinning.weightsPerNode; ++k) {
        int const entry = skinning.weightsPerNode * node + k;
        skinPositions[node] += skinning.weights[entry] * physicsPositions[skinning.indices[entry]];
      }
    }
    return skinPositions;
  }

  Actor* CreateActor(
      bool useContactSkin,
      ActorBoundaryElementType elementType = ActorBoundaryElementType::Default,
      Embedding embedding = Embedding::Identity) {
    SoftActorParams params;
    params.shape = _mochiContext->CreateModelShape(MakeModel(embedding), test::ExpectOK{});
    params.useContactSkin = useContactSkin;
    params.boundaryElementType = elementType;
    return _scene->CreateSoftActor(params, test::ExpectOK{});
  }
};

TEST_F(MochiSoftContactSkin, RejectsMissingContactSkin) {
  ModelData model = MakeModel();
  model.contactSkinMesh.reset();
  SoftActorParams params;
  params.shape = _mochiContext->CreateModelShape(model, test::ExpectOK{});
  params.useContactSkin = true;
  EXPECT_EQ(nullptr, _scene->CreateSoftActor(params, test::ExpectNotOK{}));
}

TEST_IF_F(MOCHI_ENABLE_ROM_ACTORS, MochiSoftContactSkin, RejectsRom) {
  SoftActorParams params;
  params.shape = _mochiContext->CreateModelShape(MakeModel(), test::ExpectOK{});
  params.useContactSkin = true;
  experimental::ExperimentalSoftActorParams experimentalParams;
  experimentalParams.rom = experimental::RomParams{.source = "unused"};
  Error error;
  EXPECT_EQ(nullptr, experimental::CreateSoftActor(_scene, params, experimentalParams, error));
  EXPECT_STREQ("useContactSkin is not supported for ROM soft actors.", error.GetDescription());
}

TEST_F(MochiSoftContactSkin, UseContactSkinSelectsContactRepresentation) {
  auto& reg = GetRegistry();
  entt::entity const direct = GetEntity(CreateActor(false));
  EXPECT_FALSE(reg.all_of<TagUseDeformableContactSkin>(direct));
  EXPECT_TRUE(reg.all_of<CFemBoundaryDiscretization>(direct));

  // A non-default element type checks that boundaryElementType reaches the contact skin.
  entt::entity const skin = GetEntity(CreateActor(true, ActorBoundaryElementType::P1Q6));
  EXPECT_TRUE(reg.all_of<TagUseDeformableContactSkin>(skin));
  EXPECT_FALSE(reg.all_of<CFemBoundaryDiscretization>(skin));
  EXPECT_EQ(
      6 * reg.get<CContactSkinMesh const>(skin).mesh->GetNumElements(),
      isize(reg.get<CContactSamples<TimeStep::Current> const>(skin).positions));
}

TEST_F(MochiSoftContactSkin, AuthoredSkinIsExposedIndependentlyOfContactSelection) {
  auto const tetCoordinates = test::CreateMinimalTetMeshUnitCube().first;
  // Gives each unit-cube corner a distinct displacement.
  auto const displacementAt = [](Real3 const& position) {
    return Real3{Dot(position, Real3{1_r, 2_r, 4_r}), 0_r, 0_r};
  };
  for (bool const useContactSkin : {false, true}) {
    SCOPED_TRACE(useContactSkin ? "useContactSkin" : "!useContactSkin");
    // The shifted embedding places the skin away from the tetrahedral boundary.
    Actor* const actor =
        CreateActor(useContactSkin, ActorBoundaryElementType::Default, Embedding::Shifted);

    MeshDataView const contactSkin = actor->GetContactSkinMesh();
    MeshDataView const shapeContactSkin = _mochiContext->GetShapeContactSkinMesh(
        actor->GetReferenceShape(test::ExpectOK{}), test::ExpectOK{});
    EXPECT_SPAN_EQ(shapeContactSkin.coordinates, contactSkin.coordinates);
    EXPECT_SPAN_EQ(shapeContactSkin.connectivity, contactSkin.connectivity);
    ASSERT_TRUE(contactSkin.skinning.has_value());
    ASSERT_TRUE(shapeContactSkin.skinning.has_value());
    EXPECT_EQ(shapeContactSkin.skinning->weightsPerNode, contactSkin.skinning->weightsPerNode);
    EXPECT_SPAN_EQ(shapeContactSkin.skinning->indices, contactSkin.skinning->indices);
    EXPECT_SPAN_EQ(shapeContactSkin.skinning->weights, contactSkin.skinning->weights);

    DynamicArray<real> displacements(actor->GetNumDofs(), 0_r);
    auto const displacementVectors = Unflatten<Real3>(MakeSpan(displacements));
    DynamicArray<Real3> deformedPhysicsNodes(isize(tetCoordinates), Real3{});
    for (int node = 0; node < isize(tetCoordinates); ++node) {
      displacementVectors[node] = displacementAt(tetCoordinates[node]);
      deformedPhysicsNodes[node] = tetCoordinates[node] + displacementVectors[node];
    }
    actor->SetDisplacements(displacements, test::ExpectOK{});
    actor->RegisterQueryAndCompute(QueryType::ContactSkinNodePositions, test::ExpectOK{});

    DynamicArray<Real3> const expectedSkinPositions =
        EmbedSkinNodes(contactSkin, MakeConstSpan(deformedPhysicsNodes));
    auto const skinPositions =
        Unflatten<Real3 const>(actor->GetContactSkinMeshNodePositionsLocal(test::ExpectOK{}));
    ASSERT_EQ(contactSkin.GetNumNodes(), isize(skinPositions));
    for (int node = 0; node < isize(skinPositions); ++node) {
      EXPECT_NEAR_EQ(expectedSkinPositions[node], skinPositions[node]);
    }
  }
}

TEST_F(MochiSoftContactSkin, BoundsIncludeColliderGeometry) {
  auto createActor = [this](ColliderType colliderType) {
    SoftActorParams params;
    params.shape = _mochiContext->CreateModelShape(MakeModel(Embedding::Shifted), test::ExpectOK{});
    params.useContactSkin = true;
    experimental::ExperimentalSoftActorParams experimentalParams;
    experimentalParams.colliderType = colliderType;
    return experimental::CreateSoftActor(_scene, params, experimentalParams, test::ExpectOK{});
  };

  Actor* const skinOnlyActor = createActor(ColliderType::None);
  Actor* const sdfActor = createActor(ColliderType::Sdf);
  auto& reg = GetRegistry();
  entt::entity const skinOnly = GetEntity(skinOnlyActor);
  entt::entity const sdf = GetEntity(sdfActor);

  auto const expectBounds = [&reg](entt::entity entity, Aabb const& expected) {
    Aabb const actual = GetAabb(reg.get<CBoundingVolume const>(entity).localShape);
    EXPECT_NEAR_EQ(expected.GetMin(), actual.GetMin());
    EXPECT_NEAR_EQ(expected.GetMax(), actual.GetMax());
  };

  // The skin spans x in [-0.5, 0.5] and the physics mesh spans x in [0, 1], so the union differs
  // from both.
  expectBounds(skinOnly, Aabb{Real3{-0.5_r, 0_r, 0_r}, Real3{0.5_r, 1_r, 1_r}});
  expectBounds(sdf, Aabb{Real3{-0.5_r, 0_r, 0_r}, Real3{1_r, 1_r, 1_r}});

  DynamicArray<real> velocity(skinOnlyActor->GetNumDofs(), 0_r);
  for (int node = 1; node < skinOnlyActor->GetNumDofs() / kSpaceDim3; ++node) {
    velocity[kSpaceDim3 * node] = 2_r;
  }
  for (Actor* const actor : {skinOnlyActor, sdfActor}) {
    actor->SetNodeVelocitiesLocal(MakeConstSpan(velocity), test::ExpectOK{});
  }
  UpdateMaxGeometrySpeeds(reg);

  EXPECT_NEAR_EQ(1_r, reg.get<CConservativeStepBounds const>(skinOnly).maxGeometrySpeed);
  EXPECT_NEAR_EQ(2_r, reg.get<CConservativeStepBounds const>(sdf).maxGeometrySpeed);
}

TEST_F(MochiSoftContactSkin, NormalsUseCurrentOrStageStartEmbeddedPositions) {
  // The rotated embedding maps each skin node to a different physics node, exposing index errors.
  Actor* const actor = CreateActor(true, ActorBoundaryElementType::P1Q3, Embedding::Rotated);
  auto& reg = GetRegistry();
  entt::entity const entity = GetEntity(actor);
  auto const physicsNodes = reg.get<CSimplicialMesh const>(entity).mesh->GetNodeCoordinates();
  auto const& contactSkin = reg.get<CContactSkinMesh const>(entity);
  auto const& deformedNodes = reg.get<CDeformedContactSkinNodes const>(entity);
  auto const& surfaceDisc = reg.get<CFemSurfaceDiscretization const>(entity);

  // Current normals read the deformed skin nodes, which UpdateBounds<Current> refreshes.
  auto& current = reg.get<CDisplacementSlice<real, TimeStep::Current>>(entity).value;
  ColumnVector<real> stageStart = ColumnVector<real>::Zero(actor->GetNumDofs());
  for (int node = 0; node < actor->GetNumDofs() / kSpaceDim3; ++node) {
    Real3 const& position = physicsNodes[node];
    current(kSpaceDim3 * node + 2) = 0.2_r * position[1] - 0.4_r * position[0];
    stageStart(kSpaceDim3 * node + 2) = 0.3_r * position[0] + 0.7_r * position[1];
  }
  ecs::InvokeOnEntity(linear_contact_skin::UpdateBounds<TimeStep::Current>, reg, entity);

  // Sample 0 lies on skin element 0.
  MeshDataView const skinView = actor->GetContactSkinMesh();
  Int3 const& element = Unflatten<Int3 const>(skinView.connectivity)[0];
  auto expectedNormal = [&](Span<real const> displacement) {
    auto const displacements = Unflatten<Real3 const>(displacement);
    DynamicArray<Real3> deformedPhysicsNodes(isize(physicsNodes), Real3{});
    for (int node = 0; node < isize(physicsNodes); ++node) {
      deformedPhysicsNodes[node] = physicsNodes[node] + displacements[node];
    }
    DynamicArray<Real3> const skinNodes =
        EmbedSkinNodes(skinView, MakeConstSpan(deformedPhysicsNodes));
    return Normalize(Cross(
        skinNodes[element[1]] - skinNodes[element[0]],
        skinNodes[element[2]] - skinNodes[element[0]]));
  };

  CSimulationParams simulationParams;
  CActiveCollisions</*kIsSync*/ true, TimeStep::Current> collisions(/*numPartitions*/ 1);
  entt::entity const collider{};
  collisions.SetUp(MakeSingletonConstSpan(collider));
  auto& result = collisions.front().collisionResult;
  result.sampleIndices.push_back(0);
  result.jacColliderFromWorld.resize(1, VEye<3>());
  result.jacColliderFromWorldStageStart.resize(1, VEye<3>());

  auto runTest = [&](bool explicitNormals, Real3 const& expected) {
    simulationParams.experimentalEval.explicitNormals = explicitNormals;
    linear_contact_skin::SetupActiveCollisionNormals</*kIsSync*/ true>(
        {},
        {},
        ecs::CtxGlobal<CSimulationParams const>{simulationParams},
        contactSkin,
        deformedNodes,
        surfaceDisc,
        CFinalDisplacementRef<TimeStep::StageStart>{AsConstView(stageStart)},
        reg.get<CRootTransform const>(entity),
        collisions);
    ASSERT_EQ(1, result.normalColliding.size());
    EXPECT_NEAR_EQ(expected, result.normalColliding[0]);
  };

  runTest(/*explicitNormals=*/false, expectedNormal(current.GetConstSpan()));
  runTest(/*explicitNormals=*/true, expectedNormal(stageStart.GetConstSpan()));
}

TEST_F(MochiSoftContactSkin, PlaneContactUsesSkinnedAssemblyAndQueries) {
  // The shifted embedding blends several physics nodes into each skin node, including a negative
  // weight, so per-node forces expose distribution or skin/physics index errors.
  for (Embedding const embedding : {Embedding::Identity, Embedding::Shifted}) {
    SCOPED_TRACE(embedding == Embedding::Identity ? "Identity" : "Shifted");
    SoftActorParams softParams;
    softParams.shape = _mochiContext->CreateModelShape(MakeModel(embedding), test::ExpectOK{});
    softParams.useContactSkin = true;
    softParams.boundaryElementType = ActorBoundaryElementType::P1Q1;
    softParams.worldFromLocal.SetTranslation(Real3{0_r, 0_r, -0.01_r});
    softParams.contact.penaltyCoefficient = 1e6_r;
    Actor* const softActor = _scene->CreateSoftActor(softParams, test::ExpectOK{});
    MOCHI_DEFER(_scene->DestroyActor(softActor->GetHandle()));
    softActor->RegisterQuery(QueryType::ContactPoints, test::ExpectOK{});
    softActor->RegisterQuery(QueryType::NodeContactForces, test::ExpectOK{});

    RigidActorParams planeParams;
    planeParams.shape =
        _mochiContext->CreatePlaneShape(Real3{0_r, 0_r, 1_r}, 0_r, test::ExpectOK{});
    planeParams.isStatic = true;
    planeParams.colliderType = ColliderType::Plane;
    planeParams.contact.penaltyCoefficient = softParams.contact.penaltyCoefficient;
    Actor* const planeActor = _scene->CreateRigidActor(planeParams, test::ExpectOK{});
    MOCHI_DEFER(_scene->DestroyActor(planeActor->GetHandle()));

    _scene->SetGravity({});
    _scene->Step(1e-4_r);

    auto& reg = GetRegistry();
    entt::entity const entity = GetEntity(softActor);
    auto const& contactSnle = reg.get<CSkinnedContactSnle const>(entity);
    ASSERT_TRUE(contactSnle.useInSolver);
    ASSERT_EQ(1, isize(contactSnle.residuals));

    // The actor frame is a pure translation, so local and world forces agree.
    DynamicArray<Real3> nodeForces(softActor->GetNumDofs() / kSpaceDim3, Real3{});
    int const dofOffset = reg.get<CDofOffset const>(entity).dofsOffset;
    for (auto const& [residualOffset, residual] : contactSnle.residuals) {
      for (int i = 0; i < residual.Rows(); ++i) {
        int const localDof = residualOffset + i - dofOffset;
        if (localDof >= 0 && localDof < softActor->GetNumDofs()) {
          nodeForces[localDof / kSpaceDim3][localDof % kSpaceDim3] -= residual[i];
        } else {
          EXPECT_EQ(0_r, residual[i]) << "Contact residual outside the soft actor's DoFs";
        }
      }
    }
    Real3 assembledForce{};
    for (Real3 const& force : nodeForces) {
      assembledForce += force;
    }
    ASSERT_GT(assembledForce[2], 0_r);
    real const forceTolerance = 1e-5_r * Norm(assembledForce);

    auto const contactPoints = softActor->GetContactPointsWorld(test::ExpectOK{});
    ASSERT_FALSE(contactPoints.empty());
    Real3 queryForce{};
    for (ContactPoint const& point : contactPoints) {
      queryForce += point.force;
    }
    EXPECT_NEAR_TOL(queryForce, assembledForce, forceTolerance);

    // Each contact point lies on the contact-skin triangle it reports. The shifted embedding moves
    // the skin away from the tetrahedral boundary, and this step's displacements are negligible.
    MeshDataView const contactSkin = softActor->GetContactSkinMesh();
    ASSERT_TRUE(contactSkin.skinning.has_value());
    DynamicArray<Real3> const skinNodes =
        EmbedSkinNodes(contactSkin, Unflatten<Real3 const>(softActor->GetMesh().coordinates));
    auto const skinTriangles = Unflatten<Int3 const>(contactSkin.connectivity);
    for (ContactPoint const& point : contactPoints) {
      ASSERT_GE(point.elementIndex, 0);
      ASSERT_LT(point.elementIndex, isize(skinTriangles));
      Int3 const& triangle = skinTriangles[point.elementIndex];
      Real3 localPosition{};
      for (int k = 0; k < 3; ++k) {
        localPosition += point.parametricCoords[k] * skinNodes[triangle[k]];
      }
      EXPECT_NEAR_TOL(softParams.worldFromLocal.TransformPoint(localPosition), point.posA, 1e-4_r);
    }

    // Node contact forces are reported on skin nodes; the skinning weights map them to the physics
    // nodes that received the assembled forces.
    SkinningDataView const& skinning = *contactSkin.skinning;
    DynamicArray<Real3> expectedNodeForces(isize(nodeForces), Real3{});
    for (NodeContactForce const& skinNodeForce :
         softActor->GetNodeContactForcesWorld(test::ExpectOK{})) {
      ASSERT_GE(skinNodeForce.index, 0);
      ASSERT_LT(skinNodeForce.index, contactSkin.GetNumNodes());
      for (int k = 0; k < skinning.weightsPerNode; ++k) {
        int const entry = skinning.weightsPerNode * skinNodeForce.index + k;
        expectedNodeForces[skinning.indices[entry]] +=
            skinning.weights[entry] * skinNodeForce.force;
      }
    }
    for (int node = 0; node < isize(nodeForces); ++node) {
      EXPECT_NEAR_TOL(expectedNodeForces[node], nodeForces[node], forceTolerance)
          << "Physics node " << node;
    }
  }
}
