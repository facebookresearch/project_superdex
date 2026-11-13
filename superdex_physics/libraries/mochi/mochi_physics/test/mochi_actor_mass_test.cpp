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
#include <mochi_core/utils/array_utils.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/src/mochi_actor.h>
#include <mochi_physics/src/mochi_rigid.h>
#include <mochi_physics/src/mochi_simulation.h>

#include <functional>
#include <limits>
#include <vector>

using namespace mochi;
using namespace mochi::experimental;

static_assert(
    static_cast<int>(ActorType::Count) == 6,
    "Please update the tests below if new actor types are introduced");

real constexpr kApiRelTol = kDefaultNearEqualEpsilon<real>;
real constexpr kPhysicsMassRelTol = 0.02_r;
real constexpr kRoundoffTol = 64_r * std::numeric_limits<real>::epsilon();

// =============================================================================
// Fixtures
// =============================================================================

class ActorMassTest : public test::MochiSceneTestBase {};

// =============================================================================
// Helper Functions
// =============================================================================

static real MeasureActorMass(Actor* actor) {
  // Make sure the query is registered. Redundant registration is OK.
  actor->RegisterQuery(QueryType::ContactPoints, test::ExpectOK{});

  // Step the scene several times so actor comes to rest
  auto* scene = actor->GetScene();
  for (int i = 0; i < 100; ++i) {
    scene->Step(0.01);
  }

  // Measure mass via total contact force
  real totalVerticalForce = {};
  for (auto const& c : actor->GetContactPointsWorld(test::ExpectOK{})) {
    totalVerticalForce += c.force[1];
  }

  return totalVerticalForce / Abs(scene->GetGravity()[1]);
}

static void CreateGroundPlane(Scene* scene, Context* context) {
  auto planeShape = context->CreatePlaneShape(Real3{0_r, 1_r, 0_r}, 0_r, test::ExpectOK{});
  auto planeParams = RigidActorParams{.shape = planeShape, .isStatic = true};
  scene->CreateRigidActor(planeParams, test::ExpectOK{});
}

// =============================================================================
// Physics Validation Test Infrastructure
// =============================================================================

struct PhysicsTestConfig {
  real expectedMass;
  real expectedDensity;
  bool supportsGetSetDensity;
  real newDensity;
  real newExpectedMass;
  bool supportsPhysicsValidation;
};

using ActorFactory =
    std::function<std::pair<Actor*, PhysicsTestConfig>(Scene*, Context*, Real3 const& startPos)>;

struct PhysicsValidationTestCase {
  std::string name;
  ActorFactory createActor;
};

// =============================================================================
// Actor Factory Functions
// =============================================================================

static PhysicsValidationTestCase CreateRigidActorTestCase() {
  return {
      .name = "RigidActor",
      .createActor =
          [](Scene* scene, Context* context, Real3 const& startPos) {
            auto constexpr kScale = Real3{0.1_r, 0.2_r, 0.3_r};
            auto constexpr kVolume = kScale[0] * kScale[1] * kScale[2];
            auto constexpr kDensity = 500_r;
            auto constexpr kNewDensity = 1000_r;

            auto [coords, conn] = test::CreateMinimalTetMeshUnitCube(kScale);
            auto shape = context->CreateTetMeshShape(
                Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), test::ExpectOK{});
            RigidActorParams params;
            params.shape = shape;
            params.colliderType = ColliderType::Box;
            params.density = kDensity;
            params.worldFromLocal.SetTranslation(startPos);

            return std::make_pair(
                scene->CreateRigidActor(params, test::ExpectOK{}),
                PhysicsTestConfig{
                    .expectedMass = kDensity * kVolume,
                    .expectedDensity = kDensity,
                    .supportsGetSetDensity = true,
                    .newDensity = kNewDensity,
                    .newExpectedMass = kNewDensity * kVolume,
                    .supportsPhysicsValidation = true,
                });
          },
  };
}

static PhysicsValidationTestCase CreateSoftActorTestCase() {
  return {
      .name = "SoftActor",
      .createActor =
          [](Scene* scene, Context* context, Real3 const& startPos) {
            auto constexpr kScale = Real3{0.1_r, 0.2_r, 0.3_r};
            auto constexpr kVolume = kScale[0] * kScale[1] * kScale[2];
            auto constexpr kDensity = 500_r;
            auto constexpr kNewDensity = 1000_r;

            auto [coords, conn] = test::CreateMinimalTetMeshUnitCube(kScale);
            auto shape = context->CreateTetMeshShape(
                Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), test::ExpectOK{});
            SoftActorParams params;
            params.shape = shape;
            params.material.density = kDensity;
            params.worldFromLocal.SetTranslation(startPos);

            return std::make_pair(
                scene->CreateSoftActor(params, test::ExpectOK{}),
                PhysicsTestConfig{
                    .expectedMass = kDensity * kVolume,
                    .expectedDensity = kDensity,
                    .supportsGetSetDensity = true,
                    .newDensity = kNewDensity,
                    .newExpectedMass = kNewDensity * kVolume,
                    .supportsPhysicsValidation = true,
                });
          },
  };
}

static PhysicsValidationTestCase CreateShellActorTestCase() {
  return {
      .name = "ShellActor",
      .createActor =
          [](Scene* scene, Context* context, Real3 const& startPos) {
            // Create a flat shell patch (1x1 square) that can rest on the ground
            auto constexpr kSize = 0.2_r;
            auto constexpr kArea = kSize * kSize;
            auto constexpr kDensity = 500_r;

            DynamicArray<Real3> coords = {
                Real3{0_r, 0_r, 0_r},
                Real3{kSize, 0_r, 0_r},
                Real3{kSize, 0_r, kSize},
                Real3{0_r, 0_r, kSize},
            };
            DynamicArray<Int3> conn = {
                Int3{0, 1, 2},
                Int3{0, 2, 3},
            };

            auto shape = context->CreateTriMeshShape(
                Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), test::ExpectOK{});
            ShellActorParams params;
            params.shape = shape;
            params.material.density = kDensity;
            params.worldFromLocal.SetTranslation(startPos);

            return std::make_pair(
                CreateShellActor(scene, params, test::ExpectOK{}),
                PhysicsTestConfig{
                    .expectedMass = kDensity * kArea,
                    .expectedDensity = 0_r,
                    .supportsGetSetDensity = false,
                    .newDensity = 0_r,
                    .newExpectedMass = 0_r,
                    .supportsPhysicsValidation = true,
                });
          },
  };
}

static PhysicsValidationTestCase CreateRodActorTestCase() {
  return {
      .name = "RodActor",
      .createActor =
          [](Scene* scene, Context* context, Real3 const& startPos) {
            // Create a horizontal rod that can rest on the ground
            auto constexpr kLength = 0.2_r;
            auto constexpr kLinearDensity = 10_r;

            DynamicArray<Real3> nodes = {
                Real3{0_r, 0_r, 0_r},
                Real3{kLength / 4_r, 0_r, 0_r},
                Real3{kLength / 2_r, 0_r, 0_r},
                Real3{3_r * kLength / 4_r, 0_r, 0_r},
                Real3{kLength, 0_r, 0_r},
            };

            int const numElements = isize(nodes) - 1;
            DynamicArray<Real3> elementFrameAxes(numElements, Real3{0_r, 1_r, 0_r});

            auto shape = CreatePolylineShape(
                context, nodes, elementFrameAxes, /*isClosedLoop=*/false, test::ExpectOK{});
            RodActorParams params;
            params.shape = shape;
            params.material.linearDensity = kLinearDensity;
            params.material.linearRotationalInertia = 1_r;
            params.material.axialStiffness = 1e5_r;
            params.material.torsionalStiffness = 1e3_r;
            params.material.flexuralStiffness = {1e3_r, 1e3_r};
            params.worldFromLocal.SetTranslation(startPos);

            return std::make_pair(
                scene->CreateRodActor(params, test::ExpectOK{}),
                PhysicsTestConfig{
                    .expectedMass = kLinearDensity * kLength,
                    .expectedDensity = 0_r,
                    .supportsGetSetDensity = false,
                    .newDensity = 0_r,
                    .newExpectedMass = 0_r,
                    .supportsPhysicsValidation = false,
                });
          },
  };
}

static PhysicsValidationTestCase CreateArticulatedActorTestCase() {
  return {
      .name = "ArticulatedActor",
      .createActor =
          [](Scene* scene, Context* context, Real3 const& startPos) {
            auto constexpr kScale = Real3{0.1_r, 0.1_r, 0.1_r};
            auto constexpr kVolume = kScale[0] * kScale[1] * kScale[2];
            auto constexpr kDensity = 500_r;
            int constexpr kNumLinks = 2;

            auto [coords, conn] = test::CreateMinimalTetMeshUnitCube(kScale);
            auto cubeShape = context->CreateTetMeshShape(
                Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), test::ExpectOK{});

            ArticulatedActorParams actorParams;
            actorParams.joints = {
                {
                    .type = ArticulatedJointType::Free //
                },
                {
                    .type = ArticulatedJointType::Spherical,
                    .parentLinkFromJoint = TransformRT{Real3{kScale[0], 0_r, 0_r}} //
                }};
            actorParams.links = {
                {
                    .parentLink = -1,
                    .shape = cubeShape,
                    .colliderType = ColliderType::Box,
                    .density = kDensity //
                },
                {
                    .parentLink = 0,
                    .parentJointFromLink = TransformRT{Real3{kScale[0], 0_r, 0_r}},
                    .shape = cubeShape,
                    .colliderType = ColliderType::Box,
                    .density = kDensity //
                }};
            actorParams.worldFromRoot.SetTranslation(startPos);

            return std::make_pair(
                scene->CreateArticulatedActor(actorParams, test::ExpectOK{}),
                PhysicsTestConfig{
                    .expectedMass = kDensity * kVolume * kNumLinks,
                    .expectedDensity = 0_r,
                    .supportsGetSetDensity = false,
                    .newDensity = 0_r,
                    .newExpectedMass = 0_r,
                    .supportsPhysicsValidation = false,
                });
          },
  };
}

#if MOCHI_ENABLE_ROM_ACTORS
static PhysicsValidationTestCase CreateRomActorTestCase() {
  return {
      .name = "RomActor",
      .createActor =
          [](Scene* scene, Context* context, Real3 const& startPos) {
            auto constexpr kScale = Real3{0.2_r, 0.2_r, 0.2_r};
            auto constexpr kVolume = kScale[0] * kScale[1] * kScale[2];
            auto constexpr kDensity = 500_r;
            auto constexpr kNewDensity = 1000_r;

            auto [coords, conn] = test::CreateMinimalTetMeshUnitCube(kScale);
            auto shape = context->CreateTetMeshShape(
                Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), test::ExpectOK{});
            SoftActorParams params;
            params.shape = shape;
            params.material.density = kDensity;
            params.worldFromLocal.SetTranslation(startPos);

            ExperimentalSoftActorParams experimentalParams;
            experimentalParams.rom = RomParams{.source = "polynomial_crom_order_1"};

            return std::make_pair(
                CreateSoftActor(scene, params, experimentalParams, test::ExpectOK{}),
                PhysicsTestConfig{
                    .expectedMass = kDensity * kVolume,
                    .expectedDensity = kDensity,
                    .supportsGetSetDensity = true,
                    .newDensity = kNewDensity,
                    .newExpectedMass = kNewDensity * kVolume,
                    .supportsPhysicsValidation = true,
                });
          },
  };
}
#endif // MOCHI_ENABLE_ROM_ACTORS

// =============================================================================
// Parameterized Physics Validation Test
// =============================================================================

class ActorMassPhysicsTest : public test::MochiSceneTestBase,
                             public testing::WithParamInterface<PhysicsValidationTestCase> {
 public:
  void SetUp() override {
    test::MochiSceneTestBase::SetUp();
    // Settling to rest, then reading contact force as weight, relies on backward Euler's damping.
    test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
  }
};

TEST_P(ActorMassPhysicsTest, ValidateMassViaContactForces) {
  auto const& testCase = GetParam();
  auto constexpr kStartPos = Real3{0_r, 0.1_r, 0_r};

  _scene->SetGravity(kDefaultGravity);
  CreateGroundPlane(_scene, _mochiContext);

  auto result = testCase.createActor(_scene, _mochiContext, kStartPos);
  auto* actor = result.first;
  auto config = result.second;
  MOCHI_DEFER(_scene->DestroyActor(actor));

  // Validate initial mass via physics simulation (if supported)
  if (config.supportsPhysicsValidation) {
    auto actualMass = MeasureActorMass(actor);
    EXPECT_NEAR_RTOL(actualMass, config.expectedMass, kPhysicsMassRelTol);
  }

  // Validate API consistency
  EXPECT_NEAR_RTOL(config.expectedMass, actor->GetMass(test::ExpectOK{}), kApiRelTol);

  if (config.supportsGetSetDensity) {
    EXPECT_NEAR_RTOL(config.expectedDensity, actor->GetDensity(test::ExpectOK{}), kApiRelTol);
  } else {
    EXPECT_EQ(0_r, actor->GetDensity(test::ExpectNotOK{}));
  }

  // Test density change if supported
  if (config.supportsGetSetDensity) {
    actor->SetDensity(config.newDensity, test::ExpectOK{});

    // Validate new mass via physics simulation (if supported)
    if (config.supportsPhysicsValidation) {
      auto actualMass = MeasureActorMass(actor);
      EXPECT_NEAR_RTOL(actualMass, config.newExpectedMass, kPhysicsMassRelTol);
    }

    EXPECT_NEAR_RTOL(config.newExpectedMass, actor->GetMass(test::ExpectOK{}), kApiRelTol);
    EXPECT_NEAR_RTOL(config.newDensity, actor->GetDensity(test::ExpectOK{}), kApiRelTol);
  } else {
    actor->SetDensity(1000_r, test::ExpectNotOK{});
  }
}

static std::vector<PhysicsValidationTestCase> CreateActorMassPhysicsTestCases() {
  std::vector<PhysicsValidationTestCase> testCases = {
      CreateRigidActorTestCase(),
      CreateSoftActorTestCase(),
      CreateShellActorTestCase(),
      CreateRodActorTestCase(),
      CreateArticulatedActorTestCase(),
  };
#if MOCHI_ENABLE_ROM_ACTORS
  testCases.push_back(CreateRomActorTestCase());
#endif // MOCHI_ENABLE_ROM_ACTORS
  return testCases;
}

INSTANTIATE_TEST_SUITE_P(
    AllActorTypes,
    ActorMassPhysicsTest,
    testing::ValuesIn(CreateActorMassPhysicsTestCases()),
    [](testing::TestParamInfo<PhysicsValidationTestCase> const& info) { return info.param.name; });

// =============================================================================
// Default Density Tests
// =============================================================================

TEST_F(ActorMassTest, DefaultDensity) {
  auto constexpr kScale = Real3{0.1_r, 0.2_r, 0.3_r};
  auto constexpr kVolume = kScale[0] * kScale[1] * kScale[2];
  auto constexpr kCenter = kScale / 2_r;

  auto [tetCoordinates, tetConnectivity] = test::CreateMinimalTetMeshUnitCube(kScale);
  auto [triCoordinates, triConnectivity] = test::CreateMinimalTriMeshUnitCube(kScale);
  // A tetrahedral shape provides its volume and center of mass; for a triangle shape, rigid
  // creation computes them from its surface.
  for (ShapeHandle const& shape :
       {_mochiContext->CreateTetMeshShape(
            Flatten(MakeSpan(tetCoordinates)),
            Flatten(MakeSpan(tetConnectivity)),
            test::ExpectOK{}),
        _mochiContext->CreateTriMeshShape(
            Flatten(MakeSpan(triCoordinates)),
            Flatten(MakeSpan(triConnectivity)),
            test::ExpectOK{})}) {
    RigidActorParams params{.shape = shape, .colliderType = ColliderType::Box, .isStatic = false};
    auto* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
    MOCHI_DEFER(_scene->DestroyActor(actor));

    EXPECT_NEAR_EQ(kDefaultDensity * kVolume, actor->GetMass(test::ExpectOK{}));
    EXPECT_NEAR_EQ(kDefaultDensity, actor->GetDensity(test::ExpectOK{}));
    EXPECT_NEAR_EQ(kCenter, actor->GetRigidCenterOfMassLocal(test::ExpectOK{}));
  }
}

// =============================================================================
// API Verification Tests - Per Actor Type
// =============================================================================

TEST_F(ActorMassTest, RigidActor_Api) {
  auto constexpr kScale = Real3{0.1_r, 0.2_r, 0.3_r};
  auto constexpr kVolume = kScale[0] * kScale[1] * kScale[2];
  auto constexpr kDensity = 2500_r;
  auto constexpr kMass = kDensity * kVolume;
  auto constexpr kNewDensity = 5000_r;
  auto constexpr kNewMass = kNewDensity * kVolume;

  auto [tetCoordinates, tetConnectivity] = test::CreateMinimalTetMeshUnitCube(kScale);
  auto shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(tetCoordinates)), Flatten(MakeSpan(tetConnectivity)), test::ExpectOK{});
  // Dynamic actor created from density.
  {
    RigidActorParams params{
        .shape = shape, .colliderType = ColliderType::None, .density = kDensity};
    auto* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
    MOCHI_DEFER(_scene->DestroyActor(actor));

    EXPECT_NEAR_RTOL(kMass, actor->GetMass(test::ExpectOK{}), kApiRelTol);
    EXPECT_EQ(kDensity, actor->GetDensity(test::ExpectOK{}));

    actor->SetDensity(kNewDensity, test::ExpectOK{});
    EXPECT_NEAR_RTOL(kNewMass, actor->GetMass(test::ExpectOK{}), kApiRelTol);
    EXPECT_EQ(kNewDensity, actor->GetDensity(test::ExpectOK{}));
  }

  // Dynamic actor created from mass.
  {
    RigidActorParams params{.shape = shape, .colliderType = ColliderType::None, .mass = kMass};
    auto* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
    MOCHI_DEFER(_scene->DestroyActor(actor));

    EXPECT_EQ(kMass, actor->GetMass(test::ExpectOK{}));
    EXPECT_NEAR_RTOL(kDensity, actor->GetDensity(test::ExpectOK{}), kApiRelTol);

    actor->SetDensity(kNewDensity, test::ExpectOK{});
    EXPECT_NEAR_RTOL(kNewMass, actor->GetMass(test::ExpectOK{}), kApiRelTol);
    EXPECT_EQ(kNewDensity, actor->GetDensity(test::ExpectOK{}));
  }

  // Static actor.
  {
    RigidActorParams params{.shape = shape, .colliderType = ColliderType::Box, .isStatic = true};
    auto* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
    MOCHI_DEFER(_scene->DestroyActor(actor));

    EXPECT_EQ(0_r, actor->GetMass(test::ExpectNotOK{}));
    EXPECT_EQ(0_r, actor->GetDensity(test::ExpectNotOK{}));
    actor->SetDensity(kNewDensity, test::ExpectNotOK{});
  }
}

TEST_F(ActorMassTest, SoftActor_Api) {
  auto constexpr kScale = Real3{0.1_r, 0.2_r, 0.3_r};
  auto constexpr kVolume = kScale[0] * kScale[1] * kScale[2];
  auto constexpr kDensity = 1500_r;
  auto constexpr kMass = kDensity * kVolume;
  auto constexpr kNewDensity = 3000_r;
  auto constexpr kNewMass = kNewDensity * kVolume;

  auto [tetCoordinates, tetConnectivity] = test::CreateMinimalTetMeshUnitCube(kScale);
  auto shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(tetCoordinates)), Flatten(MakeSpan(tetConnectivity)), test::ExpectOK{});
  SoftActorParams params{.shape = shape, .material = {.density = kDensity}};
  auto* actor = _scene->CreateSoftActor(params, test::ExpectOK{});
  MOCHI_DEFER(_scene->DestroyActor(actor));

  EXPECT_NEAR_RTOL(kMass, actor->GetMass(test::ExpectOK{}), kApiRelTol);
  EXPECT_NEAR_RTOL(kDensity, actor->GetDensity(test::ExpectOK{}), kApiRelTol);
  EXPECT_NEAR_EQ(kScale / 2_r, actor->GetRigidCenterOfMassLocal(test::ExpectOK{}));

  actor->SetDensity(kNewDensity, test::ExpectOK{});
  EXPECT_NEAR_RTOL(kNewMass, actor->GetMass(test::ExpectOK{}), kApiRelTol);
  EXPECT_NEAR_RTOL(kNewDensity, actor->GetDensity(test::ExpectOK{}), kApiRelTol);
  // The lumped mass follows: it adds up to the mass once per DoF component.
  auto const& lumpedMass = GetRegistry().get<CLumpedMassMatrix const>(GetEntity(actor)).values;
  EXPECT_NEAR_RTOL(3_r * kNewMass, HSum(MakeConstSpan(lumpedMass)), kApiRelTol);
}

TEST_F(ActorMassTest, ShellActor_Api) {
  auto constexpr kExpectedArea = 6_r;
  auto constexpr kDensity = 500_r;
  auto constexpr kMass = kDensity * kExpectedArea;

  auto [triCoordinates, triConnectivity] = test::CreateMinimalTriMeshUnitCube();
  auto triMeshShape = _mochiContext->CreateTriMeshShape(
      Flatten(MakeSpan(triCoordinates)), Flatten(MakeSpan(triConnectivity)), test::ExpectOK{});
  ShellActorParams params{.shape = triMeshShape, .material = {.density = kDensity}};
  auto* actor = CreateShellActor(_scene, params, test::ExpectOK{});
  MOCHI_DEFER(_scene->DestroyActor(actor));

  EXPECT_NEAR_RTOL(kMass, actor->GetMass(test::ExpectOK{}), kApiRelTol);
  EXPECT_EQ(0_r, actor->GetDensity(test::ExpectNotOK{}));
  actor->SetDensity(1000_r, test::ExpectNotOK{});
}

TEST_F(ActorMassTest, RodActor_Api) {
  auto constexpr kLength = 1_r;
  auto constexpr kLinearDensity = 2_r;
  auto constexpr kMass = kLinearDensity * kLength;

  DynamicArray<Real3> nodes = {
      Real3{0_r, 0_r, 0_r},
      Real3{0.25_r, 0_r, 0_r},
      Real3{0.5_r, 0_r, 0_r},
      Real3{0.75_r, 0_r, 0_r},
      Real3{1_r, 0_r, 0_r}};

  int const numElements = isize(nodes) - 1;
  DynamicArray<Real3> elementFrameAxes(numElements, Real3{0_r, 1_r, 0_r});

  auto shape = CreatePolylineShape(
      _mochiContext, nodes, elementFrameAxes, /*isClosedLoop=*/false, test::ExpectOK{});
  RodActorParams params{
      .shape = shape,
      .material = {
          .linearDensity = kLinearDensity,
          .linearRotationalInertia = 1_r,
          .axialStiffness = 1e3_r,
          .torsionalStiffness = 1e1_r,
          .flexuralStiffness = {1e1_r, 1e1_r}}};

  auto* actor = _scene->CreateRodActor(params, test::ExpectOK{});
  MOCHI_DEFER(_scene->DestroyActor(actor));

  EXPECT_NEAR_RTOL(kMass, actor->GetMass(test::ExpectOK{}), kApiRelTol);
  EXPECT_EQ(0_r, actor->GetDensity(test::ExpectNotOK{}));
  actor->SetDensity(1000_r, test::ExpectNotOK{});
}

TEST_F(ActorMassTest, ArticulatedActor_Api) {
  auto constexpr kScale = Real3{0.1_r, 0.2_r, 0.3_r};
  auto constexpr kVolume = kScale[0] * kScale[1] * kScale[2];
  auto constexpr kDensity = 1000_r;
  auto constexpr kLinkMass = kDensity * kVolume;
  int constexpr kNumLinks = 2;
  auto constexpr kExpectedTotalMass = kLinkMass * kNumLinks;

  auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube(kScale);
  auto cubeShape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});

  ArticulatedActorParams actorParams;
  actorParams.joints = {
      {.type = ArticulatedJointType::Free}, //
      {.type = ArticulatedJointType::Spherical}};
  actorParams.links = {
      {
          .parentLink = -1,
          .shape = cubeShape,
          .colliderType = ColliderType::None,
          .density = kDensity //
      },
      {
          .parentLink = 0,
          .shape = cubeShape,
          .colliderType = ColliderType::None,
          .density = kDensity //
      }};
  auto* actor = _scene->CreateArticulatedActor(actorParams, test::ExpectOK{});
  MOCHI_DEFER(_scene->DestroyActor(actor));

  EXPECT_NEAR_RTOL(kExpectedTotalMass, actor->GetMass(test::ExpectOK{}), kApiRelTol);
  EXPECT_EQ(0_r, actor->GetDensity(test::ExpectNotOK{}));
  actor->SetDensity(1000_r, test::ExpectNotOK{});
}

// A link welded to the world is static, but keeps the inertia of its shape. A link without a shape
// has no inertia, and setting its density has no effect.
TEST_F(ActorMassTest, ArticulatedLink_Api) {
  auto constexpr kScale = Real3{0.1_r, 0.2_r, 0.3_r};
  auto constexpr kVolume = kScale[0] * kScale[1] * kScale[2];
  auto constexpr kDensity = 1000_r;
  auto constexpr kLinkMass = kDensity * kVolume;
  auto constexpr kNewDensity = 2000_r;
  auto constexpr kNewLinkMass = kNewDensity * kVolume;

  auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube(kScale);
  auto cubeShape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});

  ArticulatedActorParams actorParams;
  actorParams.joints = {
      {.type = ArticulatedJointType::Hard},
      {.type = ArticulatedJointType::Spherical},
      {.type = ArticulatedJointType::Spherical}};
  actorParams.links = {
      {.parentLink = -1,
       .shape = cubeShape,
       .colliderType = ColliderType::None,
       .density = kDensity},
      {.parentLink = 0,
       .shape = cubeShape,
       .colliderType = ColliderType::None,
       .density = kDensity},
      {.parentLink = 1, .colliderType = ColliderType::None}};
  auto* actor = _scene->CreateArticulatedActor(actorParams, test::ExpectOK{});
  MOCHI_DEFER(_scene->DestroyActor(actor));
  auto const links = actor->GetNestedLinkActors(test::ExpectOK{});
  Actor* staticLink = _scene->GetActor(links[0]);
  Actor* dynamicLink = _scene->GetActor(links[1]);
  Actor* dummyLink = _scene->GetActor(links[2]);
  ASSERT_TRUE(staticLink->IsStatic());

  for (Actor* link : {staticLink, dynamicLink}) {
    EXPECT_NEAR_RTOL(kLinkMass, link->GetMass(test::ExpectOK{}), kApiRelTol);
    EXPECT_NEAR_RTOL(kDensity, link->GetDensity(test::ExpectOK{}), kApiRelTol);
  }
  EXPECT_EQ(0_r, dummyLink->GetMass(test::ExpectOK{}));
  EXPECT_EQ(0_r, dummyLink->GetDensity(test::ExpectOK{}));
  dummyLink->SetDensity(kNewDensity, test::ExpectOK{});
  EXPECT_EQ(0_r, dummyLink->GetMass(test::ExpectOK{}));
  EXPECT_EQ(0_r, dummyLink->GetDensity(test::ExpectOK{}));

  staticLink->SetDensity(kNewDensity, test::ExpectOK{});
  EXPECT_NEAR_RTOL(kNewLinkMass, staticLink->GetMass(test::ExpectOK{}), kApiRelTol);
  EXPECT_NEAR_RTOL(kNewDensity, staticLink->GetDensity(test::ExpectOK{}), kApiRelTol);
  EXPECT_NEAR_RTOL(kNewLinkMass + kLinkMass, actor->GetMass(test::ExpectOK{}), kApiRelTol);
}

// A link without a shape takes its inertia from an explicit mass, center of mass, and moment of
// inertia. It has no volume, so it has no density.
TEST_F(ActorMassTest, ArticulatedShapelessLinkWithInertia_Api) {
  auto constexpr kMass = 2_r;
  auto constexpr kCom = Real3{0.2_r, 0_r, 0_r};
  auto constexpr kMoi = Real6{0.01_r, 0_r, 0_r, 0.02_r, 0_r, 0.03_r};

  ArticulatedActorParams actorParams;
  actorParams.joints = {
      {.type = ArticulatedJointType::Hard},
      {.type = ArticulatedJointType::Revolute, .axis = Real3{0_r, 0_r, 1_r}}};
  actorParams.links = {
      {.parentLink = -1},
      {.parentLink = 0, .mass = kMass, .centerOfMass = kCom, .momentOfInertia = kMoi}};
  auto* actor = _scene->CreateArticulatedActor(actorParams, test::ExpectOK{});
  MOCHI_DEFER(_scene->DestroyActor(actor));
  Actor* link = _scene->GetActor(actor->GetNestedLinkActors(test::ExpectOK{})[1]);

  EXPECT_NEAR_RTOL(kMass, link->GetMass(test::ExpectOK{}), kApiRelTol);
  EXPECT_NEAR_RTOL(kMass, actor->GetMass(test::ExpectOK{}), kApiRelTol);
  EXPECT_EQ(kCom, link->GetRigidCenterOfMassLocal(test::ExpectOK{}));
  EXPECT_EQ(kMoi, link->GetRigidMomentOfInertiaLocal(test::ExpectOK{}));
  EXPECT_EQ(0_r, link->GetDensity(test::ExpectOK{}));
  link->SetDensity(1000_r, test::ExpectOK{});
  EXPECT_NEAR_RTOL(kMass, link->GetMass(test::ExpectOK{}), kApiRelTol);
}

// A pendulum whose bob has no shape swings exactly like one whose bob has a shape with the same
// explicit inertia.
TEST_F(ActorMassTest, ArticulatedShapelessLinkWithInertia_MatchesShapedLink) {
  auto constexpr kScale = Real3{0.1_r, 0.1_r, 0.1_r};
  auto constexpr kMass = 2_r;
  auto constexpr kCom = Real3{0.3_r, 0_r, 0_r};
  auto constexpr kMoi = Real6{0.01_r, 0_r, 0_r, 0.01_r, 0_r, 0.01_r};

  auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube(kScale);
  auto cubeShape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});

  auto createPendulum = [&](ShapeHandle bobShape, Real3 const& origin) {
    ArticulatedActorParams actorParams;
    actorParams.worldFromRoot.SetTranslation(origin);
    actorParams.joints = {
        {.type = ArticulatedJointType::Hard},
        {.type = ArticulatedJointType::Revolute, .axis = Real3{0_r, 0_r, 1_r}}};
    actorParams.links = {
        {.parentLink = -1},
        {.parentLink = 0,
         .shape = bobShape,
         .colliderType = ColliderType::None,
         .mass = kMass,
         .centerOfMass = kCom,
         .momentOfInertia = kMoi}};
    Actor* actor = _scene->CreateArticulatedActor(actorParams, test::ExpectOK{});
    return _scene->GetActor(actor->GetNestedLinkActors(test::ExpectOK{})[1]);
  };
  auto constexpr kShapedOrigin = Real3{0_r, 0_r, 0_r};
  auto constexpr kShapelessOrigin = Real3{0_r, 0_r, 5_r};
  Actor* shapedBob = createPendulum(cubeShape, kShapedOrigin);
  Actor* shapelessBob = createPendulum(ShapeHandle{}, kShapelessOrigin);

  for (int i = 0; i < 30; ++i) {
    _scene->Step(0.01);
  }

  TransformRT const shaped = shapedBob->GetRootTransform();
  TransformRT const shapeless = shapelessBob->GetRootTransform();
  Real3 swingAxis;
  real swingAngle = 0_r;
  shaped.GetRotation().ToAxisAngle(&swingAxis, &swingAngle);
  EXPECT_GT(swingAngle, 0.1_r) << "The pendulum should swing under gravity.";
  EXPECT_NEAR_TOL(shaped.GetRotation(), shapeless.GetRotation(), 1e-4_r);
  EXPECT_NEAR_TOL(
      shaped.GetTranslation() - kShapedOrigin,
      shapeless.GetTranslation() - kShapelessOrigin,
      1e-4_r);
}

TEST_IF_F(MOCHI_ENABLE_ROM_ACTORS, ActorMassTest, RomActor_Api) {
  auto constexpr kScale = Real3{0.2_r, 0.2_r, 0.2_r};
  auto constexpr kVolume = kScale[0] * kScale[1] * kScale[2];
  auto constexpr kDensity = 1200_r;
  auto constexpr kMass = kDensity * kVolume;
  auto constexpr kNewDensity = 5000_r;
  auto constexpr kNewMass = kNewDensity * kVolume;

  auto [tetCoordinates, tetConnectivity] = test::CreateMinimalTetMeshUnitCube(kScale);
  auto shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(tetCoordinates)), Flatten(MakeSpan(tetConnectivity)), test::ExpectOK{});
  SoftActorParams params{.shape = shape, .material = SoftMaterialParams{.density = kDensity}};

  ExperimentalSoftActorParams experimentalParams;
  experimentalParams.rom = RomParams{.source = "polynomial_crom_order_1"};

  auto* actor = CreateSoftActor(_scene, params, experimentalParams, test::ExpectOK{});
  MOCHI_DEFER(_scene->DestroyActor(actor));

  EXPECT_NEAR_RTOL(kMass, actor->GetMass(test::ExpectOK{}), kApiRelTol);
  EXPECT_NEAR_RTOL(kDensity, actor->GetDensity(test::ExpectOK{}), kApiRelTol);

  actor->SetDensity(kNewDensity, test::ExpectOK{});
  EXPECT_NEAR_RTOL(kNewMass, actor->GetMass(test::ExpectOK{}), kApiRelTol);
  EXPECT_NEAR_RTOL(kNewDensity, actor->GetDensity(test::ExpectOK{}), kApiRelTol);
}

// =============================================================================
// SetInertiaProperties Tests
// =============================================================================

TEST_F(ActorMassTest, SetInertiaProperties_RigidActor_Roundtrip) {
  auto constexpr kScale = Real3{0.1_r, 0.2_r, 0.3_r};

  auto [coords, conn] = test::CreateMinimalTetMeshUnitCube(kScale);
  auto shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), test::ExpectOK{});
  RigidActorParams params{.shape = shape, .colliderType = ColliderType::Box, .density = 500_r};
  auto* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
  MOCHI_DEFER(_scene->DestroyActor(actor));

  auto constexpr kNewMass = 5_r;
  auto constexpr kNewCom = Real3{0.1_r, 0.2_r, 0.3_r};
  auto constexpr kNewMoi = Real6{2_r, 0.1_r, 0.2_r, 3_r, 0.3_r, 4_r};

  actor->SetInertiaProperties(kNewMass, kNewCom, kNewMoi, test::ExpectOK{});

  EXPECT_EQ(kNewMass, actor->GetMass(test::ExpectOK{}));
  auto const moi = actor->GetRigidMomentOfInertiaLocal(test::ExpectOK{});
  EXPECT_EQ(kNewCom, actor->GetRigidCenterOfMassLocal(test::ExpectOK{}));
  EXPECT_SPAN_EQ(kNewMoi, moi);
}

TEST_F(ActorMassTest, SetInertiaProperties_ComChangeMatchesActorCreatedWithCom) {
  _scene->SetGravity(Real3{});

  auto [coords, conn] = test::CreateMinimalTetMeshUnitCube(Real3{0.1_r, 0.2_r, 0.3_r});
  auto shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), test::ExpectOK{});

  auto const worldFromLocal = TransformRT{
      Quaternion::FromAxisAngle(Normalize(Real3{1_r, 2_r, 3_r}), 0.3_r * kPI),
      Real3{1_r, 2_r, 3_r}};
  auto constexpr kLinearVel = Real3{0.1_r, -0.2_r, 0.3_r};
  auto constexpr kAngularVel = Real3{1_r, 2_r, -1.5_r};
  auto constexpr kMass = 5_r;
  auto constexpr kNewCom = Real3{0.4_r, -0.3_r, 0.2_r};
  auto constexpr kMoi = Real6{2_r, 0.1_r, 0.2_r, 3_r, 0.3_r, 4_r};

  // Actor whose CoM is changed while moving.
  RigidActorParams params{
      .shape = shape,
      .worldFromLocal = worldFromLocal,
      .colliderType = ColliderType::None,
      .density = 500_r};
  auto* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
  MOCHI_DEFER(_scene->DestroyActor(actor));
  actor->SetVelocity(kLinearVel, kAngularVel, test::ExpectOK{});
  Real3 const oldCom = actor->GetRigidCenterOfMassLocal(test::ExpectOK{});
  auto const rootBefore = actor->GetRootTransform();

  actor->SetInertiaProperties(kMass, kNewCom, kMoi, test::ExpectOK{});

  // The root transform is unchanged; the CoM transform follows the new local CoM.
  EXPECT_EQ(rootBefore, actor->GetRootTransform());
  EXPECT_NEAR_EQ(
      rootBefore.TransformPoint(kNewCom),
      actor->GetCenterOfMassTransform(test::ExpectOK{}).GetTranslation());

  // The rigid velocity field is preserved, so the CoM velocity follows the CoM shift.
  Real3 const expectedComVel =
      kLinearVel + Cross(kAngularVel, worldFromLocal.GetRotation() * (kNewCom - oldCom));
  EXPECT_NEAR_EQ(expectedComVel, actor->GetLinearVelocity(test::ExpectOK{}));
  EXPECT_NEAR_EQ(kAngularVel, actor->GetAngularVelocity(test::ExpectOK{}));

  // Reference actor created with the new inertia properties, same pose, same velocity field.
  RigidActorParams refParams{
      .shape = shape,
      .worldFromLocal = worldFromLocal,
      .colliderType = ColliderType::None,
      .mass = kMass,
      .centerOfMass = kNewCom,
      .momentOfInertia = kMoi,
      .linearVelocity = expectedComVel,
      .angularVelocity = kAngularVel};
  auto* refActor = _scene->CreateRigidActor(refParams, test::ExpectOK{});
  MOCHI_DEFER(_scene->DestroyActor(refActor));

  auto const& reg = GetRegistry();
  auto const e = GetEntity(actor);
  for (int i = 0; i < 10; ++i) {
    _scene->Step(0.01);

    // Passing the same CoM leaves the solver-produced state bit-exact.
    auto const stateBefore = reg.get<CRigidState<TimeStep::Current> const>(e).value;
    auto const vComBefore = reg.get<CRigidVel<TimeStep::Current> const>(e).value.GetVCom();
    auto const rootBeforeSameCom = actor->GetRootTransform();
    actor->SetInertiaProperties(kMass, kNewCom, kMoi, test::ExpectOK{});
    EXPECT_EQ(stateBefore, reg.get<CRigidState<TimeStep::Current> const>(e).value);
    EXPECT_TRUE(
        Equal<3>(vComBefore, reg.get<CRigidVel<TimeStep::Current> const>(e).value.GetVCom()));
    EXPECT_EQ(rootBeforeSameCom, actor->GetRootTransform());
  }

  auto const root = actor->GetRootTransform();
  auto const refRoot = refActor->GetRootTransform();
  EXPECT_NEAR_TOL(refRoot, root, kRoundoffTol);

  auto const stateBeforeMassChange = reg.get<CRigidState<TimeStep::Current> const>(e).value;
  auto const vComBeforeMassChange = reg.get<CRigidVel<TimeStep::Current> const>(e).value.GetVCom();
  actor->SetInertiaProperties(
      2_r * kMass, kNewCom, Real6{3_r, 0.1_r, 0.2_r, 4_r, 0.3_r, 5_r}, test::ExpectOK{});
  EXPECT_EQ(stateBeforeMassChange, reg.get<CRigidState<TimeStep::Current> const>(e).value);
  EXPECT_TRUE(
      Equal<3>(
          vComBeforeMassChange, reg.get<CRigidVel<TimeStep::Current> const>(e).value.GetVCom()));
  EXPECT_EQ(root, actor->GetRootTransform());
}

TEST_F(ActorMassTest, SetInertiaProperties_ComChangeRejectedWithConstraintOrTranslationBC) {
  auto [coords, conn] = test::CreateMinimalTetMeshUnitCube(Real3{0.1_r, 0.2_r, 0.3_r});
  auto shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), test::ExpectOK{});
  RigidActorParams params{.shape = shape, .colliderType = ColliderType::None, .density = 500_r};
  auto* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
  MOCHI_DEFER(_scene->DestroyActor(actor));

  auto constexpr kMass = 5_r;
  auto constexpr kMoi = Real6{2_r, 0.1_r, 0.2_r, 3_r, 0.3_r, 4_r};
  auto constexpr kNewCom = Real3{0.4_r, -0.3_r, 0.2_r};
  Real3 const oldCom = actor->GetRigidCenterOfMassLocal(test::ExpectOK{});

  // A translation BC rejects a CoM change but accepts mass and moment-of-inertia changes.
  int const translationDof = 0;
  real const translationValue =
      actor->GetCenterOfMassTransform(test::ExpectOK{}).GetTranslation()[0];
  actor->AddBoundaryConditionDofsWorld(
      MakeSingletonConstSpan(translationDof),
      MakeSingletonConstSpan(translationValue),
      test::ExpectOK{});
  actor->SetInertiaProperties(kMass, kNewCom, kMoi, test::ExpectNotOK{});
  actor->SetInertiaProperties(kMass, oldCom, kMoi, test::ExpectOK{});

  // Cleared BCs and rotation-only BCs do not restrict the CoM.
  actor->ClearBoundaryConditions();
  int const rotationDofs[] = {3, 4, 5};
  real const rotationValues[] = {0_r, 0_r, 0_r};
  actor->AddBoundaryConditionDofsWorld(rotationDofs, rotationValues, test::ExpectOK{});
  actor->SetInertiaProperties(kMass, kNewCom, kMoi, test::ExpectOK{});

  // A constraint rejects a CoM change but accepts mass and moment-of-inertia changes.
  RigidPivotPositionConstraintParams conParams;
  conParams.actor = actor->GetHandle();
  _scene->CreateRigidPivotPositionConstraint(conParams, test::ExpectOK{});
  actor->SetInertiaProperties(kMass, oldCom, kMoi, test::ExpectNotOK{});
  actor->SetInertiaProperties(2_r * kMass, kNewCom, kMoi, test::ExpectOK{});
}

TEST_F(ActorMassTest, SetInertiaProperties_DensityRescale) {
  auto constexpr kScale = Real3{0.1_r, 0.2_r, 0.3_r};
  auto constexpr kVolume = kScale[0] * kScale[1] * kScale[2];

  auto [coords, conn] = test::CreateMinimalTetMeshUnitCube(kScale);
  auto shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), test::ExpectOK{});
  RigidActorParams params{.shape = shape, .colliderType = ColliderType::None, .density = 500_r};
  auto* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
  MOCHI_DEFER(_scene->DestroyActor(actor));

  // Set mass properties with a custom (non-uniform-density) MoI.
  auto constexpr kSetMass = 5_r;
  auto constexpr kSetCom = Real3{0_r, 0_r, 0_r};
  auto constexpr kSetMoi = Real6{2_r, 0.1_r, 0.2_r, 3_r, 0.3_r, 4_r};
  actor->SetInertiaProperties(kSetMass, kSetCom, kSetMoi, test::ExpectOK{});

  // After SetInertiaProperties, GetDensity returns the bulk density (mass / volume).
  auto constexpr kExpectedDensity = kSetMass / kVolume;
  EXPECT_NEAR_RTOL(kExpectedDensity, actor->GetDensity(test::ExpectOK{}), kApiRelTol);

  // SetDensity should rescale mass and MoI proportionally, leaving CoM unchanged.
  auto constexpr kScaleFactor = 2_r;
  auto constexpr kNewDensity = kExpectedDensity * kScaleFactor;
  actor->SetDensity(kNewDensity, test::ExpectOK{});

  EXPECT_EQ(kNewDensity, actor->GetDensity(test::ExpectOK{}));
  EXPECT_EQ(kSetCom, actor->GetRigidCenterOfMassLocal(test::ExpectOK{}));
  EXPECT_NEAR_RTOL(kSetMass * kScaleFactor, actor->GetMass(test::ExpectOK{}), kApiRelTol);

  auto const moi = actor->GetRigidMomentOfInertiaLocal(test::ExpectOK{});
  EXPECT_NEAR_RTOL(kSetMoi[0] * kScaleFactor, moi[0], kApiRelTol);
  EXPECT_NEAR_RTOL(kSetMoi[1] * kScaleFactor, moi[1], kApiRelTol);
  EXPECT_NEAR_RTOL(kSetMoi[2] * kScaleFactor, moi[2], kApiRelTol);
  EXPECT_NEAR_RTOL(kSetMoi[3] * kScaleFactor, moi[3], kApiRelTol);
  EXPECT_NEAR_RTOL(kSetMoi[4] * kScaleFactor, moi[4], kApiRelTol);
  EXPECT_NEAR_RTOL(kSetMoi[5] * kScaleFactor, moi[5], kApiRelTol);
}

TEST_F(ActorMassTest, MomentOfInertiaValidation) {
  auto constexpr kScale = Real3{0.1_r, 0.2_r, 0.3_r};
  auto [coords, conn] = test::CreateMinimalTetMeshUnitCube(kScale);
  auto shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), test::ExpectOK{});
  RigidActorParams params{.shape = shape, .colliderType = ColliderType::Box, .density = 500_r};
  auto* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
  MOCHI_DEFER(_scene->DestroyActor(actor));

  // Direct setter: finite but physically invalid MOI warns and is still stored.
  auto constexpr kBadMoi = Real6{1_r, 5_r, 0_r, 1_r, 0_r, 1_r};
  {
    test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
    actor->SetInertiaProperties(5_r, Real3{}, kBadMoi, test::ExpectOK{});
  }

  EXPECT_EQ(kBadMoi, actor->GetRigidMomentOfInertiaLocal(test::ExpectOK{}));

  // Direct setter: non-finite MOI is an error.
  actor->SetInertiaProperties(
      5_r, Real3{}, Real6{1_r, 0_r, 0_r, 1_r, 0_r, kInf}, test::ExpectNotOK{});

  // Rigid actor creation: finite but physically invalid MOI warns and is still stored.
  auto constexpr kTriangleBadMoi = Real6{1_r, 0_r, 0_r, 1_r, 0_r, 3_r};
  RigidActorParams invalidParams{
      .shape = shape,
      .colliderType = ColliderType::Box,
      .mass = 5_r,
      .centerOfMass = Real3{},
      .momentOfInertia = kTriangleBadMoi};
  Actor* invalidActor = nullptr;
  {
    test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
    invalidActor = _scene->CreateRigidActor(invalidParams, test::ExpectOK{});
  }
  MOCHI_DEFER(_scene->DestroyActor(invalidActor));
  EXPECT_EQ(kTriangleBadMoi, invalidActor->GetRigidMomentOfInertiaLocal(test::ExpectOK{}));

  // Rigid actor creation: non-finite MOI is an error.
  RigidActorParams nonFiniteParams{
      .shape = shape,
      .colliderType = ColliderType::Box,
      .mass = 5_r,
      .centerOfMass = Real3{},
      .momentOfInertia = Real6{1_r, 0_r, 0_r, 1_r, 0_r, kInf}};
  _scene->CreateRigidActor(nonFiniteParams, test::ExpectNotOK{});
}
