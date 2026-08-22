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

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/mesh_data_utils.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/reflection.h>

#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/src/mochi_simulation.h>
#include <mochi_physics/src/mochi_soft_rom_components.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

using namespace mochi;

// The query test meshes are not shipped externally.
#if MOCHI_USE_HDF5 && MOCHI_INTERNAL
#define MOCHI_HDF5_AND_INTERNAL 1
#else
#define MOCHI_HDF5_AND_INTERNAL 0

#endif
using namespace mochi::experimental;

static constexpr std::string_view kDefaultLayer = "QueryTestDefaultLayer";

#if MOCHI_USE_HDF5 && MOCHI_ENABLE_ROM_ACTORS
#define MOCHI_TEST_ROM_HDF5 1
#else
#define MOCHI_TEST_ROM_HDF5 0
#endif

namespace {
// Enum for specifying a 90-degree rotation axis for cubes in tests
enum class RotationAxis90 {
  None, // No rotation (identity)
  X, // 90 degrees about X axis
  Y, // 90 degrees about Y axis
  Z // 90 degrees about Z axis
};
} // namespace

MOCHI_ENUM_BEGIN(RotationAxis90);
MOCHI_ENUM_ITEM(None);
MOCHI_ENUM_ITEM(X);
MOCHI_ENUM_ITEM(Y);
MOCHI_ENUM_ITEM(Z);
MOCHI_ENUM_END();

// Returns a quaternion representing a 90-degree rotation about the specified axis
static Quaternion GetRotation90(RotationAxis90 axis) {
  switch (axis) {
    case RotationAxis90::X:
      return Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, kPI / 2_r);
    case RotationAxis90::Y:
      return Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, kPI / 2_r);
    case RotationAxis90::Z:
      return Quaternion::FromAxisAngle(Real3{0_r, 0_r, 1_r}, kPI / 2_r);
    case RotationAxis90::None:
    default:
      return Quaternion::Identity();
  }
}

// Computes a TransformRT that rotates a cube about the specified axis, while keeping the cube's
// center at the same world position. The cube has one corner at the local origin and extends to
// (scale, scale, scale).
static TransformRT
GetCubeRotatedTransform(RotationAxis90 rotationAxis, real scale, Real3 position) {
  Quaternion rotation = GetRotation90(rotationAxis);
  // The cube's center in local space is at (scale/2, scale/2, scale/2).
  // After rotation about the origin, the center moves to rotation * center.
  // We need to correct the translation so the center ends up at position + center.
  Real3 center = Real3{scale / 2_r, scale / 2_r, scale / 2_r};
  // Rotate the center using quaternion * Real3
  Real3 rotatedCenter = rotation * center;
  // The corrected position shifts by the difference between original and rotated center
  Real3 correctedPosition = position + center - rotatedCenter;
  return TransformRT{rotation, correctedPosition};
}

// Test fixture which creates an empty mochi::Scene
class ActorQueryTest : public test::MochiSceneTestBase {
 public:
  void SetUp() override {
    // Call down
    test::MochiSceneTestBase::SetUp();

    // No gravity unless a specific test enables it
    _scene->SetGravity(Real3{});
  }
};

// Test fixture which creates an empty mochi::AsyncScene
class ActorQueryTestAsync : public test::MochiAsyncSceneTestBase {
 public:
  void SetUp() override {
    // Call down
    test::MochiAsyncSceneTestBase::SetUp();

    // No gravity unless a specific test enables it
    _asyncScene->QueueCommand([](Scene* scene) { scene->SetGravity(Real3{}); });
  }
};

/**
  Create a solid unit cube with one corner at (0,0,0)

        2 ------- 3    8 coordinates (all on surface)
      / |       / |    5 tets (4 have surface faces)
     /  |      /  |
    6 ------- 7   |
    |   0 ----|-- 1
    |  /      |  /
    | /       | /
    4 ------- 5
*/
static ShapeHandle CreateCubeShape(Scene* scene, real scale) {
  auto&& [unitCubeCoordinates, unitCubeConnectivity] = test::CreateMinimalTetMeshUnitCube();
  for (auto& pt : unitCubeCoordinates) {
    pt *= scale;
  }
  ShapeHandle shape = scene->GetContext()->CreateTetMeshShape(
      Flatten(MakeSpan(unitCubeCoordinates)),
      Flatten(MakeSpan(unitCubeConnectivity)),
      ErrorAssert{});
  EXPECT_EQ(true, shape.IsValid());
  return shape;
}

static Actor* CreateSoftCube(
    Scene* scene,
    real scale,
    Real3 position,
    RotationAxis90 rotationAxis = RotationAxis90::None) {
  auto shape = CreateCubeShape(scene, scale);
  SoftActorParams caparams;
  caparams.name = "SoftCube";
  caparams.shape = shape;
  caparams.worldFromLocal = GetCubeRotatedTransform(rotationAxis, scale, position);
  // Hard code contact and material params in case the default change in a way that would affect
  // this test.
  caparams.contact.penaltyCoefficient = 1e8_r;
  caparams.contact.penaltyThresholdDefault = 0.005_r;
  caparams.material.type = SoftMaterialType::NeoHookean;
  caparams.material.neoHookean.youngsModulus = 1e5_r;
  caparams.material.neoHookean.poissonRatio = 0.45_r;
  caparams.material.density = 1000_r;
  caparams.layer = kDefaultLayer;
  caparams.contact.viscousFrictionCoefficient = 2_r;
  return scene->CreateSoftActor(caparams, ErrorAssert{});
}

static Actor* CreateRomDuck(Context* context, Scene* scene, real scale, Real3 position) {
  constexpr char const* kDuckMesh = "duck/duck_1899.mochi.h5";
  auto duckShape = context->LoadShapeFromFile(
      test::GetAssetPath(kDuckMesh),
      Real3{scale, scale, scale},
      TransformRT::Identity(),
      ErrorAssert{});
  RomParams rom;
  rom.source = "rigidsoft_10";
  SampleMeshInitRandomSampling hrStrategy;
  hrStrategy.stepSizeForBoundaryElementsSelection = 5;
  hrStrategy.stepSizeForInteriorElementsSelection = 5;
  rom.hyperReduction = HyperReductionParams{hrStrategy, {}};
  SoftActorParams caparams;
  caparams.name = "RomDuck";
  caparams.shape = duckShape;
  caparams.worldFromLocal = TransformRT{position};
  // Hard code contact and material params in case the default change in a way that would affect
  // this test.
  caparams.contact.penaltyCoefficient = 1e8_r;
  caparams.contact.penaltyThresholdDefault = 0.005_r;
  caparams.material.type = SoftMaterialType::NeoHookean;
  caparams.material.neoHookean.youngsModulus = 1e5_r;
  caparams.material.neoHookean.poissonRatio = 0.45_r;
  caparams.material.density = 1000_r;
  caparams.layer = kDefaultLayer;
  caparams.contact.viscousFrictionCoefficient = 2_r;

  ExperimentalSoftActorParams experimentalParams;
  experimentalParams.rom = rom;

  return CreateSoftActor(scene, caparams, experimentalParams, ErrorAssert{});
}

static Actor* CreateRigidCube(
    Scene* scene,
    real scale,
    Real3 position,
    bool isStatic = false,
    ColliderType colliderType = ColliderType::Box,
    RotationAxis90 rotationAxis = RotationAxis90::None) {
  auto shape = CreateCubeShape(scene, scale);
  RigidActorParams caparams;
  caparams.name = "RigidCube";
  caparams.shape = shape;
  caparams.worldFromLocal = GetCubeRotatedTransform(rotationAxis, scale, position);
  caparams.contact.penaltyCoefficient = 1e8_r;
  caparams.contact.penaltyThresholdDefault = 0.005_r;
  caparams.density = 1000_r;
  caparams.isStatic = isStatic;
  caparams.layer = kDefaultLayer;
  caparams.colliderType = colliderType;
  caparams.sdf.resolutionMode = GridSdfResolutionMode::LargestAxis;
  caparams.sdf.resolutionDelta = {0.1_r, 0.1_r, 0.1_r};
  return scene->CreateRigidActor(caparams, ErrorAssert{});
}

static Actor* CreateStaticSphere(Scene* scene, real radius, Real3 position) {
  auto* context = scene->GetContext();
  auto shape = context->CreateSphereShape(Real3{}, radius, ErrorAssert{});

  RigidActorParams caparams;
  caparams.name = "RigidSphere";
  caparams.shape = shape;
  caparams.worldFromLocal = TransformRT{position};
  caparams.contact.penaltyCoefficient = 1e8_r;
  caparams.contact.penaltyThresholdDefault = 0.005_r;
  caparams.density = 1000_r;
  caparams.isStatic = true;
  caparams.layer = kDefaultLayer;
  caparams.colliderType = ColliderType::Sphere;
  caparams.contact.viscousFrictionCoefficient = 2_r;
  return scene->CreateRigidActor(caparams, ErrorAssert{});
}

static Actor* CreateStaticGroundPlane(Scene* scene, real height) {
  auto* context = scene->GetContext();
  auto shape = context->CreatePlaneShape(Real3{0_r, 1_r, 0_r}, 0_r, ErrorAssert{});
  RigidActorParams caparams;
  caparams.name = "Ground";
  caparams.isStatic = true;
  caparams.shape = shape;
  caparams.worldFromLocal.SetTranslation(Real3{0_r, height, 0_r});
  caparams.layer = kDefaultLayer;
  caparams.contact.penaltyCoefficient = 1e8_r;
  caparams.contact.penaltyThresholdDefault = 0.005_r;
  caparams.contact.viscousFrictionCoefficient = 2_r;
  return scene->CreateRigidActor(caparams, ErrorAssert{});
}

static ContactPoint MirrorContactPoint(ContactPoint const& cp) {
  ContactPoint mirrored;
  mirrored.actorA = cp.actorB;
  mirrored.actorB = cp.actorA;
  mirrored.distance = cp.distance;
  mirrored.posB = cp.posA;
  mirrored.posA = cp.posB;
  mirrored.normal = -cp.normal;
  mirrored.force = -cp.force;
  mirrored.pointVelocityA = {};
  mirrored.pointVelocityB = {};
  return mirrored;
}

static std::optional<TriangularMesh> GetActorTriangularMeshOptional(
    Actor* actor,
    Span<ContactPoint const> contacts) {
  if (!contacts) {
    return std::nullopt;
  }

  auto transform = actor->GetRootTransform();
  auto positionsLocal =
      Unflatten<Real3 const>(actor->GetSurfaceMeshNodePositionsLocal(ErrorAssert{}));
  DynamicArray<Real3> positionsWorld(positionsLocal.size());
  ArrayTransformPoints(MakeSpan(positionsWorld), positionsLocal, transform);
  TriangularMesh actorMesh(
      positionsWorld, Unflatten<Int3 const>(actor->GetSurfaceMesh().connectivity));
  return actorMesh;
}

static void
TestParametricInfo(Actor* actor, TriangularMesh const& triMesh, ContactPoint const& cp) {
  // Discard if the actor is actorB (collider) in the contact point
  if (actor->GetHandle() == cp.actorB) {
    return;
  }

  // Verify that we can reconstruct the contact point using the parametric coordinates
  auto faces = triMesh.GetElementConnectivity();
  auto coords = triMesh.GetNodeCoordinates();
  int faceIdx = cp.elementIndex;
  EXPECT_TRUE(faceIdx >= 0 && faceIdx < faces.size());
  Real3 pc = cp.parametricCoords;
  Int3 tri = faces[faceIdx];
  Real3 v[3] = {coords[tri[0]], coords[tri[1]], coords[tri[2]]};
  Real3 ptOnTriangle = v[0] * pc[0] + v[1] * pc[1] + v[2] * pc[2];
  EXPECT_NEAR_EQ(cp.posA, ptOnTriangle);
}

static void ExpectContactPointsOnGround(
    bool expectSuccess,
    Actor* actor,
    Actor* ground,
    real groundHeight,
    Real3 const& gravityAccel,
    real positionTolerance = 1e-4_r) {
  Error error;
  Span<ContactPoint const> contacts = actor->GetContactPointsWorld(error);
  std::optional<TriangularMesh> actorMesh = GetActorTriangularMeshOptional(actor, contacts);
  if (expectSuccess) {
    // Except success
    EXPECT_OK(error);
    EXPECT_NE(0, contacts.size());

    // This test does not presume to know how many contact point will be generated.
    Real3 totalForce = {};
    for (auto cp : contacts) {
      TestParametricInfo(actor, *actorMesh, cp);

      // This test does not presume to know the order in which the actors will be listed
      // within the ContactPoint structure. Mirror the contact if necessary so that our actor is
      // "actorA".
      if (actor->GetHandle() != cp.actorA) {
        cp = MirrorContactPoint(cp);
      }

      EXPECT_EQ(actor->GetHandle(), cp.actorA);
      EXPECT_EQ(ground->GetHandle(), cp.actorB);

      // Bodies should not be overlapping very much
      EXPECT_LE(-positionTolerance, cp.distance);

      // The reported position should be on the ground
      EXPECT_NEAR(groundHeight, cp.posB[1], positionTolerance);

      // The normal should point upward (away from ground)
      EXPECT_NEAR(1_r, Norm(cp.normal), 1e-6_r); // Unit length
      EXPECT_NEAR(1_r, cp.normal[1], 1e-5_r); // All in the +Y direction

      // Adding (distance * normal) should give us a point inside the actor's Aabb
      Aabb actorBounds = actor->GetAabbWorld(ErrorAssert{});
      actorBounds = ExpandShape(actorBounds, 1e-5_r); // pad for floating point error
      Real3 pt = cp.posB + (cp.distance * cp.normal);
      EXPECT_NEAR(0_r, Norm(pt - cp.posA), 1e-6_r);
      EXPECT_TRUE(ContainsPoint(actorBounds, pt));

      // Velocity of the actors should be low (coming to rest)
      EXPECT_GT(1e-3_r, Norm(cp.pointVelocityA));
      EXPECT_GT(1e-3_r, Norm(cp.pointVelocityB));

      // Force should point upward, away from the ground
      EXPECT_LT(0_r, cp.force[1]);
      totalForce += cp.force;
    }

    // The total upward force from contact, and the total downward force from gravity
    // should be equal and opposite.
    Real3 expectedTotalForce = -actor->GetMass(test::ExpectOK{}) * gravityAccel;
    real expectedTotalForceMag = Norm(expectedTotalForce);
    real reportedTotalForceMag = Norm(totalForce);
    real forceMagnitudeErrorPercent = 1_r - (reportedTotalForceMag / expectedTotalForceMag);
    real forceDirectionErrorDegrees =
        kRadiansPerDegree * std::acos(Dot(Normalize(expectedTotalForce), Normalize(totalForce)));

    // Expect force magnitude to be within 6% of our expectation.
    real const forceTolerance = 0.06_r;
    EXPECT_NEAR(forceMagnitudeErrorPercent, 0_r, forceTolerance);

    // Expect force direction to be within 0.01 degrees.
    EXPECT_NEAR(forceDirectionErrorDegrees, 0_r, 0.01_r);
  } else {
    EXPECT_NOT_OK(error);
    EXPECT_EQ(0, contacts.size());
  }
}

static QueryHandle
RegisterQuery(QueryType queryType, Actor* actor, bool computeImmediately = false) {
  return computeImmediately ? actor->RegisterQueryAndCompute(queryType, ErrorAssert{})
                            : actor->RegisterQuery(queryType, ErrorAssert{});
}

static QueryHandle
RegisterAllNecessaryQueries(QueryType queryType, Actor* actor, bool computeImmediately = false) {
  // These additional query should NOT be required for the specified `queryType` to function
  // correctly. However, they are required for the test code that runs in this file.
  if (queryType == QueryType::ContactPoints) {
    RegisterQuery(QueryType::SurfaceNodePositions, actor, computeImmediately);
  }
  if (queryType == QueryType::NodeContactForces) {
    RegisterQuery(QueryType::ContactPoints, actor, computeImmediately);
  }

  return RegisterQuery(queryType, actor, computeImmediately);
}

static void TestQuery(
    Scene* scene,
    Actor* actorA,
    Actor* actorB,
    QueryType queryType,
    std::function<void(bool expectSuccess, Actor* actor)> const& expectResults) {
  Actor const* nullActor = nullptr;
  EXPECT_NE(nullActor, actorA);
  EXPECT_NE(nullActor, actorB);

  // This test assumes that the query is not initially available
  expectResults(false, actorA);
  expectResults(false, actorB);

  // Request the query for actorA
  QueryHandle qA1 = RegisterAllNecessaryQueries(queryType, actorA);
  EXPECT_EQ(true, qA1.IsValid());

  // Results should become available after the next simulation step
  bool expectResultsAlready = actorA->IsStatic();
  expectResults(expectResultsAlready, actorA); // Most actors process queries during the step
  expectResults(false, actorB); // We didn't query actorB
  scene->Step(0.001);
  expectResults(true, actorA); // Ready
  expectResults(false, actorB); // We didn't query actorB

  // Make a redundant request and step
  QueryHandle qA2 = RegisterAllNecessaryQueries(queryType, actorA);
  EXPECT_EQ(true, qA2.IsValid());
  scene->Step(0.001);
  expectResults(true, actorA);
  expectResults(false, actorB);

  // Now request a query on actorB. This time, tell it to compute immediately (if supported):
  bool computeImmediately = false;
  switch (queryType) {
    case QueryType::NodePositions:
    case QueryType::SurfaceNodePositions:
    case QueryType::SurfaceNodeNormals:
    case QueryType::VisualNodePositions:
    case QueryType::VisualNodeNormals:
    case QueryType::ElasticEnergy:
      computeImmediately = true;
      break;
    default:
      break;
  }
  QueryHandle qB1 = RegisterAllNecessaryQueries(queryType, actorB, computeImmediately);
  EXPECT_EQ(true, qB1.IsValid());
  expectResults(true, actorA);
  expectResults(computeImmediately, actorB); // Valid before step if computeImmediately
  scene->Step(0.001);
  expectResults(true, actorA);
  expectResults(true, actorB); // And after

  // Cancel the first request and step
  actorA->CancelQuery(qA1);
  scene->Step(0.001);
  expectResults(true, actorA); // Still valid because of qA1
  expectResults(true, actorB); // Still valid because of qB1

  // Cancel the last request on actorA
  actorA->CancelQuery(qA2);
  scene->Step(0.001);
  expectResults(false, actorA); // all gone
  expectResults(true, actorB);

  // Try to cancel qA2 again (does nothing)
  actorA->CancelQuery(qA2);
  scene->Step(0.001);
  expectResults(false, actorA);
  expectResults(true, actorB);

  // Try to cancel qB1 by passing it to actorA (does nothing)
  actorA->CancelQuery(qB1);

  // Finally, cancel qB1 correctly
  actorB->CancelQuery(qB1);
  scene->Step(0.001);
  expectResults(false, actorA);
  expectResults(false, actorB); // all gone
}

// Like TestQuery, but it uses the AsyncScene API
static void TestQueryAsync(
    AsyncScene* asyncScene,
    ActorHandle actorA,
    ActorHandle actorB,
    QueryType queryType,
    std::function<void(bool expectSuccess, Actor const* actor)> const& expectResults) {
  EXPECT_EQ(true, actorA.IsValid());
  EXPECT_EQ(true, actorB.IsValid());

  // Request the query for actorA
  QueryHandle qA1 = asyncScene->RegisterActorQuery(actorA, queryType);
  EXPECT_EQ(true, qA1.IsValid());

  // Results should become available after the next simulation step
  asyncScene->QueueCommand([&](Scene* scene) {
    scene->Step(0.001); // Make sure at least one step has happened
    expectResults(true, scene->GetActor(actorA)); // ready now
    expectResults(false, scene->GetActor(actorB));
  });

  // Make a redundant request and step
  QueryHandle qA2 = asyncScene->RegisterActorQuery(actorA, queryType);
  EXPECT_EQ(true, qA2.IsValid());
  asyncScene->QueueCommand([&](Scene* scene) {
    scene->Step(0.001);
    expectResults(true, scene->GetActor(actorA)); // no change
    expectResults(false, scene->GetActor(actorB));
  });

  // Now request a query on actorB
  QueryHandle qB1 = asyncScene->RegisterActorQuery(actorB, queryType);
  EXPECT_EQ(true, qB1.IsValid());
  asyncScene->QueueCommand([&](Scene* scene) {
    scene->Step(0.001);
    expectResults(true, scene->GetActor(actorA));
    expectResults(true, scene->GetActor(actorB)); // ready now
  });

  // Cancel the first request and step
  asyncScene->CancelActorQuery(actorA, qA1);
  asyncScene->QueueCommand([&](Scene* scene) {
    scene->Step(0.001);
    expectResults(true, scene->GetActor(actorA)); // Still valid because of qA1
    expectResults(true, scene->GetActor(actorB)); // Still valid because of qB1
  });

  // Cancel the last request on actorA
  asyncScene->CancelActorQuery(actorA, qA2);
  asyncScene->QueueCommand([&](Scene* scene) {
    scene->Step(0.001);
    expectResults(false, scene->GetActor(actorA)); // all gone
    expectResults(true, scene->GetActor(actorB));
  });

  // Try to cancel qA2 again (does nothing)
  asyncScene->CancelActorQuery(actorA, qA2);
  asyncScene->QueueCommand([&](Scene* scene) {
    scene->Step(0.001);
    expectResults(false, scene->GetActor(actorA));
    expectResults(true, scene->GetActor(actorB));
  });

  // Try to cancel qB1 by passing it to actorA (does nothing)
  asyncScene->CancelActorQuery(actorA, qB1);
  asyncScene->QueueCommand([&](Scene* scene) {
    scene->Step(0.001);
    expectResults(false, scene->GetActor(actorA));
    expectResults(true, scene->GetActor(actorB));
  });

  // Finally, cancel qB1 correctly
  asyncScene->CancelActorQuery(actorB, qB1);
  asyncScene->QueueCommand([&](Scene* scene) {
    scene->Step(0.001);
    expectResults(false, scene->GetActor(actorA));
    expectResults(false, scene->GetActor(actorB)); // all gone
  });

  // Wait for the above commands complete in order
  asyncScene->WaitForQueuedCommands();
}

static void
ExpectCubeNodePositions(bool expectSuccess, Actor const* actor, Real3 expectedOffset = {}) {
  // Called by TestQuery to verify that the query results are available.
  Error error;
  int num_nodes = actor->GetMesh().GetNumNodes();
  EXPECT_EQ(8, num_nodes);
  auto positions = Unflatten<Real3 const>(actor->GetNodePositionsLocal(error));
  if (expectSuccess) {
    EXPECT_OK(error);
    auto [refCoordinates, refConnectivity] = test::CreateMinimalTetMeshUnitCube();
    // Adjust the scale of the ref coords to account for different sized cubes used in this test
    real const scale = 2_r * actor->GetAabbLocal(ErrorAssert{}).GetHalfExtents()[0];
    EXPECT_EQ(refCoordinates.size(), positions.size());
    for (int i = 0; i < actor->GetMesh().GetNumNodes(); ++i) {
      EXPECT_NEAR_EQ(refCoordinates[i] * scale + expectedOffset, positions[i]);
    }
  } else {
    EXPECT_NOT_OK(error);
  }
}

static void ExpectCubeNodePositionsRigid(bool expectSuccess, Actor const* actor) {
  // Called by TestQuery to verify that the query results are available.
  Error error;
  int num_nodes = actor->GetSurfaceMesh().GetNumNodes();
  EXPECT_EQ(8, num_nodes);
  auto positions = Unflatten<Real3 const>(actor->GetSurfaceMeshNodePositionsLocal(error));
  if (expectSuccess) {
    EXPECT_OK(error);
    auto [refCoordinates, refConnectivity] = test::CreateMinimalTriMeshUnitCube();
    // Adjust the scale of the ref coords to account for different sized cubes used in this test
    real const scale = 2_r * actor->GetAabbLocal(ErrorAssert{}).GetHalfExtents()[0];
    EXPECT_EQ(refCoordinates.size(), positions.size());
    for (int i = 0; i < actor->GetSurfaceMesh().GetNumNodes(); ++i) {
      EXPECT_NEAR_EQ(positions[i], refCoordinates[i] * scale);
    }
  } else {
    EXPECT_NOT_OK(error);
  }
}

TEST_F(ActorQueryTest, NodePositions_Soft) {
  // Test Scene::RegisterQuery (synchronous API)
  TestQuery(
      _scene,
      CreateSoftCube(_scene, 1_r, Real3{-10_r, 0_r, 0_r}),
      CreateSoftCube(_scene, 2_r, Real3{10_r, 0_r, 0_r}),
      QueryType::NodePositions,
      [](bool expectResults, Actor const* a) { return ExpectCubeNodePositions(expectResults, a); });

  // Stepping the scene with zero delta time should also compute the query
  auto* actor = CreateSoftCube(_scene, 0.5_r, Real3{5_r, 0_r, 0_r});
  actor->RegisterQuery(QueryType::NodePositions, test::ExpectOK{});
  ExpectCubeNodePositions(false, actor);
  _scene->Step(0.0);
  ExpectCubeNodePositions(true, actor, Real3{} /*expectedOffset*/);

  // If we modify the actor, then we can step the scene with zero delta time again to refresh the
  // query.
  DynamicArray<real> displacements(actor->GetDisplacements(test::ExpectOK{}));
  for (auto& d : displacements) {
    d += 1_r;
  }
  actor->SetDisplacements(displacements, test::ExpectOK{});
  _scene->Step(0.0);
  ExpectCubeNodePositions(true, actor, Real3{1_r, 1_r, 1_r} /*expectedOffset*/);
}

TEST_F(ActorQueryTest, NodePositions_Rigid) {
  // Test Scene::RegisterQuery (synchronous API)
  TestQuery(
      _scene,
      CreateRigidCube(_scene, 1_r, Real3{-10_r, 0_r, 0_r}),
      CreateRigidCube(_scene, 2_r, Real3{10_r, 0_r, 0_r}),
      QueryType::SurfaceNodePositions,
      [](bool expectResults, Actor const* a) {
        return ExpectCubeNodePositionsRigid(expectResults, a);
      });
}

TEST_F(ActorQueryTest, NodePositions_RigidStatic) {
  // Test Scene::RegisterQuery (synchronous API)
  TestQuery(
      _scene,
      CreateRigidCube(_scene, 1_r, Real3{-10_r, 0_r, 0_r}, true),
      CreateRigidCube(_scene, 2_r, Real3{10_r, 0_r, 0_r}, true),
      QueryType::SurfaceNodePositions,
      [](bool expectResults, Actor const* a) {
        return ExpectCubeNodePositionsRigid(expectResults, a);
      });
}

TEST_F(ActorQueryTestAsync, NodePositions_Soft) {
  // This time, use AsyncScene::RegisterActorQuery (asynchronous API). We don't bother testing all
  // the query types this way because they should all be identical with regard to the deferred
  // registration mechanism.
  ActorHandle actorA, actorB;
  _asyncScene->QueueCommand([&](Scene* scene) {
    actorA = CreateSoftCube(scene, 1_r, Real3{-10_r, 0_r, 0_r})->GetHandle();
    actorB = CreateSoftCube(scene, 2_r, Real3{10_r, 0_r, 0_r})->GetHandle();
  });
  _asyncScene->WaitForQueuedCommands();

  TestQueryAsync(
      _asyncScene,
      actorA,
      actorB,
      QueryType::NodePositions,
      [](bool expectResults, Actor const* a) { return ExpectCubeNodePositions(expectResults, a); });
}

static void
ExpectCubeNodeNormals(bool expectSuccess, Actor const* actor, bool isRigid, Context* context) {
  // Called by TestQuery to verify that the query results are available.
  Error error;
  EXPECT_EQ(8, isRigid ? actor->GetSurfaceMesh().GetNumNodes() : actor->GetMesh().GetNumNodes());
  auto normals = Unflatten<Real3 const>(actor->GetSurfaceMeshNodeNormalsLocal(error));
  if (expectSuccess) {
    EXPECT_OK(error);
    auto [refCoordinates, refConnectivity] = test::CreateMinimalTetMeshUnitCube();
    EXPECT_EQ(refCoordinates.size(), normals.size());

    // Expect all surface node normals to point away from the center
    Real3 center = actor->GetAabbLocal(ErrorAssert{}).GetCenter();
    auto shape = actor->GetReferenceShape(ErrorAssert{});
    auto mesh = context->GetShapeMesh(shape, ErrorAssert{});
    Span<Real3 const> referencePositions = Unflatten<Real3 const>(mesh.coordinates);
    // For the unit cube, all nodes are on the surface
    for (int i = 0; i < mesh.GetNumNodes(); ++i) {
      Real3 norm = normals[i];
      EXPECT_NEAR_EQ(1_r, Norm(norm));
      Real3 fromCenter = Normalize(referencePositions[i] - center);
      EXPECT_NEAR_EQ(1_r, Dot(norm, fromCenter));
    }
  } else {
    EXPECT_NOT_OK(error);
  }
}

TEST_F(ActorQueryTest, NodeNormals_Soft) {
  TestQuery(
      _scene,
      CreateSoftCube(_scene, 1_r, Real3{-10_r, 0_r, 0_r}),
      CreateSoftCube(_scene, 2_r, Real3{10_r, 0_r, 0_r}),
      QueryType::SurfaceNodeNormals,
      [ctx = _scene->GetContext()](bool expectResults, Actor const* a) {
        return ExpectCubeNodeNormals(expectResults, a, false, ctx);
      });
}

TEST_F(ActorQueryTest, NodeNormals_Rigid) {
  TestQuery(
      _scene,
      CreateRigidCube(_scene, 1_r, Real3{-10_r, 0_r, 0_r}),
      CreateRigidCube(_scene, 2_r, Real3{10_r, 0_r, 0_r}),
      QueryType::SurfaceNodeNormals,
      [ctx = _scene->GetContext()](bool expectResults, Actor const* a) {
        return ExpectCubeNodeNormals(expectResults, a, true, ctx);
      });
}

static void TestQueryGroundContact(Scene* scene, Actor* bodyA, Actor* bodyB, Actor* ground) {
  // Enable gravity and let the cubes fall onto the ground plane.
  // Step many times to make sure it is fully settled.
  Real3 const gravity = Real3{0_r, -10_r, 0_r};
  scene->SetGravity(gravity);
  // The at-rest checks below rely on backward Euler's damping to settle the bodies.
  test::SetSceneIntegrationMethod(scene, IntegrationMethod::BackwardEuler);
  for (int i = 0; i < 75; ++i) {
    scene->Step(0.01);
  }

  // Check the Aabbs to make sure they are both on the ground (more or less)
  // TODO: An actor should come to rest on the exact surface, not float above it.
  constexpr real kCloseEnough = 0.003_r;
  Aabb aabbA = bodyA->GetAabbWorld(ErrorAssert{});
  Aabb aabbB = bodyB->GetAabbWorld(ErrorAssert{});
  real groundHeight = ground->GetRootTransform().GetTranslation()[1];
  EXPECT_NEAR(groundHeight, aabbA.GetMin()[1], kCloseEnough);
  EXPECT_NEAR(groundHeight, aabbB.GetMin()[1], kCloseEnough);

  // Now we can expect to get a non-empty collection of contact points when the query is enabled
  TestQuery(scene, bodyA, bodyB, QueryType::ContactPoints, [&](bool expectSuccess, Actor* a) {
    ExpectContactPointsOnGround(expectSuccess, a, ground, groundHeight, gravity);
  });

  // Make sure the query is enabled for bodyA
  bodyA->RegisterQuery(QueryType::SurfaceNodePositions, ErrorAssert{});
  QueryHandle queryA = bodyA->RegisterQuery(QueryType::ContactPoints, ErrorAssert{});
  scene->Step(0.01);
  ExpectContactPointsOnGround(true, bodyA, ground, groundHeight, gravity);

  // Drop the ground plane so that the bodies must fall again
  groundHeight -= 0.1_r;
  ground->SetRootTransform(TransformRT{Real3{0_r, groundHeight, 0_r}}, test::ExpectOK{});

  // Expect no contact, even though the query is still enabled.
  scene->Step(0.01);
  Span<ContactPoint const> queryPoints = bodyA->GetContactPointsWorld(ErrorAssert{});
  EXPECT_EQ(0, queryPoints.size());

  // Let the bodies fall and settle again
  for (int i = 0; i < 30; ++i) {
    scene->Step(0.01);
  }

  // Now bodyA should be in contact at the new groundHeight
  ExpectContactPointsOnGround(true, bodyA, ground, groundHeight, gravity);

  // Disable the query
  bodyA->CancelQuery(queryA);
  scene->Step(0.01);

  // Call TestQuery again, just for good measure
  TestQuery(scene, bodyA, bodyB, QueryType::ContactPoints, [&](bool expectSuccess, Actor* a) {
    ExpectContactPointsOnGround(expectSuccess, a, ground, groundHeight, gravity);
  });
}

TEST_F(ActorQueryTest, ContactPoints_Soft_on_StaticRigid) {
  // Create two soft cubes that will fall onto the ground plane
  //
  //         -------
  //         |     | soft cube
  //         |     |
  //   ------=======------ static rigid ground
  //
  Actor* bodyA = CreateSoftCube(_scene, 0.1_r, Real3{-1_r, 0.001_r, 0_r});
  Actor* bodyB = CreateSoftCube(_scene, 0.11_r, Real3{1_r, 0.001_r, 0_r});
  Actor* ground = CreateStaticGroundPlane(_scene, -1);
  TestQueryGroundContact(_scene, bodyA, bodyB, ground);
}

TEST_F(ActorQueryTest, ContactPoints_DynamicRigid_on_StaticRigid) {
  // Create two rigid cubes that will fall onto the ground plane
  //
  //         -------
  //         |     | rigid cube
  //         |     |
  //   ------=======------ static rigid ground
  //
  Actor* bodyA = CreateRigidCube(_scene, 1_r, Real3{-10_r, 0_r, 0_r});
  Actor* bodyB = CreateRigidCube(_scene, 2_r, Real3{10_r, 0_r, 0_r});
  Actor* ground = CreateStaticGroundPlane(_scene, -1);
  TestQueryGroundContact(_scene, bodyA, bodyB, ground);
}

static void TestQueryStackedContact(
    Scene* scene,
    Actor* topBody,
    Actor* bottomBody,
    Actor* ground,
    real groundHeight,
    real boundaryHeight) {
  // Enable gravity and let the actors fall and settle
  Real3 const gravity = Real3{0_r, -10_r, 0_r};
  scene->SetGravity(gravity);
  // The velocity-near-zero and at-rest checks below rely on backward Euler's damping.
  test::SetSceneIntegrationMethod(scene, IntegrationMethod::BackwardEuler);
  for (int i = 0; i < 40; ++i) {
    scene->Step(0.05);
  }

  // Check the Aabbs. The topBody should be resting above the boundaryHeight.
  constexpr real kPositionTolerance = 0.005_r; // Wiggle room in case it wiggles around some
  Aabb aabbTop = topBody->GetAabbWorld(ErrorAssert{});
  Aabb aabbBottom = bottomBody->GetAabbWorld(ErrorAssert{});
  EXPECT_NEAR(boundaryHeight, aabbTop.GetMin()[1], kPositionTolerance);
  EXPECT_NEAR(boundaryHeight, aabbBottom.GetMax()[1], kPositionTolerance);

  // Request contact data from both actors
  topBody->RegisterQuery(QueryType::SurfaceNodePositions, ErrorAssert{});
  bottomBody->RegisterQuery(QueryType::SurfaceNodePositions, ErrorAssert{});
  QueryHandle queryTop = topBody->RegisterQuery(QueryType::ContactPoints, ErrorAssert{});
  QueryHandle queryBottom = bottomBody->RegisterQuery(QueryType::ContactPoints, ErrorAssert{});
  EXPECT_TRUE(queryTop.IsValid());
  EXPECT_TRUE(queryBottom.IsValid());
  scene->Step(0.01);

  // topBody should be contacting bottomBody as if it were the ground.
  ExpectContactPointsOnGround(
      true, topBody, bottomBody, boundaryHeight, gravity, kPositionTolerance);

  // bottomBody should be contacting both the topBody and the ground.
  {
    Span<ContactPoint const> contacts = bottomBody->GetContactPointsWorld(ErrorAssert{});
    std::optional<TriangularMesh> bottomBodyMesh =
        GetActorTriangularMeshOptional(bottomBody, contacts);
    EXPECT_LE(4, contacts.size()); // Expected at least 4 on top and 4 on bottom
    for (auto cp : contacts) {
      TestParametricInfo(bottomBody, *bottomBodyMesh, cp);

      // Mirror the contact if necessary so that bottomBody is actorA.
      if (cp.actorB == bottomBody->GetHandle()) {
        cp = MirrorContactPoint(cp);
      }
      EXPECT_EQ(cp.actorA, bottomBody->GetHandle());
      EXPECT_NEAR(1_r, Norm(cp.normal), 1e-6_r); // unit length
      EXPECT_NEAR(0_r, Norm(cp.pointVelocityA), 5e-4_r); // not moving much
      EXPECT_NEAR(0_r, Norm(cp.pointVelocityB), 5e-4_r); // not moving much
      if (cp.actorB == topBody->GetHandle()) {
        // The contacts from the top body should point down from the boundary
        EXPECT_NEAR(boundaryHeight, cp.posB[1], kPositionTolerance);
        EXPECT_NEAR(-1_r, cp.normal[1], 1e-5_r);
      } else if (cp.actorB == ground->GetHandle()) {
        // The contacts from the ground should point upward
        EXPECT_NEAR(groundHeight, cp.posB[1], kPositionTolerance);
        EXPECT_NEAR(1_r, cp.normal[1], 1e-5_r);
      } else {
        FAIL() << "Contacting an unknown actor";
      }
    }
  }
}

TEST_F(ActorQueryTest, ContactPoints_DynamicRigid_on_DynamicRigid) {
  // Create a dynamic rigid cube that will fall onto the ground plane.
  // Create a smaller dynamic rigid cube that will fall onto the dynamic rigid cube.
  //
  //           -----
  //           |   | rigid cube
  //           |   |
  //         --=====-- boundary
  //         |       |
  //         |       | rigid cube
  //         |       |
  //   ------=========------ static rigid ground
  //
  constexpr real kGroundHeight = 0_r;
  constexpr real kBoundaryHeight = kGroundHeight + 1_r;
  constexpr real kStartingGap = 0.05_r;
  Actor* ground = CreateStaticGroundPlane(_scene, kGroundHeight);
  Actor* rigidTopCube =
      CreateRigidCube(_scene, 0.5_r, Real3{0_r, kBoundaryHeight + kStartingGap, 0_r});
  Actor* rigidBottomCube = CreateRigidCube(_scene, kBoundaryHeight, Real3{0_r, kGroundHeight, 0_r});

  TestQueryStackedContact(
      _scene, rigidTopCube, rigidBottomCube, ground, kGroundHeight, kBoundaryHeight);

  _scene->DestroyActor(rigidTopCube);
  _scene->DestroyActor(rigidBottomCube);
}

TEST_F(ActorQueryTest, ContactPoints_Soft_on_DynamicRigid) {
  // Create a dynamic rigid cube that will fall onto the ground plane.
  // Create a smaller soft cube that will fall onto the dynamic rigid cube.
  //
  //           -----
  //           |   | soft cube
  //           |   |
  //         --=====-- boundary
  //         |       |
  //         |       | rigid cube
  //         |       |
  //   ------=========------ static rigid ground
  //
  constexpr real kGroundHeight = 0_r;
  constexpr real kBoundaryHeight = 1_r;
  Actor* ground = CreateStaticGroundPlane(_scene, kGroundHeight);
  Actor* softTopCube = CreateSoftCube(_scene, 0.5_r, Real3{0_r, kBoundaryHeight, 0_r});
  Actor* rigidBottomCube = CreateRigidCube(_scene, kBoundaryHeight, Real3{0_r, kGroundHeight, 0_r});

  TestQueryStackedContact(
      _scene, softTopCube, rigidBottomCube, ground, kGroundHeight, kBoundaryHeight);

  _scene->DestroyActor(softTopCube);
  _scene->DestroyActor(rigidBottomCube);
}

TEST_F(ActorQueryTest, ContactPoints_DynamicRigid_on_Soft) {
  // Create a very stiff soft cube that will fall onto the ground plane.
  // Create a dynamic rigid cube that will fall onto the soft cube.
  //
  //         ---------
  //         |       |
  //         |       | rigid cube
  //         |       |
  //         --=====-- boundary
  //         |       |
  //         |       | soft cube
  //         |       |
  //   ------=========------ static rigid ground
  //
  constexpr real kGroundHeight = 0_r;
  constexpr real kBoundaryHeight = 0.1_r;
  constexpr real kInitialAirGap = 0.01_r;
  Actor* rigidTopCube =
      CreateRigidCube(_scene, 0.1_r, Real3{0_r, kBoundaryHeight + kInitialAirGap, 0_r});
  Actor* softBottomCube = CreateSoftCube(_scene, 0.1_r, Real3{0_r, kGroundHeight, 0_r});
  Actor* ground = CreateStaticGroundPlane(_scene, kGroundHeight);

  TestQueryStackedContact(
      _scene, rigidTopCube, softBottomCube, ground, kGroundHeight, kBoundaryHeight);
}

static void ExpectEqualContactForces(bool expectSuccess, Actor* actor) {
  Error error;
  Span<NodeContactForce const> nodeForces = actor->GetNodeContactForcesWorld(error);
  Span<ContactPoint const> contacts = actor->GetContactPointsWorld(error);
  if (expectSuccess) {
    // Except success
    EXPECT_OK(error);
    EXPECT_NE(0, nodeForces.size());
    EXPECT_NE(0, contacts.size());

    // Add node contact forces
    Real3 forceSum1 = {};
    for (auto const& node : nodeForces) {
      forceSum1 += node.force;
    }

    // Add forces from contact points. Discard the ones for which the actor is the collider.
    Real3 forceSum2 = {};
    for (auto const& contact : contacts) {
      if (actor->GetHandle() == contact.actorA) {
        forceSum2 += contact.force;
      }
    }

    // Expect summed forces to be equal
    real tol = 1e-6_r * std::max(Norm(forceSum1), Norm(forceSum2));
    EXPECT_NEAR_TOL(forceSum1, forceSum2, tol);
  } else {
    EXPECT_NOT_OK(error);
    EXPECT_EQ(0, nodeForces.size());
    EXPECT_EQ(0, contacts.size());
  }
}

// Verifies that contact forces on an actor are nearly vertical (Y-dominant).
// This is expected for stacked cubes under gravity.
static void ExpectVerticalContactForces(bool expectSuccess, Actor* actor) {
  Error error;
  Span<NodeContactForce const> nodeForces = actor->GetNodeContactForcesWorld(error);
  Span<ContactPoint const> contacts = actor->GetContactPointsWorld(error);

  if (expectSuccess) {
    EXPECT_OK(error);

    constexpr real kVerticalityTolerance = 0.05_r; // Allow small deviation from pure vertical

    // Check node contact forces are vertical
    for (auto const& node : nodeForces) {
      real forceMag = Norm(node.force);
      if (forceMag > 1e-6_r) {
        Real3 normalizedForce = node.force / forceMag;
        real yComponent = Abs(normalizedForce[1]);
        EXPECT_GT(yComponent, 1_r - kVerticalityTolerance)
            << "Node contact force is not vertical: " << node.force[0] << ", " << node.force[1]
            << ", " << node.force[2];
      }
    }

    // Check contact point forces are vertical
    for (auto const& contact : contacts) {
      if (actor->GetHandle() == contact.actorA) {
        real forceMag = Norm(contact.force);
        if (forceMag > 1e-6_r) {
          Real3 normalizedForce = contact.force / forceMag;
          real yComponent = Abs(normalizedForce[1]);
          EXPECT_GT(yComponent, 1_r - kVerticalityTolerance)
              << "Contact point force is not vertical: " << contact.force[0] << ", "
              << contact.force[1] << ", " << contact.force[2];
        }
      }
    }
  } else {
    EXPECT_NOT_OK(error);
  }
}

static void TestNodeContactForces(Scene* scene, Actor* bodyA, Actor* bodyB) {
  // Enable gravity and let the bodies fall on the ground.
  // Step many times to make sure it is fully settled.
  Real3 const gravity = Real3{0_r, -10_r, 0_r};
  scene->SetGravity(gravity);
  // The near-vertical force check relies on backward Euler's damping to settle the stack.
  test::SetSceneIntegrationMethod(scene, IntegrationMethod::BackwardEuler);
  for (int i = 0; i < 75; ++i) {
    scene->Step(0.01);
  }

  // Now we can expect to get a non-empty collection of contact points when the query is enabled
  TestQuery(scene, bodyA, bodyB, QueryType::NodeContactForces, [&](bool expectSuccess, Actor* a) {
    ExpectEqualContactForces(expectSuccess, a);
    ExpectVerticalContactForces(expectSuccess, a);
  });
}

// Parameters for NodeContactForces parameterized tests
struct NodeContactForcesParams {
  RotationAxis90 topRotation;
  RotationAxis90 bottomRotation;

  std::string GetName() const {
    return std::string("Top") + SReflect::EnumToString(topRotation) + "_Bottom" +
        SReflect::EnumToString(bottomRotation);
  }
};

// Generate all combinations of rotation axes for top and bottom cubes
static std::vector<NodeContactForcesParams> GetNodeContactForcesTestCases() {
  std::vector<NodeContactForcesParams> params;
  for (auto topRot :
       {RotationAxis90::None, RotationAxis90::X, RotationAxis90::Y, RotationAxis90::Z}) {
    for (auto bottomRot :
         {RotationAxis90::None, RotationAxis90::X, RotationAxis90::Y, RotationAxis90::Z}) {
      params.push_back({topRot, bottomRot});
    }
  }
  return params;
}

// Parameterized test fixture for NodeContactForces tests
class NodeContactForcesTest : public test::MochiSceneTestBase,
                              public ::testing::WithParamInterface<NodeContactForcesParams> {
 protected:
  void SetUp() override {
    MochiSceneTestBase::SetUp();
    // Enable contact between all objects in the default layer
    _scene->EnableLayerContactSymmetric(kDefaultLayer, kDefaultLayer, true, test::ExpectOK{});
    _scene->EnableLayerContactSymmetric("GROUND", kDefaultLayer, true, test::ExpectOK{});
  }
};

TEST_P(NodeContactForcesTest, DynamicSoft_on_StaticRigid) {
  // Check that QueryType::NodeContactForces gives the same total force as QueryType::ContactPoints.
  // Soft object on the ground.
  auto params = GetParam();
  Actor* bodyA = CreateSoftCube(_scene, 0.1_r, Real3{-1_r, 0.001_r, 0_r}, params.topRotation);
  Actor* bodyB = CreateSoftCube(_scene, 0.11_r, Real3{-1_r, 0.001_r, 0_r}, params.bottomRotation);
  CreateStaticGroundPlane(_scene, -1);
  TestNodeContactForces(_scene, bodyA, bodyB);
}

TEST_P(NodeContactForcesTest, DynamicRigid_on_StaticRigid) {
  // Check that QueryType::NodeContactForces gives the same total force as QueryType::ContactPoints.
  // Rigid object on the ground.
  auto params = GetParam();
  Actor* bodyA = CreateRigidCube(
      _scene, 1_r, Real3{-10_r, 0_r, 0_r}, false, ColliderType::Box, params.topRotation);
  Actor* bodyB = CreateRigidCube(
      _scene, 2_r, Real3{10_r, 0_r, 0_r}, false, ColliderType::Box, params.bottomRotation);
  CreateStaticGroundPlane(_scene, -1);
  TestNodeContactForces(_scene, bodyA, bodyB);
}

TEST_P(NodeContactForcesTest, DynamicSoft_on_DynamicRigid) {
  // Check that QueryType::NodeContactForces gives the same total force as QueryType::ContactPoints.
  // Soft object on rigid object.
  auto params = GetParam();
  constexpr real kGroundHeight = 0_r;
  constexpr real kBoundaryHeight = kGroundHeight + 1_r;
  constexpr real kStartingGap = 0.05_r;
  Actor* softTopCube = CreateSoftCube(
      _scene, 0.5_r, Real3{0_r, kBoundaryHeight + kStartingGap, 0_r}, params.topRotation);
  Actor* rigidBottomCube = CreateRigidCube(
      _scene,
      kBoundaryHeight,
      Real3{0_r, kGroundHeight, 0_r},
      false,
      ColliderType::Box,
      params.bottomRotation);
  CreateStaticGroundPlane(_scene, kGroundHeight);
  TestNodeContactForces(_scene, softTopCube, rigidBottomCube);
}

TEST_P(NodeContactForcesTest, DynamicRigid_on_DynamicRigid) {
  // Check that QueryType::NodeContactForces gives the same total force as QueryType::ContactPoints.
  // Rigid object on rigid object.
  auto params = GetParam();
  constexpr real kGroundHeight = 0_r;
  constexpr real kBoundaryHeight = kGroundHeight + 1_r;
  constexpr real kStartingGap = 0.05_r;
  Actor* rigidTopCube = CreateRigidCube(
      _scene,
      0.5_r,
      Real3{0_r, kBoundaryHeight + kStartingGap, 0_r},
      false,
      ColliderType::Box,
      params.topRotation);
  Actor* rigidBottomCube = CreateRigidCube(
      _scene,
      kBoundaryHeight,
      Real3{0_r, kGroundHeight, 0_r},
      false,
      ColliderType::Box,
      params.bottomRotation);
  CreateStaticGroundPlane(_scene, kGroundHeight);
  TestNodeContactForces(_scene, rigidTopCube, rigidBottomCube);
}

INSTANTIATE_TEST_SUITE_P(
    RotationVariations,
    NodeContactForcesTest,
    ::testing::ValuesIn(GetNodeContactForcesTestCases()),
    [](::testing::TestParamInfo<NodeContactForcesParams> const& info) {
      return info.param.GetName();
    });

static void ExpectedElasticEnergy(bool expectSuccess, Actor const* actor) {
  // Called by TestQuery to verify that the query results are available.
  Error error;
  real energy = actor->GetElasticEnergy(error);
  if (expectSuccess) {
    EXPECT_OK(error);

    // Expect the energy not to be zero
    EXPECT_GT(energy, 1e-6_r);
  } else {
    EXPECT_NOT_OK(error);
  }
}

TEST_IF_F(MOCHI_TEST_ROM_HDF5, ActorQueryTest, ElasticEnergy) {
  auto& reg = GetRegistry();

  // Create actors (soft and ROM with hyperreduction) and add some initial velocity
  auto* actorA = CreateSoftCube(_scene, 1_r, Real3{-10_r, 0_r, 0_r});
  auto entityA = mochi::GetEntity(reg, actorA->GetHandle(), test::ExpectOK{});
  reg.get<CVelocitySlice<real, TimeStep::Current>>(entityA).value.SetRandom(123, -1_r, 1_r);
  auto* actorB = CreateRomDuck(_mochiContext, _scene, 2_r, Real3{10_r, 0_r, 0_r});
  auto entityB = mochi::GetEntity(reg, actorB->GetHandle(), test::ExpectOK{});
  reg.get<CVelocitySlice<real, TimeStep::Current>>(entityB).value.SetRandom(234, -1_r, 1_r);
  TestQuery(
      _scene, actorA, actorB, QueryType::ElasticEnergy, [](bool expectResults, Actor const* a) {
        return ExpectedElasticEnergy(expectResults, a);
      });
}

TEST_IF_F(MOCHI_HDF5_AND_INTERNAL, ActorQueryTest, ElementsGradients) {
  auto rndgen = RandomGenerator(23);

  Error error;
  real scale = 1_r;
  Real3 positionSpawn{-10_r, 0_r, 0_r};
  SoftActorParams caparams;
  caparams.name = "SoftDuck";
  constexpr char const* kDuckMesh = "duck/duck_3504.mochi.h5";
  ShapeHandle duckShape = _mochiContext->LoadShapeFromFile(
      test::GetAssetPath(kDuckMesh), Real3{scale, scale, scale}, TransformRT::Identity(), error);
  caparams.shape = duckShape;
  caparams.worldFromLocal = TransformRT{positionSpawn};

  // Hard code material params in case the default change in a way that would affect this test.
  caparams.material.type = SoftMaterialType::NeoHookean;
  caparams.material.neoHookean.youngsModulus = 1e5_r;
  caparams.material.neoHookean.poissonRatio = 0.45_r;
  caparams.material.density = 1000_r;
  caparams.layer = kDefaultLayer;
  caparams.contact.viscousFrictionCoefficient = 2_r;
  auto* actor = _scene->CreateSoftActor(caparams, error);
  EXPECT_OK(error);

  auto shape = actor->GetReferenceShape(error);
  EXPECT_OK(error);
  auto mesh = _mochiContext->GetShapeMesh(shape, error);
  EXPECT_OK(error);
  auto positions = Unflatten<Real3 const>(mesh.coordinates);

  // Test with uniform deformation gradient
  // Namely a displacement given by U = FX - X
  // So that the deformation gradient is simply F
  mochi::Matrix3x3r deformationGradientExact{};
  SetRandom(rndgen, 1.e-4_r, 2_r, deformationGradientExact);

  std::vector<Real3> deformedPositions(positions.size(), Real3{});
  for (int i = 0; i < positions.size(); ++i) {
    auto position = positions[i];
    deformedPositions[i] = DotMatVec(deformationGradientExact, position);
  }
  actor->SetNodePositionsLocal(Flatten(MakeSpan(deformedPositions)), error);
  EXPECT_OK(error);
  auto query = actor->RegisterQueryAndCompute(QueryType::ElementsDeformationGradient, error);
  auto elementsDeformationGradient =
      Unflatten<Matrix3x3r const>(actor->GetElementsDeformationGradient(error));

  for (auto elementDeformationGradient : elementsDeformationGradient) {
    EXPECT_NEAR(
        0_r,
        Norm(elementDeformationGradient - deformationGradientExact) /
            Norm(deformationGradientExact),
        5e-5_r);
  }
  actor->CancelQuery(query);

  // Test with a displacement given by a * (X \cdot X)
  // Such that the gradient is given by 2 a \outer X

  // Get the deformed node positions
  Real3 a{1_r, 2_r, 3_r};
  for (int i = 0; i < positions.size(); ++i) {
    auto position = positions[i];
    deformedPositions[i] = Dot(position, position) * a;
  }

  // Set them into the actor
  actor->SetNodePositionsLocal(Flatten(MakeSpan(deformedPositions)), error);

  // Register the query (computing immediately) and get value
  query = actor->RegisterQueryAndCompute(QueryType::ElementsDeformationGradient, error);
  elementsDeformationGradient =
      Unflatten<Matrix3x3r const>(actor->GetElementsDeformationGradient(error));

  auto barycentersFlat = ComputeElementBarycenters(mesh);
  auto barycenters = Unflatten<Real3 const>(MakeConstSpan(barycentersFlat));
  for (int i = 0; i < elementsDeformationGradient.size(); ++i) {
    auto elementDeformationGradient = elementsDeformationGradient[i];
    auto barycenter = barycenters[i];
    deformationGradientExact = 2_r * Outer(a, barycenter);
    // Note the tolerance here is large because this will introduce some
    // error as the displacement is quadratic but approximated linearly and the deformation gradient
    // is linear but approximated by piecewise constants
    EXPECT_NEAR(
        0_r,
        Norm(elementDeformationGradient - deformationGradientExact) /
            Norm(deformationGradientExact),
        5e-2_r);
  }
}

// TODO[T143757849] Add unit tests for the following:
//    QueryType::VisualNodePositions (rigid actors)
//    QueryType::VisualNodeNormals

// Test that visual node position and normal queries work for shell actors with a visual mesh
// embedding.
TEST_F(ActorQueryTest, ShellVisualNodePositionsAndNormals) {
  // Sim mesh: a square made of 2 triangles with 4 nodes.
  //   node 0: (0, 0, 0)
  //   node 1: (1, 0, 0)
  //   node 2: (1, 1, 0)
  //   node 3: (0, 1, 0)
  //   triangles: [0,1,2] and [0,2,3]
  std::vector<real> simCoords = {0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 1_r, 1_r, 0_r, 0_r, 1_r, 0_r};
  std::vector<int> simConn = {0, 1, 2, 0, 2, 3};

  // Visual mesh: 4 visual nodes, 1 triangle. Node 0 is intentionally unreferenced by
  // visual connectivity to verify visual mesh getters preserve full visual node-index space.
  //   vis 0: exactly sim node 0            -> expected (0, 0, 0)
  //   vis 1: midpoint of sim nodes 0 and 1 -> expected (0.5, 0, 0)
  //   vis 2: centroid of sim nodes 0, 1, 2 -> expected (2/3, 1/3, 0)
  //   vis 3: exactly sim node 3            -> expected (0, 1, 0)
  std::vector<real> visCoords = {
      0_r, 0_r, 0_r, 0.5_r, 0_r, 0_r, 2_r / 3_r, 1_r / 3_r, 0_r, 0_r, 1_r, 0_r};
  std::vector<int> visConn = {1, 2, 3};

  // Embedding: weightsPerNode = 3 (barycentric), 4 visual nodes -> 12 indices, 12 weights.
  int const weightsPerNode = 3;
  std::vector<int> skinIndices = {
      0,
      0,
      0, // vis 0: node 0, (unused 0, 0)
      0,
      1,
      0, // vis 1: nodes 0, 1, (unused 0)
      0,
      1,
      2, // vis 2: nodes 0, 1, 2
      3,
      0,
      0, // vis 3: node 3, (unused 0, 0)
  };
  std::vector<real> skinWeights = {
      1_r,
      0_r,
      0_r, // vis 0: exactly node 0
      0.5_r,
      0.5_r,
      0_r, // vis 1: midpoint of 0 and 1
      1_r / 3_r,
      1_r / 3_r,
      1_r / 3_r, // vis 2: centroid of 0, 1, 2
      1_r,
      0_r,
      0_r, // vis 3: exactly node 3
  };

  // Build ModelData with visual mesh and skinning.
  ModelData modelData;
  modelData.mesh.emplace();
  modelData.mesh->nodesPerElement = 3;
  modelData.mesh->coordinates = DynamicArray<real>{simCoords};
  modelData.mesh->connectivity = DynamicArray<int>{simConn};
  modelData.visualMesh.emplace();
  modelData.visualMesh->nodesPerElement = 3;
  modelData.visualMesh->coordinates = DynamicArray<real>{visCoords};
  modelData.visualMesh->connectivity = DynamicArray<int>{visConn};
  modelData.visualMesh->skinning.emplace();
  modelData.visualMesh->skinning->weightsPerNode = weightsPerNode;
  modelData.visualMesh->skinning->indices = DynamicArray<int>{skinIndices};
  modelData.visualMesh->skinning->weights = DynamicArray<real>{skinWeights};

  ShapeHandle shape = _mochiContext->CreateModelShape(modelData, test::ExpectOK{});

  ShellActorParams params;
  params.shape = shape;
  params.material.density = 1_r;
  params.layer = kDefaultLayer;
  auto* actor = CreateShellActor(_scene, params, test::ExpectOK{});
  MOCHI_DEFER(_scene->DestroyActor(actor));

  auto const actorVisualMesh = actor->GetVisualMesh();
  EXPECT_EQ(4, actorVisualMesh.GetNumNodes());
  EXPECT_SPAN_EQ(MakeConstSpan(visCoords), actorVisualMesh.coordinates);
  EXPECT_SPAN_EQ(MakeConstSpan(visConn), actorVisualMesh.connectivity);

  MeshDataView shapeVisualMesh = _mochiContext->GetShapeVisualMesh(shape, test::ExpectOK{});
  EXPECT_EQ(4, shapeVisualMesh.GetNumNodes());
  EXPECT_SPAN_EQ(MakeConstSpan(visCoords), shapeVisualMesh.coordinates);
  EXPECT_SPAN_EQ(MakeConstSpan(visConn), shapeVisualMesh.connectivity);

  Real3 constexpr kExpected0{0_r, 0_r, 0_r};
  Real3 constexpr kExpected1{0.5_r, 0_r, 0_r};
  Real3 constexpr kExpected2{2_r / 3_r, 1_r / 3_r, 0_r};
  Real3 constexpr kExpected3{0_r, 1_r, 0_r};

  auto positionsQuery =
      actor->RegisterQueryAndCompute(QueryType::VisualNodePositions, test::ExpectOK{});
  {
    MOCHI_DEFER(actor->CancelQuery(positionsQuery));
    auto visPositions =
        Unflatten<Real3 const>(actor->GetVisualMeshNodePositionsLocal(test::ExpectOK{}));
    ASSERT_EQ(4, isize(visPositions));

    // Verify: at the reference configuration, visual positions should match
    // the barycentric interpolation of the sim mesh positions.
    EXPECT_NEAR_EQ(visPositions[0], kExpected0);
    EXPECT_NEAR_EQ(visPositions[1], kExpected1);
    EXPECT_NEAR_EQ(visPositions[2], kExpected2);
    EXPECT_NEAR_EQ(visPositions[3], kExpected3);
  }

  auto normalsQuery =
      actor->RegisterQueryAndCompute(QueryType::VisualNodeNormals, test::ExpectOK{});
  MOCHI_DEFER(actor->CancelQuery(normalsQuery));

  auto visNormals = Unflatten<Real3 const>(actor->GetVisualMeshNodeNormalsLocal(test::ExpectOK{}));
  ASSERT_EQ(4, isize(visNormals));
  Real3 constexpr kExpectedReferencedNormal{0_r, 0_r, 1_r};
  EXPECT_NEAR_EQ(visNormals[0], Real3{});
  EXPECT_NEAR_EQ(visNormals[1], kExpectedReferencedNormal);
  EXPECT_NEAR_EQ(visNormals[2], kExpectedReferencedNormal);
  EXPECT_NEAR_EQ(visNormals[3], kExpectedReferencedNormal);
}

// Test fixture which creates an empty mochi::Scene
class ConstraintQueryTest : public test::MochiSceneTestBase {
 public:
  void SetUp() override {
    // Call down
    test::MochiSceneTestBase::SetUp();
  }
};

TEST_F(ConstraintQueryTest, QueryConstraintForce) {
  real constexpr kDt = 1e-2_r;

  // Create a rigid cube
  real constexpr kCubeSize = 1_r;
  auto* cube = CreateRigidCube(_scene, kCubeSize, Real3{});

  // Create a rigid pivot position constraint at the cube's root
  RigidPivotPositionConstraintParams conParams;
  conParams.stiffness = 1_r;
  conParams.damping = conParams.stiffness * kDt; // To keep stiffness and damping on the same scale
  conParams.localPosition = {};
  conParams.targetPosition = cube->GetRootTransform().GetTranslation();
  conParams.actor = cube->GetHandle();
  auto* constraint = _scene->CreateRigidPivotPositionConstraint(conParams, test::ExpectOK{});

  // Set the position and velocity of the cube and the constraint target
  Real3 posOld = Real3{1_r, 2_r, 3_r};
  auto rotOld = Quaternion::FromRotationVector(Real3{0.1_r, 0.2_r, 0.3_r});
  cube->SetRootTransform(TransformRT{rotOld, posOld}, test::ExpectOK{});
  Real3 vel = Real3{1_r, 2_r, 3_r};
  Real3 omega = Real3{-1_r, -2_r, -3_r};
  cube->SetVelocity(vel, omega, test::ExpectOK{});
  Real3 target = Real3{-2_r, -3_r, -1_r};
  constraint->SetTargetPosition(target, test::ExpectOK{});

  // Create the query
  auto query = constraint->RegisterQuery(QueryType::ConstraintForce, test::ExpectOK{});

  // Simulate a step
  _scene->Step(kDt);

  // Get the state of the cube
  auto const pos = cube->GetRootTransform().GetTranslation();
  auto const rot = cube->GetRootTransform().GetRotation();

  // Get the result of the query.
  auto const forceConstraint = constraint->GetForce(test::ExpectOK{});
  EXPECT_EQ(forceConstraint.size(), 6);
  auto const force = Unflatten<Real3 const>(forceConstraint)[0];
  auto const torque = Unflatten<Real3 const>(forceConstraint)[1];

  // Raw evaluation of the constraint force and torque.
  Real3 C = pos - target;
  Real3 targetOld = Real3{};
  Real3 COld = posOld - targetOld;
  Real3 forceTest = -conParams.stiffness * C - (conParams.damping / kDt) * (C - COld);
  Real3 pivot = rot * (-0.5_r * kCubeSize * Real3{1_r, 1_r, 1_r});
  Real3 torqueTest = Cross(pivot, forceTest);
  EXPECT_NEAR_TOL(force, forceTest, 1e-5_r);
  EXPECT_NEAR_TOL(torque, torqueTest, 1e-5_r);

  // Cancel the query
  constraint->CancelQuery(query);
}

TEST_F(ActorQueryTest, GetPointsDistanceToSurface) {
  Real3 constexpr kPosition = {0.1_r, 0.2_r, 0.3_r};
  auto constexpr kHeight = 1_r;

  auto RunTest = [](Actor* actor, Span<Real3 const> points, Span<real const> expectedDistances) {
    std::vector<real> distances(points.size(), kInf);
    actor->GetPointsDistanceToSurface(points, distances, test::ExpectOK{});
    EXPECT_TRUE(test::NearEqualSpan(distances, expectedDistances));
  };

  std::vector<Real3> const kBoxPoints = {
      Real3{0.00_r, 0.50_r, 0.50_r},
      Real3{0.50_r, 0.50_r, 0.50_r},
      Real3{1.00_r, 0.50_r, 0.50_r},
      Real3{1.50_r, 0.50_r, 0.50_r},
      Real3{2.00_r, 0.50_r, 0.50_r}};
  std::vector<Real3> const kSpherePoints = {
      Real3{0.1_r, 0.2_r, 0.3_r},
      Real3{1.1_r, 0.2_r, 0.3_r},
      Real3{2.1_r, 0.2_r, 0.3_r},
      Real3{1.1_r, 1.2_r, 0.3_r}};
  std::vector<Real3> const kPlanePoints = {
      Real3{0_r, -1_r, 0_r},
      Real3{1_r, 0_r, 0_r},
      Real3{2_r, 1_r, 0_r},
      Real3{3_r, 2_r, 0_r},
  };

  std::vector<real> const kBoxExpectedDistances = {0.1_r, -0.2_r, -0.1_r, 0.4_r, 0.9_r};
  std::vector<real> const kSphereExpectedDistances = {-1_r, 0_r, 1_r, Sqrt(2_r) - 1_r};
  std::vector<real> const kPlaneExpectedDistances = {-2_r, -1_r, 0_r, 1_r};

  // Box
  RunTest(
      CreateRigidCube(_scene, 1_r, kPosition, false, ColliderType::Box),
      kBoxPoints,
      kBoxExpectedDistances);
  // Mesh
  RunTest(
      CreateRigidCube(_scene, 1_r, kPosition, false, ColliderType::Mesh),
      kBoxPoints,
      kBoxExpectedDistances);
  // SDF
  RunTest(
      CreateRigidCube(_scene, 1_r, kPosition, false, ColliderType::Sdf),
      kBoxPoints,
      kBoxExpectedDistances);
  // Sphere
  RunTest(CreateStaticSphere(_scene, 1_r, kPosition), kSpherePoints, kSphereExpectedDistances);
  // Plane
  RunTest(CreateStaticGroundPlane(_scene, kHeight), kPlanePoints, kPlaneExpectedDistances);
}

static void Compare(Real3 const& a, Real3 const& b) {
  real normDiff = Norm(a - b);
  real normMax = Max(Norm(a), Norm(b));
  EXPECT_LE(normDiff, 1e-3_r * normMax);
}

TEST_F(ActorQueryTest, ActorContactForces) {
  std::array<Actor*, 9> actors = {};

  // Function to enable a query on all actors but the ground
  auto enableQuery = [&](QueryType queryType) {
    for (int i = 0; i < 8; ++i) {
      actors[i]->RegisterQuery(queryType, test::ExpectOK{});
    }
  };

  // Create actors: 8 cubes on a grid, first and last soft, and a ground plane.
  actors[0] = CreateSoftCube(_scene, 1_r, Real3{0_r, 0_r, 0_r});
  actors[1] = CreateRigidCube(_scene, 1_r, Real3{0_r, 0_r, 1_r});
  actors[2] = CreateRigidCube(_scene, 1_r, Real3{0_r, 1_r, 0_r});
  actors[3] = CreateRigidCube(_scene, 1_r, Real3{0_r, 1_r, 1_r});
  actors[4] = CreateRigidCube(_scene, 1_r, Real3{1_r, 0_r, 0_r});
  actors[5] = CreateRigidCube(_scene, 1_r, Real3{1_r, 0_r, 1_r});
  actors[6] = CreateRigidCube(_scene, 1_r, Real3{1_r, 1_r, 0_r});
  actors[7] = CreateSoftCube(_scene, 1_r, Real3{1_r, 1_r, 1_r});
  actors[8] = CreateStaticGroundPlane(_scene, 0_r);

  std::map<ActorHandle, int> actorToIndex = {};
  for (int i = 0; i < 9; ++i) {
    actorToIndex[actors[i]->GetHandle()] = i;
  }

  StateHandle initialState = _scene->CaptureState(ErrorAssert{});

  // Enable QueryType::TotalContactForce and step
  enableQuery(QueryType::TotalContactForce);
  _scene->Step(1e-3_r);

  // Collect pair-wise contact forces
  Matrix<Real3, 8, 9> forces;
  for (int i = 0; i < 8; ++i) { // Do not query the ground
    for (int j = 0; j < 9; ++j) { // Query against the ground too
      forces(i, j) = actors[i]->GetContactForceFromActorWorld(actors[j], test::ExpectOK{});
    }
  }

  // Collect total forces on the actors
  std::array<Real3, 8> totalForces = {};
  for (int i = 0; i < 8; ++i) {
    totalForces[i] = actors[i]->GetContactForceWorld(test::ExpectOK{});
  }

  // Check that they add up to the same values and they are non-zero
  for (int i = 0; i < 8; ++i) {
    Real3 sum = {};
    for (int j = 0; j < 9; ++j) {
      sum += forces(i, j);
    }
    EXPECT_NE(totalForces[i], Real3{});
    Compare(totalForces[i], sum);
  }

  // Restore initial state, enable QueryType::ContactPoints and step again.
  _scene->RestoreState(initialState, /* releaseImmediately */ true, ErrorAssert{});
  enableQuery(QueryType::ContactPoints);
  _scene->Step(1e-3_r);

  // Collect pair-wise contact forces by summing up point forces
  Matrix<Real3, 8, 9> forcesTest = Matrix<Real3, 8, 9>::Zero();
  for (int i = 0; i < 8; ++i) { // Do not query the ground
    auto contacts = actors[i]->GetContactPointsWorld(test::ExpectOK{});
    for (auto const& contact : contacts) {
      if (contact.actorA == actors[i]->GetHandle()) {
        // Contact registered in colliding role
        forcesTest(i, actorToIndex[contact.actorB]) += contact.force;
      } else {
        // Contact registered in collider role
        EXPECT_EQ(contact.actorB, actors[i]->GetHandle());
        forcesTest(i, actorToIndex[contact.actorA]) -= contact.force;
      }
    }
  }

  // Check that they add up to the same values
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 9; ++j) {
      Compare(forces(i, j), forcesTest(i, j));
    }
  }
}

// Verifies Actor::QueryNodesInVolumeLocal across all three volume shapes and both boundaryOnly
// branches. The tetrahedral mesh has an interior node first, so its full and surface node
// orderings differ.
TEST_F(ActorQueryTest, QueryNodesInVolumeLocal) {
  constexpr std::array kCoordinates = {
      Real3{0.25_r, 0.25_r, 0.25_r},
      Real3{0_r, 0_r, 0_r},
      Real3{1_r, 0_r, 0_r},
      Real3{0_r, 1_r, 0_r},
      Real3{0_r, 0_r, 1_r}};
  constexpr std::array kConnectivity = {
      Int4{0, 2, 3, 4}, Int4{0, 1, 4, 3}, Int4{0, 1, 2, 4}, Int4{0, 1, 3, 2}};
  ShapeHandle shape = _scene->GetContext()->CreateTetMeshShape(
      Flatten(MakeConstSpan(kCoordinates)),
      Flatten(MakeConstSpan(kConnectivity)),
      test::ExpectOK{});

  SoftActorParams softParams;
  softParams.shape = shape;
  Actor* softActor = _scene->CreateSoftActor(softParams, test::ExpectOK{});

  auto queryIndices = [](Actor* queriedActor, auto const& volume, bool boundaryOnly) {
    MeshDataView const mesh = queriedActor->GetMesh();
    auto const meshPositions = Unflatten<Real3 const>(mesh.coordinates);
    DynamicArray<int> indices;
    queriedActor->QueryNodesInVolumeLocal(
        volume,
        boundaryOnly,
        [&](int index, Real3 pos) {
          EXPECT_GE(index, 0);
          EXPECT_LT(index, isize(meshPositions));
          if (index >= 0 && index < isize(meshPositions)) {
            EXPECT_NEAR_EQ(meshPositions[index], pos);
          }
          indices.push_back(index);
        },
        test::ExpectOK{});
    std::ranges::sort(indices);
    return indices;
  };

  DynamicArray<int> const allNodes = {0, 1, 2, 3, 4};
  DynamicArray<int> const selectedNodes = {2};
  DynamicArray<int> const fullMeshBoundaryNodes = {1, 2, 3, 4};
  DynamicArray<int> const compactBoundaryNodes = {0, 1, 2, 3};

  Aabb const enclosingAabb{Real3{-1_r, -1_r, -1_r}, Real3{2_r, 2_r, 2_r}};
  Aabb const emptyAabb{Real3{10_r, 10_r, 10_r}, Real3{11_r, 11_r, 11_r}};
  Aabb const selectiveAabb{Real3{0.5_r, -1_r, -1_r}, Real3{2_r, 2_r, 2_r}};
  Sphere const enclosingSphere{Real3{0.5_r, 0.5_r, 0.5_r}, 2_r};
  Sphere const selectiveSphere{Real3{1_r, 0_r, 0_r}, 0.1_r};
  Obb const enclosingObb{
      TransformRT{Quaternion::Identity(), Real3{0.5_r, 0.5_r, 0.5_r}}, Real3{1_r, 1_r, 1_r}};
  Obb const selectiveObb{
      TransformRT{Quaternion::Identity(), Real3{1_r, 0_r, 0_r}}, Real3{0.1_r, 0.1_r, 0.1_r}};

  {
    auto query = softActor->RegisterQueryAndCompute(QueryType::NodePositions, test::ExpectOK{});
    MOCHI_DEFER(softActor->CancelQuery(query));
    auto const nodePositions =
        Unflatten<Real3 const>(softActor->GetNodePositionsLocal(test::ExpectOK{}));
    ASSERT_EQ(kCoordinates.size(), nodePositions.size());
    for (int i = 0; i < isize(nodePositions); ++i) {
      EXPECT_NEAR_EQ(kCoordinates[i], nodePositions[i]);
    }
    EXPECT_EQ(allNodes, queryIndices(softActor, enclosingAabb, false));
    EXPECT_TRUE(queryIndices(softActor, emptyAabb, false).empty());
    EXPECT_EQ(selectedNodes, queryIndices(softActor, selectiveAabb, false));
    EXPECT_EQ(allNodes, queryIndices(softActor, enclosingSphere, false));
    EXPECT_EQ(selectedNodes, queryIndices(softActor, selectiveSphere, false));
    EXPECT_EQ(allNodes, queryIndices(softActor, enclosingObb, false));
    EXPECT_EQ(selectedNodes, queryIndices(softActor, selectiveObb, false));
  }

  {
    auto query =
        softActor->RegisterQueryAndCompute(QueryType::SurfaceNodePositions, test::ExpectOK{});
    MOCHI_DEFER(softActor->CancelQuery(query));
    EXPECT_EQ(fullMeshBoundaryNodes, queryIndices(softActor, enclosingAabb, true));
    EXPECT_EQ(selectedNodes, queryIndices(softActor, selectiveAabb, true));
  }

  for (bool const isStatic : {false, true}) {
    SCOPED_TRACE(testing::Message() << "isStatic = " << isStatic);
    RigidActorParams rigidParams;
    rigidParams.shape = shape;
    rigidParams.colliderType = ColliderType::None;
    rigidParams.isStatic = isStatic;
    Actor* rigidActor = _scene->CreateRigidActor(rigidParams, test::ExpectOK{});

    auto query =
        rigidActor->RegisterQueryAndCompute(QueryType::SurfaceNodePositions, test::ExpectOK{});
    MOCHI_DEFER(rigidActor->CancelQuery(query));
    EXPECT_EQ(compactBoundaryNodes, queryIndices(rigidActor, enclosingAabb, true));
  }
}
