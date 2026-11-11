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

/**
  This file tests the mochi::Actor APIs involving Boundary Conditions
*/

#include "mochi_physics_test_fixture.h"
#include "mochi_test_scene_builders.h"

#include <mochi_core/utils/reflection.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/src/mochi_simulation.h>

#include <array>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using namespace mochi;

// Test fixture class
class MochiActorBC : public test::MochiSceneTestBase {
 public:
  using BaseClass = test::MochiSceneTestBase;
  static constexpr real kBoxWidth = 0.1_r;

  void SetUp() override {
    BaseClass::SetUp();

    // Create cube shape
    auto [coordinates, connectivity] =
        test::CreateMinimalTetMeshUnitCube(Real3{kBoxWidth, kBoxWidth, kBoxWidth});
    _cubeShape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});
  }

  void TearDown() override {
    BaseClass::TearDown();
  }

  static void ExpectBoundaryConditionSettersRejected(Actor* actor, char const* expectedError) {
    auto const expectRejected = [&](auto setBoundaryCondition) {
      auto const numDofIndicesBefore = actor->GetBoundaryConditionDofIndices().size();
      auto const numDofValuesBefore = actor->GetBoundaryConditionDofValuesWorld().size();

      Error error;
      setBoundaryCondition(error);
      EXPECT_FALSE(error.IsOK());
      EXPECT_STREQ(expectedError, error.GetDescription());
      EXPECT_EQ(numDofIndicesBefore, actor->GetBoundaryConditionDofIndices().size());
      EXPECT_EQ(numDofValuesBefore, actor->GetBoundaryConditionDofValuesWorld().size());
    };

    int const dofIndices[] = {0, 1, 2};
    int const nodeIndex = 0;
    Real3 const positionWorld{};

    expectRejected([&](Error& error) {
      actor->AddBoundaryConditionDofsWorld(MakeConstSpan(dofIndices), positionWorld, error);
    });
    expectRejected([&](Error& error) {
      actor->AddBoundaryConditionDofsWorldPermanent(
          MakeConstSpan(dofIndices), positionWorld, error);
    });
    expectRejected([&](Error& error) {
      actor->AddBoundaryConditionNodesWorld(
          MakeSingletonConstSpan(nodeIndex), positionWorld, error);
    });
    expectRejected([&](Error& error) {
      actor->AddBoundaryConditionNodesWorldPermanent(
          MakeSingletonConstSpan(nodeIndex), positionWorld, error);
    });
    expectRejected([&](Error& error) { actor->AddBoundaryConditionConstrainedNodesAtRest(error); });
    expectRejected(
        [&](Error& error) { actor->AddBoundaryConditionConstrainedNodesAtRestPermanent(error); });
  }

  ShapeHandle _cubeShape;
};

TEST_F(MochiActorBC, RigidComTranslation) {
  // Create minimal rigid cube actor
  RigidActorParams params;
  params.shape = _cubeShape;
  params.colliderType = ColliderType::Box;
  params.worldFromLocal.SetTranslation(Real3{});
  Actor* box = _scene->CreateRigidActor(params, test::ExpectOK{});
  Real3 center =
      box->GetRootTransform().TransformPoint(box->GetAabbLocal(test::ExpectOK{}).GetCenter());
  Real3 expectedCenter = 0.5_r * Real3{kBoxWidth, kBoxWidth, kBoxWidth};
  EXPECT_NEAR_EQ(expectedCenter, center);

  // Get the center-of-mass in world space. It is the first 3 DoFs.
  int const comIndices[] = {0, 1, 2};
  auto getCOM = [&comIndices](Actor* actor) {
    Real3 com;
    actor->GetDofValues(comIndices, com, test::ExpectOK{});
    return com;
  };
  Real3 com0 = getCOM(box);
  EXPECT_NEAR_EQ(com0, center);

  // Non-finite values are rejected without adding BCs
  box->AddBoundaryConditionDofsWorld(
      comIndices, Real3{0_r, std::numeric_limits<real>::quiet_NaN(), 0_r}, test::ExpectNotOK{});
  EXPECT_EQ(0, isize(box->GetBoundaryConditionDofIndices()));
  EXPECT_EQ(0, isize(box->GetBoundaryConditionDofValuesWorld()));

  // Add 3 BCs so the COM can't move
  box->AddBoundaryConditionDofsWorld(comIndices, com0, test::ExpectOK{});
  EXPECT_TRUE(test::EqualSpan(comIndices, box->GetBoundaryConditionDofIndices()));
  EXPECT_TRUE(test::NearEqualSpan(com0, box->GetBoundaryConditionDofValuesWorld()));

  // Simulate with a gravity force that will push on all 3 axes
  Real3 const kGravity = {10_r, 10_r, 10_r};
  _scene->SetGravity(kGravity);
  for (int i = 0; i < 5; ++i) {
    _scene->Step(0.05);
  }

  // Expect that the COM hasn't moved
  Real3 com1 = getCOM(box);
  EXPECT_NEAR_EQ(com0, com1); // no change

  // Now, remove the boundary condition on the Y axis only
  box->ClearBoundaryConditions();
  EXPECT_EQ(0, isize(box->GetBoundaryConditionDofIndices()));
  EXPECT_EQ(0, isize(box->GetBoundaryConditionDofValuesWorld()));
  int const dofIdxXZ[] = {0, 2};
  real const dofValXZ[] = {com1[0], com1[2]};
  box->AddBoundaryConditionDofsWorld(dofIdxXZ, dofValXZ, test::ExpectOK{});
  EXPECT_TRUE(test::EqualSpan(dofIdxXZ, box->GetBoundaryConditionDofIndices()));
  EXPECT_TRUE(test::NearEqualSpan(dofValXZ, box->GetBoundaryConditionDofValuesWorld()));

  // Simulate again
  constexpr real kTimeStep = 0.05_r;
  _scene->Step(kTimeStep);

  // Expect that only the Y component moved in the direction of gravity
  Real3 com2 = getCOM(box);
  Real3 expectedCom2 = com1 + Real3{0_r, Pow(kTimeStep, 2) * kGravity[1], 0_r};
  EXPECT_NEAR_EQ(expectedCom2, com2);

  // This time allow only X to move. Add the boundary conditions one at a time to prove we can.
  box->ClearBoundaryConditions();
  EXPECT_EQ(0, isize(box->GetBoundaryConditionDofIndices()));
  EXPECT_EQ(0, isize(box->GetBoundaryConditionDofValuesWorld()));
  int const one = 1;
  int const two = 2;
  box->AddBoundaryConditionDofsWorld(Span{&one, 1}, Span{&com2[1], 1}, test::ExpectOK{});
  box->AddBoundaryConditionDofsWorld(Span{&two, 1}, Span{&com2[2], 1}, test::ExpectOK{});

  // Simulate again
  _scene->Step(kTimeStep);

  // Expect that only the X component moved in the direction of gravity
  Real3 com3 = getCOM(box);
  Real3 expectedCom3 = com2 + Real3{Pow(kTimeStep, 2) * kGravity[1], 0_r, 0_r};
  EXPECT_NEAR_EQ(expectedCom3, com3);
}

TEST_F(MochiActorBC, DuplicateDofLastAppendedValueWins) {
  RigidActorParams params;
  params.shape = _cubeShape;
  params.colliderType = ColliderType::Box;
  Actor* box = _scene->CreateRigidActor(params, test::ExpectOK{});

  // Apply BCs to the same DoF twice with different target values.
  int const dof = 0;
  real const firstValue = 0.1_r;
  real const secondValue = 0.2_r;
  box->AddBoundaryConditionDofsWorld(
      MakeSingletonConstSpan(dof), MakeSingletonConstSpan(firstValue), test::ExpectOK{});
  box->AddBoundaryConditionDofsWorld(
      MakeSingletonConstSpan(dof), MakeSingletonConstSpan(secondValue), test::ExpectOK{});

  // Getters return both entries in append order, including the repeated index.
  int const expectedIndices[] = {dof, dof};
  real const expectedValues[] = {firstValue, secondValue};
  EXPECT_TRUE(test::EqualSpan(expectedIndices, box->GetBoundaryConditionDofIndices()));
  EXPECT_TRUE(test::NearEqualSpan(expectedValues, box->GetBoundaryConditionDofValuesWorld()));

  // The last appended value is the one enforced during the solve.
  _scene->Step(0.05_r);
  int const comIndices[] = {0, 1, 2};
  Real3 com;
  box->GetDofValues(comIndices, com, test::ExpectOK{});
  EXPECT_NEAR_EQ(secondValue, com[0]);
}

TEST_F(MochiActorBC, RigidPermanentRotationSurvivesClear) {
  RigidActorParams params;
  params.shape = _cubeShape;
  params.colliderType = ColliderType::Box;
  Actor* box = _scene->CreateRigidActor(params, test::ExpectOK{});

  DynamicArray<int> translationDofIndices = {0, 1, 2};
  DynamicArray<int> rotationDofIndices = {3, 4, 5};
  Real3 translationValues;
  box->GetDofValues(translationDofIndices, translationValues, test::ExpectOK{});
  Real3 rotationValues;
  box->GetDofValues(rotationDofIndices, rotationValues, test::ExpectOK{});

  DynamicArray<int> allDofIndices = {0, 1, 2, 3, 4, 5};
  DynamicArray<real> allDofValues = {
      translationValues[0],
      translationValues[1],
      translationValues[2],
      rotationValues[0],
      rotationValues[1],
      rotationValues[2],
  };
  box->AddBoundaryConditionDofsWorld(translationDofIndices, translationValues, test::ExpectOK{});
  box->AddBoundaryConditionDofsWorldPermanent(rotationDofIndices, rotationValues, test::ExpectOK{});
  EXPECT_TRUE(test::EqualSpan(allDofIndices, box->GetBoundaryConditionDofIndices()));
  EXPECT_TRUE(test::NearEqualSpan(allDofValues, box->GetBoundaryConditionDofValuesWorld()));

  box->ClearBoundaryConditions();
  EXPECT_TRUE(test::EqualSpan(rotationDofIndices, box->GetBoundaryConditionDofIndices()));
  EXPECT_TRUE(test::NearEqualSpan(rotationValues, box->GetBoundaryConditionDofValuesWorld()));

  box->ClearBoundaryConditions();
  EXPECT_TRUE(test::EqualSpan(rotationDofIndices, box->GetBoundaryConditionDofIndices()));
  EXPECT_TRUE(test::NearEqualSpan(rotationValues, box->GetBoundaryConditionDofValuesWorld()));

  DynamicArray<real> torqueValues = {0.001_r, 0.002_r, 0.003_r};
  box->SetExternalForcesOnDofs(rotationDofIndices, torqueValues, test::ExpectOK{});
  _scene->Step(0.01_r);

  Real3 rotationValuesAfterTorque;
  box->GetDofValues(rotationDofIndices, rotationValuesAfterTorque, test::ExpectOK{});
  EXPECT_NEAR_EQ(rotationValues, rotationValuesAfterTorque);

  // The permanent boundary conditions keep the scene from becoming differentiable.
  test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
  MakeSceneDifferentiableInternal(_scene, test::ExpectNotOK{});
}

TEST_F(MochiActorBC, MultiplePermanentRangesSurviveClear) {
  RigidActorParams params;
  params.shape = _cubeShape;
  params.colliderType = ColliderType::Box;
  Actor* box = _scene->CreateRigidActor(params, test::ExpectOK{});

  DynamicArray<int> const translationDofIndices = {0, 1, 2};
  DynamicArray<int> const rotationDofIndices = {3, 4, 5};
  Real3 translationValues;
  box->GetDofValues(translationDofIndices, translationValues, test::ExpectOK{});
  Real3 rotationValues;
  box->GetDofValues(rotationDofIndices, rotationValues, test::ExpectOK{});

  // A permanent range, a clearable entry, then another permanent range. The interleaved clearable
  // sits between the two permanent ranges, so compacting it out shifts the second range's begins --
  // exercising the remap rather than leaving offsets unchanged. Rotation contributes 3 DoFs but 4
  // pose (quaternion) entries, so a range's DoF and pose offsets diverge, exercising the
  // independent accumulation of dofBegin and poseBegin across loop iterations.
  int const clearableDof = 0;
  real const clearableValue = translationValues[0] + 1_r;
  box->AddBoundaryConditionDofsWorldPermanent(rotationDofIndices, rotationValues, test::ExpectOK{});
  box->AddBoundaryConditionDofsWorld(
      MakeSingletonSpan(clearableDof), MakeSingletonSpan(clearableValue), test::ExpectOK{});
  {
    // Expect warning due to BC count exceeding DoF count.
    test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
    box->AddBoundaryConditionDofsWorldPermanent(
        translationDofIndices, translationValues, test::ExpectOK{});
  }

  // After clearing, only the permanent ranges remain, in append order (rotation, then translation).
  DynamicArray<int> const permanentDofIndices = {3, 4, 5, 0, 1, 2};
  DynamicArray<real> const permanentDofValues = {
      rotationValues[0],
      rotationValues[1],
      rotationValues[2],
      translationValues[0],
      translationValues[1],
      translationValues[2],
  };
  box->ClearBoundaryConditions();
  EXPECT_TRUE(test::EqualSpan(permanentDofIndices, box->GetBoundaryConditionDofIndices()));
  EXPECT_TRUE(test::NearEqualSpan(permanentDofValues, box->GetBoundaryConditionDofValuesWorld()));

  // Re-add a clearable entry and clear again: the second compaction must use the remapped begins
  // stored by the first clear, not the original offsets.
  {
    // Expect warning due to BC count exceeding DoF count.
    test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
    box->AddBoundaryConditionDofsWorld(
        MakeSingletonSpan(clearableDof), MakeSingletonSpan(clearableValue), test::ExpectOK{});
  }
  box->ClearBoundaryConditions();
  EXPECT_TRUE(test::EqualSpan(permanentDofIndices, box->GetBoundaryConditionDofIndices()));
  EXPECT_TRUE(test::NearEqualSpan(permanentDofValues, box->GetBoundaryConditionDofValuesWorld()));

  // The remapped pose ranges are only observable through the solve: apply gravity and torque, then
  // confirm both permanent boundary conditions are still enforced after compaction.
  _scene->SetGravity(Real3{10_r, 10_r, 10_r});
  DynamicArray<real> const torqueValues = {0.001_r, 0.002_r, 0.003_r};
  box->SetExternalForcesOnDofs(rotationDofIndices, torqueValues, test::ExpectOK{});
  _scene->Step(0.05_r);

  Real3 translationAfterStep;
  box->GetDofValues(translationDofIndices, translationAfterStep, test::ExpectOK{});
  EXPECT_NEAR_EQ(translationValues, translationAfterStep);
  Real3 rotationAfterStep;
  box->GetDofValues(rotationDofIndices, rotationAfterStep, test::ExpectOK{});
  EXPECT_NEAR_EQ(rotationValues, rotationAfterStep);
}

TEST_F(MochiActorBC, SoftNodeTranslation) {
  // Create minimal soft cube actor with a generic rotation, far from the origin in x and y so that
  // mapping the world-space boundary values to the local frame is sensitive to roundoff of world
  // coordinates, and near it in z, where subtracting the translation in float would round.
  SoftActorParams params;
  params.shape = _cubeShape;
  params.worldFromLocal.SetRotation(Quaternion::FromRotationVector(Real3{0.3_r, -0.5_r, 0.7_r}));
  params.worldFromLocal.SetTranslation(Real3{100_r, -100_r, 0.05_r});
  Actor* actor = _scene->CreateSoftActor(params, test::ExpectOK{});
  actor->RegisterQuery(QueryType::NodePositions, test::ExpectOK{});

  // Disable recentering because it adds round-off error that makes it harder to compare values
  RecenteringParams recentering;
  recentering.useRecentering = false;
  actor->SetRecenteringParams(recentering, test::ExpectOK{});

  // Get the DoF indices (3 per node) and node positions in world-space
  auto shape = actor->GetReferenceShape(test::ExpectOK{});
  auto const& mesh = _mochiContext->GetShapeMesh(shape, test::ExpectOK{});
  int const numNodes = mesh.GetNumNodes();
  std::vector<int> nodeIndices(numNodes);
  std::iota(nodeIndices.begin(), nodeIndices.end(), 0);
  std::vector<int> dofIndices(numNodes * 3);
  std::iota(dofIndices.begin(), dofIndices.end(), 0);
  std::vector<Real3> startingWorldPositions(numNodes);
  for (int i = 0; i < numNodes; ++i) {
    auto pt = Load<3, Vec4r>(&mesh.coordinates[3 * i]);
    startingWorldPositions[i] = ToReal3(params.worldFromLocal.TransformPoint(pt));
  }

  // Expects the nodes displaced to the local positions of their world-space boundary values,
  // computed in double precision from the stored root translation and rotation matrix, up to the
  // rounding of the displacements themselves.
  auto expectAtBoundaryValues = [&](Span<int const> bcNodeIndices) {
    TransformRT const root = actor->GetRootTransform();
    auto const localFromWorldT = ToNdArray3x3(
        StaticCast<VMatrix3x3d>(ToVMatrix3x3Transpose(root.GetRotation().GetConjugate())));
    auto const refPositions = Unflatten<Real3 const>(mesh.coordinates);
    auto const displacements = Unflatten<Real3 const>(actor->GetDisplacements(test::ExpectOK{}));
    for (int i : bcNodeIndices) {
      Double3 const offsetWorld = StaticCast<Double3>(startingWorldPositions[i]) -
          StaticCast<Double3>(root.GetTranslation());
      Double3 const expected =
          DotVecMat(offsetWorld, localFromWorldT) - StaticCast<Double3>(refPositions[i]);
      double const tolerance = std::numeric_limits<real>::epsilon() * Norm(expected) +
          16.0 * std::numeric_limits<double>::epsilon() * kBoxWidth;
      EXPECT_LE(Norm(StaticCast<Double3>(displacements[i]) - expected), tolerance);
    }
  };

  // Freeze all nodes with world-space boundary conditions
  EXPECT_EQ(0, actor->GetBoundaryConditionDofIndices().size());
  EXPECT_EQ(0, actor->GetBoundaryConditionDofValuesWorld().size());
  actor->AddBoundaryConditionNodesWorld(
      nodeIndices, Flatten(MakeSpan(startingWorldPositions)), test::ExpectOK{});
  EXPECT_EQ(numNodes * 3, actor->GetBoundaryConditionDofIndices().size());
  EXPECT_EQ(numNodes * 3, actor->GetBoundaryConditionDofValuesWorld().size());
  EXPECT_TRUE(test::EqualSpan(dofIndices, actor->GetBoundaryConditionDofIndices()));
  EXPECT_TRUE(
      test::EqualSpan(
          Flatten(MakeSpan(startingWorldPositions)), actor->GetBoundaryConditionDofValuesWorld()));

  // Simulate with gravity
  Real3 const kGravity{0_r, -10_r, 0_r};
  _scene->SetGravity(kGravity);
  for (int i = 0; i < 5; ++i) {
    _scene->Step(0.05);
  }

  // Expect no movement
  EXPECT_NEAR_EQ(params.worldFromLocal, actor->GetRootTransform());
  expectAtBoundaryValues(nodeIndices);

  // Now, freeze one node (arbitrarily chosen).
  // This time, specify the DoF indices instead of the node indices.
  int const kBCNodeIndex = 1;
  actor->ClearBoundaryConditions();
  actor->AddBoundaryConditionDofsWorld(
      Span{&dofIndices[3 * kBCNodeIndex], 3},
      startingWorldPositions[kBCNodeIndex],
      test::ExpectOK{});
  EXPECT_EQ(3, actor->GetBoundaryConditionDofIndices().size());
  EXPECT_EQ(3, actor->GetBoundaryConditionDofValuesWorld().size());

  // Simulate again
  real const kTimeStep = 0.02_r;
  _scene->Step(kTimeStep);

  // Expect that node index 1 (the one with BCs) did not move
  expectAtBoundaryValues(MakeSingletonConstSpan(kBCNodeIndex));

  // Get the updated world-space positions
  auto newLocalPositions = Unflatten<Real3 const>(actor->GetNodePositionsLocal(test::ExpectOK{}));
  std::vector<Real3> newWorldPositions(newLocalPositions.size());
  for (int i = 0; i < isize(newLocalPositions); ++i) {
    newWorldPositions[i] = params.worldFromLocal.TransformPoint(newLocalPositions[i]);
  }

  // Expect that all other nodes moved in the direction of gravity (-Y)
  for (int i = 0; i < numNodes; ++i) {
    if (i != kBCNodeIndex) {
      EXPECT_GT(startingWorldPositions[i][1], newWorldPositions[i][1]); // Falling in Y direction
    }
  }
}

TEST_F(MochiActorBC, AddBoundaryConditionConstrainedNodesAtRest) {
  // Build ModelData with constrained nodes embedded in the shape
  auto tetMesh = test::CreateMinimalTetMeshUnitCube(Real3{kBoxWidth, kBoxWidth, kBoxWidth});
  auto& coordinates = tetMesh.first;
  auto& connectivity = tetMesh.second;

  ModelData model;
  model.mesh.emplace();
  model.mesh->nodesPerElement = 4;
  model.mesh->coordinates = Flatten(MakeSpan(coordinates));
  model.mesh->connectivity = Flatten(MakeSpan(connectivity));
  model.constrainedNodes = DynamicArray<int>{0, 2, 4, 6}; // nodes at x=0

  ShapeHandle modelShape = _mochiContext->CreateModelShape(model, test::ExpectOK{});

  // Create soft actor
  SoftActorParams params;
  params.shape = modelShape;
  params.worldFromLocal.SetRotation(Quaternion::FromRotationVector(Real3{0_r, 0_r, 0.5_r * kPI}));
  params.worldFromLocal.SetTranslation(Real3{0.1_r, 0.2_r, 0.3_r});
  Actor* actor = _scene->CreateSoftActor(params, test::ExpectOK{});
  actor->RegisterQuery(QueryType::NodePositions, test::ExpectOK{});

  RecenteringParams recentering;
  recentering.useRecentering = false;
  actor->SetRecenteringParams(recentering, test::ExpectOK{});

  // Expected DoF indices for constrained nodes {0, 2, 4, 6}: 3 DoFs per node
  int const expectedDofIndices[] = {0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20};
  int const expectedPermanentDofIndices[] = {0, 1, 2, 0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20};
  int const expectedNodeOneThenPermanentDofIndices[] = {
      3, 4, 5, 0, 1, 2, 0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20};
  int const expectedPermanentThenNodeOneDofIndices[] = {
      0, 1, 2, 0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 3, 4, 5};

  auto restPositionWorld = [&](int nodeIdx) {
    return params.worldFromLocal.TransformPoint(coordinates[nodeIdx]);
  };
  std::array<Real3, 4> const expectedConstrainedPositionsWorld = {
      restPositionWorld(0),
      restPositionWorld(2),
      restPositionWorld(4),
      restPositionWorld(6),
  };

  auto expectBoundaryConditionDofs = [&](Span<int const> expectedDofIndices) {
    EXPECT_EQ(expectedDofIndices.size(), actor->GetBoundaryConditionDofIndices().size());
    EXPECT_EQ(expectedDofIndices.size(), actor->GetBoundaryConditionDofValuesWorld().size());
    EXPECT_TRUE(test::EqualSpan(expectedDofIndices, actor->GetBoundaryConditionDofIndices()));
  };

  auto simulateAndVerifyConstraints = [&]() {
    Real3 const kGravity{0_r, -10_r, 0_r};
    _scene->SetGravity(kGravity);
    for (int i = 0; i < 5; ++i) {
      _scene->Step(0.05);
    }

    auto localPositions = Unflatten<Real3 const>(actor->GetNodePositionsLocal(test::ExpectOK{}));
    std::vector<Real3> worldPositions(localPositions.size());
    for (int i = 0; i < isize(localPositions); ++i) {
      worldPositions[i] = params.worldFromLocal.TransformPoint(localPositions[i]);
    }

    for (int nodeIdx : {0, 2, 4, 6}) {
      EXPECT_NEAR_EQ(restPositionWorld(nodeIdx), worldPositions[nodeIdx]);
    }
    for (int nodeIdx : {1, 3, 5, 7}) {
      EXPECT_GT(restPositionWorld(nodeIdx)[1], worldPositions[nodeIdx][1]);
    }
  };

  // Non-permanent: apply, verify, simulate, clear
  actor->AddBoundaryConditionConstrainedNodesAtRest(test::ExpectOK{});
  expectBoundaryConditionDofs(MakeConstSpan(expectedDofIndices));
  EXPECT_TRUE(
      test::NearEqualSpan(
          Flatten(MakeConstSpan(expectedConstrainedPositionsWorld)),
          actor->GetBoundaryConditionDofValuesWorld()));
  simulateAndVerifyConstraints();

  actor->ClearBoundaryConditions();
  expectBoundaryConditionDofs(Span<int const>{});

  // Non-permanent followed by permanent: clear should remove only the first batch.
  int const nonPermanentNode = 1;
  actor->AddBoundaryConditionNodesWorld(
      MakeSingletonConstSpan(nonPermanentNode),
      restPositionWorld(nonPermanentNode),
      test::ExpectOK{});
  actor->AddBoundaryConditionNodesWorldPermanent(
      MakeSingletonConstSpan(0), restPositionWorld(0), test::ExpectOK{});
  actor->AddBoundaryConditionConstrainedNodesAtRestPermanent(test::ExpectOK{});
  expectBoundaryConditionDofs(MakeConstSpan(expectedNodeOneThenPermanentDofIndices));

  actor->ClearBoundaryConditions();
  expectBoundaryConditionDofs(MakeConstSpan(expectedPermanentDofIndices));

  // Permanent followed by non-permanent: clear should still remove only the last batch.
  actor->AddBoundaryConditionNodesWorld(
      MakeSingletonConstSpan(nonPermanentNode),
      restPositionWorld(nonPermanentNode),
      test::ExpectOK{});
  expectBoundaryConditionDofs(MakeConstSpan(expectedPermanentThenNodeOneDofIndices));

  actor->ClearBoundaryConditions();
  expectBoundaryConditionDofs(MakeConstSpan(expectedPermanentDofIndices));

  simulateAndVerifyConstraints();
}

TEST_F(MochiActorBC, BoundaryConditionSettersRejectNestedActors) {
  SoftSkinnedActorParams params;
  auto& skeleton = params.skeletonParams;
  skeleton.joints = {{.type = ArticulatedJointType::Free}};
  skeleton.links = {{.parentLink = -1, .shape = _cubeShape, .colliderType = ColliderType::None}};

  auto& softParams = params.softParams.push_back();
  softParams.shape = test::CreateUnitCubeTetSoftShape(_mochiContext);
  softParams.hasGravity = false;

  Actor* actor = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
  ASSERT_NE(nullptr, actor);
  auto const& nestedSoftActors = actor->GetNestedSoftActors(test::ExpectOK{});
  ASSERT_EQ(1, isize(nestedSoftActors));
  Actor* nestedSoftActor = _scene->GetActor(nestedSoftActors[0]);
  ASSERT_NE(nullptr, nestedSoftActor);
  EXPECT_TRUE(nestedSoftActor->IsNestedSoftActor());
  Actor* nestedLinkActor = _scene->GetActor(actor->GetNestedLinkActors(test::ExpectOK{})[0]);
  ASSERT_NE(nullptr, nestedLinkActor);
  EXPECT_TRUE(nestedLinkActor->IsNestedLinkActor());

  ExpectBoundaryConditionSettersRejected(
      nestedSoftActor, "Boundary condition setters are not supported for nested soft actors.");
  ExpectBoundaryConditionSettersRejected(
      nestedLinkActor,
      "Boundary condition setters are not supported for nested link actors. Set boundary conditions on the articulated actor instead.");
}

TEST_F(MochiActorBC, BoundaryConditionSettersRejectRodActors) {
  Actor* rod = test::AddRod(_scene);
  ASSERT_NE(nullptr, rod);

  ExpectBoundaryConditionSettersRejected(
      rod, "Boundary conditions are not supported for rod actors. Use constraints instead.");
}

TEST_F(MochiActorBC, BoundaryConditionSettersRejectPartialRotations) {
  RigidActorParams rigidParams;
  rigidParams.shape = _cubeShape;
  rigidParams.colliderType = ColliderType::Box;
  Actor* rigidActor = _scene->CreateRigidActor(rigidParams, test::ExpectOK{});

  ArticulatedActorParams articulatedParams;
  articulatedParams.joints = {{.type = ArticulatedJointType::Free}};
  articulatedParams.links = {{.shape = _cubeShape, .colliderType = ColliderType::None}};
  Actor* articulatedActor = _scene->CreateArticulatedActor(articulatedParams, test::ExpectOK{});

  // DoFs 3-5 are the rotation of both actors. Constraining only some of them is rejected and adds
  // nothing.
  int const partialRotationDofIndices[] = {3, 4};
  real const partialRotationValues[] = {0_r, 0_r};
  for (Actor* actor : {rigidActor, articulatedActor}) {
    Error error;
    actor->AddBoundaryConditionDofsWorld(partialRotationDofIndices, partialRotationValues, error);
    EXPECT_STREQ("Rotation BCs must be all or none.", error.GetDescription());
    actor->AddBoundaryConditionDofsWorldPermanent(
        partialRotationDofIndices, partialRotationValues, test::ExpectNotOK{});
    EXPECT_EQ(0, isize(actor->GetBoundaryConditionDofIndices()));
    EXPECT_EQ(0, isize(actor->GetBoundaryConditionDofValuesWorld()));
  }
}

namespace {
// AMG is the only preconditioner that writes into the fixed rows of the Newton increment: it builds
// its coarse levels from the sparsity pattern alone, which keeps the couplings that boundary
// conditions zero. Direct solvers and the other preconditioners leave those rows zero.
constexpr PreconditionerType kPreconditionerWritingIntoFixedRows = PreconditionerType::AMG;

class MochiActorBCFixedDofs : public MochiActorBC,
                              public ::testing::WithParamInterface<ActorType> {};
} // namespace

TEST_P(MochiActorBCFixedDofs, FixedDofsStayAtBoundaryValues) {
  ActorType const actorType = GetParam();
  auto solverParams = _scene->GetSolverParams();
  // Small islands default to a direct solver, which runs no preconditioner.
  solverParams.linearSolver.solverType = LinearSolverType::CG;
  solverParams.linearSolver.preconditionerType = kPreconditionerWritingIntoFixedRows;
  _scene->SetSolverParams(solverParams, test::ExpectOK{});

  // A rigid actor created first and attached to the constrained actor comes first in their island.
  // Its 6 DoFs take 7 pose entries, so the constrained actor's DoF offset is nonzero and differs
  // from its pose offset: zeroing that ignores the DoF offset or uses the pose offset hits the
  // wrong rows. It has no collider, since the two actors overlap.
  RigidActorParams firstActorParams;
  firstActorParams.shape = _cubeShape;
  firstActorParams.colliderType = ColliderType::None;
  Actor* firstActor = _scene->CreateRigidActor(firstActorParams, test::ExpectOK{});

  Actor* actor = nullptr;
  if (actorType == ActorType::Soft) {
    SoftActorParams params;
    params.shape = _cubeShape;
    actor = _scene->CreateSoftActor(params, test::ExpectOK{});
    // Recentering would move the soft actor's local frame, and with it every DoF value.
    actor->SetRecenteringParams({.useRecentering = false}, test::ExpectOK{});
  } else {
    ASSERT_EQ(ActorType::Shell, actorType);
    auto [coordinates, connectivity] =
        test::CreateMinimalTriMeshUnitCube(Real3{kBoxWidth, kBoxWidth, kBoxWidth});
    experimental::ShellActorParams params;
    params.shape = _mochiContext->CreateTriMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});
    actor = experimental::CreateShellActor(_scene, params, test::ExpectOK{});
  }
  DeformableNodeToRigidConstraintParams attachment;
  attachment.rigidActor = firstActor->GetHandle();
  attachment.deformableActor = actor->GetHandle();
  attachment.deformableNodeIndex = 0;
  _scene->CreateDeformableNodeToRigidConstraint(attachment, test::ExpectOK{});

  // Fixes node 0 at rest, so its DoFs must keep their values exactly. Gravity moves the other DoFs.
  // Node 0's DoF values are displacements, but its boundary values are world positions: with the
  // identity root transform, its reference position.
  DynamicArray<int> const fixedDofs = {0, 1, 2};
  auto const mesh =
      _mochiContext->GetShapeMesh(actor->GetReferenceShape(test::ExpectOK{}), test::ExpectOK{});
  DynamicArray<real> const boundaryValues = {
      mesh.coordinates[0], mesh.coordinates[1], mesh.coordinates[2]};
  actor->AddBoundaryConditionDofsWorld(fixedDofs, boundaryValues, test::ExpectOK{});

  auto const getDofValues = [actor] {
    DynamicArray<real> values(actor->GetNumDofs());
    actor->GetDofValues({}, values, test::ExpectOK{});
    return values;
  };
  DynamicArray<real> const restDofValues = getDofValues();
  _scene->Step(0.01);
  DynamicArray<real> const dofValues = getDofValues();
  for (int i : fixedDofs) {
    EXPECT_EQ(restDofValues[i], dofValues[i]);
  }
  // The free DoFs moved, so the solve had nonzero increments.
  EXPECT_NE(restDofValues, dofValues);
  // The first actor put the constrained actor's DoFs after its own, at an offset that differs from
  // the pose offset.
  auto const& offset = GetRegistry().get<CDofOffset const>(GetEntity(actor));
  EXPECT_LT(0, offset.dofsOffset);
  EXPECT_NE(offset.poseOffset, offset.dofsOffset);
}

// Soft and shell actors add the increment to their state. Rigid and articulated actors reset
// their fixed DoFs after every increment, so their state can't show a leaked increment, and nested
// soft actors apply it through the same path as soft actors.
INSTANTIATE_TEST_SUITE_P(
    AddIncrementToState,
    MochiActorBCFixedDofs,
    ::testing::Values(ActorType::Soft, ActorType::Shell),
    [](::testing::TestParamInfo<ActorType> const& info) {
      return std::string(SReflect::EnumToString(info.param));
    });
