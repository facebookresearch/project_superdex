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

#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using namespace mochi;

/**************************************************************************************
  TransmissionTest - Test fixture for transmission functionality at the mochi_physics
  API level.
**************************************************************************************/

class TransmissionTest : public test::MochiSceneTestBase {
 protected:
  ShapeHandle _cubeShape;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();
    auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
    _cubeShape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});
  }

  // Create an articulated actor with the given joint types, connected in a chain topology.
  // All joints use the x-axis. Root joint has a translation offset; child joints connect at
  // identity.
  Actor* CreateArticulatedActorChain(Span<ArticulatedJointType const> joints) {
    auto const numBones = isize(joints);
    TransformRT const offset{Real3{1_r, 1_r, 1_r}};

    ArticulatedActorParams params;
    params.joints.resize(numBones);
    params.links.resize(numBones);
    for (int i = 0; i < numBones; ++i) {
      params.joints[i].type = joints[i];
      params.joints[i].axis = Real3{1_r, 0_r, 0_r};
      if (i == 0) {
        params.joints[i].parentLinkFromJoint = offset;
      }
      params.links[i].parentLink = i - 1;
      params.links[i].shape = _cubeShape;
    }
    return _scene->CreateArticulatedActor(params, test::ExpectOK{});
  }
};

/**************************************************************************************
  Tests for experimental::GetTransmissionDisplacement
**************************************************************************************/

TEST_F(TransmissionTest, GetTransmissionDisplacement_ZeroAtRestPose) {
  DynamicArray<ArticulatedJointType> const joints(3, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);
  ASSERT_EQ(actor->GetNumDofs(), 3);

  experimental::LinearTransmissionParams transmissionParams;
  transmissionParams.jointIndices = {0, 1, 2};
  transmissionParams.jointCoefficients = {0.5_r, -0.3_r, 0.2_r};
  int const transmissionIndex =
      experimental::AddLinearTransmission(actor, transmissionParams, test::ExpectOK{});
  EXPECT_EQ(transmissionIndex, 0);

  // At rest pose (all DOFs zero), displacement must be zero
  real const displacement =
      experimental::GetTransmissionDisplacement(actor, transmissionIndex, test::ExpectOK{});
  EXPECT_NEAR(displacement, 0_r, kTolerance);
}

TEST_F(TransmissionTest, GetTransmissionDisplacement_SingleJoint) {
  DynamicArray<ArticulatedJointType> const joints(2, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);
  ASSERT_EQ(actor->GetNumDofs(), 2);

  real constexpr kCoefficient = 0.4_r;
  experimental::LinearTransmissionParams transmissionParams;
  transmissionParams.jointIndices = {1};
  transmissionParams.jointCoefficients = {kCoefficient};
  int const transmissionIndex =
      experimental::AddLinearTransmission(actor, transmissionParams, test::ExpectOK{});

  // Set a non-zero angle on joint 1
  real constexpr kAngle = 0.7_r;
  DynamicArray<real> const pose = {0_r, kAngle};
  actor->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});

  real const displacement =
      experimental::GetTransmissionDisplacement(actor, transmissionIndex, test::ExpectOK{});
  EXPECT_NEAR(displacement, kAngle * kCoefficient, kTolerance);
}

TEST_F(TransmissionTest, GetTransmissionDisplacement_MultipleJoints) {
  DynamicArray<ArticulatedJointType> const joints(3, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);
  ASSERT_EQ(actor->GetNumDofs(), 3);

  real constexpr kC0 = 0.5_r, kC1 = 0.3_r, kC2 = 0.2_r;
  experimental::LinearTransmissionParams transmissionParams;
  transmissionParams.jointIndices = {0, 1, 2};
  transmissionParams.jointCoefficients = {kC0, kC1, -kC2};
  int const transmissionIndex =
      experimental::AddLinearTransmission(actor, transmissionParams, test::ExpectOK{});

  real constexpr kA0 = 0.2_r, kA1 = -0.4_r, kA2 = 0.6_r;
  DynamicArray<real> const pose = {kA0, kA1, kA2};
  actor->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});

  // Expected: sum of angle * signedCoefficient
  real constexpr kExpected = kA0 * kC0 + kA1 * kC1 + kA2 * (-kC2);
  real const displacement =
      experimental::GetTransmissionDisplacement(actor, transmissionIndex, test::ExpectOK{});
  EXPECT_NEAR(displacement, kExpected, kTolerance);
}

TEST_F(TransmissionTest, GetTransmissionDisplacement_AlignmentFlagEffect) {
  DynamicArray<ArticulatedJointType> const joints(2, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);
  ASSERT_EQ(actor->GetNumDofs(), 2);

  // Two transmissions on the same joint with opposite alignment flags
  real constexpr kCoefficient = 0.5_r;

  experimental::LinearTransmissionParams paramsAligned;
  paramsAligned.jointIndices = {1};
  paramsAligned.jointCoefficients = {kCoefficient};
  int const transmissionAligned =
      experimental::AddLinearTransmission(actor, paramsAligned, test::ExpectOK{});

  experimental::LinearTransmissionParams paramsOpposite;
  paramsOpposite.jointIndices = {1};
  paramsOpposite.jointCoefficients = {-kCoefficient};
  int const transmissionOpposite =
      experimental::AddLinearTransmission(actor, paramsOpposite, test::ExpectOK{});

  real constexpr kAngle = 0.3_r;
  DynamicArray<real> const pose = {0_r, kAngle};
  actor->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});

  real const posAligned =
      experimental::GetTransmissionDisplacement(actor, transmissionAligned, test::ExpectOK{});
  real const posOpposite =
      experimental::GetTransmissionDisplacement(actor, transmissionOpposite, test::ExpectOK{});

  EXPECT_NEAR(posAligned, kAngle * kCoefficient, kTolerance);
  EXPECT_NEAR(posOpposite, -(kAngle * kCoefficient), kTolerance);
}

TEST_F(TransmissionTest, GetTransmissionDisplacement_MultipleTransmissions) {
  DynamicArray<ArticulatedJointType> const joints(3, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);
  ASSERT_EQ(actor->GetNumDofs(), 3);

  // First transmission on joints 0 and 1
  real constexpr kC0a = 0.5_r, kC1a = 0.3_r;
  experimental::LinearTransmissionParams params0;
  params0.jointIndices = {0, 1};
  params0.jointCoefficients = {kC0a, kC1a};
  int const transmission0 = experimental::AddLinearTransmission(actor, params0, test::ExpectOK{});

  // Second transmission on joints 1 and 2
  real constexpr kC1b = 0.4_r, kC2b = 0.6_r;
  experimental::LinearTransmissionParams params1;
  params1.jointIndices = {1, 2};
  params1.jointCoefficients = {-kC1b, kC2b};
  int const transmission1 = experimental::AddLinearTransmission(actor, params1, test::ExpectOK{});

  EXPECT_EQ(transmission0, 0);
  EXPECT_EQ(transmission1, 1);

  real constexpr kA0 = 0.1_r, kA1 = 0.2_r, kA2 = -0.3_r;
  DynamicArray<real> const pose = {kA0, kA1, kA2};
  actor->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});

  // Each transmission's displacement is computed independently
  real constexpr kExpected0 = kA0 * kC0a + kA1 * kC1a;
  real constexpr kExpected1 = kA1 * (-kC1b) + kA2 * kC2b;

  EXPECT_NEAR(
      experimental::GetTransmissionDisplacement(actor, transmission0, test::ExpectOK{}),
      kExpected0,
      kTolerance);
  EXPECT_NEAR(
      experimental::GetTransmissionDisplacement(actor, transmission1, test::ExpectOK{}),
      kExpected1,
      kTolerance);
}

TEST_F(TransmissionTest, GetTransmissionDisplacement_UpdatesWithPose) {
  DynamicArray<ArticulatedJointType> const joints(2, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);
  ASSERT_EQ(actor->GetNumDofs(), 2);

  real constexpr kCoefficient = 0.5_r;
  experimental::LinearTransmissionParams transmissionParams;
  transmissionParams.jointIndices = {1};
  transmissionParams.jointCoefficients = {kCoefficient};
  int const transmissionIndex =
      experimental::AddLinearTransmission(actor, transmissionParams, test::ExpectOK{});

  // First pose
  real constexpr kAngle1 = 0.3_r;
  DynamicArray<real> const pose1 = {0_r, kAngle1};
  actor->SetArticulatedPoseFromJoints(pose1, test::ExpectOK{});
  EXPECT_NEAR(
      experimental::GetTransmissionDisplacement(actor, transmissionIndex, test::ExpectOK{}),
      kAngle1 * kCoefficient,
      kTolerance);

  // Update to a different pose
  real constexpr kAngle2 = -0.5_r;
  DynamicArray<real> const pose2 = {0_r, kAngle2};
  actor->SetArticulatedPoseFromJoints(pose2, test::ExpectOK{});
  EXPECT_NEAR(
      experimental::GetTransmissionDisplacement(actor, transmissionIndex, test::ExpectOK{}),
      kAngle2 * kCoefficient,
      kTolerance);
}

/**************************************************************************************
  Tests for experimental::GetTransmissionDisplacementJacobian
**************************************************************************************/

TEST_F(TransmissionTest, GetTransmissionDisplacementJacobian_LinearPlacementAndErrors) {
  DynamicArray<ArticulatedJointType> const joints(3, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);

  experimental::LinearTransmissionParams params;
  params.jointIndices = {2, 0, 2};
  params.jointCoefficients = {0.4_r, -0.3_r, 0.6_r};
  int const transmissionIndex =
      experimental::AddLinearTransmission(actor, params, test::ExpectOK{});

  DynamicArray<real> jacobian(actor->GetNumDofs(), 911_r);
  experimental::GetTransmissionDisplacementJacobian(
      actor, transmissionIndex, MakeSpan(jacobian), test::ExpectOK{});
  DynamicArray<real> const expected = {-0.3_r, 0_r, 1_r};
  EXPECT_SPAN_EQ(MakeConstSpan(jacobian), MakeConstSpan(expected));

  DynamicArray<real> wrongSize(actor->GetNumDofs() - 1);
  experimental::GetTransmissionDisplacementJacobian(
      actor, transmissionIndex, MakeSpan(wrongSize), test::ExpectNotOK{});
  experimental::GetTransmissionDisplacementJacobian(
      actor, -1, MakeSpan(jacobian), test::ExpectNotOK{});
  experimental::GetTransmissionDisplacementJacobian(
      actor, transmissionIndex + 1, MakeSpan(jacobian), test::ExpectNotOK{});
}

TEST_F(TransmissionTest, GetTransmissionDisplacementJacobian_UsesReducedDofSize) {
  DynamicArray<ArticulatedJointType> const joints(2, ArticulatedJointType::Spherical);
  Actor* actor = CreateArticulatedActorChain(joints);
  ASSERT_EQ(actor->GetNumDofs(), 6);

  experimental::SpatialTendonParams params;
  params.routingElements = {
      {.type = RoutingElementType::Waypoint, .index = 0, .localPosition = Real3{0_r, 0.2_r, 0_r}},
      {.type = RoutingElementType::Waypoint, .index = 1, .localPosition = Real3{0_r, 0_r, 0.2_r}}};
  int const transmissionIndex = experimental::AddSpatialTendon(actor, params, test::ExpectOK{});

  DynamicArray<real> reducedDofJacobian(actor->GetNumDofs());
  experimental::GetTransmissionDisplacementJacobian(
      actor, transmissionIndex, MakeSpan(reducedDofJacobian), test::ExpectOK{});
  DynamicArray<real> poseSizedJacobian(8);
  experimental::GetTransmissionDisplacementJacobian(
      actor, transmissionIndex, MakeSpan(poseSizedJacobian), test::ExpectNotOK{});
}

TEST_F(TransmissionTest, GetTransmissionDisplacementJacobian_SpatialUsesCurrentPose) {
  DynamicArray<ArticulatedJointType> const joints(3, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);
  experimental::SpatialTendonParams params;
  params.routingElements = {
      {.type = RoutingElementType::Waypoint, .index = 0, .localPosition = Real3{0_r, 0.2_r, 0_r}},
      {.type = RoutingElementType::Waypoint, .index = 1, .localPosition = Real3{0_r, 0.2_r, 0.1_r}},
      {.type = RoutingElementType::Waypoint, .index = 2, .localPosition = Real3{0_r, 0_r, 0.2_r}}};
  int const transmissionIndex = experimental::AddSpatialTendon(actor, params, test::ExpectOK{});

  DynamicArray<real> jacobianAtRest(actor->GetNumDofs());
  experimental::GetTransmissionDisplacementJacobian(
      actor, transmissionIndex, MakeSpan(jacobianAtRest), test::ExpectOK{});

  DynamicArray<real> const pose = {0.3_r, 0.4_r, -0.5_r};
  actor->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});
  DynamicArray<real> jacobianAtPose(actor->GetNumDofs(), 911_r);
  experimental::GetTransmissionDisplacementJacobian(
      actor, transmissionIndex, MakeSpan(jacobianAtPose), test::ExpectOK{});

  bool changed = false;
  for (int i = 0; i < actor->GetNumDofs(); ++i) {
    changed |= !NearEqual(jacobianAtRest[i], jacobianAtPose[i], kTolerance);
    EXPECT_NE(jacobianAtPose[i], 911_r);
  }
  EXPECT_TRUE(changed);
}

/**************************************************************************************
  Error-path tests for experimental::GetTransmissionDisplacement
**************************************************************************************/

TEST_F(TransmissionTest, GetTransmissionDisplacement_InvalidTransmissionIndex) {
  DynamicArray<ArticulatedJointType> const joints(2, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);

  experimental::LinearTransmissionParams transmissionParams;
  transmissionParams.jointIndices = {0};
  transmissionParams.jointCoefficients = {0.5_r};
  experimental::AddLinearTransmission(actor, transmissionParams, test::ExpectOK{});

  // Negative index
  (void)experimental::GetTransmissionDisplacement(actor, -1, test::ExpectNotOK{});

  // Out-of-range index (only transmission 0 exists)
  (void)experimental::GetTransmissionDisplacement(actor, 1, test::ExpectNotOK{});
}

TEST_F(TransmissionTest, GetTransmissionDisplacement_NoTransmissionsAdded) {
  DynamicArray<ArticulatedJointType> const joints(2, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);

  // No transmissions have been added; any index should fail
  (void)experimental::GetTransmissionDisplacement(actor, 0, test::ExpectNotOK{});
}

TEST_F(TransmissionTest, GetTransmissionDisplacement_NonArticulatedActor) {
  RigidActorParams rParams;
  rParams.shape = _cubeShape;
  rParams.colliderType = ColliderType::Box;
  Actor* actor = _scene->CreateRigidActor(rParams, test::ExpectOK{});

  // Transmissions are only supported for articulated actors
  (void)experimental::GetTransmissionDisplacement(actor, 0, test::ExpectNotOK{});
}

/**************************************************************************************
  Tests for transmission actuator state variables
  (GetNumTransmissionActuatorStateVariables, SetTransmissionActuatorStateVariables,
   GetTransmissionActuatorStateVariables)
**************************************************************************************/

TEST_F(TransmissionTest, StateVariables_DisplacementControlActuatorRoundtrip) {
  DynamicArray<ArticulatedJointType> const joints(3, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);

  experimental::LinearTransmissionParams transmissionParams;
  transmissionParams.jointIndices = {0, 1, 2};
  transmissionParams.jointCoefficients = {0.5_r, -0.3_r, 0.2_r};
  int const transmissionIndex =
      experimental::AddLinearTransmission(actor, transmissionParams, test::ExpectOK{});

  experimental::DisplacementControlActuatorParams actuatorParams;
  actuatorParams.targetDisplacement = 0_r;
  actuatorParams.stiffness = 1e7_r;
  experimental::AttachDisplacementControlActuator(
      actor, transmissionIndex, actuatorParams, test::ExpectOK{});

  // Displacement-control actuator has exactly 1 state variable (the target displacement)
  int const numStateVars = experimental::GetNumTransmissionActuatorStateVariables(
      actor, transmissionIndex, test::ExpectOK{});
  EXPECT_EQ(numStateVars, 1);

  // Set a target displacement
  real constexpr kTargetDisplacement = -0.25_r;
  DynamicArray<real> const inStateVars = {kTargetDisplacement};
  experimental::SetTransmissionActuatorStateVariables(
      actor, transmissionIndex, MakeSpan(inStateVars), test::ExpectOK{});

  // Read it back
  DynamicArray<real> outStateVars(numStateVars);
  experimental::GetTransmissionActuatorStateVariables(
      actor, transmissionIndex, MakeSpan(outStateVars), test::ExpectOK{});
  EXPECT_NEAR(outStateVars[0], kTargetDisplacement, kTolerance);
}

TEST_F(TransmissionTest, StateVariables_InitialValueMatchesParams) {
  DynamicArray<ArticulatedJointType> const joints(2, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);

  experimental::LinearTransmissionParams transmissionParams;
  transmissionParams.jointIndices = {1};
  transmissionParams.jointCoefficients = {0.4_r};
  int const transmissionIndex =
      experimental::AddLinearTransmission(actor, transmissionParams, test::ExpectOK{});

  // Attach actuator with a non-zero initial target displacement
  real constexpr kInitialTarget = 0.42_r;
  experimental::DisplacementControlActuatorParams actuatorParams;
  actuatorParams.targetDisplacement = kInitialTarget;
  actuatorParams.stiffness = 1e7_r;
  experimental::AttachDisplacementControlActuator(
      actor, transmissionIndex, actuatorParams, test::ExpectOK{});

  // State variable should reflect the initial target from params
  DynamicArray<real> outStateVars(1);
  experimental::GetTransmissionActuatorStateVariables(
      actor, transmissionIndex, MakeSpan(outStateVars), test::ExpectOK{});
  EXPECT_NEAR(outStateVars[0], kInitialTarget, kTolerance);
}

TEST_F(TransmissionTest, StateVariables_NoActuatorAttached) {
  DynamicArray<ArticulatedJointType> const joints(2, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);

  experimental::LinearTransmissionParams transmissionParams;
  transmissionParams.jointIndices = {0};
  transmissionParams.jointCoefficients = {0.5_r};
  int const transmissionIndex =
      experimental::AddLinearTransmission(actor, transmissionParams, test::ExpectOK{});

  // No actuator attached

  // No actuator attached — all state variable operations should fail
  (void)experimental::GetNumTransmissionActuatorStateVariables(
      actor, transmissionIndex, test::ExpectNotOK{});

  DynamicArray<real> stateVars(1, 0_r);
  experimental::SetTransmissionActuatorStateVariables(
      actor, transmissionIndex, MakeSpan(stateVars), test::ExpectNotOK{});
  experimental::GetTransmissionActuatorStateVariables(
      actor, transmissionIndex, MakeSpan(stateVars), test::ExpectNotOK{});
}

TEST_F(TransmissionTest, StateVariables_InvalidTransmissionIndex) {
  DynamicArray<ArticulatedJointType> const joints(2, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);

  experimental::LinearTransmissionParams transmissionParams;
  transmissionParams.jointIndices = {0};
  transmissionParams.jointCoefficients = {0.5_r};
  experimental::AddLinearTransmission(actor, transmissionParams, test::ExpectOK{});

  // Invalid transmission indices
  (void)experimental::GetNumTransmissionActuatorStateVariables(actor, -1, test::ExpectNotOK{});
  (void)experimental::GetNumTransmissionActuatorStateVariables(actor, 1, test::ExpectNotOK{});

  DynamicArray<real> stateVars(1, 0_r);
  experimental::SetTransmissionActuatorStateVariables(
      actor, -1, MakeSpan(stateVars), test::ExpectNotOK{});
  experimental::GetTransmissionActuatorStateVariables(
      actor, 1, MakeSpan(stateVars), test::ExpectNotOK{});
}

TEST_F(TransmissionTest, StateVariables_NonArticulatedActor) {
  RigidActorParams rParams;
  rParams.shape = _cubeShape;
  rParams.colliderType = ColliderType::Box;
  Actor* actor = _scene->CreateRigidActor(rParams, test::ExpectOK{});

  // State variable operations are only supported for articulated actors
  (void)experimental::GetNumTransmissionActuatorStateVariables(actor, 0, test::ExpectNotOK{});

  DynamicArray<real> stateVars(1, 0_r);
  experimental::SetTransmissionActuatorStateVariables(
      actor, 0, MakeSpan(stateVars), test::ExpectNotOK{});
  experimental::GetTransmissionActuatorStateVariables(
      actor, 0, MakeSpan(stateVars), test::ExpectNotOK{});
}

/**************************************************************************************
  Tests for experimental::AddSpatialTendon
**************************************************************************************/

namespace {
// Waypoints offset in the y/z plane so that rotating the chain's x-axis revolute joints changes
// the routed length. Builds waypoint elements on links 0, 1, 2.
DynamicArray<RoutingElement> SampleSpatialTendonRoutingElements() {
  return {
      {.type = RoutingElementType::Waypoint, .index = 0, .localPosition = Real3{0_r, 0.2_r, 0_r}},
      {.type = RoutingElementType::Waypoint, .index = 1, .localPosition = Real3{0_r, 0.2_r, 0.1_r}},
      {.type = RoutingElementType::Waypoint, .index = 2, .localPosition = Real3{0_r, 0_r, 0.2_r}}};
}
} // namespace

TEST_F(TransmissionTest, AddSpatialTendon_DisplacementZeroAtRest) {
  DynamicArray<ArticulatedJointType> const joints(3, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);

  experimental::SpatialTendonParams params;
  params.routingElements = SampleSpatialTendonRoutingElements();
  int const transmissionIndex = experimental::AddSpatialTendon(actor, params, test::ExpectOK{});
  EXPECT_EQ(transmissionIndex, 0);

  // The displacement is the routed-length change relative to the rest pose, so it is zero at rest.
  real const displacement =
      experimental::GetTransmissionDisplacement(actor, transmissionIndex, test::ExpectOK{});
  EXPECT_NEAR(displacement, 0_r, kTolerance);
}

TEST_F(TransmissionTest, AddSpatialTendon_DisplacementChangesWithPose) {
  DynamicArray<ArticulatedJointType> const joints(3, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);

  experimental::SpatialTendonParams params;
  params.routingElements = SampleSpatialTendonRoutingElements();
  int const transmissionIndex = experimental::AddSpatialTendon(actor, params, test::ExpectOK{});

  DynamicArray<real> const pose = {0.3_r, 0.4_r, -0.5_r};
  actor->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});
  real const displacement =
      experimental::GetTransmissionDisplacement(actor, transmissionIndex, test::ExpectOK{});
  EXPECT_GT(std::abs(displacement), kTolerance);
}

TEST_F(TransmissionTest, AddSpatialTendon_MixedElementsDisplacement) {
  // A mixed list: the routed length plus a linear-joint element. At a generic pose the reported
  // displacement must equal the pure-waypoint displacement plus coefficient * jointAngle.
  DynamicArray<ArticulatedJointType> const joints(3, ArticulatedJointType::Revolute);
  Actor* waypointActor = CreateArticulatedActorChain(joints);
  Actor* mixedActor = CreateArticulatedActorChain(joints);

  real const coefficient = 0.5_r;
  int const linearJoint = 2;

  experimental::SpatialTendonParams waypointParams;
  waypointParams.routingElements = SampleSpatialTendonRoutingElements();
  int const waypointIndex =
      experimental::AddSpatialTendon(waypointActor, waypointParams, test::ExpectOK{});

  experimental::SpatialTendonParams mixedParams;
  mixedParams.routingElements = SampleSpatialTendonRoutingElements();
  mixedParams.routingElements.push_back(
      {.type = RoutingElementType::LinearJoint, .index = linearJoint, .coefficient = coefficient});
  int const mixedIndex = experimental::AddSpatialTendon(mixedActor, mixedParams, test::ExpectOK{});

  DynamicArray<real> const pose = {0.3_r, 0.4_r, -0.5_r};
  waypointActor->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});
  mixedActor->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});

  real const waypointDisplacement =
      experimental::GetTransmissionDisplacement(waypointActor, waypointIndex, test::ExpectOK{});
  real const mixedDisplacement =
      experimental::GetTransmissionDisplacement(mixedActor, mixedIndex, test::ExpectOK{});
  EXPECT_NEAR(
      mixedDisplacement, waypointDisplacement + coefficient * pose[linearJoint], kTolerance);
}

TEST_F(TransmissionTest, AddSpatialTendon_TensionShortensRoutedPath) {
  // Isolate the tendon force from gravity so the routed-length change is unambiguous.
  _scene->SetGravity(Real3{0_r, 0_r, 0_r});
  DynamicArray<ArticulatedJointType> const joints(3, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);

  experimental::SpatialTendonParams params;
  params.routingElements = SampleSpatialTendonRoutingElements();
  int const transmissionIndex = experimental::AddSpatialTendon(actor, params, test::ExpectOK{});

  // Start from a generic (non-rest) pose so the length gradient is non-degenerate.
  DynamicArray<real> const pose = {0.3_r, 0.3_r, 0.3_r};
  actor->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});
  real const initialDisplacement =
      experimental::GetTransmissionDisplacement(actor, transmissionIndex, test::ExpectOK{});

  // A displacement-control actuator with a target below the current length applies tension that
  // pulls the routed path shorter.
  experimental::DisplacementControlActuatorParams actuatorParams;
  actuatorParams.targetDisplacement = initialDisplacement - 0.2_r;
  actuatorParams.stiffness = 1e4_r;
  experimental::AttachDisplacementControlActuator(
      actor, transmissionIndex, actuatorParams, test::ExpectOK{});

  for (int i = 0; i < 20; ++i) {
    _scene->Step(0.01);
  }
  real const finalDisplacement =
      experimental::GetTransmissionDisplacement(actor, transmissionIndex, test::ExpectOK{});
  EXPECT_LT(finalDisplacement, initialDisplacement - kTolerance);
}

TEST_F(TransmissionTest, AddSpatialTendon_RejectsEmptyElements) {
  DynamicArray<ArticulatedJointType> const joints(3, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);

  experimental::SpatialTendonParams params; // No elements.
  (void)experimental::AddSpatialTendon(actor, params, test::ExpectNotOK{});
}

TEST_F(TransmissionTest, AddSpatialTendon_RejectsInvalidLinkIndex) {
  DynamicArray<ArticulatedJointType> const joints(2, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);

  experimental::SpatialTendonParams params;
  params.routingElements = {
      {.type = RoutingElementType::Waypoint, .index = 0, .localPosition = Real3{0_r, 0.2_r, 0_r}},
      {.type = RoutingElementType::Waypoint, .index = 5, .localPosition = Real3{0_r, 0.2_r, 0_r}}};
  (void)experimental::AddSpatialTendon(actor, params, test::ExpectNotOK{});
}

TEST_F(TransmissionTest, AddSpatialTendon_RejectsNonFiniteWaypointPosition) {
  DynamicArray<ArticulatedJointType> const joints(2, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);

  real const kNaN = std::numeric_limits<real>::quiet_NaN();
  experimental::SpatialTendonParams params;
  params.routingElements = {
      {.type = RoutingElementType::Waypoint, .index = 0, .localPosition = Real3{0_r, 0.2_r, 0_r}},
      {.type = RoutingElementType::Waypoint, .index = 1, .localPosition = Real3{0_r, kNaN, 0_r}}};
  (void)experimental::AddSpatialTendon(actor, params, test::ExpectNotOK{});
}

TEST_F(TransmissionTest, AddSpatialTendon_RejectsInvalidLinearJointIndex) {
  DynamicArray<ArticulatedJointType> const joints(2, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);

  experimental::SpatialTendonParams params;
  params.routingElements = {
      {.type = RoutingElementType::Waypoint, .index = 0, .localPosition = Real3{0_r, 0.2_r, 0_r}},
      {.type = RoutingElementType::LinearJoint, .index = 5, .coefficient = 0.1_r}}; // No joint 5.
  (void)experimental::AddSpatialTendon(actor, params, test::ExpectNotOK{});
}

TEST_F(TransmissionTest, AddSpatialTendon_RejectsMultiDofLinearJoint) {
  // Linear-joint elements require single-DoF joints; a spherical joint (3 DoFs) must be rejected.
  DynamicArray<ArticulatedJointType> const joints = {
      ArticulatedJointType::Revolute, ArticulatedJointType::Spherical};
  Actor* actor = CreateArticulatedActorChain(joints);

  experimental::SpatialTendonParams params;
  params.routingElements = {
      {.type = RoutingElementType::Waypoint, .index = 0, .localPosition = Real3{0_r, 0.2_r, 0_r}},
      {.type = RoutingElementType::LinearJoint,
       .index = 1,
       .coefficient = 0.1_r}}; // Spherical joint.
  (void)experimental::AddSpatialTendon(actor, params, test::ExpectNotOK{});
}

TEST_F(TransmissionTest, AddSpatialTendon_RejectsIsolatedWaypoint) {
  // A waypoint with a linear-joint element on each side forms no segment, so it must be rejected.
  DynamicArray<ArticulatedJointType> const joints(3, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);

  experimental::SpatialTendonParams params;
  params.routingElements = {
      {.type = RoutingElementType::LinearJoint, .index = 0, .coefficient = 0.1_r},
      {.type = RoutingElementType::Waypoint, .index = 1, .localPosition = Real3{0_r, 0.2_r, 0_r}},
      {.type = RoutingElementType::LinearJoint, .index = 2, .coefficient = 0.1_r}};
  (void)experimental::AddSpatialTendon(actor, params, test::ExpectNotOK{});
}

TEST_F(TransmissionTest, AddSpatialTendon_WarnsOnLinearJointAdjacentToUnrelatedWaypoint) {
  // A linear-joint element adjacent to a waypoint whose link is neither the joint's parent nor
  // its child is accepted but should log a warning. Chain link indices: 0,1,2; joint 1 connects
  // links 0 (parent) and 1 (child). A waypoint on link 2 next to linear-joint 1 trips the warning.
  DynamicArray<ArticulatedJointType> const joints(3, ArticulatedJointType::Revolute);
  Actor* actor = CreateArticulatedActorChain(joints);

  experimental::SpatialTendonParams params;
  params.routingElements = {
      {.type = RoutingElementType::Waypoint, .index = 2, .localPosition = Real3{0_r, 0.2_r, 0_r}},
      {.type = RoutingElementType::Waypoint, .index = 2, .localPosition = Real3{0_r, 0.2_r, 0.1_r}},
      {.type = RoutingElementType::LinearJoint, .index = 1, .coefficient = 0.1_r}};
  {
    test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
    (void)experimental::AddSpatialTendon(actor, params, test::ExpectOK{});
  }
}

TEST_F(TransmissionTest, AddSpatialTendon_NonArticulatedActor) {
  RigidActorParams rParams;
  rParams.shape = _cubeShape;
  rParams.colliderType = ColliderType::Box;
  Actor* actor = _scene->CreateRigidActor(rParams, test::ExpectOK{});

  experimental::SpatialTendonParams params;
  params.routingElements = {
      {.type = RoutingElementType::Waypoint, .index = 0, .localPosition = Real3{0_r, 0.2_r, 0_r}},
      {.type = RoutingElementType::Waypoint, .index = 1, .localPosition = Real3{0_r, 0.2_r, 0_r}}};
  (void)experimental::AddSpatialTendon(actor, params, test::ExpectNotOK{});
}
