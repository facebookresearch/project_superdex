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

// TODO[T259529001]: C API coming soon...
#if 0

#include <gtest/gtest.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_physics/mochi_physics_capi.generated.h>
#include <mochi_physics/src/mochi_scene.h>

#include <cmath>
#include <vector>

using namespace mochi;

// Helper macro for checking CAPI errors with descriptive output
#define EXPECT_CAPI_OK(error)                             \
  if ((error) != MochiError_OK) {                         \
    const char* desc = Mochi_Error_GetDescription(error); \
    ADD_FAILURE() << Format(                              \
        "%s(%d) CAPI Error: %s",                          \
        __FILE__,                                         \
        __LINE__,                                         \
        (desc ? desc : "(no description available)"));    \
  }

// Helper functions for creating test meshes
namespace {
std::vector<MochiReal> CreateSmallCubeTetMeshCoords() {
  return {
      -0.1f, -0.1f, -0.1f, // 0
      +0.1f, -0.1f, -0.1f, // 1
      -0.1f, +0.1f, -0.1f, // 2
      +0.1f, +0.1f, -0.1f, // 3
      -0.1f, -0.1f, +0.1f, // 4
      +0.1f, -0.1f, +0.1f, // 5
      -0.1f, +0.1f, +0.1f, // 6
      +0.1f, +0.1f, +0.1f, // 7
  };
}

std::vector<int> CreateSmallCubeTetMeshConnectivity() {
  return {
      0, 1, 2, 4, // corner vert 0
      6, 7, 4, 2, // corner vert 6
      5, 4, 7, 1, // corner vert 5
      3, 2, 1, 7, // corner vert 3
      1, 2, 4, 7, // interior one
  };
}
} // namespace

//==================================================================================================
// Integration Tests - End-to-End Workflows
//==================================================================================================

TEST(CAPI_Integration, SimpleRigidBodyFalling) {
  MochiError error = MochiError_OK;

  // Create context and scene
  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  MochiSceneHandle scene = Mochi_CreateScene(context, "FallingBallScene", &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_DestroyScene(context, scene));

  // Set gravity
  MochiReal3 gravity = {0.0f, -9.8f, 0.0f};
  Mochi_Scene_SetGravity(scene, gravity, &error);
  EXPECT_CAPI_OK(error);

  // Create sphere at height y=10
  MochiReal3 center = MochiReal3_GetDefault();
  center.data[1] = 10.0f;
  MochiShapeHandle shape = Mochi_CreateSphereShape(context, center, 1.0f, &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_ReleaseShape(context, shape));

  // Create rigid actor (static for now, dynamic requires complex setup)
  MochiRigidActorParams params = MochiRigidActorParams_GetDefault();
  params.shape = shape;
  params.isStatic = true;
  params.colliderType = MochiColliderType_Sphere;

  MochiActorHandle actor = Mochi_Scene_CreateRigidActor(scene, params, &error);
  EXPECT_CAPI_OK(error);
  EXPECT_TRUE(Mochi_Actor_IsValid(scene, actor));

  for (int i = 0; i < 10; ++i) {
    Mochi_Scene_Step(scene, 0.01, &error);
    EXPECT_CAPI_OK(error);
  }
}

TEST(CAPI_Integration, DeformableNodePositionConstraint) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  MochiSceneHandle scene = Mochi_CreateScene(context, "DeformableConstraintScene", &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_DestroyScene(context, scene));

  // Create tet mesh shape
  auto coords = CreateSmallCubeTetMeshCoords();
  auto connectivity = CreateSmallCubeTetMeshConnectivity();
  MochiConstRealSpan coordsSpan = {coords.data(), coords.size()};
  MochiConstIntSpan connectivitySpan = {connectivity.data(), connectivity.size()};
  MochiShapeHandle shape = Mochi_CreateTetMeshShape(context, coordsSpan, connectivitySpan, &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_ReleaseShape(context, shape));

  // Create soft actor
  MochiSoftActorParams params = MochiSoftActorParams_GetDefault();
  params.shape = shape;
  MochiActorHandle actor = Mochi_Scene_CreateSoftActor(scene, params, &error);
  EXPECT_CAPI_OK(error);

  // Create a position constraint on node 0
  MochiDeformableNodePositionConstraintParams constraintParams =
      MochiDeformableNodePositionConstraintParams_GetDefault();
  constraintParams.actor = actor;
  constraintParams.nodeIndex = 0;
  constraintParams.position.data[0] = 0.0f;
  constraintParams.position.data[1] = 1.0f; // Pin node to y=1
  constraintParams.position.data[2] = 0.0f;

  MochiConstraintHandle constraint =
      Mochi_Scene_CreateDeformableNodePositionConstraint(scene, constraintParams, &error);
  EXPECT_CAPI_OK(error);
  EXPECT_NE(static_cast<MochiConstraintHandle>(MOCHI_INVALID_HANDLE), constraint);

  // Verify constraint is valid
  EXPECT_TRUE(Mochi_Constraint_IsValid(scene, constraint));

  // Verify constraint type
  MochiConstraintType type = Mochi_Constraint_GetType(scene, constraint, &error);
  EXPECT_CAPI_OK(error);
  EXPECT_EQ(MochiConstraintType_DeformableNodePosition, type);

  // Step simulation
  Mochi_Scene_Step(scene, 0.01, &error);
  EXPECT_CAPI_OK(error);
}

TEST(CAPI_Integration, MultipleActorsInteracting) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  MochiSceneHandle scene = Mochi_CreateScene(context, "MultiActorScene", &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_DestroyScene(context, scene));

  // Set gravity
  MochiReal3 gravity = {0.0f, -9.8f, 0.0f};
  Mochi_Scene_SetGravity(scene, gravity, &error);
  EXPECT_CAPI_OK(error);

  // Create ground plane (static)
  MochiReal3 planeNormal = MochiReal3_GetDefault();
  planeNormal.data[1] = 1.0f; // Normal pointing up
  MochiReal planeDistance = 0.0f;
  MochiShapeHandle planeShape = Mochi_CreatePlaneShape(context, planeNormal, planeDistance, &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_ReleaseShape(context, planeShape));

  MochiRigidActorParams planeParams = MochiRigidActorParams_GetDefault();
  planeParams.shape = planeShape;
  planeParams.isStatic = true;
  MochiActorHandle actor = Mochi_Scene_CreateRigidActor(scene, planeParams, &error);
  EXPECT_CAPI_OK(error);
  EXPECT_TRUE(Mochi_Actor_IsValid(scene, actor));

  // Create falling sphere
  MochiReal3 sphereCenter = MochiReal3_GetDefault();
  sphereCenter.data[1] = 5.0f;
  MochiShapeHandle sphereShape = Mochi_CreateSphereShape(context, sphereCenter, 0.5f, &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_ReleaseShape(context, sphereShape));

  MochiRigidActorParams sphereParams = MochiRigidActorParams_GetDefault();
  sphereParams.shape = sphereShape;
  sphereParams.isStatic = true; // Use static for now
  sphereParams.colliderType = MochiColliderType_Sphere;
  MochiActorHandle sphereActor = Mochi_Scene_CreateRigidActor(scene, sphereParams, &error);
  EXPECT_CAPI_OK(error);

  // Simulate for multiple steps
  for (int i = 0; i < 100; ++i) {
    Mochi_Scene_Step(scene, 0.01, &error);
    EXPECT_CAPI_OK(error);
  }

  // Verify sphere has settled on or near the plane
  MochiTransformRT finalTransform = Mochi_Actor_GetRootTransform(scene, sphereActor, &error);
  EXPECT_CAPI_OK(error);

  // Sphere should have fallen and be close to y=0 (accounting for sphere radius)
  EXPECT_GT(finalTransform.translation.data[1], -1.0f);
  EXPECT_LT(finalTransform.translation.data[1], 2.0f);
}

TEST(CAPI_Integration, ActorContactParameters) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  MochiSceneHandle scene = Mochi_CreateScene(context, "ContactParamsScene", &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_DestroyScene(context, scene));

  // Create sphere
  MochiReal3 center = MochiReal3_GetDefault();
  MochiShapeHandle shape = Mochi_CreateSphereShape(context, center, 1.0f, &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_ReleaseShape(context, shape));

  // Create rigid actor
  MochiRigidActorParams params = MochiRigidActorParams_GetDefault();
  params.shape = shape;
  params.isStatic = true; // Use static for now
  params.colliderType = MochiColliderType_Sphere;
  MochiActorHandle actor = Mochi_Scene_CreateRigidActor(scene, params, &error);
  EXPECT_CAPI_OK(error);

  // Set contact parameters
  MochiContactParams contactParams = MochiContactParams_GetDefault();
  contactParams.coulombFrictionCoefficient = 0.5f;

  Mochi_Actor_SetContactParams(scene, actor, contactParams, &error);
  EXPECT_CAPI_OK(error);

  // Step simulation
  Mochi_Scene_Step(scene, 0.01, &error);
  EXPECT_CAPI_OK(error);
}

//==================================================================================================
// Dynamic Rigid Body Tests
//==================================================================================================

TEST(CAPI_Integration, DynamicRigidBodyFalling) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  MochiSceneHandle scene = Mochi_CreateScene(context, "DynamicFallingScene", &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_DestroyScene(context, scene));

  // Set gravity
  MochiReal3 gravity = MochiReal3_GetDefault();
  gravity.data[1] = -9.8f;
  Mochi_Scene_SetGravity(scene, gravity, &error);
  EXPECT_CAPI_OK(error);

  // Create a tet mesh cube - needed for automatic mass property computation
  auto coords = CreateSmallCubeTetMeshCoords();
  auto connectivity = CreateSmallCubeTetMeshConnectivity();
  MochiConstRealSpan coordsSpan = {coords.data(), coords.size()};
  MochiConstIntSpan connectivitySpan = {connectivity.data(), connectivity.size()};
  MochiShapeHandle shape = Mochi_CreateTetMeshShape(context, coordsSpan, connectivitySpan, &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_ReleaseShape(context, shape));

  // Set initial position at y=5
  MochiTransformRT worldFromLocal = MochiTransformRT_GetDefault();
  worldFromLocal.translation.data[1] = 5.0f;

  // Create DYNAMIC rigid actor - system will compute mass properties automatically from tet mesh
  MochiRigidActorParams params = MochiRigidActorParams_GetDefault();
  params.shape = shape;
  params.worldFromLocal = worldFromLocal;
  params.isStatic = false; // Dynamic!
  params.colliderType = MochiColliderType_Box; // Box collider works with tet mesh
  params.hasGravity = true;
  MochiActorHandle actor = Mochi_Scene_CreateRigidActor(scene, params, &error);
  EXPECT_CAPI_OK(error);

  // Get initial position
  MochiTransformRT initialTransform = Mochi_Actor_GetRootTransform(scene, actor, &error);
  EXPECT_CAPI_OK(error);
  MochiReal initialY = initialTransform.translation.data[1];
  EXPECT_NEAR(initialY, 5.0f, 0.2f);

  // Simulate falling for 1 second (100 steps of 0.01s)
  for (int i = 0; i < 100; ++i) {
    Mochi_Scene_Step(scene, 0.01, &error);
    EXPECT_CAPI_OK(error);
  }

  // Verify cube has fallen (y position should be lower)
  MochiTransformRT finalTransform = Mochi_Actor_GetRootTransform(scene, actor, &error);
  EXPECT_CAPI_OK(error);
  MochiReal finalY = finalTransform.translation.data[1];

  // Cube should have fallen significantly under gravity
  EXPECT_LT(finalY, initialY - 2.0f);
}

TEST(CAPI_Integration, DynamicRigidBodyCollisionWithPlane) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  MochiSceneHandle scene = Mochi_CreateScene(context, "DynamicCollisionScene", &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_DestroyScene(context, scene));

  // Set gravity
  MochiReal3 gravity = MochiReal3_GetDefault();
  gravity.data[1] = -9.8f;
  Mochi_Scene_SetGravity(scene, gravity, &error);
  EXPECT_CAPI_OK(error);

  // Create ground plane (static)
  MochiReal3 planeNormal = MochiReal3_GetDefault();
  planeNormal.data[1] = 1.0f; // Normal pointing up
  MochiReal planeDistance = 0.0f;
  MochiShapeHandle planeShape = Mochi_CreatePlaneShape(context, planeNormal, planeDistance, &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_ReleaseShape(context, planeShape));

  MochiRigidActorParams planeParams = MochiRigidActorParams_GetDefault();
  planeParams.shape = planeShape;
  planeParams.isStatic = true;
  planeParams.colliderType = MochiColliderType_Plane;
  Mochi_Scene_CreateRigidActor(scene, planeParams, &error);
  EXPECT_CAPI_OK(error);

  // Create falling cube (dynamic) using tet mesh
  auto coords = CreateSmallCubeTetMeshCoords();
  auto connectivity = CreateSmallCubeTetMeshConnectivity();
  MochiConstRealSpan cubeCoords = {coords.data(), coords.size()};
  MochiConstIntSpan cubeConn = {connectivity.data(), connectivity.size()};
  MochiShapeHandle cubeShape = Mochi_CreateTetMeshShape(context, cubeCoords, cubeConn, &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_ReleaseShape(context, cubeShape));

  // Set initial position at y=3
  MochiTransformRT worldFromLocal = MochiTransformRT_GetDefault();
  worldFromLocal.translation.data[1] = 3.0f;

  MochiRigidActorParams cubeParams = MochiRigidActorParams_GetDefault();
  cubeParams.shape = cubeShape;
  cubeParams.worldFromLocal = worldFromLocal;
  cubeParams.isStatic = false; // Dynamic!
  cubeParams.colliderType = MochiColliderType_Box;
  cubeParams.hasGravity = true;
  MochiActorHandle cubeActor = Mochi_Scene_CreateRigidActor(scene, cubeParams, &error);
  EXPECT_CAPI_OK(error);

  // Settling onto the plane relies on backward Euler's damping.
  MochiSolverParams solverParams = Mochi_Scene_GetSolverParams(scene, &error);
  EXPECT_CAPI_OK(error);
  solverParams.integrationMethod = MochiIntegrationMethod_BackwardEuler;
  Mochi_Scene_SetSolverParams(scene, solverParams, &error);
  EXPECT_CAPI_OK(error);

  // Simulate for 2 seconds to let cube fall and settle on plane
  for (int i = 0; i < 200; ++i) {
    Mochi_Scene_Step(scene, 0.01, &error);
    EXPECT_CAPI_OK(error);
  }

  // Verify cube has settled on or near the plane
  MochiTransformRT finalTransform = Mochi_Actor_GetRootTransform(scene, cubeActor, &error);
  EXPECT_CAPI_OK(error);

  // Cube center should be approximately at half-height (0.1) above ground plane
  EXPECT_GT(finalTransform.translation.data[1], -0.5f);
  EXPECT_LT(finalTransform.translation.data[1], 1.0f);
}

#endif // 0
