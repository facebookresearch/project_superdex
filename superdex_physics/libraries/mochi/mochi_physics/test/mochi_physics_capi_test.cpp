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

// Helper function to create a minimal tet mesh cube for testing
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

std::vector<MochiReal> CreateSmallCubeTriMeshCoords() {
  return CreateSmallCubeTetMeshCoords();
}

std::vector<int> CreateSmallCubeTriMeshConnectivity() {
  return {
      0, 2, 1, // back
      2, 3, 1, // back
      1, 3, 5, // right
      3, 7, 5, // right
      5, 7, 4, // front
      7, 6, 4, // front
      4, 6, 0, // left
      6, 2, 0, // left
      2, 6, 3, // top
      6, 7, 3, // top
      0, 1, 4, // bottom
      4, 1, 5, // bottom
  };
}
} // namespace

//==================================================================================================
// Data Types Tests
//==================================================================================================

TEST(CAPI, Real2) {
  // Default initialization
  MochiReal2 v = MochiReal2_GetDefault();
  EXPECT_EQ(v.data[0], 0.0f);
  EXPECT_EQ(v.data[1], 0.0f);

  // Value assignment
  v.data[0] = 1.5f;
  v.data[1] = 2.5f;
  EXPECT_EQ(v.data[0], 1.5f);
  EXPECT_EQ(v.data[1], 2.5f);
}

TEST(CAPI, Real3) {
  // Default initialization
  MochiReal3 v = MochiReal3_GetDefault();
  EXPECT_EQ(v.data[0], 0.0f);
  EXPECT_EQ(v.data[1], 0.0f);
  EXPECT_EQ(v.data[2], 0.0f);

  // Value assignment
  v.data[0] = 1.0f;
  v.data[1] = 2.0f;
  v.data[2] = 3.0f;
  EXPECT_EQ(v.data[0], 1.0f);
  EXPECT_EQ(v.data[1], 2.0f);
  EXPECT_EQ(v.data[2], 3.0f);
}

TEST(CAPI, Real6) {
  // Default initialization
  MochiReal6 v = MochiReal6_GetDefault();
  for (MochiReal& i : v.data) {
    EXPECT_EQ(i, 0.0f);
  }

  // Value assignment
  for (int i = 0; i < 6; ++i) {
    v.data[i] = static_cast<MochiReal>(i + 1);
  }
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(v.data[i], static_cast<MochiReal>(i + 1));
  }
}

TEST(CAPI, Quaternion) {
  // Default initialization (identity quaternion: x, y, z, w)
  MochiQuaternion q = MochiQuaternion_GetDefault();
  EXPECT_EQ(q.data[0], 0.0f); // x
  EXPECT_EQ(q.data[1], 0.0f); // y
  EXPECT_EQ(q.data[2], 0.0f); // z
  EXPECT_EQ(q.data[3], 1.0f); // w

  // Value assignment
  q.data[0] = 0.1f;
  q.data[1] = 0.2f;
  q.data[2] = 0.3f;
  q.data[3] = 0.4f;
  EXPECT_NEAR(q.data[0], 0.1f, 1e-6f);
  EXPECT_NEAR(q.data[1], 0.2f, 1e-6f);
  EXPECT_NEAR(q.data[2], 0.3f, 1e-6f);
  EXPECT_NEAR(q.data[3], 0.4f, 1e-6f);
}

TEST(CAPI, TransformRT) {
  // Default initialization
  MochiTransformRT transform = MochiTransformRT_GetDefault();
  EXPECT_EQ(transform.rotation.data[0], 0.0f);
  EXPECT_EQ(transform.rotation.data[1], 0.0f);
  EXPECT_EQ(transform.rotation.data[2], 0.0f);
  EXPECT_EQ(transform.rotation.data[3], 1.0f);
  EXPECT_EQ(transform.translation.data[0], 0.0f);
  EXPECT_EQ(transform.translation.data[1], 0.0f);
  EXPECT_EQ(transform.translation.data[2], 0.0f);

  // Value assignment
  transform.translation.data[0] = 1.0f;
  transform.translation.data[1] = 2.0f;
  transform.translation.data[2] = 3.0f;
  EXPECT_EQ(transform.translation.data[0], 1.0f);
  EXPECT_EQ(transform.translation.data[1], 2.0f);
  EXPECT_EQ(transform.translation.data[2], 3.0f);
}

//==================================================================================================
// Error Handling Tests
//==================================================================================================

TEST(CAPI, ErrorHandling) {
  MochiError error = MochiError_OK;
  EXPECT_EQ(error, MochiError_OK);

  // Attempt to create a scene with invalid context should set error
  MochiSceneHandle invalidScene = Mochi_CreateScene(MOCHI_INVALID_HANDLE, "InvalidScene", &error);
  EXPECT_NE(error, MochiError_OK);
  EXPECT_EQ(invalidScene, static_cast<MochiSceneHandle>(MOCHI_INVALID_HANDLE));

  // Verify error description is available
  char const* description = Mochi_Error_GetDescription(error);

  EXPECT_NE(description, nullptr);
  EXPECT_EQ(strlen("Invalid context handle"), strlen(description));

  MochiContextHandle context = Mochi_CreateContext(0);
  EXPECT_NE(context, static_cast<MochiContextHandle>(MOCHI_INVALID_HANDLE));
  MOCHI_DEFER(Mochi_DestroyContext(context));

  // Ensure we don't get a scene if Error is still set.
  MochiSceneHandle scene = Mochi_CreateScene(context, "TestScene", &error);
  EXPECT_NE(error, MochiError_OK);
  EXPECT_EQ(scene, static_cast<MochiSceneHandle>(MOCHI_INVALID_HANDLE));

  // Verify error description is available
  char const* description2 = Mochi_Error_GetDescription(error);

  EXPECT_NE(description2, nullptr);
  EXPECT_EQ(strlen("Invalid context handle"), strlen(description2));
}

//==================================================================================================
// Context and Scene Lifecycle Tests
//==================================================================================================

TEST(CAPI, ContextLifecycle) {
  // Create context
  MochiContextHandle context = Mochi_CreateContext(0);
  EXPECT_NE(context, static_cast<MochiContextHandle>(MOCHI_INVALID_HANDLE));

  // Destroy context
  Mochi_DestroyContext(context);
}

TEST(CAPI, SceneLifecycle) {
  MochiError error = MochiError_OK;

  // Create context
  MochiContextHandle context = Mochi_CreateContext(0);
  EXPECT_NE(context, static_cast<MochiContextHandle>(MOCHI_INVALID_HANDLE));
  MOCHI_DEFER(Mochi_DestroyContext(context));

  // Create scene
  MochiSceneHandle scene = Mochi_CreateScene(context, "TestScene", &error);
  EXPECT_CAPI_OK(error);
  EXPECT_NE(scene, static_cast<MochiSceneHandle>(MOCHI_INVALID_HANDLE));

  // Destroy scene
  Mochi_DestroyScene(context, scene);
}

TEST(CAPI, SceneGravity) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  MochiSceneHandle scene = Mochi_CreateScene(context, "TestScene", &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_DestroyScene(context, scene));

  // Set gravity
  MochiReal3 gravity = {0.0f, -9.8f, 0.0f};
  Mochi_Scene_SetGravity(scene, gravity, &error);
  EXPECT_CAPI_OK(error);

  // Get gravity
  MochiReal3 retrievedGravity = Mochi_Scene_GetGravity(scene, &error);
  EXPECT_CAPI_OK(error);
  EXPECT_NEAR(retrievedGravity.data[0], 0.0f, 1e-6f);
  EXPECT_NEAR(retrievedGravity.data[1], -9.8f, 1e-6f);
  EXPECT_NEAR(retrievedGravity.data[2], 0.0f, 1e-6f);
}

//==================================================================================================
// Shape Creation Tests
//==================================================================================================

TEST(CAPI, CreateSphereShape) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  EXPECT_EQ(0, Mochi_GetNumShapes(context, &error));
  EXPECT_CAPI_OK(error);

  // Create sphere shape
  MochiReal3 center = MochiReal3_GetDefault();
  center.data[0] = 0.0f;
  center.data[1] = 0.0f;
  center.data[2] = 0.0f;
  MochiReal radius = 1.0f;

  MochiShapeHandle shape = Mochi_CreateSphereShape(context, center, radius, &error);
  EXPECT_EQ(1, Mochi_GetNumShapes(context, &error));
  EXPECT_CAPI_OK(error);
  EXPECT_NE(shape, static_cast<MochiShapeHandle>(MOCHI_INVALID_HANDLE));

  // Release shape
  Mochi_ReleaseShape(context, shape);
  EXPECT_EQ(0, Mochi_GetNumShapes(context, &error));
  EXPECT_CAPI_OK(error);
}

TEST(CAPI, CreatePlaneShape) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  EXPECT_EQ(0, Mochi_GetNumShapes(context, &error));
  EXPECT_CAPI_OK(error);

  // Create plane shape
  MochiReal3 normal;
  normal.data[0] = 0.0f;
  normal.data[1] = 1.0f;
  normal.data[2] = 0.0f;
  MochiReal distance = 0.0f;

  MochiShapeHandle shape = Mochi_CreatePlaneShape(context, normal, distance, &error);
  EXPECT_EQ(1, Mochi_GetNumShapes(context, &error));
  EXPECT_CAPI_OK(error);
  EXPECT_NE(shape, static_cast<MochiShapeHandle>(MOCHI_INVALID_HANDLE));

  Mochi_ReleaseShape(context, shape);
  EXPECT_EQ(0, Mochi_GetNumShapes(context, &error));
  EXPECT_CAPI_OK(error);
}

TEST(CAPI, CreateTetMeshShape) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  EXPECT_EQ(0, Mochi_GetNumShapes(context, &error));
  EXPECT_CAPI_OK(error);

  // Create tet mesh shape
  auto coords = CreateSmallCubeTetMeshCoords();
  auto connectivity = CreateSmallCubeTetMeshConnectivity();

  MochiConstRealSpan coordsSpan = {coords.data(), coords.size()};
  MochiConstIntSpan connectivitySpan = {connectivity.data(), connectivity.size()};
  MochiShapeHandle shape = Mochi_CreateTetMeshShape(context, coordsSpan, connectivitySpan, &error);
  EXPECT_EQ(1, Mochi_GetNumShapes(context, &error));
  EXPECT_CAPI_OK(error);
  EXPECT_NE(shape, static_cast<MochiShapeHandle>(MOCHI_INVALID_HANDLE));

  Mochi_ReleaseShape(context, shape);
  EXPECT_EQ(0, Mochi_GetNumShapes(context, &error));
  EXPECT_CAPI_OK(error);
}

TEST(CAPI, CreateTriMeshShape) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  EXPECT_EQ(0, Mochi_GetNumShapes(context, &error));
  EXPECT_CAPI_OK(error);

  // Create tri mesh shape
  auto coords = CreateSmallCubeTriMeshCoords();
  auto connectivity = CreateSmallCubeTriMeshConnectivity();

  MochiConstRealSpan coordsSpan = {coords.data(), coords.size()};
  MochiConstIntSpan connectivitySpan = {connectivity.data(), connectivity.size()};
  MochiShapeHandle shape = Mochi_CreateTriMeshShape(context, coordsSpan, connectivitySpan, &error);
  EXPECT_EQ(1, Mochi_GetNumShapes(context, &error));
  EXPECT_CAPI_OK(error);
  EXPECT_NE(shape, static_cast<MochiShapeHandle>(MOCHI_INVALID_HANDLE));

  Mochi_ReleaseShape(context, shape);
  EXPECT_EQ(0, Mochi_GetNumShapes(context, &error));
  EXPECT_CAPI_OK(error);
}

//==================================================================================================
// Actor Creation Tests
//==================================================================================================

TEST(CAPI, CreateRigidActor) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  MochiSceneHandle scene = Mochi_CreateScene(context, "TestScene", &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_DestroyScene(context, scene));

  // Create sphere shape
  MochiReal3 center = MochiReal3_GetDefault();
  MochiShapeHandle shape = Mochi_CreateSphereShape(context, center, 1.0f, &error);
  EXPECT_CAPI_OK(error);

  // Create rigid actor
  MochiRigidActorParams params = MochiRigidActorParams_GetDefault();
  params.shape = shape;
  params.isStatic = true; // Use static for simplicity in CAPI testing
  params.colliderType = MochiColliderType_Sphere;

  MochiActorHandle actor = Mochi_Scene_CreateRigidActor(scene, params, &error);
  EXPECT_CAPI_OK(error);
  EXPECT_NE(actor, static_cast<MochiActorHandle>(MOCHI_INVALID_HANDLE));

  // OK for actor to out-live the shape
  Mochi_ReleaseShape(context, shape);

  // Verify actor is valid
  EXPECT_TRUE(Mochi_Actor_IsValid(scene, actor));
}

TEST(CAPI, CreateSoftActor) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  MochiSceneHandle scene = Mochi_CreateScene(context, "TestScene", &error);
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
  EXPECT_NE(actor, static_cast<MochiActorHandle>(MOCHI_INVALID_HANDLE));

  // Verify actor is valid
  EXPECT_TRUE(Mochi_Actor_IsValid(scene, actor));
}

//==================================================================================================
// Actor Operations Tests
//==================================================================================================

TEST(CAPI, ActorTransform) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  MochiSceneHandle scene = Mochi_CreateScene(context, "TestScene", &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_DestroyScene(context, scene));

  // Create sphere and actor
  MochiReal3 center = MochiReal3_GetDefault();
  MochiShapeHandle shape = Mochi_CreateSphereShape(context, center, 1.0f, &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_ReleaseShape(context, shape));

  MochiRigidActorParams params = MochiRigidActorParams_GetDefault();
  params.shape = shape;
  params.isStatic = true; // Use static for simplicity in CAPI testing
  params.colliderType = MochiColliderType_Sphere;
  MochiActorHandle actor = Mochi_Scene_CreateRigidActor(scene, params, &error);
  EXPECT_CAPI_OK(error);

  // Set transform
  MochiTransformRT transform = MochiTransformRT_GetDefault();
  transform.translation.data[0] = 1.0f;
  transform.translation.data[1] = 2.0f;
  transform.translation.data[2] = 3.0f;

  Mochi_Actor_SetRootTransform(scene, actor, transform, &error);
  EXPECT_CAPI_OK(error);

  // Get transform
  MochiTransformRT retrievedTransform = Mochi_Actor_GetRootTransform(scene, actor, &error);
  EXPECT_CAPI_OK(error);
  EXPECT_NEAR(retrievedTransform.translation.data[0], 1.0f, 1e-5f);
  EXPECT_NEAR(retrievedTransform.translation.data[1], 2.0f, 1e-5f);
  EXPECT_NEAR(retrievedTransform.translation.data[2], 3.0f, 1e-5f);
}

//==================================================================================================
// Scene Stepping Test
//==================================================================================================

TEST(CAPI, SceneStep) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  MochiSceneHandle scene = Mochi_CreateScene(context, "TestScene", &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_DestroyScene(context, scene));

  // Create a simple rigid actor
  MochiReal3 center = MochiReal3_GetDefault();
  MochiShapeHandle shape = Mochi_CreateSphereShape(context, center, 1.0f, &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_ReleaseShape(context, shape));

  MochiRigidActorParams params = MochiRigidActorParams_GetDefault();
  params.shape = shape;
  params.isStatic = true; // Use static for simplicity in CAPI testing
  params.colliderType = MochiColliderType_Sphere;
  MochiActorHandle actor = Mochi_Scene_CreateRigidActor(scene, params, &error);
  EXPECT_CAPI_OK(error);
  EXPECT_TRUE(Mochi_Actor_IsValid(scene, actor));

  // Step the simulation
  MochiReal dt = 0.01f;
  Mochi_Scene_Step(scene, dt, &error);
  EXPECT_CAPI_OK(error);
}

//==================================================================================================
// Constraint Tests
//==================================================================================================

TEST(CAPI, Constraints) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  MochiSceneHandle scene = Mochi_CreateScene(context, "TestScene", &error);
  MOCHI_DEFER(Mochi_DestroyScene(context, scene));

  // Verify no linker issues with CAPI functions by insuring they're callable.
  Mochi_Constraint_IsValid(scene, MOCHI_INVALID_HANDLE);

  Mochi_Constraint_GetType(scene, MOCHI_INVALID_HANDLE, &error);

  Mochi_Constraint_GetStiffness(scene, MOCHI_INVALID_HANDLE, &error);

  Mochi_Constraint_SetStiffness(scene, MOCHI_INVALID_HANDLE, 1_r, &error);

  Mochi_Constraint_GetDamping(scene, MOCHI_INVALID_HANDLE, &error);

  Mochi_Constraint_SetDamping(scene, MOCHI_INVALID_HANDLE, 1_r, &error);

  Mochi_Constraint_GetSaturation(scene, MOCHI_INVALID_HANDLE, &error);

  Mochi_Constraint_SetSaturation(scene, MOCHI_INVALID_HANDLE, 1_r, &error);

  Mochi_Constraint_GetNumActors(scene, MOCHI_INVALID_HANDLE, &error);

  Mochi_Constraint_GetActor(scene, MOCHI_INVALID_HANDLE, 0, &error);

  Mochi_Constraint_GetDofIndicesForActor(scene, MOCHI_INVALID_HANDLE, 0, &error);

  Mochi_Constraint_SetTargetPosition(scene, MOCHI_INVALID_HANDLE, MochiReal3_GetDefault(), &error);

  auto defaultQuaternion = MochiQuaternion_GetDefault();
  Mochi_Constraint_SetTargetRotation(scene, MOCHI_INVALID_HANDLE, defaultQuaternion, &error);

  Mochi_Constraint_SetTargetDof(scene, MOCHI_INVALID_HANDLE, 0_r, &error);

  Mochi_Constraint_UpdateOldTarget(scene, MOCHI_INVALID_HANDLE, &error);

  Mochi_Constraint_SetRefRelativeRotation(
      scene, MOCHI_INVALID_HANDLE, defaultQuaternion, defaultQuaternion, &error);

  Mochi_Constraint_GetLimitMinValues(scene, MOCHI_INVALID_HANDLE, &error);

  Mochi_Constraint_GetLimitMaxValues(scene, MOCHI_INVALID_HANDLE, &error);

  auto rigidSphericalJointParams = MochiRigidSphericalJointConstraintParams_GetDefault();
  Mochi_Scene_CreateRigidSphericalJointConstraint(scene, rigidSphericalJointParams, &error);

  auto rigidPrismaticJointParams = MochiRigidPrismaticJointConstraintParams_GetDefault();
  Mochi_Scene_CreateRigidPrismaticJointConstraint(scene, rigidPrismaticJointParams, &error);

  auto deformableNodeToDeformableNodeParams =
      MochiDeformableNodeToDeformableNodeConstraintParams_GetDefault();
  Mochi_Scene_CreateDeformableNodeToDeformableNodeConstraint(
      scene, deformableNodeToDeformableNodeParams, &error);

  auto deformableNodeToRigidParams = MochiDeformableNodeToRigidConstraintParams_GetDefault();
  Mochi_Scene_CreateDeformableNodeToRigidConstraint(scene, deformableNodeToRigidParams, &error);

  auto jointRotationRangeParams = MochiJointRotationRangeConstraintParams_GetDefault();
  Mochi_Scene_CreateJointRotationRangeConstraint(scene, jointRotationRangeParams, &error);

  auto jointRotationTrackingParams = MochiJointRotationTrackingConstraintParams_GetDefault();
  Mochi_Scene_CreateJointRotationTrackingConstraint(scene, jointRotationTrackingParams, &error);

  auto rigidPivotPositionParams = MochiRigidPivotPositionConstraintParams_GetDefault();
  Mochi_Scene_CreateRigidPivotPositionConstraint(scene, rigidPivotPositionParams, &error);

  auto rigidPivotRotationParams = MochiRigidPivotRotationConstraintParams_GetDefault();
  Mochi_Scene_CreateRigidPivotRotationConstraint(scene, rigidPivotRotationParams, &error);

  auto deformableNodePositionParams = MochiDeformableNodePositionConstraintParams_GetDefault();
  Mochi_Scene_CreateDeformableNodePositionConstraint(scene, deformableNodePositionParams, &error);

  auto articulatedSingleDofTargetParams =
      MochiArticulatedSingleDofTargetConstraintParams_GetDefault();
  Mochi_Scene_CreateArticulatedSingleDofTargetConstraint(
      scene, articulatedSingleDofTargetParams, &error);

  auto articulated3dRotationTargetParams =
      MochiArticulated3dRotationTargetConstraintParams_GetDefault();
  Mochi_Scene_CreateArticulated3dRotationTargetConstraint(
      scene, articulated3dRotationTargetParams, &error);

  auto articulatedSingleDofRangeParams =
      MochiArticulatedSingleDofRangeConstraintParams_GetDefault();
  Mochi_Scene_CreateArticulatedSingleDofRangeConstraint(
      scene, articulatedSingleDofRangeParams, &error);

  auto articulated3dRotationRangeParams =
      MochiArticulated3dRotationRangeConstraintParams_GetDefault();
  Mochi_Scene_CreateArticulated3dRotationRangeConstraint(
      scene, articulated3dRotationRangeParams, &error);

  Mochi_Scene_DestroyConstraint(scene, MOCHI_INVALID_HANDLE);
  Mochi_Scene_GetNumConstraints(scene, &error);
  Mochi_Scene_ForEachConstraint(scene, nullptr, nullptr, &error);
}

//==================================================================================================
// DynamicArray Memory Handling Tests
//==================================================================================================

// Test that malloc'd memory passed in DynamicArray can be freed after C++ receives it
// This verifies the ToCpp conversion makes a deep copy of DynamicArray contents
TEST(CAPI, DynamicArrayMallocFreeAfterUse) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  // Allocate coordinates with malloc
  auto coordsVec = CreateSmallCubeTetMeshCoords();
  auto* mallocedCoords = static_cast<MochiReal*>(malloc(coordsVec.size() * sizeof(MochiReal)));
  memcpy(mallocedCoords, coordsVec.data(), coordsVec.size() * sizeof(MochiReal));

  auto connectivityVec = CreateSmallCubeTetMeshConnectivity();
  auto* mallocedConnectivity =
      static_cast<int32_t*>(malloc(connectivityVec.size() * sizeof(int32_t)));
  memcpy(mallocedConnectivity, connectivityVec.data(), connectivityVec.size() * sizeof(int32_t));

  // Create shape using the malloc'd arrays
  MochiConstRealSpan coordsSpan = {mallocedCoords, coordsVec.size()};
  MochiConstIntSpan connectivitySpan = {mallocedConnectivity, connectivityVec.size()};
  MochiShapeHandle shape = Mochi_CreateTetMeshShape(context, coordsSpan, connectivitySpan, &error);
  EXPECT_CAPI_OK(error);
  EXPECT_NE(shape, static_cast<MochiShapeHandle>(MOCHI_INVALID_HANDLE));

  // Free malloc'd memory - C++ should have its own copy
  free(mallocedCoords);
  free(mallocedConnectivity);

  // Shape should still be valid
  MochiAabb aabb = Mochi_GetShapeAabb(context, shape, &error);
  EXPECT_CAPI_OK(error);

  // The AABB should be approximately the size of our cube (-0.1 to 0.1)
  EXPECT_NEAR(aabb.min.data[0], -0.1f, 0.01f);
  EXPECT_NEAR(aabb.max.data[0], 0.1f, 0.01f);

  Mochi_ReleaseShape(context, shape);
}

// Test that DynamicArray struct fields are properly deep-copied from C to C++
// This test uses MochiArticulatedShapeParams which has multiple DynamicArray fields
// TODO: This test (and the whole file) is disabled via `#if 0` and still targets the deprecated
// Mochi_CreateArticulatedShape C API. The C API emitter in mochi_gen is currently disabled, so the
// new ArticulatedActorParamsNew C bindings are not generated. When C API emission is reactivated,
// re-enable this test and migrate it to the new APIs (Mochi_Scene_CreateArticulatedActorNew /
// MochiArticulatedActorParamsNew).
TEST(CAPI, DynamicArrayStructFieldDeepCopy) {
  MochiError error = MochiError_OK;

  MochiContextHandle context = Mochi_CreateContext(0);
  MOCHI_DEFER(Mochi_DestroyContext(context));

  // Create two link shapes
  auto coords = CreateSmallCubeTetMeshCoords();
  auto connectivity = CreateSmallCubeTetMeshConnectivity();
  MochiConstRealSpan cubeCoords = {coords.data(), coords.size()};
  MochiConstIntSpan cubeConn = {connectivity.data(), connectivity.size()};
  MochiShapeHandle cubeShape = Mochi_CreateTetMeshShape(context, cubeCoords, cubeConn, &error);
  EXPECT_CAPI_OK(error);
  MOCHI_DEFER(Mochi_ReleaseShape(context, cubeShape));

  // Allocate DynamicArray contents using malloc (simulating C allocation)
  size_t const numLinks = 2;

  // rootFromLinksAtRest: DynamicArray<TransformRT>
  auto* mallocedTransforms =
      static_cast<MochiTransformRT*>(malloc(numLinks * sizeof(MochiTransformRT)));
  mallocedTransforms[0] = MochiTransformRT_GetDefault();
  mallocedTransforms[1] = MochiTransformRT_GetDefault();

  // parents: DynamicArray<int32_t>
  auto* mallocedParents = static_cast<int32_t*>(malloc(numLinks * sizeof(int32_t)));
  mallocedParents[0] = -1; // root has no parent
  mallocedParents[1] = 0; // second link's parent is the root

  // jointTypes: DynamicArray<ArticulatedJointType>
  auto* mallocedJointTypes =
      static_cast<MochiArticulatedJointType*>(malloc(numLinks * sizeof(MochiArticulatedJointType)));
  mallocedJointTypes[0] = MochiArticulatedJointType_Free;
  mallocedJointTypes[1] = MochiArticulatedJointType_Spherical;

  // jointAxes: DynamicArray<Real3>
  auto* mallocedAxes = static_cast<MochiReal3*>(malloc(numLinks * sizeof(MochiReal3)));
  mallocedAxes[0] = MochiReal3{{0.0f, 0.0f, 0.0f}};
  mallocedAxes[1] = MochiReal3{{0.0f, 0.0f, 0.0f}};

  // jointLocals: DynamicArray<Real3>
  auto* mallocedLocals = static_cast<MochiReal3*>(malloc(numLinks * sizeof(MochiReal3)));
  mallocedLocals[0] = MochiReal3{{0.0f, 0.0f, 0.0f}};
  mallocedLocals[1] = MochiReal3{{0.0f, 0.0f, 0.0f}};

  // Build the MochiArticulatedShapeParams struct with DynamicArray fields
  MochiArticulatedShapeParams params = MochiArticulatedShapeParams_GetDefault();

  params.rootFromLinksAtRest.ptr = mallocedTransforms;
  params.rootFromLinksAtRest.size = numLinks;

  params.parents.ptr = mallocedParents;
  params.parents.size = numLinks;

  params.jointTypes.ptr = mallocedJointTypes;
  params.jointTypes.size = numLinks;

  params.jointAxes.ptr = mallocedAxes;
  params.jointAxes.size = numLinks;

  params.jointLocals.ptr = mallocedLocals;
  params.jointLocals.size = numLinks;

  // Create articulated shape - C++ should deep copy all DynamicArray contents
  MochiShapeHandle articulatedShape = Mochi_CreateArticulatedShape(context, params, &error);
  EXPECT_CAPI_OK(error);
  EXPECT_NE(articulatedShape, static_cast<MochiShapeHandle>(MOCHI_INVALID_HANDLE));

  // Free all malloc'd memory BEFORE using the shape
  // If C++ properly deep-copied, this should not cause any issues
  free(mallocedTransforms);
  // fill the mallocedParents array with garbage values in case we're running a release allocator
  memset(mallocedParents, 0xAF, numLinks * sizeof(int32_t));
  free(mallocedParents);
  // fill the mallocedParents array with garbage values in case we're running a release allocator
  free(mallocedJointTypes);
  free(mallocedAxes);
  free(mallocedLocals);

  // Clean up
  Mochi_ReleaseShape(context, articulatedShape);
}

#endif // 0
