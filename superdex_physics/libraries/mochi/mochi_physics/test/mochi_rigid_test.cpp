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
#include <mochi_physics/src/mochi_rigid.h>

#include <gtest/gtest.h>

using namespace mochi;

class AccurateRigidInertia : public test::MochiSceneTestBase {};

TEST_F(AccurateRigidInertia, AccurateRigidInertia) {
  real constexpr kDt = 1e-1_r;

  auto runTest = [&](IntegrationMethod method, bool useNewtonEulerInertia, real tol) {
    // Disable gravity
    _scene->SetGravity(Real3{});

    // Set integration method
    auto solver = _scene->GetSolverParams();
    solver.integrationMethod = method;
    _scene->SetSolverParams(solver, test::ExpectOK{});

    // Create rigid actor
    auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube(Real3{1_r, 2_r, 0.5_r});
    auto shape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});
    RigidActorParams params;
    params.shape = shape;
    params.colliderType = ColliderType::None;
    params.density = 1_r;
    auto* actor = _scene->CreateRigidActor(params, test::ExpectOK{});

    // Enable Newton-Euler inertia
    experimental::EnableNewtonEulerInertia(actor, useNewtonEulerInertia, test::ExpectOK{});

    // Set initial angular velocity
    actor->SetVelocity(Real3{}, Real3{1.5_r, 0.3_r, 0.6_r}, test::ExpectOK{});

    // Get rest-pose inertia
    auto m = actor->GetRigidMomentOfInertiaLocal(test::ExpectOK{});

    // Convert to symmetrical 3x3 matrix
    VMatrix3x3r M = {
        Vec4r{m[0], m[1], m[2]}, //
        Vec4r{m[1], m[3], m[4]}, //
        Vec4r{m[2], m[4], m[5]}};

    // lambda to evaluate angular momentum
    auto evalAngularMomentum = [&]() {
      auto q = actor->GetRootTransform().GetRotation();
      auto omega = ToSimd(actor->GetAngularVelocity(test::ExpectOK{}));
      return ToReal3(DotMatVec3x3(RotateInertia(M, q), omega));
    };

    // Initial angular momentum
    auto const initialAngularMomentum = evalAngularMomentum();

    // Step the simulation 100 times
    for (int i = 0; i < 100; ++i) {
      _scene->Step(kDt);
    }

    // Check final angular momentum
    auto const finalAngularMomentum = evalAngularMomentum();

    // Check error
    real error = Norm(finalAngularMomentum - initialAngularMomentum) / Norm(initialAngularMomentum);
    EXPECT_LE(error, tol);

    // Destroy actor
    _scene->DestroyActor(actor);
  };

  // Run test with different integration methods, with Rigid IPC inertia and Newton-Euler inertia
  runTest(
      IntegrationMethod::BackwardEuler,
      /* useNewtonEulerInertia */ false,
      6e-1_r); // 60% tolerance
  runTest(IntegrationMethod::BDF2, /* useNewtonEulerInertia */ false, 6e-1_r); // 60% tolerance
  runTest(IntegrationMethod::DIRK33, /* useNewtonEulerInertia */ false, 6e-1_r); // 50% tolerance
  runTest(
      IntegrationMethod::BackwardEuler,
      /* useNewtonEulerInertia */ true,
      2e-1_r); // 20% tolerance, due to dissipation
  runTest(IntegrationMethod::BDF2, /* useNewtonEulerInertia */ true, 3e-2_r); // 3% tolerance
  runTest(IntegrationMethod::DIRK33, /* useNewtonEulerInertia */ true, 2e-3_r); // 0.2% tolerance
}

class StaticActor : public test::MochiSceneTestBase {};

// Test that static actors can be teleported with SetVelocity(zero). We run a rigid actor with
// initial velocity touching a static actor, then run the same scene translated and rotated by 90
// degrees around the Y axis. The relative motion should be the same.
TEST_F(StaticActor, SetVelocity) {
  real constexpr kDt = 1e-2_r;
  int constexpr kNumSteps = 50;

  // Scene setup: Create a static plane and a rigid cube touching it with initial velocity
  auto runSimulation = [&](TransformRT const& sceneTransform) {
    auto const& sceneRotation = sceneTransform.GetRotation();

    // Static plane at Y=0 facing up
    Real3 const planeNormal = sceneRotation * Real3{0_r, 1_r, 0_r};
    auto planeShape = _mochiContext->CreatePlaneShape(planeNormal, 0_r, test::ExpectOK{});
    RigidActorParams planeParams;
    planeParams.name = "Plane";
    planeParams.isStatic = true;
    planeParams.shape = planeShape;
    auto* planeActor = _scene->CreateRigidActor(planeParams, test::ExpectOK{});

    // Dynamic rigid cube at the plane
    auto [coordinates, connectivity] =
        test::CreateMinimalTetMeshUnitCube(Real3{0.5_r, 0.5_r, 0.5_r});
    auto cubeShape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});
    RigidActorParams cubeParams;
    cubeParams.name = "Cube";
    cubeParams.shape = cubeShape;
    auto* cubeActor = _scene->CreateRigidActor(cubeParams, test::ExpectOK{});

    // Set initial velocity of the rigid cube
    Real3 const linearVelocity = sceneRotation * Real3{1_r, 0_r, -1_r};
    Real3 const angularVelocity = sceneRotation * Real3{0.1_r, 0.2_r, -0.1_r};
    cubeActor->SetVelocity(linearVelocity, angularVelocity, test::ExpectOK{});

    // Set root transform of both actors
    planeActor->SetRootTransform(sceneTransform, test::ExpectOK{});
    cubeActor->SetRootTransform(sceneTransform, test::ExpectOK{});

    // Call SetVelocity(zero) on the static actor — this is the behavior being tested
    planeActor->SetVelocity(Real3{}, Real3{}, test::ExpectOK{});

    // Register contact force query
    cubeActor->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});

    // Record initial transform
    TransformRT const initialTransform = cubeActor->GetRootTransform();

    // Run simulation
    real maxContactForce = 0_r;
    for (int i = 0; i < kNumSteps; ++i) {
      _scene->Step(kDt);
      Real3 const contactForce = cubeActor->GetContactForceWorld(test::ExpectOK{});
      maxContactForce = Max(maxContactForce, Norm(contactForce));
    }

    // Verify non-zero contact forces occurred
    EXPECT_GT(maxContactForce, 0_r) << "Contact forces should be non-zero";

    // Compute relative transform in local frame
    TransformRT const finalTransform = cubeActor->GetRootTransform();
    TransformRT const relativeTransform = Invert(finalTransform) * initialTransform;

    // Clean up actors for next run
    _scene->DestroyActor(cubeActor);
    _scene->DestroyActor(planeActor);

    return relativeTransform;
  };

  // Run without rotation
  TransformRT const transformA = runSimulation(TransformRT{});

  // Run with 90-degree Y rotation and translation
  Quaternion const rotation90Y = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, kPI / 2_r);
  Real3 const translation = Real3{5_r, 0_r, 3_r};
  TransformRT const transformB = runSimulation(TransformRT{rotation90Y, translation});

  // Confirm the transforms are the same
  real constexpr kTol = 1e-3_r;
  EXPECT_NEAR_TOL(transformA.GetTranslation(), transformB.GetTranslation(), kTol);
  EXPECT_NEAR_TOL(transformA.GetRotation(), transformB.GetRotation(), kTol);
}

namespace {
class RigidActorParamsTest : public test::MochiSceneTestBase {};
} // namespace

TEST_F(RigidActorParamsTest, DynamicRigidActorRequiresAMesh) {
  // Create a static sphere actor
  RigidActorParams params;
  params.name = "sphere";
  params.isStatic = true;
  params.mass = 1_r;
  params.centerOfMass = Real3{};
  params.momentOfInertia = Real6{0.4_r, 0_r, 0_r, 0.4_r, 0_r, 0.4_r};
  params.shape =
      _mochiContext->CreateSphereShape(/*center*/ Real3{}, /*radius*/ 1_r, test::ExpectOK{});
  auto* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
  ASSERT_NE(nullptr, actor);
  _scene->DestroyActor(actor);
  actor = nullptr;

  // Change only "isStatic". Fail because a dynamic actor requires a mesh.
  params.isStatic = false;
  ASSERT_EQ(nullptr, _scene->CreateRigidActor(params, test::ExpectNotOK{}));
}

TEST_F(RigidActorParamsTest, InvalidParams) {
  auto [coordinates, connectivity] = test::CreateMinimalTriMeshUnitCube();
  auto shape = _mochiContext->CreateTriMeshShape(
      Flatten(MakeConstSpan(coordinates)), Flatten(MakeConstSpan(connectivity)), test::ExpectOK{});

  // Create a valid actor
  RigidActorParams params;
  params.name = "box";
  params.isStatic = false;
  params.shape = shape;
  auto* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
  ASSERT_NE(nullptr, actor);
  _scene->DestroyActor(actor);

  // Invalid ColliderTypes
  for (auto colliderType : {static_cast<ColliderType>(-1), ColliderType::Count}) {
    auto params2 = params;
    params2.colliderType = colliderType;
    ASSERT_EQ(nullptr, _scene->CreateRigidActor(params2, test::ExpectNotOK{}));
  }

  // Invalid ActorBoundaryElementType
  for (auto boundaryElementType :
       {static_cast<ActorBoundaryElementType>(-1), ActorBoundaryElementType::Count}) {
    auto params2 = params;
    params2.boundaryElementType = boundaryElementType;
    ASSERT_EQ(nullptr, _scene->CreateRigidActor(params2, test::ExpectNotOK{}));
  }

  // Invalid BoundarySubsamplingStrategy
  for (auto boundarySubsamplingStrategy :
       {static_cast<BoundarySubsamplingStrategy>(-1), BoundarySubsamplingStrategy::Count}) {
    auto params2 = params;
    params2.boundarySubsampling.emplace();
    params2.boundarySubsampling->strategy = boundarySubsamplingStrategy;
    ASSERT_EQ(nullptr, _scene->CreateRigidActor(params2, test::ExpectNotOK{}));
  }
}
