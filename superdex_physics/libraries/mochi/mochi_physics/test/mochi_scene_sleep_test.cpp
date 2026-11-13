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

#include <mochi_core/element_operations/fem_rod.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_physics/mochi_physics_experimental.h>

// These tests inspect island sleep state, which is not exposed through the public API.
#include <mochi_physics/src/mochi_island.h>
#include <mochi_physics/src/mochi_sleep.h>

#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <string>

using namespace mochi;
using namespace mochi::experimental;

/***************************************************************************************************
  SleepParams
*/

namespace {
class SceneSleepParams : public test::MochiSceneTestBase {};
} // namespace

TEST_F(SceneSleepParams, DefaultsDisableSleep) {
  EXPECT_EQ(SleepParams{}, experimental::GetSleepParams(_scene, test::ExpectOK{}));
  EXPECT_FALSE(experimental::GetSleepParams(_scene, test::ExpectOK{}).canSleep);
}

TEST_F(SceneSleepParams, SetThenGet) {
  for (SleepParams const& params :
       {SleepParams{.canSleep = true, .sleepThreshold = 1.0, .minStepsBeforeSleep = 2},
        SleepParams{.canSleep = false, .sleepThreshold = 0.1, .minStepsBeforeSleep = 3}}) {
    experimental::SetSleepParams(_scene, params, test::ExpectOK{});
    EXPECT_EQ(params, experimental::GetSleepParams(_scene, test::ExpectOK{}));
  }
}

TEST_F(SceneSleepParams, RejectsInvalidParams) {
  auto const expectRejected = [&](SleepParams const& params) {
    experimental::SetSleepParams(_scene, params, test::ExpectNotOK{});
    EXPECT_EQ(SleepParams{}, experimental::GetSleepParams(_scene, test::ExpectOK{}))
        << "Invalid params must not be stored.";
  };
  expectRejected({.sleepThreshold = 0.0});
  expectRejected({.sleepThreshold = -0.5});
  expectRejected({.sleepThreshold = 1.001});
  expectRejected({.sleepThreshold = std::numeric_limits<double>::quiet_NaN()});
  expectRejected({.minStepsBeforeSleep = 0});
  expectRejected({.minStepsBeforeSleep = 1});
  expectRejected({.minStepsBeforeSleep = -1});
}

TEST_F(SceneSleepParams, DifferentiableSceneDisablesSleep) {
  experimental::SetSleepParams(_scene, {.canSleep = true}, test::ExpectOK{});
  test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
  {
    test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
    MakeSceneDifferentiableInternal(_scene, test::ExpectOK{});
  }
  EXPECT_FALSE(experimental::GetSleepParams(_scene, test::ExpectOK{}).canSleep);

  experimental::SetSleepParams(_scene, {.canSleep = true}, test::ExpectNotOK{});
  experimental::SetSleepParams(_scene, {.canSleep = false}, test::ExpectOK{});
}

/***************************************************************************************************
  Stage rest values
*/

static NewtonSolverStatus<real>
MakeStatus(ConvergenceStatus convergence, int numIterDone, real resNorm0, real resNorm) {
  NewtonSolverStatus<real> status;
  status.convergence = convergence;
  status.numIterDone = numIterDone;
  status.resNorm0 = resNorm0;
  status.resNorm = resNorm;
  return status;
}

TEST(SceneSleepRestValue, ZeroIterationsIsFullyAtRest) {
  EXPECT_EQ(1_r, sleep::StageRestValue(MakeStatus(ConvergenceStatus::Converged, 0, 0_r, 0_r)));
  EXPECT_EQ(
      1_r, sleep::StageRestValue(MakeStatus(ConvergenceStatus::Converged, 0, 1e-4_r, 1e-4_r)));
}

TEST(SceneSleepRestValue, NotConvergedIsNotAtRest) {
  for (auto convergence :
       {ConvergenceStatus::None, ConvergenceStatus::Stopped, ConvergenceStatus::Diverged}) {
    EXPECT_EQ(0_r, sleep::StageRestValue(MakeStatus(convergence, 0, 1_r, 1_r)));
    EXPECT_EQ(0_r, sleep::StageRestValue(MakeStatus(convergence, 3, 1_r, 1_r)));
  }
}

TEST(SceneSleepRestValue, IterationsGiveResidualRatio) {
  EXPECT_NEAR_EQ(
      0.25_r, sleep::StageRestValue(MakeStatus(ConvergenceStatus::Converged, 2, 4_r, 1_r)));
}

/***************************************************************************************************
  Test fixture
*/

static DynamicArray<ActorType> GetAllActorTypes() {
  DynamicArray<ActorType> types;
  for (int i = static_cast<int>(ActorType::None) + 1; i < static_cast<int>(ActorType::Count); ++i) {
    types.push_back(static_cast<ActorType>(i));
  }
  return types;
}

static std::string ActorTypeName(testing::TestParamInfo<ActorType> const& info) {
  return SReflect::EnumToString(info.param);
}

static DynamicArray<real> GetAllDofValues(Actor const* actor) {
  DynamicArray<real> values(actor->GetNumDofs());
  actor->GetDofValues({}, MakeSpan(values), test::ExpectOK{});
  return values;
}

namespace {

class SceneSleep : public test::MochiSceneTestBase {
 public:
  static constexpr double kDt = 0.01;
  static constexpr int kMaxStepsToSleep = 1000;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();
    // Backward Euler's numerical damping lets actors settle quickly.
    test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
    experimental::SetSleepParams(_scene, {.canSleep = true}, test::ExpectOK{});
    _ground = CreateGround();
  }

  Actor* CreateGround(Real3 normal = Real3{0_r, 1_r, 0_r}) {
    RigidActorParams params;
    params.name = "Ground";
    params.isStatic = true;
    params.shape = _mochiContext->CreatePlaneShape(Normalize(normal), 0_r, test::ExpectOK{});
    params.colliderType = ColliderType::Plane;
    return _scene->CreateRigidActor(params, test::ExpectOK{});
  }

  ShapeHandle CreateCubeShape(real size) {
    auto [coords, conn] = test::CreateMinimalTetMeshUnitCube(Real3{size, size, size});
    return _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), test::ExpectOK{});
  }

  Actor* CreateStaticBox(Real3 const& corner, real size) {
    RigidActorParams params;
    params.isStatic = true;
    params.shape = CreateCubeShape(size);
    params.colliderType = ColliderType::Box;
    params.worldFromLocal.SetTranslation(corner);
    return _scene->CreateRigidActor(params, test::ExpectOK{});
  }

  Actor* CreateStaticArticulatedBox(Real3 const& corner, real size) {
    ArticulatedActorParams params;
    params.joints = {{.type = ArticulatedJointType::Hard}};
    params.links = {
        {.parentLink = -1, .shape = CreateCubeShape(size), .colliderType = ColliderType::Box}};
    params.worldFromRoot.SetTranslation(corner);
    return _scene->CreateArticulatedActor(params, test::ExpectOK{});
  }

  // Create an actor whose bottom is at the given height, centered on x = z = 0.
  Actor* CreateTestActor(ActorType type, real bottomHeight = 0.02_r) {
    constexpr real kCubeSize = 0.2_r; // Edge length of rigid and soft cubes [m]
    constexpr real kThinActorSize = 0.8_r * kCubeSize; // Shells and rods fit on top of a cube [m]
    // The cube and patch meshes have a corner at their local origin.
    auto const cornerAt = [bottomHeight](real size) {
      return Real3{-0.5_r * size, bottomHeight, -0.5_r * size};
    };
    switch (type) {
      case ActorType::Rigid: {
        RigidActorParams params;
        params.shape = CreateCubeShape(kCubeSize);
        params.colliderType = ColliderType::Box;
        params.worldFromLocal.SetTranslation(cornerAt(kCubeSize));
        return _scene->CreateRigidActor(params, test::ExpectOK{});
      }
      case ActorType::Soft: {
        SoftActorParams params;
        params.shape = CreateCubeShape(kCubeSize);
        params.worldFromLocal.SetTranslation(cornerAt(kCubeSize));
        return _scene->CreateSoftActor(params, test::ExpectOK{});
      }
      case ActorType::Shell: {
        DynamicArray<Real3> const coords = {
            Real3{0_r, 0_r, 0_r},
            Real3{kThinActorSize, 0_r, 0_r},
            Real3{kThinActorSize, 0_r, kThinActorSize},
            Real3{0_r, 0_r, kThinActorSize}};
        DynamicArray<Int3> const conn = {Int3{0, 1, 2}, Int3{0, 2, 3}};
        ShellActorParams params;
        params.shape = _mochiContext->CreateTriMeshShape(
            Flatten(MakeConstSpan(coords)), Flatten(MakeConstSpan(conn)), test::ExpectOK{});
        params.worldFromLocal.SetTranslation(cornerAt(kThinActorSize));
        return CreateShellActor(_scene, params, test::ExpectOK{});
      }
      case ActorType::Rod: {
        // Along x, centered on the rod's local origin.
        constexpr int kNumNodes = 5;
        DynamicArray<Real3> nodes;
        for (int i = 0; i < kNumNodes; ++i) {
          real const t = StaticCast<real>(i) / (kNumNodes - 1) - 0.5_r;
          nodes.push_back(Real3{t * kThinActorSize, 0_r, 0_r});
        }
        DynamicArray<Real3> const elementFrameAxes(kNumNodes - 1, Real3{0_r, 1_r, 0_r});
        RodActorParams params;
        params.shape = CreatePolylineShape(
            _mochiContext, nodes, elementFrameAxes, /*isClosedLoop=*/false, test::ExpectOK{});
        params.material.linearDensity = 10_r;
        params.material.linearRotationalInertia = 1_r;
        params.material.axialStiffness = 1e5_r;
        params.material.torsionalStiffness = 1e3_r;
        params.material.flexuralStiffness = {1e3_r, 1e3_r};
        params.worldFromLocal.SetTranslation(Real3{0_r, bottomHeight, 0_r});
        return CreateRodActor(_scene, params, test::ExpectOK{});
      }
      case ActorType::Articulated: {
        real const linkSize = 0.5_r * kCubeSize;
        auto const linkShape = CreateCubeShape(linkSize);
        ArticulatedActorParams params;
        params.joints = {
            {.type = ArticulatedJointType::Free},
            {.type = ArticulatedJointType::Spherical,
             .parentLinkFromJoint = TransformRT{Real3{0.5_r * linkSize, 0_r, 0_r}}}};
        params.links = {
            {.parentLink = -1, .shape = linkShape, .colliderType = ColliderType::Box},
            {.parentLink = 0,
             .parentJointFromLink = TransformRT{Real3{0.5_r * linkSize, 0_r, 0_r}},
             .shape = linkShape,
             .colliderType = ColliderType::Box}};
        params.worldFromRoot.SetTranslation(Real3{-linkSize, bottomHeight, -0.5_r * linkSize});
        return _scene->CreateArticulatedActor(params, test::ExpectOK{});
      }
      case ActorType::None:
      case ActorType::Count:
        break;
    }
    static_assert(
        static_cast<int>(ActorType::Count) == 6,
        "If you add an actor type, then please update the code above.");
    ADD_FAILURE() << "Unsupported actor type.";
    return nullptr;
  }

  entt::entity GetIsland(Actor const* actor) const {
    auto const* memberInfo =
        GetRegistry().try_get<CIslandMemberInfo const>(GetEntity(actor->GetHandle()));
    EXPECT_NE(nullptr, memberInfo);
    return memberInfo ? memberInfo->island : entt::entity{entt::null};
  }

  bool IsAwake(Actor const* actor) const {
    return sleep::IsIslandAwake(GetRegistry(), GetIsland(actor));
  }

  void Step(int numSteps = 1) {
    for (int i = 0; i < numSteps; ++i) {
      _scene->Step(kDt);
    }
  }

  // Step until the actor's island is asleep. Call this right after the island was created or woken.
  // Fails if the island falls asleep sooner than SleepParams::minStepsBeforeSleep allows, or if it
  // is still awake after maxSteps.
  void StepUntilAsleep(Actor const* actor, int maxSteps = kMaxStepsToSleep) {
    // An island solved N times at rest goes to sleep during the next step.
    int const minSteps =
        experimental::GetSleepParams(_scene, test::ExpectOK{}).minStepsBeforeSleep + 1;
    int numSteps = 0;
    do {
      ASSERT_LT(numSteps, maxSteps);
      Step();
      ++numSteps;
    } while (IsAwake(actor));
    ASSERT_GE(numSteps, minSteps);
  }

  // Step and fail if the actor's island is asleep after any of the steps.
  void StepExpectAwake(Actor const* actor, int numSteps) {
    for (int i = 0; i < numSteps; ++i) {
      Step();
      ASSERT_TRUE(IsAwake(actor));
    }
  }

  // Apply pending wake-up requests without simulating, as the next step would.
  void ApplyWakeRequests() {
    sleep::PreStep(GetRegistry());
  }

  // Position of the actor, for measuring how far it moved. Articulated actors have no world AABB,
  // so use their root joint's translation DoFs, which are relative to the initial root transform.
  static Real3 GetPosition(Actor const* actor) {
    if (actor->GetType() == ActorType::Articulated) {
      auto const dofs = GetAllDofValues(actor);
      return Real3{dofs[0], dofs[1], dofs[2]};
    }
    return actor->GetAabbWorld(test::ExpectOK{}).GetCenter();
  }

  // Node positions of a deformable actor before any simulation, in the world frame.
  static DynamicArray<Real3> GetRestNodePositionsWorld(Actor const* actor) {
    TransformRT const worldFromLocal = actor->GetRootTransform();
    DynamicArray<Real3> positions;
    for (Real3 const& p : Unflatten<Real3 const>(actor->GetMesh().coordinates)) {
      positions.push_back(worldFromLocal.TransformPoint(p));
    }
    return positions;
  }

  // Hold every position DoF of the actor in place with boundary conditions, or with constraints for
  // rods, which do not support boundary conditions.
  static void ConstrainInPlace(Scene* scene, Actor* actor) {
    ActorType const type = actor->GetType();
    if (type == ActorType::Rigid || type == ActorType::Articulated) {
      auto const values = GetAllDofValues(actor);
      DynamicArray<int> dofs(isize(values));
      std::iota(dofs.begin(), dofs.end(), 0);
      actor->AddBoundaryConditionDofsWorld(
          MakeConstSpan(dofs), MakeConstSpan(values), test::ExpectOK{});
      return;
    }
    auto const positions = GetRestNodePositionsWorld(actor);
    if (type == ActorType::Rod) {
      DeformableNodePositionConstraintParams params;
      params.actor = actor->GetHandle();
      // Softer than the default, so that single-precision roundoff in the penalty forces at rest
      // stays below the solver tolerance and the island can fall asleep.
      params.stiffness = 1e4_r;
      for (int i = 0; i < isize(positions); ++i) {
        params.nodeIndex = i;
        params.position = positions[i];
        scene->CreateDeformableNodePositionConstraint(params, test::ExpectOK{});
      }
      return;
    }
    DynamicArray<int> nodes(isize(positions));
    std::iota(nodes.begin(), nodes.end(), 0);
    actor->AddBoundaryConditionNodesWorld(
        MakeConstSpan(nodes), Flatten(MakeConstSpan(positions)), test::ExpectOK{});
  }

  // Give the actor a rigid-body velocity (world frame). Deformables rotate about their local
  // origin.
  static void SetVelocity(Actor* actor, Real3 const& linear, Real3 const& angular = Real3{}) {
    ActorType const type = actor->GetType();
    if (type == ActorType::Rigid) {
      actor->SetVelocity(linear, angular, test::ExpectOK{});
    } else if (type == ActorType::Articulated) {
      // The root free joint has translation DoFs first, then rotation DoFs.
      DynamicArray<real> velocities(actor->GetNumDofs(), 0_r);
      for (int k = 0; k < 3; ++k) {
        velocities[k] = linear[k];
        velocities[3 + k] = angular[k];
      }
      actor->SetArticulatedJointVelocities(MakeConstSpan(velocities), test::ExpectOK{});
    } else {
      // Node velocities in the local frame, which has no rotation. Rod nodes also have a twist.
      int const valuesPerNode = (type == ActorType::Rod) ? 4 : 3;
      auto const positions = Unflatten<Real3 const>(actor->GetMesh().coordinates);
      DynamicArray<real> velocities(isize(positions) * valuesPerNode, 0_r);
      for (int i = 0; i < isize(positions); ++i) {
        Real3 const v = linear + Cross(angular, positions[i]);
        for (int k = 0; k < 3; ++k) {
          velocities[valuesPerNode * i + k] = v[k];
        }
      }
      actor->SetNodeVelocitiesLocal(MakeConstSpan(velocities), test::ExpectOK{});
    }
  }

 protected:
  Actor* _ground = nullptr;
};

class SceneSleepPerType : public SceneSleep, public testing::WithParamInterface<ActorType> {};

// Actor types that recenter by default.
class SceneSleepRecentering : public SceneSleepPerType {};

} // namespace

/***************************************************************************************************
  Going to sleep
*/

TEST_P(SceneSleepPerType, SettlesOnGroundThenSleeps) {
  Actor const* actor = CreateTestActor(GetParam());
  StepUntilAsleep(actor);
}

TEST_P(SceneSleepPerType, FloatsWithoutGravityThenSleeps) {
  _scene->SetGravity(Real3{});
  Actor const* actor = CreateTestActor(GetParam(), /*bottomHeight=*/1_r);
  StepUntilAsleep(actor);
}

TEST_P(SceneSleepPerType, StateIsFrozenWhileAsleep) {
  Actor const* actor = CreateTestActor(GetParam());
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));
  auto const dofValues = GetAllDofValues(actor);
  auto const rootTransform = actor->GetRootTransform();
  Step(10);
  EXPECT_FALSE(IsAwake(actor));
  EXPECT_EQ(dofValues, GetAllDofValues(actor));
  EXPECT_EQ(rootTransform.GetTranslation(), actor->GetRootTransform().GetTranslation());
}

// Queries are only refreshed for awake islands, so recentering must not move the root transform
// when the island falls asleep, or local-space queries would be left in the old frame.
TEST_P(SceneSleepRecentering, DoesNotRecenterAsItFallsAsleep) {
  Actor const* actor = CreateTestActor(GetParam());
  ASSERT_TRUE(actor->GetRecenteringParams().useRecentering);
  TransformRT const initialRoot = actor->GetRootTransform();
  TransformRT rootBeforeStep;
  int numSteps = 0;
  do {
    ASSERT_LT(numSteps, kMaxStepsToSleep);
    rootBeforeStep = actor->GetRootTransform();
    Step();
    ++numSteps;
  } while (IsAwake(actor));
  EXPECT_TRUE(NearEqual(rootBeforeStep, actor->GetRootTransform(), 0_r));
  // Otherwise, the check above would be vacuous.
  EXPECT_FALSE(NearEqual(initialRoot, actor->GetRootTransform(), 0_r));
}

TEST_P(SceneSleepPerType, SleepingActorHasNoConvergenceStatus) {
  Actor const* actor = CreateTestActor(GetParam());
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));
  EXPECT_EQ(ConvergenceStatus::None, actor->GetConvergenceStatus());

  // Changing the sleep parameters wakes all islands.
  experimental::SetSleepParams(
      _scene, {.canSleep = true, .sleepThreshold = 0.25}, test::ExpectOK{});
  Step();
  ASSERT_TRUE(IsAwake(actor));
  EXPECT_NE(ConvergenceStatus::None, actor->GetConvergenceStatus());
}

TEST_P(SceneSleepPerType, CanSleepFalsePreventsSleep) {
  experimental::SetSleepParams(_scene, {.canSleep = false}, test::ExpectOK{});
  Actor const* actor = CreateTestActor(GetParam());
  StepExpectAwake(actor, kMaxStepsToSleep);
}

/***************************************************************************************************
  Staying awake while moving
*/

namespace {
class SceneSleepMoving : public SceneSleepPerType {
 public:
  static constexpr int kNumSteps = 200;

  void SetUp() override {
    SceneSleep::SetUp();
    _scene->SetGravity(Real3{});
    // Pin the tolerance below which the solver skips iterating. The speeds in these tests keep the
    // initial residual well above it.
    auto params = _scene->GetSolverParams();
    params.nonLinearSolver.convergenceMode = NonLinearSolverConvergenceMode::PerActorWeighted;
    params.nonLinearSolver.absTol = 1e-3_r;
    _scene->SetSolverParams(params, test::ExpectOK{});
  }

  // Like StepExpectAwake, but also fail if the solver skips iterating, since the island could then
  // rightly go to sleep.
  void StepExpectIteratingAndAwake(Actor const* actor, int numSteps) {
    for (int i = 0; i < numSteps; ++i) {
      Step();
      ASSERT_GT(_scene->GetSolverStats().maxNonLinearIters, 0);
      ASSERT_TRUE(IsAwake(actor));
    }
  }
};
} // namespace

TEST_P(SceneSleepMoving, SlowConstantVelocityStaysAwake) {
  // Slow, but well above the smallest speed the solver resolves at kDt (~1e-5 m/s without
  // gravity). Below that, the solver itself stops the motion, and then sleeping is correct.
  constexpr real kSpeed = 1e-3_r; // [m/s]
  Actor* actor = CreateTestActor(GetParam(), /*bottomHeight=*/1_r);
  SetVelocity(actor, Real3{kSpeed, 0_r, 0_r});
  real const startX = GetPosition(actor)[0];
  StepExpectIteratingAndAwake(actor, kNumSteps);
  real const expectedDistance = kSpeed * kNumSteps * kDt;
  EXPECT_NEAR(expectedDistance, GetPosition(actor)[0] - startX, 0.1_r * expectedDistance);
}

TEST_P(SceneSleepMoving, SlowConstantRotationStaysAwake) {
  constexpr real kAngularSpeed = 0.1_r; // [rad/s], ~1 cm/s at the ends of an actor
  Actor* actor = CreateTestActor(GetParam(), /*bottomHeight=*/1_r);
  SetVelocity(actor, Real3{}, Real3{0_r, kAngularSpeed, 0_r});
  auto const start = GetAllDofValues(actor);
  StepExpectIteratingAndAwake(actor, kNumSteps);
  EXPECT_NE(start, GetAllDofValues(actor));
}

INSTANTIATE_TEST_SUITE_P(
    AllActorTypes,
    SceneSleepMoving,
    testing::ValuesIn(GetAllActorTypes()),
    ActorTypeName);

TEST_P(SceneSleepPerType, ThrownUpwardStaysAwakeThroughApexAndFall) {
  Actor* actor = CreateTestActor(GetParam(), /*bottomHeight=*/0.5_r);
  SetVelocity(actor, Real3{0_r, 2_r, 0_r});
  // Rises for ~0.2 s, then falls. Stay awake through the apex and part of the fall.
  StepExpectAwake(actor, 40);
  real const height = GetPosition(actor)[1];
  StepExpectAwake(actor, 5);
  EXPECT_LT(GetPosition(actor)[1], height) << "Should be falling by now.";
}

TEST_P(SceneSleepPerType, FullyConstrainedSleeps) {
  Actor* actor = CreateTestActor(GetParam(), /*bottomHeight=*/1_r);
  ConstrainInPlace(_scene, actor);
  StepUntilAsleep(actor);
}

TEST_P(SceneSleepPerType, DestroyingGroundWakesActorAndItFalls) {
  Actor const* actor = CreateTestActor(GetParam());
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));
  real const height = GetPosition(actor)[1];
  _scene->DestroyActor(_ground);
  StepExpectAwake(actor, 20);
  EXPECT_LT(GetPosition(actor)[1], height - 0.1_r);
}

TEST_P(SceneSleepPerType, LoweringGroundWakesActorUntilItSettlesAgain) {
  constexpr real kDrop = 0.1_r;
  Actor const* actor = CreateTestActor(GetParam());
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));
  real const height = GetPosition(actor)[1];
  _ground->SetRootTransform(TransformRT{Real3{0_r, -kDrop, 0_r}}, test::ExpectOK{});
  StepUntilAsleep(actor);
  EXPECT_NEAR(height - kDrop, GetPosition(actor)[1], 0.01_r);
}

namespace {
// Swinging from a pivot 0.5 m away (rigid), or from one of its nodes (deformables).
class SceneSleepPendulum : public SceneSleepPerType {};
} // namespace

TEST_P(SceneSleepPendulum, SwingingStaysAwake) {
  Actor* actor = CreateTestActor(GetParam(), /*bottomHeight=*/1_r);
  if (GetParam() == ActorType::Rigid) {
    RigidPivotPositionConstraintParams params;
    params.actor = actor->GetHandle();
    params.localPosition = Real3{-0.5_r, 0_r, 0_r};
    params.targetPosition = actor->GetRootTransform().TransformPoint(params.localPosition);
    _scene->CreateRigidPivotPositionConstraint(params, test::ExpectOK{});
  } else {
    DeformableNodePositionConstraintParams params;
    params.actor = actor->GetHandle();
    params.nodeIndex = 0;
    params.position = GetRestNodePositionsWorld(actor)[0];
    _scene->CreateDeformableNodePositionConstraint(params, test::ExpectOK{});
  }
  // A few swings. Backward Euler damping takes much longer to stop them.
  StepExpectAwake(actor, 300);
}

INSTANTIATE_TEST_SUITE_P(
    ConstrainableActorTypes,
    SceneSleepPendulum,
    testing::Values(ActorType::Rigid, ActorType::Soft, ActorType::Shell, ActorType::Rod),
    ActorTypeName);

INSTANTIATE_TEST_SUITE_P(
    AllActorTypes,
    SceneSleepPerType,
    testing::ValuesIn(GetAllActorTypes()),
    ActorTypeName);

INSTANTIATE_TEST_SUITE_P(
    RecenteringActorTypes,
    SceneSleepRecentering,
    testing::Values(ActorType::Soft, ActorType::Shell, ActorType::Rod),
    ActorTypeName);

/***************************************************************************************************
  Stacks
*/

namespace {
struct StackParam {
  ActorType bottom = {};
  ActorType top = {};
};

class SceneSleepStack : public SceneSleep, public testing::WithParamInterface<StackParam> {};
} // namespace

TEST_P(SceneSleepStack, SettlesThenSleepsTogether) {
  // Let the bottom actor settle first, then drop the top one just above it.
  Actor const* bottom = CreateTestActor(GetParam().bottom);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(bottom));
  real const bottomTop = bottom->GetAabbWorld(test::ExpectOK{}).GetMax()[1];
  Actor const* top = CreateTestActor(GetParam().top, /*bottomHeight=*/bottomTop + 0.005_r);
  StepUntilAsleep(top);
  EXPECT_EQ(GetIsland(top), GetIsland(bottom));
  EXPECT_GT(GetPosition(top)[1], GetPosition(bottom)[1])
      << "The top actor should rest on the bottom one.";
}

INSTANTIATE_TEST_SUITE_P(
    Pairs,
    SceneSleepStack,
    testing::Values(
        StackParam{ActorType::Rigid, ActorType::Rigid},
        StackParam{ActorType::Rigid, ActorType::Soft},
        StackParam{ActorType::Soft, ActorType::Rigid},
        StackParam{ActorType::Rigid, ActorType::Shell}),
    [](testing::TestParamInfo<StackParam> const& info) {
      return std::string(SReflect::EnumToString(info.param.top)) + "On" +
          SReflect::EnumToString(info.param.bottom);
    });

TEST_F(SceneSleep, ManyIslandsSleepInTheSameStep) {
  _scene->SetGravity(Real3{});
  RigidActorParams params;
  params.shape = CreateCubeShape(0.2_r);
  params.colliderType = ColliderType::Box;
  DynamicArray<Actor const*> actors;
  for (int i = 0; i < 100; ++i) {
    // 1 m apart, so each actor is its own island.
    params.worldFromLocal.SetTranslation(Real3{StaticCast<real>(i), 1_r, 0_r});
    actors.push_back(_scene->CreateRigidActor(params, test::ExpectOK{}));
  }
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actors[0]));
  for (Actor const* actor : actors) {
    EXPECT_FALSE(IsAwake(actor));
  }
}

/***************************************************************************************************
  Sleep parameters
*/

TEST_F(SceneSleep, SleepsAfterMinStepsAtMaxThreshold) {
  experimental::SetSleepParams(
      _scene,
      {.canSleep = true, .sleepThreshold = 1.0, .minStepsBeforeSleep = 5},
      test::ExpectOK{});
  _scene->SetGravity(Real3{});
  Actor const* actor = CreateTestActor(ActorType::Rigid, /*bottomHeight=*/1_r);
  // At rest from the start, and steps that take zero solver iterations record exactly 1, so it
  // sleeps as soon as the minimum step count allows.
  StepUntilAsleep(actor, /*maxSteps=*/6);
}

TEST_F(SceneSleep, DisablingSleepWakesAllIslands) {
  experimental::SetSleepParams(
      _scene, {.canSleep = true, .minStepsBeforeSleep = 2}, test::ExpectOK{});
  _scene->SetGravity(Real3{});
  Actor const* actor = CreateTestActor(ActorType::Rigid);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor, /*maxSteps=*/3));
  experimental::SetSleepParams(
      _scene, {.canSleep = false, .minStepsBeforeSleep = 2}, test::ExpectOK{});
  StepExpectAwake(actor, 3);
}

/***************************************************************************************************
  Waking up
*/

TEST_F(SceneSleep, FallingActorWakesSleepingActor) {
  Actor const* bottom = CreateTestActor(ActorType::Rigid);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(bottom));
  Actor const* top = CreateTestActor(ActorType::Rigid, /*bottomHeight=*/0.5_r);
  // Islands merge when the falling actor gets close.
  StepExpectAwake(bottom, 1);
  EXPECT_EQ(GetIsland(top), GetIsland(bottom));
  StepUntilAsleep(bottom);
}

TEST_F(SceneSleep, DestroyingSupportWakesStack) {
  Actor* bottom = CreateTestActor(ActorType::Rigid);
  Aabb const support = bottom->GetAabbWorld(test::ExpectOK{});
  Actor const* top =
      CreateTestActor(ActorType::Rigid, /*bottomHeight=*/support.GetMax()[1] + 0.01_r);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(top));
  ASSERT_EQ(GetIsland(top), GetIsland(bottom));
  real const heightBefore = GetPosition(top)[1];
  _scene->DestroyActor(bottom);
  StepUntilAsleep(top);
  EXPECT_LT(GetPosition(top)[1], heightBefore - 0.5_r * support.GetSize()[1])
      << "The top actor should fall onto the ground.";
}

TEST_F(SceneSleep, RegisteringQueryWakesIsland) {
  Actor* actor = CreateTestActor(ActorType::Rigid);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));
  auto const query = actor->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  StepUntilAsleep(actor);
  real const weight = actor->GetMass(test::ExpectOK{}) * Norm(_scene->GetGravity());
  EXPECT_NEAR(weight, actor->GetContactForceWorld(test::ExpectOK{})[1], 0.01_r * weight);
  actor->CancelQuery(query);
}

/***************************************************************************************************
  Scene statistics
*/

TEST_F(SceneSleep, SolverStatsIgnoreSleepingIslands) {
  Actor const* actor = CreateTestActor(ActorType::Rigid);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));
  Step();
  auto const stats = _scene->GetSolverStats();
  EXPECT_EQ(0, stats.maxNonLinearIters);
  EXPECT_EQ(0.0, stats.residualNorm);
  EXPECT_EQ(ConvergenceStatus::None, stats.convergenceStatus);
}

TEST_F(SceneSleep, SolverStatsAggregateAwakeIslands) {
  Actor const* sleeping = CreateTestActor(ActorType::Rigid);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(sleeping));
  Actor const* falling = CreateTestActor(ActorType::Rigid, /*bottomHeight=*/2_r);
  Step();
  ASSERT_FALSE(IsAwake(sleeping));
  ASSERT_NE(GetIsland(sleeping), GetIsland(falling));
  EXPECT_NE(ConvergenceStatus::None, falling->GetConvergenceStatus());
  EXPECT_EQ(falling->GetConvergenceStatus(), _scene->GetSolverStats().convergenceStatus);
}

/***************************************************************************************************
  Wake-up conditions

  Each case calls the public API while the actor's island is asleep, and expects the call to wake
  the island. Then the case repeats the same call on the sleeping island, and expects a wake-up only
  if OnRepeat says so. Setters that ignore unchanged values should not wake the island again. To
  cover a new wake-up condition, add a case to GetWakeCases(). A case's apply must make the same
  call each time it runs, so use fixed values rather than values relative to the current state.
*/

namespace {
// Whether repeating a case's call with the same arguments wakes the island again.
enum class OnRepeat {
  WakeUp,
  NoWake, // The value is unchanged, so there is nothing to wake for.
  NotRepeatable, // The call cannot be repeated (e.g. removing something that no longer exists).
};

struct WakeContext {
  Scene* scene = nullptr;
  Actor* actor = nullptr;
  Actor* ground = nullptr;
  Constraint* constraint = nullptr; // Created by WakeCase::setup, if needed
  StateHandle state; // Captured by WakeCase::setup, if needed
  DynamicArray<uint8_t> stateBytes; // Captured by WakeCase::setup, if needed
};

struct WakeCase {
  std::string name;
  DynamicArray<ActorType> types; // Actor types the case applies to
  OnRepeat onRepeat;
  std::function<void(WakeContext&)> apply;
  std::function<void(WakeContext&)> setup = {}; // Runs before the actor first settles
};

struct WakeTestParam {
  WakeCase const* wakeCase = nullptr;
  ActorType actorType = {};
};

class SceneSleepWake : public SceneSleep, public testing::WithParamInterface<WakeTestParam> {};
} // namespace

// Holds a rigid actor in place at its center of mass.
static void CreatePivotAtCenterOfMass(WakeContext& ctx) {
  RigidPivotPositionConstraintParams params;
  params.actor = ctx.actor->GetHandle();
  params.localPosition = ctx.actor->GetRigidCenterOfMassLocal(test::ExpectOK{});
  params.targetPosition = ctx.actor->GetRootTransform().TransformPoint(params.localPosition);
  ctx.constraint = ctx.scene->CreateRigidPivotPositionConstraint(params, test::ExpectOK{});
}

static void CreateRotationConstraint(WakeContext& ctx) {
  RigidPivotRotationConstraintParams params;
  params.actor = ctx.actor->GetHandle();
  ctx.constraint = ctx.scene->CreateRigidPivotRotationConstraint(params, test::ExpectOK{});
}

// Holds a rigid actor in place with a transform target, which has a position and a rotation.
static void CreatePivotToRigidTarget(WakeContext& ctx) {
  RigidPivotToRigidTargetConstraintParams params;
  params.actor = ctx.actor->GetHandle();
  params.targetTransform = ctx.actor->GetCenterOfMassTransform(test::ExpectOK{});
  ctx.constraint = ctx.scene->CreateRigidPivotToRigidTargetConstraint(params, test::ExpectOK{});
}

static void CreateSingleDofTargetConstraint(WakeContext& ctx) {
  ArticulatedSingleDofTargetConstraintParams params;
  params.actor = ctx.actor->GetHandle();
  ctx.constraint = ctx.scene->CreateArticulatedSingleDofTargetConstraint(params, test::ExpectOK{});
}

static void AddPoseController(WakeContext& ctx) {
  ctx.actor->AddArticulatedPoseController(PoseControllerParams{}, test::ExpectOK{});
}

static void SetZeroExternalForce(WakeContext& ctx) {
  int const dof = 0;
  real const force = 0_r;
  ctx.actor->SetExternalForcesOnDofs(Span{&dof, 1}, Span{&force, 1}, test::ExpectOK{});
}

static void SetPairFrictionOverride(WakeContext& ctx) {
  ContactPairParamsOverride const params{.coulombFrictionCoefficient = 0.25_r};
  ctx.scene->SetContactPairParamsOverride(
      ctx.actor->GetHandle(), ctx.ground->GetHandle(), params, test::ExpectOK{});
}

// Zeros except for one entry.
static DynamicArray<real> MakeUnitArray(int size, int index, real value) {
  DynamicArray<real> values(size, 0_r);
  values[index] = value;
  return values;
}

static DynamicArray<WakeCase> const& GetWakeCases() {
  DynamicArray<ActorType> const kAll = GetAllActorTypes();
  DynamicArray<ActorType> const kRigidOnly = {ActorType::Rigid};
  DynamicArray<ActorType> const kRigidAndSoft = {ActorType::Rigid, ActorType::Soft};
  DynamicArray<ActorType> const kDeformables = {ActorType::Soft, ActorType::Shell, ActorType::Rod};
  DynamicArray<ActorType> const kArticulatedOnly = {ActorType::Articulated};
  DynamicArray<ActorType> const kWithContactParams = {
      ActorType::Rigid, ActorType::Soft, ActorType::Shell, ActorType::Rod};
  DynamicArray<ActorType> const kWithExternalForces = {
      ActorType::Rigid, ActorType::Shell, ActorType::Rod, ActorType::Articulated};
  DynamicArray<ActorType> const kWithBoundaryConditions = {
      ActorType::Rigid, ActorType::Soft, ActorType::Shell, ActorType::Articulated};
  static DynamicArray<WakeCase> const cases = {
      // Actor state
      {"SetRootTransform",
       kAll,
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         c.actor->SetRootTransform(TransformRT{Real3{0_r, 1_r, 0_r}}, test::ExpectOK{});
       }},
      {"SetCenterOfMassTransform",
       kRigidOnly,
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         c.actor->SetCenterOfMassTransform(TransformRT{Real3{0_r, 1_r, 0_r}}, test::ExpectOK{});
       }},
      {"SetVelocity",
       kRigidAndSoft,
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         c.actor->SetVelocity(Real3{0_r, 0.01_r, 0_r}, Real3{}, test::ExpectOK{});
       }},
      {"SetNodeVelocitiesLocal",
       kDeformables,
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         auto const velocities = MakeUnitArray(c.actor->GetNumDofs(), 1, 0.01_r);
         c.actor->SetNodeVelocitiesLocal(MakeConstSpan(velocities), test::ExpectOK{});
       }},
      {"SetDisplacements",
       {ActorType::Soft, ActorType::Shell},
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         int const size = isize(c.actor->GetDisplacements(test::ExpectOK{}));
         auto const displacements = MakeUnitArray(size, 1, 0.001_r);
         c.actor->SetDisplacements(MakeConstSpan(displacements), test::ExpectOK{});
       }},
      {"SetRodPose",
       {ActorType::Rod},
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         // Translating every node by the same y offset keeps the frame axes valid.
         auto pose = test::GetRodPose(*c.actor);
         for (int i = 1; i < isize(pose.displacements); i += kSpaceDim3) {
           pose.displacements[i] += 0.001_r;
         }
         c.actor->SetRodPose(pose.displacements, pose.elementFrameAxes, test::ExpectOK{});
       }},
      {"SetNodePositionsLocal",
       {ActorType::Soft},
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         MeshDataView const mesh = c.actor->GetMesh();
         DynamicArray<real> positions;
         positions.assign(mesh.coordinates.begin(), mesh.coordinates.end());
         positions[1] += 0.001_r;
         c.actor->SetNodePositionsLocal(MakeConstSpan(positions), test::ExpectOK{});
       }},
      {"SetZeroDisplacementsAndVelocities",
       kDeformables,
       OnRepeat::WakeUp,
       [](WakeContext& c) { c.actor->SetZeroDisplacementsAndVelocities(test::ExpectOK{}); }},
      {"SetArticulatedPoseFromJoints",
       kArticulatedOnly,
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         auto const pose = MakeUnitArray(c.actor->GetNumDofs(), 1, 0.1_r);
         c.actor->SetArticulatedPoseFromJoints(MakeConstSpan(pose), test::ExpectOK{});
       }},
      {"SetArticulatedJointVelocities",
       kArticulatedOnly,
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         auto const velocities = MakeUnitArray(c.actor->GetNumDofs(), 1, 0.01_r);
         c.actor->SetArticulatedJointVelocities(MakeConstSpan(velocities), test::ExpectOK{});
       }},
      // Actor properties
      {"SetDensity",
       kRigidAndSoft,
       OnRepeat::NoWake,
       [](WakeContext& c) { c.actor->SetDensity(1500_r, test::ExpectOK{}); }},
      {"SetInertiaProperties",
       kRigidOnly,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         c.actor->SetInertiaProperties(
             10_r,
             c.actor->GetRigidCenterOfMassLocal(test::ExpectOK{}),
             c.actor->GetRigidMomentOfInertiaLocal(test::ExpectOK{}),
             test::ExpectOK{});
       }},
      {"SetSoftMaterialParams",
       {ActorType::Soft},
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         auto params = c.actor->GetSoftMaterialParams(test::ExpectOK{});
         params.density = 1500_r;
         c.actor->SetSoftMaterialParams(params, test::ExpectOK{});
       }},
      {"SetRecenteringParams",
       kDeformables,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         auto params = c.actor->GetRecenteringParams();
         params.translationEpsilon = 0.01_r;
         c.actor->SetRecenteringParams(params, test::ExpectOK{});
       }},
      {"SetContactParams",
       kWithContactParams,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         auto params = c.actor->GetContactParams(test::ExpectOK{});
         params.coulombFrictionCoefficient = 0.25_r;
         c.actor->SetContactParams(params, test::ExpectOK{});
       }},
      {.name = "SetContactParams_CollidingPadding",
       .types = kWithContactParams,
       .onRepeat = OnRepeat::NoWake,
       .apply =
           [](WakeContext& c) {
             auto params = c.actor->GetContactParams(test::ExpectOK{});
             params.collidingPenaltyThresholdPadding = 0.01_r;
             c.actor->SetContactParams(params, test::ExpectOK{});
           }},
      {"SetContactLayer",
       kWithContactParams,
       OnRepeat::NoWake,
       [](WakeContext& c) { c.actor->SetContactLayer("WakeTestLayer"); }},
      {"SetArticulatedJointFrictionParams",
       kArticulatedOnly,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         auto const current = c.actor->GetArticulatedJointFrictionParams(test::ExpectOK{});
         DynamicArray<ArticulatedJointFrictionParams> friction;
         friction.assign(current.begin(), current.end());
         friction[0].viscous = 0.05_r;
         c.actor->SetArticulatedJointFrictionParams(MakeConstSpan(friction), test::ExpectOK{});
       }},
      {"SetArticulatedJointInertiaParams",
       kArticulatedOnly,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         auto const current = c.actor->GetArticulatedJointInertiaParams(test::ExpectOK{});
         DynamicArray<real> inertia;
         inertia.assign(current.begin(), current.end());
         inertia[0] = 0.05_r;
         c.actor->SetArticulatedJointInertiaParams(MakeConstSpan(inertia), test::ExpectOK{});
       }},
      {"EnableNewtonEulerInertia",
       {ActorType::Rigid, ActorType::Articulated},
       OnRepeat::WakeUp,
       [](WakeContext& c) { EnableNewtonEulerInertia(c.actor, true, test::ExpectOK{}); }},
      {"SetActorPreconditionerType",
       kAll,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         SetActorPreconditionerType(c.actor, PreconditionerType::Jacobi, test::ExpectOK{});
       }},
      // Forces, boundary conditions and queries
      {"SetExternalForcesOnDofs", kWithExternalForces, OnRepeat::WakeUp, SetZeroExternalForce},
      {"ClearExternalForces",
       kWithExternalForces,
       OnRepeat::NoWake,
       [](WakeContext& c) { c.actor->ClearExternalForces(); },
       SetZeroExternalForce},
      {"AddBoundaryConditionDofsWorld",
       kRigidOnly,
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         // The first 3 DoFs of a rigid actor are its center of mass in world space.
         DynamicArray<int> const dofs = {0, 1, 2};
         auto const values = GetAllDofValues(c.actor);
         c.actor->AddBoundaryConditionDofsWorld(
             MakeConstSpan(dofs), Span{values.data(), 3}, test::ExpectOK{});
       }},
      {"AddBoundaryConditionNodesWorld",
       {ActorType::Soft, ActorType::Shell},
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         DynamicArray<int> const nodes = {0};
         DynamicArray<Real3> const positions = {SceneSleep::GetRestNodePositionsWorld(c.actor)[0]};
         c.actor->AddBoundaryConditionNodesWorld(
             MakeConstSpan(nodes), Flatten(MakeConstSpan(positions)), test::ExpectOK{});
       }},
      {"ClearBoundaryConditions",
       kWithBoundaryConditions,
       OnRepeat::NoWake,
       [](WakeContext& c) { c.actor->ClearBoundaryConditions(); },
       [](WakeContext& c) { SceneSleep::ConstrainInPlace(c.scene, c.actor); }},
      {"RegisterQuery",
       kRigidAndSoft,
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         c.actor->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
       }},
      {"RegisterQueryAndCompute_SurfaceNodePositions",
       kRigidOnly,
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         c.actor->RegisterQueryAndCompute(QueryType::SurfaceNodePositions, test::ExpectOK{});
       }},
      {"RegisterQueryAndCompute_NodePositions",
       {ActorType::Soft},
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         c.actor->RegisterQueryAndCompute(QueryType::NodePositions, test::ExpectOK{});
       }},
      // Pose controllers
      {"AddArticulatedPoseController",
       kArticulatedOnly,
       OnRepeat::NotRepeatable,
       AddPoseController},
      {"SetArticulatedTargetPose",
       kArticulatedOnly,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         auto const pose = MakeUnitArray(c.actor->GetNumDofs(), 1, 0.1_r);
         c.actor->SetArticulatedTargetPose(MakeConstSpan(pose), test::ExpectOK{});
       },
       AddPoseController},
      {"SetArticulatedTargetLinkTransforms",
       kArticulatedOnly,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         auto const numLinks = c.actor->GetNestedLinkActors(test::ExpectOK{}).size();
         DynamicArray<TransformRT> const targets(numLinks, TransformRT{Real3{0_r, 1_r, 0_r}});
         c.actor->SetArticulatedTargetLinkTransforms(MakeConstSpan(targets), test::ExpectOK{});
       },
       AddPoseController},
      {"RemoveArticulatedPoseController",
       kArticulatedOnly,
       OnRepeat::NotRepeatable,
       [](WakeContext& c) { c.actor->RemoveArticulatedPoseController(test::ExpectOK{}); },
       AddPoseController},
      // Constraints
      {"CreateConstraint", kRigidOnly, OnRepeat::WakeUp, CreatePivotAtCenterOfMass},
      {"DestroyConstraint",
       kRigidOnly,
       OnRepeat::NotRepeatable,
       [](WakeContext& c) { c.scene->DestroyConstraint(c.constraint); },
       CreatePivotAtCenterOfMass},
      {"Constraint_SetStiffness",
       kRigidOnly,
       OnRepeat::NoWake,
       [](WakeContext& c) { c.constraint->SetStiffness(12345_r, test::ExpectOK{}); },
       CreatePivotAtCenterOfMass},
      {"Constraint_SetDamping",
       kRigidOnly,
       OnRepeat::NoWake,
       [](WakeContext& c) { c.constraint->SetDamping(0.5_r, test::ExpectOK{}); },
       CreatePivotAtCenterOfMass},
      {"Constraint_SetSaturation",
       kRigidOnly,
       OnRepeat::NoWake,
       [](WakeContext& c) { c.constraint->SetSaturation(-2_r, test::ExpectOK{}); },
       CreatePivotAtCenterOfMass},
      {"Constraint_SetTargetPosition",
       kRigidOnly,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         c.constraint->SetTargetPosition(Real3{0_r, 1_r, 0_r}, test::ExpectOK{});
       },
       CreatePivotAtCenterOfMass},
      {"Constraint_SetTargetPosition_RigidTarget",
       kRigidOnly,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         c.constraint->SetTargetPosition(Real3{0_r, 1_r, 0_r}, test::ExpectOK{});
       },
       CreatePivotToRigidTarget},
      {"Constraint_SetTargetRotation",
       kRigidOnly,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         c.constraint->SetTargetRotation(
             Quaternion::FromRotationVector(Real3{0.001_r, 0_r, 0_r}), test::ExpectOK{});
       },
       CreateRotationConstraint},
      {"Constraint_SetTargetRotation_RigidTarget",
       kRigidOnly,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         c.constraint->SetTargetRotation(
             Quaternion::FromRotationVector(Real3{0.001_r, 0_r, 0_r}), test::ExpectOK{});
       },
       CreatePivotToRigidTarget},
      {"Constraint_SetTargetDof",
       kArticulatedOnly,
       OnRepeat::NoWake,
       [](WakeContext& c) { c.constraint->SetTargetDof(0.001_r, test::ExpectOK{}); },
       CreateSingleDofTargetConstraint},
      {"Constraint_UpdateOldTarget",
       kRigidOnly,
       OnRepeat::WakeUp,
       [](WakeContext& c) { c.constraint->UpdateOldTarget(test::ExpectOK{}); },
       CreatePivotAtCenterOfMass},
      {"Constraint_RegisterQuery",
       kRigidOnly,
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         c.constraint->RegisterQuery(QueryType::ConstraintForce, test::ExpectOK{});
       },
       CreatePivotAtCenterOfMass},
      // Scene settings
      {"SetGravity",
       kAll,
       OnRepeat::NoWake,
       [](WakeContext& c) { c.scene->SetGravity(Real3{0_r, -5_r, 0_r}); }},
      {"SetSolverParams",
       kAll,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         SolverParams params = c.scene->GetSolverParams();
         params.nonLinearSolver.maxIter = 37;
         c.scene->SetSolverParams(params, test::ExpectOK{});
       }},
      {"SetSleepParams",
       kAll,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         auto params = experimental::GetSleepParams(c.scene, test::ExpectOK{});
         params.sleepThreshold = 0.25;
         experimental::SetSleepParams(c.scene, params, test::ExpectOK{});
       }},
      {"EnableLayerContact",
       kAll,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         c.scene->EnableLayerContactSymmetric("unusedA", "unusedB", false, test::ExpectOK{});
       }},
      {"EnableActorContact",
       kAll,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         c.scene->EnableActorContactSymmetric(
             c.actor->GetHandle(),
             c.actor->GetHandle(),
             false,
             IncludeNestedActors::No,
             test::ExpectOK{});
       }},
      {"SetContactPairParamsOverride",
       kWithContactParams,
       OnRepeat::NoWake,
       SetPairFrictionOverride},
      {"ClearContactPairParamsOverride",
       kWithContactParams,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         c.scene->ClearContactPairParamsOverride(
             c.actor->GetHandle(), c.ground->GetHandle(), test::ExpectOK{});
       },
       SetPairFrictionOverride},
      {"RestoreState",
       kAll,
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         c.scene->RestoreState(c.state, /*releaseImmediately=*/false, test::ExpectOK{});
       },
       [](WakeContext& c) { c.state = c.scene->CaptureState(test::ExpectOK{}); }},
      {"RestoreStateFromBytes",
       kAll,
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         c.scene->RestoreStateFromBytes(MakeConstSpan(c.stateBytes), test::ExpectOK{});
       },
       [](WakeContext& c) { c.scene->CaptureStateToBytes(c.stateBytes, test::ExpectOK{}); }},
      // Static ground
      {"Ground_SetRootTransform",
       kAll,
       OnRepeat::WakeUp,
       [](WakeContext& c) {
         c.ground->SetRootTransform(TransformRT{Real3{0_r, -0.001_r, 0_r}}, test::ExpectOK{});
       }},
      {"Ground_SetVelocity",
       kAll,
       OnRepeat::WakeUp,
       [](WakeContext& c) { c.ground->SetVelocity(Real3{}, Real3{}, test::ExpectOK{}); }},
      {"Ground_SetContactParams",
       kAll,
       OnRepeat::NoWake,
       [](WakeContext& c) {
         auto params = c.ground->GetContactParams(test::ExpectOK{});
         params.coulombFrictionCoefficient = 0.25_r;
         c.ground->SetContactParams(params, test::ExpectOK{});
       }},
      {"Ground_SetContactLayer",
       kAll,
       OnRepeat::NoWake,
       [](WakeContext& c) { c.ground->SetContactLayer("GroundWakeTestLayer"); }},
      {"ClearBoundaryConditions_KeepsPermanent",
       kRigidOnly,
       OnRepeat::NoWake,
       [](WakeContext& c) { c.actor->ClearBoundaryConditions(); },
       [](WakeContext& c) {
         auto const values = GetAllDofValues(c.actor);
         DynamicArray<int> const permanent = {0, 1, 2};
         DynamicArray<int> const temporary = {3, 4, 5};
         c.actor->AddBoundaryConditionDofsWorldPermanent(
             MakeConstSpan(permanent), Span{values.data(), 3}, test::ExpectOK{});
         c.actor->AddBoundaryConditionDofsWorld(
             MakeConstSpan(temporary), Span{values.data() + 3, 3}, test::ExpectOK{});
       }},
  };
  return cases;
}

static DynamicArray<WakeTestParam> GetWakeTestParams() {
  DynamicArray<WakeTestParam> params;
  for (auto const& wakeCase : GetWakeCases()) {
    for (auto type : wakeCase.types) {
      params.push_back({&wakeCase, type});
    }
  }
  return params;
}

TEST_P(SceneSleepWake, WakesIsland) {
  auto const& [wakeCase, actorType] = GetParam();
  WakeContext context{.scene = _scene, .actor = CreateTestActor(actorType), .ground = _ground};
  if (wakeCase->setup) {
    wakeCase->setup(context);
  }
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(context.actor));

  wakeCase->apply(context);
  ApplyWakeRequests();
  ASSERT_TRUE(IsAwake(context.actor));
  if (wakeCase->onRepeat == OnRepeat::NotRepeatable) {
    return;
  }

  GetRegistry().get<CIslandSleepState>(GetIsland(context.actor)).numConsecutiveRestSteps =
      experimental::GetSleepParams(_scene, test::ExpectOK{}).minStepsBeforeSleep;

  wakeCase->apply(context);
  ApplyWakeRequests();
  EXPECT_EQ(wakeCase->onRepeat == OnRepeat::WakeUp, IsAwake(context.actor))
      << "Unexpected wake-up state after repeating the call.";
}

INSTANTIATE_TEST_SUITE_P(
    AllCases,
    SceneSleepWake,
    testing::ValuesIn(GetWakeTestParams()),
    [](testing::TestParamInfo<WakeTestParam> const& info) {
      return info.param.wakeCase->name + "_" + SReflect::EnumToString(info.param.actorType);
    });

TEST_F(SceneSleep, ReadingStateDoesNotWake) {
  Actor* actor = CreateTestActor(ActorType::Rigid);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));
  (void)actor->GetRootTransform();
  (void)GetAllDofValues(actor);
  (void)actor->GetContactParams(test::ExpectOK{});
  actor->SetUserData(nullptr);
  Step();
  EXPECT_FALSE(IsAwake(actor));
}

TEST_F(SceneSleep, RejectedSetRodPoseDoesNotWake) {
  Actor* rod = CreateTestActor(ActorType::Rod);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(rod));
  rod->SetRodPose({}, {}, test::ExpectNotOK{});
  ApplyWakeRequests();
  EXPECT_FALSE(IsAwake(rod));
}

TEST_F(SceneSleep, WakingOneIslandLeavesOthersAsleep) {
  Actor* a = CreateTestActor(ActorType::Rigid);
  Actor* b = CreateTestActor(ActorType::Rigid);
  b->SetRootTransform(
      TransformRT{b->GetRootTransform().GetTranslation() + Real3{2_r, 0_r, 0_r}}, test::ExpectOK{});
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(a));
  ASSERT_FALSE(IsAwake(b));
  ASSERT_NE(GetIsland(a), GetIsland(b));
  a->SetVelocity(Real3{0_r, 0.01_r, 0_r}, Real3{}, test::ExpectOK{});
  Step();
  EXPECT_TRUE(IsAwake(a));
  EXPECT_FALSE(IsAwake(b));
}

/***************************************************************************************************
  Static actors
*/

TEST_F(SceneSleep, AddingStaticActorNearbyWakesActor) {
  Actor const* actor = CreateTestActor(ActorType::Rigid);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));
  // A box of the same size, just beside the actor.
  Aabb const aabb = actor->GetAabbWorld(test::ExpectOK{});
  CreateStaticBox(Real3{aabb.GetMax()[0] + 0.001_r, 0_r, aabb.GetMin()[2]}, aabb.GetSize()[0]);
  Step();
  EXPECT_TRUE(IsAwake(actor));
}

TEST_F(SceneSleep, MovingStaticActorNearbyWakesActor) {
  Actor* farBox = CreateStaticBox(Real3{5_r, 0_r, 5_r}, 0.2_r);
  Actor const* actor = CreateTestActor(ActorType::Rigid);
  Aabb const aabb = actor->GetAabbWorld(test::ExpectOK{});
  Step();

  sleep::ForceSleep(GetRegistry(), GetIsland(actor));
  Step();
  EXPECT_FALSE(IsAwake(actor));

  // A static actor moving to be near the dynamic actor wakes it up.
  farBox->SetRootTransform(
      TransformRT{Real3{aabb.GetMax()[0] + 0.001_r, 0_r, aabb.GetMin()[2]}}, test::ExpectOK{});
  Step();
  EXPECT_TRUE(IsAwake(actor));

  sleep::ForceSleep(GetRegistry(), GetIsland(actor));
  Step();
  EXPECT_FALSE(IsAwake(actor));

  // A static actor moving away from the dynamic actor wakes it up.
  farBox->SetRootTransform(TransformRT{Real3{5_r, 0_r, 5_r}}, test::ExpectOK{});
  Step();
  EXPECT_TRUE(IsAwake(actor));
}

TEST_F(SceneSleep, ChangingDistantStaticActorLeavesActorAsleep) {
  Actor* farBox = CreateStaticBox(Real3{5_r, 0_r, 5_r}, 0.2_r);
  Actor const* actor = CreateTestActor(ActorType::Rigid);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));
  farBox->SetContactParams(farBox->GetContactParams(test::ExpectOK{}), test::ExpectOK{});
  farBox->SetRootTransform(TransformRT{Real3{6_r, 0_r, 6_r}}, test::ExpectOK{});
  Step();
  EXPECT_FALSE(IsAwake(actor));
}

TEST_F(SceneSleep, AddingStaticArticulatedLinkNearbyWakesActor) {
  _scene->SetGravity(Real3{});
  Actor const* actor = CreateTestActor(ActorType::Rigid, /*bottomHeight=*/1_r);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));

  Aabb const aabb = actor->GetAabbWorld(test::ExpectOK{});
  Actor* articulation = CreateStaticArticulatedBox(
      Real3{aabb.GetMax()[0] + 0.001_r, aabb.GetMin()[1], aabb.GetMin()[2]}, aabb.GetSize()[0]);
  auto const links = articulation->GetNestedLinkActors(test::ExpectOK{});
  ASSERT_EQ(1, links.size());
  Actor const* link = _scene->GetActor(links[0]);
  ASSERT_NE(nullptr, link);
  EXPECT_TRUE(link->IsStatic());
  EXPECT_EQ(GetIsland(articulation), GetIsland(link));
  EXPECT_NE(GetIsland(actor), GetIsland(link));

  Step();
  EXPECT_TRUE(IsAwake(actor));
}

TEST_F(SceneSleep, ChangingStaticArticulatedLinkContactWakesActor) {
  _scene->SetGravity(Real3{});
  Actor* actor = CreateTestActor(ActorType::Rigid, /*bottomHeight=*/1_r);
  Aabb const aabb = actor->GetAabbWorld(test::ExpectOK{});
  Actor* articulation = CreateStaticArticulatedBox(
      Real3{aabb.GetMax()[0] + 0.001_r, aabb.GetMin()[1], aabb.GetMin()[2]}, aabb.GetSize()[0]);
  Actor* link = _scene->GetActor(articulation->GetNestedLinkActors(test::ExpectOK{})[0]);
  ASSERT_NE(nullptr, link);
  ASSERT_TRUE(link->IsStatic());
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));

  auto params = link->GetContactParams(test::ExpectOK{});
  params.coulombFrictionCoefficient = 0_r;
  link->SetContactParams(params, test::ExpectOK{});
  Step();
  EXPECT_TRUE(IsAwake(actor));
}

TEST_F(SceneSleep, MovingStaticArticulatedLinkNearbyWakesActor) {
  _scene->SetGravity(Real3{});
  Actor* articulation = CreateStaticArticulatedBox(Real3{5_r, 1_r, 5_r}, 0.2_r);
  Actor const* actor = CreateTestActor(ActorType::Rigid, /*bottomHeight=*/1_r);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));

  Aabb const aabb = actor->GetAabbWorld(test::ExpectOK{});
  articulation->SetRootTransform(
      TransformRT{Real3{aabb.GetMax()[0] + 0.001_r, aabb.GetMin()[1], aabb.GetMin()[2]}},
      test::ExpectOK{});
  Step();
  EXPECT_TRUE(IsAwake(actor));
}

TEST_F(SceneSleep, DestroyingStaticArticulatedLinkWakesActor) {
  _scene->SetGravity(Real3{});
  Actor* actor = CreateTestActor(ActorType::Rigid, /*bottomHeight=*/1_r);
  Aabb const aabb = actor->GetAabbWorld(test::ExpectOK{});
  Actor* articulation = CreateStaticArticulatedBox(
      Real3{aabb.GetMax()[0] + 0.001_r, aabb.GetMin()[1], aabb.GetMin()[2]}, aabb.GetSize()[0]);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));

  _scene->DestroyActor(articulation);
  Step();
  EXPECT_TRUE(IsAwake(actor));
}

/***************************************************************************************************
  Zero step
*/

// Step(0.0) reproduces the forces of the last step and leaves the sleep state as it found it, so
// the next step resumes as if Step(0.0) had not been called.
TEST_F(SceneSleep, ZeroStepPreservesSleepState) {
  Actor const* sleeper = CreateTestActor(ActorType::Rigid);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(sleeper));
  // High enough to stay awake, and in its own island, for the rest of the test.
  Actor* faller = CreateTestActor(ActorType::Rigid, /*bottomHeight=*/3_r);
  // Step(0.0) assembles forces only if a registered query requires it.
  ASSERT_TRUE(faller->IsQuerySupported(QueryType::TotalContactForce));
  faller->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  auto& reg = GetRegistry();
  std::map<entt::entity, ColumnVector<real>> residuals;
  std::mutex mutex;
  island::SetTestCallback_AssembledResidual(
      reg, [&](entt::entity island, ColumnVectorView<real const> residual) {
        std::lock_guard lock(mutex);
        residuals[island] = residual;
      });
  MOCHI_DEFER(island::SetTestCallback_AssembledResidual(reg, {}));
  Step();
  ASSERT_FALSE(IsAwake(sleeper));
  ASSERT_TRUE(IsAwake(faller));
  ASSERT_NE(GetIsland(sleeper), GetIsland(faller));
  ASSERT_EQ(1u, residuals.count(GetIsland(faller)));
  EXPECT_EQ(0u, residuals.count(GetIsland(sleeper))) << "A step does not solve sleeping islands";

  auto const getRestSteps = [&](Actor const* actor) {
    return reg.get<CIslandSleepState const>(GetIsland(actor)).numConsecutiveRestSteps;
  };
  ColumnVector<real> const forces = residuals.at(GetIsland(faller));
  int const sleeperRestSteps = getRestSteps(sleeper);
  int const fallerRestSteps = getRestSteps(faller);
  residuals.clear();

  _scene->Step(0.0);

  ASSERT_EQ(1u, residuals.count(GetIsland(faller)));
  ColumnVector<real> const& zeroStepForces = residuals.at(GetIsland(faller));
  ASSERT_EQ(isize(forces), isize(zeroStepForces));
  for (int i = 0; i < isize(forces); ++i) {
    EXPECT_EQ(forces[i], zeroStepForces[i]) << "DOF " << i;
  }
  EXPECT_EQ(1u, residuals.count(GetIsland(sleeper))) << "Step(0.0) must evaluate sleeping islands";
  EXPECT_FALSE(IsAwake(sleeper));
  EXPECT_TRUE(IsAwake(faller));
  EXPECT_EQ(sleeperRestSteps, getRestSteps(sleeper));
  EXPECT_EQ(fallerRestSteps, getRestSteps(faller));
  EXPECT_FALSE(reg.ctx<CWakeUpRequests const>().wakeAll);

  Step();
  EXPECT_FALSE(IsAwake(sleeper)) << "Step(0.0) must not wake sleeping islands";
}

// A sleeping actor gets its query data from Step(0.0) without waking up. A wake request made before
// Step(0.0) stays pending, and wakes the island in the next step.
TEST_F(SceneSleep, ZeroStepKeepsPendingWakeRequests) {
  Actor* actor = CreateTestActor(ActorType::Rigid);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));
  auto const query = actor->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  ASSERT_TRUE(GetRegistry().all_of<TagWakeUp>(GetIsland(actor)));

  _scene->Step(0.0);
  EXPECT_FALSE(IsAwake(actor));
  EXPECT_TRUE(GetRegistry().all_of<TagWakeUp>(GetIsland(actor)));
  real const weight = actor->GetMass(test::ExpectOK{}) * Norm(_scene->GetGravity());
  EXPECT_NEAR(weight, actor->GetContactForceWorld(test::ExpectOK{})[1], 0.01_r * weight);

  Step();
  EXPECT_TRUE(IsAwake(actor));
  actor->CancelQuery(query);
}
