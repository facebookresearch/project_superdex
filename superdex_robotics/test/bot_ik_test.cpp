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

#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/ik/bot_ik.h>
#include <superdex_robotics/ik/bot_ik_config.h>
#include <superdex_robotics/utils/bot_utils.h>

#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <string_view>
#include <utility>

using namespace mochi;
using namespace mochi::test;
using namespace superdex::robotics;

namespace {

constexpr real kPoseTolerance = 2e-5_r;
constexpr real kSolveTolerance = 2e-3_r;
constexpr real kGoalStiffness = 1000_r;
constexpr int kArm = 2;
constexpr int kSlide = 3;
constexpr int kTip = 4;

template <typename... Args>
auto CreateBotIK(
    Context* context,
    BotPrefab const& prefab,
    BotIKCreateParams const& params,
    Args&&... args) {
  return std::unique_ptr<BotIK, decltype(&BotIK::Destroy)>{
      BotIK::Create(context, prefab, params, std::forward<Args>(args)...), &BotIK::Destroy};
}

// Reuse the file loader for unused operations; only shape resolution is synthetic.
struct TestBotLoader : FileBotLoader {
  mutable int shapeLoads = 0;
  int failOnShapeLoad = -1;

  ShapeHandle LoadShape(
      std::string_view path,
      Real3 const& bakeScale,
      TransformRT const& bakeTransform,
      Context* context,
      Error& error) const override {
    EXPECT_EQ(path, "test_tet");
    EXPECT_NEAR_EQ(bakeScale, (Real3{1_r, 1_r, 1_r}));
    EXPECT_NEAR_EQ(bakeTransform, TransformRT{});
    ++shapeLoads;
    if (shapeLoads == failOnShapeLoad) {
      MOCHI_ERROR_SET(error, "Shape loading failed on purpose");
      return {};
    }
    real const coordinates[] = {
        -0.4_r, -0.4_r, -0.4_r, 0.4_r, -0.4_r, -0.4_r, 0_r, 0.4_r, -0.4_r, 0_r, 0_r, 0.4_r};
    int const connectivity[] = {0, 1, 2, 3};
    return context->CreateTetMeshShape(coordinates, connectivity, error);
  }
};

BotLinkPrefab MakeLink(char const* name, int parent) {
  BotLinkPrefab link;
  link.name = name;
  link.parentLink = parent;
  link.mass = 1_r;
  link.centerOfMass = Real3{};
  link.momentOfInertia = Real6{1_r, 0_r, 0_r, 1_r, 0_r, 1_r};
  return link;
}

BotJointPrefab
MakeJoint(char const* name, ArticulatedJointType type, Real3 const& translation = {}) {
  BotJointPrefab joint;
  joint.name = name;
  joint.type = type;
  joint.axis =
      type == ArticulatedJointType::Prismatic ? Real3{1_r, 0_r, 0_r} : Real3{0_r, 0_r, 1_r};
  joint.parentLinkFromJoint = TransformRT{translation};
  return joint;
}

BotPrefab MakeRobot(ArticulatedJointType rootType = ArticulatedJointType::Hard) {
  BotPrefab prefab;
  prefab.name = "bot_ik_test";
  prefab.worldFromRoot = TransformRT{Quaternion::RotationY(0.25_r), Real3{1_r, 2_r, 3_r}};
  prefab.links = {
      MakeLink("root", kIndexNone),
      MakeLink("a_weld", 0),
      MakeLink("b_arm", 0),
      MakeLink("c_slide", kArm),
      MakeLink("d_tip", kSlide)};
  prefab.joints = {
      MakeJoint("root_joint", rootType),
      MakeJoint("weld_joint", ArticulatedJointType::Hard, {0_r, 0.2_r, 0_r}),
      MakeJoint("arm_joint", ArticulatedJointType::Revolute, {0.3_r, 0_r, 0_r}),
      MakeJoint("slide_joint", ArticulatedJointType::Prismatic, {0.5_r, 0_r, 0_r}),
      MakeJoint("tip_joint", ArticulatedJointType::Hard, {0.2_r, 0_r, 0_r})};
  prefab.defaultPose = {0.15_r, 0.1_r};
  RebuildBotData(prefab, ExpectOK{});
  return prefab;
}

// Independently authored FK for this test robot, including its nonidentity world placement.
DynamicArray<TransformRT> ExpectedLinks(TransformRT const& base, real angle, real extension) {
  TransformRT const arm = base * TransformRT{Quaternion::RotationZ(angle), Real3{0.3_r, 0_r, 0_r}};
  TransformRT const slide = arm * TransformRT{Real3{0.5_r + extension, 0_r, 0_r}};
  return {
      base,
      base * TransformRT{Real3{0_r, 0.2_r, 0_r}},
      arm,
      slide,
      slide * TransformRT{Real3{0.2_r, 0_r, 0_r}}};
}

void ExpectLinksNear(Span<TransformRT const> actual, Span<TransformRT const> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (int i = 0; i < isize(actual); ++i) {
    SCOPED_TRACE(i);
    EXPECT_TRUE(NearEqual(actual[i], expected[i], kSolveTolerance));
  }
}

void ExpectPoseNear(Span<real const> actual, Span<real const> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (int i = 0; i < isize(actual); ++i) {
    SCOPED_TRACE(i);
    EXPECT_NEAR(actual[i], expected[i], kSolveTolerance);
  }
}

experimental::IKSolverParams TestSolverParams() {
  experimental::IKSolverParams params;
  params.maxIter = 80;
  params.absTol = 1e-4_r;
  params.relTol = 1e-8_r;
  params.positionErrorThres = kSolveTolerance;
  params.rotationErrorThres = kSolveTolerance;
  return params;
}

BotIKCreateParams TestCreateParams() {
  BotIKCreateParams params;
  params.solverParams = TestSolverParams();
  return params;
}

void AddShapesAndContactOverrides(BotPrefab& prefab) {
  for (auto& link : prefab.links) {
    link.shapeFile = "test_tet";
    link.layer = link.parentLink == kIndexNone ? "authored_base" : "authored_links";
  }
  BotContactOverride override;
  override.linkA = "root";
  override.linkB = "b_arm";
  override.enable = true;
  prefab.contactOverrides.push_back(override);
  override.linkA = "a_weld";
  override.linkB = "c_slide";
  prefab.contactOverrides.push_back(override);
}

BotLinearTransmissionPrefab MakeTransmission() {
  BotLinearTransmissionPrefab transmission;
  transmission.name = "ratio_and_offset";
  transmission.jointIndices = {kArm, kSlide};
  transmission.jointCoefficients = {2_r, -1_r};
  transmission.jointAxisDisps = {0_r, 0_r};
  transmission.targetDisplacement = -0.1_r;
  transmission.stiffness = 4000_r;
  transmission.allowCompressiveForce = true;
  return transmission;
}

class BotIKTest : public testing::Test {
 protected:
  void SetUp() override {
    _mochiContext = mochi::CreateContext(0);
    ASSERT_NE(_mochiContext, nullptr);
  }

  void TearDown() override {
    if (_mochiContext != nullptr) {
      mochi::DestroyContext(_mochiContext);
      _mochiContext = nullptr;
    }
  }

  // A separate actor and experimental solver are the oracle, with transmissions installed through
  // the experimental API rather than RoboticsContext's prefab attachment path.
  void ExpectDirectParity(
      BotPrefab const& prefab,
      bool excludeShapes,
      Span<BotIKJointCouplingParams const> couplings = {}) {
    TestBotLoader loader;
    auto params = TestCreateParams();
    params.excludeCollisionShapes = excludeShapes;
    auto ik = CreateBotIK(_mochiContext, prefab, params, couplings, loader, ExpectOK{});
    ASSERT_NE(ik, nullptr);

    Scene* const scene = _mochiContext->CreateScene("BotIKDirectParity");
    ASSERT_NE(scene, nullptr);
    experimental::IKSolver* solver = nullptr;
    MOCHI_DEFER({
      if (solver != nullptr) {
        experimental::DestroyIKSolver(solver, _mochiContext, ExpectOK{});
      } else {
        _mochiContext->DestroyScene(scene);
      }
    });

    auto actorParams = BuildArticulatedActorParams(prefab, loader, _mochiContext, ExpectOK{});
    if (excludeShapes) {
      for (auto& link : actorParams.links) {
        link.shape = {};
        link.colliderType = ColliderType::None;
      }
      actorParams.skin.reset();
    }
    Actor* const actor = scene->CreateArticulatedActor(actorParams, ExpectOK{});
    ASSERT_NE(actor, nullptr);
    auto const seed = BuildArticulatedPoseFromBotPose(
        prefab, prefab.defaultPose, actor->GetNumDofs(), ExpectOK{});
    actor->SetArticulatedPoseFromJoints(seed, ExpectOK{});

    for (auto const& transmission : prefab.linearTransmissions) {
      experimental::LinearTransmissionParams direct;
      direct.jointIndices = transmission.jointIndices;
      direct.jointCoefficients = transmission.jointCoefficients;
      int const index = experimental::AddLinearTransmission(actor, direct, ExpectOK{});
      experimental::AttachDisplacementControlActuator(actor, index, transmission, ExpectOK{});
    }
    for (auto const& tendon : prefab.spatialTendons) {
      experimental::SpatialTendonParams direct;
      direct.routingElements = tendon.routingElements;
      int const index = experimental::AddSpatialTendon(actor, direct, ExpectOK{});
      experimental::AttachDisplacementControlActuator(actor, index, tendon, ExpectOK{});
    }
    for (auto const& coupling : couplings) {
      if (!coupling.enabled || coupling.stiffness == 0_r) {
        continue;
      }
      experimental::LinearTransmissionParams direct;
      // This independent oracle uses the known hinge/slider indices, not runtime name resolution.
      EXPECT_EQ(coupling.jointA, "arm_joint");
      EXPECT_EQ(coupling.jointB, "slide_joint");
      direct.jointIndices = {kArm, kSlide};
      direct.jointCoefficients = {coupling.ratio, -1_r};
      int const index = experimental::AddLinearTransmission(actor, direct, ExpectOK{});
      experimental::DisplacementControlActuatorParams actuator;
      actuator.targetDisplacement = -coupling.offset;
      actuator.stiffness = coupling.stiffness;
      actuator.damping = 0_r;
      actuator.allowCompressiveForce = true;
      experimental::AttachDisplacementControlActuator(actor, index, actuator, ExpectOK{});
    }
    scene->EnableActorContactSymmetric(
        actor->GetHandle(), actor->GetHandle(), false, IncludeNestedActors::Yes, ExpectOK{});
    solver = experimental::CreateIKSolver(scene, _mochiContext, ExpectOK{});
    ASSERT_NE(solver, nullptr);
    solver->SetSolverParams(params.solverParams);

    auto const links = actor->GetNestedLinkActors(ExpectOK{});
    ASSERT_EQ(isize(links), isize(prefab.links));
    Real3 const localPosition{0.1_r, 0.05_r, 0_r};
    Quaternion const localRotation = Quaternion::RotationX(0.2_r);
    DynamicArray<real> directPose(actor->GetNumDofs());
    DynamicArray<TransformRT> directLinks(prefab.links.size());
    int const baseDofs = actor->GetArticulatedShapeInfo(ExpectOK{}).dofInfo[0].GetSize();
    for (real const angle : {0.35_r, -0.2_r, 0.1_r}) {
      auto const goalLinks = ExpectedLinks(prefab.worldFromRoot, angle, 0.25_r);
      Real3 const worldPosition = goalLinks[kTip].TransformPoint(localPosition);
      Quaternion const worldRotation = goalLinks[kTip].GetRotation() * localRotation;
      ik->SetPositionGoal(kTip, localPosition, worldPosition, kGoalStiffness, ExpectOK{});
      ik->SetRotationGoal(kTip, localRotation, worldRotation, kGoalStiffness, ExpectOK{});
      ASSERT_NE(
          solver->CreatePositionTarget(
              links[kTip], localPosition, worldPosition, kGoalStiffness, ExpectOK{}),
          nullptr);
      ASSERT_NE(
          solver->CreateRotationTarget(
              links[kTip],
              localRotation.ToRotationVector(),
              worldRotation.ToRotationVector(),
              kGoalStiffness,
              ExpectOK{}),
          nullptr);
      EXPECT_EQ(ik->SolveIK(ExpectOK{}), solver->SolveIK(ExpectOK{}));
      actor->GetArticulatedPose(directPose, ExpectOK{});
      actor->GetArticulatedLinkTransforms(directLinks, ExpectOK{});
      ExpectPoseNear(
          ik->GetJointPositions(),
          Span<real const>{directPose.data() + baseDofs, directPose.size() - baseDofs});
      ExpectLinksNear(ik->GetWorldFromLinks(), directLinks);
    }
  }

  Context* _mochiContext = nullptr;
};

TEST_F(BotIKTest, Create_FreeHardAndForcedFixed_InitialSnapshotsExcludeRootDofs) {
  for (auto const rootType : {ArticulatedJointType::Free, ArticulatedJointType::Hard}) {
    for (bool const fixedBase : {false, true}) {
      SCOPED_TRACE(fixedBase);
      BotPrefab const prefab = MakeRobot(rootType);
      auto params = TestCreateParams();
      params.fixedBase = fixedBase;
      auto const ik = CreateBotIK(_mochiContext, prefab, params, ExpectOK{});
      ASSERT_NE(ik, nullptr);
      EXPECT_EQ(ik->GetNumLinks(), 5);
      EXPECT_EQ(ik->GetNumDofs(), 2);
      ExpectPoseNear(ik->GetJointPositions(), prefab.defaultPose);
      ExpectLinksNear(ik->GetWorldFromLinks(), ExpectedLinks(prefab.worldFromRoot, 0.15_r, 0.1_r));
      EXPECT_TRUE(ik->IsLinkRigidlyAttachedToRootLink(0));
      EXPECT_TRUE(ik->IsLinkRigidlyAttachedToRootLink(1));
      EXPECT_FALSE(ik->IsLinkRigidlyAttachedToRootLink(kArm));
      EXPECT_FALSE(ik->IsLinkRigidlyAttachedToRootLink(kTip));
      EXPECT_FALSE(ik->IsLinkRigidlyAttachedToRootLink(-1));
      EXPECT_FALSE(ik->IsLinkRigidlyAttachedToRootLink(5));
    }
  }
}

TEST_F(BotIKTest, Create_ForcedFixedBase_PreventsRootMotionWithoutChangingPrefab) {
  BotPrefab const prefab = MakeRobot(ArticulatedJointType::Free);
  auto params = TestCreateParams();
  params.fixedBase = true;
  auto const ik = CreateBotIK(_mochiContext, prefab, params, ExpectOK{});
  ASSERT_NE(ik, nullptr);
  ik->SetPositionGoal(
      0,
      {},
      prefab.worldFromRoot.GetTranslation() + Real3{0.3_r, 0.2_r, 0.1_r},
      kGoalStiffness,
      ExpectOK{});
  ik->SetRotationGoal(0, {}, Quaternion::RotationX(0.4_r), kGoalStiffness, ExpectOK{});

  EXPECT_FALSE(ik->SolveIK(ExpectOK{}));

  EXPECT_TRUE(NearEqual(ik->GetWorldFromLinks()[0], prefab.worldFromRoot, kPoseTolerance));
  EXPECT_EQ(prefab.joints[0].type, ArticulatedJointType::Free);
  EXPECT_EQ(ik->GetNumDofs(), 2);
}

TEST_F(BotIKTest, Create_PrivateOptionsAndComponentOmission_DoNotMutatePreparedPrefab) {
  BotPrefab prefab = MakeRobot(ArticulatedJointType::Free);
  AddShapesAndContactOverrides(prefab);
  BotSensorPrefab sensor;
  sensor.type = "SENSOR_CAMERA";
  sensor.name = "camera_must_not_initialize";
  sensor.params = "{invalid json";
  prefab.links[kTip].sensors.push_back(sensor);
  BotActuatorPrefab actuator;
  actuator.type = "ACTUATOR_MUST_NOT_INITIALIZE";
  actuator.name = "hardware";
  prefab.links[kTip].actuators.push_back(actuator);
  prefab.skin.emplace();
  prefab.skin->shapeFile = "skin_must_not_load";
  BotPrefab const original = prefab;
  TestBotLoader loader;
  auto params = TestCreateParams();
  params.fixedBase = true;

  auto const ik = CreateBotIK(_mochiContext, prefab, params, loader, ExpectOK{});

  ASSERT_NE(ik, nullptr);
  EXPECT_EQ(loader.shapeLoads, 0);
  EXPECT_EQ(prefab.links, original.links);
  EXPECT_EQ(prefab.joints, original.joints);
  EXPECT_EQ(prefab.defaultPose, original.defaultPose);
  EXPECT_EQ(prefab.worldFromRoot, original.worldFromRoot);
  EXPECT_EQ(prefab.skin, original.skin);
  EXPECT_EQ(prefab._dofIndices, original._dofIndices);
  EXPECT_EQ(prefab._numDofs, original._numDofs);
  ASSERT_EQ(prefab.contactOverrides.size(), original.contactOverrides.size());
  for (int i = 0; i < isize(prefab.contactOverrides); ++i) {
    EXPECT_EQ(prefab.contactOverrides[i].linkA, original.contactOverrides[i].linkA);
    EXPECT_EQ(prefab.contactOverrides[i].linkB, original.contactOverrides[i].linkB);
    EXPECT_EQ(prefab.contactOverrides[i].enable, original.contactOverrides[i].enable);
  }
  // Renaming after preparation makes lexical order differ without invalidating the cached topology.
  prefab.links[1].name = "z_weld";
  prefab.links[kArm].name = "a_arm";
  auto const orderedIK = CreateBotIK(_mochiContext, prefab, params, loader, ExpectOK{});
  ASSERT_NE(orderedIK, nullptr);
  EXPECT_TRUE(orderedIK->IsLinkRigidlyAttachedToRootLink(1));
  EXPECT_FALSE(orderedIK->IsLinkRigidlyAttachedToRootLink(kArm));
  ExpectLinksNear(
      orderedIK->GetWorldFromLinks(), ExpectedLinks(prefab.worldFromRoot, 0.15_r, 0.1_r));
}

TEST_F(BotIKTest, Create_NullContextAndInconsistentPreparedData_ReturnNoInstance) {
  auto const suppressWarning = SuppressLogWarning();
  BotPrefab const valid = MakeRobot();
  EXPECT_EQ(CreateBotIK(nullptr, valid, {}, ExpectNotOK{}), nullptr);
  EXPECT_EQ(CreateBotIK(_mochiContext, BotPrefab{}, {}, ExpectNotOK{}), nullptr);
  for (int const fault : {0, 1, 2, 3}) {
    SCOPED_TRACE(fault);
    BotPrefab prefab = valid;
    if (fault == 0) {
      prefab._numDofs = 99;
    } else if (fault == 1) {
      prefab._dofIndices[0] = kTip;
    } else if (fault == 2) {
      prefab.defaultPose.pop_back();
    } else {
      prefab.links[kTip]._index = 99;
    }
    EXPECT_EQ(CreateBotIK(_mochiContext, prefab, {}, ExpectNotOK{}), nullptr);
  }
}

TEST_F(BotIKTest, Create_CustomLoaderFailure_ReleasesPartiallyLoadedShapes) {
  BotPrefab prefab = MakeRobot();
  AddShapesAndContactOverrides(prefab);
  TestBotLoader loader;
  loader.failOnShapeLoad = 2;
  auto params = TestCreateParams();
  params.excludeCollisionShapes = false;
  int const initialShapes = _mochiContext->GetNumShapes();

  EXPECT_EQ(CreateBotIK(_mochiContext, prefab, params, loader, ExpectNotOK{}), nullptr);

  EXPECT_EQ(loader.shapeLoads, 2);
  EXPECT_EQ(_mochiContext->GetNumShapes(), initialShapes);
  loader.failOnShapeLoad = -1;
  auto ik = CreateBotIK(_mochiContext, prefab, params, loader, ExpectOK{});
  ASSERT_NE(ik, nullptr);
  ik.reset();
  EXPECT_EQ(_mochiContext->GetNumShapes(), initialShapes);
}

TEST_F(BotIKTest, Create_InvalidTransmissionAfterActorCreation_ReleasesResources) {
  BotPrefab prefab = MakeRobot();
  AddShapesAndContactOverrides(prefab);
  auto transmission = MakeTransmission();
  transmission.jointCoefficients.pop_back();
  prefab.linearTransmissions.push_back(transmission);
  TestBotLoader loader;
  auto params = TestCreateParams();
  params.excludeCollisionShapes = false;
  int const initialShapes = _mochiContext->GetNumShapes();

  EXPECT_EQ(CreateBotIK(_mochiContext, prefab, params, loader, ExpectNotOK{}), nullptr);

  EXPECT_EQ(loader.shapeLoads, 5);
  EXPECT_EQ(_mochiContext->GetNumShapes(), initialShapes);
  prefab.linearTransmissions.clear();
  auto const ik = CreateBotIK(_mochiContext, prefab, params, loader, ExpectOK{});
  ASSERT_NE(ik, nullptr);
  ExpectPoseNear(ik->GetJointPositions(), prefab.defaultPose);
}

TEST_F(BotIKTest, SetSolverParams_CreationAndUpdate_RoundTripAllFields) {
  auto params = TestCreateParams();
  params.solverParams.maxIter = 37;
  params.solverParams.lineSearchMaxIter = 13;
  params.solverParams.maxElapsedTimeSeconds = 0.5;
  auto const ik = CreateBotIK(_mochiContext, MakeRobot(), params, ExpectOK{});
  ASSERT_NE(ik, nullptr);
  for (int i = 0; i < 2; ++i) {
    auto const actual = ik->GetSolverParams();
    auto const& expected = params.solverParams;
    EXPECT_EQ(actual.maxIter, expected.maxIter);
    EXPECT_EQ(actual.verbosity, expected.verbosity);
    EXPECT_EQ(actual.absTol, expected.absTol);
    EXPECT_EQ(actual.relTol, expected.relTol);
    EXPECT_EQ(actual.positionErrorThres, expected.positionErrorThres);
    EXPECT_EQ(actual.rotationErrorThres, expected.rotationErrorThres);
    EXPECT_EQ(actual.lineSearchMaxIter, expected.lineSearchMaxIter);
    EXPECT_EQ(actual.maxElapsedTimeSeconds, expected.maxElapsedTimeSeconds);
    params.solverParams.maxIter = 43;
    params.solverParams.absTol = 0.003_r;
    params.solverParams.relTol = 0.0001_r;
    params.solverParams.positionErrorThres = 0.004_r;
    params.solverParams.rotationErrorThres = 0.005_r;
    params.solverParams.lineSearchMaxIter = 17;
    params.solverParams.maxElapsedTimeSeconds = 0.75;
    ik->SetSolverParams(params.solverParams);
  }
}

TEST_F(BotIKTest, SetJointPositions_FreeAndHardRoots_RefreshSnapshotsWithoutSolving) {
  DynamicArray<real> const seed{-0.3_r, 0.25_r};
  for (auto const rootType : {ArticulatedJointType::Free, ArticulatedJointType::Hard}) {
    BotPrefab const prefab = MakeRobot(rootType);
    auto const ik = CreateBotIK(_mochiContext, prefab, TestCreateParams(), ExpectOK{});
    ASSERT_NE(ik, nullptr);
    ik->SetJointPositions(seed, ExpectOK{});
    ExpectPoseNear(ik->GetJointPositions(), seed);
    ExpectLinksNear(ik->GetWorldFromLinks(), ExpectedLinks(prefab.worldFromRoot, seed[0], seed[1]));
    DynamicArray<real> const previousPose{ik->GetJointPositions()};
    DynamicArray<TransformRT> const previousLinks{ik->GetWorldFromLinks()};
    ik->SetJointPositions(DynamicArray<real>{0_r}, ExpectNotOK{});
    ik->SetJointPositions(DynamicArray<real>{0_r, 0_r, 0_r}, ExpectNotOK{});
    ik->SetJointPositions(
        DynamicArray<real>{std::numeric_limits<real>::quiet_NaN(), 0_r}, ExpectNotOK{});
    ik->SetJointPositions(
        DynamicArray<real>{0_r, std::numeric_limits<real>::infinity()}, ExpectNotOK{});
    EXPECT_SPAN_EQ(ik->GetJointPositions(), MakeConstSpan(previousPose));
    EXPECT_SPAN_EQ(ik->GetWorldFromLinks(), MakeConstSpan(previousLinks));
  }
}

TEST_F(BotIKTest, SetJointPositions_AfterFloatingBaseSolve_PreservesSolvedBasePlacement) {
  BotPrefab const prefab = MakeRobot(ArticulatedJointType::Free);
  auto const ik = CreateBotIK(_mochiContext, prefab, TestCreateParams(), ExpectOK{});
  ASSERT_NE(ik, nullptr);
  TransformRT const desiredBase{Quaternion::RotationX(0.3_r), Real3{-0.5_r, 1_r, 0.5_r}};
  ik->SetPositionGoal(0, {}, desiredBase.GetTranslation(), kGoalStiffness, ExpectOK{});
  ik->SetRotationGoal(0, {}, desiredBase.GetRotation(), kGoalStiffness, ExpectOK{});
  ASSERT_TRUE(ik->SolveIK(ExpectOK{}));
  TransformRT const solvedBase = ik->GetWorldFromLinks()[0];
  EXPECT_TRUE(NearEqual(solvedBase, desiredBase, kSolveTolerance));
  ASSERT_FALSE(NearEqual(solvedBase, prefab.worldFromRoot, kSolveTolerance));
  DynamicArray<real> const seed{-0.2_r, 0.3_r};

  ik->SetJointPositions(seed, ExpectOK{});

  EXPECT_TRUE(NearEqual(ik->GetWorldFromLinks()[0], solvedBase, kPoseTolerance));
  ExpectPoseNear(ik->GetJointPositions(), seed);
  ExpectLinksNear(ik->GetWorldFromLinks(), ExpectedLinks(solvedBase, seed[0], seed[1]));
  EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
}

TEST_F(BotIKTest, SolveIK_PositionRotationAndBoth_ReachIndependentlyAuthoredGoals) {
  BotPrefab const prefab = MakeRobot();
  auto const goalLinks = ExpectedLinks(prefab.worldFromRoot, 0.4_r, 0.3_r);
  Real3 const localPosition{0.1_r, 0.05_r, 0_r};
  Quaternion const localRotation = Quaternion::RotationX(0.2_r);
  for (int const mode : {0, 1, 2}) {
    SCOPED_TRACE(mode);
    auto const ik = CreateBotIK(_mochiContext, prefab, TestCreateParams(), ExpectOK{});
    ASSERT_NE(ik, nullptr);
    if (mode != 1) {
      ik->SetPositionGoal(
          kTip,
          localPosition,
          goalLinks[kTip].TransformPoint(localPosition),
          kGoalStiffness,
          ExpectOK{});
    }
    if (mode != 0) {
      ik->SetRotationGoal(
          kTip,
          localRotation,
          goalLinks[kTip].GetRotation() * localRotation,
          kGoalStiffness,
          ExpectOK{});
    }
    EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
    TransformRT const solvedTip = ik->GetWorldFromLinks()[kTip];
    if (mode != 1) {
      EXPECT_LT(
          Norm(
              solvedTip.TransformPoint(localPosition) -
              goalLinks[kTip].TransformPoint(localPosition)),
          2_r * kSolveTolerance);
    }
    if (mode != 0) {
      EXPECT_TRUE(NearEqual(
          solvedTip.GetRotation() * localRotation,
          goalLinks[kTip].GetRotation() * localRotation,
          kSolveTolerance));
    }
    if (mode == 2) {
      ExpectPoseNear(ik->GetJointPositions(), DynamicArray<real>{0.4_r, 0.3_r});
    }
    EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
  }
}

TEST_F(BotIKTest, SetGoals_SameAndChangedLocalPivots_ReplaceWorldGoalsAndStiffness) {
  BotPrefab const prefab = MakeRobot();
  auto const ik = CreateBotIK(_mochiContext, prefab, TestCreateParams(), ExpectOK{});
  ASSERT_NE(ik, nullptr);
  Real3 localPosition{0.1_r, 0.05_r, 0_r};
  Quaternion localRotation = Quaternion::RotationX(0.2_r);
  for (int update = 0; update < 4; ++update) {
    SCOPED_TRACE(update);
    if (update == 2) {
      localPosition = {0.2_r, -0.1_r, 0.05_r};
      localRotation = Quaternion::RotationY(-0.3_r);
    }
    DynamicArray<real> const previousPose{ik->GetJointPositions()};
    DynamicArray<TransformRT> const previousLinks{ik->GetWorldFromLinks()};
    real const angle = 0.1_r * (update + 1);
    real const extension = 0.05_r * (update + 1);
    auto const goal = ExpectedLinks(prefab.worldFromRoot, angle, extension)[kTip];
    ik->SetPositionGoal(
        kTip,
        localPosition,
        goal.TransformPoint(localPosition),
        kGoalStiffness * (update + 1),
        ExpectOK{});
    ik->SetRotationGoal(
        kTip,
        localRotation,
        goal.GetRotation() * localRotation,
        kGoalStiffness * (update + 1),
        ExpectOK{});
    EXPECT_SPAN_EQ(ik->GetJointPositions(), MakeConstSpan(previousPose));
    EXPECT_SPAN_EQ(ik->GetWorldFromLinks(), MakeConstSpan(previousLinks));
    EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
    ExpectPoseNear(ik->GetJointPositions(), DynamicArray<real>{angle, extension});
  }
}

TEST_F(BotIKTest, SetGoals_StiffnessUpdates_ChangeCompromiseBetweenConflictingGoals) {
  BotPrefab const prefab = MakeRobot();
  auto const ik = CreateBotIK(_mochiContext, prefab, TestCreateParams(), ExpectOK{});
  ASSERT_NE(ik, nullptr);
  auto const lowGoal = ExpectedLinks(prefab.worldFromRoot, 0_r, 0.1_r)[kArm];
  auto const highGoal = ExpectedLinks(prefab.worldFromRoot, 0.6_r, 0.1_r)[kTip];
  ik->SetRotationGoal(kArm, {}, lowGoal.GetRotation(), kGoalStiffness, ExpectOK{});
  ik->SetRotationGoal(kTip, {}, highGoal.GetRotation(), kGoalStiffness, ExpectOK{});
  EXPECT_FALSE(ik->SolveIK(ExpectOK{}));
  real const equalWeightAngle = ik->GetJointPositions()[0];
  EXPECT_GT(equalWeightAngle, 0.2_r);
  EXPECT_LT(equalWeightAngle, 0.4_r);

  ik->SetRotationGoal(kTip, {}, highGoal.GetRotation(), 4_r * kGoalStiffness, ExpectOK{});
  EXPECT_FALSE(ik->SolveIK(ExpectOK{}));
  EXPECT_GT(ik->GetJointPositions()[0], equalWeightAngle + 0.1_r);

  ik->ClearAllGoals(ExpectOK{});
  Real3 const pivot{1_r, 0_r, 0_r};
  Real3 const lowPosition = lowGoal.TransformPoint(pivot);
  Real3 const highPosition = highGoal.TransformPoint(pivot);
  ik->SetPositionGoal(kArm, pivot, lowPosition, kGoalStiffness, ExpectOK{});
  ik->SetPositionGoal(kTip, pivot, highPosition, kGoalStiffness, ExpectOK{});
  EXPECT_FALSE(ik->SolveIK(ExpectOK{}));
  real const equalWeightDistance =
      Norm(ik->GetWorldFromLinks()[kArm].TransformPoint(pivot) - lowPosition);

  ik->SetPositionGoal(kArm, pivot, lowPosition, 16_r * kGoalStiffness, ExpectOK{});
  EXPECT_FALSE(ik->SolveIK(ExpectOK{}));
  EXPECT_LT(
      Norm(ik->GetWorldFromLinks()[kArm].TransformPoint(pivot) - lowPosition),
      0.5_r * equalWeightDistance);
}

TEST_F(BotIKTest, SetPositionGoal_InvalidUpdates_PreservePreviousGoal) {
  BotPrefab const prefab = MakeRobot();
  auto const ik = CreateBotIK(_mochiContext, prefab, TestCreateParams(), ExpectOK{});
  ASSERT_NE(ik, nullptr);
  Real3 const localPosition{0.1_r, 0.05_r, 0_r};
  Real3 const goal =
      ExpectedLinks(prefab.worldFromRoot, 0.35_r, 0.2_r)[kTip].TransformPoint(localPosition);
  ik->SetPositionGoal(kTip, localPosition, goal, kGoalStiffness, ExpectOK{});
  for (real const bad :
       {-1_r, std::numeric_limits<real>::infinity(), std::numeric_limits<real>::quiet_NaN()}) {
    SCOPED_TRACE(bad);
    ik->SetPositionGoal(kTip, {0.2_r, 0_r, 0_r}, {}, bad, ExpectNotOK{});
  }
  for (real const bad :
       {std::numeric_limits<real>::infinity(), std::numeric_limits<real>::quiet_NaN()}) {
    ik->SetPositionGoal(kTip, {bad, 0_r, 0_r}, {}, kGoalStiffness, ExpectNotOK{});
    ik->SetPositionGoal(kTip, localPosition, {0_r, bad, 0_r}, kGoalStiffness, ExpectNotOK{});
  }
  for (int const index : {-1, 5, std::numeric_limits<int>::max()}) {
    ik->SetPositionGoal(index, {}, {}, kGoalStiffness, ExpectNotOK{});
    ik->ClearPositionGoal(index, ExpectNotOK{});
  }
  EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
  EXPECT_LT(
      Norm(ik->GetWorldFromLinks()[kTip].TransformPoint(localPosition) - goal),
      2_r * kSolveTolerance);
}

TEST_F(BotIKTest, SetRotationGoal_InvalidUpdates_PreservePreviousGoal) {
  BotPrefab const prefab = MakeRobot();
  auto const ik = CreateBotIK(_mochiContext, prefab, TestCreateParams(), ExpectOK{});
  ASSERT_NE(ik, nullptr);
  Quaternion const localRotation = Quaternion::RotationX(0.2_r);
  Quaternion const goal =
      ExpectedLinks(prefab.worldFromRoot, 0.35_r, 0.1_r)[kTip].GetRotation() * localRotation;
  ik->SetRotationGoal(kTip, localRotation, goal, kGoalStiffness, ExpectOK{});
  DynamicArray<Quaternion> const invalid = {
      Quaternion::Zero(),
      Quaternion{std::numeric_limits<real>::infinity(), 0_r, 0_r, 1_r},
      Quaternion{0_r, 0_r, 0_r, std::numeric_limits<real>::quiet_NaN()}};
  for (auto const& bad : invalid) {
    ik->SetRotationGoal(kTip, bad, {}, kGoalStiffness, ExpectNotOK{});
    ik->SetRotationGoal(kTip, localRotation, bad, kGoalStiffness, ExpectNotOK{});
  }
  for (real const bad :
       {-1_r, std::numeric_limits<real>::infinity(), std::numeric_limits<real>::quiet_NaN()}) {
    ik->SetRotationGoal(kTip, Quaternion::RotationY(0.3_r), {}, bad, ExpectNotOK{});
  }
  for (int const index : {-1, 5, std::numeric_limits<int>::max()}) {
    ik->SetRotationGoal(index, {}, {}, kGoalStiffness, ExpectNotOK{});
    ik->ClearRotationGoal(index, ExpectNotOK{});
  }
  EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
  EXPECT_TRUE(NearEqual(
      ik->GetWorldFromLinks()[kTip].GetRotation() * localRotation, goal, kSolveTolerance));
}

TEST_F(BotIKTest, ClearGoals_IndependentIdempotentRemoval_NoGoalSolvePreservesSnapshot) {
  BotPrefab const prefab = MakeRobot();
  auto const ik = CreateBotIK(_mochiContext, prefab, TestCreateParams(), ExpectOK{});
  ASSERT_NE(ik, nullptr);
  auto const goal = ExpectedLinks(prefab.worldFromRoot, 0.4_r, 0.25_r)[kTip];
  ik->SetPositionGoal(kTip, {}, goal.GetTranslation(), kGoalStiffness, ExpectOK{});
  ik->SetRotationGoal(kTip, {}, goal.GetRotation(), kGoalStiffness, ExpectOK{});
  ik->ClearPositionGoal(kTip, ExpectOK{});
  ik->ClearPositionGoal(kTip, ExpectOK{});
  EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
  EXPECT_NEAR(ik->GetJointPositions()[0], 0.4_r, kSolveTolerance);
  ik->SetPositionGoal(kTip, {}, goal.GetTranslation(), kGoalStiffness, ExpectOK{});
  ik->ClearRotationGoal(kTip, ExpectOK{});
  ik->ClearRotationGoal(kTip, ExpectOK{});
  EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
  EXPECT_LT(
      Norm(ik->GetWorldFromLinks()[kTip].GetTranslation() - goal.GetTranslation()),
      2_r * kSolveTolerance);
  ik->SetRotationGoal(kArm, {}, goal.GetRotation(), kGoalStiffness, ExpectOK{});
  DynamicArray<real> const previousPose{ik->GetJointPositions()};
  DynamicArray<TransformRT> const previousLinks{ik->GetWorldFromLinks()};

  ik->ClearAllGoals(ExpectOK{});
  ik->ClearAllGoals(ExpectOK{});
  EXPECT_FALSE(ik->SolveIK(ExpectNotOK{}));

  EXPECT_SPAN_EQ(ik->GetJointPositions(), MakeConstSpan(previousPose));
  EXPECT_SPAN_EQ(ik->GetWorldFromLinks(), MakeConstSpan(previousLinks));
  ik->SetRotationGoal(kTip, {}, goal.GetRotation(), kGoalStiffness, ExpectOK{});
  EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
}

TEST_F(BotIKTest, SolveIK_NoGoalsBeforeFirstSolve_PreservesInitialSnapshot) {
  auto const ik = CreateBotIK(_mochiContext, MakeRobot(), TestCreateParams(), ExpectOK{});
  ASSERT_NE(ik, nullptr);
  DynamicArray<real> const pose{ik->GetJointPositions()};
  DynamicArray<TransformRT> const links{ik->GetWorldFromLinks()};
  EXPECT_FALSE(ik->SolveIK(ExpectNotOK{}));
  EXPECT_SPAN_EQ(ik->GetJointPositions(), MakeConstSpan(pose));
  EXPECT_SPAN_EQ(ik->GetWorldFromLinks(), MakeConstSpan(links));
}

TEST_F(BotIKTest, SetGoals_ZeroStiffness_RemainPresentUntilExplicitClear) {
  auto const ik = CreateBotIK(_mochiContext, MakeRobot(), TestCreateParams(), ExpectOK{});
  ASSERT_NE(ik, nullptr);
  DynamicArray<real> const initialPose{ik->GetJointPositions()};
  ik->SetPositionGoal(kTip, {}, {9_r, 8_r, 7_r}, 0_r, ExpectOK{});
  (void)ik->SolveIK(ExpectOK{});
  ExpectPoseNear(ik->GetJointPositions(), initialPose);
  ik->SetRotationGoal(kTip, {}, Quaternion::RotationY(0.6_r), 0_r, ExpectOK{});
  ik->ClearPositionGoal(kTip, ExpectOK{});
  (void)ik->SolveIK(ExpectOK{});
  ExpectPoseNear(ik->GetJointPositions(), initialPose);
  ik->ClearRotationGoal(kTip, ExpectOK{});
  EXPECT_FALSE(ik->SolveIK(ExpectNotOK{}));
}

TEST_F(BotIKTest, SolveIK_UnreachablePositionAndRotation_RefreshReadableFiniteSnapshots) {
  BotPrefab const prefab = MakeRobot();
  auto const ik = CreateBotIK(_mochiContext, prefab, TestCreateParams(), ExpectOK{});
  ASSERT_NE(ik, nullptr);
  auto const goal = ExpectedLinks(prefab.worldFromRoot, 0.3_r, 0.2_r)[kTip];
  // The mechanism stays in the root's XY plane; this position is unreachable along root Z.
  Real3 const unreachable =
      goal.GetTranslation() + prefab.worldFromRoot.TransformDirection(Real3{0_r, 0_r, 0.5_r});
  ik->SetPositionGoal(kTip, {}, unreachable, kGoalStiffness, ExpectOK{});
  ik->SetRotationGoal(kTip, {}, goal.GetRotation(), kGoalStiffness, ExpectOK{});
  EXPECT_FALSE(ik->SolveIK(ExpectOK{}));
  ASSERT_EQ(isize(ik->GetJointPositions()), 2);
  ASSERT_EQ(isize(ik->GetWorldFromLinks()), 5);
  EXPECT_TRUE(IsFinite(ik->GetJointPositions()[0]));
  EXPECT_TRUE(IsFinite(ik->GetJointPositions()[1]));
  EXPECT_GT(Abs(ik->GetJointPositions()[0] - prefab.defaultPose[0]), 0.05_r);
  for (auto const& link : ik->GetWorldFromLinks()) {
    EXPECT_TRUE(IsFinite(link));
  }
  EXPECT_GT(Norm(ik->GetWorldFromLinks()[kTip].GetTranslation() - unreachable), 0.4_r);
  ik->ClearPositionGoal(kTip, ExpectOK{});
  ik->SetRotationGoal(
      kTip, {}, goal.GetRotation() * Quaternion::RotationX(0.5_r), kGoalStiffness, ExpectOK{});
  EXPECT_FALSE(ik->SolveIK(ExpectOK{}));
  EXPECT_TRUE(IsFinite(ik->GetWorldFromLinks()[kTip]));
  ExpectLinksNear(
      ik->GetWorldFromLinks(),
      ExpectedLinks(prefab.worldFromRoot, ik->GetJointPositions()[0], ik->GetJointPositions()[1]));
}

TEST_F(BotIKTest, Destroy_WithActiveGoals_ReleasesShapesAndLeavesBorrowedContextUsable) {
  Scene* const otherScene = _mochiContext->CreateScene("CallerOwnedScene");
  ASSERT_NE(otherScene, nullptr);
  MOCHI_DEFER(_mochiContext->DestroyScene(otherScene));
  RigidActorParams otherParams;
  otherParams.worldFromLocal = TransformRT{Real3{7_r, 8_r, 9_r}};
  otherParams.isStatic = true;
  otherParams.shape = _mochiContext->CreateSphereShape({}, 0.1_r, ExpectOK{});
  Actor* const otherActor = otherScene->CreateRigidActor(otherParams, ExpectOK{});
  ASSERT_NE(otherActor, nullptr);
  ActorHandle const otherHandle = otherActor->GetHandle();
  int const initialShapes = _mochiContext->GetNumShapes();
  BotPrefab prefab = MakeRobot();
  AddShapesAndContactOverrides(prefab);
  TestBotLoader loader;
  auto params = TestCreateParams();
  params.excludeCollisionShapes = false;
  for (int repetition = 0; repetition < 3; ++repetition) {
    auto ik = CreateBotIK(_mochiContext, prefab, params, loader, ExpectOK{});
    ASSERT_NE(ik, nullptr);
    ik->SetPositionGoal(
        kTip, {}, ik->GetWorldFromLinks()[kTip].GetTranslation(), kGoalStiffness, ExpectOK{});
    ik->SetRotationGoal(
        kTip, {}, ik->GetWorldFromLinks()[kTip].GetRotation(), kGoalStiffness, ExpectOK{});
    EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
    ik.reset();
    BotIK::Destroy(nullptr);
    EXPECT_EQ(_mochiContext->GetNumShapes(), initialShapes);
    ASSERT_TRUE(_mochiContext->IsValidScene(otherScene));
    ASSERT_EQ(otherScene->GetActor(otherHandle), otherActor);
    EXPECT_NEAR_EQ(otherActor->GetRootTransform(), otherParams.worldFromLocal);
  }
}

TEST_F(
    BotIKTest,
    SolveIK_CustomLayersAndEnabledContactOverrides_ShapeRetentionMatchesCollisionFreeOracle) {
  BotPrefab prefab = MakeRobot();
  AddShapesAndContactOverrides(prefab);
  ExpectDirectParity(prefab, false);
  ExpectDirectParity(prefab, true);
}

TEST_F(BotIKTest, SolveIK_AuthoredJointLimits_MatchesDirectExperimentalSolver) {
  BotPrefab prefab = MakeRobot();
  prefab.joints[kArm].minLimit = Real3{0_r, 0_r, -0.15_r};
  prefab.joints[kArm].maxLimit = Real3{0_r, 0_r, 0.2_r};
  prefab.joints[kArm].limitStiffness = 5000_r;
  prefab.joints[kArm].limitDamping = 2_r;
  prefab.joints[kSlide].minLimit = Real3{-0.1_r, 0_r, 0_r};
  prefab.joints[kSlide].maxLimit = Real3{0.2_r, 0_r, 0_r};
  prefab.joints[kSlide].limitStiffness = 5000_r;
  RebuildBotData(prefab, ExpectOK{});
  ExpectDirectParity(prefab, true);
}

TEST_F(BotIKTest, SolveIK_AuthoredRatioOffsetAndCompressiveTransmission_MatchesDirectSolver) {
  BotPrefab prefab = MakeRobot();
  prefab.linearTransmissions.push_back(MakeTransmission());
  ExpectDirectParity(prefab, true);
  prefab.linearTransmissions[0].allowCompressiveForce = false;
  ExpectDirectParity(prefab, true);
}

TEST_F(BotIKTest, SolveIK_SpatialTendon_MatchesDirectExperimentalSolver) {
  BotPrefab prefab = MakeRobot();
  BotSpatialTendonPrefab tendon;
  tendon.name = "root_to_tip";
  RoutingElement rootWaypoint;
  rootWaypoint.index = 0;
  rootWaypoint.localPosition = {0_r, 0.4_r, 0_r};
  RoutingElement tipWaypoint;
  tipWaypoint.index = kTip;
  tipWaypoint.localPosition = {0.1_r, 0_r, 0_r};
  tendon.routingElements = {rootWaypoint, tipWaypoint};
  tendon.targetDisplacement = 0.05_r;
  tendon.stiffness = 2000_r;
  tendon.allowCompressiveForce = true;
  prefab.spatialTendons.push_back(tendon);
  ExpectDirectParity(prefab, true);
}

TEST_F(BotIKTest, SolveIK_CycleClosingJoint_MatchesDirectExperimentalSolver) {
  BotPrefab prefab = MakeRobot();
  ArticulatedCycleJointParams cycle;
  cycle.parentLink = 1;
  cycle.childLink = kTip;
  cycle.jointFromChildLink = TransformRT{Real3{0.7_r, -0.2_r, 0_r}};
  cycle.stiffness = 2000_r;
  prefab.cycles.push_back(cycle);
  ExpectDirectParity(prefab, true);
}

TEST_F(BotIKTest, InputBotConfiguration_RetargetsOffsetsAndPreservesGoalsOnInvalidInput) {
  auto const bot = MakeRobot();
  auto const ik = CreateBotIK(_mochiContext, bot, TestCreateParams(), ExpectOK{});
  ASSERT_NE(ik, nullptr);
  auto inputBot = MakeRobot(ArticulatedJointType::Free);
  inputBot.links[kTip].name = "reference_tip";
  RebuildBotData(inputBot, ExpectOK{});
  BotIKGoalConfig goalConfig;
  goalConfig.name = "tip";
  goalConfig.outputBotLink = "d_tip";
  goalConfig.inputBotLink = "reference_tip";
  goalConfig.linkFrameFromConstraintFrame =
      TransformRT{Quaternion::RotationX(0.2_r), Real3{0.1_r, 0.05_r, 0_r}};
  goalConfig.baseOrInputLinkFrameFromConstraintFrame = goalConfig.linkFrameFromConstraintFrame;
  goalConfig.positionStiffness = kGoalStiffness;
  goalConfig.rotationStiffness = kGoalStiffness;
  DynamicArray<BotIKGoalConfig> goals{goalConfig};
  for (int i = 0; i < 2; ++i) {
    DynamicArray<real> const pose{0.25_r + 0.1_r * i, 0.2_r + 0.05_r * i};
    ik->SetInputBotConfiguration(inputBot, pose, bot.worldFromRoot, goals, ExpectOK{});
    EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
    ExpectPoseNear(ik->GetJointPositions(), pose);
    auto invalid = goals;
    invalid[0].inputBotLink = "missing";
    ik->SetInputBotConfiguration(inputBot, pose, bot.worldFromRoot, invalid, ExpectNotOK{});
    ik->SetInputBotConfiguration(
        inputBot, DynamicArray<real>{0_r}, bot.worldFromRoot, goals, ExpectNotOK{});
    invalid[0].inputBotLink = "reference_tip";
    invalid[0].outputBotLink = "missing";
    ik->SetInputBotConfiguration(inputBot, pose, bot.worldFromRoot, invalid, ExpectNotOK{});
    ik->SetJointPositions(bot.defaultPose, ExpectOK{});
    EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
    ExpectPoseNear(ik->GetJointPositions(), pose);
  }
  goals[0].enabled = false;
  ik->SetInputBotConfiguration(inputBot, {}, bot.worldFromRoot, goals, ExpectOK{});
  EXPECT_FALSE(ik->SolveIK(ExpectNotOK{}));
}

TEST_F(BotIKTest, InputBotScale_ReachableGoalsMatchIndependentGoalsWithoutMutatingAssets) {
  auto const bot = MakeRobot();
  auto const originalBot = bot;
  auto inputBot = MakeRobot(ArticulatedJointType::Free);
  inputBot.scale = 1.5_r;
  auto const originalInputBot = inputBot;
  auto const ik = CreateBotIK(_mochiContext, bot, TestCreateParams(), ExpectOK{});
  auto const direct = CreateBotIK(_mochiContext, bot, TestCreateParams(), ExpectOK{});
  ASSERT_NE(ik, nullptr);
  ASSERT_NE(direct, nullptr);
  BotIKGoalConfig goalConfig;
  goalConfig.name = "tip";
  goalConfig.outputBotLink = "d_tip";
  goalConfig.inputBotLink = "d_tip";
  goalConfig.linkFrameFromConstraintFrame =
      TransformRT{Quaternion::RotationX(0.2_r), Real3{0.1_r, 0_r, 0_r}};
  goalConfig.baseOrInputLinkFrameFromConstraintFrame = goalConfig.linkFrameFromConstraintFrame;
  goalConfig.positionStiffness = kGoalStiffness;
  goalConfig.rotationStiffness = kGoalStiffness;
  DynamicArray<BotIKGoalConfig> const goals{goalConfig};
  DynamicArray<real> const pose{0.35_r, 0.3_r};
  for (real const scale : {0.5_r, 1_r, 2_r}) {
    SCOPED_TRACE(scale);
    real const size = inputBot.scale * scale;
    // Align revolute pivots, making the scaled radial goal reachable by the output slider.
    TransformRT const root{
        bot.worldFromRoot.GetRotation(),
        bot.worldFromRoot.TransformPoint(Real3{0.3_r * (1_r - size), 0_r, 0_r})};
    real const extension = size * (0.7_r + pose[1]) + scale * 0.1_r - 0.8_r;
    TransformRT const expectedTip = ExpectedLinks(bot.worldFromRoot, pose[0], extension)[kTip];
    TransformRT const goal = expectedTip * goalConfig.linkFrameFromConstraintFrame;
    DynamicArray<real> const beforePose{ik->GetJointPositions()};
    DynamicArray<TransformRT> const beforeLinks{ik->GetWorldFromLinks()};
    ik->SetInputBotConfiguration(inputBot, pose, root, scale, goals, ExpectOK{});
    EXPECT_SPAN_EQ(ik->GetJointPositions(), MakeConstSpan(beforePose));
    EXPECT_SPAN_EQ(ik->GetWorldFromLinks(), MakeConstSpan(beforeLinks));
    direct->SetPositionGoal(
        kTip,
        goalConfig.linkFrameFromConstraintFrame.GetTranslation(),
        goal.GetTranslation(),
        kGoalStiffness,
        ExpectOK{});
    direct->SetRotationGoal(
        kTip,
        goalConfig.linkFrameFromConstraintFrame.GetRotation(),
        goal.GetRotation(),
        kGoalStiffness,
        ExpectOK{});
    EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
    EXPECT_TRUE(direct->SolveIK(ExpectOK{}));
    ExpectPoseNear(ik->GetJointPositions(), DynamicArray<real>{pose[0], extension});
    ExpectLinksNear(ik->GetWorldFromLinks(), direct->GetWorldFromLinks());
    {
      auto const suppressError = SuppressLogError();
      ik->SetInputBotConfiguration(inputBot, pose, root, 0_r, goals, ExpectNotOK{});
    }
    ik->SetJointPositions(bot.defaultPose, ExpectOK{});
    EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
    ExpectPoseNear(ik->GetJointPositions(), DynamicArray<real>{pose[0], extension});
    EXPECT_TRUE(NearEqual(
        ik->GetWorldFromLinks()[kTip] * goalConfig.linkFrameFromConstraintFrame,
        goal,
        kSolveTolerance));
    if (scale == 1_r) {
      ik->SetJointPositions(bot.defaultPose, ExpectOK{});
      ik->SetInputBotConfiguration(inputBot, pose, root, goals, ExpectOK{});
      EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
      ExpectPoseNear(ik->GetJointPositions(), DynamicArray<real>{pose[0], extension});
    }
  }
  EXPECT_EQ(inputBot.scale, originalInputBot.scale);
  EXPECT_EQ(inputBot.links, originalInputBot.links);
  EXPECT_EQ(inputBot.joints, originalInputBot.joints);
  EXPECT_EQ(inputBot.defaultPose, originalInputBot.defaultPose);
  EXPECT_EQ(inputBot.worldFromRoot, originalInputBot.worldFromRoot);
  EXPECT_EQ(bot.links, originalBot.links);
  EXPECT_EQ(bot.joints, originalBot.joints);
  EXPECT_EQ(bot.defaultPose, originalBot.defaultPose);
  EXPECT_EQ(bot.worldFromRoot, originalBot.worldFromRoot);
  EXPECT_EQ(bot.scale, originalBot.scale);
}

TEST_F(BotIKTest, InputBotScale_UnboundGoalsIgnoreScaleAndInvalidPreflightPreservesGoals) {
  auto const bot = MakeRobot();
  auto inputBot = MakeRobot(ArticulatedJointType::Free);
  inputBot.linearTransmissions.push_back(MakeTransmission());
  auto const ik = CreateBotIK(_mochiContext, bot, TestCreateParams(), ExpectOK{});
  ASSERT_NE(ik, nullptr);
  DynamicArray<real> const desiredPose{0.4_r, 0.25_r};
  TransformRT const desiredTip = ExpectedLinks({}, desiredPose[0], desiredPose[1])[kTip];
  BotIKGoalConfig goalConfig;
  goalConfig.name = "tip";
  goalConfig.outputBotLink = "d_tip";
  goalConfig.linkFrameFromConstraintFrame =
      TransformRT{Quaternion::RotationX(0.2_r), Real3{0.1_r, -0.05_r, 0_r}};
  goalConfig.baseOrInputLinkFrameFromConstraintFrame =
      desiredTip * goalConfig.linkFrameFromConstraintFrame;
  goalConfig.positionStiffness = kGoalStiffness;
  goalConfig.rotationStiffness = kGoalStiffness;
  DynamicArray<BotIKGoalConfig> const goals{goalConfig};
  TransformRT const unrelatedRoot{Quaternion::RotationZ(-0.3_r), Real3{9_r, 8_r, 7_r}};
  for (real const scale : {0.5_r, 2_r}) {
    ik->SetJointPositions(bot.defaultPose, ExpectOK{});
    ik->SetInputBotConfiguration(inputBot, {}, unrelatedRoot, scale, goals, ExpectOK{});
    EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
    ExpectPoseNear(ik->GetJointPositions(), desiredPose);
  }
  for (int fault = 0; fault < 7; ++fault) {
    SCOPED_TRACE(fault);
    auto invalid = goals;
    real scale = 2_r;
    if (fault < 4) {
      real const bad[] = {
          0_r, -1_r, std::numeric_limits<real>::infinity(), std::numeric_limits<real>::quiet_NaN()};
      scale = bad[fault];
    } else {
      // A valid changed goal precedes a late invalid binding; neither may replace old goals.
      invalid[0].baseOrInputLinkFrameFromConstraintFrame = {};
      BotIKGoalConfig late = goalConfig;
      late.name = "late";
      late.outputBotLink = "a_weld";
      late.inputBotLink = fault == 4 ? "missing" : "d_tip";
      if (fault == 5) {
        late.baseOrInputLinkFrameFromConstraintFrame.SetTranslation(
            Real3{std::numeric_limits<real>::max(), 0_r, 0_r});
      }
      if (fault == 6) {
        late.outputBotLink = "missing";
      }
      invalid.push_back(late);
    }
    DynamicArray<real> const beforePose{ik->GetJointPositions()};
    DynamicArray<TransformRT> const beforeLinks{ik->GetWorldFromLinks()};
    {
      auto const suppressError = SuppressLogError();
      ik->SetInputBotConfiguration(inputBot, {}, unrelatedRoot, scale, invalid, ExpectNotOK{});
    }
    EXPECT_SPAN_EQ(ik->GetJointPositions(), MakeConstSpan(beforePose));
    EXPECT_SPAN_EQ(ik->GetWorldFromLinks(), MakeConstSpan(beforeLinks));
    ik->SetJointPositions(bot.defaultPose, ExpectOK{});
    EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
    ExpectPoseNear(ik->GetJointPositions(), desiredPose);
  }
}

BotPrefab MakeFinger(
    ArticulatedJointType kind = ArticulatedJointType::Revolute,
    ArticulatedJointType rootType = ArticulatedJointType::Hard) {
  BotPrefab prefab;
  prefab.name = "synthetic_PIP_DIP";
  prefab.worldFromRoot = TransformRT{Quaternion::RotationY(0.25_r), Real3{1_r, 2_r, 3_r}};
  prefab.links = {MakeLink("root", kIndexNone), MakeLink("PIP", 0), MakeLink("DIP", 1)};
  prefab.joints = {
      MakeJoint("root_joint", rootType),
      MakeJoint("PIP_joint", kind, {0.3_r, 0_r, 0_r}),
      MakeJoint("DIP_joint", kind, {0.2_r, 0_r, 0_r})};
  if (kind == ArticulatedJointType::Revolute) {
    prefab.joints[1].parentLinkFromJoint.SetRotation(Quaternion::RotationZ(0.07_r));
    prefab.joints[2].parentLinkFromJoint.SetRotation(Quaternion::RotationZ(-0.04_r));
  }
  prefab.defaultPose = {0.17_r, -0.09_r};
  RebuildBotData(prefab, ExpectOK{});
  return prefab;
}

BotIKJointCouplingParams FingerCoupling(real ratio = 0.5_r, real stiffness = 10_r) {
  BotIKJointCouplingParams params;
  params.jointA = "PIP_joint";
  params.jointB = "DIP_joint";
  params.ratio = ratio;
  params.stiffness = stiffness;
  return params;
}

Quaternion FingerRotation(BotPrefab const& prefab, int index, real jointAngle) {
  real const rest = index == 1 ? 0.07_r : 0.03_r;
  return prefab.worldFromRoot.GetRotation() * Quaternion::RotationZ(rest + jointAngle);
}

void AnchorFingerBase(BotIK& ik, BotPrefab const& prefab) {
  ik.SetPositionGoal(0, {}, prefab.worldFromRoot.GetTranslation(), 1e6_r, ExpectOK{});
  ik.SetRotationGoal(0, {}, prefab.worldFromRoot.GetRotation(), 1e6_r, ExpectOK{});
}

// Stationary point of wA*(a-goalA)^2 + wSum*(a+b-goalSum)^2 +
// wCoupling*(ratio*a-b+offset)^2. Revolute fixtures use small collinear angles.
DynamicArray<real> CouplingReference(
    real ratio,
    real offset,
    real stiffness,
    real goalA,
    real goalSum,
    real weightA,
    real weightSum) {
  real const aa = weightA + weightSum + stiffness * ratio * ratio;
  real const ab = weightSum - stiffness * ratio;
  real const bb = weightSum + stiffness;
  real const rhsA = weightA * goalA + weightSum * goalSum - stiffness * ratio * offset;
  real const rhsB = weightSum * goalSum + stiffness * offset;
  real const determinant = aa * bb - ab * ab;
  return {(rhsA * bb - rhsB * ab) / determinant, (rhsB * aa - rhsA * ab) / determinant};
}

TEST_F(BotIKTest, Couplings_UnderconstrainedPIPDIP_SignedRatiosRestRotationsAndFloatingBase) {
  for (auto const rootType : {ArticulatedJointType::Hard, ArticulatedJointType::Free}) {
    for (bool const fixedBase : {false, true}) {
      for (real const ratio : {0.5_r, 2_r, -0.5_r}) {
        SCOPED_TRACE(ratio);
        auto const prefab = MakeFinger(ArticulatedJointType::Revolute, rootType);
        auto params = TestCreateParams();
        params.fixedBase = fixedBase;
        DynamicArray<BotIKJointCouplingParams> const couplings{FingerCoupling(ratio, 100_r)};
        auto ik = CreateBotIK(_mochiContext, prefab, params, MakeConstSpan(couplings), ExpectOK{});
        ASSERT_NE(ik, nullptr);
        EXPECT_SPAN_EQ(ik->GetJointPositions(), MakeConstSpan(prefab.defaultPose));
        DynamicArray<real> const initial{ik->GetJointPositions()};
        EXPECT_FALSE(ik->SolveIK(ExpectNotOK{}));
        EXPECT_SPAN_EQ(ik->GetJointPositions(), MakeConstSpan(initial));
        if (rootType == ArticulatedJointType::Free && !fixedBase) {
          AnchorFingerBase(*ik, prefab);
        }
        real const sum = 0.3_r;
        ik->SetRotationGoal(2, {}, FingerRotation(prefab, 2, sum), 100_r, ExpectOK{});
        EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
        real const a = sum / (1_r + ratio);
        ExpectPoseNear(ik->GetJointPositions(), DynamicArray<real>{a, ratio * a});
        EXPECT_TRUE(NearEqual(ik->GetWorldFromLinks()[0], prefab.worldFromRoot, kSolveTolerance));
        ik->SetJointPositions(DynamicArray<real>{-0.15_r, 0.23_r}, ExpectOK{});
        EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
        ExpectPoseNear(ik->GetJointPositions(), DynamicArray<real>{a, ratio * a});
        EXPECT_TRUE(prefab.linearTransmissions.empty());
        EXPECT_EQ(prefab.defaultPose, initial);
        EXPECT_EQ(prefab.joints[0].type, rootType);
      }
    }
  }
}

TEST_F(BotIKTest, Couplings_SoftBilateralPenalty_BothJointsMoveAndWeightsMatchReference) {
  auto prefab = MakeFinger();
  prefab.defaultPose = {0.4_r, -0.4_r};
  RebuildBotData(prefab, ExpectOK{});
  for (real const sign : {-1_r, 1_r}) {
    for (real const stiffness : {1_r, 10_r, 1000_r}) {
      for (real const ratio : {0.5_r, -0.5_r}) {
        SCOPED_TRACE(sign);
        SCOPED_TRACE(stiffness);
        SCOPED_TRACE(ratio);
        DynamicArray<BotIKJointCouplingParams> const couplings{FingerCoupling(ratio, stiffness)};
        auto ik = CreateBotIK(
            _mochiContext, prefab, TestCreateParams(), MakeConstSpan(couplings), ExpectOK{});
        ASSERT_NE(ik, nullptr);
        real const goalA = sign * 0.1_r;
        real const goalSum = sign * 0.3_r;
        ik->SetRotationGoal(1, {}, FingerRotation(prefab, 1, goalA), 100_r, ExpectOK{});
        ik->SetRotationGoal(2, {}, FingerRotation(prefab, 2, goalSum), 40_r, ExpectOK{});
        (void)ik->SolveIK(ExpectOK{});
        auto const reference =
            CouplingReference(ratio, 0_r, stiffness, goalA, goalSum, 100_r, 40_r);
        ExpectPoseNear(ik->GetJointPositions(), reference);
        EXPECT_GT(Abs(ik->GetJointPositions()[0] - prefab.defaultPose[0]), 0.02_r);
        EXPECT_GT(Abs(ik->GetJointPositions()[1] - prefab.defaultPose[1]), 0.02_r);
        // Finite stiffness cannot enforce exact linkage against incompatible independent goals.
        EXPECT_GT(Abs(ratio * ik->GetJointPositions()[0] - ik->GetJointPositions()[1]), 1e-4_r);
      }
    }
  }
}

TEST_F(BotIKTest, Couplings_PrismaticPairs_UniformNativeScaleAndSignedAnalyticalReference) {
  for (real const scale : {0.5_r, 1_r, 2_r}) {
    for (real const ratio : {0.5_r, -0.5_r}) {
      auto prefab = MakeFinger(ArticulatedJointType::Prismatic);
      prefab.scale = scale;
      DynamicArray<BotIKJointCouplingParams> const couplings{FingerCoupling(ratio, 80_r)};
      auto ik = CreateBotIK(
          _mochiContext, prefab, TestCreateParams(), MakeConstSpan(couplings), ExpectOK{});
      ASSERT_NE(ik, nullptr);
      ik->SetPositionGoal(
          1,
          {},
          prefab.worldFromRoot.TransformPoint(Real3{0.3_r * scale + 0.1_r, 0_r, 0_r}),
          100_r,
          ExpectOK{});
      ik->SetPositionGoal(
          2,
          {},
          prefab.worldFromRoot.TransformPoint(Real3{0.5_r * scale + 0.3_r, 0_r, 0_r}),
          40_r,
          ExpectOK{});
      (void)ik->SolveIK(ExpectOK{});
      ExpectPoseNear(
          ik->GetJointPositions(), CouplingReference(ratio, 0_r, 80_r, 0.1_r, 0.3_r, 100_r, 40_r));
      EXPECT_EQ(prefab.scale, scale);
      EXPECT_TRUE(prefab.linearTransmissions.empty());
    }
  }
}

TEST_F(BotIKTest, Couplings_EmptyDisabledAndZeroStiffness_MatchOldCreateBaseline) {
  auto const prefab = MakeFinger();
  auto const baseline = CreateBotIK(_mochiContext, prefab, TestCreateParams(), ExpectOK{});
  ASSERT_NE(baseline, nullptr);
  baseline->SetRotationGoal(2, {}, FingerRotation(prefab, 2, 0.3_r), 100_r, ExpectOK{});
  EXPECT_TRUE(baseline->SolveIK(ExpectOK{}));
  for (int mode = 0; mode < 3; ++mode) {
    DynamicArray<BotIKJointCouplingParams> couplings;
    if (mode != 0) {
      auto inactive = FingerCoupling();
      inactive.jointA = "stale_A";
      inactive.jointB = "stale_B";
      inactive.enabled = mode == 2;
      inactive.stiffness = mode == 2 ? 0_r : 10_r;
      couplings.push_back(inactive);
    }
    auto ik = CreateBotIK(
        _mochiContext, prefab, TestCreateParams(), MakeConstSpan(couplings), ExpectOK{});
    ASSERT_NE(ik, nullptr);
    ik->SetRotationGoal(2, {}, FingerRotation(prefab, 2, 0.3_r), 100_r, ExpectOK{});
    EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
    ExpectPoseNear(ik->GetJointPositions(), baseline->GetJointPositions());
    ExpectLinksNear(ik->GetWorldFromLinks(), baseline->GetWorldFromLinks());
  }
}

TEST_F(BotIKTest, Couplings_RuntimeValidation_ExactUniqueSingleDofNamesAndAllValues) {
  auto const prefab = MakeFinger();
  for (bool const enabled : {false, true}) {
    for (int fault = 0; fault < 11; ++fault) {
      auto coupling = FingerCoupling();
      coupling.enabled = enabled;
      if (fault == 0) {
        coupling.jointA.clear();
      }
      if (fault == 1) {
        coupling.jointB.clear();
      }
      if (fault == 2) {
        coupling.jointB = coupling.jointA;
      }
      if (fault == 3) {
        coupling.ratio = 0_r;
      }
      if (fault == 4) {
        coupling.ratio = std::numeric_limits<real>::infinity();
      }
      if (fault == 5) {
        coupling.ratio = std::numeric_limits<real>::quiet_NaN();
      }
      if (fault == 6) {
        coupling.stiffness = -1_r;
      }
      if (fault == 7) {
        coupling.stiffness = std::numeric_limits<real>::infinity();
      }
      if (fault == 8) {
        coupling.stiffness = std::numeric_limits<real>::quiet_NaN();
      }
      if (fault == 9) {
        coupling.offset = std::numeric_limits<real>::infinity();
      }
      if (fault == 10) {
        coupling.offset = std::numeric_limits<real>::quiet_NaN();
      }
      DynamicArray<BotIKJointCouplingParams> const couplings{coupling};
      BotIK::ValidateJointCouplings(prefab, couplings, ExpectNotOK{});
      EXPECT_EQ(
          CreateBotIK(
              _mochiContext, prefab, TestCreateParams(), MakeConstSpan(couplings), ExpectNotOK{}),
          nullptr);
    }
  }
  for (auto const* name : {"pip_joint", "missing", "root_joint"}) {
    auto coupling = FingerCoupling();
    coupling.jointA = name;
    DynamicArray<BotIKJointCouplingParams> couplings{coupling};
    BotIK::ValidateJointCouplings(prefab, couplings, ExpectNotOK{});
    couplings[0].enabled = false;
    BotIK::ValidateJointCouplings(prefab, couplings, ExpectOK{});
    couplings[0].enabled = true;
    couplings[0].stiffness = 0_r;
    BotIK::ValidateJointCouplings(prefab, couplings, ExpectOK{});
  }
  for (auto const type :
       {ArticulatedJointType::Hard, ArticulatedJointType::Spherical, ArticulatedJointType::Free}) {
    auto invalid = prefab;
    invalid.joints[2].type = type;
    BotIK::ValidateJointCouplings(
        invalid, DynamicArray<BotIKJointCouplingParams>{FingerCoupling()}, ExpectNotOK{});
  }
  auto ambiguous = prefab;
  ambiguous.joints[0].name = "PIP_joint";
  BotIK::ValidateJointCouplings(
      ambiguous, DynamicArray<BotIKJointCouplingParams>{FingerCoupling()}, ExpectNotOK{});
  for (real const ratio : {1e-10_r, -1e-10_r, std::numeric_limits<real>::min()}) {
    BotIK::ValidateJointCouplings(
        prefab, DynamicArray<BotIKJointCouplingParams>{FingerCoupling(ratio)}, ExpectOK{});
  }
  BotIK::ValidateJointCouplings(
      prefab,
      DynamicArray<BotIKJointCouplingParams>{FingerCoupling(), FingerCoupling()},
      ExpectOK{});
}

TEST_F(BotIKTest, Couplings_BatchPrevalidationAndLoaderFailure_NoMutationOrPartialPublishedActor) {
  auto prefab = MakeFinger();
  for (auto& link : prefab.links) {
    link.shapeFile = "test_tet";
  }
  auto const original = prefab;
  auto params = TestCreateParams();
  params.excludeCollisionShapes = false;
  TestBotLoader loader;
  DynamicArray<BotIKJointCouplingParams> couplings{FingerCoupling(), FingerCoupling()};
  couplings[1].jointB = "late_missing";
  int const initialShapes = _mochiContext->GetNumShapes();
  EXPECT_EQ(
      CreateBotIK(_mochiContext, prefab, params, MakeConstSpan(couplings), loader, ExpectNotOK{}),
      nullptr);
  EXPECT_EQ(loader.shapeLoads, 0);
  EXPECT_EQ(_mochiContext->GetNumShapes(), initialShapes);
  couplings.pop_back();
  loader.failOnShapeLoad = 2;
  EXPECT_EQ(
      CreateBotIK(_mochiContext, prefab, params, MakeConstSpan(couplings), loader, ExpectNotOK{}),
      nullptr);
  EXPECT_EQ(_mochiContext->GetNumShapes(), initialShapes);
  loader.failOnShapeLoad = -1;
  for (int repetition = 0; repetition < 3; ++repetition) {
    auto ik =
        CreateBotIK(_mochiContext, prefab, params, MakeConstSpan(couplings), loader, ExpectOK{});
    ASSERT_NE(ik, nullptr);
    ik->SetRotationGoal(2, {}, FingerRotation(prefab, 2, 0.3_r), 100_r, ExpectOK{});
    EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
    ik.reset();
    EXPECT_EQ(_mochiContext->GetNumShapes(), initialShapes);
  }
  EXPECT_EQ(prefab.links, original.links);
  EXPECT_EQ(prefab.joints, original.joints);
  EXPECT_EQ(prefab.defaultPose, original.defaultPose);
  EXPECT_EQ(prefab._dofIndices, original._dofIndices);
  EXPECT_EQ(prefab.worldFromRoot, original.worldFromRoot);
  EXPECT_TRUE(prefab.linearTransmissions.empty());
}

TEST_F(BotIKTest, Couplings_LegacyMixedKindOffsetRepeatedPairs_MatchAuthoredTransmission) {
  auto const prefab = MakeRobot();
  auto authored = prefab;
  authored.linearTransmissions = {MakeTransmission(), MakeTransmission()};
  BotIKJointCouplingParams coupling;
  coupling.jointA = "arm_joint";
  coupling.jointB = "slide_joint";
  coupling.ratio = 2_r;
  coupling.offset = 0.1_r;
  coupling.stiffness = 4000_r;
  DynamicArray<BotIKJointCouplingParams> const couplings{coupling, coupling};
  auto ik =
      CreateBotIK(_mochiContext, prefab, TestCreateParams(), MakeConstSpan(couplings), ExpectOK{});
  auto const reference = CreateBotIK(_mochiContext, authored, TestCreateParams(), ExpectOK{});
  ASSERT_NE(ik, nullptr);
  ASSERT_NE(reference, nullptr);
  for (real const angle : {-0.2_r, 0.1_r, 0.35_r}) {
    auto const goal = ExpectedLinks(prefab.worldFromRoot, angle, 0.25_r)[kTip];
    for (auto* instance : {ik.get(), reference.get()}) {
      instance->SetPositionGoal(kTip, {}, goal.GetTranslation(), 1000_r, ExpectOK{});
      instance->SetRotationGoal(kTip, {}, goal.GetRotation(), 1000_r, ExpectOK{});
    }
    EXPECT_EQ(ik->SolveIK(ExpectOK{}), reference->SolveIK(ExpectOK{}));
    ExpectPoseNear(ik->GetJointPositions(), reference->GetJointPositions());
    ExpectLinksNear(ik->GetWorldFromLinks(), reference->GetWorldFromLinks());
  }
  EXPECT_TRUE(prefab.linearTransmissions.empty());
}

TEST_F(BotIKTest, Couplings_CoexistWithAuthoredTransmissionTendonAndLimits_DirectParity) {
  auto prefab = MakeRobot();
  prefab.linearTransmissions.push_back(MakeTransmission());
  prefab.joints[kArm].minLimit = Real3{0_r, 0_r, -0.1_r};
  prefab.joints[kArm].maxLimit = Real3{0_r, 0_r, 0.25_r};
  prefab.joints[kArm].limitStiffness = 500_r;
  BotSpatialTendonPrefab tendon;
  tendon.name = "preserved_tendon";
  RoutingElement root;
  root.index = 0;
  root.localPosition = {0_r, 0.4_r, 0_r};
  RoutingElement tip;
  tip.index = kTip;
  tip.localPosition = {0.1_r, 0_r, 0_r};
  tendon.routingElements = {root, tip};
  tendon.targetDisplacement = 0.05_r;
  tendon.stiffness = 200_r;
  tendon.allowCompressiveForce = true;
  prefab.spatialTendons.push_back(tendon);
  RebuildBotData(prefab, ExpectOK{});
  auto const before = prefab;
  BotIKJointCouplingParams coupling;
  coupling.jointA = "arm_joint";
  coupling.jointB = "slide_joint";
  coupling.ratio = -0.5_r;
  coupling.stiffness = 300_r;
  ExpectDirectParity(prefab, true, DynamicArray<BotIKJointCouplingParams>{coupling});
  EXPECT_EQ(prefab.linearTransmissions.size(), before.linearTransmissions.size());
  EXPECT_EQ(
      prefab.linearTransmissions[0].jointCoefficients,
      before.linearTransmissions[0].jointCoefficients);
  EXPECT_EQ(prefab.spatialTendons.size(), before.spatialTendons.size());
  EXPECT_EQ(prefab.spatialTendons[0].routingElements, before.spatialTendons[0].routingElements);
  EXPECT_EQ(prefab.joints, before.joints);
}

TEST_F(BotIKTest, Couplings_InputBotFKIsIndependentOfPrivateOutputPenalty) {
  auto const outputBot = MakeFinger();
  auto inputBot = outputBot;
  inputBot.defaultPose = {0.25_r, -0.1_r};
  auto const originalInputBot = inputBot;
  DynamicArray<BotIKJointCouplingParams> const couplings{FingerCoupling(0.5_r, 100_r)};
  auto ik = CreateBotIK(
      _mochiContext, outputBot, TestCreateParams(), MakeConstSpan(couplings), ExpectOK{});
  ASSERT_NE(ik, nullptr);
  BotIKGoalConfig goal;
  goal.name = "DIP";
  goal.outputBotLink = "DIP";
  goal.inputBotLink = "DIP";
  goal.positionStiffness = 0_r;
  goal.rotationStiffness = 100_r;
  ik->SetInputBotConfiguration(
      inputBot, {}, outputBot.worldFromRoot, DynamicArray<BotIKGoalConfig>{goal}, ExpectOK{});
  EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
  ExpectPoseNear(ik->GetJointPositions(), DynamicArray<real>{0.1_r, 0.05_r});
  EXPECT_EQ(inputBot.defaultPose, originalInputBot.defaultPose);
  EXPECT_EQ(inputBot.joints, originalInputBot.joints);
  EXPECT_EQ(inputBot.links, originalInputBot.links);
  EXPECT_TRUE(inputBot.linearTransmissions.empty());
}

TEST_F(BotIKTest, Couplings_ReversedNativeDIPAxis_UsesJointCoordinatesNotWorldAngleSigns) {
  auto prefab = MakeFinger();
  prefab.joints[2].axis = {0_r, 0_r, -1_r};
  RebuildBotData(prefab, ExpectOK{});
  for (real const ratio : {0.5_r, -0.5_r}) {
    DynamicArray<BotIKJointCouplingParams> const couplings{FingerCoupling(ratio, 100_r)};
    auto ik = CreateBotIK(
        _mochiContext, prefab, TestCreateParams(), MakeConstSpan(couplings), ExpectOK{});
    ASSERT_NE(ik, nullptr);
    real const angle = 0.15_r;
    ik->SetRotationGoal(2, {}, FingerRotation(prefab, 2, angle), 100_r, ExpectOK{});
    EXPECT_TRUE(ik->SolveIK(ExpectOK{}));
    // DIP's reversed native axis makes the world rotation qA-qB, not qA+qB.
    real const a = angle / (1_r - ratio);
    ExpectPoseNear(ik->GetJointPositions(), DynamicArray<real>{a, ratio * a});
    EXPECT_TRUE(NearEqual(
        ik->GetWorldFromLinks()[2].GetRotation(),
        FingerRotation(prefab, 2, angle),
        kSolveTolerance));
  }
}

} // namespace
