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

#include <superdex_robotics/sensors/sensor_articulated_joint_wrench.h>

#include <superdex_robotics/core/context.h>
#include <superdex_robotics/sensors/sensor_graph.h>
#include <superdex_robotics/superdex_robotics.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_physics/mochi_physics.h>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

real constexpr kBaseSize = 0.1_r;
real constexpr kPoleLength = 0.5_r;
real constexpr kPoleWidth = 0.05_r;
real constexpr kTipSize = 0.04_r;
real constexpr kDensity = 1000_r;
real constexpr kDt = 1e-3_r;
Real3 constexpr kJointAxis = {0_r, 0_r, 1_r};
int constexpr kPoleJoint = 1;

struct DestroyRoboticsCtx {
  void operator()(RoboticsContext* ctx) const {
    DestroyRoboticsContext(ctx);
  }
};
using RoboticsCtxPtr = std::unique_ptr<RoboticsContext, DestroyRoboticsCtx>;

} // namespace

class SensorArticulatedJointWrenchTest : public testing::Test {
 protected:
  void SetUp() override {
    _context = mochi::CreateContext(0);
    ASSERT_NE(_context, nullptr);
    _scene = _context->CreateScene("SensorArticulatedJointWrenchTest");
    ASSERT_NE(_scene, nullptr);
  }

  void TearDown() override {
    _context->DestroyScene(_scene);
    mochi::DestroyContext(_context);
  }

  ShapeHandle MakeBoxShape(Real3 const& size) {
    auto&& [coordinates, connectivity] = CreateMinimalTetMeshUnitCube(size);
    return _context->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), ExpectOK{});
  }

  void BuildPendulum(
      ArticulatedJointType rootJoint = ArticulatedJointType::Hard,
      ArticulatedJointType poleJoint = ArticulatedJointType::Revolute,
      bool poleHasGravity = true,
      bool withHardTip = false) {
    ShapeHandle const baseShape = MakeBoxShape(Real3{kBaseSize, kBaseSize, kBaseSize});
    ShapeHandle const poleShape = MakeBoxShape(Real3{kPoleLength, kPoleWidth, kPoleWidth});
    ArticulatedActorParams params;
    params.name = "pendulum";
    ArticulatedJointParams root;
    root.type = rootJoint;
    ArticulatedJointParams pole;
    pole.type = poleJoint;
    pole.parentLinkFromJoint = TransformRT{Real3{kBaseSize, 0.5_r * kBaseSize, 0.5_r * kBaseSize}};
    pole.axis = kJointAxis;
    params.joints = {root, pole};
    if (withHardTip) {
      ArticulatedJointParams tip;
      tip.type = ArticulatedJointType::Hard;
      tip.parentLinkFromJoint =
          TransformRT{Real3{kPoleLength, 0.5_r * kPoleWidth, 0.5_r * kPoleWidth}};
      params.joints.push_back(tip);
    }

    ArticulatedLinkParams baseLink;
    baseLink.name = "base";
    baseLink.parentLink = -1;
    baseLink.shape = baseShape;
    baseLink.colliderType = ColliderType::Box;
    baseLink.density = kDensity;
    ArticulatedLinkParams poleLink;
    poleLink.name = "pole";
    poleLink.parentLink = 0;
    poleLink.parentJointFromLink =
        TransformRT{Real3{0_r, -0.5_r * kPoleWidth, -0.5_r * kPoleWidth}};
    poleLink.shape = poleShape;
    poleLink.colliderType = ColliderType::Box;
    poleLink.hasGravity = poleHasGravity;
    poleLink.density = kDensity;
    params.links = {baseLink, poleLink};
    if (withHardTip) {
      ArticulatedLinkParams tipLink;
      tipLink.name = "tip";
      tipLink.parentLink = kPoleJoint;
      tipLink.parentJointFromLink = TransformRT{Real3{0_r, -0.5_r * kTipSize, -0.5_r * kTipSize}};
      tipLink.shape = MakeBoxShape(Real3{kTipSize, kTipSize, kTipSize});
      tipLink.colliderType = ColliderType::Box;
      tipLink.density = kDensity;
      params.links.push_back(tipLink);
    }
    _actor = _scene->CreateArticulatedActor(params, ExpectOK{});
    ASSERT_NE(_actor, nullptr);
    Span<ActorHandle const> const links = _actor->GetNestedLinkActors(ExpectOK{});
    _root = _scene->GetActor(links[0]);
    _pole = _scene->GetActor(links[kPoleJoint]);
    ASSERT_NE(_root, nullptr);
    ASSERT_NE(_pole, nullptr);
    if (withHardTip) {
      _tip = _scene->GetActor(links[kPoleJoint + 1]);
      ASSERT_NE(_tip, nullptr);
    }
  }

  void FreezeAllDofs() {
    int const numDofs = _actor->GetNumDofs();
    std::vector<int> dofIndices(numDofs);
    std::vector<real> dofValues(numDofs);
    for (int i = 0; i < numDofs; ++i) {
      dofIndices[i] = i;
    }
    _actor->GetDofValues(dofIndices, dofValues, ExpectOK{});
    _actor->ClearBoundaryConditions();
    _actor->AddBoundaryConditionDofsWorld(dofIndices, dofValues, ExpectOK{});
  }

  void Advance(SensorGraph& graph, int firstFrame, int lastFrame) {
    Error error;
    for (int frame = firstFrame; frame <= lastFrame; ++frame) {
      _scene->Step(kDt);
      graph.Step(
          {.t = static_cast<double>(frame * kDt), .dt = static_cast<double>(kDt), .step = frame},
          error);
      ASSERT_TRUE(error.IsOK()) << error.ToString();
    }
  }

  Context* _context = nullptr;
  Scene* _scene = nullptr;
  Actor* _actor = nullptr;
  Actor* _root = nullptr;
  Actor* _pole = nullptr;
  Actor* _tip = nullptr;
};

TEST(SensorArticulatedJointWrenchLayoutTest, ClassifiesOnlyAllHardWorldChainsAsStatic) {
  struct Case {
    DynamicArray<ArticulatedJointType> joints;
    DynamicArray<int32_t> parents;
    DynamicArray<bool> expected;
  };
  DynamicArray<Case> const cases = {
      {.joints = {ArticulatedJointType::Hard, ArticulatedJointType::Revolute},
       .parents = {-1, 0},
       .expected = {true, false}},
      {.joints = {ArticulatedJointType::Free, ArticulatedJointType::Hard},
       .parents = {-1, 0},
       .expected = {false, false}},
      {.joints = {ArticulatedJointType::Prismatic, ArticulatedJointType::Hard},
       .parents = {-1, 0},
       .expected = {false, false}},
      {.joints = {ArticulatedJointType::Hard, ArticulatedJointType::Hard},
       .parents = {-1, 0},
       .expected = {true, true}},
      {.joints =
           {ArticulatedJointType::Hard, ArticulatedJointType::Revolute, ArticulatedJointType::Hard},
       .parents = {-1, 0, 1},
       .expected = {true, false, false}},
      // A static chain stays static only while every joint is Hard.
      {.joints =
           {ArticulatedJointType::Hard, ArticulatedJointType::Hard, ArticulatedJointType::Revolute},
       .parents = {-1, 0, 1},
       .expected = {true, true, false}},
      // Branches are classified independently: a Hard sibling stays static beside a moving one.
      {.joints =
           {ArticulatedJointType::Hard, ArticulatedJointType::Hard, ArticulatedJointType::Revolute},
       .parents = {-1, 0, 0},
       .expected = {true, true, false}},
  };
  for (Case const& testCase : cases) {
    EXPECT_EQ(
        testCase.expected,
        ClassifyStaticArticulatedLinks(testCase.joints, testCase.parents, ExpectOK{}));
  }
  EXPECT_TRUE(ClassifyStaticArticulatedLinks({}, {}, ExpectNotOK{}).empty());
  EXPECT_TRUE(ClassifyStaticArticulatedLinks(
                  DynamicArray<ArticulatedJointType>{ArticulatedJointType::Hard},
                  DynamicArray<int32_t>{1},
                  ExpectNotOK{})
                  .empty());
}

TEST_F(SensorArticulatedJointWrenchTest, LayoutOmitsStaticRootAndNamesMeasurableRows) {
  BuildPendulum();
  SensorArticulatedJointWrench sensor(_pole, "", ExpectOK{});
  ASSERT_EQ(1, sensor.GetJointIndices().size());
  EXPECT_EQ(kPoleJoint, sensor.GetJointIndices()[0]);
  auto const* const layout = TryGetSensorMetadata<ArticulatedJointWrenchMeta>(sensor);
  ASSERT_NE(layout, nullptr);
  EXPECT_EQ("world", layout->coordinateFrame);
  EXPECT_EQ("child_joint_origin", layout->momentReference);
  ASSERT_EQ(1, layout->rowJointIndices.size());
  ASSERT_EQ(1, layout->parentLinkIndices.size());
  ASSERT_EQ(1, layout->omittedStaticLinkIndices.size());
  EXPECT_EQ(kPoleJoint, layout->rowJointIndices[0]);
  EXPECT_EQ(0, layout->parentLinkIndices[0]);
  EXPECT_EQ(0, layout->omittedStaticLinkIndices[0]);

  SensorGraph graph;
  graph.Add("joints", &sensor, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("joints", true, mochi::test::ExpectOK{});
  Error error;
  graph.Build({.scene = _scene, .robotics = nullptr}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ((DynamicArray<SensorDimension>{1, 6}), sensor.GetFormat().shape);
  ASSERT_EQ(2, sensor.GetFormat().axes.size());
  EXPECT_EQ("joint", sensor.GetFormat().axes[0].name);
  EXPECT_EQ("wrench_component", sensor.GetFormat().axes[1].name);
}

TEST_F(SensorArticulatedJointWrenchTest, RejectsAnAllStaticActor) {
  BuildPendulum(ArticulatedJointType::Hard, ArticulatedJointType::Hard);
  SensorArticulatedJointWrench sensor(_pole, "", ExpectNotOK{});
}

TEST_F(SensorArticulatedJointWrenchTest, PhysicsQueryFailurePropagatesFromStep) {
  BuildPendulum();
  SensorArticulatedJointWrench sensor(_pole, "", ExpectOK{});
  SensorGraph graph;
  graph.Add("joints", &sensor, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("joints", true, mochi::test::ExpectOK{});
  Error error;
  graph.Build({.scene = _scene, .robotics = nullptr}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  auto const* const metadata = TryGetSensorMetadata<ArticulatedJointWrenchMeta>(sensor);
  ASSERT_NE(metadata, nullptr);
  auto const* const rowJointIndices = metadata->rowJointIndices.data();
  sensor.Shutdown();
  EXPECT_EQ(rowJointIndices, metadata->rowJointIndices.data());
  ASSERT_EQ(1, metadata->rowJointIndices.size());
  EXPECT_EQ(kPoleJoint, metadata->rowJointIndices[0]);
  EXPECT_EQ(nullptr, TryGetSensorMetadata<ArticulatedJointWrenchMeta>(sensor));

  graph.Step({.t = 0.001, .dt = 0.001, .step = 1}, error);
  EXPECT_FALSE(error.IsOK());
  EXPECT_EQ(-1, sensor.Latest().captureStep);
}

TEST_F(SensorArticulatedJointWrenchTest, GraphPublishesActorWideRowsAndLatestOnlyReads) {
  BuildPendulum();
  FreezeAllDofs();
  SensorArticulatedJointWrench sensor(_pole, "", ExpectOK{});
  SensorGraph graph;
  graph.Add("joints", &sensor, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("joints", true, mochi::test::ExpectOK{});
  Error error;
  graph.Build({.scene = _scene, .robotics = nullptr}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(sensor.GetFormat().kind, "Wrench6DList");
  EXPECT_EQ((DynamicArray<SensorDimension>{1, 6}), sensor.GetFormat().shape);
  graph.Step({.t = 0.0, .dt = 0.0, .step = 0}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(-1, sensor.Latest().captureStep);

  Advance(graph, 1, 4);
  SensorSample const first = sensor.Latest();
  SensorSample const second = sensor.Latest();
  ASSERT_EQ(4, first.captureStep);
  EXPECT_EQ(first.captureStep, second.captureStep);
  EXPECT_EQ(first.data, second.data);
  ASSERT_EQ(6, first.Floats().size());

  real const expectedForce = _pole->GetMass(ExpectOK{}) * Norm(_scene->GetGravity());
  Real3 const jointOrigin = {
      kBaseSize,
      0.5_r * kBaseSize,
      0.5_r * kBaseSize,
  };
  Real3 const poleCom =
      _pole->GetRootTransform().TransformPoint(_pole->GetRigidCenterOfMassLocal(ExpectOK{}));
  real const expectedTorque = expectedForce * (poleCom - jointOrigin)[0];
  EXPECT_NEAR(0_r, first.Floats()[0], 1e-4_r * expectedForce);
  EXPECT_NEAR(expectedForce, first.Floats()[1], 1e-4_r * expectedForce);
  EXPECT_NEAR(expectedTorque, first.Floats()[5], 1e-4_r * expectedTorque);
}

/* The BOTA mount: a Hard joint below a moving joint is not static, so its row is reported, and both
 * rows carry their whole distal subtree (the tip's weight reaches the pole joint too). */
TEST_F(SensorArticulatedJointWrenchTest, HardJointBelowAMovingJointIsReportedAndComplete) {
  BuildPendulum(
      ArticulatedJointType::Hard,
      ArticulatedJointType::Revolute,
      /*poleHasGravity=*/true,
      /*withHardTip=*/true);
  FreezeAllDofs();
  SensorArticulatedJointWrench sensor(_pole, "", ExpectOK{});
  ASSERT_EQ(2, sensor.GetJointIndices().size());
  EXPECT_EQ(kPoleJoint, sensor.GetJointIndices()[0]);
  EXPECT_EQ(kPoleJoint + 1, sensor.GetJointIndices()[1]);
  auto const* const layout = TryGetSensorMetadata<ArticulatedJointWrenchMeta>(sensor);
  ASSERT_NE(layout, nullptr);
  ASSERT_EQ(1, layout->omittedStaticLinkIndices.size());
  EXPECT_EQ(0, layout->omittedStaticLinkIndices[0]);

  SensorGraph graph;
  graph.Add("joints", &sensor, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("joints", true, mochi::test::ExpectOK{});
  Error error;
  graph.Build({.scene = _scene, .robotics = nullptr}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  Advance(graph, 1, 4);
  Span<float const> const rows = sensor.Latest().Floats();
  ASSERT_EQ(12, rows.size());

  real const g = Norm(_scene->GetGravity());
  auto com = [](Actor* link) {
    return link->GetRootTransform().TransformPoint(link->GetRigidCenterOfMassLocal(ExpectOK{}));
  };
  real const poleMass = _pole->GetMass(ExpectOK{});
  real const tipMass = _tip->GetMass(ExpectOK{});
  Real3 const poleJointOrigin = {kBaseSize, 0.5_r * kBaseSize, 0.5_r * kBaseSize};
  Real3 const tipJointOrigin = _pole->GetRootTransform().TransformPoint(
      Real3{kPoleLength, 0.5_r * kPoleWidth, 0.5_r * kPoleWidth});

  // Pole row: the pole and the tip hanging from it.
  real const poleForce = (poleMass + tipMass) * g;
  real const poleTorque = g *
      (poleMass * (com(_pole) - poleJointOrigin)[0] + tipMass * (com(_tip) - poleJointOrigin)[0]);
  EXPECT_NEAR(poleForce, rows[1], 1e-3_r * poleForce);
  EXPECT_NEAR(poleTorque, rows[5], 1e-3_r * poleTorque);
  // Tip row: the tip alone, about its own joint origin.
  real const tipForce = tipMass * g;
  real const tipTorque = tipForce * (com(_tip) - tipJointOrigin)[0];
  EXPECT_NEAR(tipForce, rows[7], 1e-3_r * tipForce);
  EXPECT_NEAR(tipTorque, rows[11], 1e-3_r * tipTorque);
}

TEST_F(SensorArticulatedJointWrenchTest, MultipleSourcesShareTheReferenceCountedQuery) {
  BuildPendulum();
  FreezeAllDofs();
  auto survivor = std::make_unique<SensorArticulatedJointWrench>(_pole, "", ExpectOK{});
  {
    SensorArticulatedJointWrench temporary(_pole, "", ExpectOK{});
    _scene->Step(kDt);
  }
  SensorGraph graph;
  graph.Add("survivor", survivor.get(), mochi::test::ExpectOK{});
  graph.SetOutputEnabled("survivor", true, mochi::test::ExpectOK{});
  Error error;
  graph.Build({.scene = _scene, .robotics = nullptr}, error);
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  Advance(graph, 1, 2);
  EXPECT_EQ(2, survivor->Latest().captureStep);
}

TEST_F(SensorArticulatedJointWrenchTest, RegisteredFactoryHasNoParallelConfigObject) {
  BuildPendulum();
  RoboticsCtxPtr const robotics{CreateRoboticsContext()};
  SensorHandle const handle = robotics->CreateSensor(
      SensorArticulatedJointWrench::TypeName(), _pole, "joint_rows", "{}", ExpectOK{});
  ASSERT_TRUE(handle.IsValid());
  EXPECT_EQ(SensorArticulatedJointWrench::TypeName(), robotics->GetSensor(handle)->GetTypeName());
  robotics->CreateSensor(
      SensorArticulatedJointWrench::TypeName(),
      _pole,
      "bad_params",
      R"({"unsupported": true})",
      ExpectNotOK{});
}
