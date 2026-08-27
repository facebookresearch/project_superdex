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
#include <mochi_physics/utils/mochi_prefab.h>

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_physics/src/mochi_articulated_body.h>
#include <mochi_physics/src/mochi_context.h>
#include <mochi_physics/src/mochi_group.h>
#include <mochi_physics/src/mochi_rigid.h>
#include <mochi_physics/src/mochi_shape.h>
#include <mochi_physics/src/mochi_snle.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

using namespace mochi;

// The articulated actor assets used by these tests are not shipped externally.
#if MOCHI_USE_HDF5 && MOCHI_INTERNAL
#define MOCHI_HDF5_AND_INTERNAL 1
#else
#define MOCHI_HDF5_AND_INTERNAL 0

#endif
using namespace mochi::articulated;

using mochi::experimental::ControlType;
using mochi::experimental::SetArticulatedForceAndTargetPose;

template <typename RandomEngine>
static auto RandomVector(RandomEngine& generator, int size) {
  std::vector<real> res(size);
  SetRandom(generator, -1_r, 1_r, MakeSpan(res));
  return res;
}

/**************************************************************************************
  ArticulatedRigidTest - Test fixture used to test functionality of the articulated body
**************************************************************************************/
class ArticulatedRigidTest : public test::MochiSceneTestBase {
 public:
  ShapeHandle _cubeShape;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp(); // call down
    _scene->SetGravity(kDefaultGravity);

    auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
    _cubeShape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});
  }
};

/**************************************************************************************
  Tests for articulated rigids (components of articulated body)
**************************************************************************************/

static void NormalizeQuaternion(ColumnVectorView<real, RigidSize::kRot> outVector) {
  auto rot = Quaternion(Load<Vec4r>(outVector.data()));
  rot = Normalize(rot);
  Store(outVector.data(), rot.data);
}

// A single-link actor with a Hard (weld) root joint has zero reduced dofs, so no CActorSnle
// component is emplaced on the compound. The scene must still step without crashing.
TEST_F(ArticulatedRigidTest, ZeroDof_SingleHardJoint_NoActorSnle) {
  entt::registry& reg = GetRegistry();

  ArticulatedActorParams params;
  params.joints = {{.type = ArticulatedJointType::Hard}};
  params.links = {{.parentLink = -1, .shape = _cubeShape, .colliderType = ColliderType::Box}};
  auto* ac = _scene->CreateArticulatedActor(params, test::ExpectOK{});
  auto eAC = GetEntity(ac->GetHandle());

  EXPECT_EQ(reg.get<CArticulatedProps const>(eAC).reducedDofsDim, 0u);
  EXPECT_FALSE(reg.all_of<CActorSnle>(eAC));

  _scene->Step(1e-2_r);
}

// A two-link actor with all Hard joints (mirrors the rigid_on_hard_articulated prefab) also has
// zero reduced dofs and thus no CActorSnle. The scene must still step without crashing.
TEST_F(ArticulatedRigidTest, ZeroDof_AllHardMultiLink_NoActorSnle) {
  entt::registry& reg = GetRegistry();

  ArticulatedActorParams params;
  params.joints = {{.type = ArticulatedJointType::Hard}, {.type = ArticulatedJointType::Hard}};
  params.links = {
      {.parentLink = -1, .shape = _cubeShape, .colliderType = ColliderType::Box},
      {.parentLink = 0, .shape = _cubeShape, .colliderType = ColliderType::Box}};
  auto* ac = _scene->CreateArticulatedActor(params, test::ExpectOK{});
  auto eAC = GetEntity(ac->GetHandle());

  EXPECT_EQ(reg.get<CArticulatedProps const>(eAC).reducedDofsDim, 0u);
  EXPECT_FALSE(reg.all_of<CActorSnle>(eAC));

  _scene->Step(1e-2_r);
}

TEST_F(ArticulatedRigidTest, Rigid_EntitySetSolution) {
  // Get registry
  entt::registry& reg = GetRegistry();

  // Create 2 rigid actors
  RigidActorParams rParams{.shape = _cubeShape, .colliderType = ColliderType::Box};

  auto* actorA = _scene->CreateRigidActor(rParams, test::ExpectOK{});
  auto eA = GetEntity(actorA);
  auto* actorB = _scene->CreateRigidActor(rParams, test::ExpectOK{});
  auto eB = GetEntity(actorB);

  // Create an articulated compound with 2 internal rigid actors
  ArticulatedActorParams acParamsNew;
  acParamsNew.joints = {
      {.type = ArticulatedJointType::Free}, {.type = ArticulatedJointType::Spherical}};
  acParamsNew.links = {
      {.parentLink = -1, .shape = _cubeShape, .colliderType = ColliderType::Box},
      {.parentLink = 0, .shape = _cubeShape, .colliderType = ColliderType::Box}};
  auto* ac = _scene->CreateArticulatedActor(acParamsNew, test::ExpectOK{});
  auto actors = ac->GetNestedLinkActors(test::ExpectOK{});

  auto eC = GetEntity(actors[0]);
  auto eD = GetEntity(actors[1]);
  auto eAC = GetEntity(ac->GetHandle());

  // Set values for solution vectors of the separate rigid actors
  ColumnVector<real> solA = {{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r}};
  NormalizeQuaternion(solA.BottomRows<RigidSize::kRot>(RigidSize::kRot));
  ColumnVector<real> solB = {{0.1_r, 0.2_r, 0.3_r, 0.4_r, 0.5_r, 0.6_r, 0.7_r}};
  NormalizeQuaternion(solB.BottomRows<RigidSize::kRot>(RigidSize::kRot));

  // Set full dofs on the articulated compound
  ColumnVector<real> solCD(solA.size() + solB.size());
  solCD.TopRows(solA.size()) = solA;
  solCD.MiddleRows(solA.size(), solB.size()) = solB;
  auto& fullDofs = reg.get<CArticulatedFullPose>(eAC);
  fullDofs.value = solCD;

  // Call SetSolution on the separate rigid actors
  auto& rbStateA = reg.get<CRigidState<TimeStep::Current>>(eA);
  mochi::rigid::EntitySetSolution(solA, {}, {}, rbStateA);

  auto& rbStateB = reg.get<CRigidState<TimeStep::Current>>(eB);
  mochi::rigid::EntitySetSolution(solB, {}, {}, rbStateB);

  // Call EntitySetSolution on the articulated compound's internal rigid actors
  auto& rbStateC = reg.get<CRigidState<TimeStep::Current>>(eC);
  auto& dofOffsetC = reg.get<CDofOffset>(eC);
  dofOffsetC.poseOffset = 0;
  mochi::articulated::rigid::EntitySetSolution(
      {}, dofOffsetC, reg.get<CArticulatedFullPoseRef>(eC), rbStateC);

  auto& rbStateD = reg.get<CRigidState<TimeStep::Current>>(eD);
  auto& dofOffsetD = reg.get<CDofOffset>(eD);
  dofOffsetD.poseOffset = RigidSize::kAll;
  mochi::articulated::rigid::EntitySetSolution(
      {}, dofOffsetD, reg.get<CArticulatedFullPoseRef>(eD), rbStateD);

  // Validate that the resulting internal state of the separate and the internal rigid actors is the
  // same
  EXPECT_EQ(rbStateA.value.GetTranslation(), rbStateC.value.GetTranslation());
  EXPECT_EQ(rbStateA.value.GetRotation(), rbStateC.value.GetRotation());
  EXPECT_EQ(rbStateB.value.GetTranslation(), rbStateD.value.GetTranslation());
  EXPECT_EQ(rbStateB.value.GetRotation(), rbStateD.value.GetRotation());
}

TEST_F(ArticulatedRigidTest, Rigid_EntityPostNewSolution) {
  // Get registry
  entt::registry& reg = GetRegistry();

  // Create 2 rigid actors
  RigidActorParams rParams{.shape = _cubeShape, .colliderType = ColliderType::Box};

  auto* actorA = _scene->CreateRigidActor(rParams, test::ExpectOK{});
  auto eA = GetEntity(actorA);
  auto* actorB = _scene->CreateRigidActor(rParams, test::ExpectOK{});
  auto eB = GetEntity(actorB);

  // Create an articulated compound with 2 internal rigid actors
  ArticulatedActorParams acParamsNew;
  acParamsNew.joints = {
      {.type = ArticulatedJointType::Free}, {.type = ArticulatedJointType::Spherical}};
  acParamsNew.links = {
      {.parentLink = -1, .shape = _cubeShape, .colliderType = ColliderType::Box},
      {.parentLink = 0, .shape = _cubeShape, .colliderType = ColliderType::Box}};
  auto* ac = _scene->CreateArticulatedActor(acParamsNew, test::ExpectOK{});
  auto actors = ac->GetNestedLinkActors(test::ExpectOK{});

  auto eC = GetEntity(actors[0]);
  auto eD = GetEntity(actors[1]);
  auto eAC = GetEntity(ac->GetHandle());

  // Define solutions
  std::vector<real> solA = {1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r};
  NormalizeQuaternion(AsView(solA).BottomRows<RigidSize::kRot>(RigidSize::kRot));
  std::vector<real> solB = {0.1_r, 0.2_r, 0.3_r, 0.4_r, 0.5_r, 0.6_r, 0.7_r};
  NormalizeQuaternion(AsView(solB).BottomRows<RigidSize::kRot>(RigidSize::kRot));
  std::vector<real> sol;
  sol.reserve(solA.size() + solB.size());
  std::copy(solA.begin(), solA.end(), std::back_inserter(sol));
  std::copy(solB.begin(), solB.end(), std::back_inserter(sol));

  // Create individual problem objects for each separate rigid
  ColumnVector<real, 2 * RigidSize::kAll> solutionAB;
  std::copy(sol.begin(), sol.end(), solutionAB.data());
  SnleProblem<real> problemAB;
  problemAB.solution.Reset(AsView(solutionAB));

  // Call EntityPostNewSolution on the separate rigid actors
  auto& dofOffsetA = reg.get<CDofOffset>(eA);
  dofOffsetA.poseOffset = 0;
  auto& dofOffsetB = reg.get<CDofOffset>(eB);
  dofOffsetB.poseOffset = RigidSize::kAll;
  std::vector<entt::entity> rEs = {eA, eB};
  ecs::InvokeForEach(
      mochi::rigid::EntityPostNewSolution, reg, rEs, AsConstView(problemAB.solution));

  // Set full dofs on articulated body
  auto& fullDofs = reg.get<CArticulatedFullPose>(eAC);
  fullDofs.value = solutionAB;

  // Call EntityPostNewSolution on the articulated compound's internal rigid actors
  auto& dofOffsetC = reg.get<CDofOffset>(eC);
  dofOffsetC.poseOffset = 0;
  auto& dofOffsetD = reg.get<CDofOffset>(eD);
  dofOffsetD.poseOffset = RigidSize::kAll;
  std::vector<entt::entity> arEs = {eC, eD};
  ecs::InvokeForEach(mochi::articulated::rigid::EntityPostNewSolution, reg, arEs);

  // Validate that the resulting internal state of the separate and the internal rigid actors is the
  // same
  auto& rbInertiaA = reg.get<CRigidBodyInertia>(eA);
  auto& rbStateA = reg.get<CRigidState<TimeStep::Current>>(eA).value;
  auto& rbRootTxA = reg.get<CRootTransform>(eA);
  auto& rbInertiaC = reg.get<CRigidBodyInertia>(eC);
  auto& rbStateC = reg.get<CRigidState<TimeStep::Current>>(eC).value;
  auto& rbRootTxC = reg.get<CRootTransform>(eC);

  EXPECT_EQ(ToReal4(rbInertiaA.GetCenterOfMassLocal()), ToReal4(rbInertiaC.GetCenterOfMassLocal()));
  EXPECT_EQ(rbInertiaA.GetSecondMomentLocal(), rbInertiaC.GetSecondMomentLocal());
  EXPECT_EQ(rbStateA.GetTranslation(), rbStateC.GetTranslation());
  EXPECT_EQ(rbStateA.GetRotation(), rbStateC.GetRotation());
  EXPECT_EQ(rbRootTxA.worldFromLocal, rbRootTxC.worldFromLocal);

  auto& rbInertiaB = reg.get<CRigidBodyInertia>(eB);
  auto& rbStateB = reg.get<CRigidState<TimeStep::Current>>(eB).value;
  auto& rbRootTxB = reg.get<CRootTransform>(eB);
  auto& rbInertiaD = reg.get<CRigidBodyInertia>(eD);
  auto& rbStateD = reg.get<CRigidState<TimeStep::Current>>(eD).value;
  auto& rbRootTxD = reg.get<CRootTransform>(eD);

  EXPECT_EQ(ToReal4(rbInertiaB.GetCenterOfMassLocal()), ToReal4(rbInertiaD.GetCenterOfMassLocal()));
  EXPECT_EQ(rbInertiaB.GetSecondMomentLocal(), rbInertiaD.GetSecondMomentLocal());
  EXPECT_EQ(rbStateB.GetTranslation(), rbStateD.GetTranslation());
  EXPECT_EQ(rbStateB.GetRotation(), rbStateD.GetRotation());
  EXPECT_EQ(rbRootTxB.worldFromLocal, rbRootTxD.worldFromLocal);
}

TEST_F(ArticulatedRigidTest, Rigid_EntityPreStep) {
  // Get registry
  entt::registry& reg = GetRegistry();

  // Create an articulated rigid and a basic rigid actor
  RigidActorParams params{.shape = _cubeShape, .colliderType = ColliderType::Box};

  Actor* rigid = _scene->CreateRigidActor(params, test::ExpectOK{});
  auto rE = GetEntity(rigid->GetHandle());

  Actor* articulatedRigid = _scene->CreateRigidActor(params, test::ExpectOK{});
  auto arE = GetEntity(articulatedRigid->GetHandle());
  reg.emplace<TagArticulatedLinkActor>(arE);

  // Set components
  auto& rbPose = reg.get<CRigidState<TimeStep::Current>>(rE).value;
  rbPose.SetTranslation(Real3{1_r, 2_r, 3_r});
  SetRotationVector(Vec4r{0.1_r, 0.2_r, 0.3_r}, rbPose);

  auto& intState = reg.get<CTimeIntegratorState>(rE);
  intState.dtStage = 0.01_r;

  auto& arbPose = reg.get<CRigidState<TimeStep::Current>>(arE).value;
  arbPose.SetTranslation(rbPose.GetTranslation());
  arbPose.SetRotation(rbPose.GetRotation());

  auto& aIntState = reg.get<CTimeIntegratorState>(arE);
  aIntState.dtStage = intState.dtStage;

  // Perform pre-step on both
  ecs::InvokeOnEntity(mochi::rigid::EntityIncrementStep, reg, rE);
  ecs::InvokeOnEntity(mochi::articulated::rigid::EntityPreStep, reg, arE);

  // Both should result in the same pose and velocity
  auto& rPrevPose = reg.get<CRigidState<TimeStep::Previous>>(rE).value;
  auto& rPrevVel = reg.get<CRigidVel<TimeStep::Previous>>(rE);
  auto& arPrevPose = reg.get<CRigidState<TimeStep::Previous>>(arE).value;
  auto& arPrevVel = reg.get<CRigidVel<TimeStep::Previous>>(arE);

  EXPECT_EQ(rPrevPose.GetTranslation(), rbPose.GetTranslation());
  EXPECT_EQ(rPrevPose.GetTranslation(), arPrevPose.GetTranslation());
  EXPECT_EQ(rPrevPose.GetRotation(), rbPose.GetRotation());
  EXPECT_EQ(rPrevPose.GetRotation(), arPrevPose.GetRotation());
  EXPECT_EQ(rPrevVel.value.GetVCom(), Vec4r{});
  EXPECT_EQ(rPrevVel.value.GetVCom(), arPrevVel.value.GetVCom());
  EXPECT_EQ(rPrevVel.value.GetOmegaAndVSym().first, Vec4r{});
  EXPECT_EQ(rPrevVel.value.GetOmegaAndVSym().first, arPrevVel.value.GetOmegaAndVSym().first);
}

/**************************************************************************************
  ArticulatedBodyTest - Test fixture used to test functionality of the articulated body
**************************************************************************************/
class ArticulatedBodyTest : public test::MochiSceneTestBase {
 public:
  ShapeHandle _cubeShape;
  entt::entity _eAC;
  std::vector<real> _expectedReducedDofsData;
  std::vector<real> _expectedFullDofsData;

  std::vector<Real3> _jointAxes;
  RestTransformArray _restTransforms;

  void CreateRigidActors() {
    // Create 4 rigid actors
    RigidActorParams rParams{.shape = _cubeShape, .colliderType = ColliderType::Box};

    _scene->CreateRigidActor(rParams, test::ExpectOK{});
    rParams.worldFromLocal.SetTranslation(Real3{1_r, 0_r, 0_r});
    _scene->CreateRigidActor(rParams, test::ExpectOK{});
    rParams.worldFromLocal.SetTranslation(Real3{2_r, 0_r, 0_r});
    _scene->CreateRigidActor(rParams, test::ExpectOK{});
    rParams.worldFromLocal.SetTranslation(Real3{3_r, 0_r, 0_r});
    _scene->CreateRigidActor(rParams, test::ExpectOK{});
  }

  entt::entity CreateArticulatedBody() {
    TransformRT const offset{Real3{1_r, 0_r, 0_r}};

    ArticulatedActorParams params;
    params.name = "myArticulation";
    params.joints = {
        {.name = "myFreeJoint", .type = ArticulatedJointType::Free},
        {.name = "mySphericalJoint",
         .type = ArticulatedJointType::Spherical,
         .parentLinkFromJoint = offset},
        {.name = "myRevoluteJoint",
         .type = ArticulatedJointType::Revolute,
         .parentLinkFromJoint = offset,
         .axis = Real3{0_r, 0_r, 1_r}},
        {.name = "myPrismaticJoint",
         .type = ArticulatedJointType::Prismatic,
         .parentLinkFromJoint = offset,
         .axis = Real3{1_r, 0_r, 0_r}}};
    params.links = {
        {.name = "myLink0",
         .parentLink = -1,
         .shape = _cubeShape,
         .colliderType = ColliderType::Box},
        {.name = "myLink1",
         .parentLink = 0,
         .shape = _cubeShape,
         .colliderType = ColliderType::Box},
        {.name = "myLink2",
         .parentLink = 1,
         .shape = _cubeShape,
         .colliderType = ColliderType::Box},
        {.name = "myLink3",
         .parentLink = 2,
         .shape = _cubeShape,
         .colliderType = ColliderType::Box}};
    auto* ac = _scene->CreateArticulatedActor(params, test::ExpectOK{});
    auto eAC = GetEntity(ac->GetHandle());

    // Get joint transforms
    entt::registry const& reg = GetRegistry();
    _restTransforms = reg.get<CArticulatedRestTransforms const>(eAC);

    // Get joint axes
    auto const shapeInfo = ac->GetArticulatedShapeInfo(test::ExpectOK{});
    _jointAxes.assign(shapeInfo.jointAxes.begin(), shapeInfo.jointAxes.end());

    // Check the actor names
    EXPECT_STREQ("myArticulation", ac->GetName());
    auto linkActors = ac->GetNestedLinkActors(test::ExpectOK{});
    EXPECT_EQ(4, isize(linkActors));
    EXPECT_STREQ("myArticulation/myLink0", _scene->GetActor(linkActors[0])->GetName());
    EXPECT_STREQ("myArticulation/myLink1", _scene->GetActor(linkActors[1])->GetName());
    EXPECT_STREQ("myArticulation/myLink2", _scene->GetActor(linkActors[2])->GetName());
    EXPECT_STREQ("myArticulation/myLink3", _scene->GetActor(linkActors[3])->GetName());

    return eAC;
  }

  void SetUp() override {
    test::MochiSceneTestBase::SetUp(); // call down
    _scene->SetGravity(kDefaultGravity);

    auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
    _cubeShape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});

    CreateRigidActors();
    _eAC = CreateArticulatedBody();

    // Define reduced dofs
    real k20 = real(20_r * kPI / 180_r);
    Real3 p0 = Real3{0_r, 1_r, 2_r};
    Quaternion q0 = Quaternion::FromRotationVector(Real3{});
    Quaternion q1 = Quaternion::FromRotationVector(Real3{0.1_r, 0.2_r, 0.3_r});
    real angle2 = k20;
    real t3 = 0.5_r;

    // Define joint transforms
    TransformRT txJ0 = TransformRT(q0, p0);
    TransformRT txJ1 = TransformRT(q1, Real3{});
    TransformRT txJ2 = TransformRT(Quaternion::FromRotationVector(angle2 * _jointAxes[2]), Real3{});
    TransformRT txJ3 = TransformRT(Quaternion::Identity(), t3 * _jointAxes[3]);
    std::vector<TransformRT> txJs = {txJ0, txJ1, txJ2, txJ3};

    // Define reduced solution
    _expectedReducedDofsData.resize(RigidSize::kAll + RigidSize::kRot + 1 + 1);
    TransformToRawPose(
        txJ0, AsView(_expectedReducedDofsData).TopRows<RigidSize::kAll>(RigidSize::kAll));
    AsView(_expectedReducedDofsData).Slice<RigidSize::kRot>(RigidSize::kAll, RigidSize::kRot) =
        AsColumnVectorView(q1.data);
    _expectedReducedDofsData[RigidSize::kAll + RigidSize::kRot] = angle2;
    _expectedReducedDofsData[RigidSize::kAll + RigidSize::kRot + 1] = t3;

    // Compute world from bone
    std::vector<TransformRT> txWs(4);
    txWs[0] = _restTransforms[0].parentFromOuter * txJs[0] * _restTransforms[0].innerFromBone;
    for (int i = 1; i < 4; ++i) {
      txWs[i] = txWs[i - 1] * _restTransforms[i].parentFromOuter * txJs[i] *
          _restTransforms[i].innerFromBone;
    }

    // Define expected full space data
    _expectedFullDofsData.resize(4 * RigidSize::kAll);
    for (int i = 0, offset = 0; i < 4; ++i, offset += RigidSize::kAll) {
      auto fullDofs = AsView(_expectedFullDofsData).Slice<RigidSize::kAll>(offset, RigidSize::kAll);
      TransformToRawPose(txWs[i], fullDofs);
    }
  }
};

/**************************************************************************************
  Tests for articulated body
**************************************************************************************/
TEST_F(ArticulatedBodyTest, Articulated_Initialization) {
  // Get registry
  entt::registry& reg = GetRegistry();

  // Collect components
  auto const& groupMembers = reg.get<CGroupMembers const>(_eAC);
  auto const& fullDofs = reg.get<CArticulatedFullPose const>(_eAC);

  // Compose full dofs from rigid actor states
  std::vector<real> expectedFullDofs(groupMembers.actors.size() * RigidSize::kAll);
  int offset = 0;
  for (auto aE : groupMembers.actors) {
    auto const& rbState = reg.get<CRigidState<TimeStep::Current> const>(aE).value;
    TransformToRawPose(
        rbState, AsView(expectedFullDofs).Slice<RigidSize::kAll>(offset, RigidSize::kAll));
    offset += RigidSize::kAll;
  }

  // Compare full dofs after initialization
  EXPECT_SPAN_EQ(fullDofs.value.GetConstSpan(), MakeSpan(expectedFullDofs));
}

TEST_F(ArticulatedBodyTest, Articulated_EntitySetSolution) {
  // Get registry
  entt::registry& reg = GetRegistry();

  // Collect components
  auto const& bodyShape = reg.get<CArticulatedBodyShape const>(_eAC);
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(_eAC);
  auto const& parents = reg.get<CArticulatedParents const>(_eAC);
  auto const& restTransforms = reg.get<CArticulatedRestTransforms const>(_eAC);
  auto const& rootTransform = reg.get<CRootTransform const>(_eAC);
  auto& reducedPose = reg.get<CArticulatedReducedPose<TimeStep::Current>>(_eAC);
  auto& jointTransforms = reg.get<CArticulatedJointTransforms<TimeStep::Current>>(_eAC);
  auto& linkTransforms = reg.get<CArticulatedLinkTransforms<TimeStep::Current>>(_eAC);
  auto& fullDofs = reg.get<CArticulatedFullPose>(_eAC);

  // Call EntitySetSolution
  Span<real> expectedReducedSolSpan = MakeSpan(_expectedReducedDofsData);
  mochi::articulated::compound::EntitySetSolution(
      AsConstView(expectedReducedSolSpan),
      {},
      bodyShape,
      poseInfo,
      parents,
      restTransforms,
      rootTransform,
      reducedPose,
      jointTransforms,
      linkTransforms,
      fullDofs);

  // Compare reduced dofs after EntitySetSolution
  EXPECT_TRUE(test::NearEqualSpan(reducedPose.value.GetConstSpan(), expectedReducedSolSpan));

  // Compare full dofs after EntitySetSolution
  EXPECT_TRUE(test::NearEqualSpan(fullDofs.value.GetConstSpan(), MakeSpan(_expectedFullDofsData)));
}

TEST_F(ArticulatedBodyTest, Articulated_EntityGetSolution) {
  // Get registry
  entt::registry& reg = GetRegistry();

  // Collect components
  auto& reducedPose = reg.get<CArticulatedReducedPose<TimeStep::Current>>(_eAC);
  Span<real> reducedPoseSpan = reducedPose.value.GetSpan();
  std::copy(
      _expectedReducedDofsData.begin(), _expectedReducedDofsData.end(), reducedPoseSpan.begin());

  // Call EntityGetSolution
  std::vector<real> solData(_expectedReducedDofsData.size());
  Span<real> solSpan = MakeSpan(solData);
  mochi::articulated::compound::EntityGetSolution(AsView(solSpan), {}, reducedPose);

  // Compare reduced dofs after EntityGetSolution
  EXPECT_SPAN_EQ(solSpan, reducedPoseSpan);
}

TEST_F(ArticulatedBodyTest, Articulated_EntityPostNewSolution) {
  // Get registry
  entt::registry& reg = GetRegistry();

  // Collect components
  auto& articulatedProps = reg.get<CArticulatedProps>(_eAC);

  // Create problem object
  ColumnVector<real> solution(articulatedProps.reducedPoseDim);
  Span solSpan = solution.GetSpan();
  memcpy(
      solSpan.data(),
      _expectedReducedDofsData.data(),
      sizeof(real) * articulatedProps.reducedPoseDim);
  SnleProblem<real> problem;
  problem.solution.Reset(solution);

  // Call EntityPostNewSolution
  ecs::InvokeOnEntity(
      mochi::articulated::compound::EntityPostNewSolution,
      reg,
      _eAC,
      AsConstView(problem.solution));

  // Compare reduced dofs after EntityPostNewSolution
  auto& reducedPose = reg.get<CArticulatedReducedPose<TimeStep::Current>>(_eAC);
  Span<real> expectedReducedSolSpan = MakeSpan(_expectedReducedDofsData);
  Span<real const> reducedSolSpan = reducedPose.value.GetConstSpan();
  EXPECT_SPAN_EQ(reducedSolSpan, expectedReducedSolSpan);

  // Compare full dofs after EntityPostNewSolution
  auto& fullDofs = reg.get<CArticulatedFullPose>(_eAC);
  Span<real> expectedFullSolSpan = MakeSpan(_expectedFullDofsData);
  Span<real const> fullDofsSpan = fullDofs.value.GetConstSpan();
  EXPECT_TRUE(test::NearEqualSpan(fullDofsSpan, expectedFullSolSpan));
}

// Test whether the data in CArticulatedLinkTransforms and the per-link rigid state are in sync.
TEST_F(ArticulatedBodyTest, Articulated_LinkTransformsInSync) {
  auto runChecks = [&]() {
    entt::registry& reg = GetRegistry();
    auto const& skeletonPoseCurr =
        reg.get<CArticulatedLinkTransforms<TimeStep::Current> const>(_eAC);
    auto const& groupMembers = reg.get<CGroupMembers const>(_eAC);
    for (int i = 0; i < isize(groupMembers.actors); ++i) {
      entt::entity aE = groupMembers.actors[i];
      auto const& rbState = reg.get<CRigidState<TimeStep::Current> const>(aE).value;
      EXPECT_EQ(skeletonPoseCurr[i].GetRotation(), rbState.GetRotation());
      EXPECT_EQ(skeletonPoseCurr[i].GetTranslation(), rbState.GetTranslation());
    }
  };

  // Compare at initialization.
  runChecks();

  // Compare after a step.
  _scene->Step(1e-2_r);
  runChecks();
}

// Guards link-authored pose projection from republishing off-manifold transforms.
TEST_F(ArticulatedBodyTest, SetPoseFromLinksPublishesCanonicalPose) {
  auto* actor = _scene->GetActor(GetActorHandle(_eAC, _scene->GetHandle()));
  ASSERT_NE(nullptr, actor);

  DynamicArray<TransformRT> baseline(4);
  actor->GetArticulatedLinkTransforms(baseline, test::ExpectOK{});

  int constexpr kRevoluteLink = 2;
  int constexpr kParentLink = 1;
  DynamicArray<TransformRT> requested = baseline;
  Real3 const worldAxis = baseline[kParentLink].GetRotation() * _jointAxes[kRevoluteLink];
  requested[kRevoluteLink].SetRotation(
      Quaternion::FromRotationVector(0.25_r * worldAxis) * requested[kRevoluteLink].GetRotation());
  requested[kRevoluteLink].SetTranslation(
      requested[kRevoluteLink].GetTranslation() + Real3{0_r, 2_r, 0_r});

  actor->SetArticulatedPoseFromLinks(requested, test::ExpectOK{});

  DynamicArray<TransformRT> canonical(baseline.size());
  actor->GetArticulatedLinkTransforms(canonical, test::ExpectOK{});
  EXPECT_FALSE(NearEqual(canonical[kRevoluteLink], requested[kRevoluteLink], kTolerance));
  EXPECT_FALSE(NearEqual(canonical[kRevoluteLink], baseline[kRevoluteLink], kTolerance));

  DynamicArray<real> pose(actor->GetNumDofs());
  actor->GetArticulatedPose(pose, test::ExpectOK{});
  actor->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});

  DynamicArray<TransformRT> replayed(baseline.size());
  actor->GetArticulatedLinkTransforms(replayed, test::ExpectOK{});
  for (int i = 0; i < isize(canonical); ++i) {
    EXPECT_TRUE(NearEqual(canonical[i], replayed[i], kTolerance));
  }
}

static void Compare(MatrixView<real const> a, MatrixView<real const> b, real tol = 1e-2_r) {
  real normMax = std::max(a.Norm(), b.Norm());
  if (normMax > 1e-6_r) {
    real normDiff = Matrix<real>(a - b).Norm();
    EXPECT_NEAR(normDiff / normMax, 0_r, tol);
  }
}

/**************************************************************************************
  ArticulatedBodyDynamicsTest - Test fixture for articulated body dynamics (velocities and forces)
**************************************************************************************/

class ArticulatedBodyDynamicsTest : public test::MochiSceneTestBase {
 protected:
  ShapeHandle _cubeShape;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp(); // call down
    _scene->SetGravity(kDefaultGravity);
    _scene->EnableLayerContactSymmetric("Articulated", "Articulated", false, test::ExpectOK{});

    auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
    _cubeShape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});
  }

  Actor* CreateActor(
      Span<ArticulatedJointType const> joints,
      real dampingCoeff,
      real inertiaCoeff,
      real frictionCoeff) {
    auto const numBones = isize(joints);
    TransformRT const offset{Real3{1_r, 1_r, 1_r}};

    // Build per-joint friction params (if any)
    ArticulatedJointFrictionParams fp{};
    bool const hasFriction = (dampingCoeff > 0 || frictionCoeff > 0);
    if (dampingCoeff > 0) {
      fp.viscous = dampingCoeff;
    }
    if (frictionCoeff > 0) {
      fp.coulomb = frictionCoeff;
    }

    // Build joints and links
    ArticulatedActorParams params;
    params.joints.resize(numBones);
    params.links.resize(numBones);
    for (int i = 0; i < numBones; ++i) {
      bool const isFreeJoint = (joints[i] == ArticulatedJointType::Free);

      auto& joint = params.joints[i];
      joint.type = joints[i];
      joint.axis = Real3{1_r, 0_r, 0_r};
      // The first link is offset from the root; subsequent links are co-located.
      if (i == 0) {
        joint.parentLinkFromJoint = offset;
      }
      // Broadcast inertia/friction to non-free joints (matches InitJointParams behavior).
      if (inertiaCoeff > 0 && !isFreeJoint) {
        joint.inertia = inertiaCoeff;
      }
      if (hasFriction && !isFreeJoint) {
        joint.friction = fp;
      }

      auto& link = params.links[i];
      link.parentLink = i - 1;
      link.shape = _cubeShape;
      link.layer = "Articulated";
      link.colliderType = ColliderType::Box;
    }
    return _scene->CreateArticulatedActor(params, test::ExpectOK{});
  }

  void TestGetVelocity(Span<ArticulatedJointType const> joints) {
    // Rationale of the test:
    // 1. Set an initial state.
    // 2. Set a target state via Dirichlet BCs.
    // 3. Define the expected velocity by finite differences.
    // 4. Run a step.
    // 5. Get the velocity and compare to the expected velocity.
    // Error should be very small.

    auto generator = mochi::RandomGenerator(20);
    real constexpr kDt = 1e-2_r;

    // Create articulated-body actor
    Actor* actor = CreateActor(joints, 0_r, 0_r, 0_r);
    auto const numDofs = actor->GetNumDofs();
    std::vector<int> inds(numDofs);
    std::iota(inds.begin(), inds.end(), 0);

    // Define random velocity and initial state
    std::vector<real> previous(numDofs);
    std::vector<real> vel(numDofs);
    SetRandom(generator, -1_r, 1_r, MakeSpan(previous));
    SetRandom(generator, -1_r, 1_r, MakeSpan(vel));

    // Set initial state
    actor->SetArticulatedPoseFromJoints(previous, test::ExpectOK{});

    // Compute target state from previous state and velocity
    std::vector<real> delta(numDofs);
    AsView(delta) = kDt * AsConstView(vel);
    std::vector<real> target(numDofs);
    actor->AddArticulatedDeltaToPose(previous, delta, target, test::ExpectOK{});

    // Check that we obtain the same delta when computing the difference
    std::vector<real> deltaTest(numDofs);
    actor->ComputeArticulatedPoseDelta(previous, target, deltaTest, test::ExpectOK{});
    Compare(AsConstView(delta), AsConstView(deltaTest), 1.25e-5_r);

    // Set target state as Dirichlet BCs and run one step
    actor->AddBoundaryConditionDofsWorld(inds, target, test::ExpectOK{});
    _scene->Step(kDt);

    // Get velocity and test
    std::vector<real> velTest(numDofs);
    actor->GetArticulatedJointVelocities(velTest, test::ExpectOK{});
    Compare(AsConstView(vel), AsConstView(velTest), 1e-3_r);

    _scene->DestroyActor(actor->GetHandle());
  }

  void TestSetVelocity(Span<ArticulatedJointType const> joints) {
    // Rationale of the test:
    // 1. Set an initial state.
    // 2. Set an initial velocity.
    // 3. Run a small step.
    // 4. Get the current velocity and compare to the initial velocity.
    // There should be some error due to acceleration.

    auto generator = mochi::RandomGenerator(20);
    real constexpr kDt = 1e-3_r;

    // Create articulated-body actor
    Actor* actor = CreateActor(joints, 0_r, 0_r, 0_r);
    auto const numDofs = actor->GetNumDofs();
    std::vector<int> inds(numDofs);
    std::iota(inds.begin(), inds.end(), 0);

    // Define random velocity and initial state
    std::vector<real> pose(numDofs);
    std::vector<real> vel(numDofs);
    SetRandom(generator, -1_r, 1_r, MakeSpan(pose));
    SetRandom(generator, -1_r, 1_r, MakeSpan(vel));

    // Set initial state and velocity; run one step
    actor->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});
    actor->SetArticulatedJointVelocities(vel, test::ExpectOK{});
    _scene->Step(kDt);

    // Get velocity and test
    std::vector<real> velTest(numDofs);
    actor->GetArticulatedJointVelocities(velTest, test::ExpectOK{});
    real normVel = AsConstView(vel).Norm();
    real normVelTest = AsConstView(velTest).Norm();
    ColumnVector<real> diff = AsConstView(vel) - AsConstView(velTest);
    real normDiff = diff.Norm();
    EXPECT_NEAR(normDiff / std::max(normVel, normVelTest), 0_r, 1.1e-2_r);

    _scene->DestroyActor(actor->GetHandle());
  }

  template <typename SystemT, typename ActorSetupCallbackT = std::nullptr_t>
  void TestJointForces(
      SystemT system,
      Span<ArticulatedJointType const> joints,
      bool testHessian,
      bool useNewtonEulerInertia,
      real residualTol = 1e-2_r,
      ActorSetupCallbackT actorSetupCallback = nullptr) {
    auto generator = mochi::RandomGenerator(20);
    real constexpr kNearWrapAround = 0.575_r * kPI;
    real constexpr kVel = 10_r;
    real constexpr kEps = 1e-3_r;
    real constexpr kOneOverTwoEps = 1_r / (2_r * kEps);

    AssemblyParams const paramsObjAndRes{.assemObj = true, .assemRes = true};
    AssemblyParams const paramsAll{
        .assemObj = true,
        .assemRes = true,
        .assemDRes = true,
        .psdDRes = false,
        .fittedSaturationHessian = SaturationHessianParams::All(false)};

    // Validate that kNearWrapAround does not produce wrap around
    Vec4r nearWrapAround{kNearWrapAround};
    EXPECT_EQ(nearWrapAround, RotVectorPiCap(kNearWrapAround));

    // Validate that the default time step produces wrap-around on some rotation vector
    real constexpr kDt = CSceneTime::kDefaultTimeStep;
    Vec4r wrapAround = nearWrapAround + Vec4r{kVel * kDt};
    EXPECT_NE(wrapAround, RotVectorPiCap(wrapAround));

    // Create articulated-body actor
    real constexpr kDamping = 1_r;
    real constexpr kInertia = 1_r;
    real constexpr kFriction = 2_r;
    Actor* actor = CreateActor(joints, kDamping, kInertia, kFriction);
    auto entity = GetEntity(actor->GetHandle());
    auto& registry = GetRegistry();

    // Add external forces
    auto const numDofs = actor->GetNumDofs();
    std::vector<int> dofIndices(numDofs);
    std::iota(dofIndices.begin(), dofIndices.end(), 0);
    std::vector<real> forceValues = RandomVector(generator, numDofs);
    actor->SetExternalForcesOnDofs(dofIndices, forceValues, test::ExpectOK{});

    // Enable/disable Newton-Euler inertia
    experimental::EnableNewtonEulerInertia(actor, useNewtonEulerInertia, test::ExpectOK{});

    // Call optional actor setup callback (e.g., to add transmissions)
    if constexpr (!std::is_same_v<ActorSetupCallbackT, std::nullptr_t>) {
      actorSetupCallback(actor);
    }

    // Fetch relevant components.
    auto& actorSnle = registry.get<CActorSnle>(entity);
    EXPECT_TRUE(actorSnle.UseReduced());
    MOCHI_ASSERT(
        std::holds_alternative<Matrix<real>>(actorSnle.reducedDResidual),
        "Expected dense storage.");
    auto const* jointsData =
        registry.get<CArticulatedBodyShape const>(entity).shape->GetJointsData();

    // Generate test poses: zero, random, near wrap-around.
    std::array<std::vector<real>, 3> testPoses = {
        {std::vector<real>(numDofs, 0_r),
         RandomVector(generator, numDofs),
         std::vector<real>(numDofs, kNearWrapAround)}};

    // Generate a random velocity
    std::vector<real> testVel = RandomVector(generator, numDofs);

    // Setup time integrator.
    registry.get<CTimeIntegratorState>(entity).dtStage = kDt;

    // Delta and pose vectors
    ColumnVector<real> delta(numDofs);
    delta.SetZero();
    ColumnVector<real> perturbedPose(numDofs);

    // Perform a consistency test - Ensure the approximate Hessians agree with their
    // analytical counterparts.
    for (auto const& testPose : testPoses) {
      // Set the desired pose and velocity
      actor->SetArticulatedPoseFromJoints(testPose, test::ExpectOK{});
      actor->SetArticulatedJointVelocities(testVel, test::ExpectOK{});

      // Simulate a step, which produces a small deviation between the current and stage-start
      // poses.
      _scene->Step(kDt);
      std::vector<real> refPose(testPose.size());
      actor->GetArticulatedPose(refPose, test::ExpectOK{});

      // Evaluate gradient and Hessian at current state.
      actorSnle.objective = 0.0;
      actorSnle.reducedResidual.SetZero();
      std::get<Matrix<real>>(actorSnle.reducedDResidual).SetZero();
      ecs::InvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
          system, registry, entity, std::cref(paramsAll));
      [[maybe_unused]] double obj = actorSnle.objective;
      ColumnVector<real> res = actorSnle.reducedResidual.Duplicate();
      Matrix<real> const dres = std::get<Matrix<real>>(actorSnle.reducedDResidual).Duplicate();

      // Approximate gradient and Hessian with central finite differences.
      ColumnVector<real> resFD = res.Duplicate();
      Matrix<real> dresFD = dres.Duplicate();

      auto evaluate = [&]() {
        actor->AddArticulatedDeltaToPose(refPose, delta, perturbedPose, test::ExpectOK{});
        actor->SetArticulatedPoseFromJoints(perturbedPose, test::ExpectOK{});
        actorSnle.objective = 0.0;
        actorSnle.reducedResidual.SetZero();
        ecs::InvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
            system, registry, entity, std::cref(paramsObjAndRes));
      };

      for (auto i = 0; i < numDofs; ++i) {
        delta[i] = kEps;
        evaluate();
        auto objFwd = actorSnle.objective;
        auto resFwd = actorSnle.reducedResidual.Duplicate();
        articulated::TransportInputOfLieJacobian(
            jointsData->jointTypes, jointsData->dofInfo, delta, resFwd.Transpose());

        delta[i] = -kEps;
        evaluate();
        auto objBwd = actorSnle.objective;
        auto resBwd = actorSnle.reducedResidual.Duplicate();
        articulated::TransportInputOfLieJacobian(
            jointsData->jointTypes, jointsData->dofInfo, delta, resBwd.Transpose());

        resFD(i) = kOneOverTwoEps * (objFwd - objBwd);
        dresFD.Col(i) = kOneOverTwoEps * (resFwd - resBwd);
        delta[i] = 0_r;
      }

      // Compare approximate and analytical gradient and Hessian.
      Compare(resFD, res, residualTol);
      if (testHessian) {
        // For external forces the analytical Hessian is zero, so there's no point in comparing to
        // finite differences.
        Compare(dresFD, dres);
      }
    }

    _scene->DestroyActor(actor->GetHandle());
  }
};

/**************************************************************************************
  Tests for articulated body dynamics (velocities and forces)
**************************************************************************************/

namespace {
std::array<std::vector<ArticulatedJointType>, 6> kBodies = {
    {{ArticulatedJointType::Spherical},
     {ArticulatedJointType::Free, ArticulatedJointType::Revolute},
     {ArticulatedJointType::Revolute, ArticulatedJointType::Spherical},
     {ArticulatedJointType::Free, ArticulatedJointType::Spherical, ArticulatedJointType::Revolute},
     {ArticulatedJointType::Free, ArticulatedJointType::Prismatic, ArticulatedJointType::Prismatic},
     {ArticulatedJointType::Free,
      ArticulatedJointType::Spherical,
      ArticulatedJointType::Revolute,
      ArticulatedJointType::Spherical}}};
}

TEST_F(ArticulatedBodyDynamicsTest, GetArticulatedJointVelocities) {
  for (auto const& body : kBodies) {
    TestGetVelocity(body);
  }
}

TEST_F(ArticulatedBodyDynamicsTest, SetArticulatedJointVelocities) {
  for (auto const& body : kBodies) {
    TestSetVelocity(body);
  }
}

TEST_F(ArticulatedBodyDynamicsTest, DampingForces) {
  for (auto const& body : kBodies) {
    TestJointForces(
        articulated::compound::AssembleDampingForces,
        body,
        /* testHessian */ true,
        /* useNewtonEulerInertia */ false);
  }
}

TEST_F(ArticulatedBodyDynamicsTest, InertiaForces) {
  for (auto const& body : kBodies) {
    TestJointForces(
        articulated::compound::AssembleInertiaForces,
        body,
        /* testHessian */ true,
        /* useNewtonEulerInertia */ false);
  }
}

TEST_F(ArticulatedBodyDynamicsTest, InertiaForcesNewtonEuler) {
  for (auto const& body : kBodies) {
    TestJointForces(
        articulated::compound::AssembleInertiaForces,
        body,
        /* testHessian */ false,
        /* useNewtonEulerInertia */ true,
        /* residualTol */ 3e-2_r);
  }
}

TEST_IF_F(MOCHI_USE_DOUBLE_PRECISION, ArticulatedBodyDynamicsTest, FrictionForces) {
  for (auto const& body : kBodies) {
    TestJointForces(
        articulated::compound::AssembleFrictionForces,
        body,
        /* testHessian */ true,
        /* useNewtonEulerInertia */ false);
  }
}

TEST_F(ArticulatedBodyDynamicsTest, ExternalForces) {
  for (auto const& body : kBodies) {
    TestJointForces(
        articulated::compound::AssembleExternalForces,
        body,
        /* testHessian */ false,
        /* useNewtonEulerInertia */ false);
  }
}

TEST_F(ArticulatedBodyDynamicsTest, TransmissionForces) {
  // Test transmission forces on a body with revolute and prismatic joints that can have
  // transmissions
  std::vector<ArticulatedJointType> body = {
      ArticulatedJointType::Free, ArticulatedJointType::Revolute, ArticulatedJointType::Prismatic};

  // Define callback to add transmission to actor
  auto addTransmission = [](Actor* actor) {
    // Add a transmission connecting joints 1 and 2
    experimental::LinearTransmissionParams transmissionParams;
    transmissionParams.jointIndices = {1, 2};
    transmissionParams.jointCoefficients = {0.5_r, -0.3_r};

    int transmissionIndex =
        experimental::AddLinearTransmission(actor, transmissionParams, test::ExpectOK{});
    EXPECT_GE(transmissionIndex, 0);

    // Attach a displacement control actuator
    experimental::DisplacementControlActuatorParams actuatorParams;
    actuatorParams.targetDisplacement = 0.1_r;
    actuatorParams.stiffness = 1e3_r;
    actuatorParams.damping = 10_r;
    experimental::AttachDisplacementControlActuator(
        actor, transmissionIndex, actuatorParams, test::ExpectOK{});
  };

  // Use TestJointForces to verify consistency via finite differences
  TestJointForces(
      articulated::compound::AssembleTransmissionForces,
      body,
      /* testHessian */ true,
      /* useNewtonEulerInertia */ false,
      /* residualTol */ 1e-2_r,
      /* actorSetupCallback */ addTransmission);
}

/**************************************************************************************
  ArticulatedActorApiTest - Test fixture for API queries on articulated actors
**************************************************************************************/

class ArticulatedActorApiTest : public test::MochiSceneTestBase {
 protected:
  Actor* _actor = nullptr;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp(); // call down

    // Create a scene
    auto rootPath = test::GetAssetPath("");
    auto prefabPath = test::GetAssetPath("benchmarks/half_cheetah/half_cheetah.mochi_scene");
    prefab::AddToScene(prefabPath, rootPath, _scene, {}, test::ExpectOK{});

    // Get the articulated actor
    std::vector<Actor*> actors(_scene->GetNumActors());
    _scene->GetActors(actors, test::ExpectOK{});
    _actor = actors[0];
    EXPECT_EQ(_actor->GetType(), ActorType::Articulated);
  }

  void AddGround() {
    RigidActorParams groundParams;
    groundParams.name = "Ground";
    Real3 normal = Real3{0_r, 1_r, 0_r};
    normal = Normalize(normal);
    groundParams.shape = _mochiContext->CreatePlaneShape(normal, 0_r, test::ExpectOK{});
    groundParams.isStatic = true;
    groundParams.colliderType = ColliderType::Plane;
    _scene->CreateRigidActor(groundParams, test::ExpectOK{});
  }

  void UpdateTrackingParams(std::optional<real> stiffness, std::optional<real> damping) {
    auto const numLinks = _actor->GetNestedLinkActors(test::ExpectOK{}).size();
    PoseControllerParams params(static_cast<int>(numLinks));
    _actor->GetArticulatedPoseControllerParams(params, test::ExpectOK{});
    auto const jointTypes = _actor->GetArticulatedShapeInfo(test::ExpectOK{}).jointTypes;
    for (size_t i = 0; i < numLinks; ++i) {
      if (stiffness) {
        params.linkPosTracking[i].stiffness = *stiffness;
        params.linkRotTracking[i].stiffness = *stiffness;
        params.jointTracking[i].stiffness =
            jointTypes[i] == ArticulatedJointType::Hard ? 0_r : *stiffness;
      }
      if (damping) {
        params.linkPosTracking[i].damping = *damping;
        params.linkRotTracking[i].damping = *damping;
        params.jointTracking[i].damping =
            jointTypes[i] == ArticulatedJointType::Hard ? 0_r : *damping;
      }
    }
    _actor->SetArticulatedPoseControllerParams(params, test::ExpectOK{});
  }

  void TestAddEmptyArticulatedPoseController() {
    PoseControllerParams params;
    _actor->RemoveArticulatedPoseController(test::ExpectOK{});
    _actor->AddArticulatedPoseController(params, test::ExpectOK{});

    auto const shapeInfo = _actor->GetArticulatedShapeInfo(test::ExpectOK{});
    auto const jointTypes = shapeInfo.jointTypes;
    int const numLinks = isize(shapeInfo.linkNames);
    int const numControllableJoints = static_cast<int>(
        std::count_if(jointTypes.begin(), jointTypes.end(), [](ArticulatedJointType type) {
          return type != ArticulatedJointType::Hard;
        }));
    auto const constraints = _actor->GetArticulatedPoseConstraints(test::ExpectOK{});
    EXPECT_EQ(2 * numLinks + numControllableJoints, isize(constraints));

    std::array<int, static_cast<int>(PoseConstraintType::Count)> typeCounts{};
    for (auto const& constraint : constraints) {
      ++typeCounts[static_cast<int>(constraint.type)];
    }
    EXPECT_EQ(numControllableJoints, typeCounts[static_cast<int>(PoseConstraintType::Joint)]);
    EXPECT_EQ(numLinks, typeCounts[static_cast<int>(PoseConstraintType::LinkTranslation)]);
    EXPECT_EQ(numLinks, typeCounts[static_cast<int>(PoseConstraintType::LinkRotation)]);

    // Empty controller params create all constraints with zero gains. Arrays are link-indexed and
    // sized to numLinks.
    PoseControllerParams trackingParams(numLinks);
    _actor->GetArticulatedPoseControllerParams(trackingParams, test::ExpectOK{});
    for (auto const* array :
         {&trackingParams.linkPosTracking,
          &trackingParams.linkRotTracking,
          &trackingParams.jointTracking}) {
      for (auto const& trackingParam : *array) {
        EXPECT_EQ(0_r, trackingParam.stiffness);
        EXPECT_EQ(0_r, trackingParam.damping);
      }
    }

    // Gains are settable. Joint slots for links without a controllable joint stay at zero gains.
    for (auto* array : {&trackingParams.linkPosTracking, &trackingParams.linkRotTracking}) {
      for (auto& trackingParam : *array) {
        trackingParam.stiffness = 1_r;
        trackingParam.damping = 2_r;
      }
    }
    for (int i = 0; i < isize(trackingParams.jointTracking); ++i) {
      if (jointTypes[i] != ArticulatedJointType::Hard) {
        trackingParams.jointTracking[i].stiffness = 1_r;
        trackingParams.jointTracking[i].damping = 2_r;
      }
    }
    _actor->SetArticulatedPoseControllerParams(trackingParams, test::ExpectOK{});

    PoseControllerParams updatedTrackingParams(numLinks);
    _actor->GetArticulatedPoseControllerParams(updatedTrackingParams, test::ExpectOK{});
    for (auto const* array :
         {&updatedTrackingParams.linkPosTracking, &updatedTrackingParams.linkRotTracking}) {
      for (auto const& trackingParam : *array) {
        EXPECT_EQ(1_r, trackingParam.stiffness);
        EXPECT_EQ(2_r, trackingParam.damping);
      }
    }
    int jointSetCount = 0;
    for (auto const& trackingParam : updatedTrackingParams.jointTracking) {
      if (trackingParam.stiffness != 0_r) {
        EXPECT_EQ(1_r, trackingParam.stiffness);
        EXPECT_EQ(2_r, trackingParam.damping);
        ++jointSetCount;
      }
    }
    EXPECT_EQ(numControllableJoints, jointSetCount);
  }
};

/**************************************************************************************
  Tests for the API of articulated actors
**************************************************************************************/

TEST_IF_F(MOCHI_HDF5_AND_INTERNAL, ArticulatedActorApiTest, GetArticulatedDofLimits) {
  auto const N = _actor->GetNumDofs();

  real constexpr kSentinel = std::numeric_limits<real>::quiet_NaN();
  std::vector<Real2> valueLimits(N, {kSentinel, kSentinel});
  _actor->GetArticulatedDofLimits(valueLimits, test::ExpectOK{});

  // Check that the limits are not empty and in the appropriate order.
  for (auto const& limit : valueLimits) {
    EXPECT_EQ(limit[0], limit[0]); // Checks for NaN
    EXPECT_EQ(limit[1], limit[1]); // Checks for NaN
    EXPECT_LE(limit[0], limit[1]); // min <= max
  }
}

TEST_IF_F(MOCHI_HDF5_AND_INTERNAL, ArticulatedActorApiTest, SetArticulatedPoseFromJoints) {
  auto const N = _actor->GetNumDofs();

  // Test setting all dofs
  std::vector<real> pose(N, 0.01_r);
  _actor->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});

  std::vector<real> poseTest(N);
  _actor->GetArticulatedPose(poseTest, test::ExpectOK{});
  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(poseTest[i], pose[i], kDefaultNearEqualEpsilon<real>);
  }
}

TEST_IF_F(MOCHI_HDF5_AND_INTERNAL, ArticulatedActorApiTest, GetArticulatedShapeInfo) {
  auto const info = _actor->GetArticulatedShapeInfo(test::ExpectOK{});

  // Check sizes
  int constexpr kNumJoints = 10;
  EXPECT_EQ(info.dofInfo.size(), kNumJoints);
  EXPECT_EQ(info.jointAxes.size(), kNumJoints);
  EXPECT_EQ(info.jointTypes.size(), kNumJoints);
  EXPECT_EQ(info.jointFromChildLink.size(), kNumJoints);
  EXPECT_EQ(info.jointNames.size(), kNumJoints);
  EXPECT_EQ(info.linkNames.size(), kNumJoints);
  EXPECT_EQ(info.parents.size(), kNumJoints);
  EXPECT_EQ(info.rootFromLinksAtRest.size(), kNumJoints);
  EXPECT_EQ(info.jointMinLimits.size(), kNumJoints);
  EXPECT_EQ(info.jointMaxLimits.size(), kNumJoints);

  // Check some values
  EXPECT_EQ(info.jointAxes[1], (Real3{0_r, 1_r, 0_r}));
  EXPECT_EQ(info.jointTypes[0], ArticulatedJointType::Prismatic);
  EXPECT_EQ(info.jointTypes[6], ArticulatedJointType::Revolute);
  EXPECT_EQ(info.jointTypes[9], ArticulatedJointType::Hard);
  EXPECT_EQ(info.jointFromChildLink[1], TransformRT{});
  EXPECT_EQ(info.jointNames[1], "RootZ");
  EXPECT_EQ(info.jointNames[5], "BackFoot");
  EXPECT_EQ(info.linkNames[2], "Torso");
  EXPECT_EQ(info.linkNames[7], "FrontShin");
  EXPECT_EQ(info.parents[0], -1);
  EXPECT_EQ(info.parents[2], 1);
  EXPECT_EQ(info.parents[6], 2);
  EXPECT_EQ(info.jointMinLimits[0], -kInf3);
  EXPECT_EQ(info.jointMaxLimits[0], kInf3);
  // In the shape file, the limits have the opposite sign, but it makes sense because the axis of
  // rotation is (0, 0, -1). Use EXPECT_NEAR_EQ to account for floating point errors.
  EXPECT_NEAR_EQ(info.jointMinLimits[4], (Real3{0_r, 0_r, 0.785_r}));
  EXPECT_NEAR_EQ(info.jointMaxLimits[4], (Real3{0_r, 0_r, -0.785_r}));
}

TEST_IF_F(MOCHI_HDF5_AND_INTERNAL, ArticulatedActorApiTest, GetArticulatedJointLayout) {
  auto const info = _actor->GetArticulatedShapeInfo(test::ExpectOK{});

  // Check size
  int constexpr kNumJoints = 10;
  EXPECT_EQ(info.jointTypes.size(), kNumJoints);

  // Check some values
  EXPECT_EQ(info.jointAxes[1], (Real3{0_r, 1_r, 0_r}));
  EXPECT_EQ(info.jointTypes[0], ArticulatedJointType::Prismatic);
  EXPECT_EQ(info.jointTypes[6], ArticulatedJointType::Revolute);
  EXPECT_EQ(info.jointTypes[9], ArticulatedJointType::Hard);
  EXPECT_EQ(info.parents[0], -1);
  EXPECT_EQ(info.parents[2], 1);
  EXPECT_EQ(info.parents[6], 2);
  EXPECT_EQ(info.dofInfo[0].offset, 0);
  EXPECT_EQ(info.dofInfo[0].GetTransOffset(), 0);
  EXPECT_EQ(info.dofInfo[0].GetRotOffset(), 1);
  EXPECT_EQ(info.dofInfo[0].GetSize(), 1);
  EXPECT_EQ(info.dofInfo[0].transSize, 1);
  EXPECT_EQ(info.dofInfo[0].rotSize, 0);
  EXPECT_EQ(info.dofInfo[6].offset, 6);
  EXPECT_EQ(info.dofInfo[6].GetTransOffset(), 6);
  EXPECT_EQ(info.dofInfo[6].GetRotOffset(), 6);
  EXPECT_EQ(info.dofInfo[6].GetSize(), 1);
  EXPECT_EQ(info.dofInfo[6].transSize, 0);
  EXPECT_EQ(info.dofInfo[6].rotSize, 1);
  EXPECT_EQ(info.dofInfo[9].offset, 9);
  EXPECT_EQ(info.dofInfo[9].GetTransOffset(), 9);
  EXPECT_EQ(info.dofInfo[9].GetRotOffset(), 9);
  EXPECT_EQ(info.dofInfo[9].GetSize(), 0);
  EXPECT_EQ(info.dofInfo[9].transSize, 0);
  EXPECT_EQ(info.dofInfo[9].rotSize, 0);
}

TEST_IF_F(MOCHI_HDF5_AND_INTERNAL, ArticulatedActorApiTest, SetGetArticulatedTargetPose) {
  // Extract number of dofs
  auto const N = _actor->GetNumDofs();

  // Extract current pose
  std::vector<real> initialPose(N);
  _actor->GetArticulatedPose(initialPose, test::ExpectOK{});

  // Compute new target pose by applying a small offset and set as current pose
  std::vector<real> targetPose = initialPose;
  real constexpr kDelta = 0.1_r;
  for (auto i = 0; i < N; ++i) {
    targetPose[i] += kDelta;
  }
  _actor->SetArticulatedTargetPose(targetPose, test::ExpectOK{});

  // Confirm that the target pose matches the one set
  std::vector<real> targetPoseTest(N);
  _actor->GetArticulatedTargetPose(targetPoseTest, test::ExpectOK{});
  EXPECT_TRUE(targetPoseTest == targetPose);

  // Extract current pose and ensure pose is the same (no simulation step yet)
  std::vector<real> poseAfterSet(N);
  _actor->GetArticulatedPose(poseAfterSet, test::ExpectOK{});

  EXPECT_TRUE(initialPose == poseAfterSet);

  // Run some simulation steps
  for (int i = 0; i < 100; ++i) {
    _scene->Step(1e-2_r);
  }

  // Confirm that the target is still the same
  _actor->GetArticulatedTargetPose(targetPoseTest, test::ExpectOK{});
  EXPECT_TRUE(targetPoseTest == targetPose);

  // Extract current pose
  std::vector<real> poseAfterRun(N);
  _actor->GetArticulatedPose(poseAfterRun, test::ExpectOK{});

  // Ensure pose is different from first one
  EXPECT_TRUE(initialPose != poseAfterRun);

  // Ensure pose is close enough to target pose.
  // Skip the first 3 dofs (root), because they're not controlled.
  real constexpr kTol = kDelta / 10_r;
  int constexpr kRoot = 3;
  for (auto i = kRoot; i < N; ++i) {
    EXPECT_NEAR(targetPose[i], poseAfterRun[i], kTol);
  }
}

TEST_IF_F(MOCHI_HDF5_AND_INTERNAL, ArticulatedActorApiTest, SetArticulatedTargetVelocity) {
  real constexpr kDt = 1e-2_r;

  // Set the controller damping to be very large
  UpdateTrackingParams({}, 1e6_r);

  // Set zero velocity on the agent
  auto const N = _actor->GetNumDofs();
  std::vector<real> vel(N);
  _actor->SetArticulatedJointVelocities(vel, test::ExpectOK{});

  // Initialize a random velocity for the controller
  auto generator = RandomGenerator(42);
  SetRandom(generator, -1_r, 1_r, MakeSpan(vel));
  _actor->SetArticulatedTargetVelocity(vel, test::ExpectOK{});

  // Simulate a step
  _scene->Step(kDt);

  // Get the velocity of the agent and compare it with the target velocity
  // Skip the first 3 dofs (root), because they're not controlled.
  std::vector<real> velTest(N);
  _actor->GetArticulatedJointVelocities(velTest, test::ExpectOK{});
  int constexpr kRoot = 3;
  for (auto i = kRoot; i < N; ++i) {
    EXPECT_NEAR(velTest[i], vel[i], 1e-2_r);
  }
}

TEST_IF_F(MOCHI_HDF5_AND_INTERNAL, ArticulatedActorApiTest, AddEmptyArticulatedPoseController) {
  TestAddEmptyArticulatedPoseController();
}

TEST_IF_F(
    MOCHI_HDF5_AND_INTERNAL,
    ArticulatedActorApiTest,
    AddEmptyArticulatedPoseControllerDifferentiable) {
  // Back-propagation requires backward Euler.
  test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
  MakeSceneDifferentiableInternal(_scene, test::ExpectOK{});
  TestAddEmptyArticulatedPoseController();
}

TEST_IF_F(MOCHI_HDF5_AND_INTERNAL, ArticulatedActorApiTest, SetArticulatedPoseControllerParams) {
  // Add a ground to prevent a simple free fall.
  AddGround();

  auto const N = _actor->GetNumDofs();
  std::vector<real> initialPose(N);
  std::vector<real> currentPose(N);
  _actor->GetArticulatedPose(initialPose, test::ExpectOK{});

  auto const M = _actor->GetArticulatedShapeInfo(test::ExpectOK{}).jointTypes.size();

  // Lambda to compute distances between poses. Skip the first 3 joints (root).
  auto IsPoseSimilarToCurrent = [&]() {
    std::vector<real> transDistances(M, 0_r);
    std::vector<real> rotDistances(M, 0_r);
    _actor->GetArticulatedPoseDistance(
        initialPose, currentPose, transDistances, rotDistances, test::ExpectOK{});
    int constexpr kRoot = 3;
    for (int i = 0; i < kRoot; ++i) {
      transDistances[i] = 0_r;
      rotDistances[i] = 0_r;
    }
    auto constexpr kMaxLinearDist = 2e-3_r;
    auto constexpr kMaxAngularDist = kRadiansPerDegree;
    return *MaxElement(transDistances.begin(), transDistances.end()) < kMaxLinearDist &&
        *MaxElement(rotDistances.begin(), rotDistances.end()) < kMaxAngularDist;
  };

  // Set stiff tracking params, set the initial pose as target and confirm it is (nearly) preserved
  // after some iterations.
  UpdateTrackingParams(1e6_r, {});
  _actor->SetArticulatedTargetPose(initialPose, test::ExpectOK{});
  for (int i = 0; i < 20; ++i) {
    _scene->Step(1e-2_r);
  }
  _actor->GetArticulatedPose(currentPose, test::ExpectOK{});
  EXPECT_TRUE(IsPoseSimilarToCurrent());

  // Set the tracking params to zero and confirm the pose is no longer preserved.
  UpdateTrackingParams(0_r, 0_r);
  for (int i = 0; i < 20; ++i) {
    _scene->Step(1e-2_r);
  }

  _actor->GetArticulatedPose(currentPose, test::ExpectOK{});
  EXPECT_FALSE(IsPoseSimilarToCurrent());
}

TEST_IF_F(MOCHI_HDF5_AND_INTERNAL, ArticulatedActorApiTest, GetArticulatedControllerForce) {
  real constexpr kDt = 1e-1_r;

  // Add a ground to prevent a simple free fall.
  AddGround();

  // Set stiff tracking params and set the initial pose as target
  UpdateTrackingParams(1e6_r, {});
  auto const N = _actor->GetNumDofs();
  std::vector<real> initialPose(N);
  _actor->GetArticulatedPose(initialPose, test::ExpectOK{});
  _actor->SetArticulatedTargetPose(initialPose, test::ExpectOK{});

  // Try to get the force, but confirm that the force is not available yet.
  EXPECT_TRUE(_actor->GetArticulatedControllerForce(test::ExpectNotOK{}).empty());

  // Enable force query, but confirm that the force is not available yet.
  auto query = _actor->RegisterQuery(QueryType::ArticulatedControllerForce, test::ExpectOK{});
  EXPECT_TRUE(_actor->GetArticulatedControllerForce(test::ExpectNotOK{}).empty());

  // Simulate a step. Confirm that the force is now available.
  _scene->Step(kDt);
  auto forceValues = DynamicArray<real>{_actor->GetArticulatedControllerForce(test::ExpectOK{})};
  EXPECT_EQ(N, isize(forceValues));

  // Confirm the force is non-zero
  EXPECT_GT(AsConstView(forceValues).Norm(), 1e-3_r);

  // Disable force query. Confirm that the force is not available anymore.
  _actor->CancelQuery(query);
  EXPECT_TRUE(_actor->GetArticulatedControllerForce(test::ExpectNotOK{}).empty());
}

TEST_IF_F(
    MOCHI_HDF5_AND_INTERNAL,
    ArticulatedActorApiTest,
    SetArticulatedForceAndTargetPose_ValidConfigurations) {
  auto const N = _actor->GetNumDofs();
  // Test 1: Empty spans (valid - no action)
  {
    Span<real const> target;
    Span<ControlType const> controlTypes;
    Span<int const> dofs;
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectOK{});
  }

  // Test 2: Force only (no pose controller required)
  {
    std::array<real, 2> target = {1.0_r, 2.0_r};
    std::array<ControlType, 2> controlTypes = {ControlType::Force, ControlType::Force};
    std::array<int, 2> dofs = {0, 1};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectOK{});
  }

  // Test 3: Single force value
  {
    std::array<real, 1> target = {0.5_r};
    std::array<ControlType, 1> controlTypes = {ControlType::Force};
    std::array<int, 1> dofs = {0};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectOK{});
  }

  // Test 4: Forces on all dofs
  {
    DynamicArray<real> target(N, 0.1_r);
    DynamicArray<ControlType> controlTypes(N, ControlType::Force);
    DynamicArray<int> dofs(N);
    std::iota(dofs.begin(), dofs.end(), 0);
    Span<ControlType const> ctSpan(controlTypes.data(), isize(controlTypes));
    SetArticulatedForceAndTargetPose(_actor, target, ctSpan, dofs, test::ExpectOK{});
  }

  // Test 5: Force on last DOF (boundary test)
  {
    std::array<real, 1> target = {1.0_r};
    std::array<ControlType, 1> controlTypes = {ControlType::Force};
    std::array<int, 1> dofs = {N - 1};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectOK{});
  }

  // Test 6: With pose controller - pose targets only
  {
    std::array<real, 2> target = {1_r, 2_r};
    std::array<ControlType, 2> controlTypes = {ControlType::SingleDof, ControlType::SingleDof};
    std::array<int, 2> dofs = {3, 4};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectOK{});
  }

  // Test 7: With pose controller - mixed force and pose
  {
    std::array<real, 3> target = {1.0_r, 2_r, 3.0_r};
    std::array<ControlType, 3> controlTypes = {
        ControlType::Force, ControlType::SingleDof, ControlType::Force};
    std::array<int, 3> dofs = {3, 4, 5};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectOK{});
  }

  // Test 8: Non-sequential DOFs
  {
    std::array<real, 3> target = {0.5_r, 0.5_r, 0.5_r};
    std::array<ControlType, 3> controlTypes = {
        ControlType::Force, ControlType::Force, ControlType::Force};
    std::array<int, 3> dofs = {0, N / 2, N - 1}; // first, middle, last
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectOK{});
  }

  // Test 9: Reverse order DOFs
  {
    std::array<real, 3> target = {0.5_r, 0.5_r, 0.5_r};
    std::array<ControlType, 3> controlTypes = {
        ControlType::Force, ControlType::Force, ControlType::Force};
    std::array<int, 3> dofs = {N - 1, N - 2, N - 3};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectOK{});
  }
}

TEST_IF_F(
    MOCHI_HDF5_AND_INTERNAL,
    ArticulatedActorApiTest,
    SetArticulatedForceAndTargetPose_InvalidConfigurations) {
  auto const N = _actor->GetNumDofs();
  // Test 1: Size mismatch - controlTypes smaller than target
  {
    std::array<real, 3> target = {1.0_r, 2.0_r, 3.0_r};
    std::array<ControlType, 2> controlTypes = {ControlType::Force, ControlType::Force};
    std::array<int, 3> dofs = {0, 1, 2};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectNotOK{});
  }

  // Test 2: Size mismatch - controlTypes larger than target
  {
    std::array<real, 2> target = {1.0_r, 2.0_r};
    std::array<ControlType, 3> controlTypes = {
        ControlType::Force, ControlType::Force, ControlType::Force};
    std::array<int, 2> dofs = {0, 1};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectNotOK{});
  }

  // Test 3: Size mismatch - dofs smaller than target
  {
    std::array<real, 3> target = {1.0_r, 2.0_r, 3.0_r};
    std::array<ControlType, 3> controlTypes = {
        ControlType::Force, ControlType::Force, ControlType::Force};
    std::array<int, 2> dofs = {0, 1}; // size 2, but target is size 3
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectNotOK{});
  }

  // Test 4: Size mismatch - dofs larger than target
  {
    std::array<real, 2> target = {1.0_r, 2.0_r};
    std::array<ControlType, 2> controlTypes = {ControlType::Force, ControlType::Force};
    std::array<int, 3> dofs = {0, 1, 2}; // size 3, but target is size 2
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectNotOK{});
  }

  // Test 5: DOF index out of range - negative index
  {
    std::array<real, 1> target = {1.0_r};
    std::array<ControlType, 1> controlTypes = {ControlType::Force};
    std::array<int, 1> dofs = {-1};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectNotOK{});
  }

  // Test 6: DOF index out of range - index equals numDofs
  {
    std::array<real, 1> target = {1.0_r};
    std::array<ControlType, 1> controlTypes = {ControlType::Force};
    std::array<int, 1> dofs = {N}; // N is out of range (valid range is 0 to N-1)
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectNotOK{});
  }

  // Test 7: DOF index out of range - index greater than numDofs
  {
    std::array<real, 1> target = {1.0_r};
    std::array<ControlType, 1> controlTypes = {ControlType::Force};
    std::array<int, 1> dofs = {N + 100};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectNotOK{});
  }

  // Test 8: Valid DOF with out-of-range DOF
  {
    std::array<real, 2> target = {1.0_r, 2.0_r};
    std::array<ControlType, 2> controlTypes = {ControlType::Force, ControlType::Force};
    std::array<int, 2> dofs = {0, N}; // first valid, second out of range
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectNotOK{});
  }

  // Test 9: Large negative DOF index
  {
    std::array<real, 1> target = {1.0_r};
    std::array<ControlType, 1> controlTypes = {ControlType::Force};
    std::array<int, 1> dofs = {-1000};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectNotOK{});
  }

  // Test 10: All spans have different sizes (triple mismatch)
  {
    std::array<real, 2> target = {1.0_r, 2.0_r};
    std::array<ControlType, 3> controlTypes = {
        ControlType::Force, ControlType::Force, ControlType::Force};
    std::array<int, 1> dofs = {0};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectNotOK{});
  }

  // Test 11: Duplicate DOF indices
  {
    std::array<real, 2> target = {1.0_r, 2.0_r};
    std::array<ControlType, 2> controlTypes = {ControlType::Force, ControlType::Force};
    std::array<int, 2> dofs = {0, 0}; // duplicate index
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectNotOK{});
  }

  // Test 12: Duplicate DOF indices in longer list
  {
    std::array<real, 4> target = {1.0_r, 2.0_r, 3.0_r, 4.0_r};
    std::array<ControlType, 4> controlTypes = {
        ControlType::Force, ControlType::Force, ControlType::Force, ControlType::Force};
    std::array<int, 4> dofs = {0, 1, 2, 1}; // index 1 appears twice
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectNotOK{});
  }

  // Test 13: Duplicate DOF indices with mixed control types (SingleDof and Force share DOF
  // namespace)
  {
    std::array<real, 2> target = {1.0_r, 2.0_r};
    std::array<ControlType, 2> controlTypes = {ControlType::SingleDof, ControlType::Force};
    std::array<int, 2> dofs = {0, 0}; // same DOF with different control types
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofs, test::ExpectNotOK{});
  }

  // Test 14: Same link with different 3D control types (Pos + Rot) is valid (6-DOF control)
  {
    std::array<real, 6> target = {1.0_r, 2.0_r, 3.0_r, 4.0_r, 5.0_r, 6.0_r};
    std::array<ControlType, 2> controlTypes = {ControlType::LinkPos, ControlType::LinkRot};
    std::array<int, 2> links = {0, 0}; // same link with different 3D control types - OK
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, links, test::ExpectOK{});
  }

  // Test 15: Same index used for DOF and link is OK (different namespaces)
  {
    std::array<real, 4> target = {1.0_r, 2.0_r, 3.0_r, 4.0_r};
    std::array<ControlType, 2> controlTypes = {ControlType::Force, ControlType::LinkPos};
    std::array<int, 2> dofOrLink = {
        0, 0}; // DOF 0 and Link 0 are separate namespaces - should be OK
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, dofOrLink, test::ExpectOK{});
  }
}

TEST_IF_F(MOCHI_HDF5_AND_INTERNAL, ArticulatedActorApiTest, GetContactForceAndTorque) {
  // Add a ground to prevent a simple free fall.
  AddGround();

  // Enable contact queries on all links with a shape. Skip the first 2 links, which are part of the
  // root and are shapeless.
  int constexpr kShapeless = 2;
  auto const linkHandles = _actor->GetNestedLinkActors(test::ExpectOK{});
  std::vector<Actor*> links(linkHandles.size() - kShapeless);
  std::vector<QueryHandle> queries(2 * linkHandles.size() - kShapeless);
  for (int i = 0; i < links.size(); ++i) {
    auto* link = _scene->GetActor(linkHandles[i + kShapeless]);
    links[i] = link;
    queries[2 * i] = link->RegisterQuery(QueryType::NodeContactForces, test::ExpectOK{});
    queries[2 * i + 1] = link->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  }

  // Step the scene some steps
  for (int i = 0; i < 50; ++i) {
    _scene->Step(1e-2_r);
  }

  // Get the contact forces and torques for each link
  // Check that at least some forces and torques are non-zero
  // Check that there are some contacts.
  Real3 totalForceA = {};
  Real3 totalForceB = {};
  Real3 totalTorque = {};
  int totalContacts = 0;
  for (auto const* link : links) {
    auto force = link->GetContactForceWorld(test::ExpectOK{});
    auto torque = link->GetContactTorqueWorld(test::ExpectOK{});
    totalForceA += force;
    totalTorque += torque;

    auto contactForces = link->GetNodeContactForcesWorld(test::ExpectOK{});
    auto numContacts = isize(contactForces);
    totalContacts += numContacts;

    if (numContacts > 0) {
      for (auto const& contactForce : contactForces) {
        totalForceB += contactForce.force;
      }
    }
  }
  EXPECT_GT(totalContacts, 0);
  EXPECT_NE(Real3{}, totalForceA);
  EXPECT_NE(Real3{}, totalForceB);
  EXPECT_NE(Real3{}, totalTorque);
  EXPECT_NEAR_TOL(totalForceA, totalForceB, 2e-4_r);

  // Disable the contact queries
  for (int i = 0; i < links.size(); ++i) {
    links[i]->CancelQuery(queries[2 * i]);
    links[i]->CancelQuery(queries[2 * i + 1]);
  }

  // Try to get contact force and torque again
  for (auto const* link : links) {
    EXPECT_EQ(Real3{}, link->GetContactForceWorld(test::ExpectNotOK{}));
    EXPECT_EQ(Real3{}, link->GetContactTorqueWorld(test::ExpectNotOK{}));
    EXPECT_TRUE(link->GetNodeContactForcesWorld(test::ExpectNotOK{}).empty());
  }
}

TEST_IF_F(
    MOCHI_HDF5_AND_INTERNAL,
    ArticulatedActorApiTest,
    SetGetArticulatedTargetPoseAndLinkTransforms) {
  // Get number of links and dofs
  auto const numLinks = _actor->GetNestedLinkActors(test::ExpectOK{}).size();
  auto const numDofs = _actor->GetNumDofs();

  // Get current link transforms, which will serve later as target
  std::vector<TransformRT> transforms(numLinks);
  _actor->GetArticulatedLinkTransforms(transforms, test::ExpectOK{});

  // Get current pose
  std::vector<real> pose(numDofs);
  _actor->GetArticulatedPose(pose, test::ExpectOK{});

  // Apply a small perturbation to the target pose
  for (auto& dof : pose) {
    dof += 0.1_r;
  }
  _actor->SetArticulatedTargetPose(pose, test::ExpectOK{});

  // Validate that it's correct
  std::vector<real> poseTest(numDofs);
  _actor->GetArticulatedTargetPose(poseTest, test::ExpectOK{});
  for (int i = 0; i < numDofs; ++i) {
    EXPECT_EQ(pose[i], poseTest[i]);
  }

  // Validate that the current target pose does not match the initial one.
  // Skip the first two links, because their target rotations don't change (the first two joints are
  // prismatic).
  std::vector<TransformRT> transformsTest(numLinks);
  _actor->GetArticulatedTargetLinkTransforms(transformsTest, test::ExpectOK{});
  for (size_t i = 2; i < numLinks; ++i) {
    EXPECT_NE(transforms[i].GetTranslation(), transformsTest[i].GetTranslation());
    EXPECT_NE(transforms[i].GetRotation(), transformsTest[i].GetRotation());
  }

  // Set the target link transforms
  _actor->SetArticulatedTargetLinkTransforms(transforms, test::ExpectOK{});

  // Get the target link transforms back and verify round-trip
  _actor->GetArticulatedTargetLinkTransforms(transformsTest, test::ExpectOK{});

  // Verify that the retrieved target transforms match what we set
  for (size_t i = 0; i < numLinks; ++i) {
    EXPECT_NEAR_EQ(transforms[i].GetTranslation(), transformsTest[i].GetTranslation());
    EXPECT_NEAR_EQ(transforms[i].GetRotation(), transformsTest[i].GetRotation());
  }
}

TEST_IF_F(
    MOCHI_HDF5_AND_INTERNAL,
    ArticulatedActorApiTest,
    SetArticulatedForceAndTargetPose_LinkPos) {
  auto const numLinks = _actor->GetNestedLinkActors(test::ExpectOK{}).size();

  // Get initial target transforms
  std::vector<TransformRT> initialTargetTransforms(numLinks);
  _actor->GetArticulatedTargetLinkTransforms(initialTargetTransforms, test::ExpectOK{});

  // Test 3D position control on a link
  // Note: The actual effect depends on the model's kinematics. This test verifies the API
  // executes without error and exercises the LinkPos code path.
  int constexpr kLinkIndex = 2;
  Real3 const targetPos{1.5_r, 2.0_r, 3.0_r};

  std::array<real, 3> target = {targetPos[0], targetPos[1], targetPos[2]};
  std::array<ControlType, 1> controlTypes = {ControlType::LinkPos};
  std::array<int, 1> links = {kLinkIndex};

  // The API should succeed without error
  SetArticulatedForceAndTargetPose(_actor, target, controlTypes, links, test::ExpectOK{});

  // Verify that target link transform has changed (IK was triggered)
  std::vector<TransformRT> targetTransforms(numLinks);
  _actor->GetArticulatedTargetLinkTransforms(targetTransforms, test::ExpectOK{});

  // The rotation should have changed (IK may not reach exactly due to joint constraints)
  Real3 const initialPos = initialTargetTransforms[kLinkIndex].GetTranslation();
  Real3 const newPos = targetTransforms[kLinkIndex].GetTranslation();
  EXPECT_NE(initialPos, newPos);
}

TEST_IF_F(
    MOCHI_HDF5_AND_INTERNAL,
    ArticulatedActorApiTest,
    SetArticulatedForceAndTargetPose_LinkRot) {
  auto const numLinks = _actor->GetNestedLinkActors(test::ExpectOK{}).size();

  // Get initial target transforms
  std::vector<TransformRT> initialTargetTransforms(numLinks);
  _actor->GetArticulatedTargetLinkTransforms(initialTargetTransforms, test::ExpectOK{});

  // Test 3D rotation control on a link (skip root links)
  int constexpr kLinkIndex = 3;
  Real3 const rotVec{0.1_r, 0.2_r, 0.3_r};

  std::array<real, 3> target = {rotVec[0], rotVec[1], rotVec[2]};
  std::array<ControlType, 1> controlTypes = {ControlType::LinkRot};
  std::array<int, 1> links = {kLinkIndex};

  SetArticulatedForceAndTargetPose(_actor, target, controlTypes, links, test::ExpectOK{});

  // Verify that target link transform has changed (IK was triggered)
  std::vector<TransformRT> targetTransforms(numLinks);
  _actor->GetArticulatedTargetLinkTransforms(targetTransforms, test::ExpectOK{});

  // The rotation should have changed (IK may not reach exactly due to joint constraints)
  Quaternion const initialRot = initialTargetTransforms[kLinkIndex].GetRotation();
  Quaternion const newRot = targetTransforms[kLinkIndex].GetRotation();
  EXPECT_NE(initialRot, newRot);
}

TEST_IF_F(
    MOCHI_HDF5_AND_INTERNAL,
    ArticulatedActorApiTest,
    SetArticulatedForceAndTargetPose_MixedModes) {
  auto const numLinks = _actor->GetNestedLinkActors(test::ExpectOK{}).size();
  auto const numDofs = _actor->GetNumDofs();

  int constexpr kDofIndex = 3;
  int constexpr kLinkIndex = 4;
  real constexpr kPoseVal = 0.5_r;

  // The root-frame merge only differs from a CoM-frame merge when the controlled link has a
  // nonzero CoM offset, so the assertions below are only meaningful for such a link.
  auto const& linkActors = _actor->GetNestedLinkActors(test::ExpectOK{});
  Actor* const controlledLink = _scene->GetActor(linkActors[kLinkIndex]);
  ASSERT_NE(nullptr, controlledLink);
  EXPECT_FALSE(NearEqual(Real3{}, controlledLink->GetRigidCenterOfMassLocal(test::ExpectOK{})));

  // Descendants compose from the merged parent transform, so pick a child of the controlled link.
  auto const parents = _actor->GetArticulatedShapeInfo(test::ExpectOK{}).parents;
  auto const childIt = std::find(parents.begin() + kLinkIndex + 1, parents.end(), kLinkIndex);
  ASSERT_NE(parents.end(), childIt);
  auto const childIndex = static_cast<int>(childIt - parents.begin());

  DynamicArray<real> fkPose(numDofs);
  _actor->GetArticulatedPose(fkPose, test::ExpectOK{});
  fkPose[kDofIndex] = kPoseVal;

  DynamicArray<TransformRT> fkTransforms(numLinks);
  DynamicArray<TransformRT> targetTransforms(numLinks);
  _actor->SetArticulatedTargetPose(fkPose, test::ExpectOK{});
  _actor->GetArticulatedTargetLinkTransforms(fkTransforms, test::ExpectOK{});
  {
    Real3 const posOffset{0.25_r, -0.1_r, 0.15_r};
    Real3 const targetPos = fkTransforms[kLinkIndex].GetTranslation() + posOffset;
    std::array<real, 4> target = {kPoseVal, targetPos[0], targetPos[1], targetPos[2]};
    std::array<ControlType, 2> controlTypes = {ControlType::SingleDof, ControlType::LinkPos};
    std::array<int, 2> indices = {kDofIndex, kLinkIndex};

    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, indices, test::ExpectOK{});

    _actor->GetArticulatedTargetLinkTransforms(targetTransforms, test::ExpectOK{});
    EXPECT_NEAR_EQ(targetPos, targetTransforms[kLinkIndex].GetTranslation());
    EXPECT_NEAR_EQ(
        fkTransforms[kLinkIndex].GetRotation(), targetTransforms[kLinkIndex].GetRotation());

    // The parent's rotation is unchanged, so the child is rigidly translated by the same offset.
    EXPECT_NEAR_EQ(
        fkTransforms[childIndex].GetTranslation() + posOffset,
        targetTransforms[childIndex].GetTranslation());
    EXPECT_NEAR_EQ(
        fkTransforms[childIndex].GetRotation(), targetTransforms[childIndex].GetRotation());
  }

  _actor->SetArticulatedTargetPose(fkPose, test::ExpectOK{});
  _actor->GetArticulatedTargetLinkTransforms(fkTransforms, test::ExpectOK{});
  {
    Real3 const rotVec{0.2_r, 0.1_r, -0.1_r};
    Quaternion const targetRot = Quaternion::FromRotationVector(rotVec);
    std::array<real, 4> target = {kPoseVal, rotVec[0], rotVec[1], rotVec[2]};
    std::array<ControlType, 2> controlTypes = {ControlType::SingleDof, ControlType::LinkRot};
    std::array<int, 2> indices = {kDofIndex, kLinkIndex};

    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, indices, test::ExpectOK{});

    _actor->GetArticulatedTargetLinkTransforms(targetTransforms, test::ExpectOK{});
    EXPECT_NEAR_EQ(
        fkTransforms[kLinkIndex].GetTranslation(), targetTransforms[kLinkIndex].GetTranslation());
    EXPECT_NEAR_EQ(targetRot, targetTransforms[kLinkIndex].GetRotation());
  }

  _actor->SetArticulatedTargetPose(fkPose, test::ExpectOK{});
  _actor->GetArticulatedTargetLinkTransforms(fkTransforms, test::ExpectOK{});
  {
    // Full 6-DoF control: both components come from the target, so the requested transform must
    // round-trip exactly through the root/CoM frame conversions.
    Real3 const targetPos =
        fkTransforms[kLinkIndex].GetTranslation() + Real3{-0.2_r, 0.3_r, 0.05_r};
    Real3 const rotVec{0.2_r, 0.1_r, -0.1_r};
    Quaternion const targetRot = Quaternion::FromRotationVector(rotVec);
    std::array<real, 7> target = {
        kPoseVal, targetPos[0], targetPos[1], targetPos[2], rotVec[0], rotVec[1], rotVec[2]};
    std::array<ControlType, 3> controlTypes = {
        ControlType::SingleDof, ControlType::LinkPos, ControlType::LinkRot};
    std::array<int, 3> indices = {kDofIndex, kLinkIndex, kLinkIndex};

    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, indices, test::ExpectOK{});

    _actor->GetArticulatedTargetLinkTransforms(targetTransforms, test::ExpectOK{});
    EXPECT_NEAR_EQ(targetPos, targetTransforms[kLinkIndex].GetTranslation());
    EXPECT_NEAR_EQ(targetRot, targetTransforms[kLinkIndex].GetRotation());
  }
}

TEST_IF_F(
    MOCHI_HDF5_AND_INTERNAL,
    ArticulatedActorApiTest,
    SetArticulatedForceAndTargetPose_MixedForceAnd3DRot) {
  auto const numLinks = _actor->GetNestedLinkActors(test::ExpectOK{}).size();

  // Get initial state
  std::vector<TransformRT> initialTargetTransforms(numLinks);
  _actor->GetArticulatedTargetLinkTransforms(initialTargetTransforms, test::ExpectOK{});

  // Mix force (Force) with 3D rotation (LinkRot)
  real constexpr kForceVal = 10.0_r;
  Real3 const rotVec{0.2_r, 0.1_r, -0.1_r};
  int constexpr kDofIndex = 5;
  int constexpr kLinkIndex = 3;

  // target = [forceVal, rotX, rotY, rotZ]
  std::array<real, 4> target = {kForceVal, rotVec[0], rotVec[1], rotVec[2]};
  std::array<ControlType, 2> controlTypes = {ControlType::Force, ControlType::LinkRot};
  std::array<int, 2> indices = {kDofIndex, kLinkIndex};

  SetArticulatedForceAndTargetPose(_actor, target, controlTypes, indices, test::ExpectOK{});

  // Verify link rotation has changed (IK was triggered)
  std::vector<TransformRT> targetTransforms(numLinks);
  _actor->GetArticulatedTargetLinkTransforms(targetTransforms, test::ExpectOK{});

  Quaternion const initialRot = initialTargetTransforms[kLinkIndex].GetRotation();
  Quaternion const newRot = targetTransforms[kLinkIndex].GetRotation();
  EXPECT_NE(initialRot, newRot);
}

TEST_IF_F(
    MOCHI_HDF5_AND_INTERNAL,
    ArticulatedActorApiTest,
    SetArticulatedForceAndTargetPose_Invalid3DModes) {
  auto const numLinks = _actor->GetNestedLinkActors(test::ExpectOK{}).size();

  // Test 1: LinkPos with out-of-range link index (negative)
  {
    std::array<real, 3> target = {1.0_r, 2.0_r, 3.0_r};
    std::array<ControlType, 1> controlTypes = {ControlType::LinkPos};
    std::array<int, 1> links = {-1};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, links, test::ExpectNotOK{});
  }

  // Test 2: LinkPos with out-of-range link index (too large)
  {
    std::array<real, 3> target = {1.0_r, 2.0_r, 3.0_r};
    std::array<ControlType, 1> controlTypes = {ControlType::LinkPos};
    std::array<int, 1> links = {static_cast<int>(numLinks)};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, links, test::ExpectNotOK{});
  }

  // Test 3: LinkRot with out-of-range link index
  {
    std::array<real, 3> target = {0.1_r, 0.2_r, 0.3_r};
    std::array<ControlType, 1> controlTypes = {ControlType::LinkRot};
    std::array<int, 1> links = {static_cast<int>(numLinks) + 100};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, links, test::ExpectNotOK{});
  }

  // Test 4: Insufficient values for 3D mode (only 2 values instead of 3)
  {
    std::array<real, 2> target = {1.0_r, 2.0_r}; // should be 3 values
    std::array<ControlType, 1> controlTypes = {ControlType::LinkPos};
    std::array<int, 1> links = {2};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, links, test::ExpectNotOK{});
  }

  // Test 5: Too many values for single control type
  {
    std::array<real, 4> target = {1.0_r, 2.0_r, 3.0_r, 4.0_r}; // should be 3 values
    std::array<ControlType, 1> controlTypes = {ControlType::LinkRot};
    std::array<int, 1> links = {2};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, links, test::ExpectNotOK{});
  }

  // Test 6: Mixed modes with incorrect total values
  {
    // SingleDof (1 value) + LinkPos (3 values) = 4 values expected
    std::array<real, 3> target = {0.5_r, 1.0_r, 2.0_r}; // only 3, should be 4
    std::array<ControlType, 2> controlTypes = {ControlType::SingleDof, ControlType::LinkPos};
    std::array<int, 2> indices = {3, 2};
    SetArticulatedForceAndTargetPose(_actor, target, controlTypes, indices, test::ExpectNotOK{});
  }
}

namespace {

// Build a TetrahedralMeshShape with single-bone skinning (every node pinned to bone 0)
// and register it with the context. Returns the resulting handle.
ShapeHandle CreateUnitCubeTetMeshShapeWithSkinning(Context* context) {
  auto&& [coords, connectivity] = test::CreateMinimalTetMeshUnitCube();
  auto mesh = std::make_shared<TetrahedralMesh const>(coords, connectivity);
  auto skinning = std::make_shared<SkinningData const>(
      test::MakeSingleBoneSkinning(mesh->GetNumNodes(), /*boneIndex=*/0));
  auto shape = std::make_shared<TetrahedralMeshShape>(mesh, skinning);
  return assert_cast<ContextImpl*>(context)->RegisterShape(shape, test::ExpectOK{});
}

// Build a TriangularMeshShape with single-bone skinning (every node pinned to bone 0)
// and register it with the context. Returns the resulting handle.
ShapeHandle CreateUnitCubeTriMeshShapeWithSkinning(Context* context) {
  auto&& [coords, connectivity] = test::CreateMinimalTriMeshUnitCube();
  auto mesh = std::make_unique<TriangularMesh>(coords, connectivity);
  int const numNodes = mesh->GetNumNodes();
  auto skinning =
      std::make_shared<SkinningData const>(test::MakeSingleBoneSkinning(numNodes, /*boneIndex=*/0));
  auto shape = std::make_shared<TriangularMeshShape>(std::move(mesh), skinning);
  return assert_cast<ContextImpl*>(context)->RegisterShape(shape, test::ExpectOK{});
}

// Build a BlendingDataMap with identity blending (every target node = source node `t` with
// weight 1) for a single source soft actor named `softName`. The blended pipeline reads slot
// 1 of each per-target pair; slot 0 is unused (weight 0).
std::shared_ptr<BlendingDataMap const> MakeIdentityBlendingMap(
    DynamicString const& softName,
    int numTargetNodes) {
  BlendingDataTargetMesh target;
  target.indices.resize(numTargetNodes * 2, 0);
  target.weights.resize(numTargetNodes * 2, 0_r);
  for (int t = 0; t < numTargetNodes; ++t) {
    target.indices[2 * t + 1] = t;
    target.weights[2 * t + 1] = 1_r;
  }
  auto map = std::make_shared<BlendingDataMap>();
  map->perSourceShapeData.emplace(softName, std::move(target));
  return map;
}

// Build a *blended* TetrahedralMeshShape: includes single-bone skinning and an identity
// blending map keyed by the given soft-actor name.
ShapeHandle CreateUnitCubeTetBlendedSkinShape(Context* context, DynamicString const& softName) {
  auto&& [coords, connectivity] = test::CreateMinimalTetMeshUnitCube();
  auto mesh = std::make_shared<TetrahedralMesh const>(coords, connectivity);
  int const numNodes = mesh->GetNumNodes();
  auto skinning =
      std::make_shared<SkinningData const>(test::MakeSingleBoneSkinning(numNodes, /*boneIndex=*/0));
  auto blending = MakeIdentityBlendingMap(softName, numNodes);
  auto shape = std::make_shared<TetrahedralMeshShape>(
      mesh, skinning, /*constrainedNodesData=*/nullptr, blending);
  return assert_cast<ContextImpl*>(context)->RegisterShape(shape, test::ExpectOK{});
}

// Build a *blended* TriangularMeshShape: includes single-bone skinning and an identity
// blending map keyed by the given soft-actor name.
ShapeHandle CreateUnitCubeTriBlendedSkinShape(Context* context, DynamicString const& softName) {
  auto&& [coords, connectivity] = test::CreateMinimalTriMeshUnitCube();
  auto mesh = std::make_unique<TriangularMesh>(coords, connectivity);
  int const numNodes = mesh->GetNumNodes();
  auto skinning =
      std::make_shared<SkinningData const>(test::MakeSingleBoneSkinning(numNodes, /*boneIndex=*/0));
  auto blending = MakeIdentityBlendingMap(softName, numNodes);
  auto shape = std::make_shared<TriangularMeshShape>(
      std::move(mesh), skinning, /*constrainedNodesData=*/nullptr, blending);
  return assert_cast<ContextImpl*>(context)->RegisterShape(shape, test::ExpectOK{});
}

// Non-trivial subsampling configuration so the subsampling code path is exercised.
BoundarySubsamplingParams MakeSubsampling() {
  BoundarySubsamplingParams subsampling;
  subsampling.subsamplingDensity = 0.5_r;
  return subsampling;
}

[[nodiscard]] bool DisplacementsMatchTranslation(
    Span<real const> before,
    Span<real const> after,
    Real3 const& translation,
    real tolerance) {
  if (before.empty() || before.size() != after.size() || before.size() % 3 != 0) {
    return false;
  }
  DynamicArray<real> expected(before);
  for (int i = 0; i < isize(expected); i += 3) {
    for (int axis = 0; axis < 3; ++axis) {
      expected[i + axis] += translation[axis];
    }
  }
  return test::NearEqualSpan(expected, after, tolerance);
}

} // namespace

class CreateSkinnedArticulatedActorTest : public test::MochiSceneTestBase {
 protected:
  // Build minimal ArticulatedActorParams describing a one-bone articulated skeleton with
  // the given shape used as the skin shape. Subsampling is enabled.
  ArticulatedActorParams MakeMinimalSkinnedParams(ShapeHandle skinShape) {
    ArticulatedActorParams params;
    params.joints = {{.type = ArticulatedJointType::Free}};
    params.links = {
        {.parentLink = -1,
         .shape = test::CreateUnitCubeTetMeshShape(_mochiContext),
         .colliderType = ColliderType::None}};
    ArticulatedSkinParams skin;
    skin.shape = skinShape;
    skin.boundarySubsampling = MakeSubsampling();
    params.skin = skin;
    return params;
  }
};

// Guards joint-pose resets from leaving skin-only public displacements stale.
TEST_F(CreateSkinnedArticulatedActorTest, JointPosePublishesSkinBeforeStep) {
  auto params = MakeMinimalSkinnedParams(CreateUnitCubeTetMeshShapeWithSkinning(_mochiContext));
  auto* actor = _scene->CreateArticulatedActor(params, test::ExpectOK{});
  ASSERT_NE(nullptr, actor);

  auto const initialSpan = actor->GetDisplacements(test::ExpectOK{});
  DynamicArray<real> const initial(initialSpan);
  DynamicArray<TransformRT> linkTransforms(1);
  actor->GetArticulatedLinkTransforms(linkTransforms, test::ExpectOK{});
  Real3 const linkTranslationBefore = linkTransforms[0].GetTranslation();
  DynamicArray<real> pose(actor->GetNumDofs());
  actor->GetArticulatedPose(pose, test::ExpectOK{});
  ASSERT_FALSE(pose.empty());
  Real3 constexpr kTranslation{1_r, 0_r, 0_r};
  pose[0] += kTranslation[0];

  actor->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});

  actor->GetArticulatedLinkTransforms(linkTransforms, test::ExpectOK{});
  EXPECT_TRUE(NearEqual(
      linkTranslationBefore + kTranslation, linkTransforms[0].GetTranslation(), kTolerance));
  EXPECT_TRUE(DisplacementsMatchTranslation(
      initial, actor->GetDisplacements(test::ExpectOK{}), kTranslation, kTolerance));
}

// Create a minimal one-bone articulated actor whose skin is a tetrahedral mesh, with
// boundary subsampling enabled. Exercises the volumetric-skin code path.
TEST_F(CreateSkinnedArticulatedActorTest, TetMesh) {
  auto params = MakeMinimalSkinnedParams(CreateUnitCubeTetMeshShapeWithSkinning(_mochiContext));

  Actor const* actor = _scene->CreateArticulatedActor(params, test::ExpectOK{});
  ASSERT_NE(nullptr, actor);
  EXPECT_EQ(ActorType::Articulated, actor->GetType());
}

// Create a minimal one-bone articulated actor whose skin is a triangular mesh, with
// boundary subsampling enabled. Exercises the surface-only skin code path.
TEST_F(CreateSkinnedArticulatedActorTest, TriMesh) {
  auto params = MakeMinimalSkinnedParams(CreateUnitCubeTriMeshShapeWithSkinning(_mochiContext));

  Actor const* actor = _scene->CreateArticulatedActor(params, test::ExpectOK{});
  ASSERT_NE(nullptr, actor);
  EXPECT_EQ(ActorType::Articulated, actor->GetType());
}

// A zero-dof (all-Hard) skeleton cannot carry a skin: there is no joint Jacobian for the skin to
// deform with, so creation must fail with an error.
TEST_F(CreateSkinnedArticulatedActorTest, ZeroDofHardSkeletonWithSkinRejected) {
  auto params = MakeMinimalSkinnedParams(CreateUnitCubeTetMeshShapeWithSkinning(_mochiContext));
  params.joints = {{.type = ArticulatedJointType::Hard}};

  EXPECT_EQ(nullptr, _scene->CreateArticulatedActor(params, test::ExpectNotOK{}));
}

class CreateBlendedActorTest : public test::MochiSceneTestBase {
 protected:
  // Build minimal SoftSkinnedActorParams describing a one-bone articulated skeleton with
  // one soft-body actor and the given shape used as the blended skin shape. Subsampling is
  // enabled.
  SoftSkinnedActorParams MakeMinimalBlendedParams(ShapeHandle blendedSkinShape) {
    SoftSkinnedActorParams params;
    auto& skeleton = params.skeletonParams;
    skeleton.joints = {{.type = ArticulatedJointType::Free}};
    skeleton.links = {
        {.parentLink = -1,
         .shape = test::CreateUnitCubeTetMeshShape(_mochiContext),
         .colliderType = ColliderType::None}};
    ArticulatedSkinParams skin;
    skin.shape = blendedSkinShape;
    skin.boundarySubsampling = MakeSubsampling();
    skeleton.skin = skin;
    SoftActorParams softParams;
    softParams.name = "soft";
    softParams.shape = test::CreateUnitCubeTetSoftShape(_mochiContext);
    softParams.hasGravity = false;
    params.softParams = {softParams};
    return params;
  }
};

class ExternalPoseResetTest : public CreateBlendedActorTest {
 protected:
  enum class Route { Links, Joints, Root };

  [[nodiscard]] static DynamicArray<real> CopyDisplacements(Actor const& actor) {
    return DynamicArray<real>(actor.GetDisplacements(test::ExpectOK{}));
  }

  void CheckExternalPoseReset(Route route) {
    _scene->SetGravity({});
    auto solverParams = _scene->GetSolverParams();
    solverParams.integrationMethod = IntegrationMethod::BDF3;
    _scene->SetSolverParams(solverParams, test::ExpectOK{});

    auto params = MakeMinimalBlendedParams(
        CreateUnitCubeTetBlendedSkinShape(_mochiContext, DynamicString{"soft"}));
    params.softParams[0].hasInertia = true;
    params.softParams[0].hasStress = false;
    params.softParams[0].hasGravity = false;

    auto* parent = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
    ASSERT_NE(nullptr, parent);
    auto const linkHandles = parent->GetNestedLinkActors(test::ExpectOK{});
    auto const softHandles = parent->GetNestedSoftActors(test::ExpectOK{});
    ASSERT_EQ(1, isize(linkHandles));
    ASSERT_EQ(1, isize(softHandles));
    auto* link = _scene->GetActor(linkHandles[0]);
    auto* nested = _scene->GetActor(softHandles[0]);
    ASSERT_NE(nullptr, link);
    ASSERT_NE(nullptr, nested);

    TransformRT const linkInitial = link->GetRootTransform();
    auto const nestedInitial = CopyDisplacements(*nested);
    DynamicArray<real> elastic(nestedInitial.size(), 0_r);
    ASSERT_GT(elastic.size(), 3u);
    elastic[3] = 0.25_r;
    nested->SetDisplacements(elastic, test::ExpectOK{});

    for (int i = 0; i < 3; ++i) {
      _scene->Step(1e-3_r);
    }

    TransformRT const linkBefore = link->GetRootTransform();
    TransformRT const rootBefore = parent->GetRootTransform();
    auto const nestedBefore = CopyDisplacements(*nested);
    auto const parentBefore = CopyDisplacements(*parent);
    ASSERT_TRUE(NearEqual(linkInitial, linkBefore, kTolerance));
    ASSERT_FALSE(test::NearEqualSpan(nestedInitial, nestedBefore, kTolerance));
    ASSERT_TRUE(test::NearEqualSpan(parentBefore, nestedBefore, kTolerance));

    auto& reg = GetRegistry();
    entt::entity const parentEntity = GetEntity(parent);
    entt::entity const linkEntity = GetEntity(link);
    entt::entity const nestedEntity = GetEntity(nested);
    ASSERT_FALSE(reg.get<CIntegrationArticulatedReducedPose const>(parentEntity).prevSteps.empty());
    ASSERT_FALSE(reg.get<CIntegrationRigidVels const>(linkEntity).prevSteps.empty());
    ASSERT_FALSE(reg.get<CIntegrationDisplacementSlices const>(nestedEntity).prevSteps.empty());
    ASSERT_FALSE(reg.get<CConservativeStepBounds const>(linkEntity).needsNextStepRelaxation);
    ASSERT_FALSE(reg.get<CConservativeStepBounds const>(nestedEntity).needsNextStepRelaxation);

    Real3 constexpr kTranslation{1_r, 0_r, 0_r};
    switch (route) {
      case Route::Links: {
        DynamicArray<TransformRT> links(1);
        parent->GetArticulatedLinkTransforms(links, test::ExpectOK{});
        links[0].SetTranslation(links[0].GetTranslation() + kTranslation);
        parent->SetArticulatedPoseFromLinks(links, test::ExpectOK{});
        break;
      }
      case Route::Joints: {
        DynamicArray<real> pose(parent->GetNumDofs());
        parent->GetArticulatedPose(pose, test::ExpectOK{});
        ASSERT_FALSE(pose.empty());
        pose[0] += kTranslation[0];
        parent->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});
        break;
      }
      case Route::Root: {
        TransformRT root = rootBefore;
        root.SetTranslation(root.GetTranslation() + kTranslation);
        parent->SetRootTransform(root, test::ExpectOK{});
        break;
      }
    }

    Real3 const translationBefore =
        route == Route::Root ? rootBefore.GetTranslation() : linkBefore.GetTranslation();
    Real3 const translationAfter = route == Route::Root
        ? parent->GetRootTransform().GetTranslation()
        : link->GetRootTransform().GetTranslation();
    EXPECT_TRUE(NearEqual(translationBefore + kTranslation, translationAfter, kTolerance));
    auto const nestedAfter = CopyDisplacements(*nested);
    auto const parentAfter = CopyDisplacements(*parent);
    EXPECT_TRUE(DisplacementsMatchTranslation(nestedBefore, nestedAfter, kTranslation, kTolerance));
    EXPECT_TRUE(test::NearEqualSpan(parentAfter, nestedAfter, kTolerance));

    EXPECT_TRUE(reg.get<CIntegrationArticulatedReducedPose const>(parentEntity).prevSteps.empty());
    EXPECT_TRUE(reg.get<CIntegrationRigidVels const>(linkEntity).prevSteps.empty());
    EXPECT_TRUE(reg.get<CIntegrationDisplacementSlices const>(nestedEntity).prevSteps.empty());
    EXPECT_TRUE(reg.get<CConservativeStepBounds const>(linkEntity).needsNextStepRelaxation);
    EXPECT_TRUE(reg.get<CConservativeStepBounds const>(nestedEntity).needsNextStepRelaxation);
  }
};

// Guards link-authored resets against stale dependent state and integration history.
TEST_F(ExternalPoseResetTest, FromLinksPublishesDerivedState) {
  CheckExternalPoseReset(Route::Links);
}

// Guards joint-authored resets against stale dependent state and integration history.
TEST_F(ExternalPoseResetTest, FromJointsPublishesDerivedState) {
  CheckExternalPoseReset(Route::Joints);
}

// Guards root-transform resets against stale dependent state and integration history.
TEST_F(ExternalPoseResetTest, FromRootPublishesDerivedState) {
  CheckExternalPoseReset(Route::Root);
}

TEST_F(CreateBlendedActorTest, TetMesh) {
  auto params = MakeMinimalBlendedParams(
      CreateUnitCubeTetBlendedSkinShape(_mochiContext, DynamicString{"soft"}));

  Actor const* actor = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
  ASSERT_NE(nullptr, actor);
  EXPECT_EQ(ActorType::Articulated, actor->GetType());
}

TEST_F(CreateBlendedActorTest, TriMesh) {
  auto params = MakeMinimalBlendedParams(
      CreateUnitCubeTriBlendedSkinShape(_mochiContext, DynamicString{"soft"}));

  Actor const* actor = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
  ASSERT_NE(nullptr, actor);
  EXPECT_EQ(ActorType::Articulated, actor->GetType());
}

// A soft-skinned actor cannot be bound to a zero-dof (all-Hard) skeleton: there is no joint
// Jacobian to deform the soft bodies, so creation must fail with an error.
TEST_F(CreateBlendedActorTest, ZeroDofHardSkeletonRejected) {
  auto params = MakeMinimalBlendedParams(
      CreateUnitCubeTetBlendedSkinShape(_mochiContext, DynamicString{"soft"}));
  params.skeletonParams.joints = {{.type = ArticulatedJointType::Hard}};

  EXPECT_EQ(nullptr, _scene->CreateSoftSkinnedActor(params, test::ExpectNotOK{}));
}
