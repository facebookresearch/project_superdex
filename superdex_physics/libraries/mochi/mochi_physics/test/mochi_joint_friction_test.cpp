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

// Elasto-plastic dry friction of articulated joints (ArticulatedJointFrictionParams).

#include "mochi_physics_test_fixture.h"

#include <mochi_physics/mochi_physics.h>

#include <mochi_core/articulated_body/joint_friction.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_physics/src/mochi_articulated_body.h>
#include <mochi_physics/src/mochi_island.h>
#include <mochi_physics/src/mochi_joint_friction.h>
#include <mochi_physics/src/mochi_simulation.h>
#include <mochi_physics/src/mochi_sleep.h>
#include <mochi_physics/src/mochi_solve.h>

#include <gtest/gtest.h>
#include <vector>

using namespace mochi;

namespace {

real constexpr kDt = 1e-2_r;

// Static peak 100 (Coulomb 60 plus extra stiction 40), with a Stribeck velocity, and enough viscous
// damping to settle the presliding oscillation.
ArticulatedJointFrictionParams const kFriction{
    .viscous = 5e4_r,
    .coulomb = 60_r,
    .preslidingDisplacement = 1e-4_r,
    .stictionExtra = 40_r,
    .stribeckVel = 1e-2_r};
real constexpr kStaticPeak = 100_r;

// Displacement from the anchor that holds a load below the static peak: S f'(u) = load, with
// f'(u) = (2 - u / d) u / d.
real HeldDisplacement(real load) {
  return kFriction.preslidingDisplacement * (1_r - Sqrt(1_r - load / kStaticPeak));
}

class JointFrictionTest : public test::MochiSceneTestBase,
                          public ::testing::WithParamInterface<ArticulatedJointType> {
 protected:
  void SetUp() override {
    test::MochiSceneTestBase::SetUp();
    _scene->SetGravity(Real3{});
  }

  // A unit cube attached to the world at its center by a single joint of the tested type, along or
  // about x. Its inertia is isotropic about the joint, so a load along or about x moves it along or
  // about x only.
  ArticulatedActorParams JointActorParams(ArticulatedJointFrictionParams const& friction) {
    ArticulatedActorParams params;
    params.joints = {{.type = GetParam(), .axis = Real3{1_r, 0_r, 0_r}, .friction = friction}};
    params.links = {
        {.parentLink = -1,
         .parentJointFromLink = TransformRT{Real3{-0.5_r, -0.5_r, -0.5_r}},
         .shape = test::CreateUnitCubeTetMeshShape(_mochiContext),
         .colliderType = ColliderType::None}};
    return params;
  }

  Actor* CreateJointActor(ArticulatedJointFrictionParams const& friction) {
    return _scene->CreateArticulatedActor(JointActorParams(friction), test::ExpectOK{});
  }

  ColumnVector<real> const& GetAnchors(Actor* actor) {
    return GetRegistry().get<CArticulatedJointFrictionAnchors>(GetEntity(actor)).value;
  }

  // The step-start pose of the last step.
  ColumnVector<real> const& GetStepStartPose(Actor* actor) {
    return GetRegistry().get<CArticulatedReducedPose<TimeStep::Previous>>(GetEntity(actor)).value;
  }

  // The current pose of the joint (at the end of the last step).
  ColumnVector<real> const& GetCurrentPose(Actor* actor) {
    return GetRegistry().get<CArticulatedReducedPose<TimeStep::Current>>(GetEntity(actor)).value;
  }

  // Expects the joint anchors of an actor to be the given ones advanced by radial return to a pose,
  // and to differ from them.
  void ExpectAnchorsAdvanced(
      Actor* actor,
      ColumnVector<real> const& anchors,
      ColumnVector<real> const& pose,
      real delta) {
    if (GetParam() == ArticulatedJointType::Spherical) {
      auto const rotation = [](ColumnVector<real> const& values) {
        return Quaternion(Load<RigidSize::kRot, Vec4r>(&values[0]));
      };
      Quaternion const expected =
          AdvanceRotationFrictionAnchor(rotation(pose), rotation(anchors), delta);
      EXPECT_FALSE(NearEqual(expected, rotation(anchors)));
      EXPECT_EQ(rotation(GetAnchors(actor)), expected);
    } else {
      real const expected = AdvanceSingleDofFrictionAnchor(pose[0], anchors[0], delta);
      EXPECT_NE(expected, anchors[0]);
      EXPECT_EQ(GetAnchors(actor)[0], expected);
    }
  }

  // Applies a constant load along or about x on the joint.
  static void SetLoad(Actor* actor, real load) {
    std::vector<int> const dofs = {0};
    std::vector<real> const loads = {load};
    actor->SetExternalForcesOnDofs(dofs, loads, test::ExpectOK{});
  }

  static std::vector<real> GetPose(Actor* actor) {
    std::vector<real> pose(actor->GetNumDofs());
    actor->GetArticulatedPose(pose, test::ExpectOK{});
    return pose;
  }

  static real GetVelocity(Actor* actor) {
    std::vector<real> velocity(actor->GetNumDofs());
    actor->GetArticulatedJointVelocities(velocity, test::ExpectOK{});
    return velocity[0];
  }

  void Step(int numSteps) {
    for (int i = 0; i < numSteps; ++i) {
      _scene->Step(kDt);
    }
  }
};

INSTANTIATE_TEST_SUITE_P(
    JointTypes,
    JointFrictionTest,
    ::testing::Values(
        ArticulatedJointType::Prismatic,
        ArticulatedJointType::Revolute,
        ArticulatedJointType::Spherical));

} // namespace

// A load below the static peak is held at the presliding displacement that balances it, without
// creep, whatever the time integrator: the joint never slips, so its anchor stays at the start
// pose. The tolerance covers the convergence tolerance of the solver, relative to the stiff
// presliding response.
TEST_P(JointFrictionTest, HoldsLoadBelowStaticPeak) {
  for (IntegrationMethod const method :
       {IntegrationMethod::BackwardEuler,
        IntegrationMethod::BDF2,
        IntegrationMethod::DIRK22,
        IntegrationMethod::DIRK23,
        IntegrationMethod::SymplecticDIRK12}) {
    SCOPED_TRACE(static_cast<int>(method));
    auto solverParams = _scene->GetSolverParams();
    solverParams.integrationMethod = method;
    _scene->SetSolverParams(solverParams, test::ExpectOK{});

    Actor* actor = CreateJointActor(kFriction);
    real const load = 0.8_r * kStaticPeak;
    SetLoad(actor, load);
    Step(1);
    ColumnVector<real> const startAnchors = GetAnchors(actor).Duplicate();
    Step(300);
    EXPECT_NEAR(GetPose(actor)[0], HeldDisplacement(load), 5e-2_r * HeldDisplacement(load));
    EXPECT_TRUE(test::NearEqualMatrices(GetAnchors(actor), startAnchors, 0_r));
    _scene->DestroyActor(actor->GetHandle());
  }
}

// Beyond the static peak, the joint slides against the Coulomb force (here without extra
// stiction, so without Stribeck term), like a frictionless joint under the load minus the Coulomb
// force.
TEST_P(JointFrictionTest, SlidesAgainstCoulombForce) {
  ArticulatedJointFrictionParams const friction{.coulomb = 60_r};
  Actor* actor = CreateJointActor(friction);
  Actor* reference = CreateJointActor({});

  real const load = 2_r * friction.coulomb;
  SetLoad(actor, load);
  SetLoad(reference, load - friction.coulomb);
  // Once sliding, both accelerate equally. Before, presliding is elastic, so friction is smaller.
  Step(50);
  real const velocity = GetVelocity(actor);
  real const referenceVelocity = GetVelocity(reference);
  Step(50);
  real const referenceGain = GetVelocity(reference) - referenceVelocity;
  EXPECT_GT(referenceGain, 0_r);
  EXPECT_NEAR(GetVelocity(actor) - velocity, referenceGain, 1e-2_r * referenceGain);
}

// While sliding steadily at the Stribeck velocity v_s, dry friction is coulomb + stictionExtra / e,
// whatever the time integrator: a load of that plus the viscous force settles the joint at v_s. A
// wrong time since the anchor state, or anchors advanced within the step, would settle it at
// another velocity. The viscous friction keeps the sliding stable, since the Stribeck term
// decreases with the velocity.
TEST_P(JointFrictionTest, SettlesAtStribeckVelocity) {
  ArticulatedJointFrictionParams const friction{
      .viscous = 5e3_r,
      .coulomb = 60_r,
      .preslidingDisplacement = 1e-4_r,
      .stictionExtra = 40_r,
      .stribeckVel = 1e-2_r};
  for (IntegrationMethod const method :
       {IntegrationMethod::BackwardEuler,
        IntegrationMethod::BDF2,
        IntegrationMethod::DIRK22,
        IntegrationMethod::DIRK23,
        IntegrationMethod::SymplecticDIRK12}) {
    SCOPED_TRACE(static_cast<int>(method));
    auto solverParams = _scene->GetSolverParams();
    solverParams.integrationMethod = method;
    // Converged solves, so that the velocity settles where the forces balance.
    solverParams.nonLinearSolver.absTol = 1e-12_r;
    solverParams.nonLinearSolver.relTol = 1e-12_r;
    solverParams.nonLinearSolver.relStepTol = 0_r;
    solverParams.nonLinearSolver.maxIter = 100;
    _scene->SetSolverParams(solverParams, test::ExpectOK{});

    Actor* actor = CreateJointActor(friction);
    SetLoad(
        actor,
        friction.coulomb + friction.stictionExtra * Exp(-1_r) +
            friction.viscous * friction.stribeckVel);
    Step(500);
    EXPECT_NEAR(GetVelocity(actor), friction.stribeckVel, 1e-2_r * friction.stribeckVel);
    _scene->DestroyActor(actor->GetHandle());
  }
}

// A quasistatic step holds a load below the static peak at the same displacement.
TEST_P(JointFrictionTest, QuasistaticStepHoldsLoad) {
  Actor* actor = CreateJointActor(kFriction);
  real const load = 0.8_r * kStaticPeak;
  SetLoad(actor, load);
  _scene->Step(kInf);
  EXPECT_NEAR(GetPose(actor)[0], HeldDisplacement(load), 1e-2_r * HeldDisplacement(load));
}

// The friction anchors are captured: a restored state resumes with its preload. Placing the actor
// drops it, so the joint takes the load again from its new anchor.
TEST_P(JointFrictionTest, CaptureRestoresAndPlacementDropsPreload) {
  Actor* actor = CreateJointActor(kFriction);
  real const load = 0.8_r * kStaticPeak;
  SetLoad(actor, load);
  Step(200);
  StateHandle const state = _scene->CaptureState(test::ExpectOK{});

  Step(50);
  std::vector<real> const resumed = GetPose(actor);

  _scene->RestoreState(state, /*releaseImmediately*/ false, test::ExpectOK{});
  actor->SetArticulatedPoseFromJoints(GetPose(actor), test::ExpectOK{});
  auto& reg = GetRegistry();
  auto const e = GetEntity(actor);
  EXPECT_EQ(reg.get<CArticulatedJointFrictionAnchors>(e).value.Rows(), 0);
  Step(200);
  EXPECT_NEAR(GetPose(actor)[0], 2_r * HeldDisplacement(load), 5e-2_r * HeldDisplacement(load));

  _scene->RestoreState(state, /*releaseImmediately*/ true, test::ExpectOK{});
  Step(50);
  EXPECT_EQ(GetPose(actor), resumed);
}

// A state restored while the joint slides resumes as it went on. Its first step advances the
// restored anchors by radial return to its start pose, the end pose of the captured step.
TEST_P(JointFrictionTest, CaptureRestoresWhileSliding) {
  Actor* actor = CreateJointActor(kFriction);
  SetLoad(actor, 2_r * kStaticPeak);
  Step(20);
  StateHandle const state = _scene->CaptureState(test::ExpectOK{});
  ColumnVector<real> const anchors = GetAnchors(actor).Duplicate();
  ColumnVector<real> const endPose = GetCurrentPose(actor).Duplicate();
  Step(20);
  std::vector<real> const resumed = GetPose(actor);
  ColumnVector<real> const resumedAnchors = GetAnchors(actor).Duplicate();

  _scene->RestoreState(state, /*releaseImmediately*/ true, test::ExpectOK{});
  Step(1);
  ExpectAnchorsAdvanced(actor, anchors, endPose, kFriction.preslidingDisplacement);
  Step(19);
  EXPECT_EQ(GetPose(actor), resumed);
  EXPECT_TRUE(test::NearEqualMatrices(GetAnchors(actor), resumedAnchors, 0_r));
}

// Changing the friction parameters completes the last step's anchor advance with the previous
// presliding displacement, so that a joint that slid keeps its displacement from the anchor. A
// larger presliding displacement alone would leave the last step's slip as preload.
TEST_P(JointFrictionTest, ParameterChangeSettlesAnchors) {
  Actor* actor = CreateJointActor(kFriction);
  SetLoad(actor, 2_r * kStaticPeak);
  Step(20);
  ColumnVector<real> const anchors = GetAnchors(actor).Duplicate();
  DynamicArray<ArticulatedJointFrictionParams> friction(1, kFriction);
  friction[0].preslidingDisplacement *= 20_r;
  actor->SetArticulatedJointFrictionParams(friction, test::ExpectOK{});
  ExpectAnchorsAdvanced(actor, anchors, GetCurrentPose(actor), kFriction.preslidingDisplacement);
}

// A sleeping joint keeps its preload: its anchors stay put, and it holds the load once awake.
TEST_P(JointFrictionTest, SleepKeepsPreload) {
  Actor* actor = CreateJointActor(kFriction);
  real const load = 0.8_r * kStaticPeak;
  SetLoad(actor, load);
  Step(300);
  experimental::SetSleepParams(
      _scene, experimental::SleepParams{.canSleep = true}, test::ExpectOK{});
  Step(1);
  ColumnVector<real> const anchors = GetAnchors(actor).Duplicate();
  auto& reg = GetRegistry();
  entt::entity const island = reg.view<CIslandDescendants const>().front();
  sleep::ForceSleep(reg, island);
  Step(10);
  ASSERT_FALSE(sleep::IsIslandAwake(reg, island));
  EXPECT_TRUE(test::NearEqualMatrices(GetAnchors(actor), anchors, 1e-6_r));

  sleep::WakeUp(reg, GetEntity(actor));
  Step(1);
  ASSERT_TRUE(sleep::IsIslandAwake(reg, island));
  Step(100);
  EXPECT_NEAR(GetPose(actor)[0], HeldDisplacement(load), 5e-2_r * HeldDisplacement(load));
}

// Setters that keep the joint positions keep the friction preload.
TEST_P(JointFrictionTest, OtherSettersKeepPreload) {
  Actor* actor = CreateJointActor(kFriction);
  SetLoad(actor, 0.8_r * kStaticPeak);
  Step(10);
  ColumnVector<real> const anchors = GetAnchors(actor).Duplicate();

  std::vector<real> const velocity(actor->GetNumDofs(), 0_r);
  actor->SetArticulatedJointVelocities(velocity, test::ExpectOK{});
  actor->SetRootTransform(TransformRT{Real3{0_r, 1_r, 0_r}}, test::ExpectOK{});
  DynamicArray<ArticulatedJointFrictionParams> friction(
      actor->GetArticulatedJointFrictionParams(test::ExpectOK{}));
  friction[0].coulomb *= 2_r;
  actor->SetArticulatedJointFrictionParams(friction, test::ExpectOK{});
  EXPECT_TRUE(test::NearEqualMatrices(GetAnchors(actor), anchors, 0_r));
}

// A diverged step releases the joint preload, as for contacts: it resets the actor to the
// step-start pose and anchors the joint there, where the next step keeps it.
TEST_P(JointFrictionTest, DivergedStepReleasesAnchors) {
  test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
  Actor* actor = CreateJointActor(kFriction);
  SetLoad(actor, 0.8_r * kStaticPeak);
  Step(20);
  ColumnVector<real> const anchors = GetAnchors(actor).Duplicate();

  // Zero tolerances and a tiny divergence threshold make the next step diverge.
  SolverParams const params = _scene->GetSolverParams();
  SolverParams divergingParams = params;
  divergingParams.nonLinearSolver.absTol = 0_r;
  divergingParams.nonLinearSolver.relTol = 0_r;
  divergingParams.nonLinearSolver.relStepTol = 0_r;
  divergingParams.nonLinearSolver.absDivTol = 1e-30_r;
  _scene->SetSolverParams(divergingParams, test::ExpectOK{});
  Step(1);
  ASSERT_EQ(ConvergenceStatus::Diverged, actor->GetConvergenceStatus());
  EXPECT_FALSE(test::NearEqualMatrices(GetAnchors(actor), anchors, 0_r));
  EXPECT_TRUE(test::NearEqualMatrices(GetAnchors(actor), GetStepStartPose(actor), 0_r));
  ColumnVector<real> const releasedAnchors = GetAnchors(actor).Duplicate();

  _scene->SetSolverParams(params, test::ExpectOK{});
  Step(1);
  ASSERT_NE(ConvergenceStatus::Diverged, actor->GetConvergenceStatus());
  EXPECT_TRUE(test::NearEqualMatrices(GetAnchors(actor), releasedAnchors, 0_r));
}

// In an actor with joint dry friction, enabling it on another joint that moved without it starts
// that joint without preload: the next step anchors it at its start pose.
TEST_P(JointFrictionTest, EnablingJointDryFrictionStartsWithoutPreload) {
  ArticulatedActorParams params = JointActorParams(kFriction);
  params.joints.push_back({.type = GetParam(), .axis = Real3{1_r, 0_r, 0_r}});
  ArticulatedLinkParams link = params.links[0];
  link.parentLink = 0;
  params.links.push_back(link);
  Actor* actor = _scene->CreateArticulatedActor(params, test::ExpectOK{});
  int const loadedDof = actor->GetArticulatedShapeInfo(test::ExpectOK{}).dofInfo[1].offset;
  std::vector<int> const dofs = {loadedDof};
  actor->SetExternalForcesOnDofs(dofs, std::vector<real>{10_r * kStaticPeak}, test::ExpectOK{});
  Step(30);
  // The joint moves much more than the presliding displacement per step, so an anchor left where
  // its friction was disabled would preload it.
  std::vector<real> velocity(actor->GetNumDofs());
  actor->GetArticulatedJointVelocities(velocity, test::ExpectOK{});
  ASSERT_GT(velocity[loadedDof] * kDt, 10_r * kFriction.preslidingDisplacement);

  DynamicArray<ArticulatedJointFrictionParams> friction(
      actor->GetArticulatedJointFrictionParams(test::ExpectOK{}));
  friction[1] = kFriction;
  actor->SetArticulatedJointFrictionParams(friction, test::ExpectOK{});
  actor->SetExternalForcesOnDofs(dofs, std::vector<real>{0_r}, test::ExpectOK{});
  Step(1);
  auto const& poseInfo = GetRegistry().get<CArticulatedJointPoseInfo const>(GetEntity(actor))[1];
  bool const isSpherical = GetParam() == ArticulatedJointType::Spherical;
  int const offset = isSpherical ? poseInfo.GetRotOffset() : poseInfo.offset;
  int const size = isSpherical ? RigidSize::kRot : 1;
  for (int k = offset; k < offset + size; ++k) {
    EXPECT_EQ(GetAnchors(actor)[k], GetStepStartPose(actor)[k]);
  }
}

// An actor without joint dry friction keeps no anchors. Setting friction makes its next step anchor
// its joints at their start pose, without preload, and removing it makes the next step drop them,
// also when a state captured with friction is restored.
TEST_P(JointFrictionTest, FrictionlessActorKeepsNoAnchors) {
  Actor* actor = CreateJointActor({});
  SetLoad(actor, 0.8_r * kStaticPeak);
  Step(5);
  EXPECT_EQ(GetAnchors(actor).Rows(), 0);

  DynamicArray<ArticulatedJointFrictionParams> friction(1, kFriction);
  actor->SetArticulatedJointFrictionParams(friction, test::ExpectOK{});
  EXPECT_EQ(GetAnchors(actor).Rows(), 0);
  Step(1);
  EXPECT_TRUE(test::NearEqualMatrices(GetAnchors(actor), GetStepStartPose(actor), 0_r));
  StateHandle const state = _scene->CaptureState(test::ExpectOK{});

  friction[0] = {};
  actor->SetArticulatedJointFrictionParams(friction, test::ExpectOK{});
  EXPECT_GT(GetAnchors(actor).Rows(), 0);
  Step(1);
  EXPECT_EQ(GetAnchors(actor).Rows(), 0);
  _scene->RestoreState(state, /*releaseImmediately*/ true, test::ExpectOK{});
  EXPECT_GT(GetAnchors(actor).Rows(), 0);
  Step(1);
  EXPECT_EQ(GetAnchors(actor).Rows(), 0);
}

// Builds the assembly problem of the only island of a scene, at the end of its last step.
static SnleProblem<real> InitAssembly(entt::registry& reg, entt::entity& outIsland) {
  outIsland = reg.view<CIslandDescendants const>().front();
  return solver::InitIslandAssembly(
      reg, outIsland, reg.get<CIslandDescendants const>(outIsland), /*syncCurrentState*/ true);
}

// The trust margin a problem reports after an assembly at its current solution.
static real AssembledTrustMargin(SnleProblem<real>& problem, AssemblyParams const& params) {
  problem.InvalidateCachedData();
  problem.UpdateObjResDRes(params);
  return problem.GetTrustMargin();
}

// Adds an increment to one DoF of a problem's solution.
static void
MoveDof(entt::registry& reg, entt::entity island, SnleProblem<real>& problem, int dof, real d) {
  ColumnVector<real> increment = ColumnVector<real>::Zero(problem.GetDofsSize());
  increment[dof] = d;
  solver::PostNewIncrementLocalPipeline(reg, island, AsConstView(increment), problem.solution);
}

// With the exact joint friction Hessian, the island assemblies report the trust margin of the
// joint: measured from the stage start, where the held joint is ahead of its anchor by its held
// displacement h, a displacement d along the load gives 1 + (h + d) / δ, which is zero at reverse
// yield. The fitted Hessian, an assembly of the Hessian alone, and fixing all the joint's DoFs
// report none.
TEST_P(JointFrictionTest, ReportsTrustMarginWithExactHessian) {
  Actor* actor = CreateJointActor(kFriction);
  SetLoad(actor, 0.8_r * kStaticPeak);
  Step(300);
  real const delta = kFriction.preslidingDisplacement;
  real const held = GetPose(actor)[0];
  ASSERT_GT(held, 0.1_r * delta);

  auto& reg = GetRegistry();
  entt::entity island = entt::null;
  SnleProblem<real> problem = InitAssembly(reg, island);
  // The stage start is the start of the last step, where the joint was held too, up to the
  // convergence tolerance of the solver.
  real constexpr kTol = 5e-2_r;
  real displacement = 0_r;
  for (real const d : {0_r, -held - delta, -held - 2_r * delta}) {
    MoveDof(reg, island, problem, 0, d - displacement);
    displacement = d;
    real const expected = 1_r + (held + d) / delta;
    EXPECT_NEAR(AssembledTrustMargin(problem, {.assemObj = true}), expected, kTol);
    EXPECT_NEAR(
        AssembledTrustMargin(problem, {.assemObj = false, .assemRes = true}), expected, kTol);
  }

  EXPECT_EQ(
      AssembledTrustMargin(
          problem,
          {.assemObj = true,
           .fittedSaturationHessian =
               {.contactFriction = false, .jointFriction = true, .constraintSaturation = false}}),
      kInf);
  EXPECT_EQ(AssembledTrustMargin(problem, {.assemObj = false, .assemDRes = true}), kInf);

  auto& dofIndices = reg.get<CDirichletBC<real>>(GetEntity(actor)).dofIndices;
  for (int dof = 0; dof < actor->GetNumDofs(); ++dof) {
    dofIndices.push_back(dof);
  }
  EXPECT_EQ(AssembledTrustMargin(problem, {.assemObj = true}), kInf);
}

// After a spherical joint, whose pose takes 4 values for 3 DoFs, a joint's pose and DoF offsets
// differ. Both joints hold a load below the static peak, so their margins exceed 1, until the child
// moves 3 presliding displacements against its load, past reverse yield: then its margin is below
// -1. Fixing the DoFs of either joint drops its margin only.
TEST_P(JointFrictionTest, TrustMarginAfterSphericalJoint) {
  ArticulatedActorParams params;
  params.joints = {
      {.type = ArticulatedJointType::Spherical, .friction = kFriction},
      {.type = GetParam(), .axis = Real3{1_r, 0_r, 0_r}, .friction = kFriction}};
  for (int parent : {-1, 0}) {
    params.links.push_back(
        {.parentLink = parent,
         .parentJointFromLink = TransformRT{Real3{-0.5_r, -0.5_r, -0.5_r}},
         .shape = test::CreateUnitCubeTetMeshShape(_mochiContext),
         .colliderType = ColliderType::None});
  }
  params.links[1].parentJointFromLink = TransformRT{};
  Actor* actor = _scene->CreateArticulatedActor(params, test::ExpectOK{});
  int constexpr kChildDof = 3;
  std::vector<int> const dofs = {0, kChildDof};
  std::vector<real> const loads(2, 0.8_r * kStaticPeak);
  actor->SetExternalForcesOnDofs(dofs, loads, test::ExpectOK{});
  Step(300);

  auto& reg = GetRegistry();
  entt::entity island = entt::null;
  SnleProblem<real> problem = InitAssembly(reg, island);
  EXPECT_GT(AssembledTrustMargin(problem, {.assemObj = true}), 1_r);
  MoveDof(reg, island, problem, kChildDof, -3_r * kFriction.preslidingDisplacement);
  EXPECT_LT(AssembledTrustMargin(problem, {.assemObj = true}), -1_r);

  auto& dofIndices = reg.get<CDirichletBC<real>>(GetEntity(actor)).dofIndices;
  dofIndices.clear();
  for (int dof = kChildDof; dof < actor->GetNumDofs(); ++dof) {
    dofIndices.push_back(dof);
  }
  EXPECT_GT(AssembledTrustMargin(problem, {.assemObj = true}), 1_r);
  dofIndices = {0, 1, 2};
  EXPECT_LT(AssembledTrustMargin(problem, {.assemObj = true}), -1_r);
}

// The deprecated falloffVel is accepted, with a warning (logged once per process), and dropped.
class JointFrictionDeprecationTest : public test::MochiSceneTestBase {};

TEST_F(JointFrictionDeprecationTest, FalloffVelIsIgnored) {
  ArticulatedJointFrictionParams friction = kFriction;
  friction.falloffVel = 1e-3_r;
  ArticulatedActorParams params;
  params.joints = {
      {.type = ArticulatedJointType::Revolute, .axis = Real3{1_r, 0_r, 0_r}, .friction = friction}};
  params.links = {
      {.parentLink = -1,
       .shape = test::CreateUnitCubeTetMeshShape(_mochiContext),
       .colliderType = ColliderType::None}};
  Actor* actor = nullptr;
  {
    test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
    actor = _scene->CreateArticulatedActor(params, test::ExpectOK{});
  }
  EXPECT_EQ(actor->GetArticulatedJointFrictionParams(test::ExpectOK{})[0], kFriction);

  // Parameters that differ from the stored ones only by falloffVel are no change: they do not wake
  // the actor.
  experimental::SetSleepParams(
      _scene, experimental::SleepParams{.canSleep = true}, test::ExpectOK{});
  _scene->Step(1e-2_r);
  auto& reg = GetRegistry();
  entt::entity const island = reg.view<CIslandDescendants const>().front();
  sleep::ForceSleep(reg, island);
  DynamicArray<ArticulatedJointFrictionParams> const newFriction(1, friction);
  actor->SetArticulatedJointFrictionParams(newFriction, test::ExpectOK{});
  EXPECT_EQ(actor->GetArticulatedJointFrictionParams(test::ExpectOK{})[0], kFriction);
  _scene->Step(1e-2_r);
  EXPECT_FALSE(sleep::IsIslandAwake(reg, island));
}
