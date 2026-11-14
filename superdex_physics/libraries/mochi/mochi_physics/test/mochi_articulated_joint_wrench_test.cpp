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

#include "mochi_articulated_test_scenes.h"
#include "mochi_physics_test_fixture.h"

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/src/mochi_articulated_body.h>
#include <mochi_physics/src/mochi_common_components.h>
#include <mochi_physics/src/mochi_island.h>
#include <mochi_physics/src/mochi_solve.h>

#include <gtest/gtest.h>

#include <array>
#include <functional>
#include <string>
#include <utility>

// Tests of QueryType::ArticulatedJointWrenches, in three families:
// - Query API: support, lifecycle and errors of the query and its getters.
// - Closed-form wrenches: simple actors whose wrenches are known in closed form, with every
//   integrator for which they hold.
// - Configuration matrix: diverse articulated actors, covering every source of link-space residual
//   terms. The wrenches match finite differences of the merit, and do not depend on threading.

using namespace mochi;

namespace {

/**************************************************************************************************
  Shared helpers
*/

struct JointWrenches {
  DynamicArray<Real3> forces;
  DynamicArray<Real3> torques;
};

JointWrenches GetWrenches(Actor const* actor, Error& error) {
  int const numLinks = isize(actor->GetNestedLinkActors(test::ExpectOK{}));
  JointWrenches wrenches{
      .forces = DynamicArray<Real3>(numLinks), .torques = DynamicArray<Real3>(numLinks)};
  actor->GetArticulatedJointWrenches(wrenches.forces, wrenches.torques, error);
  return wrenches;
}

Actor* GetLink(Scene* scene, Actor const* actor, int link) {
  return scene->GetActor(actor->GetNestedLinkActors(test::ExpectOK{})[link]);
}

// World position of a joint origin on the child side, from the joint's child link.
Real3 GetJointOrigin(Actor const* link, TransformRT const& jointFromLink) {
  return (link->GetRootTransform() * Invert(jointFromLink)).GetTranslation();
}

// Weight of an actor. For articulated actors, the weight of all its links.
real GetWeight(Actor const* actor) {
  return actor->GetMass(test::ExpectOK{}) * Norm(kDefaultGravity);
}

// Sum over the subtree of each joint's link of the given per-link wrenches (forces, and torques
// about each link's center of mass), with the torques moved to the joint origin. Link i is in the
// subtree of joint j if j is i or one of its ancestors.
JointWrenches
SumSubtreeWrenches(Scene const* scene, Actor const* actor, JointWrenches const& linkWrenches) {
  auto const links = actor->GetNestedLinkActors(test::ExpectOK{});
  auto const info = actor->GetArticulatedShapeInfo(test::ExpectOK{});
  int const numLinks = isize(links);
  JointWrenches sums{
      .forces = DynamicArray<Real3>(numLinks, Real3{}),
      .torques = DynamicArray<Real3>(numLinks, Real3{})};
  // Parents precede their children, so the origins of the ancestors of link i are known.
  DynamicArray<Real3> origins(numLinks);
  for (int i = 0; i < numLinks; ++i) {
    Actor const* link = scene->GetActor(links[i]);
    origins[i] = GetJointOrigin(link, info.jointFromChildLink[i]);
    Real3 const center = link->GetCenterOfMassTransform(test::ExpectOK{}).GetTranslation();
    for (int j = i; j >= 0; j = info.parents[j]) {
      sums.forces[j] += linkWrenches.forces[i];
      sums.torques[j] +=
          linkWrenches.torques[i] + Cross(center - origins[j], linkWrenches.forces[i]);
    }
  }
  return sums;
}

/**************************************************************************************************
  Query API
*/

class ArticulatedJointWrenchApiTest : public test::MochiSceneTestBase {};

TEST_F(ArticulatedJointWrenchApiTest, QuerySupport) {
  Actor* rigid = test::AddRigidUnitCube(_scene, TransformRT{}, ColliderType::None);
  EXPECT_FALSE(rigid->IsQuerySupported(QueryType::ArticulatedJointWrenches));
  EXPECT_FALSE(
      rigid->RegisterQuery(QueryType::ArticulatedJointWrenches, test::ExpectNotOK{}).IsValid());

  Actor* zeroDofs = test::CreateCubeTree(_scene, {{.type = ArticulatedJointType::Hard}});
  EXPECT_FALSE(zeroDofs->IsQuerySupported(QueryType::ArticulatedJointWrenches));

  Actor* actor = test::CreateCubeTree(
      _scene, {{.type = ArticulatedJointType::Hard}, {.type = ArticulatedJointType::Spherical}});
  EXPECT_TRUE(actor->IsQuerySupported(QueryType::ArticulatedJointWrenches));
  EXPECT_FALSE(GetLink(_scene, actor, 1)->IsQuerySupported(QueryType::ArticulatedJointWrenches));
}

TEST_F(ArticulatedJointWrenchApiTest, GetterLifecycle) {
  Actor* actor = test::CreateCubeTree(
      _scene, {{.type = ArticulatedJointType::Hard}, {.type = ArticulatedJointType::Spherical}});
  real constexpr kDt = 1e-2_r;
  std::array<Real3, 2> forces{};
  std::array<Real3, 2> torques{};
  ASSERT_EQ(isize(forces), isize(actor->GetNestedLinkActors(test::ExpectOK{})));

  // Not registered.
  actor->GetArticulatedJointWrenches(forces, torques, test::ExpectNotOK{});

  // Registered, but not computed until the next step.
  QueryHandle const query =
      actor->RegisterQuery(QueryType::ArticulatedJointWrenches, test::ExpectOK{});
  QueryHandle const query2 =
      actor->RegisterQuery(QueryType::ArticulatedJointWrenches, test::ExpectOK{});
  actor->GetArticulatedJointWrenches(forces, torques, test::ExpectNotOK{});

  _scene->Step(kDt);
  actor->GetArticulatedJointWrenches(forces, torques, test::ExpectOK{});

  // Wrong sizes.
  std::array<Real3, 1> tooShort{};
  actor->GetArticulatedJointWrenches(tooShort, torques, test::ExpectNotOK{});
  actor->GetArticulatedJointWrenches(forces, tooShort, test::ExpectNotOK{});

  // Not an articulated actor.
  GetLink(_scene, actor, 1)->GetArticulatedJointWrenches(forces, torques, test::ExpectNotOK{});

  // Registrations are reference counted.
  actor->CancelQuery(query);
  actor->GetArticulatedJointWrenches(forces, torques, test::ExpectOK{});
  actor->CancelQuery(query2);
  actor->GetArticulatedJointWrenches(forces, torques, test::ExpectNotOK{});
}

// The wrench of each joint matches that of all joints. Cycle joints are not supported.
TEST_F(ArticulatedJointWrenchApiTest, PerJointWrenchMatchesAllJoints) {
  Actor* actor = test::CreateParallelogramLinkage(_scene, /*useCycleJoint*/ true, 0_r);
  auto const info = actor->GetArticulatedShapeInfo(test::ExpectOK{});
  int const numLinks = isize(info.parents);
  int const numJoints = isize(info.jointTypes);
  ASSERT_GT(numJoints, numLinks);

  EXPECT_EQ(Real3{}, actor->GetArticulatedJointForce(1, test::ExpectNotOK{}));
  actor->RegisterQuery(QueryType::ArticulatedJointWrenches, test::ExpectOK{});
  EXPECT_EQ(Real3{}, actor->GetArticulatedJointForce(1, test::ExpectNotOK{}));
  EXPECT_EQ(Real3{}, actor->GetArticulatedJointTorque(1, test::ExpectNotOK{}));
  _scene->Step(1e-2_r);

  auto const wrenches = GetWrenches(actor, test::ExpectOK{});
  for (int j = 0; j < numLinks; ++j) {
    EXPECT_EQ(wrenches.forces[j], actor->GetArticulatedJointForce(j, test::ExpectOK{}));
    EXPECT_EQ(wrenches.torques[j], actor->GetArticulatedJointTorque(j, test::ExpectOK{}));
  }
  EXPECT_GT(Norm(wrenches.forces[1]), 0_r);

  for (int const invalidJoint : {-1, numLinks, numJoints}) {
    EXPECT_EQ(Real3{}, actor->GetArticulatedJointForce(invalidJoint, test::ExpectNotOK{}));
    EXPECT_EQ(Real3{}, actor->GetArticulatedJointTorque(invalidJoint, test::ExpectNotOK{}));
  }
  Actor const* link = GetLink(_scene, actor, 1);
  EXPECT_EQ(Real3{}, link->GetArticulatedJointForce(0, test::ExpectNotOK{}));
  EXPECT_EQ(Real3{}, link->GetArticulatedJointTorque(0, test::ExpectNotOK{}));
}

/**************************************************************************************************
  Closed-form wrenches
*/

// Each joint of a static actor supports the weight of its subtree:
// F_j = -Σ m_i g, τ_j = Σ (c_i - p_j) × (-m_i g).
void ExpectSupportsSubtreeWeight(Scene const* scene, Actor const* actor) {
  auto const links = actor->GetNestedLinkActors(test::ExpectOK{});
  int const numLinks = isize(links);
  // Each link is supported against its weight, with no torque about its center of mass.
  JointWrenches supports{
      .forces = DynamicArray<Real3>(numLinks), .torques = DynamicArray<Real3>(numLinks, Real3{})};
  for (int i = 0; i < numLinks; ++i) {
    supports.forces[i] = -scene->GetActor(links[i])->GetMass(test::ExpectOK{}) * kDefaultGravity;
  }
  auto const expected = SumSubtreeWrenches(scene, actor, supports);
  auto const wrenches = GetWrenches(actor, test::ExpectOK{});
  real const forceTol = 1e-4_r * GetWeight(actor);
  for (int j = 0; j < numLinks; ++j) {
    EXPECT_NEAR_TOL(expected.forces[j], wrenches.forces[j], forceTol);
    EXPECT_NEAR_TOL(expected.torques[j], wrenches.torques[j], 5_r * forceTol);
  }
}

class ArticulatedJointWrenchIntegratorTest
    : public test::MochiSceneTestBase,
      public ::testing::WithParamInterface<IntegrationMethod> {};

// Integrators whose last stage is the end of the step. The query holds the force balance of the
// last stage, so only these integrators give the forces at the end of the step.
using ArticulatedJointWrenchStepEndIntegratorTest = ArticulatedJointWrenchIntegratorTest;

std::string IntegratorName(::testing::TestParamInfo<IntegrationMethod> const& info) {
  return std::string(SReflect::EnumToString(info.param));
}

// The static chain stays static, and each joint supports the weight of its subtree, with every
// integrator. The same holds with a massless dummy link (without a shape) inside the chain. The
// chains have no colliders, so they do not interact.
TEST_P(ArticulatedJointWrenchIntegratorTest, StaticChainSupportsSubtreeWeight) {
  test::SetSceneIntegrationMethod(_scene, GetParam());
  int constexpr kDummyLink = 2;
  std::array const actors{
      test::CreateStaticChain(_scene), test::CreateStaticChain(_scene, kDummyLink)};
  for (Actor* actor : actors) {
    actor->RegisterQuery(QueryType::ArticulatedJointWrenches, test::ExpectOK{});
  }
  for (int step = 0; step < 3; ++step) {
    _scene->Step(1e-2_r);
  }

  EXPECT_EQ(0_r, GetLink(_scene, actors[1], kDummyLink)->GetMass(test::ExpectOK{}));
  for (Actor const* actor : actors) {
    SCOPED_TRACE(actor == actors[0] ? "without dummy link" : "with dummy link");
    ExpectSupportsSubtreeWeight(_scene, actor);
  }
}

// A horizontal arm on a revolute joint about a horizontal axis is held static by an external DoF
// torque that balances the moment of its weight. The joint supports the weight of the arm, and the
// torque along its axis is the holding torque.
TEST_P(ArticulatedJointWrenchIntegratorTest, HorizontalArmHeldByDofTorque) {
  test::SetSceneIntegrationMethod(_scene, GetParam());
  Real3 const axis{0_r, 0_r, 1_r};
  Actor* actor = test::CreateCubeTree(
      _scene,
      {{.type = ArticulatedJointType::Hard},
       test::MakeRevoluteJoint(Real3{1_r, 0_r, 0_r}, {}, axis)},
      {.parentJointFromLinks = {TransformRT{}, TransformRT{Real3{1_r, -0.5_r, -0.5_r}}}});
  Actor const* arm = GetLink(_scene, actor, 1);
  Real3 const lever = arm->GetCenterOfMassTransform(test::ExpectOK{}).GetTranslation() -
      GetJointOrigin(arm, actor->GetArticulatedShapeInfo(test::ExpectOK{}).jointFromChildLink[1]);
  real const holdingTorque =
      Dot(axis, Cross(lever, -arm->GetMass(test::ExpectOK{}) * kDefaultGravity));
  std::array const dofs{0};
  std::array const torques{holdingTorque};
  actor->SetExternalForcesOnDofs(dofs, torques, test::ExpectOK{});
  actor->RegisterQuery(QueryType::ArticulatedJointWrenches, test::ExpectOK{});
  for (int step = 0; step < 3; ++step) {
    _scene->Step(1e-2_r);
  }

  ExpectSupportsSubtreeWeight(_scene, actor);
  EXPECT_GT(Abs(holdingTorque), GetWeight(arm));
  EXPECT_NEAR(
      holdingTorque,
      Dot(axis, actor->GetArticulatedJointTorque(1, test::ExpectOK{})),
      5e-4_r * GetWeight(actor));
}

// Nothing holds a free-floating chain, so its root wrench vanishes at convergence, even while its
// links accelerate relative to each other.
TEST_P(ArticulatedJointWrenchIntegratorTest, FreeRootWrenchVanishes) {
  test::SetSceneIntegrationMethod(_scene, GetParam());
  // The root wrench is the residual of the root DoFs, so converge it to round-off: the solve stops
  // when the Newton step is negligible.
  auto params = _scene->GetSolverParams();
  params.nonLinearSolver.maxIter = 20;
  params.nonLinearSolver.absTol = 0_r;
  params.nonLinearSolver.relTol = 0_r;
  _scene->SetSolverParams(params, test::ExpectOK{});
  Actor* actor = test::CreateCubeTree(
      _scene,
      {{.type = ArticulatedJointType::Free},
       {.type = ArticulatedJointType::Spherical,
        .parentLinkFromJoint = TransformRT{Real3{1_r, 0.5_r, 0_r}}},
       test::MakeRevoluteJoint(Real3{0_r, 1_r, 0.5_r})});
  std::array const velocities{0.3_r, -0.2_r, 0.1_r, 0.5_r, -0.4_r, 0.2_r, 1_r, -2_r, 0.5_r, 3_r};
  ASSERT_EQ(isize(velocities), actor->GetNumDofs());
  actor->SetArticulatedJointVelocities(velocities, test::ExpectOK{});
  actor->RegisterQuery(QueryType::ArticulatedJointWrenches, test::ExpectOK{});
  _scene->Step(1e-2_r);

  auto const wrenches = GetWrenches(actor, test::ExpectOK{});
  real const forceScale = GetWeight(actor);
  // Multi-stage and midpoint integrators converge the root residual less tightly in single
  // precision.
  real const rootTol = (MOCHI_USE_DOUBLE_PRECISION ? 1e-4_r : 5e-3_r) * forceScale;
  EXPECT_NEAR_TOL(Real3{}, wrenches.forces[0], rootTol);
  EXPECT_NEAR_TOL(Real3{}, wrenches.torques[0], rootTol);

  // The inner joints do transmit forces.
  EXPECT_GT(Norm(wrenches.forces[1]), 1e-2_r * forceScale);
}

INSTANTIATE_TEST_SUITE_P(
    Integrators,
    ArticulatedJointWrenchIntegratorTest,
    ::testing::Values(
        IntegrationMethod::BackwardEuler,
        IntegrationMethod::BDF2,
        IntegrationMethod::BDF3,
        IntegrationMethod::DIRK22,
        IntegrationMethod::DIRK23,
        IntegrationMethod::DIRK33,
        IntegrationMethod::SymplecticDIRK12,
        IntegrationMethod::SymplecticDIRK22),
    IntegratorName);

// A link spinning about a fixed vertical axis, without gravity, is held by the centripetal force
// F = -m ω² r⊥, with r⊥ the radial vector from the axis to its center of mass. The cube's inertia
// is isotropic, so the torque about the joint is (c - p) × F. The first steps are skipped, because
// the discrete acceleration differs from the centripetal one in the first step after setting the
// velocity, and multistep integrators start with lower orders.
TEST_P(ArticulatedJointWrenchStepEndIntegratorTest, SpinningLinkCentripetalForce) {
  test::SetSceneIntegrationMethod(_scene, GetParam());
  _scene->SetGravity(Real3{});
  Real3 const axis{0_r, 1_r, 0_r};
  Actor* actor = test::CreateCubeTree(
      _scene,
      {{.type = ArticulatedJointType::Hard},
       test::MakeRevoluteJoint(Real3{1_r, 0_r, 0_r}, {}, axis)},
      {.parentJointFromLinks = {TransformRT{}, TransformRT{Real3{1.5_r, 0.4_r, 0_r}}}});
  std::array const initialVelocity{2_r};
  actor->SetArticulatedJointVelocities(initialVelocity, test::ExpectOK{});
  actor->RegisterQuery(QueryType::ArticulatedJointWrenches, test::ExpectOK{});
  for (int step = 0; step < 10; ++step) {
    _scene->Step(1e-2_r);
  }

  std::array<real, 1> velocity{};
  actor->GetArticulatedJointVelocities(velocity, test::ExpectOK{});
  real const rate = velocity[0];
  auto const wrenches = GetWrenches(actor, test::ExpectOK{});
  Actor const* link = GetLink(_scene, actor, 1);
  Real3 const lever = link->GetCenterOfMassTransform(test::ExpectOK{}).GetTranslation() -
      GetJointOrigin(link, actor->GetArticulatedShapeInfo(test::ExpectOK{}).jointFromChildLink[1]);
  Real3 const radial = lever - Dot(lever, axis) * axis;
  Real3 const expectedForce = -link->GetMass(test::ExpectOK{}) * Sqr(rate) * radial;
  Real3 const expectedTorque = Cross(lever, expectedForce);

  // After the first step, the discrete acceleration differs from the continuous one by O((ω h)²).
  real const tol = 3e-2_r * Norm(expectedForce);
  EXPECT_NEAR_TOL(expectedForce, wrenches.forces[1], tol);
  EXPECT_NEAR_TOL(expectedTorque, wrenches.torques[1], tol * Norm(lever));
  EXPECT_LE(Abs(Dot(wrenches.torques[1], axis)), tol * Norm(lever));
}

INSTANTIATE_TEST_SUITE_P(
    Integrators,
    ArticulatedJointWrenchStepEndIntegratorTest,
    ::testing::Values(
        IntegrationMethod::BackwardEuler,
        IntegrationMethod::BDF2,
        IntegrationMethod::BDF3,
        IntegrationMethod::DIRK22,
        IntegrationMethod::DIRK33),
    IntegratorName);

class ArticulatedJointWrenchSleepTest : public test::MochiSceneTestBase {};

// Sleeping islands keep the wrenches of their last step.
TEST_F(ArticulatedJointWrenchSleepTest, SleepingIslandKeepsWrenches) {
  test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
  experimental::SetSleepParams(
      _scene, {.canSleep = true, .minStepsBeforeSleep = 2}, test::ExpectOK{});
  Actor* actor = test::CreateStaticChain(_scene);
  actor->RegisterQuery(QueryType::ArticulatedJointWrenches, test::ExpectOK{});
  _scene->Step(1e-2_r);
  JointWrenches const awake = GetWrenches(actor, test::ExpectOK{});
  for (int step = 0; step < 20 && actor->GetConvergenceStatus() != ConvergenceStatus::None;
       ++step) {
    _scene->Step(1e-2_r);
  }
  ASSERT_EQ(ConvergenceStatus::None, actor->GetConvergenceStatus()) << "The island must sleep.";
  JointWrenches const asleep = GetWrenches(actor, test::ExpectOK{});
  real const forceTol = 1e-4_r * GetWeight(actor);
  for (int i = 0; i < isize(awake.forces); ++i) {
    EXPECT_NEAR_TOL(awake.forces[i], asleep.forces[i], forceTol);
    EXPECT_NEAR_TOL(awake.torques[i], asleep.torques[i], 5_r * forceTol);
  }
  ExpectSupportsSubtreeWeight(_scene, actor);
}

/**************************************************************************************************
  Configuration matrix

  With settings under which the residual is the gradient of the merit, the wrench of joint j equals
  the sum over the subtree of link j of the gradient of the merit with respect to the link poses,
  computed by finite differences. This holds at any state, and only if every link-dependent term
  is in the link-space residual, with the right frame, sign and reference point. The wrenches also
  do not depend on threading, although each configuration exercises different tasks and
  semaphores.
*/

constexpr real kDt = 1e-2_r;
constexpr real kJointDamping = 1e4_r;
constexpr real kConstraintStiffness = 1e6_r;
// Stiff enough for backward Euler to damp the constraint oscillations within a few dozen steps.
constexpr real kSettlingConstraintStiffness = 1e7_r;
// Relative tolerance of wrenches computed with and without worker threads.
constexpr real kThreadingTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-7_r : 2e-3_r;
// Finite differences of the merit: step [m] or [rad], and tolerance relative to the force scale.
// The step balances the truncation error of contact and the round-off error of soft actors.
constexpr real kFiniteDiffStep = 1e-6_r;
constexpr real kFiniteDiffTol = 1e-5_r;

// Joint-space terms of the cases.
constexpr real kCaseDamping = 3e3_r;
constexpr real kCaseInertia = 500_r;
constexpr real kCaseCoulomb = 2e3_r;
constexpr std::array<real, 2> kCaseDofForces{4e3_r, -2.5e3_r};
constexpr std::array<real, 2> kCaseTransmissionCoefficients{0.5_r, -0.3_r};
constexpr real kCaseTransmissionForce = 4e3_r;

// Weight of the actor's links, or of a 1000 kg body if lighter.
real ForceScale(Actor const* actor) {
  return Max(GetWeight(actor), 1e3_r * Norm(kDefaultGravity));
}

test::ThreeBoneOptions const kSteppableThreeBone{
    .linkDensity = 1e3_r,
    .softYoungsModulus = 1e5_r,
    .softHasInertia = true};

// A joint with an oblique axis, so that the topology cases do not depend on axis alignment.
ArticulatedJointParams Joint(ArticulatedJointType type, Real3 const& offset = {}) {
  return {
      .type = type,
      .parentLinkFromJoint = TransformRT{offset},
      .axis = Normalize(Real3{1_r, 0.5_r, 0.2_r})};
}

// Sets the current pose of a dynamic link, i.e., the world transform of its center of mass frame,
// without changing the joint DoFs. Updates the components that depend on the pose of the link: the
// link transform and full pose of the articulated actor, and the rigid state and root transform of
// the link.
void SetCurrentLinkPose(entt::registry& reg, entt::entity link, TransformRT const& worldFromLink) {
  entt::entity const articulated = reg.get<CArticulatedEntity const>(link).entity;
  int const poseOffset = reg.get<CDofOffset const>(link).poseOffset;
  TransformToRawPose(
      worldFromLink,
      reg.get<CArticulatedFullPose>(articulated)
          .value.Slice<RigidSize::kAll>(poseOffset, RigidSize::kAll));
  // The full pose holds the poses of the links in order.
  reg.get<CArticulatedLinkTransforms<TimeStep::Current>>(
      articulated)[poseOffset / RigidSize::kAll] = worldFromLink;
  ecs::InvokeOnEntity(&articulated::rigid::EntityPostNewSolution, reg, link);
}

// Compares the joint wrenches to the subtree sums of the given per-link wrenches (forces, and
// torques about each link's center of mass). Joints of static links are skipped: their subtrees
// hold static links, which can't be moved. Links are static only if their parent is, so the
// subtree of a dynamic link holds no static links.
void CheckSubtreeSums(
    Scene const* scene,
    Actor const* actor,
    JointWrenches const& wrenches,
    JointWrenches const& linkWrenches,
    real tol) {
  auto const expected = SumSubtreeWrenches(scene, actor, linkWrenches);
  auto const links = actor->GetNestedLinkActors(test::ExpectOK{});
  real const scale = ForceScale(actor);
  for (int j = 0; j < isize(links); ++j) {
    if (scene->GetActor(links[j])->IsStatic()) {
      continue;
    }
    EXPECT_LE(Norm(expected.forces[j] - wrenches.forces[j]), tol * scale) << "Force, joint " << j;
    EXPECT_LE(Norm(expected.torques[j] - wrenches.torques[j]), tol * scale * 4_r)
        << "Torque, joint " << j;
  }
}

// Gradient of the merit with respect to the pose of each dynamic link, by central differences in
// the convention of the link residual: translation of the center of mass, and world-frame rotation
// exp(δ) applied on the left, with torque about the center of mass. Zero for static links.
JointWrenches
ComputeMeritLinkGradients(Scene* scene, Actor const* actor, SnleProblem<real>& problem) {
  auto& reg = test::GetRegistry(scene);
  auto merit = [&problem]() {
    problem.InvalidateCachedData();
    problem.UpdateObjResDRes({.assemObj = true, .assemRes = false, .assemDRes = false});
    return problem.GetObjective();
  };
  auto const links = actor->GetNestedLinkActors(test::ExpectOK{});
  int const numLinks = isize(links);
  auto const& linkTxs = reg.get<CArticulatedLinkTransforms<TimeStep::Current> const>(
      GetEntity(reg, actor->GetHandle(), test::ExpectOK{}));
  JointWrenches gradients{
      .forces = DynamicArray<Real3>(numLinks, Real3{}),
      .torques = DynamicArray<Real3>(numLinks, Real3{})};
  for (int i = 0; i < numLinks; ++i) {
    entt::entity const link = GetEntity(reg, links[i], test::ExpectOK{});
    if (reg.all_of<TagStaticActor>(link)) {
      continue;
    }
    TransformRT const pose = linkTxs[i];
    for (int k = 0; k < RigidSize::kDAll; ++k) {
      auto perturbed = [&](real step) {
        Real3 const delta = step * BasisVector<real, 3>(k % 3);
        TransformRT out = pose;
        if (k < RigidSize::kDTrans) {
          out.SetTranslation(pose.GetTranslation() + delta);
        } else {
          out.SetRotation(Quaternion::FromRotationVector(delta) * pose.GetRotation());
        }
        return out;
      };
      SetCurrentLinkPose(reg, link, perturbed(kFiniteDiffStep));
      double const meritPlus = merit();
      SetCurrentLinkPose(reg, link, perturbed(-kFiniteDiffStep));
      double const meritMinus = merit();
      real const derivative = static_cast<real>((meritPlus - meritMinus) / (2.0 * kFiniteDiffStep));
      (k < RigidSize::kDTrans ? gradients.forces : gradients.torques)[i][k % 3] = derivative;
    }
    SetCurrentLinkPose(reg, link, pose);
  }
  return gradients;
}

struct WrenchCase {
  std::string name;
  std::function<Actor*(Context*, Scene*)> build;
  int numSteps = 1;
  // Whether its island steps with worker threads outside TSAN, i.e., it does not satisfy
  // island::ShouldRunSingleThreaded.
  bool usesWorkerThreads = false;
};

WrenchCase SoftSkinnedCase(char const* name, test::ThreeBoneSoftSkinnedMode mode) {
  return {
      .name = name,
      .build =
          [mode](Context* context, Scene* scene) {
            test::AddThreeBoneSkinCollider(scene, /*isStatic*/ true);
            return test::CreateThreeBoneSoftSkinnedActor(
                context,
                scene,
                mode,
                TransformRT::Identity(),
                0_r,
                0_r,
                0_r,
                ActorBoundaryElementType::Default,
                kSteppableThreeBone);
          },
      .usesWorkerThreads = true};
}

DynamicArray<WrenchCase> GetWrenchCases() {
  using test::ThreeBoneSoftSkinnedMode;
  using JT = ArticulatedJointType;
  DynamicArray<WrenchCase> cases;

  // Topologies without joint-space terms.
  cases.push_back(
      {.name = "FreeChain",
       .build =
           [](Context* /*context*/, Scene* scene) {
             Actor* actor = test::CreateCubeTree(
                 scene,
                 {Joint(JT::Free),
                  Joint(JT::Spherical, Real3{1.1_r, 0_r, 0_r}),
                  Joint(JT::Revolute, Real3{0_r, 1.1_r, 0_r}),
                  Joint(JT::Prismatic, Real3{0_r, 0_r, 1.1_r})});
             std::array const velocities{
                 0.3_r, -0.2_r, 0.1_r, 0.5_r, -0.4_r, 0.2_r, 1_r, -2_r, 0.5_r, 3_r, -0.5_r};
             actor->SetArticulatedJointVelocities(velocities, test::ExpectOK{});
             return actor;
           },
       .numSteps = 2});
  cases.push_back(
      {.name = "TreeWithHardJoints",
       .build =
           [](Context* /*context*/, Scene* scene) {
             return test::CreateCubeTree(
                 scene,
                 {Joint(JT::Hard),
                  Joint(JT::Spherical, Real3{1.1_r, 0_r, 0_r}),
                  Joint(JT::Revolute, Real3{0_r, 1.1_r, 0_r}),
                  Joint(JT::Prismatic, Real3{0_r, -1.1_r, 0_r}),
                  Joint(JT::Revolute, Real3{-1.1_r, 0_r, 0_r}),
                  Joint(JT::Revolute, Real3{0_r, 0_r, 1.1_r}),
                  Joint(JT::Spherical, Real3{1.1_r, 0_r, 0.5_r}),
                  Joint(JT::Spherical, Real3{0.5_r, 1.1_r, 0_r}),
                  Joint(JT::Hard, Real3{0_r, 1.1_r, 0_r})},
                 {.parents = {-1, 0, 1, 1, 0, 4, 3, 6, 7}});
           },
       .numSteps = 2});

  // Contact.
  cases.push_back(
      {.name = "LinkContactWithGround",
       .build =
           [](Context* /*context*/, Scene* scene) {
             test::AddFlatGround(scene);
             return test::CreateBridgeOnGround(scene, kJointDamping);
           },
       .numSteps = 100});
  cases.push_back(
      {.name = "LinkSelfContact",
       .build = [](Context* /*context*/,
                   Scene* scene) { return test::CreateSelfContactArm(scene, kJointDamping); },
       .numSteps = 400});
  cases.push_back(
      {.name = "LinkContactWithArticulation",
       .build =
           [](Context* /*context*/, Scene* scene) {
             // A free box link of another articulation falls onto the actor's box link.
             Actor* actor = test::CreateCubeTree(
                 scene,
                 {Joint(JT::Hard), test::MakeRevoluteJoint(Real3{0.5_r, 1.2_r, 0_r})},
                 {.colliderType = ColliderType::Box});
             test::CreateCubeTree(
                 scene,
                 {Joint(JT::Free)},
                 {.worldFromRoot = TransformRT{Real3{0.6_r, 2.18_r, 0.2_r}},
                  .colliderType = ColliderType::Box});
             return actor;
           },
       .numSteps = 3});
  cases.push_back(
      {.name = "SkinContactAsync", .build = [](Context* context, Scene* scene) {
         test::AddThreeBoneSkinCollider(scene, /*isStatic*/ true);
         return test::CreateThreeBoneSkinnedActor(
             context, scene, 0_r, 0_r, 0_r, ActorBoundaryElementType::Default, kSteppableThreeBone);
       }});
  cases.push_back(
      {.name = "SkinContactSync", .build = [](Context* context, Scene* scene) {
         test::AddThreeBoneSkinCollider(scene, /*isStatic*/ false);
         return test::CreateThreeBoneSkinnedActor(
             context, scene, 0_r, 0_r, 0_r, ActorBoundaryElementType::Default, kSteppableThreeBone);
       }});
  cases.push_back(SoftSkinnedCase("SoftSkinnedUnposed", ThreeBoneSoftSkinnedMode::Fom));
  cases.push_back(SoftSkinnedCase("SoftSkinnedPosed", ThreeBoneSoftSkinnedMode::PosedFom));
  cases.push_back(SoftSkinnedCase("Blended", ThreeBoneSoftSkinnedMode::BlendedFom));
  cases.push_back(
      SoftSkinnedCase("PartiallyBlended", ThreeBoneSoftSkinnedMode::PartiallyBlendedFom));
#if MOCHI_ENABLE_ROM_ACTORS
  cases.push_back(SoftSkinnedCase("SoftSkinnedRom", ThreeBoneSoftSkinnedMode::Rom));
  cases.push_back(SoftSkinnedCase("BlendedRom", ThreeBoneSoftSkinnedMode::BlendedRom));
#endif // MOCHI_ENABLE_ROM_ACTORS

  // Closed loops and constraints on links.
  cases.push_back(
      {.name = "CycleJoint",
       .build =
           [](Context* /*context*/, Scene* scene) {
             return test::CreateParallelogramLinkage(scene, true, kJointDamping);
           },
       .numSteps = 600});
  cases.push_back(
      {.name = "LinkConstraintToWorld",
       .build =
           [](Context* /*context*/, Scene* scene) {
             Actor* actor = test::CreateParallelogramLinkage(scene, false, kJointDamping);
             Actor const* link3 = GetLink(scene, actor, 3);
             RigidPivotPositionConstraintParams params;
             params.stiffness = kConstraintStiffness;
             params.actor = link3->GetHandle();
             params.localPosition = link3->GetRootTransform().TransformPointInverse(Real3{});
             scene->CreateRigidPivotPositionConstraint(params, test::ExpectOK{});
             return actor;
           },
       .numSteps = 600});
  cases.push_back(
      {.name = "LinkConstraintSameArticulation",
       .build =
           [](Context* /*context*/, Scene* scene) {
             Actor* actor = test::CreateDoublePendulum(scene);
             test::CreateRigidSphericalJointAt(
                 scene,
                 GetLink(scene, actor, 1),
                 GetLink(scene, actor, 2),
                 Real3{2.5_r, 0.8_r, 0.5_r},
                 kConstraintStiffness);
             return actor;
           },
       .numSteps = 2});
  cases.push_back(
      {.name = "MixedConstraintWithRigid",
       .build =
           [](Context* /*context*/, Scene* scene) {
             // A rigid cube hangs from link 1, whose joint rotates about the vertical.
             Actor* actor = test::CreateCubeTree(
                 scene,
                 {Joint(JT::Hard, Real3{0_r, 4_r, 0_r}),
                  test::MakeRevoluteJoint(Real3{1.5_r, 0_r, 0.2_r}, {}, Real3{0_r, 1_r, 0_r})});
             Actor* link1 = GetLink(scene, actor, 1);
             Real3 const anchor =
                 link1->GetRootTransform().TransformPoint(Real3{0.5_r, 0_r, 0.5_r});
             Actor* cube = test::AddRigidUnitCube(
                 scene, TransformRT{anchor - Real3{0.5_r, 1.1_r, 0.5_r}}, ColliderType::None);
             test::CreateRigidSphericalJointAt(
                 scene,
                 link1,
                 cube,
                 anchor - Real3{0_r, 0.05_r, 0_r},
                 kSettlingConstraintStiffness);
             return actor;
           },
       .numSteps = 60});
  cases.push_back(
      {.name = "MixedConstraintTwoArticulations",
       .build =
           [](Context* /*context*/, Scene* scene) {
             // The arm of the actor rotates about the vertical, and holds the link of another
             // articulation, which slides vertically, through a spherical joint.
             Actor* actor = test::CreateCubeTree(
                 scene,
                 {Joint(JT::Hard),
                  test::MakeRevoluteJoint(Real3{1.5_r, 0_r, 0_r}, {}, Real3{0_r, 1_r, 0_r})});
             Actor* other = test::CreateCubeTree(
                 scene,
                 {Joint(JT::Hard, Real3{0_r, 0.5_r, 4_r}),
                  {.type = JT::Prismatic,
                   .parentLinkFromJoint = TransformRT{Real3{1.5_r, 0_r, 0_r}},
                   .axis = Real3{0_r, 1_r, 0_r}}});
             test::CreateRigidSphericalJointAt(
                 scene,
                 GetLink(scene, actor, 1),
                 GetLink(scene, other, 1),
                 Real3{2_r, 0.8_r, 2_r},
                 kSettlingConstraintStiffness);
             return actor;
           },
       .numSteps = 60});
  cases.push_back(
      {.name = "PoseControllerLinks",
       .build =
           [](Context* /*context*/, Scene* scene) {
             Actor* actor = test::CreateDoublePendulum(scene);
             PoseControllerParams params(3);
             for (auto& tracking : params.linkPosTracking) {
               tracking.stiffness = 1e5_r;
             }
             actor->AddArticulatedPoseController(params, test::ExpectOK{});
             return actor;
           },
       .numSteps = 2});
  cases.push_back(
      {.name = "SpatialTendon",
       .build =
           [](Context* /*context*/, Scene* scene) {
             Actor* actor = test::CreateDoublePendulum(scene);
             experimental::SpatialTendonParams tendon;
             tendon.routingElements = {
                 {.index = 0, .localPosition = Real3{0.5_r, 1_r, 0.5_r}},
                 {.index = 1, .localPosition = Real3{0.2_r, 0.8_r, 0.5_r}},
                 {.index = 2, .localPosition = Real3{0.9_r, 0.1_r, 0.5_r}}};
             int const index = experimental::AddSpatialTendon(actor, tendon, test::ExpectOK{});
             experimental::AttachForceControlActuator(
                 actor, index, {.force = 5e3_r}, test::ExpectOK{});
             return actor;
           },
       .numSteps = 2});

  // Joint-space terms.
  cases.push_back(
      {.name = "JointDamping",
       .build =
           [](Context* /*context*/, Scene* scene) {
             return test::CreateDoublePendulum(scene, {.viscous = kCaseDamping});
           },
       .numSteps = 2});
  cases.push_back(
      {.name = "JointInertia",
       .build =
           [](Context* /*context*/, Scene* scene) {
             return test::CreateCubeTree(
                 scene,
                 {Joint(JT::Hard),
                  {.type = JT::Revolute,
                   .parentLinkFromJoint = TransformRT{Real3{1.2_r, 0.5_r, 0_r}},
                   .axis = Real3{0_r, 0_r, 1_r},
                   .inertia = kCaseInertia},
                  {.type = JT::Revolute,
                   .parentLinkFromJoint = TransformRT{Real3{1.1_r, -0.3_r, 0_r}},
                   .axis = Real3{0_r, 0_r, 1_r},
                   .inertia = kCaseInertia}});
           },
       .numSteps = 3});
  cases.push_back(
      {.name = "JointFriction",
       .build =
           [](Context* /*context*/, Scene* scene) {
             return test::CreateDoublePendulum(scene, {.coulomb = kCaseCoulomb});
           },
       .numSteps = 2});
  cases.push_back({.name = "ExternalDofForces", .build = [](Context* /*context*/, Scene* scene) {
                     Actor* actor = test::CreateDoublePendulum(scene);
                     std::array const dofs{0, 1};
                     actor->SetExternalForcesOnDofs(dofs, kCaseDofForces, test::ExpectOK{});
                     return actor;
                   }});
  cases.push_back({.name = "LinearTransmission", .build = [](Context* /*context*/, Scene* scene) {
                     Actor* actor = test::CreateDoublePendulum(scene);
                     experimental::LinearTransmissionParams transmission;
                     transmission.jointIndices = {1, 2};
                     transmission.jointCoefficients = {
                         kCaseTransmissionCoefficients[0], kCaseTransmissionCoefficients[1]};
                     int const index =
                         experimental::AddLinearTransmission(actor, transmission, test::ExpectOK{});
                     experimental::AttachForceControlActuator(
                         actor,
                         index,
                         {.force = kCaseTransmissionForce, .allowCompressiveForce = true},
                         test::ExpectOK{});
                     return actor;
                   }});
  cases.push_back(
      {.name = "JointTargetConstraint", .build = [](Context* /*context*/, Scene* scene) {
         Actor* actor = test::CreateDoublePendulum(scene);
         ArticulatedSingleDofTargetConstraintParams params;
         params.stiffness = 1e5_r;
         params.actor = actor->GetHandle();
         params.jointIndex = 2;
         params.targetValue = 0.3_r;
         scene->CreateArticulatedSingleDofTargetConstraint(params, test::ExpectOK{});
         return actor;
       }});
  cases.push_back(
      {.name = "JointLimits",
       .build =
           [](Context* /*context*/, Scene* scene) {
             return test::CreateCubeTree(
                 scene,
                 {Joint(JT::Hard),
                  {.type = JT::Revolute,
                   .parentLinkFromJoint = TransformRT{Real3{1.2_r, 0.5_r, 0_r}},
                   .axis = Real3{0_r, 0_r, 1_r},
                   .minLimit = Real3{-0.01_r, -0.01_r, -0.01_r},
                   .maxLimit = Real3{0.01_r, 0.01_r, 0.01_r},
                   .limitStiffness = 1e6_r}});
           },
       .numSteps = 5});
  cases.push_back(
      {.name = "PoseControllerJoints",
       .build =
           [](Context* /*context*/, Scene* scene) {
             Actor* actor = test::CreateDoublePendulum(scene);
             PoseControllerParams params(3);
             // Link 0 has a Hard joint, without a controller.
             for (int link = 1; link < 3; ++link) {
               params.jointTracking[link].stiffness = 1e5_r;
               params.jointTracking[link].damping = 1e3_r;
             }
             actor->AddArticulatedPoseController(params, test::ExpectOK{});
             return actor;
           },
       .numSteps = 2});
  return cases;
}

// Outside TSAN, islands that satisfy island::ShouldRunSingleThreaded step single-threaded even with
// worker threads, so only the cases that use worker threads compare different schedules. TSAN
// builds keep worker threads in every island, and run all the cases.
DynamicArray<WrenchCase> GetThreadingCases() {
  DynamicArray<WrenchCase> cases;
  for (WrenchCase& wrenchCase : GetWrenchCases()) {
    if (MOCHI_COMPILER_TSAN || wrenchCase.usesWorkerThreads) {
      cases.push_back(std::move(wrenchCase));
    }
  }
  return cases;
}

std::string WrenchCaseName(::testing::TestParamInfo<WrenchCase> const& info) {
  return info.param.name;
}

class ArticulatedJointWrenchMatrixTest : public test::MochiSceneTestBase,
                                         public ::testing::WithParamInterface<WrenchCase> {
 public:
  ArticulatedJointWrenchMatrixTest() {
    _numWorkerThreads = 4;
  }
};

TEST_IF_P(MOCHI_USE_DOUBLE_PRECISION, ArticulatedJointWrenchMatrixTest, FiniteDifferences) {
  Actor* actor = GetParam().build(_mochiContext, _scene);
  ASSERT_NE(nullptr, actor);
  // Improved convergence settings are a proxy for merit-based formulation. In this way we guarantee
  // that the residual is the gradient of the merit, and it can be validated by finite differences.
  // Non-merit settings cannot be tested, but that's not critical because they use the same residual
  // routing as merit-based settings.
  experimental::ApplyImprovedConvergenceSettings(_scene, test::ExpectOK{});
  actor->RegisterQuery(QueryType::ArticulatedJointWrenches, test::ExpectOK{});
  for (int step = 0; step < GetParam().numSteps; ++step) {
    _scene->Step(kDt);
  }

  auto& reg = GetRegistry();
  entt::entity const island =
      reg.get<CIslandMemberInfo const>(GetEntity(actor->GetHandle())).island;
  SnleProblem<real> problem = solver::InitIslandAssembly(
      reg, island, reg.get<CIslandDescendants const>(island), /*syncCurrentState*/ true);
  // Evaluate the joint wrenches at the unperturbed state.
  problem.UpdateObjResDRes({.assemObj = false, .assemRes = true, .assemDRes = false});
  JointWrenches const wrenches = GetWrenches(actor, test::ExpectOK{});

  CheckSubtreeSums(
      _scene, actor, wrenches, ComputeMeritLinkGradients(_scene, actor, problem), kFiniteDiffTol);

  // The joints transmit loads, so the check above is not trivially satisfied.
  real maxForce = 0_r;
  for (Real3 const& force : wrenches.forces) {
    maxForce = Max(maxForce, Norm(force));
  }
  EXPECT_GT(maxForce, 1e-4_r * ForceScale(actor));
}

INSTANTIATE_TEST_SUITE_P(
    Configurations,
    ArticulatedJointWrenchMatrixTest,
    ::testing::ValuesIn(GetWrenchCases()),
    WrenchCaseName);

using ArticulatedJointWrenchThreadingTest = ArticulatedJointWrenchMatrixTest;

TEST_P(ArticulatedJointWrenchThreadingTest, SameWrenchesWithAndWithoutWorkerThreads) {
  Actor* actor = nullptr;
  auto run = [&](bool singleThreaded) {
    _mochiContext->SetIsSingleThreaded(singleThreaded);
    _mochiContext->DestroyScene(_scene);
    _scene = _mochiContext->CreateScene("JointWrenchThreading");
    actor = GetParam().build(_mochiContext, _scene);
    actor->RegisterQuery(QueryType::ArticulatedJointWrenches, test::ExpectOK{});
    for (int step = 0; step < GetParam().numSteps; ++step) {
      _scene->Step(kDt);
    }
    return GetWrenches(actor, test::ExpectOK{});
  };
  JointWrenches const single = run(true);
  JointWrenches const multi = run(false);

  // Keeps usesWorkerThreads in sync with the heuristic. TSAN builds check every case.
  auto const& reg = GetRegistry();
  entt::entity const island = reg.get<CIslandMemberInfo const>(GetEntity(actor)).island;
  EXPECT_EQ(
      GetParam().usesWorkerThreads,
      !island::ShouldRunSingleThreaded(reg, island, reg.get<CIslandDescendants const>(island)));

  real scale = 1_r;
  for (Real3 const& force : single.forces) {
    scale = Max(scale, Norm(force));
  }
  for (int i = 0; i < isize(single.forces); ++i) {
    EXPECT_LE(Norm(single.forces[i] - multi.forces[i]), kThreadingTol * scale) << "Joint " << i;
    EXPECT_LE(Norm(single.torques[i] - multi.torques[i]), kThreadingTol * scale * 4_r)
        << "Joint " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    Configurations,
    ArticulatedJointWrenchThreadingTest,
    ::testing::ValuesIn(GetThreadingCases()),
    WrenchCaseName);

} // namespace
