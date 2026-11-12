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

#include "mochi_differentiable_friction_anchor_test_utils.h"
#include "mochi_differentiable_test_utils.h"
#include "mochi_friction_anchor_test_utils.h"
#include "mochi_physics_test_fixture.h"

#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/defer.h>
#include <mochi_physics/src/mochi_contact.h>
#include <mochi_physics/src/mochi_contact_anchors.h>
#include <mochi_physics/src/mochi_differentiable.h>
#include <mochi_physics/src/mochi_ecs.h>

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

using namespace mochi;
using namespace mochi::diffsim;

namespace {

real constexpr kDt = 0.01_r;

class MochiDifferentiableFrictionAnchor : public test::MochiSceneTestBase {
 public:
  void SetUp() override {
    test::MochiSceneTestBase::SetUp();
    // Back-propagation requires backward Euler.
    test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
  }

  Actor* CreateDifferentiableCube() {
    test::EnableElastoPlasticFriction(_scene);
    RigidActorParams cubeParams;
    cubeParams.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
    auto* cube = _scene->CreateRigidActor(cubeParams, test::ExpectOK{});
    MakeSceneDifferentiable(_scene, test::ExpectOK{});
    return cube;
  }

  // A cube pushed along x for 10 steps on three coincident static planes, so that a contact sample
  // can have an anchor on each plane. Returns the cube and the middle plane.
  std::pair<Actor*, Actor*> CreatePushedCubeOnThreePlanes() {
    test::EnableElastoPlasticFriction(_scene);
    test::CreateGroundPlane(_mochiContext, _scene);
    auto* middlePlane = test::CreateGroundPlane(_mochiContext, _scene);
    test::CreateGroundPlane(_mochiContext, _scene);
    RigidActorParams cubeParams;
    cubeParams.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
    auto* cube = _scene->CreateRigidActor(cubeParams, test::ExpectOK{});
    MakeSceneDifferentiable(_scene, test::ExpectOK{});
    int const dofs[] = {0};
    real const force[] = {5_r};
    cube->SetExternalForcesOnDofs(dofs, force, test::ExpectOK{});
    for (int i = 0; i < 10; ++i) {
      _scene->Step(kDt);
    }
    return {cube, middlePlane};
  }
};

} // namespace

// PrepareBackPropagate restores the friction anchors that the forward step used, not those it
// advanced, and back-propagation detects the step's contacts with them.
TEST_F(MochiDifferentiableFrictionAnchor, PrepareBackPropagateUsesForwardStepAnchors) {
  test::EnableElastoPlasticFriction(_scene);
  test::CreateGroundPlane(_mochiContext, _scene);

  // A 0.5 m cube with a corner at its origin, resting on the plane.
  RigidActorParams cubeParams;
  cubeParams.name = "Cube";
  cubeParams.shape = test::CreateUnitCubeTetMeshShape(_mochiContext, Real3{0.5_r, 0.5_r, 0.5_r});
  auto* cube = _scene->CreateRigidActor(cubeParams, test::ExpectOK{});
  entt::entity const cubeEntity = GetEntity(cube);

  MakeSceneDifferentiable(_scene, test::ExpectOK{});

  // An incline steeper than the friction angle (tan 35 deg > 0.5), so the cube slides and drags its
  // anchors every step.
  real constexpr kTheta = 35_r * kRadiansPerDegree;
  _scene->SetGravity(Real3{9.8_r * std::sin(kTheta), -9.8_r * std::cos(kTheta), 0_r});
  int constexpr kNumSteps = 10;
  auto const& reg = GetRegistry();
  std::vector<StateHandle> states{_scene->CaptureState(test::ExpectOK{})};
  MOCHI_DEFER({
    for (StateHandle const s : states) {
      _scene->ReleaseState(s);
    }
  });
  DynamicArray<FrictionAnchor> usedAnchors;
  DynamicArray<FrictionAnchor> advancedAnchors;
  for (int i = 0; i < kNumSteps; ++i) {
    _scene->Step(kDt);
    states.push_back(_scene->CaptureState(test::ExpectOK{}));
    usedAnchors = reg.get<CFrictionAnchors<AnchorSet::Active> const>(cubeEntity).anchors;
    advancedAnchors = reg.get<CFrictionAnchors<AnchorSet::Pending> const>(cubeEntity).anchors;
  }
  ASSERT_FALSE(usedAnchors.empty());
  ASSERT_FALSE(test::AreEqualAnchors(usedAnchors, advancedAnchors));

  PrepareBackPropagate(_scene, states[kNumSteps], states[kNumSteps - 1], test::ExpectOK{});

  test::ExpectAnchors(
      usedAnchors, reg.get<CFrictionAnchors<AnchorSet::Active> const>(cubeEntity).anchors);
  test::ExpectAnchors(
      advancedAnchors, reg.get<CFrictionAnchors<AnchorSet::Pending> const>(cubeEntity).anchors);

  // The Pending anchors are indexed, for the back-propagation of their advance.
  auto const& pending = reg.get<CFrictionAnchors<AnchorSet::Pending> const>(cubeEntity).anchors;
  auto const& pendingIndex = reg.get<CFrictionAnchorIndex<AnchorSet::Pending> const>(cubeEntity);
  for (int i = 0; i < isize(pending); ++i) {
    EXPECT_EQ(
        i,
        FindFrictionAnchor(
            pendingIndex, MakeConstSpan(pending), pending[i].sample, pending[i].collider));
  }

  // The contact results rebuilt by back-propagation carry the offsets from the used anchors.
  int numAnchoredContacts = 0;
  for (auto const& collision :
       reg.get<CActiveCollisions</*kIsSync*/ false, TimeStep::Current> const>(cubeEntity)) {
    auto const& result = collision.collisionResult;
    ASSERT_EQ(result.anchorOffset.size(), result.sampleIndices.size());
    for (int i = 0; i < isize(result.sampleIndices); ++i) {
      Real3 expected{};
      for (FrictionAnchor const& anchor : usedAnchors) {
        if (anchor.sample == result.sampleIndices[i] &&
            anchor.collider == entt::to_integral(collision.colliderEntity)) {
          expected = result.posCollidingStageStart[i] - anchor.point;
          ++numAnchoredContacts;
        }
      }
      EXPECT_EQ(expected, result.anchorOffset[i]);
    }
  }
  EXPECT_GT(numAnchoredContacts, 0);
}

// Forward drops stably compact Pending into Active; the reverse maps Active's gradient into
// Pending's layout, with zero for each dropped anchor, replacing any stale Pending gradient. Also
// for an empty Pending set.
TEST(FrictionAnchorAdjoints, ReverseEveryCombinationOfDrops) {
  DynamicArray<FrictionAnchor> const fivePrevious = {
      {.sample = 0, .collider = 10},
      {.sample = 0, .collider = 20},
      {.sample = 1, .collider = 10},
      {.sample = 2, .collider = 30},
      {.sample = 3, .collider = 10}};
  for (DynamicArray<FrictionAnchor> const& previous :
       {fivePrevious, DynamicArray<FrictionAnchor>{}}) {
    for (int mask = 0; mask < (1 << isize(previous)); ++mask) {
      SCOPED_TRACE(::testing::Message() << isize(previous) << " anchors, mask " << mask);
      entt::registry reg;
      entt::entity const e = reg.create();
      auto& active = reg.emplace<CFrictionAnchors<AnchorSet::Active>>(e);
      reg.emplace<CFrictionAnchors<AnchorSet::Pending>>(e).anchors = previous;
      auto& activeGrad = reg.emplace<CDiffFrictionAnchorGrad<AnchorSet::Active>>(e);
      auto& pendingGrad = reg.emplace<CDiffFrictionAnchorGrad<AnchorSet::Pending>>(e);
      pendingGrad.grads = {Real3{9_r, 9_r, 9_r}};
      DynamicArray<Real3> expected(previous.size(), Real3{});
      for (int i = 0; i < isize(previous); ++i) {
        if (mask & (1 << i)) {
          active.anchors.push_back(previous[i]);
          Real3 const grad{static_cast<real>(i + 1), 2_r, -3_r};
          activeGrad.grads.push_back(grad);
          expected[i] = grad;
        }
      }
      Real3 const* const data = activeGrad.grads.data();

      ecs::InvokeForEachGlobal(&ReverseFrictionAnchorDropAdjoints, reg);

      EXPECT_SPAN_EQ(MakeConstSpan(expected), MakeConstSpan(pendingGrad.grads));
      EXPECT_TRUE(activeGrad.grads.empty());
      if (mask == (1 << isize(previous)) - 1) {
        EXPECT_EQ(data, pendingGrad.grads.data());
      }
    }
  }
}

TEST_F(MochiDifferentiableFrictionAnchor, BackPropagateReversesSetterDropBeforeHandoff) {
  auto const [cube, b] = CreatePushedCubeOnThreePlanes();
  auto& reg = GetRegistry();
  entt::entity const e = GetEntity(cube);
  StateHandle const older = _scene->CaptureState(test::ExpectOK{});
  _scene->Step(kDt);
  StateHandle const beforeSetter = _scene->CaptureState(test::ExpectOK{});
  b->SetVelocity(Real3{}, Real3{}, test::ExpectOK{});
  StateHandle const pre = _scene->CaptureState(test::ExpectOK{});
  DynamicArray<FrictionAnchor> const prePending =
      reg.get<CFrictionAnchors<AnchorSet::Pending> const>(e).anchors;
  DynamicArray<FrictionAnchor> const preActive =
      reg.get<CFrictionAnchors<AnchorSet::Active> const>(e).anchors;
  _scene->Step(kDt);
  StateHandle const post = _scene->CaptureState(test::ExpectOK{});
  MOCHI_DEFER({
    _scene->ReleaseState(older);
    _scene->ReleaseState(beforeSetter);
    _scene->ReleaseState(pre);
    _scene->ReleaseState(post);
  });
  uint32_t const bId = entt::to_integral(GetEntity(b));
  DynamicArray<FrictionAnchor> const used =
      reg.get<CFrictionAnchors<AnchorSet::Active> const>(e).anchors;
  for (auto const& anchor : used) {
    EXPECT_NE(bId, anchor.collider);
  }
  ASSERT_GT(prePending.size(), used.size());

  ResetBackPropagation(_scene, test::ExpectOK{});
  PrepareBackPropagate(_scene, post, pre, test::ExpectOK{});
  real poseGrad[RigidSize::kAll] = {};
  poseGrad[0] = 1_r;
  GetCenterOfMassTransformBackward(cube, poseGrad, test::ExpectOK{});
  BackPropagate(_scene, test::ExpectOK{});

  test::ExpectAnchors(prePending, reg.get<CFrictionAnchors<AnchorSet::Pending> const>(e).anchors);
  test::ExpectAnchors(preActive, reg.get<CFrictionAnchors<AnchorSet::Active> const>(e).anchors);
  EXPECT_TRUE(reg.get<CDiffFrictionAnchorGrad<AnchorSet::Active> const>(e).grads.empty());
  auto const& result = reg.get<CDiffFrictionAnchorGrad<AnchorSet::Pending> const>(e).grads;
  ASSERT_EQ(prePending.size(), result.size());
  int dropped = 0;
  bool survivorHasGradient = false;
  for (int i = 0; i < isize(prePending); ++i) {
    if (prePending[i].collider == bId) {
      ++dropped;
      EXPECT_EQ(Real3{}, result[i]);
    } else {
      survivorHasGradient |= result[i] != Real3{};
    }
  }
  EXPECT_GT(dropped, 0);
  EXPECT_TRUE(survivorHasGradient);
  DynamicArray<Real3> const handoff = result;
  Real3 const* const data = result.data();

  PrepareBackPropagate(_scene, beforeSetter, older, test::ExpectOK{});

  auto const& incoming = reg.get<CDiffFrictionAnchorGrad<AnchorSet::Pending> const>(e).grads;
  EXPECT_EQ(data, incoming.data());
  EXPECT_SPAN_EQ(MakeConstSpan(handoff), MakeConstSpan(incoming));
  BackPropagate(_scene, test::ExpectOK{});
}

// Incoming adjoints are validated against the captured post-step Pending set, not the live one.
TEST_F(MochiDifferentiableFrictionAnchor, PrepareRejectsMismatchedDenseAdjoints) {
  auto* cube = CreateDifferentiableCube();
  auto& reg = GetRegistry();
  entt::entity const e = GetEntity(cube);
  auto& pending = reg.get<CFrictionAnchors<AnchorSet::Pending>>(e);
  pending.anchors = {{.sample = 0, .collider = 10}, {.sample = 1, .collider = 20}};
  StateHandle const state = _scene->CaptureState(test::ExpectOK{});
  MOCHI_DEFER(_scene->ReleaseState(state));
  pending.anchors.resize(1);
  auto const& grads = reg.get<CDiffFrictionAnchorGrad<AnchorSet::Pending>>(e).grads;
  DynamicArray<Real3> const mismatched = {Real3{1_r, 2_r, 3_r}};
  reg.get<CDiffFrictionAnchorGrad<AnchorSet::Pending>>(e).grads = mismatched;

  PrepareBackPropagate(_scene, state, state, test::ExpectNotOK{});

  EXPECT_SPAN_EQ(MakeConstSpan(mismatched), MakeConstSpan(grads));
  DynamicArray<Real3> const matching = {Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}};
  reg.get<CDiffFrictionAnchorGrad<AnchorSet::Pending>>(e).grads = matching;
  PrepareBackPropagate(_scene, state, state, test::ExpectOK{});
  EXPECT_SPAN_EQ(MakeConstSpan(matching), MakeConstSpan(grads));
  ResetBackPropagation(_scene, test::ExpectOK{});
}

// Only the Pending gradient is captured: a full restore restores it with its anchor layout and
// leaves the uncaptured Active working gradient untouched.
TEST_F(MochiDifferentiableFrictionAnchor, CaptureRestoresDenseAdjointsWithTheirLayout) {
  auto* cube = CreateDifferentiableCube();
  auto& reg = GetRegistry();
  entt::entity const e = GetEntity(cube);
  auto& pending = reg.get<CFrictionAnchors<AnchorSet::Pending>>(e);
  pending.anchors = {{.sample = 0, .collider = 10}, {.sample = 1, .collider = 20}};
  DynamicArray<FrictionAnchor> const captured = pending.anchors;
  auto& pendingGrad = reg.get<CDiffFrictionAnchorGrad<AnchorSet::Pending>>(e);
  auto& activeGrad = reg.get<CDiffFrictionAnchorGrad<AnchorSet::Active>>(e);
  DynamicArray<Real3> const expected = {Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}};
  pendingGrad.grads = expected;
  StateHandle const state = _scene->CaptureState(test::ExpectOK{});
  MOCHI_DEFER(_scene->ReleaseState(state));
  std::swap(pending.anchors[0], pending.anchors[1]);
  pendingGrad.grads.clear();
  DynamicArray<Real3> const live = {Real3{7_r, 8_r, 9_r}};
  activeGrad.grads = live;

  _scene->RestoreState(state, /*releaseImmediately*/ false, test::ExpectOK{});

  test::ExpectAnchors(captured, reg.get<CFrictionAnchors<AnchorSet::Pending> const>(e).anchors);
  EXPECT_SPAN_EQ(MakeConstSpan(expected), MakeConstSpan(pendingGrad.grads));
  EXPECT_SPAN_EQ(MakeConstSpan(live), MakeConstSpan(activeGrad.grads));
}

// A diverged step is back-propagated like a converged one, through the anchors it retained in both
// sets: no assertion fires, and the gradients are finite.
TEST_F(MochiDifferentiableFrictionAnchor, BackPropagatesThroughDivergedStep) {
  test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
  Actor* const cube = CreatePushedCubeOnThreePlanes().first;
  SolverParams const converging = _scene->GetSolverParams();
  // Zero tolerances and a tiny divergence threshold make a step diverge.
  SolverParams diverging = converging;
  diverging.nonLinearSolver.absTol = 0_r;
  diverging.nonLinearSolver.relTol = 0_r;
  diverging.nonLinearSolver.relStepTol = 0_r;
  diverging.nonLinearSolver.absDivTol = 1e-30_r;
  int constexpr kNumSteps = 3;
  int constexpr kDivergedStep = 1;
  int const dofs[] = {0};
  real const force[] = {5_r};
  std::vector<StateHandle> states{_scene->CaptureState(test::ExpectOK{})};
  MOCHI_DEFER({
    for (StateHandle const s : states) {
      _scene->ReleaseState(s);
    }
  });
  for (int i = 0; i < kNumSteps; ++i) {
    _scene->SetSolverParams(i == kDivergedStep ? diverging : converging, test::ExpectOK{});
    cube->SetExternalForcesOnDofs(dofs, force, test::ExpectOK{});
    _scene->Step(kDt);
    ASSERT_EQ(i == kDivergedStep, cube->GetConvergenceStatus() == ConvergenceStatus::Diverged);
    states.push_back(_scene->CaptureState(test::ExpectOK{}));
  }

  ResetBackPropagation(_scene, test::ExpectOK{});
  real poseGrad[RigidSize::kAll] = {};
  poseGrad[0] = 1_r;
  for (int i = kNumSteps - 1; i >= 0; --i) {
    PrepareBackPropagate(_scene, states[i + 1], states[i], test::ExpectOK{});
    if (i == kNumSteps - 1) {
      GetCenterOfMassTransformBackward(cube, poseGrad, test::ExpectOK{});
    }
    BackPropagate(_scene, test::ExpectOK{});
    real forceGrad[] = {0_r};
    SetExternalForcesOnDofsBackward(cube, dofs, forceGrad, test::ExpectOK{});
    EXPECT_TRUE(IsFinite(forceGrad[0])) << "step " << i;
  }
  auto const& pendingGrad =
      GetRegistry().get<CDiffFrictionAnchorGrad<AnchorSet::Pending> const>(GetEntity(cube));
  ASSERT_FALSE(pendingGrad.grads.empty());
  for (Real3 const& grad : pendingGrad.grads) {
    EXPECT_TRUE(IsFinite(grad));
  }
}
