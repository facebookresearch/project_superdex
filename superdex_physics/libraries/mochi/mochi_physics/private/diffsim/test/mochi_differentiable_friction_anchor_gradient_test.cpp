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

// Finite-difference checks of back-propagation through elasto-plastic friction anchors: actors
// pushed (or twisted) by external forces scaled by theta, then left to rest. The loss is a sum of
// components of the actors' final poses, or of a cube's reported contact force.

#include "mochi_differentiable_friction_anchor_test_utils.h"
#include "mochi_differentiable_test_utils.h"
#include "mochi_physics_test_fixture.h"

#include <mochi_core/materials/material_params.h>
#include <mochi_core/utils/defer.h>

#include <cmath>
#include <functional>
#include <initializer_list>
#include <optional>
#include <utility>
#include <vector>

using namespace mochi;
using namespace mochi::diffsim;

// The finite-difference tolerances below assume double precision.
#if MOCHI_USE_DOUBLE_PRECISION

namespace {

real constexpr kDt = 0.01_r;
real constexpr kPreslidingDisplacement = 1e-4_r;
real constexpr kCubeSize = 0.5_r;
// Density that gives the 6 x 1 x 6 box the mass of the cube.
real constexpr kBoxDensity = kDefaultDensity * kCubeSize * kCubeSize * kCubeSize / 36_r;

// An external force or torque on one DoF of an actor, at theta = 1, applied during the push.
struct Push {
  Actor* actor = nullptr;
  int dof = 0; // 0: x translation, 4: y rotation.
  real force = 0_r;
};

// A term of the loss on the state after the last step, and the seed of its gradient, given after
// PrepareBackPropagate of the last step.
struct LossTerm {
  std::function<real()> eval;
  std::function<void()> seedGrad;
};

struct Rollout {
  std::vector<Push> pushes;
  std::vector<LossTerm> loss; // Summed.
  int pushSteps = 100;
  int restSteps = 50; // Steps after the push.
  // If set, this actor is placed at its own pose before placeStep, which drops its anchors.
  Actor* placed = nullptr;
  int placeStep = 0;
  // Back-propagations of each rollout, with ResetBackPropagation but no restore of an earlier state
  // between them. The last one gives the gradient.
  int backwardPasses = 1;
};

void SetPreslidingDisplacement(Actor* actor, real preslidingDisplacement) {
  ContactParams params = actor->GetContactParams(test::ExpectOK{});
  params.preslidingDisplacement = preslidingDisplacement;
  actor->SetContactParams(params, test::ExpectOK{});
}

// Friction force at which the actors, stacked on a plane, start sliding on it.
real FrictionLimit(std::initializer_list<Actor const*> actors) {
  real mass = 0_r;
  for (Actor const* actor : actors) {
    mass += actor->GetMass(test::ExpectOK{});
  }
  return 0.5_r * mass * 9.8_r;
}

// Translation (0-2) or quaternion (3-6) component of a rigid actor's final COM transform.
LossTerm ComTransformLoss(Actor* actor, int index) {
  return {
      .eval =
          [actor, index] {
            ColumnVector<real, RigidSize::kAll> pose;
            TransformToRawPose(actor->GetCenterOfMassTransform(test::ExpectOK{}), AsView(pose));
            return pose(index);
          },
      .seedGrad =
          [actor, index] {
            real grad[RigidSize::kAll] = {};
            grad[index] = 1_r;
            GetCenterOfMassTransformBackward(actor, grad, test::ExpectOK{});
          }};
}

// Translation (0-2) or rotation vector (3-5) component of an articulated actor's root joint pose.
LossTerm RootPoseLoss(Actor* actor, int index) {
  return {
      .eval =
          [actor, index] {
            real pose[RigidSize::kDAll] = {};
            actor->GetArticulatedPose(Span<real>{pose, RigidSize::kDAll}, test::ExpectOK{});
            return pose[index];
          },
      .seedGrad =
          [actor, index] {
            real grad[RigidSize::kDAll] = {};
            grad[index] = 1_r;
            GetArticulatedPoseBackward(
                actor, Span<real const>{grad, RigidSize::kDAll}, test::ExpectOK{});
          }};
}

// The x component of an actor's total contact force at the last step. Requires the query.
LossTerm ContactForceLoss(Actor* actor) {
  return {
      .eval = [actor] { return actor->GetContactForceWorld(test::ExpectOK{})[0]; },
      .seedGrad =
          [actor] {
            real const grad[] = {1_r, 0_r, 0_r};
            GetContactForceWorldBackward(actor, grad, test::ExpectOK{});
          }};
}

// Parameterized by the number of worker threads.
class FrictionAnchorGradient : public test::MochiSceneTestBase,
                               public ::testing::WithParamInterface<int> {
 public:
  void SetUp() override {
    _numWorkerThreads = GetParam();
    test::MochiSceneTestBase::SetUp();
    test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
    auto params = _scene->GetSolverParams();
    params.nonLinearSolver.maxIter = 30;
    params.nonLinearSolver.absTol = 1e-10_r;
    params.nonLinearSolver.relTol = 1e-10_r;
    params.experimentalEval.fittedSaturationHessian.contactFriction = false;
    _scene->SetSolverParams(params, test::ExpectOK{});
  }

  Actor* CreateBox(
      char const* name,
      Real3 const& size,
      Real3 const& origin,
      std::optional<real> density = std::nullopt) {
    RigidActorParams params;
    params.name = name;
    params.shape = test::CreateUnitCubeTetMeshShape(_mochiContext, size);
    params.density = density;
    auto* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
    actor->SetRootTransform(TransformRT{Quaternion{}, origin}, test::ExpectOK{});
    return actor;
  }

  // The cube, with the given presliding displacement, which the pair with a static collider takes.
  Actor* CreateCube(Real3 const& origin, real preslidingDisplacement = kPreslidingDisplacement) {
    auto* cube = CreateBox("Cube", Real3{kCubeSize, kCubeSize, kCubeSize}, origin);
    SetPreslidingDisplacement(cube, preslidingDisplacement);
    return cube;
  }

  // A dynamic 6 x 1 x 6 box of the cube's mass on the plane, wide enough that a cube on it never
  // reaches an edge, where the box's SDF normal jumps.
  Actor* CreateDynamicBox(real preslidingDisplacement) {
    auto* box = CreateBox("Box", Real3{6_r, 1_r, 6_r}, Real3{-1_r, 0_r, -3_r}, kBoxDensity);
    SetPreslidingDisplacement(box, preslidingDisplacement);
    return box;
  }

  // Free-root articulated actor of two cube links, the second welded at the given offset along z.
  // The links keep the default contact parameters.
  Actor* CreateArticulatedCubes(Real3 const& origin, real secondLinkOffset) {
    ArticulatedActorParams params;
    params.name = "Articulated";
    params.joints.resize(2);
    params.joints[0].type = ArticulatedJointType::Free;
    params.joints[1].type = ArticulatedJointType::Hard;
    params.joints[1].parentLinkFromJoint =
        TransformRT{Quaternion{}, Real3{0_r, 0_r, secondLinkOffset}};
    params.links.resize(2);
    params.links[0].shape =
        test::CreateUnitCubeTetMeshShape(_mochiContext, Real3{kCubeSize, kCubeSize, kCubeSize});
    params.links[1].shape = params.links[0].shape;
    params.links[1].parentLink = 0;
    auto* actor = _scene->CreateArticulatedActor(params, test::ExpectOK{});
    EXPECT_EQ(ContactParams{}.preslidingDisplacement, kPreslidingDisplacement);
    actor->SetRootTransform(TransformRT{Quaternion{}, origin}, test::ExpectOK{});
    return actor;
  }

  // Makes the scene differentiable, with a back-propagation solve as tight as the forward one.
  void MakeDifferentiable() {
    MakeSceneDifferentiable(_scene, test::ExpectOK{});
    auto params = GetBackPropagationSolverParams(_scene, test::ExpectOK{});
    params.outerSolverMaxIter = 30;
    params.outerSolverAbsTol = 1e-10_r;
    params.validateFiniteDiff = true;
    SetBackPropagationSolverParams(_scene, params, test::ExpectOK{});
  }

  // Returns {finite-difference, analytic} gradients of the loss wrt theta, for each theta. Settles
  // the scene first, and starts every rollout from the settled state.
  std::vector<std::pair<real, real>> Run(Rollout const& rollout, std::vector<real> const& thetas) {
    for (int i = 0; i < 50; ++i) {
      _scene->Step(kDt);
    }
    StateHandle const initial = _scene->CaptureState(test::ExpectOK{});
    MOCHI_DEFER(_scene->ReleaseState(initial));

    int const numSteps = rollout.pushSteps + rollout.restSteps;
    auto profile = [&](int step) { return step < rollout.pushSteps ? 1_r : 0_r; };
    auto evalLoss = [&] {
      real loss = 0_r;
      for (LossTerm const& term : rollout.loss) {
        loss += term.eval();
      }
      return loss;
    };
    auto simulate = [&](real theta, std::vector<StateHandle>* pre, std::vector<StateHandle>* post) {
      _scene->RestoreState(initial, /*releaseImmediately*/ false, test::ExpectOK{});
      for (int i = 0; i < numSteps; ++i) {
        if (rollout.placed && i == rollout.placeStep) {
          rollout.placed->SetRootTransform(rollout.placed->GetRootTransform(), test::ExpectOK{});
        }
        for (Push const& push : rollout.pushes) {
          int const dofs[] = {push.dof};
          real const force[] = {theta * push.force * profile(i)};
          push.actor->SetExternalForcesOnDofs(dofs, force, test::ExpectOK{});
        }
        if (pre) {
          pre->push_back(_scene->CaptureState(test::ExpectOK{}));
        }
        _scene->Step(kDt);
        if (post) {
          post->push_back(_scene->CaptureState(test::ExpectOK{}));
        }
      }
    };
    auto analyticGradient = [&](real theta) {
      std::vector<StateHandle> pre, post;
      simulate(theta, &pre, &post);
      real grad = 0_r;
      for (int pass = 0; pass < rollout.backwardPasses; ++pass) {
        ResetBackPropagation(_scene, test::ExpectOK{});
        grad = 0_r;
        for (int i = numSteps - 1; i >= 0; --i) {
          PrepareBackPropagate(_scene, post[i], pre[i], test::ExpectOK{});
          if (i == numSteps - 1) {
            for (LossTerm const& term : rollout.loss) {
              term.seedGrad();
            }
          }
          BackPropagate(_scene, test::ExpectOK{});
          EXPECT_TRUE(GetBackPropagationSceneStats(_scene, test::ExpectOK{}).finiteDiffValid)
              << "theta " << theta << ", step " << i;
          for (Push const& push : rollout.pushes) {
            int const dofs[] = {push.dof};
            real forceGrad[] = {0_r};
            SetExternalForcesOnDofsBackward(push.actor, dofs, forceGrad, test::ExpectOK{});
            grad += forceGrad[0] * push.force * profile(i);
          }
        }
      }
      for (StateHandle const s : pre) {
        _scene->ReleaseState(s);
      }
      for (StateHandle const s : post) {
        _scene->ReleaseState(s);
      }
      return grad;
    };

    std::vector<std::pair<real, real>> grads;
    for (real const theta : thetas) {
      real const analytic = analyticGradient(theta);
      real constexpr kEps = 1e-3_r;
      simulate(theta + kEps, nullptr, nullptr);
      real const lossPlus = evalLoss();
      simulate(theta - kEps, nullptr, nullptr);
      real const lossMinus = evalLoss();
      grads.emplace_back((lossPlus - lossMinus) / (2_r * kEps), analytic);
    }
    return grads;
  }

  static void ExpectMatch(std::vector<std::pair<real, real>> const& grads, real relTol = 2e-2_r) {
    for (int i = 0; i < isize(grads); ++i) {
      auto const [fd, analytic] = grads[i];
      EXPECT_NEAR(analytic, fd, relTol * std::abs(fd) + 1e-6_r) << "theta " << i;
    }
  }
};

} // namespace

// Sticking (theta < 1) and sliding (theta > 1) under a push along x.
TEST_P(FrictionAnchorGradient, PushThenRest) {
  test::EnableElastoPlasticFriction(_scene);
  test::CreateGroundPlane(_mochiContext, _scene);
  auto* cube = CreateCube(Real3{});
  MakeDifferentiable();
  ExpectMatch(
      Run({.pushes = {{.actor = cube, .force = FrictionLimit({cube})}},
           .loss = {ComTransformLoss(cube, 0)}},
          {0.5_r, 0.9_r, 1.5_r}));
}

// Torsional sticking and slipping exercise the rotational contact Jacobians.
TEST_P(FrictionAnchorGradient, TwistThenRest) {
  test::EnableElastoPlasticFriction(_scene);
  test::CreateGroundPlane(_mochiContext, _scene);
  auto* cube = CreateCube(Real3{});
  MakeDifferentiable();
  // A torque scaled by the cube's half-size.
  ExpectMatch(Run(
      {.pushes = {{.actor = cube, .dof = 4, .force = 0.5_r * kCubeSize * FrictionLimit({cube})}},
       .loss = {ComTransformLoss(cube, 4)}},
      {0.5_r, 2_r}));
}

// A sample with anchors on several colliders: three coincident planes.
TEST_P(FrictionAnchorGradient, PushOnThreeCoincidentPlanes) {
  test::EnableElastoPlasticFriction(_scene);
  for (int i = 0; i < 3; ++i) {
    test::CreateGroundPlane(_mochiContext, _scene);
  }
  auto* cube = CreateCube(Real3{});
  MakeDifferentiable();
  ExpectMatch(
      Run({.pushes = {{.actor = cube, .force = FrictionLimit({cube})}},
           .loss = {ComTransformLoss(cube, 0)}},
          {0.5_r, 1.5_r}));
}

// Sync contact: a dynamic box of the cube's mass, pushed while the cube sticks to it, so the cube's
// anchors on the box also back-propagate to the box's DoFs. The box has 4 times the cube's
// presliding displacement, so the pair has twice the cube's (geometric mean).
TEST_P(FrictionAnchorGradient, PushOnDynamicBox) {
  test::EnableElastoPlasticFriction(_scene);
  test::CreateGroundPlane(_mochiContext, _scene);
  auto* box = CreateDynamicBox(4_r * kPreslidingDisplacement);
  auto* cube = CreateCube(Real3{0_r, 1_r, 0_r});
  MakeDifferentiable();
  ExpectMatch(
      Run({.pushes = {{.actor = box, .force = FrictionLimit({cube, box})}},
           .loss = {ComTransformLoss(cube, 0)}},
          {0.5_r, 1.5_r}));
}

// Sync contact with a rotating collider: the box turns on a frictionless plane, carrying the cube,
// which sticks to it far from the box's center, so the cube's anchors back-propagate to the box's
// rotation.
TEST_P(FrictionAnchorGradient, TwistDynamicBox) {
  test::EnableElastoPlasticFriction(_scene);
  auto* plane = test::CreateGroundPlane(_mochiContext, _scene);
  ContactParams frictionless = plane->GetContactParams(test::ExpectOK{});
  frictionless.coulombFrictionCoefficient = 0_r;
  plane->SetContactParams(frictionless, test::ExpectOK{});
  auto* box = CreateDynamicBox(kPreslidingDisplacement);
  auto* cube = CreateCube(Real3{0_r, 1_r, -2.5_r});
  MakeDifferentiable();
  ExpectMatch(
      Run({.pushes = {{.actor = box, .dof = 4, .force = 0.1_r * FrictionLimit({cube, box})}},
           .loss = {ComTransformLoss(cube, 0), ComTransformLoss(cube, 2)}},
          {0.5_r, 1_r}));
}

// The anchors are on articulated links, so the solve adjoint is mapped from the articulated DoFs
// through each link's rows of the Jacobian.
TEST_P(FrictionAnchorGradient, PushArticulatedLink) {
  test::EnableElastoPlasticFriction(_scene);
  test::CreateGroundPlane(_mochiContext, _scene);
  auto* articulated = CreateArticulatedCubes(Real3{}, 1.5_r * kCubeSize);
  MakeDifferentiable();
  ExpectMatch(
      Run({.pushes = {{.actor = articulated, .force = FrictionLimit({articulated})}},
           .loss = {RootPoseLoss(articulated, 0)}},
          {0.5_r, 1.5_r}));
}

// Twisting an articulated actor whose second link is far from the root: the second link's anchors
// move with the root's rotation times a long lever arm, so the solve adjoint mapped to them through
// the Jacobian dominates the gradient.
TEST_P(FrictionAnchorGradient, TwistArticulatedWithLongLever) {
  test::EnableElastoPlasticFriction(_scene);
  test::CreateGroundPlane(_mochiContext, _scene);
  real constexpr kLever = 4_r;
  auto* articulated = CreateArticulatedCubes(Real3{}, kLever);
  MakeDifferentiable();
  ExpectMatch(
      Run({.pushes =
               {{.actor = articulated,
                 .dof = 4,
                 .force = 0.5_r * kLever * FrictionLimit({articulated})}},
           .loss = {RootPoseLoss(articulated, 4)}},
          {0.5_r, 1.5_r}));
}

// ResetBackPropagation clears the anchor adjoints of articulated links, which have no state
// adjoint: a second back-propagation of a rollout gives the gradient again.
TEST_P(FrictionAnchorGradient, PushArticulatedLinkBackPropagatedTwice) {
  test::EnableElastoPlasticFriction(_scene);
  test::CreateGroundPlane(_mochiContext, _scene);
  auto* articulated = CreateArticulatedCubes(Real3{}, 1.5_r * kCubeSize);
  MakeDifferentiable();
  ExpectMatch(
      Run({.pushes = {{.actor = articulated, .force = FrictionLimit({articulated})}},
           .loss = {RootPoseLoss(articulated, 0)},
           .backwardPasses = 2},
          {0.5_r}));
}

// An articulated actor and a cube in separate islands, both pushed, with a loss on both.
TEST_P(FrictionAnchorGradient, PushInTwoIslands) {
  test::EnableElastoPlasticFriction(_scene);
  test::CreateGroundPlane(_mochiContext, _scene);
  auto* articulated = CreateArticulatedCubes(Real3{}, 1.5_r * kCubeSize);
  auto* cube = CreateCube(Real3{5_r, 0_r, 0_r});
  MakeDifferentiable();
  ExpectMatch(
      Run({.pushes =
               {{.actor = articulated, .force = FrictionLimit({articulated})},
                {.actor = cube, .force = FrictionLimit({cube})}},
           .loss = {RootPoseLoss(articulated, 0), ComTransformLoss(cube, 0)}},
          {0.5_r, 1.5_r}));
}

// Placing the cube when the push is released drops its anchors, so a sticking cube no longer
// springs back. No adjoint crosses the placement through the dropped anchors.
TEST_P(FrictionAnchorGradient, PlacedAtRelease) {
  test::EnableElastoPlasticFriction(_scene);
  test::CreateGroundPlane(_mochiContext, _scene);
  auto* cube = CreateCube(Real3{});
  MakeDifferentiable();
  ExpectMatch(
      Run({.pushes = {{.actor = cube, .force = FrictionLimit({cube})}},
           .loss = {ComTransformLoss(cube, 0)},
           .placed = cube,
           .placeStep = 100},
          {0.5_r}));
}

// Placing the resting cube before the first step drops its anchors. Unpushed, it does not move, so
// its contacts get new anchors at exactly zero offset, through which any push springs back.
TEST_P(FrictionAnchorGradient, PlacedAtStartUnpushed) {
  test::EnableElastoPlasticFriction(_scene);
  test::CreateGroundPlane(_mochiContext, _scene);
  auto* cube = CreateCube(Real3{});
  MakeDifferentiable();
  ExpectMatch(
      Run({.pushes = {{.actor = cube, .force = FrictionLimit({cube})}},
           .loss = {ComTransformLoss(cube, 0)},
           .placed = cube,
           .placeStep = 0},
          {0_r}));
}

// Anchors enabled on a scene that is already differentiable get their adjoint too.
TEST_P(FrictionAnchorGradient, EnabledAfterMakingSceneDifferentiable) {
  test::CreateGroundPlane(_mochiContext, _scene);
  auto* cube = CreateCube(Real3{});
  MakeDifferentiable();
  test::EnableElastoPlasticFriction(_scene);
  ExpectMatch(
      Run({.pushes = {{.actor = cube, .force = FrictionLimit({cube})}},
           .loss = {ComTransformLoss(cube, 0)}},
          {0.5_r}));
}

// A loss on the friction force: right after the push is released, the sticking cube springs back
// toward its anchors, and the friction force it reports is the anchor spring. Its stored offsets
// reach the loss both through the step's solve and directly through the reported force.
TEST_P(FrictionAnchorGradient, ContactForceLossAfterRelease) {
  test::EnableElastoPlasticFriction(_scene);
  test::CreateGroundPlane(_mochiContext, _scene);
  auto* cube = CreateCube(Real3{});
  cube->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  MakeDifferentiable();
  ExpectMatch(
      Run({.pushes = {{.actor = cube, .force = FrictionLimit({cube})}},
           .loss = {ContactForceLoss(cube)},
           .restSteps = 3},
          {0.5_r, 0.9_r}));
}

// A loss on the friction force while it holds the push. The anchor adjoint after the solve must
// read the contacts at the solution, not at the last perturbation of the solve's finite-difference
// Hessian-vector products: with a short presliding displacement, the anchor stiffness changes over
// a displacement comparable to that perturbation.
TEST_P(FrictionAnchorGradient, ContactForceLossWhileHolding) {
  test::EnableElastoPlasticFriction(_scene);
  test::CreateGroundPlane(_mochiContext, _scene);
  auto* cube = CreateCube(Real3{}, 1e-5_r);
  cube->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  MakeDifferentiable();
  ExpectMatch(
      Run({.pushes = {{.actor = cube, .force = FrictionLimit({cube})}},
           .loss = {ContactForceLoss(cube)},
           .restSteps = 0},
          {0.6_r, 0.9_r}),
      2e-3_r);
}

// A loss on the friction force of the cube on the dynamic box, a sync contact, right after the push
// on the box is released. While the box slides, or while it is pushed, the gradient of this loss
// misses its finite difference by a few percent even without anchors, so the box sticks to the
// plane.
TEST_P(FrictionAnchorGradient, ContactForceLossOnDynamicBox) {
  test::EnableElastoPlasticFriction(_scene);
  test::CreateGroundPlane(_mochiContext, _scene);
  auto* box = CreateDynamicBox(4_r * kPreslidingDisplacement);
  auto* cube = CreateCube(Real3{0_r, 1_r, 0_r});
  cube->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  MakeDifferentiable();
  ExpectMatch(
      Run({.pushes = {{.actor = box, .force = FrictionLimit({cube, box})}},
           .loss = {ContactForceLoss(cube)},
           .restSteps = 3},
          {0.5_r}));
}

INSTANTIATE_TEST_SUITE_P(ThreadingVariations, FrictionAnchorGradient, testing::Values(0, -1));

#endif // MOCHI_USE_DOUBLE_PRECISION
