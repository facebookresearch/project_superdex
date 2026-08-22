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

#include "mochi_differentiable_test_utils.h"
#include "mochi_physics_test_fixture.h"

#include <mochi_core/utils/defer.h>
#include <mochi_physics/src/mochi_contact.h>
#include <mochi_physics/src/mochi_differentiable.h>

using namespace mochi;
using namespace mochi::diffsim;
using namespace mochi::test;

// The scene assets used by these tests are not shipped externally.
#if MOCHI_USE_DOUBLE_PRECISION && MOCHI_INTERNAL
#define MOCHI_USE_DOUBLE_AND_INTERNAL 1
#else
#define MOCHI_USE_DOUBLE_AND_INTERNAL 0
#endif

namespace {
real constexpr kDt = 0.01_r;
} // namespace

/***************************************************************************************************
  Prepare back propagation for contact
*/

namespace {
class MochiDifferentiableContactPrepare : public MochiSceneTestBase {
 public:
  void SetUp() override {
    MochiSceneTestBase::SetUp();
    // Back-propagation requires backward Euler.
    test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
  }
};
} // namespace

// Validate that PrepareBackPropagate zero-initializes `forcePerUnitArea`
// The scene asset used by this test is not shipped externally.
TEST_IF_F(MOCHI_INTERNAL, MochiDifferentiableContactPrepare, InitializesContactForceAdjoints) {
  LoadScenePrefab(
      _scene,
      "mixed_rigid_and_articulated_contact_high_damping_control.mochi_scene",
      "ArticulatedActor1/ArticulatedActor/Link2_Horizontal");
  MakeSceneDifferentiable(_scene, test::ExpectOK{});

  auto const& reg = GetRegistry();
  _scene->ForEachActor([&](Actor* actor) {
    entt::entity const e = GetEntity(actor);
    if (reg.all_of<CContactSamples<TimeStep::Current> const>(e)) {
      actor->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
    }
  });

  StateHandle statePre = _scene->CaptureState(test::ExpectOK{});
  MOCHI_DEFER(_scene->ReleaseState(statePre));
  _scene->Step(kDt);
  StateHandle statePost = _scene->CaptureState(test::ExpectOK{});
  MOCHI_DEFER(_scene->ReleaseState(statePost));

  PrepareBackPropagate(_scene, statePost, statePre, test::ExpectOK{});

  int numPreparedContacts = 0;
  reg.view<CQueryActorContactForces const>().each(
      [&](entt::entity e, CQueryActorContactForces const&) {
        auto verifyCollisionResult = [&](ContactDetectionResult const& collisionResult) {
          for (Real3 const& forceAdjoint : collisionResult.forcePerUnitArea) {
            EXPECT_EQ(forceAdjoint, Real3{});
          }
          numPreparedContacts += isize(collisionResult.forcePerUnitArea);
        };

        auto const& activeCollisionsAsync =
            reg.get<CActiveCollisions<ContactType::Async, TimeStep::Current> const>(e);
        for (auto const& collision : activeCollisionsAsync) {
          verifyCollisionResult(collision.collisionResult);
        }
        auto const& activeCollisionsSync =
            reg.get<CActiveCollisions<ContactType::Sync, TimeStep::Current> const>(e);
        for (auto const& collision : activeCollisionsSync) {
          verifyCollisionResult(collision.collisionResult);
        }
        if (auto const* colliderJacs = reg.try_get<CCollJacs<CollRole::Collider> const>(e)) {
          for (auto const& jac : *colliderJacs) {
            verifyCollisionResult(*jac.query);
          }
        }
      });
  EXPECT_GT(numPreparedContacts, 0);
}

/***************************************************************************************************
  Chain-rule of contact adjoints
*/

namespace {
class MochiDifferentiableContact : public MochiSceneTestBase {
 public:
  void SetUp() override {
    MochiSceneTestBase::SetUp();

    // Back-propagation requires backward Euler.
    test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);

    MakeSceneDifferentiable(_scene, test::ExpectOK{});

    auto simParams = _scene->GetSolverParams();
    simParams.nonLinearSolver.maxIter = 15;
    simParams.nonLinearSolver.absTol = BackPropagationSolverParams{}.outerSolverAbsTol;
    simParams.nonLinearSolver.relTol = BackPropagationSolverParams{}.outerSolverRelTol;
    simParams.linearSolver.absTol = BackPropagationSolverParams{}.innerSolverAbsTol;
    simParams.experimentalEval.fittedSaturationHessian = SaturationHessianParams::All(false);
    _scene->SetSolverParams(simParams, test::ExpectOK{});
  }

 protected:
  template <ContactType kContactType>
  void ForEachCollisionResult(auto const& fn) {
    entt::registry& reg = test::GetRegistry(_scene);
    if (auto* collisions =
            reg.try_get<CActiveCollisions<kContactType, TimeStep::Current>>(_queryEntity)) {
      for (auto& collision : *collisions) {
        fn(collision.collisionResult);
      }
    }
    if constexpr (kContactType == ContactType::Sync) {
      if (auto* colliderJacs = reg.try_get<CCollJacs<CollRole::Collider>>(_queryEntity)) {
        for (auto& jac : *colliderJacs) {
          fn(*jac.query);
        }
      }
    }
  }

  template <ContactType kContactType>
  int CountContacts() {
    int count = 0;
    ForEachCollisionResult<kContactType>(
        [&](ContactDetectionResult const& result) { count += isize(result.forcePerUnitArea); });
    return count;
  }

  template <ContactType kContactType>
  void SetUnitContactForceAdjoints() {
    ForEachCollisionResult<kContactType>([&](ContactDetectionResult& result) {
      for (Real3& adjoint : result.forcePerUnitArea) {
        adjoint = Real3{1_r, 1_r, 1_r};
      }
    });
  }

  template <ContactType kContactType>
  real EvaluateLoss() {
    _scene->Step(kDt);
    real loss = 0_r;
    ForEachCollisionResult<kContactType>([&](ContactDetectionResult const& result) {
      for (auto const& force : result.forcePerUnitArea) {
        loss += Sum(force);
      }
    });
    return loss;
  }

  template <ContactType kContactType>
  void RunTest(
      std::string_view prefabName,
      std::string_view queryActorName,
      std::string_view gradActorName) {
    real constexpr kEps = 1e-7_r;
    real constexpr kTolerance = 2e-2_r;

    entt::registry& reg = test::GetRegistry(_scene);
    MOCHI_DEFER({ _scene->ReleaseAllStates(); });

    // Initialize the scene and actors
    LoadScenePrefab(_scene, prefabName);
    _queryActor = FindActorByName(_scene, queryActorName);
    _queryEntity = GetEntity(_queryActor);
    _gradActor = FindActorByName(_scene, gradActorName);
    _gradEntity = GetEntity(_gradActor);
    auto actorAddEpsFn = GetActorAddEpsFn(_gradActor->GetType());

    // Register contact queries
    _queryActor->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});

    // Compute the analytical gradient.
    StateHandle preState = _scene->CaptureState(test::ExpectOK{});
    _scene->Step(kDt);
    StateHandle postState = _scene->CaptureState(test::ExpectOK{});
    ResetBackPropagation(_scene, test::ExpectOK{});
    PrepareBackPropagate(_scene, postState, preState, test::ExpectOK{});
    int const numContacts = CountContacts<kContactType>();
    EXPECT_GT(numContacts, 0);
    SetUnitContactForceAdjoints<kContactType>();
    BackPropagate(_scene, test::ExpectOK{});
    ColumnVector<real> analyticalInternal = reg.get<CDiffStateGrad const>(_gradEntity).value;
    ColumnVector<real> analytical =
        ConvertActorGradientInternalToExternal(_gradActor, analyticalInternal);

    // Compute the finite difference gradient wrt the input state through the full backpropagation.
    ColumnVector<real> finiteDiff = analytical;
    for (int dim = 0; dim < analytical.Rows(); ++dim) {
      _scene->RestoreState(preState, /*releaseImmediately*/ false, test::ExpectOK{});
      actorAddEpsFn(_gradActor, dim, kEps);
      real const lossPlus = EvaluateLoss<kContactType>();
      int numContactsTest = CountContacts<kContactType>();
      EXPECT_EQ(numContactsTest, numContacts); // The test is fragile if #contacts varies

      _scene->RestoreState(preState, /*releaseImmediately*/ false, test::ExpectOK{});
      actorAddEpsFn(_gradActor, dim, -kEps);
      real const lossMinus = EvaluateLoss<kContactType>();
      numContactsTest = CountContacts<kContactType>();
      EXPECT_EQ(numContactsTest, numContacts); // The test is fragile if #contacts varies

      finiteDiff(dim) = (lossPlus - lossMinus) / (2_r * kEps);
    }

    // Compare
    real analyticalNorm = analytical.Norm();
    real finiteDiffNorm = finiteDiff.Norm();
    ColumnVector<real> diff = analytical - finiteDiff;
    auto diffNorm = diff.Norm();
    EXPECT_NEAR(diffNorm / Max(analyticalNorm, finiteDiffNorm), 0_r, kTolerance);
  }

  Actor* _queryActor = {};
  Actor* _gradActor = {};
  entt::entity _queryEntity = {};
  entt::entity _gradEntity = {};
};
} // namespace

TEST_IF_F(MOCHI_USE_DOUBLE_AND_INTERNAL, MochiDifferentiableContact, ConsistencyTestRigidAsync) {
  RunTest<ContactType::Async>("rigid_cube_on_plane_frictionless.mochi_scene", "Cube", "Cube");
}

TEST_IF_F(MOCHI_USE_DOUBLE_AND_INTERNAL, MochiDifferentiableContact, ConsistencyTestRigidSync) {
  RunTest<ContactType::Sync>(
      "two_rigid_cubes_on_plane_frictionless.mochi_scene", "TopCube", "TopCube");
}

TEST_IF_F(
    MOCHI_USE_DOUBLE_AND_INTERNAL,
    MochiDifferentiableContact,
    ConsistencyTestArticulatedAsync) {
  RunTest<ContactType::Async>(
      "articulated_actor_on_plane.mochi_scene", "ArticulatedActor/Link0_Root", "ArticulatedActor");
}

TEST_IF_F(
    MOCHI_USE_DOUBLE_AND_INTERNAL,
    MochiDifferentiableContact,
    ConsistencyTestArticulatedSync) {
  RunTest<ContactType::Sync>(
      "articulated_and_rigid_actor_contact.mochi_scene",
      "ArticulatedActor/Link2_Horizontal",
      "ArticulatedActor");
}

// The contact-force backward pass must not crash when an active collision has zero contacts.
// Broad-phase pairs are retained (e.g. for warm-starting) even when narrow-phase finds no contacts,
// leaving `jacColliderFromWorld` empty.
// The scene asset used by this test is not shipped externally.
TEST_IF_F(MOCHI_INTERNAL, MochiDifferentiableContact, ContactForceBackwardNoContacts) {
  MOCHI_DEFER({ _scene->ReleaseAllStates(); });

  LoadScenePrefab(_scene, "rigid_cube_on_plane_frictionless.mochi_scene");
  _queryActor = FindActorByName(_scene, "Cube");
  _queryEntity = GetEntity(_queryActor);
  _queryActor->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});

  StateHandle preState = _scene->CaptureState(test::ExpectOK{});
  _scene->Step(kDt);
  StateHandle postState = _scene->CaptureState(test::ExpectOK{});
  PrepareBackPropagate(_scene, postState, preState, test::ExpectOK{});

  // Emulate broad-phase pairs retained without contacts: clear each active collision so it has zero
  // contacts and an empty `jacColliderFromWorld`, matching the state left by narrow-phase culling.
  int numCleared = 0;
  auto clearResult = [&](ContactDetectionResult& result) {
    result.Clear();
    ++numCleared;
  };
  ForEachCollisionResult<ContactType::Async>(clearResult);
  ForEachCollisionResult<ContactType::Sync>(clearResult);
  ASSERT_GT(numCleared, 0)
      << "Expected at least one active collision to exercise the backward pass.";

  // Backward pass over the zero-contact collisions must not crash.
  DynamicArray<real> const gradOutput(RigidSize::kDTrans, 1_r);
  GetContactForceWorldBackward(_queryActor, gradOutput, test::ExpectOK{});
}
