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

#include <mochi_core/solvers/actor_preconditioner.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/src/mochi_common_components.h>
#include <mochi_physics/src/mochi_discretization_components.h>
#include <mochi_physics/src/mochi_island.h>
#include <mochi_physics/src/mochi_simulation.h>
#include <mochi_physics/src/mochi_snle.h>
#include <mochi_physics/src/mochi_solve.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <initializer_list>
#include <utility>

using namespace mochi;
using namespace mochi::experimental;

class ActorPreconditionerTest : public test::MochiSceneTestBase {
 protected:
  [[nodiscard]] ShapeHandle CreateTetShape() {
    return test::CreateUnitCubeTetMeshShape(_mochiContext);
  }

  Actor* CreateRigid(bool isStatic = false) {
    RigidActorParams params;
    params.shape = CreateTetShape();
    params.isStatic = isStatic;
    params.colliderType = ColliderType::None;
    return _scene->CreateRigidActor(params, test::ExpectOK{});
  }

  Actor* CreateSoft() {
    SoftActorParams params;
    params.shape = CreateTetShape();
    return _scene->CreateSoftActor(params, test::ExpectOK{});
  }

  Actor* CreateNestedSoft() {
    SoftSkinnedActorParams params;
    params.skeletonParams.joints = {{.type = ArticulatedJointType::Free}};
    params.skeletonParams.links = {
        {.parentLink = -1, .shape = CreateTetShape(), .colliderType = ColliderType::None}};
    auto& softParams = params.softParams.push_back();
    softParams.shape = test::CreateUnitCubeTetSoftShape(_mochiContext);
    softParams.hasGravity = false;
    auto* actor = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
    return _scene->GetActor(actor->GetNestedSoftActors(test::ExpectOK{}).front());
  }

  Actor* CreateShell() {
    auto [coordinates, connectivity] = test::CreateMinimalTriMeshUnitCube();
    ShellActorParams params;
    params.shape = _mochiContext->CreateTriMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});
    return CreateShellActor(_scene, params, test::ExpectOK{});
  }

  Actor* CreateRod() {
    DynamicArray<Real3> const nodes = {{0_r, 0_r, 0_r}, {1_r, 0_r, 0_r}};
    DynamicArray<Real3> const axes = {{0_r, 1_r, 0_r}};
    RodActorParams params;
    params.shape =
        CreatePolylineShape(_mochiContext, nodes, axes, /*isClosedLoop*/ false, test::ExpectOK{});
    return _scene->CreateRodActor(params, test::ExpectOK{});
  }

  Actor* CreateArticulated(ArticulatedJointType jointType) {
    ArticulatedActorParams params;
    params.joints = {{.type = jointType}};
    params.links = {
        {.parentLink = -1, .shape = CreateTetShape(), .colliderType = ColliderType::None}};
    return _scene->CreateArticulatedActor(params, test::ExpectOK{});
  }

  void ConfigurePerActorSolve() {
    auto solverParams = _scene->GetSolverParams();
    solverParams.linearSolver.solverType = LinearSolverType::CG;
    solverParams.linearSolver.preconditionerType = PreconditionerType::PerActor;
    _scene->SetSolverParams(solverParams, test::ExpectOK{});
  }

  static void ExpectSupportedTypes(
      std::initializer_list<Actor*> actors,
      std::initializer_list<PreconditionerType> supportedTypes) {
    for (auto* actor : actors) {
      for (int value = 0; value < static_cast<int>(PreconditionerType::Count); ++value) {
        auto const type = static_cast<PreconditionerType>(value);
        Error error;
        SetActorPreconditionerType(actor, type, error);
        EXPECT_EQ(Contains(supportedTypes, type), error.IsOK());
      }
    }
  }
};

TEST_F(ActorPreconditionerTest, PassesNodePositionsAndSupportsOnlyCompatibleTypes) {
  auto* rigid = CreateRigid();
  auto* soft = CreateSoft();
  auto* nestedSoft = CreateNestedSoft();
  auto* shell = CreateShell();
  auto* rod = CreateRod();
  auto* articulated = CreateArticulated(ArticulatedJointType::Free);
  auto* reducedSoft = CreateSoft();
  auto* reducedNestedSoft = CreateNestedSoft();

  // Full-order soft actors, nested ones included, and shell actors give their preconditioners views
  // of their reference coordinates and stage-start displacements. Other actors, reduced ones
  // included, give none.
  _scene->Step(0.01_r);
  auto& reg = GetRegistry();
  reg.get<CActorSnle>(GetEntity(reducedSoft)).EnableReduced(Matrix<real>::Zero(1, 1));
  reg.get<CActorSnle>(GetEntity(reducedNestedSoft)).EnableReduced(Matrix<real>::Zero(1, 1));
  int numWithPositions = 0;
  for (entt::entity const island : reg.view<CIslandDescendants const>()) {
    auto const& descendants = reg.get<CIslandDescendants const>(island);
    auto problem = solver::InitIslandAssembly(reg, island, descendants, /*syncCurrentState*/ true);
    problem.UpdateObjResDRes({.assemObj = false, .assemDRes = true});
    for (entt::entity const actor : descendants.actors) {
      auto const* actorSnle = reg.try_get<CActorSnle const>(actor);
      if (actorSnle == nullptr || !actorSnle->useInSolver) {
        continue;
      }
      auto const it = std::ranges::find(
          problem.actorPreconditioners,
          reg.get<CDofOffset const>(actor).dofsOffset,
          &ActorPreconditionerInfo<real>::offset);
      ASSERT_NE(problem.actorPreconditioners.end(), it);
      auto const& [reference, displacement] = it->nodePositions;
      if (!actorSnle->UseReduced() && reg.any_of<TagSoftActor, TagShellActor>(actor)) {
        auto const expectedReference =
            Flatten(reg.get<CSimplicialMesh const>(actor).mesh->GetNodeCoordinates());
        auto const expectedDisplacement =
            reg.get<CDisplacementSlice<real, TimeStep::StageStart> const>(actor)
                .value.GetConstSpan();
        EXPECT_EQ(expectedReference.data(), reference.data());
        EXPECT_EQ(expectedReference.size(), reference.size());
        EXPECT_EQ(expectedDisplacement.data(), displacement.data());
        EXPECT_EQ(expectedDisplacement.size(), displacement.size());
        ++numWithPositions;
      } else {
        EXPECT_TRUE(reference.empty() && displacement.empty());
      }
    }
  }
  EXPECT_EQ(3, numWithPositions);

  auto* staticRigid = CreateRigid(/*isStatic*/ true);
  auto* zeroDofArticulated = CreateArticulated(ArticulatedJointType::Hard);
  auto const links = articulated->GetNestedLinkActors(test::ExpectOK{});
  ASSERT_FALSE(links.empty());
  auto* articulatedLink = _scene->GetActor(links.front());

  ExpectSupportedTypes({nullptr, staticRigid, zeroDofArticulated, articulatedLink}, {});
  ExpectSupportedTypes(
      {rigid, articulated, reducedSoft, reducedNestedSoft},
      {PreconditionerType::Jacobi, PreconditionerType::SymInverse});

  // Warnings before this scope fail the test.
  test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
  ExpectSupportedTypes(
      {soft, nestedSoft, shell},
      {PreconditionerType::Jacobi,
       PreconditionerType::BlockJacobi,
       PreconditionerType::AMG,
       PreconditionerType::SymInverse});
  ExpectSupportedTypes(
      {rod},
      {PreconditionerType::Jacobi,
       PreconditionerType::BlockJacobi,
       PreconditionerType::SymInverse,
       PreconditionerType::ILU0});
  EXPECT_EQ(4, expectWarning._messagesReceived);
}

TEST_F(ActorPreconditionerTest, AppliesTypeAndPreservesItOnRejection) {
  ConfigurePerActorSolve();

  auto* actor = CreateSoft();
  auto const actorEntity = GetEntity(actor);
  auto const getActorSnle = [&]() -> CActorSnle& {
    return GetRegistry().get<CActorSnle>(actorEntity);
  };

  SetActorPreconditionerType(actor, PreconditionerType::AMG, test::ExpectOK{});
  _scene->Step(0.01_r);
  ASSERT_NE(nullptr, getActorSnle().GetActivePreconditioner());
  EXPECT_EQ(PreconditionerType::AMG, getActorSnle().GetActivePreconditioner()->GetType());

  SetActorPreconditionerType(actor, PreconditionerType::ColoredSSOR, test::ExpectNotOK{});
  EXPECT_EQ(PreconditionerType::AMG, getActorSnle().GetActivePreconditionerTypeOverride());

  SetActorPreconditionerType(actor, PreconditionerType::BlockJacobi, test::ExpectOK{});
  EXPECT_EQ(PreconditionerType::BlockJacobi, getActorSnle().GetActivePreconditionerTypeOverride());
  _scene->Step(0.01_r);
  ASSERT_NE(nullptr, getActorSnle().GetActivePreconditioner());
  EXPECT_EQ(PreconditionerType::BlockJacobi, getActorSnle().GetActivePreconditioner()->GetType());

  // Each representation keeps its own selection across ROM switches.
  getActorSnle().EnableReduced(Matrix<real>::Zero(1, 1));
  SetActorPreconditionerType(actor, PreconditionerType::Jacobi, test::ExpectOK{});
  getActorSnle().DisableReduced();
  EXPECT_EQ(PreconditionerType::BlockJacobi, getActorSnle().GetActivePreconditionerTypeOverride());
  getActorSnle().EnableReduced(Matrix<real>::Zero(1, 1));
  EXPECT_EQ(PreconditionerType::Jacobi, getActorSnle().GetActivePreconditionerTypeOverride());
}

TEST_F(ActorPreconditionerTest, ArticulatedSelectionsSurviveFirstStepInitialization) {
  ConfigurePerActorSolve();

  DynamicArray<std::pair<entt::entity, PreconditionerType>> actors;
  for (auto const type : {PreconditionerType::Jacobi, PreconditionerType::SymInverse}) {
    auto* actor = CreateArticulated(ArticulatedJointType::Free);
    actors.emplace_back(GetEntity(actor), type);
    SetActorPreconditionerType(actor, type, test::ExpectOK{});
  }

  _scene->Step(0.01_r);

  for (auto const& [actor, type] : actors) {
    auto& actorSnle = GetRegistry().get<CActorSnle>(actor);
    ASSERT_TRUE(actorSnle.UseReduced());
    EXPECT_EQ(type, actorSnle.GetActivePreconditionerTypeOverride());
    ASSERT_NE(nullptr, actorSnle.GetActivePreconditioner());
    EXPECT_EQ(type, actorSnle.GetActivePreconditioner()->GetType());
  }
}
