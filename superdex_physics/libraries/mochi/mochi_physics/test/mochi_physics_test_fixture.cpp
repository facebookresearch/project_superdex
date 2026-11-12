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

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/test/batched_work_test_utils.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_physics/src/mochi_active_collisions.h>
#include <mochi_physics/src/mochi_contact_anchors.h>
#include <mochi_physics/src/mochi_contact_pair_params.h>
#include <mochi_physics/src/mochi_island.h>
#include <mochi_physics/src/mochi_shell.h>
#include <mochi_physics/src/mochi_simulation.h>
#include <mochi_physics/src/mochi_soft.h>
#include <mochi_physics/src/mochi_soft_rom_components.h>
#include <mochi_physics/src/mochi_soft_skinned.h>
#include <mochi_physics/src/mochi_solve.h>

#include <algorithm>
#include <functional>
#include <utility>
#include <variant>

namespace mochi::test {

MochiContextTestBase::~MochiContextTestBase() {
  MOCHI_ASSERT(
      _mochiContext == nullptr,
      "Class inheriting from MochiContextTestBase has not correctly called TearDown()!"
      " This can result in memory leaks/corruption! Please clean up after yourself!");
}

/*******************************************************************************************
  MochiPhysicsTest (test fixture)
*/

void MochiContextTestBase::SetUp() {
  // Create mochi::Context
  _mochiContext = mochi::CreateContext(_numWorkerThreads);
  EXPECT_NE((Context*)nullptr, _mochiContext);

  // Take the logging callback that has been registered for this test project
  // and make sure it also applies with the mochi_physics DLL (does nothing
  // if mochi_physics is statically linked with this test).
  Context::SetLogCallback(mochi::GetLogCallback());
}

void mochi::test::MochiContextTestBase::TearDown() {
  mochi::DestroyContext(_mochiContext);
  _mochiContext = nullptr;
}

/*******************************************************************************************
  MochiContextTestWithParam (parameterized test fixture)
*/

void MochiContextTestWithParam::SetUp() {
  // Create mochi::Context with some numer of worker threads
  _numWorkerThreads = GetParam().numWorkerThreads;
  Class::SetUp();

  // Apply other parameters
  _mochiContext->SetIsSingleThreaded(GetParam().isSingleThreadedMode);
}

/*******************************************************************************************
  MochiSceneTest
*/

void MochiSceneTestBase::SetUp() {
  MochiContextTestBase::SetUp();

  _scene = _mochiContext->CreateScene("MochiPhysicsTest");
  EXPECT_NE((mochi::Scene*)nullptr, _scene);
}

void MochiSceneTestBase::TearDown() {
  EXPECT_NE((mochi::Scene*)nullptr, _scene);
  _mochiContext->DestroyScene(_scene);
  _scene = nullptr;

  MochiContextTestBase::TearDown();
}

/*******************************************************************************************
  MochiAsyncSceneTest
*/

MochiAsyncSceneTestBase::MochiAsyncSceneTestBase() {
  // Must have at least one worker thread to create an AsyncScene
  _numWorkerThreads = 1;
}

void MochiAsyncSceneTestBase::SetUp() {
  MochiContextTestBase::SetUp();

  _asyncScene = _mochiContext->CreateAsyncScene("MochiPhysicsTest", ExpectOK{});
  EXPECT_NE((mochi::AsyncScene*)nullptr, _asyncScene);
}

void MochiAsyncSceneTestBase::TearDown() {
  EXPECT_NE((mochi::AsyncScene*)nullptr, _asyncScene);
  _mochiContext->DestroyAsyncScene(_asyncScene);
  _asyncScene = nullptr;

  MochiContextTestBase::TearDown();
}

/*******************************************************************************************
  ExpectLoggingInScope
*/

ExpectLoggingInScope::ExpectLoggingInScope(Context* mochiContext, LogChannel channel)
    : _mochiContext(mochiContext),
      _channel(channel),
      _prevCallbackForMochiPhysics(mochi::Context::GetLogCallback()) {
  LogFn ourCallback = [this](LogChannel channel, char const* message, char const* file, int line) {
    if (channel == _channel) {
      ++_messagesReceived;
    } else {
      _prevCallbackForMochiPhysics(channel, message, file, line);
    }
  };
  mochi::Context::SetLogCallback(ourCallback);
}

ExpectLoggingInScope::~ExpectLoggingInScope() {
  mochi::Context::SetLogCallback(_prevCallbackForMochiPhysics);
  if (_messagesReceived == 0) {
    ADD_FAILURE() << "Expected a log message, but it never happened.";
  }
}

// Checks one body's assembly at a uniform translation, where the displacement u, stage-start
// displacement u₀ and velocity v₀ take the same value at every node. There the consistent mass
// matrix acts as the lumped mass m, stress and stiffness damping exert no force, and the stress
// energy is a constant E₀. For each DoF i, of component c, with α the actor's mass-damping
// coefficient and g the gravity in the body's frame:
//   residual_i = m_i·((u − u₀ − dt·v₀)_c/dt² + α·(u − u₀)_c/dt − g_c),
//   (dresidual·1)_i = (1/dt² + α/dt)·m_i,
//   objective = E₀ + Σᵢ m_i·((u − u₀ − dt·v₀)_c²/(2·dt²) + α·(u − u₀)_c²/(2·dt) − g_c·(u − u₀)_c).
// Without inertia, the terms in dt and α drop out; without gravity, those in g. A nested soft
// actor's posed body is assembled on the skinned displacement layer, in the world frame. Every
// output combination is assembled, into outputs that hold stale values.
static void ExpectBodyUniformTranslationAssembly(
    Scene* scene,
    Actor* actor,
    real relTol,
    bool posed,
    bool hasInertia,
    bool hasGravity) {
  constexpr Real3 kDispl = {0.3_r, -0.2_r, 0.1_r};
  constexpr Real3 kStageStartDispl = {-0.1_r, 0.25_r, 0.05_r};
  constexpr Real3 kStageStartVel = {2_r, -1_r, 3_r};
  auto& reg = GetRegistry(scene);
  entt::entity const e = GetEntity(reg, actor->GetHandle(), ExpectOK{});
  constexpr auto kSkinned = DisplacementLayer::Skinned;
  ColumnVector<real>& displ = posed
      ? reg.get<CDisplacementSlice<real, TimeStep::Current, kSkinned>>(e).value
      : reg.get<CDisplacementSlice<real, TimeStep::Current>>(e).value;
  ColumnVector<real>& stageStartDispl = posed
      ? reg.get<CDisplacementSlice<real, TimeStep::StageStart, kSkinned>>(e).value
      : reg.get<CDisplacementSlice<real, TimeStep::StageStart>>(e).value;
  auto const setUniform = [](ColumnVector<real>& values, Real3 const& value) {
    for (int i = 0; i < values.Rows(); ++i) {
      values[i] = value[i % kSpaceDim3];
    }
  };
  setUniform(stageStartDispl, kStageStartDispl);
  // The posed body has a stage-start velocity only with inertia.
  if (hasInertia) {
    setUniform(
        posed ? reg.get<CVelocitySlice<real, TimeStep::StageStart, kSkinned>>(e).value
              : reg.get<CVelocitySlice<real, TimeStep::StageStart>>(e).value,
        kStageStartVel);
  }

  real const dt = reg.get<CTimeIntegratorState const>(e).dtStage;
  real const alpha = reg.all_of<TagShellActor>(e)
      ? reg.get<CShellMaterialParams const>(e).massDampingCoefficient
      : reg.get<CSoftMaterialParams const>(e).massDampingCoefficient;
  real const inertiaScale = hasInertia ? 1_r / Sqr(dt) : 0_r;
  real const massDampingScale = hasInertia ? alpha / dt : 0_r;
  TransformRT const worldFromLocal = posed ? TransformRT::Identity() : actor->GetRootTransform();
  Real3 const gravity =
      hasGravity ? worldFromLocal.TransformDirectionInverse(scene->GetGravity()) : Real3{};
  Real3 const predTarget = kStageStartDispl + kStageStartVel * dt;
  auto const& lumpedMass = reg.get<CLumpedMassMatrix const>(e).values;
  int const numDofs = isize(lumpedMass);
  // Objective at the uniform displacement u, but for the stress energy.
  auto const objectiveBesidesStress = [&](Real3 const& u) {
    double objective = 0.0;
    for (int i = 0; i < numDofs; ++i) {
      int const c = i % kSpaceDim3;
      objective += static_cast<double>(
          lumpedMass[i] *
          (inertiaScale * Sqr(u[c] - predTarget[c]) / 2_r +
           massDampingScale * Sqr(u[c] - kStageStartDispl[c]) / 2_r -
           gravity[c] * (u[c] - kStageStartDispl[c])));
    }
    return objective;
  };
  ColumnVector<real> expectedRes(numDofs);
  ColumnVector<real> expectedDResRowSums(numDofs);
  for (int i = 0; i < numDofs; ++i) {
    int const c = i % kSpaceDim3;
    expectedRes[i] = lumpedMass[i] *
        (inertiaScale * (kDispl[c] - predTarget[c]) +
         massDampingScale * (kDispl[c] - kStageStartDispl[c]) - gravity[c]);
    expectedDResRowSums[i] = (inertiaScale + massDampingScale) * lumpedMass[i];
  }

  // CSoftSkinnedUnposedSnle holds the posed body's terms, before the skinning projection.
  ActorSnle& snle = posed ? static_cast<ActorSnle&>(reg.get<CSoftSkinnedUnposedSnle>(e))
                          : static_cast<ActorSnle&>(reg.get<CActorSnle>(e));
  auto const assemble = [&](AssemblyParams const& params) {
    if (posed) {
      // Nothing else assembles here, so the soft and articulated assemblies count as done. This
      // also writes the projected terms into the interaction SNLE and the soft and articulated
      // dresiduals, which go unchecked.
      ecs::InvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
          skinned::EntityAssembleBody, reg, e, std::cref(params), TaskSemaphore{}, TaskSemaphore{});
    } else if (reg.all_of<TagShellActor>(e)) {
      ecs::InvokeOnEntity(shell::AssembleBody, reg, e, std::cref(params));
    } else {
      ecs::InvokeOnEntity(soft::AssembleBody, reg, e, std::cref(params));
    }
  };
  // The stress energy is the same at every translation. Take it at the predicted target.
  setUniform(displ, predTarget);
  assemble({.assemObj = true, .assemRes = false, .assemDRes = false});
  double const stressEnergy = snle.objective - objectiveBesidesStress(predTarget);

  setUniform(displ, kDispl);
  for (auto const cfg : fem::kAllOutputConfigs) {
    snle.objective = 1e3;
    snle.fullResidual.SetConstant(1e3_r);
    Fill(GetValues(snle.fullDResidual), 1e3_r);
    assemble({.assemObj = cfg.energy, .assemRes = cfg.residual, .assemDRes = cfg.dresidual});
    if (cfg.energy) {
      EXPECT_NEAR_RTOL(
          stressEnergy + objectiveBesidesStress(kDispl),
          snle.objective,
          static_cast<double>(relTol));
    }
    if (cfg.residual) {
      ColumnVector<real> const resError = snle.fullResidual - expectedRes;
      EXPECT_LE(resError.Norm(), relTol * expectedRes.Norm());
    }
    // Without inertia, the dresidual holds no mass term to check.
    if (cfg.dresidual && hasInertia) {
      ColumnVector<real> const dresRowSumsError =
          ToMatrix(std::get<BlockSparseMatrix<real, 3>>(snle.fullDResidual)) *
              ColumnVector<real>::Ones(numDofs) -
          expectedDResRowSums;
      EXPECT_LE(dresRowSumsError.Norm(), relTol * expectedDResRowSums.Norm());
    }
  }
}

void ExpectUniformTranslationAssembly(Scene* scene, Actor* actor, real relTol) {
  auto& reg = GetRegistry(scene);
  entt::entity const e = GetEntity(reg, actor->GetHandle(), ExpectOK{});
  // Shells always have inertia. Only a nested soft actor puts terms on its posed body, as its
  // skinned energy records.
  bool const actorHasInertia = reg.any_of<TagShellActor, TagUseInertia>(e);
  bool const actorHasGravity = reg.all_of<TagUseGravity>(e);
  EXPECT_TRUE(actorHasInertia || actorHasGravity) << "No body has inertia or gravity to check.";
  auto const* skinnedEnergy = reg.try_get<CSkinnedEnergy const>(e);
  bool const posedInertia = skinnedEnergy != nullptr && skinnedEnergy->inertia;
  bool const posedGravity = skinnedEnergy != nullptr && skinnedEnergy->gravity;
  for (bool const posed : {false, true}) {
    bool const hasInertia = actorHasInertia && posedInertia == posed;
    bool const hasGravity = actorHasGravity && posedGravity == posed;
    if (hasInertia || hasGravity) {
      ExpectBodyUniformTranslationAssembly(scene, actor, relTol, posed, hasInertia, hasGravity);
    }
  }
}

ColumnVector<real> SumNonPotentialResiduals(SnleProblem<real> const& problem) {
  ColumnVector<real> sum = ColumnVector<real>::Zero(problem.GetDofsSize());
  for (auto const& [offset, residual] : problem.nonPotentialResiduals) {
    sum.MiddleRows(offset, residual->Rows()) += *residual;
  }
  return sum;
}

void ExpectDissipationIsNonPotential(
    entt::registry& reg,
    SnleProblem<real>& problem,
    std::optional<DissipationCoefficients> coefficients) {
  auto& simParams = reg.ctx<CSimulationParams>();
  LineSearchType const lineSearchType = simParams.nonLinearSolver.lineSearchType;
  DynamicArray<std::pair<entt::entity, ContactParams>> contactParams;
  for (auto [e, params] : reg.view<CContactParams const>().each()) {
    contactParams.emplace_back(e, params);
  }

  auto const setCoefficients = [&](DissipationCoefficients const& c) {
    for (auto [e, params] : reg.view<CContactParams>().each()) {
      params.coulombFrictionCoefficient = c.coulomb;
      params.viscousFrictionCoefficient = c.viscous;
      params.normalViscousDampingCoefficient = c.damping;
    }
  };

  struct Terms {
    double objective = 0.0;
    double nonPotentialEnergy = 0.0;
    ColumnVector<real> nonPotentialResidual;
  };
  auto const getTerms = [&] {
    return Terms{
        .objective = problem.GetObjective(),
        .nonPotentialEnergy = problem.nonPotentialEnergy,
        .nonPotentialResidual = SumNonPotentialResiduals(problem)};
  };

  auto const assemble = [&](AssemblyParams const& params) {
    problem.InvalidateCachedData();
    problem.UpdateObjResDRes(params);
  };

  simParams.nonLinearSolver.lineSearchType = LineSearchType::WolfeStrong;
  if (coefficients) {
    setCoefficients(*coefficients);
  }

  assemble({.assemObj = true});
  Terms const on = getTerms();

  problem.UpdateObjResDRes({.assemObj = true, .assemRes = true, .assemDRes = true});
  Terms const cached = getTerms();
  EXPECT_EQ(on.nonPotentialEnergy, cached.nonPotentialEnergy);
  ColumnVector<real> const cachedChange = cached.nonPotentialResidual - on.nonPotentialResidual;
  EXPECT_EQ(cachedChange.Norm(), 0_r);

  assemble({.assemObj = true, .assemRes = true});
  Terms const combined = getTerms();
  ColumnVector<real> const residual = problem.GetResidual();

  simParams.nonLinearSolver.lineSearchType = LineSearchType::ResidualNorm;
  assemble({.assemObj = true, .assemRes = true});
  EXPECT_EQ(problem.nonPotentialEnergy, 0.0);
  EXPECT_TRUE(problem.nonPotentialResiduals.empty());
  double const objectiveNoReport = problem.GetObjective();
  ColumnVector<real> const residualNoReport = problem.GetResidual();

  simParams.nonLinearSolver.lineSearchType = LineSearchType::WolfeStrong;
  setCoefficients({});
  auto& overrideTable = reg.ctx<CContactPairParamsOverrideTable>();
  CContactPairParamsOverrideTable::Records const overrides = overrideTable.GetRecords();
  overrideTable.DisableDissipation();

  assemble({.assemObj = true, .assemRes = true});
  Terms const off = getTerms();
  ColumnVector<real> const residualOff = problem.GetResidual();

  // Assemblies of different requests may evaluate an energy in the precision of real. Those of the
  // same request, with friction and damping and without, or reporting them and not, differ only in
  // summation order.
  double const energyScale = Max(Abs(combined.objective), Abs(off.objective));
  double const requestTol = (MOCHI_USE_DOUBLE_PRECISION ? 1e-10 : 1e-6) * energyScale;
  double const energyTol = 1e-10 * energyScale;
  real const residualTol = (MOCHI_USE_DOUBLE_PRECISION ? 1e-10_r : 2e-5_r) *
      Max(residual.Norm(), residualOff.Norm(), combined.nonPotentialResidual.Norm());

  EXPECT_NEAR(objectiveNoReport, combined.objective, energyTol);
  ColumnVector<real> const reportChange = residual - residualNoReport;
  EXPECT_LE(reportChange.Norm(), residualTol);

  EXPECT_NEAR(on.objective, combined.objective, requestTol);
  EXPECT_NEAR(on.nonPotentialEnergy, combined.nonPotentialEnergy, requestTol);
  ColumnVector<real> const combinedChange = combined.nonPotentialResidual - on.nonPotentialResidual;
  EXPECT_LE(combinedChange.Norm(), residualTol);

  // Friction and damping act observably, so that a dropped or extra term breaks the comparisons
  // below.
  double const objectiveChange = combined.objective - off.objective;
  ColumnVector<real> const fullResidualChange = residual - residualOff;
  EXPECT_GT(Abs(objectiveChange), 10 * energyTol);
  EXPECT_GT(fullResidualChange.Norm(), 10 * residualTol);

  double const energyChange = combined.nonPotentialEnergy - off.nonPotentialEnergy;
  ColumnVector<real> const residualChange =
      combined.nonPotentialResidual - off.nonPotentialResidual;
  if (simParams.experimentalEval.explicitNormals) {
    EXPECT_NEAR(energyChange, 0.0, energyTol);
    EXPECT_LE(residualChange.Norm(), residualTol);
  } else {
    EXPECT_NEAR(objectiveChange, energyChange, energyTol);
    ColumnVector<real> const keptResidualChange = fullResidualChange - residualChange;
    EXPECT_LE(keptResidualChange.Norm(), residualTol);
  }

  simParams.nonLinearSolver.lineSearchType = lineSearchType;
  for (auto const& [e, params] : contactParams) {
    reg.get<CContactParams>(e) = params;
  }
  for (auto const& [pair, paramsOverride] : overrides) {
    overrideTable.Set(pair.first, pair.second, paramsOverride);
  }
}

void ExpectDissipationIsNonPotentialAfterTouch(Scene* scene, real dt) {
  auto& reg = GetRegistry(scene);

  // Whether pred holds for some current active collision, sync or async.
  auto const anyCollision = [&](auto const& pred) {
    auto const anyOf = [&]<bool kIsSync>() {
      for (auto [e, collisions] :
           reg.view<CActiveCollisions<kIsSync, TimeStep::Current> const>().each()) {
        if (std::ranges::any_of(collisions, pred)) {
          return true;
        }
      }
      return false;
    };
    return anyOf.template operator()<true>() || anyOf.template operator()<false>();
  };
  auto const hasContact = [&] {
    return anyCollision([](auto const& collision) { return !collision.collisionResult.Empty(); });
  };

  int constexpr kMaxSteps = 50;
  for (int step = 0; step < kMaxSteps && !hasContact(); ++step) {
    scene->Step(dt);
  }
  ASSERT_TRUE(hasContact());
  scene->Step(dt);

  // With elasto-plastic friction, the anchors advanced by the step, as the next step uses them.
  PrepareFrictionAnchors(reg, /*promotePending*/ true);
  for (entt::entity const island : reg.view<CIslandDescendants const>()) {
    SnleProblem<real> problem = solver::InitIslandAssembly(
        reg, island, reg.get<CIslandDescendants const>(island), /*syncCurrentState*/ true);
    ColumnVector<real> increment(problem.GetDofsSize());
    increment.SetRandom(/*seed*/ 3, -1e-3_r, 1e-3_r);
    solver::PostNewIncrementLocalPipeline(reg, island, AsConstView(increment), problem.solution);
    ExpectDissipationIsNonPotential(reg, problem, kDissipationCoefficients);
  }

  if (reg.ctx<CSimulationParams const>().experimentalEval.elastoPlasticFriction) {
    bool const hasAnchoredContact = anyCollision([](auto const& collision) {
      return std::ranges::any_of(
          collision.collisionResult.anchorOffset, [](Real3 const& o) { return o != Real3{}; });
    });
    EXPECT_TRUE(hasAnchoredContact) << "Elasto-plastic friction should act on anchors.";
  }
}

} // namespace mochi::test
