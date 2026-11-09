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
#include "mochi_test_scene_builders.h"

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/integration/integration_utils.h>
#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/src/mochi_capture.h>
#include <mochi_physics/src/mochi_common_components.h>
#include <mochi_physics/src/mochi_island.h>
#include <mochi_physics/src/mochi_query.h>
#include <mochi_physics/src/mochi_scene.h>
#include <mochi_physics/src/mochi_solve.h>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

using namespace mochi;
using namespace mochi::experimental;
using namespace mochi::test;

// The soft Allegro actor used by the multi-actor scene requires an internal-only asset.
#if MOCHI_USE_HDF5 && MOCHI_INTERNAL
#define MOCHI_HDF5_AND_INTERNAL 1
#else
#define MOCHI_HDF5_AND_INTERNAL 0
#endif

namespace {
// Maps an IntegrationMethod to a stable string for parameterized test names.
char const* IntegrationMethodName(IntegrationMethod method) {
  static_assert(static_cast<int>(IntegrationMethod::Count) == 8, "Update IntegrationMethodName");
  switch (method) {
    case IntegrationMethod::BackwardEuler:
      return "BackwardEuler";
    case IntegrationMethod::BDF2:
      return "BDF2";
    case IntegrationMethod::BDF3:
      return "BDF3";
    case IntegrationMethod::DIRK22:
      return "DIRK22";
    case IntegrationMethod::DIRK23:
      return "DIRK23";
    case IntegrationMethod::DIRK33:
      return "DIRK33";
    case IntegrationMethod::SymplecticDIRK12:
      return "SymplecticDIRK12";
    case IntegrationMethod::SymplecticDIRK22:
      return "SymplecticDIRK22";
    case IntegrationMethod::Count:
      return "Count";
  }
  return "Unknown";
}

struct ZeroStepCase {
  IntegrationMethod integration;
  bool elastoPlasticFriction;
};

// Each integrator with regular friction and, if it supports it, with elasto-plastic friction.
DynamicArray<ZeroStepCase> WithAndWithoutElastoPlasticFriction(
    std::initializer_list<IntegrationMethod> integrations) {
  DynamicArray<ZeroStepCase> cases;
  for (IntegrationMethod const integration : integrations) {
    cases.push_back({.integration = integration, .elastoPlasticFriction = false});
    if (EndsAtLastStage(integration)) {
      cases.push_back({.integration = integration, .elastoPlasticFriction = true});
    }
  }
  return cases;
}

std::string ZeroStepCaseName(::testing::TestParamInfo<ZeroStepCase> const& info) {
  return std::string(IntegrationMethodName(info.param.integration)) +
      (info.param.elastoPlasticFriction ? "_ElastoPlasticFriction" : "");
}

void SetElastoPlasticFriction(Scene* scene, bool enabled) {
  SolverParams params = scene->GetSolverParams();
  params.experimentalEval.elastoPlasticFriction = enabled;
  scene->SetSolverParams(params, test::ExpectOK{});
}
} // namespace

// ---------------------------------------------------------------------------------------------
// Zero-step (solve-free) assembly with exact force reproduction during scene replay.
// ---------------------------------------------------------------------------------------------

using SceneForces = std::map<std::vector<entt::entity>, ColumnVector<real>>;

// Expects the solve-free Step(0.0) island residuals to reproduce the reference (real-step)
// residuals bit-exactly, with the same island partition.
static void CheckSameForces(SceneForces const& reference, SceneForces const& actual) {
  ASSERT_EQ(reference.size(), actual.size()) << "Different island partition after restore";
  for (auto const& [key, refRes] : reference) {
    auto it = actual.find(key);
    ASSERT_NE(it, actual.end()) << "Island missing after restore";
    ASSERT_EQ(isize(refRes), isize(it->second));
    for (int i = 0; i < isize(refRes); ++i) {
      EXPECT_EQ(refRes[i], it->second[i]) << "Mismatch at island DOF " << i;
    }
  }
}

// Runs @p run and returns the assembled residual of each island, recorded through the island test
// callback and keyed by the island's actors. Islands assemble concurrently, hence the mutex.
template <typename RunT>
static SceneForces RunAndReadAssembledForces(Scene* scene, RunT&& run) {
  auto& reg = GetRegistry(scene);
  SceneForces forces;
  std::mutex mutex;
  island::SetTestCallback_AssembledResidual(
      reg, [&](entt::entity island, ColumnVectorView<real const> residual) {
        auto const& actors = reg.get<CIslandDescendants const>(island).actors;
        std::lock_guard lock(mutex);
        forces[actors] = residual;
      });
  MOCHI_DEFER(island::SetTestCallback_AssembledResidual(reg, {}));
  run();
  return forces;
}

static SceneForces StepAndReadAssembledForces(Scene* scene, double timeStepSec) {
  return RunAndReadAssembledForces(scene, [&]() { scene->Step(timeStepSec); });
}

static void
ExpectResidualMatchesGravity(ColumnVector<real> const& res, Scene* scene, Actor* actor) {
  real const residualNorm = res.Norm();
  real const expectedForce = actor->GetMass(test::ExpectOK{}) * Norm(scene->GetGravity());
  EXPECT_NEAR(residualNorm, expectedForce, 1e-2_r * expectedForce);
}

// Step(0.0) assembles forces only if a registered query requires it.
static void RegisterAssemblyQuery(Actor* actor, QueryType type = QueryType::TotalContactForce) {
  ASSERT_TRUE(QueryRequiresAssembly(type));
  ASSERT_TRUE(actor->IsQuerySupported(type));
  actor->RegisterQuery(type, test::ExpectOK{});
}

// Runs the Step(0.0) reproduction check for @p scene: caches the residual of a real step, then
// re-derives it via a solve-free Step(0.0) both directly and after a capture/restore round-trip.
static void RunZeroStepReproduction(Scene* scene, int numSteps) {
  double constexpr kDt = 0.01;

  // Run the first N - 1 steps, then one more caching the reference forces.
  AdvanceScene(scene, numSteps - 1, kDt);
  SceneForces const referenceForces = StepAndReadAssembledForces(scene, kDt);

  // Re-assemble with a zero-step call.
  CheckSameForces(referenceForces, StepAndReadAssembledForces(scene, 0.0));

  // Capture, advance further, restore, then re-assemble with a zero-step call.
  StateHandle const state = scene->CaptureState(test::ExpectOK{});
  AdvanceScene(scene, numSteps);
  scene->RestoreState(state, /*releaseImmediately*/ true, test::ExpectOK{});
  CheckSameForces(referenceForces, StepAndReadAssembledForces(scene, 0.0));
}

namespace {
class ZeroStepAssemblyTest : public MochiContextTestBase,
                             public ::testing::WithParamInterface<ZeroStepCase> {};
} // namespace

INSTANTIATE_TEST_SUITE_P(
    Integrators,
    ZeroStepAssemblyTest,
    // Single-stage only: AssembleIsland asserts numStages == 1, and multi-stage DIRK cannot be
    // reproduced from captured state (intermediate `stages` are not serialized).
    ::testing::ValuesIn(WithAndWithoutElastoPlasticFriction(
        {IntegrationMethod::BackwardEuler, IntegrationMethod::BDF2, IntegrationMethod::BDF3})),
    ZeroStepCaseName);

// The solve-free Step(0.0) assembly reproduces a real step's forces bit-exactly, for all
// single-stage integrators, friction models and actor types (directly and after a capture/restore
// round-trip). Soft actors recenter at the start of a real step, and Step(0.0) does not recenter
// again, so both assemble in the same local frame.
TEST_IF_P(MOCHI_HDF5_AND_INTERNAL, ZeroStepAssemblyTest, ReproducesForcesBitExact) {
  auto const [integration, elastoPlasticFriction] = GetParam();
  auto [scene, agent] = CreateSceneActorTypes(_mochiContext, "Scene", integration);
  SetElastoPlasticFriction(scene, elastoPlasticFriction);
  RegisterAssemblyQuery(agent, QueryType::ArticulatedControllerForce);

  RunZeroStepReproduction(scene, GetNumSteps(integration) + 2);
}

// ---------------------------------------------------------------------------------------------
// Zero-step (solve-free) force preview before the first real step (no step history).
// ---------------------------------------------------------------------------------------------

struct RigidBoxOptions {
  Real3 translation{};
  Real3 linearVelocity{};
  Real3 angularVelocity{};
  bool isStatic = false;
  bool hasGravity = true;
};

static Actor* AddRigidBox(Scene* scene, RigidBoxOptions const& options) {
  auto* context = scene->GetContext();
  real constexpr kBoxScale = 0.2_r;
  RigidActorParams params;
  params.name = "Box";
  params.layer = "Object";
  params.shape = context->LoadShapeFromFile(
      test::GetAssetPath("cube/cube_fine_mesh.mochi.json"),
      Real3{kBoxScale, kBoxScale, kBoxScale},
      TransformRT::Identity(),
      test::ExpectOK{});
  params.colliderType = ColliderType::Box;
  params.density = 1000_r;
  params.isStatic = options.isStatic;
  params.hasGravity = options.hasGravity;
  params.worldFromLocal.SetTranslation(options.translation);
  auto* actor = scene->CreateRigidActor(params, test::ExpectOK{});
  if (!options.isStatic) {
    actor->SetVelocity(options.linearVelocity, options.angularVelocity, test::ExpectOK{});
  }
  return actor;
}

static std::tuple<Scene*, Actor*> CreateFreeRigidScene(
    Context* context,
    Real3 const& linVel,
    Real3 const& angVel,
    IntegrationMethod integrationMethod = IntegrationMethod::BackwardEuler) {
  auto* scene = context->CreateScene("PreviewScene");

  SolverParams solverParams = scene->GetSolverParams();
  solverParams.integrationMethod = integrationMethod;
  scene->SetSolverParams(solverParams, test::ExpectOK{});

  auto* actor = AddRigidBox(
      scene,
      {.translation = Real3{0_r, 1_r, 0_r}, .linearVelocity = linVel, .angularVelocity = angVel});
  return {scene, actor};
}

namespace {
class ZeroStepPreviewTest : public MochiContextTestBase,
                            public ::testing::WithParamInterface<IntegrationMethod> {};
} // namespace

INSTANTIATE_TEST_SUITE_P(
    Integrators,
    ZeroStepPreviewTest,
    // All integrators: before any real step the history is empty, so every integrator takes the
    // same constant-velocity fallback (the result is integrator-independent).
    ::testing::Values(
        IntegrationMethod::BackwardEuler,
        IntegrationMethod::BDF2,
        IntegrationMethod::BDF3,
        IntegrationMethod::DIRK22,
        IntegrationMethod::DIRK23,
        IntegrationMethod::DIRK33,
        IntegrationMethod::SymplecticDIRK12,
        IntegrationMethod::SymplecticDIRK22),
    [](::testing::TestParamInfo<IntegrationMethod> const& info) {
      return std::string(IntegrationMethodName(info.param));
    });

// Before any real step there is no completed step history. Step(0.0) must still produce a force
// preview: under the constant-velocity assumption the assembled residual reduces to the net applied
// forces -f(q0, v0). For a single free-falling rigid body started at rest the only applied force is
// gravity, so the island residual must have magnitude mass * |gravity|.
// For a free rigid body the net applied force is gravity only, independent of velocity.
static void TestNoHistoryPreviewEqualsGravity(
    Context* mochiContext,
    IntegrationMethod integration,
    bool withInitialVelocity) {
  Real3 const linVel = withInitialVelocity ? Real3{0.6_r, -0.4_r, 0.7_r} : Real3{0_r, 0_r, 0_r};
  auto [scene, actor] =
      CreateFreeRigidScene(mochiContext, linVel, Real3{0_r, 0_r, 0_r}, integration);
  RegisterAssemblyQuery(actor);

  SceneForces const previewForces = StepAndReadAssembledForces(scene, 0.0);

  ASSERT_EQ(previewForces.size(), 1u) << "Step(0.0) must produce one island's forces at sim start";
  ColumnVector<real> const& res = previewForces.begin()->second;

  ExpectResidualMatchesGravity(res, scene, actor);

  mochiContext->DestroyScene(scene);
}

TEST_P(ZeroStepPreviewTest, NoHistoryAtRestPreviewEqualsGravity) {
  TestNoHistoryPreviewEqualsGravity(_mochiContext, GetParam(), false);
}

TEST_P(ZeroStepPreviewTest, NoHistoryConstantVelocityHasNoInertialTerm) {
  TestNoHistoryPreviewEqualsGravity(_mochiContext, GetParam(), true);
}

// ---------------------------------------------------------------------------------------------
// Zero-step (solve-free) constant-velocity force preview for multi-stage DIRK integrators.
// ---------------------------------------------------------------------------------------------

namespace {
class MultiStagePreviewTest : public MochiContextTestBase,
                              public ::testing::WithParamInterface<IntegrationMethod> {};
} // namespace

INSTANTIATE_TEST_SUITE_P(
    Integrators,
    MultiStagePreviewTest,
    // Multi-stage only: the single-stage reproduction path is covered by ZeroStepAssemblyTest.
    // SymplecticDIRK12 is excluded: despite having a single Butcher stage, its reproduction
    // behavior differs from BDF integrators — it is covered by ZeroStepPreviewTest (no-history).
    ::testing::Values(
        IntegrationMethod::DIRK22,
        IntegrationMethod::DIRK23,
        IntegrationMethod::DIRK33,
        IntegrationMethod::SymplecticDIRK22),
    [](::testing::TestParamInfo<IntegrationMethod> const& info) {
      return std::string(IntegrationMethodName(info.param));
    });

// Multi-stage DIRK integrators cannot reproduce the originating step's forces from captured state
// (intermediate stage data is not serialized), so Step(0.0) presents a constant-velocity force
// preview instead. After advancing several steps (building a nonzero velocity and a populated
// integration history) the net applied force on a free body is still gravity only. The preview must
// produce exactly one island residual of magnitude mass * |gravity|, confirming the preview path
// runs for multi-stage integrators and adds no spurious inertial term from the accumulated velocity
// or the cached DIRK history.
TEST_P(MultiStagePreviewTest, ConstantVelocityPreviewEqualsGravity) {
  IntegrationMethod const integration = GetParam();
  auto [scene, actor] =
      CreateFreeRigidScene(_mochiContext, Real3{0_r, 0_r, 0_r}, Real3{0_r, 0_r, 0_r}, integration);
  RegisterAssemblyQuery(actor);
  // Advance enough steps to build a nonzero velocity and fully populate the integration history.
  AdvanceScene(scene, 4);

  SceneForces const previewForces = StepAndReadAssembledForces(scene, 0.0);

  // A single dynamic rigid actor forms exactly one island; the multi-stage preview must produce its
  // forces.
  ASSERT_EQ(previewForces.size(), 1u) << "Step(0.0) must produce one island's forces";
  ColumnVector<real> const& res = previewForces.begin()->second;

  ExpectResidualMatchesGravity(res, scene, actor);

  _mochiContext->DestroyScene(scene);
}

static void ExpectFiniteResiduals(SceneForces const& previewForces) {
  for (auto const& entry : previewForces) {
    ColumnVector<real> const& res = entry.second;
    for (int i = 0; i < isize(res); ++i) {
      EXPECT_TRUE(IsFinite(res[i])) << "Non-finite residual at DOF " << i;
    }
  }
}

// When the most recent completed step was quasistatic (infinite dt, e.g. an IK solve), the
// multi-stage constant-velocity preview must NOT feed that infinite dt into its q0 - dt*v0 backward
// extrapolation (which would yield non-finite positions). Step(0.0) detects the unusable
// quasistatic step time and falls back to the synthetic default step, still producing a finite
// force preview of magnitude mass * |gravity|.
TEST_P(MultiStagePreviewTest, QuasistaticLastStepProducesFinitePreview) {
  IntegrationMethod const integration = GetParam();
  // A nonzero velocity is what would expose a degenerate q0 - inf*v0 extrapolation.
  Real3 const linVel{0.6_r, -0.4_r, 0.7_r};
  auto [scene, actor] =
      CreateFreeRigidScene(_mochiContext, linVel, Real3{0_r, 0_r, 0_r}, integration);
  RegisterAssemblyQuery(actor);
  auto& reg = GetRegistry(scene);

  // Simulate that the most recent completed step was quasistatic: infinite current dt and a finite
  // positive previous dt (exactly what Advance(inf) leaves behind after a prior normal step).
  reg.ctx<CSceneTime>().Reset(
      0.0, CSceneTime::kDefaultTimeStep, std::numeric_limits<double>::infinity());

  SceneForces const previewForces = StepAndReadAssembledForces(scene, 0.0);

  ASSERT_EQ(previewForces.size(), 1u) << "Step(0.0) must produce one island's forces";
  ColumnVector<real> const& res = previewForces.begin()->second;

  ExpectFiniteResiduals(previewForces);

  ExpectResidualMatchesGravity(res, scene, actor);

  _mochiContext->DestroyScene(scene);
}

namespace {
class QuasistaticHistoryFallbackTest : public MochiContextTestBase,
                                       public ::testing::WithParamInterface<IntegrationMethod> {};
} // namespace

INSTANTIATE_TEST_SUITE_P(
    Integrators,
    QuasistaticHistoryFallbackTest,
    ::testing::Values(
        IntegrationMethod::BackwardEuler,
        IntegrationMethod::BDF2,
        IntegrationMethod::BDF3),
    [](::testing::TestParamInfo<IntegrationMethod> const& info) {
      return std::string(IntegrationMethodName(info.param));
    });

// After a quasistatic step followed by a dynamic one, the previous step size is infinite, but the
// last step size is finite. Step(0.0) must reproduce the last step with its real step size, which
// differs from the default step size here.
TEST_P(QuasistaticHistoryFallbackTest, DynamicStepAfterQuasistaticStepIsReproduced) {
  IntegrationMethod const integration = GetParam();
  double constexpr kDt = 0.02; // Non-default, so that a fallback to the default step is detected
  auto [scene, actor] = CreateFreeRigidScene(
      _mochiContext, Real3{0.6_r, -0.4_r, 0.7_r}, Real3{0.5_r, 0.2_r, 0.3_r}, integration);
  RegisterAssemblyQuery(actor);
  AdvanceScene(scene, GetNumSteps(integration) + 1, kDt);
  SceneForces const referenceForces = StepAndReadAssembledForces(scene, kDt);

  // Make the step before the last one quasistatic, as after Step(inf) followed by Step(kDt).
  auto& sceneTime = GetRegistry(scene).ctx<CSceneTime>();
  sceneTime.Reset(sceneTime.StepStartTime(), std::numeric_limits<double>::infinity(), kDt);

  CheckSameForces(referenceForces, StepAndReadAssembledForces(scene, 0.0));

  _mochiContext->DestroyScene(scene);
}

static CallbackHandle
ObserveFinitePreviousRigidPose(Scene* scene, entt::entity actor, bool& previousPoseIsFinite) {
  return scene->RegisterPostStepCallback(
      "Observe reconstructed previous pose",
      [scene, actor, &previousPoseIsFinite](StepInfo const&) {
        auto& reg = GetRegistry(scene);
        Real3 const translation =
            reg.get<CRigidState<TimeStep::Previous> const>(actor).value.GetTranslation();
        previousPoseIsFinite = IsFinite(translation);
      });
}

TEST_P(QuasistaticHistoryFallbackTest, ActorCreatedAfterQuasistaticStepHasFinitePreview) {
  IntegrationMethod const integration = GetParam();
  auto [scene, existingActor] =
      CreateFreeRigidScene(_mochiContext, Real3{0_r, 0_r, 0_r}, Real3{0_r, 0_r, 0_r}, integration);
  RegisterAssemblyQuery(existingActor);
  auto& reg = GetRegistry(scene);
  AdvanceScene(scene, 4);

  reg.ctx<CSceneTime>().Reset(
      0.0, CSceneTime::kDefaultTimeStep, std::numeric_limits<double>::infinity());
  Actor* newActor = AddRigidBox(
      scene,
      {.translation = Real3{2_r, 1_r, 0_r},
       .linearVelocity = Real3{0.6_r, -0.4_r, 0.7_r},
       .angularVelocity = Real3{0.2_r, -0.1_r, 0.3_r}});
  entt::entity const newActorEntity = GetEntity(reg, newActor->GetHandle(), test::ExpectOK{});
  bool previousPoseIsFinite = false;
  CallbackHandle const callback =
      ObserveFinitePreviousRigidPose(scene, newActorEntity, previousPoseIsFinite);
  ASSERT_TRUE(callback.IsValid());

  SceneForces const previewForces = StepAndReadAssembledForces(scene, 0.0);

  ASSERT_EQ(previewForces.size(), 1u);
  EXPECT_TRUE(previousPoseIsFinite);
  ExpectFiniteResiduals(previewForces);
  Real3 const gravity = scene->GetGravity();
  real const existingForce = existingActor->GetMass(test::ExpectOK{}) * Norm(gravity);
  real const newForce = newActor->GetMass(test::ExpectOK{}) * Norm(gravity);
  real const expectedForceNorm = Sqrt(Sqr(existingForce) + Sqr(newForce));
  EXPECT_NEAR(previewForces.begin()->second.Norm(), expectedForceNorm, 1e-2_r * expectedForceNorm);

  _mochiContext->DestroyScene(scene);
}

TEST_P(QuasistaticHistoryFallbackTest, InvalidatedHistoryHasFinitePreview) {
  IntegrationMethod const integration = GetParam();
  auto [scene, actor] =
      CreateFreeRigidScene(_mochiContext, Real3{0_r, 0_r, 0_r}, Real3{0_r, 0_r, 0_r}, integration);
  RegisterAssemblyQuery(actor);
  auto& reg = GetRegistry(scene);
  AdvanceScene(scene, 4);

  reg.ctx<CSceneTime>().Reset(
      0.0, CSceneTime::kDefaultTimeStep, std::numeric_limits<double>::infinity());
  actor->SetVelocity(Real3{0.6_r, -0.4_r, 0.7_r}, Real3{0_r, 0_r, 0_r}, test::ExpectOK{});
  entt::entity const actorEntity = GetEntity(reg, actor->GetHandle(), test::ExpectOK{});
  bool previousPoseIsFinite = false;
  CallbackHandle const callback =
      ObserveFinitePreviousRigidPose(scene, actorEntity, previousPoseIsFinite);
  ASSERT_TRUE(callback.IsValid());

  SceneForces const previewForces = StepAndReadAssembledForces(scene, 0.0);

  ASSERT_EQ(previewForces.size(), 1u);
  EXPECT_TRUE(previousPoseIsFinite);
  ExpectFiniteResiduals(previewForces);
  ExpectResidualMatchesGravity(previewForces.begin()->second, scene, actor);

  _mochiContext->DestroyScene(scene);
}

// ---------------------------------------------------------------------------------------------
// Contact queries through the zero-step (solve-free) assembly.
// ---------------------------------------------------------------------------------------------

namespace {

// Flat (horizontal) static rigid ground plane at y = 0.
Actor* AddFlatGroundPlane(Scene* scene) {
  auto* context = scene->GetContext();
  RigidActorParams groundParams;
  groundParams.name = "Ground";
  groundParams.shape = context->CreatePlaneShape(Real3{0_r, 1_r, 0_r}, 0_r, ErrorAssert{});
  groundParams.isStatic = true;
  groundParams.colliderType = ColliderType::Plane;
  return scene->CreateRigidActor(groundParams, test::ExpectOK{});
}

// Registers the contact-point query (and its SurfaceNodePositions prerequisite) on an actor.
void RegisterContactPointsQuery(Actor* actor) {
  actor->RegisterQuery(QueryType::SurfaceNodePositions, test::ExpectOK{});
  actor->RegisterQuery(QueryType::ContactPoints, test::ExpectOK{});
}

// GetContactPointsWorld returns a span invalidated by the next step; copy it for later comparison.
DynamicArray<ContactPoint> CopyContactPoints(Actor* actor) {
  Span<ContactPoint const> const contacts = actor->GetContactPointsWorld(test::ExpectOK{});
  return {contacts.begin(), contacts.end()};
}

class ZeroStepContactQueryTest : public MochiContextTestBase {};

} // namespace

// Callback-authored state must outlive the zero-step transaction. Pre-step changes participate in
// query generation, while post-step changes occur after the transient simulation state is restored.
TEST_F(ZeroStepContactQueryTest, PreStepCallbackStatePersistsAndMatchesQueries) {
  auto* scene = _mochiContext->CreateScene("PreStepCallbackState");
  Real3 constexpr kStaticTranslation{100_r, 20_r, -80_r};
  real constexpr kPenetration = 1e-3_r;

  Actor* staticBox = AddRigidBox(
      scene, {.translation = kStaticTranslation, .isStatic = true, .hasGravity = false});
  Aabb const staticAabb = staticBox->GetAabbWorld(test::ExpectOK{});
  Real3 callbackTranslation = kStaticTranslation;
  callbackTranslation[0] += staticAabb.GetSize()[0] - kPenetration;
  Actor* dynamicBox =
      AddRigidBox(scene, {.translation = Real3{0_r, 0_r, 0_r}, .hasGravity = false});
  RegisterContactPointsQuery(dynamicBox);

  auto const& stepCounter = GetRegistry(scene).ctx<CSceneStepCounter const>();
  uint64_t const stepCounterBefore = stepCounter.value;
  int preStepCallbackCount = 0;
  CallbackHandle const callback =
      scene->RegisterPreStepCallback("Move box into contact", [&](StepInfo const&) {
        ++preStepCallbackCount;
        dynamicBox->SetRootTransform(TransformRT{callbackTranslation}, test::ExpectOK{});
      });
  ASSERT_TRUE(callback.IsValid());

  scene->Step(0.0);

  EXPECT_EQ(stepCounterBefore, stepCounter.value);
  EXPECT_EQ(preStepCallbackCount, 1);
  EXPECT_EQ(callbackTranslation, dynamicBox->GetRootTransform().GetTranslation());
  EXPECT_FALSE(CopyContactPoints(dynamicBox).empty());

  _mochiContext->DestroyScene(scene);
}

TEST_F(ZeroStepContactQueryTest, PostStepCallbackStatePersists) {
  auto* scene = _mochiContext->CreateScene("PostStepCallbackState");
  Actor* box = AddRigidBox(scene, {.hasGravity = false});
  Real3 constexpr kCallbackTranslation{1_r, 2_r, 3_r};

  CallbackHandle const callback =
      scene->RegisterPostStepCallback("Move box after zero step", [&](StepInfo const&) {
        box->SetRootTransform(TransformRT{kCallbackTranslation}, test::ExpectOK{});
      });
  ASSERT_TRUE(callback.IsValid());

  scene->Step(0.0);

  EXPECT_EQ(kCallbackTranslation, box->GetRootTransform().GetTranslation());

  _mochiContext->DestroyScene(scene);
}

// At initialization, Step(0.0) must update a dynamic finite collider's conservative bounds before
// broad-phase contact detection. Keeping the pair far from the origin prevents the default
// point-sized dynamic bound at the origin from overlapping the static collider by accident.
TEST_F(ZeroStepContactQueryTest, InitialFiniteCollidersFarFromOriginFindContact) {
  auto* scene = _mochiContext->CreateScene("InitialFiniteContact");
  Real3 constexpr kStaticTranslation{100_r, 20_r, -80_r};
  real constexpr kPenetration = 1e-3_r;

  Actor* staticBox = AddRigidBox(
      scene,
      {.translation = kStaticTranslation, /*isStatic*/ .isStatic = true, .hasGravity = false});
  Aabb const staticAabb = staticBox->GetAabbWorld(test::ExpectOK{});
  Real3 dynamicTranslation = kStaticTranslation;
  dynamicTranslation[0] += staticAabb.GetSize()[0] - kPenetration;
  Actor* dynamicBox = AddRigidBox(scene, {.translation = dynamicTranslation, .hasGravity = false});
  RegisterContactPointsQuery(dynamicBox);

  scene->Step(0.0);

  EXPECT_FALSE(CopyContactPoints(dynamicBox).empty());

  _mochiContext->DestroyScene(scene);
}

// An external pose edit invalidates step history and requests one-step conservative-bounds
// relaxation. Step(0.0) may use that relaxation to find contacts, but its state transaction must
// restore the request so the following real step is identical to a no-probe control.
TEST_F(ZeroStepContactQueryTest, ExternalPoseEditPreservesFollowingStep) {
  Real3 constexpr kStaticTranslation{100_r, 20_r, -80_r};
  real constexpr kPenetration = 1e-3_r;
  double constexpr kDt = 0.01;

  auto createScene = [&](char const* name) {
    Scene* scene = _mochiContext->CreateScene(name);
    Actor* staticBox = AddRigidBox(
        scene, {.translation = kStaticTranslation, .isStatic = true, .hasGravity = false});
    Aabb const staticAabb = staticBox->GetAabbWorld(test::ExpectOK{});
    Real3 overlapTranslation = kStaticTranslation;
    overlapTranslation[0] += staticAabb.GetSize()[0] - kPenetration;

    Actor* dynamicBox =
        AddRigidBox(scene, {.translation = Real3{0_r, 0_r, 0_r}, .hasGravity = false});
    RegisterContactPointsQuery(dynamicBox);
    scene->Step(kDt);
    dynamicBox->SetRootTransform(TransformRT{overlapTranslation}, test::ExpectOK{});
    return std::tuple{scene, dynamicBox};
  };

  auto [controlScene, controlBox] = createScene("ExternalEditControl");
  auto [probeScene, probeBox] = createScene("ExternalEditProbe");

  auto& controlReg = GetRegistry(controlScene);
  auto& probeReg = GetRegistry(probeScene);
  entt::entity const controlEntity =
      GetEntity(controlReg, controlBox->GetHandle(), test::ExpectOK{});
  entt::entity const probeEntity = GetEntity(probeReg, probeBox->GetHandle(), test::ExpectOK{});
  EXPECT_TRUE(controlReg.get<CConservativeStepBounds const>(controlEntity).needsNextStepRelaxation);
  EXPECT_TRUE(probeReg.get<CConservativeStepBounds const>(probeEntity).needsNextStepRelaxation);

  probeScene->Step(0.0);

  EXPECT_FALSE(CopyContactPoints(probeBox).empty());
  EXPECT_TRUE(probeReg.get<CConservativeStepBounds const>(probeEntity).needsNextStepRelaxation);

  controlScene->Step(kDt);
  probeScene->Step(kDt);
  StateHandle const finalControl = controlScene->CaptureState(test::ExpectOK{});
  StateHandle const finalProbe = probeScene->CaptureState(test::ExpectOK{});
  Span<uint8_t const> const controlBytes =
      assert_cast<SceneImpl const*>(controlScene)->FindState(finalControl, test::ExpectOK{});
  Span<uint8_t const> const probeBytes =
      assert_cast<SceneImpl const*>(probeScene)->FindState(finalProbe, test::ExpectOK{});
  EXPECT_TRUE(capture::IsEqualState(controlReg, controlBytes, probeBytes));

  Aabb const controlBounds = controlReg.get<CConservativeStepBounds const>(controlEntity).worldAabb;
  Aabb const probeBounds = probeReg.get<CConservativeStepBounds const>(probeEntity).worldAabb;
  EXPECT_EQ(controlBounds, probeBounds);

  _mochiContext->DestroyScene(controlScene);
  _mochiContext->DestroyScene(probeScene);
}

// Step(0.0) must reproduce the contact-point query of the step the simulation is currently at, so
// paused rendering/inspection sees the same contact data (positions, normals, forces, velocities)
// that a real step produced.
TEST_F(ZeroStepContactQueryTest, ZeroStepReproducesContactQuery) {
  auto* scene = _mochiContext->CreateScene("ContactRepro");
  AddFlatGroundPlane(scene);
  Actor* box = AddRigidBox(
      scene, {.translation = Real3{0_r, 0.3_r, 0_r}, .linearVelocity = Real3{0.3_r, 0_r, 0.2_r}});
  RegisterContactPointsQuery(box);

  // Settle so the box is resting in contact with the ground.
  AdvanceScene(scene, 80);

  // Capture the contact query produced by a real step.
  scene->Step(0.01);
  DynamicArray<ContactPoint> const reference = CopyContactPoints(box);
  ASSERT_FALSE(reference.empty()) << "Box must be in contact with the ground after settling";

  // Reproduce it with a solve-free zero-step assembly.
  scene->Step(0.0);
  DynamicArray<ContactPoint> const reproduced = CopyContactPoints(box);

  ASSERT_EQ(reference.size(), reproduced.size())
      << "Zero-step reproduced a different number of contacts";

  for (int i = 0; i < isize(reference); ++i) {
    ContactPoint const& a = reference[i];
    ContactPoint const& b = reproduced[i];
    EXPECT_EQ(a.distance, b.distance) << "contact " << i;
    EXPECT_EQ(a.posA, b.posA) << "contact " << i;
    EXPECT_EQ(a.normal, b.normal) << "contact " << i;
    EXPECT_EQ(a.pointVelocityA, b.pointVelocityA) << "contact " << i;
    EXPECT_EQ(a.pointVelocityB, b.pointVelocityB) << "contact " << i;
    EXPECT_EQ(a.force, b.force) << "contact " << i;
  }

  _mochiContext->DestroyScene(scene);
}

namespace {
// A dynamic box resting on a static box, both away from the origin and without a ground plane.
std::tuple<Actor*, Actor*> AddBoxOnStaticBox(Scene* scene) {
  Real3 constexpr kStaticTranslation{10_r, 2_r, -8_r};
  real constexpr kPenetration = 1e-3_r;
  Actor* staticBox = AddRigidBox(scene, {.translation = kStaticTranslation, .isStatic = true});
  Aabb const staticAabb = staticBox->GetAabbWorld(test::ExpectOK{});
  Real3 dynamicTranslation = kStaticTranslation;
  dynamicTranslation[1] += staticAabb.GetSize()[1] - kPenetration;
  Actor* dynamicBox = AddRigidBox(scene, {.translation = dynamicTranslation});
  RegisterContactPointsQuery(dynamicBox);
  return {staticBox, dynamicBox};
}

enum class StaticMotion { Pose, Velocity };

char const* StaticMotionName(StaticMotion motion) {
  switch (motion) {
    case StaticMotion::Pose:
      return "Pose";
    case StaticMotion::Velocity:
      return "Velocity";
  }
  return "Unknown";
}

using MovingStaticCase = std::tuple<StaticMotion, bool /*elastoPlasticFriction*/>;

class ZeroStepMovingStaticTest : public MochiContextTestBase,
                                 public ::testing::WithParamInterface<MovingStaticCase> {};
} // namespace

INSTANTIATE_TEST_SUITE_P(
    MotionsAndFrictionModels,
    ZeroStepMovingStaticTest,
    ::testing::Combine(
        ::testing::Values(StaticMotion::Pose, StaticMotion::Velocity),
        ::testing::Bool()),
    [](::testing::TestParamInfo<MovingStaticCase> const& info) {
      return std::string(StaticMotionName(std::get<0>(info.param))) +
          (std::get<1>(info.param) ? "_ElastoPlasticFriction" : "");
    });

// Step(0.0) reproduces a step in which a static collider moved, through prescribed poses or with a
// prescribed velocity, with either friction model.
TEST_P(ZeroStepMovingStaticTest, ZeroStepReproducesMovingStaticCollider) {
  auto const [motion, elastoPlasticFriction] = GetParam();
  auto* scene = _mochiContext->CreateScene("MovingStatic");
  SetElastoPlasticFriction(scene, elastoPlasticFriction);
  auto [staticBox, dynamicBox] = AddBoxOnStaticBox(scene);
  Real3 constexpr kStaticVelocity{0.05_r, 0_r, 0_r};
  for (int i = 0; i < 20; ++i) {
    if (motion == StaticMotion::Pose) {
      TransformRT pose = staticBox->GetRootTransform();
      pose.SetTranslation(pose.GetTranslation() + 0.01_r * kStaticVelocity);
      staticBox->SetRootTransform(pose, test::ExpectOK{});
    } else {
      staticBox->SetVelocity(kStaticVelocity, Real3{}, test::ExpectOK{});
    }
    scene->Step(0.01);
  }
  DynamicArray<ContactPoint> const reference = CopyContactPoints(dynamicBox);
  ASSERT_FALSE(reference.empty()) << "Box must rest on the static box";
  EXPECT_NEAR_TOL(reference[0].pointVelocityB, kStaticVelocity, 1e-3_r);
  EXPECT_NEAR_TOL(dynamicBox->GetLinearVelocity(test::ExpectOK{}), kStaticVelocity, 5e-3_r);

  scene->Step(0.0);
  DynamicArray<ContactPoint> const reproduced = CopyContactPoints(dynamicBox);

  ASSERT_EQ(reference.size(), reproduced.size());
  for (int i = 0; i < isize(reference); ++i) {
    EXPECT_EQ(reference[i].pointVelocityB, reproduced[i].pointVelocityB) << "contact " << i;
    EXPECT_EQ(reference[i].force, reproduced[i].force) << "contact " << i;
  }

  _mochiContext->DestroyScene(scene);
}

// Step(0.0) assembles with a pending static prescription, and leaves it pending for the next step.
TEST_F(ZeroStepContactQueryTest, ZeroStepUsesAndPreservesStaticPrescription) {
  Real3 constexpr kStaticVelocity{0.05_r, 0_r, 0.02_r};
  double constexpr kDt = 0.01;
  auto createScene = [&](char const* name) {
    Scene* scene = _mochiContext->CreateScene(name);
    auto [staticBox, dynamicBox] = AddBoxOnStaticBox(scene);
    AdvanceScene(scene, 20, kDt);
    TransformRT pose = staticBox->GetRootTransform();
    pose.SetTranslation(pose.GetTranslation() + static_cast<real>(kDt) * kStaticVelocity);
    staticBox->SetRootTransform(pose, test::ExpectOK{});
    staticBox->SetVelocity(kStaticVelocity, Real3{}, test::ExpectOK{});
    return std::tuple{scene, dynamicBox};
  };
  auto [controlScene, controlBox] = createScene("PrescriptionControl");
  auto [probeScene, probeBox] = createScene("PrescriptionProbe");

  probeScene->Step(0.0);
  DynamicArray<ContactPoint> const preview = CopyContactPoints(probeBox);
  ASSERT_FALSE(preview.empty()) << "Box must rest on the static box";
  EXPECT_NEAR_TOL(preview[0].pointVelocityB, kStaticVelocity, 1e-3_r);

  controlScene->Step(kDt);
  probeScene->Step(kDt);
  StateHandle const finalControl = controlScene->CaptureState(test::ExpectOK{});
  StateHandle const finalProbe = probeScene->CaptureState(test::ExpectOK{});
  Span<uint8_t const> const controlBytes =
      assert_cast<SceneImpl const*>(controlScene)->FindState(finalControl, test::ExpectOK{});
  Span<uint8_t const> const probeBytes =
      assert_cast<SceneImpl const*>(probeScene)->FindState(finalProbe, test::ExpectOK{});
  EXPECT_TRUE(capture::IsEqualState(GetRegistry(controlScene), controlBytes, probeBytes))
      << "Step(0.0) consumed or altered the static prescription";

  _mochiContext->DestroyScene(controlScene);
  _mochiContext->DestroyScene(probeScene);
}

namespace {
// A chain of two revolute joints with a pose controller, without gravity so that only the
// controller drives it.
Actor* AddControlledChain(Scene* scene) {
  auto* context = scene->GetContext();
  auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
  ShapeHandle const shape = context->CreateTetMeshShape(
      Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});
  ArticulatedActorParams params;
  params.joints = {
      {.type = ArticulatedJointType::Revolute, .axis = Real3{1_r, 0_r, 0_r}},
      {.type = ArticulatedJointType::Revolute,
       .parentLinkFromJoint = TransformRT{Real3{1_r, 0_r, 0_r}},
       .axis = Real3{0_r, 0_r, 1_r}}};
  params.links = {
      {.parentLink = -1, .shape = shape, .colliderType = ColliderType::None},
      {.parentLink = 0, .shape = shape, .colliderType = ColliderType::None}};
  Actor* actor = scene->CreateArticulatedActor(params, test::ExpectOK{});
  PoseControllerParams controllerParams;
  controllerParams.jointTracking.resize(
      2, PoseTrackingParams{.stiffness = 1e3_r, .damping = 1e2_r});
  actor->AddArticulatedPoseController(controllerParams, test::ExpectOK{});
  actor->RegisterQuery(QueryType::ArticulatedControllerForce, test::ExpectOK{});
  return actor;
}

DynamicArray<real> CopyControllerForce(Actor const* actor) {
  return DynamicArray<real>(actor->GetArticulatedControllerForce(test::ExpectOK{}));
}

void AddToTargetPose(Actor* actor, real delta) {
  DynamicArray<real> target(actor->GetNumDofs());
  actor->GetArticulatedTargetPose(target, test::ExpectOK{});
  for (auto& value : target) {
    value += delta;
  }
  actor->SetArticulatedTargetPose(target, test::ExpectOK{});
}
} // namespace

// Step(0.0) reproduces controller damping toward a target that moved during the reproduced step,
// directly and after a restore.
TEST_F(ZeroStepContactQueryTest, ZeroStepReproducesMovingControllerTarget) {
  double constexpr kDt = 0.01;
  auto* scene = _mochiContext->CreateScene("MovingControllerTarget");
  scene->SetGravity(Real3{});
  Actor* actor = AddControlledChain(scene);
  scene->Step(kDt); // The initial target velocity determines the old target of the first step

  AddToTargetPose(actor, 0.05_r);
  scene->Step(kDt);
  DynamicArray<real> const expected = CopyControllerForce(actor);
  ASSERT_GT(AsConstView(expected).Norm(), 1e-3_r);
  StateHandle const state = scene->CaptureState(test::ExpectOK{});

  scene->Step(0.0);
  EXPECT_SPAN_EQ(expected, CopyControllerForce(actor));

  AddToTargetPose(actor, 0.05_r);
  scene->Step(kDt);
  scene->RestoreState(state, /*releaseImmediately*/ true, test::ExpectOK{});
  scene->Step(0.0);
  EXPECT_SPAN_EQ(expected, CopyControllerForce(actor));

  _mochiContext->DestroyScene(scene);
}

// Step(0.0) applies a pending target velocity, and leaves it pending for the next step.
TEST_F(ZeroStepContactQueryTest, ZeroStepKeepsPendingControllerTargetVelocity) {
  double constexpr kDt = 0.01;
  auto createScene = [&](char const* name) {
    Scene* scene = _mochiContext->CreateScene(name);
    scene->SetGravity(Real3{});
    Actor* actor = AddControlledChain(scene);
    scene->Step(kDt);
    std::array const velocity{1_r, -2_r};
    actor->SetArticulatedTargetVelocity(velocity, test::ExpectOK{});
    return scene;
  };
  Scene* controlScene = createScene("TargetVelocityControl");
  Scene* probeScene = createScene("TargetVelocityProbe");

  probeScene->Step(0.0);

  controlScene->Step(kDt);
  probeScene->Step(kDt);
  StateHandle const finalControl = controlScene->CaptureState(test::ExpectOK{});
  StateHandle const finalProbe = probeScene->CaptureState(test::ExpectOK{});
  Span<uint8_t const> const controlBytes =
      assert_cast<SceneImpl const*>(controlScene)->FindState(finalControl, test::ExpectOK{});
  Span<uint8_t const> const probeBytes =
      assert_cast<SceneImpl const*>(probeScene)->FindState(finalProbe, test::ExpectOK{});
  EXPECT_TRUE(capture::IsEqualState(GetRegistry(controlScene), controlBytes, probeBytes))
      << "Step(0.0) consumed or altered the target velocity";

  _mochiContext->DestroyScene(controlScene);
  _mochiContext->DestroyScene(probeScene);
}

// At sim start (no history) Step(0.0) presents a constant-velocity preview. A box placed in contact
// with the ground and given a purely horizontal initial velocity must report contact points whose
// velocity equals that initial velocity (the contact point on the static ground is at rest, and the
// normal is vertical). Contact forces are NOT asserted: a solve-free sim-start assembly does not
// resolve the contact to equilibrium, so its force is not a meaningful support force.
TEST_F(ZeroStepContactQueryTest, InitialConditionsContactQueryReportsVelocities) {
  auto* scene = _mochiContext->CreateScene("ContactPreview");
  AddFlatGroundPlane(scene);

  // Rest the box on the ground with a small penetration so it is in contact at sim start. Position
  // it from its world AABB rather than assuming a centered origin (the cube mesh's local origin is
  // at its base corner), so the base penetrates the y=0 plane by kPenetration. A small penetration
  // suffices; the exact depth is irrelevant to the reported contact-point velocities.
  Actor* box = AddRigidBox(scene, {.translation = Real3{0_r, 1_r, 0_r}});
  auto const aabb = box->GetAabbWorld(test::ExpectOK{});
  real const bottomY = aabb.GetMin()[1]; // world y of the box base at the y=1 placement above
  real constexpr kPenetration = 1e-3_r;
  box->SetRootTransform(
      TransformRT{Real3{0_r, 1_r - (bottomY + kPenetration), 0_r}}, test::ExpectOK{});

  RegisterContactPointsQuery(box);

  Real3 const linVel{0.6_r, 0_r, 0.4_r}; // purely horizontal, no angular velocity
  box->SetVelocity(linVel, Real3{0_r, 0_r, 0_r}, test::ExpectOK{});

  scene->Step(0.0);

  Span<ContactPoint const> const contacts = box->GetContactPointsWorld(test::ExpectOK{});
  ASSERT_FALSE(contacts.empty()) << "Box placed in contact must report contact points at sim start";

  ActorHandle const boxHandle = box->GetHandle();
  real constexpr kVelTol = 1e-2_r;
  for (ContactPoint const& cp : contacts) {
    bool const boxIsA = boxHandle == cp.actorA;
    Real3 const boxPointVel = boxIsA ? cp.pointVelocityA : cp.pointVelocityB;
    Real3 const groundPointVel = boxIsA ? cp.pointVelocityB : cp.pointVelocityA;

    // The box translates rigidly (no angular velocity), so every contact point moves at linVel.
    EXPECT_LT(Norm(boxPointVel - linVel), kVelTol);
    // The static ground's contact point is at rest.
    EXPECT_LT(Norm(groundPointVel), kVelTol);
    // Flat ground: the contact normal is vertical (its sign depends on actorA/actorB orientation).
    EXPECT_NEAR(1_r, Norm(cp.normal), 1e-5_r);
    EXPECT_NEAR(1_r, Abs(cp.normal[1]), 1e-3_r);
  }

  _mochiContext->DestroyScene(scene);
}

// Setters don't refresh bounds; Step(0.0) does, and the refreshed bounds persist after its internal
// state restore.
TEST_F(ZeroStepContactQueryTest, ZeroStepRefreshesBoundsAfterSetNodePositions) {
  auto* scene = _mochiContext->CreateScene("Bounds");
  auto [nodes, elements] = test::CreateMinimalTetMeshUnitGrid(Real3{1_r, 1_r, 1_r}, Int3{2, 2, 2});
  SoftActorParams params;
  params.shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(nodes)), Flatten(MakeSpan(elements)), test::ExpectOK{});
  Actor* actor = scene->CreateSoftActor(params, test::ExpectOK{});
  actor->RegisterQueryAndCompute(QueryType::NodePositions, test::ExpectOK{});

  real constexpr kShiftX = 5_r; // Much larger than the unit cube, so stale bounds miss every node
  DynamicArray<real> positions(actor->GetNodePositionsLocal(test::ExpectOK{}));
  for (int i = 0; i < isize(positions); i += 3) {
    positions[i] += kShiftX;
  }
  actor->SetNodePositionsLocal(positions, test::ExpectOK{});

  scene->Step(0.0);

  Aabb const bounds = actor->GetAabbLocal(test::ExpectOK{});
  for (int i = 0; i < isize(positions); i += 3) {
    Real3 const node{positions[i], positions[i + 1], positions[i + 2]};
    for (int k = 0; k < 3; ++k) {
      EXPECT_LE(bounds.GetMin()[k], node[k]) << "node " << i / 3 << ", axis " << k;
      EXPECT_GE(bounds.GetMax()[k], node[k]) << "node " << i / 3 << ", axis " << k;
    }
  }

  _mochiContext->DestroyScene(scene);
}

namespace {
// Offsets the node positions of @p actor, which leaves its node-position query stale, and returns
// the new positions.
DynamicArray<real> OffsetNodePositions(Actor* actor) {
  DynamicArray<real> positions(actor->GetNodePositionsLocal(test::ExpectOK{}));
  for (int i = 0; i < isize(positions); i += 3) {
    positions[i] += 0.1_r;
  }
  actor->SetNodePositionsLocal(positions, test::ExpectOK{});
  EXPECT_NE(positions[0], actor->GetNodePositionsLocal(test::ExpectOK{})[0]);
  return positions;
}

void ExpectNodePositionsQuery(Actor const* actor, Span<real const> expected) {
  Span<real const> const refreshed = actor->GetNodePositionsLocal(test::ExpectOK{});
  ASSERT_EQ(isize(refreshed), isize(expected));
  for (int i = 0; i < isize(expected); ++i) {
    EXPECT_NEAR(expected[i], refreshed[i], 1e-5_r) << "DoF " << i;
  }
}
} // namespace

// Without state capture support, queries that require force evaluation can't be refreshed.
// UpdateQueries refreshes the other queries, leaves those unchanged, and reports an error.
// Step(0.0) refreshes queries the same way, logs the error as a warning, and invokes callbacks.
TEST_IF_F(
    MOCHI_ENABLE_ROM_ACTORS,
    ZeroStepContactQueryTest,
    RomSceneRefreshesOnlyQueriesWithoutForceEvaluation) {
  auto* scene = _mochiContext->CreateScene("RomUnsupported");
  auto [nodes, elements] = test::CreateMinimalTetMeshUnitGrid(Real3{1_r, 1_r, 1_r}, Int3{2, 2, 2});
  ShapeHandle shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(nodes)), Flatten(MakeSpan(elements)), test::ExpectOK{});

  SoftActorParams params;
  params.shape = shape;
  ExperimentalSoftActorParams experimentalParams;
  experimentalParams.rom = RomParams{.source = "polynomial_crom_order_1"};
  CreateSoftActor(scene, params, experimentalParams, test::ExpectOK{});

  params.worldFromLocal.SetTranslation(Real3{3_r, 0_r, 0_r});
  Actor* soft = scene->CreateSoftActor(params, test::ExpectOK{});
  soft->RegisterQueryAndCompute(QueryType::NodePositions, test::ExpectOK{});
  Actor* box = AddRigidBox(scene, {.translation = Real3{-3_r, 0_r, 0_r}, .hasGravity = false});
  RegisterContactPointsQuery(box);

  auto const& stepCounter = GetRegistry(scene).ctx<CSceneStepCounter const>();
  uint64_t const stepCounterBefore = stepCounter.value;
  int callbackCount = 0;
  auto const preStepCallback = scene->RegisterPreStepCallback(
      "ROM zero-step pre", [&](StepInfo const&) { ++callbackCount; });
  auto const postStepCallback = scene->RegisterPostStepCallback(
      "ROM zero-step post", [&](StepInfo const&) { ++callbackCount; });
  ASSERT_TRUE(preStepCallback.IsValid());
  ASSERT_TRUE(postStepCallback.IsValid());

  DynamicArray<real> const updatePositions = OffsetNodePositions(soft);
  scene->UpdateQueries(test::ExpectNotOK{});
  ExpectNodePositionsQuery(soft, updatePositions);
  (void)box->GetContactPointsWorld(test::ExpectNotOK{});
  EXPECT_EQ(callbackCount, 0);

  DynamicArray<real> const stepPositions = OffsetNodePositions(soft);
  {
    // The warning is logged only once per process, so it is suppressed rather than expected.
    auto suppressWarning = test::SuppressLogWarning();
    scene->Step(0.0);
  }
  ExpectNodePositionsQuery(soft, stepPositions);
  (void)box->GetContactPointsWorld(test::ExpectNotOK{});
  EXPECT_EQ(stepCounter.value, stepCounterBefore);
  EXPECT_EQ(callbackCount, 2);

  _mochiContext->DestroyScene(scene);
}

// A diverged step resets its island to a safe state instead of solving it, so Step(0.0) doesn't
// evaluate its forces, and its queries and state stay as the diverged step left them. Other islands
// are evaluated. Overwriting the diverged actor's state clears its divergence, so Step(0.0)
// evaluates its island again.
TEST_F(ZeroStepContactQueryTest, ZeroStepSkipsDivergedIslands) {
  test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
  auto* scene = _mochiContext->CreateScene("Diverged");
  auto& reg = GetRegistry(scene);
  Actor* converging = AddRigidBox(scene, {.hasGravity = false});
  Actor* diverging = AddRigidBox(scene, {.translation = Real3{100_r, 0_r, 0_r}});
  RegisterAssemblyQuery(converging);
  RegisterAssemblyQuery(diverging);

  // Zero tolerances and a tiny divergence threshold make any nonzero residual diverge. Only the box
  // with gravity has one.
  SolverParams params = scene->GetSolverParams();
  params.nonLinearSolver.absTol = 0_r;
  params.nonLinearSolver.relTol = 0_r;
  params.nonLinearSolver.relStepTol = 0_r;
  params.nonLinearSolver.absDivTol = 1e-30_r;
  scene->SetSolverParams(params, test::ExpectOK{});
  scene->Step(0.01);
  ASSERT_EQ(ConvergenceStatus::Converged, converging->GetConvergenceStatus());
  ASSERT_EQ(ConvergenceStatus::Diverged, diverging->GetConvergenceStatus());

  auto const isAssembled = [&](SceneForces const& forces, Actor const* actor) {
    entt::entity const e = GetEntity(reg, actor->GetHandle(), test::ExpectOK{});
    return std::ranges::any_of(forces, [e](auto const& island) {
      return std::ranges::find(island.first, e) != island.first.end();
    });
  };

  Real3 const divergedForce = diverging->GetContactForceWorld(test::ExpectOK{});
  StateHandle const stateBefore = scene->CaptureState(test::ExpectOK{});
  SceneForces const forces = StepAndReadAssembledForces(scene, 0.0);
  EXPECT_TRUE(isAssembled(forces, converging));
  EXPECT_FALSE(isAssembled(forces, diverging));
  EXPECT_EQ(divergedForce, diverging->GetContactForceWorld(test::ExpectOK{}));
  EXPECT_EQ(ConvergenceStatus::Diverged, diverging->GetConvergenceStatus());
  StateHandle const stateAfter = scene->CaptureState(test::ExpectOK{});
  auto const* sceneImpl = assert_cast<SceneImpl const*>(scene);
  EXPECT_TRUE(
      capture::IsEqualState(
          reg,
          sceneImpl->FindState(stateBefore, test::ExpectOK{}),
          sceneImpl->FindState(stateAfter, test::ExpectOK{})));

  diverging->SetVelocity(Real3{}, Real3{}, test::ExpectOK{});
  EXPECT_TRUE(isAssembled(StepAndReadAssembledForces(scene, 0.0), diverging));

  _mochiContext->DestroyScene(scene);
}

// ---------------------------------------------------------------------------------------------
// Step(0.0) must not corrupt the simulation state.
// ---------------------------------------------------------------------------------------------

namespace {
class ZeroStepDoesNotCorruptTest : public MochiContextTestBase,
                                   public ::testing::WithParamInterface<ZeroStepCase> {};
class ZeroStepSymplecticDIRK22Test : public MochiContextTestBase {};
} // namespace

INSTANTIATE_TEST_SUITE_P(
    Integrators,
    ZeroStepDoesNotCorruptTest,
    ::testing::ValuesIn(WithAndWithoutElastoPlasticFriction({
        IntegrationMethod::BackwardEuler,
        IntegrationMethod::BDF2,
        IntegrationMethod::BDF3,
        IntegrationMethod::DIRK22,
        IntegrationMethod::DIRK23,
        IntegrationMethod::DIRK33,
        IntegrationMethod::SymplecticDIRK12
        // SymplecticDIRK22 is covered below with a stable free-rigid scene.
    })),
    ZeroStepCaseName);

// Runs the same real steps in two scenes, additionally probing one with Step(0.0) before the first
// step and at the midpoint, then verifies that their final captured states are bit-equal.
static void ExpectZeroStepDoesNotAffectNextSolve(Scene* controlScene, Scene* probeScene) {
  int constexpr kTotalSteps = 6;
  int constexpr kHalfSteps = kTotalSteps / 2;
  double constexpr kDt = 0.02; // Non-default

  AdvanceScene(controlScene, kTotalSteps, kDt);
  probeScene->Step(0.0);
  AdvanceScene(probeScene, kHalfSteps, kDt);
  probeScene->Step(0.0);
  AdvanceScene(probeScene, kTotalSteps - kHalfSteps, kDt);

  StateHandle const finalControl = controlScene->CaptureState(test::ExpectOK{});
  StateHandle const finalProbe = probeScene->CaptureState(test::ExpectOK{});
  Span<uint8_t const> const controlBytes =
      assert_cast<SceneImpl const*>(controlScene)->FindState(finalControl, test::ExpectOK{});
  Span<uint8_t const> const probeBytes =
      assert_cast<SceneImpl const*>(probeScene)->FindState(finalProbe, test::ExpectOK{});
  EXPECT_TRUE(capture::IsEqualState(GetRegistry(controlScene), controlBytes, probeBytes))
      << "Step(0.0) corrupted the simulation state";
}

// Two identical scenes run the same N real steps. Scene B additionally calls Step(0.0) before the
// first step and at the midpoint. The final captured states must be bit-equal, confirming that
// Step(0.0) leaves no residual side-effects in the simulation state. Both agents register a query
// that requires assembly, so that Step(0.0) assembles.
TEST_IF_P(MOCHI_HDF5_AND_INTERNAL, ZeroStepDoesNotCorruptTest, ZeroStepDoesNotAffectNextSolve) {
  auto const [integration, elastoPlasticFriction] = GetParam();

  auto [controlScene, controlAgent] =
      CreateSceneActorTypes(_mochiContext, "ControlScene", integration);
  auto [probeScene, probeAgent] = CreateSceneActorTypes(_mochiContext, "ProbeScene", integration);
  SetElastoPlasticFriction(controlScene, elastoPlasticFriction);
  SetElastoPlasticFriction(probeScene, elastoPlasticFriction);
  RegisterAssemblyQuery(controlAgent, QueryType::ArticulatedControllerForce);
  RegisterAssemblyQuery(probeAgent, QueryType::ArticulatedControllerForce);

  ExpectZeroStepDoesNotAffectNextSolve(controlScene, probeScene);

  _mochiContext->DestroyScene(controlScene);
  _mochiContext->DestroyScene(probeScene);
}

// SymplecticDIRK22 is unstable for the multi-actor scene above in some builds. A free rigid body is
// stable and still verifies that Step(0.0) preserves the state consumed by its next solve.
TEST_F(ZeroStepSymplecticDIRK22Test, ZeroStepDoesNotAffectNextSolve) {
  Real3 constexpr kLinearVelocity{0.6_r, -0.4_r, 0.7_r};
  Real3 constexpr kAngularVelocity{0.5_r, 0.2_r, 0.3_r};
  Scene* const controlScene = std::get<0>(CreateFreeRigidScene(
      _mochiContext, kLinearVelocity, kAngularVelocity, IntegrationMethod::SymplecticDIRK22));
  Scene* const probeScene = std::get<0>(CreateFreeRigidScene(
      _mochiContext, kLinearVelocity, kAngularVelocity, IntegrationMethod::SymplecticDIRK22));

  ExpectZeroStepDoesNotAffectNextSolve(controlScene, probeScene);

  _mochiContext->DestroyScene(controlScene);
  _mochiContext->DestroyScene(probeScene);
}

// ---------------------------------------------------------------------------------------------
// UpdateQueries and Step(0.0), with and without a query that requires assembly.
// ---------------------------------------------------------------------------------------------

namespace {
enum class QueryRefreshEntryPoint { UpdateQueries, ZeroStep };

char const* QueryRefreshEntryPointName(QueryRefreshEntryPoint entryPoint) {
  switch (entryPoint) {
    case QueryRefreshEntryPoint::UpdateQueries:
      return "UpdateQueries";
    case QueryRefreshEntryPoint::ZeroStep:
      return "ZeroStep";
  }
  return "Unknown";
}

using QueryRefreshCase = std::tuple<QueryRefreshEntryPoint, bool /*withAssemblyQuery*/>;

class QueryRefreshTest : public MochiContextTestBase,
                         public ::testing::WithParamInterface<QueryRefreshCase> {};
} // namespace

INSTANTIATE_TEST_SUITE_P(
    EntryPointsAndQueries,
    QueryRefreshTest,
    ::testing::Combine(
        ::testing::Values(QueryRefreshEntryPoint::UpdateQueries, QueryRefreshEntryPoint::ZeroStep),
        ::testing::Bool()),
    [](::testing::TestParamInfo<QueryRefreshCase> const& info) {
      return std::string(QueryRefreshEntryPointName(std::get<0>(info.param))) +
          (std::get<1>(info.param) ? "_AssemblyQuery" : "_StateQueriesOnly");
    });

// Both entry points refresh queries from the current state without changing it. They assemble
// exactly when a registered query requires it. Only Step(0.0) invokes step callbacks.
TEST_P(QueryRefreshTest, RefreshesQueriesWithoutChangingState) {
  auto const [entryPoint, withAssemblyQuery] = GetParam();
  bool const isZeroStep = entryPoint == QueryRefreshEntryPoint::ZeroStep;

  auto* scene = _mochiContext->CreateScene("QueryRefresh");
  auto& reg = GetRegistry(scene);

  auto [nodes, elements] = test::CreateMinimalTetMeshUnitGrid(Real3{1_r, 1_r, 1_r}, Int3{2, 2, 2});
  SoftActorParams params;
  params.shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(nodes)), Flatten(MakeSpan(elements)), test::ExpectOK{});
  Actor* soft = scene->CreateSoftActor(params, test::ExpectOK{});
  soft->RegisterQueryAndCompute(QueryType::NodePositions, test::ExpectOK{});

  Actor* box = AddRigidBox(scene, {.translation = Real3{-3_r, 0_r, 0_r}, .hasGravity = false});
  if (withAssemblyQuery) {
    RegisterContactPointsQuery(box);
  }

  // An external state edit leaves the node-position query stale.
  DynamicArray<real> const positions = OffsetNodePositions(soft);

  int numPreStepCalls = 0;
  int numPostStepCalls = 0;
  CallbackHandle const preStepCallback =
      scene->RegisterPreStepCallback("Count pre-step", [&](StepInfo const&) { ++numPreStepCalls; });
  CallbackHandle const postStepCallback = scene->RegisterPostStepCallback(
      "Count post-step", [&](StepInfo const&) { ++numPostStepCalls; });
  ASSERT_TRUE(preStepCallback.IsValid());
  ASSERT_TRUE(postStepCallback.IsValid());

  auto const& stepCounter = reg.ctx<CSceneStepCounter const>();
  uint64_t const stepCounterBefore = stepCounter.value;
  StateHandle const stateBefore = scene->CaptureState(test::ExpectOK{});

  SceneForces const forces = RunAndReadAssembledForces(scene, [&]() {
    if (isZeroStep) {
      scene->Step(0.0);
    } else {
      scene->UpdateQueries(test::ExpectOK{});
    }
  });

  // Every island is assembled if a registered query requires it, none otherwise. The assembly makes
  // the never-computed contact query available.
  size_t const numIslands = reg.view<CIslandDescendants const>().size();
  EXPECT_EQ(forces.size(), withAssemblyQuery ? numIslands : 0u);
  if (withAssemblyQuery) {
    (void)box->GetContactPointsWorld(test::ExpectOK{});
  }

  // Queries reflect the current state.
  ExpectNodePositionsQuery(soft, positions);

  // The state and the step counter are unchanged. Only Step(0.0) invokes step callbacks.
  StateHandle const stateAfter = scene->CaptureState(test::ExpectOK{});
  auto const* sceneImpl = assert_cast<SceneImpl const*>(scene);
  EXPECT_TRUE(
      capture::IsEqualState(
          reg,
          sceneImpl->FindState(stateBefore, test::ExpectOK{}),
          sceneImpl->FindState(stateAfter, test::ExpectOK{})));
  EXPECT_EQ(stepCounterBefore, stepCounter.value);
  EXPECT_EQ(numPreStepCalls, isZeroStep ? 1 : 0);
  EXPECT_EQ(numPostStepCalls, isZeroStep ? 1 : 0);

  _mochiContext->DestroyScene(scene);
}
