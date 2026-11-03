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

#include <mochi_core/utils/constants.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/src/mochi_capture.h>
#include <mochi_physics/src/mochi_context.h>
#include <mochi_physics/src/mochi_island.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace mochi;

// The recorded-scene assets used by these tests are not shipped externally.
#if MOCHI_USE_HDF5 && MOCHI_INTERNAL
#define MOCHI_HDF5_AND_INTERNAL 1
#else
#define MOCHI_HDF5_AND_INTERNAL 0

#endif
using namespace mochi::test;

// Representation of a scene's island partition that ignores island ordering, but preserves actor
// ordering within each island.
using IslandSignature = std::set<std::vector<entt::entity>>;

static IslandSignature GetIslandSignature(Scene* scene) {
  auto const& reg = test::GetRegistry(scene);
  IslandSignature signature;

  reg.view<CIslandMembers const>().each([&](CIslandMembers const& members) {
    signature.emplace(members.actors.begin(), members.actors.end());
  });

  return signature;
}

namespace {
using CreateSceneFunc = std::tuple<Scene*, Actor*> (*)(Context*, char const*, IntegrationMethod);

template <typename HandleT>
class RestoreType {
 public:
  using HandleType = HandleT;
  virtual ~RestoreType() = default;
  virtual void
  Initialize(CreateSceneFunc func, Context* mochiContext, IntegrationMethod integrationMethod) = 0;
  virtual void RestoreFromSource(HandleT handle, Error& error) const = 0;
  virtual bool IsEqualState(HandleT handleFrom, HandleT handleTo) const = 0;
  virtual void ReleaseAllStates() = 0;
  virtual void Cleanup(Context* mochiContext) = 0;

  Scene* sceneFrom = nullptr;
  Scene* sceneTo = nullptr;
  Actor* actorFrom = nullptr;
  Actor* actorTo = nullptr;
};

class RestoreTypeStateHandle : public RestoreType<StateHandle> {
 public:
  static bool IsValidState(StateHandle handle) {
    return handle.IsValid();
  }

  static bool IsEqualHandle(StateHandle handleFrom, StateHandle handleTo) {
    return handleFrom == handleTo;
  }

  static bool IsEqualStateSameScene(Scene const* scene, StateHandle handleA, StateHandle handleB) {
    return scene->IsEqualState(handleA, handleB);
  }

  static void
  RestoreState(Scene* scene, StateHandle handle, bool releaseImmediately, Error& error) {
    scene->RestoreState(handle, releaseImmediately, error);
  }

  static StateHandle CaptureState(Scene* scene, Error& error) {
    return scene->CaptureState(error);
  }

  static void ReleaseState(Scene* scene, StateHandle handle) {
    scene->ReleaseState(handle);
  }
};

class RestoreSame : public RestoreTypeStateHandle {
 public:
  void Initialize(CreateSceneFunc func, Context* mochiContext, IntegrationMethod integrationMethod)
      override {
    std::tie(sceneFrom, actorFrom) = func(mochiContext, "Scene", integrationMethod);
    sceneTo = sceneFrom;
    actorTo = actorFrom;
  }

  void RestoreFromSource(StateHandle handle, Error& error) const override {
    sceneFrom->RestoreState(handle, false, error);
  }

  bool IsEqualState(StateHandle handleFrom, StateHandle handleTo) const override {
    return sceneFrom->IsEqualState(handleFrom, handleTo);
  }

  void ReleaseAllStates() override {
    sceneFrom->ReleaseAllStates();
  }

  void Cleanup(Context* mochiContext) override {
    mochiContext->DestroyScene(sceneFrom);
  }
};

class RestoreDifferent : public RestoreTypeStateHandle {
 public:
  void Initialize(CreateSceneFunc func, Context* mochiContext, IntegrationMethod integrationMethod)
      override {
    std::tie(sceneFrom, actorFrom) = func(mochiContext, "SceneFrom", integrationMethod);
    std::tie(sceneTo, actorTo) = func(mochiContext, "SceneTo", integrationMethod);
  }

  void RestoreFromSource(StateHandle handle, Error& error) const override {
    experimental::RestoreStateFromScene(sceneTo, sceneFrom, handle, error);
  }

  bool IsEqualState(StateHandle handleFrom, StateHandle handleTo) const override {
    Error error;
    Span<uint8_t const> bytesFrom =
        assert_cast<SceneImpl const*>(sceneFrom)->FindState(handleFrom, error);
    Span<uint8_t const> bytesTo =
        assert_cast<SceneImpl const*>(sceneTo)->FindState(handleTo, error);
    return error.IsOK() &&
        capture::IsEqualState(
               assert_cast<SceneImpl const*>(sceneFrom)->GetRegistry(), bytesFrom, bytesTo);
  }

  void ReleaseAllStates() override {
    sceneFrom->ReleaseAllStates();
    sceneTo->ReleaseAllStates();
  }

  void Cleanup(Context* mochiContext) override {
    mochiContext->DestroyScene(sceneFrom);
    mochiContext->DestroyScene(sceneTo);
  }
};

struct ByteHandle {
  ByteHandle() {
    value = std::make_shared<DynamicArray<uint8_t>>(DynamicArray<uint8_t>{});
  }
  std::shared_ptr<DynamicArray<uint8_t>> value;
};

class RestoreBytes : public RestoreType<ByteHandle> {
 public:
  static bool IsValidState(ByteHandle handle) {
    return !handle.value->empty();
  }

  static bool IsEqualHandle(ByteHandle handleFrom, ByteHandle handleTo) {
    return handleFrom.value.get() == handleTo.value.get();
  }

  static bool IsEqualStateSameScene(Scene const* scene, ByteHandle handleA, ByteHandle handleB) {
    return capture::IsEqualState(
        assert_cast<SceneImpl const*>(scene)->GetRegistry(), *handleA.value, *handleB.value);
  }

  static void RestoreState(Scene* scene, ByteHandle handle, bool releaseImmediately, Error& error) {
    scene->RestoreStateFromBytes(*handle.value, error);
    if (releaseImmediately) {
      handle.value->clear();
    }
  }

  ByteHandle CaptureState(Scene* scene, Error& error) {
    ByteHandle result{};
    scene->CaptureStateToBytes(*result.value, error);
    _states.push_back(result);
    return result;
  }

  static void ReleaseState(Scene* /*scene*/, ByteHandle handle) {
    handle.value->clear();
  }

  void Initialize(CreateSceneFunc func, Context* mochiContext, IntegrationMethod integrationMethod)
      override {
    std::tie(sceneFrom, actorFrom) = func(mochiContext, "SceneFrom", integrationMethod);
    std::tie(sceneTo, actorTo) = func(mochiContext, "SceneTo", integrationMethod);
  }

  void RestoreFromSource(ByteHandle handle, Error& error) const override {
    sceneTo->RestoreStateFromBytes(*handle.value, error);
  }

  bool IsEqualState(ByteHandle handleFrom, ByteHandle handleTo) const override {
    return capture::IsEqualState(
        assert_cast<SceneImpl const*>(sceneFrom)->GetRegistry(),
        *handleFrom.value,
        *handleTo.value);
  }

  void ReleaseAllStates() override {
    for (auto& state : _states) {
      state.value->clear();
    }
  }

  void Cleanup(Context* mochiContext) override {
    mochiContext->DestroyScene(sceneFrom);
    mochiContext->DestroyScene(sceneTo);
    _states.clear();
  }

 protected:
  DynamicArray<ByteHandle> _states;
};

template <typename RestoreT>
concept RestoreTypeConcept =
    std::derived_from<RestoreT, RestoreType<typename RestoreT::HandleType>>;

enum class RestoreTypeEnum { Same, Different, Bytes };

struct TestParams {
  RestoreTypeEnum restore;
  IntegrationMethod integration;

  // For test naming
  friend std::ostream& operator<<(std::ostream& os, TestParams const& p) {
    char const* restoreStr = p.restore == RestoreTypeEnum::Same ? "Same"
        : p.restore == RestoreTypeEnum::Different               ? "Different"
                                                                : "Bytes";
    char const* integrationStr = p.integration == IntegrationMethod::BackwardEuler ? "BackwardEuler"
        : p.integration == IntegrationMethod::DIRK33                               ? "DIRK33"
                                                                                   : "BDF3";
    return os << restoreStr << "_" << integrationStr;
  }
};

class CaptureRestoreTest : public MochiContextTestBase,
                           public ::testing::WithParamInterface<TestParams> {};

} // namespace

INSTANTIATE_TEST_SUITE_P(
    RestoreAndIntegrationVariations,
    CaptureRestoreTest,
    ::testing::Values(
        TestParams{RestoreTypeEnum::Same, IntegrationMethod::BackwardEuler},
        TestParams{RestoreTypeEnum::Same, IntegrationMethod::DIRK33},
        TestParams{RestoreTypeEnum::Same, IntegrationMethod::BDF3},
        TestParams{RestoreTypeEnum::Different, IntegrationMethod::BackwardEuler},
        TestParams{RestoreTypeEnum::Different, IntegrationMethod::DIRK33},
        TestParams{RestoreTypeEnum::Different, IntegrationMethod::BDF3},
        TestParams{RestoreTypeEnum::Bytes, IntegrationMethod::BackwardEuler},
        TestParams{RestoreTypeEnum::Bytes, IntegrationMethod::DIRK33},
        TestParams{RestoreTypeEnum::Bytes, IntegrationMethod::BDF3}),
    [](::testing::TestParamInfo<TestParams> const& info) {
      std::ostringstream oss;
      oss << info.param;
      return oss.str();
    });

// Creates a scene with a rigid cube actor and configures the integration method.
// Returns the created scene and actor.
static std::tuple<Scene*, Actor*>
CreateCubeScene(Context* mochiContext, char const* sceneName, IntegrationMethod integrationMethod) {
  auto* scene = mochiContext->CreateScene(sceneName);

  // Configure the integration method.
  SolverParams solverParams = scene->GetSolverParams();
  solverParams.integrationMethod = integrationMethod;
  scene->SetSolverParams(solverParams, test::ExpectOK{});

  Actor* actor = AddRigidCube(scene);

  return {scene, actor};
}

// Common test logic for capture-restore-release tests, parameterized by restore type.
template <RestoreTypeConcept RestoreT>
static void TestCaptureRestoreRelease(Context* mochiContext, IntegrationMethod integrationMethod) {
  // Test with 0, ..., numSteps - 1 warm-up steps to test state capturing and restoring during cold
  // start.
  int const numSteps = GetNumSteps(integrationMethod);
  for (int warmUpSteps = 0; warmUpSteps < numSteps; ++warmUpSteps) {
    RestoreT restoreT;
    restoreT.Initialize(CreateCubeScene, mochiContext, integrationMethod);
    Scene* sceneFrom = restoreT.sceneFrom;
    Scene* sceneTo = restoreT.sceneTo;
    Actor* actorFrom = restoreT.actorFrom;
    Actor* actorTo = restoreT.actorTo;

    // Perform warm-up steps on the source scene.
    AdvanceScene(sceneFrom, warmUpSteps);

    // Initial state from source scene
    real height0 = actorFrom->GetRootTransform().GetTranslation()[1];
    auto state0 = restoreT.CaptureState(sceneFrom, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state0));

    // Advance the source scene so the actor falls some distance
    AdvanceScene(sceneFrom, 10);
    real height1 = actorFrom->GetRootTransform().GetTranslation()[1];
    auto state1 = restoreT.CaptureState(sceneFrom, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state1));
    EXPECT_FALSE(RestoreT::IsEqualHandle(state1, state0)); // Different handles
    EXPECT_FALSE(RestoreT::IsEqualStateSameScene(sceneFrom, state0, state1)); // Different state
    EXPECT_LT(height1, height0); // Actor fell in -Y direction

    // Restore state0 from sceneFrom to sceneTo
    restoreT.RestoreFromSource(state0, test::ExpectOK{});
    real height0B = actorTo->GetRootTransform().GetTranslation()[1];
    auto state0B = restoreT.CaptureState(sceneTo, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state0B));
    EXPECT_FALSE(RestoreT::IsEqualHandle(state0, state0B)); // Different handles
    EXPECT_TRUE(restoreT.IsEqualState(state0, state0B)); // Same state (full precision)
    EXPECT_EQ(height0, height0B); // Same height (full precision)

    // Try to restore an invalid state. Should have no effect on the scene.
    auto stateInvalid = typename RestoreT::HandleType{};
    RestoreT::RestoreState(sceneTo, stateInvalid, false, test::ExpectNotOK{});

    // Simulate forward the destination scene
    AdvanceScene(sceneTo, 10);
    real height1B = actorTo->GetRootTransform().GetTranslation()[1];
    auto state1B = restoreT.CaptureState(sceneTo, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state1B));
    EXPECT_FALSE(RestoreT::IsEqualHandle(state1, state1B)); // Different handles
    EXPECT_TRUE(restoreT.IsEqualState(state1, state1B)); // Same state (full precision)
    EXPECT_EQ(height1, height1B); // Same height (full precision)

    // Now release state0. Further attempts to use it should fail.
    RestoreT::ReleaseState(sceneFrom, state0);
    restoreT.RestoreFromSource(state0, test::ExpectNotOK{}); // stateFrom0 no longer valid
    RestoreT::ReleaseState(sceneFrom, state0); // Redundant call ignored
    EXPECT_EQ(
        height1,
        actorTo->GetRootTransform().GetTranslation()[1]); // No change. Still at state1.

    // We can still restore state0B. This time relase it as well.
    RestoreT::RestoreState(sceneTo, state0B, true, test::ExpectOK{});
    EXPECT_EQ(
        height0, actorTo->GetRootTransform().GetTranslation()[1]); // Back to the initial height

    // We can't restore state0B again because the call to RestoreState also released it.
    AdvanceScene(sceneTo, 10);
    RestoreT::RestoreState(sceneTo, state0B, false, test::ExpectNotOK{});
    EXPECT_EQ(height1, actorTo->GetRootTransform().GetTranslation()[1]); // state0B not restored

    // Release all remaining state handles
    restoreT.ReleaseAllStates();

    // All should now fail
    restoreT.RestoreFromSource(state0, test::ExpectNotOK{});
    RestoreT::RestoreState(sceneTo, state0B, false, test::ExpectNotOK{});
    restoreT.RestoreFromSource(state1, test::ExpectNotOK{});
    RestoreT::RestoreState(sceneTo, state1B, false, test::ExpectNotOK{});
    EXPECT_EQ(height1, actorTo->GetRootTransform().GetTranslation()[1]); // no change

    // Cleanup
    restoreT.Cleanup(mochiContext);
  }
}

TEST_P(CaptureRestoreTest, CaptureRestoreRelease) {
  auto const& params = GetParam();
  switch (params.restore) {
    case RestoreTypeEnum::Same:
      TestCaptureRestoreRelease<RestoreSame>(_mochiContext, params.integration);
      break;
    case RestoreTypeEnum::Different:
      TestCaptureRestoreRelease<RestoreDifferent>(_mochiContext, params.integration);
      break;
    case RestoreTypeEnum::Bytes:
      TestCaptureRestoreRelease<RestoreBytes>(_mochiContext, params.integration);
      break;
  }
}

static void TestWrongScene(Context* mochiContext, IntegrationMethod integrationMethod) {
  // Test with 0, ..., numSteps - 1 warm-up steps to test state capturing and restoring during cold
  // start.
  int const numSteps = GetNumSteps(integrationMethod);
  for (int warmUpSteps = 0; warmUpSteps < numSteps; ++warmUpSteps) {
    // Create two identical scenes
    Scene* sceneA = mochiContext->CreateScene("A");
    Scene* sceneB = mochiContext->CreateScene("B");

    // Configure the integration method for both scenes.
    SolverParams solverParamsA = sceneA->GetSolverParams();
    SolverParams solverParamsB = sceneB->GetSolverParams();
    solverParamsA.integrationMethod = integrationMethod;
    solverParamsB.integrationMethod = integrationMethod;
    sceneA->SetSolverParams(solverParamsA, test::ExpectOK{});
    sceneB->SetSolverParams(solverParamsB, test::ExpectOK{});

    Actor* actorA = AddRigidCube(sceneA);
    Actor* actorB = AddRigidCube(sceneB);

    // Perform warm-up steps.
    AdvanceScene(sceneA, warmUpSteps);
    AdvanceScene(sceneB, warmUpSteps);

    // Capture state from both scenes
    StateHandle stateA0 = sceneA->CaptureState(test::ExpectOK{});
    StateHandle stateB0 = sceneB->CaptureState(test::ExpectOK{});
    real heightA0 = actorA->GetRootTransform().GetTranslation()[1];
    real heightB0 = actorB->GetRootTransform().GetTranslation()[1];
    EXPECT_EQ(heightA0, heightB0);

    // Advance both scenes
    AdvanceScene(sceneA, 1);
    AdvanceScene(sceneB, 1);
    real heightA1 = actorA->GetRootTransform().GetTranslation()[1];
    real heightB1 = actorB->GetRootTransform().GetTranslation()[1];
    EXPECT_EQ(heightA1, heightB1); // Same to full precision
    EXPECT_LT(heightA1, heightA0); // Fell in -Y direction

    // Try to release handles for the wrong scenes. Should do nothing.
    sceneA->ReleaseState(stateB0);
    sceneB->ReleaseState(stateA0);

    // Try to restore state for the wrong scenes. Should fail.
    sceneA->RestoreState(stateB0, false, test::ExpectNotOK{});
    sceneB->RestoreState(stateA0, false, test::ExpectNotOK{});
    EXPECT_EQ(heightA1, actorA->GetRootTransform().GetTranslation()[1]); // no change
    EXPECT_EQ(heightB1, actorB->GetRootTransform().GetTranslation()[1]); // no change

    // Try to compare state for the wrong scenes. Should fail because of unknown handle values.
    EXPECT_FALSE(sceneA->IsEqualState(stateA0, stateB0));
    EXPECT_FALSE(sceneB->IsEqualState(stateA0, stateB0));

    // Restore state to prove we still can
    sceneA->RestoreState(stateA0, false, test::ExpectOK{});
    sceneB->RestoreState(stateB0, false, test::ExpectOK{});
    EXPECT_EQ(heightA0, actorA->GetRootTransform().GetTranslation()[1]);
    EXPECT_EQ(heightB0, actorB->GetRootTransform().GetTranslation()[1]);

    // Cleanup
    mochiContext->DestroyScene(sceneA);
    mochiContext->DestroyScene(sceneB);
  }
}

TEST_F(CaptureRestoreTest, WrongScene) {
  TestWrongScene(_mochiContext, IntegrationMethod::BackwardEuler);
}

TEST_F(CaptureRestoreTest, WrongSceneMultiStageIntegration) {
  TestWrongScene(_mochiContext, IntegrationMethod::DIRK33);
}

TEST_F(CaptureRestoreTest, WrongSceneMultiStepIntegration) {
  TestWrongScene(_mochiContext, IntegrationMethod::BDF3);
}

// Local bounds of all actors that have them, in actor iteration order.
static DynamicArray<Aabb> GetActorAabbsLocal(Scene* scene) {
  DynamicArray<Aabb> aabbs;
  scene->ForEachActor([&](Actor* actor) {
    Error error;
    Aabb const aabb = actor->GetAabbLocal(error);
    if (error.IsOK()) {
      aabbs.push_back(aabb);
    }
  });
  return aabbs;
}

// Common test logic for ActorTypes tests, parameterized by restore type.
template <RestoreTypeConcept RestoreT>
static void TestActorTypes(Context* mochiContext, IntegrationMethod integrationMethod) {
  // Test with 0, ..., numSteps - 1 warm-up steps to test state capturing and restoring during cold
  // start.
  int const numSteps = GetNumSteps(integrationMethod);
  for (int warmUpSteps = 0; warmUpSteps < numSteps; ++warmUpSteps) {
    RestoreT restoreT;
    restoreT.Initialize(CreateSceneActorTypes, mochiContext, integrationMethod);
    Scene* sceneFrom = restoreT.sceneFrom;
    Scene* sceneTo = restoreT.sceneTo;

    // Perform warm-up steps on the source scene.
    AdvanceScene(sceneFrom, warmUpSteps);

    // Advance the source scene and get the state
    AdvanceScene(sceneFrom, 2);
    auto state1 = restoreT.CaptureState(sceneFrom, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state1));
    auto const aabbs1 = GetActorAabbsLocal(sceneFrom);

    // Advance the source scene and get the state again
    AdvanceScene(sceneFrom, 2);
    auto state2 = restoreT.CaptureState(sceneFrom, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state2));
    EXPECT_FALSE(RestoreT::IsEqualHandle(state1, state2)); // Different handles

    // Confirm the state is different from state1
    EXPECT_FALSE(RestoreT::IsEqualStateSameScene(sceneFrom, state1, state2));

    // Restore state1 to sceneTo and verify
    restoreT.RestoreFromSource(state1, test::ExpectOK{});
    auto state1B = restoreT.CaptureState(sceneTo, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state1B));
    EXPECT_FALSE(RestoreT::IsEqualHandle(state1, state1B)); // Different handles
    EXPECT_TRUE(restoreT.IsEqualState(state1, state1B));

    // Bounds are recomputed on restore. The source's bounds are current in this scene, so they must
    // match.
    EXPECT_EQ(aabbs1, GetActorAabbsLocal(sceneTo));

    // Advance sceneTo and confirm the state matches state2
    AdvanceScene(sceneTo, 2);
    auto state2B = restoreT.CaptureState(sceneTo, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state2B));
    EXPECT_FALSE(RestoreT::IsEqualHandle(state2, state2B)); // Different handles
    EXPECT_TRUE(restoreT.IsEqualState(state2, state2B));

    restoreT.Cleanup(mochiContext);
  }
}

TEST_IF_P(MOCHI_HDF5_AND_INTERNAL, CaptureRestoreTest, ActorTypes) {
  auto const& params = GetParam();

// Test times out in debug builds with multi-stage/multi-step integration methods.
#if MOCHI_DEBUG
  if (params.integration != IntegrationMethod::BackwardEuler) {
    return;
  }
#endif
  // Disable warnings about solution explosion. DIRK33 is unstable for this problem, but that's OK
  // for the purpose of this test.
  bool const wasWarningEnabled = IsLogChannelEnabled(LogChannel::Warning);
  if (params.integration == IntegrationMethod::DIRK33) {
    EnableLogChannel(LogChannel::Warning, false);
  }

  switch (params.restore) {
    case RestoreTypeEnum::Same:
      TestActorTypes<RestoreSame>(_mochiContext, params.integration);
      break;
    case RestoreTypeEnum::Different:
      TestActorTypes<RestoreDifferent>(_mochiContext, params.integration);
      break;
    case RestoreTypeEnum::Bytes:
      TestActorTypes<RestoreBytes>(_mochiContext, params.integration);
      break;
  }

  if (params.integration == IntegrationMethod::DIRK33) {
    EnableLogChannel(LogChannel::Warning, wasWarningEnabled);
  }
}

// Creates a scene with actors for island preservation testing and configures the integration
// method. Returns the created scene and the rigid cube actor.
static std::tuple<Scene*, Actor*> CreateSceneIslandPreservation(
    Context* mochiContext,
    char const* sceneName,
    IntegrationMethod integrationMethod) {
  auto* scene = mochiContext->CreateScene(sceneName);

  // Configure the integration method.
  SolverParams solverParams = scene->GetSolverParams();
  solverParams.integrationMethod = integrationMethod;
  scene->SetSolverParams(solverParams, test::ExpectOK{});

  real offsetX = 0_r;
  AddGroundPlane(scene);
  Actor* rigidCube = AddRigidCube(scene, &offsetX);
  offsetX += 50_r; // Far apart so that they are on different islands.
  AddSoftDuck(scene, &offsetX);
#if MOCHI_USE_HDF5
  offsetX += 50_r;
  AddShellDuck(scene, &offsetX);
#endif

  return {scene, rigidCube};
}

// Common test logic for IslandPreservation tests, parameterized by restore type.
template <RestoreTypeConcept RestoreT>
static void TestIslandPreservation(Context* mochiContext, IntegrationMethod integrationMethod) {
  RestoreT restoreT;
  restoreT.Initialize(CreateSceneIslandPreservation, mochiContext, integrationMethod);
  Scene* sceneFrom = restoreT.sceneFrom;
  Scene* sceneTo = restoreT.sceneTo;

  // Get the initial state
  auto state0 = restoreT.CaptureState(sceneFrom, test::ExpectOK{});
  EXPECT_TRUE(RestoreT::IsValidState(state0));

  // Advance the source scene and record its island partition.
  AdvanceScene(sceneFrom, 1);
  auto const expectedSignature = GetIslandSignature(sceneFrom);
  EXPECT_GT(isize(expectedSignature), 1);

  // Deliberately make the target scene's island state different before restore. This keeps the test
  // non-vacuous without depending on conservative-bound heuristics changing the scene's island
  // count naturally over time.
  sceneTo->SetForceSingleIsland(true);
  AdvanceScene(sceneTo, 10);
  auto const staleSignature = GetIslandSignature(sceneTo);
  EXPECT_EQ(1, isize(staleSignature));
  EXPECT_NE(expectedSignature, staleSignature);
  sceneTo->SetForceSingleIsland(false);

  // Restore state0 to sceneTo and confirm that the island partition is restored.
  restoreT.RestoreFromSource(state0, test::ExpectOK{});
  AdvanceScene(sceneTo, 1);
  EXPECT_EQ(expectedSignature, GetIslandSignature(sceneTo));

  restoreT.Cleanup(mochiContext);
}

// The Duck meshes used by this test are not shipped externally.
TEST_IF_P(MOCHI_INTERNAL, CaptureRestoreTest, IslandPreservation) {
  auto const& params = GetParam();

  switch (params.restore) {
    case RestoreTypeEnum::Same:
      TestIslandPreservation<RestoreSame>(_mochiContext, params.integration);
      break;
    case RestoreTypeEnum::Different:
      TestIslandPreservation<RestoreDifferent>(_mochiContext, params.integration);
      break;
    case RestoreTypeEnum::Bytes:
      TestIslandPreservation<RestoreBytes>(_mochiContext, params.integration);
      break;
  }
}

static void TestSceneSharing(Context* mochiContext, IntegrationMethod integrationMethod) {
  // Run three simulations on a scene, and confirm that they produce the same results.
  // 1) 10 steps from scratch.
  // 2) 2 batches of 5 steps, alternating two scenes.
  // 3) 5 batches of 2 steps, alternating two scenes.

  // Test with 0, ..., numSteps - 1 warm-up steps to test state capturing and restoring during
  // cold start.
  int const numSteps = GetNumSteps(integrationMethod);
  for (int warmUpSteps = 0; warmUpSteps < numSteps; ++warmUpSteps) {
    auto* scene = mochiContext->CreateScene("Scene");

    // Configure the integration method.
    SolverParams solverParams = scene->GetSolverParams();
    solverParams.integrationMethod = integrationMethod;
    scene->SetSolverParams(solverParams, test::ExpectOK{});

    real offsetX = 0_r;
    AddGroundPlane(scene);
    AddRigidCube(scene, &offsetX);
    AddSoftDuck(scene, &offsetX);
#if MOCHI_USE_HDF5
    AddShellDuck(scene, &offsetX);
#endif

    // Perform warm-up steps.
    AdvanceScene(scene, warmUpSteps);

    // Get the initial state
    StateHandle state0 = scene->CaptureState(test::ExpectOK{});
    EXPECT_TRUE(state0.IsValid());

    // Run 1
    AdvanceScene(scene, 10);
    StateHandle state1 = scene->CaptureState(test::ExpectOK{});
    auto time1 = scene->GetTotalSimulationTime();

    // Run 2
    StateHandle stateA = state0;
    StateHandle stateB = state0;
    for (int i = 0; i < 2; ++i) {
      scene->RestoreState(stateA, false, test::ExpectOK{});
      AdvanceScene(scene, 5);
      stateA = scene->CaptureState(test::ExpectOK{});
      scene->RestoreState(stateB, false, test::ExpectOK{});
      AdvanceScene(scene, 5);
      stateB = scene->CaptureState(test::ExpectOK{});
    }
    StateHandle state2 = scene->CaptureState(test::ExpectOK{});
    auto time2 = scene->GetTotalSimulationTime();

    // Run 3
    stateA = state0;
    stateB = state0;
    for (int i = 0; i < 5; ++i) {
      scene->RestoreState(stateA, false, test::ExpectOK{});
      AdvanceScene(scene, 2);
      stateA = scene->CaptureState(test::ExpectOK{});
      scene->RestoreState(stateB, false, test::ExpectOK{});
      AdvanceScene(scene, 2);
      stateB = scene->CaptureState(test::ExpectOK{});
    }
    StateHandle state3 = scene->CaptureState(test::ExpectOK{});
    auto time3 = scene->GetTotalSimulationTime();

    // Compare
    EXPECT_TRUE(scene->IsEqualState(state2, state1));
    EXPECT_TRUE(scene->IsEqualState(state3, state1));
    EXPECT_EQ(time2, time1);
    EXPECT_EQ(time3, time1);

    mochiContext->DestroyScene(scene);
  }
}

// The Duck meshes used by TestSceneSharing are not shipped externally.
TEST_IF_F(MOCHI_INTERNAL, CaptureRestoreTest, SceneSharing) {
  TestSceneSharing(_mochiContext, IntegrationMethod::BackwardEuler);
}

TEST_IF_F(MOCHI_INTERNAL, CaptureRestoreTest, SceneSharingMultiStageIntegration) {
  TestSceneSharing(_mochiContext, IntegrationMethod::DIRK33);
}

TEST_IF_F(MOCHI_INTERNAL, CaptureRestoreTest, SceneSharingMultiStepIntegration) {
  TestSceneSharing(_mochiContext, IntegrationMethod::BDF3);
}

// Convergence weights of FOM soft and shell actors depend on gravity. Restoring a state captured
// before SetGravity must not bring back weights computed for the old gravity.
TEST_IF_F(MOCHI_INTERNAL, CaptureRestoreTest, SetGravityBetweenCaptureAndRestore) {
  Real3 const newGravity{0_r, -40_r, 0_r};
  int constexpr kStepsBefore = 2; // Weights are computed lazily during the first step.
  int constexpr kStepsAfter = 5;

  Scene* scene = _mochiContext->CreateScene("Scene");
  real offsetX = 0_r;
  AddGroundPlane(scene);
  AddSoftDuck(scene, &offsetX);
#if MOCHI_USE_HDF5
  AddShellDuck(scene, &offsetX);
#endif
  AdvanceScene(scene, kStepsBefore);

  // Reference: weights are recomputed for the new gravity.
  StateHandle const state0 = scene->CaptureState(test::ExpectOK{});
  scene->SetGravity(newGravity);
  AdvanceScene(scene, kStepsAfter);
  StateHandle const stateFresh = scene->CaptureState(test::ExpectOK{});

  // Restore a state captured with old-gravity weights.
  scene->RestoreState(state0, false, test::ExpectOK{});
  AdvanceScene(scene, kStepsAfter);
  StateHandle const stateRestored = scene->CaptureState(test::ExpectOK{});
  EXPECT_TRUE(scene->IsEqualState(stateFresh, stateRestored));

  _mochiContext->DestroyScene(scene);
}

// Creates a scene with an articulated agent with pose controller and configures the integration
// method. Returns the created scene and agent.
static std::tuple<Scene*, Actor*> CreateScenePoseController(
    Context* mochiContext,
    char const* sceneName,
    IntegrationMethod integrationMethod) {
  auto* scene = mochiContext->CreateScene(sceneName);

  // Configure the integration method.
  SolverParams solverParams = scene->GetSolverParams();
  solverParams.integrationMethod = integrationMethod;
  scene->SetSolverParams(solverParams, test::ExpectOK{});

  real offsetX = 0_r;
  auto* agent = AddNChainBody(
      mochiContext, scene, "Articulated Agent", -90_r * kRadiansPerDegree, 0_r, true, offsetX);
  AddPoseController(agent);

  return {scene, agent};
}

// Common test logic for PoseController tests, parameterized by restore type.
template <RestoreTypeConcept RestoreT>
static void TestPoseController(Context* mochiContext, IntegrationMethod integrationMethod) {
  // Test with 0, ..., numSteps - 1 warm-up steps to test state capturing and restoring during
  // cold start.
  int const numSteps = GetNumSteps(integrationMethod);
  for (int warmUpSteps = 0; warmUpSteps < numSteps; ++warmUpSteps) {
    RestoreT restoreT;
    restoreT.Initialize(CreateScenePoseController, mochiContext, integrationMethod);
    Scene* sceneFrom = restoreT.sceneFrom;
    Scene* sceneTo = restoreT.sceneTo;
    Actor* agentFrom = restoreT.actorFrom;
    Actor* agentTo = restoreT.actorTo;

    std::vector<real> target(agentFrom->GetNumDofs());

    auto applyTargetAndAdvance = [&](Scene* scene, Actor* agent, int stepBegin, int stepEnd) {
      for (int i = stepBegin; i < stepEnd; ++i) {
        std::ranges::fill(target, 0.01_r * i);
        agent->SetArticulatedTargetPose(target, test::ExpectOK{});
        AdvanceScene(scene, 1);
      }
    };

    // Perform warm-up steps on sceneFrom.
    applyTargetAndAdvance(sceneFrom, agentFrom, 0, warmUpSteps);

    // Advance sceneFrom and get the state
    applyTargetAndAdvance(sceneFrom, agentFrom, warmUpSteps, 10 + warmUpSteps);
    auto state1 = restoreT.CaptureState(sceneFrom, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state1));

    // Advance sceneFrom and get the state again
    applyTargetAndAdvance(sceneFrom, agentFrom, 10 + warmUpSteps, 20 + warmUpSteps);
    auto state2 = restoreT.CaptureState(sceneFrom, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state2));

    // Restore state1 to sceneTo and confirm it works
    restoreT.RestoreFromSource(state1, test::ExpectOK{});
    auto stateTest = restoreT.CaptureState(sceneTo, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(stateTest));
    EXPECT_FALSE(RestoreT::IsEqualHandle(state1, stateTest)); // Different handles
    EXPECT_TRUE(restoreT.IsEqualState(state1, stateTest)); // Same state data

    // Advance sceneTo and confirm the state matches state2
    applyTargetAndAdvance(sceneTo, agentTo, 10 + warmUpSteps, 20 + warmUpSteps);
    restoreT.ReleaseState(sceneTo, stateTest); // Release before overwriting
    stateTest = restoreT.CaptureState(sceneTo, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(stateTest));
    EXPECT_FALSE(RestoreT::IsEqualHandle(state2, stateTest)); // Different handles
    EXPECT_TRUE(restoreT.IsEqualState(state2, stateTest)); // Same state data

    restoreT.Cleanup(mochiContext);
  }
}

TEST_IF_P(MOCHI_USE_HDF5, CaptureRestoreTest, PoseController) {
  auto const& params = GetParam();

  switch (params.restore) {
    case RestoreTypeEnum::Same:
      TestPoseController<RestoreSame>(_mochiContext, params.integration);
      break;
    case RestoreTypeEnum::Different:
      TestPoseController<RestoreDifferent>(_mochiContext, params.integration);
      break;
    case RestoreTypeEnum::Bytes:
      TestPoseController<RestoreBytes>(_mochiContext, params.integration);
      break;
  }
}
