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

#pragma once

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/span.h>
#include <mochi_physics/mochi_physics.h>

// mochi_physics/test is allowed to include private src headers
#include <mochi_physics/src/mochi_context.h>
#include <mochi_physics/src/mochi_ecs_utils.h>
#include <mochi_physics/src/mochi_scene.h>
#include <mochi_physics/src/mochi_shape.h>

#include <gtest/gtest.h>

#include <memory>

namespace mochi::test {

// Get access to the registry owned by a Scene
inline entt::registry& GetRegistry(Scene* scene) {
  return assert_cast<SceneImpl*>(scene)->GetRegistry();
}
inline entt::registry const& GetRegistry(Scene const* scene) {
  return assert_cast<SceneImpl const*>(scene)->GetRegistry();
}

// Create a unit-cube tetrahedral mesh shape on the given context (no skinning).
inline ShapeHandle CreateUnitCubeTetMeshShape(Context* context) {
  auto cube = CreateMinimalTetMeshUnitCube();
  return context->CreateTetMeshShape(
      Flatten(MakeSpan(cube.first)), Flatten(MakeSpan(cube.second)), ExpectOK{});
}

// Build a minimal SkinningData with 1 weight per node (a "pinned to one bone" weight scheme),
// where every node is bound to bone `boneIndex` with weight 1.
inline SkinningData MakeSingleBoneSkinning(int numNodes, int boneIndex) {
  SkinningData data;
  data.weightsPerNode = 1;
  data.weights.resize(numNodes, 1_r);
  data.indices.resize(numNodes, boneIndex);
  return data;
}

// Build a soft-body TetrahedralMeshShape with single-bone skinning and one constrained node.
// Skinning data is required for the soft-skinned mesh code path; constrained nodes are required
// so the soft-skinned engine can apply Dirichlet boundary conditions on the soft body.
inline ShapeHandle CreateUnitCubeTetSoftShape(Context* context) {
  auto&& [coords, connectivity] = CreateMinimalTetMeshUnitCube();
  auto mesh = std::make_shared<TetrahedralMesh const>(coords, connectivity);
  auto skinning = std::make_shared<SkinningData const>(
      MakeSingleBoneSkinning(mesh->GetNumNodes(), /*boneIndex=*/0));
  auto constrained = std::make_shared<ConstrainedNodesData const>(DynamicArray<int>{0});
  auto shape = std::make_shared<TetrahedralMeshShape>(mesh, skinning, constrained);
  return assert_cast<ContextImpl*>(context)->RegisterShape(shape, ExpectOK{});
}

// Pin a scene's integrator, making explicit any test that depends on a specific integrator
// (e.g. backward Euler's numerical damping) rather than on the default.
inline void SetSceneIntegrationMethod(Scene* scene, IntegrationMethod method) {
  auto solverParams = scene->GetSolverParams();
  solverParams.integrationMethod = method;
  scene->SetSolverParams(solverParams, ExpectOK{});
}

// This generic test fixture simply creates a mochi::Context instance.
class MochiContextTestBase : public ::testing::Test {
 public:
  void SetUp() override; // Called just before each test case
  void TearDown() override; // Called just after each test case

 protected:
  ~MochiContextTestBase() override;

  static constexpr real kTolerance = 1e-5_r;
  Context* _mochiContext = nullptr;

  int _numWorkerThreads = 0; // Must be set before SetUp
};

struct MochiContextTestParams {
  int numWorkerThreads = 0; // Number of worker threads to create
  bool isSingleThreadedMode = false; // Can be true even if worker threads are available
};

// This test fixture is designed to be repeated multiple times with various combinations of
// the MochiContextTestParams.
class MochiContextTestWithParam : public MochiContextTestBase,
                                  public ::testing::WithParamInterface<MochiContextTestParams> {
 public:
  using Class = MochiContextTestBase;
  void SetUp() override;
};

// This generic test fixture creates an empty mochi::Scene (synchronous API).
class MochiSceneTestBase : public MochiContextTestBase {
 public:
  void SetUp() override; // Called just before each test case
  void TearDown() override; // Called just after each test case

  entt::registry& GetRegistry() {
    return test::GetRegistry(_scene);
  }

  entt::registry const& GetRegistry() const {
    return test::GetRegistry(_scene);
  }

  entt::entity GetEntity(ActorHandle actor) const {
    return mochi::GetEntity(GetRegistry(), actor, ExpectOK{});
  }

  entt::entity GetEntity(ConstraintHandle constraint) const {
    return mochi::GetEntity(GetRegistry(), constraint, ExpectOK{});
  }

  entt::entity GetEntity(Actor* actor) const {
    return GetEntity(actor->GetHandle());
  }

  // Return the number of entities with a given component or tag
  template <class Component>
  int CountEntitiesWith() const {
    int count = 0;
    GetRegistry().view<Component const>().each([&](auto const& /*component*/) { ++count; });
    return count;
  }

 protected:
  mochi::Scene* _scene = nullptr;
};

// This generic test fixture creates an empty mochi::AsyncScene.
class MochiAsyncSceneTestBase : public MochiContextTestBase {
 public:
  MochiAsyncSceneTestBase();
  void SetUp() override; // Called just before each test case
  void TearDown() override; // Called just after each test case

 protected:
  mochi::AsyncScene* _asyncScene = nullptr;
};

// Normally logging to the Error or Warning channel will automatically fail the test.
// If you want to explicitly test logging, you can create a temporary variable of this type.
struct ExpectLoggingInScope {
  ExpectLoggingInScope(Context* mochiContext, LogChannel channel);
  ~ExpectLoggingInScope();
  Context* _mochiContext = nullptr;
  LogChannel _channel = LogChannel::Warning;
  LogFn _prevCallbackForThisModule;
  LogFn _prevCallbackForMochiPhysics;
  int _messagesReceived = 0;
};

// Apparently buck uses a different version of gtest on macos vs windows. The macos version appears
// to be newer and complains the INSTATIATE_TEST_CASE_P has been deprecated in place
// INSTATIATE_TEST_SUITE_P, which does not exist in the current Windows version.
#define MOCHI_INSTANTIATE_TEST_SUITE_P(namePrefix, fixtureClass, values)              \
  MOCHI_WARNING_PUSH();                                                               \
  MOCHI_WARNING_IGNORE_GCC_CLANG(GCC diagnostic ignored "-Wdeprecated-declarations"); \
  INSTANTIATE_TEST_CASE_P(namePrefix, fixtureClass, values);                          \
  MOCHI_WARNING_POP();

} // namespace mochi::test
