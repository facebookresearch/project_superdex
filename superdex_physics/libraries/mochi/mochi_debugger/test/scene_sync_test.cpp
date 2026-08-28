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

#include "mochi_debugger_test.h"

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/test/wait_until.h>
#include <mochi_core/utils/span.h>
#include <mochi_physics/dbg/protocol.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/src/mochi_context.h>
#include <mochi_physics/src/mochi_scene.h>
#include <mochi_physics/src/mochi_scene_debugger.h>
#include <mochi_physics/src/mochi_shape.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <ranges>

using namespace mochi;
using namespace mochi::dbg;

// Test fixture for tests in this file.
namespace {
class SceneSyncTest : public MochiDebuggerTest {
 protected:
  static constexpr double kTimeStep = 0.01;
  static constexpr double kTimeout = 60.0; // Generous to prevent spurious CI errors [seconds]

  int _actorCount = 0;

  void SetUp() override {
    MochiDebuggerTest::SetUp(); // Call down
    _client->SetSceneStepMode(StepMode::Play);
  }

  ShapeHandle CreateUnitCube() const {
    auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
    return _context->CreateTetMeshShape(
        Flatten(MakeConstSpan(coordinates)),
        Flatten(MakeConstSpan(connectivity)),
        test::ExpectOK{});
  }

  // Create a scene. Disable gravity so we don't have to worry about dynamics.
  Scene* CreateSceneNoGravity(std::string_view name = "SceneSyncTest") const {
    Scene* scene = _context->CreateScene(name);
    scene->SetGravity({});
    return scene;
  }

  ShapeHandle CreateShapeWithVisualMesh() const {
    DynamicArray<real> const coordinates = {
        0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 1_r, 1_r, 0_r, 0_r, 1_r, 0_r};
    DynamicArray<int> const surfaceConnectivity = {0, 1, 2, 0, 2, 3};
    DynamicArray<int> const visualConnectivity = {0, 1, 3, 1, 2, 3};

    ModelData model;
    model.mesh.emplace();
    model.mesh->nodesPerElement = 3;
    model.mesh->coordinates = coordinates;
    model.mesh->connectivity = surfaceConnectivity;
    model.visualMesh.emplace();
    model.visualMesh->nodesPerElement = 3;
    model.visualMesh->coordinates = coordinates;
    model.visualMesh->connectivity = visualConnectivity;
    model.visualMesh->skinning.emplace();
    model.visualMesh->skinning->weightsPerNode = 1;
    model.visualMesh->skinning->indices = {0, 1, 2, 3};
    model.visualMesh->skinning->weights = {1_r, 1_r, 1_r, 1_r};
    return _context->CreateModelShape(model, test::ExpectOK{});
  }

  ActorHandle CreateRigidActorWithVisualMesh(Scene* scene) {
    RigidActorParams params;
    params.name = "RigidVisualMesh";
    params.shape = CreateShapeWithVisualMesh();
    params.colliderType = ColliderType::None;
    params.isStatic = true;
    return scene->CreateRigidActor(params, test::ExpectOK{})->GetHandle();
  }

  ActorHandle CreateActorWithVisualMeshOnly(Scene* scene) {
    ModelData model;
    model.mesh.emplace();
    model.mesh->nodesPerElement = 2;
    model.mesh->coordinates = {0_r, 0_r, 0_r, 1_r, 0_r, 0_r};
    model.elementFrameAxes = {0_r, 1_r, 0_r};
    model.visualMesh.emplace();
    model.visualMesh->nodesPerElement = 3;
    model.visualMesh->coordinates = {0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0_r, 1_r, 0_r};
    model.visualMesh->connectivity = {0, 1, 2};
    model.visualMesh->skinning.emplace();
    model.visualMesh->skinning->weightsPerNode = 1;
    model.visualMesh->skinning->indices = {0, 0, 0};
    model.visualMesh->skinning->weights = {1_r, 1_r, 1_r};

    experimental::RodActorParams params;
    params.name = "VisualMeshOnly";
    params.shape = _context->CreateModelShape(model, test::ExpectOK{});
    params.material.linearDensity = 1_r;
    params.material.linearRotationalInertia = 1_r;
    params.material.axialStiffness = 1e3_r;
    params.material.torsionalStiffness = 1e1_r;
    params.material.flexuralStiffness = {1e1_r, 1e1_r};
    return experimental::CreateRodActor(scene, params, test::ExpectOK{})->GetHandle();
  }

  ActorHandle CreateShellActorWithVisualMesh(Scene* scene) {
    experimental::ShellActorParams params;
    params.name = "ShellVisualMesh";
    params.shape = CreateShapeWithVisualMesh();
    params.material.density = 1_r;
    params.colliderType = ColliderType::None;
    return experimental::CreateShellActor(scene, params, test::ExpectOK{})->GetHandle();
  }

  // Create a rigid actor with properties that depend on _actorCount so they are all different.
  ActorHandle CreateRigidActor(Scene* scene, bool isStatic = false) {
    RigidActorParams params;
    params.name = Format("Rigid%2d", _actorCount);
    params.shape = CreateUnitCube();
    params.worldFromLocal.SetRotation(Quaternion::RotationX(0.1_r * _actorCount));
    params.worldFromLocal.SetTranslation(Real3{2.0_r * _actorCount, 0_r, 0_r});
    params.isStatic = isStatic;
    ++_actorCount;
    return scene->CreateRigidActor(params, test::ExpectOK{})->GetHandle();
  }

  // Create a soft actor with properties that depend on _actorCount so they are all different.
  ActorHandle CreateSoftActor(Scene* scene) {
    SoftActorParams params;
    params.name = Format("Soft%2d", _actorCount);
    params.shape = CreateUnitCube();
    params.worldFromLocal.SetRotation(Quaternion::RotationX(0.1_r * _actorCount));
    params.worldFromLocal.SetTranslation(Real3{2.0_r * _actorCount, 0_r, 0_r});
    ++_actorCount;
    return scene->CreateSoftActor(params, test::ExpectOK{})->GetHandle();
  }

  // Create a 2-link articulated actor. Its nested link actors report this actor as their parent.
  ActorHandle CreateArticulatedActor(Scene* scene) {
    ArticulatedActorParams params;
    params.name = Format("Articulated%2d", _actorCount);
    params.worldFromRoot.SetTranslation(Real3{2.0_r * _actorCount, 0_r, 0_r});
    params.joints = {
        {.type = ArticulatedJointType::Free}, {.type = ArticulatedJointType::Spherical}};
    params.links = {
        {.parentLink = -1, .shape = CreateUnitCube(), .colliderType = ColliderType::Box},
        {.parentLink = 0, .shape = CreateUnitCube(), .colliderType = ColliderType::Box}};
    ++_actorCount;
    return scene->CreateArticulatedActor(params, test::ExpectOK{})->GetHandle();
  }

  // Create a one-link articulated actor with one nested soft actor. Its nested soft actor reports
  // this actor as its parent. The soft shape carries skinning data, which binds it to the skeleton,
  // and a constrained node, which the soft-skinned engine needs for its boundary conditions.
  ActorHandle CreateSoftSkinnedActor(Scene* scene) {
    auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
    auto mesh = std::make_shared<TetrahedralMesh const>(coordinates, connectivity);
    SkinningData skinning;
    skinning.weightsPerNode = 1;
    skinning.weights.resize(mesh->GetNumNodes(), 1_r);
    skinning.indices.resize(mesh->GetNumNodes(), 0);
    auto softShape = std::make_shared<TetrahedralMeshShape>(
        mesh,
        std::make_shared<SkinningData const>(std::move(skinning)),
        std::make_shared<ConstrainedNodesData const>(DynamicArray<int>{0}));

    SoftActorParams softParams;
    softParams.name = Format("SoftSkin%2d", _actorCount);
    softParams.shape =
        assert_cast<ContextImpl*>(_context)->RegisterShape(softShape, test::ExpectOK{});
    softParams.hasGravity = false;

    SoftSkinnedActorParams params;
    params.skeletonParams.name = Format("SoftSkinned%2d", _actorCount);
    params.skeletonParams.worldFromRoot.SetTranslation(Real3{2.0_r * _actorCount, 0_r, 0_r});
    params.skeletonParams.joints = {{.type = ArticulatedJointType::Free}};
    params.skeletonParams.links = {
        {.parentLink = -1, .shape = CreateUnitCube(), .colliderType = ColliderType::None}};
    params.softParams.push_back(std::move(softParams));
    ++_actorCount;
    return scene->CreateSoftSkinnedActor(params, test::ExpectOK{})->GetHandle();
  }

  // Expect that the synced data matches the actor in the scene
  void ExpectActorData(Scene* scene, ActorHandle actorHandle, protocol::ActorSyncData const& data) {
    Actor* actor = scene->GetActor(actorHandle);
    ASSERT_NE(nullptr, actor);
    EXPECT_EQ(actorHandle, data.handle);
    EXPECT_EQ(actor->GetType(), data.type);
    EXPECT_STREQ(actor->GetName(), data.name.c_str());
    auto const rt = actor->GetRootTransform();
    EXPECT_EQ(StaticCast<Float3>(rt.GetTranslation()), data.position);
    EXPECT_EQ(StaticCast<Float4>(rt.GetRotation().ToReal4()), data.rotation);
    EXPECT_EQ(actor->IsStatic(), data.isStatic);

    // Parent: nested actors report their parent articulated actor; standalone actors report an
    // invalid handle.
    if (actor->IsNestedLinkActor() || actor->IsNestedSoftActor()) {
      EXPECT_EQ(actor->GetArticulatedActor(test::ExpectOK{}), data.parent);
    } else {
      EXPECT_FALSE(data.parent.IsValid());
    }
  }

  template <typename Fn>
  void WithSyncedMesh(ActorHandle actorHandle, Fn&& fn) const {
    _client->GetSceneSyncData([&](SceneSyncData const& data) {
      ASSERT_EQ(isize(data.actors), isize(data.actorMeshes));
      auto const it = std::ranges::find(data.actors, actorHandle, &protocol::ActorSyncData::handle);
      ASSERT_NE(data.actors.end(), it);
      fn(data.actorMeshes[static_cast<int>(it - data.actors.begin())]);
    });
  }

  // Expect that an actor's synchronized mesh matches an explicitly selected source mesh. Returns
  // the synchronized connectivity revision.
  uint64_t ExpectActorMeshMatches(ActorHandle actorHandle, MeshDataView const& expectedMesh) const {
    uint64_t revision = 0;
    WithSyncedMesh(actorHandle, [&](MeshSyncDataView const& synced) {
      // Connectivity is retained across syncs (sent once), so it must always match exactly.
      EXPECT_EQ(expectedMesh.connectivity, synced.connectivity);

      // Coordinates ride the wire as float, so compare at float precision (works in double builds).
      ASSERT_EQ(isize(expectedMesh.coordinates), isize(synced.coordinates));
      for (int i = 0; i < isize(synced.coordinates); ++i) {
        EXPECT_EQ(static_cast<float>(expectedMesh.coordinates[i]), synced.coordinates[i]);
      }
      revision = synced.versionCounter;
    });
    return revision;
  }

  uint64_t ExpectActorMeshMatches(Scene* scene, ActorHandle actorHandle) const {
    Actor* actor = scene->GetActor(actorHandle);
    EXPECT_NE(nullptr, actor);
    return actor ? ExpectActorMeshMatches(actorHandle, actor->GetSurfaceMesh()) : 0;
  }

  uint64_t ExpectActorMeshTopologyMatches(ActorHandle actorHandle, MeshDataView const& expectedMesh)
      const {
    uint64_t revision = 0;
    WithSyncedMesh(actorHandle, [&](MeshSyncDataView const& synced) {
      EXPECT_EQ(expectedMesh.connectivity, synced.connectivity);
      EXPECT_EQ(isize(expectedMesh.coordinates), isize(synced.coordinates));
      revision = synced.versionCounter;
    });
    return revision;
  }

  void ExpectActorMeshEmpty(ActorHandle actorHandle) const {
    WithSyncedMesh(actorHandle, [](MeshSyncDataView const& synced) {
      EXPECT_TRUE(synced.coordinates.empty());
      EXPECT_TRUE(synced.connectivity.empty());
      EXPECT_EQ(0, synced.versionCounter);
    });
  }

  // Return true if the DebugClient's cached scene list contains the given handle.
  bool ClientHasScene(SceneHandle handle) const {
    DynamicArray<SceneInfo> scenes;
    _client->GetSceneList(scenes);
    for (auto const& s : scenes) {
      if (s.handle == handle) {
        return true;
      }
    }
    return false;
  }

  // Return a copy of the current SceneSyncData (normally accessed via callback)
  SceneSyncData GetSceneSyncData() const {
    SceneSyncData outData;
    _client->GetSceneSyncData([&](auto const& data) { outData = data; });
    return outData;
  }

  // Pump the scene until the client has received new sync data, then return it.
  SceneSyncData WaitForSync(Scene* scene, uint64_t expectedCounter) const {
    SceneSyncData data;
    test::WaitUntil([&] {
      scene->UpdateDebugger();
      data = GetSceneSyncData();
      return data.counter >= expectedCounter;
    });
    EXPECT_EQ(data.counter, expectedCounter);
    return data;
  }

  // Expect that no new sync data arrives within a short period of time
  void ExpectNoMoreSync(Scene* scene, uint64_t expectedCounter) const {
    test::WaitUntil(
        [&] {
          scene->UpdateDebugger();
          return GetSceneSyncData().counter != expectedCounter;
        },
        /*timeout*/ 0.01f,
        /*failOnTimeout*/ false);
    EXPECT_EQ(GetSceneSyncData().counter, expectedCounter);
  }

  // Sync parameters that enable automatic synchronization of actor data.
  static SceneSyncParams ActorSyncParams() {
    SceneSyncParams params;
    params.enabled = true;
    params.syncActors = true;
    return params;
  }

  // Sync parameters that enable automatic synchronization of actor data and surface meshes.
  static SceneSyncParams MeshSyncParams() {
    SceneSyncParams params = ActorSyncParams();
    params.syncMeshes = true;
    return params;
  }

  void SetSceneSyncParams(SceneSyncParams const& params) {
    auto settings = _client->GetSettings();
    settings.sync = params;
    _client->SetSettings(settings);
  }

  static void EnableDebugDrawFeature(Scene* scene, std::string_view name, bool enable) {
    ASSERT_NE(nullptr, scene);
    auto& debugDraw = scene->GetDebugDraw();
    int index = debugDraw.FindFeature(name);
    ASSERT_LE(0, index);
    debugDraw.EnableFeature(index, enable);
    EXPECT_EQ(enable, debugDraw.IsFeatureEnabled(index));
  }

  // Enable debug draw features that depend on the positions of actors.
  static void EnableActorDebugDraw(Scene* scene) {
    scene->GetDebugDraw().Enable(true);
    EnableDebugDrawFeature(scene, "Actor Mesh", true); // Draws lines
    EnableDebugDrawFeature(scene, "Actor Contact Samples", true); // Draws spheres
  }
};
} // namespace

TEST_F(SceneSyncTest, SelectScene) {
  Scene* scene = CreateSceneNoGravity();
  SceneHandle sceneHandle = scene->GetHandle();
  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(sceneHandle); });

  test::WaitUntil([&] { return _client->GetSelectedScene() == sceneHandle; });

  // An invalid handle clears the selection.
  _client->SelectScene(SceneHandle{});
  EXPECT_EQ(SceneHandle{}, _client->GetSelectedScene());
}

TEST_F(SceneSyncTest, SelectionUpdatesSyncDataScene) {
  Scene* sceneA = CreateSceneNoGravity("A");
  Scene* sceneB = CreateSceneNoGravity("B");
  SceneHandle handleA = sceneA->GetHandle();
  SceneHandle handleB = sceneB->GetHandle();

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(handleA) && ClientHasScene(handleB); });

  test::WaitUntil([&] { return _client->GetSelectedScene() == handleB; });
  auto syncData = GetSceneSyncData();
  uint64_t const baseCounter = syncData.counter;
  EXPECT_EQ(uint64_t{1}, baseCounter); // Welcome-time auto-selection is the only increment so far.
  EXPECT_EQ(handleB, syncData.scene);

  // Enable syncing so selection drives synchronization.
  SetSceneSyncParams(ActorSyncParams());
  syncData = GetSceneSyncData();
  EXPECT_EQ(baseCounter, syncData.counter); // no change
  EXPECT_EQ(handleB, syncData.scene); // no change

  // SceneSyncData::scene reflects selection changes immediately.
  _client->SelectScene(handleA);
  syncData = GetSceneSyncData();
  EXPECT_EQ(baseCounter + 1, syncData.counter);
  EXPECT_EQ(handleA, syncData.scene);

  // Switching selection updates it immediately as well.
  _client->SelectScene(handleB);
  syncData = GetSceneSyncData();
  EXPECT_EQ(baseCounter + 2, syncData.counter);
  EXPECT_EQ(handleB, syncData.scene);

  // Deselecting clears it.
  _client->SelectScene({});
  syncData = GetSceneSyncData();
  EXPECT_EQ(baseCounter + 3, syncData.counter);
  EXPECT_EQ(SceneHandle{}, syncData.scene);
}

TEST_F(SceneSyncTest, EnableSyncThenSelectScene) {
  Scene* scene = CreateSceneNoGravity();
  SceneHandle sceneHandle = scene->GetHandle();
  ActorHandle actor = CreateRigidActor(scene);

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(sceneHandle); });

  _client->SelectScene({});
  uint64_t const noSelectionCounter = GetSceneSyncData().counter;
  EXPECT_EQ(uint64_t{2}, noSelectionCounter); // Auto-selection at welcome, then clearing it.

  // Enable actor syncing
  SetSceneSyncParams(ActorSyncParams());

  // Step the scene a few times. No data should be flowing because we haven't selected a scene.
  for (int i = 0; i < 10; ++i) {
    scene->Step(kTimeStep);
  }
  ExpectNoMoreSync(scene, noSelectionCounter);

  // Now select the scene. This should enable syncing.
  _client->SelectScene(sceneHandle);
  EXPECT_EQ(sceneHandle, _client->GetSelectedScene());
  uint64_t const baseCounter = GetSceneSyncData().counter;

  // Expect data for one actor
  auto syncData = WaitForSync(scene, baseCounter + 1);
  ASSERT_EQ(1, isize(syncData.actors));
  ExpectActorData(scene, actor, syncData.actors[0]);
  ExpectNoMoreSync(scene, baseCounter + 1); // No more data till we step the scene

  // Now step the scene on the server
  scene->Step(kTimeStep);
  syncData = WaitForSync(scene, baseCounter + 2);
  ASSERT_EQ(1, isize(syncData.actors));
  ExpectActorData(scene, actor, syncData.actors[0]);

  // Now deselect the scene. SyncData should be cleared and counter incremented immediately.
  _client->SelectScene({});
  syncData = GetSceneSyncData();
  EXPECT_EQ(SceneHandle{}, syncData.scene);
  EXPECT_EQ(baseCounter + 3, syncData.counter);
  EXPECT_EQ(0, isize(syncData.actors));
}

TEST_F(SceneSyncTest, ReselectSceneResendsMeshes) {
  Scene* scene = CreateSceneNoGravity();
  SceneHandle const sceneHandle = scene->GetHandle();
  ActorHandle const rigidActor = CreateRigidActor(scene);
  ActorHandle const softActor = CreateSoftActor(scene);

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(sceneHandle); });

  SetSceneSyncParams(MeshSyncParams());
  _client->SelectScene(sceneHandle);
  uint64_t const baseCounter = GetSceneSyncData().counter;
  WaitForSync(scene, baseCounter + 1);
  uint64_t const rigidRevision = ExpectActorMeshMatches(scene, rigidActor);
  uint64_t const softRevision = ExpectActorMeshMatches(scene, softActor);

  // Deselecting clears the client mesh cache and tells the server to reset its mesh sync state.
  _client->SelectScene({});
  EXPECT_TRUE(GetSceneSyncData().actorMeshes.empty());

  // Reselecting must provide complete meshes rather than relying on the discarded client cache.
  // Deselect, reselect, and the new reply each increment the client sync counter.
  _client->SelectScene(sceneHandle);
  WaitForSync(scene, baseCounter + 4);
  EXPECT_GT(ExpectActorMeshMatches(scene, rigidActor), rigidRevision);
  EXPECT_GT(ExpectActorMeshMatches(scene, softActor), softRevision);
}

TEST_F(SceneSyncTest, ReenableSyncResendsMeshes) {
  Scene* scene = CreateSceneNoGravity();
  SceneHandle const sceneHandle = scene->GetHandle();
  ActorHandle const rigidActor = CreateRigidActor(scene);
  ActorHandle const softActor = CreateSoftActor(scene);

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(sceneHandle); });

  SceneSyncParams params = MeshSyncParams();
  params.useVisualMesh = false;
  SetSceneSyncParams(params);
  _client->SelectScene(sceneHandle);
  uint64_t const baseCounter = GetSceneSyncData().counter;
  WaitForSync(scene, baseCounter + 1);
  uint64_t const rigidRevision = ExpectActorMeshMatches(scene, rigidActor);
  uint64_t const softRevision = ExpectActorMeshMatches(scene, softActor);
  Actor* const soft = scene->GetActor(softActor);
  ASSERT_NE(nullptr, soft);
  EXPECT_FALSE(soft->GetSurfaceMeshNodePositionsLocal(test::ExpectOK{}).empty());

  // Disabling must cancel the debugger-owned deformable mesh query.
  params.enabled = false;
  SetSceneSyncParams(params);
  test::WaitUntil([&] {
    scene->UpdateDebugger();
    Error error;
    auto const positions = soft->GetSurfaceMeshNodePositionsLocal(error);
    return !error.IsOK() && positions.empty();
  });

  // Re-enabling must start a fresh mesh epoch with complete rigid and deformable meshes.
  uint64_t const disabledCounter = GetSceneSyncData().counter;
  params.enabled = true;
  SetSceneSyncParams(params);
  WaitForSync(scene, disabledCounter + 1);
  EXPECT_GT(ExpectActorMeshMatches(scene, rigidActor), rigidRevision);
  EXPECT_GT(ExpectActorMeshMatches(scene, softActor), softRevision);
}

TEST_F(SceneSyncTest, DisablingMeshCategoryCancelsMeshQuery) {
  Scene* scene = CreateSceneNoGravity();
  SceneHandle const sceneHandle = scene->GetHandle();
  ActorHandle const softActor = CreateSoftActor(scene);

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(sceneHandle); });

  SceneSyncParams params = MeshSyncParams();
  params.useVisualMesh = false;
  SetSceneSyncParams(params);
  _client->SelectScene(sceneHandle);
  uint64_t const baseCounter = GetSceneSyncData().counter;
  WaitForSync(scene, baseCounter + 1);

  Actor* const soft = scene->GetActor(softActor);
  ASSERT_NE(nullptr, soft);
  EXPECT_FALSE(soft->GetSurfaceMeshNodePositionsLocal(test::ExpectOK{}).empty());

  params.syncMeshes = false;
  SetSceneSyncParams(params);
  test::WaitUntil([&] {
    scene->UpdateDebugger();
    Error error;
    auto const positions = soft->GetSurfaceMeshNodePositionsLocal(error);
    return !error.IsOK() && positions.empty();
  });
}

TEST_F(SceneSyncTest, SelectSceneThenEnableSync) {
  Scene* scene = CreateSceneNoGravity();
  SceneHandle sceneHandle = scene->GetHandle();
  ActorHandle actor = CreateRigidActor(scene);

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(sceneHandle); });

  // Select the scene.
  _client->SelectScene(sceneHandle);
  EXPECT_EQ(sceneHandle, _client->GetSelectedScene());
  uint64_t const baseCounter = GetSceneSyncData().counter;

  // Step the scene a few times. No data should be flowing because we haven't enabled syncing.
  for (int i = 0; i < 10; ++i) {
    scene->Step(kTimeStep);
  }
  ExpectNoMoreSync(scene, baseCounter);

  // Enable actor syncing
  SetSceneSyncParams(ActorSyncParams());

  // Expect data for one actor
  auto syncData = WaitForSync(scene, baseCounter + 1);
  ASSERT_EQ(1, isize(syncData.actors));
  ExpectActorData(scene, actor, syncData.actors[0]);
  ExpectNoMoreSync(scene, baseCounter + 1); // No more data till we step the scene

  // Now step the scene on the server
  scene->Step(kTimeStep);
  syncData = WaitForSync(scene, baseCounter + 2);
  ASSERT_EQ(1, isize(syncData.actors));
  ExpectActorData(scene, actor, syncData.actors[0]);

  // Now disable syncing. This should immediately clear data an increment the counter.
  SetSceneSyncParams(SceneSyncParams{.enabled = false});
  syncData = GetSceneSyncData();
  EXPECT_EQ(sceneHandle, syncData.scene);
  EXPECT_EQ(baseCounter + 3, syncData.counter);
  EXPECT_EQ(0, isize(syncData.actors));
}

TEST_F(SceneSyncTest, SyncMultipleActors) {
  Scene* scene = CreateSceneNoGravity();
  SceneHandle sceneHandle = scene->GetHandle();

  // A mix of rigid and soft actors, each with distinct name/position/rotation.
  CreateRigidActor(scene);
  CreateSoftActor(scene);
  CreateRigidActor(scene, /*isStatic*/ true);
  CreateSoftActor(scene);

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(sceneHandle); });

  SetSceneSyncParams(ActorSyncParams());
  _client->SelectScene(sceneHandle);
  uint64_t const baseCounter = GetSceneSyncData().counter;

  auto syncData = WaitForSync(scene, baseCounter + 1);
  ASSERT_EQ(4, isize(syncData.actors));

  // Sort by name for a deterministic order, then verify each entry against the scene.
  std::ranges::sort(syncData.actors, [](auto const& a, auto const& b) { return a.name < b.name; });
  for (auto const& actorData : syncData.actors) {
    ExpectActorData(scene, actorData.handle, actorData);
  }
}

TEST_F(SceneSyncTest, SyncReportsNestedActorParents) {
  Scene* scene = CreateSceneNoGravity();
  SceneHandle sceneHandle = scene->GetHandle();

  ActorHandle articulated = CreateArticulatedActor(scene);
  ActorHandle softSkinned = CreateSoftSkinnedActor(scene);
  CreateRigidActor(scene); // A standalone actor to confirm it reports no parent.

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(sceneHandle); });

  SetSceneSyncParams(ActorSyncParams());
  _client->SelectScene(sceneHandle);
  uint64_t const baseCounter = GetSceneSyncData().counter;

  auto syncData = WaitForSync(scene, baseCounter + 1);

  // Verify every synced actor against the scene. ExpectActorData checks that nested actors report
  // their articulated parent and that standalone actors report an invalid handle.
  for (auto const& actorData : syncData.actors) {
    ExpectActorData(scene, actorData.handle, actorData);
  }

  // Every nested actor references its articulated actor as its parent. Links and skinned soft
  // bodies are reported through separate server-side branches, so check both.
  auto const expectParents = [&](ActorHandle parent, Span<ActorHandle const> nested) {
    ASSERT_FALSE(nested.empty());
    for (ActorHandle const child : nested) {
      auto* const it = std::ranges::find(syncData.actors, child, &protocol::ActorSyncData::handle);
      ASSERT_NE(syncData.actors.end(), it);
      EXPECT_EQ(parent, it->parent);
    }
  };

  Actor* articulatedActor = scene->GetActor(articulated);
  ASSERT_NE(nullptr, articulatedActor);
  expectParents(articulated, articulatedActor->GetNestedLinkActors(test::ExpectOK{}));

  Actor* softSkinnedActor = scene->GetActor(softSkinned);
  ASSERT_NE(nullptr, softSkinnedActor);
  expectParents(softSkinned, softSkinnedActor->GetNestedSoftActors(test::ExpectOK{}));
}

TEST_F(SceneSyncTest, SyncUpdatesWhenActorMoves) {
  Scene* scene = CreateSceneNoGravity();
  SceneHandle sceneHandle = scene->GetHandle();
  ActorHandle actor = CreateRigidActor(scene);

  // Enable debug-draw features whose geometry depends on the actor's pose.
  EnableActorDebugDraw(scene);

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(sceneHandle); });

  // Sync both actors and debug draw.
  SceneSyncParams params;
  params.enabled = true;
  params.syncActors = true;
  params.syncDebugDraw = true;
  SetSceneSyncParams(params);
  _client->SelectScene(sceneHandle);
  uint64_t const baseCounter = GetSceneSyncData().counter;

  // The immediate reply has actor data but no debug draw yet (nothing stepped). Stepping generates
  // debug-draw geometry, which arrives with the next sync.
  WaitForSync(scene, baseCounter + 1);
  scene->Step(kTimeStep);
  auto before = WaitForSync(scene, baseCounter + 2);
  ASSERT_EQ(1, isize(before.actors));
  ExpectActorData(scene, actor, before.actors[0]);
  EXPECT_FALSE(before.debugDraw.lineVertices.positions.empty());
  EXPECT_FALSE(before.debugDraw.spheres.positions.empty());

  // Move the actor, then step so the new state is captured and synced.
  scene->GetActor(actor)->SetRootTransform(
      TransformRT{Quaternion::RotationX(1.0_r), Real3{5_r, 3_r, -2_r}}, test::ExpectOK{});
  scene->Step(kTimeStep);
  auto after = WaitForSync(scene, baseCounter + 3);
  ASSERT_EQ(1, isize(after.actors));
  ExpectActorData(scene, actor, after.actors[0]);

  // The actor moved, so its synced pose changes.
  EXPECT_NE(before.actors[0].position, after.actors[0].position);

  // The debug-draw geometry is attached to the actor: the same mesh and contact samples, so the
  // amount of data is unchanged, but every position moves with the actor. Comparing equal-sized,
  // non-empty position arrays confirms the geometry moved (rather than merely appearing).
  auto const& beforeLines = before.debugDraw.lineVertices;
  auto const& afterLines = after.debugDraw.lineVertices;
  auto const& beforeSpheres = before.debugDraw.spheres;
  auto const& afterSpheres = after.debugDraw.spheres;
  ASSERT_EQ(beforeLines.positions.size(), afterLines.positions.size());
  ASSERT_EQ(beforeSpheres.positions.size(), afterSpheres.positions.size());
  EXPECT_NE(beforeLines.positions, afterLines.positions);
  EXPECT_NE(beforeSpheres.positions, afterSpheres.positions);
}

TEST_F(SceneSyncTest, SyncDebugDraw) {
  Scene* scene = CreateSceneNoGravity();
  SceneHandle sceneHandle = scene->GetHandle();
  CreateRigidActor(scene);
  EnableActorDebugDraw(scene);

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(sceneHandle); });

  // Sync debug draw only.
  SceneSyncParams params;
  params.enabled = true;
  params.syncDebugDraw = true;
  SetSceneSyncParams(params);
  _client->SelectScene(sceneHandle);
  uint64_t const baseCounter = GetSceneSyncData().counter;

  // Contact-sample spheres are only generated by a step, so sync once the scene has stepped.
  WaitForSync(scene, baseCounter + 1);
  scene->Step(kTimeStep);
  auto data = WaitForSync(scene, baseCounter + 2).debugDraw;

  // Debug draw lines: 3 floats per position, 4 bytes (RGBA) per color.
  EXPECT_FALSE(data.lineVertices.positions.empty());
  EXPECT_EQ(0, isize(data.lineVertices.positions) % 3);
  EXPECT_EQ(0, isize(data.lineVertices.colors) % 4);
  EXPECT_EQ(data.lineVertices.positions.size() / 3, data.lineVertices.colors.size() / 4);

  // Debug draw spheres: 3 floats per position, 1 radius and 4 color bytes per sphere.
  EXPECT_FALSE(data.spheres.positions.empty());
  EXPECT_EQ(0, isize(data.spheres.positions) % 3);
  EXPECT_EQ(0, isize(data.spheres.colors) % 4);
  EXPECT_EQ(data.spheres.radii.size(), data.spheres.positions.size() / 3);
  EXPECT_EQ(data.spheres.radii.size(), data.spheres.colors.size() / 4);
}

TEST_F(SceneSyncTest, DisablingCategoryClearsDataSynchronously) {
  Scene* scene = CreateSceneNoGravity();
  SceneHandle sceneHandle = scene->GetHandle();
  ActorHandle actor = CreateRigidActor(scene);
  EnableActorDebugDraw(scene);

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(sceneHandle); });

  // Sync both actors and debug draw.
  SceneSyncParams params = ActorSyncParams();
  params.syncDebugDraw = true;
  SetSceneSyncParams(params);
  _client->SelectScene(sceneHandle);
  uint64_t const baseCounter = GetSceneSyncData().counter;

  // The immediate reply carries actors; stepping adds debug-draw geometry.
  WaitForSync(scene, baseCounter + 1);
  scene->Step(kTimeStep);
  auto data = WaitForSync(scene, baseCounter + 2);
  ASSERT_EQ(1, isize(data.actors));
  ExpectActorData(scene, actor, data.actors[0]);
  EXPECT_NE(protocol::DbgDrawData{}, data.debugDraw);

  // Disabling debug draw clears its cached copy synchronously and bumps the counter, while the
  // actor data is retained.
  params.syncDebugDraw = false;
  SetSceneSyncParams(params);
  data = GetSceneSyncData();
  EXPECT_EQ(baseCounter + 3, data.counter);
  EXPECT_EQ(protocol::DbgDrawData{}, data.debugDraw);
  ASSERT_EQ(1, isize(data.actors));
  ExpectActorData(scene, actor, data.actors[0]);
}

TEST_F(SceneSyncTest, ChangingSelectionClearsAndResyncs) {
  Scene* sceneA = CreateSceneNoGravity("A");
  ActorHandle actorA = CreateRigidActor(sceneA);
  Scene* sceneB = CreateSceneNoGravity("B");
  ActorHandle actorB = CreateRigidActor(sceneB);

  StartServer();
  ConnectClient();
  test::WaitUntil(
      [&] { return ClientHasScene(sceneA->GetHandle()) && ClientHasScene(sceneB->GetHandle()); });

  _client->SelectScene({});
  SetSceneSyncParams(ActorSyncParams());
  _client->SelectScene(sceneA->GetHandle());
  uint64_t const baseCounter = GetSceneSyncData().counter;
  auto dataA = WaitForSync(sceneA, baseCounter + 1);
  ASSERT_EQ(1, isize(dataA.actors));
  ExpectActorData(sceneA, actorA, dataA.actors[0]);

  // Switching selection clears the cached data synchronously and bumps the counter.
  _client->SelectScene(sceneB->GetHandle());
  EXPECT_EQ(sceneB->GetHandle(), _client->GetSelectedScene());
  EXPECT_TRUE(GetSceneSyncData().actors.empty());
  EXPECT_EQ(baseCounter + 2, GetSceneSyncData().counter); // Incremented by SelectScene

  // Scene B's data arrives after pumping it.
  auto dataB = WaitForSync(sceneB, baseCounter + 3);
  ASSERT_EQ(1, isize(dataB.actors));
  ExpectActorData(sceneB, actorB, dataB.actors[0]);
}

TEST_F(SceneSyncTest, DestroyingSelectedSceneDeselects) {
  Scene* scene = CreateSceneNoGravity();
  CreateRigidActor(scene);

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(scene->GetHandle()); });

  SetSceneSyncParams(ActorSyncParams());
  _client->SelectScene(scene->GetHandle());
  uint64_t const baseCounter = GetSceneSyncData().counter;
  WaitForSync(scene, baseCounter + 1);

  // Destroying the selected scene deselects it and clears the cached data.
  _context->DestroyScene(scene);
  test::WaitUntil([&] { return !_client->GetSelectedScene().IsValid(); });
  EXPECT_TRUE(GetSceneSyncData().actors.empty());
}

TEST_F(SceneSyncTest, DisconnectResetsSelectionAndData) {
  Scene* scene = CreateSceneNoGravity();
  CreateRigidActor(scene);

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(scene->GetHandle()); });

  SetSceneSyncParams(ActorSyncParams());
  _client->SelectScene(scene->GetHandle());
  uint64_t const baseCounter = GetSceneSyncData().counter;
  WaitForSync(scene, baseCounter + 1);

  // Disconnecting resets everything tied to the connection.
  DisconnectClient();
  EXPECT_FALSE(_client->GetSelectedScene().IsValid());
  EXPECT_TRUE(GetSceneSyncData().actors.empty());
  DynamicArray<SceneInfo> scenes;
  _client->GetSceneList(scenes);
  EXPECT_TRUE(scenes.empty());
}

TEST_F(SceneSyncTest, PausedAsyncSceneStillAutoSyncs) {
  // An AsyncScene runs on its own thread, so it services sync requests and delivers data
  // automatically without this thread pumping it, even while paused.
  AsyncScene* asyncScene = _context->CreateAsyncScene("AsyncScene", test::ExpectOK{});
  asyncScene->Pause(true);

  ShapeHandle shape = CreateUnitCube();
  SceneHandle sceneHandle;
  ActorHandle actor;
  asyncScene->QueueCommand([&](auto* scene) {
    sceneHandle = scene->GetHandle();
    actor = scene
                ->CreateRigidActor(
                    RigidActorParams{.name = "AsyncActor", .shape = shape}, test::ExpectOK{})
                ->GetHandle();
  });
  asyncScene->WaitForQueuedCommands();

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(sceneHandle); });

  SetSceneSyncParams(ActorSyncParams());
  _client->SelectScene(sceneHandle);
  uint64_t baseCounter = GetSceneSyncData().counter;

  // Data arrives on its own; note we never call UpdateDebugger/Step from this thread.
  SceneSyncData data;
  test::WaitUntil([&] {
    data = GetSceneSyncData();
    return data.counter >= baseCounter + 1;
  });
  ASSERT_EQ(1, isize(data.actors));
  EXPECT_EQ(actor, data.actors[0].handle);

  // Verify the synced data against the actor on the scene's own thread.
  asyncScene->QueueCommand([this, actorData = data.actors[0]](auto* scene) {
    ExpectActorData(scene, actorData.handle, actorData);
  });
  asyncScene->WaitForQueuedCommands();
}

TEST_F(SceneSyncTest, OneShotRequestDoesNotInterruptSync) {
  // A one-shot request (enableAutoSync left unset) fetches a single snapshot and must not change
  // the server's active syncing. Use a running AsyncScene so periodic syncs keep flowing on its own
  // thread while SendAndAwaitReply blocks this thread.
  AsyncScene* asyncScene = _context->CreateAsyncScene("AsyncScene", test::ExpectOK{});

  ShapeHandle shape = CreateUnitCube();
  SceneHandle sceneHandle;
  asyncScene->QueueCommand([&](auto* scene) {
    scene->SetGravity({});
    sceneHandle = scene->GetHandle();
    scene->CreateRigidActor(
        RigidActorParams{.name = "AsyncActor", .shape = shape}, test::ExpectOK{});
  });
  asyncScene->WaitForQueuedCommands();

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(sceneHandle); });

  // Start active periodic syncing. The running scene delivers data automatically.
  SetSceneSyncParams(ActorSyncParams());
  _client->SelectScene(sceneHandle);
  test::WaitUntil([&] { return GetSceneSyncData().counter >= 1; });

  // Confirm syncing is actively advancing before the one-shot.
  uint64_t const beforeOneShot = GetSceneSyncData().counter;
  test::WaitUntil([&] { return GetSceneSyncData().counter > beforeOneShot; });

  // Issue a one-shot request. It returns a single snapshot...
  protocol::SceneSyncRequest request;
  request.scene = sceneHandle;
  request.syncActors = true;
  auto reply = _client->SendAndAwaitReply(request, kTimeout, test::ExpectOK{});
  ASSERT_TRUE(reply.actors.has_value());
  EXPECT_EQ(1, isize(*reply.actors));

  // ...and must not interrupt the active periodic syncing, which keeps advancing afterwards.
  uint64_t const afterOneShot = GetSceneSyncData().counter;
  test::WaitUntil([&] { return GetSceneSyncData().counter > afterOneShot; });
}

TEST_F(SceneSyncTest, PeriodicSyncThrottledByInterval) {
  Scene* scene = CreateSceneNoGravity();
  SceneHandle sceneHandle = scene->GetHandle();
  CreateRigidActor(scene);

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(sceneHandle); });

  auto sceneDebugger = assert_cast<SceneImpl*>(scene)->GetDebugger();
  ASSERT_NE(nullptr, sceneDebugger);
  std::atomic<double> clockSec{0.0};
  sceneDebugger->SetClock([&] { return clockSec.load(); });

  SceneSyncParams params = ActorSyncParams();
  params.syncInterval = 100.0f;
  SetSceneSyncParams(params);
  _client->SelectScene(sceneHandle);
  uint64_t const baseCounter = GetSceneSyncData().counter;

  // The immediate reply from enabling sync arrives (after the selection snapshot) and starts the
  // interval timer at fake time zero.
  WaitForSync(scene, baseCounter + 1);

  // The fake clock advances by only one simulation frame per step, so periodic syncs are still
  // suppressed even though the playback throttle is allowed to release each step.
  for (int i = 0; i < 20; ++i) {
    clockSec.store(clockSec.load() + kTimeStep);
    scene->Step(kTimeStep);
    scene->UpdateDebugger();
  }
  ExpectNoMoreSync(scene, baseCounter + 1);

  // Once the fake interval elapses, the next step can produce a periodic sync.
  clockSec.store(params.syncInterval);
  scene->Step(kTimeStep);
  WaitForSync(scene, baseCounter + 2);
}

TEST_F(SceneSyncTest, SwitchMeshSourceResendsCompleteMeshesWithoutStepping) {
  EXPECT_TRUE(_client->GetSettings().sync.useVisualMesh);

  Scene* scene = CreateSceneNoGravity();
  SceneHandle const sceneHandle = scene->GetHandle();
  ActorHandle const rigidHandle = CreateRigidActorWithVisualMesh(scene);
  ActorHandle const visualOnlyHandle = CreateActorWithVisualMeshOnly(scene);
  ActorHandle const shellHandle = CreateShellActorWithVisualMesh(scene);
  Actor* const rigidActor = scene->GetActor(rigidHandle);
  Actor* const visualOnlyActor = scene->GetActor(visualOnlyHandle);
  Actor* const shellActor = scene->GetActor(shellHandle);
  ASSERT_NE(nullptr, rigidActor);
  ASSERT_NE(nullptr, visualOnlyActor);
  ASSERT_NE(nullptr, shellActor);
  auto const expectEqualCountsDifferentConnectivity = [](Actor const* actor) {
    ASSERT_EQ(actor->GetVisualMesh().GetNumNodes(), actor->GetSurfaceMesh().GetNumNodes());
    ASSERT_EQ(actor->GetVisualMesh().GetNumElements(), actor->GetSurfaceMesh().GetNumElements());
    ASSERT_NE(actor->GetVisualMesh().connectivity, actor->GetSurfaceMesh().connectivity);
  };
  expectEqualCountsDifferentConnectivity(rigidActor);
  expectEqualCountsDifferentConnectivity(shellActor);
  ASSERT_TRUE(visualOnlyActor->GetSurfaceMesh().connectivity.empty());

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(sceneHandle); });

  SceneSyncParams params = MeshSyncParams();
  SetSceneSyncParams(params);
  _client->SelectScene(sceneHandle);
  uint64_t const baseCounter = GetSceneSyncData().counter;
  WaitForSync(scene, baseCounter + 1);

  uint64_t const rigidVisualRevision =
      ExpectActorMeshMatches(rigidHandle, rigidActor->GetVisualMesh());
  uint64_t const visualOnlyRevision =
      ExpectActorMeshMatches(visualOnlyHandle, visualOnlyActor->GetVisualMesh());
  uint64_t const shellVisualRevision =
      ExpectActorMeshMatches(shellHandle, shellActor->GetVisualMesh());
  EXPECT_GT(rigidVisualRevision, 0);
  EXPECT_GT(visualOnlyRevision, 0);
  EXPECT_GT(shellVisualRevision, 0);

  params.useVisualMesh = false;
  SetSceneSyncParams(params);

  // The source-only setting change retains the current mesh until the scene-thread request runs.
  EXPECT_EQ(baseCounter + 1, GetSceneSyncData().counter);
  EXPECT_EQ(rigidVisualRevision, ExpectActorMeshMatches(rigidHandle, rigidActor->GetVisualMesh()));
  EXPECT_EQ(
      visualOnlyRevision,
      ExpectActorMeshMatches(visualOnlyHandle, visualOnlyActor->GetVisualMesh()));
  EXPECT_EQ(shellVisualRevision, ExpectActorMeshMatches(shellHandle, shellActor->GetVisualMesh()));

  // Pump only the debugger queue. The complete simulation meshes arrive without a simulation step.
  WaitForSync(scene, baseCounter + 2);
  uint64_t const rigidSurfaceRevision =
      ExpectActorMeshMatches(rigidHandle, rigidActor->GetSurfaceMesh());
  uint64_t const shellSurfaceRevision =
      ExpectActorMeshMatches(shellHandle, shellActor->GetSurfaceMesh());
  EXPECT_GT(rigidSurfaceRevision, rigidVisualRevision);
  EXPECT_GT(shellSurfaceRevision, shellVisualRevision);
  ExpectActorMeshEmpty(visualOnlyHandle);

  // Normal cadence resumes: rigid geometry remains cached and the shell sends live coordinates
  // while retaining the new connectivity revision.
  scene->Step(kTimeStep);
  WaitForSync(scene, baseCounter + 3);
  EXPECT_EQ(
      rigidSurfaceRevision, ExpectActorMeshMatches(rigidHandle, rigidActor->GetSurfaceMesh()));
  EXPECT_EQ(
      shellSurfaceRevision,
      ExpectActorMeshTopologyMatches(shellHandle, shellActor->GetSurfaceMesh()));
  ExpectActorMeshEmpty(visualOnlyHandle);

  // Switching back starts another epoch and re-registers the visual node-position query.
  params.useVisualMesh = true;
  SetSceneSyncParams(params);
  WaitForSync(scene, baseCounter + 4);
  EXPECT_GT(ExpectActorMeshMatches(rigidHandle, rigidActor->GetVisualMesh()), rigidSurfaceRevision);
  EXPECT_GT(
      ExpectActorMeshMatches(visualOnlyHandle, visualOnlyActor->GetVisualMesh()),
      visualOnlyRevision);
  EXPECT_GT(
      ExpectActorMeshTopologyMatches(shellHandle, shellActor->GetVisualMesh()),
      shellSurfaceRevision);
}

TEST_F(SceneSyncTest, SyncActorMeshes) {
  Scene* scene = CreateSceneNoGravity();
  SceneHandle sceneHandle = scene->GetHandle();
  ActorHandle rigidActor = CreateRigidActor(scene);
  ActorHandle softActor = CreateSoftActor(scene);

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return ClientHasScene(sceneHandle); });

  SetSceneSyncParams(MeshSyncParams());
  _client->SelectScene(sceneHandle);
  uint64_t const baseCounter = GetSceneSyncData().counter;

  // The immediate reply combines both actors and their surface meshes into one buffer.
  auto syncData = WaitForSync(scene, baseCounter + 1);
  ASSERT_EQ(2, isize(syncData.actors));
  ExpectActorMeshMatches(scene, rigidActor);
  ExpectActorMeshMatches(scene, softActor);

  // Step repeatedly to receive several more updates. On each subsequent sync the wire omits data
  // the client already has: the rigid mesh (sent once) and the soft actor's connectivity (sent
  // once, then coordinates-only). ExpectActorMeshMatches reads the client's merged cache, so it
  // passes only if the cache correctly retained the omitted rigid mesh and soft connectivity.
  for (int i = 0; i < 3; ++i) {
    scene->Step(kTimeStep);
    syncData = WaitForSync(scene, baseCounter + 2 + i);
    ASSERT_EQ(2, isize(syncData.actors));
    ExpectActorMeshMatches(scene, rigidActor);
    ExpectActorMeshMatches(scene, softActor);
  }

  // Create another actor that didn't exist before.
  ActorHandle rigidActor2 = CreateRigidActor(scene);
  scene->Step(kTimeStep);
  syncData = WaitForSync(scene, baseCounter + 5);
  ASSERT_EQ(3, isize(syncData.actors));
  ExpectActorMeshMatches(scene, rigidActor);
  ExpectActorMeshMatches(scene, rigidActor2);
  ExpectActorMeshMatches(scene, softActor);

  // Destroy the first rigid actor.
  scene->DestroyActor(rigidActor);
  scene->Step(kTimeStep);
  syncData = WaitForSync(scene, baseCounter + 6);
  ASSERT_EQ(2, isize(syncData.actors));
  ExpectActorMeshMatches(scene, rigidActor2);
  ExpectActorMeshMatches(scene, softActor);
}

TEST_F(SceneSyncTest, SyncInitialStatePaused) {
  _client->SetSceneStepMode(StepMode::Pause);
  _client->SetSettings(DebugClientSettings{.sync = MeshSyncParams()});

  StartServer();
  ConnectClient();

  // Create new scene on the server. It stays paused because the client's step mode is Pause.
  Scene* scene = CreateSceneNoGravity();
  SceneHandle sceneHandle = scene->GetHandle();
  ActorHandle rigidActor = CreateRigidActor(scene);

  ClientSelectScene(sceneHandle);
  uint64_t const baseCounter = GetSceneSyncData().counter;

  // Step the scene on another thread. This will block until we unpause the server.
  std::thread thread{[&]() { scene->Step(kTimeStep); }};

  // Wait for the client to receive sync data about the initial state (before first step).
  auto syncData = WaitForSync(scene, baseCounter + 1);
  ASSERT_EQ(1, isize(syncData.actors));
  ExpectActorData(scene, rigidActor, syncData.actors[0]);
  ExpectActorMeshMatches(scene, rigidActor);

  // Unpause so the thread can finish.
  _client->SetSceneStepMode(StepMode::Play);
  thread.join();
}
