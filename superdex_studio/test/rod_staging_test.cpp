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

#include "app/app.h"
#include "assets/asset_manager.h"
#include "rendering/rod_visualization.h"
#include "rendering/scene_stage.h"

#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/mochi_init.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_physics/utils/mochi_prefab.h>
#include <mochi_renderer/mesh.h>
#include <mochi_renderer/mochi_renderer.h>
#include <mochi_renderer/scene.h>
#include <mochi_renderer/type_conversions.h>

#include <gtest/gtest.h>
#include <imgui.h>

using namespace superdex::studio;
using namespace mochi;

namespace {

class StagingStudio : public SuperDexStudio {
 public:
  StagingStudio() {
    _persistSettings = false;
    ImGui::GetIO().IniFilename = nullptr;
    _mochiRenderer = mochi_renderer::MochiRenderer::Create();
    _mochiContext = CreateContext(0);
    _assetManager = AssetManager::Create(this);
  }

  ~StagingStudio() override {
    _assetManager.reset();
    DestroyContext(_mochiContext);
    _mochiContext = nullptr;
  }

  mochi_renderer::Scene* RenderScene() const {
    return _mochiRenderer->GetScene();
  }
};

TEST(RodStagingTest, NestedScaleReuseColliderChangesAndPoseReset) {
  Initialize();
  MOCHI_DEFER(ShutDown());
  auto directory = CreateTempDirectory("studio_rod_staging", ErrorAssert{});
  ModelData model;
  auto& mesh = model.mesh.emplace();
  mesh.nodesPerElement = 2;
  mesh.coordinates = {0_r, 0_r, 0_r, 0.1_r, 0_r, 0_r, 0.18_r, 0.06_r, 0_r};
  auto const path = (directory.Path() / "rod.mochi.json").string();
  model::SaveToFile(model, path, FileFormat::JSON, ErrorAssert{});

  prefab::ScenePrefab prefab;
  auto nested = std::make_shared<prefab::ScenePrefab>();
  auto& reference = prefab.prefabs.push_back();
  reference.prefab = nested;
  reference.name = "nested";
  reference.scale = 3_r;
  auto& rod = nested->actors.rod.push_back();
  rod.name = "rod";
  rod.shapeFile = path;
  rod.scale = {2_r, -2_r, 2_r};
  rod.shapeRotation = Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, 0.6_r);
  rod.shapeTranslation = {0.2_r, 0.1_r, -0.1_r};
  rod.colliderType = ColliderType::PointCloud;
  rod.pointCloudCollider.radius = 0.01_r;
  rod.pointCloudCollider.colliderSegmentElementType = ActorSegmentElementType::P1Q2;

  StagingStudio studio;
  SceneStage stage(&studio, "rod_staging_test");
  stage.BindRenderScene(studio.RenderScene());
  constexpr StageType kCollider = StageType::SdfFallbackToMochiModel;
  ASSERT_TRUE(stage.StagePrefab(prefab, directory.Path().string(), kCollider));
  ASSERT_EQ(stage.GetActors().size(), 1);

  auto checkNativeCollider = [&] {
    auto* context = studio.GetMochiContext();
    auto* scene = context->CreateScene("native_rod_staging_test");
    MOCHI_DEFER(context->DestroyScene(scene));
    prefab::LoadShapes(prefab, directory.Path().string(), context, ErrorAssert{});
    auto const actors = prefab::AddToScene(prefab, scene, {}, ErrorAssert{});
    ASSERT_EQ(actors.Filter(ActorType::Rod).size(), 1);
    auto& draw = scene->GetDebugDraw();
    draw.Enable(true);
    draw.EnableFeature(draw.FindFeature("Point Cloud Collider"), true);
    auto const spheres = draw.GatherData().spheres;
    auto const& staged = stage.GetActors()[0];
    ASSERT_TRUE(staged.rod->collider.has_value());
    auto const samples = ComputeRodColliderSamples(
        MakeConstSpan(staged.rod->pose.nodes),
        MakeConstSpan(staged.rod->connectivity),
        staged.rod->collider->colliderSegmentElementType);
    ASSERT_EQ(samples.size(), spheres.positions.size());
    for (size_t i = 0; i < samples.size(); ++i) {
      auto const world = staged.worldTransform.TransformPoint(samples[i]);
      EXPECT_LT(Norm(world - spheres.positions[i]), 1e-5_r);
      EXPECT_NEAR(staged.rod->collider->radius, spheres.radii[i], 1e-6_r);
    }
  };
  checkNativeCollider();
  auto* colliderMesh = stage.GetActors()[0].rodColliderMesh;
  ASSERT_NE(colliderMesh, nullptr);
  auto const initialBox = colliderMesh->GetAABB();

  rod.scale *= 0.5_r;
  rod.shape = {};
  rod.pointCloudCollider.radius *= 2_r;
  EXPECT_FALSE(stage.StagePrefab(prefab, directory.Path().string(), kCollider));
  EXPECT_EQ(stage.GetActors()[0].rodColliderMesh, colliderMesh);
  checkNativeCollider();
  EXPECT_NE(initialBox.center, colliderMesh->GetAABB().center);

  auto const restBox = colliderMesh->GetAABB();
  DeformableMeshUpdate update;
  update.stagedIndex = 0;
  update.name = stage.GetActors()[0].name;
  update.worldTransform = stage.GetActors()[0].worldTransform;
  update.rodPose = stage.GetActors()[0].rod->pose;
  for (size_t i = 1; i < update.rodPose.nodes.size(); i += 3) {
    update.rodPose.nodes[i] += 0.4_r;
  }
  auto const& converter = studio.GetEditorToRendererSpaceConverter();
  stage.ApplyDeformableMeshUpdates(Span<DeformableMeshUpdate const>{&update, 1}, converter);
  EXPECT_NE(restBox.center, colliderMesh->GetAABB().center);
  stage.ResetWorldTransforms(converter);
  EXPECT_EQ(restBox.center, colliderMesh->GetAABB().center);
  EXPECT_EQ(restBox.halfExtent, colliderMesh->GetAABB().halfExtent);

  rod.pointCloudCollider.colliderSegmentElementType.reset();
  EXPECT_TRUE(stage.StagePrefab(prefab, directory.Path().string(), kCollider));
  checkNativeCollider();
  rod.colliderType = ColliderType::None;
  EXPECT_TRUE(stage.StagePrefab(prefab, directory.Path().string(), kCollider));
  EXPECT_EQ(stage.GetActors()[0].rodColliderMesh, nullptr);
  rod.colliderType = ColliderType::Auto;
  EXPECT_FALSE(stage.StagePrefab(prefab, directory.Path().string(), kCollider));
  EXPECT_NE(stage.GetActors()[0].rodColliderMesh, nullptr);
  checkNativeCollider();
}

} // namespace
