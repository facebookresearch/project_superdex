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

#include <mochi_renderer/windows_compat.h>

#include <mochi_renderer/camera.h>
#include <mochi_renderer/debug.h>
#include <mochi_renderer/deformable_model_instance.h>
#include <mochi_renderer/mochi_renderer.h>
#include <mochi_renderer/resource.h>
#include <mochi_renderer/resource_manager.h>
#include <mochi_renderer/scene.h>
#include <mochi_renderer/types.h>
#include <mochi_renderer/utils.h>

#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/file_utils.h>

#include <gtest/gtest.h>

#include <filament/RenderableManager.h>
#include <filament/Scene.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "test_assets.h"

namespace mochi_renderer {

struct DebugDrawTestAccess {
  static size_t GetSolidSphereBatchCount(DebugDraw const& debugDraw) {
    return debugDraw.GetSolidSphereBatchCount();
  }

  static utils::Entity GetSolidSphereBatchEntity(DebugDraw const& debugDraw, size_t batchIndex) {
    return debugDraw.GetSolidSphereBatchEntity(batchIndex);
  }
};

} // namespace mochi_renderer

using namespace mochi_renderer;

// Returns true if the GPU backend (Vulkan/Metal) is available.
// Probes once by attempting to create a MochiRenderer; the result is cached.
static bool IsGpuAvailable() {
  static bool const available = (MochiRenderer::Create() != nullptr);
  return available;
}

// Macro to skip GPU tests when no GPU is present.
#define SKIP_IF_NO_GPU()                  \
  do {                                    \
    if (!IsGpuAvailable()) {              \
      GTEST_SKIP() << "No GPU available"; \
    }                                     \
  } while (false)

class DebugDrawTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!IsGpuAvailable()) {
      GTEST_SKIP() << "No GPU available";
    }

    _renderer = MochiRenderer::Create();
    ASSERT_NE(_renderer, nullptr);
    _scene = _renderer->GetScene();
    ASSERT_NE(_scene, nullptr);
    _debugDraw = _scene->CreateDebugDraw();
    ASSERT_NE(_debugDraw, nullptr);
  }

  void DrawAndCommitTestSpheres(size_t count) {
    for (size_t i = 0; i < count; ++i) {
      _debugDraw->DrawSolidSphere({0.0f, 0.0f, -3.0f}, 0.1f, {1.0f, 0.0f, 0.0f, 1.0f});
    }
    _debugDraw->Commit();
  }

  utils::Entity GetSphereBatchEntity(size_t batchIndex) const {
    return DebugDrawTestAccess::GetSolidSphereBatchEntity(*_debugDraw, batchIndex);
  }

  uint32_t GetSphereBatchInstanceCount(size_t batchIndex) const {
    auto& renderables = _renderer->GetEngine()->getRenderableManager();
    return renderables.getInstanceCount(renderables.getInstance(GetSphereBatchEntity(batchIndex)));
  }

  std::unique_ptr<MochiRenderer> _renderer;
  Scene* _scene = nullptr;
  DebugDraw* _debugDraw = nullptr;
};

TEST(MochiRendererTest, CreateAndDestroy) {
  SKIP_IF_NO_GPU();
  MochiRenderer::Config config;
  config.pipelineMode = PipelineMode::Synchronized;
  auto renderer = MochiRenderer::Create(config);
  ASSERT_NE(renderer, nullptr);
  EXPECT_NE(renderer->GetEngine(), nullptr);
  EXPECT_NE(renderer->GetScene(), nullptr);
  EXPECT_NE(renderer->GetResourceManager(), nullptr);
  EXPECT_EQ(renderer->GetPipelineMode(), PipelineMode::Synchronized);
  EXPECT_FALSE(renderer->HasPresentationTarget());
}

TEST(MochiRendererTest, HasIblSkyboxDistinguishesColorSkybox) {
  SKIP_IF_NO_GPU();
  auto renderer = MochiRenderer::Create();
  ASSERT_NE(renderer, nullptr);
  auto* scene = renderer->GetScene();
  ASSERT_NE(scene, nullptr);

  // A solid-color background skybox is not an IBL skybox, so a /scene/background
  // request must not be treated as overwriting an IBL environment.
  EXPECT_FALSE(scene->HasIblSkybox());
  scene->CreateSkybox({0.92f, 0.92f, 0.94f, 1.0f});
  EXPECT_FALSE(scene->HasIblSkybox());
}

TEST(MochiRendererTest, CreateObservationCamera) {
  SKIP_IF_NO_GPU();
  auto renderer = MochiRenderer::Create();
  ASSERT_NE(renderer, nullptr);

  auto* cam = renderer->CreateObservationCamera("test_cam", 320, 240);
  ASSERT_NE(cam, nullptr);
  EXPECT_EQ(cam->GetName(), "test_cam");
  EXPECT_EQ(cam->GetConfig().width, 320);
  EXPECT_EQ(cam->GetConfig().height, 240);

  // Duplicate name should return nullptr
  auto* cam2 = renderer->CreateObservationCamera("test_cam", 640, 480);
  EXPECT_EQ(cam2, nullptr);

  // Different name should succeed
  auto* cam3 = renderer->CreateObservationCamera("other_cam", 640, 480);
  ASSERT_NE(cam3, nullptr);

  auto names = renderer->GetObservationCameraNames();
  EXPECT_EQ(names.size(), 2);
}

TEST(MochiRendererTest, ObservationCameraTransform) {
  SKIP_IF_NO_GPU();
  auto renderer = MochiRenderer::Create();
  auto* cam = renderer->CreateObservationCamera("cam", 64, 64);
  ASSERT_NE(cam, nullptr);

  cam->SetTransform({1.0, 2.0, 3.0}, {0, 0, 0, 1});
  auto pos = cam->GetPosition();
  EXPECT_NEAR(pos.x, 1.0, 1e-5);
  EXPECT_NEAR(pos.y, 2.0, 1e-5);
  EXPECT_NEAR(pos.z, 3.0, 1e-5);
}

TEST(MochiRendererTest, ObservationCameraResolution) {
  SKIP_IF_NO_GPU();
  auto renderer = MochiRenderer::Create();
  auto* cam = renderer->CreateObservationCamera("cam", 320, 240);
  ASSERT_NE(cam, nullptr);

  cam->SetResolution(640, 480);
  EXPECT_EQ(cam->GetConfig().width, 640);
  EXPECT_EQ(cam->GetConfig().height, 480);
}

TEST(MochiRendererTest, SynchronizedRenderAndReadback) {
  SKIP_IF_NO_GPU();
  auto renderer = MochiRenderer::Create();
  auto* cam = renderer->CreateObservationCamera("cam", 64, 64);
  ASSERT_NE(cam, nullptr);

  // Render and readback
  auto results = renderer->RenderAndReadback({"cam"});
  ASSERT_TRUE(results.contains("cam"));

  auto& result = results["cam"];
  EXPECT_TRUE(result.IsValid());
  EXPECT_EQ(result.width, 64);
  EXPECT_EQ(result.height, 64);
  EXPECT_EQ(result.channels, 4);
  EXPECT_EQ(result.pixels.size(), 64 * 64 * 4);
}

TEST_F(DebugDrawTest, SpheresShareOneInstancedRenderable) {
  _debugDraw->DrawSolidSphere({-1.0f, 0.0f, -3.0f}, 0.5f, {1.0f, 0.0f, 0.0f, 1.0f});
  _debugDraw->DrawSolidSphere({1.0f, 0.0f, -3.0f}, 0.5f, {0.0f, 1.0f, 0.0f, 0.5f});
  _debugDraw->Commit();

  EXPECT_EQ(GetSphereBatchInstanceCount(0), 2);
}

TEST_F(DebugDrawTest, SphereOverflowCreatesMultipleBatches) {
  constexpr size_t kSphereCount = 65535;
  DrawAndCommitTestSpheres(kSphereCount);

  ASSERT_EQ(DebugDrawTestAccess::GetSolidSphereBatchCount(*_debugDraw), 3);
  EXPECT_NE(GetSphereBatchEntity(0), GetSphereBatchEntity(1));
  EXPECT_TRUE(_scene->GetFilamentScene()->hasEntity(GetSphereBatchEntity(0)));
  EXPECT_TRUE(_scene->GetFilamentScene()->hasEntity(GetSphereBatchEntity(1)));
  EXPECT_TRUE(_scene->GetFilamentScene()->hasEntity(GetSphereBatchEntity(2)));
  EXPECT_EQ(GetSphereBatchInstanceCount(0), 32767);
  EXPECT_EQ(GetSphereBatchInstanceCount(1), 32767);
  EXPECT_EQ(GetSphereBatchInstanceCount(2), 1);
}

TEST_F(DebugDrawTest, SphereBatchDrawCountShrinksAfterUsageDrops) {
  constexpr size_t kPeakSphereCount = 20000;
  DrawAndCommitTestSpheres(kPeakSphereCount);

  auto const entity = GetSphereBatchEntity(0);
  EXPECT_EQ(GetSphereBatchInstanceCount(0), 32767);

  _debugDraw->Clear();
  EXPECT_EQ(GetSphereBatchInstanceCount(0), 32767);
  EXPECT_FALSE(_scene->GetFilamentScene()->hasEntity(entity));

  constexpr size_t kReducedSphereCount = 10;
  DrawAndCommitTestSpheres(kReducedSphereCount);

  EXPECT_EQ(GetSphereBatchEntity(0), entity);
  EXPECT_EQ(GetSphereBatchInstanceCount(0), 16);
  EXPECT_TRUE(_scene->GetFilamentScene()->hasEntity(entity));
}

TEST_F(DebugDrawTest, SphereOverflowBatchesAreReusedAfterClear) {
  constexpr size_t kSphereCount = 32768;
  DrawAndCommitTestSpheres(kSphereCount);

  ASSERT_EQ(DebugDrawTestAccess::GetSolidSphereBatchCount(*_debugDraw), 2);
  auto const firstEntity = GetSphereBatchEntity(0);
  auto const secondEntity = GetSphereBatchEntity(1);
  uint32_t const firstInstanceCount = GetSphereBatchInstanceCount(0);
  uint32_t const secondInstanceCount = GetSphereBatchInstanceCount(1);
  auto* filamentScene = _scene->GetFilamentScene();
  _debugDraw->Clear();
  EXPECT_FALSE(filamentScene->hasEntity(firstEntity));
  EXPECT_FALSE(filamentScene->hasEntity(secondEntity));
  EXPECT_EQ(GetSphereBatchInstanceCount(0), firstInstanceCount);
  EXPECT_EQ(GetSphereBatchInstanceCount(1), secondInstanceCount);

  DrawAndCommitTestSpheres(kSphereCount);

  ASSERT_EQ(DebugDrawTestAccess::GetSolidSphereBatchCount(*_debugDraw), 2);
  EXPECT_EQ(GetSphereBatchEntity(0), firstEntity);
  EXPECT_EQ(GetSphereBatchEntity(1), secondEntity);
  EXPECT_TRUE(filamentScene->hasEntity(firstEntity));
  EXPECT_TRUE(filamentScene->hasEntity(secondEntity));
  EXPECT_EQ(GetSphereBatchInstanceCount(0), firstInstanceCount);
  EXPECT_EQ(GetSphereBatchInstanceCount(1), secondInstanceCount);
}

TEST(MochiRendererTest, PipelineModeSwitch) {
  SKIP_IF_NO_GPU();
  MochiRenderer::Config config;
  config.pipelineMode = PipelineMode::Synchronized;
  auto renderer = MochiRenderer::Create(config);
  EXPECT_EQ(renderer->GetPipelineMode(), PipelineMode::Synchronized);

  renderer->SetPipelineMode(PipelineMode::OneFrameDelay);
  EXPECT_EQ(renderer->GetPipelineMode(), PipelineMode::OneFrameDelay);

  renderer->SetPipelineMode(PipelineMode::MaxPerformance);
  EXPECT_EQ(renderer->GetPipelineMode(), PipelineMode::MaxPerformance);
}

TEST(MochiRendererTest, DestroyObservationCamera) {
  SKIP_IF_NO_GPU();
  auto renderer = MochiRenderer::Create();
  renderer->CreateObservationCamera("cam", 64, 64);
  EXPECT_NE(renderer->GetObservationCamera("cam"), nullptr);

  renderer->DestroyObservationCamera("cam");
  EXPECT_EQ(renderer->GetObservationCamera("cam"), nullptr);
}

TEST(MochiRendererTest, ViewSettings) {
  SKIP_IF_NO_GPU();
  MochiRenderer::Config config;
  config.viewSettings.msaaEnabled = false;
  config.viewSettings.bloomEnabled = false;
  auto renderer = MochiRenderer::Create(config);
  EXPECT_FALSE(renderer->GetViewSettings().msaaEnabled);
  EXPECT_FALSE(renderer->GetViewSettings().bloomEnabled);

  SceneViewSettings newSettings;
  newSettings.msaaEnabled = true;
  newSettings.msaaSampleCount = 8;
  renderer->SetViewSettings(newSettings);
  EXPECT_TRUE(renderer->GetViewSettings().msaaEnabled);
  EXPECT_EQ(renderer->GetViewSettings().msaaSampleCount, 8);
}

// --- OpenCV Camera Model Projection Tests ---

namespace {

struct Centroid {
  float x;
  float y;
  int count;
};

// Scan all pixels and return the centroid of pixels matching the predicate.
template <typename Pred>
std::optional<Centroid> FindColorCentroid(RenderResult const& result, Pred predicate) {
  double sumX = 0;
  double sumY = 0;
  int count = 0;

  for (int y = 0; y < result.height; ++y) {
    for (int x = 0; x < result.width; ++x) {
      size_t idx = (static_cast<size_t>(y) * result.width + x) * result.channels;
      uint8_t r = result.pixels[idx + 0];
      uint8_t g = result.pixels[idx + 1];
      uint8_t b = result.pixels[idx + 2];
      if (predicate(r, g, b)) {
        sumX += x;
        sumY += y;
        ++count;
      }
    }
  }

  if (count == 0) {
    return std::nullopt;
  }
  return Centroid{static_cast<float>(sumX / count), static_cast<float>(sumY / count), count};
}

bool IsRedPixel(uint8_t r, uint8_t g, uint8_t b) {
  return r > 100 && g < 80 && b < 80;
}

} // namespace

TEST(MochiRendererOpenCVCameraTest, PinholeProjection) {
  SKIP_IF_NO_GPU();
  auto renderer = MochiRenderer::Create();
  ASSERT_NE(renderer, nullptr);

  auto* cam = renderer->CreateObservationCamera("cv_cam", 640, 480);
  ASSERT_NE(cam, nullptr);

  // Pinhole camera model: no distortion
  OpenCVCameraModel model;
  model.fx = 500.0f;
  model.fy = 500.0f;
  model.cx = 320.0f;
  model.cy = 240.0f;
  cam->SetCameraModel(model);

  // Camera at origin with identity rotation
  cam->SetTransform({0, 0, 0}, {0, 0, 0, 1});

  // 3D point in OpenCV camera space: X-right, Y-down, Z-forward
  float const ocvX = 0.5f;
  float const ocvY = 0.3f;
  float const ocvZ = 3.0f;

  // Convert to OpenGL/Filament coords: X-right, Y-up, Z-backward
  float const glX = ocvX;
  float const glY = -ocvY;
  float const glZ = -ocvZ;

  // Sunlight from camera direction — illuminates the visible hemisphere uniformly
  // so directional shading doesn't bias the red-pixel centroid.
  auto* scene = renderer->GetScene();
  scene->CreateSunlight(100000.0f, {0.0f, 0.0f, -1.0f});
  auto* dd = scene->CreateDebugDraw();
  ASSERT_NE(dd, nullptr);
  dd->DrawSolidSphere({glX, glY, glZ}, 0.05f, {1.0f, 0.0f, 0.0f, 1.0f});
  dd->Commit();

  // Analytically compute expected pixel: u = fx*(X/Z) + cx, v = fy*(Y/Z) + cy
  float const expectedU = model.fx * (ocvX / ocvZ) + model.cx;
  float const expectedV = model.fy * (ocvY / ocvZ) + model.cy;

  // Render and readback
  auto results = renderer->RenderAndReadback({"cv_cam"});
  ASSERT_TRUE(results.contains("cv_cam"));
  auto& result = results["cv_cam"];
  ASSERT_TRUE(result.IsValid());

  // Find centroid of red pixels
  auto centroid = FindColorCentroid(result, IsRedPixel);
  ASSERT_TRUE(centroid.has_value()) << "No red pixels found in rendered image";

  EXPECT_NEAR(centroid->x, expectedU, 3.0f)
      << "Centroid X: " << centroid->x << " expected: " << expectedU;
  EXPECT_NEAR(centroid->y, expectedV, 3.0f)
      << "Centroid Y: " << centroid->y << " expected: " << expectedV;
}

TEST(MochiRendererOpenCVCameraTest, DistortedProjection) {
  SKIP_IF_NO_GPU();
  auto renderer = MochiRenderer::Create();
  ASSERT_NE(renderer, nullptr);

  auto* cam = renderer->CreateObservationCamera("cv_cam", 640, 480);
  ASSERT_NE(cam, nullptr);

  // Camera model with distortion
  OpenCVCameraModel model;
  model.fx = 500.0f;
  model.fy = 500.0f;
  model.cx = 320.0f;
  model.cy = 240.0f;
  model.k1 = 0.1f;
  model.k2 = 0.01f;
  model.p1 = 0.001f;
  model.p2 = 0.001f;
  model.k3 = 0.0001f;
  cam->SetCameraModel(model);

  cam->SetTransform({0, 0, 0}, {0, 0, 0, 1});

  float const ocvX = 0.5f;
  float const ocvY = 0.3f;
  float const ocvZ = 3.0f;

  auto* scene = renderer->GetScene();
  scene->CreateSunlight(100000.0f, {0.0f, 0.0f, -1.0f});
  auto* dd = scene->CreateDebugDraw();
  ASSERT_NE(dd, nullptr);
  dd->DrawSolidSphere({ocvX, -ocvY, -ocvZ}, 0.05f, {1.0f, 0.0f, 0.0f, 1.0f});
  dd->Commit();

  // Full OpenCV distortion model
  float const x = ocvX / ocvZ;
  float const y = ocvY / ocvZ;
  float const r2 = x * x + y * y;
  float const r4 = r2 * r2;
  float const r6 = r4 * r2;
  float const radial = 1.0f + model.k1 * r2 + model.k2 * r4 + model.k3 * r6;
  float const xd = x * radial + 2.0f * model.p1 * x * y + model.p2 * (r2 + 2.0f * x * x);
  float const yd = y * radial + model.p1 * (r2 + 2.0f * y * y) + 2.0f * model.p2 * x * y;
  float const expectedU = model.fx * xd + model.cx;
  float const expectedV = model.fy * yd + model.cy;

  // Sanity: distortion should shift pixel vs pure pinhole
  float const undistortedU = model.fx * x + model.cx;
  EXPECT_NE(expectedU, undistortedU) << "Distortion should shift pixel location";

  auto results = renderer->RenderAndReadback({"cv_cam"});
  ASSERT_TRUE(results.contains("cv_cam"));
  auto& result = results["cv_cam"];
  ASSERT_TRUE(result.IsValid());

  auto centroid = FindColorCentroid(result, IsRedPixel);
  ASSERT_TRUE(centroid.has_value()) << "No red pixels found in rendered image";

  // Slightly larger tolerance than pinhole: distortion shifts different parts of the
  // finite-size sphere differently, so the pixel centroid doesn't match the projected
  // center point as precisely.
  EXPECT_NEAR(centroid->x, expectedU, 5.0f)
      << "Centroid X: " << centroid->x << " expected: " << expectedU;
  EXPECT_NEAR(centroid->y, expectedV, 5.0f)
      << "Centroid Y: " << centroid->y << " expected: " << expectedV;
}

// --- STL Import + Render Tests ---

namespace {

bool IsNonBlackPixel(uint8_t r, uint8_t g, uint8_t b) {
  return (r > 10 || g > 10 || b > 10);
}

} // namespace

// --- GLB Import + Render Tests ---

TEST(GlbRenderTest, LoadGlbAsset) {
  SKIP_IF_NO_GPU();
  std::string glbPath = test::GetTestAssetPath("basic_shapes/Cube.glb");
  ASSERT_TRUE(std::filesystem::exists(glbPath)) << "Cube.glb not found: " << glbPath;

  auto renderer = MochiRenderer::Create();
  ASSERT_NE(renderer, nullptr);

  auto* resourceManager = renderer->GetResourceManager();
  ASSERT_NE(resourceManager, nullptr);

  auto* asset = resourceManager->LoadResource(glbPath);
  ASSERT_NE(asset, nullptr) << "LoadAsset returned nullptr for Cube.glb";
  EXPECT_EQ(asset->GetType(), ResourceType::RenderModel);

  auto* renderAsset = dynamic_cast<RenderModel*>(asset);
  ASSERT_NE(renderAsset, nullptr);
  EXPECT_EQ(renderAsset->GetOriginalFormat(), RenderModelFormat::Gltf);
}

TEST(GlbRenderTest, LoadGlbAndGetInstance) {
  SKIP_IF_NO_GPU();
  std::string glbPath = test::GetTestAssetPath("basic_shapes/Cube.glb");
  ASSERT_TRUE(std::filesystem::exists(glbPath));

  auto renderer = MochiRenderer::Create();
  auto* asset = renderer->GetResourceManager()->LoadRenderModel(glbPath);
  ASSERT_NE(asset, nullptr);

  auto instance = asset->GetInstance();
  ASSERT_NE(instance, nullptr) << "GetInstance returned nullptr — Filament instancing failed";
}

// Checks out well past the old fixed cap of 32 concurrent instances from a single RenderModel and
// verifies they all succeed, proving instances now grow dynamically on demand.
TEST(GlbRenderTest, DynamicInstancingExceedsLegacyCap) {
  SKIP_IF_NO_GPU();
  std::string glbPath = test::GetTestAssetPath("basic_shapes/Cube.glb");
  ASSERT_TRUE(std::filesystem::exists(glbPath));

  auto renderer = MochiRenderer::Create();
  auto* asset = renderer->GetResourceManager()->LoadRenderModel(glbPath);
  ASSERT_NE(asset, nullptr);

  constexpr int kInstanceCount = 100;
  std::vector<std::unique_ptr<SceneObject>> instances;
  for (int i = 0; i < kInstanceCount; ++i) {
    auto instance = asset->GetInstance();
    ASSERT_NE(instance, nullptr) << "GetInstance returned nullptr at index " << i;
    instances.push_back(std::move(instance));
  }
  EXPECT_EQ(asset->GetInstanceCount(), kInstanceCount);
}

// Releases some checked-out instances and re-acquires them, verifying freed slots are recycled
// rather than growing the pool without bound.
TEST(GlbRenderTest, DynamicInstancingRecyclesFreedSlots) {
  SKIP_IF_NO_GPU();
  std::string glbPath = test::GetTestAssetPath("basic_shapes/Cube.glb");
  ASSERT_TRUE(std::filesystem::exists(glbPath));

  auto renderer = MochiRenderer::Create();
  auto* asset = renderer->GetResourceManager()->LoadRenderModel(glbPath);
  ASSERT_NE(asset, nullptr);

  // Cap the pool at exactly the peak we reach. With growth forbidden past the cap, re-acquiring
  // after a release can only succeed by recycling freed slots. This lets the test distinguish
  // recycling from unbounded growth, which GetInstanceCount() alone cannot (it reports only in-use
  // slots, so it would read the same whether slots were reused or the pool grew with idle slots).
  constexpr int kPoolCap = 10;
  asset->SetMaxInstances(kPoolCap);

  std::vector<std::unique_ptr<SceneObject>> instances;
  for (int i = 0; i < kPoolCap; ++i) {
    instances.push_back(asset->GetInstance());
    ASSERT_NE(instances.back(), nullptr);
  }
  EXPECT_EQ(asset->GetInstanceCount(), kPoolCap);

  // Release half of them; the in-use count drops but the slots remain allocated for reuse.
  constexpr int kReleaseCount = 5;
  for (int i = 0; i < kReleaseCount; ++i) {
    instances.pop_back();
  }
  EXPECT_EQ(asset->GetInstanceCount(), kPoolCap - kReleaseCount);

  // Re-acquire the same number. Because the pool is already at its cap, these can only succeed by
  // recycling the freed slots; if instead the pool tried to grow it would exceed the cap and
  // return nullptr.
  for (int i = 0; i < kReleaseCount; ++i) {
    instances.push_back(asset->GetInstance());
    ASSERT_NE(instances.back(), nullptr) << "freed slot was not recycled at re-acquire " << i;
  }
  EXPECT_EQ(asset->GetInstanceCount(), kPoolCap);
}

// Verifies the soft cap: setting a low max and allocating up to it succeeds, but the next
// allocation returns nullptr.
TEST(GlbRenderTest, DynamicInstancingRespectsSoftCap) {
  SKIP_IF_NO_GPU();
  std::string glbPath = test::GetTestAssetPath("basic_shapes/Cube.glb");
  ASSERT_TRUE(std::filesystem::exists(glbPath));

  auto renderer = MochiRenderer::Create();
  auto* asset = renderer->GetResourceManager()->LoadRenderModel(glbPath);
  ASSERT_NE(asset, nullptr);

  constexpr int kCap = 3;
  asset->SetMaxInstances(kCap);
  EXPECT_EQ(asset->GetMaxInstances(), kCap);

  std::vector<std::unique_ptr<SceneObject>> instances;
  for (int i = 0; i < kCap; ++i) {
    auto instance = asset->GetInstance();
    ASSERT_NE(instance, nullptr) << "GetInstance returned nullptr at index " << i;
    instances.push_back(std::move(instance));
  }
  // The next allocation exceeds the cap and must fail. The cap path logs to LogChannel::Error by
  // design, so suppress it for this scope (the mochi test harness fails on unexpected error logs).
  {
    auto suppressError = mochi::test::SuppressLogError();
    EXPECT_EQ(asset->GetInstance(), nullptr);
  }
  EXPECT_EQ(asset->GetInstanceCount(), kCap);
}

// Builds a single large triangle spanning roughly [-10, 10] on every axis, clearly larger than the
// unit-scale Cube.glb so a geometry change is easy to detect from the instance's bounding box.
static void MakeLargeTriangle(
    std::vector<float>& positions,
    std::vector<float>& normals,
    std::vector<int>& indices) {
  positions = {-10.0f, -10.0f, -10.0f, 10.0f, -10.0f, -10.0f, 0.0f, 10.0f, 10.0f};
  normals = {0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  indices = {0, 1, 2};
}

static bool BoxesNearEqual(filament::Box const& a, filament::Box const& b, float tol) {
  auto axesNear = [tol](filament::math::float3 const& u, filament::math::float3 const& v) {
    return std::abs(u.x - v.x) <= tol && std::abs(u.y - v.y) <= tol && std::abs(u.z - v.z) <= tol;
  };
  return axesNear(a.center, b.center) && axesNear(a.halfExtent, b.halfExtent);
}

// Regression test for the bug where an instance created AFTER UpdateGeometry() rendered the
// original glTF mesh instead of the updated geometry: CreateNewInstance() built from the source
// asset and did not apply the model's owned geometry override, so a later instance kept the
// original (cube) bounds while earlier, re-pointed instances showed the updated (large-triangle)
// bounds.
TEST(GlbRenderTest, DynamicInstancingNewInstanceUsesUpdatedGeometry) {
  SKIP_IF_NO_GPU();
  std::string glbPath = test::GetTestAssetPath("basic_shapes/Cube.glb");
  ASSERT_TRUE(std::filesystem::exists(glbPath));

  auto renderer = MochiRenderer::Create();
  auto* asset = renderer->GetResourceManager()->LoadRenderModel(glbPath);
  ASSERT_NE(asset, nullptr);

  // Instance #1 reuses the initial pooled instance and reflects the original cube geometry. Keep it
  // alive so instance #2 below is forced to grow the pool via CreateNewInstance().
  auto first = asset->GetInstance();
  ASSERT_NE(first, nullptr);
  filament::Box const originalBounds = first->GetAABB();

  // Replace the model's geometry with a mesh whose bounds are clearly larger than the unit cube.
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;
  MakeLargeTriangle(positions, normals, indices);
  asset->UpdateGeometry(
      mochi::MakeConstSpan(positions),
      mochi::MakeConstSpan(normals),
      mochi::MakeConstSpan(indices));

  // Sanity: the update must actually change the existing instance's bounds, otherwise the check
  // below could pass trivially without discriminating stale vs. updated geometry.
  filament::Box const updatedFirstBounds = first->GetAABB();
  ASSERT_FALSE(BoxesNearEqual(updatedFirstBounds, originalBounds, 0.1f))
      << "UpdateGeometry did not change the existing instance's bounds; test cannot discriminate.";
  EXPECT_GT(updatedFirstBounds.halfExtent.x, 5.0f)
      << "Updated bounds should span the large triangle (~10 half-extent), not the unit cube.";

  // Instance #2 is created lazily AFTER the update (forces CreateNewInstance). It must match the
  // updated geometry (the re-pointed instance #1), not revert to the original cube.
  auto second = asset->GetInstance();
  ASSERT_NE(second, nullptr);
  EXPECT_TRUE(BoxesNearEqual(second->GetAABB(), updatedFirstBounds, 0.1f))
      << "Instance created after UpdateGeometry rendered stale (original) geometry.";
}

// The rest pose moved by @p offset, with recomputed normals, as SetVertices takes it.
static void TranslatedMesh(
    DeformableMeshView mesh,
    filament::math::float3 offset,
    std::vector<float>& positions,
    std::vector<float>& normals) {
  positions.assign(mesh.restPositions.begin(), mesh.restPositions.end());
  for (size_t i = 0; i < positions.size(); ++i) {
    positions[i] += offset[i % 3];
  }
  std::vector<int> const indices{mesh.triangleIndices.begin(), mesh.triangleIndices.end()};
  ComputeVertexNormalsAreaWeighted(positions, indices, normals);
}

TEST(DeformableModelTest, InstancesFollowSetVerticesAndCopiesStartAtTheirPose) {
  SKIP_IF_NO_GPU();
  std::string const glbPath = test::GetTestAssetPath("basic_shapes/Cube.glb");
  auto renderer = MochiRenderer::Create();
  RenderModel* const model = renderer->GetResourceManager()->LoadRenderModel(glbPath);
  ASSERT_NE(model, nullptr);
  auto mesh = model->GetDeformableMesh(mochi::test::ExpectOK{});
  ASSERT_TRUE(mesh.has_value());

  std::unique_ptr<DeformableModelInstance> const instance =
      model->GetDeformableInstance(mochi::test::ExpectOK{});
  ASSERT_NE(instance, nullptr);
  EXPECT_EQ(instance->GetVertexCount(), mesh->restPositions.size() / 3);
  filament::Box const rest = instance->GetAABB();

  std::vector<float> positions;
  std::vector<float> normals;
  TranslatedMesh(*mesh, {5.0f, 0.0f, 0.0f}, positions, normals);
  instance->SetVertices(mochi::MakeConstSpan(positions), mochi::MakeConstSpan(normals));
  EXPECT_NEAR(instance->GetAABB().center.x, rest.center.x + 5.0f, 1e-3f);

  // A copy (a highlight stand-in) starts at the deformed pose, then has vertices of its own.
  std::unique_ptr<SceneObject> const copy = instance->GetInstanceable()->GetInstance();
  ASSERT_NE(copy, nullptr);
  DeformableModelInstance* const deformableCopy = copy->AsDeformableModelInstance();
  ASSERT_NE(deformableCopy, nullptr);
  filament::Box const deformed = instance->GetAABB();
  EXPECT_TRUE(BoxesNearEqual(copy->GetAABB(), deformed, 1e-3f));
  TranslatedMesh(*mesh, {0.0f, -3.0f, 0.0f}, positions, normals);
  deformableCopy->SetVertices(mochi::MakeConstSpan(positions), mochi::MakeConstSpan(normals));
  EXPECT_NEAR(copy->GetAABB().center.y, rest.center.y - 3.0f, 1e-3f);
  EXPECT_TRUE(BoxesNearEqual(instance->GetAABB(), deformed, 1e-3f));
}

TEST(DeformableModelTest, LeavesRigidInstancesOfTheSameFileUntouched) {
  SKIP_IF_NO_GPU();
  std::string const glbPath = test::GetTestAssetPath("basic_shapes/Cube.glb");
  auto renderer = MochiRenderer::Create();
  RenderModel* const rigidModel = renderer->GetResourceManager()->LoadRenderModel(glbPath);
  ASSERT_NE(rigidModel, nullptr);
  std::unique_ptr<SceneObject> const rigid = rigidModel->GetInstance();
  filament::Box const rigidBounds = rigid->GetAABB();

  auto mesh = rigidModel->GetDeformableMesh(mochi::test::ExpectOK{});
  ASSERT_TRUE(mesh.has_value());
  std::unique_ptr<DeformableModelInstance> const instance =
      rigidModel->GetDeformableInstance(mochi::test::ExpectOK{});
  ASSERT_NE(instance, nullptr);
  std::vector<float> positions;
  std::vector<float> normals;
  TranslatedMesh(*mesh, {5.0f, 0.0f, 0.0f}, positions, normals);
  instance->SetVertices(mochi::MakeConstSpan(positions), mochi::MakeConstSpan(normals));

  EXPECT_TRUE(BoxesNearEqual(rigid->GetAABB(), rigidBounds, 1e-4f));
  EXPECT_TRUE(BoxesNearEqual(rigidModel->GetInstance()->GetAABB(), rigidBounds, 1e-4f));
}

TEST(DeformableModelTest, RendersDeformedInstance) {
  SKIP_IF_NO_GPU();
  std::string const glbPath = test::GetTestAssetPath("basic_shapes/Cube.glb");
  auto renderer = MochiRenderer::Create();
  RenderModel* const model = renderer->GetResourceManager()->LoadRenderModel(glbPath);
  ASSERT_NE(model, nullptr);
  auto mesh = model->GetDeformableMesh(mochi::test::ExpectOK{});
  ASSERT_TRUE(mesh.has_value());
  std::unique_ptr<DeformableModelInstance> instance =
      model->GetDeformableInstance(mochi::test::ExpectOK{});
  ASSERT_NE(instance, nullptr);
  std::vector<float> positions;
  std::vector<float> normals;
  TranslatedMesh(*mesh, {0.0f, 0.0f, 0.0f}, positions, normals);
  instance->SetVertices(mochi::MakeConstSpan(positions), mochi::MakeConstSpan(normals));
  auto* scene = renderer->GetScene();
  scene->CreateSunlight();
  scene->CreateIndirectLight();
  // The scene owns the instance, which must go before the model does.
  SceneObject* const staged = scene->AddSceneObjectToScene(std::move(instance));
  MOCHI_DEFER(scene->DestroySceneObject(staged));
  auto* cam = renderer->CreateObservationCamera("deformable_cam", 128, 128);
  ASSERT_NE(cam, nullptr);
  cam->LookAt({3.0, 3.0, 3.0}, {0, 0, 0});

  auto results = renderer->RenderAndReadback({"deformable_cam"});
  ASSERT_TRUE(results.contains("deformable_cam"));
  auto centroid = FindColorCentroid(results["deformable_cam"], IsNonBlackPixel);
  ASSERT_TRUE(centroid.has_value()) << "The deformable model was not rendered";
  EXPECT_GT(centroid->count, 10);
}

namespace {

// A GLB file holding a triangle whose primitive has a TEXCOORD_1, which deformation rejects.
mochi::TempFileCleanup WriteTexcoord1TriangleGlb() {
  std::vector<float> const positions = {-1, -1, 0, 1, -1, 0, 0, 1, 0};
  std::vector<float> const texcoords = {0, 0, 1, 0, 0, 1};
  std::vector<uint8_t> bin((positions.size() + texcoords.size()) * sizeof(float));
  std::memcpy(bin.data(), positions.data(), positions.size() * sizeof(float));
  std::memcpy(
      bin.data() + positions.size() * sizeof(float),
      texcoords.data(),
      texcoords.size() * sizeof(float));
  std::vector<uint8_t> const glb = test::AssembleGlb(
      R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],)"
      R"("meshes":[{"primitives":[{"attributes":{"POSITION":0,"TEXCOORD_1":1}}]}],)"
      R"("accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3",)"
      R"("min":[-1,-1,0],"max":[1,1,0]},)"
      R"({"bufferView":1,"componentType":5126,"count":3,"type":"VEC2"}],)"
      R"("bufferViews":[{"buffer":0,"byteLength":36},)"
      R"({"buffer":0,"byteOffset":36,"byteLength":24}],"buffers":[{"byteLength":60}]})",
      std::move(bin));
  mochi::TempFileCleanup file =
      mochi::CreateTempFile("deformable", ".glb", mochi::test::ExpectOK{});
  std::ofstream(file.Path(), std::ios::binary)
      .write(reinterpret_cast<char const*>(glb.data()), static_cast<std::streamsize>(glb.size()));
  return file;
}

} // namespace

TEST(DeformableModelTest, UnsupportedDeformationStillLoadsRigidly) {
  SKIP_IF_NO_GPU();
  mochi::TempFileCleanup const file = WriteTexcoord1TriangleGlb();
  auto renderer = MochiRenderer::Create();
  RenderModel* const model = renderer->GetResourceManager()->LoadRenderModel(file.Path().string());
  ASSERT_NE(model, nullptr);
  EXPECT_NE(model->GetInstance(), nullptr);
  mochi::Error error;
  EXPECT_EQ(model->GetDeformableInstance(error), nullptr);
  EXPECT_STREQ(
      error.GetDescription(),
      "The GLB's primitive has a texture-coordinate set other than TEXCOORD_0.");
}

TEST(GlbRenderTest, RenderGlbProducesPixels) {
  SKIP_IF_NO_GPU();
  std::string glbPath = test::GetTestAssetPath("basic_shapes/Cube.glb");
  ASSERT_TRUE(std::filesystem::exists(glbPath));

  auto renderer = MochiRenderer::Create();
  ASSERT_NE(renderer, nullptr);

  // Load GLB asset.
  auto* asset = renderer->GetResourceManager()->LoadRenderModel(glbPath);
  ASSERT_NE(asset, nullptr);

  // Add the mesh to the scene.
  auto instance = asset->GetInstance();
  ASSERT_NE(instance, nullptr);
  auto* scene = renderer->GetScene();
  scene->CreateSunlight();
  scene->CreateIndirectLight();
  scene->AddSceneObjectToScene(std::move(instance));

  // Set up a camera looking at the origin.
  auto* cam = renderer->CreateObservationCamera("glb_cam", 128, 128);
  ASSERT_NE(cam, nullptr);
  cam->LookAt({3.0, 3.0, 3.0}, {0, 0, 0});

  // Render and verify we got non-empty output.
  auto results = renderer->RenderAndReadback({"glb_cam"});
  ASSERT_TRUE(results.contains("glb_cam"));
  auto& result = results["glb_cam"];
  ASSERT_TRUE(result.IsValid());
  EXPECT_EQ(result.width, 128);
  EXPECT_EQ(result.height, 128);
  EXPECT_EQ(result.pixels.size(), 128u * 128u * 4u);

  // The cube should produce some non-black pixels.
  auto centroid = FindColorCentroid(result, IsNonBlackPixel);
  EXPECT_TRUE(centroid.has_value()) << "No non-black pixels — GLB mesh was not rendered";
  if (centroid.has_value()) {
    EXPECT_GT(centroid->count, 10) << "Too few non-black pixels, mesh may not have rendered";
  }
}

// --- STL Import + Render Tests ---

TEST(StlRenderTest, LoadStlAsset) {
  SKIP_IF_NO_GPU();
  std::string stlPath = test::GetTestAssetPath("basic_shapes/binary_cube.stl");
  ASSERT_TRUE(std::filesystem::exists(stlPath)) << "STL file not found: " << stlPath;

  auto renderer = MochiRenderer::Create();
  ASSERT_NE(renderer, nullptr);

  auto* resourceManager = renderer->GetResourceManager();
  ASSERT_NE(resourceManager, nullptr);

  auto* asset = resourceManager->LoadResource(stlPath);
  ASSERT_NE(asset, nullptr) << "LoadAsset returned nullptr for binary_cube.stl";
  EXPECT_EQ(asset->GetType(), ResourceType::RenderModel);

  auto* renderAsset = dynamic_cast<RenderModel*>(asset);
  ASSERT_NE(renderAsset, nullptr);
  EXPECT_EQ(renderAsset->GetOriginalFormat(), RenderModelFormat::Stl);
}

TEST(StlRenderTest, LoadStlAndGetInstance) {
  SKIP_IF_NO_GPU();
  std::string stlPath = test::GetTestAssetPath("basic_shapes/binary_cube.stl");
  ASSERT_TRUE(std::filesystem::exists(stlPath));

  auto renderer = MochiRenderer::Create();
  auto* asset = renderer->GetResourceManager()->LoadRenderModel(stlPath);
  ASSERT_NE(asset, nullptr);

  auto instance = asset->GetInstance();
  ASSERT_NE(instance, nullptr) << "GetInstance returned nullptr — Filament instancing failed";
}

TEST(StlRenderTest, RenderStlProducesPixels) {
  SKIP_IF_NO_GPU();
  std::string stlPath = test::GetTestAssetPath("basic_shapes/binary_cube.stl");
  ASSERT_TRUE(std::filesystem::exists(stlPath));

  auto renderer = MochiRenderer::Create();
  ASSERT_NE(renderer, nullptr);

  // Load STL asset.
  auto* asset = renderer->GetResourceManager()->LoadRenderModel(stlPath);
  ASSERT_NE(asset, nullptr);

  // Add the mesh to the scene.
  auto instance = asset->GetInstance();
  ASSERT_NE(instance, nullptr);
  auto* scene = renderer->GetScene();
  scene->CreateSunlight();
  scene->CreateIndirectLight();
  scene->AddSceneObjectToScene(std::move(instance));

  // Set up a camera looking at the origin. The binary_cube.stl has vertices
  // in the range [-10, 10], so the camera needs to be far enough to see it.
  auto* cam = renderer->CreateObservationCamera("stl_cam", 128, 128);
  ASSERT_NE(cam, nullptr);
  cam->LookAt({30.0, 30.0, 30.0}, {0, 0, 0});

  // Render and verify we got non-empty output.
  auto results = renderer->RenderAndReadback({"stl_cam"});
  ASSERT_TRUE(results.contains("stl_cam"));
  auto& result = results["stl_cam"];
  ASSERT_TRUE(result.IsValid());
  EXPECT_EQ(result.width, 128);
  EXPECT_EQ(result.height, 128);
  EXPECT_EQ(result.pixels.size(), 128u * 128u * 4u);

  // The cube should produce some non-black pixels (the default STL material is pink).
  auto centroid = FindColorCentroid(result, IsNonBlackPixel);
  EXPECT_TRUE(centroid.has_value()) << "No non-black pixels — STL mesh was not rendered";
  if (centroid.has_value()) {
    EXPECT_GT(centroid->count, 10) << "Too few non-black pixels, mesh may not have rendered";
  }
}
