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
#include <mochi_core/geometry/mesh_data_utils.h>
#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/src/mochi_context.h>
#include <mochi_physics/src/mochi_rigid.h>
#include <mochi_physics/src/mochi_shape.h>
#include <mochi_physics/src/mochi_soft.h>
#include <mochi_physics/src/mochi_soft_rom_systems.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <ios>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace mochi;
using namespace mochi::test;

#if MOCHI_ENABLE_ROM_ACTORS
#define MOCHI_ROM_DISABLED 0
#else
#define MOCHI_ROM_DISABLED 1
#endif

#if MOCHI_ENABLE_DEEP_FLOW_ACTORS
#define MOCHI_DEEP_FLOW_DISABLED 0
#else
#define MOCHI_DEEP_FLOW_DISABLED 1
#endif

// Test fixture used to group tests in this file.
class MochiContextTest : public MochiContextTestWithParam {
 protected:
  void TestLoadShapeFromFiles(std::initializer_list<std::string_view> paths);
  void TestFileCacheConcurrency(std::initializer_list<std::string_view> paths);
  void TestLoadShapeFromFileConcurrent(std::initializer_list<std::string_view> paths);
};

// Various parameter cominations
static constexpr MochiContextTestParams kThreadingParams[] = {
    {0, false}, // No worker threads. Implicitly single-threaded.
    {2, true}, // Has worker threads, but forces single-threaded mode anyway.
    {4, false} // Has worker threads and uses them
};

// Repeat all test in this file multiple times with different parameters
MOCHI_INSTANTIATE_TEST_SUITE_P(
    VariousParams,
    MochiContextTest,
    ::testing::ValuesIn(kThreadingParams));

// Expect that the test shape is the same as the reference shape, but with a baked-in transform.
static void ExpectTransformedMesh(
    Context const* context,
    ShapeHandle referenceShape,
    ShapeHandle testShape,
    VMatrix4x4r transform) {
  EXPECT_TRUE(referenceShape.IsValid());
  EXPECT_TRUE(testShape.IsValid());
  auto const& refMesh = context->GetShapeMesh(referenceShape, ExpectOK{});
  auto const& testMesh = context->GetShapeMesh(testShape, ExpectOK{});

  // Compare node positions
  EXPECT_LT(0, refMesh.GetNumNodes());
  ASSERT_EQ(refMesh.GetNumNodes(), testMesh.GetNumNodes());
  for (int i = 0; i < refMesh.GetNumNodes(); ++i) {
    auto refPos = Vec4r{
        refMesh.coordinates[i * 3 + 0],
        refMesh.coordinates[i * 3 + 1],
        refMesh.coordinates[i * 3 + 2],
        1_r};
    auto expectedPos = DotMatVec4x4(transform, refPos);
    auto actualPos = Vec4r{
        testMesh.coordinates[i * 3 + 0],
        testMesh.coordinates[i * 3 + 1],
        testMesh.coordinates[i * 3 + 2],
        1_r};
    EXPECT_NEAR_EQ(expectedPos, actualPos);
  }
}

// Call LoadShapeFromFile or LoadShapeFromBytes (if shouldPreload).
static ShapeHandle TestLoadShapeFullPath(
    Context* context,
    std::string_view fullPath,
    Real3 const& scale,
    TransformRT const& rt,
    bool shouldPreload,
    bool useModelData,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  if (shouldPreload) {
    auto fileBytes = ReadFileBytes(fullPath, error);
    if (useModelData) {
      ModelData model = model::LoadFromBytes(fileBytes, error);
      model::BakeTransform(model, scale, rt, error);
      return context->CreateModelShape(model, error);
    } else {
      return context->LoadShapeFromBytes(fileBytes, scale, rt, error);
    }
  } else {
    if (useModelData) {
      ModelData model = model::LoadFromFile(fullPath, error);
      model::BakeTransform(model, scale, rt, error);
      return context->CreateModelShape(model, error);
    } else {
      return context->LoadShapeFromFile(fullPath, scale, rt, error);
    }
  }
}

static ShapeHandle TestLoadShape(
    Context* context,
    std::string_view assetPath,
    Real3 const& scale,
    TransformRT const& rt,
    bool shouldPreload,
    bool useModelData,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  return TestLoadShapeFullPath(
      context, test::GetAssetPath(assetPath), scale, rt, shouldPreload, useModelData, error);
}

void MochiContextTest::TestLoadShapeFromFiles(std::initializer_list<std::string_view> paths) {
  for (auto const path : paths) {
    // Skip .h5 files if not supported
    if constexpr (!MOCHI_USE_HDF5) {
      if (path.ends_with(".h5")) {
        continue;
      }
    }
    for (bool shouldPreload : {false, true}) {
      for (bool useModelData : {false, true}) {
        // Load with with identity transform
        ShapeHandle refShape = TestLoadShape(
            _mochiContext, path, Real3{1_r, 1_r, 1_r}, {}, shouldPreload, useModelData, ExpectOK{});
        ASSERT_TRUE(refShape.IsValid());
        MOCHI_DEFER(_mochiContext->ReleaseShape(refShape));
        EXPECT_EQ(1, _mochiContext->GetNumShapes());

        // Some arbitrary non-default values
        auto const scale = Real3{0.1_r, 0.2_r, 0.3_r};
        auto const rotation = Quaternion::FromRotationVector(Real3{1.1_r, 2.2_r, 3.3_r});
        auto const translation = Real3{-0.2_r, 0.4_r, -0.5_r};
        auto const rt = TransformRT{rotation, translation};
        VMatrix4x4r sMat = VDiagonalMatrix<4>(ToSimd(scale, 1_r));
        VMatrix4x4r srtMat = Dot4x4(ToVMatrix4x4(rt), sMat);

        // Load the shape again. This time bake in some scale. Expect a transformed mesh.
        {
          ShapeHandle shape = TestLoadShape(
              _mochiContext, path, scale, {}, shouldPreload, useModelData, ExpectOK{});
          EXPECT_EQ(2, _mochiContext->GetNumShapes());
          ExpectTransformedMesh(_mochiContext, refShape, shape, sMat);
          _mochiContext->ReleaseShape(shape);
          EXPECT_EQ(1, _mochiContext->GetNumShapes());
        }

        // Load again with scale, rotation, and translation. Expect a transformed mesh.
        {
          ShapeHandle shape = TestLoadShape(
              _mochiContext, path, scale, rt, shouldPreload, useModelData, ExpectOK{});
          EXPECT_EQ(2, _mochiContext->GetNumShapes());
          ExpectTransformedMesh(_mochiContext, refShape, shape, srtMat);
          _mochiContext->ReleaseShape(shape);
          EXPECT_EQ(1, _mochiContext->GetNumShapes());
        }

        // Load again with negative scale (mirroring). Expect a transformed mesh.
        {
          auto mirrorScale = Real3{-1_r, 1_r, 1_r};
          auto mirrorMat = VDiagonalMatrix<4>(ToSimd(mirrorScale, 1_r));
          ShapeHandle shape = TestLoadShape(
              _mochiContext, path, mirrorScale, {}, shouldPreload, useModelData, ExpectOK{});
          EXPECT_EQ(2, _mochiContext->GetNumShapes());
          ExpectTransformedMesh(_mochiContext, refShape, shape, mirrorMat);
          _mochiContext->ReleaseShape(shape);
          EXPECT_EQ(1, _mochiContext->GetNumShapes());
        }
      }
    }
  }
}

TEST_P(MochiContextTest, LoadShapeFromFile) {
  TestLoadShapeFromFiles({
      "cube/cube_mesh.mochi.json",
      "cube/cube_mesh.mochi.h5",
  // This mesh is not shipped externally.
#if MOCHI_INTERNAL
      "cube/cube_minimal.mochi.json",
#endif
  });
}

// The cache test mesh is not shipped externally.
TEST_IF_P(MOCHI_INTERNAL, MochiContextTest, FileCache) {
  // Disabled by default
  EXPECT_FALSE(_mochiContext->IsFileCacheEnabled());
  _mochiContext->EnableFileCache(false);
  EXPECT_FALSE(_mochiContext->IsFileCacheEnabled());

  // Enable it
  _mochiContext->EnableFileCache(true);
  EXPECT_TRUE(_mochiContext->IsFileCacheEnabled());

  // Call LoadShapeFromFile twice with the same path. We should get different shape handles.
  auto path = test::GetAssetPath("cube/cube_minimal.mochi.json");
  auto shapeHandle1 = _mochiContext->LoadShapeFromFile(path, test::ExpectOK{});
  auto shapeHandle2 = _mochiContext->LoadShapeFromFile(path, test::ExpectOK{});
  EXPECT_TRUE(shapeHandle1.IsValid());
  EXPECT_TRUE(shapeHandle2.IsValid());
  EXPECT_NE(shapeHandle1, shapeHandle2); // Different handle values.
  EXPECT_EQ(2, _mochiContext->GetNumShapes());

  // Access the ContextImpl class to get the address of the loaded Shape objects.
  // Both handles should refer to the same address.
  auto const* contextImpl = assert_cast<ContextImpl const*>(_mochiContext);
  auto shapePtr1 = contextImpl->GetShapeSharedPtr(shapeHandle1);
  auto shapePtr2 = contextImpl->GetShapeSharedPtr(shapeHandle2);
  auto const nullShape = std::shared_ptr<Shape const>{};
  EXPECT_NE(nullShape, shapePtr1);
  EXPECT_NE(std::shared_ptr<Shape const>{}, shapePtr2);
  EXPECT_EQ(shapePtr1, shapePtr2); // Same address, thanks to the cache

  // Release the first shape handle. Redundant attempts should do nothing.
  _mochiContext->ReleaseShape(shapeHandle1);
  _mochiContext->ReleaseShape(shapeHandle1);
  _mochiContext->ReleaseShape(shapeHandle1);
  EXPECT_EQ(1, _mochiContext->GetNumShapes()); // One valid ShapeHandle

  // The first handle should be unmapped but the second should be unchanged.
  EXPECT_EQ(nullShape, contextImpl->GetShapeSharedPtr(shapeHandle1));
  EXPECT_EQ(shapePtr2, contextImpl->GetShapeSharedPtr(shapeHandle2));

  // Release the 2nd handle
  _mochiContext->ReleaseShape(shapeHandle2);
  EXPECT_EQ(0, _mochiContext->GetNumShapes()); // No valid ShapeHandles
  EXPECT_EQ(nullShape, contextImpl->GetShapeSharedPtr(shapeHandle1));
  EXPECT_EQ(nullShape, contextImpl->GetShapeSharedPtr(shapeHandle2));

  // Reload the same path as before
  auto shapeHandle1b = _mochiContext->LoadShapeFromFile(path, test::ExpectOK{});
  auto shapeHandle2b = _mochiContext->LoadShapeFromFile(path, test::ExpectOK{});
  EXPECT_TRUE(shapeHandle1b.IsValid());
  EXPECT_TRUE(shapeHandle2b.IsValid());
  EXPECT_NE(shapeHandle1, shapeHandle1b); // Handle not reused
  EXPECT_NE(shapeHandle2, shapeHandle1b); // Handle not reused
  EXPECT_NE(shapeHandle1, shapeHandle2b); // Handle not reused
  EXPECT_NE(shapeHandle2, shapeHandle2b); // Handle not reused
  EXPECT_NE(shapeHandle1b, shapeHandle2b); // Not the same
  EXPECT_EQ(2, _mochiContext->GetNumShapes());

  // The loaded address should be the same as before, thanks to the file cache
  EXPECT_EQ(shapePtr1, contextImpl->GetShapeSharedPtr(shapeHandle1b));
  EXPECT_EQ(shapePtr1, contextImpl->GetShapeSharedPtr(shapeHandle2b));

  // Load several more times, but use different scales or rotations or translations.
  for (int i = 0; i < 3 + 4 + 3; ++i) {
    Real3 scale{1_r, 1_r, 1_r};
    Quaternion rotation{};
    Real3 translation{};
    real constexpr kNudge = std::numeric_limits<real>::epsilon();

    // Modfy one value per loop iteration.
    if (i < 3) {
      scale[i] += kNudge;
    } else if (i < 7) {
      auto idx = i - 3;
      auto val = rotation.data[idx] + kNudge;
      rotation.data = Set(rotation.data, idx, val);
    } else {
      translation[i - 7] += kNudge;
    }

    // Load with baked in scale, rotation, and translation.
    auto transformedHandle1 = _mochiContext->LoadShapeFromFile(
        path, scale, TransformRT{rotation, translation}, test::ExpectOK{});
    auto transformedHandle2 = _mochiContext->LoadShapeFromFile(
        path, scale, TransformRT{rotation, translation}, test::ExpectOK{});
    EXPECT_NE(transformedHandle1, transformedHandle2);
    auto const& transformedPtr1 = contextImpl->GetShapeSharedPtr(transformedHandle1);
    auto const& transformedPtr2 = contextImpl->GetShapeSharedPtr(transformedHandle2);
    EXPECT_EQ(transformedPtr1, transformedPtr2); // Same transform --> same address
    EXPECT_NE(shapePtr1, transformedPtr1); // Different transform --> different address

    // Cleanup
    EXPECT_EQ(4, _mochiContext->GetNumShapes());
    _mochiContext->ReleaseShape(transformedHandle1);
    _mochiContext->ReleaseShape(transformedHandle2);
    EXPECT_EQ(2, _mochiContext->GetNumShapes());
  }

  // Try to load an invalid path. The second attempt should repeat the cached error (no visible side
  // effects here)
  auto badHandle1 =
      _mochiContext->LoadShapeFromFile("/no/such/file.mochi.json", test::ExpectNotOK{});
  auto badHandle2 =
      _mochiContext->LoadShapeFromFile("/no/such/file.mochi.json", test::ExpectNotOK{});
  EXPECT_FALSE(badHandle1.IsValid());
  EXPECT_FALSE(badHandle2.IsValid());
  EXPECT_EQ(2, _mochiContext->GetNumShapes()); // no change

  // Disable and re-enable the cache. This has the result of clearing it.
  _mochiContext->EnableFileCache(false);
  _mochiContext->EnableFileCache(true);
  auto handleAfterReenable = _mochiContext->LoadShapeFromFile(path, test::ExpectOK{});
  EXPECT_NE(nullShape, contextImpl->GetShapeSharedPtr(handleAfterReenable));
  EXPECT_NE(shapePtr1, contextImpl->GetShapeSharedPtr(handleAfterReenable)); // New address

  // Cleanup
  _mochiContext->ReleaseShape(shapeHandle1b);
  _mochiContext->ReleaseShape(shapeHandle2b);
  _mochiContext->ReleaseShape(handleAfterReenable);
  EXPECT_EQ(0, _mochiContext->GetNumShapes());
}

// The cache test meshes are not shipped externally.
TEST_IF_P(MOCHI_INTERNAL, MochiContextTest, ClearFileCache) {
  _mochiContext->EnableFileCache(true);

  // Load some stuff
  auto path1 = test::GetAssetPath("cube/cube_minimal.mochi.json");
  auto path2 = test::GetAssetPath("cube/cube_mesh.mochi.json");
  auto shapeHandle1a = _mochiContext->LoadShapeFromFile(path1, test::ExpectOK{});
  auto shapeHandle1b = _mochiContext->LoadShapeFromFile(path1, test::ExpectOK{});
  auto shapeHandle2a = _mochiContext->LoadShapeFromFile(path2, test::ExpectOK{});
  auto shapeHandle2b = _mochiContext->LoadShapeFromFile(path2, test::ExpectOK{});

  // Shapes with the same file path should share memory
  auto const* contextImpl = assert_cast<ContextImpl const*>(_mochiContext);
  auto const nullShape = std::shared_ptr<Shape const>{};
  auto shapePtr1a = contextImpl->GetShapeSharedPtr(shapeHandle1a);
  auto shapePtr1b = contextImpl->GetShapeSharedPtr(shapeHandle1b);
  auto shapePtr2a = contextImpl->GetShapeSharedPtr(shapeHandle2a);
  auto shapePtr2b = contextImpl->GetShapeSharedPtr(shapeHandle2b);
  EXPECT_NE(nullShape, shapePtr1a);
  EXPECT_NE(nullShape, shapePtr2a);
  EXPECT_EQ(shapePtr1a, shapePtr1b); // Same
  EXPECT_EQ(shapePtr2a, shapePtr2b); // Same
  EXPECT_NE(shapePtr1a, shapePtr2a); // Different

  // Clear the cache
  _mochiContext->ClearFileCache();

  // Request each path again. This time, they should be loaded to new addresses
  auto shapeHandle1c = _mochiContext->LoadShapeFromFile(path1, test::ExpectOK{});
  auto shapeHandle2c = _mochiContext->LoadShapeFromFile(path2, test::ExpectOK{});
  auto shapePtr1c = contextImpl->GetShapeSharedPtr(shapeHandle1c);
  auto shapePtr2c = contextImpl->GetShapeSharedPtr(shapeHandle2c);
  EXPECT_NE(nullShape, shapePtr1c);
  EXPECT_NE(nullShape, shapePtr2c);
  EXPECT_NE(shapePtr1c, shapePtr2c); // Different
  EXPECT_NE(shapePtr1a, shapePtr1c); // Different
  EXPECT_NE(shapePtr2a, shapePtr2c); // Different

  // But if we do it again (without clearing), then we should get cached addresses again.
  auto shapeHandle1d = _mochiContext->LoadShapeFromFile(path1, test::ExpectOK{});
  auto shapeHandle2d = _mochiContext->LoadShapeFromFile(path2, test::ExpectOK{});
  auto shapePtr1d = contextImpl->GetShapeSharedPtr(shapeHandle1d);
  auto shapePtr2d = contextImpl->GetShapeSharedPtr(shapeHandle2d);
  EXPECT_EQ(shapePtr1c, shapePtr1d); // Same
  EXPECT_EQ(shapePtr2c, shapePtr2d); // Same
}

// The cache test meshes are not shipped externally.
TEST_IF_P(MOCHI_INTERNAL, MochiContextTest, ClearFileFromCache) {
  _mochiContext->EnableFileCache(true);

  // Load some stuff.
  auto path1 = test::GetAssetPath("cube/cube_minimal.mochi.json");
  auto path2 = test::GetAssetPath("cube/cube_mesh.mochi.json");
  auto shapeHandle1a = _mochiContext->LoadShapeFromFile(
      path1, Real3{2_r, 1_r, 1_r}, TransformRT{}, test::ExpectOK{});
  auto shapeHandle1b = _mochiContext->LoadShapeFromFile(
      path1, Real3{3_r, 1_r, 1_r}, TransformRT{}, test::ExpectOK{});
  auto shapeHandle1c = _mochiContext->LoadShapeFromFile(
      path1, Real3{3_r, 1_r, 1_r}, TransformRT{}, test::ExpectOK{});
  auto shapeHandle2a = _mochiContext->LoadShapeFromFile(path2, test::ExpectOK{});
  auto shapeHandle2b = _mochiContext->LoadShapeFromFile(path2, test::ExpectOK{});

  // Shapes with the same file path and scale should share memory
  auto const* contextImpl = assert_cast<ContextImpl const*>(_mochiContext);
  auto const nullShape = std::shared_ptr<Shape const>{};
  auto shapePtr1a = contextImpl->GetShapeSharedPtr(shapeHandle1a);
  auto shapePtr1b = contextImpl->GetShapeSharedPtr(shapeHandle1b);
  auto shapePtr1c = contextImpl->GetShapeSharedPtr(shapeHandle1c);
  auto shapePtr2a = contextImpl->GetShapeSharedPtr(shapeHandle2a);
  auto shapePtr2b = contextImpl->GetShapeSharedPtr(shapeHandle2b);
  EXPECT_NE(nullShape, shapePtr1a);
  EXPECT_NE(nullShape, shapePtr1b);
  EXPECT_NE(nullShape, shapePtr2a);
  EXPECT_NE(shapePtr1a, shapePtr1b); // Same path. Different scale.
  EXPECT_NE(shapePtr1a, shapePtr1c); // Same path. Different scale.
  EXPECT_EQ(shapePtr1b, shapePtr1b); // Same path. Same scale.
  EXPECT_EQ(shapePtr2a, shapePtr2b); // Same path. Same scale.
  EXPECT_NE(shapePtr1a, shapePtr2a); // Different paths

  // Clear all cache entries for path1 (multiple baked scales)
  _mochiContext->ClearFileFromCache(path1);

  // Request each path again. Requests for path1 should get new addresses.
  auto shapeHandle1d = _mochiContext->LoadShapeFromFile(
      path1, Real3{2_r, 1_r, 1_r}, TransformRT{}, test::ExpectOK{});
  auto shapeHandle1e = _mochiContext->LoadShapeFromFile(
      path1, Real3{3_r, 1_r, 1_r}, TransformRT{}, test::ExpectOK{});
  auto shapeHandle1f = _mochiContext->LoadShapeFromFile(
      path1, Real3{3_r, 1_r, 1_r}, TransformRT{}, test::ExpectOK{});
  auto shapeHandle2d = _mochiContext->LoadShapeFromFile(path2, test::ExpectOK{});
  auto shapePtr1d = contextImpl->GetShapeSharedPtr(shapeHandle1d);
  auto shapePtr1e = contextImpl->GetShapeSharedPtr(shapeHandle1e);
  auto shapePtr1f = contextImpl->GetShapeSharedPtr(shapeHandle1f);
  auto shapePtr2d = contextImpl->GetShapeSharedPtr(shapeHandle2d);
  EXPECT_NE(nullShape, shapePtr1d);
  EXPECT_NE(nullShape, shapePtr1e);
  EXPECT_NE(nullShape, shapePtr1f);
  EXPECT_NE(nullShape, shapePtr2d);
  EXPECT_NE(shapePtr1a, shapePtr2d); // Same path and scale. New address.
  EXPECT_NE(shapePtr1b, shapePtr1e); // Same path and scale. New address.
  EXPECT_EQ(shapePtr1e, shapePtr1f); // Same path and scale. Same address (both loaded after clear)
  EXPECT_EQ(shapePtr2a, shapePtr2d); // Same path and scale. Same address (from cache)
}

void MochiContextTest::TestFileCacheConcurrency(std::initializer_list<std::string_view> paths) {
  // Multiple threads can attempt to load shapes while the cache is enabled. Threads should be able
  // to load different files concurrently, but if they request the same file, then only one of the
  // threads should actually load it. That requires some tricky thread synchronization. Therefore,
  // this test runs several threads that each load several shapes, several times, and the whole
  // process restarts from a clean cache several times more.
  _mochiContext->EnableFileCache(true);

  // These must be JSON files. HDF5 is not internally thread-safe, so there is a mutex that causes
  // all async HDF5 file requests to be serialized. That would greatly reduce the value of this
  // test.
  std::vector<std::string> filePaths;
  filePaths.reserve(paths.size());
  for (auto const path : paths) {
    filePaths.push_back(test::GetAssetPath(path));
  }

  int const kNumThreads = 4;
  int const kNumBatches = MOCHI_DEBUG ? 50 : 250;
  int const kNumLoadsPerFile = 5;
  int batchIdx = -1;
  auto const* contextImpl = assert_cast<ContextImpl*>(_mochiContext);
  using ResultPair = std::pair<std::string_view, ShapeHandle>;
  std::mutex mutex;
  std::condition_variable cv;
  TaskSemaphore ready(kNumThreads);
  std::vector<ResultPair> allResults;
  std::vector<std::thread> threads;

  // Worker thread procedure
  auto threadProc = [&]() {
    ready.Done();
    int lastLocalBatchIdx = -1;
    for (;;) {
      {
        // Wait for the next batch to be signaled
        std::unique_lock lock(mutex);
        cv.wait(lock, [&]() { return batchIdx > lastLocalBatchIdx; });
        lastLocalBatchIdx = batchIdx;
      }

      // Run a batch of loads
      std::vector<ResultPair> localResults;
      localResults.reserve(kNumLoadsPerFile * isize(filePaths));
      for (int i = 0; i < kNumLoadsPerFile; ++i) {
        for (auto const& path : filePaths) {
          auto handle = _mochiContext->LoadShapeFromFile(path, test::ExpectOK{});
          localResults.emplace_back(std::string_view(path), handle);
        }
      }

      // Share our results at the end
      {
        std::lock_guard lock(mutex);
        allResults.insert(allResults.end(), localResults.begin(), localResults.end());
      }

      // Signal the main thread.
      ready.Done();

      if (lastLocalBatchIdx == kNumBatches - 1) {
        // Done
        return;
      }
    }
  };

  // Start each thread
  threads.reserve(kNumThreads);
  for (int threadIdx = 0; threadIdx < kNumThreads; ++threadIdx) {
    threads.emplace_back(threadProc);
  }

  // Wait until all threads are ready
  ready.Wait();
  ready.Add(kNumThreads); // Reset

  // Repeat for several batches
  for (int i = 0; i < kNumBatches; ++i) {
    // Signal threads to start a batch of work
    {
      std::lock_guard lock(mutex);
      ++batchIdx;
    }
    cv.notify_all();

    // Wait for threads to finish a batch of work
    ready.Wait();
    ready.Add(kNumThreads); // Reset

    // Check the combined results from all threads.
    EXPECT_EQ(kNumThreads * kNumLoadsPerFile * isize(filePaths), isize(allResults));
    std::unordered_set<ShapeHandle> seenIt;
    std::unordered_map<std::string, std::shared_ptr<Shape const>> ptrMap;
    for (auto&& [path, handle] : allResults) {
      EXPECT_TRUE(seenIt.insert(handle).second); // Expect unique handles
      auto ptr = contextImpl->GetShapeSharedPtr(handle);
      EXPECT_NE(std::shared_ptr<Shape const>{}, ptr);
      auto& ptrRef = ptrMap[std::string{path}]; // Find or insert
      if (ptrRef) {
        EXPECT_EQ(ptrRef, ptr); // Same path -> same address
      } else {
        ptrRef = ptr; // First ptr for this path
      }
    }

    // Cleanup
    for (auto&& [path, handle] : allResults) {
      _mochiContext->ReleaseShape(handle);
    }
    allResults.clear();

    // Clear the cache before the next batch of loads
    _mochiContext->ClearFileCache();
  }

  // Wait for completion
  for (auto& t : threads) {
    t.join();
  }
}

TEST_P(MochiContextTest, FileCacheConcurrency) {
  TestFileCacheConcurrency({
      "cube/cube_mesh.mochi.json",
      "sphere/icosphere_3subdiv.1.mochi.json",
  // These meshes are not shipped externally.
#if MOCHI_INTERNAL
      "cube/cube_minimal.mochi.json",
      "duck/duck_coarse_mesh.mochi.json",
#endif
  });
}

TEST_P(MochiContextTest, LoadShapeFromFile_BadPath) {
  // Fail to load an invalid path
  EXPECT_EQ(ShapeHandle{}, _mochiContext->LoadShapeFromFile("no such path", ExpectNotOK{}));
  EXPECT_EQ(
      ShapeHandle{},
      _mochiContext->LoadShapeFromFile(
          "no such path", Real3{1_r, 1_r, 1_r}, TransformRT::Identity(), ExpectNotOK{}));
  EXPECT_EQ(
      ShapeHandle{},
      _mochiContext->LoadShapeFromFile(
          "no such path", Real3{1_r, 1_r, 1_r}, TransformRT::Identity(), ExpectNotOK{}));
}

TEST_P(MochiContextTest, LoadShapeFromFile_BadTransform) {
  constexpr std::string_view kPath = "cube/cube_mesh.mochi.json";
  constexpr real kSmall = std::numeric_limits<real>::min();
  for (bool shouldPreload : {false, true}) {
    for (bool useModelData : {false, true}) {
      // Load a valid path to prove we can
      {
        auto shape = TestLoadShape(
            _mochiContext,
            kPath,
            Real3{1_r, 1_r, 1_r},
            {},
            shouldPreload,
            useModelData,
            ExpectOK{});
        EXPECT_TRUE(shape.IsValid());
        _mochiContext->ReleaseShape(shape);
      }

      // Try to load with scale of zero
      TestLoadShape(
          _mochiContext,
          kPath,
          Real3{0_r, 1_r, 1_r},
          {},
          shouldPreload,
          useModelData,
          ExpectNotOK{});
      TestLoadShape(
          _mochiContext,
          kPath,
          Real3{1_r, 0_r, 1_r},
          {},
          shouldPreload,
          useModelData,
          ExpectNotOK{});
      TestLoadShape(
          _mochiContext,
          kPath,
          Real3{1_r, 1_r, 0_r},
          {},
          shouldPreload,
          useModelData,
          ExpectNotOK{});

      // Try to load with scale near zero
      TestLoadShape(
          _mochiContext,
          kPath,
          Real3{kSmall, 1_r, 1_r},
          {},
          shouldPreload,
          useModelData,
          ExpectNotOK{});
      TestLoadShape(
          _mochiContext,
          kPath,
          Real3{1_r, kSmall, 1_r},
          {},
          shouldPreload,
          useModelData,
          ExpectNotOK{});
      TestLoadShape(
          _mochiContext,
          kPath,
          Real3{1_r, 1_r, kSmall},
          {},
          shouldPreload,
          useModelData,
          ExpectNotOK{});

      // Try to load with non-finite scale
      TestLoadShape(
          _mochiContext,
          kPath,
          Real3{kInf, 1_r, 1_r},
          {},
          shouldPreload,
          useModelData,
          ExpectNotOK{});
      TestLoadShape(
          _mochiContext,
          kPath,
          Real3{1_r, kInf, 1_r},
          {},
          shouldPreload,
          useModelData,
          ExpectNotOK{});
      TestLoadShape(
          _mochiContext,
          kPath,
          Real3{1_r, 1_r, kInf},
          {},
          shouldPreload,
          useModelData,
          ExpectNotOK{});

      // Try to load with non-finite TransformRT
      TestLoadShape(
          _mochiContext,
          kPath,
          Real3{1_r, 1_r, 1_r},
          TransformRT(Quaternion{}, Real3{0_r, kInf, 0_r}),
          shouldPreload,
          useModelData,
          ExpectNotOK{});
    }
  }
}

TEST_IF_P(MOCHI_USE_HDF5, MochiContextTest, LoadShapeFromFile_Sdf) {
  // Load file with a pre-computed SDF
  constexpr std::string_view kPath = "cube/cube_mesh.mochi.h5";
  for (bool shouldPreload : {false, true}) {
    for (bool useModelData : {false, true}) {
      // Load with identity transform
      ShapeHandle refShape = TestLoadShape(
          _mochiContext, kPath, Real3{1_r, 1_r, 1_r}, {}, shouldPreload, useModelData, ExpectOK{});
      MOCHI_DEFER(_mochiContext->ReleaseShape(refShape));

      // Peak at the Impl class to verify that a GridSdf was loaded
      auto const* contextImpl = assert_cast<ContextImpl const*>(_mochiContext);
      auto refShapePtr = dynamic_pointer_cast<TetrahedralMeshShape const>(
          contextImpl->GetShapeSharedPtr(refShape));
      EXPECT_NE((TetrahedralMeshShape const*)nullptr, refShapePtr.get());
      GridSdf const* refGridSdf = refShapePtr->GetGridSdf().get();
      EXPECT_NE((GridSdf const*)nullptr, refGridSdf);

      // cube_mesh.mochi.h5 defines a unit cube spanning from (0, 0, 0) to (1, 1, 1). Check the
      // bounds with a large tolerance to account for penalty falloff, which may have been baked in.
      Aabb refSdfBounds = GetAabb(refGridSdf->GetColliderBounds());
      EXPECT_NEAR_TOL(Real3(0_r, 0_r, 0_r), refSdfBounds.GetMin(), 0.01_r);
      EXPECT_NEAR_TOL(Real3(1_r, 1_r, 1_r), refSdfBounds.GetMax(), 0.01_r);

      // Test some points on the surface
      Real3 constexpr kRefSurfacePoints[] = {
          Real3{0.5_r, 0_r, 0.5_r}, Real3{0.5_r, 1_r, 0.5_r}, Real3{1_r, 0.5_r, 0.5_r}};
      Real3 constexpr kRefSurfaceNormals[] = {
          Real3{0.0_r, -1_r, 0_r}, Real3{0_r, 1_r, 0_r}, Real3{1_r, 0_r, 0_r}};
      static_assert(std::size(kRefSurfacePoints) == std::size(kRefSurfaceNormals));
      ContactDetectionResult contactResult;
      ContactDetectionParams detectionParams;
      detectionParams.tolerance = 0.001_r;
      refGridSdf->FindPointContacts(
          kRefSurfacePoints,
          TransformRT{} /*pointsFromSdf*/,
          detectionParams,
          contactResult.sampleIndices,
          contactResult.posColliding,
          contactResult.sdfInfo,
          contactResult.isSdfGradUnitary);
      ASSERT_EQ(isize(kRefSurfacePoints), isize(contactResult.sdfInfo));
      for (int i = 0; i < isize(kRefSurfacePoints); ++i) {
        EXPECT_NEAR_EQ(0_r, contactResult.sdfInfo.val[i]);
        EXPECT_NEAR_EQ(kRefSurfacePoints[i], ToReal3(contactResult.GetApproxPosCollider(i)));
        EXPECT_NEAR_EQ(kRefSurfacePoints[i], contactResult.posColliding[i]);
        EXPECT_NEAR_EQ(kRefSurfaceNormals[i], contactResult.sdfInfo.grad[i]);
      }

      // Load the file again with non-uniform scale. This is legal, although non-uniform scale
      // cannot be baked into an SDF. Therefore, the SDF data in the file will be ignored, but
      // we can still compute one on-demand if necessary.
      {
        Real3 scale{0.1_r, 0.2_r, 0.3_r};
        ShapeHandle newShape =
            TestLoadShape(_mochiContext, kPath, scale, {}, shouldPreload, useModelData, ExpectOK{});
        MOCHI_DEFER(_mochiContext->ReleaseShape(newShape));
        auto newShapePtr = dynamic_pointer_cast<TetrahedralMeshShape const>(
            contextImpl->GetShapeSharedPtr(newShape));
        EXPECT_EQ((GridSdf const*)nullptr, newShapePtr->GetGridSdf().get()); // No SDF yet
        bool isPending = false;
        GridSdfParams sdfParams;
        sdfParams.boundaryPaddingDist = 0.005_r; // more than detectionParams.tolerance
        newShapePtr->RequestGridSdf(sdfParams, &isPending); // Start SDF processing
        EXPECT_TRUE(isPending);
        newShapePtr->GetGridSdfSemaphore().Wait(); // Wait for completion
        auto newGridSdf =
            newShapePtr->RequestGridSdf(sdfParams, &isPending); // Get newly processed SDF
        EXPECT_FALSE(isPending);
        EXPECT_NE((GridSdf const*)nullptr, newGridSdf.get());

        // Check the SDF bounds
        Aabb newSdfBounds = GetAabb(newGridSdf->GetColliderBounds());
        EXPECT_NEAR_TOL(Real3(0_r, 0_r, 0_r), newSdfBounds.GetMin(), 0.01_r);
        EXPECT_NEAR_TOL(Real3(0.1_r, 0.2_r, 0.3_r), newSdfBounds.GetMax(), 0.01_r); // scaled bounds

        // Check some points on the (scaled) surface.
        std::vector<Real3> surfacePoints(isize(kRefSurfacePoints));
        for (int i = 0; i < isize(kRefSurfacePoints); ++i) {
          surfacePoints[i] = kRefSurfacePoints[i] * scale;
        }
        contactResult.Clear();
        newGridSdf->FindPointContacts(
            surfacePoints,
            TransformRT{} /*pointsFromSdf*/,
            detectionParams,
            contactResult.sampleIndices,
            contactResult.posColliding,
            contactResult.sdfInfo,
            contactResult.isSdfGradUnitary);
        ASSERT_EQ(isize(surfacePoints), isize(contactResult.sdfInfo));
        for (int i = 0; i < isize(surfacePoints); ++i) {
          EXPECT_NEAR_EQ(0_r, contactResult.sdfInfo.val[i]);
          EXPECT_NEAR_EQ(surfacePoints[i], ToReal3(contactResult.GetApproxPosCollider(i)));
          EXPECT_NEAR_EQ(surfacePoints[i], contactResult.posColliding[i]);
          // Expect same SDF gradient as the reference mesh normal
          EXPECT_NEAR_EQ(kRefSurfaceNormals[i], contactResult.sdfInfo.grad[i]);
        }
      }
    }
  }
}

TEST_IF_P(MOCHI_USE_HDF5, MochiContextTest, LoadShapeFromFile_Bsh) {
  // TODO: Determine if a BSH was actually loaded successfully.
  constexpr std::string_view kPath = "duck/duck_730.mochi.h5";
  auto shape = _mochiContext->LoadShapeFromFile(test::GetAssetPath(kPath), ExpectOK{});
  _mochiContext->ReleaseShape(shape);
}

static void ExpectShapeDoesNotExist(Context* context, ShapeHandle shape) {
  // If the shape doesn't exist, these calls should all fail
  [[maybe_unused]] auto mesh = context->GetShapeMesh(shape, ExpectNotOK{});
  context->ReleaseShape(shape);
}
// Expects a tetrahedral mesh like the one gererated by test::CreateMinimalTetMeshUnitCube(),
// with an additional scale baked in
static void
ExpectMinimalUnitCubeTetMesh(mochi::Context const* context, ShapeHandle shape, real scale) {
  EXPECT_TRUE(shape.IsValid());

  // Volume mesh
  MeshDataView mesh = context->GetShapeMesh(shape, ExpectOK{});
  EXPECT_EQ(8, mesh.GetNumNodes());
  EXPECT_EQ(5, mesh.GetNumElements());
  EXPECT_EQ(4, mesh.nodesPerElement);
  EXPECT_EQ(20, isize(mesh.connectivity));
  EXPECT_EQ(24, isize(mesh.coordinates));

  // Edge computation
  auto edgeIndices = ComputeEdgeIndices(mesh);
  EXPECT_EQ(18, isize(edgeIndices) / 2);
  EXPECT_EQ(36, isize(edgeIndices));

  // Surface mesh
  MeshDataView surfaceMesh = context->GetShapeSurfaceMesh(shape, ExpectOK{});
  EXPECT_EQ(12, surfaceMesh.GetNumElements());
  EXPECT_EQ(3, surfaceMesh.nodesPerElement);
  EXPECT_EQ(36, isize(surfaceMesh.connectivity));

  // Surface edge computation
  auto surfaceEdges = ComputeEdgeIndices(surfaceMesh);
  EXPECT_EQ(18, isize(surfaceEdges) / 2);

  // Element barycenters
  auto barycenters = ComputeElementBarycenters(mesh);
  EXPECT_EQ(5 * 3, isize(barycenters));
  // Each barycenter should be inside the cube [0, scale]^3
  for (int e = 0; e < mesh.GetNumElements(); ++e) {
    for (int d = 0; d < 3; ++d) {
      EXPECT_GE(barycenters[e * 3 + d], 0_r);
      EXPECT_LE(barycenters[e * 3 + d], scale);
    }
  }

  EXPECT_NEAR_EQ(
      Aabb(Real3{}, Real3{scale, scale, scale}), context->GetShapeAabb(shape, ExpectOK{}));
}

TEST_P(MochiContextTest, LoadShapeFromBytes_BadData) {
  // This test attempts and fails to load bad data. Warnings and/or errors might be logged
  // in the process. Temporarily silence those, since they are expected.
  auto prevLogFn = mochi::Context::GetLogCallback();
  mochi::Context::SetLogCallback(
      [&](LogChannel /*channel*/, char const* /*message*/, char const* /*file*/, int /*line*/) {});
  MOCHI_DEFER(mochi::Context::SetLogCallback(prevLogFn));

  auto testBadData = [&](Span<char const> data) {
    // Try all overloads of LoadShapeFromBytes
    EXPECT_EQ(ShapeHandle{}, _mochiContext->LoadShapeFromBytes(data, ExpectNotOK{}));
    EXPECT_EQ(
        ShapeHandle{},
        _mochiContext->LoadShapeFromBytes(
            data, Real3{1_r, 1_r, 1_r}, TransformRT::Identity(), ExpectNotOK{}));
    EXPECT_EQ(
        ShapeHandle{},
        _mochiContext->LoadShapeFromBytes(
            data, Real3{1_r, 1_r, 1_r}, TransformRT::Identity(), ExpectNotOK{}));
  };

  // Try empty data buffer
  testBadData({});

  // Try invalid data buffer
  char const badData[] = "This is not the data you're looking for";
  testBadData(badData);

  // Expect no loaded shapes
  EXPECT_EQ(0, _mochiContext->GetNumShapes());
}

TEST_P(MochiContextTest, LoadShapeFromBytes_JSON) {
  ShapeHandle shape;

  // Fail to load an empty buffer
  {
    auto scope = ExpectLoggingInScope(_mochiContext, LogChannel::Warning);
    shape = _mochiContext->LoadShapeFromBytes(MakeSpan(""), ExpectNotOK{});
  }
  EXPECT_FALSE(shape.IsValid());
  ExpectShapeDoesNotExist(_mochiContext, shape);

  // Fail to load a malformed JSON object
  {
    auto scope = ExpectLoggingInScope(_mochiContext, LogChannel::Warning);
    shape = _mochiContext->LoadShapeFromBytes(MakeSpan("{json hates this,}"), ExpectNotOK{});
  }
  EXPECT_FALSE(shape.IsValid());
  ExpectShapeDoesNotExist(_mochiContext, shape);

  // Fail to load an empty mesh
  std::string data = SerializeTetMesh(TetMeshParams{});
  shape = _mochiContext->LoadShapeFromBytes(MakeSpan(data), ExpectNotOK{});
  EXPECT_FALSE(shape.IsValid());

  // Load a mesh with valid tets
  data = SerializeTetMesh(CreateMinimalTetMeshUnitCube());
  shape = _mochiContext->LoadShapeFromBytes(
      MakeSpan(data), Real3{123_r, 123_r, 123_r}, TransformRT::Identity(), ExpectOK{});
  ExpectMinimalUnitCubeTetMesh(_mochiContext, shape, 123_r);
}

void MochiContextTest::TestLoadShapeFromFileConcurrent(
    std::initializer_list<std::string_view> filePaths) {
  // Test multi-threaded loading and releasing of shapes. It is especially important to test
  // this for HDF5 files, because that third-party library is not inherently thread-safe.
  auto threadProc = [this, filePaths]() {
    constexpr int kNumLoads = 50;
    constexpr int kReleaseAge = 5;
    std::vector<ShapeHandle> shapes;
    for (int i = 0; i < kNumLoads; ++i) {
      // Load the next shape.
      std::string_view const filePath = filePaths.begin()[i % isize(filePaths)];
      bool const loadFromBytes = ((i % 2) == 0);
      bool const useModelData = ((i % 3) == 0);
      shapes.emplace_back(TestLoadShape(
          _mochiContext,
          filePath,
          Real3{1_r, 1_r, 1_r},
          TransformRT{},
          loadFromBytes,
          useModelData,
          test::ExpectOK{}));
      EXPECT_TRUE(shapes.back().IsValid());
      // Rlease an older shape
      if (i >= kReleaseAge) {
        _mochiContext->ReleaseShape(shapes[i - kReleaseAge]);
      }
    }
    std::sort(shapes.begin(), shapes.end());
    for (int i = 1; i < isize(shapes); ++i) {
      EXPECT_NE(shapes[i], shapes[i - 1]); // Expected all handles to be unique
    }
    EXPECT_NE(0, _mochiContext->GetNumShapes());
    for (int i = 0; i < kReleaseAge; ++i) {
      _mochiContext->ReleaseShape(shapes.back());
      shapes.pop_back();
    }
  };

  constexpr int kNumThreads = 4;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(threadProc);
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(0, _mochiContext->GetNumShapes()); // All shapes should have been released by now
}

TEST_P(MochiContextTest, LoadShapeFromFile_Concurrent) {
  TestLoadShapeFromFileConcurrent({
      "cube/cube_mesh.mochi.json",
#if MOCHI_USE_HDF5
      "cube/cube_mesh.mochi.h5",
#endif // MOCHI_USE_HDF5
  // These meshes are not shipped externally.
#if MOCHI_INTERNAL
      "cube/cube_minimal.mochi.json",
      "duck/duck_coarse_mesh.mochi.json",
#if MOCHI_USE_HDF5
      "duck/duck_359.mochi.h5",
#endif // MOCHI_USE_HDF5
#endif // MOCHI_INTERNAL
  });
}

TEST_P(MochiContextTest, CreateTetMeshShape) {
  // Create a unit cube
  auto unitCube = CreateMinimalTetMeshUnitCube();
  Span<real> unitCubeCoordinates = Flatten(MakeSpan(unitCube.first));
  Span<int> unitCubeConnectivity = Flatten(MakeSpan(unitCube.second));
  ShapeHandle shape =
      _mochiContext->CreateTetMeshShape(unitCubeCoordinates, unitCubeConnectivity, ExpectOK{});
  ExpectMinimalUnitCubeTetMesh(_mochiContext, shape, 1_r);

  // Fail when coordinates.size() is not a multiple of 3
  shape = _mochiContext->CreateTetMeshShape(
      unitCubeCoordinates.subspan(0, unitCubeCoordinates.size() - 1), // 1 too few
      unitCubeConnectivity,
      ExpectNotOK{});
  EXPECT_FALSE(shape.IsValid());
  ExpectShapeDoesNotExist(_mochiContext, shape);

  // Fail when connectivity.size() is not a multiple of 4
  shape = _mochiContext->CreateTetMeshShape(
      unitCubeCoordinates,
      unitCubeConnectivity.subspan(0, unitCubeConnectivity.size() - 1), // 1 too few,
      ExpectNotOK{});
  EXPECT_FALSE(shape.IsValid());
  ExpectShapeDoesNotExist(_mochiContext, shape);
}

TEST_IF_P(MOCHI_ENABLE_ROM_ACTORS, MochiContextTest, CreateModelShapeWithLinearRom) {
  auto unitCube = CreateMinimalTetMeshUnitCube();

  ModelData model;
  model.mesh.emplace();
  model.mesh->nodesPerElement = 4;
  model.mesh->coordinates = Flatten(MakeSpan(unitCube.first));
  model.mesh->connectivity = Flatten(MakeSpan(unitCube.second));

  // Test creation of ROM data
  DynamicArray<real> linearRomBasis(model.mesh->coordinates.size());
  std::iota(linearRomBasis.begin(), linearRomBasis.end(), 0_r);

  ShapeHandle shape = experimental::CreateModelShapeWithLinearRom(
      _mochiContext, model, "testRom", MakeConstSpan(linearRomBasis), ExpectOK{});
  EXPECT_TRUE(shape.IsValid());

  // Test the shape by creating a ROM actor
  SoftActorParams params;
  params.shape = shape;
  experimental::ExperimentalSoftActorParams experimentalParams;
  experimentalParams.rom.emplace();
  experimentalParams.rom->source = "testRom";
  auto* scene = _mochiContext->CreateScene("TestScene");
  MOCHI_DEFER(_mochiContext->DestroyScene(scene));
  experimental::CreateSoftActor(scene, params, experimentalParams, ExpectOK{});
}

TEST_IF_P(MOCHI_ROM_DISABLED, MochiContextTest, CreateModelShapeWithLinearRom_FeatureDisabled) {
  auto unitCube = CreateMinimalTetMeshUnitCube();

  ModelData model;
  model.mesh.emplace();
  model.mesh->nodesPerElement = 4;
  model.mesh->coordinates = Flatten(MakeSpan(unitCube.first));
  model.mesh->connectivity = Flatten(MakeSpan(unitCube.second));

  DynamicArray<real> linearRomBasis(model.mesh->coordinates.size());
  std::iota(linearRomBasis.begin(), linearRomBasis.end(), 0_r);

  ShapeHandle const shape = experimental::CreateModelShapeWithLinearRom(
      _mochiContext, model, "testRom", MakeConstSpan(linearRomBasis), ExpectNotOK{});
  EXPECT_FALSE(shape.IsValid());
  ExpectShapeDoesNotExist(_mochiContext, shape);
}

TEST_IF_P(MOCHI_ROM_DISABLED, MochiContextTest, CreateSoftActorWithRom_FeatureDisabled) {
  auto unitCube = CreateMinimalTetMeshUnitCube();
  Span<real> unitCubeCoordinates = Flatten(MakeSpan(unitCube.first));
  Span<int> unitCubeConnectivity = Flatten(MakeSpan(unitCube.second));
  ShapeHandle shape =
      _mochiContext->CreateTetMeshShape(unitCubeCoordinates, unitCubeConnectivity, ExpectOK{});

  SoftActorParams params;
  params.shape = shape;
  experimental::ExperimentalSoftActorParams experimentalParams;
  experimentalParams.rom.emplace();
  experimentalParams.rom->source = "testRom";
  auto* scene = _mochiContext->CreateScene("TestScene");
  MOCHI_DEFER(_mochiContext->DestroyScene(scene));

  Actor* actor = experimental::CreateSoftActor(scene, params, experimentalParams, ExpectNotOK{});
  EXPECT_EQ(nullptr, actor);
}

TEST_IF_P(MOCHI_DEEP_FLOW_DISABLED, MochiContextTest, CreateDeepFlowShape_FeatureDisabled) {
  experimental::DeepModelParams params;
  params.deepModelPath = "unused_model_path";

  ShapeHandle const shape = experimental::CreateDeepFlowShape(
      _mochiContext, params, NeuralComputeType::MochiCpu, 0, ExpectNotOK{});
  EXPECT_FALSE(shape.IsValid());
  ExpectShapeDoesNotExist(_mochiContext, shape);
}

TEST_IF_P(MOCHI_DEEP_FLOW_DISABLED, MochiContextTest, CreateSoftActorWithDeepFlow_FeatureDisabled) {
  auto unitCube = CreateMinimalTetMeshUnitCube();
  Span<real> unitCubeCoordinates = Flatten(MakeSpan(unitCube.first));
  Span<int> unitCubeConnectivity = Flatten(MakeSpan(unitCube.second));
  ShapeHandle shape =
      _mochiContext->CreateTetMeshShape(unitCubeCoordinates, unitCubeConnectivity, ExpectOK{});

  SoftActorParams params;
  params.shape = shape;
  experimental::ExperimentalSoftActorParams experimentalParams;
  experimentalParams.flow = ShapeHandle{1};
  EXPECT_TRUE(experimentalParams.flow.IsValid());
  auto* scene = _mochiContext->CreateScene("TestScene");
  MOCHI_DEFER(_mochiContext->DestroyScene(scene));

  Actor* actor = experimental::CreateSoftActor(scene, params, experimentalParams, ExpectNotOK{});
  EXPECT_EQ(nullptr, actor);
}

TEST_IF_P(MOCHI_ROM_DISABLED, MochiContextTest, CreateSoftSkinnedActorWithRom_FeatureDisabled) {
  auto unitCube = CreateMinimalTetMeshUnitCube();
  Span<real> unitCubeCoordinates = Flatten(MakeSpan(unitCube.first));
  Span<int> unitCubeConnectivity = Flatten(MakeSpan(unitCube.second));
  ShapeHandle shape =
      _mochiContext->CreateTetMeshShape(unitCubeCoordinates, unitCubeConnectivity, ExpectOK{});

  SoftSkinnedActorParams params;
  params.skeletonParams.joints = {{.type = ArticulatedJointType::Free}};
  params.skeletonParams.links = {
      {.parentLink = -1, .shape = shape, .colliderType = ColliderType::None}};
  params.softParams = {{.shape = shape}};
  experimental::ExperimentalSoftSkinnedActorParams experimentalParams;
  experimental::ExperimentalSoftActorParams softExperimentalParams;
  softExperimentalParams.rom.emplace();
  softExperimentalParams.rom->source = "polynomial_crom_order_1";
  experimentalParams.softParams = {softExperimentalParams};
  auto* scene = _mochiContext->CreateScene("TestScene");
  MOCHI_DEFER(_mochiContext->DestroyScene(scene));

  Actor* actor =
      experimental::CreateSoftSkinnedActor(scene, params, experimentalParams, ExpectNotOK{});
  EXPECT_EQ(nullptr, actor);
}

TEST_IF_P(
    MOCHI_DEEP_FLOW_DISABLED,
    MochiContextTest,
    CreateSoftSkinnedActorWithDeepFlow_FeatureDisabled) {
  auto unitCube = CreateMinimalTetMeshUnitCube();
  Span<real> unitCubeCoordinates = Flatten(MakeSpan(unitCube.first));
  Span<int> unitCubeConnectivity = Flatten(MakeSpan(unitCube.second));
  ShapeHandle shape =
      _mochiContext->CreateTetMeshShape(unitCubeCoordinates, unitCubeConnectivity, ExpectOK{});

  SoftSkinnedActorParams params;
  params.skeletonParams.joints = {{.type = ArticulatedJointType::Free}};
  params.skeletonParams.links = {
      {.parentLink = -1, .shape = shape, .colliderType = ColliderType::None}};
  params.softParams = {{.shape = shape}};
  experimental::ExperimentalSoftSkinnedActorParams experimentalParams;
  experimental::ExperimentalSoftActorParams softExperimentalParams;
  softExperimentalParams.flow = ShapeHandle{1};
  EXPECT_TRUE(softExperimentalParams.flow.IsValid());
  experimentalParams.softParams = {softExperimentalParams};
  auto* scene = _mochiContext->CreateScene("TestScene");
  MOCHI_DEFER(_mochiContext->DestroyScene(scene));

  Actor* actor =
      experimental::CreateSoftSkinnedActor(scene, params, experimentalParams, ExpectNotOK{});
  EXPECT_EQ(nullptr, actor);
}

TEST_P(MochiContextTest, ReleaseShape) {
  EXPECT_EQ(0, _mochiContext->GetNumShapes());

  // Release an invalid handle
  _mochiContext->ReleaseShape(ShapeHandle{});
  _mochiContext->ReleaseShape(ShapeHandle{911});
  EXPECT_EQ(0, _mochiContext->GetNumShapes());

  // Create two shapes
  auto unitCube = CreateMinimalTetMeshUnitCube();
  Span<real> unitCubeCoordinates = Flatten(MakeSpan(unitCube.first));
  Span<int> unitCubeConnectivity = Flatten(MakeSpan(unitCube.second));
  ShapeHandle shape1 =
      _mochiContext->CreateTetMeshShape(unitCubeCoordinates, unitCubeConnectivity, ExpectOK{});
  EXPECT_EQ(1, _mochiContext->GetNumShapes());
  ShapeHandle shape2 =
      _mochiContext->CreateTetMeshShape(unitCubeCoordinates, unitCubeConnectivity, ExpectOK{});
  EXPECT_EQ(2, _mochiContext->GetNumShapes());
  ExpectMinimalUnitCubeTetMesh(_mochiContext, shape2, 1_r);
  ExpectMinimalUnitCubeTetMesh(_mochiContext, shape2, 1_r);

  // Release one of them
  _mochiContext->ReleaseShape(shape1);
  EXPECT_EQ(1, _mochiContext->GetNumShapes());
  ExpectShapeDoesNotExist(_mochiContext, shape1); // no longer usable
  ExpectMinimalUnitCubeTetMesh(_mochiContext, shape2, 1_r); // no change

  // Try to release it again
  _mochiContext->ReleaseShape(shape1);
  EXPECT_EQ(1, _mochiContext->GetNumShapes()); // no change

  // Release the other
  _mochiContext->ReleaseShape(shape2);
  EXPECT_EQ(0, _mochiContext->GetNumShapes());
  ExpectShapeDoesNotExist(_mochiContext, shape2);
}

TEST_P(MochiContextTest, AutoReleaseShapeHandle) {
  EXPECT_EQ(0, _mochiContext->GetNumShapes());

  ShapeHandle a = _mochiContext->CreateSphereShape(Real3{}, 0.1_r, test::ExpectOK{});
  EXPECT_EQ(1, _mochiContext->GetNumShapes());

  ShapeHandle b = _mochiContext->CreateSphereShape(Real3{}, 0.1_r, test::ExpectOK{});
  EXPECT_EQ(2, _mochiContext->GetNumShapes());
  EXPECT_NE(a, b);

  ShapeHandle a2 = a; // Ref count = 2
  EXPECT_EQ(a2, a);
  a = {}; // Ref count = 1
  EXPECT_EQ(2, _mochiContext->GetNumShapes()); // No change
  a2 = {}; // Ref count = 0
  EXPECT_EQ(1, _mochiContext->GetNumShapes()); // No change

  RigidActorParams params;
  params.shape = b; // Ref count = 2
  b = {}; // Ref count = 1
  EXPECT_EQ(1, _mochiContext->GetNumShapes()); // No change

  params = {}; // Ref count = 0
  EXPECT_EQ(0, _mochiContext->GetNumShapes());

  // Create more shapes
  int constexpr kNumShapes = 100;
  DynamicArray<ShapeHandle> shapes(kNumShapes);
  for (auto& s : shapes) {
    s = _mochiContext->CreatePlaneShape(Real3{0_r, 1_r, 0_r}, 0_r, test::ExpectOK{});
  }
  EXPECT_EQ(kNumShapes, _mochiContext->GetNumShapes());

  // You can still release shapes manually if you want early cleanup (despite non-zero existing
  // handles)
  for (int i = 0; i < kNumShapes; i += 2) {
    _mochiContext->ReleaseShape(shapes[i]);
  }
  EXPECT_EQ(kNumShapes / 2, _mochiContext->GetNumShapes());
  shapes.clear();
  EXPECT_EQ(0, _mochiContext->GetNumShapes());
}

TEST(Context, ReleaseShapeAfterContext) {
  {
    Context* context = CreateContext(/*numWorkerThreads*/ 0);
    ShapeHandle shape = context->CreateSphereShape(Real3{}, 0.1_r, test::ExpectOK{});
    EXPECT_TRUE(shape.IsValid());
    EXPECT_EQ(1, context->GetNumShapes());
    DestroyContext(context);
    // It is safe to let the ShapeHandle go out of scope AFTER the context is destroyed.
  }

  {
    // Create another shape, then destroy the context
    Context* contextA = CreateContext(/*numWorkerThreads*/ 0);
    ShapeHandle shapeA = contextA->CreateSphereShape(Real3{}, 0.1_r, test::ExpectOK{});
    EXPECT_TRUE(shapeA.IsValid());
    EXPECT_EQ(1, contextA->GetNumShapes());
    DestroyContext(contextA);

    // Create another context with its own shape.
    Context* contextB = CreateContext(/*numWorkerThreads*/ 0);
    ShapeHandle shapeB = contextB->CreateSphereShape(Real3{}, 0.1_r, test::ExpectOK{});
    EXPECT_TRUE(shapeB.IsValid());
    EXPECT_EQ(1, contextB->GetNumShapes());

    // Let shapeA try to clean itself up after ContextA was destroyed. This should not affect
    // contextB, even if shapeA and shapeB happen to have the same numeric handle value.
    shapeA = {};
    EXPECT_EQ(1, contextB->GetNumShapes()); // No change

    shapeB = {};
    EXPECT_EQ(0, contextB->GetNumShapes());

    DestroyContext(contextB);
  }
}

// Helper function. Call a function multiple times from multiple threads
static void RepeatMultithreaded(Context* context, std::function<void()> const& fn, int numThreads) {
  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  for (int i = 0; i < numThreads; ++i) {
    threads.emplace_back([&]() {
      context->BindThisThread();
      fn();
      context->UnbindThisThread();
    });
  }
  for (auto& t : threads) {
    t.join();
  }
}

TEST_P(MochiContextTest, ShapeCreateMultithreaded) {
  // Create and destroy shapes from multiple threads

  auto mesh = CreateMinimalTetMeshUnitCube();
  Span<real> coordinates = Flatten(MakeSpan(mesh.first));
  Span<int> connectivity = Flatten(MakeSpan(mesh.second));
  std::string fileData = SerializeTetMesh(mesh);

  auto fn = [&]() {
    for (int i = 0; i < 100; ++i) {
      // CreateTetMeshShape
      ShapeHandle shape = _mochiContext->CreateTetMeshShape(coordinates, connectivity, ExpectOK{});
      ExpectMinimalUnitCubeTetMesh(_mochiContext, shape, 1_r);
      _mochiContext->ReleaseShape(shape);
      ExpectShapeDoesNotExist(_mochiContext, shape);

      // LoadShapeFromBytes (with JSON)
      shape = _mochiContext->LoadShapeFromBytes(fileData, ExpectOK{});
      ExpectMinimalUnitCubeTetMesh(_mochiContext, shape, 1_r);
      _mochiContext->ReleaseShape(shape);
      ExpectShapeDoesNotExist(_mochiContext, shape);
    }
  };

  RepeatMultithreaded(_mochiContext, fn, 2);
}

TEST_P(MochiContextTest, ShapeAutoReleaseMultithreaded) {
  // Create and destroy shapes from multiple threads
  EXPECT_EQ(0, _mochiContext->GetNumShapes());

  auto mesh = CreateMinimalTetMeshUnitCube();
  Span<real> coordinates = Flatten(MakeSpan(mesh.first));
  Span<int> connectivity = Flatten(MakeSpan(mesh.second));
  std::string fileData = SerializeTetMesh(mesh);

  auto fn = [&]() {
    for (int i = 0; i < 100; ++i) {
      // CreateTetMeshShape
      ShapeHandle shape = _mochiContext->CreateTetMeshShape(coordinates, connectivity, ExpectOK{});
      ExpectMinimalUnitCubeTetMesh(_mochiContext, shape, 1_r);

      // LoadShapeFromBytes (with JSON)
      // Releases the first shape on assignment.
      shape = _mochiContext->LoadShapeFromBytes(fileData, ExpectOK{});
      ExpectMinimalUnitCubeTetMesh(_mochiContext, shape, 1_r);

      // Let the shape go out-of-scope
    }
  };

  RepeatMultithreaded(_mochiContext, fn, 2);

  // All shapes cleaned up
  EXPECT_EQ(0, _mochiContext->GetNumShapes());
}

TEST_P(MochiContextTest, GetScene) {
  // Context::CreateScene
  Scene* scene1 = _mochiContext->CreateScene("Scene 1");
  Scene* scene2 = _mochiContext->CreateScene("Scene 2");
  auto handle1 = scene1->GetHandle();
  auto handle2 = scene2->GetHandle();
  EXPECT_TRUE(handle1.IsValid());
  EXPECT_TRUE(handle2.IsValid());
  EXPECT_NE(handle1, handle2);

  // Scene::GetContext
  EXPECT_EQ(_mochiContext, scene1->GetContext());
  EXPECT_EQ(_mochiContext, scene2->GetContext());

  // Scene::GetContext - const overload
  EXPECT_EQ(_mochiContext, static_cast<Scene const*>(scene1)->GetContext());
  EXPECT_EQ(_mochiContext, static_cast<Scene const*>(scene2)->GetContext());

  // Context::GetScene
  Scene* recoveredScene1 = _mochiContext->GetScene(handle1);
  Scene* recoveredScene2 = _mochiContext->GetScene(handle2);
  EXPECT_EQ(scene1, recoveredScene1);
  EXPECT_EQ(scene2, recoveredScene2);

  // Context::GetScene - const overload
  Scene const* recoveredConstScene1 = static_cast<Context const*>(_mochiContext)->GetScene(handle1);
  Scene const* recoveredConstScene2 = static_cast<Context const*>(_mochiContext)->GetScene(handle2);
  EXPECT_EQ(scene1, recoveredConstScene1);
  EXPECT_EQ(scene2, recoveredConstScene2);

  // Try to get a from a valid handle belonging to a deleted scene.
  _mochiContext->DestroyScene(scene1);
  EXPECT_FALSE(_mochiContext->IsValidScene(scene1));
  EXPECT_EQ(nullptr, _mochiContext->GetScene(handle1));

  // Try to recover from an invalid handle.
  auto invalidHandle = SceneHandle{};
  EXPECT_FALSE(invalidHandle.IsValid());
  EXPECT_EQ(nullptr, _mochiContext->GetScene(invalidHandle));
}

TEST_P(MochiContextTest, CreateSceneMultithreaded) {
  // Create and destroy scenes from multiple threads
  auto fn = [&]() {
    for (int i = 0; i < 100; ++i) {
      Scene* scene = _mochiContext->CreateScene("My Scene");
      EXPECT_NE((Scene*)nullptr, scene);
      _mochiContext->DestroyScene(scene);
    }
  };

  RepeatMultithreaded(_mochiContext, fn, 2);
}

TEST_P(MochiContextTest, CreateAsyncScene) {
  if (_mochiContext->GetNumThreads() == 0) {
    // If there are no worker threads, then CreateAsyncScene should fail gracefully.
    auto* asyncScene = _mochiContext->CreateAsyncScene("nope", ExpectNotOK{});
    EXPECT_EQ((AsyncScene*)nullptr, asyncScene);
    _mochiContext->DestroyAsyncScene(asyncScene); // Should be safe
  } else {
    // Create and destroy an AsyncScene
    auto* asyncScene = _mochiContext->CreateAsyncScene("nope", ExpectOK{});
    ASSERT_NE((AsyncScene*)nullptr, asyncScene);
    EXPECT_FALSE(asyncScene->IsPaused());
    std::thread::id const callingThreadId = std::this_thread::get_id();
    std::thread::id simulationThreadId;
    bool works = false;
    asyncScene->QueueCommand([&](Scene*) {
      simulationThreadId = std::this_thread::get_id();
      works = true;
    });
    asyncScene->WaitForQueuedCommands();
    EXPECT_TRUE(works);
    EXPECT_NE(callingThreadId, simulationThreadId);
    _mochiContext->DestroyAsyncScene(asyncScene);
  }
}

TEST_P(MochiContextTest, CreateAsyncScenePaused) {
  if (_mochiContext->GetNumThreads() == 0) {
    // If there are no worker threads, then CreateAsyncScenePaused should fail gracefully.
    auto* asyncScene = _mochiContext->CreateAsyncScenePaused("nope", ExpectNotOK{});
    EXPECT_EQ((AsyncScene*)nullptr, asyncScene);
    _mochiContext->DestroyAsyncScene(asyncScene); // Should be safe
  } else {
    // Create and destroy an AsyncScene
    auto* asyncScene = _mochiContext->CreateAsyncScenePaused("nope", ExpectOK{});
    ASSERT_NE((AsyncScene*)nullptr, asyncScene);
    EXPECT_TRUE(asyncScene->IsPaused()); // IsPaused is immediately  visible on this thread
    bool works = false;
    asyncScene->QueueCommand([&](Scene*) {
      works = true;
      EXPECT_TRUE(asyncScene->IsPaused()); // Pause state observable from the sim thread too
    });
    asyncScene->WaitForQueuedCommands();
    EXPECT_TRUE(works);
    asyncScene->Pause(false); // unpause
    asyncScene->QueueCommand([&](Scene*) {
      EXPECT_FALSE(asyncScene->IsPaused()); // Unpaused state observable from the sim thread too
    });
    asyncScene->WaitForQueuedCommands();
    EXPECT_FALSE(asyncScene->IsPaused());
    _mochiContext->DestroyAsyncScene(asyncScene);
  }
}

TEST_P(MochiContextTest, CreateAsyncSceneMultithreaded) {
  // This TEST_P is parameterized by the number of worker threads.
  // Skip this test when that number is zero because AsyncScene requires at least one.
  if (_mochiContext->GetNumThreads() > 0) {
    // Create and destroy scenes from multiple threads
    auto fn = [&]() {
      for (int i = 0; i < 100; ++i) {
        AsyncScene* scene = _mochiContext->CreateAsyncScene("My Scene", ExpectOK{});
        EXPECT_NE((AsyncScene*)nullptr, scene);
        _mochiContext->DestroyAsyncScene(scene);
      }
    };

    RepeatMultithreaded(_mochiContext, fn, 2);
  }
}

static void ExpectCreateAsyncSceneFromMochiWorkerFails(bool startPaused) {
  auto* context = CreateContext(2);
  ASSERT_NE((Context*)nullptr, context);
  MOCHI_DEFER(DestroyContext(context));

  auto* outerScene = context->CreateAsyncScene("Outer Scene", ExpectOK{});
  ASSERT_NE((AsyncScene*)nullptr, outerScene);
  MOCHI_DEFER(context->DestroyAsyncScene(outerScene));

  AsyncScene* nestedScene = nullptr;
  bool commandRan = false;
  bool errorWasSet = false;

  outerScene->QueueCommand([&](Scene*) {
    Error error;
    nestedScene = startPaused ? context->CreateAsyncScenePaused("Nested Scene", error)
                              : context->CreateAsyncScene("Nested Scene", error);
    errorWasSet = !error.IsOK();
    commandRan = true;
  });
  outerScene->WaitForQueuedCommands();

  if (nestedScene != nullptr) {
    context->DestroyAsyncScene(nestedScene);
  }
  EXPECT_TRUE(commandRan);
  EXPECT_TRUE(errorWasSet);
  EXPECT_EQ((AsyncScene*)nullptr, nestedScene);
}

TEST(MochiAsyncScene, CreateAsyncSceneExpectedFailures) {
  ExpectCreateAsyncSceneFromMochiWorkerFails(/*startPaused*/ false);
  ExpectCreateAsyncSceneFromMochiWorkerFails(/*startPaused*/ true);
}

namespace {
class MochiSceneTest : public test::MochiSceneTestBase {};

class MochiSceneForeignHandleTest : public MochiSceneTest {
 public:
  Actor* CreateRigidUnitCube() {
    auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
    auto cubeShape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeConstSpan(coordinates)), Flatten(MakeConstSpan(connectivity)), ExpectOK{});
    MOCHI_DEFER(_mochiContext->ReleaseShape(cubeShape));
    RigidActorParams params;
    params.shape = cubeShape;
    params.colliderType = ColliderType::Box;
    return _scene->CreateRigidActor(params, ExpectOK{});
  }
};
} // namespace

TEST_F(MochiSceneTest, SceneTimeAndStepCounter) {
  auto const& reg = GetRegistry();
  auto const& sceneTime = reg.ctx<CSceneTime const>();
  auto const& sceneStepCounter = reg.ctx<CSceneStepCounter const>();

  // We haven't stepped the scene yet
  EXPECT_EQ(sceneTime.DeltaTimePrev(), 0.0);
  EXPECT_EQ(sceneTime.DeltaTime(), CSceneTime::kDefaultTimeStep);
  EXPECT_EQ(sceneTime.StepStartTime(), -CSceneTime::kDefaultTimeStep);
  EXPECT_EQ(sceneTime.StepEndTime(), 0.0);
  EXPECT_EQ(sceneStepCounter.value, 0);

  // Time += 0.1
  _scene->Step(0.1);
  EXPECT_EQ(sceneTime.DeltaTimePrev(), CSceneTime::kDefaultTimeStep);
  EXPECT_EQ(sceneTime.DeltaTime(), 0.1);
  EXPECT_EQ(sceneTime.StepStartTime(), 0.0);
  EXPECT_NEAR_EQ(sceneTime.StepEndTime(), 0.1);
  EXPECT_EQ(sceneStepCounter.value, 1);

  // Time += 0.2
  _scene->Step(0.2);
  EXPECT_EQ(sceneTime.DeltaTimePrev(), 0.1);
  EXPECT_EQ(sceneTime.DeltaTime(), 0.2);
  EXPECT_NEAR_EQ(sceneTime.StepStartTime(), 0.1);
  EXPECT_NEAR_EQ(sceneTime.StepEndTime(), 0.3);
  EXPECT_EQ(sceneStepCounter.value, 2);

  // Time step of 0.0 does not advance anything
  _scene->Step(0.0);
  EXPECT_EQ(sceneTime.DeltaTimePrev(), 0.1); // no change
  EXPECT_NEAR_EQ(sceneTime.DeltaTime(), 0.2); // no change
  EXPECT_NEAR_EQ(sceneTime.StepStartTime(), 0.1); // no change
  EXPECT_NEAR_EQ(sceneTime.StepEndTime(), 0.3); // no change
  EXPECT_EQ(sceneStepCounter.value, 3);

  // Time += 0.3
  _scene->Step(0.3);
  EXPECT_EQ(sceneTime.DeltaTimePrev(), 0.2);
  EXPECT_NEAR_EQ(sceneTime.DeltaTime(), 0.3);
  EXPECT_NEAR_EQ(sceneTime.StepStartTime(), 0.3);
  EXPECT_NEAR_EQ(sceneTime.StepEndTime(), 0.6);
  EXPECT_EQ(sceneStepCounter.value, 4);
}

TEST_F(MochiSceneForeignHandleTest, RejectForeignSceneActorHandle) {
  auto* actor = CreateRigidUnitCube();
  auto* otherScene = _mochiContext->CreateScene("other");
  MOCHI_DEFER(_mochiContext->DestroyScene(otherScene));

  ActorHandle foreignHandle =
      GetActorHandle(ExtractEntity(actor->GetHandle()), otherScene->GetHandle());

  EXPECT_EQ(nullptr, _scene->GetActor(ActorHandle{}));
  _scene->DestroyActor(ActorHandle{});

  {
    test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
    EXPECT_EQ(nullptr, _scene->GetActor(foreignHandle));
  }
  {
    test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
    _scene->DestroyActor(foreignHandle);
  }

  EXPECT_EQ(1, _scene->GetNumActors());
  EXPECT_EQ(actor, _scene->GetActor(actor->GetHandle()));
}

TEST_F(MochiSceneForeignHandleTest, RejectForeignSceneConstraintHandle) {
  auto* actorA = CreateRigidUnitCube();
  auto* actorB = CreateRigidUnitCube();

  RigidSphericalJointConstraintParams params;
  params.actorA = actorA->GetHandle();
  params.actorB = actorB->GetHandle();
  auto* constraint = _scene->CreateRigidSphericalJointConstraint(params, ExpectOK{});

  auto* otherScene = _mochiContext->CreateScene("other");
  MOCHI_DEFER(_mochiContext->DestroyScene(otherScene));

  ConstraintHandle foreignHandle =
      GetConstraintHandle(ExtractEntity(constraint->GetHandle()), otherScene->GetHandle());

  EXPECT_EQ(nullptr, _scene->GetConstraint(ConstraintHandle{}));
  _scene->DestroyConstraint(ConstraintHandle{});

  {
    test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
    EXPECT_EQ(nullptr, _scene->GetConstraint(foreignHandle));
  }
  {
    test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
    _scene->DestroyConstraint(foreignHandle);
  }

  EXPECT_EQ(1, _scene->GetNumConstraints());
  EXPECT_EQ(constraint, _scene->GetConstraint(constraint->GetHandle()));
}

// Test fixture class. Used when testing the Actor class API.
class ActorTest : public MochiSceneTestBase {
 public:
  Actor* CreateRigidUnitCube(bool isStatic = false) {
    auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
    auto cubeShape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeConstSpan(coordinates)), Flatten(MakeConstSpan(connectivity)), ExpectOK{});
    MOCHI_DEFER(_mochiContext->ReleaseShape(cubeShape));
    RigidActorParams params;
    params.shape = cubeShape;
    params.isStatic = isStatic;
    params.colliderType = ColliderType::Box;
    return _scene->CreateRigidActor(params, ExpectOK{});
  }

  Actor* CreateSoftUnitCube() {
    auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
    auto cubeShape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeConstSpan(coordinates)), Flatten(MakeConstSpan(connectivity)), ExpectOK{});
    MOCHI_DEFER(_mochiContext->ReleaseShape(cubeShape));
    SoftActorParams params;
    params.shape = cubeShape;
    return _scene->CreateSoftActor(params, ExpectOK{});
  }
};

TEST_F(ActorTest, GetContext) {
  Actor* actor = CreateRigidUnitCube();
  EXPECT_EQ(_mochiContext, actor->GetContext());
  EXPECT_EQ(_mochiContext, static_cast<Actor const*>(actor)->GetContext());
}

TEST_F(ActorTest, GetScene) {
  Actor* actor = CreateRigidUnitCube();
  EXPECT_EQ(_scene, actor->GetScene());
  EXPECT_EQ(_scene, static_cast<Actor const*>(actor)->GetScene());
}

TEST_F(ActorTest, GetMesh_Rigid) {
  // Rigid actors with a tetrahedral mesh shape must return the boundary surface simulation mesh.
  Actor* rigidTetActor = CreateRigidUnitCube();
  EXPECT_EQ(rigidTetActor->GetSurfaceMesh(), rigidTetActor->GetMesh());

  // Likewise for static rigid actors.
  Actor* staticActor = CreateRigidUnitCube(/* isStatic */ true);
  EXPECT_EQ(staticActor->GetSurfaceMesh(), staticActor->GetMesh());

  // Rigid actors with a triangular mesh shape must return the compact active-node surface mesh,
  // not the raw authored shape mesh.
  auto [coordinates, connectivity] = test::CreateMinimalTriMeshUnitCube();
  coordinates.push_back(Real3{});
  auto triShape = _mochiContext->CreateTriMeshShape(
      Flatten(MakeConstSpan(coordinates)), Flatten(MakeConstSpan(connectivity)), ExpectOK{});
  MOCHI_DEFER(_mochiContext->ReleaseShape(triShape));

  RigidActorParams triParams;
  triParams.shape = triShape;
  triParams.colliderType = ColliderType::Box;
  Actor* rigidTriActor = _scene->CreateRigidActor(triParams, ExpectOK{});

  EXPECT_EQ(9, _mochiContext->GetShapeMesh(triShape, ExpectOK{}).GetNumNodes());
  EXPECT_EQ(8, rigidTriActor->GetMesh().GetNumNodes());
  EXPECT_EQ(rigidTriActor->GetSurfaceMesh(), rigidTriActor->GetMesh());

  // Rigid actors with an implicit shape must return an empty mesh.
  auto planeShape = _mochiContext->CreatePlaneShape(Real3{0_r, 1_r, 0_r}, 0_r, ExpectOK{});
  MOCHI_DEFER(_mochiContext->ReleaseShape(planeShape));
  RigidActorParams planeParams;
  planeParams.shape = planeShape;
  planeParams.isStatic = true;
  planeParams.colliderType = ColliderType::Plane;
  Actor* planeActor = _scene->CreateRigidActor(planeParams, ExpectOK{});
  EXPECT_EQ(MeshDataView{}, planeActor->GetMesh());
}

TEST_F(ActorTest, UserData) {
  Actor* actor = CreateRigidUnitCube();

  struct MyUserData {
    int val = 0;
  };

  MyUserData myUserData{42};
  EXPECT_EQ((void*)nullptr, actor->GetUserData()); // nullptr by default

  actor->SetUserData(&myUserData);
  void* userData = actor->GetUserData();
  EXPECT_NE(userData, (void*)nullptr);
  auto* asMyUserData = (MyUserData*)userData;
  EXPECT_EQ(asMyUserData->val, 42);
}

static void TestReal3(Real3 const& a, Real3 const& b, real tol) {
  real error = Norm(a - b) / Max(Norm(a), Norm(b));
  EXPECT_NEAR(0_r, error, tol);
}

TEST_F(ActorTest, GetVelocityRigid) {
  auto& reg = GetRegistry();

  // Create rigid actor
  Actor* actor = CreateRigidUnitCube();
  auto entity = mochi::GetEntity(reg, actor->GetHandle(), ExpectOK{});

  // Set a rigid transformation
  Real3 rotVec = Real3{-0.3_r, 0.4_r, 0.8_r};
  TransformRT transform{Quaternion::FromRotationVector(rotVec), Real3{0.6_r, 2.3_r, -1.4_r}};
  actor->SetRootTransform(transform, test::ExpectOK{});

  // Set rigid velocity
  Real3 linVel{-1_r, 2_r, 1_r};
  Real3 angVel{0.5_r, -0.5_r, 1_r};
  auto& actorVel = reg.get<CRigidVel<TimeStep::Current>>(entity).value;
  actorVel.SetVCom(ToSimd(linVel));
  actorVel.SetOmega(ToSimd(angVel));
  actorVel.UpdateVSymIfDirty(1e-2_r);

  // Run internal rigid-body velocity update
  ecs::InvokeOnEntity(&rigid::UpdateRigidVelocity_Dynamic, reg, entity);

  // Get rigid velocity and test
  TestReal3(linVel, actor->GetLinearVelocity(ExpectOK{}), 1e-4_r);
  TestReal3(angVel, actor->GetAngularVelocity(ExpectOK{}), 1e-4_r);
}

TEST_F(ActorTest, GetVelocitySoft) {
  auto& reg = GetRegistry();

  // Create soft actor
  auto* actor = CreateSoftUnitCube();
  auto entity = mochi::GetEntity(reg, actor->GetHandle(), ExpectOK{});
  ecs::InvokeOnEntity(&soft::UpdateBounds<TimeStep::Current>, reg, entity);
  auto pivotLocal =
      GetAabb(reg.get<CBoundingVolume<TimeStep::Current> const>(entity).localShape).GetCenter();

  // Set rigid velocity
  Real3 linVel{-1_r, 2_r, 1_r};
  Real3 angVel{0.5_r, -0.5_r, 1_r};
  real constexpr kDt = CSceneTime::kDefaultTimeStep;
  Quaternion rotation = Quaternion::FromRotationVector(angVel * kDt);
  Real3 translation = pivotLocal + linVel * kDt - rotation * pivotLocal;
  TransformRT transform{rotation, translation};
  auto const& rest = reg.get<CTetrahedralMesh const>(entity).mesh->GetNodeCoordinates();
  auto disp = Unflatten<Real3>(
      MakeSpan(reg.get<CDisplacementSlice<real, TimeStep::Current>>(entity).value));
  for (int i = 0; i < disp.size(); i++) {
    disp[i] = transform.TransformPoint(rest[i]) - rest[i];
  }

  ecs::InvokeOnEntity(&soft::UpdateRigidTransformEval, reg, entity);
  ecs::InvokeOnEntity(&soft::RecenterSolutionUsingRigidTransformEval, reg, entity);
  ecs::InvokeOnEntity(&soft::UpdateRigidVelocity, reg, entity);

  // Get rigid velocity and test
  TestReal3(linVel, actor->GetLinearVelocity(ExpectOK{}), 2e-2_r);
  TestReal3(angVel, actor->GetAngularVelocity(ExpectOK{}), 2.5e-3_r);
}

static void TestGridSdfCache(Context* context, ShapeHandle shape1, ShapeHandle shape2) {
  ShapeHandle shapes[] = {shape1, shape2};

  // Create a scene
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Create several rigid actors with ColliderType::Sdf. Alternate which shape handle is used.
  constexpr int kNumActors = 64;
  RigidActorParams actorParams;
  actorParams.colliderType = ColliderType::Sdf;
  actorParams.isStatic = true;
  actorParams.sdf = GridSdfParams{.resolutionDelta = {0.1_r, 0.1_r, 0.1_r}};
  actorParams.shape = shapes[0];
  auto* firstActor = scene->CreateRigidActor(actorParams, ExpectOK{});
  actorParams.shape = shapes[1];
  auto* secondActor = scene->CreateRigidActor(actorParams, ExpectOK{});
  ActorHandle actors[kNumActors];
  for (int i = 0; i < kNumActors; ++i) {
    actorParams.shape = shapes[i % 2];
    actors[i] = scene->CreateRigidActor(actorParams, ExpectOK{})->GetHandle();
  }

  // Destroy the first two actors. This should not mess up the GridSdf generation for the others.
  scene->DestroyActor(firstActor);
  scene->DestroyActor(secondActor);

  // Step the scene
  scene->Step(0.01);

  // By this time, every actor should have a GridSdfCollider with valid data.
  // Every even actor should share the same GridSdf pointer. Every odd one should share.
  auto& reg = test::GetRegistry(scene);
  Sdf const* sdfs[kNumActors] = {};
  for (int i = 0; i < kNumActors; ++i) {
    auto e = GetEntity(reg, actors[i], ExpectOK{});
    EXPECT_TRUE(reg.all_of<CSdfCollider const>(e));
    EXPECT_FALSE(reg.any_of<CSdfColliderPending const>(e)); // Should have been removed already
    sdfs[i] = reg.get<CSdfCollider const>(e).shape.get();
    EXPECT_NE((Sdf const*)nullptr, sdfs[i]);
  }
  for (int i = 2; i < kNumActors; ++i) {
    EXPECT_EQ(sdfs[i - 2], sdfs[i]); // Expected every other actor to share the same GridSdf
  }
}

TEST(GridSdfShapeTest, PendingGridSdfTaskKeepsSurfaceMeshAlive) {
  // RequestGridSdf snapshots the surface mesh before scheduling async work. The pending task must
  // keep that mesh alive independently of the shape, because destroying a shape tears down derived
  // mesh state before GridSdfShape waits for the pending task to finish.
  class TestGridSdfShape final : public GridSdfShape {
   public:
    TestGridSdfShape(
        std::shared_ptr<TriangularMesh const> surfaceMesh,
        std::shared_ptr<std::atomic<bool>> derivedDestroyed)
        : GridSdfShape(nullptr),
          _surfaceMesh(std::move(surfaceMesh)),
          _derivedDestroyed(std::move(derivedDestroyed)) {}

    ~TestGridSdfShape() override {
      _surfaceMesh.reset();
      _derivedDestroyed->store(true);
    }

    std::shared_ptr<TriangularMesh const> const& GetSurfaceMesh() const override {
      return _surfaceMesh;
    }

    AnyShape GetBoundingVolume(Error& /*error*/) const override {
      return {};
    }

    std::optional<real> GetVolume() const override {
      return std::nullopt;
    }

    std::optional<Real3> GetCentroid() const override {
      return std::nullopt;
    }

    ModelData GetModelData(Error& error) const override {
      MOCHI_ERROR_SET(error, "Not needed for this test.");
      return {};
    }

   private:
    std::shared_ptr<TriangularMesh const> _surfaceMesh;
    std::shared_ptr<std::atomic<bool>> _derivedDestroyed;
  };

  TaskScheduler scheduler(1);

  // Keep the single worker occupied without entering the task scheduler, so the GridSdf task stays
  // queued until this test releases it.
  auto workerRunning = std::make_shared<std::atomic<bool>>(false);
  auto releaseWorker = std::make_shared<std::atomic<bool>>(false);
  scheduler.AddTask("Occupy worker", [workerRunning, releaseWorker]() {
    workerRunning->store(true);
    while (!releaseWorker->load()) {
      std::this_thread::yield();
    }
  });
  BusyWaitFor([&] { return workerRunning->load(); });

  DynamicArray<Real3> const nodes = {
      Real3{0_r, 0_r, 0_r}, Real3{1_r, 0_r, 0_r}, Real3{0_r, 1_r, 0_r}, Real3{0_r, 0_r, 1_r}};
  DynamicArray<Int3> const triangles = {Int3{0, 2, 1}, Int3{0, 1, 3}, Int3{1, 2, 3}, Int3{2, 0, 3}};
  auto surfaceMesh = std::make_shared<TriangularMesh const>(nodes, triangles);
  std::weak_ptr<TriangularMesh const> const weakMesh = surfaceMesh;

  auto derivedDestroyed = std::make_shared<std::atomic<bool>>(false);
  auto shape = std::make_unique<TestGridSdfShape>(std::move(surfaceMesh), derivedDestroyed);

  std::thread destroyer;
  MOCHI_DEFER({
    releaseWorker->store(true);
    if (destroyer.joinable()) {
      destroyer.join();
    }
  });

  GridSdfParams params;
  params.resolutionMode = GridSdfResolutionMode::Explicit;
  params.resolutionDelta = Real3{0.25_r, 0.25_r, 0.25_r};
  params.boundaryPaddingDist = 0.01_r;

  bool isPending = false;
  auto const requested = shape->RequestGridSdf(params, &isPending);
  ASSERT_EQ(nullptr, requested);
  ASSERT_TRUE(isPending);
  ASSERT_EQ(nullptr, shape->GetGridSdf());

  destroyer = std::thread([&shape]() { shape.reset(); });
  BusyWaitFor([&] { return derivedDestroyed->load(); });

  ASSERT_FALSE(weakMesh.expired())
      << "Pending GridSdf task does not own the surface mesh independently of the shape.";
}

TEST_P(MochiContextTest, GridSdfCache_TetMesh) {
  // Create two shape handles from tetrahedral meshes
  auto tetMeshParams = test::CreateMinimalTetMeshUnitCube();
  auto coordinates = Flatten(MakeSpan(tetMeshParams.first));
  auto connectivity = Flatten(MakeSpan(tetMeshParams.second));
  ShapeHandle shape1 = _mochiContext->CreateTetMeshShape(coordinates, connectivity, ExpectOK{});
  ShapeHandle shape2 = _mochiContext->CreateTetMeshShape(coordinates, connectivity, ExpectOK{});
  MOCHI_DEFER(_mochiContext->ReleaseShape(shape1));
  MOCHI_DEFER(_mochiContext->ReleaseShape(shape2));

  // Create actors and make sure that the GridSdf colliders are cached appropriately
  TestGridSdfCache(_mochiContext, shape1, shape2);
}

TEST_P(MochiContextTest, GridSdfCache_TriMesh) {
  // Create two shape handles from triangular meshes
  auto triMeshParams = test::CreateMinimalTriMeshUnitCube();
  auto coordinates = Flatten(MakeSpan(triMeshParams.first));
  auto connectivity = Flatten(MakeSpan(triMeshParams.second));
  ShapeHandle shape1 = _mochiContext->CreateTriMeshShape(coordinates, connectivity, ExpectOK{});
  ShapeHandle shape2 = _mochiContext->CreateTriMeshShape(coordinates, connectivity, ExpectOK{});
  MOCHI_DEFER(_mochiContext->ReleaseShape(shape1));
  MOCHI_DEFER(_mochiContext->ReleaseShape(shape2));

  // Create actors and make sure that the GridSdf colliders are cached appropriately
  TestGridSdfCache(_mochiContext, shape1, shape2);
}

TEST_P(MochiContextTest, LoadImplicitShape_Sphere) {
  // Create a temporary model file
  constexpr std::string_view kJsonData = R"({
    "sphere": {
      "center": [-2, 1.4, 3],
      "radius": 1.5
    }
  })";
  auto file = CreateTempFile("implicit_sphere_test", ".mochi.json", test::ExpectOK{});
  WriteFile(file.Path(), kJsonData, test::ExpectOK{});

  for (bool shouldPreload : {false, true}) {
    for (bool useModelData : {false, true}) {
      // Load with identity transform
      ShapeHandle refShape = TestLoadShapeFullPath(
          _mochiContext,
          file.Path().string(),
          Real3{1_r, 1_r, 1_r},
          {},
          shouldPreload,
          useModelData,
          ExpectOK{});
      ASSERT_TRUE(refShape.IsValid());
      MOCHI_DEFER(_mochiContext->ReleaseShape(refShape));

      // Verify the reference sphere
      auto const* contextImpl = assert_cast<ContextImpl const*>(_mochiContext);
      auto refShapePtr =
          dynamic_pointer_cast<ImplicitRigidShape const>(contextImpl->GetShapeSharedPtr(refShape));
      EXPECT_NE((ImplicitRigidShape const*)nullptr, refShapePtr.get());
      auto const* refSphere = std::get_if<Sphere>(&refShapePtr->shape);
      ASSERT_NE((Sphere const*)nullptr, refSphere);
      EXPECT_NEAR_EQ(Real3(-2_r, 1.4_r, 3_r), refSphere->GetCenter());
      EXPECT_NEAR_EQ(1.5_r, refSphere->GetRadius());

      // Test with uniform scale, rotation, and translation
      {
        real scale = 2_r;
        Real3 scale3 = Real3{scale, scale, scale};
        Quaternion rotation = Quaternion::FromRotationVector(Real3{0.5_r, 1_r, 0.3_r});
        Real3 translation{1_r, 2_r, 3_r};
        TransformRT rt{rotation, translation};
        auto transform = Dot4x4(ToVMatrix4x4(rt), VDiagonalMatrix<4>(ToSimd(scale3, 1_r)));

        ShapeHandle shape = TestLoadShapeFullPath(
            _mochiContext,
            file.Path().string(),
            scale3,
            rt,
            shouldPreload,
            useModelData,
            ExpectOK{});
        MOCHI_DEFER(_mochiContext->ReleaseShape(shape));
        auto shapePtr =
            dynamic_pointer_cast<ImplicitRigidShape const>(contextImpl->GetShapeSharedPtr(shape));
        EXPECT_NE((ImplicitRigidShape const*)nullptr, shapePtr.get());
        auto const* sphere = std::get_if<Sphere>(&shapePtr->shape);
        ASSERT_NE((Sphere const*)nullptr, sphere);

        // Center should be transformed
        EXPECT_NEAR_EQ(
            ToReal3(DotMatVec4x4(transform, ToSimd(refSphere->GetCenter(), 1_r))),
            sphere->GetCenter());
        // Radius should be scaled
        EXPECT_NEAR_EQ(scale * refSphere->GetRadius(), sphere->GetRadius());
      }
    }
  }
}

TEST_P(MochiContextTest, LoadImplicitShape_Box) {
  // Create a temporary model file
  constexpr std::string_view kJsonData = R"({
    "box": {
      "center": [1, -2, 3],
      "halfExtents": [0.6, 0.2, 0.8],
      "rotation": [0.2, 0.3, -0.1, 0.927362]
    }
  })";
  auto file = CreateTempFile("implicit_box_test", ".mochi.json", test::ExpectOK{});
  WriteFile(file.Path(), kJsonData, test::ExpectOK{});

  for (bool shouldPreload : {false, true}) {
    for (bool useModelData : {false, true}) {
      // Load with identity transform
      ShapeHandle refShape = TestLoadShapeFullPath(
          _mochiContext,
          file.Path().string(),
          Real3{1_r, 1_r, 1_r},
          {},
          shouldPreload,
          useModelData,
          ExpectOK{});
      ASSERT_TRUE(refShape.IsValid());
      MOCHI_DEFER(_mochiContext->ReleaseShape(refShape));

      // Verify the reference box
      auto const* contextImpl = assert_cast<ContextImpl const*>(_mochiContext);
      auto refShapePtr =
          dynamic_pointer_cast<ImplicitRigidShape const>(contextImpl->GetShapeSharedPtr(refShape));
      EXPECT_NE((ImplicitRigidShape const*)nullptr, refShapePtr.get());
      auto const* refBox = std::get_if<Obb>(&refShapePtr->shape);
      ASSERT_NE((Obb const*)nullptr, refBox);
      EXPECT_NEAR_EQ(Real3(1_r, -2_r, 3_r), refBox->GetCenter());
      EXPECT_NEAR_EQ(Real3(0.6_r, 0.2_r, 0.8_r), refBox->GetHalfExtents());
      EXPECT_NEAR_EQ(
          ToMatrix3x3(Quaternion(Real4{0.2_r, 0.3_r, -0.1_r, 0.927362_r})), refBox->GetRotation());

      // Test with uniform scale, rotation, and translation
      {
        real scale = 3_r;
        Real3 scale3 = Real3{scale, scale, scale};
        Quaternion rotation = Quaternion::FromRotationVector(Real3{0.5_r, 1_r, 0.3_r});
        Real3 translation{1_r, 2_r, 3_r};
        TransformRT rt{rotation, translation};
        auto transform = Dot4x4(ToVMatrix4x4(rt), VDiagonalMatrix<4>(ToSimd(scale3, 1_r)));

        ShapeHandle shape = TestLoadShapeFullPath(
            _mochiContext,
            file.Path().string(),
            scale3,
            rt,
            shouldPreload,
            useModelData,
            ExpectOK{});
        MOCHI_DEFER(_mochiContext->ReleaseShape(shape));
        auto shapePtr =
            dynamic_pointer_cast<ImplicitRigidShape const>(contextImpl->GetShapeSharedPtr(shape));
        EXPECT_NE((ImplicitRigidShape const*)nullptr, shapePtr.get());
        auto const* box = std::get_if<Obb>(&shapePtr->shape);
        ASSERT_NE((Obb const*)nullptr, box);

        // Center should be transformed
        EXPECT_NEAR_EQ(
            ToReal3(DotMatVec4x4(transform, ToSimd(refBox->GetCenter(), 1_r))), box->GetCenter());
        // Corner should be transformed
        EXPECT_NEAR_TOL(
            ToReal3(DotMatVec4x4(transform, ToSimd(refBox->GetCorner<0>(), 1_r))),
            box->GetCorner<0>(),
            1e-5_r);
      }
    }
  }
}

TEST_P(MochiContextTest, LoadImplicitShape_Plane) {
  // Create a temporary model file
  constexpr std::string_view kJsonData = R"(
  {
    "plane": {
      "distance": 2,
      "normal": [0.3, -0.8, 0.1]
    }
  })";
  auto file = CreateTempFile("implicit_plane_test", ".mochi.json", test::ExpectOK{});
  WriteFile(file.Path(), kJsonData, test::ExpectOK{});

  for (bool shouldPreload : {false, true}) {
    for (bool useModelData : {false, true}) {
      // Load with identity transform
      ShapeHandle refShape = TestLoadShapeFullPath(
          _mochiContext,
          file.Path().string(),
          Real3{1_r, 1_r, 1_r},
          {},
          shouldPreload,
          useModelData,
          ExpectOK{});
      ASSERT_TRUE(refShape.IsValid());
      MOCHI_DEFER(_mochiContext->ReleaseShape(refShape));

      // Verify the reference plane
      auto const* contextImpl = assert_cast<ContextImpl const*>(_mochiContext);
      auto refShapePtr =
          dynamic_pointer_cast<ImplicitRigidShape const>(contextImpl->GetShapeSharedPtr(refShape));
      EXPECT_NE((ImplicitRigidShape const*)nullptr, refShapePtr.get());
      auto const* refPlane = std::get_if<Plane>(&refShapePtr->shape);
      ASSERT_NE((Plane const*)nullptr, refPlane);
      EXPECT_NEAR_EQ(Normalize(Real3(0.3_r, -0.8_r, 0.1_r)), refPlane->GetNormal());
      EXPECT_NEAR_EQ(2_r, refPlane->GetDistanceFromOrigin());

      // Test with scale, rotation, and translation
      {
        Real3 scale{2_r, 3_r, 4_r};
        Quaternion rotation = Quaternion::FromRotationVector(Real3{0.5_r, 1_r, 0.3_r});
        Real3 translation{1_r, 2_r, 3_r};
        TransformRT rt{rotation, translation};
        auto transform = Dot4x4(ToVMatrix4x4(rt), VDiagonalMatrix<4>(ToSimd(scale, 1_r)));

        ShapeHandle shape = TestLoadShapeFullPath(
            _mochiContext,
            file.Path().string(),
            scale,
            rt,
            shouldPreload,
            useModelData,
            ExpectOK{});
        MOCHI_DEFER(_mochiContext->ReleaseShape(shape));
        auto shapePtr =
            dynamic_pointer_cast<ImplicitRigidShape const>(contextImpl->GetShapeSharedPtr(shape));
        EXPECT_NE((ImplicitRigidShape const*)nullptr, shapePtr.get());
        auto const* plane = std::get_if<Plane>(&shapePtr->shape);
        ASSERT_NE((Plane const*)nullptr, plane);

        // Points on the plane should be in the transformed plane
        auto testPoint = [&](Real3 const& point) {
          auto pointTransformed = ToReal3(DotMatVec4x4(transform, ToSimd(point, 1_r)));
          EXPECT_NEAR_TOL(
              Dot(plane->GetNormal(), pointTransformed) - plane->GetDistanceFromOrigin(),
              0_r,
              1e-5_r);
        };
        auto point = refPlane->GetDistanceFromOrigin() * refPlane->GetNormal();
        testPoint(point);
        Real3 tangent = {-0.7_r, 0.5_r, 0.6_r};
        tangent -= Dot(refPlane->GetNormal(), tangent) * refPlane->GetNormal();
        tangent = Normalize(tangent);
        testPoint(point + tangent);
      }
    }
  }
}

TEST(Context, EnableLogChannel) {
  // Save original state
  std::array<bool, (size_t)LogChannel::Count> wasEnabled{};
  for (size_t i = 0; i < (size_t)LogChannel::Count; ++i) {
    wasEnabled[i] = Context::IsLogChannelEnabled((LogChannel)i);
  }
  MOCHI_DEFER(for (size_t i = 0; i < (size_t)LogChannel::Count; ++i) {
    Context::EnableLogChannel((LogChannel)i, wasEnabled[i]);
  });

  // Verify defaults: Verbose disabled, everything else enabled
  EXPECT_FALSE(Context::IsLogChannelEnabled(LogChannel::Verbose));
  EXPECT_TRUE(Context::IsLogChannelEnabled(LogChannel::Info));
  EXPECT_TRUE(Context::IsLogChannelEnabled(LogChannel::Warning));
  EXPECT_TRUE(Context::IsLogChannelEnabled(LogChannel::Error));

  // Disable all, verify all disabled
  for (size_t i = 0; i < (size_t)LogChannel::Count; ++i) {
    Context::EnableLogChannel((LogChannel)i, false);
  }
  for (size_t i = 0; i < (size_t)LogChannel::Count; ++i) {
    EXPECT_FALSE(Context::IsLogChannelEnabled((LogChannel)i));
  }

  // Enable all, verify all enabled
  for (size_t i = 0; i < (size_t)LogChannel::Count; ++i) {
    Context::EnableLogChannel((LogChannel)i, true);
  }
  for (size_t i = 0; i < (size_t)LogChannel::Count; ++i) {
    EXPECT_TRUE(Context::IsLogChannelEnabled((LogChannel)i));
  }

  // Toggle each channel individually
  for (size_t i = 0; i < (size_t)LogChannel::Count; ++i) {
    Context::EnableLogChannel((LogChannel)i, false);
    EXPECT_FALSE(Context::IsLogChannelEnabled((LogChannel)i));
    Context::EnableLogChannel((LogChannel)i, true);
    EXPECT_TRUE(Context::IsLogChannelEnabled((LogChannel)i));
  }

  // Disabled channel suppresses log callback
  std::vector<LogChannel> received;
  auto prevFn = Context::GetLogCallback();
  MOCHI_DEFER(Context::SetLogCallback(prevFn));
  Context::SetLogCallback(
      [&](LogChannel ch, char const* /*msg*/, char const* /*file*/, int /*line*/) {
        received.push_back(ch);
      });
  Context::EnableLogChannel(LogChannel::Warning, false);
  MOCHI_LOG_WARNING("should not appear");
  EXPECT_TRUE(received.empty());
  MOCHI_LOG("should appear");
  EXPECT_EQ(1, received.size());
  EXPECT_EQ(LogChannel::Info, received.front());

  // Invalid channel is silently ignored (bounds check). Note this issues a warning, but warnings
  // are suppressed at this point.
  Context::EnableLogChannel((LogChannel)(int)LogChannel::Count, true); // no crash
  EXPECT_FALSE(Context::IsLogChannelEnabled((LogChannel)(int)LogChannel::Count));
}

// Verify GetShapeMesh returns correct volume mesh data (4 nodes/element) for a tet mesh shape.
TEST_P(MochiContextTest, GetShapeMesh_TetMesh) {
  auto unitCube = CreateMinimalTetMeshUnitCube();
  ShapeHandle shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(unitCube.first)), Flatten(MakeSpan(unitCube.second)), ExpectOK{});
  ASSERT_TRUE(shape.IsValid());

  MeshDataView meshView = _mochiContext->GetShapeMesh(shape, ExpectOK{});
  EXPECT_EQ(4, meshView.nodesPerElement);
  EXPECT_EQ(8, meshView.GetNumNodes());
  EXPECT_EQ(5, meshView.GetNumElements());
  EXPECT_EQ(8 * 3, isize(meshView.coordinates));
  EXPECT_EQ(5 * 4, isize(meshView.connectivity));
  EXPECT_FALSE(meshView.skinning.has_value());
}

// Verify GetShapeMesh returns correct surface mesh data (3 nodes/element) for a tri mesh shape.
TEST_P(MochiContextTest, GetShapeMesh_TriMesh) {
  auto triCube = CreateMinimalTriMeshUnitCube();
  ShapeHandle shape = _mochiContext->CreateTriMeshShape(
      Flatten(MakeSpan(triCube.first)), Flatten(MakeSpan(triCube.second)), ExpectOK{});
  ASSERT_TRUE(shape.IsValid());

  MeshDataView meshView = _mochiContext->GetShapeMesh(shape, ExpectOK{});
  EXPECT_EQ(3, meshView.nodesPerElement);
  EXPECT_EQ(isize(triCube.first), meshView.GetNumNodes());
  EXPECT_EQ(isize(triCube.second), meshView.GetNumElements());
  EXPECT_FALSE(meshView.skinning.has_value());
}

// Verify GetShapeMesh on a ModelData-created shape matches the original tet mesh dimensions.
TEST_P(MochiContextTest, GetShapeMesh_ModelDataMatchesTetMesh) {
  auto unitCube = CreateMinimalTetMeshUnitCube();

  ModelData model;
  model.mesh.emplace();
  model.mesh->nodesPerElement = 4;
  model.mesh->coordinates = Flatten(MakeSpan(unitCube.first));
  model.mesh->connectivity = Flatten(MakeSpan(unitCube.second));

  ShapeHandle shape = _mochiContext->CreateModelShape(model, ExpectOK{});
  ASSERT_TRUE(shape.IsValid());

  MeshDataView meshView = _mochiContext->GetShapeMesh(shape, ExpectOK{});
  EXPECT_EQ(model.mesh->nodesPerElement, meshView.nodesPerElement);
  EXPECT_EQ(model.mesh->GetNumNodes(), meshView.GetNumNodes());
  EXPECT_EQ(model.mesh->GetNumElements(), meshView.GetNumElements());
}

// Verify GetShapeMesh reports an error for a default-constructed (invalid) handle.
TEST_P(MochiContextTest, GetShapeMesh_InvalidHandle) {
  [[maybe_unused]] auto mesh = _mochiContext->GetShapeMesh(ShapeHandle{}, ExpectNotOK{});
}

// Verify GetShapeMesh reports an error when the shape handle has already been released.
TEST_P(MochiContextTest, GetShapeMesh_ReleasedHandle) {
  auto unitCube = CreateMinimalTetMeshUnitCube();
  ShapeHandle shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(unitCube.first)), Flatten(MakeSpan(unitCube.second)), ExpectOK{});
  _mochiContext->ReleaseShape(shape);
  [[maybe_unused]] auto mesh = _mochiContext->GetShapeMesh(shape, ExpectNotOK{});
}

// Verify GetShapeMesh returns an empty view for non-mesh shapes (sphere).
TEST_P(MochiContextTest, GetShapeMesh_SphereShapeReturnsEmpty) {
  ShapeHandle shape = _mochiContext->CreateSphereShape(Real3{}, 1_r, ExpectOK{});
  ASSERT_TRUE(shape.IsValid());
  auto const mesh = _mochiContext->GetShapeMesh(shape, ExpectOK{});
  EXPECT_EQ(MeshDataView{}, mesh);
}

// Verify GetShapeMesh returns polyline data (2 nodes/element) with explicit sequential
// connectivity that encodes the polyline's open/closed-loop topology.
TEST_P(MochiContextTest, GetShapeMesh_Polyline) {
  DynamicArray<Real3> nodes;
  nodes.push_back({0_r, 0_r, 0_r});
  nodes.push_back({0.1_r, 0_r, 0_r});
  nodes.push_back({0.2_r, 0_r, 0_r});

  DynamicArray<Real3> elementFrameAxes;
  elementFrameAxes.push_back(Real3{0_r, 1_r, 0_r});
  elementFrameAxes.push_back(Real3{0_r, 1_r, 0_r});

  ShapeHandle shape = experimental::CreatePolylineShape(
      _mochiContext,
      MakeSpan(nodes),
      MakeSpan(elementFrameAxes),
      /*isClosedLoop=*/false,
      ExpectOK{});
  ASSERT_TRUE(shape.IsValid());

  MeshDataView meshView = _mochiContext->GetShapeMesh(shape, ExpectOK{});
  EXPECT_EQ(2, meshView.nodesPerElement);
  EXPECT_EQ(3, meshView.GetNumNodes());
  EXPECT_EQ(3 * 3, isize(meshView.coordinates));
  // Open polyline: 2*(numNodes-1) connectivity entries.
  ASSERT_EQ(2 * (isize(nodes) - 1), isize(meshView.connectivity));
  EXPECT_EQ(2, meshView.GetNumElements());
  EXPECT_FALSE(meshView.skinning.has_value());
}

// Verify GetShapeSurfaceMesh returns a non-empty triangle surface for a tet mesh shape.
TEST_P(MochiContextTest, GetShapeSurfaceMesh_TetMesh) {
  auto unitCube = CreateMinimalTetMeshUnitCube();
  ShapeHandle shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(unitCube.first)), Flatten(MakeSpan(unitCube.second)), ExpectOK{});
  ASSERT_TRUE(shape.IsValid());

  MeshDataView surfaceView = _mochiContext->GetShapeSurfaceMesh(shape, ExpectOK{});
  EXPECT_EQ(3, surfaceView.nodesPerElement);
  EXPECT_GT(surfaceView.GetNumNodes(), 0);
  EXPECT_GT(surfaceView.GetNumElements(), 0);
  EXPECT_FALSE(surfaceView.skinning.has_value());
}

// Verify GetShapeSurfaceMesh for a tri mesh returns equivalent data to GetShapeMesh.
TEST_P(MochiContextTest, GetShapeSurfaceMesh_TriMeshSameAsMainMesh) {
  auto triCube = CreateMinimalTriMeshUnitCube();
  ShapeHandle shape = _mochiContext->CreateTriMeshShape(
      Flatten(MakeSpan(triCube.first)), Flatten(MakeSpan(triCube.second)), ExpectOK{});
  ASSERT_TRUE(shape.IsValid());

  MeshDataView meshView = _mochiContext->GetShapeMesh(shape, ExpectOK{});
  MeshDataView surfaceView = _mochiContext->GetShapeSurfaceMesh(shape, ExpectOK{});
  EXPECT_EQ(meshView.nodesPerElement, surfaceView.nodesPerElement);
  EXPECT_EQ(meshView.GetNumNodes(), surfaceView.GetNumNodes());
  EXPECT_EQ(meshView.GetNumElements(), surfaceView.GetNumElements());
  EXPECT_SPAN_EQ(meshView.coordinates, surfaceView.coordinates);
  EXPECT_SPAN_EQ(meshView.connectivity, surfaceView.connectivity);
}

TEST_P(MochiContextTest, GetShapeSurfaceMesh_TriMeshOmitsUnreferencedNodes) {
  constexpr std::array kCoordinates = {
      Real3{10_r, 10_r, 10_r}, Real3{0_r, 0_r, 0_r}, Real3{1_r, 0_r, 0_r}, Real3{0_r, 1_r, 0_r}};
  constexpr std::array kConnectivity = {Int3{1, 2, 3}};
  ShapeHandle shape = _mochiContext->CreateTriMeshShape(
      Flatten(MakeConstSpan(kCoordinates)), Flatten(MakeConstSpan(kConnectivity)), ExpectOK{});
  EXPECT_TRUE(shape.IsValid());

  MeshDataView meshView = _mochiContext->GetShapeMesh(shape, ExpectOK{});
  MeshDataView surfaceView = _mochiContext->GetShapeSurfaceMesh(shape, ExpectOK{});
  EXPECT_EQ(4, meshView.GetNumNodes());
  EXPECT_EQ(3, surfaceView.GetNumNodes());
  constexpr std::array kExpectedCoordinates = {kCoordinates[1], kCoordinates[2], kCoordinates[3]};
  constexpr std::array kExpectedConnectivity = {0, 1, 2};
  EXPECT_SPAN_EQ(Flatten(MakeConstSpan(kExpectedCoordinates)), surfaceView.coordinates);
  EXPECT_SPAN_EQ(MakeConstSpan(kExpectedConnectivity), surfaceView.connectivity);
}

// Verify GetShapeSurfaceMesh reports an error for a default-constructed (invalid) handle.
TEST_P(MochiContextTest, GetShapeSurfaceMesh_InvalidHandle) {
  [[maybe_unused]] auto mesh = _mochiContext->GetShapeSurfaceMesh(ShapeHandle{}, ExpectNotOK{});
}

// Verify GetShapeSurfaceMesh returns an empty view for valid shapes without surface mesh data.
TEST_P(MochiContextTest, GetShapeSurfaceMesh_SphereShapeReturnsEmpty) {
  ShapeHandle shape = _mochiContext->CreateSphereShape(Real3{}, 1_r, ExpectOK{});
  ASSERT_TRUE(shape.IsValid());
  auto const mesh = _mochiContext->GetShapeSurfaceMesh(shape, ExpectOK{});
  EXPECT_EQ(MeshDataView{}, mesh);
}

// Verify GetShapeSurfaceMesh reports an error when the shape handle has already been released.
TEST_P(MochiContextTest, GetShapeSurfaceMesh_ReleasedHandle) {
  auto unitCube = CreateMinimalTetMeshUnitCube();
  ShapeHandle shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(unitCube.first)), Flatten(MakeSpan(unitCube.second)), ExpectOK{});
  _mochiContext->ReleaseShape(shape);
  [[maybe_unused]] auto mesh = _mochiContext->GetShapeSurfaceMesh(shape, ExpectNotOK{});
}

static ModelData CreateModelWithVisualMesh() {
  auto unitCube = CreateMinimalTetMeshUnitCube();
  auto triMesh = CreateMinimalTriMeshUnitCube();
  ModelData model;
  model.mesh.emplace();
  model.mesh->nodesPerElement = 4;
  model.mesh->coordinates = Flatten(MakeSpan(unitCube.first));
  model.mesh->connectivity = Flatten(MakeSpan(unitCube.second));
  model.visualMesh.emplace();
  model.visualMesh->nodesPerElement = 3;
  model.visualMesh->coordinates = Flatten(MakeSpan(triMesh.first));
  model.visualMesh->connectivity = Flatten(MakeSpan(triMesh.second));
  return model;
}

// Verify GetShapeVisualMesh returns correct visual mesh dimensions from a ModelData shape.
TEST_P(MochiContextTest, GetShapeVisualMesh_WithVisualMesh) {
  ModelData model = CreateModelWithVisualMesh();
  ShapeHandle shape = _mochiContext->CreateModelShape(model, ExpectOK{});
  ASSERT_TRUE(shape.IsValid());

  MeshDataView visualView = _mochiContext->GetShapeVisualMesh(shape, ExpectOK{});
  EXPECT_EQ(3, visualView.nodesPerElement);
  EXPECT_EQ(model.visualMesh->GetNumNodes(), visualView.GetNumNodes());
  EXPECT_EQ(model.visualMesh->GetNumElements(), visualView.GetNumElements());
}

TEST_P(MochiContextTest, GetShapeVisualMesh_PreservesUnreferencedNodes) {
  ModelData model = CreateModelWithVisualMesh();
  constexpr std::array kCoordinates = {
      Real3{10_r, 10_r, 10_r}, Real3{0_r, 0_r, 0_r}, Real3{1_r, 0_r, 0_r}, Real3{0_r, 1_r, 0_r}};
  constexpr std::array kConnectivity = {Int3{1, 2, 3}};
  model.visualMesh->coordinates = Flatten(MakeConstSpan(kCoordinates));
  model.visualMesh->connectivity = Flatten(MakeConstSpan(kConnectivity));

  ShapeHandle shape = _mochiContext->CreateModelShape(model, ExpectOK{});
  EXPECT_TRUE(shape.IsValid());

  MeshDataView visualView = _mochiContext->GetShapeVisualMesh(shape, ExpectOK{});
  EXPECT_EQ(4, visualView.GetNumNodes());
  EXPECT_SPAN_EQ(Flatten(MakeConstSpan(kCoordinates)), visualView.coordinates);
  EXPECT_SPAN_EQ(Flatten(MakeConstSpan(kConnectivity)), visualView.connectivity);
}

// Verify GetShapeVisualMesh populates skinning data (indices, weights) when the visual mesh
// has a linear mesh embedding.
TEST_P(MochiContextTest, GetShapeVisualMesh_WithSkinning) {
  ModelData model = CreateModelWithVisualMesh();
  int const numVisualNodes = model.visualMesh->GetNumNodes();
  int constexpr kWeightsPerNode = 1;
  model.visualMesh->skinning.emplace();
  model.visualMesh->skinning->weightsPerNode = kWeightsPerNode;
  DynamicArray<int> indices(numVisualNodes * kWeightsPerNode);
  DynamicArray<real> weights(numVisualNodes * kWeightsPerNode);
  for (int i = 0; i < numVisualNodes; ++i) {
    indices[i] = i % 8;
    weights[i] = 1_r;
  }
  model.visualMesh->skinning->indices = std::move(indices);
  model.visualMesh->skinning->weights = std::move(weights);

  ShapeHandle shape = _mochiContext->CreateModelShape(model, ExpectOK{});
  ASSERT_TRUE(shape.IsValid());

  MeshDataView visualView = _mochiContext->GetShapeVisualMesh(shape, ExpectOK{});
  EXPECT_EQ(3, visualView.nodesPerElement);
  EXPECT_EQ(numVisualNodes, visualView.GetNumNodes());
  ASSERT_TRUE(visualView.skinning.has_value());
  EXPECT_EQ(kWeightsPerNode, visualView.skinning->weightsPerNode);
  EXPECT_EQ(numVisualNodes * kWeightsPerNode, isize(visualView.skinning->indices));
  EXPECT_EQ(numVisualNodes * kWeightsPerNode, isize(visualView.skinning->weights));
}

// Verify GetShapeVisualMesh returns an empty view for a tet mesh shape with no visual mesh
// attached.
TEST_P(MochiContextTest, GetShapeVisualMesh_NoVisualMeshReturnsEmpty) {
  auto unitCube = CreateMinimalTetMeshUnitCube();
  ShapeHandle shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(unitCube.first)), Flatten(MakeSpan(unitCube.second)), ExpectOK{});
  ASSERT_TRUE(shape.IsValid());
  auto const mesh = _mochiContext->GetShapeVisualMesh(shape, ExpectOK{});
  EXPECT_EQ(MeshDataView{}, mesh);
}

// Verify GetShapeVisualMesh reports an error for a default-constructed (invalid) handle.
TEST_P(MochiContextTest, GetShapeVisualMesh_InvalidHandle) {
  [[maybe_unused]] auto mesh = _mochiContext->GetShapeVisualMesh(ShapeHandle{}, ExpectNotOK{});
}

// Verify GetShapeVisualMesh returns an empty view for valid shapes without visual mesh data.
TEST_P(MochiContextTest, GetShapeVisualMesh_SphereShapeReturnsEmpty) {
  ShapeHandle shape = _mochiContext->CreateSphereShape(Real3{}, 1_r, ExpectOK{});
  ASSERT_TRUE(shape.IsValid());
  auto const mesh = _mochiContext->GetShapeVisualMesh(shape, ExpectOK{});
  EXPECT_EQ(MeshDataView{}, mesh);
}

// Verify GetShapeVisualMesh reports an error when the shape handle has already been released.
TEST_P(MochiContextTest, GetShapeVisualMesh_ReleasedHandle) {
  ModelData model = CreateModelWithVisualMesh();
  ShapeHandle shape = _mochiContext->CreateModelShape(model, ExpectOK{});
  _mochiContext->ReleaseShape(shape);
  [[maybe_unused]] auto mesh = _mochiContext->GetShapeVisualMesh(shape, ExpectNotOK{});
}

static ModelData CreatePolylineModelWithVisualMesh() {
  auto triMesh = CreateMinimalTriMeshSingleTri();
  int const numVisualNodes = isize(triMesh.first);

  ModelData model;
  model.mesh.emplace();
  model.mesh->nodesPerElement = 2;
  model.mesh->coordinates = Flatten(MakeSpan(triMesh.first));

  model.visualMesh.emplace();
  model.visualMesh->nodesPerElement = 3;
  model.visualMesh->coordinates = Flatten(MakeSpan(triMesh.first));
  model.visualMesh->connectivity = Flatten(MakeSpan(triMesh.second));

  // Skinning is required for ComputeRodVisualMeshEmbedding to attach the visual mesh.
  model.visualMesh->skinning.emplace();
  model.visualMesh->skinning->weightsPerNode = 1;
  model.visualMesh->skinning->indices.resize(numVisualNodes, 0);
  model.visualMesh->skinning->weights.resize(numVisualNodes, 1_r);

  return model;
}

// Verify GetShapeVisualMesh returns correct visual mesh dimensions for a polyline shape.
TEST_P(MochiContextTest, GetShapeVisualMesh_PolylineWithVisualMesh) {
  ModelData model = CreatePolylineModelWithVisualMesh();
  ShapeHandle shape = _mochiContext->CreateModelShape(model, ExpectOK{});
  ASSERT_TRUE(shape.IsValid());

  MeshDataView visualView = _mochiContext->GetShapeVisualMesh(shape, ExpectOK{});
  EXPECT_EQ(3, visualView.nodesPerElement);
  EXPECT_EQ(model.visualMesh->GetNumNodes(), visualView.GetNumNodes());
  EXPECT_EQ(model.visualMesh->GetNumElements(), visualView.GetNumElements());
  EXPECT_FALSE(visualView.skinning.has_value());
}

// Verify GetShapeVisualMesh returns an empty view for a polyline shape with no visual mesh
// attached.
TEST_P(MochiContextTest, GetShapeVisualMesh_PolylineNoVisualMeshReturnsEmpty) {
  DynamicArray<Real3> nodes;
  nodes.push_back({0_r, 0_r, 0_r});
  nodes.push_back({0.1_r, 0_r, 0_r});
  nodes.push_back({0.2_r, 0_r, 0_r});

  DynamicArray<Real3> elementFrameAxes;
  elementFrameAxes.push_back(Real3{0_r, 1_r, 0_r});
  elementFrameAxes.push_back(Real3{0_r, 1_r, 0_r});

  ShapeHandle shape = experimental::CreatePolylineShape(
      _mochiContext, MakeSpan(nodes), MakeSpan(elementFrameAxes), false, ExpectOK{});
  ASSERT_TRUE(shape.IsValid());
  auto const mesh = _mochiContext->GetShapeVisualMesh(shape, ExpectOK{});
  EXPECT_EQ(MeshDataView{}, mesh);
}
