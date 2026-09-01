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

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/span.h>
#include <mochi_physics/src/mochi_scene.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include "mochi_physics_test_fixture.h"
#include "mochi_prefab_export_test_helpers.h"

#if MOCHI_INTERNAL
#include "internal/franka_hand_prefab_fixtures.h"
#endif

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace mochi;
using namespace mochi::test;

// The prefab assets used by these tests are not shipped externally.
#if MOCHI_USE_HDF5 && MOCHI_INTERNAL
#define MOCHI_HDF5_AND_INTERNAL 1
#else
#define MOCHI_HDF5_AND_INTERNAL 0
#endif

// Validates that exported and original actor counts match
template <typename T>
static void ValidateActorCounts(DynamicArray<T> const& exported, DynamicArray<T> const& original) {
  EXPECT_EQ(exported.size(), original.size());
}

// Comprehensive comparison function for RigidActorPrefab
static void ExpectEqual(
    prefab::RigidActorPrefab const& expected,
    prefab::RigidActorPrefab const& actual) {
  // Compare RigidActorParams
  EXPECT_STREQ(expected.name.c_str(), actual.name.c_str());
  EXPECT_STREQ(expected.layer.c_str(), actual.layer.c_str());
  EXPECT_EQ(expected.colliderType, actual.colliderType);
  EXPECT_EQ(expected.isStatic, actual.isStatic);

  // Compare comment (optional field)
  EXPECT_EQ(expected.comment, actual.comment);

  // NOTE: shapeFile changes during export (original -> generated file), so we don't compare it
  // NOTE: scale changes during export (gets baked into mesh, reset to [1,1,1]), so we don't compare
  // it

  // Compare shape transform fields that are preserved
  EXPECT_EQ(expected.shapeRotation, actual.shapeRotation);
  EXPECT_EQ(expected.shapeTranslation, actual.shapeTranslation);

  // Compare world transform fields
  EXPECT_EQ(expected.rotation, actual.rotation) << "Rotation should be preserved during export";
  EXPECT_EQ(expected.translation, actual.translation);
}

// Comprehensive comparison function for SoftActorPrefab
static void ExpectEqual(
    prefab::SoftActorPrefab const& expected,
    prefab::SoftActorPrefab const& actual) {
  // Compare SoftActorParams
  EXPECT_STREQ(expected.name.c_str(), actual.name.c_str());
  EXPECT_STREQ(expected.layer.c_str(), actual.layer.c_str());
  EXPECT_EQ(expected.hasGravity, actual.hasGravity);
  EXPECT_EQ(expected.hasInertia, actual.hasInertia);
  EXPECT_EQ(expected.hasStress, actual.hasStress);

  // Compare material properties
  EXPECT_EQ(expected.material, actual.material);

  // Compare comment (optional field)
  EXPECT_EQ(expected.comment, actual.comment);

  // NOTE: shapeFile changes during export (original -> generated file), so we don't compare it
  // NOTE: flowFile may change during export, so we don't compare it
  // NOTE: scale changes during export (gets baked into mesh, reset to [1,1,1]), so we don't compare
  // it

  // Compare shape transform fields that are preserved
  EXPECT_EQ(expected.shapeRotation, actual.shapeRotation);
  EXPECT_EQ(expected.shapeTranslation, actual.shapeTranslation);

  // Compare world transform fields
  EXPECT_EQ(expected.rotation, actual.rotation) << "Rotation should be preserved during export";
  EXPECT_EQ(expected.translation, actual.translation);
}

// Inline test prefabs

// Prefab for testing rigid actor export
static constexpr char const* kRigidMinimalCubeStackJson = R"({
  "actors": {
    "rigid": [
      {
        "layer": "Object",
        "name": "BottomCube",
        "colliderType": "Box",
        "scale": [0.1, 0.1, 0.1],
        "shape": "cube/cube_minimal.mochi.json",
        "translation": [0, 0.05, 0]
      },
      {
        "layer": "Object",
        "name": "MiddleCube",
        "colliderType": "Box",
        "scale": [0.1, 0.1, 0.1],
        "shape": "cube/cube_minimal.mochi.json",
        "translation": [0, 0.4, 0]
      },
      {
        "layer": "Object",
        "name": "TopCube",
        "colliderType": "Box",
        "scale": [0.1, 0.1, 0.1],
        "shape": "cube/cube_minimal.mochi.json",
        "translation": [0, 0.75, 0]
      }
    ]
  },
  "scene": {
    "description": "Stack of 3 rigid cubes using the minimal cube mesh."
  }
})";

// Prefab for testing mixed rigid/soft actor export
static constexpr char const* kSoftArmadilloOnRigidCubeJson = R"({
  "actors": {
    "rigid": [
      {
        "layer": "Object",
        "name": "Box",
        "scale": [0.1, 0.1, 0.1],
        "shape": "cube/cube_fine_mesh.mochi.json",
        "translation": [-0.05, 0.9, -0.05]
      },
      {
        "isStatic": true,
        "layer": "Environment",
        "name": "StaticCube",
        "rotation": [0.57735, 0.57735, 0.57735, 0],
        "scale": [0.1, 0.1, 0.1],
        "shape": "cube/cube_fine_mesh.mochi.json",
        "translation": [-0.05, 0.01, -0.05]
      }
    ],
    "soft": [
      {
        "layer": "Object",
        "material": {
          "type": "NeoHookean",
          "neoHookean": {
            "poissonRatio": 0.3,
            "youngsModulus": 10000
          }
        },
        "name": "Armadillo",
        "scale": [0.2, 0.2, 0.2],
        "shape": "armadillo/armadillo_coarse_mesh.mochi.json",
        "translation": [-0.1, 0.5, -0.1]
      }
    ]
  },
  "scene": {
    "description": "Soft armadillo falling on static rigid cube"
  }
})";

// Prefab for testing soft actor export
static constexpr char const* kSoftSphereOnRigidCubeJson = R"({
  "actors": {
    "rigid": [
      {
        "layer": "Object",
        "name": "Box",
        "colliderType": "Box",
        "scale": [0.4, 0.4, 0.4],
        "shape": "cube/cube_fine_mesh.mochi.json",
        "translation": [-0.2, 0.01, -0.2]
      }
    ],
    "soft": [
      {
        "layer": "Object",
        "material": {
          "type": "NeoHookean",
          "neoHookean": {
            "poissonRatio": 0.3,
            "youngsModulus": 10000
          }
        },
        "name": "Sphere",
        "scale": [0.1, 0.1, 0.1],
        "shape": "sphere/icosphere_4subdiv.1.mochi.json",
        "translation": [0, 0.52, 0]
      }
    ]
  },
  "scene": {
    "description": "Soft sphere on a dynamic rigid box on a rigid ground plane."
  }
})";

static Actor* CreateRigidCubeActor(Context* context, Scene* scene, char const* name) {
  auto shapeHandle = context->LoadShapeFromFile(
      GetAssetPath("cube/cube_minimal.mochi.json"),
      Real3{0.1_r, 0.1_r, 0.1_r},
      TransformRT{Quaternion::Identity(), Real3{0_r, 0_r, 0_r}},
      ExpectOK{});

  RigidActorParams params;
  params.name = name;
  params.shape = shapeHandle;
  params.colliderType = ColliderType::Box;
  params.worldFromLocal = TransformRT{Quaternion::Identity(), Real3{0_r, 0_r, 0_r}};
  return scene->CreateRigidActor(params, ExpectOK{});
}

[[nodiscard]] static prefab::ArticulatedActorPrefab MakeSingleLinkRobot(
    char const* actorName,
    char const* linkName = "boneA") {
  prefab::ArticulatedActorPrefab robot;
  robot.name = actorName;
  robot.joints.resize(1);
  robot.joints[0].type = ArticulatedJointType::Free;
  robot.links.resize(1);
  robot.links[0].name = linkName;
  robot.links[0].parentLink = -1;
  robot.links[0].shapeFile = "articulated/two_links_revolute/bone_a.mochi.h5";
  robot.links[0].colliderType = ColliderType::Box;
  return robot;
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, ExportScene_RigidActors) {
  // Comprehensive test for rigid actor export with physics properties
  auto tempDir = CreateTempDirectory("export_rigid_minimal_cube_stack", ExpectOK{});
  auto originalPrefabPath = CreateTempPrefabFile(
      kRigidMinimalCubeStackJson, "rigid_minimal_cube_stack.mochi_scene", tempDir.Path());
  auto originalPrefab = prefab::ShallowLoadFromFile(originalPrefabPath.string(), ExpectOK{});

  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));

  auto* scene = context->CreateScene("test_export");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::AddToScene(
      originalPrefabPath.string(), GetAssetsDir(), scene, prefab::PrefabParams{}, ExpectOK{});

  auto prefabFile = ExportScene(scene, "rigid_minimal_cube_stack", tempDir.Path());
  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});

  // Validate rigid actors and their properties
  ValidateActorCounts(exportedPrefab.actors.rigid, originalPrefab.actors.rigid);
  EXPECT_EQ(exportedPrefab.actors.rigid.size(), 3);

  // Sort original actors by name to match the alphabetical sorting done during export
  auto sortedOriginal = originalPrefab.actors.rigid;
  std::sort(sortedOriginal.begin(), sortedOriginal.end(), [](auto const& a, auto const& b) {
    return a.name < b.name;
  });

  // Actors were sorted alphabetically
  for (size_t i = 0; i < exportedPrefab.actors.rigid.size(); ++i) {
    auto const& original = sortedOriginal[i];
    auto const& exported = exportedPrefab.actors.rigid[i];

    // Comprehensive validation of ALL fields
    ExpectEqual(original, exported);
    ValidateShapeFile(std::string(exported.shapeFile), tempDir.Path(), prefabFile, "Rigid actor");
  }

  TestRoundTrip(scene, context, prefabFile, tempDir.Path());
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, ExportScene_SoftActors) {
  // Comprehensive test for soft actor export with material properties
  auto tempDir = CreateTempDirectory("export_soft_armadillo_on_rigid_cube", ExpectOK{});
  auto originalPrefabPath = CreateTempPrefabFile(
      kSoftArmadilloOnRigidCubeJson, "soft_armadillo_on_rigid_cube.mochi_scene", tempDir.Path());
  auto originalPrefab = prefab::ShallowLoadFromFile(originalPrefabPath.string(), ExpectOK{});

  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));

  auto* scene = context->CreateScene("test_export");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::AddToScene(
      originalPrefabPath.string(), GetAssetsDir(), scene, prefab::PrefabParams{}, ExpectOK{});

  auto prefabFile = ExportScene(scene, "soft_armadillo_on_rigid_cube", tempDir.Path());
  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});

  // Validate soft actors and their material properties
  ValidateActorCounts(exportedPrefab.actors.soft, originalPrefab.actors.soft);
  ValidateActorsExist(exportedPrefab.actors.soft);

  // Actors were sorted alphabetically
  for (size_t i = 0; i < exportedPrefab.actors.soft.size(); ++i) {
    auto const& original = originalPrefab.actors.soft[i];
    auto const& exported = exportedPrefab.actors.soft[i];

    // Comprehensive validation of ALL fields
    ExpectEqual(original, exported);
    ValidateShapeFile(std::string(exported.shapeFile), tempDir.Path(), prefabFile, "Soft actor");
  }

  TestRoundTrip(scene, context, prefabFile, tempDir.Path());
}

TEST_IF(MOCHI_USE_HDF5, PrefabExport, ExportScene_ExportMixedActors) {
  // Comprehensive test for mixed scene with rigids, softs, and articulated actors
  auto tempDir = CreateTempDirectory("export_soft_sphere_on_rigid_cube", ExpectOK{});
  auto originalPrefabPath = CreateTempPrefabFile(
      kSoftSphereOnRigidCubeJson, "soft_sphere_on_rigid_cube.mochi_scene", tempDir.Path());
  auto originalPrefab = prefab::ShallowLoadFromFile(originalPrefabPath.string(), ExpectOK{});

  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));

  auto* scene = context->CreateScene("test_export");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::AddToScene(
      originalPrefabPath.string(), GetAssetsDir(), scene, prefab::PrefabParams{}, ExpectOK{});

  auto prefabFile = ExportScene(scene, "soft_sphere_on_rigid_cube", tempDir.Path());
  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});

  // Validate all actor types are present and exported correctly
  ValidateActorCounts(exportedPrefab.actors.rigid, originalPrefab.actors.rigid);
  ValidateActorCounts(exportedPrefab.actors.soft, originalPrefab.actors.soft);
  ValidateActorsExist(exportedPrefab.actors.rigid);
  ValidateActorsExist(exportedPrefab.actors.soft);

  // Validate rigid actors
  for (size_t i = 0; i < exportedPrefab.actors.rigid.size(); ++i) {
    auto const& original = originalPrefab.actors.rigid[i];
    auto const& exported = exportedPrefab.actors.rigid[i];

    ExpectEqual(original, exported);
    ValidateShapeFile(std::string(exported.shapeFile), tempDir.Path(), prefabFile, "Rigid actor");
  }

  // Validate soft actors
  for (size_t i = 0; i < exportedPrefab.actors.soft.size(); ++i) {
    auto const& original = originalPrefab.actors.soft[i];
    auto const& exported = exportedPrefab.actors.soft[i];

    ExpectEqual(original, exported);
    ValidateShapeFile(std::string(exported.shapeFile), tempDir.Path(), prefabFile, "Soft actor");
  }

  TestRoundTrip(scene, context, prefabFile, tempDir.Path());
}

// Single-rigid variant of ExportActor_RoundTrip — exercises the rigid branch of the
// dispatcher added when ExportActor was generalized beyond articulated actors.
TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, ExportActor_SingleRigid) {
  static constexpr char const* kSingleRigidCubeJson = R"({
    "actors": {
      "rigid": [
        {
          "layer": "Object",
          "name": "OnlyCube",
          "colliderType": "Box",
          "scale": [0.1, 0.1, 0.1],
          "shape": "cube/cube_minimal.mochi.json",
          "translation": [0, 0.5, 0]
        }
      ]
    }
  })";

  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));

  auto* scene = context->CreateScene("test_export_actor_rigid");
  MOCHI_DEFER(context->DestroyScene(scene));

  auto tempDir = CreateTempDirectory("export_actor_single_rigid", ExpectOK{});
  auto prefabScenePath =
      CreateTempPrefabFile(kSingleRigidCubeJson, "single_rigid.mochi_scene", tempDir.Path());

  auto const loadResult = prefab::AddToScene(
      prefabScenePath.string(), GetAssetsDir(), scene, prefab::PrefabParams{}, ExpectOK{});
  auto const rigidActors = loadResult.Filter(ActorType::Rigid);
  ASSERT_EQ(rigidActors.size(), 1);

  prefab::ExportActor(rigidActors[0], "exported_rigid", tempDir.Path().string(), ExpectOK{});
  std::filesystem::path const prefabFile =
      tempDir.Path() / "exported_rigid" / "exported_rigid.mochi_scene";
  ASSERT_TRUE(std::filesystem::exists(prefabFile));

  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});
  ASSERT_EQ(exportedPrefab.actors.rigid.size(), 1);
  EXPECT_EQ(exportedPrefab.actors.soft.size(), 0);
  EXPECT_EQ(exportedPrefab.actors.articulated.size(), 0);
  EXPECT_EQ(exportedPrefab.actors.softSkinned.size(), 0);
  // ExportActor rewrites the actor name to the export name (consistent with how the
  // single-actor articulated round-trip behaves), so we only assert it is non-empty here.
  EXPECT_FALSE(exportedPrefab.actors.rigid[0].name.empty());
  ValidateShapeFile(
      std::string(exportedPrefab.actors.rigid[0].shapeFile),
      tempDir.Path(),
      prefabFile,
      "Single rigid actor");
}

TEST(PrefabExport, ExportActor_NullActorReportsError) {
  prefab::ExportActor(nullptr, "unused", "unused", ExpectNotOK{});
}

TEST(PrefabExport, ExportActor_UnsupportedActorTypeLeavesNoOutputDirectory) {
  // ExportActor rejects actor types it cannot serialize (e.g. shell). The rejection must happen
  // before any output directory is created, so a failed export leaves nothing on disk.
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  auto* scene = context->CreateScene("export_actor_unsupported_type_test");
  MOCHI_DEFER(context->DestroyScene(scene));

  auto&& [coords, tris] = CreateMinimalTriMeshUnitCube();
  experimental::ShellActorParams shellParams;
  shellParams.name = "Shell";
  shellParams.shape = context->CreateTriMeshShape(
      Flatten(MakeSpan(coords)), Flatten(MakeSpan(tris)), ErrorAssert{});
  auto* shell = experimental::CreateShellActor(scene, shellParams, ExpectOK{});

  auto tempDir = CreateTempDirectory("export_actor_unsupported_type", ExpectOK{});
  prefab::ExportActor(shell, "shell_export", tempDir.Path().string(), ExpectNotOK{});

  // The rejected export must not leave a stray output directory behind.
  EXPECT_FALSE(std::filesystem::exists(tempDir.Path() / "shell_export"));
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, ExportScene_MeshDeduplication) {
  Context* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));

  Scene* scene = context->CreateScene("deduplication_test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Load one shape that will be shared between multiple actors
  auto sharedShapeHandle = context->LoadShapeFromFile(
      GetAssetPath("duck/duck_coarse_mesh.mochi.json"),
      Real3{1_r, 1_r, 1_r}, // Scale 1x1x1
      TransformRT{Quaternion::Identity(), Real3{0_r, 0_r, 0_r}}, // No rotation/translation
      ExpectOK{});
  // Load another shape with different parameters (creates different Shape* pointer)
  auto differentShapeHandle = context->LoadShapeFromFile(
      GetAssetPath("duck/duck_coarse_mesh.mochi.json"),
      Real3{2_r, 2_r, 2_r}, // Different scale -> different Shape* pointer
      TransformRT{Quaternion::Identity(), Real3{0_r, 0_r, 0_r}},
      ExpectOK{});
  // Create 3 actors that all use the same Shape* pointer (sharedShapeHandle)
  RigidActorParams params1;
  params1.name = "duck_shared_1";
  params1.shape = sharedShapeHandle; // Same Shape* pointer
  params1.worldFromLocal = TransformRT{Quaternion::Identity(), Real3{0_r, 0_r, 0_r}};
  scene->CreateRigidActor(params1, ExpectOK{});

  RigidActorParams params2;
  params2.name = "duck_shared_2";
  params2.shape = sharedShapeHandle; // Same Shape* pointer (should share mesh file)
  params2.worldFromLocal = TransformRT{Quaternion::Identity(), Real3{5_r, 0_r, 0_r}};
  scene->CreateRigidActor(params2, ExpectOK{});

  RigidActorParams params3;
  params3.name = "duck_shared_3";
  params3.shape = sharedShapeHandle; // Same Shape* pointer (should share mesh file)
  params3.worldFromLocal = TransformRT{Quaternion::Identity(), Real3{10_r, 0_r, 0_r}};
  scene->CreateRigidActor(params3, ExpectOK{});

  // Create 1 actor with a different Shape* pointer
  RigidActorParams params4;
  params4.name = "duck_different";
  params4.shape = differentShapeHandle; // Different Shape* pointer (separate mesh file)
  params4.worldFromLocal = TransformRT{Quaternion::Identity(), Real3{15_r, 0_r, 0_r}};
  scene->CreateRigidActor(params4, ExpectOK{});

  // Export the scene
  auto tempDir = CreateTempDirectory("deduplication_test", ExpectOK{});
  auto prefabFile = ExportScene(scene, "deduplication_test", tempDir.Path());
  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});

  // Use test helpers to validate basic export correctness
  EXPECT_EQ(exportedPrefab.actors.rigid.size(), 4) << "Should export 4 rigid actors";
  ValidateActorsExist(exportedPrefab.actors.rigid);

  // Find the actors in the prefab and use helpers to validate each one
  std::unordered_map<std::string, std::string> actorNameToShapeFile;
  for (auto const& actor : exportedPrefab.actors.rigid) {
    actorNameToShapeFile[std::string(actor.name)] = actor.shapeFile;

    // Use existing helper to validate each shape file exists
    ValidateShapeFile(
        std::string(actor.shapeFile), tempDir.Path(), prefabFile, "Deduplication test rigid actor");

    // Validate actor has proper name (not empty)
    EXPECT_FALSE(actor.name.empty()) << "All actors should have names for deduplication test";
  }

  // Verify we have all expected actors
  EXPECT_TRUE(actorNameToShapeFile.contains("duck_shared_1"));
  EXPECT_TRUE(actorNameToShapeFile.contains("duck_shared_2"));
  EXPECT_TRUE(actorNameToShapeFile.contains("duck_shared_3"));
  EXPECT_TRUE(actorNameToShapeFile.contains("duck_different"));

  // Actors using sharedShapeHandle should share the same mesh file
  EXPECT_EQ(actorNameToShapeFile["duck_shared_1"], actorNameToShapeFile["duck_shared_2"]);
  EXPECT_EQ(actorNameToShapeFile["duck_shared_1"], actorNameToShapeFile["duck_shared_3"]);

  // Actor using differentShapeHandle should have a different mesh file
  EXPECT_NE(actorNameToShapeFile["duck_shared_1"], actorNameToShapeFile["duck_different"]);

  // Verify mesh files exist (already validated per-actor above via ValidateShapeFile)
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, PrefabPortability) {
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));

  auto* scene = context->CreateScene("prefab_portability");
  MOCHI_DEFER(context->DestroyScene(scene));

  auto tempDir = CreateTempDirectory("mochi_prefab_portability", ExpectOK{});

  // Create a prefab file in our temp directory
  std::string_view constexpr kSrcPrefabJson = R"({
  "actors": {
    "rigid": [
      {
        "colliderType": "Box",
        "layer": "Object",
        "name": "BottomCube",
        "scale": [
          0.10000000149011612,
          0.10000000149011612,
          0.10000000149011612
        ],
        "shape": "cube/cube_minimal.mochi.json",
        "translation": [
          0,
          0.05000000074505806,
          0
        ]
      },
      {
        "colliderType": "Box",
        "layer": "Object",
        "name": "MiddleCube",
        "scale": [
          0.10000000149011612,
          0.10000000149011612,
          0.10000000149011612
        ],
        "shape": "cube/cube_minimal.mochi.json",
        "translation": [
          0,
          0.40000000596046448,
          0
        ]
      },
      {
        "colliderType": "Box",
        "layer": "Object",
        "name": "TopCube",
        "scale": [
          0.10000000149011612,
          0.10000000149011612,
          0.10000000149011612
        ],
        "shape": "cube/cube_minimal.mochi.json",
        "translation": [
          0,
          0.75,
          0
        ]
      }
    ]
  },
  "scene": {
    "description": "Stack of 3 rigid cubes using the minimal cube mesh."
  }
})";
  auto srcFilePath = tempDir.Path() / "cube_stack_original.mochi_scene";
  WriteFile(srcFilePath, kSrcPrefabJson, test::ExpectOK{});

  // Add a simple scene with rigid actors
  prefab::AddToScene(
      srcFilePath.string(), GetAssetsDir(), scene, prefab::PrefabParams{}, ExpectOK{});

  // Export inside assets directory (normal case)
  auto prefabFileInside = ExportScene(scene, "cube_stack_inside", tempDir.Path());

  // Export outside assets directory (portability case)
  std::filesystem::path outsideDir = tempDir.Path() / "prefab_exports_outside_assets";
  prefab::ExportScene(scene, "cube_stack_outside", outsideDir.string(), ExpectOK{});
  auto prefabFileOutside = outsideDir / "cube_stack_outside" / "cube_stack_outside.mochi_scene";

  // Test round-trip loading for both locations
  TestRoundTrip(scene, context, prefabFileInside, tempDir.Path());
  TestRoundTrip(scene, context, prefabFileOutside, outsideDir);
}

TEST_IF(MOCHI_USE_HDF5, PrefabExport, ExportDeterminism) {
  // Test that prefab export is deterministic
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));

  // Test with mixed scene containing both rigid and soft actors
  {
    auto* scene = context->CreateScene("test_mixed_export");
    MOCHI_DEFER(context->DestroyScene(scene));

    auto tempDir = CreateTempDirectory("export_determinism_mixed", ExpectOK{});

    auto mixedScenePath =
        CreateTempPrefabFile(kSoftSphereOnRigidCubeJson, "mixed_scene.mochi_scene", tempDir.Path());

    // Load the mixed scene
    prefab::AddToScene(
        mixedScenePath.string(), GetAssetsDir(), scene, prefab::PrefabParams{}, ExpectOK{});

    // First export
    auto exportedPrefabFile = ExportScene(scene, "determinism_test", tempDir.Path());

    // Read first export JSON content
    std::ifstream file1(exportedPrefabFile);
    ASSERT_TRUE(file1.is_open()) << "Could not open first export file: "
                                 << exportedPrefabFile.string();
    std::string content1((std::istreambuf_iterator<char>(file1)), std::istreambuf_iterator<char>());
    file1.close();

    // Load the exported prefab into a new scene (this loads all data into memory)
    auto* scene2 = context->CreateScene("test_mixed_export");
    MOCHI_DEFER(context->DestroyScene(scene2));

    prefab::AddToScene(
        exportedPrefabFile.string(),
        tempDir.Path().string(),
        scene2,
        prefab::PrefabParams{},
        ExpectOK{});

    // Delete the first export - should be safe since prefab data is now in memory
    std::filesystem::remove_all(exportedPrefabFile.parent_path());

    // Export the loaded scene to the same location with same name
    auto exportedPrefabFile2 = ExportScene(scene2, "determinism_test", tempDir.Path());
    std::ifstream file2(exportedPrefabFile2);
    ASSERT_TRUE(file2.is_open()) << "Could not open second export file: "
                                 << exportedPrefabFile2.string();
    std::string content2((std::istreambuf_iterator<char>(file2)), std::istreambuf_iterator<char>());
    file2.close();

    // Compare JSON content directly - should be identical if export is deterministic
    EXPECT_STREQ(content1.c_str(), content2.c_str())
        << "Exported JSON content should be identical for deterministic export";
  }

  // The Franka assets used below are not shipped externally.
#if MOCHI_INTERNAL
  // Test with articulated actor scene (franka) - more complex case
  {
    auto* scene = context->CreateScene("test_articulated_export");
    MOCHI_DEFER(context->DestroyScene(scene));

    auto tempDir = CreateTempDirectory("export_determinism_articulated", ExpectOK{});

    // Use franka arm scene as it's complex
    auto frankaScenePath = CreateTempPrefabFile(
        mochi::test::details::kFrankaHandJson, "franka_hand.mochi_scene", tempDir.Path());

    // Load the franka scene
    prefab::AddToScene(
        frankaScenePath.string(), GetAssetsDir(), scene, prefab::PrefabParams{}, ExpectOK{});

    // First export
    auto exportedPrefabFile = ExportScene(scene, "franka_determinism_test", tempDir.Path());

    // Read first export string content
    std::ifstream file1(exportedPrefabFile);
    ASSERT_TRUE(file1.is_open()) << "Could not open first export file: "
                                 << exportedPrefabFile.string();
    std::string content1((std::istreambuf_iterator<char>(file1)), std::istreambuf_iterator<char>());
    file1.close();

    // Load the exported prefab into a new scene (this loads all data into memory)
    // Use the same scene name to ensure the exported description is identical
    auto* scene2 = context->CreateScene("test_articulated_export");
    MOCHI_DEFER(context->DestroyScene(scene2));

    prefab::AddToScene(
        exportedPrefabFile.string(),
        tempDir.Path().string(),
        scene2,
        prefab::PrefabParams{},
        ExpectOK{});

    // Delete the first export - should be safe since prefab data is now in memory
    std::filesystem::remove_all(exportedPrefabFile.parent_path());

    // Export the loaded scene to the same location
    auto exportedPrefabFile2 = ExportScene(scene2, "franka_determinism_test", tempDir.Path());

    // Read second export content
    std::ifstream file2(exportedPrefabFile2);
    ASSERT_TRUE(file2.is_open()) << "Could not open second export file: "
                                 << exportedPrefabFile2.string();
    std::string content2((std::istreambuf_iterator<char>(file2)), std::istreambuf_iterator<char>());
    file2.close();

    // Compare JSON content directly - should be identical if export is deterministic
    EXPECT_STREQ(content1.c_str(), content2.c_str())
        << "Exported JSON content should be identical for deterministic export";
  }
#endif
}

// The soft-skinned Allegro assets are not shipped externally.
#if MOCHI_USE_HDF5 && !MOCHI_DEBUG && MOCHI_INTERNAL
#define MOCHI_HDF5_NOT_DEBUG_AND_INTERNAL 1
#else
#define MOCHI_HDF5_NOT_DEBUG_AND_INTERNAL 0
#endif
// Test times out in debug build.
TEST_IF(MOCHI_HDF5_NOT_DEBUG_AND_INTERNAL, PrefabExport, ExportScene_SoftSkinnedActor) {
  // Test for soft skinned actor export with attachment links
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));

  auto* scene = context->CreateScene("test_soft_skinned_export");
  MOCHI_DEFER(context->DestroyScene(scene));

  auto tempDir = CreateTempDirectory("mochi_export_allegro_soft", ExpectOK{});

  prefab::AddToScene(
      GetAssetPath("allegro_soft/allegro_soft.mochi_prefab"),
      GetAssetsDir(),
      scene,
      prefab::PrefabParams{},
      ExpectOK{});

  // Capture original material properties from the scene before export
  DynamicArray<SoftMaterialParams> originalMaterials;
  scene->ForEachActor([&](Actor const* actor) {
    if (actor->GetType() == ActorType::Soft) {
      Error materialError;
      auto material = actor->GetSoftMaterialParams(materialError);
      if (materialError.IsOK()) {
        originalMaterials.push_back(material);
      }
    }
  });

  ASSERT_EQ(originalMaterials.size(), 4)
      << "Allegro soft hand should have 4 soft fingertip actors in the original scene";

  auto prefabFile = ExportScene(scene, "allegro_soft", tempDir.Path());
  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});

  // Validate soft skinned actors were exported
  EXPECT_GT(exportedPrefab.actors.softSkinned.size(), 0)
      << "Scene should contain at least one soft skinned actor";

  auto const& exportedSoftSkinned = exportedPrefab.actors.softSkinned[0];

  // Basic soft skinned actor validation
  EXPECT_FALSE(exportedSoftSkinned.skeletonParams.name.empty())
      << "Soft skinned actor should have a skeleton name";

  // Validate the soft skinned actor has soft actors attached
  EXPECT_GT(exportedSoftSkinned.softParams.size(), 0)
      << "Soft skinned actor should have nested soft actors";

  // For allegro_soft, we expect 4 soft actors (one for each fingertip)
  EXPECT_EQ(exportedSoftSkinned.softParams.size(), 4)
      << "Allegro soft hand should have 4 soft fingertip actors";

  // Validate soft attachment links
  EXPECT_EQ(exportedSoftSkinned.softAttachLinks.size(), exportedSoftSkinned.softParams.size())
      << "Each soft actor should have attachment link information";

  char const* const expectedSoftAttachLinks[] = {
      "link_15.0_digit2_sensor_base",
      "link_11.0_digit2_sensor_base",
      "link_7.0_digit2_sensor_base",
      "link_3.0_digit2_sensor_base"};
  ASSERT_EQ(isize(expectedSoftAttachLinks), isize(exportedSoftSkinned.softAttachLinks));
  for (int i = 0; i < isize(expectedSoftAttachLinks); ++i) {
    EXPECT_STREQ(expectedSoftAttachLinks[i], exportedSoftSkinned.softAttachLinks[i].c_str());
  }

  int attachedSoftActors = 0;
  for (size_t i = 0; i < exportedSoftSkinned.softAttachLinks.size(); ++i) {
    auto const& attachLink = exportedSoftSkinned.softAttachLinks[i];
    if (!attachLink.empty()) {
      attachedSoftActors++;

      // The attachment should reference a valid link name in the skeleton
      bool foundLinkName = false;
      for (auto const& link : exportedSoftSkinned.skeletonParams.links) {
        if (link.name == attachLink) {
          foundLinkName = true;
          break;
        }
      }
      EXPECT_TRUE(foundLinkName) << "Soft actor " << i << " attachment link '" << attachLink
                                 << "' should match a valid skeleton link name";
    }
  }

  EXPECT_GT(attachedSoftActors, 0) << "At least some soft actors should have attachment links";

  // Validate each soft actor properties
  for (size_t i = 0; i < exportedSoftSkinned.softParams.size(); ++i) {
    auto const& softActor = exportedSoftSkinned.softParams[i];

    EXPECT_FALSE(softActor.name.empty()) << "Soft actor " << i << " should have a name";
    EXPECT_FALSE(softActor.shapeFile.empty()) << "Soft actor " << i << " should have a shape file";

    // Verify mesh files exist on disk
    ValidateShapeFile(
        std::string(softActor.shapeFile),
        tempDir.Path(),
        prefabFile,
        "Soft skinned soft actor " + std::to_string(i));

    // Validate soft skinned specific settings
    EXPECT_FALSE(softActor.useRecentering)
        << "Soft actors in soft skinned actors should have useRecentering=false";
    EXPECT_FALSE(softActor.hasGravity)
        << "Soft actors in soft skinned actors should have hasGravity=false";

    // Validate material properties match the original scene values
    EXPECT_EQ(softActor.material, originalMaterials[i]);
  }

  // Validate the skeleton parameters (articulated actor part)
  EXPECT_GT(exportedSoftSkinned.skeletonParams.links.size(), 0) << "Skeleton should have links";
  EXPECT_GT(exportedSoftSkinned.skeletonParams.joints.size(), 0) << "Skeleton should have joints";

  // Verify link shape files exist
  for (size_t i = 0; i < exportedSoftSkinned.skeletonParams.links.size(); ++i) {
    ValidateShapeFile(
        std::string(exportedSoftSkinned.skeletonParams.links[i].shapeFile),
        tempDir.Path(),
        prefabFile,
        "Soft skinned link " + std::to_string(i));
  }

  TestRoundTrip(scene, context, prefabFile, tempDir.Path());
}
#undef MOCHI_HDF5_NOT_DEBUG_AND_INTERNAL

TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, ExportScene_SoftSkinnedActorWithoutAttachmentLinks) {
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  auto* scene = context->CreateScene("test_soft_skinned_internal_export");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::AddToScene(
      GetAssetPath("soft_character/letters/m.mochi_prefab"), GetAssetsDir(), scene, {}, ExpectOK{});

  auto tempDir = CreateTempDirectory("mochi_export_soft_skinned_internal", ExpectOK{});
  auto prefabFile = ExportScene(scene, "soft_skinned_internal", tempDir.Path());
  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});

  ASSERT_EQ(1, isize(exportedPrefab.actors.softSkinned));
  EXPECT_TRUE(exportedPrefab.actors.softSkinned[0].softAttachLinks.empty());
  TestRoundTrip(scene, context, prefabFile, tempDir.Path());
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, ExportScene_SoftSkinnedActorEffectiveSoftNames) {
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  auto* scene = context->CreateScene("effective_soft_names_export");
  MOCHI_DEFER(context->DestroyScene(scene));

  SoftSkinnedActorParams params;
  auto& skeleton = params.skeletonParams;
  skeleton.name = "Skel";
  skeleton.joints = {{.type = ArticulatedJointType::Free}};
  skeleton.links = {
      {.name = "soft_0",
       .parentLink = -1,
       .shape = CreateUnitCubeTetMeshShape(context),
       .colliderType = ColliderType::None}};

  auto makeSoft = [&](char const* name) {
    SoftActorParams soft;
    soft.name = name;
    soft.shape = CreateUnitCubeTetSoftShape(context);
    soft.hasGravity = false;
    return soft;
  };
  params.softParams = {makeSoft("named"), makeSoft(""), makeSoft(""), makeSoft("soft_2")};

  Actor* actor = scene->CreateSoftSkinnedActor(params, ExpectOK{});
  ASSERT_NE(nullptr, actor);
  auto const& runtimeSofts = actor->GetNestedSoftActors(ExpectOK{});
  ASSERT_EQ(4, isize(runtimeSofts));
  EXPECT_STREQ("Skel/named", scene->GetActor(runtimeSofts[0])->GetName());
  EXPECT_STREQ("Skel/soft_1", scene->GetActor(runtimeSofts[1])->GetName());
  EXPECT_STREQ("Skel/soft_3", scene->GetActor(runtimeSofts[2])->GetName());
  EXPECT_STREQ("Skel/soft_2", scene->GetActor(runtimeSofts[3])->GetName());

  auto* sourceCubeActor = CreateRigidCubeActor(context, scene, "Cube");
  scene->EnableActorContactSymmetric(
      sourceCubeActor->GetHandle(),
      runtimeSofts[1],
      /*enable*/ false,
      IncludeNestedActors::No,
      ExpectOK{});

  auto tempDir = CreateTempDirectory("mochi_export_effective_soft_names", ExpectOK{});
  auto prefabFile = ExportScene(scene, "effective_soft_names", tempDir.Path());
  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});

  ASSERT_EQ(1, isize(exportedPrefab.actors.softSkinned));
  auto const& softSkinned = exportedPrefab.actors.softSkinned[0];
  ASSERT_EQ(4, isize(softSkinned.softParams));
  EXPECT_STREQ("named", softSkinned.softParams[0].name.c_str());
  EXPECT_STREQ("soft_1", softSkinned.softParams[1].name.c_str());
  EXPECT_STREQ("soft_3", softSkinned.softParams[2].name.c_str());
  EXPECT_STREQ("soft_2", softSkinned.softParams[3].name.c_str());

  ASSERT_TRUE(exportedPrefab.contactFilter.has_value());
  ASSERT_TRUE(exportedPrefab.contactFilter->actorContactSymmetric.has_value());
  auto containsActor = [](prefab::ActorContactEntry const& entry, DynamicString const& actorName) {
    for (auto const& name : entry.actors) {
      if (name == actorName) {
        return true;
      }
    }
    return false;
  };

  bool foundSoftContactFilter = false;
  for (auto const& entry : *exportedPrefab.contactFilter->actorContactSymmetric) {
    if (containsActor(entry, "Cube") && containsActor(entry, "Skel/soft_1")) {
      EXPECT_FALSE(entry.enable);
      foundSoftContactFilter = true;
    }
  }
  EXPECT_TRUE(foundSoftContactFilter);

  auto reloaded =
      prefab::LoadFromFile(prefabFile.string(), tempDir.Path().string(), context, ExpectOK{});
  ASSERT_EQ(1, isize(reloaded.actors.softSkinned));
  ASSERT_EQ(4, isize(reloaded.actors.softSkinned[0].softParams));
  reloaded.actors.softSkinned[0].softParams[1].name = "";

  auto* newScene = context->CreateScene("roundtrip_effective_soft_name_reference");
  MOCHI_DEFER(context->DestroyScene(newScene));
  prefab::AddToScene(reloaded, newScene, {}, ExpectOK{});

  auto findActor = [&](DynamicString const& name) -> Actor const* {
    Actor const* found = nullptr;
    newScene->ForEachActor([&](Actor const* candidate) {
      if (name == candidate->GetName()) {
        found = candidate;
      }
    });
    return found;
  };
  auto const* roundTripCubeActor = findActor("Cube");
  auto const* roundTripSoftActor = findActor("Skel/soft_1");
  ASSERT_NE(nullptr, roundTripCubeActor);
  ASSERT_NE(nullptr, roundTripSoftActor);

  auto const* sceneImpl = static_cast<SceneImpl const*>(newScene);
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(
      roundTripCubeActor->GetHandle(), roundTripSoftActor->GetHandle(), ExpectOK{}));
}

TEST_IF(
    MOCHI_HDF5_AND_INTERNAL,
    PrefabExport,
    ExportScene_DuplicateSoftSkinnedActorsGetLocalNestedNames) {
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  auto* scene = context->CreateScene("dup_soft_skinned_export");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Two instances of the same soft-skinned prefab share a base name, so export uniquifies the
  // second actor's name (e.g. "X" -> "X1"). Its nested soft actors inherit that uniquified parent
  // name; stripping the runtime prefix instead of the export name would trip an assert (dev) or
  // emit a malformed "/name" (opt) and break the round-trip.
  for (int i = 0; i < 2; ++i) {
    prefab::AddToScene(
        GetAssetPath("allegro_soft/allegro_soft.mochi_prefab"),
        GetAssetsDir(),
        scene,
        {},
        ExpectOK{});
  }

  auto tempDir = CreateTempDirectory("mochi_export_dup_soft_skinned", ExpectOK{});
  auto prefabFile = ExportScene(scene, "dup_soft_skinned", tempDir.Path());
  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});

  ASSERT_EQ(2, isize(exportedPrefab.actors.softSkinned));
  auto const& first = exportedPrefab.actors.softSkinned[0];
  auto const& second = exportedPrefab.actors.softSkinned[1];
  EXPECT_FALSE(first.skeletonParams.name.empty());
  EXPECT_FALSE(second.skeletonParams.name.empty());
  EXPECT_NE(first.skeletonParams.name, second.skeletonParams.name);
  ASSERT_GT(isize(first.softParams), 0);
  ASSERT_EQ(first.softParams.size(), second.softParams.size());
  for (int i = 0; i < isize(first.softParams); ++i) {
    EXPECT_FALSE(first.softParams[i].name.empty());
    EXPECT_EQ(std::string::npos, std::string(first.softParams[i].name).find('/'));
    EXPECT_STREQ(first.softParams[i].name.c_str(), second.softParams[i].name.c_str());
  }
  TestRoundTrip(scene, context, prefabFile, tempDir.Path());
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, ExportContactFilter) {
  // Test contact filter export with both named and unnamed actors
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));

  auto* scene = context->CreateScene("contact_filter_export_test");
  MOCHI_DEFER(context->DestroyScene(scene));

  auto shapeHandle = context->LoadShapeFromFile(
      GetAssetPath("cube/cube_minimal.mochi.json"),
      Real3{0.1_r, 0.1_r, 0.1_r},
      TransformRT{Quaternion::Identity(), Real3{0_r, 0_r, 0_r}},
      ExpectOK{});
  // Create named actors for layer and actor contact testing
  RigidActorParams namedParams1;
  namedParams1.name = "NamedActorA";
  namedParams1.layer = "LayerA";
  namedParams1.shape = shapeHandle;
  namedParams1.colliderType = ColliderType::Box;
  namedParams1.worldFromLocal = TransformRT{Quaternion::Identity(), Real3{0_r, 0_r, 0_r}};
  auto* namedActor1 = scene->CreateRigidActor(namedParams1, ExpectOK{});

  RigidActorParams namedParams2;
  namedParams2.name = "NamedActorB";
  namedParams2.layer = "LayerB";
  namedParams2.shape = shapeHandle;
  namedParams2.colliderType = ColliderType::Box;
  namedParams2.worldFromLocal = TransformRT{Quaternion::Identity(), Real3{1_r, 0_r, 0_r}};
  auto* namedActor2 = scene->CreateRigidActor(namedParams2, ExpectOK{});

  // Create unnamed actors to test name generation
  RigidActorParams unnamedParams1;
  unnamedParams1.name = ""; // Unnamed
  unnamedParams1.layer = "LayerC";
  unnamedParams1.shape = shapeHandle;
  unnamedParams1.colliderType = ColliderType::Box;
  unnamedParams1.worldFromLocal = TransformRT{Quaternion::Identity(), Real3{2_r, 0_r, 0_r}};
  auto* unnamedActor1 = scene->CreateRigidActor(unnamedParams1, ExpectOK{});

  RigidActorParams unnamedParams2;
  unnamedParams2.name = ""; // Unnamed
  unnamedParams2.layer = "LayerC";
  unnamedParams2.shape = shapeHandle;
  unnamedParams2.colliderType = ColliderType::Box;
  unnamedParams2.worldFromLocal = TransformRT{Quaternion::Identity(), Real3{3_r, 0_r, 0_r}};
  auto* unnamedActor2 = scene->CreateRigidActor(unnamedParams2, ExpectOK{});

  // Set up contact filters
  scene->EnableLayerContactSymmetric("LayerA", "LayerB", false, ExpectOK{});
  scene->EnableActorContactSymmetric(
      namedActor1->GetHandle(),
      namedActor2->GetHandle(),
      /*enable*/ false,
      IncludeNestedActors::No,
      ExpectOK{});
  scene->EnableActorContactSymmetric(
      unnamedActor1->GetHandle(),
      unnamedActor2->GetHandle(),
      /*enable*/ false,
      IncludeNestedActors::No,
      ExpectOK{});

  // Export the scene
  auto tempDir = CreateTempDirectory("export_contact_filter", ExpectOK{});
  auto prefabFile = ExportScene(scene, "contact_filter_export", tempDir.Path());

  // Load and verify the exported prefab
  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});

  ASSERT_TRUE(exportedPrefab.contactFilter.has_value())
      << "Exported prefab should have contactFilter";

  auto const& filter = *exportedPrefab.contactFilter;

  // Verify layer contact filter
  EXPECT_TRUE(filter.layerContactSymmetric.has_value());
  EXPECT_GT(filter.layerContactSymmetric->size(), 0);
  bool foundLayerFilter = false;
  for (auto const& entry : *filter.layerContactSymmetric) {
    if ((entry.layers[0] == "LayerA" && entry.layers[1] == "LayerB") ||
        (entry.layers[0] == "LayerB" && entry.layers[1] == "LayerA")) {
      EXPECT_FALSE(entry.enable);
      foundLayerFilter = true;
    }
  }
  EXPECT_TRUE(foundLayerFilter) << "Should find LayerA-LayerB contact filter";

  // All actor names in filters should be non-empty (unnamed get auto-generated names)
  EXPECT_TRUE(filter.actorContactSymmetric.has_value());
  for (auto const& entry : *filter.actorContactSymmetric) {
    EXPECT_FALSE(entry.actors[0].empty()) << "Actor names should not be empty in filter";
    EXPECT_FALSE(entry.actors[1].empty()) << "Actor names should not be empty in filter";
    EXPECT_FALSE(entry.enable);
  }

  // Verify all exported actors have non-empty names
  EXPECT_EQ(exportedPrefab.actors.rigid.size(), 4);
  for (auto const& actor : exportedPrefab.actors.rigid) {
    EXPECT_FALSE(actor.name.empty()) << "All exported actors should have names";
  }
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, ExportContactFilter_UsesExportedNestedLinkNames) {
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  auto* scene = context->CreateScene("contact_filter_nested_link_export_test");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::ScenePrefab scenePrefab;
  scenePrefab.actors.articulated.push_back(MakeSingleLinkRobot("Robot", "boneA"));
  scenePrefab.actors.articulated.push_back(MakeSingleLinkRobot("Robot", "boneA"));
  scenePrefab.actors.articulated.push_back(MakeSingleLinkRobot("Robot", ""));
  auto& cube = scenePrefab.actors.rigid.push_back();
  cube.name = "Cube";
  cube.shapeFile = "cube/cube_minimal.mochi.json";
  cube.colliderType = ColliderType::Box;

  prefab::LoadShapes(scenePrefab, GetAssetsDir(), context, ExpectOK{});
  auto const result = prefab::AddToScene(scenePrefab, scene, {}, ExpectOK{});
  auto robots = result.Filter(ActorType::Articulated);
  auto rigids = result.Filter(ActorType::Rigid);
  ASSERT_EQ(3, isize(robots));
  ASSERT_EQ(1, isize(rigids));

  for (auto const* robot : robots) {
    auto const& links = robot->GetNestedLinkActors(ExpectOK{});
    ASSERT_EQ(1, isize(links));
    scene->EnableActorContactSymmetric(
        rigids[0]->GetHandle(),
        links[0],
        /*enable*/ false,
        IncludeNestedActors::No,
        ExpectOK{});
  }

  auto tempDir = CreateTempDirectory("export_contact_filter_nested_link_names", ExpectOK{});
  auto prefabFile = ExportScene(scene, "contact_filter_nested_link_names", tempDir.Path());
  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});

  ASSERT_TRUE(exportedPrefab.contactFilter.has_value());
  ASSERT_TRUE(exportedPrefab.contactFilter->actorContactSymmetric.has_value());
  auto containsActor = [](prefab::ActorContactEntry const& entry, DynamicString const& actorName) {
    for (auto const& name : entry.actors) {
      if (name == actorName) {
        return true;
      }
    }
    return false;
  };

  DynamicArray<DynamicString> expectedNestedLinkNames;
  int namedLinkCount = 0;
  int fallbackLinkCount = 0;
  for (auto const& actor : exportedPrefab.actors.articulated) {
    ASSERT_EQ(1, isize(actor.links));

    DynamicString nestedLinkName = actor.name;
    nestedLinkName += "/";
    nestedLinkName += actor.links[0].name;
    expectedNestedLinkNames.push_back(std::move(nestedLinkName));

    if (actor.links[0].name == "boneA") {
      ++namedLinkCount;
    } else {
      ++fallbackLinkCount;
      EXPECT_FALSE(actor.links[0].name.empty());
    }
  }
  EXPECT_EQ(2, namedLinkCount);
  EXPECT_EQ(1, fallbackLinkCount);

  for (auto const& nestedLinkName : expectedNestedLinkNames) {
    bool foundFilter = false;
    for (auto const& entry : *exportedPrefab.contactFilter->actorContactSymmetric) {
      foundFilter |= containsActor(entry, "Cube") && containsActor(entry, nestedLinkName);
    }
    EXPECT_TRUE(foundFilter);
  }

  // Re-import and confirm each entry reconnects to the correct nested link actor: the exported
  // names must match the nested link names recreated on import (a mismatch silently drops the
  // filter). The unnamed link case also verifies fallback link names are included in exported actor
  // contact-filter names. String checks above only prove the export side; this closes the round
  // trip.
  auto reloaded =
      prefab::LoadFromFile(prefabFile.string(), tempDir.Path().string(), context, ExpectOK{});
  auto* newScene = context->CreateScene("roundtrip");
  MOCHI_DEFER(context->DestroyScene(newScene));
  prefab::AddToScene(reloaded, newScene, {}, ExpectOK{});

  auto findActor = [&](DynamicString const& name) -> Actor const* {
    Actor const* actor = nullptr;
    newScene->ForEachActor([&](Actor const* candidate) {
      if (name == candidate->GetName()) {
        actor = candidate;
      }
    });
    return actor;
  };
  auto const* sceneImpl = static_cast<SceneImpl const*>(newScene);
  Actor const* cubeActor = findActor("Cube");
  ASSERT_NE(nullptr, cubeActor);
  for (auto const& nestedLinkName : expectedNestedLinkNames) {
    Actor const* linkActor = findActor(nestedLinkName);
    ASSERT_NE(nullptr, linkActor);
    EXPECT_FALSE(sceneImpl->IsActorContactEnabled(
        cubeActor->GetHandle(), linkActor->GetHandle(), ExpectOK{}));
  }
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, ExportContactFilter_UsesExportedNestedSoftNames) {
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  auto* scene = context->CreateScene("contact_filter_nested_soft_export_test");
  MOCHI_DEFER(context->DestroyScene(scene));

  SoftSkinnedActorParams params;
  auto& skeleton = params.skeletonParams;
  skeleton.name = "Skel";
  skeleton.joints = {{.type = ArticulatedJointType::Free}};
  skeleton.links = {
      {.name = "bone",
       .parentLink = -1,
       .shape = CreateUnitCubeTetMeshShape(context),
       .colliderType = ColliderType::None}};

  auto& soft = params.softParams.push_back();
  soft.name = "soft";
  soft.shape = CreateUnitCubeTetSoftShape(context);
  soft.hasGravity = false;

  Actor* skel = scene->CreateSoftSkinnedActor(params, ExpectOK{});
  ASSERT_NE(nullptr, skel);
  auto softActors = skel->GetNestedSoftActors(ExpectOK{});
  ASSERT_EQ(1, isize(softActors));
  auto* cube = CreateRigidCubeActor(context, scene, "Cube");
  scene->EnableActorContactSymmetric(
      cube->GetHandle(),
      softActors[0],
      /*enable*/ false,
      IncludeNestedActors::No,
      ExpectOK{});

  auto tempDir = CreateTempDirectory("export_contact_filter_nested_soft", ExpectOK{});
  auto prefabFile = ExportScene(scene, "contact_filter_nested_soft", tempDir.Path());
  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});

  ASSERT_EQ(1, isize(exportedPrefab.actors.softSkinned));
  auto const& exportedSoftSkinned = exportedPrefab.actors.softSkinned[0];
  ASSERT_EQ(1, isize(exportedSoftSkinned.softParams));
  DynamicString nestedSoftName = exportedSoftSkinned.skeletonParams.name;
  nestedSoftName += "/";
  nestedSoftName += exportedSoftSkinned.softParams[0].name;

  ASSERT_TRUE(exportedPrefab.contactFilter.has_value());
  ASSERT_TRUE(exportedPrefab.contactFilter->actorContactSymmetric.has_value());
  bool foundFilter = false;
  for (auto const& entry : *exportedPrefab.contactFilter->actorContactSymmetric) {
    foundFilter |= entry.actors.size() == 2 &&
        ((entry.actors[0] == "Cube" && entry.actors[1] == nestedSoftName) ||
         (entry.actors[0] == nestedSoftName && entry.actors[1] == "Cube"));
  }
  EXPECT_TRUE(foundFilter);

  auto reloaded =
      prefab::LoadFromFile(prefabFile.string(), tempDir.Path().string(), context, ExpectOK{});
  auto* newScene = context->CreateScene("roundtrip");
  MOCHI_DEFER(context->DestroyScene(newScene));
  prefab::AddToScene(reloaded, newScene, {}, ExpectOK{});

  auto findActor = [&](DynamicString const& name) -> Actor const* {
    Actor const* actor = nullptr;
    newScene->ForEachActor([&](Actor const* candidate) {
      if (name == candidate->GetName()) {
        actor = candidate;
      }
    });
    return actor;
  };
  Actor const* cubeActor = findActor("Cube");
  Actor const* softActor = findActor(nestedSoftName);
  ASSERT_NE(nullptr, cubeActor);
  ASSERT_NE(nullptr, softActor);

  auto const* sceneImpl = static_cast<SceneImpl const*>(newScene);
  EXPECT_FALSE(
      sceneImpl->IsActorContactEnabled(cubeActor->GetHandle(), softActor->GetHandle(), ExpectOK{}));
  EXPECT_FALSE(
      sceneImpl->IsActorContactEnabled(softActor->GetHandle(), cubeActor->GetHandle(), ExpectOK{}));
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, ExportContactFilter_RoundTripsSelfContact) {
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  auto* scene = context->CreateScene("contact_filter_self_contact_export_test");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::ScenePrefab scenePrefab;
  scenePrefab.actors.articulated.push_back(MakeSingleLinkRobot("Robot"));
  prefab::LoadShapes(scenePrefab, GetAssetsDir(), context, ExpectOK{});
  auto const result = prefab::AddToScene(scenePrefab, scene, {}, ExpectOK{});
  auto const robots = result.Filter(ActorType::Articulated);
  ASSERT_EQ(1, isize(robots));
  auto* robot = robots[0];
  ASSERT_EQ(1, isize(robot->GetNestedLinkActors(ExpectOK{})));

  // Export canonicalizes disabled self-pairs as actorContactSymmetric entries. Exact scope on the
  // articulated parent must survive the round trip without widening to its nested link.
  auto* cube = CreateRigidCubeActor(context, scene, "Cube");
  scene->EnableActorContactSymmetric(
      cube->GetHandle(),
      cube->GetHandle(),
      /*enable*/ false,
      IncludeNestedActors::No,
      ExpectOK{});
  scene->EnableActorContactSymmetric(
      robot->GetHandle(),
      robot->GetHandle(),
      /*enable*/ false,
      IncludeNestedActors::No,
      ExpectOK{});

  auto tempDir = CreateTempDirectory("export_contact_filter_self_contact", ExpectOK{});
  auto prefabFile = ExportScene(scene, "contact_filter_self_contact", tempDir.Path());

  std::string exportedJson;
  ReadFile(prefabFile, exportedJson, ExpectOK{});
  EXPECT_NE(std::string::npos, exportedJson.find("\"includeNestedActors\": false"));

  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});
  ASSERT_TRUE(exportedPrefab.contactFilter.has_value());
  ASSERT_TRUE(exportedPrefab.contactFilter->actorContactSymmetric.has_value());
  bool foundCubeSelfPair = false;
  bool foundRobotSelfPair = false;
  for (auto const& entry : *exportedPrefab.contactFilter->actorContactSymmetric) {
    if (entry.actors.size() == 2 && entry.actors[0] == "Cube" && entry.actors[1] == "Cube") {
      foundCubeSelfPair = true;
      EXPECT_FALSE(entry.enable);
      EXPECT_TRUE(entry.includeNestedActors);
    }
    if (entry.actors.size() == 2 && entry.actors[0] == "Robot" && entry.actors[1] == "Robot") {
      foundRobotSelfPair = true;
      EXPECT_FALSE(entry.enable);
      EXPECT_FALSE(entry.includeNestedActors);
    }
  }
  EXPECT_TRUE(foundCubeSelfPair);
  EXPECT_TRUE(foundRobotSelfPair);

  auto reloaded =
      prefab::LoadFromFile(prefabFile.string(), tempDir.Path().string(), context, ExpectOK{});
  auto* newScene = context->CreateScene("roundtrip");
  MOCHI_DEFER(context->DestroyScene(newScene));
  prefab::AddToScene(reloaded, newScene, {}, ExpectOK{});

  auto findActor = [&](DynamicString const& name) -> Actor const* {
    Actor const* actor = nullptr;
    newScene->ForEachActor([&](Actor const* candidate) {
      if (name == candidate->GetName()) {
        actor = candidate;
      }
    });
    return actor;
  };
  Actor const* cubeActor = findActor("Cube");
  Actor const* robotActor = findActor("Robot");
  Actor const* robotLink = findActor("Robot/boneA");
  ASSERT_NE(nullptr, cubeActor);
  ASSERT_NE(nullptr, robotActor);
  ASSERT_NE(nullptr, robotLink);

  auto const* sceneImpl = static_cast<SceneImpl const*>(newScene);
  EXPECT_FALSE(
      sceneImpl->IsActorContactEnabled(cubeActor->GetHandle(), cubeActor->GetHandle(), ExpectOK{}));
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(
      robotActor->GetHandle(), robotActor->GetHandle(), ExpectOK{}));
  EXPECT_TRUE(
      sceneImpl->IsActorContactEnabled(robotLink->GetHandle(), robotLink->GetHandle(), ExpectOK{}));
}

TEST_IF(
    MOCHI_HDF5_AND_INTERNAL,
    PrefabExport,
    ExportContactConfiguration_AvoidsTopLevelNestedNameCollision) {
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  auto* scene = context->CreateScene("contact_filter_top_level_nested_collision");
  MOCHI_DEFER(context->DestroyScene(scene));

  auto* topLevelActor = CreateRigidCubeActor(context, scene, "Robot/boneA");

  prefab::ArticulatedActorPrefab robot;
  robot.name = "Robot";
  robot.joints.resize(1);
  robot.joints[0].type = ArticulatedJointType::Free;
  robot.links.resize(1);
  robot.links[0].name = "boneA";
  robot.links[0].parentLink = -1;
  robot.links[0].shapeFile = "articulated/two_links_revolute/bone_a.mochi.h5";
  robot.links[0].colliderType = ColliderType::Box;

  prefab::ScenePrefab scenePrefab;
  scenePrefab.actors.articulated.push_back(robot);
  prefab::LoadShapes(scenePrefab, GetAssetsDir(), context, ExpectOK{});
  auto const result = prefab::AddToScene(scenePrefab, scene, {}, ExpectOK{});
  auto robots = result.Filter(ActorType::Articulated);
  ASSERT_EQ(1, isize(robots));

  auto const& links = robots[0]->GetNestedLinkActors(ExpectOK{});
  ASSERT_EQ(1, isize(links));
  scene->EnableActorContactSymmetric(
      topLevelActor->GetHandle(),
      links[0],
      /*enable*/ false,
      IncludeNestedActors::No,
      ExpectOK{});
  ContactPairParamsOverride paramsOverride;
  paramsOverride.penaltyCoefficient = 7_r;
  scene->SetContactPairParamsOverride(
      topLevelActor->GetHandle(), links[0], paramsOverride, ExpectOK{});

  auto tempDir =
      CreateTempDirectory("export_contact_filter_top_level_nested_collision", ExpectOK{});
  auto prefabFile = ExportScene(scene, "contact_filter_top_level_nested_collision", tempDir.Path());
  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});

  ASSERT_EQ(1, isize(exportedPrefab.actors.rigid));
  ASSERT_EQ(1, isize(exportedPrefab.actors.articulated));
  ASSERT_EQ(1, isize(exportedPrefab.actors.articulated[0].links));

  DynamicString rigidName = exportedPrefab.actors.rigid[0].name;
  DynamicString nestedLinkName = exportedPrefab.actors.articulated[0].name;
  nestedLinkName += "/";
  nestedLinkName += exportedPrefab.actors.articulated[0].links[0].name;
  EXPECT_NE(rigidName, nestedLinkName);

  ASSERT_TRUE(exportedPrefab.contactFilter.has_value());
  ASSERT_TRUE(exportedPrefab.contactFilter->actorContactSymmetric.has_value());
  auto containsActor = [](prefab::ActorContactEntry const& entry, DynamicString const& actorName) {
    for (auto const& name : entry.actors) {
      if (name == actorName) {
        return true;
      }
    }
    return false;
  };

  bool foundFilter = false;
  for (auto const& entry : *exportedPrefab.contactFilter->actorContactSymmetric) {
    if (containsActor(entry, rigidName) && containsActor(entry, nestedLinkName)) {
      EXPECT_FALSE(entry.enable);
      foundFilter = true;
    }
  }
  EXPECT_TRUE(foundFilter);

  ASSERT_TRUE(exportedPrefab.contactPairParamsOverrides.has_value());
  ASSERT_EQ(1, isize(*exportedPrefab.contactPairParamsOverrides));
  auto const& paramsOverrideEntry = (*exportedPrefab.contactPairParamsOverrides)[0];
  ASSERT_EQ(2, isize(paramsOverrideEntry.actors));
  EXPECT_TRUE(
      (paramsOverrideEntry.actors[0] == rigidName &&
       paramsOverrideEntry.actors[1] == nestedLinkName) ||
      (paramsOverrideEntry.actors[0] == nestedLinkName &&
       paramsOverrideEntry.actors[1] == rigidName));
  EXPECT_EQ(std::optional<real>{7_r}, paramsOverrideEntry.paramsOverride.penaltyCoefficient);

  auto reloaded =
      prefab::LoadFromFile(prefabFile.string(), tempDir.Path().string(), context, ExpectOK{});
  auto* newScene = context->CreateScene("roundtrip_collision");
  MOCHI_DEFER(context->DestroyScene(newScene));
  prefab::AddToScene(reloaded, newScene, {}, ExpectOK{});

  auto findActor = [&](DynamicString const& name) -> Actor const* {
    Actor const* actor = nullptr;
    newScene->ForEachActor([&](Actor const* candidate) {
      if (name == candidate->GetName()) {
        actor = candidate;
      }
    });
    return actor;
  };
  Actor const* rigidActor = findActor(rigidName);
  Actor const* linkActor = findActor(nestedLinkName);
  ASSERT_NE(nullptr, rigidActor);
  ASSERT_NE(nullptr, linkActor);

  auto const* sceneImpl = static_cast<SceneImpl const*>(newScene);
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(
      rigidActor->GetHandle(), linkActor->GetHandle(), ExpectOK{}));
  EXPECT_EQ(
      std::optional<real>{7_r},
      newScene
          ->GetContactPairParamsOverride(
              rigidActor->GetHandle(), linkActor->GetHandle(), ExpectOK{})
          .penaltyCoefficient);
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, ExportContactFilter_DropsUnsupportedActorTypes) {
  // A shell actor is unsupported for prefab export, so it must not appear in exportActorNames; a
  // contact filter naming it should be dropped rather than emitted with a dangling actor name.
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  auto* scene = context->CreateScene("contact_filter_unsupported_type_test");
  MOCHI_DEFER(context->DestroyScene(scene));

  RigidActorParams rigidParams;
  rigidParams.name = "Cube";
  rigidParams.shape = context->LoadShapeFromFile(
      GetAssetPath("cube/cube_minimal.mochi.json"),
      Real3{1_r, 1_r, 1_r},
      TransformRT{},
      ExpectOK{});
  rigidParams.colliderType = ColliderType::Box;
  auto* rigid = scene->CreateRigidActor(rigidParams, ExpectOK{});

  auto&& [coords, tris] = CreateMinimalTriMeshUnitCube();
  experimental::ShellActorParams shellParams;
  shellParams.name = "Shell";
  shellParams.shape = context->CreateTriMeshShape(
      Flatten(MakeSpan(coords)), Flatten(MakeSpan(tris)), ErrorAssert{});
  auto* shell = experimental::CreateShellActor(scene, shellParams, ExpectOK{});

  scene->EnableActorContactSymmetric(
      rigid->GetHandle(),
      shell->GetHandle(),
      /*enable*/ false,
      IncludeNestedActors::No,
      ExpectOK{});

  auto tempDir = CreateTempDirectory("export_contact_filter_unsupported", ExpectOK{});
  // Exporting the unsupported shell logs an expected "skipping actor" warning.
  auto suppressWarning = test::SuppressLogWarning();
  auto prefabFile = ExportScene(scene, "contact_filter_unsupported", tempDir.Path());
  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});

  EXPECT_EQ(1, isize(exportedPrefab.actors.rigid));
  bool const hasSymmetricEntries = exportedPrefab.contactFilter.has_value() &&
      exportedPrefab.contactFilter->actorContactSymmetric.has_value() &&
      !exportedPrefab.contactFilter->actorContactSymmetric->empty();
  EXPECT_FALSE(hasSymmetricEntries);
}

// The existing ExportContactFilter tests all exercise the symmetric emit path. These cover the
// asymmetric path: disabling contact in one direction only must be emitted under
// actorContactAsymmetric / layerContactAsymmetric (not symmetric), ordered [colliding, collider],
// and must round-trip as a one-directional disable.
TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, ExportContactFilter_AsymmetricActorRoundTrips) {
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  auto* scene = context->CreateScene("contact_filter_asymmetric_actor_export_test");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::ScenePrefab scenePrefab;
  scenePrefab.actors.articulated.push_back(MakeSingleLinkRobot("Robot"));
  prefab::LoadShapes(scenePrefab, GetAssetsDir(), context, ExpectOK{});
  auto const robotResult = prefab::AddToScene(scenePrefab, scene, {}, ExpectOK{});
  auto const robots = robotResult.Filter(ActorType::Articulated);
  ASSERT_EQ(1, isize(robots));
  auto* robot = robots[0];
  ASSERT_EQ(1, isize(robot->GetNestedLinkActors(ExpectOK{})));

  // Disable contact A->B only (A colliding against B's collider); B->A stays enabled.
  auto* actorA = CreateRigidCubeActor(context, scene, "ActorA");
  auto* actorB = CreateRigidCubeActor(context, scene, "ActorB");
  scene->EnableActorContactAsymmetric(
      actorA->GetHandle(),
      actorB->GetHandle(),
      /*enable*/ false,
      IncludeNestedActors::No,
      ExpectOK{});
  scene->EnableActorContactAsymmetric(
      robot->GetHandle(),
      actorB->GetHandle(),
      /*enable*/ false,
      IncludeNestedActors::No,
      ExpectOK{});

  auto tempDir = CreateTempDirectory("export_contact_filter_asymmetric_actor", ExpectOK{});
  auto prefabFile = ExportScene(scene, "contact_filter_asymmetric_actor", tempDir.Path());

  std::string exportedJson;
  ReadFile(prefabFile, exportedJson, ExpectOK{});
  EXPECT_NE(std::string::npos, exportedJson.find("\"includeNestedActors\": false"));

  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});

  ASSERT_TRUE(exportedPrefab.contactFilter.has_value());
  auto const& filter = *exportedPrefab.contactFilter;

  // The disabled pairs are emitted as asymmetric entries ordered [colliding, collider].
  ASSERT_TRUE(filter.actorContactAsymmetric.has_value());
  bool foundAsymmetric = false;
  bool foundArticulatedParent = false;
  for (auto const& entry : *filter.actorContactAsymmetric) {
    ASSERT_EQ(2, isize(entry.actors));
    if (entry.actors[0] == "ActorA" && entry.actors[1] == "ActorB") {
      EXPECT_FALSE(entry.enable);
      EXPECT_TRUE(entry.includeNestedActors);
      foundAsymmetric = true;
    }
    if (entry.actors[0] == "Robot" && entry.actors[1] == "ActorB") {
      EXPECT_FALSE(entry.enable);
      EXPECT_FALSE(entry.includeNestedActors);
      foundArticulatedParent = true;
    }
    // The reverse orderings must not be emitted.
    EXPECT_FALSE(entry.actors[0] == "ActorB" && entry.actors[1] == "ActorA");
    EXPECT_FALSE(entry.actors[0] == "ActorB" && entry.actors[1] == "Robot");
  }
  EXPECT_TRUE(foundAsymmetric) << "A->B disable should land under actorContactAsymmetric";
  EXPECT_TRUE(foundArticulatedParent)
      << "Robot->B disable should land under actorContactAsymmetric";

  // ...and must NOT be emitted as a symmetric pair.
  if (filter.actorContactSymmetric.has_value()) {
    for (auto const& entry : *filter.actorContactSymmetric) {
      ASSERT_EQ(2, isize(entry.actors));
      bool const mentionsPair = (entry.actors[0] == "ActorA" && entry.actors[1] == "ActorB") ||
          (entry.actors[0] == "ActorB" && entry.actors[1] == "ActorA") ||
          (entry.actors[0] == "Robot" && entry.actors[1] == "ActorB") ||
          (entry.actors[0] == "ActorB" && entry.actors[1] == "Robot");
      EXPECT_FALSE(mentionsPair) << "Asymmetric pair must not be exported as symmetric";
    }
  }

  // Round-trip: each forward pair stays disabled while reverse and nested pairs stay enabled.
  auto reloaded =
      prefab::LoadFromFile(prefabFile.string(), tempDir.Path().string(), context, ExpectOK{});
  auto* newScene = context->CreateScene("roundtrip");
  MOCHI_DEFER(context->DestroyScene(newScene));
  prefab::AddToScene(reloaded, newScene, {}, ExpectOK{});

  auto findActor = [&](DynamicString const& name) -> Actor const* {
    Actor const* actor = nullptr;
    newScene->ForEachActor([&](Actor const* candidate) {
      if (name == candidate->GetName()) {
        actor = candidate;
      }
    });
    return actor;
  };
  Actor const* reA = findActor("ActorA");
  Actor const* reB = findActor("ActorB");
  Actor const* reRobot = findActor("Robot");
  Actor const* reRobotLink = findActor("Robot/boneA");
  ASSERT_NE(nullptr, reA);
  ASSERT_NE(nullptr, reB);
  ASSERT_NE(nullptr, reRobot);
  ASSERT_NE(nullptr, reRobotLink);
  auto const* sceneImpl = static_cast<SceneImpl const*>(newScene);
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(reA->GetHandle(), reB->GetHandle(), ExpectOK{}));
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(reB->GetHandle(), reA->GetHandle(), ExpectOK{}));
  EXPECT_FALSE(
      sceneImpl->IsActorContactEnabled(reRobot->GetHandle(), reB->GetHandle(), ExpectOK{}));
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(reB->GetHandle(), reRobot->GetHandle(), ExpectOK{}));
  EXPECT_TRUE(
      sceneImpl->IsActorContactEnabled(reRobotLink->GetHandle(), reB->GetHandle(), ExpectOK{}));
  EXPECT_TRUE(
      sceneImpl->IsActorContactEnabled(reB->GetHandle(), reRobotLink->GetHandle(), ExpectOK{}));
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, ExportContactFilter_AsymmetricLayerRoundTrips) {
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  auto* scene = context->CreateScene("contact_filter_asymmetric_layer_export_test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Two rigid actors in distinct layers so the layer names exist in the exported prefab.
  auto makeLayeredCube = [&](char const* name, char const* layer, Real3 const& pos) {
    RigidActorParams params;
    params.name = name;
    params.layer = layer;
    params.shape = context->LoadShapeFromFile(
        GetAssetPath("cube/cube_minimal.mochi.json"),
        Real3{0.1_r, 0.1_r, 0.1_r},
        TransformRT{Quaternion::Identity(), pos},
        ExpectOK{});
    params.colliderType = ColliderType::Box;
    params.worldFromLocal = TransformRT{Quaternion::Identity(), pos};
    return scene->CreateRigidActor(params, ExpectOK{});
  };
  makeLayeredCube("CubeX", "LayerX", Real3{0_r, 0_r, 0_r});
  makeLayeredCube("CubeY", "LayerY", Real3{1_r, 0_r, 0_r});

  // Disable LayerX->LayerY only; LayerY->LayerX stays enabled.
  scene->EnableLayerContactAsymmetric("LayerX", "LayerY", false, ExpectOK{});

  auto tempDir = CreateTempDirectory("export_contact_filter_asymmetric_layer", ExpectOK{});
  auto prefabFile = ExportScene(scene, "contact_filter_asymmetric_layer", tempDir.Path());
  auto exportedPrefab = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});

  ASSERT_TRUE(exportedPrefab.contactFilter.has_value());
  auto const& filter = *exportedPrefab.contactFilter;

  // The disabled pair is emitted as a single asymmetric layer entry, ordered [X, Y], enable=false.
  ASSERT_TRUE(filter.layerContactAsymmetric.has_value());
  bool foundAsymmetric = false;
  for (auto const& entry : *filter.layerContactAsymmetric) {
    ASSERT_EQ(2, isize(entry.layers));
    if (entry.layers[0] == "LayerX" && entry.layers[1] == "LayerY") {
      EXPECT_FALSE(entry.enable);
      foundAsymmetric = true;
    }
    EXPECT_FALSE(entry.layers[0] == "LayerY" && entry.layers[1] == "LayerX");
  }
  EXPECT_TRUE(foundAsymmetric) << "LayerX->LayerY disable should land under layerContactAsymmetric";

  // ...and must NOT be emitted as a symmetric pair.
  if (filter.layerContactSymmetric.has_value()) {
    for (auto const& entry : *filter.layerContactSymmetric) {
      ASSERT_EQ(2, isize(entry.layers));
      bool const mentionsPair = (entry.layers[0] == "LayerX" && entry.layers[1] == "LayerY") ||
          (entry.layers[0] == "LayerY" && entry.layers[1] == "LayerX");
      EXPECT_FALSE(mentionsPair) << "Asymmetric layer pair must not be exported as symmetric";
    }
  }

  // Round-trip: reload and re-instantiate; LayerX->LayerY stays disabled, LayerY->LayerX enabled.
  auto reloaded =
      prefab::LoadFromFile(prefabFile.string(), tempDir.Path().string(), context, ExpectOK{});
  auto* newScene = context->CreateScene("roundtrip_layer");
  MOCHI_DEFER(context->DestroyScene(newScene));
  prefab::AddToScene(reloaded, newScene, {}, ExpectOK{});

  EXPECT_FALSE(newScene->IsLayerContactEnabled("LayerX", "LayerY"));
  EXPECT_TRUE(newScene->IsLayerContactEnabled("LayerY", "LayerX"));
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, ContactFiltersAreDeterministic) {
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  auto tempDirA = CreateTempDirectory("contact_filter_order_a", ExpectOK{});
  auto tempDirB = CreateTempDirectory("contact_filter_order_b", ExpectOK{});
  auto const shape = context->LoadShapeFromFile(
      GetAssetPath("cube/cube_minimal.mochi.json"),
      Real3{0.1_r, 0.1_r, 0.1_r},
      TransformRT{},
      ExpectOK{});

  constexpr int kNumPairs = 24;
  using NamePair = std::pair<std::string, std::string>;
  DynamicArray<NamePair> expectedActorAsymmetric;
  DynamicArray<NamePair> expectedActorSymmetric;
  DynamicArray<NamePair> expectedLayerAsymmetric;
  DynamicArray<NamePair> expectedLayerSymmetric;
  for (int i = 0; i < kNumPairs; ++i) {
    std::string const actorName = "Actor" + std::to_string(i);
    std::string const layerName = "Layer" + std::to_string(i);
    expectedActorAsymmetric.push_back({actorName, "AsymmetricAnchor"});
    expectedActorSymmetric.push_back({actorName, "SymmetricAnchor"});
    expectedLayerAsymmetric.push_back({layerName, "AsymmetricLayer"});
    expectedLayerSymmetric.push_back({layerName, "SymmetricLayer"});
  }
  std::sort(expectedActorAsymmetric.begin(), expectedActorAsymmetric.end());
  std::sort(expectedActorSymmetric.begin(), expectedActorSymmetric.end());
  std::sort(expectedLayerAsymmetric.begin(), expectedLayerAsymmetric.end());
  std::sort(expectedLayerSymmetric.begin(), expectedLayerSymmetric.end());

  auto actorPairs = [](DynamicArray<prefab::ActorContactEntry> const& entries) {
    DynamicArray<NamePair> pairs;
    pairs.reserve(entries.size());
    for (auto const& entry : entries) {
      pairs.push_back({std::string(entry.actors[0]), std::string(entry.actors[1])});
    }
    return pairs;
  };
  auto layerPairs = [](DynamicArray<prefab::LayerContactEntry> const& entries) {
    DynamicArray<NamePair> pairs;
    pairs.reserve(entries.size());
    for (auto const& entry : entries) {
      pairs.push_back({std::string(entry.layers[0]), std::string(entry.layers[1])});
    }
    return pairs;
  };

  auto exportWithOrder =
      [&](std::filesystem::path const& outputDir, bool reverse, std::string& json) {
        auto* scene = context->CreateScene("contact_filter_determinism");
        MOCHI_DEFER(context->DestroyScene(scene));

        auto makeActor = [&](char const* name, char const* layer) {
          RigidActorParams params;
          params.name = name;
          params.layer = layer;
          params.shape = shape;
          params.colliderType = ColliderType::Box;
          return scene->CreateRigidActor(params, ExpectOK{});
        };
        Actor* const symmetricAnchor = makeActor("SymmetricAnchor", "SymmetricLayer");
        Actor* const asymmetricAnchor = makeActor("AsymmetricAnchor", "AsymmetricLayer");

        DynamicArray<Actor*> actors;
        DynamicArray<std::string> layerNames;
        for (int i = 0; i < kNumPairs; ++i) {
          std::string const actorName = "Actor" + std::to_string(i);
          layerNames.push_back("Layer" + std::to_string(i));
          actors.push_back(makeActor(actorName.c_str(), layerNames.back().c_str()));
        }

        for (int n = 0; n < kNumPairs; ++n) {
          int const i = reverse ? kNumPairs - 1 - n : n;
          scene->EnableActorContactSymmetric(
              actors[i]->GetHandle(),
              symmetricAnchor->GetHandle(),
              false,
              IncludeNestedActors::No,
              ExpectOK{});
          scene->EnableActorContactAsymmetric(
              actors[i]->GetHandle(),
              asymmetricAnchor->GetHandle(),
              false,
              IncludeNestedActors::No,
              ExpectOK{});
          scene->EnableLayerContactSymmetric(layerNames[i], "SymmetricLayer", false, ExpectOK{});
          scene->EnableLayerContactAsymmetric(layerNames[i], "AsymmetricLayer", false, ExpectOK{});
        }

        auto const prefabFile = ExportScene(scene, "contact_filter", outputDir);
        ReadFile(prefabFile, json, ExpectOK{});
        auto const exported = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});
        ASSERT_TRUE(exported.contactFilter.has_value());
        auto const& filter = *exported.contactFilter;
        ASSERT_TRUE(filter.actorContactAsymmetric.has_value());
        ASSERT_TRUE(filter.actorContactSymmetric.has_value());
        ASSERT_TRUE(filter.layerContactAsymmetric.has_value());
        ASSERT_TRUE(filter.layerContactSymmetric.has_value());
        EXPECT_EQ(expectedActorAsymmetric, actorPairs(*filter.actorContactAsymmetric));
        EXPECT_EQ(expectedActorSymmetric, actorPairs(*filter.actorContactSymmetric));
        EXPECT_EQ(expectedLayerAsymmetric, layerPairs(*filter.layerContactAsymmetric));
        EXPECT_EQ(expectedLayerSymmetric, layerPairs(*filter.layerContactSymmetric));
      };

  std::string jsonA;
  std::string jsonB;
  exportWithOrder(tempDirA.Path(), false, jsonA);
  exportWithOrder(tempDirB.Path(), true, jsonB);
  EXPECT_EQ(jsonA, jsonB);
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, ContactPairOverridesAreDeterministic) {
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  auto tempDirA = CreateTempDirectory("contact_pair_override_order_a", ExpectOK{});
  auto tempDirB = CreateTempDirectory("contact_pair_override_order_b", ExpectOK{});

  constexpr int kNumPairs = 24;
  DynamicArray<std::pair<std::string, real>> expectedEntries;
  for (int i = 0; i < kNumPairs; ++i) {
    expectedEntries.push_back({"Actor" + std::to_string(i), static_cast<real>(i) * 0.01_r});
  }
  std::sort(expectedEntries.begin(), expectedEntries.end());

  auto exportWithOrder = [&](std::filesystem::path const& outputDir,
                             bool reverse,
                             std::string& json) {
    auto* scene = context->CreateScene("pair_override_determinism");
    MOCHI_DEFER(context->DestroyScene(scene));
    Actor* const anchor = CreateRigidCubeActor(context, scene, "Anchor");
    DynamicArray<Actor*> actors;
    for (int i = 0; i < kNumPairs; ++i) {
      actors.push_back(CreateRigidCubeActor(context, scene, ("Actor" + std::to_string(i)).c_str()));
    }

    for (int n = 0; n < kNumPairs; ++n) {
      int const i = reverse ? kNumPairs - 1 - n : n;
      ContactPairParamsOverride params;
      params.coulombFrictionCoefficient = static_cast<real>(i) * 0.01_r;
      scene->SetContactPairParamsOverride(
          actors[i]->GetHandle(), anchor->GetHandle(), params, ExpectOK{});
    }

    auto const prefabFile = ExportScene(scene, "pair_overrides", outputDir);
    ReadFile(prefabFile, json, ExpectOK{});
    auto const exported = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});
    ASSERT_TRUE(exported.contactPairParamsOverrides.has_value());
    ASSERT_EQ(kNumPairs, isize(*exported.contactPairParamsOverrides));
    for (int i = 0; i < kNumPairs; ++i) {
      auto const& entry = (*exported.contactPairParamsOverrides)[i];
      ASSERT_EQ(2, isize(entry.actors));
      EXPECT_STREQ(expectedEntries[i].first.c_str(), entry.actors[0].c_str());
      EXPECT_STREQ("Anchor", entry.actors[1].c_str());
      EXPECT_EQ(
          std::optional<real>{expectedEntries[i].second},
          entry.paramsOverride.coulombFrictionCoefficient);
    }
  };

  std::string jsonA;
  std::string jsonB;
  exportWithOrder(tempDirA.Path(), false, jsonA);
  exportWithOrder(tempDirB.Path(), true, jsonB);
  EXPECT_EQ(jsonA, jsonB);
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, PrefabExport, ContactPairOverridesExcludeAndRoundTrip) {
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  auto* scene = context->CreateScene("pair_override_exclusion");
  MOCHI_DEFER(context->DestroyScene(scene));
  Actor* const actorA = CreateRigidCubeActor(context, scene, "ActorA");
  Actor* const actorB = CreateRigidCubeActor(context, scene, "ActorB");
  Actor* const actorC = CreateRigidCubeActor(context, scene, "ActorC");

  auto setOverride = [&](Actor* first, Actor* second, real value) {
    ContactPairParamsOverride params;
    params.penaltyCoefficient = value;
    scene->SetContactPairParamsOverride(
        first->GetHandle(), second->GetHandle(), params, ExpectOK{});
  };
  setOverride(actorA, actorB, 1_r);
  setOverride(actorA, actorC, 2_r);
  setOverride(actorB, actorC, 3_r);

  auto tempDir = CreateTempDirectory("contact_pair_override_exclusion", ExpectOK{});
  DynamicArray<ActorHandle> excluded{actorB->GetHandle()};
  prefab::ExportSceneExcluding(
      scene, "pair_overrides", tempDir.Path().string(), excluded, ExpectOK{});
  auto const prefabFile = tempDir.Path() / "pair_overrides" / "pair_overrides.mochi_scene";
  auto exported = prefab::ShallowLoadFromFile(prefabFile.string(), ExpectOK{});
  ASSERT_TRUE(exported.contactPairParamsOverrides.has_value());
  ASSERT_EQ(1, isize(*exported.contactPairParamsOverrides));
  auto const& entry = (*exported.contactPairParamsOverrides)[0];
  ASSERT_EQ(2, isize(entry.actors));
  EXPECT_STREQ("ActorA", entry.actors[0].c_str());
  EXPECT_STREQ("ActorC", entry.actors[1].c_str());
  EXPECT_EQ(std::optional<real>{2_r}, entry.paramsOverride.penaltyCoefficient);

  auto loaded =
      prefab::LoadFromFile(prefabFile.string(), tempDir.Path().string(), context, ExpectOK{});
  auto* roundTripScene = context->CreateScene("roundtrip");
  MOCHI_DEFER(context->DestroyScene(roundTripScene));
  prefab::AddToScene(loaded, roundTripScene, {}, ExpectOK{});
  ActorHandle roundTripA;
  ActorHandle roundTripC;
  roundTripScene->ForEachActor([&](Actor const* actor) {
    if (std::string_view(actor->GetName()) == "ActorA") {
      roundTripA = actor->GetHandle();
    } else if (std::string_view(actor->GetName()) == "ActorC") {
      roundTripC = actor->GetHandle();
    }
  });
  ASSERT_TRUE(roundTripA.IsValid());
  ASSERT_TRUE(roundTripC.IsValid());
  EXPECT_EQ(
      std::optional<real>{2_r},
      roundTripScene->GetContactPairParamsOverride(roundTripA, roundTripC, ExpectOK{})
          .penaltyCoefficient);
}
