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
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/src/mochi_scene.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <filesystem>
#include <fstream>
#include <string>

// Validation and scratch-file helpers shared by mochi_prefab_export_test.cpp and its
// internal/ counterpart, which exercises the same export paths against prefabs that are
// not shipped externally.

namespace mochi::test {

// Validates that a shape file exists and has a valid path
inline void ValidateShapeFile(
    std::string const& shapeFile,
    std::filesystem::path const& tempDir,
    std::filesystem::path const& prefabFile,
    std::string const& actorType) {
  EXPECT_FALSE(shapeFile.empty());

  // Resolve prefab-relative paths
  std::filesystem::path meshPath;
  if (shapeFile.starts_with("./")) {
    auto prefabDir = prefabFile.parent_path();
    meshPath = prefabDir / std::filesystem::path(shapeFile).relative_path();
  } else {
    meshPath = tempDir / shapeFile;
  }

  EXPECT_TRUE(std::filesystem::exists(meshPath))
      << actorType << " mesh file should exist: " << shapeFile
      << " at resolved path: " << meshPath.string();
}

// Validates that actors exist (count > 0)
template <typename T>
inline void ValidateActorsExist(DynamicArray<T> const& actors) {
  EXPECT_GT(actors.size(), 0);
}

inline void ValidateArticulatedActorStructure(
    prefab::ArticulatedActorPrefab const& actor,
    std::filesystem::path const& tempDir,
    std::filesystem::path const& prefabFile,
    size_t expectedLinkCount) {
  EXPECT_FALSE(actor.name.empty());
  EXPECT_EQ(actor.links.size(), expectedLinkCount);
  EXPECT_EQ(actor.joints.size(), expectedLinkCount); // One joint per link (including root)

  for (size_t i = 0; i < actor.links.size(); ++i) {
    ValidateShapeFile(
        std::string(actor.links[i].shapeFile),
        tempDir,
        prefabFile,
        "Articulated link " + std::to_string(i));
  }
}

inline void ValidateArticulatedActorLinks(
    prefab::ArticulatedActorPrefab const& actor,
    std::filesystem::path const& tempDir,
    std::filesystem::path const& prefabFile) {
  for (size_t i = 0; i < actor.links.size(); ++i) {
    auto const& link = actor.links[i];

    EXPECT_FALSE(link.name.empty()) << "Link " << i << " should have a name";
    EXPECT_FALSE(link.shapeFile.empty()) << "Link " << i << " should have a shape file";

    if (!link.shapeFile.empty()) {
      ValidateShapeFile(
          std::string(link.shapeFile),
          tempDir,
          prefabFile,
          "Articulated link " + std::to_string(i));
    }

    if (!link.name.empty()) {
      EXPECT_NE(link.colliderType, ColliderType::None)
          << "Named link '" << link.name << "' should have a collider type";
    }

    EXPECT_GE(link.contact.coulombFrictionCoefficient, 0_r)
        << "Link " << i << " coulomb friction should be valid";
    EXPECT_GE(link.contact.viscousFrictionCoefficient, 0_r)
        << "Link " << i << " viscous friction should be valid";
    EXPECT_GE(link.contact.penaltyCoefficient, 0_r)
        << "Link " << i << " penalty coefficient should be valid";
  }
}

inline void ValidateJointParameters(prefab::ArticulatedActorPrefab const& actor) {
  for (size_t j = 0; j < actor.joints.size(); ++j) {
    auto const& joint = actor.joints[j];

    if (joint.inertia.has_value()) {
      EXPECT_GE(*joint.inertia, 0_r) << "Joint inertia coefficient " << j << " should be valid";
    }

    EXPECT_GE(joint.friction.viscous, 0_r)
        << "Joint viscous damping coefficient " << j << " should be valid";
    EXPECT_GE(joint.friction.coulomb, 0_r)
        << "Joint Coulomb friction coefficient " << j << " should be valid";
  }
}

inline std::filesystem::path CreateTempPrefabFile(
    std::string const& jsonContent,
    std::string const& filename,
    std::filesystem::path const& tempDir) {
  auto prefabPath = tempDir / filename;
  std::ofstream file(prefabPath);
  file << jsonContent;
  file.close();
  return prefabPath;
}

inline std::filesystem::path
ExportScene(Scene* scene, std::string const& sceneName, std::filesystem::path const& outputDir) {
  prefab::ExportScene(scene, sceneName, outputDir.string(), ExpectOK{});
  return outputDir / sceneName / (sceneName + ".mochi_scene");
}

inline void TestRoundTrip(
    Scene* originalScene,
    Context* context,
    std::filesystem::path const& prefabFile,
    std::filesystem::path const& outputDir) {
  auto exportedPrefab =
      prefab::LoadFromFile(prefabFile.string(), outputDir.string(), context, ExpectOK{});
  auto* newScene = context->CreateScene("roundtrip");
  MOCHI_DEFER(context->DestroyScene(newScene));

  prefab::AddToScene(exportedPrefab, newScene, prefab::PrefabParams{}, ExpectOK{});
  EXPECT_EQ(originalScene->GetNumActors(), newScene->GetNumActors());
}

} // namespace mochi::test
