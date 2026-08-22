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

#include "assets/asset.h"

#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/file_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

using namespace superdex::studio;
using namespace mochi::test;

namespace {

namespace fs = std::filesystem;

class AssetDiscoveryTest : public testing::Test {
 protected:
  mochi::TempDirCleanup _tempDirCleanup =
      mochi::CreateTempDirectory("asset_discovery_test", ExpectOK{});
  fs::path _tempDir = _tempDirCleanup.Path();

  // Create an empty file at `relativePath` under the temp dir, making its parents.
  fs::path Touch(std::string const& relativePath) {
    auto path = _tempDir / relativePath;
    fs::create_directories(path.parent_path());
    mochi::WriteFile(path, std::string_view{}, ExpectOK{});
    return path;
  }

  fs::path Dir(std::string const& relativePath) {
    auto path = _tempDir / relativePath;
    fs::create_directories(path);
    return path;
  }

  static std::string Find(std::string const& baseName, AssetType type, fs::path const& originDir) {
    return FindAssetForSlot(baseName, type, mochi::Path{originDir});
  }

  static void ExpectFound(std::string const& actual, fs::path const& expected) {
    EXPECT_EQ(
        superdex::robotics::NormalizeBotPath(actual),
        superdex::robotics::NormalizeBotPath(expected));
  }
};

// Classification reads the same table discovery ranks with, so it is worth pinning directly --
// especially the pairs where one extension could be mistaken for another, and the `.h5` that is
// only a mochi model when it carries the `.mochi` infix.
TEST_F(AssetDiscoveryTest, ClassifiesEveryKnownExtension) {
  EXPECT_EQ(ClassifyAssetTypeByFilename("x.glb"), AssetType::RenderModel);
  EXPECT_EQ(ClassifyAssetTypeByFilename("x.gltf"), AssetType::RenderModel);
  EXPECT_EQ(ClassifyAssetTypeByFilename("x.dae"), AssetType::RenderModel);
  EXPECT_EQ(ClassifyAssetTypeByFilename("x.obj"), AssetType::RenderModel);
  EXPECT_EQ(ClassifyAssetTypeByFilename("x.stp"), AssetType::CadModel);
  EXPECT_EQ(ClassifyAssetTypeByFilename("x.step"), AssetType::CadModel);
  EXPECT_EQ(ClassifyAssetTypeByFilename("x.stl"), AssetType::CadModel);
  EXPECT_EQ(ClassifyAssetTypeByFilename("x.STL"), AssetType::CadModel);
  EXPECT_EQ(ClassifyAssetTypeByFilename("x.mochi.h5"), AssetType::MochiModel);
  EXPECT_EQ(ClassifyAssetTypeByFilename("x.superdex_bot"), AssetType::Bot);
  EXPECT_EQ(ClassifyAssetTypeByFilename("x.superdex_bot_archive"), AssetType::Bot);
  EXPECT_EQ(ClassifyAssetTypeByFilename("x.mochi_bot"), AssetType::Bot);
  EXPECT_EQ(ClassifyAssetTypeByFilename("x.mochi_bot_archive"), AssetType::Bot);
  EXPECT_EQ(ClassifyAssetTypeByFilename("x.mochi_scene"), AssetType::MochiPrefab);
  EXPECT_EQ(ClassifyAssetTypeByFilename("x.mochi_prefab"), AssetType::MochiPrefab);

  EXPECT_EQ(ClassifyAssetTypeByFilename("x.h5"), AssetType::Unknown);
  EXPECT_EQ(ClassifyAssetTypeByFilename("x.urdf"), AssetType::Unknown);
  EXPECT_EQ(ClassifyAssetTypeByFilename("x"), AssetType::Unknown);
}

// Step 1: the role folder for the requested type, the layout the assets ship in.
TEST_F(AssetDiscoveryTest, FindsModelsInTheirRoleFolders) {
  Touch("fr3/cad/link0.step");
  auto const render = Touch("fr3/render/link0.glb");
  auto const collision = Touch("fr3/collision/link0.mochi.h5");
  auto const originDir = _tempDir / "fr3" / "cad";

  ExpectFound(Find("link0", AssetType::RenderModel, originDir), render);
  ExpectFound(Find("link0", AssetType::MochiModel, originDir), collision);
}

// Step 2: a flat asset resolves from the base folder itself.
TEST_F(AssetDiscoveryTest, FindsModelsInAFlatAsset) {
  Touch("axis_gizmos/axis_gizmo.stl");
  auto const render = Touch("axis_gizmos/axis_gizmo.glb");
  auto const originDir = _tempDir / "axis_gizmos";

  ExpectFound(Find("axis_gizmo", AssetType::RenderModel, originDir), render);
}

// Precedence, not availability, decides: the role folder outranks the base folder.
TEST_F(AssetDiscoveryTest, RoleFolderOutranksTheBaseFolder) {
  Touch("mixed/block.glb");
  auto const inRoleFolder = Touch("mixed/render/block.glb");

  ExpectFound(Find("block", AssetType::RenderModel, _tempDir / "mixed"), inRoleFolder);
}

// Rule D: several extensions classify as one type, so a fixed preference -- not the order the
// directory happens to enumerate in -- picks the winner.
TEST_F(AssetDiscoveryTest, PrefersTheHigherRankedExtension) {
  auto const glb = Touch("ranked/render/part.glb");
  Touch("ranked/render/part.obj");
  Touch("ranked/render/part.dae");
  auto const stp = Touch("ranked/cad/part.stp");
  Touch("ranked/cad/part.stl");
  auto const originDir = _tempDir / "ranked" / "render";

  ExpectFound(Find("part", AssetType::RenderModel, originDir), glb);
  ExpectFound(Find("part", AssetType::CadModel, originDir), stp);
}

// Extensions already match case-insensitively; base names have to as well, or a tree authored on
// Windows associates differently on Linux.
TEST_F(AssetDiscoveryTest, MatchesBaseNamesCaseInsensitively) {
  auto const render = Touch("cased/render/Block.glb");
  ExpectFound(Find("block", AssetType::RenderModel, _tempDir / "cased" / "cad"), render);
}

/* Step 4, the case the shipped dg5f_short_d405 bot is in: a URDF import left the `.stl` it was
 * handed in `collision/`. It is a CAD-typed file living under the collision role, and the CAD slot
 * has to find it -- while the mochi slot still resolves to the `.mochi.h5` beside it, because a
 * file's type comes from its extension and never from its folder. */
TEST_F(AssetDiscoveryTest, FindsACadModelLeftInTheCollisionFolder) {
  auto const stl = Touch("d405/collision/camera_mount_collision.stl");
  auto const mochi = Touch("d405/collision/camera_mount_collision.mochi.h5");
  auto const originDir = _tempDir / "d405" / "collision";

  ExpectFound(Find("camera_mount_collision", AssetType::CadModel, originDir), stl);
  ExpectFound(Find("camera_mount_collision", AssetType::MochiModel, originDir), mochi);
}

/* A role folder partitioned into subfolders -- `cad/internal/` for CAD that must not be
 * open-sourced -- still resolves, from a sibling role folder as well as from within. */
TEST_F(AssetDiscoveryTest, FindsAModelInsideARoleSubfolder) {
  auto const stp = Touch("ur7e/cad/internal/link0.stp");
  auto const render = Touch("ur7e/render/link0.glb");

  ExpectFound(Find("link0", AssetType::CadModel, _tempDir / "ur7e" / "render"), stp);
  ExpectFound(
      Find("link0", AssetType::RenderModel, _tempDir / "ur7e" / "cad" / "internal"), render);
}

// Breadth first: every match at one depth is considered before descending, so a copy at the role
// folder's root always outranks one tucked into a subfolder.
TEST_F(AssetDiscoveryTest, ShallowestMatchInARoleFolderWins) {
  auto const shallow = Touch("depth/render/part.glb");
  Touch("depth/render/internal/part.glb");

  ExpectFound(Find("part", AssetType::RenderModel, _tempDir / "depth" / "cad"), shallow);
}

// Two subfolders at the same depth are decided by path, not by directory-iteration order, so the
// same tree resolves the same way on every machine.
TEST_F(AssetDiscoveryTest, EquallyDeepSubfoldersAreDecidedLexicographically) {
  auto const alpha = Touch("tie/render/alpha/part.glb");
  Touch("tie/render/beta/part.glb");

  ExpectFound(Find("part", AssetType::RenderModel, _tempDir / "tie" / "cad"), alpha);
}

/* The shadowing predicate the Model Editor keys its "open alone" rule off: discovery run for a
 * shadowed file's own type resolves to the winner instead of to itself, while the winner resolves
 * to itself. That is what keeps the association a partition -- opening any member of a set opens
 * that same set, and a shadowed copy drags nothing in with it. */
TEST_F(AssetDiscoveryTest, DiscoveryResolvesToTheWinnerNotAShadowedCopy) {
  auto const winner = Touch("shadow/render/part.glb");
  auto const shadowed = Touch("shadow/render/internal/part.glb");

  ExpectFound(
      Find("part", AssetType::RenderModel, mochi::Path{winner}.GetParentPath().AsFilesystemPath()),
      winner);
  ExpectFound(
      Find(
          "part", AssetType::RenderModel, mochi::Path{shadowed}.GetParentPath().AsFilesystemPath()),
      winner);
}

/* A canonical model's generated files keep the base-name key every pipeline already on disk uses,
 * so one pipeline serves the whole cad/render/collision set and nothing needs migrating. */
TEST_F(AssetDiscoveryTest, CanonicalModelKeepsTheBaseNamedPipelinePath) {
  auto const glb = Touch("gen/render/part.glb");

  EXPECT_EQ(
      superdex::robotics::NormalizeBotPath(
          AssetGeneratedFilePath(mochi::Path{glb}, /*isCanonical=*/true, ".StudioProcessing.json")),
      superdex::robotics::NormalizeBotPath(
          _tempDir / "gen" / "intermediates" / "part.StudioProcessing.json"));
}

/* The bug this keying exists for: a shadowed model shares its base name with the model shadowing
 * it, so keying on the base name would point both at one pipeline -- opening the shadowed copy
 * would load the canonical model's work and saving would overwrite it. It keys on its own file
 * name under a mirror of its location instead. */
TEST_F(AssetDiscoveryTest, ShadowedModelGetsItsOwnPipelinePath) {
  auto const canonical = Touch("gen/render/part.glb");
  auto const shadowed = Touch("gen/render/internal/part.glb");

  std::string const canonicalJson = AssetGeneratedFilePath(
      mochi::Path{canonical}, /*isCanonical=*/true, ".StudioProcessing.json");
  std::string const shadowedJson = AssetGeneratedFilePath(
      mochi::Path{shadowed}, /*isCanonical=*/false, ".StudioProcessing.json");

  EXPECT_NE(canonicalJson, shadowedJson);
  EXPECT_EQ(
      superdex::robotics::NormalizeBotPath(shadowedJson),
      superdex::robotics::NormalizeBotPath(
          _tempDir / "gen" / "intermediates" / "render" / "internal" /
          "part.glb.StudioProcessing.json"));
  // Stable: the same model resolves to the same file, which is what lets the work load back.
  EXPECT_EQ(
      shadowedJson,
      AssetGeneratedFilePath(
          mochi::Path{shadowed}, /*isCanonical=*/false, ".StudioProcessing.json"));
}

/* With no subpath to mirror -- a loose model in the base folder shadowed by one in a role folder --
 * the extension kept in the stem is what separates the two. */
TEST_F(AssetDiscoveryTest, ShadowedModelInTheBaseFolderStillGetsItsOwnPipelinePath) {
  Touch("gen/render/part.glb");
  auto const loose = Touch("gen/part.glb");

  EXPECT_EQ(
      superdex::robotics::NormalizeBotPath(AssetGeneratedFilePath(
          mochi::Path{loose}, /*isCanonical=*/false, ".StudioProcessing.json")),
      superdex::robotics::NormalizeBotPath(
          _tempDir / "gen" / "intermediates" / "part.glb.StudioProcessing.json"));
}

/* A model alone in a role subfolder is not shadowed -- nothing outranks it, so discovery resolves
 * to it and it is the canonical model of its set. It therefore keys on the base name like any
 * other canonical model, and does not get the mirrored path. */
TEST_F(AssetDiscoveryTest, SoleCopyInARoleSubfolderIsCanonicalAndKeysOnTheBaseName) {
  auto const only = Touch("gen/cad/internal/part.stp");

  // Discovery resolves to it, which is what makes it canonical.
  ExpectFound(Find("part", AssetType::CadModel, _tempDir / "gen" / "cad" / "internal"), only);
  EXPECT_EQ(
      superdex::robotics::NormalizeBotPath(AssetGeneratedFilePath(
          mochi::Path{only}, /*isCanonical=*/true, ".StudioProcessing.json")),
      superdex::robotics::NormalizeBotPath(
          _tempDir / "gen" / "intermediates" / "part.StudioProcessing.json"));
}

// Two shadowed copies of one name in different subfolders must not share a pipeline either.
TEST_F(AssetDiscoveryTest, ShadowedCopiesInDifferentSubfoldersDoNotShareAPipeline) {
  auto const alpha = Touch("gen/render/alpha/part.glb");
  auto const beta = Touch("gen/render/beta/part.glb");

  EXPECT_NE(
      AssetGeneratedFilePath(mochi::Path{alpha}, /*isCanonical=*/false, ".StudioProcessing.json"),
      AssetGeneratedFilePath(mochi::Path{beta}, /*isCanonical=*/false, ".StudioProcessing.json"));
}

/* Recursion stops at role folders. A base folder can hold whole other assets -- a two-handed bot
 * keeps `left/` and `right/`, each with role folders of its own -- so searching it as a tree would
 * let one side's models fill the other side's slots. */
TEST_F(AssetDiscoveryTest, DoesNotDescendFromTheBaseFolderIntoNestedAssets) {
  Touch("dg5f/left/render/hand.glb");
  Touch("dg5f/right/render/hand.glb");
  auto const originDir = Dir("dg5f");

  EXPECT_TRUE(Find("hand", AssetType::RenderModel, originDir).empty());
}

// Step 4 also covers the folder name a URDF source tree uses for the render role.
TEST_F(AssetDiscoveryTest, FindsARenderModelUnderTheVisualAlias) {
  auto const dae = Touch("fr3v2/visual/link0.dae");
  Touch("fr3v2/collision/link0.stl");

  ExpectFound(Find("link0", AssetType::RenderModel, _tempDir / "fr3v2" / "collision"), dae);
}

// Two equally-plausible candidates in folders with no precedence between them: fill nothing rather
// than feed a coin-flip into a pipeline.
TEST_F(AssetDiscoveryTest, AmbiguousCrossRoleCandidatesFillNothing) {
  auto const suppressWarning = SuppressLogWarning();

  Touch("ambiguous/cad/part.glb");
  Touch("ambiguous/visual/part.glb");
  Dir("ambiguous/collision");

  EXPECT_TRUE(Find("part", AssetType::RenderModel, _tempDir / "ambiguous" / "collision").empty());
}

// Intermediates hold generated output, which must never shadow the real asset.
TEST_F(AssetDiscoveryTest, NeverResolvesToIntermediates) {
  Touch("generated/intermediates/part.glb");
  Dir("generated/cad");

  EXPECT_TRUE(Find("part", AssetType::RenderModel, _tempDir / "generated" / "cad").empty());
}

/* The anchoring regression, from the discovery side: a flat asset must not reach past itself into
 * the category folder, where a same-named model belonging to an unrelated asset would match. */
TEST_F(AssetDiscoveryTest, DoesNotReachIntoASiblingAssetsFolder) {
  Touch("internal/render/part.glb"); // category-level folder, not this asset's
  Touch("internal/paper_cups/part.step");

  EXPECT_TRUE(Find("part", AssetType::RenderModel, _tempDir / "internal" / "paper_cups").empty());
}

} // namespace
