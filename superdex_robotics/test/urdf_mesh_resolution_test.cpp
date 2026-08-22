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

#include "mochi_bots_test_helpers.h"

#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/file_utils.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <string_view>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

namespace fs = std::filesystem;

class UrdfMeshResolutionTest : public testing::Test {
 protected:
  void SetUp() override {
    // ResolveMeshPath logs on the error channel for unresolved meshes; the test
    // framework's default log callback fails the test on any error log. Suppress it.
    _prevLogFn = GetLogCallback();
    SetLogCallback([](LogChannel, char const*, char const*, int) {});
  }

  void TearDown() override {
    SetLogCallback(_prevLogFn);
  }

  void WriteFileText(fs::path const& path, std::string_view contents) {
    mochi::WriteFile(path, contents, ExpectOK{});
  }

  // Single-link URDF whose visual mesh uses the ROS `package://<pkg>/<rel>` form.
  static std::string SingleMeshUrdf(std::string_view packageMeshUri) {
    return std::string(R"(<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename=")") +
        std::string(packageMeshUri) + R"("/>
      </geometry>
    </visual>
  </link>
</robot>
)";
  }

  mochi::TempDirCleanup _tempDirCleanup =
      mochi::CreateTempDirectory("urdf_mesh_resolution_test", ExpectOK{});
  fs::path _tempDir = _tempDirCleanup.Path();
  LogFn _prevLogFn{};
};

} // namespace

// Layout (under _tempDir), with NO package.xml anywhere — the package root is only
// identifiable by its directory name matching the `package://` package segment:
//   ur_description/
//     urdf/robot.urdf          (mesh references package://ur_description/meshes/base.dae)
//     meshes/base.dae
TEST_F(UrdfMeshResolutionTest, NoPackageXmlResolvesViaPackageNamedAncestor) {
  auto const packageDir = _tempDir / "ur_description";
  auto const urdfPath = packageDir / "urdf" / "robot.urdf";
  auto const meshPath = packageDir / "meshes" / "base.dae";

  WriteFileText(meshPath, "dummy");
  WriteFileText(urdfPath, SingleMeshUrdf("package://ur_description/meshes/base.dae"));

  BotPrefab const prefab = LoadBotPrefabFromUrdfFile(urdfPath.string(), ExpectOK{});

  ASSERT_FALSE(prefab.links.empty());
  EXPECT_EQ(
      std::string(prefab.links[0].renderModelFile.c_str()),
      fs::weakly_canonical(meshPath).string());
}

// Layout (under _tempDir), with a package.xml at the package root — the canonical
// ROS case, which must continue to resolve via FindUrdfPackageRoot:
//   ur_description/
//     package.xml
//     urdf/robot.urdf
//     meshes/base.dae
TEST_F(UrdfMeshResolutionTest, WithPackageXmlResolvesViaPackageRoot) {
  auto const packageDir = _tempDir / "ur_description";
  auto const urdfPath = packageDir / "urdf" / "robot.urdf";
  auto const meshPath = packageDir / "meshes" / "base.dae";

  WriteFileText(packageDir / "package.xml", "<package/>");
  WriteFileText(meshPath, "dummy");
  WriteFileText(urdfPath, SingleMeshUrdf("package://ur_description/meshes/base.dae"));

  BotPrefab const prefab = LoadBotPrefabFromUrdfFile(urdfPath.string(), ExpectOK{});

  ASSERT_FALSE(prefab.links.empty());
  EXPECT_EQ(
      std::string(prefab.links[0].renderModelFile.c_str()),
      fs::weakly_canonical(meshPath).string());
}

// Layout (under _tempDir), with NO package.xml and a malformed `package://` URI that
// OMITS the package-name segment (as in baxter.urdf: `package://meshes/...`). The
// first segment is a real subdirectory, so the full remainder must resolve against
// the package root (an ancestor of the URDF directory):
//   baxter_description/
//     urdf/robot.urdf          (mesh references package://meshes/base.dae)
//     meshes/base.dae
TEST_F(UrdfMeshResolutionTest, NoPackageXmlResolvesOmittedPackageSegment) {
  auto const packageDir = _tempDir / "baxter_description";
  auto const urdfPath = packageDir / "urdf" / "robot.urdf";
  auto const meshPath = packageDir / "meshes" / "base.dae";

  WriteFileText(meshPath, "dummy");
  WriteFileText(urdfPath, SingleMeshUrdf("package://meshes/base.dae"));

  BotPrefab const prefab = LoadBotPrefabFromUrdfFile(urdfPath.string(), ExpectOK{});

  ASSERT_FALSE(prefab.links.empty());
  EXPECT_EQ(
      std::string(prefab.links[0].renderModelFile.c_str()),
      fs::weakly_canonical(meshPath).string());
}

// Layout (under _tempDir): no package.xml and no ancestor directory named after the
// package — the mesh cannot be resolved on disk. The resolved path field is left empty
// (path-only semantics), while the raw URDF reference is surfaced through
// UrdfMeshReferences so the importer can flag the link's mesh as missing.
TEST_F(UrdfMeshResolutionTest, UnresolvableMeshPreservesRawReference) {
  auto const packageDir = _tempDir / "some_other_dir";
  auto const urdfPath = packageDir / "urdf" / "robot.urdf";

  WriteFileText(urdfPath, SingleMeshUrdf("package://ur_description/meshes/base.dae"));

  UrdfMeshReferences meshRefs;
  BotPrefab const prefab = LoadBotPrefabFromUrdfFile(urdfPath.string(), meshRefs, ExpectOK{});

  ASSERT_FALSE(prefab.links.empty());
  EXPECT_TRUE(prefab.links[0].renderModelFile.empty());
  ASSERT_EQ(meshRefs.links.size(), prefab.links.size());
  EXPECT_EQ(
      std::string(meshRefs.links[0].visual.c_str()), "package://ur_description/meshes/base.dae");
  EXPECT_TRUE(meshRefs.links[0].collision.empty());
}
