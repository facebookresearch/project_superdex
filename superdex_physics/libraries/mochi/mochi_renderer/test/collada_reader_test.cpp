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

// Unit tests for the COLLADA (.dae) reader. Fixtures are written to temp files
// and parsed, validating section/material resolution and up-axis conversion.

#include <gtest/gtest.h>

#include <mochi_renderer/utils.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/log.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using namespace mochi_renderer;

namespace {

// Two triangles in one geometry, each bound to a distinct material; Y-up.
char const* const kTwoMaterialDae = R"DAE(<?xml version="1.0"?>
<COLLADA version="1.4.1">
  <asset><up_axis>Y_UP</up_axis></asset>
  <library_effects>
    <effect id="RedFX"><profile_COMMON><technique sid="common"><phong>
      <diffuse><color>1 0 0 1</color></diffuse>
      <shininess><float>50</float></shininess>
    </phong></technique></profile_COMMON></effect>
    <effect id="GreenFX"><profile_COMMON><technique sid="common"><lambert>
      <diffuse><color>0 1 0 1</color></diffuse>
    </lambert></technique></profile_COMMON></effect>
  </library_effects>
  <library_materials>
    <material id="RedMat"><instance_effect url="#RedFX"/></material>
    <material id="GreenMat"><instance_effect url="#GreenFX"/></material>
  </library_materials>
  <library_geometries>
    <geometry id="Geo"><mesh>
      <source id="Geo-pos">
        <float_array id="Geo-pos-array" count="9">0 0 0 1 0 0 0 1 0</float_array>
        <technique_common><accessor source="#Geo-pos-array" count="3" stride="3">
          <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
        </accessor></technique_common>
      </source>
      <vertices id="Geo-vtx"><input semantic="POSITION" source="#Geo-pos"/></vertices>
      <triangles material="RedSym" count="1">
        <input semantic="VERTEX" source="#Geo-vtx" offset="0"/>
        <p>0 1 2</p>
      </triangles>
      <triangles material="GreenSym" count="1">
        <input semantic="VERTEX" source="#Geo-vtx" offset="0"/>
        <p>0 1 2</p>
      </triangles>
    </mesh></geometry>
  </library_geometries>
  <library_visual_scenes><visual_scene id="Scene"><node>
    <instance_geometry url="#Geo"><bind_material><technique_common>
      <instance_material symbol="RedSym" target="#RedMat"/>
      <instance_material symbol="GreenSym" target="#GreenMat"/>
    </technique_common></bind_material></instance_geometry>
  </node></visual_scene></library_visual_scenes>
</COLLADA>)DAE";

// Single triangle, Z-up, to verify the Z→Y rotation (x, y, z) → (x, z, -y).
char const* const kZUpDae = R"DAE(<?xml version="1.0"?>
<COLLADA version="1.4.1">
  <asset><up_axis>Z_UP</up_axis></asset>
  <library_geometries>
    <geometry id="Geo"><mesh>
      <source id="Geo-pos">
        <float_array id="Geo-pos-array" count="9">0 0 0 0 1 0 1 0 0</float_array>
        <technique_common><accessor source="#Geo-pos-array" count="3" stride="3">
          <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
        </accessor></technique_common>
      </source>
      <vertices id="Geo-vtx"><input semantic="POSITION" source="#Geo-pos"/></vertices>
      <triangles count="1">
        <input semantic="VERTEX" source="#Geo-vtx" offset="0"/>
        <p>0 1 2</p>
      </triangles>
    </mesh></geometry>
  </library_geometries>
</COLLADA>)DAE";

// Single triangle whose geometry is in local (millimeter-like) coordinates,
// with a node <matrix> applying a 0.001 scale. Mirrors Blender DAE exports.
char const* const kNodeMatrixDae = R"DAE(<?xml version="1.0"?>
<COLLADA version="1.4.1">
  <asset><up_axis>Y_UP</up_axis></asset>
  <library_geometries>
    <geometry id="Geo"><mesh>
      <source id="Geo-pos">
        <float_array id="Geo-pos-array" count="9">0 0 0 1000 0 0 0 1000 0</float_array>
        <technique_common><accessor source="#Geo-pos-array" count="3" stride="3">
          <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
        </accessor></technique_common>
      </source>
      <vertices id="Geo-vtx"><input semantic="POSITION" source="#Geo-pos"/></vertices>
      <triangles count="1">
        <input semantic="VERTEX" source="#Geo-vtx" offset="0"/>
        <p>0 1 2</p>
      </triangles>
    </mesh></geometry>
  </library_geometries>
  <library_visual_scenes><visual_scene id="Scene"><node id="N">
    <matrix sid="transform">0.001 0 0 0 0 0.001 0 0 0 0 0.001 0 0 0 0 1</matrix>
    <instance_geometry url="#Geo"/>
  </node></visual_scene></library_visual_scenes>
</COLLADA>)DAE";

// Writes a .dae fixture to a uniquely-named temp file (auto-deleted at
// end-of-scope by the returned RAII handle).
mochi::TempFileCleanup WriteTempDae(char const* contents, char const* label) {
  mochi::TempFileCleanup file = mochi::CreateTempFile(label, ".dae", mochi::test::ExpectOK{});
  std::ofstream out(file.Path());
  out << contents;
  return file;
}

} // namespace

TEST(ColladaReaderTest, TwoMaterialSectionsResolved) {
  mochi::TempFileCleanup const file = WriteTempDae(kTwoMaterialDae, "collada_reader_two_material");
  std::vector<MeshSection> const sections = ReadColladaFromFile(file.Path().string().c_str());
  ASSERT_EQ(sections.size(), 2u);

  MeshSection const& red = sections[0];
  EXPECT_FALSE(red.hasNormals);
  EXPECT_EQ(red.positions.size(), 9u);
  EXPECT_EQ(red.indices.size(), 3u);
  EXPECT_NEAR(red.baseColor[0], 1.0f, 1e-6f);
  EXPECT_NEAR(red.baseColor[1], 0.0f, 1e-6f);
  EXPECT_NEAR(red.baseColor[2], 0.0f, 1e-6f);
  EXPECT_NEAR(red.metallic, 0.0f, 1e-6f);
  // shininess 50 → roughness = sqrt(2/52).
  EXPECT_NEAR(red.roughness, std::sqrt(2.0f / 52.0f), 1e-5f);

  MeshSection const& green = sections[1];
  EXPECT_NEAR(green.baseColor[0], 0.0f, 1e-6f);
  EXPECT_NEAR(green.baseColor[1], 1.0f, 1e-6f);
  EXPECT_NEAR(green.baseColor[2], 0.0f, 1e-6f);
  // No shininess → default 20 → roughness = sqrt(2/22).
  EXPECT_NEAR(green.roughness, std::sqrt(2.0f / 22.0f), 1e-5f);
}

TEST(ColladaReaderTest, ZUpConvertedToYUp) {
  mochi::TempFileCleanup const file = WriteTempDae(kZUpDae, "collada_reader_zup");
  std::vector<MeshSection> const sections = ReadColladaFromFile(file.Path().string().c_str());
  ASSERT_EQ(sections.size(), 1u);

  // Source vertex 1 is (0, 1, 0); Z→Y maps it to (0, 0, -1).
  MeshSection const& s = sections[0];
  ASSERT_EQ(s.positions.size(), 9u);
  EXPECT_NEAR(s.positions[3], 0.0f, 1e-6f);
  EXPECT_NEAR(s.positions[4], 0.0f, 1e-6f);
  EXPECT_NEAR(s.positions[5], -1.0f, 1e-6f);
}

TEST(ColladaReaderTest, NodeMatrixScalesGeometryToMeters) {
  mochi::TempFileCleanup const file = WriteTempDae(kNodeMatrixDae, "collada_reader_node_matrix");
  std::vector<MeshSection> const sections = ReadColladaFromFile(file.Path().string().c_str());
  ASSERT_EQ(sections.size(), 1u);

  // Source vertex 1 is (1000, 0, 0); the node's 0.001 scale maps it to (1, 0, 0).
  MeshSection const& s = sections[0];
  ASSERT_EQ(s.positions.size(), 9u);
  EXPECT_NEAR(s.positions[3], 1.0f, 1e-6f);
  EXPECT_NEAR(s.positions[4], 0.0f, 1e-6f);
  EXPECT_NEAR(s.positions[5], 0.0f, 1e-6f);
}

TEST(ColladaReaderTest, MissingFileReturnsEmpty) {
  // The reader warns on a failed open; silence it so the test harness does not
  // treat the expected warning as a failure.
  auto const prevLogFn = mochi::GetLogCallback();
  mochi::SetLogCallback(nullptr);
  MOCHI_DEFER(mochi::SetLogCallback(prevLogFn));

  EXPECT_TRUE(ReadColladaFromFile("/nonexistent/path/missing.dae").empty());
}
