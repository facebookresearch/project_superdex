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

// Headless test that validates STL file parsing and GLB conversion
// without requiring a Filament Engine or GPU.

#include <gtest/gtest.h>

#include <mochi_renderer/utils.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <vector>

#include "test_assets.h"
#include "third_party/stl_reader.h"

using namespace mochi_renderer;

namespace {

std::string GetBinaryCubeStlPath() {
  return test::GetTestAssetPath("basic_shapes/binary_cube.stl");
}

} // namespace

// --- STL Parsing Tests ---

TEST(StlImportTest, BinaryCubeFileExists) {
  std::string path = GetBinaryCubeStlPath();
  ASSERT_TRUE(std::filesystem::exists(path)) << "binary_cube.stl not found at: " << path;
}

TEST(StlImportTest, BinaryCubeParsesSuccessfully) {
  std::string path = GetBinaryCubeStlPath();
  ASSERT_TRUE(std::filesystem::exists(path)) << path;

  std::vector<float> positions;
  std::vector<float> faceNormals;
  std::vector<int> indices;
  std::vector<int> solids;

  ASSERT_NO_THROW(stl_reader::ReadStlFile(path.c_str(), positions, faceNormals, indices, solids))
      << "stl_reader::ReadStlFile threw an exception";

  // A cube has 8 unique vertices and 12 triangles (2 per face × 6 faces).
  EXPECT_EQ(positions.size() % 3, 0u) << "Position count must be a multiple of 3";
  EXPECT_EQ(faceNormals.size() % 3, 0u) << "Normal count must be a multiple of 3";
  EXPECT_EQ(indices.size() % 3, 0u) << "Index count must be a multiple of 3";

  size_t numVertices = positions.size() / 3;
  size_t numTriangles = indices.size() / 3;

  EXPECT_GT(numVertices, 0u) << "No vertices parsed";
  EXPECT_EQ(numTriangles, 12u) << "A cube should have 12 triangles";
  EXPECT_EQ(faceNormals.size() / 3, numTriangles);

  // Solids: stl_reader returns [begin, end) pairs, so a single solid = [0, numTris].
  EXPECT_LE(solids.size(), 2u) << "Expected at most one solid (LoadStl rejects multi-solid)";

  // All indices must be in range [0, numVertices).
  for (size_t i = 0; i < indices.size(); ++i) {
    EXPECT_GE(indices[i], 0) << "Negative index at position " << i;
    EXPECT_LT(static_cast<size_t>(indices[i]), numVertices)
        << "Out-of-range index at position " << i;
  }

  // All positions and normals must be finite.
  for (size_t i = 0; i < positions.size(); ++i) {
    EXPECT_TRUE(std::isfinite(positions[i])) << "Non-finite position at index " << i;
  }
  for (size_t i = 0; i < faceNormals.size(); ++i) {
    EXPECT_TRUE(std::isfinite(faceNormals[i])) << "Non-finite normal at index " << i;
  }
}

TEST(StlImportTest, BinaryCubeVertexNormals) {
  std::string path = GetBinaryCubeStlPath();
  ASSERT_TRUE(std::filesystem::exists(path));

  std::vector<float> positions, faceNormals, vertexNormals;
  std::vector<int> indices, solids;
  stl_reader::ReadStlFile(path.c_str(), positions, faceNormals, indices, solids);

  ComputeVertexNormalsAngleWeighted(positions, faceNormals, indices, vertexNormals);

  EXPECT_EQ(vertexNormals.size(), positions.size())
      << "Vertex normals should have same size as positions";

  // All vertex normals should be unit-length (or zero for degenerate verts).
  size_t numVertices = positions.size() / 3;
  for (size_t v = 0; v < numVertices; ++v) {
    float nx = vertexNormals[v * 3 + 0];
    float ny = vertexNormals[v * 3 + 1];
    float nz = vertexNormals[v * 3 + 2];
    float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    EXPECT_NEAR(len, 1.0f, 1e-4f) << "Vertex normal " << v << " not unit length: " << len;
  }
}

TEST(StlImportTest, BinaryCubeGlbConversion) {
  std::string path = GetBinaryCubeStlPath();
  ASSERT_TRUE(std::filesystem::exists(path));

  std::vector<float> positions, faceNormals, vertexNormals;
  std::vector<int> indices, solids;
  stl_reader::ReadStlFile(path.c_str(), positions, faceNormals, indices, solids);
  ComputeVertexNormalsAngleWeighted(positions, faceNormals, indices, vertexNormals);

  MeshSection section;
  section.positions = positions;
  section.normals = vertexNormals;
  section.indices = indices;
  section.hasNormals = (section.normals.size() == section.positions.size());
  std::vector<uint8_t> glb = BuildGlbFromMeshSections({std::move(section)});

  // Validate GLB header.
  ASSERT_GE(glb.size(), 12u) << "GLB too small for header";

  uint32_t magic = 0, version = 0, length = 0;
  std::memcpy(&magic, glb.data(), 4);
  std::memcpy(&version, glb.data() + 4, 4);
  std::memcpy(&length, glb.data() + 8, 4);

  EXPECT_EQ(magic, 0x46546C67u) << "GLB magic should be 'glTF'";
  EXPECT_EQ(version, 2u) << "GLB version should be 2";
  EXPECT_EQ(length, static_cast<uint32_t>(glb.size())) << "GLB length mismatch";

  // Validate JSON chunk header.
  ASSERT_GE(glb.size(), 20u);
  uint32_t jsonChunkLen = 0, jsonChunkType = 0;
  std::memcpy(&jsonChunkLen, glb.data() + 12, 4);
  std::memcpy(&jsonChunkType, glb.data() + 16, 4);
  EXPECT_EQ(jsonChunkType, 0x4E4F534Au) << "First chunk should be JSON";
  EXPECT_GT(jsonChunkLen, 0u);

  // Validate BIN chunk header.
  size_t binChunkOffset = 12 + 8 + jsonChunkLen;
  ASSERT_GE(glb.size(), binChunkOffset + 8);
  uint32_t binChunkLen = 0, binChunkType = 0;
  std::memcpy(&binChunkLen, glb.data() + binChunkOffset, 4);
  std::memcpy(&binChunkType, glb.data() + binChunkOffset + 4, 4);
  EXPECT_EQ(binChunkType, 0x004E4942u) << "Second chunk should be BIN";

  // BIN chunk should contain positions + normals + indices.
  size_t expectedBinData = positions.size() * sizeof(float) + vertexNormals.size() * sizeof(float) +
      indices.size() * sizeof(uint32_t);
  EXPECT_GE(binChunkLen, static_cast<uint32_t>(expectedBinData));
}

TEST(StlImportTest, BinaryCubeFullPipeline) {
  // End-to-end: parse STL → compute normals → build GLB → verify no crash.
  // This mirrors ResourceManager::LoadStl() minus the Filament calls.
  std::string path = GetBinaryCubeStlPath();
  ASSERT_TRUE(std::filesystem::exists(path));

  std::vector<float> positions, faceNormals, vertexNormals;
  std::vector<int> indices, solids;
  ASSERT_NO_THROW(stl_reader::ReadStlFile(path.c_str(), positions, faceNormals, indices, solids));
  ASSERT_LE(solids.size(), 2u) << "LoadStl rejects multi-solid STL files";

  ComputeVertexNormalsAngleWeighted(positions, faceNormals, indices, vertexNormals);

  MeshSection section;
  section.positions = positions;
  section.normals = vertexNormals;
  section.indices = indices;
  section.hasNormals = (section.normals.size() == section.positions.size());
  std::vector<uint8_t> glb;
  ASSERT_NO_THROW(glb = BuildGlbFromMeshSections({std::move(section)}));
  EXPECT_GT(glb.size(), 12u) << "GLB output is empty or too small";
}
