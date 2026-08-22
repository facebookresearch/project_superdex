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

// Headless test for the multi-primitive GLB writer (BuildGlbFromMeshSections).
// Validates GLB container structure and the per-section JSON without requiring
// a Filament Engine or GPU.

#include <gtest/gtest.h>

#include <mochi_renderer/utils.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/log.h>

#include <meshoptimizer.h>

#include "test_assets.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using namespace mochi_renderer;

namespace {

// Returns the JSON chunk text from a GLB byte buffer.
std::string ExtractJson(std::vector<uint8_t> const& glb) {
  EXPECT_GE(glb.size(), 20u);
  uint32_t jsonChunkLen = 0;
  std::memcpy(&jsonChunkLen, glb.data() + 12, 4);
  return {reinterpret_cast<char const*>(glb.data() + 20), jsonChunkLen};
}

size_t CountOccurrences(std::string const& haystack, std::string const& needle) {
  size_t count = 0;
  for (size_t pos = haystack.find(needle); pos != std::string::npos;
       pos = haystack.find(needle, pos + needle.size())) {
    ++count;
  }
  return count;
}

// Assembles a GLB byte buffer from a JSON descriptor and binary chunk, applying
// the spec-required 4-byte chunk padding (spaces for JSON, zeros for BIN).
std::vector<uint8_t> AssembleGlb(std::string json, std::vector<uint8_t> bin) {
  json.append((4 - (json.size() % 4)) % 4, ' ');
  bin.insert(bin.end(), (4 - (bin.size() % 4)) % 4, uint8_t{0});

  auto const jsonLen = static_cast<uint32_t>(json.size());
  auto const binLen = static_cast<uint32_t>(bin.size());
  auto const total = static_cast<uint32_t>(12 + 8 + jsonLen + 8 + binLen);

  std::vector<uint8_t> glb;
  glb.reserve(total);
  auto push32 = [&](uint32_t v) {
    uint8_t b[4];
    std::memcpy(b, &v, 4);
    glb.insert(glb.end(), b, b + 4);
  };
  push32(0x46546C67); // "glTF"
  push32(2);
  push32(total);
  push32(jsonLen);
  push32(0x4E4F534A); // "JSON"
  glb.insert(glb.end(), json.begin(), json.end());
  push32(binLen);
  push32(0x004E4942); // "BIN\0"
  glb.insert(glb.end(), bin.begin(), bin.end());
  return glb;
}

// Builds a GLB with a single non-indexed triangle primitive (POSITION only).
std::vector<uint8_t> BuildNoIndexGlb(std::vector<float> const& positions) {
  std::vector<uint8_t> bin(positions.size() * sizeof(float));
  std::memcpy(bin.data(), positions.data(), bin.size());
  size_t const byteLength = bin.size();
  std::string json =
      R"({"asset":{"version":"2.0"},"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],)";
  json += R"("accessors":[{"bufferView":0,"componentType":5126,"count":)";
  json += std::to_string(positions.size() / 3);
  json += R"(,"type":"VEC3"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":)";
  json += std::to_string(byteLength);
  json += R"(}],"buffers":[{"byteLength":)";
  json += std::to_string(byteLength);
  json += R"(}]})";
  return AssembleGlb(std::move(json), std::move(bin));
}

// A single triangle in the z=0 plane.
MeshSection MakeTriangle(std::array<float, 4> const& color) {
  MeshSection s;
  s.positions = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  s.normals = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
  s.hasNormals = true;
  s.indices = {0, 1, 2};
  s.baseColor = color;
  return s;
}

// Writes a binary GLB fixture to a uniquely-named temp file (auto-deleted at
// end-of-scope by the returned RAII handle).
mochi::TempFileCleanup WriteTempGlb(std::vector<uint8_t> const& glb, char const* label) {
  mochi::TempFileCleanup file = mochi::CreateTempFile(label, ".glb", mochi::test::ExpectOK{});
  std::ofstream out(file.Path(), std::ios::binary);
  out.write(reinterpret_cast<char const*>(glb.data()), static_cast<std::streamsize>(glb.size()));
  return file;
}

// The checked-in SOLIDWORKS export of a unit cube, Draco-compressed
// (extensionsRequired: KHR_draco_mesh_compression).
std::string DracoCubeGlbPath() {
  return test::GetTestAssetPath("basic_shapes/Cube.glb");
}

std::vector<uint8_t> ReadFileBytes(std::string const& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Offset of the BIN chunk's payload within a GLB byte buffer.
size_t BinChunkOffset(std::vector<uint8_t> const& glb) {
  uint32_t jsonChunkLen = 0;
  std::memcpy(&jsonChunkLen, glb.data() + 12, 4);
  return 12 + 8 + jsonChunkLen + 8;
}

// Builds a GLB whose POSITION and index bufferViews are EXT_meshopt_compression
// streams, encoded with the same library the reader decodes with.
std::vector<uint8_t> BuildMeshoptGlb(
    std::vector<float> const& positions,
    std::vector<uint32_t> const& indices) {
  size_t const vertexCount = positions.size() / 3;
  size_t const vertexStride = 3 * sizeof(float);

  std::vector<uint8_t> bin(meshopt_encodeVertexBufferBound(vertexCount, vertexStride));
  size_t const positionBytes = meshopt_encodeVertexBuffer(
      bin.data(), bin.size(), positions.data(), vertexCount, vertexStride);
  // EXT_meshopt_compression requires 4-byte-aligned stream offsets.
  bin.resize(positionBytes + (4 - (positionBytes % 4)) % 4);
  size_t const indexOffset = bin.size();

  bin.resize(indexOffset + meshopt_encodeIndexBufferBound(indices.size(), vertexCount));
  size_t const indexBytes = meshopt_encodeIndexBuffer(
      bin.data() + indexOffset, bin.size() - indexOffset, indices.data(), indices.size());
  bin.resize(indexOffset + indexBytes);

  std::string json = R"({"asset":{"version":"2.0"},)"
                     R"("extensionsUsed":["EXT_meshopt_compression"],)"
                     R"("extensionsRequired":["EXT_meshopt_compression"],)"
                     R"("meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],)"
                     R"("accessors":[{"bufferView":0,"componentType":5126,"count":)";
  json += std::to_string(vertexCount);
  json += R"(,"type":"VEC3"},{"bufferView":1,"componentType":5125,"count":)";
  json += std::to_string(indices.size());
  json += R"(,"type":"SCALAR"}],"bufferViews":[)";
  json += R"({"buffer":0,"byteOffset":0,"byteLength":)";
  json += std::to_string(vertexCount * vertexStride);
  json += R"(,"byteStride":)";
  json += std::to_string(vertexStride);
  json += R"(,"extensions":{"EXT_meshopt_compression":{"buffer":0,"byteOffset":0,"byteLength":)";
  json += std::to_string(positionBytes);
  json += R"(,"byteStride":)";
  json += std::to_string(vertexStride);
  json += R"(,"count":)";
  json += std::to_string(vertexCount);
  json += R"(,"mode":"ATTRIBUTES"}}},)";
  json += R"({"buffer":0,"byteOffset":0,"byteLength":)";
  json += std::to_string(indices.size() * sizeof(uint32_t));
  json += R"(,"extensions":{"EXT_meshopt_compression":{"buffer":0,"byteOffset":)";
  json += std::to_string(indexOffset);
  json += R"(,"byteLength":)";
  json += std::to_string(indexBytes);
  json += R"(,"byteStride":4,"count":)";
  json += std::to_string(indices.size());
  json += R"(,"mode":"TRIANGLES"}}}],"buffers":[{"byteLength":)";
  json += std::to_string(bin.size());
  json += R"(}]})";
  return AssembleGlb(std::move(json), std::move(bin));
}

// Builds a GLB with KHR_mesh_quantization SHORT positions under a node whose
// uniform scale restores their real extent.
std::vector<uint8_t> BuildQuantizedGlb(std::vector<int16_t> const& positions, float scale) {
  std::vector<uint8_t> bin(positions.size() * sizeof(int16_t));
  std::memcpy(bin.data(), positions.data(), bin.size());

  std::string json = R"({"asset":{"version":"2.0"},)"
                     R"("extensionsUsed":["KHR_mesh_quantization"],)"
                     R"("extensionsRequired":["KHR_mesh_quantization"],)"
                     R"("scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"scale":[)";
  std::string const scaleText = std::to_string(scale);
  json += scaleText + ',' + scaleText + ',' + scaleText;
  json += R"(]}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],)";
  json += R"("accessors":[{"bufferView":0,"componentType":5122,"count":)";
  json += std::to_string(positions.size() / 3);
  json += R"(,"type":"VEC3"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":)";
  json += std::to_string(bin.size());
  json += R"(}],"buffers":[{"byteLength":)";
  json += std::to_string(bin.size());
  json += R"(}]})";
  return AssembleGlb(std::move(json), std::move(bin));
}

} // namespace

TEST(GlbSectionsTest, TwoSectionsHeaderAndStructure) {
  std::vector<MeshSection> sections = {
      MakeTriangle({1.0f, 0.0f, 0.0f, 1.0f}),
      MakeTriangle({0.0f, 1.0f, 0.0f, 1.0f}),
  };

  std::vector<uint8_t> const glb = BuildGlbFromMeshSections(sections);

  ASSERT_GE(glb.size(), 12u);
  uint32_t magic = 0;
  uint32_t version = 0;
  uint32_t length = 0;
  std::memcpy(&magic, glb.data(), 4);
  std::memcpy(&version, glb.data() + 4, 4);
  std::memcpy(&length, glb.data() + 8, 4);
  EXPECT_EQ(magic, 0x46546C67u) << "GLB magic should be 'glTF'";
  EXPECT_EQ(version, 2u);
  EXPECT_EQ(length, static_cast<uint32_t>(glb.size()));

  std::string const json = ExtractJson(glb);
  // One primitive and one material per section.
  EXPECT_EQ(CountOccurrences(json, "\"attributes\""), 2u);
  EXPECT_EQ(CountOccurrences(json, "pbrMetallicRoughness"), 2u);
  // Every section's material is emitted two-sided.
  EXPECT_EQ(CountOccurrences(json, "\"doubleSided\":true"), 2u);
  // Each section emits 3 accessors / 3 bufferViews.
  EXPECT_EQ(CountOccurrences(json, "\"componentType\""), 6u);
  // The second section's POSITION accessor is index 3.
  EXPECT_NE(json.find("\"POSITION\":3"), std::string::npos);
  // Distinct base colors survive into the JSON.
  EXPECT_NE(json.find("\"baseColorFactor\":[1,0,0,1]"), std::string::npos);
  EXPECT_NE(json.find("\"baseColorFactor\":[0,1,0,1]"), std::string::npos);
}

TEST(GlbSectionsTest, EmptySectionsSkipped) {
  std::vector<MeshSection> sections = {
      MakeTriangle({1.0f, 1.0f, 1.0f, 1.0f}),
      MeshSection{}, // empty: no positions/indices
  };

  std::vector<uint8_t> const glb = BuildGlbFromMeshSections(sections);
  std::string const json = ExtractJson(glb);
  EXPECT_EQ(CountOccurrences(json, "\"attributes\""), 1u) << "Empty section should be skipped";
}

TEST(GlbSectionsTest, LeadingEmptySectionKeepsMaterialAlignment) {
  // A skipped empty section must not shift the material assigned to later
  // sections (regression: materials were once indexed against the original
  // section list rather than the non-empty layout list).
  std::vector<MeshSection> sections = {
      MeshSection{}, // leading empty section, skipped
      MakeTriangle({1.0f, 0.0f, 0.0f, 1.0f}),
      MakeTriangle({0.0f, 1.0f, 0.0f, 1.0f}),
  };

  std::vector<uint8_t> const glb = BuildGlbFromMeshSections(sections);
  std::string const json = ExtractJson(glb);

  EXPECT_EQ(CountOccurrences(json, "\"attributes\""), 2u);
  // The two non-empty sections keep their colors, in order.
  size_t const redPos = json.find("\"baseColorFactor\":[1,0,0,1]");
  size_t const greenPos = json.find("\"baseColorFactor\":[0,1,0,1]");
  ASSERT_NE(redPos, std::string::npos);
  ASSERT_NE(greenPos, std::string::npos);
  EXPECT_LT(redPos, greenPos);
  // The skipped section's default-gray material must not leak in.
  EXPECT_EQ(json.find("\"baseColorFactor\":[0.5,0.5,0.5,1]"), std::string::npos);
}

TEST(GlbSectionsTest, SectionWithoutNormalsOmitsNormalAttribute) {
  // A section lacking normals must emit only POSITION + indices (two accessors),
  // never a NORMAL accessor pointing at an empty bufferView.
  MeshSection triangle = MakeTriangle({1.0f, 1.0f, 1.0f, 1.0f});
  triangle.normals.clear();
  triangle.hasNormals = false;

  std::vector<uint8_t> const glb = BuildGlbFromMeshSections({triangle});
  std::string const json = ExtractJson(glb);

  EXPECT_EQ(json.find("NORMAL"), std::string::npos);
  EXPECT_EQ(CountOccurrences(json, "\"componentType\""), 2u);
}

TEST(GlbSectionsTest, NoGeometryReturnsEmpty) {
  std::vector<MeshSection> sections = {MeshSection{}};
  EXPECT_TRUE(BuildGlbFromMeshSections(sections).empty());
  EXPECT_TRUE(BuildGlbFromMeshSections({}).empty());
}

TEST(GlbSectionsTest, BinChunkContainsAllSectionData) {
  std::vector<MeshSection> sections = {
      MakeTriangle({1.0f, 0.0f, 0.0f, 1.0f}),
      MakeTriangle({0.0f, 1.0f, 0.0f, 1.0f}),
  };

  std::vector<uint8_t> const glb = BuildGlbFromMeshSections(sections);

  uint32_t jsonChunkLen = 0;
  std::memcpy(&jsonChunkLen, glb.data() + 12, 4);
  size_t const binChunkOffset = 12 + 8 + jsonChunkLen;
  ASSERT_GE(glb.size(), binChunkOffset + 8);
  uint32_t binChunkType = 0;
  std::memcpy(&binChunkType, glb.data() + binChunkOffset + 4, 4);
  EXPECT_EQ(binChunkType, 0x004E4942u) << "Second chunk should be BIN";

  size_t expectedBin = 0;
  for (MeshSection const& s : sections) {
    expectedBin += s.positions.size() * sizeof(float);
    expectedBin += s.normals.size() * sizeof(float);
    expectedBin += s.indices.size() * sizeof(uint32_t);
  }
  uint32_t binChunkLen = 0;
  std::memcpy(&binChunkLen, glb.data() + binChunkOffset, 4);
  EXPECT_GE(binChunkLen, static_cast<uint32_t>(expectedBin));
}

TEST(GlbReaderTest, RoundTripPositionsIndicesNormalsMaterial) {
  std::vector<MeshSection> sections = {
      MakeTriangle({1.0f, 0.0f, 0.0f, 1.0f}),
      MakeTriangle({0.0f, 1.0f, 0.0f, 1.0f}),
  };
  sections[0].metallic = 0.25f;
  sections[0].roughness = 0.75f;

  std::vector<uint8_t> const glb = BuildGlbFromMeshSections(sections);
  mochi::TempFileCleanup const file = WriteTempGlb(glb, "glb_sections_roundtrip");
  std::vector<MeshSection> const read = ReadGlbFromFile(file.Path().string().c_str());

  ASSERT_EQ(read.size(), 2u);
  std::vector<int> const expectedIndices = {0, 1, 2};
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(read[i].positions, sections[i].positions);
    EXPECT_EQ(read[i].normals, sections[i].normals);
    EXPECT_TRUE(read[i].hasNormals);
    EXPECT_EQ(read[i].indices, expectedIndices);
    EXPECT_EQ(read[i].baseColor, sections[i].baseColor);
  }
  EXPECT_NEAR(read[0].metallic, 0.25f, 1e-6f);
  EXPECT_NEAR(read[0].roughness, 0.75f, 1e-6f);
}

TEST(GlbReaderTest, MalformedFileReturnsEmpty) {
  // The reader warns on parse/open failure; silence it so the test harness does
  // not treat the expected warnings as failures.
  auto const prevLogFn = mochi::GetLogCallback();
  mochi::SetLogCallback(nullptr);
  MOCHI_DEFER(mochi::SetLogCallback(prevLogFn));

  std::vector<uint8_t> const garbage(64, 0xAB);
  mochi::TempFileCleanup const file = WriteTempGlb(garbage, "glb_sections_garbage");
  EXPECT_TRUE(ReadGlbFromFile(file.Path().string().c_str()).empty());
  EXPECT_TRUE(ReadGlbFromFile(nullptr).empty());
  EXPECT_TRUE(ReadGlbFromFile("/nonexistent/path/missing.glb").empty());
}

TEST(GlbReaderTest, NoIndexPrimitiveGeneratesSequentialIndices) {
  std::vector<float> const positions = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  std::vector<uint8_t> const glb = BuildNoIndexGlb(positions);

  mochi::TempFileCleanup const file = WriteTempGlb(glb, "glb_sections_noindex");
  std::vector<MeshSection> const read = ReadGlbFromFile(file.Path().string().c_str());

  ASSERT_EQ(read.size(), 1u);
  EXPECT_EQ(read[0].positions, positions);
  EXPECT_FALSE(read[0].hasNormals);
  std::vector<int> const expectedIndices = {0, 1, 2};
  EXPECT_EQ(read[0].indices, expectedIndices);
}

TEST(GlbReaderTest, DracoCompressedCubeYieldsRealGeometry) {
  std::string const path = DracoCubeGlbPath();
  ASSERT_TRUE(std::filesystem::exists(path)) << "Cube.glb not found at: " << path;

  std::vector<MeshSection> const read = ReadGlbFromFile(path.c_str());

  ASSERT_EQ(read.size(), 1u);
  MeshSection const& cube = read[0];
  // 24 vertices (4 per face, unshared) and 12 triangles.
  EXPECT_EQ(cube.positions.size(), 24u * 3u);
  EXPECT_EQ(cube.indices.size(), 36u);
  EXPECT_TRUE(cube.hasNormals);
  EXPECT_EQ(cube.normals.size(), cube.positions.size());

  // The regression guard: before Draco decoding, unpacking produced a
  // correctly-sized but entirely zero-filled mesh.
  float minPos = std::numeric_limits<float>::max();
  float maxPos = std::numeric_limits<float>::lowest();
  for (float const p : cube.positions) {
    minPos = std::min(minPos, p);
    maxPos = std::max(maxPos, p);
  }
  EXPECT_NEAR(minPos, -0.5f, 1e-4f);
  EXPECT_NEAR(maxPos, 0.5f, 1e-4f);

  for (int const index : cube.indices) {
    EXPECT_GE(index, 0);
    EXPECT_LT(index, 24);
  }
}

TEST(GlbReaderTest, CorruptDracoStreamReturnsEmpty) {
  std::vector<uint8_t> glb = ReadFileBytes(DracoCubeGlbPath());
  ASSERT_FALSE(glb.empty());
  // Overwrite the start of the Draco bitstream so decoding fails rather than
  // falling back to zeroed geometry.
  size_t const binOffset = BinChunkOffset(glb);
  ASSERT_LT(binOffset + 16, glb.size());
  std::fill(
      glb.begin() + static_cast<ptrdiff_t>(binOffset),
      glb.begin() + static_cast<ptrdiff_t>(binOffset) + 16,
      uint8_t{0xAB});

  auto const prevLogFn = mochi::GetLogCallback();
  mochi::SetLogCallback(nullptr);
  MOCHI_DEFER(mochi::SetLogCallback(prevLogFn));

  mochi::TempFileCleanup const file = WriteTempGlb(glb, "glb_sections_draco_corrupt");
  EXPECT_TRUE(ReadGlbFromFile(file.Path().string().c_str()).empty());
}

TEST(GlbReaderTest, MeshoptCompressedPrimitiveDecodes) {
  std::vector<float> const positions = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f};
  std::vector<uint32_t> const indices = {0, 2, 1};
  std::vector<uint8_t> const glb = BuildMeshoptGlb(positions, indices);

  mochi::TempFileCleanup const file = WriteTempGlb(glb, "glb_sections_meshopt");
  std::vector<MeshSection> const read = ReadGlbFromFile(file.Path().string().c_str());

  ASSERT_EQ(read.size(), 1u);
  EXPECT_EQ(read[0].positions, positions);
  std::vector<int> const expectedIndices = {0, 2, 1};
  EXPECT_EQ(read[0].indices, expectedIndices);
}

TEST(GlbReaderTest, QuantizedPositionsAreDequantizedByNodeScale) {
  // KHR_mesh_quantization needs no decode pass: cgltf widens the integer
  // components and the node transform supplies the scale.
  std::vector<int16_t> const quantized = {0, 0, 0, 1000, 0, 0, 0, 2000, 0};
  std::vector<uint8_t> const glb = BuildQuantizedGlb(quantized, 0.001f);

  mochi::TempFileCleanup const file = WriteTempGlb(glb, "glb_sections_quantized");
  std::vector<MeshSection> const read = ReadGlbFromFile(file.Path().string().c_str());

  ASSERT_EQ(read.size(), 1u);
  std::vector<float> const expected = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f};
  ASSERT_EQ(read[0].positions.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(read[0].positions[i], expected[i], 1e-5f) << "component " << i;
  }
}
