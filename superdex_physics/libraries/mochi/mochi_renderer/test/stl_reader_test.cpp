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

// Unit tests for the public ReadStlFromFile MeshSection API (the inverse-free
// STL importer in utils.cpp). Covers ASCII and binary parsing, vertex
// deduplication, the single-section / no-normals / default-material contract,
// and the multi-solid and missing-file rejection paths. Fixtures are written to
// uniquely-named temp files via the canonical mochi temp-file utility.

#include <gtest/gtest.h>

#include <mochi_renderer/utils.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/log.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace mochi_renderer;

namespace {

// A single triangle (3 distinct vertices), Y-up, one solid.
char const* const kAsciiTriangleStl = R"STL(solid tri
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid tri
)STL";

// Two triangles forming a quad in the z=0 plane. Six corners reference only four
// distinct positions, so the reader's deduplication should collapse them.
char const* const kAsciiQuadStl = R"STL(solid quad
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 1 1 0
  endloop
endfacet
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 1 0
    vertex 0 1 0
  endloop
endfacet
endsolid quad
)STL";

// Two solids in one file; the importer rejects multi-solid STL.
char const* const kAsciiTwoSolidStl = R"STL(solid a
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid a
solid b
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 0 1
  endloop
endfacet
endsolid b
)STL";

// Builds a binary STL byte stream with a single triangle. Layout: 80-byte
// header, uint32 triangle count, then per triangle [normal(3f), v0(3f), v1(3f),
// v2(3f), uint16 attribute byte count]. The zeroed header ensures the reader's
// ASCII sniffing (which looks for "solid"/"facet") selects the binary path.
std::string MakeBinaryTriangleStl() {
  std::string data(80, '\0');
  auto appendBytes = [&](void const* p, size_t n) { data.append(static_cast<char const*>(p), n); };
  auto appendU32 = [&](uint32_t v) { appendBytes(&v, sizeof(v)); };
  auto appendF32 = [&](float v) { appendBytes(&v, sizeof(v)); };

  appendU32(1); // triangle count
  // Face normal.
  appendF32(0.0f);
  appendF32(0.0f);
  appendF32(1.0f);
  // Three corners.
  float const corners[9] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  for (float c : corners) {
    appendF32(c);
  }
  uint16_t const attributeByteCount = 0;
  appendBytes(&attributeByteCount, sizeof(attributeByteCount));
  return data;
}

// Writes raw STL bytes to a uniquely-named temp file (auto-deleted at
// end-of-scope by the returned RAII handle).
mochi::TempFileCleanup WriteTempStl(std::string const& bytes, char const* label) {
  mochi::TempFileCleanup file = mochi::CreateTempFile(label, ".stl", mochi::test::ExpectOK{});
  std::ofstream out(file.Path(), std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return file;
}

// Reconstructs each triangle corner's coordinates by following the section's
// indices, then sorts them so comparisons are independent of the reader's
// deduplication ordering.
std::vector<std::array<float, 3>> SortedCornerCoords(MeshSection const& section) {
  std::vector<std::array<float, 3>> corners;
  corners.reserve(section.indices.size());
  for (int const index : section.indices) {
    size_t const base = static_cast<size_t>(index) * 3;
    corners.push_back(
        {section.positions[base], section.positions[base + 1], section.positions[base + 2]});
  }
  std::ranges::sort(corners);
  return corners;
}

} // namespace

TEST(StlReaderTest, AsciiTriangleYieldsSingleDefaultSection) {
  mochi::TempFileCleanup const file = WriteTempStl(kAsciiTriangleStl, "stl_reader_triangle");
  std::vector<MeshSection> const sections = ReadStlFromFile(file.Path().string().c_str());
  ASSERT_EQ(sections.size(), 1u);

  MeshSection const& s = sections[0];
  // STL carries only per-face normals, so the reader leaves normals to the caller.
  EXPECT_FALSE(s.hasNormals);
  EXPECT_TRUE(s.normals.empty());
  EXPECT_EQ(s.positions.size(), 9u);
  EXPECT_EQ(s.indices.size(), 3u);

  // STL has no material; MeshSection defaults apply.
  EXPECT_NEAR(s.baseColor[0], 0.5f, 1e-6f);
  EXPECT_NEAR(s.baseColor[1], 0.5f, 1e-6f);
  EXPECT_NEAR(s.baseColor[2], 0.5f, 1e-6f);
  EXPECT_NEAR(s.baseColor[3], 1.0f, 1e-6f);
  EXPECT_NEAR(s.metallic, 0.0f, 1e-6f);
  EXPECT_NEAR(s.roughness, 0.5f, 1e-6f);

  // Indices and positions together reproduce the input triangle's corners.
  std::vector<std::array<float, 3>> const expected = {
      {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
  EXPECT_EQ(SortedCornerCoords(s), expected);
}

TEST(StlReaderTest, SharedVerticesAreDeduplicated) {
  mochi::TempFileCleanup const file = WriteTempStl(kAsciiQuadStl, "stl_reader_quad");
  std::vector<MeshSection> const sections = ReadStlFromFile(file.Path().string().c_str());
  ASSERT_EQ(sections.size(), 1u);

  MeshSection const& s = sections[0];
  // Six triangle corners collapse to four unique vertices; two triangles remain.
  EXPECT_EQ(s.positions.size(), 12u);
  EXPECT_EQ(s.indices.size(), 6u);

  // Every index must reference a vertex that actually exists.
  size_t const vertexCount = s.positions.size() / 3;
  for (int const index : s.indices) {
    EXPECT_GE(index, 0);
    EXPECT_LT(static_cast<size_t>(index), vertexCount);
  }
}

TEST(StlReaderTest, BinaryTriangleParsed) {
  mochi::TempFileCleanup const file = WriteTempStl(MakeBinaryTriangleStl(), "stl_reader_binary");
  std::vector<MeshSection> const sections = ReadStlFromFile(file.Path().string().c_str());
  ASSERT_EQ(sections.size(), 1u);

  MeshSection const& s = sections[0];
  EXPECT_FALSE(s.hasNormals);
  EXPECT_EQ(s.positions.size(), 9u);
  EXPECT_EQ(s.indices.size(), 3u);

  std::vector<std::array<float, 3>> const expected = {
      {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
  EXPECT_EQ(SortedCornerCoords(s), expected);
}

TEST(StlReaderTest, MultiSolidReturnsEmpty) {
  // The reader warns when rejecting a multi-solid file; silence it so the test
  // harness does not treat the expected warning as a failure.
  auto const prevLogFn = mochi::GetLogCallback();
  mochi::SetLogCallback(nullptr);
  MOCHI_DEFER(mochi::SetLogCallback(prevLogFn));

  mochi::TempFileCleanup const file = WriteTempStl(kAsciiTwoSolidStl, "stl_reader_two_solid");
  EXPECT_TRUE(ReadStlFromFile(file.Path().string().c_str()).empty());
}

TEST(StlReaderTest, MissingFileReturnsEmpty) {
  // The reader warns on a failed open; silence it so the test harness does not
  // treat the expected warning as a failure.
  auto const prevLogFn = mochi::GetLogCallback();
  mochi::SetLogCallback(nullptr);
  MOCHI_DEFER(mochi::SetLogCallback(prevLogFn));

  EXPECT_TRUE(ReadStlFromFile("/nonexistent/path/missing.stl").empty());
}

TEST(StlReaderTest, NullPathReturnsEmpty) {
  EXPECT_TRUE(ReadStlFromFile(nullptr).empty());
}
