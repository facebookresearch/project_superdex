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

#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/file_utils.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

// Order-robust golden regression tests for surface-mesh loading (OBJ/PLY/STL/OFF), covering both
// the file (LoadFromFile) and bytes (LoadFromBytes) entry points and both ASCII and binary
// encodings. They assert implementation-independent geometry (unique-vertex set, triangle set by
// coordinate, bounding box, centroid) rather than raw vertex/face ordering.

using namespace mochi;
using namespace mochi::test;

namespace {

using Vec3 = std::array<double, 3>;
using Tri = std::array<Vec3, 3>;

// Canonical cube: 8 corners at {0,2}^3, 12 triangles (0-based indices, consistent outward winding).
std::array<Vec3, 8> const kCubeVertices = {{
    {0, 0, 0},
    {2, 0, 0},
    {2, 2, 0},
    {0, 2, 0},
    {0, 0, 2},
    {2, 0, 2},
    {2, 2, 2},
    {0, 2, 2},
}};

std::array<std::array<int, 3>, 12> const kCubeTriangles = {{
    {0, 2, 1},
    {0, 3, 2},
    {4, 5, 6},
    {4, 6, 7},
    {0, 1, 5},
    {0, 5, 4},
    {2, 7, 6},
    {2, 3, 7},
    {0, 4, 7},
    {0, 7, 3},
    {1, 6, 5},
    {1, 2, 6},
}};

void WriteBytes(std::ofstream& file, void const* data, size_t size) {
  file.write(reinterpret_cast<char const*>(data), static_cast<std::streamsize>(size));
}

template <class T>
void WritePod(std::ofstream& file, T value) {
  WriteBytes(file, &value, sizeof(value));
}

// ---------------------------------------------------------------------------
// Fixture: writes a 2x2x2 cube in each format/encoding to a temp dir.
// ---------------------------------------------------------------------------

class ModelUtilsMeshIoTest : public testing::Test {
 protected:
  void SetUp() override {
    _tempDir.emplace(CreateTempDirectory("mochi_meshio_test", ExpectOK{}));
  }

  std::filesystem::path const& TempDir() const {
    return _tempDir->Path();
  }

  // OBJ uses 1-based indices and shared vertices.
  std::string WriteObj() {
    auto path = (TempDir() / "cube.obj").string();
    std::ofstream file(path);
    for (auto const& v : kCubeVertices) {
      file << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
    }
    for (auto const& f : kCubeTriangles) {
      file << "f " << f[0] + 1 << " " << f[1] + 1 << " " << f[2] + 1 << "\n";
    }
    return path;
  }

  // A single quad face (four shared corners). With config.triangulate = true the loader must fan it
  // into two triangles over the same four corners.
  std::string WriteObjQuad() {
    auto path = (TempDir() / "quad.obj").string();
    std::ofstream file(path);
    file << "v 0 0 0\nv 2 0 0\nv 2 2 0\nv 0 2 0\n";
    file << "f 1 2 3 4\n";
    return path;
  }

  // OFF uses 0-based indices and shared vertices.
  std::string WriteOff() {
    auto path = (TempDir() / "cube.off").string();
    std::ofstream file(path);
    file << "OFF\n8 12 0\n";
    for (auto const& v : kCubeVertices) {
      file << v[0] << " " << v[1] << " " << v[2] << "\n";
    }
    for (auto const& f : kCubeTriangles) {
      file << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }
    return path;
  }

  // ASCII PLY with shared vertices (0-based indices).
  std::string WritePlyAscii() {
    auto path = (TempDir() / "cube_ascii.ply").string();
    std::ofstream file(path);
    file << "ply\nformat ascii 1.0\n";
    file << "element vertex 8\n";
    file << "property float x\nproperty float y\nproperty float z\n";
    file << "element face 12\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";
    for (auto const& v : kCubeVertices) {
      file << v[0] << " " << v[1] << " " << v[2] << "\n";
    }
    for (auto const& f : kCubeTriangles) {
      file << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }
    return path;
  }

  // Binary little-endian PLY with shared vertices.
  std::string WritePlyBinary() {
    auto path = (TempDir() / "cube_binary.ply").string();
    std::ofstream file(path, std::ios::binary);
    file << "ply\nformat binary_little_endian 1.0\n";
    file << "element vertex 8\n";
    file << "property float x\nproperty float y\nproperty float z\n";
    file << "element face 12\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";
    for (auto const& v : kCubeVertices) {
      WritePod(file, static_cast<float>(v[0]));
      WritePod(file, static_cast<float>(v[1]));
      WritePod(file, static_cast<float>(v[2]));
    }
    for (auto const& f : kCubeTriangles) {
      WritePod(file, static_cast<uint8_t>(3));
      WritePod(file, static_cast<int32_t>(f[0]));
      WritePod(file, static_cast<int32_t>(f[1]));
      WritePod(file, static_cast<int32_t>(f[2]));
    }
    return path;
  }

  // ASCII PLY where every triangle has its own copy of each vertex (36 vertices); welding on load
  // must reduce these to the 8 unique cube corners.
  std::string WritePlyDuplicated() {
    auto path = (TempDir() / "cube_duped.ply").string();
    std::ofstream file(path);
    file << "ply\nformat ascii 1.0\n";
    file << "element vertex 36\n";
    file << "property float x\nproperty float y\nproperty float z\n";
    file << "element face 12\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";
    for (auto const& f : kCubeTriangles) {
      for (int vi : f) {
        auto const& v = kCubeVertices[vi];
        file << v[0] << " " << v[1] << " " << v[2] << "\n";
      }
    }
    for (int i = 0; i < 12; ++i) {
      file << "3 " << 3 * i << " " << 3 * i + 1 << " " << 3 * i + 2 << "\n";
    }
    return path;
  }

  // ASCII PLY with a single quad face (four shared corners); fan-triangulation must split it into
  // two triangles over the same four corners.
  std::string WritePlyQuad() {
    auto path = (TempDir() / "quad.ply").string();
    std::ofstream file(path);
    file << "ply\nformat ascii 1.0\n";
    file << "element vertex 4\n";
    file << "property float x\nproperty float y\nproperty float z\n";
    file << "element face 1\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";
    file << "0 0 0\n2 0 0\n2 2 0\n0 2 0\n";
    file << "4 0 1 2 3\n";
    return path;
  }

  // ASCII STL stores each triangle's vertices independently (36 vertices; merged to 8 on load).
  std::string WriteStlAscii() {
    auto path = (TempDir() / "cube_ascii.stl").string();
    std::ofstream file(path);
    file << "solid cube\n";
    for (auto const& f : kCubeTriangles) {
      file << "  facet normal 0 0 0\n    outer loop\n";
      for (int vi : f) {
        auto const& v = kCubeVertices[vi];
        file << "      vertex " << v[0] << " " << v[1] << " " << v[2] << "\n";
      }
      file << "    endloop\n  endfacet\n";
    }
    file << "endsolid cube\n";
    return path;
  }

  // Binary STL: 80-byte header + uint32 triangle count + 50 bytes per triangle.
  std::string WriteStlBinary() {
    auto path = (TempDir() / "cube_binary.stl").string();
    std::ofstream file(path, std::ios::binary);
    char header[80] = {};
    // Header must not begin with "solid" so it is detected as binary, not ASCII.
    std::memcpy(header, "binary STL cube", 15);
    WriteBytes(file, header, sizeof(header));
    WritePod(file, static_cast<uint32_t>(kCubeTriangles.size()));
    for (auto const& f : kCubeTriangles) {
      WritePod(file, 0.0f); // normal x
      WritePod(file, 0.0f); // normal y
      WritePod(file, 0.0f); // normal z
      for (int vi : f) {
        auto const& v = kCubeVertices[vi];
        WritePod(file, static_cast<float>(v[0]));
        WritePod(file, static_cast<float>(v[1]));
        WritePod(file, static_cast<float>(v[2]));
      }
      WritePod(file, static_cast<uint16_t>(0)); // attribute byte count
    }
    return path;
  }

  std::optional<TempDirCleanup> _tempDir;
};

// Expected unique-vertex set, sorted (independent of any loader's indexing).
DynamicArray<Vec3> ExpectedUniqueVertices() {
  DynamicArray<Vec3> verts(kCubeVertices.begin(), kCubeVertices.end());
  std::sort(verts.begin(), verts.end());
  return verts;
}

// Expected triangle set: each triangle as its 3 vertex coordinates, sorted within the triangle (so
// winding/rotation does not matter) and the list of triangles sorted (so face order does not
// matter).
DynamicArray<Tri> ExpectedTriangles() {
  DynamicArray<Tri> tris;
  for (auto const& f : kCubeTriangles) {
    Tri tri = {kCubeVertices[f[0]], kCubeVertices[f[1]], kCubeVertices[f[2]]};
    std::sort(tri.begin(), tri.end());
    tris.push_back(tri);
  }
  std::sort(tris.begin(), tris.end());
  return tris;
}

// Asserts that the loaded mesh is geometrically the canonical cube, robust to vertex/face
// reordering.
void ExpectCubeGeometry(ModelData const& data) {
  ASSERT_TRUE(data.mesh.has_value());
  MeshData const& mesh = *data.mesh;
  ASSERT_EQ(mesh.nodesPerElement, 3);
  ASSERT_EQ(isize(mesh.coordinates) % 3, 0);
  ASSERT_EQ(isize(mesh.connectivity) % 3, 0);
  EXPECT_EQ(mesh.GetNumElements(), 12);

  int const numNodes = mesh.GetNumNodes();
  DynamicArray<Vec3> nodes(numNodes);
  for (int i = 0; i < numNodes; ++i) {
    nodes[i] = {
        static_cast<double>(mesh.coordinates[3 * i + 0]),
        static_cast<double>(mesh.coordinates[3 * i + 1]),
        static_cast<double>(mesh.coordinates[3 * i + 2])};
  }

  // Unique vertices (welded to 8 for the cube).
  DynamicArray<Vec3> unique = nodes;
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  EXPECT_EQ(unique, ExpectedUniqueVertices());

  // Triangles by coordinate.
  DynamicArray<Tri> tris;
  for (int t = 0; t < mesh.GetNumElements(); ++t) {
    // Connectivity comes straight from the loader; bounds-check before indexing so a bad index
    // surfaces as a clean failure instead of an out-of-bounds read.
    for (int k = 0; k < 3; ++k) {
      int const index = mesh.connectivity[3 * t + k];
      ASSERT_GE(index, 0);
      ASSERT_LT(index, numNodes);
    }
    Tri tri = {
        nodes[mesh.connectivity[3 * t + 0]],
        nodes[mesh.connectivity[3 * t + 1]],
        nodes[mesh.connectivity[3 * t + 2]]};
    std::sort(tri.begin(), tri.end());
    tris.push_back(tri);
  }
  std::sort(tris.begin(), tris.end());
  EXPECT_EQ(tris, ExpectedTriangles());

  // Bounding box + centroid of the unique vertices.
  ASSERT_FALSE(unique.empty());
  Vec3 bmin = unique.front();
  Vec3 bmax = unique.front();
  Vec3 centroid = {0, 0, 0};
  for (auto const& v : unique) {
    for (int k = 0; k < 3; ++k) {
      bmin[k] = std::min(bmin[k], v[k]);
      bmax[k] = std::max(bmax[k], v[k]);
      centroid[k] += v[k];
    }
  }
  for (int k = 0; k < 3; ++k) {
    centroid[k] /= static_cast<double>(unique.size());
  }
  EXPECT_EQ(bmin, (Vec3{0, 0, 0}));
  EXPECT_EQ(bmax, (Vec3{2, 2, 2}));
  EXPECT_EQ(centroid, (Vec3{1, 1, 1}));
}

ModelData LoadBytes(std::string const& path, MeshFileType format) {
  // Read as raw bytes (ReadFileString rejects the null bytes in binary PLY/STL).
  DynamicArray<char> const bytes = ReadFileBytes(path, ExpectOK{});
  ExpectOK expectOK;
  return model::LoadFromBytes(MakeConstSpan(bytes), format, expectOK);
}

} // namespace

// ---------------------------------------------------------------------------
// File-based loading.
// ---------------------------------------------------------------------------

TEST_F(ModelUtilsMeshIoTest, LoadObjFile) {
  ExpectCubeGeometry(model::LoadFromFile(WriteObj(), ExpectOK{}));
}

TEST_F(ModelUtilsMeshIoTest, LoadOffFile) {
  ExpectCubeGeometry(model::LoadFromFile(WriteOff(), ExpectOK{}));
}

TEST_F(ModelUtilsMeshIoTest, LoadPlyAsciiFile) {
  ExpectCubeGeometry(model::LoadFromFile(WritePlyAscii(), ExpectOK{}));
}

TEST_F(ModelUtilsMeshIoTest, LoadPlyBinaryFile) {
  ExpectCubeGeometry(model::LoadFromFile(WritePlyBinary(), ExpectOK{}));
}

TEST_F(ModelUtilsMeshIoTest, LoadStlAsciiFile) {
  ExpectCubeGeometry(model::LoadFromFile(WriteStlAscii(), ExpectOK{}));
}

TEST_F(ModelUtilsMeshIoTest, LoadStlBinaryFile) {
  ExpectCubeGeometry(model::LoadFromFile(WriteStlBinary(), ExpectOK{}));
}

TEST_F(ModelUtilsMeshIoTest, LoadNonexistentFile) {
  ExpectNotOK expectNotOK;
  ModelData const data = model::LoadFromFile((TempDir() / "nonexistent.obj").string(), expectNotOK);
  EXPECT_FALSE(data.mesh.has_value());
}

// ---------------------------------------------------------------------------
// OBJ permissive-loader behavior (tiny_obj_loader path; no CGAL needed).
// ---------------------------------------------------------------------------

// A non-triangle (quad) face must be triangulated on load: one quad becomes two triangles.
TEST_F(ModelUtilsMeshIoTest, LoadObjQuadIsTriangulated) {
  ModelData const data = model::LoadFromFile(WriteObjQuad(), ExpectOK{});
  ASSERT_TRUE(data.mesh.has_value());
  MeshData const& mesh = *data.mesh;
  EXPECT_EQ(mesh.nodesPerElement, 3);
  EXPECT_EQ(mesh.GetNumElements(), 2);
  EXPECT_EQ(mesh.GetNumNodes(), 4);
}

// A face that references a vertex index past the vertex list must be rejected, not silently loaded.
TEST_F(ModelUtilsMeshIoTest, LoadObjOutOfRangeIndexFails) {
  auto const path = (TempDir() / "bad_index.obj").string();
  {
    std::ofstream file(path);
    file << "v 0 0 0\nv 2 0 0\nv 0 2 0\nf 1 2 5\n"; // only 3 vertices; index 5 is out of range
  }
  ExpectNotOK expectNotOK;
  // ExpectNotOK asserts the load reports an error; the returned data is irrelevant here.
  (void)model::LoadFromFile(path, expectNotOK);
}

// An OBJ with no geometry must be rejected.
TEST_F(ModelUtilsMeshIoTest, LoadObjNoVerticesFails) {
  auto const path = (TempDir() / "empty.obj").string();
  {
    std::ofstream file(path);
    file << "# empty OBJ, no vertices or faces\n";
  }
  ExpectNotOK expectNotOK;
  // ExpectNotOK asserts the load reports an error; the returned data is irrelevant here.
  (void)model::LoadFromFile(path, expectNotOK);
}

// ---------------------------------------------------------------------------
// STL permissive-loader behavior (hand-rolled reader; no CGAL needed).
// ---------------------------------------------------------------------------

// A zero-area (collapsed) triangle must be dropped when welding: only the real triangle survives.
TEST_F(ModelUtilsMeshIoTest, LoadStlDropsDegenerateTriangle) {
  auto const path = (TempDir() / "degenerate.stl").string();
  {
    std::ofstream file(path);
    file << "solid d\n";
    // One real triangle.
    file << "  facet normal 0 0 0\n    outer loop\n";
    file << "      vertex 0 0 0\n      vertex 2 0 0\n      vertex 0 2 0\n";
    file << "    endloop\n  endfacet\n";
    // One degenerate triangle: three coincident vertices (all an existing corner) collapse to a
    // single welded index, so the triangle is dropped and adds no new vertex.
    file << "  facet normal 0 0 0\n    outer loop\n";
    file << "      vertex 0 0 0\n      vertex 0 0 0\n      vertex 0 0 0\n";
    file << "    endloop\n  endfacet\n";
    file << "endsolid d\n";
  }
  ModelData const data = model::LoadFromFile(path, ExpectOK{});
  ASSERT_TRUE(data.mesh.has_value());
  MeshData const& mesh = *data.mesh;
  EXPECT_EQ(mesh.GetNumElements(), 1);
  EXPECT_EQ(mesh.GetNumNodes(), 3);
}

// A solid name that contains the word "vertex" must not be misparsed as a vertex record.
TEST_F(ModelUtilsMeshIoTest, LoadStlSolidNameContainingVertex) {
  auto const path = (TempDir() / "named.stl").string();
  {
    std::ofstream file(path);
    file << "solid exported vertex mesh\n";
    file << "  facet normal 0 0 0\n    outer loop\n";
    file << "      vertex 0 0 0\n      vertex 2 0 0\n      vertex 0 2 0\n";
    file << "    endloop\n  endfacet\n";
    file << "endsolid exported vertex mesh\n";
  }
  ModelData const data = model::LoadFromFile(path, ExpectOK{});
  ASSERT_TRUE(data.mesh.has_value());
  EXPECT_EQ(data.mesh->GetNumElements(), 1);
}

// ---------------------------------------------------------------------------
// Bytes-based loading (explicit MeshFileType).
// ---------------------------------------------------------------------------

TEST_F(ModelUtilsMeshIoTest, LoadObjBytes) {
  ExpectCubeGeometry(LoadBytes(WriteObj(), MeshFileType::OBJ));
}

TEST_F(ModelUtilsMeshIoTest, LoadOffBytes) {
  ExpectCubeGeometry(LoadBytes(WriteOff(), MeshFileType::OFF));
}

TEST_F(ModelUtilsMeshIoTest, LoadPlyAsciiBytes) {
  ExpectCubeGeometry(LoadBytes(WritePlyAscii(), MeshFileType::PLY));
}

TEST_F(ModelUtilsMeshIoTest, LoadPlyBinaryBytes) {
  ExpectCubeGeometry(LoadBytes(WritePlyBinary(), MeshFileType::PLY));
}

TEST_F(ModelUtilsMeshIoTest, LoadStlAsciiBytes) {
  ExpectCubeGeometry(LoadBytes(WriteStlAscii(), MeshFileType::STL));
}

TEST_F(ModelUtilsMeshIoTest, LoadStlBinaryBytes) {
  ExpectCubeGeometry(LoadBytes(WriteStlBinary(), MeshFileType::STL));
}

// Duplicate-vertex welding: a PLY soup of 36 duplicated vertices must weld to the 8-vertex cube.
TEST_F(ModelUtilsMeshIoTest, LoadPlyDuplicatedWeldsFromFile) {
  ExpectCubeGeometry(model::LoadFromFile(WritePlyDuplicated(), ExpectOK{}));
}

TEST_F(ModelUtilsMeshIoTest, LoadPlyDuplicatedWeldsFromBytes) {
  ExpectCubeGeometry(LoadBytes(WritePlyDuplicated(), MeshFileType::PLY));
}

// ---------------------------------------------------------------------------
// PLY permissive-loader behavior (happly path; no CGAL needed).
// ---------------------------------------------------------------------------

// A quad PLY face must be fan-triangulated into two triangles over the four corners.
TEST_F(ModelUtilsMeshIoTest, LoadPlyQuadIsTriangulated) {
  ModelData const data = model::LoadFromFile(WritePlyQuad(), ExpectOK{});
  ASSERT_TRUE(data.mesh.has_value());
  MeshData const& mesh = *data.mesh;
  EXPECT_EQ(mesh.nodesPerElement, 3);
  EXPECT_EQ(mesh.GetNumElements(), 2);
  EXPECT_EQ(mesh.GetNumNodes(), 4);
}

// A face index past the vertex list must be rejected, not used to index out of bounds.
TEST_F(ModelUtilsMeshIoTest, LoadPlyOutOfRangeIndexFails) {
  auto const path = (TempDir() / "bad_index.ply").string();
  {
    std::ofstream file(path);
    file << "ply\nformat ascii 1.0\n";
    file << "element vertex 3\n";
    file << "property float x\nproperty float y\nproperty float z\n";
    file << "element face 1\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";
    file << "0 0 0\n2 0 0\n0 2 0\n";
    file << "3 0 1 5\n"; // only 3 vertices; index 5 is out of range
  }
  ExpectNotOK expectNotOK;
  // ExpectNotOK asserts the load reports an error; the returned data is irrelevant here.
  (void)model::LoadFromFile(path, expectNotOK);
}
