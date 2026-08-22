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

#include "model_utils_meshio.h"

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/file_utils.h>

#include <happly/happly.h>
#include <tiny_obj_loader.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <ios>
#include <istream>
#include <map>
#include <sstream>
#include <streambuf>
#include <string>
#include <string_view>
#include <vector>

using namespace mochi;

namespace {

// Merges exactly-coincident vertices in a triangle soup, remaps the connectivity, and drops
// degenerate triangles (those left with a repeated vertex). Operates in place. Used by loaders
// whose source format stores each triangle's vertices independently.
void WeldVertices(DynamicArray<real>& coordinates, DynamicArray<int>& connectivity) {
  int const numOldVertices = isize(coordinates) / 3;
  std::map<std::array<real, 3>, int> uniqueIndexByPosition;
  DynamicArray<int> remap;
  remap.resize_noinit(numOldVertices);
  DynamicArray<real> weldedCoordinates;
  for (int v = 0; v < numOldVertices; ++v) {
    std::array<real, 3> const position = {
        coordinates[3 * v + 0], coordinates[3 * v + 1], coordinates[3 * v + 2]};
    auto const [it, inserted] =
        uniqueIndexByPosition.try_emplace(position, isize(weldedCoordinates) / 3);
    if (inserted) {
      weldedCoordinates.push_back(position[0]);
      weldedCoordinates.push_back(position[1]);
      weldedCoordinates.push_back(position[2]);
    }
    remap[v] = it->second;
  }

  DynamicArray<int> weldedConnectivity;
  for (int t = 0; t + 2 < isize(connectivity); t += 3) {
    int const a = remap[connectivity[t + 0]];
    int const b = remap[connectivity[t + 1]];
    int const c = remap[connectivity[t + 2]];
    if (a == b || b == c || a == c) {
      continue; // Drop degenerate triangle.
    }
    weldedConnectivity.push_back(a);
    weldedConnectivity.push_back(b);
    weldedConnectivity.push_back(c);
  }

  coordinates = std::move(weldedCoordinates);
  connectivity = std::move(weldedConnectivity);
}

} // namespace

// ---------------------------------------------------------------------------
// OBJ loading via tiny_obj_loader.
// ---------------------------------------------------------------------------

namespace {

tinyobj::ObjReaderConfig MakeObjReaderConfig() {
  tinyobj::ObjReaderConfig config;
  config.triangulate = true; // Triangulate any non-triangle faces on load.
  config.vertex_color = false; // We do not consume vertex colors.
  return config;
}

// Flattens a parsed OBJ (already triangulated) into MeshData. OBJ stores shared vertices, so no
// vertex welding is needed. attrib.vertices is a flat xyz array; each face index references a
// vertex.
void BuildModelFromObjReader(ModelData& outData, tinyobj::ObjReader const& reader, Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const& attrib = reader.GetAttrib();
  MOCHI_ERROR_IF(attrib.vertices.empty(), error, "OBJ mesh contains no vertices.");
  MOCHI_ERROR_IF(
      isize(attrib.vertices) % 3 != 0, error, "OBJ vertex array size must be a multiple of 3.");
  MOCHI_ERROR_RETURN(error);

  int const numVertices = isize(attrib.vertices) / 3;

  outData = {};
  outData.mesh.emplace();
  outData.mesh->nodesPerElement = 3;

  auto& coordinates = outData.mesh->coordinates;
  coordinates.resize_noinit(isize(attrib.vertices));
  for (int i = 0; i < isize(attrib.vertices); ++i) {
    coordinates[i] = StaticCast<real>(attrib.vertices[i]);
  }

  auto& connectivity = outData.mesh->connectivity;
  size_t totalIndexCount = 0;
  for (auto const& shape : reader.GetShapes()) {
    totalIndexCount += shape.mesh.indices.size();
  }
  connectivity.reserve(totalIndexCount);
  for (auto const& shape : reader.GetShapes()) {
    for (auto const& index : shape.mesh.indices) {
      int const vertexIndex = StaticCast<int>(index.vertex_index);
      MOCHI_ERROR_IF(
          vertexIndex < 0 || vertexIndex >= numVertices,
          error,
          "OBJ face references an out-of-range vertex index.");
      connectivity.push_back(vertexIndex);
    }
  }
  MOCHI_ERROR_RETURN(error);

  MOCHI_ERROR_IF(connectivity.empty(), error, "OBJ mesh contains no faces.");
  MOCHI_ERROR_IF(
      isize(connectivity) % 3 != 0, error, "OBJ mesh faces did not triangulate to triangles.");
}

} // namespace

void mochi::model::LoadObjFromFile(ModelData& outData, std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error);
  tinyobj::ObjReader reader;
  bool const ok = reader.ParseFromFile(std::string(path), MakeObjReaderConfig());
  MOCHI_ERROR_IF(!ok || !reader.Valid(), error, "Failed to read OBJ file.");
  MOCHI_ERROR_RETURN(error);
  BuildModelFromObjReader(outData, reader, error);
}

void mochi::model::LoadObjFromBytes(ModelData& outData, Span<char const> data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  tinyobj::ObjReader reader;
  bool const ok = reader.ParseFromString(
      std::string(data.data(), data.size()), /*mtl_text*/ std::string(), MakeObjReaderConfig());
  MOCHI_ERROR_IF(!ok || !reader.Valid(), error, "Failed to parse OBJ data.");
  MOCHI_ERROR_RETURN(error);
  BuildModelFromObjReader(outData, reader, error);
}

// ---------------------------------------------------------------------------
// STL loading (binary + ASCII).
// ---------------------------------------------------------------------------

namespace {

// Binary STL layout: an 80-byte header, a uint32 triangle count, then one fixed-size record per
// triangle (a 3-float normal, three 3-float vertices, and a 2-byte attribute count).
constexpr size_t kStlHeaderSize = 80;
constexpr size_t kStlCountSize = sizeof(uint32_t);
constexpr size_t kStlMinSize = kStlHeaderSize + kStlCountSize;
constexpr size_t kStlNormalSize = 3 * sizeof(float);
constexpr size_t kStlVertexSize = 3 * sizeof(float);
constexpr size_t kStlAttributeSize = sizeof(uint16_t);
constexpr size_t kStlTriangleRecordSize = kStlNormalSize + 3 * kStlVertexSize + kStlAttributeSize;

// Reads a little-endian float from a byte pointer. All Mochi targets are little-endian, so this is
// a plain copy.
float ReadLittleEndianFloat(char const* p) {
  float value = 0.0f;
  std::memcpy(&value, p, sizeof(value));
  return value;
}

// Binary STL's size is fully determined by its triangle count, which makes it unambiguous to
// detect.
bool LooksLikeBinaryStl(Span<char const> data) {
  if (data.size() < kStlMinSize) {
    return false;
  }
  uint32_t count = 0;
  std::memcpy(&count, data.data() + kStlHeaderSize, sizeof(count));
  return data.size() == kStlMinSize + static_cast<size_t>(count) * kStlTriangleRecordSize;
}

// ASCII STL starts with the "solid" keyword followed by whitespace or end-of-data.
bool LooksLikeAsciiStl(Span<char const> data) {
  constexpr std::string_view kKeyword = "solid";
  size_t i = 0;
  while (i < data.size() &&
         (data[i] == ' ' || data[i] == '\t' || data[i] == '\r' || data[i] == '\n')) {
    ++i;
  }
  if (data.size() - i < kKeyword.size() ||
      std::memcmp(data.data() + i, kKeyword.data(), kKeyword.size()) != 0) {
    return false;
  }
  size_t const after = i + kKeyword.size();
  return after == data.size() || data[after] == ' ' || data[after] == '\t' || data[after] == '\n' ||
      data[after] == '\r';
}

// Binary STL stores each triangle's three vertices independently, so coincident vertices are welded
// on load.
void LoadStlBinary(ModelData& outData, Span<char const> data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(data.size() < kStlMinSize, error, "Binary STL data is too small.");
  MOCHI_ERROR_RETURN(error);

  uint32_t triangleCount = 0;
  std::memcpy(&triangleCount, data.data() + kStlHeaderSize, sizeof(triangleCount));
  MOCHI_ERROR_IF(
      data.size() != kStlMinSize + static_cast<size_t>(triangleCount) * kStlTriangleRecordSize,
      error,
      "Binary STL size does not match its triangle count.");
  MOCHI_ERROR_IF(triangleCount == 0, error, "STL mesh contains no triangles.");
  MOCHI_ERROR_RETURN(error);

  // The triangle count is known up front, so both arrays are sized once. Each triangle emits three
  // fresh vertices (9 reals) and three sequential indices, so connectivity is in range by
  // construction (later collapsed by WeldVertices); no index bounds check is needed.
  DynamicArray<real> coordinates;
  coordinates.resize_noinit(static_cast<size_t>(triangleCount) * 9);
  DynamicArray<int> connectivity;
  connectivity.resize_noinit(static_cast<size_t>(triangleCount) * 3);
  char const* const records = data.data() + kStlMinSize;
  for (uint32_t t = 0; t < triangleCount; ++t) {
    char const* const triangleVertices =
        records + static_cast<size_t>(t) * kStlTriangleRecordSize + kStlNormalSize;
    for (int v = 0; v < 3; ++v) {
      char const* const vertex = triangleVertices + static_cast<size_t>(v) * kStlVertexSize;
      int const vertexIndex = static_cast<int>(t) * 3 + v;
      for (int c = 0; c < 3; ++c) {
        coordinates[static_cast<size_t>(vertexIndex) * 3 + c] =
            StaticCast<real>(ReadLittleEndianFloat(vertex + c * sizeof(float)));
      }
      connectivity[vertexIndex] = vertexIndex;
    }
  }

  WeldVertices(coordinates, connectivity);
  outData = {};
  outData.mesh.emplace();
  outData.mesh->nodesPerElement = 3;
  outData.mesh->coordinates = std::move(coordinates);
  outData.mesh->connectivity = std::move(connectivity);
}

// ASCII STL is a sequence of `facet`/`outer loop`/`vertex x y z` records. We collect every `vertex`
// triple in order; consecutive triples form triangles, and coincident vertices are welded on load.
void LoadStlAscii(ModelData& outData, Span<char const> data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  // Parse line by line and act only on lines whose first token is `vertex`. Scanning the raw token
  // stream would misfire on the word "vertex" appearing inside a `solid <name>` / `endsolid <name>`
  // line (a valid file could be named e.g. `solid exported vertex mesh`).
  std::istringstream stream(std::string(data.data(), data.size()));
  std::string line;
  DynamicArray<real> coordinates;
  DynamicArray<int> connectivity;
  int vertexCount = 0;
  while (std::getline(stream, line)) {
    std::istringstream lineStream(line);
    std::string keyword;
    if (!(lineStream >> keyword) || keyword != "vertex") {
      continue;
    }
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    MOCHI_ERROR_IF(!(lineStream >> x >> y >> z), error, "Malformed vertex in ASCII STL.");
    MOCHI_ERROR_RETURN(error);
    coordinates.push_back(StaticCast<real>(x));
    coordinates.push_back(StaticCast<real>(y));
    coordinates.push_back(StaticCast<real>(z));
    connectivity.push_back(vertexCount);
    ++vertexCount;
  }

  MOCHI_ERROR_IF(vertexCount == 0, error, "STL mesh contains no vertices.");
  MOCHI_ERROR_IF(
      vertexCount % 3 != 0, error, "ASCII STL vertex count is not a multiple of 3 (triangles).");
  MOCHI_ERROR_RETURN(error);

  WeldVertices(coordinates, connectivity);
  outData = {};
  outData.mesh.emplace();
  outData.mesh->nodesPerElement = 3;
  outData.mesh->coordinates = std::move(coordinates);
  outData.mesh->connectivity = std::move(connectivity);
}

// Binary headers can also start with "solid", so check binary first.
void LoadStlImpl(ModelData& outData, Span<char const> data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (LooksLikeBinaryStl(data)) {
    LoadStlBinary(outData, data, error);
  } else if (LooksLikeAsciiStl(data)) {
    LoadStlAscii(outData, data, error);
  } else {
    MOCHI_ERROR_SET(error, "Data does not appear to be a valid STL file.");
  }
}

} // namespace

void mochi::model::LoadStlFromFile(ModelData& outData, std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error);
  DynamicArray<char> const bytes = ReadFileBytes(std::filesystem::path(std::string(path)), error);
  MOCHI_ERROR_RETURN(error);
  LoadStlImpl(outData, MakeConstSpan(bytes), error);
}

void mochi::model::LoadStlFromBytes(ModelData& outData, Span<char const> data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  LoadStlImpl(outData, data, error);
}

// ---------------------------------------------------------------------------
// PLY loading via happly.
// ---------------------------------------------------------------------------

namespace {

// Read-only std::streambuf that wraps a Span<char const> without copying, so happly can parse
// in-memory bytes through a std::istream.
class SpanStreamBuf : public std::streambuf {
 public:
  explicit SpanStreamBuf(Span<char const> data) {
    auto* begin = const_cast<char*>(data.data());
    setg(begin, begin, begin + data.size());
  }

 protected:
  pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which)
      override {
    if (!(which & std::ios_base::in)) {
      return {-1};
    }
    char* newPos = nullptr;
    if (dir == std::ios_base::beg) {
      newPos = eback() + off;
    } else if (dir == std::ios_base::cur) {
      newPos = gptr() + off;
    } else if (dir == std::ios_base::end) {
      newPos = egptr() + off;
    }
    if (newPos == nullptr || newPos < eback() || newPos > egptr()) {
      return {-1};
    }
    setg(eback(), newPos, egptr());
    return {newPos - eback()};
  }

  pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
    return seekoff(off_type(pos), std::ios_base::beg, which);
  }
};

// Flattens a parsed PLY into MeshData. Polygon faces are fan-triangulated and coincident vertices
// are welded on load.
void BuildModelFromPly(ModelData& outData, happly::PLYData& ply, Error& error) {
  MOCHI_ERROR_RETURN(error);
  std::vector<std::array<double, 3>> const vertices = ply.getVertexPositions();
  std::vector<std::vector<int>> const faces = ply.getFaceIndices<int>();
  MOCHI_ERROR_IF(vertices.empty(), error, "PLY mesh contains no vertices.");
  MOCHI_ERROR_IF(faces.empty(), error, "PLY mesh contains no faces.");
  MOCHI_ERROR_RETURN(error);

  int const numVertices = isize(vertices);
  DynamicArray<real> coordinates;
  // Size with size_t so a huge vertex count cannot overflow the int multiply; resize_noinit is safe
  // because real is trivially copyable and every element is written below.
  coordinates.resize_noinit(static_cast<size_t>(numVertices) * 3);
  for (int i = 0; i < numVertices; ++i) {
    // Compute the write offset in size_t (matching the STL loader) so a huge vertex count cannot
    // overflow the index arithmetic even though the allocation above is already size_t-safe.
    size_t const base = static_cast<size_t>(i) * 3;
    coordinates[base + 0] = StaticCast<real>(vertices[i][0]);
    coordinates[base + 1] = StaticCast<real>(vertices[i][1]);
    coordinates[base + 2] = StaticCast<real>(vertices[i][2]);
  }

  DynamicArray<int> connectivity;
  for (auto const& face : faces) {
    MOCHI_ERROR_IF(face.size() < 3, error, "PLY face has fewer than 3 vertices.");
    MOCHI_ERROR_RETURN(error);
    // Reject out-of-range indices before they are used to index the welded vertex table.
    for (int const index : face) {
      MOCHI_ERROR_IF(
          index < 0 || index >= numVertices,
          error,
          "PLY face references an out-of-range vertex index.");
    }
    MOCHI_ERROR_RETURN(error);
    // Fan-triangulate: (0, k, k+1) for k in [1, n-1).
    for (size_t k = 1; k + 1 < face.size(); ++k) {
      connectivity.push_back(face[0]);
      connectivity.push_back(face[k]);
      connectivity.push_back(face[k + 1]);
    }
  }

  WeldVertices(coordinates, connectivity);
  outData = {};
  outData.mesh.emplace();
  outData.mesh->nodesPerElement = 3;
  outData.mesh->coordinates = std::move(coordinates);
  outData.mesh->connectivity = std::move(connectivity);
}

} // namespace

void mochi::model::LoadPlyFromFile(ModelData& outData, std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error);
  try {
    std::string const pathString(path);
    happly::PLYData ply(pathString);
    BuildModelFromPly(outData, ply, error);
  } catch (std::exception const& e) {
    // happly reports parse failures (bad header, unknown element, ...) via exceptions. Error stores
    // a non-owning string literal, so log the detail here and set a stable message.
    MOCHI_LOG_WARNING("Failed to read PLY file: %s\n", e.what());
    MOCHI_ERROR_SET(error, "Failed to read PLY file.");
  } catch (...) {
    MOCHI_ERROR_SET(error, "Failed to read PLY file.");
  }
}

void mochi::model::LoadPlyFromBytes(ModelData& outData, Span<char const> data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  try {
    SpanStreamBuf buf(data);
    std::istream is(&buf);
    happly::PLYData ply(is);
    BuildModelFromPly(outData, ply, error);
  } catch (std::exception const& e) {
    MOCHI_LOG_WARNING("Failed to parse PLY data: %s\n", e.what());
    MOCHI_ERROR_SET(error, "Failed to parse PLY data.");
  } catch (...) {
    MOCHI_ERROR_SET(error, "Failed to parse PLY data.");
  }
}

// ---------------------------------------------------------------------------
// OFF loading (hand-rolled ASCII parser).
// ---------------------------------------------------------------------------

namespace {

// Parses an ASCII OFF mesh. Skips full-line comments (lines whose first non-space character is '#')
// and consumes the optional "OFF" header token; tagged variants (COFF, NOFF, ...) carry extra
// per-vertex columns and are explicitly rejected rather than misparsed. Reads the `nV nF nE`
// counts, then the vertices and faces. Polygon faces are fan-triangulated. Standard OFF stores xyz
// vertices with shared indices (no welding needed).
void LoadOffImpl(ModelData& outData, Span<char const> data, Error& error) {
  MOCHI_ERROR_RETURN(error);

  // Strip full-line comments so the token stream sees only data. The cleaned text is bounded by the
  // input size, so reserve once to avoid repeated reallocations.
  std::string cleaned;
  cleaned.reserve(data.size());
  {
    std::istringstream raw(std::string(data.data(), data.size()));
    std::string line;
    while (std::getline(raw, line)) {
      size_t i = 0;
      while (i < line.size() && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r')) {
        ++i;
      }
      if (i < line.size() && line[i] == '#') {
        continue;
      }
      cleaned += line;
      cleaned += '\n';
    }
  }

  std::istringstream stream(cleaned);
  std::string header;
  MOCHI_ERROR_IF(!(stream >> header), error, "OFF data is empty.");
  MOCHI_ERROR_RETURN(error);

  int numVertices = 0;
  int numFaces = 0;
  int numEdges = 0;
  if (header == "OFF") {
    MOCHI_ERROR_IF(
        !(stream >> numVertices >> numFaces >> numEdges), error, "OFF header is malformed.");
  } else {
    // No header token: the first token must itself be the integer vertex count. Variants such as
    // COFF/NOFF (which add per-vertex color/normal columns) are not supported; reject them cleanly
    // here rather than silently misparsing the extra columns as vertex coordinates.
    std::istringstream headerStream(header);
    MOCHI_ERROR_IF(
        !(headerStream >> numVertices) || !headerStream.eof(),
        error,
        "Unsupported OFF variant (only plain OFF is supported).");
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF(!(stream >> numFaces >> numEdges), error, "OFF header is malformed.");
  }
  (void)numEdges; // The edge count is informational and not used.
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(numVertices <= 0 || numFaces <= 0, error, "OFF mesh has no vertices or no faces.");
  MOCHI_ERROR_RETURN(error);

  // Read vertices incrementally so a malformed/huge declared count cannot overflow the int multiply
  // or pre-allocate a huge buffer; a short stream simply fails on the next read.
  DynamicArray<real> coordinates;
  for (int v = 0; v < numVertices; ++v) {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    MOCHI_ERROR_IF(!(stream >> x >> y >> z), error, "OFF vertex data is malformed.");
    MOCHI_ERROR_RETURN(error);
    coordinates.push_back(StaticCast<real>(x));
    coordinates.push_back(StaticCast<real>(y));
    coordinates.push_back(StaticCast<real>(z));
  }

  DynamicArray<int> connectivity;
  DynamicArray<int> faceIndices;
  for (int f = 0; f < numFaces; ++f) {
    int vertexCount = 0;
    MOCHI_ERROR_IF(!(stream >> vertexCount), error, "OFF face data is malformed.");
    MOCHI_ERROR_IF(vertexCount < 3, error, "OFF face has fewer than 3 vertices.");
    MOCHI_ERROR_RETURN(error);
    faceIndices.resize_noinit(vertexCount);
    for (int j = 0; j < vertexCount; ++j) {
      MOCHI_ERROR_IF(!(stream >> faceIndices[j]), error, "OFF face index is malformed.");
      MOCHI_ERROR_RETURN(error);
      // OFF stores shared indices (no welding), so reject out-of-range indices here rather than
      // emitting connectivity that would index out of bounds.
      MOCHI_ERROR_IF(
          faceIndices[j] < 0 || faceIndices[j] >= numVertices,
          error,
          "OFF face references an out-of-range vertex index.");
      MOCHI_ERROR_RETURN(error);
    }
    // Fan-triangulate: (0, j, j+1).
    for (int j = 1; j + 1 < vertexCount; ++j) {
      connectivity.push_back(faceIndices[0]);
      connectivity.push_back(faceIndices[j]);
      connectivity.push_back(faceIndices[j + 1]);
    }
  }

  // numFaces > 0 and each face has >= 3 vertices (both validated above), so every face contributes
  // at least one triangle and `connectivity` is necessarily non-empty here -- no separate check
  // needed.
  outData = {};
  outData.mesh.emplace();
  outData.mesh->nodesPerElement = 3;
  outData.mesh->coordinates = std::move(coordinates);
  outData.mesh->connectivity = std::move(connectivity);
}

} // namespace

void mochi::model::LoadOffFromFile(ModelData& outData, std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error);
  DynamicArray<char> const bytes = ReadFileBytes(std::filesystem::path(std::string(path)), error);
  MOCHI_ERROR_RETURN(error);
  LoadOffImpl(outData, MakeConstSpan(bytes), error);
}

void mochi::model::LoadOffFromBytes(ModelData& outData, Span<char const> data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  LoadOffImpl(outData, data, error);
}
