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

#include "meshing/processing_modifiers/processing_mesh_utils.h"

#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_physics/utils/mochi_model_utils.h>
#include <mochi_renderer/mesh.h> // BuildMochiModelGeometry
#include <mochi_renderer/render_space.h> // RenderSpace
#include <mochi_renderer/utils.h> // Read*FromFile, Write*ToFile, ConvertMeshSectionsSpace, normals

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <utility>

namespace superdex::studio::processing {

using mochi_renderer::MeshSection;

// --- File-size estimate calibration constants --------------------------------------------------
// These drive EstimateGlbSizeBytes / EstimateObjSizeBytes / EstimateMochiH5SizeBytes. They are
// intentionally coarse; tune them against real exported files (the estimates target ~10% accuracy,
// not exactness).
namespace {

// GLB (binary glTF): float3 position + float3 normal per vertex, 32-bit indices per triangle
// corner, plus a fixed chunk/header/material overhead.
constexpr int64_t kGlbBytesPerVertex = 24; // 3 * float32 position + 3 * float32 normal
constexpr int64_t kGlbBytesPerIndex = 4; // one uint32 index (3 per triangle)
constexpr int64_t kGlbOverheadBytes = 1024; // JSON chunk + GLB/binary headers + one material

// OBJ (text): "v x y z" + "vn x y z" lines per vertex and an "f a//na b//nb c//nc" line per
// triangle. Text size varies with coordinate magnitude/precision, so this is coarser than GLB.
constexpr int64_t kObjBytesPerVertex = 52; // v line + vn line, both ~26 chars
constexpr int64_t kObjBytesPerTriangle = 22; // one face line with normal references

// .mochi.h5 (HDF5, gzip/deflate-6 + shuffle filter). Uncompressed, the surface mesh is float coords
// + int32 connectivity and the SDF is one real per grid cell; the file is
//     overhead + meshCompression * meshUncompressed + sdfCompression * sdfUncompressed.
// Calibrated against a set of Studio exports (2026-07-14): mesh and SDF both deflate to ~0.44, and
// there is a large, roughly-fixed ~66 KiB overhead that dominates small exports (it explains why a
// coarse grid with a tiny mesh still lands ~200 KB). SDF compressibility is somewhat shape
// dependent (~+/-12% across models at a fixed cell count), so very-well- or very-poorly-compressing
// shapes can fall outside 10%; these are tuned to the representative case. Complex surface
// geometries tend to be underestimated. Re-fit here if the exports drift.
constexpr double kH5MeshCompression = 0.44;
constexpr double kH5SdfCompression = 0.44;
constexpr int64_t kH5OverheadBytes = 68000; // ~66 KiB fixed HDF5 / filter overhead

// --- Mesh-file space conversion ----------------------------------------------------------------
// The two directions of the convention documented in the header. Built once: a converter is a pair
// of basis matrices, and these two are fixed for the life of the process.

mochi::CoordinateSpaceConverter const& MochiToRenderConverter() {
  static mochi::CoordinateSpaceConverter const converter(
      mochi::CoordinateSpace::Default(), mochi_renderer::RenderSpace());
  return converter;
}

mochi::CoordinateSpaceConverter const& RenderToMochiConverter() {
  static mochi::CoordinateSpaceConverter const converter(
      mochi_renderer::RenderSpace(), mochi::CoordinateSpace::Default());
  return converter;
}

} // namespace

int64_t EstimateGlbSizeBytes(mochi::MeshData const& mesh) {
  auto const numVerts = static_cast<int64_t>(mesh.GetNumNodes());
  auto const numTris = static_cast<int64_t>(mesh.GetNumElements());
  if (numTris == 0) {
    return 0;
  }
  return kGlbOverheadBytes + numVerts * kGlbBytesPerVertex + numTris * 3 * kGlbBytesPerIndex;
}

int64_t EstimateObjSizeBytes(mochi::MeshData const& mesh) {
  auto const numVerts = static_cast<int64_t>(mesh.GetNumNodes());
  auto const numTris = static_cast<int64_t>(mesh.GetNumElements());
  if (numTris == 0) {
    return 0;
  }
  return numVerts * kObjBytesPerVertex + numTris * kObjBytesPerTriangle;
}

int64_t EstimateMochiH5SizeBytes(mochi::MeshData const& mesh, int64_t sdfValueCount) {
  auto const numVerts = static_cast<int64_t>(mesh.GetNumNodes());
  auto const numTris = static_cast<int64_t>(mesh.GetNumElements());
  int64_t const coordBytes = numVerts * 3 * static_cast<int64_t>(sizeof(mochi::real));
  int64_t const connBytes = numTris * 3 * static_cast<int64_t>(sizeof(int32_t));
  int64_t const sdfBytes =
      std::max<int64_t>(0, sdfValueCount) * static_cast<int64_t>(sizeof(mochi::real));
  double const compressed = kH5OverheadBytes +
      static_cast<double>(coordBytes + connBytes) * kH5MeshCompression +
      static_cast<double>(sdfBytes) * kH5SdfCompression;
  return static_cast<int64_t>(compressed);
}

mochi::MeshData MeshDataFromSections(std::vector<MeshSection> const& sections) {
  mochi::MeshData mesh;
  mesh.nodesPerElement = 3;
  for (auto const& section : sections) {
    int const base = static_cast<int>(mesh.coordinates.size() / 3);
    for (float const value : section.positions) {
      mesh.coordinates.push_back(static_cast<mochi::real>(value));
    }
    for (int const index : section.indices) {
      mesh.connectivity.push_back(base + index);
    }
  }
  return mesh;
}

MeshSection SectionFromMeshData(mochi::MeshData const& mesh) {
  MeshSection section;
  section.positions.reserve(mesh.coordinates.size());
  for (mochi::real const coordinate : mesh.coordinates) {
    section.positions.push_back(static_cast<float>(coordinate));
  }
  section.indices.assign(mesh.connectivity.begin(), mesh.connectivity.end());
  std::vector<float> faceNormals;
  mochi_renderer::ComputeFaceNormals(section.positions, section.indices, faceNormals);
  mochi_renderer::ComputeVertexNormalsAngleWeighted(
      section.positions, faceNormals, section.indices, section.normals);
  section.hasNormals = section.normals.size() == section.positions.size();
  return section;
}

void ApplyTransform(
    mochi::MeshData& mesh,
    mochi::Real3 const& scale,
    mochi::Quaternion const& rotation,
    mochi::Real3 const& translation) {
  using namespace mochi; // Real3 / Quaternion arithmetic
  for (std::size_t i = 0; i + 3 <= mesh.coordinates.size(); i += 3) {
    Real3 const local{mesh.coordinates[i], mesh.coordinates[i + 1], mesh.coordinates[i + 2]};
    Real3 const world = rotation * (local * scale) + translation;
    mesh.coordinates[i] = world[0];
    mesh.coordinates[i + 1] = world[1];
    mesh.coordinates[i + 2] = world[2];
  }
}

bool EndsWithNoCase(std::string const& text, char const* suffix) {
  std::size_t const n = std::strlen(suffix);
  if (text.size() < n) {
    return false;
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (std::tolower(static_cast<unsigned char>(text[text.size() - n + i])) !=
        std::tolower(static_cast<unsigned char>(suffix[i]))) {
      return false;
    }
  }
  return true;
}

// No .step/.stp case here on purpose: STEP is tessellated by the superdex_mesh_cli helper, which
// converts to renderer space itself. See TessellateCadModelFile (cad_model_asset.cpp).
std::vector<MeshSection> ReadSectionsInRenderSpace(std::string const& path) {
  std::vector<MeshSection> sections;
  bool sourceIsMochiSpace = false;
  if (EndsWithNoCase(path, ".glb") || EndsWithNoCase(path, ".gltf")) {
    sections = mochi_renderer::ReadGlbFromFile(path.c_str());
  } else if (EndsWithNoCase(path, ".obj")) {
    sections = mochi_renderer::ReadObjFromFile(path.c_str());
    sourceIsMochiSpace = true;
  } else if (EndsWithNoCase(path, ".stl")) {
    sections = mochi_renderer::ReadStlFromFile(path.c_str());
    sourceIsMochiSpace = true;
  } else if (EndsWithNoCase(path, ".dae")) {
    sections = mochi_renderer::ReadColladaFromFile(path.c_str());
  }
  if (sourceIsMochiSpace) {
    mochi_renderer::ConvertMeshSectionsSpace(sections, MochiToRenderConverter());
  }
  return sections;
}

mochi::MeshData LoadRenderMesh(std::string const& path, mochi::Error& error) {
  MOCHI_ERROR_IF_NOT(
      EndsWithNoCase(path, ".glb") || EndsWithNoCase(path, ".gltf") ||
          EndsWithNoCase(path, ".obj") || EndsWithNoCase(path, ".stl") ||
          EndsWithNoCase(path, ".dae"),
      error,
      "Unsupported render-model file extension (expected obj/glb/stl/dae).");
  MOCHI_ERROR_RETURN(error, {});
  mochi::MeshData mesh = MeshDataFromSections(ReadSectionsInRenderSpace(path));
  MOCHI_ERROR_IF(mesh.GetNumElements() == 0, error, "Render model produced no triangles.");
  MOCHI_ERROR_RETURN(error, {});
  return mesh;
}

mochi::MeshData LoadMochiMesh(std::string const& path, mochi::Error& error) {
  mochi::ModelData const model = mochi::model_utils::LoadFromFile(path, error);
  MOCHI_ERROR_RETURN(error, {});
  if (model.mesh.has_value() && model.mesh->nodesPerElement == 3 &&
      model.mesh->GetNumElements() > 0) {
    mochi::MeshData mesh = *model.mesh; // stored triangle surface, in Mochi space
    MochiToRenderConverter().TranslationsToOutput(
        mochi::MakeSpan(mesh.coordinates), mochi::ErrorAssert{});
    return mesh;
  }
  // Tet mesh or procedural shape: extract a surface. A null converter makes BuildMochiModelGeometry
  // apply the same Mochi -> RenderSpace mapping internally.
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;
  if (!mochi_renderer::BuildMochiModelGeometry(model, nullptr, positions, normals, indices)) {
    MOCHI_ERROR_SET(error, "Mochi model has no usable surface geometry.");
    return {};
  }
  mochi::MeshData mesh;
  mesh.nodesPerElement = 3;
  for (float const value : positions) {
    mesh.coordinates.push_back(static_cast<mochi::real>(value));
  }
  for (int const index : indices) {
    mesh.connectivity.push_back(index);
  }
  return mesh;
}

void WriteObjFile(std::string const& path, mochi::MeshData const& mesh, mochi::Error& error) {
  std::vector<MeshSection> sections{SectionFromMeshData(mesh)};
  mochi_renderer::ConvertMeshSectionsSpace(sections, RenderToMochiConverter());
  if (!mochi_renderer::WriteObjToFile(path.c_str(), sections)) {
    MOCHI_ERROR_SET(error, "Failed to write the OBJ file.");
  }
}

void WriteGlbFile(
    std::string const& path,
    mochi::MeshData const& mesh,
    std::array<float, 4> const& baseColor,
    mochi::Error& error) {
  MeshSection section = SectionFromMeshData(mesh);
  section.baseColor = baseColor;
  std::vector<MeshSection> const sections{std::move(section)};
  if (!mochi_renderer::WriteGlbToFile(path.c_str(), sections)) {
    MOCHI_ERROR_SET(error, "Failed to write the GLB file.");
  }
}

std::string FileBaseName(std::string const& path) {
  std::filesystem::path const p(path);
  std::string const filename = p.filename().string();
  for (char const* const doubleExt : {".mochi.h5", ".mochi.json"}) {
    if (EndsWithNoCase(filename, doubleExt)) {
      return filename.substr(0, filename.size() - std::strlen(doubleExt));
    }
  }
  return p.stem().string();
}

// Whether @p path is a string that could name a file at all. Deliberately NOT an existence test:
// pointing at a folder that is not there yet is normal (the export creates it on write), so only a
// string no filesystem would accept is rejected. Screens out the characters Windows forbids in a
// name, plus control characters -- applied on every platform so a path typed on Linux stays usable
// on Windows, which is the whole point of keeping these files portable. Testing the path relative
// to its root name lets a drive letter's colon through.
static bool IsWellFormedPath(std::string const& path) {
  if (path.empty()) {
    return false;
  }
  std::string const tail = std::filesystem::path(path).relative_path().generic_string();
  return std::none_of(tail.begin(), tail.end(), [](char const c) {
    return static_cast<unsigned char>(c) < 0x20 ||
        std::string_view(R"(<>:"|?*)").find(c) != std::string_view::npos;
  });
}

std::string ExportDialogStartPath(
    std::string const& currentPath,
    std::string const& suggestedPath,
    std::string const& modelFolder) {
  if (IsWellFormedPath(currentPath)) {
    return currentPath;
  }
  if (IsWellFormedPath(suggestedPath)) {
    return suggestedPath;
  }
  if (modelFolder.empty()) {
    return {};
  }
  // Appending an empty element leaves a trailing separator, which is how the native dialog is told
  // "start here, with no file name pre-filled" -- without it the folder name itself is offered as
  // the name to save under.
  std::filesystem::path folder = std::filesystem::path(modelFolder) / "";
  folder.make_preferred();
  return folder.string();
}

bool SamePath(std::string_view a, std::string_view b) {
  if (a.empty() || b.empty()) {
    return false; // an unset path collides with nothing
  }
  return std::filesystem::path(a).lexically_normal().generic_string() ==
      std::filesystem::path(b).lexically_normal().generic_string();
}

} // namespace superdex::studio::processing
