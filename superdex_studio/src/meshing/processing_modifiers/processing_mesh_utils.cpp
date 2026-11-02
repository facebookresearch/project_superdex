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
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_mesh/step_tessellation.h>
#include <mochi_physics/utils/mochi_model_utils.h>
#include <mochi_renderer/model_geometry.h> // BuildMochiModelGeometry
#include <mochi_renderer/render_space.h> // RenderSpace
#include <mochi_renderer/utils.h> // Read*FromFile, Write*ToFile, ConvertMeshSectionsSpace, normals

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace superdex::studio::processing {

using mochi_renderer::MeshSection;

// --- File-size estimate calibration constants --------------------------------------------------
// These drive EstimateGlbSizeBytes / EstimateVtkSizeBytes / EstimateObjSizeBytes /
// EstimateMochiH5SizeBytes. They are intentionally coarse; tune them against real exported files
// (the estimates target ~10% accuracy, not exactness).
namespace {

// GLB (binary glTF): float3 position + float3 normal per vertex, 32-bit indices per triangle
// corner, plus a fixed chunk/header/material overhead.
constexpr int64_t kGlbBytesPerVertex = 24; // 3 * float32 position + 3 * float32 normal
constexpr int64_t kGlbBytesPerIndex = 4; // one uint32 index (3 per triangle)
constexpr int64_t kGlbOverheadBytes = 1024; // JSON chunk + GLB/binary headers + one material

// ASCII Legacy VTK: one XYZ line per node, and per tetrahedron one connectivity line plus a cell
// type line, plus the fixed headers. Text width varies with values and index counts.
constexpr int64_t kVtkBytesPerNode = 37;
constexpr int64_t kVtkBytesPerTet = 24;
constexpr int64_t kVtkOverheadBytes = 130;

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

// sRGB transfer function, decoding direction (IEC 61966-2-1:1999): converts a nonlinear sRGB
// component in [0, 1], as CAD tools and color pickers store it, to linear light, PBR
float SrgbToLinear(float srgb) {
  constexpr float kLinearThreshold = 0.04045f;
  constexpr float kLinearSlope = 12.92f;
  constexpr float kOffset = 0.055f;
  constexpr float kExponent = 2.4f;
  return srgb <= kLinearThreshold ? srgb / kLinearSlope
                                  : std::pow((srgb + kOffset) / (1.0f + kOffset), kExponent);
}

// One 8-bit channel of a packed 0xRRGGBB color as a component in [0, 1]. @p shift selects it:
// 16 = red, 8 = green, 0 = blue.
float UnpackChannel(uint32_t rgb, int shift) {
  return static_cast<float>((rgb >> shift) & 0xFFu) / 255.0f;
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

int64_t EstimateVtkSizeBytes(mochi::MeshData const& mesh) {
  auto const numNodes = static_cast<int64_t>(mesh.GetNumNodes());
  auto const numTets = static_cast<int64_t>(mesh.GetNumElements());
  if (mesh.nodesPerElement != 4 || numTets == 0) {
    return 0;
  }
  return kVtkOverheadBytes + numNodes * kVtkBytesPerNode + numTets * kVtkBytesPerTet;
}

int64_t EstimateObjSizeBytes(mochi::MeshData const& mesh) {
  auto const numVerts = static_cast<int64_t>(mesh.GetNumNodes());
  auto const numTris = static_cast<int64_t>(mesh.GetNumElements());
  if (numTris == 0) {
    return 0;
  }
  return numVerts * kObjBytesPerVertex + numTris * kObjBytesPerTriangle;
}

int64_t EstimateMochiH5SizeBytes(mochi::ModelData const& model) {
  auto const bytes = [](std::size_t count, std::size_t size) {
    return static_cast<int64_t>(count * size);
  };
  int64_t meshBytes = 0;
  for (std::optional<mochi::MeshData> const* mesh :
       {&model.mesh, &model.contactSkinMesh, &model.visualMesh}) {
    if (!mesh->has_value()) {
      continue;
    }
    meshBytes += bytes((*mesh)->coordinates.size(), sizeof(mochi::real)) +
        bytes((*mesh)->connectivity.size(), sizeof(int32_t));
    if ((*mesh)->skinning.has_value()) {
      meshBytes += bytes((*mesh)->skinning->indices.size(), sizeof(int32_t)) +
          bytes((*mesh)->skinning->weights.size(), sizeof(mochi::real));
    }
  }
  int64_t const sdfBytes =
      model.sdf.has_value() ? bytes(model.sdf->values.size(), sizeof(mochi::real)) : 0;
  double const compressed = kH5OverheadBytes + static_cast<double>(meshBytes) * kH5MeshCompression +
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

mochi::MeshData BufferedStageOutput(
    mochi::MeshData const& tetMesh,
    std::vector<MeshSection> const& sections) {
  if (tetMesh.GetNumElements() > 0) {
    return tetMesh;
  }
  return MeshDataFromSections(sections);
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

void WeldIdenticalVertices(mochi::MeshData& mesh) {
  using Position = std::array<mochi::real, 3>;
  struct PositionHash {
    std::size_t operator()(Position const& p) const {
      std::size_t seed = 0;
      for (mochi::real const value : p) {
        seed ^= std::hash<mochi::real>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
      }
      return seed;
    }
  };
  int const numNodes = mesh.GetNumNodes();
  std::unordered_map<Position, int, PositionHash> indexOf;
  indexOf.reserve(static_cast<std::size_t>(numNodes));
  std::vector<int> remap(static_cast<std::size_t>(numNodes));
  mochi::DynamicArray<mochi::real> coordinates;
  coordinates.reserve(mesh.coordinates.size());
  for (int v = 0; v < numNodes; ++v) {
    Position const p = {
        mesh.coordinates[3 * v], mesh.coordinates[3 * v + 1], mesh.coordinates[3 * v + 2]};
    auto const [it, inserted] = indexOf.try_emplace(p, static_cast<int>(coordinates.size() / 3));
    if (inserted) {
      for (mochi::real const value : p) {
        coordinates.push_back(value);
      }
    }
    remap[static_cast<std::size_t>(v)] = it->second;
  }
  for (int& index : mesh.connectivity) {
    index = remap[static_cast<std::size_t>(index)];
  }
  mesh.coordinates = std::move(coordinates);
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
// converts to renderer space itself. See TessellateStepSections.
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

std::vector<MeshSection> TessellateStepSections(
    std::string const& path,
    mochi::mesh::StepTessellationParams const& params,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  std::vector<mochi::mesh::StepColorMesh> const meshes =
      mochi::mesh::TessellateStepByColor(path, params, error);
  MOCHI_ERROR_RETURN(error, {});

  std::vector<MeshSection> sections;
  sections.reserve(meshes.size());
  for (auto const& [rgb, mesh] : meshes) {
    MeshSection& section = sections.emplace_back(SectionFromMeshData(mesh));
    if (rgb) {
      section.baseColor = {
          SrgbToLinear(UnpackChannel(*rgb, 16)),
          SrgbToLinear(UnpackChannel(*rgb, 8)),
          SrgbToLinear(UnpackChannel(*rgb, 0)),
          1.0f};
    }
  }
  return sections;
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

mochi::MeshData DisplaySurface(mochi::MeshData const& mesh) {
  if (mesh.nodesPerElement != 4) {
    return mesh;
  }
  mochi::TriangularMesh const boundary = mochi::CreateBoundaryMesh(
      mochi::TetrahedralMesh{
          mochi::Unflatten<mochi::Real3 const>(mochi::MakeConstSpan(mesh.coordinates)),
          mochi::Unflatten<mochi::Int4 const>(mochi::MakeConstSpan(mesh.connectivity))});
  mochi::Span<mochi::real const> const coordinates = mochi::Flatten(boundary.GetNodeCoordinates());
  mochi::Span<int const> const connectivity = boundary.GetFlatConnectivity();

  mochi::MeshData surface;
  surface.nodesPerElement = mochi::TriangularMesh::kNodesPerElement;
  surface.coordinates.assign(coordinates.begin(), coordinates.end());
  surface.connectivity.assign(connectivity.begin(), connectivity.end());
  return surface;
}

mochi::MeshData ToMochiSpace(mochi::MeshData mesh) {
  RenderToMochiConverter().TranslationsToOutput(
      mochi::MakeSpan(mesh.coordinates), mochi::ErrorAssert{});
  return mesh;
}

void WriteVtkFile(std::string const& path, mochi::MeshData const& mesh, mochi::Error& error) {
  MOCHI_ERROR_IF(
      mesh.nodesPerElement != 4, error, "Only tetrahedral meshes can be written as .vtk.");
  MOCHI_ERROR_RETURN(error);
  std::ofstream out(path);
  MOCHI_ERROR_IF(!out, error, "Failed to open the .vtk file for writing.");
  MOCHI_ERROR_RETURN(error);
  out << std::setprecision(std::numeric_limits<mochi::real>::max_digits10);
  out << "# vtk DataFile Version 3.0\nSuperdex Studio tetrahedral mesh\nASCII\n"
      << "DATASET UNSTRUCTURED_GRID\nPOINTS " << mesh.GetNumNodes() << " double\n";
  auto const& converter = RenderToMochiConverter();
  for (int node = 0; node < mesh.GetNumNodes(); ++node) {
    mochi::Real3 const position = converter.TranslationToOutput(
        mochi::Real3{
            mesh.coordinates[3 * node],
            mesh.coordinates[(3 * node) + 1],
            mesh.coordinates[(3 * node) + 2]});
    for (int axis = 0; axis < 3; ++axis) {
      // Adding zero writes -0 as 0.
      out << position[axis] + mochi::real{0} << (axis < 2 ? ' ' : '\n');
    }
  }
  out << "CELLS " << mesh.GetNumElements() << ' ' << 5 * mesh.GetNumElements() << '\n';
  for (int tet = 0; tet < mesh.GetNumElements(); ++tet) {
    out << '4';
    for (int corner = 0; corner < 4; ++corner) {
      out << ' ' << mesh.connectivity[(4 * tet) + corner];
    }
    out << '\n';
  }
  // Cell type 10 is VTK_TETRA.
  out << "CELL_TYPES " << mesh.GetNumElements() << '\n';
  for (int tet = 0; tet < mesh.GetNumElements(); ++tet) {
    out << "10\n";
  }
  out.close();
  MOCHI_ERROR_IF(!out, error, "Failed to write the .vtk file.");
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

} // namespace superdex::studio::processing
