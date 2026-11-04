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

#include "meshing/link_meshes.h"

#include "meshing/mesh_conversion.h"
#include "meshing/primitive_meshes.h"

#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/quaternion_utils.h>
#include <mochi_core/utils/rigid_body_utils.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_physics/utils/mochi_model_utils.h>
#include <mochi_renderer/render_space.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace superdex::studio {

using superdex::robotics::GeomReference;
using superdex::robotics::GeomType;
using superdex::robotics::IsPrimitiveGeom;

namespace {

mochi::Real3 const kUnitScale{mochi::real(1), mochi::real(1), mochi::real(1)};

std::string GeomLabel(GeomReference const& geom) {
  if (!geom.name.empty()) {
    return "geom \"" + std::string(geom.name) + "\"";
  }
  if (!geom.meshRef.empty()) {
    return "mesh \"" + std::string(geom.meshRef) + "\"";
  }
  return "an unnamed geom";
}

// The first whitespace-separated token of `line`, and where it ends.
std::string_view LineKeyword(std::string const& line, size_t& outEnd) {
  size_t const start = line.find_first_not_of(" \t");
  outEnd = line.find_first_of(" \t", start);
  return start == std::string::npos ? std::string_view{}
                                    : std::string_view(line).substr(start, outEnd - start);
}

// True if an OBJ file names a material library that exists, so its faces carry their own colors.
bool ObjHasMaterialLibrary(std::string const& objPath) {
  std::ifstream in(objPath);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    size_t end = 0;
    if (LineKeyword(line, end) != "mtllib" || end == std::string::npos) {
      continue;
    }
    size_t const start = line.find_first_not_of(" \t", end);
    std::error_code error;
    if (start != std::string::npos &&
        std::filesystem::exists(
            std::filesystem::path(objPath).parent_path() / line.substr(start), error)) {
      return true;
    }
  }
  return false;
}

// A mesh file's render sections in its own frame, in Mochi space, and whether the file colors them
// itself (DAE and GLB always do; an OBJ does with a material library). Empty if the format is not
// OBJ, STL, DAE, or GLB, or the file could not be read.
std::vector<mochi_renderer::MeshSection> ReadMeshSections(
    std::string const& file,
    bool& outOwnMaterials) {
  std::string const lower = mochi::Path{file}.AsLowercaseString();
  outOwnMaterials = false;
  if (lower.ends_with(".obj")) {
    outOwnMaterials = ObjHasMaterialLibrary(file);
    return mochi_renderer::ReadObjFromFile(file.c_str());
  }
  if (lower.ends_with(".stl")) {
    return mochi_renderer::ReadStlFromFile(file.c_str());
  }
  bool const isCollada = lower.ends_with(".dae");
  if (!isCollada && !lower.ends_with(".glb")) {
    return {};
  }
  std::vector<mochi_renderer::MeshSection> sections = isCollada
      ? mochi_renderer::ReadColladaFromFile(file.c_str())
      : mochi_renderer::ReadGlbFromFile(file.c_str());
  // These readers return geometry in the renderer's space.
  mochi_renderer::ConvertMeshSectionsSpace(
      sections,
      mochi::CoordinateSpaceConverter(
          mochi_renderer::RenderSpace(), mochi::CoordinateSpace::Default()));
  outOwnMaterials = true;
  return sections;
}

// Scales positions per axis and then applies `transform`. Normals follow the inverse-transpose of
// the scale, then the rotation.
void ScaleAndTransform(
    mochi_renderer::MeshSection& section,
    mochi::Real3 const& scale,
    mochi::TransformRT const& transform) {
  for (size_t i = 0; i + 2 < section.positions.size(); i += 3) {
    mochi::Real3 const p{
        section.positions[i] * scale[0],
        section.positions[i + 1] * scale[1],
        section.positions[i + 2] * scale[2]};
    mochi::Real3 const q = transform.TransformPoint(p);
    section.positions[i] = static_cast<float>(q[0]);
    section.positions[i + 1] = static_cast<float>(q[1]);
    section.positions[i + 2] = static_cast<float>(q[2]);
  }
  if (scale[0] == 0 || scale[1] == 0 || scale[2] == 0) {
    // A flattened mesh has no inverse scale for its normals; the caller recomputes them.
    section.normals.clear();
    section.hasNormals = false;
    return;
  }
  for (size_t i = 0; i + 2 < section.normals.size(); i += 3) {
    mochi::Real3 n{
        section.normals[i] / scale[0],
        section.normals[i + 1] / scale[1],
        section.normals[i + 2] / scale[2]};
    n = transform.TransformDirection(mochi::Normalize(n));
    section.normals[i] = static_cast<float>(n[0]);
    section.normals[i + 1] = static_cast<float>(n[1]);
    section.normals[i + 2] = static_cast<float>(n[2]);
  }
}

// Render section of a tessellated primitive, with one vertex per face corner and normals that are
// smooth across curved surfaces but sharp across creases (box edges, cylinder rims).
mochi_renderer::MeshSection PrimitiveSection(mochi::MeshData const& mesh) {
  constexpr mochi::real kCreaseCos = 0.5; // 60 degrees
  auto point = [&](int v) {
    return mochi::Real3{
        mesh.coordinates[3 * v], mesh.coordinates[3 * v + 1], mesh.coordinates[3 * v + 2]};
  };
  size_t const numFaces = mesh.connectivity.size() / 3;
  std::vector<mochi::Real3> faceNormals(numFaces); // area-weighted
  std::vector<std::vector<size_t>> facesOfVertex(mesh.coordinates.size() / 3);
  for (size_t f = 0; f < numFaces; ++f) {
    int const* corner = &mesh.connectivity[3 * f];
    faceNormals[f] =
        mochi::Cross(point(corner[1]) - point(corner[0]), point(corner[2]) - point(corner[0]));
    for (int k = 0; k < 3; ++k) {
      facesOfVertex[corner[k]].push_back(f);
    }
  }
  mochi_renderer::MeshSection section;
  section.hasNormals = true;
  for (size_t f = 0; f < numFaces; ++f) {
    mochi::Real3 const faceDir = mochi::Normalize(faceNormals[f]);
    for (int k = 0; k < 3; ++k) {
      int const v = mesh.connectivity[3 * f + k];
      mochi::Real3 normal{};
      for (size_t g : facesOfVertex[v]) {
        if (mochi::Dot(mochi::Normalize(faceNormals[g]), faceDir) > kCreaseCos) {
          normal += faceNormals[g];
        }
      }
      normal = mochi::Normalize(normal);
      mochi::Real3 const p = point(v);
      for (int axis = 0; axis < 3; ++axis) {
        section.positions.push_back(static_cast<float>(p[axis]));
        section.normals.push_back(static_cast<float>(normal[axis]));
      }
      section.indices.push_back(static_cast<int>(3 * f + k));
    }
  }
  return section;
}

} // namespace

std::vector<mochi_renderer::MeshSection> GeomRenderSections(
    GeomReference const& geom,
    std::vector<std::string>& warnings) {
  std::vector<mochi_renderer::MeshSection> sections;
  bool ownMaterials = false;
  if (geom.type == GeomType::Mesh) {
    if (geom.meshFile.empty()) {
      warnings.push_back("Missing mesh file for " + GeomLabel(geom) + ".");
      return {};
    }
    std::string const file(geom.meshFile);
    sections = ReadMeshSections(file, ownMaterials);
    if (sections.empty()) {
      warnings.push_back(
          "Could not read " + GeomLabel(geom) + " as an OBJ, STL, DAE, or GLB mesh: " + file);
      return {};
    }
    for (auto& section : sections) {
      ScaleAndTransform(section, geom.meshScale, geom.linkFromGeom * geom.geomFromMesh);
    }
  } else if (IsPrimitiveGeom(geom.type)) {
    sections.push_back(PrimitiveSection(TessellatePrimitive(geom.type, geom.size)));
    ScaleAndTransform(sections.back(), kUnitScale, geom.linkFromGeom);
  } else {
    return {};
  }

  // MuJoCo colors a geom by its rgba or material and ignores the mesh's own materials. URDF colors
  // only meshes without materials of their own.
  if (geom.hasMaterial && !(geom.keepMeshMaterials && ownMaterials)) {
    std::shared_ptr<mochi_renderer::TextureImage const> texture;
    if (!geom.textureFile.empty()) {
      texture = mochi_renderer::ReadTextureImageFromFile(geom.textureFile.c_str());
      if (!texture) {
        warnings.push_back(
            "Texture " + std::string(geom.textureFile) + " is missing or not a PNG/JPEG.");
      }
    }
    for (auto& section : sections) {
      section.baseColor = {
          static_cast<float>(geom.rgba[0]),
          static_cast<float>(geom.rgba[1]),
          static_cast<float>(geom.rgba[2]),
          static_cast<float>(geom.rgba[3])};
      section.roughness = std::clamp(1.0f - static_cast<float>(geom.shininess), 0.0f, 1.0f);
      section.metallic = 0.0f;
      section.baseColorTexture = nullptr;
      if (!texture) {
        continue;
      }
      if (!section.hasTexcoords) {
        warnings.push_back(
            GeomLabel(geom) + " has a texture but no texture coordinates; using its color.");
        continue;
      }
      section.baseColorTexture = texture;
      // texrepeat scales MuJoCo's (bottom-left origin) coordinates; ours have a top-left origin.
      for (size_t i = 0; i + 1 < section.texcoords.size(); i += 2) {
        section.texcoords[i] *= static_cast<float>(geom.texRepeat[0]);
        section.texcoords[i + 1] =
            1.0f - (1.0f - section.texcoords[i + 1]) * static_cast<float>(geom.texRepeat[1]);
      }
    }
  }
  return sections;
}

bool BuildLinkGlb(
    GeomList const& geoms,
    mochi::Path const& outputPath,
    mochi::CoordinateSpaceConverter const& toRender,
    std::vector<std::string>& warnings) {
  std::vector<mochi_renderer::MeshSection> sections;
  for (GeomReference const& geom : geoms) {
    std::vector<mochi_renderer::MeshSection> geomSections = GeomRenderSections(geom, warnings);
    std::ranges::move(geomSections, std::back_inserter(sections));
  }
  if (sections.empty()) {
    return false;
  }
  mochi_renderer::ConvertMeshSectionsSpace(sections, toRender);
  for (mochi_renderer::MeshSection& section : sections) {
    if (!section.hasNormals || section.normals.size() != section.positions.size()) {
      mochi_renderer::ComputeVertexNormalsAreaWeighted(
          section.positions, section.indices, section.normals);
      section.hasNormals = true;
    }
  }
  return mochi_renderer::WriteGlbToFile(outputPath.ToString().c_str(), sections);
}

mochi::MeshData
GeomCollisionMesh(GeomReference const& geom, bool inLinkFrame, std::vector<std::string>& warnings) {
  mochi::ModelData data;
  mochi::Real3 scale = kUnitScale;
  mochi::TransformRT linkFromData = geom.linkFromGeom;
  if (geom.type == GeomType::Mesh) {
    if (geom.meshFile.empty()) {
      warnings.push_back("Missing mesh file for " + GeomLabel(geom) + ".");
      return {};
    }
    std::string const file(geom.meshFile);
    std::string const lower = mochi::Path{file}.AsLowercaseString();
    mochi::ErrorLog error;
    if (lower.ends_with(".dae") || lower.ends_with(".glb")) {
      bool ownMaterials = false;
      data = MeshSectionsToModel(ReadMeshSections(file, ownMaterials), nullptr);
    } else {
      data = mochi::model_utils::LoadFromFile(file, error);
    }
    if (!error.IsOK() || !data.mesh.has_value() || data.mesh->nodesPerElement != 3) {
      warnings.push_back("Could not load " + GeomLabel(geom) + " as a triangle mesh.");
      return {};
    }
    scale = geom.meshScale;
    linkFromData = geom.linkFromGeom * geom.geomFromMesh;
  } else if (IsPrimitiveGeom(geom.type)) {
    data.mesh = TessellatePrimitive(geom.type, geom.size);
  } else {
    return {};
  }
  mochi::ErrorLog error;
  mochi::model_utils::BakeTransform(
      data, scale, inLinkFrame ? linkFromData : mochi::TransformRT::Identity(), error);
  return error.IsOK() ? std::move(*data.mesh) : mochi::MeshData{};
}

namespace {

// Volume, centroid, and second moment about the centroid (per unit density) of a mesh, as MuJoCo's
// default ("legacy") mesh inertia computes them: every face spans a tetrahedron with an apex,
// counted with its absolute volume, so open or inconsistently wound meshes still weigh something.
// The apex is first the area-weighted face centroid, then the centroid that this yields. Empty if
// the mesh encloses no volume.
struct LegacyMeshVolume {
  mochi::real volume = 0;
  mochi::Vec4r centroid{};
  mochi::VMatrix3x3r secondMoment{};
};

std::optional<LegacyMeshVolume> ComputeLegacyMeshVolume(mochi::MeshData mesh) {
  int const numNodes = mesh.GetNumNodes();
  if (std::ranges::any_of(
          mesh.connectivity, [&](int index) { return index < 0 || index >= numNodes; })) {
    return std::nullopt;
  }
  mochi::Span<mochi::Real3> const coordinates{
      mochi::ReinterpretCast<mochi::Real3*>(mesh.coordinates.data()), mesh.coordinates.size() / 3};
  mochi::Span<mochi::Int3> const faces{
      mochi::ReinterpretCast<mochi::Int3*>(mesh.connectivity.data()), mesh.connectivity.size() / 3};
  mochi::Span<mochi::Real3 const> const tetCoordinates{coordinates.data(), coordinates.size()};
  mochi::Span<mochi::Int3 const> const tetFaces{faces.data(), faces.size()};

  mochi::Real3 apex{};
  mochi::real area = 0;
  for (mochi::Int3 const& face : faces) {
    mochi::Real3 const& a = coordinates[face[0]];
    mochi::Real3 const& b = coordinates[face[1]];
    mochi::Real3 const& c = coordinates[face[2]];
    mochi::real const faceArea = mochi::real(0.5) * mochi::Norm(mochi::Cross(b - a, c - a));
    apex += (a + b + c) * (faceArea / 3);
    area += faceArea;
  }
  if (area > 0) {
    apex = apex / area;
  }

  // The apex becomes the origin, and faces are re-wound to make every tetrahedron's volume
  // positive about it.
  mochi::Vec4r origin{};
  LegacyMeshVolume out;
  for (int pass = 0; pass < 2; ++pass) {
    for (mochi::Real3& p : coordinates) {
      p -= apex;
    }
    origin += mochi::ToSimd(apex);
    for (mochi::Int3& face : faces) {
      mochi::Real3 const& a = coordinates[face[0]];
      if (mochi::Dot(a, mochi::Cross(coordinates[face[1]] - a, coordinates[face[2]] - a)) < 0) {
        std::swap(face[1], face[2]);
      }
    }
    mochi::ComputeVolumeAndCentroid(tetCoordinates, tetFaces, out.centroid, out.volume);
    if (!mochi::IsFinite(out.volume) || out.volume <= 0) {
      return std::nullopt;
    }
    apex = mochi::ToReal3(out.centroid);
  }
  out.secondMoment = mochi::ComputeSecondMomentOfInertia(tetCoordinates, tetFaces, 1, {}) -
      mochi::Outer3(out.centroid, out.centroid) * out.volume;
  out.centroid += origin;
  return out;
}

} // namespace

namespace {

// Volume, centroid (link frame), and second moment about the centroid (per unit density) of a
// primitive geom, from the closed-form solids MuJoCo uses. Empty if it encloses no volume.
std::optional<LegacyMeshVolume> ComputePrimitiveVolume(GeomReference const& geom) {
  constexpr mochi::real kPi = mochi::kPI;
  mochi::real const a = geom.size[0];
  mochi::real const b = geom.size[1];
  mochi::real const c = geom.size[2];
  mochi::real volume = 0;
  mochi::Real3 moments{}; // integrals of x^2, y^2, z^2 over the solid, in the geom frame
  switch (geom.type) {
    case GeomType::Sphere:
      volume = 4 * kPi * a * a * a / 3;
      moments = mochi::Real3{a * a, a * a, a * a} * (volume / 5);
      break;
    case GeomType::Ellipsoid:
      volume = 4 * kPi * a * b * c / 3;
      moments = mochi::Real3{a * a, b * b, c * c} * (volume / 5);
      break;
    case GeomType::Box:
      volume = 8 * a * b * c;
      moments = mochi::Real3{a * a, b * b, c * c} * (volume / 3);
      break;
    case GeomType::Cylinder: // radius a, half-height b along Z
      volume = 2 * kPi * a * a * b;
      moments = mochi::Real3{volume * a * a / 4, volume * a * a / 4, volume * b * b / 3};
      break;
    case GeomType::Capsule: { // radius a, cylinder half-height b along Z, hemispherical caps
      mochi::real const cylinder = 2 * kPi * a * a * b;
      mochi::real const sphere = 4 * kPi * a * a * a / 3;
      volume = cylinder + sphere;
      mochi::real const radial = cylinder * a * a / 4 + sphere * a * a / 5;
      moments = mochi::Real3{
          radial, radial, cylinder * b * b / 3 + sphere * (b * b + 3 * b * a / 4 + a * a / 5)};
      break;
    }
    case GeomType::Mesh:
    case GeomType::Plane:
    case GeomType::Hfield:
    case GeomType::Sdf:
      return std::nullopt;
  }
  if (!mochi::IsFinite(volume) || volume <= 0) {
    return std::nullopt;
  }
  LegacyMeshVolume out;
  out.volume = volume;
  out.centroid = mochi::ToSimd(geom.linkFromGeom.GetTranslation());
  for (int axis = 0; axis < 3; ++axis) {
    mochi::Real3 basis{};
    basis[axis] = 1;
    mochi::Vec4r const column = mochi::ToSimd(geom.linkFromGeom.GetRotation() * basis);
    out.secondMoment += mochi::Outer3(column, column) * moments[axis];
  }
  return out;
}

} // namespace

std::optional<GeomsInertia> ComputeGeomsInertia(
    GeomList const& geoms,
    std::vector<std::string>& warnings) {
  constexpr mochi::real kMjcfDefaultDensity = 1000;
  mochi::real totalMass = 0;
  mochi::Vec4r firstMoment{};
  mochi::VMatrix3x3r secondMoment{};
  for (GeomReference const& geom : geoms) {
    mochi::real const density = geom.density.value_or(kMjcfDefaultDensity);
    if (geom.mass ? *geom.mass <= 0 : density <= 0) {
      continue;
    }
    std::optional<LegacyMeshVolume> volume;
    if (geom.type == GeomType::Mesh) {
      mochi::MeshData mesh = GeomCollisionMesh(geom, /*inLinkFrame=*/true, warnings);
      if (mesh.IsEmpty()) {
        continue;
      }
      volume = ComputeLegacyMeshVolume(std::move(mesh));
    } else if (IsPrimitiveGeom(geom.type)) {
      volume = ComputePrimitiveVolume(geom);
    } else {
      continue;
    }
    if (!volume) {
      warnings.push_back("Ignored the mass of " + GeomLabel(geom) + ", which encloses no volume.");
      continue;
    }
    mochi::real const geomDensity = geom.mass ? *geom.mass / volume->volume : density;
    mochi::real const mass = geomDensity * volume->volume;
    totalMass += mass;
    firstMoment += volume->centroid * mass;
    secondMoment += volume->secondMoment * geomDensity +
        mochi::Outer3(volume->centroid, volume->centroid) * mass;
  }
  if (totalMass <= 0) {
    return std::nullopt;
  }
  mochi::Vec4r const com = firstMoment / totalMass;
  mochi::VMatrix3x3r const moi =
      mochi::SecondMomentToMomentOfInertia(secondMoment - mochi::Outer3(com, com) * totalMass);
  return GeomsInertia{
      totalMass,
      mochi::ToReal3(com),
      mochi::Real6{moi[0][0], moi[0][1], moi[0][2], moi[1][1], moi[1][2], moi[2][2]}};
}

namespace {

// Copies the material library `source` to `copy`, with relative texture paths made absolute so
// they still resolve from the copy's folder.
void CopyMaterialLibrary(std::filesystem::path const& source, std::filesystem::path const& copy) {
  std::ifstream in(source);
  if (!in) {
    return;
  }
  std::string text;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    size_t end = 0;
    std::string_view const keyword = LineKeyword(line, end);
    bool const isTexture = keyword.starts_with("map_") || keyword == "bump" || keyword == "disp" ||
        keyword == "decal" || keyword == "norm" || keyword == "refl";
    // The texture file is the statement's last token, after any options.
    size_t const fileStart = line.find_last_of(" \t");
    if (isTexture && end != std::string::npos && fileStart != std::string::npos &&
        fileStart >= end) {
      std::filesystem::path const file(line.substr(fileStart + 1));
      if (file.is_relative()) {
        line = line.substr(0, fileStart + 1) + (source.parent_path() / file).string();
      }
    }
    text += line;
    text += '\n';
  }
  mochi::ErrorLog error;
  mochi::WriteFile(copy, text, error);
}

} // namespace

std::string FirstObjObjectFile(std::string const& objPath, std::filesystem::path const& outDir) {
  std::ifstream in(objPath);
  if (!in) {
    return objPath;
  }
  std::filesystem::path const sourceDir = std::filesystem::path(objPath).parent_path();
  // Different folders can hold OBJs and material libraries of the same name.
  std::string const prefix = std::to_string(std::hash<std::string>{}(objPath)) + "_";
  std::vector<std::pair<std::filesystem::path, std::string>> libraries; // source, copy name
  std::string kept;
  std::string line;
  bool sawFace = false;
  bool truncated = false;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    size_t end = 0;
    std::string_view const keyword = LineKeyword(line, end);
    if (keyword == "o" || keyword == "g") {
      truncated = truncated || sawFace;
    } else if (keyword == "f" || keyword == "l" || keyword == "p") {
      if (truncated) {
        continue;
      }
      sawFace = true;
    } else if (
        keyword == "mtllib" && end != std::string::npos &&
        line.find_first_not_of(" \t", end) != std::string::npos) {
      // tinyobjloader looks material libraries up next to the OBJ, so the copy gets its own.
      std::filesystem::path const library(line.substr(line.find_first_not_of(" \t", end)));
      std::string const copyName = prefix + library.filename().string();
      libraries.emplace_back(sourceDir / library, copyName);
      line = "mtllib " + copyName;
    }
    kept += line;
    kept += '\n';
  }
  if (!truncated) {
    return objPath;
  }
  for (auto const& [source, copyName] : libraries) {
    CopyMaterialLibrary(source, outDir / copyName);
  }
  std::filesystem::path const copy =
      outDir / (prefix + std::filesystem::path(objPath).filename().string());
  mochi::ErrorLog error;
  mochi::WriteFile(copy, kept, error);
  return error.IsOK() ? copy.string() : objPath;
}

bool FitPrimitiveToMesh(GeomReference& geom, std::vector<std::string>& warnings) {
  GeomReference asMesh = geom;
  asMesh.type = GeomType::Mesh;
  mochi::MeshData mesh = GeomCollisionMesh(asMesh, /*inLinkFrame=*/false, warnings);
  if (mesh.IsEmpty()) {
    return false;
  }
  std::optional<LegacyMeshVolume> const volume = ComputeLegacyMeshVolume(std::move(mesh));
  if (!volume) {
    warnings.push_back("Could not fit " + GeomLabel(geom) + ": its mesh encloses no volume.");
    return false;
  }
  // Principal moments in descending order, so the primitive's Z axis is the mesh's long axis.
  mochi::Real3 moments;
  mochi::Matrix3x3r axes;
  mochi::AnalyticalEigendecompSym(
      mochi::ToNdArray3x3(mochi::SecondMomentToMomentOfInertia(volume->secondMoment)),
      moments,
      &axes);
  // Half-extents of the uniform box with the same volume and principal moments.
  mochi::Real3 box;
  for (int i = 0; i < 3; ++i) {
    box[i] = mochi::Sqrt(
        mochi::Max(
            mochi::real(0),
            mochi::real(1.5) * (moments[(i + 1) % 3] + moments[(i + 2) % 3] - moments[i]) /
                volume->volume));
  }
  mochi::Real3 size{};
  switch (geom.type) {
    case GeomType::Sphere:
      size[0] = (box[0] + box[1] + box[2]) / 3;
      break;
    case GeomType::Capsule:
      size[0] = (box[0] + box[1]) / 2;
      size[1] = mochi::Max(mochi::real(0), box[2] - size[0] / 2);
      break;
    case GeomType::Cylinder:
      size[0] = (box[0] + box[1]) / 2;
      size[1] = box[2];
      break;
    case GeomType::Ellipsoid:
    case GeomType::Box:
      size = box;
      break;
    case GeomType::Mesh:
    case GeomType::Plane:
    case GeomType::Hfield:
    case GeomType::Sdf:
      return false;
  }
  geom.size = size * geom.fitScale;
  geom.linkFromGeom = geom.linkFromGeom * geom.geomFromMesh *
      mochi::TransformRT{mochi::QuaternionFromMatrix(axes), mochi::ToReal3(volume->centroid)};
  geom.meshFile = {};
  geom.meshScale = kUnitScale;
  geom.geomFromMesh = mochi::TransformRT::Identity();
  return true;
}

} // namespace superdex::studio
