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

#include "meshing/mesh_conversion.h"

#include <mochi_core/geometry/grid_sdf_params.h>
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/log.h>
#include <mochi_mesh/surface_remeshing.h>
#include <mochi_physics/utils/mochi_model_utils.h>
#include <mochi_renderer/utils.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace superdex::studio {

bool ConvertToGlb(
    mochi::Path const& inputPath,
    mochi::Path const& outputPath,
    mochi::CoordinateSpaceConverter const* converter) {
  std::string const& lowerInput = inputPath.AsLowercaseString();
  std::vector<mochi_renderer::MeshSection> sections;
  if (lowerInput.ends_with(".dae")) {
    sections = mochi_renderer::ReadColladaFromFile(inputPath.ToString().c_str());
  } else if (lowerInput.ends_with(".obj")) {
    sections = mochi_renderer::ReadObjFromFile(inputPath.ToString().c_str());
  } else if (lowerInput.ends_with(".stl")) {
    sections = mochi_renderer::ReadStlFromFile(inputPath.ToString().c_str());
  } else {
    MOCHI_LOG_WARNING(
        "ConvertToGlb supports only .dae, .obj, and .stl inputs; skipping '%s'.",
        inputPath.ToString().c_str());
    return false;
  }

  if (sections.empty()) {
    MOCHI_LOG_WARNING("Failed to parse mesh file '%s'.", inputPath.ToString().c_str());
    return false;
  }

  // Convert before computing normals so recomputed normals come out in the
  // target space; any normals already supplied by the reader (e.g. OBJ vn) are
  // converted here too.
  if (converter != nullptr) {
    mochi_renderer::ConvertMeshSectionsSpace(sections, *converter);
  }

  for (mochi_renderer::MeshSection& section : sections) {
    if (!section.hasNormals || section.normals.size() != section.positions.size()) {
      mochi_renderer::ComputeVertexNormalsAreaWeighted(
          section.positions, section.indices, section.normals);
    }
  }

  std::vector<uint8_t> const glb = mochi_renderer::BuildGlbFromMeshSections(sections);
  if (glb.empty()) {
    MOCHI_LOG_WARNING("Mesh '%s' produced no GLB geometry.", inputPath.ToString().c_str());
    return false;
  }

  std::ofstream out(outputPath.AsFilesystemPath(), std::ios::binary);
  if (!out) {
    MOCHI_LOG_ERROR("Failed to open '%s' for writing.", outputPath.ToString().c_str());
    return false;
  }
  out.write(reinterpret_cast<char const*>(glb.data()), static_cast<std::streamsize>(glb.size()));
  return out.good();
}

mochi::ModelData MeshSectionsToModel(
    std::vector<mochi_renderer::MeshSection> const& sections,
    mochi::CoordinateSpaceConverter const* converter) {
  size_t totalCoords = 0;
  size_t totalIndices = 0;
  for (mochi_renderer::MeshSection const& section : sections) {
    if (section.positions.empty() || section.indices.empty()) {
      continue;
    }
    totalCoords += section.positions.size();
    totalIndices += section.indices.size();
  }

  mochi::MeshData mesh;
  mesh.nodesPerElement = 3;
  mesh.coordinates.reserve(totalCoords);
  mesh.connectivity.reserve(totalIndices);

  int vertexOffset = 0;
  for (mochi_renderer::MeshSection const& section : sections) {
    if (section.positions.empty() || section.indices.empty()) {
      continue;
    }
    for (size_t v = 0; v + 2 < section.positions.size(); v += 3) {
      mesh.coordinates.push_back(static_cast<mochi::real>(section.positions[v]));
      mesh.coordinates.push_back(static_cast<mochi::real>(section.positions[v + 1]));
      mesh.coordinates.push_back(static_cast<mochi::real>(section.positions[v + 2]));
    }
    for (int const index : section.indices) {
      mesh.connectivity.push_back(index + vertexOffset);
    }
    vertexOffset += static_cast<int>(section.positions.size() / 3);
  }
  if (converter != nullptr) {
    converter->TranslationsToOutput(mochi::MakeSpan(mesh.coordinates), mochi::ErrorAssert{});
  }

  mochi::ModelData data;
  if (!mesh.coordinates.empty() && !mesh.connectivity.empty()) {
    data.mesh = std::move(mesh);
  }
  return data;
}

bool ConvertToH5(
    mochi::Path const& inputPath,
    mochi::Path const& outputPath,
    mochi::CoordinateSpaceConverter const* converter,
    bool remesh,
    bool bakeSdf) {
  std::string const& lowerInput = inputPath.AsLowercaseString();
  mochi::ErrorLog error;
  mochi::ModelData data;

  if (lowerInput.ends_with(".stl") || lowerInput.ends_with(".obj") ||
      lowerInput.ends_with(".off") || lowerInput.ends_with(".ply")) {
    // Collision formats are loaded directly into Mochi space by CGAL; they carry
    // no section geometry, so `converter` does not apply to this path.
    data = mochi::model_utils::LoadFromFile(inputPath.ToString(), error);
    if (!error.IsOK()) {
      return false;
    }
  } else if (lowerInput.ends_with(".dae")) {
    std::vector<mochi_renderer::MeshSection> const sections =
        mochi_renderer::ReadColladaFromFile(inputPath.ToString().c_str());
    if (sections.empty()) {
      MOCHI_LOG_WARNING("Failed to parse COLLADA file '%s'.", inputPath.ToString().c_str());
      return false;
    }
    data = MeshSectionsToModel(sections, converter);
  } else if (lowerInput.ends_with(".glb")) {
    std::vector<mochi_renderer::MeshSection> const sections =
        mochi_renderer::ReadGlbFromFile(inputPath.ToString().c_str());
    if (sections.empty()) {
      MOCHI_LOG_WARNING("Failed to parse GLB file '%s'.", inputPath.ToString().c_str());
      return false;
    }
    data = MeshSectionsToModel(sections, converter);
  } else {
    MOCHI_LOG_WARNING(
        "ConvertToH5 supports only .stl, .obj, .off, .ply, .dae, and .glb inputs; skipping '%s'.",
        inputPath.ToString().c_str());
    return false;
  }

  if (!data.mesh.has_value()) {
    MOCHI_LOG_WARNING("Mesh '%s' produced no usable geometry.", inputPath.ToString().c_str());
    return false;
  }

  // Remesh the surface. Only triangle meshes are supported by RemeshSurface; on
  // failure we keep the original geometry (a separate Error keeps a remesh failure
  // from aborting the save below).
  if (remesh && data.mesh->nodesPerElement == 3) {
    mochi::mesh::SurfaceRemeshingParams params;
    params.relativeToMeshSize = true;
    params.edgeSize = 0.05f;
    mochi::ErrorLog remeshError;
    mochi::MeshData remeshed = mochi::mesh::RemeshSurface(*data.mesh, params, remeshError);
    if (remeshError.IsOK()) {
      data.mesh = std::move(remeshed);
    } else {
      MOCHI_LOG_WARNING(
          "Mesh remesh failed for '%s'; saving original geometry.", inputPath.ToString().c_str());
    }
  }

  // Bake a signed distance field. BakeSdf requires a triangle/tetrahedral mesh; on
  // failure we save the mesh without an SDF (a separate Error keeps a bake failure
  // from aborting the save below).
  if (bakeSdf) {
    mochi::ErrorLog sdfError;
    mochi::GridSdfParams params;
    mochi::model_utils::BakeSdf(data, params, sdfError);
    if (!sdfError.IsOK()) {
      MOCHI_LOG_WARNING(
          "SDF bake failed for '%s'; saving without an SDF.", inputPath.ToString().c_str());
    }
  }

  mochi::model_utils::SaveToFile(data, outputPath.ToString(), mochi::FileFormat::H5, error);
  return error.IsOK();
}

} // namespace superdex::studio
