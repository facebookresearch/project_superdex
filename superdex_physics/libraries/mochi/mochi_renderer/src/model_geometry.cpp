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

#include <mochi_renderer/model_geometry.h>

#include <mochi_renderer/render_space.h>
#include <mochi_renderer/utils.h>

#include <mochi_core/geometry/mesh_data_utils.h>
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <memory>
#include <utility>
#include <vector>

namespace mochi_renderer {

//--------------------------------------------------------------------------------------------------
// PROCEDURAL MESH GENERATORS (for Mochi models defined by an analytic shape)
//--------------------------------------------------------------------------------------------------

namespace {

struct ProceduralMesh {
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;
};

ProceduralMesh GenerateBoxMesh(mochi::Box const& box) {
  ProceduralMesh mesh;
  mesh.positions.reserve(72);
  mesh.normals.reserve(72);
  mesh.indices.reserve(36);
  auto const& c = box.center;
  auto const& h = box.halfExtents;
  auto const& q = box.rotation;
  mochi::Real3 const localCorners[8] = {
      {-h[0], -h[1], -h[2]},
      {+h[0], -h[1], -h[2]},
      {+h[0], +h[1], -h[2]},
      {-h[0], +h[1], -h[2]},
      {-h[0], -h[1], +h[2]},
      {+h[0], -h[1], +h[2]},
      {+h[0], +h[1], +h[2]},
      {-h[0], +h[1], +h[2]},
  };
  struct Face {
    int v[4];
    mochi::Real3 normal;
  };
  using R = mochi::real;
  Face const faces[6] = {
      {{1, 2, 6, 5}, {R(1), R(0), R(0)}},
      {{0, 4, 7, 3}, {R(-1), R(0), R(0)}},
      {{3, 7, 6, 2}, {R(0), R(1), R(0)}},
      {{0, 1, 5, 4}, {R(0), R(-1), R(0)}},
      {{4, 5, 6, 7}, {R(0), R(0), R(1)}},
      {{0, 3, 2, 1}, {R(0), R(0), R(-1)}},
  };
  for (auto const& face : faces) {
    int const base = static_cast<int>(mesh.positions.size()) / 3;
    mochi::Real3 const n = q * face.normal;
    for (int const corner : face.v) {
      mochi::Real3 const v = c + q * localCorners[corner];
      mesh.positions.push_back(static_cast<float>(v[0]));
      mesh.positions.push_back(static_cast<float>(v[1]));
      mesh.positions.push_back(static_cast<float>(v[2]));
      mesh.normals.push_back(static_cast<float>(n[0]));
      mesh.normals.push_back(static_cast<float>(n[1]));
      mesh.normals.push_back(static_cast<float>(n[2]));
    }
    mesh.indices.push_back(base);
    mesh.indices.push_back(base + 1);
    mesh.indices.push_back(base + 2);
    mesh.indices.push_back(base);
    mesh.indices.push_back(base + 2);
    mesh.indices.push_back(base + 3);
  }
  return mesh;
}

ProceduralMesh GeneratePlaneMesh(mochi::Plane const& plane) {
  ProceduralMesh mesh;
  mesh.positions.reserve(12);
  mesh.normals.reserve(12);
  mochi::Real3 const n = mochi::Normalize(plane.normal);
  mochi::Real3 const origin = n * plane.distance;
  mochi::Real3 tangent;
  if (std::abs(n[0]) < mochi::real(0.9)) {
    tangent = mochi::Normalize(
        mochi::Cross(n, mochi::Real3{mochi::real(1), mochi::real(0), mochi::real(0)}));
  } else {
    tangent = mochi::Normalize(
        mochi::Cross(n, mochi::Real3{mochi::real(0), mochi::real(1), mochi::real(0)}));
  }
  mochi::Real3 const bitangent = mochi::Cross(n, tangent);
  auto const halfSize = mochi::real(5);
  mochi::Real3 const corners[4] = {
      origin - tangent * halfSize - bitangent * halfSize,
      origin + tangent * halfSize - bitangent * halfSize,
      origin + tangent * halfSize + bitangent * halfSize,
      origin - tangent * halfSize + bitangent * halfSize,
  };
  for (auto const& v : corners) {
    mesh.positions.push_back(static_cast<float>(v[0]));
    mesh.positions.push_back(static_cast<float>(v[1]));
    mesh.positions.push_back(static_cast<float>(v[2]));
    mesh.normals.push_back(static_cast<float>(n[0]));
    mesh.normals.push_back(static_cast<float>(n[1]));
    mesh.normals.push_back(static_cast<float>(n[2]));
  }
  mesh.indices = {0, 1, 2, 0, 2, 3};
  return mesh;
}

ProceduralMesh GenerateSphereMesh(mochi::Sphere const& sphere) {
  ProceduralMesh mesh;
  int const numLon = 32;
  int const numLat = 16;
  int const numVerts = (numLat + 1) * (numLon + 1);
  mesh.positions.reserve(numVerts * 3);
  mesh.normals.reserve(numVerts * 3);
  mesh.indices.reserve(numLat * numLon * 6);
  auto const r = static_cast<float>(sphere.radius);
  auto const cx = static_cast<float>(sphere.center[0]);
  auto const cy = static_cast<float>(sphere.center[1]);
  auto const cz = static_cast<float>(sphere.center[2]);
  for (int lat = 0; lat <= numLat; ++lat) {
    float const theta =
        static_cast<float>(mochi::kPI) * static_cast<float>(lat) / static_cast<float>(numLat);
    float const sinT = std::sin(theta);
    float const cosT = std::cos(theta);
    for (int lon = 0; lon <= numLon; ++lon) {
      float const phi = 2.0f * static_cast<float>(mochi::kPI) * static_cast<float>(lon) /
          static_cast<float>(numLon);
      float const nx = sinT * std::cos(phi);
      float const ny = sinT * std::sin(phi);
      float const nz = cosT;
      mesh.positions.push_back(cx + r * nx);
      mesh.positions.push_back(cy + r * ny);
      mesh.positions.push_back(cz + r * nz);
      mesh.normals.push_back(nx);
      mesh.normals.push_back(ny);
      mesh.normals.push_back(nz);
    }
  }
  for (int lat = 0; lat < numLat; ++lat) {
    for (int lon = 0; lon < numLon; ++lon) {
      int const curr = lat * (numLon + 1) + lon;
      int const next = curr + numLon + 1;
      if (lat != 0) {
        mesh.indices.push_back(curr);
        mesh.indices.push_back(next);
        mesh.indices.push_back(curr + 1);
      }
      if (lat != numLat - 1) {
        mesh.indices.push_back(curr + 1);
        mesh.indices.push_back(next);
        mesh.indices.push_back(next + 1);
      }
    }
  }
  return mesh;
}

} // namespace

bool BuildMochiModelGeometry(
    mochi::ModelData const& modelData,
    mochi::CoordinateSpaceConverter const* converter,
    std::vector<float>& positions,
    std::vector<float>& vertexNormals,
    std::vector<int>& indices) {
  mochi::CoordinateSpaceConverter const defaultConverter(
      mochi::CoordinateSpace::Default(), RenderSpace());
  mochi::CoordinateSpaceConverter const& spaceConverter = converter ? *converter : defaultConverter;

  positions.clear();
  vertexNormals.clear();
  indices.clear();

  if (modelData.mesh) {
    mochi::Span<mochi::Real3 const> surfaceNodes;
    mochi::Span<mochi::Int3 const> surfaceTris;
    std::shared_ptr<mochi::TriangularMesh const> boundaryMesh;
    int const nodesPerElement = modelData.mesh->nodesPerElement;
    if (nodesPerElement == 4) {
      // Tetrahedral mesh: extract boundary surface
      auto nodes =
          mochi::Unflatten<mochi::Real3 const>(mochi::MakeConstSpan(modelData.mesh->coordinates));
      auto tets =
          mochi::Unflatten<mochi::Int4 const>(mochi::MakeConstSpan(modelData.mesh->connectivity));
      mochi::TetrahedralMesh tetMesh(nodes, tets);
      boundaryMesh = tetMesh.GetBoundaryMesh();
      surfaceNodes = boundaryMesh->GetNodeCoordinates();
      surfaceTris = boundaryMesh->GetElementConnectivity();
    } else if (nodesPerElement == 3) {
      // Surface mesh: use directly
      surfaceNodes =
          mochi::Unflatten<mochi::Real3 const>(mochi::MakeConstSpan(modelData.mesh->coordinates));
      surfaceTris =
          mochi::Unflatten<mochi::Int3 const>(mochi::MakeConstSpan(modelData.mesh->connectivity));
    } else {
      return false;
    }
    positions.reserve(surfaceNodes.size() * 3);
    for (auto const& node : surfaceNodes) {
      auto const pos = spaceConverter.TranslationToOutput(StaticCast<mochi::Float3>(node));
      positions.push_back(pos[0]);
      positions.push_back(pos[1]);
      positions.push_back(pos[2]);
    }

    indices.reserve(surfaceTris.size() * 3);
    for (auto const& tri : surfaceTris) {
      indices.push_back(tri[0]);
      indices.push_back(tri[1]);
      indices.push_back(tri[2]);
    }

    std::vector<float> faceNormals;
    faceNormals.reserve(surfaceTris.size() * 3);
    for (auto const& tri : surfaceTris) {
      auto const v0 = StaticCast<mochi::Float3>(surfaceNodes[tri[0]]);
      auto const v1 = StaticCast<mochi::Float3>(surfaceNodes[tri[1]]);
      auto const v2 = StaticCast<mochi::Float3>(surfaceNodes[tri[2]]);
      auto const e1 = v1 - v0;
      auto const e2 = v2 - v0;
      auto n = mochi::Cross(e1, e2);
      float const len = mochi::Norm(n);
      if (len > 0) {
        n = n / len;
      }
      auto const norm = spaceConverter.DirectionToOutput(n);
      faceNormals.push_back(norm[0]);
      faceNormals.push_back(norm[1]);
      faceNormals.push_back(norm[2]);
    }
    ComputeVertexNormalsAngleWeighted(positions, faceNormals, indices, vertexNormals);
  } else {
    ProceduralMesh procMesh;
    if (modelData.box) {
      procMesh = GenerateBoxMesh(*modelData.box);
    } else if (modelData.plane) {
      procMesh = GeneratePlaneMesh(*modelData.plane);
    } else if (modelData.sphere) {
      procMesh = GenerateSphereMesh(*modelData.sphere);
    } else {
      return false;
    }
    spaceConverter.TranslationsToOutput(mochi::MakeSpan(procMesh.positions), mochi::ErrorAssert{});
    spaceConverter.DirectionsToOutput(mochi::MakeSpan(procMesh.normals), mochi::ErrorAssert{});
    positions = std::move(procMesh.positions);
    vertexNormals = std::move(procMesh.normals);
    indices = std::move(procMesh.indices);
  }

  if (positions.empty() || indices.empty() || positions.size() % 3 != 0 ||
      vertexNormals.size() != positions.size()) {
    return false;
  }
  return true;
}

bool BuildMochiPolylineGeometry(
    mochi::MeshDataView const& mesh,
    mochi::CoordinateSpaceConverter const* converter,
    std::vector<float>& positions,
    std::vector<int>& segmentIndices) {
  int const numNodes = mesh.GetNumNodes();
  if (mesh.nodesPerElement != 2 || numNodes < 2) {
    return false;
  }
  mochi::CoordinateSpaceConverter const defaultConverter(
      mochi::CoordinateSpace::Default(), RenderSpace());
  mochi::CoordinateSpaceConverter const& spaceConverter = converter ? *converter : defaultConverter;

  positions.clear();
  positions.reserve(mesh.coordinates.size());
  for (auto const& node : mochi::Unflatten<mochi::Real3 const>(mesh.coordinates)) {
    auto const pos = spaceConverter.TranslationToOutput(StaticCast<mochi::Float3>(node));
    positions.push_back(pos[0]);
    positions.push_back(pos[1]);
    positions.push_back(pos[2]);
  }
  mochi::DynamicArray<int> const segments =
      mochi::MakeSequentialPolylineConnectivity(numNodes, mochi::IsPolylineClosedLoop(mesh));
  segmentIndices.assign(segments.begin(), segments.end());
  return true;
}

} // namespace mochi_renderer
