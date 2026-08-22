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

#include <mochi_mesh/mesh_statistics.h>

#include "mesh_cli_adapter.h"
#include "mesh_cli_client.h"

#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>

#include <mochi_mesh/mochi_mesh_cli_encoding.h>

#include <cmath>
#include <map>
#include <utility>
#include <vector>

// Mesh statistics are computed natively: vertex/face counts, edge-length and interior-angle
// distributions, and watertightness. Only the optional Hausdorff distance to a reference mesh is
// routed to the superdex_mesh_cli helper.

using namespace mochi;
using namespace mochi::mesh;

using Vec3d = NdArray<double, 3>;

static DistributionStatistics ComputeDistributionStatistics(Span<double const> values) {
  DistributionStatistics stats;
  if (values.empty()) {
    return stats;
  }

  auto const [min, max] = MinMax(values);
  stats.min = min;
  stats.max = max;
  stats.mean = HSum(values) / static_cast<double>(values.size());

  double sumSqDev = 0;
  size_t i = 0;
  auto const vMean = Broadcast<Vec4d>(stats.mean);
  Vec4d acc = {};
  for (; i + Vec4d::kSize <= values.size(); i += Vec4d::kSize) {
    auto const dev = Load<Vec4d>(&values[i]) - vMean;
    acc += dev * dev;
  }
  sumSqDev = HSum(acc);
  for (; i < values.size(); ++i) {
    double const dev = values[i] - stats.mean;
    sumSqDev += dev * dev;
  }
  stats.standardDeviation = std::sqrt(sumSqDev / static_cast<double>(values.size()));

  return stats;
}

static Vec3d VertexAt(MeshDataView const& mesh, int vertex) {
  return {
      static_cast<double>(mesh.coordinates[3 * vertex + 0]),
      static_cast<double>(mesh.coordinates[3 * vertex + 1]),
      static_cast<double>(mesh.coordinates[3 * vertex + 2])};
}

// Routes the approximate Hausdorff distance to the superdex_mesh_cli helper process.
static double
ComputeHausdorffViaCli(MeshDataView const& mesh, MeshDataView const& referenceMesh, Error& error) {
  MOCHI_ERROR_RETURN(error, -1.0);

  cli::PayloadWriter writer;
  {
    cli::MeshData const cliMesh = cli_adapter::ToCliMeshData(mesh, error);
    MOCHI_ERROR_RETURN(error, -1.0);
    writer.WriteMeshData(cliMesh);
  }
  {
    cli::MeshData const cliReferenceMesh = cli_adapter::ToCliMeshData(referenceMesh, error);
    MOCHI_ERROR_RETURN(error, -1.0);
    writer.WriteMeshData(cliReferenceMesh);
  }

  auto const response =
      InvokeMeshCli(cli::GeometryOp::ApproximateHausdorffDistance, writer.Bytes(), error);
  MOCHI_ERROR_RETURN(error, -1.0);

  cli::PayloadReader reader(response);
  double distance = -1.0;
  MOCHI_ERROR_IF(
      !reader.ReadDouble(distance) || !reader.AtEnd(),
      error,
      "Malformed ApproximateHausdorffDistance response from superdex_mesh_cli.");
  MOCHI_ERROR_RETURN(error, -1.0);
  return distance;
}

MeshStatistics mochi::mesh::ComputeMeshStatistics(
    MeshDataView const& mesh,
    MeshDataView const* referenceMesh,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  MOCHI_ERROR_IF(
      mesh.nodesPerElement != 3,
      error,
      "Input mesh must be a triangle mesh (nodesPerElement == 3).");
  MOCHI_ERROR_IF(mesh.coordinates.empty(), error, "Input mesh has no vertices.");
  MOCHI_ERROR_IF(mesh.connectivity.empty(), error, "Input mesh has no faces.");
  MOCHI_ERROR_RETURN(error, {});

  int const numVertices = mesh.GetNumNodes();
  int const numFaces = mesh.GetNumElements();

  // Validate connectivity bounds.
  for (int i = 0; i < isize(mesh.connectivity); ++i) {
    MOCHI_ERROR_IF(
        mesh.connectivity[i] < 0 || mesh.connectivity[i] >= numVertices,
        error,
        "Connectivity index out of bounds.");
  }
  MOCHI_ERROR_RETURN(error, {});

  MeshStatistics stats;
  stats.numVertices = numVertices;
  stats.numFaces = numFaces;

  // Count how many triangles share each undirected edge (used for both unique edges and the
  // watertightness check).
  std::map<std::pair<int, int>, int> edgeUseCount;
  for (int i = 0; i < numFaces; ++i) {
    int const tri[3] = {
        mesh.connectivity[3 * i + 0], mesh.connectivity[3 * i + 1], mesh.connectivity[3 * i + 2]};
    for (int e = 0; e < 3; ++e) {
      int const vi = tri[e];
      int const vj = tri[(e + 1) % 3];
      ++edgeUseCount[std::make_pair(std::min(vi, vj), std::max(vi, vj))];
    }
  }

  // Edge lengths over the unique edges.
  DynamicArray<double> edgeLengths;
  edgeLengths.reserve(edgeUseCount.size());
  for (auto const& [edge, count] : edgeUseCount) {
    edgeLengths.push_back(Norm(VertexAt(mesh, edge.second) - VertexAt(mesh, edge.first)));
  }
  stats.edgeLengths = ComputeDistributionStatistics(MakeConstSpan(edgeLengths));

  // Watertight iff there is at least one edge and every edge is shared by exactly two triangles.
  stats.isClosed = !edgeUseCount.empty();
  for (auto const& [edge, count] : edgeUseCount) {
    if (count != 2) {
      stats.isClosed = false;
      break;
    }
  }

  // Interior angles.
  double constexpr kRadToDeg = 180.0 / static_cast<double>(kPI);
  DynamicArray<double> angles;
  angles.reserve(3 * numFaces);
  for (int i = 0; i < numFaces; ++i) {
    Vec3d const p[3] = {
        VertexAt(mesh, mesh.connectivity[3 * i + 0]),
        VertexAt(mesh, mesh.connectivity[3 * i + 1]),
        VertexAt(mesh, mesh.connectivity[3 * i + 2])};
    for (int v = 0; v < 3; ++v) {
      Vec3d const e1 = p[(v + 1) % 3] - p[v];
      Vec3d const e2 = p[(v + 2) % 3] - p[v];
      angles.push_back(std::atan2(Norm(Cross(e1, e2)), Dot(e1, e2)) * kRadToDeg);
    }
  }
  stats.angles = ComputeDistributionStatistics(MakeConstSpan(angles));

  // Hausdorff distance to a reference mesh (routed to the helper) is only computed on
  // request.
  if (referenceMesh != nullptr) {
    MOCHI_ERROR_IF(
        referenceMesh->nodesPerElement != 3,
        error,
        "Reference mesh must be a triangle mesh (nodesPerElement == 3).");
    MOCHI_ERROR_IF(referenceMesh->coordinates.empty(), error, "Reference mesh has no vertices.");
    MOCHI_ERROR_IF(referenceMesh->connectivity.empty(), error, "Reference mesh has no faces.");
    MOCHI_ERROR_RETURN(error, {});

    stats.hausdorffDistance = ComputeHausdorffViaCli(mesh, *referenceMesh, error);
    MOCHI_ERROR_RETURN(error, {});
  }

  return stats;
}
