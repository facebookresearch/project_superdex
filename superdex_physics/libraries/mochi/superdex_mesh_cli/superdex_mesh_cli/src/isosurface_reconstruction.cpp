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

#include "cgal_mesh_utils.h"
#include "mesh_cli_geometry.h"

#include <CGAL/Isosurfacing_3/Cartesian_grid_3.h>
#include <CGAL/Isosurfacing_3/Interpolated_discrete_values_3.h>
#include <CGAL/Isosurfacing_3/Marching_cubes_domain_3.h>
#include <CGAL/Isosurfacing_3/marching_cubes_3.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <span>
#include <vector>

using namespace mochi::mesh::cli;
using K = cgal_utils::K;

static bool IsAllFinite(std::span<double const> values) {
  return std::all_of(
      values.begin(), values.end(), [](double value) { return std::isfinite(value); });
}

MeshData mochi::mesh::cli::ReconstructSurfaceFromSdf(
    ScalarField3d const& sdfData,
    CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});

  // Validate dimensions
  MOCHI_MESH_CLI_ERROR_IF(
      sdfData.dims[0] < 2 || sdfData.dims[1] < 2 || sdfData.dims[2] < 2,
      error,
      "SDF grid dimensions must all be >= 2.");
  int64_t const expectedSize = int64_t{sdfData.dims[0]} * sdfData.dims[1] * sdfData.dims[2];
  MOCHI_MESH_CLI_ERROR_IF(
      static_cast<int64_t>(sdfData.values.size()) != expectedSize,
      error,
      "SDF values size does not match dims[0]*dims[1]*dims[2].");

  // Validate bounds
  Vector3d const bMin = sdfData.boundsMin;
  Vector3d const bMax = sdfData.boundsMax;
  MOCHI_MESH_CLI_ERROR_IF(
      !std::isfinite(bMin[0]) || !std::isfinite(bMin[1]) || !std::isfinite(bMin[2]) ||
          !std::isfinite(bMax[0]) || !std::isfinite(bMax[1]) || !std::isfinite(bMax[2]),
      error,
      "SDF bounds contain non-finite values.");
  MOCHI_MESH_CLI_ERROR_IF(
      bMin[0] >= bMax[0] || bMin[1] >= bMax[1] || bMin[2] >= bMax[2],
      error,
      "SDF bounds are degenerate (min >= max in at least one axis).");
  MOCHI_MESH_CLI_ERROR_IF(
      !IsAllFinite(sdfData.values), error, "SDF values must all be finite (no NaN or Inf).");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});

  try {
    using Grid = CGAL::Isosurfacing::Cartesian_grid_3<K>;
    using Values = CGAL::Isosurfacing::Interpolated_discrete_values_3<Grid>;

    K::Point_3 const pMin(
        static_cast<double>(bMin[0]), static_cast<double>(bMin[1]), static_cast<double>(bMin[2]));
    K::Point_3 const pMax(
        static_cast<double>(bMax[0]), static_cast<double>(bMax[1]), static_cast<double>(bMax[2]));
    K::Iso_cuboid_3 const bbox(pMin, pMax);

    int const xDim = sdfData.dims[0];
    int const yDim = sdfData.dims[1];
    int const zDim = sdfData.dims[2];

    Grid grid(bbox, CGAL::make_array<std::size_t>(xDim, yDim, zDim));
    Values values(grid);

    // Populate values: CGAL uses (i,j,k) with i=x, j=y, k=z.
    // Mochi stores values in x-slowest, z-fastest order: index = dims[1]*dims[2]*x + dims[2]*y + z
    for (int x = 0; x < xDim; ++x) {
      for (int y = 0; y < yDim; ++y) {
        for (int z = 0; z < zDim; ++z) {
          int const mochiIdx = zDim * yDim * x + zDim * y + z;
          values(x, y, z) = static_cast<double>(sdfData.values[mochiIdx]);
        }
      }
    }

    auto const domain = CGAL::Isosurfacing::create_marching_cubes_domain_3(grid, values);

    std::vector<K::Point_3> points;
    std::vector<std::vector<std::size_t>> triangles;
    CGAL::Isosurfacing::marching_cubes<CGAL::Sequential_tag>(domain, 0.0, points, triangles);

    MOCHI_MESH_CLI_ERROR_IF(
        points.empty() || triangles.empty(), error, "Marching Cubes produced no output.");
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});

    // Convert polygon soup to Surface_mesh
    cgal_utils::CgalSurfaceMesh surfaceMesh;
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, triangles, surfaceMesh);

    return cgal_utils::SurfaceMeshToMeshData(surfaceMesh);
  } catch (std::exception const& e) {
    MOCHI_MESH_CLI_LOG_WARNING("Isosurface reconstruction failed: %s", e.what());
    MOCHI_MESH_CLI_ERROR_SET(error, "Isosurface reconstruction failed with an exception.");
  } catch (...) {
    MOCHI_MESH_CLI_ERROR_SET(error, "Isosurface reconstruction failed with an unknown exception.");
  }

  return {};
}
