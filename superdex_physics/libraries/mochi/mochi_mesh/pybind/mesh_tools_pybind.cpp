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

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/error.h>
#include <mochi_mesh/isosurface_reconstruction.h>
#include <mochi_mesh/mesh_statistics.h>
#include <mochi_mesh/surface_remeshing.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>

namespace py = pybind11;

using namespace mochi;
using namespace mochi::mesh;

#if MOCHI_USE_DOUBLE_PRECISION
#define MODULE_NAME mochi_mesh_double
#else
#define MODULE_NAME mochi_mesh
#endif

namespace {

class MochiMeshErrorException : public std::runtime_error {
 public:
  explicit MochiMeshErrorException(Error const& e) : std::runtime_error(e.ToString()) {}
};

MeshData MeshDataFromNumpy(
    py::array_t<real, py::array::c_style | py::array::forcecast> const& vertices,
    py::array_t<int, py::array::c_style | py::array::forcecast> const& faces) {
  auto vBuf = vertices.request();
  auto fBuf = faces.request();

  if (vBuf.ndim != 2 || vBuf.shape[1] != 3) {
    throw std::invalid_argument("vertices must be an (N, 3) array");
  }
  if (fBuf.ndim != 2 || fBuf.shape[1] != 3) {
    throw std::invalid_argument("faces must be an (M, 3) array");
  }

  MeshData mesh;
  mesh.nodesPerElement = 3;

  int const numVerts = StaticCast<int>(vBuf.shape[0]);
  int const numFaces = StaticCast<int>(fBuf.shape[0]);

  mesh.coordinates.resize(numVerts * 3);
  auto const* vPtr = static_cast<real const*>(vBuf.ptr);
  for (int i = 0; i < numVerts * 3; ++i) {
    mesh.coordinates[i] = vPtr[i];
  }

  mesh.connectivity.resize(numFaces * 3);
  auto const* fPtr = static_cast<int const*>(fBuf.ptr);
  for (int i = 0; i < numFaces * 3; ++i) {
    mesh.connectivity[i] = fPtr[i];
  }

  return mesh;
}

std::tuple<py::array_t<real>, py::array_t<int>> MeshDataToNumpy(MeshData const& mesh) {
  int const numVerts = mesh.GetNumNodes();
  int const numElems = mesh.GetNumElements();

  py::array_t<real> vertices({numVerts, 3});
  auto vBuf = vertices.mutable_unchecked<2>();
  for (int i = 0; i < numVerts; ++i) {
    vBuf(i, 0) = mesh.coordinates[3 * i + 0];
    vBuf(i, 1) = mesh.coordinates[3 * i + 1];
    vBuf(i, 2) = mesh.coordinates[3 * i + 2];
  }

  py::array_t<int> elements({numElems, mesh.nodesPerElement});
  auto eBuf = elements.mutable_unchecked<2>();
  for (int i = 0; i < numElems; ++i) {
    for (int j = 0; j < mesh.nodesPerElement; ++j) {
      eBuf(i, j) = mesh.connectivity[i * mesh.nodesPerElement + j];
    }
  }

  return {vertices, elements};
}

} // namespace

PYBIND11_MODULE(MODULE_NAME, m) {
  m.doc() =
      "Mochi mesh processing operations (heavy geometry runs in the superdex_mesh_cli helper)";

  py::register_exception<MochiMeshErrorException>(m, "Error", PyExc_RuntimeError);

  py::enum_<RemeshMethod>(m, "RemeshMethod")
      .value("NONE", RemeshMethod::None)
      .value("ALPHA_WRAP", RemeshMethod::AlphaWrap)
      .value("ACVD", RemeshMethod::ACVD)
      .value("SURFACE_DELAUNAY", RemeshMethod::SurfaceDelaunay)
      .export_values();

  // Surface remeshing
  py::class_<SurfaceRemeshingParams>(m, "SurfaceRemeshingParams")
      .def(py::init<>())
      .def_readwrite("edge_size", &SurfaceRemeshingParams::edgeSize)
      .def_readwrite("detect_features", &SurfaceRemeshingParams::detectFeatures)
      .def_readwrite("relative_to_mesh_size", &SurfaceRemeshingParams::relativeToMeshSize)
      .def_readwrite("alpha_wrap_relative_alpha", &SurfaceRemeshingParams::alphaWrapRelativeAlpha)
      .def_readwrite("alpha_wrap_relative_offset", &SurfaceRemeshingParams::alphaWrapRelativeOffset)
      .def_readwrite("smoothing_iterations", &SurfaceRemeshingParams::smoothingIterations)
      .def_readwrite(
          "angle_smoothing_iterations", &SurfaceRemeshingParams::angleSmoothingIterations)
      .def_readwrite("sharp_feature_angle", &SurfaceRemeshingParams::sharpFeatureAngle)
      .def_readwrite("protect_constraints", &SurfaceRemeshingParams::protectConstraints)
      .def_readwrite("relax_constraints", &SurfaceRemeshingParams::relaxConstraints)
      .def_readwrite("use_adaptive_sizing", &SurfaceRemeshingParams::useAdaptiveSizing)
      .def_readwrite("adaptive_sizing_tolerance", &SurfaceRemeshingParams::adaptiveSizingTolerance)
      .def_readwrite("min_edge_size_factor", &SurfaceRemeshingParams::minEdgeSizeFactor)
      .def_readwrite("max_edge_size_factor", &SurfaceRemeshingParams::maxEdgeSizeFactor)
      .def_readwrite("repair_mesh", &SurfaceRemeshingParams::repairMesh)
      .def_readwrite("method", &SurfaceRemeshingParams::method)
      .def_readwrite(
          "relaxation_steps_per_iteration", &SurfaceRemeshingParams::relaxationStepsPerIteration)
      .def_readwrite(
          "tangential_relaxation_iterations",
          &SurfaceRemeshingParams::tangentialRelaxationIterations)
      .def_readwrite("target_vertex_count", &SurfaceRemeshingParams::targetVertexCount)
      .def_readwrite("acvd_gradation_factor", &SurfaceRemeshingParams::acvdGradationFactor)
      .def_readwrite("facet_angle_bound", &SurfaceRemeshingParams::facetAngleBound)
      .def_readwrite("facet_distance_bound", &SurfaceRemeshingParams::facetDistanceBound);

  m.def(
      "remesh_surface",
      [](py::array_t<real, py::array::c_style | py::array::forcecast> const& vertices,
         py::array_t<int, py::array::c_style | py::array::forcecast> const& faces,
         SurfaceRemeshingParams const& params) {
        MeshData inputMesh = MeshDataFromNumpy(vertices, faces);
        Error error;
        MeshData result = RemeshSurface(inputMesh, params, error);
        if (!error.IsOK()) {
          throw MochiMeshErrorException(error);
        }
        return MeshDataToNumpy(result);
      },
      py::arg("vertices"),
      py::arg("faces"),
      py::arg("params") = SurfaceRemeshingParams{},
      "Remesh a triangular surface mesh.\n\n"
      "Args:\n"
      "    vertices: (N, 3) array of vertex coordinates\n"
      "    faces: (M, 3) array of triangle vertex indices\n"
      "    params: SurfaceRemeshingParams\n\n"
      "Returns:\n"
      "    Tuple of (vertices, faces) numpy arrays");

  // Mesh statistics
  py::class_<DistributionStatistics>(m, "DistributionStatistics")
      .def_readonly("mean", &DistributionStatistics::mean)
      .def_readonly("standard_deviation", &DistributionStatistics::standardDeviation)
      .def_readonly("min", &DistributionStatistics::min)
      .def_readonly("max", &DistributionStatistics::max);

  py::class_<MeshStatistics>(m, "MeshStatistics")
      .def_readonly("num_vertices", &MeshStatistics::numVertices)
      .def_readonly("num_faces", &MeshStatistics::numFaces)
      .def_readonly("edge_lengths", &MeshStatistics::edgeLengths)
      .def_readonly("angles", &MeshStatistics::angles)
      .def_readonly("hausdorff_distance", &MeshStatistics::hausdorffDistance)
      .def_readonly("is_closed", &MeshStatistics::isClosed);

  m.def(
      "compute_mesh_statistics",
      [](py::array_t<real, py::array::c_style | py::array::forcecast> const& vertices,
         py::array_t<int, py::array::c_style | py::array::forcecast> const& faces,
         std::optional<py::array_t<real, py::array::c_style | py::array::forcecast>> const&
             refVertices,
         std::optional<py::array_t<int, py::array::c_style | py::array::forcecast>> const&
             refFaces) {
        MeshData mesh = MeshDataFromNumpy(vertices, faces);
        Error error;
        MeshStatistics stats;
        if (refVertices.has_value() != refFaces.has_value()) {
          throw std::invalid_argument("Both ref_vertices and ref_faces must be provided together.");
        }
        if (refVertices.has_value() && refFaces.has_value()) {
          MeshData refMesh = MeshDataFromNumpy(*refVertices, *refFaces);
          MeshDataView refView(refMesh);
          stats = ComputeMeshStatistics(mesh, &refView, error);
        } else {
          stats = ComputeMeshStatistics(mesh, nullptr, error);
        }
        if (!error.IsOK()) {
          throw MochiMeshErrorException(error);
        }
        return stats;
      },
      py::arg("vertices"),
      py::arg("faces"),
      py::arg("ref_vertices") = py::none(),
      py::arg("ref_faces") = py::none(),
      "Compute quality statistics for a triangle surface mesh.\n\n"
      "Args:\n"
      "    vertices: (N, 3) array of vertex coordinates\n"
      "    faces: (M, 3) array of triangle vertex indices\n"
      "    ref_vertices: Optional (N, 3) reference mesh vertices for Hausdorff distance\n"
      "    ref_faces: Optional (M, 3) reference mesh faces for Hausdorff distance\n\n"
      "Returns:\n"
      "    MeshStatistics object with edge_lengths, angles, and hausdorff_distance");

  // Isosurface reconstruction from SDF
  m.def(
      "reconstruct_surface_from_sdf",
      [](py::array_t<int, py::array::c_style | py::array::forcecast> const& dims,
         py::array_t<real, py::array::c_style | py::array::forcecast> const& values,
         py::array_t<real, py::array::c_style | py::array::forcecast> const& boundsMin,
         py::array_t<real, py::array::c_style | py::array::forcecast> const& boundsMax) {
        auto dimsBuf = dims.request();
        auto valuesBuf = values.request();
        auto bMinBuf = boundsMin.request();
        auto bMaxBuf = boundsMax.request();

        if (dimsBuf.ndim != 1 || dimsBuf.shape[0] != 3) {
          throw std::invalid_argument("dims must be a (3,) integer array");
        }
        if (bMinBuf.ndim != 1 || bMinBuf.shape[0] != 3) {
          throw std::invalid_argument("bounds_min must be a (3,) float array");
        }
        if (bMaxBuf.ndim != 1 || bMaxBuf.shape[0] != 3) {
          throw std::invalid_argument("bounds_max must be a (3,) float array");
        }
        if (valuesBuf.ndim != 1) {
          throw std::invalid_argument("values must be a flat 1D float array");
        }

        auto const* dimsPtr = static_cast<int const*>(dimsBuf.ptr);
        int64_t const expectedValueCount = int64_t{dimsPtr[0]} * dimsPtr[1] * dimsPtr[2];
        if (static_cast<int64_t>(valuesBuf.size) != expectedValueCount) {
          throw std::invalid_argument("values must have size dims[0] * dims[1] * dims[2]");
        }

        auto const* valuesPtr = static_cast<real const*>(valuesBuf.ptr);
        auto const* bMinPtr = static_cast<real const*>(bMinBuf.ptr);
        auto const* bMaxPtr = static_cast<real const*>(bMaxBuf.ptr);

        if (valuesBuf.size > std::numeric_limits<int>::max()) {
          throw std::invalid_argument("values array exceeds maximum supported size (INT_MAX).");
        }

        GridSdfDataView sdfView;
        sdfView.dims = {dimsPtr[0], dimsPtr[1], dimsPtr[2]};
        sdfView.values = Span<real const>(valuesPtr, StaticCast<int>(valuesBuf.size));
        sdfView.bounds = Aabb(
            Real3{bMinPtr[0], bMinPtr[1], bMinPtr[2]}, Real3{bMaxPtr[0], bMaxPtr[1], bMaxPtr[2]});

        Error error;
        MeshData result = ReconstructSurfaceFromSdf(sdfView, error);
        if (!error.IsOK()) {
          throw MochiMeshErrorException(error);
        }
        return MeshDataToNumpy(result);
      },
      py::arg("dims"),
      py::arg("values"),
      py::arg("bounds_min"),
      py::arg("bounds_max"),
      "Reconstruct a triangle mesh from a grid SDF using Marching Cubes.\n\n"
      "Args:\n"
      "    dims: (3,) integer array of grid dimensions [x, y, z]\n"
      "    values: Flat float array of SDF values in x-slowest, z-fastest order\n"
      "            (index = dims[1]*dims[2]*x + dims[2]*y + z, size = dims[0]*dims[1]*dims[2])\n"
      "    bounds_min: (3,) float array of grid minimum bounds\n"
      "    bounds_max: (3,) float array of grid maximum bounds\n\n"
      "Returns:\n"
      "    Tuple of (vertices, faces) numpy arrays");
}
