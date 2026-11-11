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
#include <mochi_mesh/tetrahedralization.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>

namespace nb = nanobind;

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

using RealArray = nb::ndarray<nb::numpy, real const, nb::c_contig, nb::device::cpu>;
using IntArray = nb::ndarray<nb::numpy, int const, nb::c_contig, nb::device::cpu>;

MeshDataView MeshDataViewFromNumpy(RealArray const& vertices, IntArray const& faces) {
  if (vertices.ndim() != 2 || vertices.shape(1) != 3) {
    throw std::invalid_argument("vertices must be an (N, 3) array");
  }
  if (faces.ndim() != 2 || faces.shape(1) != 3) {
    throw std::invalid_argument("faces must be an (M, 3) array");
  }
  constexpr auto kMaxTriangleRows = static_cast<size_t>(std::numeric_limits<int>::max() / 3);
  if (vertices.shape(0) > kMaxTriangleRows) {
    throw std::invalid_argument("vertices array is too large");
  }
  if (faces.shape(0) > kMaxTriangleRows) {
    throw std::invalid_argument("faces array is too large");
  }

  MeshDataView mesh;
  mesh.nodesPerElement = 3;
  mesh.coordinates = Span<real const>(vertices.data(), vertices.size());
  mesh.connectivity = Span<int const>(faces.data(), faces.size());
  return mesh;
}

template <typename T>
nb::object MakeOwningNumpy2D(T const* source, size_t rows, size_t columns) {
  size_t const count = rows * columns;
  auto data = std::make_unique_for_overwrite<T[]>(count == 0 ? 1 : count);
  if (count != 0) {
    std::copy(source, source + count, data.get());
  }
  T* dataPtr = data.get();
  nb::capsule owner(dataPtr, [](void* pointer) noexcept { delete[] static_cast<T*>(pointer); });
  // The capsule owns dataPtr after successful construction.
  // @lint-ignore CLANGTIDY facebook-hte-UnassignedReleasedUniquePointer
  data.release();
  nb::ndarray<nb::numpy, T, nb::ndim<2>, nb::c_contig> array(dataPtr, {rows, columns}, owner);
  return nb::cast(std::move(array));
}

nb::tuple MeshDataToNumpy(MeshData const& mesh) {
  int const numVerts = mesh.GetNumNodes();
  int const numElems = mesh.GetNumElements();
  return nb::make_tuple(
      MakeOwningNumpy2D(mesh.coordinates.data(), StaticCast<size_t>(numVerts), 3),
      MakeOwningNumpy2D(
          mesh.connectivity.data(),
          StaticCast<size_t>(numElems),
          StaticCast<size_t>(mesh.nodesPerElement)));
}

} // namespace

NB_MODULE(MODULE_NAME, m) {
  nb::set_leak_warnings(false);
  m.doc() =
      "Mochi mesh processing operations (heavy geometry runs in the superdex_mesh_cli helper)";

  // Register the C++ -> Python exception translator for MochiMeshErrorException (surfaced in
  // Python as mochi_mesh.Error). The returned handle is the registration object; retain it in a
  // named variable so the registering call is not mistaken for a discarded temporary. It lives
  // for the module's lifetime.
  [[maybe_unused]] auto meshErrorException =
      nb::exception<MochiMeshErrorException>(m, "Error", PyExc_RuntimeError);

  nb::enum_<RemeshMethod>(m, "RemeshMethod", nb::is_arithmetic())
      .value("NONE", RemeshMethod::None)
      .value("ALPHA_WRAP", RemeshMethod::AlphaWrap)
      .value("ACVD", RemeshMethod::ACVD)
      .value("SURFACE_DELAUNAY", RemeshMethod::SurfaceDelaunay)
      .export_values();

  // Surface remeshing
  nb::class_<SurfaceRemeshingParams>(m, "SurfaceRemeshingParams")
      .def(nb::init<>())
      .def_rw("edge_size", &SurfaceRemeshingParams::edgeSize)
      .def_rw("detect_features", &SurfaceRemeshingParams::detectFeatures)
      .def_rw("relative_to_mesh_size", &SurfaceRemeshingParams::relativeToMeshSize)
      .def_rw("alpha_wrap_relative_alpha", &SurfaceRemeshingParams::alphaWrapRelativeAlpha)
      .def_rw("alpha_wrap_relative_offset", &SurfaceRemeshingParams::alphaWrapRelativeOffset)
      .def_rw("smoothing_iterations", &SurfaceRemeshingParams::smoothingIterations)
      .def_rw("angle_smoothing_iterations", &SurfaceRemeshingParams::angleSmoothingIterations)
      .def_rw("sharp_feature_angle", &SurfaceRemeshingParams::sharpFeatureAngle)
      .def_rw("protect_constraints", &SurfaceRemeshingParams::protectConstraints)
      .def_rw("relax_constraints", &SurfaceRemeshingParams::relaxConstraints)
      .def_rw("use_adaptive_sizing", &SurfaceRemeshingParams::useAdaptiveSizing)
      .def_rw("adaptive_sizing_tolerance", &SurfaceRemeshingParams::adaptiveSizingTolerance)
      .def_rw("min_edge_size_factor", &SurfaceRemeshingParams::minEdgeSizeFactor)
      .def_rw("max_edge_size_factor", &SurfaceRemeshingParams::maxEdgeSizeFactor)
      .def_rw("repair_mesh", &SurfaceRemeshingParams::repairMesh)
      .def_rw("method", &SurfaceRemeshingParams::method)
      .def_rw(
          "relaxation_steps_per_iteration", &SurfaceRemeshingParams::relaxationStepsPerIteration)
      .def_rw(
          "tangential_relaxation_iterations",
          &SurfaceRemeshingParams::tangentialRelaxationIterations)
      .def_rw("target_vertex_count", &SurfaceRemeshingParams::targetVertexCount)
      .def_rw("acvd_gradation_factor", &SurfaceRemeshingParams::acvdGradationFactor)
      .def_rw("facet_angle_bound", &SurfaceRemeshingParams::facetAngleBound)
      .def_rw("facet_distance_bound", &SurfaceRemeshingParams::facetDistanceBound);

  m.def(
      "remesh_surface",
      [](RealArray const& vertices, IntArray const& faces, SurfaceRemeshingParams const& params) {
        MeshDataView const inputMesh = MeshDataViewFromNumpy(vertices, faces);
        Error error;
        MeshData result = RemeshSurface(inputMesh, params, error);
        if (!error.IsOK()) {
          throw MochiMeshErrorException(error);
        }
        return MeshDataToNumpy(result);
      },
      nb::arg("vertices"),
      nb::arg("faces"),
      nb::arg("params") = SurfaceRemeshingParams{},
      "Remesh a triangular surface mesh.\n\n"
      "Args:\n"
      "    vertices: (N, 3) array of vertex coordinates\n"
      "    faces: (M, 3) array of triangle vertex indices\n"
      "    params: SurfaceRemeshingParams\n\n"
      "Returns:\n"
      "    Tuple of (vertices, faces) numpy arrays");

  // Tetrahedralization
  nb::class_<TetrahedralizationParams>(m, "TetrahedralizationParams")
      .def(nb::init<>())
      .def_rw("edge_size", &TetrahedralizationParams::edgeSize)
      .def_rw("relative_to_mesh_size", &TetrahedralizationParams::relativeToMeshSize)
      .def_rw("facet_distance_bound", &TetrahedralizationParams::facetDistanceBound)
      .def_rw("facet_angle_bound", &TetrahedralizationParams::facetAngleBound)
      .def_rw("cell_radius_edge_ratio", &TetrahedralizationParams::cellRadiusEdgeRatio)
      .def_rw("detect_features", &TetrahedralizationParams::detectFeatures)
      .def_rw("sharp_feature_angle", &TetrahedralizationParams::sharpFeatureAngle)
      .def_rw("perturb", &TetrahedralizationParams::perturb)
      .def_rw("exude", &TetrahedralizationParams::exude)
      .def_rw("odt", &TetrahedralizationParams::odt);

  m.def(
      "tetrahedralize",
      [](RealArray const& vertices, IntArray const& faces, TetrahedralizationParams const& params) {
        MeshDataView const inputMesh = MeshDataViewFromNumpy(vertices, faces);
        Error error;
        MeshData result = Tetrahedralize(inputMesh, params, error);
        if (!error.IsOK()) {
          throw MochiMeshErrorException(error);
        }
        return MeshDataToNumpy(result);
      },
      nb::arg("vertices"),
      nb::arg("faces"),
      nb::arg("params") = TetrahedralizationParams{},
      "Generate a tetrahedral volume mesh from a closed triangle surface.\n\n"
      "Args:\n"
      "    vertices: (N, 3) array of vertex coordinates\n"
      "    faces: (M, 3) array of triangle vertex indices\n"
      "    params: TetrahedralizationParams\n\n"
      "Returns:\n"
      "    Tuple of (vertices, tetrahedra) numpy arrays");

  nb::class_<ConstrainedTetrahedralizationParams>(m, "ConstrainedTetrahedralizationParams")
      .def(nb::init<>())
      .def_rw("edge_size", &ConstrainedTetrahedralizationParams::edgeSize)
      .def_rw("relative_to_mesh_size", &ConstrainedTetrahedralizationParams::relativeToMeshSize)
      .def_rw("iterations", &ConstrainedTetrahedralizationParams::iterations)
      .def_rw("strict", &ConstrainedTetrahedralizationParams::strict);

  m.def(
      "tetrahedralize_constrained",
      [](RealArray const& vertices,
         IntArray const& faces,
         ConstrainedTetrahedralizationParams const& params) {
        MeshDataView const inputMesh = MeshDataViewFromNumpy(vertices, faces);
        Error error;
        MeshData result = TetrahedralizeConstrained(inputMesh, params, error);
        if (!error.IsOK()) {
          throw MochiMeshErrorException(error);
        }
        return MeshDataToNumpy(result);
      },
      nb::arg("vertices"),
      nb::arg("faces"),
      nb::arg("params") = ConstrainedTetrahedralizationParams{},
      "Generate a tetrahedral volume mesh whose boundary is a closed triangle surface.\n\n"
      "The first output vertices are the input vertices, in input order.\n\n"
      "Args:\n"
      "    vertices: (N, 3) array of vertex coordinates\n"
      "    faces: (M, 3) array of triangle vertex indices\n"
      "    params: ConstrainedTetrahedralizationParams\n\n"
      "Returns:\n"
      "    Tuple of (vertices, tetrahedra) numpy arrays");

  // Mesh statistics
  nb::class_<DistributionStatistics>(m, "DistributionStatistics")
      .def_ro("mean", &DistributionStatistics::mean)
      .def_ro("standard_deviation", &DistributionStatistics::standardDeviation)
      .def_ro("min", &DistributionStatistics::min)
      .def_ro("max", &DistributionStatistics::max);

  nb::class_<MeshStatistics>(m, "MeshStatistics")
      .def_ro("num_vertices", &MeshStatistics::numVertices)
      .def_ro("num_faces", &MeshStatistics::numFaces)
      .def_ro("edge_lengths", &MeshStatistics::edgeLengths)
      .def_ro("angles", &MeshStatistics::angles)
      .def_ro("hausdorff_distance", &MeshStatistics::hausdorffDistance)
      .def_ro("is_closed", &MeshStatistics::isClosed);

  m.def(
      "compute_mesh_statistics",
      [](RealArray const& vertices,
         IntArray const& faces,
         std::optional<RealArray> const& refVertices,
         std::optional<IntArray> const& refFaces) {
        MeshDataView const mesh = MeshDataViewFromNumpy(vertices, faces);
        Error error;
        MeshStatistics stats;
        if (refVertices.has_value() != refFaces.has_value()) {
          throw std::invalid_argument("Both ref_vertices and ref_faces must be provided together.");
        }
        if (refVertices.has_value() && refFaces.has_value()) {
          MeshDataView const refMesh = MeshDataViewFromNumpy(*refVertices, *refFaces);
          stats = ComputeMeshStatistics(mesh, &refMesh, error);
        } else {
          stats = ComputeMeshStatistics(mesh, nullptr, error);
        }
        if (!error.IsOK()) {
          throw MochiMeshErrorException(error);
        }
        return stats;
      },
      nb::arg("vertices"),
      nb::arg("faces"),
      nb::arg("ref_vertices") = nb::none(),
      nb::arg("ref_faces") = nb::none(),
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
      [](IntArray const& dims,
         RealArray const& values,
         RealArray const& boundsMin,
         RealArray const& boundsMax) {
        if (dims.ndim() != 1 || dims.shape(0) != 3) {
          throw std::invalid_argument("dims must be a (3,) integer array");
        }
        if (boundsMin.ndim() != 1 || boundsMin.shape(0) != 3) {
          throw std::invalid_argument("bounds_min must be a (3,) float array");
        }
        if (boundsMax.ndim() != 1 || boundsMax.shape(0) != 3) {
          throw std::invalid_argument("bounds_max must be a (3,) float array");
        }
        if (values.ndim() != 1) {
          throw std::invalid_argument("values must be a flat 1D float array");
        }

        auto const* dimsPtr = dims.data();
        int64_t expectedValueCount = 1;
        for (int i = 0; i < 3; ++i) {
          if (dimsPtr[i] <= 0) {
            throw std::invalid_argument("dims entries must be positive");
          }
          if (expectedValueCount > std::numeric_limits<int>::max() / dimsPtr[i]) {
            throw std::invalid_argument("dims product exceeds maximum supported size (INT_MAX)");
          }
          expectedValueCount *= dimsPtr[i];
        }
        if (static_cast<int64_t>(values.size()) != expectedValueCount) {
          throw std::invalid_argument("values must have size dims[0] * dims[1] * dims[2]");
        }

        auto const* valuesPtr = values.data();
        auto const* bMinPtr = boundsMin.data();
        auto const* bMaxPtr = boundsMax.data();

        GridSdfDataView sdfView;
        sdfView.dims = {dimsPtr[0], dimsPtr[1], dimsPtr[2]};
        sdfView.values = Span<real const>(valuesPtr, StaticCast<int>(values.size()));
        sdfView.bounds = Aabb(
            Real3{bMinPtr[0], bMinPtr[1], bMinPtr[2]}, Real3{bMaxPtr[0], bMaxPtr[1], bMaxPtr[2]});

        Error error;
        MeshData result = ReconstructSurfaceFromSdf(sdfView, error);
        if (!error.IsOK()) {
          throw MochiMeshErrorException(error);
        }
        return MeshDataToNumpy(result);
      },
      nb::arg("dims"),
      nb::arg("values"),
      nb::arg("bounds_min"),
      nb::arg("bounds_max"),
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
