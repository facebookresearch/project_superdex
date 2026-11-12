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

#include "config.h"

#include <mochi_core/geometry/bvh_tree.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/grid_sdf.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/simd.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

using namespace mochi;
using namespace mochi_benchmark;

/****************************************************************************************
  Nearest-triangle queries on real meshes, single-threaded and one point at a time: a baseline for
  batched or parallel queries. Query points form a 32^3 grid over the mesh bounds, padded on every
  side by 10% of the longest axis. Coherent visits them in z-fastest grid order, as SDF baking does.
  Shuffled visits the same points in random order.
*/
static std::shared_ptr<TriangularMesh const> LoadSurfaceMesh(std::string const& meshPath) {
  auto const tetMesh = LoadTetrahedralMesh(GetAssetPath(meshPath), ErrorAssert{});
  return std::make_shared<TriangularMesh const>(CreateBoundaryMesh(*tetMesh));
}

static DynamicArray<Real3> MakeGridQueryPoints(Aabb const& bounds, bool shuffle) {
  constexpr int kPointsPerAxis = 32;
  Aabb const padded = ExpandShape(bounds, 0.1_r * Max(bounds.GetSize()));
  Real3 const step = padded.GetSize() / static_cast<real>(kPointsPerAxis - 1);
  DynamicArray<Real3> points;
  for (int x = 0; x < kPointsPerAxis; ++x) {
    for (int y = 0; y < kPointsPerAxis; ++y) {
      for (int z = 0; z < kPointsPerAxis; ++z) {
        points.push_back(padded.GetMin() + StaticCast<Real3>(Int3{x, y, z}) * step);
      }
    }
  }
  if (shuffle) {
    std::shuffle(points.begin(), points.end(), RandomGenerator(3));
  }
  return points;
}

static void BvhTreeFindClosest(benchmark::State& state, std::string const& meshPath, bool shuffle) {
  auto const mesh = LoadSurfaceMesh(meshPath);
  TriangularMeshBvhObject<Aabb> const object(mesh);
  AabbTree const tree(&object, BvhTreeParams{});
  auto const points = MakeGridQueryPoints(mesh->GetAabb(), shuffle);

  real sum = 0_r;
  for (auto _ : state) {
    for (auto const& p : points) {
      real distSqr = {};
      tree.VFindClosest(ToSimd(p), &distSqr);
      sum += distSqr;
    }
  }
  benchmark::DoNotOptimize(sum);
  state.counters["points/second"] =
      benchmark::Counter(state.iterations() * points.size(), benchmark::Counter::kIsRate);
  state.counters["triangles"] = mesh->GetNumFaces();

  // Work per query, counted in a separate untimed pass.
  BvhNearestQueryStats stats;
  for (auto const& p : points) {
    tree.FindClosestWithStats(ToSimd(p), stats);
  }
  auto const perPoint = [&](int64_t count) { return static_cast<double>(count) / points.size(); };
  state.counters["nodes/point"] = perPoint(stats.nodesVisited);
  state.counters["bvTests/point"] = perPoint(stats.bvTests);
  state.counters["triTests/point"] = perPoint(stats.elementTests);
}

// clang-format off
BENCHMARK_CAPTURE(BvhTreeFindClosest, cube/Coherent, "cube/cube_mesh.mochi.json", false);
BENCHMARK_CAPTURE(BvhTreeFindClosest, cube/Shuffled, "cube/cube_mesh.mochi.json", true);
#if MOCHI_INTERNAL // These meshes are not shipped externally.
BENCHMARK_CAPTURE(BvhTreeFindClosest, duck_coarse/Coherent, "duck/duck_coarse_mesh.mochi.json", false);
BENCHMARK_CAPTURE(BvhTreeFindClosest, duck_coarse/Shuffled, "duck/duck_coarse_mesh.mochi.json", true);
BENCHMARK_CAPTURE(BvhTreeFindClosest, armadillo/Coherent, "armadillo/armadillo_coarse_mesh.mochi.json", false);
BENCHMARK_CAPTURE(BvhTreeFindClosest, armadillo/Shuffled, "armadillo/armadillo_coarse_mesh.mochi.json", true);
BENCHMARK_CAPTURE(BvhTreeFindClosest, icosphere_5subdiv/Coherent, "sphere/icosphere_5subdiv.1.mochi.json", false);
BENCHMARK_CAPTURE(BvhTreeFindClosest, icosphere_5subdiv/Shuffled, "sphere/icosphere_5subdiv.1.mochi.json", true);
#endif
// clang-format on

/****************************************************************************************
  GridSdf construction from a mesh. The voxel size is at most the mesh's longest axis divided by
  numCells.
*/
static void GridSdfBuild(benchmark::State& state, std::string const& meshPath, int numCells) {
  auto const mesh = LoadSurfaceMesh(meshPath);
  GridSdfParams params;
  params.resolutionMode = GridSdfResolutionMode::LargestAxis;
  params.resolutionDelta = Real3{1_r, 1_r, 1_r} / static_cast<real>(numCells);

  Int3 cells{};
  for (auto _ : state) {
    GridSdf const sdf(mesh, params, ErrorAssert{});
    cells = sdf.GetCellResolution();
  }
  int64_t const numSamples = int64_t{cells[0] + 1} * (cells[1] + 1) * (cells[2] + 1);
  state.counters["samples"] = static_cast<double>(numSamples);
  state.counters["samples/second"] =
      benchmark::Counter(state.iterations() * numSamples, benchmark::Counter::kIsRate);
  state.counters["triangles"] = mesh->GetNumFaces();
}

// clang-format off
BENCHMARK_CAPTURE(GridSdfBuild, cube/32, "cube/cube_mesh.mochi.json", 32)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(GridSdfBuild, cube/64, "cube/cube_mesh.mochi.json", 64)->Unit(benchmark::kMillisecond);
#if MOCHI_INTERNAL // These meshes are not shipped externally.
BENCHMARK_CAPTURE(GridSdfBuild, duck_coarse/32, "duck/duck_coarse_mesh.mochi.json", 32)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(GridSdfBuild, duck_coarse/64, "duck/duck_coarse_mesh.mochi.json", 64)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(GridSdfBuild, armadillo/32, "armadillo/armadillo_coarse_mesh.mochi.json", 32)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(GridSdfBuild, armadillo/64, "armadillo/armadillo_coarse_mesh.mochi.json", 64)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(GridSdfBuild, icosphere_5subdiv/32, "sphere/icosphere_5subdiv.1.mochi.json", 32)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(GridSdfBuild, icosphere_5subdiv/64, "sphere/icosphere_5subdiv.1.mochi.json", 64)->Unit(benchmark::kMillisecond);
#endif
// clang-format on
