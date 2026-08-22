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

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>

#include <functional>
#include <numeric>
#include <vector>

using namespace mochi;

namespace mochi_benchmark {

template <typename T>
T GetExample();

template <>
Sphere GetExample<Sphere>() {
  return {Vec4r{1.0_r, 2.0_r, 3.0_r}, 2.0_r};
}

template <>
Aabb GetExample<Aabb>() {
  return {Vec4r{-1.0_r, -1.0_r, -1.0_r}, Vec4r{2.0_r, 2.0_r, 2.0_r}};
}

template <>
Plane GetExample<Plane>() {
  return {Vec4r{0.0_r, 1.0_r, 0.0_r}, 0.0_r};
}

template <>
Obb GetExample<Obb>() {
  return {TransformRT{Real3{0.0_r, 0.5_r, 0.0_r}}, Real3{1.0_r, 2.0_r, 3.0_r}};
}

template <typename ObjectT, typename TargetT, int kBatchSize>
static void HasOverlap(benchmark::State& state) {
  std::vector<ObjectT> objects(kBatchSize, GetExample<ObjectT>());
  TargetT target = GetExample<TargetT>();
  bool results[kBatchSize] = {false};

  std::function<void()> fn = [&]() {
    for (int i = 0; i < kBatchSize; i++) {
      results[i] = HasOverlap(objects[i], target);
    }
  };

  for (auto _ : state) {
    CallNoInline(fn);
  }

  state.counters["Checks/second"] =
      benchmark::Counter(state.iterations() * kBatchSize, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(results);
}

// NOTE: These benchmarks measure performance of overlap tests with fixed geometries. Since the
// tested functions have no conditional branches based on geometry configuration, we don't need
// separate benchmarks for intersecting vs non-intersecting cases.
// NOTE: Timings include function call overhead.
BENCHMARK_TEMPLATE(HasOverlap, Sphere, Sphere, 1)
    ->Name("Geometry/HasOverlap/SphereVsSphere/Batch1");
BENCHMARK_TEMPLATE(HasOverlap, Sphere, Sphere, 16)
    ->Name("Geometry/HasOverlap/SphereVsSphere/Batch16");
BENCHMARK_TEMPLATE(HasOverlap, Sphere, Aabb, 1)->Name("Geometry/HasOverlap/SphereVsAabb/Batch1");
BENCHMARK_TEMPLATE(HasOverlap, Sphere, Aabb, 16)->Name("Geometry/HasOverlap/SphereVsAabb/Batch16");
BENCHMARK_TEMPLATE(HasOverlap, Sphere, Obb, 1)->Name("Geometry/HasOverlap/SphereVsObb/Batch1");
BENCHMARK_TEMPLATE(HasOverlap, Sphere, Obb, 16)->Name("Geometry/HasOverlap/SphereVsObb/Batch16");
BENCHMARK_TEMPLATE(HasOverlap, Sphere, Plane, 1)->Name("Geometry/HasOverlap/SphereVsPlane/Batch1");
BENCHMARK_TEMPLATE(HasOverlap, Sphere, Plane, 16)
    ->Name("Geometry/HasOverlap/SphereVsPlane/Batch16");
BENCHMARK_TEMPLATE(HasOverlap, Aabb, Aabb, 1)->Name("Geometry/HasOverlap/AabbVsAabb/Batch1");
BENCHMARK_TEMPLATE(HasOverlap, Aabb, Aabb, 16)->Name("Geometry/HasOverlap/AabbVsAabb/Batch16");
BENCHMARK_TEMPLATE(HasOverlap, Aabb, Sphere, 1)->Name("Geometry/HasOverlap/AabbVsSphere/Batch1");
BENCHMARK_TEMPLATE(HasOverlap, Aabb, Sphere, 16)->Name("Geometry/HasOverlap/AabbVsSphere/Batch16");
BENCHMARK_TEMPLATE(HasOverlap, Aabb, Obb, 1)->Name("Geometry/HasOverlap/AabbVsObb/Batch1");
BENCHMARK_TEMPLATE(HasOverlap, Aabb, Obb, 16)->Name("Geometry/HasOverlap/AabbVsObb/Batch16");
BENCHMARK_TEMPLATE(HasOverlap, Aabb, Plane, 1)->Name("Geometry/HasOverlap/AabbVsPlane/Batch1");
BENCHMARK_TEMPLATE(HasOverlap, Aabb, Plane, 16)->Name("Geometry/HasOverlap/AabbVsPlane/Batch16");
BENCHMARK_TEMPLATE(HasOverlap, Obb, Obb, 1)->Name("Geometry/HasOverlap/ObbVsObb/Batch1");
BENCHMARK_TEMPLATE(HasOverlap, Obb, Obb, 16)->Name("Geometry/HasOverlap/ObbVsObb/Batch16");
BENCHMARK_TEMPLATE(HasOverlap, Obb, Plane, 1)->Name("Geometry/HasOverlap/ObbVsPlane/Batch1");
BENCHMARK_TEMPLATE(HasOverlap, Obb, Plane, 16)->Name("Geometry/HasOverlap/ObbVsPlane/Batch16");

static void CalcAabb(benchmark::State& state, size_t numPoints) {
  std::vector<Real3> points(numPoints);
  auto pointsSpan = MakeConstSpan(points);
  Aabb aabb = {};
  for (auto _ : state) {
    aabb = CalcAabb(pointsSpan);
  }
  benchmark::DoNotOptimize(aabb);
}

BENCHMARK_CAPTURE(CalcAabb, 10, 10);
BENCHMARK_CAPTURE(CalcAabb, 100, 100);
BENCHMARK_CAPTURE(CalcAabb, 1000, 1000);
BENCHMARK_CAPTURE(CalcAabb, 10000, 10000);
BENCHMARK_CAPTURE(CalcAabb, 100000, 100000);

static void CalcAabbWithDisplacements(benchmark::State& state, size_t numPoints) {
  std::vector<Real3> points(numPoints);
  std::vector<Real3> displacements(numPoints);
  auto pointsSpan = MakeConstSpan(points);
  auto displacementsSpan = MakeConstSpan(displacements);
  Aabb aabb = {};
  for (auto _ : state) {
    aabb = CalcAabbWithDisplacements(pointsSpan, displacementsSpan);
  }
  benchmark::DoNotOptimize(aabb);
}

BENCHMARK_CAPTURE(CalcAabbWithDisplacements, 10, 10);
BENCHMARK_CAPTURE(CalcAabbWithDisplacements, 100, 100);
BENCHMARK_CAPTURE(CalcAabbWithDisplacements, 1000, 1000);
BENCHMARK_CAPTURE(CalcAabbWithDisplacements, 10000, 10000);
BENCHMARK_CAPTURE(CalcAabbWithDisplacements, 100000, 100000);

static void CalcAabbWithSortedIndices(benchmark::State& state, std::string const& meshPath) {
  // This overload of CalcAabb takes points, displacements, and indices. In practice, it is used to
  // find the bounds of a deformed soft actor. We load a real mesh to ensure a realistic
  // distribution of boundary indices.
  auto mesh = LoadTetrahedralMesh(GetAssetPath(meshPath), ErrorAssert{});
  auto points = mesh->GetNodeCoordinates();
  auto indices = mesh->GetBoundaryNodes();
  std::vector<Real3> displacements(points.size());
  auto displacementsSpan = MakeConstSpan(displacements);

  Aabb aabb = {};
  for (auto _ : state) {
    aabb = CalcAabbWithSortedIndices(points, displacementsSpan, indices);
  }
  benchmark::DoNotOptimize(aabb);
}

// clang-format off
BENCHMARK_CAPTURE(CalcAabbWithSortedIndices, icosphere_3subdiv, "sphere/icosphere_3subdiv.1.mochi.json");
BENCHMARK_CAPTURE(CalcAabbWithSortedIndices, icosphere_4subdiv, "sphere/icosphere_4subdiv.1.mochi.json");
// The 5-subdivision sphere is not shipped externally.
#if MOCHI_INTERNAL
BENCHMARK_CAPTURE(CalcAabbWithSortedIndices, icosphere_5subdiv, "sphere/icosphere_5subdiv.1.mochi.json");
#endif
// clang-format on

} // namespace mochi_benchmark
