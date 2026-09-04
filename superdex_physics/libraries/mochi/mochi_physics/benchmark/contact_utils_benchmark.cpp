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

#include <mochi_core/contact/contact_utils.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/sdf_bv.h>
#include <mochi_core/geometry/sphere_tree.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_physics/mochi_physics.h>

// src includes for registry access
#include <mochi_physics/src/mochi_contact.h>
#include <mochi_physics/src/mochi_ecs_utils.h>
#include <mochi_physics/src/mochi_scene.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

using namespace mochi;
using namespace mochi_benchmark;

// Most benchmarks use the 5-subdivision sphere that is not shipped externally.
#if MOCHI_INTERNAL

// Mesh used to generate sample points and for mesh & sdf colliders.
static constexpr std::string_view kMeshPath = "sphere/icosphere_5subdiv.1.mochi.json";

// Number of random points for benchmarks. Should match the number of samples points in kMeshPath,
// if you want to compare random points vs mesh points.
static constexpr int kNumRandPoints = 15360;

// Sample points are not usually random. They follow the surface mesh and often have neighboring
// points are often near each other spatially. To ensure a realistic distribution, we create a real
// actor and then extract CContactSamples from the registry.
static Span<Real3 const> GetSamplesPositionsFromMesh(
    Context* context,
    std::string_view meshPath,
    MeshCollider const** outMeshCollider = nullptr, // Optionally return a mesh collider
    SdfCollider const** outSdfCollider = nullptr // Optionally return an SDF collider
) {
  auto shape = context->LoadShapeFromFile(
      mochi_benchmark::GetAssetPath(std::string(meshPath)), ErrorAssert{});

  // This scene will continue living beyond the scope of the function.
  // That's OK. It keeps the data alive until the end of the benchmark.
  auto* scene = assert_cast<SceneImpl*>(context->CreateScene("BenchmarkScene"));

  auto colliderType = ColliderType::None;
  if (outMeshCollider) {
    colliderType = ColliderType::Mesh;
  } else if (outSdfCollider) {
    colliderType = ColliderType::Sdf;
  }

  auto& reg = scene->GetRegistry();
  auto* actor = scene->CreateRigidActor(
      RigidActorParams{.shape = shape, .colliderType = colliderType}, ErrorAssert{});
  entt::entity e = GetEntity(reg, actor->GetHandle(), ErrorAssert{});

  if (outMeshCollider) {
    *outMeshCollider = &reg.get<CMeshCollider const>(e);
  } else if (outSdfCollider) {
    auto& sdfCollider = reg.get<CSdfCollider>(e);
    if (!sdfCollider.shape) {
      auto const& gridSdfShape = reg.get<CSdfColliderPending const>(e).gridSdfShape;
      gridSdfShape->GetGridSdfSemaphore().Wait();
      sdfCollider.shape = gridSdfShape->GetGridSdf();
    }
    *outSdfCollider = &sdfCollider;
  }

  auto const& contactSamples = reg.get<CContactSamples<TimeStep::Current> const>(e);
  return contactSamples.positions;
}

template <FindPointContactsCollider ColliderT>
static void BenchmarkFindPointContact(
    benchmark::State& state,
    ColliderT const* collider,
    Span<Real3 const> positions,
    real percentVolumeOverlap,
    bool expectAllContactsFor100PercentOverlap = true) {
  // Use a non-identity transform in case identity receives special treatment.
  TransformRT pointsFromCollider =
      TransformRT{Quaternion::RotationX(kPI * 0.25_r), Real3{0.1_r, 0.2_r, 0.3_r}}; // Not identity
  DynamicArray<Real3> positionsTransformed;
  positionsTransformed.resize_noinit(positions.size());
  ArrayTransformPoints(MakeSpan(positionsTransformed), positions, pointsFromCollider);

  ContactDetectionParams cdParams;

  // Results go here
  DynamicArray<int> resultIndices;
  DynamicArray<Real3> resultContacts;
  SdfInfo resultSdf;
  resultIndices.reserve(positions.size());
  resultContacts.reserve(positions.size());
  resultSdf.reserve(positions.size());
  bool isSdfGradUnitary = false;

  // Go!
  for (auto _ : state) {
    resultIndices.clear();
    resultContacts.clear();
    resultSdf.clear();
    CallNoInline([&]() {
      FindPointContactsT(
          positionsTransformed,
          collider,
          cdParams,
          pointsFromCollider,
          resultIndices,
          resultContacts,
          resultSdf,
          isSdfGradUnitary);
    });
  }

  // Stats
  state.counters["points/second"] =
      benchmark::Counter(state.iterations() * positions.size(), benchmark::Counter::kIsRate);
  state.counters["hits"] = benchmark::Counter(
      state.iterations() * resultIndices.size(), benchmark::Counter::kAvgIterations);

  // Sanity check
  if (percentVolumeOverlap <= 0_r) { // No hits
    MOCHI_ASSERT(resultIndices.empty());
    MOCHI_ASSERT(resultContacts.empty());
    MOCHI_ASSERT(resultSdf.empty());
  } else if (percentVolumeOverlap >= 1_r && expectAllContactsFor100PercentOverlap) { // All hits
    MOCHI_ASSERT(resultIndices.size() == positions.size());
    MOCHI_ASSERT(resultContacts.size() == positions.size());
    MOCHI_ASSERT(resultSdf.size() == positions.size());
  } else { // Some hits
    MOCHI_ASSERT(!resultIndices.empty());
    MOCHI_ASSERT(resultContacts.size() == resultIndices.size());
    MOCHI_ASSERT(resultSdf.size() == resultIndices.size());
  }
}

static void BenchmarkFindPointContact_Plane(
    benchmark::State& state,
    Span<Real3 const> positions,
    real percentVolumeOverlap) {
  // Position the plane based on percentVolumeOverlap
  auto bounds = CalcAabb(positions);
  auto planeOffset = Lerp(bounds.GetMin()[1], bounds.GetMax()[1], percentVolumeOverlap);
  if (percentVolumeOverlap == 0_r) {
    planeOffset -= 1_r; // Move down to ensure zero contacts on the boundary
  }
  Plane plane{Real3{0_r, 1_r, 0_r}, planeOffset};
  BenchmarkFindPointContact(state, &plane, positions, percentVolumeOverlap);
}

static void FindPointContacts_Mesh_vs_Plane(
    benchmark::State& state,
    std::string_view meshPath,
    real percentVolumeOverlap) {
  // Use multiple threads to initialize the SDF grids
  auto* context = mochi::CreateContext(TaskScheduler::GetNumSupportedPhysicalProcessors());
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto positions = GetSamplesPositionsFromMesh(context, meshPath);

  // Benchmark is single-threaded
  context->SetIsSingleThreaded(true);
  BenchmarkFindPointContact_Plane(state, positions, percentVolumeOverlap);
}

// clang-format off
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Plane, FindPointContacts_Mesh_vs_Plane_OverlapPct00, kMeshPath, 0.00_r)->Name("FindPointContacts/MeshVsPlane/OverlapPct00");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Plane, FindPointContacts_Mesh_vs_Plane_OverlapPct01, kMeshPath, 0.01_r)->Name("FindPointContacts/MeshVsPlane/OverlapPct01");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Plane, FindPointContacts_Mesh_vs_Plane_OverlapPct05, kMeshPath, 0.05_r)->Name("FindPointContacts/MeshVsPlane/OverlapPct05");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Plane, FindPointContacts_Mesh_vs_Plane_OverlapPct10, kMeshPath, 0.1_r)->Name("FindPointContacts/MeshVsPlane/OverlapPct10");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Plane, FindPointContacts_Mesh_vs_Plane_OverlapPct25, kMeshPath, 0.25_r)->Name("FindPointContacts/MeshVsPlane/OverlapPct25");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Plane, FindPointContacts_Mesh_vs_Plane_OverlapPct50, kMeshPath, 0.5_r)->Name("FindPointContacts/MeshVsPlane/OverlapPct50");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Plane, FindPointContacts_Mesh_vs_Plane_OverlapPct75, kMeshPath, 0.75_r)->Name("FindPointContacts/MeshVsPlane/OverlapPct75");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Plane, FindPointContacts_Mesh_vs_Plane_OverlapPct100, kMeshPath, 1.0_r)->Name("FindPointContacts/MeshVsPlane/OverlapPct100");
// clang-format on

static void FindPointContacts_Rand_vs_Plane(
    benchmark::State& state,
    size_t numSamples,
    real percentVolumeOverlap) {
  auto rng = RandomGenerator(42);
  DynamicArray<Real3> positions(numSamples);
  SetRandom(rng, 0_r, 1_r, MakeSpan(positions)); // Random points within a unit cube
  BenchmarkFindPointContact_Plane(state, positions, percentVolumeOverlap);
}

// clang-format off
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Plane, FindPointContacts_Rand_vs_Plane_OverlapPct00, kNumRandPoints, 0.00_r)->Name("FindPointContacts/RandVsPlane/OverlapPct00");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Plane, FindPointContacts_Rand_vs_Plane_OverlapPct01, kNumRandPoints, 0.01_r)->Name("FindPointContacts/RandVsPlane/OverlapPct01");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Plane, FindPointContacts_Rand_vs_Plane_OverlapPct05, kNumRandPoints, 0.05_r)->Name("FindPointContacts/RandVsPlane/OverlapPct05");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Plane, FindPointContacts_Rand_vs_Plane_OverlapPct10, kNumRandPoints, 0.1_r)->Name("FindPointContacts/RandVsPlane/OverlapPct10");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Plane, FindPointContacts_Rand_vs_Plane_OverlapPct25, kNumRandPoints, 0.25_r)->Name("FindPointContacts/RandVsPlane/OverlapPct25");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Plane, FindPointContacts_Rand_vs_Plane_OverlapPct50, kNumRandPoints, 0.5_r)->Name("FindPointContacts/RandVsPlane/OverlapPct50");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Plane, FindPointContacts_Rand_vs_Plane_OverlapPct75, kNumRandPoints, 0.75_r)->Name("FindPointContacts/RandVsPlane/OverlapPct75");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Plane, FindPointContacts_Rand_vs_Plane_OverlapPct100, kNumRandPoints, 1.0_r)->Name("FindPointContacts/RandVsPlane/OverlapPct100");
// clang-format on

static void BenchmarkFindPointContact_Sphere(
    benchmark::State& state,
    Span<Real3 const> positions,
    real percentVolumeOverlap) {
  // Position the sphere based on percentVolumeOverlap
  auto bounds = CalcAabb(positions);
  auto center = bounds.GetCenter();
  auto radius = Norm(bounds.GetHalfExtents());
  auto height = bounds.GetMax()[1] - bounds.GetMin()[1];
  auto offset = Lerp(-0.5_r * height - radius, 0_r, percentVolumeOverlap);
  if (percentVolumeOverlap == 0_r) {
    offset -= 1_r; // Move down to ensure zero contacts on the boundary
  }
  center[1] += offset;
  Sphere sphere(center, radius);
  BenchmarkFindPointContact(state, &sphere, positions, percentVolumeOverlap);
}

static void FindPointContacts_Mesh_vs_Sphere(
    benchmark::State& state,
    std::string_view meshPath,
    real percentVolumeOverlap) {
  auto* context = mochi::CreateContext(0 /*numWorkerThreads*/);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto positions = GetSamplesPositionsFromMesh(context, meshPath);
  BenchmarkFindPointContact_Sphere(state, positions, percentVolumeOverlap);
}

// clang-format off
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sphere, FindPointContacts_Mesh_vs_Sphere_OverlapPct00, kMeshPath, 0.00_r)->Name("FindPointContacts/MeshVsSphere/OverlapPct00");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sphere, FindPointContacts_Mesh_vs_Sphere_OverlapPct01, kMeshPath, 0.01_r)->Name("FindPointContacts/MeshVsSphere/OverlapPct01");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sphere, FindPointContacts_Mesh_vs_Sphere_OverlapPct05, kMeshPath, 0.05_r)->Name("FindPointContacts/MeshVsSphere/OverlapPct05");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sphere, FindPointContacts_Mesh_vs_Sphere_OverlapPct10, kMeshPath, 0.1_r)->Name("FindPointContacts/MeshVsSphere/OverlapPct10");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sphere, FindPointContacts_Mesh_vs_Sphere_OverlapPct25, kMeshPath, 0.25_r)->Name("FindPointContacts/MeshVsSphere/OverlapPct25");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sphere, FindPointContacts_Mesh_vs_Sphere_OverlapPct50, kMeshPath, 0.5_r)->Name("FindPointContacts/MeshVsSphere/OverlapPct50");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sphere, FindPointContacts_Mesh_vs_Sphere_OverlapPct75, kMeshPath, 0.75_r)->Name("FindPointContacts/MeshVsSphere/OverlapPct75");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sphere, FindPointContacts_Mesh_vs_Sphere_OverlapPct100, kMeshPath, 1.0_r)->Name("FindPointContacts/MeshVsSphere/OverlapPct100");
// clang-format on

static void FindPointContacts_Rand_vs_Sphere(
    benchmark::State& state,
    size_t numSamples,
    real percentVolumeOverlap) {
  auto rng = RandomGenerator(42);
  DynamicArray<Real3> positions(numSamples);
  SetRandom(rng, 0_r, 1_r, MakeSpan(positions)); // Random points within a unit cube
  BenchmarkFindPointContact_Sphere(state, positions, percentVolumeOverlap);
}

// clang-format off
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Sphere, FindPointContacts_Rand_vs_Sphere_OverlapPct00, kNumRandPoints, 0.00_r)->Name("FindPointContacts/RandVsSphere/OverlapPct00");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Sphere, FindPointContacts_Rand_vs_Sphere_OverlapPct01, kNumRandPoints, 0.01_r)->Name("FindPointContacts/RandVsSphere/OverlapPct01");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Sphere, FindPointContacts_Rand_vs_Sphere_OverlapPct05, kNumRandPoints, 0.05_r)->Name("FindPointContacts/RandVsSphere/OverlapPct05");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Sphere, FindPointContacts_Rand_vs_Sphere_OverlapPct10, kNumRandPoints, 0.1_r)->Name("FindPointContacts/RandVsSphere/OverlapPct10");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Sphere, FindPointContacts_Rand_vs_Sphere_OverlapPct25, kNumRandPoints, 0.25_r)->Name("FindPointContacts/RandVsSphere/OverlapPct25");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Sphere, FindPointContacts_Rand_vs_Sphere_OverlapPct50, kNumRandPoints, 0.5_r)->Name("FindPointContacts/RandVsSphere/OverlapPct50");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Sphere, FindPointContacts_Rand_vs_Sphere_OverlapPct75, kNumRandPoints, 0.75_r)->Name("FindPointContacts/RandVsSphere/OverlapPct75");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Sphere, FindPointContacts_Rand_vs_Sphere_OverlapPct100, kNumRandPoints, 1.0_r)->Name("FindPointContacts/RandVsSphere/OverlapPct100");
// clang-format on

static void BenchmarkFindPointContact_Obb(
    benchmark::State& state,
    Span<Real3 const> positions,
    real percentVolumeOverlap) {
  // Position the Obb based on percentVolumeOverlap
  auto bounds = CalcAabb(positions);
  auto min = bounds.GetMin();
  auto max = bounds.GetMax();
  auto height = max[1] - min[1];
  auto offset = Lerp(-height, 0_r, percentVolumeOverlap);
  if (percentVolumeOverlap == 0_r) {
    offset -= 1_r; // Move down to ensure zero contacts on the boundary
  }
  auto aabb = Aabb{Real3{min[0], min[1] + offset, min[2]}, Real3{max[0], max[1] + offset, max[2]}};
  auto obb = GetObb(aabb);
  BenchmarkFindPointContact(state, &obb, positions, percentVolumeOverlap);
}

static void FindPointContacts_Mesh_vs_Obb(
    benchmark::State& state,
    std::string_view meshPath,
    real percentVolumeOverlap) {
  auto* context = mochi::CreateContext(0 /*numWorkerThreads*/);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto positions = GetSamplesPositionsFromMesh(context, meshPath);
  BenchmarkFindPointContact_Obb(state, positions, percentVolumeOverlap);
}

// clang-format off
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Obb, FindPointContacts_Mesh_vs_Obb_OverlapPct00, kMeshPath, 0.00_r)->Name("FindPointContacts/MeshVsObb/OverlapPct00");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Obb, FindPointContacts_Mesh_vs_Obb_OverlapPct01, kMeshPath, 0.01_r)->Name("FindPointContacts/MeshVsObb/OverlapPct01");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Obb, FindPointContacts_Mesh_vs_Obb_OverlapPct05, kMeshPath, 0.05_r)->Name("FindPointContacts/MeshVsObb/OverlapPct05");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Obb, FindPointContacts_Mesh_vs_Obb_OverlapPct10, kMeshPath, 0.1_r)->Name("FindPointContacts/MeshVsObb/OverlapPct10");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Obb, FindPointContacts_Mesh_vs_Obb_OverlapPct25, kMeshPath, 0.25_r)->Name("FindPointContacts/MeshVsObb/OverlapPct25");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Obb, FindPointContacts_Mesh_vs_Obb_OverlapPct50, kMeshPath, 0.5_r)->Name("FindPointContacts/MeshVsObb/OverlapPct50");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Obb, FindPointContacts_Mesh_vs_Obb_OverlapPct75, kMeshPath, 0.75_r)->Name("FindPointContacts/MeshVsObb/OverlapPct75");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Obb, FindPointContacts_Mesh_vs_Obb_OverlapPct100, kMeshPath, 1.0_r)->Name("FindPointContacts/MeshVsObb/OverlapPct100");
// clang-format on

static void FindPointContacts_Rand_vs_Obb(
    benchmark::State& state,
    size_t numSamples,
    real percentVolumeOverlap) {
  auto rng = RandomGenerator(42);
  DynamicArray<Real3> positions(numSamples);
  SetRandom(rng, 0_r, 1_r, MakeSpan(positions)); // Random points within a unit cube
  BenchmarkFindPointContact_Obb(state, positions, percentVolumeOverlap);
}

// clang-format off
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Obb, FindPointContacts_Rand_vs_Obb_OverlapPct00, kNumRandPoints, 0.00_r)->Name("FindPointContacts/RandVsObb/OverlapPct00");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Obb, FindPointContacts_Rand_vs_Obb_OverlapPct01, kNumRandPoints, 0.01_r)->Name("FindPointContacts/RandVsObb/OverlapPct01");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Obb, FindPointContacts_Rand_vs_Obb_OverlapPct05, kNumRandPoints, 0.05_r)->Name("FindPointContacts/RandVsObb/OverlapPct05");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Obb, FindPointContacts_Rand_vs_Obb_OverlapPct10, kNumRandPoints, 0.1_r)->Name("FindPointContacts/RandVsObb/OverlapPct10");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Obb, FindPointContacts_Rand_vs_Obb_OverlapPct25, kNumRandPoints, 0.25_r)->Name("FindPointContacts/RandVsObb/OverlapPct25");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Obb, FindPointContacts_Rand_vs_Obb_OverlapPct50, kNumRandPoints, 0.5_r)->Name("FindPointContacts/RandVsObb/OverlapPct50");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Obb, FindPointContacts_Rand_vs_Obb_OverlapPct75, kNumRandPoints, 0.75_r)->Name("FindPointContacts/RandVsObb/OverlapPct75");
BENCHMARK_CAPTURE(FindPointContacts_Rand_vs_Obb, FindPointContacts_Rand_vs_Obb_OverlapPct100, kNumRandPoints, 1.0_r)->Name("FindPointContacts/RandVsObb/OverlapPct100");
// clang-format on

static void BenchmarkFindPointContact_Mesh(
    benchmark::State& state,
    MeshCollider const* collider,
    Span<Real3 const> positions,
    real percentVolumeOverlap,
    bool wasRandomized) {
  MOCHI_ASSERT(collider != nullptr);

  // This benchmark assumes the points came from the same mesh as the collider, and thus fit within
  // the collider's bounds.
  Aabb colliderBounds = collider->GetMesh()->GetAabb();
  Aabb pointBounds = CalcAabb(positions);
  MOCHI_ASSERT(
      AllTrue(pointBounds.VGetMin() >= colliderBounds.VGetMin()) &&
          AllTrue(pointBounds.VGetMax() <= colliderBounds.VGetMax()),
      "Expected points to be inside the collider bounds");

  // Now shift those points up or down to achieve the requested percentVolumeOverlap
  real offset = Lerp(colliderBounds.GetSize()[1], 0_r, percentVolumeOverlap);
  DynamicArray<Real3> adjustedPoints(positions);
  for (auto& pt : adjustedPoints) {
    pt[1] += offset;
  }

  BenchmarkFindPointContact(state, collider, adjustedPoints, percentVolumeOverlap, !wasRandomized);
}

static void FindPointContacts_Mesh_vs_Mesh(
    benchmark::State& state,
    std::string_view meshPath,
    bool randomize,
    real percentVolumeOverlap) {
  auto* context = mochi::CreateContext(0 /*numWorkerThreads*/);
  MOCHI_DEFER(mochi::DestroyContext(context));
  MeshCollider const* meshCollider = nullptr;
  DynamicArray<Real3> positions(
      GetSamplesPositionsFromMesh(context, meshPath, &meshCollider, nullptr /*sdfCollider*/));
  MOCHI_ASSERT(meshCollider != nullptr);
  if (randomize) {
    Aabb bounds = meshCollider->GetMesh()->GetAabb();
    auto rng = RandomGenerator(42);
    for (auto& pt : positions) {
      for (int i = 0; i < 3; ++i) {
        pt[i] = RandomUniformValue<real>(rng, bounds.GetMin()[i], bounds.GetMax()[i]);
      }
    }
  }
  BenchmarkFindPointContact_Mesh(state, meshCollider, positions, percentVolumeOverlap, randomize);
}

// clang-format off
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Mesh, FindPointContacts_Mesh_vs_Mesh_OverlapPct00, kMeshPath, false, 0.00_r)->Name("FindPointContacts/MeshVsMesh/OverlapPct00");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Mesh, FindPointContacts_Mesh_vs_Mesh_OverlapPct01, kMeshPath, false, 0.01_r)->Name("FindPointContacts/MeshVsMesh/OverlapPct01");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Mesh, FindPointContacts_Mesh_vs_Mesh_OverlapPct05, kMeshPath, false, 0.05_r)->Name("FindPointContacts/MeshVsMesh/OverlapPct05");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Mesh, FindPointContacts_Mesh_vs_Mesh_OverlapPct10, kMeshPath, false, 0.1_r)->Name("FindPointContacts/MeshVsMesh/OverlapPct10");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Mesh, FindPointContacts_Mesh_vs_Mesh_OverlapPct25, kMeshPath, false, 0.25_r)->Name("FindPointContacts/MeshVsMesh/OverlapPct25");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Mesh, FindPointContacts_Mesh_vs_Mesh_OverlapPct50, kMeshPath, false, 0.5_r)->Name("FindPointContacts/MeshVsMesh/OverlapPct50");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Mesh, FindPointContacts_Mesh_vs_Mesh_OverlapPct75, kMeshPath, false, 0.75_r)->Name("FindPointContacts/MeshVsMesh/OverlapPct75");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Mesh, FindPointContacts_Mesh_vs_Mesh_OverlapPct100, kMeshPath, false, 1.0_r)->Name("FindPointContacts/MeshVsMesh/OverlapPct100");

BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Mesh, FindPointContacts_Rand_vs_Mesh_OverlapPct00, kMeshPath, true, 0.00_r)->Name("FindPointContacts/RandVsMesh/OverlapPct00");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Mesh, FindPointContacts_Rand_vs_Mesh_OverlapPct01, kMeshPath, true, 0.01_r)->Name("FindPointContacts/RandVsMesh/OverlapPct01");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Mesh, FindPointContacts_Rand_vs_Mesh_OverlapPct05, kMeshPath, true, 0.05_r)->Name("FindPointContacts/RandVsMesh/OverlapPct05");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Mesh, FindPointContacts_Rand_vs_Mesh_OverlapPct10, kMeshPath, true, 0.1_r)->Name("FindPointContacts/RandVsMesh/OverlapPct10");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Mesh, FindPointContacts_Rand_vs_Mesh_OverlapPct25, kMeshPath, true, 0.25_r)->Name("FindPointContacts/RandVsMesh/OverlapPct25");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Mesh, FindPointContacts_Rand_vs_Mesh_OverlapPct50, kMeshPath, true, 0.5_r)->Name("FindPointContacts/RandVsMesh/OverlapPct50");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Mesh, FindPointContacts_Rand_vs_Mesh_OverlapPct75, kMeshPath, true, 0.75_r)->Name("FindPointContacts/RandVsMesh/OverlapPct75");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Mesh, FindPointContacts_Rand_vs_Mesh_OverlapPct100, kMeshPath, true, 1.0_r)->Name("FindPointContacts/RandVsMesh/OverlapPct100");
// clang-format on

static void BenchmarkFindPointContact_Sdf(
    benchmark::State& state,
    GridSdf const* collider,
    Span<Real3 const> positions,
    real percentVolumeOverlap,
    bool wasRandomized) {
  MOCHI_ASSERT(collider != nullptr);

  // This benchmark assumes the points came from the same mesh as the collider, and thus fit within
  // the collider's bounds.
  Aabb colliderBounds = GetAabb(collider->GetColliderBounds());
  Aabb pointBounds = CalcAabb(positions);
  MOCHI_ASSERT(
      AllTrue(pointBounds.VGetMin() >= colliderBounds.VGetMin()) &&
          AllTrue(pointBounds.VGetMax() <= colliderBounds.VGetMax()),
      "Expected points to be inside the collider bounds");

  // Now shift those points up or down to achieve the requested percentVolumeOverlap
  real offset = Lerp(colliderBounds.GetSize()[1], 0_r, percentVolumeOverlap);
  DynamicArray<Real3> adjustedPoints(positions);
  for (auto& pt : adjustedPoints) {
    pt[1] += offset;
  }

  BenchmarkFindPointContact(state, collider, adjustedPoints, percentVolumeOverlap, !wasRandomized);
}

static void FindPointContacts_Mesh_vs_Sdf(
    benchmark::State& state,
    std::string_view meshPath,
    bool randomize,
    real percentVolumeOverlap) {
  // Create multi-threaded context for SDF construction
  auto* context = mochi::CreateContext(TaskScheduler::GetNumSupportedPhysicalProcessors());
  MOCHI_DEFER(mochi::DestroyContext(context));
  DynamicArray<Real3> positions(GetSamplesPositionsFromMesh(context, meshPath));

  // Building an SDF for a high resolution mesh is crazy slow, if it wasn't done offline.
  // Lets make sure it only happens once for all the benchmarks.
  static std::shared_ptr<Sdf const> s_sdf = [&]() {
    SdfCollider const* sdfCollider = nullptr;
    GetSamplesPositionsFromMesh(context, meshPath, nullptr /*meshCollider*/, &sdfCollider);
    return sdfCollider->shape;
  }();

  if (randomize) {
    Aabb bounds = GetAabb(s_sdf->GetColliderBounds());
    auto rng = RandomGenerator(42);
    for (auto& pt : positions) {
      for (int i = 0; i < 3; ++i) {
        pt[i] = RandomUniformValue<real>(rng, bounds.GetMin()[i], bounds.GetMax()[i]);
      }
    }
  }

  context->SetIsSingleThreaded(true); // Benchmark is single-threaded
  BenchmarkFindPointContact_Sdf(
      state, static_cast<GridSdf const*>(s_sdf.get()), positions, percentVolumeOverlap, randomize);
}

// clang-format off
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sdf, FindPointContacts_Mesh_vs_Sdf_OverlapPct00, kMeshPath, false, 0.00_r)->Name("FindPointContacts/MeshVsSdf/OverlapPct00");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sdf, FindPointContacts_Mesh_vs_Sdf_OverlapPct01, kMeshPath, false, 0.01_r)->Name("FindPointContacts/MeshVsSdf/OverlapPct01");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sdf, FindPointContacts_Mesh_vs_Sdf_OverlapPct05, kMeshPath, false, 0.05_r)->Name("FindPointContacts/MeshVsSdf/OverlapPct05");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sdf, FindPointContacts_Mesh_vs_Sdf_OverlapPct10, kMeshPath, false, 0.1_r)->Name("FindPointContacts/MeshVsSdf/OverlapPct10");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sdf, FindPointContacts_Mesh_vs_Sdf_OverlapPct25, kMeshPath, false, 0.25_r)->Name("FindPointContacts/MeshVsSdf/OverlapPct25");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sdf, FindPointContacts_Mesh_vs_Sdf_OverlapPct50, kMeshPath, false, 0.5_r)->Name("FindPointContacts/MeshVsSdf/OverlapPct50");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sdf, FindPointContacts_Mesh_vs_Sdf_OverlapPct75, kMeshPath, false, 0.75_r)->Name("FindPointContacts/MeshVsSdf/OverlapPct75");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sdf, FindPointContacts_Mesh_vs_Sdf_OverlapPct100, kMeshPath, false, 1.0_r)->Name("FindPointContacts/MeshVsSdf/OverlapPct100");

BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sdf, FindPointContacts_Rand_vs_Sdf_OverlapPct00, kMeshPath, true, 0.00_r)->Name("FindPointContacts/RandVsSdf/OverlapPct00");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sdf, FindPointContacts_Rand_vs_Sdf_OverlapPct01, kMeshPath, true, 0.01_r)->Name("FindPointContacts/RandVsSdf/OverlapPct01");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sdf, FindPointContacts_Rand_vs_Sdf_OverlapPct05, kMeshPath, true, 0.05_r)->Name("FindPointContacts/RandVsSdf/OverlapPct05");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sdf, FindPointContacts_Rand_vs_Sdf_OverlapPct10, kMeshPath, true, 0.1_r)->Name("FindPointContacts/RandVsSdf/OverlapPct10");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sdf, FindPointContacts_Rand_vs_Sdf_OverlapPct25, kMeshPath, true, 0.25_r)->Name("FindPointContacts/RandVsSdf/OverlapPct25");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sdf, FindPointContacts_Rand_vs_Sdf_OverlapPct50, kMeshPath, true, 0.5_r)->Name("FindPointContacts/RandVsSdf/OverlapPct50");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sdf, FindPointContacts_Rand_vs_Sdf_OverlapPct75, kMeshPath, true, 0.75_r)->Name("FindPointContacts/RandVsSdf/OverlapPct75");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_Sdf, FindPointContacts_Rand_vs_Sdf_OverlapPct100, kMeshPath, true, 1.0_r)->Name("FindPointContacts/RandVsSdf/OverlapPct100");
// clang-format on

namespace {

// Trivial BV used by BenchmarkFindIntersectingSamples_Bsh.
// Acts like a plane with normal (0, 1, 0) and offset y.
struct DummyPlaneBv {
  real y = 0_r;
};

// Single overlap test for BvTree
MOCHI_FORCE_INLINE bool HasOverlap(DummyPlaneBv const& planeBv, Sphere const& sphere) {
  return sphere.GetCenter()[1] + sphere.GetRadius() >= planeBv.y;
}

template <int kBatchSize>
MOCHI_FORCE_INLINE Simd<real, kBatchSize> HasOverlap(
    DummyPlaneBv const& planeBv,
    BatchSphere<kBatchSize> const& sphere) {
  return (sphere.center[1] + sphere.radius) >= planeBv.y;
}

// Batch overlap test for BvhTree
template <int kMaxBatchSize>
void HasOverlapBatch(
    int batchSize,
    DummyPlaneBv const& planeBv,
    Span<Sphere const> spheres,
    Span<bool> outHasOverlap) {
  for (int i = 0; i < batchSize; ++i) {
    outHasOverlap[i] = spheres[i].GetCenter()[1] + spheres[i].GetRadius() >= planeBv.y;
  }
}

} // namespace

// This benchmark case focuses on BSH culling. It uses DummyPlaneBv which does a minimal amount
// of work to give us results that still reflect the requested percentVolumeOverlap. Thus, this
// benchmark focuses on the BSH culling performance in (relative) isolation.
template <class Bsh>
static void BenchmarkFindIntersectingSamples_Bsh(
    benchmark::State& state,
    Bsh const& bsh,
    Span<Real3 const> positions,
    real percentVolumeOverlap) {
  // Compute Y threshold based on percentVolumeOverlap.
  Aabb bounds = CalcAabb(positions);
  real yThreshold = Lerp(bounds.GetMax()[1], bounds.GetMin()[1], percentVolumeOverlap);
  if (percentVolumeOverlap == 0_r) {
    yThreshold += 1_r; // Make sure there are zero hits for this case
  }
  DummyPlaneBv dummyBv{.y = yThreshold};

  DynamicArray<int> culledIndices;
  culledIndices.reserve(positions.size());

  std::function<void()> const benchmarkIteration = [&]() {
    bsh.FindIntersectingSamples(dummyBv, culledIndices);
  };
  for (auto _ : state) {
    CallNoInline(benchmarkIteration);
  }

  // Stats
  state.counters["points/second"] =
      benchmark::Counter(state.iterations() * positions.size(), benchmark::Counter::kIsRate);
  state.counters["hits"] = benchmark::Counter(
      state.iterations() * culledIndices.size(), benchmark::Counter::kAvgIterations);
}

static void FindIntersectingSamples_Mesh_vs_Bsh(
    benchmark::State& state,
    std::string_view meshPath,
    real percentVolumeOverlap,
    bool useNewBsh) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto positions = GetSamplesPositionsFromMesh(context, meshPath);
  if (useNewBsh) {
    // (kMaxPerLeaf == 8) matches the behavior of ContactSamplesBvh, making it a fair comparison.
    int constexpr kMaxPerLeaf = 8;
    auto bsh = SphereTree<8>::FromPoints(positions, kMaxPerLeaf);
    BenchmarkFindIntersectingSamples_Bsh(state, bsh, positions, percentVolumeOverlap);
  } else {
    ContactSamplesBvh<Sphere> bsh(positions);
    BenchmarkFindIntersectingSamples_Bsh(state, bsh, positions, percentVolumeOverlap);
  }
}

// clang-format off
BENCHMARK_CAPTURE(FindIntersectingSamples_Mesh_vs_Bsh, FindIntersectingSamples_Mesh_vs_Bsh_OverlapPct00, kMeshPath, 0.00_r, false)->Name("FindIntersectingSamples/MeshVsBsh/OverlapPct00");
BENCHMARK_CAPTURE(FindIntersectingSamples_Mesh_vs_Bsh, FindIntersectingSamples_Mesh_vs_Bsh_OverlapPct01, kMeshPath, 0.01_r, false)->Name("FindIntersectingSamples/MeshVsBsh/OverlapPct01");
BENCHMARK_CAPTURE(FindIntersectingSamples_Mesh_vs_Bsh, FindIntersectingSamples_Mesh_vs_Bsh_OverlapPct05, kMeshPath, 0.05_r, false)->Name("FindIntersectingSamples/MeshVsBsh/OverlapPct05");
BENCHMARK_CAPTURE(FindIntersectingSamples_Mesh_vs_Bsh, FindIntersectingSamples_Mesh_vs_Bsh_OverlapPct10, kMeshPath, 0.1_r, false)->Name("FindIntersectingSamples/MeshVsBsh/OverlapPct10");
BENCHMARK_CAPTURE(FindIntersectingSamples_Mesh_vs_Bsh, FindIntersectingSamples_Mesh_vs_Bsh_OverlapPct25, kMeshPath, 0.25_r, false)->Name("FindIntersectingSamples/MeshVsBsh/OverlapPct25");
BENCHMARK_CAPTURE(FindIntersectingSamples_Mesh_vs_Bsh, FindIntersectingSamples_Mesh_vs_Bsh_OverlapPct50, kMeshPath, 0.5_r, false)->Name("FindIntersectingSamples/MeshVsBsh/OverlapPct50");
BENCHMARK_CAPTURE(FindIntersectingSamples_Mesh_vs_Bsh, FindIntersectingSamples_Mesh_vs_Bsh_OverlapPct75, kMeshPath, 0.75_r, false)->Name("FindIntersectingSamples/MeshVsBsh/OverlapPct75");
BENCHMARK_CAPTURE(FindIntersectingSamples_Mesh_vs_Bsh, FindIntersectingSamples_Mesh_vs_Bsh_OverlapPct100, kMeshPath, 1.0_r, false)->Name("FindIntersectingSamples/MeshVsBsh/OverlapPct100");

BENCHMARK_CAPTURE(FindIntersectingSamples_Mesh_vs_Bsh, FindIntersectingSamples_Mesh_vs_Bsh_OverlapPct00, kMeshPath, 0.00_r, true)->Name("FindIntersectingSamples/MeshVsBshNew/OverlapPct00");
BENCHMARK_CAPTURE(FindIntersectingSamples_Mesh_vs_Bsh, FindIntersectingSamples_Mesh_vs_Bsh_OverlapPct01, kMeshPath, 0.01_r, true)->Name("FindIntersectingSamples/MeshVsBshNew/OverlapPct01");
BENCHMARK_CAPTURE(FindIntersectingSamples_Mesh_vs_Bsh, FindIntersectingSamples_Mesh_vs_Bsh_OverlapPct05, kMeshPath, 0.05_r, true)->Name("FindIntersectingSamples/MeshVsBshNew/OverlapPct05");
BENCHMARK_CAPTURE(FindIntersectingSamples_Mesh_vs_Bsh, FindIntersectingSamples_Mesh_vs_Bsh_OverlapPct10, kMeshPath, 0.1_r, true)->Name("FindIntersectingSamples/MeshVsBshNew/OverlapPct10");
BENCHMARK_CAPTURE(FindIntersectingSamples_Mesh_vs_Bsh, FindIntersectingSamples_Mesh_vs_Bsh_OverlapPct25, kMeshPath, 0.25_r, true)->Name("FindIntersectingSamples/MeshVsBshNew/OverlapPct25");
BENCHMARK_CAPTURE(FindIntersectingSamples_Mesh_vs_Bsh, FindIntersectingSamples_Mesh_vs_Bsh_OverlapPct50, kMeshPath, 0.5_r, true)->Name("FindIntersectingSamples/MeshVsBshNew/OverlapPct50");
BENCHMARK_CAPTURE(FindIntersectingSamples_Mesh_vs_Bsh, FindIntersectingSamples_Mesh_vs_Bsh_OverlapPct75, kMeshPath, 0.75_r, true)->Name("FindIntersectingSamples/MeshVsBshNew/OverlapPct75");
BENCHMARK_CAPTURE(FindIntersectingSamples_Mesh_vs_Bsh, FindIntersectingSamples_Mesh_vs_Bsh_OverlapPct100, kMeshPath, 1.0_r, true)->Name("FindIntersectingSamples/MeshVsBshNew/OverlapPct100");
// clang-format on

// This benchmark uses both BSH culling and an SDF collider, similar to the actual code in
// mochi_contact.cpp.
static void BenchmarkFindPointContact_SdfWithBsh(
    benchmark::State& state,
    GridSdf const* sdf,
    Span<Real3 const> positions,
    real percentVolumeOverlap,
    bool useNewBsh) {
  MOCHI_ASSERT(sdf != nullptr);

  // Adjust points based on percentVolumeOverlap (same as existing SDF benchmark)
  Aabb colliderBounds = GetAabb(sdf->GetColliderBounds());
  real offset = Lerp(colliderBounds.GetSize()[1], 0_r, percentVolumeOverlap);
  DynamicArray<Real3> adjustedPoints(positions);
  for (auto& pt : adjustedPoints) {
    pt[1] += offset;
  }

  // Build BSH from adjusted points
  ContactSamplesBvh<Sphere> bsh(adjustedPoints);

  int constexpr kMaxPerLeaf = 16;
  auto bsh2 = SphereTree<8>::FromPoints(adjustedPoints, kMaxPerLeaf);

  ContactDetectionParams cdParams;
  // Match the fine-contact tolerance so culling cannot reject valid contacts.
  VMatrix4x4r gridFromPointsT = sdf->GetGridFromActorTranspose();
  SdfBv sdfBv{
      .gridSdf = sdf, .distanceThreshold = cdParams.tolerance, .gridFromPointsT = gridFromPointsT};

  // Non-identity transform for FindPointContactsT
  TransformRT pointsFromCollider =
      TransformRT{Quaternion::RotationX(kPI * 0.25_r), Real3{0.1_r, 0.2_r, 0.3_r}};
  DynamicArray<Real3> positionsTransformed;
  positionsTransformed.resize_noinit(adjustedPoints.size());
  ArrayTransformPoints(
      MakeSpan(positionsTransformed), MakeConstSpan(adjustedPoints), pointsFromCollider);

  DynamicArray<int> culledIndices;
  DynamicArray<Real3> culledPositions;
  DynamicArray<int> resultIndices;
  DynamicArray<Real3> resultContacts;
  SdfInfo resultSdf;
  bool isSdfGradUnitary = false;

  culledIndices.reserve(positions.size());
  culledPositions.reserve(positions.size());
  resultIndices.reserve(positions.size());
  resultContacts.reserve(positions.size());
  resultSdf.reserve(positions.size());

  DynamicArray<int> expectedResultIndices;
  DynamicArray<Real3> expectedContacts;
  SdfInfo expectedSdf;
  FindPointContactsT(
      positionsTransformed,
      sdf,
      cdParams,
      pointsFromCollider,
      expectedResultIndices,
      expectedContacts,
      expectedSdf,
      isSdfGradUnitary);

  std::function<void()> const benchmarkIteration = [&]() {
    // Step 1: BSH culling
    if (useNewBsh) {
      bsh2.FindIntersectingSamples(sdfBv, culledIndices);
    } else {
      bsh.FindIntersectingSamples(sdfBv, culledIndices);
    }

    // Step 2: Gather culled positions
    culledPositions.resize_noinit(culledIndices.size());
    for (int i = 0; i < isize(culledIndices); ++i) {
      culledPositions[i] = positionsTransformed[culledIndices[i]];
    }

    // Step 3: SDF query on culled points only
    resultIndices.clear();
    resultContacts.clear();
    resultSdf.clear();
    FindPointContactsT(
        culledPositions,
        sdf,
        cdParams,
        pointsFromCollider,
        resultIndices,
        resultContacts,
        resultSdf,
        isSdfGradUnitary);
  };
  benchmarkIteration();
  DynamicArray<int> actualResultIndices;
  actualResultIndices.resize_noinit(resultIndices.size());
  for (int i = 0; i < isize(resultIndices); ++i) {
    actualResultIndices[i] = culledIndices[resultIndices[i]];
  }
  std::sort(expectedResultIndices.begin(), expectedResultIndices.end());
  std::sort(actualResultIndices.begin(), actualResultIndices.end());
  if (actualResultIndices != expectedResultIndices) {
    state.SkipWithError("BSH culling changed the contact result");
    return;
  }

  for (auto _ : state) {
    CallNoInline(benchmarkIteration);
  }
  // Stats
  state.counters["points/second"] =
      benchmark::Counter(state.iterations() * positions.size(), benchmark::Counter::kIsRate);
  state.counters["culled"] = benchmark::Counter(
      state.iterations() * culledIndices.size(), benchmark::Counter::kAvgIterations);
  state.counters["hits"] = benchmark::Counter(
      state.iterations() * resultIndices.size(), benchmark::Counter::kAvgIterations);
}

static void FindPointContacts_Mesh_vs_SdfWithBsh(
    benchmark::State& state,
    std::string_view meshPath,
    real percentVolumeOverlap,
    bool useNewBsh) {
  // Create multi-threaded context for SDF construction
  auto* context = mochi::CreateContext(TaskScheduler::GetNumSupportedPhysicalProcessors());
  MOCHI_DEFER(mochi::DestroyContext(context));
  DynamicArray<Real3> positions(GetSamplesPositionsFromMesh(context, meshPath));
  static std::shared_ptr<Sdf const> s_sdf = [&]() {
    SdfCollider const* sdfCollider = nullptr;
    GetSamplesPositionsFromMesh(context, meshPath, nullptr, &sdfCollider);
    return sdfCollider->shape;
  }();

  context->SetIsSingleThreaded(true); // Benchmark is single-threaded
  BenchmarkFindPointContact_SdfWithBsh(
      state, static_cast<GridSdf const*>(s_sdf.get()), positions, percentVolumeOverlap, useNewBsh);
}

// clang-format off
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_SdfWithBsh, FindPointContacts_MeshVsSdfWithBsh_OverlapPct00, kMeshPath, 0.00_r, false)->Name("FindPointContacts/MeshVsSdfWithBsh/OverlapPct00");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_SdfWithBsh, FindPointContacts_MeshVsSdfWithBsh_OverlapPct01, kMeshPath, 0.01_r, false)->Name("FindPointContacts/MeshVsSdfWithBsh/OverlapPct01");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_SdfWithBsh, FindPointContacts_MeshVsSdfWithBsh_OverlapPct05, kMeshPath, 0.05_r, false)->Name("FindPointContacts/MeshVsSdfWithBsh/OverlapPct05");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_SdfWithBsh, FindPointContacts_MeshVsSdfWithBsh_OverlapPct10, kMeshPath, 0.1_r, false)->Name("FindPointContacts/MeshVsSdfWithBsh/OverlapPct10");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_SdfWithBsh, FindPointContacts_MeshVsSdfWithBsh_OverlapPct25, kMeshPath, 0.25_r, false)->Name("FindPointContacts/MeshVsSdfWithBsh/OverlapPct25");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_SdfWithBsh, FindPointContacts_MeshVsSdfWithBsh_OverlapPct50, kMeshPath, 0.5_r, false)->Name("FindPointContacts/MeshVsSdfWithBsh/OverlapPct50");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_SdfWithBsh, FindPointContacts_MeshVsSdfWithBsh_OverlapPct75, kMeshPath, 0.75_r, false)->Name("FindPointContacts/MeshVsSdfWithBsh/OverlapPct75");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_SdfWithBsh, FindPointContacts_MeshVsSdfWithBsh_OverlapPct100, kMeshPath, 1.0_r, false)->Name("FindPointContacts/MeshVsSdfWithBsh/OverlapPct100");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_SdfWithBsh, FindPointContacts_MeshVsSdfWithBshNew_OverlapPct00, kMeshPath, 0.00_r, true)->Name("FindPointContacts/MeshVsSdfWithBshNew/OverlapPct00");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_SdfWithBsh, FindPointContacts_MeshVsSdfWithBshNew_OverlapPct01, kMeshPath, 0.01_r, true)->Name("FindPointContacts/MeshVsSdfWithBshNew/OverlapPct01");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_SdfWithBsh, FindPointContacts_MeshVsSdfWithBshNew_OverlapPct05, kMeshPath, 0.05_r, true)->Name("FindPointContacts/MeshVsSdfWithBshNew/OverlapPct05");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_SdfWithBsh, FindPointContacts_MeshVsSdfWithBshNew_OverlapPct10, kMeshPath, 0.1_r, true)->Name("FindPointContacts/MeshVsSdfWithBshNew/OverlapPct10");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_SdfWithBsh, FindPointContacts_MeshVsSdfWithBshNew_OverlapPct25, kMeshPath, 0.25_r, true)->Name("FindPointContacts/MeshVsSdfWithBshNew/OverlapPct25");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_SdfWithBsh, FindPointContacts_MeshVsSdfWithBshNew_OverlapPct50, kMeshPath, 0.5_r, true)->Name("FindPointContacts/MeshVsSdfWithBshNew/OverlapPct50");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_SdfWithBsh, FindPointContacts_MeshVsSdfWithBshNew_OverlapPct75, kMeshPath, 0.75_r, true)->Name("FindPointContacts/MeshVsSdfWithBshNew/OverlapPct75");
BENCHMARK_CAPTURE(FindPointContacts_Mesh_vs_SdfWithBsh, FindPointContacts_MeshVsSdfWithBshNew_OverlapPct100, kMeshPath, 1.0_r, true)->Name("FindPointContacts/MeshVsSdfWithBshNew/OverlapPct100");
// clang-format on

#endif // MOCHI_INTERNAL
