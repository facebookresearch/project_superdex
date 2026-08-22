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

#include <mochi_core/geometry/aabb.h>
#include <mochi_core/geometry/scalar_field.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

#include <array>
#include <cstdint>

using namespace mochi;

namespace mochi_benchmark {

// Batch size = Native SIMD size.
static constexpr int kBatchSize = Simd<real>::kSize;

// Number of points and batches per benchmark iteration.
static constexpr int kNumPointsPerIter = 2048;
static constexpr int kNumBatchesPerIter = kNumPointsPerIter / kBatchSize;
static_assert(kNumPointsPerIter % kBatchSize == 0);

using PointBatch = NdArray<Simd<real, kBatchSize>, 3>;

// Dense grid holding the signed distance to a sphere.
static DenseGrid3D<real> MakeSphereSdfGrid() {
  constexpr Int3 kGridSize{32, 32, 32};
  constexpr Real3 kCenter{0.5_r, 0.5_r, 0.5_r};
  constexpr real kRadius = 0.35_r;
  constexpr real kPadding = 0.15_r;
  Aabb const bounds{kCenter - (kRadius + kPadding), kCenter + (kRadius + kPadding)};

  DenseGrid3D<real> grid{kGridSize, bounds, bounds};
  Real3 const delta = bounds.GetSize() / StaticCast<Real3>(kGridSize - 1);
  for (int x = 0; x < kGridSize[0]; ++x) {
    for (int y = 0; y < kGridSize[1]; ++y) {
      for (int z = 0; z < kGridSize[2]; ++z) {
        Int3 const index{x, y, z};
        Real3 const pt = StaticCast<Real3>(index) * delta + bounds.GetMin();
        grid(index) = Norm(pt - kCenter) - kRadius;
      }
    }
  }
  return grid;
}

// Build kNumBatchesPerIter batches with a controlled batch-level interior fraction (either all or
// none of the points in a batch are interior).
static DynamicArray<PointBatch> MakeQueryBatches(DenseGrid3D<real> const& grid, int interiorPct) {
  MOCHI_ASSERT(interiorPct >= 0 && interiorPct <= 100, "Invalid interior percentage.");
  Real3 const minCorner = grid.GetGridLowerCorner();
  Real3 const maxCorner = grid.GetGridUpperCorner();
  Real3 const boundsSize = maxCorner - minCorner;
  Real3 const delta = grid.GetVoxelSize();
  Int3 const cells = grid.GetDimensions() - 1; // (dims - 1) cells per axis
  int const numCells = cells[0] * cells[1] * cells[2];
  constexpr int kStride = 7919; // prime, coprime to numCells -> spreads coverage

  int const numInteriorBatches = (kNumBatchesPerIter * interiorPct) / 100;

  DynamicArray<PointBatch> batches(kNumBatchesPerIter);
  for (int b = 0; b < kNumBatchesPerIter; ++b) {
    bool const interiorBatch = b < numInteriorBatches;
    std::array<Real3, kBatchSize> pts{};
    for (int i = 0; i < kBatchSize; ++i) {
      int const c = ((b * kBatchSize + i) * kStride) % numCells;
      int const cx = c % cells[0];
      int const cy = (c / cells[0]) % cells[1];
      int const cz = c / (cells[0] * cells[1]);
      if (interiorBatch) {
        // Voxel-cell center: strictly interior, parametric coord ~(0.5, 0.5, 0.5).
        pts[i] = minCorner + (StaticCast<Real3>(Int3{cx, cy, cz}) + 0.5_r) * delta;
      } else {
        // Beyond the max corner on all axes (varied per lane) -> exercises extrapolation.
        real const frac = StaticCast<real>(c) / StaticCast<real>(numCells);
        pts[i] = maxCorner + boundsSize * (0.25_r + 0.5_r * frac);
      }
    }
    LoadTransposed(&pts[0][0], batches[b]);
  }
  return batches;
}

template <GridExtrapolation kMode, bool kValues, bool kGradients, int kInteriorPct>
static void BenchmarkTrilinearSampleBatch(benchmark::State& state) {
  static DenseGrid3D<real> const grid = MakeSphereSdfGrid();
  static DynamicArray<PointBatch> const batches = MakeQueryBatches(grid, kInteriorPct);

  Simd<real, kBatchSize> values{};
  PointBatch gradients{};
  for (auto _ : state) {
    for (auto const& pts : batches) {
      if constexpr (kValues && kGradients) {
        grid.template TrilinearSampleBatch<kBatchSize, kMode, true, true>(pts, &values, &gradients);
      } else if constexpr (kValues) {
        grid.template TrilinearSampleBatch<kBatchSize, kMode, true, false>(pts, &values);
      } else {
        static_assert(kGradients, "Must compute something");
        grid.template TrilinearSampleBatch<kBatchSize, kMode, false, true>(
            pts, nullptr, &gradients);
      }
      if constexpr (kValues) {
        MOCHI_NO_DISCARD_IN_LOOP(values);
      }
      if constexpr (kGradients) {
        MOCHI_NO_DISCARD_IN_LOOP(gradients);
      }
    }
  }
  state.counters["Points/second"] = benchmark::Counter(
      state.iterations() * static_cast<int64_t>(kNumBatchesPerIter) * kBatchSize,
      benchmark::Counter::kIsRate);
}

#define MOCHI_BENCHMARK_TRILINEAR_CASE(Mode, Values, Gradients, Label, InteriorPct)           \
  BENCHMARK_TEMPLATE(                                                                         \
      BenchmarkTrilinearSampleBatch, GridExtrapolation::Mode, Values, Gradients, InteriorPct) \
      ->Name("ScalarField/TrilinearSampleBatch/" #Mode "/" Label "/Interior" #InteriorPct);

#define MOCHI_BENCHMARK_TRILINEAR_SWEEP(Mode, Values, Gradients, Label) \
  MOCHI_BENCHMARK_TRILINEAR_CASE(Mode, Values, Gradients, Label, 100)   \
  MOCHI_BENCHMARK_TRILINEAR_CASE(Mode, Values, Gradients, Label, 50)    \
  MOCHI_BENCHMARK_TRILINEAR_CASE(Mode, Values, Gradients, Label, 0)

// GridExtrapolation::LowerBound. Used by production collision culling. Only value sampling is
// implemented, so gradient variants are intentionally omitted.
MOCHI_BENCHMARK_TRILINEAR_SWEEP(LowerBound, true, false, "Value");

// GridExtrapolation::UpperBound. Used by production SDF queries.
MOCHI_BENCHMARK_TRILINEAR_SWEEP(UpperBound, true, false, "Value");
MOCHI_BENCHMARK_TRILINEAR_SWEEP(UpperBound, false, true, "Gradient");
MOCHI_BENCHMARK_TRILINEAR_SWEEP(UpperBound, true, true, "ValueAndGradient");

// GridExtrapolation::Unsupported. Used by production SDF queries. Only valid at 100% interior.
// Compare against the 100%-interior UpperBound cases above to estimate the overhead of the
// UpperBound path.
MOCHI_BENCHMARK_TRILINEAR_CASE(Unsupported, true, false, "Value", 100);
MOCHI_BENCHMARK_TRILINEAR_CASE(Unsupported, false, true, "Gradient", 100);
MOCHI_BENCHMARK_TRILINEAR_CASE(Unsupported, true, true, "ValueAndGradient", 100);

#undef MOCHI_BENCHMARK_TRILINEAR_SWEEP
#undef MOCHI_BENCHMARK_TRILINEAR_CASE

} // namespace mochi_benchmark
