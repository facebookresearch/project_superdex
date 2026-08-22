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

#pragma once

#include <mochi_core/element_operations/batched_element_utils.h>
#include <mochi_core/elements/tetrahedral/simplex_quadrature.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>

#include <gtest/gtest.h>

#include <utility>

namespace mochi::fem {

inline constexpr int kNumTrials = 10;
inline constexpr real kRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 5e-4_r;

struct OutputConfig {
  bool energy;
  bool residual;
  bool dresidual;
};

// All 7 valid permutations of energy/residual/dresidual (excludes all-false which is illegal).
inline constexpr OutputConfig kAllOutputConfigs[] = {
    {.energy = true, .residual = true, .dresidual = true},
    {.energy = true, .residual = true, .dresidual = false},
    {.energy = true, .residual = false, .dresidual = true},
    {.energy = true, .residual = false, .dresidual = false},
    {.energy = false, .residual = true, .dresidual = true},
    {.energy = false, .residual = true, .dresidual = false},
    {.energy = false, .residual = false, .dresidual = true},
};

// Invokes `fn.template operator()<kBS>()` for each batch size in `kBSs`.
template <int... kBSs, class TestFn>
void RunBatchSizes(TestFn const& fn) {
  (fn.template operator()<kBSs>(), ...);
}

// Invokes `fn` for scalar coverage and the explicitly-instantiated production FEM batch size.
template <class TestFn>
void RunSupportedFemShellRodBatchSizes(TestFn const& fn) {
  RunBatchSizes<1, kDefaultFemBatchSize>(fn);
}

// Invokes `fn` for the canonical set of batch sizes covering scalar (1), unaligned (2,3,5,6,7,9),
// aligned (4,8), and large (16) configurations.
template <class TestFn>
void RunAllBatchSizes(TestFn const& fn) {
  RunBatchSizes<1, 2, 3, 4, 5, 6, 7, 8, 9, 16>(fn);
}

template <class ElementT>
struct TestTetMeshData {
  static constexpr int kDim = ElementT::kSpaceDim * ElementT::kNumDofs;

  TetrahedralMesh mesh;
  DynamicArray<ElementT> elements;

  TestTetMeshData() : mesh(test::CreateMinimalTetMeshSingleTet()) {
    PopulateElements();
  }

  static TestTetMeshData CreateMinimalCube() {
    return TestTetMeshData(test::CreateMinimalTetMeshUnitCube());
  }

 private:
  explicit TestTetMeshData(test::TetMeshParams params) : mesh(params) {
    PopulateElements();
  }

  void PopulateElements() {
    int const numElements = isize(mesh.GetElementConnectivity());
    for (int i = 0; i < numElements; ++i) {
      if constexpr (ElementT::kNumQuadPoints == 1) {
        elements.emplace_back(
            i,
            mesh.GetNodeCoordinates(),
            mesh.GetElementConnectivity(),
            tetrahedral::kTetrahedralQuadrature1);
      } else {
        static_assert(ElementT::kNumQuadPoints == 4);
        elements.emplace_back(
            i,
            mesh.GetNodeCoordinates(),
            mesh.GetElementConnectivity(),
            tetrahedral::kTetrahedralQuadrature4);
      }
    }
  }
};

template <class ElementT>
struct TestTriMeshData {
  static constexpr int kDim = ElementT::kSpaceDim * ElementT::kNumDofs;

  TriangularMesh mesh;
  DynamicArray<ElementT> elements;

  static TestTriMeshData CreateMinimalCube() {
    return TestTriMeshData(test::CreateMinimalTriMeshUnitCube());
  }

 private:
  explicit TestTriMeshData(TriangularMesh mesh_) : mesh(std::move(mesh_)) {
    int const numElements = isize(mesh.GetElementConnectivity());
    for (int i = 0; i < numElements; ++i) {
      elements.emplace_back(i, mesh.GetNodeCoordinates(), mesh.GetElementConnectivity());
    }
  }
};

inline void ExpectNearEnergy(double ref, double actual, real relTol = kRelTol) {
  double const scale = Max(Abs(ref), static_cast<double>(relTol));
  EXPECT_NEAR(ref, actual, relTol * scale);
}

template <size_t kSize>
void ExpectNearL2(
    NdArray<real, kSize> const& ref,
    NdArray<real, kSize> const& actual,
    real relTol = kRelTol) {
  EXPECT_LE(Norm(ref - actual), relTol * Max(Norm(ref), relTol));
}

template <int kDim>
NdArray<real, kDim> MakeRandomArray(unsigned int seed, real lo = -0.3_r, real hi = 0.3_r) {
  auto gen = RandomGenerator(seed);
  NdArray<real, kDim> arr;
  SetRandom(gen, lo, hi, arr);
  return arr;
}

} // namespace mochi::fem
