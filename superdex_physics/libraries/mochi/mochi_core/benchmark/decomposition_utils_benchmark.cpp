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

#include <mochi_core/test/batch_helpers.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/matrix_utils.h>

#include <functional>

using namespace mochi;

namespace mochi_benchmark {

static constexpr int kBS = Min(2 * Simd<real>::kSize, 8);
using V3 = BatchReal3<kBS>;
using V3x3 = BatchReal3x3<kBS>;

static VMatrix3x3r GetArbitraryMatrix3x3() {
  return {Vec4r{1_r, 0.5_r, -0.2_r}, Vec4r{0.15_r, -2_r, -0.75_r}, Vec4r{1.2_r, -0.3_r, -0.5_r}};
}

static void AnalyticalEigendecompSym3x3(benchmark::State& state, VMatrix3x3r const& A) {
  VSymMatrix3x3r const Asym = SimdFullToSym(A);
  VMatrix3x3r VT;
  Vec4r lambda;
  std::function<void()> fn = [&]() { mochi::AnalyticalEigendecompSym3x3(Asym, lambda, &VT); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
  state.counters["matrices/s"] =
      benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK_CAPTURE(AnalyticalEigendecompSym3x3, "AnalyticalEigendecompSym3x3_Identity", VEye<3>());
BENCHMARK_CAPTURE(
    AnalyticalEigendecompSym3x3,
    "AnalyticalEigendecompSym3x3_ArbitraryMatrix",
    GetArbitraryMatrix3x3());

static void RotationVariantSvd3x3(benchmark::State& state, VMatrix3x3r const& A) {
  VMatrix3x3r U, VT;
  Vec4r sigma;
  std::function<void()> fn = [&]() { mochi::RotationVariantSvd3x3(A, U, sigma, VT); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
  state.counters["matrices/s"] =
      benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK_CAPTURE(RotationVariantSvd3x3, "RotationVariantSvd3x3_Identity", VEye<3>());
BENCHMARK_CAPTURE(
    RotationVariantSvd3x3,
    "RotationVariantSvd3x3_ArbitraryMatrix",
    GetArbitraryMatrix3x3());

static void BatchedAnalyticalEigendecompSym3x3(benchmark::State& state, VMatrix3x3r const& A) {
  NdArray<Matrix3x3r, kBS> mats;
  for (auto& m : mats) {
    m = ToNdArray3x3(A);
  }
  auto sym = mochi::test::LoadBatchSymMatrix3x3<kBS>(mats);
  V3 eigvals{};
  V3x3 eigvecs{};
  std::function<void()> fn = [&]() {
    mochi::BatchedAnalyticalEigendecompSym3x3<kBS>(sym, eigvals, &eigvecs);
  };
  for (auto _ : state) {
    CallNoInline(fn);
  }
  state.counters["matrices/s"] =
      benchmark::Counter(state.iterations() * kBS, benchmark::Counter::kIsRate);
}

BENCHMARK_CAPTURE(
    BatchedAnalyticalEigendecompSym3x3,
    "BatchedAnalyticalEigendecompSym3x3_Identity",
    VEye<3>());
BENCHMARK_CAPTURE(
    BatchedAnalyticalEigendecompSym3x3,
    "BatchedAnalyticalEigendecompSym3x3_ArbitraryMatrix",
    GetArbitraryMatrix3x3());

static void BatchedRotationVariantSvd3x3(benchmark::State& state, VMatrix3x3r const& A) {
  NdArray<Matrix3x3r, kBS> mats;
  for (auto& m : mats) {
    m = ToNdArray3x3(A);
  }
  auto fm = mochi::test::LoadBatchMatrix3x3<kBS>(MakeSpan(mats));
  V3x3 U{}, VT{};
  V3 sigma{};
  std::function<void()> fn = [&]() { mochi::BatchedRotationVariantSvd3x3<kBS>(fm, U, sigma, VT); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
  state.counters["matrices/s"] =
      benchmark::Counter(state.iterations() * kBS, benchmark::Counter::kIsRate);
}

BENCHMARK_CAPTURE(BatchedRotationVariantSvd3x3, "BatchedRotationVariantSvd3x3_Identity", VEye<3>());
BENCHMARK_CAPTURE(
    BatchedRotationVariantSvd3x3,
    "BatchedRotationVariantSvd3x3_ArbitraryMatrix",
    GetArbitraryMatrix3x3());

} // namespace mochi_benchmark
