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

#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/dynamic_array.h>

namespace mochi_benchmark {

static void ArrayTransformPoints(benchmark::State& state, int numPoints) {
  using namespace mochi;
  DynamicArray<Real3> src, dst;
  src.resize(numPoints);
  dst.resize(numPoints);
  auto srcSpan = MakeConstSpan(src);
  auto dstSpan = MakeSpan(dst);

  // Arbitrary non-idenity transform
  TransformSRT transform{0.5_r, Quaternion::RotationX(0.5_r * kPI), Real3{1_r, 2_r, 3_r}};

  for (auto _ : state) {
    CallNoInline([&]() { ArrayTransformPoints<true>(dstSpan, srcSpan, transform); });
  }
}

BENCHMARK_CAPTURE(ArrayTransformPoints, ArrayTransformPoints_10, 10)
    ->Name("ArrayUtils/TransformPoints/Count10");
BENCHMARK_CAPTURE(ArrayTransformPoints, ArrayTransformPoints_50, 50)
    ->Name("ArrayUtils/TransformPoints/Count50");
BENCHMARK_CAPTURE(ArrayTransformPoints, ArrayTransformPoints_100, 100)
    ->Name("ArrayUtils/TransformPoints/Count100");
BENCHMARK_CAPTURE(ArrayTransformPoints, ArrayTransformPoints_1000, 1000)
    ->Name("ArrayUtils/TransformPoints/Count1000");
BENCHMARK_CAPTURE(ArrayTransformPoints, ArrayTransformPoints_10000, 10000)
    ->Name("ArrayUtils/TransformPoints/Count10000");
BENCHMARK_CAPTURE(ArrayTransformPoints, ArrayTransformPoints_100000, 100000)
    ->Name("ArrayUtils/TransformPoints/Count100000");

static void ArrayTransformDisplacements(benchmark::State& state, int numPoints) {
  using namespace mochi;
  DynamicArray<Real3> srcRef, srcDisp, dst;
  srcRef.resize(numPoints);
  srcDisp.resize(numPoints);
  dst.resize(numPoints);
  auto srcRefSpan = MakeConstSpan(srcRef);
  auto srcDispSpan = MakeConstSpan(srcDisp);
  auto dstSpan = MakeSpan(dst);

  // Arbitrary non-idenity transform
  TransformRT transform{Quaternion::RotationX(0.5_r * kPI), Real3{1_r, 2_r, 3_r}};

  for (auto _ : state) {
    CallNoInline(
        [&]() { ArrayTransformDisplacements<true>(dstSpan, srcRefSpan, srcDispSpan, transform); });
  }
}

BENCHMARK_CAPTURE(ArrayTransformDisplacements, ArrayTransformDisplacements_10, 10)
    ->Name("ArrayUtils/TransformDisplacements/Count10");
BENCHMARK_CAPTURE(ArrayTransformDisplacements, ArrayTransformDisplacements_50, 50)
    ->Name("ArrayUtils/TransformDisplacements/Count50");
BENCHMARK_CAPTURE(ArrayTransformDisplacements, ArrayTransformDisplacements_100, 100)
    ->Name("ArrayUtils/TransformDisplacements/Count100");
BENCHMARK_CAPTURE(ArrayTransformDisplacements, ArrayTransformDisplacements_1000, 1000)
    ->Name("ArrayUtils/TransformDisplacements/Count1000");
BENCHMARK_CAPTURE(ArrayTransformDisplacements, ArrayTransformDisplacements_10000, 10000)
    ->Name("ArrayUtils/TransformDisplacements/Count10000");
BENCHMARK_CAPTURE(ArrayTransformDisplacements, ArrayTransformDisplacements_100000, 100000)
    ->Name("ArrayUtils/TransformDisplacements/Count100000");

static void ArrayRotateVectors(benchmark::State& state, int numPoints) {
  using namespace mochi;
  DynamicArray<Real3> src, dst;
  src.resize(numPoints);
  dst.resize(numPoints);
  auto srcSpan = MakeConstSpan(src);
  auto dstSpan = MakeSpan(dst);

  // Arbitrary non-idenity rotation
  auto rot = Quaternion::RotationX(0.5_r * kPI);

  for (auto _ : state) {
    CallNoInline([&]() { ArrayRotateVectors<true>(dstSpan, srcSpan, rot); });
  }
}

BENCHMARK_CAPTURE(ArrayRotateVectors, ArrayRotateVectors_10, 10)
    ->Name("ArrayUtils/RotateVectors/Count10");
BENCHMARK_CAPTURE(ArrayRotateVectors, ArrayRotateVectors_50, 50)
    ->Name("ArrayUtils/RotateVectors/Count50");
BENCHMARK_CAPTURE(ArrayRotateVectors, ArrayRotateVectors_100, 100)
    ->Name("ArrayUtils/RotateVectors/Count100");
BENCHMARK_CAPTURE(ArrayRotateVectors, ArrayRotateVectors_1000, 1000)
    ->Name("ArrayUtils/RotateVectors/Count1000");
BENCHMARK_CAPTURE(ArrayRotateVectors, ArrayRotateVectors_10000, 10000)
    ->Name("ArrayUtils/RotateVectors/Count10000");
BENCHMARK_CAPTURE(ArrayRotateVectors, ArrayRotateVectors_100000, 100000)
    ->Name("ArrayUtils/RotateVectors/Count100000");

static void Fill(benchmark::State& state, int numPoints) {
  using namespace mochi;
  DynamicArray<Real3> arr(numPoints);
  auto span = MakeSpan(arr);
  constexpr auto kValue = Real3{1_r, 2_r, 3_r};
  for (auto _ : state) {
    CallNoInline([&]() { Fill(span, kValue); });
  }
}

BENCHMARK_CAPTURE(Fill, Real3[10], 10)->Name("ArrayUtils/Fill/Real3/Count10");
BENCHMARK_CAPTURE(Fill, Real3[50], 50)->Name("ArrayUtils/Fill/Real3/Count50");
BENCHMARK_CAPTURE(Fill, Real3[100], 100)->Name("ArrayUtils/Fill/Real3/Count100");
BENCHMARK_CAPTURE(Fill, Real3[1000], 1000)->Name("ArrayUtils/Fill/Real3/Count1000");
BENCHMARK_CAPTURE(Fill, Real3[10000], 10000)->Name("ArrayUtils/Fill/Real3/Count10000");
BENCHMARK_CAPTURE(Fill, Real3[100000], 100000)->Name("ArrayUtils/Fill/Real3/Count100000");

} // namespace mochi_benchmark
