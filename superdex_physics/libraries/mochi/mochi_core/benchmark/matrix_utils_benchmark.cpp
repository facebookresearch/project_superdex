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

#include <mochi_core/utils/matrix_utils.h>

#include <functional>

using namespace mochi;

namespace mochi_benchmark {

static void DotMatVec3x3(benchmark::State& state) {
  VMatrix3x3r A = {};
  Vec4r x = {}, b = {};
  std::function<void()> fn = [&]() { b = mochi::DotMatVec3x3(A, x); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
}

BENCHMARK(DotMatVec3x3); // Timings include function call overhead

static void DotVecMat3x3(benchmark::State& state) {
  VMatrix3x3r A = {};
  Vec4r x = {}, b = {};
  std::function<void()> fn = [&]() { b = mochi::DotVecMat3x3(x, A); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
}

BENCHMARK(DotVecMat3x3); // Timings include function call overhead

static void DotVecMat3x3WithTranspose(benchmark::State& state) {
  VMatrix3x3r A = {};
  Vec4r x = {}, b = {};
  std::function<void()> fn = [&]() { b = mochi::DotVecMat3x3(x, Transpose3x3(A)); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
}

BENCHMARK(DotVecMat3x3WithTranspose); // Timings include function call overhead

} // namespace mochi_benchmark
