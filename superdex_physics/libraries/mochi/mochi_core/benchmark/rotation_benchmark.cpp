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

#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/rodrigues_utils.h>

#include <functional>

using namespace mochi;

namespace mochi_benchmark {

static Vec4r GetRotVec() {
  return {0.5_r, 0.5_r, 0.5_r};
}

static Quaternion GetQuaternion() {
  return Quaternion::FromRotationVector(GetRotVec());
}

static VMatrix3x3r GetVMatrix3x3() {
  return ToVMatrix3x3(GetQuaternion());
}

template <bool kTranspose>
static void QuaternionToVMatrix3x3Impl(benchmark::State& state) {
  Quaternion q = GetQuaternion();
  VMatrix3x3r R = {};
  std::function<void()> fn = [&]() {
    if constexpr (kTranspose) {
      R = ToVMatrix3x3Transpose(q);
    } else {
      R = ToVMatrix3x3(q);
    }
  };
  for (auto _ : state) {
    CallNoInline(fn);
  }
  benchmark::DoNotOptimize(R);
}

static void QuaternionToVMatrix3x3(benchmark::State& state) {
  QuaternionToVMatrix3x3Impl<false>(state);
}

static void QuaternionToVMatrix3x3Transpose(benchmark::State& state) {
  QuaternionToVMatrix3x3Impl<true>(state);
}

static void RotationVectorToVMatrix3x3(benchmark::State& state) {
  Vec4r rotVec = GetRotVec();
  VMatrix3x3r R = {};
  std::function<void()> fn = [&]() { R = Rodrigues(rotVec); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
  benchmark::DoNotOptimize(R);
}

static void VMatrix3x3ToQuaternion(benchmark::State& state) {
  VMatrix3x3r R = GetVMatrix3x3();
  Quaternion q = {};
  std::function<void()> fn = [&]() { q = QuaternionFromMatrix(R); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
  benchmark::DoNotOptimize(q);
}

static void RotationVectorToQuaternion(benchmark::State& state) {
  Vec4r rotVec = GetRotVec();
  Quaternion q = {};
  std::function<void()> fn = [&]() { q = Quaternion::FromRotationVector(rotVec); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
  benchmark::DoNotOptimize(q);
}

static void QuaternionToRotationVector(benchmark::State& state) {
  Quaternion q = GetQuaternion();
  Vec4r rotVec = {};
  std::function<void()> fn = [&]() { rotVec = q.VToRotationVector(); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
  benchmark::DoNotOptimize(rotVec);
}

static void VMatrix3x3ToRotationVector(benchmark::State& state) {
  VMatrix3x3r R = GetVMatrix3x3();
  Vec4r rotVec = {};
  std::function<void()> fn = [&]() { rotVec = InvRodrigues(R); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
  benchmark::DoNotOptimize(rotVec);
}

// Timings include function call overhead.
BENCHMARK(QuaternionToVMatrix3x3);
BENCHMARK(QuaternionToVMatrix3x3Transpose);
BENCHMARK(RotationVectorToVMatrix3x3);
BENCHMARK(VMatrix3x3ToQuaternion);
BENCHMARK(RotationVectorToQuaternion);
BENCHMARK(QuaternionToRotationVector);
BENCHMARK(VMatrix3x3ToRotationVector);

static void DRotVectorDRotIncrement(benchmark::State& state) {
  Vec4r rotVec = GetRotVec();
  VMatrix3x3r DRotDinc = {};
  std::function<void()> fn = [&]() { DRotDinc = DRotVectorDRotIncrement(rotVec); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
}

static void DRotIncrementDRotVector(benchmark::State& state) {
  Vec4r rotVec = GetRotVec();
  VMatrix3x3r DincDrot = {};
  std::function<void()> fn = [&]() { DincDrot = DRotIncrementDRotVector(rotVec); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
}

// Timings include function call overhead.
BENCHMARK(DRotVectorDRotIncrement);
BENCHMARK(DRotIncrementDRotVector);

} // namespace mochi_benchmark
