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

namespace mochi_benchmark {

// Matrices A and B are passed to this non-inlined function so that the optimizer can't presume to
// know their initial values.
MOCHI_NO_INLINE static void
Dot3x3(benchmark::State& state, mochi::VMatrix3x3r const& A0, mochi::VMatrix3x3r const& B0) {
  mochi::VMatrix3x3r A = A0;
  mochi::VMatrix3x3r B = B0;
  for (auto _ : state) {
    B = mochi::Dot3x3(A, B);
    MOCHI_NO_DISCARD_IN_LOOP(B);
  }
  benchmark::DoNotOptimize(B);
}
BENCHMARK_CAPTURE(Dot3x3, Dot3x3, {}, {});

MOCHI_NO_INLINE static void Det3x3(benchmark::State& state, mochi::VMatrix3x3r const& A0) {
  mochi::VMatrix3x3r A = A0;
  mochi::real det = 0;
  for (auto _ : state) {
    // Use += to prevent the optimizer from throwing anything out.
    det += mochi::Det3x3(A);
    MOCHI_NO_DISCARD_IN_LOOP(det);
  }
  benchmark::DoNotOptimize(det);
}
BENCHMARK_CAPTURE(Det3x3, Det3x3, mochi::VEye<3>());

} // namespace mochi_benchmark
