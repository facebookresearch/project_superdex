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

#include <mochi_core/utils/nd_array.h>
#include <numbers>

namespace mochi::segment {

// Gaussian quadrature rules on the unit interval [0,1].

template <int kNumQuadPoints>
struct SegmentQuadrature final {};

template <>
struct SegmentQuadrature<1> final {
  static constexpr size_t kNumQuadPoints = 1;
  static constexpr NdArray<real, 1, 1> points = {Real1{1_r / 2_r}};
  static constexpr NdArray<real, 1> weights = {1_r};
};

// Using compile-time arithmetic to transform from (-1,1) expressions for points/weights found in
// standard references on numerical analysis.
template <>
struct SegmentQuadrature<2> final {
  static constexpr size_t kNumQuadPoints = 2;
  static constexpr real kInvSqrt3 = std::numbers::inv_sqrt3_v<real>;
  static constexpr NdArray<real, 2, 1> points = {
      Real1{0.5_r * (1_r - kInvSqrt3)},
      Real1{0.5_r * (1_r + kInvSqrt3)}};
  static constexpr NdArray<real, 2> weights = {0.5_r, 0.5_r};
};

template <>
struct SegmentQuadrature<3> final {
  static constexpr size_t kNumQuadPoints = 3;
  // std::sqrt is not constexpr in C++20; precomputing as literal
  static constexpr real kSqrt3Over5 = 0.7745966692414834_r;
  static constexpr NdArray<real, 3, 1> points = {
      Real1{0.5_r * (1_r - kSqrt3Over5)},
      Real1{0.5_r},
      Real1{0.5_r * (1_r + kSqrt3Over5)}};
  static constexpr NdArray<real, 3> weights = {2.5_r / 9_r, 4_r / 9_r, 2.5_r / 9_r};
};

} // namespace mochi::segment
