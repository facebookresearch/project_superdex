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

#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>

/* Values from a seed, such as a @ref Derive sub-seed: SplitMix64 with uniform, normal and
 * uniform-rotation draws. Each draw advances the state it is given. */

namespace superdex::robotics {

/* @brief Advance @p state and return its next SplitMix64 output. */
inline uint64_t SplitMix64(uint64_t& state) {
  uint64_t z = (state += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

/* @brief A uniform draw from [@p lo, @p hi). */
inline double Uniform(uint64_t& state, double lo, double hi) {
  double const unit = static_cast<double>(SplitMix64(state) >> 11) * 0x1.0p-53;
  return lo + (hi - lo) * unit;
}

/* @brief A standard normal draw (Box-Muller). */
inline double Normal(uint64_t& state) {
  double const u = 1.0 - Uniform(state, 0.0, 1.0);
  double const v = Uniform(state, 0.0, 1.0);
  constexpr double kPi = std::numbers::pi;
  return std::sqrt(-2.0 * std::log(u)) * std::cos(2.0 * kPi * v);
}

/* @brief A rotation uniformly random on SO(3) from three uniform draws (Shoemake's method), as a
 * unit quaternion (x, y, z, w). */
inline std::array<double, 4> UniformQuat(uint64_t& state) {
  constexpr double kTwoPi = 2.0 * std::numbers::pi;
  double const u1 = Uniform(state, 0.0, 1.0);
  double const u2 = Uniform(state, 0.0, 1.0);
  double const u3 = Uniform(state, 0.0, 1.0);
  double const a = std::sqrt(1.0 - u1);
  double const b = std::sqrt(u1);
  return {
      a * std::sin(kTwoPi * u2),
      a * std::cos(kTwoPi * u2),
      b * std::sin(kTwoPi * u3),
      b * std::cos(kTwoPi * u3)};
}

} // namespace superdex::robotics
