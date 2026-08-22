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

#include <limits>

namespace mochi {

template <typename T>
constexpr T kMinusOne = T{0} - 1;
inline constexpr real kPI = 3.14159265358979323846_r;
inline constexpr real kRadiansPerDegree = kPI / 180_r;
inline constexpr real kDegreesPerRadian = 180_r / kPI;
inline constexpr real kSqrt2 = 1.41421356237309504880_r;
inline constexpr real kSqrtPi = 1.772453850905515881919427556567825376987457275390625_r;
inline constexpr real kSqrt3Over2 = 0.86602540378443864676_r;
inline constexpr real kInf = std::numeric_limits<real>::infinity();
inline constexpr Real2 kMinus2PiPlus2Pi = {-2_r * kPI, 2_r * kPI};
inline constexpr Real2 kUnitInterval = {0_r, 1_r};
inline constexpr Real3 kReal3Zeros = {0_r, 0_r, 0_r};
inline constexpr Real3 kReal3Ones = {1_r, 1_r, 1_r};
inline constexpr Real3 kReal3XAxis = {1_r, 0_r, 0_r};
inline constexpr Real3 kReal3YAxis = {0_r, 1_r, 0_r};
inline constexpr Real3 kReal3ZAxis = {0_r, 0_r, 1_r};
inline constexpr Real3 kInf3 = {kInf, kInf, kInf};

inline constexpr NdArray<real, 3, 3> kSecondOrderIdentity{
    Real3{1_r, 0_r, 0_r},
    Real3{0_r, 1_r, 0_r},
    Real3{0_r, 0_r, 1_r}};

// Fourth order identity tensor
inline constexpr auto FourthOrderEye() {
  NdArray<real, 3, 3, 3, 3> val = {};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      val[i][j][i][j] += 1_r;
    }
  }
  return val;
}
static constexpr NdArray<real, 3, 3, 3, 3> kFourthOrderEye = FourthOrderEye();

inline constexpr auto FourthOrderEyeSym() {
  NdArray<real, 3, 3, 3, 3> val = {};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          val[i][j][k][l] = 1_r / 2_r *
              (static_cast<real>(i == k) * static_cast<real>(j == l) +
               static_cast<real>(i == l) * static_cast<real>(j == k));
        }
      }
    }
  }
  return val;
}
static constexpr NdArray<real, 3, 3, 3, 3> kFourthOrderEyeSym = FourthOrderEyeSym();

inline constexpr auto IouterI() {
  NdArray<real, 3, 3, 3, 3> val = {};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          val[i][j][k][l] = static_cast<real>(i == j) * static_cast<real>(k == l);
        }
      }
    }
  }
  return val;
}
static constexpr NdArray<real, 3, 3, 3, 3> kIouterI = IouterI();

// Indices of face nodes within a parent tetrahedron. Note that 3-faceNum is excluded from each set
// and the winding order ensures outward normals.
// This ordering is used in two different processes:
// - Create a boundary discretization of a finite-element tetrahedral mesh.
// - Extract a triangle mesh from a tetrahedral mesh.
struct TetFaces {
  static constexpr NdArray<int, 4, 3> kIndices = {
      Int3{0, 2, 1},
      Int3{0, 1, 3},
      Int3{0, 3, 2},
      Int3{1, 2, 3}};
};

// This is intended to be used as a placeholder index in linear algebra data structures.
constexpr int kSentinelIndex = -1;

} // namespace mochi
