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

// Shared low-level material-test utilities: tet deformation helpers, deformation-gradient
// datasets, PSD checks, comparison helpers.

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/materials/material_traits.h>
#include <mochi_core/materials/material_types.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/vmatrix.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <utility>

#if MOCHI_USE_EIGEN
MOCHI_WARNING_PUSH()
MOCHI_WARNING_SUPPRESS_EIGEN()
#include <Eigen/Dense>
MOCHI_WARNING_POP()
#endif

namespace mochi {

// PSD strategies that support analytic projection.
inline constexpr auto kAnalyticPsdProjectionStrategies =
    std::array{MaterialPsdStrategy::Projection, MaterialPsdStrategy::AbsEigenProjection};

namespace details {
template <typename ParamsType>
[[nodiscard]] constexpr int CountSupportedPsdStrategies() {
  static_assert(
      // Please update API documentation if this list changes.
      materials::IsPsdStrategySupported<ParamsType>(MaterialPsdStrategy::MaterialDefault) &&
          materials::IsPsdStrategySupported<ParamsType>(MaterialPsdStrategy::None) &&
          materials::IsPsdStrategySupported<ParamsType>(MaterialPsdStrategy::Projection) &&
          materials::IsPsdStrategySupported<ParamsType>(MaterialPsdStrategy::AbsEigenProjection),
      "MaterialDefault, None, Projection and AbsEigenProjection must be supported by all material models.");
  int n = 0;
  for (int i = 0; i < static_cast<int>(MaterialPsdStrategy::Count); ++i) {
    if (materials::IsPsdStrategySupported<ParamsType>(static_cast<MaterialPsdStrategy>(i))) {
      ++n;
    }
  }
  return n;
}
} // namespace details

// Collect the supported PSD strategies for a material model into a compile-time array.
template <typename ParamsType, int N = details::CountSupportedPsdStrategies<ParamsType>()>
[[nodiscard]] constexpr std::array<MaterialPsdStrategy, N> GetSupportedPsdStrategies() {
  std::array<MaterialPsdStrategy, N> result{};
  int idx = 0;
  for (int i = 0; i < static_cast<int>(MaterialPsdStrategy::Count); ++i) {
    auto const s = static_cast<MaterialPsdStrategy>(i);
    if (materials::IsPsdStrategySupported<ParamsType>(s)) {
      result[idx++] = s;
    }
  }
  return result;
}

Matrix<real, 3, 3> ComputeDmInv(
    ColumnVector<real, 3> const& V0,
    ColumnVector<real, 3> const& V1,
    ColumnVector<real, 3> const& V2,
    ColumnVector<real, 3> const& V3);

Matrix<real, 3, 3> ComputeF(
    ColumnVector<real, 3> const& v0,
    ColumnVector<real, 3> const& v1,
    ColumnVector<real, 3> const& v2,
    ColumnVector<real, 3> const& v3,
    Matrix<real, 3, 3> const& dmInv);

VMatrix3x3r ToSimdMatrix(RowMatrix<real, 3, 3> const& mat);

template <typename Generator>
[[nodiscard]] Matrix<real, 3, 3> GetRandomRotationMatrix(Generator& generator) {
  real const angle0 = RandomUniformValue(generator, -kPI, kPI);
  real const angle1 = RandomUniformValue(generator, -kPI, kPI);
  real const angle2 = RandomUniformValue(generator, -kPI, kPI);

  RowMatrix<real, 3, 3> pitch = {
      {1_r, 0_r, 0_r}, {0_r, Cos(angle0), Sin(angle0)}, {0_r, -Sin(angle0), Cos(angle0)}};

  RowMatrix<real, 3, 3> yaw = {
      {Cos(angle1), 0_r, -Sin(angle1)}, {0_r, 1_r, 0_r}, {Sin(angle1), 0_r, Cos(angle1)}};

  RowMatrix<real, 3, 3> roll = {
      {Cos(angle2), Sin(angle2), 0_r}, {-Sin(angle2), Cos(angle2), 0_r}, {0_r, 0_r, 1_r}};

  return pitch * yaw * roll;
}

// Curated deformation gradient test cases covering identity, random, isochoric, symmetric,
// dilation, contraction, shear, and inversion.
inline constexpr Matrix3x3r kTestDeformations[] = {
    // Identity
    Matrix3x3r{Real3{1_r, 0_r, 0_r}, Real3{0_r, 1_r, 0_r}, Real3{0_r, 0_r, 1_r}},
    // Purely random
    Matrix3x3r{
        Real3{1.757904e+00_r, 8.375675e-01_r, 8.940403e-01_r},
        Real3{7.781580e-01_r, 1.293040e+00_r, 8.082672e-01_r},
        Real3{9.259674e-01_r, 3.419398e-01_r, 1.178375e+00_r}},
    // Isochoric
    Matrix3x3r{
        Real3{1.852487e+00_r, 3.954069e-01_r, 8.651419e-01_r},
        Real3{1.217431e-01_r, 1.074060e+00_r, 2.034998e-01_r},
        Real3{1.669658e-01_r, 6.375304e-01_r, 1.053620e+00_r}},
    // Symmetric
    Matrix3x3r{
        Real3{3.474406e+00_r, 9.696913e-01_r, 1.803357e+00_r},
        Real3{9.696913e-01_r, 1.716397e+00_r, 1.232369e+00_r},
        Real3{1.803357e+00_r, 1.232369e+00_r, 1.899997e+00_r}},
    // Pure dilation
    Matrix3x3r{Real3{4_r, 0_r, 0_r}, Real3{0_r, 4_r, 0_r}, Real3{0_r, 0_r, 4_r}},
    // Pure contraction
    Matrix3x3r{Real3{0.25_r, 0_r, 0_r}, Real3{0_r, 0.25_r, 0_r}, Real3{0_r, 0_r, 0.25_r}},
    // Shear
    Matrix3x3r{Real3{1_r, 0.5_r, 0_r}, Real3{0.5_r, 1_r, 0_r}, Real3{0_r, 0_r, 1_r}},
    // Inverted (det < 0)
    Matrix3x3r{Real3{-0.5_r, 0_r, 0_r}, Real3{0_r, 1_r, 0_r}, Real3{0_r, 0_r, 1_r}}};

namespace details {

inline constexpr real kRandomDeformationMinAbsStretch = 0.05_r;
inline constexpr real kRandomDeformationMinAbsStretchSeparation = 0.1_r;

[[nodiscard]] inline bool AreRandomDeformationStretchesSeparated(Real3 sigma) {
  for (int i = 0; i < 3; ++i) {
    for (int j = i + 1; j < 3; ++j) {
      if (Abs(sigma[i] - sigma[j]) < kRandomDeformationMinAbsStretchSeparation) {
        return false;
      }
    }
  }
  return true;
}

template <typename Generator>
void RandomlyPermuteRandomDeformationStretches(Generator& generator, Real3& sigma) {
  for (int i = 2; i > 0; --i) {
    int const j = RandomUniformValue(generator, 0, i);
    std::swap(sigma[i], sigma[j]);
  }
}

template <typename Generator>
void RandomlyInvertRandomDeformation(Generator& generator, Real3& sigma) {
  if (RandomUniformValue(generator, 0, 1) == 1) {
    sigma[RandomUniformValue(generator, 0, 2)] *= -1_r;
  }
}

template <typename Generator>
[[nodiscard]] Matrix3x3r MakeRandomDeformationGradient(
    Generator& generator,
    real maxAbsStretch,
    bool separateSingularValues = false) {
  MOCHI_ASSERT(maxAbsStretch >= kRandomDeformationMinAbsStretch);
  MOCHI_ASSERT(
      !separateSingularValues ||
      maxAbsStretch - kRandomDeformationMinAbsStretch >=
          2_r * kRandomDeformationMinAbsStretchSeparation);
  Real3 sigma{};
  if (separateSingularValues) {
    // Sample three stretches, sort them, then add monotonically increasing offsets (0, +sep,
    // +2*sep). After sorting, each adjacent pair differs by at least
    // kRandomDeformationMinAbsStretchSeparation, so every pair of stretches is separated.
    // maxShiftedStretch reserves headroom (2*sep) so the largest shifted value stays within
    // maxAbsStretch. The final permutation removes the sorted ordering.
    real const maxShiftedStretch = maxAbsStretch - 2_r * kRandomDeformationMinAbsStretchSeparation;
    for (int i = 0; i < 3; ++i) {
      sigma[i] = RandomUniformValue(generator, kRandomDeformationMinAbsStretch, maxShiftedStretch);
    }
    std::ranges::sort(sigma);
    sigma[1] += kRandomDeformationMinAbsStretchSeparation;
    sigma[2] += 2_r * kRandomDeformationMinAbsStretchSeparation;
    RandomlyPermuteRandomDeformationStretches(generator, sigma);
  } else {
    for (int i = 0; i < 3; ++i) {
      sigma[i] = RandomUniformValue(generator, kRandomDeformationMinAbsStretch, maxAbsStretch);
    }
  }
  RandomlyInvertRandomDeformation(generator, sigma);

  auto const U = GetRandomRotationMatrix(generator);
  auto const V = GetRandomRotationMatrix(generator);
  auto USigma = U;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      USigma(i, j) *= sigma[j];
    }
  }
  Matrix<real, 3, 3> const out = USigma * V.Transpose();
  return ToNdArray3x3(ToSimdMatrix(out));
}

} // namespace details

// Generate a test set of deformation gradients: curated cases + random.
inline DynamicArray<Matrix3x3r>
GenerateTestSet(int randomCases, real maxRandomStretch, bool separateRandomSingularValues = false) {
  DynamicArray<Matrix3x3r> testSet;
  testSet.reserve(randomCases + std::size(kTestDeformations));
  for (auto const& test : kTestDeformations) {
    testSet.push_back(test);
  }
  auto generator = RandomGenerator(42);
  for (int i = 0; i < randomCases; ++i) {
    testSet.push_back(
        details::MakeRandomDeformationGradient(
            generator, maxRandomStretch, separateRandomSingularValues));
  }
  return testSet;
}

#if MOCHI_USE_EIGEN
// Reference PSD projection of a 3x3x3x3 tangent tensor via Eigen eigendecomposition.
// Supports Projection (Max(λ, ε)) and AbsEigenProjection (Max(|λ|, ε)).
inline void EigenProjectPsd(NdArray<real, 3, 3, 3, 3>& tensor, MaterialPsdStrategy psdStrategy) {
  constexpr real kEps = std::numeric_limits<real>::epsilon();
  using EMatType = Eigen::Matrix<real, 9, 9, Eigen::RowMajor | Eigen::DontAlign>;
  auto eigenMat = Eigen::Map<EMatType>(&tensor[0][0][0][0]);
  auto eigenSolver = Eigen::SelfAdjointEigenSolver<EMatType>(eigenMat);
  auto projectedEigenvals = eigenSolver.eigenvalues().unaryExpr([&](real f) {
    if (psdStrategy == MaterialPsdStrategy::Projection) {
      return Max(f, kEps);
    } else {
      EXPECT_EQ(MaterialPsdStrategy::AbsEigenProjection, psdStrategy);
      return Max(Abs(f), kEps);
    }
  });
  eigenMat = eigenSolver.eigenvectors() * projectedEigenvals.asDiagonal() *
      eigenSolver.eigenvectors().transpose();
}

template <typename EigenMatT>
[[nodiscard]] real GetPsdProjectionEigenvalueTol(EigenMatT const& eigenMat) {
  constexpr real kScale = MOCHI_USE_DOUBLE_PRECISION ? 300_r : 100_r;
  return kScale * std::numeric_limits<real>::epsilon() * eigenMat.norm();
}
#endif // MOCHI_USE_EIGEN

} // namespace mochi
