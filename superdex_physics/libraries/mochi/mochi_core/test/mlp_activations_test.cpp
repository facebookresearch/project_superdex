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

#include <mochi_core/ai/mlp.h>
#include <mochi_core/test/mochi_test_helpers.h>

#include <cmath>
#include <type_traits>
#include <vector>

using namespace mochi;

// TODO(T186384485): Extend test coverage to cover all specializations.

constexpr real kTolerance = std::is_same_v<real, float> ? 1e-3 : 1e-10;

template <typename ActivationType>
void ActivationTestImpl(
    ActivationType f,
    std::vector<real> const& z,
    std::vector<real> const& gold_z,
    std::vector<real> const& gold_dfdz) {
  MOCHI_ASSERT_VERBOSE((z.size() == gold_z.size()) && (z.size() == gold_dfdz.size()));
  for (int i = 0; i < z.size(); ++i) {
    {
      auto curr_z = z[i];
      f(curr_z); // modifies in place
      EXPECT_NEAR(curr_z, gold_z[i], kTolerance);
    }
    {
      auto curr_z = z[i];
      real dfdz = 0_r;
      f(curr_z, dfdz); // modifies in place
      EXPECT_NEAR(curr_z, gold_z[i], kTolerance);
      EXPECT_NEAR(dfdz, gold_dfdz[i], kTolerance);
    }
  }
}

template <
    int kRowsAtCompile,
    int kColsAtCompile,
    krylov::Direction kMajorDir,
    int kLeadDim = krylov::kAutomaticLeadDim,
    typename ActivationFunctor>
void ActivationSIMDApply(ActivationFunctor f) {
  static_assert((kRowsAtCompile > 0) && (kColsAtCompile > 0) && (kLeadDim >= 0));
  constexpr int kC = (kLeadDim > 0) ? krylov::kDynamic : kColsAtCompile;
  constexpr int kR = (kLeadDim > 0) ? krylov::kDynamic : kRowsAtCompile;
  constexpr int kLD = (kLeadDim > 0) ? krylov::kDynamic : kLeadDim;
  using MatType = Matrix<real, kR, kC, kMajorDir, krylov::Ownership::Owner, kLD>;
  int newLD = kLeadDim;
  if (kLeadDim == krylov::kAutomaticLeadDim) {
    newLD = (kMajorDir == krylov::Direction::ColMajor) ? kRowsAtCompile : kColsAtCompile;
  }
  MatType w(kRowsAtCompile, kColsAtCompile, newLD);
  w.SetRandom(123, -2.3_r, 2.0_r);
  MatType w0(w), gold_w(w);
  for (int i = 0; i < gold_w.Rows(); ++i) {
    for (int j = 0; j < gold_w.Cols(); ++j) {
      f(gold_w(i, j));
    }
  }
  ai::details::ActivationInPlace(f, w);
  EXPECT_TRUE(test::NearEqualMatrices(w, gold_w));
  //
  w = w0;
  MatType dfdw(w0), gold_dfdw(w0);
  gold_w = w0;
  for (int i = 0; i < w0.Rows(); ++i) {
    for (int j = 0; j < w0.Cols(); ++j) {
      f(gold_w(i, j), gold_dfdw(i, j));
    }
  }
  ai::details::ActivationInPlace(f, w, dfdw);
  EXPECT_TRUE(test::NearEqualMatrices(w, gold_w));
  EXPECT_TRUE(test::NearEqualMatrices(dfdw, gold_dfdw));
}

TEST(MlpActivation, Identity) {
  ai::IdentityActivation<real> f;
  std::vector<real> z{-1.2_r, 0_r, 2_r};
  std::vector<real> gold_z{-1.2_r, 0_r, 2_r};
  std::vector<real> gold_dfdz{1_r, 1_r, 1_r};
  ActivationTestImpl(f, z, gold_z, gold_dfdz);
  //
  ActivationSIMDApply<19, 3, krylov::Direction::ColMajor>(f);
  ActivationSIMDApply<7, 17, krylov::Direction::RowMajor>(f);
  ActivationSIMDApply<19, 3, krylov::Direction::ColMajor, 32>(f);
  ActivationSIMDApply<7, 17, krylov::Direction::RowMajor, 32>(f);
}

TEST(MlpActivation, ELU) {
  for (real alpha : {-0.5_r, 0_r, 2_r}) {
    ai::ELUActivation<real> f{alpha};
    std::vector<real> z{-1.2_r, 0_r, 2_r};
    std::vector<real> gold_z{alpha * std::exp(-1.2_r) - alpha, 0_r, 2_r};
    std::vector<real> gold_dfdz{alpha * std::exp(-1.2_r), 1_r, 1_r};
    ActivationTestImpl(f, z, gold_z, gold_dfdz);
    //
    ActivationSIMDApply<19, 3, krylov::Direction::ColMajor>(f);
    ActivationSIMDApply<7, 17, krylov::Direction::RowMajor>(f);
    ActivationSIMDApply<19, 3, krylov::Direction::ColMajor, 32>(f);
    ActivationSIMDApply<7, 17, krylov::Direction::RowMajor, 32>(f);
  }
}

TEST(MlpActivation, ReLU) {
  ai::ReLUActivation<real> f;
  std::vector<real> z{-1.2_r, 0_r, 2_r};
  std::vector<real> gold_z{0_r, 0_r, 2_r};
  std::vector<real> gold_dfdz{0_r, 1_r, 1_r};
  ActivationTestImpl(f, z, gold_z, gold_dfdz);
  //
  ActivationSIMDApply<19, 3, krylov::Direction::ColMajor>(f);
  ActivationSIMDApply<7, 17, krylov::Direction::RowMajor>(f);
  ActivationSIMDApply<19, 3, krylov::Direction::ColMajor, 32>(f);
  ActivationSIMDApply<7, 17, krylov::Direction::RowMajor, 32>(f);
}
