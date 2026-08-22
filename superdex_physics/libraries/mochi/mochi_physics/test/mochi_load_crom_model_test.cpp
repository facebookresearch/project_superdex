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

#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/src/mochi_hdf5.h>
#include "mochi_physics_test_fixture.h"

#include <type_traits>

using namespace mochi;

constexpr real kTolerance = std::is_same_v<real, float> ? 1e-4 : 1e-13;

template <typename AT, typename BT>
static void AssertNearMatrix(AT const& a, BT const& b) {
  ASSERT_TRUE(a.Rows() == b.Rows()) << a.Rows() << " " << b.Rows() << '\n';
  ASSERT_TRUE(a.Cols() == b.Cols()) << a.Cols() << " " << b.Cols() << '\n';

  for (int i = 0; i < a.Rows(); ++i) {
    for (int j = 0; j < a.Cols(); ++j) {
      ASSERT_NEAR(a(i, j), b(i, j), kTolerance) << "i = " << i << " , j = " << j << '\n';
    }
  }
}

static void AssertVerifyLayerIndex0(ai::Mlp<real>& m) {
  auto const& layer = m.GetLayer(0);

  // weight
  auto const W = layer.WeightsConstView();
  ASSERT_TRUE(W.Rows() == 6);
  ASSERT_TRUE(W.Cols() == 5);
  Matrix<real> gold_W(6, 5);
  gold_W(0, 0) = 0.01_r;
  gold_W(0, 1) = 0.02_r;
  gold_W(0, 2) = 0.015_r;
  gold_W(0, 3) = -0.016_r;
  gold_W(0, 4) = 0.0125_r;

  gold_W(1, 0) = 0.06_r;
  gold_W(1, 1) = 0.07_r;
  gold_W(1, 2) = 0.04_r;
  gold_W(1, 3) = -0.036_r;
  gold_W(1, 4) = 0.025_r;

  gold_W(2, 0) = 0.11_r;
  gold_W(2, 1) = 0.12_r;
  gold_W(2, 2) = 0.065_r;
  gold_W(2, 3) = -0.056_r;
  gold_W(2, 4) = 0.0375_r;

  gold_W(3, 0) = 0.16_r;
  gold_W(3, 1) = 0.17_r;
  gold_W(3, 2) = 0.09_r;
  gold_W(3, 3) = -0.076_r;
  gold_W(3, 4) = 0.05_r;

  gold_W(4, 0) = 0.21_r;
  gold_W(4, 1) = 0.22_r;
  gold_W(4, 2) = 0.115_r;
  gold_W(4, 3) = -0.096_r;
  gold_W(4, 4) = 0.0625_r;

  gold_W(5, 0) = 0.26_r;
  gold_W(5, 1) = 0.27_r;
  gold_W(5, 2) = 0.14_r;
  gold_W(5, 3) = -0.116_r;
  gold_W(5, 4) = 0.075_r;

  AssertNearMatrix(gold_W, W);

  // bias
  auto const b = layer.BiasConstView();
  ASSERT_TRUE(b.size() == 6);
  ColumnVector<real> gold_b(6);
  gold_b(0) = 0.923_r;
  gold_b(1) = 1.833_r;
  gold_b(2) = 2.743_r;
  gold_b(3) = 3.653_r;
  gold_b(4) = 4.563_r;
  gold_b(5) = 5.473_r;
  AssertNearMatrix(gold_b, b);
}

static void AssertVerifyLayerIndex1(ai::Mlp<real>& m) {
  auto const& layer = m.GetLayer(1);

  // weight
  auto const W = layer.WeightsConstView();
  ASSERT_TRUE(W.Rows() == 6);
  ASSERT_TRUE(W.Cols() == 6);
  Matrix<real> gold_W(6, 6);
  gold_W(0, 0) = 0.50_r;
  gold_W(0, 1) = 0.51_r;
  gold_W(0, 2) = 0.52_r;
  gold_W(0, 3) = 0.53_r;
  gold_W(0, 4) = 0.54_r;
  gold_W(0, 5) = 0.55_r;

  gold_W(1, 0) = 0.56_r;
  gold_W(1, 1) = 0.57_r;
  gold_W(1, 2) = 0.58_r;
  gold_W(1, 3) = 0.59_r;
  gold_W(1, 4) = 0.60_r;
  gold_W(1, 5) = 0.61_r;

  gold_W(2, 0) = 0.62_r;
  gold_W(2, 1) = 0.63_r;
  gold_W(2, 2) = 0.64_r;
  gold_W(2, 3) = 0.65_r;
  gold_W(2, 4) = 0.66_r;
  gold_W(2, 5) = 0.67_r;

  gold_W(3, 0) = 0.68_r;
  gold_W(3, 1) = 0.69_r;
  gold_W(3, 2) = 0.70_r;
  gold_W(3, 3) = 0.71_r;
  gold_W(3, 4) = 0.72_r;
  gold_W(3, 5) = 0.73_r;

  gold_W(4, 0) = 0.74_r;
  gold_W(4, 1) = 0.75_r;
  gold_W(4, 2) = 0.76_r;
  gold_W(4, 3) = 0.77_r;
  gold_W(4, 4) = 0.78_r;
  gold_W(4, 5) = 0.79_r;

  gold_W(5, 0) = 0.80_r;
  gold_W(5, 1) = 0.81_r;
  gold_W(5, 2) = 0.82_r;
  gold_W(5, 3) = 0.83_r;
  gold_W(5, 4) = 0.84_r;
  gold_W(5, 5) = 0.85_r;

  AssertNearMatrix(gold_W, W);

  // bias
  auto const b = layer.BiasConstView();
  ASSERT_TRUE(b.size() == 6);
  ColumnVector<real> gold_b(6);
  gold_b(0) = 1_r;
  gold_b(1) = 2_r;
  gold_b(2) = 3_r;
  gold_b(3) = 4_r;
  gold_b(4) = 5_r;
  gold_b(5) = 6_r;
  AssertNearMatrix(gold_b, b);
}

static void AssertVerifyLayerIndex2(ai::Mlp<real>& m) {
  auto const& layer = m.GetLayer(2);

  // weight
  auto const W = layer.WeightsConstView();
  ASSERT_TRUE(W.Rows() == 3);
  ASSERT_TRUE(W.Cols() == 6);
  Matrix<real> gold_W(3, 6);
  gold_W(0, 0) = 1_r;
  gold_W(0, 1) = 1.01_r;
  gold_W(0, 2) = 1.02_r;
  gold_W(0, 3) = 1.03_r;
  gold_W(0, 4) = 1.04_r;
  gold_W(0, 5) = 1.05_r;

  gold_W(1, 0) = 1.06_r;
  gold_W(1, 1) = 1.07_r;
  gold_W(1, 2) = 1.08_r;
  gold_W(1, 3) = 1.09_r;
  gold_W(1, 4) = 1.10_r;
  gold_W(1, 5) = 1.11_r;

  gold_W(2, 0) = 1.12_r;
  gold_W(2, 1) = 1.13_r;
  gold_W(2, 2) = 1.14_r;
  gold_W(2, 3) = 1.15_r;
  gold_W(2, 4) = 1.16_r;
  gold_W(2, 5) = 1.17_r;
  AssertNearMatrix(gold_W, W);

  // bias
  auto const b = layer.BiasConstView();
  ASSERT_TRUE(b.size() == 3);
  ColumnVector<real> gold_b(3);
  gold_b(0) = 1_r;
  gold_b(1) = 2_r;
  gold_b(2) = 3_r;
  AssertNearMatrix(gold_b, b);
}

// The CROM model data is not shipped externally.
#if MOCHI_USE_HDF5 && !MOCHI_USE_DOUBLE_PRECISION && MOCHI_INTERNAL
#define MOCHI_CAN_TEST_CROM_H5 1
#else
#define MOCHI_CAN_TEST_CROM_H5 0
#endif

TEST_IF(MOCHI_CAN_TEST_CROM_H5, MochiCROM, LoadModelWithNativeMlpFromH5) {
  real const scale = 1_r;
  auto path = test::GetAssetPath("miscellanea/crom_model_for_h5_load_test/model.h5");
  auto [network, metricsForInverseStdOutput] =
      *hdf5::LoadCromDecoderModel(path, "rom/test", scale, test::ExpectOK{});

  AssertVerifyLayerIndex0(network);
  AssertVerifyLayerIndex1(network);
  AssertVerifyLayerIndex2(network);

  ASSERT_TRUE(metricsForInverseStdOutput[0] == 1.5_r);
  ASSERT_TRUE(metricsForInverseStdOutput[1] == 3_r);
  ASSERT_TRUE(metricsForInverseStdOutput[2] == -2_r);
  ASSERT_TRUE(metricsForInverseStdOutput[3] == 2.5_r);
  ASSERT_TRUE(metricsForInverseStdOutput[4] == -3.5_r);
  ASSERT_TRUE(metricsForInverseStdOutput[5] == 4.5_r);

  Matrix<real> X(5, 1), Y(3, 1), J(3, 5);
  X.SetConstant(0.001);
  Y.SetConstant(0);
  J.SetConstant(0);
  network.ForwardAndJacobian(X, Y, J);

  ColumnVector<real> goldY(3, 1);
  goldY(0, 0) = 260.3588_r;
  goldY(1, 0) = -383.8342_r;
  goldY(2, 0) = 526.7705_r;

  Matrix<real> goldJ(3, 5);
  goldJ(0, 0) = 8.5621_r;
  goldJ(0, 1) = 9.1863_r;
  goldJ(0, 2) = 4.9053_r;
  goldJ(0, 3) = -4.1739_r;
  goldJ(0, 4) = 2.7648_r;
  goldJ(1, 0) = -12.6868_r;
  goldJ(1, 1) = -13.6118_r;
  goldJ(1, 2) = -7.2684_r;
  goldJ(1, 3) = 6.1847_r;
  goldJ(1, 4) = -4.0967_r;

  goldJ(2, 0) = 17.2116_r;
  goldJ(2, 1) = 18.4664_r;
  goldJ(2, 2) = 9.8607_r;
  goldJ(2, 3) = -8.3905_r;
  goldJ(2, 4) = 5.5578_r;

  // we need to account for the output "normalization"
  for (int k = 0; k < Y.Rows(); ++k) {
    Y(k, 0) = Y(k, 0) * metricsForInverseStdOutput[k + 3] + metricsForInverseStdOutput[k];

    for (int c = 0; c < J.Cols(); ++c) {
      J(k, c) *= metricsForInverseStdOutput[k + 3];
    }
  }

  AssertNearMatrix(goldY, Y);
  AssertNearMatrix(goldJ, J);
}
