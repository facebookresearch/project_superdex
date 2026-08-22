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
#include <utility>
#include <variant>

using namespace mochi;
using namespace mochi::test;

// TODO(T186384485): Extend test coverage to cover all specializations.

using MlpLayerType = ai::MlpLayer<real>;
using WeightType = typename MlpLayerType::WeightType;
using BiasType = typename MlpLayerType::BiasType;
using MatrixType = Matrix<real>;

constexpr real kTolerance = std::is_same_v<real, float> ? 1e-3 : 1e-10;

template <typename ActivationType>
static void TestConstructionFromData() {
  WeightType W(5, 3);
  BiasType b(5);
  MlpLayerType l(std::move(W), std::move(b), ActivationType());

  EXPECT_EQ(3, l.InputDim());
  EXPECT_EQ(5, l.OutputDim());
  EXPECT_EQ(5, l.WeightsView().Rows());
  EXPECT_EQ(3, l.WeightsConstView().Cols());
  EXPECT_EQ(5, l.BiasView().size());
  EXPECT_EQ(5, l.BiasConstView().size());

  std::visit(
      [&](auto&& act) {
        EXPECT_TRUE((std::is_same_v<ActivationType const, std::decay_t<decltype(act)> const>));
      },
      l.GetActivation());

  // The above constructor for MlpLayer layer accepts rvalue references and by design it should move
  // W and b. Both W and b are instances of the Matrix class. The Matrix class supports a move
  // constructor that actually moves things, and sets to to null the underlying ptr of the
  // moved-from object, so we can check that W and b are effectively moved.
  EXPECT_EQ(nullptr, W.data()); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(nullptr, b.data()); // NOLINT(bugprone-use-after-move)
}

TEST(MlpLayer, DefaultConstructor) {
  MlpLayerType l;
  EXPECT_EQ(0, l.InputDim());
  EXPECT_EQ(0, l.OutputDim());
  EXPECT_EQ(0, l.WeightsView().Rows());
  EXPECT_EQ(0, l.WeightsConstView().Cols());
  EXPECT_EQ(0, l.BiasView().size());
  EXPECT_EQ(0, l.BiasConstView().size());
}

TEST(MlpLayer, ConstructorFromData) {
  TestConstructionFromData<ai::IdentityActivation<real>>();
  TestConstructionFromData<ai::ReLUActivation<real>>();
  TestConstructionFromData<ai::ELUActivation<real>>();
}

struct MlpLayerTester : public testing::Test {
  int const inputDim = 3;
  int const outputDim = 4;
  int const batchSize = 2;
  MlpLayerType m_l;

  template <typename ActivType>
  void CreateLayer(ActivType F) {
    WeightType W(outputDim, inputDim);
    BiasType b(outputDim);

    int count = 0;
    for (int i = 0; i < W.Rows(); ++i) {
      for (int j = 0; j < W.Cols(); ++j) {
        W(i, j) = real(count++);
        if (count % 2 == 0) {
          W(i, j) *= -1_r;
        }
      }
      b(i) = real(i);
    }

    m_l = MlpLayerType(std::move(W), std::move(b), F);
  }

  void operator()(MatrixType const& X, MatrixType& goldY, MatrixType& goldJ) {
    {
      auto Y = MatrixType::Zero(outputDim, batchSize);
      m_l.Forward(X, Y);
      EXPECT_TRUE(NearEqualMatrices(goldY, Y, kTolerance));
    }

    {
      auto Y = MatrixType::Zero(outputDim, batchSize);
      auto J = MatrixType::Zero(outputDim, batchSize);
      m_l.template Forward<true>(X, Y, J);
      EXPECT_TRUE(NearEqualMatrices(goldY, Y, kTolerance));
      EXPECT_TRUE(NearEqualMatrices(goldJ, J, kTolerance));
    }
  }
};

TEST_F(MlpLayerTester, WithIdentityActivation) {
  this->CreateLayer(ai::IdentityActivation<real>());

  MatrixType X(inputDim, batchSize);
  X.Col(0).SetConstant(1);
  X.Col(1).SetConstant(2);

  // Prepare "gold" variables and fill with expected values
  MatrixType goldY(4, batchSize);
  goldY(0, 0) = 1_r;
  goldY(1, 0) = -3_r;
  goldY(2, 0) = 9_r;
  goldY(3, 0) = -7_r;
  goldY(0, 1) = 2_r;
  goldY(1, 1) = -7_r;
  goldY(2, 1) = 16_r;
  goldY(3, 1) = -17_r;

  MatrixType goldJ(outputDim, batchSize);
  goldJ.SetConstant(1);

  (*this)(X, goldY, goldJ);
}

TEST_F(MlpLayerTester, WithELUActivation) {
  ai::ELUActivation<real> activFunc;

  this->CreateLayer(activFunc);

  MatrixType X(inputDim, batchSize);
  X.Col(0).SetConstant(1);
  X.Col(1).SetConstant(2);

  // Prepare "gold" variables and fill with expected values

  // **Before** applying activation: this is just W X + b
  MatrixType goldY_preAct(outputDim, batchSize);
  auto const alpha = activFunc.alpha;
  goldY_preAct(0, 0) = 1_r;
  goldY_preAct(1, 0) = -3_r;
  goldY_preAct(2, 0) = 9_r;
  goldY_preAct(3, 0) = -7_r;
  goldY_preAct(0, 1) = 2_r;
  goldY_preAct(1, 1) = -7_r;
  goldY_preAct(2, 1) = 16_r;
  goldY_preAct(3, 1) = -17_r;

  // **After** applying activation: f(goldY_preAct)
  MatrixType goldY_postAct = goldY_preAct;
  goldY_postAct(1, 0) = alpha * (std::exp(goldY_preAct(1, 0)) - 1_r);
  goldY_postAct(1, 1) = alpha * (std::exp(goldY_preAct(1, 1)) - 1_r);
  goldY_postAct(3, 0) = alpha * (std::exp(goldY_preAct(3, 0)) - 1_r);
  goldY_postAct(3, 1) = alpha * (std::exp(goldY_preAct(3, 1)) - 1_r);

  MatrixType goldJ(outputDim, batchSize);
  goldJ.SetConstant(1);
  goldJ(1, 0) = alpha * std::exp(goldY_preAct(1, 0));
  goldJ(1, 1) = alpha * std::exp(goldY_preAct(1, 1));
  goldJ(3, 0) = alpha * std::exp(goldY_preAct(3, 0));
  goldJ(3, 1) = alpha * std::exp(goldY_preAct(3, 1));

  (*this)(X, goldY_postAct, goldJ);
}

TEST_F(MlpLayerTester, WithReLUActivation) {
  ai::ReLUActivation<real> activFunc;

  this->CreateLayer(activFunc);

  MatrixType X(inputDim, batchSize);
  X.Col(0).SetConstant(1);
  X.Col(1).SetConstant(2);

  // Prepare "gold" variables and fill with expected values

  // **Before** applying activation: this is just W X + b
  MatrixType goldY_preAct(outputDim, batchSize);
  goldY_preAct(0, 0) = 1_r;
  goldY_preAct(1, 0) = -3_r;
  goldY_preAct(2, 0) = 9_r;
  goldY_preAct(3, 0) = -7_r;
  goldY_preAct(0, 1) = 2_r;
  goldY_preAct(1, 1) = -7_r;
  goldY_preAct(2, 1) = 16_r;
  goldY_preAct(3, 1) = -17_r;

  // **After** applying activation: f(goldY_preAct)
  MatrixType goldY_postAct = goldY_preAct;
  goldY_postAct(1, 0) = 0_r;
  goldY_postAct(1, 1) = 0_r;
  goldY_postAct(3, 0) = 0_r;
  goldY_postAct(3, 1) = 0_r;

  MatrixType goldJ(outputDim, batchSize);
  goldJ.SetConstant(1);
  goldJ(1, 0) = 0_r;
  goldJ(1, 1) = 0_r;
  goldJ(3, 0) = 0_r;
  goldJ(3, 1) = 0_r;

  (*this)(X, goldY_postAct, goldJ);
}
