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

#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using namespace mochi;
using namespace mochi::test;

// TODO(T186384485): Extend test coverage to cover all specializations.

using MlpLayerType = ai::MlpLayer<real>;
using W_t = typename MlpLayerType::WeightType;
using b_t = typename MlpLayerType::BiasType;
using MlpType = ai::Mlp<real>;
using MatrixType = Matrix<real>;
using VecType = ColumnVector<real>;

constexpr real kTolerance = std::is_same_v<real, float> ? 1e-3 : 1e-10;

static void FillRandom(W_t& W, b_t& b) {
  W.SetRandom(123);
  b.SetRandom(234);
};

template <class y_t, class A_t, class x_t, class s_t>
void MatVecPlusShift(y_t& y, A_t const& A, x_t const& x, s_t const& shift) {
  // Computes y = A x + shift. We need at some point to take for granted some functionalities,
  // otherwise we reimplement everything just for the test. Here, we take for granted that operator*
  // on matrices works correctly.
  y = A * x;
  y += shift;
}

template <typename ActivationFunctor, typename T>
auto GoldMlpTwoLayers(
    MatrixType const& Xin,
    ai::MlpLayer<T> const& l1,
    ai::MlpLayer<T> const& l2,
    ActivationFunctor f) {
  /*
   * Yout = f( W2 * f(W1 * x + b1) + b2 )
   *
   * f is the activation function
   *
   * J is dYout/dx is the Jacobian we want to compute which can be written as:
   *
   *	J = [df/dy2] W2 [df/dy1] W1
   *
   * where y2 = W2 * y1 + b2 and y1 = f(W1 * x + b1)
   */

  auto const W1 = l1.WeightsView();
  auto const b1 = l1.BiasView();
  auto const W2 = l2.WeightsConstView();
  auto const b2 = l2.BiasConstView();

  int const iDim = Xin.Rows();
  int const batchSize = Xin.Cols();
  int const hDim = W1.Rows();
  int const oDim = W2.Rows();

  auto Yout = MatrixType::Zero(oDim, batchSize);
  auto Jout = MatrixType::Zero(oDim * batchSize, iDim);

  for (int batchId = 0; batchId < batchSize; ++batchId) {
    auto x = Xin.Col(batchId);

    auto df1dy1 = MatrixType::Zero(hDim, hDim);
    auto y1 = VecType::Zero(hDim);
    MatVecPlusShift(y1, W1, x, b1);
    // y1 = W1 * x + b1, so need to apply now the activation in place
    for (int r = 0; r < hDim; ++r) {
      f(y1(r), df1dy1(r, r)); // modifies operands in place
    }

    auto df2dy2 = MatrixType::Zero(oDim, oDim);
    auto y2 = VecType::Zero(oDim);
    MatVecPlusShift(y2, W2, y1, b2);
    // y2 = W2 * f(W1 * x + b1) + b2, so need to apply now the activation in place
    for (int r = 0; r < oDim; ++r) {
      f(y2(r), df2dy2(r, r)); // modifies operands in place
    }

    // Store into output
    Yout.Col(batchId) = y2;

    auto const A = df1dy1 * W1;
    auto const B = df2dy2 * W2;
    Jout.Block(batchId * oDim, 0, oDim, iDim) = B * A;
  }
  return std::make_pair(Yout, Jout);
}

template <typename ActivationFunctor, typename T>
auto GoldMlpThreeLayers(
    MatrixType const& Xin,
    ai::MlpLayer<T> const& l1,
    ai::MlpLayer<T> const& l2,
    ai::MlpLayer<T> const& l3,
    ActivationFunctor f) {
  // Reasoning below is similar to GoldMlpTwoLayers

  auto const W1 = l1.WeightsView();
  auto const b1 = l1.BiasView();
  auto const W2 = l2.WeightsConstView();
  auto const b2 = l2.BiasConstView();
  auto const W3 = l3.WeightsConstView();
  auto const b3 = l3.BiasConstView();

  int const iDim = Xin.Rows();
  int const batchSize = Xin.Cols();
  int const hDim1 = W1.Rows();
  int const hDim2 = W2.Rows();
  int const oDim = W3.Rows();

  auto Yout = MatrixType::Zero(oDim, batchSize);
  auto Jout = MatrixType::Zero(oDim * batchSize, iDim);

  for (int batchId = 0; batchId < batchSize; ++batchId) {
    auto x = Xin.Col(batchId);

    auto df1dy1 = MatrixType::Zero(hDim1, hDim1);
    auto y1 = VecType::Zero(hDim1);
    MatVecPlusShift(y1, W1, x, b1);
    // y1 = W1 * x + b1, so need to apply now the activation in place
    for (int r = 0; r < hDim1; ++r) {
      f(y1(r), df1dy1(r, r)); // modifies operands in place
    }

    auto df2dy2 = MatrixType::Zero(hDim2, hDim2);
    auto y2 = VecType::Zero(hDim2);
    MatVecPlusShift(y2, W2, y1, b2);
    // y2 = W2 * y1 + b2, so need to apply now the activation in place
    for (int r = 0; r < hDim2; ++r) {
      f(y2(r), df2dy2(r, r)); // modifies operands in place
    }

    auto df3dy3 = MatrixType::Zero(oDim, oDim);
    auto y3 = VecType::Zero(oDim);
    MatVecPlusShift(y3, W3, y2, b3);
    // y3 = W3 * y2 + b3, so need to apply now the activation in place
    for (int r = 0; r < oDim; ++r) {
      f(y3(r), df3dy3(r, r)); // modifies operands in place
    }

    // Store into output
    Yout.Col(batchId) = y3;

    auto const A = df1dy1 * W1;
    auto const B = df2dy2 * W2;
    auto const C = df3dy3 * W3;
    auto const D = B * A;
    Jout.Block(batchId * oDim, 0, oDim, iDim) = C * D;
  }
  return std::make_pair(Yout, Jout);
}

template <typename ActivationFunctor>
void TestMlpTwoOrThreeLayersImpl(
    ActivationFunctor activation,
    int batchSize,
    int inputDim,
    std::vector<int> const& hiddenDims,
    int outputDim) {
  /*
    If hiddenDims has two entries, then we are doing:
            outputDim |  hiddenDim1  |  inputDim
    else:
            outputDim |  hiddenDim2  |  hiddenDim1  |  inputDim
    the reason for thinking right-to-left is because of how the MLP is implemented/designed.
    Note: we call this "three" layers because have three weight matrices and bias vectors.
  */

  int const hiddenCount = hiddenDims.size();
  MOCHI_ASSERT(hiddenCount <= 2);
  // hidden1 always exists
  int const hiddenDim1 = hiddenDims[0];

  DynamicArray<MlpLayerType> layers;

  // hidden1 <- input
  W_t W1(hiddenDim1, inputDim);
  b_t b1(hiddenDim1);
  FillRandom(W1, b1);
  layers.emplace_back(std::move(W1), std::move(b1), activation);

  if (hiddenCount == 1) {
    // output <- hidden1
    W_t W2(outputDim, hiddenDim1);
    b_t b2(outputDim);
    FillRandom(W2, b2);
    layers.emplace_back(std::move(W2), std::move(b2), activation);
  } else {
    int const hiddenDim2 = hiddenDims[1];

    // hidden2 <- hidden1
    W_t W2(hiddenDim2, hiddenDim1);
    b_t b2(hiddenDim2);
    FillRandom(W2, b2);
    layers.emplace_back(std::move(W2), std::move(b2), activation);

    // output <- hidden2
    W_t W3(outputDim, hiddenDim2);
    b_t b3(outputDim);
    FillRandom(W3, b3);
    layers.emplace_back(std::move(W3), std::move(b3), activation);
  }

  MlpType model(std::move(layers));

  // Create input
  MatrixType X(inputDim, batchSize);
  X.SetRandom(-0.001_r, 0.001_r);

  // Compute the "gold" solution
  MatrixType goldY, goldJ;
  auto const& l1 = model.GetLayer(0);
  auto const& l2 = model.GetLayer(1);
  if (hiddenDims.size() == 1) {
    std::tie(goldY, goldJ) = GoldMlpTwoLayers(X, l1, l2, activation);
  } else {
    auto const& l3 = model.GetLayer(2);
    std::tie(goldY, goldJ) = GoldMlpThreeLayers(X, l1, l2, l3, activation);
  }

  // Forward only
  auto Y = MatrixType::Zero(outputDim, batchSize);
  model.Forward(X, Y);
  EXPECT_TRUE(NearEqualMatrices(goldY, Y, kTolerance));

  // Forward and Jacobian
  auto Y2 = MatrixType::Zero(outputDim, batchSize);
  MatrixType J(outputDim * batchSize, inputDim);
  J.SetConstant(0);
  model.ForwardAndJacobian(X, Y2, J);
  EXPECT_TRUE(NearEqualMatrices(goldY, Y2, kTolerance));
  EXPECT_TRUE(NearEqualMatrices(goldJ, J, kTolerance));
}

std::vector<int> const batchSizesToTest = {1, 2, 4, 8};
std::vector<int> const inputDimsToTest = {1, 2, 3};
std::vector<int> const hiddenDimsToTest = {2, 3, 13, 57};
std::vector<int> const outDimsToTest = {1, 2, 3};

#define MOCHI_RUN_LOOP_TEST_TWO_LAYERS(ACTIVATION)                          \
  for (int batchSize : batchSizesToTest) {                                  \
    for (int iDim : inputDimsToTest) {                                      \
      for (int hDim : hiddenDimsToTest) {                                   \
        for (int oDim : outDimsToTest) {                                    \
          TestMlpTwoOrThreeLayersImpl(                                      \
              ACTIVATION, batchSize, iDim, std::vector<int>({hDim}), oDim); \
        }                                                                   \
      }                                                                     \
    }                                                                       \
  }

#define MOCHI_RUN_LOOP_TEST_THREE_LAYERS(ACTIVATION)                                  \
  for (int batchSize : batchSizesToTest) {                                            \
    for (int iDim : inputDimsToTest) {                                                \
      for (int hDim1 : hiddenDimsToTest) {                                            \
        for (int hDim2 : hiddenDimsToTest) {                                          \
          for (int oDim : outDimsToTest) {                                            \
            TestMlpTwoOrThreeLayersImpl(                                              \
                ACTIVATION, batchSize, iDim, std::vector<int>({hDim1, hDim2}), oDim); \
          }                                                                           \
        }                                                                             \
      }                                                                               \
    }                                                                                 \
  }

TEST(Mlp, MlpTwoLayersIdentityActivation) {
  ai::IdentityActivation<real> activation;
  MOCHI_RUN_LOOP_TEST_TWO_LAYERS(activation)
}

TEST(Mlp, MlpTwoLayersELUActivation) {
  ai::ELUActivation<real> activation;
  MOCHI_RUN_LOOP_TEST_TWO_LAYERS(activation)
}

TEST(Mlp, MlpTwoLayersReLUActivation) {
  ai::ReLUActivation<real> activation;
  MOCHI_RUN_LOOP_TEST_TWO_LAYERS(activation)
}

TEST(Mlp, MlpThreeLayersIdentityActivation) {
  ai::IdentityActivation<real> activation;
  MOCHI_RUN_LOOP_TEST_THREE_LAYERS(activation)
}

TEST(Mlp, MlpThreeLayersELUActivation) {
  ai::ELUActivation<real> activation;
  MOCHI_RUN_LOOP_TEST_THREE_LAYERS(activation)
}

TEST(Mlp, MlpThreeLayersReLUActivation) {
  ai::ReLUActivation<real> activation;
  MOCHI_RUN_LOOP_TEST_THREE_LAYERS(activation)
}

TEST(Mlp, MlpTwoLayersTrivial) {
  ai::IdentityActivation<real> f;

  auto filler = [](auto& W, auto& b, int countIn) {
    int count = countIn;
    for (int i = 0; i < W.Rows(); ++i) {
      for (int j = 0; j < W.Cols(); ++j) {
        W(i, j) = real(count++);
        if (count % 2 == 0) {
          W(i, j) *= -1_r;
        }
      }
      b(i) = real(i);
    }
  };

  int const batchSize = 1;
  int const inputDim = 2;
  int const hiddenDim = 4;
  int const outputDim = 3;

  DynamicArray<MlpLayerType> layers;

  // layer1: hidden1 <- input
  // b1 = [0,1,2,3]^T
  // W1 = [[0,-1], [2,-3], [4,-5], [6,-7]]
  W_t W1(hiddenDim, inputDim);
  b_t b1(hiddenDim);
  filler(W1, b1, 0);
  layers.emplace_back(std::move(W1), std::move(b1), f);

  // layer2: output <- hidden1
  // b2 = [0,1,2,3]^T
  // W2 = [[-1 2 -3 4], [-5 6 -7 8], [-9 10 -11 12]]
  W_t W2(outputDim, hiddenDim);
  b_t b2(outputDim);
  filler(W2, b2, 1);
  layers.emplace_back(std::move(W2), std::move(b2), f);

  MlpType M(std::move(layers));

  MatrixType X(inputDim, batchSize);
  X.SetConstant(2);
  MatrixType Y(outputDim, batchSize);
  MatrixType J(outputDim * batchSize, inputDim);
  M.ForwardAndJacobian(X, Y, J);

  MatrixType goldY(outputDim, 1);
  MatrixType goldJ(outputDim, inputDim);
  goldY(0, 0) = 4_r;
  goldY(1, 0) = 13_r;
  goldY(2, 0) = 22_r;

  goldJ(0, 0) = 16_r;
  goldJ(0, 1) = -18_r;
  goldJ(1, 0) = 32_r;
  goldJ(1, 1) = -34_r;
  goldJ(2, 0) = 48_r;
  goldJ(2, 1) = -50_r;

  EXPECT_TRUE(NearEqualMatrices(goldY, Y, kTolerance));
  EXPECT_TRUE(NearEqualMatrices(goldJ, J, kTolerance));
}
