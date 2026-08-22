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

#include <mochi_core/integration/integration_utils.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>

#include <gtest/gtest.h>

#include <type_traits>

using namespace mochi;

TEST(IntegrationUtils, Coefficients) {
  // Test correctness of the coefficients of all supported time integration schemes.

  auto runChecks = [](auto const& params,
                      int numSteps,
                      int numStages,
                      auto const& alphaExpected,
                      real betaExpected,
                      auto const& Aexpected,
                      auto const& bExpected,
                      auto const& cExpected) {
    using BType = std::decay_t<decltype(bExpected)>;
    BType bTildeExpected = bExpected * Inverse(Aexpected);
    EXPECT_EQ(numSteps, params.numSteps);
    EXPECT_EQ(numStages, params.numStages);
    EXPECT_TRUE(test::NearEqualMatrices(alphaExpected, params.alpha));
    EXPECT_NEAR_EQ(betaExpected, params.beta);
    EXPECT_TRUE(test::NearEqualMatrices(Aexpected, params.A));
    EXPECT_TRUE(test::NearEqualMatrices(Inverse(Aexpected), params.Ainv));
    EXPECT_TRUE(test::NearEqualMatrices(bExpected, params.b));
    EXPECT_TRUE(test::NearEqualMatrices(bTildeExpected, params.bTilde));
    EXPECT_TRUE(test::NearEqualMatrices(cExpected, params.c));
  };

  {
    int const numSteps = 1;
    int const numStages = 1;
    RowVector<real> alphaExpected = {{1_r}};
    real const betaExpected = 1_r;
    Matrix<real> Aexpected = {{1_r}};
    RowVector<real> bExpected = {{1_r}};
    RowVector<real> cExpected = {{1_r}};
    auto const params = CreateTimeIntegratorParams(IntegrationMethod::BackwardEuler);
    runChecks(
        params, numSteps, numStages, alphaExpected, betaExpected, Aexpected, bExpected, cExpected);
  }

  {
    int const numSteps = 2;
    int const numStages = 1;
    RowVector<real> alphaExpected = {{4_r / 3_r, -1_r / 3_r}};
    real const betaExpected = 2_r / 3_r;
    Matrix<real> Aexpected = {{1_r}};
    RowVector<real> bExpected = {{1_r}};
    RowVector<real> cExpected = {{1_r}};
    auto const params = CreateTimeIntegratorParams(IntegrationMethod::BDF2);
    runChecks(
        params, numSteps, numStages, alphaExpected, betaExpected, Aexpected, bExpected, cExpected);
  }

  {
    int const numSteps = 3;
    int const numStages = 1;
    RowVector<real> alphaExpected = {{18_r / 11_r, -9_r / 11_r, 2_r / 11_r}};
    real const betaExpected = 6_r / 11_r;
    Matrix<real> Aexpected = {{1_r}};
    RowVector<real> bExpected = {{1_r}};
    RowVector<real> cExpected = {{1_r}};
    auto const params = CreateTimeIntegratorParams(IntegrationMethod::BDF3);
    runChecks(
        params, numSteps, numStages, alphaExpected, betaExpected, Aexpected, bExpected, cExpected);
  }

  {
    int const numSteps = 1;
    int const numStages = 2;
    RowVector<real> alphaExpected = {{1_r}};
    real const betaExpected = 1_r;
    real const alpha = 1_r - Sqrt(2_r) / 2_r;
    Matrix<real> Aexpected = {{alpha, 1_r - alpha}, {0_r, alpha}};
    RowVector<real> bExpected = {{1_r - alpha, alpha}};
    RowVector<real> cExpected = {{alpha, 1_r}};
    auto const params = CreateTimeIntegratorParams(IntegrationMethod::DIRK22);
    runChecks(
        params, numSteps, numStages, alphaExpected, betaExpected, Aexpected, bExpected, cExpected);
  }

  {
    int const numSteps = 1;
    int const numStages = 3;
    RowVector<real> alphaExpected = {{1_r}};
    real const betaExpected = 1_r;
    real const gamma = 0.43586652150845899941601945_r;
    real const b1 = -1.5_r * Pow(gamma, 2) + 4_r * gamma - 0.25_r;
    real const b2 = 1.5_r * Pow(gamma, 2) - 5_r * gamma + 1.25_r;
    real const c2 = (1_r + gamma) / 2_r;
    Matrix<real> Aexpected = {{gamma, c2 - gamma, b1}, {0_r, gamma, b2}, {0_r, 0_r, gamma}};
    RowVector<real> bExpected = {{b1, b2, gamma}};
    RowVector<real> cExpected = {{gamma, c2, 1_r}};
    auto const params = CreateTimeIntegratorParams(IntegrationMethod::DIRK33);
    runChecks(
        params, numSteps, numStages, alphaExpected, betaExpected, Aexpected, bExpected, cExpected);
  }

  {
    int const numSteps = 1;
    int const numStages = 2;
    RowVector<real> alphaExpected = {{1_r}};
    real const betaExpected = 1_r;
    Matrix<real> Aexpected = {
        {0.5_r + 0.5_r / Sqrt(3_r), -1_r / Sqrt(3_r)}, {0_r, 0.5_r + 0.5_r / Sqrt(3_r)}};
    RowVector<real> bExpected = {{0.5_r, 0.5_r}};
    RowVector<real> cExpected = {{0.5_r + 0.5_r / Sqrt(3_r), 0.5_r - 0.5_r / Sqrt(3_r)}};
    auto const params = CreateTimeIntegratorParams(IntegrationMethod::DIRK23);
    runChecks(
        params, numSteps, numStages, alphaExpected, betaExpected, Aexpected, bExpected, cExpected);
  }

  {
    int const numSteps = 1;
    int const numStages = 1;
    RowVector<real> alphaExpected = {{1_r}};
    real const betaExpected = 1_r;
    Matrix<real> Aexpected = {{0.5_r}};
    RowVector<real> bExpected = {{1_r}};
    RowVector<real> cExpected = {{0.5_r}};
    auto const params = CreateTimeIntegratorParams(IntegrationMethod::SymplecticDIRK12);
    runChecks(
        params, numSteps, numStages, alphaExpected, betaExpected, Aexpected, bExpected, cExpected);
  }

  {
    int const numSteps = 1;
    int const numStages = 2;
    RowVector<real> alphaExpected = {{1_r}};
    real const betaExpected = 1_r;
    Matrix<real> Aexpected = {{0.25_r, 0.5_r}, {0_r, 0.25_r}};
    RowVector<real> bExpected = {{0.5_r, 0.5_r}};
    RowVector<real> cExpected = {{0.25_r, 0.75_r}};
    auto const params = CreateTimeIntegratorParams(IntegrationMethod::SymplecticDIRK22);
    runChecks(
        params, numSteps, numStages, alphaExpected, betaExpected, Aexpected, bExpected, cExpected);
  }
}

TEST(IntegrationUtils, TimeIntegratorStateFromParams) {
  // Test TimeIntegratorState constructor from params.
  auto const params = CreateTimeIntegratorParams(IntegrationMethod::DIRK33);
  int const currentStage = 1;
  real const dt = 1.5e-2_r;
  TimeIntegratorState state;
  state.Set(params, currentStage, dt);

  EXPECT_EQ(state.numSteps, params.numSteps);
  EXPECT_EQ(state.numStages, params.numStages);
  EXPECT_EQ(state.currentStage, currentStage);
  EXPECT_NEAR_EQ(state.dtStage, params.beta * params.A(currentStage, currentStage) * dt);
  EXPECT_TRUE(test::NearEqualMatrices(state.alpha, params.alpha));
  EXPECT_TRUE(
      test::NearEqualMatrices(
          state.aTilde,
          Matrix<real>{
              params.A.Block(currentStage, 0, 1, currentStage) *
              params.Ainv.Block(0, 0, currentStage, currentStage)}));
  EXPECT_TRUE(test::NearEqualMatrices(state.bTilde, params.bTilde));
}

TEST(IntegrationUtils, GetNumSteps) {
  EXPECT_EQ(1, GetNumSteps(IntegrationMethod::BackwardEuler));
  EXPECT_EQ(2, GetNumSteps(IntegrationMethod::BDF2));
  EXPECT_EQ(3, GetNumSteps(IntegrationMethod::BDF3));
  EXPECT_EQ(1, GetNumSteps(IntegrationMethod::DIRK22));
  EXPECT_EQ(1, GetNumSteps(IntegrationMethod::DIRK23));
  EXPECT_EQ(1, GetNumSteps(IntegrationMethod::DIRK33));
  EXPECT_EQ(1, GetNumSteps(IntegrationMethod::SymplecticDIRK12));
  EXPECT_EQ(1, GetNumSteps(IntegrationMethod::SymplecticDIRK22));
}

TEST(IntegrationUtils, GetNumStages) {
  EXPECT_EQ(1, GetNumStages(IntegrationMethod::BackwardEuler));
  EXPECT_EQ(1, GetNumStages(IntegrationMethod::BDF2));
  EXPECT_EQ(1, GetNumStages(IntegrationMethod::BDF3));
  EXPECT_EQ(2, GetNumStages(IntegrationMethod::DIRK22));
  EXPECT_EQ(2, GetNumStages(IntegrationMethod::DIRK23));
  EXPECT_EQ(3, GetNumStages(IntegrationMethod::DIRK33));
  EXPECT_EQ(1, GetNumStages(IntegrationMethod::SymplecticDIRK12));
  EXPECT_EQ(2, GetNumStages(IntegrationMethod::SymplecticDIRK22));
}
