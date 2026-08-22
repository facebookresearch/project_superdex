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

#include <mochi_core/integration/integration_params.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/no_copy.h>

#include <limits>

namespace mochi {

/// @brief Maximum number of supported integration steps and stages. Please update as needed.
constexpr int kMaxIntegrationSteps = 3;
constexpr int kMaxIntegrationStages = 3;

struct TimeIntegratorParams final {
  TimeIntegratorParams(
      int numStepsIn,
      RowVector<real> const& alphaIn,
      real betaIn,
      int numStagesIn,
      RowMatrix<real> const& Ain,
      RowVector<real> const& bIn,
      RowVector<real> const& cIn)
      : // Multi-step parameters.
        numSteps(numStepsIn),
        alpha(alphaIn),
        beta(betaIn),
        // Multi-stage parameters.
        numStages(numStagesIn),
        A(Ain),
        Ainv(Inverse(Ain)),
        b(bIn),
        bTilde(bIn * Ainv),
        c(cIn) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    MOCHI_ASSERT_VERBOSE(
        (numSteps > 0) && (numSteps <= kMaxIntegrationSteps), "Invalid number of steps.");
    MOCHI_ASSERT_VERBOSE(
        (numStages > 0) && (numStages <= kMaxIntegrationStages), "Invalid number of stages.");
    MOCHI_ASSERT_VERBOSE((alpha.Rows() == 1) && (alpha.Cols() == numSteps), "Inconsistent size.");
    MOCHI_ASSERT_VERBOSE((A.Rows() == numStages) && (A.Cols() == numStages), "Inconsistent size.");
    MOCHI_ASSERT_VERBOSE((b.Rows() == 1) && (b.Cols() == numStages), "Inconsistent size.");
    MOCHI_ASSERT_VERBOSE((c.Rows() == 1) && (c.Cols() == numStages), "Inconsistent size.");

    // Check the scheme is at least 1st-order and thus consistent.
    real alphaSum = 0_r;
    real alphaWeightedSum = 0_r;
    for (int i = 0; i < numSteps; ++i) {
      alphaSum += alpha(i);
      alphaWeightedSum += (i + 1) * alpha(i);
    }
    MOCHI_ASSERT_VERBOSE(Abs(alphaSum - 1_r) <= numSteps * std::numeric_limits<real>::epsilon());
    MOCHI_ASSERT_VERBOSE(
        Abs(alphaWeightedSum - beta) <= numSteps * std::numeric_limits<real>::epsilon());
    real bSum = 0_r;
    for (int i = 0; i < numStages; ++i) {
      MOCHI_ASSERT_VERBOSE(
          // May be problematic for hard-coded finite differences.
          A(i, i) >= 0_r,
          "Schemes with negative diagonal coefficients not supported yet.");
      MOCHI_ASSERT_VERBOSE(A(i, i) != 0_r, "Explicit Runge-Kutta (ERK) schemes not supported yet.");
      bSum += b(i);
      real aRowSum = 0_r;
      for (int j = 0; j < numStages; ++j) {
        aRowSum += A(i, j);
        if (j > i) {
          MOCHI_ASSERT_VERBOSE(
              A(i, j) == 0_r, "Fully implicit Runge-Kutta (FIRK) schemes not supported yet.");
        }
      }
      MOCHI_ASSERT_VERBOSE(Abs(aRowSum - c(i)) <= numStages * std::numeric_limits<real>::epsilon());
    }
    MOCHI_ASSERT_VERBOSE(Abs(bSum - 1_r) <= numStages * std::numeric_limits<real>::epsilon());
#endif
  }

  int const numSteps = {}; // Number of steps. >1 for multi-step methods.
  RowVector<real> const alpha = {}; // 'alpha' coefficients for multi-step methods.
  real const beta = {}; // 'beta' coefficient for multi-step methods.
  int const numStages = {}; // Number of stages. >1 for multi-stage methods.
  RowMatrix<real> const A = {}; // 'A' matrix of Butcher tableau.
  RowMatrix<real> const Ainv = {}; // Ainv = Inverse(A).
  RowVector<real> const b = {}; // 'b' vector of Butcher tableau.
  RowVector<real> const bTilde = {}; // bTilde = b * Inverse(A).
  RowVector<real> const c = {}; // 'c' vector of Butcher tableau.
};

/**
 * Structure storing the current state of the time integrator.
 */
struct TimeIntegratorState {
  TimeIntegratorState() = default;
  MOCHI_DECLARE_MOVE_ONLY(TimeIntegratorState);

  void Set(TimeIntegratorParams const& params, int iStage, real dt) {
    dtStage = static_cast<real>(params.beta * params.A(iStage, iStage) * dt);
    currentStage = iStage;
    numStages = params.numStages;
    numSteps = params.numSteps;
    // Resize RowVectors before copying values, in case dimensions have changed
    alpha.Resize(params.alpha.Cols());
    alpha = params.alpha;
    aTilde.Resize(iStage); // May resize to zero
    aTilde = params.A.Block(iStage, 0, 1, iStage) * params.Ainv.Block(0, 0, iStage, iStage);
    bTilde.Resize(params.bTilde.Cols());
    bTilde = params.bTilde;
  }

  real dtStage = -1_r; // Time increment in the current stage.
  int currentStage = 0; // Current stage, in the range 0, 1, ..., numStages - 1.
  int numSteps = 0; // Number of steps. >1 for multi-step methods.
  RowVector<real> alpha = {}; // 'alpha' coefficients for multi-step methods.
  int numStages = 0; // Number of stages. >1 for multi-stage methods.
  RowVector<real> aTilde = {}; // aTilde = A.Block(currentStage, 0, 1, currentStage) *
                               //          Ainv.Block(0, 0, currentStage, currentStage)
  RowVector<real> bTilde = {}; // bTilde = b * Inverse(A).
};

[[nodiscard]] inline TimeIntegratorParams CreateTimeIntegratorParams(
    IntegrationMethod const& method) {
  // Reference for DIRK schemes: C.A. Kennedy, M.H. Carpenter, Diagonally Implicit Runge-Kutta
  // Methods for Ordinary Differential Equations. A Review. NASA/TM–2016–219173.
  // https://ntrs.nasa.gov/api/citations/20160005923/downloads/20160005923.pdf

  switch (method) {
    case IntegrationMethod::BackwardEuler: {
      // 1 step, 1 stage, 1st order, L-stable. Also known as BDF1 and DIRK11.
      return TimeIntegratorParams{1, {{1_r}}, 1_r, 1, {{1_r}}, {{1_r}}, {{1_r}}};
    }
    case IntegrationMethod::BDF2: {
      // 2 steps, 1 stage, 2nd order, A-stable.
      return TimeIntegratorParams{
          2, {{4_r / 3_r, -1_r / 3_r}}, 2_r / 3_r, 1, {{1_r}}, {{1_r}}, {{1_r}}};
    }
    case IntegrationMethod::BDF3: {
      // 3 steps, 1 stage, 3rd order.
      return TimeIntegratorParams{
          3, {{18_r / 11_r, -9_r / 11_r, 2_r / 11_r}}, 6_r / 11_r, 1, {{1_r}}, {{1_r}}, {{1_r}}};
    }
    case IntegrationMethod::DIRK22: {
      // 1 step, 2 stages, 2nd order, L-stable.
      real const alpha = 1_r - Sqrt(2_r) / 2_r;
      return TimeIntegratorParams{
          1,
          {{1_r}},
          1_r,
          2,
          {{alpha, 0_r}, {1_r - alpha, alpha}},
          {{1_r - alpha, alpha}},
          {{alpha, 1_r}}};
    }
    case IntegrationMethod::DIRK23: {
      // 1 step, 2 stages, 3rd order.
      return TimeIntegratorParams{
          1,
          {{1_r}},
          1_r,
          2,
          {{0.5_r + 0.5_r / Sqrt(3_r), 0_r}, {-1_r / Sqrt(3_r), 0.5_r + 0.5_r / Sqrt(3_r)}},
          {{0.5_r, 0.5_r}},
          {{0.5_r + 0.5_r / Sqrt(3_r), 0.5_r - 0.5_r / Sqrt(3_r)}}};
    }
    case IntegrationMethod::DIRK33: {
      // 1 step, 3 stages, 3rd order, L-stable.
      // gamma = 0.158983899989 is L(α)-stable with smaller error coefficients than 0.435866521508.
      real const gamma = 0.43586652150845899941601945_r;
      real const b1 = -1.5_r * Sqr(gamma) + 4_r * gamma - 0.25_r;
      real const b2 = 1.5_r * Sqr(gamma) - 5_r * gamma + 1.25_r;
      real const c2 = (1_r + gamma) / 2_r;
      return TimeIntegratorParams{
          1,
          {{1_r}},
          1_r,
          3,
          {{gamma, 0_r, 0_r}, {c2 - gamma, gamma, 0_r}, {b1, b2, gamma}},
          {{b1, b2, gamma}},
          {{gamma, c2, 1_r}}};
    }
    case IntegrationMethod::SymplecticDIRK12: {
      // 1 step, 1 stage, 2nd order, A-stable, symplectic. Also known as Gauss and implicit
      // midpoint. Symplecticity is lost with variable step size unless a supplementary procedure is
      // used.
      return TimeIntegratorParams{1, {{1_r}}, 1_r, 1, {{0.5_r}}, {{1_r}}, {{0.5_r}}};
    }
    case IntegrationMethod::SymplecticDIRK22: {
      // 1 step, 2 stages, 2nd order, A-stable, symplectic. Symplecticity is lost with variable step
      // size unless a supplementary procedure is used.
      return TimeIntegratorParams{
          1,
          {{1_r}},
          1_r,
          2,
          {{0.25_r, 0_r}, {0.5_r, 0.25_r}},
          {{0.5_r, 0.5_r}},
          {{0.25_r, 0.75_r}}};
    }
    default: {
      MOCHI_ASSERT(false, "Time integration method (%i) not supported.", static_cast<int>(method));
    }
  }
  static_assert(
      static_cast<int>(IntegrationMethod::Count) == 8,
      "Please update the switch statement above if IntegrationMethod enumerator changes");

  return TimeIntegratorParams{0, {}, 0_r, 0, {}, {}, {}};
}

[[nodiscard]] inline int GetNumSteps(IntegrationMethod const& method) {
  switch (method) {
    case IntegrationMethod::BackwardEuler:
    case IntegrationMethod::DIRK22:
    case IntegrationMethod::DIRK23:
    case IntegrationMethod::DIRK33:
    case IntegrationMethod::SymplecticDIRK12:
    case IntegrationMethod::SymplecticDIRK22: {
      return 1;
    }
    case IntegrationMethod::BDF2: {
      return 2;
    }
    case IntegrationMethod::BDF3: {
      return 3;
    }
    default: {
      MOCHI_ASSERT(false, "Time integration method (%i) not supported.", static_cast<int>(method));
      return {};
    }
  }
  static_assert(
      static_cast<int>(IntegrationMethod::Count) == 8,
      "Please update the switch statement above if IntegrationMethod enumerator changes");
}

[[nodiscard]] inline int GetNumStages(IntegrationMethod const& method) {
  switch (method) {
    case IntegrationMethod::BackwardEuler:
    case IntegrationMethod::BDF2:
    case IntegrationMethod::BDF3:
    case IntegrationMethod::SymplecticDIRK12: {
      return 1;
    }
    case IntegrationMethod::DIRK22:
    case IntegrationMethod::DIRK23:
    case IntegrationMethod::SymplecticDIRK22: {
      return 2;
    }
    case IntegrationMethod::DIRK33: {
      return 3;
    }
    default: {
      MOCHI_ASSERT(false, "Time integration method (%i) not supported.", static_cast<int>(method));
      return {};
    }
  }
  static_assert(
      static_cast<int>(IntegrationMethod::Count) == 8,
      "Please update the switch statement above if IntegrationMethod enumerator changes");
}

} // namespace mochi
