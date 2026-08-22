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

#include <mochi_core/linear_algebra/krylov/stopping_criterion.h>

#include <gtest/gtest.h>

#include <limits>

using namespace mochi;

static krylov::IterationStatus CheckStatusFor(real residualNormSqr) {
  krylov::details::StopTest<real> stop(1e-3_r, 1e-6_r, 10_r);
  stop.SetScaling(1_r);
  return stop.CheckStatus(0, residualNormSqr);
}

TEST(StopTest, ClassifiesFiniteResidualNormSquared) {
  EXPECT_EQ(krylov::IterationStatus::DivergedRes, CheckStatusFor(-1_r));
  EXPECT_EQ(
      krylov::IterationStatus::ConvergedAtol,
      CheckStatusFor(-std::numeric_limits<real>::epsilon()));
  EXPECT_EQ(krylov::IterationStatus::ConvergedAtol, CheckStatusFor(1e-14_r));
  EXPECT_EQ(krylov::IterationStatus::ConvergedRtol, CheckStatusFor(1e-8_r));
  EXPECT_EQ(krylov::IterationStatus::Active, CheckStatusFor(1e-3_r));
  EXPECT_EQ(krylov::IterationStatus::DivergedRes, CheckStatusFor(101_r));
}

TEST(StopTest, ClassifiesNonFiniteResidualNormSquared) {
  EXPECT_EQ(
      krylov::IterationStatus::DivergedRes, CheckStatusFor(std::numeric_limits<real>::quiet_NaN()));
  EXPECT_EQ(
      krylov::IterationStatus::DivergedRes, CheckStatusFor(std::numeric_limits<real>::infinity()));
}

TEST(StopTest, StoresLatestResidualNormSquared) {
  krylov::details::StopTest<real> stop(1e-3_r, 1e-6_r, 10_r);
  stop.SetScaling(1_r);

  stop.CheckStatus(0, 0.25_r);
  EXPECT_EQ(0.25_r, stop.GetLatestResidualNormSqr());

  // A tiny negative residual-norm-squared (round-off, within the convergence threshold) is clamped
  // to 0 rather than stored verbatim.
  stop.CheckStatus(0, -std::numeric_limits<real>::epsilon() / 2_r);
  EXPECT_EQ(0_r, stop.GetLatestResidualNormSqr());
}
