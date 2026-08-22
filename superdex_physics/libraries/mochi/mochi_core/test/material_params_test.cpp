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

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/materials/active_aniso_arap_params.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>

#include <gtest/gtest.h>

using namespace mochi;

static constexpr int kTestSize = 1000;
static constexpr real kTolerance = MOCHI_USE_DOUBLE_PRECISION ? 5e-7_r : 5e-4_r;

TEST(ActiveAnisoArapMaterialParams, AnisoDirectionRoundTrip) {
  ActiveAnisoArapMaterialParams params;
  for (int idx = 0; idx < kTestSize; ++idx) {
    auto paramRand = ColumnVector<real, 2>::Random(idx, -2 * kPI, 2 * kPI);
    params.anisoDir =
        ActiveAnisoArapMaterialParams::ComputeFiberDirection(paramRand(0), paramRand(1));
    EXPECT_NEAR_TOL(
        params.anisoDir,
        ActiveAnisoArapMaterialParams::ComputeFiberDirection(params.GetTheta(), params.GetPhi()),
        kTolerance);
  }

  // Edge cases: k*pi/2.
  DynamicArray<real> angles = {
      -2 * kPI, -3 * kPI / 2, -kPI, -kPI / 2, 0_r, kPI / 2, kPI, 3 * kPI / 2, 2 * kPI};
  for (real theta : angles) {
    for (real phi : angles) {
      params.anisoDir = ActiveAnisoArapMaterialParams::ComputeFiberDirection(theta, phi);
      EXPECT_NEAR_TOL(
          params.anisoDir,
          ActiveAnisoArapMaterialParams::ComputeFiberDirection(params.GetTheta(), params.GetPhi()),
          kTolerance);
    }
  }
}
