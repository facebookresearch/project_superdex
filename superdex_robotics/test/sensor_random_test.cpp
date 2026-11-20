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

#include <superdex_robotics/domain_randomization/derive.h>
#include <superdex_robotics/domain_randomization/random.h>

#include <gtest/gtest.h>

using namespace superdex::robotics;

TEST(SensorRandomTest, DeriveMatchesExistingEpisodeAndAcquisitionGoldens) {
  EXPECT_EQ(Derive(0, "episode"), 0xB8585E8A9C0DBDCFULL);
  EXPECT_EQ(Derive(0, "episode", 0), 0x7EFA5F8AD6B623CBULL);
  EXPECT_EQ(Derive(0, "dr/scene/light"), 0x5030FA1CE5A18882ULL);
  EXPECT_EQ(Derive(1, "dr/scene/light"), 0x514779E16CBE3052ULL);
  EXPECT_EQ(Derive(1, "dr/scene/light", 0), 0x213A364625D030BDULL);
  EXPECT_EQ(Derive(1, "dr/scene/light", 1), 0x349CF06D1B43E6DFULL);
  EXPECT_EQ(Derive(12345, "dr/robot/obs_noise", 7), 0x77412850768BACA3ULL);
  EXPECT_EQ(Derive(UINT64_MAX, "dr/task/shape_pose", 100), 0x55834CAD71CF82E0ULL);
  EXPECT_EQ(Derive(2026, "dr/global/difficulty"), 0x9CED29E36394A848ULL);
  EXPECT_EQ(Derive(2026, "", 0), 0xD757873921833D22ULL);
}

TEST(SensorRandomTest, LocalStreamsReproduceWithoutDependingOnOtherDraws) {
  uint64_t first = Derive(42, "left/index/noise", 3);
  uint64_t second = first;
  uint64_t unrelated = Derive(42, "right/thumb/noise", 3);
  for (int i = 0; i < 100; ++i) {
    static_cast<void>(Normal(unrelated));
    EXPECT_DOUBLE_EQ(Normal(first), Normal(second));
  }
  EXPECT_NE(Derive(42, "left/index/noise", 3), Derive(42, "left/index/noise", 4));
}

TEST(SensorRandomTest, UniformAndNormalHaveExpectedRangesAndMoments) {
  uint64_t state = 123;
  double sum = 0.0;
  double squares = 0.0;
  for (int i = 0; i < 10000; ++i) {
    double const uniform = Uniform(state, -2.0, 3.0);
    ASSERT_GE(uniform, -2.0);
    ASSERT_LT(uniform, 3.0);
    double const normal = Normal(state);
    ASSERT_TRUE(std::isfinite(normal));
    sum += normal;
    squares += normal * normal;
  }
  EXPECT_NEAR(sum / 10000.0, 0.0, 0.05);
  EXPECT_NEAR(squares / 10000.0, 1.0, 0.08);
}

TEST(SensorRandomTest, UniformQuatIsUnitLengthAndHasNoPreferredComponent) {
  uint64_t state = 456;
  std::array<double, 4> squares{};
  for (int i = 0; i < 10000; ++i) {
    auto const quaternion = UniformQuat(state);
    double norm = 0.0;
    for (size_t component = 0; component < 4; ++component) {
      ASSERT_TRUE(std::isfinite(quaternion[component]));
      norm += quaternion[component] * quaternion[component];
      squares[component] += quaternion[component] * quaternion[component];
    }
    ASSERT_NEAR(norm, 1.0, 1e-12);
  }
  for (double const square : squares) {
    EXPECT_NEAR(square / 10000.0, 0.25, 0.02);
  }
}
