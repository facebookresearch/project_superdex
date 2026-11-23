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

TEST(DrRandomTest, SplitMix64GoldenOutputsAndWrappedState) {
  uint64_t state = 0;
  EXPECT_EQ(SplitMix64(state), 0xe220a8397b1dcdafull);
  EXPECT_EQ(state, 0x9e3779b97f4a7c15ull);
  state = UINT64_MAX;
  EXPECT_EQ(SplitMix64(state), 0xe4d971771b652c20ull);
  EXPECT_EQ(state, 0x9e3779b97f4a7c14ull);
}

TEST(DrRandomTest, PreservesNativeDrawSequenceAndStateAdvancement) {
  struct Golden {
    uint64_t initial;
    double uniform;
    double normal;
    std::array<double, 4> quaternion;
    uint64_t final;
  };
  std::array<Golden, 3> const goldens{{
      {0,
       2.4165540410682134,
       1.0482038676528576,
       {0.10572360733745068, 0.13394230361513149, 0.87130805772518172, -0.46011329767460507},
       13064056694810536062ull},
      {123,
       1.5324561088185336,
       1.7423070317180693,
       {-0.51511955894409023, -0.21875443526032828, -0.7188048973758705, -0.41245345986625565},
       13064056694810536185ull},
      {UINT64_MAX,
       2.4697146014159221,
       0.42076043798948504,
       {-0.72814970763971709, -0.20871883915695796, -0.58231848517664508, 0.29519422634384934},
       13064056694810536061ull},
  }};
  for (auto const& golden : goldens) {
    uint64_t state = golden.initial;
    EXPECT_DOUBLE_EQ(Uniform(state, -2.0, 3.0), golden.uniform);
    EXPECT_EQ(state, golden.initial + 0x9e3779b97f4a7c15ull);
    EXPECT_NEAR(Normal(state), golden.normal, 1e-12);
    EXPECT_EQ(state, golden.initial + 3 * 0x9e3779b97f4a7c15ull);
    auto const quaternion = UniformQuat(state);
    for (std::size_t i = 0; i < quaternion.size(); ++i) {
      EXPECT_NEAR(quaternion[i], golden.quaternion[i], 1e-12);
    }
    EXPECT_EQ(state, golden.final);
  }
}
