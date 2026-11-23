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

#include <mochi_core/test/mochi_test_helpers.h>
#include <superdex_robotics/domain_randomization/derive.h>
#include <superdex_robotics/domain_randomization/random.h>
#include <superdex_robotics/sensors/sensor_function.h>
#include <superdex_robotics/sensors/sensor_graph.h>
#include <superdex_robotics/sensors/sensor_uniform_noise.h>

#include <array>
#include <string_view>

using namespace mochi;
using namespace superdex::robotics;

namespace {

/* Publishes {step, step + 0.5} at even graph steps and holds it at odd ones. */
class PairSource final : public SensorBase {
 public:
  explicit PairSource(Error& error) : SensorBase(error) {}
  [[nodiscard]] std::string_view GetTypeName() const override {
    return "TEST_PAIR_SOURCE";
  }
  [[nodiscard]] SensorStepResult Step(StepTime const& now, Error& error) override {
    if (now.step % 2 != 0) {
      return SensorStepResult::NoUpdate;
    }
    std::array<float, 2> const values{static_cast<float>(now.step), now.step + 0.5F};
    return PublishSample(now, values, error) ? SensorStepResult::Published
                                             : SensorStepResult::NoUpdate;
  }

 private:
  void ResetSensor() override {}
  SensorFormat InitSensor(SensorScope const& /*scope*/, Error& /*error*/) override {
    return {.kind = "Pair", .shape = {2}, .axes = {{.name = "value"}}};
  }
};

void Build(SensorGraph& graph, PairSource& source, SensorUniformNoise& noise) {
  graph.Add("source", &source, test::ExpectOK{});
  DynamicArray<DynamicString> const input{"source"};
  graph.Add("noise", &noise, input, test::ExpectOK{});
  graph.SetOutputEnabled("noise", true, test::ExpectOK{});
  graph.Build({}, test::ExpectOK{});
  graph.Reset(test::ExpectOK{});
}

DynamicArray<float> Values(SensorBase const& sensor) {
  return {sensor.Latest().Floats().begin(), sensor.Latest().Floats().end()};
}

// Each value gains a uniform draw from (episode seed, process key, the input's published step): a
// held sample keeps its noise, and a reset replays it.
TEST(SensorUniformNoiseTest, AddsADrawPerValueKeyedByThePublication) {
  uint64_t const seed = 0xfedcba9876543210ULL;
  PairSource source(test::ExpectOK{});
  SensorUniformNoise noise(
      {.low = -0.5, .high = 0.25, .random = {.episodeSeed = seed, .processKey = "index/noise"}},
      test::ExpectOK{});
  SensorGraph graph;
  Build(graph, source, noise);
  uint64_t state = Derive(seed, "index/noise", 2);
  float const first = 2.0F + static_cast<float>(Uniform(state, -0.5, 0.25));
  float const second = 2.5F + static_cast<float>(Uniform(state, -0.5, 0.25));

  graph.Step({.t = 0.02, .dt = 0.01, .step = 2}, test::ExpectOK{});
  DynamicArray<float> const noisy = Values(noise);
  graph.Step({.t = 0.03, .dt = 0.01, .step = 3}, test::ExpectOK{});
  DynamicArray<float> const held = Values(noise);
  graph.Reset(test::ExpectOK{});
  graph.Step({.t = 0.02, .dt = 0.01, .step = 2}, test::ExpectOK{});

  EXPECT_EQ(noisy, (DynamicArray<float>{first, second}));
  EXPECT_EQ(held, noisy);
  EXPECT_EQ(Values(noise), noisy);
  EXPECT_EQ(noise.Latest().captureStep, 2);
}

// Pins the stream: these values change only if Derive or Uniform does.
TEST(SensorUniformNoiseTest, DrawsPinnedValues) {
  PairSource source(test::ExpectOK{});
  SensorUniformNoise noise(
      {.low = -0.5, .high = 0.25, .random = {.episodeSeed = 7, .processKey = "golden"}},
      test::ExpectOK{});
  SensorGraph graph;
  Build(graph, source, noise);

  graph.Step({.t = 0.02, .dt = 0.01, .step = 2}, test::ExpectOK{});

  EXPECT_EQ(Values(noise), (DynamicArray<float>{1.73835325F, 2.23207951F}));
}

// A source that republishes a held measurement every step keeps its capture, and each publication
// gets its own draw.
TEST(SensorUniformNoiseTest, EachPublicationOfAHeldCaptureDrawsAgain) {
  SensorSample const held{.captureStep = 1, .captureTime = 0.01};
  SensorFunction source(
      SensorFunction::ValuesFormat(1),
      [&held](Span<float> out) {
        out[0] = 1.0F;
        return held;
      },
      test::ExpectOK{});
  SensorUniformNoise noise({.high = 1.0, .random = {.processKey = "held"}}, test::ExpectOK{});
  SensorGraph graph;
  graph.Add("source", &source, test::ExpectOK{});
  DynamicArray<DynamicString> const input{"source"};
  graph.Add("noise", &noise, input, test::ExpectOK{});
  graph.SetOutputEnabled("noise", true, test::ExpectOK{});
  graph.Build({}, test::ExpectOK{});
  graph.Reset(test::ExpectOK{});
  uint64_t state = Derive(0, "held", 3);
  float const expected = 1.0F + static_cast<float>(Uniform(state, 0.0, 1.0));

  graph.Step({.t = 0.02, .dt = 0.01, .step = 2}, test::ExpectOK{});
  DynamicArray<float> const first = Values(noise);
  graph.Step({.t = 0.03, .dt = 0.01, .step = 3}, test::ExpectOK{});

  EXPECT_NE(Values(noise), first);
  EXPECT_EQ(Values(noise), (DynamicArray<float>{expected}));
  EXPECT_EQ(noise.Latest().captureStep, 1);
  EXPECT_EQ(noise.Latest().publishedStep, 3);
}

// A node without a process key draws from the name its graph added it under, so two such nodes
// draw independently even in a graph nobody seeds; a node with a key keeps drawing from the key.
TEST(SensorUniformNoiseTest, AnEmptyProcessKeyDrawsFromTheGraphName) {
  uint64_t constexpr kSeed = 0;
  PairSource source(test::ExpectOK{});
  SensorUniformNoise left({.high = 1.0}, test::ExpectOK{});
  SensorUniformNoise right({.high = 1.0}, test::ExpectOK{});
  SensorUniformNoise keyed({.high = 1.0, .random = {.processKey = "k"}}, test::ExpectOK{});
  SensorGraph graph;
  graph.Add("source", &source, test::ExpectOK{});
  DynamicArray<DynamicString> const input{"source"};
  graph.Add("left/noise", &left, input, test::ExpectOK{});
  graph.Add("right/noise", &right, input, test::ExpectOK{});
  graph.Add("keyed/noise", &keyed, input, test::ExpectOK{});
  graph.SetOutputEnabled("left/noise", true, test::ExpectOK{});
  graph.SetOutputEnabled("right/noise", true, test::ExpectOK{});
  graph.SetOutputEnabled("keyed/noise", true, test::ExpectOK{});
  graph.Build({}, test::ExpectOK{});
  graph.Reset(test::ExpectOK{});
  auto const expected = [](std::string_view key) {
    uint64_t state = Derive(kSeed, key, 2);
    float const first = 2.0F + static_cast<float>(Uniform(state, 0.0, 1.0));
    float const second = 2.5F + static_cast<float>(Uniform(state, 0.0, 1.0));
    return DynamicArray<float>{first, second};
  };

  graph.Step({.t = 0.02, .dt = 0.01, .step = 2}, test::ExpectOK{});

  EXPECT_EQ(Values(left), expected("left/noise"));
  EXPECT_EQ(Values(right), expected("right/noise"));
  EXPECT_NE(Values(left), Values(right));
  EXPECT_EQ(Values(keyed), expected("k"));
}

// Params load from a scene file's JSON, bounds out of order are refused, and the process key may be
// left out.
// The graph's seed overrides the params' and survives SetParams: the two calls' order draws the
// same values, which differ from seed 0's.
TEST(SensorUniformNoiseTest, TheGraphsSeedSurvivesSetParams) {
  SensorUniformNoise::Params const params{.low = -0.5, .high = 0.25};
  auto const draw = [&params](uint64_t seed, bool seedFirst) {
    PairSource source(test::ExpectOK{});
    SensorUniformNoise noise(params, test::ExpectOK{});
    SensorGraph graph;
    Build(graph, source, noise);
    if (seedFirst) {
      graph.SetEpisodeSeed(seed);
    }
    noise.SetParams(params, test::ExpectOK{});
    if (!seedFirst) {
      graph.SetEpisodeSeed(seed);
    }
    graph.Reset(test::ExpectOK{});
    graph.Step({.t = 0.02, .dt = 0.01, .step = 2}, test::ExpectOK{});
    return Values(noise);
  };

  EXPECT_EQ(draw(7, true), draw(7, false));
  EXPECT_NE(draw(7, true), draw(0, true));
}

TEST(SensorUniformNoiseTest, LoadsAndChecksParams) {
  SensorUniformNoise noise(
      nullptr,
      R"({"low": -1, "high": 2, "random": {"episodeSeed": 7, "processKey": "k"}})",
      test::ExpectOK{});

  EXPECT_EQ(noise.GetParams().low, -1.0);
  EXPECT_EQ(noise.GetParams().high, 2.0);
  EXPECT_EQ(noise.GetParams().random.episodeSeed, 7);
  noise.SetParams({.low = 1, .high = 0, .random = {.processKey = "k"}}, test::ExpectNotOK{});
  EXPECT_EQ(noise.GetParams().low, -1.0);
  SensorUniformNoise unkeyed({}, test::ExpectOK{});
}

} // namespace
