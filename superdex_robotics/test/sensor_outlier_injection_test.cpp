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
#include <superdex_robotics/sensors/sensor_outlier_injection.h>

#include <cmath>

using namespace mochi;
using namespace superdex::robotics;

namespace {

/* Publishes `width` values step, step + 1, ... at even graph steps and holds them at odd ones. */
class CountingSource final : public SensorBase {
 public:
  CountingSource(int32_t width, Error& error) : SensorBase(error), _values(width) {}
  [[nodiscard]] std::string_view GetTypeName() const override {
    return "TEST_COUNTING_SOURCE";
  }
  [[nodiscard]] SensorStepResult Step(StepTime const& now, Error& error) override {
    if (now.step % 2 != 0) {
      return SensorStepResult::NoUpdate;
    }
    for (std::size_t i = 0; i < _values.size(); ++i) {
      _values[i] = static_cast<float>(now.step + i);
    }
    return PublishSample(now, MakeConstSpan(_values), error) ? SensorStepResult::Published
                                                             : SensorStepResult::NoUpdate;
  }

 private:
  void ResetSensor() override {}
  SensorFormat InitSensor(SensorScope const& /*scope*/, Error& /*error*/) override {
    return {
        .kind = "Counts",
        .shape = {static_cast<int32_t>(_values.size())},
        .axes = {{.name = "value"}}};
  }
  DynamicArray<float> _values;
};

void Build(
    SensorGraph& graph,
    CountingSource& source,
    SensorOutlierInjection& outliers,
    Error& error) {
  graph.Add("source", &source, test::ExpectOK{});
  DynamicArray<DynamicString> const input{"source"};
  graph.Add("outliers", &outliers, input, test::ExpectOK{});
  graph.SetOutputEnabled("outliers", true, test::ExpectOK{});
  graph.Build({}, error);
  graph.Reset(error);
}

DynamicArray<float> Values(SensorBase const& sensor) {
  return {sensor.Latest().Floats().begin(), sensor.Latest().Floats().end()};
}

// A replaced sample's values are uniform draws from (episode seed, process key, the input's
// published step), after the draw that decided the replacement; it keeps the input's capture
// identity, and a held sample keeps its outlier.
TEST(SensorOutlierInjectionTest, ReplacesACaptureByUniformValues) {
  CountingSource source(2, test::ExpectOK{});
  SensorOutlierInjection outliers(
      {.probability = 1.0, .range = 3.0, .random = {.episodeSeed = 7, .processKey = "cube"}},
      test::ExpectOK{});
  SensorGraph graph;
  Build(graph, source, outliers, test::ExpectOK{});
  uint64_t state = Derive(7, "cube", 4);
  (void)Uniform(state, 0.0, 1.0);
  auto const first = static_cast<float>(Uniform(state, -3.0, 3.0));
  auto const second = static_cast<float>(Uniform(state, -3.0, 3.0));

  graph.Step({.t = 0.04, .dt = 0.01, .step = 4}, test::ExpectOK{});
  DynamicArray<float> const replaced = Values(outliers);
  graph.Step({.t = 0.05, .dt = 0.01, .step = 5}, test::ExpectOK{});

  EXPECT_EQ(replaced, (DynamicArray<float>{first, second}));
  EXPECT_EQ(Values(outliers), replaced);
  EXPECT_EQ(outliers.Latest().captureStep, 4);
}

// Pins the stream: these values change only if Derive or Uniform does.
TEST(SensorOutlierInjectionTest, DrawsPinnedValues) {
  CountingSource source(2, test::ExpectOK{});
  SensorOutlierInjection outliers(
      {.probability = 1.0, .range = 3.0, .random = {.episodeSeed = 7, .processKey = "golden"}},
      test::ExpectOK{});
  SensorGraph graph;
  Build(graph, source, outliers, test::ExpectOK{});

  graph.Step({.t = 0.04, .dt = 0.01, .step = 4}, test::ExpectOK{});

  EXPECT_EQ(Values(outliers), (DynamicArray<float>{2.50519586F, 0.0337455757F}));
}

// A source that republishes a held measurement every step keeps its capture, and each publication
// gets its own outlier draw.
TEST(SensorOutlierInjectionTest, EachPublicationOfAHeldCaptureDrawsAgain) {
  SensorSample const held{.captureStep = 1, .captureTime = 0.01};
  SensorFunction source(
      SensorFunction::ValuesFormat(1),
      [&held](Span<float> out) {
        out[0] = 1.0F;
        return held;
      },
      test::ExpectOK{});
  SensorOutlierInjection outliers(
      {.probability = 1.0, .range = 3.0, .random = {.processKey = "held"}}, test::ExpectOK{});
  SensorGraph graph;
  graph.Add("source", &source, test::ExpectOK{});
  DynamicArray<DynamicString> const input{"source"};
  graph.Add("outliers", &outliers, input, test::ExpectOK{});
  graph.SetOutputEnabled("outliers", true, test::ExpectOK{});
  graph.Build({}, test::ExpectOK{});
  graph.Reset(test::ExpectOK{});
  uint64_t state = Derive(0, "held", 3);
  (void)Uniform(state, 0.0, 1.0);
  auto const expected = static_cast<float>(Uniform(state, -3.0, 3.0));

  graph.Step({.t = 0.02, .dt = 0.01, .step = 2}, test::ExpectOK{});
  DynamicArray<float> const first = Values(outliers);
  graph.Step({.t = 0.03, .dt = 0.01, .step = 3}, test::ExpectOK{});

  EXPECT_NE(Values(outliers), first);
  EXPECT_EQ(Values(outliers), (DynamicArray<float>{expected}));
  EXPECT_EQ(outliers.Latest().captureStep, 1);
  EXPECT_EQ(outliers.Latest().publishedStep, 3);
}

// With probability 0 every capture passes unchanged.
TEST(SensorOutlierInjectionTest, PassesOtherCapturesThrough) {
  CountingSource source(2, test::ExpectOK{});
  SensorOutlierInjection outliers(
      {.probability = 0.0, .range = 3.0, .random = {.processKey = "cube"}}, test::ExpectOK{});
  SensorGraph graph;
  Build(graph, source, outliers, test::ExpectOK{});

  graph.Step({.t = 0.04, .dt = 0.01, .step = 4}, test::ExpectOK{});

  EXPECT_EQ(Values(outliers), (DynamicArray<float>{4.0F, 5.0F}));
}

// A rotation outlier is two orthonormal rows.
TEST(SensorOutlierInjectionTest, ReplacesRotationsByTwoRowsOfARandomRotation) {
  CountingSource rows(6, test::ExpectOK{});
  SensorOutlierInjection outliers(
      {.probability = 1.0, .rotation = true, .random = {.processKey = "cube/rot"}},
      test::ExpectOK{});
  SensorGraph graph;
  Build(graph, rows, outliers, test::ExpectOK{});

  graph.Step({.t = 0.02, .dt = 0.01, .step = 2}, test::ExpectOK{});

  Span<float const> const r = outliers.Latest().Floats();
  EXPECT_NEAR(r[0] * r[0] + r[1] * r[1] + r[2] * r[2], 1.0, 1e-5);
  EXPECT_NEAR(r[3] * r[3] + r[4] * r[4] + r[5] * r[5], 1.0, 1e-5);
  EXPECT_NEAR(r[0] * r[3] + r[1] * r[4] + r[2] * r[5], 0.0, 1e-5);
}

// Rotation outliers need an input of 6 values, at build and when params change after it.
// The graph's seed overrides the params' and survives SetParams: the two calls' order draws the
// same outliers, which differ from seed 0's.
TEST(SensorOutlierInjectionTest, TheGraphsSeedSurvivesSetParams) {
  SensorOutlierInjection::Params const params{.probability = 1.0, .range = 3.0};
  auto const draw = [&params](uint64_t seed, bool seedFirst) {
    CountingSource source(2, test::ExpectOK{});
    SensorOutlierInjection outliers(params, test::ExpectOK{});
    SensorGraph graph;
    Build(graph, source, outliers, test::ExpectOK{});
    if (seedFirst) {
      graph.SetEpisodeSeed(seed);
    }
    outliers.SetParams(params, test::ExpectOK{});
    if (!seedFirst) {
      graph.SetEpisodeSeed(seed);
    }
    graph.Reset(test::ExpectOK{});
    graph.Step({.t = 0.04, .dt = 0.01, .step = 4}, test::ExpectOK{});
    return Values(outliers);
  };

  EXPECT_EQ(draw(7, true), draw(7, false));
  EXPECT_NE(draw(7, true), draw(0, true));
}

TEST(SensorOutlierInjectionTest, RotationNeedsSixValues) {
  CountingSource pair(2, test::ExpectOK{});
  SensorOutlierInjection rotation(
      {.probability = 1.0, .rotation = true, .random = {.processKey = "cube/rot"}},
      test::ExpectOK{});
  SensorGraph rotationGraph;
  CountingSource otherPair(2, test::ExpectOK{});
  SensorOutlierInjection uniform(
      {.probability = 1.0, .random = {.processKey = "x"}}, test::ExpectOK{});
  SensorGraph uniformGraph;
  Build(uniformGraph, otherPair, uniform, test::ExpectOK{});

  Build(rotationGraph, pair, rotation, test::ExpectNotOK{});
  uniform.SetParams(
      {.probability = 1.0, .rotation = true, .random = {.processKey = "x"}}, test::ExpectNotOK{});

  EXPECT_FALSE(uniform.GetParams().rotation);
}

// A probability outside [0, 1], a negative range or a missing process key is refused.
TEST(SensorOutlierInjectionTest, ChecksParams) {
  SensorOutlierInjection outliers(
      {.probability = 0.5, .random = {.processKey = "cube"}}, test::ExpectOK{});

  outliers.SetParams({.probability = 1.5, .random = {.processKey = "cube"}}, test::ExpectNotOK{});
  outliers.SetParams(
      {.probability = 0.5, .range = -1.0, .random = {.processKey = "cube"}}, test::ExpectNotOK{});
  SensorOutlierInjection unkeyed({}, test::ExpectOK{});

  EXPECT_EQ(outliers.GetParams().probability, 0.5);
}

} // namespace
