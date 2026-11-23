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
#include <superdex_robotics/sensors/sensor_function.h>
#include <superdex_robotics/sensors/sensor_graph.h>

#include <algorithm>
#include <optional>

using namespace mochi;
using namespace superdex::robotics;

namespace {

void Build(SensorGraph& graph, SensorFunction& source) {
  graph.Add("function", &source, test::ExpectOK{});
  graph.SetOutputEnabled("function", true, test::ExpectOK{});
  graph.Build({}, test::ExpectOK{});
  graph.Reset(test::ExpectOK{});
}

DynamicArray<float> Values(SensorBase const& sensor) {
  return {sensor.Latest().Floats().begin(), sensor.Latest().Floats().end()};
}

// Values captured at the step publish at every graph step, the reset step (dt = 0) included.
TEST(SensorFunctionTest, PublishesWhatItsFunctionWritesAtEveryStep) {
  int calls = 0;
  SensorFunction source(
      SensorFunction::ValuesFormat(2),
      [&calls](Span<float> out) -> std::optional<SensorSample> {
        ++calls;
        out[0] = static_cast<float>(calls);
        out[1] = -static_cast<float>(calls);
        return std::nullopt;
      },
      test::ExpectOK{});
  SensorGraph graph;
  Build(graph, source);

  graph.Step({.t = 0.0, .dt = 0.0, .step = 0}, test::ExpectOK{});
  DynamicArray<float> const atReset = Values(source);
  int64_t const resetCapture = source.Latest().captureStep;
  graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, test::ExpectOK{});

  EXPECT_EQ(source.GetFormat(), SensorFunction::ValuesFormat(2));
  EXPECT_EQ(atReset, (DynamicArray<float>{1.0F, -1.0F}));
  EXPECT_EQ(resetCapture, 0);
  EXPECT_EQ(Values(source), (DynamicArray<float>{2.0F, -2.0F}));
  EXPECT_EQ(source.Latest().captureStep, 1);
  EXPECT_EQ(source.Latest().publishedStep, 1);
}

// A mirror publishes in the format it is given, and its values keep the capture step and time of
// the sample they derive from while publishing every step; nothing publishes while that sample has
// no capture.
TEST(SensorFunctionTest, MirrorKeepsTheFormatAndCaptureOfWhatItReads) {
  SensorFormat const wrench = {
      .kind = "Wrench6D", .shape = {6}, .axes = {MakeWrenchComponentAxis()}};
  SensorSample measured;
  SensorFunction source(
      wrench,
      [&measured](Span<float> out) {
        std::fill(out.begin(), out.end(), 5.0F);
        return measured;
      },
      test::ExpectOK{});
  SensorGraph graph;
  Build(graph, source);

  graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, test::ExpectOK{});
  int64_t const unmeasured = source.Latest().publishedStep;
  measured.captureStep = 1;
  measured.captureTime = 0.01;
  graph.Step({.t = 0.02, .dt = 0.01, .step = 2}, test::ExpectOK{});
  graph.Step({.t = 0.03, .dt = 0.01, .step = 3}, test::ExpectOK{});

  EXPECT_EQ(source.GetFormat(), wrench);
  EXPECT_EQ(unmeasured, -1);
  EXPECT_EQ(Values(source), (DynamicArray<float>(6, 5.0F)));
  EXPECT_EQ(source.Latest().captureStep, 1);
  EXPECT_DOUBLE_EQ(source.Latest().captureTime, 0.01);
  EXPECT_EQ(source.Latest().publishedStep, 3);
  EXPECT_DOUBLE_EQ(source.Latest().publishedTime, 0.03);
}

TEST(SensorFunctionTest, RefusesAnInvalidFormatOrNoFunction) {
  auto const nothing = [](Span<float> /*out*/) -> std::optional<SensorSample> {
    return std::nullopt;
  };
  SensorFunction empty(SensorFunction::ValuesFormat(0), nothing, test::ExpectNotOK{});
  SensorFunction unnamed(
      {.kind = "Values", .shape = {1}, .axes = {SensorAxis{}}}, nothing, test::ExpectNotOK{});
  SensorFunction unread(
      SensorFunction::ValuesFormat(1), SensorFunction::Read{}, test::ExpectNotOK{});
}

} // namespace
