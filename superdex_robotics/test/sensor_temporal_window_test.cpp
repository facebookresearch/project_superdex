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
#include <superdex_robotics/sensors/sensor_graph.h>
#include <superdex_robotics/sensors/sensor_temporal_window.h>

#include <array>

using namespace mochi;
using namespace superdex::robotics;

namespace {

class PairSource final : public SensorBase {
 public:
  explicit PairSource(Error& error) : SensorBase(nullptr, error) {}

  static constexpr std::string_view TypeName() {
    return "TEST_PAIR_SOURCE";
  }

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  [[nodiscard]] SensorStepResult Step(StepTime const& now, Error& error) override {
    Span<SensorBase const* const> const inputs = Inputs();
    MOCHI_ERROR_IF_NOT(inputs.empty(), error, "PairSource accepts no inputs");
    MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
    std::array<float, 2> const values = {
        static_cast<float>(now.step), static_cast<float>(now.step) + 0.5F};
    return PublishSample(now, Span<float const>{values}, error) ? SensorStepResult::Published
                                                                : SensorStepResult::NoUpdate;
  }

 private:
  void ResetSensor() override {}

  [[nodiscard]] SensorFormat InitSensor(SensorScope const& /*scope*/, Error& error) override {
    MOCHI_ERROR_IF_NOT(Inputs().empty(), error, "PairSource accepts no inputs");
    MOCHI_ERROR_RETURN(error, {});
    return {
        .kind = "Pair",
        .shape = {2},
        .axes = {{.name = "value", .labels = {}, .units = {}}},
    };
  }
};

TEST(SensorTemporalWindowTest, PublishesOldestFirstFixedWindowWithRowIdentity) {
  Error constructionError;
  PairSource source(constructionError);
  SensorTemporalWindow buffer({.historyLength = 3}, constructionError);
  ASSERT_TRUE(constructionError.IsOK()) << constructionError.ToString();
  SensorGraph graph;
  graph.Add("source", &source, mochi::test::ExpectOK{});
  DynamicArray<DynamicString> const input = {"source"};
  graph.Add("history", &buffer, input, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("history", true, mochi::test::ExpectOK{});

  Error error;
  graph.Build({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(buffer.GetFormat().kind, "Pair");
  EXPECT_EQ(buffer.GetFormat().shape, (DynamicArray<int32_t>{3, 2}));
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  void const* allocation = nullptr;
  for (int frame = 1; frame <= 5; ++frame) {
    StepTime const now{.t = 0.01 * frame, .dt = 0.01, .step = frame};
    EXPECT_TRUE(graph.Step(now, error));
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    SensorSample const sample = buffer.Latest();
    ASSERT_EQ(sample.captureStep, frame);
    EXPECT_EQ(sample.publishedStep, frame);
    EXPECT_DOUBLE_EQ(sample.publishedTime, now.t);
    if (allocation == nullptr) {
      allocation = sample.data;
    }
    EXPECT_EQ(sample.data, allocation);
  }

  SensorSample const sample = buffer.Latest();
  ASSERT_EQ(sample.Floats().size(), 6);
  EXPECT_EQ(
      DynamicArray<float>(sample.Floats().begin(), sample.Floats().end()),
      (DynamicArray<float>{3.0F, 3.5F, 4.0F, 4.5F, 5.0F, 5.5F}));
  EXPECT_EQ(
      DynamicArray<int64_t>(buffer.GetRowCaptureSteps().begin(), buffer.GetRowCaptureSteps().end()),
      (DynamicArray<int64_t>{3, 4, 5}));
  EXPECT_EQ(
      DynamicArray<int64_t>(
          buffer.GetRowPublishedSteps().begin(), buffer.GetRowPublishedSteps().end()),
      (DynamicArray<int64_t>{3, 4, 5}));
  EXPECT_EQ(
      DynamicArray<double>(
          buffer.GetRowPublishedTimes().begin(), buffer.GetRowPublishedTimes().end()),
      (DynamicArray<double>{0.03, 0.04, 0.05}));
  EXPECT_TRUE(graph.Step({.t = 0.05, .dt = 0.01, .step = 5}, error));
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(buffer.Latest().data, allocation);
  EXPECT_EQ(buffer.Latest().captureStep, 5);
}

TEST(SensorTemporalWindowTest, PartialWindowIsRightAlignedAndResetClearsRows) {
  Error constructionError;
  PairSource source(constructionError);
  SensorTemporalWindow buffer({.historyLength = 3}, constructionError);
  ASSERT_TRUE(constructionError.IsOK()) << constructionError.ToString();
  SensorGraph graph;
  graph.Add("source", &source, mochi::test::ExpectOK{});
  DynamicArray<DynamicString> const input = {"source"};
  graph.Add("history", &buffer, input, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("history", true, mochi::test::ExpectOK{});
  Error error;
  graph.Build({}, error);
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  EXPECT_TRUE(graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, error));
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(
      DynamicArray<float>(buffer.Latest().Floats().begin(), buffer.Latest().Floats().end()),
      (DynamicArray<float>{0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.5F}));
  EXPECT_EQ(
      DynamicArray<int64_t>(buffer.GetRowCaptureSteps().begin(), buffer.GetRowCaptureSteps().end()),
      (DynamicArray<int64_t>{-1, -1, 1}));
  EXPECT_EQ(
      DynamicArray<int64_t>(
          buffer.GetRowPublishedSteps().begin(), buffer.GetRowPublishedSteps().end()),
      (DynamicArray<int64_t>{-1, -1, 1}));
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(buffer.Latest().captureStep, -1);
  EXPECT_EQ(buffer.Latest().publishedStep, -1);
  EXPECT_EQ(
      DynamicArray<int64_t>(buffer.GetRowCaptureSteps().begin(), buffer.GetRowCaptureSteps().end()),
      (DynamicArray<int64_t>{-1, -1, -1}));
}

// With padWithFirst, rows before the first sample since a reset repeat it, its capture included.
TEST(SensorTemporalWindowTest, PadWithFirstRepeatsTheFirstSample) {
  PairSource source(mochi::test::ExpectOK{});
  SensorTemporalWindow buffer({.historyLength = 3, .padWithFirst = true}, mochi::test::ExpectOK{});
  SensorGraph graph;
  graph.Add("source", &source, mochi::test::ExpectOK{});
  DynamicArray<DynamicString> const input = {"source"};
  graph.Add("history", &buffer, input, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("history", true, mochi::test::ExpectOK{});
  graph.Build({}, mochi::test::ExpectOK{});
  graph.Reset(mochi::test::ExpectOK{});

  graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, mochi::test::ExpectOK{});
  DynamicArray<float> const first(buffer.Latest().Floats().begin(), buffer.Latest().Floats().end());
  graph.Step({.t = 0.02, .dt = 0.01, .step = 2}, mochi::test::ExpectOK{});

  EXPECT_EQ(first, (DynamicArray<float>{1.0F, 1.5F, 1.0F, 1.5F, 1.0F, 1.5F}));
  EXPECT_EQ(
      DynamicArray<float>(buffer.Latest().Floats().begin(), buffer.Latest().Floats().end()),
      (DynamicArray<float>{1.0F, 1.5F, 1.0F, 1.5F, 2.0F, 2.5F}));
  EXPECT_EQ(
      DynamicArray<int64_t>(buffer.GetRowCaptureSteps().begin(), buffer.GetRowCaptureSteps().end()),
      (DynamicArray<int64_t>{1, 1, 2}));
  EXPECT_EQ(
      DynamicArray<double>(buffer.GetRowCaptureTimes().begin(), buffer.GetRowCaptureTimes().end()),
      (DynamicArray<double>{0.01, 0.01, 0.02}));
  EXPECT_EQ(
      DynamicArray<int64_t>(
          buffer.GetRowPublishedSteps().begin(), buffer.GetRowPublishedSteps().end()),
      (DynamicArray<int64_t>{1, 1, 2}));
  graph.Reset(mochi::test::ExpectOK{});
  graph.Step({.t = 0.03, .dt = 0.01, .step = 3}, mochi::test::ExpectOK{});
  EXPECT_EQ(
      DynamicArray<float>(buffer.Latest().Floats().begin(), buffer.Latest().Floats().end()),
      (DynamicArray<float>{3.0F, 3.5F, 3.0F, 3.5F, 3.0F, 3.5F}));
}

} // namespace
