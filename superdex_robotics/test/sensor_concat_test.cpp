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

#include <superdex_robotics/sensors/sensor_concat.h>
#include <superdex_robotics/sensors/sensor_graph.h>

#include <mochi_core/test/mochi_test_helpers.h>

#include <gtest/gtest.h>

#include <array>
#include <utility>

using namespace mochi;
using namespace mochi::test;
using namespace superdex::robotics;

namespace {

class JawSource final : public SensorBase {
 public:
  explicit JawSource(
      SensorFormat format =
          {.kind = "DasFingerTactile",
           .shape = {4},
           .axes =
               {{.name = "component",
                 .labels = {"normal", "tangential", "direction", "mutual_prox"},
                 .units = {"N", "N", "rad", ""}}}})
      : SensorBase(ExpectOK{}), _format(std::move(format)) {}

  std::string_view GetTypeName() const override {
    return "TEST_JAW_SOURCE";
  }

  SensorStepResult Step(StepTime const&, superdex::Error&) override {
    return SensorStepResult::NoUpdate;
  }

  void Capture(StepTime const& now, float normal) {
    std::array<float, 4> const values{normal, 2.0F, 3.0F, 0.0F};
    PublishSample(now, values, ExpectOK{});
  }

  void Republish(StepTime const& now, float normal) {
    SensorSample const source = Latest();
    std::array<float, 4> const values{normal, 2.0F, 3.0F, 0.0F};
    PublishSample(now, values, ExpectOK{}, &source);
  }

 private:
  SensorFormat InitSensor(SensorScope const&, superdex::Error&) override {
    return _format;
  }
  void ResetSensor() override {}
  SensorFormat _format;
};

void BuildConcat(
    SensorGraph& graph,
    JawSource& left,
    JawSource& right,
    SensorConcat& concat,
    superdex::Error& error) {
  graph.Add("left", &left, error);
  graph.Add("right", &right, error);
  graph.Add("concat", &concat, DynamicArray<DynamicString>{"left", "right"}, error);
  graph.SetOutputEnabled("concat", true, error);
  graph.Build({}, error);
}

} // namespace

TEST(SensorConcatTest, HoldsTheOtherJawAndKeepsBothCaptureClocks) {
  JawSource left;
  JawSource right;
  SensorConcat concat({.axisName = "jaw", .inputLabels = {"left", "right"}}, ExpectOK{});
  SensorGraph graph;
  BuildConcat(graph, left, right, concat, ExpectOK{});
  graph.Reset(ExpectOK{});
  EXPECT_EQ("DasFingerTactile", concat.GetFormat().kind);
  EXPECT_EQ((DynamicArray<SensorDimension>{2, 4}), concat.GetFormat().shape);
  EXPECT_EQ("jaw", concat.GetFormat().axes[0].name);
  EXPECT_EQ((DynamicArray<DynamicString>{"left", "right"}), concat.GetFormat().axes[0].labels);
  EXPECT_EQ(left.GetFormat().axes[0], concat.GetFormat().axes[1]);

  left.Capture({.t = 1.0, .dt = 0.01, .step = 100}, 1.0F);
  EXPECT_EQ(SensorStepResult::NoUpdate, concat.Step({.t = 1.0, .step = 100}, ExpectOK{}));
  right.Capture({.t = 1.01, .dt = 0.01, .step = 101}, 4.0F);
  EXPECT_EQ(SensorStepResult::Published, concat.Step({.t = 1.01, .step = 101}, ExpectOK{}));
  EXPECT_EQ(1.0F, concat.Latest().Floats()[0]);
  EXPECT_EQ(4.0F, concat.Latest().Floats()[4]);
  EXPECT_DOUBLE_EQ(1.0, concat.GetInputCaptureTimes()[0]);
  EXPECT_DOUBLE_EQ(1.01, concat.GetInputCaptureTimes()[1]);
  EXPECT_EQ(100, concat.GetInputCaptureSteps()[0]);
  EXPECT_EQ(101, concat.GetInputCaptureSteps()[1]);
  EXPECT_DOUBLE_EQ(1.01, concat.Latest().captureTime);

  left.Capture({.t = 1.02, .dt = 0.01, .step = 102}, 5.0F);
  // The output owns its values: changing an upstream buffer cannot change it before publication.
  EXPECT_EQ(1.0F, concat.Latest().Floats()[0]);
  EXPECT_EQ(SensorStepResult::Published, concat.Step({.t = 1.02, .step = 102}, ExpectOK{}));
  EXPECT_EQ(5.0F, concat.Latest().Floats()[0]);
  EXPECT_EQ(4.0F, concat.Latest().Floats()[4]);
  EXPECT_DOUBLE_EQ(1.02, concat.GetInputCaptureTimes()[0]);
  EXPECT_DOUBLE_EQ(1.01, concat.GetInputCaptureTimes()[1]);
  EXPECT_EQ(SensorStepResult::NoUpdate, concat.Step({.t = 1.03, .step = 103}, ExpectOK{}));
  EXPECT_EQ(102, concat.Latest().publishedStep);

  // A new upstream publication can retain its capture identity, for example after a delay.
  left.Republish({.t = 1.04, .step = 104}, 6.0F);
  EXPECT_EQ(SensorStepResult::Published, concat.Step({.t = 1.04, .step = 104}, ExpectOK{}));
  EXPECT_EQ(6.0F, concat.Latest().Floats()[0]);
  EXPECT_EQ(102, concat.GetInputCaptureSteps()[0]);
  EXPECT_EQ(104, concat.Latest().publishedStep);

  graph.Reset(ExpectOK{});
  EXPECT_EQ(-1, concat.Latest().captureStep);
  EXPECT_EQ(-1, concat.GetInputCaptureSteps()[0]);
  EXPECT_EQ(-1, concat.GetInputCaptureSteps()[1]);
  EXPECT_EQ(SensorStepResult::NoUpdate, concat.Step({.t = 0, .step = 0}, ExpectOK{}));
}

TEST(SensorConcatTest, RejectsDifferentComponentMeaning) {
  JawSource left;
  JawSource right(
      {.kind = "DasFingerTactile",
       .shape = {4},
       .axes = {
           {.name = "component",
            .labels = {"normal", "tangential", "direction", "mutual_prox"},
            .units = {"N", "N", "degrees", ""}}}});
  SensorConcat concat(SensorConcat::Params{}, ExpectOK{});
  SensorGraph graph;
  BuildConcat(graph, left, right, concat, ExpectNotOK{});
}

TEST(SensorConcatTest, RejectsMissingOrMislabelledInputs) {
  SensorConcat empty(SensorConcat::Params{}, ExpectOK{});
  SensorGraph graph;
  graph.Add("concat", &empty, ExpectOK{});
  graph.SetOutputEnabled("concat", true, ExpectOK{});
  graph.Build({}, ExpectNotOK{});

  JawSource left;
  JawSource right;
  SensorConcat mislabelled({.axisName = "jaw", .inputLabels = {"left"}}, ExpectOK{});
  SensorGraph other;
  BuildConcat(other, left, right, mislabelled, ExpectNotOK{});
}

TEST(SensorConcatTest, RejectsDifferentShapesEvenWithTheSameElementCount) {
  JawSource left;
  JawSource right(
      {.kind = "DasFingerTactile",
       .shape = {2, 2},
       .axes = {{.name = "row"}, {.name = "component"}}});
  SensorConcat concat(SensorConcat::Params{}, ExpectOK{});
  SensorGraph graph;
  BuildConcat(graph, left, right, concat, ExpectNotOK{});
}
