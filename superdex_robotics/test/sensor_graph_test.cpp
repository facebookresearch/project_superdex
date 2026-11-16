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

#include <superdex_robotics/sensors/sensor_graph.h>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string_view>

using namespace mochi;
using namespace superdex::robotics;

namespace superdex::robotics {

struct TestMeta {
  DynamicString value;
};

template <>
struct SensorMetadataKey<TestMeta> {
  static constexpr std::string_view Key = "test.meta.v1";
};

} // namespace superdex::robotics

namespace {

SensorFormat ScalarFormat() {
  return {
      .kind = "TestScalar",
      .shape = {1},
      .axes = {{.name = "value", .labels = {}, .units = {}}},
  };
}

DynamicArray<DynamicString> Inputs(std::initializer_list<std::string_view> names) {
  DynamicArray<DynamicString> result;
  for (std::string_view const name : names) {
    result.emplace_back(name);
  }
  return result;
}

class CountingSource final : public SensorBase {
 public:
  explicit CountingSource(Error& error) : SensorBase(error) {}

  [[nodiscard]] std::string_view GetTypeName() const override {
    return "TEST_COUNTING_SOURCE";
  }

  SensorStepResult Step(StepTime const& now, Error& error) override {
    MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
    ++stepCalls;
    std::array<float, 1> const value{static_cast<float>(now.step)};
    return PublishSample(now, value, error) ? SensorStepResult::Published
                                            : SensorStepResult::NoUpdate;
  }

  int initCalls = 0;
  int stepCalls = 0;
  int resetCalls = 0;

 protected:
  SensorFormat InitSensor(SensorScope const& /*scope*/, Error& error) override {
    MOCHI_ERROR_IF_NOT(Inputs().empty(), error, "CountingSource does not accept inputs");
    ++initCalls;
    return ScalarFormat();
  }

  void ResetSensor() override {
    ++resetCalls;
  }
};

class DoubleSensor final : public SensorBase {
 public:
  explicit DoubleSensor(Error& error) : SensorBase(error) {}

  [[nodiscard]] std::string_view GetTypeName() const override {
    return "TEST_DOUBLE";
  }

  SensorStepResult Step(StepTime const& now, Error& error) override {
    MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
    SensorSample const source = Inputs()[0]->Latest();
    if (source.captureStep < 0 || source.captureStep == Latest().captureStep) {
      return SensorStepResult::NoUpdate;
    }
    ++stepCalls;
    std::array<float, 1> const value{2.0F * source.Floats()[0]};
    return PublishSample(now, value, error, &source) ? SensorStepResult::Published
                                                     : SensorStepResult::NoUpdate;
  }

  int stepCalls = 0;

 protected:
  void ResetSensor() override {}

  SensorFormat InitSensor(SensorScope const& /*scope*/, Error& error) override {
    MOCHI_ERROR_IF_NOT(
        Inputs().size() == 1 && Inputs()[0]->GetFormat().kind == "TestScalar",
        error,
        "DoubleSensor requires one TestScalar input");
    return ScalarFormat();
  }
};

class MetadataSource final : public SensorBase {
 public:
  explicit MetadataSource(Error& error) : SensorBase(error) {}

  [[nodiscard]] std::string_view GetTypeName() const override {
    return "TEST_METADATA";
  }

  SensorStepResult Step(StepTime const& /*now*/, Error& /*error*/) override {
    return SensorStepResult::NoUpdate;
  }

  [[nodiscard]] Span<std::byte const> TryGetMetadata(std::string_view key) const override {
    if (key != SensorMetadataKey<TestMeta>::Key) {
      return {};
    }
    return {reinterpret_cast<std::byte const*>(&_meta), sizeof(_meta)};
  }

 protected:
  void ResetSensor() override {}

  SensorFormat InitSensor(SensorScope const& /*scope*/, Error& /*error*/) override {
    return ScalarFormat();
  }

 private:
  TestMeta _meta{.value = "borrowed metadata"};
};

class InvalidFormatSensor final : public SensorBase {
 public:
  explicit InvalidFormatSensor(Error& error) : SensorBase(error) {}

  [[nodiscard]] std::string_view GetTypeName() const override {
    return "TEST_INVALID_FORMAT";
  }

  SensorStepResult Step(StepTime const& /*now*/, Error& /*error*/) override {
    return SensorStepResult::NoUpdate;
  }

 protected:
  void ResetSensor() override {}

  SensorFormat InitSensor(SensorScope const& /*scope*/, Error& /*error*/) override {
    return {.kind = "Invalid", .shape = {2}, .axes = {}};
  }
};

class WrongSizeSource final : public SensorBase {
 public:
  explicit WrongSizeSource(Error& error) : SensorBase(error) {}

  [[nodiscard]] std::string_view GetTypeName() const override {
    return "TEST_WRONG_SIZE_SOURCE";
  }

  SensorStepResult Step(StepTime const& now, Error& error) override {
    std::array<float, 2> const value{};
    return PublishSample(now, value, error) ? SensorStepResult::Published
                                            : SensorStepResult::NoUpdate;
  }

 protected:
  void ResetSensor() override {}

  SensorFormat InitSensor(SensorScope const& /*scope*/, Error& /*error*/) override {
    return ScalarFormat();
  }
};

TEST(SensorGraphTest, BuildsTopologicallyAndPublishesBorrowedLatest) {
  Error constructionError;
  CountingSource source(constructionError);
  DoubleSensor derived(constructionError);
  ASSERT_TRUE(constructionError.IsOK());

  SensorGraph graph;
  graph.Add("derived", &derived, Inputs({"source"}));
  graph.Add("source", &source);
  graph.AddOutput("derived");
  Error error;
  graph.Build({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(source.initCalls, 1);

  graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(derived.Latest().captureStep, 1);
  EXPECT_DOUBLE_EQ(derived.Latest().captureTime, 0.01);
  EXPECT_FLOAT_EQ(derived.Latest().Floats()[0], 2.0F);
  EXPECT_EQ(derived.stepCalls, 1);

  SensorSample const firstRead = derived.Latest();
  SensorSample const secondRead = derived.Latest();
  EXPECT_EQ(firstRead.data, secondRead.data);
  EXPECT_EQ(derived.stepCalls, 1);
}

TEST(SensorGraphTest, FindOnlyReturnsNodesFromTheCurrentBuild) {
  Error constructionError;
  CountingSource source(constructionError);
  CountingSource addedLater(constructionError);
  ASSERT_TRUE(constructionError.IsOK());

  SensorGraph graph;
  graph.Add("source", &source);
  graph.AddOutput("source");
  EXPECT_EQ(graph.Find("source"), nullptr);

  Error error;
  graph.Build({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(graph.Find("source"), &source);
  EXPECT_EQ(graph.Find("missing"), nullptr);

  graph.Add("added-later", &addedLater);
  EXPECT_EQ(graph.Find("source"), nullptr);
  EXPECT_EQ(graph.Find("added-later"), nullptr);
}

TEST(SensorGraphTest, SelectedOutputPrunesUnneededNodes) {
  Error constructionError;
  CountingSource selected(constructionError);
  CountingSource unused(constructionError);
  ASSERT_TRUE(constructionError.IsOK());
  SensorGraph graph;
  graph.Add("selected", &selected);
  graph.Add("unused", &unused);
  graph.AddOutput("selected");
  Error error;
  graph.Build({}, error);
  graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(selected.stepCalls, 1);
  EXPECT_EQ(unused.initCalls, 0);
  EXPECT_EQ(unused.stepCalls, 0);
}

TEST(SensorGraphTest, RejectsMissingInputCycleDuplicateAndUnknownOutput) {
  Error constructionError;
  CountingSource a(constructionError);
  CountingSource b(constructionError);
  ASSERT_TRUE(constructionError.IsOK());

  SensorGraph missing;
  missing.Add("a", &a, Inputs({"missing"}));
  Error missingError;
  missing.Build({}, missingError);
  EXPECT_FALSE(missingError.IsOK());
  EXPECT_STREQ(
      "SensorGraph input name does not match any graph node", missingError.GetDescription());

  SensorGraph cycle;
  cycle.Add("a", &a, Inputs({"b"}));
  cycle.Add("b", &b, Inputs({"a"}));
  Error cycleError;
  cycle.Build({}, cycleError);
  EXPECT_FALSE(cycleError.IsOK());

  SensorGraph duplicate;
  duplicate.Add("a", &a);
  duplicate.Add("also_a", &a);
  Error duplicateError;
  duplicate.Build({}, duplicateError);
  EXPECT_FALSE(duplicateError.IsOK());

  SensorGraph unknownOutput;
  unknownOutput.Add("a", &a);
  unknownOutput.AddOutput("unknown");
  Error outputError;
  unknownOutput.Build({}, outputError);
  EXPECT_FALSE(outputError.IsOK());
  EXPECT_STREQ(
      "SensorGraph selected output name does not match any graph node",
      outputError.GetDescription());
}

TEST(SensorGraphTest, RepeatedStepIsIdempotentAndInconsistentTimeFails) {
  Error constructionError;
  CountingSource source(constructionError);
  ASSERT_TRUE(constructionError.IsOK());
  SensorGraph graph;
  graph.Add("source", &source);
  Error error;
  graph.Build({}, error);
  graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, error);
  graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(source.stepCalls, 1);

  graph.Step({.t = 0.02, .dt = 0.01, .step = 1}, error);
  EXPECT_FALSE(error.IsOK());
}

TEST(SensorGraphTest, ResetClearsPublicationBeforeSensorHook) {
  Error constructionError;
  CountingSource source(constructionError);
  ASSERT_TRUE(constructionError.IsOK());
  SensorGraph graph;
  graph.Add("source", &source);
  Error error;
  graph.Build({}, error);
  graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, error);
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(source.Latest().captureStep, -1);
  EXPECT_FLOAT_EQ(source.Latest().Floats()[0], 0.0F);
  EXPECT_EQ(source.resetCalls, 1);
}

TEST(SensorGraphTest, ResetSupportsExplicitNoOpSensorHook) {
  Error constructionError;
  CountingSource source(constructionError);
  DoubleSensor derived(constructionError);
  ASSERT_TRUE(constructionError.IsOK());

  SensorGraph graph;
  graph.Add("source", &source);
  graph.Add("derived", &derived, Inputs({"source"}));
  graph.AddOutput("derived");
  Error error;
  graph.Build({}, error);
  graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, error);
  graph.Reset(error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(derived.Latest().captureStep, -1);
  EXPECT_FLOAT_EQ(derived.Latest().Floats()[0], 0.0F);
}

TEST(SensorGraphTest, BuildValidatesTheImmutableFormatOnce) {
  Error constructionError;
  InvalidFormatSensor sensor(constructionError);
  ASSERT_TRUE(constructionError.IsOK());
  SensorGraph graph;
  graph.Add("sensor", &sensor);
  Error error;
  graph.Build({}, error);
  EXPECT_FALSE(error.IsOK());
}

TEST(SensorGraphTest, StepRejectsAValueCountThatDoesNotMatchTheFormat) {
  Error constructionError;
  WrongSizeSource source(constructionError);
  ASSERT_TRUE(constructionError.IsOK());
  SensorGraph graph;
  graph.Add("source", &source);
  graph.AddOutput("source");
  Error error;
  graph.Build({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_FALSE(graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, error));
  EXPECT_FALSE(error.IsOK());
  EXPECT_STREQ("Sensor sample element count does not match its format", error.GetDescription());
  EXPECT_EQ(source.Latest().captureStep, -1);
}

TEST(SensorGraphTest, UnpublishedInputIsNoUpdateRatherThanAnError) {
  Error constructionError;
  MetadataSource source(constructionError);
  DoubleSensor derived(constructionError);
  ASSERT_TRUE(constructionError.IsOK());
  SensorGraph graph;
  graph.Add("source", &source);
  graph.Add("derived", &derived, Inputs({"source"}));
  graph.AddOutput("derived");
  Error error;
  graph.Build({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_TRUE(graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, error));
  EXPECT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(derived.Latest().captureStep, -1);
}

TEST(SensorGraphTest, TypedMetadataQueryReturnsProducerOwnedObject) {
  Error constructionError;
  MetadataSource source(constructionError);
  ASSERT_TRUE(constructionError.IsOK());
  SensorGraph graph;
  graph.Add("source", &source);
  Error error;
  graph.Build({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  auto const* const meta = TryGetSensorMetadata<TestMeta>(source);
  ASSERT_NE(meta, nullptr);
  EXPECT_EQ(meta->value, "borrowed metadata");
  EXPECT_TRUE(source.TryGetMetadata("unknown").empty());
}

TEST(SensorGraphTest, BuildPreservesErrorsAndValidatesOutputNames) {
  SensorGraph graph;
  graph.AddOutput("");
  Error existing;
  MOCHI_ERROR_SET(existing, "existing error");
  auto const message = existing.ToString();
  graph.Build({}, existing);
  EXPECT_EQ(message, existing.ToString());
  Error emptyName;
  graph.Build({}, emptyName);
  EXPECT_FALSE(emptyName.IsOK());

  SensorGraph unknown;
  unknown.AddOutput("unknown");
  Error unknownName;
  unknown.Build({}, unknownName);
  EXPECT_FALSE(unknownName.IsOK());
}

} // namespace
