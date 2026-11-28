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
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/defer.h>
#include <mochi_physics/mochi_physics.h>
#include <superdex_robotics/sensors/sensor_delay_line.h>
#include <superdex_robotics/sensors/sensor_gaussian_noise.h>
#include <superdex_robotics/sensors/sensor_graph.h>
#include <superdex_robotics/sensors/sensor_outlier_injection.h>
#include <superdex_robotics/sensors/sensor_uniform_noise.h>
#include <superdex_robotics/superdex_robotics.h>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if MOCHI_ASSERT_ENABLED && MOCHI_PHYSICS_DYNAMICALLY_LINKED
#if MOCHI_PLATFORM_WINDOWS
#include <io.h>
#else
#include <unistd.h>
#endif
#endif

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

template <typename Call>
void ExpectMissingInputDiagnostic(std::string_view node, std::string_view input, Call const& call) {
#if MOCHI_PHYSICS_DYNAMICALLY_LINKED
  testing::internal::CaptureStdout();
#else
  std::string diagnostic;
  int errorLogs = 0;
  auto previous = GetLogCallback();
  MOCHI_DEFER(SetLogCallback(std::move(previous)));
  SetLogCallback([&](LogChannel channel, char const* message, char const*, int) {
    EXPECT_EQ(channel, LogChannel::Error);
    ++errorLogs;
    diagnostic = message;
  });
#endif
  call();
#if MOCHI_PHYSICS_DYNAMICALLY_LINKED
  std::string const diagnostic = testing::internal::GetCapturedStdout();
#else
  EXPECT_EQ(errorLogs, 1);
#endif
  std::string const expected = "SensorGraph node '" + std::string(node) +
      "' references missing input '" + std::string(input) + "'";
  EXPECT_NE(diagnostic.find(expected), std::string::npos);
}

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
  explicit CountingSource(Error& error, int* destructionCount = nullptr)
      : SensorBase(error), _destructionCount(destructionCount) {}
  using SensorBase::EpisodeSeed;
  using SensorBase::GraphName;

  ~CountingSource() override {
    if (_destructionCount != nullptr) {
      ++*_destructionCount;
    }
  }

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

 private:
  int* _destructionCount;
};

class DoubleSensor final : public SensorBase {
 public:
  explicit DoubleSensor(Error& error) : SensorBase(error) {}
  using SensorBase::GraphName;

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
  graph.Add("derived", &derived, Inputs({"source"}), mochi::test::ExpectOK{});
  graph.Add("source", &source, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("derived", true, mochi::test::ExpectOK{});
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

TEST(SensorGraphTest, EnumeratesEnabledNodesSeparatelyFromActiveDependencies) {
  Error error;
  CountingSource source(error);
  CountingSource unused(error);
  DoubleSensor derived(error);
  SensorHandle const sourceHandle{1};
  SensorHandle const derivedHandle{2};
  SensorGraph graph;
  graph.Add("derived", derivedHandle, &derived, Inputs({"source"}), error);
  graph.Add("source", sourceHandle, &source, {}, error);
  graph.Add("unused", &unused, error);
  graph.SetOutputEnabled("derived", true, error);
  EXPECT_TRUE(graph.GetEnabledNodes().empty());
  graph.Build({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  auto enabled = graph.GetEnabledNodes();
  ASSERT_EQ(enabled.size(), 1);
  EXPECT_EQ(enabled[0].name, "derived");
  EXPECT_EQ(enabled[0].handle, derivedHandle);
  EXPECT_EQ(enabled[0].sensor, &derived);
  ASSERT_TRUE(graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, error)) << error.ToString();
  EXPECT_FLOAT_EQ(enabled[0].sensor->Latest().Floats()[0], 2.0F);
  EXPECT_EQ(source.stepCalls, 1);
  EXPECT_EQ(unused.stepCalls, 0);

  graph.SetOutputEnabled(sourceHandle, true, error);
  enabled = graph.GetEnabledNodes();
  ASSERT_EQ(enabled.size(), 2);
  EXPECT_EQ(enabled[0].name, "source");
  EXPECT_EQ(enabled[1].name, "derived");
  std::array<SensorHandle, 1> const removed{sourceHandle};
  graph.UnbindNode(removed, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_TRUE(graph.GetEnabledNodes().empty());

  graph.SetOutputEnabled("unused", true, error);
  graph.Build({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  enabled = graph.GetEnabledNodes();
  ASSERT_EQ(enabled.size(), 1);
  EXPECT_EQ(enabled[0].sensor, &unused);
  EXPECT_FALSE(enabled[0].handle.IsValid());
}

TEST(SensorGraphTest, NamesEveryNodeOfTheBuiltGraphInAddedOrder) {
  CountingSource source(mochi::test::ExpectOK{});
  CountingSource unused(mochi::test::ExpectOK{});
  DoubleSensor derived(mochi::test::ExpectOK{});
  SensorGraph graph;
  graph.Add("source", &source, mochi::test::ExpectOK{});
  graph.Add("unused", &unused, mochi::test::ExpectOK{});
  graph.Add("derived", &derived, Inputs({"source"}), mochi::test::ExpectOK{});
  graph.SetOutputEnabled("derived", true, mochi::test::ExpectOK{});
  DynamicArray<std::string_view> const unbuilt = graph.GetNodeNames();

  graph.Build({}, mochi::test::ExpectOK{});

  EXPECT_TRUE(unbuilt.empty());
  EXPECT_EQ(graph.GetNodeNames(), (DynamicArray<std::string_view>{"source", "unused", "derived"}));
}

TEST(SensorGraphTest, FindOnlyReturnsNodesFromTheCurrentBuild) {
  Error constructionError;
  CountingSource source(constructionError);
  CountingSource addedLater(constructionError);
  ASSERT_TRUE(constructionError.IsOK());

  SensorGraph graph;
  graph.Add("source", &source, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("source", true, mochi::test::ExpectOK{});
  EXPECT_EQ(graph.Find("source"), nullptr);

  Error error;
  graph.Build({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(graph.Find("source"), &source);
  EXPECT_EQ(graph.Find("missing"), nullptr);

  graph.Add("added-later", &addedLater, mochi::test::ExpectOK{});
  EXPECT_EQ(graph.Find("source"), nullptr);
  EXPECT_EQ(graph.Find("added-later"), nullptr);
}

TEST(SensorGraphTest, UnbindReleasesDependantsAndPreservesIndependentPublicationAndClock) {
  Error error;
  CountingSource retained(error);
  CountingSource removed(error);
  DoubleSensor retainedOutput(error);
  DoubleSensor removedOutput(error);
  DoubleSensor removedGrandchild(error);
  SensorHandle const retainedHandle{1};
  SensorHandle const removedHandle{2};
  SensorGraph graph;
  graph.Add("retained", retainedHandle, &retained, {}, error);
  graph.Add("removed", removedHandle, &removed, {}, error);
  graph.Add("retained-output", &retainedOutput, Inputs({"retained"}), error);
  graph.Add("removed-output", &removedOutput, Inputs({"removed"}), error);
  graph.Add("removed-grandchild", &removedGrandchild, Inputs({"removed-output"}), error);
  graph.SetOutputEnabled("retained-output", true, error);
  graph.SetOutputEnabled("removed-grandchild", true, error);
  graph.Build({}, error);
  graph.Reset(error);
  graph.SetOutputEnabled(retainedHandle, true, error);
  StepTime const last{.t = 5.0, .dt = 0.01, .step = 500};
  ASSERT_TRUE(graph.Step(last, error)) << error.ToString();
  SensorSample const sample = retainedOutput.Latest();

  std::array<SensorHandle, 1> const removedHandles{removedHandle};
  auto const removedOutputs = graph.UnbindNode(removedHandles, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_EQ(removedOutputs.size(), 1);
  EXPECT_EQ(removedOutputs[0].name, "removed-grandchild");
  EXPECT_EQ(graph.Find("removed"), nullptr);
  EXPECT_EQ(graph.Find("removed-output"), nullptr);
  EXPECT_EQ(graph.Find("removed-grandchild"), nullptr);
  EXPECT_EQ(graph.Find("retained-output"), &retainedOutput);
  EXPECT_EQ(
      graph.GetNodeNames(),
      (DynamicArray<std::string_view>{
          "retained", "removed", "retained-output", "removed-output", "removed-grandchild"}));
  EXPECT_TRUE(removed.GraphName().empty());
  EXPECT_TRUE(removedOutput.GraphName().empty());
  EXPECT_TRUE(removedGrandchild.GraphName().empty());
  EXPECT_TRUE(graph.GetOutputEnabled(retainedHandle, error));
  EXPECT_EQ(retained.initCalls, 1);
  EXPECT_EQ(retained.resetCalls, 1);
  EXPECT_EQ(retainedOutput.Latest().data, sample.data);
  EXPECT_EQ(retainedOutput.Latest().captureStep, 500);
  EXPECT_DOUBLE_EQ(retainedOutput.Latest().captureTime, 5.0);
  EXPECT_FLOAT_EQ(retainedOutput.Latest().Floats()[0], 1000.0F);
  ASSERT_TRUE(graph.Step(last, error)) << error.ToString();
  EXPECT_EQ(retained.stepCalls, 1);
  ASSERT_TRUE(graph.Step({.t = 5.01, .dt = 0.01, .step = 501}, error)) << error.ToString();
  EXPECT_EQ(retained.stepCalls, 2);
  EXPECT_EQ(removed.stepCalls, 1);
  EXPECT_EQ(removedGrandchild.stepCalls, 1);
  EXPECT_FLOAT_EQ(retainedOutput.Latest().Floats()[0], 1002.0F);
}

TEST(SensorGraphTest, AddedOutputsJoinCurrentStepWithoutReinitializingExistingSensors) {
  Error error;
  CountingSource retained(error);
  CountingSource added(error);
  MetadataSource notReady(error);
  SensorGraph graph;
  graph.Add("retained", &retained, error);
  graph.Add("added", &added, error);
  graph.Add("not-ready", &notReady, error);
  graph.SetOutputEnabled("retained", true, error);
  graph.Build({}, error);
  graph.Reset(error);
  StepTime const last{.t = 5.0, .dt = 0.01, .step = 500};
  ASSERT_TRUE(graph.Step(last, error)) << error.ToString();
  SensorSample const sample = retained.Latest();

  graph.AddOutputs(Inputs({"added", "not-ready"}), error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(retained.initCalls, 1);
  EXPECT_EQ(retained.resetCalls, 1);
  EXPECT_EQ(retained.Latest().data, sample.data);
  EXPECT_FLOAT_EQ(retained.Latest().Floats()[0], 500.0F);
  EXPECT_EQ(added.initCalls, 1);
  EXPECT_EQ(added.resetCalls, 1);
  EXPECT_EQ(added.Latest().captureStep, -1);
  ASSERT_TRUE(graph.Step(last, error)) << error.ToString();
  EXPECT_EQ(retained.stepCalls, 1);
  EXPECT_EQ(added.stepCalls, 0);
  ASSERT_TRUE(graph.Step({.t = 5.01, .dt = 0.01, .step = 501}, error)) << error.ToString();
  EXPECT_EQ(added.stepCalls, 1);
  EXPECT_EQ(added.Latest().captureStep, 501);
  EXPECT_DOUBLE_EQ(added.Latest().captureTime, 5.01);
  EXPECT_EQ(notReady.Latest().captureStep, -1);
}

TEST(SensorGraphTest, UnbindIgnoresNonmembersAndRejectsInvalidHandlesBeforeChangingTheGraph) {
  Error error;
  CountingSource source(error);
  SensorHandle const sourceHandle{1};
  SensorGraph graph;
  graph.Add("source", sourceHandle, &source, {}, error);
  graph.SetOutputEnabled("source", true, error);
  graph.Build({}, error);
  ASSERT_TRUE(graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, error)) << error.ToString();
  std::array<SensorHandle, 1> const nonmember{SensorHandle{2}};
  graph.UnbindNode(nonmember, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  Error removeError;
  std::array<SensorHandle, 2> const invalid{sourceHandle, SensorHandle{}};
  graph.UnbindNode(invalid, removeError);
  EXPECT_FALSE(removeError.IsOK());
  EXPECT_EQ(graph.Find("source"), &source);
  ASSERT_TRUE(graph.Step({.t = 0.02, .dt = 0.01, .step = 2}, error)) << error.ToString();
  EXPECT_EQ(source.stepCalls, 2);
}

TEST(SensorGraphTest, EmptyOutputSelectionLeavesInactiveNodesInactive) {
  Error error;
  CountingSource existing(error);
  CountingSource added(error);
  SensorGraph graph;
  graph.Add("existing", &existing, error);
  graph.Add("added", &added, error);
  graph.SetOutputEnabled("existing", true, error);
  graph.Build({}, error);
  graph.Reset(error);
  ASSERT_TRUE(graph.Step({.t = 5.0, .dt = 0.01, .step = 500}, error)) << error.ToString();
  graph.AddOutputs({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_EQ(graph.GetEnabledNodes().size(), 1);
  EXPECT_EQ(graph.GetEnabledNodes()[0].name, "existing");
  ASSERT_TRUE(graph.Step({.t = 5.01, .dt = 0.01, .step = 501}, error)) << error.ToString();
  EXPECT_EQ(existing.initCalls, 1);
  EXPECT_EQ(existing.resetCalls, 1);
  EXPECT_EQ(added.initCalls, 0);
  EXPECT_EQ(added.stepCalls, 0);

  graph.AddOutputs(Inputs({"added"}), error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_TRUE(graph.Step({.t = 5.02, .dt = 0.01, .step = 502}, error)) << error.ToString();
  EXPECT_EQ(existing.initCalls, 1);
  EXPECT_EQ(existing.resetCalls, 1);
  EXPECT_EQ(added.initCalls, 1);
  EXPECT_EQ(added.resetCalls, 1);
  EXPECT_EQ(added.stepCalls, 1);
  EXPECT_EQ(added.Latest().captureStep, 502);
}

TEST(SensorGraphTest, NodeDeclarationOwnsInputsAfterSourceStorageIsReleased) {
  Error error;
  CountingSource source(error);
  DoubleSensor derived(error);
  SensorGraphNode node;
  {
    auto inputs = Inputs({"source"});
    node = {.name = "derived", .sensor = &derived, .inputs = inputs};
    inputs[0] = "missing";
  }
  SensorGraphNode const moved = std::move(node);
  SensorGraph graph;
  graph.Add("source", &source, error);
  graph.Add(moved, error);
  graph.SetOutputEnabled("derived", true, error);
  graph.Build({}, error);
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_TRUE(graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, error)) << error.ToString();
  EXPECT_FLOAT_EQ(derived.Latest().Floats()[0], 2.0F);
}

TEST(SensorGraphTest, BuildWithNoEnabledOutputsCanEnableADependentSensor) {
  Error error;
  CountingSource source(error);
  DoubleSensor derived(error);
  SensorGraph graph;
  graph.Add("source", &source, error);
  graph.Add("derived", &derived, Inputs({"source"}), error);
  graph.Build({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(graph.Find("source"), &source);
  EXPECT_TRUE(graph.GetEnabledNodes().empty());
  EXPECT_EQ(source.initCalls, 0);
  ASSERT_TRUE(graph.Step({.t = 5.0, .dt = 0.01, .step = 500}, error)) << error.ToString();
  EXPECT_EQ(source.stepCalls, 0);

  graph.AddOutputs(Inputs({"derived"}), error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(source.initCalls, 1);
  ASSERT_EQ(graph.GetEnabledNodes().size(), 1);
  EXPECT_EQ(graph.GetEnabledNodes()[0].name, "derived");
  ASSERT_TRUE(graph.Step({.t = 5.01, .dt = 0.01, .step = 501}, error)) << error.ToString();
  EXPECT_EQ(source.stepCalls, 1);
  EXPECT_EQ(derived.stepCalls, 1);
  EXPECT_EQ(derived.Latest().captureStep, 501);
  EXPECT_FLOAT_EQ(derived.Latest().Floats()[0], 1002.0F);
}

TEST(SensorGraphTest, OutputBatchRejectsMissingOrUnboundNodesBeforeActivatingAnything) {
  Error error;
  CountingSource existing(error);
  CountingSource added(error);
  SensorGraph graph;
  graph.Add("existing", &existing, error);
  graph.Add("added", &added, error);
  graph.AddNode("unbound", {}, error);
  graph.SetOutputEnabled("existing", true, error);
  graph.Build({}, error);
  graph.Reset(error);
  StepTime const now{.t = 5.0, .dt = 0.01, .step = 500};
  ASSERT_TRUE(graph.Step(now, error)) << error.ToString();
  SensorSample const sample = existing.Latest();

  for (std::string_view const unavailable : {"", "missing", "unbound"}) {
    Error selectionError;
    graph.AddOutputs(Inputs({"added", unavailable}), selectionError);
    EXPECT_FALSE(selectionError.IsOK());
    ASSERT_EQ(graph.GetEnabledNodes().size(), 1);
    EXPECT_EQ(graph.GetEnabledNodes()[0].name, "existing");
    EXPECT_EQ(existing.Latest().data, sample.data);
    EXPECT_EQ(existing.Latest().captureStep, 500);
    EXPECT_EQ(existing.initCalls, 1);
    EXPECT_EQ(existing.resetCalls, 1);
    EXPECT_EQ(added.initCalls, 0);
    ASSERT_TRUE(graph.Step(now, error)) << error.ToString();
    EXPECT_EQ(existing.stepCalls, 1);
  }

  graph.AddOutputs(Inputs({"added", "added", "existing"}), error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(graph.GetEnabledNodes().size(), 2);
  EXPECT_EQ(existing.initCalls, 1);
  EXPECT_EQ(existing.resetCalls, 1);
  EXPECT_EQ(added.initCalls, 1);
  EXPECT_EQ(added.resetCalls, 1);
  ASSERT_TRUE(graph.Step({.t = 5.01, .dt = 0.01, .step = 501}, error)) << error.ToString();
  EXPECT_EQ(existing.stepCalls, 2);
  EXPECT_EQ(added.stepCalls, 1);
}

TEST(SensorGraphTest, FailedOutputInitializationPreservesTheOriginalSelectionAndClock) {
  Error error;
  CountingSource existing(error);
  DoubleSensor rejected(error);
  SensorGraph graph;
  graph.Add("existing", &existing, error);
  graph.Add("rejected", &rejected, error);
  graph.SetOutputEnabled("existing", true, error);
  graph.Build({}, error);
  graph.Reset(error);
  StepTime const now{.t = 5.0, .dt = 0.01, .step = 500};
  ASSERT_TRUE(graph.Step(now, error)) << error.ToString();
  SensorSample const sample = existing.Latest();
  Error selectionError;
  graph.AddOutputs(Inputs({"rejected"}), selectionError);
  ASSERT_FALSE(selectionError.IsOK());
  EXPECT_NE(
      selectionError.ToString().find("DoubleSensor requires one TestScalar input"),
      std::string::npos);
  EXPECT_EQ(graph.Find("rejected"), &rejected);
  ASSERT_EQ(graph.GetEnabledNodes().size(), 1);
  EXPECT_EQ(graph.GetEnabledNodes()[0].name, "existing");
  EXPECT_EQ(existing.Latest().data, sample.data);
  EXPECT_FLOAT_EQ(existing.Latest().Floats()[0], 500.0F);
  EXPECT_EQ(existing.initCalls, 1);
  EXPECT_EQ(existing.resetCalls, 1);
  ASSERT_TRUE(graph.Step(now, error)) << error.ToString();
  EXPECT_EQ(existing.stepCalls, 1);
  ASSERT_TRUE(graph.Step({.t = 5.01, .dt = 0.01, .step = 501}, error)) << error.ToString();
  EXPECT_EQ(existing.stepCalls, 2);
}

// A sensor keeps its node's name only while the graph holds it: UnbindNode and Clear take the name
// back, so a removed keyless random node keeps no stale stream key.
TEST(SensorGraphTest, GraphNameLastsWhileAGraphHoldsTheSensor) {
  CountingSource kept(mochi::test::ExpectOK{});
  CountingSource removed(mochi::test::ExpectOK{});
  SensorHandle const removedHandle{1};
  SensorGraph graph;
  graph.Add("kept", &kept, mochi::test::ExpectOK{});
  graph.Add("removed", removedHandle, &removed, {}, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("kept", true, mochi::test::ExpectOK{});
  graph.Build({}, mochi::test::ExpectOK{});
  std::string const named{removed.GraphName()};
  std::array<SensorHandle, 1> const removedHandles{removedHandle};

  graph.UnbindNode(removedHandles, mochi::test::ExpectOK{});
  std::string const keptBeforeClear{kept.GraphName()};
  graph.Clear();

  EXPECT_EQ(named, "removed");
  EXPECT_EQ(removed.GraphName(), "");
  EXPECT_EQ(keptBeforeClear, "kept");
  EXPECT_EQ(kept.GraphName(), "");
}

TEST(SensorGraphTest, ASharedSensorKeepsTheFirstGraphsNameUntilTheLastGraphReleasesIt) {
  CountingSource shared(mochi::test::ExpectOK{});
  CountingSource other(mochi::test::ExpectOK{});
  SensorGraph first;
  SensorGraph second;
  first.Add("first", &shared, mochi::test::ExpectOK{});
  second.Add("other", &other, mochi::test::ExpectOK{});
  second.Build({}, mochi::test::ExpectOK{});
  EXPECT_EQ(shared.GraphName(), "first");
  second.Add("second", &shared, mochi::test::ExpectOK{});
  second.Clear();
  std::string const afterSecond{shared.GraphName()};
  first.Clear();

  EXPECT_EQ(afterSecond, "first");
  EXPECT_EQ(shared.GraphName(), "");
}

TEST(SensorGraphTest, AddRejectsInvalidNamesWithoutChangingTheBuiltGraph) {
  Error error;
  CountingSource source(error);
  CountingSource rejected(error);
  SensorGraph graph;
  graph.Add("source", &source, error);
  graph.SetOutputEnabled("source", true, error);
  graph.Build({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

#if MOCHI_PHYSICS_DYNAMICALLY_LINKED
  testing::internal::CaptureStdout();
#else
  std::string diagnostic;
  auto previousLogCallback = GetLogCallback();
  MOCHI_DEFER(SetLogCallback(std::move(previousLogCallback)));
  SetLogCallback([&](LogChannel, char const* message, char const*, int) { diagnostic = message; });
#endif
  Error duplicateError;
  graph.Add("source", &rejected, duplicateError);
#if MOCHI_PHYSICS_DYNAMICALLY_LINKED
  std::string const diagnostic = testing::internal::GetCapturedStdout();
#endif
  ASSERT_FALSE(duplicateError.IsOK());
  EXPECT_NE(diagnostic.find("duplicate node name 'source'"), std::string::npos);
  EXPECT_EQ(graph.Find("source"), &source);

  Error emptyError;
  graph.Add("", &rejected, emptyError);
  EXPECT_FALSE(emptyError.IsOK());
  Error nullError;
  graph.Add("null", nullptr, nullError);
  EXPECT_FALSE(nullError.IsOK());
  EXPECT_EQ(graph.Find("null"), nullptr);
  EXPECT_TRUE(graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, error));
  EXPECT_TRUE(error.IsOK()) << error.ToString();

  graph.Clear();
  graph.Add("source", &rejected, error);
  graph.SetOutputEnabled("source", true, error);
  graph.Build({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(graph.Find("source"), &rejected);
}

TEST(SensorGraphTest, SelectedOutputPrunesUnneededNodes) {
  Error constructionError;
  CountingSource selected(constructionError);
  CountingSource unused(constructionError);
  ASSERT_TRUE(constructionError.IsOK());
  SensorGraph graph;
  graph.Add("selected", &selected, mochi::test::ExpectOK{});
  graph.Add("unused", &unused, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("selected", true, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("unused", true, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("unused", false, mochi::test::ExpectOK{});
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
  missing.Add("a", &a, Inputs({"missing"}), mochi::test::ExpectOK{});
  Error missingError;
#if MOCHI_PHYSICS_DYNAMICALLY_LINKED
  testing::internal::CaptureStdout();
  missing.Build({}, missingError);
  std::string const diagnostic = testing::internal::GetCapturedStdout();
#else
  std::string diagnostic;
  {
    auto previousCallback = GetLogCallback();
    MOCHI_DEFER(SetLogCallback(std::move(previousCallback)));
    SetLogCallback(
        [&](LogChannel, char const* message, char const*, int) { diagnostic = message; });
    missing.Build({}, missingError);
  }
#endif
  EXPECT_FALSE(missingError.IsOK());
  EXPECT_NE(diagnostic.find("'a'"), std::string::npos);
  EXPECT_NE(diagnostic.find("'missing'"), std::string::npos);
  EXPECT_STREQ(
      "SensorGraph input name does not match any graph node", missingError.GetDescription());

  SensorGraph cycle;
  cycle.Add("a", &a, Inputs({"b"}), mochi::test::ExpectOK{});
  cycle.Add("b", &b, Inputs({"a"}), mochi::test::ExpectOK{});
  Error cycleError;
  cycle.Build({}, cycleError);
  EXPECT_FALSE(cycleError.IsOK());

  SensorGraph duplicate;
  duplicate.Add("a", &a, mochi::test::ExpectOK{});
  duplicate.Add("also_a", &a, mochi::test::ExpectOK{});
  Error duplicateError;
  duplicate.Build({}, duplicateError);
  EXPECT_FALSE(duplicateError.IsOK());

  SensorGraph unknownOutput;
  unknownOutput.Add("a", &a, mochi::test::ExpectOK{});
  unknownOutput.SetOutputEnabled("unknown", true, mochi::test::ExpectOK{});
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
  graph.Add("source", &source, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("source", true, mochi::test::ExpectOK{});
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
  graph.Add("source", &source, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("source", true, mochi::test::ExpectOK{});
  Error error;
  graph.Build({}, error);
  graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, error);
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(source.Latest().captureStep, -1);
  EXPECT_FLOAT_EQ(source.Latest().Floats()[0], 0.0F);
  EXPECT_EQ(source.resetCalls, 1);
}

TEST(SensorGraphTest, LastStepIsTheLastSuccessfulStepUntilReset) {
  CountingSource source(mochi::test::ExpectOK{});
  SensorGraph graph;
  graph.Add("source", &source, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("source", true, mochi::test::ExpectOK{});
  graph.Build({}, mochi::test::ExpectOK{});
  bool const steppedBeforeFirstStep = graph.LastStep().has_value();

  graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, mochi::test::ExpectOK{});
  graph.Step({.t = 0.0, .dt = 0.01, .step = 0}, mochi::test::ExpectNotOK{});
  std::optional<StepTime> const last = graph.LastStep();
  graph.Reset(mochi::test::ExpectOK{});

  EXPECT_FALSE(steppedBeforeFirstStep);
  ASSERT_TRUE(last.has_value());
  EXPECT_EQ(last->step, 1);
  EXPECT_DOUBLE_EQ(last->t, 0.01);
  EXPECT_DOUBLE_EQ(last->dt, 0.01);
  EXPECT_FALSE(graph.LastStep().has_value());
}

// One call seeds every node that draws random streams: a chain of them draws as when each node's
// params carry the seed.
TEST(SensorGraphTest, SetEpisodeSeedSeedsEveryRandomNode) {
  uint64_t constexpr kSeed = 0xfedcba9876543210ULL;
  // The noisy, jittered chain's output at each of 50 steps of 1 kHz, its nodes' params seeded with
  // `paramsSeed` and the graph with `graphSeed`, if any.
  auto const outputs = [](uint64_t paramsSeed, std::optional<uint64_t> graphSeed) {
    SensorRandomParams const random{.episodeSeed = paramsSeed};
    CountingSource source(mochi::test::ExpectOK{});
    SensorUniformNoise uniform(
        {.low = -1.0, .high = 1.0, .random = random}, mochi::test::ExpectOK{});
    SensorGaussianNoise gaussian({.std = {0.5}, .random = random}, mochi::test::ExpectOK{});
    SensorOutlierInjection outliers(
        {.probability = 0.5, .range = 3.0, .random = random}, mochi::test::ExpectOK{});
    SensorDelayLine delay(
        {.delayMs = 1.0, .jitterMs = 8.0, .random = random}, mochi::test::ExpectOK{});
    SensorGraph graph;
    graph.Add("source", &source, mochi::test::ExpectOK{});
    graph.Add("uniform", &uniform, Inputs({"source"}), mochi::test::ExpectOK{});
    graph.Add("gaussian", &gaussian, Inputs({"uniform"}), mochi::test::ExpectOK{});
    graph.Add("outliers", &outliers, Inputs({"gaussian"}), mochi::test::ExpectOK{});
    graph.Add("delay", &delay, Inputs({"outliers"}), mochi::test::ExpectOK{});
    graph.SetOutputEnabled("delay", true, mochi::test::ExpectOK{});
    graph.Build({}, mochi::test::ExpectOK{});
    if (graphSeed) {
      graph.SetEpisodeSeed(*graphSeed);
    }
    graph.Reset(mochi::test::ExpectOK{});
    std::vector<float> values;
    for (int64_t step = 1; step <= 50; ++step) {
      double const t = static_cast<double>(step) / 1000.0;
      graph.Step({.t = t, .dt = 0.001, .step = step}, mochi::test::ExpectOK{});
      values.push_back(delay.Latest().Floats()[0]);
    }
    return values;
  };

  std::vector<float> const seeded = outputs(0, kSeed);

  EXPECT_EQ(seeded, outputs(kSeed, std::nullopt));
  EXPECT_NE(seeded, outputs(0, std::nullopt));
}

TEST(SensorGraphTest, ResetSupportsExplicitNoOpSensorHook) {
  Error constructionError;
  CountingSource source(constructionError);
  DoubleSensor derived(constructionError);
  ASSERT_TRUE(constructionError.IsOK());

  SensorGraph graph;
  graph.Add("source", &source, mochi::test::ExpectOK{});
  graph.Add("derived", &derived, Inputs({"source"}), mochi::test::ExpectOK{});
  graph.SetOutputEnabled("derived", true, mochi::test::ExpectOK{});
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
  graph.Add("sensor", &sensor, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("sensor", true, mochi::test::ExpectOK{});
  Error error;
  graph.Build({}, error);
  EXPECT_FALSE(error.IsOK());
}

TEST(SensorGraphTest, StepRejectsAValueCountThatDoesNotMatchTheFormat) {
  Error constructionError;
  WrongSizeSource source(constructionError);
  ASSERT_TRUE(constructionError.IsOK());
  SensorGraph graph;
  graph.Add("source", &source, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("source", true, mochi::test::ExpectOK{});
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
  graph.Add("source", &source, mochi::test::ExpectOK{});
  graph.Add("derived", &derived, Inputs({"source"}), mochi::test::ExpectOK{});
  graph.SetOutputEnabled("derived", true, mochi::test::ExpectOK{});
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
  graph.Add("source", &source, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("source", true, mochi::test::ExpectOK{});
  Error error;
  graph.Build({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  auto const* const meta = TryGetSensorMetadata<TestMeta>(source);
  ASSERT_NE(meta, nullptr);
  EXPECT_EQ(meta->value, "borrowed metadata");
  EXPECT_TRUE(source.TryGetMetadata("unknown").empty());
}

TEST(SensorGraphTest, OutputSelectionPreservesErrorsAndValidatesNamesAtBuild) {
  SensorGraph graph;
  Error existing;
  MOCHI_ERROR_SET(existing, "existing error");
  auto const message = existing.ToString();
  graph.SetOutputEnabled("", true, existing);
  EXPECT_EQ(message, existing.ToString());
  Error emptyName;
  graph.SetOutputEnabled("", true, emptyName);
  EXPECT_FALSE(emptyName.IsOK());
  Error unknownName;
  graph.SetOutputEnabled("unknown", true, unknownName);
  ASSERT_TRUE(unknownName.IsOK());
  graph.Build({}, unknownName);
  EXPECT_FALSE(unknownName.IsOK());
}

TEST(SensorGraphTest, BuildPreservesErrorsAndValidatesOutputNames) {
  SensorGraph graph;
  graph.SetOutputEnabled("unknown", true, mochi::test::ExpectOK{});
  Error existing;
  MOCHI_ERROR_SET(existing, "existing error");
  auto const message = existing.ToString();
  graph.Build({}, existing);
  EXPECT_EQ(message, existing.ToString());
  Error unknownName;
  graph.Build({}, unknownName);
  EXPECT_FALSE(unknownName.IsOK());
}

#if MOCHI_ASSERT_ENABLED && MOCHI_PHYSICS_DYNAMICALLY_LINKED
void RedirectAssertionOutput() {
  std::fflush(stdout);
#if MOCHI_PLATFORM_WINDOWS
  ASSERT_NE(_dup2(_fileno(stderr), _fileno(stdout)), -1);
#else
  ASSERT_NE(dup2(fileno(stderr), fileno(stdout)), -1);
#endif
}
#endif

TEST(SensorGraphDeathTest, RefusesDestructionUntilEveryGraphReleasesTheSensor) {
  int destructionCount = 0;
  auto* context = CreateRoboticsContext();
  MOCHI_DEFER(DestroyRoboticsContext(context));
  context->RegisterSensorType("TEST_COUNTING_SOURCE", [&](Actor*, std::string_view, Error& error) {
    return new CountingSource(error, &destructionCount);
  });
  Error error;
  SensorHandle const handle =
      context->CreateSensor("TEST_COUNTING_SOURCE", nullptr, "source", "", error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  SensorBase* const source = context->GetSensor(handle);
  ASSERT_NE(source, nullptr);

#if !MOCHI_PHYSICS_DYNAMICALLY_LINKED
  int assertionCount = 0;
  auto previousCallback = GetAssertionFailureCallback();
  MOCHI_DEFER(SetAssertionFailureCallback(std::move(previousCallback)));
  SetAssertionFailureCallback([&](char const*, char const*, char const*, int) {
    ++assertionCount;
    return false;
  });
#endif

  auto const expectRefusedDestruction = [&] {
    EXPECT_EQ(destructionCount, 0);
#if MOCHI_ASSERT_ENABLED && MOCHI_PHYSICS_DYNAMICALLY_LINKED
    EXPECT_DEATH_IF_SUPPORTED(
        {
          RedirectAssertionOutput();
          context->DestroySensor(handle);
        },
        "context\\.cpp");
#else
    context->DestroySensor(handle);
    EXPECT_EQ(context->GetSensor(handle), source);
#endif
  };

  SensorGraph first;
  first.Add("source", source, Inputs({"missing"}), mochi::test::ExpectOK{});
  expectRefusedDestruction();
  Error buildError;
  ExpectMissingInputDiagnostic("source", "missing", [&] { first.Build({}, buildError); });
  ASSERT_FALSE(buildError.IsOK());
  expectRefusedDestruction();
  {
    SensorGraph second;
    second.Add("source", source, mochi::test::ExpectOK{});
    second.SetOutputEnabled("source", true, error);
    second.Build({}, error);
    second.Build({}, error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    first.Clear();
    first.Clear();
    expectRefusedDestruction();
    EXPECT_TRUE(second.Step({.t = 0.01, .dt = 0.01, .step = 1}, error));
    EXPECT_TRUE(error.IsOK()) << error.ToString();
  }
#if MOCHI_ASSERT_ENABLED && !MOCHI_PHYSICS_DYNAMICALLY_LINKED
  EXPECT_EQ(assertionCount, 3);
#endif
  EXPECT_EQ(destructionCount, 0);
  context->DestroySensor(handle);
  EXPECT_EQ(context->GetSensor(handle), nullptr);
  EXPECT_EQ(destructionCount, 1);
  context->DestroySensor(handle);
  EXPECT_EQ(destructionCount, 1);
}

TEST(SensorGraphDeathTest, RefusesBotDestructionWhileItsSensorIsBorrowed) {
  auto* physics = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(physics));
  Scene* const scene = physics->CreateScene("sensor_lifetime");
  auto* context = CreateRoboticsContext();
  MOCHI_DEFER(DestroyRoboticsContext(context));
  context->RegisterSensorType("TEST_COUNTING_SOURCE", [](Actor*, std::string_view, Error& error) {
    return new CountingSource(error);
  });
  Error error;
  auto prefab =
      LoadBotPrefabFromFile(mochi::test::GetAssetPath("bots/arms/fr3/fr3.superdex_bot"), error);
  Bot* const bot = CreateBot(scene, prefab, context, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(bot, nullptr);
  BotHandle const botHandle = bot->GetHandle();
  SensorHandle const handle = context->CreateSensor(
      "TEST_COUNTING_SOURCE", bot->GetArticulatedActor(), "source", "", error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  SensorBase* const source = context->GetSensor(handle);
  ASSERT_NE(source, nullptr);

#if !MOCHI_PHYSICS_DYNAMICALLY_LINKED
  auto previousCallback = GetAssertionFailureCallback();
  MOCHI_DEFER(SetAssertionFailureCallback(std::move(previousCallback)));
  SetAssertionFailureCallback([](char const*, char const*, char const*, int) { return false; });
#endif
  SensorGraph graph;
  graph.Add("source", source, mochi::test::ExpectOK{});
#if MOCHI_ASSERT_ENABLED && MOCHI_PHYSICS_DYNAMICALLY_LINKED
  EXPECT_DEATH_IF_SUPPORTED(
      {
        RedirectAssertionOutput();
        context->DestroyBot(scene, botHandle);
      },
      "Clear every SensorGraph before destroying its sensor's bot");
#else
  context->DestroyBot(scene, botHandle);
  EXPECT_EQ(context->GetBot(botHandle), bot);
  EXPECT_EQ(context->GetSensor(handle), source);
  EXPECT_NE(bot->GetArticulatedActor(), nullptr);
#endif
  graph.Clear();
  context->DestroyBot(scene, botHandle);
  EXPECT_EQ(context->GetBot(botHandle), nullptr);
  EXPECT_EQ(context->GetSensor(handle), nullptr);
}

} // namespace

TEST(SensorGraphTest, NodesRejectCyclesAndMissingInputsBeforeBinding) {
  auto const firstInputs = Inputs({"second"});
  auto const secondInputs = Inputs({"first"});
  SensorGraph graph;
  graph.AddNode("first", firstInputs, mochi::test::ExpectOK{});
  graph.AddNode("second", secondInputs, mochi::test::ExpectOK{});
  graph.Build({}, mochi::test::ExpectNotOK{});
  EXPECT_TRUE(graph.GetNodeNames().empty());

  graph.Clear();
  graph.AddNode("first", firstInputs, mochi::test::ExpectOK{});
  ExpectMissingInputDiagnostic(
      "first", "second", [&] { graph.Build({}, mochi::test::ExpectNotOK{}); });
}

TEST(SensorGraphTest, ProgrammaticBindingsCanArriveInReverseDependencyOrder) {
  auto const inputNames = Inputs({"source"});
  SensorGraph graph;
  graph.AddNode("source", {}, mochi::test::ExpectOK{});
  graph.AddNode("double", inputNames, mochi::test::ExpectOK{});
  graph.Build({}, mochi::test::ExpectOK{});
  EXPECT_EQ(graph.GetNodeNames().size(), 2);
  EXPECT_EQ(graph.Find("source"), nullptr);
  EXPECT_FALSE(graph.InputsReady("double"));

  CountingSource source(mochi::test::ExpectOK{});
  DoubleSensor doubled(mochi::test::ExpectOK{});
  MOCHI_DEFER(graph.Clear());
  graph.BindNode(
      SensorGraphNode{.name = "double", .sensor = &doubled, .inputs = Inputs({"source"})},
      mochi::test::ExpectOK{});
  EXPECT_EQ(source.initCalls, 0);
  EXPECT_FALSE(graph.InputsReady("double"));
  graph.BindNode(SensorGraphNode{.name = "source", .sensor = &source}, mochi::test::ExpectOK{});
  EXPECT_TRUE(graph.InputsReady("double"));
  graph.SetOutputEnabled("double", true, mochi::test::ExpectOK{});
  graph.Build({}, mochi::test::ExpectOK{});
  EXPECT_EQ(source.initCalls, 1);
  EXPECT_TRUE(graph.Step({.t = 0.007, .dt = 0.001, .step = 7}, mochi::test::ExpectOK{}));
  EXPECT_FLOAT_EQ(doubled.Latest().Floats()[0], 14.0F);
}

TEST(SensorGraphTest, GraphNodesOwnInputNamesAfterCallerStorageIsDestroyed) {
  for (bool const deferred : {false, true}) {
    CountingSource source(mochi::test::ExpectOK{});
    DoubleSensor doubled(mochi::test::ExpectOK{});
    SensorGraph graph;
    {
      auto inputNames = Inputs({"source"});
      if (deferred) {
        graph.AddNode("source", {}, mochi::test::ExpectOK{});
        graph.AddNode("double", inputNames, mochi::test::ExpectOK{});
      } else {
        graph.Add("source", &source, mochi::test::ExpectOK{});
        graph.Add("double", &doubled, inputNames, mochi::test::ExpectOK{});
      }
      inputNames[0] = "not-in-the-graph";
    }
    graph.Build({}, mochi::test::ExpectOK{});
    if (deferred) {
      graph.BindNode(SensorGraphNode{.name = "source", .sensor = &source}, mochi::test::ExpectOK{});
      graph.BindNode(
          SensorGraphNode{.name = "double", .sensor = &doubled, .inputs = Inputs({"source"})},
          mochi::test::ExpectOK{});
    }
    graph.SetOutputEnabled("double", true, mochi::test::ExpectOK{});
    graph.Build({}, mochi::test::ExpectOK{});
    EXPECT_TRUE(graph.Step({.t = 0.007, .dt = 0.001, .step = 7}, mochi::test::ExpectOK{}));
    EXPECT_FLOAT_EQ(doubled.Latest().Floats()[0], 14.0F);
  }
}

TEST(SensorGraphTest, BindingChecksInputNamesWithoutChangingStoredWiring) {
  CountingSource source(mochi::test::ExpectOK{});
  DoubleSensor doubled(mochi::test::ExpectOK{});
  SensorGraph graph;
  graph.AddNode("source", {}, mochi::test::ExpectOK{});
  graph.AddNode("double", Inputs({"source"}), mochi::test::ExpectOK{});
  graph.Build({}, mochi::test::ExpectOK{});
  graph.BindNode(SensorGraphNode{.name = "source", .sensor = &source}, mochi::test::ExpectOK{});
  graph.BindNode(
      SensorGraphNode{.name = "double", .sensor = &doubled, .inputs = Inputs({"other"})},
      mochi::test::ExpectNotOK{});
  EXPECT_EQ(graph.Find("double"), nullptr);
  SensorGraphNode const binding{.name = "double", .sensor = &doubled, .inputs = Inputs({"source"})};
  graph.BindNode(binding, mochi::test::ExpectOK{});
  graph.BindNode(binding, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("double", true, mochi::test::ExpectOK{});
  graph.Build({}, mochi::test::ExpectOK{});
  EXPECT_TRUE(graph.Step({.t = 0.007, .dt = 0.001, .step = 7}, mochi::test::ExpectOK{}));
  EXPECT_FLOAT_EQ(doubled.Latest().Floats()[0], 14.0F);
}

TEST(SensorGraphTest, HandleSelectionRequiresReadyUpstreamBindings) {
  CountingSource source(mochi::test::ExpectOK{});
  DoubleSensor doubled(mochi::test::ExpectOK{});
  SensorHandle const sourceHandle{1};
  SensorHandle const doubledHandle{2};
  SensorGraph graph;
  graph.AddNode("source", {}, mochi::test::ExpectOK{});
  graph.AddNode("double", Inputs({"source"}), mochi::test::ExpectOK{});
  graph.Build({}, mochi::test::ExpectOK{});
  graph.BindNode(
      {.name = "double", .handle = doubledHandle, .sensor = &doubled, .inputs = Inputs({"source"})},
      mochi::test::ExpectOK{});
  graph.SetOutputEnabled(doubledHandle, true, mochi::test::ExpectNotOK{});
  EXPECT_FALSE(graph.GetOutputEnabled(doubledHandle, mochi::test::ExpectOK{}));
  graph.BindNode(
      {.name = "source", .handle = sourceHandle, .sensor = &source}, mochi::test::ExpectOK{});
  graph.SetOutputEnabled(doubledHandle, true, mochi::test::ExpectOK{});
  EXPECT_TRUE(graph.GetOutputEnabled(doubledHandle, mochi::test::ExpectOK{}));
  EXPECT_EQ(source.initCalls, 0);
  graph.Build({}, mochi::test::ExpectOK{});
  EXPECT_EQ(source.initCalls, 1);
  EXPECT_TRUE(graph.Step({.t = 0.007, .dt = 0.001, .step = 7}, mochi::test::ExpectOK{}));
  EXPECT_FLOAT_EQ(doubled.Latest().Floats()[0], 14.0F);
}

TEST(SensorGraphTest, UnbindingDoesNotApplyPendingOutputSelection) {
  CountingSource source(mochi::test::ExpectOK{});
  CountingSource unrelated(mochi::test::ExpectOK{});
  DoubleSensor doubled(mochi::test::ExpectOK{});
  SensorHandle const sourceHandle{1};
  SensorHandle const doubledHandle{2};
  SensorHandle const unrelatedHandle{3};
  SensorGraph graph;
  graph.AddNode("source", {}, mochi::test::ExpectOK{});
  graph.AddNode("double", Inputs({"source"}), mochi::test::ExpectOK{});
  graph.AddNode("unrelated", {}, mochi::test::ExpectOK{});
  graph.Build({}, mochi::test::ExpectOK{});
  graph.BindNode(
      {.name = "source", .handle = sourceHandle, .sensor = &source}, mochi::test::ExpectOK{});
  graph.BindNode(
      {.name = "double", .handle = doubledHandle, .sensor = &doubled, .inputs = Inputs({"source"})},
      mochi::test::ExpectOK{});
  graph.SetOutputEnabled(doubledHandle, true, mochi::test::ExpectOK{});
  graph.BindNode(
      {.name = "unrelated", .handle = unrelatedHandle, .sensor = &unrelated},
      mochi::test::ExpectOK{});
  std::array<SensorHandle, 1> const removed{unrelatedHandle};
  graph.UnbindNode(removed, mochi::test::ExpectOK{});
  EXPECT_FALSE(graph.GetOutputEnabled(unrelatedHandle, mochi::test::ExpectNotOK{}));
  EXPECT_EQ(source.initCalls, 0);
  EXPECT_TRUE(graph.Step({.t = 0.005, .dt = 0.001, .step = 5}, mochi::test::ExpectOK{}));
  EXPECT_EQ(source.stepCalls, 0);
  graph.Reset(mochi::test::ExpectOK{});
  EXPECT_EQ(source.initCalls, 1);
  EXPECT_EQ(source.resetCalls, 1);
  EXPECT_TRUE(graph.Step({.t = 0.006, .dt = 0.001, .step = 6}, mochi::test::ExpectOK{}));
  EXPECT_FLOAT_EQ(doubled.Latest().Floats()[0], 12.0F);
}

TEST(SensorGraphTest, NodesAddedOrBoundLaterReceiveTheCurrentEpisodeSeed) {
  CountingSource added(mochi::test::ExpectOK{});
  CountingSource bound(mochi::test::ExpectOK{});
  CountingSource lateAdded(mochi::test::ExpectOK{});
  SensorGraph graph;
  graph.SetEpisodeSeed(42);
  graph.Add("added", &added, mochi::test::ExpectOK{});
  graph.AddNode("bound", {}, mochi::test::ExpectOK{});
  graph.Build({}, mochi::test::ExpectOK{});
  graph.BindNode({.name = "bound", .sensor = &bound}, mochi::test::ExpectOK{});
  EXPECT_EQ(added.EpisodeSeed(0), 42);
  EXPECT_EQ(bound.EpisodeSeed(0), 42);
  graph.SetEpisodeSeed(43);
  graph.Add("late-added", &lateAdded, mochi::test::ExpectOK{});
  graph.Build({}, mochi::test::ExpectOK{});
  EXPECT_EQ(added.EpisodeSeed(0), 43);
  EXPECT_EQ(bound.EpisodeSeed(0), 43);
  EXPECT_EQ(lateAdded.EpisodeSeed(0), 43);
}

TEST(SensorGraphTest, UnbindingStopsUnusedUpstreamNodesButPreservesSharedInputs) {
  CountingSource source(mochi::test::ExpectOK{});
  DoubleSensor first(mochi::test::ExpectOK{});
  DoubleSensor second(mochi::test::ExpectOK{});
  SensorHandle const sourceHandle{1};
  SensorHandle const firstHandle{2};
  SensorHandle const secondHandle{3};
  SensorGraph graph;
  graph.Add("source", sourceHandle, &source, {}, mochi::test::ExpectOK{});
  graph.Add("first", firstHandle, &first, Inputs({"source"}), mochi::test::ExpectOK{});
  graph.Add("second", secondHandle, &second, Inputs({"source"}), mochi::test::ExpectOK{});
  graph.SetOutputEnabled("first", true, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("second", true, mochi::test::ExpectOK{});
  graph.Build({}, mochi::test::ExpectOK{});
  EXPECT_TRUE(graph.Step({.t = 0.001, .dt = 0.001, .step = 1}, mochi::test::ExpectOK{}));
  std::array<SensorHandle, 1> const removeFirst{firstHandle};
  graph.UnbindNode(removeFirst, mochi::test::ExpectOK{});
  EXPECT_TRUE(graph.Step({.t = 0.002, .dt = 0.001, .step = 2}, mochi::test::ExpectOK{}));
  EXPECT_EQ(source.stepCalls, 2);
  EXPECT_EQ(first.stepCalls, 1);
  EXPECT_EQ(second.stepCalls, 2);
  EXPECT_FLOAT_EQ(second.Latest().Floats()[0], 4.0F);
  std::array<SensorHandle, 1> const removeSecond{secondHandle};
  graph.UnbindNode(removeSecond, mochi::test::ExpectOK{});
  EXPECT_TRUE(graph.Step({.t = 0.003, .dt = 0.001, .step = 3}, mochi::test::ExpectOK{}));
  EXPECT_EQ(source.stepCalls, 2);
  EXPECT_EQ(source.initCalls, 1);
  EXPECT_EQ(source.resetCalls, 0);
  EXPECT_EQ(graph.Find("source"), &source);
  EXPECT_TRUE(graph.GetEnabledNodes().empty());
  ASSERT_TRUE(graph.LastStep().has_value());
  EXPECT_EQ(graph.LastStep()->step, 3);
}

TEST(SensorGraphTest, UnbindingDoesNotApplyPendingOutputDisablement) {
  CountingSource source(mochi::test::ExpectOK{});
  DoubleSensor doubled(mochi::test::ExpectOK{});
  SensorHandle const sourceHandle{1};
  SensorHandle const doubledHandle{2};
  SensorGraph graph;
  graph.Add("source", sourceHandle, &source, {}, mochi::test::ExpectOK{});
  graph.Add("double", doubledHandle, &doubled, Inputs({"source"}), mochi::test::ExpectOK{});
  graph.SetOutputEnabled("source", true, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("double", true, mochi::test::ExpectOK{});
  graph.Build({}, mochi::test::ExpectOK{});
  EXPECT_TRUE(graph.Step({.t = 0.001, .dt = 0.001, .step = 1}, mochi::test::ExpectOK{}));
  graph.SetOutputEnabled(sourceHandle, false, mochi::test::ExpectOK{});
  std::array<SensorHandle, 1> const removed{doubledHandle};
  graph.UnbindNode(removed, mochi::test::ExpectOK{});
  EXPECT_TRUE(graph.Step({.t = 0.002, .dt = 0.001, .step = 2}, mochi::test::ExpectOK{}));
  EXPECT_EQ(source.stepCalls, 2);
  graph.Reset(mochi::test::ExpectOK{});
  EXPECT_TRUE(graph.Step({.t = 0.003, .dt = 0.001, .step = 3}, mochi::test::ExpectOK{}));
  EXPECT_EQ(source.stepCalls, 2);
  EXPECT_EQ(source.resetCalls, 1);
}
