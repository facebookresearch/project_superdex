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
#include <superdex_robotics/sensors/sensor_graph.h>
#include <superdex_robotics/superdex_robotics.h>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>

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
  graph.Remove(removed, error);
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

TEST(SensorGraphTest, RemovePrunesDependantsAndPreservesIndependentPublicationAndClock) {
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
  auto const removedOutputs = graph.Remove(removedHandles, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_EQ(removedOutputs.size(), 1);
  EXPECT_EQ(removedOutputs[0].name, "removed-grandchild");
  EXPECT_EQ(graph.Find("removed"), nullptr);
  EXPECT_EQ(graph.Find("removed-output"), nullptr);
  EXPECT_EQ(graph.Find("removed-grandchild"), nullptr);
  EXPECT_EQ(graph.Find("retained-output"), &retainedOutput);
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

TEST(SensorGraphTest, AddedNodesJoinCurrentStepWithoutReinitializingExistingSensors) {
  Error error;
  CountingSource retained(error);
  CountingSource added(error);
  MetadataSource notReady(error);
  SensorGraph graph;
  graph.Add("retained", &retained, error);
  graph.SetOutputEnabled("retained", true, error);
  graph.Build({}, error);
  graph.Reset(error);
  StepTime const last{.t = 5.0, .dt = 0.01, .step = 500};
  ASSERT_TRUE(graph.Step(last, error)) << error.ToString();
  SensorSample const sample = retained.Latest();

  std::array<SensorGraphNode, 2> const additions{
      SensorGraphNode{.name = "added", .sensor = &added},
      SensorGraphNode{.name = "not-ready", .sensor = &notReady}};
  graph.Extend(additions, Inputs({"added", "not-ready"}), error);
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

TEST(SensorGraphTest, RemoveIgnoresNonmembersAndRejectsInvalidHandlesBeforeChangingTheGraph) {
  Error error;
  CountingSource source(error);
  SensorHandle const sourceHandle{1};
  SensorGraph graph;
  graph.Add("source", sourceHandle, &source, {}, error);
  graph.SetOutputEnabled("source", true, error);
  graph.Build({}, error);
  ASSERT_TRUE(graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, error)) << error.ToString();
  std::array<SensorHandle, 1> const nonmember{SensorHandle{2}};
  graph.Remove(nonmember, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  Error removeError;
  std::array<SensorHandle, 2> const invalid{sourceHandle, SensorHandle{}};
  graph.Remove(invalid, removeError);
  EXPECT_FALSE(removeError.IsOK());
  EXPECT_EQ(graph.Find("source"), &source);
  ASSERT_TRUE(graph.Step({.t = 0.02, .dt = 0.01, .step = 2}, error)) << error.ToString();
  EXPECT_EQ(source.stepCalls, 2);
}

TEST(SensorGraphTest, ExtensionWithNoEnabledNamesLeavesAddedNodesInactive) {
  Error error;
  CountingSource existing(error);
  CountingSource added(error);
  SensorGraph graph;
  graph.Add("existing", &existing, error);
  graph.SetOutputEnabled("existing", true, error);
  graph.Build({}, error);
  graph.Reset(error);
  ASSERT_TRUE(graph.Step({.t = 5.0, .dt = 0.01, .step = 500}, error)) << error.ToString();
  std::array<SensorGraphNode, 1> const nodes{SensorGraphNode{.name = "added", .sensor = &added}};
  graph.Extend(nodes, {}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_EQ(graph.GetEnabledNodes().size(), 1);
  EXPECT_EQ(graph.GetEnabledNodes()[0].name, "existing");
  ASSERT_TRUE(graph.Step({.t = 5.01, .dt = 0.01, .step = 501}, error)) << error.ToString();
  EXPECT_EQ(existing.initCalls, 1);
  EXPECT_EQ(existing.resetCalls, 1);
  EXPECT_EQ(added.initCalls, 0);
  EXPECT_EQ(added.stepCalls, 0);

  graph.Extend({}, Inputs({"added"}), error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_TRUE(graph.Step({.t = 5.02, .dt = 0.01, .step = 502}, error)) << error.ToString();
  EXPECT_EQ(existing.initCalls, 1);
  EXPECT_EQ(existing.resetCalls, 1);
  EXPECT_EQ(added.initCalls, 1);
  EXPECT_EQ(added.resetCalls, 1);
  EXPECT_EQ(added.stepCalls, 1);
  EXPECT_EQ(added.Latest().captureStep, 502);
}

TEST(SensorGraphTest, BuildWithNoEnabledOutputsCanExtendWithAnEnabledDependentSensor) {
  Error error;
  CountingSource source(error);
  DoubleSensor derived(error);
  SensorGraph graph;
  graph.Add("source", &source, error);
  graph.Build({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(graph.Find("source"), &source);
  EXPECT_TRUE(graph.GetEnabledNodes().empty());
  EXPECT_EQ(source.initCalls, 0);
  ASSERT_TRUE(graph.Step({.t = 5.0, .dt = 0.01, .step = 500}, error)) << error.ToString();
  EXPECT_EQ(source.stepCalls, 0);

  auto const inputs = Inputs({"source"});
  std::array<SensorGraphNode, 1> const nodes{
      SensorGraphNode{.name = "derived", .sensor = &derived, .inputs = inputs}};
  graph.Extend(nodes, Inputs({"derived"}), error);
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

TEST(SensorGraphTest, FailedExtensionPreservesTheOriginalGraphAndClock) {
  Error error;
  CountingSource existing(error);
  DoubleSensor rejected(error);
  SensorGraph graph;
  graph.Add("existing", &existing, error);
  graph.SetOutputEnabled("existing", true, error);
  graph.Build({}, error);
  graph.Reset(error);
  StepTime const now{.t = 5.0, .dt = 0.01, .step = 500};
  ASSERT_TRUE(graph.Step(now, error)) << error.ToString();
  SensorSample const sample = existing.Latest();
  auto const missing = Inputs({"missing"});
  std::array<SensorGraphNode, 1> const nodes{
      SensorGraphNode{.name = "rejected", .sensor = &rejected, .inputs = missing}};
  Error extensionError;
  graph.Extend(nodes, Inputs({"rejected"}), extensionError);
  ASSERT_FALSE(extensionError.IsOK());
  EXPECT_EQ(graph.Find("rejected"), nullptr);
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
  missing.Build({}, missingError);
  EXPECT_FALSE(missingError.IsOK());
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
  first.Build({}, buildError);
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
