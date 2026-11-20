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

#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/json_utils.h>
#include <superdex_robotics/domain_randomization/derive.h>
#include <superdex_robotics/domain_randomization/random.h>
#include <superdex_robotics/sensors/sensor_gaussian_noise.h>
#include <superdex_robotics/sensors/sensor_graph.h>

#include <superdex_robotics/utils/file_utils.h>
#include <array>
#include <bit>

using namespace mochi;
using namespace superdex::robotics;

namespace {

class CadencedScalarSource final : public SensorBase {
 public:
  CadencedScalarSource(int64_t periodSteps, Error& error, int32_t elementCount = 1)
      : SensorBase(nullptr, error), _periodSteps(periodSteps), _elementCount(elementCount) {}

  [[nodiscard]] std::string_view GetTypeName() const override {
    return "TEST_CADENCED_SCALAR_SOURCE";
  }

  [[nodiscard]] SensorStepResult Step(StepTime const& now, Error& error) override {
    MOCHI_ERROR_IF_NOT(Inputs().empty(), error, "CadencedScalarSource accepts no inputs");
    MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
    if (now.step <= 0 || now.step % _periodSteps != 0) {
      return SensorStepResult::NoUpdate;
    }
    DynamicArray<float> const value(_elementCount, static_cast<float>(now.step));
    return PublishSample(now, Span<float const>{value}, error) ? SensorStepResult::Published
                                                               : SensorStepResult::NoUpdate;
  }

 private:
  void ResetSensor() override {}

  [[nodiscard]] SensorFormat InitSensor(SensorScope const& /*scope*/, Error& error) override {
    MOCHI_ERROR_IF_NOT(Inputs().empty(), error, "CadencedScalarSource accepts no inputs");
    MOCHI_ERROR_RETURN(error, {});
    return {
        .kind = "TestScalar",
        .shape = {_elementCount},
        .axes = {{.name = "value", .labels = {}, .units = {}}},
    };
  }

  int64_t _periodSteps;
  int32_t _elementCount;
};

void BuildNoiseGraph(
    SensorGraph& graph,
    CadencedScalarSource& source,
    SensorGaussianNoise& noise,
    Error& error) {
  graph.Add("source", &source, mochi::test::ExpectOK{});
  DynamicArray<DynamicString> const input = {"source"};
  graph.Add("noise", &noise, input, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("noise", true, mochi::test::ExpectOK{});
  graph.Build({}, error);
  graph.Reset(error);
}

void StepNoise(SensorGraph& graph, int64_t captureStep, Error& error) {
  EXPECT_TRUE(graph.Step({.t = 0.005 * captureStep, .dt = 0.005, .step = captureStep}, error));
  ASSERT_TRUE(error.IsOK()) << error.ToString();
}

TEST(SensorGaussianNoiseTest, ZeroNoisePreservesCaptureAndHeldSamples) {
  Error error;
  CadencedScalarSource source(2, error);
  SensorGaussianNoise noise({.std = {0.0F}, .random = {.processKey = "hand/index/noise"}}, error);
  SensorGraph graph;
  BuildNoiseGraph(graph, source, noise, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  StepNoise(graph, 1, error);
  EXPECT_EQ(noise.Latest().captureStep, -1);
  StepNoise(graph, 2, error);
  EXPECT_FLOAT_EQ(noise.Latest().Floats()[0], 2.0F);
  EXPECT_EQ(noise.Latest().captureStep, source.Latest().captureStep);
  EXPECT_EQ(noise.Latest().captureTime, source.Latest().captureTime);
  StepNoise(graph, 3, error);
  EXPECT_EQ(noise.Latest().publishedStep, 2);
  EXPECT_FLOAT_EQ(noise.Latest().Floats()[0], 2.0F);
}

TEST(SensorGaussianNoiseTest, UsesFullWidthSeedAndAcquisitionIndex) {
  Error error;
  CadencedScalarSource source(2, error);
  uint64_t const seed = 0xfedcba9876543210ULL;
  SensorGaussianNoise noise(
      {.std = {0.25}, .random = {.episodeSeed = seed, .processKey = "index/noise"}}, error);
  SensorGraph graph;
  BuildNoiseGraph(graph, source, noise, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  StepNoise(graph, 2, error);
  uint64_t state = Derive(seed, "index/noise", 2);
  auto const expected = static_cast<float>(2.0 + 0.25 * Normal(state));
  EXPECT_FLOAT_EQ(noise.Latest().Floats()[0], expected);
  StepNoise(graph, 3, error);
  EXPECT_FLOAT_EQ(noise.Latest().Floats()[0], expected);
  graph.Reset(error);
  StepNoise(graph, 2, error);
  EXPECT_FLOAT_EQ(noise.Latest().Floats()[0], expected);
}

TEST(SensorGaussianNoiseTest, InvalidTypedParamsDoNotChangeEffectiveConfiguration) {
  Error error;
  CadencedScalarSource source(1, error);
  SensorGaussianNoise noise({.std = {0.25}, .random = {.processKey = "noise"}}, error);
  SensorGraph graph;
  BuildNoiseGraph(graph, source, noise, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  auto params = noise.GetParams();
  params.std = {-1.0};
  Error negative;
  noise.SetParams(params, negative);
  EXPECT_FALSE(negative.IsOK());
  EXPECT_DOUBLE_EQ(noise.GetParams().std[0], 0.25);
  params.std = {0.1F, 0.2F};
  Error wrongShape;
  noise.SetParams(params, wrongShape);
  EXPECT_FALSE(wrongShape.IsOK());
  EXPECT_EQ(noise.GetParams().std.size(), 1);
}

TEST(SensorGaussianNoiseTest, RejectsMissingStreamIdentity) {
  Error error;
  SensorGaussianNoise noise({}, error);
  EXPECT_FALSE(error.IsOK());
}

class RepublishedCapture final : public SensorBase {
 public:
  explicit RepublishedCapture(Error& error) : SensorBase(error) {}
  int64_t captureStep = 1;
  double captureTime = 0.005;
  [[nodiscard]] std::string_view GetTypeName() const override {
    return "TEST_REPUBLISHED_CAPTURE";
  }
  [[nodiscard]] SensorStepResult Step(StepTime const& now, Error& error) override {
    std::array<float, 1> const value{1.0F};
    SensorSample const identity{.captureStep = captureStep, .captureTime = captureTime};
    return PublishSample(now, value, error, &identity) ? SensorStepResult::Published
                                                       : SensorStepResult::NoUpdate;
  }

 private:
  void ResetSensor() override {}
  SensorFormat InitSensor(SensorScope const& /*scope*/, Error& /*error*/) override {
    return {
        .kind = "TestScalar", .shape = {1}, .axes = {{.name = "value", .labels = {}, .units = {}}}};
  }
};

TEST(SensorGaussianNoiseTest, RepublishedCaptureDoesNotRedraw) {
  Error error;
  RepublishedCapture source(error);
  SensorGaussianNoise noise({.std = {0.25}, .random = {.processKey = "noise"}}, error);
  SensorGraph graph;
  graph.Add("source", &source, mochi::test::ExpectOK{});
  DynamicArray<DynamicString> const input{"source"};
  graph.Add("noise", &noise, input, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("noise", true, mochi::test::ExpectOK{});
  graph.Build({}, error);
  graph.Reset(error);
  StepNoise(graph, 1, error);
  float const first = noise.Latest().Floats()[0];
  StepNoise(graph, 2, error);
  EXPECT_EQ(source.Latest().publishedStep, 2);
  EXPECT_EQ(noise.Latest().publishedStep, 1);
  EXPECT_FLOAT_EQ(noise.Latest().Floats()[0], first);
  source.captureTime = 0.006;
  StepNoise(graph, 3, error);
  EXPECT_EQ(noise.Latest().publishedStep, 1);
  EXPECT_FLOAT_EQ(noise.Latest().Floats()[0], first);
}

TEST(SensorGaussianNoiseTest, ReflectedParamsPreserveFullWidthSeedsAndSigma) {
  for (uint64_t const seed :
       {0ULL, 42ULL, 0x8000000000000000ULL, 0xfedcba9876543210ULL, 0xffffffffffffffffULL}) {
    SensorGaussianNoiseParams params;
    params.std = {0.05F, 0.02F};
    params.random = {.episodeSeed = seed, .processKey = "index/noise"};
    picojson::value json;
    SReflect::ToJsonValue(params, json);
    EXPECT_EQ(
        json.get<picojson::object>()
            .at("random")
            .get<picojson::object>()
            .at("episodeSeed")
            .get<int64_t>(),
        std::bit_cast<int64_t>(seed));
    Error error;
    SensorGaussianNoise noise(nullptr, json.serialize(), error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    EXPECT_EQ(noise.GetParams().random.episodeSeed, seed);
    EXPECT_EQ(noise.GetParams().std, params.std);
  }
}

TEST(SensorGaussianNoiseTest, ZeroSigmaDoesNotShiftLaterElementsDraws) {
  Error error;
  CadencedScalarSource source(1, error, 3);
  SensorGaussianNoise noise(
      {.std = {0.0F, 0.1F, 0.2F}, .random = {.episodeSeed = 42, .processKey = "noise"}}, error);
  SensorGraph graph;
  BuildNoiseGraph(graph, source, noise, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  StepNoise(graph, 1, error);
  std::array<float, 3> const first{
      noise.Latest().Floats()[0], noise.Latest().Floats()[1], noise.Latest().Floats()[2]};
  EXPECT_EQ(first[0], 1.0F);
  auto params = noise.GetParams();
  params.std[0] = 0.01;
  noise.SetParams(params, error);
  graph.Reset(error);
  StepNoise(graph, 1, error);
  EXPECT_NE(noise.Latest().Floats()[0], first[0]);
  EXPECT_EQ(noise.Latest().Floats()[1], first[1]);
  EXPECT_EQ(noise.Latest().Floats()[2], first[2]);
}

TEST(SensorGaussianNoiseTest, Native120HzCapturesAreIndependentOfPolicyReads) {
  Error error;
  CadencedScalarSource source(2, error);
  SensorGaussianNoise noise(
      {.std = {0.25}, .random = {.episodeSeed = 42, .processKey = "index/noise"}}, error);
  SensorGraph graph;
  BuildNoiseGraph(graph, source, noise, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  DynamicArray<float> reference;
  for (int64_t policyPeriod : {12, 4, 1}) {
    graph.Reset(error);
    std::size_t captures = 0;
    for (int64_t captureStep = 1; captureStep <= 240; ++captureStep) {
      ASSERT_TRUE(
          graph.Step({.t = captureStep / 240.0, .dt = 1.0 / 240.0, .step = captureStep}, error));
      if (captureStep % policyPeriod == 0) {
        auto const first = noise.Latest();
        EXPECT_EQ(noise.Latest().captureStep, first.captureStep);
        EXPECT_EQ(noise.Latest().captureTime, first.captureTime);
      }
      if (captureStep % 2 == 0) {
        if (policyPeriod == 12) {
          reference.push_back(noise.Latest().Floats()[0]);
        } else {
          EXPECT_FLOAT_EQ(noise.Latest().Floats()[0], reference[captures]);
        }
        ++captures;
      }
    }
    EXPECT_EQ(captures, 120);
  }
}

TEST(SensorGaussianNoiseTest, DeclarativeParamsPreserveTheNativeProcessContext) {
  Error error;
  SensorGaussianNoise noise(
      nullptr,
      R"({"std":[0.25],"random":{"episodeSeed":-81985529216486896,"processKey":"index/noise"}})",
      error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(noise.GetParams().random.episodeSeed, 0xfedcba9876543210ULL);
  EXPECT_EQ(noise.GetParams().random.processKey, "index/noise");
  EXPECT_EQ(noise.GetParams().std[0], 0.25);
}

TEST(SensorGaussianNoiseTest, DeclarativeSeedRejectsAmbiguousOrMalformedValues) {
  auto const warnings = mochi::test::SuppressLogWarning();
  for (std::string_view seed :
       {"18364758544493064720",
        "18446744073709551616",
        "1.5",
        "\"-1\"",
        "\"42oops\"",
        "\"18446744073709551616\""}) {
    Error error;
    std::string const json =
        R"({"std":[0],"random":{"episodeSeed":)" + std::string{seed} + R"(,"processKey":"noise"}})";
    SensorGaussianNoise noise(nullptr, json, error);
    EXPECT_FALSE(error.IsOK()) << seed;
  }
}

} // namespace
