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
#include <superdex_robotics/core/context.h>
#include <superdex_robotics/domain_randomization/derive.h>
#include <superdex_robotics/domain_randomization/random.h>
#include <superdex_robotics/sensors/sensor_delay_line.h>
#include <superdex_robotics/sensors/sensor_graph.h>
#include <superdex_robotics/sensors/sensor_temporal_window.h>
#include <superdex_robotics/superdex_robotics.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

using namespace mochi;
using namespace superdex::robotics;

namespace {

class CadencedScalarSource final : public SensorBase {
 public:
  CadencedScalarSource(int64_t periodSteps, Error& error, double frequencyHzEstimate = 0.0)
      : SensorBase(nullptr, error),
        _periodSteps(periodSteps),
        _frequencyHzEstimate(frequencyHzEstimate) {}

  [[nodiscard]] double GetPublicationFrequencyHzEstimate() const override {
    return _frequencyHzEstimate;
  }

  [[nodiscard]] std::string_view GetTypeName() const override {
    return "TEST_CADENCED_SCALAR_SOURCE";
  }

  [[nodiscard]] SensorStepResult Step(StepTime const& now, Error& error) override {
    MOCHI_ERROR_IF_NOT(Inputs().empty(), error, "CadencedScalarSource accepts no inputs");
    MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
    if (now.step <= 0 || now.step % _periodSteps != 0) {
      return SensorStepResult::NoUpdate;
    }
    std::array<float, 1> const value{static_cast<float>(now.step)};
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
        .shape = {1},
        .axes = {{.name = "value", .labels = {}, .units = {}}},
    };
  }

  int64_t _periodSteps;
  double _frequencyHzEstimate;
};

void AddDelayGraph(SensorGraph& graph, CadencedScalarSource& source, SensorDelayLine& delay) {
  graph.Add("source", &source, mochi::test::ExpectOK{});
  DynamicArray<DynamicString> const input = {"source"};
  graph.Add("delay", &delay, input, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("delay", true, mochi::test::ExpectOK{});
}

void Step(SensorGraph& graph, int64_t step, Error& error) {
  double const t = 0.005 * static_cast<double>(step);
  EXPECT_TRUE(graph.Step({.t = t, .dt = 0.005, .step = step}, error));
  ASSERT_TRUE(error.IsOK()) << error.ToString();
}

TEST(SensorDelayLineTest, RejectsNegativeDelay) {
  Error error;
  SensorDelayLine delay({.delayMs = -1_r}, error);
  EXPECT_FALSE(error.IsOK());
}

TEST(SensorDelayLineTest, RejectsInvalidBufferEstimates) {
  Error negativeDuration;
  SensorDelayLine negative({.bufferMs = -1.0}, negativeDuration);
  EXPECT_FALSE(negativeDuration.IsOK());
  Error zeroFrequency;
  SensorDelayLine zero({.bufferAssumedFrequencyHzEstimate = 0.0}, zeroFrequency);
  EXPECT_FALSE(zeroFrequency.IsOK());
  Error overflow;
  CadencedScalarSource source(1, overflow);
  SensorDelayLine huge({.bufferMs = 1e300}, overflow);
  SensorGraph graph;
  AddDelayGraph(graph, source, huge);
  graph.Build({}, overflow);
  EXPECT_FALSE(overflow.IsOK());
}

TEST(SensorDelayLineTest, ZeroDelayPreservesValuesAndIdentity) {
  Error constructionError;
  CadencedScalarSource source(1, constructionError);
  SensorDelayLine delay({.delayMs = 0_r}, constructionError);
  ASSERT_TRUE(constructionError.IsOK()) << constructionError.ToString();
  SensorGraph graph;
  AddDelayGraph(graph, source, delay);
  Error error;
  graph.Build({}, error);
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  Step(graph, 7, error);
  SensorSample const input = source.Latest();
  SensorSample const sample = delay.Latest();
  ASSERT_EQ(sample.Floats().size(), 1);
  EXPECT_EQ(sample.Floats()[0], 7.0F);
  EXPECT_EQ(sample.captureStep, 7);
  EXPECT_EQ(sample.captureTime, input.captureTime);
  EXPECT_EQ(sample.publishedStep, 7);
  EXPECT_EQ(sample.publishedTime, input.publishedTime);
}

// The first sample after a reset publishes at once and holds until the second is due, 10 ms after
// its publication.
TEST(SensorDelayLineTest, FirstSamplePublishesAtOnceAndLaterOnesAfterTheDelay) {
  Error constructionError;
  CadencedScalarSource source(1, constructionError);
  SensorDelayLine delay({.delayMs = 10_r}, constructionError);
  ASSERT_TRUE(constructionError.IsOK()) << constructionError.ToString();
  SensorGraph graph;
  AddDelayGraph(graph, source, delay);
  Error error;
  graph.Build({}, error);
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  Step(graph, 1, error);
  SensorSample const first = delay.Latest();
  ASSERT_EQ(first.Floats().size(), 1);
  EXPECT_FLOAT_EQ(first.Floats()[0], 1.0F);
  EXPECT_EQ(first.captureStep, 1);
  EXPECT_DOUBLE_EQ(first.captureTime, 0.005);
  EXPECT_EQ(first.publishedStep, 1);
  EXPECT_DOUBLE_EQ(first.publishedTime, 0.005);
  Step(graph, 2, error);
  Step(graph, 3, error);
  EXPECT_EQ(delay.Latest().captureStep, 1);
  EXPECT_EQ(delay.Latest().publishedStep, 1);

  Step(graph, 4, error);
  SensorSample const second = delay.Latest();
  EXPECT_FLOAT_EQ(second.Floats()[0], 2.0F);
  EXPECT_EQ(second.captureStep, 2);
  EXPECT_EQ(second.publishedStep, 4);
  EXPECT_DOUBLE_EQ(second.publishedTime, 0.02);
}

TEST(SensorDelayLineTest, ResetDiscardsSamplesInFlight) {
  Error constructionError;
  CadencedScalarSource source(1, constructionError);
  SensorDelayLine delay({.delayMs = 10_r}, constructionError);
  ASSERT_TRUE(constructionError.IsOK()) << constructionError.ToString();
  SensorGraph graph;
  AddDelayGraph(graph, source, delay);
  Error error;
  graph.Build({}, error);
  graph.Reset(error);
  Step(graph, 1, error);

  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(delay.Latest().captureStep, -1);
  Step(graph, 2, error);
  Step(graph, 3, error);
  Step(graph, 4, error);
  EXPECT_EQ(delay.Latest().captureStep, 2);
  EXPECT_EQ(delay.Latest().publishedStep, 2);
  Step(graph, 5, error);
  EXPECT_EQ(delay.Latest().captureStep, 3);
  EXPECT_EQ(delay.Latest().publishedStep, 5);
}

TEST(SensorDelayLineTest, DelaysOneHundredHertzSamplesBeforeTemporalWindow) {
  Error constructionError;
  CadencedScalarSource source(2, constructionError);
  SensorDelayLine delay({.delayMs = 20_r}, constructionError);
  SensorTemporalWindow history({.historyLength = 5}, constructionError);
  ASSERT_TRUE(constructionError.IsOK()) << constructionError.ToString();
  SensorGraph graph;
  graph.Add("source", &source, mochi::test::ExpectOK{});
  DynamicArray<DynamicString> const sourceInput = {"source"};
  DynamicArray<DynamicString> const delayInput = {"delay"};
  graph.Add("delay", &delay, sourceInput, mochi::test::ExpectOK{});
  graph.Add("history", &history, delayInput, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("history", true, mochi::test::ExpectOK{});
  Error error;
  graph.Build({}, error);
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  for (int64_t step = 1; step <= 20; ++step) {
    Step(graph, step, error);
  }

  EXPECT_EQ(
      DynamicArray<float>(history.Latest().Floats().begin(), history.Latest().Floats().end()),
      (DynamicArray<float>{8.0F, 10.0F, 12.0F, 14.0F, 16.0F}));
  EXPECT_EQ(
      DynamicArray<int64_t>(
          history.GetRowCaptureSteps().begin(), history.GetRowCaptureSteps().end()),
      (DynamicArray<int64_t>{8, 10, 12, 14, 16}));
  EXPECT_EQ(
      DynamicArray<int64_t>(
          history.GetRowPublishedSteps().begin(), history.GetRowPublishedSteps().end()),
      (DynamicArray<int64_t>{12, 14, 16, 18, 20}));
  EXPECT_EQ(history.Latest().captureStep, 16);
  EXPECT_EQ(history.Latest().publishedStep, 20);
}

TEST(SensorDelayLineTest, NonIntegralDelayPublishesAtTheFirstGraphBoundaryAfterDueTime) {
  Error constructionError;
  CadencedScalarSource source(1, constructionError);
  SensorDelayLine delay({.delayMs = 7.5_r}, constructionError);
  ASSERT_TRUE(constructionError.IsOK()) << constructionError.ToString();
  SensorGraph graph;
  AddDelayGraph(graph, source, delay);
  Error error;
  graph.Build({}, error);
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  Step(graph, 1, error);
  Step(graph, 2, error);
  Step(graph, 3, error);
  EXPECT_EQ(delay.Latest().captureStep, 1);
  Step(graph, 4, error);
  EXPECT_EQ(delay.Latest().captureStep, 2);
  EXPECT_DOUBLE_EQ(delay.Latest().captureTime, 0.01);
  EXPECT_DOUBLE_EQ(delay.Latest().publishedTime, 0.02);
}

TEST(SensorDelayLineTest, PendingSampleBoundFailsLoudlyWithoutDroppingAQueuedSample) {
  Error constructionError;
  CadencedScalarSource source(1, constructionError);
  SensorDelayLine delay(
      {.delayMs = 100_r, .bufferAssumedFrequencyHzEstimate = 1.0}, constructionError);
  ASSERT_TRUE(constructionError.IsOK()) << constructionError.ToString();
  SensorGraph graph;
  AddDelayGraph(graph, source, delay);
  Error error;
  graph.Build({}, error);
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  // The first sample publishes at once; the next two fill the ring's two slots.
  Step(graph, 1, error);
  Step(graph, 2, error);
  Step(graph, 3, error);
  EXPECT_EQ(delay.Latest().captureStep, 1);

  Error overflowError;
  EXPECT_FALSE(graph.Step({.t = 0.02, .dt = 0.005, .step = 4}, overflowError));
  EXPECT_FALSE(overflowError.IsOK());
  EXPECT_EQ(delay.Latest().captureStep, 1);
}

TEST(SensorDelayLineTest, FallbackEstimateSupportsKilohertzInputWithOneSecondDelay) {
  Error error;
  CadencedScalarSource source(1, error);
  SensorDelayLine delay(
      {.delayMs = 1000.0, .bufferMs = 1000.0, .bufferAssumedFrequencyHzEstimate = 1024.0}, error);
  SensorGraph graph;
  AddDelayGraph(graph, source, delay);
  graph.Build({}, error);
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  // Binary-exact intervals isolate capacity and wraparound from timestamp rounding.
  for (int64_t step = 1; step <= 3073; ++step) {
    double const t = static_cast<double>(step) / 1024.0;
    ASSERT_TRUE(graph.Step({.t = t, .dt = 1.0 / 1024.0, .step = step}, error)) << error.ToString();
    SensorSample const sample = delay.Latest();
    if (step <= 1025) {
      EXPECT_EQ(sample.captureStep, 1);
      EXPECT_EQ(sample.publishedStep, 1);
    } else {
      ASSERT_EQ(sample.Floats().size(), 1);
      EXPECT_FLOAT_EQ(sample.Floats()[0], static_cast<float>(step - 1024));
      EXPECT_EQ(sample.captureStep, step - 1024);
      EXPECT_EQ(sample.publishedStep, step);
      EXPECT_DOUBLE_EQ(sample.publishedTime - sample.captureTime, 1.0);
    }
  }
}

TEST(SensorDelayLineTest, UpstreamEstimatePassesThroughTransformsAndOverridesFallback) {
  Error error;
  CadencedScalarSource source(1, error, 1024.0);
  SensorTemporalWindow history({.historyLength = 2}, error);
  SensorDelayLine delay({.delayMs = 1000.0, .bufferAssumedFrequencyHzEstimate = 1.0}, error);
  SensorGraph graph;
  graph.Add("source", &source, error);
  graph.Add("history", &history, DynamicArray<DynamicString>{"source"}, error);
  graph.Add("delay", &delay, DynamicArray<DynamicString>{"history"}, error);
  graph.SetOutputEnabled("delay", true, error);
  graph.Build({}, error);
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  for (int64_t step = 1; step <= 2049; ++step) {
    ASSERT_TRUE(graph.Step(
        {.t = static_cast<double>(step) / 1024.0, .dt = 1.0 / 1024.0, .step = step}, error))
        << error.ToString();
  }
  EXPECT_EQ(delay.Latest().captureStep, 1025);
  EXPECT_EQ(delay.Latest().publishedStep, 2049);
  EXPECT_DOUBLE_EQ(history.GetPublicationFrequencyHzEstimate(), 1024.0);
}

TEST(SensorDelayLineTest, PublishesADueSampleBeforeReusingItsBoundedRingSlot) {
  Error constructionError;
  CadencedScalarSource source(1, constructionError);
  // Two slots: ceil(10 ms * 100 Hz) + 1.
  SensorDelayLine delay({.delayMs = 10_r, .bufferMs = 10.0}, constructionError);
  ASSERT_TRUE(constructionError.IsOK()) << constructionError.ToString();
  SensorGraph graph;
  AddDelayGraph(graph, source, delay);
  Error error;
  graph.Build({}, error);
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  // Sample 1 publishes at once, samples 2 and 3 fill the ring, and sample 2, due at step 4, frees
  // its slot for sample 4.
  Step(graph, 1, error);
  Step(graph, 2, error);
  Step(graph, 3, error);
  Step(graph, 4, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(delay.Latest().captureStep, 2);
  EXPECT_EQ(delay.Latest().publishedStep, 4);
}

TEST(SensorDelayLineTest, DrainsEveryDueSampleAfterAGraphTimeJump) {
  Error constructionError;
  CadencedScalarSource source(1, constructionError);
  SensorDelayLine delay(
      {.delayMs = 100_r, .bufferAssumedFrequencyHzEstimate = 3.0}, constructionError);
  ASSERT_TRUE(constructionError.IsOK()) << constructionError.ToString();
  SensorGraph graph;
  AddDelayGraph(graph, source, delay);
  Error error;
  graph.Build({}, error);
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  Step(graph, 1, error);
  Step(graph, 2, error);
  Step(graph, 3, error);
  EXPECT_EQ(delay.Latest().publishedStep, 1);

  Step(graph, 100, error);
  EXPECT_EQ(delay.Latest().captureStep, 3);
  EXPECT_EQ(delay.Latest().publishedStep, 100);
}

// A delay of whole steps stays whole all episode on a clock that sums its steps, as BotScene's
// does, even written in rounded milliseconds.
TEST(SensorDelayLineTest, WholeStepDelaysStayWholeOnAnAccumulatedClock) {
  struct Case {
    double hz = 0.0;
    double delayMs = 0.0;
    int64_t steps = 0;
  };
  std::array<Case, 3> constexpr kCases = {{
      {.hz = 200.0, .delayMs = 50.0, .steps = 10},
      {.hz = 240.0, .delayMs = 16.666667, .steps = 4},
      {.hz = 500.0, .delayMs = 10.0, .steps = 5},
  }};
  for (Case const& c : kCases) {
    Error error;
    CadencedScalarSource source(1, error);
    SensorDelayLine delay({.delayMs = c.delayMs}, error);
    SensorGraph graph;
    AddDelayGraph(graph, source, delay);
    graph.Build({}, error);
    graph.Reset(error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();

    double t = 0.0;
    int64_t late = 0;
    for (int64_t step = 1; step <= 300; ++step) {
      t += 1.0 / c.hz;
      ASSERT_TRUE(graph.Step({.t = t, .dt = 1.0 / c.hz, .step = step}, error)) << error.ToString();
      late += delay.Latest().captureStep != std::max<int64_t>(step - c.steps, 1) ? 1 : 0;
    }
    EXPECT_EQ(late, 0) << c.hz << " Hz";
  }
}

TEST(SensorDelayLineTest, TypedParamsRejectInvalidValuesWithoutMutation) {
  Error error;
  SensorDelayLine delay({.delayMs = 8.0}, error);
  delay.SetParams({.delayMs = 12.0}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_DOUBLE_EQ(delay.GetParams().delayMs, 12.0);
  Error rejected;
  delay.SetParams({.delayMs = -1.0}, rejected);
  EXPECT_FALSE(rejected.IsOK());
  EXPECT_DOUBLE_EQ(delay.GetParams().delayMs, 12.0);
}

// The delay and the jitter must fit in the buffer, at construction and in SetParams.
TEST(SensorDelayLineTest, DelayAndJitterMustFitInBufferMs) {
  SensorDelayLine tooLong(
      {.delayMs = 30.0, .bufferMs = 40.0, .jitterMs = 11.0}, test::ExpectNotOK{});
  SensorDelayLine delay({.delayMs = 30.0, .bufferMs = 40.0, .jitterMs = 10.0}, test::ExpectOK{});

  delay.SetParams({.delayMs = 35.0, .bufferMs = 40.0, .jitterMs = 6.0}, test::ExpectNotOK{});

  EXPECT_DOUBLE_EQ(delay.GetParams().delayMs, 30.0);
  EXPECT_DOUBLE_EQ(delay.GetParams().jitterMs, 10.0);
}

// The ring holds ceil(bufferMs * rate estimate / 1000) + 1 samples, the input's estimate or else
// the fallback: for 20 ms at 200 Hz, five. The input publishes at 1 kHz, so after the first sample,
// which publishes at once, five samples fill the ring and a sixth overflows.
TEST(SensorDelayLineTest, CapacityFollowsBufferMsAndTheRateEstimate) {
  for (bool const fallback : {false, true}) {
    Error error;
    CadencedScalarSource source(1, error, fallback ? 0.0 : 200.0);
    SensorDelayLine delay(
        {.delayMs = 20.0,
         .bufferMs = 20.0,
         .bufferAssumedFrequencyHzEstimate = fallback ? 200.0 : 1.0},
        error);
    SensorGraph graph;
    AddDelayGraph(graph, source, delay);
    graph.Build({}, error);
    graph.Reset(error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();

    for (int64_t step = 1; step <= 6; ++step) {
      double const t = static_cast<double>(step) / 1000.0;
      ASSERT_TRUE(graph.Step({.t = t, .dt = 0.001, .step = step}, error))
          << error.ToString() << " at step " << step;
    }
    Error overflow;
    EXPECT_FALSE(graph.Step({.t = 0.007, .dt = 0.001, .step = 7}, overflow));
    EXPECT_NE(overflow.ToString().find("exceeded its configured capacity"), std::string::npos)
        << overflow.ToString();
  }
}

TEST(SensorDelayLineTest, EpisodeParamsTakeEffectAfterExplicitReset) {
  Error error;
  CadencedScalarSource source(1, error);
  SensorDelayLine delay({.delayMs = 10.0}, error);
  SensorGraph graph;
  AddDelayGraph(graph, source, delay);
  graph.Build({}, error);
  graph.Reset(error);
  Step(graph, 1, error);
  Step(graph, 2, error);
  ASSERT_EQ(delay.Latest().captureStep, 1);
  delay.SetParams({.delayMs = 0.0}, error);
  graph.Reset(error);
  Step(graph, 3, error);
  Step(graph, 4, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(delay.Latest().captureStep, 4);
  EXPECT_FLOAT_EQ(delay.Latest().Floats()[0], 4.0F);
  Error rejected;
  delay.SetParams({.delayMs = 20.0, .bufferMs = 0.0}, rejected);
  EXPECT_FALSE(rejected.IsOK());
  EXPECT_DOUBLE_EQ(delay.GetParams().delayMs, 0.0);
}

TEST(SensorDelayLineTest, JitterNeedsNoProcessKey) {
  SensorDelayLine unkeyed({.jitterMs = 1.0}, test::ExpectOK{});
}

// Each sample waits delayMs plus its own draw from (episode seed, process key, capture step) but
// never overtakes an earlier sample.
TEST(SensorDelayLineTest, JitterDelaysEachSampleByItsDrawInOrder) {
  Error error;
  CadencedScalarSource source(1, error);
  SensorDelayLine delay(
      {.delayMs = 4.0, .jitterMs = 8.0, .random = {.episodeSeed = 3, .processKey = "camera"}},
      error);
  SensorGraph graph;
  AddDelayGraph(graph, source, delay);
  graph.Build({}, error);
  graph.Reset(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  // Sample k is captured and published at step k, t = k / 1024 s. Sample 1 publishes at once; a
  // later sample publishes once its due time and those of the samples before it have passed.
  int64_t constexpr kSteps = 200;
  std::array<double, kSteps + 1> releases{};
  bool held = false;
  for (int64_t k = 2; k <= kSteps; ++k) {
    uint64_t state = Derive(3, "camera", static_cast<uint64_t>(k));
    double const due = static_cast<double>(k) / 1024.0 + (4.0 + Uniform(state, 0.0, 8.0)) / 1000.0;
    releases[k] = std::max(due, releases[k - 1]);
    held = held || due < releases[k - 1];
  }
  EXPECT_TRUE(held) << "no sample waited behind an earlier one";

  int64_t expected = 1;
  for (int64_t step = 1; step <= kSteps; ++step) {
    double const t = static_cast<double>(step) / 1024.0;
    ASSERT_TRUE(graph.Step({.t = t, .dt = 1.0 / 1024.0, .step = step}, error)) << error.ToString();
    while (expected < step && releases[expected + 1] <= t) {
      ++expected;
    }
    EXPECT_EQ(delay.Latest().captureStep, expected) << "step " << step;
  }
}

// The graph's seed overrides the params' and survives SetParams: the two calls' order jitters the
// samples alike, otherwise than seed 0.
TEST(SensorDelayLineTest, TheGraphsSeedSurvivesSetParams) {
  SensorDelayLine::Params const params{.delayMs = 4.0, .jitterMs = 8.0};
  // The capture step published at each of 200 steps of 1024 Hz.
  auto const published = [&params](uint64_t seed, bool seedFirst) {
    CadencedScalarSource source(1, test::ExpectOK{});
    SensorDelayLine delay(params, test::ExpectOK{});
    SensorGraph graph;
    AddDelayGraph(graph, source, delay);
    graph.Build({}, test::ExpectOK{});
    if (seedFirst) {
      graph.SetEpisodeSeed(seed);
    }
    delay.SetParams(params, test::ExpectOK{});
    if (!seedFirst) {
      graph.SetEpisodeSeed(seed);
    }
    graph.Reset(test::ExpectOK{});
    std::vector<int64_t> captures;
    for (int64_t step = 1; step <= 200; ++step) {
      double const t = static_cast<double>(step) / 1024.0;
      graph.Step({.t = t, .dt = 1.0 / 1024.0, .step = step}, test::ExpectOK{});
      captures.push_back(delay.Latest().captureStep);
    }
    return captures;
  };

  EXPECT_EQ(published(7, true), published(7, false));
  EXPECT_NE(published(7, true), published(0, true));
}

TEST(SensorDelayLineTest, SceneAndBotFilesCanDeclareIt) {
  RoboticsContext* const context = CreateRoboticsContext();
  EXPECT_TRUE(context->IsSensorTypeRegistered(SensorDelayLine::TypeName()));
  DestroyRoboticsContext(context);
}

} // namespace
