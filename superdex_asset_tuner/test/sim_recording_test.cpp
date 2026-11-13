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

#include "superdex_asset_tuner/sim_recording.h"
#include "tuner_test_fixture.h"

// Builds without superdex_recorder record no sim.rrd and cannot read one.
#if __has_include(<superdex_recorder/episode_reader.h>)
#include <superdex_recorder/episode_reader.h>
#define SUPERDEX_TUNER_TEST_RECORDER 1
#else
#define SUPERDEX_TUNER_TEST_RECORDER 0
#endif

#include <memory>
#include <string>
#include <vector>

namespace superdex::tuner::test {
namespace {

class SimRecordingTest : public TunerTest {
 protected:
  void SetUp() override {
    TunerTest::SetUp();
    if (!SimRecording::IsSupported()) {
      GTEST_SKIP() << "Built without superdex_recorder.";
    }
  }

  // A pushed cube, recorded to a run folder.
  [[nodiscard]] RunCapture RecordPushedCube() const {
    TunerSpec spec = MakeSpec(kRigidCubePrefab);
    spec.loads.push_back(MakeLoad("push", LoadType::Force, "cube", {1.0f, 0.0f, 0.0f}, 1));
    spec.probes.push_back(MakeProbe(ProbeKind::BodyPose, "cube"));
    spec.run.duration = 0.1;
    return Run(spec, {.outDir = Dir() / "runs"});
  }

  [[nodiscard]] static std::filesystem::path RecordingPath(RunCapture const& run) {
    return std::filesystem::path(std::string_view(run.summary.runDir)) / "sim.rrd";
  }
};

TEST_F(SimRecordingTest, DisabledRecordingWritesNoFile) {
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  spec.run.duration = 0.05;
  spec.run.record.enabled = false;

  RunCapture const run = Run(spec, {.outDir = Dir() / "runs"});

  EXPECT_FALSE(run.summary.simRecording);
  EXPECT_FALSE(std::filesystem::exists(RecordingPath(run)));
}

#if SUPERDEX_TUNER_TEST_RECORDER
std::unique_ptr<recorder::EpisodeReader> OpenRecording(std::filesystem::path const& path) {
  mochi::Error error;
  auto reader = recorder::EpisodeReader::Open(path, error);
  EXPECT_TRUE(error.IsOK()) << error.ToString();
  return reader;
}

TEST_F(SimRecordingTest, RecordsEveryStepsChannelsAndLoads) {
  RunCapture const run = RecordPushedCube();
  auto const reader = OpenRecording(RecordingPath(run));
  ASSERT_NE(reader, nullptr);
  mochi::Error error;

  std::vector<std::string> const channelNames =
      reader->ReadStaticTexts(kRecordingTunerPath, kRecordingChannelNames, error);
  std::vector<std::vector<double>> channels;
  channels.reserve(run.channels.size());
  for (std::string const& name : run.channels) {
    channels.push_back(
        reader
            ->ReadScalars(
                std::string(kRecordingChannelsPath) + "/" + name, "Scalars:scalars", error)
            .values);
  }
  recorder::ScalarRows const loadVectors =
      reader->ReadScalars(std::string(kRecordingLoadsPath) + "/push", kRecordingLoadVector, error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_TRUE(run.summary.simRecording) << run.summary.message.c_str();
  std::vector<std::vector<double>> expectedChannels(run.channels.size());
  std::vector<double> expectedLoadVectors;
  for (Sample const& sample : run.samples) {
    for (size_t c = 0; c < run.channels.size(); ++c) {
      expectedChannels[c].push_back(sample.values[c]);
    }
    ASSERT_EQ(sample.loads.size(), 1u);
    Real3 const& vector = sample.loads[0].vectorWorld;
    expectedLoadVectors.insert(expectedLoadVectors.end(), {vector[0], vector[1], vector[2]});
  }
  EXPECT_EQ(channelNames, run.channels);
  EXPECT_EQ(channels, expectedChannels);
  EXPECT_EQ(loadVectors.values, expectedLoadVectors);
  ASSERT_EQ(loadVectors.simulationTimesNs.size(), run.samples.size());
  for (size_t i = 0; i < run.samples.size(); ++i) {
    // Rerun stores times in whole nanoseconds.
    EXPECT_NEAR(
        static_cast<double>(loadVectors.simulationTimesNs[i]) * 1e-9, run.samples[i].time, 1e-9);
  }
}

TEST_F(SimRecordingTest, KeepsTheResolvedSpecAndSummary) {
  RunCapture const run = RecordPushedCube();
  auto const reader = OpenRecording(RecordingPath(run));
  ASSERT_NE(reader, nullptr);
  mochi::Error error;

  std::vector<std::string> const specJson =
      reader->ReadStaticTexts(kRecordingTunerPath, kRecordingSpecJson, error);
  std::vector<std::string> const summaryJson =
      reader->ReadStaticTexts(kRecordingTunerPath, kRecordingSummaryJson, error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_EQ(specJson.size(), 1u);
  ASSERT_EQ(summaryJson.size(), 1u);
  TunerSpec spec;
  std::string specError;
  ASSERT_TRUE(LoadTunerSpecFromJson(specJson[0], {}, spec, specError)) << specError;
  ASSERT_EQ(spec.loads.size(), 1u);
  EXPECT_EQ(spec.loads[0].name, "push");
  RunSummary summary;
  ASSERT_TRUE(SReflect::FromJsonString(summary, summaryJson[0]));
  EXPECT_EQ(summary.steps, run.summary.steps);
  EXPECT_EQ(summary.status, run.summary.status);
}

TEST_F(SimRecordingTest, RecordsContactForcesOnlyWhenAsked) {
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  spec.run.gravity = Real3{0.0f, 0.0f, -9.81f};
  spec.run.groundPlane = true;
  spec.run.duration = 0.1;
  auto const recordsContacts = [&](bool contactForces) {
    spec.run.record.contactForces = contactForces;
    std::filesystem::path const out = Dir() / (contactForces ? "with" : "without");
    RunCapture const run = Run(spec, {.outDir = out});
    auto const reader = OpenRecording(RecordingPath(run));
    return reader != nullptr &&
        reader->HasComponent(
            "data/physics/contacts", "superdex.physics.contact_force_on_a_world_xyz", false);
  };

  EXPECT_TRUE(recordsContacts(true));
  EXPECT_FALSE(recordsContacts(false));
}
#endif

} // namespace
} // namespace superdex::tuner::test
