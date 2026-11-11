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

#include <gtest/gtest.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <superdex_recorder/episode_reader.h>
#include <superdex_recorder/link_kinematics.h>
#include <superdex_recorder/recorder.h>
#include <superdex_recorder/series.h>
#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/utils/bot_utils.h>
#include "simulation/ik_recording_player.h"

#include <rerun.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>

using namespace mochi;
using namespace superdex::robotics;
using namespace superdex::recorder;
using namespace superdex::studio;
namespace {
constexpr char kEntity[] = "data/physics/actors/fixture_hand";
constexpr char kArchetype[] = "superdex.archetypes.PhysicsActor";
constexpr char kPositions[] = "superdex.physics.joint_positions";
constexpr char kTranslation[] = "superdex.physics.root_translation_xyz";
constexpr char kRotation[] = "superdex.physics.root_quaternion_xyzw";
constexpr char kNames[] = "superdex.physics.joint_names";
constexpr char kTypes[] = "superdex.physics.joint_types";
constexpr char kOffsets[] = "superdex.physics.joint_dof_offsets";
constexpr char kTranslations[] = "superdex.physics.joint_translation_dof_counts";
constexpr char kRotations[] = "superdex.physics.joint_rotation_dof_counts";
using Key = std::pair<std::string, std::string>;
BotPrefab Robot(size_t dofs = 2, bool fixedTip = false, bool freeRoot = true) {
  BotPrefab inputBot;
  inputBot.name = "fixture_hand";
  for (size_t i = 0; i < dofs + 1 + static_cast<size_t>(fixedTip); ++i) {
    BotLinkPrefab link;
    link.name = i == 0 ? "root" : "joint" + std::to_string(i);
    link.parentLink = static_cast<int>(i) - 1;
    link.mass = 1_r;
    link.centerOfMass = Real3{};
    link.momentOfInertia = Real6{1_r, 0_r, 0_r, 1_r, 0_r, 1_r};
    inputBot.links.push_back(link);
    BotJointPrefab joint;
    joint.name = link.name;
    joint.type = i == 0 ? (freeRoot ? ArticulatedJointType::Free : ArticulatedJointType::Hard)
        : i > dofs      ? ArticulatedJointType::Hard
                        : ArticulatedJointType::Revolute;
    joint.axis = Real3{0_r, 0_r, 1_r};
    inputBot.joints.push_back(joint);
  }
  RebuildBotData(inputBot, test::ExpectOK{});
  return inputBot;
}
struct Metadata {
  std::map<Key, std::vector<std::string>> texts;
  std::map<Key, std::vector<double>> scalars;
  void Merge(Metadata const& other) {
    texts.insert(other.texts.begin(), other.texts.end());
    scalars.insert(other.scalars.begin(), other.scalars.end());
  }
};
Metadata Meta(
    BotPrefab const& inputBot = Robot(),
    std::string const& path = kEntity,
    std::string runtime = "",
    bool roboticsIdentity = true) {
  Metadata metadata;
  if (runtime.empty())
    runtime = std::string{inputBot.name};
  auto const id = path.substr(std::string{"data/physics/actors/"}.size());
  metadata.texts[{path, "superdex.physics.actor_id"}] = {id};
  metadata.texts[{path, "superdex.physics.runtime_name"}] = {runtime};
  metadata.texts[{path, "superdex.physics.actor_type"}] = {"articulated"};
  metadata.scalars[{path, "superdex.physics.is_static"}] = {0};
  std::string const layout = path + "/joint_layout";
  size_t cursor = 0;
  for (auto const& joint : inputBot.joints) {
    size_t trans = 0, rot = 0;
    std::string type;
    switch (joint.type) {
      case ArticulatedJointType::Free:
        type = "free";
        trans = 3;
        rot = 3;
        break;
      case ArticulatedJointType::Spherical:
        type = "spherical";
        rot = 3;
        break;
      case ArticulatedJointType::Prismatic:
        type = "prismatic";
        trans = 1;
        break;
      case ArticulatedJointType::Revolute:
        type = "revolute";
        rot = 1;
        break;
      default:
        type = "hard";
        break;
    }
    metadata.texts[{layout, kNames}].emplace_back(joint.name);
    metadata.texts[{layout, kTypes}].push_back(type);
    metadata.scalars[{layout, kOffsets}].push_back(static_cast<double>(cursor));
    metadata.scalars[{layout, kTranslations}].push_back(static_cast<double>(trans));
    metadata.scalars[{layout, kRotations}].push_back(static_cast<double>(rot));
    cursor += trans + rot;
  }
  metadata.scalars[{path, "superdex.physics.num_dofs"}] = {static_cast<double>(cursor)};
  if (roboticsIdentity) {
    auto const robotPath = "data/robotics/robots/" + id;
    metadata.texts[{robotPath, "superdex.robotics.robot_id"}] = {runtime};
    metadata.texts[{robotPath, "superdex.robotics.robot_prefab_name"}] = {
        std::string{inputBot.name}};
    metadata.scalars[{robotPath, "superdex.robotics.num_dofs"}] = {static_cast<double>(cursor)};
  }
  return metadata;
}
struct Snapshot {
  int64_t time;
  int64_t frame;
  std::vector<double> positions;
  std::vector<double> translation;
  std::vector<double> rotation;
};
Snapshot Sample(int64_t time, int64_t frame, double q = 0, size_t dofs = 2, bool freeRoot = true) {
  size_t const root = freeRoot ? 6 : 0;
  Snapshot row{time, frame, std::vector<double>(dofs + root, 0), {q / 10, 2, 3}, {0, 0, 0, 1}};
  row.positions[root] = q;
  return row;
}
std::vector<Snapshot> Samples() {
  return {Sample(10, 10, 10), Sample(20, 20, 20), Sample(30, 30, 30)};
}
struct Row {
  int64_t time;
  int64_t frame;
  std::vector<double> values;
};
using Columns = std::map<Key, std::vector<Row>>;
Columns State(std::vector<Snapshot> const& samples = Samples(), std::string const& path = kEntity) {
  Columns columns;
  for (auto const& sample : samples) {
    columns[{path, kPositions}].push_back({sample.time, sample.frame, sample.positions});
    columns[{path, kTranslation}].push_back({sample.time, sample.frame, sample.translation});
    columns[{path, kRotation}].push_back({sample.time, sample.frame, sample.rotation});
  }
  return columns;
}
template <typename Component>
rerun::ComponentBatch Batch(std::vector<Component> const& values, std::string const& component) {
  return rerun::ComponentBatch::from_loggable(
             rerun::Collection<Component>::borrow(values),
             rerun::ComponentDescriptor{
                 kArchetype, component, rerun::Loggable<Component>::ComponentType})
      .value;
}
struct RawOptions {
  bool frameTimeline = true;
  bool simTimeline = true;
  bool staticState = false;
  std::string schema = "1.1";
  std::string space = "FLU";
  double units = 1;
};
class Fixture {
 public:
  Fixture() {
    static std::atomic_uint64_t serial = 0;
    directory = std::filesystem::temp_directory_path() /
        ("superdex_studio_rrd_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
         std::to_string(serial++));
    std::filesystem::create_directories(directory);
  }
  ~Fixture() {
    std::error_code error;
    std::filesystem::remove_all(directory, error);
  }
  void Write(std::vector<Snapshot> const& samples = Samples(), Metadata const& metadata = Meta()) {
    Recorder recorder{{.outputDirectory = directory}};
    Error error;
    auto episode = recorder.BeginEpisode({}, error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    ASSERT_NE(episode, nullptr);
    for (auto const& [key, values] : metadata.texts) {
      episode->WriteConstantTexts(key.first, {kArchetype, key.second}, values, error);
    }
    for (auto const& [key, values] : metadata.scalars) {
      episode->WriteConstantScalars(key.first, {kArchetype, key.second}, values, error);
    }
    auto* series = episode->CreateScalarSeries(
        kEntity,
        {{kArchetype, kPositions}, {kArchetype, kTranslation}, {kArchetype, kRotation}},
        error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    ASSERT_NE(series, nullptr);
    for (auto const& row : samples) {
      std::vector<real> q{row.positions.begin(), row.positions.end()};
      std::vector<real> xyz{row.translation.begin(), row.translation.end()};
      std::vector<real> xyzw{row.rotation.begin(), row.rotation.end()};
      std::array<Span<real const>, 3> fields{
          MakeConstSpan(q), MakeConstSpan(xyz), MakeConstSpan(xyzw)};
      series->Append(
          {static_cast<uint64_t>(row.frame), static_cast<double>(row.time) / 1e9}, fields, error);
      ASSERT_TRUE(error.IsOK()) << error.ToString();
    }
    episode->Commit(error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    path = episode->GetOutputPath();
  }
  // Native Rerun SDK columns allow deliberate malformed fields/keys that Recorder rejects.
  // Different fields are independently chunked and written in reverse order, not packed rows.
  void WriteRaw(
      Columns const& columns = State(),
      Metadata const& metadata = Meta(),
      RawOptions const& options = {}) {
    path = directory / "raw.rrd";
    rerun::RecordingStream const stream{"superdex_studio_native_physics_test"};
    ASSERT_TRUE(stream.save(path.string()).is_ok());
    stream.log_static(
        "episode",
        Batch(
            std::vector<rerun::components::Text>{"superdex.episode"},
            "superdex.episode.schema_name"),
        Batch(
            std::vector<rerun::components::Text>{options.schema},
            "superdex.episode.schema_version"),
        Batch(
            std::vector<rerun::components::Text>{options.space},
            "superdex.episode.coordinate_space"),
        Batch(
            std::vector<rerun::components::Scalar>{options.units},
            "superdex.episode.units_per_meter"));
    for (auto const& [key, values] : metadata.texts) {
      stream.log_static(
          key.first,
          Batch(std::vector<rerun::components::Text>{values.begin(), values.end()}, key.second));
    }
    for (auto const& [key, values] : metadata.scalars) {
      stream.log_static(
          key.first,
          Batch(std::vector<rerun::components::Scalar>{values.begin(), values.end()}, key.second));
    }
    for (auto column = columns.rbegin(); column != columns.rend(); ++column) {
      auto const& [key, rows] = *column;
      for (auto row = rows.rbegin(); row != rows.rend(); ++row) {
        std::vector<rerun::components::Scalar> values{row->values.begin(), row->values.end()};
        if (options.staticState) {
          stream.log_static(key.first, Batch(values, key.second));
          continue;
        }
        auto data = rerun::ComponentColumn::from_loggable_with_lengths(
            rerun::Collection<rerun::components::Scalar>::borrow(values),
            std::vector<uint32_t>{static_cast<uint32_t>(values.size())},
            rerun::ComponentDescriptor{
                kArchetype, key.second, rerun::Loggable<rerun::components::Scalar>::ComponentType});
        ASSERT_TRUE(data.is_ok());
        std::vector<rerun::TimeColumn> times{rerun::TimeColumn::from_nanos_since_epoch(
            "log_time", std::vector<int64_t>{1'700'000'000'000'000'000LL - row->time})};
        if (options.frameTimeline)
          times.push_back(
              rerun::TimeColumn::from_sequence("frame", std::vector<int64_t>{row->frame}));
        if (options.simTimeline)
          times.push_back(
              rerun::TimeColumn::from_duration_nanoseconds(
                  "sim_time", std::vector<int64_t>{row->time}));
        ASSERT_TRUE(stream
                        .try_send_columns(
                            key.first, times, rerun::Collection<rerun::ComponentColumn>{data.value})
                        .is_ok());
      }
    }
    ASSERT_TRUE(stream.flush_blocking().is_ok());
  }
  std::shared_ptr<IKRecordingClip const> Load(
      BotPrefab const& inputBot = Robot(),
      std::atomic_bool const* cancelled = nullptr) {
    return IKRecordingClip::Load(path.string(), inputBot, diagnostic, cancelled);
  }
  std::shared_ptr<IKRecordingClip const> Good(BotPrefab const& inputBot = Robot()) {
    auto clip = Load(inputBot);
    EXPECT_NE(clip, nullptr) << diagnostic;
    return clip;
  }
  void Reject(std::string_view expected, BotPrefab const& inputBot = Robot()) {
    EXPECT_EQ(Load(inputBot), nullptr);
    EXPECT_FALSE(diagnostic.empty());
    EXPECT_NE(diagnostic.find(expected), std::string::npos) << diagnostic;
  }
  std::filesystem::path directory;
  std::filesystem::path path;
  std::string diagnostic;
};

TEST(IKRecordingClipTest, NativePhysicsColumnsReadBackAndFreeRootIsNotPartOfBotQ) {
  auto inputBot = Robot(26);
  Fixture file;
  file.Write({Sample(0, 0, 1, 26), Sample(20, 1, 2, 26)}, Meta(inputBot));
  Error error;
  auto reader = EpisodeReader::Open(file.path, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(reader, nullptr);
  auto rows = reader->ReadScalars(kEntity, kPositions, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(rows.offsets, (std::vector<size_t>{0, 32, 64}));
  EXPECT_EQ(rows.frames, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(rows.simulationTimesNs, (std::vector<int64_t>{0, 20}));
  auto clip = file.Good(inputBot);
  ASSERT_NE(clip, nullptr);
  EXPECT_EQ(clip->SampleAt(20)->inputBotConfiguration.size(), 26);
  EXPECT_EQ(clip->SampleAt(20)->inputBotConfiguration[0], 2_r);
  EXPECT_TRUE(clip->SampleAt(20)->hasPose);
  EXPECT_TRUE(clip->SampleAt(20)->tracked); // Availability, not raw tracked confidence.
}
TEST(IKRecordingClipTest, BothHandsUseNativePrefabIdentityEvenWithIdenticalJointLayouts) {
  auto left = Robot(), right = Robot();
  left.name = "left_model";
  right.name = "right_model";
  std::string const leftPath = "data/physics/actors/scene.spawn.42";
  std::string const rightPath = "data/physics/actors/scene.spawn.7";
  auto metadata = Meta(left, leftPath, "runtime left alias");
  metadata.Merge(Meta(right, rightPath, "runtime right alias"));
  auto columns = State({Sample(0, 0, 0.1)}, leftPath);
  columns.merge(State({Sample(0, 0, 0.8)}, rightPath));
  Fixture file;
  file.WriteRaw(columns, metadata);
  auto l = file.Good(left), r = file.Good(right);
  ASSERT_NE(l, nullptr);
  ASSERT_NE(r, nullptr);
  EXPECT_FLOAT_EQ(l->SampleAt(0)->inputBotConfiguration[0], 0.1_r);
  EXPECT_FLOAT_EQ(r->SampleAt(0)->inputBotConfiguration[0], 0.8_r);
  EXPECT_NE(l->GetCompatibility().find(leftPath), std::string::npos);
  EXPECT_NE(r->GetCompatibility().find(rightPath), std::string::npos);
  auto unknown = Robot();
  unknown.name = "unknown_model";
  file.Reject("No compatible", unknown);
}
TEST(IKRecordingClipTest, MultipleCompatibleActorsFailRatherThanChoosingFirstOrSuffix) {
  std::string const second = std::string{kEntity} + "_1";
  auto metadata = Meta();
  // The same original runtime name is preserved by PhysicsRecorder's unique ID suffix.
  metadata.Merge(Meta(Robot(), second, "fixture_hand", false));
  auto columns = State();
  columns.merge(State(Samples(), second));
  Fixture file;
  file.WriteRaw(columns, metadata);
  file.Reject("Ambiguous");
  EXPECT_NE(file.diagnostic.find(second), std::string::npos);
}
TEST(IKRecordingClipTest, AuthoritativePrefabMismatchDoesNotFallBackToActorName) {
  auto metadata = Meta();
  metadata.texts[{"data/robotics/robots/fixture_hand", "superdex.robotics.robot_prefab_name"}] = {
      "other_hand"};
  Fixture file;
  file.WriteRaw(State(), metadata);
  file.Reject("No compatible");
}
TEST(IKRecordingClipTest, PurePhysicsIdentityRequiresExactNameNotHandSubstring) {
  Fixture exact;
  exact.WriteRaw(State(), Meta(Robot(), kEntity, "", false));
  ASSERT_NE(exact.Good(), nullptr);
  std::string const path = "data/physics/actors/not_fixture_hand_1";
  Fixture suffix;
  suffix.WriteRaw(State(Samples(), path), Meta(Robot(), path, "not_fixture_hand_1", false));
  suffix.Reject("No compatible");
  auto metadata = Meta(Robot(), path, "different_runtime", false);
  metadata.texts[{path, "superdex.physics.spawn_name"}] = {"fixture_hand"};
  Fixture spawn;
  spawn.WriteRaw(State(Samples(), path), metadata);
  ASSERT_NE(spawn.Good(), nullptr);
}
TEST(IKRecordingClipTest, NativeInt64TimelinesAreExactAndIgnoreWallClockAndChunkOrder) {
  Fixture file;
  constexpr int64_t first = 9'007'199'254'740'993LL;
  constexpr int64_t frame = std::numeric_limits<int64_t>::max();
  file.WriteRaw(State(
      {Sample(first, frame - 2, 10),
       Sample(first + 1, frame - 1, 20),
       Sample(std::numeric_limits<int64_t>::max(), frame, 30)}));
  auto clip = file.Good();
  ASSERT_NE(clip, nullptr);
  EXPECT_EQ(clip->GetFirstTime(), first);
  EXPECT_EQ(clip->GetLastTime(), std::numeric_limits<int64_t>::max());
  EXPECT_EQ(clip->SampleAt(first + 1)->frame, static_cast<uint64_t>(frame - 1));
  EXPECT_EQ(clip->SampleAt(first + 1)->inputBotConfiguration[0], 20_r);
}
TEST(IKRecordingClipTest, EqualSimulationTimesChooseLatestCompleteFrameAndAnchorIt) {
  Fixture file;
  file.WriteRaw(State({Sample(10, 1, 10), Sample(10, 2, 20), Sample(20, 3, 30)}));
  auto clip = file.Good();
  ASSERT_NE(clip, nullptr);
  EXPECT_EQ(clip->SampleAt(9), nullptr);
  EXPECT_EQ(clip->SampleAt(10)->inputBotConfiguration[0], 20_r);
  EXPECT_EQ(clip->SampleAt(19)->inputBotConfiguration[0], 20_r);
  EXPECT_EQ(clip->SampleAt(99)->inputBotConfiguration[0], 30_r);
  EXPECT_EQ(clip->GetTimes(), (std::vector<int64_t>{10, 20}));
  TransformRT anchor{Quaternion::RotationZ(0.3_r), Real3{5_r, 6_r, 7_r}};
  EXPECT_TRUE(NearEqual(clip->InputBotRoot(*clip->SampleAt(10), anchor, false), anchor, 1e-4_r));
  EXPECT_TRUE(NearEqual(
      clip->InputBotRoot(*clip->SampleAt(20), anchor, false),
      anchor * TransformRT{Real3{1_r, 0_r, 0_r}},
      1e-4_r));
}
TEST(IKRecordingClipTest, RootUsesImmutableAuthoredAnchorAndDoesNotAccumulatePlaybackMotion) {
  Fixture file;
  file.Write();
  auto clip = file.Good();
  ASSERT_NE(clip, nullptr);
  TransformRT anchor{Quaternion::RotationZ(0.3_r), Real3{5_r, 6_r, 7_r}};
  EXPECT_TRUE(NearEqual(clip->InputBotRoot(*clip->SampleAt(30), anchor, true), anchor, 1e-4_r));
  EXPECT_TRUE(NearEqual(clip->InputBotRoot(*clip->SampleAt(10), anchor, false), anchor, 1e-4_r));
  EXPECT_TRUE(NearEqual(
      clip->InputBotRoot(*clip->SampleAt(30), anchor, false),
      anchor * TransformRT{Real3{2_r, 0_r, 0_r}},
      1e-4_r));
  EXPECT_TRUE(NearEqual(clip->InputBotRoot(*clip->SampleAt(10), anchor, false), anchor, 1e-4_r));
  EXPECT_DOUBLE_EQ(clip->GetDurationSeconds(), 20e-9);
}
TEST(IKRecordingClipTest, FreeRootTranslationAndRotationVectorCarryMotionWithConstantActorAnchor) {
  auto first = Sample(0, 0, 0.1), last = Sample(10, 1, 0.2);
  first.translation = last.translation = {2, 3, 4};
  last.positions[0] = 0.3;
  last.positions[1] = 0.4;
  last.positions[5] = 0.7;
  Fixture file;
  file.Write({first, last});
  auto clip = file.Good();
  ASSERT_NE(clip, nullptr);
  auto expected = TransformRT{Real3{2_r, 3_r, 4_r}} *
      TransformRT{Quaternion::RotationZ(0.7_r), Real3{0.3_r, 0.4_r, 0_r}};
  EXPECT_TRUE(
      NearEqual(clip->SampleAt(10)->recordedWorldFrameFromInputBaseFrame, expected, 1e-4_r));
  EXPECT_TRUE(NearEqual(
      clip->InputBotRoot(*clip->SampleAt(10), {}, false),
      TransformRT{Quaternion::RotationZ(0.7_r), Real3{0.3_r, 0.4_r, 0_r}},
      1e-4_r));
  EXPECT_TRUE(NearEqual(clip->InputBotRoot(*clip->SampleAt(10), {}, true), TransformRT{}, 1e-4_r));
}
TEST(IKRecordingClipTest, HardRootAddsNoDofsAndUsesRecordedRootPose) {
  auto inputBot = Robot(2, false, false);
  Fixture file;
  file.Write({Sample(0, 0, 0.1, 2, false)}, Meta(inputBot));
  auto clip = file.Good(inputBot);
  ASSERT_NE(clip, nullptr);
  EXPECT_FLOAT_EQ(clip->SampleAt(0)->inputBotConfiguration[0], 0.1_r);
  EXPECT_TRUE(NearEqual(
      clip->SampleAt(0)->recordedWorldFrameFromInputBaseFrame,
      TransformRT{Real3{0.01_r, 2_r, 3_r}},
      1e-4_r));
}
TEST(IKRecordingClipTest, NameMappingHandlesReorderedDofsAndNonPrefixFreeRootSpan) {
  auto metadata = Meta();
  // Root's six DoFs are after the two nonroot coordinates. The layout, not subtraction, maps them.
  metadata.scalars[{std::string{kEntity} + "/joint_layout", kOffsets}] = {2, 1, 0};
  auto row = Sample(0, 0);
  row.positions = {0.8, 0.2, 0.3, 0.4, 0.5, 0, 0, 0.7};
  Fixture file;
  file.WriteRaw(State({row}), metadata);
  auto clip = file.Good();
  ASSERT_NE(clip, nullptr);
  EXPECT_FLOAT_EQ(clip->SampleAt(0)->inputBotConfiguration[0], 0.2_r);
  EXPECT_FLOAT_EQ(clip->SampleAt(0)->inputBotConfiguration[1], 0.8_r);
  EXPECT_TRUE(NearEqual(
      clip->SampleAt(0)->recordedWorldFrameFromInputBaseFrame,
      TransformRT{Real3{0_r, 2_r, 3_r}} *
          TransformRT{Quaternion::RotationZ(0.7_r), Real3{0.3_r, 0.4_r, 0.5_r}},
      1e-4_r));
}
TEST(IKRecordingClipTest, SphericalPrismaticAndHardJointsRetainNativeComponents) {
  auto inputBot = Robot(3, true);
  inputBot.joints[1].type = ArticulatedJointType::Spherical;
  inputBot.joints[2].type = ArticulatedJointType::Prismatic;
  RebuildBotData(inputBot, test::ExpectOK{});
  ASSERT_EQ(inputBot._numDofs, 5);
  auto row = Sample(0, 0, 0.1, 5);
  row.positions = {0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5};
  Fixture file;
  file.Write({row}, Meta(inputBot));
  auto clip = file.Good(inputBot);
  ASSERT_NE(clip, nullptr);
  EXPECT_EQ(
      clip->SampleAt(0)->inputBotConfiguration,
      (DynamicArray<real>{0.1_r, 0.2_r, 0.3_r, 0.4_r, 0.5_r}));
}
TEST(IKRecordingClipTest, MissingFieldsOrMismatchedFrameTimeKeysFailWholeClip) {
  for (auto field : {kPositions, kTranslation, kRotation}) {
    for (int mutation = 0; mutation < 4; ++mutation) {
      SCOPED_TRACE(field);
      SCOPED_TRACE(mutation);
      auto columns = State();
      auto& rows = columns[{kEntity, field}];
      if (mutation == 0)
        columns.erase({kEntity, field});
      if (mutation == 1)
        rows.pop_back();
      if (mutation == 2)
        rows[1].frame += 1;
      if (mutation == 3)
        rows[1].time += 1;
      Fixture file;
      file.WriteRaw(columns);
      file.Reject(mutation == 0 ? "no temporal rows" : "alignment mismatch");
    }
  }
}
TEST(IKRecordingClipTest, DuplicateFrameTimeReversalNegativeAndMissingTimelinesReject) {
  for (auto samples :
       {std::vector<Snapshot>{Sample(0, 0), Sample(1, 0)},
        std::vector<Snapshot>{Sample(10, 0), Sample(9, 1)},
        std::vector<Snapshot>{Sample(-1, 0)},
        std::vector<Snapshot>{Sample(0, -1)}}) {
    Fixture file;
    file.WriteRaw(State(samples));
    file.Reject("nonnegative");
  }
  for (bool frame : {false, true}) {
    Fixture file;
    file.WriteRaw(State(), Meta(), {.frameTimeline = frame, .simTimeline = !frame});
    file.Reject("timeline");
  }
}
TEST(IKRecordingClipTest, CorruptWidthsAndNonfiniteValuesFailWholeClipWithoutHoldingPose) {
  for (auto field : {kPositions, kTranslation, kRotation}) {
    for (size_t width : {size_t{0}, size_t{1}, size_t{9}}) {
      auto columns = State();
      columns[{kEntity, field}][1].values.resize(width);
      Fixture file;
      file.WriteRaw(columns);
      file.Reject("row 1");
    }
    for (auto bad :
         {std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::quiet_NaN(),
          1e100}) {
      auto columns = State();
      columns[{kEntity, field}][1].values[0] = bad;
      Fixture file;
      file.WriteRaw(columns);
      file.Reject("row 1");
    }
  }
}
TEST(IKRecordingClipTest, RootQuaternionMustBeUnitIncludingNegativeHemisphere) {
  for (double w : {0.0, 0.5, 2.0}) {
    auto rows = Samples();
    rows[1].rotation[3] = w;
    Fixture file;
    file.Write(rows);
    file.Reject("unit rotation");
  }
  auto row = Sample(0, 0);
  row.rotation[3] = -1;
  Fixture file;
  file.Write({row});
  ASSERT_NE(file.Good(), nullptr);
}
TEST(IKRecordingClipTest, RequiredStaticLayoutAndIdentityMetadataCannotBeMissingOrMistyped) {
  auto const baseline = Meta();
  for (auto const& [key, values] : baseline.texts) {
    if (key.second == "superdex.robotics.robot_id")
      continue; // No robotics identity uses exact physics name.
    auto metadata = baseline;
    metadata.texts.erase(key);
    Fixture file;
    file.WriteRaw(State(), metadata);
    file.Reject("No compatible");
  }
  for (auto field : {kOffsets, kTranslations, kRotations, "superdex.physics.num_dofs"}) {
    auto metadata = baseline;
    std::string const path = std::string{field} == "superdex.physics.num_dofs"
        ? kEntity
        : std::string{kEntity} + "/joint_layout";
    metadata.scalars.erase({path, field});
    Fixture file;
    file.WriteRaw(State(), metadata);
    file.Reject("No compatible");
  }
  auto metadata = Meta();
  Key const key{std::string{kEntity} + "/joint_layout", kNames};
  metadata.texts.erase(key);
  metadata.scalars[key] = {1, 2, 3};
  Fixture file;
  file.WriteRaw(State(), metadata);
  file.Reject("text");
}
TEST(IKRecordingClipTest, JointNamesTypesCountsAndPartitionAreStrict) {
  std::string const layout = std::string{kEntity} + "/joint_layout";
  for (auto values :
       std::vector<std::vector<double>>{{0, 6, 6}, {0, 6, 6.5}, {0, 6, -1}, {0, 6}, {0, 6, 9}}) {
    auto metadata = Meta();
    metadata.scalars[{layout, kOffsets}] = values;
    Fixture file;
    file.WriteRaw(State(), metadata);
    file.Reject("layout");
  }
  for (int mutation = 0; mutation < 7; ++mutation) {
    auto metadata = Meta();
    if (mutation == 0)
      metadata.texts[{layout, kNames}][1] = "unknown_joint";
    if (mutation == 1)
      metadata.texts[{layout, kNames}][1] = "root";
    if (mutation == 2)
      metadata.texts[{layout, kTypes}][1] = "prismatic";
    if (mutation == 3)
      metadata.scalars[{layout, kTranslations}][1] = 1;
    if (mutation == 4)
      metadata.scalars[{layout, kRotations}][0] = 2;
    if (mutation == 5)
      metadata.texts[{layout, kTypes}][0] = "spherical";
    if (mutation == 6)
      metadata.scalars[{kEntity, "superdex.physics.num_dofs"}] = {9};
    Fixture file;
    file.WriteRaw(State(), metadata);
    file.Reject("No compatible");
  }
}
TEST(IKRecordingClipTest, StaticStateAndTemporalMetadataDoNotBecomeSnapshots) {
  Fixture stat;
  stat.WriteRaw(State(), Meta(), {.staticState = true});
  stat.Reject("static constants");
  auto metadata = Meta();
  auto columns = State();
  Key const layout{std::string{kEntity} + "/joint_layout", kOffsets};
  columns[layout] = {{0, 0, {0, 6, 7}}};
  Fixture temporal;
  temporal.WriteRaw(columns, metadata);
  temporal.Reject("must be static");
  columns = State();
  columns[{"episode", "superdex.episode.units_per_meter"}] = {{0, 0, {1}}};
  Fixture units;
  units.WriteRaw(columns);
  units.Reject("static episode");
}
TEST(IKRecordingClipTest, EpisodeCoordinatesUnitsAndSchemaAreStrict) {
  Fixture minor;
  minor.WriteRaw(State(), Meta(), {.schema = "1.7"});
  ASSERT_NE(minor.Good(), nullptr);
  Fixture major;
  major.WriteRaw(State(), Meta(), {.schema = "2.0"});
  major.Reject("schema");
  Fixture space;
  space.WriteRaw(State(), Meta(), {.space = "FRU"});
  space.Reject("FLU");
  for (double units : {0.0, -1.0, 100.0, std::numeric_limits<double>::infinity()}) {
    Fixture file;
    file.WriteRaw(State(), Meta(), {.units = units});
    file.Reject("units_per_meter");
  }
}
TEST(IKRecordingClipTest, InputBotLayoutAndUnsupportedRootFramesAreRejected) {
  Fixture file;
  file.Write();
  for (int mutation = 0; mutation < 7; ++mutation) {
    auto inputBot = Robot();
    if (mutation == 0)
      inputBot.name.clear();
    if (mutation == 1)
      inputBot._numDofs = 3;
    if (mutation == 2)
      inputBot._dofIndices[0] = 0;
    if (mutation == 3)
      inputBot.joints[1].name = "root";
    if (mutation == 4)
      inputBot.joints[0].type = ArticulatedJointType::Spherical;
    if (mutation == 5)
      inputBot.joints[0].parentLinkFromJoint.SetTranslation(Real3{1_r, 0_r, 0_r});
    if (mutation == 6)
      inputBot.joints[1].axis[0] = std::numeric_limits<real>::quiet_NaN();
    file.Reject("Input bot", inputBot);
  }
}
TEST(IKRecordingClipTest, EmptyRejectsAndSingleZeroTimePoseIsAvailable) {
  Fixture empty;
  empty.Write({});
  empty.Reject("no temporal rows");
  Fixture single;
  single.Write({Sample(0, 0)});
  auto clip = single.Good();
  ASSERT_NE(clip, nullptr);
  EXPECT_EQ(clip->GetTimes(), (std::vector<int64_t>{0}));
  EXPECT_EQ(clip->GetDurationSeconds(), 0);
  EXPECT_TRUE(clip->SampleAt(0)->hasPose);
}
TEST(IKRecordingClipTest, CancellationFailedReplacementAndUnsupportedFilesPreserveExistingClip) {
  Fixture file;
  file.Write();
  auto current = file.Good();
  ASSERT_NE(current, nullptr);
  auto const before = current->SampleAt(20)->inputBotConfiguration;
  std::atomic_bool cancelled{true};
  EXPECT_EQ(file.Load(Robot(), &cancelled), nullptr);
  EXPECT_NE(file.diagnostic.find("cancelled"), std::string::npos);
  file.path = file.directory / "missing.rrd";
  file.Reject("Cannot read recording");
  file.path = file.directory / "legacy.mcap";
  file.Reject("expected native RRD");
  file.path = file.directory / "truncated.rrd";
  {
    std::ofstream stream{file.path, std::ios::binary};
    stream << "not an RRD";
  }
  file.Reject("");
  EXPECT_EQ(current->SampleAt(20)->inputBotConfiguration, before);
}
TEST(IKRecordingClipTest, FileAndLayoutSizeCapsFailBeforeConstructingUnboundedClip) {
  Fixture file;
  file.path = file.directory / "too_large.rrd";
  // Sparse file: this tests the pre-open cap without writing a gigabyte of payload.
  {
    std::ofstream stream{file.path, std::ios::binary};
    stream.seekp(1024ULL * 1024 * 1024);
    stream.put('x');
  }
  file.Reject("1 GiB");
  auto metadata = Meta();
  metadata.scalars[{kEntity, "superdex.physics.num_dofs"}] = {4097};
  Fixture layout;
  layout.WriteRaw(State(), metadata);
  layout.Reject("num_dofs");
}
TEST(IKRecordingClipTest, LoadsUserSceneRecordingAndReadsBackNativeQAndRootForEachInputBot) {
  auto const* path = std::getenv("SUPERDEX_TEST_INPUT_RRD");
  auto const* inputBotPath = std::getenv("SUPERDEX_TEST_INPUT_BOT");
  if (!path || !inputBotPath)
    GTEST_SKIP() << "Set actual native RRD and input-bot paths";
  Error error;
  auto inputBot = FileBotLoader{}.LoadBotPrefab(inputBotPath, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  std::string diagnostic;
  auto clip = IKRecordingClip::Load(path, inputBot, diagnostic);
  ASSERT_NE(clip, nullptr) << diagnostic;
  auto reader = EpisodeReader::Open(path, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  auto const entity = "data/physics/actors/" + std::string{inputBot.name};
  auto const native = reader->ReadScalars(entity, kPositions, error);
  auto const xyz = reader->ReadScalars(entity, kTranslation, error);
  auto const xyzw = reader->ReadScalars(entity, kRotation, error);
  auto const offsets = reader->ReadStaticScalars(entity + "/joint_layout", kOffsets, error);
  auto const transCounts =
      reader->ReadStaticScalars(entity + "/joint_layout", kTranslations, error);
  auto const rotCounts = reader->ReadStaticScalars(entity + "/joint_layout", kRotations, error);
  auto const names = reader->ReadStaticTexts(entity + "/joint_layout", kNames, error);
  auto const rootEntity =
      "world/" + std::string{inputBot.name} + "/links/" + std::string{inputBot.links[0].name};
  auto const rootXyz = reader->ReadScalars(rootEntity, "Transform3D:translation", error);
  auto const rootXyzw = reader->ReadScalars(rootEntity, "Transform3D:quaternion", error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_EQ(native.GetNumRows(), clip->GetTimes().size());
  ASSERT_EQ(inputBot._numDofs, 26);
  ASSERT_EQ(native.GetNumRows(), 5158);
  EXPECT_EQ(clip->GetFirstTime(), 6'332'254'100LL);
  EXPECT_EQ(clip->GetLastTime(), 15'388'685'900LL);
  EXPECT_DOUBLE_EQ(clip->GetDurationSeconds(), 9.0564318);
  for (size_t i = 0; i < native.GetNumRows(); ++i) {
    auto const* sample = clip->SampleAt(native.simulationTimesNs[i]);
    ASSERT_NE(sample, nullptr);
    ASSERT_TRUE(sample->hasPose);
    ASSERT_EQ(sample->inputBotConfiguration.size(), 26);
    size_t cursor = 0;
    for (size_t joint = 1; joint < inputBot.joints.size(); ++joint) {
      auto const found =
          std::find(names.begin(), names.end(), std::string_view{inputBot.joints[joint].name});
      ASSERT_NE(found, names.end());
      size_t const j = static_cast<size_t>(found - names.begin());
      for (size_t component = 0; component < static_cast<size_t>(transCounts[j] + rotCounts[j]);
           ++component) {
        EXPECT_EQ(
            sample->inputBotConfiguration[cursor++],
            static_cast<real>(native.GetRow(i)[static_cast<size_t>(offsets[j]) + component]));
      }
    }
    EXPECT_TRUE(NearEqual(clip->InputBotRoot(*sample, {}, true), TransformRT{}, 1e-4_r));
    EXPECT_TRUE(IsFinite(clip->InputBotRoot(*sample, {}, false)));
  }
  // Native root-link transforms are sparse/deduplicated. Compare by exact frame key, never zip.
  for (size_t i = 0; i < rootXyz.GetNumRows(); ++i) {
    auto const found =
        std::lower_bound(rootXyzw.frames.begin(), rootXyzw.frames.end(), rootXyz.frames[i]);
    ASSERT_NE(found, rootXyzw.frames.end());
    size_t const j = static_cast<size_t>(found - rootXyzw.frames.begin());
    ASSERT_EQ(rootXyzw.frames[j], rootXyz.frames[i]);
    ASSERT_EQ(rootXyzw.simulationTimesNs[j], rootXyz.simulationTimesNs[i]);
    auto const t = rootXyz.GetRow(i), r = rootXyzw.GetRow(j);
    TransformRT expected{
        Quaternion{
            static_cast<real>(r[0]),
            static_cast<real>(r[1]),
            static_cast<real>(r[2]),
            static_cast<real>(r[3])},
        Real3{static_cast<real>(t[0]), static_cast<real>(t[1]), static_cast<real>(t[2])}};
    auto const anchorT = xyz.GetRow(0), anchorR = xyzw.GetRow(0);
    TransformRT anchor{
        Quaternion{
            static_cast<real>(anchorR[0]),
            static_cast<real>(anchorR[1]),
            static_cast<real>(anchorR[2]),
            static_cast<real>(anchorR[3])},
        Real3{
            static_cast<real>(anchorT[0]),
            static_cast<real>(anchorT[1]),
            static_cast<real>(anchorT[2])}};
    EXPECT_TRUE(NearEqual(
        clip->SampleAt(rootXyz.simulationTimesNs[i])->recordedWorldFrameFromInputBaseFrame,
        anchor * expected,
        1e-4_r));
  }
  // Compare with the SAME native reduced-pose FK used by PhysicsRecorder, including both rest
  // frames. Quaternion signs are not physical orientation differences. Aggregate errors once per
  // link so even an incompatible model cannot flood the test runner's output.
  DynamicArray<int> parents;
  DynamicArray<ArticulatedJointType> jointTypes;
  DynamicArray<Real3> axes;
  DynamicArray<TransformRT> childRest, parentRest;
  for (size_t link = 0; link < inputBot.links.size(); ++link) {
    parents.push_back(inputBot.links[link].parentLink);
    jointTypes.push_back(inputBot.joints[link].type);
    axes.push_back(inputBot.joints[link].axis);
    auto child = inputBot.links[link].parentJointFromLink;
    auto parent = inputBot.joints[link].parentLinkFromJoint;
    child.SetTranslation(child.GetTranslation() * inputBot.scale);
    parent.SetTranslation(parent.GetTranslation() * inputBot.scale);
    childRest.push_back(child);
    parentRest.push_back(parent);
  }
  // Geometry size is independent of DOF/basis compatibility. This UE recording has runtime
  // uniform hand scaling, which native prefab_name/layout does not encode. Measure it from a
  // nonzero hard-joint rest translation, then require that SAME scale to fit every native link.
  std::vector<std::string> linkPaths(inputBot.links.size());
  linkPaths[0] = rootEntity;
  real recordedGeometryScale = 0;
  for (size_t link = 1; link < inputBot.links.size(); ++link) {
    auto const parent = static_cast<size_t>(inputBot.links[link].parentLink);
    linkPaths[link] = linkPaths[parent] + "/" + std::string{inputBot.links[link].name};
    auto const restTranslation = (parentRest[link] * childRest[link]).GetTranslation();
    if (recordedGeometryScale == 0 && jointTypes[link] == ArticulatedJointType::Hard &&
        NormSqr(restTranslation) > 1e-8_r) {
      auto const rows = reader->ReadScalars(linkPaths[link], "Transform3D:translation", error);
      ASSERT_TRUE(error.IsOK()) << error.ToString();
      ASSERT_GT(rows.GetNumRows(), 0);
      auto const t = rows.GetRow(0);
      Real3 const recorded{
          static_cast<real>(t[0]), static_cast<real>(t[1]), static_cast<real>(t[2])};
      recordedGeometryScale = Dot(recorded, restTranslation) / NormSqr(restTranslation);
      ASSERT_TRUE(IsFinite(recordedGeometryScale));
      ASSERT_GT(recordedGeometryScale, 0);
    }
  }
  ASSERT_GT(recordedGeometryScale, 0);
  std::cout << "Recorded geometry/input uniform scale=" << recordedGeometryScale << "\n";
  for (size_t link = 0; link < inputBot.links.size(); ++link) {
    childRest[link].SetTranslation(childRest[link].GetTranslation() * recordedGeometryScale);
    parentRest[link].SetTranslation(parentRest[link].GetTranslation() * recordedGeometryScale);
  }
  superdex::recorder::detail::LinkKinematics kinematics{
      parents, jointTypes, axes, childRest, parentRest, error};
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  size_t verifiedLinkRows = 0;
  for (size_t link = 1; link < inputBot.links.size(); ++link) {
    SCOPED_TRACE(inputBot.links[link].name.c_str());
    auto const parent = static_cast<size_t>(inputBot.links[link].parentLink);
    linkPaths[link] = linkPaths[parent] + "/" + std::string{inputBot.links[link].name};
    auto const tRows = reader->ReadScalars(linkPaths[link], "Transform3D:translation", error);
    auto const rRows = reader->ReadScalars(linkPaths[link], "Transform3D:quaternion", error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    ASSERT_EQ(tRows.frames, rRows.frames);
    ASSERT_EQ(tRows.simulationTimesNs, rRows.simulationTimesNs);
    real maxTranslationError = 0, maxRotationError = 0;
    for (size_t i = 0; i < tRows.GetNumRows(); ++i) {
      auto const state =
          std::lower_bound(native.frames.begin(), native.frames.end(), tRows.frames[i]);
      ASSERT_NE(state, native.frames.end());
      size_t const stateIndex = static_cast<size_t>(state - native.frames.begin());
      ASSERT_EQ(native.frames[stateIndex], tRows.frames[i]);
      ASSERT_EQ(native.simulationTimesNs[stateIndex], tRows.simulationTimesNs[i]);
      auto const q = native.GetRow(stateIndex);
      DynamicArray<real> positions;
      for (auto value : q)
        positions.push_back(static_cast<real>(value));
      kinematics.Compute(positions, error);
      ASSERT_TRUE(error.IsOK()) << error.ToString();
      auto const expected = kinematics.GetParentFromLinks()[link];
      auto const t = tRows.GetRow(i), r = rRows.GetRow(i);
      TransformRT recorded{
          Quaternion{
              static_cast<real>(r[0]),
              static_cast<real>(r[1]),
              static_cast<real>(r[2]),
              static_cast<real>(r[3])},
          Real3{static_cast<real>(t[0]), static_cast<real>(t[1]), static_cast<real>(t[2])}};
      maxTranslationError = std::max(
          maxTranslationError, Norm(expected.GetTranslation() - recorded.GetTranslation()));
      maxRotationError = std::max(
          maxRotationError,
          std::abs(Normalize(expected.GetRotation().GetConjugate() * recorded.GetRotation())
                       .GetAngle()));
      ++verifiedLinkRows;
    }
    std::cout << "Native FK link=" << inputBot.links[link].name << " rows=" << tRows.GetNumRows()
              << " max_translation_m=" << maxTranslationError
              << " max_rotation_rad=" << maxRotationError << "\n";
    EXPECT_LE(maxTranslationError, 1e-4_r);
    EXPECT_LE(maxRotationError, 1e-4_r);
  }
  std::cout << "Native nonroot FK link rows verified=" << verifiedLinkRows << "\n";
  EXPECT_TRUE(NearEqual(
      clip->InputBotRoot(*clip->SampleAt(clip->GetFirstTime()), {}, false), TransformRT{}, 1e-4_r));
  EXPECT_FALSE(NearEqual(
      clip->SampleAt(clip->GetFirstTime())->recordedWorldFrameFromInputBaseFrame,
      clip->SampleAt(clip->GetTimes()[clip->GetTimes().size() / 2])
          ->recordedWorldFrameFromInputBaseFrame,
      1e-4_r));
  std::cout << std::setprecision(12) << "Native scene model=" << inputBot.name
            << " samples=" << native.GetNumRows() << " first_frame=" << native.frames.front()
            << " last_frame=" << native.frames.back() << " first_ns=" << clip->GetFirstTime()
            << " last_ns=" << clip->GetLastTime() << " duration_s=" << clip->GetDurationSeconds()
            << "\n";
  for (size_t i : {size_t{0}, native.GetNumRows() / 2, native.GetNumRows() - 1}) {
    auto const* sample = clip->SampleAt(native.simulationTimesNs[i]);
    std::cout << "frame=" << sample->frame << " sim_ns=" << sample->time << " q26:";
    for (auto q : sample->inputBotConfiguration)
      std::cout << " " << q;
    std::cout << " wrist_xyz:";
    for (int axis = 0; axis < 3; ++axis)
      std::cout << " " << sample->recordedWorldFrameFromInputBaseFrame.GetTranslation()[axis];
    std::cout << " wrist_xyzw:";
    for (int axis = 0; axis < 4; ++axis)
      std::cout << " " << sample->recordedWorldFrameFromInputBaseFrame.GetRotation().data[axis];
    std::cout << "\n";
  }
}
using Clock = IKRecordingClock::Clock;
Clock::time_point Now(int64_t ns) {
  return Clock::time_point{std::chrono::nanoseconds{ns}};
}
TEST(IKRecordingClockTest, IrregularCadenceUsesElapsedNanosecondsAndWraps) {
  IKRecordingClock clock;
  clock.Configure(100, 1100, {100, 300, 1100});
  clock.Play(Now(0));
  EXPECT_EQ(clock.Tick(Now(37)), 137);
  EXPECT_EQ(clock.Tick(Now(600)), 700);
  EXPECT_EQ(clock.Tick(Now(1000)), 100);
  EXPECT_EQ(clock.Tick(Now(2300)), 400);
}
TEST(IKRecordingClockTest, PauseAndDeactivationExcludeIdleTime) {
  IKRecordingClock clock;
  clock.Configure(0, 1000, {0, 1000});
  clock.Play(Now(0));
  clock.Pause(Now(200));
  EXPECT_EQ(clock.Tick(Now(10000)), 200);
  clock.Play(Now(10000));
  EXPECT_EQ(clock.Tick(Now(10100)), 300);
  clock.Pause(Now(10100));
  clock.Play(Now(20000));
  EXPECT_EQ(clock.Tick(Now(20100)), 400);
}
TEST(IKRecordingClockTest, StepDistinctTimesAndStopRestoreCursor) {
  IKRecordingClock clock;
  clock.Configure(10, 50, {50, 10, 20, 20, 40});
  clock.Step(Now(0));
  EXPECT_EQ(clock.GetCursor(), 20);
  clock.Step(Now(0));
  EXPECT_EQ(clock.GetCursor(), 40);
  clock.Step(Now(0));
  EXPECT_EQ(clock.GetCursor(), 50);
  clock.Step(Now(0));
  EXPECT_EQ(clock.GetCursor(), 10);
  clock.Play(Now(0));
  clock.Tick(Now(8));
  clock.Stop();
  EXPECT_EQ(clock.GetCursor(), 10);
  EXPECT_FALSE(clock.IsPlaying());
}
TEST(IKRecordingClockTest, ZeroDurationHoldsAndNegativeOriginsWork) {
  IKRecordingClock clock;
  clock.Configure(-10, -10, {-10});
  clock.Play(Now(0));
  EXPECT_EQ(clock.Tick(Now(100)), -10);
  clock.Step(Now(100));
  EXPECT_EQ(clock.GetCursor(), -10);
  clock.Configure(-100, 100, {-100, 0, 100});
  clock.Play(Now(0));
  EXPECT_EQ(clock.Tick(Now(150)), 50);
  EXPECT_EQ(clock.Tick(Now(200)), -100);
}
TEST(IKRecordingClockTest, ExtremeOriginsDoNotOverflowSignedArithmetic) {
  IKRecordingClock clock;
  auto const first = std::numeric_limits<int64_t>::min();
  clock.Configure(first, std::numeric_limits<int64_t>::max(), {first, 0});
  clock.Play(Now(0));
  EXPECT_EQ(clock.Tick(Now(200)), first + 200);
}
} // namespace
