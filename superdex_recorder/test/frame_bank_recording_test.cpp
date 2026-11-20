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

#include <superdex_recorder/episode_reader.h>
#include <superdex_recorder/frame_bank.h>
#include <superdex_recorder/physics_recorder.h>
#include <superdex_recorder/robotics_recorder.h>

#include <superdex_robotics/controllers/controller_basic_jsc_pd.h>
#include <superdex_robotics/core/context.h>
#include <superdex_robotics/utils/bot_utils.h>

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace superdex::recorder {
namespace {

using Values = mochi::DynamicArray<real>;

constexpr real kDt = 0.01_r;
constexpr char const* kGains = R"({"Kp": [10], "Kd": [1], "saturation": [-1], "deadband": [0]})";
constexpr int kNumFrames = 5;
// The efforts applied to the hand before frames 0 and 1. It holds the second.
constexpr std::array<real, 2> kHandEfforts = {0.5_r, 0.75_r};
constexpr real kNaN = std::numeric_limits<real>::quiet_NaN();

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    auto const suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    _path = std::filesystem::temp_directory_path() /
        ("superdex_frame_bank_recording_test_" + std::to_string(suffix) + "_" +
         std::to_string(std::random_device{}()));
    std::filesystem::create_directories(_path);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(_path, error);
  }

  [[nodiscard]] std::filesystem::path const& Path() const {
    return _path;
  }

 private:
  std::filesystem::path _path;
};

/* A gripper controller that records a closure target and a target pose, each the value it was last
 * given, at each Record after it was given one. */
class GripController final : public robotics::ControllerBase {
 public:
  GripController(robotics::BotPrefab const* prefab, Actor* actor, Error& error)
      : ControllerBase(prefab, actor, error) {}

  static constexpr std::string_view TypeName() {
    return "TEST_GRIP";
  }

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  void Reset() override {}

  void ConfigureFromSceneEntry(
      std::string_view /*paramArgs*/,
      std::string_view /*initArgs*/,
      Error& /*error*/) override {}

  void Record(robotics::ControllerRecordingSink& sink, Error& error) const override {
    if (closure.empty()) {
      return;
    }
    std::array<robotics::ControllerScalarValues, 1> const fields{{{"closure_target", closure}}};
    sink.WriteScalars("state", fields, error);
    sink.WriteTransform("target", TransformRT{Real3{closure[0], 0_r, 0_r}}, error);
  }

  std::vector<real> closure;
};

/* A robot of one revolute joint on a fixed base. */
robotics::BotPrefab MakeRobotPrefab(char const* name, Error& error) {
  robotics::BotPrefab prefab;
  prefab.name = name;
  prefab.links.resize(2);
  prefab.links[0].name = "base";
  prefab.links[0].parentLink = robotics::kIndexNone;
  prefab.links[1].name = "link";
  prefab.links[1].parentLink = 0;
  for (auto& link : prefab.links) {
    link.mass = 1_r;
    link.momentOfInertia = Real6{0.01_r, 0.01_r, 0.01_r, 0_r, 0_r, 0_r};
  }
  prefab.joints.resize(2);
  prefab.joints[0].name = "base_joint";
  prefab.joints[0].type = ArticulatedJointType::Hard;
  prefab.joints[1].name = "joint";
  prefab.joints[1].type = ArticulatedJointType::Revolute;
  prefab.joints[1].axis = {0_r, 0_r, 1_r};
  robotics::RebuildBotData(prefab, error);
  return prefab;
}

/* An arm driven by BASIC_JSC_PD, and a hand with a gripper controller, recorded with the physics
 * and robotics recorders. */
class FrameBankRecordingTest : public testing::Test {
 protected:
  void SetUp() override {
    context = CreateContext(0);
    scene = context->CreateScene("frame bank recording test");
    roboticsContext = robotics::CreateRoboticsContext();
    roboticsContext->RegisterController<GripController>();
    arm = robotics::CreateBot(scene, MakeRobotPrefab("arm", error), roboticsContext, error);
    hand = robotics::CreateBot(scene, MakeRobotPrefab("hand", error), roboticsContext, error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    jsc = static_cast<robotics::ControllerBasicJscPd*>(
        arm->CreateController("BASIC_JSC_PD", "jsc", error));
    ASSERT_NE(jsc, nullptr);
    jsc->ConfigureFromSceneEntry(kGains, "", error);
    grip = static_cast<GripController*>(hand->CreateController("TEST_GRIP", "grip", error));
    ASSERT_NE(grip, nullptr);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
  }

  void TearDown() override {
    robotics::DestroyRoboticsContext(roboticsContext);
    context->DestroyScene(scene);
    DestroyContext(context);
  }

  /* Records frames 0 to kNumFrames - 1 of the swinging arm into a new episode in `run`. Before each
   * frame in `armFrames` the arm's controller computes toward 0.1 times the frame number, and
   * before each frame in `gripFrames` the gripper takes a closure of the frame number. The hand is
   * recorded at frames 0 and 1 only, after each of its efforts is applied. */
  std::filesystem::path
  Record(std::string const& run, std::set<int> const& armFrames, std::set<int> const& gripFrames) {
    jsc->Reset();
    grip->closure.clear();
    auto episode = Recorder{{.outputDirectory = directory.Path() / run}}.BeginEpisode({}, error);
    {
      PhysicsRecorder physics{*episode, *scene, error};
      RoboticsRecorder armRecorder{*episode, *arm, error};
      RoboticsRecorder handRecorder{*episode, *hand, error};
      std::array<real, 1> const velocity{1_r};
      arm->GetArticulatedActor()->SetArticulatedJointVelocities(velocity, error);
      for (int frame = 0; frame < kNumFrames; ++frame) {
        if (armFrames.contains(frame)) {
          robotics::ControllerBasicJscPd::Obsv obsv = jsc->GetCurrentObservationsFromMochi(error);
          obsv.dt = kDt;
          robotics::ControllerBasicJscPd::Target target;
          target.targetPose = {0.1_r * static_cast<real>(frame)};
          jsc->ComputeOutput(obsv, target, error);
        }
        if (gripFrames.contains(frame)) {
          grip->closure = {static_cast<real>(frame)};
        }
        if (frame > 0) {
          scene->Step(kDt);
        }
        SampleTime const time{
            .frame = static_cast<uint64_t>(frame),
            .simulationTimeSec = scene->GetTotalSimulationTime()};
        physics.Record(time, error);
        armRecorder.Record(time, error);
        if (std::cmp_less(frame, kHandEfforts.size())) {
          std::array<int, 1> const dofs{0};
          std::array<real, 1> const effort{kHandEfforts[static_cast<size_t>(frame)]};
          hand->GetArticulatedActor()->SetExternalForcesOnDofs(dofs, effort, error);
          handRecorder.Record(time, error);
        }
      }
    }
    episode->CommitSync(error);
    EXPECT_TRUE(error.IsOK()) << error.ToString();
    return episode->GetOutputPath();
  }

  TemporaryDirectory directory;
  Context* context = nullptr;
  Scene* scene = nullptr;
  robotics::RoboticsContext* roboticsContext = nullptr;
  robotics::Bot* arm = nullptr;
  robotics::Bot* hand = nullptr;
  robotics::ControllerBasicJscPd* jsc = nullptr;
  GripController* grip = nullptr;
  Error error;
};

/* Compares values bit for bit, so NaN equals NaN. */
void ExpectSameValues(Values const& actual, std::vector<real> const& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_TRUE(actual[i] == expected[i] || (std::isnan(actual[i]) && std::isnan(expected[i])))
        << "value " << i << ": " << actual[i] << " vs " << expected[i];
  }
}

/* Every physics frame is read. Each target field holds from its latest sample, with NaN before the
 * first: the hand records only at frames 0 and 1, so its gripper's target and its effort recorded
 * at frame 1 hold at later frames. */
TEST_F(FrameBankRecordingTest, ReadsEveryFrameWithHeldTargetsAndEfforts) {
  std::filesystem::path const path = Record("run", {2, 4}, {1});
  auto const recording = EpisodeReader::Open(path, error);

  BankEpisode const episode = ReadBankEpisode(*recording, error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(episode.frames, (mochi::DynamicArray<int64_t>{0, 1, 2, 3, 4}));
  ASSERT_EQ(episode.actors.size(), 2);
  ScalarRows const joints =
      recording->ReadScalars("data/physics/actors/arm", "superdex.physics.joint_positions", error);
  EXPECT_EQ(
      episode.actors[0].jointPositions,
      Values(std::vector<real>(joints.values.begin(), joints.values.end())));
  EXPECT_EQ(episode.actors[1].runtimeName, "hand");
  real const effort = kHandEfforts[1];
  EXPECT_EQ(
      episode.actors[1].jointEfforts, (Values{kHandEfforts[0], effort, effort, effort, effort}));

  ASSERT_EQ(episode.controllers.size(), 2);
  BankController const& arm = episode.controllers[0];
  EXPECT_EQ(arm.robotActor, 0);
  EXPECT_EQ(arm.id, "jsc");
  ASSERT_EQ(arm.targets.size(), 1);
  EXPECT_EQ(arm.targets[0].name, "joint_position_target");
  EXPECT_EQ(arm.targets[0].width, 1);
  ExpectSameValues(arm.targets[0].values, {kNaN, kNaN, 0.2_r, 0.2_r, 0.4_r});
  BankController const& hand = episode.controllers[1];
  EXPECT_EQ(hand.robotActor, 1);
  EXPECT_EQ(hand.id, "grip");
  ASSERT_EQ(hand.targets.size(), 3);
  EXPECT_EQ(hand.targets[0].name, "closure_target");
  ExpectSameValues(hand.targets[0].values, {kNaN, 1_r, 1_r, 1_r, 1_r});
  EXPECT_EQ(hand.targets[1].name, "target_quaternion_xyzw");
  EXPECT_EQ(hand.targets[1].width, 4);
  EXPECT_EQ(hand.targets[2].name, "target_translation_xyz");
  ExpectSameValues(
      hand.targets[2].values,
      {kNaN, kNaN, kNaN, 1_r, 0_r, 0_r, 1_r, 0_r, 0_r, 1_r, 0_r, 0_r, 1_r, 0_r, 0_r});
}

TEST_F(FrameBankRecordingTest, RefusesATargetThatChangesSize) {
  grip->closure = {0_r};
  std::filesystem::path const path = [&] {
    auto episode = Recorder{{.outputDirectory = directory.Path() / "run"}}.BeginEpisode({}, error);
    {
      PhysicsRecorder physics{*episode, *scene, error};
      RoboticsRecorder handRecorder{*episode, *hand, error};
      for (int frame = 0; frame < 2; ++frame) {
        SampleTime const time{.frame = static_cast<uint64_t>(frame), .simulationTimeSec = 0.0};
        physics.Record(time, error);
        handRecorder.Record(time, error);
        grip->closure.push_back(1_r);
      }
    }
    episode->CommitSync(error);
    return episode->GetOutputPath();
  }();
  auto const recording = EpisodeReader::Open(path, error);

  ReadBankEpisode(*recording, error);

  EXPECT_NE(std::string{error.GetDescription()}.find("changes size at frame 1"), std::string::npos)
      << error.ToString();
}

/* Episodes are packed in the order given, with their sources relative to the recordings' common
 * directory. */
TEST_F(FrameBankRecordingTest, PacksEpisodesInOrderWithTheirSources) {
  std::array<std::filesystem::path, 2> const recordings{
      Record("run_b", {1, 2, 3, 4}, {}), Record("run_a", {2, 3, 4}, {})};
  std::filesystem::path const file = directory.Path() / "bank.superdex_framebank";

  FrameBank const bank = PackFrameBank(recordings, error);
  SaveFrameBank(bank, file, error);
  auto const loaded = LoadFrameBank(file, error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_EQ(loaded->episodes.size(), 2);
  EXPECT_EQ(
      std::string_view{loaded->episodes[0].source}, "run_b/" + recordings[0].filename().string());
  EXPECT_EQ(
      std::string_view{loaded->episodes[1].source}, "run_a/" + recordings[1].filename().string());
  ExpectSameValues(
      loaded->episodes[1].controllers[0].targets[0].values, {kNaN, kNaN, 0.2_r, 0.3_r, 0.4_r});
}

TEST_F(FrameBankRecordingTest, PackingNamesTheRecordingItCannotRead) {
  std::array<std::filesystem::path, 1> const recordings{directory.Path() / "missing.rrd"};

  FrameBank const bank = PackFrameBank(recordings, error);

  std::string const message = error.GetDescription();
  EXPECT_NE(message.find("Failed to read '" + recordings[0].string() + "'"), std::string::npos)
      << message;
  EXPECT_TRUE(bank.episodes.empty());
}

} // namespace
} // namespace superdex::recorder
