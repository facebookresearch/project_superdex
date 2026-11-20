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

#include <superdex_recorder/robot_shadow.h>
#include <superdex_recorder/robotics_recorder.h>
#include <superdex_recorder/skeleton_shadow.h>

#include <superdex_robotics/core/context.h>
#include <superdex_robotics/utils/bot_utils.h>

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace superdex::recorder {
namespace {

using RecordingHook = std::function<void(robotics::ControllerRecordingSink&, Error&)>;

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    auto const suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    _path = std::filesystem::temp_directory_path() /
        ("superdex_recorder_robotics_test_" + std::to_string(suffix) + "_" +
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

class TestController final : public robotics::ControllerBase {
 public:
  TestController(robotics::BotPrefab const* prefab, Actor* actor, Error& error)
      : ControllerBase(prefab, actor, error) {}

  static constexpr std::string_view TypeName() {
    return "TEST";
  }

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  void Reset() override {}

  void ConfigureFromSceneEntry(
      std::string_view /*paramArgs*/,
      std::string_view /*initArgs*/,
      Error& /*error*/) override {}

  void ConfigureRecording(robotics::ControllerRecordingSink& sink, Error& error) const override {
    configureRecording(sink, error);
  }

  void Record(robotics::ControllerRecordingSink& sink, Error& error) const override {
    record(sink, error);
  }

  RecordingHook configureRecording = [](auto& /*sink*/, auto& /*error*/) {};
  RecordingHook record = [](auto& /*sink*/, auto& /*error*/) {};
};

robotics::BotPrefab MakeRobotPrefab(Error& error) {
  robotics::BotPrefab prefab;
  prefab.name = "robot";
  prefab.links.resize(2);
  prefab.links[0].name = "base";
  prefab.links[0].parentLink = robotics::kIndexNone;
  prefab.links[1].name = "link";
  prefab.links[1].parentLink = 0;
  prefab.joints.resize(2);
  prefab.joints[0].name = "base_joint";
  prefab.joints[0].type = ArticulatedJointType::Hard;
  prefab.joints[1].name = "joint";
  prefab.joints[1].type = ArticulatedJointType::Revolute;
  prefab.joints[1].axis = {0_r, 0_r, 1_r};
  robotics::RebuildBotData(prefab, error);
  return prefab;
}

SkeletonShadowConfig MakeSkeletonConfig(std::string id) {
  return {
      .id = std::move(id),
      .keypointNames = {"wrist", "knuckle", "tip"},
      .parentIndices = {-1, 0, 1}};
}

void WriteState(robotics::ControllerRecordingSink& sink, std::string_view field, Error& error) {
  std::array<real, 2> const values{1_r, 2_r};
  std::array<robotics::ControllerScalarValues, 1> const fields{{{field, values}}};
  sink.WriteScalars("state", fields, error);
}

class RoboticsRecorderTest : public testing::Test {
 protected:
  void SetUp() override {
    context = CreateContext(0);
    ASSERT_NE(context, nullptr);
    scene = context->CreateScene("recorder robotics test");
    ASSERT_NE(scene, nullptr);
    roboticsContext = robotics::CreateRoboticsContext();
    ASSERT_NE(roboticsContext, nullptr);
    roboticsContext->RegisterController<TestController>();
    robot = robotics::CreateBot(scene, MakeRobotPrefab(error), roboticsContext, error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    controller = static_cast<TestController*>(
        robot->CreateController(TestController::TypeName(), "test", error));
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    episode = recorder.BeginEpisode({}, error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    ASSERT_NE(episode, nullptr);
  }

  void TearDown() override {
    if (roboticsContext != nullptr) {
      robotics::DestroyRoboticsContext(roboticsContext);
    }
    if (scene != nullptr) {
      context->DestroyScene(scene);
    }
    if (context != nullptr) {
      DestroyContext(context);
    }
  }

  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  std::unique_ptr<Episode> episode;
  Context* context = nullptr;
  Scene* scene = nullptr;
  robotics::RoboticsContext* roboticsContext = nullptr;
  robotics::Bot* robot = nullptr;
  TestController* controller = nullptr;
  Error error;
};

TEST_F(RoboticsRecorderTest, CommitsControllerConstantsAndSamples) {
  controller->configureRecording = [](auto& sink, auto& error) {
    std::array<real, 2> const gains{1_r, 2_r};
    sink.WriteConstantScalars("gains", gains, error);
    sink.WriteConstantTransform("ee_link_from_ee", TransformRT::Identity(), error);
  };
  controller->record = [](auto& sink, auto& error) {
    WriteState(sink, "target", error);
    sink.WriteTransform("target_pose", TransformRT::Identity(), error);
  };
  RoboticsRecorder controllers{*episode, *robot, error};

  controllers.Record({.frame = 0, .simulationTimeSec = 0.0}, error);
  controllers.Record({.frame = 1, .simulationTimeSec = 0.01}, error);
  episode->CommitSync(error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_TRUE(std::filesystem::exists(episode->GetOutputPath()));
}

TEST_F(RoboticsRecorderTest, RecordAppendsToTheSameSeries) {
  controller->record = [](auto& sink, auto& error) { WriteState(sink, "target", error); };
  RoboticsRecorder controllers{*episode, *robot, error};

  controllers.Record({.frame = 0, .simulationTimeSec = 0.0}, error);
  controllers.Record({.frame = 1, .simulationTimeSec = 0.01}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  /* The series already holds frame 1. */
  controllers.Record({.frame = 1, .simulationTimeSec = 0.01}, error);
  EXPECT_FALSE(error.IsOK());
}

TEST_F(RoboticsRecorderTest, ConfigureRecordingRejectsScalarSamples) {
  controller->configureRecording = [](auto& sink, auto& error) {
    WriteState(sink, "target", error);
  };
  RoboticsRecorder controllers{*episode, *robot, error};

  EXPECT_FALSE(error.IsOK());
}

TEST_F(RoboticsRecorderTest, ConfigureRecordingRejectsTransformSamples) {
  controller->configureRecording = [](auto& sink, auto& error) {
    sink.WriteTransform("target_pose", TransformRT::Identity(), error);
  };
  RoboticsRecorder controllers{*episode, *robot, error};

  EXPECT_FALSE(error.IsOK());
}

TEST_F(RoboticsRecorderTest, RecordRejectsChangedSeriesFields) {
  std::string_view field = "target";
  controller->record = [&field](auto& sink, auto& error) { WriteState(sink, field, error); };
  RoboticsRecorder controllers{*episode, *robot, error};
  controllers.Record({.frame = 0, .simulationTimeSec = 0.0}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  field = "output";
  controllers.Record({.frame = 1, .simulationTimeSec = 0.01}, error);

  EXPECT_FALSE(error.IsOK());
}

TEST_F(RoboticsRecorderTest, RecordRejectsEmptyFieldNames) {
  controller->record = [](auto& sink, auto& error) { WriteState(sink, "", error); };
  RoboticsRecorder controllers{*episode, *robot, error};
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  controllers.Record({.frame = 0, .simulationTimeSec = 0.0}, error);

  EXPECT_FALSE(error.IsOK());
}

TEST_F(RoboticsRecorderTest, RecordRejectsSamplesWithoutFields) {
  controller->record = [](auto& sink, auto& error) { sink.WriteScalars("state", {}, error); };
  RoboticsRecorder controllers{*episode, *robot, error};
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  controllers.Record({.frame = 0, .simulationTimeSec = 0.0}, error);

  EXPECT_FALSE(error.IsOK());
}

TEST_F(RoboticsRecorderTest, RecordFailsAfterRecordedRobotIsDestroyed) {
  robotics::Bot* const bare =
      robotics::CreateBot(scene, MakeRobotPrefab(error), roboticsContext, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(bare, nullptr);
  RoboticsRecorder recorded{*episode, *bare, error};
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  roboticsContext->DestroyBot(scene, bare->GetHandle());
  recorded.Record({.frame = 0, .simulationTimeSec = 0.0}, error);

  EXPECT_NE(error.ToString().find("The recorded robot was destroyed."), std::string::npos)
      << error.ToString();
}

TEST_F(RoboticsRecorderTest, SkeletonShadowCommitsKeypointsAndUntrackedSamples) {
  SkeletonShadow skeleton{*episode, *robot, MakeSkeletonConfig("tracked_hand"), error};
  std::array<Real3, 3> const keypoints{Real3{0_r, 0_r, 0_r}, {0.1_r, 0_r, 0_r}, {0.2_r, 0_r, 0_r}};

  skeleton.Record({.frame = 0, .simulationTimeSec = 0.0}, keypoints, error);
  skeleton.RecordUntracked({.frame = 1, .simulationTimeSec = 0.01}, error);
  skeleton.Record({.frame = 2, .simulationTimeSec = 0.02}, keypoints, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  Error repeatedFrameError;
  skeleton.RecordUntracked({.frame = 2, .simulationTimeSec = 0.02}, repeatedFrameError);
  episode->CommitSync(error);

  EXPECT_FALSE(repeatedFrameError.IsOK());
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_TRUE(std::filesystem::exists(episode->GetOutputPath()));
}

TEST_F(RoboticsRecorderTest, SkeletonShadowRejectsInvalidConfigs) {
  auto const withTopology =
      [](std::string id, std::vector<std::string> names, std::vector<int32_t> parents) {
        auto config = MakeSkeletonConfig(std::move(id));
        config.keypointNames = std::move(names);
        config.parentIndices = std::move(parents);
        return config;
      };
  auto const withRadius = [](std::string id, float radius) {
    auto config = MakeSkeletonConfig(std::move(id));
    config.keypointRadiusMeters = radius;
    return config;
  };
  std::vector<std::string> tooManyNames(size_t{std::numeric_limits<uint16_t>::max()} + 2);
  for (size_t i = 0; i < tooManyNames.size(); ++i) {
    tooManyNames[i] = std::to_string(i);
  }
  // Each case has its own ID, so no case fails because an earlier case created its series.
  std::vector<std::pair<SkeletonShadowConfig, std::string_view>> const cases{
      {MakeSkeletonConfig(""), "Shadow ID must not be empty."},
      {withTopology("no_keypoints", {}, {}), "at least one keypoint"},
      {withTopology("missing_parent", {"wrist", "tip"}, {-1}), "differ in size"},
      {withTopology("duplicate_name", {"wrist", "tip", "tip"}, {-1, 0, 1}), "must be unique"},
      {withTopology("empty_name", {"wrist", "", "tip"}, {-1, 0, 1}), "must be unique"},
      {withTopology("own_parent", {"wrist", "knuckle", "tip"}, {-1, 1, 1}), "must precede it"},
      {withTopology("later_parent", {"wrist", "knuckle", "tip"}, {-1, 2, 0}), "must precede it"},
      {withTopology("invalid_parent", {"wrist", "knuckle", "tip"}, {-2, 0, 1}), "must precede it"},
      {withTopology(
           "too_many_keypoints", tooManyNames, std::vector<int32_t>(tooManyNames.size(), -1)),
       "too many keypoints"},
      {withRadius("zero_radius", 0.0f), "radius must be finite and positive"},
      {withRadius("nan_radius", std::numeric_limits<float>::quiet_NaN()),
       "radius must be finite and positive"},
      {withRadius("infinite_radius", std::numeric_limits<float>::infinity()),
       "radius must be finite and positive"},
  };

  for (auto const& [config, expectedError] : cases) {
    SCOPED_TRACE(config.id);
    Error constructionError;
    SkeletonShadow skeleton{*episode, *robot, config, constructionError};
    EXPECT_NE(constructionError.ToString().find(expectedError), std::string::npos)
        << constructionError.ToString();
  }
}

TEST_F(RoboticsRecorderTest, ShadowsOfOneRobotNeedDistinctIds) {
  RobotShadow command{*episode, *robot, {.id = "command"}, error};
  SkeletonShadow trackedHand{*episode, *robot, MakeSkeletonConfig("tracked_hand"), error};
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  Error skeletonError;
  SkeletonShadow skeletonClash{*episode, *robot, MakeSkeletonConfig("command"), skeletonError};
  Error robotError;
  RobotShadow robotClash{*episode, *robot, {.id = "tracked_hand"}, robotError};
  Error duplicateError;
  SkeletonShadow duplicate{*episode, *robot, MakeSkeletonConfig("tracked_hand"), duplicateError};
  Error recordError;
  skeletonClash.RecordUntracked({.frame = 0, .simulationTimeSec = 0.0}, recordError);
  Error robotRecordError;
  robotClash.Record(
      {.frame = 0, .simulationTimeSec = 0.0}, TransformRT::Identity(), {}, robotRecordError);

  for (Error const* clashError : {&skeletonError, &robotError, &duplicateError}) {
    EXPECT_NE(clashError->ToString().find("A series already exists"), std::string::npos)
        << clashError->ToString();
  }
  for (Error const* failedRecordError : {&recordError, &robotRecordError}) {
    EXPECT_NE(failedRecordError->ToString().find("construction failed"), std::string::npos)
        << failedRecordError->ToString();
  }
}

TEST_F(RoboticsRecorderTest, SkeletonShadowWithAnInvalidRadiusRecordsNothing) {
  auto config = MakeSkeletonConfig("tracked_hand");
  config.keypointRadiusMeters = -0.01f;
  std::array<Real3, 3> const keypoints{Real3{0_r, 0_r, 0_r}, {0.1_r, 0_r, 0_r}, {0.2_r, 0_r, 0_r}};

  Error constructionError;
  SkeletonShadow invalid{*episode, *robot, config, constructionError};
  Error recordError;
  invalid.Record({.frame = 0, .simulationTimeSec = 0.0}, keypoints, recordError);
  // The failed shadow left its path free.
  SkeletonShadow valid{*episode, *robot, MakeSkeletonConfig("tracked_hand"), error};

  EXPECT_NE(constructionError.ToString().find("radius"), std::string::npos)
      << constructionError.ToString();
  EXPECT_NE(recordError.ToString().find("construction failed"), std::string::npos)
      << recordError.ToString();
  EXPECT_TRUE(error.IsOK()) << error.ToString();
}

TEST_F(RoboticsRecorderTest, SkeletonShadowRejectsMissingAndNonFiniteKeypoints) {
  SkeletonShadow skeleton{*episode, *robot, MakeSkeletonConfig("tracked_hand"), error};
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  std::array<Real3, 3> keypoints{Real3{0_r, 0_r, 0_r}, {0.1_r, 0_r, 0_r}, {0.2_r, 0_r, 0_r}};

  Error missingError;
  skeleton.Record(
      {.frame = 0, .simulationTimeSec = 0.0},
      Span<Real3 const>{keypoints}.subspan(0, 2),
      missingError);
  keypoints[1][2] = std::numeric_limits<real>::quiet_NaN();
  Error nanError;
  skeleton.Record({.frame = 1, .simulationTimeSec = 0.01}, keypoints, nanError);
  keypoints[1][2] = std::numeric_limits<real>::infinity();
  Error infinityError;
  skeleton.Record({.frame = 2, .simulationTimeSec = 0.02}, keypoints, infinityError);

  EXPECT_NE(missingError.ToString().find("one position per keypoint"), std::string::npos)
      << missingError.ToString();
  EXPECT_NE(nanError.ToString().find("must be finite"), std::string::npos) << nanError.ToString();
  EXPECT_NE(infinityError.ToString().find("must be finite"), std::string::npos)
      << infinityError.ToString();
}

} // namespace
} // namespace superdex::recorder
