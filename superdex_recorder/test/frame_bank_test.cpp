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

#include <superdex_recorder/frame_bank.h>

#include <superdex_recorder/episode_restorer.h>

#include <mochi_physics/mochi_physics.h>

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace superdex::recorder {
namespace {

using mochi::operator""_r;
using mochi::real;
using Values = mochi::DynamicArray<real>;

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    auto const suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    _path = std::filesystem::temp_directory_path() /
        ("superdex_frame_bank_test_" + std::to_string(suffix) + "_" +
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

/* `count` values from `first` on, stepping by `step`. */
Values Ramp(size_t count, double first, double step = 0.1) {
  Values values(count);
  for (size_t i = 0; i < count; ++i) {
    values[i] = static_cast<real>(first + step * static_cast<double>(i));
  }
  return values;
}

/* `rows` unit quaternions, each turned a little further about z, from `first` turns on. */
Values Rotations(size_t rows, size_t first = 0) {
  Values values;
  for (size_t row = first; row < first + rows; ++row) {
    double const halfAngle = 0.1 * static_cast<double>(row);
    for (double const value : {0.0, 0.0, std::sin(halfAngle), std::cos(halfAngle)}) {
      values.push_back(static_cast<real>(value));
    }
  }
  return values;
}

BankActor MakeActor(char const* name, char const* type, bool isStatic = false) {
  BankActor actor;
  actor.path = std::string{"data/physics/actors/"} + name;
  actor.runtimeName = name;
  actor.type = type;
  actor.isStatic = isStatic;
  return actor;
}

/* An episode of `frames` frames, with an actor of each kind a bank holds, the static table placed
 * `tableShift` along x, and the arm's joint target from the second frame on. */
BankEpisode MakeEpisode(std::vector<int64_t> const& frames, double tableShift) {
  size_t const n = frames.size();
  BankEpisode episode;
  episode.source = "run/episode_" + std::to_string(frames[0]) + ".rrd";
  episode.properties.push_back(
      {.key = "seed", .value = mochi::DynamicString{std::to_string(frames[0])}});
  episode.frames = frames;
  for (int64_t const frame : frames) {
    episode.simulationTimesSec.push_back(0.01 * static_cast<double>(frame));
    episode.stepDtsSec.push_back(0.01);
  }
  auto const start = static_cast<double>(frames[0]);
  BankActor arm = MakeActor("arm", "articulated");
  arm.spawnName = "robot";
  arm.rootRotations = Rotations(n);
  arm.rootTranslations = Ramp(3 * n, 1.0);
  arm.jointPositions = Ramp(2 * n, start - 0.5);
  arm.jointVelocities = Ramp(2 * n, 2.0);
  arm.controllerTargets = Ramp(2 * n, 0.25);
  arm.jointEfforts = Ramp(2 * n, -3.0);
  BankActor cube = MakeActor("cube", "rigid");
  cube.rootRotations = Rotations(n);
  cube.comTranslations = Ramp(3 * n, 0.3);
  cube.linearVelocities = Ramp(3 * n, start - 1.0);
  cube.angularVelocities = Ramp(3 * n, 0.5);
  BankActor duck = MakeActor("duck", "soft");
  duck.rootRotations = Rotations(n);
  duck.rootTranslations = Ramp(3 * n, 0.0);
  duck.displacements = Ramp(12 * n, start, 0.001);
  duck.nodeVelocities = Ramp(12 * n, -0.2, 0.01);
  BankActor table = MakeActor("table", "rigid", true);
  table.rootRotations = Rotations(1, static_cast<size_t>(frames[0]));
  table.rootTranslations = {static_cast<real>(tableShift), -0.5_r, 0_r};
  episode.actors = {arm, cube, duck, table};
  BankTarget target{.name = "joint_position_target", .width = 2, .values = Ramp(2 * n, 0.3)};
  target.values[0] = target.values[1] = std::numeric_limits<real>::quiet_NaN();
  episode.controllers.push_back({.robotActor = 0, .id = "jsc", .targets = {target}});
  return episode;
}

/* Two episodes of two frames and one, which differ in every value and in the table's pose. */
FrameBank TwoEpisodes() {
  FrameBank bank;
  bank.episodes = {MakeEpisode({0, 1}, -0.4), MakeEpisode({7}, 0.6)};
  return bank;
}

/* Restores a cube resting on the static table at x = 0.6 into a new scene whose table starts at
 * `tableX`, steps twice, and returns the cube's center of mass and velocity. */
std::vector<real>
RestoreCubeOnTableAndStep(mochi::Context& context, mochi::ShapeHandle box, real tableX) {
  mochi::Error error;
  mochi::Scene* const scene = context.CreateScene("cube on table");
  for (bool const isStatic : {true, false}) {
    mochi::RigidActorParams params;
    params.name = isStatic ? "table" : "cube";
    params.shape = box;
    params.isStatic = isStatic;
    params.density = 1000_r;
    params.worldFromLocal =
        mochi::TransformRT{mochi::Real3{tableX, isStatic ? -0.5_r : -0.25_r, 0_r}};
    scene->CreateRigidActor(params, error);
  }
  BankActor cube = MakeActor("cube", "rigid");
  cube.rootRotations = Rotations(1);
  cube.comTranslations = {0.6_r, -0.2502_r, 0_r};
  cube.linearVelocities = {0.3_r, -0.1_r, 0_r};
  cube.angularVelocities = {0_r, 0_r, 0.2_r};
  BankActor table = MakeActor("table", "rigid", true);
  table.rootRotations = Rotations(1);
  table.rootTranslations = {0.6_r, -0.5_r, 0_r};
  FrameBank bank;
  bank.episodes.emplace_back();
  BankEpisode& episode = bank.episodes.back();
  episode.frames = {0};
  episode.simulationTimesSec = {0.0};
  episode.stepDtsSec = {0.01};
  episode.actors = {cube, table};
  auto const restorer =
      EpisodeRestorer::Create(*scene, std::make_shared<FrameBank const>(std::move(bank)), error);
  if (restorer == nullptr) {
    ADD_FAILURE() << error.ToString();
    context.DestroyScene(scene);
    return {};
  }
  restorer->RestoreFrame(0, 0, error);
  scene->Step(0.01);
  scene->Step(0.01);
  mochi::Actor* const cubeActor = scene->GetActor(restorer->GetActors()[0].handle);
  mochi::TransformRT const com = cubeActor->GetCenterOfMassTransform(error);
  mochi::Real3 const velocity = cubeActor->GetLinearVelocity(error);
  EXPECT_TRUE(error.IsOK()) << error.ToString();
  context.DestroyScene(scene);
  return {
      com.GetTranslation()[0],
      com.GetTranslation()[1],
      com.GetTranslation()[2],
      com.GetRotation().data[0],
      com.GetRotation().data[1],
      com.GetRotation().data[2],
      com.GetRotation().data[3],
      velocity[0],
      velocity[1],
      velocity[2]};
}

std::string ReadBytes(std::filesystem::path const& path) {
  std::ifstream file{path, std::ios::binary};
  return {std::istreambuf_iterator<char>{file}, {}};
}

void WriteBytes(std::filesystem::path const& path, std::string const& bytes) {
  std::ofstream file{path, std::ios::binary | std::ios::trunc};
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

/* The schema hash a bank file's header holds, after its magic. */
uint64_t SchemaOf(std::string const& bytes) {
  uint64_t schema = 0;
  std::memcpy(&schema, bytes.data() + sizeof(uint64_t), sizeof(schema));
  return schema;
}

std::string WithSchema(std::string bytes, uint64_t schema) {
  std::memcpy(bytes.data() + sizeof(uint64_t), &schema, sizeof(schema));
  return bytes;
}

/* The schemas earlier builds wrote for the bank structs of today: the hash of their reflection
 * description before and after simple_reflection's D125296135. */
constexpr std::array<uint64_t, 2> kLegacySchemas{0x83C7'7162'05B2'E53C, 0x8C14'6B6C'489E'4C6E};

class FrameBankTest : public testing::Test {
 protected:
  /* Saves TwoEpisodes(), applies `edit` to the file's bytes, and returns the error of loading it.
   */
  std::string LoadEdited(std::function<void(std::string&)> const& edit) {
    mochi::Error error;
    SaveFrameBank(TwoEpisodes(), path, error);
    EXPECT_TRUE(error.IsOK()) << error.ToString();
    std::string bytes = ReadBytes(path);
    edit(bytes);
    WriteBytes(path, bytes);
    auto const loaded = LoadFrameBank(path, error);
    EXPECT_EQ(loaded == nullptr, !error.IsOK());
    return error.GetDescription();
  }

  TemporaryDirectory directory;
  std::filesystem::path const path = directory.Path() / "bank.superdex_framebank";
};

TEST_F(FrameBankTest, LoadsWhatWasSavedAndSavesItToTheSameBytes) {
  mochi::Error error;
  std::filesystem::path const again = directory.Path() / "again.superdex_framebank";

  SaveFrameBank(TwoEpisodes(), path, error);
  auto const loaded = LoadFrameBank(path, error);
  SaveFrameBank(*loaded, again, error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(ReadBytes(again), ReadBytes(path));
  ASSERT_EQ(loaded->episodes.size(), 2);
  BankEpisode const& second = loaded->episodes[1];
  EXPECT_EQ(second.source, "run/episode_7.rrd");
  EXPECT_EQ(second.properties[0].value, "7");
  EXPECT_EQ(second.frames, mochi::DynamicArray<int64_t>{7});
  EXPECT_EQ(second.actors[3].rootTranslations, (Values{0.6_r, -0.5_r, 0_r}));
  EXPECT_EQ(second.actors[2].displacements, TwoEpisodes().episodes[1].actors[2].displacements);
  EXPECT_TRUE(std::isnan(loaded->episodes[0].controllers[0].targets[0].values[0]));
}

// The hash of the bank structs' layout. Changing the structs changes it: update it here, and remove
// frame_bank.cpp's legacy schemas, which hold for this layout only.
TEST_F(FrameBankTest, WritesTheHashOfTheBankStructsLayout) {
  mochi::Error error;

  SaveFrameBank(TwoEpisodes(), path, error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(SchemaOf(ReadBytes(path)), uint64_t{0xE6B5'0867'CE8A'BE7D});
}

TEST_F(FrameBankTest, LoadsBanksOfTheEarlierSchemasOfTheSameLayoutToTheSameBank) {
  mochi::Error error;
  std::filesystem::path const legacy = directory.Path() / "legacy.superdex_framebank";
  std::filesystem::path const again = directory.Path() / "again.superdex_framebank";
  SaveFrameBank(TwoEpisodes(), path, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  for (uint64_t const schema : kLegacySchemas) {
    WriteBytes(legacy, WithSchema(ReadBytes(path), schema));
    auto const loaded = LoadFrameBank(legacy, error);
    ASSERT_NE(loaded, nullptr) << error.ToString();
    SaveFrameBank(*loaded, again, error);

    EXPECT_EQ(ReadBytes(again), ReadBytes(path)) << std::hex << schema;
  }
}

TEST_F(FrameBankTest, RefusesFilesThatAreNotBanks) {
  mochi::Error error;
  EXPECT_EQ(LoadFrameBank(directory.Path() / "missing", error), nullptr);
  EXPECT_NE(std::string{error.GetDescription()}.find("Failed to read"), std::string::npos);

  std::string const message = LoadEdited([](std::string& bytes) { bytes = "{\"actors\": []}"; });

  EXPECT_NE(message.find("is not a frame bank file"), std::string::npos) << message;
}

TEST_F(FrameBankTest, RefusesBanksOfAnotherPrecisionOrLayout) {
  // The eighth byte holds the bytes per real; the next eight, the bank structs' layout hash.
  std::string const precision = LoadEdited([](std::string& bytes) { bytes[7] ^= 12; });
  std::string const layout = LoadEdited([](std::string& bytes) { bytes[8] ^= 1; });

  EXPECT_NE(precision.find("another Mochi precision; re-pack"), std::string::npos) << precision;
  EXPECT_NE(layout.find("another build or an older format"), std::string::npos) << layout;
}

TEST_F(FrameBankTest, RefusesTruncatedFilesAndTrailingBytes) {
  std::string const truncated = LoadEdited([](std::string& bytes) { bytes.pop_back(); });
  std::string const trailing = LoadEdited([](std::string& bytes) { bytes.push_back('\0'); });

  EXPECT_NE(truncated.find("truncated, corrupt or empty"), std::string::npos) << truncated;
  EXPECT_NE(trailing.find("truncated, corrupt or empty"), std::string::npos) << trailing;
}

constexpr std::array<real, 24> kBox = {
    -0.125_r, -0.125_r, -0.125_r, 0.125_r,  -0.125_r, -0.125_r, -0.125_r, 0.125_r,
    -0.125_r, 0.125_r,  0.125_r,  -0.125_r, -0.125_r, -0.125_r, 0.125_r,  0.125_r,
    -0.125_r, 0.125_r,  -0.125_r, 0.125_r,  0.125_r,  0.125_r,  0.125_r,  0.125_r,
};
constexpr std::array<int, 36> kBoxConnectivity = {
    0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 5, 0, 5, 4,
    2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5,
};
constexpr std::array<real, 12> kTetCoordinates =
    {0_r, 0_r, 0_r, 0.25_r, 0_r, 0_r, 0_r, 0.25_r, 0_r, 0_r, 0_r, 0.25_r};
constexpr std::array<int, 4> kTetConnectivity = {0, 1, 2, 3};

/* Scenes for TwoEpisodes(): its actors bind by name, so a test renames or leaves out scene actors
 * to change what binds. */
class FrameBankSceneTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NE(context, nullptr);
    scene = context->CreateScene("frame bank test");
    box = context->CreateTriMeshShape(kBox, kBoxConnectivity, error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
  }

  void TearDown() override {
    context->DestroyScene(scene);
  }

  mochi::Actor* AddBox(char const* name, bool isStatic = false) {
    mochi::RigidActorParams params;
    params.name = name;
    params.shape = box;
    params.isStatic = isStatic;
    params.density = 1000_r;
    return scene->CreateRigidActor(params, error);
  }

  mochi::Actor* AddDuck(char const* name) {
    mochi::SoftActorParams params;
    params.name = name;
    params.shape = context->CreateTetMeshShape(kTetCoordinates, kTetConnectivity, error);
    return scene->CreateSoftActor(params, error);
  }

  /* A fixed base and `numJoints` - 1 revolute joints, with a pose controller. */
  mochi::Actor* AddArm(char const* name, int numJoints) {
    mochi::ArticulatedActorParams params;
    params.name = name;
    for (int i = 0; i < numJoints; ++i) {
      params.joints.push_back({
          .type =
              i == 0 ? mochi::ArticulatedJointType::Hard : mochi::ArticulatedJointType::Revolute,
          .parentLinkFromJoint = mochi::TransformRT{mochi::Real3{i == 0 ? 0_r : 0.5_r, 0_r, 0_r}},
          .axis = {0_r, 0_r, 1_r},
      });
      params.links.push_back({
          .parentLink = i - 1,
          .shape = box,
          .colliderType = mochi::ColliderType::Box,
          .density = 500_r,
      });
    }
    mochi::Actor* const actor = scene->CreateArticulatedActor(params, error);
    if (actor != nullptr) {
      actor->AddArticulatedPoseController(mochi::PoseControllerParams{numJoints}, error);
    }
    return actor;
  }

  /* The scene actors that TwoEpisodes() binds to. */
  void AddActorsOfTwoEpisodes() {
    arm = AddArm("robot/arm", 3);
    cube = AddBox("cube");
    duck = AddDuck("duck");
    table = AddBox("table", true);
  }

  std::string CreateError(FrameBank bank) {
    mochi::Error createError;
    auto const restorer = EpisodeRestorer::Create(
        *scene, std::make_shared<FrameBank const>(std::move(bank)), createError);
    EXPECT_EQ(restorer == nullptr, !createError.IsOK());
    return createError.GetDescription();
  }

  std::unique_ptr<mochi::Context, decltype(&mochi::DestroyContext)> context{
      mochi::CreateContext(0),
      &mochi::DestroyContext};
  mochi::Scene* scene = nullptr;
  mochi::ShapeHandle box;
  mochi::Error error;
  mochi::Actor* arm = nullptr;
  mochi::Actor* cube = nullptr;
  mochi::Actor* duck = nullptr;
  mochi::Actor* table = nullptr;
};

TEST_F(FrameBankSceneTest, BindsEachActorToItsSceneActorOnce) {
  AddActorsOfTwoEpisodes();

  auto const restorer =
      EpisodeRestorer::Create(*scene, std::make_shared<FrameBank const>(TwoEpisodes()), error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  std::vector<mochi::ActorHandle> handles;
  for (RecordedActor const& actor : restorer->GetActors()) {
    handles.push_back(actor.handle);
  }
  EXPECT_EQ(
      handles,
      (std::vector<mochi::ActorHandle>{
          arm->GetHandle(), cube->GetHandle(), duck->GetHandle(), table->GetHandle()}));
  EXPECT_EQ(restorer->GetActors()[0].type, mochi::ActorType::Articulated);
  EXPECT_TRUE(restorer->GetActors()[3].isStatic);
  EXPECT_EQ(restorer->GetFrames(), (std::vector<int64_t>{0, 1}));
}

TEST_F(FrameBankSceneTest, RefusesScenesThatDoNotFit) {
  struct Case {
    char const* expected;
    std::function<void()> addActors;
  };
  auto const addOthers = [this] {
    AddBox("cube");
    AddDuck("duck");
    AddBox("table", true);
  };
  std::vector<Case> const cases = {
      {"missing recorded actor 'robot/arm'", addOthers},
      {"degrees of freedom",
       [&] {
         AddArm("robot/arm", 4);
         addOthers();
       }},
      {"'cube' was recorded as rigid",
       [this] {
         AddArm("robot/arm", 3);
         AddDuck("cube");
         AddDuck("duck");
         AddBox("table", true);
       }},
      {"unique actor names: 'wall'",
       [this] {
         AddActorsOfTwoEpisodes();
         AddBox("wall", true);
         AddBox("wall", true);
       }},
  };
  for (Case const& testCase : cases) {
    TearDown();
    SetUp();
    testCase.addActors();
    ASSERT_TRUE(error.IsOK()) << error.ToString();

    std::string const message = CreateError(TwoEpisodes());

    EXPECT_NE(message.find(testCase.expected), std::string::npos) << message;
  }
}

TEST_F(FrameBankSceneTest, RefusesEpisodesThatListOtherActors) {
  AddActorsOfTwoEpisodes();
  FrameBank bank = TwoEpisodes();
  bank.episodes[1].actors[1].runtimeName = "ball";

  std::string const message = CreateError(bank);

  EXPECT_NE(message.find("episode 1 lists other actors"), std::string::npos) << message;
}

TEST_F(FrameBankSceneTest, RestoresAnyFrameOfAnyEpisodeAndLeavesOtherActors) {
  AddActorsOfTwoEpisodes();
  mochi::Actor* const pebble = AddBox("pebble");
  pebble->SetVelocity(mochi::Real3{0_r, 0_r, 1_r}, mochi::Real3{0_r, 0_r, 0_r}, error);
  FrameBank const bank = TwoEpisodes();
  auto const restorer =
      EpisodeRestorer::Create(*scene, std::make_shared<FrameBank const>(bank), error);

  restorer->RestoreFrame(1, 0, error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  BankEpisode const& episode = bank.episodes[1];
  EXPECT_DOUBLE_EQ(scene->GetTotalSimulationTime(), 0.07);
  std::array<real, 2> joints{};
  arm->GetArticulatedPose(mochi::Span<real>{joints.data(), joints.size()}, error);
  EXPECT_EQ(joints[0], episode.actors[0].jointPositions[0]);
  EXPECT_EQ(joints[1], episode.actors[0].jointPositions[1]);
  auto const displacements = duck->GetDisplacements(error);
  ASSERT_EQ(displacements.size(), 12);
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_EQ(displacements[i], episode.actors[2].displacements[i]);
  }
  EXPECT_EQ(cube->GetLinearVelocity(error)[2], episode.actors[1].linearVelocities[2]);
  EXPECT_EQ(pebble->GetLinearVelocity(error)[2], 1_r);
  restorer->RestoreFrame(1, 1, error);
  EXPECT_FALSE(error.IsOK());
}

TEST_F(FrameBankSceneTest, PlacesStaticActorsAtTheirEpisodesPoses) {
  AddActorsOfTwoEpisodes();
  auto const restorer =
      EpisodeRestorer::Create(*scene, std::make_shared<FrameBank const>(TwoEpisodes()), error);

  for (auto const [episode, x] : {std::pair{1, 0.6}, std::pair{0, -0.4}}) {
    restorer->RestoreFrame(episode, 0, error);
    mochi::Real3 const translation = table->GetRootTransform().GetTranslation();
    EXPECT_NEAR(translation[0], x, 1e-6) << "episode " << episode;
    EXPECT_NEAR(translation[1], -0.5, 1e-6) << "episode " << episode;
  }
  EXPECT_TRUE(error.IsOK()) << error.ToString();
}

TEST_F(FrameBankSceneTest, RestoresBanksOfTheEarlierSchemasBitwise) {
  AddActorsOfTwoEpisodes();
  TemporaryDirectory const directory;
  std::filesystem::path const current = directory.Path() / "current.superdex_framebank";
  std::filesystem::path const legacy = directory.Path() / "legacy.superdex_framebank";
  SaveFrameBank(TwoEpisodes(), current, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  // Loads `path`, restores episode 0's second frame and returns every restored value.
  auto const restore = [&](std::filesystem::path const& path) {
    std::vector<real> state;
    auto const bank = LoadFrameBank(path, error);
    auto const restorer = bank != nullptr ? EpisodeRestorer::Create(*scene, bank, error) : nullptr;
    if (restorer == nullptr) {
      ADD_FAILURE() << error.ToString();
      return state;
    }
    restorer->RestoreFrame(0, 1, error);
    std::array<real, 2> joints{};
    arm->GetArticulatedPose(mochi::Span<real>{joints.data(), joints.size()}, error);
    state.insert(state.end(), joints.begin(), joints.end());
    for (real const value : duck->GetDisplacements(error)) {
      state.push_back(value);
    }
    mochi::TransformRT const com = cube->GetCenterOfMassTransform(error);
    for (mochi::Real3 const& v :
         {com.GetTranslation(),
          cube->GetLinearVelocity(error),
          cube->GetAngularVelocity(error),
          table->GetRootTransform().GetTranslation()}) {
      state.insert(state.end(), {v[0], v[1], v[2]});
    }
    for (int i = 0; i < 4; ++i) {
      state.push_back(com.GetRotation().data[i]);
    }
    EXPECT_TRUE(error.IsOK()) << error.ToString();
    return state;
  };

  std::vector<real> const expected = restore(current);

  ASSERT_FALSE(expected.empty());
  for (uint64_t const schema : kLegacySchemas) {
    WriteBytes(legacy, WithSchema(ReadBytes(current), schema));
    EXPECT_EQ(restore(legacy), expected) << std::hex << schema;
  }
}

TEST_F(FrameBankSceneTest, StepsOnAMovedStaticActorAsIfItHadBeenThere) {
  EXPECT_EQ(
      RestoreCubeOnTableAndStep(*context, box, 0.6_r),
      RestoreCubeOnTableAndStep(*context, box, -0.4_r));
}

} // namespace
} // namespace superdex::recorder
