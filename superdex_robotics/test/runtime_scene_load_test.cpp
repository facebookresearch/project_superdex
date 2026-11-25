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

#include "mochi_bots_test_helpers.h"

#include <mochi_core/utils/defer.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/mochi_physics.h>
#include <superdex_robotics/controllers/controller_basic_jsc_pd.h>
#include <superdex_robotics/controllers/controller_basic_osc_pd.h>
#include <superdex_robotics/controllers/controller_mochi_articulated_pose.h>
#include <superdex_robotics/scene/bot_scene.h>
#include <superdex_robotics/scene/bot_task.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <filesystem>
#include <set>
#include <string>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

std::string ToStd(DynamicString const& value) {
  return std::string(value.c_str(), value.size());
}

std::string GetSceneDir() {
  return GetAssetsDir() + "scenes/runtime_scene_load/";
}

} // namespace

// The runtime_scene_load scene mirrors SuperDex Unreal's L_Sampler level: each slot offers two
// arms, driven through their `<bot>/osc` arm and `<bot>/jsc` hand controllers, and four proxy
// hands, driven by their `<bot>/pose` controller. Every bot starts in its initial pose, without
// gravity.
TEST(RuntimeSceneLoadTest, BotsLoadWithTheControllersTeleopDrives) {
  Context* context = CreateContext(0);
  ASSERT_NE(context, nullptr);
  MOCHI_DEFER(DestroyContext(context));
  RoboticsContext* botsCtx = CreateRoboticsContext();
  ASSERT_NE(botsCtx, nullptr);
  MOCHI_DEFER(DestroyRoboticsContext(botsCtx));

  BotScene botScene = LoadBotScene(
      GetSceneDir() + "runtime_scene_load.superdex_scene", context, botsCtx, ExpectOK{});
  auto const& bots = botScene.GetBotScenePrefab().bots;
  ASSERT_EQ(bots.size(), 12);
  for (auto const& entry : bots) {
    std::string const name = ToStd(entry.name);
    SCOPED_TRACE(name);
    EXPECT_FALSE(entry.hasGravity);
    Bot* bot = botScene.GetBot(name);
    ASSERT_NE(bot, nullptr);
    Actor* actor = bot->GetArticulatedActor();
    ASSERT_NE(actor, nullptr);

    auto controllerType = [&](std::string const& controller) -> std::string_view {
      ControllerBase const* base =
          botsCtx->GetController(botScene.GetControllerHandle(name + "/" + controller));
      return base ? base->GetTypeName() : std::string_view{};
    };
    if (entry.controllers.size() == 1) {
      EXPECT_EQ(controllerType("pose"), ControllerMochiArticulatedPose::TypeName());
    } else {
      EXPECT_EQ(controllerType("osc"), ControllerBasicOscPd::TypeName());
      EXPECT_EQ(controllerType("jsc"), ControllerBasicJscPd::TypeName());
    }

    // The scene's initial pose, or the bot's default pose, which the loader puts in defaultPose.
    // A proxy hand's free root adds actor DOFs that the bot-space pose does not list.
    BotPrefab const& prefab = bot->GetBotPrefab();
    DynamicArray<real> const expected = BuildArticulatedPoseFromBotPose(
        prefab, MakeConstSpan(prefab.defaultPose), actor->GetNumDofs(), ExpectOK{});
    DynamicArray<real> pose(actor->GetNumDofs());
    actor->GetArticulatedPose(pose, ExpectOK{});
    ASSERT_EQ(pose.size(), expected.size());
    for (int i = 0; i < isize(pose); ++i) {
      EXPECT_NEAR(pose[i], expected[i], 1e-4_r) << "DOF " << i;
    }

    // The loader clamps a bot's default pose to its joint limits, but not a scene's initial pose.
    BotPrefab clamped = prefab;
    RebuildBotData(clamped, ExpectOK{});
    ASSERT_EQ(clamped.defaultPose.size(), prefab.defaultPose.size());
    for (int i = 0; i < isize(prefab.defaultPose); ++i) {
      EXPECT_NEAR(prefab.defaultPose[i], clamped.defaultPose[i], 1e-4_r) << "DOF " << i;
    }
  }
}

// Each spawnable prefab has one task in tasks/, which places it, and it spawns.
TEST(RuntimeSceneLoadTest, EveryTaskSpawnsASpawnablePrefab) {
  Context* context = CreateContext(0);
  ASSERT_NE(context, nullptr);
  MOCHI_DEFER(DestroyContext(context));
  RoboticsContext* botsCtx = CreateRoboticsContext();
  ASSERT_NE(botsCtx, nullptr);
  MOCHI_DEFER(DestroyRoboticsContext(botsCtx));

  BotScene botScene = LoadBotScene(
      GetSceneDir() + "runtime_scene_load.superdex_scene", context, botsCtx, ExpectOK{});
  auto const& scene = botScene.GetBotScenePrefab().scene;

  std::set<std::string> taskPrefabs;
  for (auto const& file : std::filesystem::directory_iterator(GetSceneDir() + "tasks")) {
    if (!IsBotTaskPath(file.path().string())) {
      continue;
    }
    SCOPED_TRACE(file.path().filename().string());
    BotTaskPrefab const task = LoadBotTaskPrefabFromFile(file.path().string(), ExpectOK{});
    auto const spawns = ResolveResetState(task, {}, ComputeWorldFromTask(scene, task), ExpectOK{});
    ASSERT_EQ(spawns.size(), 1);
    auto const& spawn = spawns[0];
    EXPECT_TRUE(taskPrefabs.insert(ToStd(spawn.prefabName)).second) << "a second task places it";
    auto const actors = botScene.SpawnPrefabInstance(
        ToStd(spawn.name), ToStd(spawn.prefabName), spawn.worldFromSpawn, "", ExpectOK{});
    EXPECT_FALSE(actors.empty());
    botScene.DestroyAllPrefabInstances();
  }
  std::set<std::string> spawnablePrefabs;
  for (auto const& prefab : scene.spawnablePrefabs) {
    spawnablePrefabs.insert(ToStd(prefab.name));
  }
  EXPECT_EQ(taskPrefabs, spawnablePrefabs);
}
