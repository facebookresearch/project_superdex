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

#include "tuner_test_fixture.h"

#include <superdex_asset_tuner/tuner_session.h>

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/quaternion.h>
#include <picojson/picojson.h>
#include <superdex_robotics/scene/bot_scene.h>
#include <superdex_robotics/scene/bot_task.h>

#include <fstream>
#include <string>

namespace superdex::tuner::test {
namespace {

constexpr char kJscParams[] = R"({"Kp": [200, 200, 200, 200, 200, 200, 200],
  "Kd": [20, 20, 20, 20, 20, 20, 20], "saturation": [40, 40, 40, 40, 40, 40, 40],
  "deadband": [0, 0, 0, 0, 0, 0, 0]})";
constexpr char kOscParams[] = R"({"Kp_p": 900, "Kd_p": 75, "Kp_r": 30, "Kd_r": 3,
  "maxTranslationError": 0.05, "maxRotationError": 0.4})";
constexpr char kJsc[] =
    R"([{"type": "BASIC_JSC_PD", "name": "jsc", "params": "jsc.superdex_controller"}])";
constexpr char kOsc[] = R"([{"type": "BASIC_OSC_PD", "name": "osc",
    "initArgs": {"baseLinkName": "fr3_link0", "eeLinkName": "fr3_link8"},
    "params": "osc.superdex_controller"}])";

// An FR3 arm with `controllers` (a scene's JSON array), with gravity off and no prefab.
class TunerBotTest : public TunerTest {
 protected:
  [[nodiscard]] TunerSpec MakeBotSpec(char const* controllers) const {
    TunerSpec spec = MakeSpec(kRigidCubePrefab);
    spec.prefab = "";
    // Bot scenes name bots through a root marker's tags, never by absolute path.
    std::filesystem::path const bots = std::filesystem::path(std::string(spec.assetsRoot)) / "bots";
    picojson::value const root{picojson::object{{"@bots", picojson::value(bots.string())}}};
    std::ofstream(Dir() / ".superdex_root") << root.serialize();
    std::ofstream(Dir() / "jsc.superdex_controller") << kJscParams;
    std::ofstream(Dir() / "osc.superdex_controller") << kOscParams;
    std::ofstream(Dir() / "arm.superdex_scene")
        << R"({"bots": [{"name": "arm", "path": "@bots/arms/fr3/fr3.superdex_bot", "controllers": )"
        << controllers << "}]}";
    spec.scene = "./arm.superdex_scene";
    spec.run.duration = 2.0;
    return spec;
  }

  // The FR3 scene with a task that spawns two cubes 0.3 m apart, and the task's other `fields`
  // (its criteria over the cubes). Gravity is off, so the cubes stay where they spawn.
  [[nodiscard]] TunerSpec MakeTaskSpec(std::string const& fields) const {
    TunerSpec spec = MakeBotSpec(kJsc);
    std::filesystem::path const mesh =
        std::filesystem::path(std::string(spec.assetsRoot)) / "cube/cube_fine_mesh.mochi.h5";
    std::ofstream(Dir() / "cube.mochi_scene")
        << R"({"actors": {"rigid": [{"name": "cube", "shape": )"
        << picojson::value(mesh.string()).serialize()
        << R"(, "scale": [0.1, 0.1, 0.1], "colliderType": "Box", "density": 1000}]}})";
    std::ofstream(Dir() / "arm.superdex_scene")
        << R"({"scene": {"spawnablePrefabs": [{"name": "cube", "path": "./cube.mochi_scene"}]},
              "bots": [{"name": "arm", "path": "@bots/arms/fr3/fr3.superdex_bot",
                        "controllers": )"
        << kJsc << "}]}";
    std::ofstream(Dir() / "cubes.superdex_task")
        << R"({"resetState": [{"name": "a", "prefabName": "cube"},
              {"name": "b", "prefabName": "cube",
               "parentFromSpawn": {"rotation": [0, 0, 0, 1], "translation": [0.3, 0, 0]}}], )"
        << fields << "}";
    spec.task = "./cubes.superdex_task";
    spec.run.duration = 1.0;
    return spec;
  }
};

constexpr char kCubesNear[] = R"({"criteria": [{"SphericalProximity": {"taskObjectA": "a",
    "taskObjectB": "b", "comparator": "LessThan", "thresholdM": 0.5}}], "validationTimeS": 0.2})";
constexpr char kCubesTouch[] = R"({"criteria": [{"SphericalProximity": {"taskObjectA": "a",
    "taskObjectB": "b", "comparator": "LessThan", "thresholdM": 0.1}}]})";

LoadSpec MakeInput(std::string_view target) {
  LoadSpec load;
  load.name = "drive";
  load.type = LoadType::ControllerInput;
  load.target = DynamicString(target);
  return load;
}

TEST_F(TunerBotTest, DescribeListsTheBotsControllersWithTheirStartingInput) {
  SceneTargets const targets = Describe(MakeBotSpec(kJsc));

  ASSERT_EQ(targets.controllers.size(), 1u);
  ControllerTarget const& jsc = targets.controllers[0];
  EXPECT_EQ(jsc.name, "arm/jsc");
  EXPECT_EQ(jsc.type, "BASIC_JSC_PD");
  EXPECT_EQ(jsc.inputKind, "JointPositions");
  EXPECT_EQ(jsc.input.size(), 7u);
  // Each entry's joint and its limits, from the FR3's bot file.
  ASSERT_EQ(jsc.inputJoints.size(), 7u);
  EXPECT_EQ(jsc.inputJoints[3].name, "fr3_joint4");
  EXPECT_EQ(jsc.inputJoints[3].type, mochi::ArticulatedJointType::Revolute);
  EXPECT_NEAR(jsc.inputJoints[3].min, -3.0421f, 1e-3f);
  EXPECT_NEAR(jsc.inputJoints[3].max, -0.1518f, 1e-3f);
}

TEST_F(TunerBotTest, DescribeGivesAPoseInputsFrameAsTheBotsRoot) {
  TunerSpec spec = MakeBotSpec(kOsc);
  std::ofstream(Dir() / "arm.superdex_scene") << R"({"bots": [{"name": "arm",
      "path": "@bots/arms/fr3/fr3.superdex_bot", "parentFromBot": {"translation": [0.5, 0, 0]},
      "controllers": )" << kOsc << "}]}";

  ControllerTarget const osc = Describe(spec).controllers[0];

  EXPECT_EQ(osc.inputKind, "Pose");
  EXPECT_TRUE(osc.inputJoints.empty());
  EXPECT_NEAR(osc.rootPosition[0], 0.5f, 1e-6f);
  EXPECT_NEAR(osc.rootPosition[1], 0.0f, 1e-6f);
}

TEST_F(TunerBotTest, ControllerInputMovesItsJointsAndTheOthersHoldTheirStart) {
  TunerSpec spec = MakeBotSpec(kJsc);
  real const start = Describe(spec).controllers[0].input[3];
  LoadSpec load = MakeInput("arm/jsc");
  load.joints = {"fr3_joint4"};
  load.inputKeys = {{1.0, {start + 0.3f}, Interpolation::Linear}};
  spec.loads.push_back(load);
  spec.probes.push_back(MakeProbe(ProbeKind::JointState, "fr3/fr3_joint4"));
  spec.probes.push_back(MakeProbe(ProbeKind::JointState, "fr3/fr3_joint2"));
  real const joint2 = Describe(spec).controllers[0].input[1];

  RunCapture const run = Run(spec);

  EXPECT_EQ(run.summary.status, "ok");
  EXPECT_NEAR(run.Final("fr3/fr3_joint4.q0"), start + 0.3f, 0.01f);
  EXPECT_NEAR(run.Final("fr3/fr3_joint2.q0"), joint2, 0.01f);
  EXPECT_LT(run.Final("load.drive.error"), 0.01f);
  // Each sample carries the controller's input as it reports it: here, the joint positions.
  ASSERT_EQ(run.samples.back().loads[0].input.size(), 7u);
  EXPECT_NEAR(run.samples.back().loads[0].input[3], run.Final("fr3/fr3_joint4.q0"), 1e-5f);
}

TEST_F(TunerBotTest, AControllerInputEditedMidRunMovesItsJoint) {
  TunerSpec spec = MakeBotSpec(kJsc);
  real const start = Describe(spec).controllers[0].input[3];
  LoadSpec load = MakeInput("arm/jsc");
  load.joints = {"fr3_joint4"};
  load.inputKeys = {{0.1, {start}, Interpolation::Linear}};
  spec.loads.push_back(load);
  // A disabled load that does not resolve does not block edits to the others.
  LoadSpec unused = MakeInput("arm/none");
  unused.name = "unused";
  unused.enabled = false;
  spec.loads.push_back(unused);
  spec.probes.push_back(MakeProbe(ProbeKind::JointState, "fr3/fr3_joint4"));
  // Built and bound as a run does, and stepped here so the load can be edited between steps, as
  // Studio does.
  mochi::Context* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  mochi::Scene* scene = context->CreateScene("edit");
  MOCHI_DEFER(context->DestroyScene(scene));
  TunerBots bots;
  std::optional<real> groundHeight;
  DynamicArray<mochi::Actor*> prefabActors;
  std::string error;
  ASSERT_TRUE(
      BuildTunerScene(spec, TunerPath(), context, *scene, bots, groundHeight, prefabActors, error))
      << error;
  TunerSession session;
  ASSERT_TRUE(session.Bind(*scene, spec, error, bots.Get())) << error;
  MOCHI_DEFER(session.Unbind(*scene));
  Sample sample;
  mochi::CallbackHandle const loads = scene->RegisterPreStepCallback(
      "edit::Loads", [&](mochi::StepInfo const& info) { session.OnPreStep(info); });
  MOCHI_DEFER(scene->CancelCallback(loads));
  mochi::CallbackHandle const probes = scene->RegisterPostStepCallback(
      "edit::Probes",
      [&](mochi::StepInfo const& info) { session.OnPostStep(info, sample); },
      kProbesCallbackPriority);
  MOCHI_DEFER(scene->CancelCallback(probes));
  auto const runFor = [&](double seconds) {
    for (int i = 0; i < static_cast<int>(std::llround(seconds / spec.run.dt)); ++i) {
      scene->Step(spec.run.dt);
    }
  };
  auto const names = session.GetChannelNames();
  auto const joint4 = static_cast<size_t>(
      std::find(names.begin(), names.end(), "fr3/fr3_joint4.q0") - names.begin());
  ASSERT_LT(joint4, names.size());

  runFor(0.5);
  spec.loads[0].inputKeys[0].values[0] = start + 0.3f;
  ASSERT_TRUE(session.UpdateLoads(mochi::MakeConstSpan(spec.loads), error)) << error;
  runFor(1.0);

  EXPECT_NEAR(sample.values[joint4], start + 0.3f, 0.01f);
}

TEST_F(TunerBotTest, EnablingASecondDriverOfAControllerMidRunIsRefused) {
  TunerSpec spec = MakeBotSpec(kJsc);
  LoadSpec load = MakeInput("arm/jsc");
  load.inputKeys = {{1.0, Describe(spec).controllers[0].input, Interpolation::Linear}};
  spec.loads = {load, load};
  spec.loads[1].name = "again";
  spec.loads[1].enabled = false;
  mochi::Context* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  mochi::Scene* scene = context->CreateScene("edit");
  MOCHI_DEFER(context->DestroyScene(scene));
  TunerBots bots;
  std::optional<real> groundHeight;
  DynamicArray<mochi::Actor*> prefabActors;
  std::string error;
  ASSERT_TRUE(
      BuildTunerScene(spec, TunerPath(), context, *scene, bots, groundHeight, prefabActors, error))
      << error;
  TunerSession session;
  ASSERT_TRUE(session.Bind(*scene, spec, error, bots.Get())) << error;
  MOCHI_DEFER(session.Unbind(*scene));

  spec.loads[1].enabled = true;

  EXPECT_FALSE(session.UpdateLoads(mochi::MakeConstSpan(spec.loads), error));
  EXPECT_NE(error.find("both drive 'arm/jsc'"), std::string::npos) << error;
}

TEST_F(TunerBotTest, ControllerInputDrivesAToolPoseAlongWaypoints) {
  TunerSpec spec = MakeBotSpec(kOsc);
  DynamicArray<real> const tool = Describe(spec).controllers[0].input;
  PoseWaypoint goal;
  goal.time = 1.0;
  goal.position = Real3{tool[0] + 0.05f, tool[1], tool[2]};
  goal.rotation = mochi::Quaternion(tool[3], tool[4], tool[5], tool[6]).ToRotationVector();
  LoadSpec load = MakeInput("arm/osc");
  load.waypoints = {goal};
  spec.loads.push_back(load);

  RunCapture const run = Run(spec);

  EXPECT_LT(run.Value("load.drive.error", 0), 0.001f);
  EXPECT_LT(run.Final("load.drive.error"), 0.005f);
  EXPECT_LT(run.Final("load.drive.angleError"), 0.02f);
}

TEST_F(TunerBotTest, TwoLoadsCannotDriveOneController) {
  TunerSpec spec = MakeBotSpec(kJsc);
  LoadSpec load = MakeInput("arm/jsc");
  load.inputKeys = {{1.0, Describe(spec).controllers[0].input, Interpolation::Linear}};
  spec.loads = {load, load};
  spec.loads[1].name = "again";
  RunSummary summary;
  std::string error;

  EXPECT_FALSE(RunTuner(spec, TunerPath(), {}, summary, error));

  EXPECT_NE(error.find("both drive 'arm/jsc'"), std::string::npos) << error;
}

TEST_F(TunerBotTest, AJointListedTwiceIsRefused) {
  TunerSpec spec = MakeBotSpec(kJsc);
  LoadSpec load = MakeInput("arm/jsc");
  load.joints = {"fr3_joint4", "fr3_joint4"};
  load.inputKeys = {{1.0, {0.0f, 0.0f}, Interpolation::Linear}};
  spec.loads.push_back(load);
  RunSummary summary;
  std::string error;

  EXPECT_FALSE(RunTuner(spec, TunerPath(), {}, summary, error));

  EXPECT_NE(error.find("lists joint 'fr3_joint4' more than once"), std::string::npos) << error;
}

TEST_F(TunerBotTest, APrefabAndASceneTogetherAreRefused) {
  TunerSpec spec = MakeBotSpec(kJsc);
  spec.prefab = DynamicString(kRigidCubePrefab);
  RunSummary summary;
  std::string error;

  EXPECT_FALSE(RunTuner(spec, TunerPath(), {}, summary, error));

  EXPECT_NE(error.find("a prefab or a scene"), std::string::npos) << error;
}

TEST_F(TunerBotTest, ATasksSuccessEndsTheRunOnceItHoldsForItsValidationTime) {
  TunerSpec spec =
      MakeTaskSpec(std::string(R"("terminationCriteria": {"success": )") + kCubesNear + "}");
  // A task's objects are spawned as named instances, which probes can target.
  spec.probes.push_back(MakeProbe(ProbeKind::BodyPose, "b/cube"));

  RunCapture const run = Run(spec);

  // Held from the first step (0.01 s), the success is reported on the step that has held it for
  // 0.2 s (0.21 s, or 0.22 s as the summed time steps round), and the run ends on that step.
  EXPECT_EQ(run.summary.status, "succeeded") << run.summary.message;
  EXPECT_GE(run.summary.simTime, 0.21 - 1e-6);
  EXPECT_LE(run.summary.simTime, 0.22 + 1e-6);
  EXPECT_NEAR(run.Final("b/cube.pos.x"), 0.3f, 1e-4f);
}

TEST_F(TunerBotTest, EachSampleCarriesTheTaskOutcomeOfItsOwnStep) {
  TunerSpec const spec =
      MakeTaskSpec(std::string(R"("terminationCriteria": {"success": )") + kCubesNear + "}");
  mochi::Context* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  mochi::Scene* scene = context->CreateScene("task");
  MOCHI_DEFER(context->DestroyScene(scene));
  TunerBots bots;
  std::optional<real> groundHeight;
  DynamicArray<mochi::Actor*> prefabActors;
  std::string error;
  ASSERT_TRUE(
      BuildTunerScene(spec, TunerPath(), context, *scene, bots, groundHeight, prefabActors, error))
      << error;
  TunerSession session;
  ASSERT_TRUE(session.Bind(*scene, spec, error, bots.Get())) << error;
  MOCHI_DEFER(session.Unbind(*scene));
  Sample sample;
  // Registered as a run registers it, before the bot scene's own termination callback.
  mochi::CallbackHandle const probes = scene->RegisterPostStepCallback(
      "task::Probes",
      [&](mochi::StepInfo const& info) { session.OnPostStep(info, sample); },
      kProbesCallbackPriority);
  MOCHI_DEFER(scene->CancelCallback(probes));

  bool succeeded = false;
  for (int step = 0; step < 40; ++step) {
    scene->Step(spec.run.dt);
    succeeded = bots.Get()->GetTerminationOutcome().success;
    EXPECT_EQ(sample.task == TaskOutcome::Success, succeeded) << "step " << step;
  }
  EXPECT_TRUE(succeeded);
}

TEST_F(TunerBotTest, PrefabOverridesChangeTheTasksObjects) {
  TunerSpec spec = MakeTaskSpec(R"("terminationCriteria": {})");
  PrefabOverride density;
  density.path = "actors.rigid[name=cube].density";
  density.value = "500";
  spec.prefabOverrides.push_back(density);

  SceneTargets const targets = Describe(spec);

  auto const cube = std::find_if(targets.rigid.begin(), targets.rigid.end(), [](auto const& body) {
    return body.name == "a/cube";
  });
  ASSERT_NE(cube, targets.rigid.end());
  EXPECT_NEAR(cube->mass, 0.5f, 1e-4f); // a 0.1 m cube at 500 kg/m^3
}

TEST_F(TunerBotTest, ATasksFailureFailsTheRun) {
  TunerSpec const spec =
      MakeTaskSpec(std::string(R"("terminationCriteria": {"failure": )") + kCubesNear + "}");

  RunCapture const run = Run(spec);

  EXPECT_EQ(run.summary.status, "failed") << run.summary.message;
  EXPECT_GE(run.summary.simTime, 0.21 - 1e-6);
  EXPECT_LE(run.summary.simTime, 0.22 + 1e-6);
}

TEST_F(TunerBotTest, AStopConditionThatEndsTheRunBeforeTheTaskSucceedsFailsIt) {
  TunerSpec spec =
      MakeTaskSpec(std::string(R"("terminationCriteria": {"success": )") + kCubesTouch + "}");
  spec.probes.push_back(MakeProbe(ProbeKind::BodyPose, "b/cube"));
  StopCondition stop;
  stop.name = "early";
  stop.kind = StopKind::ChannelThreshold;
  stop.channel = "b/cube.pos.x";
  stop.comparison = StopComparison::Above;
  stop.threshold = 0.2f; // true from the first step
  spec.stopConditions.push_back(stop);

  RunCapture const run = Run(spec);

  EXPECT_EQ(run.summary.status, "failed");
  EXPECT_NE(std::string_view(run.summary.message).find("'early'"), std::string_view::npos)
      << run.summary.message;
}

TEST_F(TunerBotTest, ATaskThatDoesNotSucceedByItsTimeLimitFailsTheRun) {
  TunerSpec const spec = MakeTaskSpec(
      std::string(R"("terminationCriteria": {"success": )") + kCubesTouch +
      R"(}, "timeLimitS": 0.5)");

  RunCapture const run = Run(spec);

  EXPECT_EQ(run.summary.status, "failed");
  EXPECT_NEAR(run.summary.simTime, 0.5, 1e-6);
  EXPECT_NE(std::string_view(run.summary.message).find("did not succeed"), std::string_view::npos)
      << run.summary.message;
}

TEST_F(TunerBotTest, AnUnknownControllerListsTheBotsControllers) {
  TunerSpec spec = MakeBotSpec(kJsc);
  LoadSpec load = MakeInput("arm/osc");
  load.inputKeys = {{1.0, {0.0f}, Interpolation::Linear}};
  spec.loads.push_back(load);
  RunSummary summary;
  std::string error;

  EXPECT_FALSE(RunTuner(spec, TunerPath(), {}, summary, error));

  EXPECT_NE(error.find("Available: [arm/jsc]"), std::string::npos) << error;
}

} // namespace
} // namespace superdex::tuner::test
