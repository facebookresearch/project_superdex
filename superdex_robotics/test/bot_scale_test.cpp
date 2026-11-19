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

/* Tests for the experimental uniform BotPrefab::scale. */

#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_physics/mochi_physics.h>
#include <superdex_robotics/controllers/controller_mochi_articulated_pose.h>
#include <superdex_robotics/core/context.h>
#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/sensors/sensor_base.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <cmath>
#include <filesystem>
#include <limits>
#include <string>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

constexpr real kScale = 0.8_r;
constexpr char kHandPath[] = "bots/hands/oculus_xr/right/oculus_xr_hand_lowpoly_right.superdex_bot";

BotPrefab LoadHand() {
  return LoadBotPrefabFromFile(GetAssetPath(kHandPath), ExpectOK{});
}

BotPrefab LoadFR3() {
  return LoadBotPrefabFromFile(GetAssetPath("bots/arms/fr3/fr3.superdex_bot"), ExpectOK{});
}

/* FR3 with its last revolute joint turned into a limited prismatic joint, a default pose on it, and
 * nonzero joint dynamics everywhere, so every scaled quantity is exercised. */
BotPrefab LoadFR3WithPrismatic(int& outPrismaticJoint) {
  BotPrefab bot = LoadFR3();
  outPrismaticJoint = -1;
  for (int i = isize(bot.joints) - 1; i >= 0; --i) {
    if (bot.joints[i].type == ArticulatedJointType::Revolute) {
      outPrismaticJoint = i;
      break;
    }
  }
  MOCHI_ASSERT(outPrismaticJoint > 0);
  BotJointPrefab& prismatic = bot.joints[outPrismaticJoint];
  prismatic.type = ArticulatedJointType::Prismatic;
  prismatic.axis = Real3{0_r, 0_r, 1_r};
  prismatic.minLimit = Real3{0_r, 0_r, -0.05_r};
  prismatic.maxLimit = Real3{0_r, 0_r, 0.05_r};
  for (auto& joint : bot.joints) {
    joint.friction.viscous = 0.3_r;
    joint.friction.coulomb = 0.2_r;
    joint.friction.stictionExtra = 0.1_r;
    joint.friction.preslidingDisplacement = 0.01_r;
    joint.friction.stribeckVel = 0.02_r;
    joint.inertia = 0.05_r;
    joint.limitStiffness = 200_r;
    joint.limitDamping = 3_r;
    joint.effortLimit = 10_r;
  }
  RebuildBotData(bot, ExpectOK{});
  bot.defaultPose.resize(isize(bot._dofIndices));
  for (int dof = 0; dof < isize(bot._dofIndices); ++dof) {
    bot.defaultPose[dof] = bot._dofIndices[dof] == outPrismaticJoint ? 0.04_r : 0.1_r;
  }
  return bot;
}

real Pow(real x, int n) {
  return static_cast<real>(std::pow(x, n));
}

class BotScaleTest : public testing::Test {
 protected:
  void SetUp() override {
    _mochiContext = mochi::CreateContext(0);
    _botsCtx = CreateRoboticsContext();
  }

  void TearDown() override {
    DestroyRoboticsContext(_botsCtx);
    for (Scene* scene : _scenes) {
      _mochiContext->DestroyScene(scene);
    }
    mochi::DestroyContext(_mochiContext);
  }

  Scene* NewScene() {
    Scene* scene = _mochiContext->CreateScene("BotScaleTest");
    scene->SetGravity(Real3{0_r, 0_r, 0_r});
    _scenes.push_back(scene);
    return scene;
  }

  Bot* Spawn(BotPrefab prefab, real scale) {
    prefab.scale = scale;
    return CreateBot(NewScene(), prefab, _botsCtx, ExpectOK{});
  }

  std::string WriteFile(std::string_view name, std::string_view content) {
    auto path = _tempDir.Path() / name;
    mochi::WriteFile(path, content, ExpectOK{});
    return path.generic_string();
  }

  Context* _mochiContext = nullptr;
  RoboticsContext* _botsCtx = nullptr;
  DynamicArray<Scene*> _scenes;
  mochi::TempDirCleanup _tempDir = mochi::CreateTempDirectory("bot_scale_test", ExpectOK{});
};

TEST_F(BotScaleTest, SerializesOnlyWhenNotOne) {
  BotPrefab bot = LoadHand();
  EXPECT_EQ(bot.scale, 1_r);
  std::string const nominal(WriteToJsonString(bot, ExpectOK{}).c_str());
  EXPECT_EQ(nominal.find("\"scale\""), std::string::npos);

  bot.scale = kScale;
  DynamicString const scaled = WriteToJsonString(bot, ExpectOK{});
  EXPECT_NEAR(ReadFromJsonString(scaled, ExpectOK{}).scale, kScale, 1e-6_r);
}

TEST_F(BotScaleTest, ValidateRejectsInvalidScale) {
  BotPrefab bot = LoadHand();
  auto const suppress = SuppressLogWarning();
  for (real const bad :
       {0_r, -1_r, std::numeric_limits<real>::infinity(), std::numeric_limits<real>::quiet_NaN()}) {
    bot.scale = bad;
    Validate(bot, nullptr, ExpectNotOK{});
  }
  bot.scale = kScale;
  Validate(bot, nullptr, ExpectOK{});
}

TEST_F(BotScaleTest, ValidateRejectsScaledTransmissions) {
  BotPrefab bot = LoadFR3();
  bot.linearTransmissions.emplace_back();
  bot.scale = kScale;
  auto const suppress = SuppressLogWarning();
  Validate(bot, nullptr, ExpectNotOK{});
}

TEST_F(BotScaleTest, ModsRejectScaledBots) {
  // A scaled copy of the hand, written beside the mod bots that reference it.
  std::string hand = mochi::ReadFileString(GetAssetPath(kHandPath), ExpectOK{});
  WriteFile("hand.superdex_bot", hand);
  hand.insert(hand.find('{') + 1, "\n  \"scale\": 0.8,");
  WriteFile("scaled_hand.superdex_bot", hand);

  std::string const attach =
      R"(, "modifications": [ { "AttachBot": { "parentLinkName": "bone_00_wrist_root", "prefix": "c_", "path": "./hand.superdex_bot", "joint": { "name": "attach", "type": "Hard" } } } ] })";
  // Controls: the scaled copy loads on its own, and the same mods on unscaled bots build.
  EXPECT_NEAR(
      LoadBotPrefabFromFile(WriteFile("scaled_copy.superdex_bot", hand), ExpectOK{}).scale,
      kScale,
      1e-6_r);
  std::string const unscaledMod = WriteFile(
      "unscaled.superdex_bot", R"({ "name": "Mod", "base": "./hand.superdex_bot")" + attach);
  EXPECT_FALSE(LoadBotPrefabFromFile(unscaledMod, ExpectOK{}).links.empty());

  auto const suppressWarnings = SuppressLogWarning();
  auto const suppressErrors = SuppressLogError();
  std::string const scaledBase = WriteFile(
      "scaled_base.superdex_bot", R"({ "name": "Mod", "base": "./scaled_hand.superdex_bot" })");
  LoadBotPrefabFromFile(scaledBase, ExpectNotOK{});

  std::string const scaledChild = WriteFile(
      "scaled_child.superdex_bot",
      R"({ "name": "Mod", "base": "./hand.superdex_bot", "modifications": [ { "AttachBot": { "parentLinkName": "bone_00_wrist_root", "prefix": "c_", "path": "./scaled_hand.superdex_bot", "joint": { "name": "attach", "type": "Hard" } } } ] })");
  LoadBotPrefabFromFile(scaledChild, ExpectNotOK{});
}

TEST_F(BotScaleTest, ScalesGeometryAndMass) {
  BotPrefab const hand = LoadHand();
  Actor* const nominal = Spawn(hand, 1_r)->GetArticulatedActor();
  Actor* const scaled = Spawn(hand, kScale)->GetArticulatedActor();

  ArticulatedShapeInfo const nominalInfo = nominal->GetArticulatedShapeInfo(ExpectOK{});
  ArticulatedShapeInfo const scaledInfo = scaled->GetArticulatedShapeInfo(ExpectOK{});
  ASSERT_EQ(isize(nominalInfo.rootFromLinksAtRest), isize(scaledInfo.rootFromLinksAtRest));
  for (int i = 0; i < isize(nominalInfo.rootFromLinksAtRest); ++i) {
    TransformRT const& a = nominalInfo.rootFromLinksAtRest[i];
    TransformRT const& b = scaledInfo.rootFromLinksAtRest[i];
    EXPECT_TRUE(NearEqual(b.GetTranslation(), a.GetTranslation() * kScale)) << "link " << i;
    EXPECT_TRUE(NearEqual(b.GetRotation(), a.GetRotation())) << "link " << i;
  }

  // Link masses come from the scaled collision shapes at the nominal density.
  Span<ActorHandle const> const nominalLinks = nominal->GetNestedLinkActors(ExpectOK{});
  Span<ActorHandle const> const scaledLinks = scaled->GetNestedLinkActors(ExpectOK{});
  ASSERT_EQ(isize(nominalLinks), isize(scaledLinks));
  int checked = 0;
  for (int i = 0; i < isize(nominalLinks); ++i) {
    if (hand.links[i].shapeFile.empty()) {
      continue;
    }
    real const m = _scenes[0]->GetActor(nominalLinks[i])->GetMass(ExpectOK{});
    real const ms = _scenes[1]->GetActor(scaledLinks[i])->GetMass(ExpectOK{});
    EXPECT_NEAR(ms, m * Pow(kScale, 3), 2e-3_r * m) << "link " << i;
    ++checked;
  }
  EXPECT_GT(checked, 0);
}

TEST_F(BotScaleTest, ScalesJointsForDynamicSimilarity) {
  int prismaticJoint = -1;
  BotPrefab bot = LoadFR3WithPrismatic(prismaticJoint);
  ArticulatedActorParams const nominal =
      BuildArticulatedActorParams(bot, FileBotLoader{}, _mochiContext, ExpectOK{});
  bot.scale = kScale;
  ArticulatedActorParams const scaled =
      BuildArticulatedActorParams(bot, FileBotLoader{}, _mochiContext, ExpectOK{});

  real const s = kScale;
  real const tol = 1e-5_r;
  for (int i = 0; i < isize(bot.joints); ++i) {
    ArticulatedJointParams const& a = nominal.joints[i];
    ArticulatedJointParams const& b = scaled.joints[i];
    SCOPED_TRACE("joint " + std::to_string(i));
    EXPECT_TRUE(NearEqual(
        b.parentLinkFromJoint.GetTranslation(), a.parentLinkFromJoint.GetTranslation() * s));
    if (a.type == ArticulatedJointType::Hard || a.type == ArticulatedJointType::Free) {
      EXPECT_EQ(b.friction, a.friction);
      EXPECT_EQ(b.limitStiffness, a.limitStiffness);
      continue;
    }
    bool const prismatic = a.type == ArticulatedJointType::Prismatic;
    real const gain = prismatic ? Pow(s, 3) : Pow(s, 5);
    real const effort = prismatic ? Pow(s, 4) : Pow(s, 5);
    real const linear = prismatic ? s : 1_r;
    EXPECT_NEAR(b.friction.viscous, a.friction.viscous * gain, tol);
    EXPECT_NEAR(b.friction.coulomb, a.friction.coulomb * effort, tol);
    EXPECT_NEAR(b.friction.stictionExtra, a.friction.stictionExtra * effort, tol);
    EXPECT_NEAR(b.friction.preslidingDisplacement, a.friction.preslidingDisplacement * linear, tol);
    EXPECT_NEAR(b.friction.stribeckVel, a.friction.stribeckVel * linear, tol);
    EXPECT_NEAR(*b.inertia, *a.inertia * gain, tol);
    EXPECT_NEAR(b.limitStiffness, a.limitStiffness * gain, 1e-3_r);
    EXPECT_NEAR(b.limitDamping, a.limitDamping * gain, tol);
    if (a.minLimit.has_value()) {
      real const limitScale = prismatic ? s : 1_r;
      EXPECT_TRUE(NearEqual(*b.minLimit, *a.minLimit * limitScale));
      EXPECT_TRUE(NearEqual(*b.maxLimit, *a.maxLimit * limitScale));
    }
  }

  DynamicArray<real> const limits = GetEffortLimitsPerDof(bot, ExpectOK{});
  for (int dof = 0; dof < isize(limits); ++dof) {
    bool const prismatic = bot._dofIndices[dof] == prismaticJoint;
    EXPECT_NEAR(limits[dof], 10_r * (prismatic ? Pow(s, 4) : Pow(s, 5)), tol);
  }
}

TEST_F(BotScaleTest, ScalesPrismaticDefaultPose) {
  int prismaticJoint = -1;
  BotPrefab const bot = LoadFR3WithPrismatic(prismaticJoint);
  Actor* const actor = Spawn(bot, kScale)->GetArticulatedActor();
  DynamicArray<real> pose(actor->GetNumDofs());
  actor->GetArticulatedPose(pose, ExpectOK{});
  int const numBaseDofs = actor->GetNumDofs() - isize(bot._dofIndices);
  for (int dof = 0; dof < isize(bot._dofIndices); ++dof) {
    real const expected = bot._dofIndices[dof] == prismaticJoint ? 0.04_r * kScale : 0.1_r;
    EXPECT_NEAR(pose[numBaseDofs + dof], expected, 1e-5_r) << "dof " << dof;
  }
}

TEST_F(BotScaleTest, ScalesSensorOffsets) {
  BotPrefab bot = LoadFR3();
  BotSensorPrefab camera;
  camera.type = DynamicString("SENSOR_CAMERA");
  camera.name = DynamicString("camera");
  camera.parentFromSensor = TransformRT(Quaternion{}, Real3{0.1_r, 0.2_r, 0.3_r});
  bot.links.back().sensors.push_back(camera);
  Bot* const spawned = Spawn(bot, kScale);
  ASSERT_EQ(isize(spawned->GetSensorHandles()), 1);
  SensorBase const* const sensor = _botsCtx->GetSensor(spawned->GetSensorHandles()[0]);
  ASSERT_NE(sensor, nullptr);
  EXPECT_TRUE(NearEqual(
      sensor->GetParentFromSensor().GetTranslation(), Real3{0.1_r, 0.2_r, 0.3_r} * kScale));
}

/* Pose-controller gains are nominal; the controller hands Mochi gains scaled for the bot. */
TEST_F(BotScaleTest, PoseControllerScalesGains) {
  Bot* const bot = Spawn(LoadHand(), kScale);
  auto* const controller = dynamic_cast<ControllerMochiArticulatedPose*>(
      bot->CreateController("MOCHI_ARTICULATED_POSE", "pose", ExpectOK{}));
  ASSERT_NE(controller, nullptr);
  EXPECT_NEAR(controller->GetBotScale(), kScale, 1e-6_r);

  ControllerMochiArticulatedPose::Params params;
  params.poseControllerParams.linkPosTracking = {PoseTrackingParams{1000_r, 40_r, 0.02_r}};
  params.poseControllerParams.linkRotTracking = {PoseTrackingParams{10_r, 0.1_r, 0.5_r}};
  params.poseControllerParams.jointTracking = {PoseTrackingParams{2_r, 0.01_r, -1_r}};
  controller->SetParams(params, ExpectOK{});
  controller->Initialize(/*removeExisting=*/true, ExpectOK{});
  // GetParams reports the nominal gains.
  EXPECT_EQ(
      controller->GetParams().poseControllerParams.linkRotTracking[0].stiffness,
      params.poseControllerParams.linkRotTracking[0].stiffness);

  PoseControllerParams applied(isize(bot->GetBotPrefab().links));
  bot->GetArticulatedActor()->GetArticulatedPoseControllerParams(applied, ExpectOK{});
  real const s3 = Pow(kScale, 3);
  real const s5 = Pow(kScale, 5);
  auto expectNear = [](PoseTrackingParams const& a, PoseTrackingParams const& b) {
    EXPECT_NEAR(a.stiffness, b.stiffness, 1e-4_r * b.stiffness);
    EXPECT_NEAR(a.damping, b.damping, 1e-4_r * b.damping);
    EXPECT_NEAR(a.saturation, b.saturation, 1e-6_r);
  };
  ASSERT_FALSE(applied.linkPosTracking.empty());
  for (auto const& p : applied.linkPosTracking) {
    expectNear(p, PoseTrackingParams{1000_r * s3, 40_r * s3, 0.02_r * kScale});
  }
  for (auto const& p : applied.linkRotTracking) {
    expectNear(p, PoseTrackingParams{10_r * s5, 0.1_r * s5, 0.5_r});
  }
  // Joint tracking applies to the actuated joints only.
  BotPrefab const& prefab = bot->GetBotPrefab();
  for (int i = 0; i < isize(applied.jointTracking); ++i) {
    ArticulatedJointType const type = prefab.joints[i].type;
    if (type == ArticulatedJointType::Revolute || type == ArticulatedJointType::Spherical) {
      expectNear(applied.jointTracking[i], PoseTrackingParams{2_r * s5, 0.01_r * s5, -1_r});
    }
  }
}

/* End to end: a scaled hand driven by the same nominal gains toward the same joint targets follows
 * the same joint trajectory as the nominal hand. */
TEST_F(BotScaleTest, ScaledHandTracksLikeNominal) {
  // Contact margins are absolute distances, so self-contact is disabled to compare the articulated
  // dynamics alone.
  BotPrefab hand = LoadHand();
  for (auto& link : hand.links) {
    link.layer = DynamicString("bot_scale_test");
  }
  struct Run {
    Bot* bot = nullptr;
    ControllerMochiArticulatedPose* controller = nullptr;
  };
  auto start = [&](real scale) {
    Run run;
    BotPrefab prefab = hand;
    prefab.scale = scale;
    Scene* const scene = NewScene();
    scene->EnableLayerContactSymmetric("bot_scale_test", "bot_scale_test", false, ExpectOK{});
    run.bot = CreateBot(scene, prefab, _botsCtx, ExpectOK{});
    run.controller = dynamic_cast<ControllerMochiArticulatedPose*>(
        run.bot->CreateController("MOCHI_ARTICULATED_POSE", "pose", ExpectOK{}));
    int const numLinks = isize(hand.links);
    ControllerMochiArticulatedPose::Params params;
    params.poseControllerParams.linkPosTracking.resize(numLinks);
    params.poseControllerParams.linkRotTracking.resize(numLinks);
    params.poseControllerParams.linkPosTracking[0] = PoseTrackingParams{1000_r, 40_r, -1_r};
    params.poseControllerParams.linkRotTracking[0] = PoseTrackingParams{50_r, 0.4_r, -1_r};
    params.poseControllerParams.jointTracking = {PoseTrackingParams{0.05_r, 0.002_r, -1_r}};
    run.controller->SetParams(params, ExpectOK{});
    run.controller->Initialize(/*removeExisting=*/true, ExpectOK{});
    return run;
  };
  Run nominal = start(1_r);
  Run scaled = start(kScale);

  // Curl every finger joint a little.
  ControllerMochiArticulatedPose::Target target;
  target.worldFromRoot = hand.worldFromRoot;
  target.poseDofs.resize(isize(hand._dofIndices));
  std::fill(target.poseDofs.begin(), target.poseDofs.end(), 0.3_r);
  ControllerMochiArticulatedPose::Obsv const obsv;

  int const numDofs = nominal.bot->GetArticulatedActor()->GetNumDofs();
  DynamicArray<real> poseA(numDofs);
  DynamicArray<real> poseB(numDofs);
  int const numBaseDofs = numDofs - isize(hand._dofIndices);
  real maxError = 0_r;
  real maxMotion = 0_r;
  for (int step = 0; step < 200; ++step) {
    for (Run* run : {&nominal, &scaled}) {
      run->controller->ComputeOutput(obsv, target, ExpectOK{});
      run->bot->GetScene()->Step(1_r / 500_r);
    }
    nominal.bot->GetArticulatedActor()->GetArticulatedPose(poseA, ExpectOK{});
    scaled.bot->GetArticulatedActor()->GetArticulatedPose(poseB, ExpectOK{});
    for (int dof = numBaseDofs; dof < numDofs; ++dof) {
      maxError = Max(maxError, Abs(poseA[dof] - poseB[dof]));
      maxMotion = Max(maxMotion, Abs(poseA[dof]));
    }
  }
  EXPECT_GT(maxMotion, 0.1_r);
  EXPECT_LT(maxError, 0.01_r * maxMotion) << "max joint error " << maxError;
}

} // namespace

/* On a bot with both prismatic and rotational joints, joint tracking is scaled per joint: a
 * broadcast entry is expanded per link, hard joints keep default (ignored) entries, and a per-link
 * array of the wrong size is rejected. */
TEST_F(BotScaleTest, PoseControllerScalesMixedJointTracking) {
  int prismaticJoint = -1;
  Bot* const bot = Spawn(LoadFR3WithPrismatic(prismaticJoint), kScale);
  BotPrefab const& prefab = bot->GetBotPrefab();
  int const numLinks = isize(prefab.links);
  real const s3 = Pow(kScale, 3);
  real const s5 = Pow(kScale, 5);
  auto expectedFor = [&](int link, PoseTrackingParams const& nominal) {
    switch (prefab.joints[link].type) {
      case ArticulatedJointType::Hard:
        return PoseTrackingParams{};
      case ArticulatedJointType::Prismatic:
        return PoseTrackingParams{
            nominal.stiffness * s3, nominal.damping * s3, nominal.saturation * kScale};
      default:
        return PoseTrackingParams{nominal.stiffness * s5, nominal.damping * s5, nominal.saturation};
    }
  };
  auto applyAndCheck = [&](std::string_view name,
                           DynamicArray<PoseTrackingParams> const& jointTracking,
                           auto const& nominalFor) {
    auto* const controller = dynamic_cast<ControllerMochiArticulatedPose*>(
        bot->CreateController("MOCHI_ARTICULATED_POSE", name, ExpectOK{}));
    ASSERT_NE(controller, nullptr);
    ControllerMochiArticulatedPose::Params params;
    params.poseControllerParams.jointTracking = jointTracking;
    controller->SetParams(params, ExpectOK{});
    controller->Initialize(/*removeExisting=*/true, ExpectOK{});
    PoseControllerParams applied(numLinks);
    bot->GetArticulatedActor()->GetArticulatedPoseControllerParams(applied, ExpectOK{});
    ASSERT_EQ(isize(applied.jointTracking), numLinks);
    int numPrismatic = 0;
    for (int i = 0; i < numLinks; ++i) {
      PoseTrackingParams const expected = expectedFor(i, nominalFor(i));
      EXPECT_NEAR(applied.jointTracking[i].stiffness, expected.stiffness, 1e-4_r) << "link " << i;
      EXPECT_NEAR(applied.jointTracking[i].damping, expected.damping, 1e-6_r) << "link " << i;
      EXPECT_NEAR(applied.jointTracking[i].saturation, expected.saturation, 1e-6_r) << "link " << i;
      numPrismatic += prefab.joints[i].type == ArticulatedJointType::Prismatic ? 1 : 0;
    }
    EXPECT_EQ(numPrismatic, 1);
  };

  // A broadcast entry.
  PoseTrackingParams const broadcast{20_r, 0.5_r, 0.3_r};
  applyAndCheck("broadcast", {broadcast}, [&](int) { return broadcast; });

  // One entry per link (default on hard joints, which Mochi ignores).
  DynamicArray<PoseTrackingParams> perLink(numLinks);
  for (int i = 0; i < numLinks; ++i) {
    if (prefab.joints[i].type != ArticulatedJointType::Hard) {
      perLink[i] = PoseTrackingParams{real(i + 1), 0.1_r * real(i + 1), 0.2_r};
    }
  }
  applyAndCheck("per_link", perLink, [&](int i) { return perLink[i]; });

  // A per-link array of the wrong size.
  auto* const controller = dynamic_cast<ControllerMochiArticulatedPose*>(
      bot->CreateController("MOCHI_ARTICULATED_POSE", "wrong_size", ExpectOK{}));
  ASSERT_NE(controller, nullptr);
  ControllerMochiArticulatedPose::Params params;
  params.poseControllerParams.jointTracking = {broadcast, broadcast};
  controller->SetParams(params, ExpectOK{});
  controller->Initialize(/*removeExisting=*/true, ExpectNotOK{});
}

/* CreateBot does not validate, so it enforces the scale restrictions itself. */
TEST_F(BotScaleTest, CreateBotRejectsScaledTransmissions) {
  BotPrefab bot = LoadFR3();
  bot.linearTransmissions.emplace_back();
  bot.scale = kScale;
  auto const suppress = SuppressLogError();
  EXPECT_EQ(CreateBot(NewScene(), bot, _botsCtx, ExpectNotOK{}), nullptr);
}

/* The skin's rest geometry scales with the bot about the root, on both skin paths: the cached
 * file shape (bones in link order) and the remapped shape of a composed bot. */
TEST_F(BotScaleTest, ScalesSkin) {
  BotPrefab const composed = LoadBotPrefabFromFile(
      GetAssetPath("bots/hands/wuji_hand2_beta1/left/wuji_hand2_beta1_skinned_left.superdex_bot"),
      ExpectOK{});
  ASSERT_TRUE(composed.skin.has_value());
  ASSERT_FALSE(composed._skinBoneLinks.empty());
  // This skin's bones already follow the link order, so dropping the remap selects the cached file
  // shape without changing the skin.
  ASSERT_LE(isize(composed._skinBoneLinks), isize(composed.links));
  for (int b = 0; b < isize(composed._skinBoneLinks); ++b) {
    ASSERT_EQ(composed._skinBoneLinks[b], composed.links[b].name);
  }
  BotPrefab linkOrder = composed;
  linkOrder._skinBoneLinks.clear();

  for (BotPrefab const* bot : {&composed, static_cast<BotPrefab const*>(&linkOrder)}) {
    BotPrefab scaled = *bot;
    scaled.scale = kScale;
    ArticulatedActorParams const nominalParams =
        BuildArticulatedActorParams(*bot, FileBotLoader{}, _mochiContext, ExpectOK{});
    ArticulatedActorParams const scaledParams =
        BuildArticulatedActorParams(scaled, FileBotLoader{}, _mochiContext, ExpectOK{});
    ASSERT_TRUE(nominalParams.skin.has_value());
    ASSERT_TRUE(scaledParams.skin.has_value());
    Span<real const> const nominal =
        _mochiContext->GetShapeMesh(nominalParams.skin->shape, ExpectOK{}).coordinates;
    Span<real const> const actual =
        _mochiContext->GetShapeMesh(scaledParams.skin->shape, ExpectOK{}).coordinates;
    ASSERT_EQ(isize(actual), isize(nominal));
    ASSERT_FALSE(nominal.empty());
    real maxError = 0_r;
    for (int i = 0; i < isize(nominal); ++i) {
      maxError = Max(maxError, Abs(actual[i] - nominal[i] * kScale));
    }
    EXPECT_LT(maxError, 1e-5_r) << (bot == &composed ? "remapped skin" : "cached skin");
  }
}
