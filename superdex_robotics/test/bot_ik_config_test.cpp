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
#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/file_utils.h>
#include <picojson/picojson.h>
#include <superdex_robotics/ik/bot_ik_config.h>
#include <superdex_robotics/utils/bot_utils.h>

#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

using namespace mochi;
using namespace mochi::test;
using namespace superdex::robotics;

namespace {

void SetInvalidRotation(TransformRT& frame, Quaternion const& rotation) {
  auto const expectedInvalidRotationWarning = SuppressLogWarning();
  frame.SetRotation(rotation);
}

BotIKConfig Config() {
  BotIKConfig spec;
  spec.outputBot = "robot.superdex_bot";
  BotIKGoalConfig goalConfig;
  goalConfig.name = "goal";
  goalConfig.outputBotLink = "tip";
  goalConfig.baseOrInputLinkFrameFromConstraintFrame = TransformRT{
      Quaternion::FromRotationVector(Real3{0.1_r, 0.2_r, -0.1_r}), Real3{0.2_r, -0.3_r, 0.4_r}};
  spec.goals.push_back(goalConfig);
  return spec;
}

picojson::value Json(BotIKConfig const& spec) {
  picojson::value value;
  SReflect::ToJsonValue(spec, value);
  return value;
}

// Historical fixtures retain the complete v1-v7 Target or v8 Reference vocabulary.
picojson::value LegacyJson(BotIKConfig const& spec, int version = 7) {
  auto value = Json(spec);
  auto& fields = value.get<picojson::object>();
  fields["version"] = picojson::value(int64_t(version));
  for (auto const& [current, historical] :
       {std::pair{"outputBot", "bot"},
        std::pair{"outputBotConfiguration", "botConfiguration"},
        std::pair{"inputBot", version < 8 ? "targetBot" : "referenceBot"},
        std::pair{
            "outputBaseFrameFromInputBaseFrame",
            version < 8 ? "baseFromTargetBot" : "baseFromReferenceBot"},
        std::pair{"inputBotScale", version < 8 ? "targetBotScale" : "referenceBotScale"},
        std::pair{
            "inputBotConfiguration",
            version < 8 ? "targetBotConfiguration" : "referenceBotConfiguration"}}) {
    fields[historical] = std::move(fields.at(current));
    fields.erase(current);
  }
  for (auto& item : fields.at("goals").get<picojson::array>()) {
    auto& goal = item.get<picojson::object>();
    for (auto const& [current, historical] :
         {std::pair{"outputBotLink", "link"},
          std::pair{"linkFrameFromConstraintFrame", "linkRelativeFrame"},
          std::pair{"inputBotLink", version < 8 ? "targetBotLink" : "referenceBotLink"},
          std::pair{
              "baseOrInputLinkFrameFromConstraintFrame",
              version < 8 ? "targetFrame" : "goalFrame"}}) {
      goal[historical] = std::move(goal.at(current));
      goal.erase(current);
    }
  }
  if (version < 8) {
    fields["targets"] = std::move(fields.at("goals"));
    fields.erase("goals");
  }
  return value;
}

// Preserve historical feature gates, not only the vocabulary, in versioned fixtures.
picojson::value VersionedJson(BotIKConfig const& spec, int version) {
  auto value = version < 9 ? LegacyJson(spec, version) : Json(spec);
  auto& fields = value.get<picojson::object>();
  fields["version"] = picojson::value(int64_t(version));
  if (version < 3) {
    fields.erase("targetBotScale");
  }
  if (version < 4) {
    fields.erase("botConfiguration");
  }
  if (version < 6) {
    fields.erase("jointCouplings");
  }
  return value;
}

BotPrefab Robot() {
  BotPrefab bot;
  bot.name = "leaf_defaults";
  bot.worldFromRoot = TransformRT{Quaternion::RotationY(0.4_r), Real3{1_r, 2_r, 3_r}};
  BotLinkPrefab root;
  root.name = "root";
  root.parentLink = kIndexNone;
  BotLinkPrefab arm;
  arm.name = "arm";
  arm.parentLink = 0;
  BotLinkPrefab tip;
  tip.name = "tip";
  tip.parentLink = 1;
  BotLinkPrefab weld;
  weld.name = "weld";
  weld.parentLink = 0;
  bot.links = {root, arm, tip, weld};
  BotJointPrefab base;
  base.name = "base";
  base.type = ArticulatedJointType::Hard;
  BotJointPrefab turn;
  turn.name = "turn";
  turn.type = ArticulatedJointType::Revolute;
  turn.axis = {0_r, 0_r, 1_r};
  turn.parentLinkFromJoint = TransformRT{Real3{0.3_r, 0_r, 0_r}};
  BotJointPrefab end;
  end.name = "end";
  end.type = ArticulatedJointType::Hard;
  end.parentLinkFromJoint = TransformRT{Real3{0.7_r, 0_r, 0_r}};
  BotJointPrefab fixed;
  fixed.name = "fixed";
  fixed.type = ArticulatedJointType::Hard;
  fixed.parentLinkFromJoint = TransformRT{Quaternion::RotationX(-0.2_r), Real3{0_r, 0.5_r, 0_r}};
  bot.joints = {base, turn, end, fixed};
  bot.defaultPose = {0.35_r};
  RebuildBotData(bot, ExpectOK{});
  return bot;
}

TEST(IKConfigTest, RoundTrip_AllFieldsAndNativeDefaults) {
  auto spec = Config();
  EXPECT_TRUE(spec.createParams.fixedBase);
  EXPECT_EQ(spec.version, kIKConfigVersion);
  EXPECT_EQ(spec.version, 9);
  EXPECT_TRUE(spec.outputBotConfiguration.empty());
  spec.outputBotConfiguration = {0.3_r, -0.4_r, 0_r};
  EXPECT_EQ(spec.createParams.solverParams.maxElapsedTimeSeconds, kIKInitialSolveBudgetSeconds);
  EXPECT_EQ(spec.createParams.solverParams.maxElapsedTimeSeconds, 0.01);
  EXPECT_EQ(spec.createParams.solverParams.maxIter, experimental::IKSolverParams{}.maxIter);
  EXPECT_FALSE(BotIKCreateParams{}.fixedBase);
  EXPECT_EQ(spec.goals[0].positionStiffness, kIKInitialPositionStiffness);
  EXPECT_EQ(spec.goals[0].rotationStiffness, kIKInitialRotationStiffness);
  EXPECT_EQ(spec.goals[0].positionStiffness, 100_r);
  EXPECT_EQ(spec.goals[0].rotationStiffness, 1_r);
  EXPECT_EQ(spec.goals[0].linkFrameFromConstraintFrame, TransformRT{});
  spec.goals[0].linkFrameFromConstraintFrame =
      TransformRT{Quaternion::RotationX(0.2_r), Real3{0.1_r, -0.2_r, 0.3_r}};
  spec.createParams.fixedBase = false;
  spec.createParams.excludeCollisionShapes = false;
  auto& p = spec.createParams.solverParams;
  p.maxIter = 37;
  p.verbosity = VerbosityLevel::Verbose;
  p.absTol = 0.003_r;
  p.relTol = 0.0002_r;
  p.positionErrorThres = 0.004_r;
  p.rotationErrorThres = 0.005_r;
  p.lineSearchMaxIter = 13;
  p.maxElapsedTimeSeconds = 0.075;
  spec.inputBot = "reference.superdex_bot";
  spec.outputBaseFrameFromInputBaseFrame =
      TransformRT{Quaternion::RotationZ(0.3_r), Real3{0.5_r, 0.2_r, 0_r}};
  EXPECT_EQ(spec.inputBotScale, 1_r);
  spec.inputBotScale = 0.5_r;
  spec.inputBotConfiguration = {0.1_r, -0.2_r};
  spec.goals[0].inputBotLink = "reference_tip";
  spec.goals[0].enabled = false;
  spec.goals[0].positionStiffness = 0_r;
  spec.goals[0].rotationStiffness = 57_r;
  DynamicString diagnostic;
  BotIKConfig loaded;
  ASSERT_TRUE(ParseIKConfig(SerializeIKConfig(spec), loaded, diagnostic)) << diagnostic;
  EXPECT_TRUE(diagnostic.empty());
  EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(spec));
  EXPECT_EQ(loaded.outputBotConfiguration, spec.outputBotConfiguration);
  EXPECT_EQ(Json(loaded).get<picojson::object>().count("targetBotRecording"), 0);
  EXPECT_TRUE(
      Json(loaded).get<picojson::object>().at("outputBotConfiguration").is<picojson::array>());
  EXPECT_EQ(
      Json(loaded)
          .get<picojson::object>()
          .at("createParams")
          .get<picojson::object>()
          .at("solverParams")
          .get<picojson::object>()
          .size(),
      8);
  EXPECT_TRUE(NearEqual(
      loaded.goals[0].baseOrInputLinkFrameFromConstraintFrame,
      spec.goals[0].baseOrInputLinkFrameFromConstraintFrame));
  EXPECT_TRUE(NearEqual(
      loaded.goals[0].linkFrameFromConstraintFrame, spec.goals[0].linkFrameFromConstraintFrame));
}

TEST(IKConfigTest, Parse_OmittedOptionalFieldsRetainStudioDefaults) {
  DynamicString diagnostic;
  for (int const version : {1, 2, 3, 4, 5, 6, 7, 8, 9}) {
    auto const value = picojson::value(
        picojson::object{
            {"version", picojson::value(int64_t(version))},
            {version < 9 ? "bot" : "outputBot", picojson::value("robot.superdex_bot")}});
    auto loaded = Config();
    loaded.outputBotConfiguration = {0.2_r};
    loaded.inputBotConfiguration = {-0.3_r};
    loaded.inputBot = "reference.superdex_bot";
    ASSERT_TRUE(ParseIKConfig(value.serialize(), loaded, diagnostic)) << diagnostic;
    EXPECT_EQ(loaded.version, 9);
    EXPECT_TRUE(loaded.createParams.fixedBase);
    EXPECT_EQ(loaded.createParams.solverParams.maxElapsedTimeSeconds, 0.01);
    EXPECT_TRUE(loaded.outputBotConfiguration.empty());
    EXPECT_TRUE(loaded.inputBotConfiguration.empty());
    EXPECT_TRUE(loaded.goals.empty());
  }
}

TEST(IKConfigTest, Parse_OmittedLinkRelativeFrameUsesIdentity) {
  auto value = Json(Config());
  value.get<picojson::object>().at("goals").get<picojson::array>()[0].get<picojson::object>().erase(
      "linkFrameFromConstraintFrame");
  BotIKConfig loaded;
  DynamicString diagnostic;
  ASSERT_TRUE(ParseIKConfig(value.serialize(), loaded, diagnostic)) << diagnostic;
  ASSERT_EQ(loaded.goals.size(), 1);
  EXPECT_EQ(loaded.goals[0].linkFrameFromConstraintFrame, TransformRT{});
}

TEST(IKConfigTest, Validate_InvalidLinkRelativeFramePreservesParseDestination) {
  auto const initial = Config();
  auto loaded = initial;
  DynamicString diagnostic;
  for (int fault = 0; fault < 3; ++fault) {
    auto invalid = initial;
    auto& frame = invalid.goals[0].linkFrameFromConstraintFrame;
    if (fault == 0) {
      frame.SetTranslation(Real3{std::numeric_limits<real>::quiet_NaN(), 0_r, 0_r});
    } else {
      SetInvalidRotation(frame, fault == 1 ? Quaternion::Zero() : Quaternion{2_r, 0_r, 0_r, 0_r});
    }
    EXPECT_FALSE(ValidateIKConfig(invalid, diagnostic));
    EXPECT_NE(std::string(diagnostic).find("linkFrameFromConstraintFrame"), std::string::npos);
    if (fault != 0) {
      EXPECT_FALSE(ParseIKConfig(SerializeIKConfig(invalid), loaded, diagnostic));
    }
    EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
  }
}

TEST(IKConfigTest, Parse_MalformedAndUnsupportedDocumentsPreserveDestination) {
  auto const initial = Config();
  auto out = initial;
  DynamicString diagnostic;
  for (std::string const text :
       {"{",
        "[]",
        "null",
        "{}",
        R"({"version":10,"outputBot":"x"})",
        R"({"version":1.2,"bot":"x"})",
        R"({"version":1e30,"bot":"x"})",
        R"({"version":1,"bot":null})",
        R"({"version":1,"bot":""})",
        R"({"version":1,"bot":"x","unknown":5})"}) {
    SCOPED_TRACE(text);
    EXPECT_FALSE(ParseIKConfig(text, out, diagnostic));
    EXPECT_FALSE(diagnostic.empty());
    EXPECT_EQ(SerializeIKConfig(out), SerializeIKConfig(initial));
  }
}

TEST(IKConfigTest, Parse_EveryNestedNumericFieldRejectsCoercionAndInvalidRange) {
  auto const initial = Config();
  DynamicString diagnostic;
  for (auto const* field :
       {"maxIter",
        "absTol",
        "relTol",
        "positionErrorThres",
        "rotationErrorThres",
        "lineSearchMaxIter",
        "maxElapsedTimeSeconds"}) {
    for (auto const& bad :
         {picojson::value("2"), picojson::value(true), picojson::value{}, picojson::value(-1.0)}) {
      auto value = Json(initial);
      value.get<picojson::object>()
          .at("createParams")
          .get<picojson::object>()
          .at("solverParams")
          .get<picojson::object>()[field] = bad;
      auto out = initial;
      SCOPED_TRACE(field);
      EXPECT_FALSE(ParseIKConfig(value.serialize(), out, diagnostic));
      EXPECT_FALSE(diagnostic.empty());
      EXPECT_EQ(SerializeIKConfig(out), SerializeIKConfig(initial));
    }
  }
  for (auto const* field : {"maxIter", "lineSearchMaxIter"}) {
    for (double const bad : {0.0, 1.5, 1e30}) {
      auto value = Json(initial);
      value.get<picojson::object>()
          .at("createParams")
          .get<picojson::object>()
          .at("solverParams")
          .get<picojson::object>()[field] = picojson::value(bad);
      BotIKConfig out;
      EXPECT_FALSE(ParseIKConfig(value.serialize(), out, diagnostic));
    }
  }
  for (auto const& bad :
       {picojson::value("Unknown"),
        picojson::value(2.0),
        picojson::value(false),
        picojson::value{}}) {
    auto value = Json(initial);
    value.get<picojson::object>()
        .at("createParams")
        .get<picojson::object>()
        .at("solverParams")
        .get<picojson::object>()["verbosity"] = bad;
    BotIKConfig out;
    EXPECT_FALSE(ParseIKConfig(value.serialize(), out, diagnostic));
  }
}

TEST(IKConfigTest, Parse_GoalTypesAndShapesAreStrict) {
  auto const initial = Config();
  DynamicString diagnostic;
  for (auto const* field :
       {"name",
        "enabled",
        "outputBotLink",
        "inputBotLink",
        "baseOrInputLinkFrameFromConstraintFrame",
        "positionStiffness",
        "rotationStiffness"}) {
    auto value = Json(initial);
    value.get<picojson::object>()
        .at("goals")
        .get<picojson::array>()[0]
        .get<picojson::object>()[field] = picojson::value{};
    BotIKConfig out;
    SCOPED_TRACE(field);
    EXPECT_FALSE(ParseIKConfig(value.serialize(), out, diagnostic));
  }
  for (auto const* field : {"translation", "rotation"}) {
    for (picojson::array const& bad :
         {picojson::array{},
          picojson::array{picojson::value(1.0), picojson::value(2.0)},
          picojson::array{picojson::value(1.0), picojson::value("2"), picojson::value(3.0)}}) {
      auto value = Json(initial);
      value.get<picojson::object>()
          .at("goals")
          .get<picojson::array>()[0]
          .get<picojson::object>()
          .at("baseOrInputLinkFrameFromConstraintFrame")
          .get<picojson::object>()[field] = picojson::value(bad);
      BotIKConfig out;
      EXPECT_FALSE(ParseIKConfig(value.serialize(), out, diagnostic));
    }
  }
}

TEST(IKConfigTest, Validate_NonfiniteValuesRejectAndUnlimitedBudgetIsLegal) {
  DynamicString diagnostic;
  for (real const bad :
       {std::numeric_limits<real>::quiet_NaN(), std::numeric_limits<real>::infinity(), -1_r}) {
    for (int field = 0; field < 6; ++field) {
      auto spec = Config();
      real* const fields[] = {
          &spec.createParams.solverParams.absTol,
          &spec.createParams.solverParams.relTol,
          &spec.createParams.solverParams.positionErrorThres,
          &spec.createParams.solverParams.rotationErrorThres,
          &spec.goals[0].positionStiffness,
          &spec.goals[0].rotationStiffness};
      *fields[field] = bad;
      EXPECT_FALSE(ValidateIKConfig(spec, diagnostic));
      EXPECT_FALSE(diagnostic.empty());
    }
  }
  auto spec = Config();
  spec.createParams.solverParams.maxElapsedTimeSeconds = 0;
  spec.goals[0].positionStiffness = 0_r;
  spec.goals[0].rotationStiffness = 0_r;
  EXPECT_TRUE(ValidateIKConfig(spec, diagnostic));
  spec.goals[0].baseOrInputLinkFrameFromConstraintFrame.SetTranslation(
      Real3{0_r, std::numeric_limits<real>::infinity(), 0_r});
  EXPECT_FALSE(ValidateIKConfig(spec, diagnostic));
  spec = Config();
  SetInvalidRotation(spec.goals[0].baseOrInputLinkFrameFromConstraintFrame, Quaternion::Zero());
  EXPECT_FALSE(ValidateIKConfig(spec, diagnostic));
  spec = Config();
  spec.createParams.solverParams.maxElapsedTimeSeconds = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(ValidateIKConfig(spec, diagnostic));
}

TEST(IKConfigTest, Parse_DuplicatesRemainInspectableButCannotValidateOrSave) {
  auto spec = Config();
  spec.goals.push_back(BotIKGoalConfig{spec.goals[0]});
  DynamicString diagnostic;
  BotIKConfig loaded;
  ASSERT_TRUE(ParseIKConfig(SerializeIKConfig(spec), loaded, diagnostic));
  EXPECT_EQ(loaded.goals.size(), 2);
  EXPECT_NE(diagnostic.find("Duplicate goal name"), std::string::npos);
  EXPECT_NE(diagnostic.find("Duplicate goal link"), std::string::npos);
  EXPECT_FALSE(ValidateIKConfig(loaded, diagnostic));
  loaded.goals[1].name = "distinct";
  EXPECT_FALSE(ValidateIKGoalBindings(loaded, diagnostic));
  loaded.goals[1].outputBotLink = "unresolved_link";
  EXPECT_TRUE(ValidateIKConfig(loaded, diagnostic));
}

TEST(IKConfigTest, Paths_RelativeToDocumentAndLexicalRoundTrip) {
  std::string const doc =
      (std::filesystem::temp_directory_path() / "ik_paths" / "goals" / "arm.superdex_ik")
          .generic_string();
  std::string const bot =
      (std::filesystem::path(doc).parent_path().parent_path() / "robots" / "arm.superdex_bot")
          .generic_string();
  EXPECT_EQ(MakeIKBotReference(doc, bot), "../robots/arm.superdex_bot");
  EXPECT_EQ(std::string(ResolveIKBotPath(doc, MakeIKBotReference(doc, bot))), bot);
  EXPECT_EQ(std::string(ResolveIKBotPath(doc, bot)), bot);
  EXPECT_TRUE(ResolveIKBotPath(doc, "").empty());
}

TEST(IKConfigTest, LoadSave_FileRoundTripAndFailedLoadPreservation) {
  auto cleanup = CreateTempDirectory("ik_config_test", ExpectOK{});
  std::string const path = (cleanup.Path() / "goals.superdex_ik").generic_string();
  auto spec = Config();
  spec.inputBot = "reference.superdex_bot";
  DynamicString diagnostic;
  ASSERT_TRUE(SaveIKConfig(path, spec, diagnostic)) << diagnostic;
  BotIKConfig loaded;
  ASSERT_TRUE(LoadIKConfig(path, loaded, diagnostic)) << diagnostic;
  EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(spec));
  std::string const malformed = "{broken";
  WriteFile(path, Span<char const>{malformed.data(), malformed.size()}, ExpectOK{});
  EXPECT_FALSE(LoadIKConfig(path, loaded, diagnostic));
  EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(spec));
  EXPECT_FALSE(
      LoadIKConfig((cleanup.Path() / "missing.superdex_ik").generic_string(), loaded, diagnostic));
  auto duplicate = spec;
  duplicate.goals.push_back(BotIKGoalConfig{duplicate.goals[0]});
  EXPECT_FALSE(SaveIKConfig(path, duplicate, diagnostic));
  EXPECT_EQ(ReadFileString(path, ExpectOK{}), malformed);
}

TEST(IKConfigTest, Initialize_AuthoredBranchedPoseIncludesHardLeavesAndDoesNotMutateBot) {
  auto const bot = Robot();
  auto const original = bot;
  auto spec = Config();
  DynamicString diagnostic;
  ASSERT_TRUE(InitializeIKGoals(bot, spec, diagnostic)) << diagnostic;
  ASSERT_EQ(spec.goals.size(), 2);
  EXPECT_EQ(spec.goals[0].outputBotLink, "tip");
  EXPECT_EQ(spec.goals[1].outputBotLink, "weld");
  TransformRT const expectedTip =
      TransformRT{Quaternion::RotationZ(0.35_r), Real3{0.3_r, 0_r, 0_r}} *
      TransformRT{Real3{0.7_r, 0_r, 0_r}};
  TransformRT const expectedWeld{Quaternion::RotationX(-0.2_r), Real3{0_r, 0.5_r, 0_r}};
  EXPECT_TRUE(NearEqual(IKGoalConstraintTransform(spec.goals[0]), expectedTip, 2e-5_r));
  EXPECT_TRUE(NearEqual(IKGoalConstraintTransform(spec.goals[1]), expectedWeld, 2e-5_r));
  EXPECT_EQ(bot.defaultPose, original.defaultPose);
  EXPECT_EQ(bot.worldFromRoot, original.worldFromRoot);
  EXPECT_EQ(bot.links, original.links);
  EXPECT_EQ(bot.joints, original.joints);
}

TEST(IKConfigTest, Initialize_SingleRootAndInvalidBotPreservation) {
  auto bot = Robot();
  bot.links.resize(1);
  bot.joints.resize(1);
  bot.defaultPose.clear();
  RebuildBotData(bot, ExpectOK{});
  auto spec = Config();
  DynamicString diagnostic;
  ASSERT_TRUE(InitializeIKGoals(bot, spec, diagnostic));
  ASSERT_EQ(spec.goals.size(), 1);
  EXPECT_EQ(spec.goals[0].outputBotLink, "root");
  EXPECT_TRUE(NearEqual(IKGoalConstraintTransform(spec.goals[0]), TransformRT{}, 2e-5_r));
  auto const before = SerializeIKConfig(spec);
  {
    auto const expectedInvalidBotWarning = SuppressLogWarning();
    EXPECT_FALSE(InitializeIKGoals(BotPrefab{}, spec, diagnostic));
  }
  EXPECT_EQ(SerializeIKConfig(spec), before);
}

TEST(IKConfigTest, Initialize_StalePreparedDataAndNonfiniteSeedPreserveAuthoredGoals) {
  auto const bot = Robot();
  auto const original = Config();
  DynamicString diagnostic;
  for (int fault = 0; fault < 3; ++fault) {
    auto bad = bot;
    if (fault == 0) {
      bad.links[1]._index = 99;
    }
    if (fault == 1) {
      bad.links[1]._parentFromLink = TransformRT{};
    }
    if (fault == 2) {
      bad.defaultPose[0] = std::numeric_limits<real>::quiet_NaN();
    }
    auto out = original;
    EXPECT_FALSE(InitializeIKGoals(bad, out, diagnostic));
    EXPECT_FALSE(diagnostic.empty());
    EXPECT_EQ(SerializeIKConfig(out), SerializeIKConfig(original));
  }
}

TEST(IKConfigTest, GoalTransform_NonidentityAnchorRoundTripTranslationAndRotation) {
  TransformRT const anchor{Quaternion::RotationY(0.4_r), Real3{1_r, 2_r, 3_r}};
  TransformRT const local{
      Quaternion::RotationZ(-0.3_r) * Quaternion::RotationX(0.2_r), Real3{0.3_r, -0.4_r, 0.5_r}};
  TransformRT const delta{Quaternion::RotationX(0.25_r), Real3{-0.2_r, 0.3_r, 0.1_r}};
  TransformRT const editedWorld = delta * (anchor * local);
  BotIKGoalConfig goalConfig;
  SetIKGoalConstraintTransform(goalConfig, Invert(anchor) * editedWorld);
  EXPECT_TRUE(NearEqual(anchor * IKGoalConstraintTransform(goalConfig), editedWorld, 2e-5_r));
  EXPECT_FALSE(
      NearEqual(anchor * IKGoalConstraintTransform(goalConfig), anchor * delta * local, 2e-5_r));
}

TEST(IKConfigTest, Parse_TrailingContentAndNegativeUnderflowPreserveDestination) {
  auto const original = Config();
  auto out = original;
  DynamicString diagnostic;
  for (DynamicString const suffix : {" trailing", " {}", " null"}) {
    EXPECT_FALSE(ParseIKConfig(SerializeIKConfig(original) + suffix, out, diagnostic));
    EXPECT_EQ(SerializeIKConfig(out), SerializeIKConfig(original));
  }
  EXPECT_TRUE(ParseIKConfig(SerializeIKConfig(original) + " \n\t\r", out, diagnostic));
  for (auto const* field : {"absTol", "relTol", "positionErrorThres", "rotationErrorThres"}) {
    auto value = Json(original);
    value.get<picojson::object>()
        .at("createParams")
        .get<picojson::object>()
        .at("solverParams")
        .get<picojson::object>()[field] = picojson::value(-1e-100);
    EXPECT_FALSE(ParseIKConfig(value.serialize(), out, diagnostic));
    EXPECT_EQ(SerializeIKConfig(out), SerializeIKConfig(original));
  }
  for (auto const* field : {"positionStiffness", "rotationStiffness"}) {
    auto value = Json(original);
    value.get<picojson::object>()
        .at("goals")
        .get<picojson::array>()[0]
        .get<picojson::object>()[field] = picojson::value(-1e-100);
    EXPECT_FALSE(ParseIKConfig(value.serialize(), out, diagnostic));
    EXPECT_EQ(SerializeIKConfig(out), SerializeIKConfig(original));
  }
}

TEST(IKConfigTest, Parse_MigratesLegacyRotationVectorAndWritesVersionNine) {
  BotIKConfig config;
  DynamicString diagnostic;
  ASSERT_TRUE(ParseIKConfig(
      R"({"version":1,"bot":"robot.superdex_bot","targets":[{"name":"tip","link":"tip","position":[0.2,-0.3,0.4],"rotation":[0.1,0.2,-0.1]}]})",
      config,
      diagnostic))
      << diagnostic;
  EXPECT_EQ(config.version, 9);
  EXPECT_TRUE(config.outputBotConfiguration.empty());
  EXPECT_EQ(config.inputBotScale, 1_r);
  ASSERT_EQ(config.goals.size(), 1);
  EXPECT_TRUE(NearEqual(
      config.goals[0].baseOrInputLinkFrameFromConstraintFrame,
      Config().goals[0].baseOrInputLinkFrameFromConstraintFrame));
  auto const fields = Json(config)
                          .get<picojson::object>()
                          .at("goals")
                          .get<picojson::array>()[0]
                          .get<picojson::object>();
  EXPECT_EQ(fields.count("position"), 0);
  EXPECT_EQ(fields.count("rotation"), 0);
  EXPECT_EQ(fields.count("baseOrInputLinkFrameFromConstraintFrame"), 1);
  auto const before = SerializeIKConfig(config);
  for (
      auto const* invalid :
      {R"({"version":1,"bot":"x","targets":[{"name":"tip","link":"tip","position":[1,2]}]})",
       R"({"version":1,"bot":"x","targets":[{"name":"tip","link":"tip","rotation":[0,"1",2]}]})",
       R"({"version":1,"bot":"x","targets":[{"name":"tip","link":"tip","position":[0,0,0],"targetFrame":{}}]})",
       R"({"version":2,"bot":"x","targets":[{"name":"tip","link":"tip","position":[0,0,0]}]})"}) {
    EXPECT_FALSE(ParseIKConfig(invalid, config, diagnostic));
    EXPECT_EQ(SerializeIKConfig(config), before);
  }
}

TEST(IKConfigTest, InputBotFK_ExplicitPlacementOffsetsAndUnboundBaseFrame) {
  auto bot = Robot();
  TransformRT const root{Quaternion::RotationX(0.2_r), Real3{-1_r, 0.5_r, 2_r}};
  TransformRT const base{Quaternion::RotationY(-0.3_r), Real3{2_r, 1_r, -1_r}};
  TransformRT const offset{Quaternion::RotationY(0.15_r), Real3{0.1_r, 0.2_r, -0.1_r}};
  DynamicArray<TransformRT> links;
  DynamicArray<real> const pose{-0.4_r};
  ComputeIKInputBotTransforms(bot, pose, root, links, ExpectOK{});
  auto const tip = root * TransformRT{Quaternion::RotationZ(-0.4_r), Real3{0.3_r, 0_r, 0_r}} *
      TransformRT{Real3{0.7_r, 0_r, 0_r}};
  ASSERT_EQ(links.size(), bot.links.size());
  EXPECT_TRUE(NearEqual(links[2], tip));
  BotIKGoalConfig goalConfig;
  goalConfig.inputBotLink = "tip";
  goalConfig.baseOrInputLinkFrameFromConstraintFrame = offset;
  TransformRT goal;
  ComputeIKGoalWorldConstraintTransform(goalConfig, base, &bot, links, goal, ExpectOK{});
  EXPECT_TRUE(NearEqual(goal, tip * offset));
  goalConfig.inputBotLink.clear();
  ComputeIKGoalWorldConstraintTransform(goalConfig, base, nullptr, {}, goal, ExpectOK{});
  EXPECT_TRUE(NearEqual(goal, base * offset));
  auto const original = links;
  ComputeIKInputBotTransforms(bot, DynamicArray<real>{1_r, 2_r}, root, links, ExpectNotOK{});
  EXPECT_EQ(links, original);
  ComputeIKInputBotTransforms(
      bot, DynamicArray<real>{std::numeric_limits<real>::quiet_NaN()}, root, links, ExpectNotOK{});
  EXPECT_EQ(links, original);
  goalConfig.inputBotLink = "missing";
  auto const previousGoal = goal;
  ComputeIKGoalWorldConstraintTransform(goalConfig, base, &bot, links, goal, ExpectNotOK{});
  EXPECT_EQ(goal, previousGoal);
  ComputeIKGoalWorldConstraintTransform(goalConfig, base, nullptr, links, goal, ExpectNotOK{});
  EXPECT_EQ(goal, previousGoal);
  bot.joints[0].type = ArticulatedJointType::Free;
  RebuildBotData(bot, ExpectOK{});
  ComputeIKInputBotTransforms(bot, {}, root, links, ExpectOK{});
  EXPECT_TRUE(NearEqual(links[0], root));
  EXPECT_TRUE(NearEqual(
      links[2],
      root * TransformRT{Quaternion::RotationZ(0.35_r), Real3{0.3_r, 0_r, 0_r}} *
          TransformRT{Real3{0.7_r, 0_r, 0_r}}));
}

TEST(IKConfigTest, Validate_InputBotFramesAndConfigurationAreStrict) {
  auto spec = Config();
  DynamicString diagnostic;
  spec.goals[0].inputBotLink = "tip";
  EXPECT_FALSE(ValidateIKConfig(spec, diagnostic));
  spec.inputBot = "reference.superdex_bot";
  EXPECT_TRUE(ValidateIKConfig(spec, diagnostic));
  SetInvalidRotation(spec.outputBaseFrameFromInputBaseFrame, Quaternion::Zero());
  EXPECT_FALSE(ValidateIKConfig(spec, diagnostic));
  spec.outputBaseFrameFromInputBaseFrame = {};
  spec.inputBotConfiguration = {std::numeric_limits<real>::infinity()};
  EXPECT_FALSE(ValidateIKConfig(spec, diagnostic));
  spec.inputBotConfiguration.clear();
  SetInvalidRotation(
      spec.goals[0].baseOrInputLinkFrameFromConstraintFrame, Quaternion{0_r, 0_r, 0_r, 2_r});
  EXPECT_FALSE(ValidateIKConfig(spec, diagnostic));
}

TEST(IKConfigTest, Parse_ScaleDefaultsMigrationAndOlderSchemaScaleRejection) {
  auto const initial = Config();
  DynamicString diagnostic;
  for (int const version : {1, 2, 3, 4}) {
    auto value = LegacyJson(initial);
    auto& fields = value.get<picojson::object>();
    fields["version"] = picojson::value(int64_t(version));
    fields.erase("targetBotScale");
    fields.erase("botConfiguration");
    fields.erase("jointCouplings");
    auto loaded = initial;
    loaded.inputBotScale = 2_r;
    ASSERT_TRUE(ParseIKConfig(value.serialize(), loaded, diagnostic)) << diagnostic;
    EXPECT_EQ(loaded.version, 9);
    EXPECT_TRUE(loaded.outputBotConfiguration.empty());
    EXPECT_EQ(loaded.inputBotScale, 1_r);
    EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
    if (version < 3) {
      for (auto const& scale : {picojson::value(1.0), picojson::value(2.0), picojson::value{}}) {
        fields["targetBotScale"] = scale;
        EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
        EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
      }
    }
  }
}

TEST(IKConfigTest, Parse_VersionThreeMigrationPreservesAuthoredRetargeting) {
  auto expected = Config();
  expected.inputBot = "reference.superdex_bot";
  expected.inputBotScale = 2.5_r;
  expected.outputBaseFrameFromInputBaseFrame =
      TransformRT{Quaternion::RotationY(0.4_r), Real3{1_r, -2_r, 3_r}};
  expected.inputBotConfiguration = {0.2_r, -0.5_r};
  expected.createParams.fixedBase = false;
  expected.createParams.solverParams.maxElapsedTimeSeconds = 0.025;
  expected.goals[0].inputBotLink = "reference_tip";
  expected.goals[0].linkFrameFromConstraintFrame = TransformRT{Real3{0.1_r, 0.2_r, 0.3_r}};
  expected.goals[0].enabled = false;
  auto value = LegacyJson(expected);
  auto& fields = value.get<picojson::object>();
  fields["version"] = picojson::value(int64_t(3));
  fields.erase("botConfiguration");
  fields.erase("jointCouplings");
  auto loaded = Config();
  loaded.outputBotConfiguration = {0.7_r};
  DynamicString diagnostic;
  ASSERT_TRUE(ParseIKConfig(value.serialize(), loaded, diagnostic)) << diagnostic;
  EXPECT_TRUE(diagnostic.empty());
  EXPECT_EQ(loaded.version, 9);
  EXPECT_TRUE(loaded.outputBotConfiguration.empty());
  EXPECT_EQ(loaded.inputBotScale, expected.inputBotScale);
  EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(expected));
}

TEST(IKConfigTest, Parse_RecordingReferencesRejectOlderSchemasAndPreserveDestination) {
  auto initial = Config();
  initial.inputBot = "reference.superdex_bot";
  auto loaded = initial;
  DynamicString diagnostic;
  for (int const version : {1, 2, 3, 4}) {
    for (auto const& reference :
         {picojson::value(""), picojson::value("recording.rrd"), picojson::value{}}) {
      auto const value = picojson::value(
          picojson::object{
              {"version", picojson::value(int64_t(version))},
              {"bot", picojson::value("robot.superdex_bot")},
              {"targetBot", picojson::value("reference.superdex_bot")},
              {"targetBotRecording", reference}});
      SCOPED_TRACE(value.serialize());
      EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
      EXPECT_NE(
          diagnostic.find("targetBotRecording requires IK configuration version 5"),
          std::string::npos);
      EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
    }
  }
}

TEST(IKConfigTest, Parse_LegacyRecordingReferenceRequiresStringWithoutCoercion) {
  auto initial = Config();
  initial.inputBot = "reference.superdex_bot";
  auto loaded = initial;
  DynamicString diagnostic;
  for (int const version : {5, 6}) {
    for (auto const& invalid :
         {picojson::value{},
          picojson::value(true),
          picojson::value(1.0),
          picojson::value(picojson::array{}),
          picojson::value(picojson::object{})}) {
      auto value = LegacyJson(initial);
      auto& fields = value.get<picojson::object>();
      fields["version"] = picojson::value(int64_t(version));
      fields.erase("jointCouplings");
      fields["targetBotRecording"] = invalid;
      SCOPED_TRACE(value.serialize());
      EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
      EXPECT_NE(diagnostic.find("targetBotRecording: expected a string"), std::string::npos);
      EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
    }
  }
}

TEST(IKConfigTest, Parse_LegacyRecordingIsDroppedAndNeverSerializedOrSaved) {
  auto cleanup = CreateTempDirectory("ik_recording_migration_test", ExpectOK{});
  std::string const path = (cleanup.Path() / "goals.superdex_ik").generic_string();
  DynamicString diagnostic;
  for (int const version : {5, 6}) {
    for (auto const* recording : {"", "../recordings/old.rrd"}) {
      for (bool const hasInputBot : {false, true}) {
        auto expected = Config();
        expected.outputBotConfiguration = {0.4_r, -0.2_r};
        if (hasInputBot) {
          expected.inputBot = "reference.superdex_bot";
          expected.inputBotConfiguration = {-0.5_r};
          expected.goals[0].inputBotLink = "reference_tip";
        }
        expected.inputBotScale = 2.5_r;
        expected.outputBaseFrameFromInputBaseFrame =
            TransformRT{Quaternion::RotationY(0.4_r), Real3{1_r, -2_r, 3_r}};
        if (version == 6) {
          BotIKJointCouplingConfig coupling;
          coupling.jointA = "PIP";
          coupling.jointB = "DIP";
          expected.jointCouplings = {coupling};
        }
        auto value = LegacyJson(expected);
        auto& fields = value.get<picojson::object>();
        fields["version"] = picojson::value(int64_t(version));
        if (version == 5) {
          fields.erase("jointCouplings");
        }
        fields["targetBotRecording"] = picojson::value(recording);
        BotIKConfig loaded;
        ASSERT_TRUE(ParseIKConfig(value.serialize(), loaded, diagnostic)) << diagnostic;
        EXPECT_NE(diagnostic.find("targetBotRecording"), std::string::npos);
        EXPECT_NE(diagnostic.find("ignored"), std::string::npos);
        EXPECT_EQ(loaded.version, 9);
        EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(expected));
        EXPECT_EQ(Json(loaded).get<picojson::object>().count("targetBotRecording"), 0);
        ASSERT_TRUE(SaveIKConfig(path, loaded, diagnostic)) << diagnostic;
        EXPECT_EQ(ReadFileString(path, ExpectOK{}), std::string(SerializeIKConfig(expected)));
        BotIKConfig reloaded;
        ASSERT_TRUE(LoadIKConfig(path, reloaded, diagnostic)) << diagnostic;
        EXPECT_TRUE(diagnostic.empty());
        EXPECT_EQ(SerializeIKConfig(reloaded), SerializeIKConfig(expected));
      }
    }
  }
}

TEST(IKConfigTest, Parse_RecordingCompatibilityDoesNotRelaxUnknownFields) {
  auto const initial = Config();
  auto loaded = initial;
  DynamicString diagnostic;
  for (auto const& recording :
       {picojson::value(""), picojson::value("old.rrd"), picojson::value{}}) {
    auto value = Json(initial);
    value.get<picojson::object>()["targetBotRecording"] = recording;
    EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
    EXPECT_NE(diagnostic.find("targetBotRecording: unknown field"), std::string::npos);
    EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
  }
  for (int const version : {5, 6}) {
    auto value = LegacyJson(initial);
    auto& fields = value.get<picojson::object>();
    fields["version"] = picojson::value(int64_t(version));
    fields.erase("jointCouplings");
    fields["targetBotRecording"] = picojson::value("old.rrd");
    fields["unrecognized"] = picojson::value(1.0);
    EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
    EXPECT_NE(diagnostic.find("unrecognized: unknown field"), std::string::npos);
    EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
    fields.erase("unrecognized");
    fields.at("targets").get<picojson::array>()[0].get<picojson::object>()["targetBotRecording"] =
        picojson::value("nested.rrd");
    EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
    EXPECT_NE(diagnostic.find("targets[0].targetBotRecording: unknown field"), std::string::npos);
    EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
  }
}

TEST(IKConfigTest, Parse_LegacyRecordingWarningRetainsDuplicateBindingDiagnostics) {
  auto expected = Config();
  expected.goals.push_back(BotIKGoalConfig{expected.goals[0]});
  auto value = LegacyJson(expected);
  auto& fields = value.get<picojson::object>();
  fields["version"] = picojson::value(int64_t(6));
  fields["targetBotRecording"] = picojson::value("old.rrd");
  BotIKConfig loaded;
  DynamicString diagnostic;
  ASSERT_TRUE(ParseIKConfig(value.serialize(), loaded, diagnostic)) << diagnostic;
  EXPECT_NE(diagnostic.find("targetBotRecording"), std::string::npos);
  EXPECT_NE(diagnostic.find("Duplicate goal name"), std::string::npos);
  EXPECT_NE(diagnostic.find("Duplicate goal link"), std::string::npos);
  EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(expected));
  EXPECT_FALSE(ValidateIKConfig(loaded, diagnostic));
}

TEST(IKConfigTest, Parse_VersionFourMigrationPreservesAuthoredPose) {
  auto expected = Config();
  expected.outputBotConfiguration = {0.4_r, -0.2_r};
  expected.inputBot = "reference.superdex_bot";
  expected.inputBotConfiguration = {-0.5_r};
  expected.inputBotScale = 2.5_r;
  expected.outputBaseFrameFromInputBaseFrame =
      TransformRT{Quaternion::RotationY(0.4_r), Real3{1_r, -2_r, 3_r}};
  expected.createParams.fixedBase = false;
  expected.goals[0].inputBotLink = "reference_tip";
  auto value = LegacyJson(expected);
  auto& fields = value.get<picojson::object>();
  fields["version"] = picojson::value(int64_t(4));
  fields.erase("jointCouplings");
  auto loaded = expected;
  DynamicString diagnostic;
  ASSERT_TRUE(ParseIKConfig(value.serialize(), loaded, diagnostic)) << diagnostic;
  EXPECT_EQ(loaded.version, 9);
  EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(expected));
}

TEST(IKConfigTest, Parse_VersionFourEmptyAndIntegerOutputArraysRoundTrip) {
  auto loaded = Config();
  loaded.outputBotConfiguration = {0.7_r};
  DynamicString diagnostic;
  ASSERT_TRUE(ParseIKConfig(
      R"({"version":4,"bot":"robot.superdex_bot","botConfiguration":[]})", loaded, diagnostic))
      << diagnostic;
  EXPECT_TRUE(loaded.outputBotConfiguration.empty());
  auto defaults = Config();
  defaults.goals.clear();
  EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(defaults));
  ASSERT_TRUE(ParseIKConfig(
      R"({"version":4,"bot":"robot.superdex_bot","botConfiguration":[0,-1,2]})",
      loaded,
      diagnostic))
      << diagnostic;
  DynamicArray<real> const expected{0_r, -1_r, 2_r};
  EXPECT_EQ(loaded.outputBotConfiguration, expected);
  BotIKConfig roundTrip;
  ASSERT_TRUE(ParseIKConfig(SerializeIKConfig(loaded), roundTrip, diagnostic)) << diagnostic;
  EXPECT_EQ(roundTrip.outputBotConfiguration, expected);
}

TEST(IKConfigTest, Parse_LegacyOutputConfigurationIsRejectedEvenWhenEmpty) {
  auto initial = Config();
  initial.outputBotConfiguration = {0.4_r, -0.2_r};
  auto loaded = initial;
  DynamicString diagnostic;
  for (int const version : {1, 2, 3}) {
    for (auto const& configuration :
         {picojson::value(picojson::array{}),
          picojson::value(picojson::array{picojson::value(0.2)}),
          picojson::value{}}) {
      auto value = LegacyJson(Config());
      auto& fields = value.get<picojson::object>();
      fields["version"] = picojson::value(int64_t(version));
      fields.erase("targetBotScale");
      fields.erase("jointCouplings");
      fields["botConfiguration"] = configuration;
      SCOPED_TRACE(value.serialize());
      EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
      EXPECT_NE(diagnostic.find("botConfiguration requires"), std::string::npos);
      EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
    }
  }
}

TEST(IKConfigTest, Parse_JointConfigurationArraysRejectMalformedValuesWithoutMutation) {
  auto initial = Config();
  initial.outputBotConfiguration = {0.4_r, -0.2_r};
  initial.inputBotConfiguration = {-0.5_r};
  auto loaded = initial;
  DynamicString diagnostic;
  for (auto const* field : {"outputBotConfiguration", "inputBotConfiguration"}) {
    for (auto const& bad :
         {picojson::value("2"),
          picojson::value(true),
          picojson::value{},
          picojson::value(2.0),
          picojson::value(picojson::object{}),
          picojson::value(picojson::array{picojson::value(0.1), picojson::value("2")}),
          picojson::value(picojson::array{picojson::value(0.1), picojson::value(true)}),
          picojson::value(picojson::array{picojson::value(0.1), picojson::value{}}),
          picojson::value(picojson::array{picojson::value(picojson::array{})}),
          picojson::value(picojson::array{picojson::value(picojson::object{})})}) {
      auto value = Json(initial);
      value.get<picojson::object>()[field] = bad;
      SCOPED_TRACE(value.serialize());
      EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
      EXPECT_NE(diagnostic.find(field), std::string::npos);
      EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
    }
    for (auto const* bad : {"1e400", "-1e400", "NaN", "Infinity", "-Infinity"}) {
      std::string const text =
          std::string(R"({"version":9,"outputBot":"x",")") + field + "\":[0," + bad + "]}";
      SCOPED_TRACE(text);
      EXPECT_FALSE(ParseIKConfig(text, loaded, diagnostic));
      EXPECT_FALSE(diagnostic.empty());
      EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
    }
  }
}

TEST(IKConfigTest, Validate_NonfiniteJointConfigurationsCannotOverwriteSavedDocument) {
  auto initial = Config();
  initial.outputBotConfiguration = {0.4_r, -0.2_r};
  auto cleanup = CreateTempDirectory("ik_joint_config_test", ExpectOK{});
  std::string const path = (cleanup.Path() / "goals.superdex_ik").generic_string();
  DynamicString diagnostic;
  ASSERT_TRUE(SaveIKConfig(path, initial, diagnostic)) << diagnostic;
  auto const saved = ReadFileString(path, ExpectOK{});
  for (bool const output : {true, false}) {
    for (real const bad :
         {std::numeric_limits<real>::quiet_NaN(),
          std::numeric_limits<real>::infinity(),
          -std::numeric_limits<real>::infinity()}) {
      auto invalid = initial;
      auto& positions = output ? invalid.outputBotConfiguration : invalid.inputBotConfiguration;
      positions = {0_r, bad};
      EXPECT_FALSE(ValidateIKConfig(invalid, diagnostic));
      EXPECT_NE(
          diagnostic.find(output ? "outputBotConfiguration" : "inputBotConfiguration"),
          std::string::npos);
      EXPECT_FALSE(SaveIKConfig(path, invalid, diagnostic));
      EXPECT_EQ(ReadFileString(path, ExpectOK{}), saved);
    }
  }
}

TEST(IKConfigTest, Scale_InvalidTypesRangesAndNonfiniteValuesPreserveDestinations) {
  auto const initial = Config();
  auto out = initial;
  DynamicString diagnostic;
  for (auto const& bad :
       {picojson::value("2"),
        picojson::value(true),
        picojson::value{},
        picojson::value(picojson::array{}),
        picojson::value(picojson::object{}),
        picojson::value(0.0),
        picojson::value(-1.0),
        picojson::value(-1e-100)}) {
    auto value = Json(initial);
    value.get<picojson::object>()["inputBotScale"] = bad;
    SCOPED_TRACE(value.serialize());
    EXPECT_FALSE(ParseIKConfig(value.serialize(), out, diagnostic));
    EXPECT_NE(diagnostic.find("inputBotScale"), std::string::npos);
    EXPECT_EQ(SerializeIKConfig(out), SerializeIKConfig(initial));
  }
  auto cleanup = CreateTempDirectory("ik_scale_test", ExpectOK{});
  std::string const path = (cleanup.Path() / "goals.superdex_ik").generic_string();
  ASSERT_TRUE(SaveIKConfig(path, initial, diagnostic));
  auto const saved = ReadFileString(path, ExpectOK{});
  for (real const bad :
       {0_r, -1_r, std::numeric_limits<real>::infinity(), std::numeric_limits<real>::quiet_NaN()}) {
    auto invalid = initial;
    invalid.inputBotScale = bad;
    EXPECT_FALSE(ValidateIKConfig(invalid, diagnostic));
    EXPECT_FALSE(SaveIKConfig(path, invalid, diagnostic));
    EXPECT_EQ(ReadFileString(path, ExpectOK{}), saved);
  }
  for (real const scale : {0.5_r, 2_r, std::numeric_limits<real>::min()}) {
    auto valid = initial;
    valid.inputBotScale = scale;
    ASSERT_TRUE(ParseIKConfig(SerializeIKConfig(valid), out, diagnostic)) << diagnostic;
    EXPECT_EQ(out.inputBotScale, scale);
  }
  auto const before = SerializeIKConfig(out);
  EXPECT_FALSE(
      ParseIKConfig(R"({"version":4,"bot":"x","targetBotScale":1e-400})", out, diagnostic));
  EXPECT_EQ(SerializeIKConfig(out), before);
  EXPECT_FALSE(ParseIKConfig(R"({"version":4,"bot":"x","targetBotScale":1e400})", out, diagnostic));
  EXPECT_EQ(SerializeIKConfig(out), before);
}

TEST(IKConfigTest, InputBotFK_ScalesNativeRevoluteAndPrismaticTranslationsBeforeRotatedRoot) {
  auto bot = Robot();
  bot.scale = 1.7_r;
  bot.joints[2].type = ArticulatedJointType::Prismatic;
  bot.joints[2].axis = {1_r, 0_r, 0_r};
  bot.joints[2].parentLinkFromJoint =
      TransformRT{Quaternion::RotationY(0.3_r), Real3{0.7_r, 0_r, 0_r}};
  bot.defaultPose = {0.35_r, 0.2_r};
  RebuildBotData(bot, ExpectOK{});
  auto const original = bot;
  TransformRT const root{
      Quaternion::RotationX(-0.4_r) * Quaternion::RotationY(0.2_r), Real3{-2_r, 3_r, 1_r}};
  DynamicArray<real> const pose{-0.45_r, 0.3_r};
  TransformRT const arm{Quaternion::RotationZ(pose[0]), Real3{0.3_r, 0_r, 0_r}};
  TransformRT const slide =
      bot.joints[2].parentLinkFromJoint * TransformRT{Real3{pose[1], 0_r, 0_r}};
  TransformRT const nativeTip = arm * slide;
  DynamicArray<TransformRT> links;
  for (real const scale : {0.5_r, 1_r, 2_r}) {
    ComputeIKInputBotTransforms(bot, pose, root, scale, links, ExpectOK{});
    EXPECT_TRUE(NearEqual(links[0], root));
    EXPECT_TRUE(NearEqual(
        links[2],
        root *
            TransformRT{nativeTip.GetRotation(), nativeTip.GetTranslation() * (bot.scale * scale)},
        2e-5_r));
    EXPECT_TRUE(NearEqual(links[2].GetRotation(), (root * nativeTip).GetRotation()));
    if (scale == 1_r) {
      DynamicArray<TransformRT> legacy;
      ComputeIKInputBotTransforms(bot, pose, root, legacy, ExpectOK{});
      EXPECT_EQ(links, legacy);
    }
  }
  ComputeIKInputBotTransforms(bot, {}, root, 0.5_r, links, ExpectOK{});
  TransformRT const defaultTip =
      TransformRT{Quaternion::RotationZ(0.35_r), Real3{0.3_r, 0_r, 0_r}} *
      bot.joints[2].parentLinkFromJoint * TransformRT{Real3{0.2_r, 0_r, 0_r}};
  EXPECT_TRUE(NearEqual(
      links[2],
      root * TransformRT{defaultTip.GetRotation(), defaultTip.GetTranslation() * 0.85_r},
      2e-5_r));
  EXPECT_EQ(bot.scale, original.scale);
  EXPECT_EQ(bot.worldFromRoot, original.worldFromRoot);
  EXPECT_EQ(bot.defaultPose, original.defaultPose);
  EXPECT_EQ(bot.links, original.links);
  EXPECT_EQ(bot.joints, original.joints);
}

TEST(IKConfigTest, GoalGoal_ScaledOffsetsInverseEditsReanchoringAndClearingPreserveWorldPose) {
  auto bot = Robot();
  bot.scale = 1.7_r;
  TransformRT const base{Quaternion::RotationY(-0.3_r), Real3{2_r, 1_r, -1_r}};
  TransformRT const root{Quaternion::RotationX(0.2_r), Real3{-1_r, 0.5_r, 2_r}};
  TransformRT const local{Quaternion::RotationY(0.15_r), Real3{0.1_r, 0.2_r, -0.1_r}};
  TransformRT const delta{Quaternion::RotationZ(-0.25_r), Real3{0.3_r, -0.1_r, 0.2_r}};
  for (real const scale : {0.5_r, 2_r}) {
    DynamicArray<TransformRT> links;
    ComputeIKInputBotTransforms(bot, {}, root, scale, links, ExpectOK{});
    BotIKGoalConfig goalConfig;
    goalConfig.inputBotLink = "tip";
    goalConfig.baseOrInputLinkFrameFromConstraintFrame = local;
    TransformRT goal;
    ComputeIKGoalWorldConstraintTransform(goalConfig, base, &bot, links, scale, goal, ExpectOK{});
    TransformRT const expected =
        links[2] * TransformRT{local.GetRotation(), local.GetTranslation() * scale};
    EXPECT_TRUE(NearEqual(goal, expected, 2e-5_r));
    EXPECT_TRUE(NearEqual(
        IKGoalLocalConstraintTransform(goalConfig, links[2], goal, scale), local, 2e-5_r));
    TransformRT const edited = delta * expected;
    SetIKGoalConstraintTransform(
        goalConfig, IKGoalLocalConstraintTransform(goalConfig, links[2], edited, scale));
    ComputeIKGoalWorldConstraintTransform(goalConfig, base, &bot, links, scale, goal, ExpectOK{});
    EXPECT_TRUE(NearEqual(goal, edited, 2e-5_r));
    goalConfig.inputBotLink = "weld";
    SetIKGoalConstraintTransform(
        goalConfig, IKGoalLocalConstraintTransform(goalConfig, links[3], edited, scale));
    ComputeIKGoalWorldConstraintTransform(goalConfig, base, &bot, links, scale, goal, ExpectOK{});
    EXPECT_TRUE(NearEqual(goal, edited, 2e-5_r));
    SetIKGoalConstraintTransform(
        goalConfig, IKGoalLocalConstraintTransform(goalConfig, links[3], links[3], scale));
    EXPECT_TRUE(
        NearEqual(goalConfig.baseOrInputLinkFrameFromConstraintFrame, TransformRT{}, 2e-5_r));
    goalConfig.inputBotLink.clear();
    SetIKGoalConstraintTransform(
        goalConfig, IKGoalLocalConstraintTransform(goalConfig, base, edited, scale));
    EXPECT_TRUE(NearEqual(
        goalConfig.baseOrInputLinkFrameFromConstraintFrame, Invert(base) * edited, 2e-5_r));
    ComputeIKGoalWorldConstraintTransform(goalConfig, base, nullptr, {}, scale, goal, ExpectOK{});
    EXPECT_TRUE(NearEqual(goal, edited, 2e-5_r));
    goalConfig.baseOrInputLinkFrameFromConstraintFrame = local;
    ComputeIKGoalWorldConstraintTransform(goalConfig, base, nullptr, {}, scale, goal, ExpectOK{});
    EXPECT_TRUE(NearEqual(goal, base * local, 2e-5_r));
    TransformRT legacy;
    ComputeIKGoalWorldConstraintTransform(goalConfig, base, nullptr, {}, legacy, ExpectOK{});
    EXPECT_EQ(goal, legacy);
  }
}

TEST(IKConfigTest, GoalScale_InvalidInputAndOverflowPreserveHelperOutputs) {
  auto bot = Robot();
  DynamicArray<TransformRT> const previousLinks{TransformRT{Real3{7_r, 8_r, 9_r}}};
  TransformRT const previousGoal{Quaternion::RotationY(0.2_r), Real3{4_r, 5_r, 6_r}};
  DynamicArray<TransformRT> links = previousLinks;
  TransformRT goal = previousGoal;
  BotIKGoalConfig goalConfig;
  goalConfig.inputBotLink = "tip";
  DynamicArray<TransformRT> anchors;
  ComputeIKInputBotTransforms(bot, {}, {}, anchors, ExpectOK{});
  for (real const bad :
       {0_r, -1_r, std::numeric_limits<real>::infinity(), std::numeric_limits<real>::quiet_NaN()}) {
    ComputeIKInputBotTransforms(bot, {}, {}, bad, links, ExpectNotOK{});
    EXPECT_EQ(links, previousLinks);
    ComputeIKGoalWorldConstraintTransform(goalConfig, {}, &bot, anchors, bad, goal, ExpectNotOK{});
    EXPECT_EQ(goal, previousGoal);
    goalConfig.inputBotLink.clear();
    ComputeIKGoalWorldConstraintTransform(goalConfig, {}, nullptr, {}, bad, goal, ExpectNotOK{});
    EXPECT_EQ(goal, previousGoal);
    goalConfig.inputBotLink = "tip";
  }
  bot.scale = std::numeric_limits<real>::max();
  ComputeIKInputBotTransforms(bot, {}, {}, 4_r, links, ExpectNotOK{});
  EXPECT_EQ(links, previousLinks);
  goalConfig.baseOrInputLinkFrameFromConstraintFrame.SetTranslation(
      Real3{std::numeric_limits<real>::max(), 0_r, 0_r});
  ComputeIKGoalWorldConstraintTransform(goalConfig, {}, &bot, anchors, 2_r, goal, ExpectNotOK{});
  EXPECT_EQ(goal, previousGoal);
  goalConfig.baseOrInputLinkFrameFromConstraintFrame = {};
  SetInvalidRotation(anchors[2], Quaternion::Zero());
  ComputeIKGoalWorldConstraintTransform(goalConfig, {}, &bot, anchors, 1_r, goal, ExpectNotOK{});
  EXPECT_EQ(goal, previousGoal);
}

TEST(IKConfigTest, Couplings_RoundTripDefaultsConversionAndSaveLoad) {
  auto spec = Config();
  EXPECT_TRUE(spec.jointCouplings.empty());
  BotIKJointCouplingConfig coupling;
  EXPECT_EQ(coupling.ratio, 1_r);
  EXPECT_EQ(coupling.stiffness, 1_r);
  EXPECT_TRUE(coupling.enabled);
  coupling.jointA = "PIP";
  coupling.jointB = "DIP";
  coupling.ratio = -0.5_r;
  coupling.stiffness = 23_r;
  coupling.enabled = false;
  spec.jointCouplings = {coupling};
  auto const runtime = ToBotIKJointCouplingParams(spec.jointCouplings);
  ASSERT_EQ(runtime.size(), 1);
  EXPECT_EQ(runtime[0].jointA, coupling.jointA);
  EXPECT_EQ(runtime[0].jointB, coupling.jointB);
  EXPECT_EQ(runtime[0].ratio, coupling.ratio);
  EXPECT_EQ(runtime[0].offset, 0_r);
  EXPECT_EQ(runtime[0].stiffness, coupling.stiffness);
  EXPECT_EQ(runtime[0].enabled, coupling.enabled);
  auto cleanup = CreateTempDirectory("ik_coupling_config_test", ExpectOK{});
  std::string const path = (cleanup.Path() / "coupled.superdex_ik").generic_string();
  DynamicString diagnostic;
  ASSERT_TRUE(SaveIKConfig(path, spec, diagnostic)) << diagnostic;
  BotIKConfig loaded;
  ASSERT_TRUE(LoadIKConfig(path, loaded, diagnostic)) << diagnostic;
  EXPECT_EQ(loaded.jointCouplings, spec.jointCouplings);
  EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(spec));
  auto const fields = Json(loaded).get<picojson::object>();
  auto const& entry = fields.at("jointCouplings").get<picojson::array>()[0];
  EXPECT_EQ(entry.get<picojson::object>().size(), 5);
  EXPECT_EQ(entry.get<picojson::object>().count("offset"), 0);
  EXPECT_EQ(fields.at("createParams").get<picojson::object>().count("jointCouplings"), 0);
  auto const saved = ReadFileString(path, ExpectOK{});
  spec.jointCouplings.push_back(coupling);
  EXPECT_FALSE(SaveIKConfig(path, spec, diagnostic));
  EXPECT_EQ(ReadFileString(path, ExpectOK{}), saved);
  auto value = Json(loaded);
  auto& object = value.get<picojson::object>()
                     .at("jointCouplings")
                     .get<picojson::array>()[0]
                     .get<picojson::object>();
  object.erase("ratio");
  object.erase("stiffness");
  object.erase("enabled");
  ASSERT_TRUE(ParseIKConfig(value.serialize(), loaded, diagnostic)) << diagnostic;
  EXPECT_EQ(loaded.jointCouplings[0].ratio, 1_r);
  EXPECT_EQ(loaded.jointCouplings[0].stiffness, 1_r);
  EXPECT_TRUE(loaded.jointCouplings[0].enabled);
}

TEST(IKConfigTest, Couplings_StrictReflectedTypesAndNativePrecisionPreserveDestination) {
  auto initial = Config();
  BotIKJointCouplingConfig coupling;
  coupling.jointA = "PIP";
  coupling.jointB = "DIP";
  initial.jointCouplings = {coupling};
  auto loaded = initial;
  DynamicString diagnostic;
  for (auto const& bad :
       {picojson::value{},
        picojson::value(true),
        picojson::value(2.0),
        picojson::value("[]"),
        picojson::value(picojson::object{}),
        picojson::value(picojson::array{picojson::value(2.0)}),
        picojson::value(picojson::array{picojson::value{}})}) {
    auto value = Json(initial);
    value.get<picojson::object>()["jointCouplings"] = bad;
    EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
    EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
  }
  for (auto const* field : {"jointA", "jointB", "ratio", "stiffness", "enabled", "offset"}) {
    for (auto const& bad :
         {picojson::value{},
          picojson::value(picojson::array{}),
          picojson::value(picojson::object{})}) {
      auto value = Json(initial);
      value.get<picojson::object>()
          .at("jointCouplings")
          .get<picojson::array>()[0]
          .get<picojson::object>()[field] = bad;
      SCOPED_TRACE(field);
      EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
      EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
    }
  }
  for (auto const* field : {"jointA", "jointB", "enabled"}) {
    for (auto const& bad : {picojson::value(2.0), picojson::value("true")}) {
      if (std::string_view(field) != "enabled" && bad.is<std::string>()) {
        continue;
      }
      auto value = Json(initial);
      value.get<picojson::object>()
          .at("jointCouplings")
          .get<picojson::array>()[0]
          .get<picojson::object>()[field] = bad;
      EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
      EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
    }
  }
  for (auto const* field : {"ratio", "stiffness"}) {
    for (auto const& bad : {picojson::value("2"), picojson::value(true)}) {
      auto value = Json(initial);
      value.get<picojson::object>()
          .at("jointCouplings")
          .get<picojson::array>()[0]
          .get<picojson::object>()[field] = bad;
      EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
      EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
    }
  }
  {
    auto value = Json(initial);
    value.get<picojson::object>()
        .at("jointCouplings")
        .get<picojson::array>()[0]
        .get<picojson::object>()["ratio"] = picojson::value(0.0);
    EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
    EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
  }
  {
    auto value = Json(initial);
    value.get<picojson::object>()
        .at("jointCouplings")
        .get<picojson::array>()[0]
        .get<picojson::object>()["stiffness"] = picojson::value(-1e-100);
    EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
    EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
  }
  for (auto const* field : {"ratio", "stiffness"}) {
    for (auto const* bad : {"1e400", "-1e400", "NaN", "Infinity"}) {
      std::string const text =
          std::string(R"({"version":6,"bot":"x","jointCouplings":[{"jointA":"A","jointB":"B",")") +
          field + "\":" + bad + "}]}";
      EXPECT_FALSE(ParseIKConfig(text, loaded, diagnostic));
      EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
    }
  }
  for (real const ratio : {1e-10_r, -1e-10_r, std::numeric_limits<real>::min()}) {
    auto valid = initial;
    valid.jointCouplings[0].ratio = ratio;
    ASSERT_TRUE(ParseIKConfig(SerializeIKConfig(valid), loaded, diagnostic)) << diagnostic;
    EXPECT_EQ(loaded.jointCouplings[0].ratio, ratio);
  }
}

TEST(IKConfigTest, Parse_RealPrecisionBoundariesAndNarrowingPreserveDestination) {
  auto initial = Config();
  BotIKJointCouplingConfig coupling;
  coupling.jointA = "A";
  coupling.jointB = "B";
  initial.jointCouplings = {coupling};
  DynamicString diagnostic;
  for (real const number : {std::numeric_limits<real>::min(), std::numeric_limits<real>::max()}) {
    auto expected = initial;
    expected.inputBotScale = number;
    expected.outputBotConfiguration = {number, -number};
    expected.inputBotConfiguration = {-number};
    expected.goals[0].baseOrInputLinkFrameFromConstraintFrame.SetTranslation(
        Real3{number, -number, 0_r});
    expected.jointCouplings[0].ratio = -number;
    expected.jointCouplings[0].stiffness = number;
    BotIKConfig loaded;
    ASSERT_TRUE(ParseIKConfig(SerializeIKConfig(expected), loaded, diagnostic)) << diagnostic;
    EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(expected));
  }

  double const aboveFloatMax = std::nextafter(
      double(std::numeric_limits<float>::max()), std::numeric_limits<double>::infinity());
  double const belowFloatMin = std::ldexp(
      double(std::numeric_limits<float>::min()), -std::numeric_limits<float>::digits - 1);
  for (double const number : {aboveFloatMax, -aboveFloatMax, belowFloatMin, -belowFloatMin}) {
    for (auto const* field :
         {"inputBotScale",
          "ratio",
          "stiffness",
          "outputBotConfiguration",
          "inputBotConfiguration",
          "translation"}) {
      auto value = Json(initial);
      auto& fields = value.get<picojson::object>();
      if (std::string_view(field) == "ratio" || std::string_view(field) == "stiffness") {
        fields.at("jointCouplings").get<picojson::array>()[0].get<picojson::object>()[field] =
            picojson::value(number);
      } else if (std::string_view(field) == "translation") {
        fields.at("goals")
            .get<picojson::array>()[0]
            .get<picojson::object>()
            .at("baseOrInputLinkFrameFromConstraintFrame")
            .get<picojson::object>()[field] = picojson::value(
            picojson::array{picojson::value(number), picojson::value(0.0), picojson::value(0.0)});
      } else if (
          std::string_view(field) == "outputBotConfiguration" ||
          std::string_view(field) == "inputBotConfiguration") {
        fields[field] = picojson::value(picojson::array{picojson::value(number)});
      } else {
        fields[field] = picojson::value(number);
      }
      bool const positive =
          std::string_view(field) == "inputBotScale" || std::string_view(field) == "stiffness";
      bool const preserveNonzero = positive || std::string_view(field) == "ratio";
      bool const narrowing = std::is_same_v<real, float>;
      bool const overflow = std::abs(number) > double(std::numeric_limits<real>::max());
      bool const underflow = narrowing && std::abs(number) == belowFloatMin;
      bool const valid = !(positive && number < 0) && !overflow && !(preserveNonzero && underflow);
      auto loaded = initial;
      SCOPED_TRACE(value.serialize());
      ASSERT_EQ(ParseIKConfig(value.serialize(), loaded, diagnostic), valid) << diagnostic;
      if (valid) {
        auto roundTrip = initial;
        ASSERT_TRUE(ParseIKConfig(SerializeIKConfig(loaded), roundTrip, diagnostic)) << diagnostic;
        EXPECT_EQ(roundTrip.outputBotConfiguration, loaded.outputBotConfiguration);
        EXPECT_EQ(roundTrip.inputBotConfiguration, loaded.inputBotConfiguration);
        EXPECT_EQ(
            roundTrip.goals[0].baseOrInputLinkFrameFromConstraintFrame,
            loaded.goals[0].baseOrInputLinkFrameFromConstraintFrame);
        EXPECT_EQ(roundTrip.inputBotScale, loaded.inputBotScale);
        EXPECT_EQ(roundTrip.jointCouplings, loaded.jointCouplings);
        if (std::string_view(field) == "ratio") {
          EXPECT_EQ(loaded.jointCouplings[0].ratio, real(number));
        } else if (std::string_view(field) == "stiffness") {
          EXPECT_EQ(loaded.jointCouplings[0].stiffness, real(number));
        } else if (std::string_view(field) == "inputBotScale") {
          EXPECT_EQ(loaded.inputBotScale, real(number));
        } else if (std::string_view(field) == "outputBotConfiguration") {
          EXPECT_EQ(loaded.outputBotConfiguration, (DynamicArray<real>{real(number)}));
        } else if (std::string_view(field) == "inputBotConfiguration") {
          EXPECT_EQ(loaded.inputBotConfiguration, (DynamicArray<real>{real(number)}));
        } else {
          EXPECT_EQ(
              loaded.goals[0].baseOrInputLinkFrameFromConstraintFrame.GetTranslation(),
              (Real3{real(number), 0_r, 0_r}));
        }
      } else {
        EXPECT_FALSE(diagnostic.empty());
        EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
      }
    }
  }
}

TEST(IKConfigTest, Parse_LegacyPoseCardinalityIsStrictForEveryGoal) {
  auto const initial = Config();
  DynamicString diagnostic;
  for (auto const* field : {"position", "rotation"}) {
    for (int const count : {0, 1, 2, 4}) {
      picojson::array components(count, picojson::value(0.0));
      picojson::value value(
          picojson::object{
              {"version", picojson::value(int64_t(1))},
              {"bot", picojson::value("robot")},
              {"targets",
               picojson::value(
                   picojson::array{
                       picojson::value(
                           picojson::object{
                               {"name", picojson::value("first")},
                               {"link", picojson::value("first")}}),
                       picojson::value(
                           picojson::object{
                               {"name", picojson::value("second")},
                               {"link", picojson::value("second")},
                               {field, picojson::value(components)}})})}});
      auto loaded = initial;
      SCOPED_TRACE(value.serialize());
      EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
      EXPECT_NE(
          diagnostic.find(std::string("targets[1].") + field + ": expected 3 elements"),
          std::string::npos);
      EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
    }
  }
}

TEST(IKConfigTest, Couplings_AllEntriesRequireValidValuesAndUniqueUnorderedPairs) {
  auto const initial = Config();
  auto loaded = initial;
  DynamicString diagnostic;
  for (bool const enabled : {false, true}) {
    for (int fault = 0; fault < 10; ++fault) {
      BotIKJointCouplingConfig coupling;
      coupling.jointA = "PIP";
      coupling.jointB = "DIP";
      coupling.enabled = enabled;
      if (fault == 0) {
        coupling.jointA.clear();
      }
      if (fault == 1) {
        coupling.jointB.clear();
      }
      if (fault == 2) {
        coupling.jointB = coupling.jointA;
      }
      if (fault == 3) {
        coupling.ratio = 0_r;
      }
      if (fault == 4) {
        coupling.ratio = std::numeric_limits<real>::infinity();
      }
      if (fault == 5) {
        coupling.ratio = std::numeric_limits<real>::quiet_NaN();
      }
      if (fault == 6) {
        coupling.stiffness = -1_r;
      }
      if (fault == 7) {
        coupling.stiffness = std::numeric_limits<real>::infinity();
      }
      auto invalid = initial;
      invalid.jointCouplings = {coupling};
      if (fault >= 8) {
        auto duplicate = coupling;
        if (fault == 9) {
          std::swap(duplicate.jointA, duplicate.jointB);
        }
        duplicate.enabled = false;
        duplicate.stiffness = 0_r;
        invalid.jointCouplings.push_back(duplicate);
      }
      SCOPED_TRACE(fault);
      EXPECT_FALSE(ValidateIKConfig(invalid, diagnostic));
      if (IsFinite(coupling.ratio) && IsFinite(coupling.stiffness)) {
        EXPECT_FALSE(ParseIKConfig(SerializeIKConfig(invalid), loaded, diagnostic));
      }
      EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
    }
  }
  auto valid = initial;
  BotIKJointCouplingConfig ab;
  ab.jointA = "A";
  ab.jointB = "B";
  auto bc = ab;
  bc.jointA = "B";
  bc.jointB = "C";
  auto ac = ab;
  ac.jointB = "C";
  valid.jointCouplings = {ab, bc, ac};
  EXPECT_TRUE(ValidateIKConfig(valid, diagnostic)) << diagnostic;
}

TEST(IKConfigTest, Couplings_ModelBindingSameKindUniqueCaseAndStaleInactive) {
  auto bot = Robot();
  bot.joints[2].type = ArticulatedJointType::Revolute;
  bot.joints[2].axis = {0_r, 0_r, 1_r};
  bot.defaultPose = {0.2_r, -0.1_r};
  RebuildBotData(bot, ExpectOK{});
  BotIKJointCouplingConfig coupling;
  coupling.jointA = "turn";
  coupling.jointB = "end";
  DynamicArray<BotIKJointCouplingConfig> couplings{coupling};
  DynamicString diagnostic;
  ASSERT_TRUE(ValidateIKJointCouplings(bot, couplings, diagnostic)) << diagnostic;
  for (auto const* name : {"Turn", "missing", "base", "fixed"}) {
    couplings[0].jointA = name;
    EXPECT_FALSE(ValidateIKJointCouplings(bot, couplings, diagnostic));
    EXPECT_NE(diagnostic.find("jointCouplings[0]"), std::string::npos);
    EXPECT_NE(diagnostic.find(name), std::string::npos);
    EXPECT_NE(diagnostic.find("end"), std::string::npos);
    couplings[0].enabled = false;
    EXPECT_TRUE(ValidateIKJointCouplings(bot, couplings, diagnostic)) << diagnostic;
    couplings[0].enabled = true;
    couplings[0].stiffness = 0_r;
    EXPECT_TRUE(ValidateIKJointCouplings(bot, couplings, diagnostic)) << diagnostic;
    couplings[0].stiffness = 10_r;
  }
  BotIKJointCouplingConfig stale;
  stale.jointA = "stale_A";
  stale.jointB = "stale_B";
  stale.enabled = false;
  couplings = {stale, coupling};
  couplings[1].jointA = "late_missing";
  EXPECT_FALSE(ValidateIKJointCouplings(bot, couplings, diagnostic));
  EXPECT_NE(diagnostic.find("jointCouplings[1] (late_missing, end)"), std::string::npos);
  couplings = {coupling};
  bot.joints[2].type = ArticulatedJointType::Prismatic;
  EXPECT_FALSE(ValidateIKJointCouplings(bot, couplings, diagnostic));
  EXPECT_NE(diagnostic.find("jointCouplings[0] (turn, end)"), std::string::npos);
  BotIK::ValidateJointCouplings(bot, ToBotIKJointCouplingParams(couplings), ExpectOK{});
  bot.joints[1].type = ArticulatedJointType::Prismatic;
  EXPECT_TRUE(ValidateIKJointCouplings(bot, couplings, diagnostic)) << diagnostic;
  bot.joints[2].type = ArticulatedJointType::Spherical;
  EXPECT_FALSE(ValidateIKJointCouplings(bot, couplings, diagnostic));
  bot.joints[2].type = ArticulatedJointType::Prismatic;
  bot.joints[3].name = "turn";
  EXPECT_FALSE(ValidateIKJointCouplings(bot, couplings, diagnostic));
  couplings[0].enabled = false;
  EXPECT_TRUE(ValidateIKJointCouplings(bot, couplings, diagnostic)) << diagnostic;
  couplings.push_back(coupling);
  EXPECT_FALSE(ValidateIKJointCouplings(bot, couplings, diagnostic));
}

TEST(IKConfigTest, Couplings_VersionsOneThroughFiveMigrateEmptyAndRejectNewField) {
  DynamicString diagnostic;
  for (int const version : {1, 2, 3, 4, 5}) {
    auto const minimal = picojson::value(
        picojson::object{
            {"version", picojson::value(int64_t(version))}, {"bot", picojson::value("robot")}});
    auto loaded = Config();
    BotIKJointCouplingConfig stale;
    stale.jointA = "oldA";
    stale.jointB = "oldB";
    loaded.jointCouplings = {stale};
    ASSERT_TRUE(ParseIKConfig(minimal.serialize(), loaded, diagnostic)) << diagnostic;
    EXPECT_EQ(loaded.version, 9);
    EXPECT_TRUE(loaded.jointCouplings.empty());
    auto const before = SerializeIKConfig(loaded);
    for (auto const& field : {picojson::value(picojson::array{}), picojson::value{}}) {
      auto value = minimal;
      value.get<picojson::object>()["jointCouplings"] = field;
      EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
      EXPECT_NE(
          diagnostic.find("jointCouplings requires IK configuration version 6"), std::string::npos);
      EXPECT_EQ(SerializeIKConfig(loaded), before);
    }
  }
  auto expected = Config();
  expected.outputBotConfiguration = {0.2_r, -0.4_r};
  expected.inputBot = "reference";
  expected.inputBotConfiguration = {-0.1_r};
  expected.inputBotScale = 2_r;
  expected.createParams.fixedBase = false;
  expected.createParams.solverParams.maxElapsedTimeSeconds = 0.125;
  auto value = LegacyJson(expected);
  value.get<picojson::object>()["version"] = picojson::value(int64_t(5));
  value.get<picojson::object>().erase("jointCouplings");
  BotIKConfig loaded;
  ASSERT_TRUE(ParseIKConfig(value.serialize(), loaded, diagnostic)) << diagnostic;
  EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(expected));
}

TEST(IKConfigTest, Parse_VersionSevenOldKeysMigrateToCanonicalNine_AllFieldsExact) {
  auto expected = Config();
  expected.outputBotConfiguration = {0.375_r, -0.625_r, 0_r};
  expected.inputBot = "../reference/reference.superdex_bot";
  expected.outputBaseFrameFromInputBaseFrame =
      TransformRT{Quaternion::RotationY(0.3125_r), Real3{1.25_r, -2.5_r, 3.75_r}};
  expected.inputBotScale = 1.75_r;
  expected.inputBotConfiguration = {-0.125_r, 0.25_r, 0.5_r, -0.75_r};
  expected.createParams.fixedBase = false;
  expected.createParams.excludeCollisionShapes = false;
  auto& solver = expected.createParams.solverParams;
  solver.maxIter = 47;
  solver.verbosity = VerbosityLevel::Verbose;
  solver.absTol = 0.003125_r;
  solver.relTol = 0.000125_r;
  solver.positionErrorThres = 0.00625_r;
  solver.rotationErrorThres = 0.0125_r;
  solver.lineSearchMaxIter = 17;
  solver.maxElapsedTimeSeconds = 0.0375;
  auto& goal = expected.goals[0];
  goal.enabled = false;
  goal.inputBotLink = "reference_tip";
  goal.linkFrameFromConstraintFrame =
      TransformRT{Quaternion::RotationX(-0.25_r), Real3{0.125_r, -0.25_r, 0.375_r}};
  goal.positionStiffness = 37.5_r;
  goal.rotationStiffness = 2.25_r;
  auto second = goal;
  second.name = "second_goal";
  second.outputBotLink = "second_tip";
  second.enabled = true;
  second.inputBotLink.clear();
  second.baseOrInputLinkFrameFromConstraintFrame =
      TransformRT{Quaternion::RotationZ(0.5_r), Real3{-1_r, 2_r, -3_r}};
  second.positionStiffness = 0_r;
  expected.goals.push_back(second);
  BotIKJointCouplingConfig coupling;
  coupling.jointA = "PIP";
  coupling.jointB = "DIP";
  coupling.ratio = -0.625_r;
  coupling.stiffness = 12.5_r;
  coupling.enabled = false;
  expected.jointCouplings = {coupling};

  auto const historical = VersionedJson(expected, 7);
  auto const& oldFields = historical.get<picojson::object>();
  EXPECT_EQ(oldFields.count("targets"), 1);
  EXPECT_EQ(oldFields.count("goals"), 0);
  BotIKConfig loaded;
  DynamicString diagnostic;
  ASSERT_TRUE(ParseIKConfig(historical.serialize(), loaded, diagnostic)) << diagnostic;
  EXPECT_TRUE(diagnostic.empty());
  EXPECT_EQ(loaded.version, 9);
  EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(expected));
  EXPECT_EQ(loaded.outputBotConfiguration, expected.outputBotConfiguration);
  EXPECT_EQ(loaded.inputBotConfiguration, expected.inputBotConfiguration);
  EXPECT_EQ(loaded.outputBaseFrameFromInputBaseFrame, expected.outputBaseFrameFromInputBaseFrame);
  EXPECT_EQ(loaded.inputBotScale, expected.inputBotScale);
  EXPECT_EQ(loaded.jointCouplings, expected.jointCouplings);
  ASSERT_EQ(loaded.goals.size(), expected.goals.size());
  for (int i = 0; i < isize(loaded.goals); ++i) {
    EXPECT_EQ(
        loaded.goals[i].baseOrInputLinkFrameFromConstraintFrame,
        expected.goals[i].baseOrInputLinkFrameFromConstraintFrame);
    EXPECT_EQ(
        loaded.goals[i].linkFrameFromConstraintFrame,
        expected.goals[i].linkFrameFromConstraintFrame);
  }
  auto const canonical = Json(loaded);
  auto const& fields = canonical.get<picojson::object>();
  for (auto const* name :
       {"inputBot",
        "outputBaseFrameFromInputBaseFrame",
        "inputBotScale",
        "inputBotConfiguration",
        "goals"}) {
    EXPECT_EQ(fields.count(name), 1) << name;
  }
  for (auto const* name :
       {"targetBot",
        "baseFromTargetBot",
        "targetBotScale",
        "targetBotConfiguration",
        "targets",
        "targetBotRecording",
        "inputBotRecording"}) {
    EXPECT_EQ(fields.count(name), 0) << name;
  }
  for (auto const& item : fields.at("goals").get<picojson::array>()) {
    auto const& entry = item.get<picojson::object>();
    EXPECT_EQ(entry.count("inputBotLink"), 1);
    EXPECT_EQ(entry.count("baseOrInputLinkFrameFromConstraintFrame"), 1);
    EXPECT_EQ(entry.count("targetBotLink"), 0);
    EXPECT_EQ(entry.count("targetFrame"), 0);
  }
  BotIKConfig roundTrip;
  ASSERT_TRUE(ParseIKConfig(SerializeIKConfig(loaded), roundTrip, diagnostic)) << diagnostic;
  EXPECT_EQ(SerializeIKConfig(roundTrip), SerializeIKConfig(expected));
}

TEST(IKConfigTest, Parse_VersionEightRejectsLegacyKeysEvenWithoutCanonicalCounterparts) {
  auto const initial = Config();
  DynamicString diagnostic;
  for (auto const& [canonical, historical] :
       {std::pair{"referenceBot", "targetBot"},
        std::pair{"baseFromReferenceBot", "baseFromTargetBot"},
        std::pair{"referenceBotScale", "targetBotScale"},
        std::pair{"referenceBotConfiguration", "targetBotConfiguration"},
        std::pair{"goals", "targets"}}) {
    auto value = VersionedJson(initial, 8);
    auto& fields = value.get<picojson::object>();
    fields[historical] = std::move(fields.at(canonical));
    fields.erase(canonical);
    auto loaded = initial;
    SCOPED_TRACE(value.serialize());
    EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
    EXPECT_NE(diagnostic.find(std::string(historical) + ": unknown field"), std::string::npos);
    EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
  }
  for (auto const& [canonical, historical] :
       {std::pair{"referenceBotLink", "targetBotLink"}, std::pair{"goalFrame", "targetFrame"}}) {
    auto value = VersionedJson(initial, 8);
    auto& entry =
        value.get<picojson::object>().at("goals").get<picojson::array>()[0].get<picojson::object>();
    entry[historical] = std::move(entry.at(canonical));
    entry.erase(canonical);
    auto loaded = initial;
    EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
    EXPECT_NE(diagnostic.find(std::string(historical) + ": unknown field"), std::string::npos);
    EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
  }
}

TEST(IKConfigTest, Parse_MixedOldAndNewKeysRejectTransactionallyInEveryVersion) {
  auto const initial = Config();
  DynamicString diagnostic;
  for (int const version : {1, 2, 3, 4, 5, 6, 7, 8}) {
    for (auto const& [canonical, historical] :
         {std::pair{"referenceBot", "targetBot"},
          std::pair{"baseFromReferenceBot", "baseFromTargetBot"},
          std::pair{"referenceBotScale", "targetBotScale"},
          std::pair{"referenceBotConfiguration", "targetBotConfiguration"},
          std::pair{"goals", "targets"}}) {
      auto value = VersionedJson(initial, version);
      auto& fields = value.get<picojson::object>();
      auto const sameValue = LegacyJson(initial, 8).get<picojson::object>().at(canonical);
      fields[canonical] = sameValue;
      fields[historical] = sameValue;
      auto loaded = initial;
      SCOPED_TRACE(value.serialize());
      EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
      EXPECT_FALSE(diagnostic.empty());
      EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
    }
    for (auto const& [canonical, historical] :
         {std::pair{"referenceBotLink", "targetBotLink"}, std::pair{"goalFrame", "targetFrame"}}) {
      auto value = VersionedJson(initial, version);
      auto& entry = value.get<picojson::object>()
                        .at(version < 8 ? "targets" : "goals")
                        .get<picojson::array>()[0]
                        .get<picojson::object>();
      auto const sameValue = entry.at(version < 8 ? historical : canonical);
      entry[canonical] = sameValue;
      entry[historical] = sameValue;
      auto loaded = initial;
      SCOPED_TRACE(value.serialize());
      EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
      EXPECT_FALSE(diagnostic.empty());
      EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
    }
  }
}

TEST(IKConfigTest, Parse_AllVersionsTypeCheckContainersAndLateEntriesBeforeMigration) {
  auto const initial = Config();
  DynamicString diagnostic;
  for (int const version : {1, 2, 3, 4, 5, 6, 7, 8, 9}) {
    for (auto const* key :
         {version < 8 ? "targets" : "goals",
          version < 8        ? "targetBotConfiguration"
              : version == 8 ? "referenceBotConfiguration"
                             : "inputBotConfiguration"}) {
      for (auto const& invalid :
           {picojson::value{},
            picojson::value("[]"),
            picojson::value(true),
            picojson::value(1.0),
            picojson::value(picojson::object{}),
            picojson::value(picojson::array{picojson::value{}}),
            picojson::value(picojson::array{picojson::value("invalid")})}) {
        auto value = VersionedJson(initial, version);
        value.get<picojson::object>()[key] = invalid;
        auto loaded = initial;
        SCOPED_TRACE(value.serialize());
        EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
        EXPECT_NE(diagnostic.find(key), std::string::npos);
        EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
      }
    }
    for (auto const* component : {"translation", "rotation"}) {
      for (int const count : {0, 1, 2, 5}) {
        auto value = VersionedJson(initial, version);
        auto& goals = value.get<picojson::object>()
                          .at(version < 8 ? "targets" : "goals")
                          .get<picojson::array>();
        auto late = goals[0];
        late.get<picojson::object>()["name"] = picojson::value("late");
        late.get<picojson::object>()[version < 9 ? "link" : "outputBotLink"] =
            picojson::value("late");
        late.get<picojson::object>()
            .at(version < 8        ? "targetFrame"
                    : version == 8 ? "goalFrame"
                                   : "baseOrInputLinkFrameFromConstraintFrame")
            .get<picojson::object>()[component] =
            picojson::value(picojson::array(count, picojson::value(0.0)));
        goals.push_back(late);
        auto loaded = initial;
        SCOPED_TRACE(value.serialize());
        EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
        EXPECT_NE(
            diagnostic.find(
                std::string(version < 8 ? "targets[1]." : "goals[1].") +
                (version < 8        ? "targetFrame."
                     : version == 8 ? "goalFrame."
                                    : "baseOrInputLinkFrameFromConstraintFrame.") +
                component),
            std::string::npos);
        EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
      }
    }
  }
}

TEST(IKConfigTest, Parse_RecordingAdmissionRemainsExclusiveToLegacyVersionsFiveAndSix) {
  auto const initial = Config();
  DynamicString diagnostic;
  for (int const version : {1, 2, 3, 4, 5, 6, 7, 8, 9}) {
    for (auto const* key : {"targetBotRecording", "referenceBotRecording", "inputBotRecording"}) {
      auto value = VersionedJson(initial, version);
      value.get<picojson::object>()[key] = picojson::value("old.rrd");
      auto loaded = initial;
      bool const admitted =
          (version == 5 || version == 6) && std::string_view(key) == "targetBotRecording";
      SCOPED_TRACE(value.serialize());
      ASSERT_EQ(ParseIKConfig(value.serialize(), loaded, diagnostic), admitted) << diagnostic;
      EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
      if (admitted) {
        EXPECT_NE(diagnostic.find("ignored"), std::string::npos);
      } else {
        EXPECT_FALSE(diagnostic.empty());
      }
    }
  }
}

TEST(IKConfigTest, Parse_VersionEightReferenceSchemaMigratesEveryValueExactlyToNine) {
  auto expected = Config();
  expected.outputBotConfiguration = {0.375_r, -0.625_r, 0_r};
  expected.inputBot = "../reference/reference.superdex_bot";
  expected.inputBotConfiguration = {-0.125_r, 0.25_r, 0.5_r, -0.75_r};
  expected.inputBotScale = 1.75_r;
  expected.outputBaseFrameFromInputBaseFrame =
      TransformRT{Quaternion::RotationY(0.3125_r), Real3{1.25_r, -2.5_r, 3.75_r}};
  expected.createParams.fixedBase = false;
  expected.createParams.excludeCollisionShapes = false;
  auto& solver = expected.createParams.solverParams;
  solver.maxIter = 47;
  solver.verbosity = VerbosityLevel::Verbose;
  solver.absTol = 0.003125_r;
  solver.relTol = 0.000125_r;
  solver.positionErrorThres = 0.00625_r;
  solver.rotationErrorThres = 0.0125_r;
  solver.lineSearchMaxIter = 17;
  solver.maxElapsedTimeSeconds = 0.0375;
  auto& goal = expected.goals[0];
  goal.inputBotLink = "reference_tip";
  goal.enabled = false;
  goal.linkFrameFromConstraintFrame =
      TransformRT{Quaternion::RotationX(-0.25_r), Real3{0.125_r, -0.25_r, 0.375_r}};
  goal.positionStiffness = 37.5_r;
  goal.rotationStiffness = 2.25_r;
  auto second = goal;
  second.name = "second_goal";
  second.outputBotLink = "second_tip";
  second.enabled = true;
  second.inputBotLink.clear();
  second.baseOrInputLinkFrameFromConstraintFrame =
      TransformRT{Quaternion::RotationZ(0.5_r), Real3{-1_r, 2_r, -3_r}};
  second.positionStiffness = 0_r;
  expected.goals.push_back(second);
  BotIKJointCouplingConfig coupling;
  coupling.jointA = "PIP";
  coupling.jointB = "DIP";
  coupling.ratio = -0.625_r;
  coupling.stiffness = 12.5_r;
  coupling.enabled = false;
  expected.jointCouplings = {coupling};

  auto const historical = VersionedJson(expected, 8);
  auto const& fields = historical.get<picojson::object>();
  EXPECT_EQ(fields.count("bot"), 1);
  EXPECT_EQ(fields.count("referenceBot"), 1);
  EXPECT_EQ(fields.count("baseFromReferenceBot"), 1);
  EXPECT_EQ(fields.count("outputBot"), 0);
  EXPECT_EQ(fields.count("inputBot"), 0);
  auto const& entry = fields.at("goals").get<picojson::array>()[0].get<picojson::object>();
  EXPECT_EQ(entry.count("link"), 1);
  EXPECT_EQ(entry.count("linkRelativeFrame"), 1);
  EXPECT_EQ(entry.count("referenceBotLink"), 1);
  EXPECT_EQ(entry.count("goalFrame"), 1);
  EXPECT_EQ(entry.count("outputBotLink"), 0);
  BotIKConfig loaded;
  DynamicString diagnostic;
  ASSERT_TRUE(ParseIKConfig(historical.serialize(), loaded, diagnostic)) << diagnostic;
  EXPECT_TRUE(diagnostic.empty());
  EXPECT_EQ(loaded.version, 9);
  EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(expected));
  EXPECT_EQ(loaded.outputBaseFrameFromInputBaseFrame, expected.outputBaseFrameFromInputBaseFrame);
  ASSERT_EQ(loaded.goals.size(), expected.goals.size());
  for (int i = 0; i < isize(loaded.goals); ++i) {
    EXPECT_EQ(
        loaded.goals[i].linkFrameFromConstraintFrame,
        expected.goals[i].linkFrameFromConstraintFrame);
    EXPECT_EQ(
        loaded.goals[i].baseOrInputLinkFrameFromConstraintFrame,
        expected.goals[i].baseOrInputLinkFrameFromConstraintFrame);
  }
  auto const canonical = Json(loaded);
  for (auto const* key :
       {"bot",
        "botConfiguration",
        "referenceBot",
        "referenceBotConfiguration",
        "referenceBotScale",
        "baseFromReferenceBot",
        "targetBot",
        "targets"}) {
    EXPECT_EQ(canonical.get<picojson::object>().count(key), 0) << key;
  }
  BotIKConfig roundTrip;
  ASSERT_TRUE(ParseIKConfig(SerializeIKConfig(loaded), roundTrip, diagnostic)) << diagnostic;
  EXPECT_EQ(SerializeIKConfig(roundTrip), SerializeIKConfig(expected));
  auto cleanup = CreateTempDirectory("ik_v8_to_v9", ExpectOK{});
  std::string const path = (cleanup.Path() / "canonical.superdex_ik").generic_string();
  ASSERT_TRUE(SaveIKConfig(path, loaded, diagnostic)) << diagnostic;
  ASSERT_TRUE(LoadIKConfig(path, roundTrip, diagnostic)) << diagnostic;
  EXPECT_EQ(SerializeIKConfig(roundTrip), SerializeIKConfig(expected));
}

TEST(IKConfigTest, Parse_EveryVersionRejectsForeignVocabularyWithOrWithoutEqualCounterparts) {
  struct Names {
    char const* current;
    char const* versionEight;
    char const* versionSeven;
    char const* At(int version) const {
      return version < 8 ? versionSeven : version == 8 ? versionEight : current;
    }
  };
  auto const initial = Config();
  DynamicString diagnostic;
  for (int const version : {1, 2, 3, 4, 5, 6, 7, 8, 9}) {
    for (auto const& names :
         {Names{"outputBot", "bot", "bot"},
          Names{"outputBotConfiguration", "botConfiguration", "botConfiguration"},
          Names{"inputBot", "referenceBot", "targetBot"},
          Names{"inputBotConfiguration", "referenceBotConfiguration", "targetBotConfiguration"},
          Names{"inputBotScale", "referenceBotScale", "targetBotScale"},
          Names{"outputBaseFrameFromInputBaseFrame", "baseFromReferenceBot", "baseFromTargetBot"},
          Names{"goals", "goals", "targets"}}) {
      for (int const vocabulary : {7, 8, 9}) {
        char const* const foreign = names.At(vocabulary);
        if (std::string_view(foreign) == names.At(version)) {
          continue;
        }
        for (bool const mixed : {false, true}) {
          auto value = VersionedJson(initial, version);
          auto& fields = value.get<picojson::object>();
          if (!mixed) {
            fields.erase(names.At(version));
          }
          fields[foreign] = Json(initial).get<picojson::object>().at(names.current);
          auto loaded = initial;
          SCOPED_TRACE(value.serialize());
          EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
          EXPECT_FALSE(diagnostic.empty());
          EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
        }
      }
    }
    for (auto const& names :
         {Names{"outputBotLink", "link", "link"},
          Names{"linkFrameFromConstraintFrame", "linkRelativeFrame", "linkRelativeFrame"},
          Names{"inputBotLink", "referenceBotLink", "targetBotLink"},
          Names{"baseOrInputLinkFrameFromConstraintFrame", "goalFrame", "targetFrame"}}) {
      for (int const vocabulary : {7, 8, 9}) {
        char const* const foreign = names.At(vocabulary);
        if (std::string_view(foreign) == names.At(version)) {
          continue;
        }
        for (bool const mixed : {false, true}) {
          auto value = VersionedJson(initial, version);
          auto& entry = value.get<picojson::object>()
                            .at(version < 8 ? "targets" : "goals")
                            .get<picojson::array>()[0]
                            .get<picojson::object>();
          auto const sameValue = entry.at(names.At(version));
          if (!mixed) {
            entry.erase(names.At(version));
          }
          entry[foreign] = sameValue;
          auto loaded = initial;
          SCOPED_TRACE(value.serialize());
          EXPECT_FALSE(ParseIKConfig(value.serialize(), loaded, diagnostic));
          EXPECT_FALSE(diagnostic.empty());
          EXPECT_EQ(SerializeIKConfig(loaded), SerializeIKConfig(initial));
        }
      }
    }
  }
}

} // namespace
