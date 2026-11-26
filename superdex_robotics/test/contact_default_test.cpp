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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/mochi_physics.h>
#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace mochi;
using namespace mochi::test;
using namespace superdex::robotics;

namespace {

// A bot under construction: each link is shaped and joined to its parent by `type`.
struct Chain {
  BotPrefab bot;

  int Add(int parent, ArticulatedJointType type) {
    BotLinkPrefab link;
    link.name = ("link" + std::to_string(bot.links.size())).c_str();
    link.parentLink = parent;
    link.shapeFile = "link.mochi.h5";
    BotJointPrefab joint;
    joint.name = ("joint" + std::to_string(bot.joints.size())).c_str();
    joint.type = type;
    bot.links.push_back(std::move(link));
    bot.joints.push_back(std::move(joint));
    return isize(bot.links) - 1;
  }

  void Cycle(int parent, int child) {
    ArticulatedCycleJointParams cycle;
    cycle.parentLink = parent;
    cycle.childLink = child;
    bot.cycles.push_back(cycle);
  }

  [[nodiscard]] std::pair<bool, bool> Disabled(int a, int b) const {
    return {IsContactImplicitlyDisabled(bot, a, b), IsContactImplicitlyDisabled(bot, b, a)};
  }
};

// Silences the warning and error channels, which a refusal logs, for its scope.
struct QuietRefusals {
  bool const warnings = IsLogChannelEnabled(LogChannel::Warning);
  bool const errors = IsLogChannelEnabled(LogChannel::Error);
  QuietRefusals() {
    EnableLogChannel(LogChannel::Warning, false);
    EnableLogChannel(LogChannel::Error, false);
  }
  ~QuietRefusals() {
    EnableLogChannel(LogChannel::Warning, warnings);
    EnableLogChannel(LogChannel::Error, errors);
  }
  QuietRefusals(QuietRefusals const&) = delete;
  QuietRefusals& operator=(QuietRefusals const&) = delete;
};

constexpr ArticulatedJointType kHard = ArticulatedJointType::Hard;
constexpr ArticulatedJointType kRevolute = ArticulatedJointType::Revolute;

// X -revolute- Y -hard- Z: the search from X stops at Y, which is shaped and entered over a moving
// joint, so X and Z touch, in both argument orders (from Z alone, it would reach X).
TEST(ContactDefaultTest, AWeldUnderAMovingJointTouchesTheParentInBothOrders) {
  Chain chain;
  int const x = chain.Add(kIndexNone, kHard);
  int const y = chain.Add(x, kRevolute);
  int const z = chain.Add(y, kHard);

  EXPECT_EQ(chain.Disabled(x, y), std::make_pair(true, true));
  EXPECT_EQ(chain.Disabled(y, z), std::make_pair(true, true));
  EXPECT_EQ(chain.Disabled(x, z), std::make_pair(false, false));
}

// A welded link and a revolute link under one parent: the answer depends on which comes first in
// spawn order, as in Mochi, but not on the argument order.
TEST(ContactDefaultTest, SiblingsDependOnSpawnOrderNotArgumentOrder) {
  Chain weldFirst;
  int const parent = weldFirst.Add(kIndexNone, kHard);
  int const weld = weldFirst.Add(parent, kHard);
  int const moving = weldFirst.Add(parent, kRevolute);
  Chain movingFirst;
  int const parent2 = movingFirst.Add(kIndexNone, kHard);
  int const moving2 = movingFirst.Add(parent2, kRevolute);
  int const weld2 = movingFirst.Add(parent2, kHard);

  EXPECT_EQ(weldFirst.Disabled(weld, moving), std::make_pair(true, true));
  EXPECT_EQ(movingFirst.Disabled(moving2, weld2), std::make_pair(false, false));
}

// A cycle joint is a moving edge, as in Mochi: the two links it joins do not touch, and the search
// does not pass through them.
TEST(ContactDefaultTest, ACycleJointDisablesItsLinksPair) {
  Chain chain;
  int const base = chain.Add(kIndexNone, kHard);
  int const knuckle = chain.Add(base, kRevolute);
  int const tip = chain.Add(knuckle, kRevolute);
  int const inner = chain.Add(base, kRevolute);
  EXPECT_EQ(chain.Disabled(inner, tip), std::make_pair(false, false));

  chain.Cycle(inner, tip);

  EXPECT_EQ(chain.Disabled(inner, tip), std::make_pair(true, true));
  EXPECT_EQ(chain.Disabled(knuckle, inner), std::make_pair(false, false));
}

// A link reached along two paths, a hard chain and a cycle joint: Mochi takes each link's joints
// first and its cycle joints after, so the search crosses the weld first and goes on past it.
// Taking the cycle first would stop there and leave the far link touching.
TEST(ContactDefaultTest, LinkJointsComeBeforeCycleJoints) {
  Chain chain;
  int const root = chain.Add(kIndexNone, kHard);
  int const weld = chain.Add(root, kHard);
  int const middle = chain.Add(weld, kHard);
  int const moving = chain.Add(middle, kRevolute);
  int const beyond = chain.Add(middle, kHard);
  chain.Cycle(root, middle);

  EXPECT_EQ(chain.Disabled(root, beyond), std::make_pair(true, true));
  EXPECT_EQ(chain.Disabled(root, moving), std::make_pair(true, true));
}

// Every pair of the 2F-85's shaped links, whose cycle joints close its four-bar, is enabled in the
// spawned bot exactly where the mirror says Mochi leaves it, but for the pairs the bot's own
// contactOverrides decide.
TEST(ContactDefaultTest, MatchesTheSpawnedTwoFingerGripper) {
  Context* context = CreateContext(0);
  Scene* scene = context->CreateScene("ContactDefaultTest");
  RoboticsContext* robotics = CreateRoboticsContext();
  BotPrefab const prefab =
      LoadBotPrefabFromFile(GetAssetsDir() + "bots/grippers/2f_85/2f_85.superdex_bot", ExpectOK{});
  ASSERT_FALSE(prefab.cycles.empty());
  Bot* bot = CreateBot(scene, prefab, robotics, ExpectOK{});
  ASSERT_NE(bot, nullptr);
  Span<ActorHandle const> const links = bot->GetArticulatedActor()->GetNestedLinkActors(ExpectOK{});
  ASSERT_EQ(isize(links), isize(prefab.links));
  auto const overridden = [&](int a, int b) {
    DynamicString const& nameA = prefab.links[a].name;
    DynamicString const& nameB = prefab.links[b].name;
    return std::any_of(
        prefab.contactOverrides.begin(),
        prefab.contactOverrides.end(),
        [&](BotContactOverride const& o) {
          return (o.linkA == nameA && o.linkB == nameB) || (o.linkA == nameB && o.linkB == nameA);
        });
  };
  int checked = 0;
  int disabled = 0;

  for (int a = 0; a < isize(prefab.links); ++a) {
    for (int b = a + 1; b < isize(prefab.links); ++b) {
      if (prefab.links[a].shapeFile.empty() || prefab.links[b].shapeFile.empty() ||
          overridden(a, b)) {
        continue;
      }
      bool const enabled = scene->IsContactEnabled(links[a], links[b], ExpectOK{});
      EXPECT_EQ(enabled, !IsContactImplicitlyDisabled(prefab, a, b))
          << prefab.links[a].name << " ~ " << prefab.links[b].name;
      ++checked;
      disabled += enabled ? 0 : 1;
    }
  }

  EXPECT_GT(checked, 0);
  EXPECT_GT(disabled, 0);
  DestroyBot(scene, bot);
  DestroyRoboticsContext(robotics);
  DestroyContext(context);
}

// MuJoCo's rule keeps a weld and its parent weld apart, links further down the tree together, even
// through a shapeless link (which Mochi's rule passes over), and a cycle joint's links apart.
TEST(ContactDefaultTest, MujocosRuleFiltersWeldsAndTheirParentsOnly) {
  Chain chain;
  int const root = chain.Add(kIndexNone, kHard);
  int const upper = chain.Add(root, kRevolute);
  int const weld = chain.Add(upper, kHard);
  int const lower = chain.Add(weld, kRevolute);
  int const tip = chain.Add(lower, kRevolute);
  int const hidden = chain.Add(root, kRevolute);
  int const beyond = chain.Add(hidden, kRevolute);
  chain.bot.links[hidden].shapeFile.clear();
  chain.Cycle(root, lower);

  EXPECT_TRUE(MujocoFiltersSelfContact(chain.bot, root, upper));
  EXPECT_TRUE(MujocoFiltersSelfContact(chain.bot, upper, weld));
  EXPECT_TRUE(MujocoFiltersSelfContact(chain.bot, root, weld));
  EXPECT_TRUE(MujocoFiltersSelfContact(chain.bot, weld, lower));
  EXPECT_TRUE(MujocoFiltersSelfContact(chain.bot, lower, root));
  EXPECT_TRUE(MujocoFiltersSelfContact(chain.bot, upper, lower));
  EXPECT_FALSE(MujocoFiltersSelfContact(chain.bot, upper, tip));
  EXPECT_FALSE(MujocoFiltersSelfContact(chain.bot, root, beyond));
  EXPECT_EQ(chain.Disabled(root, beyond), std::make_pair(true, true));
}

// Both rules on shipped bots count the shaped pairs the defaults audit's independent script counts
// (selfcontact_master_mujoco_mode.tsv, 25 Nov: Mochi's rule in spawn order with cycle joints, and
// MuJoCo's with the root as its own weld and cycle-joined pairs excluded).
TEST(ContactDefaultTest, BothRulesMatchTheAuditsCountsOnShippedBots) {
  struct Expected {
    char const* bot;
    int mochiOff;
    int mujocoOff;
  };
  for (Expected const& expected : {
           Expected{"bots/grippers/2f_85/2f_85.superdex_bot", 12, 14},
           Expected{"bots/arm_hand_combos/fr3_v2_2f_85/fr3_v2_2f_85.superdex_bot", 24, 29},
           Expected{
               "bots/arm_hand_combos/fr3_dg5f_long/right/fr3_dg5f_long_right.superdex_bot", 53, 61},
           Expected{
               "bots/hands/wuji_hand2_beta1/right/wuji_hand2_beta1_right.superdex_bot", 35, 45},
       }) {
    BotPrefab const prefab = LoadBotPrefabFromFile(GetAssetsDir() + expected.bot, ExpectOK{});
    int mochiOff = 0;
    int mujocoOff = 0;
    for (int a = 0; a < isize(prefab.links); ++a) {
      for (int b = a + 1; b < isize(prefab.links); ++b) {
        if (prefab.links[a].shapeFile.empty() || prefab.links[b].shapeFile.empty()) {
          continue;
        }
        mochiOff += IsContactImplicitlyDisabled(prefab, a, b) ? 1 : 0;
        mujocoOff += MujocoFiltersSelfContact(prefab, a, b) ? 1 : 0;
      }
    }
    EXPECT_EQ(mochiOff, expected.mochiOff) << expected.bot;
    EXPECT_EQ(mujocoOff, expected.mujocoOff) << expected.bot;
  }
}

// A 2F-85 spawned with selfContactFilter mujoco has MuJoCo's table on every pair of shaped links
// its own contactOverrides leave alone.
TEST(ContactDefaultTest, AMujocoFilterBotSpawnsMujocosTable) {
  Context* context = CreateContext(0);
  Scene* scene = context->CreateScene("ContactDefaultTest");
  RoboticsContext* robotics = CreateRoboticsContext();
  BotPrefab prefab =
      LoadBotPrefabFromFile(GetAssetsDir() + "bots/grippers/2f_85/2f_85.superdex_bot", ExpectOK{});
  prefab.selfContactFilter = SelfContactFilter::Mujoco;
  Bot* bot = CreateBot(scene, prefab, robotics, ExpectOK{});
  ASSERT_NE(bot, nullptr);
  Span<ActorHandle const> const links = bot->GetArticulatedActor()->GetNestedLinkActors(ExpectOK{});
  ASSERT_EQ(isize(links), isize(prefab.links));
  int changed = 0;

  for (int a = 0; a < isize(prefab.links); ++a) {
    for (int b = a + 1; b < isize(prefab.links); ++b) {
      bool const overridden = std::any_of(
          prefab.contactOverrides.begin(),
          prefab.contactOverrides.end(),
          [&](BotContactOverride const& o) {
            return (o.linkA == prefab.links[a].name && o.linkB == prefab.links[b].name) ||
                (o.linkA == prefab.links[b].name && o.linkB == prefab.links[a].name);
          });
      if (prefab.links[a].shapeFile.empty() || prefab.links[b].shapeFile.empty() || overridden) {
        continue;
      }
      bool const enabled = scene->IsContactEnabled(links[a], links[b], ExpectOK{});
      EXPECT_EQ(enabled, !MujocoFiltersSelfContact(prefab, a, b))
          << prefab.links[a].name << " ~ " << prefab.links[b].name;
      changed += enabled == !IsContactImplicitlyDisabled(prefab, a, b) ? 0 : 1;
    }
  }

  EXPECT_GT(changed, 0);
  DestroyBot(scene, bot);
  DestroyRoboticsContext(robotics);
  DestroyContext(context);
}

// The key round-trips through a bot file: the default is not written, Mujoco is written and read
// back; an unknown name and Count refuse the bot.
TEST(ContactDefaultTest, TheFilterRoundTripsAndRefusesUnknownValues) {
  BotPrefab prefab =
      LoadBotPrefabFromFile(GetAssetsDir() + "bots/grippers/2f_85/2f_85.superdex_bot", ExpectOK{});
  std::string const plain{WriteToJsonString(prefab, ExpectOK{}).c_str()};
  prefab.selfContactFilter = SelfContactFilter::Mujoco;
  std::string const mujoco{WriteToJsonString(prefab, ExpectOK{}).c_str()};
  std::string misspelled = mujoco;
  misspelled.replace(misspelled.find("\"Mujoco\""), 8, "\"Mujuco\"");

  EXPECT_EQ(plain.find("selfContactFilter"), std::string::npos);
  EXPECT_NE(mujoco.find(R"("selfContactFilter": "Mujoco")"), std::string::npos) << mujoco;
  EXPECT_EQ(
      ReadFromJsonString(mujoco, ExpectOK{}).selfContactFilter,
      std::optional{SelfContactFilter::Mujoco});
  QuietRefusals const quiet;
  (void)ReadFromJsonString(misspelled, ExpectNotOK{});
  prefab.selfContactFilter = SelfContactFilter::Count;
  Validate(prefab, nullptr, ExpectNotOK{});
}

// A combo has one self-contact rule: its base's, else one an attached bot names. A base and an
// attached bot that name different rules refuse the combo; a bot that names none takes the combo's.
TEST(ContactDefaultTest, ACombosBotsShareOneFilter) {
  // Loads every bot file as written, but gives the named ones the rule given.
  struct RuleLoader : FileBotLoader {
    std::vector<std::pair<std::string, SelfContactFilter>> rules;
    BotPrefab LoadBotPrefab(std::string_view path, superdex::Error& error) const override {
      BotPrefab prefab = FileBotLoader::LoadBotPrefab(path, error);
      for (auto const& [name, rule] : rules) {
        if (path.ends_with(name)) {
          prefab.selfContactFilter = rule;
        }
      }
      return prefab;
    }
  };
  std::string const combo =
      GetAssetsDir() + "bots/arm_hand_combos/fr3_v2_2f_85/fr3_v2_2f_85.superdex_bot";
  std::string const arm = "fr3_v2/fr3_v2.superdex_bot";
  std::string const gripper = "2f_85/2f_85.superdex_bot";
  auto const build = [&](std::vector<std::pair<std::string, SelfContactFilter>> rules,
                         superdex::Error& error) {
    RuleLoader loader;
    loader.rules = std::move(rules);
    ModBotPrefab const recipe = loader.LoadModBotPrefab(combo, ExpectOK{});
    return BuildBot(recipe, loader, /*validate*/ false, error);
  };

  BotPrefab const plain = build({}, ExpectOK{});
  BotPrefab const baseOnly = build({{arm, SelfContactFilter::Mujoco}}, ExpectOK{});
  BotPrefab const childOnly = build({{gripper, SelfContactFilter::Mujoco}}, ExpectOK{});
  BotPrefab const both =
      build({{arm, SelfContactFilter::Mujoco}, {gripper, SelfContactFilter::Mujoco}}, ExpectOK{});

  EXPECT_FALSE(plain.selfContactFilter.has_value());
  EXPECT_EQ(baseOnly.selfContactFilter, std::optional{SelfContactFilter::Mujoco});
  EXPECT_EQ(childOnly.selfContactFilter, std::optional{SelfContactFilter::Mujoco});
  EXPECT_EQ(both.selfContactFilter, std::optional{SelfContactFilter::Mujoco});
  QuietRefusals const quiet;
  (void)build(
      {{arm, SelfContactFilter::Mujoco}, {gripper, SelfContactFilter::Mochi}}, ExpectNotOK{});
}

// A shapeless link passes the search on: links joined through it do not touch.
TEST(ContactDefaultTest, AShapelessLinkPassesTheSearchOn) {
  Chain chain;
  int const a = chain.Add(kIndexNone, kHard);
  int const hidden = chain.Add(a, kRevolute);
  int const b = chain.Add(hidden, kRevolute);
  chain.bot.links[hidden].shapeFile.clear();

  EXPECT_EQ(chain.Disabled(a, b), std::make_pair(true, true));
  EXPECT_EQ(chain.Disabled(a, hidden), std::make_pair(false, false));
}

} // namespace
