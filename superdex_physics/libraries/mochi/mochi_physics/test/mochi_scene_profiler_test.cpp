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

#include "mochi_physics_test_fixture.h"

#include <mochi_physics/src/mochi_island.h>
#include <mochi_physics/src/mochi_query.h>
#include <mochi_physics/src/mochi_scene_profiler.h>
#include <mochi_physics/src/mochi_sleep.h>

#include <gtest/gtest.h>

using namespace mochi;

namespace {

class SceneProfiler : public test::MochiSceneTestBase {
 protected:
  static constexpr double kDt = 0.01;

  // A unit box with its lowest corner at the given position.
  Actor* CreateBox(Real3 const& corner, bool isStatic) {
    RigidActorParams params;
    params.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
    params.colliderType = ColliderType::Box;
    params.isStatic = isStatic;
    params.worldFromLocal.SetTranslation(corner);
    return _scene->CreateRigidActor(params, test::ExpectOK{});
  }

  // A dynamic box sinking slightly into a static box, so they are in contact.
  Actor* CreateBoxOnGround() {
    CreateBox(Real3{0_r, -1_r, 0_r}, /*isStatic*/ true);
    return CreateBox(Real3{0_r, -0.01_r, 0_r}, /*isStatic*/ false);
  }

  entt::entity GetIsland(Actor const* actor) const {
    return GetRegistry().get<CIslandMemberInfo const>(GetEntity(actor->GetHandle())).island;
  }

  void Step() {
    _scene->Step(kDt);
  }
};

} // namespace

TEST_F(SceneProfiler, EnableAddsStatsToExistingAndNewIslandsAndDisableRemovesThem) {
  auto& reg = GetRegistry();
  auto const first = GetIsland(CreateBox(Real3{}, /*isStatic*/ false));
  EXPECT_EQ(nullptr, reg.try_get<CIslandStats>(first));
  EXPECT_FALSE(profiler::IsEnabled(reg));

  profiler::Enable(reg);
  profiler::Enable(reg);
  EXPECT_TRUE(profiler::IsEnabled(reg));
  ASSERT_NE(nullptr, reg.try_get<CIslandStats>(first));
  EXPECT_EQ(CIslandStats{}, reg.get<CIslandStats>(first));
  auto const second = GetIsland(CreateBox(Real3{100_r, 0_r, 0_r}, /*isStatic*/ false));
  EXPECT_NE(first, second);
  EXPECT_NE(nullptr, reg.try_get<CIslandStats>(second));

  profiler::Disable(reg);
  profiler::Disable(reg);
  EXPECT_FALSE(profiler::IsEnabled(reg));
  EXPECT_TRUE(reg.view<CIslandStats>().empty());
  auto const third = GetIsland(CreateBox(Real3{200_r, 0_r, 0_r}, /*isStatic*/ false));
  EXPECT_EQ(nullptr, reg.try_get<CIslandStats>(third));
}

TEST_F(SceneProfiler, StepRecordsExclusiveIslandTimes) {
  Actor const* box = CreateBoxOnGround();
  auto& reg = GetRegistry();
  profiler::Enable(reg);
  // The next step must reset previous measurements, not accumulate across steps.
  reg.get<CIslandStats>(GetIsland(box)) = {1.0, 1.0, 1.0, 1.0};
  Step();

  auto const& stats = reg.get<CIslandStats>(GetIsland(box));
  EXPECT_GT(stats.contactSec, 0.0);
  EXPECT_GT(stats.assemblySec, 0.0);
  EXPECT_GT(stats.solveSec, 0.0);
  EXPECT_GT(stats.queriesSec, 0.0);
  EXPECT_LE(
      stats.contactSec + stats.assemblySec + stats.solveSec + stats.queriesSec,
      _scene->GetPerformanceStats().solveStepDurationSec);
}

TEST_F(SceneProfiler, SleepingIslandsAreNotMeasured) {
  Actor const* box = CreateBoxOnGround();
  auto& reg = GetRegistry();
  profiler::Enable(reg);
  Step();
  auto const island = GetIsland(box);
  auto const previousStats = reg.get<CIslandStats>(island);

  sleep::ForceSleep(reg, island);
  Step();
  EXPECT_EQ(previousStats, reg.get<CIslandStats>(island));
}

TEST_F(SceneProfiler, QueryComponentsAreTimedWithoutQueryHandles) {
  SoftActorParams params;
  params.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
  Actor* actor = _scene->CreateSoftActor(params, test::ExpectOK{});
  auto& reg = GetRegistry();
  auto const e = GetEntity(actor->GetHandle());
  reg.emplace<CQuerySurfaceNodePositions>(e);
  profiler::Enable(reg);
  Step();
  EXPECT_FALSE(reg.all_of<CActiveQuerySet>(e));
  EXPECT_FALSE(reg.get<CQuerySurfaceNodePositions>(e).nodePositions.empty());
  EXPECT_GT(reg.get<CIslandStats>(GetIsland(actor)).queriesSec, 0.0);
}

TEST_F(SceneProfiler, TimerOnlyWritesToTheSelectedIslandIfPresent) {
  auto& reg = GetRegistry();
  auto const first = island::CreateEmpty(reg);
  auto const second = island::CreateEmpty(reg);
  {
    profiler::ScopedIslandTimer timer(reg, first, &CIslandStats::assemblySec);
  }
  EXPECT_EQ(nullptr, reg.try_get<CIslandStats>(first));

  profiler::Enable(reg);
  {
    profiler::ScopedIslandTimer timer(reg, first, &CIslandStats::assemblySec);
  }
  EXPECT_GE(reg.get<CIslandStats>(first).assemblySec, 0.0);
  EXPECT_EQ(CIslandStats{}, reg.get<CIslandStats>(second));
}
