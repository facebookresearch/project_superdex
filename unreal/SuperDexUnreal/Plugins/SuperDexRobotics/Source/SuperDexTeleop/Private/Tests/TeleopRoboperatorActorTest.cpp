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

#include "TeleopRoboperatorActor.h"

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

namespace {

constexpr double kDt = 0.25;
constexpr double kSettleSeconds = 0.5;
constexpr double kTolerance = 0.01;
constexpr double kMaxWaitSeconds = 2.0;

/* Feeds HandScales one per tick and returns the tick (1-based) and scale of the spawn, or 0. */
TPair<int32, double> RunSettle(TConstArrayView<double> HandScales) {
  ATeleopRoboperatorActor::FSpawnSettle Settle;
  for (int32 Tick = 0; Tick < HandScales.Num(); ++Tick) {
    const double Scale = ATeleopRoboperatorActor::AdvanceSpawnSettle(
        Settle, HandScales[Tick], kDt, kSettleSeconds, kTolerance, kMaxWaitSeconds);
    if (Scale > 0.0) {
      return {Tick + 1, Scale};
    }
  }
  return {0, 0.0};
}

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FTeleopRoboperatorSpawnSettleTest,
    "Teleop.TeleopRoboperatorActor.SpawnSettle",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FTeleopRoboperatorSpawnSettleTest::RunTest(const FString& Parameters) {
  TArray<double> Scales;

  // A steady scale spawns once it has been steady for kSettleSeconds: tick 1 sets the reference.
  Scales.Init(0.8, 10);
  TestEqual(TEXT("Steady tick"), RunSettle(Scales).Key, 3);
  TestEqual(TEXT("Steady scale"), RunSettle(Scales).Value, 0.8);

  // Changes within the tolerance count as steady.
  Scales = {0.8, 0.805, 0.795, 0.8};
  TestEqual(TEXT("Within tolerance tick"), RunSettle(Scales).Key, 3);

  // A drifting scale never settles, so it spawns at kMaxWaitSeconds with the latest scale.
  Scales.Reset();
  for (int32 Tick = 0; Tick < 10; ++Tick) {
    Scales.Add(0.7 + 0.05 * Tick);
  }
  TestEqual(TEXT("Drifting tick"), RunSettle(Scales).Key, 8);
  TestEqual(TEXT("Drifting scale"), RunSettle(Scales).Value, 0.7 + 0.05 * 7);

  // A scale that is never valid still spawns at kMaxWaitSeconds, at nominal size.
  for (const double Invalid : {0.0, -1.0, TNumericLimits<double>::Max() * 10.0}) {
    Scales.Init(Invalid, 10);
    TestEqual(TEXT("Invalid tick"), RunSettle(Scales).Key, 8);
    TestEqual(TEXT("Invalid scale"), RunSettle(Scales).Value, 1.0);
  }
  return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
