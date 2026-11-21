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

#include "Retargeters/ClutchSkeletonRootToVrRootRetargeter.h"

#include "Engine/World.h"
#include "Misc/AutomationTest.h"
#include "MochiArticulationBase.h"
#include "Skeletons/SuperDexSkeletonBots.h"

#if WITH_DEV_AUTOMATION_TESTS

#define CLUTCH_RETARGETER_TEST_FLAGS \
  (EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FClutchRetargeterFailedReinitializationTest,
    "SuperDexTeleop.ClutchSkeletonRootToVrRootRetargeter.FailedReinitialization",
    CLUTCH_RETARGETER_TEST_FLAGS)

bool FClutchRetargeterFailedReinitializationTest::RunTest(const FString& Parameters) {
  UClutchSkeletonRootToVrRootRetargeter* Clutch =
      NewObject<UClutchSkeletonRootToVrRootRetargeter>();

  Clutch->SetClutchedSkeletonRootToVrRoot(FTransform::Identity);
  Clutch->SetClutchEnabled(true);
  TestTrue(TEXT("Clutch starts initialized"), Clutch->IsClutchInitialized());
  TestTrue(TEXT("Clutch starts enabled"), Clutch->GetClutchEnabled());

  AddExpectedError(
      TEXT("was passed an empty string for leaf initialization"),
      EAutomationExpectedErrorFlags::Contains,
      1);
  TestFalse(
      TEXT("Reinitialization fails without a poseable name"),
      Clutch->InitializeClutchFromCurrentPoseableLeafBone(TEXT("")));
  TestFalse(
      TEXT("Failed reinitialization clears initialized state"), Clutch->IsClutchInitialized());
  TestFalse(TEXT("Failed reinitialization disables clutch"), Clutch->GetClutchEnabled());
  TestFalse(TEXT("Stale transform cannot be re-enabled"), Clutch->SetClutchEnabled(true));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FClutchRetargeterJointPositionsTest,
    "SuperDexTeleop.ClutchSkeletonRootToVrRootRetargeter.HoldsAndBlendsJointPositions",
    CLUTCH_RETARGETER_TEST_FLAGS)

bool FClutchRetargeterJointPositionsTest::RunTest(const FString& Parameters) {
  UMochiArticulationBase* Bot = GetSkeletonBot_OpenXR(ESuperDexHandedness::Right);
  if (!TestNotNull(TEXT("OpenXR skeleton bot"), Bot)) {
    return false;
  }
  const auto Settings = UWorld::InitializationValues()
                            .CreatePhysicsScene(false)
                            .CreateNavigation(false)
                            .CreateAISystem(false)
                            .ShouldSimulatePhysics(false);
  UWorld* World = UWorld::CreateWorld(
      EWorldType::Game, false, NAME_None, nullptr, true, ERHIFeatureLevel::Num, &Settings);
  AActor* Actor = World->SpawnActor<AActor>();
  auto* Clutch = NewObject<UClutchSkeletonRootToVrRootRetargeter>(Actor);
  Actor->AddInstanceComponent(Clutch);
  Clutch->RegisterComponent();
  Clutch->TimeInSecondsToLerpToAbsolute = 1.0;
  Clutch->TimeInSecondsToLerpOnUnpause = 0.25;
  Clutch->SetClutchedSkeletonRootToVrRoot(FTransform::Identity);

  // Poses whose joint positions all equal Q.
  const auto MakePose = [Bot](double Q) {
    FSuperDexPose Pose;
    Pose.Bot = Bot;
    Pose.DofPositions.Init(Q, Bot->GetNumDofs());
    Pose.bIsTrackingEnabled = Pose.bIsTrackingConfident = true;
    Pose.TrackingConfidence = 1;
    return Pose;
  };
  // The joint position the clutch outputs at Time for a source pose at Q.
  const auto Output = [&](double Time, const FSuperDexPose& Source) {
    World->TimeSeconds = Time;
    FSuperDexPose Pose;
    if (!Clutch->GetTransformedPose_Implementation(Source, true, Pose) ||
        Pose.DofPositions.Num() != Bot->GetNumDofs()) {
      return TNumericLimits<double>::Max();
    }
    for (const double Q : Pose.DofPositions) {
      if (Q != Pose.DofPositions[0]) {
        return TNumericLimits<double>::Lowest();
      }
    }
    return Pose.DofPositions[0];
  };
  constexpr double kTolerance = 1e-9;

  TestEqual(TEXT("Disengaged: the first pose"), Output(1.0, MakePose(0.1)), 0.1, kTolerance);
  TestEqual(TEXT("Disengaged: holds it"), Output(2.0, MakePose(0.5)), 0.1, kTolerance);
  Clutch->SetClutchEnabled(true);
  TestEqual(
      TEXT("Engaging: starts at the held pose"), Output(10.0, MakePose(0.5)), 0.1, kTolerance);
  TestEqual(TEXT("Engaged: blends"), Output(10.5, MakePose(0.5)), 0.3, kTolerance);
  TestEqual(TEXT("Engaged: blends the live pose"), Output(10.75, MakePose(0.9)), 0.7, kTolerance);
  TestEqual(TEXT("Engaged: passes through after"), Output(11.0, MakePose(0.5)), 0.5, kTolerance);
  TestEqual(TEXT("Engaged: passes through"), Output(12.0, MakePose(0.2)), 0.2, kTolerance);

  // Disengaging mid-blend holds what was output.
  Clutch->SetClutchEnabled(false);
  Clutch->SetClutchEnabled(true);
  Output(20.0, MakePose(0.6));
  TestEqual(TEXT("Engaged again: blends"), Output(20.5, MakePose(0.6)), 0.4, kTolerance);
  Clutch->SetClutchEnabled(false);
  TestEqual(TEXT("Disengaged mid-blend: holds"), Output(20.6, MakePose(0.0)), 0.4, kTolerance);

  // Pausing holds; unpausing blends over its own time.
  Clutch->bPauseClutchInsteadOfAutoDisengaging = true;
  Clutch->SetClutchEnabled(true);
  Output(30.0, MakePose(0.4));
  TestEqual(TEXT("Engaged"), Output(31.0, MakePose(0.4)), 0.4, kTolerance);
  FSuperDexPose Untracked = MakePose(0.0);
  Untracked.bIsTrackingConfident = false;
  TestEqual(TEXT("Paused: holds"), Output(32.0, Untracked), 0.4, kTolerance);
  TestTrue(TEXT("Paused, not disengaged"), Clutch->GetClutchEnabled());
  TestEqual(
      TEXT("Unpausing: starts at the held pose"), Output(33.0, MakePose(0.8)), 0.4, kTolerance);
  TestEqual(TEXT("Unpaused: blends"), Output(33.125, MakePose(0.8)), 0.6, kTolerance);
  TestEqual(TEXT("Unpaused: passes through after"), Output(33.25, MakePose(0.8)), 0.8, kTolerance);

  Clutch->DestroyComponent();
  World->DestroyWorld(false);
  return true;
}

#undef CLUTCH_RETARGETER_TEST_FLAGS

#endif // WITH_DEV_AUTOMATION_TESTS
