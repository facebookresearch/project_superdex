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

#include "Tests/SuperDexRetargeterCacheTestTypes.h"

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "Engine/World.h"
#include "HAL/PlatformTLS.h"
#include "MochiArticulationBase.h"
#include "Skeletons/SuperDexSkeletonBots.h"
#include "Skeletons/SuperDexSkeleton_OpenXR.h"
#include "SuperDexPoseSubsystem.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexRetargeterReadsItsCacheWhileComputingTest,
    "SuperDexTeleop.SuperDexRetargeter.ReadsItsCacheWhileComputingItsPose",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSuperDexRetargeterReadsItsCacheWhileComputingTest::RunTest(const FString& Parameters) {
  // A retargeter computes its pose holding its cache for writing. Reads of the cache from that
  // computation, on the same thread, must not wait on that lock.
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
  auto* Reader = NewObject<USuperDexRetargeterCacheTestReader>(Actor);
  Actor->AddInstanceComponent(Reader);
  Reader->RegisterComponent();
  auto* Source = NewObject<USuperDexRetargeterCacheTestSource>(Actor);
  Actor->AddInstanceComponent(Source);
  Source->SetupAttachment(Reader);
  Source->RegisterComponent();
  Source->TestPose.Bot = Bot;
  Source->TestPose.DofPositions.Init(0.0, Bot->GetNumDofs());
  Source->TestPose.bIsTrackingEnabled = Source->TestPose.bIsTrackingConfident = true;
  Source->TestPose.TrackingConfidence = 1;
  Reader->LinkName =
      SuperDexBoneNames_OXR[static_cast<int32>(GetFingerTipBone_OXR(ESuperDexFinger::Index))];

  // A new cache key for each request, so that each one computes the pose.
  double Key = 0;
  auto& Threads = const_cast<TMap<uint32, ThreadInfoForCaching>&>(
      World->GetSubsystem<USuperDexPoseSubsystem>()->GetThreadInfoForCaching());
  Threads.FindChecked(FPlatformTLS::GetCurrentThreadId()).GetCacheKeyCallback = [&Key] {
    return Key;
  };

  for (int32 Request = 0; Request < 2; ++Request) {
    ++Key;
    FSuperDexPose Pose;
    TestTrue(TEXT("GetPose computes the pose"), Reader->GetPose(Pose));
  }
  TestEqual(TEXT("Each request computed the pose"), Reader->Calls, 2);
  TestEqual(TEXT("The cached pose reads see the previous pose"), Reader->CachedPoseReads, 1);
  TestEqual(TEXT("The cached link reads see the previous pose"), Reader->CachedLinkReads, 1);
  FSuperDexPose Cached;
  TestTrue(TEXT("The cache is readable afterwards"), Reader->GetCachedPose(Cached));
  TestTrue(TEXT("The cache holds the computed pose"), Cached.Bot.Get() == Bot);

  Source->DestroyComponent();
  Reader->DestroyComponent();
  World->DestroyWorld(false);
  return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
