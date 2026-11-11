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

#include "Misc/AutomationTest.h"

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

#undef CLUTCH_RETARGETER_TEST_FLAGS

#endif // WITH_DEV_AUTOMATION_TESTS
