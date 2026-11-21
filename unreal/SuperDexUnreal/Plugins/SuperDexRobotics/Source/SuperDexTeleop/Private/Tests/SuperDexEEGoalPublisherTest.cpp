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

#include "Retargeters/SuperDexEEGoalPublisher.h"

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#define EE_GOAL_PUBLISHER_TEST_FLAGS \
  (EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexEEGoalPublisherGoalMathTest,
    "SuperDexTeleop.SuperDexEEGoalPublisher.GoalMath",
    EE_GOAL_PUBLISHER_TEST_FLAGS)

bool FSuperDexEEGoalPublisherGoalMathTest::RunTest(const FString& Parameters) {
  const FTransform EndEffectorToSourceRoot(
      FQuat(FVector::UpVector, PI), FVector(0, 0, 3.5), FVector::OneVector);
  const FTransform SourceRootToVrRoot(
      FQuat(FVector(1, 1, 0).GetSafeNormal(), 0.7), FVector(40, -12, 95), FVector::OneVector);
  const FTransform ArmBaseToActor(
      FQuat(FVector::UpVector, 0.3), FVector(5, 10, -2), FVector::OneVector);

  const FTransform Goal = USuperDexEEGoalPublisher::ComputeEndEffectorGoal(
      EndEffectorToSourceRoot, SourceRootToVrRoot, ArmBaseToActor);

  // The goal is the end effector in the arm base frame: re-expressing it in the actor frame must
  // give the end effector carried by the source root.
  const FTransform ExpectedEndEffectorToActor = EndEffectorToSourceRoot * SourceRootToVrRoot;
  TestTrue(
      TEXT("Goal * ArmBaseToActor == EndEffectorToSourceRoot * SourceRootToVrRoot"),
      (Goal * ArmBaseToActor).Equals(ExpectedEndEffectorToActor, 1e-6));

  const FTransform GoalAtActorRoot = USuperDexEEGoalPublisher::ComputeEndEffectorGoal(
      EndEffectorToSourceRoot, SourceRootToVrRoot, FTransform::Identity);
  TestTrue(
      TEXT("An arm base at the actor root leaves the goal in the actor frame"),
      GoalAtActorRoot.Equals(ExpectedEndEffectorToActor, 1e-6));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexEEGoalPublisherUninitializedTest,
    "SuperDexTeleop.SuperDexEEGoalPublisher.Uninitialized",
    EE_GOAL_PUBLISHER_TEST_FLAGS)

bool FSuperDexEEGoalPublisherUninitializedTest::RunTest(const FString& Parameters) {
  USuperDexEEGoalPublisher* Publisher = NewObject<USuperDexEEGoalPublisher>();
  TestFalse(TEXT("Not initialized before BeginPlay"), Publisher->IsInitialized());
  TestTrue(
      TEXT("Runs on invalid source poses so the goal can be invalidated"),
      Publisher->bGetTransformedPoseDoesNotNeedValidSourcePose);

  FSuperDexPose Source;
  FSuperDexPose Transformed;
  TestFalse(
      TEXT("Produces no pose before initialization (valid source)"),
      Publisher->GetTransformedPose_Implementation(Source, true, Transformed));
  TestFalse(
      TEXT("Produces no pose before initialization (invalid source)"),
      Publisher->GetTransformedPose_Implementation(Source, false, Transformed));
  TestTrue(TEXT("No joint positions cached"), Publisher->GetCachedJointPositions().IsEmpty());
  return true;
}

#undef EE_GOAL_PUBLISHER_TEST_FLAGS

#endif // WITH_DEV_AUTOMATION_TESTS
