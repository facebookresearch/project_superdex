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

#include "Engine/SkeletalMesh.h"
#include "Misc/AutomationTest.h"
#include "SuperDexBot.h"

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

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexEEGoalPublisherMeshCompatibilityTest,
    "SuperDexTeleop.SuperDexEEGoalPublisher.MeshCompatibilityRequiresInputs",
    EE_GOAL_PUBLISHER_TEST_FLAGS)

bool FSuperDexEEGoalPublisherMeshCompatibilityTest::RunTest(const FString& Parameters) {
  FString Reason;
  TestFalse(
      TEXT("No mesh and no articulation is incompatible"),
      USuperDexEEGoalPublisher::IsMeshCompatibleWithArticulation(nullptr, nullptr, &Reason));
  TestFalse(TEXT("A reason is reported"), Reason.IsEmpty());
  TestFalse(
      TEXT("A null reason pointer is allowed"),
      USuperDexEEGoalPublisher::IsMeshCompatibleWithArticulation(nullptr, nullptr));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexEEGoalPublisherArmDofsTest,
    "SuperDexTeleop.SuperDexEEGoalPublisher.ArmDofsOfCombinedBot",
    EE_GOAL_PUBLISHER_TEST_FLAGS)

bool FSuperDexEEGoalPublisherArmDofsTest::RunTest(const FString& Parameters) {
  // The FR3 V2 arm with a DG-5F hand: 7 arm DOFs followed by 20 hand DOFs.
  const auto* Combo = LoadObject<USuperDexBot>(
      nullptr,
      TEXT("/SuperDexAssets/bots/arm_hand_combos/fr3_v2_dg5f_long_seed/right/"
           "SDB_fr3_v2_dg5f_long_seed_right.SDB_fr3_v2_dg5f_long_seed_right"));
  if (!TestNotNull(TEXT("Combined arm + hand bot"), Combo))
    return false;
  const USkinnedAsset* Mesh = Combo->GetSkeletalMesh();
  TestTrue(
      TEXT("The combined bot's own mesh is compatible"),
      USuperDexEEGoalPublisher::IsMeshCompatibleWithArticulation(Mesh, Combo));
  const int32 NumDofs = Combo->GetNumDofs();
  TestEqual(TEXT("Combined bot DOFs"), NumDofs, 27);

  TArray<int32> Arm;
  FString Reason;
  TestTrue(
      TEXT("Arm joints up to the flange"),
      USuperDexEEGoalPublisher::ComputeArmDofIndices(Combo, Mesh, TEXT("fr3_link8"), Arm, &Reason));
  TestEqual(TEXT("The flange chain has the 7 FR3 joints"), Arm, TArray<int32>{0, 1, 2, 3, 4, 5, 6});
  for (int32 Joint = 0; Joint < Arm.Num(); ++Joint) {
    TestEqual(
        TEXT("Arm joint i moves fr3_link<i+1>"),
        Combo->GetDofLinkName(Arm[Joint]),
        FName(*FString::Printf(TEXT("fr3_link%d"), Joint + 1)));
  }

  TestTrue(
      TEXT("Arm joints up to a mid-arm link"),
      USuperDexEEGoalPublisher::ComputeArmDofIndices(Combo, Mesh, TEXT("fr3_link4"), Arm));
  TestEqual(TEXT("fr3_link4's chain has 4 joints"), Arm, TArray<int32>{0, 1, 2, 3});

  // A chain into the hand (to the thumb's seed sensor mount) takes the arm and that finger's
  // joints, never another finger's.
  TestTrue(
      TEXT("Arm joints up to the thumb's sensor mount"),
      USuperDexEEGoalPublisher::ComputeArmDofIndices(
          Combo, Mesh, TEXT("dg5f_link_1_4_seed_mount"), Arm));
  int32 ThumbDofs = 0;
  for (int32 Dof = 0; Dof < NumDofs; ++Dof) {
    ThumbDofs += Combo->GetDofLinkName(Dof).ToString().StartsWith(TEXT("dg5f_link_1_")) ? 1 : 0;
  }
  TestTrue(TEXT("The thumb has joints"), ThumbDofs > 0);
  TestEqual(TEXT("Thumb chain: arm + thumb joints"), Arm.Num(), 7 + ThumbDofs);
  for (int32 Dof : Arm) {
    const FString Link = Combo->GetDofLinkName(Dof).ToString();
    TestTrue(
        TEXT("Thumb chain joints move the arm or the thumb"),
        Link.StartsWith(TEXT("fr3_link")) || Link.StartsWith(TEXT("dg5f_link_1_")));
  }

  TestTrue(
      TEXT("No end effector: every DOF"),
      USuperDexEEGoalPublisher::ComputeArmDofIndices(Combo, Mesh, NAME_None, Arm));
  TestEqual(TEXT("Every DOF is an arm joint"), Arm.Num(), NumDofs);

  Reason.Reset();
  TestFalse(
      TEXT("An end effector that is not a link"),
      USuperDexEEGoalPublisher::ComputeArmDofIndices(Combo, Mesh, TEXT("not_a_link"), Arm, &Reason));
  TestFalse(TEXT("A reason is reported"), Reason.IsEmpty());
  TestTrue(TEXT("No arm joints on failure"), Arm.IsEmpty());
  TestFalse(
      TEXT("No articulation"),
      USuperDexEEGoalPublisher::ComputeArmDofIndices(nullptr, Mesh, TEXT("fr3_link8"), Arm));
  return true;
}

#undef EE_GOAL_PUBLISHER_TEST_FLAGS

#endif // WITH_DEV_AUTOMATION_TESTS
