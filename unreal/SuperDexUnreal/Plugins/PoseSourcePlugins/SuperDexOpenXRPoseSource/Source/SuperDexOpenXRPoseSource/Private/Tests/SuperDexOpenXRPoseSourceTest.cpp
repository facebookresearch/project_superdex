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

#include "SuperDexOpenXRPoseSource.h"
#include "Skeletons/SuperDexSkeletonBots.h"
#include "Skeletons/SuperDexSkeleton_OpenXR.h"

#include "MochiArticulationBase.h"
#include "MochiConversions.h"
#include "MochiTypes.h"

#include "HeadMountedDisplayTypes.h"
#include "Math/RandomStream.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#define OPENXR_POSE_SOURCE_TEST_FLAGS \
  (EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

namespace {

/* Quaternions below hold OpenXR (right-handed) rotations as raw (x, y, z, w); the Hamilton product
 * of FQuat composes them correctly regardless of handedness. */

// UE's OpenXR plugin conversion (OpenXRCore.h ToFQuat).
FQuat XrToUnreal(const FQuat& Xr) {
  return FQuat(-Xr.Z, Xr.X, Xr.Y, -Xr.W);
}

// What the superdex importer makes of a right-handed bot rotation.
FQuat BotToUnreal(const FQuat& Bot) {
  const TMochiToUnrealConverter<double> M2U(kSuperDexSpace);
  return M2U.RotationToUnreal(
      mochi::Quaternion(
          static_cast<mochi::real>(Bot.X),
          static_cast<mochi::real>(Bot.Y),
          static_cast<mochi::real>(Bot.Z),
          static_cast<mochi::real>(Bot.W)));
}

// What the superdex importer made a right-handed bot rotation into, back in the bot's space.
FQuat UnrealToBot(const FQuat& Unreal) {
  const TUnrealToMochiConverter<double> U2M(kSuperDexSpace);
  const mochi::Real4 V = U2M.RotationToMochi(Unreal).ToReal4();
  return FQuat(V[0], V[1], V[2], V[3]);
}

// FQuat::AngularDistance takes acos of a value that rounds above 1 for equal rotations.
double AngleBetween(const FQuat& A, const FQuat& B) {
  const FQuat D = A.Inverse() * B;
  return 2.0 * FMath::Atan2(FVector(D.X, D.Y, D.Z).Size(), FMath::Abs(D.W));
}

FQuat RandomRotation(FRandomStream& Rng) {
  return FQuat(Rng.GetUnitVector(), Rng.FRandRange(-PI, PI));
}

struct FRandomHand {
  TArray<FQuat> XrLocals; // parent-from-joint, ESuperDexBone_OXR order
  TArray<FQuat> XrWorld; // tracking-from-joint, EHandKeypoint order
};

// A hand with the given parent-from-joint rotations (ESuperDexBone_OXR order).
FRandomHand MakeHand(TArray<FQuat> XrLocals) {
  constexpr int32 NumBones = static_cast<int32>(ESuperDexBone_OXR::Count);
  check(XrLocals.Num() == NumBones);
  FRandomHand Hand;
  Hand.XrLocals = MoveTemp(XrLocals);
  Hand.XrWorld.SetNum(EHandKeypointCount);
  TArray<FQuat> BoneWorld;
  BoneWorld.SetNum(NumBones);
  for (int32 Bone = 0; Bone < NumBones; ++Bone) {
    const ESuperDexBone_OXR Parent = kSuperDexBoneParents_OXR[Bone];
    BoneWorld[Bone] = Parent == ESuperDexBone_OXR::Invalid
        ? Hand.XrLocals[Bone]
        : BoneWorld[static_cast<int32>(Parent)] * Hand.XrLocals[Bone];
    Hand.XrWorld[GetXrHandJointOfBone_OXR(static_cast<ESuperDexBone_OXR>(Bone))] = BoneWorld[Bone];
  }
  return Hand;
}

FRandomHand MakeRandomHand(FRandomStream& Rng) {
  TArray<FQuat> XrLocals;
  for (int32 Bone = 0; Bone < static_cast<int32>(ESuperDexBone_OXR::Count); ++Bone) {
    XrLocals.Add(RandomRotation(Rng));
  }
  return MakeHand(MoveTemp(XrLocals));
}

FSuperDexPose ConvertAsUnrealReportsIt(
    FAutomationTestBase& Test,
    const FRandomHand& Hand,
    ESuperDexHandedness Handedness) {
  TArray<FQuat> UnrealRotations;
  for (const FQuat& Q : Hand.XrWorld) {
    UnrealRotations.Add(XrToUnreal(Q));
  }
  FSuperDexPose Pose;
  Test.TestTrue(
      TEXT("Converted"),
      USuperDexOpenXRPoseSource::ConvertHandKeypointsToPose(
          UnrealRotations, FVector(1, 2, 3), Handedness, Pose));
  return Pose;
}

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FOpenXRPoseSourceJointPositionsTest,
    "SuperDexOpenXRPoseSource.JointPositionsAreTheTrackedBotJointPositions",
    OPENXR_POSE_SOURCE_TEST_FLAGS)

bool FOpenXRPoseSourceJointPositionsTest::RunTest(const FString& Parameters) {
  FRandomStream Rng(42);
  for (const ESuperDexHandedness Handedness :
       {ESuperDexHandedness::Left, ESuperDexHandedness::Right}) {
    const TCHAR* Side = Handedness == ESuperDexHandedness::Left ? TEXT("Left") : TEXT("Right");
    UMochiArticulationBase* Bot = GetSkeletonBot_OpenXR(Handedness);
    if (!TestNotNull(TEXT("Bot"), Bot)) {
      continue;
    }
    for (int32 Trial = 0; Trial < 16; ++Trial) {
      // The OpenXR joint frames of the bot at random joint positions, as the runtime tracks them:
      // the bot's link frames are the OpenXR joint frames.
      TArray<double> Expected;
      Expected.SetNum(Bot->GetNumDofs());
      for (double& Position : Expected) {
        Position = Rng.FRandRange(-0.5, 0.5);
      }
      TArray<FTransform> LinkToParent;
      if (!TestTrue(
              TEXT("Link transforms"),
              Bot->ComputeLinkToParentTransforms(Expected, LinkToParent))) {
        return false;
      }
      TArray<FQuat> XrLocals;
      for (const FTransform& Link : LinkToParent) {
        XrLocals.Add(UnrealToBot(Link.GetRotation()));
      }

      const FSuperDexPose Pose =
          ConvertAsUnrealReportsIt(*this, MakeHand(MoveTemp(XrLocals)), Handedness);
      TestTrue(TEXT("Bot"), Pose.Bot == Bot);
      if (!TestEqual(TEXT("Joint position count"), Pose.DofPositions.Num(), Expected.Num())) {
        continue;
      }
      for (int32 Dof = 0; Dof < Expected.Num(); ++Dof) {
        TestNearlyEqual(
            FString::Printf(TEXT("%s trial %d DOF %d"), Side, Trial, Dof),
            Pose.DofPositions[Dof],
            Expected[Dof],
            1e-4);
      }
    }
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FOpenXRPoseSourceRootTest,
    "SuperDexOpenXRPoseSource.RootIsTheImportedWrist",
    OPENXR_POSE_SOURCE_TEST_FLAGS)

bool FOpenXRPoseSourceRootTest::RunTest(const FString& Parameters) {
  // OpenXR tracking space (+Y up, -Z forward, +X right) in the superdex bots' world (+Z up,
  // +X forward, -Y right).
  const FQuat XrToBotWorld = FQuat(
      FMatrix(FPlane(0, -1, 0, 0), FPlane(0, 0, 1, 0), FPlane(-1, 0, 0, 0), FPlane(0, 0, 0, 1)));
  FRandomStream Rng(7);
  for (int32 Trial = 0; Trial < 16; ++Trial) {
    const FRandomHand Hand = MakeRandomHand(Rng);
    const FSuperDexPose Pose = ConvertAsUnrealReportsIt(*this, Hand, ESuperDexHandedness::Right);
    const FQuat Wrist = Hand.XrWorld[static_cast<int32>(EHandKeypoint::Wrist)];
    const FQuat Expected = BotToUnreal(XrToBotWorld * Wrist);
    TestTrue(
        FString::Printf(TEXT("Trial %d root rotation"), Trial),
        AngleBetween(Pose.SkeletonRootToVrRoot.GetRotation(), Expected) < 1e-4);
    TestEqual(TEXT("Root location"), Pose.SkeletonRootToVrRoot.GetLocation(), FVector(1, 2, 3));
    TestEqual(TEXT("Root scale"), Pose.SkeletonRootToVrRoot.GetScale3D(), FVector::OneVector);
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FOpenXRPoseSourceHandScaleTest,
    "SuperDexOpenXRPoseSource.HandScaleIsTheRuntimeSkeletonScale",
    OPENXR_POSE_SOURCE_TEST_FLAGS)

bool FOpenXRPoseSourceHandScaleTest::RunTest(const FString& Parameters) {
  // Rest offsets [m] of the right openxr bot's links in their parents (ESuperDexBone_OXR order),
  // the OpenXR runtime's unit hand skeleton in its right-handed joint frames.
  const FVector Offsets[] = {
      {0, 0, 0},
      {-0.00212, -0.00548, -0.065331},
      {-0.028028, -0.019158, -0.035958},
      {0, 0, -0.032513},
      {0, 0, -0.033793},
      {0.00067, 0.001027, -0.024591},
      {-0.018724, -0.011042, -0.037178},
      {-0.004826, 0.003726, -0.058818},
      {0, 0, -0.037927},
      {0, 0, -0.024304},
      {0.000296, 0.001025, -0.022363},
      {-0.002515, -0.008416, -0.035016},
      {0.000789, 0.005873, -0.060631},
      {0, 0, -0.042927},
      {0, 0, -0.02755},
      {0.000309, 0.001137, -0.024965},
      {0.014992, -0.006016, -0.034776},
      {0.002473, -0.000514, -0.053918},
      {0, 0, -0.038996},
      {0, 0, -0.026573},
      {0.000258, 0.001608, -0.024326},
      {0.022999, -0.00942, -0.034074},
      {0, 0, -0.045651},
      {0, 0, -0.03072},
      {0, 0, -0.020311},
      {-0.000246, 0.001216, -0.021922},
  };
  constexpr int32 NumBones = static_cast<int32>(ESuperDexBone_OXR::Count);
  static_assert(UE_ARRAY_COUNT(Offsets) == NumBones);
  // Unreal units per meter: the bots' 100, and a world shown at twice that size.
  constexpr double kBotUnitsPerMeter = 100.0;
  constexpr double WorldScales[] = {1.0, 2.0};
  FRandomStream Rng(11);
  for (const double WorldScale : WorldScales) {
    for (const double HandScale : {0.75, 1.0, 1.2}) {
      for (int32 Trial = 0; Trial < 8; ++Trial) {
        // Joints placed as the runtime places them: the unit skeleton scaled by the hand scale, in
        // a random pose.
        const FRandomHand Hand = MakeRandomHand(Rng);
        TArray<FVector> XrPositions;
        XrPositions.SetNumZeroed(NumBones);
        TArray<FVector> Locations;
        Locations.SetNum(EHandKeypointCount);
        for (int32 Bone = 0; Bone < NumBones; ++Bone) {
          const ESuperDexBone_OXR Parent = kSuperDexBoneParents_OXR[Bone];
          if (Parent != ESuperDexBone_OXR::Invalid) {
            const int32 P = static_cast<int32>(Parent);
            const FQuat ParentWorld = Hand.XrWorld[GetXrHandJointOfBone_OXR(Parent)];
            XrPositions[Bone] =
                XrPositions[P] + ParentWorld.RotateVector(Offsets[Bone] * HandScale);
          }
          const FVector& X = XrPositions[Bone];
          Locations[GetXrHandJointOfBone_OXR(static_cast<ESuperDexBone_OXR>(Bone))] =
              FVector(-X.Z, X.X, X.Y) * kBotUnitsPerMeter * WorldScale;
        }
        // The bot must match the hand as the world shows it.
        TestNearlyEqual(
            FString::Printf(TEXT("World %.0f scale %.2f trial %d"), WorldScale, HandScale, Trial),
            USuperDexOpenXRPoseSource::EstimateHandScale(Locations),
            HandScale * WorldScale,
            1e-4);
      }
    }
  }

  // Coincident joints are degenerate.
  TArray<FVector> Degenerate;
  Degenerate.Init(FVector(1, 2, 3), EHandKeypointCount);
  TestEqual(TEXT("Degenerate"), USuperDexOpenXRPoseSource::EstimateHandScale(Degenerate), 1.0);
  return true;
}

#undef OPENXR_POSE_SOURCE_TEST_FLAGS

#endif // WITH_DEV_AUTOMATION_TESTS
