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

#include "SuperDexPoseFunctionLibrary.h"
#include "Skeletons/SuperDexSkeletonBots.h"

#include "Math/RandomStream.h"
#include "Misc/AutomationTest.h"
#include "MochiArticulationBase.h"
#include "MochiTypes.h"

#if WITH_DEV_AUTOMATION_TESTS

#define SUPERDEX_POSE_TEST_FLAGS \
  (EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

namespace {

// FQuat::AngularDistance takes acos of a value that rounds above 1 for equal rotations.
double AngleBetween(const FQuat& A, const FQuat& B) {
  const FQuat D = A.Inverse() * B;
  return 2.0 * FMath::Atan2(FVector(D.X, D.Y, D.Z).Size(), FMath::Abs(D.W));
}

FTransform RandomTransform(FRandomStream& Rng, double Scale = 1.0) {
  return FTransform(
      FQuat(Rng.GetUnitVector(), Rng.FRandRange(-PI, PI)),
      Rng.GetUnitVector() * Rng.FRandRange(0.0, 10.0),
      FVector(Scale));
}

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexPoseFromLinkTransformsTest,
    "SuperDexPose.FunctionLibrary.PoseFromLinkTransformsPlacesEveryLink",
    SUPERDEX_POSE_TEST_FLAGS)

bool FSuperDexPoseFromLinkTransformsTest::RunTest(const FString& Parameters) {
  // Link transforms of a skeleton bot at random joint positions with its root link moved off the
  // pose root, which is not a joint position: in both spaces, the pose has those joint positions
  // and, through SkeletonRootToVrRoot, puts every link where the transforms do.
  FRandomStream Rng(5);
  for (UMochiArticulationBase* Bot :
       {GetSkeletonBot_OpenXR(ESuperDexHandedness::Left),
        GetSkeletonBot_OculusXR(ESuperDexHandedness::Right)}) {
    if (!TestNotNull(TEXT("Bot"), Bot)) {
      continue;
    }
    for (const bool bToParent : {true, false}) {
      const TCHAR* SpaceName = bToParent ? TEXT("parent") : TEXT("root");
      for (int32 Trial = 0; Trial < 8; ++Trial) {
        TArray<double> Expected;
        Expected.SetNum(Bot->GetNumDofs());
        for (double& Position : Expected) {
          Position = Rng.FRandRange(-0.5, 0.5);
        }
        TArray<FTransform> LinkToRoot;
        TArray<FTransform> LinkTransforms;
        if (!TestTrue(
                TEXT("Link transforms"),
                Bot->ComputeLinkToArticulationRootTransforms(Expected, LinkToRoot) &&
                    (bToParent ? Bot->ComputeLinkToParentTransforms(Expected, LinkTransforms)
                               : Bot->ComputeLinkToArticulationRootTransforms(
                                     Expected, LinkTransforms)))) {
          return false;
        }
        // The root link moved off the pose root: only its own transform changes relative to its
        // parent, every transform changes relative to the root.
        const FTransform RootLinkOffset = RandomTransform(Rng);
        if (bToParent) {
          LinkTransforms[0] = LinkTransforms[0] * RootLinkOffset;
        } else {
          for (FTransform& Link : LinkTransforms) {
            Link = Link * RootLinkOffset;
          }
        }
        const FTransform PoseRoot = RandomTransform(Rng, Rng.FRandRange(0.5, 1.5));

        FSuperDexPose Pose;
        Pose.SkeletonRootToVrRoot = PoseRoot;
        if (!TestTrue(
                TEXT("Pose from link transforms"),
                bToParent
                    ? USuperDexPoseFunctionLibrary::SetSuperDexPoseFromLinkToParentTransforms(
                          Bot, LinkTransforms, Pose)
                    : USuperDexPoseFunctionLibrary::SetSuperDexPoseFromLinkToSkeletonRootTransforms(
                          Bot, LinkTransforms, Pose))) {
          continue;
        }
        TestTrue(TEXT("Bot"), Pose.Bot == Bot);
        if (!TestEqual(TEXT("Joint position count"), Pose.DofPositions.Num(), Expected.Num())) {
          continue;
        }
        for (int32 Dof = 0; Dof < Expected.Num(); ++Dof) {
          TestNearlyEqual(
              FString::Printf(
                  TEXT("%s %s trial %d DOF %d"), *Bot->GetName(), SpaceName, Trial, Dof),
              Pose.DofPositions[Dof],
              Expected[Dof],
              1e-4);
        }
        TArray<FTransform> PoseLinkToRoot;
        if (!TestTrue(
                TEXT("Pose link transforms"),
                USuperDexPoseFunctionLibrary::ComputeSuperDexPoseLinkToSkeletonRootTransforms(
                    Pose, PoseLinkToRoot))) {
          continue;
        }
        for (int32 Link = 0; Link < LinkToRoot.Num(); ++Link) {
          const FTransform ExpectedToVrRoot = LinkToRoot[Link] * RootLinkOffset * PoseRoot;
          const FTransform Actual = PoseLinkToRoot[Link] * Pose.SkeletonRootToVrRoot;
          const FString What = FString::Printf(
              TEXT("%s %s trial %d link %d"), *Bot->GetName(), SpaceName, Trial, Link);
          TestTrue(
              What + TEXT(" location"),
              FVector::Dist(Actual.GetLocation(), ExpectedToVrRoot.GetLocation()) < 1e-3);
          TestTrue(
              What + TEXT(" rotation"),
              AngleBetween(Actual.GetRotation(), ExpectedToVrRoot.GetRotation()) < 1e-4);
          TestTrue(
              What + TEXT(" scale"),
              Actual.GetScale3D().Equals(ExpectedToVrRoot.GetScale3D(), 1e-6));
        }
      }
    }
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexPoseFromWrongLinkCountTest,
    "SuperDexPose.FunctionLibrary.PoseFromLinkTransformsNeedsOnePerLink",
    SUPERDEX_POSE_TEST_FLAGS)

bool FSuperDexPoseFromWrongLinkCountTest::RunTest(const FString& Parameters) {
  UMochiArticulationBase* Bot = GetSkeletonBot_OpenXR(ESuperDexHandedness::Right);
  if (!TestNotNull(TEXT("Bot"), Bot)) {
    return false;
  }
  const FTransform PoseRoot(FQuat(FVector::ZAxisVector, 1.0), FVector(1, 2, 3));
  FSuperDexPose Pose;
  Pose.SkeletonRootToVrRoot = PoseRoot;
  TArray<FQuat> TooFew;
  TooFew.Init(FQuat::Identity, 3);
  TestFalse(
      TEXT("Too few links"),
      USuperDexPoseFunctionLibrary::SetSuperDexPoseFromLinkRotations(Bot, TooFew, Pose));
  TestNull(TEXT("Bot unchanged"), Pose.Bot.Get());
  TestEqual(TEXT("Joint positions unchanged"), Pose.DofPositions.Num(), 0);
  TestTrue(TEXT("Root unchanged"), Pose.SkeletonRootToVrRoot.Equals(PoseRoot));
  TestFalse(TEXT("Not on a bot"), USuperDexPoseFunctionLibrary::IsSuperDexPoseOnBot(Pose));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexPoseLinkToVrRootTest,
    "SuperDexPose.FunctionLibrary.LinkToVrRootPlacesNamedLink",
    SUPERDEX_POSE_TEST_FLAGS)

bool FSuperDexPoseLinkToVrRootTest::RunTest(const FString& Parameters) {
  // A link found by name, at random joint positions, is where the bot's link transforms place it
  // under the pose root.
  UMochiArticulationBase* Bot = GetSkeletonBot_OpenXR(ESuperDexHandedness::Left);
  if (!TestNotNull(TEXT("Bot"), Bot)) {
    return false;
  }
  FRandomStream Rng(11);
  TArray<double> JointPositions;
  JointPositions.SetNum(Bot->GetNumDofs());
  for (double& Position : JointPositions) {
    Position = Rng.FRandRange(-0.5, 0.5);
  }
  FSuperDexPose Pose;
  Pose.SkeletonRootToVrRoot = RandomTransform(Rng, 0.8);
  TArray<FTransform> LinkToRoot;
  if (!TestTrue(
          TEXT("Pose from joint positions"),
          USuperDexPoseFunctionLibrary::ComputeSuperDexPoseFromJointPositions(
              Bot, JointPositions, Pose)) ||
      !TestTrue(
          TEXT("Link transforms"),
          Bot->ComputeLinkToArticulationRootTransforms(JointPositions, LinkToRoot))) {
    return false;
  }
  const TArray<FName> LinkNames = Bot->GetLinkNames();
  for (int32 Link = 0; Link < LinkNames.Num(); ++Link) {
    const int32 Found =
        USuperDexPoseFunctionLibrary::FindSuperDexPoseLinkIndex(Pose, LinkNames[Link]);
    TestEqual(TEXT("Link found by name"), Found, Link);
    FTransform LinkToVrRoot;
    if (!TestTrue(
            TEXT("Link to VrRoot"),
            USuperDexPoseFunctionLibrary::ComputeSuperDexPoseLinkToVrRoot(
                Pose, Found, LinkToVrRoot))) {
      continue;
    }
    const FTransform Expected = LinkToRoot[Link] * Pose.SkeletonRootToVrRoot;
    TestTrue(
        FString::Printf(TEXT("Link %d location"), Link),
        FVector::Dist(LinkToVrRoot.GetLocation(), Expected.GetLocation()) < 1e-3);
    TestTrue(
        FString::Printf(TEXT("Link %d rotation"), Link),
        AngleBetween(LinkToVrRoot.GetRotation(), Expected.GetRotation()) < 1e-4);
  }
  TestEqual(
      TEXT("Unknown link"),
      USuperDexPoseFunctionLibrary::FindSuperDexPoseLinkIndex(Pose, FName(TEXT("not_a_link"))),
      static_cast<int32>(INDEX_NONE));
  FTransform Unchanged = FTransform::Identity;
  TestFalse(
      TEXT("No link index"),
      USuperDexPoseFunctionLibrary::ComputeSuperDexPoseLinkToVrRoot(Pose, INDEX_NONE, Unchanged));
  Pose.DofPositions.Pop();
  TestFalse(
      TEXT("Not on its bot"),
      USuperDexPoseFunctionLibrary::ComputeSuperDexPoseLinkToVrRoot(Pose, 0, Unchanged));
  TestTrue(TEXT("Output unchanged on failure"), Unchanged.Equals(FTransform::Identity));
  return true;
}

#undef SUPERDEX_POSE_TEST_FLAGS

#endif // WITH_DEV_AUTOMATION_TESTS
