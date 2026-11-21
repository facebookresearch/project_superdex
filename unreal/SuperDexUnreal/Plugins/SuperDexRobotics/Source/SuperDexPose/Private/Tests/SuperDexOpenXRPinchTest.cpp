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

#include "Skeletons/SuperDexOpenXRPinch.h"
#include "Skeletons/SuperDexSkeletonBots.h"
#include "SuperDexPoseFunctionLibrary.h"

#include "HAL/IConsoleManager.h"
#include "Math/RandomStream.h"
#include "Misc/AutomationTest.h"
#include "Misc/ScopeExit.h"
#include "MochiArticulationBase.h"
#include "MochiTypes.h"

#if WITH_DEV_AUTOMATION_TESTS

#define SUPERDEX_OPENXR_PINCH_TEST_FLAGS \
  (EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

namespace {
// FQuat::AngularDistance takes acos of a value that rounds above 1 for equal rotations.
double PinchTestAngleBetween(const FQuat& A, const FQuat& B) {
  const FQuat D = A.Inverse() * B;
  return 2.0 * FMath::Atan2(FVector(D.X, D.Y, D.Z).Size(), FMath::Abs(D.W));
}
} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FThumbFitReachableTest,
    "SuperDexPose.OpenXRPinch.ThumbFitReproducesReachableDistalFrames",
    SUPERDEX_OPENXR_PINCH_TEST_FLAGS)

bool FThumbFitReachableTest::RunTest(const FString&) {
  constexpr int32 Wrist = static_cast<int32>(ESuperDexBone_OXR::Wrist);
  constexpr int32 Base = static_cast<int32>(ESuperDexBone_OXR::ThumbMetacarpal);
  constexpr int32 Ip = static_cast<int32>(ESuperDexBone_OXR::ThumbDistal);
  for (const ESuperDexHandedness Hand : {ESuperDexHandedness::Left, ESuperDexHandedness::Right}) {
    auto* Bot = GetSkeletonBot_OpenXR(Hand);
    if (!TestNotNull(TEXT("OpenXR bot"), Bot)) {
      return false;
    }
    FRandomStream Rng(Hand == ESuperDexHandedness::Left ? 11 : 12);
    for (int32 Trial = 0; Trial < 16; ++Trial) {
      // A pose the bot reaches, within its thumb hinge limits.
      TArray<double> Dofs;
      Dofs.SetNum(Bot->GetNumDofs());
      for (double& Position : Dofs) {
        Position = Rng.FRandRange(-0.2, 0.6);
      }
      TArray<FTransform> LinkToParent, LinkToRoot;
      if (!TestTrue(
              TEXT("Bot FK"),
              Bot->ComputeLinkToParentTransforms(Dofs, LinkToParent) &&
                  Bot->ComputeLinkToArticulationRootTransforms(Dofs, LinkToRoot))) {
        return false;
      }
      const FTransform DistalToWrist = LinkToRoot[Ip].GetRelativeTransform(LinkToRoot[Wrist]);
      TArray<FQuat> Rotations;
      for (const FTransform& Link : LinkToParent) {
        Rotations.Add(Link.GetRotation());
      }
      // Only the MCP rotation is kept, to pick the MCP branch.
      Rotations[Base] = FQuat::Identity;
      Rotations[Ip] = FQuat::Identity;
      if (!TestTrue(
              TEXT("Thumb fit"), FitOpenXRThumbToDistalFrame(Hand, DistalToWrist, Rotations))) {
        return false;
      }
      FSuperDexPose Fit;
      TArray<FTransform> FitToRoot;
      if (!TestTrue(
              TEXT("Fit FK"),
              USuperDexPoseFunctionLibrary::SetSuperDexPoseFromLinkRotations(Bot, Rotations, Fit) &&
                  USuperDexPoseFunctionLibrary::ComputeSuperDexPoseLinkToSkeletonRootTransforms(
                      Fit, FitToRoot))) {
        return false;
      }
      const FTransform Reached = FitToRoot[Ip].GetRelativeTransform(FitToRoot[Wrist]);
      const FString What = FString::Printf(
          TEXT("%s trial %d"),
          Hand == ESuperDexHandedness::Left ? TEXT("Left") : TEXT("Right"),
          Trial);
      TestTrue(
          What + TEXT(" distal origin"),
          FVector::Dist(Reached.GetLocation(), DistalToWrist.GetLocation()) < 0.001);
      TestTrue(
          What + TEXT(" distal orientation"),
          PinchTestAngleBetween(Reached.GetRotation(), DistalToWrist.GetRotation()) < 1e-4);
    }
  }
  return true;
}

namespace {
// Link-to-wrist frames of an OpenXR pose given as link-to-parent rotations.
TArray<FTransform> OpenXRLinksToWrist(UMochiArticulationBase* Bot, const TArray<FQuat>& Rotations) {
  FSuperDexPose Pose;
  TArray<FTransform> ToRoot, ToWrist;
  if (!USuperDexPoseFunctionLibrary::SetSuperDexPoseFromLinkRotations(Bot, Rotations, Pose) ||
      !USuperDexPoseFunctionLibrary::ComputeSuperDexPoseLinkToSkeletonRootTransforms(
          Pose, ToRoot)) {
    return ToWrist;
  }
  for (const FTransform& Link : ToRoot) {
    ToWrist.Add(Link.GetRelativeTransform(ToRoot[0]));
  }
  return ToWrist;
}

bool RandomOpenXRPose(
    UMochiArticulationBase* Bot,
    FRandomStream& Rng,
    TArray<FQuat>& OutRotations,
    TArray<FTransform>& OutToWrist) {
  TArray<double> Dofs;
  Dofs.SetNum(Bot->GetNumDofs());
  for (double& Position : Dofs) {
    Position = Rng.FRandRange(-0.2, 0.6);
  }
  TArray<FTransform> LinkToParent;
  if (!Bot->ComputeLinkToParentTransforms(Dofs, LinkToParent)) {
    return false;
  }
  OutRotations.Reset();
  for (const FTransform& Link : LinkToParent) {
    OutRotations.Add(Link.GetRotation());
  }
  OutToWrist = OpenXRLinksToWrist(Bot, OutRotations);
  return OutToWrist.Num() == OutRotations.Num();
}
} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FPinchBlendReachedThumbTest,
    "SuperDexPose.OpenXRPinch.PinchBlendLeavesFingersWhenTheThumbReaches",
    SUPERDEX_OPENXR_PINCH_TEST_FLAGS)

bool FPinchBlendReachedThumbTest::RunTest(const FString&) {
  constexpr int32 Ip = static_cast<int32>(ESuperDexBone_OXR::ThumbDistal);
  for (const ESuperDexHandedness Hand : {ESuperDexHandedness::Left, ESuperDexHandedness::Right}) {
    auto* Bot = GetSkeletonBot_OpenXR(Hand);
    if (!TestNotNull(TEXT("OpenXR bot"), Bot)) {
      return false;
    }
    FRandomStream Rng(Hand == ESuperDexHandedness::Left ? 21 : 22);
    for (int32 Trial = 0; Trial < 16; ++Trial) {
      TArray<FQuat> Rotations;
      TArray<FTransform> ToWrist;
      if (!TestTrue(TEXT("Pose"), RandomOpenXRPose(Bot, Rng, Rotations, ToWrist))) {
        return false;
      }
      // Wide blend: every finger is weighted, and none moves without a thumb shortfall.
      for (const double Allowance : {0.0, 1.0}) {
        FOpenXRPinchBlend Blend;
        Blend.FullWeightDistance = 50.0;
        Blend.ZeroWeightDistance = 100.0;
        for (double& A : Blend.FlexAllowance) {
          A = Allowance;
        }
        TArray<FQuat> Blended = Rotations;
        if (!TestTrue(
                TEXT("Blend"),
                BlendOpenXRFingersIntoPinch(Hand, ToWrist[Ip].GetLocation(), Blended, Blend))) {
          return false;
        }
        for (int32 Link = 0; Link < Rotations.Num(); ++Link) {
          TestTrue(
              FString::Printf(
                  TEXT("Allowance %.0f trial %d link %d unchanged"), Allowance, Trial, Link),
              PinchTestAngleBetween(Rotations[Link], Blended[Link]) < 1e-6);
        }
      }
    }
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FPinchBlendShortfallTest,
    "SuperDexPose.OpenXRPinch.PinchBlendCarriesNearFingerPadsWithTheThumbShortfall",
    SUPERDEX_OPENXR_PINCH_TEST_FLAGS)

bool FPinchBlendShortfallTest::RunTest(const FString&) {
  constexpr int32 Ip = static_cast<int32>(ESuperDexBone_OXR::ThumbDistal);
  constexpr int32 ThumbTip = static_cast<int32>(ESuperDexBone_OXR::ThumbTip);
  constexpr int32 Proximal = static_cast<int32>(ESuperDexBone_OXR::IndexProximal);
  constexpr int32 Distal = static_cast<int32>(ESuperDexBone_OXR::IndexDistal);
  constexpr int32 Tip = static_cast<int32>(ESuperDexBone_OXR::IndexTip);
  constexpr int32 LittleDistal = static_cast<int32>(ESuperDexBone_OXR::LittleDistal);
  constexpr int32 LittleTip = static_cast<int32>(ESuperDexBone_OXR::LittleTip);
  for (const ESuperDexHandedness Hand : {ESuperDexHandedness::Left, ESuperDexHandedness::Right}) {
    auto* Bot = GetSkeletonBot_OpenXR(Hand);
    if (!TestNotNull(TEXT("OpenXR bot"), Bot)) {
      return false;
    }
    FRandomStream Rng(Hand == ESuperDexHandedness::Left ? 31 : 32);
    for (int32 Trial = 0; Trial < 16; ++Trial) {
      TArray<FQuat> Rotations;
      TArray<FTransform> ToWrist;
      if (!TestTrue(TEXT("Pose"), RandomOpenXRPose(Bot, Rng, Rotations, ToWrist))) {
        return false;
      }
      // A tracked thumb distal origin touching the index distal centerline, short of which the
      // bot's thumb distal lies.
      const FVector IndexEnd = ToWrist[Distal].TransformPosition(GetOpenXRFingerEndInDistal(
          ESuperDexFinger::Index,
          ToWrist[Tip].GetRelativeTransform(ToWrist[Distal]).GetLocation()));
      const FVector IndexMid = (ToWrist[Distal].GetLocation() + IndexEnd) / 2;
      const FVector Tracked = IndexMid + Rng.GetUnitVector() * 0.3;
      const FVector Shortfall = ToWrist[Ip].GetLocation() - Tracked;
      TArray<FQuat> Blended = Rotations;
      // Zero flex allowance: the knuckle turn alone.
      FOpenXRPinchBlend Turn;
      for (double& A : Turn.FlexAllowance) {
        A = 0.0;
      }
      if (!TestTrue(TEXT("Blend"), BlendOpenXRFingersIntoPinch(Hand, Tracked, Blended, Turn))) {
        return false;
      }
      const TArray<FTransform> After = OpenXRLinksToWrist(Bot, Blended);
      if (!TestEqual(TEXT("FK"), After.Num(), ToWrist.Num())) {
        return false;
      }
      // Pad point as in BlendOpenXRFingersIntoPinch.
      const FVector PadInDistal = GetOpenXRFingerPadInDistal(
          ESuperDexFinger::Index, ToWrist[Tip].GetRelativeTransform(ToWrist[Distal]).GetLocation());
      const FVector Knuckle = ToWrist[Proximal].GetLocation();
      const FVector PadBefore = ToWrist[Distal].TransformPosition(PadInDistal);
      const FVector PadAfter = After[Distal].TransformPosition(PadInDistal);
      const FString What = FString::Printf(
          TEXT("%s trial %d"),
          Hand == ESuperDexHandedness::Left ? TEXT("Left") : TEXT("Right"),
          Trial);
      TestTrue(
          What + TEXT(" index pad turns toward its shifted position"),
          FVector::CrossProduct(
              (PadAfter - Knuckle).GetSafeNormal(),
              (PadBefore + Shortfall - Knuckle).GetSafeNormal())
                      .Size() < 1e-5 &&
              ((PadAfter - Knuckle) | (PadBefore + Shortfall - Knuckle)) > 0);
      TestTrue(
          What + TEXT(" index knuckle-to-pad distance kept"),
          FMath::IsNearlyEqual(
              FVector::Dist(PadAfter, Knuckle), FVector::Dist(PadBefore, Knuckle), 1e-4));
      // A finger beyond the zero-weight distance is untouched.
      FVector OnThumb, OnLittle;
      const FVector TrackedTip = Tracked +
          ToWrist[Ip].GetRotation().RotateVector(GetOpenXRFingerEndInDistal(
              ESuperDexFinger::Thumb,
              ToWrist[ThumbTip].GetRelativeTransform(ToWrist[Ip]).GetLocation()));
      FMath::SegmentDistToSegmentSafe(
          Tracked,
          TrackedTip,
          ToWrist[LittleDistal].GetLocation(),
          ToWrist[LittleDistal].TransformPosition(GetOpenXRFingerEndInDistal(
              ESuperDexFinger::Pinky,
              ToWrist[LittleTip].GetRelativeTransform(ToWrist[LittleDistal]).GetLocation())),
          OnThumb,
          OnLittle);
      if (FVector::Dist(OnThumb, OnLittle) >= FOpenXRPinchBlend().ZeroWeightDistance) {
        TestTrue(
            What + TEXT(" far little finger unchanged"),
            After[LittleDistal].Equals(ToWrist[LittleDistal], 1e-6));
      }
    }
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FPinchBlendFlexAllowanceTest,
    "SuperDexPose.OpenXRPinch.PinchBlendFlexAllowanceSpansKnuckleTurnAndLeastSquares",
    SUPERDEX_OPENXR_PINCH_TEST_FLAGS)

bool FPinchBlendFlexAllowanceTest::RunTest(const FString&) {
  constexpr int32 Ip = static_cast<int32>(ESuperDexBone_OXR::ThumbDistal);
  constexpr int32 ThumbTip = static_cast<int32>(ESuperDexBone_OXR::ThumbTip);
  const int32 Proximals[] = {
      static_cast<int32>(ESuperDexBone_OXR::IndexProximal),
      static_cast<int32>(ESuperDexBone_OXR::MiddleProximal),
      static_cast<int32>(ESuperDexBone_OXR::RingProximal),
      static_cast<int32>(ESuperDexBone_OXR::LittleProximal)};
  int32 DistalContacts = 0;
  int32 IntermediateContacts = 0;
  for (const ESuperDexHandedness Hand : {ESuperDexHandedness::Left, ESuperDexHandedness::Right}) {
    auto* Bot = GetSkeletonBot_OpenXR(Hand);
    if (!TestNotNull(TEXT("OpenXR bot"), Bot)) {
      return false;
    }
    FRandomStream Rng(Hand == ESuperDexHandedness::Left ? 51 : 52);
    for (int32 Trial = 0; Trial < 16; ++Trial) {
      TArray<FQuat> Rotations;
      TArray<FTransform> ToWrist;
      if (!TestTrue(TEXT("Pose"), RandomOpenXRPose(Bot, Rng, Rotations, ToWrist))) {
        return false;
      }
      // A 0.5 cm thumb shortfall, with every finger fully weighted.
      const FVector Tracked = ToWrist[Ip].GetLocation() + Rng.GetUnitVector() * 0.5;
      auto Run = [&](double Allowance) {
        FOpenXRPinchBlend Blend;
        Blend.FullWeightDistance = 50.0;
        Blend.ZeroWeightDistance = 100.0;
        for (double& A : Blend.FlexAllowance) {
          A = Allowance;
        }
        TArray<FQuat> Out = Rotations;
        BlendOpenXRFingersIntoPinch(Hand, Tracked, Out, Blend);
        return Out;
      };
      const TArray<FQuat> Turn = Run(0.0);
      const TArray<FQuat> Flex = Run(1.0);
      const TArray<FTransform> AfterTurn = OpenXRLinksToWrist(Bot, Turn);
      const TArray<FTransform> AfterFlex = OpenXRLinksToWrist(Bot, Flex);
      if (!TestEqual(TEXT("FK"), AfterTurn.Num(), ToWrist.Num()) ||
          !TestEqual(TEXT("FK"), AfterFlex.Num(), ToWrist.Num())) {
        return false;
      }
      const FVector TrackedTip = Tracked +
          ToWrist[Ip].GetRotation().RotateVector(GetOpenXRFingerEndInDistal(
              ESuperDexFinger::Thumb,
              ToWrist[ThumbTip].GetRelativeTransform(ToWrist[Ip]).GetLocation()));
      for (const int32 Proximal : Proximals) {
        const int32 Intermediate = Proximal + 1, Distal = Proximal + 2, Tip = Proximal + 3;
        // Distal links 9, 14, 19, 24 are the index, middle, ring and little fingers'.
        const ESuperDexFinger Finger = static_cast<ESuperDexFinger>((Distal - 4) / 5);
        const FVector TipInDistal =
            ToWrist[Tip].GetRelativeTransform(ToWrist[Distal]).GetLocation();
        const FString What = FString::Printf(TEXT("Trial %d finger %d"), Trial, Distal);
        // Zero allowance is the knuckle turn: only the proximal rotation changes.
        TestTrue(
            What + TEXT(" zero allowance keeps the intermediate joint"),
            PinchTestAngleBetween(Turn[Intermediate], Rotations[Intermediate]) < 1e-6);
        TestTrue(
            What + TEXT(" zero allowance keeps the distal joint"),
            PinchTestAngleBetween(Turn[Distal], Rotations[Distal]) < 1e-6);
        // Where the thumb is, as BlendOpenXRFingersIntoPinch gates the flex allowance.
        FVector A, B;
        FMath::SegmentDistToSegmentSafe(
            Tracked,
            TrackedTip,
            ToWrist[Distal].GetLocation(),
            ToWrist[Distal].TransformPosition(GetOpenXRFingerEndInDistal(Finger, TipInDistal)),
            A,
            B);
        const double ToDistal = FVector::Dist(A, B);
        FMath::SegmentDistToSegmentSafe(
            Tracked,
            TrackedTip,
            ToWrist[Intermediate].GetLocation(),
            ToWrist[Distal].GetLocation(),
            A,
            B);
        const double Lead = FVector::Dist(A, B) - ToDistal;
        if (Lead <= 0.1) {
          for (const int32 Link : {Proximal, Intermediate, Distal}) {
            TestTrue(
                What + TEXT(" an intermediate contact keeps the knuckle turn"),
                PinchTestAngleBetween(Flex[Link], Turn[Link]) < 1e-6);
          }
          ++IntermediateContacts;
        }
      }
    }
  }
  TestTrue(TEXT("Some intermediate contacts were checked"), IntermediateContacts > 0);

  // Distal contacts, which random poses rarely give: sample poses until the thumb, 0.5 cm from the
  // bot's, is enough nearer a finger's distal segment than its intermediate one.
  for (const ESuperDexHandedness Hand : {ESuperDexHandedness::Left, ESuperDexHandedness::Right}) {
    auto* Bot = GetSkeletonBot_OpenXR(Hand);
    FRandomStream Rng(Hand == ESuperDexHandedness::Left ? 61 : 62);
    int32 HandContacts = 0;
    for (int32 Sample = 0; Sample < 4000 && HandContacts < 12; ++Sample) {
      TArray<FQuat> Rotations;
      TArray<FTransform> ToWrist;
      if (!TestTrue(TEXT("Pose"), RandomOpenXRPose(Bot, Rng, Rotations, ToWrist))) {
        return false;
      }
      const FVector Tracked = ToWrist[Ip].GetLocation() + Rng.GetUnitVector() * 0.5;
      const FVector TrackedTip = Tracked +
          ToWrist[Ip].GetRotation().RotateVector(GetOpenXRFingerEndInDistal(
              ESuperDexFinger::Thumb,
              ToWrist[ThumbTip].GetRelativeTransform(ToWrist[Ip]).GetLocation()));
      TArray<FTransform> AfterTurn, AfterFlex;
      for (const int32 Proximal : Proximals) {
        const int32 Intermediate = Proximal + 1, Distal = Proximal + 2, Tip = Proximal + 3;
        const ESuperDexFinger Finger = static_cast<ESuperDexFinger>((Distal - 4) / 5);
        const FVector TipInDistal =
            ToWrist[Tip].GetRelativeTransform(ToWrist[Distal]).GetLocation();
        FVector A, B;
        FMath::SegmentDistToSegmentSafe(
            Tracked,
            TrackedTip,
            ToWrist[Distal].GetLocation(),
            ToWrist[Distal].TransformPosition(GetOpenXRFingerEndInDistal(Finger, TipInDistal)),
            A,
            B);
        const double ToDistal = FVector::Dist(A, B);
        FMath::SegmentDistToSegmentSafe(
            Tracked,
            TrackedTip,
            ToWrist[Intermediate].GetLocation(),
            ToWrist[Distal].GetLocation(),
            A,
            B);
        if (FVector::Dist(A, B) - ToDistal < 0.6) {
          continue;
        }
        if (AfterTurn.IsEmpty()) {
          auto Run = [&](double Allowance) {
            FOpenXRPinchBlend Blend;
            Blend.FullWeightDistance = 50.0;
            Blend.ZeroWeightDistance = 100.0;
            for (double& Value : Blend.FlexAllowance) {
              Value = Allowance;
            }
            TArray<FQuat> Out = Rotations;
            BlendOpenXRFingersIntoPinch(Hand, Tracked, Out, Blend);
            return OpenXRLinksToWrist(Bot, Out);
          };
          AfterTurn = Run(0.0);
          AfterFlex = Run(1.0);
          if (!TestEqual(TEXT("FK"), AfterFlex.Num(), ToWrist.Num()) ||
              !TestEqual(TEXT("FK"), AfterTurn.Num(), ToWrist.Num())) {
            return false;
          }
        }
        // Least squares also follows the knuckle-to-pad component the knuckle turn leaves.
        const FVector PadInDistal = GetOpenXRFingerPadInDistal(Finger, TipInDistal);
        const FVector Goal =
            ToWrist[Distal].TransformPosition(PadInDistal) + (ToWrist[Ip].GetLocation() - Tracked);
        const double FlexMiss =
            FVector::Dist(AfterFlex[Distal].TransformPosition(PadInDistal), Goal);
        const double TurnMiss =
            FVector::Dist(AfterTurn[Distal].TransformPosition(PadInDistal), Goal);
        TestTrue(
            FString::Printf(
                TEXT(
                    "Sample %d finger %d full allowance on a distal contact comes at least as close"),
                Sample,
                Distal),
            FlexMiss <= TurnMiss + 0.02);
        ++HandContacts;
        ++DistalContacts;
      }
    }
  }
  TestTrue(TEXT("Some distal contacts were checked"), DistalContacts > 0);
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FFingertipSphereTest,
    "SuperDexPose.OpenXRPinch.FingertipSpheresSitInsideTheDistalLinks",
    SUPERDEX_OPENXR_PINCH_TEST_FLAGS)

bool FFingertipSphereTest::RunTest(const FString&) {
  // The tip bones are fingertip-sphere centers on the distal centerline, about one radius inside
  // the fingertip, mirrored between the hands.
  TArray<FTransform> Rest[2];
  int32 Hand = 0;
  for (const ESuperDexHandedness Handedness :
       {ESuperDexHandedness::Left, ESuperDexHandedness::Right}) {
    auto* Bot = GetSkeletonBot_OpenXR(Handedness);
    TArray<double> Dofs;
    if (!TestNotNull(TEXT("OpenXR bot"), Bot)) {
      return false;
    }
    Dofs.SetNumZeroed(Bot->GetNumDofs());
    if (!TestTrue(TEXT("FK"), Bot->ComputeLinkToParentTransforms(Dofs, Rest[Hand++]))) {
      return false;
    }
  }
  for (const ESuperDexFinger Finger :
       {ESuperDexFinger::Thumb,
        ESuperDexFinger::Index,
        ESuperDexFinger::Middle,
        ESuperDexFinger::Ring,
        ESuperDexFinger::Pinky}) {
    const int32 Tip = static_cast<int32>(GetFingerTipBone_OXR(Finger));
    const double Radius = GetFingerTipRadius_OXR(Finger);
    const FVector Left = Rest[0][Tip].GetLocation();
    const FVector Right = Rest[1][Tip].GetLocation();
    const FString What = FString::Printf(TEXT("Finger %d"), static_cast<int32>(Finger));
    TestTrue(What + TEXT(" radius"), Radius > 0.4 && Radius < 1.0);
    TestTrue(What + TEXT(" center along the distal link"), Right.Z < -1.2 && Right.Z > -2.0);
    TestTrue(What + TEXT(" center near the centerline"), FVector2D(Right.X, Right.Y).Size() < 0.25);
    TestTrue(
        What + TEXT(" tip rotated like its distal link"),
        Rest[1][Tip].GetRotation().Equals(FQuat::Identity, 1e-6));
    TestTrue(
        What + TEXT(" left mirrors right"), Left.Equals(FVector(-Right.X, Right.Y, Right.Z), 1e-6));
    TestTrue(
        What + TEXT(" pad below the center, end beyond it"),
        GetOpenXRFingerPadInDistal(Finger, Right).Y > Right.Y &&
            GetOpenXRFingerEndInDistal(Finger, Right).Z < Right.Z);
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FPinchCorrectionOverrideTest,
    "SuperDexPose.OpenXRPinch.ConsoleOverrideBeatsTheProjectSetting",
    SUPERDEX_OPENXR_PINCH_TEST_FLAGS)

bool FPinchCorrectionOverrideTest::RunTest(const FString&) {
  IConsoleVariable* Enable =
      IConsoleManager::Get().FindConsoleVariable(TEXT("superdex.OpenXRPinchCorrection.Enable"));
  if (!TestNotNull(TEXT("superdex.OpenXRPinchCorrection.Enable"), Enable)) {
    return false;
  }
  const int32 Saved = Enable->GetInt();
  ON_SCOPE_EXIT {
    Enable->Set(Saved, ECVF_SetByCode);
  };
  Enable->Set(-1, ECVF_SetByCode);
  TestTrue(TEXT("-1 follows an enabled setting"), ShouldApplyOpenXRPinchCorrection(true));
  TestFalse(TEXT("-1 follows a disabled setting"), ShouldApplyOpenXRPinchCorrection(false));
  Enable->Set(0, ECVF_SetByCode);
  TestFalse(TEXT("0 turns it off"), ShouldApplyOpenXRPinchCorrection(true));
  Enable->Set(1, ECVF_SetByCode);
  TestTrue(TEXT("1 turns it on"), ShouldApplyOpenXRPinchCorrection(false));
  return true;
}

#undef SUPERDEX_OPENXR_PINCH_TEST_FLAGS

#endif // WITH_DEV_AUTOMATION_TESTS
