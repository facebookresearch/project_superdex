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

#include "MochiTypes.h"
#include "Skeletons/SuperDexSkeletonBots.h"
#include "SuperDexBot.h"

#include "Algo/MinElement.h"
#include "HAL/IConsoleManager.h"
#include "Misc/ScopeLock.h"

DEFINE_LOG_CATEGORY_STATIC(LogSuperDexOpenXRPinch, Log, All);

namespace {

TAutoConsoleVariable<int32> CVarPinchCorrectionEnable(
    TEXT("superdex.OpenXRPinchCorrection.Enable"),
    -1,
    TEXT("Override of every pose source's pinch correction setting (OpenXR thumb refit and finger ")
        TEXT("blend). -1 (default): follow each source's project setting. 0: off. 1: on."));

constexpr int32 kThumbBase = static_cast<int32>(ESuperDexBone_OXR::ThumbMetacarpal);
constexpr int32 kThumbMcp = static_cast<int32>(ESuperDexBone_OXR::ThumbProximal);
constexpr int32 kThumbIp = static_cast<int32>(ESuperDexBone_OXR::ThumbDistal);
constexpr int32 kThumbTip = static_cast<int32>(ESuperDexBone_OXR::ThumbTip);
constexpr int32 kNumOpenXRBones = static_cast<int32>(ESuperDexBone_OXR::Count);

struct FFingerLinks {
  ESuperDexFinger Finger;
  int32 Proximal;
  int32 Distal;
  int32 Tip;
};
constexpr FFingerLinks kFingers[] = {
    {ESuperDexFinger::Index,
     static_cast<int32>(ESuperDexBone_OXR::IndexProximal),
     static_cast<int32>(ESuperDexBone_OXR::IndexDistal),
     static_cast<int32>(ESuperDexBone_OXR::IndexTip)},
    {ESuperDexFinger::Middle,
     static_cast<int32>(ESuperDexBone_OXR::MiddleProximal),
     static_cast<int32>(ESuperDexBone_OXR::MiddleDistal),
     static_cast<int32>(ESuperDexBone_OXR::MiddleTip)},
    {ESuperDexFinger::Ring,
     static_cast<int32>(ESuperDexBone_OXR::RingProximal),
     static_cast<int32>(ESuperDexBone_OXR::RingDistal),
     static_cast<int32>(ESuperDexBone_OXR::RingTip)},
    {ESuperDexFinger::Pinky,
     static_cast<int32>(ESuperDexBone_OXR::LittleProximal),
     static_cast<int32>(ESuperDexBone_OXR::LittleDistal),
     static_cast<int32>(ESuperDexBone_OXR::LittleTip)},
};

// The least-squares pad move: damping (cm^2 per rad^2) of the knuckle rotation, and how much more
// the intermediate and distal flexion is damped. Chosen on the 2026-11-14 captures.
constexpr double kPinchDamping = 1.0;
constexpr double kPinchFlexDamping = 4.0;
constexpr int32 kPinchSteps = 2;

// The flex allowance fades from 0 to 1 as the thumb's distance to the finger's intermediate
// centerline exceeds its distance to the distal centerline by these (cm).
// Recorded contacts on the distal segment exceed 0.78 cm; contacts on the intermediate are below 0.
constexpr double kDistalContactStart = 0.1;
constexpr double kDistalContactFull = 0.6;

/* A revolute joint of the OpenXR bot: LinkToParent(A) = (Rotation(A), Offset). */
struct FHinge {
  FQuat Rest = FQuat::Identity;
  FVector Offset = FVector::ZeroVector;
  // Unit axis in the joint (rest child) frame, so Rotation(A) = Rest * FQuat(Axis, A).
  FVector Axis = FVector::XAxisVector;
  double Min = -UE_PI;
  double Max = UE_PI;

  FQuat Rotation(double Angle) const {
    return (Rest * FQuat(Axis, Angle)).GetNormalized();
  }
  // Nearest angle of a LinkToParent rotation: its twist about the axis.
  double Angle(const FQuat& LinkToParent) const {
    FQuat Motion = (Rest.Inverse() * LinkToParent).GetNormalized();
    if (Motion.W < 0) {
      Motion *= -1.0;
    }
    return 2.0 * FMath::Atan2(FVector(Motion.X, Motion.Y, Motion.Z) | Axis, Motion.W);
  }
};

/* The OpenXR bot's thumb: a spherical metacarpal at a fixed pivot on the wrist, then the MCP and
 * IP hinges. */
struct FThumbKinematics {
  const UMochiArticulationBase* Bot = nullptr;
  bool bValid = false;
  FVector Pivot = FVector::ZeroVector; // Metacarpal origin in the wrist frame.
  FHinge Mcp;
  FHinge Ip;
  // Every link's origin in its parent's frame (the OpenXR bot has no prismatic joints).
  TArray<FVector, TInlineAllocator<kNumOpenXRBones>> Offsets;
  // Every revolute link's hinge (default elsewhere).
  TArray<FHinge, TInlineAllocator<kNumOpenXRBones>> Hinges;
};

int32 NumPoseDofs(mochi::ArticulatedJointType Type) {
  switch (Type) {
    case mochi::ArticulatedJointType::Revolute:
    case mochi::ArticulatedJointType::Prismatic:
      return 1;
    case mochi::ArticulatedJointType::Spherical:
      return 3;
    case mochi::ArticulatedJointType::Free:
      return 6;
    case mochi::ArticulatedJointType::Hard:
    case mochi::ArticulatedJointType::Cycle:
    case mochi::ArticulatedJointType::Count:
      break;
  }
  return 0;
}

FThumbKinematics ComputeThumbKinematics(const USuperDexBot& Bot) {
  FThumbKinematics K;
  K.Bot = &Bot;
  const superdex::robotics::BotPrefab& Prefab = Bot.GetBotPrefab();
  if (static_cast<int32>(Prefab.joints.size()) != static_cast<int32>(ESuperDexBone_OXR::Count) ||
      Prefab.links[kThumbBase].parentLink != static_cast<int32>(ESuperDexBone_OXR::Wrist) ||
      Prefab.joints[kThumbBase].type != mochi::ArticulatedJointType::Spherical) {
    return K;
  }
  TArray<double> Dofs;
  Dofs.SetNumZeroed(Bot.GetNumDofs());
  TArray<FTransform> Rest;
  if (!Bot.ComputeLinkToParentTransforms(Dofs, Rest)) {
    return K;
  }
  K.Pivot = Rest[kThumbBase].GetLocation();
  for (int32 Link = 0; Link < kNumOpenXRBones; ++Link) {
    if (Link > 0 &&
        Prefab.links[Link].parentLink != static_cast<int32>(SuperDexBoneParents_OXR[Link])) {
      return K;
    }
    K.Offsets.Add(Rest[Link].GetLocation());
  }

  K.Hinges.SetNum(kNumOpenXRBones);
  int32 Dof = 0;
  for (int32 Link = 1; Link < static_cast<int32>(Prefab.joints.size()); ++Link) {
    const auto& Joint = Prefab.joints[Link];
    if ((Link == kThumbMcp || Link == kThumbIp) &&
        Joint.type != mochi::ArticulatedJointType::Revolute) {
      return K;
    }
    for (const FFingerLinks& Finger : kFingers) {
      if ((Link == Finger.Proximal + 1 || Link == Finger.Distal) &&
          Joint.type != mochi::ArticulatedJointType::Revolute) {
        return K;
      }
    }
    if (Joint.type == mochi::ArticulatedJointType::Revolute) {
      FHinge& Hinge = K.Hinges[Link];
      // The DOF's motion, read back from the bot's own kinematics in Unreal space.
      constexpr double kProbe = 0.5;
      Dofs[Dof] = kProbe;
      TArray<FTransform> Moved;
      const bool bMoved = Bot.ComputeLinkToParentTransforms(Dofs, Moved);
      Dofs[Dof] = 0.0;
      if (!bMoved) {
        return K;
      }
      Hinge.Rest = Rest[Link].GetRotation();
      Hinge.Offset = Rest[Link].GetLocation();
      FVector Axis;
      double Angle;
      (Hinge.Rest.Inverse() * Moved[Link].GetRotation())
          .GetNormalized()
          .ToAxisAndAngle(Axis, Angle);
      if (!FMath::IsNearlyEqual(Angle, kProbe, 1e-6) ||
          !Moved[Link].GetLocation().Equals(Hinge.Offset, 1e-6)) {
        return K;
      }
      Hinge.Axis = Axis.GetSafeNormal();
      // Mochi limits a revolute joint to the limit vectors' components along its axis.
      const mochi::Real3 NativeAxis = mochi::Normalize(Joint.axis);
      const double A =
          Joint.minLimit ? static_cast<double>(mochi::Dot(NativeAxis, *Joint.minLimit)) : -UE_PI;
      const double B =
          Joint.maxLimit ? static_cast<double>(mochi::Dot(NativeAxis, *Joint.maxLimit)) : UE_PI;
      Hinge.Min = FMath::Min(A, B);
      Hinge.Max = FMath::Max(A, B);
    }
    Dof += NumPoseDofs(Joint.type);
  }
  K.Mcp = K.Hinges[kThumbMcp];
  K.Ip = K.Hinges[kThumbIp];
  K.bValid = true;
  return K;
}

FThumbKinematics GetThumbKinematics(ESuperDexHandedness Handedness) {
  static FCriticalSection Lock;
  static FThumbKinematics Cache[2];
  const auto* Bot = Cast<USuperDexBot>(GetSkeletonBot_OpenXR(Handedness));
  if (Bot == nullptr) {
    return {};
  }
  FScopeLock Guard(&Lock);
  FThumbKinematics& K = Cache[Handedness == ESuperDexHandedness::Right ? 1 : 0];
  if (K.Bot != Bot) {
    K = ComputeThumbKinematics(*Bot);
    if (!K.bValid) {
      UE_LOG(
          LogSuperDexOpenXRPinch,
          Warning,
          TEXT(
              "OpenXR thumb is not a spherical pivot and two hinges; thumb positions are not fit"));
    }
  }
  return K;
}

/* Angles in [Min, Max] where Base + C cos(x) + S sin(x) = Target, or else the angle there nearest
 * to it. */
TArray<double, TInlineAllocator<4>>
SolveSinusoid(double Base, double C, double S, double Target, double Min, double Max) {
  TArray<double, TInlineAllocator<4>> Angles;
  const double Amplitude = FMath::Sqrt(C * C + S * S);
  const double Phase = FMath::Atan2(S, C);
  if (Amplitude > UE_DOUBLE_SMALL_NUMBER && FMath::Abs(Target - Base) <= Amplitude) {
    const double Offset = FMath::Acos((Target - Base) / Amplitude);
    for (const double Sign : {-1.0, 1.0}) {
      for (int32 Turn = -2; Turn <= 2; ++Turn) {
        const double X = Phase + Sign * Offset + UE_TWO_PI * Turn;
        if (X >= Min && X <= Max) {
          Angles.Add(X);
        }
      }
    }
  }
  if (Angles.IsEmpty()) {
    // Unreachable: the extremes of the sinusoid in the range, or the range ends.
    TArray<double, TInlineAllocator<8>> Candidates = {Min, Max};
    for (int32 Turn = -3; Turn <= 3; ++Turn) {
      const double X = Phase + UE_PI * Turn;
      if (X > Min && X < Max) {
        Candidates.Add(X);
      }
    }
    auto Miss = [&](double X) {
      return FMath::Abs(Base + C * FMath::Cos(X) + S * FMath::Sin(X) - Target);
    };
    Angles.Add(*Algo::MinElementBy(Candidates, Miss));
  }
  return Angles;
}

/* Solves A x = b for a symmetric positive-definite 5x5 A (Cholesky). */
bool SolveSpd5(double A[5][5], const double B[5], double X[5]) {
  double L[5][5] = {};
  for (int32 I = 0; I < 5; ++I) {
    for (int32 J = 0; J <= I; ++J) {
      double Sum = A[I][J];
      for (int32 K = 0; K < J; ++K) {
        Sum -= L[I][K] * L[J][K];
      }
      if (I == J) {
        if (Sum <= 0.0) {
          return false;
        }
        L[I][I] = FMath::Sqrt(Sum);
      } else {
        L[I][J] = Sum / L[J][J];
      }
    }
  }
  double Y[5];
  for (int32 I = 0; I < 5; ++I) {
    double Sum = B[I];
    for (int32 K = 0; K < I; ++K) {
      Sum -= L[I][K] * Y[K];
    }
    Y[I] = Sum / L[I][I];
  }
  for (int32 I = 4; I >= 0; --I) {
    double Sum = Y[I];
    for (int32 K = I + 1; K < 5; ++K) {
      Sum -= L[K][I] * X[K];
    }
    X[I] = Sum / L[I][I];
  }
  return true;
}

/* The least-squares pad move: moves a finger's pad point toward Goal with kPinchSteps
 * damped least-squares steps over its proximal rotation and intermediate and distal hinges. */
void MoveFingerPad(
    const FThumbKinematics& K,
    const FFingerLinks& Finger,
    const FQuat& ParentToWrist,
    const FVector& Knuckle,
    const FVector& PadInDistal,
    const FVector& Goal,
    TArray<FQuat>& InOutLinkToParent) {
  const int32 Intermediate = Finger.Proximal + 1;
  const FHinge& Pip = K.Hinges[Intermediate];
  const FHinge& Dip = K.Hinges[Finger.Distal];
  FQuat& ProximalRotation = InOutLinkToParent[Finger.Proximal];
  FQuat& PipRotation = InOutLinkToParent[Intermediate];
  FQuat& DipRotation = InOutLinkToParent[Finger.Distal];
  for (int32 Step = 0; Step < kPinchSteps; ++Step) {
    const FQuat Rp = (ParentToWrist * ProximalRotation).GetNormalized();
    const FQuat Ri = (Rp * PipRotation).GetNormalized();
    const FQuat Rd = (Ri * DipRotation).GetNormalized();
    const FVector PipOrigin = Knuckle + Rp.RotateVector(K.Offsets[Intermediate]);
    const FVector DipOrigin = PipOrigin + Ri.RotateVector(K.Offsets[Finger.Distal]);
    const FVector Pad = DipOrigin + Rd.RotateVector(PadInDistal);
    // Columns: pad velocity per knuckle angular velocity (x, y, z), then per hinge angle.
    const FVector Columns[5] = {
        FVector::XAxisVector ^ (Pad - Knuckle),
        FVector::YAxisVector ^ (Pad - Knuckle),
        FVector::ZAxisVector ^ (Pad - Knuckle),
        Ri.RotateVector(Pip.Axis) ^ (Pad - PipOrigin),
        Rd.RotateVector(Dip.Axis) ^ (Pad - DipOrigin)};
    const FVector Error = Goal - Pad;
    double A[5][5];
    double B[5];
    for (int32 I = 0; I < 5; ++I) {
      for (int32 J = 0; J < 5; ++J) {
        A[I][J] = Columns[I] | Columns[J];
      }
      A[I][I] += kPinchDamping * (I < 3 ? 1.0 : kPinchFlexDamping);
      B[I] = Columns[I] | Error;
    }
    double Delta[5];
    if (!SolveSpd5(A, B, Delta)) {
      return;
    }
    const FVector Turn(Delta[0], Delta[1], Delta[2]);
    const double TurnAngle = Turn.Size();
    if (TurnAngle > UE_DOUBLE_SMALL_NUMBER) {
      ProximalRotation =
          (ParentToWrist.Inverse() * FQuat(Turn / TurnAngle, TurnAngle) * Rp).GetNormalized();
    }
    PipRotation = (PipRotation * FQuat(Pip.Axis, Delta[3])).GetNormalized();
    DipRotation = (DipRotation * FQuat(Dip.Axis, Delta[4])).GetNormalized();
  }
  PipRotation = Pip.Rotation(FMath::Clamp(Pip.Angle(PipRotation), Pip.Min, Pip.Max));
  DipRotation = Dip.Rotation(FMath::Clamp(Dip.Angle(DipRotation), Dip.Min, Dip.Max));
}

} // namespace

bool FitOpenXRThumbToDistalFrame(
    ESuperDexHandedness Handedness,
    const FTransform& DistalToWrist,
    TArray<FQuat>& InOutLinkToParent) {
  const FThumbKinematics K = GetThumbKinematics(Handedness);
  if (!K.bValid || InOutLinkToParent.Num() != static_cast<int32>(ESuperDexBone_OXR::Count)) {
    return false;
  }
  const FQuat DistalRotation = DistalToWrist.GetRotation().GetNormalized();
  const FVector ToDistal = DistalToWrist.GetLocation() - K.Pivot;

  // MCP: the pivot-to-distal distance depends on it alone. With the distal origin
  // D(b) = McpOffset + Rest * Rot(b) * IpOffset in the metacarpal frame, |D(b)|^2 is a sinusoid.
  const FVector& McpOffset = K.Mcp.Offset;
  const FVector& IpOffset = K.Ip.Offset;
  const FVector McpOffsetInJoint = K.Mcp.Rest.UnrotateVector(McpOffset);
  const FVector Along = K.Mcp.Axis * (K.Mcp.Axis | IpOffset);
  const double Base =
      McpOffset.SizeSquared() + IpOffset.SizeSquared() + 2.0 * (McpOffsetInJoint | Along);
  const double C = 2.0 * (McpOffsetInJoint | (IpOffset - Along));
  const double S = 2.0 * (McpOffsetInJoint | (K.Mcp.Axis ^ IpOffset));
  const double Hint = K.Mcp.Angle(InOutLinkToParent[kThumbMcp]);
  const auto Candidates = SolveSinusoid(Base, C, S, ToDistal.SizeSquared(), K.Mcp.Min, K.Mcp.Max);
  const double Mcp =
      *Algo::MinElementBy(Candidates, [&](double X) { return FMath::Abs(X - Hint); });
  const FQuat McpRotation = K.Mcp.Rotation(Mcp);
  const FVector Distal = McpOffset + McpRotation.RotateVector(IpOffset);

  // IP: turn the bot's pivot-to-distal direction, seen from the distal frame, onto the target's.
  const FVector Reached = (McpRotation * K.Ip.Rest).UnrotateVector(Distal);
  const FVector Target = DistalRotation.UnrotateVector(ToDistal);
  const FVector ReachedAcross = Reached - K.Ip.Axis * (K.Ip.Axis | Reached);
  const FVector TargetAcross = Target - K.Ip.Axis * (K.Ip.Axis | Target);
  const double Ip = ReachedAcross.IsNearlyZero() || TargetAcross.IsNearlyZero()
      ? K.Ip.Angle(InOutLinkToParent[kThumbIp])
      : FMath::Atan2(K.Ip.Axis | (TargetAcross ^ ReachedAcross), TargetAcross | ReachedAcross);
  const FQuat IpRotation = K.Ip.Rotation(FMath::Clamp(Ip, K.Ip.Min, K.Ip.Max));

  // Metacarpal: whatever orientation gives the distal link the target's exactly.
  InOutLinkToParent[kThumbBase] =
      (DistalRotation * (McpRotation * IpRotation).Inverse()).GetNormalized();
  InOutLinkToParent[kThumbMcp] = McpRotation;
  InOutLinkToParent[kThumbIp] = IpRotation;
  return true;
}

FVector GetOpenXRFingerPadInDistal(ESuperDexFinger Finger, const FVector& TipInDistal) {
  return TipInDistal + FVector(0.0, 0.5 * GetFingerTipRadius_OXR(Finger), 0.0);
}

FVector GetOpenXRFingerEndInDistal(ESuperDexFinger Finger, const FVector& TipInDistal) {
  return TipInDistal - FVector(0.0, 0.0, GetFingerTipRadius_OXR(Finger));
}

bool BlendOpenXRFingersIntoPinch(
    ESuperDexHandedness Handedness,
    const FVector& TrackedThumbDistal,
    TArray<FQuat>& InOutLinkToParent,
    const FOpenXRPinchBlend& Blend) {
  const FThumbKinematics K = GetThumbKinematics(Handedness);
  if (!K.bValid || InOutLinkToParent.Num() != kNumOpenXRBones) {
    return false;
  }
  // Link-to-wrist frames of the pose.
  TArray<FQuat, TInlineAllocator<kNumOpenXRBones>> Rotation;
  TArray<FVector, TInlineAllocator<kNumOpenXRBones>> Location;
  Rotation.SetNum(kNumOpenXRBones);
  Location.SetNum(kNumOpenXRBones);
  Rotation[0] = FQuat::Identity;
  Location[0] = FVector::ZeroVector;
  for (int32 Link = 1; Link < kNumOpenXRBones; ++Link) {
    const int32 Parent = static_cast<int32>(SuperDexBoneParents_OXR[Link]);
    Rotation[Link] = (Rotation[Parent] * InOutLinkToParent[Link]).GetNormalized();
    Location[Link] = Location[Parent] + Rotation[Parent].RotateVector(K.Offsets[Link]);
  }

  // The thumb distal link keeps the tracked orientation, so all of it is offset by this.
  const FVector Shortfall = Location[kThumbIp] - TrackedThumbDistal;
  const FVector ThumbTip = TrackedThumbDistal +
      Rotation[kThumbIp].RotateVector(
          GetOpenXRFingerEndInDistal(ESuperDexFinger::Thumb, K.Offsets[kThumbTip]));
  const double Span = Blend.ZeroWeightDistance - Blend.FullWeightDistance;
  for (const FFingerLinks& Finger : kFingers) {
    // The fingers are still tracked here, so this is the tracked thumb-to-finger distance.
    const FVector Tip = Location[Finger.Distal] +
        Rotation[Finger.Distal].RotateVector(
            GetOpenXRFingerEndInDistal(Finger.Finger, K.Offsets[Finger.Tip]));
    FVector OnThumb, OnFinger;
    FMath::SegmentDistToSegmentSafe(
        TrackedThumbDistal, ThumbTip, Location[Finger.Distal], Tip, OnThumb, OnFinger);
    const double Distance = FVector::Dist(OnThumb, OnFinger);
    const double X = Span > UE_DOUBLE_SMALL_NUMBER
        ? FMath::Clamp((Blend.ZeroWeightDistance - Distance) / Span, 0.0, 1.0)
        : (Distance <= Blend.FullWeightDistance ? 1.0 : 0.0);
    const double Weight = X * X * (3.0 - 2.0 * X);
    if (Weight <= 0.0) {
      continue;
    }
    const FVector PadInDistal = GetOpenXRFingerPadInDistal(Finger.Finger, K.Offsets[Finger.Tip]);
    const FVector Pad = Location[Finger.Distal] + Rotation[Finger.Distal].RotateVector(PadInDistal);
    const FVector& Knuckle = Location[Finger.Proximal];
    const int32 Parent = static_cast<int32>(SuperDexBoneParents_OXR[Finger.Proximal]);
    const FQuat Turn = FQuat::FindBetweenVectors(Pad - Knuckle, Pad + Weight * Shortfall - Knuckle);
    const FQuat TurnedProximal =
        (Rotation[Parent].Inverse() * Turn * Rotation[Finger.Proximal]).GetNormalized();
    // Flex allowance, less where the thumb is nearer the intermediate segment than the distal one.
    const int32 FingerIndex = static_cast<int32>(&Finger - kFingers);
    const int32 Intermediate = Finger.Proximal + 1;
    FVector OnThumbI, OnIntermediate;
    FMath::SegmentDistToSegmentSafe(
        TrackedThumbDistal,
        ThumbTip,
        Location[Intermediate],
        Location[Finger.Distal],
        OnThumbI,
        OnIntermediate);
    const double Lead = FVector::Dist(OnThumbI, OnIntermediate) - Distance;
    const double Y = FMath::Clamp(
        (Lead - kDistalContactStart) / (kDistalContactFull - kDistalContactStart), 0.0, 1.0);
    const double Allowance =
        FMath::Clamp(Blend.FlexAllowance[FingerIndex], 0.0, 1.0) * Y * Y * (3.0 - 2.0 * Y);
    if (Allowance <= 0.0) {
      InOutLinkToParent[Finger.Proximal] = TurnedProximal;
      continue;
    }
    // Interpolate each joint from the knuckle turn toward the least-squares result.
    TArray<FQuat> Moved = InOutLinkToParent;
    MoveFingerPad(
        K, Finger, Rotation[Parent], Knuckle, PadInDistal, Pad + Weight * Shortfall, Moved);
    InOutLinkToParent[Finger.Proximal] =
        FQuat::Slerp(TurnedProximal, Moved[Finger.Proximal], Allowance).GetNormalized();
    InOutLinkToParent[Intermediate] =
        FQuat::Slerp(InOutLinkToParent[Intermediate], Moved[Intermediate], Allowance)
            .GetNormalized();
    InOutLinkToParent[Finger.Distal] =
        FQuat::Slerp(InOutLinkToParent[Finger.Distal], Moved[Finger.Distal], Allowance)
            .GetNormalized();
  }
  return true;
}

bool ShouldApplyOpenXRPinchCorrection(bool bProjectSetting) {
  const int32 Override = CVarPinchCorrectionEnable.GetValueOnAnyThread();
  return Override < 0 ? bProjectSetting : Override != 0;
}

bool ApplyOpenXRPinchCorrection(
    ESuperDexHandedness Handedness,
    const FTransform& TrackedThumbDistalToWrist,
    TArray<FQuat>& InOutLinkToParent) {
  if (!FitOpenXRThumbToDistalFrame(Handedness, TrackedThumbDistalToWrist, InOutLinkToParent)) {
    return false;
  }
  // Where the thumb falls short, move the fingers near it to keep pinches closed.
  BlendOpenXRFingersIntoPinch(
      Handedness, TrackedThumbDistalToWrist.GetLocation(), InOutLinkToParent);
  return true;
}
