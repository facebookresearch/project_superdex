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

#include "Skeletons/SuperDexSkeleton_OpenXR.h"

#include "DrawDebugHelpers.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "HeadMountedDisplayTypes.h"
#include "IXRTrackingSystem.h"
#include "Interfaces/IPluginManager.h"

namespace {

TAutoConsoleVariable<bool> CVarOpenXRPoseSourceDebugDraw(
    TEXT("superdex.OpenXRPoseSource.DebugDraw"),
    false,
    TEXT("Draw the bone frame of every tracked OpenXR hand joint (X red, Y green, Z blue)."));

/* Nominal length [m] of each finger phalanx and tip, keyed by the child bone: the lengths of the
 * bone offsets in the openxr bots' rest skeleton. That skeleton is the Meta runtime's XRHand bind
 * skeleton, the unit hand it scales to the user's hand (ovrp_GetSkeleton3; see OVRSkeletonData.cs
 * in the Meta XR Interaction SDK).
 *
 * Core OpenXR (XR_EXT_hand_tracking) reports neither this skeleton nor the hand scale. Meta's
 * XR_FB_hand_tracking_mesh reports both (xrGetHandMeshFB bind poses, XrHandTrackingScaleFB), but
 * UE's OpenXR plugin does not use it. */
struct FNominalPhalanx {
  ESuperDexBone_OXR Bone;
  double Length;
};
constexpr FNominalPhalanx kNominalPhalanges[] = {
    {ESuperDexBone_OXR::ThumbProximal, 0.0325129},
    {ESuperDexBone_OXR::ThumbDistal, 0.0337931},
    {ESuperDexBone_OXR::ThumbTip, 0.0246213},
    {ESuperDexBone_OXR::IndexIntermediate, 0.0379273},
    {ESuperDexBone_OXR::IndexDistal, 0.0243037},
    {ESuperDexBone_OXR::IndexTip, 0.0223888},
    {ESuperDexBone_OXR::MiddleIntermediate, 0.0429270},
    {ESuperDexBone_OXR::MiddleDistal, 0.0275496},
    {ESuperDexBone_OXR::MiddleTip, 0.0249927},
    {ESuperDexBone_OXR::RingIntermediate, 0.0389961},
    {ESuperDexBone_OXR::RingDistal, 0.0265734},
    {ESuperDexBone_OXR::RingTip, 0.0243806},
    {ESuperDexBone_OXR::LittleIntermediate, 0.0307204},
    {ESuperDexBone_OXR::LittleDistal, 0.0203114},
    {ESuperDexBone_OXR::LittleTip, 0.0219575},
};

// Bots are shown in Unreal at 1 cm per Unreal unit (kUnrealCentimetersSpace).
constexpr double kBotUnitsPerMeter = 100.0;

/* C: maps an OpenXR joint frame, as UE's OpenXR plugin reports it, to the imported bone frame. */
const FQuat& GetJointToBoneRotation() {
  // C = B * S maps local X -> +Y, Y -> -Z, Z -> -X (FMatrix rows are the images of the axes).
  static const FQuat C = FQuat(FMatrix(
      FPlane(0.0, 1.0, 0.0, 0.0),
      FPlane(0.0, 0.0, -1.0, 0.0),
      FPlane(-1.0, 0.0, 0.0, 0.0),
      FPlane(0.0, 0.0, 0.0, 1.0)));
  return C;
}

} // namespace

double USuperDexOpenXRPoseSource::EstimateHandScale(TConstArrayView<FVector> KeypointLocations) {
  check(KeypointLocations.Num() == EHandKeypointCount);
  double Tracked = 0.0;
  double Nominal = 0.0;
  for (const FNominalPhalanx& Phalanx : kNominalPhalanges) {
    const ESuperDexBone_OXR Parent = kSuperDexBoneParents_OXR[static_cast<int32>(Phalanx.Bone)];
    Tracked += FVector::Dist(
        KeypointLocations[GetXrHandJointOfBone_OXR(Phalanx.Bone)],
        KeypointLocations[GetXrHandJointOfBone_OXR(Parent)]);
    Nominal += Phalanx.Length;
  }
  const double Scale = Tracked / (Nominal * kBotUnitsPerMeter);
  return FMath::IsFinite(Scale) && Scale > UE_KINDA_SMALL_NUMBER ? Scale : 1.0;
}

bool USuperDexOpenXRPoseSource::ShouldCreatePoseSource(UWorld* World) const {
  // Hand tracking comes from Epic's OpenXR plugin, with XR_EXT_hand_tracking from
  // OpenXRHandTracking.
  IPluginManager& PluginManager = IPluginManager::Get();
  return PluginManager.FindEnabledPlugin(TEXT("OpenXR")).IsValid() &&
      PluginManager.FindEnabledPlugin(TEXT("OpenXRHandTracking")).IsValid();
}

bool USuperDexOpenXRPoseSource::GetPose(
    const ESuperDexHandedness Handedness,
    FSuperDexPose& OutPose) {
  check(Handedness == ESuperDexHandedness::Left || Handedness == ESuperDexHandedness::Right);
  FScopeLock Lock(&StateLock);
  OutPose = (Handedness == ESuperDexHandedness::Left) ? LeftPoseCache : RightPoseCache;
  return OutPose.bIsTrackingEnabled;
}

void USuperDexOpenXRPoseSource::ConvertHandKeypointsToPose(
    TConstArrayView<FQuat> KeypointRotations,
    const FVector& WristLocation,
    FSuperDexPose& OutPose) {
  check(KeypointRotations.Num() == EHandKeypointCount);
  const FQuat& C = GetJointToBoneRotation();
  constexpr int32 NumBones = static_cast<int32>(ESuperDexBone_OXR::Count);

  TArray<FQuat, TInlineAllocator<NumBones>> BoneToTracking;
  BoneToTracking.SetNum(NumBones);
  for (int32 Bone = 0; Bone < NumBones; ++Bone) {
    const FQuat& Joint =
        KeypointRotations[GetXrHandJointOfBone_OXR(static_cast<ESuperDexBone_OXR>(Bone))];
    BoneToTracking[Bone] = (Joint.GetNormalized() * C).GetNormalized();
  }

  OutPose.SkeletonName = kSkeletonName_OpenXR;
  OutPose.BoneLocations.Reset();
  OutPose.BoneRotations.SetNum(NumBones);
  OutPose.BoneRotations[0] = FQuat::Identity;
  for (int32 Bone = 1; Bone < NumBones; ++Bone) {
    const int32 Parent = static_cast<int32>(kSuperDexBoneParents_OXR[Bone]);
    OutPose.BoneRotations[Bone] =
        (BoneToTracking[Parent].Inverse() * BoneToTracking[Bone]).GetNormalized();
  }
  OutPose.SkeletonRootToVrRoot = FTransform(BoneToTracking[0], WristLocation, FVector::OneVector);
}

void USuperDexOpenXRPoseSource::CachePose(const ESuperDexHandedness Handedness) {
  FSuperDexPose& Cache = (Handedness == ESuperDexHandedness::Left) ? LeftPoseCache : RightPoseCache;
  Cache = FSuperDexPose();
  Cache.SourceName = TEXT("OpenXR");
  Cache.SkeletonName = kSkeletonName_OpenXR;
  Cache.tags.Add(TEXT("OpenXR"));
  Cache.tags.Add((Handedness == ESuperDexHandedness::Left) ? TEXT("Left") : TEXT("Right"));

  if (GEngine == nullptr || !GEngine->XRSystem.IsValid()) {
    return;
  }
  IXRTrackingSystem& XRSystem = *GEngine->XRSystem;
  FXRHandTrackingState State;
  // UE's OpenXR hand tracker reports keypoints in Unreal world space whatever space is requested,
  // so ask for world space and bring them back to tracking (VrRoot) space here.
  XRSystem.GetHandTrackingState(
      GetWorld(),
      EXRSpaceType::UnrealWorldSpace,
      Handedness == ESuperDexHandedness::Left ? EControllerHand::Left : EControllerHand::Right,
      State);
  if (!State.bValid || State.HandKeyRotations.Num() != EHandKeypointCount ||
      State.HandKeyLocations.Num() != EHandKeypointCount) {
    return;
  }

  const FTransform TrackingToWorld = XRSystem.GetTrackingToWorldTransform();
  const FTransform WorldToTracking = TrackingToWorld.Inverse();
  TArray<FQuat, TInlineAllocator<EHandKeypointCount>> Rotations;
  Rotations.SetNum(EHandKeypointCount);
  TArray<FVector, TInlineAllocator<EHandKeypointCount>> Locations;
  Locations.SetNum(EHandKeypointCount);
  for (int32 i = 0; i < EHandKeypointCount; ++i) {
    Rotations[i] = (FTransform(State.HandKeyRotations[i]) * WorldToTracking).GetRotation();
    Locations[i] = WorldToTracking.TransformPosition(State.HandKeyLocations[i]);
  }
  ConvertHandKeypointsToPose(Rotations, Locations[(int32)EHandKeypoint::Wrist], Cache);
  Cache.SkeletonRootToVrRoot.SetScale3D(FVector(EstimateHandScale(Locations)));

  Cache.bIsTrackingEnabled = true;
  Cache.bIsTrackingConfident = State.TrackingStatus == ETrackingStatus::Tracked;
  Cache.TrackingConfidence = Cache.bIsTrackingConfident ? 1.f : .5f;

  if (CVarOpenXRPoseSourceDebugDraw.GetValueOnGameThread() && GetWorld() != nullptr) {
    const FQuat& C = GetJointToBoneRotation();
    for (int32 i = 0; i < EHandKeypointCount; ++i) {
      DrawDebugCoordinateSystem(
          GetWorld(),
          State.HandKeyLocations[i],
          (State.HandKeyRotations[i] * C).Rotator(),
          1.5f,
          false,
          -1.f,
          SDPG_Foreground,
          0.1f);
    }
  }
}

UWorld* USuperDexOpenXRPoseSource::GetTickableGameObjectWorld() const {
  return GetWorld();
}

ETickableTickType USuperDexOpenXRPoseSource::GetTickableTickType() const {
  return IsTemplate() ? ETickableTickType::Never : FTickableGameObject::GetTickableTickType();
}

bool USuperDexOpenXRPoseSource::IsTickable() const {
  return !IsTemplate();
}

bool USuperDexOpenXRPoseSource::IsTickableWhenPaused() const {
  return false;
}

bool USuperDexOpenXRPoseSource::IsTickableInEditor() const {
  return false;
}

void USuperDexOpenXRPoseSource::Tick(float DeltaTime) {
  if (NecessaryThreadId == kInvalidThreadId) {
    NecessaryThreadId = FPlatformTLS::GetCurrentThreadId();
  }
  FScopeLock Lock(&StateLock);
  CachePose(ESuperDexHandedness::Left);
  CachePose(ESuperDexHandedness::Right);
}

TStatId USuperDexOpenXRPoseSource::GetStatId() const {
  RETURN_QUICK_DECLARE_CYCLE_STAT(USuperDexOpenXRPoseSource, STATGROUP_Tickables);
}
