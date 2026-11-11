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

#include "Engine/SkeletalMesh.h"
#include "SuperDexPoseUtils.h"
#include "SuperDexTeleopModule.h"
#include "TimerManager.h"

// #pragma optimize("", off)

void UClutchSkeletonRootToVrRootRetargeter::BeginPlay() {
  Super::BeginPlay();

  if (bInitializeClutchFromEndEffectorOfPoseable) {
    AttemptToInitialize_RecursiveOnTick();
  }
}

void UClutchSkeletonRootToVrRootRetargeter::AttemptToInitialize_RecursiveOnTick() {
  // A retry already queued when an explicit initialization succeeds would re-initialize, which
  // clears the enable that initialization was followed by.
  if (bClutchInitialized) {
    return;
  }
  InitializeClutchBasedOnLeafBone(InitializeClutchFromPosableWithName, true);
  if (!bClutchInitialized) {
    if (bInitializeClutchFromEndEffectorOfPoseable) {
      GetWorld()->GetTimerManager().SetTimerForNextTick(
          FTimerDelegate::CreateWeakLambda(
              this, [this]() { AttemptToInitialize_RecursiveOnTick(); }));
    }
  }
}

bool UClutchSkeletonRootToVrRootRetargeter::SetClutchEnabled(bool clutchEnabled) {
  FWriteScopeLock lock(ClutchLock);
  // only allowed to do things if initialized
  if (bClutchInitialized) {
    if (bClutchEnabled != clutchEnabled) {
      if (clutchEnabled == true) {
        // if the clutch just went from disabled to enabled then flag that we should collect a new
        // relative transform
        bUpdateClutchRelativeTransform = true;
      }
      bClutchEnabled = clutchEnabled;
      OnClutchStateChanged.Broadcast(this, clutchEnabled);
    }
  }
  return bClutchEnabled;
}

FTransform LerpTransform(const FTransform& A, const FTransform& B, float Alpha) {
  return FTransform(
      FMath::Lerp(A.GetRotation(), B.GetRotation(), Alpha),
      FMath::Lerp(A.GetTranslation(), B.GetTranslation(), Alpha),
      FMath::Lerp(A.GetScale3D(), B.GetScale3D(), Alpha));
}

bool UClutchSkeletonRootToVrRootRetargeter::GetTransformedPose_Implementation(
    FSuperDexPose const& SourcePose,
    bool bSourceIsValid,
    FSuperDexPose& TransformedPose) {
  if (bSourceIsValid == false) {
    return false;
  }

  FWriteScopeLock lock(ClutchLock);
  if (bClutchInitialized == false) {
    return false;
  }

  if (bLastEnabledSourcePoseInitialized == false) {
    LastEnabledSourcePose = SourcePose;
    bLastEnabledSourcePoseInitialized = true;
  }
  // ClutchRootToSkeletonRoot_location =
  //     ClutchedSkeletonRootToVrRoot * SourcePose.SkeletonRootToVrRoot.Inverse();

  //  A*B
  //	Q(AxB) = Q(B)*Q(A)
  //	S(AxB) = S(A)*S(B)
  //	T(AxB) = Q(B)*S(B)*T(A)*-Q(B) + T(B)
  if (bClutchEnabled) {
    if (bUpdateClutchRelativeTransform) {
      InitialClutchRootToSkeletonRoot_location = FTransform(
          ClutchedSkeletonRootToVrRoot.GetLocation() -
          SourcePose.SkeletonRootToVrRoot.GetLocation());
      InitialClutchVrRootToVrRoot_rotation = FTransform(
          ClutchedSkeletonRootToVrRoot.GetRotation().Inverse() *
          SourcePose.SkeletonRootToVrRoot.GetRotation());

      // Reset our lerp start time
      LastClutchEnableTime = GetWorld()->GetTimeSeconds();

      // lower the flag now that we have updated the clutch position
      bUpdateClutchRelativeTransform = false;
    }

    double currentClutchTime = GetWorld()->GetTimeSeconds();
    double deltaTime = currentClutchTime - LastClutchEnableTime;
    double lerpFraction = 1.0;
    if ((deltaTime < TimeInSecondsToLerpToAbsolute) && (TimeInSecondsToLerpToAbsolute > 0.001)) {
      lerpFraction = FMath::Clamp(deltaTime / TimeInSecondsToLerpToAbsolute, 0.0, 1.0);
    }
    FTransform ClutchRootToSkeletonRoot_location = InitialClutchRootToSkeletonRoot_location;
    FTransform ClutchVrRootToVrRoot_rotation = InitialClutchVrRootToVrRoot_rotation;
    // Identity out the transforms that are meant to be absolute
    switch (ClutchType) {
      case ESuperDexClutchType::RelativeTranslationAbsoluteRotation:
        ClutchVrRootToVrRoot_rotation =
            LerpTransform(InitialClutchVrRootToVrRoot_rotation, FTransform::Identity, lerpFraction);
        break;

      case ESuperDexClutchType::RelativeTranslationRelativeRotation:
        break;

      case ESuperDexClutchType::AbsoluteTranslationRelativeRotation:
        ClutchRootToSkeletonRoot_location = LerpTransform(
            InitialClutchRootToSkeletonRoot_location, FTransform::Identity, lerpFraction);
        break;

      case ESuperDexClutchType::AbsoluteTranslationAbsoluteRotation:
        ClutchRootToSkeletonRoot_location = LerpTransform(
            InitialClutchRootToSkeletonRoot_location, FTransform::Identity, lerpFraction);
        ClutchVrRootToVrRoot_rotation =
            LerpTransform(InitialClutchVrRootToVrRoot_rotation, FTransform::Identity, lerpFraction);
        break;
    }

    static_assert(static_cast<int>(ESuperDexClutchType::Count) == 4);
    // compute the general transform, if we expand ESuperDexClutchType we may want to add a switch
    // here too
    FTransform ClutchRootToVrRoot = FTransform(
        SourcePose.SkeletonRootToVrRoot.GetRotation() *
            ClutchVrRootToVrRoot_rotation.GetRotation().Inverse(),
        SourcePose.SkeletonRootToVrRoot.GetLocation() +
            ClutchRootToSkeletonRoot_location.GetLocation(),
        SourcePose.SkeletonRootToVrRoot.GetScale3D());

    // Check our safety checks to disengage the clutch prior to setting it.
    bool clutchEnabled = bClutchEnabled;
    FVector deltaPos =
        ClutchRootToVrRoot.GetLocation() - ClutchedSkeletonRootToVrRoot.GetLocation();
    clutchEnabled &= (deltaPos.Length() <= DisengageClutchIfDeltaDistanceOverAmount_cm);
    clutchEnabled &=
        (SourcePose.TrackingConfidence >= DisengageClutchIfSourceConfidenceBelowAmount);
    clutchEnabled &= (SourcePose.bIsTrackingConfident || !bDisengageClutchIfSourceIsNotConfident);

    // check if the clutch changed state due to safety checks
    if (clutchEnabled != bClutchEnabled) {
      if (bPauseClutchInsteadOfAutoDisengaging == false) {
        bClutchEnabled = clutchEnabled;
        OnClutchStateChanged.Broadcast(this, bClutchEnabled);
      } else {
        bClutchWasPaused = true;
        OnClutchPauseUpause.Broadcast(this, bClutchWasPaused);
      }
    }

    // only update clutch transform if we have not hit our safety checks
    if (clutchEnabled) {
      LastEnabledSourcePose = SourcePose;

      // If the clutch was paused and is now enabled, we need to capture the relative position
      // between the clutch transforms so we can lerp instead of teleport
      if (bClutchWasPaused) {
        UnpausedClutchedSkeletonRootToClutchRoot =
            ClutchedSkeletonRootToVrRoot * ClutchRootToVrRoot.Inverse();
        LastClutchUnpauseTime = GetWorld()->GetTimeSeconds();
        bClutchWasPaused = false;
        OnClutchPauseUpause.Broadcast(this, bClutchWasPaused);
      }

      double currentUnpauseTime = GetWorld()->GetTimeSeconds();
      double unpauseDeltaTime = currentUnpauseTime - LastClutchUnpauseTime;
      double unpauseLerpFraction = 1.0;
      if ((unpauseDeltaTime < TimeInSecondsToLerpOnUnpause) &&
          (TimeInSecondsToLerpOnUnpause > 0.001)) {
        unpauseLerpFraction =
            FMath::Clamp(unpauseDeltaTime / TimeInSecondsToLerpOnUnpause, 0.0, 1.0);
      }

      // Set the clutch transform to ClutchRootToVrRoot plus any residual from unpausing
      ClutchedSkeletonRootToVrRoot =
          LerpTransform(
              UnpausedClutchedSkeletonRootToClutchRoot, FTransform::Identity, unpauseLerpFraction) *
          ClutchRootToVrRoot;
    }
  }
  TransformedPose = LastEnabledSourcePose;
  TransformedPose.tags.Add(TEXT("ClutchedPose"));
  // Does not change the SkeletonName
  TransformedPose.SkeletonRootToVrRoot = ClutchedSkeletonRootToVrRoot;
  // Since the SourcePose is guaranteed to be valid the transformed pose will also be valid
  return true;
}

bool UClutchSkeletonRootToVrRootRetargeter::InitializeClutchBasedOnLeafBone(
    FString const& PosableWithName,
    bool UpdateClutchToLeafBoneTransform) {
  return InitializeClutchBasedOnLeafBoneInternal(
      PosableWithName, UpdateClutchToLeafBoneTransform, /*bRequireCachedPose=*/true);
}

bool UClutchSkeletonRootToVrRootRetargeter::InitializeClutchFromCurrentPoseableLeafBone(
    FString const& PosableWithName,
    bool UpdateClutchToLeafBoneTransform) {
  return InitializeClutchBasedOnLeafBoneInternal(
      PosableWithName, UpdateClutchToLeafBoneTransform, /*bRequireCachedPose=*/false);
}

bool UClutchSkeletonRootToVrRootRetargeter::InitializeClutchBasedOnLeafBoneInternal(
    FString const& PosableWithName,
    bool UpdateClutchToLeafBoneTransform,
    bool bRequireCachedPose) {
  bool bWasEnabled = false;
  {
    FWriteScopeLock lock(ClutchLock);
    bWasEnabled = bClutchEnabled;
    bClutchInitialized = false;
    bClutchEnabled = false;
  }
  if (bWasEnabled) {
    OnClutchStateChanged.Broadcast(this, false);
  }

  if (PosableWithName.IsEmpty()) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "ClutchPoseTransformer [%s] was passed an empty string for leaf initialization. Stopping intialization."),
        *GetName());
    bInitializeClutchFromEndEffectorOfPoseable = false;
    return false;
  }
  USuperDexPoseableComponent* RobotArmPosablePtr =
      FindComponentByName<USuperDexPoseableComponent>(GetOwner(), PosableWithName);
  if (RobotArmPosablePtr == nullptr) {
    // Stop trying to initialize if the component literally does not exist
    // Going to assume that we are not dynamically constructing our robots during runtime
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "ClutchPoseTransformer [%s] could not find component with name [%s] to initialize from. Stopping intialization."),
        *GetName(),
        *PosableWithName);
    bInitializeClutchFromEndEffectorOfPoseable = false;
    return false;
  }
  // TODO: need to make sure we have a valid pose. The cache is not a guaranteed way of doing that
  FSuperDexPose ArmCachedPose;
  const bool bHasCache = RobotArmPosablePtr->GetCachedPose(ArmCachedPose);
  if (bRequireCachedPose && !bHasCache) {
    return false;
  }

  auto const* const PoseableAsset = RobotArmPosablePtr->GetSkinnedAsset();
  if (PoseableAsset == nullptr) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "ClutchPoseTransformer [%s] identified component with name [%s] as having no skeletal mesh. Stopping intialization."),
        *GetName(),
        *PosableWithName);
    bInitializeClutchFromEndEffectorOfPoseable = false;
    return false;
  }
  FReferenceSkeleton const& skele = PoseableAsset->GetRefSkeleton();
  int numBones = skele.GetNum();
  if (numBones == 0) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "ClutchPoseTransformer [%s] identified component with name [%s] as having 0 bones in its skeleton. Stopping intialization."),
        *GetName(),
        *PosableWithName);
    bInitializeClutchFromEndEffectorOfPoseable = false;
    return false;
  }
  TSet<int> BoneParents;
  for (int iBone = 0; iBone < numBones; iBone++) {
    BoneParents.Add(skele.GetParentIndex(iBone));
  }
  TArray<int> LeafBones;
  for (int iBone = 0; iBone < numBones; iBone++) {
    if (!BoneParents.Contains(iBone)) {
      LeafBones.Add(iBone);
    }
  }
  if ((LeafBones.Num() != 1) && (InitializeClutchRelativeToBoneIndex < 0)) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "ClutchPoseTransformer [%s] identified component with name [%s] as having more than one [%d] leaf bones. Cannot determine end effector pose. Stopping intialization."),
        *GetName(),
        *PosableWithName,
        LeafBones.Num());
    bInitializeClutchFromEndEffectorOfPoseable = false;
    return false;
  }

  const int EndEffectorBone = (InitializeClutchRelativeToBoneIndex >= 0)
      ? InitializeClutchRelativeToBoneIndex
      : LeafBones[0];
  FTransform EndEffectorToWorld = RobotArmPosablePtr->GetBoneTransform(EndEffectorBone);
  FTransform VrRootToWorld = GetOwner()->GetActorTransform();
  if (UpdateClutchToLeafBoneTransform &&
      !bOverrideInitializeClutchSkeletonRootToPoseableEndEffector) {
    FTransform ClutchToWorld = GetComponentTransform();
    InitializeClutchSkeletonRootToPoseableEndEffector =
        ClutchToWorld * EndEffectorToWorld.Inverse();
  }
  FTransform ClutchToRoot = InitializeClutchSkeletonRootToPoseableEndEffector * EndEffectorToWorld *
      VrRootToWorld.Inverse();
  SetClutchedSkeletonRootToVrRoot(ClutchToRoot);
  UE_LOG(
      LogSuperDexTeleop,
      Log,
      TEXT(
          "ClutchPoseTransformer [%s] succeeded in initialization from leaf bone [%s][%d] and UpdateClutchToLeafBoneTransform[%d]."),
      *GetName(),
      *PosableWithName,
      EndEffectorBone,
      UpdateClutchToLeafBoneTransform);
  return true;
}
