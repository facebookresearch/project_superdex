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

#pragma once

#include "Retargeters/SuperDexRetargeter.h"

#include "ClutchSkeletonRootToVrRootRetargeter.generated.h"

UENUM(BlueprintType)
enum class ESuperDexClutchType : uint8 {
  // The relative translation of the input to output SkeletonRoot persists while clutched. The
  // rotation of the output skeletonRoot equals the input rotation while clutched.
  RelativeTranslationAbsoluteRotation,
  // The relative translation and relative rotation are both applied when clutched. The rotation
  // will be applied after the translation so that rotating the input SkeletonRoot does not
  // translate the output SkeletonRoot.
  RelativeTranslationRelativeRotation,
  // The translation of the output skeletonRoot equals the input translation while clutched. The
  // relative rotation of the input to output SkeletonRoot persists while clutched.
  AbsoluteTranslationRelativeRotation,
  // The translation and rotation of the output skeletonRoot equals the input translation and
  // rotation while clutched
  AbsoluteTranslationAbsoluteRotation,

  Count UMETA(Hidden),
};

UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class SUPERDEXTELEOP_API UClutchSkeletonRootToVrRootRetargeter : public USuperDexRetargeter {
  GENERATED_BODY()
 public:
  // Determines what mechanics that clutch uses. This should not be changed while running as the
  // system is not currently written to support that without additional initialization.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex")
  ESuperDexClutchType ClutchType = ESuperDexClutchType::RelativeTranslationAbsoluteRotation;

  /**
   * If the ClutchType involves an absolute translation or rotation the initial clutch when engaged
   * will report a relative relationship and then lerp to the absolute relationship over the course
   * of this much time in seconds. For example, if you have an absolute rotation ClutchType and have
   * engaged the clutch with a 90 degree rotation between your input and current output. If you hold
   * still, the output rotation will slowly lerp from its initial position toward your current
   * position over the specified time.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  double TimeInSecondsToLerpToAbsolute = 1.0;

  /**
   * If true, the clutch will not disengage when it meets the "disengage clutch if" criteria.
   * Instead it will ignore the input leaving itself unchanged and then resume updating when the
   * data becomes valid again. This will not broadcast a clutch disabled event. If false, the clutch
   * will disengage when the criteria is met.
   * NOTE: The clutch pose will instantaneously jump to the new position once the pause is over if
   * the hand is within the DisengageClutchIfDeltaDistanceOverAmount_cm. If the hand was moving when
   * the clutch paused due to loss of tracking it is likely to resume near where the hand was. If it
   * pauses due to excessive velocity then the jump on resume will be maximized. Disable and
   * reenabling clutch will still properly reset as expected.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  bool bPauseClutchInsteadOfAutoDisengaging = false;

  /**
   * If bPauseClutchInsteadOfAutoDisengaging is true, then when the clutch unpauses it will lerp its
   * current position with the new clutch position over this amount of time. Effectively bleeding
   * off the delta transform over time rather than instantly teleporting to the new position. This
   * helps smooth out any bumps or jitters that may be caused by intermittent or flaky hand
   * tracking.
   * NOTE: The pose must be valid to unpause so it must be both confident and within
   * DisengageClutchIfDeltaDistanceOverAmount_cm of the original paused location.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  double TimeInSecondsToLerpOnUnpause = 0.25;

  /**
   * A safety catch will automatically disengage the clutch if the distance between clutched
   * transforms is greater than this amount. This will generally detect moving too quickly or hands
   * teleporting due to tracking failures.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  double DisengageClutchIfDeltaDistanceOverAmount_cm = 15.0;

  /**
   * A safety catch will automatically disengage the clutch if SourcePose.TrackingConfidence is
   * below this amount. Tracking sources report their confidence.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  double DisengageClutchIfSourceConfidenceBelowAmount = 0.6;

  /**
   * If true, enables a safety catch that will automatically disengage the clutch if
   * SourcePose.bIsTrackingConfident is false. Tracking sources have their own internal threshold
   * for determining this value and should set it appropriately.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  bool bDisengageClutchIfSourceIsNotConfident = true;

  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex")
  bool bInitializeClutchFromEndEffectorOfPoseable = false;

  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex")
  FString InitializeClutchFromPosableWithName = "";

  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex")
  int InitializeClutchRelativeToBoneIndex = -1;

  // If true, InitializeClutchSkeletonRootToPoseableEndEffector will be set to its value. If false,
  // it will be derived from the relative position from the clutch component to the leaf bone of the
  // SuperDexPoseableComponent found using InitializeClutchFromPosableWithName on BeginPlay. If
  // false and no valid component is present, it will be identity.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  bool bOverrideInitializeClutchSkeletonRootToPoseableEndEffector = false;

  // This should be the same transform used by an arm to transform pose. Its inverse will be applied
  // to the end effector transform of the posable to compute the initial clutch
  // SkeletonRootToVrRoot. This value is used internally to pose the clutch with respect to the end
  // effector.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta = (EditCondition = "bOverrideInitializeClutchSkeletonRootToPoseableEndEffector"),
      Category = "SuperDex")
  FTransform InitializeClutchSkeletonRootToPoseableEndEffector = FTransform::Identity;

  /**
   * Finds the component on the parent actor with the name provided in
   * InitializeClutchFromPosableWithName. Then checks if the actor is of type
   * USuperDexPoseableComponent and has a skeletal mesh attached that has only one leaf node. It
   * then calls SetClutchedSkeletonRootToVrRoot() passing in the transform of that leaf node
   * relative to the VrRoot. If UpdateClutchToLeafBoneTransform is true, then the system will update
   * InitializeClutchSkeletonRootToPoseableEndEffector with the correct relative pose with respect
   * to the leaf node. In general this should only really be set to true when first initializing
   * the clutch from the spawn pose. If false, it will use the existing relative pose, usually
   * assigned when it was first initialized. If any stage has issues the function will abort and
   * return false.
   */
  UFUNCTION(BlueprintCallable, meta = (ReturnDisplayName = "Success"), Category = "SuperDex")
  bool InitializeClutchBasedOnLeafBone(
      FString const& PosableWithName,
      bool UpdateClutchToLeafBoneTransform = false);

  bool InitializeClutchFromCurrentPoseableLeafBone(
      FString const& PosableWithName,
      bool UpdateClutchToLeafBoneTransform = false);

  /**
   * Set the clutch state. The state is only set if the clutch transform has been initialized.
   * Returns the new state of the clutch (which may not be equal to the command and false if not
   * initialized). When the clutch is enabled from a disabled state the relative transform between
   * the clutch transform and SourcePose transform is recorded to compute the new TransformedPose.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex")
  bool SetClutchEnabled(bool clutchEnabled);

  UFUNCTION(BlueprintPure, Category = "SuperDex")
  bool GetClutchEnabled() const {
    FReadScopeLock lock(ClutchLock);
    return bClutchEnabled;
  }

  UFUNCTION(BlueprintPure, Category = "SuperDex")
  bool IsClutchInitialized() const {
    FReadScopeLock lock(ClutchLock);
    return bClutchInitialized;
  }

  DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(
      FOnClutchChanged,
      UClutchSkeletonRootToVrRootRetargeter*,
      ClutchRetargeter,
      bool,
      NewClutchState);

  DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(
      FOnClutchPaused,
      UClutchSkeletonRootToVrRootRetargeter*,
      ClutchRetargeter,
      bool,
      IsNowPaused);

  // Emitted when the clutch enable state changes. Useful for knowing when the clutch has disengaged
  // itself via the safety checks and informing the user.
  UPROPERTY(BlueprintAssignable, Category = "SuperDex")
  FOnClutchChanged OnClutchStateChanged;

  // Emitted when the clutch pauses or unpauses. Useful for knowing when the clutch has disengaged
  // itself via the safety checks and informing the user.
  UPROPERTY(BlueprintAssignable, Category = "SuperDex")
  FOnClutchPaused OnClutchPauseUpause;

  /**
   * Get the currently stored and applied SkeletonRootToVrRoot. This value will be updated
   * continuously while the clutch is enabled.
   */
  UFUNCTION(BlueprintPure, Category = "SuperDex")
  FTransform GetClutchedSkeletonRootToVrRoot() const {
    FReadScopeLock lock(ClutchLock);
    return ClutchedSkeletonRootToVrRoot;
  }

  /**
   * Set the clutched SkeletonRootToVrRoot to a new value. This is used to initialize the clutch
   * start position. You may want to set this to a specific default value or based on the end
   * effector position of some other component such as a robot arm. When the clutch is disengaged
   * the TransformedPose will return the SourcePose with the SkeletonRootToVrRoot replaced by the
   * clutched version.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex")
  void SetClutchedSkeletonRootToVrRoot(FTransform const& newSkeletonRootToVrRoot) {
    FWriteScopeLock lock(ClutchLock);
    ClutchedSkeletonRootToVrRoot = newSkeletonRootToVrRoot;
    bClutchInitialized = true;
  }

  virtual bool GetTransformedPose_Implementation(
      FSuperDexPose const& SourcePose,
      bool SourceIsValid,
      FSuperDexPose& TransformedPose) override;

 protected:
  virtual void BeginPlay() override;

  void AttemptToInitialize_RecursiveOnTick();

  bool InitializeClutchBasedOnLeafBoneInternal(
      FString const& PosableWithName,
      bool UpdateClutchToLeafBoneTransform,
      bool bRequireCachedPose);

  // Guards access to the our clutch in case it is being evaluated on the mochi thread.
  // Explicity should guard
  mutable FRWLock ClutchLock;

  // Do not allow the user to clutch the robot until an initial value has been set.
  bool bClutchInitialized = false;
  bool bClutchEnabled = false;
  bool bClutchWasPaused = false;

  // The relative transforms are collected in GetTransformedPose() so that it is accurate to the
  // first call after the clutch was enabled. Resulting in zero unintended movement. The relative
  // transform is applied to the SourcePose differently depending on the settings of the clutch
  // action.
  bool bUpdateClutchRelativeTransform = false;
  FTransform InitialClutchRootToSkeletonRoot_location = FTransform::Identity;
  FTransform InitialClutchVrRootToVrRoot_rotation = FTransform::Identity;
  double LastClutchEnableTime = -1;
  double LastClutchUnpauseTime = -1;

  // The actual computed transform being applied to the SourcePose to get the TransformedPose
  FTransform ClutchedSkeletonRootToVrRoot = FTransform::Identity;
  FTransform UnpausedClutchedSkeletonRootToClutchRoot = FTransform::Identity;

  // Source pose used for clutching so that bones stay stationary when clutch is disabled
  FSuperDexPose LastEnabledSourcePose;
  bool bLastEnabledSourcePoseInitialized = false;
};
