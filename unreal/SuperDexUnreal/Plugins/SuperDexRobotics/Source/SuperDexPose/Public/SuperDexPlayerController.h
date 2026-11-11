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

#include "CoreMinimal.h"
#include "Delegates/Delegate.h"
#include "GameFramework/PlayerController.h"
#include "SuperDexPoseSource.h"
#include "SuperDexPoseTypes.h"

#include "SuperDexPlayerController.generated.h"

class ASuperDexPlayerPawn;
class ASuperDexPlayerStart;
class USuperDexPoseSource;

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FTriggerAlignment);

/** Delegate for pose source changed events with handedness information. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(
    FPoseSourceChangedWithHandedness,
    TSubclassOf<USuperDexPoseSource>,
    PoseSource,
    ESuperDexHandedness,
    Handedness);

/**
 * Player controller to be used with SuperDexPose.
 *
 * Manages hand pose source selection for the player. Gets pose from the pose subsystem and passes
 * it to its possessed pawn.
 */
UCLASS()
class SUPERDEXPOSE_API ASuperDexPlayerController : public APlayerController {
  GENERATED_BODY()

 public:
  /** Delegate for pose source changed events. */
  UPROPERTY(BlueprintAssignable)
  FPoseSourceChangedWithHandedness OnPoseSourceChanged;

  /** Called when the alignment keybind is pressed */
  UPROPERTY(BlueprintAssignable)
  FTriggerAlignment OnDoAlginmentKeypress;

  /** SuperDexPlayerPawn possessed by this controller. */
  UPROPERTY(BlueprintReadOnly, Category = "SuperDexPlayerController")
  ASuperDexPlayerPawn* PlayerPawn;

 private:
  /** First SuperDexPlayerStart in the level. */
  ASuperDexPlayerStart* PlayerStart = nullptr;

  /** Current index within the SuperDexPlayerStart's left pose sources array. */
  int LeftPoseSourceIndex = 0;

  /** Current index within the SuperDexPlayerStart's right pose sources array. */
  int RightPoseSourceIndex = 0;

 public:
  /**
   * Get pose, using currently selected pose source (from the SuperDexPlayerStart's pose
   * sources configuration), for given handeness.
   *
   * @param Handedness The handedness to get pose for.
   * @param OutPose The output pose.
   * @return true if successful, false otherwise.
   */
  UFUNCTION(
      BlueprintCallable,
      Category = "ASuperDexPlayerController",
      meta = (ReturnDisplayName = "Success"))
  bool GetPose(const ESuperDexHandedness Handedness, FSuperDexPose& OutPose);

  /**
   * Switch to next pose source for given handedness.
   *
   * @param Handedness The handedness to switch pose sources for.
   * @param bPrevious Whether to switch to previous pose source instead of the next one.
   * @return true if successful, false otherwise.
   */
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDexPlayerController",
      meta = (ReturnDisplayName = "Success"))
  bool NextPoseSource(const ESuperDexHandedness Handedness, bool bPrevious = false);

  UFUNCTION(BlueprintCallable, Category = "ASuperDexPlayerController")
  void TriggerAlignment();

  /**
   * Get current pose source for given handedness.
   *
   * @param Handedness The handedness to get current pose source for.
   * @return The pose source class, nullptr if unavailable (e.g. due to misconfiguration of the
   * SuperDexPlayerStart).
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDexPlayerController")
  TSubclassOf<USuperDexPoseSource> GetPoseSource(const ESuperDexHandedness Handedness) const;

  /**
   * Set pose source for given handedness. The pose source class must be in the player start listed
   * pose sources for that handedness for this function to work.
   *
   * @param Handedness The handedness to set pose sources for.
   * @param PoseSourceClass The pose source class to set to.
   * @return true if successful, false otherwise.
   */
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDexPlayerController",
      meta = (ReturnDisplayName = "Success"))
  bool SetPoseSource(
      const ESuperDexHandedness Handedness,
      TSubclassOf<USuperDexPoseSource> PoseSourceClass);

 protected:
  virtual void BeginPlay() override;
  virtual void OnPossess(APawn* InPawn) override;
  virtual void OnUnPossess() override;
};
