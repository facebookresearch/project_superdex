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
#include "GameFramework/Actor.h"
#include "SuperDexPoseSource.h"
#include "SuperDexPoseTypes.h"

#include "SuperDexPoseableActor.generated.h"

class ASuperDexPlayerPawn;
class USuperDexPoseSource;
class USuperDexPoseableComponent;

/** Delegate for pose source changed events. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(
    FPoseSourceChanged,
    TSubclassOf<USuperDexPoseSource>,
    PoseSource);

/**
 * Actor for poseable objects in SuperDexPose.
 *
 * Component hierarchy should contain SuperDexPoseableComponents, which will inherit some properties
 * (e.g. handedness) and get pose from this actor. Usually managed by a SuperDexPlayerPawn, from
 * which this actor inherits some properties and gets pose, though can be used standalone if
 * overrides are set. Usually used for hands, but could be anything representable by skeletal mesh
 * (e.g. robots).
 */
UCLASS(Abstract)
class SUPERDEXPOSE_API ASuperDexPoseableActor : public AActor {
  GENERATED_BODY()

 public:
  /** Delegate for pose source changed events. */
  UPROPERTY(BlueprintAssignable)
  FPoseSourceChanged OnPoseSourceChanged;

  /** Display name of this actor, as shown on the wristband component. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDexPoseableActor")
  FText DisplayName = FText::FromString(TEXT("Poseable Actor"));

  /**
   * Override pose source class for this actor.
   * If set, will get pose from pose subsystem directly instead of from owning pawn.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDexPoseableActor")
  TSubclassOf<USuperDexPoseSource> PoseSourceOverride;

  /** Owning pawn currently managing this poseable actor. */
  UPROPERTY(BlueprintReadOnly, Category = "SuperDexPoseableActor")
  TObjectPtr<ASuperDexPlayerPawn> OwningPawn;

  /** Handedness of this actor, applies to attached poseable components w/o override set. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDexPoseableActor")
  ESuperDexHandedness Handedness = ESuperDexHandedness::Right;

  /** If false, always return false and early out when GetPose() is called */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDexPoseableActor")
  bool bEnableGetPose = true;

 public:
  ASuperDexPoseableActor();

  /**
   * Get pose from owning pawn, or directly from pose subsystem if override is set.
   *
   * @param OutPose The output pose.
   * @return true if successful, false otherwise.
   */
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDexPoseableActor",
      meta = (ReturnDisplayName = "Success"))
  virtual bool GetPose(FSuperDexPose& OutPose);

  /**
   * Get the primary/displayed poseable component.
   * You *must* implement this function in your poseable actor blueprint, or things won't work.
   *
   * @param OutPrimaryPoseable The output primary poseable component.
   * @return true if successful, false otherwise.
   */
  UFUNCTION(
      BlueprintCallable,
      BlueprintImplementableEvent,
      Category = "SuperDexPoseableActor",
      meta = (ReturnDisplayName = "Success"))
  bool GetPrimaryPoseableComponent(USuperDexPoseableComponent*& OutPrimaryPoseable) const;

  /**
   * Get the tracked/root poseable component.
   * You *must* implement this function in your poseable actor blueprint, or things won't work.
   *
   * @param OutTrackedPoseable The output tracked poseable component.
   * @return true if successful, false otherwise.
   */
  UFUNCTION(
      BlueprintCallable,
      BlueprintImplementableEvent,
      Category = "SuperDexPoseableActor",
      meta = (ReturnDisplayName = "Success"))
  bool GetTrackedPoseableComponent(USuperDexPoseableComponent*& OutTrackedPoseable) const;
};
