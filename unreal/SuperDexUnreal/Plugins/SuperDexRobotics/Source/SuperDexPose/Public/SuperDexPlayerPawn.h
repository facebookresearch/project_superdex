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
#include "GameFramework/Pawn.h"
#include "SuperDexPoseTypes.h"
#include "SuperDexPoseableActor.h"

#include "SuperDexPlayerPawn.generated.h"

class ASuperDexPoseableActor;
class ASuperDexPlayerController;
class ASuperDexPlayerStart;
class UCameraComponent;
class UCapsuleComponent;
class USceneComponent;

/** Delegate for poseable actor spawned events with handedness information. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(
    FPoseableActorSpawned,
    ASuperDexPoseableActor*,
    PoseableActor,
    ESuperDexHandedness,
    Handedness);

/** Delegate for poseable actor destroyed events with handedness information. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(
    FPoseableActorDestroyed,
    ASuperDexPoseableActor*,
    PoseableActor,
    ESuperDexHandedness,
    Handedness);

/**
 * Pawn class to be used with SuperDexPose.
 *
 * Manages poseable (hand) actors for the player. Gets pose from player controller and passes it to
 * its managed poseable actors.
 */
UCLASS()
class SUPERDEXPOSE_API ASuperDexPlayerPawn : public APawn {
  GENERATED_BODY()

 public:
  /** Delegate for poseable actor spawned events. */
  UPROPERTY(BlueprintAssignable)
  FPoseableActorSpawned OnPoseableActorSpawned;

  /** Delegate for poseable actor destroyed events. */
  UPROPERTY(BlueprintAssignable)
  FPoseableActorDestroyed OnPoseableActorDestroyed;

  /** Camera component for this pawn. */
  UPROPERTY(BlueprintReadOnly, Category = "SuperDexPlayerPawn")
  TObjectPtr<UCameraComponent> CameraComponent;

  /**
   * VR root location marker for this pawn. Placed at the player's feet.
   * Hand tracking transforms are applied relative to this component's world location.
   */
  UPROPERTY(BlueprintReadOnly, Category = "SuperDexPlayerPawn")
  TObjectPtr<USceneComponent> VrRootComponent;

  /** Collision component for this pawn. */
  UPROPERTY(BlueprintReadOnly, Category = "SuperDexPlayerPawn")
  TObjectPtr<UCapsuleComponent> CollisionComponent;

  /** Current left poseable actor for this pawn. */
  UPROPERTY(BlueprintReadOnly, Category = "SuperDexPlayerPawn", meta = (Keywords = "Hand"))
  TObjectPtr<ASuperDexPoseableActor> LeftPoseableActor = nullptr;

  /** Current right poseable actor for this pawn. */
  UPROPERTY(BlueprintReadOnly, Category = "SuperDexPlayerPawn", meta = (Keywords = "Hand"))
  TObjectPtr<ASuperDexPoseableActor> RightPoseableActor = nullptr;

  /** SuperDexPlayerController possessing this pawn. */
  UPROPERTY(BlueprintReadOnly, Category = "SuperDexPlayerPawn")
  TObjectPtr<ASuperDexPlayerController> SuperDexPlayerController = nullptr;

  /** First SuperDexPlayerStart in the level. */
  UPROPERTY(BlueprintReadOnly, Category = "SuperDexPlayerPawn")
  TObjectPtr<ASuperDexPlayerStart> SuperDexPlayerStart = nullptr;

 private:
  /** Current index within the SuperDexPlayerStart's left pose sources array. */
  int LeftPoseableActorIndex = 0;

  /** Current index within the SuperDexPlayerStart's right pose sources array. */
  int RightPoseableActorIndex = 0;

 public:
  ASuperDexPlayerPawn();

  /**
   * Get pose, for given handedness, from our player controller (using its current selections).
   *
   * @param Handedness The handedness to get the pose for.
   * @param OutPose The output pose.
   * @return true if successful, false otherwise.
   */
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDexPlayerPawn",
      meta = (ReturnDisplayName = "Success"))
  bool GetPose(ESuperDexHandedness Handedness, FSuperDexPose& OutPose);

  /**
   * Spawn next poseable actor, according to the SuperDexPlayerStart's poseable actors
   * configuration, for given handedness.
   *
   * @param Handedness The handedness to spawn the poseable actor for.
   * @param bPrevious Whether to spawn previous poseable actor instead of the next one.
   * @return true if successful, false otherwise.
   */
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDexPlayerPawn",
      meta = (ReturnDisplayName = "Success"))
  bool SpawnNextPoseableActor(ESuperDexHandedness Handedness, bool bPrevious = false);

  /**
   * Spawn poseable actor by class, for given handedness.
   *
   * @param PoseableActorClass The class of the poseable actor to spawn.
   * @param Handedness The handedness to spawn the poseable actor for.
   * @return true if successful, false otherwise.
   */
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDexPlayerPawn",
      meta = (ReturnDisplayName = "Success"))
  bool SpawnPoseableActorByClass(
      TSubclassOf<ASuperDexPoseableActor> PoseableActorClass,
      ESuperDexHandedness Handedness);

  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDexPlayerPawn",
      meta = (ReturnDisplayName = "Success"))
  bool DestroyPoseableActor(ESuperDexHandedness Handedness);

  /**
   * Get current poseable actor for given handedness.
   *
   * @param Handedness The handedness to get the poseable actor for.
   * @return The current poseable actor for given handedness, nullptr if it doesn't exist.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDexPlayerPawn")
  ASuperDexPoseableActor* GetPoseableActor(ESuperDexHandedness Handedness) const;

 protected:
  virtual void BeginPlay() override;

  /**
   * Get currently selected poseable actor class, for given handedness.
   *
   * @param Handedness The handedness to get currently selected poseable actor class for.
   * @return The currently selected poseable actor class, nullptr if unavailable (e.g. due to
   * misconfiguration of the SuperDexPlayerStart).
   */
  TSubclassOf<ASuperDexPoseableActor> GetCurrentPoseableActorClass(ESuperDexHandedness Handedness);
};
