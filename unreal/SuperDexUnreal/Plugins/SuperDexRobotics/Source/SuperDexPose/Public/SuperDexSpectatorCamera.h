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
#include "GameFramework/Actor.h"

#include "SuperDexSpectatorCamera.generated.h"

class USceneCaptureComponent2D;
class UTextureRenderTarget2D;

/**
 * Spectator camera for recording VR sessions.
 *
 * Follows the player's view (the HMD) with optional smoothing, renders it into a 16:9 render
 * target, and shows that render target on the VR spectator screen (the desktop window of a PC VR
 * build, or the VR Preview window in PIE). Place one in a level. Does nothing without an HMD whose
 * spectator screen can be controlled (e.g. non-VR PIE or standalone Quest).
 */
UCLASS(Blueprintable)
class SUPERDEXPOSE_API ASuperDexSpectatorCamera : public AActor {
  GENERATED_BODY()

 public:
  /** Scene capture that renders the spectator view. */
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SuperDexSpectatorCamera")
  TObjectPtr<USceneCaptureComponent2D> CaptureComponent;

  /**
   * Render target to capture into and show on the spectator screen. If unset, one of
   * `Resolution` is created at BeginPlay. An assigned 8-bit one needs a Target Gamma of 2.2, or
   * the spectator screen shows it too dark.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDexSpectatorCamera")
  TObjectPtr<UTextureRenderTarget2D> RenderTarget;

  /** Resolution of the render target created when `RenderTarget` is unset. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDexSpectatorCamera")
  FIntPoint Resolution = FIntPoint(1920, 1080);

  /** Horizontal field of view of the spectator view, in degrees. */
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "SuperDexSpectatorCamera",
      meta = (ClampMin = "10.0", ClampMax = "170.0", Units = "deg"))
  float FieldOfView = 100.0f;

  /**
   * Offset of the spectator view from the head, in the view's own frame (X forward, Y right,
   * Z up). A negative X pulls the view back behind the head to show more of what's in front.
   */
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "SuperDexSpectatorCamera",
      meta = (Units = "cm"))
  FVector ViewOffset = FVector(-20.0, 0.0, 0.0);

  /** How fast the view follows the head's position (1/s). 0 follows it exactly. */
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "SuperDexSpectatorCamera",
      meta = (ClampMin = "0.0"))
  float LocationSmoothingSpeed = 0.0f;

  /** How fast the view follows the head's rotation (1/s). 0 follows it exactly. */
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "SuperDexSpectatorCamera",
      meta = (ClampMin = "0.0"))
  float RotationSmoothingSpeed = 8.0f;

  /** Keep the view level by dropping the head's roll. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDexSpectatorCamera")
  bool bLockRoll = true;

  ASuperDexSpectatorCamera();

  virtual void OnConstruction(const FTransform& Transform) override;
  virtual void Tick(float DeltaSeconds) override;

 protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

 private:
  /** Whether this actor took over the spectator screen at BeginPlay. */
  bool bSpectatorScreenActive = false;

  /** Spectator screen mode before BeginPlay, restored at EndPlay. */
  uint8 PreviousSpectatorScreenMode = 0;

  /** Whether the view has been placed at least once (the first placement is not smoothed). */
  bool bHasView = false;

  /** Smoothed head location, before `ViewOffset`. */
  FVector HeadLocation = FVector::ZeroVector;
};
