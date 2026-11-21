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

#include "SuperDexSpectatorCamera.h"

#include "Camera/PlayerCameraManager.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/Engine.h"
#include "Engine/TextureRenderTarget2D.h"
#include "HeadMountedDisplayTypes.h"
#include "IHeadMountedDisplay.h"
#include "ISpectatorScreenController.h"
#include "IXRTrackingSystem.h"
#include "Kismet/GameplayStatics.h"
#include "SuperDexPoseModule.h"

namespace {

/** The spectator screen of the active, stereo-enabled HMD, or nullptr. */
ISpectatorScreenController* GetSpectatorScreenController() {
  if (!GEngine || !GEngine->XRSystem.IsValid() || !GEngine->IsStereoscopic3D()) {
    return nullptr;
  }
  IHeadMountedDisplay* Hmd = GEngine->XRSystem->GetHMDDevice();
  if (!Hmd || !Hmd->IsHMDEnabled()) {
    return nullptr;
  }
  return Hmd->GetSpectatorScreenController();
}

/** Fraction of the remaining distance to cover this frame when following at Speed (1/s). */
float SmoothingAlpha(float Speed, float DeltaSeconds) {
  return Speed > 0.0f ? 1.0f - FMath::Exp(-Speed * DeltaSeconds) : 1.0f;
}

} // namespace

ASuperDexSpectatorCamera::ASuperDexSpectatorCamera() {
  PrimaryActorTick.bCanEverTick = true;
  // Follow the camera after the player controller has updated it this frame.
  PrimaryActorTick.TickGroup = TG_PostUpdateWork;

  CaptureComponent = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("Capture"));
  RootComponent = CaptureComponent;
  CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
  CaptureComponent->bCaptureEveryFrame = true;
  CaptureComponent->bCaptureOnMovement = false;
  // Keeps temporal AA history and eye adaptation between frames.
  CaptureComponent->bAlwaysPersistRenderingState = true;
  // Match the main view: scene captures turn temporal AA and separate translucency off by default.
  CaptureComponent->ShowFlags.SetTemporalAA(true);
  CaptureComponent->ShowFlags.SetSeparateTranslucency(true);
  CaptureComponent->ShowFlags.SetMotionBlur(false);
  CaptureComponent->FOVAngle = FieldOfView;
}

void ASuperDexSpectatorCamera::OnConstruction(const FTransform& Transform) {
  Super::OnConstruction(Transform);
  // Show this actor's field of view on the capture in the editor too, not only once playing.
  CaptureComponent->FOVAngle = FieldOfView;
}

void ASuperDexSpectatorCamera::BeginPlay() {
  Super::BeginPlay();

  ISpectatorScreenController* SpectatorScreen = GetSpectatorScreenController();
  if (!SpectatorScreen) {
    UE_LOG(
        LogSuperDexPose,
        Log,
        TEXT("[%hs::%d] No VR spectator screen (no stereo HMD); [%s] stays idle."),
        __FUNCTION__,
        __LINE__,
        *GetName());
    CaptureComponent->bCaptureEveryFrame = false;
    SetActorTickEnabled(false);
    return;
  }

  if (!RenderTarget) {
    RenderTarget = NewObject<UTextureRenderTarget2D>(this, TEXT("SpectatorRenderTarget"));
    RenderTarget->RenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA8;
    RenderTarget->ClearColor = FLinearColor::Black;
    // The spectator screen shows the texture's bytes as they are, so they must be gamma encoded
    // like the back buffer. By default a new RGBA8 render target is linear and gets linear output.
    RenderTarget->TargetGamma = 2.2f;
    RenderTarget->InitAutoFormat(FMath::Max(Resolution.X, 1), FMath::Max(Resolution.Y, 1));
    RenderTarget->UpdateResourceImmediate(true);
  }
  CaptureComponent->TextureTarget = RenderTarget;
  CaptureComponent->FOVAngle = FieldOfView;

  PreviousSpectatorScreenMode = static_cast<uint8>(SpectatorScreen->GetSpectatorScreenMode());
  SpectatorScreen->SetSpectatorScreenTexture(RenderTarget);
  SpectatorScreen->SetSpectatorScreenMode(ESpectatorScreenMode::Texture);
  bSpectatorScreenActive = true;
}

void ASuperDexSpectatorCamera::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  if (bSpectatorScreenActive) {
    if (ISpectatorScreenController* SpectatorScreen = GetSpectatorScreenController()) {
      SpectatorScreen->SetSpectatorScreenMode(
          static_cast<ESpectatorScreenMode>(PreviousSpectatorScreenMode));
      SpectatorScreen->SetSpectatorScreenTexture(nullptr);
    }
    bSpectatorScreenActive = false;
  }
  Super::EndPlay(EndPlayReason);
}

void ASuperDexSpectatorCamera::Tick(float DeltaSeconds) {
  Super::Tick(DeltaSeconds);

  const APlayerCameraManager* CameraManager = UGameplayStatics::GetPlayerCameraManager(this, 0);
  if (!CameraManager) {
    return;
  }

  const FVector TargetLocation = CameraManager->GetCameraLocation();
  FRotator TargetRotator = CameraManager->GetCameraRotation();
  if (bLockRoll) {
    TargetRotator.Roll = 0.0;
  }
  const FQuat TargetRotation = TargetRotator.Quaternion();

  FVector Location = TargetLocation;
  FQuat Rotation = TargetRotation;
  if (bHasView) {
    Location = FMath::Lerp(
        HeadLocation, TargetLocation, SmoothingAlpha(LocationSmoothingSpeed, DeltaSeconds));
    Rotation = FQuat::Slerp(
        GetActorQuat(), TargetRotation, SmoothingAlpha(RotationSmoothingSpeed, DeltaSeconds));
  }
  HeadLocation = Location;
  SetActorLocationAndRotation(Location + Rotation.RotateVector(ViewOffset), Rotation);
  bHasView = true;

  // Scene captures pick up post process volumes but not the camera's own post process settings
  // (e.g. the pawn's camera component), so copy those from the player's view.
  const FMinimalViewInfo& View = CameraManager->GetCameraCacheView();
  CaptureComponent->PostProcessSettings = View.PostProcessSettings;
  CaptureComponent->PostProcessBlendWeight = View.PostProcessBlendWeight;

  CaptureComponent->FOVAngle = FieldOfView;
}
