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

#include "SuperDexPlayerPawn.h"

#include "Camera/CameraComponent.h"
#include "Components/CapsuleComponent.h"
#include "Kismet/GameplayStatics.h"
#include "SuperDexPlayerController.h"
#include "SuperDexPlayerStart.h"
#include "SuperDexPoseModule.h"
#include "SuperDexPoseTypes.h"
#include "SuperDexPoseableActor.h"

// this exact number seems to be crucial to making "local floor" relative hand positions work
static constexpr float kCapsuleHalfHeight = 88.0f;

ASuperDexPlayerPawn::ASuperDexPlayerPawn() {
  PrimaryActorTick.bCanEverTick = true;

  // create capsule collision component, set as root, set half height to kCapsuleHalfHeight
  CollisionComponent = CreateDefaultSubobject<UCapsuleComponent>(TEXT("Collision"));
  RootComponent = CollisionComponent;
  CollisionComponent->SetCapsuleHalfHeight(kCapsuleHalfHeight);
  CollisionComponent->SetCollisionProfileName(TEXT("OverlapAllDynamic"), true);

  // create vr root component, attach to collision component, offset z to bottom of capsule
  VrRootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("VrRoot"));
  VrRootComponent->SetupAttachment(CollisionComponent);
  VrRootComponent->SetRelativeLocation(FVector(0, 0, -kCapsuleHalfHeight));

  // create camera component, attach to vr root component
  CameraComponent = CreateDefaultSubobject<UCameraComponent>(TEXT("Camera"));
  CameraComponent->SetupAttachment(VrRootComponent);
  CameraComponent->SetRelativeLocation(FVector(0, 0, 2 * kCapsuleHalfHeight));
}

bool ASuperDexPlayerPawn::GetPose(ESuperDexHandedness Handedness, FSuperDexPose& OutPose) {
  if (IsValid(SuperDexPlayerController)) {
    return SuperDexPlayerController->GetPose(Handedness, OutPose);
  } else {
    return false;
  }
}

bool ASuperDexPlayerPawn::SpawnNextPoseableActor(ESuperDexHandedness Handedness, bool bPrevious) {
  if ((Handedness < static_cast<ESuperDexHandedness>(0)) ||
      (Handedness >= ESuperDexHandedness::Count)) {
    UE_LOG(
        LogSuperDexPose,
        Error,
        TEXT(
            "[%hs::%d] called with invalid handedness [%d]."
            " No change to posable actor made."),
        __FUNCTION__,
        __LINE__,
        Handedness);
    return false;
  }

  const int Change = bPrevious ? -1 : 1;

  // increment left/right poseable actor index, wrap around to 0 if over poseable actor count
  if (Handedness == ESuperDexHandedness::Left) {
    if (SuperDexPlayerStart->LeftPoseableActors.Num() > 0) {
      LeftPoseableActorIndex =
          (LeftPoseableActorIndex + Change + SuperDexPlayerStart->LeftPoseableActors.Num()) %
          SuperDexPlayerStart->LeftPoseableActors.Num();
    }
  } else if (Handedness == ESuperDexHandedness::Right) {
    if (SuperDexPlayerStart->RightPoseableActors.Num() > 0) {
      RightPoseableActorIndex =
          (RightPoseableActorIndex + Change + SuperDexPlayerStart->RightPoseableActors.Num()) %
          SuperDexPlayerStart->RightPoseableActors.Num();
    }
  } else {
    // TODO: clean up how the sources are stored to support more than left and right hand sources.
    // In general we probably won't use that generalization, so deferring doing this till it is
    // needed.
    UE_LOG(
        LogSuperDexPose,
        Error,
        TEXT(
            "[%hs::%d] called with unhandled handedness [%d]."
            " Make sure the code is updated to account for new handedness options added."
            " No change to posable actor made."),
        __FUNCTION__,
        __LINE__,
        Handedness);
    return false;
  }

  // only try spawning the next poseable if we've ended up with a valid current poseable class
  TSubclassOf<ASuperDexPoseableActor> CurrentClass = GetCurrentPoseableActorClass(Handedness);
  return IsValid(CurrentClass) ? SpawnPoseableActorByClass(CurrentClass, Handedness) : false;
}

bool ASuperDexPlayerPawn::SpawnPoseableActorByClass(
    TSubclassOf<ASuperDexPoseableActor> PoseableActorClass,
    ESuperDexHandedness Handedness) {
  if ((Handedness != ESuperDexHandedness::Left) && (Handedness != ESuperDexHandedness::Right)) {
    UE_LOG(
        LogSuperDexPose,
        Error,
        TEXT(
            "[%hs::%d] called with unhandled handedness [%d]."
            " No change to posable actor made."),
        __FUNCTION__,
        __LINE__,
        Handedness);
    return false;
  }

  if (!IsValid(PoseableActorClass)) {
    // if an invalid hand or none was in the array dont spawn a hand
    return false;
  }

  TObjectPtr<ASuperDexPoseableActor>* PoseablePtr =
      (Handedness == ESuperDexHandedness::Left) ? &LeftPoseableActor : &RightPoseableActor;

  // destroy existing poseable actor, if there is one, spawn new poseable actor
  if (IsValid(*PoseablePtr) && !DestroyPoseableActor(Handedness)) {
    return false;
  }
  check(IsValid(GetWorld()));
  *PoseablePtr = GetWorld()->SpawnActor<ASuperDexPoseableActor>(PoseableActorClass);

  // attach new poseable actor to VrRoot with zero location/rotation offset
  (*PoseablePtr)
      ->AttachToComponent(
          VrRootComponent, FAttachmentTransformRules::SnapToTargetNotIncludingScale);
  (*PoseablePtr)->SetActorRelativeLocation(FVector::ZeroVector);
  (*PoseablePtr)->SetActorRelativeRotation(FRotator::ZeroRotator);

  (*PoseablePtr)->SetOwner(this);
  (*PoseablePtr)->OwningPawn = this;

  // set poseable index to -1 unless specified class exists in player start's config
  TArray<TSubclassOf<ASuperDexPoseableActor>>* PoseableArrayPtr =
      (Handedness == ESuperDexHandedness::Left) ? &SuperDexPlayerStart->LeftPoseableActors
                                                : &SuperDexPlayerStart->RightPoseableActors;
  int* PoseableIndexPtr = (Handedness == ESuperDexHandedness::Left) ? &LeftPoseableActorIndex
                                                                    : &RightPoseableActorIndex;

  if (PoseableArrayPtr->Contains(PoseableActorClass)) {
    (*PoseableIndexPtr) = PoseableArrayPtr->Find(PoseableActorClass);
  } else {
    (*PoseableIndexPtr) = -1;
  }

  OnPoseableActorSpawned.Broadcast(*PoseablePtr, Handedness);
  return IsValid(*PoseablePtr);
}

bool ASuperDexPlayerPawn::DestroyPoseableActor(ESuperDexHandedness Handedness) {
  TObjectPtr<ASuperDexPoseableActor>* PoseablePtr =
      (Handedness == ESuperDexHandedness::Left) ? &LeftPoseableActor : &RightPoseableActor;
  int* PoseableIndexPtr = (Handedness == ESuperDexHandedness::Left) ? &LeftPoseableActorIndex
                                                                    : &RightPoseableActorIndex;

  if (IsValid(*PoseablePtr)) {
    (*PoseablePtr)->Destroy();
    (*PoseableIndexPtr) = -1;
    OnPoseableActorDestroyed.Broadcast(*PoseablePtr, Handedness);
    return true;
  }

  return false;
}

ASuperDexPoseableActor* ASuperDexPlayerPawn::GetPoseableActor(
    ESuperDexHandedness Handedness) const {
  if ((Handedness != ESuperDexHandedness::Left) && (Handedness != ESuperDexHandedness::Right)) {
    UE_LOG(
        LogSuperDexPose,
        Error,
        TEXT(
            "[%hs::%d] called with unhandled handedness [%d]."
            " Returning nullptr."),
        __FUNCTION__,
        __LINE__,
        Handedness);
    return nullptr;
  }
  return Handedness == ESuperDexHandedness::Left ? LeftPoseableActor : RightPoseableActor;
}

void ASuperDexPlayerPawn::BeginPlay() {
  Super::BeginPlay();

  // grab pointers to our SuperDex player controller and SuperDex player start
  check(IsValid(GetWorld()));
  SuperDexPlayerController = CastChecked<ASuperDexPlayerController>(GetController());
  SuperDexPlayerStart = CastChecked<ASuperDexPlayerStart>(
      UGameplayStatics::GetActorOfClass(GetWorld(), ASuperDexPlayerStart::StaticClass()));

  // ensure we've got a SuperDexPlayerStart in the level
  // only spawn a poseable if we have at least 1 configured for that side
  if (IsValid(SuperDexPlayerStart)) {
    if (SuperDexPlayerStart->bSpawnPoseablesAtBeginPlay) {
      if (SuperDexPlayerStart->LeftPoseableActors.Num() > 0) {
        SpawnPoseableActorByClass(
            GetCurrentPoseableActorClass(ESuperDexHandedness::Left), ESuperDexHandedness::Left);
      }

      if (SuperDexPlayerStart->RightPoseableActors.Num() > 0) {
        SpawnPoseableActorByClass(
            GetCurrentPoseableActorClass(ESuperDexHandedness::Right), ESuperDexHandedness::Right);
      }
    }
  } else {
    checkNoEntry();
  }
}

TSubclassOf<ASuperDexPoseableActor> ASuperDexPlayerPawn::GetCurrentPoseableActorClass(
    ESuperDexHandedness Handedness) {
  if ((Handedness < static_cast<ESuperDexHandedness>(0)) ||
      (Handedness >= ESuperDexHandedness::Count)) {
    UE_LOG(
        LogSuperDexPose,
        Error,
        TEXT(
            "[%hs::%d] called with invalid handedness [%d]."
            " Returning nullptr."),
        __FUNCTION__,
        __LINE__,
        Handedness);
    return nullptr;
  }
  check(IsValid(SuperDexPlayerStart));

  // get current poseable class for specified handedness, return nullptr if poseables list is empty
  if (Handedness == ESuperDexHandedness::Left) {
    return (SuperDexPlayerStart->LeftPoseableActors.Num() > 0)
        ? SuperDexPlayerStart->LeftPoseableActors[LeftPoseableActorIndex]
        : nullptr;
  } else if (Handedness == ESuperDexHandedness::Right) {
    return (SuperDexPlayerStart->RightPoseableActors.Num() > 0)
        ? SuperDexPlayerStart->RightPoseableActors[RightPoseableActorIndex]
        : nullptr;
  } else {
    // TODO: clean up how the sources are stored to support more than left and right hand sources.
    // In general we probably won't use that generalization, so deferring doing this till it is
    // needed.
    UE_LOG(
        LogSuperDexPose,
        Error,
        TEXT(
            "[%hs::%d] called with unhandled handedness [%d]."
            " Make sure the code is updated to account for new handedness options added."
            " Returning nullptr."),
        __FUNCTION__,
        __LINE__,
        Handedness);
    return nullptr;
  }
}
