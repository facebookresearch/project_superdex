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

#include "SuperDexPlayerController.h"

#include "Kismet/GameplayStatics.h"
#include "SuperDexPlayerPawn.h"
#include "SuperDexPlayerPawnPoseSource.h"
#include "SuperDexPlayerStart.h"
#include "SuperDexPoseModule.h"
#include "SuperDexPoseSubsystem.h"
#include "SuperDexPoseTypes.h"

bool ASuperDexPlayerController::GetPose(
    const ESuperDexHandedness Handedness,
    FSuperDexPose& OutPose) {
  if (GetPoseSource(Handedness) == nullptr) {
    return false;
  }
  // break the infinite loop that can be caused by a source asking the player pawn what its pose is
  if (GetPoseSource(Handedness)->IsChildOf(USuperDexPlayerPawnPoseSource::StaticClass())) {
    return false;
  }
  // get pose, for specified handedness,from pose subsystem
  return GetWorld()->GetSubsystem<USuperDexPoseSubsystem>()->GetPose(
      GetPoseSource(Handedness), Handedness, OutPose);
}

bool ASuperDexPlayerController::NextPoseSource(
    const ESuperDexHandedness Handedness,
    bool bPrevious) {
  if ((Handedness < static_cast<ESuperDexHandedness>(0)) ||
      (Handedness >= ESuperDexHandedness::Count)) {
    UE_LOG(
        LogSuperDexPose,
        Error,
        TEXT(
            "[%hs::%d] called with invalid handedness [%d]."
            " No change to source made."),
        __FUNCTION__,
        __LINE__,
        Handedness);
    return false;
  }

  const int Delta = bPrevious ? -1 : 1;
  bool bChanged = false;

  // increment left/right pose source index, wrapping back to 0
  if (Handedness == ESuperDexHandedness::Left) {
    const int numLeftSources = PlayerStart->LeftPoseSources.Num();
    if (numLeftSources > 1) {
      LeftPoseSourceIndex = (LeftPoseSourceIndex + Delta + numLeftSources) % numLeftSources;
      bChanged = true;
    }
  } else if (Handedness == ESuperDexHandedness::Right) {
    const int numRightSources = PlayerStart->RightPoseSources.Num();
    if (numRightSources > 1) {
      RightPoseSourceIndex = (RightPoseSourceIndex + Delta + numRightSources) % numRightSources;
      bChanged = true;
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
            " No change to source made."),
        __FUNCTION__,
        __LINE__,
        Handedness);
    return false;
  }

  // early return if no change to hand pose source
  if (!bChanged)
    return false;

  // grab new current hand pose source, use it to broadcast some OnPoseSourceChanged events
  // Sources could be nullptr if they were set to none in the list
  TSubclassOf<USuperDexPoseSource> CurrentHandPoseSource = GetPoseSource(Handedness);
  OnPoseSourceChanged.Broadcast(CurrentHandPoseSource, Handedness);

  // grab the current poseable actor for this handedness and fire the same event on it
  ASuperDexPoseableActor* CurrentPoseableActor = Handedness == ESuperDexHandedness::Left
      ? PlayerPawn->LeftPoseableActor
      : PlayerPawn->RightPoseableActor;
  if (IsValid(CurrentPoseableActor)) {
    CurrentPoseableActor->OnPoseSourceChanged.Broadcast(CurrentHandPoseSource);
  }

  return true;
}

void ASuperDexPlayerController::TriggerAlignment() {
  OnDoAlginmentKeypress.Broadcast();
}

bool ASuperDexPlayerController::SetPoseSource(
    const ESuperDexHandedness Handedness,
    TSubclassOf<USuperDexPoseSource> PoseSourceClass) {
  if ((Handedness < static_cast<ESuperDexHandedness>(0)) ||
      (Handedness >= ESuperDexHandedness::Count)) {
    UE_LOG(
        LogSuperDexPose,
        Error,
        TEXT(
            "[%hs::%d] called with invalid handedness [%d]."
            " No change to source made."),
        __FUNCTION__,
        __LINE__,
        Handedness);
    return false;
  }

  // check to make sure the specified pose source exists within the player start's config
  // increment left/right pose source index, wrapping back to 0
  if (Handedness == ESuperDexHandedness::Left) {
    if (PlayerStart->LeftPoseSources.Contains(PoseSourceClass)) {
      LeftPoseSourceIndex = PlayerStart->LeftPoseSources.Find(PoseSourceClass);
    } else {
      return false;
    }
  } else if (Handedness == ESuperDexHandedness::Right) {
    if (PlayerStart->RightPoseSources.Contains(PoseSourceClass)) {
      RightPoseSourceIndex = PlayerStart->RightPoseSources.Find(PoseSourceClass);
    } else {
      return false;
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
            " No change to source made."),
        __FUNCTION__,
        __LINE__,
        Handedness);
    return false;
  }

  // grab new current hand pose source, use it to broadcast some OnPoseSourceChanged events
  // Sources could be nullptr if they were set to none in the list
  TSubclassOf<USuperDexPoseSource> CurrentHandPoseSource = GetPoseSource(Handedness);
  OnPoseSourceChanged.Broadcast(CurrentHandPoseSource, Handedness);

  // grab the current poseable actor for this handedness and fire the same event on it
  ASuperDexPoseableActor* CurrentPoseableActor = Handedness == ESuperDexHandedness::Left
      ? PlayerPawn->LeftPoseableActor
      : PlayerPawn->RightPoseableActor;
  if (IsValid(CurrentPoseableActor)) {
    CurrentPoseableActor->OnPoseSourceChanged.Broadcast(CurrentHandPoseSource);
  }

  return true;
}

void ASuperDexPlayerController::BeginPlay() {
  // ensure there's a SuperDexPlayerStart in the level and store a reference to it
  PlayerStart = (ASuperDexPlayerStart*)UGameplayStatics::GetActorOfClass(
      GetWorld(), ASuperDexPlayerStart::StaticClass());
  if (IsValid(PlayerStart)) {
    // doing an extra check here to see if the PlayerStart in the level is completely unconfigured.
    // fully empty poseable and pose source lists won't break anything, but most of the time this
    // happens it's because somebody didn't know to configure it (here's your notice)
    if ((PlayerStart->LeftPoseableActors.IsEmpty() && PlayerStart->RightPoseableActors.IsEmpty() &&
         PlayerStart->LeftPoseSources.IsEmpty() && PlayerStart->RightPoseSources.IsEmpty())) {
      UE_LOG(
          LogSuperDexPose,
          Error,
          TEXT(
              "[%hs::%d] Player start has no poseable actors or pose sources defined."
              " The pawn will not have hands or tracking."),
          __FUNCTION__,
          __LINE__);
    }
  } else {
    UE_LOG(
        LogSuperDexPose,
        Error,
        TEXT(
            "[%hs::%d] Could not find an ASuperDexPlayerStart in the level and thus cannot properly handle the pose sources."),
        __FUNCTION__,
        __LINE__);
  }

  Super::BeginPlay();
}

void ASuperDexPlayerController::OnPossess(APawn* InPawn) {
  Super::OnPossess(InPawn);
  PlayerPawn = Cast<ASuperDexPlayerPawn>(InPawn);
}

void ASuperDexPlayerController::OnUnPossess() {
  PlayerPawn = nullptr;
}

TSubclassOf<USuperDexPoseSource> ASuperDexPlayerController::GetPoseSource(
    const ESuperDexHandedness Handedness) const {
  // return left/right pose source at current index if sources list is non-empty
  if (PlayerStart) {
    if (Handedness == ESuperDexHandedness::Left) {
      return PlayerStart->LeftPoseSources.Num() > 0
          ? PlayerStart->LeftPoseSources[LeftPoseSourceIndex]
          : nullptr;
    } else if (Handedness == ESuperDexHandedness::Right) {
      return PlayerStart->RightPoseSources.Num() > 0
          ? PlayerStart->RightPoseSources[RightPoseSourceIndex]
          : nullptr;
    } else {
      // TODO: clean up how the sources are stored to support more than left and right hand sources.
      // In general we probably won't use that generalization, so deferring doing this till it is
      // needed.
    }
  }

  return nullptr;
}
