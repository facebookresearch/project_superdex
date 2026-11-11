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

#include "SuperDexPoseableActor.h"

#include "SuperDexPlayerPawn.h"
#include "SuperDexPoseSubsystem.h"

ASuperDexPoseableActor::ASuperDexPoseableActor() {
  PrimaryActorTick.bCanEverTick = true;
}

bool ASuperDexPoseableActor::GetPose(FSuperDexPose& OutPose) {
  if (bEnableGetPose == false) {
    return false;
  }

  // don't bother getting pose if this is being called before BeginPlay
  // this usually happens when the Mochi thread asks for pose too early
  if (!HasActorBegunPlay())
    return false;

  // if hand pose source override is set, use that
  if (PoseSourceOverride) {
    return GetWorld()->GetSubsystem<USuperDexPoseSubsystem>()->GetPose(
        PoseSourceOverride, Handedness, OutPose);
  }

  // if attached to a SuperDex player pawn, use what it says
  if (OwningPawn) {
    return OwningPawn->GetPose(Handedness, OutPose);
  }

  // There is no pose source, just return false as though disabled
  return false;
}
