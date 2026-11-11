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

#include "Retargeters/VrRootToActorRootRetargeter.h"

#include "SuperDexPlayerPawn.h"
#include "SuperDexPoseModule.h"

// #pragma optimize("", off)

bool UVrRootToActorRootRetargeter::GetTransformedPose_Implementation(
    FSuperDexPose const& SourcePose,
    bool bSourceIsValid,
    FSuperDexPose& TransformedPose) {
  ASuperDexPlayerPawn* playerPawn =
      Cast<ASuperDexPlayerPawn>(GetWorld()->GetFirstPlayerController()->GetPawn());
  if (playerPawn == nullptr) {
    return false;
  }

  FTransform VrRootToWorld = playerPawn->VrRootComponent->GetComponentTransform();
  FTransform WorldToActorRoot = GetOwner()->GetTransform().Inverse();
  FTransform SkeletonRootToActorRoot =
      SourcePose.SkeletonRootToVrRoot * VrRootToWorld * WorldToActorRoot;

  TransformedPose = SourcePose;
  TransformedPose.tags.Add(TEXT("VrRootToActorRoot"));
  // Does not change the SkeletonName
  TransformedPose.SkeletonRootToVrRoot = SkeletonRootToActorRoot;
  // Since the SourcePose is guaranteed to be valid the transformed pose will also be valid
  return bSourceIsValid;
}
