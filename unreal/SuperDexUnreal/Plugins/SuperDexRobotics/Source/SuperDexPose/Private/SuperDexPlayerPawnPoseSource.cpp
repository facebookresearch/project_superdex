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

#include "SuperDexPlayerPawnPoseSource.h"

#include "SuperDexPlayerPawn.h"
#include "SuperDexPoseModule.h"
#include "SuperDexPoseTypes.h"

#include "Kismet/GameplayStatics.h"

bool USuperDexPlayerPawnPoseSource::GetPose(
    const ESuperDexHandedness Handedness,
    FSuperDexPose& OutPose) {
  ASuperDexPlayerPawn* superDexPawn =
      Cast<ASuperDexPlayerPawn>(UGameplayStatics::GetPlayerPawn(GetWorld(), 0));
  if (superDexPawn != nullptr) {
    return superDexPawn->GetPose(Handedness, OutPose);
  }
  return false;
}
