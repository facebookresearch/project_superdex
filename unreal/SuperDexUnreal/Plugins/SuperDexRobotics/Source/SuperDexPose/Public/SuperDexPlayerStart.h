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
#include "GameFramework/PlayerStart.h"
#include "SuperDexPoseTypes.h"

#include "SuperDexPlayerStart.generated.h"

class USuperDexPoseSource;
class ASuperDexPoseableActor;

/**
 * Player start class to be used with SuperDexPose.
 *
 * Handles pose source and poseable actor configuration for a level. Placing a SuperDexPlayerPawn
 * into a level directly, without a SuperDexPlayerStart, will not work.
 */
UCLASS()
class SUPERDEXPOSE_API ASuperDexPlayerStart : public APlayerStart {
  GENERATED_BODY()

 public:
  /** Array of pose sources to use for the left hand, first element is the default. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDexPlayerStart")
  TArray<TSubclassOf<USuperDexPoseSource>> LeftPoseSources;

  /** Array of pose sources to use for the right hand, first element is the default. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDexPlayerStart")
  TArray<TSubclassOf<USuperDexPoseSource>> RightPoseSources;

  /** Array of poseable actors to use for the left hand, first element is the default. */
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "SuperDexPlayerStart",
      meta = (Keywords = "Hand"))
  TArray<TSubclassOf<ASuperDexPoseableActor>> LeftPoseableActors;

  /** Array of poseable actors to use for the right hand, first element is the default. */
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "SuperDexPlayerStart",
      meta = (Keywords = "Hand"))
  TArray<TSubclassOf<ASuperDexPoseableActor>> RightPoseableActors;

  /** Whether to automatically spawn poseables at begin play or not. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDexPlayerStart")
  bool bSpawnPoseablesAtBeginPlay = true;
};
