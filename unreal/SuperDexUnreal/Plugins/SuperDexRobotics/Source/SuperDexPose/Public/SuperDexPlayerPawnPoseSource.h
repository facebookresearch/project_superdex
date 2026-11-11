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
#include "SuperDexPoseSource.h"
#include "SuperDexPoseTypes.h"
#include "UObject/Object.h"

#include "SuperDexPlayerPawnPoseSource.generated.h"

// Redirects any GetPose calls to the player pawn. If this component is attached to the player pawn,
// then it will always return false.
UCLASS()
class SUPERDEXPOSE_API USuperDexPlayerPawnPoseSource : public USuperDexPoseSource {
  GENERATED_BODY()

 public:
  bool GetPose(const ESuperDexHandedness Handedness, FSuperDexPose& OutPose) override;
};
