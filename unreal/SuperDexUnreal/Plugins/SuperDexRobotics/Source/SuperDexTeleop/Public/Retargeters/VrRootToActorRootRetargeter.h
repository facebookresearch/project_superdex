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

#include "Retargeters/SuperDexRetargeter.h"

#include "VrRootToActorRootRetargeter.generated.h"

// Modifies the SkeletonRootToVrRoot transform to be SkeletonRootToActorRoot. Effectively changing
// the pose so that it is relative to the parent actor of this component rather than the pawn's
// VrRoot leaving the SkeletonRoot stationary in UEWorldSpace.
UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class SUPERDEXTELEOP_API UVrRootToActorRootRetargeter : public USuperDexRetargeter {
  GENERATED_BODY()
 public:
  virtual bool GetTransformedPose_Implementation(
      FSuperDexPose const& SourcePose,
      bool SourceIsValid,
      FSuperDexPose& TransformedPose) override;
};
