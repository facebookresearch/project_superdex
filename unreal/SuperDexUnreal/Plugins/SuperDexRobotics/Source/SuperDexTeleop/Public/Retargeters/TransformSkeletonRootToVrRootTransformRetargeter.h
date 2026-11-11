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

#include "TransformSkeletonRootToVrRootTransformRetargeter.generated.h"

// This class is used to add transforms and offsets to tracking poses to provide a different
// SkeletonRootToVrRoot to components further down the chain. It is otherwise identical to a normal
// USuperDexPoseableComponent. Note that the FTransforms are not protected and thus may result in
// tearing when changed across threads.
UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class SUPERDEXTELEOP_API UTransformSkeletonRootToVrRootTransformRetargeter
    : public USuperDexRetargeter {
  GENERATED_BODY()
 public:
  // The TransformedPose_SkeletonRootToVrRoot = SkeletonOutRootToSkeletonInRoot *
  // InputSource_SkeletonRootToVrRoot * VrInRootToVrOutRoot
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  FTransform SkeletonOutRootToSkeletonInRoot;

  // The TransformedPose_SkeletonRootToVrRoot = SkeletonOutRootToSkeletonInRoot *
  // InputSource_SkeletonRootToVrRoot * VrInRootToVrOutRoot
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  FTransform VrInRootToVrOutRoot;

  virtual bool GetTransformedPose_Implementation(
      FSuperDexPose const& SourcePose,
      bool SourceIsValid,
      FSuperDexPose& TransformedPose) override;
};
