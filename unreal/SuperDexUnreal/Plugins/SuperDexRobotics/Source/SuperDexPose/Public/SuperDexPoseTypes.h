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

#include "SuperDexHandTypes.h"

#include "Engine/DataAsset.h"
#include "Misc/EnumRange.h"

#include "SuperDexPoseTypes.generated.h"

// Describes the source types that a SuperDexPoseableComponent can use to get its pose in the order
// they are checked.
UENUM(BlueprintType)
enum class ESuperDexPoseSourceType : uint8 {
  SuperDexPoseSubsystem,
  SuperDexPoseableComponent,
  SuperDexPoseableActor,
  SuperDexPlayerController,
  Invalid = 255,
};

USTRUCT(BlueprintType)
struct FSuperDexPoseSourcePtrTypePair {
  GENERATED_BODY()
  ESuperDexPoseSourceType Type;
  UObject* Source;
};

USTRUCT(BlueprintType) struct SUPERDEXPOSE_API FSuperDexPose {
  GENERATED_BODY()

  // The name of the source the pose was derived from, this can be an official source provided by an
  // USuperDexPoseSource or it can be an unofficial source such as a pose derived from mochi.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  FName SourceName = "Invalid";

  // The name associated with the skeleton topology being used. These are commonly defined alongside
  // the enums that list bones.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  FName SkeletonName = "Invalid";

  // Any additional string tags that we want to add to the pose. These include any tags coming from
  // the pose source or additional tags accumulated from systems along the pose pipeline. Tags can
  // be used by anyone when filtering or modifying pose for any reason. Ideally tag modifications
  // should be additive and nondestructive so they accrue over the pose chain.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  TArray<FName> tags;

  // The bone rotations associated with the pose. These are in LocalToParent space and in bone index
  // order for their specific skeleton. Whole pose requires knowing the position offsets of the
  // bones gotten from the rest pose combined with the rotations provided here. Scale is fixed.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  TArray<FQuat> BoneRotations;

  // [Optional] The bone locations associated with the pose. These are in LocalToParent space and in
  // bone index order for their specific skeleton. Whole pose requires knowing the position offsets
  // of the bones gotten from the rest pose combined with the locations provided here. Scale is
  // fixed. If you set this, it must be the same length as BoneRotations.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  TArray<FVector> BoneLocations;

  // The translation, rotation, and scale of the root of the skeleton relative to its actor's VrRoot
  // location. For poseable actors attached to the pawn the actor is placed at the VrRoot scene
  // component on the pawn. Mochi and other systems do not handle nonuniform scale well so scale
  // should always be left uniform despite UE allowing for float3 scales.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  FTransform SkeletonRootToVrRoot = FTransform::Identity;

  // True if the source is updating the tracking pose
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  bool bIsTrackingEnabled = false;

  // True if the source reports confidence in its pose. The meaning of this is independent per
  // source and may be equivilent to something like (TrackingConfidence > 0.5)
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  bool bIsTrackingConfident = false;

  // The tracking confidence float provided by the source. Many systems rely on this to know if they
  // should trust the pose so some explicitly set the TrackingConfidence to 1 if they are not able
  // to derive a value but still have confidence.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  float TrackingConfidence = 0.f;

  // Set to true if the system has detected getting back to the same component more than once during
  // pose evaluation. Since pose generation path is functionally static this indicates an infinite
  // loop has formed. This value is automatically set to false when calling GetPose() and will
  // remain true so long as you are unwinding from an infinite loop.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  bool bIsInInfiniteLoop = false;
};

UCLASS(BlueprintType)
class SUPERDEXPOSE_API USuperDexPoseAsset : public UDataAsset {
  GENERATED_BODY()
 public:
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex")
  FSuperDexPose Pose;
};

// After looking it up because FTransform does not follow the standard class definitions it cannot
// be serialized directly. So to use the FJsonSerializer on this type we must wrap it in a class.
USTRUCT(BlueprintType)
struct SUPERDEXPOSE_API FSuperDexSerializableTransform {
  GENERATED_BODY()
  UPROPERTY(EditAnywhere, Category = "SuperDex") FTransform Data;
};
