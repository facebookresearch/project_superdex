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

class UMochiArticulationBase;

/**
 * Describes the source types that a SuperDexPoseableComponent can use to get its pose in the order
 * they are checked.
 */
UENUM(BlueprintType)
enum class ESuperDexPoseSourceType : uint8 {
  SuperDexPoseSubsystem,
  SuperDexPoseableComponent,
  SuperDexPoseableActor,
  SuperDexPlayerController,
  Invalid = 255,
};

/** A pose source object paired with the kind of source it is. */
USTRUCT(BlueprintType)
struct FSuperDexPoseSourcePtrTypePair {
  GENERATED_BODY()
  ESuperDexPoseSourceType Type;
  UObject* Source;
};

/** A hand or skeleton pose in a named skeleton topology, with tracking state from its source. */
USTRUCT(BlueprintType) struct SUPERDEXPOSE_API FSuperDexPose {
  GENERATED_BODY()

  /**
   * The name of the source the pose was derived from, this can be an official source provided by an
   * USuperDexPoseSource or it can be an unofficial source such as a pose derived from mochi.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  FName SourceName = "Invalid";

  /**
   * Any additional string tags that we want to add to the pose. These include any tags coming from
   * the pose source or additional tags accumulated from systems along the pose pipeline. Tags can
   * be used by anyone when filtering or modifying pose for any reason. Ideally tag modifications
   * should be additive and nondestructive so they accrue over the pose chain.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  TArray<FName> Tags;

  /**
   * The bot whose skeleton the pose is on. Its links are the skeleton's bones, DofPositions are its
   * joint positions, and SkeletonRootToVrRoot places its root link. Pose sources report poses on
   * the bots of their skeletons (USuperDexPoseSource::GetBot), such as the pose skeleton bots
   * (Skeletons/SuperDexSkeletonBots.h); poses on the same skeleton share their bot.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  TObjectPtr<UMochiArticulationBase> Bot = nullptr;

  /**
   * The joint positions of Bot [rad or m], in its DOF order and excluding the DOFs of a Free root
   * joint (UMochiArticulationBase::GetNumDofs): one for a revolute or prismatic joint, and three, a
   * rotation vector in the joint frame, for a spherical joint. Prismatic positions are in nominal
   * bot units, unaffected by the pose's scale. The bone transforms of the pose are Bot's link
   * transforms at these joint positions
   * (USuperDexPoseFunctionLibrary::ComputeSuperDexPoseLinkToParentTransforms).
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  TArray<double> DofPositions;

  /**
   * The translation, rotation, and scale of the root of the skeleton (Bot's root link) relative to
   * its actor's VrRoot location. For poseable actors attached to the pawn the actor is placed at
   * the VrRoot scene component on the pawn. The scale is the size of the pose relative to Bot.
   * Mochi and other systems do not handle nonuniform scale well so scale should always be left
   * uniform despite UE allowing for float3 scales.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  FTransform SkeletonRootToVrRoot = FTransform::Identity;

  /** True if the source is updating the tracking pose */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  bool bIsTrackingEnabled = false;

  /**
   * True if the source reports confidence in its pose. The meaning of this is independent per
   * source and may be equivilent to something like (TrackingConfidence > 0.5)
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  bool bIsTrackingConfident = false;

  /**
   * The tracking confidence float provided by the source. Many systems rely on this to know if they
   * should trust the pose so some explicitly set the TrackingConfidence to 1 if they are not able
   * to derive a value but still have confidence.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  float TrackingConfidence = 0.f;

  /**
   * Set to true if the system has detected getting back to the same component more than once during
   * pose evaluation. Since pose generation path is functionally static this indicates an infinite
   * loop has formed. This value is automatically set to false when calling GetPose() and will
   * remain true so long as you are unwinding from an infinite loop.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  bool bIsInInfiniteLoop = false;
};

/** Data asset that stores a single pose. */
UCLASS(BlueprintType)
class SUPERDEXPOSE_API USuperDexPoseAsset : public UDataAsset {
  GENERATED_BODY()
 public:
  /** The stored pose. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex")
  FSuperDexPose Pose;
};

/**
 * After looking it up because FTransform does not follow the standard class definitions it cannot
 * be serialized directly. So to use the FJsonSerializer on this type we must wrap it in a class.
 */
USTRUCT(BlueprintType)
struct SUPERDEXPOSE_API FSuperDexSerializableTransform {
  GENERATED_BODY()
  /** The wrapped transform. */
  UPROPERTY(EditAnywhere, Category = "SuperDex") FTransform Data;
};
