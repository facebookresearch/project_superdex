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
#include "MochiTypes.h"

#include "MochiArticulationUtil.generated.h"

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiArticulationState {
  GENERATED_BODY()
  // The current transform of the articulation base in world space.
  FTransform BaseTransform;
  // Current joint positions [rad or m], excluding base DOFs.
  TArray<mochi::real> JointPositions;
  // Current joint velocities [rad/s or m/s], excluding base DOFs.
  TArray<mochi::real> JointVelocities;
};

SUPERDEXPHYSICS_API FMochiArticulationState
ExtractArticulatedActorState(const mochi::Actor* ArticulatedActor);
SUPERDEXPHYSICS_API void SetArticulatedActorJointEfforts(
    mochi::Actor* ArticulatedActor,
    const TArray<mochi::real>& JointEfforts);
SUPERDEXPHYSICS_API TArray<mochi::real> BuildPoseControllerTargets(
    const mochi::Actor* ArticulatedActor,
    const FTransform& BaseTransformTarget,
    const TArray<mochi::real>& JointPositionTargets);
// Convert joint DOF values (excluding base DOFs) of an articulated actor from the simulation space
// convention to Unreal's, or back. Only Spherical and non-root Free joints are affected; see
// ConvertArticulatedDofsToUnreal.
SUPERDEXPHYSICS_API void ConvertArticulatedActorDofsToUnreal(
    const mochi::Actor* ArticulatedActor,
    TArray<mochi::real>& JointDofs);
SUPERDEXPHYSICS_API void ConvertArticulatedActorDofsToMochi(
    const mochi::Actor* ArticulatedActor,
    TArray<mochi::real>& JointDofs);
