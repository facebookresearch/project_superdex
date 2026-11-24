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

/** Base pose, joint motion and contact forces gathered for an articulation controller. */
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiArticulationState {
  GENERATED_BODY()
  /** The current transform of the articulation base in world space. */
  FTransform BaseTransform;

  /** Current joint positions [rad or m], excluding base DOFs. */
  TArray<mochi::real> JointPositions;

  /** Current joint velocities [rad/s or m/s], excluding base DOFs. */
  TArray<mochi::real> JointVelocities;
};

/**
 * Gather the native articulation state used by a controller.
 *
 * @param ArticulatedActor Native articulation supplying or receiving motion data.
 * @return State gathered from the native articulation.
 */
SUPERDEXPHYSICS_API FMochiArticulationState
ExtractArticulatedActorState(const mochi::Actor* ArticulatedActor);
/**
 * Command joint efforts on a native articulation.
 *
 * @param ArticulatedActor Native articulation supplying or receiving motion data.
 * @param JointEfforts Joint efforts to command.
 */
SUPERDEXPHYSICS_API void SetArticulatedActorJointEfforts(
    mochi::Actor* ArticulatedActor,
    const TArray<mochi::real>& JointEfforts);

/**
 * Teleport an articulated actor: place its links at WorldFromLinks (one per link, projected onto
 * its joints) at rest. Mochi resets the target of its pose controller, if it has one, to the new
 * pose. Call on the Mochi thread.
 */
SUPERDEXPHYSICS_API void TeleportArticulatedActor(
    mochi::Actor* ArticulatedActor,
    mochi::Span<const mochi::TransformRT> WorldFromLinks,
    mochi::Error& Error);

/**
 * Build native pose-controller targets from base and joint targets.
 *
 * @param ArticulatedActor Native articulation supplying or receiving motion data.
 * @param BaseTransformTarget Desired world transform of the articulation base.
 * @param JointPositionTargets Desired joint-position targets.
 * @return Native pose-controller target values.
 */
SUPERDEXPHYSICS_API TArray<mochi::real> BuildPoseControllerTargets(
    const mochi::Actor* ArticulatedActor,
    const FTransform& BaseTransformTarget,
    const TArray<mochi::real>& JointPositionTargets);
/**
 * Convert joint DOF values (excluding base DOFs) of an articulated actor from the simulation space
 * convention to Unreal's, or back. Only Spherical and non-root Free joints are affected; see
 * ConvertArticulatedDofsToUnreal.
 *
 * @param ArticulatedActor Native articulation supplying or receiving motion data.
 * @param JointDofs Motion-coordinate values to read and convert in place.
 */
SUPERDEXPHYSICS_API void ConvertArticulatedActorDofsToUnreal(
    const mochi::Actor* ArticulatedActor,
    TArray<mochi::real>& JointDofs);
/**
 * Convert an articulation's motion values to native conventions in place.
 *
 * @param ArticulatedActor Native articulation supplying or receiving motion data.
 * @param JointDofs Motion-coordinate values to read and convert in place.
 */
SUPERDEXPHYSICS_API void ConvertArticulatedActorDofsToMochi(
    const mochi::Actor* ArticulatedActor,
    TArray<mochi::real>& JointDofs);
