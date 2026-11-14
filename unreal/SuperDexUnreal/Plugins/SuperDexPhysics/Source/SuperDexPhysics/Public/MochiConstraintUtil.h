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

#include "MochiPhysicsModule.h"

#include "Components/ActorComponent.h"
#include "Components/SceneComponent.h"
#include "CoreMinimal.h"

#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_physics/mochi_physics.h>

class UMochiArticulationComponent;
class UMochiSceneObject;

SUPERDEXPHYSICS_API USceneComponent*
FindComponentRecursive(UActorComponent* Caller, FName ComponentName, const AActor* Actor = nullptr);

SUPERDEXPHYSICS_API bool EnsureMochiActor(UActorComponent* Caller, USceneComponent* MochiComponent);

// Whether a body is ready to be constrained.
enum class EMochiBodyState {
  Ready,
  // Its Mochi actor doesn't exist yet: an articulation spawns its actor on its first tick.
  Pending,
  // There is no such body, or it can't have a Mochi actor (logged).
  Failed,
};

// A Mochi body a constraint component names: a Mochi mesh component (rigid, soft, shell, rod or
// implicit shape), an articulation, or a link of an articulation.
struct SUPERDEXPHYSICS_API FMochiConstraintBody {
  // The Mochi mesh component, or the articulation.
  USceneComponent* Component = nullptr;
  // Set when the body is an articulation or one of its links.
  UMochiArticulationComponent* Articulation = nullptr;
  // The link, or None for the articulation itself.
  FName Link = NAME_None;

  // The body's Mochi scene and actor. Creates a mesh component's Mochi actor if needed.
  EMochiBodyState GetMochiActor(
      UActorComponent* Caller,
      UMochiSceneObject*& OutScene,
      mochi::ActorHandle& OutActor) const;
  // A position [cm] in the body's local frame (the component's scaled space, or the link frame)
  // in the Mochi actor's local frame [m].
  mochi::Real3 LocalPositionToMochi(const FVector& Position) const;
};

// Finds the body Path names, relative to Caller's owner: a component ("lid"), a link of an
// articulation component ("arm/elbow"), or a body inside a child actor, such as an imported nested
// prefab ("inner/lid", "inner/arm/elbow"). When the owner has no such component, the first segment
// names another actor in the world by its label or the name of its root component, as in a map
// imported from a prefab. Logs and returns false if there is no such body.
SUPERDEXPHYSICS_API bool ResolveMochiConstraintBody(
    UActorComponent* Caller,
    const FString& Path,
    FMochiConstraintBody& OutBody);
