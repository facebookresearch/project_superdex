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
#include "MochiConversions.h"

#include <mochi_physics/utils/mochi_prefab.h>

// Helpers shared by the prefab importer's source files.
namespace SuperDexPrefabImporterInternal {

FString ToFString(const mochi::DynamicString& S);

// The normalized path of a prefab Path refers to from the prefab at SourcePath.
FString ResolvePrefabPath(
    const FString& RootDir,
    const FString& SourcePath,
    const mochi::DynamicString& Path);

mochi::prefab::ScenePrefab LoadPrefab(const FString& SourcePath, bool& bOutSuccess);

// Where a rigid or soft actor's component sits in the actor's frame: Mochi bakes the actor's scale,
// then the shape offset, into the shape.
template <typename TActorPrefab>
FTransform GetShapeToActor(const TActorPrefab& Actor, const FMochiToUnrealConverter& M2U) {
  return FTransform(
      M2U.RotationToUnreal(Actor.shapeRotation),
      M2U.TranslationToUnreal(Actor.shapeTranslation),
      M2U.DirectionToUnreal(Actor.scale).GetAbs());
}

} // namespace SuperDexPrefabImporterInternal
