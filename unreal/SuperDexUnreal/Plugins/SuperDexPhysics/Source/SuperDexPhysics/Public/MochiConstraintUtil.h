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

SUPERDEXPHYSICS_API USceneComponent*
FindComponentRecursive(UActorComponent* Caller, FName ComponentName, const AActor* Actor = nullptr);

SUPERDEXPHYSICS_API bool EnsureMochiActor(UActorComponent* Caller, USceneComponent* MochiComponent);
