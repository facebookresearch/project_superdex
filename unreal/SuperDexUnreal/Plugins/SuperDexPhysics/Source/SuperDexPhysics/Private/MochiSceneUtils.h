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

#include "MochiGameInstanceSubsystem.h"
#include "MochiSceneObject.h"

#include <mochi_core/utils/profile.h>

// Shared helpers used by both MochiActorObject and MochiArticulatedActorObject.
//
// Extracted into this header because both .cpp files had identical copies inside
// anonymous namespaces, which causes redefinition errors in UE5 unity builds
// (where multiple .cpp files are concatenated into one translation unit).

inline UMochiSceneObject* GetMochiGameInstanceScene(const UObject* WorldContextObject) {
  MOCHI_PROFILE_SCOPE();
  const UGameInstance* game = WorldContextObject->GetWorld()->GetGameInstance();
  const UMochiGameInstanceSubsystem* subsys = game->GetSubsystem<UMochiGameInstanceSubsystem>();
  if (subsys) {
    return subsys->GetScene();
  }
  return nullptr;
}

inline bool
VerifyOuterAndScene(const UObject* Outer, UMochiSceneObject*& Scene, const TCHAR* ClassName) {
  MOCHI_PROFILE_SCOPE();
  if (Outer == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%s] Failed to create %s because Outer was null!"),
        ClassName,
        ClassName)
    return false;
  }
  if (Scene == nullptr) {
    Scene = GetMochiGameInstanceScene(Outer);
  }
  if (Scene == nullptr || !Scene->IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%s] Failed to create %s because Scene was null or invalid!"),
        ClassName,
        ClassName)
    return false;
  }
  return true;
}

// For factories that create new Mochi actors: an externally owned scene takes none from Unreal.
// Wrapping an actor that already exists (CreateFromExisting*) does not go through this.
inline bool VerifySceneAcceptsNewActors(
    const UMochiSceneObject* Scene,
    const TCHAR* ClassName,
    const FString& ActorName) {
  if (!Scene->IsContentsExternallyOwned()) {
    return true;
  }
  UE_LOG(
      LogMochiPhysics,
      Verbose,
      TEXT("[%s] Scene contents are externally owned; not creating %s"),
      ClassName,
      *ActorName);
  return false;
}
