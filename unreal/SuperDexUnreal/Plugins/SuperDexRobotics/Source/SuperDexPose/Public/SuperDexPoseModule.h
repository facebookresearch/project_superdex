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
#include "Modules/ModuleManager.h"

DECLARE_LOG_CATEGORY_EXTERN(LogSuperDexPose, Log, All);

/** Module class for SuperDexPose functionality. */
class FSuperDexPoseModule : public IModuleInterface {
 private:
  FTransform TrackingSpaceToHandParentSpaceTransform = FTransform::Identity;

 public:
  /**
   * Get the SuperDexPose module instance.
   *
   * @return The SuperDexPose module instance, or nullptr if not loaded.
   */
  static FSuperDexPoseModule* Get() {
    return FModuleManager::LoadModulePtr<FSuperDexPoseModule>("SuperDexPose");
  }

  // Transform that takes a hand root/wrist pose from the source tracking space (e.g. mocap), to
  // the space of the parent of the hand meshes (likely either the player pawn root, or a sub-node
  // of the player pawn, such as 'VROrigin', at the player's feet). This transform is applied to
  // hand root/wrist poses to get them into a convenient space for consumption by callers.
  virtual void SetTrackingSpaceToHandParentSpaceTransform(const FTransform& Transform);

  virtual FTransform GetTrackingSpaceToHandParentSpaceTransform() const;

  // Common utility functions that require .cpp implementation
  // This isn't a great place to put these, but, no other home for them right now
  // Modules are not statically-linked together, so must go through some virtual interface to be
  // accessible to other modules
  virtual bool
  LoadFromJsonFile(const UStruct* structDefinition, void* structData, TCHAR const* filePath);

  virtual bool
  SaveToJsonFile(const UStruct* structDefinition, const void* structData, TCHAR const* filePath);

  /** IModuleInterface implementation */
  virtual void StartupModule() override;
  virtual void ShutdownModule() override;
};
