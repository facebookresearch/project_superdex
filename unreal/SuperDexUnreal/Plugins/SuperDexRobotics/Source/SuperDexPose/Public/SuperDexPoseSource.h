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
#include "SuperDexPoseTypes.h"
#include "UObject/NoExportTypes.h"

#include "SuperDexPoseSource.generated.h"

class UWorld;

/** Invalid thread ID constant. */
constexpr uint32 kInvalidThreadId = std::numeric_limits<uint32>::max();

/**
 * The connection between some source of pose and SuperDexPose.
 *
 * Manages the connection between some, usually external, source of pose and SuperDexPose. Has the
 * same lifetime as the pose subsystem, which is tied to UWorld.
 *
 * The pose subsystem creates one instance of every non-abstract subclass in every loaded module,
 * so a pose source in another plugin (see Plugins/PoseSourcePlugins) is available wherever that
 * plugin is enabled, with no registration. A subclass that can not work in some setup (no device
 * SDK, or not configured) overrides ShouldCreatePoseSource so that it is not created there.
 */
UCLASS(Abstract)
class SUPERDEXPOSE_API USuperDexPoseSource : public UObject {
  GENERATED_BODY()

 public:
  USuperDexPoseSource();
  ~USuperDexPoseSource();

  /**
   * Get pose for given handedness.
   *
   * @param Handedness The handedness to get pose for.
   * @param OutPose The output pose.
   * @return true if successful, false otherwise.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDexPoseSource")
  virtual bool GetPose(const ESuperDexHandedness Handedness, FSuperDexPose& OutPose);

  /**
   * Whether the pose subsystem of World should create this pose source, like
   * USubsystem::ShouldCreateSubsystem. Called on the class default object. A pose source that is
   * not created does not exist in that world: the subsystem returns no pose for it.
   *
   * @param World The world of the pose subsystem.
   * @return true (the default) to create the pose source, false to leave it out.
   */
  virtual bool ShouldCreatePoseSource(UWorld* World) const;

  /** Deinitialize this pose source. */
  virtual void Deinitialize();

  /**
   * Get the necessary thread ID to update the cache for this pose source.
   *
   * @return The necessary thread ID.
   */
  uint32 GetNecessaryThreadId() {
    return NecessaryThreadId;
  }

 protected:
  /** Initialize this pose source. */
  virtual void Initialize();

  /** Necessary thread ID to update the cache for this pose source. */
  uint32 NecessaryThreadId = kInvalidThreadId;
};
