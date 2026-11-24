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
#include "Kismet/BlueprintFunctionLibrary.h"

#include "SuperDexRoboticsFunctionLibrary.generated.h"

/** Blueprint helpers for SuperDex robotics scenes. */
UCLASS(BlueprintType)
class SUPERDEXROBOTICS_API USuperDexRoboticsFunctionLibrary : public UBlueprintFunctionLibrary {
  GENERATED_BODY()

 public:
  /**
   * Exports the SuperDexPhysics game-instance Mochi scene to the archive
   * OutputDir + ".superdex_scene_archive", staging its contents in OutputDir. Bots are exported as
   * bot archives beside the base scene.
   *
   * By default the task's spawned actors are removed first and its spawnable prefabs are exported
   * separately, so the archive is a reusable scene for tasks to populate. With bCaptureLiveScene,
   * the scene is captured as it is, task and spawnables included, e.g. to archive the scene a
   * recording was made in.
   *
   * Base-scene actors are exported in their current pose: articulated joint configurations and
   * deformations as well as root transforms. Spawnables are exported as spawned.
   *
   * Pauses the simulation while it exports. Editor only (PIE included): returns false, doing
   * nothing, in packaged builds.
   *
   * @param WorldContext Object whose game instance owns the Mochi scene.
   * @param ExportName Name of the exported scene and its .superdex_scene file.
   * @param OutputDir Directory to stage the export in; the archive is OutputDir +
   *   ".superdex_scene_archive".
   * @param bCaptureLiveScene Capture the live scene, task and spawnables included, instead of the
   *   reusable base scene.
   * @return True if the scene was exported.
   */
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDex Robotics|Scene",
      meta = (WorldContext = "WorldContext"))
  static bool ExportSuperDexBotScene(
      UObject* WorldContext,
      const FString& ExportName,
      const FString& OutputDir,
      bool bCaptureLiveScene = false);
};
