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

UCLASS(BlueprintType)
class SUPERDEXROBOTICS_API USuperDexRoboticsFunctionLibrary : public UBlueprintFunctionLibrary {
  GENERATED_BODY()

 public:
  /*
   * Exports the SuperDexPhysics game-instance Mochi scene to the archive
   * OutputDir + ".superdex_scene_archive", staging its contents in OutputDir. Bots are exported as
   * bot archives beside the base scene.
   *
   * By default the task's spawned actors are removed first and its spawnable prefabs are exported
   * separately, so the archive is a reusable scene for tasks to populate. With bCaptureLiveScene,
   * the scene is captured as it is, task and spawnables included, e.g. to archive the scene a
   * recording was made in.
   *
   * Pauses the simulation while it exports. Editor only (PIE included): returns false, doing
   * nothing, in packaged builds.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex", meta = (WorldContext = "WorldContext"))
  static bool ExportSuperDexBotScene(
      UObject* WorldContext,
      const FString& ExportName,
      const FString& OutputDir,
      bool bCaptureLiveScene = false);
};
