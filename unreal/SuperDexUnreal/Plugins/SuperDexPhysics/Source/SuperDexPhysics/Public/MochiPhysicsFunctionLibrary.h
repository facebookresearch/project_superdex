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
#include "MochiActorObject.h"
#include "MochiArticulatedActorObject.h"
#include "MochiGameInstanceSubsystem.h"
#include "MochiTypes.h"

#include "MochiPhysicsFunctionLibrary.generated.h"

/** Blueprint helpers for the game-instance Mochi scene. */
UCLASS(BlueprintType)
class SUPERDEXPHYSICS_API UMochiPhysicsFunctionLibrary : public UBlueprintFunctionLibrary {
  GENERATED_BODY()

 public:
  /**
   * Set gravity on the game-instance Mochi scene.
   *
   * @param WorldContext Object used to resolve the world and game instance.
   * @param GravityVector Gravity vector to apply to the scene.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void SetMochiGravity(UObject* WorldContext, FVector GravityVector);

  /**
   * Set a layer-to-layer contact rule on the game-instance scene.
   *
   * @param WorldContext Object used to resolve the world and game instance.
   * @param Entry Layer-to-layer contact rule.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void SetMochiContactMatrixEntry(
      UObject* WorldContext,
      const FMochiContactMatrixEntry& Entry);

  /**
   * Disable contact symmetrically between two actors.
   *
   * @param WorldContext Object used to resolve the world and game instance.
   * @param ActorA First actor of the contact pair.
   * @param ActorB Second actor of the contact pair.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void DisableContact(
      UObject* WorldContext,
      const UMochiActorObject* ActorA,
      const UMochiActorObject* ActorB);

  /**
   * Disable contact symmetrically between an actor and selected links of an articulation.
   *
   * @param WorldContext Object used to resolve the world and game instance.
   * @param ActorA First actor of the contact pair.
   * @param ActorB Second actor of the contact pair.
   * @param Link Articulation link index; -1 selects all links.
   */
  UFUNCTION(
      BlueprintCallable,
      DisplayName = "Disable Contact (Articulated)",
      Category = "Mochi",
      meta = (WorldContext = "WorldContext"))
  static void DisableContact_Articulated(
      UObject* WorldContext,
      const UMochiActorObject* ActorA,
      const UMochiArticulatedActorObject* ActorB,
      int Link = -1);

  /**
   * Enable debug drawing for the game-instance scene.
   *
   * @param WorldContext Object used to resolve the world and game instance.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void EnableMochiDebugDraw(UObject* WorldContext);

  /**
   * Disable debug drawing for the game-instance scene.
   *
   * @param WorldContext Object used to resolve the world and game instance.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void DisableMochiDebugDraw(UObject* WorldContext);

  /**
   * Toggle debug drawing for the game-instance scene.
   *
   * @param WorldContext Object used to resolve the world and game instance.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void ToggleMochiDebugDraw(UObject* WorldContext);

  /**
   * Get whether debug drawing is enabled for the game instance.
   *
   * @param WorldContext Object used to resolve the world and game instance.
   * @return Whether game-instance debug drawing is enabled.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static bool IsMochiDebugDrawEnabled(UObject* WorldContext);

  /**
   * Set the pause state of the game-instance scene.
   *
   * @param WorldContext Object used to resolve the world and game instance.
   * @param bPaused Requested pause state.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void SetPaused(UObject* WorldContext, bool bPaused);

  /**
   * Get the pause state of the game-instance scene.
   *
   * @param WorldContext Object used to resolve the world and game instance.
   * @return Pause state of the scene, or true when no scene is available.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static bool IsPaused(UObject* WorldContext);

  /**
   * Get the scene step count accumulated since startup.
   *
   * @param WorldContext Object used to resolve the world and game instance.
   * @return Number of native scene steps accumulated since startup.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static int GetNumMochiStepsSinceStart(UObject* WorldContext);

  /**
   * Get the simulated time accumulated since startup.
   *
   * @param WorldContext Object used to resolve the world and game instance.
   * @return Accumulated simulated time in seconds.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static double GetTotalSimulatedTimeSinceStartSec(UObject* WorldContext);

  /**
   * Start recording the game-instance scene in the requested directory.
   *
   * @param WorldContext Object used to resolve the world and game instance.
   * @param FilePath Directory in which to store the recording or exported scene.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void StartRecording(UObject* WorldContext, const FString& FilePath);

  /**
   * Stop recording the game-instance scene.
   *
   * @param WorldContext Object used to resolve the world and game instance.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void StopRecording(UObject* WorldContext);

  /**
   * Export the game-instance scene as a named native prefab in the requested directory.
   *
   * @param WorldContext Object used to resolve the world and game instance.
   * @param ExportName Name of the exported native prefab.
   * @param FilePath Directory in which to store the recording or exported scene.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void
  ExportMochiScene(UObject* WorldContext, const FString& ExportName, const FString& FilePath);

  /**
   * Get the Mochi subsystem for the context object's game instance.
   *
   * @param WorldContext Object used to resolve the world and game instance.
   * @return The physics subsystem for the context's game instance.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static UMochiGameInstanceSubsystem* GetMochiGameInstanceSubsystem(UObject* WorldContext);
};
