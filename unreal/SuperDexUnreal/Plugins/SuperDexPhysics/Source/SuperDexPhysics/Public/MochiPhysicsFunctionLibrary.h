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

UCLASS(BlueprintType)
class SUPERDEXPHYSICS_API UMochiPhysicsFunctionLibrary : public UBlueprintFunctionLibrary {
  GENERATED_BODY()

 public:
  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void SetMochiGravity(UObject* WorldContext, FVector GravityVector);

  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void SetMochiContactMatrixEntry(
      UObject* WorldContext,
      const FMochiContactMatrixEntry& Entry);

  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void DisableContact(
      UObject* WorldContext,
      const UMochiActorObject* ActorA,
      const UMochiActorObject* ActorB);

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

  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void EnableMochiDebugDraw(UObject* WorldContext);

  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void DisableMochiDebugDraw(UObject* WorldContext);

  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void ToggleMochiDebugDraw(UObject* WorldContext);

  UFUNCTION(BlueprintPure, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static bool IsMochiDebugDrawEnabled(UObject* WorldContext);

  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void SetPaused(UObject* WorldContext, bool bPaused);

  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static bool IsPaused(UObject* WorldContext);

  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static int GetNumMochiStepsSinceStart(UObject* WorldContext);

  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static double GetTotalSimulatedTimeSinceStartSec(UObject* WorldContext);

  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void StartRecording(UObject* WorldContext, const FString& filePath);

  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void StopRecording(UObject* WorldContext);

  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static void
  ExportMochiScene(UObject* WorldContext, const FString& exportName, const FString& filePath);

  UFUNCTION(BlueprintCallable, Category = "Mochi", meta = (WorldContext = "WorldContext"))
  static UMochiGameInstanceSubsystem* GetMochiGameInstanceSubsystem(UObject* WorldContext);
};
