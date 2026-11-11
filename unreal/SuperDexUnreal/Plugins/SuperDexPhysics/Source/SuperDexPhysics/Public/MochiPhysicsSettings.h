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

#include <Engine/DeveloperSettings.h>
#include <UObject/UnrealType.h>
#include "CoreMinimal.h"
#include "GameFramework/Info.h"
#include "MochiAssets.h"
#include "MochiDebugDraw.h"
#include "MochiSpaceConverter.h"
#include "UObject/Object.h"

#include "MochiPhysicsSettings.generated.h"

USTRUCT(BlueprintType)
struct FMochiSceneSettings {
  GENERATED_BODY()
 public:
  // If true, the SimulationSpace defined below will override the project-level SimulationSpace.
  UPROPERTY(BlueprintReadOnly, EditAnywhere, Category = "Scene")
  bool bOverrideSimulationSpace = false;

  // The coordinate space used for simulation. Only used if bOverrideSimulationSpace is true.
  UPROPERTY(
      BlueprintReadOnly,
      EditAnywhere,
      Category = "Scene",
      meta = (EditCondition = "bOverrideSimulationSpace"))
  FMochiSpaceDefinition SimulationSpace = kSuperDexSpace;

  // Gravity vector in m/s^2.
  UPROPERTY(BlueprintReadOnly, EditAnywhere, Category = "Scene")
  FVector GravityVector = FVector(0, 0, -9.81);

  // Parameters that control how the scene steps on the simulation thread.
  UPROPERTY(BlueprintReadOnly, EditAnywhere, Category = "Scene")
  FMochiAsyncStepParams StepParams;

  UPROPERTY(
      Config,
      BlueprintReadOnly,
      EditAnywhere,
      Category = "Scene",
      meta = (AllowedClasses = "/Script/SuperDexPhysics.MochiSolverParams"))
  FSoftObjectPath SolverParams;

  UPROPERTY(
      Config,
      BlueprintReadOnly,
      EditAnywhere,
      Category = "Scene",
      meta = (AllowedClasses = "/Script/SuperDexPhysics.MochiContactMatrix"))
  FSoftObjectPath ContactMatrix;

  UPROPERTY(BlueprintReadOnly, EditAnywhere, Category = "Scene")
  bool bStartPaused = false;

  UPROPERTY(BlueprintReadOnly, EditAnywhere, Category = "DebugDraw")
  bool bEnableDebugDraw = false;

  // Scale the length of all lines drawn by mochi. This is useful for scaling elements like contact
  // forces, but will also alter how elements like mesh edges are drawn. Use at your own risk.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "DebugDraw")
  float DebugDraw_Line_Length_Multiplier = 1.0f;

  UPROPERTY(Config, BlueprintReadOnly, EditAnywhere, Category = "DebugDraw")
  FMochiDebugDrawFlags DebugDrawFlags;

  UMochiSolverParams* GetSolverParams() const;
  UMochiContactMatrix* GetContactMatrix() const;
};

// Project-level SuperDex Physics settings, shown under Project Settings > Plugins > SuperDex
// Physics.
UCLASS(Config = Game, defaultconfig, DisplayName = "SuperDex Physics")
class SUPERDEXPHYSICS_API UMochiPhysicsSettings : public UDeveloperSettings {
  GENERATED_BODY()
 public:
  // The coordinate space used for simulation. This choice has no effect on the Unreal
  // authoring experience, which retains Unreal's left-hand convention, but affects the
  // coordinate space in which Mochi is simulated as well as exported prefabs. This is
  // useful if you want to author scenes in Unreal and export them to another application
  // which simulates Mochi in a different coordinate space. By default, this is the SuperDex
  // space (+X forward, +Y left, +Z up) with meter scaling, the convention of SuperDex bot and
  // scene files.
  UPROPERTY(Config, BlueprintReadOnly, EditAnywhere, Category = "Simulation")
  FMochiSpaceDefinition SimulationSpace = kSuperDexSpace;

  // Start the Mochi physics debugger server when the plugin initializes. This setting is ignored
  // in Shipping builds. The server advertises the effective simulation space. A scene override
  // restarts it on the same port so connected debugger clients can immediately reconnect.
  UPROPERTY(Config, BlueprintReadOnly, EditAnywhere, Category = "Debugging")
  bool bEnablePhysicsDebuggerConnection = false;

  // Number of worker threads Mochi should use. If the value is <= zero, then Mochi will
  // automatically determine how many threads to use.
  UPROPERTY(Config, BlueprintReadOnly, EditAnywhere, Category = "Simulation")
  int NumWorkerThreads = 0;

  // Normally, Mochi actors are divided into simulation islands dynamically. However, you still have
  // the option to force them into a single island. This may be useful for debugging, but it will
  // hurt performance.
  UPROPERTY(Config, BlueprintReadOnly, EditAnywhere, Category = "Simulation")
  bool bForceSingleIsland = false;

  // Default settings used for scene creation if the scene does not contain a
  // MochiSceneSettingsActor.
  UPROPERTY(
      Config,
      BlueprintReadOnly,
      EditAnywhere,
      meta = (ShowOnlyInnerProperties),
      Category = "Default Scene")
  FMochiSceneSettings DefaultSceneSettings;

  // This setting allows the Editor to run at full performance while not in focus
  // during Play or Simulate.
  // **Note that minimizing the Editor will cause the Editor to throttle no matter what.**
  // This has no effect in Packaged Builds.
  UPROPERTY(
      Config,
      BlueprintReadOnly,
      EditAnywhere,
      Category = "Editor",
      meta = (DisplayName = "Turn off 'Use Less CPU when in Background' during Play/Simulate"))
  bool bTurnOffUseLessCPUInBackgroundDuringPlay = true;

  // If true, imported SuperDex prefabs, Mochi scenes and SuperDex scenes are organized under
  // PrefabRootFolder, mirroring their on-disk directory structure relative to .superdex_root.
  // Otherwise they are imported into the folder they are dropped into. Prefabs and bots that an
  // import references are always imported into subfolders of its folder.
  UPROPERTY(Config, BlueprintReadOnly, EditAnywhere, Category = "Editor")
  bool bOrganizePrefabsInRootFolder = false;

  // Content Browser root folder for imported SuperDex prefabs and Mochi scenes.
  UPROPERTY(
      Config,
      BlueprintReadOnly,
      EditAnywhere,
      Category = "Editor",
      meta = (EditCondition = "bOrganizePrefabsInRootFolder", EditConditionHides))
  FDirectoryPath PrefabRootFolder = {TEXT("/Game/Prefabs")};

  // Map copied to start new maps generated from SuperDex prefab and Mochi scene imports in Map
  // mode. Defaults to the Basic template of the New Level wizard. If empty, new maps start empty.
  UPROPERTY(
      Config,
      BlueprintReadOnly,
      EditAnywhere,
      Category = "Editor",
      meta = (AllowedClasses = "/Script/Engine.World"))
  FSoftObjectPath PrefabMapTemplate{
      TEXT("/Engine/Maps/Templates/Template_Default.Template_Default")};

 public:
  UMochiPhysicsSettings();
  virtual FName GetCategoryName() const override;

  // Returns the effective SimulationSpace: the scene-level override if one has been set,
  // otherwise the project-level setting. Simulation consumers should call this instead of
  // reading SimulationSpace directly.
  static const FMochiSpaceDefinition& GetActiveSimulationSpace();

  // Return whether a scene override has been set.
  static bool HasActiveSimulationSpaceOverride();

  // Set a runtime override for SimulationSpace (called when a scene settings actor
  // with bOverrideSimulationSpace is found). Does not modify the CDO or .ini. Outside Shipping,
  // restarts a listening debugger on the same port when the active space changes.
  static void SetActiveSimulationSpaceOverride(const FMochiSpaceDefinition& Space);

  // Clear the runtime override, reverting to the project-level setting. Outside Shipping,
  // restarts a listening debugger on the same port when the active space changes.
  static void ClearActiveSimulationSpaceOverride();

 private:
  static FMochiSpaceDefinition ActiveSimulationSpace;
  static bool bHasActiveOverride;
};

// Actor which can *optionally* be placed into a level to override project level Mochi scene
// settings.
UCLASS(
    HideCategories =
        (Actor,
         Advanced,
         Display,
         Events,
         Object,
         Attachment,
         Info,
         Input,
         Blueprint,
         Layers,
         Tags,
         Replication,
         Networking,
         Physics,
         Cooking))
class SUPERDEXPHYSICS_API AMochiSceneSettingsActor : public AActor {
  GENERATED_BODY()
 public:
  UPROPERTY(
      BlueprintReadOnly,
      EditAnywhere,
      meta = (ShowOnlyInnerProperties),
      Category = "Scene Settings")
  FMochiSceneSettings SceneSettings;

 public:
  AMochiSceneSettingsActor();
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

 public:
  bool bApplied = false;
};
