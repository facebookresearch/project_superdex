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

#include "MochiPhysicsSettings.h"

#include "MochiGameInstanceSubsystem.h"
#include "MochiPhysicsModule.h"

#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "UObject/ConstructorHelpers.h"

#if !UE_BUILD_SHIPPING
namespace {

void RestartDebuggerIfActiveSpaceChanged(const FMochiSpaceDefinition& PreviousActiveSpace) {
  const FMochiSpaceDefinition& ActiveSpace = UMochiPhysicsSettings::GetActiveSimulationSpace();
  IMochiPhysicsModule* Module = IMochiPhysicsModule::Get();
  mochi::Context* Context = Module ? Module->GetMochiContext() : nullptr;
  if (!Context || ActiveSpace == PreviousActiveSpace) {
    return;
  }

  auto& DebugServer = Context->GetDebugServer();
  const uint16 PreviousPort = DebugServer.GetPort();
  if (PreviousPort == 0) {
    return;
  }

  const mochi::LogFn PreviousLogCallback = mochi::Context::GetLogCallback();
  DebugServer.Stop();
  DebugServer.SetCoordinateSpace(ToMochiCoordinateSpace(ActiveSpace));
  DebugServer.Start(PreviousPort);
  mochi::Context::SetLogCallback(PreviousLogCallback);

  if (DebugServer.GetPort() != PreviousPort) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "Mochi debug server restart could not rebind port %u; now on port %u (0 means not hosting)."),
        PreviousPort,
        DebugServer.GetPort());
  }
}

} // namespace
#endif

UMochiSolverParams* FMochiSceneSettings::GetSolverParams() const {
  return Cast<UMochiSolverParams>(SolverParams.TryLoad());
}

UMochiContactMatrix* FMochiSceneSettings::GetContactMatrix() const {
  return Cast<UMochiContactMatrix>(ContactMatrix.TryLoad());
}

UMochiPhysicsSettings::UMochiPhysicsSettings() {
  DefaultSceneSettings.SolverParams =
      ConstructorHelpers::FObjectFinder<UMochiSolverParams>(
          TEXT(
              "MochiSolverParams'/SuperDexPhysics/Params/MSP_Default.MSP_Default'"))
          .Object;
  DefaultSceneSettings.ContactMatrix =
      ConstructorHelpers::FObjectFinder<UMochiContactMatrix>(
          TEXT(
              "MochiContactMatrix'/SuperDexPhysics/Params/MCM_Default.MCM_Default'"))
          .Object;
}

FMochiSpaceDefinition UMochiPhysicsSettings::ActiveSimulationSpace = kSuperDexSpace;
bool UMochiPhysicsSettings::bHasActiveOverride = false;

const FMochiSpaceDefinition& UMochiPhysicsSettings::GetActiveSimulationSpace() {
  if (bHasActiveOverride) {
    return ActiveSimulationSpace;
  }
  return GetDefault<UMochiPhysicsSettings>()->SimulationSpace;
}

bool UMochiPhysicsSettings::HasActiveSimulationSpaceOverride() {
  return bHasActiveOverride;
}

void UMochiPhysicsSettings::SetActiveSimulationSpaceOverride(const FMochiSpaceDefinition& Space) {
#if !UE_BUILD_SHIPPING
  const FMochiSpaceDefinition PreviousActiveSpace = GetActiveSimulationSpace();
#endif

  ActiveSimulationSpace = Space;
  bHasActiveOverride = true;

#if !UE_BUILD_SHIPPING
  RestartDebuggerIfActiveSpaceChanged(PreviousActiveSpace);
#endif
}

void UMochiPhysicsSettings::ClearActiveSimulationSpaceOverride() {
#if !UE_BUILD_SHIPPING
  const FMochiSpaceDefinition PreviousActiveSpace = GetActiveSimulationSpace();
#endif

  bHasActiveOverride = false;

#if !UE_BUILD_SHIPPING
  RestartDebuggerIfActiveSpaceChanged(PreviousActiveSpace);
#endif
}

FName UMochiPhysicsSettings::GetCategoryName() const {
  return FName(TEXT("Plugins"));
}

AMochiSceneSettingsActor::AMochiSceneSettingsActor() {
  // try to get settings from project settings
  const auto projectSettings = GetDefault<UMochiPhysicsSettings>();
  if (projectSettings) {
    SceneSettings = projectSettings->DefaultSceneSettings;
    // initialize SimulationSpace from project-level setting so the default value in the details
    // panel matches what the project is currently using
    SceneSettings.SimulationSpace = projectSettings->SimulationSpace;
  } else {
    SceneSettings.SolverParams =
        ConstructorHelpers::FObjectFinder<UMochiSolverParams>(
            TEXT(
                "MochiSolverParams'/SuperDexPhysics/Params/MSP_Default.MSP_Default'"))
            .Object;
    SceneSettings.ContactMatrix =
        ConstructorHelpers::FObjectFinder<UMochiContactMatrix>(
            TEXT(
                "MochiContactMatrix'/SuperDexPhysics/Params/MCM_Default.MCM_Default'"))
            .Object;
  }
}

void AMochiSceneSettingsActor::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  // Clear the SimulationSpace override when this actor is torn down (e.g. during a level
  // transition via OpenLevel). Without this, the override from this level would leak into
  // the next level since the GameInstance subsystem persists across level transitions.
  if (SceneSettings.bOverrideSimulationSpace) {
    UMochiPhysicsSettings::ClearActiveSimulationSpaceOverride();
    UE_LOG(
        LogMochiPhysics,
        Display,
        TEXT("AMochiSceneSettingsActor: Cleared SimulationSpace override on EndPlay"));
  }
  Super::EndPlay(EndPlayReason);
}

void AMochiSceneSettingsActor::BeginPlay() {
  Super::BeginPlay();
  // return if this instance has already been applied by UMochiGameInstanceSubsystem
  if (bApplied) {
    return;
  }
  // apply simulation space override so all converters pick it up
  if (SceneSettings.bOverrideSimulationSpace) {
    UMochiPhysicsSettings::SetActiveSimulationSpaceOverride(SceneSettings.SimulationSpace);
    UE_LOG(
        LogMochiPhysics,
        Display,
        TEXT("AMochiSceneSettingsActor: Overriding SimulationSpace from level settings"));
  }
  // else get the subsystem scene and reconfigure settings
  const auto game = GetWorld()->GetGameInstance();
  const auto subsys = game->GetSubsystem<UMochiGameInstanceSubsystem>();
  if (subsys) {
    subsys->bStartPaused = SceneSettings.bStartPaused;
    if (const auto scene = subsys->GetScene()) {
      scene->ConfigureSettings(SceneSettings);
      bApplied = true;
      UE_LOG(
          LogMochiPhysics,
          Display,
          TEXT("AMochiSceneSettingsActor: Applied settings on BeginPlay"));
    }
  }
}
