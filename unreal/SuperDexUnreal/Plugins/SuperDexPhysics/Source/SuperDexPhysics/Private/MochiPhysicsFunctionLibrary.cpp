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

#include "MochiPhysicsFunctionLibrary.h"

#include "MochiPhysicsModule.h"

#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "HAL/IConsoleManager.h"

#include <mochi_physics/utils/mochi_prefab.h>

//------------------------------------------------------------------------------
// Console Commands
//------------------------------------------------------------------------------

static FAutoConsoleCommandWithWorld GMochiPause(
    TEXT("Mochi.Pause"),
    TEXT("Pause Mochi simulation."),
    FConsoleCommandWithWorldDelegate::CreateLambda([](UWorld* World) {
      const auto game = World->GetGameInstance();
      const auto subsys = game->GetSubsystem<UMochiGameInstanceSubsystem>();
      if (subsys && subsys->GetScene()) {
        subsys->GetScene()->SetPaused(true);
      }
    }));

static FAutoConsoleCommandWithWorld GMochiUnpause(
    TEXT("Mochi.Unpause"),
    TEXT("Unpause Mochi simulation."),
    FConsoleCommandWithWorldDelegate::CreateLambda([](UWorld* World) {
      const auto game = World->GetGameInstance();
      const auto subsys = game->GetSubsystem<UMochiGameInstanceSubsystem>();
      if (subsys && subsys->GetScene()) {
        subsys->GetScene()->SetPaused(false);
      }
    }));

static FAutoConsoleCommandWithWorld GMochiStepPause(
    TEXT("Mochi.StepPause"),
    TEXT("Step the Mochi simulation by exactly one step then pause (or re-pause)."),
    FConsoleCommandWithWorldDelegate::CreateLambda([](UWorld* World) {
      const auto game = World->GetGameInstance();
      const auto subsys = game->GetSubsystem<UMochiGameInstanceSubsystem>();
      if (subsys && subsys->GetScene()) {
        subsys->GetScene()->StepPause();
      }
    }));

//------------------------------------------------------------------------------
// UMochiPhysicsFunctionLibrary
//------------------------------------------------------------------------------

void UMochiPhysicsFunctionLibrary::SetMochiGravity(UObject* WorldContext, FVector GravityVector) {
  const auto subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (subsys) {
    if (const auto scene = subsys->GetScene()) {
      scene->SetGravity(GravityVector);
    }
  }
}

void UMochiPhysicsFunctionLibrary::SetMochiContactMatrixEntry(
    UObject* WorldContext,
    const FMochiContactMatrixEntry& Entry) {
  const auto subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (subsys) {
    if (const auto scene = subsys->GetScene()) {
      scene->SetContactMatrixEntry(Entry);
    }
  }
}

void UMochiPhysicsFunctionLibrary::DisableContact(
    UObject* WorldContext,
    const UMochiActorObject* ActorA,
    const UMochiActorObject* ActorB) {
  const auto subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (subsys) {
    const auto scene = subsys->GetScene();
    if (scene && ActorA && ActorB && ActorA->IsValid() && ActorB->IsValid()) {
      auto handleA = ActorA->GetMochiActorHandle();
      auto handleB = ActorB->GetMochiActorHandle();
      scene->QueueOrExecuteCommand([handleA, handleB](mochi::Scene* Scene) {
        mochi::ErrorLog e;
        Scene->EnableActorContactSymmetric(
            handleA, handleB, /*enable*/ false, mochi::IncludeNestedActors::No, e);
      });
    }
  } else {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "[UMochiPhysicsFunctionLibrary::AllowContact] ActorA and/or ActorB was null or invalid!"));
  }
}

void UMochiPhysicsFunctionLibrary::DisableContact_Articulated(
    UObject* WorldContext,
    const UMochiActorObject* ActorA,
    const UMochiArticulatedActorObject* ActorB,
    int Link) {
  const auto subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (subsys) {
    const auto scene = subsys->GetScene();
    auto handlesB = ActorB->GetNestedLinkActorHandles();
    if (Link >= 0) {
      handlesB = {handlesB[Link]};
    }
    if (scene && ActorA && ActorB && ActorA->IsValid() && ActorB->IsValid()) {
      auto handleA = ActorA->GetMochiActorHandle();
      scene->QueueOrExecuteCommand([handleA, handlesB](mochi::Scene* Scene) {
        for (auto& handleB : handlesB) {
          mochi::ErrorLog e;
          Scene->EnableActorContactSymmetric(
              handleA, handleB, /*enable*/ false, mochi::IncludeNestedActors::No, e);
        }
      });
    }
  } else {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "[UMochiPhysicsFunctionLibrary::AllowContact] ActorA and/or ActorB was null or invalid!"));
  }
}

void UMochiPhysicsFunctionLibrary::EnableMochiDebugDraw(UObject* WorldContext) {
  const auto subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (subsys) {
    subsys->EnableDebugDraw();
  }
}

void UMochiPhysicsFunctionLibrary::DisableMochiDebugDraw(UObject* WorldContext) {
  const auto subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (subsys) {
    subsys->DisableDebugDraw();
  }
}

void UMochiPhysicsFunctionLibrary::ToggleMochiDebugDraw(UObject* WorldContext) {
  const auto subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (subsys) {
    if (subsys->IsDebugDrawEnabled()) {
      subsys->DisableDebugDraw();
    } else {
      subsys->EnableDebugDraw();
    }
  }
}

bool UMochiPhysicsFunctionLibrary::IsMochiDebugDrawEnabled(UObject* WorldContext) {
  const auto subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (subsys) {
    return subsys->IsDebugDrawEnabled();
  }
  return false;
}

void UMochiPhysicsFunctionLibrary::SetPaused(UObject* WorldContext, bool bPaused) {
  const auto subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (subsys && subsys->GetScene()) {
    subsys->GetScene()->SetPaused(bPaused);
  }
}

bool UMochiPhysicsFunctionLibrary::IsPaused(UObject* WorldContext) {
  const auto subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (subsys && subsys->GetScene()) {
    return subsys->GetScene()->IsPaused();
  }
  return true;
}

int UMochiPhysicsFunctionLibrary::GetNumMochiStepsSinceStart(UObject* WorldContext) {
  const auto subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (subsys && subsys->GetScene()) {
    return subsys->GetScene()->GetNumMochiStepsSinceStart();
  }
  return 0;
}

double UMochiPhysicsFunctionLibrary::GetTotalSimulatedTimeSinceStartSec(UObject* WorldContext) {
  const auto subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (subsys && subsys->GetScene()) {
    return subsys->GetScene()->GetTotalSimulatedTimeSinceStartSec();
  }
  return 0;
}

void UMochiPhysicsFunctionLibrary::StartRecording(UObject* WorldContext, const FString& filePath) {
  const auto subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (subsys && subsys->GetScene()) {
    subsys->GetScene()->StartRecording(filePath);
  }
}

void UMochiPhysicsFunctionLibrary::StopRecording(UObject* WorldContext) {
  const auto subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (subsys && subsys->GetScene()) {
    subsys->GetScene()->StopRecording();
  }
}

void UMochiPhysicsFunctionLibrary::ExportMochiScene(
    UObject* WorldContext,
    const FString& exportName,
    const FString& filePath) {
  const auto subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (subsys && subsys->GetScene()) {
    subsys->GetScene()->QueueOrExecuteCommand([exportName, filePath](mochi::Scene* Scene) {
      mochi::Error e;
      mochi::prefab::ExportScene(Scene, TCHAR_TO_UTF8(*exportName), TCHAR_TO_UTF8(*filePath), e);
    });
  }
}

UMochiGameInstanceSubsystem* UMochiPhysicsFunctionLibrary::GetMochiGameInstanceSubsystem(
    UObject* WorldContext) {
  const auto game = WorldContext->GetWorld()->GetGameInstance();
  return game->GetSubsystem<UMochiGameInstanceSubsystem>();
}
