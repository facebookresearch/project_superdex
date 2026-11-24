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
      UGameInstance* const Game = World->GetGameInstance();
      UMochiGameInstanceSubsystem* const Subsys = Game->GetSubsystem<UMochiGameInstanceSubsystem>();
      if (Subsys && Subsys->GetScene()) {
        Subsys->GetScene()->SetPaused(true);
      }
    }));

static FAutoConsoleCommandWithWorld GMochiUnpause(
    TEXT("Mochi.Unpause"),
    TEXT("Unpause Mochi simulation."),
    FConsoleCommandWithWorldDelegate::CreateLambda([](UWorld* World) {
      UGameInstance* const Game = World->GetGameInstance();
      UMochiGameInstanceSubsystem* const Subsys = Game->GetSubsystem<UMochiGameInstanceSubsystem>();
      if (Subsys && Subsys->GetScene()) {
        Subsys->GetScene()->SetPaused(false);
      }
    }));

static FAutoConsoleCommandWithWorld GMochiStepPause(
    TEXT("Mochi.StepPause"),
    TEXT("Step the Mochi simulation by exactly one step then pause (or re-pause)."),
    FConsoleCommandWithWorldDelegate::CreateLambda([](UWorld* World) {
      UGameInstance* const Game = World->GetGameInstance();
      UMochiGameInstanceSubsystem* const Subsys = Game->GetSubsystem<UMochiGameInstanceSubsystem>();
      if (Subsys && Subsys->GetScene()) {
        Subsys->GetScene()->StepPause();
      }
    }));

//------------------------------------------------------------------------------
// UMochiPhysicsFunctionLibrary
//------------------------------------------------------------------------------

void UMochiPhysicsFunctionLibrary::SetMochiGravity(UObject* WorldContext, FVector GravityVector) {
  UMochiGameInstanceSubsystem* const Subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (Subsys) {
    if (UMochiSceneObject* const Scene = Subsys->GetScene()) {
      Scene->SetGravity(GravityVector);
    }
  }
}

void UMochiPhysicsFunctionLibrary::SetMochiContactMatrixEntry(
    UObject* WorldContext,
    const FMochiContactMatrixEntry& Entry) {
  UMochiGameInstanceSubsystem* const Subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (Subsys) {
    if (UMochiSceneObject* const Scene = Subsys->GetScene()) {
      Scene->SetContactMatrixEntry(Entry);
    }
  }
}

void UMochiPhysicsFunctionLibrary::DisableContact(
    UObject* WorldContext,
    const UMochiActorObject* ActorA,
    const UMochiActorObject* ActorB) {
  UMochiGameInstanceSubsystem* const Subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (Subsys) {
    UMochiSceneObject* const PhysicsScene = Subsys->GetScene();
    if (PhysicsScene && ActorA && ActorB && ActorA->IsValid() && ActorB->IsValid()) {
      mochi::ActorHandle HandleA = ActorA->GetMochiActorHandle();
      mochi::ActorHandle HandleB = ActorB->GetMochiActorHandle();
      PhysicsScene->QueueOrExecuteCommand([HandleA, HandleB](mochi::Scene* Scene) {
        mochi::ErrorLog E;
        Scene->EnableActorContactSymmetric(
            HandleA, HandleB, /*enable*/ false, mochi::IncludeNestedActors::No, E);
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
  UMochiGameInstanceSubsystem* const Subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (Subsys) {
    UMochiSceneObject* const PhysicsScene = Subsys->GetScene();
    TArray<mochi::ActorHandle> HandlesB = ActorB->GetNestedLinkActorHandles();
    if (Link >= 0) {
      HandlesB = {HandlesB[Link]};
    }
    if (PhysicsScene && ActorA && ActorB && ActorA->IsValid() && ActorB->IsValid()) {
      mochi::ActorHandle HandleA = ActorA->GetMochiActorHandle();
      PhysicsScene->QueueOrExecuteCommand([HandleA, HandlesB](mochi::Scene* Scene) {
        for (const mochi::ActorHandle& HandleB : HandlesB) {
          mochi::ErrorLog E;
          Scene->EnableActorContactSymmetric(
              HandleA, HandleB, /*enable*/ false, mochi::IncludeNestedActors::No, E);
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
  UMochiGameInstanceSubsystem* const Subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (Subsys) {
    Subsys->EnableDebugDraw();
  }
}

void UMochiPhysicsFunctionLibrary::DisableMochiDebugDraw(UObject* WorldContext) {
  UMochiGameInstanceSubsystem* const Subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (Subsys) {
    Subsys->DisableDebugDraw();
  }
}

void UMochiPhysicsFunctionLibrary::ToggleMochiDebugDraw(UObject* WorldContext) {
  UMochiGameInstanceSubsystem* const Subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (Subsys) {
    if (Subsys->IsDebugDrawEnabled()) {
      Subsys->DisableDebugDraw();
    } else {
      Subsys->EnableDebugDraw();
    }
  }
}

bool UMochiPhysicsFunctionLibrary::IsMochiDebugDrawEnabled(UObject* WorldContext) {
  UMochiGameInstanceSubsystem* const Subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (Subsys) {
    return Subsys->IsDebugDrawEnabled();
  }
  return false;
}

void UMochiPhysicsFunctionLibrary::SetPaused(UObject* WorldContext, bool bPaused) {
  UMochiGameInstanceSubsystem* const Subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (Subsys && Subsys->GetScene()) {
    Subsys->GetScene()->SetPaused(bPaused);
  }
}

bool UMochiPhysicsFunctionLibrary::IsPaused(UObject* WorldContext) {
  UMochiGameInstanceSubsystem* const Subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (Subsys && Subsys->GetScene()) {
    return Subsys->GetScene()->IsPaused();
  }
  return true;
}

int UMochiPhysicsFunctionLibrary::GetNumMochiStepsSinceStart(UObject* WorldContext) {
  UMochiGameInstanceSubsystem* const Subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (Subsys && Subsys->GetScene()) {
    return Subsys->GetScene()->GetNumMochiStepsSinceStart();
  }
  return 0;
}

double UMochiPhysicsFunctionLibrary::GetTotalSimulatedTimeSinceStartSec(UObject* WorldContext) {
  UMochiGameInstanceSubsystem* const Subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (Subsys && Subsys->GetScene()) {
    return Subsys->GetScene()->GetTotalSimulatedTimeSinceStartSec();
  }
  return 0;
}

void UMochiPhysicsFunctionLibrary::StartRecording(UObject* WorldContext, const FString& filePath) {
  UMochiGameInstanceSubsystem* const Subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (Subsys && Subsys->GetScene()) {
    Subsys->GetScene()->StartRecording(filePath);
  }
}

void UMochiPhysicsFunctionLibrary::StopRecording(UObject* WorldContext) {
  UMochiGameInstanceSubsystem* const Subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (Subsys && Subsys->GetScene()) {
    Subsys->GetScene()->StopRecording();
  }
}

void UMochiPhysicsFunctionLibrary::ExportMochiScene(
    UObject* WorldContext,
    const FString& exportName,
    const FString& filePath) {
  UMochiGameInstanceSubsystem* const Subsys = GetMochiGameInstanceSubsystem(WorldContext);
  if (Subsys && Subsys->GetScene()) {
    Subsys->GetScene()->QueueOrExecuteCommand([exportName, filePath](mochi::Scene* Scene) {
      mochi::Error E;
      mochi::prefab::ExportScene(Scene, TCHAR_TO_UTF8(*exportName), TCHAR_TO_UTF8(*filePath), E);
    });
  }
}

UMochiGameInstanceSubsystem* UMochiPhysicsFunctionLibrary::GetMochiGameInstanceSubsystem(
    UObject* WorldContext) {
  UGameInstance* const Game = WorldContext->GetWorld()->GetGameInstance();
  return Game->GetSubsystem<UMochiGameInstanceSubsystem>();
}
