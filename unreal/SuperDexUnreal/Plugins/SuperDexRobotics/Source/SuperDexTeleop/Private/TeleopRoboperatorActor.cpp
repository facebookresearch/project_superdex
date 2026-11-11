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

#include "TeleopRoboperatorActor.h"

#include "Controllers/Controller_MochiArticulatedPose.h"
#include "Retargeters/ClutchSkeletonRootToVrRootRetargeter.h"
#include "SuperDexBotComponent.h"
#include "SuperDexPoseableComponent.h"
#include "SuperDexTeleopModule.h"
#include "TeleopNode.h"

ATeleopRoboperatorActor::ATeleopRoboperatorActor() {
  PrimaryActorTick.bCanEverTick = true;

  RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));

  TeleopNodeOperator =
      CreateDefaultSubobject<UTeleopNodeOperatorComponent>(TEXT("TeleopNodeOperator"));
  TeleopNodeRobot = CreateDefaultSubobject<UTeleopNodeRobotComponent>(TEXT("TeleopNodeRobot"));
}

void ATeleopRoboperatorActor::SetClutchEnabled_Implementation(bool bClutchEnabled) {
  TArray<UClutchSkeletonRootToVrRootRetargeter*> Clutches;
  GetComponents<UClutchSkeletonRootToVrRootRetargeter>(Clutches);
  for (UClutchSkeletonRootToVrRootRetargeter* Clutch : Clutches) {
    // Re-initialize clutch origin before enabling so the offset resets cleanly
    if (bClutchEnabled) {
      Clutch->InitializeClutchBasedOnLeafBone(Clutch->InitializeClutchFromPosableWithName, false);
    }
    Clutch->SetClutchEnabled(bClutchEnabled);
  }
}

void ATeleopRoboperatorActor::PostInitializeComponents() {
  Super::PostInitializeComponents();
  if (!bSpawnBotsAtTrackedHand) {
    return;
  }
  // Before Begin Play, so no bot is created at the actor before its hand is seen.
  TArray<USuperDexBotComponent*> Bots;
  GetComponents<USuperDexBotComponent>(Bots);
  for (USuperDexBotComponent* Bot : Bots) {
    Bot->SetAutoSpawn(false);
    PendingBotSpawns.Add({Bot});
  }
}

void ATeleopRoboperatorActor::Tick(float DeltaSeconds) {
  Super::Tick(DeltaSeconds);
  if (!PendingBotSpawns.IsEmpty() && GetWorld() != nullptr && GetWorld()->IsGameWorld()) {
    PendingBotSpawns.RemoveAll(
        [&](FPendingBotSpawn& Pending) { return UpdatePendingBotSpawn(Pending, DeltaSeconds); });
  }
}

double ATeleopRoboperatorActor::AdvanceSpawnSettle(
    FSpawnSettle& Settle,
    double HandScale,
    double DeltaSeconds,
    double SettleSeconds,
    double ScaleTolerance,
    double MaxWaitSeconds) {
  Settle.TrackedSeconds += DeltaSeconds;
  const bool bTimedOut = Settle.TrackedSeconds >= MaxWaitSeconds;
  if (!FMath::IsFinite(HandScale) || HandScale <= UE_KINDA_SMALL_NUMBER) {
    Settle.SteadySeconds = 0.0;
    Settle.ReferenceScale = 0.0;
    return bTimedOut ? 1.0 : 0.0;
  }
  if (Settle.ReferenceScale <= 0.0 ||
      FMath::Abs(HandScale - Settle.ReferenceScale) > ScaleTolerance * Settle.ReferenceScale) {
    Settle.ReferenceScale = HandScale;
    Settle.SteadySeconds = 0.0;
  } else {
    Settle.SteadySeconds += DeltaSeconds;
  }
  return Settle.SteadySeconds >= SettleSeconds || bTimedOut ? HandScale : 0.0;
}

bool ATeleopRoboperatorActor::UpdatePendingBotSpawn(FPendingBotSpawn& Pending, float DeltaSeconds) {
  USuperDexBotComponent* Bot = Pending.Bot.Get();
  if (Bot == nullptr || Bot->IsBotSpawned()) {
    return true;
  }
  if (!Pending.Source.IsValid()) {
    USuperDexPoseableComponent* Source = nullptr;
    if (const auto* PoseController = Cast<UController_MochiArticulatedPose>(Bot->GetController())) {
      Source = PoseController->FindChildSourcePoseable(Bot);
    }
    if (Source == nullptr) {
      for (USceneComponent* Child : Bot->GetAttachChildren()) {
        if ((Source = Cast<USuperDexPoseableComponent>(Child)) != nullptr) {
          break;
        }
      }
    }
    if (Source == nullptr) {
      // The bot's own auto-spawn creates it at the component's transform and scale, retrying until
      // it succeeds.
      UE_LOG(
          LogSuperDexTeleop,
          Error,
          TEXT("[%hs] %s: bot '%s' has no tracked hand to spawn at; spawning it at the actor."),
          __FUNCTION__,
          *GetName(),
          *Bot->GetName());
      Bot->SetAutoSpawn(true);
      return true;
    }
    Pending.Source = Source;
  }

  FSuperDexPose Pose;
  if (!Pending.Source->GetCachedPose(Pose) || !Pose.bIsTrackingEnabled ||
      !Pose.bIsTrackingConfident) {
    Pending.Settle.SteadySeconds = 0.0;
    return false;
  }
  // A mirrored hand reports a negative scale; its size is the magnitude.
  const double HandScale =
      bScaleBotsToTrackedHand ? FMath::Abs(Pose.SkeletonRootToVrRoot.GetScale3D().X) : 1.0;
  const double Scale = AdvanceSpawnSettle(
      Pending.Settle,
      HandScale,
      DeltaSeconds,
      SpawnSettleSeconds,
      SpawnScaleTolerance,
      SpawnMaxWaitSeconds);
  if (Scale <= 0.0) {
    return false;
  }
  if (Scale != HandScale && !Pending.bReportedFailure) {
    UE_LOG(
        LogSuperDexTeleop,
        Warning,
        TEXT(
            "[%hs] %s: the tracked hand of bot '%s' reports the invalid scale %f; spawning it at nominal size."),
        __FUNCTION__,
        *GetName(),
        *Bot->GetName(),
        HandScale);
  }

  // The same root the bot's pose controller targets: the source pose's root, relative to the
  // source's actor without its scale. (The root's own scale does not reach the composition's
  // location and rotation.)
  FTransform ActorToWorld = Pending.Source->GetOwner()->GetActorTransform();
  ActorToWorld.SetScale3D(FVector::OneVector);
  const FTransform RootToWorld = Pose.SkeletonRootToVrRoot * ActorToWorld;
  if (Bot->SpawnBot(RootToWorld, static_cast<float>(Scale))) {
    UE_LOG(
        LogSuperDexTeleop,
        Log,
        TEXT("[%hs] %s: spawned bot '%s' at the tracked hand with scale %.3f after %.2f s"),
        __FUNCTION__,
        *GetName(),
        *Bot->GetName(),
        Scale,
        Pending.Settle.TrackedSeconds);
    return true;
  }
  if (!Pending.bReportedFailure) {
    UE_LOG(
        LogSuperDexTeleop,
        Warning,
        TEXT("[%hs] %s: failed to spawn bot '%s' at the tracked hand; retrying every tick."),
        __FUNCTION__,
        *GetName(),
        *Bot->GetName());
    Pending.bReportedFailure = true;
  }
  return false;
}
