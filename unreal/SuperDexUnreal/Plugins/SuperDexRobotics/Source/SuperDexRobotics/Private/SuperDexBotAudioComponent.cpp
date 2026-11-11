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

#include "SuperDexBotAudioComponent.h"

#include "MochiPhysicsModule.h"
#include "SuperDexBot.h"
#include "SuperDexBotComponent.h"

#include "Components/AudioComponent.h"
#include "Curves/CurveFloat.h"
#include "DrawDebugHelpers.h"
#include "GameFramework/Actor.h"
#include "Sound/SoundBase.h"

#if WITH_EDITOR
#include "Engine/BlueprintGeneratedClass.h"
#include "Engine/SCS_Node.h"
#include "Engine/SimpleConstructionScript.h"
#endif

namespace {

UAudioComponent*
SpawnAudioAtBone(USkinnedMeshComponent* SkinnedMeshComponent, USoundBase* Sound, FName BoneName) {
  if (SkinnedMeshComponent == nullptr || Sound == nullptr) {
    return nullptr;
  }
  UAudioComponent* audioComp = NewObject<UAudioComponent>(SkinnedMeshComponent);
  if (audioComp) {
    audioComp->RegisterComponent();
    audioComp->SetSound(Sound);
    audioComp->SetPitchMultiplier(1.0f);
    audioComp->bAutoDestroy = false;
    audioComp->AttachToComponent(
        SkinnedMeshComponent, FAttachmentTransformRules::SnapToTargetIncludingScale, BoneName);
    audioComp->SetRelativeLocation(FVector::ZeroVector);
    audioComp->SetRelativeRotation(FRotator::ZeroRotator);
    audioComp->Play();
    audioComp->SetVolumeMultiplier(0);
  }
  return audioComp;
}

#if WITH_EDITOR
// The bot component beside Component: on its owner when placed, or, for a component template being
// edited in a Blueprint, among the Blueprint's (and its parents') components.
const USuperDexBotComponent* FindBotComponent(const UActorComponent* Component) {
  if (const AActor* Owner = Component->GetOwner()) {
    return Owner->FindComponentByClass<USuperDexBotComponent>();
  }
  for (const UBlueprintGeneratedClass* Class =
           Cast<UBlueprintGeneratedClass>(Component->GetOuter());
       Class != nullptr;
       Class = Cast<UBlueprintGeneratedClass>(Class->GetSuperClass())) {
    if (const USimpleConstructionScript* Construction = Class->SimpleConstructionScript) {
      for (const USCS_Node* Node : Construction->GetAllNodes()) {
        if (const auto* Bot = Cast<USuperDexBotComponent>(Node->ComponentTemplate)) {
          return Bot;
        }
      }
    }
    if (const AActor* Defaults = Cast<AActor>(Class->GetDefaultObject())) {
      if (const auto* Bot = Defaults->FindComponentByClass<USuperDexBotComponent>()) {
        return Bot;
      }
    }
  }
  return nullptr;
}
#endif

} // namespace

USuperDexBotAudioComponent::USuperDexBotAudioComponent() {
  PrimaryComponentTick.bCanEverTick = true;
}

void USuperDexBotAudioComponent::BeginPlay() {
  Super::BeginPlay();
  BotComponent = GetOwner()->FindComponentByClass<USuperDexBotComponent>();
  if (BotComponent == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[USuperDexBotAudioComponent] %s has no USuperDexBotComponent!"),
        *GetOwner()->GetName());
    return;
  }
  // The bot spawns on its component's first tick, which may come after this BeginPlay.
  if (BotComponent->GetMochiArticulatedActor() != nullptr) {
    SetUpAudio();
  } else {
    BotComponent->OnBotSpawned.AddDynamic(this, &USuperDexBotAudioComponent::HandleBotSpawned);
  }
}

void USuperDexBotAudioComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  if (BotComponent != nullptr) {
    BotComponent->OnBotSpawned.RemoveDynamic(this, &USuperDexBotAudioComponent::HandleBotSpawned);
  }
  Super::EndPlay(EndPlayReason);
}

void USuperDexBotAudioComponent::HandleBotSpawned(USuperDexBotComponent* InBotComponent) {
  if (InBotComponent == BotComponent && !bAudioSetUp) {
    SetUpAudio();
  }
}

void USuperDexBotAudioComponent::SetUpAudio() {
  const USuperDexBot* const Bot = BotComponent->GetBotAsset();
  if (Bot == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[USuperDexBotAudioComponent] The USuperDexBotComponent of %s has no bot asset!"),
        *GetOwner()->GetName());
    return;
  }
  bAudioSetUp = true;
  // set up joint velocity estimators
  const int numDofs = Bot->GetNumDofs();
  JointPositionsPrevious.SetNumZeroed(numDofs);
  EstimatedJointVelocities.SetNumZeroed(numDofs);
  OneEuroFilters.Init(TSuperDexOneEuroFilter<double>(3.0, 1.0, 1.0), numDofs);
  // spawn audio instances at the links the joints move
  for (const auto& params : BotAudioParams) {
    if (!params.AudioLoop || !params.VelocityVolumeCurve || !params.VelocityPitchCurve) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[USuperDexBotAudioComponent] Encountered incomplete bot audio params entry!"));
      continue;
    }

    FSuperDexBotAudioInstance audioInstance;
    audioInstance.VelocityVolumeCurve = params.VelocityVolumeCurve;
    audioInstance.VelocityPitchCurve = params.VelocityPitchCurve;
    audioInstance.JointAggregateMode = params.JointAggregateMode;
    audioInstance.bEstimateVelocity = params.bEstimateVelocity;
    for (const auto& jName : params.Joints) {
      const int iDof = Bot->GetDofIndex(jName);
      if (iDof == INDEX_NONE) {
        UE_LOG(
            LogMochiPhysics,
            Error,
            TEXT(
                "[USuperDexBotAudioComponent] Joint [%s] does not exist or is not a degree of freedom!"),
            *jName.ToString());
        continue;
      }
      if (params.JointAggregateMode == ESuperDexBotAudioJointAggregateMode::Individual) {
        FSuperDexBotAudioInstance jointInstance = audioInstance;
        jointInstance.DofIndices.Add(iDof);
        jointInstance.AudioComponent =
            SpawnAudioAtBone(BotComponent, params.AudioLoop, Bot->GetDofLinkName(iDof));
        JointAudioInstances.Add(jointInstance);
      } else {
        if (audioInstance.DofIndices.IsEmpty()) {
          audioInstance.AudioComponent =
              SpawnAudioAtBone(BotComponent, params.AudioLoop, Bot->GetDofLinkName(iDof));
        }
        audioInstance.DofIndices.Add(iDof);
      }
    }
    if (params.JointAggregateMode == ESuperDexBotAudioJointAggregateMode::Sum &&
        !audioInstance.DofIndices.IsEmpty()) {
      JointAudioInstances.Add(audioInstance);
    }
  }
}

void USuperDexBotAudioComponent::TickComponent(
    float DeltaTime,
    ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction) {
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
  if (!bAudioSetUp || BotComponent == nullptr || DeltaTime <= 0.0f) {
    return;
  }
  const TArray<double>& JointVelocities = BotComponent->GetCurrentJointVelocities();
  const TArray<double>& JointPositions = BotComponent->GetCurrentJointPositions();
  // Nothing to play until the bot has been simulated.
  if (JointPositions.Num() != OneEuroFilters.Num() ||
      JointVelocities.Num() != OneEuroFilters.Num()) {
    return;
  }
  if (bFirstTick) {
    JointPositionsPrevious = JointPositions;
    bFirstTick = false;
  }
  // estimate velocities from positions
  for (int i = 0; i < JointPositions.Num(); ++i) {
    EstimatedJointVelocities[i] = (JointPositions[i] - JointPositionsPrevious[i]) / DeltaTime;
  }
  JointPositionsPrevious = JointPositions;
  for (auto& instance : JointAudioInstances) {
    if (instance.AudioComponent == nullptr) {
      continue;
    }
    float velocity = 0.0f;
    for (const int dofIndex : instance.DofIndices) {
      velocity += FMath::Abs(
          OneEuroFilters[dofIndex].Filter(
              instance.bEstimateVelocity ? EstimatedJointVelocities[dofIndex]
                                         : JointVelocities[dofIndex],
              DeltaTime));
    }
    float volume = instance.VelocityVolumeCurve->GetFloatValue(velocity);
    float pitch = instance.VelocityPitchCurve->GetFloatValue(velocity);
    instance.AudioComponent->SetVolumeMultiplier(volume);
    instance.AudioComponent->SetPitchMultiplier(pitch);
    if (bDebugDraw) {
      float radius = volume * 10;
      DrawDebugSphere(
          GetWorld(),
          instance.AudioComponent->GetComponentLocation(),
          radius,
          16,
          FColor::Blue,
          false,
          -1,
          SDPG_Foreground);
    }
  }
}

#if WITH_EDITOR
TArray<FName> USuperDexBotAudioComponent::GetJointNames_EditorOnly() const {
  const USuperDexBotComponent* const Bot = FindBotComponent(this);
  const USuperDexBot* const BotAsset = Bot != nullptr ? Bot->GetBotAsset() : nullptr;
  return BotAsset != nullptr ? BotAsset->GetJointNames() : TArray<FName>{};
}
#endif
