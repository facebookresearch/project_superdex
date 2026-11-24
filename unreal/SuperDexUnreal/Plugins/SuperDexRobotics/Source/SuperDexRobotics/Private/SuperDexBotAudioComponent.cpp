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
  UAudioComponent* AudioComp = NewObject<UAudioComponent>(SkinnedMeshComponent);
  if (AudioComp) {
    AudioComp->RegisterComponent();
    AudioComp->SetSound(Sound);
    AudioComp->SetPitchMultiplier(1.0f);
    AudioComp->bAutoDestroy = false;
    AudioComp->AttachToComponent(
        SkinnedMeshComponent, FAttachmentTransformRules::SnapToTargetIncludingScale, BoneName);
    AudioComp->SetRelativeLocation(FVector::ZeroVector);
    AudioComp->SetRelativeRotation(FRotator::ZeroRotator);
    AudioComp->Play();
    AudioComp->SetVolumeMultiplier(0);
  }
  return AudioComp;
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
        if (const USuperDexBotComponent* Bot =
                Cast<USuperDexBotComponent>(Node->ComponentTemplate)) {
          return Bot;
        }
      }
    }
    if (const AActor* Defaults = Cast<AActor>(Class->GetDefaultObject())) {
      if (const USuperDexBotComponent* Bot =
              Defaults->FindComponentByClass<USuperDexBotComponent>()) {
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
  const int NumDofs = Bot->GetNumDofs();
  JointPositionsPrevious.SetNumZeroed(NumDofs);
  EstimatedJointVelocities.SetNumZeroed(NumDofs);
  OneEuroFilters.Init(TSuperDexOneEuroFilter<double>(3.0, 1.0, 1.0), NumDofs);
  // spawn audio instances at the links the joints move
  for (const FSuperDexBotAudioParams& Params : BotAudioParams) {
    if (!Params.AudioLoop || !Params.VelocityVolumeCurve || !Params.VelocityPitchCurve) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[USuperDexBotAudioComponent] Encountered incomplete bot audio params entry!"));
      continue;
    }

    FSuperDexBotAudioInstance AudioInstance;
    AudioInstance.VelocityVolumeCurve = Params.VelocityVolumeCurve;
    AudioInstance.VelocityPitchCurve = Params.VelocityPitchCurve;
    AudioInstance.JointAggregateMode = Params.JointAggregateMode;
    AudioInstance.bEstimateVelocity = Params.bEstimateVelocity;
    for (const FName& JName : Params.Joints) {
      const int DofIndex = Bot->GetDofIndex(JName);
      if (DofIndex == INDEX_NONE) {
        UE_LOG(
            LogMochiPhysics,
            Error,
            TEXT(
                "[USuperDexBotAudioComponent] Joint [%s] does not exist or is not a degree of freedom!"),
            *JName.ToString());
        continue;
      }
      if (Params.JointAggregateMode == ESuperDexBotAudioJointAggregateMode::Individual) {
        FSuperDexBotAudioInstance JointInstance = AudioInstance;
        JointInstance.DofIndices.Add(DofIndex);
        JointInstance.AudioComponent =
            SpawnAudioAtBone(BotComponent, Params.AudioLoop, Bot->GetDofLinkName(DofIndex));
        JointAudioInstances.Add(JointInstance);
      } else {
        if (AudioInstance.DofIndices.IsEmpty()) {
          AudioInstance.AudioComponent =
              SpawnAudioAtBone(BotComponent, Params.AudioLoop, Bot->GetDofLinkName(DofIndex));
        }
        AudioInstance.DofIndices.Add(DofIndex);
      }
    }
    if (Params.JointAggregateMode == ESuperDexBotAudioJointAggregateMode::Sum &&
        !AudioInstance.DofIndices.IsEmpty()) {
      JointAudioInstances.Add(AudioInstance);
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
  for (FSuperDexBotAudioInstance& Instance : JointAudioInstances) {
    if (Instance.AudioComponent == nullptr) {
      continue;
    }
    float Velocity = 0.0f;
    for (const int DofIndex : Instance.DofIndices) {
      Velocity += FMath::Abs(
          OneEuroFilters[DofIndex].Filter(
              Instance.bEstimateVelocity ? EstimatedJointVelocities[DofIndex]
                                         : JointVelocities[DofIndex],
              DeltaTime));
    }
    float Volume = Instance.VelocityVolumeCurve->GetFloatValue(Velocity);
    float Pitch = Instance.VelocityPitchCurve->GetFloatValue(Velocity);
    Instance.AudioComponent->SetVolumeMultiplier(Volume);
    Instance.AudioComponent->SetPitchMultiplier(Pitch);
    if (bDebugDraw) {
      float Radius = Volume * 10;
      DrawDebugSphere(
          GetWorld(),
          Instance.AudioComponent->GetComponentLocation(),
          Radius,
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
