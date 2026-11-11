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

#include "SuperDexRecordingSources.h"

#include "MochiConversions.h"
#include "SuperDexRecorderModule.h"
#include "SuperDexSceneRecorder.h"

#include "Camera/PlayerCameraManager.h"
#include "Components/SceneComponent.h"
#include "Engine/World.h"
#include "GameFramework/PlayerController.h"

namespace {

// Returns a sampler, run on the Mochi thread, that converts the cached Unreal transform to Mochi.
TFunction<bool(mochi::TransformRT&)> MakeTransformSampler(
    FSuperDexRecordingCachedTransformPtr Cached) {
  return [Cached = MoveTemp(Cached)](mochi::TransformRT& OutWorldTransform) {
    FTransform WorldTransform;
    if (!Cached->Get(WorldTransform) || WorldTransform.ContainsNaN()) {
      return false;
    }
    OutWorldTransform = FUnrealToMochiConverter{}.TransformToMochi_RT(WorldTransform);
    return true;
  };
}

bool GetFirstPlayerCameraTransform(const UWorld* World, FTransform& OutWorldTransform) {
  const APlayerController* const Controller =
      World != nullptr ? World->GetFirstPlayerController() : nullptr;
  const APlayerCameraManager* const Camera =
      Controller != nullptr ? Controller->PlayerCameraManager.Get() : nullptr;
  if (Camera == nullptr) {
    return false;
  }
  OutWorldTransform = FTransform(Camera->GetCameraRotation(), Camera->GetCameraLocation());
  return true;
}

void CacheTransform(
    FSuperDexRecordingCachedTransform& Cached,
    bool bIsValid,
    const FTransform& Transform) {
  if (bIsValid) {
    Cached.Set(Transform);
  } else {
    Cached.Clear();
  }
}

} // namespace

USuperDexRecordingPlayerCameraSourceComponent::USuperDexRecordingPlayerCameraSourceComponent() {
  SourceId = TEXT("head");
  PrimaryComponentTick.bCanEverTick = true;
  // After the camera follows the headset for this frame.
  PrimaryComponentTick.TickGroup = TG_PostUpdateWork;
}

void USuperDexRecordingPlayerCameraSourceComponent::ConfigureRecording(
    FSuperDexSceneRecorderConfig& Config) {
  FTransform WorldTransform;
  const bool bIsValid = GetFirstPlayerCameraTransform(GetWorld(), WorldTransform);
  CacheTransform(*Cached, bIsValid, WorldTransform);
  Config.TransformSources.Add({GetResolvedSourceId(), MakeTransformSampler(Cached)});
}

void USuperDexRecordingPlayerCameraSourceComponent::OnRecordingEnded() {
  Cached->Clear();
}

void USuperDexRecordingPlayerCameraSourceComponent::TickComponent(
    float DeltaTime,
    ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction) {
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
  FTransform WorldTransform;
  const bool bIsValid = GetFirstPlayerCameraTransform(GetWorld(), WorldTransform);
  CacheTransform(*Cached, bIsValid, WorldTransform);
}

void USuperDexRecordingPlayerCameraSourceComponent::EndPlay(
    const EEndPlayReason::Type EndPlayReason) {
  Cached->Clear();
  Super::EndPlay(EndPlayReason);
}

USuperDexRecordingTransformSourceComponent::USuperDexRecordingTransformSourceComponent() {
  SourceId = TEXT("objects");
  PrimaryComponentTick.bCanEverTick = true;
  PrimaryComponentTick.TickGroup = TG_PostUpdateWork;
}

void USuperDexRecordingTransformSourceComponent::ConfigureRecording(
    FSuperDexSceneRecorderConfig& Config) {
  Recorded.Reset();
  const FString Prefix = GetResolvedSourceId();
  for (const FSuperDexRecordingTransformSourceEntry& Entry : Components) {
    USceneComponent* const Component =
        Cast<USceneComponent>(Entry.Component.GetComponent(GetOwner()));
    if (Component == nullptr) {
      UE_LOG(
          LogSuperDexRecorder,
          Warning,
          TEXT("%s: not recording '%s', whose component wasn't found."),
          *GetName(),
          *Entry.Id);
      continue;
    }
    const FString Id = Entry.Id.IsEmpty()
        ? FString::Printf(TEXT("%s.%s"), *Component->GetOwner()->GetName(), *Component->GetName())
        : Entry.Id;
    FRecordedComponent& Item = Recorded.Add_GetRef(
        {Component, MakeShared<FSuperDexRecordingCachedTransform, ESPMode::ThreadSafe>()});
    Item.Cached->Set(Component->GetComponentTransform());
    Config.TransformSources.Add({Prefix + TEXT(".") + Id, MakeTransformSampler(Item.Cached)});
  }
}

void USuperDexRecordingTransformSourceComponent::OnRecordingEnded() {
  for (const FRecordedComponent& Item : Recorded) {
    Item.Cached->Clear();
  }
  Recorded.Reset();
}

void USuperDexRecordingTransformSourceComponent::TickComponent(
    float DeltaTime,
    ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction) {
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
  for (const FRecordedComponent& Item : Recorded) {
    const USceneComponent* const Component = Item.Component.Get();
    CacheTransform(
        *Item.Cached,
        Component != nullptr,
        Component != nullptr ? Component->GetComponentTransform() : FTransform::Identity);
  }
}

void USuperDexRecordingTransformSourceComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  for (const FRecordedComponent& Item : Recorded) {
    Item.Cached->Clear();
  }
  Recorded.Reset();
  Super::EndPlay(EndPlayReason);
}
