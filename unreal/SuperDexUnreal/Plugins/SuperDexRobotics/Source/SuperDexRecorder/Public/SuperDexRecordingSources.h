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
#include "Engine/EngineTypes.h"
#include "SuperDexRecordingSourceComponent.h"

#include "SuperDexRecordingSources.generated.h"

/*
 * Records the first local player's camera, which follows the headset in VR, under
 * "world/sources/<SourceId>" (default "head").
 */
UCLASS(Blueprintable, ClassGroup = SuperDex, meta = (BlueprintSpawnableComponent))
class SUPERDEXRECORDER_API USuperDexRecordingPlayerCameraSourceComponent
    : public USuperDexRecordingSourceComponent {
  GENERATED_BODY()

 public:
  USuperDexRecordingPlayerCameraSourceComponent();

  void ConfigureRecording(FSuperDexSceneRecorderConfig& Config) override;
  void OnRecordingEnded() override;
  void TickComponent(
      float DeltaTime,
      ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;
  void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

 private:
  FSuperDexRecordingCachedTransformPtr Cached =
      MakeShared<FSuperDexRecordingCachedTransform, ESPMode::ThreadSafe>();
};

USTRUCT(BlueprintType)
struct SUPERDEXRECORDER_API FSuperDexRecordingTransformSourceEntry {
  GENERATED_BODY()

  /* Names this component's stream "world/sources/<SourceId>.<Id>". Empty uses
   * "<actor>.<component>". */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex|Recording")
  FString Id;

  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "SuperDex|Recording",
      meta = (UseComponentPicker, AllowedClasses = "/Script/Engine.SceneComponent"))
  FComponentReference Component;
};

/*
 * Records the world transforms of scene components, such as objects that have no Mochi physics
 * (Mochi bodies are already recorded by the physics recorder), under
 * "world/sources/<SourceId>.<Id>".
 */
UCLASS(Blueprintable, ClassGroup = SuperDex, meta = (BlueprintSpawnableComponent))
class SUPERDEXRECORDER_API USuperDexRecordingTransformSourceComponent
    : public USuperDexRecordingSourceComponent {
  GENERATED_BODY()

 public:
  USuperDexRecordingTransformSourceComponent();

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex|Recording")
  TArray<FSuperDexRecordingTransformSourceEntry> Components;

  void ConfigureRecording(FSuperDexSceneRecorderConfig& Config) override;
  void OnRecordingEnded() override;
  void TickComponent(
      float DeltaTime,
      ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;
  void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

 private:
  struct FRecordedComponent {
    TWeakObjectPtr<USceneComponent> Component;
    FSuperDexRecordingCachedTransformPtr Cached;
  };

  /* The components of the episode being recorded, cached each tick. */
  TArray<FRecordedComponent> Recorded;
};
