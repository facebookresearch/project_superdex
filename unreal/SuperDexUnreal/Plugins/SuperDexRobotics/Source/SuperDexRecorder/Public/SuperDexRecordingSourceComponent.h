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

#include "Components/ActorComponent.h"
#include "CoreMinimal.h"
#include "HAL/CriticalSection.h"

#include "SuperDexRecordingSourceComponent.generated.h"

struct FSuperDexSceneRecorderConfig;

/*
 * The latest Unreal world transform of a source, written on the game thread and read on the Mochi
 * thread when the recorder samples it.
 */
class SUPERDEXRECORDER_API FSuperDexRecordingCachedTransform {
 public:
  void Set(const FTransform& WorldTransform);
  /* Marks the source untracked until the next Set. */
  void Clear();
  bool Get(FTransform& OutWorldTransform) const;

 private:
  mutable FCriticalSection Lock;
  FTransform Transform;
  bool bIsValid = false;
};

using FSuperDexRecordingCachedTransformPtr =
    TSharedPtr<FSuperDexRecordingCachedTransform, ESPMode::ThreadSafe>;

/*
 * Base class of a component that adds streams to an ASuperDexRecorder's episodes. Place it on the
 * recorder, or on an actor listed in the recorder's SourceActors. It ticks only while an episode is
 * recording.
 */
UCLASS(Abstract, ClassGroup = SuperDex)
class SUPERDEXRECORDER_API USuperDexRecordingSourceComponent : public UActorComponent {
  GENERATED_BODY()

 public:
  USuperDexRecordingSourceComponent();

  /* Identifies this source's streams in the recording. Empty uses the component's name. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex|Recording")
  FString SourceId;

  /* Called on the game thread when an episode begins, to add this source's streams. */
  virtual void ConfigureRecording(FSuperDexSceneRecorderConfig& Config) {}
  /* Called on the game thread once the episode has ended (committed, aborted or failed). */
  virtual void OnRecordingEnded() {}

 protected:
  FString GetResolvedSourceId() const;
};
