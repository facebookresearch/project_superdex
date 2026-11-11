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
#include "SuperDexOneEuroFilter.h"

#include "SuperDexBotAudioComponent.generated.h"

class UAudioComponent;
class UCurveFloat;
class USoundBase;
class USuperDexBotComponent;

UENUM()
enum class ESuperDexBotAudioJointAggregateMode {
  // Each joint plays its own loop, driven by its own velocity.
  Individual,
  // One loop, at the first joint, driven by the sum of the joints' speeds.
  Sum
};

USTRUCT()
struct FSuperDexBotAudioParams {
  GENERATED_BODY()
  // The joints these params are applied to.
  UPROPERTY(EditAnywhere, meta = (GetOptions = "GetJointNames_EditorOnly"), Category = "SuperDex Bot Audio")
  TArray<FName> Joints;
  // The audio clip to use. Should be seamless loop with constant vol/pitch/tone.
  UPROPERTY(EditAnywhere, Category = "SuperDex Bot Audio")
  USoundBase* AudioLoop = nullptr;
  // Maps joint speed to volume.
  UPROPERTY(EditAnywhere, Category = "SuperDex Bot Audio")
  UCurveFloat* VelocityVolumeCurve = nullptr;
  // Maps joint speed to pitch.
  UPROPERTY(EditAnywhere, Category = "SuperDex Bot Audio")
  UCurveFloat* VelocityPitchCurve = nullptr;
  // Should each joint have an AudioComponent attached or should all joint velocities
  // be aggregated
  UPROPERTY(EditAnywhere, Category = "SuperDex Bot Audio")
  ESuperDexBotAudioJointAggregateMode JointAggregateMode =
      ESuperDexBotAudioJointAggregateMode::Individual;
  // Estimate joint velocity from joint positions instead of using Mochi's joint velocity.
  UPROPERTY(EditAnywhere, Category = "SuperDex Bot Audio")
  bool bEstimateVelocity = false;
};

USTRUCT()
struct FSuperDexBotAudioInstance {
  GENERATED_BODY()
  UPROPERTY()
  ESuperDexBotAudioJointAggregateMode JointAggregateMode =
      ESuperDexBotAudioJointAggregateMode::Individual;
  UPROPERTY()
  TArray<int> DofIndices;
  UPROPERTY()
  UAudioComponent* AudioComponent = nullptr;
  UPROPERTY()
  UCurveFloat* VelocityVolumeCurve = nullptr;
  UPROPERTY()
  UCurveFloat* VelocityPitchCurve = nullptr;
  UPROPERTY()
  bool bEstimateVelocity = false;
};

/*
 * Plays looping sounds at a SuperDex bot's joints, with volume and pitch following each joint's
 * speed. Add it to an actor with a USuperDexBotComponent; the sounds attach to the links the joints
 * move. Sounds and curves are assigned per entry of BotAudioParams.
 */
UCLASS(ClassGroup = (SuperDex), meta = (BlueprintSpawnableComponent))
class SUPERDEXROBOTICS_API USuperDexBotAudioComponent : public UActorComponent {
  GENERATED_BODY()
 protected:
  // Parameters controlling the bot's audio.
  UPROPERTY(EditAnywhere, Category = "SuperDex Bot Audio")
  TArray<FSuperDexBotAudioParams> BotAudioParams;
  // Draw a sphere at each sound, sized by its volume.
  UPROPERTY(EditAnywhere, Category = "SuperDex Bot Audio")
  bool bDebugDraw = false;

#if WITH_EDITOR
  UFUNCTION(Category = "SuperDex Bot Audio")
  TArray<FName> GetJointNames_EditorOnly() const;
#endif

 public:
  USuperDexBotAudioComponent();

 protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

 public:
  virtual void TickComponent(
      float DeltaTime,
      ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;

 private:
  UFUNCTION()
  void HandleBotSpawned(USuperDexBotComponent* InBotComponent);
  // Spawns the sounds once the bot exists.
  void SetUpAudio();

  UPROPERTY(Transient)
  TObjectPtr<USuperDexBotComponent> BotComponent;
  bool bAudioSetUp = false;
  bool bFirstTick = true;
  UPROPERTY(Transient)
  TArray<FSuperDexBotAudioInstance> JointAudioInstances;
  TArray<double> JointPositionsPrevious;
  TArray<double> EstimatedJointVelocities;
  TArray<TSuperDexOneEuroFilter<double>> OneEuroFilters;
};
