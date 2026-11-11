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

#include "MochiArticulationUtil.h"

#include <superdex_robotics/scene/bot_scene.h>

#include "SuperDexBotController.generated.h"

class USuperDexBotComponent;

UCLASS(Abstract, BlueprintType, DefaultToInstanced)
class SUPERDEXROBOTICS_API USuperDexBotController : public UObject {
  GENERATED_BODY()

 public:
  /* @brief Return the most recent composed target pose for recorder visualization. */
  virtual bool GetRecordingTarget(
      int32& OutAnchorLinkIndex,
      mochi::TransformRT& OutWorldFromAnchor,
      TConstArrayView<mochi::real>& OutJointPositions) const;

 protected:
  friend class USuperDexBotComponent;
  // Initialize the controller. Called on the game thread in BeginPlay.
  // Return true on success, false to cancel controller deployment.
  virtual bool Initialize(const USuperDexBotComponent* Component);
  // Called when the bot has been spawned. Game thread.
  virtual void OnBotSpawned(USuperDexBotComponent* Component);
  // Shutdown the controller. Called on the game thread in EndPlay or when swapped.
  virtual void Shutdown(const USuperDexBotComponent* Component);
  // Step the controller on the Mochi simulation thread in a pre-step callback.
  // MochiDeltaTime is the duration of the step about to be taken.
  virtual bool StepController(
      const USuperDexBotComponent* Component,
      const FMochiArticulationState& State,
      double MochiDeltaTime,
      mochi::Actor* ArticulatedActor,
      mochi::Scene* Scene);
  // Tick the controller on the game thread. Called in TickComponent and is thread safe with
  // StepController.
  virtual void TickController(const USuperDexBotComponent* Component, double GameDeltaTime);

 public:
  virtual TArray<superdex::robotics::ControllerEntry> ExportControllerEntries(
      const FString& OutputDir) const;
};
