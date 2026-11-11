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

#include "SuperDexBotController.h"

bool USuperDexBotController::GetRecordingTarget(
    int32& OutAnchorLinkIndex,
    mochi::TransformRT& OutWorldFromAnchor,
    TConstArrayView<mochi::real>& OutJointPositions) const {
  return false;
}

bool USuperDexBotController::Initialize(const USuperDexBotComponent* Component) {
  return true;
}

void USuperDexBotController::OnBotSpawned(USuperDexBotComponent* Component) {}

void USuperDexBotController::Shutdown(const USuperDexBotComponent* Component) {}

bool USuperDexBotController::StepController(
    const USuperDexBotComponent* Component,
    const FMochiArticulationState& State,
    double MochiDeltaTime,
    mochi::Actor* ArticulatedActor,
    mochi::Scene* Scene) {
  return false;
}

void USuperDexBotController::TickController(
    const USuperDexBotComponent* Component,
    double GameDeltaTime) {}

TArray<superdex::robotics::ControllerEntry> USuperDexBotController::ExportControllerEntries(
    const FString& OutputDir) const {
  return {};
}
