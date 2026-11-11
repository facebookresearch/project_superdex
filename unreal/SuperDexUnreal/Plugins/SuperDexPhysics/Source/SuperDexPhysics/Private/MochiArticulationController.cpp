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

#include "MochiArticulationController.h"

bool UMochiArticulationController::Initialize_Implementation(
    const UMochiArticulationComponent* Component) {
  return true;
}

void UMochiArticulationController::OnArticulationSpawned(UMochiArticulationComponent* Component) {}

void UMochiArticulationController::Shutdown_Implementation(
    const UMochiArticulationComponent* Component) {}

bool UMochiArticulationController::Control(
    const UMochiArticulationComponent* Component,
    const FMochiArticulationState& State,
    double MochiDeltaTime,
    FMochiArticulationControllerCommands& OutCommands,
    mochi::Scene* mochiScene) {
  return false;
}

void UMochiArticulationController::TickController_Implementation(
    const UMochiArticulationComponent* Component,
    double GameDeltaTime) {}
