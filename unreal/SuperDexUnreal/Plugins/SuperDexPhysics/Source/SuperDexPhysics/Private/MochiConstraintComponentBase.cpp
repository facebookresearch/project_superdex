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

#include "MochiConstraintComponentBase.h"
#include "MochiPhysicsModule.h"
#include "MochiSceneObject.h"

#include <algorithm>

UMochiConstraintComponentBase::UMochiConstraintComponentBase() {
  // Ticks only while waiting for the bodies' Mochi actors.
  PrimaryComponentTick.bCanEverTick = true;
  PrimaryComponentTick.bStartWithTickEnabled = false;
}

void UMochiConstraintComponentBase::SetStiffness(float NewStiffness) {
  Stiffness = NewStiffness;
  ForEachConstraint(
      [NewStiffness](mochi::Constraint& C) { C.SetStiffness(NewStiffness, mochi::ErrorLog{}); });
}

void UMochiConstraintComponentBase::SetDamping(float NewDamping) {
  Damping = NewDamping;
  ForEachConstraint(
      [NewDamping](mochi::Constraint& C) { C.SetDamping(NewDamping, mochi::ErrorLog{}); });
}

float UMochiConstraintComponentBase::GetMochiSaturation() const {
  return Saturation < 0 ? -1.0f : static_cast<float>(Saturation * GetSaturationScale());
}

void UMochiConstraintComponentBase::SetSaturation(float NewSaturation) {
  Saturation = NewSaturation;
  const float saturation = GetMochiSaturation();
  ForEachConstraint(
      [saturation](mochi::Constraint& C) { C.SetSaturation(saturation, mochi::ErrorLog{}); });
}

void UMochiConstraintComponentBase::CreateConstraints() {
  DestroyConstraints();
  bCreatePending = true;
  TryCreateConstraints();
  SetComponentTickEnabled(bCreatePending);
}

void UMochiConstraintComponentBase::DestroyConstraints() {
  bCreatePending = false;
  if (!bCreated) {
    return;
  }
  bCreated = false;
  if (MochiScene.IsValid()) {
    MochiScene->QueueOrExecuteCommand([state = std::move(MochiThreadState)](mochi::Scene* Scene) {
      for (const mochi::ConstraintHandle handle : state->Constraints) {
        if (handle.IsValid()) {
          Scene->DestroyConstraint(handle);
        }
      }
    });
    MochiScene->WaitForQueuedCommands();
  }
  MochiThreadState = std::make_shared<FMochiState>();
}

void UMochiConstraintComponentBase::FillCommonParams(mochi::ConstraintParams& Params) const {
  Params.stiffness = Stiffness;
  Params.damping = Damping;
  Params.saturation = GetMochiSaturation();
}

void UMochiConstraintComponentBase::LogError(const FString& Message) const {
  UE_LOG(
      LogMochiPhysics, Error, TEXT("[%s] %s: %s"), *GetNameSafe(GetOwner()), *GetName(), *Message);
}

void UMochiConstraintComponentBase::ForEachConstraint(
    std::function<void(mochi::Constraint&)> Fn) const {
  if (!bCreated || !MochiScene.IsValid()) {
    return;
  }
  MochiScene->QueueOrExecuteCommand(
      [Fn = std::move(Fn), state = MochiThreadState](mochi::Scene* Scene) {
        for (const mochi::ConstraintHandle handle : state->Constraints) {
          if (mochi::Constraint* constraint = Scene->GetConstraint(handle)) {
            Fn(*constraint);
          }
        }
      });
}

EMochiBodyState UMochiConstraintComponentBase::ResolveBody(
    const FString& Path,
    FMochiConstraintBody& OutBody,
    UMochiSceneObject*& OutScene,
    mochi::ActorHandle& OutActor) {
  if (!ResolveMochiConstraintBody(this, Path, OutBody)) {
    return EMochiBodyState::Failed;
  }
  return OutBody.GetMochiActor(this, OutScene, OutActor);
}

void UMochiConstraintComponentBase::BeginPlay() {
  Super::BeginPlay();
  CreateConstraints();
}

void UMochiConstraintComponentBase::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  DestroyConstraints();
  Super::EndPlay(EndPlayReason);
}

void UMochiConstraintComponentBase::TickComponent(
    float DeltaTime,
    ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction) {
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
  if (bCreatePending) {
    TryCreateConstraints();
  }
  if (!bCreatePending) {
    SetComponentTickEnabled(false);
  }
}

void UMochiConstraintComponentBase::TryCreateConstraints() {
  if (GetWorld() == nullptr || !GetWorld()->IsGameWorld()) {
    bCreatePending = false;
    return;
  }
  FConstraintRequest request;
  const EMochiBodyState state = BuildConstraints(request);
  if (state == EMochiBodyState::Pending) {
    return;
  }
  bCreatePending = false;
  if (state == EMochiBodyState::Failed || !request.Create || request.Scene == nullptr) {
    return;
  }
  MochiScene = request.Scene;
  bCreated = true;
  request.Scene->QueueOrExecuteCommand(
      [request, mochiState = MochiThreadState](mochi::Scene* Scene) {
        std::vector<mochi::ConstraintHandle>& constraints = mochiState->Constraints;
        constraints = request.Create(Scene);
        const bool bAllCreated = !constraints.empty() &&
            std::all_of(constraints.begin(), constraints.end(), [](mochi::ConstraintHandle Handle) {
              return Handle.IsValid();
            });
        if (!bAllCreated) {
          return;
        }
        for (const auto& [a, b] : request.NoContactPairs) {
          Scene->EnableActorContactSymmetric(
              a, b, /*enable*/ false, mochi::IncludeNestedActors::No, mochi::ErrorLog{});
        }
      });
}
