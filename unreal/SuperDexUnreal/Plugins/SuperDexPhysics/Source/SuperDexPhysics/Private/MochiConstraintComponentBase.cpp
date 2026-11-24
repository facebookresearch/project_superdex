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
  const float MochiSaturation = GetMochiSaturation();
  ForEachConstraint([MochiSaturation](mochi::Constraint& C) {
    C.SetSaturation(MochiSaturation, mochi::ErrorLog{});
  });
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
    MochiScene->QueueOrExecuteCommand([State = std::move(MochiThreadState)](mochi::Scene* Scene) {
      for (const mochi::ConstraintHandle Handle : State->Constraints) {
        if (Handle.IsValid()) {
          Scene->DestroyConstraint(Handle);
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
      [Fn = std::move(Fn), State = MochiThreadState](mochi::Scene* Scene) {
        for (const mochi::ConstraintHandle Handle : State->Constraints) {
          if (mochi::Constraint* Constraint = Scene->GetConstraint(Handle)) {
            Fn(*Constraint);
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
  FConstraintRequest Request;
  const EMochiBodyState State = BuildConstraints(Request);
  if (State == EMochiBodyState::Pending) {
    return;
  }
  bCreatePending = false;
  if (State == EMochiBodyState::Failed || !Request.Create || Request.Scene == nullptr) {
    return;
  }
  MochiScene = Request.Scene;
  bCreated = true;
  Request.Scene->QueueOrExecuteCommand(
      [Request, MochiState = MochiThreadState](mochi::Scene* Scene) {
        std::vector<mochi::ConstraintHandle>& Constraints = MochiState->Constraints;
        Constraints = Request.Create(Scene);
        const bool bAllCreated = !Constraints.empty() &&
            std::all_of(
                Constraints.begin(), Constraints.end(), [](mochi::ConstraintHandle Handle) -> bool {
                  return Handle.IsValid();
                });
        if (!bAllCreated) {
          return;
        }
        for (const std::pair<mochi::ActorHandle, mochi::ActorHandle>& NoContactPair :
             Request.NoContactPairs) {
          const mochi::ActorHandle& A = NoContactPair.first;
          const mochi::ActorHandle& B = NoContactPair.second;
          Scene->EnableActorContactSymmetric(
              A, B, /*enable*/ false, mochi::IncludeNestedActors::No, mochi::ErrorLog{});
        }
      });
}
