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

#include "MochiNodeBoundaryConditionComponent.h"
#include "Components/BoxComponent.h"
#include "Components/SphereComponent.h"
#include "MochiConstraintUtil.h"
#include "MochiConversions.h"
#include "MochiPhysicsModule.h"
#include "MochiSceneObject.h"

#include <mochi_core/geometry/geometry_utils.h>

bool UMochiNodeBoundaryConditionComponent::GetPinnedNodeIndices(TArray<int>& OutNodeIndices) const {
  if (!bCreated || !MochiThreadState->bReady.load(std::memory_order_acquire)) {
    return false;
  }
  OutNodeIndices = MochiThreadState->PinnedNodeIndices;
  return true;
}

void UMochiNodeBoundaryConditionComponent::BeginPlay() {
  Super::BeginPlay();
  CreateBoundaryConditions();
}

void UMochiNodeBoundaryConditionComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  DestroyBoundaryConditions();
  Super::EndPlay(EndPlayReason);
}

void UMochiNodeBoundaryConditionComponent::CreateBoundaryConditions() {
  DestroyBoundaryConditions();
  FMochiConstraintBody body;
  if (!ResolveMochiConstraintBody(this, Body, body)) {
    return;
  }
  const USceneComponent* volume = FindComponentRecursive(this, Volume);
  const UBoxComponent* box = Cast<UBoxComponent>(volume);
  const USphereComponent* sphere = Cast<USphereComponent>(volume);
  if (box == nullptr && sphere == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%s] %s: %s is not a Box Collision or Sphere Collision component."),
        *GetNameSafe(GetOwner()),
        *GetName(),
        *Volume.ToString());
    return;
  }
  UMochiSceneObject* scene = nullptr;
  mochi::ActorHandle actorHandle;
  if (body.Articulation || body.GetMochiActor(this, scene, actorHandle) != EMochiBodyState::Ready) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%s] %s: %s has no deformable Mochi actor."),
        *GetNameSafe(GetOwner()),
        *GetName(),
        *Body);
    return;
  }
  MochiSceneObject = scene;
  bCreated = true;

  const FUnrealToMochiConverter u2m;
  const mochi::TransformRT volWorldFromLocal =
      u2m.TransformToMochi_RT(volume->GetComponentTransform());
  const mochi::Real3 volBoxExtents =
      box ? u2m.TranslationToMochi(box->GetUnscaledBoxExtent()) : mochi::Real3{};
  const mochi::real volSphereRadius =
      sphere ? u2m.MeasureToMochi(sphere->GetUnscaledSphereRadius()) : 0_r;
  const bool bVolUseBox = box != nullptr;
  // Capture by value so we don't have to wait
  const FString logPrefix = FString::Printf(TEXT("[%s] %s"), *GetNameSafe(GetOwner()), *GetName());
  scene->QueueOrExecuteCommand([bVolUseBox,
                                logPrefix,
                                volWorldFromLocal,
                                volBoxExtents,
                                volSphereRadius,
                                actorHandle,
                                MochiThreadState = this->MochiThreadState](mochi::Scene* Scene) {
    mochi::ErrorLog e;
    mochi::Actor* softActor = Scene->GetActor(actorHandle);
    if (softActor == nullptr ||
        (softActor->GetType() != mochi::ActorType::Soft &&
         softActor->GetType() != mochi::ActorType::Shell)) {
      UE_LOG(
          LogMochiPhysics, Error, TEXT("%s: the body is not a soft or shell actor."), *logPrefix);
      return;
    }
    MochiThreadState->DeformableActorHandle = actorHandle;
    const auto softWorldFromLocal = softActor->GetRootTransform();
    const auto softLocalFromWorld = Invert(softWorldFromLocal);
    const auto query = softActor->RegisterQueryAndCompute(mochi::QueryType::NodePositions, e);
    // pin each node at its current world position
    auto onEach = [&](int iNode, mochi::Real3 pos) {
      std::array<int, 1> node = {iNode};
      pos = softWorldFromLocal.TransformPoint(pos);
      softActor->AddBoundaryConditionNodesWorld(node, pos, e);
      MochiThreadState->PinnedNodeIndices.Add(iNode);
    };
    if (bVolUseBox) {
      auto volLocalFromSoft = softLocalFromWorld * volWorldFromLocal;
      mochi::Obb obb(volLocalFromSoft, volBoxExtents);
      softActor->QueryNodesInVolumeLocal(obb, false, onEach, e);
    } else {
      auto sphereWorld = TransformShape(
          volWorldFromLocal, mochi::Sphere(mochi::Real3{0_r, 0_r, 0_r}, volSphereRadius));
      auto sphereLocal = TransformShape(softLocalFromWorld, sphereWorld);
      softActor->QueryNodesInVolumeLocal(sphereLocal, false, onEach, e);
    }
    softActor->CancelQuery(query);
    MochiThreadState->bReady.store(true, std::memory_order_release);
  });
}

void UMochiNodeBoundaryConditionComponent::DestroyBoundaryConditions() {
  if (!bCreated) {
    return;
  }
  bCreated = false;
  if (MochiSceneObject.IsValid()) {
    // Pass ownership of MochiThreadState to the lambda so we don't have to wait
    MochiSceneObject->QueueOrExecuteCommand(
        [MochiThreadState = std::move(MochiThreadState)](mochi::Scene* Scene) {
          if (mochi::Actor* deformableActor =
                  Scene->GetActor(MochiThreadState->DeformableActorHandle)) {
            deformableActor->ClearBoundaryConditions();
          }
        });
    MochiSceneObject->WaitForQueuedCommands();
  }
  MochiThreadState = std::make_shared<FMochiState>();
}
