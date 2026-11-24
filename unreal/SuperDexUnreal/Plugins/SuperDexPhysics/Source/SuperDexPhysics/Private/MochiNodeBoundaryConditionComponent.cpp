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
  FMochiConstraintBody ResolvedBody;
  if (!ResolveMochiConstraintBody(this, Body, ResolvedBody)) {
    return;
  }
  const USceneComponent* VolumeComponent = FindComponentRecursive(this, Volume);
  const UBoxComponent* Box = Cast<UBoxComponent>(VolumeComponent);
  const USphereComponent* Sphere = Cast<USphereComponent>(VolumeComponent);
  if (Box == nullptr && Sphere == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%s] %s: %s is not a Box Collision or Sphere Collision component."),
        *GetNameSafe(GetOwner()),
        *GetName(),
        *Volume.ToString());
    return;
  }
  UMochiSceneObject* PhysicsScene = nullptr;
  mochi::ActorHandle ActorHandle;
  if (ResolvedBody.Articulation ||
      ResolvedBody.GetMochiActor(this, PhysicsScene, ActorHandle) != EMochiBodyState::Ready) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%s] %s: %s has no deformable Mochi actor."),
        *GetNameSafe(GetOwner()),
        *GetName(),
        *Body);
    return;
  }
  MochiSceneObject = PhysicsScene;
  bCreated = true;

  const FUnrealToMochiConverter U2m;
  const mochi::TransformRT VolWorldFromLocal =
      U2m.TransformToMochi_RT(VolumeComponent->GetComponentTransform());
  const mochi::Real3 VolBoxExtents =
      Box ? U2m.TranslationToMochi(Box->GetUnscaledBoxExtent()) : mochi::Real3{};
  const mochi::real VolSphereRadius =
      Sphere ? U2m.MeasureToMochi(Sphere->GetUnscaledSphereRadius()) : 0_r;
  const bool bVolUseBox = Box != nullptr;
  // Capture by value so we don't have to wait
  const FString LogPrefix = FString::Printf(TEXT("[%s] %s"), *GetNameSafe(GetOwner()), *GetName());
  PhysicsScene->QueueOrExecuteCommand([bVolUseBox,
                                       LogPrefix,
                                       VolWorldFromLocal,
                                       VolBoxExtents,
                                       VolSphereRadius,
                                       ActorHandle,
                                       MochiThreadState =
                                           this->MochiThreadState](mochi::Scene* Scene) {
    mochi::ErrorLog E;
    mochi::Actor* SoftActor = Scene->GetActor(ActorHandle);
    if (SoftActor == nullptr ||
        (SoftActor->GetType() != mochi::ActorType::Soft &&
         SoftActor->GetType() != mochi::ActorType::Shell)) {
      UE_LOG(
          LogMochiPhysics, Error, TEXT("%s: the body is not a soft or shell actor."), *LogPrefix);
      return;
    }
    MochiThreadState->DeformableActorHandle = ActorHandle;
    const mochi::TransformRT SoftWorldFromLocal = SoftActor->GetRootTransform();
    const mochi::TransformRT SoftLocalFromWorld = Invert(SoftWorldFromLocal);
    const mochi::QueryHandle Query =
        SoftActor->RegisterQueryAndCompute(mochi::QueryType::NodePositions, E);
    // pin each node at its current world position
    auto OnEach = [&](int NodeIndex, mochi::Real3 Pos) {
      std::array<int, 1> Node = {NodeIndex};
      Pos = SoftWorldFromLocal.TransformPoint(Pos);
      SoftActor->AddBoundaryConditionNodesWorld(Node, Pos, E);
      MochiThreadState->PinnedNodeIndices.Add(NodeIndex);
    };
    if (bVolUseBox) {
      mochi::TransformRT VolLocalFromSoft = SoftLocalFromWorld * VolWorldFromLocal;
      mochi::Obb Obb(VolLocalFromSoft, VolBoxExtents);
      SoftActor->QueryNodesInVolumeLocal(Obb, false, OnEach, E);
    } else {
      mochi::Sphere SphereWorld = TransformShape(
          VolWorldFromLocal, mochi::Sphere(mochi::Real3{0_r, 0_r, 0_r}, VolSphereRadius));
      mochi::Sphere SphereLocal = TransformShape(SoftLocalFromWorld, SphereWorld);
      SoftActor->QueryNodesInVolumeLocal(SphereLocal, false, OnEach, E);
    }
    SoftActor->CancelQuery(Query);
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
          if (mochi::Actor* DeformableActor =
                  Scene->GetActor(MochiThreadState->DeformableActorHandle)) {
            DeformableActor->ClearBoundaryConditions();
          }
        });
    MochiSceneObject->WaitForQueuedCommands();
  }
  MochiThreadState = std::make_shared<FMochiState>();
}
