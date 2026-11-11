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

#include "MochiConstraintUtil.h"
#include "MochiComponentTags.h"
#include "MochiMeshComponentInterface.h"
#include "MochiPhysicsModule.h"

USceneComponent*
FindComponentRecursive(UActorComponent* Caller, FName ComponentName, const AActor* Actor) {
  if (ComponentName.IsNone()) {
    return nullptr;
  }
  // first search this component's owning actor
  const auto actor = Actor ? Actor : Caller->GetOwner();
  TArray<USceneComponent*> comps;
  actor->GetComponents(comps);
  for (USceneComponent* comp : comps) {
    if (comp->GetName() == ComponentName) {
      return comp;
    }
  }

  // next search any child actor components
  TArray<AActor*> childActors;
  actor->GetAllChildActors(childActors, false);
  for (const auto childActor : childActors) {
    if (const auto result = FindComponentRecursive(Caller, ComponentName, childActor)) {
      return result;
    }
  }
  // next search actor's parent *IF* it is a ChildActor
  // note: this will cause a redundant search, but will not loop indefinitely
  if (actor->IsChildActor() && Actor == nullptr) {
    if (const auto parent = actor->GetAttachParentActor()) {
      if (const auto result = FindComponentRecursive(Caller, ComponentName, parent)) {
        return result;
      }
    }
  }
  // give up
  return nullptr;
}

bool EnsureMochiActor(UActorComponent* Caller, USceneComponent* MochiComponent) {
  if (MochiComponent && MochiComponentTags::AreComponentsDisabled(MochiComponent->GetOwner())) {
    return false;
  }
  if (const auto mochiMeshComp = Cast<IMochiMeshComponentInterface>(MochiComponent)) {
    if (mochiMeshComp->GetMochiActor() == nullptr) {
      mochiMeshComp->CreateMochiActor();
      const bool success = mochiMeshComp->GetMochiActor() != nullptr;
      if (success) {
        UE_LOG(
            LogMochiPhysics,
            Display,
            TEXT("Component [%s] created Mochi actor for Component [%s]"),
            *Caller->GetName(),
            *MochiComponent->GetName());
      }
      return success;
    }
    return true;
  }
  UE_LOG(
      LogMochiPhysics,
      Warning,
      TEXT("Specified component [%s] is not an IMochiMeshComponentInterface!"),
      *MochiComponent->GetName());
  return false;
}
