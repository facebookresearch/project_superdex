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

#include "MochiImplicitShapeComponent.h"
#include "MochiComponentTags.h"
#include "MochiPhysicsModule.h"

#include <mochi_core/utils/profile.h>

void UMochiImplicitShapeComponent::CreateMochiActor() {
  MOCHI_PROFILE_SCOPE();
  if (!GetOwner()->HasAuthority()) {
    return; // The server or solo player has authority. Multiplayer clients don't.
  }

  DestroyMochiActor();
  if (Shape == EMochiImplicitShape::Plane) {
    MochiActorObject = UMochiActorObject::CreateImplicitPlane(
        this,
        GetName(),
        GetComponentTransform(),
        FVector(0, 0, 1),
        0,
        MaterialParams,
        ContactParams,
        Layer);
  } else if (Shape == EMochiImplicitShape::Sphere && SphereRadius > 0) {
    MochiActorObject = UMochiActorObject::CreateImplicitSphere(
        this,
        GetName(),
        GetComponentTransform(),
        FVector::Zero(),
        SphereRadius,
        MaterialParams,
        ContactParams,
        Layer);
  } else {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiImplicitShapeComponent] Did not define a valid implicit shape!"));
  }
}

void UMochiImplicitShapeComponent::DestroyMochiActor() {
  MOCHI_PROFILE_SCOPE();
  if (MochiActorObject) {
    MochiActorObject->DestroyMochiResources();
    MochiActorObject = nullptr;
  }
}

UMochiActorObject* UMochiImplicitShapeComponent::GetMochiActor() const {
  return MochiActorObject;
}

bool UMochiImplicitShapeComponent::GetContactsEnabled() const {
  return false;
}

UMochiModel* UMochiImplicitShapeComponent::GetMochiModel() const {
  return nullptr;
}

void UMochiImplicitShapeComponent::BeginPlay() {
  MOCHI_PROFILE_SCOPE();
  Super::BeginPlay();

  if (MochiComponentTags::AreComponentsDisabled(GetOwner())) {
    return;
  }

  // create the actor if it hasn't been created externally (e.g. by MochiConstraintComponent)
  if (MochiActorObject == nullptr) {
    CreateMochiActor();
  }
}

void UMochiImplicitShapeComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  MOCHI_PROFILE_SCOPE();
  Super::EndPlay(EndPlayReason);
  DestroyMochiActor();
}
