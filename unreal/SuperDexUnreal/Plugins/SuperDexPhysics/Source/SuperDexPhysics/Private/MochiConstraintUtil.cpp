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
#include "Components/ChildActorComponent.h"
#include "EngineUtils.h"
#include "MochiArticulatedActorObject.h"
#include "MochiArticulation.h"
#include "MochiArticulationComponent.h"
#include "MochiComponentTags.h"
#include "MochiConversions.h"
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

EMochiBodyState FMochiConstraintBody::GetMochiActor(
    UActorComponent* Caller,
    UMochiSceneObject*& OutScene,
    mochi::ActorHandle& OutActor) const {
  if (Articulation) {
    const UMochiArticulatedActorObject* actor = Articulation->GetMochiArticulatedActor();
    if (actor == nullptr) {
      return EMochiBodyState::Pending;
    }
    OutScene = actor->GetMochiScene();
    if (Link.IsNone()) {
      OutActor = actor->GetArticulatedActorHandle();
      return EMochiBodyState::Ready;
    }
    const int link = actor->GetNestedLinkActorIndex(Link);
    if (!actor->GetNestedLinkActorHandles().IsValidIndex(link)) {
      return EMochiBodyState::Failed;
    }
    OutActor = actor->GetNestedLinkActorHandles()[link];
    return EMochiBodyState::Ready;
  }
  if (!EnsureMochiActor(Caller, Component)) {
    return EMochiBodyState::Failed;
  }
  const UMochiActorObject* actor = Cast<IMochiMeshComponentInterface>(Component)->GetMochiActor();
  OutScene = actor->GetMochiScene();
  OutActor = actor->GetMochiActorHandle();
  return EMochiBodyState::Ready;
}

mochi::Real3 FMochiConstraintBody::LocalPositionToMochi(const FVector& Position) const {
  const FUnrealToMochiConverter u2m;
  // A Mochi mesh component creates its actor at its unscaled transform and bakes its scale into
  // the shape. Link frames are not scaled.
  return u2m.TranslationToMochi(
      Articulation ? Position : Position * Component->GetComponentTransform().GetScale3D());
}

namespace {

// Finds the body named by Segments[Index..] among Actor's components.
bool ResolveInActor(
    const AActor* Actor,
    const TArray<FString>& Segments,
    int Index,
    FMochiConstraintBody& OutBody) {
  const FString& name = Segments[Index];
  const int numAfter = Segments.Num() - Index - 1;
  TArray<USceneComponent*> components;
  Actor->GetComponents(components);
  for (USceneComponent* component : components) {
    if (component->GetName() != name) {
      continue;
    }
    if (const UChildActorComponent* child = Cast<UChildActorComponent>(component)) {
      return numAfter > 0 && child->GetChildActor() &&
          ResolveInActor(child->GetChildActor(), Segments, Index + 1, OutBody);
    }
    if (UMochiArticulationComponent* articulation = Cast<UMochiArticulationComponent>(component)) {
      if (numAfter > 1) {
        return false;
      }
      const FName link = numAfter == 1 ? FName(Segments[Index + 1]) : NAME_None;
      const UMochiArticulation* asset = articulation->GetArticulationAsset();
      if (!link.IsNone() && (asset == nullptr || asset->GetLinkIndex(link) < 0)) {
        return false;
      }
      OutBody = {articulation, articulation, link};
      return true;
    }
    if (numAfter == 0 && Cast<IMochiMeshComponentInterface>(component)) {
      OutBody = {component, nullptr, NAME_None};
      return true;
    }
  }
  return false;
}

bool FindBody(UActorComponent* Caller, TArray<FString> Segments, FMochiConstraintBody& OutBody) {
  const AActor* owner = Caller->GetOwner();
  if (owner && ResolveInActor(owner, Segments, 0, OutBody)) {
    return true;
  }
  if (Caller->GetWorld() == nullptr) {
    return false;
  }
  for (const AActor* actor : TActorRange<AActor>(Caller->GetWorld())) {
    const USceneComponent* root = actor->GetRootComponent();
    if (actor == owner || root == nullptr ||
        (actor->GetActorNameOrLabel() != Segments[0] && root->GetName() != Segments[0])) {
      continue;
    }
    // The first segment names the actor's root component (a body, as a map import spawns one
    // actor per body) or the actor itself (an imported nested prefab).
    const FString first = Segments[0];
    Segments[0] = root->GetName();
    if (ResolveInActor(actor, Segments, 0, OutBody)) {
      return true;
    }
    Segments[0] = first;
    if (Segments.Num() > 1 && ResolveInActor(actor, Segments, 1, OutBody)) {
      return true;
    }
  }
  return false;
}

} // namespace

bool ResolveMochiConstraintBody(
    UActorComponent* Caller,
    const FString& Path,
    FMochiConstraintBody& OutBody) {
  TArray<FString> segments;
  Path.ParseIntoArray(segments, TEXT("/"));
  if (segments.IsEmpty() || !FindBody(Caller, MoveTemp(segments), OutBody)) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%s] %s: no Mochi body named \"%s\"."),
        *GetNameSafe(Caller->GetOwner()),
        *Caller->GetName(),
        *Path);
    return false;
  }
  return true;
}
