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
  const AActor* const SearchActor = Actor ? Actor : Caller->GetOwner();
  TArray<USceneComponent*> Comps;
  SearchActor->GetComponents(Comps);
  for (USceneComponent* Comp : Comps) {
    if (Comp->GetName() == ComponentName) {
      return Comp;
    }
  }

  // next search any child actor components
  TArray<AActor*> ChildActors;
  SearchActor->GetAllChildActors(ChildActors, false);
  for (AActor* const ChildActor : ChildActors) {
    if (USceneComponent* const Result = FindComponentRecursive(Caller, ComponentName, ChildActor)) {
      return Result;
    }
  }
  // next search actor's parent *IF* it is a ChildActor
  // note: this will cause a redundant search, but will not loop indefinitely
  if (SearchActor->IsChildActor() && Actor == nullptr) {
    if (AActor* const Parent = SearchActor->GetAttachParentActor()) {
      if (USceneComponent* const Result = FindComponentRecursive(Caller, ComponentName, Parent)) {
        return Result;
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
  if (IMochiMeshComponentInterface* const MochiMeshComp =
          Cast<IMochiMeshComponentInterface>(MochiComponent)) {
    if (MochiMeshComp->GetMochiActor() == nullptr) {
      MochiMeshComp->CreateMochiActor();
      const bool bSuccess = MochiMeshComp->GetMochiActor() != nullptr;
      if (bSuccess) {
        UE_LOG(
            LogMochiPhysics,
            Display,
            TEXT("Component [%s] created Mochi actor for Component [%s]"),
            *Caller->GetName(),
            *MochiComponent->GetName());
      }
      return bSuccess;
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
    const UMochiArticulatedActorObject* Actor = Articulation->GetMochiArticulatedActor();
    if (Actor == nullptr) {
      return EMochiBodyState::Pending;
    }
    OutScene = Actor->GetMochiScene();
    if (Link.IsNone()) {
      OutActor = Actor->GetArticulatedActorHandle();
      return EMochiBodyState::Ready;
    }
    const int LinkIndex = Actor->GetNestedLinkActorIndex(Link);
    if (!Actor->GetNestedLinkActorHandles().IsValidIndex(LinkIndex)) {
      return EMochiBodyState::Failed;
    }
    OutActor = Actor->GetNestedLinkActorHandles()[LinkIndex];
    return EMochiBodyState::Ready;
  }
  if (!EnsureMochiActor(Caller, Component)) {
    return EMochiBodyState::Failed;
  }
  const UMochiActorObject* Actor = Cast<IMochiMeshComponentInterface>(Component)->GetMochiActor();
  OutScene = Actor->GetMochiScene();
  OutActor = Actor->GetMochiActorHandle();
  return EMochiBodyState::Ready;
}

mochi::Real3 FMochiConstraintBody::LocalPositionToMochi(const FVector& Position) const {
  const FUnrealToMochiConverter U2m;
  // A Mochi mesh component creates its actor at its unscaled transform and bakes its scale into
  // the shape. Link frames are not scaled.
  return U2m.TranslationToMochi(
      Articulation ? Position : Position * Component->GetComponentTransform().GetScale3D());
}

namespace {

// Finds the body named by Segments[Index..] among Actor's components.
bool ResolveInActor(
    const AActor* Actor,
    const TArray<FString>& Segments,
    int Index,
    FMochiConstraintBody& OutBody) {
  const FString& Name = Segments[Index];
  const int NumAfter = Segments.Num() - Index - 1;
  TArray<USceneComponent*> Components;
  Actor->GetComponents(Components);
  for (USceneComponent* Component : Components) {
    if (Component->GetName() != Name) {
      continue;
    }
    if (const UChildActorComponent* Child = Cast<UChildActorComponent>(Component)) {
      return NumAfter > 0 && Child->GetChildActor() &&
          ResolveInActor(Child->GetChildActor(), Segments, Index + 1, OutBody);
    }
    if (UMochiArticulationComponent* Articulation = Cast<UMochiArticulationComponent>(Component)) {
      if (NumAfter > 1) {
        return false;
      }
      const FName Link = NumAfter == 1 ? FName(Segments[Index + 1]) : NAME_None;
      const UMochiArticulation* Asset = Articulation->GetArticulationAsset();
      if (!Link.IsNone() && (Asset == nullptr || Asset->GetLinkIndex(Link) < 0)) {
        return false;
      }
      OutBody = {Articulation, Articulation, Link};
      return true;
    }
    if (NumAfter == 0 && Cast<IMochiMeshComponentInterface>(Component)) {
      OutBody = {Component, nullptr, NAME_None};
      return true;
    }
  }
  return false;
}

bool FindBody(UActorComponent* Caller, TArray<FString> Segments, FMochiConstraintBody& OutBody) {
  const AActor* Owner = Caller->GetOwner();
  if (Owner && ResolveInActor(Owner, Segments, 0, OutBody)) {
    return true;
  }
  if (Caller->GetWorld() == nullptr) {
    return false;
  }
  for (const AActor* Actor : TActorRange<AActor>(Caller->GetWorld())) {
    const USceneComponent* Root = Actor->GetRootComponent();
    if (Actor == Owner || Root == nullptr ||
        (Actor->GetActorNameOrLabel() != Segments[0] && Root->GetName() != Segments[0])) {
      continue;
    }
    // The first segment names the actor's root component (a body, as a map import spawns one
    // actor per body) or the actor itself (an imported nested prefab).
    const FString First = Segments[0];
    Segments[0] = Root->GetName();
    if (ResolveInActor(Actor, Segments, 0, OutBody)) {
      return true;
    }
    Segments[0] = First;
    if (Segments.Num() > 1 && ResolveInActor(Actor, Segments, 1, OutBody)) {
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
  TArray<FString> Segments;
  Path.ParseIntoArray(Segments, TEXT("/"));
  if (Segments.IsEmpty() || !FindBody(Caller, MoveTemp(Segments), OutBody)) {
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
