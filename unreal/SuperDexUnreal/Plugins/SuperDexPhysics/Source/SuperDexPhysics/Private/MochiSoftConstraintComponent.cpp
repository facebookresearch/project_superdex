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

#include "MochiSoftConstraintComponent.h"
#include "Components/BoxComponent.h"
#include "Components/SphereComponent.h"
#include "MochiConstraintUtil.h"
#include "MochiConversions.h"
#include "MochiPhysicsModule.h"
#include "MochiSceneObject.h"

#include <mochi_core/geometry/geometry_utils.h>

void UMochiSoftConstraintComponent::SetStiffness(float NewStiffness) {
  Stiffness = NewStiffness;
  if (MochiSceneObject.IsValid()) {
    MochiSceneObject->QueueOrExecuteCommand(
        [NewStiffness, MochiThreadState = this->MochiThreadState](mochi::Scene* Scene) {
          for (auto handle : MochiThreadState->ConstraintHandles) {
            if (auto* con = Scene->GetConstraint(handle)) {
              con->SetStiffness(NewStiffness, mochi::ErrorLog{});
            }
          }
        });
  }
}

void UMochiSoftConstraintComponent::SetDamping(float NewDamping) {
  Damping = NewDamping;
  if (MochiSceneObject.IsValid()) {
    MochiSceneObject->QueueOrExecuteCommand(
        [NewDamping, MochiThreadState = this->MochiThreadState](mochi::Scene* Scene) {
          mochi::ErrorLog e;
          for (auto handle : MochiThreadState->ConstraintHandles) {
            if (auto* con = Scene->GetConstraint(handle)) {
              con->SetDamping(NewDamping, e);
            }
          }
        });
  }
}

bool UMochiSoftConstraintComponent::GetConstrainedNodeIndices(TArray<int>& ConstrainedNodeIndices) {
  // NB: This *should* be thread safe since bConstraintsCreated will only be true
  // after the mochi thread has finished populating ConstrainedNodeIndices
  if (MochiThreadState && bConstraintsCreated) {
    ConstrainedNodeIndices = MochiThreadState->ConstrainedNodeIndices;
    return true;
  }
  return false;
}

UMochiSoftConstraintComponent::UMochiSoftConstraintComponent() {
  PrimaryComponentTick.bCanEverTick = true;
}

void UMochiSoftConstraintComponent::BeginPlay() {
  Super::BeginPlay();
  CreateConstraints();
}

void UMochiSoftConstraintComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  Super::EndPlay(EndPlayReason);
  DestroyConstraints();
}

void UMochiSoftConstraintComponent::TickComponent(
    float DeltaTime,
    ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction) {
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
}

void UMochiSoftConstraintComponent::TryCreateConstraints_OneBody() {
  FUnrealToMochiConverter u2m;
  const auto mochiMeshComp1 = Cast<IMochiMeshComponentInterface>(Component1);
  const auto mochiActor = mochiMeshComp1->GetMochiActor();
  MochiSceneObject = mochiActor->GetMochiScene();
  const auto actorHandle = mochiActor->GetMochiActorHandle();

  bool bVolUseBox = false;
  auto volWorldFromLocal = u2m.TransformToMochi_RT(OneBodyVolume->GetComponentTransform());
  mochi::Real3 volBoxExtents = {};
  mochi::real volSphereRadius = 0_r;
  const auto volBox = Cast<UBoxComponent>(OneBodyVolume);
  const auto volSphere = Cast<USphereComponent>(OneBodyVolume);
  if (volBox == nullptr && volSphere == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiSoftConstraintComponent] OneBodyVolume must be BoxCollision or SphereCollision component!"));
    return;
  }
  if (volBox) {
    bVolUseBox = true;
    volBoxExtents = u2m.TranslationToMochi(volBox->GetUnscaledBoxExtent());
  } else if (volSphere) {
    bVolUseBox = false;
    volSphereRadius = u2m.MeasureToMochi(volSphere->GetUnscaledSphereRadius());
  }

  // Capture by value so we don't have to wait
  MochiSceneObject->QueueOrExecuteCommand(
      [&bConstraintsCreated = bConstraintsCreated,
       bVolUseBox,
       volWorldFromLocal,
       volBoxExtents,
       volSphereRadius,
       actorHandle,
       MochiThreadState = this->MochiThreadState](mochi::Scene* Scene) {
        mochi::ErrorLog e;
        if (Scene->GetActor(actorHandle)->GetType() == mochi::ActorType::Soft ||
            Scene->GetActor(actorHandle)->GetType() == mochi::ActorType::Shell) {
          MochiThreadState->DeformableActorHandle = actorHandle;
          const auto softActor = Scene->GetActor(actorHandle);
          const auto softWorldFromLocal = softActor->GetRootTransform();
          const auto softLocalFromWorld = Invert(softWorldFromLocal);
          const auto query = softActor->RegisterQueryAndCompute(mochi::QueryType::NodePositions, e);
          // constrain each node to current world position
          MochiThreadState->ConstrainedNodeIndices.SetNum(0);
          auto onEach = [&](int iNode, mochi::Real3 pos) {
            std::array<int, 1> node = {iNode};
            pos = softWorldFromLocal.TransformPoint(pos);
            softActor->AddBoundaryConditionNodesWorld(node, pos, e);
            MochiThreadState->ConstrainedNodeIndices.Add(iNode);
          };
          // query nodes in box
          if (bVolUseBox) {
            auto volLocalFromSoft = softLocalFromWorld * volWorldFromLocal;
            mochi::Obb obb(volLocalFromSoft, volBoxExtents);
            softActor->QueryNodesInVolumeLocal(obb, false, onEach, e);
          }
          // query nodes in sphere
          else {
            auto sphere = mochi::Sphere(mochi::Real3{0.0f, 0.0f, 0.0f}, volSphereRadius);
            auto sphereWorld = TransformShape(volWorldFromLocal, sphere);
            auto sphereLocal = TransformShape(softLocalFromWorld, sphereWorld);
            softActor->QueryNodesInVolumeLocal(sphereLocal, false, onEach, e);
          }
          softActor->CancelQuery(query);
          bConstraintsCreated = true;
        } else {
          UE_LOG(
              LogMochiPhysics,
              Error,
              TEXT("[UMochiSoftConstraintComponent] ActorType is not Soft or Shell!"));
        }
      });
}

void UMochiSoftConstraintComponent::TryCreateConstraints_TwoBody() {
  const auto mochiMeshComp1 = Cast<IMochiMeshComponentInterface>(Component1);
  const auto mochiMeshComp2 = Cast<IMochiMeshComponentInterface>(Component2);
  const auto mochiActor1 = mochiMeshComp1->GetMochiActor();
  const auto mochiActor2 = mochiMeshComp2->GetMochiActor();
  if (mochiActor1->GetMochiScene() != mochiActor2->GetMochiScene()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiSoftConstraintComponent] Component1 and Component2 are not in the same Mochi scene."));
    return;
  }
  MochiSceneObject = mochiActor1->GetMochiScene();
  const auto actorHandle1 = mochiActor1->GetMochiActorHandle();
  const auto actorHandle2 = mochiActor2->GetMochiActorHandle();

  // Capture by value so we don't have to wait
  MochiSceneObject->QueueOrExecuteCommand([&bConstraintsCreated = bConstraintsCreated,
                                           actorHandle1,
                                           actorHandle2,
                                           Stiffness = this->Stiffness,
                                           Damping = this->Damping,
                                           AllowContact = this->AllowContact,
                                           MochiThreadState =
                                               this->MochiThreadState](mochi::Scene* Scene) {
    mochi::ErrorLog e;
    // determine actor types (deformable: Soft/Shell, or Rigid)
    auto actor1Type = Scene->GetActor(actorHandle1)->GetType();
    auto actor2Type = Scene->GetActor(actorHandle2)->GetType();
    const bool bActor1Deformable =
        actor1Type == mochi::ActorType::Soft || actor1Type == mochi::ActorType::Shell;
    const bool bActor2Deformable =
        actor2Type == mochi::ActorType::Soft || actor2Type == mochi::ActorType::Shell;
    const bool bActor1Rigid = actor1Type == mochi::ActorType::Rigid;
    const bool bActor2Rigid = actor2Type == mochi::ActorType::Rigid;
    if (bActor1Rigid && bActor2Rigid) {
      // both actors Rigid not supported
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT(
              "[UMochiSoftConstraintComponent] Components are both Rigid! This is not supported."));
    } else if (bActor1Deformable && bActor2Deformable) {
      // TODO: implement deformable/deformable constraint
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT(
              "[UMochiSoftConstraintComponent] Constraint does not yet support Soft-Soft constraints. One actor must be Rigid."));
    } else if ((bActor1Deformable && bActor2Rigid) || (bActor1Rigid && bActor2Deformable)) {
      const auto deformableActorHandle = bActor1Deformable ? actorHandle1 : actorHandle2;
      const auto rigidActorHandle = bActor1Deformable ? actorHandle2 : actorHandle1;
      MochiThreadState->ConstrainedNodeIndices.SetNum(0);
      auto onEach = [&](int iNode, mochi::Real3 /*pos*/) {
        mochi::DeformableNodeToRigidConstraintParams params;
        params.deformableActor = deformableActorHandle;
        params.rigidActor = rigidActorHandle;
        params.fixToDeformablePos = true;
        params.deformableNodeIndex = iNode;
        params.stiffness = Stiffness;
        params.damping = Damping;
        auto* constraint = Scene->CreateDeformableNodeToRigidConstraint(params, e);
        if (e.IsOK()) {
          auto handle = constraint->GetHandle();
          MochiThreadState->ConstraintHandles.Add(handle);
          MochiThreadState->ConstrainedNodeIndices.Add(iNode);
        }
      };
      // / Find nodes in the deformable actor which overlap the rigid actor's Aabb.
      const auto deformableActor = Scene->GetActor(deformableActorHandle);
      const auto rigidActor = Scene->GetActor(rigidActorHandle);
      const mochi::Aabb aabbRigidWorld = rigidActor->GetAabbWorld(e);
      const auto deformableLocalFromWorld = Invert(deformableActor->GetRootTransform());
      const auto obbRigidLocal =
          mochi::TransformShape(deformableLocalFromWorld, mochi::GetObb(aabbRigidWorld));
      const auto query =
          deformableActor->RegisterQueryAndCompute(mochi::QueryType::NodePositions, e);
      deformableActor->QueryNodesInVolumeLocal(obbRigidLocal, false, onEach, e);
      deformableActor->CancelQuery(query);
      // Optionally disable contact between these actors
      if (!AllowContact) {
        // Prevent contact in both directions
        Scene->EnableActorContactSymmetric(
            deformableActorHandle,
            rigidActorHandle,
            /*enable*/ false,
            mochi::IncludeNestedActors::No,
            e);
      }
      bConstraintsCreated = true;
    } else {
      // actors types are not supported
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiSoftConstraintComponent] ActorTypes are not supported!"));
    }
  });
}

void UMochiSoftConstraintComponent::CreateConstraints() {
  // destroy existing constraints
  if (bConstraintsCreated) {
    DestroyConstraints();
  }
  if (Mode == EMochiConstraintComponentMode::OneBody) {
    // find component 1
    if (ComponentName1.IsNone()) {
      UE_LOG(
          LogMochiPhysics, Error, TEXT("[UMochiSoftConstraintComponent] ComponentName1 is None!"))
      return;
    }
    Component1 = FindComponentRecursive(this, ComponentName1);
    if (Component1 == nullptr) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiSoftConstraintComponent] Failed to find Component1 [%s]!"),
          *ComponentName1.ToString());
      return;
    }
    OneBodyVolume = FindComponentRecursive(this, OneBodyVolumeName);
    if (OneBodyVolume == nullptr) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiSoftConstraintComponent] Failed to find OneBodyVolume [%s]!"),
          *OneBodyVolumeName.ToString());
      return;
    }
    // ensure component 1 mochi actor created
    if (EnsureMochiActor(this, Component1)) {
      // create constraints
      TryCreateConstraints_OneBody();
    }
  } else if (Mode == EMochiConstraintComponentMode::TwoBody) {
    // find component 1 and component 2
    if (ComponentName1.IsNone() || ComponentName2.IsNone()) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiSoftConstraintComponent] ComponentName1 and/or ComponentName 2 is None!"))
      return;
    }
    if (ComponentName1 == ComponentName2) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiSoftConstraintComponent] ComponentName1 and ComponentName2 must be unique!"))
      return;
    }
    Component1 = FindComponentRecursive(this, ComponentName1);
    if (Component1 == nullptr) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiSoftConstraintComponent] Failed to find Component1 %s!"),
          *ComponentName1.ToString());
      return;
    }
    Component2 = FindComponentRecursive(this, ComponentName2);
    if (Component2 == nullptr) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiSoftConstraintComponent] Failed to find Component2 %s!"),
          *ComponentName2.ToString());
      return;
    }
    // ensure component 1 and component 2 mochi actors created
    if (EnsureMochiActor(this, Component1) && EnsureMochiActor(this, Component2)) {
      // create constraints
      TryCreateConstraints_TwoBody();
    }
  }
}

void UMochiSoftConstraintComponent::DestroyConstraints() {
  if (bConstraintsCreated == true && MochiSceneObject.IsValid()) {
    // Pass ownership of MochiThreadState to the lambda so we don't have to wait
    MochiSceneObject->QueueOrExecuteCommand([&bConstraintsCreated = bConstraintsCreated,
                                             MochiThreadState =
                                                 std::move(MochiThreadState)](mochi::Scene* Scene) {
      for (auto& handle : MochiThreadState->ConstraintHandles) {
        Scene->DestroyConstraint(handle);
      }
      // clear boundary conditions on component 1 actor (for one body case)
      if (const auto deformableActor = Scene->GetActor(MochiThreadState->DeformableActorHandle)) {
        deformableActor->ClearBoundaryConditions();
      }
      bConstraintsCreated = false;
    });
    MochiSceneObject->WaitForQueuedCommands();
    MochiThreadState = std::make_shared<MochiState>();
  }
}
