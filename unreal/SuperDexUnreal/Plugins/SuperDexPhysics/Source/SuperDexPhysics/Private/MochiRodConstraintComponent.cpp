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

#include "MochiRodConstraintComponent.h"
#include "MochiConstraintUtil.h"
#include "MochiPhysicsModule.h"
#include "MochiSceneObject.h"

UMochiRodConstraintComponent::UMochiRodConstraintComponent() {
  PrimaryComponentTick.bCanEverTick = false;
}

void UMochiRodConstraintComponent::BeginPlay() {
  Super::BeginPlay();
  CreateConstraints();
}

void UMochiRodConstraintComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  Super::EndPlay(EndPlayReason);
  DestroyConstraints();
}

void UMochiRodConstraintComponent::SetPositionStiffness(float NewStiffness) {
  PositionStiffness = NewStiffness;
  if (MochiSceneObject.IsValid()) {
    MochiSceneObject->QueueOrExecuteCommand([NewStiffness,
                                             State = MochiThreadState](mochi::Scene* Scene) {
      if (mochi::Constraint* Constraint = Scene->GetConstraint(State->PositionConstraintHandle)) {
        Constraint->SetStiffness(NewStiffness, mochi::ErrorLog{});
      }
    });
  }
}

void UMochiRodConstraintComponent::SetPositionDamping(float NewDamping) {
  PositionDamping = NewDamping;
  if (MochiSceneObject.IsValid()) {
    MochiSceneObject->QueueOrExecuteCommand([NewDamping,
                                             State = MochiThreadState](mochi::Scene* Scene) {
      if (mochi::Constraint* Constraint = Scene->GetConstraint(State->PositionConstraintHandle)) {
        Constraint->SetDamping(NewDamping, mochi::ErrorLog{});
      }
    });
  }
}

void UMochiRodConstraintComponent::SetRotationStiffness(float NewStiffness) {
  RotationStiffness = NewStiffness;
  if (MochiSceneObject.IsValid()) {
    MochiSceneObject->QueueOrExecuteCommand([NewStiffness,
                                             State = MochiThreadState](mochi::Scene* Scene) {
      if (mochi::Constraint* Constraint = Scene->GetConstraint(State->RotationConstraintHandle)) {
        Constraint->SetStiffness(NewStiffness, mochi::ErrorLog{});
      }
    });
  }
}

void UMochiRodConstraintComponent::SetRotationDamping(float NewDamping) {
  RotationDamping = NewDamping;
  if (MochiSceneObject.IsValid()) {
    MochiSceneObject->QueueOrExecuteCommand([NewDamping,
                                             State = MochiThreadState](mochi::Scene* Scene) {
      if (mochi::Constraint* Constraint = Scene->GetConstraint(State->RotationConstraintHandle)) {
        Constraint->SetDamping(NewDamping, mochi::ErrorLog{});
      }
    });
  }
}

void UMochiRodConstraintComponent::CreateConstraints() {
  if (bConstraintsCreated) {
    DestroyConstraints();
  }

  // Validate component names
  if (RodComponentName.IsNone() || RigidComponentName.IsNone()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiRodConstraintComponent] RodComponentName and/or RigidComponentName is None!"));
    return;
  }

  // Find components
  USceneComponent* RodComponent = FindComponentRecursive(this, RodComponentName);
  if (RodComponent == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiRodConstraintComponent] Failed to find RodComponent [%s]!"),
        *RodComponentName.ToString());
    return;
  }
  USceneComponent* RigidComponent = FindComponentRecursive(this, RigidComponentName);
  if (RigidComponent == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiRodConstraintComponent] Failed to find RigidComponent [%s]!"),
        *RigidComponentName.ToString());
    return;
  }

  // Ensure Mochi actors exist
  if (!EnsureMochiActor(this, RodComponent) || !EnsureMochiActor(this, RigidComponent)) {
    return;
  }

  // Get Mochi actor objects. EnsureMochiActor above guarantees both components implement
  // IMochiMeshComponentInterface, so these casts are expected to succeed. Guard defensively
  // in case that invariant ever changes.
  IMochiMeshComponentInterface* const RodMochiMesh =
      Cast<IMochiMeshComponentInterface>(RodComponent);
  IMochiMeshComponentInterface* const RigidMochiMesh =
      Cast<IMochiMeshComponentInterface>(RigidComponent);
  if (RodMochiMesh == nullptr || RigidMochiMesh == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiRodConstraintComponent] RodComponent or RigidComponent does not implement IMochiMeshComponentInterface!"));
    return;
  }
  UMochiActorObject* const RodActorObj = RodMochiMesh->GetMochiActor();
  UMochiActorObject* const RigidActorObj = RigidMochiMesh->GetMochiActor();
  if (RodActorObj->GetMochiScene() != RigidActorObj->GetMochiScene()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiRodConstraintComponent] RodComponent and RigidComponent are not in the same Mochi scene."));
    return;
  }
  MochiSceneObject = RodActorObj->GetMochiScene();
  const mochi::ActorHandle RodHandle = RodActorObj->GetMochiActorHandle();
  const mochi::ActorHandle RigidHandle = RigidActorObj->GetMochiActorHandle();

  // Capture parameters by value into lambda
  MochiSceneObject->QueueOrExecuteCommand([RodHandle,
                                           RigidHandle,
                                           AttachEnd = AttachmentEnd,
                                           bConstrainPosition = this->bConstrainPosition,
                                           bConstrainRotation = this->bConstrainRotation,
                                           PosStiffness = this->PositionStiffness,
                                           PosDamping = this->PositionDamping,
                                           RotStiffness = this->RotationStiffness,
                                           RotDamping = this->RotationDamping,
                                           bAllowContact = this->bAllowContact,
                                           State = MochiThreadState](mochi::Scene* Scene) {
    mochi::ErrorLog E;
    mochi::Actor* RodActor = Scene->GetActor(RodHandle);
    mochi::Actor* RigidActor = Scene->GetActor(RigidHandle);

    // Guard against actors that were destroyed between queuing and execution.
    if (RodActor == nullptr || RigidActor == nullptr) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT(
              "[UMochiRodConstraintComponent] Rod or Rigid actor no longer exists; skipping constraint creation."));
      return;
    }

    // Validate actor types
    if (RodActor->GetType() != mochi::ActorType::Rod) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiRodConstraintComponent] RodComponent actor is not of type Rod!"));
      return;
    }
    if (RigidActor->GetType() != mochi::ActorType::Rigid) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiRodConstraintComponent] RigidComponent actor is not of type Rigid!"));
      return;
    }

    const mochi::MeshDataView RodMesh = RodActor->GetMesh();

    // Compute indices based on attachment end
    const int ElementIndex =
        (AttachEnd == EMochiRodAttachmentEnd::First) ? 0 : RodMesh.GetNumElements() - 1;
    const int NodeIndex =
        (AttachEnd == EMochiRodAttachmentEnd::First) ? 0 : RodMesh.GetNumNodes() - 1;

    // Position constraint (DeformableNodeToRigid)
    if (bConstrainPosition) {
      mochi::DeformableNodeToRigidConstraintParams Params;
      Params.deformableActor = RodHandle;
      Params.rigidActor = RigidHandle;
      Params.deformableNodeIndex = NodeIndex;
      Params.stiffness = PosStiffness;
      Params.damping = PosDamping;
      mochi::Constraint* Constraint = Scene->CreateDeformableNodeToRigidConstraint(Params, E);
      if (E.IsOK() && Constraint) {
        State->PositionConstraintHandle = Constraint->GetHandle();
      }
    }

    // Rotation constraint (RodElementRotationToRigid)
    if (bConstrainRotation) {
      mochi::RodElementRotationToRigidConstraintParams Params;
      Params.rodActor = RodHandle;
      Params.rigidActor = RigidHandle;
      Params.elementIndex = ElementIndex;
      Params.stiffness = RotStiffness;
      Params.damping = RotDamping;
      mochi::Constraint* Constraint = Scene->CreateRodElementRotationToRigidConstraint(Params, E);
      if (E.IsOK() && Constraint) {
        State->RotationConstraintHandle = Constraint->GetHandle();
      }
    }

    // Optionally disable contact between the pair
    if (!bAllowContact) {
      Scene->EnableActorContactSymmetric(
          RodHandle, RigidHandle, /*enable*/ false, mochi::IncludeNestedActors::No, E);
    }
  });

  bConstraintsCreated = true;
}

void UMochiRodConstraintComponent::DestroyConstraints() {
  if (!bConstraintsCreated) {
    return;
  }
  // If the scene is still alive, queue destruction of the constraints.
  if (MochiSceneObject.IsValid()) {
    MochiSceneObject->QueueOrExecuteCommand(
        [State = std::move(MochiThreadState)](mochi::Scene* Scene) {
          auto TryDestroyConstraint = [Scene](mochi::ConstraintHandle& Handle) {
            if (Handle.IsValid()) {
              Scene->DestroyConstraint(Handle);
            }
            Handle = {};
          };
          TryDestroyConstraint(State->PositionConstraintHandle);
          TryDestroyConstraint(State->RotationConstraintHandle);
        });
    MochiSceneObject->WaitForQueuedCommands();
  }
  MochiThreadState = std::make_shared<FMochiState>();
  bConstraintsCreated = false;
}
