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
    MochiSceneObject->QueueOrExecuteCommand(
        [NewStiffness, state = MochiThreadState](mochi::Scene* Scene) {
          if (auto* constraint = Scene->GetConstraint(state->PositionConstraintHandle)) {
            constraint->SetStiffness(NewStiffness, mochi::ErrorLog{});
          }
        });
  }
}

void UMochiRodConstraintComponent::SetPositionDamping(float NewDamping) {
  PositionDamping = NewDamping;
  if (MochiSceneObject.IsValid()) {
    MochiSceneObject->QueueOrExecuteCommand(
        [NewDamping, state = MochiThreadState](mochi::Scene* Scene) {
          if (auto* constraint = Scene->GetConstraint(state->PositionConstraintHandle)) {
            constraint->SetDamping(NewDamping, mochi::ErrorLog{});
          }
        });
  }
}

void UMochiRodConstraintComponent::SetRotationStiffness(float NewStiffness) {
  RotationStiffness = NewStiffness;
  if (MochiSceneObject.IsValid()) {
    MochiSceneObject->QueueOrExecuteCommand(
        [NewStiffness, state = MochiThreadState](mochi::Scene* Scene) {
          if (auto* constraint = Scene->GetConstraint(state->RotationConstraintHandle)) {
            constraint->SetStiffness(NewStiffness, mochi::ErrorLog{});
          }
        });
  }
}

void UMochiRodConstraintComponent::SetRotationDamping(float NewDamping) {
  RotationDamping = NewDamping;
  if (MochiSceneObject.IsValid()) {
    MochiSceneObject->QueueOrExecuteCommand(
        [NewDamping, state = MochiThreadState](mochi::Scene* Scene) {
          if (auto* constraint = Scene->GetConstraint(state->RotationConstraintHandle)) {
            constraint->SetDamping(NewDamping, mochi::ErrorLog{});
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
  auto* RodComponent = FindComponentRecursive(this, RodComponentName);
  if (RodComponent == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiRodConstraintComponent] Failed to find RodComponent [%s]!"),
        *RodComponentName.ToString());
    return;
  }
  auto* RigidComponent = FindComponentRecursive(this, RigidComponentName);
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
  const auto rodMochiMesh = Cast<IMochiMeshComponentInterface>(RodComponent);
  const auto rigidMochiMesh = Cast<IMochiMeshComponentInterface>(RigidComponent);
  if (rodMochiMesh == nullptr || rigidMochiMesh == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiRodConstraintComponent] RodComponent or RigidComponent does not implement IMochiMeshComponentInterface!"));
    return;
  }
  const auto rodActorObj = rodMochiMesh->GetMochiActor();
  const auto rigidActorObj = rigidMochiMesh->GetMochiActor();
  if (rodActorObj->GetMochiScene() != rigidActorObj->GetMochiScene()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiRodConstraintComponent] RodComponent and RigidComponent are not in the same Mochi scene."));
    return;
  }
  MochiSceneObject = rodActorObj->GetMochiScene();
  const auto rodHandle = rodActorObj->GetMochiActorHandle();
  const auto rigidHandle = rigidActorObj->GetMochiActorHandle();

  // Capture parameters by value into lambda
  MochiSceneObject->QueueOrExecuteCommand([rodHandle,
                                           rigidHandle,
                                           attachEnd = AttachmentEnd,
                                           bConstrainPosition = this->bConstrainPosition,
                                           bConstrainRotation = this->bConstrainRotation,
                                           posStiffness = this->PositionStiffness,
                                           posDamping = this->PositionDamping,
                                           rotStiffness = this->RotationStiffness,
                                           rotDamping = this->RotationDamping,
                                           allowContact = this->bAllowContact,
                                           state = MochiThreadState](mochi::Scene* Scene) {
    mochi::ErrorLog e;
    auto* rodActor = Scene->GetActor(rodHandle);
    auto* rigidActor = Scene->GetActor(rigidHandle);

    // Guard against actors that were destroyed between queuing and execution.
    if (rodActor == nullptr || rigidActor == nullptr) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT(
              "[UMochiRodConstraintComponent] Rod or Rigid actor no longer exists; skipping constraint creation."));
      return;
    }

    // Validate actor types
    if (rodActor->GetType() != mochi::ActorType::Rod) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiRodConstraintComponent] RodComponent actor is not of type Rod!"));
      return;
    }
    if (rigidActor->GetType() != mochi::ActorType::Rigid) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiRodConstraintComponent] RigidComponent actor is not of type Rigid!"));
      return;
    }

    const mochi::MeshDataView rodMesh = rodActor->GetMesh();

    // Compute indices based on attachment end
    const int elementIndex =
        (attachEnd == EMochiRodAttachmentEnd::First) ? 0 : rodMesh.GetNumElements() - 1;
    const int nodeIndex =
        (attachEnd == EMochiRodAttachmentEnd::First) ? 0 : rodMesh.GetNumNodes() - 1;

    // Position constraint (DeformableNodeToRigid)
    if (bConstrainPosition) {
      mochi::DeformableNodeToRigidConstraintParams params;
      params.deformableActor = rodHandle;
      params.rigidActor = rigidHandle;
      params.deformableNodeIndex = nodeIndex;
      params.fixToDeformablePos = true;
      params.stiffness = posStiffness;
      params.damping = posDamping;
      auto* constraint = Scene->CreateDeformableNodeToRigidConstraint(params, e);
      if (e.IsOK() && constraint) {
        state->PositionConstraintHandle = constraint->GetHandle();
      }
    }

    // Rotation constraint (RodElementRotationToRigid)
    if (bConstrainRotation) {
      mochi::RodElementRotationToRigidConstraintParams params;
      params.rodActor = rodHandle;
      params.rigidActor = rigidHandle;
      params.elementIndex = elementIndex;
      params.stiffness = rotStiffness;
      params.damping = rotDamping;
      auto* constraint = Scene->CreateRodElementRotationToRigidConstraint(params, e);
      if (e.IsOK() && constraint) {
        state->RotationConstraintHandle = constraint->GetHandle();
      }
    }

    // Optionally disable contact between the pair
    if (!allowContact) {
      Scene->EnableActorContactSymmetric(
          rodHandle, rigidHandle, /*enable*/ false, mochi::IncludeNestedActors::No, e);
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
        [state = std::move(MochiThreadState)](mochi::Scene* Scene) {
          auto TryDestroyConstraint = [Scene](mochi::ConstraintHandle& Handle) {
            if (Handle.IsValid()) {
              Scene->DestroyConstraint(Handle);
            }
            Handle = {};
          };
          TryDestroyConstraint(state->PositionConstraintHandle);
          TryDestroyConstraint(state->RotationConstraintHandle);
        });
    MochiSceneObject->WaitForQueuedCommands();
  }
  MochiThreadState = std::make_shared<MochiState>();
  bConstraintsCreated = false;
}
