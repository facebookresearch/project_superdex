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

#include "MochiMeshComponentInterface.h"

#include "MochiConversions.h"

void IMochiMeshComponentInterface::SetWorldTransformMochi(const FTransform& NewTransform) {
  if (UMochiActorObject* const MochiActorObj = GetMochiActor()) {
    MochiActorObj->SetRootTransform(NewTransform);
  }
}

void IMochiMeshComponentInterface::SetWorldLocationAndRotationMochi(
    FVector NewLocation,
    FRotator NewRotation) {
  FTransform Transform;
  Transform.SetLocation(NewLocation);
  Transform.SetScale3D(FVector::OneVector);
  Transform.SetRotation(NewRotation.Quaternion());
  SetWorldTransformMochi(Transform);
}

void IMochiMeshComponentInterface::SetContactParams(const UMochiContactParams* ContactParams) {
  if (UMochiActorObject* const MochiActorObj = GetMochiActor()) {
    MochiActorObj->SetContactParams(ContactParams);
  }
}

void IMochiMeshComponentInterface::SetMaterialParams(const UMochiMaterialParams* MaterialParams) {
  if (UMochiActorObject* const MochiActorObj = GetMochiActor()) {
    MochiActorObj->SetMaterialParams(MaterialParams);
  }
}

void IMochiMeshComponentInterface::ClearBoundaryConditions() {
  if (UMochiActorObject* const MochiActorObj = GetMochiActor()) {
    mochi::ActorHandle Handle = MochiActorObj->GetMochiActorHandle();
    GetMochiActor()->GetMochiScene()->QueueOrExecuteActorCommand(
        Handle, [](mochi::Actor* Actor) { Actor->ClearBoundaryConditions(); });
  }
}

void IMochiMeshComponentInterface::SetZeroDeformation() const {
  mochi::ActorHandle Handle = GetMochiActor()->GetMochiActorHandle();
  GetMochiActor()->GetMochiScene()->QueueOrExecuteActorCommand(Handle, [](mochi::Actor* Actor) {
    mochi::ErrorLog E{};
    Actor->ClearBoundaryConditions();
    Actor->SetZeroDisplacementsAndVelocities(E);
  });
}

void IMochiMeshComponentInterface::SetVelocities(FVector LinearVelocity, FVector AngularVelocity)
    const {
  if (UMochiActorObject* const MochiActorObj = GetMochiActor()) {
    MochiActorObj->SetVelocity(LinearVelocity, AngularVelocity);
  }
}

void IMochiMeshComponentInterface::SetZeroVelocities() {
  if (UMochiActorObject* const MochiActorObj = GetMochiActor()) {
    MochiActorObj->SetVelocity(FVector::ZeroVector, FVector::ZeroVector);
  }
}

void IMochiMeshComponentInterface::GetInertialParams(
    double& Mass,
    FVector& CenterOfMass,
    FMochiMomentOfInertia& MomentOfInertia) {
  if (UMochiActorObject* const MochiActorObj = GetMochiActor()) {
    MochiActorObj->GetInertialParams(Mass, CenterOfMass, MomentOfInertia);
  }
}
