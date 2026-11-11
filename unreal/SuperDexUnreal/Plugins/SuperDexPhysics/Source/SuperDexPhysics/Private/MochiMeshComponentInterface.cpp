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
  if (const auto mochiActorObj = GetMochiActor()) {
    mochiActorObj->SetRootTransform(NewTransform);
  }
}

void IMochiMeshComponentInterface::SetWorldLocationAndRotationMochi(
    FVector NewLocation,
    FRotator NewRotation) {
  FTransform transform;
  transform.SetLocation(NewLocation);
  transform.SetScale3D(FVector::OneVector);
  transform.SetRotation(NewRotation.Quaternion());
  SetWorldTransformMochi(transform);
}

void IMochiMeshComponentInterface::SetContactParams(const UMochiContactParams* ContactParams) {
  if (const auto mochiActorObj = GetMochiActor()) {
    mochiActorObj->SetContactParams(ContactParams);
  }
}

void IMochiMeshComponentInterface::SetMaterialParams(const UMochiMaterialParams* MaterialParams) {
  if (const auto mochiActorObj = GetMochiActor()) {
    mochiActorObj->SetMaterialParams(MaterialParams);
  }
}

void IMochiMeshComponentInterface::ClearBoundaryConditions() {
  if (const auto mochiActorObj = GetMochiActor()) {
    auto handle = mochiActorObj->GetMochiActorHandle();
    GetMochiActor()->GetMochiScene()->QueueOrExecuteActorCommand(
        handle, [](mochi::Actor* Actor) { Actor->ClearBoundaryConditions(); });
  }
}

void IMochiMeshComponentInterface::SetZeroDeformation() const {
  auto handle = GetMochiActor()->GetMochiActorHandle();
  GetMochiActor()->GetMochiScene()->QueueOrExecuteActorCommand(handle, [](mochi::Actor* Actor) {
    mochi::ErrorLog e{};
    Actor->ClearBoundaryConditions();
    Actor->SetZeroDisplacementsAndVelocities(e);
  });
}

void IMochiMeshComponentInterface::SetVelocities(FVector LinearVelocity, FVector AngularVelocity)
    const {
  if (const auto mochiActorObj = GetMochiActor()) {
    mochiActorObj->SetVelocity(LinearVelocity, AngularVelocity);
  }
}

void IMochiMeshComponentInterface::SetZeroVelocities() {
  if (const auto mochiActorObj = GetMochiActor()) {
    mochiActorObj->SetVelocity(FVector::ZeroVector, FVector::ZeroVector);
  }
}

void IMochiMeshComponentInterface::GetInertialParams(
    double& Mass,
    FVector& CenterOfMass,
    FMochiMomentOfInertia& MomentOfInertia) {
  if (const auto mochiActorObj = GetMochiActor()) {
    mochiActorObj->GetInertialParams(Mass, CenterOfMass, MomentOfInertia);
  }
}
