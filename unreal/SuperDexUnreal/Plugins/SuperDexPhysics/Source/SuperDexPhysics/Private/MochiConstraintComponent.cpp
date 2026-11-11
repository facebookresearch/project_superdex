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

#include "MochiConstraintComponent.h"
#include "MochiConstraintUtil.h"
#include "MochiConversions.h"
#include "MochiPhysicsModule.h"

// TODO: UMochiConstraintComponent should be notified when either the constrained component
// or constrained component's underlying Mochi actor have been destroyed so that it can
// destroy the Mochi constraints. IMochiMeshComponentInterface could possibly expose
// an OnMochiActorCreated event.

void UMochiConstraintComponent::SetLinearStiffness(float Stiffness) {
  LinearStiffness = Stiffness;
  if (MochiSceneObject.IsValid()) {
    MochiSceneObject->QueueOrExecuteCommand(
        [Stiffness, state = MochiThreadState](mochi::Scene* Scene) {
          // one-body case
          if (auto* constraint = Scene->GetConstraint(state->RigidPivotPositionConstraintHandle)) {
            constraint->SetStiffness(Stiffness, mochi::ErrorLog{});
          }

          // two-body case
          if (auto* constraint = Scene->GetConstraint(state->RigidSphericalJointConstraintHandle)) {
            constraint->SetStiffness(Stiffness, mochi::ErrorLog{});
          }
        });
  }
}

void UMochiConstraintComponent::SetLinearDamping(float Damping) {
  LinearDamping = Damping;
  if (MochiSceneObject.IsValid()) {
    MochiSceneObject->QueueOrExecuteCommand(
        [Damping, state = MochiThreadState](mochi::Scene* Scene) {
          // one-body case
          if (auto* constraint = Scene->GetConstraint(state->RigidPivotPositionConstraintHandle)) {
            constraint->SetDamping(Damping, mochi::ErrorLog{});
          }

          // two-body case
          if (auto* constraint = Scene->GetConstraint(state->RigidSphericalJointConstraintHandle)) {
            constraint->SetDamping(Damping, mochi::ErrorLog{});
          }
        });
  }
}

void UMochiConstraintComponent::SetTorsionalStiffness(float Stiffness) {
  TorsionalStiffness = Stiffness;
  if (MochiSceneObject.IsValid()) {
    MochiSceneObject->QueueOrExecuteCommand([Stiffness,
                                             state = MochiThreadState](mochi::Scene* Scene) {
      // one-body case
      if (auto* constraint = Scene->GetConstraint(state->RigidPivotRotationConstraintHandle)) {
        constraint->SetStiffness(Stiffness, mochi::ErrorLog{});
      }

      // two-body case
      if (auto* constraint = Scene->GetConstraint(state->JointRotationTrackingConstraintHandle)) {
        constraint->SetStiffness(Stiffness, mochi::ErrorLog{});
      }
    });
  }
}

void UMochiConstraintComponent::SetTorsionalDamping(float Damping) {
  TorsionalDamping = Damping;
  if (MochiSceneObject.IsValid()) {
    MochiSceneObject->QueueOrExecuteCommand([Damping,
                                             state = MochiThreadState](mochi::Scene* Scene) {
      // one-body case
      if (auto* constraint = Scene->GetConstraint(state->RigidPivotRotationConstraintHandle)) {
        constraint->SetDamping(Damping, mochi::ErrorLog{});
      }

      // two-body case
      if (auto* constraint = Scene->GetConstraint(state->JointRotationTrackingConstraintHandle)) {
        constraint->SetDamping(Damping, mochi::ErrorLog{});
      }
    });
  }
}

UMochiConstraintComponent::UMochiConstraintComponent() {
  PrimaryComponentTick.bCanEverTick = true;
  bWantsOnUpdateTransform = true;
}

void UMochiConstraintComponent::BeginPlay() {
  Super::BeginPlay();
  CreateConstraints();
}

void UMochiConstraintComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  Super::EndPlay(EndPlayReason);
  DestroyConstraints();
}

void UMochiConstraintComponent::OnUpdateTransform(
    EUpdateTransformFlags UpdateTransformFlags,
    ETeleportType Teleport) {
  Super::OnUpdateTransform(UpdateTransformFlags, Teleport);
  if (MochiSceneObject.IsValid()) {
    FUnrealToMochiConverter u2m;
    const auto location = GetComponentLocation();
    const auto rotation = GetComponentRotation().Quaternion();
    MochiSceneObject->QueueOrExecuteCommand(
        [u2m, location, rotation, MochiThreadState = this->MochiThreadState](mochi::Scene* Scene) {
          if (auto* constraint =
                  Scene->GetConstraint(MochiThreadState->RigidPivotPositionConstraintHandle)) {
            constraint->SetTargetPosition(u2m.TranslationToMochi(location), mochi::ErrorLog{});
          }
          if (auto* constraint =
                  Scene->GetConstraint(MochiThreadState->RigidPivotRotationConstraintHandle)) {
            constraint->SetTargetRotation(u2m.RotationToMochi(rotation), mochi::ErrorLog{});
          }
        });
  }
}

void UMochiConstraintComponent::TickComponent(
    float DeltaTime,
    ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction) {
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
}

void UMochiConstraintComponent::TryCreateConstraints_OneBody() {
  FUnrealToMochiConverter u2m;
  const auto mochiMeshComp1 = Cast<IMochiMeshComponentInterface>(Component1);
  const auto mochiActor = mochiMeshComp1->GetMochiActor();
  MochiSceneObject = mochiActor->GetMochiScene();
  const auto actorHandle = mochiActor->GetMochiActorHandle();

  const FVector componentLocalLocation = bOverrideComponent1ConnectionPoint
      ? Component1ConnectionPointOverride
      : Component1->GetComponentTransform().InverseTransformPositionNoScale(GetComponentLocation());

  const FVector componentFrameRotationVector =
      Component1->GetComponentTransform()
          .InverseTransformRotation(GetComponentRotation().Quaternion())
          .GetNormalized()
          .ToRotationVector();

  // constrain location with RigidPivotPositionParams
  mochi::RigidPivotPositionConstraintParams locationParams;
  if (bConstrainLocation) {
    locationParams.actor = actorHandle;
    locationParams.localPosition = u2m.TranslationToMochi(componentLocalLocation);
    locationParams.targetPosition = u2m.TranslationToMochi(GetComponentLocation());
    locationParams.stiffness = LinearStiffness;
    locationParams.damping = LinearDamping;
  }

  // constrain rotation with RigidPivotRotationParams
  mochi::RigidPivotRotationConstraintParams rotationParams;
  if (bConstrainRotation) {
    rotationParams.actor = actorHandle;
    rotationParams.localRotation = u2m.DirectionToMochi(componentFrameRotationVector);
    rotationParams.targetRotation =
        u2m.RotationToMochi(GetComponentRotation().Quaternion()).ToRotationVector();
    rotationParams.stiffness = TorsionalStiffness * TorsionalGain;
    rotationParams.damping = TorsionalDamping * TorsionalGain;
  }

  // Capture by value so we don't have to wait
  MochiSceneObject->QueueOrExecuteCommand(
      [locationParams,
       rotationParams,
       bConstrainLocation = this->bConstrainLocation,
       bConstrainRotation = this->bConstrainRotation,
       MochiThreadState = this->MochiThreadState](mochi::Scene* Scene) {
        mochi::ErrorLog e;
        // constrain location with RigidPivotPositionParams
        if (bConstrainLocation) {
          auto* constraint = Scene->CreateRigidPivotPositionConstraint(locationParams, e);
          auto constraintHandle = constraint ? constraint->GetHandle() : mochi::ConstraintHandle{};
          MochiThreadState->RigidPivotPositionConstraintHandle = constraintHandle;
        }
        // constrain rotation with RigidPivotRotationParams
        if (bConstrainRotation) {
          auto* constraint = Scene->CreateRigidPivotRotationConstraint(rotationParams, e);
          auto constraintHandle = constraint ? constraint->GetHandle() : mochi::ConstraintHandle{};
          MochiThreadState->RigidPivotRotationConstraintHandle = constraintHandle;
        }
      });
}

void UMochiConstraintComponent::TryCreateConstraints_TwoBody() {
  FUnrealToMochiConverter u2m;
  const auto mochiMeshComp1 = Cast<IMochiMeshComponentInterface>(Component1);
  const auto mochiMeshComp2 = Cast<IMochiMeshComponentInterface>(Component2);
  const auto mochiActor1 = mochiMeshComp1->GetMochiActor();
  const auto mochiActor2 = mochiMeshComp2->GetMochiActor();
  if (mochiActor1->GetMochiScene() != mochiActor2->GetMochiScene()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiConstraintComponent] Component1 and Component2 are not in the same Mochi scene."));
    return;
  }
  MochiSceneObject = mochiActor1->GetMochiScene();
  const auto actorHandle1 = mochiActor1->GetMochiActorHandle();
  const auto actorHandle2 = mochiActor2->GetMochiActorHandle();

  const FVector componentLocalLocation1 = bOverrideComponent1ConnectionPoint
      ? Component1ConnectionPointOverride
      : Component1->GetComponentTransform().InverseTransformPositionNoScale(GetComponentLocation());

  const FVector componentFrameRotationVector1 =
      Component1->GetComponentTransform()
          .InverseTransformRotation(GetComponentRotation().Quaternion())
          .GetNormalized()
          .ToRotationVector();
  const FVector componentLocalLocation2 = bOverrideComponent2ConnectionPoint
      ? Component2ConnectionPointOverride
      : Component2->GetComponentTransform().InverseTransformPositionNoScale(GetComponentLocation());

  const FVector componentFrameRotationVector2 =
      Component2->GetComponentTransform()
          .InverseTransformRotation(GetComponentRotation().Quaternion())
          .GetNormalized()
          .ToRotationVector();
  const FVector refFrameRotationVector =
      GetComponentRotation().Quaternion().GetNormalized().ToRotationVector();

  // Location Constraint Params
  mochi::RigidPrismaticJointConstraintParams paramsRPJ;
  mochi::RigidSphericalJointConstraintParams paramsRSJ;
  bool useRPJ = false;
  bool useRSJ = false;
  if (bConstrainLocation) {
    // prismatic joint case
    if (FreeAxis != EAxis::None) {
      useRPJ = true;
      paramsRPJ.actorA = actorHandle1;
      paramsRPJ.actorB = actorHandle2;
      switch (FreeAxis) {
        case EAxis::X:
          paramsRPJ.freeAxis = u2m.TranslationToMochi(GetForwardVector());
          break;
        case EAxis::Y:
          paramsRPJ.freeAxis = u2m.TranslationToMochi(GetRightVector());
          break;
        case EAxis::Z:
          paramsRPJ.freeAxis = u2m.TranslationToMochi(GetUpVector());
          break;
        default:
          break;
      }
      paramsRPJ.stiffness = LinearStiffness;
      paramsRPJ.damping = LinearDamping;
      if (bLimitFreeAxis) {
        paramsRPJ.min = u2m.MeasureToMochi(FreeAxisRange.GetLowerBoundValue());
        paramsRPJ.max = u2m.MeasureToMochi(FreeAxisRange.GetUpperBoundValue());
      }
      // create additional spherical joint if user wants spring free-axis
      if (FreeAxisLinearStiffness != 0 || FreeAxisLinearDamping != 0) {
        useRSJ = true;
        paramsRSJ.actorA = actorHandle1;
        paramsRSJ.actorB = actorHandle2;
        paramsRSJ.stiffness = FreeAxisLinearStiffness;
        paramsRSJ.damping = FreeAxisLinearDamping;
        paramsRSJ.localPosA = u2m.TranslationToMochi(componentLocalLocation1);
        paramsRSJ.localPosB = u2m.TranslationToMochi(componentLocalLocation2);
      }
    }
    // spherical joint case
    else {
      useRSJ = true;
      paramsRSJ.actorA = actorHandle1;
      paramsRSJ.actorB = actorHandle2;
      paramsRSJ.stiffness = LinearStiffness;
      paramsRSJ.damping = LinearDamping;
      paramsRSJ.localPosA = u2m.TranslationToMochi(componentLocalLocation1);
      paramsRSJ.localPosB = u2m.TranslationToMochi(componentLocalLocation2);
    }
  }

  // Rotation constraint params
  mochi::JointRotationTrackingConstraintParams paramsJRT;
  bool useJRT = false;
  if (bConstrainRotation) {
    useJRT = true;
    paramsJRT.actorA = actorHandle1;
    paramsJRT.actorB = actorHandle2;
    paramsJRT.stiffness = TorsionalStiffness * TorsionalGain;
    paramsJRT.damping = TorsionalDamping * TorsionalGain;
    paramsJRT.refFrameRotVec = u2m.DirectionToMochi(refFrameRotationVector);
  }

  // Rotation range constraint params
  mochi::JointRotationRangeConstraintParams paramsJRR;
  bool useJRR = false;
  if (bLimitRotation) {
    useJRR = true;
    paramsJRR.actorA = actorHandle1;
    paramsJRR.actorB = actorHandle2;
    paramsJRR.refFrameRotVec = u2m.DirectionToMochi(refFrameRotationVector);
    paramsJRR.angleRangeX = {
        FMath::DegreesToRadians(AngleRangeX.GetLowerBoundValue()),
        FMath::DegreesToRadians(AngleRangeX.GetUpperBoundValue())};
    paramsJRR.angleRangeY = {
        FMath::DegreesToRadians(AngleRangeY.GetLowerBoundValue()),
        FMath::DegreesToRadians(AngleRangeY.GetUpperBoundValue())};
    paramsJRR.angleRangeZ = {
        FMath::DegreesToRadians(AngleRangeZ.GetLowerBoundValue()),
        FMath::DegreesToRadians(AngleRangeZ.GetUpperBoundValue())};
    paramsJRR.stiffness = RotationLimitStiffness;
    paramsJRR.damping = RotationLimitDamping;
  }

  // Capture by value so we don't have to wait
  MochiSceneObject->QueueOrExecuteCommand(
      [actorHandle1,
       actorHandle2,
       useRPJ,
       useRSJ,
       useJRT,
       useJRR,
       paramsRPJ,
       paramsRSJ,
       paramsJRT,
       paramsJRR,
       AllowContact = this->AllowContact,
       MochiThreadState = this->MochiThreadState](mochi::Scene* Scene) {
        mochi::ErrorLog e;
        std::vector<mochi::ConstraintHandle> newConstraints;
        newConstraints.reserve(4);
        if (useRPJ) {
          auto* constraint = Scene->CreateRigidPrismaticJointConstraint(paramsRPJ, e);
          if (e.IsOK()) {
            MochiThreadState->RigidPrismaticJointConstraintHandle = constraint->GetHandle();
            newConstraints.push_back(constraint->GetHandle());
          }
        }
        if (useRSJ) {
          auto* constraint = Scene->CreateRigidSphericalJointConstraint(paramsRSJ, e);
          if (e.IsOK()) {
            MochiThreadState->RigidSphericalJointConstraintHandle = constraint->GetHandle();
            newConstraints.push_back(constraint->GetHandle());
          }
        }
        if (useJRT) {
          auto* constraint = Scene->CreateJointRotationTrackingConstraint(paramsJRT, e);
          if (e.IsOK()) {
            MochiThreadState->JointRotationTrackingConstraintHandle = constraint->GetHandle();
            newConstraints.push_back(constraint->GetHandle());
          }
        }
        if (useJRR) {
          auto* constraint = Scene->CreateJointRotationRangeConstraint(paramsJRR, e);
          if (e.IsOK()) {
            MochiThreadState->JointRotationRangeConstraintHandle = constraint->GetHandle();
            newConstraints.push_back(constraint->GetHandle());
          }
        }
        if (!AllowContact) {
          // Prevent contact in both directions
          Scene->EnableActorContactSymmetric(
              actorHandle1,
              actorHandle2,
              /*enable*/ false,
              mochi::IncludeNestedActors::No,
              e);
        }
      });
}

void UMochiConstraintComponent::CreateConstraints() {
  // destroy existing constraints
  if (bConstraintsCreated) {
    DestroyConstraints();
  }
  // one-body case
  if (Mode == EMochiConstraintComponentMode::OneBody) {
    // find component 1
    if (ComponentName1.IsNone() && Component1 == nullptr) {
      UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiConstraintComponent] ComponentName1 is None!"))
      return;
    }

    if (Component1 == nullptr) {
      Component1 = FindComponentRecursive(this, ComponentName1);
    }
    if (Component1 == nullptr) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiConstraintComponent] Failed to find Component1 [%s]!"),
          *ComponentName1.ToString());
      return;
    }
    // ensure component 1 mochi actor created
    if (EnsureMochiActor(this, Component1)) {
      // create constraints
      TryCreateConstraints_OneBody();
    }
  }
  // two-body case
  else if (Mode == EMochiConstraintComponentMode::TwoBody) {
    // find component 1 and component 2
    if ((ComponentName1.IsNone() && Component1 == nullptr) ||
        (ComponentName2.IsNone() && Component2 == nullptr)) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiConstraintComponent] ComponentName1 and/or ComponentName 2 is None!"))
      return;
    }
    if (Component1 == nullptr) {
      Component1 = FindComponentRecursive(this, ComponentName1);
    }
    if (Component1 == nullptr) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiConstraintComponent] Failed to find Component1 %s!"),
          *ComponentName1.ToString());
      return;
    }
    if (Component2 == nullptr) {
      Component2 = FindComponentRecursive(this, ComponentName2);
    }
    if (Component2 == nullptr) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiConstraintComponent] Failed to find Component2 %s!"),
          *ComponentName2.ToString());
      return;
    }
    if (Component1 == Component2) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiConstraintComponent] Component1 and Component2 must be unique!"))
      return;
    }
    // ensure component 1 and component 2 mochi actors created
    if (EnsureMochiActor(this, Component1) && EnsureMochiActor(this, Component2)) {
      // create constraints
      TryCreateConstraints_TwoBody();
    }
  }
  bConstraintsCreated = true;
}

void UMochiConstraintComponent::DestroyConstraints() {
  if (bConstraintsCreated && MochiSceneObject.IsValid()) {
    // Pass ownership of MochiThreadState to the lambda so we don't have to block
    MochiSceneObject->QueueOrExecuteCommand(
        [state = std::move(MochiThreadState)](mochi::Scene* Scene) {
          auto TryDestroyConstraint = [Scene](mochi::ConstraintHandle& Handle) {
            if (Handle.IsValid()) {
              Scene->DestroyConstraint(Handle);
            }
            Handle = {};
          };
          // clean up constraints/callbacks
          TryDestroyConstraint(state->RigidSphericalJointConstraintHandle);
          TryDestroyConstraint(state->JointRotationTrackingConstraintHandle);
          TryDestroyConstraint(state->JointRotationRangeConstraintHandle);
          TryDestroyConstraint(state->RigidPivotPositionConstraintHandle);
          TryDestroyConstraint(state->RigidPivotRotationConstraintHandle);
        });
    MochiSceneObject->WaitForQueuedCommands();
    MochiThreadState = std::make_shared<MochiState>();
    bConstraintsCreated = false;
  }
}

bool UMochiConstraintComponent::OverrideComponent1(
    USceneComponent* Component,
    FVector ConnectionPoint) {
  if (bConstraintsCreated) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("Unable to override component after constraint has been created."))
    return false;
  }
  Component1 = Component;
  bOverrideComponent1ConnectionPoint = true;
  Component1ConnectionPointOverride = ConnectionPoint;
  return true;
}

bool UMochiConstraintComponent::OverrideComponent2(
    USceneComponent* Component,
    FVector ConnectionPoint) {
  if (Mode != EMochiConstraintComponentMode::TwoBody) {
    UE_LOG(
        LogMochiPhysics, Error, TEXT("Constraint must be two body to override a second component."))
    return false;
  }
  if (bConstraintsCreated) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("Unable to override component after constraint has been created."))
    return false;
  }
  Component2 = Component;
  bOverrideComponent2ConnectionPoint = true;
  Component2ConnectionPointOverride = ConnectionPoint;
  return true;
}
