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

#include "MochiRigidMeshComponent.h"
#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "MochiComponentTags.h"
#include "MochiConversions.h"
#include "MochiGameInstanceSubsystem.h"
#include "MochiPhysicsModule.h"

#include "GameFramework/Actor.h"

namespace {
bool IsSceneExternallyOwned(const UWorld* World) {
  if (!World) {
    return false;
  }
  if (const UGameInstance* GI = World->GetGameInstance()) {
    if (const UMochiGameInstanceSubsystem* Subsys =
            GI->GetSubsystem<UMochiGameInstanceSubsystem>()) {
      return Subsys->IsSceneExternallyOwned();
    }
  }
  return false;
}
} // namespace

#if WITH_EDITOR
#include "AssetRegistry/AssetData.h"
#include "EditorAssetLibrary.h"
#include "Engine/StaticMesh.h"
#endif

bool UMochiRigidMeshComponent::SetModel(UMochiModel* NewModel) {
  if (MochiActorObject) {
    // TODO: Should be possible to allow this by destroying/recreating the Mochi actor
    return false;
  } else {
    Model = NewModel;
    return true;
  }
}

FVector UMochiRigidMeshComponent::GetWorldLinearVelocity() const {
  return LinearVelocity;
}

FVector UMochiRigidMeshComponent::GetWorldAngularVelocity() const {
  return AngularVelocity;
}

const TArray<FMochiContactPoint>& UMochiRigidMeshComponent::GetContactPoints() const {
  return ContactEventHandler.ContactPoints;
}

void UMochiRigidMeshComponent::CreateMochiActor() {
  MOCHI_PROFILE_SCOPE();
  // Guarded here rather than only in BeginPlay because the constraint components also create
  // actors through this entry point.
  if (IsSceneExternallyOwned(GetWorld())) {
    UE_LOG(
        LogMochiPhysics,
        Verbose,
        TEXT("[%hs] Scene is externally owned; skipping actor creation on %s/%s"),
        __FUNCTION__,
        *GetNameSafe(GetOwner()),
        *GetName());
    return;
  }
  if (!GetOwner()->HasAuthority()) {
    return; // The server or solo player has authority. Multiplayer clients don't.
  }

  DestroyMochiActor();
  if (bUseStaticMeshAsMochiModel) {
    MochiActorObject = UMochiActorObject::CreateRigidFromSurfaceMesh(
        this,
        GetName(),
        GetComponentTransform(),
        GetStaticMesh(),
        CollisionSource,
        MaterialParams,
        ContactParams,
        ColliderType,
        Layer,
        bIsStatic,
        bHasGravity,
        BoundaryElementType,
        bEnableBoundarySubsampling ? &BoundarySubsamplingParams : nullptr,
        bUseStaticMeshAsMochiModel ? &GridSdfParams : nullptr,
        bOverrideMass ? &Mass : nullptr,
        bOverrideInertialParams ? &CenterOfMass : nullptr,
        bOverrideInertialParams ? &MomentOfInertia : nullptr,
        false,
        false,
        bEnableContacts,
        nullptr);
  } else {
    MochiActorObject = UMochiActorObject::CreateRigid(
        this,
        GetName(),
        GetComponentTransform(),
        Model,
        MaterialParams,
        ContactParams,
        ColliderType,
        Layer,
        bIsStatic,
        bHasGravity,
        BoundaryElementType,
        bEnableBoundarySubsampling ? &BoundarySubsamplingParams : nullptr,
        bUseStaticMeshAsMochiModel ? &GridSdfParams : nullptr,
        bOverrideMass ? &Mass : nullptr,
        bOverrideInertialParams ? &CenterOfMass : nullptr,
        bOverrideInertialParams ? &MomentOfInertia : nullptr,
        false,
        false,
        bEnableContacts,
        nullptr);
  }
}

void UMochiRigidMeshComponent::DestroyMochiActor() {
  MOCHI_PROFILE_SCOPE();
  if (MochiActorObject) {
    MochiActorObject->DestroyMochiResources();
    MochiActorObject = nullptr;
  }
}

UMochiActorObject* UMochiRigidMeshComponent::GetMochiActor() const {
  return MochiActorObject;
}

bool UMochiRigidMeshComponent::GetContactsEnabled() const {
  return bEnableContacts;
}

UMochiRigidMeshComponent::UMochiRigidMeshComponent(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  PrimaryComponentTick.bCanEverTick = true;
}

void UMochiRigidMeshComponent::SetMochiModel(UMochiModel* NewMochiModel) {
  MOCHI_PROFILE_SCOPE();
  UE_LOG(
      LogMochiPhysics,
      Display,
      TEXT("[%hs]   %hs(%s)"),
      __FUNCTION__,
      __FUNCTION__,
      *NewMochiModel->GetPackage()->GetName());
  Model = NewMochiModel;
}

UMochiModel* UMochiRigidMeshComponent::GetMochiModel() const {
  return Model;
}

void UMochiRigidMeshComponent::BeginPlay() {
  MOCHI_PROFILE_SCOPE();
  Super::BeginPlay();

  if (MochiComponentTags::AreComponentsDisabled(GetOwner())) {
    return;
  }

  // Nothing to create or configure when the scene's contents come from a runtime loader; the
  // missing-Model warning below would be noise for props the loaded scene already supplies.
  if (IsSceneExternallyOwned(GetWorld())) {
    return;
  }

  if (!bUseStaticMeshAsMochiModel && !Model) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[%hs] %hs failed because not all params were set."),
        __FUNCTION__,
        __FUNCTION__);
    return;
  }
  // create the actor if it hasn't been created externally (e.g. by a constraint component)
  if (MochiActorObject == nullptr) {
    CreateMochiActor();
  }

  // freeze degrees of freedom
  if (MochiActorObject && (FreezeConstraints != (uint8)EMochiFreezeConstraint::None)) {
    mochi::ActorHandle Handle = MochiActorObject->GetMochiActorHandle();
    MochiActorObject->GetMochiScene()->QueueOrExecuteActorCommand(
        Handle, [this](mochi::Actor* Actor) {
          mochi::ErrorLog E{};
          std::vector<int> DofIndices;
          DofIndices.reserve(6);
          for (int DofIndex = 0; DofIndex < 6; ++DofIndex) {
            const uint8 DofFlag = 1 << DofIndex;
            if ((FreezeConstraints & DofFlag) == DofFlag) {
              DofIndices.push_back(DofIndex);
            }
          }
          std::vector<mochi::real> DofValues(DofIndices.size(), 0.0f);
          Actor->GetDofValues(DofIndices, DofValues, E);
          Actor->AddBoundaryConditionDofsWorld(DofIndices, DofValues, E);
        });
  }
}

// Set the world location and rotation of the Mochi mesh component.
// For convenience allows to use a transform but the scale is filtered out.
void UMochiRigidMeshComponent::SetWorldLocationAndRotationMochiWithDynamicsTransform(
    const FTransform& NewTransform) {
  SetWorldLocationAndRotationMochiWithDynamics(
      NewTransform.GetLocation(), UE::Math::TRotator(NewTransform.GetRotation()));
}

// Set the world location and rotation of the Mochi mesh component.
void UMochiRigidMeshComponent::SetWorldLocationAndRotationMochiWithDynamics(
    const FVector& NewLocation,
    const FRotator& NewRotation) {
  FTransform NewTransform(NewRotation, NewLocation);
  mochi::ActorHandle Handle = GetMochiActor()->GetMochiActorHandle();
  FUnrealToMochiConverter U2m;
  GetMochiActor()->GetMochiScene()->QueueOrExecuteActorCommand(
      Handle, [NewTransform, U2m](mochi::Actor* Actor) {
        std::array<int, 6> DofIndices{0, 1, 2, 3, 4, 5};
        std::array<mochi::real, 6> DofValues{};
        mochi::ErrorLog E{};

        // Get the degrees of freedom value
        Actor->GetDofValues(DofIndices, DofValues, E);
        mochi::Real3 DofWorldPosition{DofValues[0], DofValues[1], DofValues[2]};

        // Get Root Transform Information
        mochi::TransformRT Root = Actor->GetRootTransform();
        mochi::Real3 RootWorldPosition = Root.GetTranslation();

        // The dof transform is about the center of mass.
        // The root transform is about the origin of the local space.
        // So the position of the center of mass in world space is merely
        // the dof translation or the root translation + the local com coordinates rotated to world
        // space.
        mochi::Real3 ComLocalPos =
            Root.TransformDirectionInverse(DofWorldPosition - RootWorldPosition);

        // Get the new transform
        mochi::TransformRT NewTransform_Mochi =
            U2m.TransformToMochi_RT(NewTransform); // world from local of grip
        mochi::TransformRT NewTransformInverse_Mochi =
            mochi::Invert(U2m.TransformToMochi_RT(NewTransform));
        mochi::Real3 Translation = NewTransform_Mochi.GetTranslation();
        mochi::Real3 RotationVec = NewTransform_Mochi.GetRotation().ToRotationVector();

        // Get new dof
        mochi::Real3 TranslationDof =
            Translation + NewTransform_Mochi.TransformDirection(ComLocalPos);
        mochi::Real3 RotationDof = RotationVec;

        for (int i = 0; i < 3; ++i) {
          DofValues[i] = TranslationDof[i];
        }
        for (int i = 3; i < 6; ++i) {
          DofValues[i] = RotationDof[i - 3];
        }

        Actor->ClearBoundaryConditions();
        Actor->AddBoundaryConditionDofsWorld(DofIndices, DofValues, E);
      });
}

void UMochiRigidMeshComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  MOCHI_PROFILE_SCOPE();
  Super::EndPlay(EndPlayReason);
  DestroyMochiActor();
}

void UMochiRigidMeshComponent::TickComponent(
    float DeltaTime,
    ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction) {
  MOCHI_PROFILE_SCOPE();
  if (MochiActorObject == nullptr) {
    return;
  }
  // Update dynamic mesh vertex positions, normals, and actor transform if needed
  MochiActorObject->TryGetQueryData([this](const FMochiActorQueryData& QueryData) {
    // set mesh transform
    // NOTE: Scale is baked into the Mochi shape, but not the UStaticMesh, so set 6-dof pose only
    SetWorldLocationAndRotation(
        QueryData.RootTransform.GetLocation(), QueryData.RootTransform.GetRotation());
    LinearVelocity = QueryData.LinearVelocity;
    AngularVelocity = QueryData.AngularVelocity;
    // update contact events
    if (bEnableContacts) {
      ContactEventHandler.Update(
          this,
          OnMochiComponentBeginContact,
          OnMochiComponentEndContact,
          OnMochiComponentContact,
          QueryData.ContactPoints);
    }
  });
}

#if WITH_EDITOR
void UMochiRigidMeshComponent::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) {
  MOCHI_PROFILE_SCOPE();
  // Check for changes in the Model property...
  const FName PropName = PropertyChangedEvent.GetPropertyName();
  if (PropName == GET_MEMBER_NAME_CHECKED(UMochiRigidMeshComponent, Model)) {
    UE_LOG(
        LogMochiPhysics,
        Display,
        TEXT("[%hs] %hs() - %s changed"),
        __FUNCTION__,
        __FUNCTION__,
        *PropName.ToString());
    // If we have a valid Model assigned...
    if (Model && Model->IsAsset()) {
      UE_LOG(
          LogMochiPhysics, Display, TEXT("  - package name = %s"), *Model->GetPackage()->GetName());

      // Attempt to auto-assign corresponding static mesh asset if it exists
      FString StaticMeshPath = Model->GetPackage()->GetName() + TEXT("_StaticMesh");
      if (UEditorAssetLibrary::DoesAssetExist(StaticMeshPath)) {
        FAssetData Data = UEditorAssetLibrary::FindAssetData(StaticMeshPath);
        // If it's indeed a static mesh, try to assign
        if (UStaticMesh* Mesh = Cast<UStaticMesh>(Data.GetAsset())) {
          SetStaticMesh(Mesh);
        }
      }
    }
  }
  // check static mesh CPU access
  if (bUseStaticMeshAsMochiModel) {
    if (TObjectPtr<UStaticMesh> Mesh = GetStaticMesh()) {
      if (!Mesh->bAllowCPUAccess) {
        UE_LOG(
            LogMochiPhysics,
            Warning,
            TEXT("[%hs] %s is true but static mesh %s is false; expect undefined behavior!"),
            __FUNCTION__,
            *GET_MEMBER_NAME_CHECKED(UMochiRigidMeshComponent, bUseStaticMeshAsMochiModel)
                 .ToString(),
            *GET_MEMBER_NAME_CHECKED(UStaticMesh, bAllowCPUAccess).ToString());
      }
    }
  }
  // Make sure we invoke the base class implementation our work, so that it commits our changes.
  Super::PostEditChangeProperty(PropertyChangedEvent);
}
#endif // WITH_EDITOR
