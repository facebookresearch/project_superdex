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

#include "MochiActors.h"

#include "Logging/MessageLog.h"
#include "Misc/MapErrors.h"
#include "Misc/UObjectToken.h"
#include "MochiArticulationComponent.h"

#include <mochi_core/utils/profile.h>

#define LOCTEXT_NAMESPACE "MochiActors"

//------------------------------------------------------------------------------
// MOCHI SOFT MESH ACTOR
//------------------------------------------------------------------------------

FName AMochiSoftMeshActor::MochiSoftMeshComponentName(TEXT("MochiSoftMeshComponent0"));

AMochiSoftMeshActor::AMochiSoftMeshActor(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  MOCHI_PROFILE_SCOPE();
  SetCanBeDamaged(false);

  MochiSoftMeshComponent =
      CreateDefaultSubobject<UMochiSoftMeshComponent>(MochiSoftMeshComponentName);
  RootComponent = MochiSoftMeshComponent;
  bCanBeInCluster = (GetClass() == AMochiSoftMeshActor::StaticClass());
}

void AMochiSoftMeshActor::BeginPlay() {
  Super::BeginPlay();
}

#if WITH_EDITOR

void AMochiSoftMeshActor::CheckForErrors() {
  MOCHI_PROFILE_SCOPE();
  Super::CheckForErrors();
  FMessageLog MapCheck("MapCheck");
  if (!MochiSoftMeshComponent) {
    FFormatNamedArguments Arguments;
    Arguments.Add(TEXT("ActorClass"), FText::FromString(StaticClass()->GetName()));
    Arguments.Add(TEXT("ActorName"), FText::FromString(GetName()));
    Arguments.Add(
        TEXT("ComponentClass"),
        FText::FromString(UMochiSoftMeshComponent::StaticClass()->GetName()));
    MapCheck.Warning()
        ->AddToken(FUObjectToken::Create(this))
        ->AddToken(
            FTextToken::Create(
                FText::Format(
                    LOCTEXT(
                        "MapCheck_Message_MochiSoftMeshComponent",
                        "{ActorClass} named {ActorName} has NULL {ComponentClass} property - please delete"),
                    Arguments)));
  }
}

bool AMochiSoftMeshActor::GetReferencedContentObjects(TArray<UObject*>& Objects) const {
  MOCHI_PROFILE_SCOPE();
  Super::GetReferencedContentObjects(Objects);
  if (MochiSoftMeshComponent && MochiSoftMeshComponent->GetMochiModel()) {
    Objects.Add(MochiSoftMeshComponent->GetMochiModel());
  }
  return true;
}

#endif

//------------------------------------------------------------------------------
// MOCHI ARTICULATION ACTOR
//------------------------------------------------------------------------------

FName AMochiArticulationActor::MochiArticulationComponentName(TEXT("MochiArticulationComponent0"));

AMochiArticulationActor::AMochiArticulationActor() {
  PrimaryActorTick.bCanEverTick = true;
  ArticulationComponent =
      CreateDefaultSubobject<UMochiArticulationComponent>(MochiArticulationComponentName);
  RootComponent = ArticulationComponent;
  bCanBeInCluster = (GetClass() == StaticClass());
}

void AMochiArticulationActor::BeginPlay() {
  Super::BeginPlay();
}

void AMochiArticulationActor::Tick(float DeltaTime) {
  Super::Tick(DeltaTime);
}

void AMochiArticulationActor::SetArticulationAsset(UMochiArticulation* NewArticulation) {
  ArticulationComponent->SetArticulationAsset(NewArticulation);
}

UMochiArticulation* AMochiArticulationActor::GetArticulationAsset() const {
  return ArticulationComponent->GetArticulationAsset();
}

UMochiArticulationComponent* AMochiArticulationActor::GetArticulationComponent() const {
  return ArticulationComponent;
}

#if WITH_EDITOR

void AMochiArticulationActor::CheckForErrors() {
  Super::CheckForErrors();
  FMessageLog MapCheck("MapCheck");
  if (!ArticulationComponent) {
    FFormatNamedArguments Arguments;
    Arguments.Add(TEXT("ActorName"), FText::FromString(GetName()));
    MapCheck.Warning()
        ->AddToken(FUObjectToken::Create(this))
        ->AddToken(
            FTextToken::Create(
                FText::Format(
                    LOCTEXT(
                        "MapCheck_Message_AMochiArticulationActor",
                        "Mochi Articulation Editor actor {ActorName} has NULL ArticulationComponent property - please delete"),
                    Arguments)));
  }
}

bool AMochiArticulationActor::GetReferencedContentObjects(TArray<UObject*>& Objects) const {
  Super::GetReferencedContentObjects(Objects);
  if (ArticulationComponent) {
    Objects.Add(ArticulationComponent);
    if (ArticulationComponent->GetArticulationAsset()) {
      if (ArticulationComponent->GetArticulationAsset()->GetSkeletalMesh()) {
        Objects.Add(ArticulationComponent->GetArticulationAsset()->GetSkeletalMesh());
      }
    }
  }
  return true;
}

#endif

//------------------------------------------------------------------------------
// MOCHI ARTICULATION PREVIEW ACTOR
//------------------------------------------------------------------------------

FName AMochiArticulationPreviewActor::PoseableMeshComponentName(TEXT("PoseableMeshComponent0"));

AMochiArticulationPreviewActor::AMochiArticulationPreviewActor() {
  PrimaryActorTick.bCanEverTick = true;
  PoseableMeshComponent =
      CreateDefaultSubobject<UMochiArticulationPreviewComponent>(PoseableMeshComponentName);
  RootComponent = PoseableMeshComponent;
  bCanBeInCluster = (GetClass() == StaticClass());
}

void AMochiArticulationPreviewActor::BeginPlay() {
  Super::BeginPlay();
}

void AMochiArticulationPreviewActor::Tick(float DeltaTime) {
  Super::Tick(DeltaTime);
}

void AMochiArticulationPreviewActor::SetArticulation(UMochiArticulationBase* NewArticulation) {
  PoseableMeshComponent->SetArticulation(NewArticulation);
}

UMochiArticulationBase* AMochiArticulationPreviewActor::GetArticulation() const {
  return PoseableMeshComponent->GetArticulation();
}

void AMochiArticulationPreviewActor::SetJointPositions(const TArray<double>& NewJointPositions) {
  PoseableMeshComponent->SetJointPositions(NewJointPositions);
}

UMochiArticulationPreviewComponent* AMochiArticulationPreviewActor::GetPoseableMeshComponent()
    const {
  return PoseableMeshComponent;
}

#if WITH_EDITOR

void AMochiArticulationPreviewActor::CheckForErrors() {
  Super::CheckForErrors();
  FMessageLog MapCheck("MapCheck");
  if (!PoseableMeshComponent) {
    FFormatNamedArguments Arguments;
    Arguments.Add(TEXT("ActorName"), FText::FromString(GetName()));
    MapCheck.Warning()
        ->AddToken(FUObjectToken::Create(this))
        ->AddToken(
            FTextToken::Create(
                FText::Format(
                    LOCTEXT(
                        "MapCheck_Message_AMochiArticulationPreviewActor",
                        "Mochi Articulation Editor actor {ActorName} has NULL PoseableMeshComponent property - please delete"),
                    Arguments)));
  }
}

bool AMochiArticulationPreviewActor::GetReferencedContentObjects(TArray<UObject*>& Objects) const {
  Super::GetReferencedContentObjects(Objects);
  if (PoseableMeshComponent) {
    Objects.Add(PoseableMeshComponent);
    if (PoseableMeshComponent->GetArticulation()) {
      Objects.Add(PoseableMeshComponent->GetArticulation()->GetSkeletalMesh());
      if (PoseableMeshComponent->GetArticulation()->GetSkeletalMesh()) {
        Objects.Add(PoseableMeshComponent->GetArticulation()->GetSkeletalMesh());
      }
    }
  }
  return true;
}

#endif

#undef LOCTEXT_NAMESPACE
