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

#include "SuperDexWristbandComponent.h"

#include "Engine/EngineTypes.h"
#include "SuperDexPoseableActor.h"
#include "SuperDexPoseableComponent.h"

USuperDexWristbandComponent::USuperDexWristbandComponent() {
  PrimaryComponentTick.bCanEverTick = true;
  PrimaryComponentTick.TickInterval = 0.250f;
}

void USuperDexWristbandComponent::OnComponentDestroyed(bool bDestroyingHierarchy) {
  Super::OnComponentDestroyed(bDestroyingHierarchy);
  if (NameTextRender != nullptr) {
    NameTextRender->DestroyComponent();
    NameTextRender = nullptr;
  }
  if (SourceTextRender != nullptr) {
    SourceTextRender->DestroyComponent();
    SourceTextRender = nullptr;
  }
}

void USuperDexWristbandComponent::OnRegister() {
  if ((GetOwner() != nullptr) && !IsRunningCommandlet()) {
    if (NameTextRender == nullptr) {
      NameTextRender = CreateTextRenderComponent(WristbandToNameText);
      NameTextRender->SetText(FText::FromString("Poseable Name"));
    }
    if (SourceTextRender == nullptr) {
      SourceTextRender = CreateTextRenderComponent(WristbandToSourceText);
      SourceTextRender->SetText(FText::FromString("Source Name"));
    }
  }
  Super::OnRegister();
}

UTextRenderComponent* USuperDexWristbandComponent::CreateTextRenderComponent(
    FTransform RelativeTransform) {
  // spawn the text render component with undo and transient flags
  UTextRenderComponent* OutTextRenderComponent = NewObject<UTextRenderComponent>(
      GetOwner(), NAME_None, RF_Transactional | RF_TextExportTransient);
  OutTextRenderComponent->SetupAttachment(this);

  // UTextRenderComponent variables setup
  OutTextRenderComponent->SetTextMaterial(TextMaterial);
  OutTextRenderComponent->SetHorizontalAlignment(EHorizTextAligment::EHTA_Center);
  OutTextRenderComponent->SetTextRenderColor(FColor::Black);
  OutTextRenderComponent->SetWorldSize(1.0);
  OutTextRenderComponent->SetVisibility(this->IsVisible());
  OutTextRenderComponent->CastShadow = false;
  OutTextRenderComponent->CreationMethod = CreationMethod; // same as this component

  OutTextRenderComponent->SetRelativeTransform(RelativeTransform);

  // Because the owning actor is part of the world already can just register directly
  OutTextRenderComponent->RegisterComponent();

  return OutTextRenderComponent;
}

void USuperDexWristbandComponent::TickComponent(
    float DeltaTime,
    enum ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction) {
  if ((NameTextRender == nullptr) || (SourceTextRender == nullptr)) {
    return;
  }

  // Set visibility of text render components to match this component's visibility
  NameTextRender->SetVisibility(this->IsVisible());
  SourceTextRender->SetVisibility(this->IsVisible());

  // climb the component hierarchy to find the first USuperDexPoseableComponent ancestor
  USuperDexPoseableComponent* AncestorPoseableComponent = nullptr;
  USceneComponent* checkComp = GetAttachParent();
  while (AncestorPoseableComponent == nullptr) {
    AncestorPoseableComponent = Cast<USuperDexPoseableComponent>(checkComp);
    if (checkComp != nullptr) {
      checkComp = checkComp->GetAttachParent();
    } else {
      break;
    }
  }
  if (AncestorPoseableComponent != nullptr) {
    // we have a valid ancestor, set the text based on its pose source and the owning actor
    NameTextRender->SetText(FText::FromString(GetOwner()->GetName()));

    FSuperDexPose Pose;
    bool bHavePose = AncestorPoseableComponent->GetPose(Pose);
    if (bHavePose) {
      SourceTextRender->SetText(FText::FromName(Pose.SourceName));
    }
  } else {
    // if we didn't find a USuperDexPoseableComponent ancestor, try the owner
    ASuperDexPoseableActor* OwnerPoseableActor = Cast<ASuperDexPoseableActor>(GetOwner());
    if (OwnerPoseableActor != nullptr) {
      NameTextRender->SetText(FText::FromString(OwnerPoseableActor->GetName()));

      FSuperDexPose Pose;
      bool bHavePose = OwnerPoseableActor->GetPose(Pose);
      if (bHavePose) {
        SourceTextRender->SetText(FText::FromName(Pose.SourceName));
      }
    } else {
      // All the way up the chain there is not a poseable
      NameTextRender->SetText(FText::FromString("No Poseable Ancestor"));
      SourceTextRender->SetText(FText::FromString("No Poseable Ancestor"));
    }
  }
}
