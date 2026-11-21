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

#include "MochiTypes.h"

#include "MochiConversions.h"
#include "MochiGenSupport.h"
#include "MochiPhysicsModule.h"

#include "Components/SceneComponent.h"
#include "Serialization/CustomVersion.h"

#include <mochi_physics/mochi_physics.h>

#include <variant>

// Unique Mochi Object version id
// Arbitrary, permanent, and matches nothing elsewhere
const FGuid FMochiObjectVersion::GUID(0x7B3C1D42, 0x9A6E4F08, 0xB25D7C31, 0x0E8A46F9);

// Must be registered in exactly one translation unit.
static FCustomVersionRegistration GRegisterMochiObjectVersion(
    FMochiObjectVersion::GUID,
    FMochiObjectVersion::LatestVersion,
    TEXT("MochiObjectVer"));

FMochiRomParams::FMochiRomParams() {
  FromMochi(mochi::experimental::RomParams{});
}

void FMochiRomParams::FromMochi(const mochi::experimental::RomParams& Mochi) {
  Source = MochiGenSupport::ToUnrealString(Mochi.source);
  HyperReductionStrategy = EMochiHyperReductionStrategy::NoHyperReduction;
  auto sampling = mochi::experimental::SampleMeshInitRandomSampling{};
  if (Mochi.hyperReduction) {
    if (const auto* randomSampling = std::get_if<mochi::experimental::SampleMeshInitRandomSampling>(
            &Mochi.hyperReduction->initializationStrategy)) {
      HyperReductionStrategy =
          EMochiHyperReductionStrategy::StaticHyperReductionUsingRandomSampling;
      sampling = *randomSampling;
    } else {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT("FMochiRomParams only supports random-sampling hyper-reduction; ignoring it."));
    }
  }
  StepSizeForBoundaryElementsSelection = sampling.stepSizeForBoundaryElementsSelection;
  StepSizeForInteriorElementsSelection = sampling.stepSizeForInteriorElementsSelection;
}

mochi::experimental::RomParams FMochiRomParams::ToMochi() const {
  mochi::experimental::RomParams params;
  params.source = std::string(TCHAR_TO_ANSI(*Source));
  if (HyperReductionStrategy == EMochiHyperReductionStrategy::NoHyperReduction) {
    params.hyperReduction = {};
  } else if (
      HyperReductionStrategy ==
      EMochiHyperReductionStrategy::StaticHyperReductionUsingRandomSampling) {
    params.hyperReduction = mochi::experimental::HyperReductionParams{
        mochi::experimental::SampleMeshInitRandomSampling{
            StepSizeForBoundaryElementsSelection, StepSizeForInteriorElementsSelection},
        {}};
  }
  return params;
}

mochi::Real6 FMochiShapeTargetTensor::ToMochi() const {
  return mochi::Real6{XX, XY, XZ, YY, YZ, ZZ};
}

void FMochiShapeTargetTensor::FromMochi(const mochi::Real6& Mochi) {
  XX = Mochi[0];
  XY = Mochi[1];
  XZ = Mochi[2];
  YY = Mochi[3];
  YZ = Mochi[4];
  ZZ = Mochi[5];
}

//------------------------------------------------------------------------------
// CONTACT HANDLING
//------------------------------------------------------------------------------

namespace {

inline void ContactEventHandlerBehavior(
    const USceneComponent* CallingMochiComponent,
    TArray<FMochiContactPoint>& OutContactPoints,
    TSet<USceneComponent*>& OuContactingComponents,
    TSet<USceneComponent*>& OutPreviousContactingComponents,
    const TArray<FMochiContactPoint>& NewContactPoints,
    bool bFilterSelfContacts,
    const TFunction<void(USceneComponent*, const TArray<FMochiContactPoint>&)>& OnBeginContact,
    const TFunction<void(USceneComponent*)>& OnEndContact,
    const TFunction<void(const TArray<FMochiContactPoint>&)>& OnPersistContact) {
  // collect contacts and contacting components
  OutContactPoints.SetNum(0);
  OuContactingComponents.Reset();
  for (auto& contact : NewContactPoints) {
    // We can sometimes get null component pointers for some reason so we will remove them as they
    // make no sense and are untraceable
    if (contact.OtherMochiComponent &&
        (!bFilterSelfContacts || CallingMochiComponent != contact.OtherMochiComponent)) {
      OutContactPoints.Add(contact);
      OuContactingComponents.Add(contact.OtherMochiComponent);
    }
  }
  // begin contact events
  for (const auto otherComp : OuContactingComponents) {
    if (!OutPreviousContactingComponents.Contains(otherComp)) {
      // create temp array of contacts for only this component
      thread_local TArray<FMochiContactPoint> beginContacts;
      beginContacts.Reset();
      for (auto& contact : OutContactPoints) {
        if (contact.OtherMochiComponent == otherComp) {
          beginContacts.Add(contact);
        }
      }
      OnBeginContact(otherComp, beginContacts);
    }
  }
  // end contact events
  for (const auto otherComp : OutPreviousContactingComponents) {
    if (!OuContactingComponents.Contains(otherComp)) {
      OnEndContact(otherComp);
    }
  }
  // persistent contact event
  if (OutContactPoints.Num() > 0) {
    OnPersistContact(OutContactPoints);
  }
  // swap buffers
  OutPreviousContactingComponents = OuContactingComponents;
}

} // namespace

void FMochiContactEventHandler::Update(
    USceneComponent* CallingMochiComponent,
    const FOnMochiComponentBeginContact& OnMochiComponentBeginContact,
    const FOnMochiComponentEndContact& OnMochiComponentEndContact,
    const FOnMochiComponentContact& OnMochiComponentContact,
    const TArray<FMochiContactPoint>& NewContactPoints,
    bool bFilterSelfContacts) {
  ContactEventHandlerBehavior(
      CallingMochiComponent,
      ContactPoints,
      ContactingComponents,
      PreviousContactingComponents,
      NewContactPoints,
      bFilterSelfContacts,
      [&](USceneComponent* OtherMochiComponent, const TArray<FMochiContactPoint>& BeginContacts) {
        OnMochiComponentBeginContact.Broadcast(
            CallingMochiComponent,
            OtherMochiComponent->GetOwner(),
            OtherMochiComponent,
            BeginContacts);
      },
      [&](USceneComponent* OtherMochiComponent) {
        OnMochiComponentEndContact.Broadcast(
            CallingMochiComponent, OtherMochiComponent->GetOwner(), OtherMochiComponent);
      },
      [&](const TArray<FMochiContactPoint>& PersistContacts) {
        OnMochiComponentContact.Broadcast(CallingMochiComponent, PersistContacts);
      });
}

void FMochiNestedContactEventHandler::Update(
    USceneComponent* CallingMochiComponent,
    const FOnMochiComponentBeginNestedContact& OnMochiComponentBeginNestedContact,
    const FOnMochiComponentEndNestedContact& OnMochiComponentEndNestedContact,
    const FOnMochiComponentNestedContact& OnMochiComponentNestedContact,
    const TArray<FMochiContactPoint>& NewContactPoints,
    bool bFilterSelfContacts) {
  // sort and copy new points by nested name
  SortedContactPoints.Reset();
  // first we need to add previously observed nested names otherwise we'll never call EndContact
  for (auto& pair : PreviousContactingComponents) {
    SortedContactPoints.Add(pair.Key, {});
  }
  for (auto& contact : NewContactPoints) {
    SortedContactPoints.FindOrAdd(contact.NestedName).Add(contact);
  }
  for (auto& pair : SortedContactPoints) {
    FName name = pair.Key;
    TArray<FMochiContactPoint>& newContactPoints = pair.Value;
    ContactEventHandlerBehavior(
        CallingMochiComponent,
        ContactPoints.FindOrAdd(name),
        ContactingComponents.FindOrAdd(name),
        PreviousContactingComponents.FindOrAdd(name),
        newContactPoints,
        bFilterSelfContacts,
        [&](USceneComponent* OtherMochiComponent, const TArray<FMochiContactPoint>& BeginContacts) {
          OnMochiComponentBeginNestedContact.Broadcast(
              CallingMochiComponent,
              name,
              OtherMochiComponent->GetOwner(),
              OtherMochiComponent,
              BeginContacts);
        },
        [&](USceneComponent* OtherMochiComponent) {
          OnMochiComponentEndNestedContact.Broadcast(
              CallingMochiComponent, name, OtherMochiComponent->GetOwner(), OtherMochiComponent);
        },
        [&](const TArray<FMochiContactPoint>& PersistContacts) {
          OnMochiComponentNestedContact.Broadcast(CallingMochiComponent, name, PersistContacts);
        });
  }
}
