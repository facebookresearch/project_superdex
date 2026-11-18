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
#include "MochiPhysicsModule.h"

#include "Components/SceneComponent.h"
#include "Serialization/CustomVersion.h"

#include <mochi_physics/mochi_physics.h>

// Unique Mochi Object version id
// Arbitrary, permanent, and matches nothing elsewhere
const FGuid FMochiObjectVersion::GUID(0x7B3C1D42, 0x9A6E4F08, 0xB25D7C31, 0x0E8A46F9);

// Must be registered in exactly one translation unit.
static FCustomVersionRegistration GRegisterMochiObjectVersion(
    FMochiObjectVersion::GUID,
    FMochiObjectVersion::LatestVersion,
    TEXT("MochiObjectVer"));

FMochiAsyncStepParams::FMochiAsyncStepParams() {
  FromMochi(mochi::AsyncStepParams{});
}

mochi::AsyncStepParams FMochiAsyncStepParams::ToMochi() const {
  mochi::AsyncStepParams Out;
  Out.useFixedTimeStep = bUseFixedTimeStep;
  Out.fixedTimeStepSeconds = FixedTimeStepSeconds;
  Out.dynamicTimeStepMinSeconds = DynamicTimeStepMinSeconds;
  Out.dynamicTimeStepMaxSeconds = DynamicTimeStepMaxSeconds;
  return Out;
}

void FMochiAsyncStepParams::FromMochi(const mochi::AsyncStepParams& Mochi) {
  bUseFixedTimeStep = Mochi.useFixedTimeStep;
  FixedTimeStepSeconds = Mochi.fixedTimeStepSeconds;
  DynamicTimeStepMinSeconds = Mochi.dynamicTimeStepMinSeconds;
  DynamicTimeStepMaxSeconds = Mochi.dynamicTimeStepMaxSeconds;
}

EMochiActorBoundaryElementType FromMochi(mochi::ActorBoundaryElementType Type) {
  switch (Type) {
    case mochi::ActorBoundaryElementType::P1Q1:
      return EMochiActorBoundaryElementType::P1Q1;
    case mochi::ActorBoundaryElementType::P1Q3:
      return EMochiActorBoundaryElementType::P1Q3;
    case mochi::ActorBoundaryElementType::P1Q6:
      return EMochiActorBoundaryElementType::P1Q6;
    case mochi::ActorBoundaryElementType::ExperimentalP1Q7:
    case mochi::ActorBoundaryElementType::ExperimentalP1Q12:
    case mochi::ActorBoundaryElementType::ExperimentalP1Q16:
    case mochi::ActorBoundaryElementType::Count:
      break;
  }
  return EMochiActorBoundaryElementType::Default;
}

FMochiBoundarySubsamplingParams FromMochi(const mochi::BoundarySubsamplingParams& Params) {
  FMochiBoundarySubsamplingParams result;
  result.SubsamplingDensity = Params.subsamplingDensity;
  result.Strategy = Params.strategy == mochi::BoundarySubsamplingStrategy::AreaProportional
      ? EMochiBoundarySubsamplingStrategy::AreaProportional
      : EMochiBoundarySubsamplingStrategy::UniformProbability;
  return result;
}

mochi::BoundarySubsamplingParams ToMochi(const FMochiBoundarySubsamplingParams& Params) {
  mochi::BoundarySubsamplingParams result;
  result.subsamplingDensity = static_cast<mochi::real>(Params.SubsamplingDensity);
  switch (Params.Strategy) {
    case EMochiBoundarySubsamplingStrategy::UniformProbability:
      result.strategy = mochi::BoundarySubsamplingStrategy::UniformProbability;
      break;
    case EMochiBoundarySubsamplingStrategy::AreaProportional:
      result.strategy = mochi::BoundarySubsamplingStrategy::AreaProportional;
      break;
  }
  return result;
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

FMochiGridSdfParams::FMochiGridSdfParams() {
  FromMochi(mochi::GridSdfParams{});
}

mochi::GridSdfParams FMochiGridSdfParams::ToMochi() const {
  mochi::GridSdfParams result;
  result.resolutionMode = static_cast<mochi::GridSdfResolutionMode>(ResolutionMode);
  result.resolutionDelta = mochi::Real3{
      static_cast<mochi::real>(ResolutionDelta.X),
      static_cast<mochi::real>(ResolutionDelta.Y),
      static_cast<mochi::real>(ResolutionDelta.Z)};
  // Add small value to padding so collision checks up to and including
  // the distance of the padding selected aren't subjected to floating point precision issues
  result.boundaryPaddingDist = BoundaryPaddingDist + 0.000001f;
  return result;
}

void FMochiGridSdfParams::FromMochi(const mochi::GridSdfParams& Mochi) {
  ResolutionMode = static_cast<EMochiGridSdfResolutionMode>(Mochi.resolutionMode);
  ResolutionDelta.X = Mochi.resolutionDelta[0];
  ResolutionDelta.Y = Mochi.resolutionDelta[1];
  ResolutionDelta.Z = Mochi.resolutionDelta[2];
  BoundaryPaddingDist = Mochi.boundaryPaddingDist;
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

FMochiIKSolverParams::FMochiIKSolverParams() {
  FromMochi(mochi::experimental::IKSolverParams{});
}

mochi::experimental::IKSolverParams FMochiIKSolverParams::ToMochi() const {
  mochi::experimental::IKSolverParams Out;
  Out.maxIter = MaxIter;
  Out.absTol = static_cast<mochi::real>(AbsTol);
  Out.relTol = static_cast<mochi::real>(RelTol / RelTolScalar);
  Out.positionErrorThres = static_cast<mochi::real>(PositionErrorThres);
  Out.rotationErrorThres = static_cast<mochi::real>(RotationErrorThres);
  Out.lineSearchMaxIter = LineSearchMaxIter;
  Out.maxElapsedTimeSeconds = MaxElapsedTimeSeconds;
  return Out;
}

void FMochiIKSolverParams::FromMochi(const mochi::experimental::IKSolverParams& Mochi) {
  MaxIter = Mochi.maxIter;
  AbsTol = static_cast<double>(Mochi.absTol);
  RelTolScalar = 1e6;
  RelTol = static_cast<double>(Mochi.relTol) * RelTolScalar;
  PositionErrorThres = static_cast<double>(Mochi.positionErrorThres);
  RotationErrorThres = static_cast<double>(Mochi.rotationErrorThres);
  LineSearchMaxIter = Mochi.lineSearchMaxIter;
  MaxElapsedTimeSeconds = Mochi.maxElapsedTimeSeconds;
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
