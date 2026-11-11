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

// clang-format off
// Please keep these enums in sync

// Can't use mochi::float because it doesn't work with Unreal's code generation

// TODO: Convert between the Mochi and UE enums with functions (like the ToMochi overloads in
// MochiTypes.h) instead of a static_cast at each use, even when the function is only a cast. One
// place per enum pair keeps a cast from picking the wrong enum.

static_assert(sizeof(float) == sizeof(mochi::real), "mochi::real unexpected size");

static_assert(static_cast<int>(EMochiActorType::None) == static_cast<int>(mochi::ActorType::None));
static_assert(static_cast<int>(EMochiActorType::Rigid) == static_cast<int>(mochi::ActorType::Rigid));
static_assert(static_cast<int>(EMochiActorType::Soft) == static_cast<int>(mochi::ActorType::Soft));
static_assert(static_cast<int>(EMochiActorType::Articulated) == static_cast<int>(mochi::ActorType::Articulated));
static_assert(static_cast<int>(EMochiActorType::Shell) == static_cast<int>(mochi::ActorType::Shell));
static_assert(static_cast<int>(EMochiActorType::Rod) == static_cast<int>(mochi::ActorType::Rod));

static_assert(static_cast<int>(EMochiColliderType::None) == static_cast<int>(mochi::ColliderType::None));
static_assert(static_cast<int>(EMochiColliderType::Auto) == static_cast<int>(mochi::ColliderType::Auto));
static_assert(static_cast<int>(EMochiColliderType::Sphere) == static_cast<int>(mochi::ColliderType::Sphere));
static_assert(static_cast<int>(EMochiColliderType::Box) == static_cast<int>(mochi::ColliderType::Box));
static_assert(static_cast<int>(EMochiColliderType::Mesh) == static_cast<int>(mochi::ColliderType::Mesh));
static_assert(static_cast<int>(EMochiColliderType::Plane) == static_cast<int>(mochi::ColliderType::Plane));
static_assert(static_cast<int>(EMochiColliderType::Sdf) == static_cast<int>(mochi::ColliderType::Sdf));
static_assert(static_cast<int>(EMochiColliderType::PointCloud) == static_cast<int>(mochi::ColliderType::PointCloud));

static_assert(static_cast<int>(EMochiSoftMaterialType::NeoHookean) == static_cast<int>(mochi::SoftMaterialType::NeoHookean));
static_assert(static_cast<int>(EMochiSoftMaterialType::StVenantKirchhoff) == static_cast<int>(mochi::SoftMaterialType::StVenantKirchhoff));
static_assert(static_cast<int>(EMochiSoftMaterialType::LinearElastic) == static_cast<int>(mochi::SoftMaterialType::LinearElastic));
static_assert(static_cast<int>(EMochiSoftMaterialType::ActiveNeoHookean) == static_cast<int>(mochi::SoftMaterialType::ActiveNeoHookean));
static_assert(static_cast<int>(EMochiSoftMaterialType::ActiveShapeTargetingArap) == static_cast<int>(mochi::SoftMaterialType::ActiveShapeTargetingArap));
static_assert(static_cast<int>(EMochiSoftMaterialType::Arap) == static_cast<int>(mochi::SoftMaterialType::Arap));

static_assert(static_cast<int>(EMochiMaterialPsdStrategy::MaterialDefault) == static_cast<int>(mochi::MaterialPsdStrategy::MaterialDefault));
static_assert(static_cast<int>(EMochiMaterialPsdStrategy::None) == static_cast<int>(mochi::MaterialPsdStrategy::None));
static_assert(static_cast<int>(EMochiMaterialPsdStrategy::Projection) == static_cast<int>(mochi::MaterialPsdStrategy::Projection));
static_assert(static_cast<int>(EMochiMaterialPsdStrategy::Fast) == static_cast<int>(mochi::MaterialPsdStrategy::Fast));
static_assert(static_cast<int>(EMochiMaterialPsdStrategy::AbsEigenProjection) == static_cast<int>(mochi::MaterialPsdStrategy::AbsEigenProjection));
static_assert(static_cast<int>(EMochiMaterialPsdStrategy::PerTermProjection) == static_cast<int>(mochi::MaterialPsdStrategy::PerTermProjection));

static_assert(static_cast<int>(EMochiPsdProjectionMode::Never) == static_cast<int>(mochi::PsdProjectionMode::Never));
static_assert(static_cast<int>(EMochiPsdProjectionMode::Always) == static_cast<int>(mochi::PsdProjectionMode::Always));
static_assert(static_cast<int>(EMochiPsdProjectionMode::IfFailRetry) == static_cast<int>(mochi::PsdProjectionMode::IfFailRetry));
static_assert(static_cast<int>(EMochiPsdProjectionMode::IfFailAlways) == static_cast<int>(mochi::PsdProjectionMode::IfFailAlways));
static_assert(static_cast<int>(EMochiPsdProjectionMode::Default) == static_cast<int>(mochi::PsdProjectionMode::Default));

static_assert(static_cast<int>(EMochiNonLinearSolverType::Newton) == static_cast<int>(mochi::NonLinearSolverType::Newton));
static_assert(static_cast<int>(EMochiNonLinearSolverType::Default) == static_cast<int>(mochi::NonLinearSolverType::Default));

static_assert(static_cast<int>(EMochiLsType::None) == static_cast<int>(mochi::LineSearchType::None));
static_assert(static_cast<int>(EMochiLsType::Simple) == static_cast<int>(mochi::LineSearchType::Simple));
static_assert(static_cast<int>(EMochiLsType::Armijo) == static_cast<int>(mochi::LineSearchType::Armijo));
static_assert(static_cast<int>(EMochiLsType::WolfeWeak) == static_cast<int>(mochi::LineSearchType::WolfeWeak));
static_assert(static_cast<int>(EMochiLsType::WolfeStrong) == static_cast<int>(mochi::LineSearchType::WolfeStrong));
static_assert(static_cast<int>(EMochiLsType::ResidualNorm) == static_cast<int>(mochi::LineSearchType::ResidualNorm));
static_assert(static_cast<int>(EMochiLsType::Default) == static_cast<int>(mochi::LineSearchType::Default));

static_assert(static_cast<int>(EMochiLinearSolverType::CG) == static_cast<int>(mochi::LinearSolverType::CG));
static_assert(static_cast<int>(EMochiLinearSolverType::GMRes) == static_cast<int>(mochi::LinearSolverType::GMRES));
static_assert(static_cast<int>(EMochiLinearSolverType::CudaCG) == static_cast<int>(mochi::LinearSolverType::CudaCG));
static_assert(static_cast<int>(EMochiLinearSolverType::CudaGMRes) == static_cast<int>(mochi::LinearSolverType::CudaGMRES));
static_assert(static_cast<int>(EMochiLinearSolverType::AugmentedPCG) == static_cast<int>(mochi::LinearSolverType::AugmentedCG));
static_assert(static_cast<int>(EMochiLinearSolverType::LDLt) == static_cast<int>(mochi::LinearSolverType::LDLT));
static_assert(static_cast<int>(EMochiLinearSolverType::LU) == static_cast<int>(mochi::LinearSolverType::LU));
static_assert(static_cast<int>(EMochiLinearSolverType::AsyncCG) == static_cast<int>(mochi::LinearSolverType::AsyncCG));
static_assert(static_cast<int>(EMochiLinearSolverType::ParallelCG) == static_cast<int>(mochi::LinearSolverType::ParallelCG));
static_assert(static_cast<int>(EMochiLinearSolverType::Default) == static_cast<int>(mochi::LinearSolverType::Default));

static_assert(static_cast<int>(EMochiPreconditionerType::None) == static_cast<int>(mochi::PreconditionerType::None));
static_assert(static_cast<int>(EMochiPreconditionerType::Jacobi) == static_cast<int>(mochi::PreconditionerType::Jacobi));
static_assert(static_cast<int>(EMochiPreconditionerType::BlockJacobi) == static_cast<int>(mochi::PreconditionerType::BlockJacobi));
static_assert(static_cast<int>(EMochiPreconditionerType::SSOR) == static_cast<int>(mochi::PreconditionerType::SSOR));
static_assert(static_cast<int>(EMochiPreconditionerType::BlockSSOR) == static_cast<int>(mochi::PreconditionerType::BlockSSOR));
static_assert(static_cast<int>(EMochiPreconditionerType::LU) == static_cast<int>(mochi::PreconditionerType::LU));
static_assert(static_cast<int>(EMochiPreconditionerType::AMG) == static_cast<int>(mochi::PreconditionerType::AMG));
static_assert(static_cast<int>(EMochiPreconditionerType::SymInverse) == static_cast<int>(mochi::PreconditionerType::SymInverse));
static_assert(static_cast<int>(EMochiPreconditionerType::LDLt) == static_cast<int>(mochi::PreconditionerType::LDLT));
static_assert(static_cast<int>(EMochiPreconditionerType::ILU0) == static_cast<int>(mochi::PreconditionerType::ILU0));
static_assert(static_cast<int>(EMochiPreconditionerType::IChol0) == static_cast<int>(mochi::PreconditionerType::IC0));
static_assert(static_cast<int>(EMochiPreconditionerType::PerActor) == static_cast<int>(mochi::PreconditionerType::PerActor));
static_assert(static_cast<int>(EMochiPreconditionerType::AdaptivePerActor) == static_cast<int>(mochi::PreconditionerType::AdaptivePerActor));
static_assert(static_cast<int>(EMochiPreconditionerType::Default) == static_cast<int>(mochi::PreconditionerType::Default));

static_assert(static_cast<int>(EMochiLinearSolverConvergenceNorm::L2Norm) == static_cast<int>(mochi::LinearSolverConvergenceNorm::ResidualL2));
static_assert(static_cast<int>(EMochiLinearSolverConvergenceNorm::L2NormPreconditioned) == static_cast<int>(mochi::LinearSolverConvergenceNorm::PreconditionedResidualL2));
static_assert(static_cast<int>(EMochiLinearSolverConvergenceNorm::L2NormPerActorWeighted) == static_cast<int>(mochi::LinearSolverConvergenceNorm::ResidualPerActorWeighted));
static_assert(static_cast<int>(EMochiLinearSolverConvergenceNorm::Default) == static_cast<int>(mochi::LinearSolverConvergenceNorm::Default));

static_assert(static_cast<int>(EMochiIntegrationMethod::BackwardEuler) == static_cast<int>(mochi::IntegrationMethod::BackwardEuler));
static_assert(static_cast<int>(EMochiIntegrationMethod::BDF2) == static_cast<int>(mochi::IntegrationMethod::BDF2));
static_assert(static_cast<int>(EMochiIntegrationMethod::Default) == static_cast<int>(mochi::IntegrationMethod::Default));

static_assert(static_cast<int>(EMochiVerbosityLevel::Silent) == static_cast<int>(mochi::VerbosityLevel::Silent));
static_assert(static_cast<int>(EMochiVerbosityLevel::Error) == static_cast<int>(mochi::VerbosityLevel::Error));
static_assert(static_cast<int>(EMochiVerbosityLevel::Warning) == static_cast<int>(mochi::VerbosityLevel::Warning));
static_assert(static_cast<int>(EMochiVerbosityLevel::Verbose) == static_cast<int>(mochi::VerbosityLevel::Verbose));

static_assert(static_cast<int>(EMochiGridSdfResolutionMode::LargestAxis) == static_cast<int>(mochi::GridSdfResolutionMode::LargestAxis));
static_assert(static_cast<int>(EMochiGridSdfResolutionMode::SmallestAxis) == static_cast<int>(mochi::GridSdfResolutionMode::SmallestAxis));
static_assert(static_cast<int>(EMochiGridSdfResolutionMode::MeanAxis) == static_cast<int>(mochi::GridSdfResolutionMode::MeanAxis));
static_assert(static_cast<int>(EMochiGridSdfResolutionMode::LargestEdge) == static_cast<int>(mochi::GridSdfResolutionMode::LargestEdge));
static_assert(static_cast<int>(EMochiGridSdfResolutionMode::SmallestEdge) == static_cast<int>(mochi::GridSdfResolutionMode::SmallestEdge));
static_assert(static_cast<int>(EMochiGridSdfResolutionMode::MeanEdge) == static_cast<int>(mochi::GridSdfResolutionMode::MeanEdge));
static_assert(static_cast<int>(EMochiGridSdfResolutionMode::Explicit) == static_cast<int>(mochi::GridSdfResolutionMode::Explicit));

static_assert((int)EMochiTrackingConstraintType::Joint == static_cast<int>(mochi::PoseConstraintType::Joint));
static_assert((int)EMochiTrackingConstraintType::BoneRotation == static_cast<int>(mochi::PoseConstraintType::LinkRotation));
static_assert((int)EMochiTrackingConstraintType::BoneTranslation == static_cast<int>(mochi::PoseConstraintType::LinkTranslation));
static_assert((int)EMochiTrackingConstraintType::Count == static_cast<int>(mochi::PoseConstraintType::Count));
static_assert((int)EMochiTrackingConstraintType::Invalid == static_cast<int>(mochi::PoseConstraintType::Invalid));

static_assert((int)EMochiArticulatedJointType::Free ==  static_cast<int>(mochi::ArticulatedJointType::Free));
static_assert((int)EMochiArticulatedJointType::Prismatic ==  static_cast<int>(mochi::ArticulatedJointType::Prismatic));
static_assert((int)EMochiArticulatedJointType::Revolute ==  static_cast<int>(mochi::ArticulatedJointType::Revolute));
static_assert((int)EMochiArticulatedJointType::Spherical ==  static_cast<int>(mochi::ArticulatedJointType::Spherical));
static_assert((int)EMochiArticulatedJointType::Hard ==  static_cast<int>(mochi::ArticulatedJointType::Hard));
static_assert((int)EMochiArticulatedJointType::Cycle ==  static_cast<int>(mochi::ArticulatedJointType::Cycle));

// clang-format on

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
std::optional<mochi::ActorSegmentElementType> ToMochi(EMochiActorSegmentElementType Type) {
  switch (Type) {
    case EMochiActorSegmentElementType::Nodal:
      return std::nullopt;
    case EMochiActorSegmentElementType::P1Q1:
      return mochi::ActorSegmentElementType::P1Q1;
    case EMochiActorSegmentElementType::P1Q2:
      return mochi::ActorSegmentElementType::P1Q2;
    case EMochiActorSegmentElementType::P1Q3:
      return mochi::ActorSegmentElementType::P1Q3;
  }
  return std::nullopt;
}

std::optional<mochi::ActorBoundaryElementType> ToMochi(EMochiActorBoundaryElementType Type) {
  switch (Type) {
    case EMochiActorBoundaryElementType::Nodal:
      return std::nullopt;
    case EMochiActorBoundaryElementType::P1Q1:
      return mochi::ActorBoundaryElementType::P1Q1;
    case EMochiActorBoundaryElementType::P1Q3:
      return mochi::ActorBoundaryElementType::P1Q3;
    case EMochiActorBoundaryElementType::P1Q6:
      return mochi::ActorBoundaryElementType::P1Q6;
  }
  return std::nullopt;
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
