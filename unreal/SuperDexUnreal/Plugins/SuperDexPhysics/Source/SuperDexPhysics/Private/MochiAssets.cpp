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

#include "MochiAssets.h"
#include "MochiConversions.h"

#include "MochiPhysicsModule.h"

#if WITH_EDITOR
#include "Framework/Notifications/NotificationManager.h"
#include "JsonObjectConverter.h"
#include "Widgets/Notifications/SNotificationList.h"
#endif

template <typename T, size_t N>
int IndexOf(const T (&array)[N], const T& Value) {
  const T* It = std::find(std::begin(array), std::end(array), Value);
  if (It != std::end(array)) {
    return std::distance(std::begin(array), It);
  }
  UE_LOG(LogMochiPhysics, Error, TEXT("Failed to find index of Mochi string argument!"));
  return -1;
}

//------------------------------------------------------------------------------
// MOCHI MATERIAL PARAMS
//------------------------------------------------------------------------------

UMochiMaterialParams::UMochiMaterialParams() {
  FromMochi(mochi::SoftMaterialParams{});
}

mochi::SoftMaterialParams UMochiMaterialParams::ToMochi() const {
  constexpr float MPaToPa = 1e6;
  mochi::SoftMaterialParams Result;
  Result.type = ::ToMochi(MaterialType);
  Result.density = mochi::real(Density);
  Result.massDampingCoefficient = mochi::real(MassDampingCoefficient);
  Result.stiffnessDampingCoefficient = mochi::real(StiffnessDampingCoefficient);

  const mochi::real Ym = mochi::real(YoungsModulus * MPaToPa);
  const mochi::real Pr = mochi::real(PoissonRatio);
  const mochi::MaterialPsdStrategy Psd = ::ToMochi(PsdStrategy);
  const mochi::real MochiArapStiffness = mochi::real(ArapStiffness);
  switch (Result.type) {
    case mochi::SoftMaterialType::NeoHookean:
      Result.neoHookean.youngsModulus = Ym;
      Result.neoHookean.poissonRatio = Pr;
      Result.neoHookean.psdStrategy = Psd;
      break;
    case mochi::SoftMaterialType::StVenantKirchhoff:
      Result.stVenantKirchhoff.youngsModulus = Ym;
      Result.stVenantKirchhoff.poissonRatio = Pr;
      Result.stVenantKirchhoff.psdStrategy = Psd;
      break;
    case mochi::SoftMaterialType::LinearElastic:
      Result.linearElastic.youngsModulus = Ym;
      Result.linearElastic.poissonRatio = Pr;
      break;
    case mochi::SoftMaterialType::ActiveNeoHookean:
      Result.activeNeoHookean.passiveIsotropic.youngsModulus = Ym;
      Result.activeNeoHookean.passiveIsotropic.poissonRatio = Pr;
      Result.activeNeoHookean.passiveIsotropic.psdStrategy = Psd;
      Result.activeNeoHookean.activeAnisotropic.alpha = mochi::real(AnisoAlpha);
      Result.activeNeoHookean.activeAnisotropic.length = mochi::real(AnisoLength);
      Result.activeNeoHookean.activeAnisotropic.anisoDir =
          mochi::Real3(AnisoDirection.X, AnisoDirection.Y, AnisoDirection.Z);
      Result.activeNeoHookean.activeAnisotropic.psdStrategy = Psd;
      break;
    case mochi::SoftMaterialType::ActiveShapeTargetingArap:
      Result.activeShapeTargetingArap.stiffness = MochiArapStiffness;
      Result.activeShapeTargetingArap.shapeTargetTensor = ShapeTargetTensor.ToMochi();
      Result.activeShapeTargetingArap.psdStrategy = Psd;
      break;
    case mochi::SoftMaterialType::Arap:
      Result.arap.stiffness = MochiArapStiffness;
      Result.arap.psdStrategy = Psd;
      break;
    case mochi::SoftMaterialType::Count:
      checkNoEntry();
      break;
  }
  return Result;
}

void UMochiMaterialParams::FromMochi(const mochi::SoftMaterialParams& Mochi) {
  constexpr mochi::real PaToMPa = 1e-6;
  MaterialType = ToUnreal(Mochi.type);
  Density = Mochi.density;
  MassDampingCoefficient = Mochi.massDampingCoefficient;
  StiffnessDampingCoefficient = Mochi.stiffnessDampingCoefficient;

  // mochi::SoftMaterialParams carries every material's parameters at once and picks one
  // by type, where this struct is flat. Seed the type-specific fields from their own
  // sub-struct so they hold Mochi's defaults whatever the type is, then let the switch
  // below overwrite the ones the active type actually owns.
  ArapStiffness = Mochi.arap.stiffness;
  AnisoAlpha = Mochi.activeNeoHookean.activeAnisotropic.alpha;
  AnisoLength = Mochi.activeNeoHookean.activeAnisotropic.length;
  AnisoDirection = FVector3f(
      Mochi.activeNeoHookean.activeAnisotropic.anisoDir[0],
      Mochi.activeNeoHookean.activeAnisotropic.anisoDir[1],
      Mochi.activeNeoHookean.activeAnisotropic.anisoDir[2]);
  ShapeTargetTensor.FromMochi(Mochi.activeShapeTargetingArap.shapeTargetTensor);

  switch (Mochi.type) {
    case mochi::SoftMaterialType::NeoHookean:
      YoungsModulus = Mochi.neoHookean.youngsModulus * PaToMPa;
      PoissonRatio = Mochi.neoHookean.poissonRatio;
      PsdStrategy = ToUnreal(Mochi.neoHookean.psdStrategy);
      break;
    case mochi::SoftMaterialType::StVenantKirchhoff:
      YoungsModulus = Mochi.stVenantKirchhoff.youngsModulus * PaToMPa;
      PoissonRatio = Mochi.stVenantKirchhoff.poissonRatio;
      PsdStrategy = ToUnreal(Mochi.stVenantKirchhoff.psdStrategy);
      break;
    case mochi::SoftMaterialType::LinearElastic:
      // Linear elastic has a constant, already-PSD Hessian and so no strategy to mirror.
      YoungsModulus = Mochi.linearElastic.youngsModulus * PaToMPa;
      PoissonRatio = Mochi.linearElastic.poissonRatio;
      break;
    case mochi::SoftMaterialType::ActiveNeoHookean:
      YoungsModulus = Mochi.activeNeoHookean.passiveIsotropic.youngsModulus * PaToMPa;
      PoissonRatio = Mochi.activeNeoHookean.passiveIsotropic.poissonRatio;
      PsdStrategy = ToUnreal(Mochi.activeNeoHookean.passiveIsotropic.psdStrategy);
      break;
    case mochi::SoftMaterialType::ActiveShapeTargetingArap:
      ArapStiffness = Mochi.activeShapeTargetingArap.stiffness;
      PsdStrategy = ToUnreal(Mochi.activeShapeTargetingArap.psdStrategy);
      break;
    case mochi::SoftMaterialType::Arap:
      PsdStrategy = ToUnreal(Mochi.arap.psdStrategy);
      break;
    case mochi::SoftMaterialType::Count:
      checkNoEntry();
      break;
  }
}

//------------------------------------------------------------------------------
// MOCHI SHELL MATERIAL PARAMS
//------------------------------------------------------------------------------

UMochiShellMaterialParams::UMochiShellMaterialParams() {
  FromMochi(mochi::experimental::ShellMaterialParams{});
}

#if WITH_EDITOR
bool UMochiShellMaterialParams::TryDeriveFrom3DMaterial(
    mochi::experimental::ShellMaterialParams& Result) const {
  mochi::Error Error;
  Result = mochi::experimental::ShellMaterialParamsFrom3dIsotropic(
      mochi::real(YoungModulus3D),
      mochi::real(PoissonsRatio3D),
      mochi::real(Density3D),
      mochi::real(Thickness),
      Error);
  if (Error.IsOK()) {
    return true;
  }
  UE_LOG(
      LogMochiPhysics,
      Error,
      TEXT("[UMochiShellMaterialParams] Failed to derive shell material from 3D params: %hs"),
      Error.GetDescription());
  FNotificationInfo Notification(
      FText::Format(
          NSLOCTEXT(
              "MochiShellMaterialParams",
              "EstimateFailed",
              "Could not estimate shell material: {0}"),
          FText::FromString(UTF8_TO_TCHAR(Error.GetDescription()))));
  Notification.ExpireDuration = 8.0f;
  Notification.bFireAndForget = true;
  if (TSharedPtr<SNotificationItem> NotificationItem =
          FSlateNotificationManager::Get().AddNotification(Notification)) {
    NotificationItem->SetCompletionState(SNotificationItem::CS_Fail);
  }
  return false;
}
#endif

void UMochiShellMaterialParams::SetElasticAndDensityParameters(
    const mochi::experimental::ShellMaterialParams& Mochi) {
  MembraneLambda = Mochi.membraneLambda;
  MembraneMu = Mochi.membraneMu;
  BendingAlpha = Mochi.bendingAlpha;
  BendingBeta = Mochi.bendingBeta;
  Density = Mochi.density;
}

mochi::experimental::ShellMaterialParams UMochiShellMaterialParams::ToMochi() const {
  mochi::experimental::ShellMaterialParams Result;
  Result.membraneLambda = mochi::real(MembraneLambda);
  Result.membraneMu = mochi::real(MembraneMu);
  Result.bendingAlpha = mochi::real(BendingAlpha);
  Result.bendingBeta = mochi::real(BendingBeta);
  Result.density = mochi::real(Density);
  Result.massDampingCoefficient = mochi::real(MassDampingCoefficient);
  Result.stiffnessDampingCoefficient = mochi::real(StiffnessDampingCoefficient);
  Result.membraneCollapseBarrierScale = mochi::real(MembraneCollapseBarrierScale);
  return Result;
}

// Populate from native Mochi type.
void UMochiShellMaterialParams::FromMochi(const mochi::experimental::ShellMaterialParams& Mochi) {
  SetElasticAndDensityParameters(Mochi);
  MassDampingCoefficient = Mochi.massDampingCoefficient;
  StiffnessDampingCoefficient = Mochi.stiffnessDampingCoefficient;
  MembraneCollapseBarrierScale = Mochi.membraneCollapseBarrierScale;
}

#if WITH_EDITOR
void UMochiShellMaterialParams::Apply3DParametersTo2D() {
  mochi::experimental::ShellMaterialParams Material;
  if (!TryDeriveFrom3DMaterial(Material)) {
    return;
  }
  Modify();
  SetElasticAndDensityParameters(Material);
  PostEditChange();
}
#endif

//------------------------------------------------------------------------------
// MOCHI ROD MATERIAL PARAMS
//------------------------------------------------------------------------------

UMochiRodMaterialParams::UMochiRodMaterialParams() {
  FromMochi(mochi::RodMaterialParams{});
}

mochi::RodMaterialParams UMochiRodMaterialParams::ToMochi() const {
  mochi::RodMaterialParams Result;
  Result.linearDensity = mochi::real(LinearDensity);
  Result.linearRotationalInertia = mochi::real(LinearRotationalInertia);
  Result.axialStiffness = mochi::real(AxialStiffness);
  Result.torsionalStiffness = mochi::real(TorsionalStiffness);
  Result.flexuralStiffness = {mochi::real(FlexuralStiffness.X), mochi::real(FlexuralStiffness.Y)};
  Result.massDampingCoefficient = mochi::real(MassDampingCoefficient);
  Result.stiffnessDampingCoefficient = mochi::real(StiffnessDampingCoefficient);
  Result.axialCollapseBarrierScale = mochi::real(AxialCollapseBarrierScale);
  return Result;
}
// Populate from native Mochi type.
void UMochiRodMaterialParams::FromMochi(const mochi::RodMaterialParams& Mochi) {
  LinearDensity = Mochi.linearDensity;
  LinearRotationalInertia = Mochi.linearRotationalInertia;
  AxialStiffness = Mochi.axialStiffness;
  TorsionalStiffness = Mochi.torsionalStiffness;
  FlexuralStiffness = FVector2D(Mochi.flexuralStiffness[0], Mochi.flexuralStiffness[1]);
  MassDampingCoefficient = Mochi.massDampingCoefficient;
  StiffnessDampingCoefficient = Mochi.stiffnessDampingCoefficient;
  AxialCollapseBarrierScale = Mochi.axialCollapseBarrierScale;
}

//------------------------------------------------------------------------------
// MOCHI CONTACT PARAMS
//------------------------------------------------------------------------------

UMochiContactParams::UMochiContactParams() {
  FromMochi(mochi::ContactParams{});
}

mochi::ContactParams UMochiContactParams::ToMochi() const {
  constexpr float MPaToPa = 1e6;
  mochi::ContactParams Result;
  Result.penaltyCoefficient = mochi::real(PenaltyCoefficient * MPaToPa);
  Result.penaltyThresholdDefault = mochi::real(PenaltyThresholdDist);
  Result.penaltyThresholdExtraPadding = mochi::real(PenaltyExtraPadding);
  Result.collidingPenaltyThresholdPadding = mochi::real(CollidingPenaltyThresholdPadding);
  Result.penaltySmoothingHalfDistance = mochi::real(PenaltySmoothingHalfDistance);
  Result.distanceErrorBound = mochi::real(DistanceErrorBound);
  Result.viscousFrictionCoefficient = mochi::real(ViscousFrictionCoefficient);
  Result.coulombFrictionCoefficient = mochi::real(CoulombFrictionCoefficient);
  Result.preferElastoPlasticFriction = bPreferElastoPlasticFriction;
  Result.frictionFalloffVel = mochi::real(FrictionFalloffVel);
  Result.preslidingDisplacement = mochi::real(PreslidingDisplacement);
  Result.normalViscousDampingCoefficient = mochi::real(NormalViscousDampingCoefficient);
  Result.frictionWithColliderNormal = bFrictionWithColliderNormal;
  Result.maxAlignmentNormals = mochi::real(MaxAlignmentNormals);
  Result.alignmentSmoothingFraction = mochi::real(AlignmentSmoothingFraction);
  return Result;
}

void UMochiContactParams::FromMochi(const mochi::ContactParams& Mochi) {
  constexpr mochi::real PaToMPa = 1e-6;
  PenaltyCoefficient = Mochi.penaltyCoefficient * PaToMPa;
  PenaltyThresholdDist = Mochi.GetPenaltyThresholdDist(false);
  PenaltyExtraPadding = Mochi.GetPenaltyThresholdDist(true) - PenaltyThresholdDist;
  CollidingPenaltyThresholdPadding = Mochi.collidingPenaltyThresholdPadding;
  PenaltySmoothingHalfDistance = Mochi.penaltySmoothingHalfDistance;
  DistanceErrorBound = Mochi.distanceErrorBound;
  ViscousFrictionCoefficient = Mochi.viscousFrictionCoefficient;
  CoulombFrictionCoefficient = Mochi.coulombFrictionCoefficient;
  bPreferElastoPlasticFriction = Mochi.preferElastoPlasticFriction;
  FrictionFalloffVel = Mochi.frictionFalloffVel;
  PreslidingDisplacement = Mochi.preslidingDisplacement;
  NormalViscousDampingCoefficient = Mochi.normalViscousDampingCoefficient;
  bFrictionWithColliderNormal = Mochi.frictionWithColliderNormal;
  MaxAlignmentNormals = Mochi.maxAlignmentNormals;
  AlignmentSmoothingFraction = Mochi.alignmentSmoothingFraction;
}

//------------------------------------------------------------------------------
// MOCHI SOLVER PARAMS
//------------------------------------------------------------------------------

UMochiSolverParams::UMochiSolverParams() {
  mochi::SolverParams Mochi = mochi::SolverParams{};
  IntegrationMethod = ::ToUnreal(Mochi.integrationMethod);
}

mochi::SolverParams UMochiSolverParams::ToMochi() const {
  mochi::SolverParams Result;
  Result.nonLinearSolver = ::ToMochi(NonLinearSolverParams);
  Result.linearSolver = ::ToMochi(LinearSolverParams);
  Result.integrationMethod = ::ToMochi(IntegrationMethod);
  Result.experimentalEval = ::ToMochi(ExperimentalEvalParams);
  return Result;
}

void UMochiSolverParams::FromMochi(const mochi::SolverParams& Mochi) {
  NonLinearSolverParams = ::ToUnreal(Mochi.nonLinearSolver);
  LinearSolverParams = ::ToUnreal(Mochi.linearSolver);
  IntegrationMethod = ::ToUnreal(Mochi.integrationMethod);
  ExperimentalEvalParams = ::ToUnreal(Mochi.experimentalEval);
}
