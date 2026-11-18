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
int IndexOf(const T (&array)[N], const T& value) {
  auto it = std::find(std::begin(array), std::end(array), value);
  if (it != std::end(array)) {
    return std::distance(std::begin(array), it);
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
  constexpr float k_MPaToPa = 1e6;
  mochi::SoftMaterialParams result;
  result.type = ::ToMochi(MaterialType);
  result.density = mochi::real(Density);
  result.massDampingCoefficient = mochi::real(MassDampingCoefficient);
  result.stiffnessDampingCoefficient = mochi::real(StiffnessDampingCoefficient);

  auto const ym = mochi::real(YoungsModulus * k_MPaToPa);
  auto const pr = mochi::real(PoissonRatio);
  auto const psd = ::ToMochi(PsdStrategy);
  auto const arapStiffness = mochi::real(ArapStiffness);
  switch (result.type) {
    case mochi::SoftMaterialType::NeoHookean:
      result.neoHookean.youngsModulus = ym;
      result.neoHookean.poissonRatio = pr;
      result.neoHookean.psdStrategy = psd;
      break;
    case mochi::SoftMaterialType::StVenantKirchhoff:
      result.stVenantKirchhoff.youngsModulus = ym;
      result.stVenantKirchhoff.poissonRatio = pr;
      result.stVenantKirchhoff.psdStrategy = psd;
      break;
    case mochi::SoftMaterialType::LinearElastic:
      result.linearElastic.youngsModulus = ym;
      result.linearElastic.poissonRatio = pr;
      break;
    case mochi::SoftMaterialType::ActiveNeoHookean:
      result.activeNeoHookean.passiveIsotropic.youngsModulus = ym;
      result.activeNeoHookean.passiveIsotropic.poissonRatio = pr;
      result.activeNeoHookean.passiveIsotropic.psdStrategy = psd;
      result.activeNeoHookean.activeAnisotropic.alpha = mochi::real(AnisoAlpha);
      result.activeNeoHookean.activeAnisotropic.length = mochi::real(AnisoLength);
      result.activeNeoHookean.activeAnisotropic.anisoDir =
          mochi::Real3(AnisoDirection.X, AnisoDirection.Y, AnisoDirection.Z);
      result.activeNeoHookean.activeAnisotropic.psdStrategy = psd;
      break;
    case mochi::SoftMaterialType::ActiveShapeTargetingArap:
      result.activeShapeTargetingArap.stiffness = arapStiffness;
      result.activeShapeTargetingArap.shapeTargetTensor = ShapeTargetTensor.ToMochi();
      result.activeShapeTargetingArap.psdStrategy = psd;
      break;
    case mochi::SoftMaterialType::Arap:
      result.arap.stiffness = arapStiffness;
      result.arap.psdStrategy = psd;
      break;
    case mochi::SoftMaterialType::Count:
      checkNoEntry();
      break;
  }
  return result;
}

void UMochiMaterialParams::FromMochi(const mochi::SoftMaterialParams& Mochi) {
  constexpr mochi::real k_PaToMPa = 1e-6;
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
      YoungsModulus = Mochi.neoHookean.youngsModulus * k_PaToMPa;
      PoissonRatio = Mochi.neoHookean.poissonRatio;
      PsdStrategy = ToUnreal(Mochi.neoHookean.psdStrategy);
      break;
    case mochi::SoftMaterialType::StVenantKirchhoff:
      YoungsModulus = Mochi.stVenantKirchhoff.youngsModulus * k_PaToMPa;
      PoissonRatio = Mochi.stVenantKirchhoff.poissonRatio;
      PsdStrategy = ToUnreal(Mochi.stVenantKirchhoff.psdStrategy);
      break;
    case mochi::SoftMaterialType::LinearElastic:
      // Linear elastic has a constant, already-PSD Hessian and so no strategy to mirror.
      YoungsModulus = Mochi.linearElastic.youngsModulus * k_PaToMPa;
      PoissonRatio = Mochi.linearElastic.poissonRatio;
      break;
    case mochi::SoftMaterialType::ActiveNeoHookean:
      YoungsModulus = Mochi.activeNeoHookean.passiveIsotropic.youngsModulus * k_PaToMPa;
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
  mochi::Error error;
  Result = mochi::experimental::ShellMaterialParamsFrom3dIsotropic(
      mochi::real(YoungModulus3D),
      mochi::real(PoissonsRatio3D),
      mochi::real(Density3D),
      mochi::real(Thickness),
      error);
  if (error.IsOK()) {
    return true;
  }
  UE_LOG(
      LogMochiPhysics,
      Error,
      TEXT("[UMochiShellMaterialParams] Failed to derive shell material from 3D params: %hs"),
      error.GetDescription());
  FNotificationInfo notification(
      FText::Format(
          NSLOCTEXT(
              "MochiShellMaterialParams",
              "EstimateFailed",
              "Could not estimate shell material: {0}"),
          FText::FromString(UTF8_TO_TCHAR(error.GetDescription()))));
  notification.ExpireDuration = 8.0f;
  notification.bFireAndForget = true;
  if (TSharedPtr<SNotificationItem> notificationItem =
          FSlateNotificationManager::Get().AddNotification(notification)) {
    notificationItem->SetCompletionState(SNotificationItem::CS_Fail);
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
  mochi::experimental::ShellMaterialParams result;
  result.membraneLambda = mochi::real(MembraneLambda);
  result.membraneMu = mochi::real(MembraneMu);
  result.bendingAlpha = mochi::real(BendingAlpha);
  result.bendingBeta = mochi::real(BendingBeta);
  result.density = mochi::real(Density);
  result.massDampingCoefficient = mochi::real(MassDampingCoefficient);
  result.stiffnessDampingCoefficient = mochi::real(StiffnessDampingCoefficient);
  result.membraneCollapseBarrierScale = mochi::real(MembraneCollapseBarrierScale);
  return result;
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
  mochi::experimental::ShellMaterialParams material;
  if (!TryDeriveFrom3DMaterial(material)) {
    return;
  }
  Modify();
  SetElasticAndDensityParameters(material);
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
  mochi::RodMaterialParams result;
  result.linearDensity = mochi::real(LinearDensity);
  result.linearRotationalInertia = mochi::real(LinearRotationalInertia);
  result.axialStiffness = mochi::real(AxialStiffness);
  result.torsionalStiffness = mochi::real(TorsionalStiffness);
  result.flexuralStiffness = {mochi::real(FlexuralStiffness.X), mochi::real(FlexuralStiffness.Y)};
  result.massDampingCoefficient = mochi::real(MassDampingCoefficient);
  result.stiffnessDampingCoefficient = mochi::real(StiffnessDampingCoefficient);
  result.axialCollapseBarrierScale = mochi::real(AxialCollapseBarrierScale);
  return result;
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
  constexpr float k_MPaToPa = 1e6;
  mochi::ContactParams result;
  result.penaltyCoefficient = mochi::real(PenaltyCoefficient * k_MPaToPa);
  result.penaltyThresholdDefault = mochi::real(PenaltyThresholdDist);
  result.penaltyThresholdExtraPadding = mochi::real(PenaltyExtraPadding);
  result.collidingPenaltyThresholdPadding = mochi::real(CollidingPenaltyThresholdPadding);
  result.penaltySmoothingHalfDistance = mochi::real(PenaltySmoothingHalfDistance);
  result.distanceErrorBound = mochi::real(DistanceErrorBound);
  result.viscousFrictionCoefficient = mochi::real(ViscousFrictionCoefficient);
  result.coulombFrictionCoefficient = mochi::real(CoulombFrictionCoefficient);
  result.frictionFalloffVel = mochi::real(FrictionFalloffVel);
  result.preslidingDisplacement = mochi::real(PreslidingDisplacement);
  result.normalViscousDampingCoefficient = mochi::real(NormalViscousDampingCoefficient);
  result.frictionWithColliderNormal = bFrictionWithColliderNormal;
  result.maxAlignmentNormals = mochi::real(MaxAlignmentNormals);
  result.alignmentSmoothingFraction = mochi::real(AlignmentSmoothingFraction);
  return result;
}

void UMochiContactParams::FromMochi(const mochi::ContactParams& Mochi) {
  constexpr mochi::real k_PaToMPa = 1e-6;
  PenaltyCoefficient = Mochi.penaltyCoefficient * k_PaToMPa;
  PenaltyThresholdDist = Mochi.GetPenaltyThresholdDist(false);
  PenaltyExtraPadding = Mochi.GetPenaltyThresholdDist(true) - PenaltyThresholdDist;
  CollidingPenaltyThresholdPadding = Mochi.collidingPenaltyThresholdPadding;
  PenaltySmoothingHalfDistance = Mochi.penaltySmoothingHalfDistance;
  DistanceErrorBound = Mochi.distanceErrorBound;
  ViscousFrictionCoefficient = Mochi.viscousFrictionCoefficient;
  CoulombFrictionCoefficient = Mochi.coulombFrictionCoefficient;
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
  auto mochi = mochi::SolverParams{};
  IntegrationMethod = ::ToUnreal(mochi.integrationMethod);
}

mochi::SolverParams UMochiSolverParams::ToMochi() const {
  mochi::SolverParams result;
  result.nonLinearSolver = ::ToMochi(NonLinearSolverParams);
  result.linearSolver = ::ToMochi(LinearSolverParams);
  result.integrationMethod = ::ToMochi(IntegrationMethod);
  result.experimentalEval = ::ToMochi(ExperimentalEvalParams);
  return result;
}

void UMochiSolverParams::FromMochi(const mochi::SolverParams& Mochi) {
  NonLinearSolverParams = ::ToUnreal(Mochi.nonLinearSolver);
  LinearSolverParams = ::ToUnreal(Mochi.linearSolver);
  IntegrationMethod = ::ToUnreal(Mochi.integrationMethod);
  ExperimentalEvalParams = ::ToUnreal(Mochi.experimentalEval);
}
