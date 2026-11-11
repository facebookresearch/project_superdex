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

#include "MochiArticulationTypes.h"

#include "MochiAssets.h"

#include <limits>

namespace {

// Rotational joint-space vectors (revolute axes, spherical limits and DOFs) are pseudovectors:
// a basis change that flips handedness also negates them. This sign keeps a positive joint
// position meaning the same rotation in both spaces.
double PseudovectorSign(EMochiArticulatedJointType Type, bool bFlipsHandedness) {
  const bool bRotational =
      Type == EMochiArticulatedJointType::Revolute || Type == EMochiArticulatedJointType::Spherical;
  return bRotational && bFlipsHandedness ? -1.0 : 1.0;
}

FVector ToFVector(const mochi::Real3& V) {
  return FVector(V[0], V[1], V[2]);
}

mochi::Real3 ToReal3(const FVector& V) {
  return mochi::Real3(
      static_cast<mochi::real>(V.X), static_cast<mochi::real>(V.Y), static_cast<mochi::real>(V.Z));
}

// Basis changes are matrix products, so unlimited (infinite) limit components go through them as
// huge finite values instead.
constexpr double kHugeLimit = 1e30;

// Spherical limits bound each rotation vector component separately. A basis change that negates a
// component turns its lower bound into an upper bound and vice versa, so a one-sided limit can
// become two-sided. ComponentSigns is the converted (1, 1, 1) vector; the limits are already
// converted.
void SortSphericalLimits(
    const FVector& ComponentSigns,
    bool& bInOutHasMin,
    FVector& InOutMin,
    bool& bInOutHasMax,
    FVector& InOutMax) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  FVector lower(-kInf);
  FVector upper(kInf);
  for (int32 i = 0; i < 3; ++i) {
    const bool bFlipped = ComponentSigns[i] < 0;
    const bool bHasLower = bFlipped ? bInOutHasMax : bInOutHasMin;
    const bool bHasUpper = bFlipped ? bInOutHasMin : bInOutHasMax;
    const double lowerValue = bFlipped ? InOutMax[i] : InOutMin[i];
    const double upperValue = bFlipped ? InOutMin[i] : InOutMax[i];
    if (bHasLower && lowerValue > -kHugeLimit) {
      lower[i] = lowerValue;
    }
    if (bHasUpper && upperValue < kHugeLimit) {
      upper[i] = upperValue;
    }
  }
  bInOutHasMin = lower != FVector(-kInf);
  bInOutHasMax = upper != FVector(kInf);
  InOutMin = bInOutHasMin ? lower : FVector::ZeroVector;
  InOutMax = bInOutHasMax ? upper : FVector::ZeroVector;
}

// Shape and render model scales are per-axis magnitudes; a basis change only permutes them.
FVector ScaleToUnreal(const mochi::Real3& S, const FMochiToUnrealConverter& M2U) {
  return M2U.DirectionToUnreal(S).GetAbs();
}

template <typename TVectorConverter>
void ConvertArticulatedDofs(
    TConstArrayView<EMochiArticulatedJointType> Types,
    bool bIncludesRootFreeDofs,
    TArrayView<double> Dofs,
    TVectorConverter VectorConverter) {
  int32 iDof = 0;
  auto convertVector = [&](bool bRotational) {
    if (iDof + 3 > Dofs.Num()) {
      return;
    }
    const FVector converted =
        VectorConverter(FVector(Dofs[iDof], Dofs[iDof + 1], Dofs[iDof + 2]), bRotational);
    Dofs[iDof] = converted.X;
    Dofs[iDof + 1] = converted.Y;
    Dofs[iDof + 2] = converted.Z;
  };
  for (int32 iJoint = 0; iJoint < Types.Num(); ++iJoint) {
    switch (Types[iJoint]) {
      case EMochiArticulatedJointType::Free:
        if (iJoint == 0 && !bIncludesRootFreeDofs) {
          break;
        }
        convertVector(false);
        iDof += 3;
        convertVector(true);
        iDof += 3;
        break;
      case EMochiArticulatedJointType::Spherical:
        convertVector(true);
        iDof += 3;
        break;
      case EMochiArticulatedJointType::Prismatic:
      case EMochiArticulatedJointType::Revolute:
        iDof += 1;
        break;
      case EMochiArticulatedJointType::Hard:
      case EMochiArticulatedJointType::Cycle:
        break;
    }
  }
}

} // namespace

int32 GetArticulatedJointNumDofs(EMochiArticulatedJointType Type) {
  switch (Type) {
    case EMochiArticulatedJointType::Free:
      return 6;
    case EMochiArticulatedJointType::Spherical:
      return 3;
    case EMochiArticulatedJointType::Prismatic:
    case EMochiArticulatedJointType::Revolute:
      return 1;
    case EMochiArticulatedJointType::Hard:
    case EMochiArticulatedJointType::Cycle:
      return 0;
  }
  return 0;
}

//------------------------------------------------------------------------------
// FMochiArticulatedJointFriction
//------------------------------------------------------------------------------

mochi::ArticulatedJointFrictionParams FMochiArticulatedJointFriction::ToMochi() const {
  mochi::ArticulatedJointFrictionParams params;
  params.viscous = Viscous;
  params.coulomb = Coulomb;
  params.falloffVel = FalloffVel;
  params.stictionExtra = StictionExtra;
  params.stribeckVel = StribeckVel;
  return params;
}

void FMochiArticulatedJointFriction::FromMochi(const mochi::ArticulatedJointFrictionParams& Mochi) {
  Viscous = Mochi.viscous;
  Coulomb = Mochi.coulomb;
  FalloffVel = Mochi.falloffVel;
  StictionExtra = Mochi.stictionExtra;
  StribeckVel = Mochi.stribeckVel;
}

//------------------------------------------------------------------------------
// FMochiArticulatedJoint
//------------------------------------------------------------------------------

int32 FMochiArticulatedJoint::GetNumDofs() const {
  return GetArticulatedJointNumDofs(Type);
}

void FMochiArticulatedJoint::GetDofLimits(int32 DofIndex, double& OutLower, double& OutUpper)
    const {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  OutLower = -kInf;
  OutUpper = +kInf;
  if (Type == EMochiArticulatedJointType::Prismatic ||
      Type == EMochiArticulatedJointType::Revolute) {
    const FVector axis = Axis.GetSafeNormal();
    if (bHasMinLimit) {
      OutLower = FVector::DotProduct(MinLimit, axis);
    }
    if (bHasMaxLimit) {
      OutUpper = FVector::DotProduct(MaxLimit, axis);
    }
  } else if (Type == EMochiArticulatedJointType::Spherical && DofIndex >= 0 && DofIndex < 3) {
    if (bHasMinLimit) {
      OutLower = MinLimit[DofIndex];
    }
    if (bHasMaxLimit) {
      OutUpper = MaxLimit[DofIndex];
    }
  }
  // Mochi accepts limits in either order.
  if (OutLower > OutUpper) {
    Swap(OutLower, OutUpper);
  }
}

mochi::ArticulatedJointParams FMochiArticulatedJoint::ToMochi(
    const FUnrealToMochiConverter& U2M) const {
  const double sign = PseudovectorSign(Type, U2M.FlipsHandedness());
  mochi::ArticulatedJointParams params;
  params.name = TCHAR_TO_ANSI(*Name.ToString());
  params.type = static_cast<mochi::ArticulatedJointType>(Type);
  params.parentLinkFromJoint = U2M.TransformToMochi_RT(ParentLinkFromJoint);
  if (Type == EMochiArticulatedJointType::Prismatic ||
      Type == EMochiArticulatedJointType::Revolute) {
    params.axis = U2M.DirectionToMochi(sign * Axis);
  }
  params.friction = Friction.ToMochi();
  if (bOverrideInertia) {
    params.inertia = Inertia;
  }
  // Limits are joint-space quantities: change their basis like Axis but keep their units.
  bool bHasMin = bHasMinLimit;
  FVector minLimit = ToFVector(U2M.DirectionToMochi(sign * MinLimit.BoundToCube(kHugeLimit)));
  bool bHasMax = bHasMaxLimit;
  FVector maxLimit = ToFVector(U2M.DirectionToMochi(sign * MaxLimit.BoundToCube(kHugeLimit)));
  if (Type == EMochiArticulatedJointType::Spherical) {
    SortSphericalLimits(
        ToFVector(U2M.DirectionToMochi(sign * FVector::OneVector)),
        bHasMin,
        minLimit,
        bHasMax,
        maxLimit);
  }
  if (bHasMin) {
    params.minLimit = ToReal3(minLimit);
  }
  if (bHasMax) {
    params.maxLimit = ToReal3(maxLimit);
  }
  params.limitStiffness = LimitStiffness;
  params.limitDamping = LimitDamping;
  return params;
}

void FMochiArticulatedJoint::FromMochi(
    const mochi::ArticulatedJointParams& Mochi,
    const FMochiToUnrealConverter& M2U) {
  Name = FName(UTF8_TO_TCHAR(Mochi.name.c_str()));
  Type = static_cast<EMochiArticulatedJointType>(Mochi.type);
  const double sign = PseudovectorSign(Type, M2U.FlipsHandedness());
  ParentLinkFromJoint = M2U.TransformToUnreal(Mochi.parentLinkFromJoint);
  if (Type == EMochiArticulatedJointType::Prismatic ||
      Type == EMochiArticulatedJointType::Revolute) {
    Axis = sign * M2U.DirectionToUnreal(Mochi.axis);
  }
  Friction.FromMochi(Mochi.friction);
  bOverrideInertia = Mochi.inertia.has_value();
  Inertia = Mochi.inertia.value_or(0);
  bHasMinLimit = Mochi.minLimit.has_value();
  MinLimit = bHasMinLimit ? sign * M2U.DirectionToUnreal(*Mochi.minLimit) : FVector::ZeroVector;
  bHasMaxLimit = Mochi.maxLimit.has_value();
  MaxLimit = bHasMaxLimit ? sign * M2U.DirectionToUnreal(*Mochi.maxLimit) : FVector::ZeroVector;
  if (Type == EMochiArticulatedJointType::Spherical) {
    SortSphericalLimits(
        sign * M2U.DirectionToUnreal(ToReal3(FVector::OneVector)),
        bHasMinLimit,
        MinLimit,
        bHasMaxLimit,
        MaxLimit);
  }
  LimitStiffness = Mochi.limitStiffness;
  LimitDamping = Mochi.limitDamping;
}

//------------------------------------------------------------------------------
// FMochiArticulatedLink
//------------------------------------------------------------------------------

FTransform FMochiArticulatedLink::GetShapeTransform() const {
  return FTransform(ShapeRotation, ShapeTranslation);
}

FTransform FMochiArticulatedLink::GetRenderModelTransform() const {
  return FTransform(RenderModelRotation, RenderModelTranslation, RenderModelScale);
}

mochi::ArticulatedLinkParams FMochiArticulatedLink::ToMochi(
    const FUnrealToMochiConverter& U2M,
    mochi::ShapeHandle InShape,
    FName DefaultLayer,
    const mochi::ContactParams& DefaultContact,
    bool bInHasGravity) const {
  mochi::ArticulatedLinkParams params;
  params.name = TCHAR_TO_ANSI(*Name.ToString());
  params.parentLink = ParentLink;
  params.parentJointFromLink = U2M.TransformToMochi_RT(ParentJointFromLink);
  params.shape = InShape;
  params.layer = TCHAR_TO_ANSI(*(Layer.IsNone() ? DefaultLayer : Layer).ToString());
  params.colliderType = static_cast<mochi::ColliderType>(ColliderType);
  params.contact = ContactParams ? ContactParams->ToMochi() : DefaultContact;
  params.hasGravity = bHasGravity && bInHasGravity;
  if (bOverrideDensity) {
    params.density = Density;
  }
  if (bOverrideMass) {
    params.mass = Mass;
  }
  if (bOverrideCenterOfMass) {
    params.centerOfMass = U2M.TranslationToMochi(CenterOfMass);
  }
  if (bOverrideMomentOfInertia) {
    params.momentOfInertia = U2M.InertiaToMochi(MomentOfInertia);
  }
  if (const auto boundaryElementType = ::ToMochi(BoundaryElementType)) {
    params.boundaryElementType = *boundaryElementType;
  }
  if (bEnableBoundarySubsampling) {
    params.boundarySubsampling = ::ToMochi(BoundarySubsampling);
  }
  return params;
}

void FMochiArticulatedLink::FromMochi(
    const mochi::prefab::ArticulatedLinkPrefab& Mochi,
    const FMochiToUnrealConverter& M2U) {
  Name = FName(UTF8_TO_TCHAR(Mochi.name.c_str()));
  ParentLink = Mochi.parentLink;
  ParentJointFromLink = M2U.TransformToUnreal(Mochi.parentJointFromLink);
  ShapeScale = ScaleToUnreal(Mochi.shapeScale, M2U);
  ShapeRotation = M2U.RotationToUnreal(Mochi.shapeRotation);
  ShapeTranslation = M2U.TranslationToUnreal(Mochi.shapeTranslation);
  RenderModelScale = ScaleToUnreal(Mochi.renderModelScale, M2U);
  RenderModelRotation = M2U.RotationToUnreal(Mochi.renderModelRotation);
  RenderModelTranslation = M2U.TranslationToUnreal(Mochi.renderModelTranslation);
  Layer = Mochi.layer.empty() ? NAME_None : FName(UTF8_TO_TCHAR(Mochi.layer.c_str()));
  ColliderType = static_cast<EMochiColliderType>(Mochi.colliderType);
  bHasGravity = Mochi.hasGravity;
  bOverrideDensity = Mochi.density.has_value();
  Density = Mochi.density.value_or(0);
  bOverrideMass = Mochi.mass.has_value();
  Mass = Mochi.mass.value_or(0);
  bOverrideCenterOfMass = Mochi.centerOfMass.has_value();
  CenterOfMass =
      bOverrideCenterOfMass ? M2U.TranslationToUnreal(*Mochi.centerOfMass) : FVector::ZeroVector;
  bOverrideMomentOfInertia = Mochi.momentOfInertia.has_value();
  MomentOfInertia = bOverrideMomentOfInertia ? M2U.InertiaToUnreal(*Mochi.momentOfInertia)
                                             : FMochiMomentOfInertia{};
  BoundaryElementType = ::FromMochi(Mochi.boundaryElementType);
  bEnableBoundarySubsampling = Mochi.boundarySubsampling.has_value();
  BoundarySubsampling = bEnableBoundarySubsampling ? ::FromMochi(*Mochi.boundarySubsampling)
                                                   : FMochiBoundarySubsamplingParams{};
}

//------------------------------------------------------------------------------
// FMochiArticulatedSkin
//------------------------------------------------------------------------------

mochi::ArticulatedSkinParams FMochiArticulatedSkin::ToMochi(
    mochi::ShapeHandle InShape,
    FName DefaultLayer,
    const mochi::ContactParams& DefaultContact) const {
  mochi::ArticulatedSkinParams params;
  params.shape = InShape;
  params.layer = TCHAR_TO_ANSI(*(Layer.IsNone() ? DefaultLayer : Layer).ToString());
  params.contact = ContactParams ? ContactParams->ToMochi() : DefaultContact;
  if (const auto boundaryElementType = ::ToMochi(BoundaryElementType)) {
    params.boundaryElementType = *boundaryElementType;
  }
  if (bEnableBoundarySubsampling) {
    params.boundarySubsampling = ::ToMochi(BoundarySubsampling);
  }
  if (bOverrideNonCollidingLinks) {
    mochi::DynamicArray<mochi::DynamicString> links;
    links.reserve(NonCollidingLinks.Num());
    for (const FName& link : NonCollidingLinks) {
      links.push_back(mochi::DynamicString(TCHAR_TO_ANSI(*link.ToString())));
    }
    params.nonCollidingLinks = std::move(links);
  }
  return params;
}

void FMochiArticulatedSkin::FromMochi(const mochi::prefab::ArticulatedSkinPrefab& Mochi) {
  Layer = Mochi.layer.empty() ? NAME_None : FName(UTF8_TO_TCHAR(Mochi.layer.c_str()));
  BoundaryElementType = ::FromMochi(Mochi.boundaryElementType);
  bEnableBoundarySubsampling = Mochi.boundarySubsampling.has_value();
  BoundarySubsampling = bEnableBoundarySubsampling ? ::FromMochi(*Mochi.boundarySubsampling)
                                                   : FMochiBoundarySubsamplingParams{};
  bOverrideNonCollidingLinks = Mochi.nonCollidingLinks.has_value();
  NonCollidingLinks.Reset();
  if (bOverrideNonCollidingLinks) {
    for (const auto& link : *Mochi.nonCollidingLinks) {
      NonCollidingLinks.Add(FName(UTF8_TO_TCHAR(link.c_str())));
    }
  }
}

//------------------------------------------------------------------------------
// FMochiArticulatedCycleJoint
//------------------------------------------------------------------------------

mochi::ArticulatedCycleJointParams FMochiArticulatedCycleJoint::ToMochi(
    const FUnrealToMochiConverter& U2M) const {
  mochi::ArticulatedCycleJointParams params;
  params.parentLink = ParentLink;
  params.childLink = ChildLink;
  params.jointFromChildLink = U2M.TransformToMochi_RT(JointFromChildLink);
  params.stiffness = Stiffness;
  return params;
}

void FMochiArticulatedCycleJoint::FromMochi(
    const mochi::ArticulatedCycleJointParams& Mochi,
    const FMochiToUnrealConverter& M2U) {
  ParentLink = Mochi.parentLink;
  ChildLink = Mochi.childLink;
  JointFromChildLink = M2U.TransformToUnreal(Mochi.jointFromChildLink);
  Stiffness = Mochi.stiffness;
}

//------------------------------------------------------------------------------
// FMochiJointControlGains
//------------------------------------------------------------------------------

void FMochiJointControlGains::PopulateTable(
    UDataTable* Table,
    const TArray<FMochiArticulatedJoint>& Joints,
    float Stiffness,
    float Damping) {
  Table->RowStruct = FMochiJointControlGains::StaticStruct();
  Table->EmptyTable();
  for (int32 iJoint = 0; iJoint < Joints.Num(); ++iJoint) {
    const FMochiArticulatedJoint& joint = Joints[iJoint];
    // The root joint's DOFs belong to the base, which has separate gains.
    const bool bRootFree = iJoint == 0 && joint.Type == EMochiArticulatedJointType::Free;
    if (joint.GetNumDofs() > 0 && !bRootFree) {
      FMochiJointControlGains gains;
      gains.JointName = joint.Name;
      gains.Stiffness = Stiffness;
      gains.Damping = Damping;
      Table->AddRow(joint.Name, gains);
    }
  }
}

//------------------------------------------------------------------------------
// DOF conversions
//------------------------------------------------------------------------------

void ConvertArticulatedDofsToMochi(
    TConstArrayView<EMochiArticulatedJointType> Types,
    bool bIncludesRootFreeDofs,
    TArrayView<double> Dofs,
    const FUnrealToMochiConverter& U2M) {
  const double sign = U2M.FlipsHandedness() ? -1.0 : 1.0;
  ConvertArticulatedDofs(
      Types, bIncludesRootFreeDofs, Dofs, [&](const FVector& V, bool bRotational) {
        return ToFVector(U2M.DirectionToMochi(bRotational ? sign * V : V));
      });
}

void ConvertArticulatedDofsToUnreal(
    TConstArrayView<EMochiArticulatedJointType> Types,
    bool bIncludesRootFreeDofs,
    TArrayView<double> Dofs,
    const FMochiToUnrealConverter& M2U) {
  const double sign = M2U.FlipsHandedness() ? -1.0 : 1.0;
  ConvertArticulatedDofs(
      Types, bIncludesRootFreeDofs, Dofs, [&](const FVector& V, bool bRotational) {
        const FVector converted = M2U.DirectionToUnreal(ToReal3(V));
        return bRotational ? sign * converted : converted;
      });
}
