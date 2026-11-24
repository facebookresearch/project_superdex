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
constexpr double HugeLimit = 1e30;

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
  constexpr double Inf = std::numeric_limits<double>::infinity();
  FVector Lower(-Inf);
  FVector Upper(Inf);
  for (int32 i = 0; i < 3; ++i) {
    const bool bFlipped = ComponentSigns[i] < 0;
    const bool bHasLower = bFlipped ? bInOutHasMax : bInOutHasMin;
    const bool bHasUpper = bFlipped ? bInOutHasMin : bInOutHasMax;
    const double LowerValue = bFlipped ? InOutMax[i] : InOutMin[i];
    const double UpperValue = bFlipped ? InOutMin[i] : InOutMax[i];
    if (bHasLower && LowerValue > -HugeLimit) {
      Lower[i] = LowerValue;
    }
    if (bHasUpper && UpperValue < HugeLimit) {
      Upper[i] = UpperValue;
    }
  }
  bInOutHasMin = Lower != FVector(-Inf);
  bInOutHasMax = Upper != FVector(Inf);
  InOutMin = bInOutHasMin ? Lower : FVector::ZeroVector;
  InOutMax = bInOutHasMax ? Upper : FVector::ZeroVector;
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
  int32 DofIndex = 0;
  auto ConvertVector = [&](bool bRotational) {
    if (DofIndex + 3 > Dofs.Num()) {
      return;
    }
    const FVector Converted = VectorConverter(
        FVector(Dofs[DofIndex], Dofs[DofIndex + 1], Dofs[DofIndex + 2]), bRotational);
    Dofs[DofIndex] = Converted.X;
    Dofs[DofIndex + 1] = Converted.Y;
    Dofs[DofIndex + 2] = Converted.Z;
  };
  for (int32 JointIndex = 0; JointIndex < Types.Num(); ++JointIndex) {
    switch (Types[JointIndex]) {
      case EMochiArticulatedJointType::Free:
        if (JointIndex == 0 && !bIncludesRootFreeDofs) {
          break;
        }
        ConvertVector(false);
        DofIndex += 3;
        ConvertVector(true);
        DofIndex += 3;
        break;
      case EMochiArticulatedJointType::Spherical:
        ConvertVector(true);
        DofIndex += 3;
        break;
      case EMochiArticulatedJointType::Prismatic:
      case EMochiArticulatedJointType::Revolute:
        DofIndex += 1;
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
  mochi::ArticulatedJointFrictionParams Params;
  Params.viscous = Viscous;
  Params.coulomb = Coulomb;
  Params.preslidingDisplacement = PreslidingDisplacement;
  Params.stictionExtra = StictionExtra;
  Params.stribeckVel = StribeckVel;
  return Params;
}

void FMochiArticulatedJointFriction::FromMochi(const mochi::ArticulatedJointFrictionParams& Mochi) {
  Viscous = Mochi.viscous;
  Coulomb = Mochi.coulomb;
  PreslidingDisplacement = Mochi.preslidingDisplacement;
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
  constexpr double Inf = std::numeric_limits<double>::infinity();
  OutLower = -Inf;
  OutUpper = +Inf;
  if (Type == EMochiArticulatedJointType::Prismatic ||
      Type == EMochiArticulatedJointType::Revolute) {
    const FVector NormalizedAxis = Axis.GetSafeNormal();
    if (bHasMinLimit) {
      OutLower = FVector::DotProduct(MinLimit, NormalizedAxis);
    }
    if (bHasMaxLimit) {
      OutUpper = FVector::DotProduct(MaxLimit, NormalizedAxis);
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
  const double Sign = PseudovectorSign(Type, U2M.FlipsHandedness());
  mochi::ArticulatedJointParams Params;
  Params.name = TCHAR_TO_ANSI(*Name.ToString());
  Params.type = ::ToMochi(Type);
  Params.parentLinkFromJoint = U2M.TransformToMochi_RT(ParentLinkFromJoint);
  if (Type == EMochiArticulatedJointType::Prismatic ||
      Type == EMochiArticulatedJointType::Revolute) {
    Params.axis = U2M.DirectionToMochi(Sign * Axis);
  }
  Params.friction = Friction.ToMochi();
  if (bOverrideInertia) {
    Params.inertia = Inertia;
  }
  // Limits are joint-space quantities: change their basis like Axis but keep their units.
  bool bHasMin = bHasMinLimit;
  FVector MochiMinLimit = ToFVector(U2M.DirectionToMochi(Sign * MinLimit.BoundToCube(HugeLimit)));
  bool bHasMax = bHasMaxLimit;
  FVector MochiMaxLimit = ToFVector(U2M.DirectionToMochi(Sign * MaxLimit.BoundToCube(HugeLimit)));
  if (Type == EMochiArticulatedJointType::Spherical) {
    SortSphericalLimits(
        ToFVector(U2M.DirectionToMochi(Sign * FVector::OneVector)),
        bHasMin,
        MochiMinLimit,
        bHasMax,
        MochiMaxLimit);
  }
  if (bHasMin) {
    Params.minLimit = ToReal3(MochiMinLimit);
  }
  if (bHasMax) {
    Params.maxLimit = ToReal3(MochiMaxLimit);
  }
  Params.limitStiffness = LimitStiffness;
  Params.limitDamping = LimitDamping;
  return Params;
}

void FMochiArticulatedJoint::FromMochi(
    const mochi::ArticulatedJointParams& Mochi,
    const FMochiToUnrealConverter& M2U) {
  Name = FName(UTF8_TO_TCHAR(Mochi.name.c_str()));
  Type = ::ToUnreal(Mochi.type);
  const double Sign = PseudovectorSign(Type, M2U.FlipsHandedness());
  ParentLinkFromJoint = M2U.TransformToUnreal(Mochi.parentLinkFromJoint);
  if (Type == EMochiArticulatedJointType::Prismatic ||
      Type == EMochiArticulatedJointType::Revolute) {
    Axis = Sign * M2U.DirectionToUnreal(Mochi.axis);
  }
  Friction.FromMochi(Mochi.friction);
  bOverrideInertia = Mochi.inertia.has_value();
  Inertia = Mochi.inertia.value_or(0);
  bHasMinLimit = Mochi.minLimit.has_value();
  MinLimit = bHasMinLimit ? Sign * M2U.DirectionToUnreal(*Mochi.minLimit) : FVector::ZeroVector;
  bHasMaxLimit = Mochi.maxLimit.has_value();
  MaxLimit = bHasMaxLimit ? Sign * M2U.DirectionToUnreal(*Mochi.maxLimit) : FVector::ZeroVector;
  if (Type == EMochiArticulatedJointType::Spherical) {
    SortSphericalLimits(
        Sign * M2U.DirectionToUnreal(ToReal3(FVector::OneVector)),
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
  mochi::ArticulatedLinkParams Params;
  Params.name = TCHAR_TO_ANSI(*Name.ToString());
  Params.parentLink = ParentLink;
  Params.parentJointFromLink = U2M.TransformToMochi_RT(ParentJointFromLink);
  Params.shape = InShape;
  Params.layer = TCHAR_TO_ANSI(*(Layer.IsNone() ? DefaultLayer : Layer).ToString());
  Params.colliderType = ::ToMochi(ColliderType);
  Params.contact = ContactParams ? ContactParams->ToMochi() : DefaultContact;
  Params.hasGravity = bHasGravity && bInHasGravity;
  if (bOverrideDensity) {
    Params.density = Density;
  }
  if (bOverrideMass) {
    Params.mass = Mass;
  }
  if (bOverrideCenterOfMass) {
    Params.centerOfMass = U2M.TranslationToMochi(CenterOfMass);
  }
  if (bOverrideMomentOfInertia) {
    Params.momentOfInertia = U2M.InertiaToMochi(MomentOfInertia);
  }
  Params.boundaryElementType = ::ToMochi(BoundaryElementType);
  if (bEnableBoundarySubsampling) {
    Params.boundarySubsampling = ::ToMochi(BoundarySubsampling);
  }
  return Params;
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
  ColliderType = ::ToUnreal(Mochi.colliderType);
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
  BoundaryElementType = ::ToUnreal(Mochi.boundaryElementType);
  bEnableBoundarySubsampling = Mochi.boundarySubsampling.has_value();
  BoundarySubsampling = bEnableBoundarySubsampling ? ::ToUnreal(*Mochi.boundarySubsampling)
                                                   : FMochiBoundarySubsamplingParams{};
}

//------------------------------------------------------------------------------
// FMochiArticulatedSkin
//------------------------------------------------------------------------------

mochi::ArticulatedSkinParams FMochiArticulatedSkin::ToMochi(
    mochi::ShapeHandle InShape,
    FName DefaultLayer,
    const mochi::ContactParams& DefaultContact) const {
  mochi::ArticulatedSkinParams Params;
  Params.shape = InShape;
  Params.layer = TCHAR_TO_ANSI(*(Layer.IsNone() ? DefaultLayer : Layer).ToString());
  Params.contact = ContactParams ? ContactParams->ToMochi() : DefaultContact;
  Params.boundaryElementType = ::ToMochi(BoundaryElementType);
  if (bEnableBoundarySubsampling) {
    Params.boundarySubsampling = ::ToMochi(BoundarySubsampling);
  }
  if (bOverrideNonCollidingLinks) {
    mochi::DynamicArray<mochi::DynamicString> Links;
    Links.reserve(NonCollidingLinks.Num());
    for (const FName& Link : NonCollidingLinks) {
      Links.push_back(mochi::DynamicString(TCHAR_TO_ANSI(*Link.ToString())));
    }
    Params.nonCollidingLinks = std::move(Links);
  }
  return Params;
}

void FMochiArticulatedSkin::FromMochi(const mochi::prefab::ArticulatedSkinPrefab& Mochi) {
  Layer = Mochi.layer.empty() ? NAME_None : FName(UTF8_TO_TCHAR(Mochi.layer.c_str()));
  BoundaryElementType = ::ToUnreal(Mochi.boundaryElementType);
  bEnableBoundarySubsampling = Mochi.boundarySubsampling.has_value();
  BoundarySubsampling = bEnableBoundarySubsampling ? ::ToUnreal(*Mochi.boundarySubsampling)
                                                   : FMochiBoundarySubsamplingParams{};
  bOverrideNonCollidingLinks = Mochi.nonCollidingLinks.has_value();
  NonCollidingLinks.Reset();
  if (bOverrideNonCollidingLinks) {
    for (const mochi::DynamicString& Link : *Mochi.nonCollidingLinks) {
      NonCollidingLinks.Add(FName(UTF8_TO_TCHAR(Link.c_str())));
    }
  }
}

//------------------------------------------------------------------------------
// FMochiArticulatedCycleJoint
//------------------------------------------------------------------------------

mochi::ArticulatedCycleJointParams FMochiArticulatedCycleJoint::ToMochi(
    const FUnrealToMochiConverter& U2M) const {
  mochi::ArticulatedCycleJointParams Params;
  Params.parentLink = ParentLink;
  Params.childLink = ChildLink;
  Params.jointFromChildLink = U2M.TransformToMochi_RT(JointFromChildLink);
  Params.stiffness = Stiffness;
  return Params;
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
  for (int32 JointIndex = 0; JointIndex < Joints.Num(); ++JointIndex) {
    const FMochiArticulatedJoint& Joint = Joints[JointIndex];
    // The root joint's DOFs belong to the base, which has separate gains.
    const bool bRootFree = JointIndex == 0 && Joint.Type == EMochiArticulatedJointType::Free;
    if (Joint.GetNumDofs() > 0 && !bRootFree) {
      FMochiJointControlGains Gains;
      Gains.JointName = Joint.Name;
      Gains.Stiffness = Stiffness;
      Gains.Damping = Damping;
      Table->AddRow(Joint.Name, Gains);
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
  const double Sign = U2M.FlipsHandedness() ? -1.0 : 1.0;
  ConvertArticulatedDofs(
      Types, bIncludesRootFreeDofs, Dofs, [&](const FVector& V, bool bRotational) -> FVector {
        return ToFVector(U2M.DirectionToMochi(bRotational ? Sign * V : V));
      });
}

void ConvertArticulatedDofsToUnreal(
    TConstArrayView<EMochiArticulatedJointType> Types,
    bool bIncludesRootFreeDofs,
    TArrayView<double> Dofs,
    const FMochiToUnrealConverter& M2U) {
  const double Sign = M2U.FlipsHandedness() ? -1.0 : 1.0;
  ConvertArticulatedDofs(
      Types, bIncludesRootFreeDofs, Dofs, [&](const FVector& V, bool bRotational) {
        const FVector Converted = M2U.DirectionToUnreal(ToReal3(V));
        return bRotational ? Sign * Converted : Converted;
      });
}

namespace {

// Converts a one-hot DOF array with ConvertDofs to find where DOF Dof goes.
template <typename TConvertDofs>
bool ConvertArticulatedDofIndex(
    EMochiArticulatedJointType Type,
    int32 Dof,
    TConvertDofs ConvertDofs,
    int32& OutDof,
    double& OutSign) {
  TArray<double> Dofs;
  Dofs.SetNumZeroed(GetArticulatedJointNumDofs(Type));
  if (!Dofs.IsValidIndex(Dof)) {
    return false;
  }
  Dofs[Dof] = 1.0;
  ConvertDofs(MakeArrayView(Dofs));
  OutDof = Dofs.IndexOfByPredicate([](double Value) { return !FMath::IsNearlyZero(Value); });
  OutSign = Dofs[OutDof] > 0 ? 1.0 : -1.0;
  return true;
}

// Converts Min and Max as spherical joint DOFs with ConvertDofs, then reorders each component.
template <typename TConvertDofs>
void ConvertRotationBounds(
    const FVector& Min,
    const FVector& Max,
    TConvertDofs ConvertDofs,
    FVector& OutMin,
    FVector& OutMax) {
  double A[3] = {Min.X, Min.Y, Min.Z};
  double B[3] = {Max.X, Max.Y, Max.Z};
  ConvertDofs(MakeArrayView(A));
  ConvertDofs(MakeArrayView(B));
  OutMin = FVector(A[0], A[1], A[2]).ComponentMin(FVector(B[0], B[1], B[2]));
  OutMax = FVector(A[0], A[1], A[2]).ComponentMax(FVector(B[0], B[1], B[2]));
}

constexpr EMochiArticulatedJointType Spherical = EMochiArticulatedJointType::Spherical;

} // namespace

bool ConvertArticulatedDofIndexToMochi(
    EMochiArticulatedJointType Type,
    int32 Dof,
    const FUnrealToMochiConverter& U2M,
    int32& OutDof,
    double& OutSign) {
  return ConvertArticulatedDofIndex(
      Type,
      Dof,
      [&](TArrayView<double> Dofs) {
        ConvertArticulatedDofsToMochi(MakeArrayView(&Type, 1), true, Dofs, U2M);
      },
      OutDof,
      OutSign);
}

bool ConvertArticulatedDofIndexToUnreal(
    EMochiArticulatedJointType Type,
    int32 Dof,
    const FMochiToUnrealConverter& M2U,
    int32& OutDof,
    double& OutSign) {
  return ConvertArticulatedDofIndex(
      Type,
      Dof,
      [&](TArrayView<double> Dofs) {
        ConvertArticulatedDofsToUnreal(MakeArrayView(&Type, 1), true, Dofs, M2U);
      },
      OutDof,
      OutSign);
}

void ConvertRotationBoundsToMochi(
    const FVector& Min,
    const FVector& Max,
    const FUnrealToMochiConverter& U2M,
    mochi::Real3& OutMin,
    mochi::Real3& OutMax) {
  FVector ConvertedMin;
  FVector ConvertedMax;
  ConvertRotationBounds(
      Min,
      Max,
      [&](TArrayView<double> Dofs) {
        ConvertArticulatedDofsToMochi(MakeArrayView(&Spherical, 1), true, Dofs, U2M);
      },
      ConvertedMin,
      ConvertedMax);
  OutMin = ToReal3(ConvertedMin);
  OutMax = ToReal3(ConvertedMax);
}

void ConvertRotationBoundsToUnreal(
    const mochi::Real3& Min,
    const mochi::Real3& Max,
    const FMochiToUnrealConverter& M2U,
    FVector& OutMin,
    FVector& OutMax) {
  ConvertRotationBounds(
      ToFVector(Min),
      ToFVector(Max),
      [&](TArrayView<double> Dofs) {
        ConvertArticulatedDofsToUnreal(MakeArrayView(&Spherical, 1), true, Dofs, M2U);
      },
      OutMin,
      OutMax);
}
