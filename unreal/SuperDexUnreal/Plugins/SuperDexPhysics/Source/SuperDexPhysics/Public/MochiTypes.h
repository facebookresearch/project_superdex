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

#pragma once

#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/coordinate_space.h>
#include <mochi_core/utils/profile.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include <optional>

#include "MochiTypes.generated.h"

//------------------------------------------------------------------------------
// FORWARDS
//------------------------------------------------------------------------------

class UMochiModel;

//------------------------------------------------------------------------------
// LAYERS
//------------------------------------------------------------------------------

struct FMochiLayer {
  static constexpr const char* LeftHand = "LeftHand";
  static constexpr const char* RightHand = "RightHand";
  static constexpr const char* GenericObject = "GenericObject";
};

//------------------------------------------------------------------------------
// SERIALIZATION VERSION
//------------------------------------------------------------------------------

// Serialization version for SuperDexPhysics assets. Saving stamps LatestVersion into
// the package; an asset saved before an entry was added reads that entry back as -1.
struct SUPERDEXPHYSICS_API FMochiObjectVersion {
  enum Type {
    BeforeCustomVersionWasAdded = 0,
    // UMochiModel holds an FMochiModelData, with the SDF values in bulk data
    TypedModelData,
    // -----<new versions can be added above this line>-----
    VersionPlusOne,
    LatestVersion = VersionPlusOne - 1,
  };

  static const FGuid GUID;

  FMochiObjectVersion() = delete;
};

//------------------------------------------------------------------------------
// COORDINATE SPACES
//------------------------------------------------------------------------------

// clang-format off

// 48 FRU combinations, named by the axis Forward, Right, and Up each point along. This
// exhaustive list prevents the user from selecting an invalid space and also provides a
// mechanism to select common spaces by name. Values are mochi::CoordinateSpaceAxes, which
// defines what each convention means; config and assets store the names, not the values.
UENUM()
enum class EMochiSpaceAxes : uint16 {
    SuperDex  = static_cast<uint16>(mochi::CoordinateSpaceAxes::FLU) UMETA(DisplayName = "SuperDex (+X, -Y, +Z)"),
    Mochi     = static_cast<uint16>(mochi::CoordinateSpaceAxes::RUB) UMETA(DisplayName = "Mochi Legacy (-Z, +X, +Y)"),
    Unreal    = static_cast<uint16>(mochi::CoordinateSpaceAxes::FRU) UMETA(DisplayName = "Unreal (+X, +Y, +Z)"),
    Unity     = static_cast<uint16>(mochi::CoordinateSpaceAxes::RUF) UMETA(DisplayName = "Unity (+Z, +X, +Y)"),
    ROS       = static_cast<uint16>(mochi::CoordinateSpaceAxes::FLU) UMETA(DisplayName = "ROS (+X, -Y, +Z)"),

    pXpYpZ = static_cast<uint16>(mochi::CoordinateSpaceAxes::FRU) UMETA(DisplayName = "+X, +Y, +Z"),
    pXpYnZ = static_cast<uint16>(mochi::CoordinateSpaceAxes::FRD) UMETA(DisplayName = "+X, +Y, -Z"),
    pXnYpZ = static_cast<uint16>(mochi::CoordinateSpaceAxes::FLU) UMETA(DisplayName = "+X, -Y, +Z"),
    pXnYnZ = static_cast<uint16>(mochi::CoordinateSpaceAxes::FLD) UMETA(DisplayName = "+X, -Y, -Z"),
    pXpZpY = static_cast<uint16>(mochi::CoordinateSpaceAxes::FUR) UMETA(DisplayName = "+X, +Z, +Y"),
    pXpZnY = static_cast<uint16>(mochi::CoordinateSpaceAxes::FDR) UMETA(DisplayName = "+X, +Z, -Y"),
    pXnZpY = static_cast<uint16>(mochi::CoordinateSpaceAxes::FUL) UMETA(DisplayName = "+X, -Z, +Y"),
    pXnZnY = static_cast<uint16>(mochi::CoordinateSpaceAxes::FDL) UMETA(DisplayName = "+X, -Z, -Y"),

    nXpYpZ = static_cast<uint16>(mochi::CoordinateSpaceAxes::BRU) UMETA(DisplayName = "-X, +Y, +Z"),
    nXpYnZ = static_cast<uint16>(mochi::CoordinateSpaceAxes::BRD) UMETA(DisplayName = "-X, +Y, -Z"),
    nXnYpZ = static_cast<uint16>(mochi::CoordinateSpaceAxes::BLU) UMETA(DisplayName = "-X, -Y, +Z"),
    nXnYnZ = static_cast<uint16>(mochi::CoordinateSpaceAxes::BLD) UMETA(DisplayName = "-X, -Y, -Z"),
    nXpZpY = static_cast<uint16>(mochi::CoordinateSpaceAxes::BUR) UMETA(DisplayName = "-X, +Z, +Y"),
    nXpZnY = static_cast<uint16>(mochi::CoordinateSpaceAxes::BDR) UMETA(DisplayName = "-X, +Z, -Y"),
    nXnZpY = static_cast<uint16>(mochi::CoordinateSpaceAxes::BUL) UMETA(DisplayName = "-X, -Z, +Y"),
    nXnZnY = static_cast<uint16>(mochi::CoordinateSpaceAxes::BDL) UMETA(DisplayName = "-X, -Z, -Y"),

    pYpZpX = static_cast<uint16>(mochi::CoordinateSpaceAxes::UFR) UMETA(DisplayName = "+Y, +Z, +X"),
    pYpZnX = static_cast<uint16>(mochi::CoordinateSpaceAxes::DFR) UMETA(DisplayName = "+Y, +Z, -X"),
    pYnZpX = static_cast<uint16>(mochi::CoordinateSpaceAxes::UFL) UMETA(DisplayName = "+Y, -Z, +X"),
    pYnZnX = static_cast<uint16>(mochi::CoordinateSpaceAxes::DFL) UMETA(DisplayName = "+Y, -Z, -X"),
    pYpXpZ = static_cast<uint16>(mochi::CoordinateSpaceAxes::RFU) UMETA(DisplayName = "+Y, +X, +Z"),
    pYpXnZ = static_cast<uint16>(mochi::CoordinateSpaceAxes::RFD) UMETA(DisplayName = "+Y, +X, -Z"),
    pYnXpZ = static_cast<uint16>(mochi::CoordinateSpaceAxes::LFU) UMETA(DisplayName = "+Y, -X, +Z"),
    pYnXnZ = static_cast<uint16>(mochi::CoordinateSpaceAxes::LFD) UMETA(DisplayName = "+Y, -X, -Z"),

    nYpZpX = static_cast<uint16>(mochi::CoordinateSpaceAxes::UBR) UMETA(DisplayName = "-Y, +Z, +X"),
    nYpZnX = static_cast<uint16>(mochi::CoordinateSpaceAxes::DBR) UMETA(DisplayName = "-Y, +Z, -X"),
    nYnZpX = static_cast<uint16>(mochi::CoordinateSpaceAxes::UBL) UMETA(DisplayName = "-Y, -Z, +X"),
    nYnZnX = static_cast<uint16>(mochi::CoordinateSpaceAxes::DBL) UMETA(DisplayName = "-Y, -Z, -X"),
    nYpXpZ = static_cast<uint16>(mochi::CoordinateSpaceAxes::RBU) UMETA(DisplayName = "-Y, +X, +Z"),
    nYpXnZ = static_cast<uint16>(mochi::CoordinateSpaceAxes::RBD) UMETA(DisplayName = "-Y, +X, -Z"),
    nYnXpZ = static_cast<uint16>(mochi::CoordinateSpaceAxes::LBU) UMETA(DisplayName = "-Y, -X, +Z"),
    nYnXnZ = static_cast<uint16>(mochi::CoordinateSpaceAxes::LBD) UMETA(DisplayName = "-Y, -X, -Z"),

    pZpXpY = static_cast<uint16>(mochi::CoordinateSpaceAxes::RUF) UMETA(DisplayName = "+Z, +X, +Y"),
    pZpXnY = static_cast<uint16>(mochi::CoordinateSpaceAxes::RDF) UMETA(DisplayName = "+Z, +X, -Y"),
    pZnXpY = static_cast<uint16>(mochi::CoordinateSpaceAxes::LUF) UMETA(DisplayName = "+Z, -X, +Y"),
    pZnXnY = static_cast<uint16>(mochi::CoordinateSpaceAxes::LDF) UMETA(DisplayName = "+Z, -X, -Y"),
    pZpYpX = static_cast<uint16>(mochi::CoordinateSpaceAxes::URF) UMETA(DisplayName = "+Z, +Y, +X"),
    pZpYnX = static_cast<uint16>(mochi::CoordinateSpaceAxes::DRF) UMETA(DisplayName = "+Z, +Y, -X"),
    pZnYpX = static_cast<uint16>(mochi::CoordinateSpaceAxes::ULF) UMETA(DisplayName = "+Z, -Y, +X"),
    pZnYnX = static_cast<uint16>(mochi::CoordinateSpaceAxes::DLF) UMETA(DisplayName = "+Z, -Y, -X"),

    nZpXpY = static_cast<uint16>(mochi::CoordinateSpaceAxes::RUB) UMETA(DisplayName = "-Z, +X, +Y"),
    nZpXnY = static_cast<uint16>(mochi::CoordinateSpaceAxes::RDB) UMETA(DisplayName = "-Z, +X, -Y"),
    nZnXpY = static_cast<uint16>(mochi::CoordinateSpaceAxes::LUB) UMETA(DisplayName = "-Z, -X, +Y"),
    nZnXnY = static_cast<uint16>(mochi::CoordinateSpaceAxes::LDB) UMETA(DisplayName = "-Z, -X, -Y"),
    nZpYpX = static_cast<uint16>(mochi::CoordinateSpaceAxes::URB) UMETA(DisplayName = "-Z, +Y, +X"),
    nZpYnX = static_cast<uint16>(mochi::CoordinateSpaceAxes::DRB) UMETA(DisplayName = "-Z, +Y, -X"),
    nZnYpX = static_cast<uint16>(mochi::CoordinateSpaceAxes::ULB) UMETA(DisplayName = "-Z, -Y, +X"),
    nZnYnX = static_cast<uint16>(mochi::CoordinateSpaceAxes::DLB) UMETA(DisplayName = "-Z, -Y, -X")
};
// clang-format on

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiSpaceDefinition {
  GENERATED_BODY()
  // The axes used for Forward, Right, and Up
  UPROPERTY(EditAnywhere, BlueprintReadOnly, DisplayName = "Axes (FRU)")
  EMochiSpaceAxes Axes = EMochiSpaceAxes::Unreal;

  // How many units are in a meter?
  // Note: This feature is intentionally read-only and locked to 1.0 at this time.
  // Mochi should always be simulated at meter scale.
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
  double UnitsPerMeter = 1.0;

  constexpr FMochiSpaceDefinition() = default;
  constexpr FMochiSpaceDefinition(EMochiSpaceAxes InAxes, double InMeterScale)
      : Axes(InAxes), UnitsPerMeter(InMeterScale) {}
};

constexpr bool operator==(const FMochiSpaceDefinition& lhs, const FMochiSpaceDefinition& rhs) {
  return lhs.Axes == rhs.Axes && lhs.UnitsPerMeter == rhs.UnitsPerMeter;
}
constexpr bool operator!=(const FMochiSpaceDefinition& lhs, const FMochiSpaceDefinition& rhs) {
  return !(lhs == rhs);
}

static constexpr FMochiSpaceDefinition kMochiSpace(EMochiSpaceAxes::Mochi, 1.0);
static constexpr FMochiSpaceDefinition kSuperDexSpace(EMochiSpaceAxes::SuperDex, 1.0);
static constexpr FMochiSpaceDefinition kUnrealCentimetersSpace(EMochiSpaceAxes::Unreal, 100.0);
static constexpr FMochiSpaceDefinition kUnrealMetersSpace(EMochiSpaceAxes::Unreal, 1.0);

FORCEINLINE constexpr mochi::CoordinateSpace ToMochiCoordinateSpace(
    const FMochiSpaceDefinition& SpaceDef) {
  return {static_cast<mochi::CoordinateSpaceAxes>(SpaceDef.Axes), SpaceDef.UnitsPerMeter};
}

//------------------------------------------------------------------------------
// ENUMS
//------------------------------------------------------------------------------

// This UEnum matches mochi::ActorType from mochi_physics.h
UENUM(BlueprintType)
enum class EMochiActorType : uint8 { None, Rigid, Soft, Articulated, Shell, Rod };

UENUM(BlueprintType)
enum class EMochiColliderType : uint8 { None, Auto, Sphere, Box, Mesh, Plane, Sdf, PointCloud };

UENUM(BlueprintType)
enum class EMochiSoftMaterialType : uint8 {
  // WARNING: If supporting new material types, please update UMochiMaterialParams::ToMochi
  // and UMochiMaterialParams::FromMochi.
  NeoHookean = 0,
  StVenantKirchhoff = 1,
  LinearElastic = 2,
  ActiveNeoHookean = 3,
  ActiveShapeTargetingArap = 4,
  Arap = 5,
};

UENUM(BlueprintType)
enum class EMochiMaterialPsdStrategy : uint8 {
  MaterialDefault = static_cast<uint8>(mochi::MaterialPsdStrategy::MaterialDefault),
  None = static_cast<uint8>(mochi::MaterialPsdStrategy::None),
  Projection = static_cast<uint8>(mochi::MaterialPsdStrategy::Projection),
  Fast = static_cast<uint8>(mochi::MaterialPsdStrategy::Fast),
  AbsEigenProjection = static_cast<uint8>(mochi::MaterialPsdStrategy::AbsEigenProjection),
  PerTermProjection = static_cast<uint8>(mochi::MaterialPsdStrategy::PerTermProjection),
};

UENUM(BlueprintType)
enum class EMochiPsdProjectionMode : uint8 {
  Never,
  Always,
  IfFailRetry,
  IfFailAlways,
  Default = static_cast<uint8>(mochi::PsdProjectionMode::Default),
};

UENUM(BlueprintType)
enum class EMochiNonLinearSolverType : uint8 {
  Newton,
  Default = static_cast<uint8>(mochi::NonLinearSolverType::Default),
};

UENUM(BlueprintType)
enum class EMochiLsType : uint8 {
  None,
  Simple,
  Armijo,
  WolfeWeak,
  WolfeStrong,
  ResidualNorm,
  Default = static_cast<uint8>(mochi::LineSearchType::Default),
};

UENUM(BlueprintType)
enum class EMochiLinearSolverType : uint8 {
  CG,
  GMRes,
  CudaCG,
  CudaGMRes,
  AugmentedPCG,
  LDLt,
  LU,
  AsyncCG,
  ParallelCG,
  Default = static_cast<uint8>(mochi::LinearSolverType::Default),
};

UENUM(BlueprintType)
enum class EMochiPreconditionerType : uint8 {
  None,
  Jacobi,
  BlockJacobi,
  SSOR,
  BlockSSOR,
  LU,
  AMG,
  SymInverse,
  LDLt,
  ILU0,
  IChol0,
  PerActor,
  AdaptivePerActor = static_cast<uint8>(mochi::PreconditionerType::AdaptivePerActor),
  Default = static_cast<uint8>(mochi::PreconditionerType::Default),
};

UENUM(BlueprintType)
enum class EMochiLinearSolverConvergenceNorm : uint8 {
  L2Norm,
  L2NormPreconditioned,
  L2NormPerActorWeighted =
      static_cast<uint8>(mochi::LinearSolverConvergenceNorm::ResidualPerActorWeighted),
  Default = static_cast<uint8>(mochi::LinearSolverConvergenceNorm::Default),
};

UENUM(BlueprintType)
enum class EMochiIntegrationMethod : uint8 {
  BackwardEuler,
  BDF2,
  Default = static_cast<uint8>(mochi::IntegrationMethod::Default),
};

UENUM(BlueprintType)
enum class EMochiCoulombFrictionModel : uint8 {
  C1Regularized,
  CinfRegularized,
  Default = static_cast<uint8>(mochi::CoulombFrictionModel::Default),
};

UENUM(BlueprintType)
enum class EMochiVerbosityLevel : uint8 {
  Silent,
  Error,
  Warning,
  Verbose,
};

UENUM(BlueprintType)
enum class EMochiGridSdfResolutionMode : uint8 {
  // Aabb dependent
  LargestAxis,
  // Aabb dependent
  SmallestAxis,
  // Aabb dependent
  MeanAxis,
  // Edge length dependent
  LargestEdge,
  // Edge length dependent
  SmallestEdge,
  // Edge length dependent
  MeanEdge,
  // User provided
  Explicit UMETA(Hidden)
};

UENUM(BlueprintType)
enum class EMochiImplicitShape : uint8 { Plane, Sphere };

UENUM(BlueprintType, meta = (Bitflags, UseEnumValuesAsMaskValuesInEditor = "true"))
enum class EMochiFreezeConstraint : uint8 {
  None = 0 UMETA(Hidden),
  FreezeLocationX = 1 << 0,
  FreezeLocationY = 1 << 1,
  FreezeLocationZ = 1 << 2,
  FreezeRotationX = 1 << 3,
  FreezeRotationY = 1 << 4,
  FreezeRotationZ = 1 << 5,
};

UENUM(BlueprintType)
enum class EMochiHyperReductionStrategy : uint8 {
  NoHyperReduction,
  StaticHyperReductionUsingRandomSampling
};

UENUM(Blueprintable)
enum class EMochiContactDirection : uint8 {
  // 'this' object is contacting the 'other' object
  // (i.e the contact point in question lies on the surface of 'this' object)
  ThisContactsOther,
  // the 'other' object is contacting 'this' object
  // (i.e the contact point in question lies on the surface of the 'other' object)
  OtherContactsThis
};

// Matches mochi::ArticulatedJointType
UENUM(Blueprintable)
enum class EMochiArticulatedJointType : uint8 { Free, Prismatic, Revolute, Spherical, Hard, Cycle };

// Tracking constraint type, replica of Mochi's TrackingConstraintType
UENUM(BlueprintType)
enum class EMochiTrackingConstraintType : uint8 {
  Joint = 0, // Joint tracking constraint
  BoneTranslation = 1, // Bone translation tracking constraint
  BoneRotation = 2, // Bone rotation tracking constraint
  Count = 3 UMETA(Hidden),
  Any = 100,
  Invalid = 255 UMETA(Hidden)
};

UENUM(BlueprintType)
enum class EMochiArticulationLinkTransformSpace : uint8 { LinkToParent, LinkToRoot };

// One-dimensional discretization element types for segment (polyline) elements
// used by rod point-cloud colliders. Mirrors mochi::ActorSegmentElementType
// with an extra "Nodal" sentinel representing std::nullopt (no quadrature
// collider - one contact point per node, weighted by lumped segment lengths).
UENUM(BlueprintType)
enum class EMochiActorSegmentElementType : uint8 {
  // No quadrature collider: one contact point per node, weighted by lumped
  // segment lengths. Maps to std::nullopt on the Mochi side.
  Nodal,
  // Linear elements (P1), Gauss-Legendre quadrature with 1 point (Q1).
  P1Q1,
  // Linear elements (P1), Gauss-Legendre quadrature with 2 points (Q2).
  P1Q2,
  // Linear elements (P1), Gauss-Legendre quadrature with 3 points (Q3).
  P1Q3,
  Default = P1Q3,
};

// Convert an EMochiActorSegmentElementType to the corresponding Mochi value.
// Returns std::nullopt for Nodal (the per-node, no-quadrature collider).
std::optional<mochi::ActorSegmentElementType> ToMochi(EMochiActorSegmentElementType Type);

// Two-dimensional discretization element types for triangle (surface) elements
// used by shell point-cloud colliders. Mirrors mochi::ActorBoundaryElementType
// with an extra "Nodal" sentinel representing std::nullopt (no quadrature
// collider - one contact point per node, weighted by lumped triangle areas).
UENUM(BlueprintType)
enum class EMochiActorBoundaryElementType : uint8 {
  // No quadrature collider: one contact point per node, weighted by lumped
  // triangle areas. Maps to std::nullopt on the Mochi side.
  Nodal,
  // Linear elements (P1), Dunavant quadrature with 1 point (Q1).
  P1Q1,
  // Linear elements (P1), Dunavant quadrature with 3 points (Q3).
  P1Q3,
  // Linear elements (P1), Dunavant quadrature with 6 points (Q6).
  P1Q6,
  Default = P1Q3,
};

// Convert an EMochiActorBoundaryElementType to the corresponding Mochi value.
// Returns std::nullopt for Nodal (the per-node, no-quadrature collider).
std::optional<mochi::ActorBoundaryElementType> ToMochi(EMochiActorBoundaryElementType Type);
// Convert a Mochi boundary element type to the closest EMochiActorBoundaryElementType.
SUPERDEXPHYSICS_API EMochiActorBoundaryElementType FromMochi(mochi::ActorBoundaryElementType Type);

UENUM(BlueprintType)
enum class EMochiBoundarySubsamplingStrategy : uint8 {
  UniformProbability,
  AreaProportional,
};

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiBoundarySubsamplingParams {
  GENERATED_BODY()

  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta = (ClampMin = "0.0", ClampMax = "1.0"),
      Category = "Mochi")
  float SubsamplingDensity = 1.0f;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  EMochiBoundarySubsamplingStrategy Strategy =
      EMochiBoundarySubsamplingStrategy::UniformProbability;
};

mochi::BoundarySubsamplingParams ToMochi(const FMochiBoundarySubsamplingParams& Params);
SUPERDEXPHYSICS_API FMochiBoundarySubsamplingParams
FromMochi(const mochi::BoundarySubsamplingParams& Params);

// Which end of a rod a constraint is attached to.
UENUM(BlueprintType)
enum class EMochiRodAttachmentEnd : uint8 {
  First,
  Last,
};

//------------------------------------------------------------------------------
// STRUCTS
//------------------------------------------------------------------------------

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiAsyncStepParams {
  GENERATED_BODY()
  // Whether to use fixed or dynamic time stepping.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bUseFixedTimeStep;
  // Ignored if !useFixedTimeStep
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      meta = (EditCondition = "bUseFixedTimeStep"),
      Category = "Mochi")
  double FixedTimeStepSeconds;
  // Clamp to this minimum time step (fastest rate)
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      meta = (EditCondition = "!bUseFixedTimeStep"),
      Category = "Mochi")
  double DynamicTimeStepMinSeconds;
  // Clamp to this maximum time step (slowest rate)
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      meta = (EditCondition = "!bUseFixedTimeStep"),
      Category = "Mochi")
  double DynamicTimeStepMaxSeconds;

  FMochiAsyncStepParams();
  mochi::AsyncStepParams ToMochi() const;
  void FromMochi(const mochi::AsyncStepParams& Mochi);
};

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiContactPoint {
  GENERATED_BODY()

  // The Actor receiving the FMochiContactPoint data.
  // NOTE: this is the Actor from which the contact point was generated
  //       if and only if ContactDirection equals ThisContactsOther
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  AActor* MochiActor = nullptr;

  // The Component receiving the FMochiContactPoint data.
  // NOTE: this is the Component from which the contact point was generated
  //       if and only if ContactDirection equals ThisContactsOther
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  USceneComponent* MochiComponent = nullptr;

  // If the contact came from a internally nested Mochi actor belonging to an articulated actor,
  // this will be the name of that actor. For a skin, which has no nested actor of its own, it is
  // the name of the articulated actor the skin belongs to. Otherwise it will be None.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FName NestedName = NAME_None;

  // The other Actor in contact.
  // NOTE: this is the actor from which the contact point was generated
  //       if and only if ContactDirection equals OtherContactsThis
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  AActor* OtherMochiActor = nullptr;

  // The other Component in contact.
  // NOTE: this is the Component from which the contact point was generated
  //       if and only if ContactDirection equals OtherContactsThis
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  USceneComponent* OtherMochiComponent = nullptr;

  // The distance between MochiComponent and OtherMochiComponent
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  double Distance = 0;

  // The location of the contact on MochiComponent in world-space.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector Location = FVector::ZeroVector;

  // The location of the contact on OtherMochiComponent in world-space.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector OtherLocation = FVector::ZeroVector;

  // The point velocity of the contact on MochiComponent in world-space.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector Velocity = FVector::ZeroVector;

  // The point velocity of the contact on OtherMochiComponent in world-space.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector OtherVelocity = FVector::ZeroVector;

  // The contact normal pointing from OtherMochiComponent to MochiComponent.
  // NOTE: this corresponds to surface normals on the surface of the Component from which the
  //       contact point was generated, which is determined by the value of ContactDirection:
  //         ThisContactsOther -> MochiComponent
  //         OtherContactsThis -> OtherMochiComponent
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector Normal = FVector::ZeroVector;

  // The contact force applied to MochiComponent from OtherMochiComponent.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector Force = FVector::ZeroVector;

  // ADVANCED

  // The index of the Mochi surface triangle associated with this contact.
  // NOTE: this index corresponds to the object from which the contact point was generated,
  //       which is determined by the value of ContactDirection:
  //         ThisContactsOther -> MochiComponent
  //         OtherContactsThis -> OtherMochiComponent
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  int TriangleIndex = -1;

  // The barycentric coords of the contact within its associated Mochi surface triangle.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector ParametricCoords = FVector::ZeroVector;

  // The integration area of the contact in cm^2 (divide Force by Area to get pressure).
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  float Area = 0;

  // Which direction is the contact in? (this->other or other->this)
  // Equivalently, which object generated the contact point? (other or this)
  // NOTE: this determines how TriangleIndex should be interpreted
  // NOTE: several members of this struct are swapped in the 'OtherContactsThis' case
  //       so that force vectors always point in a consistent direction
  //       (and such that summing over all contacts for an object
  //       yields the correct total net contact force for that object)
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  EMochiContactDirection ContactDirection = EMochiContactDirection::ThisContactsOther;

  // The name of the internal Mochi actor originating contact, not to be confused with Unreal
  // actors.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FName InternalMochiActor;

  // The name of the other internal Mochi actor originating contact, not to be confused with Unreal
  // actors.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FName OtherInternalMochiActor;
};

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiContactMatrixEntry {
  GENERATED_BODY()
  // Name of the layer of the "colliding" actor (the one checking for contact).
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FName LayerA;
  // Name of the layer of the "collider" actor (the one with a ColliderType (e.g. SDF) being
  // tested).
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FName LayerB;
  // True to enable contact, false to disable it.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  bool bEnabled = true;
  // If true, an additional entry is created with swapped layer arguments.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  bool bSymmetric = false;
};

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiRomParams {
  GENERATED_BODY()
  // Specifies a ROM inside the rom group of a .mochi.h5 file.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  FString Source;
  // Hyper-reduction quadrature strategy for this ROM.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  EMochiHyperReductionStrategy HyperReductionStrategy =
      EMochiHyperReductionStrategy::StaticHyperReductionUsingRandomSampling;
  // Parameter that defines a sample mesh when subsampling the boundary elements.
  UPROPERTY(
      EditAnywhere,
      meta =
          (EditCondition =
               "HyperReductionStrategy == EMochiHyperReductionStrategy::StaticHyperReductionUsingRandomSampling",
           EditConditionHides),
      Category = "Mochi")
  int StepSizeForBoundaryElementsSelection = 1;
  // Parameter that defines a sample mesh when subsampling the interior elements.
  UPROPERTY(
      EditAnywhere,
      meta =
          (EditCondition =
               "HyperReductionStrategy == EMochiHyperReductionStrategy::StaticHyperReductionUsingRandomSampling",
           EditConditionHides),
      Category = "Mochi")
  int StepSizeForInteriorElementsSelection = 1;

 public:
  // Convert to native Mochi type.
  mochi::experimental::RomParams ToMochi() const;
};

// Moments of inertia in kg * cm^2
// Taken at the link center of mass and aligned with parent joint coordinate system.
// Note that inertial values from imported URDFs assume a negative product of
// inertia convention per the URDF spec:
//
// [+Ixx -Ixy -Ixz]
// [-Iyx +Iyy -Iyz]
// [-Izx -Izy +Izz]
//
// whereas inertia from CAD source may assume a positive convention:
//
// [+Ixx +Ixy +Ixz]
// [+Iyx +Iyy +Iyz]
// [+Izx +Izy +Izz]
//
// Read more at:
// https://wiki.ros.org/urdf/XML/link
// https://www.mathworks.com/help/releases/R2021b/physmod/sm/ug/specify-custom-inertia.html#mw_b043ec69-835b-4ca9-8769-af2e6f1b190c
//
// Handling the conversion of these and all permutations is tricky. For future
// generations, here's how we go from Solidworks, to .urdf, to Unreal, to Mochi.
// clang-format off
//
// COM: [x y z]
// MOI: [ixx ixy ixz iyy iyz izz]
//
// 1. Solidworks (RH FLU, meters, "positive" notation)
// COM: [0.0167    0.0833    0.2000]
// MOI: [0.0933    0.0117    0.0050    0.0533    0.0250    0.0667]
//
// 2. foo.urdf (RH FLU, meters, "negative" notation)
// COM: [0.0167    0.0833    0.2000] (no change)
// MOI: [0.0933   -0.0117   -0.0050    0.0533   -0.0250    0.0667] (negate off-diag products of inertia)
//
// 3. FMochiMomentOfInertia (LH FRU, centimeters, "negative" notation)
// COM: [1.6667   -8.3333   20.0000] (scale by 100; negate y)
// MOI: [933.3333  116.6667  -50.0000  533.3333  250.0000  666.6667] (scale by 10,000; negate ixy and iyz)
//
// 4. FMochiInertialParams (LH FRU, centimeters, "negative" notation)
// COM: [1.666667, -8.333333, 20] (same)
// MOI: [933.3333  116.6667  -50.0000  533.3333  250.0000  666.6667] (same)
//
// 5. mochi::InertialParams (LH FRU, m, "negative" notation)
// COM: [0.0167   -0.0833    0.2000] (scale by 0.01)
// MOI: [0.0933    0.0117   -0.0050    0.0533    0.0250    0.0667] (scale by 0.0001)
//
// clang-format on

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiMomentOfInertia {
  GENERATED_BODY()
  // The moment of inertia tensor taken at the actor's local center of mass and aligned
  // with the local ref system (root transform), using *negative* tensor notation:
  //     [ixx, ixy, ixz]   [ +∫(y^2 + z^2)dm, -∫(xy)dm,        -∫(xz)dm        ]
  // I = [ixy, iyy, iyz] = [ -∫(xy)dm,        +∫(x^2 + z^2)dm, -∫(yz)dm        ]
  //     [ixz, iyz, izz]   [ -∫(xz)dm,        -∫(yz)dm,        +∫(x^2 + y^2)dm ]
  // This is consistent with the formal definition and tensors defined in e.g. robotics
  // URDF files, but many CAD software (e.g. Solidworks) use a *positive* tensor
  // notation and require negation of the off-diagonal products of inertia upon entry.
  // Read https://www.mathworks.com/help/sm/ug/specify-custom-inertia.html.

  // Ixx = I(1,1) [kg*cm^2]
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  double Ixx = 0;
  // Ixy = I(1,2) [kg*cm^2]
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  double Ixy = 0;
  // Ixz = I(1,3) [kg*cm^2]
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  double Ixz = 0;
  // Iyy = I(2,2) [kg*cm^2]
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  double Iyy = 0;
  // Iyz = I(2,3) [kg*cm^2]
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  double Iyz = 0;
  // Izz = I(3,3) [kg*cm^2]
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  double Izz = 0;

 public:
  // Returns true if all elements are zero.
  bool IsZero() const {
    return Ixx == 0 && Ixy == 0 && Ixz == 0 && Iyy == 0 && Iyz == 0 && Izz == 0;
  }
};

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiGridSdfParams {
  GENERATED_BODY()
  // Reference measurement used for computing the size of the SDF voxel
  UPROPERTY(EditAnywhere, Category = "Mochi")
  EMochiGridSdfResolutionMode ResolutionMode;
  // For a measurement M (e.g., mean edge length), delta = resolutionDelta * M
  UPROPERTY(EditAnywhere, Category = "Mochi")
  FVector ResolutionDelta;
  // Distance around the Aabb to consider when computing the range of the SDF
  UPROPERTY(EditAnywhere, Category = "Mochi")
  float BoundaryPaddingDist;

  FMochiGridSdfParams();
  mochi::GridSdfParams ToMochi() const;
  void FromMochi(const mochi::GridSdfParams& Mochi);
};

// Mirrors mochi::ActiveShapeTargetingArapMaterialParams::shapeTargetTensor, which is a
// Real6 that UHT cannot express. The desired local shape is the identity plus the
// symmetric matrix [[XX, XY, XZ], [XY, YY, YZ], [XZ, YZ, ZZ]]; the material resists
// deformation away from it while still allowing rigid rotation. All zero means no
// targeting, which is why that is the default.
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiShapeTargetTensor {
  GENERATED_BODY()
  UPROPERTY(EditAnywhere, Category = "Mochi")
  float XX = 0.0f;
  UPROPERTY(EditAnywhere, Category = "Mochi")
  float XY = 0.0f;
  UPROPERTY(EditAnywhere, Category = "Mochi")
  float XZ = 0.0f;
  UPROPERTY(EditAnywhere, Category = "Mochi")
  float YY = 0.0f;
  UPROPERTY(EditAnywhere, Category = "Mochi")
  float YZ = 0.0f;
  UPROPERTY(EditAnywhere, Category = "Mochi")
  float ZZ = 0.0f;

  mochi::Real6 ToMochi() const;
  void FromMochi(const mochi::Real6& Mochi);
};

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiMaterialModifier {
  GENERATED_BODY()

  FMochiMaterialModifier() {
    constexpr mochi::real k_PaToMPa = 1e-6;
    YoungsModulus = mochi::kDefaultYoungsModulus * k_PaToMPa;
    PoissonRatio = mochi::kDefaultPoissonRatio;
  }

  UPROPERTY(
      EditAnywhere,
      DisplayName = "Young's Modulus",
      meta = (ClampMin = 0, UIMin = 0),
      meta = (ForceUnit = "MPa"),
      Category = "Mochi")
  float YoungsModulus;

  UPROPERTY(EditAnywhere, Category = "Mochi")
  float PoissonRatio;
};

USTRUCT(BlueprintType)
struct FMochiArticulationSoftLink {
  GENERATED_BODY()
  // The Model to use for the soft link. The Model H5 must have constrained
  // nodes, which currently cannot be authored in Unreal.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  UMochiModel* Model = nullptr;
  // Scale applied to the Mochi Model when its shape is loaded.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector ModelScale = FVector::OneVector;
  // Rotation applied to the Mochi Model when its shape is loaded.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector ModelOriginRPY = FVector::ZeroVector;
  // Translation applied to the Mochi Model when its shape is loaded.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector ModelOriginXYZ = FVector::ZeroVector;
  // Name of the soft link. This should be uniquely different from the link names
  // used in UMochiArticulation::Links
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FName Name = NAME_None;
  // Name of the rigid link that will be the attach parent.
  // This must be a link named in UMochiArticulation::Links
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta = (GetOptions = "GetLinkNames_EditorOnly"),
      Category = "Mochi")
  FName ParentName = NAME_None;
};

USTRUCT(BlueprintType)
struct FMochiArticulationConstraint {
  GENERATED_USTRUCT_BODY()
  UPROPERTY(
      EditAnywhere,
      DisplayName = "Link A Name",
      BlueprintReadWrite,
      meta = (GetOptions = "GetLinkNames_EditorOnly"),
      Category = "Mochi")
  FName LinkAName = NAME_None;
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector LinkAPosition = FVector::ZeroVector;
  UPROPERTY(
      EditAnywhere,
      DisplayName = "Link B Name",
      BlueprintReadWrite,
      meta = (GetOptions = "GetLinkNames_EditorOnly"),
      Category = "Mochi")
  FName LinkBName = NAME_None;
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector LinkBPosition = FVector::ZeroVector;
  // Stiffness of constraint [N/m]
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  float Stiffness = 10000.0f;
  // Damping on the constraint limit [Ns/m]
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  float Damping = 0.0f;
  // Displacement at which the constraint stiffness saturates, -1 for no saturation
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  float Saturation = -1.0f;
};

// Enables or disables contact between two links of an articulation, after the articulation is
// created. Adjacent links have contact disabled by default.
USTRUCT(BlueprintType)
struct FMochiArticulationContactOverride {
  GENERATED_USTRUCT_BODY()
  UPROPERTY(
      EditAnywhere,
      DisplayName = "Link A Name",
      BlueprintReadWrite,
      meta = (GetOptions = "GetLinkNames_EditorOnly"),
      Category = "Mochi")
  FName LinkAName = NAME_None;
  UPROPERTY(
      EditAnywhere,
      DisplayName = "Link B Name",
      BlueprintReadWrite,
      meta = (GetOptions = "GetLinkNames_EditorOnly"),
      Category = "Mochi")
  FName LinkBName = NAME_None;
  // Whether contact between the links is enabled.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  bool bEnable = false;
};

// Kept here rather than in SuperDexRobotics: the wrapped mochi::experimental::IKSolverParams
// is owned by mochi_physics, so this needs no superdex_robotics. Only the solver object does.
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiIKSolverParams {
  GENERATED_BODY()
  // Maximum Newton solver iterations
  UPROPERTY(EditAnywhere, BlueprintReadOnly)
  int MaxIter;
  // Absolute error tolerance for Newton solver convergence
  UPROPERTY(EditAnywhere, BlueprintReadOnly)
  double AbsTol;
  // Relative error tolerance for Newton solver convergence (scaled by RelTolScalar).
  // Actual relTol = RelTol / RelTolScalar
  UPROPERTY(EditAnywhere, BlueprintReadOnly)
  double RelTol;
  // Scalar applied to RelTol for UI display. Actual relTol = RelTol / RelTolScalar
  UPROPERTY(EditAnywhere, BlueprintReadOnly)
  double RelTolScalar;
  // Position error threshold for determining if IK targets are reachable [m]
  UPROPERTY(EditAnywhere, BlueprintReadOnly)
  double PositionErrorThres;
  // Rotation error threshold for determining if IK targets are reachable [radians]
  UPROPERTY(EditAnywhere, BlueprintReadOnly)
  double RotationErrorThres;
  // Maximum number of line search iterations per Newton step
  UPROPERTY(EditAnywhere, BlueprintReadOnly)
  int LineSearchMaxIter;
  // Maximum elapsed wall-clock time [seconds] before the solver stops. 0 means no time limit.
  UPROPERTY(EditAnywhere, BlueprintReadOnly)
  double MaxElapsedTimeSeconds;

  FMochiIKSolverParams();
  mochi::experimental::IKSolverParams ToMochi() const;
  void FromMochi(const mochi::experimental::IKSolverParams& Mochi);
};

// Couples two single-dof joints of one articulation so that
//   JointB = Ratio * JointA + Offset.
// Wraps a Mochi linear transmission driven by a displacement-control actuator, so the coupling is
// a spring penalty rather than a hard constraint: it is satisfied to roughly (solver absolute
// tolerance / Stiffness). The penalty is symmetric, so both joints move to satisfy it; A and B
// only name the direction of Ratio. Useful for resolving a redundant dof, e.g. tying a finger's
// distal joint to its proximal one.
// Entries are independent and their penalties simply add, so a joint may appear in more than one
// coupling and chains such as A->B, B->C are allowed. Nothing checks that a set of couplings is
// mutually consistent; an over-constrained set settles at a least-squares compromise silently.
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiJointCoupling {
  GENERATED_BODY()
  // Uncheck to ignore this entry without deleting it.
  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  bool bEnabled = true;
  // Name of the first joint, as it appears in the articulation. Must be a single-dof joint
  // (revolute or prismatic); welded and free joints have no dof to couple and are rejected.
  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  FName JointAName;
  // Name of the second joint. Same requirements as JointAName, and must differ from it.
  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  FName JointBName;
  // Coupling ratio in JointB = Ratio * JointA + Offset. Dimensionless when both joints are the
  // same kind; for a revolute/prismatic pair it is a moment arm [m/rad].
  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  double Ratio = 1.0;
  // Constant offset in JointB = Ratio * JointA + Offset. Carries the joints' own units, in Mochi's
  // SI convention rather than Unreal's: radians for revolute joints, metres for prismatic ones.
  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  double Offset = 0.0;
  // Penalty stiffness, in the joints' own units: [Nm/rad] for revolute joints, [N/m] for
  // prismatic. Higher holds the coupling more tightly but worsens the conditioning of the solve,
  // and can overpower a joint's limits, which are themselves only penalties. Keep it in the same
  // order as the solver's other weights.
  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  double Stiffness = 10.0;
};

//------------------------------------------------------------------------------
// EVENTS
//------------------------------------------------------------------------------

// clang-format off
DECLARE_DYNAMIC_MULTICAST_DELEGATE_FourParams(FOnMochiComponentBeginContact, USceneComponent*, MochiComponent, AActor*, OtherMochiActor, USceneComponent*, OtherMochiComponent, const TArray<FMochiContactPoint>&, Contacts);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(FOnMochiComponentEndContact, USceneComponent*, MochiComponent, AActor*, OtherMochiActor, USceneComponent*, OtherMochiComponent);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnMochiComponentContact, USceneComponent*, MochiComponent, const TArray<FMochiContactPoint>&, Contacts);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnMochiComponentContactProcessing, USceneComponent*, MochiComponent);

DECLARE_DYNAMIC_MULTICAST_DELEGATE_FiveParams(FOnMochiComponentBeginNestedContact, USceneComponent*, MochiComponent, FName, NestedName, AActor*, OtherMochiActor, USceneComponent*, OtherMochiComponent, const TArray<FMochiContactPoint>&, Contacts);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_FourParams(FOnMochiComponentEndNestedContact, USceneComponent*, MochiComponent, FName, NestedName, AActor*, OtherMochiActor, USceneComponent*, OtherMochiComponent);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(FOnMochiComponentNestedContact, USceneComponent*, MochiComponent, FName, NestedName, const TArray<FMochiContactPoint>&, Contacts);
// clang-format on

//------------------------------------------------------------------------------
// CONTACT HANDLING
//------------------------------------------------------------------------------

struct SUPERDEXPHYSICS_API FMochiContactEventHandler {
  void Update(
      USceneComponent* CallingMochiComponent,
      const FOnMochiComponentBeginContact& OnMochiComponentBeginContact,
      const FOnMochiComponentEndContact& OnMochiComponentEndContact,
      const FOnMochiComponentContact& OnMochiComponentContact,
      const TArray<FMochiContactPoint>& NewContactPoints,
      bool bFilterSelfContacts = true);
  TArray<FMochiContactPoint> ContactPoints;
  TSet<USceneComponent*> ContactingComponents;
  TSet<USceneComponent*> PreviousContactingComponents;
};

struct SUPERDEXPHYSICS_API FMochiNestedContactEventHandler {
  void Update(
      USceneComponent* CallingMochiComponent,
      const FOnMochiComponentBeginNestedContact& OnMochiComponentBeginNestedContact,
      const FOnMochiComponentEndNestedContact& OnMochiComponentEndNestedContact,
      const FOnMochiComponentNestedContact& OnMochiComponentNestedContact,
      const TArray<FMochiContactPoint>& NewContactPoints,
      bool bFilterSelfContacts = true);
  TMap<FName, TArray<FMochiContactPoint>> SortedContactPoints;
  TMap<FName, TArray<FMochiContactPoint>> ContactPoints;
  TMap<FName, TSet<USceneComponent*>> ContactingComponents;
  TMap<FName, TSet<USceneComponent*>> PreviousContactingComponents;
};
