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

#include "MochiGenTypes.h"
#include "MochiTypes.generated.h"

//------------------------------------------------------------------------------
// FORWARDS
//------------------------------------------------------------------------------

class UMochiMaterialParams;
class UMochiModel;

//------------------------------------------------------------------------------
// LAYERS
//------------------------------------------------------------------------------

/** Common names used for native contact layers. */
struct FMochiLayer {
  /** Contact-layer name for the left hand. */
  static constexpr const char* LeftHand = "LeftHand";
  /** Contact-layer name for the right hand. */
  static constexpr const char* RightHand = "RightHand";
  /** Default contact-layer name for generic objects. */
  static constexpr const char* GenericObject = "GenericObject";
};

//------------------------------------------------------------------------------
// SERIALIZATION VERSION
//------------------------------------------------------------------------------

/**
 * Serialization version for SuperDexPhysics assets. Saving stamps LatestVersion into
 * the package; an asset saved before an entry was added reads that entry back as -1.
 */
struct SUPERDEXPHYSICS_API FMochiObjectVersion {
  /** Milestones recorded by the plugin custom-version registration. */
  enum Type {
    BeforeCustomVersionWasAdded = 0,

    /** UMochiModel holds an FMochiModelData, with the SDF values in bulk data */
    TypedModelData,

    /** Native-shaped generated structs replace the legacy bounds and IK tolerance formats. */
    NativeGeneratedStructs,
    // -----<new versions can be added above this line>-----
    VersionPlusOne,
    LatestVersion = VersionPlusOne - 1,
  };

  /** Identifier of the plugin's custom-version registration. */
  static const FGuid GUID;

  /** Prevent construction of the custom-version registration type. */
  FMochiObjectVersion() = delete;
};

//------------------------------------------------------------------------------
// COORDINATE SPACES
//------------------------------------------------------------------------------

// clang-format off

/**
 * 48 FRU combinations, named by the axis Forward, Right, and Up each point along. This
 * exhaustive list prevents the user from selecting an invalid space and also provides a
 * mechanism to select common spaces by name. Values are mochi::CoordinateSpaceAxes, which
 * defines what each convention means; config and assets store the names, not the values.
 */
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

/** Axis convention and units describing a coordinate space. */
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiSpaceDefinition {
  GENERATED_BODY()
  /** The axes used for Forward, Right, and Up */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, DisplayName = "Axes (FRU)")
  EMochiSpaceAxes Axes = EMochiSpaceAxes::Unreal;

  // Note: This feature is intentionally read-only and locked to 1.0 at this time.
  // Mochi should always be simulated at meter scale.
  /** How many units are in a meter? */
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
  double UnitsPerMeter = 1.0;

  /** Initialize the default coordinate-space definition. */
  constexpr FMochiSpaceDefinition() = default;
  /**
   * Initialize an axis convention and units-per-meter value.
   *
   * @param InAxes Coordinate-axis convention.
   * @param InMeterScale Units-per-meter value.
   */
  constexpr FMochiSpaceDefinition(EMochiSpaceAxes InAxes, double InMeterScale)
      : Axes(InAxes), UnitsPerMeter(InMeterScale) {}
};

/**
 * Compare coordinate-axis conventions and units-per-meter values.
 *
 * @param Lhs First coordinate-space definition.
 * @param Rhs Second coordinate-space definition.
 * @return Whether both definitions have identical axes and units-per-meter values.
 */
constexpr bool operator==(const FMochiSpaceDefinition& Lhs, const FMochiSpaceDefinition& Rhs) {
  return Lhs.Axes == Rhs.Axes && Lhs.UnitsPerMeter == Rhs.UnitsPerMeter;
}
/**
 * Compare coordinate-space definitions for inequality.
 *
 * @param Lhs First coordinate-space definition.
 * @param Rhs Second coordinate-space definition.
 * @return Whether the axis convention or units-per-meter value differs.
 */
constexpr bool operator!=(const FMochiSpaceDefinition& Lhs, const FMochiSpaceDefinition& Rhs) {
  return !(Lhs == Rhs);
}

/** Legacy Mochi axis convention with meter scaling. */
static constexpr FMochiSpaceDefinition MochiSpace(EMochiSpaceAxes::Mochi, 1.0);
/** SuperDex axis convention with meter scaling. */
static constexpr FMochiSpaceDefinition SuperDexSpace(EMochiSpaceAxes::SuperDex, 1.0);
/** Unreal axis convention with centimeter scaling. */
static constexpr FMochiSpaceDefinition UnrealCentimetersSpace(EMochiSpaceAxes::Unreal, 100.0);
/** Unreal axis convention with meter scaling. */
static constexpr FMochiSpaceDefinition UnrealMetersSpace(EMochiSpaceAxes::Unreal, 1.0);

/**
 * Convert the Unreal coordinate-space definition to native axes and scaling.
 *
 * @param SpaceDef Coordinate-space definition used to build a basis.
 * @return Native axis convention and scale.
 */
FORCEINLINE constexpr mochi::CoordinateSpace ToMochiCoordinateSpace(
    const FMochiSpaceDefinition& SpaceDef) {
  return {static_cast<mochi::CoordinateSpaceAxes>(SpaceDef.Axes), SpaceDef.UnitsPerMeter};
}

//------------------------------------------------------------------------------
// ENUMS
//------------------------------------------------------------------------------

/** Implicit geometry supported by the shape component. */
UENUM(BlueprintType)
enum class EMochiImplicitShape : uint8 { Plane, Sphere };

/** World-aligned translation and rotation axes that may be frozen. */
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

/** Quadrature strategy used for reduced-order-model hyper-reduction. */
UENUM(BlueprintType)
enum class EMochiHyperReductionStrategy : uint8 {
  NoHyperReduction,
  StaticHyperReductionUsingRandomSampling
};

/** Direction of contact generation relative to the receiving component. */
UENUM(Blueprintable)
enum class EMochiContactDirection : uint8 {
  /**
   * 'this' object is contacting the 'other' object
   * (i.e the contact point in question lies on the surface of 'this' object)
   */
  ThisContactsOther,

  /**
   * the 'other' object is contacting 'this' object
   * (i.e the contact point in question lies on the surface of the 'other' object)
   */
  OtherContactsThis
};

/** Which end of a rod a constraint is attached to. */
UENUM(BlueprintType)
enum class EMochiRodAttachmentEnd : uint8 {
  First,
  Last,
};

//------------------------------------------------------------------------------
// STRUCTS
//------------------------------------------------------------------------------

/** Contact data expressed relative to the receiving Unreal component. */
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiContactPoint {
  GENERATED_BODY()

  // NOTE: this is the Actor from which the contact point was generated
  //       if and only if ContactDirection equals ThisContactsOther
  /** The Actor receiving the FMochiContactPoint data. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  AActor* MochiActor = nullptr;

  // NOTE: this is the Component from which the contact point was generated
  //       if and only if ContactDirection equals ThisContactsOther
  /** The Component receiving the FMochiContactPoint data. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  USceneComponent* MochiComponent = nullptr;

  /**
   * If the contact came from a internally nested Mochi actor belonging to an articulated actor,
   * this will be the name of that actor. For a skin, which has no nested actor of its own, it is
   * the name of the articulated actor the skin belongs to. Otherwise it will be None.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FName NestedName = NAME_None;

  // NOTE: this is the actor from which the contact point was generated
  //       if and only if ContactDirection equals OtherContactsThis
  /** The other Actor in contact. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  AActor* OtherMochiActor = nullptr;

  // NOTE: this is the Component from which the contact point was generated
  //       if and only if ContactDirection equals OtherContactsThis
  /** The other Component in contact. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  USceneComponent* OtherMochiComponent = nullptr;

  /** The distance between MochiComponent and OtherMochiComponent */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  double Distance = 0;

  /** The location of the contact on MochiComponent in world-space. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector Location = FVector::ZeroVector;

  /** The location of the contact on OtherMochiComponent in world-space. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector OtherLocation = FVector::ZeroVector;

  /** The point velocity of the contact on MochiComponent in world-space. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector Velocity = FVector::ZeroVector;

  /** The point velocity of the contact on OtherMochiComponent in world-space. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector OtherVelocity = FVector::ZeroVector;

  // NOTE: this corresponds to surface normals on the surface of the Component from which the
  //       contact point was generated, which is determined by the value of ContactDirection:
  //         ThisContactsOther -> MochiComponent
  //         OtherContactsThis -> OtherMochiComponent
  /** The contact normal pointing from OtherMochiComponent to MochiComponent. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector Normal = FVector::ZeroVector;

  /** The contact force applied to MochiComponent from OtherMochiComponent. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector Force = FVector::ZeroVector;

  // ADVANCED

  // NOTE: this index corresponds to the object from which the contact point was generated,
  //       which is determined by the value of ContactDirection:
  //         ThisContactsOther -> MochiComponent
  //         OtherContactsThis -> OtherMochiComponent
  /** The index of the Mochi surface triangle associated with this contact. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  int TriangleIndex = -1;

  /** The barycentric coords of the contact within its associated Mochi surface triangle. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector ParametricCoords = FVector::ZeroVector;

  /** The integration area of the contact in cm^2 (divide Force by Area to get pressure). */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  float Area = 0;

  // NOTE: this determines how TriangleIndex should be interpreted
  // NOTE: several members of this struct are swapped in the 'OtherContactsThis' case
  //       so that force vectors always point in a consistent direction
  //       (and such that summing over all contacts for an object
  //       yields the correct total net contact force for that object)
  /**
   * Which direction is the contact in? (this->other or other->this)
   * Equivalently, which object generated the contact point? (other or this)
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  EMochiContactDirection ContactDirection = EMochiContactDirection::ThisContactsOther;

  /**
   * The name of the internal Mochi actor originating contact, not to be confused with Unreal
   * actors.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FName InternalMochiActor;

  /**
   * The name of the other internal Mochi actor originating contact, not to be confused with Unreal
   * actors.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FName OtherInternalMochiActor;
};

/** Contact rule between two named native layers. */
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiContactMatrixEntry {
  GENERATED_BODY()
  /** Name of the layer of the "colliding" actor (the one checking for contact). */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FName LayerA;
  /**
   * Name of the layer of the "collider" actor (the one with a ColliderType (e.g. SDF) being
   * tested).
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FName LayerB;
  /** True to enable contact, false to disable it. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  bool bEnabled = true;
  /** If true, an additional entry is created with swapped layer arguments. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  bool bSymmetric = false;
};

/** Reduced-order-model source and element-selection settings. */
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiRomParams {
  GENERATED_BODY()

  FMochiRomParams();
  /** Specifies a ROM inside the rom group of a .mochi.h5 file. */
  UPROPERTY(EditAnywhere, Category = "Mochi")
  FString Source;
  /** Hyper-reduction quadrature strategy for this ROM. */
  UPROPERTY(EditAnywhere, Category = "Mochi")
  EMochiHyperReductionStrategy HyperReductionStrategy;
  /** Parameter that defines a sample mesh when subsampling the boundary elements. */
  UPROPERTY(
      EditAnywhere,
      meta =
          (EditCondition =
               "HyperReductionStrategy == EMochiHyperReductionStrategy::StaticHyperReductionUsingRandomSampling",
           EditConditionHides),
      Category = "Mochi")
  int StepSizeForBoundaryElementsSelection;
  /** Parameter that defines a sample mesh when subsampling the interior elements. */
  UPROPERTY(
      EditAnywhere,
      meta =
          (EditCondition =
               "HyperReductionStrategy == EMochiHyperReductionStrategy::StaticHyperReductionUsingRandomSampling",
           EditConditionHides),
      Category = "Mochi")
  int StepSizeForInteriorElementsSelection;

 public:
  /**
   * Convert to native Mochi type.
   *
   * @return Native parameter values converted from this object.
   */
  mochi::experimental::RomParams ToMochi() const;

  /** Convert from native Mochi type. */
  void FromMochi(const mochi::experimental::RomParams& Mochi);
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

/** Moment-of-inertia tensor about the center of mass, in kg*cm^2. */
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

  /** Ixx = I(1,1) [kg*cm^2] */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  double Ixx = 0;
  /** Ixy = I(1,2) [kg*cm^2] */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  double Ixy = 0;
  /** Ixz = I(1,3) [kg*cm^2] */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  double Ixz = 0;
  /** Iyy = I(2,2) [kg*cm^2] */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  double Iyy = 0;
  /** Iyz = I(2,3) [kg*cm^2] */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  double Iyz = 0;
  /** Izz = I(3,3) [kg*cm^2] */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  double Izz = 0;

 public:
  /**
   * Returns true if all elements are zero.
   *
   * @return Whether every inertia entry is zero.
   */
  bool IsZero() const {
    return Ixx == 0 && Ixy == 0 && Ixz == 0 && Iyy == 0 && Iyz == 0 && Izz == 0;
  }
};

/**
 * Mirrors mochi::ActiveShapeTargetingArapMaterialParams::shapeTargetTensor, which is a
 * Real6 that UHT cannot express. The desired local shape is the identity plus the
 * symmetric matrix [[XX, XY, XZ], [XY, YY, YZ], [XZ, YZ, ZZ]]; the material resists
 * deformation away from it while still allowing rigid rotation. All zero means no
 * targeting, which is why that is the default.
 */
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiShapeTargetTensor {
  GENERATED_BODY()
  FMochiShapeTargetTensor() {
    FromMochi(mochi::ActiveShapeTargetingArapMaterialParams{}.shapeTargetTensor);
  }

  /** XX entry of the symmetric shape-target offset. */
  UPROPERTY(EditAnywhere, Category = "Mochi")
  float XX;
  /** XY entry of the symmetric shape-target offset. */
  UPROPERTY(EditAnywhere, Category = "Mochi")
  float XY;
  /** XZ entry of the symmetric shape-target offset. */
  UPROPERTY(EditAnywhere, Category = "Mochi")
  float XZ;
  /** YY entry of the symmetric shape-target offset. */
  UPROPERTY(EditAnywhere, Category = "Mochi")
  float YY;
  /** YZ entry of the symmetric shape-target offset. */
  UPROPERTY(EditAnywhere, Category = "Mochi")
  float YZ;
  /** ZZ entry of the symmetric shape-target offset. */
  UPROPERTY(EditAnywhere, Category = "Mochi")
  float ZZ;

  /**
   * Convert the symmetric shape-target entries to native storage order.
   *
   * @return Native parameter values converted from this object.
   */
  mochi::Real6 ToMochi() const;
  /**
   * Load the symmetric shape-target entries from native storage order.
   *
   * @param Mochi Native values supplied to the conversion.
   */
  void FromMochi(const mochi::Real6& Mochi);
};

/** Material values assigned to selected model elements. */
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiMaterialModifier {
  GENERATED_BODY()

  /** Initialize material-modifier values from native material defaults. */
  FMochiMaterialModifier() {
    constexpr mochi::real PaToMPa = 1e-6;
    YoungsModulus = mochi::kDefaultYoungsModulus * PaToMPa;
    PoissonRatio = mochi::kDefaultPoissonRatio;
  }

  /** Young's modulus assigned by the material modifier. */
  UPROPERTY(
      EditAnywhere,
      DisplayName = "Young's Modulus",
      meta = (ClampMin = 0, UIMin = 0),
      meta = (ForceUnit = "MPa"),
      Category = "Mochi")
  float YoungsModulus;

  /** Poisson ratio assigned by the material modifier. */
  UPROPERTY(EditAnywhere, Category = "Mochi")
  float PoissonRatio;
};

/** Deformable model attached to a named rigid link of an articulation. */
USTRUCT(BlueprintType)
struct FMochiArticulationSoftLink {
  GENERATED_BODY()
  /**
   * The Model to use for the soft link. The Model H5 must have constrained
   * nodes, which currently cannot be authored in Unreal.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  UMochiModel* Model = nullptr;
  /** Material of the soft link. Uses Mochi's default material when null. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  UMochiMaterialParams* MaterialParams = nullptr;

  /** Scale applied to the Mochi Model when its shape is loaded. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector ModelScale = FVector::OneVector;
  /** Rotation applied to the Mochi Model when its shape is loaded. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector ModelOriginRPY = FVector::ZeroVector;
  /** Translation applied to the Mochi Model when its shape is loaded. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector ModelOriginXYZ = FVector::ZeroVector;
  /**
   * Name of the soft link. This should be uniquely different from the link names
   * used in UMochiArticulation::Links
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FName Name = NAME_None;
  /**
   * Name of the rigid link that will be the attach parent.
   * This must be a link named in UMochiArticulation::Links
   */
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta = (GetOptions = "GetLinkNames_EditorOnly"),
      Category = "Mochi")
  FName ParentName = NAME_None;
};

/** Distance-constraint settings between two articulation links. */
USTRUCT(BlueprintType)
struct FMochiArticulationConstraint {
  GENERATED_USTRUCT_BODY()
  /** Name of the first constrained link. */
  UPROPERTY(
      EditAnywhere,
      DisplayName = "Link A Name",
      BlueprintReadWrite,
      meta = (GetOptions = "GetLinkNames_EditorOnly"),
      Category = "Mochi")
  FName LinkAName = NAME_None;
  /** Constraint point in the first link's local frame. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector LinkAPosition = FVector::ZeroVector;
  /** Name of the second constrained link. */
  UPROPERTY(
      EditAnywhere,
      DisplayName = "Link B Name",
      BlueprintReadWrite,
      meta = (GetOptions = "GetLinkNames_EditorOnly"),
      Category = "Mochi")
  FName LinkBName = NAME_None;
  /** Constraint point in the second link's local frame. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector LinkBPosition = FVector::ZeroVector;
  /** Stiffness of constraint [N/m] */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  float Stiffness = mochi::RigidSphericalJointConstraintParams{}.stiffness;
  /** Damping on the constraint limit [Ns/m] */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  float Damping = mochi::RigidSphericalJointConstraintParams{}.damping;
  /** Displacement at which the constraint stiffness saturates, -1 for no saturation */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  float Saturation = mochi::RigidSphericalJointConstraintParams{}.saturation;
};

/**
 * Enables or disables contact between two links of an articulation, after the articulation is
 * created. Adjacent links have contact disabled by default.
 */
USTRUCT(BlueprintType)
struct FMochiArticulationContactOverride {
  GENERATED_USTRUCT_BODY()
  /** Name of the first link whose contact is overridden. */
  UPROPERTY(
      EditAnywhere,
      DisplayName = "Link A Name",
      BlueprintReadWrite,
      meta = (GetOptions = "GetLinkNames_EditorOnly"),
      Category = "Mochi")
  FName LinkAName = NAME_None;
  /** Name of the second link whose contact is overridden. */
  UPROPERTY(
      EditAnywhere,
      DisplayName = "Link B Name",
      BlueprintReadWrite,
      meta = (GetOptions = "GetLinkNames_EditorOnly"),
      Category = "Mochi")
  FName LinkBName = NAME_None;
  /** Whether contact between the links is enabled. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  bool bEnable = false;
};

/**
 * Couples two single-dof joints of one articulation so that
 *   JointB = Ratio * JointA + Offset.
 * Wraps a Mochi linear transmission driven by a displacement-control actuator, so the coupling is
 * a spring penalty rather than a hard constraint: it is satisfied to roughly (solver absolute
 * tolerance / Stiffness). The penalty is symmetric, so both joints move to satisfy it; A and B
 * only name the direction of Ratio. Useful for resolving a redundant dof, e.g. tying a finger's
 * distal joint to its proximal one.
 * Entries are independent and their penalties simply add, so a joint may appear in more than one
 * coupling and chains such as A->B, B->C are allowed. Nothing checks that a set of couplings is
 * mutually consistent; an over-constrained set settles at a least-squares compromise silently.
 */
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiJointCoupling {
  GENERATED_BODY()
  /** Uncheck to ignore this entry without deleting it. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  bool bEnabled = true;
  /**
   * Name of the first joint, as it appears in the articulation. Must be a single-dof joint
   * (revolute or prismatic); welded and free joints have no dof to couple and are rejected.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  FName JointAName;
  /** Name of the second joint. Same requirements as JointAName, and must differ from it. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  FName JointBName;
  /**
   * Coupling ratio in JointB = Ratio * JointA + Offset. Dimensionless when both joints are the
   * same kind; for a revolute/prismatic pair it is a moment arm [m/rad].
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  double Ratio = 1.0;
  /**
   * Constant offset in JointB = Ratio * JointA + Offset. Carries the joints' own units, in Mochi's
   * SI convention rather than Unreal's: radians for revolute joints, metres for prismatic ones.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  double Offset = 0.0;
  /**
   * Penalty stiffness, in the joints' own units: [Nm/rad] for revolute joints, [N/m] for
   * prismatic. Higher holds the coupling more tightly but worsens the conditioning of the solve,
   * and can overpower a joint's limits, which are themselves only penalties. Keep it in the same
   * order as the solver's other weights.
   */
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

/** Maintains component contact history for begin, end and current-contact events. */
struct SUPERDEXPHYSICS_API FMochiContactEventHandler {
  /**
   * Update contact history and emit component contact events.
   *
   * @param CallingMochiComponent Component receiving the contact events.
   * @param OnMochiComponentBeginContact Delegate invoked for newly contacting components.
   * @param OnMochiComponentEndContact Delegate invoked for components no longer in contact.
   * @param OnMochiComponentContact Delegate invoked for current component contacts.
   * @param NewContactPoints New contact data used to update contact history.
   * @param bFilterSelfContacts Whether self-contacts are removed from event generation.
   */
  void Update(
      USceneComponent* CallingMochiComponent,
      const FOnMochiComponentBeginContact& OnMochiComponentBeginContact,
      const FOnMochiComponentEndContact& OnMochiComponentEndContact,
      const FOnMochiComponentContact& OnMochiComponentContact,
      const TArray<FMochiContactPoint>& NewContactPoints,
      bool bFilterSelfContacts = true);
  /** Contact points retained for component event generation. */
  TArray<FMochiContactPoint> ContactPoints;
  /** Components contacting the receiver in the current update. */
  TSet<USceneComponent*> ContactingComponents;
  /** Components contacting the receiver in the previous update. */
  TSet<USceneComponent*> PreviousContactingComponents;
};

/** Maintains nested actor contact history for articulation events. */
struct SUPERDEXPHYSICS_API FMochiNestedContactEventHandler {
  /**
   * Update nested contact history and emit nested component contact events.
   *
   * @param CallingMochiComponent Component receiving the contact events.
   * @param OnMochiComponentBeginNestedContact Delegate invoked for newly contacting nested actors.
   * @param OnMochiComponentEndNestedContact Delegate invoked for nested actors no longer in
   * contact.
   * @param OnMochiComponentNestedContact Delegate invoked for current nested contacts.
   * @param NewContactPoints New contact data used to update contact history.
   * @param bFilterSelfContacts Whether self-contacts are removed from event generation.
   */
  void Update(
      USceneComponent* CallingMochiComponent,
      const FOnMochiComponentBeginNestedContact& OnMochiComponentBeginNestedContact,
      const FOnMochiComponentEndNestedContact& OnMochiComponentEndNestedContact,
      const FOnMochiComponentNestedContact& OnMochiComponentNestedContact,
      const TArray<FMochiContactPoint>& NewContactPoints,
      bool bFilterSelfContacts = true);
  /** Contact points grouped by nested actor name. */
  TMap<FName, TArray<FMochiContactPoint>> SortedContactPoints;
  /** Current contact points grouped by nested actor name. */
  TMap<FName, TArray<FMochiContactPoint>> ContactPoints;
  /** Currently contacting components grouped by nested actor name. */
  TMap<FName, TSet<USceneComponent*>> ContactingComponents;
  /** Previously contacting components grouped by nested actor name. */
  TMap<FName, TSet<USceneComponent*>> PreviousContactingComponents;
};
