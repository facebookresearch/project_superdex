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

#include "CoreMinimal.h"
#include "Engine/DataTable.h"
#include "MochiConversions.h"
#include "MochiTypes.h"

#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include "MochiArticulationTypes.generated.h"

class UMochiContactParams;
class UMochiModel;
class UStaticMesh;

// The structs below mirror the Mochi articulated actor types (mochi::Articulated*Params and
// mochi::prefab::Articulated*Prefab) member for member. Spatial quantities (transforms, axes,
// centers of mass, inertia tensors, shape offsets) are stored in Unreal space (LH, Z up, cm) and
// converted to the simulation space by ToMochi. Joint-space quantities (limits, friction, joint
// inertia, positions, velocities) keep Mochi's units: [m] for translational DOFs and [rad] for
// rotational DOFs.

// Number of DOFs a joint of the given type contributes to a Mochi articulated actor.
SUPERDEXPHYSICS_API int32 GetArticulatedJointNumDofs(EMochiArticulatedJointType Type);

// Mirrors mochi::ArticulatedJointFrictionParams.
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiArticulatedJointFriction {
  GENERATED_BODY()
  // Viscous friction coefficient [N*s/m or N*m*s/rad].
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Friction")
  double Viscous = 0;
  // Coulomb friction coefficient [N or N*m].
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Friction")
  double Coulomb = 0;
  // Velocity threshold for dry friction smoothing [m/s or rad/s].
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Friction")
  double FalloffVel = 1e-3;
  // [Experimental] Extra stiction force/torque [N or N*m].
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Friction", AdvancedDisplay)
  double StictionExtra = 0;
  // [Experimental] Stribeck velocity [m/s or rad/s].
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Friction", AdvancedDisplay)
  double StribeckVel = 0;

  mochi::ArticulatedJointFrictionParams ToMochi() const;
  void FromMochi(const mochi::ArticulatedJointFrictionParams& Mochi);
};

// Mirrors mochi::ArticulatedJointParams. Joint i connects link i to its parent link.
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiArticulatedJoint {
  GENERATED_BODY()
  // Joint name. Must be unique among the joints of the articulation.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Joint")
  FName Name = NAME_None;
  // Joint type. Free joints are only supported on the root link.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Joint",
      meta = (ValidEnumValues = "Free, Prismatic, Revolute, Spherical, Hard"))
  EMochiArticulatedJointType Type = EMochiArticulatedJointType::Hard;
  // Rotation and translation of the joint with respect to the parent link [cm].
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Joint")
  FTransform ParentLinkFromJoint = FTransform::Identity;
  // Axis of motion in the joint frame. Only used by Prismatic and Revolute joints. A positive
  // revolute joint position rotates the child link about this axis in Unreal's convention.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Joint",
      meta =
          (EditCondition =
               "Type==EMochiArticulatedJointType::Prismatic || Type==EMochiArticulatedJointType::Revolute",
           EditConditionHides))
  FVector Axis = FVector::ZAxisVector;
  // Joint friction. Ignored for Free and Hard joints.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Dynamics")
  FMochiArticulatedJointFriction Friction;
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Dynamics",
      meta = (InlineEditConditionToggle))
  bool bOverrideInertia = false;
  // Joint inertia [kg or kg*m^2]. Ignored for Free and Hard joints.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Dynamics",
      meta = (EditCondition = "bOverrideInertia"))
  double Inertia = 0;
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Limits",
      meta = (InlineEditConditionToggle))
  bool bHasMinLimit = false;
  // Minimum joint limits [m or rad]. For 1-DOF joints this is the scalar limit times Axis; for
  // Spherical joints each component limits the rotation about the corresponding joint frame axis.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Limits",
      meta = (EditCondition = "bHasMinLimit"))
  FVector MinLimit = FVector::ZeroVector;
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Limits",
      meta = (InlineEditConditionToggle))
  bool bHasMaxLimit = false;
  // Maximum joint limits [m or rad]. See MinLimit.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Limits",
      meta = (EditCondition = "bHasMaxLimit"))
  FVector MaxLimit = FVector::ZeroVector;
  // Stiffness of the joint limit constraints [N/m or N*m/rad].
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Limits")
  double LimitStiffness = 100;
  // Damping of the joint limit constraints [N*s/m or N*m*s/rad].
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Limits")
  double LimitDamping = 0;

 public:
  // Number of DOFs this joint contributes to a Mochi articulated actor.
  int32 GetNumDofs() const;
  // Lower and upper limits of one of the joint's DOFs [m or rad], or -/+infinity when unlimited.
  // For 1-DOF joints, DofIndex must be 0 and the limit is projected onto Axis.
  void GetDofLimits(int32 DofIndex, double& OutLower, double& OutUpper) const;

  mochi::ArticulatedJointParams ToMochi(const FUnrealToMochiConverter& U2M) const;
  void FromMochi(const mochi::ArticulatedJointParams& Mochi, const FMochiToUnrealConverter& M2U);
};

// Mirrors mochi::prefab::ArticulatedLinkPrefab (mochi::ArticulatedLinkParams plus its shape and
// render model references and offsets).
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiArticulatedLink {
  GENERATED_BODY()
  // Link name. Must be unique among the links of the articulation.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Link")
  FName Name = NAME_None;
  // Index of the parent link, or -1 for the root link. Must be less than this link's index.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Link")
  int32 ParentLink = INDEX_NONE;
  // Rotation and translation of the link with respect to its parent joint [cm].
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Link")
  FTransform ParentJointFromLink = FTransform::Identity;
  // Collision shape of the link. Links without a shape are massless dummies.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Shape")
  TObjectPtr<UMochiModel> Shape = nullptr;
  // Scale baked into the shape.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Shape")
  FVector ShapeScale = FVector::OneVector;
  // Rotation of the shape in the link frame, applied after ShapeScale.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Shape")
  FQuat ShapeRotation = FQuat::Identity;
  // Translation of the shape in the link frame [cm], applied after ShapeRotation.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Shape")
  FVector ShapeTranslation = FVector::ZeroVector;
  // Render model of the link.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Render Model")
  TObjectPtr<UStaticMesh> RenderModel = nullptr;
  // Scale of the render model.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Render Model")
  FVector RenderModelScale = FVector::OneVector;
  // Rotation of the render model in the link frame, applied after RenderModelScale.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Render Model")
  FQuat RenderModelRotation = FQuat::Identity;
  // Translation of the render model in the link frame [cm], applied after RenderModelRotation.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Render Model")
  FVector RenderModelTranslation = FVector::ZeroVector;
  // Material slots of the render model that are not merged with others when generating the
  // skeletal mesh (e.g. to keep fingertip materials separate).
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Render Model", AdvancedDisplay)
  TArray<int32> UniqueMaterialSlots;
  // Contact layer of the link. Defaults to the layer of the articulation component when None.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Contact")
  FName Layer = NAME_None;
  // Collider type of the link.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Contact",
      meta = (ValidEnumValues = "None, Auto, Sphere, Box, Mesh, Plane, Sdf"))
  EMochiColliderType ColliderType = EMochiColliderType::Auto;
  // Contact params of the link. Defaults to those of the articulation component when null.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Contact")
  TObjectPtr<UMochiContactParams> ContactParams = nullptr;
  // Boundary element type used for the link's surface discretization.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Contact",
      meta = (ValidEnumValues = "P1Q1, P1Q3, P1Q6, Default"))
  EMochiActorBoundaryElementType BoundaryElementType = EMochiActorBoundaryElementType::Default;
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Contact",
      meta = (InlineEditConditionToggle))
  bool bEnableBoundarySubsampling = false;
  // Boundary subsampling parameters.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Contact",
      meta = (EditCondition = "bEnableBoundarySubsampling"))
  FMochiBoundarySubsamplingParams BoundarySubsampling;
  // Whether the link is affected by gravity.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Inertial")
  bool bHasGravity = true;
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Inertial",
      meta = (InlineEditConditionToggle))
  bool bOverrideDensity = false;
  // Density of the link [kg/m^3]. Mutually exclusive with Mass.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Inertial",
      meta = (EditCondition = "bOverrideDensity"))
  double Density = 0;
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Inertial",
      meta = (InlineEditConditionToggle))
  bool bOverrideMass = false;
  // Mass of the link [kg]. Mutually exclusive with Density.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Inertial",
      meta = (EditCondition = "bOverrideMass", Units = "Kilograms"))
  double Mass = 0;
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Inertial",
      meta = (InlineEditConditionToggle))
  bool bOverrideCenterOfMass = false;
  // Center of mass in the link frame [cm].
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Inertial",
      meta = (EditCondition = "bOverrideCenterOfMass"))
  FVector CenterOfMass = FVector::ZeroVector;
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Inertial",
      meta = (InlineEditConditionToggle))
  bool bOverrideMomentOfInertia = false;
  // Moment of inertia about the center of mass, aligned with the link frame [kg*cm^2].
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Inertial",
      meta = (EditCondition = "bOverrideMomentOfInertia"))
  FMochiMomentOfInertia MomentOfInertia;

 public:
  // Transform of the shape in the link frame (ShapeRotation and ShapeTranslation).
  FTransform GetShapeTransform() const;
  // Transform of the render model in the link frame, including RenderModelScale.
  FTransform GetRenderModelTransform() const;

  // Build the Mochi link params. Layer and contact params fall back to the given defaults.
  mochi::ArticulatedLinkParams ToMochi(
      const FUnrealToMochiConverter& U2M,
      mochi::ShapeHandle InShape,
      FName DefaultLayer,
      const mochi::ContactParams& DefaultContact,
      bool bInHasGravity) const;
  // Copy everything except the Shape, RenderModel and ContactParams asset references, which the
  // caller is expected to resolve from the prefab's shapeFile, renderModelFile and contact.
  void FromMochi(
      const mochi::prefab::ArticulatedLinkPrefab& Mochi,
      const FMochiToUnrealConverter& M2U);
};

// Mirrors mochi::prefab::ArticulatedSkinPrefab, minus its render model which Unreal does not use.
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiArticulatedSkin {
  GENERATED_BODY()
  // Collision shape of the skin, expressed in the articulation root frame.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Skin")
  TObjectPtr<UMochiModel> Shape = nullptr;
  // Contact layer of the skin. Defaults to the layer of the articulation component when None.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Skin")
  FName Layer = NAME_None;
  // Contact params of the skin. Defaults to those of the articulation component when null.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Skin")
  TObjectPtr<UMochiContactParams> ContactParams = nullptr;
  // Boundary element type used for the skin's surface discretization.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Skin",
      meta = (ValidEnumValues = "P1Q1, P1Q3, P1Q6, Default"))
  EMochiActorBoundaryElementType BoundaryElementType = EMochiActorBoundaryElementType::Default;
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Skin", meta = (InlineEditConditionToggle))
  bool bEnableBoundarySubsampling = false;
  // Boundary subsampling parameters.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Skin",
      meta = (EditCondition = "bEnableBoundarySubsampling"))
  FMochiBoundarySubsamplingParams BoundarySubsampling;
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Skin", meta = (InlineEditConditionToggle))
  bool bOverrideNonCollidingLinks = false;
  // Links the skin does not collide with.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Skin",
      meta = (EditCondition = "bOverrideNonCollidingLinks"))
  TArray<FName> NonCollidingLinks;

 public:
  mochi::ArticulatedSkinParams ToMochi(
      mochi::ShapeHandle InShape,
      FName DefaultLayer,
      const mochi::ContactParams& DefaultContact) const;
  // Copy everything except the Shape and ContactParams asset references.
  void FromMochi(const mochi::prefab::ArticulatedSkinPrefab& Mochi);
};

// Mirrors mochi::ArticulatedCycleJointParams.
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiArticulatedCycleJoint {
  GENERATED_BODY()
  // Index of the parent link.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cycle")
  int32 ParentLink = INDEX_NONE;
  // Index of the child link.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cycle")
  int32 ChildLink = INDEX_NONE;
  // Pose of the cycle joint in the child link's frame [cm]. The translation is the pivot of the
  // spherical constraint that closes the cycle; the rotation is currently unused.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cycle")
  FTransform JointFromChildLink = FTransform::Identity;
  // Stiffness of the cycle-closing constraint [N/m].
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cycle")
  double Stiffness = 50000;

  mochi::ArticulatedCycleJointParams ToMochi(const FUnrealToMochiConverter& U2M) const;
  void
  FromMochi(const mochi::ArticulatedCycleJointParams& Mochi, const FMochiToUnrealConverter& M2U);
};

// Per-joint control gains, typically authored in a UDataTable keyed by joint name.
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiJointControlGains : public FTableRowBase {
  GENERATED_USTRUCT_BODY()
  // Joint name should match that specified in the articulation asset
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MochiJointControlGains")
  FName JointName = NAME_None;
  // Control gain on error, stiffness value in N/m (P) or Nm/rad (R)
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MochiJointControlGains")
  float Stiffness = 0.0f;
  // Control damping, in Ns/m (P) or Nms/rad (R)
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MochiJointControlGains")
  float Damping = 0.0f;
  // Maximum force in N (P) or torque in Nm (R) effort the joint can exert, -1 for "unspecified"
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MochiJointControlGains")
  float LimitEffort = -1.0f;
  // Maximum velocity in m/s (P) or rad/s (R) the joint can safely move at, -1 for "unspecified"
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MochiJointControlGains")
  float LimitVelocity = -1.0f;

 public:
  // Fill Table with one row for each joint with DOFs.
  static void PopulateTable(
      UDataTable* Table,
      const TArray<FMochiArticulatedJoint>& Joints,
      float Stiffness = 0.0f,
      float Damping = 0.0f);
};

// Convert per-DOF values (positions, velocities or efforts) between the Unreal and simulation space
// conventions. Dofs is laid out as Mochi lays out the DOFs of Types (one entry per type, in link
// order), optionally without the 6 DOFs of a Free root joint. Only Free and Spherical joints are
// affected: their DOFs are vectors expressed in the joint frame, while 1-DOF joint values are the
// same in both conventions.
SUPERDEXPHYSICS_API void ConvertArticulatedDofsToMochi(
    TConstArrayView<EMochiArticulatedJointType> Types,
    bool bIncludesRootFreeDofs,
    TArrayView<double> Dofs,
    const FUnrealToMochiConverter& U2M);
SUPERDEXPHYSICS_API void ConvertArticulatedDofsToUnreal(
    TConstArrayView<EMochiArticulatedJointType> Types,
    bool bIncludesRootFreeDofs,
    TArrayView<double> Dofs,
    const FMochiToUnrealConverter& M2U);
