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

#include "MochiArticulationBase.h"
#include "MochiArticulationTypes.h"
#include "MochiAssets.h"
#include "MochiModel.h"

#include "MochiArticulation.generated.h"

class FMochiEditorModule;
class FMochiArticulationDetailsCustomization;
class FMochiArticulationEditorToolkit;
class FMochiArticulationAssetUtils;

// An articulated body described with the same data model as a Mochi articulated actor
// (mochi::ArticulatedActorParams): one joint per link, where Joints[i] connects Links[i] to
// Links[Links[i].ParentLink]. The root joint (Joints[0]) attaches the root link to the frame of
// the component that spawns the articulation.
//
// Joint positions (DefaultJointPositions, FK inputs, etc.) contain one value per DOF of every
// joint except a Free root joint, whose 6 DOFs belong to the base: 1 for Revolute and Prismatic
// joints, and 3 (a rotation vector in the joint frame) for Spherical joints.
UCLASS(
    BlueprintType,
    HideCategories = Object,
    DisplayName = "Mochi Articulation",
    PrioritizeCategories = ("Articulation", "Assets", "Internal"),
    AutoCollapseCategories = ("Internal"))
class SUPERDEXPHYSICS_API UMochiArticulation : public UMochiArticulationBase {
  GENERATED_BODY()
 protected:
  // The name of the articulation described.
  UPROPERTY(EditAnywhere, Category = "Articulation", BlueprintReadOnly, AssetRegistrySearchable)
  FString ArticulationName;
  // Optional joint positions that define a default "home" pose of the articulation [rad or m]
  UPROPERTY(
      EditAnywhere,
      Category = "Articulation",
      BlueprintReadOnly,
      AssetRegistrySearchable,
      EditFixedSize,
      meta = (EditFixedOrder))
  TArray<double> DefaultJointPositions;
  // The joints of the articulation, one per link.
  UPROPERTY(
      EditAnywhere,
      Category = "Articulation",
      BlueprintReadOnly,
      meta = (TitleProperty = "Name"))
  TArray<FMochiArticulatedJoint> Joints;
  // The links of the articulation. Parents must precede their children.
  UPROPERTY(
      EditAnywhere,
      Category = "Articulation",
      BlueprintReadOnly,
      meta = (TitleProperty = "Name"))
  TArray<FMochiArticulatedLink> Links;
  // Cycle-closing joints for closed-loop topologies.
  UPROPERTY(EditAnywhere, Category = "Articulation", BlueprintReadOnly)
  TArray<FMochiArticulatedCycleJoint> Cycles;
  UPROPERTY(
      EditAnywhere,
      Category = "Articulation",
      BlueprintReadOnly,
      meta = (InlineEditConditionToggle))
  bool bHasSkin = false;
  // Optional deformable collision skin wrapped over the links.
  UPROPERTY(
      EditAnywhere,
      Category = "Articulation",
      BlueprintReadOnly,
      meta = (EditCondition = "bHasSkin"))
  FMochiArticulatedSkin Skin;
  // Link pairs whose contact is enabled or disabled, applied in order after the links are created.
  // Adjacent links have contact disabled by default.
  UPROPERTY(EditAnywhere, Category = "Articulation", BlueprintReadOnly, AssetRegistrySearchable)
  TArray<FMochiArticulationContactOverride> ContactOverrides;
  // Additional constraints to create between rigid links.
  UPROPERTY(EditAnywhere, Category = "Articulation", BlueprintReadOnly, AssetRegistrySearchable)
  TArray<FMochiArticulationConstraint> AdditionalConstraints;
  // Soft links to attach to rigid links
  UPROPERTY(EditAnywhere, Category = "Articulation", BlueprintReadOnly, AssetRegistrySearchable)
  TArray<FMochiArticulationSoftLink> SoftLinks;

  // The SkeletalMesh asset associated with the articulation. Can be regenerated with Generate
  // Skeletal Mesh.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Assets", AssetRegistrySearchable)
  TObjectPtr<USkeletalMesh> SkeletalMesh = nullptr;

  // User added notes/README about the articulation.
  UPROPERTY(EditAnywhere, Category = "Internal", meta = (MultiLine = true))
  FString Notes;

 public:
  // Check that the articulation is ready to be used (assets generated and data consistent).
  // This is a mildly expensive operation; call only once in BeginPlay or similar.
  bool IsReady(bool bLogWarnings = true, TArray<FString>* OutWarnings = nullptr) const;
  // Replace the description of the articulation. Rebuilds derived data and resizes
  // DefaultJointPositions; the SkeletalMesh must be regenerated afterwards.
  void SetDefinition(
      const FString& InArticulationName,
      TArray<FMochiArticulatedJoint> InJoints,
      TArray<FMochiArticulatedLink> InLinks,
      TArray<FMochiArticulatedCycleJoint> InCycles = {},
      const FMochiArticulatedSkin* InSkin = nullptr);
  // Get the name of the articulation.
  virtual const FString& GetArticulationName() const override;
  // Get the list of links.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  const TArray<FMochiArticulatedLink>& GetLinks() const;
  // Get the list of joints, one per link.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  const TArray<FMochiArticulatedJoint>& GetJoints() const;
  // Get the list of cycle-closing joints.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  const TArray<FMochiArticulatedCycleJoint>& GetCycles() const;
  // Returns true if the articulation has a collision skin.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  bool HasSkin() const;
  // Get the collision skin. Only meaningful if HasSkin().
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  const FMochiArticulatedSkin& GetSkin() const;
  // Returns true if the articulation is configured with soft link attachments.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  bool HasSoftLinks() const;
  // Get the list of *soft* links.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  const TArray<FMochiArticulationSoftLink>& GetSoftLinks() const;
  // Get the link contact overrides.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  const TArray<FMochiArticulationContactOverride>& GetContactOverrides() const;
  // Replace the link contact overrides.
  void SetContactOverrides(TArray<FMochiArticulationContactOverride> InContactOverrides);
  // Get the list of additional constraints
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  const TArray<FMochiArticulationConstraint>& GetAdditionalConstraints() const;
  // Get the default or home pose joint positions.
  virtual const TArray<double>& GetDefaultJointPositions() const override;
  // Returns true if the root joint does not let the articulation move freely, i.e. it is not a
  // Free joint. This is the default for UMochiArticulationComponent::bFixedBase.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  bool DefaultFixedBase() const;
  // The root joint type a component spawns the articulation with. bFixedBase overrides a Free or
  // Hard root joint (Hard if true, Free otherwise); other root joint types are kept as is.
  EMochiArticulatedJointType GetEffectiveRootJointType(bool bFixedBase) const;
  // Get a link index by name. This indexes into the array returned by GetLinks().
  // Return -1 if name does not refer to a valid link.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  int GetLinkIndex(FName LinkName) const;
  // Get a link by name. Returns true if found, false otherwise.
  UFUNCTION(
      BlueprintCallable,
      meta = (ExpandBoolAsExecs = "ReturnValue"),
      Category = "Mochi|Articulation")
  bool GetLink(FName LinkName, FMochiArticulatedLink& OutLink) const;
  // Get a joint index by name. This indexes into the array returned by GetJoints().
  // Return -1 if name does not refer to a valid joint.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  int GetJointIndex(FName JointName) const;
  // Get a joint by name. Returns true if found, false otherwise.
  UFUNCTION(
      BlueprintCallable,
      meta = (ExpandBoolAsExecs = "ReturnValue"),
      Category = "Mochi|Articulation")
  bool GetJoint(FName JointName, FMochiArticulatedJoint& OutJoint) const;
  // Get the joint connecting a link to its parent. Returns false if the index is out of bounds.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  bool GetLinkJoint(int LinkIndex, FMochiArticulatedJoint& OutJoint) const;
  // Get the indices of a link's children.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  TArray<int> GetChildLinkIndices(int LinkIndex) const;
  // Get all the leaf link indices, i.e. those without child links.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  TArray<int> GetLeafLinkIndices() const;
  // Get either the last link or first forking link. Useful for getting the
  // "end-effector" of articulation arms with hand attachments.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  FMochiArticulatedLink GetLastOrFirstForkingLink() const;
  // Get the number of DOFs, excluding the 6 DOFs of a Free root joint.
  virtual int GetNumDofs() const override;
  // Get the joint index of each DOF. Spherical joints appear three times.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  const TArray<int>& GetDofIndices() const;
  // Get the index of the first DOF of a joint. Returns -1 if the joint has no DOFs.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  int GetDofIndex(FName JointName) const;
  // Get the joint for a given *DOF* index; not to be confused with joint index.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  bool GetDof(int DofIndex, FMochiArticulatedJoint& OutJoint) const;
  // Get the lower and upper limits of a DOF [rad or m]. Unlimited DOFs return -/+infinity.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  bool GetDofLimits(int DofIndex, double& OutLower, double& OutUpper) const;
  // Get the transform of a link relative to its parent link when all joint positions are zero.
  FTransform GetRestLinkToParent(int LinkIndex) const;
  // Get the transform of a link relative to the articulation root frame when all joint positions
  // are zero.
  FTransform GetRestLinkToRoot(int LinkIndex) const;
  // Get the ordered array of link names.
  virtual TArray<FName> GetLinkNames() const override;
  // Get the ordered array of joint names
  virtual TArray<FName> GetJointNames() const override;
  // Get the SkeletalMesh to use with this articulation.
  virtual USkeletalMesh* GetSkeletalMesh() const override;
  // Compute link transforms for a given joint configuration. AngleScale and LengthScale
  // should convert your JointPositions units to radians/centimeters. The default values
  // assume JointPositions is defined in radians/meters. FinalLinkIndex can be used to
  // minimize unnecessary computations; a value of -1 will compute all link transforms.
  // LinkToRoot transforms are relative to the articulation root frame (the frame the root
  // joint attaches to), not the root link.
  virtual bool ComputeLinkTransformsFromJointPositions(
      const TArray<double>& JointPositions,
      TArray<FTransform>& OutLinkTransforms,
      EMochiArticulationLinkTransformSpace TransformSpace,
      double AngleScale = 1.0,
      double LengthScale = 100.0,
      int FinalLinkIndex = -1) const override;
  virtual bool ComputeJointPositionsFromLinkTransforms(
      const TArray<FTransform>& LinkTransforms,
      TArray<double>& OutJointPositions,
      EMochiArticulationLinkTransformSpace TransformSpace,
      double AngleScale = 1.0,
      double LengthScale = 100.0) const override;
  // Convenience wrapper around ComputeLinkTransformsFromJointPositions to compute the
  // forward kinematics for a single link. Prefer ComputeVisualBoundsFromLinkTransforms
  // with LinksToTransformSpace if you need to compute forward kinematics for multiple links.
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  bool ForwardKinematics(
      FName LinkName,
      const TArray<double>& JointPositions,
      FTransform& OutLinkTransform,
      double AngleScale = 1.0,
      double LengthScale = 100.0) const;
  // Compute the bounds of the articulation from an array of LinksToTransformSpace transforms. This
  // can be used as a replacement for USkeletalMesh::GetBounds() which in contrast does not change
  // with pose.
  virtual bool ComputeVisualBoundsFromLinkTransforms(
      const TArray<FTransform>& LinksToTransformSpace,
      EMochiArticulationLinkTransformSpace TransformSpace,
      FBoxSphereBounds& OutBounds) const override;
  // Clamp joint DOF positions between their lower and upper limits.
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  void ClampJointPositions(TArray<double>& JointPositions) const;
  // Generate an array of random joints positions within the joint limits.
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  void RandomJointPositions(TArray<double>& JointPositions, bool NormalDistribution = false) const;

 public:
  // Load the Mochi link shapes. This MUST be called on the UE game thread.
  TArray<mochi::ShapeHandle> GetLinkShapes() const;
  // Load the Mochi soft link shapes. This MUST be called on the UE game thread.
  TArray<mochi::ShapeHandle> GetSoftShapes() const;
  // Load the Mochi skin shape, if any. This MUST be called on the UE game thread.
  mochi::ShapeHandle GetSkinShape() const;
  // Build a mochi::ArticulatedActorParams. InitialJointVelocities follows the joint position
  // layout and may be empty. This MUST be called on the UE game thread.
  mochi::ArticulatedActorParams GetArticulatedActorParams(
      bool bInFixedBase,
      const UMochiContactParams* InContactParams,
      FName InLayer,
      bool bInHasGravity,
      mochi::Span<mochi::ShapeHandle const> InLinkShapes,
      mochi::ShapeHandle InSkinShape,
      const TArray<double>& InitialJointVelocities) const;

  virtual void PostLoad() override;
#if WITH_EDITOR
 protected:
  virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
  virtual void PostEditUndo() override;

 protected:
  // Get the ordered array of link names; for editor drop down menus.
  UFUNCTION(Category = "Mochi|Articulation")
  TArray<FName> GetLinkNames_EditorOnly() const;
  // Get the ordered array of joint names; for editor drop down menus.
  UFUNCTION(Category = "Mochi|Articulation")
  TArray<FName> GetJointNames_EditorOnly() const;
#endif

 protected:
  // True if any changes have invalidated the SkeletalMesh asset
  UPROPERTY()
  bool bNeedGenerateSkeletalMesh = true;

  // Derived from Joints and Links by BuildDerivedData.
  TArray<int> DofIndices;
  TArray<FTransform> RestLinksToParent;
  TArray<FTransform> RestLinksToRoot;
  TArray<TArray<int>> ChildLinkIndices;

 protected:
  // Find link returning pointer; non-const and const versions.
  FMochiArticulatedLink* FindLink(FName LinkName);
  const FMochiArticulatedLink* FindLink(FName LinkName) const;
  // Find joint returning pointer; non-const and const versions.
  FMochiArticulatedJoint* FindJoint(FName JointName);
  const FMochiArticulatedJoint* FindJoint(FName JointName) const;
  // Returns true if all link names are unique; this is a requirement for SkeletalMeshes.
  bool ValidateLinkNames() const;
  // Rebuilds DOF indices, rest transforms and child indices from Joints and Links, and resizes
  // DefaultJointPositions to match the number of DOFs. Returns false if the topology is invalid.
  bool BuildDerivedData();
  // Called just before the articulation is used to generate assets. Returns true on success.
  virtual bool PreGenValidateOrUpdateDerivedData();
  // Strip a leading articulation asset type prefix ("MA_" or the legacy "URDF_") from an asset
  // name. Sets bOutHadPrefix, if given, to whether one was found.
  static FString RemoveAssetTypePrefix(const FString& AssetName, bool* bOutHadPrefix = nullptr);

  friend class FMochiEditorModule;
  friend class FMochiArticulationDetailsCustomization;
  friend class FMochiArticulationEditorToolkit;
  friend class FMochiArticulationAssetUtils;
};
