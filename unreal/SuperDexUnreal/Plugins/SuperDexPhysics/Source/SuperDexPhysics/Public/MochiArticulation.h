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

/**
 * An articulated body described with the same data model as a Mochi articulated actor
 * (mochi::ArticulatedActorParams): one joint per link, where Joints[i] connects Links[i] to
 * Links[Links[i].ParentLink]. The root joint (Joints[0]) attaches the root link to the frame of
 * the component that spawns the articulation.
 *
 * Joint positions (DefaultJointPositions, FK inputs, etc.) contain one value per DOF of every
 * joint except a Free root joint, whose 6 DOFs belong to the base: 1 for Revolute and Prismatic
 * joints, and 3 (a rotation vector in the joint frame) for Spherical joints.
 */
UCLASS(
    BlueprintType,
    HideCategories = Object,
    DisplayName = "Mochi Articulation",
    PrioritizeCategories = ("Articulation", "Assets", "Internal"),
    AutoCollapseCategories = ("Internal"))
class SUPERDEXPHYSICS_API UMochiArticulation : public UMochiArticulationBase {
  GENERATED_BODY()
 protected:
  /** The name of the articulation described. */
  UPROPERTY(EditAnywhere, Category = "Articulation", BlueprintReadOnly, AssetRegistrySearchable)
  FString ArticulationName;

  /** Optional joint positions that define a default "home" pose of the articulation [rad or m] */
  UPROPERTY(
      EditAnywhere,
      Category = "Articulation",
      BlueprintReadOnly,
      AssetRegistrySearchable,
      EditFixedSize,
      meta = (EditFixedOrder))
  TArray<double> DefaultJointPositions;

  /** The joints of the articulation, one per link. */
  UPROPERTY(
      EditAnywhere,
      Category = "Articulation",
      BlueprintReadOnly,
      meta = (TitleProperty = "Name"))
  TArray<FMochiArticulatedJoint> Joints;

  /** The links of the articulation. Parents must precede their children. */
  UPROPERTY(
      EditAnywhere,
      Category = "Articulation",
      BlueprintReadOnly,
      meta = (TitleProperty = "Name"))
  TArray<FMochiArticulatedLink> Links;

  /** Cycle-closing joints for closed-loop topologies. */
  UPROPERTY(EditAnywhere, Category = "Articulation", BlueprintReadOnly)
  TArray<FMochiArticulatedCycleJoint> Cycles;

  /** Whether the articulation includes a collision skin. */
  UPROPERTY(
      EditAnywhere,
      Category = "Articulation",
      BlueprintReadOnly,
      meta = (InlineEditConditionToggle))
  bool bHasSkin = false;

  /** Optional deformable collision skin wrapped over the links. */
  UPROPERTY(
      EditAnywhere,
      Category = "Articulation",
      BlueprintReadOnly,
      meta = (EditCondition = "bHasSkin"))
  FMochiArticulatedSkin Skin;

  /**
   * Link pairs whose contact is enabled or disabled, applied in order after the links are created.
   * Adjacent links have contact disabled by default.
   */
  UPROPERTY(EditAnywhere, Category = "Articulation", BlueprintReadOnly, AssetRegistrySearchable)
  TArray<FMochiArticulationContactOverride> ContactOverrides;

  /** Additional constraints to create between rigid links. */
  UPROPERTY(EditAnywhere, Category = "Articulation", BlueprintReadOnly, AssetRegistrySearchable)
  TArray<FMochiArticulationConstraint> AdditionalConstraints;

  /** Soft links to attach to rigid links */
  UPROPERTY(EditAnywhere, Category = "Articulation", BlueprintReadOnly, AssetRegistrySearchable)
  TArray<FMochiArticulationSoftLink> SoftLinks;

  /** Whether the rigid links also collide with other actors when there are soft links. */
  UPROPERTY(EditAnywhere, Category = "Articulation", BlueprintReadOnly)
  bool bEnableCollidingLinks = mochi::SoftSkinnedActorParams{}.enableCollidingLinks;

  /**
   * The SkeletalMesh asset associated with the articulation. Can be regenerated with Generate
   * Skeletal Mesh.
   */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Assets", AssetRegistrySearchable)
  TObjectPtr<USkeletalMesh> SkeletalMesh = nullptr;

  /** User added notes/README about the articulation. */
  UPROPERTY(EditAnywhere, Category = "Internal", meta = (MultiLine = true))
  FString Notes;

 public:
  /**
   * Check that the articulation is ready to be used (assets generated and data consistent).
   * This is a mildly expensive operation; call only once in BeginPlay or similar.
   *
   * @param bLogWarnings Whether validation warnings are logged.
   * @param OutWarnings Warning storage used while validating the articulation.
   * @return Whether the articulation definition passes validation.
   */
  bool IsReady(bool bLogWarnings = true, TArray<FString>* OutWarnings = nullptr) const;

  /**
   * Replace the description of the articulation. Rebuilds derived data and resizes
   * DefaultJointPositions; the SkeletalMesh must be regenerated afterwards.
   *
   * @param InArticulationName Name to assign to the articulation.
   * @param InJoints Joint definitions to assign to the articulation.
   * @param InLinks Link definitions to assign to the articulation.
   * @param InCycles Cycle-joint definitions to assign to the articulation.
   * @param InSkin Collision-skin definition to assign to the articulation.
   */
  void SetDefinition(
      const FString& InArticulationName,
      TArray<FMochiArticulatedJoint> InJoints,
      TArray<FMochiArticulatedLink> InLinks,
      TArray<FMochiArticulatedCycleJoint> InCycles = {},
      const FMochiArticulatedSkin* InSkin = nullptr);

  /**
   * Get the name of the articulation.
   *
   * @return The articulation name.
   */
  virtual const FString& GetArticulationName() const override;

  /**
   * Get the list of links.
   *
   * @return Link definitions in articulation order.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  const TArray<FMochiArticulatedLink>& GetLinks() const;

  /**
   * Get the list of joints, one per link.
   *
   * @return Joint definitions, one per link.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  const TArray<FMochiArticulatedJoint>& GetJoints() const;

  /**
   * Get the list of cycle-closing joints.
   *
   * @return Cycle-closing joint definitions.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  const TArray<FMochiArticulatedCycleJoint>& GetCycles() const;

  /**
   * Returns true if the articulation has a collision skin.
   *
   * @return Whether a collision skin is configured.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  bool HasSkin() const;

  /**
   * Get the collision skin. Only meaningful if HasSkin().
   *
   * @return The configured collision-skin definition.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  const FMochiArticulatedSkin& GetSkin() const;

  /**
   * Returns true if the articulation is configured with soft link attachments.
   *
   * @return Whether soft-link attachments are configured.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  bool HasSoftLinks() const;

  /**
   * Get the list of *soft* links.
   *
   * @return The configured soft-link attachments.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  const TArray<FMochiArticulationSoftLink>& GetSoftLinks() const;

  /** Whether the rigid links also collide with other actors when there are soft links. */
  bool GetEnableCollidingLinks() const;

  /**
   * Get the link contact overrides.
   *
   * @return The contact overrides.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  const TArray<FMochiArticulationContactOverride>& GetContactOverrides() const;

  /**
   * Replace the link contact overrides.
   *
   * @param InContactOverrides Link contact overrides to assign to the articulation.
   */
  void SetContactOverrides(TArray<FMochiArticulationContactOverride> InContactOverrides);

  /**
   * Get the list of additional constraints
   *
   * @return The additional constraints.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  const TArray<FMochiArticulationConstraint>& GetAdditionalConstraints() const;

  /**
   * Get the default or home pose joint positions.
   *
   * @return The configured default joint positions.
   */
  virtual const TArray<double>& GetDefaultJointPositions() const override;

  /**
   * Returns true if the root joint does not let the articulation move freely, i.e. it is not a
   * Free joint. This is the default for UMochiArticulationComponent::bFixedBase.
   *
   * @return Whether the articulation's root joint is fixed by default.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  bool DefaultFixedBase() const;

  /**
   * The root joint type a component spawns the articulation with. bFixedBase overrides a Free or
   * Hard root joint (Hard if true, Free otherwise); other root joint types are kept as is.
   *
   * @param bFixedBase Whether to fix the articulation base.
   * @return Root-joint type after applying the fixed-base choice.
   */
  EMochiArticulatedJointType GetEffectiveRootJointType(bool bFixedBase) const;

  /**
   * Get a link index by name. This indexes into the array returned by GetLinks().
   * Return -1 if name does not refer to a valid link.
   *
   * @param LinkName Name of the requested articulation link.
   * @return Index of the named link, or -1 when it is not found.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  int GetLinkIndex(FName LinkName) const;

  /**
   * Get a link by name. Returns true if found, false otherwise.
   *
   * @param LinkName Name of the requested articulation link.
   * @param OutLink Storage for the requested link definition.
   * @return Whether the named link was found.
   */
  UFUNCTION(
      BlueprintCallable,
      meta = (ExpandBoolAsExecs = "ReturnValue"),
      Category = "Mochi|Articulation")
  bool GetLink(FName LinkName, FMochiArticulatedLink& OutLink) const;

  /**
   * Get a joint index by name. This indexes into the array returned by GetJoints().
   * Return -1 if name does not refer to a valid joint.
   *
   * @param JointName Name of the requested articulation joint.
   * @return Index of the named joint, or -1 when it is not found.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  int GetJointIndex(FName JointName) const;

  /**
   * Get a joint by name. Returns true if found, false otherwise.
   *
   * @param JointName Name of the requested articulation joint.
   * @param OutJoint Storage for the requested joint definition.
   * @return Whether the named joint was found.
   */
  UFUNCTION(
      BlueprintCallable,
      meta = (ExpandBoolAsExecs = "ReturnValue"),
      Category = "Mochi|Articulation")
  bool GetJoint(FName JointName, FMochiArticulatedJoint& OutJoint) const;

  /**
   * Get the joint connecting a link to its parent. Returns false if the index is out of bounds.
   *
   * @param LinkIndex Index of the articulation link.
   * @param OutJoint Storage for the requested joint definition.
   * @return Whether the link index resolved to a joint.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  bool GetLinkJoint(int LinkIndex, FMochiArticulatedJoint& OutJoint) const;

  /**
   * Get the indices of a link's children.
   *
   * @param LinkIndex Index of the articulation link.
   * @return The child link indices.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  TArray<int> GetChildLinkIndices(int LinkIndex) const;

  /**
   * Get all the leaf link indices, i.e. those without child links.
   *
   * @return The leaf link indices.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  TArray<int> GetLeafLinkIndices() const;

  /**
   * Get either the last link or first forking link. Useful for getting the
   * "end-effector" of articulation arms with hand attachments.
   *
   * @return The last or first forking link.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  FMochiArticulatedLink GetLastOrFirstForkingLink() const;

  /**
   * Get the number of DOFs, excluding the 6 DOFs of a Free root joint.
   *
   * @return Number of degrees of freedom in the articulation or joint.
   */
  virtual int GetNumDofs() const override;

  /**
   * Get the joint index of each DOF. Spherical joints appear three times.
   *
   * @return Joint index for each articulated degree of freedom.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  const TArray<int>& GetDofIndices() const;

  /**
   * Get the index of the first DOF of a joint. Returns -1 if the joint has no DOFs.
   *
   * @param JointName Name of the requested articulation joint.
   * @return Index of the joint's first DOF, or -1 when it has no DOFs.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  int GetDofIndex(FName JointName) const;

  /**
   * Get the joint for a given *DOF* index; not to be confused with joint index.
   *
   * @param DofIndex Index of the degree of freedom.
   * @param OutJoint Storage for the requested joint definition.
   * @return Whether the DOF index resolved to a joint.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  bool GetDof(int DofIndex, FMochiArticulatedJoint& OutJoint) const;

  /**
   * Get the lower and upper limits of a DOF [rad or m]. Unlimited DOFs return -/+infinity.
   *
   * @param DofIndex Index of the degree of freedom.
   * @param OutLower Storage for the lower joint-motion limit.
   * @param OutUpper Storage for the upper joint-motion limit.
   * @return Whether limits were available for the requested DOF.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  bool GetDofLimits(int DofIndex, double& OutLower, double& OutUpper) const;

  /**
   * Get the transform of a link relative to its parent link when all joint positions are zero.
   *
   * @param LinkIndex Index of the articulation link.
   * @return Rest transform of the link relative to its parent.
   */
  FTransform GetRestLinkToParent(int LinkIndex) const;

  /**
   * Get the transform of a link relative to the articulation root frame when all joint positions
   * are zero.
   *
   * @param LinkIndex Index of the articulation link.
   * @return Rest transform of the link relative to the articulation root.
   */
  FTransform GetRestLinkToRoot(int LinkIndex) const;

  /**
   * Get the ordered array of link names.
   *
   * @return Link names in articulation order.
   */
  virtual TArray<FName> GetLinkNames() const override;

  /**
   * Get the ordered array of joint names
   *
   * @return Joint names in articulation order.
   */
  virtual TArray<FName> GetJointNames() const override;

  /**
   * Get the SkeletalMesh to use with this articulation.
   *
   * @return The skeletal mesh associated with the articulation.
   */
  virtual USkeletalMesh* GetSkeletalMesh() const override;

  /**
   * Compute forward kinematics for a single link in the articulation root frame.
   * Prefer ComputeLinkToArticulationRootTransforms when computing multiple links.
   *
   * @param LinkName Name of the requested articulation link.
   * @param JointPositions Joint-position values supplied to this operation.
   * @param OutLinkTransform Storage for the computed link transform.
   * @param AngleScale Scale converting angular joint values to radians.
   * @param LengthScale Scale converting linear joint values to Unreal lengths.
   * @return Whether the requested link transform was computed.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  bool ForwardKinematics(
      FName LinkName,
      const TArray<double>& JointPositions,
      FTransform& OutLinkTransform,
      double AngleScale = 1.0,
      double LengthScale = 100.0) const;

  /**
   * Clamp joint DOF positions between their lower and upper limits.
   *
   * @param JointPositions Joint-position values to read and clamp in place.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  void ClampJointPositions(TArray<double>& JointPositions) const;

  /**
   * Generate an array of random joints positions within the joint limits.
   *
   * @param JointPositions Array to fill with sampled joint-position values.
   * @param bNormalDistribution Whether to sample from a normal distribution instead of a uniform
   * distribution.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  void RandomJointPositions(TArray<double>& JointPositions, bool bNormalDistribution = false) const;

 public:
  /**
   * Load the Mochi link shapes. This MUST be called on the UE game thread.
   *
   * @return The link shapes.
   */
  TArray<mochi::ShapeHandle> GetLinkShapes() const;

  /**
   * Load the Mochi soft link shapes. This MUST be called on the UE game thread.
   *
   * @return The soft shapes.
   */
  TArray<mochi::ShapeHandle> GetSoftShapes() const;

  /**
   * Load the Mochi skin shape, if any. This MUST be called on the UE game thread.
   *
   * @return The skin shape.
   */
  mochi::ShapeHandle GetSkinShape() const;

  /**
   * Build a mochi::ArticulatedActorParams. InitialJointVelocities follows the joint position
   * layout and may be empty. This MUST be called on the UE game thread.
   *
   * @param bInFixedBase Whether to fix the articulation base.
   * @param InContactParams Contact parameters supplied to actor conversion.
   * @param InLayer Native contact layer supplied to actor conversion.
   * @param bInHasGravity Whether gravity affects the created articulation.
   * @param InLinkShapes Native shapes of the articulation links.
   * @param InSkinShape Native shape handle for the articulation skin.
   * @param InitialJointVelocities Joint velocities used when creating the articulation.
   * @return The articulated actor params.
   */
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
  /** Get the ordered array of link names; for editor drop down menus. */
  UFUNCTION(Category = "Mochi|Articulation")
  TArray<FName> GetLinkNames_EditorOnly() const;

  /** Get the ordered array of joint names; for editor drop down menus. */
  UFUNCTION(Category = "Mochi|Articulation")
  TArray<FName> GetJointNames_EditorOnly() const;
#endif

 protected:
  /** True if any changes have invalidated the SkeletalMesh asset */
  UPROPERTY()
  bool bNeedGenerateSkeletalMesh = true;

  /** Derived from Joints and Links by BuildDerivedData. */
  TArray<int> DofIndices;
  TArray<FTransform> RestLinksToParent;
  TArray<FTransform> RestLinksToRoot;
  TArray<TArray<int>> ChildLinkIndices;

 protected:
  virtual bool ComputeLinkTransformsInternal(
      const TArray<double>& JointPositions,
      TArray<FTransform>& OutLinkTransforms,
      bool bToArticulationRoot,
      double AngleScale,
      double LengthScale,
      int FinalLinkIndex) const override;

  virtual bool ComputeJointPositionsInternal(
      const TArray<FTransform>& LinkTransforms,
      TArray<double>& OutJointPositions,
      bool bFromArticulationRoot,
      double AngleScale,
      double LengthScale) const override;

  virtual bool ComputeVisualBoundsInternal(
      const TArray<FTransform>& LinksToTransformSpace,
      bool bToArticulationRoot,
      FBoxSphereBounds& OutBounds) const override;

  /** Find link returning pointer; non-const and const versions. */
  FMochiArticulatedLink* FindLink(FName LinkName);
  const FMochiArticulatedLink* FindLink(FName LinkName) const;

  /** Find joint returning pointer; non-const and const versions. */
  FMochiArticulatedJoint* FindJoint(FName JointName);
  const FMochiArticulatedJoint* FindJoint(FName JointName) const;

  /** Returns true if all link names are unique; this is a requirement for SkeletalMeshes. */
  bool ValidateLinkNames() const;

  /**
   * Rebuilds DOF indices, rest transforms and child indices from Joints and Links, and resizes
   * DefaultJointPositions to match the number of DOFs. Returns false if the topology is invalid.
   */
  bool BuildDerivedData();

  /** Called just before the articulation is used to generate assets. Returns true on success. */
  virtual bool PreGenValidateOrUpdateDerivedData();

  /**
   * Strip a leading articulation asset type prefix ("MA_" or the legacy "URDF_") from an asset
   * name. Sets bOutHadPrefix, if given, to whether one was found.
   */
  static FString RemoveAssetTypePrefix(const FString& AssetName, bool* bOutHadPrefix = nullptr);

  friend class FMochiEditorModule;
  friend class FMochiArticulationDetailsCustomization;
  friend class FMochiArticulationEditorToolkit;
  friend class FMochiArticulationAssetUtils;
};
