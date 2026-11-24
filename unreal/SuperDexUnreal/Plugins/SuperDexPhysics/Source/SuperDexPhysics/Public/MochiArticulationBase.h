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

#include "MochiArticulationBase.generated.h"

class USkeletalMesh;
class FSceneView;
class FViewport;
class FCanvas;
class FPrimitiveDrawInterface;
class FMaterialRenderProxy;

/**
 * Abstract base class for articulated body assets (UMochiArticulation, USuperDexBot).
 *
 * Provides the minimal interface needed by consumers that work with any articulated body
 * regardless of its source format.
 */
UCLASS(Abstract, BlueprintType)
class SUPERDEXPHYSICS_API UMochiArticulationBase : public UObject {
  GENERATED_BODY()
 public:
  /**
   * Get the name of the articulation (e.g. "my_arm", "left_hand").
   *
   * @return The articulation name.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  virtual const FString& GetArticulationName() const PURE_VIRTUAL(
      UMochiArticulationBase::GetArticulationName, static FString Empty; return Empty;);

  /**
   * Get the number of revolute/prismatic DOFs.
   *
   * @return Number of degrees of freedom in the articulation or joint.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  virtual int GetNumDofs() const PURE_VIRTUAL(UMochiArticulationBase::GetNumDofs, return 0;);

  /**
   * Get the default or home pose joint positions [rad or m].
   *
   * @return The configured default joint positions.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  virtual const TArray<double>& GetDefaultJointPositions() const PURE_VIRTUAL(
      UMochiArticulationBase::GetDefaultJointPositions, static TArray<double> Empty; return Empty;);

  /**
   * Compute each link relative to its parent. The root entry retains the asset's root
   * convention: UMochiArticulation includes its root-joint transform; USuperDexBot uses identity.
   * AngleScale and LengthScale convert joint units to radians/centimeters (default: rad/m).
   * FinalLinkIndex limits computation; -1 computes all link transforms.
   *
   * @param JointPositions Joint-position values supplied to this operation.
   * @param OutLinkTransforms Storage for the computed link transforms.
   * @param AngleScale Scale converting angular joint values to radians.
   * @param LengthScale Scale converting linear joint values to Unreal lengths.
   * @param FinalLinkIndex Last link to compute; -1 selects all links.
   * @return Whether the requested computation completed.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  bool ComputeLinkToParentTransforms(
      const TArray<double>& JointPositions,
      TArray<FTransform>& OutLinkTransforms,
      double AngleScale = 1.0,
      double LengthScale = 100.0,
      int FinalLinkIndex = -1) const;

  /**
   * Compute accumulated link transforms in the asset's articulation root frame.
   * For UMochiArticulation this is the frame its root joint attaches to, not the root link;
   * for USuperDexBot this is the native bot's root link (whose transform remains identity).
   * AngleScale and LengthScale convert joint units to radians/centimeters (default: rad/m).
   * FinalLinkIndex limits computation; -1 computes all link transforms.
   *
   * @param JointPositions Joint-position values supplied to this operation.
   * @param OutLinkTransforms Storage for the computed link transforms.
   * @param AngleScale Scale converting angular joint values to radians.
   * @param LengthScale Scale converting linear joint values to Unreal lengths.
   * @param FinalLinkIndex Last link to compute; -1 selects all links.
   * @return Whether the requested computation completed.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  bool ComputeLinkToArticulationRootTransforms(
      const TArray<double>& JointPositions,
      TArray<FTransform>& OutLinkTransforms,
      double AngleScale = 1.0,
      double LengthScale = 100.0,
      int FinalLinkIndex = -1) const;

  /**
   * Compute OutJointPositions from one parent-relative transform per link.
   * Inverts ComputeLinkToParentTransforms, dividing joint motion by AngleScale/LengthScale.
   * Revolute motion is projected onto its axis (the nearest twist in [-pi, pi]); prismatic
   * motion onto its axis; spherical motion becomes a rotation vector. Unsupported motion
   * is discarded, and a link's error only affects its own joint.
   *
   * @param LinkTransforms Link transforms supplied to the computation.
   * @param OutJointPositions Storage for the computed joint positions.
   * @param AngleScale Scale converting angular joint values to radians.
   * @param LengthScale Scale converting linear joint values to Unreal lengths.
   * @return Whether the requested computation completed.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  bool ComputeJointPositionsFromLinkToParentTransforms(
      const TArray<FTransform>& LinkTransforms,
      TArray<double>& OutJointPositions,
      double AngleScale = 1.0,
      double LengthScale = 100.0) const;

  /**
   * Compute OutJointPositions from one articulation-root-relative transform per link.
   * Inverts ComputeLinkToArticulationRootTransforms with the same scales and joint-motion
   * projection as ComputeJointPositionsFromLinkToParentTransforms. The articulation root
   * frame follows each asset's convention, as in the forward method.
   *
   * @param LinkTransforms Link transforms supplied to the computation.
   * @param OutJointPositions Storage for the computed joint positions.
   * @param AngleScale Scale converting angular joint values to radians.
   * @param LengthScale Scale converting linear joint values to Unreal lengths.
   * @return Whether the requested computation completed.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  bool ComputeJointPositionsFromLinkToArticulationRootTransforms(
      const TArray<FTransform>& LinkTransforms,
      TArray<double>& OutJointPositions,
      double AngleScale = 1.0,
      double LengthScale = 100.0) const;

  /**
   * Get the SkeletalMesh associated with this articulation.
   *
   * @return The skeletal mesh associated with the articulation.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  virtual USkeletalMesh* GetSkeletalMesh() const
      PURE_VIRTUAL(UMochiArticulationBase::GetSkeletalMesh, return nullptr;);

  /**
   * Get the ordered array of link names.
   *
   * @return Link names in articulation order.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  virtual TArray<FName> GetLinkNames() const
      PURE_VIRTUAL(UMochiArticulationBase::GetLinkNames, return {};);

  /**
   * Get the ordered array of joint names
   *
   * @return Joint names in articulation order.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  virtual TArray<FName> GetJointNames() const
      PURE_VIRTUAL(UMochiArticulationBase::GetJointNames, return {};);

  /**
   * Compute pose-dependent visual bounds by accumulating link-to-parent transforms.
   * OutBounds uses the asset's articulation root frame, as in
   * ComputeLinkToArticulationRootTransforms.
   *
   * @param LinksToParent Parent-relative link transforms.
   * @param OutBounds Storage for the computed visual bounds.
   * @return Whether the requested computation completed.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  bool ComputeVisualBoundsFromLinkToParentTransforms(
      const TArray<FTransform>& LinksToParent,
      FBoxSphereBounds& OutBounds) const;

  /**
   * Compute pose-dependent visual bounds from already accumulated articulation-root transforms.
   * OutBounds uses the asset's articulation root frame, as in
   * ComputeLinkToArticulationRootTransforms.
   *
   * @param LinksToArticulationRoot Articulation-root-relative link transforms.
   * @param OutBounds Storage for the computed visual bounds.
   * @return Whether the requested computation completed.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  bool ComputeVisualBoundsFromLinkToArticulationRootTransforms(
      const TArray<FTransform>& LinksToArticulationRoot,
      FBoxSphereBounds& OutBounds) const;

  /**
   * Whether ToUrdfXml can export the articulation.
   *
   * @return Whether this articulation type supports URDF export.
   */
  virtual bool SupportsUrdfExport() const {
    return false;
  }

  /**
   * Generate the URDF XML string, or an empty string if the articulation can't be exported. Visual
   * and collision geometry entries are optionally encoded when VisualRelPath and/or
   * CollisionRelPath are provided (e.g., "./visual" and "./collision"). Specifying these does NOT
   * actually export the visual meshes and/or collision Mochi models.
   *
   * @param VisualRelPath Relative path used for visual geometry entries in exported URDF.
   * @param CollisionRelPath Relative path used for collision geometry entries in exported URDF.
   * @return URDF XML, or an empty string when the articulation cannot be exported.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  virtual FString ToUrdfXml(
      const FString& VisualRelPath = TEXT(""),
      const FString& CollisionRelPath = TEXT("")) const {
    return {};
  }

 public:
  /**
   * Compute the bounds of the articulation in a given joint configuration. This can be
   * used as a replacement for USkeletalMesh::GetBounds() which in contrast does
   * not change with pose.
   *
   * @param JointPositions Joint-position values supplied to this operation.
   * @param OutBounds Storage for the computed visual bounds.
   * @param AngleScale Scale converting angular joint values to radians.
   * @param LengthScale Scale converting linear joint values to Unreal lengths.
   * @return Whether the requested computation completed.
   */
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  bool ComputeVisualBoundsFromJointPositions(
      const TArray<double>& JointPositions,
      FBoxSphereBounds& OutBounds,
      double AngleScale = 1.0,
      double LengthScale = 100.0) const;

  /**
   * Returns {TotalMass [kg], NumSurfaceFaces} for the articulation editor viewport's info
   * overlay panel (FMochiArticulationEditorToolkit::SpawnTab_Viewport). Default
   * implementation returns zero. UMochiArticulation computes this inline in the editor
   * toolkit instead (it has direct access to its own per-link mesh data there via
   * Cast<UMochiArticulation>), so this hook exists purely for other subclasses (e.g.
   * bots) that the toolkit cannot Cast<> to without a cross-plugin dependency.
   *
   * @param OutMass Storage for the computed articulation mass.
   * @param OutNumSurfaceFaces Storage for the computed surface-face count.
   */
  virtual void ComputeOverlayStats(double& OutMass, int& OutNumSurfaceFaces) const {
    OutMass = 0.0;
    OutNumSurfaceFaces = 0;
  }

  /**
   * Editor-only viewport draw hooks (FMochiArticulationEditorViewportClient::Draw/
   * DrawCanvas) for subclasses whose draw logic can't be expressed generically here
   * without a compile-time dependency on their concrete type/module (e.g. bots, which
   * live in the sibling SuperDexRobotics plugin). UMochiArticulation's own drawing is handled
   * directly by FMochiArticulationEditorViewportClient::DrawArticulation/
   * DrawCanvasArticulation and does not go through these hooks. Default implementations
   * draw nothing. Show-flags and the joint gizmo material are passed in explicitly since
   * they are private state owned by the viewport client, not the articulation asset.
   *
   * @param View Scene view used for editor rendering.
   * @param PDI Primitive drawing interface used for debug geometry.
   * @param bShowLinkTransforms Whether to draw link transforms.
   * @param bShowJoints Whether to draw joint annotations.
   * @param bShowBounds Whether to draw visual bounds.
   * @param JointMaterialRenderProxy Material proxy used to draw joints.
   */
  virtual void DrawCustom(
      const FSceneView* View,
      FPrimitiveDrawInterface* PDI,
      bool bShowLinkTransforms,
      bool bShowJoints,
      bool bShowBounds,
      FMaterialRenderProxy* JointMaterialRenderProxy) const {}

  /**
   * Draw asset-specific articulation annotations on the viewport canvas.
   *
   * @param InViewport Viewport used by the editor operation.
   * @param View Scene view used for editor rendering.
   * @param Canvas Canvas used for editor annotations.
   * @param DPIScale Display scale used for canvas annotations.
   * @param bShowLinkTransforms Whether to draw link transforms.
   * @param bShowJoints Whether to draw joint annotations.
   */
  virtual void DrawCanvasCustom(
      FViewport& InViewport,
      FSceneView& View,
      FCanvas& Canvas,
      float DPIScale,
      bool bShowLinkTransforms,
      bool bShowJoints) const {}

 protected:
  virtual bool ComputeLinkTransformsInternal(
      const TArray<double>& JointPositions,
      TArray<FTransform>& OutLinkTransforms,
      bool bToArticulationRoot,
      double AngleScale,
      double LengthScale,
      int FinalLinkIndex) const
      PURE_VIRTUAL(UMochiArticulationBase::ComputeLinkTransformsInternal, return false;);

  virtual bool ComputeJointPositionsInternal(
      const TArray<FTransform>& LinkTransforms,
      TArray<double>& OutJointPositions,
      bool bFromArticulationRoot,
      double AngleScale,
      double LengthScale) const
      PURE_VIRTUAL(UMochiArticulationBase::ComputeJointPositionsInternal, return false;);

  virtual bool ComputeVisualBoundsInternal(
      const TArray<FTransform>& LinksToTransformSpace,
      bool bToArticulationRoot,
      FBoxSphereBounds& OutBounds) const
      PURE_VIRTUAL(UMochiArticulationBase::ComputeVisualBoundsInternal, return false;);

  /**
   * Hook for FMochiArticulationEditorToolkit which needs to know when the SKM data
   * changes but doesn't necessarily result in a change of pointer address.
   */
  TFunction<void(USkeletalMesh*)> OnSkeletalMeshRegenerated;

  friend class FMochiEditorModule;
  friend class FMochiArticulationDetailsCustomization;
  friend class FMochiArticulationEditorToolkit;
  friend class FMochiArticulationAssetUtils;
};
