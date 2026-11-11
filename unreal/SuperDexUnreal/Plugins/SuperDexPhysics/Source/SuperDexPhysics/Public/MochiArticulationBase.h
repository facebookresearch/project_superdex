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

enum class EMochiArticulationLinkTransformSpace : uint8;
class USkeletalMesh;
class FSceneView;
class FViewport;
class FCanvas;
class FPrimitiveDrawInterface;
class FMaterialRenderProxy;

/// Abstract base class for articulated body assets (UMochiArticulation, USuperDexBot).
///
/// Provides the minimal interface needed by consumers that work with any articulated body
/// regardless of its source format.
UCLASS(Abstract, BlueprintType)
class SUPERDEXPHYSICS_API UMochiArticulationBase : public UObject {
  GENERATED_BODY()
 public:
  /// Get the name of the articulation (e.g. "my_arm", "left_hand").
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  virtual const FString& GetArticulationName() const PURE_VIRTUAL(
      UMochiArticulationBase::GetArticulationName, static FString empty; return empty;);
  /// Get the number of revolute/prismatic DOFs.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  virtual int GetNumDofs() const PURE_VIRTUAL(UMochiArticulationBase::GetNumDofs, return 0;);
  /// Get the default or home pose joint positions [rad or m].
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  virtual const TArray<double>& GetDefaultJointPositions() const PURE_VIRTUAL(
      UMochiArticulationBase::GetDefaultJointPositions, static TArray<double> empty; return empty;);
  /// Compute link transforms for a given joint configuration. AngleScale and LengthScale
  /// convert JointPositions units to radians/centimeters. Defaults assume radians/meters.
  /// FinalLinkIndex can minimize unnecessary computations; -1 computes all link transforms.
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  virtual bool ComputeLinkTransformsFromJointPositions(
      const TArray<double>& JointPositions,
      TArray<FTransform>& OutLinkTransforms,
      EMochiArticulationLinkTransformSpace TransformSpace,
      double AngleScale = 1.0,
      double LengthScale = 100.0,
      int FinalLinkIndex = -1) const
      PURE_VIRTUAL(UMochiArticulationBase::ComputeLinkTransformsFromJointPositions, return false;);
  /// Compute the joint configuration of given link transforms (one per link): the inverse of
  /// ComputeLinkTransformsFromJointPositions, with the same TransformSpace and scales (the output
  /// is divided by AngleScale/LengthScale). Each joint is read from its link's transform relative
  /// to the link's parent as given (in either space), limited to what the joint can do: a revolute
  /// angle is the twist of the joint motion about the axis (the nearest rotation about it, in
  /// [-pi, pi]), a prismatic position is the joint motion's translation along the axis, and a
  /// spherical joint is the rotation vector of the joint motion. Anything else, such as swing off a
  /// revolute axis, is discarded, so transforms from ComputeLinkTransformsFromJointPositions
  /// round-trip and an error in one link's transform only affects its own joint.
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  virtual bool ComputeJointPositionsFromLinkTransforms(
      const TArray<FTransform>& LinkTransforms,
      TArray<double>& OutJointPositions,
      EMochiArticulationLinkTransformSpace TransformSpace,
      double AngleScale = 1.0,
      double LengthScale = 100.0) const
      PURE_VIRTUAL(UMochiArticulationBase::ComputeJointPositionsFromLinkTransforms, return false;);
  /// Get the SkeletalMesh associated with this articulation.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  virtual USkeletalMesh* GetSkeletalMesh() const
      PURE_VIRTUAL(UMochiArticulationBase::GetSkeletalMesh, return nullptr;);
  /// Get the ordered array of link names.
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  virtual TArray<FName> GetLinkNames() const
      PURE_VIRTUAL(UMochiArticulationBase::GetLinkNames, return {};);
  // Get the ordered array of joint names
  UFUNCTION(BlueprintPure, Category = "Mochi|Articulation")
  virtual TArray<FName> GetJointNames() const
      PURE_VIRTUAL(UMochiArticulationBase::GetJointNames, return {};);
  // Compute the bounds of the articulation from an array of LinksToTransformSpace transforms. This
  // can be used as a replacement for USkeletalMesh::GetBounds() which in contrast does not change
  // with pose.
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  virtual bool ComputeVisualBoundsFromLinkTransforms(
      const TArray<FTransform>& LinksToTransformSpace,
      EMochiArticulationLinkTransformSpace TransformSpace,
      FBoxSphereBounds& OutBounds) const
      PURE_VIRTUAL(UMochiArticulationBase::ComputeVisualBoundsFromLinkTransforms, return false;);
  // Whether ToUrdfXml can export the articulation.
  virtual bool SupportsUrdfExport() const {
    return false;
  }
  // Generate the URDF XML string, or an empty string if the articulation can't be exported. Visual
  // and collision geometry entries are optionally encoded when VisualRelPath and/or
  // CollisionRelPath are provided (e.g., "./visual" and "./collision"). Specifying these does NOT
  // actually export the visual meshes and/or collision Mochi models.
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  virtual FString ToUrdfXml(
      const FString& VisualRelPath = TEXT(""),
      const FString& CollisionRelPath = TEXT("")) const {
    return {};
  }

 public:
  // Compute the bounds of the articulation in a given joint configuration. This can be
  // used as a replacement for USkeletalMesh::GetBounds() which in contrast does
  // not change with pose.
  UFUNCTION(BlueprintCallable, Category = "Mochi|Articulation")
  bool ComputeVisualBoundsFromJointPositions(
      const TArray<double>& JointPositions,
      FBoxSphereBounds& OutBounds,
      double AngleScale = 1.0,
      double LengthScale = 100.0) const;

  /// Returns {TotalMass [kg], NumSurfaceFaces} for the articulation editor viewport's info
  /// overlay panel (FMochiArticulationEditorToolkit::SpawnTab_Viewport). Default
  /// implementation returns zero. UMochiArticulation computes this inline in the editor
  /// toolkit instead (it has direct access to its own per-link mesh data there via
  /// Cast<UMochiArticulation>), so this hook exists purely for other subclasses (e.g.
  /// bots) that the toolkit cannot Cast<> to without a cross-plugin dependency.
  virtual void ComputeOverlayStats(double& OutMass, int& OutNumSurfaceFaces) const {
    OutMass = 0.0;
    OutNumSurfaceFaces = 0;
  }

  /// Editor-only viewport draw hooks (FMochiArticulationEditorViewportClient::Draw/
  /// DrawCanvas) for subclasses whose draw logic can't be expressed generically here
  /// without a compile-time dependency on their concrete type/module (e.g. bots, which
  /// live in the sibling SuperDexRobotics plugin). UMochiArticulation's own drawing is handled
  /// directly by FMochiArticulationEditorViewportClient::DrawArticulation/
  /// DrawCanvasArticulation and does not go through these hooks. Default implementations
  /// draw nothing. Show-flags and the joint gizmo material are passed in explicitly since
  /// they are private state owned by the viewport client, not the articulation asset.
  virtual void DrawCustom(
      const FSceneView* View,
      FPrimitiveDrawInterface* PDI,
      bool bShowLinkTransforms,
      bool bShowJoints,
      bool bShowBounds,
      FMaterialRenderProxy* JointMaterialRenderProxy) const {}
  virtual void DrawCanvasCustom(
      FViewport& InViewport,
      FSceneView& View,
      FCanvas& Canvas,
      float DPIScale,
      bool bShowLinkTransforms,
      bool bShowJoints) const {}

 protected:
  // Hook for FMochiArticulationEditorToolkit which needs to know when the SKM data
  // changes but doesn't necessarily result in a change of pointer address.
  TFunction<void(USkeletalMesh*)> OnSkeletalMeshRegenerated;

  friend class FMochiEditorModule;
  friend class FMochiArticulationDetailsCustomization;
  friend class FMochiArticulationEditorToolkit;
  friend class FMochiArticulationAssetUtils;
};
