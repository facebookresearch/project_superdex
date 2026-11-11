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
#include "MochiActorObject.h"
#include "MochiDeformableMeshComponent.h"

#include "MochiRodMeshComponent.generated.h"

UCLASS(ClassGroup = (Custom), ShowCategories = (Mobility), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiRodMeshComponent : public UMochiDeformableMeshComponent {
  GENERATED_BODY()

 public:
  // Parameters controlling rod material properties.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  UMochiRodMaterialParams* RodMaterialParams = nullptr;
  UPROPERTY(EditAnywhere, Category = "Mochi", meta = (ValidEnumValues = "None, Auto, PointCloud"))
  EMochiColliderType ColliderType = EMochiColliderType::None;
  // Use the rod's contact skin for contact sampling instead of the centerline.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  bool bUseContactSkin = false;
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi",
      meta = (ValidEnumValues = "P1Q1, P1Q2, P1Q3, Default", EditCondition = "!bUseContactSkin"))
  EMochiActorSegmentElementType ContactElementType = EMochiActorSegmentElementType::Default;
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi",
      meta = (ValidEnumValues = "P1Q1, P1Q3, P1Q6, Default", EditCondition = "bUseContactSkin"))
  EMochiActorBoundaryElementType ContactSkinElementType = EMochiActorBoundaryElementType::Default;

  // When the assigned Mochi Model contains a contact skin, use it instead of
  // generating a tube. If the model has no contact skin, a tube is generated from
  // the radius and segment counts below.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi|Contact Skin",
      meta = (DisplayName = "Use Model Contact Skin (If Available)"))
  bool bUseModelContactSkin = true;

  // Read-only indicator of whether the assigned Mochi Model currently provides a
  // contact skin. Refreshed whenever the model geometry is (re)loaded.
  UPROPERTY(
      VisibleAnywhere,
      Transient,
      Category = "Mochi|Contact Skin",
      meta = (DisplayName = "Model Has Contact Skin"))
  bool bModelHasContactSkin = false;

  // Radius of the procedural tubular contact skin [m]. Only used when a tube is
  // generated procedurally (see bUseModelContactSkin).
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi|Contact Skin",
      meta = (Units = "m", EditCondition = "!bUseModelContactSkin || !bModelHasContactSkin"))
  float ProceduralTubeRadius = 0.001f;
  // Number of segments around the tube circumference. Only used when a tube is
  // generated procedurally (see bUseModelContactSkin).
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi|Contact Skin",
      meta =
          (UIMin = 3,
           UIMax = 32,
           ClampMin = 3,
           EditCondition = "!bUseModelContactSkin || !bModelHasContactSkin"))
  int32 NumCrossSectionSegments = 8;

  // Point-cloud collider interaction radius [m].
  UPROPERTY(EditAnywhere, Category = "Mochi|Point Cloud Collider", meta = (Units = "m"))
  float PointCloudRadius = 0.01f;
  // Enable self-contact for the rod.
  UPROPERTY(EditAnywhere, Category = "Mochi|Point Cloud Collider")
  bool bPointCloudSelfContact = false;
  // Self-contact exclusion ratio. Should be > 1.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi|Point Cloud Collider",
      meta = (EditCondition = "bPointCloudSelfContact"))
  float PointCloudSelfContactExclusionRatio = 1.5f;
  // Discretization element type for the rod's point-cloud collider. Nodal (the
  // default) corresponds to one-contact-per-node. P1Q1/P1Q2/P1Q3 enable denser, quadrature-based
  // contact sampling on linear segments.
  UPROPERTY(EditAnywhere, Category = "Mochi|Point Cloud Collider")
  EMochiActorSegmentElementType ColliderSegmentElementType = EMochiActorSegmentElementType::Nodal;

 public:
  // Begin IMochiMeshComponentInterface
  virtual void CreateMochiActor() override;

 public:
  UMochiRodMeshComponent(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

 public:
  /// UObject lifecycle hooks that funnel through RebakeIfNeeded so the editor
  /// preview tube radius stays in sync with the current ComponentToWorld scale.
  virtual void PostLoad() override;
  virtual void PostEditImport() override;
  virtual void PostDuplicate(bool bDuplicateForPIE) override;

#if WITH_EDITOR
  virtual bool CanEditChange(const FProperty* InProperty) const override;
  virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
  virtual void PostEditUndo() override;
#endif

 protected:
  virtual void OnRegister() override;
  virtual void OnUpdateTransform(EUpdateTransformFlags UpdateTransformFlags, ETeleportType Teleport)
      override;

 private:
  /// Rebuild the editor preview's dynamic mesh from the rod shape using the
  /// component's current scale. Mirrors CreateMochiActor's build path so the
  /// preview is WYSIWYG, and updates LastBakedScale. Skipped on archetypes/CDOs
  /// to avoid creating a private UBodySetup on the Blueprint CDO.
  void RebuildPreviewMesh();

  /// Funnel for editor-only lifecycle hooks; skips on archetypes and after
  /// BeginPlay.
  void RebakeIfNeeded();

  /// Mark any UBodySetup on this component RF_Public so it serializes legally
  /// when attached to a Blueprint CDO. See body for the failure mode.
  void EnsureBodySetupPublic();

  mochi::ShapeHandle CreateRodShape(const FVector& BakeScale, bool& bOutModelHasContactSkin) const;

  /// Scale at which the editor preview mesh was last baked; initialized to an
  /// impossible value so the first bake always occurs.
  FVector LastBakedScale = FVector(TNumericLimits<double>::Lowest());

  // One-shot guard so a continuously animated parent scale doesn't flood the
  // log every frame after BeginPlay.
  bool bWarnedRuntimeScaleChange = false;
};
