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

/** Dynamic mesh component backed by a native rod actor. */
UCLASS(ClassGroup = (Custom), ShowCategories = (Mobility), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiRodMeshComponent : public UMochiDeformableMeshComponent {
  GENERATED_BODY()

 public:
  /** Parameters controlling rod material properties. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics")
  UMochiRodMaterialParams* RodMaterialParams = nullptr;

  /** Collider representation used by the rod actor. */
  UPROPERTY(
      EditAnywhere,
      Category = "SuperDex Physics",
      meta = (ValidEnumValues = "None, Auto, PointCloud"))
  EMochiColliderType ColliderType = ::ToUnreal(mochi::RodActorParams{}.colliderType);

  /** Use the rod's contact skin for contact sampling instead of the centerline. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics")
  bool bUseContactSkin = mochi::RodActorParams{}.useContactSkin;

  /** Segment elements used for rod contact sampling. */
  UPROPERTY(
      EditAnywhere,
      Category = "SuperDex Physics",
      meta = (ValidEnumValues = "P1Q1, P1Q2, P1Q3, Default", EditCondition = "!bUseContactSkin"))
  EMochiActorSegmentElementType ContactElementType =
      ::ToUnreal(mochi::RodActorParams{}.contactElementType);

  /** Boundary elements used for contact-skin sampling. */
  UPROPERTY(
      EditAnywhere,
      Category = "SuperDex Physics",
      meta = (ValidEnumValues = "P1Q1, P1Q3, P1Q6, Default", EditCondition = "bUseContactSkin"))
  EMochiActorBoundaryElementType ContactSkinElementType =
      ::ToUnreal(mochi::RodActorParams{}.contactSkinElementType);

  /** Point-cloud collider interaction radius [m]. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics|Point Cloud Collider", meta = (Units = "m"))
  float PointCloudRadius = mochi::PointCloudColliderParams{}.radius;

  /** Enable self-contact for the rod. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics|Point Cloud Collider")
  bool bPointCloudSelfContact = mochi::PointCloudColliderParams{}.selfContact;

  /** Self-contact exclusion ratio. Should be > 1. */
  UPROPERTY(
      EditAnywhere,
      Category = "SuperDex Physics|Point Cloud Collider",
      meta = (EditCondition = "bPointCloudSelfContact"))
  float PointCloudSelfContactExclusionRatio =
      mochi::PointCloudColliderParams{}.selfContactExclusionRatio;

  /** Unset uses one collider point per node; a value selects quadrature-based placement. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics|Point Cloud Collider")
  TOptional<EMochiActorSegmentElementType> PointCloudColliderSegmentElementType;

 public:
  // Begin IMochiMeshComponentInterface
  virtual void CreateMochiActor() override;

#if WITH_EDITOR
  virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

 protected:
  virtual void OnRegister() override;

 private:
  /**
   * Rebuild the editor preview's dynamic mesh from Model. Skipped on archetypes/CDOs to avoid
   * creating a private UBodySetup on the Blueprint CDO.
   */
  void RebuildPreviewMesh();

  /**
   * Mark any UBodySetup on this component RF_Public so it serializes legally
   * when attached to a Blueprint CDO. See body for the failure mode.
   */
  void EnsureBodySetupPublic();
};
