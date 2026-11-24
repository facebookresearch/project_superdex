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

#include "MochiShellMeshComponent.generated.h"

/** Dynamic mesh component backed by a native shell actor. */
UCLASS(ClassGroup = (Custom), ShowCategories = (Mobility), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiShellMeshComponent : public UMochiDeformableMeshComponent {
  GENERATED_BODY()

 public:
  /** Parameters controlling shell material properties. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics")
  UMochiShellMaterialParams* ShellMaterialParams = nullptr;

  /** Collider representation used by the shell actor. */
  UPROPERTY(
      EditAnywhere,
      Category = "SuperDex Physics",
      meta = (ValidEnumValues = "None, Auto, PointCloud"))
  EMochiColliderType ColliderType =
      ::ToUnreal(mochi::experimental::ShellActorParams{}.colliderType);

  /** Use the shell's authored contact skin for contact sampling instead of the physics mesh. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics")
  bool bUseContactSkin = mochi::experimental::ShellActorParams{}.useContactSkin;

  /** Boundary elements used for shell contact sampling. */
  UPROPERTY(
      EditAnywhere,
      Category = "SuperDex Physics",
      meta = (ValidEnumValues = "P1Q1, P1Q3, P1Q6, Default"))
  EMochiActorBoundaryElementType ContactElementType =
      ::ToUnreal(mochi::experimental::ShellActorParams{}.contactElementType);

  /**
   * Point-cloud collider interaction radius [m]. Contact samples spaced further apart than this
   * may allow nodes to slip through. Increase for coarser meshes.
   */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics|Point Cloud Collider", meta = (Units = "m"))
  float PointCloudRadius = mochi::PointCloudColliderParams{}.radius;

  /** Enable self-contact for the shell. Experimental - may degrade performance. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics|Point Cloud Collider")
  bool bPointCloudSelfContact = mochi::PointCloudColliderParams{}.selfContact;

  /**
   * Self-contact exclusion ratio. Contact samples within (radius * ratio) of a node are excluded
   * from self-contact. Should be > 1.
   */
  UPROPERTY(
      EditAnywhere,
      Category = "SuperDex Physics|Point Cloud Collider",
      meta = (EditCondition = "bPointCloudSelfContact"))
  float PointCloudSelfContactExclusionRatio =
      mochi::PointCloudColliderParams{}.selfContactExclusionRatio;

  /** Unset uses one collider point per node; a value selects quadrature-based placement. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics|Point Cloud Collider")
  TOptional<EMochiActorBoundaryElementType> PointCloudColliderTriangleElementType;

 public:
  // Begin IMochiMeshComponentInterface
  virtual void CreateMochiActor() override;

 public:
  /**
   * Initialize the shell mesh component and its native-actor lifecycle settings.
   *
   * @param ObjectInitializer Unreal object-initialization settings.
   */
  UMochiShellMeshComponent(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

#if WITH_EDITOR
  virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

 protected:
  virtual void OnRegister() override;
};
