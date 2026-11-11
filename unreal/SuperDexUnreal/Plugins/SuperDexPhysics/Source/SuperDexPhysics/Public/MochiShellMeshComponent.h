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

UCLASS(ClassGroup = (Custom), ShowCategories = (Mobility), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiShellMeshComponent : public UMochiDeformableMeshComponent {
  GENERATED_BODY()

 public:
  // Parameters controlling shell material properties.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  UMochiShellMaterialParams* ShellMaterialParams = nullptr;
  UPROPERTY(EditAnywhere, Category = "Mochi", meta = (ValidEnumValues = "None, Auto, PointCloud"))
  EMochiColliderType ColliderType = EMochiColliderType::PointCloud;
  // Use the shell's authored contact skin for contact sampling instead of the physics mesh.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  bool bUseContactSkin = false;
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi",
      meta = (ValidEnumValues = "P1Q1, P1Q3, P1Q6, Default"))
  EMochiActorBoundaryElementType ContactElementType = EMochiActorBoundaryElementType::Default;

  // Point-cloud collider interaction radius [m]. Contact samples spaced further apart than this
  // may allow nodes to slip through. Increase for coarser meshes.
  UPROPERTY(EditAnywhere, Category = "Mochi|Point Cloud Collider", meta = (Units = "m"))
  float PointCloudRadius = 0.01f;
  // Enable self-contact for the shell. Experimental - may degrade performance.
  UPROPERTY(EditAnywhere, Category = "Mochi|Point Cloud Collider")
  bool bPointCloudSelfContact = false;
  // Self-contact exclusion ratio. Contact samples within (radius * ratio) of a node are excluded
  // from self-contact. Should be > 1.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi|Point Cloud Collider",
      meta = (EditCondition = "bPointCloudSelfContact"))
  float PointCloudSelfContactExclusionRatio = 1.5f;
  // Discretization element type for the shell's point-cloud collider. Nodal (the
  // default) corresponds to one-contact-per-node. P1Q1/P1Q3/P1Q6 enable denser, quadrature-based
  // contact sampling on triangles.
  UPROPERTY(EditAnywhere, Category = "Mochi|Point Cloud Collider")
  EMochiActorBoundaryElementType ColliderTriangleElementType =
      EMochiActorBoundaryElementType::Nodal;

 public:
  // Begin IMochiMeshComponentInterface
  virtual void CreateMochiActor() override;

 public:
  UMochiShellMeshComponent(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

#if WITH_EDITOR
  virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

 protected:
  virtual void OnRegister() override;
};
