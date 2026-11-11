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
#include "EditorFramework/AssetImportData.h"
#include "Materials/MaterialInterface.h"
#include "MochiAssets.h"
#include "MochiModelData.h"
#include "Serialization/BulkData.h"
#include "UObject/Object.h"

#include "MochiModel.generated.h"

namespace mochi {
struct ModelData;
} // namespace mochi

UENUM(BlueprintType)
enum class ETriangularMeshCollisionSource : uint8 {
  Simple UMETA(DisplayName = "Use Simple Collision (Without Spheres)"),
  Complex UMETA(DisplayName = "Use Complex Collision")
};

//------------------------------------------------------------------------------
// MOCHI MODEL
//------------------------------------------------------------------------------

UCLASS(BlueprintType, HideCategories = Object, DisplayName = "Mochi Model")
class SUPERDEXPHYSICS_API UMochiModel : public UObject {
  GENERATED_BODY()
 public:
  // Number of nodes (vertices) in the volume mesh
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Volume Mesh", AssetRegistrySearchable)
  int NumNodes = 0;
  // Number of unique edges (line segments) on the polyhedrons
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Volume Mesh", AssetRegistrySearchable)
  int NumEdges = 0;
  // Number of polyhedrons within the volume (e.g. tetrahedrons)
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Volume Mesh", AssetRegistrySearchable)
  int NumPolyhedrons = 0;

  // Number of unique nodes (vertices) in the surface mesh
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Surface Mesh", AssetRegistrySearchable)
  int NumSurfaceNodes = 0;
  // Number of unique edges (line segments) in the surface mesh
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Surface Mesh", AssetRegistrySearchable)
  int NumSurfaceEdges = 0;
  // Number of unique faces (e.g. triangles) on the surface mesh
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Surface Mesh", AssetRegistrySearchable)
  int NumSurfaceFaces = 0;

  // Thumbnail and asset editor preview will be rendered using this material if set
  // TODO: "link" this similarly to SKEL<->SKM<->PHYS assets
  UPROPERTY(EditAnywhere, Category = "Mochi")
  TObjectPtr<UMaterialInterface> PreviewRenderMaterial;

  // Material modifier to edit using this model
  // TODO: "link" this similarly to SKEL<->SKM<->PHYS assets
  UPROPERTY(EditAnywhere, Category = "Mochi")
  UMochiModelMaterialModifiers* MaterialModifiersAsset;

  // The model's geometry.
  UPROPERTY(VisibleAnywhere, Category = "Model Data")
  FMochiModelData ModelData;

#if WITH_EDITORONLY_DATA
  // Values to assign to selected material modifiers
  // TODO: Give this a nice interface to live outside of the details panel
  UPROPERTY(EditAnywhere, Category = "Mochi")
  FMochiMaterialModifier MaterialModifierTemplate;
#endif

 public:
  virtual void Serialize(FArchive& Ar) override;
  virtual void PostLoad() override;
  bool IsValid() const;

  // Load the Mochi shape. This MUST be called on the UE game thread.
  // This function loads the shape with a transform that incorporates, in the following order:
  //    1. ImportTransform (SRT) expressed in LH, Z up, cm (i.e. Unreal space)
  //    2. BakeScale expressed in LH, Z up, (i.e. Unreal space)
  //    3. BakeTransformRT expressed in LH, Z up, cm (i.e. Unreal space)
  //    4. Coordinate space conversion to the project settings' simulation space.
  // In other words, BakeScale and BakeTransformRT should come from Unreal component
  // transforms or transforms otherwise expressed Unreal's LH, Z up, cm space.
  virtual mochi::ShapeHandle GetMochiShape(
      const FVector& BakeScale,
      const FTransform& BakeTransformRT) const;
  // [Internal] Same as above, with BakeTransformRT as a mochi::VMatrix4x4r.
  virtual mochi::ShapeHandle GetMochiShape(
      const FVector& BakeScale,
      const mochi::VMatrix4x4r& BakeTransformRT) const;
  // [Internal] Load the Mochi shape WITHOUT implicit ImportTransform or space conversion.
  virtual mochi::ShapeHandle GetMochiShape_Raw(const mochi::VMatrix4x4r& BakeMatrix) const;

  static mochi::ShapeHandle GetTriangularMochiShapeFromStaticMesh(
      const UStaticMesh* mesh,
      const ETriangularMeshCollisionSource collisionSource,
      FVector scale3D);

  // [Internal] Get the model's raw data (mesh + skinning) from ModelData.
  // Unlike GetMochiShape (which builds a physics ShapeHandle), this returns owning CPU-side
  // ModelData so callers can inspect or transform it -- e.g. the bot loader remaps a skin's baked
  // bone indices onto a composed skeleton before creating a shape. Returns default-constructed data
  // with `error` set on failure (including when the asset holds no data).
  mochi::ModelData GetMochiModelData(mochi::Error& error) const;

#if WITH_EDITOR
  virtual void PostInitProperties() override;
  virtual void GetAssetRegistryTags(FAssetRegistryTagsContext Context) const override;
  // Absolute path of the file this model was last imported from, or empty.
  FString GetSourceFilePath() const;

  // Prompts for a Mochi model file and replaces this asset's contents with it.
  UFUNCTION(Category = "Import", CallInEditor)
  void Import();
  // Repeats the last import from the recorded source file.
  UFUNCTION(Category = "Import", CallInEditor)
  void Reimport();
  // Replaces this asset's contents with the Mochi model file at InFilename. On success, records
  // it as the model's source file.
  bool ImportFromFilename(const FString& InFilename);
  // Writes this asset out as a Mochi model file.
  bool ExportToFilename(const FString& InFilename) const;
  // Replaces the mesh with the surface extracted from it.
  // Returns true on success. Fails if the mesh is not tetrahedral.
  bool ConvertTetMeshToSurfaceMesh();
  // Bakes an SDF into the model data using the provided parameters.
  // If no parameters are provided, uses default GridSdfParams.
  bool BakeSdf(const FMochiGridSdfParams* SdfParams = nullptr);
  void SetImportTransform(FTransform transform) {
    ImportTransform = transform;
  }
#endif

#if WITH_EDITOR
 private:
  // Replaces the model's contents and refreshes the mesh counts. Used by every editor
  // path that produces geometry, so that none of them has to go through a file.
  bool ApplyModelData(const mochi::ModelData& Data);
  // Recomputes NumNodes, NumSurfaceFaces and the rest by creating a shape and measuring
  // it. Returns false if the model does not produce a usable shape.
  bool RefreshMeshCounts();
#endif

 private:
  // Moves ModelData.Sdf->Values in and out of SdfValues, which is where it lives on disk.
  void SerializeSdfValues(FArchive& Ar);

  UPROPERTY(EditAnywhere, Category = "Import")
  FTransform ImportTransform = FTransform::Identity;

#if WITH_EDITORONLY_DATA
 public:
  // The source file this model was imported from, for reimport. Like other imported assets, a
  // file under the project's Data Source Folder is stored relative to that folder.
  UPROPERTY(VisibleAnywhere, Category = "Model Data")
  TObjectPtr<UAssetImportData> AssetImportData;

 private:
  // Absolute path of the source file, from before the model used AssetImportData. Moved into
  // AssetImportData on load.
  UPROPERTY()
  FString Filename_DEPRECATED;
#endif

 private:
  // Storage for ModelData.Sdf->Values. Bulk data sizes are 64-bit and sit outside the
  // tagged property stream, so a grid larger than 2 GB still serializes.
  FByteBulkData SdfValues;
};
