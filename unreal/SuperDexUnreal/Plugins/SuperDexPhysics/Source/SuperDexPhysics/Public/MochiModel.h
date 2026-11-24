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
#include "MochiGenTypes.h"
#include "Serialization/BulkData.h"
#include "UObject/Object.h"

#include "MochiModel.generated.h"

namespace mochi {
struct ModelData;
} // namespace mochi

/** Source of collision geometry extracted from a static mesh. */
UENUM(BlueprintType)
enum class ETriangularMeshCollisionSource : uint8 {
  Simple UMETA(DisplayName = "Use Simple Collision (Without Spheres)"),
  Complex UMETA(DisplayName = "Use Complex Collision")
};

//------------------------------------------------------------------------------
// MOCHI MODEL
//------------------------------------------------------------------------------

/** Asset holding model geometry and its import and preview settings. */
UCLASS(BlueprintType, HideCategories = Object, DisplayName = "Mochi Model")
class SUPERDEXPHYSICS_API UMochiModel : public UObject {
  GENERATED_BODY()
 public:
  /** Number of nodes (vertices) in the volume mesh */
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Volume Mesh", AssetRegistrySearchable)
  int NumNodes = 0;
  /** Number of unique edges (line segments) on the polyhedrons */
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Volume Mesh", AssetRegistrySearchable)
  int NumEdges = 0;
  /** Number of polyhedrons within the volume (e.g. tetrahedrons) */
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Volume Mesh", AssetRegistrySearchable)
  int NumPolyhedrons = 0;

  /** Number of unique nodes (vertices) in the surface mesh */
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Surface Mesh", AssetRegistrySearchable)
  int NumSurfaceNodes = 0;
  /** Number of unique edges (line segments) in the surface mesh */
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Surface Mesh", AssetRegistrySearchable)
  int NumSurfaceEdges = 0;
  /** Number of unique faces (e.g. triangles) on the surface mesh */
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Surface Mesh", AssetRegistrySearchable)
  int NumSurfaceFaces = 0;

  // TODO: "link" this similarly to SKEL<->SKM<->PHYS assets
  /** Thumbnail and asset editor preview will be rendered using this material if set */
  UPROPERTY(EditAnywhere, Category = "Mochi")
  TObjectPtr<UMaterialInterface> PreviewRenderMaterial;

  // TODO: "link" this similarly to SKEL<->SKM<->PHYS assets
  /** Material modifier to edit using this model */
  UPROPERTY(EditAnywhere, Category = "Mochi")
  UMochiModelMaterialModifiers* MaterialModifiersAsset;

  /** The model's geometry. */
  UPROPERTY(VisibleAnywhere, Category = "Model Data")
  FMochiModelData ModelData;

#if WITH_EDITORONLY_DATA
  // TODO: Give this a nice interface to live outside of the details panel
  /** Values to assign to selected material modifiers */
  UPROPERTY(EditAnywhere, Category = "Mochi")
  FMochiMaterialModifier MaterialModifierTemplate;
#endif

 public:
  virtual void Serialize(FArchive& Ar) override;
  virtual void PostLoad() override;
  /**
   * Get whether model data contains a supported geometry member.
   *
   * @return Whether a supported geometry member is present.
   */
  bool IsValid() const;

  /**
   * Load the Mochi shape. This MUST be called on the UE game thread.
   * This function loads the shape with a transform that incorporates, in the following order:
   *    1. ImportTransform (SRT) expressed in LH, Z up, cm (i.e. Unreal space)
   *    2. BakeScale expressed in LH, Z up, (i.e. Unreal space)
   *    3. BakeTransformRT expressed in LH, Z up, cm (i.e. Unreal space)
   *    4. Coordinate space conversion to the project settings' simulation space.
   * In other words, BakeScale and BakeTransformRT should come from Unreal component
   * transforms or transforms otherwise expressed Unreal's LH, Z up, cm space.
   *
   * @param BakeScale Scale baked into the shape geometry.
   * @param BakeTransformRT Transform baked into the shape geometry.
   * @return The mochi shape.
   */
  virtual mochi::ShapeHandle GetMochiShape(
      const FVector& BakeScale,
      const FTransform& BakeTransformRT) const;
  /**
   * [Internal] Same as above, with BakeTransformRT as a mochi::VMatrix4x4r.
   *
   * @param BakeScale Scale baked into the shape geometry.
   * @param BakeTransformRT Transform baked into the shape geometry.
   * @return The mochi shape.
   */
  virtual mochi::ShapeHandle GetMochiShape(
      const FVector& BakeScale,
      const mochi::VMatrix4x4r& BakeTransformRT) const;
  /**
   * [Internal] Load the Mochi shape WITHOUT implicit ImportTransform or space conversion.
   *
   * @param BakeMatrix Matrix baked into the shape geometry.
   * @return The mochi shape raw.
   */
  virtual mochi::ShapeHandle GetMochiShape_Raw(const mochi::VMatrix4x4r& BakeMatrix) const;

  /**
   * Create a native triangular shape from the requested static-mesh collision source.
   *
   * @param Mesh Static mesh supplying collision geometry.
   * @param CollisionSource Static-mesh collision geometry to extract.
   * @param Scale3D Scale applied to the geometry.
   * @return The triangular mochi shape from static mesh.
   */
  static mochi::ShapeHandle GetTriangularMochiShapeFromStaticMesh(
      const UStaticMesh* Mesh,
      const ETriangularMeshCollisionSource CollisionSource,
      FVector Scale3D);

  /**
   * [Internal] Get the model's raw data (mesh + skinning) from ModelData.
   * Unlike GetMochiShape (which builds a physics ShapeHandle), this returns owning CPU-side
   * ModelData so callers can inspect or transform it -- e.g. the bot loader remaps a skin's baked
   * bone indices onto a composed skeleton before creating a shape. Returns default-constructed data
   * with `Error` set on failure (including when the asset holds no data).
   *
   * @param Error Native error state passed through the operation.
   * @return Owning model data produced by the asset-loading operation.
   */
  mochi::ModelData GetMochiModelData(mochi::Error& Error) const;

#if WITH_EDITOR
  virtual void PostInitProperties() override;
  virtual void GetAssetRegistryTags(FAssetRegistryTagsContext Context) const override;
  /**
   * Absolute path of the file this model was last imported from, or empty.
   *
   * @return The source file path.
   */
  FString GetSourceFilePath() const;

  /** Prompts for a Mochi model file and replaces this asset's contents with it. */
  UFUNCTION(Category = "Import", CallInEditor)
  void Import();
  /** Repeats the last import from the recorded source file. */
  UFUNCTION(Category = "Import", CallInEditor)
  void Reimport();
  /**
   * Replaces this asset's contents with the Mochi model file at InFilename. On success, records
   * it as the model's source file.
   *
   * @param InFilename Model file to read or write.
   * @return The result of importing the source asset.
   */
  bool ImportFromFilename(const FString& InFilename);
  /**
   * Writes this asset out as a Mochi model file.
   *
   * @param InFilename Model file to read or write.
   * @return The result of exporting the asset.
   */
  bool ExportToFilename(const FString& InFilename) const;
  /**
   * Replaces the mesh with the surface extracted from it.
   * Returns true on success. Fails if the mesh is not tetrahedral.
   *
   * @return Whether the model geometry was converted to a surface mesh.
   */
  bool ConvertTetMeshToSurfaceMesh();
  /**
   * Bakes an SDF into the model data using the provided parameters.
   * If no parameters are provided, uses default GridSdfParams.
   *
   * @param SdfParams Grid signed-distance-field parameters to use while baking.
   * @return Whether signed-distance-field baking completed.
   */
  bool BakeSdf(const FMochiGridSdfParams* SdfParams = nullptr);
  /**
   * Set the transform applied when producing imported model and shape data.
   *
   * @param Transform Transform supplied to the operation.
   */
  void SetImportTransform(FTransform Transform) {
    ImportTransform = Transform;
  }
#endif

#if WITH_EDITOR
 private:
  /**
   * Replaces the model's contents and refreshes the mesh counts. Used by every editor
   * path that produces geometry, so that none of them has to go through a file.
   */
  bool ApplyModelData(const mochi::ModelData& Data);

  /**
   * Recomputes NumNodes, NumSurfaceFaces and the rest by creating a shape and measuring
   * it. Returns false if the model does not produce a usable shape.
   */
  bool RefreshMeshCounts();
#endif

 private:
  /** Bulk-backed ModelData.Sdf->Values; the reflected array is empty during property saves. */
  void SerializeSdfValues(FArchive& Ar);

  /** Transform applied to imported geometry when producing native model or shape data. */
  UPROPERTY(EditAnywhere, Category = "Import")
  FTransform ImportTransform = FTransform::Identity;

#if WITH_EDITORONLY_DATA
 public:
  /**
   * The source file this model was imported from, for reimport. Like other imported assets, a
   * file under the project's Data Source Folder is stored relative to that folder.
   */
  UPROPERTY(VisibleAnywhere, Category = "Model Data")
  TObjectPtr<UAssetImportData> AssetImportData;

 private:
  /**
   * Absolute path of the source file, from before the model used AssetImportData. Moved into
   * AssetImportData on load.
   */
  UPROPERTY()
  FString Filename_DEPRECATED;
#endif

 private:
  /**
   * Storage for ModelData.Sdf->Values. Bulk data sizes are 64-bit and sit outside the
   * tagged property stream, so a grid larger than 2 GB still serializes.
   */
  FByteBulkData SdfValues;
};
