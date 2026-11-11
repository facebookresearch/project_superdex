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
#include "Subsystems/GameInstanceSubsystem.h"

#include "SuperDexRuntimeRenderUtils.generated.h"

SUPERDEXPHYSICS_API DECLARE_LOG_CATEGORY_EXTERN(LogSuperDexRuntimeMesh, Log, All);

struct FMeshDescription;
class UInterchangePipelineBase;
class UMaterialInterface;
class USkeletalMesh;
class USuperDexRuntimeMeshPipeline;
class UStaticMesh;

// Runtime loading of render models (.glb) that were never imported in the editor, e.g. those of a
// runtime-loaded .superdex_scene.
namespace SuperDexRuntimeRender {

// Read the first skinned mesh of a .glb into OutMesh, in Unreal's basis and units as Interchange
// imports a .glb: static mesh attributes plus vertex skin weights. Joint j of the glTF skin binds
// to bone JointToBone[j]. Influences on a joint that maps to no bone (INDEX_NONE or out of range)
// are dropped and the rest renormalized; a vertex left with none binds fully to bone 0.
// OutMaterialNames receives the glTF material name of each polygon group, indexed by polygon group
// ID. Returns false if the file cannot be read or has no skinned mesh, or the platform has no glTF
// reader (SUPERDEX_WITH_RUNTIME_MESH_IMPORT=0, e.g. Android).
SUPERDEXPHYSICS_API bool ReadSkinnedGlb(
    const FString& GlbPath,
    TConstArrayView<int32> JointToBone,
    FMeshDescription& OutMesh,
    TArray<FString>& OutMaterialNames);

// The material of StaticMesh -- a skinned .glb imported as a static mesh -- for the polygon group
// ReadSkinnedGlb named MaterialName: the slot of that (imported) name, or else slot 0.
SUPERDEXPHYSICS_API UMaterialInterface* FindMaterialForSlot(
    const UStaticMesh& StaticMesh,
    const FString& MaterialName);

} // namespace SuperDexRuntimeRender

// A skinned render model (.glb) and the flat skeleton to build it over (see
// USuperDexRuntimeMeshSubsystem::RequestSkinnedMesh).
struct FSuperDexSkinnedMeshRequest {
  // Absolute .glb path.
  FString FilePath;
  // Per joint of the glTF skin: the bone (index into the built skeleton, where bone 0 is the root
  // and bone i + 1 is BoneNames[i]) it binds to, or INDEX_NONE to drop its influence.
  TArray<int32> JointToBone;
  // The name of skeleton bone i + 1.
  TArray<FName> BoneNames;
  // The reference pose of skeleton bone i + 1 in component space [cm], one per BoneNames entry.
  // Scale is ignored.
  TArray<FTransform> RestBoneToComponent;
  // The placement of the .glb's vertices in component space, scale (including mirroring) applied.
  FTransform MeshToComponent = FTransform::Identity;
};

/**
 * Imports render models (.glb) into transient UStaticMeshes at runtime, so content that was never
 * imported in the editor -- e.g. the render models inside a runtime-loaded .superdex_scene -- can
 * still be drawn in a packaged build.
 *
 * Imports go through Interchange with the same mesh settings as the editor's MochiAssetsPipeline,
 * so a runtime mesh matches the one the editor importers would have produced. glTF materials become
 * instances of the engine's glTF reference materials (/InterchangeAssets/gltf), since full
 * materials cannot be built at runtime. Results are cached for the lifetime of the game instance,
 * keyed on file content, so identical files at different paths (two bots extracted from separate
 * archives) import once.
 *
 * Skinned render models (RequestSkinnedMesh) are built into transient USkeletalMeshes over a flat
 * skeleton the caller describes, with the materials of the file's static mesh. Interchange cannot
 * create skeletal meshes at runtime, so their render data is filled directly. They are cached for
 * the lifetime of the game instance too, keyed on file content and skeleton.
 *
 * Packaged builds must cook /InterchangeAssets/gltf (e.g. DirectoriesToAlwaysCook in the project's
 * DefaultGame.ini): Interchange names those parent materials only by path, so the cook does not
 * find them on its own, and without them every imported material is left without a parent.
 */
UCLASS()
class SUPERDEXPHYSICS_API USuperDexRuntimeMeshSubsystem final : public UGameInstanceSubsystem {
  GENERATED_BODY()

 public:
  using FOnMeshLoaded = TFunction<void(UStaticMesh* /*Mesh, null on failure*/)>;
  using FOnSkinnedMeshLoaded = TFunction<void(USkeletalMesh* /*Mesh, null on failure*/)>;

  /**
   * Request the static mesh for an absolute .glb path. OnLoaded runs on the game thread:
   * immediately if the file was already imported (or already failed), otherwise when the import
   * finishes. It is passed null if the import failed, this process cannot render, or the platform
   * has no runtime import (SUPERDEX_WITH_RUNTIME_MESH_IMPORT=0, e.g. Android).
   */
  void RequestStaticMesh(const FString& FilePath, FOnMeshLoaded OnLoaded);

  /**
   * Request a transient skeletal mesh of a skinned .glb over the flat skeleton the request
   * describes: bone 0 is a root at the component origin and bone i + 1 is BoneNames[i], parented to
   * the root. Its materials are those of the file's static mesh (RequestStaticMesh), matched by
   * slot name. OnLoaded runs on the game thread, as for RequestStaticMesh, and is passed null if
   * the file cannot be read or the platform has no runtime import. Meshes are cached for the
   * lifetime of the game instance, keyed on file content and skeleton.
   */
  void RequestSkinnedMesh(FSuperDexSkinnedMeshRequest Request, FOnSkinnedMeshLoaded OnLoaded);

  virtual void Initialize(FSubsystemCollectionBase& Collection) override;
  virtual void Deinitialize() override;

 private:
  void OnImportDone(
      const FString& ContentKey,
      const FString& FilePath,
      UStaticMesh* Mesh,
      const TArray<UObject*>& Imported);

  // Typed as the base class (InterchangeCore) so this header does not need the Interchange plugin's
  // pipelines, which only exist on desktop platforms.
  UPROPERTY()
  TObjectPtr<UInterchangePipelineBase> Pipeline;

  UPROPERTY()
  TObjectPtr<UInterchangePipelineBase> GltfPipeline;

  UPROPERTY()
  TObjectPtr<USuperDexRuntimeMeshPipeline> RuntimeMeshPipeline;

  // Every object the imports created (meshes, materials, textures), kept alive for the session.
  UPROPERTY()
  TArray<TObjectPtr<UObject>> ImportedObjects;

  UPROPERTY()
  TMap<FString, TObjectPtr<UStaticMesh>> MeshesByContent;

  // Built skinned meshes, keyed on file content and skeleton; null for a failed build.
  UPROPERTY()
  TMap<FString, TObjectPtr<USkeletalMesh>> SkinnedMeshesByKey;

  TSet<FString> FailedContent;
  TMap<FString, TArray<FOnMeshLoaded>> PendingByContent;
  TMap<FString, FString> ContentKeyByPath;
  bool bDeinitialized = false;
};
