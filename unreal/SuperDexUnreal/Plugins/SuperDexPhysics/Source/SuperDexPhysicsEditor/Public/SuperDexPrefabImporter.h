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
#include "MochiConversions.h"
#include "SuperDexPrefab.h"

#include <mochi_physics/utils/mochi_prefab.h>

class AActor;
class FSuperDexImportSession;
class UBlueprint;
class UMochiArticulation;
class UMochiContactParams;
class UMochiMaterialParams;
class UMochiModel;
class UMochiRodMaterialParams;
class USceneComponent;
class UStaticMesh;
class UWorld;

/**
 * Imports .superdex_prefab, .mochi_prefab and .mochi_scene files authored in SuperDex Studio into
 * USuperDexPrefab assets and the assets generated from them: render meshes, Mochi models, contact,
 * material and rod material params, articulations, and either a Blueprint with a component per
 * Mochi actor or a Map with an actor per Mochi actor plus the prefab's scene settings. Nested
 * prefabs are imported recursively as their own USuperDexPrefab Blueprints, each in a subfolder of
 * the folder of the prefab that nests it, and placed as child actor components or actors.
 */
class SUPERDEXPHYSICSEDITOR_API FSuperDexPrefabImporter {
 public:
  /**
   * Where generated actors are added: a Blueprint's construction script or a map. Everything
   * added is tagged so that the next import can replace it.
   */
  class FOutput {
   public:
    /** Destroy the output builder through its interface. */
    virtual ~FOutput() = default;
    /**
     * Add the root component of an actor. Its relative transform places it in the output.
     *
     * @param ComponentClass Class of the component to create in the output.
     * @param Name Name used by this operation.
     * @return The component returned by the output builder.
     */
    virtual USceneComponent* AddActor(UClass* ComponentClass, const FString& Name) = 0;
    /**
     * Add a component attached to one returned by AddActor or AddChild.
     *
     * @param Parent Existing component to which the new child attaches.
     * @param ComponentClass Class of the component to create in the output.
     * @param Name Name used by this operation.
     * @return The child component returned by the output builder.
     */
    virtual USceneComponent*
    AddChild(USceneComponent* Parent, UClass* ComponentClass, const FString& Name) = 0;
    /**
     * Add an instance of an imported prefab's Blueprint.
     *
     * @param Nested Blueprint of the nested prefab.
     * @param Name Name used by this operation.
     * @param Transform Transform applied to the nested prefab.
     */
    virtual void
    AddNestedPrefab(UBlueprint* Nested, const FString& Name, const FTransform& Transform) = 0;
    /**
     * Add an actor of ActorClass, such as scene settings. Returns nullptr for Blueprints, which
     * cannot contain actors.
     *
     * @param ActorClass Class of the scene actor to create.
     * @param Name Name used by this operation.
     * @return The scene actor returned by the output builder.
     */
    virtual AActor* AddSceneActor(UClass* ActorClass, const FString& Name) = 0;
  };

  /**
   * Create a prefab importer using the supplied import session.
   *
   * @param InSession Import session sharing asset caches and naming.
   */
  explicit FSuperDexPrefabImporter(FSuperDexImportSession& InSession);

  /**
   * Import the prefab at SourcePath into Prefab, generating assets next to it. Returns false if
   * the prefab could not be imported.
   *
   * @param Prefab Prefab asset whose generated output is updated.
   * @param SourcePath Source prefab or scene file.
   * @param Mode Editor or prefab import mode.
   * @return The result of importing the source asset.
   */
  bool Import(USuperDexPrefab* Prefab, const FString& SourcePath, ESuperDexPrefabImportMode Mode);
  /**
   * Import the prefab at SourcePath as a Blueprint into its own USuperDexPrefab asset in
   * DestFolder, unless it is already up to date. Returns nullptr if the prefab could not be
   * imported.
   *
   * @param SourcePath Source prefab or scene file.
   * @param DestFolder Destination Content Browser folder.
   * @return The result of importing the source asset.
   */
  USuperDexPrefab* ImportPrefab(const FString& SourcePath, const FString& DestFolder);
  /**
   * Add the scene settings (gravity, solver, contact filter) of the prefab at SourcePath to
   * Output, generating their assets under DestFolder. The paths of the generated assets relative
   * to DestFolder are added to OutGeneratedAssetPaths.
   *
   * @param SourcePath Source prefab or scene file.
   * @param DestFolder Destination Content Browser folder.
   * @param Output Existing Blueprint or Map output to populate.
   * @param OutGeneratedAssetPaths Generated-asset paths to extend during import.
   */
  void AddPrefabSceneSettings(
      const FString& SourcePath,
      const FString& DestFolder,
      FOutput& Output,
      TArray<FString>& OutGeneratedAssetPaths);

  /**
   * Create or update the actor Blueprint at PackagePath: remove the components generated by the
   * previous import, then call Populate to add the new ones. Returns nullptr on failure.
   *
   * @param PackagePath Package path for the generated asset or output.
   * @param Populate Callback that populates the new output.
   * @return The generated Blueprint output.
   */
  static UBlueprint* BuildBlueprint(
      const FString& PackagePath,
      TFunctionRef<void(FOutput&)> Populate);
  /**
   * Create or update the map at PackagePath: remove the actors generated by the previous import,
   * then call Populate to add the new ones. New maps start from the map template setting. Maps
   * not open in the editor are saved. Returns nullptr on failure.
   *
   * @param PackagePath Package path for the generated asset or output.
   * @param Populate Callback that populates the new output.
   * @return The generated Map output.
   */
  static UWorld* BuildMap(const FString& PackagePath, TFunctionRef<void(FOutput&)> Populate);

  /**
   * Name of the prefab at SourcePath, derived from its file name.
   *
   * @param SourcePath Source prefab or scene file.
   * @return The prefab name.
   */
  static FString GetPrefabName(const FString& SourcePath);
  /**
   * Package folder a prefab or SuperDex scene dropped into FallbackFolder imports into:
   * FallbackFolder, or its source folder mirrored under the prefab root folder setting.
   *
   * @param SourcePath Source prefab or scene file.
   * @param FallbackFolder Destination folder used when organized import is disabled.
   * @return The destination folder.
   */
  static FString GetDestinationFolder(const FString& SourcePath, const FString& FallbackFolder);
  /**
   * Find the USuperDexPrefab asset of a prefab in its destination folder, or nullptr.
   *
   * @param DestFolder Destination Content Browser folder.
   * @param PrefabName Name of the prefab asset.
   * @return The prefab asset found in the destination folder.
   */
  static USuperDexPrefab* FindPrefabAsset(const FString& DestFolder, const FString& PrefabName);
  /**
   * Create the USuperDexPrefab asset of a prefab in its destination folder.
   *
   * @param DestFolder Destination Content Browser folder.
   * @param PrefabName Name of the prefab asset.
   * @return The object returned by creation.
   */
  static USuperDexPrefab* CreatePrefabAsset(const FString& DestFolder, const FString& PrefabName);
  /**
   * Hash of the prefab at SourcePath and every file it references, including nested prefabs.
   *
   * @param SourcePath Source prefab or scene file.
   * @return Hash of the source file and its referenced dependencies.
   */
  static FString ComputeSourceHash(const FString& SourcePath);
  /**
   * Whether every asset generated by the prefab's last import still exists.
   *
   * @param Prefab Prefab asset inspected for generated-asset presence.
   * @return Whether recorded generated assets and nested prefab outputs are present.
   */
  static bool AreGeneratedAssetsPresent(const USuperDexPrefab* Prefab);
  /**
   * Whether the prefab is up to date with its source and its generated assets exist.
   *
   * @param Prefab Prefab asset inspected for source freshness and generated-asset presence.
   * @return Whether the source hash matches and the generated assets are present.
   */
  static bool IsUpToDate(const USuperDexPrefab* Prefab);
  /**
   * Add the path of Asset relative to Folder to OutPaths, if Asset is in Folder. Assets elsewhere
   * belong to other imports.
   *
   * @param Folder Folder against which generated asset paths are recorded.
   * @param Asset Asset supplied to the operation.
   * @param OutPaths Generated-asset paths to extend with this asset.
   */
  static void
  RecordGeneratedAsset(const FString& Folder, const UObject* Asset, TArray<FString>& OutPaths);
  /**
   * Whether an asset other than CurrentOutput, the import's previous Blueprint or map, exists at
   * PackagePath. An import must not overwrite it, since it belongs to another import or the user.
   *
   * @param PackagePath Package path for the generated asset or output.
   * @param CurrentOutput Current generated output used in collision checks.
   * @return The is output taken state.
   */
  static bool IsOutputTaken(const FString& PackagePath, const FSoftObjectPath& CurrentOutput);

 private:
  /** State of the import of one prefab file. */
  struct FPrefabContext {
    USuperDexPrefab* Prefab = nullptr;
    FString PrefabName;
    FString SourcePath;
    FString RootDir;
    FString DestFolder;
    mochi::prefab::ScenePrefab Data;
    TArray<FString> GeneratedAssetPaths;
    TArray<UMochiContactParams*> ContactParams;
    TArray<UMochiMaterialParams*> MaterialParams;
    TArray<UMochiRodMaterialParams*> RodMaterialParams;

    /**
     * Generated asset name to what it was generated from, to keep same-named sources from
     * overwriting each other.
     */
    TMap<FString, FString> AssetSources;
  };

  /** Load the prefab at SourcePath, an absolute normalized path, into Context. */
  bool LoadContext(FPrefabContext& Context, const FString& SourcePath, const FString& DestFolder);
  FString ResolvePath(const FPrefabContext& Context, const mochi::DynamicString& Path) const;

  /** Name for an asset generated from SourcePath, made unique among this import's assets. */
  FString MakeAssetName(
      FPrefabContext& Context,
      const FString& Prefix,
      const FString& BaseName,
      const FString& SourcePath) const;

  /** Name for an asset that no other asset of this import may share. */
  FString MakeUniqueAssetName(
      FPrefabContext& Context,
      const FString& Prefix,
      const FString& BaseName) const;
  void RecordAsset(FPrefabContext& Context, const UObject* Asset) const;
  void RecordMeshAndMaterials(FPrefabContext& Context, const UStaticMesh* Mesh) const;

  UStaticMesh* ImportRenderModel(FPrefabContext& Context, const mochi::DynamicString& File);
  UMochiModel* ImportShape(FPrefabContext& Context, const mochi::DynamicString& File);
  template <typename TParams>
  TParams* FindOrCreateParams(
      FPrefabContext& Context,
      const TParams* Candidate,
      TArray<TParams*>& Existing,
      const FString& Prefix,
      const FString& OwnerName);
  UMochiContactParams* FindOrCreateContactParams(
      FPrefabContext& Context,
      const mochi::ContactParams& Params,
      const FString& OwnerName);
  UMochiMaterialParams* FindOrCreateMaterialParams(
      FPrefabContext& Context,
      const UMochiMaterialParams* Candidate,
      const FString& OwnerName);
  UMochiArticulation* ImportArticulation(
      FPrefabContext& Context,
      const mochi::prefab::ArticulatedActorPrefab& Actor,
      const FString& ActorName);

  void AddActors(
      FPrefabContext& Context,
      FOutput& Output,
      const TArray<USuperDexPrefab*>& NestedPrefabs);
  void AddRigidActor(
      FPrefabContext& Context,
      FOutput& Output,
      const mochi::prefab::RigidActorPrefab& Actor,
      const FString& ActorName);
  void AddSoftActor(
      FPrefabContext& Context,
      FOutput& Output,
      const mochi::prefab::SoftActorPrefab& Actor,
      const FString& ActorName);
  void AddRodActor(
      FPrefabContext& Context,
      FOutput& Output,
      const mochi::prefab::RodActorPrefab& Actor,
      const FString& ActorName);
  void AddArticulatedActor(
      FPrefabContext& Context,
      FOutput& Output,
      const mochi::prefab::ArticulatedActorPrefab& Actor,
      const FString& ActorName);
  void AddSceneSettings(FPrefabContext& Context, FOutput& Output);

  /** Add a constraint component per prefab constraint (SuperDexPrefabImporterConstraints.cpp). */
  void AddConstraints(FPrefabContext& Context, FOutput& Output);

  FSuperDexImportSession& Session;

  /** SuperDex Studio space (right-handed, X forward, Y left, Z up, meters) to Unreal. */
  FMochiToUnrealConverter M2U;

  /** Prefabs imported by this importer, by source path. */
  TMap<FString, USuperDexPrefab*> ImportedPrefabs;

  /** Prefabs currently being imported, to detect cycles. */
  TSet<FString> ImportStack;
};
