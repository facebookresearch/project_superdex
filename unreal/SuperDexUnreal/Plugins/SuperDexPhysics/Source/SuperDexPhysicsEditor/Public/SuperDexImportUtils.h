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
#include "UObject/StrongObjectPtr.h"

class UMochiModel;
class UStaticMesh;

// Helpers shared by the SuperDex Studio asset importers (bots, prefabs, SuperDex scenes).
namespace SuperDexImportUtils {

// Replace '.' and ' ' with '_'. Other characters are kept, so the result is not guaranteed to be
// a valid package name.
SUPERDEXPHYSICSEDITOR_API FString SanitizeForPackagePath(const FString& InStr);

// Replace every character other than alphanumerics, '_' and '-' with '_'. Returns "link" for an
// empty name.
SUPERDEXPHYSICSEDITOR_API FString SanitizeForFileName(const FString& InStr);

// Name of the component an importer generates for something named Name (e.g. a prefab actor),
// before it is made unique: SanitizeForFileName, with a leading '_' if it starts with a digit.
SUPERDEXPHYSICSEDITOR_API FString SanitizeForComponentName(const FString& Name);

// Absolute form of a file path with '/' separators, to compare and key source paths by.
SUPERDEXPHYSICSEDITOR_API FString NormalizePath(const FString& Path);

// Asset base name of a .mochi.h5 file: both extensions stripped, sanitized for package paths.
SUPERDEXPHYSICSEDITOR_API FString GetMochiModelBaseName(const FString& H5Path);

// Walk up the directory tree from StartPath looking for a .superdex_root marker file. Returns
// the directory containing it, or empty if none was found.
SUPERDEXPHYSICSEDITOR_API FString FindSuperDexRootDir(const FString& StartPath);

// Map a source file to <RootFolder>/<source dir relative to the .superdex_root dir>. Returns empty
// if the source file is not beneath a .superdex_root dir.
SUPERDEXPHYSICSEDITOR_API FString
ComputeDestinationPackagePath(const FString& SourceFilePath, const FString& RootFolder);

// Folder that a file referenced by an import into ParentFolder (e.g. a nested prefab or a bot of a
// SuperDex scene) imports into: <ParentFolder>/<source file name>.
SUPERDEXPHYSICSEDITOR_API FString
GetNestedImportFolder(const FString& ParentFolder, const FString& SourceFilePath);

} // namespace SuperDexImportUtils

// Imports the render (.glb) and collision (.mochi.h5) files referenced by SuperDex Studio assets.
// Results are cached by source path so that one import (including recursive ones, e.g. the bots
// and prefabs of a SuperDex scene) imports each file only once: a file referenced again returns the
// asset of its first import, wherever that was. The cache does not keep assets alive, so a session
// should not outlive the import it serves.
class SUPERDEXPHYSICSEDITOR_API FSuperDexImportSession {
 public:
  // Import a .glb as SM_<AssetName> under DestPackagePath, replacing any existing asset.
  //
  // Its materials and textures are named after the GLB's material and image names. Those of named
  // materials and images are rebuilt from the GLB on every import, replacing existing assets of
  // those names, also when the mesh already exists (Interchange reimports an existing mesh without
  // its materials); unnamed ones keep an existing asset as it is. GLBs whose same-named materials
  // or images have the same content share one asset. If an asset of that name holds a different
  // GLB's different content (imported earlier in this session, or found on disk with that GLB as
  // its source), this GLB's asset is named <GLB file name>_<name>, or if that is taken too,
  // <GLB file name>_<name>_<digest prefix>.
  UStaticMesh* ImportGlbToStaticMesh(
      const FString& GlbPath,
      const FString& DestPackagePath,
      const FString& AssetName);

  // Import a .mochi.h5 as MM_<AssetName> under DestPackagePath. An existing asset is reimported in
  // place so references to it from other assets are preserved.
  UMochiModel* ImportH5ToMochiModel(
      const FString& H5Path,
      const FString& DestPackagePath,
      const FString& AssetName);

  void Reset();

 private:
  // Content of a material or texture asset imported in this session.
  struct FGlbAssetContent {
    FString GlbPath; // Normalized.
    FString Digest;
  };

  // Whether the asset ObjectPath, named Name, holds content other than Digest from a GLB other than
  // the one at the normalized path GlbPath.
  bool HoldsOtherContent(
      const FString& ObjectPath,
      const FString& Name,
      const FString& GlbPath,
      const FString& Digest);

  // Replace every reference to each displaced asset (by the object path it was moved from) with
  // the asset rebuilt at that path, then discard it; put back those that were not rebuilt.
  static void ReplaceDisplacedAssets(
      const FString& GlbPath,
      const TArray<TPair<FString, UObject*>>& Displaced);

  TMap<FString, UStaticMesh*> StaticMeshCache;
  TMap<FString, UMochiModel*> MochiModelCache;
  // By object path.
  TMap<FString, FGlbAssetContent> GlbAssetContents;
  // Material and image digests of GLBs that earlier imports' assets came from, by normalized path.
  TMap<FString, TMap<FString, FString>> SourceGlbDigests;
  // The Mochi assets pipeline, set not to reuse existing materials, and a copy of it that imports
  // only materials and textures.
  TStrongObjectPtr<UObject> GlbAssetsPipeline;
  TStrongObjectPtr<UObject> GlbMaterialsPipeline;
};
