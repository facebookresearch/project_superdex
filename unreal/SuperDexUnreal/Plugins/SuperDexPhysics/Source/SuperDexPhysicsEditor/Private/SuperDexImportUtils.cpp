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

#include "SuperDexImportUtils.h"

#include "AssetRegistry/AssetRegistryModule.h"
#include "EditorFramework/AssetImportData.h"
#include "Engine/StaticMesh.h"
#include "Engine/Texture.h"
#include "HAL/FileManager.h"
#include "InterchangeGenericAssetsPipeline.h"
#include "InterchangeGenericMaterialPipeline.h"
#include "InterchangeGenericMeshPipeline.h"
#include "InterchangeManager.h"
#include "Materials/MaterialInterface.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "MochiModel.h"
#include "MochiPhysicsEditorModule.h"
#include "ObjectTools.h"
#include "SuperDexGlbAssetDigests.h"
#include "SuperDexGlbAssetNamesPipeline.h"

namespace {

using SuperDexGlbAssetDigests::AssetName;
using SuperDexGlbAssetDigests::AssetObjectPath;

constexpr const TCHAR* kMochiAssetsPipelinePath =
    TEXT("/SuperDexPhysics/Editor/MochiAssetsPipeline.MochiAssetsPipeline");

// The asset at ObjectPath, loaded if it exists on disk, or null.
UObject* LoadExistingObject(const FString& ObjectPath) {
  if (UObject* asset = FindObject<UObject>(nullptr, *ObjectPath)) {
    return asset;
  }
  return FPackageName::DoesPackageExist(FPackageName::ObjectPathToPackageName(ObjectPath))
      ? LoadObject<UObject>(nullptr, *ObjectPath, nullptr, LOAD_NoWarn | LOAD_Quiet)
      : nullptr;
}

// The first source file of an existing asset's import data, or empty.
FString GetAssetSourceFile(const FString& ObjectPath) {
  UObject* asset = LoadExistingObject(ObjectPath);
  if (!asset) {
    return FString();
  }
  const FObjectProperty* property =
      FindFProperty<FObjectProperty>(asset->GetClass(), TEXT("AssetImportData"));
  const UAssetImportData* importData = property
      ? Cast<UAssetImportData>(property->GetObjectPropertyValue_InContainer(asset))
      : nullptr;
  return importData ? importData->GetFirstFilename() : FString();
}

// The asset names a GLB's material or image may import as, in order of preference: its own, then
// prefixed with the GLB file name, then also suffixed with its content digest.
TArray<FString, TInlineAllocator<3>>
CandidateAssetNames(const FString& GlbPath, const FString& Name, const FString& Digest) {
  const FString prefixed = AssetName(FPaths::GetBaseFilename(GlbPath) + TEXT("_") + Name);
  return {Name, prefixed, AssetName(prefixed + TEXT("_") + Digest.Left(8))};
}

} // namespace

namespace SuperDexImportUtils {

FString SanitizeForPackagePath(const FString& InStr) {
  FString Result = InStr;
  Result.ReplaceCharInline(TEXT('.'), TEXT('_'));
  Result.ReplaceCharInline(TEXT(' '), TEXT('_'));
  return Result;
}

FString SanitizeForFileName(const FString& InStr) {
  FString Result;
  Result.Reserve(InStr.Len());
  for (const TCHAR C : InStr) {
    Result.AppendChar(FChar::IsAlnum(C) || C == TEXT('_') || C == TEXT('-') ? C : TEXT('_'));
  }
  return Result.IsEmpty() ? FString(TEXT("link")) : Result;
}

FString SanitizeForComponentName(const FString& Name) {
  const FString name = SanitizeForFileName(Name);
  return FChar::IsDigit(name[0]) ? TEXT("_") + name : name;
}

FString NormalizePath(const FString& Path) {
  FString normalized = FPaths::ConvertRelativePathToFull(Path);
  FPaths::NormalizeFilename(normalized);
  return normalized;
}

FString GetMochiModelBaseName(const FString& H5Path) {
  // GetBaseFilename twice to strip the .mochi.h5 double extension.
  return SanitizeForPackagePath(FPaths::GetBaseFilename(FPaths::GetBaseFilename(H5Path)));
}

// Pure FString/FPaths implementation - avoids std::filesystem entirely
// (its wchar_t internal strings crash when freed through UE's mimalloc on Windows).
FString FindSuperDexRootDir(const FString& StartPath) {
  FString SearchDir = FPaths::GetPath(StartPath);
  while (!SearchDir.IsEmpty()) {
    FString MarkerPath = FPaths::Combine(SearchDir, TEXT(".superdex_root"));
    if (IFileManager::Get().FileExists(*MarkerPath)) {
      return SearchDir;
    }
    FString ParentDir = FPaths::GetPath(SearchDir);
    if (ParentDir == SearchDir) {
      break;
    }
    SearchDir = ParentDir;
  }
  return FString();
}

FString ComputeDestinationPackagePath(const FString& SourceFilePath, const FString& RootFolder) {
  FString RootDir = FindSuperDexRootDir(SourceFilePath);
  if (RootDir.IsEmpty()) {
    return FString();
  }

  FString SourceDir = FPaths::GetPath(SourceFilePath);
  FPaths::NormalizeDirectoryName(RootDir);
  FPaths::NormalizeDirectoryName(SourceDir);

  if (!SourceDir.StartsWith(RootDir)) {
    return FString();
  }

  FString RelPath = SourceDir.Mid(RootDir.Len());
  RelPath.RemoveFromStart(TEXT("/"));
  return FPaths::Combine(RootFolder, SanitizeForPackagePath(RelPath));
}

FString GetNestedImportFolder(const FString& ParentFolder, const FString& SourceFilePath) {
  return FPaths::Combine(
      ParentFolder, SanitizeForPackagePath(FPaths::GetBaseFilename(SourceFilePath)));
}

} // namespace SuperDexImportUtils

UStaticMesh* FSuperDexImportSession::ImportGlbToStaticMesh(
    const FString& GlbPath,
    const FString& DestPackagePath,
    const FString& AssetName) {
  if (UStaticMesh** Found = StaticMeshCache.Find(GlbPath)) {
    return *Found;
  }

  if (!FPaths::FileExists(GlbPath)) {
    UE_LOG(LogMochiPhysicsEditor, Warning, TEXT("Render mesh file not found: %s"), *GlbPath);
    return nullptr;
  }

  if (!GlbAssetsPipeline) {
    const UInterchangeGenericAssetsPipeline* pipeline = Cast<UInterchangeGenericAssetsPipeline>(
        FSoftObjectPath(kMochiAssetsPipelinePath).TryLoad());
    if (!pipeline) {
      UE_LOG(
          LogMochiPhysicsEditor,
          Error,
          TEXT("Failed to load Interchange pipeline %s, cannot import GLB meshes"),
          kMochiAssetsPipelinePath);
      return nullptr;
    }
    // Left to search, the pipeline would bind a GLB material to any existing material of its name
    // and skip importing it, so changes to the GLB's materials would never reach Unreal.
    UInterchangeGenericAssetsPipeline* copy = DuplicateObject(pipeline, GetTransientPackage());
    copy->MaterialPipeline->SearchLocation = EInterchangeMaterialSearchLocation::DoNotSearch;
    GlbAssetsPipeline.Reset(copy);
    // The same, importing only the materials and textures.
    UInterchangeGenericAssetsPipeline* materialsOnly = DuplicateObject(copy, GetTransientPackage());
    materialsOnly->MeshPipeline->bImportStaticMeshes = false;
    materialsOnly->MeshPipeline->bImportSkeletalMeshes = false;
    GlbMaterialsPipeline.Reset(materialsOnly);
  }

  // Name the GLB's named materials and images so they don't replace different ones of another GLB.
  const FString glbKey = SuperDexImportUtils::NormalizePath(GlbPath);
  TStrongObjectPtr<USuperDexGlbAssetNamesPipeline> namesPipeline(
      NewObject<USuperDexGlbAssetNamesPipeline>(GetTransientPackage()));
  TMap<FString, FString> digests;
  if (!SuperDexGlbAssetDigests::Read(GlbPath, digests)) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Warning,
        TEXT(
            "%s is not a readable .glb; its existing materials and textures are kept as they are."),
        *GlbPath);
  }
  TMap<FString, FString> assetNames;
  // Existing materials and textures to rebuild, moved out of the way so Interchange, which never
  // overwrites an existing material or texture outside a reimport of that asset, makes new ones in
  // their place.
  TArray<TPair<FString, UObject*>> displaced;
  for (const TPair<FString, FString>& digest : digests) {
    FString name;
    for (const FString& candidate : CandidateAssetNames(GlbPath, digest.Key, digest.Value)) {
      name = candidate;
      if (!HoldsOtherContent(AssetObjectPath(DestPackagePath, name), name, glbKey, digest.Value)) {
        break;
      }
    }
    if (name != digest.Key) {
      namesPipeline->AssetRenames.Add(digest.Key, name);
      UE_LOG(
          LogMochiPhysicsEditor,
          Display,
          TEXT("%s: another GLB's %s differs, importing this GLB's as %s"),
          *GlbPath,
          *digest.Key,
          *name);
    }
    assetNames.Add(digest.Key, name);
    const FString assetPath = AssetObjectPath(DestPackagePath, name);
    const FGlbAssetContent* built = GlbAssetContents.Find(assetPath);
    if (!built || built->Digest != digest.Value) {
      if (UObject* existing = LoadExistingObject(assetPath)) {
        if (existing->IsA<UMaterialInterface>() || existing->IsA<UTexture>()) {
          existing->Rename(
              *MakeUniqueObjectName(GetTransientPackage(), existing->GetClass()).ToString(),
              GetTransientPackage(),
              REN_DontCreateRedirectors | REN_NonTransactional | REN_DoNotDirty);
          displaced.Emplace(assetPath, existing);
        }
      }
    }
    GlbAssetContents.Add(assetPath, {glbKey, digest.Value});
  }

  UInterchangeManager& InterchangeManager = UInterchangeManager::GetInterchangeManager();
  const FString meshName = FString::Printf(TEXT("SM_%s"), *AssetName);
  const FString meshPackage = FPaths::Combine(DestPackagePath, meshName);
  const bool bMeshExists =
      FindPackage(nullptr, *meshPackage) != nullptr || FPackageName::DoesPackageExist(meshPackage);

  FImportAssetParameters ImportParams;
  ImportParams.bIsAutomated = true;
  ImportParams.bReplaceExisting = true;
  ImportParams.OverridePipelines.Add(FSoftObjectPath(GlbAssetsPipeline.Get()));
  ImportParams.OverridePipelines.Add(FSoftObjectPath(namesPipeline.Get()));
  ImportParams.DestinationName = meshName;

  TArray<UObject*> ImportedObjects;
  InterchangeManager.ImportAsset(
      DestPackagePath,
      UInterchangeManager::CreateSourceData(GlbPath),
      ImportParams,
      ImportedObjects);

  UStaticMesh* Result = nullptr;
  for (UObject* Obj : ImportedObjects) {
    if (UStaticMesh* Mesh = Cast<UStaticMesh>(Obj)) {
      Result = Mesh;
      break;
    }
  }

  if (bMeshExists) {
    // Interchange turns an import over an existing mesh into a reimport of only the mesh, which
    // keeps its materials and textures as they were. Import those again on their own, under a
    // name no asset has: an import over an existing asset (by default, one named after the GLB
    // file) would become a reimport of that asset alone.
    FImportAssetParameters MaterialParams;
    MaterialParams.bIsAutomated = true;
    MaterialParams.bReplaceExisting = true;
    MaterialParams.OverridePipelines.Add(FSoftObjectPath(GlbMaterialsPipeline.Get()));
    MaterialParams.OverridePipelines.Add(FSoftObjectPath(namesPipeline.Get()));
    MaterialParams.DestinationName = meshName + TEXT("_GlbMaterials");
    TArray<UObject*> ImportedMaterials;
    InterchangeManager.ImportAsset(
        DestPackagePath,
        UInterchangeManager::CreateSourceData(GlbPath),
        MaterialParams,
        ImportedMaterials);
  }

  ReplaceDisplacedAssets(GlbPath, displaced);

  // Point the mesh's material slots, named after the GLB's materials, at this import's materials:
  // a reimported mesh keeps the materials its slots had, which may have been renamed since.
  if (Result) {
    bool bChanged = false;
    for (FStaticMaterial& slot : Result->GetStaticMaterials()) {
      const FString* name = assetNames.Find(
          SuperDexGlbAssetDigests::AssetName(slot.ImportedMaterialSlotName.ToString()));
      UMaterialInterface* material = name ? LoadObject<UMaterialInterface>(
                                                nullptr,
                                                *AssetObjectPath(DestPackagePath, *name),
                                                nullptr,
                                                LOAD_NoWarn | LOAD_Quiet)
                                          : nullptr;
      if (material && slot.MaterialInterface != material) {
        Result->Modify();
        slot.MaterialInterface = material;
        bChanged = true;
      }
    }
    if (bChanged) {
      Result->PostEditChange();
    }
  }

  StaticMeshCache.Add(GlbPath, Result);
  return Result;
}

UMochiModel* FSuperDexImportSession::ImportH5ToMochiModel(
    const FString& H5Path,
    const FString& DestPackagePath,
    const FString& AssetName) {
  if (UMochiModel** Found = MochiModelCache.Find(H5Path)) {
    return *Found;
  }

  if (!FPaths::FileExists(H5Path)) {
    UE_LOG(LogMochiPhysicsEditor, Warning, TEXT("MochiModel file not found: %s"), *H5Path);
    return nullptr;
  }

  FString ModelAssetName = FString::Printf(TEXT("MM_%s"), *AssetName);
  FString FullAssetPath = FPaths::Combine(DestPackagePath, ModelAssetName);

  // Reuse an existing MochiModel asset at this path if one is present, updating it
  // in place rather than constructing a new UObject. A shared asset (e.g. one
  // already imported by another bot in the same folder) may be referenced by other
  // bots and their skeletal meshes; replacing the object would break those
  // references. Re-importing into the existing object preserves them.
  UMochiModel* Model = LoadObject<UMochiModel>(nullptr, *FullAssetPath);
  const bool bCreatedNewAsset = (Model == nullptr);
  if (bCreatedNewAsset) {
    UPackage* Package = CreatePackage(*FullAssetPath);
    Model = NewObject<UMochiModel>(
        Package, UMochiModel::StaticClass(), FName(*ModelAssetName), RF_Public | RF_Standalone);
  }

  if (!Model->ImportFromFilename(H5Path)) {
    if (bCreatedNewAsset) {
      // The new asset was never registered (AssetCreated only runs on success), so
      // discard the empty, half-constructed object rather than leaving it dangling.
      Model->ClearFlags(RF_Public | RF_Standalone);
      Model->MarkAsGarbage();
      UE_LOG(
          LogMochiPhysicsEditor, Error, TEXT("Failed to import new MochiModel from: %s"), *H5Path);
    } else {
      // Reimport over an existing (possibly shared) asset failed partway through. We
      // cannot cleanly revert its contents, so surface this loudly: the asset may be
      // left in an inconsistent state and should be reimported once the source is fixed.
      UE_LOG(
          LogMochiPhysicsEditor,
          Error,
          TEXT(
              "Failed to reimport existing MochiModel '%s' from: %s; it may be left in an "
              "inconsistent state and should be reimported."),
          *FullAssetPath,
          *H5Path);
    }
    return nullptr;
  }

  // SuperDex Studio files are in its right-handed X-forward, Y-left, Z-up frame; flip Y into
  // Unreal's left-handed frame.
  FTransform ImportTransform(FRotator::ZeroRotator, FVector::ZeroVector, FVector(1.0, -1.0, 1.0));
  Model->SetImportTransform(ImportTransform);

  if (bCreatedNewAsset) {
    FAssetRegistryModule::AssetCreated(Model);
  }
  Model->MarkPackageDirty();

  MochiModelCache.Add(H5Path, Model);
  return Model;
}

void FSuperDexImportSession::Reset() {
  StaticMeshCache.Empty();
  MochiModelCache.Empty();
  GlbAssetContents.Empty();
  SourceGlbDigests.Empty();
  GlbAssetsPipeline.Reset();
  GlbMaterialsPipeline.Reset();
}

bool FSuperDexImportSession::HoldsOtherContent(
    const FString& ObjectPath,
    const FString& Name,
    const FString& GlbPath,
    const FString& Digest) {
  if (const FGlbAssetContent* content = GlbAssetContents.Find(ObjectPath)) {
    // Both must differ: another GLB's same content is shared, and a GLB's own record never
    // conflicts with it (its digests are keyed by asset name, so it has one entry per name).
    return content->GlbPath != GlbPath && content->Digest != Digest;
  }
  // An asset from an earlier import: compare with what its source GLB holds now. One whose source
  // is gone is stale and may be replaced.
  const FString sourcePath = GetAssetSourceFile(ObjectPath);
  if (sourcePath.IsEmpty() || !FPaths::FileExists(sourcePath) ||
      SuperDexImportUtils::NormalizePath(sourcePath) == GlbPath) {
    return false;
  }
  const FString sourceKey = SuperDexImportUtils::NormalizePath(sourcePath);
  if (!SourceGlbDigests.Contains(sourceKey)) {
    SuperDexGlbAssetDigests::Read(sourcePath, SourceGlbDigests.Add(sourceKey));
  }
  for (const TPair<FString, FString>& sourceDigest : SourceGlbDigests[sourceKey]) {
    if (CandidateAssetNames(sourcePath, sourceDigest.Key, sourceDigest.Value).Contains(Name)) {
      return sourceDigest.Value != Digest;
    }
  }
  return false;
}

void FSuperDexImportSession::ReplaceDisplacedAssets(
    const FString& GlbPath,
    const TArray<TPair<FString, UObject*>>& Displaced) {
  TArray<UObject*> olds;
  TArray<UObject*> rebuilts;
  for (const TPair<FString, UObject*>& old : Displaced) {
    const FString name = FPackageName::ObjectPathToObjectName(old.Key);
    UPackage* package = FindPackage(nullptr, *FPackageName::ObjectPathToPackageName(old.Key));
    UObject* rebuilt = package ? StaticFindObject(nullptr, package, *name) : nullptr;
    if (!rebuilt) {
      if (!package) {
        package = CreatePackage(*FPackageName::ObjectPathToPackageName(old.Key));
      }
      old.Value->Rename(
          *name, package, REN_DontCreateRedirectors | REN_NonTransactional | REN_DoNotDirty);
      UE_LOG(
          LogMochiPhysicsEditor,
          Warning,
          TEXT("%s: %s was not rebuilt and is kept as it was."),
          *GlbPath,
          *old.Key);
      continue;
    }
    if (rebuilt->GetClass() != old.Value->GetClass()) {
      UE_LOG(
          LogMochiPhysicsEditor,
          Warning,
          TEXT("%s: %s was rebuilt as a %s instead of a %s; references to it were cleared."),
          *GlbPath,
          *old.Key,
          *rebuilt->GetClass()->GetName(),
          *old.Value->GetClass()->GetName());
      rebuilt = nullptr;
    }
    olds.Add(old.Value);
    rebuilts.Add(rebuilt);
  }
  if (olds.IsEmpty()) {
    return;
  }
  // Replace through the editor, which tells every referencer of the change (Pre/PostEditChange, so
  // meshes, components and materials refresh their render state) before the old objects go away.
  // (This engine doesn't export the batched overload.)
  for (int32 i = 0; i < olds.Num(); ++i) {
    TArray<UObject*> old = {olds[i]};
    ObjectTools::ForceReplaceReferences(rebuilts[i], old);
  }
  for (UObject* old : olds) {
    old->ClearFlags(RF_Public | RF_Standalone);
    old->MarkAsGarbage();
  }
}
