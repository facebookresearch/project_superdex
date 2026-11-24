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

constexpr const TCHAR* MochiAssetsPipelinePath =
    TEXT("/SuperDexPhysics/Editor/MochiAssetsPipeline.MochiAssetsPipeline");

// The asset at ObjectPath, loaded if it exists on disk, or null.
UObject* LoadExistingObject(const FString& ObjectPath) {
  if (UObject* Asset = FindObject<UObject>(nullptr, *ObjectPath)) {
    return Asset;
  }
  return FPackageName::DoesPackageExist(FPackageName::ObjectPathToPackageName(ObjectPath))
      ? LoadObject<UObject>(nullptr, *ObjectPath, nullptr, LOAD_NoWarn | LOAD_Quiet)
      : nullptr;
}

// The first source file of an existing asset's import data, or empty.
FString GetAssetSourceFile(const FString& ObjectPath) {
  UObject* Asset = LoadExistingObject(ObjectPath);
  if (!Asset) {
    return FString();
  }
  const FObjectProperty* Property =
      FindFProperty<FObjectProperty>(Asset->GetClass(), TEXT("AssetImportData"));
  const UAssetImportData* ImportData = Property
      ? Cast<UAssetImportData>(Property->GetObjectPropertyValue_InContainer(Asset))
      : nullptr;
  return ImportData ? ImportData->GetFirstFilename() : FString();
}

// The asset names a GLB's material or image may import as, in order of preference: its own, then
// prefixed with the GLB file name, then also suffixed with its content digest.
TArray<FString, TInlineAllocator<3>>
CandidateAssetNames(const FString& GlbPath, const FString& Name, const FString& Digest) {
  const FString Prefixed = AssetName(FPaths::GetBaseFilename(GlbPath) + TEXT("_") + Name);
  return {Name, Prefixed, AssetName(Prefixed + TEXT("_") + Digest.Left(8))};
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
  const FString SanitizedName = SanitizeForFileName(Name);
  return FChar::IsDigit(SanitizedName[0]) ? TEXT("_") + SanitizedName : SanitizedName;
}

FString NormalizePath(const FString& Path) {
  FString Normalized = FPaths::ConvertRelativePathToFull(Path);
  FPaths::NormalizeFilename(Normalized);
  return Normalized;
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
    const UInterchangeGenericAssetsPipeline* Pipeline =
        Cast<UInterchangeGenericAssetsPipeline>(FSoftObjectPath(MochiAssetsPipelinePath).TryLoad());
    if (!Pipeline) {
      UE_LOG(
          LogMochiPhysicsEditor,
          Error,
          TEXT("Failed to load Interchange pipeline %s, cannot import GLB meshes"),
          MochiAssetsPipelinePath);
      return nullptr;
    }
    // Left to search, the pipeline would bind a GLB material to any existing material of its name
    // and skip importing it, so changes to the GLB's materials would never reach Unreal.
    UInterchangeGenericAssetsPipeline* Copy = DuplicateObject(Pipeline, GetTransientPackage());
    Copy->MaterialPipeline->SearchLocation = EInterchangeMaterialSearchLocation::DoNotSearch;
    GlbAssetsPipeline.Reset(Copy);
    // The same, importing only the materials and textures.
    UInterchangeGenericAssetsPipeline* MaterialsOnly = DuplicateObject(Copy, GetTransientPackage());
    MaterialsOnly->MeshPipeline->bImportStaticMeshes = false;
    MaterialsOnly->MeshPipeline->bImportSkeletalMeshes = false;
    GlbMaterialsPipeline.Reset(MaterialsOnly);
  }

  // Name the GLB's named materials and images so they don't replace different ones of another GLB.
  const FString GlbKey = SuperDexImportUtils::NormalizePath(GlbPath);
  TStrongObjectPtr<USuperDexGlbAssetNamesPipeline> NamesPipeline(
      NewObject<USuperDexGlbAssetNamesPipeline>(GetTransientPackage()));
  TMap<FString, FString> Digests;
  if (!SuperDexGlbAssetDigests::Read(GlbPath, Digests)) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Warning,
        TEXT(
            "%s is not a readable .glb; its existing materials and textures are kept as they are."),
        *GlbPath);
  }
  TMap<FString, FString> AssetNames;
  // Existing materials and textures to rebuild, moved out of the way so Interchange, which never
  // overwrites an existing material or texture outside a reimport of that asset, makes new ones in
  // their place.
  TArray<TPair<FString, UObject*>> Displaced;
  for (const TPair<FString, FString>& Digest : Digests) {
    FString Name;
    for (const FString& Candidate : CandidateAssetNames(GlbPath, Digest.Key, Digest.Value)) {
      Name = Candidate;
      if (!HoldsOtherContent(AssetObjectPath(DestPackagePath, Name), Name, GlbKey, Digest.Value)) {
        break;
      }
    }
    if (Name != Digest.Key) {
      NamesPipeline->AssetRenames.Add(Digest.Key, Name);
      UE_LOG(
          LogMochiPhysicsEditor,
          Display,
          TEXT("%s: another GLB's %s differs, importing this GLB's as %s"),
          *GlbPath,
          *Digest.Key,
          *Name);
    }
    AssetNames.Add(Digest.Key, Name);
    const FString AssetPath = AssetObjectPath(DestPackagePath, Name);
    const FGlbAssetContent* Built = GlbAssetContents.Find(AssetPath);
    if (!Built || Built->Digest != Digest.Value) {
      if (UObject* Existing = LoadExistingObject(AssetPath)) {
        if (Existing->IsA<UMaterialInterface>() || Existing->IsA<UTexture>()) {
          Existing->Rename(
              *MakeUniqueObjectName(GetTransientPackage(), Existing->GetClass()).ToString(),
              GetTransientPackage(),
              REN_DontCreateRedirectors | REN_NonTransactional | REN_DoNotDirty);
          Displaced.Emplace(AssetPath, Existing);
        }
      }
    }
    GlbAssetContents.Add(AssetPath, {GlbKey, Digest.Value});
  }

  UInterchangeManager& InterchangeManager = UInterchangeManager::GetInterchangeManager();
  const FString MeshName = FString::Printf(TEXT("SM_%s"), *AssetName);
  const FString MeshPackage = FPaths::Combine(DestPackagePath, MeshName);
  const bool bMeshExists =
      FindPackage(nullptr, *MeshPackage) != nullptr || FPackageName::DoesPackageExist(MeshPackage);

  FImportAssetParameters ImportParams;
  ImportParams.bIsAutomated = true;
  ImportParams.bReplaceExisting = true;
  ImportParams.OverridePipelines.Add(FSoftObjectPath(GlbAssetsPipeline.Get()));
  ImportParams.OverridePipelines.Add(FSoftObjectPath(NamesPipeline.Get()));
  ImportParams.DestinationName = MeshName;

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
    MaterialParams.OverridePipelines.Add(FSoftObjectPath(NamesPipeline.Get()));
    MaterialParams.DestinationName = MeshName + TEXT("_GlbMaterials");
    TArray<UObject*> ImportedMaterials;
    InterchangeManager.ImportAsset(
        DestPackagePath,
        UInterchangeManager::CreateSourceData(GlbPath),
        MaterialParams,
        ImportedMaterials);
  }

  ReplaceDisplacedAssets(GlbPath, Displaced);

  // Point the mesh's material slots, named after the GLB's materials, at this import's materials:
  // a reimported mesh keeps the materials its slots had, which may have been renamed since.
  if (Result) {
    bool bChanged = false;
    for (FStaticMaterial& Slot : Result->GetStaticMaterials()) {
      const FString* Name = AssetNames.Find(
          SuperDexGlbAssetDigests::AssetName(Slot.ImportedMaterialSlotName.ToString()));
      UMaterialInterface* Material = Name ? LoadObject<UMaterialInterface>(
                                                nullptr,
                                                *AssetObjectPath(DestPackagePath, *Name),
                                                nullptr,
                                                LOAD_NoWarn | LOAD_Quiet)
                                          : nullptr;
      if (Material && Slot.MaterialInterface != Material) {
        Result->Modify();
        Slot.MaterialInterface = Material;
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
  if (const FGlbAssetContent* Content = GlbAssetContents.Find(ObjectPath)) {
    // Both must differ: another GLB's same content is shared, and a GLB's own record never
    // conflicts with it (its digests are keyed by asset name, so it has one entry per name).
    return Content->GlbPath != GlbPath && Content->Digest != Digest;
  }
  // An asset from an earlier import: compare with what its source GLB holds now. One whose source
  // is gone is stale and may be replaced.
  const FString SourcePath = GetAssetSourceFile(ObjectPath);
  if (SourcePath.IsEmpty() || !FPaths::FileExists(SourcePath) ||
      SuperDexImportUtils::NormalizePath(SourcePath) == GlbPath) {
    return false;
  }
  const FString SourceKey = SuperDexImportUtils::NormalizePath(SourcePath);
  if (!SourceGlbDigests.Contains(SourceKey)) {
    SuperDexGlbAssetDigests::Read(SourcePath, SourceGlbDigests.Add(SourceKey));
  }
  for (const TPair<FString, FString>& SourceDigest : SourceGlbDigests[SourceKey]) {
    if (CandidateAssetNames(SourcePath, SourceDigest.Key, SourceDigest.Value).Contains(Name)) {
      return SourceDigest.Value != Digest;
    }
  }
  return false;
}

void FSuperDexImportSession::ReplaceDisplacedAssets(
    const FString& GlbPath,
    const TArray<TPair<FString, UObject*>>& Displaced) {
  TArray<UObject*> Olds;
  TArray<UObject*> Rebuilts;
  for (const TPair<FString, UObject*>& Old : Displaced) {
    const FString Name = FPackageName::ObjectPathToObjectName(Old.Key);
    UPackage* Package = FindPackage(nullptr, *FPackageName::ObjectPathToPackageName(Old.Key));
    UObject* Rebuilt = Package ? StaticFindObject(nullptr, Package, *Name) : nullptr;
    if (!Rebuilt) {
      if (!Package) {
        Package = CreatePackage(*FPackageName::ObjectPathToPackageName(Old.Key));
      }
      Old.Value->Rename(
          *Name, Package, REN_DontCreateRedirectors | REN_NonTransactional | REN_DoNotDirty);
      UE_LOG(
          LogMochiPhysicsEditor,
          Warning,
          TEXT("%s: %s was not rebuilt and is kept as it was."),
          *GlbPath,
          *Old.Key);
      continue;
    }
    if (Rebuilt->GetClass() != Old.Value->GetClass()) {
      UE_LOG(
          LogMochiPhysicsEditor,
          Warning,
          TEXT("%s: %s was rebuilt as a %s instead of a %s; references to it were cleared."),
          *GlbPath,
          *Old.Key,
          *Rebuilt->GetClass()->GetName(),
          *Old.Value->GetClass()->GetName());
      Rebuilt = nullptr;
    }
    Olds.Add(Old.Value);
    Rebuilts.Add(Rebuilt);
  }
  if (Olds.IsEmpty()) {
    return;
  }
  // Replace through the editor, which tells every referencer of the change (Pre/PostEditChange, so
  // meshes, components and materials refresh their render state) before the old objects go away.
  // (This engine doesn't export the batched overload.)
  for (int32 i = 0; i < Olds.Num(); ++i) {
    TArray<UObject*> Old = {Olds[i]};
    ObjectTools::ForceReplaceReferences(Rebuilts[i], Old);
  }
  for (UObject* Old : Olds) {
    Old->ClearFlags(RF_Public | RF_Standalone);
    Old->MarkAsGarbage();
  }
}
