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
#include "Engine/StaticMesh.h"
#include "HAL/FileManager.h"
#include "InterchangeGenericAssetsPipeline.h"
#include "InterchangeManager.h"
#include "Misc/Paths.h"
#include "MochiModel.h"
#include "MochiPhysicsEditorModule.h"

namespace {

constexpr const TCHAR* kMochiAssetsPipelinePath =
    TEXT("/SuperDexPhysics/Editor/MochiAssetsPipeline.MochiAssetsPipeline");

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

  const FSoftObjectPath MochiAssetPipelinePath(kMochiAssetsPipelinePath);
  if (!Cast<UInterchangeGenericAssetsPipeline>(MochiAssetPipelinePath.TryLoad())) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Error,
        TEXT("Failed to load Interchange pipeline %s, cannot import GLB meshes"),
        kMochiAssetsPipelinePath);
    return nullptr;
  }

  UInterchangeManager& InterchangeManager = UInterchangeManager::GetInterchangeManager();
  UInterchangeSourceData* SourceData = UInterchangeManager::CreateSourceData(GlbPath);

  FImportAssetParameters ImportParams;
  ImportParams.bIsAutomated = true;
  ImportParams.bReplaceExisting = true;
  ImportParams.OverridePipelines.Add(MochiAssetPipelinePath);
  ImportParams.DestinationName = FString::Printf(TEXT("SM_%s"), *AssetName);

  TArray<UObject*> ImportedObjects;
  InterchangeManager.ImportAsset(DestPackagePath, SourceData, ImportParams, ImportedObjects);

  UStaticMesh* Result = nullptr;
  for (UObject* Obj : ImportedObjects) {
    if (UStaticMesh* Mesh = Cast<UStaticMesh>(Obj)) {
      Result = Mesh;
      break;
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
}
