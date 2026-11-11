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

#include "SuperDexSceneFactory.h"

#include "AssetRegistry/AssetRegistryModule.h"
#include "Misc/App.h"
#include "Misc/Paths.h"
#include "Misc/ScopedSlowTask.h"
#include "SuperDexPrefabFactory.h"
#include "SuperDexPrefabImporter.h"
#include "SuperDexScene.h"
#include "SuperDexSceneImporter.h"

DEFINE_LOG_CATEGORY_STATIC(LogSuperDexSceneFactory, Log, All);

#define LOCTEXT_NAMESPACE "SuperDexSceneFactory"

///////////////////////////////////////////////////////////////////////////////
// USuperDexSceneFactory
///////////////////////////////////////////////////////////////////////////////

USuperDexSceneFactory::USuperDexSceneFactory(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  Formats.Add(FString(TEXT("superdex_scene;SuperDex Scene")));
  Formats.Add(FString(TEXT("superdex_scene_archive;SuperDex Scene Archive")));
  SupportedClass = USuperDexScene::StaticClass();
  bCreateNew = false;
  bEditorImport = true;
}

UObject* USuperDexSceneFactory::FactoryCreateFile(
    UClass* InClass,
    UObject* InParent,
    FName InName,
    EObjectFlags Flags,
    const FString& Filename,
    const TCHAR* Parms,
    FFeedbackContext* Warn,
    bool& bOutOperationCanceled) {
  bOutOperationCanceled = false;
  const FString SceneName = FSuperDexSceneImporter::GetSceneName(Filename);
  const FString DestFolder = FSuperDexPrefabImporter::GetDestinationFolder(
      Filename, FPackageName::GetLongPackagePath(InParent->GetPathName()));

  if (USuperDexScene* Existing = FSuperDexSceneImporter::FindSceneAsset(DestFolder, SceneName)) {
    UE_LOG(
        LogSuperDexSceneFactory,
        Display,
        TEXT("SuperDex scene %s already exists in %s, reimporting it from %s"),
        *SceneName,
        *DestFolder,
        *Filename);
    if (Existing->AssetImportData) {
      Existing->AssetImportData->UpdateFilenameOnly(Filename);
    }
    return FReimportManager::Instance()->Reimport(Existing, false, true) ? Existing : nullptr;
  }

  ESuperDexPrefabImportMode Mode = ESuperDexPrefabImportMode::Map;
  if (!IsAutomatedImport() && !FApp::IsUnattended() &&
      !PromptForSuperDexPrefabImportMode(
          LOCTEXT("ImportModeTitle", "Import SuperDex Scene"), SceneName, Mode)) {
    bOutOperationCanceled = true;
    return nullptr;
  }

  USuperDexScene* Scene = FSuperDexSceneImporter::CreateSceneAsset(DestFolder, SceneName);
  FScopedSlowTask SlowTask(
      0,
      FText::Format(
          LOCTEXT("ImportingScene", "Importing SuperDex scene {0}..."),
          FText::FromString(SceneName)));
  SlowTask.MakeDialog();
  FSuperDexSceneImporter Importer(ImportSession);
  if (!Importer.Import(Scene, Filename, Mode)) {
    // Discard the empty asset rather than leaving it behind.
    FAssetRegistryModule::AssetDeleted(Scene);
    Scene->ClearFlags(RF_Public | RF_Standalone);
    Scene->MarkAsGarbage();
    return nullptr;
  }
  return Scene;
}

void USuperDexSceneFactory::CleanUp() {
  Super::CleanUp();
  ImportSession.Reset();
}

///////////////////////////////////////////////////////////////////////////////
// UReimportSuperDexSceneFactory
///////////////////////////////////////////////////////////////////////////////

UReimportSuperDexSceneFactory::UReimportSuperDexSceneFactory() {
  ImportPriority = DefaultImportPriority - 1;
}

bool UReimportSuperDexSceneFactory::CanReimport(UObject* Obj, TArray<FString>& OutFilenames) {
  USuperDexScene* Scene = Cast<USuperDexScene>(Obj);
  if (!Scene) {
    return false;
  }
  if (Scene->AssetImportData) {
    Scene->AssetImportData->ExtractFilenames(OutFilenames);
  }
  return true;
}

void UReimportSuperDexSceneFactory::SetReimportPaths(
    UObject* Obj,
    const TArray<FString>& NewReimportPaths) {
  USuperDexScene* Scene = Cast<USuperDexScene>(Obj);
  if (Scene && Scene->AssetImportData && NewReimportPaths.Num() > 0) {
    Scene->AssetImportData->UpdateFilenameOnly(NewReimportPaths[0]);
  }
}

EReimportResult::Type UReimportSuperDexSceneFactory::Reimport(UObject* Obj) {
  USuperDexScene* Scene = Cast<USuperDexScene>(Obj);
  if (!Scene) {
    return EReimportResult::Failed;
  }
  const FString SourcePath = Scene->GetSourceFilePath();
  if (SourcePath.IsEmpty() || !FPaths::FileExists(SourcePath)) {
    UE_LOG(LogSuperDexSceneFactory, Error, TEXT("Source file not found: %s"), *SourcePath);
    return EReimportResult::Failed;
  }
  if (FSuperDexSceneImporter::IsUpToDate(Scene)) {
    UE_LOG(
        LogSuperDexSceneFactory,
        Display,
        TEXT("SuperDex scene %s is already up to date, skipping reimport"),
        *Scene->GetName());
    return EReimportResult::Succeeded;
  }
  FScopedSlowTask SlowTask(
      0,
      FText::Format(
          LOCTEXT("ReimportingScene", "Reimporting SuperDex scene {0}..."),
          FText::FromString(Scene->SceneName)));
  SlowTask.MakeDialog();
  ImportSession.Reset();
  FSuperDexSceneImporter Importer(ImportSession);
  return Importer.Import(Scene, SourcePath, Scene->ImportMode) ? EReimportResult::Succeeded
                                                               : EReimportResult::Failed;
}

int32 UReimportSuperDexSceneFactory::GetPriority() const {
  return ImportPriority;
}

#undef LOCTEXT_NAMESPACE
