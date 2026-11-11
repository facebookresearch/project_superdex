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

#include "SuperDexPrefabFactory.h"

#include "AssetRegistry/AssetRegistryModule.h"
#include "Dialog/SCustomDialog.h"
#include "Misc/App.h"
#include "Misc/Paths.h"
#include "Misc/ScopedSlowTask.h"
#include "MochiPhysicsEditorModule.h"
#include "SuperDexPrefab.h"
#include "SuperDexPrefabImporter.h"
#include "Widgets/Text/STextBlock.h"

#define LOCTEXT_NAMESPACE "SuperDexPrefabFactory"

bool PromptForSuperDexPrefabImportMode(
    const FText& Title,
    const FString& AssetName,
    ESuperDexPrefabImportMode& OutMode) {
  constexpr int32 kBlueprintButton = 0;
  constexpr int32 kMapButton = 1;
  TSharedRef<SCustomDialog> dialog =
      SNew(SCustomDialog)
          .Title(Title)
          .Content()[SNew(STextBlock)
                         .Text(
                             FText::Format(
                                 LOCTEXT(
                                     "ImportModeMessage",
                                     "Import {0} as an actor Blueprint or as a Map?"),
                                 FText::FromString(AssetName)))]
          .Buttons(
              {SCustomDialog::FButton(LOCTEXT("ImportAsBlueprint", "Blueprint")),
               SCustomDialog::FButton(LOCTEXT("ImportAsMap", "Map")),
               SCustomDialog::FButton(LOCTEXT("ImportCancel", "Cancel"))});
  const int32 choice = dialog->ShowModal();
  if (choice != kBlueprintButton && choice != kMapButton) {
    return false;
  }
  OutMode = choice == kBlueprintButton ? ESuperDexPrefabImportMode::Blueprint
                                       : ESuperDexPrefabImportMode::Map;
  return true;
}

//------------------------------------------------------------------------------
// USuperDexPrefabFactory
//------------------------------------------------------------------------------

USuperDexPrefabFactory::USuperDexPrefabFactory(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  Formats.Add(FString(TEXT("superdex_prefab;SuperDex Prefab")));
  Formats.Add(FString(TEXT("mochi_prefab;SuperDex Prefab")));
  Formats.Add(FString(TEXT("mochi_scene;Mochi Scene")));
  SupportedClass = USuperDexPrefab::StaticClass();
  bCreateNew = false;
  bEditorImport = true;
}

UObject* USuperDexPrefabFactory::FactoryCreateFile(
    UClass* InClass,
    UObject* InParent,
    FName InName,
    EObjectFlags Flags,
    const FString& Filename,
    const TCHAR* Parms,
    FFeedbackContext* Warn,
    bool& bOutOperationCanceled) {
  bOutOperationCanceled = false;
  const FString prefabName = FSuperDexPrefabImporter::GetPrefabName(Filename);
  const FString destFolder = FSuperDexPrefabImporter::GetDestinationFolder(
      Filename, FPackageName::GetLongPackagePath(InParent->GetPathName()));

  if (USuperDexPrefab* existing =
          FSuperDexPrefabImporter::FindPrefabAsset(destFolder, prefabName)) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Display,
        TEXT("SuperDex prefab %s already exists in %s, reimporting it from %s"),
        *prefabName,
        *destFolder,
        *Filename);
    if (existing->AssetImportData) {
      existing->AssetImportData->UpdateFilenameOnly(Filename);
    }
    return FReimportManager::Instance()->Reimport(existing, false, true) ? existing : nullptr;
  }

  ESuperDexPrefabImportMode mode = Filename.EndsWith(TEXT(".mochi_scene"))
      ? ESuperDexPrefabImportMode::Map
      : ESuperDexPrefabImportMode::Blueprint;
  if (!IsAutomatedImport() && !FApp::IsUnattended() &&
      !PromptForSuperDexPrefabImportMode(
          LOCTEXT("ImportModeTitle", "Import SuperDex Prefab"), prefabName, mode)) {
    bOutOperationCanceled = true;
    return nullptr;
  }

  USuperDexPrefab* prefab = FSuperDexPrefabImporter::CreatePrefabAsset(destFolder, prefabName);

  FScopedSlowTask slowTask(
      0,
      FText::Format(
          LOCTEXT("ImportingPrefab", "Importing SuperDex prefab {0}..."),
          FText::FromString(prefabName)));
  slowTask.MakeDialog();
  FSuperDexPrefabImporter importer(ImportSession);
  if (!importer.Import(prefab, Filename, mode)) {
    // Discard the empty asset rather than leaving it behind.
    FAssetRegistryModule::AssetDeleted(prefab);
    prefab->ClearFlags(RF_Public | RF_Standalone);
    prefab->MarkAsGarbage();
    return nullptr;
  }
  return prefab;
}

void USuperDexPrefabFactory::CleanUp() {
  Super::CleanUp();
  ImportSession.Reset();
}

//------------------------------------------------------------------------------
// UReimportSuperDexPrefabFactory
//------------------------------------------------------------------------------

UReimportSuperDexPrefabFactory::UReimportSuperDexPrefabFactory() {
  ImportPriority = DefaultImportPriority - 1;
}

bool UReimportSuperDexPrefabFactory::CanReimport(UObject* Obj, TArray<FString>& OutFilenames) {
  USuperDexPrefab* prefab = Cast<USuperDexPrefab>(Obj);
  if (!prefab) {
    return false;
  }
  if (prefab->AssetImportData) {
    prefab->AssetImportData->ExtractFilenames(OutFilenames);
  }
  return true;
}

void UReimportSuperDexPrefabFactory::SetReimportPaths(
    UObject* Obj,
    const TArray<FString>& NewReimportPaths) {
  USuperDexPrefab* prefab = Cast<USuperDexPrefab>(Obj);
  if (prefab && prefab->AssetImportData && NewReimportPaths.Num() > 0) {
    prefab->AssetImportData->UpdateFilenameOnly(NewReimportPaths[0]);
  }
}

EReimportResult::Type UReimportSuperDexPrefabFactory::Reimport(UObject* Obj) {
  USuperDexPrefab* prefab = Cast<USuperDexPrefab>(Obj);
  if (!prefab) {
    return EReimportResult::Failed;
  }
  const FString sourcePath = prefab->GetSourceFilePath();
  if (sourcePath.IsEmpty() || !FPaths::FileExists(sourcePath)) {
    UE_LOG(LogMochiPhysicsEditor, Error, TEXT("Source file not found: %s"), *sourcePath);
    return EReimportResult::Failed;
  }
  if (FSuperDexPrefabImporter::IsUpToDate(prefab)) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Display,
        TEXT("SuperDex prefab %s is already up to date, skipping reimport"),
        *prefab->GetName());
    return EReimportResult::Succeeded;
  }
  FScopedSlowTask slowTask(
      0,
      FText::Format(
          LOCTEXT("ReimportingPrefab", "Reimporting SuperDex prefab {0}..."),
          FText::FromString(prefab->PrefabName)));
  slowTask.MakeDialog();
  ImportSession.Reset();
  FSuperDexPrefabImporter importer(ImportSession);
  return importer.Import(prefab, sourcePath, prefab->ImportMode) ? EReimportResult::Succeeded
                                                                 : EReimportResult::Failed;
}

int32 UReimportSuperDexPrefabFactory::GetPriority() const {
  return ImportPriority;
}

#undef LOCTEXT_NAMESPACE
