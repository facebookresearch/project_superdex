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
  constexpr int32 BlueprintButton = 0;
  constexpr int32 MapButton = 1;
  TSharedRef<SCustomDialog> Dialog =
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
  const int32 Choice = Dialog->ShowModal();
  if (Choice != BlueprintButton && Choice != MapButton) {
    return false;
  }
  OutMode = Choice == BlueprintButton ? ESuperDexPrefabImportMode::Blueprint
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
  const FString PrefabName = FSuperDexPrefabImporter::GetPrefabName(Filename);
  const FString DestFolder = FSuperDexPrefabImporter::GetDestinationFolder(
      Filename, FPackageName::GetLongPackagePath(InParent->GetPathName()));

  if (USuperDexPrefab* Existing =
          FSuperDexPrefabImporter::FindPrefabAsset(DestFolder, PrefabName)) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Display,
        TEXT("SuperDex prefab %s already exists in %s, reimporting it from %s"),
        *PrefabName,
        *DestFolder,
        *Filename);
    if (Existing->AssetImportData) {
      Existing->AssetImportData->UpdateFilenameOnly(Filename);
    }
    return FReimportManager::Instance()->Reimport(Existing, false, true) ? Existing : nullptr;
  }

  ESuperDexPrefabImportMode Mode = Filename.EndsWith(TEXT(".mochi_scene"))
      ? ESuperDexPrefabImportMode::Map
      : ESuperDexPrefabImportMode::Blueprint;
  if (!IsAutomatedImport() && !FApp::IsUnattended() &&
      !PromptForSuperDexPrefabImportMode(
          LOCTEXT("ImportModeTitle", "Import SuperDex Prefab"), PrefabName, Mode)) {
    bOutOperationCanceled = true;
    return nullptr;
  }

  USuperDexPrefab* Prefab = FSuperDexPrefabImporter::CreatePrefabAsset(DestFolder, PrefabName);

  FScopedSlowTask SlowTask(
      0,
      FText::Format(
          LOCTEXT("ImportingPrefab", "Importing SuperDex prefab {0}..."),
          FText::FromString(PrefabName)));
  SlowTask.MakeDialog();
  FSuperDexPrefabImporter Importer(ImportSession);
  if (!Importer.Import(Prefab, Filename, Mode)) {
    // Discard the empty asset rather than leaving it behind.
    FAssetRegistryModule::AssetDeleted(Prefab);
    Prefab->ClearFlags(RF_Public | RF_Standalone);
    Prefab->MarkAsGarbage();
    return nullptr;
  }
  return Prefab;
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
  USuperDexPrefab* Prefab = Cast<USuperDexPrefab>(Obj);
  if (!Prefab) {
    return false;
  }
  if (Prefab->AssetImportData) {
    Prefab->AssetImportData->ExtractFilenames(OutFilenames);
  }
  return true;
}

void UReimportSuperDexPrefabFactory::SetReimportPaths(
    UObject* Obj,
    const TArray<FString>& NewReimportPaths) {
  USuperDexPrefab* Prefab = Cast<USuperDexPrefab>(Obj);
  if (Prefab && Prefab->AssetImportData && NewReimportPaths.Num() > 0) {
    Prefab->AssetImportData->UpdateFilenameOnly(NewReimportPaths[0]);
  }
}

EReimportResult::Type UReimportSuperDexPrefabFactory::Reimport(UObject* Obj) {
  USuperDexPrefab* Prefab = Cast<USuperDexPrefab>(Obj);
  if (!Prefab) {
    return EReimportResult::Failed;
  }
  const FString SourcePath = Prefab->GetSourceFilePath();
  if (SourcePath.IsEmpty() || !FPaths::FileExists(SourcePath)) {
    UE_LOG(LogMochiPhysicsEditor, Error, TEXT("Source file not found: %s"), *SourcePath);
    return EReimportResult::Failed;
  }
  if (FSuperDexPrefabImporter::IsUpToDate(Prefab)) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Display,
        TEXT("SuperDex prefab %s is already up to date, skipping reimport"),
        *Prefab->GetName());
    return EReimportResult::Succeeded;
  }
  FScopedSlowTask SlowTask(
      0,
      FText::Format(
          LOCTEXT("ReimportingPrefab", "Reimporting SuperDex prefab {0}..."),
          FText::FromString(Prefab->PrefabName)));
  SlowTask.MakeDialog();
  ImportSession.Reset();
  FSuperDexPrefabImporter Importer(ImportSession);
  return Importer.Import(Prefab, SourcePath, Prefab->ImportMode) ? EReimportResult::Succeeded
                                                                 : EReimportResult::Failed;
}

int32 UReimportSuperDexPrefabFactory::GetPriority() const {
  return ImportPriority;
}

#undef LOCTEXT_NAMESPACE
