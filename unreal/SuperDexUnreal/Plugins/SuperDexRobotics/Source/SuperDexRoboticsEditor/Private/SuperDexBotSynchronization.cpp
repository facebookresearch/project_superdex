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

#include "SuperDexBotSynchronization.h"

#include "AssetRegistry/AssetRegistryModule.h"
#include "Dialog/SCustomDialog.h"
#include "MochiPhysicsEditorModule.h"
#include "SuperDexBot.h"
#include "UObject/SavePackage.h"
#include "Widgets/Layout/SScrollBox.h"
#include "Widgets/Text/STextBlock.h"

#if WITH_EDITOR
#include "EditorReimportHandler.h"
#endif

#define LOCTEXT_NAMESPACE "SuperDexBotSynchronization"

TArray<FAssetData> FSuperDexBotSyncUtils::FindAllBotAssets() {
  FAssetRegistryModule const& AssetRegistryModule =
      FModuleManager::LoadModuleChecked<FAssetRegistryModule>("AssetRegistry");
  TArray<FAssetData> AssetDataArray;
  AssetRegistryModule.Get().GetAssetsByClass(
      USuperDexBot::StaticClass()->GetClassPathName(), AssetDataArray, true);
  return AssetDataArray;
}

TArray<USuperDexBot*> FSuperDexBotSyncUtils::FindOutOfSyncBots() {
  TArray<FAssetData> AllBotAssets = FindAllBotAssets();
  TArray<USuperDexBot*> OutOfSync;
  UE_LOG(
      LogMochiPhysicsEditor,
      Display,
      TEXT("[FSuperDexBotSyncUtils] Checking sync status of %d SuperDexBot asset(s)..."),
      AllBotAssets.Num());
  for (FAssetData const& AssetData : AllBotAssets) {
    USuperDexBot* Bot = Cast<USuperDexBot>(AssetData.GetAsset());
    if (!Bot) {
      continue;
    }
    if (Bot->IsOutOfSync()) {
      UE_LOG(
          LogMochiPhysicsEditor,
          Warning,
          TEXT("[FSuperDexBotSyncUtils] Out of sync: %s (source: %s)"),
          *Bot->GetBotName(),
          *Bot->GetSourceFilePath());
      OutOfSync.Add(Bot);
    }
  }
  if (OutOfSync.Num() == 0) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Display,
        TEXT("[FSuperDexBotSyncUtils] All %d SuperDexBot(s) are up to date."),
        AllBotAssets.Num());
  } else {
    UE_LOG(
        LogMochiPhysicsEditor,
        Warning,
        TEXT("[FSuperDexBotSyncUtils] %d of %d SuperDexBot(s) are out of sync."),
        OutOfSync.Num(),
        AllBotAssets.Num());
  }
  return OutOfSync;
}

int32 FSuperDexBotSyncUtils::ReimportOutOfSyncBots(TArray<USuperDexBot*> const& Bots) {
#if WITH_EDITOR
  int32 Count = 0;
  for (USuperDexBot* Bot : Bots) {
    if (!Bot) {
      continue;
    }
    UE_LOG(
        LogMochiPhysicsEditor,
        Display,
        TEXT("[FSuperDexBotSyncUtils] Reimporting: %s"),
        *Bot->GetBotName());
    bool bSuccess = FReimportManager::Instance()->Reimport(Bot, false, false);
    if (bSuccess) {
      ++Count;
    } else {
      UE_LOG(
          LogMochiPhysicsEditor,
          Error,
          TEXT("[FSuperDexBotSyncUtils] Failed to reimport: %s"),
          *Bot->GetBotName());
    }
  }
  UE_LOG(
      LogMochiPhysicsEditor,
      Display,
      TEXT("[FSuperDexBotSyncUtils] Reimport complete: %d of %d bot(s) reimported successfully."),
      Count,
      Bots.Num());
  return Count;
#else
  return 0;
#endif
}

int32 FSuperDexBotSyncUtils::ShowOutOfSyncDialog(TArray<USuperDexBot*> const& OutOfSyncBots) {
  if (OutOfSyncBots.IsEmpty()) {
    return 0;
  }

  TSharedRef<SScrollBox> BotList = SNew(SScrollBox);
  for (USuperDexBot const* Bot : OutOfSyncBots) {
    if (!Bot) {
      continue;
    }
    BotList->AddSlot()
        [SNew(SHorizontalBox) +
         SHorizontalBox::Slot().AutoWidth().Padding(
             4, 2)[SNew(STextBlock)
                       .Text(FText::FromString(Bot->GetBotName()))
                       .Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))] +
         SHorizontalBox::Slot().AutoWidth().Padding(
             4, 2)[SNew(STextBlock)
                       .Text(FText::FromString(Bot->GetSourceFilePath()))
                       .ColorAndOpacity(FSlateColor(FLinearColor::Gray))]];
  }

  // clang-format off
  TSharedRef<SCustomDialog> Dialog =
      SNew(SCustomDialog)
          .Title(LOCTEXT("OutOfSyncTitle", "Out-of-Sync SuperDex Robotics"))
          .Content()
              [SNew(SVerticalBox) +
               SVerticalBox::Slot()
                   .AutoHeight()
                   .Padding(10)
                       [SNew(STextBlock)
                            .Text(FText::Format(
                                LOCTEXT(
                                    "OutOfSyncMessage",
                                    "{0} SuperDexBot asset(s) have changed on disk since last import:"),
                                FText::AsNumber(OutOfSyncBots.Num())))] +
               SVerticalBox::Slot()
                   .FillHeight(1.0)
                   .MaxHeight(300)
                   .Padding(10, 0)
                       [BotList]]
          .Buttons(
              {SCustomDialog::FButton(LOCTEXT("ReimportAllButton", "Reimport All")),
               SCustomDialog::FButton(LOCTEXT("DismissButton", "Dismiss"))});
  // clang-format on

  int32 ButtonPressed = Dialog->ShowModal();
  if (ButtonPressed == 0) {
    return ReimportOutOfSyncBots(OutOfSyncBots);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// UReimportSuperDexBotsCommandlet
// ---------------------------------------------------------------------------

UReimportSuperDexBotsCommandlet::UReimportSuperDexBotsCommandlet() {
  IsClient = false;
  IsEditor = true;
  IsServer = false;
  LogToConsole = true;
}

int32 UReimportSuperDexBotsCommandlet::Main(const FString& Params) {
  UE_LOG(LogMochiPhysicsEditor, Display, TEXT("[ReimportSuperDexBotsCommandlet] Starting..."));

  // Ensure the Asset Registry has finished scanning all content directories.
  IAssetRegistry& AssetRegistry =
      FModuleManager::LoadModuleChecked<FAssetRegistryModule>("AssetRegistry").Get();
  AssetRegistry.SearchAllAssets(true);

  TArray<USuperDexBot*> OutOfSyncBots = FSuperDexBotSyncUtils::FindOutOfSyncBots();
  if (OutOfSyncBots.IsEmpty()) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Display,
        TEXT("[ReimportSuperDexBotsCommandlet] All bots are up to date. Nothing to do."));
    return 0;
  }

  int32 ReimportedCount = 0;
  TArray<USuperDexBot*> ReimportedBots;
  for (USuperDexBot* Bot : OutOfSyncBots) {
    if (!Bot) {
      continue;
    }
    UE_LOG(
        LogMochiPhysicsEditor,
        Display,
        TEXT("[ReimportSuperDexBotsCommandlet] Reimporting: %s"),
        *Bot->GetBotName());
    bool bSuccess = FReimportManager::Instance()->Reimport(Bot, false, false);
    if (bSuccess) {
      ++ReimportedCount;
      ReimportedBots.Add(Bot);
    } else {
      UE_LOG(
          LogMochiPhysicsEditor,
          Error,
          TEXT("[ReimportSuperDexBotsCommandlet] Failed to reimport: %s (skipping package save)"),
          *Bot->GetBotName());
    }
  }

  // Collect all unique packages that need saving: the bot itself, its skeletal mesh,
  // skeleton, and all MochiModels live in separate UE packages.
  TSet<UPackage*> PackagesToSave;
  for (USuperDexBot* Bot : ReimportedBots) {
    if (!Bot) {
      continue;
    }
    if (UPackage* BotPackage = Bot->GetOutermost()) {
      PackagesToSave.Add(BotPackage);
    }
    if (USkeletalMesh* SkelMesh = Bot->GetSkeletalMesh()) {
      if (UPackage* SkelMeshPackage = SkelMesh->GetOutermost()) {
        PackagesToSave.Add(SkelMeshPackage);
      }
      if (USkeleton* Skeleton = SkelMesh->GetSkeleton()) {
        if (UPackage* SkeletonPackage = Skeleton->GetOutermost()) {
          PackagesToSave.Add(SkeletonPackage);
        }
      }
    }
    for (UMochiModel* Model : Bot->GetMochiModels()) {
      if (Model) {
        if (UPackage* ModelPackage = Model->GetOutermost()) {
          PackagesToSave.Add(ModelPackage);
        }
      }
    }
  }

  int32 SavedCount = 0;
  for (UPackage* Package : PackagesToSave) {
    FString PackageFilename;
    if (FPackageName::TryConvertLongPackageNameToFilename(
            Package->GetName(), PackageFilename, FPackageName::GetAssetPackageExtension())) {
      FSavePackageArgs SaveArgs;
      if (UPackage::SavePackage(Package, nullptr, *PackageFilename, SaveArgs)) {
        ++SavedCount;
      } else {
        UE_LOG(
            LogMochiPhysicsEditor,
            Error,
            TEXT("[ReimportSuperDexBotsCommandlet] Failed to save package: %s"),
            *Package->GetName());
      }
    }
  }

  UE_LOG(
      LogMochiPhysicsEditor,
      Display,
      TEXT("[ReimportSuperDexBotsCommandlet] Done. Reimported %d bot(s), saved %d package(s)."),
      ReimportedCount,
      SavedCount);

  return ReimportedCount > 0 ? 0 : 1;
}

#undef LOCTEXT_NAMESPACE
