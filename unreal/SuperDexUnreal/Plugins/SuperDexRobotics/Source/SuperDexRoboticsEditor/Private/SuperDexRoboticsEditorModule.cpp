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

#include "SuperDexRoboticsEditorModule.h"

#include "AssetToolsModule.h"
#include "ContentBrowserMenuContexts.h"
#include "Developer/DesktopPlatform/Public/DesktopPlatformModule.h"
#include "Developer/DesktopPlatform/Public/IDesktopPlatform.h"
#include "Dialog/SCustomDialog.h"
#include "Editor.h"
#include "IAssetTools.h"
#include "MochiArticulationThumbnail.h"
#include "MochiPhysicsEditorModule.h"
#include "MochiPhysicsFunctionLibrary.h"
#include "Subsystems/PlacementSubsystem.h"
#include "SuperDexBot.h"
#include "SuperDexBotActorFactory.h"
#include "SuperDexBotSynchronization.h"
#include "SuperDexRoboticsAssetsEditor.h"
#include "SuperDexRoboticsFunctionLibrary.h"
#include "SuperDexSplitBlueprintChildren.h"
#include "ToolMenus.h"

#define LOCTEXT_NAMESPACE "FSuperDexRoboticsEditorModule"

void FSuperDexRoboticsEditorModule::StartupModule() {
  // get asset tools -- RegisterAdvancedAssetCategory is idempotent, so this is safe
  // regardless of whether FMochiPhysicsEditorModule has already registered "Mochi".
  IAssetTools& assetTools =
      FModuleManager::LoadModuleChecked<FAssetToolsModule>("AssetTools").Get();
  EAssetTypeCategories::Type mochiCategory =
      assetTools.RegisterAdvancedAssetCategory(FName(TEXT("Mochi")), FText::FromString("SuperDex"));
  const TSharedRef<IAssetTypeActions> botAssetActions =
      MakeShareable(new FSuperDexBotAssetActions(mochiCategory));
  assetTools.RegisterAssetTypeActions(botAssetActions);
  RegisteredAssetTypeActions.Add(botAssetActions);
  const TSharedRef<IAssetTypeActions> sceneAssetActions =
      MakeShareable(new FSuperDexSceneAssetActions(mochiCategory));
  assetTools.RegisterAssetTypeActions(sceneAssetActions);
  RegisteredAssetTypeActions.Add(sceneAssetActions);

  const TSharedRef<IAssetTypeActions> ikAssetActions =
      MakeShareable(new FSuperDexBotIKConfigAssetActions(mochiCategory));
  assetTools.RegisterAssetTypeActions(ikAssetActions);
  RegisteredAssetTypeActions.Add(ikAssetActions);

  for (const TSharedRef<IAssetTypeActions>& dataAssetActions : {
           TSharedRef<IAssetTypeActions>(MakeShared<FSuperDexPoseAssetActions>(mochiCategory)),
           TSharedRef<IAssetTypeActions>(
               MakeShared<FSuperDexStudyQuestionDataAssetActions>(mochiCategory)),
       }) {
    assetTools.RegisterAssetTypeActions(dataAssetActions);
    RegisteredAssetTypeActions.Add(dataAssetActions);
  }

  // Class names without their U prefix; icons live in SuperDexRobotics/Resources.
  static const TCHAR* const kClassIcons[] = {
      TEXT("SuperDexBot"),
      TEXT("SuperDexScene"),
      TEXT("SuperDexBotIKConfig"),
      TEXT("SuperDexPoseAsset"),
      TEXT("SuperDexStudyQuestionData"),
  };
  ClassIconStyle = MakeShared<FSuperDexClassIconStyle>(
      TEXT("SuperDexRoboticsEditorStyle"), TEXT("SuperDexRobotics"), kClassIcons);

  // register the "Split Mochi Children..." content-browser context menu entry
  RegisterAssetContextMenu();

  // register actor factory
  if (GEditor) {
    auto BotActorFactory = NewObject<USuperDexBotActorFactory>();
    GEditor->ActorFactories.Add(BotActorFactory);
    if (UPlacementSubsystem* PlacementSubsystem =
            GEditor->GetEditorSubsystem<UPlacementSubsystem>()) {
      PlacementSubsystem->RegisterAssetFactory(BotActorFactory);
    }
    UE_LOG(
        LogMochiPhysicsEditor,
        Display,
        TEXT("[%hs] Registered SuperDexRobotics actor factory."),
        __FUNCTION__);
  } else {
    UE_LOG(
        LogMochiPhysicsEditor,
        Warning,
        TEXT("[%hs] Failed to register SuperDexRobotics actor factory!"),
        __FUNCTION__);
  }

  // register thumbnail renderer (reuses SuperDexPhysics' UMochiArticulationThumbnailRenderer,
  // since USuperDexBot : public UMochiArticulationBase and the renderer is bot-agnostic)
  UThumbnailManager::Get().RegisterCustomRenderer(
      USuperDexBot::StaticClass(), UMochiArticulationThumbnailRenderer::StaticClass());

  // extend the "Mochi" menu (built by FMochiPhysicsEditorModule::RegisterMochiMenu) with
  // bot-specific entries once it exists.
  UToolMenus::RegisterStartupCallback(
      FSimpleMulticastDelegate::FDelegate::CreateRaw(
          this, &FSuperDexRoboticsEditorModule::RegisterSuperDexRoboticsMenu));
}

void FSuperDexRoboticsEditorModule::ShutdownModule() {
  // unregister assets
  FAssetToolsModule* AssetToolsModule =
      FModuleManager::GetModulePtr<FAssetToolsModule>("AssetTools");
  if (AssetToolsModule != nullptr) {
    IAssetTools& AssetTools = AssetToolsModule->Get();
    for (auto Action : RegisteredAssetTypeActions) {
      AssetTools.UnregisterAssetTypeActions(Action);
    }
  }
  ClassIconStyle.Reset();
}

void FSuperDexRoboticsEditorModule::RegisterAssetContextMenu() {
  // Extend the asset browser for Blueprints - split top-level SuperDex-bearing
  // children of a parent Blueprint into separate BP assets and write out a
  // matching .superdex_task JSON. Entry is shown for every Blueprint but
  // greys out when nothing in the BP looks splittable.
  if (UToolMenu* ToolMenu =
          UToolMenus::Get()->ExtendMenu("ContentBrowser.AssetContextMenu.Blueprint")) {
    FToolMenuSection& Section = ToolMenu->FindOrAddSection("GetAssetActions");
    Section.AddDynamicEntry(
        "SplitMochiChildren",
        FNewToolMenuSectionDelegate::CreateLambda([](FToolMenuSection& InSection) {
          const TAttribute<FText> Label =
              LOCTEXT("SplitMochiChildrenLabel", "Split SuperDex Children...");
          const TAttribute<FText> ToolTip = LOCTEXT(
              "SplitMochiChildrenTooltip",
              "Split each top-level SuperDex-bearing child of this Blueprint into its own BP "
              "(under <BPName>_Children/) and write a .superdex_task that respawns the layout.");
          const FSlateIcon Icon =
              FSlateIcon(FAppStyle::GetAppStyleSetName(), "BlueprintEditor.AddNewMacroDeclaration");
          FToolUIAction Action;
          Action.ExecuteAction =
              FToolMenuExecuteAction::CreateLambda([](const FToolMenuContext& InMenuContext) {
                if (const UContentBrowserAssetContextMenuContext* Context =
                        InMenuContext.FindContext<UContentBrowserAssetContextMenuContext>()) {
                  TArray<FAssetData> assets =
                      Context->GetSelectedAssetsOfType(UBlueprint::StaticClass());
                  for (FAssetData const& asset : assets) {
                    if (UBlueprint* BP = Cast<UBlueprint>(asset.GetAsset())) {
                      SuperDexSplitBlueprintChildren::ExecuteSplitBlueprintChildren(BP);
                    }
                  }
                }
              });
          Action.CanExecuteAction = FToolMenuCanExecuteAction::CreateLambda(
              [](const FToolMenuContext& InMenuContext) -> bool {
                if (const UContentBrowserAssetContextMenuContext* Context =
                        InMenuContext.FindContext<UContentBrowserAssetContextMenuContext>()) {
                  TArray<FAssetData> assets =
                      Context->GetSelectedAssetsOfType(UBlueprint::StaticClass());
                  for (FAssetData const& asset : assets) {
                    if (UBlueprint const* BP = Cast<UBlueprint>(asset.GetAsset())) {
                      if (SuperDexSplitBlueprintChildren::IsBlueprintSplittable(BP)) {
                        return true;
                      }
                    }
                  }
                }
                return false;
              });
          const FToolUIActionChoice UIAction(Action);
          InSection.AddMenuEntry("SplitMochiChildren", Label, ToolTip, Icon, UIAction);
        }));
  }
}

void FSuperDexRoboticsEditorModule::RegisterSuperDexRoboticsMenu() {
  // Extend -- do not re-add -- the "Mochi" submenu that FMochiPhysicsEditorModule registers as
  // "LevelEditor.MainMenu.Mochi". ExtendMenu() creates a pending extension if that module has
  // not registered the menu yet, so this is independent of module startup order.
  UToolMenu* Menu = UToolMenus::Get()->ExtendMenu("LevelEditor.MainMenu.Mochi");
  if (!Menu) {
    return;
  }

  FToolMenuSection& SubSection = Menu->FindOrAddSection("MochiExport");
  SubSection.AddMenuEntry(
      "ExportSuperDexBotScene",
      LOCTEXT("ExportSuperDexBotSceneLabel", "Export SuperDex Bot Scene Archive..."),
      LOCTEXT(
          "ExportSuperDexBotSceneTooltip",
          "Export a .superdex_scene_archive from the current level. "
          "Requires PIE and SuperDexBotSceneExportData components on robot actors."),
      FSlateIcon(FAppStyle::GetAppStyleSetName(), "MainFrame.SaveAll"),
      FUIAction(
          FExecuteAction::CreateStatic(
              &FSuperDexRoboticsEditorModule::ExecuteExportSuperDexBotScene),
          FCanExecuteAction::CreateStatic(&FMochiPhysicsEditorModule::CanExecuteExportMochiScene)));

  SubSection.AddMenuEntry(
      "ExportLiveSuperDexBotScene",
      LOCTEXT(
          "ExportLiveSuperDexBotSceneLabel", "Export SuperDex Bot Scene Archive (Live Scene)..."),
      LOCTEXT(
          "ExportLiveSuperDexBotSceneTooltip",
          "Export a .superdex_scene_archive capturing the current live scene "
          "(task and spawnables) directly into the base scene, with bots exported "
          "separately, like a SuperDex Recorder's live-scene archive. Requires PIE."),
      FSlateIcon(FAppStyle::GetAppStyleSetName(), "MainFrame.SaveAll"),
      FUIAction(
          FExecuteAction::CreateStatic(
              &FSuperDexRoboticsEditorModule::ExecuteExportLiveSuperDexBotScene),
          FCanExecuteAction::CreateStatic(&FMochiPhysicsEditorModule::CanExecuteExportMochiScene)));

  FToolMenuSection& BotSection = Menu->FindOrAddSection("SuperDexRobotics");
  BotSection.AddMenuEntry(
      "ReimportOutOfSyncBots",
      LOCTEXT("ReimportOutOfSyncBotsLabel", "Reimport Bots"),
      LOCTEXT(
          "ReimportOutOfSyncBotsTooltip",
          "Scans all SuperDexBot assets and reimports those whose source files have changed since last import."),
      FSlateIcon(FAppStyle::GetAppStyleSetName(), "Icons.Refresh"),
      FUIAction(
          FExecuteAction::CreateStatic(
              &FSuperDexRoboticsEditorModule::ExecuteReimportOutOfSyncBots)));
}

void FSuperDexRoboticsEditorModule::ExecuteExportSuperDexBotScene() {
  ExecuteExportSuperDexBotSceneInternal(/*bCaptureLiveScene=*/false);
}

void FSuperDexRoboticsEditorModule::ExecuteExportLiveSuperDexBotScene() {
  ExecuteExportSuperDexBotSceneInternal(/*bCaptureLiveScene=*/true);
}

void FSuperDexRoboticsEditorModule::ExecuteExportSuperDexBotSceneInternal(bool bCaptureLiveScene) {
  if (!FMochiPhysicsEditorModule::CanExecuteExportMochiScene()) {
    FMessageDialog::Open(
        EAppMsgType::Ok,
        LOCTEXT(
            "ExportSuperDexBotSceneNotInPIE",
            "Cannot export SuperDex bot scene: No Play-In-Editor session is active.\n\nPlease start a PIE session first."));
    return;
  }

  IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
  if (!DesktopPlatform) {
    return;
  }

  TArray<FString> OutFileNames;
  const void* ParentWindowHandle =
      FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr);

  bool bFileSelected = DesktopPlatform->SaveFileDialog(
      ParentWindowHandle,
      bCaptureLiveScene ? TEXT("Export SuperDex Bot Scene Archive (Live Scene)")
                        : TEXT("Export SuperDex Bot Scene Archive"),
      FPaths::ProjectSavedDir(),
      bCaptureLiveScene ? TEXT("BotScene_Live") : TEXT("BotScene"),
      TEXT("SuperDex Bot Scene Archive (*.superdex_scene_archive)|*.superdex_scene_archive"),
      EFileDialogFlags::None,
      OutFileNames);

  if (!bFileSelected || OutFileNames.IsEmpty()) {
    return;
  }

  FString SelectedFilePath = OutFileNames[0];
  FString ExportName = FPaths::GetBaseFilename(SelectedFilePath);
  FString OutputDir = FPaths::Combine(FPaths::GetPath(SelectedFilePath), ExportName);

  // Get the PIE world context.
  UWorld* PIEWorld = nullptr;
  for (const FWorldContext& Context : GEngine->GetWorldContexts()) {
    if (Context.WorldType == EWorldType::PIE && Context.World()) {
      PIEWorld = Context.World();
      break;
    }
  }

  if (!PIEWorld) {
    FMessageDialog::Open(
        EAppMsgType::Ok,
        LOCTEXT(
            "ExportSuperDexBotSceneNoPIEWorld",
            "Cannot export SuperDex bot scene: Could not find PIE world."));
    return;
  }

  UE_LOG(
      LogMochiPhysicsEditor,
      Display,
      TEXT("[%hs] Exporting SuperDex bot scene '%s' to '%s'"),
      __FUNCTION__,
      *ExportName,
      *OutputDir);

  GWarn->BeginSlowTask(
      FText::Format(
          LOCTEXT("SuperDexBotSceneExportingSlowTask", "Exporting SuperDex bot scene '{0}'..."),
          FText::FromString(ExportName)),
      true,
      false);

  bool bSuccess = USuperDexRoboticsFunctionLibrary::ExportSuperDexBotScene(
      PIEWorld, ExportName, OutputDir, bCaptureLiveScene);

  GWarn->EndSlowTask();

  FString ParentDir = FPaths::GetPath(SelectedFilePath);

  if (bSuccess) {
    auto onOpenFileLocation = [ParentDir]() {
      FString NormalizedPath = ParentDir;
      FPaths::NormalizeDirectoryName(NormalizedPath);
      FPaths::MakePlatformFilename(NormalizedPath);
      FPlatformProcess::ExploreFolder(*NormalizedPath);
    };

    TSharedRef<SCustomDialog> successDialog =
        SNew(SCustomDialog)
            .Title(LOCTEXT("SuperDexBotSceneExportSuccessTitle", "Export Complete"))
            .Content()
                [SNew(SHorizontalBox) +
                 SHorizontalBox::Slot().AutoWidth().Padding(
                     10)[SNew(SBox).WidthOverride(64).HeightOverride(
                     32)[SNew(SImage).Image(FAppStyle::GetBrush("Symbols.Check"))]] +
                 SHorizontalBox::Slot().AutoWidth().Padding(0, 10, 10, 10)
                     [SNew(STextBlock)
                          .Text(
                              FText::Format(
                                  LOCTEXT(
                                      "SuperDexBotSceneExportSuccessBody",
                                      "SuperDex bot scene archive exported successfully.\n\nExport Name: {0}\nLocation: {1}"),
                                  FText::FromString(ExportName),
                                  FText::FromString(ParentDir)))]]
            .Buttons(
                {SCustomDialog::FButton(
                     LOCTEXT("BotSceneOpenLocationButton", "Open Location"),
                     FSimpleDelegate::CreateLambda(onOpenFileLocation)),
                 SCustomDialog::FButton(LOCTEXT("BotSceneDialogButtonDone", "Done"))});
    successDialog->Show();
  } else {
    FMessageDialog::Open(
        EAppMsgType::Ok,
        FText::Format(
            LOCTEXT(
                "SuperDexBotSceneExportFailed",
                "SuperDex bot scene export failed.\n\nCheck the Output Log for details."),
            FText::FromString(ExportName)));
  }
}

void FSuperDexRoboticsEditorModule::ExecuteReimportOutOfSyncBots() {
  TArray<USuperDexBot*> OutOfSyncBots = FSuperDexBotSyncUtils::FindOutOfSyncBots();
  if (OutOfSyncBots.IsEmpty()) {
    FMessageDialog::Open(
        EAppMsgType::Ok,
        LOCTEXT("AllBotsUpToDate", "All SuperDexBot assets are up to date. No reimport needed."));
    return;
  }
  FSuperDexBotSyncUtils::ShowOutOfSyncDialog(OutOfSyncBots);
}

IMPLEMENT_MODULE(FSuperDexRoboticsEditorModule, SuperDexRoboticsEditor)

#undef LOCTEXT_NAMESPACE
