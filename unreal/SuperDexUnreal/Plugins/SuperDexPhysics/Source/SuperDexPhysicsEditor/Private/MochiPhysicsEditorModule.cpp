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

#include "MochiPhysicsEditorModule.h"

#include "AssetRegistry/AssetRegistryModule.h"
#include "ContentBrowserMenuContexts.h"
#include "Developer/DesktopPlatform/Public/DesktopPlatformModule.h"
#include "Developer/DesktopPlatform/Public/IDesktopPlatform.h"
#include "Dialog/SCustomDialog.h"
#include "Editor.h"
#include "IAssetTools.h"
#include "LevelEditor.h"
#include "MochiArticulationAssetUtils.h"
#include "MochiArticulationComponent.h"
#include "MochiArticulationDetailCustomization.h"
#include "MochiArticulationThumbnail.h"
#include "MochiAssets.h"
#include "MochiFactory.h"
#include "MochiModel.h"
#include "MochiModelThumbnailRenderer.h"
#include "MochiNumericFormatCustomization.h"
#include "MochiPhysicsFunctionLibrary.h"
#include "Styling/SlateIconFinder.h"
#include "Subsystems/PlacementSubsystem.h"
#include "ToolMenus.h"

DEFINE_LOG_CATEGORY(LogMochiPhysicsEditor);

#define LOCTEXT_NAMESPACE "FMochiPhysicsEditorModule"

void FMochiPhysicsEditorModule::StartupModule() {
  // get asset tools
  IAssetTools& assetTools =
      FModuleManager::LoadModuleChecked<FAssetToolsModule>("AssetTools").Get();
  // create Mochi category
  EAssetTypeCategories::Type mochiCategory =
      assetTools.RegisterAdvancedAssetCategory(FName(TEXT("Mochi")), FText::FromString("SuperDex"));
  // register assets
  RegisterAssetActions<FMochiMaterialParamsAssetActions>(assetTools, mochiCategory);
  RegisterAssetActions<FMochiShellMaterialParamsAssetActions>(assetTools, mochiCategory);
  RegisterAssetActions<FMochiRodMaterialParamsAssetActions>(assetTools, mochiCategory);
  RegisterAssetActions<FMochiContactParamsAssetActions>(assetTools, mochiCategory);
  RegisterAssetActions<FMochiSolverParamsAssetActions>(assetTools, mochiCategory);
  RegisterAssetActions<FMochiContactMatrixAssetActions>(assetTools, mochiCategory);
  RegisterAssetActions<FMochiModelAssetActions>(assetTools, mochiCategory);
  RegisterAssetActions<FMochiModelMaterialModifiersAssetActions>(assetTools, mochiCategory);
  RegisterAssetActions<FMochiArticulationAssetActions>(assetTools, mochiCategory);
  RegisterAssetActions<FSuperDexPrefabAssetActions>(assetTools, mochiCategory);
  // register asset context menu actions
  RegisterAssetContextMenu();
  // register actor factories
  if (GEditor) {
    auto SoftMeshActorFactory = NewObject<UMochiSoftMeshActorFactory>();
    auto ArticulationPreviewActorFactory = NewObject<UMochiArticulationActorFactory>();

    GEditor->ActorFactories.Add(SoftMeshActorFactory);
    GEditor->ActorFactories.Add(ArticulationPreviewActorFactory);

    if (UPlacementSubsystem* PlacementSubsystem =
            GEditor->GetEditorSubsystem<UPlacementSubsystem>()) {
      PlacementSubsystem->RegisterAssetFactory(SoftMeshActorFactory);
      PlacementSubsystem->RegisterAssetFactory(ArticulationPreviewActorFactory);
    }
    UE_LOG(
        LogMochiPhysicsEditor,
        Display,
        TEXT("[%hs] Registered Mochi Physics actor factories."),
        __FUNCTION__);
  } else {
    UE_LOG(
        LogMochiPhysicsEditor,
        Warning,
        TEXT("[%hs] Failed to register Mochi Physics actor factories!"),
        __FUNCTION__);
  }
  // register thumbnail renderer
  UThumbnailManager::Get().RegisterCustomRenderer(
      UMochiModel::StaticClass(), UMochiModelThumbnailRenderer::StaticClass());
  UThumbnailManager::Get().RegisterCustomRenderer(
      UMochiArticulation::StaticClass(), UMochiArticulationThumbnailRenderer::StaticClass());
  // register details customization
  FPropertyEditorModule& propEd =
      FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");
  propEd.RegisterCustomClassLayout(
      UMochiArticulation::StaticClass()->GetFName(),
      FOnGetDetailCustomizationInstance::CreateStatic(
          &FMochiArticulationDetailsCustomization::MakeInstance));
  propEd.RegisterCustomClassLayout(
      UMochiArticulationComponent::StaticClass()->GetFName(),
      FOnGetDetailCustomizationInstance::CreateStatic(
          &FMochiArticulationComponentDetailsCustomization::MakeInstance));
  propEd.RegisterCustomClassLayout(
      UMochiShellMaterialParams::StaticClass()->GetFName(),
      FOnGetDetailCustomizationInstance::CreateStatic(
          &FMochiScientificNotationCustomization::MakeInstance));
  propEd.RegisterCustomClassLayout(
      UMochiRodMaterialParams::StaticClass()->GetFName(),
      FOnGetDetailCustomizationInstance::CreateStatic(
          &FMochiScientificNotationCustomization::MakeInstance));
  ToolbarExtensibilityManager = MakeShareable(new FExtensibilityManager);
  // register the Mochi menu in the Level Editor's main menu bar
  UToolMenus::RegisterStartupCallback(
      FSimpleMulticastDelegate::FDelegate::CreateRaw(
          this, &FMochiPhysicsEditorModule::RegisterMochiMenu));
}

void FMochiPhysicsEditorModule::ShutdownModule() {
  // unregister assets
  FAssetToolsModule* AssetToolsModule =
      FModuleManager::GetModulePtr<FAssetToolsModule>("AssetTools");
  if (AssetToolsModule != nullptr) {
    IAssetTools& AssetTools = AssetToolsModule->Get();
    for (auto Action : RegisteredAssetTypeActions) {
      AssetTools.UnregisterAssetTypeActions(Action);
    }
  }
  // unregister details customization
  if (FModuleManager::Get().IsModuleLoaded("PropertyEditor")) {
    FPropertyEditorModule& propEdit =
        FModuleManager::GetModuleChecked<FPropertyEditorModule>("PropertyEditor");
    propEdit.UnregisterCustomClassLayout(UMochiArticulation::StaticClass()->GetFName());
    propEdit.UnregisterCustomClassLayout(UMochiArticulationComponent::StaticClass()->GetFName());
    propEdit.UnregisterCustomClassLayout(UMochiShellMaterialParams::StaticClass()->GetFName());
    propEdit.UnregisterCustomClassLayout(UMochiRodMaterialParams::StaticClass()->GetFName());
  }
}

void FMochiPhysicsEditorModule::RegisterAssetContextMenu() {
  // Extend the asset browser for MochiModel (export)
  if (UToolMenu* ToolMenu =
          UToolMenus::Get()->ExtendMenu("ContentBrowser.AssetContextMenu.MochiModel")) {
    FToolMenuSection& Section = ToolMenu->FindOrAddSection("GetAssetActions");
    Section.AddDynamicEntry(
        "ExportToHDF5", FNewToolMenuSectionDelegate::CreateLambda([](FToolMenuSection& InSection) {
          const TAttribute<FText> Label = LOCTEXT("ExportToHDF5Label", "Export to HDF5");
          const TAttribute<FText> ToolTip = LOCTEXT(
              "ExportToHDF5Tooltip",
              "Export the asset to an HDF5 file usable with other Mochi systems.");
          FSlateIcon Icon = FSlateIcon(FAppStyle::GetAppStyleSetName(), "MainFrame.SaveAll");
          const FToolMenuExecuteAction UIAction =
              FToolMenuExecuteAction::CreateLambda([](const FToolMenuContext& InMenuContext) {
                if (const UContentBrowserAssetContextMenuContext* Context =
                        InMenuContext.FindContext<UContentBrowserAssetContextMenuContext>()) {
                  TArray<FAssetData> assets =
                      Context->GetSelectedAssetsOfType(UMochiModel::StaticClass());
                  for (FAssetData const& asset : assets) {
                    if (UMochiModel* model = Cast<UMochiModel>(asset.GetAsset())) {
                      ExecuteExportMochiModelToHDF5(model, asset.AssetName);
                    }
                  }
                }
              });

          InSection.AddMenuEntry("ExportToHDF5", Label, ToolTip, Icon, UIAction);
        }));
  }

  // Extend the asset browser for MochiModel (tet mesh operations)
  if (UToolMenu* ToolMenu =
          UToolMenus::Get()->ExtendMenu("ContentBrowser.AssetContextMenu.MochiModel")) {
    FToolMenuSection& Section = ToolMenu->FindOrAddSection("GetAssetActions");
    Section.AddDynamicEntry(
        "ConvertToSurfaceMesh",
        FNewToolMenuSectionDelegate::CreateLambda([](FToolMenuSection& InSection) {
          const TAttribute<FText> Label =
              LOCTEXT("ConvertToSurfaceMeshLabel", "Convert to Surface Mesh");
          const TAttribute<FText> ToolTip = LOCTEXT(
              "ConvertToSurfaceMeshTooltip",
              "Extracts the surface mesh from a tetrahedral mesh and replaces the H5 data.");
          FSlateIcon Icon = FSlateIcon(FAppStyle::GetAppStyleSetName(), "ClassIcon.StaticMesh");
          const FToolMenuExecuteAction UIAction =
              FToolMenuExecuteAction::CreateLambda([](const FToolMenuContext& InMenuContext) {
                if (const UContentBrowserAssetContextMenuContext* Context =
                        InMenuContext.FindContext<UContentBrowserAssetContextMenuContext>()) {
                  TArray<FAssetData> assets =
                      Context->GetSelectedAssetsOfType(UMochiModel::StaticClass());
                  for (FAssetData const& asset : assets) {
                    if (UMochiModel* model = Cast<UMochiModel>(asset.GetAsset())) {
                      model->ConvertTetMeshToSurfaceMesh();
                    }
                  }
                }
              });

          InSection.AddMenuEntry("ConvertToSurfaceMesh", Label, ToolTip, Icon, UIAction);
        }));

    Section.AddDynamicEntry(
        "BakeSdf", FNewToolMenuSectionDelegate::CreateLambda([](FToolMenuSection& InSection) {
          const TAttribute<FText> Label = LOCTEXT("BakeSdfLabel", "Bake SDF...");
          const TAttribute<FText> ToolTip = LOCTEXT(
              "BakeSdfTooltip", "Computes and bakes a signed distance field (SDF) for collision.");
          FSlateIcon Icon =
              FSlateIcon(FAppStyle::GetAppStyleSetName(), "LevelEditor.Tabs.Viewports");
          const FToolMenuExecuteAction UIAction =
              FToolMenuExecuteAction::CreateLambda([](const FToolMenuContext& InMenuContext) {
                if (const UContentBrowserAssetContextMenuContext* Context =
                        InMenuContext.FindContext<UContentBrowserAssetContextMenuContext>()) {
                  TArray<FAssetData> assets =
                      Context->GetSelectedAssetsOfType(UMochiModel::StaticClass());
                  if (assets.Num() == 0) {
                    return;
                  }

                  // Create options object
                  UBakeSdfOptions* opt = NewObject<UBakeSdfOptions>();

                  // Create details view for the options
                  FPropertyEditorModule& PropertyModule =
                      FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");
                  FDetailsViewArgs DetailsViewArgs;
                  DetailsViewArgs.bUpdatesFromSelection = false;
                  DetailsViewArgs.bLockable = false;
                  DetailsViewArgs.bAllowSearch = false;
                  DetailsViewArgs.NameAreaSettings = FDetailsViewArgs::HideNameArea;
                  TSharedRef<IDetailsView> DetailsView =
                      PropertyModule.CreateDetailView(DetailsViewArgs);
                  DetailsView->SetObject(opt);

                  auto onBakeLambda = [assets, opt]() {
                    for (FAssetData const& asset : assets) {
                      if (UMochiModel* model = Cast<UMochiModel>(asset.GetAsset())) {
                        model->BakeSdf(&opt->GridSdfParams);
                      }
                    }
                  };

                  TSharedRef<SCustomDialog> dialog =
                      SNew(SCustomDialog)
                          .Title(LOCTEXT("BakeSdfDialogTitle", "Bake SDF Options"))
                          .UseScrollBox(false)
                          .Content()[SNew(SBox).MinDesiredWidth(400).MinDesiredHeight(
                              200)[DetailsView]]
                          .Buttons(
                              {SCustomDialog::FButton(
                                   LOCTEXT("BakeSdfButtonBake", "Bake"),
                                   FSimpleDelegate::CreateLambda(onBakeLambda)),
                               SCustomDialog::FButton(LOCTEXT("BakeSdfButtonCancel", "Cancel"))});
                  dialog->Show();
                }
              });

          InSection.AddMenuEntry("BakeSdf", Label, ToolTip, Icon, UIAction);
        }));
  }

  // Extend the asset browser for UMochiArticulation assets
  if (UToolMenu* ToolMenu =
          UToolMenus::Get()->ExtendMenu("ContentBrowser.AssetContextMenu.MochiArticulation")) {
    FToolMenuSection& Section = ToolMenu->FindOrAddSection("GetAssetActions");

    Section.AddDynamicEntry(
        "CreateJointControlGains",
        FNewToolMenuSectionDelegate::CreateLambda([](FToolMenuSection& InSection) {
          const TAttribute<FText> Label =
              LOCTEXT("CreateJointControlGains", "Create Joint Control Gains");
          const TAttribute<FText> ToolTip = LOCTEXT(
              "CreateJointControlGainsToolTip",
              "Creates a table of joint control gains for the joints described by this MochiArticulation");
          const FSlateIcon Icon = FSlateIconFinder::FindIconForClass(UDataTable::StaticClass());
          const FToolMenuExecuteAction UIAction =
              FToolMenuExecuteAction::CreateLambda([](const FToolMenuContext& InMenuContext) {
                if (const UContentBrowserAssetContextMenuContext* Context =
                        InMenuContext.FindContext<UContentBrowserAssetContextMenuContext>()) {
                  TArray<FAssetData> assets =
                      Context->GetSelectedAssetsOfType(UMochiArticulation::StaticClass());
                  for (FAssetData const& asset : assets) {
                    if (UMochiArticulation* articulation =
                            Cast<UMochiArticulation>(asset.GetAsset())) {
                      UE_LOG(
                          LogMochiPhysicsEditor,
                          Display,
                          TEXT("[%hs] Attempting to create joint control gains on %s"),
                          __FUNCTION__,
                          *articulation->GetName());
                      FMochiArticulationAssetUtils::GenerateJointControlGainsTable(articulation);
                    }
                  }
                }
              });

          InSection.AddMenuEntry("CreateJointControlGains", Label, ToolTip, Icon, UIAction);
        }));
  }
}

void FMochiPhysicsEditorModule::ExecuteExportMochiModelToHDF5(
    UMochiModel* MochiModel,
    FName BaseName) {
  if (!MochiModel) {
    return;
  }

  IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
  if (!DesktopPlatform) {
    return;
  }
  TArray<FString> OutFileNames;
  const void* ParentWindowHandle =
      FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr);

  // Prompt for filepath
  bool bFileSelected = DesktopPlatform->SaveFileDialog(
      ParentWindowHandle,
      TEXT("Export to HDF5"),
      FPaths::ProfilingDir(),
      MochiModel->GetName() + TEXT(".mochi.h5"),
      TEXT("HDF5 files (*.h5)|*.h5|All files (*.*)|*.*"),
      EFileDialogFlags::None,
      OutFileNames);

  if (bFileSelected && !OutFileNames.IsEmpty()) {
    // Attempt to write the file
    FString SelectedFilePath = OutFileNames[0];
    if (MochiModel->ExportToFilename(SelectedFilePath)) {
      UE_LOG(
          LogMochiPhysicsEditor,
          Display,
          TEXT("[%hs] Output file: %s"),
          __FUNCTION__,
          *SelectedFilePath);

      auto onOpenFileLocation = [MochiModel, SelectedFilePath]() {
        FPlatformProcess::ExploreFolder(*SelectedFilePath);
      };

      // Success dialog with "open file location"
      TSharedRef<SCustomDialog> successDialog =
          SNew(SCustomDialog)
              .Title(LOCTEXT("HDF5ExportSuccessTitle", "Success"))
              .Content()
                  [SNew(SHorizontalBox) +
                   SHorizontalBox::Slot().AutoWidth().Padding(
                       10)[SNew(SImage).Image(FAppStyle::GetBrush("Symbols.Check"))] +
                   SHorizontalBox::Slot().AutoWidth().Padding(0, 10, 10, 10)
                       [SNew(STextBlock)
                            .Text(
                                FText::Format(
                                    LOCTEXT("HDF5ExportSuccessBody", "Exported HDF5 to {0}"),
                                    FText::FromString(SelectedFilePath)))]]
              .Buttons(
                  {SCustomDialog::FButton(
                       LOCTEXT("OpenFileLocationButton", "Open Location"),
                       FSimpleDelegate::CreateLambda(onOpenFileLocation)),
                   SCustomDialog::FButton(LOCTEXT("DialogButtonDone", "Done"))});
      successDialog->Show();
    }
  }
}

void FMochiPhysicsEditorModule::RegisterMochiMenu() {
  UToolMenu* Menu = UToolMenus::Get()->ExtendMenu("LevelEditor.MainMenu");
  if (!Menu) {
    return;
  }

  // Add a "Mochi" submenu to the main menu bar. The submenu is a *registered* menu
  // ("LevelEditor.MainMenu.Mochi") rather than an inline construct delegate so that sibling
  // editor modules -- FSuperDexRoboticsEditorModule in the SuperDexRobotics plugin -- can
  // contribute entries via ExtendMenu(). Two modules adding a submenu named "Mochi" to this same
  // section would collide on the entry name and one set of entries would silently disappear.
  FToolMenuSection& Section = Menu->FindOrAddSection("MochiSection");
  Section.AddSubMenu(
      "Mochi",
      LOCTEXT("MochiMenuLabel", "SuperDex"),
      LOCTEXT("MochiMenuTooltip", "SuperDex Physics Tools"),
      FNewToolMenuChoice());

  UToolMenu* MochiMenu = UToolMenus::Get()->RegisterMenu("LevelEditor.MainMenu.Mochi");
  if (!MochiMenu) {
    return;
  }

  FToolMenuSection& SubSection = MochiMenu->FindOrAddSection("MochiExport");
  SubSection.AddMenuEntry(
      "ExportMochiScene",
      LOCTEXT("ExportMochiSceneLabel", "Export Mochi Scene..."),
      LOCTEXT(
          "ExportMochiSceneTooltip",
          "Export the current Mochi physics scene to a file. Requires an active Play-In-Editor session."),
      FSlateIcon(FAppStyle::GetAppStyleSetName(), "MainFrame.SaveAll"),
      FUIAction(
          FExecuteAction::CreateStatic(&FMochiPhysicsEditorModule::ExecuteExportMochiScene),
          FCanExecuteAction::CreateStatic(&FMochiPhysicsEditorModule::CanExecuteExportMochiScene)));
}

bool FMochiPhysicsEditorModule::CanExecuteExportMochiScene() {
  return GEditor && GEditor->IsPlayingSessionInEditor();
}

void FMochiPhysicsEditorModule::ExecuteExportMochiScene() {
  if (!CanExecuteExportMochiScene()) {
    FMessageDialog::Open(
        EAppMsgType::Ok,
        LOCTEXT(
            "ExportMochiSceneNotInPIE",
            "Cannot export Mochi scene: No Play-In-Editor session is active.\n\nPlease start a PIE session first."));
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
      TEXT("Export Mochi Scene"),
      FPaths::ProjectSavedDir(),
      TEXT("MochiScene"),
      TEXT("Mochi Scene files (*.mochi)|*.mochi|All files (*.*)|*.*"),
      EFileDialogFlags::None,
      OutFileNames);

  if (!bFileSelected || OutFileNames.IsEmpty()) {
    return;
  }

  FString SelectedFilePath = OutFileNames[0];

  // Extract exportName (filename without extension) and filePath (directory)
  FString ExportName = FPaths::GetBaseFilename(SelectedFilePath);
  FString FilePath = FPaths::GetPath(SelectedFilePath);

  // Get the PIE world context
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
            "ExportMochiSceneNoPIEWorld", "Cannot export Mochi scene: Could not find PIE world."));
    return;
  }

  UMochiPhysicsFunctionLibrary::ExportMochiScene(PIEWorld, ExportName, FilePath);

  UE_LOG(
      LogMochiPhysicsEditor,
      Display,
      TEXT("[%hs] Exporting Mochi scene '%s' to '%s'"),
      __FUNCTION__,
      *ExportName,
      *FilePath);

  // Use the directory path since the file may not exist yet (export is async)
  auto onOpenFileLocation = [FilePath]() {
    FString NormalizedPath = FilePath;
    FPaths::NormalizeDirectoryName(NormalizedPath);
    FPaths::MakePlatformFilename(NormalizedPath);
    FPlatformProcess::ExploreFolder(*NormalizedPath);
  };

  TSharedRef<SCustomDialog> successDialog =
      SNew(SCustomDialog)
          .Title(LOCTEXT("MochiExportSuccessTitle", "Export Started"))
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
                                    "MochiExportSuccessBody",
                                    "Mochi scene export queued.\n\nExport Name: {0}\nLocation: {1}"),
                                FText::FromString(ExportName),
                                FText::FromString(FilePath)))]]
          .Buttons(
              {SCustomDialog::FButton(
                   LOCTEXT("OpenFileLocationButton", "Open Location"),
                   FSimpleDelegate::CreateLambda(onOpenFileLocation)),
               SCustomDialog::FButton(LOCTEXT("DialogButtonDone", "Done"))});
  successDialog->Show();
}

IMPLEMENT_MODULE(FMochiPhysicsEditorModule, SuperDexPhysicsEditor)

#undef LOCTEXT_NAMESPACE
