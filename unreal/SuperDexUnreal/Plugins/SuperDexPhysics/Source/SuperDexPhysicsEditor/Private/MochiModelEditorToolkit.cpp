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

#include "MochiModelEditorToolkit.h"

#include "MochiPhysicsEditorModule.h"

#include "AssetEditorModeManager.h"
#include "ComponentReregisterContext.h"
#include "MochiAssetEditorTabs.h"
#include "MochiAssetsEditor.h"
#include "MochiModelEditorCommands.h"
#include "MochiModelEditorViewportClient.h"
#include "MochiModelPreviewScene.h"
#include "UObject/UObjectGlobals.h"
#include "Widgets/Input/SSegmentedControl.h"

#define LOCTEXT_NAMESPACE "FMochiPhysicsEditorModule"

const FName MochiModelEditorAppIdentifier = FName(TEXT("MochiModelEditorApp"));
const FName FMochiModelEditorToolkit::ViewportTabID(TEXT("MochiModelEditorToolkit_Viewport"));
const FName FMochiModelEditorToolkit::DetailsTabID(TEXT("MochiModelEditorToolkit_Details"));

FMochiModelEditorToolkit::FMochiModelEditorToolkit()
    : StandaloneDefaultLayout(
          MochiAssetEditorTabs::MakeDefaultLayout(
              TEXT("Standalone_MochiModelEditor_Layout_v1"),
              ViewportTabID,
              DetailsTabID)) {}

FMochiModelEditorToolkit::~FMochiModelEditorToolkit() {
  FCoreUObjectDelegates::OnObjectModified.RemoveAll(this);
}

void FMochiModelEditorToolkit::InitMochiModelEditor(
    const EToolkitMode::Type Mode,
    const TSharedPtr<class IToolkitHost>& InitToolkitHost,
    UMochiModel* ObjectToEdit) {
  // Commands should be bound to UI elements at their respective level,
  // so commands that are inside the Viewport are bound in Viewport or ViewportClient
  // See MochiModelEditorCommands.h for full list of commands
  BindCommands();

  MochiModel = Cast<UMochiModel>(ObjectToEdit);

  // This custom viewport allows us to do deep customization
  ModelEditorViewport = SNew(SMochiModelEditorViewport, SharedThis(this), CreatePreviewScene());

  const bool bCreateDefaultStandaloneMenu = true;
  const bool bCreateDefaultToolbar = true;

  InitAssetEditor(
      Mode,
      InitToolkitHost,
      MochiModelEditorAppIdentifier,
      GetDefaultLayout(),
      bCreateDefaultStandaloneMenu,
      bCreateDefaultToolbar,
      ObjectToEdit);

  // Top level buttons (e.g. "Reimport Base Mesh" Static Mesh Editor)
  ExtendToolBars();

  FocusViewport();

  FCoreUObjectDelegates::OnObjectModified.AddSP(this, &FMochiModelEditorToolkit::OnObjectModified);
}

void FMochiModelEditorToolkit::BindCommands() {
  FMochiModelEditorCommands::Register();

  const FMochiModelEditorCommands& Commands = FMochiModelEditorCommands::Get();

  ToolkitCommands->MapAction(
      Commands.FocusViewport,
      FExecuteAction::CreateSP(this, &FMochiModelEditorToolkit::FocusViewport));

  ToolkitCommands->MapAction(
      Commands.ApplyMaterialModifer,
      FExecuteAction::CreateSP(this, &FMochiModelEditorToolkit::AddMaterialModifiers));

  ToolkitCommands->MapAction(
      Commands.RemoveMaterialModifier,
      FExecuteAction::CreateSP(this, &FMochiModelEditorToolkit::RemoveMaterialModifiers));
}

void FMochiModelEditorToolkit::ExtendToolBars() {
  // Follows the toolbar extension pattern of the engine's asset editor toolkits.
  struct Local {
    static void FillToolbar(FToolBarBuilder& ToolBarBuilder) {
      ToolBarBuilder.BeginSection("ExtendToolbarItem");
      {
        ToolBarBuilder.AddToolBarButton(
            FMochiModelEditorCommands::Get().FocusViewport,
            NAME_None,
            LOCTEXT("FocusViewport", "Focus Viewport"),
            LOCTEXT("FocusViewportTooltip", "Focuses viewport on Mochi model"),
            FSlateIcon(FAppStyle::GetAppStyleSetName(), "ClassIcon.CameraComponent"));
        // Reuses the material modifiers asset thumbnail as the button icon. That icon is defined
        // in MochiAssetsEditor.cpp; its style set is referenced by name because it has no static
        // Get() accessor.
        ToolBarBuilder.AddToolBarButton(
            FMochiModelEditorCommands::Get().ApplyMaterialModifer,
            NAME_None,
            LOCTEXT("ApplyMaterialModifier", "Apply Material Modifiers"),
            LOCTEXT(
                "ApplyMaterialModifierTooltip",
                "Adds to the Material Modifiers Map"
                " using the selected elements"
                " and the values in the template."),
            FSlateIcon("MochiPhysicsEditorStyle", "ClassThumbnail.MochiModelMaterialModifiers"));
        ToolBarBuilder.AddToolBarButton(
            FMochiModelEditorCommands::Get().RemoveMaterialModifier,
            NAME_None,
            LOCTEXT("DeleteMaterialModifier", "Remove from Material Modifiers"),
            LOCTEXT(
                "DeleteMaterialModifierTooltip",
                "Removes the selected elements from the Material Modifiers Map"),
            FSlateIcon(FAppStyle::GetAppStyleSetName(), "Icons.Delete"));
      }
      ToolBarBuilder.EndSection();
    }
  };

  const TSharedPtr<FExtender> ToolBarExtender = MakeShareable(new FExtender);

  ToolBarExtender->AddToolBarExtension(
      "Asset",
      EExtensionHook::After,
      GetToolkitCommands(),
      FToolBarExtensionDelegate::CreateStatic(&Local::FillToolbar));

  ToolBarExtender->AddToolBarExtension(
      "Asset",
      EExtensionHook::After,
      GetToolkitCommands(),
      FToolBarExtensionDelegate::CreateLambda([this](FToolBarBuilder& ParentToolbarBuilder) {
        // Choose between "Always Clear Selection", "Add to Selection", and
        // "Subtract from Selection"
        ParentToolbarBuilder.BeginSection("SelectionMode");
        ParentToolbarBuilder.AddWidget(
            SNew(STextBlock).Text(LOCTEXT("_", "Selection Mode")).Margin(FMargin(10.0f, 10.0f)));
        ParentToolbarBuilder.AddWidget(
            SNew(SSegmentedControl<EMochiModelEditorSelectionMode>)
                .Visibility(EVisibility::Visible)
                .OnValueChanged_Lambda([this](EMochiModelEditorSelectionMode InMode) {
                  if (InMode != GetViewportSelectionMode())
                    SetViewportSelectionMode(InMode);
                })
                .Value(this, &FMochiModelEditorToolkit::GetViewportSelectionMode)

            + SSegmentedControl<EMochiModelEditorSelectionMode>::Slot(
                  EMochiModelEditorSelectionMode::ClearFirst)
                  .Icon(FAppStyle::Get().GetBrush("Icons.Refresh"))
                  .ToolTip(LOCTEXT("_", "New selections always clear the current selection"))

            + SSegmentedControl<EMochiModelEditorSelectionMode>::Slot(
                  EMochiModelEditorSelectionMode::Additive)
                  .Icon(FAppStyle::Get().GetBrush("Icons.Plus"))
                  .ToolTip(LOCTEXT("_", "New selections add to the current selection"))

            + SSegmentedControl<EMochiModelEditorSelectionMode>::Slot(
                  EMochiModelEditorSelectionMode::Subtractive)
                  .Icon(FAppStyle::Get().GetBrush("Icons.Minus"))
                  .ToolTip(LOCTEXT("_", "New selections subtract from the current selection")));
        ParentToolbarBuilder.EndSection();

        // Choose between showing Young's Modulus modifiers, Poisson Ratio modifiers,
        // or show all elements
        ParentToolbarBuilder.BeginSection("ElementViewMode");
        ParentToolbarBuilder.AddWidget(
            SNew(STextBlock).Text(LOCTEXT("_", "Element View")).Margin(FMargin(10.0f, 10.0f)));
        ParentToolbarBuilder.AddWidget(
            SNew(SSegmentedControl<EMochiModelEditorElementViewMode>)
                .Visibility(EVisibility::Visible)
                .OnValueChanged_Lambda([this](EMochiModelEditorElementViewMode InMode) {
                  if (InMode != GetViewportElementViewMode())
                    SetViewportElementViewMode(InMode);
                })
                .Value(this, &FMochiModelEditorToolkit::GetViewportElementViewMode)

            + SSegmentedControl<EMochiModelEditorElementViewMode>::Slot(
                  EMochiModelEditorElementViewMode::All)
                  .Text(LOCTEXT("_", "All"))
                  .ToolTip(LOCTEXT("_", "Show all elements"))

            + SSegmentedControl<EMochiModelEditorElementViewMode>::Slot(
                  EMochiModelEditorElementViewMode::ModifiedYoungsModulus)
                  .Text(LOCTEXT("_", "Y M"))
                  .ToolTip(LOCTEXT(
                      "_",
                      "Show only elements with Young's Modulus modifiers,"
                      "\n and color them by the value."))

            + SSegmentedControl<EMochiModelEditorElementViewMode>::Slot(
                  EMochiModelEditorElementViewMode::ModifiedPoissonRatio)
                  .Text(LOCTEXT("_", "P R"))
                  .ToolTip(LOCTEXT(
                      "_",
                      "Show only elements with Poisson Ratio modifiers,"
                      "\n and color them by the value.")));
        ParentToolbarBuilder.EndSection();
      }));

  AddToolbarExtender(ToolBarExtender);

  FMochiPhysicsEditorModule* MochiPhysicsEditorModule =
      &FModuleManager::LoadModuleChecked<FMochiPhysicsEditorModule>("SuperDexPhysicsEditor");
  AddToolbarExtender(
      MochiPhysicsEditorModule->GetEditorToolbarExtensibilityManager()->GetAllExtenders());

  RegenerateMenusAndToolbars();
}

void FMochiModelEditorToolkit::FocusViewport() const {
  if (ModelEditorViewport.IsValid()) {
    // Calling Viewport's built-in method that has been overridden
    ModelEditorViewport->OnFocusViewportToSelection();
  }
}

TSharedPtr<FMochiModelPreviewScene> FMochiModelEditorToolkit::CreatePreviewScene() {
  if (!ModelPreviewScene.IsValid()) {
    ModelPreviewScene = MakeShareable(new FMochiModelPreviewScene(
        FPreviewScene::ConstructionValues()
            .AllowAudioPlayback(true)
            .ShouldSimulatePhysics(true)
            .ForceUseMovementComponentInNonGameWorld(true),
        StaticCastSharedRef<FMochiModelEditorToolkit>(AsShared())));
  }

  return ModelPreviewScene;
}

void FMochiModelEditorToolkit::RegisterTabSpawners(const TSharedRef<FTabManager>& InTabManager) {
  // The base class picks AssetEditorTabsCategory.
  Super::RegisterTabSpawners(InTabManager);
  DetailsView = MochiAssetEditorTabs::MakeDetailsView(MochiModel, this);
  MochiAssetEditorTabs::RegisterTabSpawners(
      *InTabManager,
      AssetEditorTabsCategory.ToSharedRef(),
      ViewportTabID,
      FOnSpawnTab::CreateSP(this, &FMochiModelEditorToolkit::SpawnTab_Viewport),
      DetailsTabID,
      FOnSpawnTab::CreateSP(this, &FMochiModelEditorToolkit::SpawnTab_Details));
}

void FMochiModelEditorToolkit::UnregisterTabSpawners(const TSharedRef<FTabManager>& InTabManager) {
  Super::UnregisterTabSpawners(InTabManager);

  InTabManager->UnregisterTabSpawner(DetailsTabID);
  InTabManager->UnregisterTabSpawner(ViewportTabID);
}

const TSharedRef<FTabManager::FLayout> FMochiModelEditorToolkit::GetDefaultLayout() const {
  return StandaloneDefaultLayout.ToSharedRef();
}

void FMochiModelEditorToolkit::AddMaterialModifiers() {
  if (ModelEditorViewport) {
    auto ViewportClient = ModelEditorViewport->GetViewportClient();
    if (ViewportClient) {
      ViewportClient.Get()->AddSelectionToMaterialModifiers();
    }
  }
}

void FMochiModelEditorToolkit::RemoveMaterialModifiers() {
  if (ModelEditorViewport) {
    auto ViewportClient = ModelEditorViewport->GetViewportClient();
    if (ViewportClient) {
      ViewportClient.Get()->RemoveSelectionFromMaterialModifiers();
    }
  }
}

TSharedRef<SDockTab> FMochiModelEditorToolkit::SpawnTab_Viewport(const FSpawnTabArgs& Args) {
  TSharedRef<SDockTab> SpawnedTab = SNew(SDockTab).Label(LOCTEXT("ViewportTab_Title", "Viewport"));
  if (ModelEditorViewport.IsValid()) {
    SpawnedTab->SetContent(ModelEditorViewport.ToSharedRef());

    int nNodes = 0;
    int nElements = 0;
    if (UMochiModel* model = Cast<UMochiModel>(MochiModel)) {
      nNodes = model->NumNodes;
      nElements = model->NumPolyhedrons;
    }

    TArray<FText> overlayItems;
    overlayItems.Add(FText::Format(LOCTEXT("Nodes_F", "Nodes: {0}"), FText::AsNumber(nNodes)));
    overlayItems.Add(
        FText::Format(LOCTEXT("Elements_F", "Elements: {0}"), FText::AsNumber(nElements)));

    UMochiSoftMeshComponent* PreviewMeshComponent = ModelEditorViewport->GetPreviewMeshComponent();
    if (PreviewMeshComponent) {
      FVector box = PreviewMeshComponent->GetLocalBounds().BoxExtent * 2.0;
      FNumberFormattingOptions opt;
      opt.SetMaximumFractionalDigits(2);
      overlayItems.Add(
          FText::Format(
              LOCTEXT("ApproxSize_F", "Approx Size: {0}x{1}x{2}"),
              FText::AsNumber(box.X, &opt),
              FText::AsNumber(box.Y, &opt),
              FText::AsNumber(box.Z, &opt)));
    }

    overlayItems.Add(LOCTEXT(
        "Tips",
        "\nOrbit Camera: Hold down ALT+LeftMouseButton"
        "\nSelection: Use CTRL+ALT+LeftMouseButton"
        "\nSelection Modes: Use the toolbar to select between modes,"
        "\n such as always adding to the current selection,"
        "\n or subtracting from the current selection."
        "\nTip: Elements hard to see? Use Wireframe mode."
        "\nTip: Lasso selection always red? Draw in the opposite direction."));

    ModelEditorViewport->PopulateOverlayText(overlayItems);
  }

  return SpawnedTab;
}

TSharedRef<SDockTab> FMochiModelEditorToolkit::SpawnTab_Details(const FSpawnTabArgs& Args) {
  TSharedPtr<SDockTab> DetailsTab = SNew(SDockTab).Label(
      LOCTEXT("MochiModelDetailsTitle", "MochiModelDetails"))[DetailsView.ToSharedRef()];

  return DetailsTab.ToSharedRef();
}

void FMochiModelEditorToolkit::RegisterToolbar() {}

void FMochiModelEditorToolkit::CreateEditorModeManager() {
  EditorModeManager = MakeShared<FAssetEditorModeManager>();
}

void FMochiModelEditorToolkit::OnObjectModified(UObject* InObject) {
  // TODO: handle more objects!

  if (MochiModel == InObject && ModelEditorViewport) {
    auto PreviewMeshComponent = ModelEditorViewport->GetPreviewMeshComponent();
    UE_LOG(
        LogMochiPhysicsEditor,
        Display,
        TEXT("[%hs](%s) - updating preview mesh"),
        __FUNCTION__,
        *InObject->GetName());
    FComponentReregisterContext reregister(PreviewMeshComponent);
  }
}

void FMochiModelEditorToolkit::NotifyPostChange(
    const FPropertyChangedEvent& PropertyChangedEvent,
    FProperty* PropertyThatChanged) {
  const FName PropName = PropertyChangedEvent.GetPropertyName();

  if (MochiModel && ModelEditorViewport) {
    auto PreviewMeshComponent = ModelEditorViewport->GetPreviewMeshComponent();
    if (PreviewMeshComponent) {
      PreviewMeshComponent->SetMaterial(0, MochiModel->PreviewRenderMaterial);
    }
  }
}

TObjectPtr<UMochiModel> FMochiModelEditorToolkit::GetModel() {
  return MochiModel;
}

void FMochiModelEditorToolkit::OnClose() {
  MochiModel = nullptr;

  if (ModelPreviewScene.IsValid()) {
    ModelPreviewScene.Reset();
  }

  if (ModelEditorViewport.IsValid()) {
    ModelEditorViewport.Reset();
  }
}

EMochiModelEditorSelectionMode FMochiModelEditorToolkit::GetViewportSelectionMode() const {
  return ModelEditorViewport->GetSelectionMode();
}

void FMochiModelEditorToolkit::SetViewportSelectionMode(
    EMochiModelEditorSelectionMode NewSelectionMode) const {
  // Note: a breakpoint here may not be hit when the mode is changed from the toolbar.
  ModelEditorViewport->SetSelectionMode(NewSelectionMode);
}

EMochiModelEditorElementViewMode FMochiModelEditorToolkit::GetViewportElementViewMode() const {
  return ModelEditorViewport->GetViewportClient()->GetElementViewMode();
}

void FMochiModelEditorToolkit::SetViewportElementViewMode(
    EMochiModelEditorElementViewMode NewViewMode) const {
  ModelEditorViewport->GetViewportClient()->SetElementViewMode(NewViewMode);
}

TSharedRef<FMochiModelEditorToolkit> FMochiModelEditorToolkit::CreateEditor(
    const EToolkitMode::Type Mode,
    const TSharedPtr<IToolkitHost>& InitToolkitHost,
    UMochiModel* ObjectToEdit) {
  UE_LOG(LogMochiPhysicsEditor, Display, TEXT("[%hs]"), __FUNCTION__);

  TSharedRef<FMochiModelEditorToolkit> NewEditor(new FMochiModelEditorToolkit());
  NewEditor->InitMochiModelEditor(Mode, InitToolkitHost, ObjectToEdit);

  return NewEditor;
}

#undef LOCTEXT_NAMESPACE
