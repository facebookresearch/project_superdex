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

#include "MochiArticulationEditor.h"
#include "MochiArticulationAssetUtils.h"
#include "MochiAssetEditorTabs.h"
#include "MochiViewportFraming.h"

#include "MochiPhysicsEditorModule.h"
#include "MochiPhysicsModule.h"
#include "MochiSkeletonHeaderExport.h"

#include "AssetEditorModeManager.h"
#include "CanvasItem.h"
#include "CanvasTypes.h"
#include "ComponentReregisterContext.h"
#include "DesktopPlatformModule.h"
#include "Eigen/Dense"
#include "InputCoreTypes.h"
#include "MochiConversions.h"
#include "PreviewProfileController.h"
#include "Rendering/SkeletalMeshRenderData.h"
#include "SEditorViewportToolBarMenu.h"
#include "UObject/UObjectGlobals.h"

#define LOCTEXT_NAMESPACE "FMochiEditorModule"

//------------------------------------------------------------------------------
// TOOLKIT
//------------------------------------------------------------------------------

const FName MochiArticulationEditorAppIdentifier = FName(TEXT("MochiArticulationEditorApp"));
const FName FMochiArticulationEditorToolkit::ViewportTabID(
    TEXT("MochiArticulationEditorToolkit_Viewport"));
const FName FMochiArticulationEditorToolkit::DetailsTabID(
    TEXT("MochiArticulationEditorToolkit_Details"));

static constexpr FColor kPurpleColor = FColor(138, 43, 226, 255);

FMochiArticulationEditorToolkit::FMochiArticulationEditorToolkit()
    : StandaloneDefaultLayout(
          MochiAssetEditorTabs::MakeDefaultLayout(
              TEXT("Standalone_MochiArticulationEditor_Layout_v1"),
              ViewportTabID,
              DetailsTabID)) {}

FMochiArticulationEditorToolkit::~FMochiArticulationEditorToolkit() {
  FCoreUObjectDelegates::OnObjectModified.RemoveAll(this);
}

FName FMochiArticulationEditorToolkit::GetToolkitFName() const {
  return FName("MochiArticulationEditor");
}

FText FMochiArticulationEditorToolkit::GetBaseToolkitName() const {
  return FText::FromString("Articulation Editor");
}

FString FMochiArticulationEditorToolkit::GetWorldCentricTabPrefix() const {
  return TEXT("Articulation ");
}

FLinearColor FMochiArticulationEditorToolkit::GetWorldCentricTabColorScale() const {
  return FLinearColor::White;
}

void FMochiArticulationEditorToolkit::RegisterTabSpawners(
    const TSharedRef<FTabManager>& InTabManager) {
  // The base class picks AssetEditorTabsCategory.
  FAssetEditorToolkit::RegisterTabSpawners(InTabManager);
  DetailsView = MochiAssetEditorTabs::MakeDetailsView(Articulation, this);
  MochiAssetEditorTabs::RegisterTabSpawners(
      *InTabManager,
      AssetEditorTabsCategory.ToSharedRef(),
      ViewportTabID,
      FOnSpawnTab::CreateSP(this, &FMochiArticulationEditorToolkit::SpawnTab_Viewport),
      DetailsTabID,
      FOnSpawnTab::CreateSP(this, &FMochiArticulationEditorToolkit::SpawnTab_Details));
}

void FMochiArticulationEditorToolkit::UnregisterTabSpawners(
    const TSharedRef<FTabManager>& InTabManager) {
  FAssetEditorToolkit::UnregisterTabSpawners(InTabManager);
  InTabManager->UnregisterTabSpawner(DetailsTabID);
  InTabManager->UnregisterTabSpawner(ViewportTabID);
}

const TSharedRef<FTabManager::FLayout> FMochiArticulationEditorToolkit::GetDefaultLayout() const {
  return StandaloneDefaultLayout.ToSharedRef();
}

void FMochiArticulationEditorToolkit::CreateEditorModeManager() {
  EditorModeManager = MakeShared<FAssetEditorModeManager>();
}

void FMochiArticulationEditorToolkit::OnClose() {
  if (Articulation) {
    Articulation->OnSkeletalMeshRegenerated = nullptr;
    Articulation = nullptr;
  }
  if (PreviewScene.IsValid()) {
    PreviewScene.Reset();
  }
  if (EditorViewport.IsValid()) {
    EditorViewport.Reset();
  }
}

void FMochiArticulationEditorToolkit::NotifyPostChange(
    const FPropertyChangedEvent& PropertyChangedEvent,
    FProperty* PropertyThatChanged) {
  // Called by the details view, whose NotifyHook is this toolkit, after a property changes.
  const FName PropName = PropertyChangedEvent.GetPropertyName();
  if (Articulation && EditorViewport) {
    auto PreviewMeshComponent = EditorViewport->GetPreviewMeshComponent();
    if (PreviewMeshComponent) {
      PreviewMeshComponent->SetArticulation(Articulation);
    }
  }
}

TObjectPtr<UMochiArticulationBase> FMochiArticulationEditorToolkit::GetArticulation() {
  return Articulation;
}

TSharedRef<SDockTab> FMochiArticulationEditorToolkit::SpawnTab_Details(const FSpawnTabArgs& Args) {
  TSharedPtr<SDockTab> DetailsTab = SNew(SDockTab).Label(LOCTEXT(
      "MochiArticulationDetailsTitle", "MochiUArticulationDetails"))[DetailsView.ToSharedRef()];
  return DetailsTab.ToSharedRef();
}

TSharedRef<SDockTab> FMochiArticulationEditorToolkit::SpawnTab_Viewport(const FSpawnTabArgs& Args) {
  TSharedRef<SDockTab> SpawnedTab = SNew(SDockTab).Label(LOCTEXT("ViewportTab_Title", "Viewport"));
  if (EditorViewport.IsValid()) {
    SpawnedTab->SetContent(EditorViewport.ToSharedRef());
    int numLinks = 0;
    int numJoints = 0;
    int numDofs = 0;
    double mass = 0;
    numLinks = Articulation->GetLinkNames().Num();
    numJoints = Articulation->GetJointNames().Num();
    numDofs = Articulation->GetNumDofs();
    int numSurfaceFaces = 0;
    if (auto asArticulation = Cast<UMochiArticulation>(Articulation)) {
      for (auto& l : asArticulation->GetLinks()) {
        mass += l.bOverrideMass ? l.Mass : 0;
        numSurfaceFaces += l.Shape ? l.Shape->NumSurfaceFaces : 0;
      }
    } else {
      Articulation->ComputeOverlayStats(mass, numSurfaceFaces);
    }
    TArray<FText> overlayItems;
    overlayItems.Emplace(
        FText::Format(LOCTEXT("Links_F", "Links:  {0}"), FText::AsNumber(numLinks)));
    overlayItems.Emplace(
        FText::Format(LOCTEXT("Joints_F", "Joints:  {0}"), FText::AsNumber(numJoints)));
    overlayItems.Emplace(FText::Format(LOCTEXT("Dofs_F", "DOF:  {0}"), FText::AsNumber(numDofs)));
    overlayItems.Emplace(FText::Format(LOCTEXT("Mass_F", "Mass:  {0} kg"), FText::AsNumber(mass)));
    overlayItems.Emplace(
        FText::Format(
            LOCTEXT("Faces_F", "Mochi Surface Faces:  {0}"), FText::AsNumber(numSurfaceFaces)));
    UMochiArticulationPreviewComponent* PreviewMeshComponent =
        EditorViewport->GetPreviewMeshComponent();
    if (PreviewMeshComponent) {
      auto skm = PreviewMeshComponent->GetSkinnedAsset();
      if (skm) {
        const FSkeletalMeshRenderData* renderData = skm->GetResourceForRendering();
        const FSkeletalMeshLODRenderData& lod0 = renderData->LODRenderData[0];
        int numVerts = lod0.GetNumVertices();
        int numTris = 0;
        for (const FSkelMeshRenderSection& section : lod0.RenderSections) {
          numTris += section.NumTriangles;
        }
        overlayItems.Emplace(
            FText::Format(
                NSLOCTEXT("UnrealEd", "Triangles_F", "Skeletal Mesh Triangles:  {0}"), numTris));
        overlayItems.Emplace(
            FText::Format(
                NSLOCTEXT("UnrealEd", "Vertices_F", "Skeletal Mesh Vertices:  {0}"), numVerts));
      }
      FBoxSphereBounds bounds;
      if (!Articulation->ComputeVisualBoundsFromJointPositions(
              Articulation->GetDefaultJointPositions(), bounds)) {
        bounds = PreviewMeshComponent->GetLocalBounds();
      }
      FVector box = bounds.BoxExtent * 2.0;
      FNumberFormattingOptions opt;
      opt.SetMaximumFractionalDigits(2);
      overlayItems.Emplace(
          FText::Format(
              LOCTEXT("ApproxSize_F", "Approx Size:  {0} x {1} x {2} cm"),
              FText::AsNumber(box.X, &opt),
              FText::AsNumber(box.Y, &opt),
              FText::AsNumber(box.Z, &opt)));
    }
    EditorViewport->PopulateOverlayText(overlayItems);
  }
  return SpawnedTab;
}

void FMochiArticulationEditorToolkit::BindCommands() {
  FMochiArticulationEditorCommands::Register();

  const FMochiArticulationEditorCommands& Commands = FMochiArticulationEditorCommands::Get();

  ToolkitCommands->MapAction(
      Commands.FocusViewport,
      FExecuteAction::CreateSP(this, &FMochiArticulationEditorToolkit::FocusViewport));

  ToolkitCommands->MapAction(
      Commands.ToggleScreenPercentage,
      FExecuteAction::CreateSP(this, &FMochiArticulationEditorToolkit::ToggleScreenPercentage));

  ToolkitCommands->MapAction(
      Commands.Export,
      FExecuteAction::CreateSP(this, &FMochiArticulationEditorToolkit::Export),
      FCanExecuteAction::CreateSP(this, &FMochiArticulationEditorToolkit::CanExport));

  ToolkitCommands->MapAction(
      Commands.GenerateSkeletalMesh,
      FExecuteAction::CreateSP(this, &FMochiArticulationEditorToolkit::GenerateSkeletalMesh),
      FCanExecuteAction::CreateSP(this, &FMochiArticulationEditorToolkit::CanGenerateSkeletalMesh));

  ToolkitCommands->MapAction(
      Commands.GenerateSkeletonHeader,
      FExecuteAction::CreateSP(this, &FMochiArticulationEditorToolkit::GenerateSkeletonHeader),
      FCanExecuteAction::CreateSP(
          this, &FMochiArticulationEditorToolkit::CanGenerateSkeletonHeader));
}

TSharedPtr<FMochiArticulationEditorPreviewScene>
FMochiArticulationEditorToolkit::CreatePreviewScene() {
  if (!PreviewScene.IsValid()) {
    PreviewScene = MakeShareable(new FMochiArticulationEditorPreviewScene(
        FPreviewScene::ConstructionValues()
            .AllowAudioPlayback(true)
            .ShouldSimulatePhysics(true)
            .ForceUseMovementComponentInNonGameWorld(true),
        StaticCastSharedRef<FMochiArticulationEditorToolkit>(AsShared())));
    PreviewScene->SetProfileIndex(1);
  }
  return PreviewScene;
}

void FMochiArticulationEditorToolkit::ExtendToolbars() {
  struct Local {
    static void
    FillToolbar(FToolBarBuilder& ToolBarBuilder, bool bShowGenerateSkeletalMesh, bool bShowExport) {
      ToolBarBuilder.BeginSection("ExtendToolbarItem");
      {
        ToolBarBuilder.AddToolBarButton(
            FMochiArticulationEditorCommands::Get().FocusViewport,
            NAME_None,
            LOCTEXT("FocusViewport", "Focus Viewport"),
            LOCTEXT("FocusViewportTooltip", "Focuses viewport on articulation"),
            FSlateIcon(FAppStyle::GetAppStyleSetName(), "ClassIcon.CameraComponent"));
        ToolBarBuilder.AddToolBarButton(
            FMochiArticulationEditorCommands::Get().ToggleScreenPercentage,
            NAME_None,
            LOCTEXT("ToggleScreenPercentage", "Toggle Screen Percentage"),
            LOCTEXT(
                "ToggleScreenPercentageTip",
                "Toggles between 200% and 100% rendering resolution (for screenshots)"),
            FSlateIcon(FAppStyle::GetAppStyleSetName(), "ClassIcon.SceneCaptureComponent"));
        ToolBarBuilder.AddSeparator();
        if (bShowGenerateSkeletalMesh) {
          ToolBarBuilder.AddToolBarButton(
              FMochiArticulationEditorCommands::Get().GenerateSkeletalMesh,
              NAME_None,
              LOCTEXT("GenerateSkeletalMesh", "Generate Skeletal Mesh"),
              LOCTEXT(
                  "GenerateSkeletalMeshTooltip",
                  "Regenerate the articulation's skeleton and skeletal mesh from its links"),
              FSlateIcon(FAppStyle::GetAppStyleSetName(), "ClassIcon.SkeletalMesh"));
        }
        ToolBarBuilder.AddToolBarButton(
            FMochiArticulationEditorCommands::Get().GenerateSkeletonHeader,
            NAME_None,
            LOCTEXT("GenerateSkeletonHeader", "Generate Skeleton Header"),
            LOCTEXT(
                "GenerateSkeletonHeaderTooltip",
                "Write a SuperDexSkeleton header (bone enum, names and parents) for the skeletal mesh"),
            FSlateIcon(FAppStyle::GetAppStyleSetName(), "ClassIcon.Skeleton"));
        if (bShowExport) {
          ToolBarBuilder.AddToolBarButton(
              FMochiArticulationEditorCommands::Get().Export,
              NAME_None,
              LOCTEXT("Export", "Export URDF"),
              LOCTEXT("ExportTooltip", "Export the articulation asset to a .urdf XML package"),
              FSlateIcon(FAppStyle::GetAppStyleSetName(), "ClassIcon.StringTable"));
        }
      }
      ToolBarBuilder.EndSection();
    }
  };
  const TSharedPtr<FExtender> ToolBarExtender = MakeShareable(new FExtender);
  ToolBarExtender->AddToolBarExtension(
      "Asset",
      EExtensionHook::After,
      GetToolkitCommands(),
      FToolBarExtensionDelegate::CreateStatic(
          &Local::FillToolbar, CanGenerateSkeletalMesh(), CanExport()));
  AddToolbarExtender(ToolBarExtender);
  FMochiPhysicsEditorModule* MochiEditorModule =
      &FModuleManager::LoadModuleChecked<FMochiPhysicsEditorModule>("SuperDexPhysicsEditor");
  AddToolbarExtender(MochiEditorModule->GetEditorToolbarExtensibilityManager()->GetAllExtenders());
  RegenerateMenusAndToolbars();
}

void FMochiArticulationEditorToolkit::FocusViewport() {
  if (EditorViewport.IsValid()) {
    // Calling Viewport's built-in method that has been overridden
    EditorViewport->OnFocusViewportToSelection();
  }
}

void FMochiArticulationEditorToolkit::ToggleScreenPercentage() const {
  if (!EditorViewport.IsValid()) {
    return;
  }

  TSharedPtr<FMochiArticulationEditorViewportClient> ViewportClient =
      EditorViewport->GetViewportClient();

  if (!ViewportClient.IsValid()) {
    return;
  }

  ViewportClient->SetPreviewScreenPercentage(
      ViewportClient->GetPreviewScreenPercentage() != kHighPreviewScreenPercentage
          ? kHighPreviewScreenPercentage
          : kNominalPreviewScreenPercentage);
}

void FMochiArticulationEditorToolkit::Export() const {
  // Exports URDF XML via the polymorphic ToUrdfXml()/GetArticulationName() hooks, with no mesh
  // re-export (e.g. USuperDexBot's mesh data is already baked into its own asset).
  IDesktopPlatform* desktopPlatform = FDesktopPlatformModule::Get();
  if (desktopPlatform) {
    const void* windowHandle =
        FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr);
    FString fileTypes = TEXT("URDF (*.urdf)|*.urdf|All files (*.*)|*.*");
    TArray<FString> savePaths;
    bool bOpened = desktopPlatform->SaveFileDialog(
        windowHandle,
        TEXT("URDF Save Location"),
        FPaths::ProjectDir(),
        Articulation->GetArticulationName() + TEXT(".urdf"),
        fileTypes,
        EFileDialogFlags::None,
        savePaths);
    if (bOpened) {
      check(savePaths.Num() == 1);
      const FString xml = Articulation->ToUrdfXml();
      if (xml.IsEmpty()) {
        UE_LOG(
            LogMochiPhysicsEditor,
            Error,
            TEXT("URDF export is not supported for this articulation; nothing was written."));
      } else {
        FFileHelper::SaveStringToFile(xml, *savePaths[0]);
      }
    }
  }
}

void FMochiArticulationEditorToolkit::GenerateSkeletalMesh() const {
  if (auto asArticulation = Cast<UMochiArticulation>(Articulation)) {
    FMochiArticulationAssetUtils::GenerateSkeletalMesh(asArticulation);
  }
}

bool FMochiArticulationEditorToolkit::CanExport() const {
  return Articulation && Articulation->SupportsUrdfExport();
}

bool FMochiArticulationEditorToolkit::CanGenerateSkeletalMesh() const {
  return Articulation && Articulation->IsA<UMochiArticulation>();
}

void FMochiArticulationEditorToolkit::GenerateSkeletonHeader() const {
  MochiSkeletonHeaderExport::Generate(Articulation);
}

bool FMochiArticulationEditorToolkit::CanGenerateSkeletonHeader() const {
  return MochiSkeletonHeaderExport::CanGenerate(Articulation);
}

void FMochiArticulationEditorToolkit::OnObjectModified(UObject* InObject) {
  if (Articulation == InObject && EditorViewport) {
    UMochiArticulationPreviewComponent* PreviewMeshComponent =
        EditorViewport->GetPreviewMeshComponent();
    PreviewMeshComponent->SetArticulation(Articulation);
    {
      FComponentReregisterContext reregister(PreviewMeshComponent);
    }
  }
}

void FMochiArticulationEditorToolkit::InitEditor(
    const EToolkitMode::Type Mode,
    const TSharedPtr<IToolkitHost>& InitToolkitHost,
    UMochiArticulationBase* ObjectToEdit) {
  // Commands should be bound to UI elements at their respective level,
  // so commands that are inside the Viewport are bound in Viewport or ViewportClient
  BindCommands();
  Articulation = Cast<UMochiArticulationBase>(ObjectToEdit);
  Articulation->OnSkeletalMeshRegenerated = [this](USkeletalMesh*) {
    if (Articulation && EditorViewport) {
      auto PreviewMeshComponent = EditorViewport->GetPreviewMeshComponent();
      if (PreviewMeshComponent) {
        // We need to call this with null so that the next call to it in SetArticulation
        // won't early out because the SKM is the same pointer address. When that
        // happens, important cached transform information doesn't get recomputed,
        // which causes crashes if the regeneration of the articulation SKM resulted in
        // new links/bones. This is a workaround, but it's the simplest solution
        // outside of manually calling the internals of SetSkinnedAssetAndUpdate.
        PreviewMeshComponent->SetSkinnedAssetAndUpdate(nullptr);
        PreviewMeshComponent->SetArticulation(Articulation);
      }
    }
  };

  // This custom viewport allows us to do deep customization
  EditorViewport = SNew(SMochiArticulationEditorViewport, SharedThis(this), CreatePreviewScene());

  const bool bCreateDefaultStandaloneMenu = true;
  const bool bCreateDefaultToolbar = true;

  InitAssetEditor(
      Mode,
      InitToolkitHost,
      MochiArticulationEditorAppIdentifier,
      GetDefaultLayout(),
      bCreateDefaultStandaloneMenu,
      bCreateDefaultToolbar,
      ObjectToEdit);

  // Top level buttons (e.g. "Reimport Base Mesh" Static Mesh Editor)
  ExtendToolbars();
  FocusViewport();
  FCoreUObjectDelegates::OnObjectModified.AddSP(
      this, &FMochiArticulationEditorToolkit::OnObjectModified);
}

TSharedRef<FMochiArticulationEditorToolkit> FMochiArticulationEditorToolkit::CreateEditor(
    const EToolkitMode::Type Mode,
    const TSharedPtr<IToolkitHost>& InitToolkitHost,
    UMochiArticulationBase* ObjectToEdit) {
  TSharedRef<FMochiArticulationEditorToolkit> NewEditor(new FMochiArticulationEditorToolkit());
  NewEditor->InitEditor(Mode, InitToolkitHost, ObjectToEdit);
  return NewEditor;
}

//------------------------------------------------------------------------------
// COMMANDS
//------------------------------------------------------------------------------

void FMochiArticulationEditorCommands::RegisterCommands() {
  UI_COMMAND(
      FocusViewport,
      "Focus Viewport",
      "Focus Viewport on Articulation",
      EUserInterfaceActionType::Button,
      FInputChord(EKeys::F));
  UI_COMMAND(
      ToggleScreenPercentage,
      "Toggle Screen Percentage",
      "Toggles between 200% and 100% rendering resolution (for screenshots)",
      EUserInterfaceActionType::Button,
      FInputChord());
  UI_COMMAND(
      Export,
      "Export URDF",
      "Export the articulation asset to a .urdf XML package",
      EUserInterfaceActionType::Button,
      FInputChord());
  UI_COMMAND(
      GenerateSkeletalMesh,
      "Generate Skeletal Mesh",
      "Regenerate the articulation's skeleton and skeletal mesh from its links",
      EUserInterfaceActionType::Button,
      FInputChord());
  UI_COMMAND(
      GenerateSkeletonHeader,
      "Generate Skeleton Header",
      "Write a SuperDexSkeleton header (bone enum, names and parents) for the skeletal mesh",
      EUserInterfaceActionType::Button,
      FInputChord());
  UI_COMMAND(
      SetShowLinkTransforms,
      "Link Transforms",
      "Show link root transforms",
      EUserInterfaceActionType::ToggleButton,
      FInputChord(EKeys::L));
  UI_COMMAND(
      SetShowJoints,
      "Joints",
      "Show joints (revolute = cylinder, prismatic = prism, fixed = point)",
      EUserInterfaceActionType::ToggleButton,
      FInputChord(EKeys::J));
  UI_COMMAND(
      SetShowBounds,
      "Bounds",
      "Show articulation bounds in current configuration",
      EUserInterfaceActionType::ToggleButton,
      FInputChord(EKeys::B));
}

//------------------------------------------------------------------------------
// VIEWPORT CLIENT
//------------------------------------------------------------------------------

constexpr FLinearColor kOrange{1.0, 0.5, 0};

FMochiArticulationEditorViewportClient::FMochiArticulationEditorViewportClient(
    const TSharedRef<SMochiArticulationEditorViewport> InEditorViewport,
    const TSharedRef<FAdvancedPreviewScene> InPreviewScene,
    UMochiArticulationBase* InArticulation,
    UMochiArticulationPreviewComponent* InPreviewMeshComponent)
    : FEditorViewportClient(
          nullptr,
          &InPreviewScene.Get(),
          StaticCastSharedRef<SMochiArticulationEditorViewport>(InEditorViewport)),
      ViewportPtr(InEditorViewport) {
  SetRealtime(true);
  DrawHelper.bDrawGrid = false;
  DrawHelper.bDrawPivot = false;
  DrawHelper.AxesLineThickness = 5;
  DrawHelper.PivotSize = 5;
  WidgetMode = UE::Widget::WM_None;
  SetViewLocation(FVector(50.0, 50.0, 50.0));
  SetViewRotation(FVector(-75.0, -75.0, 0).Rotation());
  ELevelViewportType NewViewportType = LVT_Perspective;
  FEditorViewportClient::SetViewportType(NewViewportType);
  AdvancedPreviewScene = static_cast<FAdvancedPreviewScene*>(PreviewScene);
  SetPreviewMesh(InArticulation, InPreviewMeshComponent);
  SetViewModes(VMI_Lit, VMI_Lit);
  OverrideNearClipPlane(1.0f);
  // Setup defaults for the common draw helper if user wants to show grid.
  bUsesDrawHelper = true;
  DrawHelper.bDrawPivot = false;
  DrawHelper.bDrawWorldBox = false;
  DrawHelper.bDrawKillZ = false;
  DrawHelper.bDrawGrid = true;
  DrawHelper.GridColorAxis = FColor(70, 70, 70);
  DrawHelper.GridColorMajor = FColor(40, 40, 40);
  DrawHelper.GridColorMinor = FColor(20, 20, 20);
  DrawHelper.PerspectiveGridSize = UE_OLD_HALF_WORLD_MAX1;
  // disable grid by default for clean look
  if (IsSetShowGridChecked()) {
    SetShowGrid();
  }
  // Load materials for joints, CoM, etc.
  Materials = TStrongObjectPtr(
      NewObject<UMochiArticulationEditorViewportClientMaterials>(GetTransientPackage()));
  Materials->SolidGizmoMaterial = Cast<UMaterial>(StaticLoadObject(
      UMaterial::StaticClass(),
      nullptr,
      TEXT("/SuperDexPhysics/Editor/M_MochiSolidGizmoMaterial.M_MochiSolidGizmoMaterial")));
  Materials->TransGizmoMaterial = Cast<UMaterial>(StaticLoadObject(
      UMaterial::StaticClass(),
      nullptr,
      TEXT("/SuperDexPhysics/Editor/M_MochiTransGizmoMaterial.M_MochiTransGizmoMaterial")));
  Materials->ComMaterial = UMaterialInstanceDynamic::Create(Materials->SolidGizmoMaterial, nullptr);
  Materials->ComMaterial->SetVectorParameterValue("GizmoColor", kOrange);
  Materials->JointMaterial =
      UMaterialInstanceDynamic::Create(Materials->SolidGizmoMaterial, nullptr);
  Materials->JointMaterial->SetVectorParameterValue("GizmoColor", kPurpleColor);
}

void FMochiArticulationEditorViewportClient::SetWidgetMode(UE::Widget::EWidgetMode NewMode) {
  FEditorViewportClient::SetWidgetMode(NewMode);
  WidgetMode = NewMode;
  Invalidate();
}

UE::Widget::EWidgetMode FMochiArticulationEditorViewportClient::GetWidgetMode() const {
  return UE::Widget::WM_Max;
}

bool FMochiArticulationEditorViewportClient::CanSetWidgetMode(
    UE::Widget::EWidgetMode NewMode) const {
  return true;
}

void FMochiArticulationEditorViewportClient::Draw(
    const FSceneView* View,
    FPrimitiveDrawInterface* PDI) {
  FEditorViewportClient::Draw(View, PDI);
  if (!PreviewArticulation || !PreviewMeshComponent) {
    return;
  }
  if (auto asArticulation = Cast<UMochiArticulation>(PreviewArticulation)) {
    DrawArticulation(asArticulation, View, PDI);
  } else {
    PreviewArticulation->DrawCustom(
        View,
        PDI,
        bShowLinkTransforms,
        bShowJoints,
        bShowBounds,
        Materials->JointMaterial->GetRenderProxy());
  }
}

void FMochiArticulationEditorViewportClient::DrawArticulation(
    UMochiArticulation* Articulation,
    const FSceneView* View,
    FPrimitiveDrawInterface* PDI) {
  const auto& links = Articulation->GetLinks();
  const auto& jointPos = Articulation->GetDefaultJointPositions();
  TArray<FTransform> linkTransforms;
  if (!Articulation->ComputeLinkTransformsFromJointPositions(
          jointPos, linkTransforms, EMochiArticulationLinkTransformSpace::LinkToRoot)) {
    return;
  }
  auto rootToWorld = PreviewMeshComponent->GetComponentTransform();
  for (int iLink = 0; iLink < links.Num(); ++iLink) {
    auto& link = links[iLink];
    auto& linkToRoot = linkTransforms[iLink];
    auto linkLoc = linkToRoot.GetLocation();
    if (bShowLinkTransforms) {
      FRotationMatrix R(linkToRoot.GetRotation().Rotator());
      FVector const X = R.GetScaledAxis(EAxis::X);
      FVector const Y = R.GetScaledAxis(EAxis::Y);
      FVector const Z = R.GetScaledAxis(EAxis::Z);
      float scale = 5.0f;
      PDI->DrawLine(linkLoc, linkLoc + X * scale, FLinearColor::Red, SDPG_Foreground, 0.1f, 1.0f);
      PDI->DrawLine(linkLoc, linkLoc + Y * scale, FLinearColor::Green, SDPG_Foreground, 0.1f, 1.0f);
      PDI->DrawLine(linkLoc, linkLoc + Z * scale, FLinearColor::Blue, SDPG_Foreground, 0.1f, 1.0f);
    }
    if (bShowJoints) {
      FMochiArticulatedJoint joint;
      if (Articulation->GetLinkJoint(iLink, joint)) {
        const FTransform jointToRoot = link.ParentJointFromLink.Inverse() * linkToRoot;
        const FVector jointLoc = jointToRoot.GetLocation();
        if (joint.Type == EMochiArticulatedJointType::Revolute ||
            joint.Type == EMochiArticulatedJointType::Prismatic) {
          const int sides = joint.Type == EMochiArticulatedJointType::Prismatic ? 3 : 16;
          auto axis = jointToRoot.TransformVector(joint.Axis).GetSafeNormal();
          auto transform = jointToRoot;
          transform.SetRotation(FQuat::FindBetweenNormals(FVector(0, 0, 1), axis));
          FVector X = transform.TransformVector({1.0, 0.0, 0.0});
          FVector Y = transform.TransformVector({0.0, 1.0, 0.0});
          FVector Z = transform.TransformVector({0.0, 0.0, 1.0});
          DrawCylinder(
              PDI,
              jointLoc,
              X,
              Y,
              Z,
              0.4f,
              1.5f,
              sides,
              Materials->JointMaterial->GetRenderProxy(),
              SDPG_Foreground);
        } else if (joint.Type == EMochiArticulatedJointType::Spherical) {
          DrawWireSphere(
              PDI, jointToRoot, kPurpleColor, 1.5f, 16, SDPG_Foreground, 0.1f, 0.0f, true);
        } else if (joint.Type == EMochiArticulatedJointType::Hard) {
          PDI->DrawPoint(jointLoc, kPurpleColor, 4, SDPG_Foreground);
        }
      }
    }
  }
  if (bShowBounds) {
    FBoxSphereBounds bounds;
    if (Articulation->ComputeVisualBoundsFromJointPositions(jointPos, bounds)) {
      DrawWireBox(PDI, bounds.GetBox(), kPurpleColor, SDPG_World);
      // uncomment to render sphere bounds
      // DrawCircle(PDI, bounds.Origin, {1,0,0}, {0,1,0}, kPurpleColor, bounds.SphereRadius, 32,
      // SDPG_World); DrawCircle(PDI, bounds.Origin, {0,1,0}, {0,0,1}, kPurpleColor,
      // bounds.SphereRadius, 32, SDPG_World); DrawCircle(PDI, bounds.Origin, {1,0,0}, {0,0,1},
      // kPurpleColor, bounds.SphereRadius, 32, SDPG_World);
    }
  }
}

inline void DrawWorldText(
    const FString& Text,
    const FVector& WorldLocation,
    FSceneView& View,
    FCanvas& Canvas,
    float DPIScale,
    bool bCenter) {
  if (View.ViewFrustum.IntersectPoint(WorldLocation)) {
    FVector2D screenLoc;
    if (View.WorldToPixel(WorldLocation, screenLoc)) {
      screenLoc /= DPIScale;
      screenLoc = screenLoc.RoundToVector();
      FCanvasTextItem TextItem(
          screenLoc, FText::FromString(Text), GEngine->GetSmallFont(), FLinearColor::White);
      TextItem.EnableShadow(FLinearColor::Black);
      TextItem.bCentreX = bCenter;
      TextItem.bCentreY = bCenter;
      Canvas.DrawItem(TextItem);
    }
  }
}

// Draws labels on new lines if they have the same screen space loc
struct StackedLabelDrawer {
  StackedLabelDrawer(FSceneView& View, FCanvas& Canvas, float DPIScale)
      : View(View), Canvas(Canvas), DPIScale(DPIScale) {}
  void Add(const FString& Text, const FVector& WorldLocation) {
    if (View.ViewFrustum.IntersectPoint(WorldLocation)) {
      FVector2D screenLoc;
      if (View.WorldToPixel(WorldLocation, screenLoc)) {
        screenLoc /= DPIScale;
        screenLoc = screenLoc.RoundToVector();
        TPair<int, int> xy = {(int)screenLoc.X, (int)screenLoc.Y};
        if (Labels.Contains(xy)) {
          Labels[xy].AddUnique(Text);
        } else {
          Labels.Add(xy, {Text});
        }
      }
    }
  }
  void Draw() {
    for (auto& [loc, labels] : Labels) {
      FString string;
      for (int i = 0; i < labels.Num(); ++i) {
        FString eol = i == labels.Num() - 1 ? TEXT("") : TEXT("\n");
        string += labels[i] + eol;
      }
      FCanvasTextItem TextItem(
          FVector2D(loc.Get<0>(), loc.Get<1>()),
          FText::FromString(string),
          GEngine->GetSmallFont(),
          FLinearColor::White);
      TextItem.EnableShadow(FLinearColor::Black);
      Canvas.DrawItem(TextItem);
    }
  }

 private:
  FSceneView& View;
  FCanvas& Canvas;
  float DPIScale = 1.0f;
  TMap<TPair<int, int>, TArray<FString>> Labels;
};

void FMochiArticulationEditorViewportClient::DrawCanvas(
    FViewport& InViewport,
    FSceneView& View,
    FCanvas& Canvas) {
  FEditorViewportClient::DrawCanvas(InViewport, View, Canvas);
  if (!PreviewArticulation || !PreviewMeshComponent) {
    return;
  }
  if (auto asArticulation = Cast<UMochiArticulation>(PreviewArticulation)) {
    DrawCanvasArticulation(asArticulation, InViewport, View, Canvas);
  } else {
    PreviewArticulation->DrawCanvasCustom(
        InViewport, View, Canvas, GetDPIScale(), bShowLinkTransforms, bShowJoints);
  }
}

void FMochiArticulationEditorViewportClient::DrawCanvasArticulation(
    UMochiArticulation* Articulation,
    FViewport& InViewport,
    FSceneView& View,
    FCanvas& Canvas) {
  StackedLabelDrawer drawer(View, Canvas, GetDPIScale());
  const auto& links = Articulation->GetLinks();
  const auto& joints = Articulation->GetJoints();
  const auto& jointPos = Articulation->GetDefaultJointPositions();
  TArray<FTransform> transforms;
  if (!Articulation->ComputeLinkTransformsFromJointPositions(
          jointPos, transforms, EMochiArticulationLinkTransformSpace::LinkToRoot)) {
    return;
  }
  auto rootToWorld = PreviewMeshComponent->GetComponentTransform();
  for (int iLink = 0; iLink < links.Num(); ++iLink) {
    auto& link = links[iLink];
    auto& linkToRoot = transforms[iLink];
    auto linkLoc = linkToRoot.GetLocation();
    if (bShowLinkTransforms) {
      bool bDummy = link.RenderModel == nullptr || link.Shape == nullptr;
      FString txt = bDummy ? link.Name.ToString() + TEXT("*") : link.Name.ToString();
      drawer.Add(txt, linkLoc);
    }
    if (bShowJoints) {
      FMochiArticulatedJoint joint;
      if (Articulation->GetLinkJoint(iLink, joint)) {
        drawer.Add(
            joint.Name.ToString(), (link.ParentJointFromLink.Inverse() * linkToRoot).GetLocation());
      }
    }
  }
  drawer.Draw();
}

void FMochiArticulationEditorViewportClient::AddReferencedObjects(FReferenceCollector& Collector) {
  FEditorViewportClient::AddReferencedObjects(Collector);
  Collector.AddReferencedObject(PreviewArticulation);
  Collector.AddReferencedObject(PreviewMeshComponent);
}

FString FMochiArticulationEditorViewportClient::GetReferencerName() const {
  return "FMochiArticulationEditorViewportClient";
}

void FMochiArticulationEditorViewportClient::SetPreviewMesh(
    UMochiArticulationBase* InArticulation,
    UMochiArticulationPreviewComponent* InPreviewMeshComponent) {
  PreviewArticulation = InArticulation;
  PreviewMeshComponent = InPreviewMeshComponent;
}

#ifndef MOCHI_ARTICULATION_VIEWPORT_SHOW_IMPL
#define MOCHI_ARTICULATION_VIEWPORT_SHOW_IMPL(NAME)                            \
  void FMochiArticulationEditorViewportClient::SetShow##NAME(bool bShowOn) {   \
    bShow##NAME = bShowOn;                                                     \
    Invalidate();                                                              \
  }                                                                            \
  void FMochiArticulationEditorViewportClient::ToggleShow##NAME() {            \
    SetShow##NAME(!bShow##NAME);                                               \
  }                                                                            \
  bool FMochiArticulationEditorViewportClient::IsShow##NAME##Checked() const { \
    return bShow##NAME;                                                        \
  }
#endif

MOCHI_ARTICULATION_VIEWPORT_SHOW_IMPL(LinkTransforms)
MOCHI_ARTICULATION_VIEWPORT_SHOW_IMPL(Joints);
MOCHI_ARTICULATION_VIEWPORT_SHOW_IMPL(Bounds);

//------------------------------------------------------------------------------
// VIEWPORT TOOLBAR SLATE
//------------------------------------------------------------------------------

void SMochiArticulationEditorViewportToolbar::Construct(
    const FArguments& InArgs,
    TSharedPtr<ICommonEditorViewportToolbarInfoProvider> InInfoProvider) {
  SCommonEditorViewportToolbarBase::Construct(
      SCommonEditorViewportToolbarBase::FArguments().PreviewProfileController(
          MakeShared<FPreviewProfileController>()),
      InInfoProvider);
}

void SMochiArticulationEditorViewportToolbar::ExtendLeftAlignedToolbarSlots(
    TSharedPtr<SHorizontalBox> MainBoxPtr,
    TSharedPtr<SViewportToolBar> ParentToolBarPtr) const {
  const FMargin ToolbarSlotPadding(2.0f, 2.0f);
  if (!MainBoxPtr.IsValid()) {
    return;
  }
  // Articulation menu
  MainBoxPtr->AddSlot().AutoWidth().Padding(ToolbarSlotPadding)
      [SNew(SEditorViewportToolbarMenu)
           .Label(LOCTEXT("ArticulationMenu", "Articulation"))
           .OnGetMenuContent(
               this, &SMochiArticulationEditorViewportToolbar::GenerateArticulationMenu)
           .Cursor(EMouseCursor::Default)
           .ParentToolBar(ParentToolBarPtr)];
}

TSharedPtr<FExtender> SMochiArticulationEditorViewportToolbar::GetViewMenuExtender() const {
  return SCommonEditorViewportToolbarBase::GetViewMenuExtender();
}

TSharedRef<SWidget> SMochiArticulationEditorViewportToolbar::GenerateArticulationMenu() const {
  const FMochiArticulationEditorCommands& Commands = FMochiArticulationEditorCommands::Get();
  TSharedRef<SEditorViewport> BaseViewportRef = GetInfoProvider().GetViewportWidget();
  TSharedRef<SMochiArticulationEditorViewport> ViewportRef =
      StaticCastSharedRef<SMochiArticulationEditorViewport, SEditorViewport>(BaseViewportRef);
  TSharedPtr<FExtender> MenuExtender = GetInfoProvider().GetExtenders();
  constexpr bool bShouldCloseAfterSelect = false;
  FMenuBuilder InMenuBuilder(bShouldCloseAfterSelect, ViewportRef->GetCommandList(), MenuExtender);
  InMenuBuilder.PushCommandList(ViewportRef->GetCommandList().ToSharedRef());
  if (MenuExtender.IsValid()) {
    InMenuBuilder.PushExtender(MenuExtender.ToSharedRef());
  }
  InMenuBuilder.BeginSection(
      "MochiArticulationEditorViewportSelectionTool", LOCTEXT("Links_PreviewLabel", "Links"));
  InMenuBuilder.AddMenuEntry(Commands.SetShowLinkTransforms);
  InMenuBuilder.EndSection();
  InMenuBuilder.BeginSection(
      "MochiArticulationEditorViewportSelectionTool", LOCTEXT("Joints_PreviewLabel", "Joints"));
  InMenuBuilder.AddMenuEntry(Commands.SetShowJoints);
  InMenuBuilder.EndSection();
  InMenuBuilder.BeginSection(
      "MochiArticulationEditorViewportSelectionTool", LOCTEXT("Misc_PreviewLabel", "Misc"));
  InMenuBuilder.AddMenuEntry(Commands.SetShowBounds);
  InMenuBuilder.EndSection();
  InMenuBuilder.PopCommandList();
  if (MenuExtender.IsValid()) {
    InMenuBuilder.PopExtender();
  }
  return InMenuBuilder.MakeWidget();
}

//------------------------------------------------------------------------------
// VIEWPORT SLATE
//------------------------------------------------------------------------------

void SMochiArticulationEditorViewport::Construct(
    const FArguments& InArgs,
    TSharedPtr<FMochiArticulationEditorToolkit> InEditorToolkit,
    TSharedPtr<FMochiArticulationEditorPreviewScene> InPreviewScene) {
  EditorToolkit = InEditorToolkit;
  PreviewScene = InPreviewScene;
  PreviewMeshComponent =
      NewObject<UMochiArticulationPreviewComponent>(GetTransientPackage(), NAME_None, RF_Transient);
  const TObjectPtr<UMochiArticulationBase> articulation = EditorToolkit->GetArticulation();
  PreviewScene->AddComponent(PreviewMeshComponent, FTransform::Identity);
  PreviewMeshComponent->SetArticulation(articulation);
  const auto skelMesh = PreviewMeshComponent->GetSkinnedAsset();
  if (skelMesh) {
    PreviewScene->SetFloorOffset(
        static_cast<float>(-skelMesh->GetBounds().Origin.Z + skelMesh->GetBounds().BoxExtent.Z));
  }
  SEditorViewport::Construct(SEditorViewport::FArguments());
}

SMochiArticulationEditorViewport::~SMochiArticulationEditorViewport() {
  if (ViewportClient.IsValid()) {
    ViewportClient->Viewport = nullptr;
  }
}

void SMochiArticulationEditorViewport::BindCommands() {
  SAssetEditorViewport::BindCommands();
  const FMochiArticulationEditorCommands& Commands = FMochiArticulationEditorCommands::Get();
  const TSharedRef<FMochiArticulationEditorViewportClient> EditorViewportClient =
      ViewportClient.ToSharedRef();
  // LinkTransform
  CommandList->MapAction(
      Commands.SetShowLinkTransforms,
      FExecuteAction::CreateSP(
          EditorViewportClient, &FMochiArticulationEditorViewportClient::ToggleShowLinkTransforms),
      FCanExecuteAction(),
      FIsActionChecked::CreateSP(
          EditorViewportClient,
          &FMochiArticulationEditorViewportClient::IsShowLinkTransformsChecked));
  // JointDofs
  CommandList->MapAction(
      Commands.SetShowJoints,
      FExecuteAction::CreateSP(
          EditorViewportClient, &FMochiArticulationEditorViewportClient::ToggleShowJoints),
      FCanExecuteAction(),
      FIsActionChecked::CreateSP(
          EditorViewportClient, &FMochiArticulationEditorViewportClient::IsShowJointsChecked));
  // Bounds
  CommandList->MapAction(
      Commands.SetShowBounds,
      FExecuteAction::CreateSP(
          EditorViewportClient, &FMochiArticulationEditorViewportClient::ToggleShowBounds),
      FCanExecuteAction(),
      FIsActionChecked::CreateSP(
          EditorViewportClient, &FMochiArticulationEditorViewportClient::IsShowBoundsChecked));
}

TSharedRef<SEditorViewport> SMochiArticulationEditorViewport::GetViewportWidget() {
  return SharedThis(this);
}

TSharedPtr<FExtender> SMochiArticulationEditorViewport::GetExtenders() const {
  TSharedPtr<FExtender> Result(MakeShareable(new FExtender));
  return Result;
}

void SMochiArticulationEditorViewport::OnFloatingButtonClicked() {}

void SMochiArticulationEditorViewport::OnFocusViewportToSelection() {
  SAssetEditorViewport::OnFocusViewportToSelection();
  if (PreviewScene.IsValid() && EditorToolkit->GetArticulation()) {
    const auto& jointPos = EditorToolkit->GetArticulation()->GetDefaultJointPositions();
    FBoxSphereBounds bounds;
    if (EditorToolkit->GetArticulation()->ComputeVisualBoundsFromJointPositions(jointPos, bounds)) {
      MochiViewportFraming::FocusOnBounds(*ViewportClient, bounds, false);
    } else if (PreviewMeshComponent) {
      MochiViewportFraming::FocusOnBounds(*ViewportClient, PreviewMeshComponent->Bounds, false);
    }
  }
}

TSharedRef<FEditorViewportClient> SMochiArticulationEditorViewport::MakeEditorViewportClient() {
  UMochiArticulationBase* articulation = EditorToolkit->GetArticulation();
  ViewportClient = MakeShareable(new FMochiArticulationEditorViewportClient(
      SharedThis(this), PreviewScene.ToSharedRef(), articulation, PreviewMeshComponent));
  ViewportClient->SetPreviewingScreenPercentage(!ViewportClient->IsPreviewingScreenPercentage());
  ViewportClient->SetPreviewScreenPercentage(
      FMochiArticulationEditorToolkit::kNominalPreviewScreenPercentage);
  ViewportClient->Invalidate();
  return ViewportClient.ToSharedRef();
}

TSharedRef<SWidget> SMochiArticulationEditorViewport::MakeLegacyViewportToolbar() {
  return SNew(SMochiArticulationEditorViewportToolbar, SharedThis(this));
}

#if UE_VERSION_OLDER_THAN(5, 7, 0)
TSharedPtr<SWidget> SMochiArticulationEditorViewport::MakeViewportToolbar() {
  return MakeLegacyViewportToolbar();
}
#endif

void SMochiArticulationEditorViewport::PopulateViewportOverlays(TSharedRef<SOverlay> Overlay) {
  SEditorViewport::PopulateViewportOverlays(Overlay);
#if !UE_VERSION_OLDER_THAN(5, 7, 0)
  // Placed here by SEditorViewport itself before UE 5.7.
  Overlay->AddSlot().VAlign(VAlign_Top)[MakeLegacyViewportToolbar()];
#endif
  Overlay->AddSlot()
      .VAlign(VAlign_Top)
      .HAlign(HAlign_Left)
      .Padding(FMargin(
          6.0f, 36.0f, 6.0f, 6.0f))[SNew(SBorder)
                                        .BorderImage(FAppStyle::Get().GetBrush("FloatingBorder"))
                                        .Padding(4.f)[SAssignNew(OverlayText, SRichTextBlock)]];
}

void SMochiArticulationEditorViewport::AddReferencedObjects(FReferenceCollector& Collector) {
  Collector.AddReferencedObject(PreviewMeshComponent);
}

FString SMochiArticulationEditorViewport::GetReferencerName() const {
  return "SMochiArticulationEditorViewport";
}

void SMochiArticulationEditorViewport::PopulateOverlayText(const TArray<FText>& TextItems) const {
  FTextBuilder FinalText;
  for (FText const& item : TextItems) {
    FinalText.AppendLine(item);
  }
  OverlayText->SetText(FinalText.ToText());
}

TSharedPtr<FMochiArticulationEditorViewportClient>
SMochiArticulationEditorViewport::GetViewportClient() {
  return ViewportClient;
}

UMochiArticulationPreviewComponent* SMochiArticulationEditorViewport::GetPreviewMeshComponent() {
  return PreviewMeshComponent;
}

//------------------------------------------------------------------------------
// PREVIEW SCENE
//------------------------------------------------------------------------------

FMochiArticulationEditorPreviewScene::FMochiArticulationEditorPreviewScene(
    ConstructionValues CVS,
    const TSharedRef<FMochiArticulationEditorToolkit>& EditorToolkit)
    : FAdvancedPreviewScene(CVS), EditorToolkitPtr(EditorToolkit) {
  AWorldSettings* WorldSettings = GetWorld()->GetWorldSettings(true);
  WorldSettings->bEnableWorldBoundsChecks = false;
  SetFloorVisibility(false, true);
}

FMochiArticulationEditorPreviewScene::~FMochiArticulationEditorPreviewScene() {}

void FMochiArticulationEditorPreviewScene::Tick(float DeltaTime) {
  FAdvancedPreviewScene::Tick(DeltaTime);
}

#undef LOCTEXT_NAMESPACE
