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

static constexpr FColor PurpleColor = FColor(138, 43, 226, 255);

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
    UMochiArticulationPreviewComponent* PreviewMeshComponent =
        EditorViewport->GetPreviewMeshComponent();
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
    int NumLinks = 0;
    int NumJoints = 0;
    int NumDofs = 0;
    double Mass = 0;
    NumLinks = Articulation->GetLinkNames().Num();
    NumJoints = Articulation->GetJointNames().Num();
    NumDofs = Articulation->GetNumDofs();
    int NumSurfaceFaces = 0;
    if (UMochiArticulation* AsArticulation = Cast<UMochiArticulation>(Articulation)) {
      for (const FMochiArticulatedLink& L : AsArticulation->GetLinks()) {
        Mass += L.bOverrideMass ? L.Mass : 0;
        NumSurfaceFaces += L.Shape ? L.Shape->NumSurfaceFaces : 0;
      }
    } else {
      Articulation->ComputeOverlayStats(Mass, NumSurfaceFaces);
    }
    TArray<FText> OverlayItems;
    OverlayItems.Emplace(
        FText::Format(LOCTEXT("Links_F", "Links:  {0}"), FText::AsNumber(NumLinks)));
    OverlayItems.Emplace(
        FText::Format(LOCTEXT("Joints_F", "Joints:  {0}"), FText::AsNumber(NumJoints)));
    OverlayItems.Emplace(FText::Format(LOCTEXT("Dofs_F", "DOF:  {0}"), FText::AsNumber(NumDofs)));
    OverlayItems.Emplace(FText::Format(LOCTEXT("Mass_F", "Mass:  {0} kg"), FText::AsNumber(Mass)));
    OverlayItems.Emplace(
        FText::Format(
            LOCTEXT("Faces_F", "Mochi Surface Faces:  {0}"), FText::AsNumber(NumSurfaceFaces)));
    UMochiArticulationPreviewComponent* PreviewMeshComponent =
        EditorViewport->GetPreviewMeshComponent();
    if (PreviewMeshComponent) {
      USkinnedAsset* Skm = PreviewMeshComponent->GetSkinnedAsset();
      if (Skm) {
        const FSkeletalMeshRenderData* RenderData = Skm->GetResourceForRendering();
        const FSkeletalMeshLODRenderData& Lod0 = RenderData->LODRenderData[0];
        int NumVerts = Lod0.GetNumVertices();
        int NumTris = 0;
        for (const FSkelMeshRenderSection& Section : Lod0.RenderSections) {
          NumTris += Section.NumTriangles;
        }
        OverlayItems.Emplace(
            FText::Format(
                NSLOCTEXT("UnrealEd", "Triangles_F", "Skeletal Mesh Triangles:  {0}"), NumTris));
        OverlayItems.Emplace(
            FText::Format(
                NSLOCTEXT("UnrealEd", "Vertices_F", "Skeletal Mesh Vertices:  {0}"), NumVerts));
      }
      FBoxSphereBounds Bounds;
      if (!Articulation->ComputeVisualBoundsFromJointPositions(
              Articulation->GetDefaultJointPositions(), Bounds)) {
        Bounds = PreviewMeshComponent->GetLocalBounds();
      }
      FVector Box = Bounds.BoxExtent * 2.0;
      FNumberFormattingOptions Opt;
      Opt.SetMaximumFractionalDigits(2);
      OverlayItems.Emplace(
          FText::Format(
              LOCTEXT("ApproxSize_F", "Approx Size:  {0} x {1} x {2} cm"),
              FText::AsNumber(Box.X, &Opt),
              FText::AsNumber(Box.Y, &Opt),
              FText::AsNumber(Box.Z, &Opt)));
    }
    EditorViewport->PopulateOverlayText(OverlayItems);
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
  struct FLocal {
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
          &FLocal::FillToolbar, CanGenerateSkeletalMesh(), CanExport()));
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
  IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
  if (DesktopPlatform) {
    const void* WindowHandle =
        FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr);
    FString FileTypes = TEXT("URDF (*.urdf)|*.urdf|All files (*.*)|*.*");
    TArray<FString> SavePaths;
    bool bOpened = DesktopPlatform->SaveFileDialog(
        WindowHandle,
        TEXT("URDF Save Location"),
        FPaths::ProjectDir(),
        Articulation->GetArticulationName() + TEXT(".urdf"),
        FileTypes,
        EFileDialogFlags::None,
        SavePaths);
    if (bOpened) {
      check(SavePaths.Num() == 1);
      const FString Xml = Articulation->ToUrdfXml();
      if (Xml.IsEmpty()) {
        UE_LOG(
            LogMochiPhysicsEditor,
            Error,
            TEXT("URDF export is not supported for this articulation; nothing was written."));
      } else {
        FFileHelper::SaveStringToFile(Xml, *SavePaths[0]);
      }
    }
  }
}

void FMochiArticulationEditorToolkit::GenerateSkeletalMesh() const {
  if (UMochiArticulation* AsArticulation = Cast<UMochiArticulation>(Articulation)) {
    FMochiArticulationAssetUtils::GenerateSkeletalMesh(AsArticulation);
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
      FComponentReregisterContext Reregister(PreviewMeshComponent);
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
      UMochiArticulationPreviewComponent* PreviewMeshComponent =
          EditorViewport->GetPreviewMeshComponent();
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

constexpr FLinearColor Orange{1.0, 0.5, 0};

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
  // Fix the exposure rather than inherit the host project's auto exposure: against the mostly empty
  // preview background, eye adaptation keeps brightening the bot until it washes out. EV100 -1
  // matches a project with r.DefaultFeature.AutoExposure=False and the default exposure bias of 1.
  ExposureSettings.bFixed = true;
  ExposureSettings.FixedEV100 = -1.0f;
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
  Materials->ComMaterial->SetVectorParameterValue("GizmoColor", Orange);
  Materials->JointMaterial =
      UMaterialInstanceDynamic::Create(Materials->SolidGizmoMaterial, nullptr);
  Materials->JointMaterial->SetVectorParameterValue("GizmoColor", PurpleColor);
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
  if (UMochiArticulation* AsArticulation = Cast<UMochiArticulation>(PreviewArticulation)) {
    DrawArticulation(AsArticulation, View, PDI);
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
  const TArray<FMochiArticulatedLink>& Links = Articulation->GetLinks();
  const TArray<double>& JointPos = Articulation->GetDefaultJointPositions();
  TArray<FTransform> LinkTransforms;
  if (!Articulation->ComputeLinkToArticulationRootTransforms(JointPos, LinkTransforms)) {
    return;
  }
  FTransform RootToWorld = PreviewMeshComponent->GetComponentTransform();
  for (int LinkIndex = 0; LinkIndex < Links.Num(); ++LinkIndex) {
    const FMochiArticulatedLink& Link = Links[LinkIndex];
    FTransform& LinkToRoot = LinkTransforms[LinkIndex];
    FVector LinkLoc = LinkToRoot.GetLocation();
    if (bShowLinkTransforms) {
      FRotationMatrix R(LinkToRoot.GetRotation().Rotator());
      FVector const X = R.GetScaledAxis(EAxis::X);
      FVector const Y = R.GetScaledAxis(EAxis::Y);
      FVector const Z = R.GetScaledAxis(EAxis::Z);
      float Scale = 5.0f;
      PDI->DrawLine(LinkLoc, LinkLoc + X * Scale, FLinearColor::Red, SDPG_Foreground, 0.1f, 1.0f);
      PDI->DrawLine(LinkLoc, LinkLoc + Y * Scale, FLinearColor::Green, SDPG_Foreground, 0.1f, 1.0f);
      PDI->DrawLine(LinkLoc, LinkLoc + Z * Scale, FLinearColor::Blue, SDPG_Foreground, 0.1f, 1.0f);
    }
    if (bShowJoints) {
      FMochiArticulatedJoint Joint;
      if (Articulation->GetLinkJoint(LinkIndex, Joint)) {
        const FTransform JointToRoot = Link.ParentJointFromLink.Inverse() * LinkToRoot;
        const FVector JointLoc = JointToRoot.GetLocation();
        if (Joint.Type == EMochiArticulatedJointType::Revolute ||
            Joint.Type == EMochiArticulatedJointType::Prismatic) {
          const int Sides = Joint.Type == EMochiArticulatedJointType::Prismatic ? 3 : 16;
          FVector Axis = JointToRoot.TransformVector(Joint.Axis).GetSafeNormal();
          FTransform Transform = JointToRoot;
          Transform.SetRotation(FQuat::FindBetweenNormals(FVector(0, 0, 1), Axis));
          FVector X = Transform.TransformVector({1.0, 0.0, 0.0});
          FVector Y = Transform.TransformVector({0.0, 1.0, 0.0});
          FVector Z = Transform.TransformVector({0.0, 0.0, 1.0});
          DrawCylinder(
              PDI,
              JointLoc,
              X,
              Y,
              Z,
              0.4f,
              1.5f,
              Sides,
              Materials->JointMaterial->GetRenderProxy(),
              SDPG_Foreground);
        } else if (Joint.Type == EMochiArticulatedJointType::Spherical) {
          DrawWireSphere(
              PDI, JointToRoot, PurpleColor, 1.5f, 16, SDPG_Foreground, 0.1f, 0.0f, true);
        } else if (Joint.Type == EMochiArticulatedJointType::Hard) {
          PDI->DrawPoint(JointLoc, PurpleColor, 4, SDPG_Foreground);
        }
      }
    }
  }
  if (bShowBounds) {
    FBoxSphereBounds Bounds;
    if (Articulation->ComputeVisualBoundsFromJointPositions(JointPos, Bounds)) {
      DrawWireBox(PDI, Bounds.GetBox(), PurpleColor, SDPG_World);
      // uncomment to render sphere bounds
      // DrawCircle(PDI, Bounds.Origin, {1,0,0}, {0,1,0}, PurpleColor, Bounds.SphereRadius, 32,
      // SDPG_World); DrawCircle(PDI, Bounds.Origin, {0,1,0}, {0,0,1}, PurpleColor,
      // Bounds.SphereRadius, 32, SDPG_World); DrawCircle(PDI, Bounds.Origin, {1,0,0}, {0,0,1},
      // PurpleColor, Bounds.SphereRadius, 32, SDPG_World);
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
    FVector2D ScreenLoc;
    if (View.WorldToPixel(WorldLocation, ScreenLoc)) {
      ScreenLoc /= DPIScale;
      ScreenLoc = ScreenLoc.RoundToVector();
      FCanvasTextItem TextItem(
          ScreenLoc, FText::FromString(Text), GEngine->GetSmallFont(), FLinearColor::White);
      TextItem.EnableShadow(FLinearColor::Black);
      TextItem.bCentreX = bCenter;
      TextItem.bCentreY = bCenter;
      Canvas.DrawItem(TextItem);
    }
  }
}

// Draws labels on new lines if they have the same screen space loc
struct FStackedLabelDrawer {
  FStackedLabelDrawer(FSceneView& View, FCanvas& Canvas, float DPIScale)
      : View(View), Canvas(Canvas), DPIScale(DPIScale) {}
  void Add(const FString& Text, const FVector& WorldLocation) {
    if (View.ViewFrustum.IntersectPoint(WorldLocation)) {
      FVector2D ScreenLoc;
      if (View.WorldToPixel(WorldLocation, ScreenLoc)) {
        ScreenLoc /= DPIScale;
        ScreenLoc = ScreenLoc.RoundToVector();
        TPair<int, int> Xy = {(int)ScreenLoc.X, (int)ScreenLoc.Y};
        if (Labels.Contains(Xy)) {
          Labels[Xy].AddUnique(Text);
        } else {
          Labels.Add(Xy, {Text});
        }
      }
    }
  }
  void Draw() {
    for (TPair<TPair<int, int>, TArray<FString>>& LabelPair : Labels) {
      TPair<int, int>& Loc = LabelPair.Key;
      TArray<FString>& PositionLabels = LabelPair.Value;
      FString String;
      for (int i = 0; i < PositionLabels.Num(); ++i) {
        FString Eol = i == PositionLabels.Num() - 1 ? TEXT("") : TEXT("\n");
        String += PositionLabels[i] + Eol;
      }
      FCanvasTextItem TextItem(
          FVector2D(Loc.Get<0>(), Loc.Get<1>()),
          FText::FromString(String),
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
  if (UMochiArticulation* AsArticulation = Cast<UMochiArticulation>(PreviewArticulation)) {
    DrawCanvasArticulation(AsArticulation, InViewport, View, Canvas);
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
  FStackedLabelDrawer Drawer(View, Canvas, GetDPIScale());
  const TArray<FMochiArticulatedLink>& Links = Articulation->GetLinks();
  const TArray<FMochiArticulatedJoint>& Joints = Articulation->GetJoints();
  const TArray<double>& JointPos = Articulation->GetDefaultJointPositions();
  TArray<FTransform> Transforms;
  if (!Articulation->ComputeLinkToArticulationRootTransforms(JointPos, Transforms)) {
    return;
  }
  FTransform RootToWorld = PreviewMeshComponent->GetComponentTransform();
  for (int LinkIndex = 0; LinkIndex < Links.Num(); ++LinkIndex) {
    const FMochiArticulatedLink& Link = Links[LinkIndex];
    FTransform& LinkToRoot = Transforms[LinkIndex];
    FVector LinkLoc = LinkToRoot.GetLocation();
    if (bShowLinkTransforms) {
      bool bDummy = Link.RenderModel == nullptr || Link.Shape == nullptr;
      FString Txt = bDummy ? Link.Name.ToString() + TEXT("*") : Link.Name.ToString();
      Drawer.Add(Txt, LinkLoc);
    }
    if (bShowJoints) {
      FMochiArticulatedJoint Joint;
      if (Articulation->GetLinkJoint(LinkIndex, Joint)) {
        Drawer.Add(
            Joint.Name.ToString(), (Link.ParentJointFromLink.Inverse() * LinkToRoot).GetLocation());
      }
    }
  }
  Drawer.Draw();
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
  const TObjectPtr<UMochiArticulationBase> Articulation = EditorToolkit->GetArticulation();
  PreviewScene->AddComponent(PreviewMeshComponent, FTransform::Identity);
  PreviewMeshComponent->SetArticulation(Articulation);
  USkinnedAsset* const SkelMesh = PreviewMeshComponent->GetSkinnedAsset();
  if (SkelMesh) {
    PreviewScene->SetFloorOffset(
        static_cast<float>(-SkelMesh->GetBounds().Origin.Z + SkelMesh->GetBounds().BoxExtent.Z));
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
    const TArray<double>& JointPos = EditorToolkit->GetArticulation()->GetDefaultJointPositions();
    FBoxSphereBounds Bounds;
    if (EditorToolkit->GetArticulation()->ComputeVisualBoundsFromJointPositions(JointPos, Bounds)) {
      MochiViewportFraming::FocusOnBounds(*ViewportClient, Bounds, false);
    } else if (PreviewMeshComponent) {
      MochiViewportFraming::FocusOnBounds(*ViewportClient, PreviewMeshComponent->Bounds, false);
    }
  }
}

TSharedRef<FEditorViewportClient> SMochiArticulationEditorViewport::MakeEditorViewportClient() {
  UMochiArticulationBase* Articulation = EditorToolkit->GetArticulation();
  ViewportClient = MakeShareable(new FMochiArticulationEditorViewportClient(
      SharedThis(this), PreviewScene.ToSharedRef(), Articulation, PreviewMeshComponent));
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
  for (FText const& Item : TextItems) {
    FinalText.AppendLine(Item);
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
