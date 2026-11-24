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

#pragma once

#include "Misc/EngineVersionComparison.h"
#include "MochiArticulation.h"
#include "MochiArticulationPreviewComponent.h"

#include "AdvancedPreviewScene.h"
#include "CoreMinimal.h"
#include "Misc/NotifyHook.h"
#include "SAssetEditorViewport.h"
#include "SCommonEditorViewportToolbarBase.h"
#include "SEditorViewport.h"
#include "UObject/Object.h"
#include "Widgets/Text/SRichTextBlock.h"

#include "MochiArticulationEditor.generated.h"

//------------------------------------------------------------------------------
// FORWARDS
//------------------------------------------------------------------------------

class FMochiArticulationEditorToolkit;
class FMochiArticulationEditorCommands;
class FMochiArticulationEditorViewportClient;
class SMochiArticulationEditorViewportToolbar;
class SMochiArticulationEditorViewport;
class FMochiArticulationEditorPreviewScene;

//------------------------------------------------------------------------------
// TOOLKIT (ENTRY POINT)
//------------------------------------------------------------------------------

/** Asset editor for articulation definitions and previews. */
class SUPERDEXPHYSICSEDITOR_API FMochiArticulationEditorToolkit : public FAssetEditorToolkit,
                                                                  public FNotifyHook {
 private:
  using Super = FAssetEditorToolkit;

 public:
  /** Initialize the articulation asset editor toolkit. */
  FMochiArticulationEditorToolkit();
  virtual ~FMochiArticulationEditorToolkit() override;

 public:
  // Begin FAssetEditorToolkit Interface
  virtual FName GetToolkitFName() const override;
  virtual FText GetBaseToolkitName() const override;
  virtual FString GetWorldCentricTabPrefix() const override;
  virtual FLinearColor GetWorldCentricTabColorScale() const override;
  virtual void RegisterTabSpawners(const TSharedRef<class FTabManager>& InTabManager) override;
  virtual void UnregisterTabSpawners(const TSharedRef<class FTabManager>& InTabManager) override;
  /**
   * Get the articulation editor's tab layout.
   *
   * @return The default layout.
   */
  const TSharedRef<FTabManager::FLayout> GetDefaultLayout() const;
  virtual void CreateEditorModeManager() override;
  virtual void OnClose() override;
  // End FAssetEditorToolkit Interface

 public:
  // Begin FNotifyHook Interface
  virtual void NotifyPostChange(
      const FPropertyChangedEvent& PropertyChangedEvent,
      FProperty* PropertyThatChanged) override;
  // End FNotifyHookInterface

 public:
  /**
   * Get the articulation asset being edited.
   *
   * @return The articulation.
   */
  TObjectPtr<UMochiArticulationBase> GetArticulation();
  /**
   * Create the articulation-details tab.
   *
   * @param Args Tab-spawning arguments.
   * @return The tab containing articulation details.
   */
  TSharedRef<SDockTab> SpawnTab_Details(const FSpawnTabArgs& Args);
  /**
   * Create the articulation-preview tab.
   *
   * @param Args Tab-spawning arguments.
   * @return The tab containing the articulation preview.
   */
  TSharedRef<SDockTab> SpawnTab_Viewport(const FSpawnTabArgs& Args);
  /** Bind articulation-editor actions to commands. */
  void BindCommands();
  /**
   * Create the articulation editor's preview scene.
   *
   * @return The object returned by creation.
   */
  TSharedPtr<FMochiArticulationEditorPreviewScene> CreatePreviewScene();
  /** Add articulation-editor toolbar extensions. */
  void ExtendToolbars();
  /** Focus the articulation preview viewport. */
  void FocusViewport();
  /** Toggle the preview screen-percentage setting. */
  void ToggleScreenPercentage() const;
  /** Export the articulation as URDF. */
  void Export() const;
  /**
   * Get whether the articulation can be exported as URDF.
   *
   * @return Whether the articulation supports URDF export.
   */
  bool CanExport() const;
  /** Generate the articulation's skeletal mesh. */
  void GenerateSkeletalMesh() const;
  /**
   * Get whether skeletal-mesh generation is available.
   *
   * @return Whether the editor holds an asset of type UMochiArticulation.
   */
  bool CanGenerateSkeletalMesh() const;
  /** Generate a source header describing the articulation skeleton. */
  void GenerateSkeletonHeader() const;
  /**
   * Get whether skeleton-header generation is available.
   *
   * @return Whether skeleton-header generation is available for the articulation.
   */
  bool CanGenerateSkeletonHeader() const;
  /**
   * Handle modifications to objects used by the editor.
   *
   * @param InObject Object whose modification is being handled.
   */
  void OnObjectModified(UObject* InObject);
  /**
   * Open the articulation asset in the requested editor mode and host.
   *
   * @param Mode Editor or prefab import mode.
   * @param InitToolkitHost Host for the asset editor.
   * @param ObjectToEdit Asset to open in the editor.
   */
  void InitEditor(
      const EToolkitMode::Type Mode,
      const TSharedPtr<class IToolkitHost>& InitToolkitHost,
      UMochiArticulationBase* ObjectToEdit);

 public:
  /**
   * Create and initialize an articulation asset editor.
   *
   * @param Mode Editor or prefab import mode.
   * @param InitToolkitHost Host for the asset editor.
   * @param ObjectToEdit Asset to open in the editor.
   * @return The newly initialized asset editor.
   */
  static TSharedRef<FMochiArticulationEditorToolkit> CreateEditor(
      const EToolkitMode::Type Mode,
      const TSharedPtr<IToolkitHost>& InitToolkitHost,
      UMochiArticulationBase* ObjectToEdit);

 public:
  /** Identifier used to register the articulation viewport tab. */
  static const FName ViewportTabID;
  /** Identifier used to register the articulation details tab. */
  static const FName DetailsTabID;
  /** High preview-render resolution percentage. */
  static constexpr int HighPreviewScreenPercentage = 200;
  /** Nominal preview-render resolution percentage. */
  static constexpr int NominalPreviewScreenPercentage = 100;

 private:
  TObjectPtr<UMochiArticulationBase> Articulation;
  TSharedPtr<IDetailsView> DetailsView;
  TSharedPtr<FTabManager::FLayout> StandaloneDefaultLayout;
  TSharedPtr<SMochiArticulationEditorViewport> EditorViewport;
  TSharedPtr<FMochiArticulationEditorPreviewScene> PreviewScene;
};

//------------------------------------------------------------------------------
// COMMANDS
//------------------------------------------------------------------------------

/** Commands exposed by the articulation editor. */
class FMochiArticulationEditorCommands : public TCommands<FMochiArticulationEditorCommands> {
 public:
  /** Initialize the articulation editor's command context. */
  FMochiArticulationEditorCommands()
      : TCommands<FMochiArticulationEditorCommands>(
            "MochiArticulationEditor",
            NSLOCTEXT("Contexts", "MochiArticulationEditor", "Articulation Editor"),
            "EditorViewport",
            FAppStyle::GetAppStyleSetName()) {}

  // Viewport manipulation
  TSharedPtr<FUICommandInfo> FocusViewport;

  /** Preview screen percentage/resolution */
  TSharedPtr<FUICommandInfo> ToggleScreenPercentage;

  /** Export */
  TSharedPtr<FUICommandInfo> Export;

  /** Regenerate the skeletal mesh of a UMochiArticulation */
  TSharedPtr<FUICommandInfo> GenerateSkeletalMesh;

  /** Write a SuperDexSkeleton header for the skeletal mesh (MochiSkeletonHeaderExport) */
  TSharedPtr<FUICommandInfo> GenerateSkeletonHeader;
  // Articulation "Show" menu
  TSharedPtr<FUICommandInfo> SetShowLinkTransforms;
  /** Command controlling joint annotations in the viewport. */
  TSharedPtr<FUICommandInfo> SetShowJoints;
  /** Command controlling visual-bounds annotations in the viewport. */
  TSharedPtr<FUICommandInfo> SetShowBounds;

  virtual void RegisterCommands() override;
};

//------------------------------------------------------------------------------
// VIEWPORT CLIENT
//------------------------------------------------------------------------------

/** Object retaining materials used by the articulation preview viewport. */
UCLASS()
class UMochiArticulationEditorViewportClientMaterials : public UObject {
  GENERATED_BODY()
 public:
  UPROPERTY()
  UMaterial* SolidGizmoMaterial;
  UPROPERTY()
  UMaterial* TransGizmoMaterial;
  UPROPERTY()
  UMaterialInstanceDynamic* ComMaterial;
  UPROPERTY()
  UMaterialInstanceDynamic* JointMaterial;
};

#ifndef MOCHI_ARTICULATION_VIEWPORT_SHOW
#define MOCHI_ARTICULATION_VIEWPORT_SHOW(NAME) \
 public:                                       \
  void SetShow##NAME(bool bShowOn);            \
  void ToggleShow##NAME();                     \
  bool IsShow##NAME##Checked() const;          \
                                               \
 private:                                      \
  bool bShow##NAME = false;
#endif

/** Viewport client rendering articulation previews and debug annotations. */
class FMochiArticulationEditorViewportClient : public FEditorViewportClient {
 public:
  /**
   * Create a viewport client for the articulation preview.
   *
   * @param InEditorViewport Viewport widget associated with this client.
   * @param InPreviewScene Scene containing the preview objects.
   * @param InArticulation Articulation asset supplied to the operation.
   * @param InPreviewMeshComponent Mesh component displayed in the preview.
   */
  FMochiArticulationEditorViewportClient(
      const TSharedRef<SMochiArticulationEditorViewport> InEditorViewport,
      const TSharedRef<FAdvancedPreviewScene> InPreviewScene,
      UMochiArticulationBase* InArticulation,
      UMochiArticulationPreviewComponent* InPreviewMeshComponent);

 public:
  // Begin FEditorViewportClient Overrides
  /**
   * Select the manipulation mode stored by this articulation viewport.
   *
   * @param NewMode Requested editor viewport widget mode.
   */
  virtual void SetWidgetMode(UE::Widget::EWidgetMode NewMode) override;
  virtual UE::Widget::EWidgetMode GetWidgetMode() const override;
  virtual bool CanSetWidgetMode(UE::Widget::EWidgetMode NewMode) const override;
  virtual void Draw(const FSceneView* View, FPrimitiveDrawInterface* PDI) override;
  virtual void DrawCanvas(FViewport& InViewport, FSceneView& View, FCanvas& Canvas) override;
  // End FEditorViewportClient Overrides

  /**
   * Draw articulation geometry and debug annotations in the viewport.
   *
   * @param Articulation Articulation asset supplied to the operation.
   * @param View Scene view used for editor rendering.
   * @param PDI Primitive drawing interface used for debug geometry.
   */
  void DrawArticulation(
      UMochiArticulation* Articulation,
      const FSceneView* View,
      FPrimitiveDrawInterface* PDI);
  /**
   * Draw articulation annotations on the viewport canvas.
   *
   * @param Articulation Articulation asset supplied to the operation.
   * @param InViewport Viewport used by the editor operation.
   * @param View Scene view used for editor rendering.
   * @param Canvas Canvas used for editor annotations.
   */
  void DrawCanvasArticulation(
      UMochiArticulation* Articulation,
      FViewport& InViewport,
      FSceneView& View,
      FCanvas& Canvas);

 public:
  // Begin FGCObject Interface
  virtual void AddReferencedObjects(FReferenceCollector& Collector) override;
  virtual FString GetReferencerName() const override;
  // End FGCObject Interface

 public:
  /**
   * Set the articulation and mesh component shown in the preview.
   *
   * @param InArticulation Articulation asset supplied to the operation.
   * @param InPreviewMeshComponent Mesh component displayed in the preview.
   */
  void SetPreviewMesh(
      UMochiArticulationBase* InArticulation,
      UMochiArticulationPreviewComponent* InPreviewMeshComponent);

 public:
  MOCHI_ARTICULATION_VIEWPORT_SHOW(LinkTransforms)
  MOCHI_ARTICULATION_VIEWPORT_SHOW(Joints);
  MOCHI_ARTICULATION_VIEWPORT_SHOW(Bounds)

 public:
  /** Viewport widget associated with this client. */
  TWeakPtr<SMochiArticulationEditorViewport> ViewportPtr;
  /** Manipulation mode stored by this viewport client. */
  UE::Widget::EWidgetMode WidgetMode;

 private:
  FAdvancedPreviewScene* AdvancedPreviewScene;
  TObjectPtr<UMochiArticulationBase> PreviewArticulation;
  TObjectPtr<UMochiArticulationPreviewComponent> PreviewMeshComponent;
  SMochiArticulationEditorViewport* ArticulationViewport;
  TStrongObjectPtr<UMochiArticulationEditorViewportClientMaterials> Materials;
};

//------------------------------------------------------------------------------
// VIEWPORT TOOLBAR SLATE
//------------------------------------------------------------------------------

/** Toolbar widget for the articulation preview viewport. */
class SMochiArticulationEditorViewportToolbar : public SCommonEditorViewportToolbarBase {
 public:
  SLATE_BEGIN_ARGS(SMochiArticulationEditorViewportToolbar) {}
  SLATE_END_ARGS()
  /**
   * Construct the viewport toolbar using its information provider.
   *
   * @param InArgs Slate construction arguments.
   * @param InInfoProvider Provider of the viewport and its command bindings.
   */
  void Construct(
      const FArguments& InArgs,
      TSharedPtr<class ICommonEditorViewportToolbarInfoProvider> InInfoProvider);

 private:
  // virtual TSharedRef<SWidget> GenerateShowMenu() const override;
  virtual void ExtendLeftAlignedToolbarSlots(
      TSharedPtr<SHorizontalBox> MainBoxPtr,
      TSharedPtr<SViewportToolBar> ParentToolBarPtr) const override;
  virtual TSharedPtr<FExtender> GetViewMenuExtender() const override;

  /** Generate Articulation Menu */
  TSharedRef<SWidget> GenerateArticulationMenu() const;
};

//------------------------------------------------------------------------------
// VIEWPORT SLATE
//------------------------------------------------------------------------------

/** Widget hosting the articulation preview viewport. */
class SMochiArticulationEditorViewport : public SAssetEditorViewport,
                                         public FGCObject,
                                         public ICommonEditorViewportToolbarInfoProvider {
 public:
  SLATE_BEGIN_ARGS(SMochiArticulationEditorViewport) {}
  SLATE_END_ARGS()
  /**
   * Construct the articulation preview widget and connect its toolkit and preview scene.
   *
   * @param InArgs Slate construction arguments.
   * @param InEditorToolkit Toolkit owning the articulation editor.
   * @param InPreviewScene Scene containing the articulation preview objects.
   */
  void Construct(
      const FArguments& InArgs,
      TSharedPtr<FMochiArticulationEditorToolkit> InEditorToolkit,
      TSharedPtr<FMochiArticulationEditorPreviewScene> InPreviewScene);
  virtual ~SMochiArticulationEditorViewport() override;

 public:
  // Begin SAssetEditorViewport Interface
  virtual void BindCommands() override;
  virtual TSharedRef<SEditorViewport> GetViewportWidget() override;
  virtual TSharedPtr<FExtender> GetExtenders() const override;
  virtual void OnFloatingButtonClicked() override;
  virtual void OnFocusViewportToSelection() override;
  virtual TSharedRef<FEditorViewportClient> MakeEditorViewportClient() override;
#if UE_VERSION_OLDER_THAN(5, 7, 0)
  virtual TSharedPtr<SWidget> MakeViewportToolbar() override;
#endif
  virtual void PopulateViewportOverlays(TSharedRef<SOverlay> Overlay) override;
  /**
   * Makes the viewport toolbar. SEditorViewport adds it through MakeViewportToolbar() before
   * UE 5.7; from 5.7 that is deprecated and final, so PopulateViewportOverlays() adds it.
   *
   * @return The viewport toolbar widget.
   */
  TSharedRef<SWidget> MakeLegacyViewportToolbar();
  // End SAssetEditorViewport Interface

 public:
  // Begin FGCObject Interface
  virtual void AddReferencedObjects(FReferenceCollector& Collector) override;
  virtual FString GetReferencerName() const override;
  ;
  // End FGCObject Interface

 public:
  /**
   * Set the text displayed in the viewport overlay.
   *
   * @param TextItems Text entries displayed in the viewport overlay.
   */
  void PopulateOverlayText(const TArray<FText>& TextItems) const;
  /**
   * Get the client controlling this preview viewport.
   *
   * @return The viewport client.
   */
  TSharedPtr<class FMochiArticulationEditorViewportClient> GetViewportClient();
  /**
   * Get the mesh component displayed in the preview.
   *
   * @return The preview mesh component.
   */
  UMochiArticulationPreviewComponent* GetPreviewMeshComponent();

 public:
  /** Toolkit associated with the articulation viewport widget. */
  TSharedPtr<FMochiArticulationEditorToolkit> EditorToolkit;

 private:
  TSharedPtr<FMochiArticulationEditorPreviewScene> PreviewScene;
  TSharedPtr<SRichTextBlock> OverlayText;
  TSharedPtr<FMochiArticulationEditorViewportClient> ViewportClient;
  TObjectPtr<UMochiArticulationPreviewComponent> PreviewMeshComponent;
  float PerspectiveFOV = 90.0f;
};

//------------------------------------------------------------------------------
// PREVIEW SCENE
//------------------------------------------------------------------------------

/** Preview scene containing the articulation editor's display objects. */
class FMochiArticulationEditorPreviewScene : public FAdvancedPreviewScene {
 public:
  /**
   * Create the articulation editor's preview scene.
   *
   * @param CVS Preview-scene construction settings.
   * @param EditorToolkit Toolkit owning the preview scene.
   */
  FMochiArticulationEditorPreviewScene(
      ConstructionValues CVS,
      const TSharedRef<FMochiArticulationEditorToolkit>& EditorToolkit);
  virtual ~FMochiArticulationEditorPreviewScene() override;
  virtual void Tick(float DeltaTime) override;
  /**
   * Get the toolkit owning this preview scene.
   *
   * @return The editor.
   */
  TSharedRef<FMochiArticulationEditorToolkit> GetEditor() const {
    return EditorToolkitPtr.Pin().ToSharedRef();
  }

 private:
  TWeakPtr<FMochiArticulationEditorToolkit> EditorToolkitPtr;
};
