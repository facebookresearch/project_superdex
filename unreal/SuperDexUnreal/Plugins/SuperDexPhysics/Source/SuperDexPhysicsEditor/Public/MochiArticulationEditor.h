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

class SUPERDEXPHYSICSEDITOR_API FMochiArticulationEditorToolkit : public FAssetEditorToolkit,
                                                                  public FNotifyHook {
 private:
  using Super = FAssetEditorToolkit;

 public:
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
  TObjectPtr<UMochiArticulationBase> GetArticulation();
  TSharedRef<SDockTab> SpawnTab_Details(const FSpawnTabArgs& Args);
  TSharedRef<SDockTab> SpawnTab_Viewport(const FSpawnTabArgs& Args);
  void BindCommands();
  TSharedPtr<FMochiArticulationEditorPreviewScene> CreatePreviewScene();
  void ExtendToolbars();
  void FocusViewport();
  void ToggleScreenPercentage() const;
  void Export() const;
  bool CanExport() const;
  void GenerateSkeletalMesh() const;
  bool CanGenerateSkeletalMesh() const;
  void GenerateSkeletonHeader() const;
  bool CanGenerateSkeletonHeader() const;
  void OnObjectModified(UObject* InObject);
  void InitEditor(
      const EToolkitMode::Type Mode,
      const TSharedPtr<class IToolkitHost>& InitToolkitHost,
      UMochiArticulationBase* ObjectToEdit);

 public:
  static TSharedRef<FMochiArticulationEditorToolkit> CreateEditor(
      const EToolkitMode::Type Mode,
      const TSharedPtr<IToolkitHost>& InitToolkitHost,
      UMochiArticulationBase* ObjectToEdit);

 public:
  static const FName ViewportTabID;
  static const FName DetailsTabID;
  static constexpr int kHighPreviewScreenPercentage = 200;
  static constexpr int kNominalPreviewScreenPercentage = 100;

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

class FMochiArticulationEditorCommands : public TCommands<FMochiArticulationEditorCommands> {
 public:
  FMochiArticulationEditorCommands()
      : TCommands<FMochiArticulationEditorCommands>(
            "MochiArticulationEditor",
            NSLOCTEXT("Contexts", "MochiArticulationEditor", "Articulation Editor"),
            "EditorViewport",
            FAppStyle::GetAppStyleSetName()) {}

  // Viewport manipulation
  TSharedPtr<FUICommandInfo> FocusViewport;
  // Preview screen percentage/resolution
  TSharedPtr<FUICommandInfo> ToggleScreenPercentage;
  // Export
  TSharedPtr<FUICommandInfo> Export;
  // Regenerate the skeletal mesh of a UMochiArticulation
  TSharedPtr<FUICommandInfo> GenerateSkeletalMesh;
  // Write a SuperDexSkeleton header for the skeletal mesh (MochiSkeletonHeaderExport)
  TSharedPtr<FUICommandInfo> GenerateSkeletonHeader;
  // Articulation "Show" menu
  TSharedPtr<FUICommandInfo> SetShowLinkTransforms;
  TSharedPtr<FUICommandInfo> SetShowJoints;
  TSharedPtr<FUICommandInfo> SetShowBounds;

  virtual void RegisterCommands() override;
};

//------------------------------------------------------------------------------
// VIEWPORT CLIENT
//------------------------------------------------------------------------------

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

class FMochiArticulationEditorViewportClient : public FEditorViewportClient {
 public:
  FMochiArticulationEditorViewportClient(
      const TSharedRef<SMochiArticulationEditorViewport> InEditorViewport,
      const TSharedRef<FAdvancedPreviewScene> InPreviewScene,
      UMochiArticulationBase* InArticulation,
      UMochiArticulationPreviewComponent* InPreviewMeshComponent);

 public:
  // Begin FEditorViewportClient Overrides
  virtual void SetWidgetMode(UE::Widget::EWidgetMode NewMode) override;
  virtual UE::Widget::EWidgetMode GetWidgetMode() const override;
  virtual bool CanSetWidgetMode(UE::Widget::EWidgetMode NewMode) const override;
  virtual void Draw(const FSceneView* View, FPrimitiveDrawInterface* PDI) override;
  virtual void DrawCanvas(FViewport& InViewport, FSceneView& View, FCanvas& Canvas) override;
  // End FEditorViewportClient Overrides

  void DrawArticulation(
      UMochiArticulation* Articulation,
      const FSceneView* View,
      FPrimitiveDrawInterface* PDI);
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
  void SetPreviewMesh(
      UMochiArticulationBase* InArticulation,
      UMochiArticulationPreviewComponent* InPreviewMeshComponent);

 public:
  MOCHI_ARTICULATION_VIEWPORT_SHOW(LinkTransforms)
  MOCHI_ARTICULATION_VIEWPORT_SHOW(Joints);
  MOCHI_ARTICULATION_VIEWPORT_SHOW(Bounds)

 public:
  TWeakPtr<SMochiArticulationEditorViewport> ViewportPtr;
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

class SMochiArticulationEditorViewportToolbar : public SCommonEditorViewportToolbarBase {
 public:
  SLATE_BEGIN_ARGS(SMochiArticulationEditorViewportToolbar) {}
  SLATE_END_ARGS()
  void Construct(
      const FArguments& InArgs,
      TSharedPtr<class ICommonEditorViewportToolbarInfoProvider> InInfoProvider);

 private:
  // virtual TSharedRef<SWidget> GenerateShowMenu() const override;
  virtual void ExtendLeftAlignedToolbarSlots(
      TSharedPtr<SHorizontalBox> MainBoxPtr,
      TSharedPtr<SViewportToolBar> ParentToolBarPtr) const override;
  virtual TSharedPtr<FExtender> GetViewMenuExtender() const override;
  // Generate Articulation Menu
  TSharedRef<SWidget> GenerateArticulationMenu() const;
};

//------------------------------------------------------------------------------
// VIEWPORT SLATE
//------------------------------------------------------------------------------

class SMochiArticulationEditorViewport : public SAssetEditorViewport,
                                         public FGCObject,
                                         public ICommonEditorViewportToolbarInfoProvider {
 public:
  SLATE_BEGIN_ARGS(SMochiArticulationEditorViewport) {}
  SLATE_END_ARGS()
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
  // Makes the viewport toolbar. SEditorViewport adds it through MakeViewportToolbar() before
  // UE 5.7; from 5.7 that is deprecated and final, so PopulateViewportOverlays() adds it.
  TSharedRef<SWidget> MakeLegacyViewportToolbar();
  // End SAssetEditorViewport Interface

 public:
  // Begin FGCObject Interface
  virtual void AddReferencedObjects(FReferenceCollector& Collector) override;
  virtual FString GetReferencerName() const override;
  ;
  // End FGCObject Interface

 public:
  void PopulateOverlayText(const TArray<FText>& TextItems) const;
  TSharedPtr<class FMochiArticulationEditorViewportClient> GetViewportClient();
  UMochiArticulationPreviewComponent* GetPreviewMeshComponent();

 public:
  TSharedPtr<FMochiArticulationEditorToolkit> EditorToolkit;

 private:
  TSharedPtr<FMochiArticulationEditorPreviewScene> PreviewScene;
  TSharedPtr<SRichTextBlock> OverlayText;
  TSharedPtr<FMochiArticulationEditorViewportClient> ViewportClient;
  TObjectPtr<UMochiArticulationPreviewComponent> PreviewMeshComponent;
  float kPerspectiveFOV = 90.0f;
};

//------------------------------------------------------------------------------
// PREVIEW SCENE
//------------------------------------------------------------------------------

class FMochiArticulationEditorPreviewScene : public FAdvancedPreviewScene {
 public:
  FMochiArticulationEditorPreviewScene(
      ConstructionValues CVS,
      const TSharedRef<FMochiArticulationEditorToolkit>& EditorToolkit);
  virtual ~FMochiArticulationEditorPreviewScene() override;
  virtual void Tick(float DeltaTime) override;
  TSharedRef<FMochiArticulationEditorToolkit> GetEditor() const {
    return EditorToolkitPtr.Pin().ToSharedRef();
  }

 private:
  TWeakPtr<FMochiArticulationEditorToolkit> EditorToolkitPtr;
};
