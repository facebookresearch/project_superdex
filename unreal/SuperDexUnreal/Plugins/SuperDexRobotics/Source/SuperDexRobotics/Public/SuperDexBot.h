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

#include "CoreMinimal.h"
#include "EditorFramework/AssetImportData.h"
#include "MochiArticulationBase.h"

#include <superdex_robotics/superdex_robotics.h>

#include <string_view>

#include "SuperDexBot.generated.h"

class UMochiModel;
class FSceneView;
class FViewport;
class FCanvas;
class FPrimitiveDrawInterface;
class FMaterialRenderProxy;

/// Marker written over a sensor's `params` at import once the params file it named has been
/// captured into USuperDexBot::SensorParamFiles. The remainder of the string is the key into
/// that map. Chosen so superdex_robotics still classifies the value as a path rather than
/// inline JSON (@ref superdex::robotics::IsInlineJson), and so it collides with neither of
/// that library's own path forms (`//` root-relative and `@tag/`).
inline constexpr std::string_view kSuperDexBotDataPrefix = "botdata:";

enum class ESuperDexBotComponentParamsKind : uint8 {
  Empty,
  InlineJson,
  CapturedFile,
  LoaderPath,
  FilePath,
};

/// Mirrors superdex::robotics::IsInlineJson: a params value is inline JSON when its first
/// non-whitespace character is '{' or '[', and any other non-empty value is a file path.
/// Reimplemented rather than called because the declaration in the vendored file_utils.h sits
/// behind MOCHI_USE_REFLECTION, which none of the Unreal plugin modules define.
inline bool IsInlineJsonParams(std::string_view Value) {
  for (const char C : Value) {
    if (C == ' ' || C == '\t' || C == '\n' || C == '\r') {
      continue;
    }
    return C == '{' || C == '[';
  }
  return false; // empty or all-whitespace -> treat as (empty) path, not inline JSON
}

inline ESuperDexBotComponentParamsKind ClassifyBotComponentParams(std::string_view Value) {
  if (Value.empty()) {
    return ESuperDexBotComponentParamsKind::Empty;
  }
  if (IsInlineJsonParams(Value)) {
    return ESuperDexBotComponentParamsKind::InlineJson;
  }
  if (Value.starts_with(kSuperDexBotDataPrefix)) {
    return ESuperDexBotComponentParamsKind::CapturedFile;
  }
  if (Value.starts_with("//") || Value.starts_with("@")) {
    return ESuperDexBotComponentParamsKind::LoaderPath;
  }
  return ESuperDexBotComponentParamsKind::FilePath;
}

/// One file captured out of a bot's sensor params at import time.
USTRUCT()
struct FSuperDexBotDataFile {
  GENERATED_BODY()

  UPROPERTY()
  TArray<uint8> Contents;
};

UCLASS()
class SUPERDEXROBOTICS_API USuperDexBotImportData : public UAssetImportData {
  GENERATED_BODY()
 public:
  UPROPERTY(VisibleAnywhere)
  FString SourceFileHash;
  UPROPERTY(VisibleAnywhere)
  TArray<FString> RenderModelHashes;
  UPROPERTY(VisibleAnywhere)
  TArray<FString> MochiModelHashes;
  /// Content hash of the skin collision model (skin.shape .mochi.h5), used to skip
  /// reimporting an unchanged skin collision mesh. Empty if the bot has no skin.
  UPROPERTY(VisibleAnywhere)
  FString SkinMochiModelHash;
  /// Content hash of the skin render model (skin.renderModel .glb), used to skip reimporting an
  /// unchanged skin render mesh. Empty if the bot's skin has no render model.
  UPROPERTY(VisibleAnywhere)
  FString SkinRenderModelHash;
  /// Object paths (relative to this bot's package folder) of every asset generated
  /// by the last import -- render/collision meshes, the skeleton and skeletal mesh,
  /// and the materials they reference. Reimport checks these still exist to detect
  /// deleted outputs even when the source .superdex_bot is unchanged. Stored relative
  /// so the bot's folder can be moved and still reimport correctly.
  UPROPERTY(VisibleAnywhere)
  TArray<FString> GeneratedAssetPaths;
};

UCLASS(
    BlueprintType,
    HideCategories = Object,
    DisplayName = "SuperDex Bot",
    PrioritizeCategories = ("SuperDex Bot", "Assets", "Internal"),
    AutoCollapseCategories = ("Internal"))
/// Imported Unreal asset representing a SuperDex bot.
///
/// Created by USuperDexBotFactory when a .superdex_bot file is dragged into the Content
/// Browser. Stores the source JSON, a generated skeletal mesh, and any associated
/// MochiModels. Supports reimport via FReimportManager (right-click "Reimport" in
/// the Content Browser, or the Reimport button in the Details panel).
class SUPERDEXROBOTICS_API USuperDexBot : public UMochiArticulationBase {
  GENERATED_BODY()
 public:
#if WITH_EDITOR
  /// Creates the UAssetImportData subobject for source file tracking.
  virtual void PostInitProperties() override;
  // Upgrades AssetImportData to USuperDexBotImportData if necessary.
  virtual void PostLoad() override;
  /// Exposes the source file path to the Asset Registry for reimport discovery.
  virtual void GetAssetRegistryTags(FAssetRegistryTagsContext Context) const override;
  /// Triggers reimport of this asset from its source .superdex_bot file via FReimportManager.
  void Reimport();
  /// Returns the absolute path to the source .superdex_bot file on disk.
  FString GetSourceFilePath() const;
  /// Computes a SHA1 hash of the source .superdex_bot file and all referenced asset
  /// files (.glb, .mochi.h5). If the bot was a ModBotPrefab, the hash is
  /// computed on the resolved (built)
  FString ComputeSourceHash() const;
  /// Returns true if the source .superdex_bot on disk has changed since the last import
  /// (i.e. ComputeSourceHash() differs from the stored hash).
  bool IsOutOfSync() const;
#endif
  /// Returns the bot name extracted from the BotPrefab JSON (e.g. "left_hand").
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  const FString& GetBotName() const;
  /// Returns the generated skeletal mesh, or nullptr if generation failed.
  virtual USkeletalMesh* GetSkeletalMesh() const override;
  /// Returns the imported MochiModel assets (one per link that has a mochiModel entry).
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  const TArray<UMochiModel*>& GetMochiModels() const;
  /// Returns the fully-resolved BotPrefab JSON.
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  const FString& GetBotPrefabJson() const;
  /// Returns the BotPrefab
  const superdex::robotics::BotPrefab& GetBotPrefab() const;

  virtual const FString& GetArticulationName() const override;
  virtual int GetNumDofs() const override;
  virtual const TArray<double>& GetDefaultJointPositions() const override;
  virtual bool ComputeLinkTransformsFromJointPositions(
      const TArray<double>& JointPositions,
      TArray<FTransform>& OutLinkTransforms,
      EMochiArticulationLinkTransformSpace TransformSpace,
      double AngleScale = 1.0,
      double LengthScale = 100.0,
      int FinalLinkIndex = -1) const override;
  virtual bool ComputeJointPositionsFromLinkTransforms(
      const TArray<FTransform>& LinkTransforms,
      TArray<double>& OutJointPositions,
      EMochiArticulationLinkTransformSpace TransformSpace,
      double AngleScale = 1.0,
      double LengthScale = 100.0) const override;
  virtual bool ComputeVisualBoundsFromLinkTransforms(
      const TArray<FTransform>& LinksToTransformSpace,
      EMochiArticulationLinkTransformSpace TransformSpace,
      FBoxSphereBounds& OutBounds) const override;
  virtual TArray<FName> GetLinkNames() const override;
  virtual TArray<FName> GetJointNames() const override;
  /// Returns the index of a joint's first DOF among the bot's DOFs (the order of joint positions,
  /// e.g. USuperDexBotComponent::GetCurrentJointPositions), or INDEX_NONE if the joint doesn't
  /// exist or has no DOFs.
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  int32 GetDofIndex(FName JointName) const;
  /// Returns the name of the link a DOF moves, or NAME_None for an invalid DOF index.
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  FName GetDofLinkName(int32 DofIndex) const;
  virtual bool SupportsUrdfExport() const override {
    return true;
  }
  virtual FString ToUrdfXml(
      const FString& VisualRelPath = TEXT(""),
      const FString& CollisionRelPath = TEXT("")) const override;

  // --- UMochiArticulationBase editor-viewport hooks (Part D of the plugin split: these
  // let MochiArticulationEditor.cpp stay fully bot-agnostic at compile time). See the
  // base class declarations in MochiArticulationBase.h for the full rationale. ---
  virtual void ComputeOverlayStats(double& OutMass, int& OutNumSurfaceFaces) const override;
  virtual void DrawCustom(
      const FSceneView* View,
      FPrimitiveDrawInterface* PDI,
      bool bShowLinkTransforms,
      bool bShowJoints,
      bool bShowBounds,
      FMaterialRenderProxy* JointMaterialRenderProxy) const override;
  virtual void DrawCanvasCustom(
      FViewport& InViewport,
      FSceneView& View,
      FCanvas& Canvas,
      float DPIScale,
      bool bShowLinkTransforms,
      bool bShowJoints) const override;

 protected:
  friend class USuperDexBotFactory;
  friend class UReimportSuperDexBotFactory;
  friend class FSuperDexBotEditorToolkit;
  // Calls MaterializeSensorParamFiles: the loader is the single point every BotPrefab this
  // asset produces passes through, including one resolved by UE path for a nested bot.
  friend class FSuperDexBotLoader;

  /// Bot name extracted from the BotPrefab JSON (sanitized for use as a package path).
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, AssetRegistrySearchable, Category = "SuperDex Bot")
  FString BotName;
#if WITH_EDITORONLY_DATA
  /// UE import data subobject storing the source file path for reimport support.
  UPROPERTY(VisibleAnywhere, Category = "Internal")
  TObjectPtr<USuperDexBotImportData> AssetImportData;
#endif

  /// Fully-resolved BotPrefab JSON.
  UPROPERTY()
  FString BotPrefabJson;
  /// Generated skeletal mesh with per-link bones and render geometry baked in.
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, AssetRegistrySearchable, Category = "Assets")
  TObjectPtr<USkeletalMesh> SkeletalMesh;
  /// Imported MochiModel (.mochi.h5) assets: one per link that defines a mochiModel entry,
  /// plus the optional skin collision model (skin.shape) when the bot has a skin. Held here
  /// to keep them GC-alive and saved with the bot; not indexed by link at runtime.
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Assets")
  TArray<TObjectPtr<UMochiModel>> MochiModels;
  /// Per-link local bounding boxes for visual bounds computation.
  UPROPERTY()
  TArray<FBoxSphereBounds> LinkBounds;

  /// Every sensor params file this bot's links name, plus the companion files those params
  /// reference (a learned sensor's model weights, say), captured at import and keyed by a
  /// bot-relative path.
  ///
  /// superdex_robotics opens a sensor's params through the filesystem, and the path the
  /// importer hands it is absolute and rooted in the *importing* machine's archive-extraction
  /// temp dir (see ExtractBotArchiveToCache). Baking that path into the asset makes the bot
  /// loadable only where that temp dir happens to survive, so the bytes travel inside the
  /// asset instead and are written back out by MaterializeSensorParamFiles before the bot is
  /// built. Empty for bots whose sensors all use inline JSON params, and for bots imported
  /// before this capture existed.
  UPROPERTY()
  TMap<FString, FSuperDexBotDataFile> SensorParamFiles;

 private:
  /// Lazily loads and caches BotPrefab from the stored JSON.
  bool EnsureBotPrefabCached(bool bForceReload = false) const;
  /// Writes SensorParamFiles to a scratch directory and repoints every `botdata:`-marked
  /// sensor params in @p Prefab at the file written for it. Returns false if any captured
  /// file could not be written or any marker had no matching entry; the bot is not loadable
  /// in that case, so the caller should treat it as a load failure rather than build a bot
  /// whose sensors would fail one by one.
  bool MaterializeSensorParamFiles(superdex::robotics::BotPrefab& Prefab) const;
  /// Returns the per-link bounding boxes.
  const TArray<FBoxSphereBounds>& GetLinkBounds() const;

 private:
  mutable superdex::robotics::BotPrefab CachedBotPrefab;
  mutable bool bHasCachedBotPrefab = false;
  mutable FCriticalSection SensorParamFilesMaterializationMutex;
  mutable FString CachedMaterializedSensorParamFilesDir;
  mutable double NextSensorParamFilesMaterializationAttemptSeconds = 0.0;
  mutable TArray<double> CachedDefaultJointPositions;
};
