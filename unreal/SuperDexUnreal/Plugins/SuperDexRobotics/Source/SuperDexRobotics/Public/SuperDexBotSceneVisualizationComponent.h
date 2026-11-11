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

#include <mochi_core/utils/transform_rt.h>
#include <mochi_physics/cpp_api/mochi_handle.h>
#include "Components/SceneComponent.h"
#include "CoreMinimal.h"

#include "SuperDexBotSceneVisualizationComponent.generated.h"

class UMeshComponent;
class UMochiSceneObject;
class UPoseableMeshComponent;
class USkeletalMesh;
class UStaticMesh;

namespace UE::Geometry {
class FDynamicMesh3;
}

namespace superdex::robotics {
class BotScene;
}

struct FBotSceneVizSnapshotEntry;
struct FBotSceneVizSkinSnapshot;

/**
 * Runtime mirror of the loaded Mochi BotScene into Unreal.
 *
 * Placed on (or spawned by) a holder actor in the scene level. When the
 * USuperDexBotSceneSubsystem is in the Loaded state, this component enumerates
 * the BotScene's base-scene actors, spawned-prefab actors, and bot articulated
 * link actors on the Mochi thread via QueueOrExecuteCommand+WaitForQueuedCommands,
 * snapshots their world transforms, caches per-ReferenceShape geometry via
 * MochiDynamicMeshUtil into FDynamicMesh3, and creates movable UDynamicMeshComponents
 * on the game thread. Each tick, link transforms are re-snapshotted and each
 * mesh component's world transform is updated (no per-frame mesh rebuild;
 * winding + meters->cm is baked at cache time).
 *
 * An entry with a render model (.glb) is drawn with that model instead, imported at runtime by
 * USuperDexRuntimeMeshSubsystem. Its visual appears once the import finishes; if the import fails
 * the entry falls back to its Mochi shape. -NoRenderModels on the command line draws every entry
 * from its Mochi shape.
 *
 * A bot's deformable skin with a render model is drawn as a transient skeletal mesh (built by
 * USuperDexRuntimeMeshSubsystem::RequestSkinnedMesh) on a UPoseableMeshComponent, with one bone per
 * skin bone posed from its link. Once it is drawn, the links it covers are hidden; until then, or
 * if it cannot be built, they keep their own visuals.
 *
 * Teardown: owned mesh components are destroyed when BotScene is cleared
 * or EndPlay fires; geometry cache is dropped on failure/reload.
 *
 * This component never loads a scene. It builds from whatever
 * USuperDexBotSceneSubsystem has loaded, either immediately at BeginPlay or when
 * the subsystem's OnSceneLoaded / OnSceneContentsChanged delegates fire, so it
 * makes no assumption about actor initialization order.
 */
UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class SUPERDEXROBOTICS_API USuperDexBotSceneVisualizationComponent : public USceneComponent {
  GENERATED_BODY()

 public:
  USuperDexBotSceneVisualizationComponent();

 protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
  virtual void TickComponent(
      float DeltaTime,
      ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;

 private:
  void RebuildIfNeeded();
  void DestroyVisuals();
  void CancelPostStepCallback();
  bool TrySnapshotLinkTransforms(
      UMochiSceneObject* SceneObject,
      const superdex::robotics::BotScene* BotScene,
      TArray<FBotSceneVizSnapshotEntry>& OutEntries,
      TArray<FBotSceneVizSkinSnapshot>& OutSkins,
      TArray<mochi::ActorHandle>& OutArticulatedActors) const;
  void EnsureVisualsForSnapshot(
      const TArray<FBotSceneVizSnapshotEntry>& Snapshot,
      UMochiSceneObject* SceneObject);
  void EnsureSkinsForSnapshot(const TArray<FBotSceneVizSkinSnapshot>& Snapshot);
  void OnSkinnedMeshLoaded(uint64 ActorKey, uint32 RequestId, USkeletalMesh* Mesh);
  // Hides the visuals of links a drawn skin covers, and shows every other link visual.
  void UpdateCoveredLinkVisibility();
  // Poses each drawn skin from LinkToWorldByHandle.
  void PoseSkins();
  UMeshComponent* CreateShapeMeshComponent(
      mochi::ShapeHandle RefShape,
      uint64 HandleKey,
      UMochiSceneObject* SceneObject);
  UMeshComponent* NewVisualComponent(TSubclassOf<UMeshComponent> ComponentClass);
  void RequestRenderModel(uint64 HandleKey, const FString& File, UMochiSceneObject* SceneObject);
  void OnRenderModelLoaded(
      uint64 HandleKey,
      const FString& File,
      UStaticMesh* Mesh,
      UMochiSceneObject* SceneObject);

  // Per-ShapeHandle cache: geometry built once via MochiDynamicMeshUtil,
  // winding + meters->cm baked, normals computed, inverse-scale baked.
  // Key is ShapeHandle.value (handle hash); value is a shared FDynamicMesh3.
  TMap<uint64, TSharedPtr<UE::Geometry::FDynamicMesh3>> GeometryCache;

  // The component scale baked into everything currently in GeometryCache. The cache key does
  // not carry it, so the cache is dropped when this stops matching the live scale.
  FVector BakedGeometryScale = FVector::OneVector;

  struct FVisualEntry {
    uint64 HandleValue = 0;
    uint64 ShapeKey = 0;
    mochi::ShapeHandle RefShape;
    // Absolute .glb path, or empty to draw the Mochi shape.
    FString RenderModelFile;
    // Null while the render model is importing.
    TObjectPtr<UMeshComponent> MeshComp = nullptr;
    FTransform ComponentToLink = FTransform::Identity;
    FTransform LinkToWorld = FTransform::Identity;
  };
  TArray<FVisualEntry> Visuals;

  struct FSkinVisual {
    // The bot's articulated actor handle.
    uint64 ActorKey = 0;
    FString File;
    // Per skin bone: the link it follows (kNoLink for none). Skeleton bone i + 1 is skin bone i.
    TArray<uint64> BoneLinks;
    // The skin bone whose link the component follows; bones are posed relative to it.
    int32 AnchorBone = 0;
    TArray<uint64> CoveredLinks;
    // What the mesh is built from, besides File and BoneLinks (see FSuperDexSkinnedMeshRequest).
    FTransform MeshToComponent;
    TArray<FTransform> RestBoneToComponent;
    // Identifies the mesh request, so a superseded request's mesh is dropped.
    uint32 RequestId = 0;
    // Null while the mesh is being built, or if it could not be.
    TObjectPtr<UPoseableMeshComponent> MeshComp = nullptr;
  };
  TArray<FSkinVisual> Skins;
  uint32 LastSkinRequestId = 0;

  // Latest world transform of every link a visual or skin follows.
  TMap<uint64, FTransform> LinkToWorldByHandle;

  // Tick runs only when visuals exist.
  bool bHasSceneSnapshot = false;

  // False under -NoRenderModels.
  bool bUseRenderModels = true;

  // Viz publisher: Mochi PostStep writes LatestTransforms, Game tick reads it.
  // Matches legacy UMochiArticulatedActorObject triple-buffer polling - no
  // Game→Mochi QueueOrExecuteCommand+WaitForQueuedCommands on the tick path.
  mutable FCriticalSection LatestTransformsLock;
  TMap<uint64, mochi::TransformRT> LatestTransforms;
  mochi::CallbackHandle VizPostStepHandle;
  TWeakObjectPtr<UMochiSceneObject> VizSceneObject;

  FDelegateHandle SceneLoadedHandle;
  FDelegateHandle SceneContentsChangedHandle;
};
