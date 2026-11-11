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

#include "SuperDexBotSceneVisualizationComponent.h"

#include "SuperDexBotSceneSubsystem.h"

#include "Algo/Compare.h"
#include "Components/DynamicMeshComponent.h"
#include "Components/PoseableMeshComponent.h"
#include "Components/StaticMeshComponent.h"
#include "DynamicMesh/DynamicMesh3.h"
#include "DynamicMesh/MeshNormals.h"
#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "MaterialDomain.h"
#include "Materials/Material.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "MochiConversions.h"
#include "MochiDynamicMeshUtil.h"
#include "MochiGameInstanceSubsystem.h"
#include "MochiSceneObject.h"
#include "SuperDexRuntimeRenderUtils.h"
#include "UDynamicMesh.h"

#include <atomic>

#include <mochi_core/utils/error.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_physics/cpp_api/mochi_handle.h>
#include <superdex_robotics/scene/bot_scene.h>
#include <superdex_robotics/superdex_robotics.h>

// A render model drawn in place of a Mochi shape, posed in its actor's (or link's) frame.
struct FBotSceneVizRenderModel {
  // Absolute .glb path; empty draws the Mochi shape.
  FString File;
  mochi::TransformRT LinkFromModel;
  mochi::Real3 Scale = {1_r, 1_r, 1_r};
};

struct FBotSceneVizSnapshotEntry {
  mochi::ActorHandle LinkHandle;
  mochi::ShapeHandle RefShape; // Invalid when only a render model is drawn.
  uint64 ShapeKey = 0; // RefShape.value
  mochi::TransformRT WorldFromLink;
  FBotSceneVizRenderModel RenderModel;
};

// A bot's skin render model (see superdex::robotics::ActorSkinRenderModel), with the world
// transforms of its bones' links.
struct FBotSceneVizSkinSnapshot {
  uint64 ActorKey = 0; // The bot's articulated actor handle.
  FString File;
  mochi::TransformRT RootFromModel;
  mochi::Real3 Scale = {1_r, 1_r, 1_r};
  TArray<uint64> BoneLinks; // kNoLink for a bone without a link.
  TArray<mochi::TransformRT> RootFromBoneRest;
  TArray<mochi::TransformRT> WorldFromBoneLink;
  TArray<uint64> CoveredLinks;
};

namespace {
// A skin bone that follows no link.
constexpr uint64 kNoLink = MAX_uint64;

// Stable per-mesh palette - 16 distinct hues spanning the wheel so you do not
// get all-reds when shape handles are sequential.
FLinearColor GetDistinctVizColor(uint64 ShapeKey, uint64 HandleValue) {
  static const FLinearColor Palette[16] = {
      FLinearColor(0.90f, 0.20f, 0.20f), // red
      FLinearColor(0.95f, 0.55f, 0.10f), // orange
      FLinearColor(0.92f, 0.80f, 0.15f), // yellow
      FLinearColor(0.35f, 0.85f, 0.30f), // green
      FLinearColor(0.20f, 0.80f, 0.65f), // teal
      FLinearColor(0.22f, 0.60f, 0.92f), // sky
      FLinearColor(0.35f, 0.35f, 0.90f), // blue
      FLinearColor(0.65f, 0.30f, 0.90f), // purple
      FLinearColor(0.90f, 0.30f, 0.60f), // magenta
      FLinearColor(0.90f, 0.50f, 0.75f), // pink
      FLinearColor(0.55f, 0.35f, 0.20f), // brown
      FLinearColor(0.70f, 0.70f, 0.70f), // light grey (kept saturated enough to read)
      FLinearColor(0.15f, 0.85f, 0.85f), // cyan
      FLinearColor(0.60f, 0.85f, 0.35f), // lime
      FLinearColor(0.85f, 0.55f, 0.95f), // lavender
      FLinearColor(0.90f, 0.90f, 0.40f), // pale yellow
  };
  // Spread hashes with splitmix64 so sequential handles do not cluster in palette
  uint64 Hash = ShapeKey ^ (HandleValue * 0x9e3779b97f4a7c15ULL);
  Hash ^= Hash >> 30;
  Hash *= 0xbf58476d1ce4e5b9ULL;
  Hash ^= Hash >> 27;
  Hash *= 0x94d049bb133111ebULL;
  Hash ^= Hash >> 31;
  const uint32 Idx = (uint32)(Hash % 16u);
  // Small value jitter only - do not touch hue, keep palette hue intact
  const float VJitter = (((Hash >> 20) & 0xFF) / 255.0f - 0.5f) * 0.12f;
  FLinearColor C = Palette[Idx];
  C.R = FMath::Clamp(C.R + VJitter, 0.05f, 1.0f);
  C.G = FMath::Clamp(C.G + VJitter, 0.05f, 1.0f);
  C.B = FMath::Clamp(C.B + VJitter, 0.05f, 1.0f);
  return C;
}

UMaterialInstanceDynamic* CreateColoredVizMaterial(UObject* Outer, const FLinearColor& Color) {
  // Use BasicShapeMaterial if available (has exposed "Color" param), otherwise
  // fall back to the default surface material.
  UMaterialInterface* Base = nullptr;
  if (UMaterialInterface* Alt = LoadObject<UMaterialInterface>(
          nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"))) {
    Base = Alt;
  } else {
    Base = UMaterial::GetDefaultMaterial(MD_Surface);
  }
  UMaterialInstanceDynamic* MID = UMaterialInstanceDynamic::Create(Base, Outer);
  if (MID) {
    // BasicShapeMaterial's param is literally "Color" - that is the one that
    // actually tints the mesh. Also set common aliases for future-proofing.
    MID->SetVectorParameterValue(FName(TEXT("Color")), Color);
    MID->SetVectorParameterValue(FName(TEXT("BaseColor")), Color);
    MID->SetVectorParameterValue(FName(TEXT("Base Color")), Color);
    MID->SetVectorParameterValue(FName(TEXT("DiffuseColor")), Color);
    // Log when the base material has no Color parameter, which would leave the mesh untinted.
    {
      FLinearColor Dummy;
      if (!MID->GetVectorParameterValue(FName(TEXT("Color")), Dummy)) {
        UE_LOG(
            LogTemp,
            Verbose,
            TEXT("VizMaterial: Color param not found on base %s"),
            *GetNameSafe(Base));
      }
    }
  }
  return MID;
}

// Also destroys the components other systems attached to the visual (UnrealCV's
// UAnnotationComponent, for one): USceneComponent::OnComponentDestroyed would otherwise re-attach
// them to this component, where they outlive the mesh they shadow.
void DestroyVisualMeshComponent(UMeshComponent* MeshComp) {
  if (!IsValid(MeshComp)) {
    return;
  }
  TArray<USceneComponent*> Descendants;
  MeshComp->GetChildrenComponents(/*bIncludeAllDescendants=*/true, Descendants);
  for (int32 i = Descendants.Num() - 1; i >= 0; --i) {
    if (IsValid(Descendants[i]) && Descendants[i]->GetOwner() == MeshComp->GetOwner()) {
      Descendants[i]->DestroyComponent();
    }
  }
  MeshComp->DestroyComponent();
}
} // namespace

USuperDexBotSceneVisualizationComponent::USuperDexBotSceneVisualizationComponent() {
  PrimaryComponentTick.bCanEverTick = true;
  PrimaryComponentTick.bStartWithTickEnabled = false;
  bTickInEditor = false;
}

void USuperDexBotSceneVisualizationComponent::BeginPlay() {
  Super::BeginPlay();

  bUseRenderModels = !FParse::Param(FCommandLine::Get(), TEXT("NoRenderModels"));
  if (!bUseRenderModels) {
    UE_LOG(LogTemp, Display, TEXT("BotSceneViz: -NoRenderModels; drawing Mochi shapes only."));
  }

  // Subscribe first, then build: if the scene is already loaded the build below covers it, and if
  // it loads later (editor iteration, where a level actor supplies the path) the delegate does.
  // Either order of BeginPlay between this component and whoever triggers the load is fine.
  UGameInstance* GI = GetWorld() ? GetWorld()->GetGameInstance() : nullptr;
  if (USuperDexBotSceneSubsystem* Subsys =
          GI ? GI->GetSubsystem<USuperDexBotSceneSubsystem>() : nullptr) {
    SceneLoadedHandle = Subsys->OnSceneLoaded().AddUObject(
        this, &USuperDexBotSceneVisualizationComponent::RebuildIfNeeded);
    SceneContentsChangedHandle = Subsys->OnSceneContentsChanged().AddUObject(
        this, &USuperDexBotSceneVisualizationComponent::RebuildIfNeeded);
  }
  RebuildIfNeeded();
}

void USuperDexBotSceneVisualizationComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  UGameInstance* GI = GetWorld() ? GetWorld()->GetGameInstance() : nullptr;
  if (USuperDexBotSceneSubsystem* Subsys =
          GI ? GI->GetSubsystem<USuperDexBotSceneSubsystem>() : nullptr) {
    Subsys->OnSceneLoaded().Remove(SceneLoadedHandle);
    Subsys->OnSceneContentsChanged().Remove(SceneContentsChangedHandle);
  }
  SceneLoadedHandle.Reset();
  SceneContentsChangedHandle.Reset();

  DestroyVisuals();
  Super::EndPlay(EndPlayReason);
}

void USuperDexBotSceneVisualizationComponent::CancelPostStepCallback() {
  if (VizSceneObject.IsValid() && VizPostStepHandle.IsValid()) {
    VizSceneObject->CancelCallback(VizPostStepHandle);
    // CancelCallback does not wait for an in-flight callback, which dereferences this component.
    VizSceneObject->QueueOrExecuteCommand([](mochi::Scene*) {});
    VizSceneObject->WaitForQueuedCommands();
  }
  VizPostStepHandle = {};
  VizSceneObject.Reset();
}

void USuperDexBotSceneVisualizationComponent::DestroyVisuals() {
  CancelPostStepCallback();
  {
    FScopeLock Lock(&LatestTransformsLock);
    LatestTransforms.Reset();
  }
  for (auto& V : Visuals) {
    DestroyVisualMeshComponent(V.MeshComp);
  }
  Visuals.Reset();
  for (auto& Skin : Skins) {
    DestroyVisualMeshComponent(Skin.MeshComp);
  }
  Skins.Reset();
  LinkToWorldByHandle.Reset();
  GeometryCache.Reset();
  bHasSceneSnapshot = false;
  SetComponentTickEnabled(false);
}

bool USuperDexBotSceneVisualizationComponent::TrySnapshotLinkTransforms(
    UMochiSceneObject* SceneObject,
    const superdex::robotics::BotScene* BotScene,
    TArray<FBotSceneVizSnapshotEntry>& OutEntries,
    TArray<FBotSceneVizSkinSnapshot>& OutSkins,
    TArray<mochi::ActorHandle>& OutArticulatedActors) const {
  if (!SceneObject || !SceneObject->IsValid() || !BotScene) {
    return false;
  }

  struct SnapshotState {
    TArray<FBotSceneVizSnapshotEntry> Entries;
    TArray<FBotSceneVizSkinSnapshot> Skins;
    TArray<mochi::ActorHandle> ArticulatedActors;
    bool bOk = false;
  };
  TSharedPtr<SnapshotState> State = MakeShared<SnapshotState>();

  SceneObject->QueueOrExecuteCommand([BotScene, State](mochi::Scene* Scene) mutable {
    mochi::ErrorLog ErrLog;
    TArray<FBotSceneVizSnapshotEntry> Local;
    TArray<mochi::ActorHandle> ArticulatedActors;

    // Collect one actor's renderable entries, each with the render model the BotScene recorded for
    // it, if any.
    //
    // An articulated actor holds no shape of its own -- the geometry is on its nested link actors
    // -- so it has to be expanded. Every source of actors goes through here rather than only the
    // bot pass, because a task object can be articulated too: a prefab instance that asked for its
    // reference shape directly got an invalid handle and vanished.
    auto AppendActor = [&Local, &ArticulatedActors, BotScene, Scene](mochi::Actor* Actor) {
      if (!Actor) {
        return;
      }

      // Returns false when there is nothing to draw: no shape and no render model.
      auto AppendEntry = [&Local, BotScene](
                             mochi::Actor* EntryActor, const mochi::TransformRT& WorldFromLink) {
        mochi::Error ShapeErr;
        const mochi::ShapeHandle Shape = EntryActor->GetReferenceShape(ShapeErr);
        const bool bHasShape = ShapeErr.IsOK() && Shape.IsValid();
        const superdex::robotics::ActorRenderModel* RenderModel =
            BotScene->FindRenderModel(EntryActor->GetHandle());
        if (!bHasShape && !RenderModel) {
          return false;
        }
        FBotSceneVizSnapshotEntry E;
        E.LinkHandle = EntryActor->GetHandle();
        if (bHasShape) {
          E.RefShape = Shape;
          E.ShapeKey = Shape.value;
        }
        E.WorldFromLink = WorldFromLink;
        if (RenderModel) {
          E.RenderModel.File = UTF8_TO_TCHAR(RenderModel->file.c_str());
          E.RenderModel.LinkFromModel = RenderModel->actorFromModel;
          E.RenderModel.Scale = RenderModel->scale;
        }
        Local.Add(MoveTemp(E));
        return true;
      };

      if (Actor->GetType() != mochi::ActorType::Articulated) {
        if (!AppendEntry(Actor, Actor->GetRootTransform())) {
          // Warn once rather than per actor per snapshot: a shapeless actor is a standing
          // condition, and the point is to name the actor type that cannot be drawn instead of
          // leaving it to be discovered as a missing object.
          static std::atomic<bool> bLoggedShapelessActor{false};
          if (!bLoggedShapelessActor.exchange(true)) {
            UE_LOG(
                LogTemp,
                Warning,
                TEXT(
                    "BotSceneViz: actor '%hs' (type %d) has no reference shape and will not be "
                    "drawn. Further occurrences are not logged."),
                Actor->GetName(),
                static_cast<int>(Actor->GetType()));
          }
        }
        return;
      }

      ArticulatedActors.Add(Actor->GetHandle());

      mochi::Error LinkErr;
      auto LinkHandles = Actor->GetNestedLinkActors(LinkErr);
      if (!LinkErr.IsOK()) {
        // Warn once, like a shapeless actor: without its links the whole articulation goes undrawn.
        static std::atomic<bool> bLoggedUnreadableLinks{false};
        if (!bLoggedUnreadableLinks.exchange(true)) {
          UE_LOG(
              LogTemp,
              Warning,
              TEXT(
                  "BotSceneViz: articulated actor '%hs' has unreadable nested links (%hs) and will "
                  "not be drawn. Further occurrences are not logged."),
              Actor->GetName(),
              LinkErr.ToString().c_str());
        }
        return;
      }
      TArray<mochi::TransformRT> WorldFromLinks;
      WorldFromLinks.SetNumUninitialized((int32)LinkHandles.size());
      if (LinkHandles.size() > 0) {
        mochi::Error XformErr;
        Actor->GetArticulatedLinkTransforms(
            mochi::Span<mochi::TransformRT>(WorldFromLinks.GetData(), (size_t)WorldFromLinks.Num()),
            XformErr);
        if (!XformErr.IsOK()) {
          WorldFromLinks.Reset();
          WorldFromLinks.SetNumUninitialized((int32)LinkHandles.size());
          for (int32 i = 0; i < (int32)LinkHandles.size(); ++i) {
            mochi::Actor* LinkActor = Scene ? Scene->GetActor(LinkHandles[i]) : nullptr;
            WorldFromLinks[i] =
                LinkActor ? LinkActor->GetRootTransform() : mochi::TransformRT::Identity();
          }
        }
      }

      for (int32 i = 0; i < (int32)LinkHandles.size(); ++i) {
        mochi::Actor* LinkActor = Scene ? Scene->GetActor(LinkHandles[i]) : nullptr;
        if (!LinkActor) {
          continue;
        }
        AppendEntry(
            LinkActor,
            (i < WorldFromLinks.Num()) ? WorldFromLinks[i] : LinkActor->GetRootTransform());
      }
    };

    // 1) Base-scene actors
    for (mochi::ActorHandle H : BotScene->GetBaseSceneActorHandles()) {
      AppendActor(Scene ? Scene->GetActor(H) : nullptr);
    }

    // 2) Spawnable-prefab instances (placed via BotScene::SpawnPrefabInstance). Keyed by instance
    //    name, so several instances of one prefab each get their own visuals.
    for (auto const& InstanceName : BotScene->GetPrefabInstanceNames()) {
      for (mochi::ActorHandle H : BotScene->GetPrefabInstanceActors(
               std::string_view(InstanceName.c_str(), InstanceName.size()))) {
        AppendActor(Scene ? Scene->GetActor(H) : nullptr);
      }
    }

    // 3) Bots.
    for (auto const& BotEntry : BotScene->GetBotScenePrefab().bots) {
      std::string NameStr(BotEntry.name.c_str(), BotEntry.name.size());
      superdex::robotics::Bot* Bot = BotScene->GetBot(NameStr);
      if (!Bot) {
        continue;
      }
      mochi::Actor* const BotActor = Bot->GetArticulatedActor();
      AppendActor(BotActor);

      const superdex::robotics::ActorSkinRenderModel* const Skin =
          BotActor ? BotScene->FindSkinRenderModel(BotActor->GetHandle()) : nullptr;
      if (!Skin) {
        continue;
      }
      FBotSceneVizSkinSnapshot& S = State->Skins.AddDefaulted_GetRef();
      S.ActorKey = BotActor->GetHandle().value;
      S.File = UTF8_TO_TCHAR(Skin->file.c_str());
      S.RootFromModel = Skin->rootFromModel;
      S.Scale = Skin->scale;
      for (int32 i = 0; i < (int32)Skin->boneLinks.size(); ++i) {
        const mochi::ActorHandle Link = Skin->boneLinks[i];
        mochi::Actor* const LinkActor = Link.IsValid() && Scene ? Scene->GetActor(Link) : nullptr;
        S.BoneLinks.Add(LinkActor ? Link.value : kNoLink);
        S.RootFromBoneRest.Add(Skin->rootFromBoneRest[i]);
        S.WorldFromBoneLink.Add(
            LinkActor ? LinkActor->GetRootTransform() : mochi::TransformRT::Identity());
      }
      for (const mochi::ActorHandle Link : Skin->hiddenLinks) {
        S.CoveredLinks.Add(Link.value);
      }
    }

    State->Entries = MoveTemp(Local);
    State->ArticulatedActors = MoveTemp(ArticulatedActors);
    State->bOk = true;
  });

  SceneObject->WaitForQueuedCommands();
  if (!State->bOk) {
    return false;
  }
  OutEntries = MoveTemp(State->Entries);
  OutSkins = MoveTemp(State->Skins);
  OutArticulatedActors = MoveTemp(State->ArticulatedActors);
  return true;
}

void USuperDexBotSceneVisualizationComponent::EnsureVisualsForSnapshot(
    const TArray<FBotSceneVizSnapshotEntry>& Snapshot,
    UMochiSceneObject* SceneObject) {
  // The cache is keyed on the shape alone, but the geometry has the component scale baked in,
  // so everything built under a previous scale has to go when the owner is rescaled.
  const FVector Scale3D = GetComponentTransform().GetScale3D();
  if (GeometryCache.Num() > 0 && !Scale3D.Equals(BakedGeometryScale)) {
    for (auto& V : Visuals) {
      DestroyVisualMeshComponent(V.MeshComp);
    }
    Visuals.Reset();
    GeometryCache.Reset();
  }
  BakedGeometryScale = Scale3D;

  TMap<uint64, int32> HandleToVisualIdx;
  HandleToVisualIdx.Reserve(Visuals.Num());
  for (int32 i = 0; i < Visuals.Num(); ++i) {
    HandleToVisualIdx.Add(Visuals[i].HandleValue, i);
  }

  TSet<uint64> SeenThisFrame;
  SeenThisFrame.Reserve(Snapshot.Num());
  FMochiToUnrealConverter M2U;

  for (auto const& Entry : Snapshot) {
    const uint64 HandleKey = Entry.LinkHandle.value;
    if (SeenThisFrame.Contains(HandleKey)) {
      continue;
    }
    SeenThisFrame.Add(HandleKey);

    const FTransform LinkToWorld = M2U.TransformToUnreal(Entry.WorldFromLink);
    LinkToWorldByHandle.Add(HandleKey, LinkToWorld);
    const FString RenderModelFile = bUseRenderModels ? Entry.RenderModel.File : FString();
    FVisualEntry* V = nullptr;
    if (int32* ExistingIdx = HandleToVisualIdx.Find(HandleKey)) {
      FVisualEntry& Existing = Visuals[*ExistingIdx];
      Existing.LinkToWorld = LinkToWorld;
      // Actor handles are recycled, so a handle seen before can now name a different actor (a task
      // swap respawns its prefab instances). Keep the visual only if it still draws the same thing.
      if (Existing.ShapeKey == Entry.ShapeKey && Existing.RenderModelFile == RenderModelFile) {
        if (IsValid(Existing.MeshComp)) {
          Existing.MeshComp->SetWorldTransform(Existing.ComponentToLink * LinkToWorld);
        }
        continue;
      }
      DestroyVisualMeshComponent(Existing.MeshComp);
      Existing = FVisualEntry{};
      V = &Existing;
    } else {
      V = &Visuals.AddDefaulted_GetRef();
    }

    V->HandleValue = HandleKey;
    V->ShapeKey = Entry.ShapeKey;
    V->RefShape = Entry.RefShape;
    V->LinkToWorld = LinkToWorld;

    if (!RenderModelFile.IsEmpty()) {
      const mochi::Real3& Scale = Entry.RenderModel.Scale;
      V->RenderModelFile = RenderModelFile;
      V->ComponentToLink = M2U.TransformToUnreal(Entry.RenderModel.LinkFromModel);
      V->ComponentToLink.SetScale3D(FVector(Scale[0], Scale[1], Scale[2]));
      // The entry exists before the request: an already-imported model completes inside it.
      RequestRenderModel(HandleKey, RenderModelFile, SceneObject);
      continue;
    }

    V->MeshComp = CreateShapeMeshComponent(Entry.RefShape, HandleKey, SceneObject);
    if (IsValid(V->MeshComp)) {
      V->MeshComp->SetWorldTransform(LinkToWorld);
    }
  }

  // Drops actors gone from the scene, and entries left with nothing to draw.
  for (int32 i = Visuals.Num() - 1; i >= 0; --i) {
    const bool bPending = !Visuals[i].RenderModelFile.IsEmpty();
    if (!SeenThisFrame.Contains(Visuals[i].HandleValue) ||
        (!bPending && !IsValid(Visuals[i].MeshComp))) {
      DestroyVisualMeshComponent(Visuals[i].MeshComp);
      Visuals.RemoveAtSwap(i);
    }
  }
}

void USuperDexBotSceneVisualizationComponent::EnsureSkinsForSnapshot(
    const TArray<FBotSceneVizSkinSnapshot>& Snapshot) {
  // -NoRenderModels draws no skins, so Skins stays empty.
  if (!bUseRenderModels) {
    return;
  }
  FMochiToUnrealConverter M2U;
  UGameInstance* GI = GetWorld() ? GetWorld()->GetGameInstance() : nullptr;
  USuperDexRuntimeMeshSubsystem* Meshes =
      GI ? GI->GetSubsystem<USuperDexRuntimeMeshSubsystem>() : nullptr;
  TSet<uint64> SeenThisFrame;
  for (const FBotSceneVizSkinSnapshot& S : Snapshot) {
    const int32 AnchorBone =
        S.BoneLinks.IndexOfByPredicate([](uint64 Link) { return Link != kNoLink; });
    if (AnchorBone == INDEX_NONE) {
      continue;
    }
    SeenThisFrame.Add(S.ActorKey);
    for (int32 i = 0; i < S.BoneLinks.Num(); ++i) {
      if (S.BoneLinks[i] != kNoLink) {
        LinkToWorldByHandle.Add(S.BoneLinks[i], M2U.TransformToUnreal(S.WorldFromBoneLink[i]));
      }
    }

    // The component follows the anchor bone's link, so the mesh is built relative to that link's
    // rest pose: its bounds then stay with it.
    const mochi::TransformRT ComponentFromRoot = mochi::Invert(S.RootFromBoneRest[AnchorBone]);
    FTransform MeshToComponent = M2U.TransformToUnreal(ComponentFromRoot * S.RootFromModel);
    MeshToComponent.SetScale3D(FVector(S.Scale[0], S.Scale[1], S.Scale[2]));
    TArray<FTransform> RestBoneToComponent;
    for (const mochi::TransformRT& RootFromBoneRest : S.RootFromBoneRest) {
      RestBoneToComponent.Add(M2U.TransformToUnreal(ComponentFromRoot * RootFromBoneRest));
    }

    // Actor handles are recycled: keep a skin only if it is built from the same inputs.
    const int32 ExistingIdx = Skins.IndexOfByPredicate(
        [&S](const FSkinVisual& Skin) { return Skin.ActorKey == S.ActorKey; });
    if (ExistingIdx != INDEX_NONE) {
      FSkinVisual& Existing = Skins[ExistingIdx];
      const auto SameTransform = [](const FTransform& A, const FTransform& B) {
        return A.Equals(B);
      };
      if (Existing.File == S.File && Existing.BoneLinks == S.BoneLinks &&
          SameTransform(Existing.MeshToComponent, MeshToComponent) &&
          Algo::Compare(Existing.RestBoneToComponent, RestBoneToComponent, SameTransform)) {
        Existing.CoveredLinks = S.CoveredLinks;
        continue;
      }
      DestroyVisualMeshComponent(Existing.MeshComp);
      Skins.RemoveAtSwap(ExistingIdx);
    }
    if (!Meshes) {
      continue;
    }

    FSuperDexSkinnedMeshRequest Request;
    Request.FilePath = S.File;
    Request.MeshToComponent = MeshToComponent;
    Request.RestBoneToComponent = RestBoneToComponent;
    for (int32 i = 0; i < S.BoneLinks.Num(); ++i) {
      Request.BoneNames.Add(FName(*FString::Printf(TEXT("skin_bone_%d"), i)));
      Request.JointToBone.Add(S.BoneLinks[i] != kNoLink ? i + 1 : INDEX_NONE);
    }

    FSkinVisual& Skin = Skins.AddDefaulted_GetRef();
    Skin.ActorKey = S.ActorKey;
    Skin.File = S.File;
    Skin.BoneLinks = S.BoneLinks;
    Skin.AnchorBone = AnchorBone;
    Skin.CoveredLinks = S.CoveredLinks;
    Skin.MeshToComponent = MoveTemp(MeshToComponent);
    Skin.RestBoneToComponent = MoveTemp(RestBoneToComponent);
    Skin.RequestId = ++LastSkinRequestId;
    Meshes->RequestSkinnedMesh(
        MoveTemp(Request),
        [WeakThis = TWeakObjectPtr<ThisClass>(this),
         ActorKey = S.ActorKey,
         RequestId = Skin.RequestId](USkeletalMesh* Mesh) {
          if (ThisClass* Self = WeakThis.Get()) {
            Self->OnSkinnedMeshLoaded(ActorKey, RequestId, Mesh);
          }
        });
  }

  for (int32 i = Skins.Num() - 1; i >= 0; --i) {
    if (!SeenThisFrame.Contains(Skins[i].ActorKey)) {
      DestroyVisualMeshComponent(Skins[i].MeshComp);
      Skins.RemoveAtSwap(i);
    }
  }
  UpdateCoveredLinkVisibility();
}

void USuperDexBotSceneVisualizationComponent::OnSkinnedMeshLoaded(
    uint64 ActorKey,
    uint32 RequestId,
    USkeletalMesh* Mesh) {
  FSkinVisual* Skin = Skins.FindByPredicate([ActorKey, RequestId](const FSkinVisual& Entry) {
    return Entry.ActorKey == ActorKey && Entry.RequestId == RequestId;
  });
  // Gone or replaced if the visuals were rebuilt while the mesh was built.
  if (!Skin) {
    return;
  }
  if (!Mesh) {
    UE_LOG(
        LogTemp,
        Warning,
        TEXT("BotSceneViz: cannot draw skin '%s'; drawing the links it covers instead."),
        *Skin->File);
    return;
  }
  UPoseableMeshComponent* Comp =
      Cast<UPoseableMeshComponent>(NewVisualComponent(UPoseableMeshComponent::StaticClass()));
  if (!Comp) {
    return;
  }
  Comp->SetSkinnedAssetAndUpdate(Mesh);
  Comp->SetCollisionEnabled(ECollisionEnabled::NoCollision);
  Skin->MeshComp = Comp;
  UE_LOG(
      LogTemp,
      Log,
      TEXT("BotSceneViz: drawing skin '%s' over %d bones, covering %d links."),
      *Skin->File,
      Skin->BoneLinks.Num(),
      Skin->CoveredLinks.Num());
  PoseSkins();
  UpdateCoveredLinkVisibility();
}

void USuperDexBotSceneVisualizationComponent::UpdateCoveredLinkVisibility() {
  TSet<uint64> Covered;
  for (const FSkinVisual& Skin : Skins) {
    if (IsValid(Skin.MeshComp)) {
      Covered.Append(Skin.CoveredLinks);
    }
  }
  for (const FVisualEntry& V : Visuals) {
    if (IsValid(V.MeshComp)) {
      V.MeshComp->SetVisibility(!Covered.Contains(V.HandleValue));
    }
  }
}

void USuperDexBotSceneVisualizationComponent::PoseSkins() {
  for (const FSkinVisual& Skin : Skins) {
    UPoseableMeshComponent* Comp = Skin.MeshComp;
    const FTransform* AnchorToWorld = LinkToWorldByHandle.Find(Skin.BoneLinks[Skin.AnchorBone]);
    if (!IsValid(Comp) || !AnchorToWorld) {
      continue;
    }
    Comp->SetWorldTransform(*AnchorToWorld);
    // Skin bone i is skeleton bone i + 1, a child of the root at the component origin, so its
    // bone-space transform is its component-space one.
    for (int32 i = 0; i < Skin.BoneLinks.Num() && i + 1 < Comp->BoneSpaceTransforms.Num(); ++i) {
      if (const FTransform* LinkToWorld = LinkToWorldByHandle.Find(Skin.BoneLinks[i])) {
        Comp->BoneSpaceTransforms[i + 1] = LinkToWorld->GetRelativeTransform(*AnchorToWorld);
      }
    }
    Comp->MarkRefreshTransformDirty();
    Comp->RefreshBoneTransforms();
  }
}

UMeshComponent* USuperDexBotSceneVisualizationComponent::CreateShapeMeshComponent(
    mochi::ShapeHandle RefShape,
    uint64 HandleKey,
    UMochiSceneObject* SceneObject) {
  if (!RefShape.IsValid()) {
    return nullptr;
  }
  const uint64 ShapeKey = RefShape.value;

  TSharedPtr<UE::Geometry::FDynamicMesh3>* CachedMesh = GeometryCache.Find(ShapeKey);
  if (!CachedMesh) {
    auto NewMesh = MakeShared<UE::Geometry::FDynamicMesh3>();
    // Build once via the MochiPhysics utility path: visual mesh preferred,
    // fallback to actor handle (handles Acquire from AsyncScene if needed).
    {
      UE::Geometry::FDynamicMesh3 Tmp;
      PopulateMeshFromMochiShape(
          &Tmp,
          RefShape,
          /*bUseVisualMeshIfPresent=*/true,
          /*bUseContactSkinIfPresent=*/false);
      if (Tmp.VertexCount() == 0) {
        Tmp.Clear();
        PopulateMeshFromMochiActor(&Tmp, mochi::ActorHandle(HandleKey), SceneObject, true);
      }
      // Copy into the shared mesh we cache (keeps geometry even if Tmp goes out of scope)
      NewMesh->Copy(Tmp);
    }
    if (NewMesh->VertexCount() == 0) {
      return nullptr;
    }
    ComputeOrDiscardVertexNormals(*NewMesh, /*bComputeNormals=*/true);
    PrepareDynamicMeshForRuntime(BakedGeometryScale, *NewMesh);
    CachedMesh = &GeometryCache.Add(ShapeKey, MoveTemp(NewMesh));
  }

  UDynamicMeshComponent* DynComp =
      Cast<UDynamicMeshComponent>(NewVisualComponent(UDynamicMeshComponent::StaticClass()));
  if (!DynComp) {
    return nullptr;
  }
  DynComp->SetEnableFlatShading(false);
  DynComp->EditMesh([CachedMesh](UE::Geometry::FDynamicMesh3& Mesh) { Mesh.Copy(**CachedMesh); });
  DynComp->NotifyMeshUpdated();

  // Distinct color per mesh so the scene is not all-grey - deterministic from
  // handle identity so each link keeps its color across rebuilds.
  {
    const FLinearColor Color = GetDistinctVizColor(ShapeKey, HandleKey);
    if (UMaterialInstanceDynamic* MID = CreateColoredVizMaterial(DynComp, Color)) {
      DynComp->SetMaterial(0, MID);
    }
  }

  DynComp->SetCollisionEnabled(ECollisionEnabled::QueryOnly);
  DynComp->SetCollisionObjectType(ECC_WorldStatic);
  return DynComp;
}

UMeshComponent* USuperDexBotSceneVisualizationComponent::NewVisualComponent(
    TSubclassOf<UMeshComponent> ComponentClass) {
  AActor* Owner = GetOwner();
  if (!Owner) {
    return nullptr;
  }
  UMeshComponent* Comp = NewObject<UMeshComponent>(Owner, ComponentClass);
  Comp->SetMobility(EComponentMobility::Movable);
  Owner->AddInstanceComponent(Comp);
  Comp->RegisterComponent();
  Comp->AttachToComponent(this, FAttachmentTransformRules::KeepWorldTransform);
  return Comp;
}

void USuperDexBotSceneVisualizationComponent::RequestRenderModel(
    uint64 HandleKey,
    const FString& File,
    UMochiSceneObject* SceneObject) {
  UGameInstance* GI = GetWorld() ? GetWorld()->GetGameInstance() : nullptr;
  USuperDexRuntimeMeshSubsystem* Meshes =
      GI ? GI->GetSubsystem<USuperDexRuntimeMeshSubsystem>() : nullptr;
  if (!Meshes) {
    OnRenderModelLoaded(HandleKey, File, nullptr, SceneObject);
    return;
  }
  Meshes->RequestStaticMesh(
      File,
      [WeakThis = TWeakObjectPtr<ThisClass>(this),
       WeakScene = TWeakObjectPtr<UMochiSceneObject>(SceneObject),
       HandleKey,
       File](UStaticMesh* Mesh) {
        if (ThisClass* Self = WeakThis.Get()) {
          Self->OnRenderModelLoaded(HandleKey, File, Mesh, WeakScene.Get());
        }
      });
}

void USuperDexBotSceneVisualizationComponent::OnRenderModelLoaded(
    uint64 HandleKey,
    const FString& File,
    UStaticMesh* Mesh,
    UMochiSceneObject* SceneObject) {
  FVisualEntry* V = Visuals.FindByPredicate([HandleKey, &File](const FVisualEntry& Entry) {
    return Entry.HandleValue == HandleKey && Entry.RenderModelFile == File;
  });
  // Gone or already drawn if the visuals were rebuilt while the import ran.
  if (!V || IsValid(V->MeshComp)) {
    return;
  }

  if (Mesh) {
    UStaticMeshComponent* Comp =
        Cast<UStaticMeshComponent>(NewVisualComponent(UStaticMeshComponent::StaticClass()));
    if (!Comp) {
      return;
    }
    Comp->SetStaticMesh(Mesh);
    Comp->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    V->MeshComp = Comp;
  } else {
    V->ComponentToLink = FTransform::Identity;
    V->MeshComp = CreateShapeMeshComponent(V->RefShape, V->HandleValue, SceneObject);
    if (!V->MeshComp) {
      return;
    }
  }
  V->MeshComp->SetWorldTransform(V->ComponentToLink * V->LinkToWorld);
  UpdateCoveredLinkVisibility();
}

void USuperDexBotSceneVisualizationComponent::RebuildIfNeeded() {
  UWorld* World = GetWorld();
  if (!World) {
    return;
  }
  UGameInstance* GI = World->GetGameInstance();
  if (!GI) {
    return;
  }
  USuperDexBotSceneSubsystem* Subsys = GI->GetSubsystem<USuperDexBotSceneSubsystem>();
  UMochiGameInstanceSubsystem* MochiSubsys = GI->GetSubsystem<UMochiGameInstanceSubsystem>();
  if (!Subsys || !MochiSubsys) {
    return;
  }

  if (Subsys->GetState() != ESuperDexBotSceneState::Loaded) {
    if (Visuals.Num() > 0 || Skins.Num() > 0 || GeometryCache.Num() > 0) {
      DestroyVisuals();
    }
    SetComponentTickEnabled(false);
    bHasSceneSnapshot = false;
    return;
  }

  UMochiSceneObject* SceneObject = MochiSubsys->GetScene();
  const superdex::robotics::BotScene* BotScene = Subsys->GetBotScene();
  if (!SceneObject || !SceneObject->IsValid() || !BotScene) {
    return;
  }

  TArray<FBotSceneVizSnapshotEntry> Snapshot;
  TArray<FBotSceneVizSkinSnapshot> SkinSnapshot;
  TArray<mochi::ActorHandle> ArticulatedActors;
  if (!TrySnapshotLinkTransforms(
          SceneObject, BotScene, Snapshot, SkinSnapshot, ArticulatedActors)) {
    return;
  }

  // Handles are recycled, so both passes refill the link transforms from scratch.
  LinkToWorldByHandle.Reset();
  EnsureVisualsForSnapshot(Snapshot, SceneObject);
  EnsureSkinsForSnapshot(SkinSnapshot);
  bHasSceneSnapshot = true;
  SetComponentTickEnabled(Visuals.Num() > 0 || Skins.Num() > 0);

  // Register async publisher: Mochi PostStep writes LatestTransforms, Game tick
  // polls it without any QueueOrExecuteCommand+WaitForQueuedCommands.
  CancelPostStepCallback();
  {
    TArray<uint64> Handles;
    Handles.Reserve(Visuals.Num());
    for (auto const& V : Visuals) {
      Handles.Add(V.HandleValue);
    }
    for (auto const& Skin : Skins) {
      for (uint64 Link : Skin.BoneLinks) {
        if (Link != kNoLink) {
          Handles.AddUnique(Link);
        }
      }
    }
    TWeakObjectPtr<USuperDexBotSceneVisualizationComponent> WeakThis(this);
    VizSceneObject = SceneObject;
    // Captures actor handles rather than the BotScene: the callback runs on the Mochi thread and
    // must not depend on the subsystem's BotScene outliving it. A destroyed actor resolves to null.
    VizPostStepHandle = SceneObject->RegisterPostStepCallback(
        "BotSceneViz::PublishTransforms",
        [WeakThis, Handles, ArticulatedActors](mochi::StepInfo const& Info) {
          auto* Self = WeakThis.Get();
          if (!Self || !Info.scene) {
            return;
          }
          // Bulk lookup from articulated link transforms; fallback to GetRootTransform.
          TMap<uint64, mochi::TransformRT> Bulk;
          Bulk.Reserve(Handles.Num());
          for (mochi::ActorHandle ArtHandle : ArticulatedActors) {
            mochi::Actor* ArtActor = Info.scene->GetActor(ArtHandle);
            if (!ArtActor) {
              continue;
            }
            mochi::Error E1;
            auto LinkHandles = ArtActor->GetNestedLinkActors(E1);
            if (!E1.IsOK() || LinkHandles.size() == 0) {
              continue;
            }
            TArray<mochi::TransformRT> WFT;
            WFT.SetNumUninitialized((int32)LinkHandles.size());
            mochi::Error E2;
            ArtActor->GetArticulatedLinkTransforms(
                mochi::Span<mochi::TransformRT>(WFT.GetData(), (size_t)WFT.Num()), E2);
            if (E2.IsOK() && WFT.Num() == (int32)LinkHandles.size()) {
              Bulk.Reserve(Bulk.Num() + WFT.Num());
              for (int32 i = 0; i < WFT.Num(); ++i) {
                Bulk.Add(LinkHandles[i].value, WFT[i]);
              }
            }
          }
          TMap<uint64, mochi::TransformRT> Published;
          Published.Reserve(Handles.Num());
          for (uint64 H : Handles) {
            if (mochi::TransformRT* BulkT = Bulk.Find(H)) {
              Published.Add(H, *BulkT);
            } else {
              mochi::Actor* A = Info.scene->GetActor(mochi::ActorHandle(H));
              if (A) {
                Published.Add(H, A->GetRootTransform());
              }
            }
          }
          FScopeLock Lock(&Self->LatestTransformsLock);
          // Work around TMap cross-allocator move-assignment deduction on UBT
          Self->LatestTransforms.Reset();
          for (auto& Pair : Published) {
            Self->LatestTransforms.Add(Pair.Key, Pair.Value);
          }
        });
  }
}

void USuperDexBotSceneVisualizationComponent::TickComponent(
    float DeltaTime,
    ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction) {
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

  if (!bHasSceneSnapshot || (Visuals.Num() == 0 && Skins.Num() == 0)) {
    return;
  }
  // Poll latest published transforms - no Mochi sync, matches legacy
  // UMochiArticulatedActorObject::TryGetQueryData polling.
  TMap<uint64, mochi::TransformRT> Snapshot;
  {
    FScopeLock Lock(&LatestTransformsLock);
    Snapshot.Reserve(LatestTransforms.Num());
    for (auto const& Pair : LatestTransforms) {
      Snapshot.Add(Pair.Key, Pair.Value);
    }
  }
  if (Snapshot.Num() == 0) {
    return;
  }
  FMochiToUnrealConverter M2U;
  for (auto const& Pair : Snapshot) {
    LinkToWorldByHandle.Add(Pair.Key, M2U.TransformToUnreal(Pair.Value));
  }
  for (auto& V : Visuals) {
    if (const FTransform* LinkToWorld = LinkToWorldByHandle.Find(V.HandleValue)) {
      V.LinkToWorld = *LinkToWorld;
      if (IsValid(V.MeshComp)) {
        V.MeshComp->SetWorldTransform(V.ComponentToLink * V.LinkToWorld);
      }
    }
  }
  PoseSkins();
}
