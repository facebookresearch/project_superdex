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

#include "SuperDexBotComponent.h"

#include "ISuperDexRoboticsModule.h"
#include "MochiConversions.h"
#include "MochiGameInstanceSubsystem.h"
#include "MochiPhysicsModule.h"
#include "SuperDexBot.h"
#include "SuperDexBotLoader.h"

#include "Engine/GameInstance.h"
#include "Engine/SkeletalMesh.h"
#include "Engine/World.h"
#include "Misc/Paths.h"

#include <superdex_robotics/utils/bot_utils.h>

namespace {
bool IsSceneExternallyOwned(const UWorld* World) {
  if (!World) {
    return false;
  }
  if (const UGameInstance* GI = World->GetGameInstance()) {
    if (const auto* Subsys = GI->GetSubsystem<UMochiGameInstanceSubsystem>()) {
      return Subsys->IsSceneExternallyOwned();
    }
  }
  return false;
}
} // namespace

#if WITH_EDITOR
#include "Editor.h"
#include "Subsystems/ImportSubsystem.h"
#endif

USuperDexBotComponent::USuperDexBotComponent() {
  PrimaryComponentTick.bCanEverTick = true;
  PrimaryComponentTick.TickGroup = TG_StartPhysics;
  EnsureArrayProperties();
}

void USuperDexBotComponent::SetBotAsset(USuperDexBot* InBot) {
  if (ArticulatedActorObject) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[%hs] Attempted to set SuperDexBot asset but bot has already spawned [%s]!"),
        __FUNCTION__,
        *GetName());
    return;
  }
  LastBotCreationFailure.Reset();
  if (InBot) {
    Bot = InBot;
    if (Bot->GetSkeletalMesh()) {
      // The bot was replaced so update the skinned asset as a convinience and to prevent incorrect
      // bone transforms. This is done whenever the bot is intentionally changed.
      SetSkinnedAssetAndUpdate(Bot->GetSkeletalMesh());
    }
    EnsureArrayProperties();
    PoseMeshFromJointPositions(InitialPose);
  } else {
    Bot = nullptr;
    SetSkinnedAssetAndUpdate(nullptr);
    InitialPose.Reset();
    ContactGeneratingLinks.Reset();
  }
}

USuperDexBot* USuperDexBotComponent::GetBotAsset() const {
  return Bot;
}

bool USuperDexBotComponent::SetInitialPose(const TArray<double>& NewInitialPose) {
  if (ArticulatedActorObject == nullptr) {
    InitialPose = NewInitialPose;
    return true;
  }
  return false;
}

const TArray<double>& USuperDexBotComponent::GetInitialPose() const {
  return InitialPose;
}

bool USuperDexBotComponent::HasGravity() const {
  return bHasGravity;
}

UMochiArticulatedActorObject* USuperDexBotComponent::GetMochiArticulatedActor() const {
  return ArticulatedActorObject;
}

void USuperDexBotComponent::SetWorldTransform(const FTransform& WorldTransform) {
  if (ArticulatedActorObject) {
    ArticulatedActorObject->SetRootTransform(WorldTransform);
  }
}

void USuperDexBotComponent::SetPose(TArray<double> const& Pose) {
  if (ArticulatedActorObject) {
    ArticulatedActorObject->SetArticulatedPoseFromJoints(Pose);
  }
}

FTransform USuperDexBotComponent::GetLinkToWorldTransform(FName LinkName) const {
  if (ArticulatedActorObject) {
    const int iLink = ArticulatedActorObject->GetNestedLinkActorIndex(LinkName);
    if (iLink != INDEX_NONE && iLink < QueriedLinkTransforms.Num()) {
      return QueriedLinkTransforms[iLink];
    }
  }
  return FTransform();
}

const TArray<double>& USuperDexBotComponent::GetCurrentJointPositions() const {
  return CurrentJointPositions;
}

const TArray<double>& USuperDexBotComponent::GetCurrentJointVelocities() const {
  return CurrentJointVelocities;
}

bool USuperDexBotComponent::SetController(TSubclassOf<USuperDexBotController> NewControllerClass) {
  FScopeLock lock(&BotControllerMutex);
  if (BotController != nullptr) {
    BotController->Shutdown(this);
    BotController = nullptr;
  }
  BotControllerClass = NewControllerClass;
  if (BotControllerClass) {
    BotController = NewObject<USuperDexBotController>(this, BotControllerClass);
    if (BotController) {
      if (!BotController->Initialize(this)) {
        UE_LOG(
            LogMochiPhysics,
            Warning,
            TEXT("[USuperDexBotComponent] Failed to initialize SuperDexBotController of class %s"),
            *BotControllerClass->GetName());
        BotController = nullptr;
        return false;
      }
    } else {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[USuperDexBotComponent] Failed to create SuperDexBotController of class %s"),
          *BotControllerClass->GetName());
      return false;
    }
  }
  return true;
}

USuperDexBotController* USuperDexBotComponent::GetController() const {
  return BotController;
}

bool USuperDexBotComponent::GetEnableContacts() const {
  return bEnableContacts;
}

const TArray<FName>& USuperDexBotComponent::GetContactGeneratingLinks() const {
  return ContactGeneratingLinks;
}

bool USuperDexBotComponent::GetSkinGeneratesContacts() const {
  return bSkinGeneratesContacts;
}

bool USuperDexBotComponent::GetFilterSelfContacts() const {
  return bFilterSelfContacts;
}

void USuperDexBotComponent::HandleBotControllerOnBotSpawned(USuperDexBotComponent* InBot) {
  if (BotController) {
    BotController->OnBotSpawned(InBot);
  }
}

void USuperDexBotComponent::BeginPlay() {
  Super::BeginPlay();

  // A runtime-loaded .superdex_scene already created this level's bots, so a placed component
  // must not create a second copy. Its owning actor is normally not spawned at all in that mode;
  // this guard covers bots authored directly into the level.
  if (IsSceneExternallyOwned(GetWorld())) {
    UE_LOG(
        LogMochiPhysics,
        Verbose,
        TEXT("[%hs] Scene is externally owned; skipping bot init on %s/%s"),
        __FUNCTION__,
        *GetNameSafe(GetOwner()),
        *GetName());
    return;
  }
  // TODO: superdex_robotics does not yet support loading a bot into coordinate spaces
  // other than RH, Z up. Until it does, throw an error and ensure.
  const auto& activeSpace = UMochiPhysicsSettings::GetActiveSimulationSpace();
  if (activeSpace != kSuperDexSpace) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[%hs] SuperDexBotComponent only supports a RH, Z-up space! Please reconfigured your project settings!"),
        __FUNCTION__);
    ensure(false);
    return;
  }
  if (Bot) {
    if (BotController == nullptr) {
      SetController(BotControllerClass);
    }
    if (BotController) {
      OnBotSpawned.AddDynamic(this, &USuperDexBotComponent::HandleBotControllerOnBotSpawned);
    }
  } else {
    UE_LOG(
        LogMochiPhysics, Error, TEXT("[%hs] Bot asset was null on %s!"), __FUNCTION__, *GetName());
  }
}

void USuperDexBotComponent::TickComponent(
    float DeltaTime,
    ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction) {
  if (GetWorld()->IsGameWorld()) {
    // A borrowed bot must keep ticking -- polling its query data is what poses the skeletal mesh
    // from the scene's articulation, which the teleop clutch depends on. Only an unbound component
    // has nothing to read.
    if (IsSceneExternallyOwned(GetWorld()) && !bIsBorrowedBot) {
      Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
      return;
    }
    if (ArticulatedActorObject == nullptr && Bot && bAutoSpawn) {
      CreateBotAndArticulatedActor();
    }
    if (ArticulatedActorObject) {
      ArticulatedActorObject->TryGetQueryData(
          [this](const FMochiArticulatedActorQueryData& QueryData) {
            QueriedLinkTransforms = QueryData.LinksToWorld;
            CurrentJointPositions = QueryData.JointPositions;
            CurrentJointVelocities = QueryData.JointVelocities;
            SetWorldLocationAndRotation(
                QueriedLinkTransforms[0].GetLocation(), QueriedLinkTransforms[0].GetRotation());
            PoseMeshFromLinkWorldTransforms(QueriedLinkTransforms);
            UpdateBounds();
            if (bEnableContacts) {
              OnMochiComponentBeginContactProcessing.Broadcast(this);
              NestedContactEventHandler.Update(
                  this,
                  OnMochiComponentBeginNestedContact,
                  OnMochiComponentEndNestedContact,
                  OnMochiComponentNestedContact,
                  QueryData.ContactPoints,
                  bFilterSelfContacts);
              ContactEventHandler.Update(
                  this,
                  OnMochiComponentBeginContact,
                  OnMochiComponentEndContact,
                  OnMochiComponentContact,
                  QueryData.ContactPoints,
                  bFilterSelfContacts);
              OnMochiComponentEndContactProcessing.Broadcast(this);
            }
          });
    }
    // SuperDexBotController game thread tick
    if (ArticulatedActorObject && BotController) {
      FScopeLock lock(&BotControllerMutex);
      BotController->TickController(this, DeltaTime);
    }
  }
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
}

void USuperDexBotComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  MOCHI_PROFILE_SCOPE();
  SetController(nullptr);
  Super::EndPlay(EndPlayReason);
  DestroyBotAndArticulatedActor();
}

void USuperDexBotComponent::CreateBotAndArticulatedActor() {
  MOCHI_PROFILE_SCOPE();
  if (IsSceneExternallyOwned(GetWorld())) {
    UE_LOG(
        LogMochiPhysics,
        Verbose,
        TEXT("[%hs] Scene is externally owned; skipping bot creation on %s/%s"),
        __FUNCTION__,
        *GetNameSafe(GetOwner()),
        *GetName());
    return;
  }
  if (!GetOwner()->HasAuthority()) {
    return;
  }
  double Scale = 1.0;
  const EBotScaleFit ScaleFit = ResolveUniformBotScale(GetComponentScale(), Scale);
  if (ScaleFit == EBotScaleFit::Degenerate) {
    const FString Failure = TEXT("DegenerateScale");
    if (LastBotCreationFailure != Failure) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT(
              "[%hs] Cannot create bot '%s' at the degenerate scale %s of component '%s' on actor '%s'."),
          __FUNCTION__,
          *GetPathNameSafe(Bot),
          *GetComponentScale().ToString(),
          *GetName(),
          *GetNameSafe(GetOwner()));
      LastBotCreationFailure = Failure;
    }
    return;
  }
  DestroyBotAndArticulatedActor();
  if (Bot == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%hs] Failed to create articulated actor because Bot was null!"),
        __FUNCTION__);
    return;
  }
  FSuperDexBotLoader Loader;
  mochi::ErrorLog e{};
  superdex::robotics::BotPrefab BotPrefab = Bot->GetBotPrefab();
  // override default pose if InitialPose is set
  if (InitialPose.Num() == BotPrefab._numDofs) {
    for (int i = 0; i < BotPrefab._numDofs; ++i) {
      BotPrefab.defaultPose[i] = static_cast<mochi::real>(InitialPose[i]);
    }
  } else if (InitialPose.Num() > 0) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[%hs] InitialPose size (%d) does not match DOF count (%d); using default pose."),
        __FUNCTION__,
        InitialPose.Num(),
        BotPrefab._numDofs);
  }
  ApplyPrefabOverrides(BotPrefab);
  if (ScaleFit == EBotScaleFit::NonUniform) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "[%hs] Bots only scale uniformly; creating bot '%s' of component '%s' on actor '%s' at scale %f instead of %s."),
        __FUNCTION__,
        *GetPathNameSafe(Bot),
        *GetName(),
        *GetNameSafe(GetOwner()),
        Scale,
        *GetComponentScale().ToString());
  }
  BotScale = static_cast<float>(Scale);
  BotPrefab.scale = static_cast<mochi::real>(Scale);
  // set world transform from component location (convert UE → mochi simulation space)
  FUnrealToMochiConverter u2m;
  BotPrefab.worldFromRoot = u2m.TransformToMochi_RT(
      FTransform(GetComponentQuat(), GetComponentLocation(), FVector::OneVector));
  // get scene
  const UGameInstance* GameInstance = GetWorld()->GetGameInstance();
  const UMochiGameInstanceSubsystem* Subsys =
      GameInstance->GetSubsystem<UMochiGameInstanceSubsystem>();
  UMochiSceneObject* SceneObject = Subsys ? Subsys->GetScene() : nullptr;
  if (!SceneObject || !SceneObject->IsValid()) {
    const FString Failure = TEXT("InvalidMochiScene");
    if (LastBotCreationFailure != Failure) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[%hs] Failed to get valid Mochi scene for bot '%s' on actor '%s'!"),
          __FUNCTION__,
          *GetPathNameSafe(Bot),
          *GetNameSafe(GetOwner()));
      LastBotCreationFailure = Failure;
    }
    return;
  }
  // create articulated actor on the Mochi thread via AddToScene
  mochi::ActorHandle ActorHandle;
  SceneObject->QueueOrExecuteCommand([&](mochi::Scene* SyncScene) {
    auto* botsCtx = ISuperDexRoboticsModule::Get()->GetRoboticsContext();
    BotPtr = botsCtx->CreateBot(SyncScene, BotPrefab, Loader, e);
    mochi::Actor* Actor = BotPtr ? BotPtr->GetArticulatedActor() : nullptr;
    if (Actor) {
      ActorHandle = Actor->GetHandle();
    }
  });
  SceneObject->WaitForQueuedCommands();
  if (!e.IsOK() || !ActorHandle.IsValid()) {
    const char* ErrorDescription = e._error.GetDescription();
    if (ErrorDescription == nullptr) {
      ErrorDescription = "Unknown bot creation failure";
    }
    const FString Failure = FString::Printf(TEXT("CreateBot:%hs"), ErrorDescription);
    if (LastBotCreationFailure != Failure) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[%hs] Failed to add bot '%s' (component '%s' on actor '%s') to scene: %hs"),
          __FUNCTION__,
          *GetPathNameSafe(Bot),
          *GetName(),
          *GetNameSafe(GetOwner()),
          ErrorDescription);
      // superdex_robotics only emits this exact error for a filesystem params load. List the
      // paths once; unrelated bot-creation failures should not implicate every file-backed sensor.
      const bool bParamsLoadFailure =
          std::string_view(ErrorDescription).find("Failed to load params from file") !=
          std::string_view::npos;
      if (bParamsLoadFailure) {
        for (const auto& Link : BotPrefab.links) {
          for (const auto& Sensor : Link.sensors) {
            const std::string_view Params(Sensor.params.c_str(), Sensor.params.size());
            const ESuperDexBotComponentParamsKind ParamsKind = ClassifyBotComponentParams(Params);
            if (ParamsKind == ESuperDexBotComponentParamsKind::Empty ||
                ParamsKind == ESuperDexBotComponentParamsKind::InlineJson) {
              continue;
            }
            const FString ParamsPath = UTF8_TO_TCHAR(Sensor.params.c_str());
            FString PathToCheck = ParamsPath;
            const TCHAR* Availability = nullptr;
            if (ParamsKind == ESuperDexBotComponentParamsKind::CapturedFile) {
              Availability = TEXT("captured asset data; materialization did not complete");
            } else if (ParamsKind == ESuperDexBotComponentParamsKind::LoaderPath) {
              Availability = TEXT("loader-specific path; existence not checked");
            } else {
              PathToCheck = FPaths::ConvertRelativePathToFull(PathToCheck);
              Availability = FPaths::FileExists(PathToCheck) ? TEXT("present")
                                                             : TEXT("MISSING ON THIS MACHINE");
            }
            UE_LOG(
                LogMochiPhysics,
                Error,
                TEXT("[%hs]   sensor '%hs' (type '%hs') on link '%hs' reads params '%s' -- %s"),
                __FUNCTION__,
                Sensor.name.c_str(),
                Sensor.type.c_str(),
                Link.name.c_str(),
                *PathToCheck,
                Availability);
          }
        }
      }
      LastBotCreationFailure = Failure;
    }
    return;
  }
  LastBotCreationFailure.Reset();
  // wrap with UE infrastructure
  ArticulatedActorObject = UMochiArticulatedActorObject::CreateFromExistingArticulatedActor(
      this,
      GetOwner()->GetName() + TEXT("___") + GetName(),
      ActorHandle,
      nullptr,
      [this](
          FMochiArticulatedActorPoseVariant& WorldFromLinks,
          mochi::Actor* ArticulatedActor,
          mochi::Scene* Scene,
          double TimeStepSec) {
        return OnArticulatedActorControl(WorldFromLinks, ArticulatedActor, Scene, TimeStepSec);
      },
      bEnableContacts ? ContactGeneratingLinks : TArray<FName>{},
      bEnableContacts && bSkinGeneratesContacts,
      SceneObject);
  if (ArticulatedActorObject && ArticulatedActorObject->IsValid()) {
    // The destruction of BotPtr by superdex::robotics::DestroyBot will destroy the articulation.
    ArticulatedActorObject->bDestroyMochiActorOnObjectDestruction = false;
    // Validate against the actual runtime driving names; any link without a matching
    // bone will be silently skipped by PoseMeshFromLinkWorldTransforms each tick.
    WarnOnUnmatchedLinkBones(ArticulatedActorObject->GetNestedLinkActorNames());
    // The mesh is posed with scaled bone transforms (PoseMeshFromLinkWorldTransforms).
    SetWorldScale3D(FVector(BotScale));
    OnBotSpawned.Broadcast(this);
    bArticulatedActorReady = true;
  }
}

USuperDexBotComponent::EBotScaleFit USuperDexBotComponent::ResolveUniformBotScale(
    const FVector& ComponentScale,
    double& OutScale) {
  const FVector Magnitude = ComponentScale.GetAbs();
  OutScale = (Magnitude.X + Magnitude.Y + Magnitude.Z) / 3.0;
  if (ComponentScale.ContainsNaN() || Magnitude.GetMin() <= UE_KINDA_SMALL_NUMBER) {
    return EBotScaleFit::Degenerate;
  }
  return ComponentScale.Equals(FVector(OutScale), UE_KINDA_SMALL_NUMBER * OutScale)
      ? EBotScaleFit::Uniform
      : EBotScaleFit::NonUniform;
}

bool USuperDexBotComponent::SpawnBot(const FTransform& WorldFromRoot, float Scale) {
  if (ArticulatedActorObject || BotPtr) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("[%hs] %s already has a bot"), __FUNCTION__, *GetName());
    return false;
  }
  if (!(Scale > 0.f) || !FMath::IsFinite(Scale)) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[%hs] invalid scale %f"), __FUNCTION__, Scale);
    return false;
  }
  SetWorldLocationAndRotation(WorldFromRoot.GetLocation(), WorldFromRoot.GetRotation());
  SetWorldScale3D(FVector(Scale));
  CreateBotAndArticulatedActor();
  return IsBotSpawned();
}

bool USuperDexBotComponent::IsBotSpawned() const {
  return ArticulatedActorObject != nullptr && ArticulatedActorObject->IsValid();
}

bool USuperDexBotComponent::BindToExistingBot(superdex::robotics::Bot* SceneBot) {
  if (!SceneBot) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("[%hs] SceneBot is null"), __FUNCTION__);
    return false;
  }
  if (ArticulatedActorObject || BotPtr) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("[%hs] %s already has a bot"), __FUNCTION__, *GetName());
    return false;
  }
  const UGameInstance* GameInstance = GetWorld() ? GetWorld()->GetGameInstance() : nullptr;
  const UMochiGameInstanceSubsystem* Subsystem =
      GameInstance ? GameInstance->GetSubsystem<UMochiGameInstanceSubsystem>() : nullptr;
  UMochiSceneObject* SceneObject = Subsystem ? Subsystem->GetScene() : nullptr;
  if (!SceneObject || !SceneObject->IsValid()) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("[%hs] no valid Mochi scene"), __FUNCTION__);
    return false;
  }

  // The BotScene owns this bot and its simulation steps it asynchronously, so everything read
  // off it has to be read on the scene thread: a game-thread traversal races a step or a
  // teardown and can hand back an actor that is already gone. Collect the actor handle, the
  // prefab link names, and the user data the wrapper is about to overwrite in one pass.
  mochi::ActorHandle ActorHandle{};
  TArray<FName> PrefabLinkNames;
  float BorrowedScale = 1.f;
  BorrowedActorUserData.Reset();
  SceneObject->QueueOrExecuteCommand([&](mochi::Scene* Scene) {
    mochi::Actor* Actor = SceneBot->GetArticulatedActor();
    if (!Actor || !Actor->GetHandle().IsValid()) {
      return;
    }
    ActorHandle = Actor->GetHandle();
    for (auto const& Link : SceneBot->GetBotPrefab().links) {
      PrefabLinkNames.Add(FName(UTF8_TO_TCHAR(Link.name.c_str())));
    }
    BorrowedScale = static_cast<float>(SceneBot->GetBotPrefab().scale);
    BorrowedActorUserData.Add({ActorHandle, Actor->GetUserData()});
    mochi::ErrorLog e{};
    auto SaveNestedUserData = [&](auto const& Handles) {
      for (mochi::ActorHandle const Handle : Handles) {
        if (mochi::Actor* Nested = Scene->GetActor(Handle)) {
          BorrowedActorUserData.Add({Handle, Nested->GetUserData()});
        }
      }
    };
    SaveNestedUserData(Actor->GetNestedLinkActors(e));
    SaveNestedUserData(Actor->GetNestedSoftActors(e));
  });
  SceneObject->WaitForQueuedCommands();
  if (!ActorHandle.IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[%hs] scene bot has no valid articulated actor"),
        __FUNCTION__);
    BorrowedActorUserData.Reset();
    return false;
  }

  // Contact queries are registered when the wrapper is created, so the link list has to be final
  // before that call.
  if (bEnableContacts && ContactGeneratingLinks.Num() == 0) {
    ContactGeneratingLinks = PrefabLinkNames;
  }
  ArticulatedActorObject = UMochiArticulatedActorObject::CreateFromExistingArticulatedActor(
      this,
      GetOwner() ? GetOwner()->GetName() + TEXT("___") + GetName() : GetName(),
      ActorHandle,
      nullptr,
      nullptr,
      bEnableContacts ? ContactGeneratingLinks : TArray<FName>{},
      bEnableContacts && bSkinGeneratesContacts,
      SceneObject);
  if (!ArticulatedActorObject || !ArticulatedActorObject->IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%hs] failed to wrap the scene bot's articulated actor"),
        __FUNCTION__);
    ArticulatedActorObject = nullptr;
    // Setup stamps user data as it walks the actor tree, so a failure partway through still
    // leaves part of it overwritten.
    RestoreBorrowedActorUserData();
    return false;
  }
  // The BotScene created the actor and will destroy it; this wrapper is only a view onto it.
  ArticulatedActorObject->bDestroyMochiActorOnObjectDestruction = false;

  // A borrowed component is a view onto a BotScene-owned bot. Its scene controllers are driven by
  // the BotScene bridge, so retaining the level-authored controller would register a second force
  // writer for the same articulation.
  SetController(nullptr);
  BotPtr = SceneBot;
  bIsBorrowedBot = true;
  BotScale = BorrowedScale;
  WarnOnUnmatchedLinkBones(ArticulatedActorObject->GetNestedLinkActorNames());

  OnBotSpawned.Broadcast(this);
  bArticulatedActorReady = true;

  // The scene owns this bot, so the BotScene visualization draws it from Mochi collision
  // geometry. Keep posing -- the retargeter chain depends on it -- but stop drawing, or the
  // Unreal mesh and the Mochi volumes render on top of each other.
  if (bHideMeshWhenBoundToSceneBot && IsVisible()) {
    SetVisibility(false, /*bPropagateToChildren=*/false);
    bHidMeshForSceneBot = true;
  }

  UE_LOG(
      LogMochiPhysics,
      Display,
      TEXT("[%hs] %s/%s now posing from scene bot '%hs'"),
      __FUNCTION__,
      *GetNameSafe(GetOwner()),
      *GetName(),
      SceneBot->GetName());
  return true;
}

void USuperDexBotComponent::RestoreBorrowedActorUserData() {
  if (BorrowedActorUserData.Num() == 0) {
    return;
  }
  const UGameInstance* GameInstance = GetWorld() ? GetWorld()->GetGameInstance() : nullptr;
  const UMochiGameInstanceSubsystem* Subsystem =
      GameInstance ? GameInstance->GetSubsystem<UMochiGameInstanceSubsystem>() : nullptr;
  UMochiSceneObject* SceneObject = Subsystem ? Subsystem->GetScene() : nullptr;
  if (SceneObject && SceneObject->IsValid()) {
    SceneObject->QueueOrExecuteCommand([this](mochi::Scene* Scene) {
      for (auto const& Saved : BorrowedActorUserData) {
        if (mochi::Actor* Actor = Scene->GetActor(Saved.Key)) {
          Actor->SetUserData(Saved.Value);
        }
      }
    });
    SceneObject->WaitForQueuedCommands();
  }
  BorrowedActorUserData.Reset();
}

void USuperDexBotComponent::DestroyBotAndArticulatedActor() {
  CurrentJointPositions.Reset();
  CurrentJointVelocities.Reset();
  if (ArticulatedActorObject) {
    ArticulatedActorObject->DestroyMochiResources();
    ArticulatedActorObject = nullptr;
    bArticulatedActorReady = false;
  }
  // A borrowed bot belongs to the BotScene, which destroys it on its own teardown.
  if (bIsBorrowedBot) {
    RestoreBorrowedActorUserData();
    BotPtr = nullptr;
    bIsBorrowedBot = false;
    if (bHidMeshForSceneBot) {
      SetVisibility(true, /*bPropagateToChildren=*/false);
      bHidMeshForSceneBot = false;
    }
    return;
  }
  if (BotPtr) {
    const UGameInstance* GameInstance = GetWorld()->GetGameInstance();
    const UMochiGameInstanceSubsystem* Subsys =
        GameInstance->GetSubsystem<UMochiGameInstanceSubsystem>();
    if (UMochiSceneObject* SceneObject = Subsys ? Subsys->GetScene() : nullptr) {
      SceneObject->QueueOrExecuteCommand([&](mochi::Scene* scene) {
        mochi::ErrorLog e{};
        superdex::robotics::DestroyBot(scene, BotPtr);
        BotPtr = nullptr;
      });
      SceneObject->WaitForQueuedCommands();
    }
  }
}

void USuperDexBotComponent::ApplyPrefabOverrides(superdex::robotics::BotPrefab& InOutPrefab) const {
  for (auto& link : InOutPrefab.links) {
    link.hasGravity = bHasGravity;
  }
  if (LayerOverride.IsEmpty()) {
    return;
  }
  // The layer override must reach every collidable the prefab spawns, not just the links:
  // the scene's contact matrix filters per layer, so a skin left on the layer baked into
  // the asset would keep colliding with whatever the overridden links were moved away from.
  const auto layer = mochi::DynamicString(TCHAR_TO_UTF8(*LayerOverride));
  for (auto& link : InOutPrefab.links) {
    link.layer = layer;
  }
  if (InOutPrefab.skin.has_value()) {
    InOutPrefab.skin->layer = layer;
  }
}

bool USuperDexBotComponent::OnArticulatedActorControl(
    FMochiArticulatedActorPoseVariant& OutPoseTarget,
    mochi::Actor* ArticulatedActor,
    mochi::Scene* Scene,
    double TimeStepSec) const {
  FScopeLock lock(&BotControllerMutex);
  if (!bArticulatedActorReady || !BotController) {
    return false;
  }
  auto state = ExtractArticulatedActorState(ArticulatedActor);
  return BotController->StepController(this, state, TimeStepSec, ArticulatedActor, Scene);
}

bool USuperDexBotComponent::EnsureArrayProperties() {
  bool ret = false;
  if (Bot) {
    auto& params = Bot->GetBotPrefab();
    if (InitialPose.Num() != params._numDofs) {
      InitialPose.SetNum(params._numDofs);
      for (int i = 0; i < params._numDofs; ++i) {
        InitialPose[i] = static_cast<double>(params.defaultPose[i]);
      }
      ret = true;
    }
    if (ContactGeneratingLinks.Num() == 0) {
      for (auto const& link : params.links) {
        ContactGeneratingLinks.Add(FName(UTF8_TO_TCHAR(link.name.c_str())));
      }
      ret = true;
    }
  }
  return ret;
}

void USuperDexBotComponent::PoseMeshFromJointPositions(const TArray<double>& InJointPositions) {
  if (!GetSkinnedAsset() || !RequiredBones.IsValid()) {
    return;
  }
  if (Bot) {
    ensure(InJointPositions.Num() == Bot->GetDefaultJointPositions().Num());
    TArray<FTransform> transforms;
    auto linkNames = Bot->GetLinkNames();
    if (Bot->ComputeLinkTransformsFromJointPositions(
            InJointPositions, transforms, EMochiArticulationLinkTransformSpace::LinkToParent)) {
      for (int iLink = 0; iLink < transforms.Num(); ++iLink) {
        const int32 boneIndex = GetBoneIndex(linkNames[iLink]);
        if (boneIndex >= 0 && boneIndex < BoneSpaceTransforms.Num()) {
          BoneSpaceTransforms[boneIndex] = transforms[iLink];
        }
      }
      MarkRefreshTransformDirty();
      TickPose(0, true);
      RefreshBoneTransforms();
      RecreateRenderState_Concurrent();
      UpdateOverlaps();
    }
  }
}

void USuperDexBotComponent::PoseMeshFromLinkWorldTransforms(
    const TArray<FTransform>& InLinkWorldTransforms) {
  if (!ArticulatedActorObject) {
    return;
  }
  const auto& LinkNames = ArticulatedActorObject->GetNestedLinkActorNames();
  check(LinkNames.Num() == InLinkWorldTransforms.Num());
  // Mochi link transforms carry no scale. Give the bones the bot's scale so the mesh renders at the
  // size of the simulated bot whatever the component's scale.
  const FVector Scale(BotScale);
  for (int iLink = 0; iLink < InLinkWorldTransforms.Num(); ++iLink) {
    FTransform LinkToWorld = InLinkWorldTransforms[iLink];
    LinkToWorld.SetScale3D(Scale);
    SetBoneTransformByName(LinkNames[iLink], LinkToWorld, EBoneSpaces::WorldSpace);
  }
  UpdateOverlaps();
}

void USuperDexBotComponent::WarnOnUnmatchedLinkBones(const TArray<FName>& LinkNames) const {
  const USkinnedAsset* Mesh = GetSkinnedAsset();
  if (!Mesh) {
    return;
  }
  TArray<FName> UnmatchedLinks;
  for (const FName& LinkName : LinkNames) {
    if (GetBoneIndex(LinkName) == INDEX_NONE) {
      UnmatchedLinks.Add(LinkName);
    }
  }
  if (UnmatchedLinks.Num() > 0) {
    FString UnmatchedList;
    for (int32 i = 0; i < UnmatchedLinks.Num(); ++i) {
      UnmatchedList += (i == 0 ? TEXT("") : TEXT(", "));
      UnmatchedList += UnmatchedLinks[i].ToString();
    }
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "[%hs] %d of %d bot links have no matching bone on skeletal mesh '%s' (on '%s'); these "
            "links cannot be posed because Mochi drives the mesh by bone name. An override mesh "
            "must name its bones after the bot's links (extra bones are fine). Unmatched: %s"),
        __FUNCTION__,
        UnmatchedLinks.Num(),
        LinkNames.Num(),
        *GetNameSafe(Mesh),
        *GetName(),
        *UnmatchedList);
  }
}

void USuperDexBotComponent::OnRegister() {
  Super::OnRegister();
  EnsureArrayProperties();
  if (Bot) {
    PoseMeshFromJointPositions(InitialPose);
  }
#if WITH_EDITOR
  if (GEditor) {
    if (UImportSubsystem* ImportSubsystem = GEditor->GetEditorSubsystem<UImportSubsystem>()) {
      ReimportDelegateHandle = ImportSubsystem->OnAssetReimport.AddUObject(
          this, &USuperDexBotComponent::OnAssetReimported);
    }
  }
#endif
}

void USuperDexBotComponent::OnUnregister() {
#if WITH_EDITOR
  if (ReimportDelegateHandle.IsValid() && GEditor) {
    if (UImportSubsystem* ImportSubsystem = GEditor->GetEditorSubsystem<UImportSubsystem>()) {
      ImportSubsystem->OnAssetReimport.Remove(ReimportDelegateHandle);
    }
    ReimportDelegateHandle.Reset();
  }
#endif
  Super::OnUnregister();
}

FBoxSphereBounds USuperDexBotComponent::CalcBounds(const FTransform& LocalToWorld) const {
  if (Bot) {
    FBoxSphereBounds localBounds;
    if (Bot->ComputeVisualBoundsFromLinkTransforms(
            BoneSpaceTransforms, EMochiArticulationLinkTransformSpace::LinkToParent, localBounds)) {
      return localBounds.TransformBy(LocalToWorld);
    }
  }
  return Super::CalcBounds(LocalToWorld);
}

#if WITH_EDITOR
void USuperDexBotComponent::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) {
  Super::PostEditChangeProperty(PropertyChangedEvent);
  const FName PropertyName = (PropertyChangedEvent.Property != nullptr)
      ? PropertyChangedEvent.Property->GetFName()
      : NAME_None;
  if (PropertyName == GET_MEMBER_NAME_CHECKED(USuperDexBotComponent, Bot)) {
    SetBotAsset(Bot);
  }
  if (PropertyName == GET_MEMBER_NAME_CHECKED(USuperDexBotComponent, InitialPose)) {
    PoseMeshFromJointPositions(InitialPose);
  }
  // Validate the (possibly newly-assigned) skeletal mesh against the bot's links so
  // a non-conforming override mesh is caught at edit time, before entering play.
  if (Bot) {
    WarnOnUnmatchedLinkBones(Bot->GetLinkNames());
  }
}

void USuperDexBotComponent::OnAssetReimported(UObject* ReimportedAsset) {
  if (ReimportedAsset == Bot) {
    // During reimport-in-place the USkeletalMesh* pointer is unchanged, so
    // SetSkinnedAssetAndUpdate early-returns without resizing BoneSpaceTransforms.
    // We only need to force a full reallocation (clear then re-set) when the bone
    // count actually changed -- that is the case that would otherwise assert in
    // FillComponentSpaceTransforms. When the bone count is unchanged (the common
    // case: refined collision/render geometry on the same skeleton) we leave the
    // component's current skeletal mesh alone, which also preserves any editor
    // override mesh.
    USkeletalMesh* ReimportedMesh = Bot->GetSkeletalMesh();
    const int32 NewBoneCount = ReimportedMesh ? ReimportedMesh->GetRefSkeleton().GetNum() : 0;
    if (NewBoneCount != BoneSpaceTransforms.Num()) {
      SetSkinnedAssetAndUpdate(nullptr);
      if (ReimportedMesh) {
        SetSkinnedAssetAndUpdate(ReimportedMesh);
      }
      EnsureArrayProperties();
      PoseMeshFromJointPositions(InitialPose);
      Modify(true);
    }
  }
}
#endif
