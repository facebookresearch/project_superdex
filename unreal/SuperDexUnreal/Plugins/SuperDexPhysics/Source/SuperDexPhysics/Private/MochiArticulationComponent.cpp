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

#include "MochiArticulationComponent.h"

#include "MochiConversions.h"
#include "MochiGameInstanceSubsystem.h"
#include "MochiPhysicsModule.h"
#include "MochiSoftMeshComponent.h"

#include <numeric>

void UMochiArticulationComponent::SetArticulationAsset(UMochiArticulation* InArticulation) {
  if (ArticulatedActorObject) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[%hs] Attempted to set MochiArticulation asset but %s has already spawned [%s]!"),
        __FUNCTION__,
        *GET_MEMBER_NAME_CHECKED(UMochiArticulationComponent, Articulation).ToString(),
        *GetName());
    return;
  }
  if (InArticulation) {
    Articulation = InArticulation;
    if (Articulation->GetSkeletalMesh()) {
      SetSkinnedAssetAndUpdate(Articulation->GetSkeletalMesh());
    }
    InitialJointPositions = Articulation->GetDefaultJointPositions();
    bFixedBase = Articulation->DefaultFixedBase();
    ContactGeneratingLinks.Reset();
    EnsureArrayProperties();
    PoseMeshFromJointPositions(InitialJointPositions);
  } else {
    Articulation = nullptr;
    SetSkinnedAssetAndUpdate(nullptr);
    InitialJointPositions.Reset();
    InitialJointVelocities.Reset();
    bFixedBase = false;
    ContactGeneratingLinks.Reset();
  }
}

UMochiArticulation* UMochiArticulationComponent::GetArticulationAsset() const {
  return Articulation;
}

bool UMochiArticulationComponent::SetInitialJointPositions(
    const TArray<double>& NewInitialJointPositions) {
  if (Articulation && NewInitialJointPositions.Num() == Articulation->GetNumDofs() &&
      ArticulatedActorObject == nullptr) {
    InitialJointPositions = NewInitialJointPositions;
    return true;
  }
  return false;
}

const TArray<double>& UMochiArticulationComponent::GetInitialJointPositions() const {
  return InitialJointPositions;
}

bool UMochiArticulationComponent::IsFixedBase() const {
  return bFixedBase;
}

bool UMochiArticulationComponent::GetMochiGravityEnabled() const {
  return bGravityEnabled;
}

const UMochiSoftMeshComponent* UMochiArticulationComponent::GetSoftLink(int SoftLinIndex) {
  if (SoftLinIndex > 0 && SoftLinIndex < SoftLinkDynamicMeshComponents.Num()) {
    return SoftLinkDynamicMeshComponents[SoftLinIndex];
  }
  return nullptr;
}

void UMochiArticulationComponent::SetArticulationWorldTransform(const FTransform& WorldTransform) {
  if (bFixedBase) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[%hs] Attempted to set world transform of articulation [%s] while %s is set!"),
        __FUNCTION__,
        *GetName(),
        *GET_MEMBER_NAME_CHECKED(UMochiArticulationComponent, bFixedBase).ToString());
    return;
  }
  if (ArticulatedActorObject) {
    ArticulatedActorObject->SetRootTransform(WorldTransform);
  }
}

void UMochiArticulationComponent::SetArticulatedPoseFromJoints(
    TArray<double> const& JointPositions) {
  if (ArticulatedActorObject) {
    ArticulatedActorObject->SetArticulatedPoseFromJoints(JointPositions);
  }
}

UMochiArticulatedActorObject* UMochiArticulationComponent::GetMochiArticulatedActor() const {
  return ArticulatedActorObject;
}

FTransform UMochiArticulationComponent::GetLinkToWorldTransform(FName LinkName) const {
  if (Articulation) {
    const int iLink = Articulation->GetLinkIndex(LinkName);
    if (iLink != INDEX_NONE && iLink < QueriedLinkTransforms.Num()) {
      return QueriedLinkTransforms[iLink];
    }
  }
  return FTransform();
}

UMochiArticulationComponent::UMochiArticulationComponent() {
  PrimaryComponentTick.bCanEverTick = true;
  PrimaryComponentTick.TickGroup = TG_StartPhysics;
  EnsureArrayProperties();
}

void UMochiArticulationComponent::BeginPlay() {
  Super::BeginPlay();
  if (Articulation && Articulation->IsReady()) {
    NumArticulationDofs = Articulation->GetNumDofs();
    // MochiArticulationController setup
    if (ArticulationController == nullptr) {
      SetArticulationController(ArticulationControllerClass);
    }
    if (ArticulationController) {
      OnArticulationSpawned.AddDynamic(
          this, &UMochiArticulationComponent::HandleArticulationControllerOnArticulationSpawned);
    }
  } else {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%hs] %s was null or not ready!"),
        __FUNCTION__,
        *GET_MEMBER_NAME_CHECKED(UMochiArticulationComponent, Articulation).ToString());
    Articulation = nullptr;
  }
}

void UMochiArticulationComponent::TickComponent(
    float DeltaTime,
    ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction) {
  if (GetWorld()->IsGameWorld()) {
    if (ArticulatedActorObject == nullptr) {
      // NB: We defer actor creation to the first tick so that users can e.g.
      // script starting pose via SetInitialJointPositions in Blueprint BeginPlay
      CreateArticulatedActor();
    }
    if (ArticulatedActorObject) {
      ArticulatedActorObject->TryGetQueryData(
          [this](const FMochiArticulatedActorQueryData& QueryData) {
            QueriedLinkTransforms = QueryData.LinksToWorld;
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
    // MochiArticulationController tick
    if (ArticulatedActorObject && ArticulationController) {
      FScopeLock lock(&ArticulationControllerMutex);
      ArticulationController->TickController(this, DeltaTime);
    }
  }
  // NB: Call UPoseableMeshComponent::TickComponent last to avoid one-frame render delay
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
}

void UMochiArticulationComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  MOCHI_PROFILE_SCOPE();
  SetArticulationController(nullptr);
  Super::EndPlay(EndPlayReason);
  for (const auto& dynMesh : SoftLinkDynamicMeshComponents) {
    dynMesh->GetMochiActor()->DestroyMochiResources();
  }
  DestroyArticulatedActor();
}

void UMochiArticulationComponent::OnRegister() {
  Super::OnRegister();
  EnsureArrayProperties();
  if (Articulation) {
    PoseMeshFromJointPositions(InitialJointPositions);
  }
}

FBoxSphereBounds UMochiArticulationComponent::CalcBounds(const FTransform& LocalToWorld) const {
  if (Articulation) {
    FBoxSphereBounds localBounds;
    if (Articulation->ComputeVisualBoundsFromLinkTransforms(
            BoneSpaceTransforms, EMochiArticulationLinkTransformSpace::LinkToParent, localBounds)) {
      return localBounds.TransformBy(LocalToWorld);
    }
  }
  return Super::CalcBounds(LocalToWorld);
}

bool UMochiArticulationComponent::WantsPoseController() const {
  return false;
}

FMochiPoseControllerGains UMochiArticulationComponent::GetPoseControllerGains() const {
  // TODO: Remove GetPoseControllerGains once its last legacy caller has been deleted
  return FMochiPoseControllerGains{};
}

void UMochiArticulationComponent::CreateArticulatedActor() {
  MOCHI_PROFILE_SCOPE();
  if (!GetOwner()->HasAuthority()) {
    return;
  }
  DestroyArticulatedActor();
  if (Articulation == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%hs] Failed to create articulated actor because %s was null!"),
        __FUNCTION__,
        *GET_MEMBER_NAME_CHECKED(UMochiArticulationComponent, Articulation).ToString());
    return;
  }
  TArray<FTransform> rootFromLinks;
  if (!Articulation->ComputeLinkTransformsFromJointPositions(
          InitialJointPositions, rootFromLinks, EMochiArticulationLinkTransformSpace::LinkToRoot)) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[%hs] Failed to create articulated actor because %s could not be converted to link transforms!"),
        __FUNCTION__,
        *GET_MEMBER_NAME_CHECKED(UMochiArticulationComponent, InitialJointPositions).ToString());
    return;
  }
  ArticulatedActorObject = UMochiArticulatedActorObject::CreateArticulation(
      this,
      GetOwner()->GetName() + "___" + GetName(),
      Articulation,
      GetComponentTransform(),
      rootFromLinks,
      ContactParams,
      Layer,
      bFixedBase,
      bGravityEnabled,
      InitialJointVelocities,
      [this](
          const TArray<mochi::TransformRT>& WorldFromLinks,
          const TArray<mochi::real>& JointPositions,
          mochi::Actor* ArticulatedActor,
          mochi::Scene* Scene) {
        OnArticulatedActorGetPose(WorldFromLinks, JointPositions, ArticulatedActor, Scene);
      },
      WantsPoseController(),
      GetPoseControllerGains(),
      [this](
          FMochiArticulatedActorPoseVariant& WorldFromLinks,
          mochi::Actor* ArticulatedActor,
          mochi::Scene* Scene,
          double /*TimeStepSec*/) {
        return OnArticulatedActorControl(WorldFromLinks, ArticulatedActor, Scene);
      },
      bEnableContacts ? ContactGeneratingLinks : TArray<FName>{},
      /* bSkinQueriesContactPoints */ false,
      nullptr);
  if (ArticulatedActorObject && ArticulatedActorObject->IsValid()) {
    // We will have issues if the number of DOFs reported by Mochi ever mismatches
    // that reported by the MochiArticulation; throw errors when they are not equal or when we
    // haven't accounted for known discrepancies.
    NumMochiDofs = ArticulatedActorObject->GetNumDofs();
    NumMochiBaseDofs =
        Articulation->GetEffectiveRootJointType(bFixedBase) == EMochiArticulatedJointType::Free
        ? GetArticulatedJointNumDofs(EMochiArticulatedJointType::Free)
        : 0;
    NumMochiJointDofs = NumMochiDofs - NumMochiBaseDofs;
    if (NumMochiJointDofs != NumArticulationDofs) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT(
              "[%hs] %s (%d) on articulated actor created by %s::%s "
              "not equal to %s (%d) from articulation %s!"),
          __FUNCTION__,
          *GET_MEMBER_NAME_CHECKED(UMochiArticulationComponent, NumMochiJointDofs).ToString(),
          NumMochiJointDofs,
          *GetOwner()->GetName(),
          *GetName(),
          *GET_MEMBER_NAME_CHECKED(UMochiArticulationComponent, NumArticulationDofs).ToString(),
          NumArticulationDofs,
          *Articulation->GetName());
    }
    // robot spawn event
    OnArticulationSpawned.Broadcast(this);
    // create wrapper dynamic mesh component around nested soft actors
    CreateSoftLinksDynamicMeshComponents(ArticulatedActorObject->GetNestedSoftActorHandles());
    // articulated actor callbacks can proceed now
    bArticulatedActorReady = true;
  }
}

void UMochiArticulationComponent::DestroyArticulatedActor() {
  if (ArticulatedActorObject) {
    ArticulatedActorObject->DestroyMochiResources();
    ArticulatedActorObject = nullptr;
  }
}

void UMochiArticulationComponent::OnArticulatedActorGetPose(
    const TArray<mochi::TransformRT>& InWorldFromLinks,
    const TArray<mochi::real>& InJointDofs,
    mochi::Actor* ArticulatedActor,
    mochi::Scene* Scene) {}

bool UMochiArticulationComponent::OnArticulatedActorControl(
    FMochiArticulatedActorPoseVariant& OutPoseTarget,
    mochi::Actor* ArticulatedActor,
    mochi::Scene* Scene) {
  if (!bArticulatedActorReady || !ArticulationController) {
    return false;
  }
  FScopeLock lock(&ArticulationControllerMutex);
  mochi::ErrorLog e{};
  auto state = ExtractArticulatedActorState(ArticulatedActor);
  ConvertArticulatedActorDofsToUnreal(ArticulatedActor, state.JointPositions);
  ConvertArticulatedActorDofsToUnreal(ArticulatedActor, state.JointVelocities);
  thread_local FMochiArticulationControllerCommands commands;
  commands.BaseTransformTarget = FTransform::Identity;
  commands.JointPositionTargets.SetNumZeroed(NumMochiJointDofs);
  commands.JointEfforts.SetNumZeroed(NumMochiJointDofs);
  const double mochiDeltaTime = Scene->GetLastTimeStep();
  if (ArticulationController->Control(this, state, mochiDeltaTime, commands, Scene)) {
    ensure(commands.JointPositionTargets.Num() == NumMochiJointDofs);
    ensure(commands.JointEfforts.Num() == NumMochiJointDofs);
    ConvertArticulatedActorDofsToMochi(ArticulatedActor, commands.JointPositionTargets);
    ConvertArticulatedActorDofsToMochi(ArticulatedActor, commands.JointEfforts);
    auto poseTarget = BuildPoseControllerTargets(
        ArticulatedActor, commands.BaseTransformTarget, commands.JointPositionTargets);
    OutPoseTarget.Set<TArray<mochi::real>>(poseTarget);
    SetArticulatedActorJointEfforts(ArticulatedActor, commands.JointEfforts);
    return true;
  }
  return false;
}

bool UMochiArticulationComponent::SetArticulationController(
    TSubclassOf<UMochiArticulationController> NewControllerClass) {
  FScopeLock lock(&ArticulationControllerMutex);
  if (ArticulationController != nullptr) {
    ArticulationController->Shutdown(this);
    ArticulationController = nullptr;
  }
  ArticulationControllerClass = NewControllerClass;
  if (ArticulationControllerClass) {
    ArticulationController =
        NewObject<UMochiArticulationController>(this, ArticulationControllerClass);
    if (ArticulationController) {
      if (!ArticulationController->Initialize(this)) {
        UE_LOG(
            LogMochiPhysics,
            Warning,
            TEXT(
                "[UMochiArticulationComponent] Failed to initialize MochiArticulationController of class %s"),
            *ArticulationControllerClass->GetName());
        ArticulationController = nullptr;
        return false;
      }
      if (ArticulatedActorObject) {
        ArticulatedActorObject->AddPoseController(GetPoseControllerGains(), false);
      }
    } else {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT(
              "[UMochiArticulationComponent] Failed to create MochiArticulationController of class %s"),
          *ArticulationControllerClass->GetName());
      return false;
    }
  } else {
    if (ArticulatedActorObject && ArticulatedActorObject->HasPoseController()) {
      ArticulatedActorObject->RemovePoseController(false);
    }
  }
  return true;
}

UMochiArticulationController* UMochiArticulationComponent::GetArticulationController() const {
  return ArticulationController;
}

void UMochiArticulationComponent::HandleArticulationControllerOnArticulationSpawned(
    UMochiArticulationComponent* InArticulation) {
  if (ArticulationController) {
    ArticulationController->OnArticulationSpawned(InArticulation);
  }
}

bool UMochiArticulationComponent::EnsureArrayProperties() {
  bool ret = false;
  if (Articulation) {
    if (InitialJointPositions.Num() != Articulation->GetDefaultJointPositions().Num()) {
      InitialJointPositions = Articulation->GetDefaultJointPositions();
      ret = true;
    }
    if (ContactGeneratingLinks.Num() == 0) {
      ContactGeneratingLinks = Articulation->GetLinkNames();
      for (auto& softLink : Articulation->GetSoftLinks()) {
        ContactGeneratingLinks.Add(softLink.Name);
      }
      ret = true;
    }
  }
  return ret;
}

void UMochiArticulationComponent::PoseMeshFromJointPositions(
    const TArray<double>& InJointPositions) {
  if (!GetSkinnedAsset() || !RequiredBones.IsValid()) {
    return;
  }
  if (Articulation) {
    ensure(InJointPositions.Num() == Articulation->GetDefaultJointPositions().Num());
    TArray<FTransform> transforms;
    if (Articulation->ComputeLinkTransformsFromJointPositions(
            InJointPositions, transforms, EMochiArticulationLinkTransformSpace::LinkToParent)) {
      for (int iLink = 0; iLink < transforms.Num(); ++iLink) {
        const int32 boneIndex = GetBoneIndex(Articulation->GetLinks()[iLink].Name);
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

void UMochiArticulationComponent::PoseMeshFromLinkWorldTransforms(
    const TArray<FTransform>& InLinkWorldTransforms) {
  auto& links = Articulation->GetLinks();
  check(links.Num() == InLinkWorldTransforms.Num());
  for (int iLink = 0; iLink < InLinkWorldTransforms.Num(); iLink++) {
    SetBoneTransformByName(
        links[iLink].Name, InLinkWorldTransforms[iLink], EBoneSpaces::WorldSpace);
  }
  UpdateOverlaps();
}

void UMochiArticulationComponent::CreateSoftLinksDynamicMeshComponents(
    const TArray<mochi::ActorHandle>& SoftActors) {
  const auto& softLinks = Articulation->GetSoftLinks();
  const int numSoftLinks = softLinks.Num();
  if (numSoftLinks == 0) {
    return;
  }
  check(SoftActors.Num() == numSoftLinks);
  AActor* owner = GetOwner();
  SoftLinkDynamicMeshComponents.Reset();
  for (int iLink = 0; iLink < numSoftLinks; ++iLink) {
    auto dynMesh = Cast<UMochiSoftMeshComponent>(owner->AddComponentByClass(
        UMochiSoftMeshComponent::StaticClass(), true, FTransform::Identity, true));
    dynMesh->SetupAttachment(this);
    FString name = FString::Printf(
        TEXT("%s_MochiSoftMeshComponent%d_%s"),
        *GetName(),
        iLink,
        *softLinks[iLink].Name.ToString());
    dynMesh->Rename(*name);
    dynMesh->SetExistingMochiActorHandle(SoftActors[iLink]);
    dynMesh->SetMochiModel(softLinks[iLink].Model);
    dynMesh->SetSimulatePhysics(false);
    dynMesh->SetVisibility(true);
    dynMesh->SetMaterial(0, softLinks[0].Model->PreviewRenderMaterial);
    owner->FinishAddComponent(dynMesh, true, FTransform::Identity);
    dynMesh->SetCollisionProfileName(UCollisionProfile::BlockAllDynamic_ProfileName);
    dynMesh->SetGenerateOverlapEvents(true);
    dynMesh->GetMochiActor()->bDestroyMochiActorOnObjectDestruction = false;
    SoftLinkDynamicMeshComponents.Add(dynMesh);
  }
  OnSoftLinksSpawned.Broadcast(this, SoftLinkDynamicMeshComponents);
}

#if WITH_EDITOR
void UMochiArticulationComponent::PostEditChangeProperty(
    FPropertyChangedEvent& PropertyChangedEvent) {
  Super::PostEditChangeProperty(PropertyChangedEvent);
  const FName PropertyName = (PropertyChangedEvent.Property != nullptr)
      ? PropertyChangedEvent.Property->GetFName()
      : NAME_None;
  if (PropertyName == GET_MEMBER_NAME_CHECKED(UMochiArticulationComponent, Articulation)) {
    SetArticulationAsset(Articulation);
  }
  if (PropertyName == GET_MEMBER_NAME_CHECKED(UMochiArticulationComponent, InitialJointPositions)) {
    PoseMeshFromJointPositions(InitialJointPositions);
  }
}

void UMochiArticulationComponent::PostLoad() {
  Super::PostLoad();
  if (EnsureArrayProperties()) {
    Modify(true);
  }
}
#endif
