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

#include "SuperDexPoseableComponent.h"

#include "Skeletons/SuperDexSkeleton_OpenXR.h"
#include "SuperDexPlayerController.h"
#include "SuperDexPoseFunctionLibrary.h"
#include "SuperDexPoseModule.h"
#include "SuperDexPoseSource.h"
#include "SuperDexPoseSubsystem.h"
#include "SuperDexPoseTypes.h"
#include "SuperDexPoseableActor.h"

#include "Engine/SkeletalMesh.h"
#include "MochiArticulationBase.h"
#include "MochiTypes.h"
#include "Net/UnrealNetwork.h"

// #pragma optimize("", off)

/**
 * A file thread_local global map shared by all USuperDexPoseableComponents to detect infinite loops
 * in the GetPose chain formed by these components. Each component adds its this pointer to the map
 * when it calls GetInputPoseFromSource setting the value to true, and sets the value to false
 * when returning from that function. If the value is true when entering, that means we are
 * currently scoped inside the same component twice and have formed a loop. We should return false
 * to break the infinite loop. The component will remove itself from the map when it is destroyed.
 * However, this can still leave orphaned keys in other thread's maps. We do not expect the number
 * of created components to be large enough to make lookup times meaningful.
 *
 * This is a global because the compiler does not allow us to use thread_local in a class member,
 * since the class is exposed via dll export. The other option is to create a map of threadIDs that
 * is mutex protected, but that will result in taking 2 mutex locks per call into GetPose. This
 * variable should not be exposed outside of this file.
 */
namespace {
thread_local TMap<USuperDexPoseableComponent*, bool> GLoopDetectionMap;
}

USuperDexPoseableComponent::USuperDexPoseableComponent() {
  PrimaryComponentTick.bCanEverTick = true;
  PrimaryComponentTick.bStartWithTickEnabled = true;
  bTickInEditor = true;

  // Tick before physics but after components in group TG_PrePhysics (e.g. MochiHandComponent)
  PrimaryComponentTick.TickGroup = TG_StartPhysics;
  bAutoActivate = true;
  bWantsInitializeComponent = true;

  // By default, we probably don't want extra gravity on our hands
  SetEnableGravity(false);
}

USuperDexPoseableComponent::~USuperDexPoseableComponent() {
  GLoopDetectionMap.Remove(this);
}

FSuperDexPoseCacheWriteScope::FSuperDexPoseCacheWriteScope(
    const USuperDexPoseableComponent& InComponent)
    : Component(InComponent) {
  Component.CacheLock.WriteLock();
  // Only the writing thread ever finds its own id here, so relaxed ordering suffices.
  Component.CacheWriterThreadId.store(
      FPlatformTLS::GetCurrentThreadId(), std::memory_order_relaxed);
}

FSuperDexPoseCacheWriteScope::~FSuperDexPoseCacheWriteScope() {
  Component.CacheWriterThreadId.store(0, std::memory_order_relaxed);
  Component.CacheLock.WriteUnlock();
}

bool USuperDexPoseableComponent::IsCacheWriterThread() const {
  return CacheWriterThreadId.load(std::memory_order_relaxed) == FPlatformTLS::GetCurrentThreadId();
}

bool USuperDexPoseableComponent::GetCachedPose(FSuperDexPose& OutPose) const {
  if (IsCacheWriterThread()) {
    // This thread is writing the cache and has it to itself.
    if (bCacheInitialized) {
      OutPose = CachedOutputPose;
    }
    return bCacheInitialized;
  }
  FReadScopeLock Lock(CacheLock);
  if (bCacheInitialized) {
    OutPose = CachedOutputPose;
  }
  return bCacheInitialized;
}

FTransform USuperDexPoseableComponent::GetVrRootToWorld() const {
  if (IsCacheWriterThread()) {
    return CachedVrRootToWorld;
  }
  FReadScopeLock Lock(CacheLock);
  return CachedVrRootToWorld;
}

bool USuperDexPoseableComponent::GetPose(FSuperDexPose& OutPose) {
  if (bEnableGetPose == false) {
    return false;
  }

  // don't update pose unless we're in a game world
  check(IsValid(GetWorld()));
  if (!GetWorld()->IsGameWorld()) {
    return false;
  }

  bool bHavePose = GetInputPoseFromSource(OutPose);
  if (bHavePose && bOverrideOutputPoseScale) {
    OutPose.SkeletonRootToVrRoot.SetScale3D(FVector(OutputPoseScaleOverride));
  }
  return bHavePose;
}

USuperDexPoseableComponent* USuperDexPoseableComponent::GetChildSourcePoseable() const {
  // Use a copy of the AttachedChildren so that it cannot change from under us.
  // This function can be called from the mochi thread as part of GetPose().
  // Unfortunately there is no way to guard the internal AttachedChildren array so that it is always
  // safe to access without changing UE internals. If an array iterator bug hits again here we might
  // need to do something like keep a cache of children that is updated from the UE thread and mutex
  // protected to make it truely safe.
  const TArray<USceneComponent*> AttachedChildren = GetAttachChildren();
  for (USceneComponent* Child : AttachedChildren) {
    USuperDexPoseableComponent* ChildSource = Cast<USuperDexPoseableComponent>(Child);
    if (IsValid(ChildSource)) {
      if (!ChildSourcePoseableExcludeList.Contains(ChildSource)) {
        if (ChildSource->GetName().Contains(ChildSourceNameContainsSubstring)) {
          return ChildSource;
        }
      }
    }
  }
  return nullptr;
}

FSuperDexPoseSourcePtrTypePair USuperDexPoseableComponent::GetSourcePoseable() const {
  // if pose source override is set, directly call pose subsystem with that
  // if pose source override is not valid, return an invalid source
  if (PoseSourceOverride != nullptr) {
    if (IsValid(PoseSourceOverride)) {
      USuperDexPoseSubsystem* Hpss = GetWorld()->GetSubsystem<USuperDexPoseSubsystem>();
      return FSuperDexPoseSourcePtrTypePair{ESuperDexPoseSourceType::SuperDexPoseSubsystem, Hpss};
    } else {
      return FSuperDexPoseSourcePtrTypePair{ESuperDexPoseSourceType::Invalid, nullptr};
    }
  }

  // if source poseable override is set, use it
  // if the source poseable is not valid, return an invalid source
  if (SourcePoseableOverride != nullptr) {
    if (IsValid(SourcePoseableOverride)) {
      return FSuperDexPoseSourcePtrTypePair{
          ESuperDexPoseSourceType::SuperDexPoseableComponent, SourcePoseableOverride};
    } else {
      return FSuperDexPoseSourcePtrTypePair{ESuperDexPoseSourceType::Invalid, nullptr};
    }
  }

  // if child component is a SuperDex poseable component, use what it says
  // if there is more than one child, use the one defined by SourceChildNameContainsSubstring
  USuperDexPoseableComponent* ChildSource = GetChildSourcePoseable();
  if (ChildSource != nullptr) {
    return FSuperDexPoseSourcePtrTypePair{
        ESuperDexPoseSourceType::SuperDexPoseableComponent, ChildSource};
  }

  check(IsValid(GetOwner()));

  // if attached to a SuperDex poseable actor, use what it says
  if (GetOwner()->IsA<ASuperDexPoseableActor>()) {
    return FSuperDexPoseSourcePtrTypePair{
        ESuperDexPoseSourceType::SuperDexPoseableActor, GetOwner()};
  }

  // if owning actor is possessed by a SuperDex player controller, use what it says
  if (GetOwner()->IsA<APawn>()) {
    const APawn* OwnerPawn = static_cast<APawn*>(GetOwner());

    if (OwnerPawn->GetController()->IsA<ASuperDexPlayerController>()) {
      return FSuperDexPoseSourcePtrTypePair{
          ESuperDexPoseSourceType::SuperDexPlayerController, OwnerPawn->GetController()};
    }
  }

  // There is no pose source, just return false as though disabled
  return FSuperDexPoseSourcePtrTypePair{ESuperDexPoseSourceType::Invalid, nullptr};
}

bool USuperDexPoseableComponent::CheckIfInInfiniteLoop(FSuperDexPose& Pose) {
  bool& bWasGetPoseCalledAlready = GLoopDetectionMap.FindOrAdd(this);
  if (bWasGetPoseCalledAlready) {
    UE_LOG(
        LogSuperDexPose,
        Error,
        TEXT(
            "Infinite loop detected for poseID [%lu]!"
            " Component [%30s] on actor [%s] is already in the pose source chain!"
            " This is the end of the loop, returning false and unwinding."),
        FPlatformTLS::GetCurrentThreadId(),
        *GetName(),
        *GetOwner()->GetName());
    Pose.bIsInInfiniteLoop = true;
    ensure(false);
    // wasGetPoseCalledAlready will be set to false up the stack when exiting the first entry point
    return true;
  }
  return false;
}

bool USuperDexPoseableComponent::GetInputPoseFromSource(FSuperDexPose& InputPose) {
  if (bEnableGetPose == false) {
    return false;
  }

  // don't update pose unless we're in a game world
  check(IsValid(GetWorld()));
  if (!GetWorld()->IsGameWorld()) {
    return false;
  }

  FSuperDexPoseSourcePtrTypePair Source = GetSourcePoseable();
  if (Source.Source == nullptr) {
    return false;
  }

  // Check if we are in an infinite loop.
  InputPose.bIsInInfiniteLoop = false;
  bool bInfiniteLoopDetected = CheckIfInInfiniteLoop(InputPose);
  if (bInfiniteLoopDetected) {
    return false;
  }
  // If not in infinite loop, set this component's flag to true so we can detect one later
  GLoopDetectionMap.FindOrAdd(this) = true;

  bool bHavePose = false;
  switch (Source.Type) {
    case ESuperDexPoseSourceType::SuperDexPoseSubsystem:
      bHavePose = static_cast<USuperDexPoseSubsystem*>(Source.Source)
                      ->GetPose(PoseSourceOverride, Handedness, InputPose);
      break;
    case ESuperDexPoseSourceType::SuperDexPoseableComponent:
      bHavePose = static_cast<USuperDexPoseableComponent*>(Source.Source)->GetPose(InputPose);
      break;
    case ESuperDexPoseSourceType::SuperDexPoseableActor:
      bHavePose = static_cast<ASuperDexPoseableActor*>(Source.Source)->GetPose(InputPose);
      break;
    case ESuperDexPoseSourceType::SuperDexPlayerController:
      bHavePose =
          static_cast<ASuperDexPlayerController*>(Source.Source)->GetPose(Handedness, InputPose);
      break;
    default:
      bHavePose = false; // return false later so we can clean up elegantly
  }
  if (bHavePose && bOverrideInputPoseScale) {
    InputPose.SkeletonRootToVrRoot.SetScale3D(FVector(InputPoseScaleOverride));
  }

  if (InputPose.bIsInInfiniteLoop == true) {
    UE_LOG(
        LogSuperDexPose,
        Error,
        TEXT(
            "Infinite loop detected for poseID [%llu]!"
            " Component [%30s] on actor [%s] got pose from above."),
        FPlatformTLS::GetCurrentThreadId(),
        *GetName(),
        *GetOwner()->GetName());
  }
  // about to exit the function, set flag back to false
  GLoopDetectionMap.FindChecked(this) = false;
  return bHavePose;
}

FTransform USuperDexPoseableComponent::GetBoneToRestPoseRootTransform(int BoneIndex) const {
  if (!InitBoneToRestPoseRootTransforms()) {
    return FTransform::Identity;
  }

  if (BoneIndex >= BoneToRestPoseRootTransforms.Num()) {
    UE_LOG(
        LogSuperDexPose, Warning, TEXT("[%s] Provided invalid bone!"), ANSI_TO_TCHAR(__FUNCTION__));
    return FTransform::Identity;
  }

  return BoneToRestPoseRootTransforms[BoneIndex];
}

FVector USuperDexPoseableComponent::TransformRestPoseBoneLocationToRoot(
    int BoneIndex,
    const FVector& BoneLocalLocation) const {
  if (!InitBoneToRestPoseRootTransforms()) {
    return FVector::ZeroVector;
  }

  if (BoneIndex >= BoneToRestPoseRootTransforms.Num()) {
    UE_LOG(
        LogSuperDexPose, Warning, TEXT("[%s] Provided invalid bone!"), ANSI_TO_TCHAR(__FUNCTION__));
    return FVector::ZeroVector;
  }

  const FTransform& BoneToRestPoseHandRoot = BoneToRestPoseRootTransforms[BoneIndex];
  return BoneToRestPoseHandRoot.TransformPosition(BoneLocalLocation);
}

bool USuperDexPoseableComponent::GetCachedPoseLinkTransform(
    FName LinkName,
    FTransform& OutLinkToWorld) const {
  FSuperDexPose Pose;
  FTransform LinkToSkeletonRoot;
  if (!GetCachedPose(Pose) ||
      !USuperDexPoseFunctionLibrary::ComputeSuperDexPoseLinkToSkeletonRoot(
          Pose,
          USuperDexPoseFunctionLibrary::FindSuperDexPoseLinkIndex(Pose, LinkName),
          LinkToSkeletonRoot)) {
    return false;
  }
  OutLinkToWorld = LinkToSkeletonRoot * GetComponentTransform();
  return true;
}

FTransform USuperDexPoseableComponent::GetFingerTipTransform(ESuperDexFinger Finger) const {
  const ESuperDexBone_OXR TipBone = GetFingerTipBone_OXR(Finger);
  if (TipBone == ESuperDexBone_OXR::Invalid) {
    return GetComponentTransform();
  }
  const FName TipName = SuperDexBoneNames_OXR[static_cast<int32>(TipBone)];
  FTransform TipToWorld;
  if (GetCachedPoseLinkTransform(TipName, TipToWorld)) {
    return TipToWorld;
  }
  return GetSocketTransform(TipName);
}

const TArray<int32>& USuperDexPoseableComponent::GetMeshBonesOfLinks(
    const UMochiArticulationBase& Bot) {
  const USkinnedAsset* Mesh = GetSkinnedAsset();
  if (MeshBonesOfLinksBot.Get() == &Bot && MeshBonesOfLinksMesh.Get() == Mesh) {
    return MeshBonesOfLinks;
  }
  MeshBonesOfLinksBot = &Bot;
  MeshBonesOfLinksMesh = Mesh;
  const FReferenceSkeleton& RefSkeleton = Mesh->GetRefSkeleton();
  const TArray<FName> LinkNames = Bot.GetLinkNames();
  MeshBonesOfLinks.Init(INDEX_NONE, LinkNames.Num());
  bool bAnyLinkHasBone = false;
  for (int32 Link = 0; Link < LinkNames.Num(); ++Link) {
    MeshBonesOfLinks[Link] = RefSkeleton.FindBoneIndex(LinkNames[Link]);
    bAnyLinkHasBone |= MeshBonesOfLinks[Link] != INDEX_NONE;
  }
  if (!bAnyLinkHasBone) {
    if (LinkNames.Num() == RefSkeleton.GetNum()) {
      // A mesh of the bot's skeleton with other bone names, such as a Nimble hand mesh showing
      // OculusXR poses: its bones are the links in order.
      for (int32 Link = 0; Link < LinkNames.Num(); ++Link) {
        MeshBonesOfLinks[Link] = Link;
      }
    } else {
      UE_LOG(
          LogSuperDexPose,
          Warning,
          TEXT(
              "Component [%s]: no bone of mesh [%s] is a link of bot [%s], and they have %d bones"
              " and %d links. The mesh does not show its poses."),
          *GetName(),
          *Mesh->GetName(),
          *Bot.GetName(),
          RefSkeleton.GetNum(),
          LinkNames.Num());
    }
  }
  return MeshBonesOfLinks;
}

void USuperDexPoseableComponent::ApplyCachedPoseToMesh() {
  if (!bCacheInitialized || !GetSkinnedAsset()) {
    return;
  }

  UMochiArticulationBase* Bot = nullptr;
  TArray<FTransform> LinkToParent;
  {
    FReadScopeLock Lock(CacheLock);
    if (!USuperDexPoseFunctionLibrary::ComputeSuperDexPoseLinkToParentTransforms(
            CachedOutputPose, LinkToParent)) {
      return;
    }
    Bot = CachedOutputPose.Bot;
  }

  const TArray<int32>& MeshBones = GetMeshBonesOfLinks(*Bot);
  const bool bSetLocations = Bot->GetSkeletalMesh() == GetSkinnedAsset();
  for (int32 Link = 0; Link < LinkToParent.Num() && Link < MeshBones.Num(); ++Link) {
    const int32 Bone = MeshBones[Link];
    if (!BoneSpaceTransforms.IsValidIndex(Bone)) {
      continue;
    }
    const FTransform& LinkTransform = LinkToParent[Link];
    if (!LinkTransform.GetRotation().ContainsNaN()) {
      BoneSpaceTransforms[Bone].SetRotation(LinkTransform.GetRotation());
    }
    if (bSetLocations && !LinkTransform.GetLocation().ContainsNaN()) {
      BoneSpaceTransforms[Bone].SetLocation(LinkTransform.GetLocation());
    }
  }

  MarkRefreshTransformDirty();
}

void USuperDexPoseableComponent::OnRegister() {
  Super::OnRegister();
  WhenNotPlayingInitPoseCache();
}

#if WITH_EDITOR
void USuperDexPoseableComponent::PostEditChangeProperty(
    FPropertyChangedEvent& PropertyChangedEvent) {
  Super::PostEditChangeProperty(PropertyChangedEvent);
  const FName PropName = PropertyChangedEvent.GetMemberPropertyName();
  if (PropName == GET_MEMBER_NAME_CHECKED(USuperDexPoseableComponent, EditorViewportPose)) {
    WhenNotPlayingInitPoseCache();
  }
}
#endif

void USuperDexPoseableComponent::WhenNotPlayingInitPoseCache() {
  if (GWorld->IsGameWorld()) {
    return;
  }
  USkinnedAsset* SkelMesh = GetSkinnedAsset();
  if (!SkelMesh) {
    return;
  }
  if (GetOwner() == nullptr) {
    return;
  }

  if (EditorViewportPose != nullptr) {
    if (USuperDexPoseFunctionLibrary::IsSuperDexPoseOnBot(EditorViewportPose->Pose)) {
      // Write the pose to cache and set initialized true, then apply to mesh for editor preview.
      FWriteScopeLock Lock(CacheLock);
      CachedOutputPose = EditorViewportPose->Pose;
      bCacheInitialized = true;
    } else {
      UE_LOG(
          LogSuperDexPose,
          Error,
          TEXT(
              "[%hs::%d] EditorViewportPose has no joint positions of a bot, on component [%s] on actor [%s]."
              " The EditorViewportPose will not be used."),
          __FUNCTION__,
          __LINE__,
          *GetName(),
          *GetOwner()->GetName());
    }
  } else {
    // EditorViewportPose == nullptr: the mesh shows its rest pose. The cached pose has no bot, so
    // it only holds the pose metadata.
    BoneSpaceTransforms = SkelMesh->GetRefSkeleton().GetRefBonePose();
    MarkRefreshTransformDirty();

    FWriteScopeLock Lock(CacheLock);
    CachedOutputPose = FSuperDexPose();
    CachedOutputPose.SourceName = TEXT("RefSkeleton");
    CachedOutputPose.Tags.Add(TEXT("InEditorOnly"));
    CachedOutputPose.SkeletonRootToVrRoot = FTransform::Identity;
    CachedOutputPose.TrackingConfidence = 1;
    CachedOutputPose.bIsTrackingConfident = true;
    CachedOutputPose.bIsTrackingEnabled = true;

    bCacheInitialized = true;
  }

  // Apply the cached pose directly to the mesh bones for editor preview
  ApplyCachedPoseToMesh();
}

void USuperDexPoseableComponent::BeginPlay() {
  Super::BeginPlay();

  // Scale bounds to prevent frustum culling.
  SetBoundsScale(1000);

  // figure out our handedness
  if (HandednessOverride != ESuperDexHandedness::None) {
    Handedness = HandednessOverride;
  } else {
    // if you don't have a handedness override set, try to get handedness from your attached
    // ASuperDexPoseableActor if possible
    ASuperDexPoseableActor* PoseableActor = Cast<ASuperDexPoseableActor>(GetOwner());
    if (PoseableActor != nullptr) {
      Handedness = PoseableActor->Handedness;
    }
  }

  if (Handedness == ESuperDexHandedness::None) {
    UE_LOG(
        LogSuperDexPose,
        Warning,
        TEXT(
            "[%hs::%d] Handedness not set on component [%s]."
            " If unexpected, make sure HandednessOverride is set, or the component is attached to a valid posable actor"),
        __FUNCTION__,
        __LINE__,
        *GetName());
  }

  // check for multiple matching children in the initial setup
  int NumMatchingChildren = 0;
  TArray<USceneComponent*> const& AttachedChildren = GetAttachChildren();
  for (USceneComponent* Child : AttachedChildren) {
    USuperDexPoseableComponent* ChildSource = Cast<USuperDexPoseableComponent>(Child);
    if (IsValid(ChildSource)) {
      if (!ChildSourcePoseableExcludeList.Contains(ChildSource)) {
        if (ChildSource->GetName().Contains(ChildSourceNameContainsSubstring)) {
          NumMatchingChildren++;
        }
      }
    }
  }
  if (NumMatchingChildren > 1) {
    UE_LOG(
        LogSuperDexPose,
        Warning,
        TEXT(
            "[%hs::%d] Multiple children with name containing substring [%s] found on component [%s]."
            " Only the first child indexed will be used as the pose source."),
        __FUNCTION__,
        __LINE__,
        *ChildSourceNameContainsSubstring,
        *GetName());
  }
}

void USuperDexPoseableComponent::TickComponent(
    float DeltaTime,
    enum ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction) {
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

  // Always call GetPose() on UE Tick() to ensure a recent pose and cache have been generated.
  // Derived classes should have a minimalistic, pass through, or appropriately cached GetPose().
  FSuperDexPose Pose;
  bool bHavePose = GetPose(Pose);
  // Update the cache if needed
  OnTickManageCache(Pose, bHavePose);
  // Apply the cached pose to the mesh's BoneSpaceTransforms for rendering
  ApplyCachedPoseToMesh();
  // Update the component relative transform based on the cached pose
  OnTickUpdateComponentActorRelativeTransformFromCache();
}

void USuperDexPoseableComponent::OnTickManageCache(FSuperDexPose& Pose, bool bIsValid) {
  // The base class just stores whatever pose was handed in the cache if it is valid
  if (bIsValid) {
    FWriteScopeLock Lock(CacheLock);
    CachedOutputPose = Pose;
    bCacheInitialized = true;
  }
}

void USuperDexPoseableComponent::OnTickUpdateComponentActorRelativeTransformFromCache() {
  // if we can get a pose, and we're in a game world
  if (GWorld->IsGameWorld()) {
    if (bCacheInitialized == false) {
      return; // do nothing if no cache
    }
    // get the transform of the poseable component who is placed at the pawn's VR root component
    FTransform ActorToWorld = GetOwner()->GetActorTransform();
    ActorToWorld.SetScale3D(FVector::OneVector);
    FTransform const& VrRootToWorld = ActorToWorld;

    FTransform SkeletonRootToVrRoot;
    {
      FWriteScopeLock Lock(CacheLock);
      // get this component's transform relative to vrRoot
      SkeletonRootToVrRoot = CachedOutputPose.SkeletonRootToVrRoot;
      CachedVrRootToWorld = VrRootToWorld;
    }
    if (bUpdateComponentToRootTransformOnTick) {
      // compute the skeletonRoot pose in world space
      // No need to set skeletonRootToVrRoot scale to 1, it is not applied to location/rotation
      const FTransform SkeletonRootToWorld = SkeletonRootToVrRoot * ActorToWorld;
      // update location/rotation
      SetWorldLocationAndRotation(
          SkeletonRootToWorld.GetLocation(), SkeletonRootToWorld.GetRotation(), false);

      // update scale
      if (bOverrideOutputPoseScale) {
        SetWorldScale3D(FVector(OutputPoseScaleOverride));
      } else {
        SetWorldScale3D(SkeletonRootToVrRoot.GetScale3D());
      }
    }
  }
}

bool USuperDexPoseableComponent::InitBoneToRestPoseRootTransforms() const {
  if (BoneToRestPoseRootTransforms.Num()) {
    return true; // Already initialized
  }

  // Compute the transform from each bone to the rest pose hand root
  USkinnedAsset* SkelMesh = GetSkinnedAsset();
  if (!SkelMesh) {
    return false;
  }
  const FReferenceSkeleton& RefSkeleton = SkelMesh->GetRefSkeleton();
  BoneToRestPoseRootTransforms.SetNum(RefSkeleton.GetNum());

  // First we get bone-to-parent
  for (int BoneIndex = 0; BoneIndex < RefSkeleton.GetNum(); ++BoneIndex) {
    BoneToRestPoseRootTransforms[BoneIndex] = RefSkeleton.GetRefBonePose()[BoneIndex];
  }

  // Then concatenate each transform with its parent transform to get bone-to-root
  static float ScaleThreshold = 0.001f;
  int BadScales = 0;

  for (int BoneIndex = 1; BoneIndex < RefSkeleton.GetNum(); ++BoneIndex) {
    const int ParentBone = RefSkeleton.GetParentIndex(BoneIndex);
    check(ParentBone < BoneIndex);

    // C = A * B will yield a transform C that logically first
    // applies A then B to any subsequent transformation.
    FTransform BoneToParent = BoneToRestPoseRootTransforms[BoneIndex];
    FTransform ParentToRoot = BoneToRestPoseRootTransforms[ParentBone];
    BoneToRestPoseRootTransforms[BoneIndex] = BoneToParent * ParentToRoot;

    // Check skeletal mesh transform scales
    // (Blender -> UE export can lead to non-unit scales on bone transforms if not done correctly)
    FVector Scale = BoneToRestPoseRootTransforms[BoneIndex].GetScale3D();
    float ScaleMax = Scale.GetAbsMax();
    float ScaleMin = Scale.GetAbsMin();
    if (ScaleMax > (1.0f + ScaleThreshold) || ScaleMin < (1.0f - ScaleThreshold)) {
      BadScales++;
    }
  }

  if (BadScales > 0) {
    UE_LOG(
        LogSuperDexPose,
        Warning,
        TEXT(
            "[%s] Skeletal mesh %s has %d bad bone transform scales (scale should be 1.0 in all cases). Possible problem with Blender -> UE export process."),
        ANSI_TO_TCHAR(__FUNCTION__),
        GetSkinnedAsset()->GetName().GetCharArray().GetData(),
        BadScales);
  }

  return true;
}

void USuperDexPoseableComponent::GetLifetimeReplicatedProps(
    TArray<FLifetimeProperty>& OutLifetimeProps) const {
  Super::GetLifetimeReplicatedProps(OutLifetimeProps);

  // Replicate to everyone
  DOREPLIFETIME(USuperDexPoseableComponent, CachedOutputPose);
}
