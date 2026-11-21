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

#include "SuperDexPoseFunctionLibrary.h"

#include "MochiArticulationBase.h"
#include "MochiTypes.h"
#include "SuperDexPoseModule.h"
#include "SuperDexPoseTypes.h"
#include "SuperDexPoseUtils.h"

#if WITH_EDITOR
#include "EditorAssetLibrary.h"
#endif

// #pragma optimize("", off)

bool USuperDexPoseFunctionLibrary::UpdateSuperDexPoseInAsset(
    USuperDexPoseAsset* Asset,
    FSuperDexPose Pose) {
  if (Asset == nullptr) {
    return false;
  }
  Asset->Pose = Pose;

#if WITH_EDITOR
  // UE runs in a sandbox mode and so does not mark the package dirty if the normal
  // asset->MarkPackageDirty() is called. So we must instead make sure we have the actual editor
  // package and we are running with the editor rather than in a packaged build.
  if (GIsEditor && !IsRunningCommandlet()) {
    UPackage* Package = Asset->GetOutermost();
    // Ensure we're not in a PIE world duplicate
    if (!Package->HasAnyPackageFlags(PKG_PlayInEditor)) {
      Package->SetDirtyFlag(true);
    }
  }
#endif

  return true;
}

bool USuperDexPoseFunctionLibrary::ComputeSuperDexPoseFromJointPositions(
    UMochiArticulationBase* Articulation,
    const TArray<double>& JointPositions,
    FSuperDexPose& OutPose) {
  if (Articulation == nullptr || JointPositions.Num() != Articulation->GetNumDofs()) {
    return false;
  }
  OutPose.Bot = Articulation;
  OutPose.DofPositions = JointPositions;
  return true;
}

namespace {

// SetSuperDexPoseFromLinkToParentTransforms and SetSuperDexPoseFromLinkToSkeletonRootTransforms:
// DofPositions are the joint positions LinkTransforms hold.
bool SetSuperDexPoseFromJointPositionsAndLinkTransforms(
    UMochiArticulationBase& Bot,
    const TArray<FTransform>& LinkTransforms,
    TArray<double>&& DofPositions,
    FSuperDexPose& Pose) {
  // The root link's transform relative to the pose root, in either space, at the joint positions;
  // the pose root moves so that it becomes LinkTransforms[0].
  TArray<FTransform> RootLink;
  if (!Bot.ComputeLinkToParentTransforms(
          DofPositions,
          RootLink,
          /*AngleScale=*/1.0,
          /*LengthScale=*/100.0,
          /*FinalLinkIndex=*/0)) {
    return false;
  }
  Pose.SkeletonRootToVrRoot = RootLink[0].Inverse() * LinkTransforms[0] * Pose.SkeletonRootToVrRoot;
  Pose.Bot = &Bot;
  Pose.DofPositions = MoveTemp(DofPositions);
  return true;
}

} // namespace

bool USuperDexPoseFunctionLibrary::SetSuperDexPoseFromLinkToParentTransforms(
    UMochiArticulationBase* Bot,
    const TArray<FTransform>& LinkToParent,
    FSuperDexPose& Pose) {
  TArray<double> DofPositions;
  return Bot != nullptr && !LinkToParent.IsEmpty() &&
      LinkToParent.Num() == Bot->GetLinkNames().Num() &&
      Bot->ComputeJointPositionsFromLinkToParentTransforms(LinkToParent, DofPositions) &&
      SetSuperDexPoseFromJointPositionsAndLinkTransforms(
             *Bot, LinkToParent, MoveTemp(DofPositions), Pose);
}

bool USuperDexPoseFunctionLibrary::SetSuperDexPoseFromLinkToSkeletonRootTransforms(
    UMochiArticulationBase* Bot,
    const TArray<FTransform>& LinkToSkeletonRoot,
    FSuperDexPose& Pose) {
  TArray<double> DofPositions;
  return Bot != nullptr && !LinkToSkeletonRoot.IsEmpty() &&
      LinkToSkeletonRoot.Num() == Bot->GetLinkNames().Num() &&
      Bot->ComputeJointPositionsFromLinkToArticulationRootTransforms(
          LinkToSkeletonRoot, DofPositions) &&
      SetSuperDexPoseFromJointPositionsAndLinkTransforms(
             *Bot, LinkToSkeletonRoot, MoveTemp(DofPositions), Pose);
}

bool USuperDexPoseFunctionLibrary::SetSuperDexPoseFromLinkRotations(
    UMochiArticulationBase* Bot,
    const TArray<FQuat>& LinkToParentRotations,
    FSuperDexPose& Pose) {
  TArray<FTransform> LinkToParent;
  LinkToParent.Reserve(LinkToParentRotations.Num());
  for (const FQuat& Rotation : LinkToParentRotations) {
    LinkToParent.Emplace(Rotation);
  }
  return SetSuperDexPoseFromLinkToParentTransforms(Bot, LinkToParent, Pose);
}

bool USuperDexPoseFunctionLibrary::ComputeSuperDexPoseLinkToParentTransforms(
    const FSuperDexPose& Pose,
    TArray<FTransform>& OutLinkToParent) {
  return IsSuperDexPoseOnBot(Pose) &&
      Pose.Bot->ComputeLinkToParentTransforms(Pose.DofPositions, OutLinkToParent);
}

bool USuperDexPoseFunctionLibrary::ComputeSuperDexPoseLinkToSkeletonRootTransforms(
    const FSuperDexPose& Pose,
    TArray<FTransform>& OutLinkToSkeletonRoot) {
  return IsSuperDexPoseOnBot(Pose) &&
      Pose.Bot->ComputeLinkToArticulationRootTransforms(Pose.DofPositions, OutLinkToSkeletonRoot);
}

bool USuperDexPoseFunctionLibrary::IsSuperDexPoseOnBot(const FSuperDexPose& Pose) {
  return Pose.Bot != nullptr && Pose.DofPositions.Num() == Pose.Bot->GetNumDofs();
}

int32 USuperDexPoseFunctionLibrary::FindSuperDexPoseLinkIndex(
    const FSuperDexPose& Pose,
    FName LinkName) {
  return Pose.Bot != nullptr ? Pose.Bot->GetLinkNames().Find(LinkName) : INDEX_NONE;
}

bool USuperDexPoseFunctionLibrary::ComputeSuperDexPoseLinkToSkeletonRoot(
    const FSuperDexPose& Pose,
    int32 LinkIndex,
    FTransform& OutLinkToSkeletonRoot) {
  TArray<FTransform> LinkToRoot;
  if (LinkIndex < 0 || !IsSuperDexPoseOnBot(Pose) ||
      !Pose.Bot->ComputeLinkToArticulationRootTransforms(
          Pose.DofPositions,
          LinkToRoot,
          /*AngleScale=*/1.0,
          /*LengthScale=*/100.0,
          /*FinalLinkIndex=*/LinkIndex) ||
      !LinkToRoot.IsValidIndex(LinkIndex)) {
    return false;
  }
  OutLinkToSkeletonRoot = LinkToRoot[LinkIndex];
  return true;
}

bool USuperDexPoseFunctionLibrary::ComputeSuperDexPoseLinkToVrRoot(
    const FSuperDexPose& Pose,
    int32 LinkIndex,
    FTransform& OutLinkToVrRoot) {
  FTransform LinkToSkeletonRoot;
  if (!ComputeSuperDexPoseLinkToSkeletonRoot(Pose, LinkIndex, LinkToSkeletonRoot)) {
    return false;
  }
  OutLinkToVrRoot = LinkToSkeletonRoot * Pose.SkeletonRootToVrRoot;
  return true;
}

void USuperDexPoseFunctionLibrary::FireAndForgetSeconds(
    const UObject* WorldContextObject,
    float Duration,
    ESuperDexFireAndForgetExec& Exec,
    float& Time,
    float& PercentComplete,
    FLatentActionInfo LatentInfo) {
  if (Duration <= 0) {
    return;
  }
  if (UWorld* World = GEngine->GetWorldFromContextObject(
          WorldContextObject, EGetWorldErrorMode::LogAndReturnNull)) {
    FLatentActionManager& LatentActionManager = World->GetLatentActionManager();
    if (LatentActionManager.FindExistingAction<FSuperDexFireAndForgetSecondsAction>(
            LatentInfo.CallbackTarget, LatentInfo.UUID) == nullptr) {
      LatentActionManager.AddNewAction(
          LatentInfo.CallbackTarget,
          LatentInfo.UUID,
          new FSuperDexFireAndForgetSecondsAction(
              Duration, Time, PercentComplete, Exec, LatentInfo));
    }
  }
}

void USuperDexPoseFunctionLibrary::FireAndForgetFrames(
    const UObject* WorldContextObject,
    int Frames,
    ESuperDexFireAndForgetExec& Exec,
    int& Frame,
    FLatentActionInfo LatentInfo) {
  if (Frames <= 0) {
    return;
  }
  if (UWorld* World = GEngine->GetWorldFromContextObject(
          WorldContextObject, EGetWorldErrorMode::LogAndReturnNull)) {
    FLatentActionManager& LatentActionManager = World->GetLatentActionManager();
    if (LatentActionManager.FindExistingAction<FSuperDexFireAndForgetFramesAction>(
            LatentInfo.CallbackTarget, LatentInfo.UUID) == nullptr) {
      LatentActionManager.AddNewAction(
          LatentInfo.CallbackTarget,
          LatentInfo.UUID,
          new FSuperDexFireAndForgetFramesAction(Frames, Frame, Exec, LatentInfo));
    }
  }
}

void USuperDexPoseFunctionLibrary::FireAndForgetIterations(
    const UObject* WorldContextObject,
    int Iterations,
    float IterationDuration,
    ESuperDexFireAndForgetIterationExec& Exec,
    int& Iteration,
    float& IterationTime,
    float& IterationPercentComplete,
    float& Time,
    float& PercentComplete,
    FLatentActionInfo LatentInfo) {
  if (Iterations <= 0 || IterationDuration <= 0) {
    return;
  }
  if (UWorld* World = GEngine->GetWorldFromContextObject(
          WorldContextObject, EGetWorldErrorMode::LogAndReturnNull)) {
    FLatentActionManager& LatentActionManager = World->GetLatentActionManager();
    if (LatentActionManager.FindExistingAction<FSuperDexFireAndForgetIterationsAction>(
            LatentInfo.CallbackTarget, LatentInfo.UUID) == nullptr) {
      LatentActionManager.AddNewAction(
          LatentInfo.CallbackTarget,
          LatentInfo.UUID,
          new FSuperDexFireAndForgetIterationsAction(
              Iterations,
              IterationDuration,
              Iteration,
              IterationTime,
              IterationPercentComplete,
              Time,
              PercentComplete,
              Exec,
              LatentInfo));
    }
  }
}

void USuperDexPoseFunctionLibrary::FireAndForgetCurve(
    const UObject* WorldContextObject,
    UCurveFloat* Curve,
    float Speed,
    ESuperDexFireAndForgetExec& Exec,
    float& Value,
    float& Time,
    float& PercentComplete,
    FLatentActionInfo LatentInfo) {
  if (Curve == nullptr || Speed == 0) {
    return;
  }
  if (UWorld* World = GEngine->GetWorldFromContextObject(
          WorldContextObject, EGetWorldErrorMode::LogAndReturnNull)) {
    FLatentActionManager& LatentActionManager = World->GetLatentActionManager();
    if (LatentActionManager.FindExistingAction<FSuperDexFireAndForgetCurveAction>(
            LatentInfo.CallbackTarget, LatentInfo.UUID) == nullptr) {
      LatentActionManager.AddNewAction(
          LatentInfo.CallbackTarget,
          LatentInfo.UUID,
          new FSuperDexFireAndForgetCurveAction(
              Curve, Speed, Value, Time, PercentComplete, Exec, LatentInfo));
    }
  }
}
