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
    const UMochiArticulationBase* Articulation,
    const TArray<double>& JointPositions,
    FSuperDexPose& OutPose) {
  TArray<FTransform> LinkToParent;
  if (Articulation == nullptr ||
      !Articulation->ComputeLinkTransformsFromJointPositions(
          JointPositions, LinkToParent, EMochiArticulationLinkTransformSpace::LinkToParent)) {
    return false;
  }
  OutPose.BoneRotations.SetNum(LinkToParent.Num());
  OutPose.BoneLocations.SetNum(LinkToParent.Num());
  for (int32 I = 0; I < LinkToParent.Num(); ++I) {
    OutPose.BoneRotations[I] = LinkToParent[I].GetRotation();
    OutPose.BoneLocations[I] = LinkToParent[I].GetLocation();
  }
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
