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
#include "Kismet/BlueprintFunctionLibrary.h"
#include "SuperDexLatentActions.h"
#include "SuperDexPoseTypes.h"

#include "SuperDexPoseFunctionLibrary.generated.h"

class UMochiArticulationBase;

/** Blueprint helpers for building, combining and storing SuperDex poses. */
UCLASS(BlueprintType)
class SUPERDEXPOSE_API USuperDexPoseFunctionLibrary : public UBlueprintFunctionLibrary {
  GENERATED_BODY()

 public:
  /**
   * This function multiplies two transforms together (A*B) treating them as a pair of position
   * rotation transforms without scale but preserving the scale terms at the end. This is useful
   * when compositing SuperDexPose::SkeletonRootToVrRoot but you dont want to have the source pose
   * scale influence the result.
   *
   * S(AxB) = S(A)*S(B)
   * Q(AxB) = Q(B)*Q(A)
   * T(AxB) = Q(B)*T(A)*-Q(B) + T(B)
   *
   * @param A First transform.
   * @param B Second transform.
   * @return A*B composed without scale, with the scale set to S(A)*S(B).
   */
  UFUNCTION(BlueprintPure, Category = "SuperDex Robotics|Pose")
  static FTransform MultiplyTransformsWithoutScalePreservingScale(
      FTransform const& A,
      FTransform const& B) {
    FTransform ANoScale = FTransform(A.GetRotation(), A.GetTranslation(), FVector::OneVector);
    FTransform BNoScale = FTransform(B.GetRotation(), B.GetTranslation(), FVector::OneVector);
    FTransform Result = ANoScale * BNoScale;
    Result.SetScale3D(A.GetScale3D() * B.GetScale3D());
    return Result;
  }

  /**
   * This function multiplies two transforms together (A*B) stripping out the scale first.
   *
   * S(AxB) = 1
   * Q(AxB) = Q(B)*Q(A)
   * T(AxB) = Q(B)*T(A)*-Q(B) + T(B)
   *
   * @param A First transform.
   * @param B Second transform.
   * @return A*B composed with both scales removed.
   */
  UFUNCTION(BlueprintPure, Category = "SuperDex Robotics|Pose")
  static FTransform MultiplyTransformsStrippingScale(FTransform const& A, FTransform const& B) {
    FTransform ANoScale = FTransform(A.GetRotation(), A.GetTranslation(), FVector::OneVector);
    FTransform BNoScale = FTransform(B.GetRotation(), B.GetTranslation(), FVector::OneVector);
    FTransform Result = ANoScale * BNoScale;
    return Result;
  }

  /**
   * Copies a pose into a pose asset. In the editor (outside commandlets) the asset's package is
   * also marked dirty unless it is a Play In Editor package.
   *
   * @param Asset Pose asset to update.
   * @param Pose Pose to store.
   * @return False if Asset is null, otherwise true.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Robotics|Pose")
  static bool UpdateSuperDexPoseInAsset(USuperDexPoseAsset* Asset, FSuperDexPose Pose);

  /**
   * Sets OutPose to a joint configuration [rad or m] of an articulation: OutPose.Bot becomes the
   * articulation and OutPose.DofPositions the joint positions. Fails, leaving OutPose unchanged, if
   * there is not one joint position per DOF of the articulation. Leaves the rest of OutPose
   * unchanged.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Robotics|Pose")
  static bool ComputeSuperDexPoseFromJointPositions(
      UMochiArticulationBase* Articulation,
      const TArray<double>& JointPositions,
      FSuperDexPose& OutPose);

  /**
   * Sets Pose to the joint configuration of Bot that its links' transforms relative to their
   * parents hold (one per link, in nominal bot space, as
   * UMochiArticulationBase::ComputeJointPositionsFromLinkToParentTransforms takes them): Pose.Bot
   * becomes Bot and Pose.DofPositions its joint positions. What the joints can't produce, such as a
   * link's swing off its revolute axis, is discarded. The root link's transform is not a joint
   * position, so it is composed into Pose.SkeletonRootToVrRoot: the pose places the root link where
   * LinkToParent places it relative to the pose root Pose had. Fails, leaving Pose unchanged, if
   * there is not one transform per link of Bot. Leaves the rest of Pose unchanged.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex")
  static bool SetSuperDexPoseFromLinkToParentTransforms(
      UMochiArticulationBase* Bot,
      const TArray<FTransform>& LinkToParent,
      UPARAM(ref) FSuperDexPose& Pose);

  /**
   * SetSuperDexPoseFromLinkToParentTransforms for the links' transforms relative to the skeleton
   * root (UMochiArticulationBase::ComputeJointPositionsFromLinkToArticulationRootTransforms).
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex")
  static bool SetSuperDexPoseFromLinkToSkeletonRootTransforms(
      UMochiArticulationBase* Bot,
      const TArray<FTransform>& LinkToSkeletonRoot,
      UPARAM(ref) FSuperDexPose& Pose);

  /**
   * SetSuperDexPoseFromLinkToParentTransforms for link rotations relative to their parents
   * (link-to-parent transforms with no translation), such as a hand tracker's bone rotations. Only
   * prismatic joints read translations, so the rotations of a bot without them hold all its joint
   * positions.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex")
  static bool SetSuperDexPoseFromLinkRotations(
      UMochiArticulationBase* Bot,
      const TArray<FQuat>& LinkToParentRotations,
      UPARAM(ref) FSuperDexPose& Pose);

  /**
   * The transforms of a pose's links relative to their parents: its Bot's links at its
   * DofPositions (UMochiArticulationBase::ComputeLinkToParentTransforms), in nominal bot space
   * [cm]. SkeletonRootToVrRoot scales and places them. Fails if the pose is not on a bot
   * (IsSuperDexPoseOnBot).
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex")
  static bool ComputeSuperDexPoseLinkToParentTransforms(
      const FSuperDexPose& Pose,
      TArray<FTransform>& OutLinkToParent);

  /**
   * The transforms of a pose's links relative to its skeleton root (its Bot's articulation root,
   * UMochiArticulationBase::ComputeLinkToArticulationRootTransforms), as
   * ComputeSuperDexPoseLinkToParentTransforms.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex")
  static bool ComputeSuperDexPoseLinkToSkeletonRootTransforms(
      const FSuperDexPose& Pose,
      TArray<FTransform>& OutLinkToSkeletonRoot);

  /** Whether a pose is on a bot: it has a Bot and one joint position per DOF of it. */
  UFUNCTION(BlueprintPure, Category = "SuperDex")
  static bool IsSuperDexPoseOnBot(const FSuperDexPose& Pose);

  /**
   * The index of the link of a pose's Bot named LinkName (UMochiArticulationBase::GetLinkNames), or
   * INDEX_NONE if the pose has no Bot or its Bot has no such link.
   */
  UFUNCTION(BlueprintPure, Category = "SuperDex")
  static int32 FindSuperDexPoseLinkIndex(const FSuperDexPose& Pose, FName LinkName);

  /**
   * The transform of a link of a pose relative to its skeleton root: the link's transform relative
   * to its Bot's root at the pose's DofPositions (nominal size, in cm). Fails if the pose is not on
   * a bot (IsSuperDexPoseOnBot) or LinkIndex is not one of its links.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex")
  static bool ComputeSuperDexPoseLinkToSkeletonRoot(
      const FSuperDexPose& Pose,
      int32 LinkIndex,
      FTransform& OutLinkToSkeletonRoot);

  /**
   * The transform of a link of a pose relative to its VrRoot: its transform relative to the
   * skeleton root (ComputeSuperDexPoseLinkToSkeletonRoot), scaled and placed by
   * SkeletonRootToVrRoot. Where a mesh showing the pose puts that link's bone, relative to the
   * VrRoot.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex")
  static bool ComputeSuperDexPoseLinkToVrRoot(
      const FSuperDexPose& Pose,
      int32 LinkIndex,
      FTransform& OutLinkToVrRoot);

  //----------------------------------------------------------------------------
  // LATENT ACTIONS (FIRE AND FORGET)
  //----------------------------------------------------------------------------

  /**
   * A latent action which runs for Duration seconds and outputs Time and Percent Complete each
   * frame. The output values on the final execution of Update a guaranteed to be at exactly full
   * completion -- no more, no less. Finished is executed one frame after the final Update.
   *
   * @param WorldContextObject Object whose world runs the latent action.
   * @param Duration Run time [s]. Nothing starts unless it is positive.
   * @param Exec Output execution pin to fire.
   * @param Time Elapsed time [s], written on each update.
   * @param PercentComplete Fraction of Duration elapsed, written on each update.
   * @param LatentInfo Latent action info supplied by the Blueprint call node.
   */
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDex Robotics|Fire and Forget",
      meta =
          (Latent,
           DisplayName = "Fire and Forget (Seconds)",
           WorldContext = "WorldContextObject",
           ExpandEnumAsExecs = "Exec",
           LatentInfo = "LatentInfo",
           Duration = "10.0f",
           Keywords = "coroutine, async, one shot"))
  static void FireAndForgetSeconds(
      const UObject* WorldContextObject,
      float Duration,
      ESuperDexFireAndForgetExec& Exec,
      float& Time,
      float& PercentComplete,
      struct FLatentActionInfo LatentInfo);

  /**
   * A latent function which runs for a specified number of Frames. The output Frame is zero
   * indexed. The first Frame is output during the same frame in which the function is called.
   * Finished is executed one frame after the last Frame.
   *
   * @param WorldContextObject Object whose world runs the latent action.
   * @param Frames Number of frames to run. Nothing starts unless it is positive.
   * @param Exec Output execution pin to fire.
   * @param Frame Zero-based current frame, written on each update.
   * @param LatentInfo Latent action info supplied by the Blueprint call node.
   */
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDex Robotics|Fire and Forget",
      meta =
          (Latent,
           DisplayName = "Fire and Forget (Frames)",
           WorldContext = "WorldContextObject",
           ExpandEnumAsExecs = "Exec",
           LatentInfo = "LatentInfo",
           Frames = "10",
           Keywords = "coroutine, async, one shot"))
  static void FireAndForgetFrames(
      const UObject* WorldContextObject,
      int Frames,
      ESuperDexFireAndForgetExec& Exec,
      int& Frame,
      struct FLatentActionInfo LatentInfo);

  /**
   * A latent function which runs a specified number of Iterations, each lasting IterationDuration
   * seconds. The current Iteration, time in that Iteration, and total time in the function are
   * provided. The output values at the end of each iteration are guaranteed to be exactly the full
   * IterationDuration -- no more, no less. Iteration Finished is executed between Updates at
   * Iteration boundaries. Finished is executed one frame after the final Iteration Finished.
   * Beware that the total real-time duration may be greater than the indicated Time due to
   * one frame delays between Iteration transitions. This is a limitation of LatentActions, but
   * should matter little in practice.
   *
   * @param WorldContextObject Object whose world runs the latent action.
   * @param Iterations Number of iterations. Nothing starts unless it is positive.
   * @param IterationDuration Length of each iteration [s]. Nothing starts unless it is positive.
   * @param Exec Output execution pin to fire.
   * @param Iteration Current iteration, written on each update.
   * @param IterationTime Elapsed time in the current iteration [s].
   * @param IterationPercentComplete Fraction of the current iteration elapsed.
   * @param Time Total elapsed time [s].
   * @param PercentComplete Fraction of the total duration elapsed.
   * @param LatentInfo Latent action info supplied by the Blueprint call node.
   */
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDex Robotics|Fire and Forget",
      meta =
          (Latent,
           DisplayName = "Fire and Forget (Iterations)",
           WorldContext = "WorldContextObject",
           ExpandEnumAsExecs = "Exec",
           LatentInfo = "LatentInfo",
           Iterations = "10",
           IterationDuration = "1.0f",
           Keywords = "coroutine, async, one shot"))
  static void FireAndForgetIterations(
      const UObject* WorldContextObject,
      int Iterations,
      float IterationDuration,
      ESuperDexFireAndForgetIterationExec& Exec,
      int& Iteration,
      float& IterationTime,
      float& IterationPercentComplete,
      float& Time,
      float& PercentComplete,
      struct FLatentActionInfo LatentInfo);

  /**
   * A latent action which samples a single Curve at a given speed and outputs its value.
   *
   * @param WorldContextObject Object whose world runs the latent action.
   * @param Curve Curve to sample over its time range. Nothing starts if it is null.
   * @param Speed Playback rate: the time range plays over (range / Speed) seconds. Nothing starts
   *   if it is 0.
   * @param Exec Output execution pin to fire.
   * @param Value Curve value at the current playback time.
   * @param Time Elapsed time [s].
   * @param PercentComplete Fraction of the playback elapsed.
   * @param LatentInfo Latent action info supplied by the Blueprint call node.
   */
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDex Robotics|Fire and Forget",
      meta =
          (Latent,
           DisplayName = "Fire and Forget (Curve)",
           WorldContext = "WorldContextObject",
           ExpandEnumAsExecs = "Exec",
           LatentInfo = "LatentInfo",
           Speed = "1.0f",
           Keywords = "coroutine, async, one shot"))
  static void FireAndForgetCurve(
      const UObject* WorldContextObject,
      UCurveFloat* Curve,
      float Speed,
      ESuperDexFireAndForgetExec& Exec,
      float& Value,
      float& Time,
      float& PercentComplete,
      struct FLatentActionInfo LatentInfo);
};
