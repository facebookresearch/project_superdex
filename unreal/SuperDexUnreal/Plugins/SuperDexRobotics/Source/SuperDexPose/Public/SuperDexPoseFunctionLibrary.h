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
   */
  UFUNCTION(BlueprintPure, Category = "SuperDex")
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
   */
  UFUNCTION(BlueprintPure, Category = "SuperDex")
  static FTransform MultiplyTransformsStrippingScale(FTransform const& A, FTransform const& B) {
    FTransform ANoScale = FTransform(A.GetRotation(), A.GetTranslation(), FVector::OneVector);
    FTransform BNoScale = FTransform(B.GetRotation(), B.GetTranslation(), FVector::OneVector);
    FTransform Result = ANoScale * BNoScale;
    return Result;
  }

  UFUNCTION(BlueprintCallable, Category = "SuperDex")
  static bool UpdateSuperDexPoseInAsset(USuperDexPoseAsset* Asset, FSuperDexPose Pose);

  // Sets OutPose's bones (one per link, parent-relative) to an articulation's link transforms at a
  // joint configuration [rad or m]. Leaves the rest of OutPose unchanged.
  UFUNCTION(BlueprintCallable, Category = "SuperDex")
  static bool ComputeSuperDexPoseFromJointPositions(
      const UMochiArticulationBase* Articulation,
      const TArray<double>& JointPositions,
      FSuperDexPose& OutPose);

  //----------------------------------------------------------------------------
  // LATENT ACTIONS (FIRE AND FORGET)
  //----------------------------------------------------------------------------

  // A latent action which runs for Duration seconds and outputs Time and Percent Complete each
  // frame. The output values on the final execution of Update a guaranteed to be at exactly full
  // completion -- no more, no less. Finished is executed one frame after the final Update.
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDex|Fire and Forget",
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

  // A latent function which runs for a specified number of Frames. The output Frame is zero
  // indexed. The first Frame is output during the same frame in which the function is called.
  // Finished is executed one frame after the last Frame.
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDex|Fire and Forget",
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

  // A latent function which runs a specified number of Iterations, each lasting IterationDuration
  // seconds. The current Iteration, time in that Iteration, and total time in the function are
  // provided. The output values at the end of each iteration are guaranteed to be exactly the full
  // IterationDuration -- no more, no less. Iteration Finished is executed between Updates at
  // Iteration boundaries. Finished is executed one frame after the final Iteration Finished.
  // Beware that the total real-time duration may be greater than the indicated Time due to
  // one frame delays between Iteration transitions. This is a limitation of LatentActions, but
  // should matter little in practice.
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDex|Fire and Forget",
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

  // A latent action which samples a single Curve at a given speed and outputs its value.
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDex|Fire and Forget",
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
