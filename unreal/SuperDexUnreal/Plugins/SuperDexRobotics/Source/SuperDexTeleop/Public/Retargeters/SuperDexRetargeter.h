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

#include "SuperDexPoseableComponent.h"

#include "SuperDexRetargeter.generated.h"

class USuperDexPoseTransformAction;

/**
 * A poseable component that computes (and displays) a transformed pose from an input pose.
 *
 * Contains the overridable function GetTransformedPose that modifies a pose. It returns the output
 * of this function when GetPose() is called on it. Manages its own cache for this purpose as well.
 */
UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class SUPERDEXTELEOP_API USuperDexRetargeter : public USuperDexPoseableComponent {
  GENERATED_BODY()

 public:
  /**
   * If true, the PoseTransformAction will be invoked even if the SourcePose returned with false on
   * GetPose(). This value should be set by actions that are designed to handle invalid input poses.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDexPoseableComponent")
  bool bGetTransformedPoseDoesNotNeedValidSourcePose = false;

  /**
   * Computes the transformed pose from the source pose. The USuperDexRetargeter will
   * only overwrite its outPose if both this function and its GetPose() for the SourcePose returned
   * true.
   *
   * @param SourcePose The incoming pose to be transformed.
   * @param TransformedPose The output pose, i.e. SourcePose after transformation.
   * @return true if successful, false otherwise
   */
  UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "SuperDexPoseableComponent")
  bool GetTransformedPose(
      FSuperDexPose const& SourcePose,
      bool bSourceIsValid,
      FSuperDexPose& TransformedPose);
  virtual bool GetTransformedPose_Implementation(
      FSuperDexPose const& SourcePose,
      bool bSourceIsValid,
      FSuperDexPose& TransformedPose);

  /**
   * Get pose for this component using the managed pose transformation action.
   *
   * @param OutPose The output pose after transformation.
   * @return true if successful, false otherwise.
   */
  virtual bool GetPose(FSuperDexPose& OutPose) override;

 protected:
  virtual void OnTickManageCache(FSuperDexPose& Pose, bool isValid) override;

  /**
   * Get the highest priority last called within a certain time frame.
   * Duplicate function to the one found in the USuperDexPoseSubsystem.
   *
   * @param currentTime The current time.
   * @param withinDeltaSecondsAgo The time frame to check within.
   * @return The highest priority last called within the specified time frame.
   */
  uint32 GetHighestPriorityLastCalledWithin(double CurrentTime, double WithinDeltaSecondsAgo);

 private:
  /** Key as of the last cache update. If a new key does not match it means the cache is invalid. */
  double CacheKey = -1;

  /** Map of cache priorities to their corresponding last request times. */
  TMap<uint32, double> CachePriorityToLastRequestTimeMap;
};
