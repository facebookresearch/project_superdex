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

#include "MochiArticulation.h"
#include "Retargeters/SuperDexRetargeter.h"

#include "ZeroOrderHoldRetargeter.generated.h"

/** Retargeter that outputs a held joint configuration of Articulation, ignoring the source pose. */
UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class SUPERDEXTELEOP_API UZeroOrderHoldRetargeter : public USuperDexRetargeter {
  GENERATED_BODY()

 public:
  /** The articulation whose held joint configuration is output. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Robotics|Teleop")
  TObjectPtr<UMochiArticulation> Articulation;

  /**
   * Set the output pose of the zero order hold filter
   *
   * @param ConfigurationVector Joint positions [rad or m]; ignored unless it has one value per held
   *   DOF.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Robotics|Teleop")
  void SetPose(const TArray<double>& ConfigurationVector);

  /** Runs without a valid source pose. */
  UZeroOrderHoldRetargeter();

  /**
   * Lets the retargeter run without a valid source pose, and takes Articulation from the owner's
   * UMochiArticulationComponent when it is not set.
   */
  virtual void BeginPlay() override;
  virtual bool GetTransformedPose_Implementation(
      FSuperDexPose const& SourcePose,
      bool bSourceIsValid,
      FSuperDexPose& TransformedPose) override;

 protected:
  FRWLock HeldPoseMutex;

  /**
   * The held joint positions [rad or m]; initialized to Articulation's default pose on BeginPlay.
   */
  UPROPERTY(EditAnywhere, BlueprintSetter = SetPose, Category = "SuperDex Robotics|Teleop")
  TArray<double> HeldPose;
};
