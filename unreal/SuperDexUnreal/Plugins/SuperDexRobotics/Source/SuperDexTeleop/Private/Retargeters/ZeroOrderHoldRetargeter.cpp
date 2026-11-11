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

#include "Retargeters/ZeroOrderHoldRetargeter.h"

#include "MochiArticulationComponent.h"
#include "SuperDexPoseFunctionLibrary.h"
#include "SuperDexTeleopModule.h"

void UZeroOrderHoldRetargeter::SetPose(const TArray<double>& ConfigurationVector) {
  // Verify that the dimension of the new pose corresponds to the length we're expecting based on
  // the robot description articulation.
  if (ConfigurationVector.Num() == HeldPose.Num()) {
    FWriteScopeLock lock(HeldPoseMutex);
    HeldPose = ConfigurationVector;
  }
}

UZeroOrderHoldRetargeter::UZeroOrderHoldRetargeter() {
  bGetTransformedPoseDoesNotNeedValidSourcePose = true;
}

void UZeroOrderHoldRetargeter::BeginPlay() {
  Super::BeginPlay();

  bGetTransformedPoseDoesNotNeedValidSourcePose = true;

  // Find an attached UMochiArticulationComponent that holds the articulation asset describing our
  // kinematics
  if (!Articulation) {
    auto const* robot = GetOwner()->FindComponentByClass<UMochiArticulationComponent>();
    if (robot) {
      Articulation = robot->GetArticulationAsset();
    } else {
      UE_LOG(
          LogSuperDexTeleop,
          Warning,
          TEXT("[ZeroOrderHoldRetargeter] Cound not find robot component on actor!"));
    }
  }
  // Initialize the held pose array with home pose from articulation
  if (Articulation) {
    if (HeldPose.Num() != Articulation->GetDefaultJointPositions().Num()) {
      HeldPose = Articulation->GetDefaultJointPositions();
    }
  }
}

bool UZeroOrderHoldRetargeter::GetTransformedPose_Implementation(
    FSuperDexPose const& SourcePose,
    bool SourceIsValid,
    FSuperDexPose& TransformedPose) {
  if (!Articulation) {
    return false;
  }

  // Just forward the held pose into the TransformedPose output
  FReadScopeLock lock(HeldPoseMutex);
  bool result = USuperDexPoseFunctionLibrary::ComputeSuperDexPoseFromJointPositions(
      Articulation.Get(), HeldPose, TransformedPose);
  TransformedPose.bIsTrackingEnabled = result;
  TransformedPose.bIsTrackingConfident = result;
  TransformedPose.TrackingConfidence = result ? 1.0f : 0.0f;
  TransformedPose.tags.Add(TEXT("ZeroOrderHold"));
  TransformedPose.SkeletonName = FName(Articulation->GetArticulationName());
  return result;
}
