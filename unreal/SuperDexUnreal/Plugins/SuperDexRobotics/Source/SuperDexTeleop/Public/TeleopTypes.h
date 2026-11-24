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

#include <atomic>
#include <string>

#include "TeleopTypes.generated.h"

//------------------------------------------------------------------------------
// TELEOP CONSTANTS
//------------------------------------------------------------------------------

namespace Teleop {

/** Topic names, prefixed by BuildTeleopTopic with "/teleop/" and the robot name, if any. */
constexpr const char* ArmPoseTargetTopic = "arm_pose_target";
constexpr const char* ArmJointTargetsTopic = "arm_joint_targets";
constexpr const char* ArmJointStateTopic = "arm_joint_states";
constexpr const char* ArmEndEffectorPoseStateTopic = "arm_ee_pose_state";
constexpr const char* ArmEndEffectorStiffnessTargetTopic = "arm_ee_stiffness_target";
constexpr const char* ArmEndEffectorStiffnessStateTopic = "arm_ee_stiffness_state";

constexpr const char* HandJointTargetTopic = "hand_joint_targets";
constexpr const char* HandJointStateTopic = "hand_joint_states";
constexpr const char* HandStiffnessTargetTopic = "hand_stiffness_targets";
constexpr const char* HandStiffnessStateTopic = "hand_stiffness_states";
constexpr const char* HandSensorStateTopic = "fingertip_pressures";

constexpr const char* ArmJointVelocitiesTopic = "arm_joint_velocities";
constexpr const char* ArmJointEffortsTopic = "arm_joint_efforts";
constexpr const char* ArmControlPolicyEffortsTopic = "arm_joint_control_policy_efforts";

constexpr const char* HandJointVelocitiesTopic = "hand_joint_velocities";
constexpr const char* HandJointEffortsTopic = "hand_joint_efforts";
constexpr const char* HandControlPolicyEffortsTopic = "hand_joint_control_policy_efforts";

/** Default channel sizes of a teleop node. */
constexpr int NumDefaultArmJoints = 7;
constexpr int NumDefaultGripperJoints = 1;
constexpr int NumDefaultHandJoints = 16;
constexpr int NumDefaultHandSensors = 5;

} // namespace Teleop

//------------------------------------------------------------------------------
// HANDLE TYPES
//------------------------------------------------------------------------------

/**
 * Handle to a publisher created by a teleop communication backend; invalid while Id is negative.
 */
USTRUCT(BlueprintType)
struct FTeleopPublisherHandle {
  GENERATED_BODY()
 public:
  UPROPERTY()
  int32 Id = -1;

  /** Whether Id is nonnegative; does not check registration with a backend. */
  bool IsValid() const {
    return Id >= 0;
  }

  /** Makes the handle invalid. */
  void Reset() {
    Id = -1;
  }
};

/**
 * Handle to a subscriber created by a teleop communication backend; invalid while Id is negative.
 */
USTRUCT(BlueprintType)
struct FTeleopSubscriberHandle {
  GENERATED_BODY()
 public:
  UPROPERTY()
  int32 Id = -1;

  /** Whether Id is nonnegative; does not check registration with a backend. */
  bool IsValid() const {
    return Id >= 0;
  }

  /** Makes the handle invalid. */
  void Reset() {
    Id = -1;
  }
};

//------------------------------------------------------------------------------
// THREAD-SAFE DATA STRUCTURES
//------------------------------------------------------------------------------

// clang-format off
/** A mutex guarded dynamic array that is gettable only once set. */
template <typename T>
struct TSafeLatentArray {
  TSafeLatentArray() : Data({}) { }

  /** Resizes to N zeros and marks the array unset. */
  void Init(int N) { FWriteScopeLock Lock(Mutex); Data.SetNumZeroed(N); bIsSet = false; }

  /** Edits the array under the lock and marks it set. */
  void Mutate(const TFunction<void(TArray<T>&)> Func) { FWriteScopeLock Lock(Mutex); Func(Data); bIsSet = true; }

  /** Copies the array into Out and returns true once it is set. */
  bool TryGet(TArray<T>& Out) const { FReadScopeLock Lock(Mutex); if (bIsSet) { Out = Data; } return bIsSet; }

  /** Copies the first element into Out and returns true once the array is set and not empty. */
  bool TryGetFirst(T& Out) const { FReadScopeLock Lock(Mutex); bool bResult = bIsSet && !Data.IsEmpty(); if (bResult) { Out = Data[0]; } return bResult; }

  /** Whether the array has been set since the last Init, and its size. */
  bool IsSet() const { return bIsSet; }
  int Num() const { return Data.Num(); }
private:
  mutable FRWLock Mutex;
  TArray<T> Data = {};
  std::atomic<bool> bIsSet = false;
};
// clang-format on

/**
 * Container for related pub/sub and associated data buffers
 * Instances of this type don't need to use both pub and sub, but most do
 */
template <typename T>
struct TTeleopPubSub {
  /** Resizes both buffers to N zeros and marks them unset. */
  void Init(int N) {
    PubData.Init(N);
    SubData.Init(N);
  }

  /** The publisher and the data it publishes. */
  FTeleopPublisherHandle Pub;
  TSafeLatentArray<T> PubData = {};

  /** The subscriber and the data it last received. */
  FTeleopSubscriberHandle Sub;
  TSafeLatentArray<T> SubData = {};

  /** The size of the published data. */
  int Num() const {
    return PubData.Num();
  }
};

//------------------------------------------------------------------------------
// HELPER FUNCTIONS
//------------------------------------------------------------------------------

/** Helper function to remove whitespace and leading/trailing slashes from a string */
inline FString SanitizeTopicPathlet(const FString& Str) {
  FString Result;
  for (int32 i = 0; i < Str.Len(); ++i) {
    TCHAR C = Str[i];
    if (!FChar::IsWhitespace(C)) {
      Result.AppendChar(C);
    }
  }
  // Remove leading slashes
  while (!Result.IsEmpty() && Result[0] == TEXT('/')) {
    Result.RemoveAt(0);
  }
  // Remove trailing slashes
  while (!Result.IsEmpty() && Result[Result.Len() - 1] == TEXT('/')) {
    Result.RemoveAt(Result.Len() - 1);
  }
  return Result;
}

/**
 * Concatenates strings using '/' for constructing topics
 * Always starts with "/teleop/"
 * e.g. /teleop/left/arm_joint_states
 */
template <typename... Args>
FString BuildTeleopTopic(Args... Arguments) {
  FString Result = TEXT("/teleop");
  auto Append = [&](const auto& Arg) {
    FString SanitizedArg = SanitizeTopicPathlet(FString(Arg));
    if (!SanitizedArg.IsEmpty()) {
      Result += TEXT('/');
      Result += SanitizedArg;
    }
  };
  (Append(Arguments), ...);
  return Result;
}
