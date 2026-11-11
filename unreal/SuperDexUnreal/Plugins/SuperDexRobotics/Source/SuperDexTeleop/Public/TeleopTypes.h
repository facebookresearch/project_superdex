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

constexpr const char* kArmPoseTargetTopic = "arm_pose_target";
constexpr const char* kArmJointTargetsTopic = "arm_joint_targets";
constexpr const char* kArmJointStateTopic = "arm_joint_states";
constexpr const char* kArmEndEffectorPoseStateTopic = "arm_ee_pose_state";
constexpr const char* kArmEndEffectorStiffnessTargetTopic = "arm_ee_stiffness_target";
constexpr const char* kArmEndEffectorStiffnessStateTopic = "arm_ee_stiffness_state";

constexpr const char* kHandJointTargetTopic = "hand_joint_targets";
constexpr const char* kHandJointStateTopic = "hand_joint_states";
constexpr const char* kHandStiffnessTargetTopic = "hand_stiffness_targets";
constexpr const char* kHandStiffnessStateTopic = "hand_stiffness_states";
constexpr const char* kHandSensorStateTopic = "fingertip_pressures";

constexpr const char* kArmJointVelocitiesTopic = "arm_joint_velocities";
constexpr const char* kArmJointEffortsTopic = "arm_joint_efforts";
constexpr const char* kArmControlPolicyEffortsTopic = "arm_joint_control_policy_efforts";

constexpr const char* kHandJointVelocitiesTopic = "hand_joint_velocities";
constexpr const char* kHandJointEffortsTopic = "hand_joint_efforts";
constexpr const char* kHandControlPolicyEffortsTopic = "hand_joint_control_policy_efforts";

constexpr int kNumDefaultArmJoints = 7;
constexpr int kNumDefaultGripperJoints = 1;
constexpr int kNumDefaultHandJoints = 16;
constexpr int kNumDefaultHandSensors = 5;

} // namespace Teleop

//------------------------------------------------------------------------------
// HANDLE TYPES
//------------------------------------------------------------------------------

USTRUCT(BlueprintType)
struct FTeleopPublisherHandle {
  GENERATED_BODY()
 public:
  UPROPERTY()
  int32 Id = -1;

  bool IsValid() const {
    return Id >= 0;
  }

  void Reset() {
    Id = -1;
  }
};

USTRUCT(BlueprintType)
struct FTeleopSubscriberHandle {
  GENERATED_BODY()
 public:
  UPROPERTY()
  int32 Id = -1;

  bool IsValid() const {
    return Id >= 0;
  }

  void Reset() {
    Id = -1;
  }
};

//------------------------------------------------------------------------------
// THREAD-SAFE DATA STRUCTURES
//------------------------------------------------------------------------------

// clang-format off
// A mutex guarded dynamic array that is gettable only once set.
template <typename T>
struct TSafeLatentArray {
  TSafeLatentArray() : Data({}) { }
  void Init(int N) { FWriteScopeLock lock(Mutex); Data.SetNumZeroed(N); bIsSet = false; }
  void Mutate(const TFunction<void(TArray<T>&)> Func) { FWriteScopeLock lock(Mutex); Func(Data); bIsSet = true; }
  bool TryGet(TArray<T>& Out) const { FReadScopeLock lock(Mutex); if (bIsSet) { Out = Data; } return bIsSet; }
  bool TryGetFirst(T& Out) const { FReadScopeLock lock(Mutex); bool b = bIsSet && !Data.IsEmpty(); if (b) { Out = Data[0]; } return b; }
  bool IsSet() const { return bIsSet; }
  int Num() const { return Data.Num(); }
private:
  mutable FRWLock Mutex;
  TArray<T> Data = {};
  std::atomic<bool> bIsSet = false;
};
// clang-format on

// Container for related pub/sub and associated data buffers
// Instances of this type don't need to use both pub and sub, but most do
template <typename T>
struct TTeleopPubSub {
  void Init(int N) {
    PubData.Init(N);
    SubData.Init(N);
  }
  FTeleopPublisherHandle Pub;
  TSafeLatentArray<T> PubData = {};
  FTeleopSubscriberHandle Sub;
  TSafeLatentArray<T> SubData = {};
  int Num() const {
    return PubData.Num();
  }
};

//------------------------------------------------------------------------------
// HELPER FUNCTIONS
//------------------------------------------------------------------------------

// Helper function to remove whitespace and leading/trailing slashes from a string
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

// Concatenates strings using '/' for constructing topics
// Always starts with "/teleop/"
// e.g. /teleop/left/arm_joint_states
template <typename... Args>
FString BuildTeleopTopic(Args... args) {
  FString Result = TEXT("/teleop");
  auto Append = [&](const auto& Arg) {
    FString SanitizedArg = SanitizeTopicPathlet(FString(Arg));
    if (!SanitizedArg.IsEmpty()) {
      Result += TEXT('/');
      Result += SanitizedArg;
    }
  };
  (Append(args), ...);
  return Result;
}
