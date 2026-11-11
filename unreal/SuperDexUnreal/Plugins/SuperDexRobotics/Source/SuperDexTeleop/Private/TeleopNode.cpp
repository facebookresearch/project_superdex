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

#include "TeleopNode.h"

#include "SuperDexTeleopModule.h"
#include "TeleopNodeImplBase.h"
#include "TeleopNodeSettings.h"
#include "TeleopTypes.h"
#include "UObject/CoreRedirects.h"

//------------------------------------------------------------------------------
// TEMPLATE HELPER IMPLEMENTATIONS
//------------------------------------------------------------------------------

namespace {

// SetPubDataWithSizeCheck: Validates size and copies data with warning on mismatch. Returns
// whether the full target was written.
template <typename T>
bool SetPubDataWithSizeCheck(
    const TArray<T>& Source,
    TTeleopPubSub<T>& PubSub,
    const TCHAR* Context,
    const TCHAR* Func) {
  if (!PubSub.Pub.IsValid()) {
    return false;
  }
  auto& SafeDest = PubSub.PubData;
  const int32 Count = FMath::Min(SafeDest.Num(), Source.Num());
  if (SafeDest.Num() != Source.Num()) {
    UE_LOG(
        LogSuperDexTeleop,
        Warning,
        TEXT("[%s] %s passed %d values but expected %d; setting only %d values!"),
        Context,
        Func,
        Source.Num(),
        SafeDest.Num(),
        Count);
  }
  SafeDest.Mutate([&](auto& Data) {
    for (int32 i = 0; i < Count; ++i) {
      Data[i] = Source[i];
    }
  });
  return SafeDest.Num() == Source.Num();
}

// CreatePublisher: Creates a publisher and stores the handle
template <typename T>
void CreatePublisher(
    UTeleopNodeImplBase* Comm,
    const FString& RobotName,
    const char* Topic,
    TTeleopPubSub<T>& PubSub);

template <>
void CreatePublisher<FTransform>(
    UTeleopNodeImplBase* Comm,
    const FString& RobotName,
    const char* Topic,
    TTeleopPubSub<FTransform>& PubSub) {
  PubSub.Pub = Comm->CreatePublisher_TransformArray(BuildTeleopTopic(RobotName, Topic));
}

template <>
void CreatePublisher<double>(
    UTeleopNodeImplBase* Comm,
    const FString& RobotName,
    const char* Topic,
    TTeleopPubSub<double>& PubSub) {
  PubSub.Pub = Comm->CreatePublisher_DoubleArray(BuildTeleopTopic(RobotName, Topic));
}

// CreateSubscriber: Creates a subscriber with auto-callback to populate SubData
template <typename T>
void CreateSubscriber(
    UTeleopNodeImplBase* Comm,
    const FString& RobotName,
    const char* Topic,
    TTeleopPubSub<T>& PubSub);

template <>
void CreateSubscriber<FTransform>(
    UTeleopNodeImplBase* Comm,
    const FString& RobotName,
    const char* Topic,
    TTeleopPubSub<FTransform>& PubSub) {
  PubSub.Sub = Comm->CreateSubscriber_TransformArray(
      BuildTeleopTopic(RobotName, Topic), [&PubSub](const TArray<FTransform>& Data) {
        PubSub.SubData.Mutate([&](auto& Arr) { Arr = Data; });
      });
}

template <>
void CreateSubscriber<double>(
    UTeleopNodeImplBase* Comm,
    const FString& RobotName,
    const char* Topic,
    TTeleopPubSub<double>& PubSub) {
  PubSub.Sub = Comm->CreateSubscriber_DoubleArray(
      BuildTeleopTopic(RobotName, Topic), [&PubSub](const TArray<double>& Data) {
        PubSub.SubData.Mutate([&](auto& Arr) { Arr = Data; });
      });
}

// Publish: Publishes data if handle is valid and data is available
template <typename T>
void Publish(UTeleopNodeImplBase* Comm, TTeleopPubSub<T>& PubSub);

template <>
void Publish<FTransform>(UTeleopNodeImplBase* Comm, TTeleopPubSub<FTransform>& PubSub) {
  if (!PubSub.Pub.IsValid()) {
    return;
  }
  TArray<FTransform> Data;
  if (PubSub.PubData.TryGet(Data)) {
    Comm->Publish_TransformArray(PubSub.Pub, Data);
  }
}

template <>
void Publish<double>(UTeleopNodeImplBase* Comm, TTeleopPubSub<double>& PubSub) {
  if (!PubSub.Pub.IsValid()) {
    return;
  }
  TArray<double> Data;
  if (PubSub.PubData.TryGet(Data)) {
    Comm->Publish_DoubleArray(PubSub.Pub, Data);
  }
}

} // namespace

//------------------------------------------------------------------------------
// BASE COMPONENT IMPLEMENTATION
//------------------------------------------------------------------------------

UTeleopNodeBaseComponent::UTeleopNodeBaseComponent() {
  PrimaryComponentTick.bCanEverTick = true;
  PrimaryComponentTick.bStartWithTickEnabled = false;
  PrimaryComponentTick.TickGroup = TG_PrePhysics;

  SecondaryComponentTick.bCanEverTick = true;
  SecondaryComponentTick.bStartWithTickEnabled = false;
  SecondaryComponentTick.TickGroup = TG_PostUpdateWork;
}

void UTeleopNodeBaseComponent::BeginPlay() {
  Super::BeginPlay();

  NodeName = GetOwner()->GetName() + TEXT("_") + GetName();

  ArmPose.Init(1);
  ArmEndEffectorStiffness.Init(9);
  ArmJointPositions.Init(NumArmJoints);
  HandJointPositions.Init(NumHandJoints);
  HandStiffness.Init(NumHandJoints);
  ArmJointVelocities.Init(NumArmJoints);
  HandJointVelocities.Init(NumHandJoints);
  ArmJointEfforts.Init(NumArmJoints);
  ArmJointControlPolicyEfforts.Init(NumArmJoints);
  HandJointEfforts.Init(NumHandJoints);
  HandJointControlPolicyEfforts.Init(NumHandJoints);
  HandSensors.Init(bHandSensorsMeasureForce ? NumHandSensors * 3 : NumHandSensors);

  UClass* CommunicationClass = nullptr;

  if (CommunicationClassName.IsEmpty()) {
    CommunicationClassName = GetDefault<UTeleopNodeSettings>()->DefaultCommunicationClassName;
    if (CommunicationClassName.IsEmpty()) {
      CommunicationClassName = TEXT("/Script/SuperDexTeleop.TeleopNodeImplDirect");
    }
    UE_LOG(
        LogSuperDexTeleop,
        Warning,
        TEXT("TeleopNode [%s]: Empty CommunicationClassName, defaulting to [%s]"),
        *NodeName,
        *CommunicationClassName);
  }

  // CommunicationClassName is a plain string, and LoadClass applies only package redirects. Load
  // the name as stored first, so that a broad class redirect cannot replace a class that exists,
  // and fall back to the class redirects for names saved before a class was renamed or moved.
  CommunicationClass = LoadClass<UTeleopNodeImplBase>(
      nullptr, *CommunicationClassName, nullptr, LOAD_Quiet | LOAD_NoWarn);
  if (CommunicationClass == nullptr) {
    const FString RedirectedName =
        FCoreRedirects::GetRedirectedName(
            ECoreRedirectFlags::Type_Class, FCoreRedirectObjectName(CommunicationClassName))
            .ToString();
    if (!RedirectedName.IsEmpty() && RedirectedName != CommunicationClassName) {
      CommunicationClass = LoadClass<UTeleopNodeImplBase>(nullptr, *RedirectedName);
      if (CommunicationClass != nullptr) {
        CommunicationClassName = RedirectedName;
      }
    }
  }

  if (CommunicationClass == nullptr) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("Could not find CommunicationClass %s. Is its module not loaded?"),
        *CommunicationClassName);
    return;
  }

  Communication = NewObject<UTeleopNodeImplBase>(this, CommunicationClass);
  if (!Communication) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("TeleopNode [%s]: Failed to create Communication object"),
        *NodeName);
    return;
  }

  Communication->SetName(NodeName);

  if (!Communication->Initialize()) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("TeleopNode [%s]: Communication failed to initialize"),
        *NodeName);
    return;
  }

  bInitialized = OnCreatePubSubs();
  if (!bInitialized) {
    UE_LOG(LogSuperDexTeleop, Error, TEXT("TeleopNode [%s]: Failed to create pub/subs"), *NodeName);
    return;
  }

  UE_LOG(LogSuperDexTeleop, Log, TEXT("TeleopNode [%s]: Initialized successfully"), *NodeName);
  PrimaryComponentTick.SetTickFunctionEnable(bTickOnPrePhysics);
  SecondaryComponentTick.SetTickFunctionEnable(bTickOnPostUpdateWork);
}

void UTeleopNodeBaseComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  if (Communication) {
    Communication->Deinitialize();
  }
  bInitialized = false;
  Super::EndPlay(EndPlayReason);
}

void UTeleopNodeBaseComponent::RegisterComponentTickFunctions(bool bRegister) {
  Super::RegisterComponentTickFunctions(bRegister);

  if (bRegister) {
    if (SetupActorComponentTickFunction(&SecondaryComponentTick)) {
      SecondaryComponentTick.Target = this;
      SecondaryComponentTick.AddPrerequisite(this, PrimaryComponentTick);
    }
  } else {
    if (SecondaryComponentTick.IsTickFunctionRegistered()) {
      SecondaryComponentTick.UnRegisterTickFunction();
    }
  }
}

void UTeleopNodeBaseComponent::TickComponent(
    float DeltaTime,
    ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction) {
  if (!bInitialized || !Communication) {
    return;
  }
  if (!bTickOnPrePhysics && ThisTickFunction->TickGroup == TG_PrePhysics) {
    return;
  }
  if (!bTickOnPostUpdateWork && ThisTickFunction->TickGroup == TG_PostUpdateWork) {
    return;
  }
  TRACE_CPUPROFILER_EVENT_SCOPE_STR("TeleopNode: Communication pump");
  OnPublish();
  // Pump subscriber callbacks
  Communication->Pump();

  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
}

bool UTeleopNodeBaseComponent::OnCreatePubSubs() {
  return true;
}

void UTeleopNodeBaseComponent::OnPublish() {}

#if WITH_EDITOR
TArray<FString> UTeleopNodeBaseComponent::GetCommunicationClassOptions_EditorOnly() const {
  TArray<FString> Options;

  TArray<UClass*> DerivedClasses;
  GetDerivedClasses(UTeleopNodeImplBase::StaticClass(), DerivedClasses, true);

  for (UClass* Class : DerivedClasses) {
    if (Class && !Class->HasAnyClassFlags(CLASS_Abstract)) {
      Options.Add(Class->GetPathName());
    }
  }

  return Options;
}
#endif

//------------------------------------------------------------------------------
// OPERATOR COMPONENT IMPLEMENTATION
//------------------------------------------------------------------------------

UTeleopNodeOperatorComponent::UTeleopNodeOperatorComponent() {}

bool UTeleopNodeOperatorComponent::OnCreatePubSubs() {
  if (!Communication) {
    return false;
  }

  const FString& PubName = RobotNameForPublishers;
  const FString& SubName = RobotNameForSubscribers;

  // clang-format off

  // the Operator publishes arm pose targets
  CreatePublisher(Communication, PubName, Teleop::kArmPoseTargetTopic, ArmPose);
  // the Operator publishes arm end-effector stiffness targets (3x3 matrix)
  CreatePublisher(Communication, PubName, Teleop::kArmEndEffectorStiffnessTargetTopic, ArmEndEffectorStiffness);
  // the Operator publishes arm joint targets
  CreatePublisher(Communication, PubName, Teleop::kArmJointTargetsTopic, ArmJointPositions);
  // the Operator publishes hand joint targets
  CreatePublisher(Communication, PubName, Teleop::kHandJointTargetTopic, HandJointPositions);
  // the Operator publishes hand stiffness targets
  CreatePublisher(Communication, PubName, Teleop::kHandStiffnessTargetTopic, HandStiffness);

  // the Operator subscribes end-effector (wrist) pose
  CreateSubscriber(Communication, SubName, Teleop::kArmEndEffectorPoseStateTopic, ArmPose);
  // the Operator subscribes end-effector (wrist) stiffness
  CreateSubscriber(Communication, SubName, Teleop::kArmEndEffectorStiffnessStateTopic, ArmEndEffectorStiffness);
  // the Operator subscribes arm joint state (init data to home position)
  CreateSubscriber(Communication, SubName, Teleop::kArmJointStateTopic, ArmJointPositions);
  // the Operator subscribes hand joint state
  CreateSubscriber(Communication, SubName, Teleop::kHandJointStateTopic, HandJointPositions);
  // the Operator subscribes hand stiffness state
  CreateSubscriber(Communication, SubName, Teleop::kHandStiffnessStateTopic, HandStiffness);
  // the Operator subscribes hand sensor state
  CreateSubscriber(Communication, SubName, Teleop::kHandSensorStateTopic, HandSensors);

  // the Operator subscribes to joint velocities
  CreateSubscriber(Communication, SubName, Teleop::kArmJointVelocitiesTopic, ArmJointVelocities);
  CreateSubscriber(Communication, SubName, Teleop::kHandJointVelocitiesTopic, HandJointVelocities);

  // the Operator subscribes to joint efforts
  CreateSubscriber(Communication, SubName, Teleop::kArmJointEffortsTopic, ArmJointEfforts);
  CreateSubscriber(Communication, SubName, Teleop::kHandJointEffortsTopic, HandJointEfforts);
  CreateSubscriber(Communication, SubName, Teleop::kArmControlPolicyEffortsTopic, ArmJointControlPolicyEfforts);
  CreateSubscriber(Communication, SubName, Teleop::kHandControlPolicyEffortsTopic, HandJointControlPolicyEfforts);

  // clang-format on

  return true;
}

void UTeleopNodeOperatorComponent::OnPublish() {
  TRACE_CPUPROFILER_EVENT_SCOPE_STR("TeleopNode: Operator Publish");

  if (!Communication) {
    return;
  }

  if (bPublishArmPoseTarget) {
    Publish(Communication, ArmPose);
  }
  if (bPublishArmEndEffectorStiffnessTarget) {
    Publish(Communication, ArmEndEffectorStiffness);
  }
  if (bPublishArmJointTargets) {
    Publish(Communication, ArmJointPositions);
  }
  if (bPublishHandJointTargets) {
    Publish(Communication, HandJointPositions);
  }
  if (bPublishHandStiffnessTargets) {
    Publish(Communication, HandStiffness);
  }
}

// Arm pose
void UTeleopNodeOperatorComponent::SetArmPoseTarget(const FTransform& Target) {
  ArmPose.PubData.Mutate([&](TArray<FTransform>& Arr) {
    if (Arr.Num() > 0) {
      Arr[0] = Target;
    }
  });
  // Set after the data, so a concurrent reader that sees "live" is reading this target and not
  // the previous one.
  bArmPoseTargetLive = true;
}

bool UTeleopNodeOperatorComponent::GetArmPoseClampedTarget(FTransform& Pose) {
  return ArmPose.SubData.TryGetFirst(Pose);
}

// Arm joints
void UTeleopNodeOperatorComponent::SetArmJointTargets(const TArray<double>& Targets) {
  bArmJointTargetsLive = SetPubDataWithSizeCheck(
      Targets, ArmJointPositions, TEXT("UTeleopNodeOperatorComponent"), TEXT("SetArmJointTargets"));
}

bool UTeleopNodeOperatorComponent::GetArmJointStates(TArray<double>& State) {
  return ArmJointPositions.SubData.TryGet(State);
}

// Arm end-effector stiffness
void UTeleopNodeOperatorComponent::SetArmEndEffectorStiffnessTarget(const TArray<double>& Targets) {
  SetPubDataWithSizeCheck(
      Targets,
      ArmEndEffectorStiffness,
      TEXT("UTeleopNodeOperatorComponent"),
      TEXT("SetArmEndEffectorStiffnessTarget"));
}

bool UTeleopNodeOperatorComponent::GetArmEndEffectorStiffnessState(TArray<double>& State) {
  return ArmEndEffectorStiffness.SubData.TryGet(State);
}

// Arm velocities and efforts
bool UTeleopNodeOperatorComponent::GetArmJointVelocities(TArray<double>& Velocities) {
  return ArmJointVelocities.SubData.TryGet(Velocities);
}

bool UTeleopNodeOperatorComponent::GetArmJointEfforts(TArray<double>& Efforts) {
  return ArmJointEfforts.SubData.TryGet(Efforts);
}

bool UTeleopNodeOperatorComponent::GetArmJointControlPolicyEfforts(TArray<double>& Efforts) {
  return ArmJointControlPolicyEfforts.SubData.TryGet(Efforts);
}

// Hand joints
void UTeleopNodeOperatorComponent::SetHandJointTargets(const TArray<double>& Targets) {
  bHandJointTargetsLive = SetPubDataWithSizeCheck(
      Targets,
      HandJointPositions,
      TEXT("UTeleopNodeOperatorComponent"),
      TEXT("SetHandJointTargets"));
}

bool UTeleopNodeOperatorComponent::GetHandJointStates(TArray<double>& State) {
  return HandJointPositions.SubData.TryGet(State);
}

bool UTeleopNodeOperatorComponent::GetLastPublishedHandJointTargets(TArray<double>& Targets) const {
  // Liveness is re-read after the copy: an invalidation that lands mid-copy must lose, or the
  // caller would act on a target the operator no longer intends.
  return bHandJointTargetsLive && HandJointPositions.PubData.TryGet(Targets) &&
      bHandJointTargetsLive;
}

bool UTeleopNodeOperatorComponent::GetLastPublishedArmPoseTarget(FTransform& Target) const {
  return bArmPoseTargetLive && ArmPose.PubData.TryGetFirst(Target) && bArmPoseTargetLive;
}

bool UTeleopNodeOperatorComponent::GetLastPublishedArmJointTargets(TArray<double>& Targets) const {
  return bArmJointTargetsLive && ArmJointPositions.PubData.TryGet(Targets) && bArmJointTargetsLive;
}

void UTeleopNodeOperatorComponent::InvalidateLastPublishedCommandTargets() {
  // Every pose-derived command channel, so none of them survives a recalibration. Stiffness is
  // deliberately left alone: it is not derived from the operator's pose, so a new clutch origin
  // does not make it stale.
  InvalidateArmPoseTarget();
  InvalidateArmJointTargets();
  InvalidateHandJointTargets();
}

void UTeleopNodeOperatorComponent::InvalidateArmPoseTarget() {
  bArmPoseTargetLive = false;
}

void UTeleopNodeOperatorComponent::InvalidateArmJointTargets() {
  bArmJointTargetsLive = false;
}

void UTeleopNodeOperatorComponent::InvalidateHandJointTargets() {
  bHandJointTargetsLive = false;
}

// Hand stiffness
void UTeleopNodeOperatorComponent::SetHandStiffnessTargets(const TArray<double>& Targets) {
  SetPubDataWithSizeCheck(
      Targets,
      HandStiffness,
      TEXT("UTeleopNodeOperatorComponent"),
      TEXT("SetHandStiffnessTargets"));
}

bool UTeleopNodeOperatorComponent::GetHandStiffnessStates(TArray<double>& State) {
  return HandStiffness.SubData.TryGet(State);
}

// Hand velocities and efforts
bool UTeleopNodeOperatorComponent::GetHandJointVelocities(TArray<double>& Velocities) {
  return HandJointVelocities.SubData.TryGet(Velocities);
}

bool UTeleopNodeOperatorComponent::GetHandJointEfforts(TArray<double>& Efforts) {
  return HandJointEfforts.SubData.TryGet(Efforts);
}

bool UTeleopNodeOperatorComponent::GetHandJointControlPolicyEfforts(TArray<double>& Efforts) {
  return HandJointControlPolicyEfforts.SubData.TryGet(Efforts);
}

// Hand sensors
bool UTeleopNodeOperatorComponent::GetHandSensorStates_Pressure(TArray<double>& State) {
  return HandSensors.SubData.TryGet(State);
}

bool UTeleopNodeOperatorComponent::GetHandSensorStates_Force(TArray<FVector>& State) {
  TArray<double> RawData;
  if (!HandSensors.SubData.TryGet(RawData)) {
    return false;
  }
  State.SetNumUninitialized(RawData.Num() / 3);
  for (int i = 0; i < State.Num(); ++i) {
    State[i] = FVector(RawData[i * 3], RawData[i * 3 + 1], RawData[i * 3 + 2]);
  }
  return true;
}

//------------------------------------------------------------------------------
// ROBOT COMPONENT IMPLEMENTATION
//------------------------------------------------------------------------------

UTeleopNodeRobotComponent::UTeleopNodeRobotComponent() {}

bool UTeleopNodeRobotComponent::OnCreatePubSubs() {
  if (!Communication) {
    return false;
  }

  const FString& PubName = RobotNameForPublishers;
  const FString& SubName = RobotNameForSubscribers;

  // clang-format off

  // the Robot publishes end-effector (wrist) pose
  CreatePublisher(Communication, PubName, Teleop::kArmEndEffectorPoseStateTopic, ArmPose);
  CreatePublisher(Communication, PubName, Teleop::kArmEndEffectorStiffnessStateTopic, ArmEndEffectorStiffness);
  // the Robot publishes arm joint state
  CreatePublisher(Communication, PubName, Teleop::kArmJointStateTopic, ArmJointPositions);
  // the Robot publishes hand joint state
  CreatePublisher(Communication, PubName, Teleop::kHandJointStateTopic, HandJointPositions);
  CreatePublisher(Communication, PubName, Teleop::kHandStiffnessStateTopic, HandStiffness);
  // the Robot publishes hand sensor state
  CreatePublisher(Communication, PubName, Teleop::kHandSensorStateTopic, HandSensors);

  // the Robot publishes joint velocities
  CreatePublisher(Communication, PubName, Teleop::kArmJointVelocitiesTopic, ArmJointVelocities);
  CreatePublisher(Communication, PubName, Teleop::kHandJointVelocitiesTopic, HandJointVelocities);

  // the Robot publishes joint efforts
  CreatePublisher(Communication, PubName, Teleop::kArmJointEffortsTopic, ArmJointEfforts);
  CreatePublisher(Communication, PubName, Teleop::kHandJointEffortsTopic, HandJointEfforts);
  CreatePublisher(Communication, PubName, Teleop::kArmControlPolicyEffortsTopic, ArmJointControlPolicyEfforts);
  CreatePublisher(Communication, PubName, Teleop::kHandControlPolicyEffortsTopic, HandJointControlPolicyEfforts);

  // the Robot subscribes arm pose targets
  CreateSubscriber(Communication, SubName, Teleop::kArmPoseTargetTopic, ArmPose);
  // the Robot subscribes arm joint targets
  CreateSubscriber(Communication, SubName, Teleop::kArmJointTargetsTopic, ArmJointPositions);
  // the Robot subscribes arm stiffness targets
  CreateSubscriber(Communication, SubName, Teleop::kArmEndEffectorStiffnessTargetTopic, ArmEndEffectorStiffness);
  // the Robot subscribes hand joint targets
  CreateSubscriber(Communication, SubName, Teleop::kHandJointTargetTopic, HandJointPositions);
  // the Robot subscribes hand stiffness targets
  CreateSubscriber(Communication, SubName, Teleop::kHandStiffnessTargetTopic, HandStiffness);

  // clang-format on

  return true;
}

void UTeleopNodeRobotComponent::OnPublish() {
  TRACE_CPUPROFILER_EVENT_SCOPE_STR("TeleopNode: Robot Publish");
  if (!Communication) {
    return;
  }

  if (bPublishArmPoseState) {
    Publish(Communication, ArmPose);
  }
  if (bPublishArmEndEffectorStiffnessState) {
    Publish(Communication, ArmEndEffectorStiffness);
  }
  if (bPublishArmJointStates) {
    Publish(Communication, ArmJointPositions);
  }
  if (bPublishHandJointStates) {
    Publish(Communication, HandJointPositions);
  }
  if (bPublishHandStiffnessStates) {
    Publish(Communication, HandStiffness);
  }
  if (bPublishHandSensorStates) {
    Publish(Communication, HandSensors);
  }
  if (bPublishArmJointVelocitiesAndEfforts) {
    Publish(Communication, ArmJointVelocities);
    Publish(Communication, ArmJointEfforts);
  }
  if (bPublishHandJointVelocitiesAndEfforts) {
    Publish(Communication, HandJointVelocities);
    Publish(Communication, HandJointEfforts);
  }
  if (bPublishArmJointControlPolicyEfforts) {
    Publish(Communication, ArmJointControlPolicyEfforts);
  }
  if (bPublishHandJointControlPolicyEfforts) {
    Publish(Communication, HandJointControlPolicyEfforts);
  }
}

// Arm pose
bool UTeleopNodeRobotComponent::GetArmPoseTarget(FTransform& Target) {
  return ArmPose.SubData.TryGetFirst(Target);
}

void UTeleopNodeRobotComponent::SetArmPoseClampedTarget(const FTransform& Pose) {
  ArmPose.PubData.Mutate([&](TArray<FTransform>& Arr) {
    if (Arr.Num() > 0) {
      Arr[0] = Pose;
    }
  });
}

// Arm joints
bool UTeleopNodeRobotComponent::GetArmJointTargets(TArray<double>& Targets) {
  return ArmJointPositions.SubData.TryGet(Targets);
}

void UTeleopNodeRobotComponent::SetArmJointStates(const TArray<double>& State) {
  SetPubDataWithSizeCheck(
      State, ArmJointPositions, TEXT("UTeleopNodeRobotComponent"), TEXT("SetArmJointStates"));
}

// Arm end-effector stiffness
bool UTeleopNodeRobotComponent::GetArmEndEffectorStiffnessTarget(TArray<double>& Targets) {
  return ArmEndEffectorStiffness.SubData.TryGet(Targets);
}

void UTeleopNodeRobotComponent::SetArmEndEffectorStiffnessState(const TArray<double>& State) {
  SetPubDataWithSizeCheck(
      State,
      ArmEndEffectorStiffness,
      TEXT("UTeleopNodeRobotComponent"),
      TEXT("SetArmEndEffectorStiffnessState"));
}

// Arm velocities and efforts
void UTeleopNodeRobotComponent::SetArmJointVelocities(const TArray<double>& Velocities) {
  SetPubDataWithSizeCheck(
      Velocities,
      ArmJointVelocities,
      TEXT("UTeleopNodeRobotComponent"),
      TEXT("SetArmJointVelocities"));
}

void UTeleopNodeRobotComponent::SetArmJointEfforts(const TArray<double>& Efforts) {
  SetPubDataWithSizeCheck(
      Efforts, ArmJointEfforts, TEXT("UTeleopNodeRobotComponent"), TEXT("SetArmJointEfforts"));
}

void UTeleopNodeRobotComponent::SetArmJointControlPolicyEfforts(const TArray<double>& Efforts) {
  SetPubDataWithSizeCheck(
      Efforts,
      ArmJointControlPolicyEfforts,
      TEXT("UTeleopNodeRobotComponent"),
      TEXT("SetArmJointControlPolicyEfforts"));
}

// Hand joints
bool UTeleopNodeRobotComponent::GetHandJointTargets(TArray<double>& Targets) {
  return HandJointPositions.SubData.TryGet(Targets);
}

void UTeleopNodeRobotComponent::SetHandJointStates(const TArray<double>& State) {
  SetPubDataWithSizeCheck(
      State, HandJointPositions, TEXT("UTeleopNodeRobotComponent"), TEXT("SetHandJointStates"));
}

// Hand stiffness
bool UTeleopNodeRobotComponent::GetHandStiffnessTargets(TArray<double>& Targets) {
  return HandStiffness.SubData.TryGet(Targets);
}

void UTeleopNodeRobotComponent::SetHandStiffnessStates(const TArray<double>& State) {
  SetPubDataWithSizeCheck(
      State, HandStiffness, TEXT("UTeleopNodeRobotComponent"), TEXT("SetHandStiffnessStates"));
}

// Hand velocities and efforts
void UTeleopNodeRobotComponent::SetHandJointVelocities(const TArray<double>& Velocities) {
  SetPubDataWithSizeCheck(
      Velocities,
      HandJointVelocities,
      TEXT("UTeleopNodeRobotComponent"),
      TEXT("SetHandJointVelocities"));
}

void UTeleopNodeRobotComponent::SetHandJointEfforts(const TArray<double>& Efforts) {
  SetPubDataWithSizeCheck(
      Efforts, HandJointEfforts, TEXT("UTeleopNodeRobotComponent"), TEXT("SetHandJointEfforts"));
}

void UTeleopNodeRobotComponent::SetHandJointControlPolicyEfforts(const TArray<double>& Efforts) {
  SetPubDataWithSizeCheck(
      Efforts,
      HandJointControlPolicyEfforts,
      TEXT("UTeleopNodeRobotComponent"),
      TEXT("SetHandJointControlPolicyEfforts"));
}

// Hand sensors
void UTeleopNodeRobotComponent::SetHandSensorStates_Pressure(const TArray<double>& State) {
  SetPubDataWithSizeCheck(
      State, HandSensors, TEXT("UTeleopNodeRobotComponent"), TEXT("SetHandSensorStates_Pressure"));
}

void UTeleopNodeRobotComponent::SetHandSensorStates_Force(const TArray<FVector>& State) {
  HandSensors.PubData.Mutate([&](TArray<double>& Arr) {
    Arr.SetNumUninitialized(State.Num() * 3);
    for (int i = 0; i < State.Num(); ++i) {
      Arr[i * 3] = State[i].X;
      Arr[i * 3 + 1] = State[i].Y;
      Arr[i * 3 + 2] = State[i].Z;
    }
  });
}
