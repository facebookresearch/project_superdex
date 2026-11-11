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

#include "Components/ActorComponent.h"
#include "CoreMinimal.h"
#include "TeleopTypes.h"

#include <atomic>

#include "TeleopNode.generated.h"

class UTeleopNodeImplBase;

//------------------------------------------------------------------------------
// BASE COMPONENT
//
// IMPORTANT: If you add, remove, or rename any UPROPERTY in
// UTeleopNodeBaseComponent or UTeleopNodeRobotComponent, you MUST update the
// matching Linux stub provided by the consuming project, if it has one. The stub
// carries the same UPROPERTYs so that levels saved on Linux round-trip without
// losing Teleop data. Forgetting to update it will silently drop properties
// when a level is opened and saved on Linux.
//------------------------------------------------------------------------------

UCLASS(Abstract)
class SUPERDEXTELEOP_API UTeleopNodeBaseComponent : public UActorComponent {
  GENERATED_BODY()

 public:
  // The UTeleopNodeImplBase class that carries this node's messages. When empty, the node uses the
  // project's Default Communication Class Name (Project Settings > Plugins > SuperDex Teleop), or
  // UTeleopNodeImplDirect if that is not set.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "TeleopNode|Configuration",
      meta = (GetOptions = "GetCommunicationClassOptions_EditorOnly"))
  FString CommunicationClassName;

  // !! IMPORTANT: Keep UPROPERTYs in sync with the Linux stub (see note above) !!
  // Otherwise save in editor on Linux will silently drop properties

  // The number of arm joints to allocate storage for.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Configuration")
  int NumArmJoints = Teleop::kNumDefaultArmJoints;

  // The number of hand joints to allocate storage for.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Configuration")
  int NumHandJoints = Teleop::kNumDefaultHandJoints;

  // The number of hand sensors to allocate storage for.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Configuration")
  int NumHandSensors = Teleop::kNumDefaultHandSensors;

  // If true, hand sensors are 3D force sensors instead of 1D pressure sensors,
  // and Get/SetHandSensorStates_Force should be used to read/write states.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Configuration")
  bool bHandSensorsMeasureForce = false;

  // RobotName is usually either 'left' or 'right' for bimanual setups, can be blank
  // For example, result will either be /teleop/my_robot_name/arm_joint_states
  // ... or /teleop/arm_joint_states
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Configuration")
  FString RobotNameForPublishers = "";

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Configuration")
  FString RobotNameForSubscribers = "";

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Scheduling")
  bool bTickOnPrePhysics = true;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Scheduling")
  bool bTickOnPostUpdateWork = false;

  UFUNCTION(BlueprintCallable, Category = "Teleop")
  void PublishNow() {
    OnPublish();
  }

 public:
  UTeleopNodeBaseComponent();

  virtual void TickComponent(
      float DeltaTime,
      ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;

 protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
  virtual void RegisterComponentTickFunctions(bool bRegister) override;

  // Override in subclasses to create publishers and subscribers
  virtual bool OnCreatePubSubs();

  // Override in subclasses to publish data each tick
  virtual void OnPublish();

  // Check if the node is properly initialized
  bool IsInitialized() const {
    return bInitialized;
  }

  // Get the communication backend
  UTeleopNodeImplBase* GetCommunication() const {
    return Communication;
  }

#if WITH_EDITOR
 public:
  UFUNCTION(Category = "TeleopNode")
  TArray<FString> GetCommunicationClassOptions_EditorOnly() const;
#endif

 protected:
  // Data buffers for all pub/sub channels
  TTeleopPubSub<FTransform> ArmPose;
  TTeleopPubSub<double> ArmEndEffectorStiffness;
  TTeleopPubSub<double> ArmJointPositions;
  TTeleopPubSub<double> HandJointPositions;
  TTeleopPubSub<double> HandStiffness;
  TTeleopPubSub<double> ArmJointVelocities;
  TTeleopPubSub<double> HandJointVelocities;
  TTeleopPubSub<double> ArmJointEfforts;
  TTeleopPubSub<double> ArmJointControlPolicyEfforts;
  TTeleopPubSub<double> HandJointEfforts;
  TTeleopPubSub<double> HandJointControlPolicyEfforts;
  TTeleopPubSub<double> HandSensors;

  // Node name for logging
  FString NodeName;

  // Communication backend instance
  UPROPERTY()
  TObjectPtr<UTeleopNodeImplBase> Communication;

 private:
  FActorComponentTickFunction SecondaryComponentTick;
  bool bInitialized = false;
};

//------------------------------------------------------------------------------
// OPERATOR SIDE COMPONENT
//------------------------------------------------------------------------------

UCLASS(
    ClassGroup = (Custom),
    meta = (BlueprintSpawnableComponent),
    AutoExpandCategories = ("TeleopNode|Scheduling", "TeleopNode|Operator"))
class SUPERDEXTELEOP_API UTeleopNodeOperatorComponent : public UTeleopNodeBaseComponent {
  GENERATED_BODY()

 public:
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Operator")
  bool bPublishArmPoseTarget = false;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Operator")
  bool bPublishArmEndEffectorStiffnessTarget = false;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Operator")
  bool bPublishArmJointTargets = false;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Operator")
  bool bPublishHandJointTargets = false;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Operator")
  bool bPublishHandStiffnessTargets = false;

 public:
  // Arm pose
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  void SetArmPoseTarget(const FTransform& Target);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetArmPoseClampedTarget(FTransform& Pose);

  // Arm joints
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  void SetArmJointTargets(const TArray<double>& Targets);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetArmJointStates(TArray<double>& State);

  // Arm end-effector stiffness
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  void SetArmEndEffectorStiffnessTarget(const TArray<double>& Targets);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetArmEndEffectorStiffnessState(TArray<double>& State);

  // Arm velocities and efforts
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetArmJointVelocities(TArray<double>& Velocities);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetArmJointEfforts(TArray<double>& Efforts);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetArmJointControlPolicyEfforts(TArray<double>& Efforts);

  // Hand joints
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  void SetHandJointTargets(const TArray<double>& Targets);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetHandJointStates(TArray<double>& State);

  // The targets this node last published, as opposed to what it has received back over ROS. A
  // consumer that drives simulated hardware in-process (the bot-scene teleop bridge) reads these
  // so it is not gated on the ROS round trip having completed. False once the channel has been
  // invalidated, so a command from before the operator lost tracking is not read back as live.
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetLastPublishedHandJointTargets(TArray<double>& Targets) const;

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetLastPublishedArmPoseTarget(FTransform& Target) const;

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetLastPublishedArmJointTargets(TArray<double>& Targets) const;

  // Invalidate commands published before a discontinuous retargeter recalibration.
  void InvalidateLastPublishedCommandTargets();

  // Mark a channel's last command stale for GetLastPublished* until its next Set*Target. These
  // affect the in-process readback only: what goes out on the wire is unchanged, so a subscribed
  // robot keeps receiving the last good command rather than seeing the topic fall silent.
  void InvalidateArmPoseTarget();
  void InvalidateArmJointTargets();
  void InvalidateHandJointTargets();

  // Hand stiffness
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  void SetHandStiffnessTargets(const TArray<double>& Targets);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetHandStiffnessStates(TArray<double>& State);

  // Hand velocities and efforts
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetHandJointVelocities(TArray<double>& Velocities);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetHandJointEfforts(TArray<double>& Efforts);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetHandJointControlPolicyEfforts(TArray<double>& Efforts);

  // Hand sensors
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetHandSensorStates_Pressure(TArray<double>& State);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetHandSensorStates_Force(TArray<FVector>& State);

 public:
  UTeleopNodeOperatorComponent();

 protected:
  virtual bool OnCreatePubSubs() override;
  virtual void OnPublish() override;

 private:
  // Whether each command channel's PubData is a live operator intent rather than the last value
  // from before tracking was lost. Kept here instead of on the channel's own set-flag because that
  // one also gates publishing: clearing it would mute the ROS topic, which only the in-process
  // readback wants. Read off the game thread by the bot-scene bridge's Mochi pre-step.
  std::atomic<bool> bArmPoseTargetLive{false};
  std::atomic<bool> bArmJointTargetsLive{false};
  std::atomic<bool> bHandJointTargetsLive{false};
};

//------------------------------------------------------------------------------
// ROBOT SIDE COMPONENT
// IMPORTANT: Keep UPROPERTYs in sync with the Linux stub (see note above).
//------------------------------------------------------------------------------

UCLASS(
    ClassGroup = (Custom),
    meta = (BlueprintSpawnableComponent),
    AutoExpandCategories = ("TeleopNode|Scheduling", "TeleopNode|Robot"))
class SUPERDEXTELEOP_API UTeleopNodeRobotComponent : public UTeleopNodeBaseComponent {
  GENERATED_BODY()

 public:
  // !! IMPORTANT: Keep UPROPERTYs in sync with the Linux stub (see note above) !!
  // Otherwise save in editor on Linux will silently drop properties

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishArmPoseState = false;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishArmEndEffectorStiffnessState = false;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishArmJointStates = false;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishHandJointStates = false;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishHandStiffnessStates = false;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishHandSensorStates = false;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishArmJointVelocitiesAndEfforts = false;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishHandJointVelocitiesAndEfforts = false;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishArmJointControlPolicyEfforts = false;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishHandJointControlPolicyEfforts = false;

 public:
  // Arm pose
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  bool GetArmPoseTarget(FTransform& Target);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetArmPoseClampedTarget(const FTransform& Pose);

  // Arm joints
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  bool GetArmJointTargets(TArray<double>& Targets);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetArmJointStates(const TArray<double>& State);

  // Arm end-effector stiffness
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  bool GetArmEndEffectorStiffnessTarget(TArray<double>& Targets);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetArmEndEffectorStiffnessState(const TArray<double>& State);

  // Arm velocities and efforts
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetArmJointVelocities(const TArray<double>& Velocities);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetArmJointEfforts(const TArray<double>& Efforts);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetArmJointControlPolicyEfforts(const TArray<double>& Efforts);

  // Hand joints
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  bool GetHandJointTargets(TArray<double>& Targets);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetHandJointStates(const TArray<double>& State);

  // Hand stiffness
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  bool GetHandStiffnessTargets(TArray<double>& Targets);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetHandStiffnessStates(const TArray<double>& State);

  // Hand velocities and efforts
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetHandJointVelocities(const TArray<double>& Velocities);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetHandJointEfforts(const TArray<double>& Efforts);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetHandJointControlPolicyEfforts(const TArray<double>& Efforts);

  // Hand sensors
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetHandSensorStates_Pressure(const TArray<double>& State);

  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetHandSensorStates_Force(const TArray<FVector>& State);

 public:
  UTeleopNodeRobotComponent();

 protected:
  virtual bool OnCreatePubSubs() override;
  virtual void OnPublish() override;
};
