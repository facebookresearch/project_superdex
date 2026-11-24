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

/**
 * Base of the teleop node components: owns the pub/sub channels between an operator and a robot,
 * created on BeginPlay through the configured communication backend, and publishes each tick.
 */
UCLASS(Abstract)
class SUPERDEXTELEOP_API UTeleopNodeBaseComponent : public UActorComponent {
  GENERATED_BODY()

 public:
  /**
   * The UTeleopNodeImplBase class that carries this node's messages. When empty, the node uses the
   * project's Default Communication Class Name (Project Settings > Plugins > SuperDex Teleop), or
   * UTeleopNodeImplDirect if that is not set.
   */
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "TeleopNode|Configuration",
      meta = (GetOptions = "GetCommunicationClassOptions_EditorOnly"))
  FString CommunicationClassName;

  // !! IMPORTANT: Keep UPROPERTYs in sync with the Linux stub (see note above) !!
  // Otherwise save in editor on Linux will silently drop properties

  /** The number of arm joints to allocate storage for. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Configuration")
  int NumArmJoints = Teleop::NumDefaultArmJoints;

  /** The number of hand joints to allocate storage for. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Configuration")
  int NumHandJoints = Teleop::NumDefaultHandJoints;

  /** The number of hand sensors to allocate storage for. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Configuration")
  int NumHandSensors = Teleop::NumDefaultHandSensors;

  /**
   * If true, hand sensors are 3D force sensors instead of 1D pressure sensors,
   * and Get/SetHandSensorStates_Force should be used to read/write states.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Configuration")
  bool bHandSensorsMeasureForce = false;

  /**
   * RobotName is usually either 'left' or 'right' for bimanual setups, can be blank
   * For example, result will either be /teleop/my_robot_name/arm_joint_states
   * ... or /teleop/arm_joint_states
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Configuration")
  FString RobotNameForPublishers = "";

  /** Robot name inserted into the subscribed topics, like RobotNameForPublishers. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Configuration")
  FString RobotNameForSubscribers = "";

  /** Publish in the pre-physics tick group. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Scheduling")
  bool bTickOnPrePhysics = true;

  /** Publish in the post-update-work tick group. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Scheduling")
  bool bTickOnPostUpdateWork = false;

  /** Publishes the enabled channels immediately instead of waiting for the next tick. */
  UFUNCTION(BlueprintCallable, Category = "Teleop")
  void PublishNow() {
    OnPublish();
  }

 public:
  /**
   * Creates the pre-physics and post-update-work tick functions disabled; BeginPlay enables them
   * from bTickOnPrePhysics and bTickOnPostUpdateWork.
   */
  UTeleopNodeBaseComponent();

  virtual void TickComponent(
      float DeltaTime,
      ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;

 protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
  virtual void RegisterComponentTickFunctions(bool bRegister) override;

  /** Override in subclasses to create publishers and subscribers */
  virtual bool OnCreatePubSubs();

  /** Override in subclasses to publish data each tick */
  virtual void OnPublish();

  /** Check if the node is properly initialized */
  bool IsInitialized() const {
    return bInitialized;
  }

  /** Get the communication backend */
  UTeleopNodeImplBase* GetCommunication() const {
    return Communication;
  }

#if WITH_EDITOR
 public:
  /**
   * Lists the path names of the non-abstract UTeleopNodeImplBase classes, for the
   * CommunicationClassName picker.
   */
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

  /** Node name for logging */
  FString NodeName;

  /** Communication backend instance */
  UPROPERTY()
  TObjectPtr<UTeleopNodeImplBase> Communication;

 private:
  FActorComponentTickFunction SecondaryComponentTick;
  bool bInitialized = false;
};

//------------------------------------------------------------------------------
// OPERATOR SIDE COMPONENT
//------------------------------------------------------------------------------

/** Operator side of a teleop link: publishes targets and receives the robot's states. */
UCLASS(
    ClassGroup = (Custom),
    meta = (BlueprintSpawnableComponent),
    AutoExpandCategories = ("TeleopNode|Scheduling", "TeleopNode|Operator"))
class SUPERDEXTELEOP_API UTeleopNodeOperatorComponent : public UTeleopNodeBaseComponent {
  GENERATED_BODY()

 public:
  /** Publish the arm pose target each tick. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Operator")
  bool bPublishArmPoseTarget = false;

  /** Publish the arm end-effector stiffness target each tick. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Operator")
  bool bPublishArmEndEffectorStiffnessTarget = false;

  /** Publish the arm joint targets each tick. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Operator")
  bool bPublishArmJointTargets = false;

  /** Publish the hand joint targets each tick. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Operator")
  bool bPublishHandJointTargets = false;

  /** Publish the hand stiffness targets each tick. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Operator")
  bool bPublishHandStiffnessTargets = false;

 public:
  // Arm pose
  /**
   * Sets the arm pose target to publish and marks it live for GetLastPublishedArmPoseTarget.
   *
   * @param Target The arm pose target.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  void SetArmPoseTarget(const FTransform& Target);

  /**
   * Gets the last received end-effector (wrist) pose state from the robot.
   *
   * @param Pose Set to the received pose; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetArmPoseClampedTarget(FTransform& Pose);

  // Arm joints
  /**
   * Sets the arm joint targets to publish. Ignored without a publisher; a size other than
   * NumArmJoints logs a warning and only the overlapping values are set.
   *
   * @param Targets The values to publish.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  void SetArmJointTargets(const TArray<double>& Targets);

  /**
   * Gets the last received arm joint states.
   *
   * @param State Set to the received values; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetArmJointStates(TArray<double>& State);

  // Arm end-effector stiffness
  /**
   * Sets the arm end-effector stiffness target (3x3 matrix) to publish. Ignored without a
   * publisher; a size other than 9 logs a warning and only the overlapping values are set.
   *
   * @param Targets The values to publish.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  void SetArmEndEffectorStiffnessTarget(const TArray<double>& Targets);

  /**
   * Gets the last received arm end-effector stiffness state.
   *
   * @param State Set to the received values; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetArmEndEffectorStiffnessState(TArray<double>& State);

  // Arm velocities and efforts
  /**
   * Gets the last received arm joint velocities.
   *
   * @param Velocities Set to the received values; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetArmJointVelocities(TArray<double>& Velocities);

  /**
   * Gets the last received arm joint efforts.
   *
   * @param Efforts Set to the received values; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetArmJointEfforts(TArray<double>& Efforts);

  /**
   * Gets the last received arm joint control policy efforts.
   *
   * @param Efforts Set to the received values; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetArmJointControlPolicyEfforts(TArray<double>& Efforts);

  // Hand joints
  /**
   * Sets the hand joint targets to publish. Ignored without a publisher; a size other than
   * NumHandJoints logs a warning and only the overlapping values are set.
   *
   * @param Targets The values to publish.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  void SetHandJointTargets(const TArray<double>& Targets);

  /**
   * Gets the last received hand joint states.
   *
   * @param State Set to the received values; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetHandJointStates(TArray<double>& State);

  /**
   * The targets this node last published, as opposed to what it has received back over ROS. A
   * consumer that drives simulated hardware in-process (the bot-scene teleop bridge) reads these
   * so it is not gated on the ROS round trip having completed. False once the channel has been
   * invalidated, so a command from before the operator lost tracking is not read back as live.
   *
   * @param Targets Set to the last published hand joint targets when they are live.
   * @return False if no live hand joint targets have been published.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetLastPublishedHandJointTargets(TArray<double>& Targets) const;

  /**
   * Like GetLastPublishedHandJointTargets, for the arm pose target.
   *
   * @param Target Set to the last published arm pose target when it is live.
   * @return False if no live arm pose target has been published.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetLastPublishedArmPoseTarget(FTransform& Target) const;

  /**
   * Like GetLastPublishedHandJointTargets, for the arm joint targets.
   *
   * @param Targets Set to the last published arm joint targets when they are live.
   * @return False if no live arm joint targets have been published.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetLastPublishedArmJointTargets(TArray<double>& Targets) const;

  /** Invalidate commands published before a discontinuous retargeter recalibration. */
  void InvalidateLastPublishedCommandTargets();

  /**
   * Mark a channel's last command stale for GetLastPublished* until its next Set*Target. These
   * affect the in-process readback only: what goes out on the wire is unchanged, so a subscribed
   * robot keeps receiving the last good command rather than seeing the topic fall silent.
   */
  void InvalidateArmPoseTarget();
  void InvalidateArmJointTargets();
  void InvalidateHandJointTargets();

  // Hand stiffness
  /**
   * Sets the hand stiffness targets to publish. Ignored without a publisher; a size other than
   * NumHandJoints logs a warning and only the overlapping values are set.
   *
   * @param Targets The values to publish.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  void SetHandStiffnessTargets(const TArray<double>& Targets);

  /**
   * Gets the last received hand stiffness states.
   *
   * @param State Set to the received values; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetHandStiffnessStates(TArray<double>& State);

  // Hand velocities and efforts
  /**
   * Gets the last received hand joint velocities.
   *
   * @param Velocities Set to the received values; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetHandJointVelocities(TArray<double>& Velocities);

  /**
   * Gets the last received hand joint efforts.
   *
   * @param Efforts Set to the received values; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetHandJointEfforts(TArray<double>& Efforts);

  /**
   * Gets the last received hand joint control policy efforts.
   *
   * @param Efforts Set to the received values; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetHandJointControlPolicyEfforts(TArray<double>& Efforts);

  // Hand sensors
  /**
   * Gets the last received hand sensor states as 1D pressures.
   *
   * @param State Set to the received values; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetHandSensorStates_Pressure(TArray<double>& State);

  /**
   * Gets the last received hand sensor states as 3D forces (see bHandSensorsMeasureForce).
   *
   * @param State Set to one force per sensor; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Operator")
  bool GetHandSensorStates_Force(TArray<FVector>& State);

 public:
  UTeleopNodeOperatorComponent();

 protected:
  virtual bool OnCreatePubSubs() override;
  virtual void OnPublish() override;

 private:
  /**
   * Whether each command channel's PubData is a live operator intent rather than the last value
   * from before tracking was lost. Kept here instead of on the channel's own set-flag because that
   * one also gates publishing: clearing it would mute the ROS topic, which only the in-process
   * readback wants. Read off the game thread by the bot-scene bridge's Mochi pre-step.
   */
  std::atomic<bool> bArmPoseTargetLive{false};
  std::atomic<bool> bArmJointTargetsLive{false};
  std::atomic<bool> bHandJointTargetsLive{false};
};

//------------------------------------------------------------------------------
// ROBOT SIDE COMPONENT
// IMPORTANT: Keep UPROPERTYs in sync with the Linux stub (see note above).
//------------------------------------------------------------------------------

/**
 * Robot side of a teleop link: receives the operator's targets and publishes the robot's states.
 */
UCLASS(
    ClassGroup = (Custom),
    meta = (BlueprintSpawnableComponent),
    AutoExpandCategories = ("TeleopNode|Scheduling", "TeleopNode|Robot"))
class SUPERDEXTELEOP_API UTeleopNodeRobotComponent : public UTeleopNodeBaseComponent {
  GENERATED_BODY()

 public:
  // !! IMPORTANT: Keep UPROPERTYs in sync with the Linux stub (see note above) !!
  // Otherwise save in editor on Linux will silently drop properties

  /** Publish the end-effector (wrist) pose state each tick. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishArmPoseState = false;

  /** Publish the arm end-effector stiffness state each tick. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishArmEndEffectorStiffnessState = false;

  /** Publish the arm joint states each tick. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishArmJointStates = false;

  /** Publish the hand joint states each tick. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishHandJointStates = false;

  /** Publish the hand stiffness states each tick. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishHandStiffnessStates = false;

  /** Publish the hand sensor states each tick. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishHandSensorStates = false;

  /** Publish the arm joint velocities and efforts each tick. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishArmJointVelocitiesAndEfforts = false;

  /** Publish the hand joint velocities and efforts each tick. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishHandJointVelocitiesAndEfforts = false;

  /** Publish the arm joint control policy efforts each tick. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishArmJointControlPolicyEfforts = false;

  /** Publish the hand joint control policy efforts each tick. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopNode|Robot")
  bool bPublishHandJointControlPolicyEfforts = false;

 public:
  // Arm pose
  /**
   * Gets the last received arm pose target from the operator.
   *
   * @param Target Set to the received target; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  bool GetArmPoseTarget(FTransform& Target);

  /**
   * Sets the end-effector (wrist) pose state to publish.
   *
   * @param Pose The value to publish.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetArmPoseClampedTarget(const FTransform& Pose);

  // Arm joints
  /**
   * Gets the last received arm joint targets.
   *
   * @param Targets Set to the received values; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  bool GetArmJointTargets(TArray<double>& Targets);

  /**
   * Sets the arm joint states to publish. Ignored without a publisher; a size other than
   * NumArmJoints logs a warning and only the overlapping values are set.
   *
   * @param State The values to publish.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetArmJointStates(const TArray<double>& State);

  // Arm end-effector stiffness
  /**
   * Gets the last received arm end-effector stiffness target.
   *
   * @param Targets Set to the received values; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  bool GetArmEndEffectorStiffnessTarget(TArray<double>& Targets);

  /**
   * Sets the arm end-effector stiffness state to publish. Ignored without a publisher; a size other
   * than 9 logs a warning and only the overlapping values are set.
   *
   * @param State The values to publish.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetArmEndEffectorStiffnessState(const TArray<double>& State);

  // Arm velocities and efforts
  /**
   * Sets the arm joint velocities to publish. Ignored without a publisher; a size other than
   * NumArmJoints logs a warning and only the overlapping values are set.
   *
   * @param Velocities The values to publish.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetArmJointVelocities(const TArray<double>& Velocities);

  /**
   * Sets the arm joint efforts to publish. Ignored without a publisher; a size other than
   * NumArmJoints logs a warning and only the overlapping values are set.
   *
   * @param Efforts The values to publish.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetArmJointEfforts(const TArray<double>& Efforts);

  /**
   * Sets the arm joint control policy efforts to publish. Ignored without a publisher; a size other
   * than NumArmJoints logs a warning and only the overlapping values are set.
   *
   * @param Efforts The values to publish.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetArmJointControlPolicyEfforts(const TArray<double>& Efforts);

  // Hand joints
  /**
   * Gets the last received hand joint targets.
   *
   * @param Targets Set to the received values; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  bool GetHandJointTargets(TArray<double>& Targets);

  /**
   * Sets the hand joint states to publish. Ignored without a publisher; a size other than
   * NumHandJoints logs a warning and only the overlapping values are set.
   *
   * @param State The values to publish.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetHandJointStates(const TArray<double>& State);

  // Hand stiffness
  /**
   * Gets the last received hand stiffness targets.
   *
   * @param Targets Set to the received values; unchanged until a message has been received.
   * @return False until a message has been received.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  bool GetHandStiffnessTargets(TArray<double>& Targets);

  /**
   * Sets the hand stiffness states to publish. Ignored without a publisher; a size other than
   * NumHandJoints logs a warning and only the overlapping values are set.
   *
   * @param State The values to publish.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetHandStiffnessStates(const TArray<double>& State);

  // Hand velocities and efforts
  /**
   * Sets the hand joint velocities to publish. Ignored without a publisher; a size other than
   * NumHandJoints logs a warning and only the overlapping values are set.
   *
   * @param Velocities The values to publish.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetHandJointVelocities(const TArray<double>& Velocities);

  /**
   * Sets the hand joint efforts to publish. Ignored without a publisher; a size other than
   * NumHandJoints logs a warning and only the overlapping values are set.
   *
   * @param Efforts The values to publish.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetHandJointEfforts(const TArray<double>& Efforts);

  /**
   * Sets the hand joint control policy efforts to publish. Ignored without a publisher; a size
   * other than NumHandJoints logs a warning and only the overlapping values are set.
   *
   * @param Efforts The values to publish.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetHandJointControlPolicyEfforts(const TArray<double>& Efforts);

  // Hand sensors
  /**
   * Sets the hand sensor states as 1D pressures to publish. Ignored without a publisher; a size
   * other than NumHandSensors logs a warning and only the overlapping values are set.
   *
   * @param State The values to publish.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetHandSensorStates_Pressure(const TArray<double>& State);

  /**
   * Sets the hand sensor states to publish as 3D forces (see bHandSensorsMeasureForce).
   *
   * @param State One force per sensor.
   */
  UFUNCTION(BlueprintCallable, Category = "TeleopNode|Robot")
  void SetHandSensorStates_Force(const TArray<FVector>& State);

 public:
  UTeleopNodeRobotComponent();

 protected:
  virtual bool OnCreatePubSubs() override;
  virtual void OnPublish() override;
};
