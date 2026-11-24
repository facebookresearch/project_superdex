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

#include "Components/SceneComponent.h"
#include "CoreMinimal.h"
#include "SuperDexBotIKConfig.h"

#include "SuperDexBotIKComponent.generated.h"

class USuperDexIKSolverObject;

/** Space a bot IK goal transform is given in: this component's space or world space. */
UENUM(BlueprintType)
enum class ESuperDexBotIKGoalSpace : uint8 {
  Component,
  World,
};

/**
 * Solves inverse kinematics for a bot with a USuperDexBotIKConfig, with goals that can be updated
 * at runtime. Solver calls are game-thread-only.
 */
UCLASS(ClassGroup = (SuperDex), meta = (BlueprintSpawnableComponent))
class SUPERDEXROBOTICS_API USuperDexBotIKComponent : public USceneComponent {
  GENERATED_BODY()

 public:
  USuperDexBotIKComponent();

  /** The IK config the solver is created from. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex Bot IK")
  TObjectPtr<USuperDexBotIKConfig> Config;

  /**
   * Creates the solver from Config and resets the goals to Config's goals. Called in BeginPlay.
   *
   * @return False off the game thread, or if Config is missing, invalid or its solver cannot be
   * created.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot IK")
  bool InitializeSolver();

  /**
   * Restores every goal to the authored goals captured by InitializeSolver, in component space.
   *
   * @return False off the game thread or without a solver, otherwise true.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot IK")
  bool ResetGoalsToConfig();

  /**
   * Names are case-sensitive. Invalid updates preserve the previous goal.
   *
   * @param Name Goal name.
   * @param Transform Goal transform; its scale is ignored, and its rotation must be normalized.
   * @param Space Space Transform is given in.
   * @return False if the goal is unknown or the transform is invalid, otherwise true.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot IK")
  bool
  SetGoalTransform(const FString& Name, const FTransform& Transform, ESuperDexBotIKGoalSpace Space);

  /**
   * Enables or disables a goal. Names are case-sensitive.
   *
   * @param Name Goal name.
   * @param bEnabled Whether the goal is enabled.
   * @return False if the goal is unknown, otherwise true.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot IK")
  bool SetGoalEnabled(const FString& Name, bool bEnabled);

  /**
   * Stiffnesses are in native N/m and N m/rad units.
   *
   * @param Name Goal name.
   * @param Position Position stiffness [N/m]; must be finite and non-negative.
   * @param Rotation Rotation stiffness [N m/rad]; must be finite and non-negative.
   * @return False if the goal is unknown or a stiffness is invalid, otherwise true.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot IK")
  bool SetGoalStiffness(const FString& Name, double Position, double Rotation);

  /**
   * Returns solve success; bOutGoalsReached separately reports convergence.
   *
   * @param bOutGoalsReached Set to whether the goals were reached; false whenever the call returns
   *   false.
   * @return False off the game thread, without a solver, with no active goal or if the solve fails;
   *   otherwise true.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot IK")
  bool SolveIK(bool& bOutGoalsReached);

  /**
   * Native joint-only positions in metres/radians, excluding root DOFs.
   *
   * @return The solved joint positions, or empty off the game thread.
   */
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot IK")
  TArray<double> GetSolvedJointPositions() const;

  /**
   * World-space transforms in the bot's link order. IK ignores component and goal scale.
   *
   * @return The solved link world transforms, or empty off the game thread.
   */
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot IK")
  TArray<FTransform> GetSolvedLinkTransforms() const;

 protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
  virtual void BeginDestroy() override;

 private:
  void ReleaseSolver();
  void UpdateResults();
  int32 FindGoal(const FString& Name) const;

  UPROPERTY(Transient)
  TObjectPtr<USuperDexIKSolverObject> Solver;

  UPROPERTY(Transient)
  TArray<FSuperDexBotIKGoalConfig> RuntimeGoals;

  UPROPERTY(Transient)
  TArray<FSuperDexBotIKGoalConfig> AuthoredGoals;

  TArray<ESuperDexBotIKGoalSpace> GoalSpaces;
  TArray<int32> GoalLinks;
  TArray<double> JointPositions;
  TArray<FTransform> WorldFromLinks;
};
