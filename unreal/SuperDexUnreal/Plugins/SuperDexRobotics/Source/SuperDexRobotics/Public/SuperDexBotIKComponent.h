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

UENUM(BlueprintType)
enum class ESuperDexBotIKGoalSpace : uint8 {
  Component,
  World,
};

UCLASS(ClassGroup = (SuperDex), meta = (BlueprintSpawnableComponent))
class SUPERDEXROBOTICS_API USuperDexBotIKComponent : public USceneComponent {
  GENERATED_BODY()

 public:
  USuperDexBotIKComponent();

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex Bot IK")
  TObjectPtr<USuperDexBotIKConfig> Config;

  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot IK")
  bool InitializeSolver();

  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot IK")
  bool ResetGoalsToConfig();

  /** Names are case-sensitive. Invalid updates preserve the previous goal. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot IK")
  bool
  SetGoalTransform(const FString& Name, const FTransform& Transform, ESuperDexBotIKGoalSpace Space);

  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot IK")
  bool SetGoalEnabled(const FString& Name, bool bEnabled);

  /** Stiffnesses are in native N/m and N m/rad units. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot IK")
  bool SetGoalStiffness(const FString& Name, double Position, double Rotation);

  /** Returns solve success; OutGoalsReached separately reports convergence. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot IK")
  bool SolveIK(bool& OutGoalsReached);

  /** Native joint-only positions in metres/radians, excluding root DOFs. */
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot IK")
  TArray<double> GetSolvedJointPositions() const;

  /** World-space transforms in the bot's link order. IK ignores component and goal scale. */
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
