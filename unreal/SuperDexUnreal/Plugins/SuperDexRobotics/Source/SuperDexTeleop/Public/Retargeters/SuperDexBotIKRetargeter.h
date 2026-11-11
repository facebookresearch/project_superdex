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

#include <superdex_robotics/superdex_robotics.h>
#include "Retargeters/SuperDexRetargeter.h"
#include "SuperDexBotIKConfig.h"

#include "SuperDexBotIKRetargeter.generated.h"

class USuperDexIKSolverObject;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(
    FSuperDexBotIKConfigurationSolved,
    const TArray<double>&,
    JointPositions,
    bool,
    bGoalsReached);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(
    FSuperDexBotIKRetargeterStatusChanged,
    bool,
    bCurrentlyValid,
    const FString&,
    Diagnostic);

/** Input-bot pose -> fixed-output-base document-authored Bot IK, for any input bot with a Free/Hard
 * root and Hard/Revolute/Prismatic/Spherical joints. Input poses are parent-relative link
 * transforms in the input bot's link order (only their count is checked);
 * rotations suffice unless the input bot has prismatic joints. Each joint keeps what it can do of
 * its link's motion (see superdex::robotics::ComputePoseFromLinkTransforms). The root link's local
 * transform places the input bot's base. Root motion is carried via captured input/output
 * calibration, not a floating IK output base. Solves on the game thread. Physics consumers read
 * snapshots; they must not pull a solve. Configuration/model edits require InitializeRetargeter.
 * No recording is opened at runtime.
 */
UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class SUPERDEXTELEOP_API USuperDexBotIKRetargeter : public USuperDexRetargeter {
  GENERATED_BODY()

 public:
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bot IK")
  TObjectPtr<USuperDexBotIKConfig> Config;

  /** Opt-in hand-only bridge for existing arm + hand roboperators. Does not initialize transport.
   * The existing TeleopNode's publisher/clutch settings still govern external communication.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bot IK|Teleop")
  bool bPublishTeleopHandJointTargets = false;

  /** Use available hand feedback for the displayed pose, without changing solved command joints. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bot IK|Teleop")
  bool bSubscribeToTeleopHandJointPositions = false;

  /** Preserve legacy finger bias in hand-only prefab DOF order (never combined arm indices). */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bot IK|Teleop")
  double JointAngleMultiplier = 1.0;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bot IK|Teleop")
  TArray<int32> JointAngleMultiplierExcludeIndex;

  /** Native joint-only coordinates in radians/metres, in Config.OutputBot prefab DOF order.
   * An opted-in legacy bias is included; bGoalsReached describes the authored solve BEFORE bias,
   * not convergence of the calibrated command or feedback visualization.
   */
  UPROPERTY(BlueprintAssignable, Category = "Bot IK")
  FSuperDexBotIKConfigurationSolved OnJointConfigurationSolved;

  /** Tracking loss is reported here, without publishing an old solution as a fresh command. */
  UPROPERTY(BlueprintAssignable, Category = "Bot IK")
  FSuperDexBotIKRetargeterStatusChanged OnRetargetingStatusChanged;

  UFUNCTION(BlueprintCallable, Category = "Bot IK")
  bool InitializeRetargeter();

  UFUNCTION(BlueprintCallable, CallInEditor, Category = "Bot IK")
  bool ValidateConfiguration(FString& Diagnostic) const;

  /** Returns whether any solution exists; bCurrentlyValid distinguishes held data from tracking. */
  UFUNCTION(BlueprintPure, Category = "Bot IK")
  bool GetCachedJointConfiguration(TArray<double>& JointPositions, bool& bCurrentlyValid) const;

  UFUNCTION(BlueprintPure, Category = "Bot IK")
  FString GetRetargetingDiagnostic() const;

  virtual bool GetTransformedPose_Implementation(
      const FSuperDexPose& SourcePose,
      bool bSourceIsValid,
      FSuperDexPose& TransformedPose) override;
  virtual bool GetPose(FSuperDexPose& OutPose) override;
  virtual bool GetCachedPose(FSuperDexPose& OutPose) const override;

 protected:
  USuperDexBotIKRetargeter();
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
  virtual void OnComponentDestroyed(bool bDestroyingHierarchy) override;
#if WITH_EDITOR
  virtual void PostEditChangeProperty(FPropertyChangedEvent& Event) override;
  virtual EDataValidationResult IsDataValid(FDataValidationContext& Context) const override;
#endif

 private:
  friend struct FSuperDexBotIKRetargeterTestAccess;
  bool ValidateSourceProvider(FString& Diagnostic) const;
  bool ComputeSuperDexPoseFromJointPositions(
      TConstArrayView<double> DisplayPositions,
      FSuperDexPose& OutPose,
      FString& Diagnostic);
  void ReleaseSolver();
  void SetStatus(bool bValid, const FString& Diagnostic, bool bConfigurationError);
  void PublishPending();

  UPROPERTY(Transient)
  TObjectPtr<USuperDexIKSolverObject> Solver;
  UPROPERTY(Transient)
  TObjectPtr<USuperDexBotIKConfig> InitializedConfig;
  UPROPERTY(Transient)
  TObjectPtr<USuperDexBot> InitializedOutputBot;
  UPROPERTY(Transient)
  TObjectPtr<USuperDexBot> InitializedInputBot;

  superdex::robotics::BotIKConfig NativeConfig;
  superdex::robotics::BotPrefab OutputModel;
  superdex::robotics::BotPrefab InputModel;
  FString OutputModelJson;
  FString InputModelJson;
  // Legacy root-driven pose contract: captured output root relative to the selected component
  // source (normally the real clutch). Solver reference alignment is NOT output placement.
  // InitializeRetargeter explicitly recaptures this; solves, clutch toggles and FK never do.
  // Subsystem/actor pose sources with no component use the legacy identity relation.
  FTransform InputRootFrameFromOutputRootFrame = FTransform::Identity;
  // Input joint positions of the last applied pose: the reference for the next one's revolute
  // turns.
  mochi::DynamicArray<mochi::real> PreviousInputPositions;
  // Frames whose input pose didn't fit the InputBot's joints; throttles their warning.
  int32 LargeInputResidualFrames = 0;
  bool bInputBotHasPrismatic = false;
  TArray<double> JointCache;
  mutable FCriticalSection ResultMutex;
  FString LastDiagnostic;
  bool bHasSolution = false;
  bool bCurrentValid = false;
  bool bLastGoalsReached = false;
  bool bPendingStatus = false;
  bool bPublishing = false;
  bool bEvaluatingPose = false;
  uint64 SolutionSequence = 0;
  uint64 PublishedSequence = 0;
  uint64 StatusSequence = 0;
  uint64 PublishedStatusSequence = 0;
  uint64 ErrorMessageKey = 0;
  uint64 SolverGeneration = 0;
};
