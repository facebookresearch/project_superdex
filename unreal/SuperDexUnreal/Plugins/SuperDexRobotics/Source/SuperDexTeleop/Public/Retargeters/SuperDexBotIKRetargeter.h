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
class UMaterialInterface;

/**
 * Input-bot pose -> fixed-output-base document-authored Bot IK, for any input bot with a Free/Hard
 * root and Hard/Revolute/Prismatic/Spherical joints. Input poses hold joint positions of the input
 * bot, or of another bot of its skeleton: their parent-relative link transforms, in the input bot's
 * link order (only their count is checked), pose the input bot. Each joint keeps what it can do of
 * its link's motion (see superdex::robotics::ComputePoseFromLinkTransforms). The root link's local
 * transform places the input bot's base. Output poses hold joint positions of the output bot. Root
 * motion is carried through Config->OutputBaseFrameFromInputBaseFrame (the input bot's base
 * relative to the output bot's base), not a floating IK output base: the output root is that
 * frame's inverse applied to the source root. A component source attached to this retargeter
 * (normally the clutch) is placed at that frame, so everything that measures it agrees. Solves on
 * the game thread. Physics consumers read snapshots; they must not pull a solve.
 * Configuration/model edits require InitializeRetargeter. No recording is opened at runtime.
 */
UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class SUPERDEXTELEOP_API USuperDexBotIKRetargeter : public USuperDexRetargeter {
  GENERATED_BODY()

 public:
  /**
   * The bot IK config solved from the input bot pose. Edits take effect after InitializeRetargeter.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bot IK")
  TObjectPtr<USuperDexBotIKConfig> Config;

  /**
   * Publish each fresh solve's joint positions as hand joint targets on the owner's TeleopNode
   * operator component, whose own settings govern communication. Targets are invalidated when no
   * fresh solve is available.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bot IK|Teleop")
  bool bPublishTeleopHandJointTargets = false;

  /** Use available hand feedback for the displayed pose, without changing solved command joints.
   * The display follows feedback even on frames without a fresh solve (tracking lost, clutch not
   * yet initialized): only the mesh is posed then, and the pose stays invalid to consumers.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bot IK|Teleop")
  bool bSubscribeToTeleopHandJointPositions = false;

  /**
   * Scales the solved joint positions, except those in JointAngleMultiplierExcludeIndex, then
   * clamps every joint to the output bot's authored limits. 1.0 leaves the solve as is (no
   * clamping). Applies to the published command and its display, not to the solver or feedback.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bot IK|Teleop")
  double JointAngleMultiplier = 1.0;

  /** Output bot DOF indices JointAngleMultiplier does not scale (they are still clamped). */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bot IK|Teleop")
  TArray<int32> JointAngleMultiplierExcludeIndex;

  /**
   * Draw each active IK goal every frame with a fresh solve: the goal frame resolved from the
   * input pose and the constraint frame on the solved output link (both as small coordinate
   * frames), a black line between them (the remaining error), and lines from each frame to its
   * link (white: output link, yellow: input link, for goals bound to one).
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bot IK|Debug")
  bool bDebugDrawIKCoordinateFrames = false;

  /** Show the input bot (e.g. the tracked OpenXR hand) as the IK sees it, on every frame with a
   * fresh solve: its mesh posed at the joint positions fitted to the input pose, placed at
   * Config->OutputBaseFrameFromInputBaseFrame on this component's root bone and scaled by the
   * input bot's scale times Config->InputBotScale.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bot IK|Debug")
  bool bDebugDrawInputPoseSkeleton = false;

  /** Material for every slot of the input pose skeleton, the translucent tracked-hand material by
   * default; null keeps the input bot mesh's own materials.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bot IK|Debug")
  TObjectPtr<UMaterialInterface> DebugDrawInputPoseSkeletonMaterialOverride = nullptr;

  /** The input pose skeleton mesh, created on first use; null before. */
  UPoseableMeshComponent* GetDebugInputPoseMesh() const {
    return DebugInputPoseMesh;
  }

  /** One active goal's frames from the latest solve, relative to the output bot's base (this
   * component's root bone). Captured only while bDebugDrawIKCoordinateFrames is set.
   */
  struct FDebugGoalFrames {
    /** Where the input pose wants the constraint frame. */
    FTransform GoalInOutputBase;
    /** The constraint frame on the solved output link. */
    FTransform ConstraintInOutputBase;
    FTransform OutputLinkInOutputBase;

    /** Valid only if bHasInputLink. */
    FTransform InputLinkInOutputBase;
    /** Valid only if bHasInputLink. */
    FName InputLinkName;
    bool bHasInputLink = false;
  };
  const TArray<FDebugGoalFrames>& GetDebugGoalFrames() const {
    return DebugGoalFrames;
  }

  /**
   * Validates the configuration and creates the IK solver, replacing any previous one.
   *
   * @return False off the game thread, if the configuration is invalid or if the solver cannot be
   *   created; otherwise true.
   */
  UFUNCTION(BlueprintCallable, Category = "Bot IK")
  bool InitializeRetargeter();

  /**
   * Checks that Config is complete and valid for this retargeter without creating a solver.
   *
   * @param Diagnostic Set to why the configuration is invalid, or cleared.
   * @return True if the configuration is valid.
   */
  UFUNCTION(BlueprintCallable, CallInEditor, Category = "Bot IK")
  bool ValidateConfiguration(FString& Diagnostic) const;

  /** The last published joint positions [rad or m] in Config.OutputBot DOF order, including
   * JointAngleMultiplier. Returns whether any solution exists; bCurrentlyValid tells whether it is
   * from the current frame or held from before tracking was lost.
   */
  UFUNCTION(BlueprintPure, Category = "Bot IK")
  bool GetCachedJointConfiguration(TArray<double>& JointPositions, bool& bCurrentlyValid) const;

  virtual bool GetTransformedPose_Implementation(
      const FSuperDexPose& SourcePose,
      bool bSourceIsValid,
      FSuperDexPose& TransformedPose) override;
  virtual bool GetPose(FSuperDexPose& OutPose) override;
  virtual bool GetCachedPose(FSuperDexPose& OutPose) const override;

 protected:
  /** Runs without a valid source pose and hides the editor viewport pose. */
  USuperDexBotIKRetargeter();
  virtual void OnRegister() override;
  virtual void OnChildAttached(USceneComponent* ChildComponent) override;
  virtual void BeginPlay() override;

  /** Without a fresh solve this tick, poses the mesh from hand feedback (display only). */
  virtual void OnTickManageCache(FSuperDexPose& Pose, bool bIsValid) override;
  virtual void TickComponent(
      float DeltaTime,
      enum ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
  virtual void OnComponentDestroyed(bool bDestroyingHierarchy) override;
#if WITH_EDITOR
  virtual void PostEditChangeProperty(FPropertyChangedEvent& Event) override;
  virtual EDataValidationResult IsDataValid(FDataValidationContext& Context) const override;
#endif

 private:
  friend struct FSuperDexBotIKRetargeterTestAccess;
  friend struct FBotIKBiasedCommandTestAccess;
  struct FBiasedCommandLimit {
    double Minimum = 0;
    double Maximum = 0;
    bool bHasMinimum = false;
    bool bHasMaximum = false;
  };

  /** Prepared once in instantiated command units; no random-pose fallback bounds. */
  TArray<FBiasedCommandLimit> BiasedCommandLimits;
  void PrepareBiasedCommandLimits();
  bool ApplyJointBiasAndLimits(TArray<double>& Positions) const;

  /** The component pose source, or null for subsystem/actor sources. */
  USuperDexPoseableComponent* GetComponentPoseSource() const;

  /**
   * The source's relative transform that puts the input bot's base at
   * Config->OutputBaseFrameFromInputBaseFrame. False if the source is not attached to this
   * component, whose placement is then left to its owner.
   */
  bool ComputeSourcePlacement(
      const USuperDexPoseableComponent& Source,
      FTransform& OutSourceRelative) const;

  /**
   * The output root in the source root's frame implied by
   * Config->OutputBaseFrameFromInputBaseFrame.
   */
  FTransform ComputeInputRootFrameFromOutputRootFrame() const;

  /** Place an attached component source per ComputeSourcePlacement. */
  void PlaceSourceAtConfigInputBaseFrame();

  /** Resolve the active goals' frames exactly as the native solver does, after a solve. */
  void CaptureDebugGoalFrames(const mochi::TransformRT& WorldFrameFromInputBaseFrame);
  void DrawDebugGoalFrames() const;
  TArray<FDebugGoalFrames> DebugGoalFrames;

  /**
   * The input bot as the latest solve saw it, captured while bDebugDrawInputPoseSkeleton is set.
   */
  void CaptureDebugInputPose(const mochi::TransformRT& WorldFrameFromInputBaseFrame);
  void UpdateDebugInputPoseMesh(bool bFreshSolve);
  UPROPERTY(Transient)
  TObjectPtr<UPoseableMeshComponent> DebugInputPoseMesh;

  /** Includes the display scale. */
  FTransform DebugInputBaseInOutputBase = FTransform::Identity;
  /** Nominal (unscaled), input bot link order. */
  TArray<FTransform> DebugInputLinksInParent;
  bool bDebugInputPoseMeshMatchesBot = false;
  bool bHasDebugInputPose = false;
  void ReleaseSolver();
  void SetStatus(bool bValid, const FString& Diagnostic, bool bConfigurationError);

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

  /**
   * Root-driven pose contract: the output root in the source root's frame, so the output root is
   * this applied to the source pose's SkeletonRootToVrRoot. Derived from
   * Config->OutputBaseFrameFromInputBaseFrame on InitializeRetargeter; solves, clutch toggles,
   * FK and component movement never change it.
   */
  FTransform InputRootFrameFromOutputRootFrame = FTransform::Identity;

  /**
   * Input joint positions of the last applied pose: the reference for the next one's revolute
   * turns.
   */
  mochi::DynamicArray<mochi::real> PreviousInputPositions;

  /** Frames whose input pose didn't fit the InputBot's joints; throttles their warning. */
  int32 LargeInputResidualFrames = 0;
  bool bInputBotHasPrismatic = false;

  /** Guarded by ResultMutex: read from other threads through the cached getters. */
  TArray<double> JointCache;
  mutable FCriticalSection ResultMutex;
  bool bHasSolution = false;
  bool bCurrentValid = false;

  /** The last status, only to log and show each change once. */
  FString LastDiagnostic;
  uint64 ErrorMessageKey = 0;
};
