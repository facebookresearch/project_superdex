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

#include "MochiActorObject.h"
#include "MochiArticulatedActorObject.h"
#include "MochiArticulation.h"
#include "MochiSceneObject.h"

#include "Engine/DataTable.h"

#include <superdex_robotics/ik/bot_ik.h>
#include <superdex_robotics/ik/bot_ik_config.h>
#include "CoreMinimal.h"
#include "MochiAssets.h"
#include "MochiDebugDraw.h"
#include "MochiPhysicsSettings.h"
#include "UObject/Object.h"

#include <mochi_core/utils/transform_srt.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include "SuperDexIKSolverObject.generated.h"

UCLASS()
class SUPERDEXROBOTICS_API USuperDexIKSolverObject : public UObject {
  GENERATED_BODY()
 public:
  static USuperDexIKSolverObject* CreateBotIKSolver(
      UObject* Outer,
      const class USuperDexBot* OutputBot,
      const superdex::robotics::BotIKCreateParams& Params);
  // Document couplings are validated as a complete batch; invalid input creates no solver.
  static USuperDexIKSolverObject* CreateBotIKSolver(
      UObject* Outer,
      const class USuperDexBot* OutputBot,
      const superdex::robotics::BotIKCreateParams& Params,
      mochi::Span<superdex::robotics::BotIKJointCouplingConfig const> JointCouplings);

  /* Validate the complete document and seed the private prefab once at creation. The bot asset
   * and subsequent warm starts are unchanged. Reference bot bindings are applied separately below.
   */
  static USuperDexIKSolverObject* CreateBotIKSolver(
      UObject* Outer,
      const class USuperDexBot* OutputBot,
      const superdex::robotics::BotIKConfig& Config);

  bool SetBotGoal(
      int32 OutputBotLinkIndex,
      const mochi::TransformRT& WorldFrameFromConstraintFrame,
      mochi::real PositionStiffness,
      mochi::real RotationStiffness);
  /* Full native-unit local attachment and world goal; zero stiffness clears that component. */
  bool SetBotGoal(
      int32 OutputBotLinkIndex,
      const mochi::TransformRT& LinkFrameFromConstraintFrame,
      const mochi::TransformRT& WorldFrameFromConstraintFrame,
      mochi::real PositionStiffness,
      mochi::real RotationStiffness);
  /* Synchronous borrowed inputs. Preserves all reference-link bindings, attachment frames and
   * independent retarget scale through BotIK; invalid input leaves the native goals unchanged.
   */
  bool SetBotInputConfiguration(
      superdex::robotics::BotPrefab const& InputBot,
      mochi::Span<mochi::real const> Configuration,
      mochi::TransformRT const& WorldFrameFromInputBaseFrame,
      mochi::real InputBotScale,
      mochi::Span<superdex::robotics::BotIKGoalConfig const> Goals);
  bool SolveBotIK(bool& OutGoalsReached);
  /* Native link-frame snapshots in metres/FLU, including geometry scale. Game-thread-only;
   * borrowed until the next mutation or destruction. Does not trigger a solve or expose the actor.
   */
  mochi::Span<mochi::TransformRT const> GetOutputBotWorldFromLinks() const;

  /* Native FK at joint-only coordinates, including prefab scale and joint/link rest frames.
   * Game-thread-only. Copies the snapshot and restores the private solver's original warm start;
   * performs no solve and does not change the cached solved command. Invalidates borrowed views.
   */
  bool CopyOutputBotFKAtJointConfiguration(
      TConstArrayView<double> Positions,
      mochi::DynamicArray<mochi::TransformRT>& OutWorldFromLinks);

  // Create a IKSolver scene with articulation. This will own its own scene and copy of the
  // articulation. When bFixedBase is true the articulation's root link is welded to WorldFromRoot
  // and the solve has no free base dofs, so every target must be reachable by the joints alone.
  // For a USuperDexBot this welds a copy of the bot prefab; the asset is left untouched.
  // JointCouplings ties pairs of joints together for the duration of the solve; an empty array
  // leaves the solve unchanged.
  // When bExcludeCollisionShapes is true, the articulation is built with no collision geometry.
  // The IK scene never uses collision (contacts are disabled below), and shape-bearing links add
  // per-solve broadphase cost, so excluding shapes speeds up the solve without changing its result.
  // Only honored for the USuperDexBot path; ignored (with a warning) for UMochiArticulation.
  static USuperDexIKSolverObject* CreateIKSolver(
      UObject* Outer,
      FString SolverName,
      const UMochiArticulationBase* Articulation,
      const FTransform& WorldFromRoot,
      const TArray<FTransform>& RootFromLinks,
      bool bFixedBase,
      const FMochiIKSolverParams& IKSolverParams,
      const TArray<FMochiJointCoupling>& JointCouplings = {},
      bool bExcludeCollisionShapes = false);

  /**
  Create a position target for a specific link. The mochi API restricts this to one constraint
  target per link so calling this function more than once will overwrite a previously assigned
  target.

  @param LinkIndex The index of the link/link in the articulation associated with the target
  constraint
  @param LocalPositionToLink The position in local link space that the constraint will be tied to
  @param TargetLocalPositionToWorld The position in world space the constraint will be pulling
  towards
  @param Weight The weight used by the IKSolver to determin final pose. It is best to have your
  weights across all constraints sum to something in the 1-10 range to avoid numerical precision
  values. The constraints act similarly to a spring model so the relative strength of the weights
  will result in a relative equilibrium. For positions imagine your weights as N/m and for rotations
  imagine them as Nm/rad to get your proportional effects.
  */
  void CreatePositionTargetForLink(
      int LinkIndex,
      FVector LocalPositionToLink,
      FVector TargetPositionToWorld,
      float Weight);
  // Removes the position target associated with a link
  void ClearPositionTargetForLink(int LinkIndex);

  /**
  Create a Rotation target for a specific link. The mochi API restricts this to one constraint
  target per link so calling this function more than once will overwrite a previously assigned
  target.

  @param LinkIndex The index of the link/link in the articulation associated with the target
  constraint
  @param LocalRotationToLink The rotation in local link space that the constraint will be tied to
  @param TargetLocalRotationToWorld The rotation in world space the constraint will be pulling
  towards
  @param Weight The weight used by the IKSolver to determin final pose. It is best to have your
  weights across all constraints sum to something in the 1-10 range to avoid numerical precision
  values. The constraints act similarly to a spring model so the relative strength of the weights
  will result in a relative equilibrium. For positions imagine your weights as N/m and for rotations
  imagine them as Nm/rad to get your proportional effects.
  */
  void CreateRotationTargetForLink(
      int LinkIndex,
      FQuat LocalRotationToLink,
      FQuat TargetRotationToWorld,
      float Weight);
  // Removes the rotation target associated with a link
  void ClearRotationTargetForLink(int LinkIndex);

  // Removes all goals on all links
  void ClearAllTargets();

  // Blocking call to run the IKSolver and generate a pose
  bool SolveIK();

  // Get the link transforms from the previous solve. These represent each link's world transform.
  const TArray<FTransform>& GetSolvedLinkToWorldTransforms() const;
  // Get the dofs from the previous solve. These represent the dofs of the articulated actor joints.
  // The additional 6 dofs provided when bFixedBase == false are removed. Simply get
  // LinkToWorldTransforms[0] if you want the transform of the base.
  const TArray<double>& GetSolvedDofs() const;

 public:
  // Returns true if the USuperDexIKSolverObject is associated with valid Scene and Mochi actor
  // handle.
  bool IsValid() const;
  // Returns the number of degrees of freedom in the articulated actor.
  int GetNumDofs() const;
  // True if LinkIndex is connected to the root link through Hard (weld) joints only, meaning no
  // dof can move it relative to the root. When the base is fixed such a link is immovable, so any
  // IK target on it is unsatisfiable. Returns false for an out of range LinkIndex.
  bool IsLinkRigidlyAttachedToRootLink(int32 LinkIndex) const;
  // Force destroy the underlying Mochi actor and cancel the registered query callback.
  // This will also happen in BeginDestroy when GC'd.
  void DestroyMochiResources();

 protected:
  // Ensures Mochi actor is destroyed when this USuperDexIKSolverObject is GC.
  virtual void BeginDestroy() override;

 private:
  static USuperDexIKSolverObject* CreatePreparedBotIKSolver(
      UObject* Outer,
      superdex::robotics::BotPrefab const& Prefab,
      superdex::robotics::BotIKCreateParams const& Params,
      mochi::Span<superdex::robotics::BotIKJointCouplingConfig const> JointCouplings);
  void UpdateBotResults();
  void OnArticulatedActorGetPose(
      const TArray<mochi::TransformRT>& InWorldFromLink,
      const TArray<mochi::real>& InJointDofs,
      mochi::Actor* ArticulatedActor,
      mochi::Scene* /*Scene*/);

 private:
  UPROPERTY()
  UMochiSceneObject* MochiSceneObject = nullptr;
  superdex::robotics::BotIK* NativeIK = nullptr;
  // Actual bot root layout. Native snapshots have already removed these DOFs.
  int NumBotBaseDofs = 0;
  bool bNativeJointLayout = false;
  UPROPERTY()
  UMochiArticulatedActorObject* ArticulatedActorObject = nullptr;

  bool bIsInitialized = false;
  int NumLinks = 0;
  mochi::experimental::IKSolver* IKSolver = nullptr;
  TArray<FTransform> SolvedLinkToWorld;
  TArray<double> SolvedDofs;
  // Legacy UE result slice: 0 for a requested fixed base, 6 otherwise. For bots, only the
  // difference from NumBotBaseDofs is stripped from the native joint-only snapshot.
  int NumMochiBaseDofs = 0;
  // Per link, whether it reaches the root link through Hard joints only. Indexed by link index.
  TArray<bool> LinkIsRigidlyAttachedToRootLink;

  static constexpr char const* Layer = "NoCollisionLayer";
};
