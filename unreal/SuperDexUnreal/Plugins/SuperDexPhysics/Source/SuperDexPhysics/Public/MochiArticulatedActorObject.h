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

#include "Engine/DataTable.h"
#include "MochiActorObject.h"
#include "MochiArticulation.h"

#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/transform_srt.h>
#include <mochi_physics/mochi_physics.h>

#include "MochiArticulatedActorObject.generated.h"

struct FMochiPoseControllerGains {
  TArray<mochi::PoseTrackingParams> PositionTrackingGains;
  TArray<mochi::PoseTrackingParams> RotationTrackingGains;
  TArray<mochi::PoseTrackingParams> JointTrackingGains;
};

// Signature for user callback called on Mochi post-step to get the articulated actor pose.
// Both the link transforms and the joint positions are passed for convenience. The actor
// and scene pointers are passed if the user wants to query additional information.
typedef TFunction<void // no return
                  (const TArray<mochi::TransformRT>&, // WorldFromLink[nLinks]
                   const TArray<mochi::real>&, // JointPositions[nDofs]
                   mochi::Actor*, // ArticulatedActor (advanced use)
                   mochi::Scene* // Scene
                   )>
    FOnMochiArticulatedActorGetPose;

// Allows describing a pose target as either an array of WorldFromLink transforms
// of size #link, or an array of joint positions of size #DOF
typedef TVariant<TArray<mochi::TransformRT>, TArray<mochi::real>> FMochiArticulatedActorPoseVariant;

// Signature for user callback called on Mochi pre-step to set the pose controller target
// or do more sophisticated control directly on the actor. FMochiArticulatedActorPoseVariant
// OutPoseTarget allows setting either target link transforms or target joint positions. It is
// only relevant if bAddPoseController was true on actor creation. If it was false, it is assumed
// the user will be implementing direct torque control or similar through the actor pointer.
// TimeStepSec is the duration of the step about to be taken; during pre-step,
// Scene->GetLastTimeStep() still returns the previous step's.
typedef TFunction<bool // return true if you wish the pose target to be set
                  (FMochiArticulatedActorPoseVariant&, // OutPoseTarget (WorldFromLink[nLinks] or
                                                       // JointPositions[nDofs])
                   mochi::Actor*, // ArticulatedActor (advanced use)
                   mochi::Scene*, // Scene
                   double // TimeStepSec [s]
                   )>
    FOnMochiArticulatedActorControl;

// General purpose query data for use with UMochiArticulatedActorObject::TryGetQueryData.
struct FMochiArticulatedActorQueryData {
  // The root transform of the Mochi actor, in Unreal units (always available).
  // This is the articulated actor root transform, not the root link actor transform.
  // Their equivalence is not enforced by Mochi, and this is likely always identity.
  // Use LinksToWorld[0] to get the effective transform of the articulated actor.
  FTransform RootToWorld;
  // The linkToWorld transforms of the Mochi actor, in Unreal units (always available).
  TArray<FTransform> LinksToWorld;
  // The joint positions [rad or m] and velocities [rad/s or m/s], excluding base DOFs, in Unreal's
  // DOF convention (always available).
  TArray<double> JointPositions;
  TArray<double> JointVelocities;
  // The contact points of every actor that was registered for contact queries: the nested link
  // and soft actors named in LinksThatQueryContactPoints, plus the skin when
  // bSkinQueriesContactPoints was set. Empty if none were registered.
  TArray<FMochiContactPoint> ContactPoints;
};

UCLASS()
class SUPERDEXPHYSICS_API UMochiArticulatedActorObject : public UObject {
  GENERATED_BODY()
 public:
  // Create an articulated actor from a UMochiArticulation asset.
  static UMochiArticulatedActorObject* CreateArticulation(
      UObject* Outer,
      FString ArticulationName,
      const UMochiArticulation* Articulation,
      const FTransform& WorldFromRoot,
      const TArray<FTransform>& RootFromLinks,
      const UMochiContactParams* ContactParams,
      FName Layer,
      bool bFixedBase = false,
      bool bHasGravity = true,
      const TArray<double>& InitialJointVelocities = {},
      const FOnMochiArticulatedActorGetPose& GetPoseCallback = nullptr,
      bool bAddPoseController = false,
      const FMochiPoseControllerGains& PoseControllerGains = {},
      const FOnMochiArticulatedActorControl& ControlCallback = nullptr,
      const TArray<FName>& LinksThatQueryContactPoints = {},
      bool bSkinQueriesContactPoints = false,
      UMochiSceneObject* Scene = nullptr);

  // Wrap an already-created articulated actor handle with the UE-side infrastructure
  // (triple-buffered pose extraction, contact queries, callback management).
  // The actor must already exist in the given scene.
  // A skin is not a nested actor: its contact samples live on the top-level articulated actor, so
  // it cannot be named in LinksThatQueryContactPoints and is enabled by bSkinQueriesContactPoints
  // instead. Skin contacts are reported under the top-level actor's name.
  static UMochiArticulatedActorObject* CreateFromExistingArticulatedActor(
      UObject* Outer,
      FString Name,
      mochi::ActorHandle ArticulatedActorHandle,
      const FOnMochiArticulatedActorGetPose& GetPoseCallback = nullptr,
      const FOnMochiArticulatedActorControl& ControlCallback = nullptr,
      const TArray<FName>& LinksThatQueryContactPoints = {},
      bool bSkinQueriesContactPoints = false,
      UMochiSceneObject* Scene = nullptr);

 public:
  // Returns true if the UMochiArticulatedActorObject is associated with valid Scene and Mochi actor
  // handle.
  bool IsValid() const;
  // Get the Scene this Mochi actor is associated with.
  UMochiSceneObject* GetMochiScene() const;
  // Returns the number of degrees of freedom in the articulated actor.
  int GetNumDofs() const;
  // Returns true if the articulated actor has a pose controller.
  bool HasPoseController() const;
  // Add a pose controller. If one already exists, it will be removed.
  void AddPoseController(
      const FMochiPoseControllerGains& PoseControllerGains,
      bool bBlocking = false);
  // Remove pose controller if it exists.
  void RemovePoseController(bool bBlocking = false);
  // Get the articulated actor handle.
  mochi::ActorHandle GetArticulatedActorHandle() const;
  // Get the link actor handles.
  const TArray<mochi::ActorHandle>& GetNestedLinkActorHandles() const;
  // Get the link actor names.
  const TArray<FName>& GetNestedLinkActorNames() const;
  // Get the index of a link actor by name; -1 if the Name is invalid.
  int GetNestedLinkActorIndex(FName Name) const;
  // Get the soft actor handles (if there are any.)
  const TArray<mochi::ActorHandle>& GetNestedSoftActorHandles() const;
  // Get the soft actor names (if there are any.)
  const TArray<FName>& GetNestedSoftActorNames() const;
  // Get the index of a soft actor by name; -1 if the Name is invalid.
  int GetNestedSoftActorIndex(FName Name) const;
  // Try to get the latest query data if available.
  void TryGetQueryData(
      TFunctionRef<void(const FMochiArticulatedActorQueryData& QueryData)> Callback);
  // Teleport the articulation to a new root transform.
  void SetRootTransform(const FTransform& WorldFromLocal, bool bBlocking = false) const;
  // Teleport the articulation to the given pose based on joint positions.
  void SetArticulatedPoseFromJoints(TArray<double> const& JointPositions, bool bBlocking = false)
      const;
  // Get the contact params of each rigid link actor. Returns an array of size NumLinks.
  // Returns nullptr for links without contact params (e.g., dummy links, static actors).
  // This is a synchronous query that blocks until the Mochi thread has processed the request.
  TArray<UMochiContactParams*> GetLinkContactParams() const;
  // Set the contact params of a single rigid link actor by index.
  void SetLinkContactParams(
      int LinkIndex,
      const UMochiContactParams* ContactParams,
      bool bBlocking = false) const;
  // Get the density of each rigid link actor. Returns an array of size NumLinks.
  // This is a synchronous query that blocks until the Mochi thread has processed the request.
  TArray<float> GetLinkDensities() const;
  // Set the density of a single rigid link actor by index.
  void SetLinkDensity(int LinkIndex, float Density, bool bBlocking = false) const;
  // Force destroy the underlying Mochi actor and cancel the registered query callback.
  // This will also happen in BeginDestroy when GC'd.
  void DestroyMochiResources();

 protected:
  // Ensures Mochi actor is destroyed when this UMochiArticulatedActorObject is GC.
  virtual void BeginDestroy() override;

 private:
  void ExtractQueryData_Internal(mochi::Actor* Actor, mochi::Scene* SyncScene);

  // Data collected from the Mochi thread during actor setup, used to populate
  // the UMochiArticulatedActorObject on the game thread.
  struct FActorSetupData {
    FString ActorName;
    mochi::ActorHandle ActorHandle;
    TArray<mochi::ActorHandle> LinkActorHandles;
    TArray<FName> LinkActorNames;
    TArray<mochi::ActorHandle> SoftActorHandles;
    TArray<FName> SoftActorNames;
    int NumDofs = 0;
    bool bHasPoseController = false;
    TArray<TPair<FName, mochi::ActorHandle>> ContactableActorHandles;
    mochi::QueryHandle ArticulatedControllerForceQueryHandle = {};
    mochi::CallbackHandle PreStepCallback = {};
    mochi::CallbackHandle PostStepCallback = {};
  };

  // Mochi thread: extract handles/names from a created articulated actor, set user data,
  // register contact queries, and register GetPose/Control callbacks.
  static void SetupActorHandlesAndCallbacks(
      mochi::Scene* SyncScene,
      mochi::Actor* ArticulatedActor,
      UObject* Outer,
      const FOnMochiArticulatedActorGetPose& GetPoseCallback,
      const FOnMochiArticulatedActorControl& ControlCallback,
      const TArray<FName>& LinksThatQueryContactPoints,
      bool bSkinQueriesContactPoints,
      FActorSetupData& OutData,
      mochi::ErrorLog& Error);

  // Game thread: create and populate a UMochiArticulatedActorObject from collected setup data.
  static UMochiArticulatedActorObject*
  CreateFromSetupData(UObject* Outer, UMochiSceneObject* Scene, const FActorSetupData& Data);

 public:
  // [ADVANCED] If true, the destruction/GC of this MochiArticulatedActorObject will destroy
  // the underlying Mochi actor. This is the DEFAULT behavior. The only reason to
  // set this to false is if the actor lifetime needs to be managed by a higher
  // level object, component, or system (e.g. Bots). Note that registered queries owned
  // by this MochiArticulatedActorObject WILL be canceled regardless of what this is set to.
  bool bDestroyMochiActorOnObjectDestruction = true;

 private:
  UMochiSceneObject* MochiSceneObject = nullptr;
  mochi::ActorHandle MochiActorHandle = {};
  TArray<mochi::ActorHandle> LinkActorHandles = {};
  TArray<FName> LinkActorNames = {};
  TArray<mochi::ActorHandle> SoftActorHandles = {};
  TArray<FName> SoftActorNames = {};
  mochi::CallbackHandle PreStepCallbackHandle = {};
  mochi::CallbackHandle PostStepCallbackHandle = {};
  mochi::CallbackHandle ExtractQueryDataCallbackHandle = {};
  mochi::QueryHandle ArticulatedControllerForceQueryHandle = {};
  int NumDofs = 0;
  std::atomic<bool> bHasArticulatedPoseController{false};
  // contact querying data
  bool bQueryContactPoints = false;
  TArray<TPair<FName, mochi::ActorHandle>> ContactableActorHandles = {};
  // query data synchronization from Mochi thread to UE thread
  FCriticalSection PendingQueryDataLock;
  FMochiArticulatedActorQueryData QueryDataStorage[3];
  FMochiArticulatedActorQueryData* MochiThreadQueryData = &QueryDataStorage[0];
  FMochiArticulatedActorQueryData* UnrealThreadQueryData = &QueryDataStorage[2];
  FMochiArticulatedActorQueryData* PendingQueryData = &QueryDataStorage[1];
  std::atomic<bool> bHasPendingQueryData{false};
  std::atomic<bool> bIsUpdatingUnrealQueryData{false};
};
