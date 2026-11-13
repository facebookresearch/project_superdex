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
#include "MatrixTypes.h"
#include "MochiModel.h"
#include "MochiSceneObject.h"
#include "MochiTypes.h"

#include <mochi_physics/mochi_physics.h>

#include <atomic>

#include "MochiActorObject.generated.h"

class UMochiShellMaterialParams;
class UMochiRodMaterialParams;

// General purpose query data for use with UMochiActorObject::TryGetQueryData.
struct FMochiActorQueryData {
  // The root transform of the Mochi actor, in Unreal units (always available).
  FTransform RootTransform;
  // Linear world velocity (always available).
  FVector LinearVelocity;
  // Angular world velocity (always available).
  FVector AngularVelocity;
  // The node positions of the Mochi actor (available if bQueryVolumeNodePositions true).
  TArray<mochi::real> VolumeNodePositions;
  // The node positions of the Mochi actor (available if bQuerySurfaceNodePositions true).
  TArray<mochi::real> SurfaceNodePositions;
  // The node normals of the Mochi actor (available if bQueryNodeNormals true).
  TArray<mochi::real> NodeNormals;
  // Elastic energy of a soft body relative to rest (available if bQueryElasticEnergy true).
  float ElasticEnergy = 0.0f;
  // The contact points of the Mochi actor (available if bQueryContactPoints true).
  TArray<FMochiContactPoint> ContactPoints;
};

UCLASS()
class SUPERDEXPHYSICS_API UMochiActorObject : public UObject {
  GENERATED_BODY()

 public:
  // Creates a rigid Mochi actor from a tetrahedral mesh.
  static UMochiActorObject* CreateRigid(
      UObject* Outer,
      FString ActorName,
      const FTransform& WorldFromLocal,
      const UMochiModel* Model,
      const UMochiMaterialParams* MaterialParams,
      const UMochiContactParams* ContactParams,
      EMochiColliderType ColliderType,
      FName Layer,
      bool bIsStatic,
      bool bHasGravity,
      const EMochiActorBoundaryElementType BoundaryElementType =
          EMochiActorBoundaryElementType::Default,
      const FMochiBoundarySubsamplingParams* BoundarySubsampling = nullptr,
      const FMochiGridSdfParams* GridSdfParams = nullptr,
      const double* MassOverride = nullptr,
      const FVector* CenterOfMassOverride = nullptr,
      const FMochiMomentOfInertia* MomentOfInertiaOverride = nullptr,
      bool bQuerySurfaceNodePositions = false,
      bool bQueryNodeNormals = false,
      bool bQueryContactPoints = false,
      UMochiSceneObject* Scene = nullptr);

  static UMochiActorObject* CreateRigidFromSurfaceMesh(
      UObject* Outer,
      FString ActorName,
      const FTransform& WorldFromLocal,
      const UStaticMesh* Model,
      const ETriangularMeshCollisionSource collisionSource,
      const UMochiMaterialParams* MaterialParams,
      const UMochiContactParams* ContactParams,
      EMochiColliderType ColliderType,
      FName Layer,
      bool bIsStatic,
      bool bHasGravity,
      const EMochiActorBoundaryElementType BoundaryElementType =
          EMochiActorBoundaryElementType::Default,
      const FMochiBoundarySubsamplingParams* BoundarySubsampling = nullptr,
      const FMochiGridSdfParams* GridSdfParams = nullptr,
      const double* MassOverride = nullptr,
      const FVector* CenterOfMassOverride = nullptr,
      const FMochiMomentOfInertia* MomentOfInertiaOverride = nullptr,
      bool bQuerySurfaceNodePositions = false,
      bool bQueryNodeNormals = false,
      bool bQueryContactPoints = false,
      UMochiSceneObject* Scene = nullptr);

  static UMochiActorObject* CreateRigidFromMochiShape(
      UObject* Outer,
      FString ActorName,
      const FTransform& WorldFromLocal,
      const mochi::ShapeHandle ModelShape,
      const UMochiMaterialParams* MaterialParams,
      const UMochiContactParams* ContactParams,
      EMochiColliderType ColliderType,
      FName Layer,
      bool bIsStatic,
      bool bHasGravity,
      const EMochiActorBoundaryElementType BoundaryElementType =
          EMochiActorBoundaryElementType::Default,
      const FMochiBoundarySubsamplingParams* BoundarySubsampling = nullptr,
      const FMochiGridSdfParams* GridSdfParams = nullptr,
      const double* MassOverride = nullptr,
      const FVector* CenterOfMassOverride = nullptr,
      const FMochiMomentOfInertia* MomentOfInertiaOverride = nullptr,
      bool bQuerySurfaceNodePositions = false,
      bool bQueryNodeNormals = false,
      bool bQueryContactPoints = false,
      UMochiSceneObject* Scene = nullptr);

  // Creates a soft Mochi actor.
  static UMochiActorObject* CreateSoft(
      UObject* Outer,
      FString ActorName,
      const FTransform& WorldFromLocal,
      const UMochiModel* Model,
      const UMochiMaterialParams* MaterialParams,
      const UMochiContactParams* ContactParams,
      EMochiColliderType ColliderType,
      FName Layer,
      bool bHasGravity,
      const EMochiActorBoundaryElementType BoundaryElementType =
          EMochiActorBoundaryElementType::Default,
      bool bUseContactSkin = false,
      const FMochiRomParams* RomParams = nullptr,
      bool bUseRecentering = true,
      bool bQuerySurfaceNodePositions = false,
      bool bQueryVolumeNodePositions = false,
      bool bQueryNodeNormals = false,
      bool bQueryContactPoints = false,
      bool bQueryElasticEnergy = false,
      UMochiSceneObject* Scene = nullptr);

  // Creates a shell Mochi actor from a triangular mesh.
  static UMochiActorObject* CreateShell(
      UObject* Outer,
      FString ActorName,
      const FTransform& WorldFromLocal,
      mochi::ShapeHandle Shape,
      const UMochiShellMaterialParams* ShellMaterialParams,
      const UMochiContactParams* ContactParams,
      EMochiColliderType ColliderType,
      FName Layer,
      bool bHasGravity,
      const mochi::PointCloudColliderParams* PointCloudParams = nullptr,
      bool bUseContactSkin = false,
      EMochiActorBoundaryElementType ContactElementType = EMochiActorBoundaryElementType::Default,
      bool bQuerySurfaceNodePositions = false,
      bool bQueryNodeNormals = false,
      bool bQueryContactPoints = false,
      UMochiSceneObject* Scene = nullptr);

  // Creates a rod Mochi actor from a polyline mesh with an optional contact skin.
  static UMochiActorObject* CreateRod(
      UObject* Outer,
      FString ActorName,
      const FTransform& WorldFromLocal,
      mochi::ShapeHandle Shape,
      const UMochiRodMaterialParams* RodMaterialParams,
      const UMochiContactParams* ContactParams,
      EMochiColliderType ColliderType,
      FName Layer,
      bool bHasGravity,
      const mochi::PointCloudColliderParams* PointCloudParams = nullptr,
      bool bUseContactSkin = false,
      EMochiActorSegmentElementType ContactElementType = EMochiActorSegmentElementType::Default,
      EMochiActorBoundaryElementType ContactSkinElementType =
          EMochiActorBoundaryElementType::Default,
      bool bQuerySurfaceNodePositions = false,
      bool bQueryNodeNormals = false,
      bool bQueryContactPoints = false,
      UMochiSceneObject* Scene = nullptr);

  // Creates a UMochiActorObject wrapper around an existing Mochi actor handle.
  static UMochiActorObject* CreateFromExistingActor(
      UObject* Outer,
      mochi::ActorHandle MochiActorHandle,
      bool bQuerySurfaceNodePositions = false,
      bool bQueryVolumeNodePositions = false,
      bool bQueryNodeNormals = false,
      bool bQueryContactPoints = false,
      UMochiSceneObject* Scene = nullptr,
      bool bUseContactSkinIfPresent = false);

  // Creates a static rigid implicit plane.
  static UMochiActorObject* CreateImplicitPlane(
      UObject* Outer,
      FString ActorName,
      const FTransform& WorldFromLocal,
      const FVector& Normal,
      float Distance,
      const UMochiMaterialParams* MaterialParams,
      const UMochiContactParams* ContactParams,
      FName Layer,
      UMochiSceneObject* Scene = nullptr);

  // Creates a static rigid implicit sphere.
  static UMochiActorObject* CreateImplicitSphere(
      UObject* Outer,
      FString ActorName,
      const FTransform& WorldFromLocal,
      const FVector& Center,
      float Radius,
      const UMochiMaterialParams* MaterialParams,
      const UMochiContactParams* ContactParams,
      FName Layer,
      UMochiSceneObject* Scene = nullptr);

 public:
  // Returns true if the MochiActorObject is associated with valid Scene and Mochi actor handle.
  bool IsValid() const;
  // Get the Scene this Mochi actor is associated with.
  UMochiSceneObject* GetMochiScene() const;
  // Get the Mochi actor handle.
  mochi::ActorHandle GetMochiActorHandle() const;
  // Try to get the latest query data if available.
  void TryGetQueryData(TFunctionRef<void(const FMochiActorQueryData& QueryData)> Callback);
  // Set the root transform of the Mochi actor.
  void SetRootTransform(const FTransform& WorldFromLocal, bool bBlocking = false) const;
  // Set the contact params of the Mochi actor.
  void SetContactParams(const UMochiContactParams* ContactParams, bool bBlocking = false) const;
  // Set the material params of the Mochi actor.
  void SetMaterialParams(const UMochiMaterialParams* MaterialParams, bool bBlocking = false);
  // Set the material params of individual element(s) in the Mochi actor
  // by default is additive, but can be reset to original first.
  void ApplyMaterialModifiers(
      const TMap<int, FMochiMaterialModifier>& Modifiers,
      bool bResetExisting = false,
      bool bBlocking = false);
  // Retrieve cached pointer to last set material parameters.
  const UMochiMaterialParams* GetMaterialParams() const;
  // Set linear and angular velocity of the Mochi actor
  void SetVelocity(FVector LinearVelocity, FVector AngularVelocity) const;
  // Get inertial properties of the actor (mass [kg], COM [cm], MOI [kg*cm^2])
  void GetInertialParams(
      double& Mass,
      FVector& CenterOfMass,
      FMochiMomentOfInertia& MomentOfInertia) const;
  // Force destroy the underlying Mochi actor and cancel the registered query callback.
  // This will also happen in BeginDestroy when GC'd. If bDestroyMochiActorOnObjectDestruction
  // is false, the actor will NOT be destroyed and is assumed to be destroyed elsewhere afterwards.
  void DestroyMochiResources();

 protected:
  // Ensures Mochi actor is destroyed when this UMochiActorObject is GC.
  virtual void BeginDestroy() override;

 private:
  static UMochiActorObject* Create_Internal(
      UObject* Outer,
      mochi::ActorHandle ActorHandle,
      bool bQuerySurfaceNodePositions,
      bool bQueryVolumeNodePositions,
      bool bQueryNodeNormals,
      bool bQueryContactPoints,
      bool bQueryElasticEnergy,
      const UMochiMaterialParams* MaterialParams,
      UMochiSceneObject* Scene,
      bool bUseVisualMeshIfPresent = true,
      bool bUseContactSkinIfPresent = false);

  void ExtractQueryData_Internal(const mochi::Actor* Actor, mochi::Scene* Scene);

 public:
  // [ADVANCED] If true, the destruction/GC of this MochiActorObject will destroy
  // the underlying Mochi actor. This is the DEFAULT behavior. The only reason to
  // set this to false is if the actor lifetime needs to be managed by a higher
  // level object, component, or system (a.g. articulated actors). Note that
  // registered queries owned by this MochiActorObject WILL be canceled regardless
  // of what this is set to.
  bool bDestroyMochiActorOnObjectDestruction = true;

 private:
  UPROPERTY()
  TObjectPtr<UMochiSceneObject> MochiSceneObject;
  mochi::ActorHandle MochiActorHandle = {};
  bool bQuerySurfaceNodePositions = false;
  bool bQueryVolumeNodePositions = false;
  bool bQueryNodeNormals = false;
  bool bQueryElasticEnergy = false;
  bool bQueryContactPoints = false;
  bool bUseVisualMeshIfPresent = true;
  bool bUseContactSkinIfPresent = false;
  mochi::CallbackHandle ExtractQueryDataCallbackHandle = {};

  // Cached copy of material parameters to avoid having to query the Mochi actor
  const UMochiMaterialParams* MochiMaterialParams = nullptr;

  // Query data synchronization from Mochi thread to UE thread
  FCriticalSection PendingQueryDataLock;
  FMochiActorQueryData QueryDataStorage[3];
  FMochiActorQueryData* MochiThreadQueryData = &QueryDataStorage[0];
  FMochiActorQueryData* UnrealThreadQueryData = &QueryDataStorage[2];
  FMochiActorQueryData* PendingQueryData = &QueryDataStorage[1];
  std::atomic<bool> bHasPendingQueryData{false};
  std::atomic<bool> bIsUpdatingUnrealQueryData{false};
};
