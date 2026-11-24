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

/** General purpose query data for use with UMochiActorObject::TryGetQueryData. */
struct FMochiActorQueryData {
  /** The root transform of the Mochi actor, in Unreal units (always available). */
  FTransform RootTransform;
  /** Linear world velocity (always available). */
  FVector LinearVelocity;
  /** Angular world velocity (always available). */
  FVector AngularVelocity;
  /** The node positions of the Mochi actor (available if bQueryVolumeNodePositions true). */
  TArray<mochi::real> VolumeNodePositions;
  /** The node positions of the Mochi actor (available if bQuerySurfaceNodePositions true). */
  TArray<mochi::real> SurfaceNodePositions;
  /** The node normals of the Mochi actor (available if bQueryNodeNormals true). */
  TArray<mochi::real> NodeNormals;
  /** Elastic energy of a soft body relative to rest (available if bQueryElasticEnergy true). */
  float ElasticEnergy = 0.0f;
  /** The contact points of the Mochi actor (available if bQueryContactPoints true). */
  TArray<FMochiContactPoint> ContactPoints;
};

/** Unreal wrapper that creates, queries and updates native rigid and deformable actors. */
UCLASS()
class SUPERDEXPHYSICS_API UMochiActorObject : public UObject {
  GENERATED_BODY()

 public:
  /**
   * Creates a rigid Mochi actor from a tetrahedral mesh.
   *
   * @param Outer Unreal outer for the created wrapper object.
   * @param ActorName Name to assign to the created native actor.
   * @param WorldFromLocal Actor local-to-world transform.
   * @param Model Model or mesh supplying actor geometry.
   * @param MaterialParams Material-parameter asset supplied to the operation.
   * @param ContactParams Contact-parameter asset or native values supplied to the operation.
   * @param ColliderType Native collider representation.
   * @param Layer Native contact layer.
   * @param bIsStatic Whether the created actor is static.
   * @param bHasGravity Whether gravity affects the created actor.
   * @param BoundaryElementType Boundary discretization used for contact sampling.
   * @param BoundarySubsampling Boundary-subsampling parameters supplied to actor creation.
   * @param GridSdfParams Grid signed-distance-field parameters supplied to actor creation.
   * @param MassOverride Mass override supplied to actor creation.
   * @param CenterOfMassOverride Center-of-mass override supplied to actor creation.
   * @param MomentOfInertiaOverride Inertia override supplied to actor creation.
   * @param bQuerySurfaceNodePositions Whether to register surface-node position queries.
   * @param bQueryNodeNormals Whether to register node-normal queries.
   * @param bQueryContactPoints Whether to register contact-point queries.
   * @param Scene Scene wrapper associated with the operation.
   * @return The object returned by creation.
   */
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

  /**
   * Create a rigid actor wrapper from collision geometry extracted from a static mesh.
   *
   * @param Outer Unreal outer for the created wrapper object.
   * @param ActorName Name to assign to the created native actor.
   * @param WorldFromLocal Actor local-to-world transform.
   * @param Model Model or mesh supplying actor geometry.
   * @param CollisionSource Static-mesh collision geometry to extract.
   * @param MaterialParams Material-parameter asset supplied to the operation.
   * @param ContactParams Contact-parameter asset or native values supplied to the operation.
   * @param ColliderType Native collider representation.
   * @param Layer Native contact layer.
   * @param bIsStatic Whether the created actor is static.
   * @param bHasGravity Whether gravity affects the created actor.
   * @param BoundaryElementType Boundary discretization used for contact sampling.
   * @param BoundarySubsampling Boundary-subsampling parameters supplied to actor creation.
   * @param GridSdfParams Grid signed-distance-field parameters supplied to actor creation.
   * @param MassOverride Mass override supplied to actor creation.
   * @param CenterOfMassOverride Center-of-mass override supplied to actor creation.
   * @param MomentOfInertiaOverride Inertia override supplied to actor creation.
   * @param bQuerySurfaceNodePositions Whether to register surface-node position queries.
   * @param bQueryNodeNormals Whether to register node-normal queries.
   * @param bQueryContactPoints Whether to register contact-point queries.
   * @param Scene Scene wrapper associated with the operation.
   * @return The object returned by creation.
   */
  static UMochiActorObject* CreateRigidFromSurfaceMesh(
      UObject* Outer,
      FString ActorName,
      const FTransform& WorldFromLocal,
      const UStaticMesh* Model,
      const ETriangularMeshCollisionSource CollisionSource,
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

  /**
   * Create a rigid actor wrapper from an existing native shape.
   *
   * @param Outer Unreal outer for the created wrapper object.
   * @param ActorName Name to assign to the created native actor.
   * @param WorldFromLocal Actor local-to-world transform.
   * @param ModelShape Native shape handle supplying rigid geometry.
   * @param MaterialParams Material-parameter asset supplied to the operation.
   * @param ContactParams Contact-parameter asset or native values supplied to the operation.
   * @param ColliderType Native collider representation.
   * @param Layer Native contact layer.
   * @param bIsStatic Whether the created actor is static.
   * @param bHasGravity Whether gravity affects the created actor.
   * @param BoundaryElementType Boundary discretization used for contact sampling.
   * @param BoundarySubsampling Boundary-subsampling parameters supplied to actor creation.
   * @param GridSdfParams Grid signed-distance-field parameters supplied to actor creation.
   * @param MassOverride Mass override supplied to actor creation.
   * @param CenterOfMassOverride Center-of-mass override supplied to actor creation.
   * @param MomentOfInertiaOverride Inertia override supplied to actor creation.
   * @param bQuerySurfaceNodePositions Whether to register surface-node position queries.
   * @param bQueryNodeNormals Whether to register node-normal queries.
   * @param bQueryContactPoints Whether to register contact-point queries.
   * @param Scene Scene wrapper associated with the operation.
   * @return The object returned by creation.
   */
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

  /**
   * Creates a soft Mochi actor.
   *
   * @param Outer Unreal outer for the created wrapper object.
   * @param ActorName Name to assign to the created native actor.
   * @param WorldFromLocal Actor local-to-world transform.
   * @param Model Model or mesh supplying actor geometry.
   * @param MaterialParams Material-parameter asset supplied to the operation.
   * @param ContactParams Contact-parameter asset or native values supplied to the operation.
   * @param ColliderType Native collider representation.
   * @param Layer Native contact layer.
   * @param bHasGravity Whether gravity affects the created actor.
   * @param BoundaryElementType Boundary discretization used for contact sampling.
   * @param bUseContactSkin Whether to use the model contact skin.
   * @param RomParams Reduced-order-model settings supplied to actor creation.
   * @param bUseRecentering Whether to enable native actor recentering.
   * @param bQuerySurfaceNodePositions Whether to register surface-node position queries.
   * @param bQueryVolumeNodePositions Whether to register volume-node position queries.
   * @param bQueryNodeNormals Whether to register node-normal queries.
   * @param bQueryContactPoints Whether to register contact-point queries.
   * @param bQueryElasticEnergy Whether to register an elastic-energy query.
   * @param Scene Scene wrapper associated with the operation.
   * @return The object returned by creation.
   */
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

  /**
   * Creates a shell Mochi actor from a triangular mesh.
   *
   * @param Outer Unreal outer for the created wrapper object.
   * @param ActorName Name to assign to the created native actor.
   * @param WorldFromLocal Actor local-to-world transform.
   * @param Shape Native shape handle supplying actor geometry.
   * @param ShellMaterialParams Shell-material asset supplied to actor creation.
   * @param ContactParams Contact-parameter asset or native values supplied to the operation.
   * @param ColliderType Native collider representation.
   * @param Layer Native contact layer.
   * @param bHasGravity Whether gravity affects the created actor.
   * @param PointCloudParams Point-cloud collider parameters supplied to actor creation.
   * @param bUseContactSkin Whether to use the model contact skin.
   * @param ContactElementType Contact-sampling discretization.
   * @param bQuerySurfaceNodePositions Whether to register surface-node position queries.
   * @param bQueryNodeNormals Whether to register node-normal queries.
   * @param bQueryContactPoints Whether to register contact-point queries.
   * @param Scene Scene wrapper associated with the operation.
   * @return The object returned by creation.
   */
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

  /**
   * Creates a rod Mochi actor from a polyline mesh with an optional contact skin.
   *
   * @param Outer Unreal outer for the created wrapper object.
   * @param ActorName Name to assign to the created native actor.
   * @param WorldFromLocal Actor local-to-world transform.
   * @param Shape Native shape handle supplying actor geometry.
   * @param RodMaterialParams Rod-material asset supplied to actor creation.
   * @param ContactParams Contact-parameter asset or native values supplied to the operation.
   * @param ColliderType Native collider representation.
   * @param Layer Native contact layer.
   * @param bHasGravity Whether gravity affects the created actor.
   * @param PointCloudParams Point-cloud collider parameters supplied to actor creation.
   * @param bUseContactSkin Whether to use the model contact skin.
   * @param ContactElementType Contact-sampling discretization.
   * @param ContactSkinElementType Boundary discretization used for contact-skin sampling.
   * @param bQuerySurfaceNodePositions Whether to register surface-node position queries.
   * @param bQueryNodeNormals Whether to register node-normal queries.
   * @param bQueryContactPoints Whether to register contact-point queries.
   * @param Scene Scene wrapper associated with the operation.
   * @return The object returned by creation.
   */
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

  /**
   * Creates a UMochiActorObject wrapper around an existing Mochi actor handle.
   *
   * @param Outer Unreal outer for the created wrapper object.
   * @param MochiActorHandle Handle of the existing native actor.
   * @param bQuerySurfaceNodePositions Whether to register surface-node position queries.
   * @param bQueryVolumeNodePositions Whether to register volume-node position queries.
   * @param bQueryNodeNormals Whether to register node-normal queries.
   * @param bQueryContactPoints Whether to register contact-point queries.
   * @param Scene Scene wrapper associated with the operation.
   * @param bUseContactSkinIfPresent Whether to use an available model contact skin.
   * @return The object returned by creation.
   */
  static UMochiActorObject* CreateFromExistingActor(
      UObject* Outer,
      mochi::ActorHandle MochiActorHandle,
      bool bQuerySurfaceNodePositions = false,
      bool bQueryVolumeNodePositions = false,
      bool bQueryNodeNormals = false,
      bool bQueryContactPoints = false,
      UMochiSceneObject* Scene = nullptr,
      bool bUseContactSkinIfPresent = false);

  /**
   * Creates a static rigid implicit plane.
   *
   * @param Outer Unreal outer for the created wrapper object.
   * @param ActorName Name to assign to the created native actor.
   * @param WorldFromLocal Actor local-to-world transform.
   * @param Normal Normal of the implicit plane.
   * @param Distance Offset of the implicit plane.
   * @param MaterialParams Material-parameter asset supplied to the operation.
   * @param ContactParams Contact-parameter asset or native values supplied to the operation.
   * @param Layer Native contact layer.
   * @param Scene Scene wrapper associated with the operation.
   * @return The object returned by creation.
   */
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

  /**
   * Creates a static rigid implicit sphere.
   *
   * @param Outer Unreal outer for the created wrapper object.
   * @param ActorName Name to assign to the created native actor.
   * @param WorldFromLocal Actor local-to-world transform.
   * @param Center Center of the implicit sphere.
   * @param Radius Radius of the implicit sphere.
   * @param MaterialParams Material-parameter asset supplied to the operation.
   * @param ContactParams Contact-parameter asset or native values supplied to the operation.
   * @param Layer Native contact layer.
   * @param Scene Scene wrapper associated with the operation.
   * @return The object returned by creation.
   */
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
  /**
   * Returns true if the MochiActorObject is associated with valid Scene and Mochi actor handle.
   *
   * @return Whether a native scene is assigned and the actor handle is non-invalid.
   */
  bool IsValid() const;
  /**
   * Get the Scene this Mochi actor is associated with.
   *
   * @return The scene wrapper associated with the actor.
   */
  UMochiSceneObject* GetMochiScene() const;
  /**
   * Get the Mochi actor handle.
   *
   * @return The native actor handle associated with the wrapper.
   */
  mochi::ActorHandle GetMochiActorHandle() const;
  /**
   * Try to get the latest query data if available.
   *
   * @param Callback Callback supplied to the query or command.
   */
  void TryGetQueryData(TFunctionRef<void(const FMochiActorQueryData& QueryData)> Callback);
  /**
   * Set the root transform of the Mochi actor.
   *
   * @param WorldFromLocal Actor local-to-world transform.
   * @param bBlocking Whether to wait for queued commands after submitting this operation.
   */
  void SetRootTransform(const FTransform& WorldFromLocal, bool bBlocking = false) const;
  /**
   * Set the contact params of the Mochi actor.
   *
   * @param ContactParams Contact-parameter asset or native values supplied to the operation.
   * @param bBlocking Whether to wait for queued commands after submitting this operation.
   */
  void SetContactParams(const UMochiContactParams* ContactParams, bool bBlocking = false) const;
  /**
   * Set the material params of the Mochi actor.
   *
   * @param MaterialParams Material-parameter asset supplied to the operation.
   * @param bBlocking Whether to wait for queued commands after submitting this operation.
   */
  void SetMaterialParams(const UMochiMaterialParams* MaterialParams, bool bBlocking = false);
  /**
   * Set the material params of individual element(s) in the Mochi actor
   * by default is additive, but can be reset to original first.
   *
   * @param Modifiers Element material modifiers to apply.
   * @param bResetExisting Whether to reset existing element modifiers before applying these values.
   * @param bBlocking Whether to wait for queued commands after submitting this operation.
   */
  void ApplyMaterialModifiers(
      const TMap<int, FMochiMaterialModifier>& Modifiers,
      bool bResetExisting = false,
      bool bBlocking = false);
  /**
   * Retrieve cached pointer to last set material parameters.
   *
   * @return The material params.
   */
  const UMochiMaterialParams* GetMaterialParams() const;
  /**
   * Set linear and angular velocity of the Mochi actor
   *
   * @param LinearVelocity Linear velocity to apply.
   * @param AngularVelocity Angular velocity to apply.
   */
  void SetVelocity(FVector LinearVelocity, FVector AngularVelocity) const;
  /**
   * Get inertial properties of the actor (mass [kg], COM [cm], MOI [kg*cm^2])
   *
   * @param Mass Mass storage supplied to the native query.
   * @param CenterOfMass Center-of-mass storage supplied to the native query.
   * @param MomentOfInertia Inertia storage supplied to the native query.
   */
  void GetInertialParams(
      double& Mass,
      FVector& CenterOfMass,
      FMochiMomentOfInertia& MomentOfInertia) const;
  /**
   * Force destroy the underlying Mochi actor and cancel the registered query callback.
   * This will also happen in BeginDestroy when GC'd. If bDestroyMochiActorOnObjectDestruction
   * is false, the actor will NOT be destroyed and is assumed to be destroyed elsewhere afterwards.
   */
  void DestroyMochiResources();

 protected:
  /** Ensures Mochi actor is destroyed when this UMochiActorObject is GC. */
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
  /**
   * [ADVANCED] If true, the destruction/GC of this MochiActorObject will destroy
   * the underlying Mochi actor. This is the DEFAULT behavior. The only reason to
   * set this to false is if the actor lifetime needs to be managed by a higher
   * level object, component, or system (a.g. articulated actors). Note that
   * registered queries owned by this MochiActorObject WILL be canceled regardless
   * of what this is set to.
   */
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

  /** Cached copy of material parameters to avoid having to query the Mochi actor */
  const UMochiMaterialParams* MochiMaterialParams = nullptr;

  /** Query data synchronization from Mochi thread to UE thread */
  FCriticalSection PendingQueryDataLock;
  FMochiActorQueryData QueryDataStorage[3];
  FMochiActorQueryData* MochiThreadQueryData = &QueryDataStorage[0];
  FMochiActorQueryData* UnrealThreadQueryData = &QueryDataStorage[2];
  FMochiActorQueryData* PendingQueryData = &QueryDataStorage[1];
  std::atomic<bool> bHasPendingQueryData{false};
  std::atomic<bool> bIsUpdatingUnrealQueryData{false};
};
