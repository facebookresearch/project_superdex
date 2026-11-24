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

#include "MochiActorObject.h"

#include "MochiAssets.h"
#include "MochiConversions.h"
#include "MochiGameInstanceSubsystem.h"
#include "MochiModel.h"
#include "MochiPhysicsModule.h"

#include "Engine/GameInstance.h"
#include "Engine/StaticMesh.h"
#include "Engine/World.h"
#include "PhysicsEngine/BodySetup.h"
#include "StaticMeshResources.h"

#include "MochiSceneUtils.h"

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/profile.h>
#include <mochi_physics/mochi_physics_experimental.h>

namespace MochiActorObjectInternal {

mochi::RigidActorParams CreateRigidActorParams(
    const char* Name,
    const FTransform& WorldFromLocal,
    const char* LayerStr,
    const mochi::ShapeHandle Shape,
    const UMochiMaterialParams* MaterialParams,
    const UMochiContactParams* ContactParams,
    const FMochiGridSdfParams* GridSdfParams,
    EMochiColliderType ColliderType,
    bool bIsStatic,
    bool bHasGravity,
    const EMochiActorBoundaryElementType BoundaryElementType,
    const FMochiBoundarySubsamplingParams* BoundarySubsampling,
    const double* MassOverride,
    const FVector* CenterOfMassOverride,
    const FMochiMomentOfInertia* MomentOfInertiaOverride) {
  MOCHI_PROFILE_SCOPE();
  mochi::RigidActorParams ActorParams;
  FUnrealToMochiConverter U2m;
  ActorParams.name = Name;
  ActorParams.shape = Shape;
  ActorParams.isStatic = bIsStatic;
  ActorParams.hasGravity = bHasGravity;
  ActorParams.worldFromLocal = U2m.TransformToMochi_RT(WorldFromLocal);
  ActorParams.colliderType = mochi::ColliderType(ColliderType);
  ActorParams.layer = LayerStr;
  if (ContactParams) {
    ActorParams.contact = ContactParams->ToMochi();
  }
  if (GridSdfParams) {
    ActorParams.sdf = ::ToMochi(*GridSdfParams);
  }

  // Set mass or density, but not both.
  if (MassOverride) {
    ActorParams.mass = *MassOverride;
  } else if (MaterialParams) {
    ActorParams.density = MaterialParams->Density;
  }

  if (CenterOfMassOverride && MomentOfInertiaOverride) {
    ActorParams.centerOfMass = U2m.TranslationToMochi(*CenterOfMassOverride);
    ActorParams.momentOfInertia = U2m.InertiaToMochi(*MomentOfInertiaOverride);
  }
  ActorParams.boundaryElementType = ToMochi(BoundaryElementType);
  if (BoundarySubsampling) {
    ActorParams.boundarySubsampling = ::ToMochi(*BoundarySubsampling);
  }
  return ActorParams;
}

static mochi::SoftActorParams CreateSoftActorParams(
    const char* Name,
    const FTransform& WorldFromLocal,
    const char* LayerStr,
    const mochi::ShapeHandle Shape,
    const UMochiMaterialParams* MaterialParams,
    const UMochiContactParams* ContactParams,
    bool bHasGravity,
    bool bUseContactSkin,
    const EMochiActorBoundaryElementType BoundaryElementType) {
  MOCHI_PROFILE_SCOPE();
  FUnrealToMochiConverter U2m;
  mochi::SoftActorParams ActorParams;
  ActorParams.name = Name;
  ActorParams.shape = Shape;
  ActorParams.worldFromLocal = U2m.TransformToMochi_RT(WorldFromLocal);
  ActorParams.hasGravity = bHasGravity;
  ActorParams.useContactSkin = bUseContactSkin;
  ActorParams.layer = LayerStr;
  if (MaterialParams) {
    ActorParams.material = MaterialParams->ToMochi();
  }
  if (ContactParams) {
    ActorParams.contact = ContactParams->ToMochi();
  }
  ActorParams.boundaryElementType = ToMochi(BoundaryElementType);
  return ActorParams;
}

mochi::experimental::ShellActorParams CreateShellActorParams(
    const char* Name,
    const FTransform& WorldFromLocal,
    const char* LayerStr,
    const mochi::ShapeHandle Shape,
    const UMochiShellMaterialParams* ShellMaterialParams,
    const UMochiContactParams* ContactParams,
    bool bHasGravity,
    EMochiColliderType ColliderType,
    const mochi::PointCloudColliderParams* PointCloudParams,
    bool bUseContactSkin,
    EMochiActorBoundaryElementType ContactElementType) {
  MOCHI_PROFILE_SCOPE();
  FUnrealToMochiConverter U2m;
  mochi::experimental::ShellActorParams ShellParams;
  ShellParams.name = Name;
  ShellParams.layer = LayerStr;
  ShellParams.worldFromLocal = U2m.TransformToMochi_RT(WorldFromLocal);
  ShellParams.shape = Shape;
  ShellParams.hasGravity = bHasGravity;
  ShellParams.colliderType = mochi::ColliderType(ColliderType);
  ShellParams.useContactSkin = bUseContactSkin;
  if (ShellMaterialParams) {
    ShellParams.material = ShellMaterialParams->ToMochi();
  }
  if (ContactParams) {
    ShellParams.contact = ContactParams->ToMochi();
  }
  if (PointCloudParams) {
    ShellParams.pointCloudCollider = *PointCloudParams;
  }
  ShellParams.contactElementType = ToMochi(ContactElementType);
  return ShellParams;
}

mochi::RodActorParams CreateRodActorParams(
    const char* Name,
    const FTransform& WorldFromLocal,
    const char* LayerStr,
    const mochi::ShapeHandle Shape,
    const UMochiRodMaterialParams* RodMaterialParams,
    const UMochiContactParams* ContactParams,
    bool bHasGravity,
    EMochiColliderType ColliderType,
    bool bUseContactSkin,
    const mochi::PointCloudColliderParams* PointCloudParams,
    EMochiActorSegmentElementType ContactElementType,
    EMochiActorBoundaryElementType ContactSkinElementType) {
  MOCHI_PROFILE_SCOPE();
  FUnrealToMochiConverter U2m;
  mochi::RodActorParams RodParams;
  RodParams.name = Name;
  RodParams.layer = LayerStr;
  RodParams.worldFromLocal = U2m.TransformToMochi_RT(WorldFromLocal);
  RodParams.shape = Shape;
  RodParams.hasGravity = bHasGravity;
  RodParams.colliderType = mochi::ColliderType(ColliderType);
  RodParams.useContactSkin = bUseContactSkin;
  if (RodMaterialParams) {
    RodParams.material = RodMaterialParams->ToMochi();
  }
  if (ContactParams) {
    RodParams.contact = ContactParams->ToMochi();
  }
  if (PointCloudParams) {
    RodParams.pointCloudCollider = *PointCloudParams;
  }
  RodParams.contactElementType = ToMochi(ContactElementType);
  RodParams.contactSkinElementType = ToMochi(ContactSkinElementType);
  return RodParams;
}

} // namespace MochiActorObjectInternal
using namespace MochiActorObjectInternal;

UMochiActorObject* UMochiActorObject::CreateRigid(
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
    const EMochiActorBoundaryElementType BoundaryElementType,
    const FMochiBoundarySubsamplingParams* BoundarySubsampling,
    const FMochiGridSdfParams* GridSdfParams,
    const double* MassOverride,
    const FVector* CenterOfMassOverride,
    const FMochiMomentOfInertia* MomentOfInertiaOverride,
    bool bQuerySurfaceNodePositions,
    bool bQueryNodeNormals,
    bool bQueryContactPoints,
    UMochiSceneObject* Scene) {
  MOCHI_PROFILE_SCOPE();
  const mochi::ShapeHandle Shape =
      Model->GetMochiShape(WorldFromLocal.GetScale3D(), FTransform::Identity);

  return CreateRigidFromMochiShape(
      Outer,
      ActorName,
      WorldFromLocal,
      Shape,
      MaterialParams,
      ContactParams,
      ColliderType,
      Layer,
      bIsStatic,
      bHasGravity,
      BoundaryElementType,
      BoundarySubsampling,
      GridSdfParams,
      MassOverride,
      CenterOfMassOverride,
      MomentOfInertiaOverride,
      bQuerySurfaceNodePositions,
      bQueryNodeNormals,
      bQueryContactPoints,
      Scene);
}

UMochiActorObject* UMochiActorObject::CreateRigidFromSurfaceMesh(
    UObject* Outer,
    FString ActorName,
    const FTransform& WorldFromLocal,
    const UStaticMesh* SurfaceMesh,
    const ETriangularMeshCollisionSource CollisionSource,
    const UMochiMaterialParams* MaterialParams,
    const UMochiContactParams* ContactParams,
    EMochiColliderType ColliderType,
    FName Layer,
    bool bIsStatic,
    bool bHasGravity,
    const EMochiActorBoundaryElementType BoundaryElementType,
    const FMochiBoundarySubsamplingParams* BoundarySubsampling,
    const FMochiGridSdfParams* GridSdfParams,
    const double* MassOverride,
    const FVector* CenterOfMassOverride,
    const FMochiMomentOfInertia* MomentOfInertiaOverride,
    bool bQuerySurfaceNodePositions,
    bool bQueryNodeNormals,
    bool bQueryContactPoints,
    UMochiSceneObject* Scene) {
  MOCHI_PROFILE_SCOPE();
  const mochi::ShapeHandle Shape = UMochiModel::GetTriangularMochiShapeFromStaticMesh(
      SurfaceMesh, CollisionSource, WorldFromLocal.GetScale3D());
  if (Shape.IsValid()) {
    return CreateRigidFromMochiShape(
        Outer,
        ActorName,
        WorldFromLocal,
        Shape,
        MaterialParams,
        ContactParams,
        ColliderType,
        Layer,
        bIsStatic,
        bHasGravity,
        BoundaryElementType,
        BoundarySubsampling,
        GridSdfParams,
        MassOverride,
        CenterOfMassOverride,
        MomentOfInertiaOverride,
        bQuerySurfaceNodePositions,
        bQueryNodeNormals,
        bQueryContactPoints,
        Scene);
  }
  UE_LOG(
      LogMochiPhysics,
      Error,
      TEXT(
          "[UMochiActorObject] Failed to create rigid actor from surface mesh because shape handle was invalid"));
  return nullptr;
}

UMochiActorObject* UMochiActorObject::CreateRigidFromMochiShape(
    UObject* Outer,
    FString ActorName,
    const FTransform& WorldFromLocal,
    const mochi::ShapeHandle Shape,
    const UMochiMaterialParams* MaterialParams,
    const UMochiContactParams* ContactParams,
    EMochiColliderType ColliderType,
    FName Layer,
    bool bIsStatic,
    bool bHasGravity,
    const EMochiActorBoundaryElementType BoundaryElementType,
    const FMochiBoundarySubsamplingParams* BoundarySubsampling,
    const FMochiGridSdfParams* GridSdfParams,
    const double* MassOverride,
    const FVector* CenterOfMassOverride,
    const FMochiMomentOfInertia* MomentOfInertiaOverride,
    bool bQuerySurfaceNodePositions,
    bool bQueryNodeNormals,
    bool bQueryContactPoints,
    UMochiSceneObject* Scene) {
  MOCHI_PROFILE_SCOPE();
  if (!VerifyOuterAndScene(Outer, Scene, TEXT("UMochiActorObject"))) {
    return nullptr;
  }
  if (!VerifySceneAcceptsNewActors(Scene, TEXT("UMochiActorObject"), ActorName)) {
    return nullptr;
  }
  mochi::Context* const MochiWorld = IMochiPhysicsModule::Get()->GetMochiContext();
  mochi::ActorHandle CreatedActorHandle;
  // Reference captures rely on the wait below before this scope ends. It completes queued work
  // only off the async simulation thread; synchronous scenes run this callback inline.
  Scene->QueueOrExecuteCommand([&](mochi::Scene* PhysicsScene) {
    mochi::ErrorLog E{};
    MOCHI_ERROR_RETURN(E);
    const FString NameStr = ActorName + "___Actor";
    const FString LayerStr = Layer.ToString();
    const mochi::RigidActorParams ActorParams = CreateRigidActorParams(
        TCHAR_TO_ANSI(*NameStr),
        WorldFromLocal,
        TCHAR_TO_ANSI(*LayerStr),
        Shape,
        MaterialParams,
        ContactParams,
        GridSdfParams,
        ColliderType,
        bIsStatic,
        bHasGravity,
        BoundaryElementType,
        BoundarySubsampling,
        MassOverride,
        CenterOfMassOverride,
        MomentOfInertiaOverride);
    mochi::Actor* Actor = PhysicsScene->CreateRigidActor(ActorParams, E);
    MOCHI_ERROR_RETURN(E);
    CreatedActorHandle = Actor->GetHandle();
  });
  Scene->WaitForQueuedCommands();
  return Create_Internal(
      Outer,
      CreatedActorHandle,
      bQuerySurfaceNodePositions,
      false, /* volume positions */
      bQueryNodeNormals,
      bQueryContactPoints,
      false,
      MaterialParams,
      Scene);
}

UMochiActorObject* UMochiActorObject::CreateSoft(
    UObject* Outer,
    FString ActorName,
    const FTransform& WorldFromLocal,
    const UMochiModel* Model,
    const UMochiMaterialParams* MaterialParams,
    const UMochiContactParams* ContactParams,
    EMochiColliderType ColliderType,
    FName Layer,
    bool bHasGravity,
    const EMochiActorBoundaryElementType BoundaryElementType,
    bool bUseContactSkin,
    const FMochiRomParams* RomParams,
    bool bUseRecentering,
    bool bQuerySurfaceNodePositions,
    bool bQueryVolumeNodePositions,
    bool bQueryNodeNormals,
    bool bQueryContactPoints,
    bool bQueryElasticEnergy,
    UMochiSceneObject* Scene) {
  MOCHI_PROFILE_SCOPE();
  if (!VerifyOuterAndScene(Outer, Scene, TEXT("UMochiActorObject"))) {
    return nullptr;
  }
  if (!VerifySceneAcceptsNewActors(Scene, TEXT("UMochiActorObject"), ActorName)) {
    return nullptr;
  }
  mochi::Context* const MochiWorld = IMochiPhysicsModule::Get()->GetMochiContext();
  mochi::ActorHandle CreatedActorHandle;
  const mochi::ShapeHandle Shape =
      Model->GetMochiShape(WorldFromLocal.GetScale3D(), FTransform::Identity);
  // Reference captures rely on the wait below before this scope ends. It completes queued work
  // only off the async simulation thread; synchronous scenes run this callback inline.
  Scene->QueueOrExecuteCommand([&](mochi::Scene* PhysicsScene) {
    mochi::ErrorLog E{};
    MOCHI_ERROR_RETURN(E);
    const FString NameStr = ActorName + "___SoftActor";
    const FString LayerStr = Layer.ToString();
    const mochi::SoftActorParams ActorParams = CreateSoftActorParams(
        TCHAR_TO_ANSI(*NameStr),
        WorldFromLocal,
        TCHAR_TO_ANSI(*LayerStr),
        Shape,
        MaterialParams,
        ContactParams,
        bHasGravity,
        bUseContactSkin,
        BoundaryElementType);

    mochi::experimental::ExperimentalSoftActorParams ExperimentalParams;
    ExperimentalParams.useRecentering = bUseRecentering;
    ExperimentalParams.colliderType = mochi::ColliderType(ColliderType);

    if (RomParams) {
      ExperimentalParams.rom = RomParams->ToMochi();
    }

    mochi::Actor* Actor =
        mochi::experimental::CreateSoftActor(PhysicsScene, ActorParams, ExperimentalParams, E);
    MOCHI_ERROR_RETURN(E);
    CreatedActorHandle = Actor->GetHandle();
  });
  Scene->WaitForQueuedCommands();
  return Create_Internal(
      Outer,
      CreatedActorHandle,
      bQuerySurfaceNodePositions,
      bQueryVolumeNodePositions,
      bQueryNodeNormals,
      bQueryContactPoints,
      bQueryElasticEnergy,
      MaterialParams,
      Scene,
      true /* use visual mesh if present */,
      true /* use contact skin if present */);
}

UMochiActorObject* UMochiActorObject::CreateShell(
    UObject* Outer,
    FString ActorName,
    const FTransform& WorldFromLocal,
    mochi::ShapeHandle Shape,
    const UMochiShellMaterialParams* ShellMaterialParams,
    const UMochiContactParams* ContactParams,
    EMochiColliderType ColliderType,
    FName Layer,
    bool bHasGravity,
    const mochi::PointCloudColliderParams* PointCloudParams,
    bool bUseContactSkin,
    EMochiActorBoundaryElementType ContactElementType,
    bool bQuerySurfaceNodePositions,
    bool bQueryNodeNormals,
    bool bQueryContactPoints,
    UMochiSceneObject* Scene) {
  MOCHI_PROFILE_SCOPE();
  if (!VerifyOuterAndScene(Outer, Scene, TEXT("UMochiActorObject"))) {
    return nullptr;
  }
  if (!VerifySceneAcceptsNewActors(Scene, TEXT("UMochiActorObject"), ActorName)) {
    return nullptr;
  }
  mochi::ActorHandle CreatedActorHandle;
  // Reference captures rely on the wait below before this scope ends. It completes queued work
  // only off the async simulation thread; synchronous scenes run this callback inline.
  Scene->QueueOrExecuteCommand([&](mochi::Scene* PhysicsScene) {
    mochi::ErrorLog E{};
    MOCHI_ERROR_RETURN(E);
    const FString NameStr = ActorName + "___ShellActor";
    const FString LayerStr = Layer.ToString();
    const mochi::experimental::ShellActorParams ShellParams = CreateShellActorParams(
        TCHAR_TO_ANSI(*NameStr),
        WorldFromLocal,
        TCHAR_TO_ANSI(*LayerStr),
        Shape,
        ShellMaterialParams,
        ContactParams,
        bHasGravity,
        ColliderType,
        PointCloudParams,
        bUseContactSkin,
        ContactElementType);

    mochi::Actor* Actor = mochi::experimental::CreateShellActor(PhysicsScene, ShellParams, E);
    MOCHI_ERROR_RETURN(E);
    CreatedActorHandle = Actor->GetHandle();
  });
  Scene->WaitForQueuedCommands();
  return Create_Internal(
      Outer,
      CreatedActorHandle,
      bQuerySurfaceNodePositions,
      false, /* volume positions */
      bQueryNodeNormals,
      bQueryContactPoints,
      false, /* elastic energy */
      nullptr, /* material params */
      Scene,
      true /* use visual mesh if present */,
      true /* use contact skin if present */);
}

UMochiActorObject* UMochiActorObject::CreateRod(
    UObject* Outer,
    FString ActorName,
    const FTransform& WorldFromLocal,
    mochi::ShapeHandle Shape,
    const UMochiRodMaterialParams* RodMaterialParams,
    const UMochiContactParams* ContactParams,
    EMochiColliderType ColliderType,
    FName Layer,
    bool bHasGravity,
    const mochi::PointCloudColliderParams* PointCloudParams,
    bool bUseContactSkin,
    EMochiActorSegmentElementType ContactElementType,
    EMochiActorBoundaryElementType ContactSkinElementType,
    bool bQuerySurfaceNodePositions,
    bool bQueryNodeNormals,
    bool bQueryContactPoints,
    UMochiSceneObject* Scene) {
  MOCHI_PROFILE_SCOPE();
  if (!VerifyOuterAndScene(Outer, Scene, TEXT("UMochiActorObject"))) {
    return nullptr;
  }
  if (!VerifySceneAcceptsNewActors(Scene, TEXT("UMochiActorObject"), ActorName)) {
    return nullptr;
  }
  mochi::ActorHandle CreatedActorHandle;
  // Reference captures rely on the wait below before this scope ends. It completes queued work
  // only off the async simulation thread; synchronous scenes run this callback inline.
  Scene->QueueOrExecuteCommand([&](mochi::Scene* PhysicsScene) {
    mochi::ErrorLog E{};
    MOCHI_ERROR_RETURN(E);
    const FString NameStr = ActorName + "___RodActor";
    const FString LayerStr = Layer.ToString();
    const mochi::RodActorParams RodParams = CreateRodActorParams(
        TCHAR_TO_ANSI(*NameStr),
        WorldFromLocal,
        TCHAR_TO_ANSI(*LayerStr),
        Shape,
        RodMaterialParams,
        ContactParams,
        bHasGravity,
        ColliderType,
        bUseContactSkin,
        PointCloudParams,
        ContactElementType,
        ContactSkinElementType);

    mochi::Actor* Actor = PhysicsScene->CreateRodActor(RodParams, E);
    MOCHI_ERROR_RETURN(E);
    CreatedActorHandle = Actor->GetHandle();
  });
  Scene->WaitForQueuedCommands();
  return Create_Internal(
      Outer,
      CreatedActorHandle,
      bQuerySurfaceNodePositions,
      false, /* volume positions */
      bQueryNodeNormals,
      bQueryContactPoints,
      false, /* elastic energy */
      nullptr, /* material params */
      Scene,
      false /* use visual mesh if present */,
      true /* use contact skin if present */);
}

UMochiActorObject* UMochiActorObject::CreateFromExistingActor(
    UObject* Outer,
    mochi::ActorHandle MochiActorHandle,
    bool bQuerySurfaceNodePositions,
    bool bQueryVolumeNodePositions,
    bool bQueryNodeNormals,
    bool bQueryContactPoints,
    UMochiSceneObject* Scene,
    bool bUseContactSkinIfPresent) {
  MOCHI_PROFILE_SCOPE();
  if (!VerifyOuterAndScene(Outer, Scene, TEXT("UMochiActorObject"))) {
    return nullptr;
  }
  return Create_Internal(
      Outer,
      MochiActorHandle,
      bQuerySurfaceNodePositions,
      bQueryVolumeNodePositions,
      bQueryNodeNormals,
      bQueryContactPoints,
      false,
      nullptr,
      Scene,
      true /* use visual mesh if present */,
      bUseContactSkinIfPresent);
}

UMochiActorObject* UMochiActorObject::CreateImplicitPlane(
    UObject* Outer,
    FString ActorName,
    const FTransform& WorldFromLocal,
    const FVector& Normal,
    float Distance,
    const UMochiMaterialParams* MaterialParams,
    const UMochiContactParams* ContactParams,
    FName Layer,
    UMochiSceneObject* Scene) {
  MOCHI_PROFILE_SCOPE();
  if (!VerifyOuterAndScene(Outer, Scene, TEXT("UMochiActorObject"))) {
    return nullptr;
  }
  if (!VerifySceneAcceptsNewActors(Scene, TEXT("UMochiActorObject"), ActorName)) {
    return nullptr;
  }
  mochi::Context* const MochiWorld = IMochiPhysicsModule::Get()->GetMochiContext();
  mochi::ActorHandle CreatedActorHandle;
  // Reference captures rely on the wait below before this scope ends. It completes queued work
  // only off the async simulation thread; synchronous scenes run this callback inline.
  Scene->QueueOrExecuteCommand([&](mochi::Scene* PhysicsScene) {
    mochi::ErrorLog E{};
    FUnrealToMochiConverter U2m;
    const mochi::ShapeHandle Shape =
        MochiWorld->CreatePlaneShape(U2m.DirectionToMochi(Normal), U2m.MeasureToMochi(Distance), E);
    MOCHI_ERROR_RETURN(E);
    const FString NameStr = ActorName + "___Plane";
    const FString LayerStr = Layer.ToString();
    const mochi::RigidActorParams ActorParams = CreateRigidActorParams(
        TCHAR_TO_ANSI(*NameStr),
        WorldFromLocal,
        TCHAR_TO_ANSI(*LayerStr),
        Shape,
        MaterialParams,
        ContactParams,
        nullptr,
        EMochiColliderType::Plane,
        true,
        true,
        EMochiActorBoundaryElementType::Default,
        nullptr, // BoundarySubsampling
        nullptr, // MassOverride
        nullptr, // CenterOfMassOverride
        nullptr // MomentOfInertiaOverride
    );
    mochi::Actor* Actor = PhysicsScene->CreateRigidActor(ActorParams, E);
    MOCHI_ERROR_RETURN(E);
    CreatedActorHandle = Actor->GetHandle();
  });
  Scene->WaitForQueuedCommands();
  return Create_Internal(
      Outer,
      CreatedActorHandle,
      false,
      false, /* volume positions */
      false,
      false,
      false,
      MaterialParams,
      Scene);
}

UMochiActorObject* UMochiActorObject::CreateImplicitSphere(
    UObject* Outer,
    FString ActorName,
    const FTransform& WorldFromLocal,
    const FVector& Center,
    float Radius,
    const UMochiMaterialParams* MaterialParams,
    const UMochiContactParams* ContactParams,
    FName Layer,
    UMochiSceneObject* Scene) {
  MOCHI_PROFILE_SCOPE();
  if (!VerifyOuterAndScene(Outer, Scene, TEXT("UMochiActorObject"))) {
    return nullptr;
  }
  if (!VerifySceneAcceptsNewActors(Scene, TEXT("UMochiActorObject"), ActorName)) {
    return nullptr;
  }
  mochi::Context* const MochiWorld = IMochiPhysicsModule::Get()->GetMochiContext();
  mochi::ActorHandle CreatedActorHandle;
  // Reference captures rely on the wait below before this scope ends. It completes queued work
  // only off the async simulation thread; synchronous scenes run this callback inline.
  Scene->QueueOrExecuteCommand([&](mochi::Scene* PhysicsScene) {
    mochi::ErrorLog E{};
    FUnrealToMochiConverter U2m;
    const mochi::ShapeHandle Shape = MochiWorld->CreateSphereShape(
        U2m.TranslationToMochi(Center), U2m.MeasureToMochi(Radius), E);
    MOCHI_ERROR_RETURN(E);
    const FString NameStr = ActorName + "___Sphere";
    const FString LayerStr = Layer.ToString();
    const mochi::RigidActorParams ActorParams = CreateRigidActorParams(
        TCHAR_TO_ANSI(*NameStr),
        WorldFromLocal,
        TCHAR_TO_ANSI(*LayerStr),
        Shape,
        MaterialParams,
        ContactParams,
        nullptr,
        EMochiColliderType::Sphere,
        true,
        true,
        EMochiActorBoundaryElementType::Default,
        nullptr, // BoundarySubsampling
        nullptr, // MassOverride
        nullptr, // CenterOfMassOverride
        nullptr // MomentOfInertiaOverride
    );
    mochi::Actor* Actor = PhysicsScene->CreateRigidActor(ActorParams, E);
    MOCHI_ERROR_RETURN(E);
    CreatedActorHandle = Actor->GetHandle();
  });
  Scene->WaitForQueuedCommands();
  return Create_Internal(
      Outer,
      CreatedActorHandle,
      false,
      false, /* volume positions */
      false,
      false,
      false,
      MaterialParams,
      Scene);
}

UMochiActorObject* UMochiActorObject::Create_Internal(
    UObject* Outer,
    mochi::ActorHandle ActorHandle,
    bool bQuerySurfaceNodePositions,
    bool bQueryVolumeNodePositions,
    bool bQueryNodeNormals,
    bool bQueryContactPoints,
    bool bQueryElasticEnergy,
    const UMochiMaterialParams* MaterialParams,
    UMochiSceneObject* Scene,
    bool bUseVisualMeshIfPresent,
    bool bUseContactSkinIfPresent) {
  MOCHI_PROFILE_SCOPE();
  if (!ActorHandle.IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiActorObject] Failed to create UMochiActorObject because ActorHandle was invalid!"))
    return nullptr;
  }
  FString ActorName;
  // Reference captures rely on the wait below before this scope ends. It completes queued work
  // only off the async simulation thread; synchronous scenes run this callback inline.
  Scene->QueueOrExecuteCommand([&](mochi::Scene* PhysicsScene) {
    mochi::ErrorLog E{};
    if (mochi::Actor* const Actor = PhysicsScene->GetActor(ActorHandle)) {
      if ((bQuerySurfaceNodePositions || bQueryNodeNormals) &&
          Actor->GetType() == mochi::ActorType::Rod &&
          !(bUseVisualMeshIfPresent &&
            Actor->IsQuerySupported(mochi::QueryType::VisualNodePositions)) &&
          !(bUseContactSkinIfPresent &&
            Actor->IsQuerySupported(mochi::QueryType::ContactSkinNodePositions))) {
        UE_LOG(
            LogMochiPhysics,
            Warning,
            TEXT(
                "Rod actor '%hs' has no drawable visual mesh or contact skin and will only be "
                "visible in Mochi debug draw."),
            Actor->GetName());
        bQuerySurfaceNodePositions = false;
        bQueryNodeNormals = false;
      }
      bool bUseVisualMesh = bUseVisualMeshIfPresent && !Actor->GetVisualMesh().IsEmpty();
      bool bUseContactSkin =
          !bUseVisualMesh && bUseContactSkinIfPresent && !Actor->GetContactSkinMesh().IsEmpty();

      // register queries
      if (bQuerySurfaceNodePositions) {
        if (bUseVisualMesh) {
          Actor->RegisterQuery(mochi::QueryType::VisualNodePositions, E);
        } else if (bUseContactSkin) {
          Actor->RegisterQuery(mochi::QueryType::ContactSkinNodePositions, E);
        } else {
          Actor->RegisterQuery(mochi::QueryType::SurfaceNodePositions, E);
        }
        MOCHI_ERROR_RETURN(E);
      }
      if (bQueryVolumeNodePositions) {
        Actor->RegisterQuery(mochi::QueryType::NodePositions, E);
        MOCHI_ERROR_RETURN(E);
      }
      if (bQueryNodeNormals) {
        if (bUseVisualMesh) {
          Actor->RegisterQuery(mochi::QueryType::VisualNodeNormals, E);
        } else if (bUseContactSkin) {
          Actor->RegisterQuery(mochi::QueryType::ContactSkinNodeNormals, E);
        } else {
          Actor->RegisterQuery(mochi::QueryType::SurfaceNodeNormals, E);
        }
        MOCHI_ERROR_RETURN(E);
      }
      if (bQueryContactPoints) {
        Actor->RegisterQuery(mochi::QueryType::ContactPoints, E);
        MOCHI_ERROR_RETURN(E);
      }
      if (bQueryElasticEnergy) {
        Actor->RegisterQuery(mochi::QueryType::ElasticEnergy, E);
        MOCHI_ERROR_RETURN(E);
      }
      // Set the user data to the owning component Outer so that can trace contacts.
      // Note that in the case of CreateFromExistingActor, this may have already
      // been set, so we null check it first.
      if (Actor->GetUserData() == nullptr) {
        Actor->SetUserData(Outer);
      } else if (bQueryContactPoints) {
        const UObject* Obj = reinterpret_cast<UObject*>(Actor->GetUserData());
        const FString ObjName = Obj ? *Obj->GetName() : TEXT("[Unknown]");
        UE_LOG(
            LogMochiPhysics,
            Warning,
            TEXT(
                "[UMochiActorObject] bQueryContactPoints enabled but actor already has Outer named %s in user data. "
                "This will result in contacts not being traced back to the component which created this UMochiActorObject. "
                "Are you using CreateFromExistingActor on an actor handle that was created by another component?"),
            *ObjName);
      }
      ActorName = Actor->GetName();
    }
  });
  Scene->WaitForQueuedCommands();
  UMochiActorObject* MochiActorObj = NewObject<UMochiActorObject>(Outer, FName(*ActorName));
  MochiActorObj->MochiSceneObject = Scene;
  MochiActorObj->MochiMaterialParams = MaterialParams;
  MochiActorObj->MochiActorHandle = ActorHandle;
  MochiActorObj->bQuerySurfaceNodePositions = bQuerySurfaceNodePositions;
  MochiActorObj->bQueryVolumeNodePositions = bQueryVolumeNodePositions;
  MochiActorObj->bQueryNodeNormals = bQueryNodeNormals;
  MochiActorObj->bQueryElasticEnergy = bQueryElasticEnergy;
  MochiActorObj->bQueryContactPoints = bQueryContactPoints;
  MochiActorObj->bUseVisualMeshIfPresent = bUseVisualMeshIfPresent;
  MochiActorObj->bUseContactSkinIfPresent = bUseContactSkinIfPresent;

  // register post step callbacks to extract queried data
  MochiActorObj->ExtractQueryDataCallbackHandle = Scene->RegisterPostStepCallback(
      "UMochiActorObject::ExtractQueryData_Internal", [MochiActorObj](mochi::StepInfo const& Info) {
        mochi::Actor* Actor = Info.scene->GetActor(MochiActorObj->MochiActorHandle);
        MochiActorObj->ExtractQueryData_Internal(Actor, Info.scene);
      });
  return MochiActorObj;
}

bool UMochiActorObject::IsValid() const {
  return MochiSceneObject && MochiSceneObject->IsValid() && MochiActorHandle.IsValid();
}

UMochiSceneObject* UMochiActorObject::GetMochiScene() const {
  return MochiSceneObject;
}

mochi::ActorHandle UMochiActorObject::GetMochiActorHandle() const {
  return MochiActorHandle;
}

void UMochiActorObject::TryGetQueryData(TFunctionRef<void(const FMochiActorQueryData&)> Callback) {
  MOCHI_PROFILE_SCOPE();
  if (bHasPendingQueryData.load()) {
    // Sanity check
    bool bWasIsUpdatingUnrealQueryData = bIsUpdatingUnrealQueryData.exchange(true);
    if (bWasIsUpdatingUnrealQueryData) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiActorObject] TryGetQueryData only supports single-threaded access!"))
      return;
    }
    MOCHI_DEFER(bIsUpdatingUnrealQueryData.store(false));

    // Synchronize
    bool bHasNewData = false;
    {
      FScopeLock Lock(&PendingQueryDataLock);
      if (bHasPendingQueryData) {
        std::swap(PendingQueryData, UnrealThreadQueryData);
        bHasPendingQueryData = false;
        bHasNewData = true;
      }
    }

    if (bHasNewData) {
      // Fix up data here on the Unreal (render) thread. We don't do this part on the Mochi
      // simulation thread because it requires access to the Unreal components and actors.
      if (UnrealThreadQueryData->ContactPoints.Num() > 0) {
        UObject* const Outer = GetOuter();
        for (FMochiContactPoint& Contact : UnrealThreadQueryData->ContactPoints) {
          Contact.MochiActor =
              Contact.MochiComponent ? Contact.MochiComponent->GetOwner() : nullptr;
          Contact.OtherMochiActor =
              Contact.OtherMochiComponent ? Contact.OtherMochiComponent->GetOwner() : nullptr;
          // There's no guarantee that OtherMochiActor is registered.
          // ensure(contact.MochiActor && contact.OtherMochiActor);
          constexpr bool bVerboseContactWarnings = false;
          if (bVerboseContactWarnings) {
            if (!Contact.MochiActor || !Contact.OtherMochiActor || !Contact.MochiComponent ||
                !Contact.MochiComponent) {
              FString AA = Contact.MochiActor ? *Contact.MochiActor->GetName() : TEXT("NULL");
              FString CA =
                  Contact.MochiComponent ? *Contact.MochiComponent->GetName() : TEXT("NULL");
              FString MA = Contact.InternalMochiActor.ToString();
              FString AB =
                  Contact.OtherMochiActor ? *Contact.OtherMochiActor->GetName() : TEXT("NULL");
              FString CB = Contact.OtherMochiComponent ? *Contact.OtherMochiComponent->GetName()
                                                       : TEXT("NULL");
              FString MB = Contact.OtherInternalMochiActor.ToString();
              UE_LOG(
                  LogMochiPhysics,
                  Warning,
                  TEXT(
                      "[UMochiArticulatedActorObject] Untraceable contact detected: "
                      "AActorA=%s : USceneComponentA=%s : MochiActorA=%s  <---> AActorB=%s : USceneComponentB=%s : MochiActorB=%s"),
                  *AA,
                  *CA,
                  *MA,
                  *AB,
                  *CB,
                  *MB);
              continue;
            }
            if (Contact.MochiComponent && Contact.MochiComponent == Contact.OtherMochiComponent) {
              UE_LOG(
                  LogMochiPhysics,
                  Warning,
                  TEXT("[UMochiArticulatedActorObject] Self-contact detected on component %s"),
                  *Contact.MochiComponent->GetName());
            }
          }
          if (Contact.MochiComponent != Outer) {
            // Swap the direction of this contact (by convention)
            ensure(Contact.OtherMochiComponent == Outer);
            std::swap(Contact.InternalMochiActor, Contact.OtherInternalMochiActor);
            std::swap(Contact.MochiComponent, Contact.OtherMochiComponent);
            std::swap(Contact.MochiActor, Contact.OtherMochiActor);
            std::swap(Contact.Location, Contact.OtherLocation);
            std::swap(Contact.Velocity, Contact.OtherVelocity);
            Contact.Normal = -Contact.Normal;
            Contact.Force = -Contact.Force;

            // Unambiguously record that this is a swapped (other->this) contact,
            // for clients that need to process contacts in an asymmetric manner
            Contact.ContactDirection = EMochiContactDirection::OtherContactsThis;
          }
        }
      }

      // Finally, pass the data back to the caller
      Callback(*UnrealThreadQueryData);
    }
  }
}

void UMochiActorObject::SetRootTransform(const FTransform& WorldFromLocal, bool bBlocking) const {
  MOCHI_PROFILE_SCOPE();
  if (!IsValid()) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("[UMochiActorObject] UMochiActorObject is invalid!"))
    return;
  }
  MochiSceneObject->QueueOrExecuteActorCommand(
      MochiActorHandle, [WorldFromLocal](mochi::Actor* Actor) {
        mochi::ErrorLog E{};
        FUnrealToMochiConverter U2m;
        Actor->SetRootTransform(U2m.TransformToMochi_RT(WorldFromLocal), E);
      });
  if (bBlocking) {
    MochiSceneObject->WaitForQueuedCommands();
  }
}

void UMochiActorObject::SetContactParams(const UMochiContactParams* ContactParams, bool bBlocking)
    const {
  MOCHI_PROFILE_SCOPE();
  if (!IsValid()) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("[UMochiActorObject] UMochiActorObject is invalid!"))
    return;
  }
  MochiSceneObject->QueueOrExecuteActorCommand(
      MochiActorHandle, [ContactParams](mochi::Actor* Actor) {
        mochi::ErrorLog E{};
        Actor->SetContactParams(ContactParams->ToMochi(), E);
      });
  if (bBlocking) {
    MochiSceneObject->WaitForQueuedCommands();
  }
}

void UMochiActorObject::SetMaterialParams(
    const UMochiMaterialParams* MaterialParams,
    bool bBlocking) {
  MOCHI_PROFILE_SCOPE();
  if (!IsValid()) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("[UMochiActorObject] UMochiActorObject is invalid!"))
    return;
  }
  MochiSceneObject->QueueOrExecuteActorCommand(
      MochiActorHandle, [MaterialParams](mochi::Actor* Actor) {
        mochi::ErrorLog E{};
        if (Actor->GetType() == mochi::ActorType::Soft) {
          Actor->SetSoftMaterialParams(MaterialParams->ToMochi(), E);
        } else if (Actor->GetType() == mochi::ActorType::Rigid) {
          Actor->SetDensity(MaterialParams->Density, E);
        }
      });
  if (bBlocking) {
    MochiSceneObject->WaitForQueuedCommands();
  }
  // Cache a local copy for GetMaterialParams()
  MochiMaterialParams = MaterialParams;
}

void UMochiActorObject::ApplyMaterialModifiers(
    const TMap<int, FMochiMaterialModifier>& Modifiers,
    bool bResetExisting,
    bool bBlocking) {
  MOCHI_PROFILE_SCOPE();
  if (!IsValid()) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("[UMochiActorObject] UMochiActorObject is invalid!"))
    return;
  }

  if (Modifiers.Num() > 0 || bResetExisting) {
    // create a "template" of material params, because if the type doesn't match
    // the original material SetSoftMaterialParamsField will fail
    const UMochiMaterialParams* ModelMaterialParams = GetMaterialParams();
    MochiSceneObject->QueueOrExecuteActorCommand(
        MochiActorHandle, [ModelMaterialParams, Modifiers, bResetExisting](mochi::Actor* Actor) {
          mochi::SoftMaterialParams Params = ModelMaterialParams->ToMochi();
          mochi::ErrorLog E{};
          if (bResetExisting) {
            if (Actor->GetType() == mochi::ActorType::Soft) {
              Actor->SetSoftMaterialParams(Params, E);
            } else if (Actor->GetType() == mochi::ActorType::Rigid) {
              Actor->SetDensity(Params.density, E);
            }
          }
          if (Actor->GetType() == mochi::ActorType::Soft) {
            // Modifiers only supported for soft actors.
            for (const TPair<int, FMochiMaterialModifier> Modifier : Modifiers) {
              // TODO: should we check for zeros/negative?
              // indices are already checked in mochi
              constexpr mochi::real MPaToPa = 1e6_r;
              int Idx = Modifier.Key;
              FMochiMaterialModifier Values = Modifier.Value;
              const mochi::real Ym = Values.YoungsModulus * MPaToPa;
              const float Pr = Values.PoissonRatio;
              static_assert(
                  static_cast<int>(mochi::SoftMaterialType::Count) == 6,
                  "Please update the switch statement below if SoftMaterialType enum changes");
              switch (Params.type) {
                case mochi::SoftMaterialType::StVenantKirchhoff:
                  Params.stVenantKirchhoff.youngsModulus = Ym;
                  Params.stVenantKirchhoff.poissonRatio = Pr;
                  break;
                case mochi::SoftMaterialType::LinearElastic:
                  Params.linearElastic.youngsModulus = Ym;
                  Params.linearElastic.poissonRatio = Pr;
                  break;
                case mochi::SoftMaterialType::ActiveNeoHookean:
                  Params.activeNeoHookean.passiveIsotropic.youngsModulus = Ym;
                  Params.activeNeoHookean.passiveIsotropic.poissonRatio = Pr;
                  break;
                case mochi::SoftMaterialType::NeoHookean:
                  Params.neoHookean.youngsModulus = Ym;
                  Params.neoHookean.poissonRatio = Pr;
                  break;
                // Both are pure ARAP, parameterised by a stiffness rather than by a
                // Young's modulus and Poisson ratio, so a modifier has nothing to say
                // about them.
                case mochi::SoftMaterialType::ActiveShapeTargetingArap:
                case mochi::SoftMaterialType::Arap:
                  break;
                case mochi::SoftMaterialType::Count:
                  checkNoEntry();
                  break;
              }
              mochi::experimental::SetSoftMaterialParamsField(Actor, Params, Idx, E);
            }
          }
        });
    if (bBlocking) {
      MochiSceneObject->WaitForQueuedCommands();
    }
  }
}

const UMochiMaterialParams* UMochiActorObject::GetMaterialParams() const {
  return MochiMaterialParams;
}

void UMochiActorObject::SetVelocity(FVector LinearVelocity, FVector AngularVelocity) const {
  MOCHI_PROFILE_SCOPE();
  if (!IsValid()) {
    return;
  }

  MochiSceneObject->QueueOrExecuteActorCommand(
      MochiActorHandle, [LinearVelocity, AngularVelocity](mochi::Actor* Actor) {
        FUnrealToMochiConverter U2m;
        Actor->SetVelocity(
            U2m.TranslationToMochi(LinearVelocity),
            U2m.DirectionToMochi(AngularVelocity),
            mochi::ErrorLog{});
      });
}

void UMochiActorObject::GetInertialParams(
    double& Mass,
    FVector& CenterOfMass,
    FMochiMomentOfInertia& MomentOfInertia) const {
  MochiSceneObject->QueueOrExecuteActorCommand(
      MochiActorHandle, [&Mass, &CenterOfMass, &MomentOfInertia](mochi::Actor* Actor) {
        mochi::ErrorLog E{};
        Mass = Actor->GetMass(E);
        FMochiToUnrealConverter M2u;
        CenterOfMass = M2u.TranslationToUnreal(Actor->GetRigidCenterOfMassLocal(E));
        mochi::Real6 Moi = Actor->GetRigidMomentOfInertiaLocal(E);
        MomentOfInertia = M2u.InertiaToUnreal(Moi);
        MomentOfInertia.Ixy = M2u.MeasureToUnreal(Moi[1], 2);
        MomentOfInertia.Ixz = M2u.MeasureToUnreal(Moi[2], 2);
        MomentOfInertia.Iyy = M2u.MeasureToUnreal(Moi[3], 2);
        MomentOfInertia.Iyz = M2u.MeasureToUnreal(Moi[4], 2);
        MomentOfInertia.Izz = M2u.MeasureToUnreal(Moi[5], 2);
      });
  MochiSceneObject->WaitForQueuedCommands();
}

void UMochiActorObject::DestroyMochiResources() {
  MOCHI_PROFILE_SCOPE();
  if (MochiSceneObject) {
    MochiSceneObject->QueueOrExecuteCommand([this](mochi::Scene* Scene) {
      Scene->CancelCallback(ExtractQueryDataCallbackHandle);
      if (bDestroyMochiActorOnObjectDestruction) {
        Scene->DestroyActor(MochiActorHandle);
      }
    });
    MochiSceneObject->WaitForQueuedCommands();
    MochiSceneObject = nullptr;
  }
  ExtractQueryDataCallbackHandle = {};
  MochiActorHandle = {};
}

void UMochiActorObject::BeginDestroy() {
  MOCHI_PROFILE_SCOPE();
  UObject::BeginDestroy();
  DestroyMochiResources();
}

// Helper
template <typename Elem>
static void CopyToUnrealArray(mochi::Span<Elem const> In, TArray<Elem>& Out, mochi::Error& Error) {
  if (Error.IsOK()) {
    Out.SetNum(In.size());
    std::copy(In.data(), In.data() + In.size(), Out.GetData());
  } else {
    Out.SetNum(0);
  }
}

void UMochiActorObject::ExtractQueryData_Internal(const mochi::Actor* Actor, mochi::Scene* Scene) {
  MOCHI_PROFILE_SCOPE();
  if (Actor == nullptr || Scene == nullptr) {
    return;
  }
  mochi::ErrorLog Err;

  // extract root transform
  FMochiToUnrealConverter M2u;
  MochiThreadQueryData->RootTransform = M2u.TransformToUnreal(Actor->GetRootTransform());

  // We may only query velocity from rigid and soft bodies, or individual bones within an
  // articulated actor. Invalid calls will throw an error.
  mochi::ActorType ActorType = Actor->GetType();
  if (ActorType == mochi::ActorType::Rigid || ActorType == mochi::ActorType::Soft) {
    MochiThreadQueryData->LinearVelocity =
        M2u.TranslationToUnreal(Actor->GetLinearVelocity(mochi::ErrorLog{}));
    MochiThreadQueryData->AngularVelocity =
        M2u.DirectionToUnreal(Actor->GetAngularVelocity(mochi::ErrorLog{}));
  }
  const bool bUseVisualMesh = bUseVisualMeshIfPresent && !Actor->GetVisualMesh().IsEmpty();
  const bool bUseContactSkin =
      !bUseVisualMesh && bUseContactSkinIfPresent && !Actor->GetContactSkinMesh().IsEmpty();
  // extract node positions
  if (bQuerySurfaceNodePositions) {
    mochi::Span<const mochi::real> MochiNodePositions;
    if (bUseVisualMesh) {
      MochiNodePositions = Actor->GetVisualMeshNodePositionsLocal(Err);
    } else if (bUseContactSkin) {
      MochiNodePositions = Actor->GetContactSkinMeshNodePositionsLocal(Err);
    } else {
      MochiNodePositions = Actor->GetSurfaceMeshNodePositionsLocal(Err);
    }
    CopyToUnrealArray(MochiNodePositions, MochiThreadQueryData->SurfaceNodePositions, Err);
  }
  if (bQueryVolumeNodePositions) {
    mochi::Span<const mochi::real> MochiVolumeNodePositions = Actor->GetNodePositionsLocal(Err);
    CopyToUnrealArray(MochiVolumeNodePositions, MochiThreadQueryData->VolumeNodePositions, Err);
  }
  // extract node normals
  if (bQueryNodeNormals) {
    mochi::Span<const mochi::real> MochiNodeNormals;
    if (bUseVisualMesh) {
      MochiNodeNormals = Actor->GetVisualMeshNodeNormalsLocal(Err);
    } else if (bUseContactSkin) {
      MochiNodeNormals = Actor->GetContactSkinMeshNodeNormalsLocal(Err);
    } else {
      MochiNodeNormals = Actor->GetSurfaceMeshNodeNormalsLocal(Err);
    }
    CopyToUnrealArray(MochiNodeNormals, MochiThreadQueryData->NodeNormals, Err);
  }

  if (bQueryElasticEnergy) {
    mochi::real Energy = Actor->GetElasticEnergy(Err);
    MochiThreadQueryData->ElasticEnergy = Energy;
  }

  // extract contact points
  if (bQueryContactPoints) {
    const mochi::Span<const mochi::ContactPoint> ContactPoints = Actor->GetContactPointsWorld(Err);
    if (Err.IsOK()) {
      MOCHI_PROFILE_SCOPE_N("Convert Contact Points For Unreal");
      // TODO: Convert the contacts on the game thread to take this work off the Mochi thread.
      // ContactPointToUnreal looks both actors up in the Mochi scene, so capture what it reads
      // from them here first.
      MochiThreadQueryData->ContactPoints.SetNum(ContactPoints.size());
      for (int ContactIndex = 0; ContactIndex < ContactPoints.size(); ++ContactIndex) {
        const mochi::ContactPoint& MochiContact = ContactPoints[ContactIndex];
        FMochiContactPoint& UnrealContact = MochiThreadQueryData->ContactPoints[ContactIndex];
        ContactPointToUnreal(MochiContact, Scene, NAME_None, M2u, UnrealContact);
      }
    }
  }

  // Synchronize
  {
    FScopeLock Lock(&PendingQueryDataLock);
    std::swap(MochiThreadQueryData, PendingQueryData);
    bHasPendingQueryData = true;
  }
}
