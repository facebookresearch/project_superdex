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
  mochi::RigidActorParams actorParams;
  FUnrealToMochiConverter u2m;
  actorParams.name = Name;
  actorParams.shape = Shape;
  actorParams.isStatic = bIsStatic;
  actorParams.hasGravity = bHasGravity;
  actorParams.worldFromLocal = u2m.TransformToMochi_RT(WorldFromLocal);
  actorParams.colliderType = mochi::ColliderType(ColliderType);
  actorParams.layer = LayerStr;
  if (ContactParams) {
    actorParams.contact = ContactParams->ToMochi();
  }
  if (GridSdfParams) {
    actorParams.sdf = GridSdfParams->ToMochi();
  }

  // Set mass or density, but not both.
  if (MassOverride) {
    actorParams.mass = *MassOverride;
  } else if (MaterialParams) {
    actorParams.density = MaterialParams->Density;
  }

  if (CenterOfMassOverride && MomentOfInertiaOverride) {
    actorParams.centerOfMass = u2m.TranslationToMochi(*CenterOfMassOverride);
    actorParams.momentOfInertia = u2m.InertiaToMochi(*MomentOfInertiaOverride);
  }
  const auto ActorBoundaryElementType = ToMochi(BoundaryElementType);
  if (ActorBoundaryElementType) {
    actorParams.boundaryElementType = *ActorBoundaryElementType;
  }
  if (BoundarySubsampling) {
    actorParams.boundarySubsampling = ToMochi(*BoundarySubsampling);
  }
  return actorParams;
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
  FUnrealToMochiConverter u2m;
  mochi::SoftActorParams actorParams;
  actorParams.name = Name;
  actorParams.shape = Shape;
  actorParams.worldFromLocal = u2m.TransformToMochi_RT(WorldFromLocal);
  actorParams.hasGravity = bHasGravity;
  actorParams.useContactSkin = bUseContactSkin;
  actorParams.layer = LayerStr;
  if (MaterialParams) {
    actorParams.material = MaterialParams->ToMochi();
  }
  if (ContactParams) {
    actorParams.contact = ContactParams->ToMochi();
  }
  const auto ActorBoundaryElementType = ToMochi(BoundaryElementType);
  if (ActorBoundaryElementType) {
    actorParams.boundaryElementType = *ActorBoundaryElementType;
  }
  return actorParams;
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
    const mochi::experimental::PointCloudColliderParams* PointCloudParams,
    bool bUseContactSkin,
    EMochiActorBoundaryElementType ContactElementType) {
  MOCHI_PROFILE_SCOPE();
  FUnrealToMochiConverter u2m;
  mochi::experimental::ShellActorParams shellParams;
  shellParams.name = Name;
  shellParams.layer = LayerStr;
  shellParams.worldFromLocal = u2m.TransformToMochi_RT(WorldFromLocal);
  shellParams.shape = Shape;
  shellParams.hasGravity = bHasGravity;
  shellParams.colliderType = mochi::ColliderType(ColliderType);
  shellParams.useContactSkin = bUseContactSkin;
  if (ShellMaterialParams) {
    shellParams.material = ShellMaterialParams->ToMochi();
  }
  if (ContactParams) {
    shellParams.contact = ContactParams->ToMochi();
  }
  if (PointCloudParams) {
    shellParams.pointCloudCollider = *PointCloudParams;
  }
  const auto ActorContactElementType = ToMochi(ContactElementType);
  if (ActorContactElementType) {
    shellParams.contactElementType = *ActorContactElementType;
  }
  return shellParams;
}

mochi::experimental::RodActorParams CreateRodActorParams(
    const char* Name,
    const FTransform& WorldFromLocal,
    const char* LayerStr,
    const mochi::ShapeHandle Shape,
    const UMochiRodMaterialParams* RodMaterialParams,
    const UMochiContactParams* ContactParams,
    bool bHasGravity,
    EMochiColliderType ColliderType,
    bool bUseContactSkin,
    const mochi::experimental::PointCloudColliderParams* PointCloudParams,
    EMochiActorSegmentElementType ContactElementType,
    EMochiActorBoundaryElementType ContactSkinElementType) {
  MOCHI_PROFILE_SCOPE();
  FUnrealToMochiConverter u2m;
  mochi::experimental::RodActorParams rodParams;
  rodParams.name = Name;
  rodParams.layer = LayerStr;
  rodParams.worldFromLocal = u2m.TransformToMochi_RT(WorldFromLocal);
  rodParams.shape = Shape;
  rodParams.hasGravity = bHasGravity;
  rodParams.colliderType = mochi::ColliderType(ColliderType);
  rodParams.useContactSkin = bUseContactSkin;
  if (RodMaterialParams) {
    rodParams.material = RodMaterialParams->ToMochi();
  }
  if (ContactParams) {
    rodParams.contact = ContactParams->ToMochi();
  }
  if (PointCloudParams) {
    rodParams.pointCloudCollider = *PointCloudParams;
  }
  const auto ActorContactElementType = ToMochi(ContactElementType);
  if (ActorContactElementType) {
    rodParams.contactElementType = *ActorContactElementType;
  }
  const auto ActorContactSkinElementType = ToMochi(ContactSkinElementType);
  if (ActorContactSkinElementType) {
    rodParams.contactSkinElementType = *ActorContactSkinElementType;
  }
  return rodParams;
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
  const auto shape = Model->GetMochiShape(WorldFromLocal.GetScale3D(), FTransform::Identity);

  return CreateRigidFromMochiShape(
      Outer,
      ActorName,
      WorldFromLocal,
      shape,
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
  const auto shape = UMochiModel::GetTriangularMochiShapeFromStaticMesh(
      SurfaceMesh, CollisionSource, WorldFromLocal.GetScale3D());
  if (shape.IsValid()) {
    return CreateRigidFromMochiShape(
        Outer,
        ActorName,
        WorldFromLocal,
        shape,
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
    const mochi::ShapeHandle shape,
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
  const auto mochiWorld = IMochiPhysicsModule::Get()->GetMochiContext();
  mochi::ActorHandle mochiActorHandle;
  Scene->QueueOrExecuteCommand([&](mochi::Scene* scene) {
    mochi::ErrorLog e{};
    MOCHI_ERROR_RETURN(e);
    const auto nameStr = ActorName + "___Actor";
    const auto layerStr = Layer.ToString();
    const auto actorParams = CreateRigidActorParams(
        TCHAR_TO_ANSI(*nameStr),
        WorldFromLocal,
        TCHAR_TO_ANSI(*layerStr),
        shape,
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
    auto* actor = scene->CreateRigidActor(actorParams, e);
    MOCHI_ERROR_RETURN(e);
    mochiActorHandle = actor->GetHandle();
  });
  Scene->WaitForQueuedCommands();
  return Create_Internal(
      Outer,
      mochiActorHandle,
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
  const auto mochiWorld = IMochiPhysicsModule::Get()->GetMochiContext();
  mochi::ActorHandle mochiActorHandle;
  const auto shape = Model->GetMochiShape(WorldFromLocal.GetScale3D(), FTransform::Identity);
  Scene->QueueOrExecuteCommand([&](mochi::Scene* scene) {
    mochi::ErrorLog e{};
    MOCHI_ERROR_RETURN(e);
    const auto nameStr = ActorName + "___SoftActor";
    const auto layerStr = Layer.ToString();
    const auto actorParams = CreateSoftActorParams(
        TCHAR_TO_ANSI(*nameStr),
        WorldFromLocal,
        TCHAR_TO_ANSI(*layerStr),
        shape,
        MaterialParams,
        ContactParams,
        bHasGravity,
        bUseContactSkin,
        BoundaryElementType);

    mochi::experimental::ExperimentalSoftActorParams experimentalParams;
    experimentalParams.useRecentering = bUseRecentering;
    experimentalParams.colliderType = mochi::ColliderType(ColliderType);

    if (RomParams) {
      experimentalParams.rom = RomParams->ToMochi();
    }

    auto* actor = mochi::experimental::CreateSoftActor(scene, actorParams, experimentalParams, e);
    MOCHI_ERROR_RETURN(e);
    mochiActorHandle = actor->GetHandle();
  });
  Scene->WaitForQueuedCommands();
  return Create_Internal(
      Outer,
      mochiActorHandle,
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
    const mochi::experimental::PointCloudColliderParams* PointCloudParams,
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
  mochi::ActorHandle mochiActorHandle;
  Scene->QueueOrExecuteCommand([&](mochi::Scene* scene) {
    mochi::ErrorLog e{};
    MOCHI_ERROR_RETURN(e);
    const auto nameStr = ActorName + "___ShellActor";
    const auto layerStr = Layer.ToString();
    const auto shellParams = CreateShellActorParams(
        TCHAR_TO_ANSI(*nameStr),
        WorldFromLocal,
        TCHAR_TO_ANSI(*layerStr),
        Shape,
        ShellMaterialParams,
        ContactParams,
        bHasGravity,
        ColliderType,
        PointCloudParams,
        bUseContactSkin,
        ContactElementType);

    auto* actor = mochi::experimental::CreateShellActor(scene, shellParams, e);
    MOCHI_ERROR_RETURN(e);
    mochiActorHandle = actor->GetHandle();
  });
  Scene->WaitForQueuedCommands();
  return Create_Internal(
      Outer,
      mochiActorHandle,
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
    const mochi::experimental::PointCloudColliderParams* PointCloudParams,
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
  mochi::ActorHandle mochiActorHandle;
  Scene->QueueOrExecuteCommand([&](mochi::Scene* scene) {
    mochi::ErrorLog e{};
    MOCHI_ERROR_RETURN(e);
    const auto nameStr = ActorName + "___RodActor";
    const auto layerStr = Layer.ToString();
    const auto rodParams = CreateRodActorParams(
        TCHAR_TO_ANSI(*nameStr),
        WorldFromLocal,
        TCHAR_TO_ANSI(*layerStr),
        Shape,
        RodMaterialParams,
        ContactParams,
        bHasGravity,
        ColliderType,
        bUseContactSkin,
        PointCloudParams,
        ContactElementType,
        ContactSkinElementType);

    auto* actor = mochi::experimental::CreateRodActor(scene, rodParams, e);
    MOCHI_ERROR_RETURN(e);
    mochiActorHandle = actor->GetHandle();
  });
  Scene->WaitForQueuedCommands();
  return Create_Internal(
      Outer,
      mochiActorHandle,
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
  const auto mochiWorld = IMochiPhysicsModule::Get()->GetMochiContext();
  mochi::ActorHandle mochiActorHandle;
  Scene->QueueOrExecuteCommand([&](mochi::Scene* scene) {
    mochi::ErrorLog e{};
    FUnrealToMochiConverter u2m;
    const auto shape =
        mochiWorld->CreatePlaneShape(u2m.DirectionToMochi(Normal), u2m.MeasureToMochi(Distance), e);
    MOCHI_ERROR_RETURN(e);
    const auto nameStr = ActorName + "___Plane";
    const auto layerStr = Layer.ToString();
    const auto actorParams = CreateRigidActorParams(
        TCHAR_TO_ANSI(*nameStr),
        WorldFromLocal,
        TCHAR_TO_ANSI(*layerStr),
        shape,
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
    auto* actor = scene->CreateRigidActor(actorParams, e);
    MOCHI_ERROR_RETURN(e);
    mochiActorHandle = actor->GetHandle();
  });
  Scene->WaitForQueuedCommands();
  return Create_Internal(
      Outer,
      mochiActorHandle,
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
  const auto mochiWorld = IMochiPhysicsModule::Get()->GetMochiContext();
  mochi::ActorHandle mochiActorHandle;
  Scene->QueueOrExecuteCommand([&](mochi::Scene* scene) {
    mochi::ErrorLog e{};
    FUnrealToMochiConverter u2m;
    const auto shape = mochiWorld->CreateSphereShape(
        u2m.TranslationToMochi(Center), u2m.MeasureToMochi(Radius), e);
    MOCHI_ERROR_RETURN(e);
    const auto nameStr = ActorName + "___Sphere";
    const auto layerStr = Layer.ToString();
    const auto actorParams = CreateRigidActorParams(
        TCHAR_TO_ANSI(*nameStr),
        WorldFromLocal,
        TCHAR_TO_ANSI(*layerStr),
        shape,
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
    auto* actor = scene->CreateRigidActor(actorParams, e);
    MOCHI_ERROR_RETURN(e);
    mochiActorHandle = actor->GetHandle();
  });
  Scene->WaitForQueuedCommands();
  return Create_Internal(
      Outer,
      mochiActorHandle,
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
  FString actorName;
  Scene->QueueOrExecuteCommand([&](mochi::Scene* scene) {
    mochi::ErrorLog e{};
    if (const auto actor = scene->GetActor(ActorHandle)) {
      bool useVisualMesh = bUseVisualMeshIfPresent && !actor->GetVisualMesh().IsEmpty();
      bool useContactSkin =
          !useVisualMesh && bUseContactSkinIfPresent && !actor->GetContactSkinMesh().IsEmpty();

      // register queries
      if (bQuerySurfaceNodePositions) {
        if (useVisualMesh) {
          actor->RegisterQuery(mochi::QueryType::VisualNodePositions, e);
        } else if (useContactSkin) {
          actor->RegisterQuery(mochi::QueryType::ContactSkinNodePositions, e);
        } else {
          actor->RegisterQuery(mochi::QueryType::SurfaceNodePositions, e);
        }
        MOCHI_ERROR_RETURN(e);
      }
      if (bQueryVolumeNodePositions) {
        actor->RegisterQuery(mochi::QueryType::NodePositions, e);
        MOCHI_ERROR_RETURN(e);
      }
      if (bQueryNodeNormals) {
        if (useVisualMesh) {
          actor->RegisterQuery(mochi::QueryType::VisualNodeNormals, e);
        } else if (useContactSkin) {
          actor->RegisterQuery(mochi::QueryType::ContactSkinNodeNormals, e);
        } else {
          actor->RegisterQuery(mochi::QueryType::SurfaceNodeNormals, e);
        }
        MOCHI_ERROR_RETURN(e);
      }
      if (bQueryContactPoints) {
        actor->RegisterQuery(mochi::QueryType::ContactPoints, e);
        MOCHI_ERROR_RETURN(e);
      }
      if (bQueryElasticEnergy) {
        actor->RegisterQuery(mochi::QueryType::ElasticEnergy, e);
        MOCHI_ERROR_RETURN(e);
      }
      // Set the user data to the owning component Outer so that can trace contacts.
      // Note that in the case of CreateFromExistingActor, this may have already
      // been set, so we null check it first.
      if (actor->GetUserData() == nullptr) {
        actor->SetUserData(Outer);
      } else if (bQueryContactPoints) {
        const UObject* obj = reinterpret_cast<UObject*>(actor->GetUserData());
        const FString objName = obj ? *obj->GetName() : TEXT("[Unknown]");
        UE_LOG(
            LogMochiPhysics,
            Warning,
            TEXT(
                "[UMochiActorObject] bQueryContactPoints enabled but actor already has Outer named %s in user data. "
                "This will result in contacts not being traced back to the component which created this UMochiActorObject. "
                "Are you using CreateFromExistingActor on an actor handle that was created by another component?"),
            *objName);
      }
      actorName = actor->GetName();
    }
  });
  Scene->WaitForQueuedCommands();
  auto mochiActorObj = NewObject<UMochiActorObject>(Outer, FName(*actorName));
  mochiActorObj->MochiSceneObject = Scene;
  mochiActorObj->MochiMaterialParams = MaterialParams;
  mochiActorObj->MochiActorHandle = ActorHandle;
  mochiActorObj->bQuerySurfaceNodePositions = bQuerySurfaceNodePositions;
  mochiActorObj->bQueryVolumeNodePositions = bQueryVolumeNodePositions;
  mochiActorObj->bQueryNodeNormals = bQueryNodeNormals;
  mochiActorObj->bQueryElasticEnergy = bQueryElasticEnergy;
  mochiActorObj->bQueryContactPoints = bQueryContactPoints;
  mochiActorObj->bUseVisualMeshIfPresent = bUseVisualMeshIfPresent;
  mochiActorObj->bUseContactSkinIfPresent = bUseContactSkinIfPresent;

  // register post step callbacks to extract queried data
  mochiActorObj->ExtractQueryDataCallbackHandle = Scene->RegisterPostStepCallback(
      "UMochiActorObject::ExtractQueryData_Internal", [mochiActorObj](mochi::StepInfo const& info) {
        mochi::Actor* actor = info.scene->GetActor(mochiActorObj->MochiActorHandle);
        mochiActorObj->ExtractQueryData_Internal(actor, info.scene);
      });
  return mochiActorObj;
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
    bool wasIsUpdatingUnrealQueryData = bIsUpdatingUnrealQueryData.exchange(true);
    if (wasIsUpdatingUnrealQueryData) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiActorObject] TryGetQueryData only supports single-threaded access!"))
      return;
    }
    MOCHI_DEFER(bIsUpdatingUnrealQueryData.store(false));

    // Synchronize
    bool hasNewData = false;
    {
      FScopeLock lock(&PendingQueryDataLock);
      if (bHasPendingQueryData) {
        std::swap(PendingQueryData, UnrealThreadQueryData);
        bHasPendingQueryData = false;
        hasNewData = true;
      }
    }

    if (hasNewData) {
      // Fix up data here on the Unreal (render) thread. We don't do this part on the Mochi
      // simulation thread because it requires access to the Unreal components and actors.
      if (UnrealThreadQueryData->ContactPoints.Num() > 0) {
        const auto outer = GetOuter();
        for (auto& contact : UnrealThreadQueryData->ContactPoints) {
          contact.MochiActor =
              contact.MochiComponent ? contact.MochiComponent->GetOwner() : nullptr;
          contact.OtherMochiActor =
              contact.OtherMochiComponent ? contact.OtherMochiComponent->GetOwner() : nullptr;
          // There's no guarantee that OtherMochiActor is registered.
          // ensure(contact.MochiActor && contact.OtherMochiActor);
          constexpr bool bVerboseContactWarnings = false;
          if (bVerboseContactWarnings) {
            if (!contact.MochiActor || !contact.OtherMochiActor || !contact.MochiComponent ||
                !contact.MochiComponent) {
              FString aA = contact.MochiActor ? *contact.MochiActor->GetName() : TEXT("NULL");
              FString cA =
                  contact.MochiComponent ? *contact.MochiComponent->GetName() : TEXT("NULL");
              FString mA = contact.InternalMochiActor.ToString();
              FString aB =
                  contact.OtherMochiActor ? *contact.OtherMochiActor->GetName() : TEXT("NULL");
              FString cB = contact.OtherMochiComponent ? *contact.OtherMochiComponent->GetName()
                                                       : TEXT("NULL");
              FString mB = contact.OtherInternalMochiActor.ToString();
              UE_LOG(
                  LogMochiPhysics,
                  Warning,
                  TEXT(
                      "[UMochiArticulatedActorObject] Untraceable contact detected: "
                      "AActorA=%s : USceneComponentA=%s : MochiActorA=%s  <---> AActorB=%s : USceneComponentB=%s : MochiActorB=%s"),
                  *aA,
                  *cA,
                  *mA,
                  *aB,
                  *cB,
                  *mB);
              continue;
            }
            if (contact.MochiComponent && contact.MochiComponent == contact.OtherMochiComponent) {
              UE_LOG(
                  LogMochiPhysics,
                  Warning,
                  TEXT("[UMochiArticulatedActorObject] Self-contact detected on component %s"),
                  *contact.MochiComponent->GetName());
            }
          }
          if (contact.MochiComponent != outer) {
            // Swap the direction of this contact (by convention)
            ensure(contact.OtherMochiComponent == outer);
            std::swap(contact.InternalMochiActor, contact.OtherInternalMochiActor);
            std::swap(contact.MochiComponent, contact.OtherMochiComponent);
            std::swap(contact.MochiActor, contact.OtherMochiActor);
            std::swap(contact.Location, contact.OtherLocation);
            std::swap(contact.Velocity, contact.OtherVelocity);
            contact.Normal = -contact.Normal;
            contact.Force = -contact.Force;

            // Unambiguously record that this is a swapped (other->this) contact,
            // for clients that need to process contacts in an asymmetric manner
            contact.ContactDirection = EMochiContactDirection::OtherContactsThis;
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
      MochiActorHandle, [WorldFromLocal](mochi::Actor* actor) {
        mochi::ErrorLog e{};
        FUnrealToMochiConverter u2m;
        actor->SetRootTransform(u2m.TransformToMochi_RT(WorldFromLocal), e);
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
      MochiActorHandle, [ContactParams](mochi::Actor* actor) {
        mochi::ErrorLog e{};
        actor->SetContactParams(ContactParams->ToMochi(), e);
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
      MochiActorHandle, [MaterialParams](mochi::Actor* actor) {
        mochi::ErrorLog e{};
        if (actor->GetType() == mochi::ActorType::Soft) {
          actor->SetSoftMaterialParams(MaterialParams->ToMochi(), e);
        } else if (actor->GetType() == mochi::ActorType::Rigid) {
          actor->SetDensity(MaterialParams->Density, e);
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
    auto ModelMaterialParams = GetMaterialParams();
    MochiSceneObject->QueueOrExecuteActorCommand(
        MochiActorHandle, [ModelMaterialParams, Modifiers, bResetExisting](mochi::Actor* actor) {
          mochi::SoftMaterialParams params = ModelMaterialParams->ToMochi();
          mochi::ErrorLog e{};
          if (bResetExisting) {
            if (actor->GetType() == mochi::ActorType::Soft) {
              actor->SetSoftMaterialParams(params, e);
            } else if (actor->GetType() == mochi::ActorType::Rigid) {
              actor->SetDensity(params.density, e);
            }
          }
          if (actor->GetType() == mochi::ActorType::Soft) {
            // Modifiers only supported for soft actors.
            for (const auto Modifier : Modifiers) {
              // TODO: should we check for zeros/negative?
              // indices are already checked in mochi
              constexpr mochi::real k_MPaToPa = 1e6_r;
              int idx = Modifier.Key;
              auto values = Modifier.Value;
              auto const ym = values.YoungsModulus * k_MPaToPa;
              auto const pr = values.PoissonRatio;
              static_assert(
                  static_cast<int>(mochi::SoftMaterialType::Count) == 6,
                  "Please update the switch statement below if SoftMaterialType enum changes");
              switch (params.type) {
                case mochi::SoftMaterialType::StVenantKirchhoff:
                  params.stVenantKirchhoff.youngsModulus = ym;
                  params.stVenantKirchhoff.poissonRatio = pr;
                  break;
                case mochi::SoftMaterialType::LinearElastic:
                  params.linearElastic.youngsModulus = ym;
                  params.linearElastic.poissonRatio = pr;
                  break;
                case mochi::SoftMaterialType::ActiveNeoHookean:
                  params.activeNeoHookean.passiveIsotropic.youngsModulus = ym;
                  params.activeNeoHookean.passiveIsotropic.poissonRatio = pr;
                  break;
                case mochi::SoftMaterialType::NeoHookean:
                  params.neoHookean.youngsModulus = ym;
                  params.neoHookean.poissonRatio = pr;
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
              mochi::experimental::SetSoftMaterialParamsField(actor, params, idx, e);
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
        FUnrealToMochiConverter u2m;
        Actor->SetVelocity(
            u2m.TranslationToMochi(LinearVelocity),
            u2m.DirectionToMochi(AngularVelocity),
            mochi::ErrorLog{});
      });
}

void UMochiActorObject::GetInertialParams(
    double& Mass,
    FVector& CenterOfMass,
    FMochiMomentOfInertia& MomentOfInertia) const {
  MochiSceneObject->QueueOrExecuteActorCommand(
      MochiActorHandle, [&Mass, &CenterOfMass, &MomentOfInertia](mochi::Actor* actor) {
        mochi::ErrorLog e{};
        Mass = actor->GetMass(e);
        FMochiToUnrealConverter m2u;
        CenterOfMass = m2u.TranslationToUnreal(actor->GetRigidCenterOfMassLocal(e));
        auto moi = actor->GetRigidMomentOfInertiaLocal(e);
        MomentOfInertia = m2u.InertiaToUnreal(moi);
        MomentOfInertia.Ixy = m2u.MeasureToUnreal(moi[1], 2);
        MomentOfInertia.Ixz = m2u.MeasureToUnreal(moi[2], 2);
        MomentOfInertia.Iyy = m2u.MeasureToUnreal(moi[3], 2);
        MomentOfInertia.Iyz = m2u.MeasureToUnreal(moi[4], 2);
        MomentOfInertia.Izz = m2u.MeasureToUnreal(moi[5], 2);
      });
  MochiSceneObject->WaitForQueuedCommands();
}

void UMochiActorObject::DestroyMochiResources() {
  MOCHI_PROFILE_SCOPE();
  if (MochiSceneObject) {
    MochiSceneObject->QueueOrExecuteCommand([this](mochi::Scene* scene) {
      scene->CancelCallback(ExtractQueryDataCallbackHandle);
      if (bDestroyMochiActorOnObjectDestruction) {
        scene->DestroyActor(MochiActorHandle);
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
static void CopyToUnrealArray(mochi::Span<Elem const> in, TArray<Elem>& out, mochi::Error& error) {
  if (error.IsOK()) {
    out.SetNum(in.size());
    std::copy(in.data(), in.data() + in.size(), out.GetData());
  } else {
    out.SetNum(0);
  }
}

void UMochiActorObject::ExtractQueryData_Internal(const mochi::Actor* Actor, mochi::Scene* Scene) {
  MOCHI_PROFILE_SCOPE();
  if (Actor == nullptr || Scene == nullptr) {
    return;
  }
  mochi::ErrorLog err;

  // extract root transform
  FMochiToUnrealConverter m2u;
  MochiThreadQueryData->RootTransform = m2u.TransformToUnreal(Actor->GetRootTransform());

  // We may only query velocity from rigid and soft bodies, or individual bones within an
  // articulated actor. Invalid calls will throw an error.
  mochi::ActorType actorType = Actor->GetType();
  if (actorType == mochi::ActorType::Rigid || actorType == mochi::ActorType::Soft) {
    MochiThreadQueryData->LinearVelocity =
        m2u.TranslationToUnreal(Actor->GetLinearVelocity(mochi::ErrorLog{}));
    MochiThreadQueryData->AngularVelocity =
        m2u.DirectionToUnreal(Actor->GetAngularVelocity(mochi::ErrorLog{}));
  }
  const bool useVisualMesh = bUseVisualMeshIfPresent && !Actor->GetVisualMesh().IsEmpty();
  const bool useContactSkin =
      !useVisualMesh && bUseContactSkinIfPresent && !Actor->GetContactSkinMesh().IsEmpty();
  // extract node positions
  if (bQuerySurfaceNodePositions) {
    mochi::Span<const mochi::real> mochiNodePositions;
    if (useVisualMesh) {
      mochiNodePositions = Actor->GetVisualMeshNodePositionsLocal(err);
    } else if (useContactSkin) {
      mochiNodePositions = Actor->GetContactSkinMeshNodePositionsLocal(err);
    } else {
      mochiNodePositions = Actor->GetSurfaceMeshNodePositionsLocal(err);
    }
    CopyToUnrealArray(mochiNodePositions, MochiThreadQueryData->SurfaceNodePositions, err);
  }
  if (bQueryVolumeNodePositions) {
    mochi::Span<const mochi::real> mochiVolumeNodePositions = Actor->GetNodePositionsLocal(err);
    CopyToUnrealArray(mochiVolumeNodePositions, MochiThreadQueryData->VolumeNodePositions, err);
  }
  // extract node normals
  if (bQueryNodeNormals) {
    mochi::Span<const mochi::real> mochiNodeNormals;
    if (useVisualMesh) {
      mochiNodeNormals = Actor->GetVisualMeshNodeNormalsLocal(err);
    } else if (useContactSkin) {
      mochiNodeNormals = Actor->GetContactSkinMeshNodeNormalsLocal(err);
    } else {
      mochiNodeNormals = Actor->GetSurfaceMeshNodeNormalsLocal(err);
    }
    CopyToUnrealArray(mochiNodeNormals, MochiThreadQueryData->NodeNormals, err);
  }

  if (bQueryElasticEnergy) {
    mochi::real energy = Actor->GetElasticEnergy(err);
    MochiThreadQueryData->ElasticEnergy = energy;
  }

  // extract contact points
  if (bQueryContactPoints) {
    const mochi::Span<const mochi::ContactPoint> contactPoints = Actor->GetContactPointsWorld(err);
    if (err.IsOK()) {
      MOCHI_PROFILE_SCOPE_N("Convert Contact Points For Unreal");
      // TODO: Convert the contacts on the game thread to take this work off the Mochi thread.
      // ContactPointToUnreal looks both actors up in the Mochi scene, so capture what it reads
      // from them here first.
      MochiThreadQueryData->ContactPoints.SetNum(contactPoints.size());
      for (int iContact = 0; iContact < contactPoints.size(); ++iContact) {
        const auto& mochiContact = contactPoints[iContact];
        auto& unrealContact = MochiThreadQueryData->ContactPoints[iContact];
        ContactPointToUnreal(mochiContact, Scene, NAME_None, m2u, unrealContact);
      }
    }
  }

  // Synchronize
  {
    FScopeLock lock(&PendingQueryDataLock);
    std::swap(MochiThreadQueryData, PendingQueryData);
    bHasPendingQueryData = true;
  }
}
