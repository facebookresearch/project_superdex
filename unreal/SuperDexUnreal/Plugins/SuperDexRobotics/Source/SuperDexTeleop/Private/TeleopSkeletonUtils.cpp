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

#include "TeleopSkeletonUtils.h"

#include "SuperDexTeleopModule.h"

#include "Animation/Skeleton.h"
#include "Components/SkeletalMeshComponent.h"
#include "Engine/SkeletalMesh.h"
#include "ReferenceSkeleton.h"
#include "UObject/Package.h"

// ---------------------------------------------------------------------------------------------------
// One-off skeleton-rebase utility toggle.
//
// UTeleopSkeletonUtils::ClearReferenceRootBoneRotation rebases a hand mesh's root bone (and the
// matching geometry) in place. It was a one-shot asset fix and is DISABLED by default so it does
// not drag in editor-only mesh-editing module dependencies. The BlueprintCallable entry point still
// exists but logs a warning and no-ops while disabled.
//
// To re-enable:
//   1) Set TELEOP_ENABLE_SKELETON_REBASE to 1 (below).
//   2) In Teleop.Build.cs, inside the Target.bBuildEditor block, add these to
//      PrivateDependencyModuleNames: "MeshDescription", "StaticMeshDescription",
//      "SkeletalMeshDescription".
// ---------------------------------------------------------------------------------------------------
#define TELEOP_ENABLE_SKELETON_REBASE 0

#if WITH_EDITOR && TELEOP_ENABLE_SKELETON_REBASE
#include "MeshDescription.h"
#include "SkeletalMeshAttributes.h"
#include "StaticMeshAttributes.h"
#endif

#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

namespace {

// Packs an XYZ triple into a JSON array.
TSharedPtr<FJsonValue> MakeVectorValue(FVector const& V) {
  TArray<TSharedPtr<FJsonValue>> Out;
  Out.Add(MakeShared<FJsonValueNumber>(V.X));
  Out.Add(MakeShared<FJsonValueNumber>(V.Y));
  Out.Add(MakeShared<FJsonValueNumber>(V.Z));
  return MakeShared<FJsonValueArray>(Out);
}

// Packs a quaternion into a JSON array in XYZW order.
TSharedPtr<FJsonValue> MakeQuatValue(FQuat const& Q) {
  TArray<TSharedPtr<FJsonValue>> Out;
  Out.Add(MakeShared<FJsonValueNumber>(Q.X));
  Out.Add(MakeShared<FJsonValueNumber>(Q.Y));
  Out.Add(MakeShared<FJsonValueNumber>(Q.Z));
  Out.Add(MakeShared<FJsonValueNumber>(Q.W));
  return MakeShared<FJsonValueArray>(Out);
}

// Serializes a transform as translation / rotation (quat XYZW + XYZ-degree
// Euler) / scale. Euler angles are about the X (Roll), Y (Pitch), Z (Yaw) axes.
TSharedPtr<FJsonObject> MakeTransformObject(FTransform const& T) {
  TSharedPtr<FJsonObject> Obj = MakeShared<FJsonObject>();
  Obj->SetField(TEXT("translation"), MakeVectorValue(T.GetTranslation()));
  Obj->SetField(TEXT("rotationQuatXYZW"), MakeQuatValue(T.GetRotation()));
  FRotator const Rot = T.GetRotation().Rotator();
  Obj->SetField(
      TEXT("rotationEulerXYZDeg"), MakeVectorValue(FVector(Rot.Roll, Rot.Pitch, Rot.Yaw)));
  Obj->SetField(TEXT("scale"), MakeVectorValue(T.GetScale3D()));
  return Obj;
}

#if WITH_EDITOR && TELEOP_ENABLE_SKELETON_REBASE
// UObject::MarkPackageDirty() is suppressed while the editor is in its PIE sandbox, so a change
// made during Play-In-Editor never flags the source asset for save. Setting the dirty flag directly
// on the real (non-PIE) package works from both PIE and normal editor contexts. Mirrors the
// approach in USuperDexPoseFunctionLibrary::UpdateSuperDexPoseInAsset.
void MarkAssetDirtyForSave(UObject* Asset) {
  if (Asset == nullptr || !GIsEditor || IsRunningCommandlet()) {
    return;
  }
  UPackage* Package = Asset->GetOutermost();
  if (Package != nullptr && !Package->HasAnyPackageFlags(PKG_PlayInEditor)) {
    Package->SetDirtyFlag(true);
  }
}

// Rebases the root bone in every LOD's source MeshDescription so the asset stays internally
// consistent (and visually transparent) after removing the root's rest rotation:
//   1) bone 0's pose rotation is cleared to identity (translation/scale preserved), and
//   2) the geometry (positions, normals, tangents) is counter-rotated by InverseRootRotation.
// Clearing bone 0 globally rotates the skeleton's component frames by InverseRootRotation, which
// pushes that same rotation into every bone's inverse-bind matrix; counter-rotating the geometry by
// the same amount cancels it for every vertex, so the skinned result is unchanged for any pose. The
// MeshDescription is also the editor source-of-truth the render data is derived from, so without
// this a render-data rebuild on reload reverts the change. Returns false if a LOD that has a
// MeshDescription could not be updated. (Assumes the root bone's rest translation is zero, i.e. the
// rotation is about the component origin - true for these hand skeletons.)
bool RebaseSourceMeshDescriptionRootBone(
    USkeletalMesh* SkeletalMesh,
    const FQuat& InverseRootRotation) {
  const FQuat4f GeomRot(
      static_cast<float>(InverseRootRotation.X),
      static_cast<float>(InverseRootRotation.Y),
      static_cast<float>(InverseRootRotation.Z),
      static_cast<float>(InverseRootRotation.W));
  bool bAllOk = true;
  int32 NumUpdated = 0;
  const int32 NumLODs = SkeletalMesh->GetLODNum();
  for (int32 LODIndex = 0; LODIndex < NumLODs; ++LODIndex) {
    if (!SkeletalMesh->HasMeshDescription(LODIndex)) {
      continue;
    }
    FMeshDescription MeshDescription;
    if (!SkeletalMesh->CloneMeshDescription(LODIndex, MeshDescription)) {
      UE_LOG(
          LogSuperDexTeleop,
          Error,
          TEXT("[%hs] Failed to clone MeshDescription for LOD %d of '%s'."),
          __FUNCTION__,
          LODIndex,
          *SkeletalMesh->GetName());
      bAllOk = false;
      continue;
    }
    FSkeletalMeshAttributes Attributes(MeshDescription);
    FSkeletalMeshAttributes::FBonePoseAttributesRef BonePoses = Attributes.GetBonePoses();
    if (!BonePoses.IsValid() || Attributes.GetNumBones() == 0) {
      UE_LOG(
          LogSuperDexTeleop,
          Error,
          TEXT("[%hs] LOD %d of '%s' has no usable bone-pose attributes."),
          __FUNCTION__,
          LODIndex,
          *SkeletalMesh->GetName());
      bAllOk = false;
      continue;
    }

    // 1) Clear bone 0's rest rotation (preserve its translation/scale).
    const FBoneID RootBoneID(0);
    const FTransform OldDescRoot = BonePoses.Get(RootBoneID);
    BonePoses.Set(
        RootBoneID,
        FTransform(FQuat::Identity, OldDescRoot.GetLocation(), OldDescRoot.GetScale3D()));

    // 2) Counter-rotate the geometry so the skinned result is unchanged for any pose.
    TVertexAttributesRef<FVector3f> Positions = Attributes.GetVertexPositions();
    for (const FVertexID VertexID : MeshDescription.Vertices().GetElementIDs()) {
      Positions.Set(VertexID, GeomRot.RotateVector(Positions.Get(VertexID)));
    }
    TVertexInstanceAttributesRef<FVector3f> Normals = Attributes.GetVertexInstanceNormals();
    TVertexInstanceAttributesRef<FVector3f> Tangents = Attributes.GetVertexInstanceTangents();
    const bool bHasNormals = Normals.IsValid();
    const bool bHasTangents = Tangents.IsValid();
    for (const FVertexInstanceID InstanceID : MeshDescription.VertexInstances().GetElementIDs()) {
      if (bHasNormals) {
        Normals.Set(InstanceID, GeomRot.RotateVector(Normals.Get(InstanceID)));
      }
      if (bHasTangents) {
        Tangents.Set(InstanceID, GeomRot.RotateVector(Tangents.Get(InstanceID)));
      }
    }

    SkeletalMesh->CreateMeshDescription(LODIndex, MoveTemp(MeshDescription));
    SkeletalMesh->CommitMeshDescription(LODIndex);
    ++NumUpdated;
  }
  if (NumUpdated == 0) {
    UE_LOG(
        LogSuperDexTeleop,
        Warning,
        TEXT(
            "[%hs] '%s' has no editable MeshDescription on any LOD; the source geometry/bind pose "
            "was not updated and a render-data rebuild may revert the change. (Legacy/cooked asset?)"),
        __FUNCTION__,
        *SkeletalMesh->GetName());
  }
  return bAllOk;
}
#endif

// Resolves the destination path: absolute paths are used verbatim, relative
// paths are anchored under the project's Saved/ directory, and an empty path
// falls back to Saved/SkeletonExports/<MeshName>.json.
FString ResolveOutputPath(FString const& FilePath, FString const& MeshName) {
  if (FilePath.IsEmpty()) {
    return FPaths::Combine(
        FPaths::ProjectSavedDir(), TEXT("SkeletonExports"), MeshName + TEXT(".json"));
  }
  if (FPaths::IsRelative(FilePath)) {
    return FPaths::Combine(FPaths::ProjectSavedDir(), FilePath);
  }
  return FilePath;
}

} // namespace

bool UTeleopSkeletonUtils::ExportSkeletalMeshSkeleton(
    USkeletalMesh const* SkeletalMesh,
    FString const& FilePath) {
  if (SkeletalMesh == nullptr) {
    UE_LOG(LogSuperDexTeleop, Error, TEXT("[%hs] SkeletalMesh is null."), __FUNCTION__);
    return false;
  }

  FReferenceSkeleton const& RefSkeleton = SkeletalMesh->GetRefSkeleton();
  TArray<FMeshBoneInfo> const& BoneInfos = RefSkeleton.GetRefBoneInfo();
  TArray<FTransform> const& LocalPoses = RefSkeleton.GetRefBonePose();
  int32 const NumBones = RefSkeleton.GetNum();

  if (NumBones == 0) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("[%hs] SkeletalMesh '%s' has an empty reference skeleton."),
        __FUNCTION__,
        *SkeletalMesh->GetName());
    return false;
  }

  // Accumulate component-space (root-relative) transforms. The reference
  // skeleton guarantees parents precede their children, so a single forward
  // pass suffices: ComponentSpace[i] = Local[i] * ComponentSpace[parent].
  TArray<FTransform> ComponentPoses;
  ComponentPoses.SetNum(NumBones);
  for (int32 i = 0; i < NumBones; ++i) {
    int32 const ParentIndex = BoneInfos[i].ParentIndex;
    ComponentPoses[i] =
        (ParentIndex == INDEX_NONE) ? LocalPoses[i] : LocalPoses[i] * ComponentPoses[ParentIndex];
  }

  TArray<TSharedPtr<FJsonValue>> BoneValues;
  BoneValues.Reserve(NumBones);
  for (int32 i = 0; i < NumBones; ++i) {
    int32 const ParentIndex = BoneInfos[i].ParentIndex;
    TSharedPtr<FJsonObject> Bone = MakeShared<FJsonObject>();
    Bone->SetNumberField(TEXT("index"), i);
    Bone->SetStringField(TEXT("name"), BoneInfos[i].Name.ToString());
    Bone->SetNumberField(TEXT("parentIndex"), ParentIndex);
    Bone->SetStringField(
        TEXT("parentName"),
        ParentIndex == INDEX_NONE ? FString() : BoneInfos[ParentIndex].Name.ToString());
    Bone->SetObjectField(TEXT("local"), MakeTransformObject(LocalPoses[i]));
    Bone->SetObjectField(TEXT("component"), MakeTransformObject(ComponentPoses[i]));
    BoneValues.Add(MakeShared<FJsonValueObject>(Bone));
  }

  TSharedPtr<FJsonObject> Root = MakeShared<FJsonObject>();
  Root->SetStringField(TEXT("sourceAsset"), SkeletalMesh->GetPathName());
  Root->SetStringField(
      TEXT("coordinateSystem"),
      TEXT(
          "UnrealEngine: left-handed, X=forward, Y=right, Z=up, units=cm; "
          "rotationQuatXYZW order; rotationEulerXYZDeg = degrees about X(Roll),Y(Pitch),Z(Yaw)"));
  Root->SetNumberField(TEXT("boneCount"), NumBones);
  Root->SetArrayField(TEXT("bones"), BoneValues);

  FString JsonString;
  TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonString);
  if (!FJsonSerializer::Serialize(Root.ToSharedRef(), Writer)) {
    UE_LOG(
        LogSuperDexTeleop, Error, TEXT("[%hs] Failed to serialize skeleton JSON."), __FUNCTION__);
    return false;
  }

  FString const OutPath = ResolveOutputPath(FilePath, SkeletalMesh->GetName());
  if (!FFileHelper::SaveStringToFile(JsonString, *OutPath)) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("[%hs] Failed to write skeleton JSON to '%s'."),
        __FUNCTION__,
        *OutPath);
    return false;
  }

  UE_LOG(
      LogSuperDexTeleop,
      Display,
      TEXT("[%hs] Wrote %d bones from '%s' to '%s'."),
      __FUNCTION__,
      NumBones,
      *SkeletalMesh->GetName(),
      *OutPath);
  return true;
}

bool UTeleopSkeletonUtils::ExportSkeletalMeshComponentSkeleton(
    USkeletalMeshComponent const* Component,
    FString const& FilePath) {
  if (Component == nullptr) {
    UE_LOG(LogSuperDexTeleop, Error, TEXT("[%hs] Component is null."), __FUNCTION__);
    return false;
  }
  return ExportSkeletalMeshSkeleton(Component->GetSkeletalMeshAsset(), FilePath);
}

bool UTeleopSkeletonUtils::ClearReferenceRootBoneRotation(
    USkeletalMesh* SkeletalMesh,
    bool bAlsoUpdateSharedSkeleton) {
#if WITH_EDITOR && TELEOP_ENABLE_SKELETON_REBASE
  if (SkeletalMesh == nullptr) {
    UE_LOG(LogSuperDexTeleop, Error, TEXT("[%hs] SkeletalMesh is null."), __FUNCTION__);
    return false;
  }

  const FReferenceSkeleton& RefSkeleton = SkeletalMesh->GetRefSkeleton();
  if (RefSkeleton.GetNum() == 0) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("[%hs] SkeletalMesh '%s' has an empty reference skeleton."),
        __FUNCTION__,
        *SkeletalMesh->GetName());
    return false;
  }

  // Keep the root's translation and scale; only the (spurious) rotation is removed. The root
  // location is expected to be zero for these hand skeletons, but preserving it keeps the operation
  // general and harmless.
  const FTransform OldRoot = RefSkeleton.GetRefBonePose()[0];
  if (OldRoot.GetRotation().IsIdentity(KINDA_SMALL_NUMBER)) {
    UE_LOG(
        LogSuperDexTeleop,
        Display,
        TEXT("[%hs] Root bone of '%s' already has identity rotation; nothing to do."),
        __FUNCTION__,
        *SkeletalMesh->GetName());
    return true;
  }
  const FTransform NewRoot(FQuat::Identity, OldRoot.GetLocation(), OldRoot.GetScale3D());

  USkeleton* const Skeleton = SkeletalMesh->GetSkeleton();

  SkeletalMesh->Modify();
  SkeletalMesh->PreEditChange(nullptr);

  // 1) Rebase the source MeshDescription first: clear bone 0's rotation and counter-rotate the
  // geometry by the inverse of the removed rotation. This is the editor source of truth the render
  // data is derived from; without it a rebuild on reload re-skins against the old data (bones look
  // corrected but the mesh reverts/distorts). Abort before touching RefSkeleton if this fails, so
  // we don't create an inconsistent state.
  if (!RebaseSourceMeshDescriptionRootBone(SkeletalMesh, OldRoot.GetRotation().Inverse())) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "[%hs] Failed to rebase source MeshDescription for '%s'; aborting without changing the "
            "reference skeleton. If this is a multi-LOD mesh it may be partially modified - revert it."),
        __FUNCTION__,
        *SkeletalMesh->GetName());
    SkeletalMesh->PostEditChange();
    return false;
  }

  // 2) Mutate a copy of the reference skeleton and assign it back via SetRefSkeleton (mirrors the
  // pattern in MochiArticulationAssetUtils). The modifier rebuilds the lookup tables on scope exit.
  FReferenceSkeleton NewRefSkeleton = RefSkeleton;
  {
    FReferenceSkeletonModifier Modifier(NewRefSkeleton, Skeleton);
    Modifier.UpdateRefPoseTransform(0, NewRoot);
  }
  SkeletalMesh->SetRefSkeleton(NewRefSkeleton);

  // 3) Rebuild the per-bone inverse bind matrices from the new reference pose. With the source
  // MeshDescription (step 1) and RefSkeleton (step 2) now in agreement, the rest pose renders
  // unchanged and survives a render-data rebuild.
  SkeletalMesh->CalculateInvRefMatrices();
  SkeletalMesh->PostEditChange();
  MarkAssetDirtyForSave(SkeletalMesh);

  if (bAlsoUpdateSharedSkeleton && Skeleton != nullptr &&
      Skeleton->GetReferenceSkeleton().GetNum() > 0) {
    Skeleton->Modify();
    {
      // Single-arg modifier targets the skeleton's own reference skeleton.
      FReferenceSkeletonModifier SkeletonModifier(Skeleton);
      SkeletonModifier.UpdateRefPoseTransform(0, NewRoot);
    }
    Skeleton->PostEditChange();
    MarkAssetDirtyForSave(Skeleton);
  }

  const FRotator OldRot = OldRoot.GetRotation().Rotator();
  UE_LOG(
      LogSuperDexTeleop,
      Display,
      TEXT(
          "[%hs] Rebased root bone on '%s': removed rotation (Roll=%.2f Pitch=%.2f Yaw=%.2f) and "
          "counter-rotated geometry. Save the asset to persist. Shared skeleton updated: %s."),
      __FUNCTION__,
      *SkeletalMesh->GetName(),
      OldRot.Roll,
      OldRot.Pitch,
      OldRot.Yaw,
      (bAlsoUpdateSharedSkeleton && Skeleton != nullptr) ? TEXT("yes") : TEXT("no"));
  return true;
#else
  // Disabled (TELEOP_ENABLE_SKELETON_REBASE == 0) or non-editor build. The entry point is kept so
  // existing Blueprint/Python callers still resolve; it no-ops with a warning.
  (void)SkeletalMesh;
  (void)bAlsoUpdateSharedSkeleton;
  UE_LOG(
      LogSuperDexTeleop,
      Warning,
      TEXT(
          "[%hs] is disabled. To re-enable: set TELEOP_ENABLE_SKELETON_REBASE to 1 in "
          "TeleopSkeletonUtils.cpp and re-add the MeshDescription/StaticMeshDescription/"
          "SkeletalMeshDescription editor module deps in Teleop.Build.cs."),
      __FUNCTION__);
  return false;
#endif
}
