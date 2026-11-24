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

#include "MochiArticulationAssetUtils.h"
#include "MochiPhysicsEditorModule.h"

#include "AssetRegistry/AssetRegistryModule.h"
#include "ComponentReregisterContext.h"
#include "Components/SkinnedMeshComponent.h"
#include "DynamicMesh/DynamicMesh3.h"
#include "DynamicMeshToMeshDescription.h"
#include "Engine/SkeletalMesh.h"
#include "Engine/StaticMesh.h"
#include "Misc/EngineVersionComparison.h"
#include "SkeletalMeshAttributes.h"
#include "SkeletalMeshOperations.h"
#include "StaticMeshAttributes.h"
#include "StaticMeshLODResourcesToDynamicMesh.h"
#include "StaticMeshOperations.h"
#include "StaticMeshResources.h"
#include "StaticToSkeletalMeshConverter.h"
#include "UObject/UObjectIterator.h"

namespace {
template <typename TAsset>
TAsset* CreateOrReuseAsset(const FString& AssetPath, EObjectFlags Flags, bool bForceNew) {
  UPackage* Package = FindPackage(nullptr, *AssetPath);
  TAsset* Asset = nullptr;
  if (Package) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Display,
        TEXT("[MochiArticulationAssetUtils] Package %s already exists; reusing it,"),
        *AssetPath);
    Asset = Cast<TAsset>(Package->FindAssetInPackage());
  } else {
    Package = CreatePackage(*AssetPath);
  }
  if (Asset == nullptr || bForceNew) {
    if (Asset) {
      FAssetRegistryModule::AssetDeleted(Asset);
    }
    FString ObjName = FPackageName::GetLongPackageAssetName(AssetPath);
    Asset = NewObject<TAsset>(Package, FName(*ObjName), Flags);
    FAssetRegistryModule::AssetCreated(Asset);
  }
  return Asset;
}

// Unregisters the components that show any of Meshes for as long as it lives. Regenerating a
// skeletal mesh and its skeleton replaces them in place, which releases the render data, and
// invalidates the bone transforms, that those components were set up with.
class FScopedUnregisterMeshComponents {
 public:
  explicit FScopedUnregisterMeshComponents(TArrayView<const USkinnedAsset* const> Meshes) {
    for (TObjectIterator<USkinnedMeshComponent> It; It; ++It) {
      if (It->IsRegistered() && It->GetSkinnedAsset() && Meshes.Contains(It->GetSkinnedAsset())) {
        Contexts.Add(MakeUnique<FComponentReregisterContext>(*It));
      }
    }
  }

 private:
  TArray<TUniquePtr<FComponentReregisterContext>> Contexts;
};

// How a mesh is merged into each LOD of a generated skeletal mesh.
struct FMeshPart {
  // Rest transform of the mesh relative to the articulation root.
  FTransform MeshToRoot = FTransform::Identity;
  // The skeletal mesh material index of each polygon group, indexed by polygon group ID.
  TArray<int32> GroupMaterials;
};

// Builds the materials of a generated skeletal mesh and gives each one a slot name not used before.
class FMaterialListBuilder {
 public:
  int32 Add(UMaterialInterface* Material, const FString& BaseName) {
    FName SlotName(*BaseName);
    for (int32 Suffix = 1; TakenSlotNames.Contains(SlotName); ++Suffix) {
      SlotName = FName(*FString::Printf(TEXT("%s_%03d"), *BaseName, Suffix));
    }
    TakenSlotNames.Add(SlotName);
    return Materials.Emplace(Material, SlotName, SlotName);
  }
  TArray<FSkeletalMaterial> Materials;

 private:
  TSet<FName> TakenSlotNames;
};

// Converts the render geometry of one LOD of a link's static mesh (its last LOD if it has fewer)
// into a mesh description bound rigidly to Bone. The render geometry is what the static mesh shows
// after its own build, e.g. with build scale applied and normals computed. Returns false if the
// static mesh has no render data or its render geometry cannot be converted.
bool MakeRigidLinkMesh(
    const UStaticMesh& StaticMesh,
    int32 LODIndex,
    int32 Bone,
    FMeshDescription& OutMesh) {
  const FStaticMeshRenderData* RenderData = StaticMesh.GetRenderData();
  if (RenderData == nullptr || RenderData->LODResources.IsEmpty()) {
    return false;
  }
  const FStaticMeshLODResources& LODResources =
      RenderData->LODResources[FMath::Min(LODIndex, RenderData->LODResources.Num() - 1)];
  const FColorVertexBuffer& Colors = LODResources.VertexBuffers.ColorVertexBuffer;
  UE::Geometry::FDynamicMesh3 DynamicMesh;
  if (!UE::Geometry::FStaticMeshLODResourcesToDynamicMesh::Convert(
          &LODResources,
          UE::Geometry::FStaticMeshLODResourcesToDynamicMesh::ConversionOptions(),
          DynamicMesh,
          Colors.GetNumVertices() > 0,
          [&Colors](int32 VertexIndex) -> FColor { return Colors.VertexColor(VertexIndex); })) {
    return false;
  }
  FSkeletalMeshAttributes Attributes(OutMesh);
  Attributes.Register();
  FDynamicMeshToMeshDescription Converter;
  Converter.ConversionOptions.bSetPolyGroups = false;
  // Keep the vertex colors as the render buffers store them.
  Converter.ConversionOptions.bTransformVtxColorsSRGBToLinear = false;
  Converter.Convert(&DynamicMesh, OutMesh, /*bCopyTangents=*/true);

  // The normals are final, so no edge is smoothed over again.
  TEdgeAttributesRef<bool> EdgeHardnesses = Attributes.GetEdgeHardnesses();
  for (const FEdgeID EdgeID : OutMesh.Edges().GetElementIDs()) {
    EdgeHardnesses[EdgeID] = true;
  }

  // Polygon groups are the static mesh's material indices. Bind every vertex fully to the link.
  FSkinWeightsVertexAttributesRef SkinWeights = Attributes.GetVertexSkinWeights();
  const UE::AnimationCore::FBoneWeights RigidBinding = UE::AnimationCore::FBoneWeights::Create(
      {UE::AnimationCore::FBoneWeight(static_cast<FBoneIndexType>(Bone), 1.0f)});
  for (const FVertexID VertexID : OutMesh.Vertices().GetElementIDs()) {
    SkinWeights.Set(VertexID, RigidBinding);
  }
  return true;
}

// A LOD mesh description of a skeletal mesh with RefSkeleton's bones and one empty polygon group
// per material, so that polygon group IDs are material indices.
void InitLODMesh(
    const FReferenceSkeleton& RefSkeleton,
    const TArray<FSkeletalMaterial>& Materials,
    FMeshDescription& OutMesh) {
  FSkeletalMeshAttributes Attributes(OutMesh);
  Attributes.Register();
  for (int32 Bone = 0; Bone < RefSkeleton.GetRawBoneNum(); ++Bone) {
    const FBoneID BoneID = Attributes.CreateBone();
    Attributes.GetBoneNames().Set(BoneID, RefSkeleton.GetBoneName(Bone));
    Attributes.GetBoneParentIndices().Set(BoneID, RefSkeleton.GetParentIndex(Bone));
    Attributes.GetBonePoses().Set(BoneID, RefSkeleton.GetRefBonePose()[Bone]);
  }
  TPolygonGroupAttributesRef<FName> SlotNames = Attributes.GetPolygonGroupMaterialSlotNames();
  for (const FSkeletalMaterial& Material : Materials) {
    SlotNames[OutMesh.CreatePolygonGroup()] = Material.MaterialSlotName;
  }
}

// Whether every vertex instance of Mesh has a nonzero vector in Vectors, e.g. the normals of a file
// that stores them.
bool HasAllVectors(
    const FMeshDescription& Mesh,
    TVertexInstanceAttributesConstRef<FVector3f> Vectors) {
  for (const FVertexInstanceID ID : Mesh.VertexInstances().GetElementIDs()) {
    if (Vectors[ID].IsNearlyZero()) {
      return false;
    }
  }
  return true;
}

// Appends Mesh, a mesh with vertex skin weights whose bone indices index the skeleton's bones, to
// Target, a LOD mesh description set up by InitLODMesh for a skeleton with NumBones bones.
void AppendMeshPart(
    const FMeshDescription& Mesh,
    const FMeshPart& Part,
    int32 NumBones,
    FMeshDescription& Target) {
  // AppendSkinWeight places each source vertex at a fixed offset from its ID, which holds only
  // while both meshes' vertex IDs are compact.
  auto IsCompact = [](const FMeshDescription& Mesh) {
    return Mesh.Vertices().Num() == Mesh.Vertices().GetArraySize();
  };
  if (!ensure(IsCompact(Target) && IsCompact(Mesh))) {
    return;
  }
  FStaticMeshOperations::FAppendSettings AppendSettings;
  AppendSettings.MeshTransform = Part.MeshToRoot;
  AppendSettings.PolygonGroupsDelegate.BindLambda(
      [&Part](const FMeshDescription& Source, FMeshDescription& Target, PolygonGroupMap& Remap) {
        for (const FPolygonGroupID SourceID : Source.PolygonGroups().GetElementIDs()) {
          const int32 MaterialIndex = Part.GroupMaterials.IsValidIndex(SourceID.GetValue())
              ? Part.GroupMaterials[SourceID.GetValue()]
              : INDEX_NONE;
          Remap.Add(
              SourceID,
              Target.PolygonGroups().IsValid(FPolygonGroupID(MaterialIndex))
                  ? FPolygonGroupID(MaterialIndex)
                  : Target.CreatePolygonGroup());
        }
      });
  FSkeletalMeshOperations::FSkeletalMeshAppendSettings SkinWeightSettings;
  SkinWeightSettings.SourceRemapBoneIndex.SetNum(NumBones);
  for (int32 Bone = 0; Bone < NumBones; ++Bone) {
    SkinWeightSettings.SourceRemapBoneIndex[Bone] = static_cast<FBoneIndexType>(Bone);
  }
  SkinWeightSettings.SourceVertexIDOffset = Target.Vertices().Num();
  FStaticMeshOperations::AppendMeshDescription(Mesh, Target, AppendSettings);
  FSkeletalMeshOperations::AppendSkinWeight(Mesh, Target, SkinWeightSettings);
}

} // namespace

FString FMochiArticulationAssetUtils::MakeAssetName(
    const FString& Prefix,
    const FString& ArticulationName,
    const FString& AssetName,
    const FString& Postfix) {
  FString Name;
  Name += !Prefix.IsEmpty() ? Prefix + "_" : "";
  Name += ArticulationName;
  Name += !AssetName.IsEmpty() ? "_" + AssetName : "";
  Name += !Postfix.IsEmpty() ? "_" + Postfix : "";
  return Name;
}

TArray<FMochiArticulationSkeletonLink> FMochiArticulationAssetUtils::GetSkeletonLinks(
    const UMochiArticulation* Articulation) {
  const TArray<FMochiArticulatedLink>& Links = Articulation->GetLinks();
  TArray<FMochiArticulationSkeletonLink> SkeletonLinks;
  SkeletonLinks.SetNum(Links.Num());
  for (int LinkIndex = 0; LinkIndex < Links.Num(); ++LinkIndex) {
    const FMochiArticulatedLink& Link = Links[LinkIndex];
    FMochiArticulationSkeletonLink& SkeletonLink = SkeletonLinks[LinkIndex];
    SkeletonLink.Name = Link.Name;
    SkeletonLink.ParentIndex = Link.ParentLink;
    SkeletonLink.LinkToParent = Articulation->GetRestLinkToParent(LinkIndex);
    SkeletonLink.LinkToRoot = Articulation->GetRestLinkToRoot(LinkIndex);
    SkeletonLink.RenderModel = Link.RenderModel;
    SkeletonLink.RenderModelToLink = Link.GetRenderModelTransform();
    SkeletonLink.UniqueMaterialSlots = Link.UniqueMaterialSlots;
  }
  return SkeletonLinks;
}

UDataTable* FMochiArticulationAssetUtils::GenerateJointControlGainsTable(
    const TArray<FMochiArticulatedJoint>& Joints,
    const FString& PackagePath,
    const FString& TableAssetName,
    EObjectFlags Flags) {
  // If package already exists, try numerical suffixes until we find a new one.
  FString AssetPath = FPaths::Combine(PackagePath, TableAssetName);
  UPackage* Package = FindPackage(nullptr, *AssetPath);
  for (int i = 1; Package; ++i) {
    const FString Suffixed = FString::Printf(TEXT("%s_%d"), *AssetPath, i);
    Package = FindPackage(nullptr, *Suffixed);
    if (!Package) {
      AssetPath = Suffixed;
    }
  }
  // Create the package and asset within
  Package = CreatePackage(*AssetPath);
  FString ObjName = FPackageName::GetLongPackageAssetName(AssetPath);
  UDataTable* DataTable = NewObject<UDataTable>(Package, FName(*ObjName), Flags);
  FAssetRegistryModule::AssetCreated(DataTable);
  FMochiJointControlGains::PopulateTable(DataTable, Joints);
  DataTable->MarkPackageDirty();
  DataTable->PostEditChange();
  return DataTable;
}

USkeleton* FMochiArticulationAssetUtils::GenerateSkeleton(
    const TArray<FMochiArticulationSkeletonLink>& Links,
    const FString& PackagePath,
    const FString& SkeletonName,
    EObjectFlags Flags) {
  const FString AssetPath = FPaths::Combine(PackagePath, SkeletonName);
  USkeleton* Skeleton = CreateOrReuseAsset<USkeleton>(AssetPath, Flags, true);
  // One bone per link, at the link's rest transform relative to its parent.
  FReferenceSkeletonModifier BoneAdder(Skeleton);
  for (const FMochiArticulationSkeletonLink& Link : Links) {
    BoneAdder.Add(
        FMeshBoneInfo(Link.Name, Link.Name.ToString(), Link.ParentIndex), Link.LinkToParent);
  }
  Skeleton->MarkPackageDirty();
  Skeleton->PostEditChange();
  return Skeleton;
}

USkeletalMesh* FMochiArticulationAssetUtils::GenerateSkeletalMesh(
    const TArray<FMochiArticulationSkeletonLink>& Links,
    USkeleton* Skeleton,
    const FString& PackagePath,
    const FString& SkeletalMeshAssetName,
    EObjectFlags Flags,
    const TArray<FMochiArticulationSkinnedPart>& SkinnedParts) {
  const FReferenceSkeleton& RefSkeleton = Skeleton->GetReferenceSkeleton();
  const int32 NumBones = RefSkeleton.GetRawBoneNum();

  // Materials: those of the links' render models first, then those of the skinned parts. A
  // material shared by several render models gets one slot, unless a link keeps its slot unique.
  FMaterialListBuilder Materials;
  TMap<UMaterialInterface*, int32> MergeableMaterials;
  // Links with a render model.
  struct FRigidLink {
    const UStaticMesh* RenderModel = nullptr;
    int32 Bone = INDEX_NONE;
    FMeshPart Part;
  };
  TArray<FRigidLink> RigidLinks;
  int32 NumLODs = 0;
  for (int32 LinkIndex = 0; LinkIndex < Links.Num(); ++LinkIndex) {
    const FMochiArticulationSkeletonLink& Link = Links[LinkIndex];
    const FStaticMeshRenderData* RenderData =
        Link.RenderModel ? Link.RenderModel->GetRenderData() : nullptr;
    if (RenderData == nullptr || RenderData->LODResources.IsEmpty()) {
      continue;
    }
    NumLODs = FMath::Max(NumLODs, RenderData->LODResources.Num());
    FRigidLink& RigidLink = RigidLinks.AddDefaulted_GetRef();
    RigidLink.RenderModel = Link.RenderModel;
    RigidLink.Bone = LinkIndex;
    FMeshPart& Part = RigidLink.Part;
    Part.MeshToRoot = Link.RenderModelToLink * Link.LinkToRoot;
    int32 NumSlots = Link.RenderModel->GetStaticMaterials().Num();
    for (const FStaticMeshLODResources& Lod : RenderData->LODResources) {
      for (const FStaticMeshSection& Section : Lod.Sections) {
        NumSlots = FMath::Max(NumSlots, Section.MaterialIndex + 1);
      }
    }
    for (int32 Slot = 0; Slot < NumSlots; ++Slot) {
      UMaterialInterface* Material = Link.RenderModel->GetMaterial(Slot);
      const bool bMergeable = !Link.UniqueMaterialSlots.Contains(Slot);
      const int32* Existing = bMergeable ? MergeableMaterials.Find(Material) : nullptr;
      if (Existing) {
        Part.GroupMaterials.Add(*Existing);
        continue;
      }
      const int32 Index =
          Materials.Add(Material, Material ? Material->GetName() : FString(TEXT("Material")));
      Part.GroupMaterials.Add(Index);
      if (bMergeable) {
        MergeableMaterials.Add(Material, Index);
      }
    }
  }
  if (NumLODs == 0 && SkinnedParts.IsEmpty()) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Error,
        TEXT(
            "[MochiArticulationAssetUtils] No link has a render model and there are no skinned parts"));
    return nullptr;
  }

  TArray<FMeshPart> SkinParts;
  for (const FMochiArticulationSkinnedPart& SkinnedPart : SkinnedParts) {
    FMeshPart& Part = SkinParts.AddDefaulted_GetRef();
    Part.MeshToRoot = SkinnedPart.MeshToRoot;
    for (const FPolygonGroupID GroupID : SkinnedPart.Mesh.PolygonGroups().GetElementIDs()) {
      UMaterialInterface* Material = SkinnedPart.Materials.IsValidIndex(GroupID.GetValue())
          ? SkinnedPart.Materials[GroupID.GetValue()]
          : nullptr;
      if (Part.GroupMaterials.Num() <= GroupID.GetValue()) {
        Part.GroupMaterials.SetNum(GroupID.GetValue() + 1);
      }
      Part.GroupMaterials[GroupID.GetValue()] =
          Materials.Add(Material, Material ? Material->GetName() : FString(TEXT("Skin")));
    }
  }

  // One mesh description per LOD: every link's render model at that LOD, bound rigidly to its
  // bone, followed by the skinned parts.
  TArray<FMeshDescription> LodMeshes;
  LodMeshes.SetNum(FMath::Max(NumLODs, 1));
  for (int32 LODIndex = 0; LODIndex < LodMeshes.Num(); ++LODIndex) {
    FMeshDescription& LodMesh = LodMeshes[LODIndex];
    InitLODMesh(RefSkeleton, Materials.Materials, LodMesh);
    for (const FRigidLink& RigidLink : RigidLinks) {
      FMeshDescription LinkMesh;
      if (MakeRigidLinkMesh(*RigidLink.RenderModel, LODIndex, RigidLink.Bone, LinkMesh)) {
        AppendMeshPart(LinkMesh, RigidLink.Part, NumBones, LodMesh);
      } else {
        UE_LOG(
            LogMochiPhysicsEditor,
            Warning,
            TEXT(
                "[MochiArticulationAssetUtils] Skipping %s in LOD %d of %s: its render geometry could not be read"),
            *RigidLink.RenderModel->GetName(),
            LODIndex,
            *SkeletalMeshAssetName);
      }
    }
    for (int32 SkinIndex = 0; SkinIndex < SkinnedParts.Num(); ++SkinIndex) {
      AppendMeshPart(SkinnedParts[SkinIndex].Mesh, SkinParts[SkinIndex], NumBones, LodMesh);
    }
  }

  const FString AssetPath = FPaths::Combine(PackagePath, SkeletalMeshAssetName);
  USkeletalMesh* SkeletalMesh = CreateOrReuseAsset<USkeletalMesh>(AssetPath, Flags, true);
  TArray<const FMeshDescription*> LodMeshPointers;
  for (const FMeshDescription& LodMesh : LodMeshes) {
    LodMeshPointers.Add(&LodMesh);
  }
  // The render models come with the normals and tangents of their own build, and the skinned parts
  // with those of their files. A mesh made only of skinned parts has them recomputed only if a part
  // lacks them: recomputing splits the normals along the parts' UV seams.
  bool bSkinNormals = true;
  bool bSkinTangents = true;
  for (const FMochiArticulationSkinnedPart& skinnedPart : SkinnedParts) {
    const FStaticMeshConstAttributes attributes(skinnedPart.Mesh);
    bSkinNormals =
        bSkinNormals && HasAllVectors(skinnedPart.Mesh, attributes.GetVertexInstanceNormals());
    bSkinTangents =
        bSkinTangents && HasAllVectors(skinnedPart.Mesh, attributes.GetVertexInstanceTangents());
  }
  const bool bRecomputeNormals = RigidLinks.IsEmpty() && !bSkinNormals;
  const bool bRecomputeTangents = RigidLinks.IsEmpty() && !bSkinTangents;
#if UE_VERSION_OLDER_THAN(5, 8, 0)
  const bool bBuilt = FStaticToSkeletalMeshConverter::InitializeSkeletalMeshFromMeshDescriptions(
      SkeletalMesh,
      LodMeshPointers,
      Materials.Materials,
      RefSkeleton,
      bRecomputeNormals,
      bRecomputeTangents);
#else
  FStaticToSkeletalMeshConverter::FInitializationParams InitParams;
  InitParams.Materials = Materials.Materials;
  InitParams.bRecomputeNormals = bRecomputeNormals;
  InitParams.bRecomputeTangents = bRecomputeTangents;
  const bool bBuilt = FStaticToSkeletalMeshConverter::InitializeSkeletalMeshFromMeshDescriptions(
      SkeletalMesh, LodMeshPointers, RefSkeleton, InitParams);
#endif
  if (!bBuilt) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Error,
        TEXT("[MochiArticulationAssetUtils] Failed to build skeletal mesh %s"),
        *AssetPath);
    return nullptr;
  }

  // Bounds around the vertices of LOD 0 in the reference pose.
  TArray<FVector> Lod0Positions;
  const TVertexAttributesConstRef<FVector3f> Positions =
      FStaticMeshConstAttributes(LodMeshes[0]).GetVertexPositions();
  for (const FVertexID VertexID : LodMeshes[0].Vertices().GetElementIDs()) {
    Lod0Positions.Add(FVector(Positions[VertexID]));
  }
  if (!Lod0Positions.IsEmpty()) {
    SkeletalMesh->SetImportedBounds(FBoxSphereBounds(Lod0Positions.GetData(), Lod0Positions.Num()));
  }

  SkeletalMesh->SetSkeleton(Skeleton);
  Skeleton->MergeAllBonesToBoneTree(SkeletalMesh);
  if (!Skeleton->GetPreviewMesh()) {
    Skeleton->SetPreviewMesh(SkeletalMesh);
  }
  Skeleton->PostEditChange();

  SkeletalMesh->MarkPackageDirty();
  SkeletalMesh->PostEditChange();

  return SkeletalMesh;
}

bool FMochiArticulationAssetUtils::GenerateSkeletalMesh(
    UMochiArticulation* Articulation,
    const FString& PackagePath) {
  if (Articulation->PreGenValidateOrUpdateDerivedData()) {
    bool bPrefix = false;
    const FString RootName =
        UMochiArticulation::RemoveAssetTypePrefix(Articulation->GetName(), &bPrefix);
    const FString SkelName = MakeAssetName(
        bPrefix ? TEXT("SKEL") : TEXT(""),
        RootName,
        TEXT(""),
        bPrefix ? TEXT("") : TEXT("Skeleton"));
    const FString SkelMeshName = MakeAssetName(
        bPrefix ? TEXT("SKM") : TEXT(""),
        RootName,
        TEXT(""),
        bPrefix ? TEXT("") : TEXT("SkeletalMesh"));
    constexpr EObjectFlags Flags = RF_Public | RF_Standalone;
    const TArray<FMochiArticulationSkeletonLink> Links = GetSkeletonLinks(Articulation);
    {
      // Components showing the mesh being replaced, e.g. in an open level, would otherwise keep
      // rendering with its released render data.
      const USkinnedAsset* const ReplacedMeshes[] = {
          Articulation->GetSkeletalMesh(),
          FindObject<USkeletalMesh>(
              nullptr, *(FPaths::Combine(PackagePath, SkelMeshName) + TEXT(".") + SkelMeshName))};
      const FScopedUnregisterMeshComponents UnregisterComponents(ReplacedMeshes);
      USkeleton* const Skeleton = GenerateSkeleton(Links, PackagePath, SkelName, Flags);
      Articulation->SkeletalMesh =
          GenerateSkeletalMesh(Links, Skeleton, PackagePath, SkelMeshName, Flags);
      FProperty* Prop = FindFieldChecked<FProperty>(
          Articulation->StaticClass(), GET_MEMBER_NAME_CHECKED(UMochiArticulation, SkeletalMesh));
      FPropertyChangedEvent Event(Prop, EPropertyChangeType::ValueSet);
      Articulation->PostEditChangeProperty(Event);
      Articulation->Modify(true);
      if (Articulation->OnSkeletalMeshRegenerated) {
        Articulation->OnSkeletalMeshRegenerated(Articulation->SkeletalMesh);
      }
    }
    const bool bResult = Articulation->SkeletalMesh != nullptr;
    Articulation->bNeedGenerateSkeletalMesh = !bResult;
    return bResult;
  }
  UE_LOG(
      LogMochiPhysicsEditor,
      Error,
      TEXT(
          "[UMochiArticulation] Failed to generate asset because internal representation could not be validated for %s!"),
      *Articulation->GetName());
  return false;
}

void FMochiArticulationAssetUtils::GenerateSkeletalMesh(UMochiArticulation* Articulation) {
  const FString ArticulationFolder =
      FPackageName::GetLongPackagePath(Articulation->GetPackage()->GetName());
  const USkeletalMesh* CurrentMesh = Articulation->GetSkeletalMesh();
  const FString CurrentMeshFolder = CurrentMesh
      ? FPackageName::GetLongPackagePath(CurrentMesh->GetPackage()->GetName())
      : FString();
  // Meshes generated before skeletal meshes moved out of "Generated" subfolders are regenerated
  // next to the articulation instead.
  const bool bRegenerateInPlace = !CurrentMeshFolder.IsEmpty() &&
      FPaths::GetCleanFilename(CurrentMeshFolder) != TEXT("Generated");
  GenerateSkeletalMesh(Articulation, bRegenerateInPlace ? CurrentMeshFolder : ArticulationFolder);
}

UDataTable* FMochiArticulationAssetUtils::GenerateJointControlGainsTable(
    UMochiArticulation* Articulation) {
  bool bPrefix = false;
  const FString RootName =
      UMochiArticulation::RemoveAssetTypePrefix(Articulation->GetName(), &bPrefix);
  if (Articulation->PreGenValidateOrUpdateDerivedData()) {
    const FString PackPath =
        FPackageName::GetLongPackagePath(Articulation->GetPackage()->GetName());
    const FString GainsTableName = MakeAssetName(
        bPrefix ? TEXT("DT_Gains") : TEXT(""),
        RootName,
        TEXT(""),
        bPrefix ? TEXT("") : TEXT("DataTable"));
    UDataTable* GainsTable = GenerateJointControlGainsTable(
        Articulation->Joints, PackPath, GainsTableName, RF_Public | RF_Standalone);
    return GainsTable;
  }
  return nullptr;
}
