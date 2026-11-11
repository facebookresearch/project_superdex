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

namespace {
template <typename TAsset>
TAsset* CreateOrReuseAsset(const FString& AssetPath, EObjectFlags Flags, bool bForceNew) {
  UPackage* package = FindPackage(nullptr, *AssetPath);
  TAsset* asset = nullptr;
  if (package) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Display,
        TEXT("[MochiArticulationAssetUtils] Package %s already exists; reusing it,"),
        *AssetPath);
    asset = Cast<TAsset>(package->FindAssetInPackage());
  } else {
    package = CreatePackage(*AssetPath);
  }
  if (asset == nullptr || bForceNew) {
    if (asset) {
      FAssetRegistryModule::AssetDeleted(asset);
    }
    FString objName = FPackageName::GetLongPackageAssetName(AssetPath);
    asset = NewObject<TAsset>(package, FName(*objName), Flags);
    FAssetRegistryModule::AssetCreated(asset);
  }
  return asset;
}

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
          [&Colors](int32 VertexIndex) { return Colors.VertexColor(VertexIndex); })) {
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
  FString name;
  name += !Prefix.IsEmpty() ? Prefix + "_" : "";
  name += ArticulationName;
  name += !AssetName.IsEmpty() ? "_" + AssetName : "";
  name += !Postfix.IsEmpty() ? "_" + Postfix : "";
  return name;
}

TArray<FMochiArticulationSkeletonLink> FMochiArticulationAssetUtils::GetSkeletonLinks(
    const UMochiArticulation* Articulation) {
  const TArray<FMochiArticulatedLink>& links = Articulation->GetLinks();
  TArray<FMochiArticulationSkeletonLink> skeletonLinks;
  skeletonLinks.SetNum(links.Num());
  for (int iLink = 0; iLink < links.Num(); ++iLink) {
    const FMochiArticulatedLink& link = links[iLink];
    FMochiArticulationSkeletonLink& skeletonLink = skeletonLinks[iLink];
    skeletonLink.Name = link.Name;
    skeletonLink.ParentIndex = link.ParentLink;
    skeletonLink.LinkToParent = Articulation->GetRestLinkToParent(iLink);
    skeletonLink.LinkToRoot = Articulation->GetRestLinkToRoot(iLink);
    skeletonLink.RenderModel = link.RenderModel;
    skeletonLink.RenderModelToLink = link.GetRenderModelTransform();
    skeletonLink.UniqueMaterialSlots = link.UniqueMaterialSlots;
  }
  return skeletonLinks;
}

UDataTable* FMochiArticulationAssetUtils::GenerateJointControlGainsTable(
    const TArray<FMochiArticulatedJoint>& Joints,
    const FString& PackagePath,
    const FString& TableAssetName,
    EObjectFlags Flags) {
  // If package already exists, try numerical suffixes until we find a new one.
  FString assetPath = FPaths::Combine(PackagePath, TableAssetName);
  UPackage* package = FindPackage(nullptr, *assetPath);
  for (int i = 1; package; ++i) {
    const FString suffixed = FString::Printf(TEXT("%s_%d"), *assetPath, i);
    package = FindPackage(nullptr, *suffixed);
    if (!package) {
      assetPath = suffixed;
    }
  }
  // Create the package and asset within
  package = CreatePackage(*assetPath);
  FString objName = FPackageName::GetLongPackageAssetName(assetPath);
  UDataTable* dataTable = NewObject<UDataTable>(package, FName(*objName), Flags);
  FAssetRegistryModule::AssetCreated(dataTable);
  FMochiJointControlGains::PopulateTable(dataTable, Joints);
  dataTable->MarkPackageDirty();
  dataTable->PostEditChange();
  return dataTable;
}

USkeleton* FMochiArticulationAssetUtils::GenerateSkeleton(
    const TArray<FMochiArticulationSkeletonLink>& Links,
    const FString& PackagePath,
    const FString& SkeletonName,
    EObjectFlags Flags) {
  const FString assetPath = FPaths::Combine(PackagePath, SkeletonName);
  USkeleton* skeleton = CreateOrReuseAsset<USkeleton>(assetPath, Flags, true);
  // One bone per link, at the link's rest transform relative to its parent.
  FReferenceSkeletonModifier boneAdder(skeleton);
  for (const FMochiArticulationSkeletonLink& link : Links) {
    boneAdder.Add(
        FMeshBoneInfo(link.Name, link.Name.ToString(), link.ParentIndex), link.LinkToParent);
  }
  skeleton->MarkPackageDirty();
  skeleton->PostEditChange();
  return skeleton;
}

USkeletalMesh* FMochiArticulationAssetUtils::GenerateSkeletalMesh(
    const TArray<FMochiArticulationSkeletonLink>& Links,
    USkeleton* Skeleton,
    const FString& PackagePath,
    const FString& SkeletalMeshAssetName,
    EObjectFlags Flags,
    const TArray<FMochiArticulationSkinnedPart>& SkinnedParts) {
  const FReferenceSkeleton& refSkeleton = Skeleton->GetReferenceSkeleton();
  const int32 numBones = refSkeleton.GetRawBoneNum();

  // Materials: those of the links' render models first, then those of the skinned parts. A
  // material shared by several render models gets one slot, unless a link keeps its slot unique.
  FMaterialListBuilder materials;
  TMap<UMaterialInterface*, int32> mergeableMaterials;
  // Links with a render model.
  struct FRigidLink {
    const UStaticMesh* RenderModel = nullptr;
    int32 Bone = INDEX_NONE;
    FMeshPart Part;
  };
  TArray<FRigidLink> rigidLinks;
  int32 numLODs = 0;
  for (int32 iLink = 0; iLink < Links.Num(); ++iLink) {
    const FMochiArticulationSkeletonLink& link = Links[iLink];
    const FStaticMeshRenderData* renderData =
        link.RenderModel ? link.RenderModel->GetRenderData() : nullptr;
    if (renderData == nullptr || renderData->LODResources.IsEmpty()) {
      continue;
    }
    numLODs = FMath::Max(numLODs, renderData->LODResources.Num());
    FRigidLink& rigidLink = rigidLinks.AddDefaulted_GetRef();
    rigidLink.RenderModel = link.RenderModel;
    rigidLink.Bone = iLink;
    FMeshPart& part = rigidLink.Part;
    part.MeshToRoot = link.RenderModelToLink * link.LinkToRoot;
    int32 numSlots = link.RenderModel->GetStaticMaterials().Num();
    for (const FStaticMeshLODResources& lod : renderData->LODResources) {
      for (const FStaticMeshSection& section : lod.Sections) {
        numSlots = FMath::Max(numSlots, section.MaterialIndex + 1);
      }
    }
    for (int32 slot = 0; slot < numSlots; ++slot) {
      UMaterialInterface* material = link.RenderModel->GetMaterial(slot);
      const bool bMergeable = !link.UniqueMaterialSlots.Contains(slot);
      const int32* existing = bMergeable ? mergeableMaterials.Find(material) : nullptr;
      if (existing) {
        part.GroupMaterials.Add(*existing);
        continue;
      }
      const int32 index =
          materials.Add(material, material ? material->GetName() : FString(TEXT("Material")));
      part.GroupMaterials.Add(index);
      if (bMergeable) {
        mergeableMaterials.Add(material, index);
      }
    }
  }
  if (numLODs == 0 && SkinnedParts.IsEmpty()) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Error,
        TEXT(
            "[MochiArticulationAssetUtils] No link has a render model and there are no skinned parts"));
    return nullptr;
  }

  TArray<FMeshPart> skinParts;
  for (const FMochiArticulationSkinnedPart& skinnedPart : SkinnedParts) {
    FMeshPart& part = skinParts.AddDefaulted_GetRef();
    part.MeshToRoot = skinnedPart.MeshToRoot;
    for (const FPolygonGroupID groupID : skinnedPart.Mesh.PolygonGroups().GetElementIDs()) {
      UMaterialInterface* material = skinnedPart.Materials.IsValidIndex(groupID.GetValue())
          ? skinnedPart.Materials[groupID.GetValue()]
          : nullptr;
      if (part.GroupMaterials.Num() <= groupID.GetValue()) {
        part.GroupMaterials.SetNum(groupID.GetValue() + 1);
      }
      part.GroupMaterials[groupID.GetValue()] =
          materials.Add(material, material ? material->GetName() : FString(TEXT("Skin")));
    }
  }

  // One mesh description per LOD: every link's render model at that LOD, bound rigidly to its
  // bone, followed by the skinned parts.
  TArray<FMeshDescription> lodMeshes;
  lodMeshes.SetNum(FMath::Max(numLODs, 1));
  for (int32 iLOD = 0; iLOD < lodMeshes.Num(); ++iLOD) {
    FMeshDescription& lodMesh = lodMeshes[iLOD];
    InitLODMesh(refSkeleton, materials.Materials, lodMesh);
    for (const FRigidLink& rigidLink : rigidLinks) {
      FMeshDescription linkMesh;
      if (MakeRigidLinkMesh(*rigidLink.RenderModel, iLOD, rigidLink.Bone, linkMesh)) {
        AppendMeshPart(linkMesh, rigidLink.Part, numBones, lodMesh);
      } else {
        UE_LOG(
            LogMochiPhysicsEditor,
            Warning,
            TEXT(
                "[MochiArticulationAssetUtils] Skipping %s in LOD %d of %s: its render geometry could not be read"),
            *rigidLink.RenderModel->GetName(),
            iLOD,
            *SkeletalMeshAssetName);
      }
    }
    for (int32 iSkin = 0; iSkin < SkinnedParts.Num(); ++iSkin) {
      AppendMeshPart(SkinnedParts[iSkin].Mesh, skinParts[iSkin], numBones, lodMesh);
    }
  }

  const FString assetPath = FPaths::Combine(PackagePath, SkeletalMeshAssetName);
  USkeletalMesh* skeletalMesh = CreateOrReuseAsset<USkeletalMesh>(assetPath, Flags, true);
  TArray<const FMeshDescription*> lodMeshPointers;
  for (const FMeshDescription& lodMesh : lodMeshes) {
    lodMeshPointers.Add(&lodMesh);
  }
  // The render models come with the normals and tangents of their own build. A mesh made only of
  // skinned parts has its normals and tangents recomputed.
  const bool bRecompute = rigidLinks.IsEmpty();
#if UE_VERSION_OLDER_THAN(5, 8, 0)
  const bool bBuilt = FStaticToSkeletalMeshConverter::InitializeSkeletalMeshFromMeshDescriptions(
      skeletalMesh, lodMeshPointers, materials.Materials, refSkeleton, bRecompute, bRecompute);
#else
  FStaticToSkeletalMeshConverter::FInitializationParams initParams;
  initParams.Materials = materials.Materials;
  initParams.bRecomputeNormals = bRecompute;
  initParams.bRecomputeTangents = bRecompute;
  const bool bBuilt = FStaticToSkeletalMeshConverter::InitializeSkeletalMeshFromMeshDescriptions(
      skeletalMesh, lodMeshPointers, refSkeleton, initParams);
#endif
  if (!bBuilt) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Error,
        TEXT("[MochiArticulationAssetUtils] Failed to build skeletal mesh %s"),
        *assetPath);
    return nullptr;
  }

  // Bounds around the vertices of LOD 0 in the reference pose.
  TArray<FVector> lod0Positions;
  const TVertexAttributesConstRef<FVector3f> positions =
      FStaticMeshConstAttributes(lodMeshes[0]).GetVertexPositions();
  for (const FVertexID vertexID : lodMeshes[0].Vertices().GetElementIDs()) {
    lod0Positions.Add(FVector(positions[vertexID]));
  }
  if (!lod0Positions.IsEmpty()) {
    skeletalMesh->SetImportedBounds(FBoxSphereBounds(lod0Positions.GetData(), lod0Positions.Num()));
  }

  skeletalMesh->SetSkeleton(Skeleton);
  Skeleton->MergeAllBonesToBoneTree(skeletalMesh);
  if (!Skeleton->GetPreviewMesh()) {
    Skeleton->SetPreviewMesh(skeletalMesh);
  }
  Skeleton->PostEditChange();

  skeletalMesh->MarkPackageDirty();
  skeletalMesh->PostEditChange();

  return skeletalMesh;
}

bool FMochiArticulationAssetUtils::GenerateSkeletalMesh(
    UMochiArticulation* Articulation,
    const FString& PackagePath) {
  if (Articulation->PreGenValidateOrUpdateDerivedData()) {
    bool bPrefix = false;
    const FString rootName =
        UMochiArticulation::RemoveAssetTypePrefix(Articulation->GetName(), &bPrefix);
    const FString skelName = MakeAssetName(
        bPrefix ? TEXT("SKEL") : TEXT(""),
        rootName,
        TEXT(""),
        bPrefix ? TEXT("") : TEXT("Skeleton"));
    const FString skelMeshName = MakeAssetName(
        bPrefix ? TEXT("SKM") : TEXT(""),
        rootName,
        TEXT(""),
        bPrefix ? TEXT("") : TEXT("SkeletalMesh"));
    constexpr EObjectFlags kFlags = RF_Public | RF_Standalone;
    const TArray<FMochiArticulationSkeletonLink> links = GetSkeletonLinks(Articulation);
    const auto skeleton = GenerateSkeleton(links, PackagePath, skelName, kFlags);
    Articulation->SkeletalMesh =
        GenerateSkeletalMesh(links, skeleton, PackagePath, skelMeshName, kFlags);
    FProperty* prop = FindFieldChecked<FProperty>(
        Articulation->StaticClass(), GET_MEMBER_NAME_CHECKED(UMochiArticulation, SkeletalMesh));
    FPropertyChangedEvent event(prop, EPropertyChangeType::ValueSet);
    Articulation->PostEditChangeProperty(event);
    Articulation->Modify(true);
    if (Articulation->OnSkeletalMeshRegenerated) {
      Articulation->OnSkeletalMeshRegenerated(Articulation->SkeletalMesh);
    }
    const bool result = Articulation->SkeletalMesh != nullptr;
    Articulation->bNeedGenerateSkeletalMesh = !result;
    return result;
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
  const FString articulationFolder =
      FPackageName::GetLongPackagePath(Articulation->GetPackage()->GetName());
  const USkeletalMesh* currentMesh = Articulation->GetSkeletalMesh();
  const FString currentMeshFolder = currentMesh
      ? FPackageName::GetLongPackagePath(currentMesh->GetPackage()->GetName())
      : FString();
  // Meshes generated before skeletal meshes moved out of "Generated" subfolders are regenerated
  // next to the articulation instead.
  const bool bRegenerateInPlace = !currentMeshFolder.IsEmpty() &&
      FPaths::GetCleanFilename(currentMeshFolder) != TEXT("Generated");
  GenerateSkeletalMesh(Articulation, bRegenerateInPlace ? currentMeshFolder : articulationFolder);
}

UDataTable* FMochiArticulationAssetUtils::GenerateJointControlGainsTable(
    UMochiArticulation* Articulation) {
  bool bPrefix = false;
  const FString rootName =
      UMochiArticulation::RemoveAssetTypePrefix(Articulation->GetName(), &bPrefix);
  if (Articulation->PreGenValidateOrUpdateDerivedData()) {
    const FString packPath =
        FPackageName::GetLongPackagePath(Articulation->GetPackage()->GetName());
    const FString gainsTableName = MakeAssetName(
        bPrefix ? TEXT("DT_Gains") : TEXT(""),
        rootName,
        TEXT(""),
        bPrefix ? TEXT("") : TEXT("DataTable"));
    UDataTable* gainsTable = GenerateJointControlGainsTable(
        Articulation->Joints, packPath, gainsTableName, RF_Public | RF_Standalone);
    return gainsTable;
  }
  return nullptr;
}
