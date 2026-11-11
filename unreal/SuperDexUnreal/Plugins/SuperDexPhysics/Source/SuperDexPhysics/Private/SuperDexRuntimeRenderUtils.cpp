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

#include "SuperDexRuntimeRenderUtils.h"

#include "SuperDexRuntimeRenderInterchange.h"

#include "Algo/Find.h"
#include "Animation/Skeleton.h"
#include "BoneWeights.h"
#include "Engine/SkeletalMesh.h"
#include "Engine/SkinnedAssetCommon.h"
#include "Engine/StaticMesh.h"
#include "Engine/Texture2D.h"
#include "HAL/PlatformTime.h"
#include "ImageCore.h"
#if SUPERDEX_WITH_RUNTIME_MESH_IMPORT
#include "GLTFAsset.h"
#include "GLTFMeshFactory.h"
#include "GLTFReader.h"
#include "InterchangeGenericAnimationPipeline.h"
#include "InterchangeGenericAssetsPipeline.h"
#include "InterchangeGenericAssetsPipelineSharedSettings.h"
#include "InterchangeGenericMaterialPipeline.h"
#include "InterchangeGenericMeshPipeline.h"
#include "InterchangeGenericTexturePipeline.h"
#include "InterchangeglTFPipeline.h"
#endif
#include "InterchangeManager.h"
#include "InterchangePipelineBase.h"
#include "InterchangeSourceData.h"
#include "Materials/MaterialInstance.h"
#include "MeshDescription.h"
#include "Misc/App.h"
#include "Misc/Paths.h"
#include "Misc/SecureHash.h"
#include "ReferenceSkeleton.h"
#include "Rendering/SkeletalMeshLODRenderData.h"
#include "Rendering/SkeletalMeshRenderData.h"
#include "RenderingThread.h"
#include "SkeletalMeshAttributes.h"
#include "StaticMeshAttributes.h"
#include "StaticMeshOperations.h"
#include "StaticMeshResources.h"
#include "TextureResource.h"

DEFINE_LOG_CATEGORY(LogSuperDexRuntimeMesh);

namespace {

// Texture inputs of the glTF reference materials that hold data rather than colour, and so must be
// sampled without sRGB decoding. Matches the colour spaces Interchange's glTF translator assigns.
const FName kLinearTextureInputs[] = {
    TEXT("NormalTexture"),
    TEXT("MetallicRoughnessTexture"),
    TEXT("OcclusionTexture"),
    TEXT("ClearCoatTexture"),
    TEXT("ClearCoatRoughnessTexture"),
    TEXT("ClearCoatNormalTexture"),
    TEXT("TransmissionTexture"),
    TEXT("SheenRoughnessTexture"),
    TEXT("IridescenceTexture"),
    TEXT("IridescenceThicknessTexture"),
    TEXT("AnisotropyTexture"),
};

// Rewrites the single mip of a G8 texture as BGRA8 with the grey value in every colour channel.
void ExpandGrayscaleToBgra8(UTexture2D& Texture) {
  FTexturePlatformData* Data = Texture.GetPlatformData();
  if (!Data || Data->Mips.Num() != 1 || !Data->Mips[0].BulkData.IsBulkDataLoaded()) {
    return;
  }
  FTexture2DMipMap& Mip = Data->Mips[0];
  const int64 NumPixels = int64(Mip.SizeX) * Mip.SizeY;
  if (Mip.BulkData.GetBulkDataSize() != NumPixels) {
    return;
  }
  TArray64<uint8> Gray;
  Gray.SetNumUninitialized(NumPixels);
  FMemory::Memcpy(Gray.GetData(), Mip.BulkData.LockReadOnly(), NumPixels);
  Mip.BulkData.Unlock();

  Mip.BulkData.Lock(LOCK_READ_WRITE);
  FColor* Colors = reinterpret_cast<FColor*>(Mip.BulkData.Realloc(NumPixels * sizeof(FColor)));
  for (int64 i = 0; i < NumPixels; ++i) {
    Colors[i] = FColor(Gray[i], Gray[i], Gray[i], 255);
  }
  Mip.BulkData.Unlock();
  Data->PixelFormat = PF_B8G8R8A8;
}

// Appends a full mip chain down to 1x1 below the single mip, filtered in the texture's colour
// space.
void BuildMipChain(UTexture2D& Texture) {
  FTexturePlatformData* Data = Texture.GetPlatformData();
  if (!Data || Data->Mips.Num() != 1 || Data->PixelFormat != PF_B8G8R8A8 ||
      !Data->Mips[0].BulkData.IsBulkDataLoaded()) {
    return;
  }
  const EGammaSpace Gamma = Texture.SRGB ? EGammaSpace::sRGB : EGammaSpace::Linear;
  constexpr FImageCore::EResizeImageFilter kFilter = FImageCore::EResizeImageFilter::Box |
      FImageCore::EResizeImageFilter::Flag_WrapX | FImageCore::EResizeImageFilter::Flag_WrapY;

  FTexture2DMipMap& Top = Data->Mips[0];
  FImage Previous(Top.SizeX, Top.SizeY, ERawImageFormat::BGRA8, Gamma);
  FMemory::Memcpy(Previous.RawData.GetData(), Top.BulkData.LockReadOnly(), Previous.RawData.Num());
  Top.BulkData.Unlock();

  while (Previous.SizeX > 1 || Previous.SizeY > 1) {
    FImage Next(
        FMath::Max(1, Previous.SizeX / 2),
        FMath::Max(1, Previous.SizeY / 2),
        ERawImageFormat::BGRA8,
        Gamma);
    FImageCore::ResizeImage(Previous, Next, kFilter);
    auto* Mip = new FTexture2DMipMap(Next.SizeX, Next.SizeY);
    Mip->BulkData.Lock(LOCK_READ_WRITE);
    FMemory::Memcpy(
        Mip->BulkData.Realloc(Next.RawData.Num()), Next.RawData.GetData(), Next.RawData.Num());
    Mip->BulkData.Unlock();
    Data->Mips.Add(Mip);
    Previous = MoveTemp(Next);
  }
  // The texture streamer can only stream mips loaded from disk; these all stay resident.
  Texture.NeverStream = true;
}

// A runtime import skips the work an editor import does when it compiles a texture: every texture
// is left sRGB with a single mip, and an all-grey image is stored single-channel. This applies what
// the editor would have produced:
// - Data textures (normal, roughness/metallic, occlusion, ...) linear, by the material slot they
//   fill. Decoding a normal map as sRGB bends every normal the same way and flattens the surface.
// - Grey images as BGRA8. A single-channel texture is sampled as (grey, 0, 0): red.
// - A full mip chain. Without one, a surface seen at less than one texel per pixel samples
//   scattered texels: the detail averages away to a flat colour under temporal AA, and the two
//   eyes of a stereo view sample different texels, so their shading differs.
void FixUpRuntimeTextures(const TArray<UObject*>& Imported) {
  TMap<UTexture2D*, bool> SRGBByTexture;
  for (UObject* Object : Imported) {
    UTexture2D* Texture = Cast<UTexture2D>(Object);
    // Only textures the runtime import created; textures built in the editor are already compiled.
    if (Texture && Texture->bNotOfflineProcessed) {
      SRGBByTexture.Add(Texture, true);
    }
  }
  if (SRGBByTexture.IsEmpty()) {
    return;
  }
  for (UObject* Object : Imported) {
    if (const UMaterialInstance* Material = Cast<UMaterialInstance>(Object)) {
      for (const FTextureParameterValue& Value : Material->TextureParameterValues) {
        bool* SRGB = SRGBByTexture.Find(Cast<UTexture2D>(Value.ParameterValue));
        if (SRGB && Algo::Find(kLinearTextureInputs, Value.ParameterInfo.Name)) {
          *SRGB = false;
        }
      }
    }
  }

  // The import has already started creating each texture's render resource from its single mip,
  // on the render thread. Changing the mips underneath it crashes, so release and flush first.
  for (const TPair<UTexture2D*, bool>& Entry : SRGBByTexture) {
    Entry.Key->ReleaseResource();
  }
  FlushRenderingCommands();
  for (const TPair<UTexture2D*, bool>& Entry : SRGBByTexture) {
    UTexture2D& Texture = *Entry.Key;
    if (Texture.GetPixelFormat() == PF_G8) {
      ExpandGrayscaleToBgra8(Texture);
    }
    Texture.SRGB = Entry.Value;
    BuildMipChain(Texture);
    Texture.UpdateResource();
  }
}

// The runtime mesh build gives every mesh the full set of UV channels and leaves the ones the file
// does not provide uninitialized (-inf). The glTF materials choose their UV channel by weighting
// every channel with 0 or 1, and 0 * -inf is NaN, so a surface whose texture reads such a mesh
// samples a single texel everywhere. Returns whether any value was replaced.
bool ZeroNonFiniteUVs(UStaticMesh& Mesh) {
  FStaticMeshRenderData* RenderData = Mesh.GetRenderData();
  if (!RenderData) {
    return false;
  }
  auto IsBad = [](const FVector2f& UV) { return !FMath::IsFinite(UV.X) || !FMath::IsFinite(UV.Y); };
  bool bAnyBad = false;
  for (const FStaticMeshLODResources& Lod : RenderData->LODResources) {
    const FStaticMeshVertexBuffer& UVs = Lod.VertexBuffers.StaticMeshVertexBuffer;
    for (uint32 V = 0; V < UVs.GetNumVertices() && !bAnyBad; ++V) {
      for (uint32 C = 0; C < UVs.GetNumTexCoords() && !bAnyBad; ++C) {
        bAnyBad = IsBad(UVs.GetVertexUV(V, C));
      }
    }
  }
  if (!bAnyBad) {
    return false;
  }
  // The render thread reads the vertex buffers, so they are rewritten with the resources released.
  Mesh.ReleaseResources();
  FlushRenderingCommands();
  for (FStaticMeshLODResources& Lod : RenderData->LODResources) {
    FStaticMeshVertexBuffer& UVs = Lod.VertexBuffers.StaticMeshVertexBuffer;
    for (uint32 V = 0; V < UVs.GetNumVertices(); ++V) {
      for (uint32 C = 0; C < UVs.GetNumTexCoords(); ++C) {
        if (IsBad(UVs.GetVertexUV(V, C))) {
          UVs.SetVertexUV(V, C, FVector2f::ZeroVector);
        }
      }
    }
  }
  Mesh.InitResources();
  return true;
}

// Verbose: what each slot of an imported mesh draws with, to diagnose a mesh that renders wrong.
void LogMeshMaterials(const FString& FilePath, const UStaticMesh& Mesh) {
  UE_LOG(LogSuperDexRuntimeMesh, Verbose, TEXT("'%s' materials:"), *FilePath);
  for (const FStaticMaterial& Slot : Mesh.GetStaticMaterials()) {
    const UMaterialInterface* Material = Slot.MaterialInterface;
    const UMaterialInstance* Instance = Cast<UMaterialInstance>(Material);
    UE_LOG(
        LogSuperDexRuntimeMesh,
        Verbose,
        TEXT("  slot '%s': %s (%s), parent %s"),
        *Slot.MaterialSlotName.ToString(),
        *GetNameSafe(Material),
        Material ? *Material->GetClass()->GetName() : TEXT("none"),
        Instance ? *GetPathNameSafe(Instance->Parent) : TEXT("none"));
    if (!Instance) {
      continue;
    }
    for (const FTextureParameterValue& Value : Instance->TextureParameterValues) {
      const UTexture2D* Texture = Cast<UTexture2D>(Value.ParameterValue);
      UE_LOG(
          LogSuperDexRuntimeMesh,
          Verbose,
          TEXT("    %s = %s %dx%d %s, %d mip(s), %s"),
          *Value.ParameterInfo.Name.ToString(),
          *GetNameSafe(Value.ParameterValue),
          Texture ? Texture->GetSizeX() : 0,
          Texture ? Texture->GetSizeY() : 0,
          Texture ? GetPixelFormatString(Texture->GetPixelFormat()) : TEXT("-"),
          Texture ? Texture->GetNumMips() : 0,
          Texture && Texture->SRGB ? TEXT("sRGB") : TEXT("linear"));
    }
    for (const FScalarParameterValue& Value : Instance->ScalarParameterValues) {
      UE_LOG(
          LogSuperDexRuntimeMesh,
          Verbose,
          TEXT("    %s = %g"),
          *Value.ParameterInfo.Name.ToString(),
          Value.ParameterValue);
    }
    for (const FVectorParameterValue& Value : Instance->VectorParameterValues) {
      UE_LOG(
          LogSuperDexRuntimeMesh,
          Verbose,
          TEXT("    %s = %s"),
          *Value.ParameterInfo.Name.ToString(),
          *Value.ParameterValue.ToString());
    }
  }
}

// Most influences kept per vertex.
constexpr int32 kMaxInfluences = 8;

// Creates a transient skeletal mesh from Mesh (as SuperDexRuntimeRender::ReadSkinnedGlb returns
// it) over a flat skeleton: bone 0 is a root at the component origin, and bone i + 1 is named
// BoneNames[i], parented to the root, with reference pose RestBoneToComponent[i]. The skin weights
// of Mesh index these bones. Mesh is placed by MeshToComponent. Materials holds the material of
// each polygon group, indexed by polygon group ID; a missing or null entry renders with the default
// material. Interchange cannot create skeletal meshes outside the editor, and
// USkeletalMesh::Build / CommitMeshDescription are editor-only, so this fills the render data
// directly, as the engine's runtime FSkeletalMeshMerge does. Returns null on failure.
USkeletalMesh* CreateTransientSkeletalMesh(
    UObject* Outer,
    const FMeshDescription& Mesh,
    const FTransform& MeshToComponent,
    TConstArrayView<FName> BoneNames,
    TConstArrayView<FTransform> RestBoneToComponent,
    TConstArrayView<UMaterialInterface*> Materials) {
  check(IsInGameThread());
  if (BoneNames.Num() != RestBoneToComponent.Num() || Mesh.Triangles().Num() == 0) {
    return nullptr;
  }
  const int32 NumBones = BoneNames.Num() + 1;

  FSkeletalMeshConstAttributes Attributes(Mesh);
  TVertexAttributesConstRef<FVector3f> Positions = Attributes.GetVertexPositions();
  TVertexInstanceAttributesConstRef<FVector3f> Normals = Attributes.GetVertexInstanceNormals();
  TVertexInstanceAttributesConstRef<FVector3f> Tangents = Attributes.GetVertexInstanceTangents();
  TVertexInstanceAttributesConstRef<float> BinormalSigns =
      Attributes.GetVertexInstanceBinormalSigns();
  TVertexInstanceAttributesConstRef<FVector2f> UVs = Attributes.GetVertexInstanceUVs();
  FSkinWeightsVertexAttributesConstRef SkinWeights = Attributes.GetVertexSkinWeights();
  if (!SkinWeights.IsValid()) {
    return nullptr;
  }
  const int32 NumUVs = FMath::Clamp(UVs.GetNumChannels(), 1, int32(MAX_TEXCOORDS));
  // A mirroring placement flips the triangles' winding and the tangent frame's handedness.
  const bool bMirrored = MeshToComponent.GetDeterminant() < 0.f;
  const FMatrix MeshToComponentMatrix = MeshToComponent.ToMatrixWithScale();
  // The inverse transpose up to a positive factor, signed so mirrored normals still point out.
  const FMatrix NormalToComponent =
      MeshToComponentMatrix.TransposeAdjoint() * (bMirrored ? -1.0 : 1.0);

  // The triangles of each polygon group: each group becomes one render section.
  TMap<FPolygonGroupID, TArray<FTriangleID>> TrianglesByGroup;
  for (const FTriangleID TriangleID : Mesh.Triangles().GetElementIDs()) {
    TrianglesByGroup.FindOrAdd(Mesh.GetTrianglePolygonGroup(TriangleID)).Add(TriangleID);
  }
  TrianglesByGroup.KeySort([](FPolygonGroupID A, FPolygonGroupID B) { return A < B; });

  USkeletalMesh* SkeletalMesh = NewObject<USkeletalMesh>(Outer, NAME_None, RF_Transient);

  FReferenceSkeleton RefSkeleton;
  {
    FReferenceSkeletonModifier Modifier(RefSkeleton, nullptr);
    Modifier.Add(FMeshBoneInfo(TEXT("root"), TEXT("root"), INDEX_NONE), FTransform::Identity);
    for (int32 i = 0; i < BoneNames.Num(); ++i) {
      FTransform Rest = RestBoneToComponent[i];
      Rest.SetScale3D(FVector::OneVector);
      Rest.NormalizeRotation();
      Modifier.Add(FMeshBoneInfo(BoneNames[i], BoneNames[i].ToString(), 0), Rest);
    }
  }
  SkeletalMesh->SetRefSkeleton(RefSkeleton);
  SkeletalMesh->CalculateInvRefMatrices();

  FSkeletalMeshLODInfo& LODInfo = SkeletalMesh->AddLODInfo();
  LODInfo.BuildSettings.bUseFullPrecisionUVs = true;

  SkeletalMesh->AllocateResourceForRendering();
  FSkeletalMeshRenderData* RenderData = SkeletalMesh->GetResourceForRendering();
  FSkeletalMeshLODRenderData* LODData = new FSkeletalMeshLODRenderData();
  RenderData->LODRenderData.Add(LODData);

  TArray<FBoneIndexType> AllBones;
  for (int32 Bone = 0; Bone < NumBones; ++Bone) {
    AllBones.Add(static_cast<FBoneIndexType>(Bone));
  }

  struct FRenderVertex {
    FVector3f Position;
    FVector3f TangentX;
    FVector3f TangentY;
    FVector3f TangentZ;
    FVector2f UVs[MAX_TEXCOORDS];
  };
  TArray<FRenderVertex> Vertices;
  TArray<FSkinWeightInfo> Weights;
  TArray<uint32> Indices;
  uint32 MaxBoneInfluences = 1;
  FBox Bounds(ForceInit);

  for (const TPair<FPolygonGroupID, TArray<FTriangleID>>& Group : TrianglesByGroup) {
    FSkelMeshRenderSection& Section = LODData->RenderSections.AddDefaulted_GetRef();
    Section.MaterialIndex = SkeletalMesh->GetMaterials().Num();
    Section.BaseIndex = Indices.Num();
    Section.NumTriangles = Group.Value.Num();
    Section.BaseVertexIndex = Vertices.Num();
    Section.BoneMap = AllBones;

    UMaterialInterface* Material =
        Materials.IsValidIndex(Group.Key.GetValue()) ? Materials[Group.Key.GetValue()] : nullptr;
    const FName SlotName = Attributes.GetPolygonGroupMaterialSlotNames()[Group.Key];
    SkeletalMesh->GetMaterials().Add(FSkeletalMaterial(Material, SlotName));

    // One render vertex per vertex instance of the group.
    TMap<FVertexInstanceID, uint32> RenderIndexByInstance;
    for (const FTriangleID TriangleID : Group.Value) {
      TArrayView<const FVertexInstanceID> Corners = Mesh.GetTriangleVertexInstances(TriangleID);
      uint32 Triangle[3];
      for (int32 Corner = 0; Corner < 3; ++Corner) {
        const FVertexInstanceID InstanceID = Corners[Corner];
        if (const uint32* Existing = RenderIndexByInstance.Find(InstanceID)) {
          Triangle[Corner] = *Existing;
          continue;
        }
        const FVertexID VertexID = Mesh.GetVertexInstanceVertex(InstanceID);
        FRenderVertex& V = Vertices.AddDefaulted_GetRef();
        V.Position = FVector3f(MeshToComponent.TransformPosition(FVector(Positions[VertexID])));
        Bounds += FVector(V.Position);
        // Normals take the inverse transpose of the placement, which keeps them perpendicular to
        // the surface under non-uniform scale; tangents lie in the surface and take the placement.
        const FVector Normal =
            NormalToComponent.TransformVector(FVector(Normals[InstanceID])).GetSafeNormal();
        FVector Tangent =
            MeshToComponentMatrix.TransformVector(FVector(Tangents[InstanceID])).GetSafeNormal();
        Tangent = (Tangent - FVector::DotProduct(Tangent, Normal) * Normal).GetSafeNormal();
        if (Tangent.IsNearlyZero()) {
          Tangent = FVector::CrossProduct(Normal, FVector::UpVector).GetSafeNormal();
        }
        if (Tangent.IsNearlyZero()) {
          Tangent = FVector::CrossProduct(Normal, FVector::ForwardVector).GetSafeNormal();
        }
        const float Sign = BinormalSigns[InstanceID] * (bMirrored ? -1.f : 1.f);
        V.TangentX = FVector3f(Tangent);
        V.TangentZ = FVector3f(Normal);
        V.TangentY = FVector3f(FVector::CrossProduct(Normal, Tangent).GetSafeNormal() * Sign);
        for (int32 Channel = 0; Channel < NumUVs; ++Channel) {
          V.UVs[Channel] =
              Channel < UVs.GetNumChannels() ? UVs.Get(InstanceID, Channel) : FVector2f::ZeroVector;
        }

        // The strongest influences, quantized to 16 bits summing to 65535.
        TArray<UE::AnimationCore::FBoneWeight, TInlineAllocator<kMaxInfluences * 2>> Influences;
        for (const UE::AnimationCore::FBoneWeight BoneWeight : SkinWeights.Get(VertexID)) {
          if (BoneWeight.GetWeight() > 0.f && BoneWeight.GetBoneIndex() < NumBones) {
            Influences.Add(BoneWeight);
          }
        }
        Influences.Sort([](const auto& A, const auto& B) { return A.GetWeight() > B.GetWeight(); });
        if (Influences.Num() > kMaxInfluences) {
          Influences.SetNum(kMaxInfluences);
        }
        FSkinWeightInfo& Weight = Weights.AddZeroed_GetRef();
        if (Influences.IsEmpty()) {
          Weight.InfluenceBones[0] = 0;
          Weight.InfluenceWeights[0] = MAX_uint16;
        } else {
          float Total = 0.f;
          for (const auto& Influence : Influences) {
            Total += Influence.GetWeight();
          }
          int32 Remaining = MAX_uint16;
          for (int32 i = 0; i < Influences.Num(); ++i) {
            const int32 Quantized = i + 1 == Influences.Num()
                ? Remaining
                : FMath::Min(
                      Remaining,
                      FMath::RoundToInt(Influences[i].GetWeight() / Total * float(MAX_uint16)));
            Weight.InfluenceBones[i] = static_cast<FBoneIndexType>(Influences[i].GetBoneIndex());
            Weight.InfluenceWeights[i] = static_cast<uint16>(Quantized);
            Remaining -= Quantized;
          }
          MaxBoneInfluences = FMath::Max(MaxBoneInfluences, uint32(Influences.Num()));
        }

        Triangle[Corner] = Vertices.Num() - 1;
        RenderIndexByInstance.Add(InstanceID, Triangle[Corner]);
      }
      Indices.Add(Triangle[0]);
      Indices.Add(bMirrored ? Triangle[2] : Triangle[1]);
      Indices.Add(bMirrored ? Triangle[1] : Triangle[2]);
    }

    Section.NumVertices = Vertices.Num() - Section.BaseVertexIndex;
    // No overlapping vertices: the buffer only has to exist, sized to the section.
    Section.DuplicatedVerticesBuffer.DupVertData.ResizeBuffer(1);
    Section.DuplicatedVerticesBuffer.DupVertIndexData.ResizeBuffer(Section.NumVertices);
    FMemory::Memzero(
        Section.DuplicatedVerticesBuffer.DupVertIndexData.GetDataPointer(),
        Section.NumVertices * sizeof(FIndexLengthPair));
    FMemory::Memzero(Section.DuplicatedVerticesBuffer.DupVertData.GetDataPointer(), sizeof(uint32));
    Section.DuplicatedVerticesBuffer.bHasOverlappingVertices = false;
  }
  for (FSkelMeshRenderSection& Section : LODData->RenderSections) {
    Section.MaxBoneInfluences = MaxBoneInfluences;
  }

  LODData->ActiveBoneIndices = AllBones;
  LODData->RequiredBones = AllBones;

  constexpr bool bNeedsCPUAccess = false;
  FStaticMeshVertexBuffers& VertexBuffers = LODData->StaticVertexBuffers;
  VertexBuffers.PositionVertexBuffer.Init(Vertices.Num(), bNeedsCPUAccess);
  VertexBuffers.StaticMeshVertexBuffer.SetUseFullPrecisionUVs(true);
  VertexBuffers.StaticMeshVertexBuffer.Init(Vertices.Num(), NumUVs, bNeedsCPUAccess);
  for (int32 i = 0; i < Vertices.Num(); ++i) {
    const FRenderVertex& V = Vertices[i];
    VertexBuffers.PositionVertexBuffer.VertexPosition(i) = V.Position;
    VertexBuffers.StaticMeshVertexBuffer.SetVertexTangents(i, V.TangentX, V.TangentY, V.TangentZ);
    for (int32 Channel = 0; Channel < NumUVs; ++Channel) {
      VertexBuffers.StaticMeshVertexBuffer.SetVertexUV(i, Channel, V.UVs[Channel]);
    }
  }

  LODData->SkinWeightVertexBuffer.SetMaxBoneInfluences(MaxBoneInfluences);
  LODData->SkinWeightVertexBuffer.SetUse16BitBoneIndex(NumBones > MAX_uint8);
  LODData->SkinWeightVertexBuffer.SetUse16BitBoneWeight(true);
  LODData->SkinWeightVertexBuffer.SetNeedsCPUAccess(bNeedsCPUAccess);
  LODData->SkinWeightVertexBuffer = Weights;

  LODData->MultiSizeIndexContainer.RebuildIndexBuffer(sizeof(uint32), Indices);

  // Bounds in the reference pose; deformation can move the surface beyond them.
  SkeletalMesh->SetImportedBounds(FBoxSphereBounds(Bounds.ExpandBy(Bounds.GetExtent().GetMax())));
  // There is nothing on disk to stream from.
  SkeletalMesh->NeverStream = true;

  USkeleton* Skeleton = NewObject<USkeleton>(Outer, NAME_None, RF_Transient);
  Skeleton->MergeAllBonesToBoneTree(SkeletalMesh, /*bShowProgress=*/false);
  SkeletalMesh->SetSkeleton(Skeleton);

  SkeletalMesh->InitResources();
  return SkeletalMesh;
}

} // namespace

namespace SuperDexRuntimeRender {

bool ReadSkinnedGlb(
    const FString& GlbPath,
    TConstArrayView<int32> JointToBone,
    FMeshDescription& OutMesh,
    TArray<FString>& OutMaterialNames) {
#if !SUPERDEX_WITH_RUNTIME_MESH_IMPORT
  return false;
#else
  GLTF::FAsset Asset;
  GLTF::FFileReader Reader;
  Reader.ReadFile(GlbPath, /*bInLoadImageData=*/false, /*bInLoadMetadata=*/false, Asset);
  for (const GLTF::FLogMessage& Message : Reader.GetLogMessages()) {
    if (Message.Get<0>() == GLTF::EMessageSeverity::Error) {
      UE_LOG(
          LogSuperDexRuntimeMesh,
          Error,
          TEXT("Failed to read %s: %s"),
          *GlbPath,
          *Message.Get<1>().ToString());
      return false;
    }
  }
  const GLTF::FNode* SkinnedNode = Asset.Nodes.FindByPredicate([&Asset](const GLTF::FNode& Node) {
    return Asset.Meshes.IsValidIndex(Node.MeshIndex) && Asset.Skins.IsValidIndex(Node.Skindex);
  });
  if (!SkinnedNode) {
    UE_LOG(LogSuperDexRuntimeMesh, Error, TEXT("%s has no skinned mesh"), *GlbPath);
    return false;
  }
  const GLTF::FMesh& Mesh = Asset.Meshes[SkinnedNode->MeshIndex];

  // glTF ignores a skinned mesh node's own transform: vertices are in the skeleton's bind space.
  // glTF is in meters, Unreal in centimeters.
  constexpr float kGltfToUnrealScale = 100.f;
  GLTF::FMeshFactory MeshFactory;
  MeshFactory.SetUniformScale(kGltfToUnrealScale);
  OutMesh.Empty();
  MeshFactory.FillMeshDescription(Mesh, FTransform::Identity, &OutMesh);

  // The mesh factory names each polygon group after the index of its glTF material.
  TPolygonGroupAttributesRef<FName> SlotNames =
      FStaticMeshAttributes(OutMesh).GetPolygonGroupMaterialSlotNames();
  OutMaterialNames.Reset();
  OutMaterialNames.SetNum(OutMesh.PolygonGroups().GetArraySize());
  for (const FPolygonGroupID GroupID : OutMesh.PolygonGroups().GetElementIDs()) {
    int32 MaterialIndex = INDEX_NONE;
    LexFromString(MaterialIndex, *SlotNames[GroupID].ToString());
    if (Asset.Materials.IsValidIndex(MaterialIndex)) {
      OutMaterialNames[GroupID.GetValue()] = Asset.Materials[MaterialIndex].Name;
      SlotNames[GroupID] = FName(*Asset.Materials[MaterialIndex].Name);
    }
  }

  FSkeletalMeshAttributes SkeletalAttributes(OutMesh);
  SkeletalAttributes.Register(/*bKeepExistingAttribute=*/true);
  FSkinWeightsVertexAttributesRef SkinWeights = SkeletalAttributes.GetVertexSkinWeights();
  const TArray<GLTF::FMeshFactory::FIndexVertexIdMap>& PositionToVertexPerPrimitive =
      MeshFactory.GetPositionIndexToVertexIdPerPrim();
  for (int32 PrimIndex = 0; PrimIndex < Mesh.Primitives.Num(); ++PrimIndex) {
    const GLTF::FPrimitive& Primitive = Mesh.Primitives[PrimIndex];
    if (!Primitive.IsValid() || !Primitive.HasJointWeights() ||
        !PositionToVertexPerPrimitive.IsValidIndex(PrimIndex)) {
      continue;
    }
    TArray<GLTF::FJointInfluence> Influences;
    Primitive.GetJointInfluences(Influences);
    for (int32 PositionIndex = 0; PositionIndex < Influences.Num(); ++PositionIndex) {
      const FVertexID* VertexID = PositionToVertexPerPrimitive[PrimIndex].Find(PositionIndex);
      if (!VertexID) {
        continue;
      }
      constexpr int32 kNumSlots = UE_ARRAY_COUNT(GLTF::FJointInfluence::ID);
      TArray<UE::AnimationCore::FBoneWeight, TInlineAllocator<kNumSlots>> BoneWeights;
      for (int32 Slot = 0; Slot < kNumSlots; ++Slot) {
        const int32 Joint = Influences[PositionIndex].ID[Slot];
        const float Weight = static_cast<float>(Influences[PositionIndex].Weight[Slot]);
        const int32 Bone = JointToBone.IsValidIndex(Joint) ? JointToBone[Joint] : INDEX_NONE;
        if (Weight <= 0.f || Bone == INDEX_NONE) {
          continue;
        }
        UE::AnimationCore::FBoneWeight* Existing =
            BoneWeights.FindByPredicate([Bone](const UE::AnimationCore::FBoneWeight& BoneWeight) {
              return BoneWeight.GetBoneIndex() == Bone;
            });
        if (Existing) {
          Existing->SetWeight(Existing->GetWeight() + Weight);
        } else {
          BoneWeights.Emplace(static_cast<FBoneIndexType>(Bone), Weight);
        }
      }
      if (BoneWeights.IsEmpty()) {
        BoneWeights.Emplace(static_cast<FBoneIndexType>(0), 1.f);
      }
      // Normalized by the attribute.
      SkinWeights.Set(*VertexID, MakeArrayView(BoneWeights));
    }
  }

  // Tangents are optional in glTF; skinned meshes are built without recomputing them.
  FStaticMeshOperations::ComputeTriangleTangentsAndNormals(OutMesh, 0.f, *GlbPath);
  FStaticMeshOperations::RecomputeNormalsAndTangentsIfNeeded(
      OutMesh, EComputeNTBsFlags::UseMikkTSpace | EComputeNTBsFlags::BlendOverlappingNormals);
  return true;
#endif
}

UMaterialInterface* FindMaterialForSlot(
    const UStaticMesh& StaticMesh,
    const FString& MaterialName) {
  const FName SlotName(*MaterialName);
  const FStaticMaterial* Slot =
      StaticMesh.GetStaticMaterials().FindByPredicate([&SlotName](const FStaticMaterial& Material) {
        return Material.ImportedMaterialSlotName == SlotName ||
            Material.MaterialSlotName == SlotName;
      });
  return Slot ? Slot->MaterialInterface.Get() : StaticMesh.GetMaterial(0);
}

} // namespace SuperDexRuntimeRender

void USuperDexRuntimeMeshSubsystem::Initialize(FSubsystemCollectionBase& Collection) {
  Super::Initialize(Collection);

#if WITH_EDITOR
  if (!GIsEditor) {
    UInterchangeManager::GetInterchangeManager().SetEditorUtilities(
        USuperDexInterchangeGameUtilities::StaticClass());
  }
#endif

#if SUPERDEX_WITH_RUNTIME_MESH_IMPORT
  // Mirrors the non-default settings of /SuperDexPhysics/Editor/MochiAssetsPipeline, which
  // the editor importers use, so runtime and editor meshes agree. That asset is editor content and
  // is not cooked. Collision is also off: these meshes only render.
  UInterchangeGenericAssetsPipeline* AssetsPipeline =
      NewObject<UInterchangeGenericAssetsPipeline>(GetTransientPackage());
  AssetsPipeline->CommonMeshesProperties->ForceAllMeshAsType =
      EInterchangeForceMeshType::IFMT_StaticMesh;
  AssetsPipeline->CommonMeshesProperties->bRecomputeNormals = false;
  AssetsPipeline->CommonMeshesProperties->bComputeWeightedNormals = true;
  // Newer engines replaced bCombineStaticMeshes with the CombineStaticMeshesBehavior enum.
  [](auto* MeshPipeline) {
    if constexpr (requires { MeshPipeline->CombineStaticMeshesBehavior; }) {
      MeshPipeline->CombineStaticMeshesBehavior =
          decltype(MeshPipeline->CombineStaticMeshesBehavior)::All;
    } else {
      MeshPipeline->bCombineStaticMeshes = true;
    }
  }(AssetsPipeline->MeshPipeline.Get());
  AssetsPipeline->MeshPipeline->bBuildNanite = false;
  AssetsPipeline->MeshPipeline->bImportSkeletalMeshes = false;
  AssetsPipeline->MeshPipeline->bCollision = false;
  AssetsPipeline->AnimationPipeline->bImportAnimations = false;
  AssetsPipeline->MaterialPipeline->TexturePipeline->bAllowNonPowerOfTwo = true;
  Pipeline = AssetsPipeline;

  // Runs after the assets pipeline, as in the engine's default glTF stack. At runtime the assets
  // pipeline can only make instances of a generic parent and drops the glTF material values; this
  // replaces them with instances of the glTF reference materials carrying those values.
  GltfPipeline = NewObject<UInterchangeGLTFPipeline>(GetTransientPackage());
  RuntimeMeshPipeline = NewObject<USuperDexRuntimeMeshPipeline>(GetTransientPackage());
#endif
}

void USuperDexRuntimeMeshSubsystem::Deinitialize() {
  bDeinitialized = true;
  // Imports are created RF_Standalone, which would otherwise outlive the game instance.
  for (UObject* Object : ImportedObjects) {
    if (IsValid(Object)) {
      Object->ClearFlags(RF_Standalone);
    }
  }
  ImportedObjects.Reset();
  MeshesByContent.Reset();
  SkinnedMeshesByKey.Reset();
  FailedContent.Reset();
  PendingByContent.Reset();
  ContentKeyByPath.Reset();
  Pipeline = nullptr;
  GltfPipeline = nullptr;
  RuntimeMeshPipeline = nullptr;
  Super::Deinitialize();
}

void USuperDexRuntimeMeshSubsystem::RequestStaticMesh(
    const FString& FilePath,
    FOnMeshLoaded OnLoaded) {
  check(IsInGameThread());
  if (!FApp::CanEverRender() || !Pipeline || !GltfPipeline) {
    OnLoaded(nullptr);
    return;
  }

  const FString* CachedKey = ContentKeyByPath.Find(FilePath);
  FString ContentKey;
  if (CachedKey) {
    ContentKey = *CachedKey;
  } else {
    const FMD5Hash Hash = FMD5Hash::HashFile(*FilePath);
    if (!Hash.IsValid()) {
      UE_LOG(LogSuperDexRuntimeMesh, Warning, TEXT("Runtime mesh: cannot read '%s'."), *FilePath);
      OnLoaded(nullptr);
      return;
    }
    ContentKey = LexToString(Hash);
    ContentKeyByPath.Add(FilePath, ContentKey);
  }

  if (TObjectPtr<UStaticMesh>* Mesh = MeshesByContent.Find(ContentKey)) {
    OnLoaded(*Mesh);
    return;
  }
  if (FailedContent.Contains(ContentKey)) {
    OnLoaded(nullptr);
    return;
  }
  if (TArray<FOnMeshLoaded>* Pending = PendingByContent.Find(ContentKey)) {
    Pending->Add(MoveTemp(OnLoaded));
    return;
  }
  // Registered before importing: a failed import can complete inside ImportAssetAsync.
  PendingByContent.Add(ContentKey).Add(MoveTemp(OnLoaded));

  FImportAssetParameters Params;
  Params.bIsAutomated = true;
  Params.OverridePipelines.Add(FSoftObjectPath(Pipeline.Get()));
  Params.OverridePipelines.Add(FSoftObjectPath(GltfPipeline.Get()));
  Params.OverridePipelines.Add(FSoftObjectPath(RuntimeMeshPipeline.Get()));
  Params.DestinationName = FPaths::GetBaseFilename(FilePath);

  const double StartSeconds = FPlatformTime::Seconds();
  UE::Interchange::FAssetImportResultRef Result =
      UInterchangeManager::GetInterchangeManager().ImportAssetAsync(
          FString::Printf(TEXT("%s/%s"), USuperDexRuntimeMeshPipeline::ImportRoot, *ContentKey),
          UInterchangeManager::CreateSourceData(FilePath),
          Params);
  Result->OnDone([WeakThis = TWeakObjectPtr<USuperDexRuntimeMeshSubsystem>(this),
                  ContentKey,
                  FilePath,
                  StartSeconds](UE::Interchange::FImportResult& Done) {
    UStaticMesh* Mesh = Cast<UStaticMesh>(Done.GetFirstAssetOfClass(UStaticMesh::StaticClass()));
    if (Mesh) {
      UE_LOG(
          LogSuperDexRuntimeMesh,
          Verbose,
          TEXT("Runtime mesh: imported '%s' in %.2f s."),
          *FilePath,
          FPlatformTime::Seconds() - StartSeconds);
    } else {
      UE_LOG(
          LogSuperDexRuntimeMesh, Warning, TEXT("Runtime mesh: failed to import '%s'."), *FilePath);
    }
    USuperDexRuntimeMeshSubsystem* Self = WeakThis.Get();
    if (!Self || Self->bDeinitialized) {
      // Deinitialize has already released the other imports; release this late one too.
      for (UObject* Object : Done.GetImportedObjects()) {
        if (IsValid(Object)) {
          Object->ClearFlags(RF_Standalone);
        }
      }
      return;
    }
    Self->OnImportDone(ContentKey, FilePath, Mesh, Done.GetImportedObjects());
  });
}

void USuperDexRuntimeMeshSubsystem::OnImportDone(
    const FString& ContentKey,
    const FString& FilePath,
    UStaticMesh* Mesh,
    const TArray<UObject*>& Imported) {
  ImportedObjects.Append(Imported);

  const double FixUpStart = FPlatformTime::Seconds();
  FixUpRuntimeTextures(Imported);
  if (Mesh && ZeroNonFiniteUVs(*Mesh)) {
    UE_LOG(
        LogSuperDexRuntimeMesh,
        Verbose,
        TEXT("Runtime mesh: zeroed undefined UVs in '%s'."),
        *FilePath);
  }
  UE_LOG(
      LogSuperDexRuntimeMesh,
      Verbose,
      TEXT("Runtime mesh: fixed up textures of '%s' in %.2f s."),
      *FilePath,
      FPlatformTime::Seconds() - FixUpStart);

  // Without this, a packaged build draws the imported materials with the reference material's
  // placeholder textures (white) instead of their own: the materials' cached texture bindings are
  // not refreshed once the runtime-created textures have their render resources. In an editor build
  // texture compilation refreshes them, which is why only packaged builds lost their textures.
  for (UObject* Object : Imported) {
    if (UMaterialInstance* Material = Cast<UMaterialInstance>(Object)) {
      if (!Material->Parent) {
        UE_LOG(
            LogSuperDexRuntimeMesh,
            Warning,
            TEXT(
                "Runtime mesh: material '%s' of '%s' has no parent; is /InterchangeAssets/gltf "
                "cooked (DirectoriesToAlwaysCook)?"),
            *Material->GetName(),
            *FilePath);
      }
      Material->RecacheUniformExpressions(false);
    }
  }

  if (Mesh && UE_LOG_ACTIVE(LogSuperDexRuntimeMesh, Verbose)) {
    LogMeshMaterials(FilePath, *Mesh);
  }

  if (Mesh) {
    MeshesByContent.Add(ContentKey, Mesh);
  } else {
    FailedContent.Add(ContentKey);
  }
  TArray<FOnMeshLoaded> Callbacks;
  PendingByContent.RemoveAndCopyValue(ContentKey, Callbacks);
  for (FOnMeshLoaded& Callback : Callbacks) {
    Callback(Mesh);
  }
}

void USuperDexRuntimeMeshSubsystem::RequestSkinnedMesh(
    FSuperDexSkinnedMeshRequest Request,
    FOnSkinnedMeshLoaded OnLoaded) {
  check(IsInGameThread());
  // The static mesh supplies the materials, and caches the file's content key.
  const FString FilePath = Request.FilePath;
  RequestStaticMesh(
      FilePath,
      [WeakThis = TWeakObjectPtr<USuperDexRuntimeMeshSubsystem>(this),
       Request = MoveTemp(Request),
       OnLoaded = MoveTemp(OnLoaded)](UStaticMesh* StaticMesh) {
        USuperDexRuntimeMeshSubsystem* Self = WeakThis.Get();
        const FString* ContentKey = Self ? Self->ContentKeyByPath.Find(Request.FilePath) : nullptr;
        if (!Self || Self->bDeinitialized || !ContentKey) {
          OnLoaded(nullptr);
          return;
        }

        // The skeleton is part of the key: the same file skinned over a different rest pose is a
        // different mesh.
        FMD5 Hash;
        auto Append = [&Hash](const void* Data, int64 Size) {
          Hash.Update(static_cast<const uint8*>(Data), Size);
        };
        auto AppendTransform = [&Append](const FTransform& Transform) {
          const FVector Translation = Transform.GetTranslation();
          const FQuat Rotation = Transform.GetRotation();
          const FVector Scale = Transform.GetScale3D();
          Append(&Translation, sizeof(Translation));
          Append(&Rotation, sizeof(Rotation));
          Append(&Scale, sizeof(Scale));
        };
        const FTCHARToUTF8 ContentKeyUtf8(**ContentKey);
        Append(ContentKeyUtf8.Get(), ContentKeyUtf8.Length());
        Append(Request.JointToBone.GetData(), Request.JointToBone.Num() * sizeof(int32));
        for (const FName& BoneName : Request.BoneNames) {
          const FTCHARToUTF8 NameUtf8(*BoneName.ToString());
          Append(NameUtf8.Get(), NameUtf8.Length() + 1);
        }
        AppendTransform(Request.MeshToComponent);
        for (const FTransform& Rest : Request.RestBoneToComponent) {
          AppendTransform(Rest);
        }
        uint8 Digest[16];
        Hash.Final(Digest);
        const FString Key = BytesToHex(Digest, UE_ARRAY_COUNT(Digest));
        if (const TObjectPtr<USkeletalMesh>* Cached = Self->SkinnedMeshesByKey.Find(Key)) {
          OnLoaded(*Cached);
          return;
        }

        const double StartSeconds = FPlatformTime::Seconds();
        FMeshDescription Mesh;
        TArray<FString> MaterialNames;
        USkeletalMesh* SkeletalMesh = nullptr;
        if (SuperDexRuntimeRender::ReadSkinnedGlb(
                Request.FilePath, Request.JointToBone, Mesh, MaterialNames)) {
          TArray<UMaterialInterface*> Materials;
          for (const FString& MaterialName : MaterialNames) {
            Materials.Add(
                StaticMesh ? SuperDexRuntimeRender::FindMaterialForSlot(*StaticMesh, MaterialName)
                           : nullptr);
          }
          SkeletalMesh = CreateTransientSkeletalMesh(
              Self,
              Mesh,
              Request.MeshToComponent,
              Request.BoneNames,
              Request.RestBoneToComponent,
              Materials);
        }
        if (SkeletalMesh) {
          UE_LOG(
              LogSuperDexRuntimeMesh,
              Log,
              TEXT("Runtime mesh: built skinned mesh of '%s' (%d bones) in %.2f s."),
              *Request.FilePath,
              Request.BoneNames.Num(),
              FPlatformTime::Seconds() - StartSeconds);
        } else {
          UE_LOG(
              LogSuperDexRuntimeMesh,
              Warning,
              TEXT("Runtime mesh: failed to build the skinned mesh of '%s'."),
              *Request.FilePath);
        }
        Self->SkinnedMeshesByKey.Add(Key, SkeletalMesh);
        OnLoaded(SkeletalMesh);
      });
}
