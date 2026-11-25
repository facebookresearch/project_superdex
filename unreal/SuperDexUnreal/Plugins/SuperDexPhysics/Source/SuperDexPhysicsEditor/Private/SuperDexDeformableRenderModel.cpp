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

#include "SuperDexDeformableRenderModel.h"

#include "Algo/AnyOf.h"
#include "GLTFAsset.h"
#include "GLTFReader.h"
#include "MochiGenTypes.h"

namespace SuperDexDeformableRenderModel {

bool ReadRenderModelAttributes(
    const FString& Path,
    FRenderModelAttributes& OutAttributes,
    FString& OutReason) {
  GLTF::FAsset Asset;
  GLTF::FFileReader Reader;
  Reader.ReadFile(Path, /*bInLoadImageData=*/false, /*bInLoadMetadata=*/false, Asset);
  for (const GLTF::FLogMessage& Message : Reader.GetLogMessages()) {
    if (Message.Get<0>() == GLTF::EMessageSeverity::Error) {
      OutReason =
          FString::Printf(TEXT("The GLB could not be read: %s"), *Message.Get<1>().ToString());
      return false;
    }
  }
  // The reader records required extensions it can't decode, such as EXT_meshopt_compression,
  // without failing; their accessors would then hold no data. Only the extensions GLTFCore lists
  // as supported are accepted, which conservatively excludes some it does decode.
  for (const FString& Extension : Asset.ExtensionsRequired) {
    if (!Algo::AnyOf(GLTF::FAsset::SupportedExtensions, [&Extension](GLTF::EExtension Supported) {
          return Extension.Equals(GLTF::ToString(Supported), ESearchCase::CaseSensitive);
        })) {
      OutReason = FString::Printf(
          TEXT(
              "The GLB requires %s, which deformable-actor render models don't support in Unreal."),
          *Extension);
      return false;
    }
  }
  if (Asset.Meshes.Num() != 1 || Asset.Meshes[0].Primitives.Num() != 1) {
    OutReason = TEXT("The GLB does not have exactly one mesh primitive.");
    return false;
  }
  GLTF::FPrimitive& Primitive = Asset.Meshes[0].Primitives[0];
  if (!Primitive.IsValid() || !Primitive.HasPositions()) {
    OutReason = TEXT("The GLB's primitive geometry is invalid.");
    return false;
  }
  // The imported material may sample TEXCOORD_1, which the deformable mesh does not supply.
  if (Primitive.HasTexCoords(1)) {
    OutReason = TEXT("The GLB's primitive has a texture-coordinate set other than TEXCOORD_0.");
    return false;
  }

  // Raw accessor data, in file order: GLTF::FMeshFactory would weld and reorder vertices.
  OutAttributes.NumVertices =
      Asset.Accessors[Primitive.GetAttributeAccessorIndex(GLTF::EMeshAttributeType::POSITION)]
          .Count;
  OutAttributes.TriangleIndices.Reset();
  Primitive.GetTriangleIndices(OutAttributes.TriangleIndices);
  OutAttributes.UVs.Reset();
  if (Primitive.HasTexCoords(0)) {
    Primitive.GetTexCoords(0, OutAttributes.UVs);
  }
  OutAttributes.Colors.Reset();
  if (Primitive.HasColors()) {
    TArray<FVector4f> Colors;
    Primitive.GetColors(Colors);
    OutAttributes.Colors.Reserve(Colors.Num());
    for (const FVector4f& Color : Colors) {
      OutAttributes.Colors.Add(Color);
    }
  }
  return true;
}

FString CheckRenderModelLink(
    const FRenderModelAttributes& Attributes,
    const FMochiMeshData* VisualMesh) {
  if (!VisualMesh || VisualMesh->Coordinates.IsEmpty()) {
    return TEXT("The Mochi model has no visual mesh.");
  }
  if (!VisualMesh->Skinning.IsSet()) {
    return TEXT("The visual mesh has no deformation embedding.");
  }
  const int32 VisualNodeCount = VisualMesh->Coordinates.Num() / 3;
  if (Attributes.NumVertices != VisualNodeCount) {
    return FString::Printf(
        TEXT("The render model has %d vertices, but the visual mesh has %d."),
        Attributes.NumVertices,
        VisualNodeCount);
  }
  const auto Fits = [&Attributes](int32 Num) { return Num == 0 || Num == Attributes.NumVertices; };
  if (!Fits(Attributes.UVs.Num()) || !Fits(Attributes.Colors.Num())) {
    return TEXT("The render model's UV or color count differs from its vertex count.");
  }
  const TArray<int32>& Connectivity = VisualMesh->Connectivity;
  bool bSameTriangles =
      Attributes.TriangleIndices.Num() == Connectivity.Num() && Connectivity.Num() % 3 == 0;
  bool bReversedTriangles = bSameTriangles;
  // A mirrored bake reverses every triangle, but preserves the vertex-to-attribute mapping.
  for (int32 i = 0; (bSameTriangles || bReversedTriangles) && i < Connectivity.Num(); i += 3) {
    bSameTriangles = bSameTriangles &&
        Attributes.TriangleIndices[i] == static_cast<uint32>(Connectivity[i]) &&
        Attributes.TriangleIndices[i + 1] == static_cast<uint32>(Connectivity[i + 1]) &&
        Attributes.TriangleIndices[i + 2] == static_cast<uint32>(Connectivity[i + 2]);
    bReversedTriangles = bReversedTriangles &&
        Attributes.TriangleIndices[i] == static_cast<uint32>(Connectivity[i + 1]) &&
        Attributes.TriangleIndices[i + 1] == static_cast<uint32>(Connectivity[i]) &&
        Attributes.TriangleIndices[i + 2] == static_cast<uint32>(Connectivity[i + 2]);
  }
  if (!bSameTriangles && !bReversedTriangles) {
    return TEXT("The render model's triangles differ from the visual mesh's.");
  }
  return FString();
}

} // namespace SuperDexDeformableRenderModel
