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

#include "SuperDexSoftRenderModel.h"

#include "Algo/AnyOf.h"
#include "GLTFAsset.h"
#include "GLTFReader.h"
#include "MochiModelData.h"

namespace SuperDexSoftRenderModel {

bool ReadRenderModelAttributes(
    const FString& Path,
    FRenderModelAttributes& OutAttributes,
    FString& OutReason) {
  GLTF::FAsset asset;
  GLTF::FFileReader reader;
  reader.ReadFile(Path, /*bInLoadImageData=*/false, /*bInLoadMetadata=*/false, asset);
  for (const GLTF::FLogMessage& message : reader.GetLogMessages()) {
    if (message.Get<0>() == GLTF::EMessageSeverity::Error) {
      OutReason =
          FString::Printf(TEXT("The GLB could not be read: %s"), *message.Get<1>().ToString());
      return false;
    }
  }
  // The reader records required extensions it can't decode, such as EXT_meshopt_compression,
  // without failing; their accessors would then hold no data. Only the extensions GLTFCore lists
  // as supported are accepted, which conservatively excludes some it does decode.
  for (const FString& extension : asset.ExtensionsRequired) {
    if (!Algo::AnyOf(GLTF::FAsset::SupportedExtensions, [&extension](GLTF::EExtension Supported) {
          return extension.Equals(GLTF::ToString(Supported), ESearchCase::CaseSensitive);
        })) {
      OutReason = FString::Printf(
          TEXT("The GLB requires %s, which soft-actor render models don't support in Unreal."),
          *extension);
      return false;
    }
  }
  if (asset.Meshes.Num() != 1 || asset.Meshes[0].Primitives.Num() != 1) {
    OutReason = TEXT("The GLB does not have exactly one mesh primitive.");
    return false;
  }
  GLTF::FPrimitive& primitive = asset.Meshes[0].Primitives[0];
  if (!primitive.IsValid() || !primitive.HasPositions()) {
    OutReason = TEXT("The GLB's primitive geometry is invalid.");
    return false;
  }
  // The imported material may sample TEXCOORD_1, which the soft mesh does not supply.
  if (primitive.HasTexCoords(1)) {
    OutReason = TEXT("The GLB's primitive has a texture-coordinate set other than TEXCOORD_0.");
    return false;
  }

  // Raw accessor data, in file order: GLTF::FMeshFactory would weld and reorder vertices.
  OutAttributes.NumVertices =
      asset.Accessors[primitive.GetAttributeAccessorIndex(GLTF::EMeshAttributeType::POSITION)]
          .Count;
  OutAttributes.TriangleIndices.Reset();
  primitive.GetTriangleIndices(OutAttributes.TriangleIndices);
  OutAttributes.UVs.Reset();
  if (primitive.HasTexCoords(0)) {
    primitive.GetTexCoords(0, OutAttributes.UVs);
  }
  OutAttributes.Colors.Reset();
  if (primitive.HasColors()) {
    TArray<FVector4f> colors;
    primitive.GetColors(colors);
    OutAttributes.Colors.Reserve(colors.Num());
    for (const FVector4f& color : colors) {
      OutAttributes.Colors.Add(color);
    }
  }
  return true;
}

FString CheckSoftRenderModelLink(
    const FRenderModelAttributes& Attributes,
    const FMochiMeshData* VisualMesh) {
  if (!VisualMesh || VisualMesh->Coordinates.IsEmpty()) {
    return TEXT("The Mochi model has no visual mesh.");
  }
  if (!VisualMesh->Skinning.IsSet()) {
    return TEXT("The visual mesh has no deformation embedding.");
  }
  const int32 visualNodeCount = VisualMesh->Coordinates.Num() / 3;
  if (Attributes.NumVertices != visualNodeCount) {
    return FString::Printf(
        TEXT("The render model has %d vertices, but the visual mesh has %d."),
        Attributes.NumVertices,
        visualNodeCount);
  }
  const auto fits = [&Attributes](int32 Num) { return Num == 0 || Num == Attributes.NumVertices; };
  if (!fits(Attributes.UVs.Num()) || !fits(Attributes.Colors.Num())) {
    return TEXT("The render model's UV or color count differs from its vertex count.");
  }
  const TArray<int32>& connectivity = VisualMesh->Connectivity;
  bool bSameTriangles = Attributes.TriangleIndices.Num() == connectivity.Num();
  for (int32 i = 0; bSameTriangles && i < connectivity.Num(); ++i) {
    bSameTriangles = Attributes.TriangleIndices[i] == static_cast<uint32>(connectivity[i]);
  }
  if (!bSameTriangles) {
    return TEXT("The render model's triangles differ from the visual mesh's.");
  }
  return FString();
}

} // namespace SuperDexSoftRenderModel
