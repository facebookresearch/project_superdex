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

struct FMochiMeshData;

// A SuperDex Studio soft actor's GLB render model, as Unreal draws it: the actor's visual mesh,
// with the render model's material and per-vertex attributes. Unlike Studio, Unreal never draws
// the render model's own positions, so they and its placement in the prefab are not checked.
namespace SuperDexSoftRenderModel {

// The per-vertex attributes of a GLB's one mesh primitive, in accessor order.
struct FRenderModelAttributes {
  int32 NumVertices = 0;
  TArray<uint32> TriangleIndices;
  // TEXCOORD_0, or empty if the primitive has none.
  TArray<FVector2f> UVs;
  // COLOR_0 as linear RGBA, or empty if the primitive has none.
  TArray<FLinearColor> Colors;
};

// Reads the GLB at Path, which must have exactly one mesh primitive, with no texture-coordinate set
// other than TEXCOORD_0, and require no glTF extension beyond those GLTFCore lists as supported,
// which excludes mesh compression. Otherwise returns false with the reason.
bool ReadRenderModelAttributes(
    const FString& Path,
    FRenderModelAttributes& OutAttributes,
    FString& OutReason);

// Why Attributes can't be drawn on VisualMesh (null if the model has none), or empty if they can:
// the visual mesh must have a deformation embedding, and the render model must have its vertex
// count and triangle indices, so that render-model vertex i is visual-mesh node i. UVs and colors
// must each be empty or have one entry per vertex.
FString CheckSoftRenderModelLink(
    const FRenderModelAttributes& Attributes,
    const FMochiMeshData* VisualMesh);

} // namespace SuperDexSoftRenderModel
