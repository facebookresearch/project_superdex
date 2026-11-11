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

#include "MeshDescription.h"
#include "MochiArticulation.h"

// A bone of a generated skeleton and the render model bound to it.
struct FMochiArticulationSkeletonLink {
  FName Name = NAME_None;
  int32 ParentIndex = INDEX_NONE;
  // Rest transform of the link relative to its parent (or the articulation root for the root).
  FTransform LinkToParent = FTransform::Identity;
  // Rest transform of the link relative to the articulation root.
  FTransform LinkToRoot = FTransform::Identity;
  UStaticMesh* RenderModel = nullptr;
  FTransform RenderModelToLink = FTransform::Identity;
  // Material slots of the render model that are not merged with others.
  TArray<int32> UniqueMaterialSlots;
};

// A mesh skinned over several bones of a generated skeleton, e.g. a deformable glove over a hand's
// links. It is merged into the generated skeletal mesh alongside the links' rigid render models.
struct FMochiArticulationSkinnedPart {
  // Static mesh attributes plus vertex skin weights, whose bone indices index the skeleton links.
  FMeshDescription Mesh;
  // Rest transform of the mesh relative to the articulation root.
  FTransform MeshToRoot = FTransform::Identity;
  // Material of each polygon group of Mesh, indexed by polygon group ID. A missing or null entry
  // renders with the default material. Not visible to the garbage collector: the caller keeps the
  // materials alive until GenerateSkeletalMesh returns, e.g. as assets of a saved package.
  TArray<UMaterialInterface*> Materials;
};

class SUPERDEXPHYSICSEDITOR_API FMochiArticulationAssetUtils {
 public:
  static FString MakeAssetName(
      const FString& Prefix,
      const FString& ArticulationName,
      const FString& AssetName,
      const FString& Postfix);
  // Get the skeleton links of an articulation in its rest pose.
  static TArray<FMochiArticulationSkeletonLink> GetSkeletonLinks(
      const UMochiArticulation* Articulation);
  static UDataTable* GenerateJointControlGainsTable(
      const TArray<FMochiArticulatedJoint>& Joints,
      const FString& PackagePath,
      const FString& TableAssetName,
      EObjectFlags Flags);
  static UDataTable* GenerateJointControlGainsTable(UMochiArticulation* Articulation);
  static USkeleton* GenerateSkeleton(
      const TArray<FMochiArticulationSkeletonLink>& Links,
      const FString& PackagePath,
      const FString& SkeletonName,
      EObjectFlags Flags);
  // Generate a skeletal mesh binding each link's render model rigidly to its bone, plus any
  // SkinnedParts with their own weights.
  static USkeletalMesh* GenerateSkeletalMesh(
      const TArray<FMochiArticulationSkeletonLink>& Links,
      USkeleton* Skeleton,
      const FString& PackagePath,
      const FString& SkeletalMeshAssetName,
      EObjectFlags Flags,
      const TArray<FMochiArticulationSkinnedPart>& SkinnedParts = {});
  // Generate the skeleton and skeletal mesh of an articulation in the folder of its current
  // skeletal mesh, or next to the articulation if it has none or it is in a legacy "Generated"
  // folder.
  static void GenerateSkeletalMesh(UMochiArticulation* Articulation);
  // Generate the skeleton and skeletal mesh of an articulation in PackagePath. Returns false on
  // failure.
  static bool GenerateSkeletalMesh(UMochiArticulation* Articulation, const FString& PackagePath);
};
