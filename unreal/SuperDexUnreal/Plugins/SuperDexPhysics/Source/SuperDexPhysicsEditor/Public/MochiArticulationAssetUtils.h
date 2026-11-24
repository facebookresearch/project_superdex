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

/** A bone of a generated skeleton and the render model bound to it. */
struct FMochiArticulationSkeletonLink {
  /** Name of the skeleton link. */
  FName Name = NAME_None;
  /** Index of the parent skeleton link. */
  int32 ParentIndex = INDEX_NONE;
  /** Rest transform of the link relative to its parent (or the articulation root for the root). */
  FTransform LinkToParent = FTransform::Identity;
  /** Rest transform of the link relative to the articulation root. */
  FTransform LinkToRoot = FTransform::Identity;
  /** Render geometry associated with the skeleton link. */
  UStaticMesh* RenderModel = nullptr;
  /** Transform from render-model coordinates to the link frame. */
  FTransform RenderModelToLink = FTransform::Identity;
  /** Material slots of the render model that are not merged with others. */
  TArray<int32> UniqueMaterialSlots;
};

/**
 * A mesh skinned over several bones of a generated skeleton, e.g. a deformable glove over a hand's
 * links. It is merged into the generated skeletal mesh alongside the links' rigid render models.
 */
struct FMochiArticulationSkinnedPart {
  /**
   * Static mesh attributes plus vertex skin weights, whose bone indices index the skeleton links.
   */
  FMeshDescription Mesh;
  /** Rest transform of the mesh relative to the articulation root. */
  FTransform MeshToRoot = FTransform::Identity;
  /**
   * Material of each polygon group of Mesh, indexed by polygon group ID. A missing or null entry
   * renders with the default material. Not visible to the garbage collector: the caller keeps the
   * materials alive until GenerateSkeletalMesh returns, e.g. as assets of a saved package.
   */
  TArray<UMaterialInterface*> Materials;
};

/** Editor helpers generating articulation control tables, skeletons and skeletal meshes. */
class SUPERDEXPHYSICSEDITOR_API FMochiArticulationAssetUtils {
 public:
  /**
   * Compose an articulation asset name from its name parts.
   *
   * @param Prefix Prefix of the generated asset name.
   * @param ArticulationName Name of the articulation.
   * @param AssetName Name of the asset to create.
   * @param Postfix Suffix of the generated asset name.
   * @return The composed asset name.
   */
  static FString MakeAssetName(
      const FString& Prefix,
      const FString& ArticulationName,
      const FString& AssetName,
      const FString& Postfix);
  /**
   * Get the skeleton links of an articulation in its rest pose.
   *
   * @param Articulation Articulation asset supplied to the operation.
   * @return The skeleton links.
   */
  static TArray<FMochiArticulationSkeletonLink> GetSkeletonLinks(
      const UMochiArticulation* Articulation);
  /**
   * Generate a joint-control gains table for the supplied joints.
   *
   * @param Joints Joint definitions used to populate controller gains.
   * @param PackagePath Package path for the generated asset or output.
   * @param TableAssetName Name of the generated gains table.
   * @param Flags Object flags used for generated assets.
   * @return The result of asset generation.
   */
  static UDataTable* GenerateJointControlGainsTable(
      const TArray<FMochiArticulatedJoint>& Joints,
      const FString& PackagePath,
      const FString& TableAssetName,
      EObjectFlags Flags);
  /**
   * Generate a joint-control gains table for the articulation asset.
   *
   * @param Articulation Articulation asset supplied to the operation.
   * @return The result of asset generation.
   */
  static UDataTable* GenerateJointControlGainsTable(UMochiArticulation* Articulation);
  /**
   * Generate a skeleton for the supplied articulation links.
   *
   * @param Links Link definitions used for skeleton and mesh generation.
   * @param PackagePath Package path for the generated asset or output.
   * @param SkeletonName Name of the generated skeleton.
   * @param Flags Object flags used for generated assets.
   * @return The result of asset generation.
   */
  static USkeleton* GenerateSkeleton(
      const TArray<FMochiArticulationSkeletonLink>& Links,
      const FString& PackagePath,
      const FString& SkeletonName,
      EObjectFlags Flags);
  /**
   * Generate a skeletal mesh binding each link's render model rigidly to its bone, plus any
   * SkinnedParts with their own weights.
   *
   * @param Links Link definitions used for skeleton and mesh generation.
   * @param Skeleton Skeleton used by the generated skeletal mesh.
   * @param PackagePath Package path for the generated asset or output.
   * @param SkeletalMeshAssetName Name of the generated skeletal mesh.
   * @param Flags Object flags used for generated assets.
   * @param SkinnedParts Geometry and skinning data combined in the skeletal mesh.
   * @return The result of asset generation.
   */
  static USkeletalMesh* GenerateSkeletalMesh(
      const TArray<FMochiArticulationSkeletonLink>& Links,
      USkeleton* Skeleton,
      const FString& PackagePath,
      const FString& SkeletalMeshAssetName,
      EObjectFlags Flags,
      const TArray<FMochiArticulationSkinnedPart>& SkinnedParts = {});
  /**
   * Generate the skeleton and skeletal mesh of an articulation in the folder of its current
   * skeletal mesh, or next to the articulation if it has none or it is in a legacy "Generated"
   * folder.
   *
   * @param Articulation Articulation asset supplied to the operation.
   */
  static void GenerateSkeletalMesh(UMochiArticulation* Articulation);
  /**
   * Generate the skeleton and skeletal mesh of an articulation in PackagePath. Returns false on
   * failure.
   *
   * @param Articulation Articulation asset supplied to the operation.
   * @param PackagePath Package path for the generated asset or output.
   * @return The result of asset generation.
   */
  static bool GenerateSkeletalMesh(UMochiArticulation* Articulation, const FString& PackagePath);
};
