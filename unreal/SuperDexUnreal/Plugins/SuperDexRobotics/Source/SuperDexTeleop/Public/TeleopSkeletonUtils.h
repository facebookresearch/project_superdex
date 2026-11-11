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
#include "Kismet/BlueprintFunctionLibrary.h"

#include "TeleopSkeletonUtils.generated.h"

class USkeletalMesh;
class USkeletalMeshComponent;

/**
 * Blueprint function library with utilities for inspecting and exporting the
 * coordinate frames of a skeletal mesh's reference skeleton.
 *
 * The primary use case is offline analysis: dump the bind-pose bone transforms
 * of a known-good skeleton and of a generated (e.g. SuperDex bot) skeleton so the
 * per-bone relative transform between them can be computed and folded back into
 * the asset-generation tooling.
 */
UCLASS()
class SUPERDEXTELEOP_API UTeleopSkeletonUtils : public UBlueprintFunctionLibrary {
  GENERATED_BODY()

 public:
  /**
   * Extracts the reference-skeleton bone hierarchy and bind-pose transforms of
   * a skeletal mesh and writes them to a JSON file.
   *
   * For every bone the file records, in Unreal coordinates (left-handed,
   * X=forward, Y=right, Z=up, centimeters):
   *   - the bone name, index, and parent index/name
   *   - the local (parent-relative) bind-pose transform
   *   - the component-space (root-relative) bind-pose transform
   * Rotations are stored as quaternions (XYZW) and, for readability, as
   * XYZ-degree Euler angles. Translation and scale are stored as XYZ triples.
   *
   * @param SkeletalMesh The mesh whose reference skeleton is exported.
   * @param FilePath Destination file. Absolute paths are used verbatim; a
   * relative path is resolved under the project's Saved/ directory. If empty, a
   * default of Saved/SkeletonExports/<MeshName>.json is used.
   * @return True if the file was written successfully.
   */
  UFUNCTION(BlueprintCallable, Category = "Teleop|Skeleton")
  static bool ExportSkeletalMeshSkeleton(
      USkeletalMesh const* SkeletalMesh,
      FString const& FilePath);

  /**
   * Convenience overload that exports the reference skeleton of the skeletal
   * mesh currently assigned to a component. See @ref ExportSkeletalMeshSkeleton.
   *
   * @param Component The component whose skeletal mesh asset is exported.
   * @param FilePath Destination file (see @ref ExportSkeletalMeshSkeleton).
   * @return True if the file was written successfully.
   */
  UFUNCTION(BlueprintCallable, Category = "Teleop|Skeleton")
  static bool ExportSkeletalMeshComponentSkeleton(
      USkeletalMeshComponent const* Component,
      FString const& FilePath);

  /**
   * Removes the rotation from the reference-pose transform of the root bone (index 0) of a skeletal
   * mesh while keeping the asset internally consistent and the skinned result unchanged.
   *
   * Purpose: some hand skeletons ship with a spurious rotation (e.g. 90 degrees) baked into the
   * root bone. USuperDexPoseableComponent masks it at runtime (it overwrites every bone's rotation
   * with the tracked pose) and in the editor (WhenNotPlayingInitPoseCache forces the root bone to
   * identity), so it is invisible there but appears in every other use of the mesh.
   *
   * The operation: clear bone 0's rest rotation R to identity (preserving translation/scale), and
   * counter-rotate the mesh geometry (positions, normals, tangents) by R^-1. Clearing bone 0
   * rotates the whole skeleton's component frames by R^-1, which pushes R into every bone's
   * inverse-bind matrix; the geometry counter-rotation cancels it for every vertex. Because the
   * root bone's rotation is overwritten by the tracked pose at runtime, this is transparent to
   * USuperDexPoseableComponent (and the mochi pose controller, which reads only rest *locations*,
   * which are unchanged). Child bone local transforms are left untouched. The visible effect is
   * that the raw mesh (and any non-SuperDexPose use) now displays in the corrected orientation that
   * SuperDexPoseable already showed.
   *
   * Both the runtime RefSkeleton and each LOD's source MeshDescription (bone poses + geometry) are
   * updated, the inverse bind matrices are recomputed, and the asset is marked dirty. Editing only
   * RefSkeleton is not enough: a render-data rebuild on reload re-derives from the source
   * MeshDescription and reverts the change.
   *
   * Editor-only. Assumes the root bone's rest translation is zero. After running this and
   * verifying, the root-forcing line in USuperDexPoseableComponent::WhenNotPlayingInitPoseCache can
   * be removed. Note: morph-target position deltas, if any, are not counter-rotated (hand meshes
   * have none).
   *
   * @param SkeletalMesh The mesh to rebase. Modified in place.
   * @param bAlsoUpdateSharedSkeleton If true, also clears the root rotation in the shared
   * USkeleton's reference pose so the two stay consistent. Affects every mesh sharing that skeleton
   * (set true on exactly one mesh per shared skeleton).
   * @return True if the mesh was modified successfully.
   */
  UFUNCTION(BlueprintCallable, Category = "Teleop|Skeleton")
  static bool ClearReferenceRootBoneRotation(
      USkeletalMesh* SkeletalMesh,
      bool bAlsoUpdateSharedSkeleton = true);
};
