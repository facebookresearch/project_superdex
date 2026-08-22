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

#include <mochi_core/articulated_body/articulated_body_params.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/rigid_body_size.h>
#include <mochi_core/utils/rodrigues_utils.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>

#include <cstdint>
#include <utility>
#include <vector>

/************************************************************************************************/
// Definition of lib-level articulated compound data structures and algorithms
//
// Based on the definition in chapter 2 of Physics-Based Animation, by K. Erleben et al.
//
// To describe the transformation to/from a bone and its parent we use two auxiliary frames:
// - An "outer" frame of the joint, i.e., the frame of the joint fixed in the reference system of
// the parent
// - An "inner" frame of the joint, i.e., the frame of the joint fixed in the reference system of
// the bone
//
// The joint transform is the transformation from inner to outer frame, which is defined by 3
// transforms:
// - T_po = parent from outer which remains constant (aka. body from outer)
// - T_oi = outer from inner (the time - varying joint transform)
// - T_ib = inner from bone,  which remains constant (aka. inverse of body from inner)
//
// At runtime we compute parent from bone as:
//
//   T_pb = T_po * T_oi * T_ib.
//
// These transforms are concatenated to find the state of any internal rigid body in world space.
//
// Terminology:
//  - Reduced pose/dofs: joint-space representation of the position (i.e. pose) and its dofs.
//  - Full pose/dofs: link-space representation of the position (i.e. pose) and its dofs.
//  - Joint Info:
//    - Joint type
//      - FREE: 6 dofs {tx, ty, tz, rx, ry, rz}; pose as 3D translation and quaternion.
//      - SPHERICAL: 3 dofs {rx, ry, rz}; pose as quaternion.
//      - PRISMATIC: 1 dof/pose {t}, as the translation along the joint axis.
//      - REVOLUTE: 1 dof/pose {r}, as the rotation around the joint axis.
//    - Axis: axis onto which joint dofs are applied
//  - Link hierarchy: structure defining the hierarchy of links.
/************************************************************************************************/

namespace mochi {

struct ArticulatedProperties {
  uint32_t numLinks = 0;
  // Pose refers to the position state representation, while DoFs refer to the local tangent space
  // of the pose. For 3D translation, 1D translation and 1D rotation they match, but for 3D rotation
  // pose is represented using quaternion and DoFs are represented using Lie algebra (local rotation
  // vector). This distinction applies both to 3D rotations of links and 3D rotations of joints
  // (free and spherical).
  uint32_t fullDofsDim = 0; // Number of DoFs of the link-level pose (3 DoFs per 3D rotation)
  uint32_t fullPoseDim = 0; // Size of the link-level pose (4 values per 3D rotation quaternion)
  uint32_t reducedDofsDim = 0; // Number of DoFs of the joint-level pose (3 DoFs per 3D rotation)
  uint32_t reducedPoseDim = 0; // Size of the joint-level pose (4 values per 3D rotation quaternion)
};

// @brief Pose-space layout for one articulated joint: the offset and translation/rotation sizes
// within the flattened reduced-pose array. Mirrors @ref ArticulatedDofInfo but for pose space,
// where a 3D rotation occupies 4 values (quaternion) rather than 3 DoFs.
struct ArticulatedPoseInfo {
  int offset = -1;
  int transSize = 0;
  int rotSize = 0;

  int GetTransOffset() const {
    return offset;
  }
  int GetRotOffset() const {
    return offset + transSize;
  }
  int GetSize() const {
    return transSize + rotSize;
  }

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ArticulatedPoseInfo const&) const = default;
#endif
};

// @brief Rest-pose transforms between a bone/link's center-of-mass and an articulated joint.
class ArticulatedRestTransform {
 public:
  /** @brief Transform from the child bone (link) frame to the joint's inner frame. */
  TransformRT innerFromBone;
  /** @brief Transform from the joint's outer frame to the parent bone (link) frame. */
  TransformRT parentFromOuter;
  /** @brief Inverse of @ref innerFromBone, computed automatically. */
  TransformRT boneFromInner;
  /** @brief Inverse of @ref parentFromOuter, computed automatically. */
  TransformRT outerFromParent;

  ArticulatedRestTransform() = default;
  ArticulatedRestTransform(TransformRT const& innerFromBone, TransformRT const& parentFromOuter)
      : innerFromBone(innerFromBone),
        parentFromOuter(parentFromOuter),
        boneFromInner(Invert(innerFromBone)),
        outerFromParent(Invert(parentFromOuter)) {}
};
} // namespace mochi

namespace mochi::articulated {

/************************************************************************************************/
// Exposed interface
/************************************************************************************************/

using RestTransformArray = DynamicArray<ArticulatedRestTransform>;

// Index of the parent bone per bone. -1 if the bone has no parent (root).
using ParentIndexArray = DynamicArray<int>;

template <int Value>
void AssertJointTypeCount() {
  static_assert(
      static_cast<int>(ArticulatedJointType::Count) == Value,
      "Please update the caller if you add a new ArticulatedJointType.");
}

// Get the number of DoFs for a specific joint type (first: translation, second: rotation)
[[nodiscard]] inline std::pair<int, int> constexpr GetJointTypeNumDofs(ArticulatedJointType type) {
  switch (type) {
    case ArticulatedJointType::Free:
      return {RigidSize::kDTrans, RigidSize::kDRot};
    case ArticulatedJointType::Spherical:
      return {0, RigidSize::kDRot};
    case ArticulatedJointType::Prismatic:
      return {1, 0};
    case ArticulatedJointType::Revolute:
      return {0, 1};
    case ArticulatedJointType::Hard:
    case ArticulatedJointType::Cycle:
      return {0, 0};
    case ArticulatedJointType::Count: // fallthrough
    default:
      AssertJointTypeCount<6>();
      return {0, 0};
  }
}

// Get the pose size for a specific joint type (first: translation, second: rotation)
[[nodiscard]] inline std::pair<int, int> constexpr GetJointTypeNumPose(ArticulatedJointType type) {
  switch (type) {
    case ArticulatedJointType::Free:
      return {RigidSize::kTrans, RigidSize::kRot};
    case ArticulatedJointType::Spherical:
      return {0, RigidSize::kRot};
    case ArticulatedJointType::Prismatic:
      return {1, 0};
    case ArticulatedJointType::Revolute:
      return {0, 1};
    case ArticulatedJointType::Hard:
    case ArticulatedJointType::Cycle:
      return {0, 0};
    case ArticulatedJointType::Count: // fallthrough
    default:
      AssertJointTypeCount<6>();
      return {0, 0};
  }
}

// Set up per-joint dofs based on joint types
[[nodiscard]] inline DynamicArray<ArticulatedDofInfo> SetupJointDofs(
    Span<ArticulatedJointType const> types) {
  // Loop over joints and set joint-specific sizes and offsets.
  DynamicArray<ArticulatedDofInfo> result;
  result.reserve(types.size());
  int offset = 0;
  for (auto const& type : types) {
    auto const size = GetJointTypeNumDofs(type);
    result.push_back(ArticulatedDofInfo{offset, size.first, size.second});
    offset += result.back().GetSize();
  }
  return result;
}

// Set up per-joint pose based on joint types
[[nodiscard]] inline DynamicArray<ArticulatedPoseInfo> SetupJointPose(
    Span<ArticulatedJointType const> types) {
  // Loop over joints and set joint-specific sizes and offsets.
  DynamicArray<ArticulatedPoseInfo> result;
  result.reserve(types.size());
  int offset = 0;
  for (auto const& type : types) {
    auto const size = GetJointTypeNumPose(type);
    result.push_back(ArticulatedPoseInfo{offset, size.first, size.second});
    offset += result.back().GetSize();
  }
  return result;
}

// Get the size of the reduced pose from the joint pose info
[[nodiscard]] int GetReducedPoseSize(Span<ArticulatedPoseInfo const> poseInfo);

// Get the number of reduced dofs from the joint dofs info
[[nodiscard]] int GetReducedDofsSize(Span<ArticulatedDofInfo const> dofInfo);

// Create 2D storage for Jacobian of mapping from reduced dofs to full dofs
// Returned storage is initialized to zeros
RowMatrix<real> CreateJacobianStorage(int fullSize, int reducedSize);

// Helper structure that stores, for each bone, the indices of its governing joint DoFs
struct ReducedDofsMap {
  std::vector<std::vector<int>> dofs;
};

// Create the sparse map from reduced dofs to full dofs based on the hierarchy and the joint info.
ReducedDofsMap CreateBonesToReducedDofsMap(
    Span<int const> parents,
    Span<ArticulatedDofInfo const> dofInfo);

// Compute Jacobian
void Jacobian(
    Span<ArticulatedJointType const> jointTypes,
    Span<int const> parents,
    Span<Real3 const> jointAxes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT const> jointTransforms,
    Span<TransformRT const> linkTransforms,
    RowMatrixView<real> outJacobian);

// Given a Lie Jacobian dx/dq, with q an articulated pose, this function computes the Jacobian
// dx/du, with u expressing 3D rotations as full rotation vectors.
void TransportInputOfLieJacobian(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    ColumnVectorView<real const> u,
    RowMatrixView<real> outJacobian);

// Given a Lie Jacobian dq/dx, with q an articulated pose, this function computes the Jacobian
// du/dx, with u expressing 3D rotations as full rotation vectors.
void TransportOutputOfLieJacobian(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    ColumnVectorView<real const> u,
    RowMatrixView<real> outJacobian);

// Given two articulated poses qNew and qOld, with relative pose qDelta, this operation
// implements the chain rule df/dqOld = df/dqDelta * dqDelta/dqOld.
void ChainArticulatedGradientDDeltaDOld(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> qNew,
    ColumnVectorView<real const> qOld,
    ColumnVectorView<real> inOutGrad);

// Given two articulated poses qNew and qOld, with relative pose qDelta, this operation
// implements the chain rule df/dqOld = df/dqNew * dqNew/dqOld.
void ChainArticulatedGradientDNewDOld(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> qNew,
    ColumnVectorView<real const> qOld,
    ColumnVectorView<real> inOutGrad);

// Convert a pose (using quaternion) to dofs (using rotation vector)
void ConvertPoseToDofs(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> pose,
    ColumnVectorView<real> outDofs);

// Convert dofs (using rotation vector) to pose (using quaternion)
void ConvertDofsToPose(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> dofs,
    ColumnVectorView<real> outPose);

// Convert flags on dofs to flags on pose
void ConvertDofFlagsToPoseFlags(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    Span<bool const> dofs,
    Span<bool> outPose);

// Computes the distances between two reduced poses. The distance of 3D rotation joints is
// evaluated as the angle of the relative rotation. Start and end pose vectors must be of
// the same size as reported by GetReducedPoseSize. Output distances must be a span of the same
// size as the number of joints. Note that some joints may not have translational or rotational
// DoFs, in which case the distances are set to zero.
void ReducedPoseDistance(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> poseA,
    ColumnVectorView<real const> poseB,
    Span<real> outTransDistances,
    Span<real> outRotDistances);

// Add a pose increment to a reduced pose.
void AddDeltaToReducedPose(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> pose,
    ColumnVectorView<real const> delta,
    ColumnVectorView<real> outPose);

// Add a Lie-algebra pose increment (i.e., using incremental rotation vectors for 3D rotations).
void AddLieDeltaToReducedPose(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> pose,
    ColumnVectorView<real const> delta,
    ColumnVectorView<real> outPose);

// Compute the difference between two reduced poses.
void ComputeDeltaReducedPose(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> start,
    ColumnVectorView<real const> end,
    ColumnVectorView<real> outDelta);

// Compute the Lie-algebra difference between two reduced poses (i.e., using incremental rotation
// vectors for 3D rotations).
void ComputeLieDeltaReducedPose(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> start,
    ColumnVectorView<real const> end,
    ColumnVectorView<real> outDelta);

// Compute the Lie-algebra difference between two full poses (i.e., using incremental rotation
// vectors for 3D rotations).
void ComputeLieDeltaFullPose(
    ColumnVectorView<real const> start,
    ColumnVectorView<real const> end,
    ColumnVectorView<real> outDelta);

// Compute joint and link transforms from reduced dofs.
void ComputeTransformsFromReducedPose(
    Span<ArticulatedJointType const> jointTypes,
    Span<Real3 const> jointAxes,
    Span<ArticulatedPoseInfo const> poseInfo,
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    ColumnVectorView<real const> reducedPose,
    Span<TransformRT> outJointTransforms,
    Span<TransformRT> outLinkTransforms);

// Compute full pose from reduced pose. The function also outputs the joint and link transforms,
// which are necessary intermediate steps.
void ComputeFullPose(
    Span<ArticulatedJointType const> jointTypes,
    Span<Real3 const> jointAxes,
    Span<ArticulatedPoseInfo const> poseInfo,
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    ColumnVectorView<real const> reducedPose,
    Span<TransformRT> outJointTransforms,
    Span<TransformRT> outLinkTransforms,
    ColumnVectorView<real> outFullPose);

// Compute reduced pose from full pose
void ComputeReducedPose(
    Span<ArticulatedJointType const> jointTypes,
    Span<Real3 const> jointAxes,
    Span<ArticulatedPoseInfo const> poseInfo,
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    ColumnVectorView<real const> fullPose,
    Span<TransformRT> outJointTransforms,
    Span<TransformRT> outLinkTransforms,
    ColumnVectorView<real> outReducedPose);

// Compute joint transforms from link transforms
void ComputeJointTransformsFromLinkTransforms(
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT const> worldFromBoneTransforms,
    Span<TransformRT> outJointTransforms);

// Compute reduced pose from link transforms
void ComputeReducedPoseFromTransforms(
    Span<ArticulatedJointType const> jointTypes,
    Span<Real3 const> jointAxes,
    Span<ArticulatedPoseInfo const> poseInfo,
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT const> worldFromBoneTransforms,
    Span<TransformRT> outJointTransforms,
    ColumnVectorView<real> outReducedPose);

// Create rest transforms of joints wrt bones. It takes as input joint-link transforms expressed in
// link root, and it outputs joint-link transforms expressed in link CoM.
RestTransformArray CreateRestTransforms(
    Span<Real3 const> comLocals,
    Span<int const> jointsChildLinks,
    Span<int const> jointsParentLinks,
    Span<TransformRT const> jointFromChildLink,
    Span<TransformRT const> parentLinkFromJoint);

// Compute joint transform based on the full joint pose of the articulated body. Requires joint info
// to interpret the pose.This function can only be called on active joints, not on passive joints
// (cycles).
TransformRT ComputeJointTransform(
    ColumnVectorView<real const> pose,
    ArticulatedJointType type,
    Real3 const& axis,
    ArticulatedPoseInfo const& poseInfo);

// Compute reduced joint pose based on joint transform. Requires joint info to produce the expected
// output. The joint information is used to write onto the appropriate indices in the output pose
// vector.
void ComputeJointPose(
    ArticulatedJointType type,
    Real3 const& axis,
    ArticulatedPoseInfo const& poseInfo,
    TransformRT const& jointTransform,
    ColumnVectorView<real> outReducedPose);

// Given a reduced pose, normalize all quaternion representations of 3D rotations.
void NormalizeQuaternions(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real> outReducedPose);

/************************************************************************************************/
// Internal utility functions to convert between reduced dofs, full pose and transforms
/************************************************************************************************/
namespace internal {
// Compute array of active joint transforms based on reduced pose
// Requires array of joint info to interpret the pose
void ComputeActiveJointTransforms(
    Span<ArticulatedJointType const> jointTypes,
    Span<Real3 const> jointAxes,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> reducedPose,
    Span<TransformRT> outJointTransforms);

// Compute reduced pose based on array of joint transforms
// Requires array of joint info to produce the expected reduced pose
void ComputeReducedPose(
    Span<ArticulatedJointType const> jointTypes,
    Span<Real3 const> jointAxes,
    Span<ArticulatedPoseInfo const> poseInfo,
    Span<TransformRT const> jointTransforms,
    ColumnVectorView<real> outReducedPose);

// Compute array of parent from bone transforms based on array of joint transforms
void ComputeParentFromBone(
    Span<ArticulatedRestTransform const> restTransforms,
    Span<TransformRT const> jointTransforms,
    Span<TransformRT> outParentFromBoneTransforms);

// Compute array of active joint transforms based on array of parent from bone transforms
void ComputeActiveJointTransforms(
    Span<ArticulatedRestTransform const> restTransforms,
    Span<TransformRT const> parentFromBoneTransforms,
    Span<TransformRT> outJointTransforms);

// Compute array of world from bone transforms based on array of parent from bone transforms
void ComputeWorldFromBone(
    Span<int const> parents,
    Span<TransformRT const> parentFromBoneTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT> outWorldFromBoneTransforms);

// Compute array of parent from bone transforms based on array of world from bone transforms
void ComputeParentFromBone(
    Span<int const> parents,
    Span<TransformRT const> worldFromBoneTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT> outParentFromBoneTransforms);

// Compute full pose based on array of world from bone transforms
void ComputeFullPose(
    Span<TransformRT const> worldFromBoneTransforms,
    ColumnVectorView<real> outFullPose);

// Compute array of world from bone transforms based on full pose
void ComputeWorldFromBone(
    ColumnVectorView<real const> fullPose,
    Span<TransformRT> outWorldFromBoneTransforms);

} // namespace internal
} // namespace mochi::articulated
