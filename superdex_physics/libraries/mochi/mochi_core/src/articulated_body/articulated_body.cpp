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

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/dtransform.h>
#include <mochi_core/utils/lie.h>
#include <mochi_core/utils/rodrigues_utils.h>
#include <mochi_core/utils/sparsity_utils.h>

#include <algorithm>
#include <numeric>
#include <vector>

// TODO[T217402682]
#if MOCHI_COMPILER_GCC
MOCHI_WARNING_IGNORE_GCC_CLANG(GCC diagnostic ignored "-Wstringop-overread")
#endif

namespace mochi::articulated {

int GetReducedPoseSize(Span<ArticulatedPoseInfo const> poseInfo) {
  int size = 0;
  for (auto info : poseInfo) {
    size += info.GetSize();
  }
  return size;
}

int GetReducedDofsSize(Span<ArticulatedDofInfo const> dofInfo) {
  int size = 0;
  for (auto info : dofInfo) {
    size += info.GetSize();
  }
  return size;
}

RowMatrix<real> CreateJacobianStorage(int fullSize, int reducedSize) {
  return RowMatrix<real>::Zero(fullSize, reducedSize);
}

ReducedDofsMap CreateBonesToReducedDofsMap(
    Span<int const> parents,
    Span<ArticulatedDofInfo const> dofInfo) {
  // Traverse hierarchy from start to end. Bones are sorted depth-first from root to children.
  // For each bone, if it has a parent, grab the reduced dofs it depends on, then append the joint
  // dofs for the current bone. If no parent, collect the joint dofs of the current bone.
  ReducedDofsMap out;
  out.dofs.resize(parents.size());
  for (auto boneIdx = 0; boneIdx < parents.size(); ++boneIdx) {
    if (parents[boneIdx] > -1) {
      auto parentIdx = parents[boneIdx];
      out.dofs[boneIdx].insert(
          out.dofs[boneIdx].end(), out.dofs[parentIdx].begin(), out.dofs[parentIdx].end());
    }

    auto& boneDofs = out.dofs[boneIdx];

    int const sizeDofsTrans = dofInfo[boneIdx].transSize;
    if (sizeDofsTrans > 0) {
      int const firstIdx = dofInfo[boneIdx].GetTransOffset();
      for (int j = 0; j < sizeDofsTrans; ++j) {
        boneDofs.push_back(firstIdx + j);
      }
    }

    int const sizeDofsRot = dofInfo[boneIdx].rotSize;
    if (sizeDofsRot > 0) {
      int const firstIdx = dofInfo[boneIdx].GetRotOffset();
      for (int j = 0; j < sizeDofsRot; ++j) {
        boneDofs.push_back(firstIdx + j);
      }
    }
    std::sort(boneDofs.begin(), boneDofs.end());
    auto last = std::unique(boneDofs.begin(), boneDofs.end());
    boneDofs.erase(last, boneDofs.end());
  }
  return out;
}

static void SetJacobianSelfBlock(
    Span<TransformRT const> linkTransforms,
    Span<TransformRT const> jointTransforms,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    int link,
    ArticulatedJointType const& jointType,
    Real3 const& jointAxis,
    ArticulatedDofInfo const& dofInfo,
    int parent,
    RowMatrixView<real> outJacobian) {
  // Skip if the joint has no dofs, i.e., it is a hard joint
  if (jointType == ArticulatedJointType::Hard) {
    return;
  }

  // Common link-joint Jacobian (dTLinkdTJoint = dRLinkdRJoint)
  Quaternion const& qOuter = restTransforms[link].parentFromOuter.GetRotation();
  Quaternion const& qParent = (parent != -1 ? linkTransforms[parent] : worldFromRoot).GetRotation();
  auto dXLinkdXJoint = ToVMatrix3x3(qParent * qOuter);

  // Set translation Jacobians. Joint translation affects only link translation.
  int const linkTransOffset = link * RigidSize::kDAll;
  int const jointTransSize = dofInfo.transSize;
  int const jointTransOffset = dofInfo.offset;
  if (jointTransSize == 1) {
    outJacobian.template Block<3, 1>(linkTransOffset, jointTransOffset, 3, 1) =
        AsColumnVectorView<3>(DotMatVec3x3(dXLinkdXJoint, ToSimd(jointAxis)));
  } else if (jointTransSize == 3) {
    outJacobian.template Block<3, 3>(linkTransOffset, jointTransOffset, 3, 3) =
        AsMatrixView(dXLinkdXJoint);
  } else {
    MOCHI_ASSERT_VERBOSE(jointTransSize == 0, "Unexpected joint translation size");
  }

  int const jointRotSize = dofInfo.rotSize;
  if (jointRotSize <= 0) {
    return;
  }

  // Link-joint Jacobian dTLinkdRJoint
  Quaternion const& qJoint = jointTransforms[link].GetRotation();
  Vec4r tInner = restTransforms[link].innerFromBone.VGetTranslation();
  auto dTLinkdRJoint = Dot3x3(dXLinkdXJoint, lie::DMultRotVecDRot(qJoint * tInner));

  // Set rotation Jacobians. Joint rotation affects both link translation and rotation.
  int const linkRotOffset = link * RigidSize::kDAll + RigidSize::kDTrans;
  int const jointRotOffset = dofInfo.GetRotOffset();
  if (jointRotSize == 1) {
    outJacobian.template Block<3, 1>(linkTransOffset, jointRotOffset, 3, 1) =
        AsColumnVectorView<3>(DotMatVec3x3(dTLinkdRJoint, ToSimd(jointAxis)));
    outJacobian.template Block<3, 1>(linkRotOffset, jointRotOffset, 3, 1) =
        AsColumnVectorView<3>(DotMatVec3x3(dXLinkdXJoint, ToSimd(jointAxis)));
  } else {
    MOCHI_ASSERT_VERBOSE(jointRotSize == 3, "Unexpected joint rotation size");
    outJacobian.template Block<3, 3>(linkTransOffset, jointRotOffset, 3, 3) =
        AsMatrixView(dTLinkdRJoint);
    outJacobian.template Block<3, 3>(linkRotOffset, jointRotOffset, 3, 3) =
        AsMatrixView(dXLinkdXJoint);
  }
}

static void PropagateJacobianBlock(
    VMatrix3x3r const& dTLinkdRParent,
    int link,
    ArticulatedJointType const& jointType,
    ArticulatedDofInfo const& dofInfo,
    int parent,
    RowMatrixView<real> outJacobian) {
  // Skip if the joint has no dofs, i.e., it is a hard joint
  if (jointType == ArticulatedJointType::Hard) {
    return;
  }

  // Set translation Jacobian. Joint translation affects only link translation.
  int const linkTransOffset = link * RigidSize::kDAll;
  int const parentTransOffset = parent * RigidSize::kDAll;
  int const jointTransSize = dofInfo.transSize;
  int const jointTransOffset = dofInfo.GetTransOffset();
  if (jointTransSize == 1) {
    outJacobian.template Block<3, 1>(linkTransOffset, jointTransOffset, 3, 1) =
        outJacobian.template Block<3, 1>(parentTransOffset, jointTransOffset, 3, 1);
  } else if (jointTransSize == 3) {
    outJacobian.template Block<3, 3>(linkTransOffset, jointTransOffset, 3, 3) =
        outJacobian.template Block<3, 3>(parentTransOffset, jointTransOffset, 3, 3);
  } else {
    MOCHI_ASSERT_VERBOSE(jointTransSize == 0, "Unexpected joint translation size");
  }

  int const jointRotSize = dofInfo.rotSize;
  if (jointRotSize <= 0) {
    return;
  }

  // Set rotation Jacobians. Joint rotation affects both link translation and rotation.
  int const linkRotOffset = link * RigidSize::kDAll + RigidSize::kDTrans;
  int const parentRotOffset = parent * RigidSize::kDAll + RigidSize::kDTrans;
  int const jointRotOffset = dofInfo.GetRotOffset();
  if (jointRotSize == 1) {
    auto dTParentdJoint = outJacobian.template Block<3, 1>(parentTransOffset, jointRotOffset, 3, 1);
    auto dRParentdJoint = outJacobian.template Block<3, 1>(parentRotOffset, jointRotOffset, 3, 1);
    outJacobian.template Block<3, 1>(linkTransOffset, jointRotOffset, 3, 1) =
        dTParentdJoint + AsMatrixView(dTLinkdRParent) * dRParentdJoint;
    outJacobian.template Block<3, 1>(linkRotOffset, jointRotOffset, 3, 1) = dRParentdJoint;
  } else {
    MOCHI_ASSERT_VERBOSE(jointRotSize == 3, "Unexpected joint rotation size");
    auto dTParentdJoint = outJacobian.template Block<3, 3>(parentTransOffset, jointRotOffset, 3, 3);
    auto dRParentdJoint = outJacobian.template Block<3, 3>(parentRotOffset, jointRotOffset, 3, 3);
    outJacobian.template Block<3, 3>(linkTransOffset, jointRotOffset, 3, 3) =
        dTParentdJoint + AsMatrixView(dTLinkdRParent) * dRParentdJoint;
    outJacobian.template Block<3, 3>(linkRotOffset, jointRotOffset, 3, 3) = dRParentdJoint;
  }
}

void Jacobian(
    Span<ArticulatedJointType const> jointTypes,
    Span<int const> parents,
    Span<Real3 const> jointAxes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT const> jointTransforms,
    Span<TransformRT const> linkTransforms,
    RowMatrixView<real> outJacobian) {
  // For each link, traverse joints toward the root.
  outJacobian.SetZero();
  for (int link = 0; link < isize(parents); ++link) {
    // Process the link's joint.
    SetJacobianSelfBlock(
        linkTransforms,
        jointTransforms,
        restTransforms,
        worldFromRoot,
        link,
        jointTypes[link],
        jointAxes[link],
        dofInfo[link],
        parents[link],
        outJacobian);

    int parent = parents[link];
    if (parent == -1) {
      continue;
    }

    // Link-parent Jacobian dTLinkdRParent
    Vec4r tLink = linkTransforms[link].VGetTranslation();
    Vec4r tParent = linkTransforms[parent].VGetTranslation();
    VMatrix3x3r dTLinkdRParent = lie::DMultRotVecDRot(tLink - tParent);
    // Propagate other joint Jacobians.
    for (int joint = parent; joint != -1; joint = parents[joint]) {
      PropagateJacobianBlock(
          dTLinkdRParent, link, jointTypes[joint], dofInfo[joint], parent, outJacobian);
    }
  }
}

void TransportInputOfLieJacobian(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    ColumnVectorView<real const> u,
    RowMatrixView<real> outJacobian) {
  MOCHI_ASSERT_VERBOSE(u.Rows() == GetReducedDofsSize(dofInfo), "Invalid u size.");
  MOCHI_ASSERT_VERBOSE(outJacobian.Cols() == GetReducedDofsSize(dofInfo), "Invalid Jacobian size.");
  for (int joint = 0; joint < isize(jointTypes); ++joint) {
    // We only need to transport the rotation Jacobian of free and spherical joints
    if (jointTypes[joint] != ArticulatedJointType::Free &&
        jointTypes[joint] != ArticulatedJointType::Spherical) {
      continue; // Nothing to do
    }

    // Determine the rotation vector and the columns to transport
    int const dofOffset = dofInfo[joint].GetRotOffset();
    TransportInputOfLieJacobian(
        Load<RigidSize::kDRot, Vec4r>(u.data() + dofOffset),
        outJacobian.MiddleCols<RigidSize::kDRot>(dofOffset, RigidSize::kDRot));
  }
}

void TransportOutputOfLieJacobian(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    ColumnVectorView<real const> u,
    RowMatrixView<real> outJacobian) {
  MOCHI_ASSERT_VERBOSE(u.Rows() == GetReducedDofsSize(dofInfo), "Invalid u size.");
  MOCHI_ASSERT_VERBOSE(outJacobian.Rows() == GetReducedDofsSize(dofInfo), "Invalid Jacobian size.");
  for (int joint = 0; joint < isize(jointTypes); ++joint) {
    // We only need to transport the rotation Jacobian of free and spherical joints
    if (jointTypes[joint] != ArticulatedJointType::Free &&
        jointTypes[joint] != ArticulatedJointType::Spherical) {
      continue; // Nothing to do
    }

    // Determine the rotation vector and the rows to transport
    int const dofOffset = dofInfo[joint].GetRotOffset();
    TransportOutputOfLieJacobian(
        Load<RigidSize::kDRot, Vec4r>(u.data() + dofOffset),
        outJacobian.MiddleRows<RigidSize::kDRot>(dofOffset, RigidSize::kDRot));
  }
}

void ChainArticulatedGradientDDeltaDOld(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> qNew,
    ColumnVectorView<real const> qOld,
    ColumnVectorView<real> inOutGrad) {
  MOCHI_ASSERT_VERBOSE(qNew.Rows() == GetReducedPoseSize(poseInfo), "Invalid pose size.");
  MOCHI_ASSERT_VERBOSE(qOld.Rows() == qNew.Rows(), "Invalid pose size.");
  MOCHI_ASSERT_VERBOSE(inOutGrad.Rows() == GetReducedDofsSize(dofInfo), "Invalid gradient size.");
  // Change the sign of the full gradient, which addresses translation and revolute dofs.
  inOutGrad *= -1_r;
  // Shift the rotation gradient of free and spherical joints
  for (int joint = 0; joint < isize(jointTypes); ++joint) {
    if (jointTypes[joint] != ArticulatedJointType::Free &&
        jointTypes[joint] != ArticulatedJointType::Spherical) {
      continue; // Nothing to do
    }

    // Fetch the rotations and the gradient to shift (which is negated)
    Quaternion rotNew(Load<RigidSize::kRot, Vec4r>(&qNew[poseInfo[joint].GetRotOffset()]));
    Quaternion rotOld(Load<RigidSize::kRot, Vec4r>(&qOld[poseInfo[joint].GetRotOffset()]));
    Vec4r grad = -Load<RigidSize::kDRot, Vec4r>(&inOutGrad[dofInfo[joint].GetRotOffset()]);
    ChainRotationGradientDDeltaDOld(rotNew, rotOld, grad);
    Store<RigidSize::kDRot>(&inOutGrad[dofInfo[joint].GetRotOffset()], grad);
  }
}

void ChainArticulatedGradientDNewDOld(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> qNew,
    ColumnVectorView<real const> qOld,
    ColumnVectorView<real> inOutGrad) {
  MOCHI_ASSERT_VERBOSE(qNew.Rows() == GetReducedPoseSize(poseInfo), "Invalid pose size.");
  MOCHI_ASSERT_VERBOSE(qOld.Rows() == qNew.Rows(), "Invalid pose size.");
  MOCHI_ASSERT_VERBOSE(inOutGrad.Rows() == GetReducedDofsSize(dofInfo), "Invalid gradient size.");
  // Nop translation and revolute dofs.
  // Shift the rotation gradient of free and spherical joints
  for (int joint = 0; joint < isize(jointTypes); ++joint) {
    if (jointTypes[joint] != ArticulatedJointType::Free &&
        jointTypes[joint] != ArticulatedJointType::Spherical) {
      continue; // Nothing to do
    }

    // Fetch the rotations and the gradient to shift
    Quaternion rotNew(Load<RigidSize::kRot, Vec4r>(&qNew[poseInfo[joint].GetRotOffset()]));
    Quaternion rotOld(Load<RigidSize::kRot, Vec4r>(&qOld[poseInfo[joint].GetRotOffset()]));
    Vec4r grad = Load<RigidSize::kDRot, Vec4r>(&inOutGrad[dofInfo[joint].GetRotOffset()]);
    ChainRotationGradientDNewDOld(rotNew, rotOld, grad);
    Store<RigidSize::kDRot>(&inOutGrad[dofInfo[joint].GetRotOffset()], grad);
  }
}

void ConvertPoseToDofs(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> pose,
    ColumnVectorView<real> outDofs) {
  MOCHI_ASSERT_VERBOSE(pose.Rows() == GetReducedPoseSize(poseInfo), "Invalid pose size.");
  MOCHI_ASSERT_VERBOSE(outDofs.Rows() == GetReducedDofsSize(dofInfo), "Invalid dofs size.");
  for (int i = 0; i < isize(jointTypes); ++i) {
    switch (jointTypes[i]) {
      case ArticulatedJointType::Free: {
        auto translation = Load<Vec4r>(&pose[poseInfo[i].GetTransOffset()]);
        Store(&outDofs[dofInfo[i].GetTransOffset()], translation);
      }
        [[fallthrough]];
      case ArticulatedJointType::Spherical: {
        auto quaternion =
            Quaternion(Load<RigidSize::kRot, Vec4r>(&pose[poseInfo[i].GetRotOffset()]));
        auto rotVector = quaternion.VToRotationVector();
        Store<RigidSize::kDRot>(&outDofs[dofInfo[i].GetRotOffset()], rotVector);
      } break;
      case ArticulatedJointType::Prismatic:
      case ArticulatedJointType::Revolute: {
        outDofs[dofInfo[i].offset] = pose[poseInfo[i].offset];
      } break;
      default:
        AssertJointTypeCount<6>();
    }
  }
}

void ConvertDofsToPose(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> dofs,
    ColumnVectorView<real> outPose) {
  MOCHI_ASSERT_VERBOSE(dofs.Rows() == GetReducedDofsSize(dofInfo), "Invalid dofs size.");
  MOCHI_ASSERT_VERBOSE(outPose.Rows() == GetReducedPoseSize(poseInfo), "Invalid pose size.");
  for (int i = 0; i < isize(jointTypes); ++i) {
    switch (jointTypes[i]) {
      case ArticulatedJointType::Free: {
        auto translation = Load<Vec4r>(&dofs[dofInfo[i].GetTransOffset()]);
        Store(&outPose[poseInfo[i].GetTransOffset()], translation);
      }
        [[fallthrough]];
      case ArticulatedJointType::Spherical: {
        auto rotVector = Load<RigidSize::kDRot, Vec4r>(&dofs[dofInfo[i].GetRotOffset()]);
        auto quaternion = Quaternion::FromRotationVector(rotVector);
        Store<RigidSize::kRot>(&outPose[poseInfo[i].GetRotOffset()], quaternion.data);
      } break;
      case ArticulatedJointType::Prismatic:
      case ArticulatedJointType::Revolute: {
        outPose[poseInfo[i].offset] = dofs[dofInfo[i].offset];
      } break;
      default:
        AssertJointTypeCount<6>();
    }
  }
}

void ConvertDofFlagsToPoseFlags(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    Span<bool const> dofs,
    Span<bool> outPose) {
  MOCHI_ASSERT_VERBOSE(isize(dofs) == GetReducedDofsSize(dofInfo), "Invalid dofs size.");
  MOCHI_ASSERT_VERBOSE(isize(outPose) == GetReducedPoseSize(poseInfo), "Invalid pose size.");
  for (int i = 0; i < isize(jointTypes); ++i) {
    switch (jointTypes[i]) {
      case ArticulatedJointType::Free: {
        // Copy translation indices
        int const src = dofInfo[i].GetTransOffset();
        int const dst = poseInfo[i].GetTransOffset();
        std::copy(&dofs[src], &dofs[src] + RigidSize::kDTrans, &outPose[dst]);
      }
        [[fallthrough]];
      case ArticulatedJointType::Spherical: {
        // Set rotation indices
        int const src = dofInfo[i].GetRotOffset();
        int const dst = poseInfo[i].GetRotOffset();
        MOCHI_ASSERT_VERBOSE(
            dofs[src] == dofs[src + 1] && dofs[src] == dofs[src + 2],
            "Rotation indices must be all or none");
        std::fill(&outPose[dst], &outPose[dst] + RigidSize::kRot, dofs[src]);
      } break;
      case ArticulatedJointType::Prismatic:
      case ArticulatedJointType::Revolute: {
        outPose[poseInfo[i].offset] = dofs[dofInfo[i].offset];
      } break;
      default:
        AssertJointTypeCount<6>();
    }
  }
}

void ReducedPoseDistance(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> poseA,
    ColumnVectorView<real const> poseB,
    Span<real> outTransDistances,
    Span<real> outRotDistances) {
  MOCHI_ASSERT_VERBOSE(poseA.Rows() == poseB.Rows(), "Inconsistent sizes.");
  MOCHI_ASSERT_VERBOSE(poseA.Rows() == GetReducedPoseSize(poseInfo), "Invalid reduced pose size.");
  MOCHI_ASSERT_VERBOSE(outTransDistances.size() == jointTypes.size(), "Invalid output size.");
  MOCHI_ASSERT_VERBOSE(outRotDistances.size() == jointTypes.size(), "Invalid output size.");

  for (int i = 0; i < isize(jointTypes); ++i) {
    auto const& jp = poseInfo[i];

    outTransDistances[i] = 0_r;
    outRotDistances[i] = 0_r;

    switch (jointTypes[i]) {
      case ArticulatedJointType::Free: {
        auto const startPosition = Load<RigidSize::kTrans, Vec4r>(&poseA[jp.GetTransOffset()]);
        auto const endPosition = Load<RigidSize::kTrans, Vec4r>(&poseB[jp.GetTransOffset()]);
        outTransDistances[i] = Norm<3>(endPosition - startPosition);
        [[fallthrough]];
      }
      case ArticulatedJointType::Spherical: {
        auto const startRotation =
            Quaternion(Load<RigidSize::kRot, Vec4r>(&poseA[jp.GetRotOffset()]));
        auto const endRotation =
            Quaternion(Load<RigidSize::kRot, Vec4r>(&poseB[jp.GetRotOffset()]));
        outRotDistances[i] = (endRotation * startRotation.GetConjugate()).GetAngle();
        break;
      }
      case ArticulatedJointType::Revolute: {
        auto const startAngle = poseA[jp.GetRotOffset()];
        auto const endAngle = poseB[jp.GetRotOffset()];
        outRotDistances[i] = Abs(endAngle - startAngle);
        break;
      }
      case ArticulatedJointType::Prismatic: {
        auto const startDistance = poseA[jp.GetTransOffset()];
        auto const endDistance = poseB[jp.GetTransOffset()];
        outTransDistances[i] = Abs(endDistance - startDistance);
        break;
      }
      case ArticulatedJointType::Hard:
      case ArticulatedJointType::Cycle: {
        // Nothing to do.
        break;
      }
      default:
        AssertJointTypeCount<6>();
    }
  }
}

void AddDeltaToReducedPose(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> pose,
    ColumnVectorView<real const> delta,
    ColumnVectorView<real> outPose) {
  MOCHI_ASSERT_VERBOSE(outPose.Rows() == pose.Rows(), "Inconsistent sizes.");
  MOCHI_ASSERT_VERBOSE(delta.Rows() == pose.Rows(), "Inconsistent sizes.");

  for (int i = 0; i < isize(jointTypes); ++i) {
    auto const& jp = poseInfo[i];
    MOCHI_ASSERT_VERBOSE(jp.offset + jp.GetSize() <= pose.Rows(), "Inconsistent sizes.");
    switch (jointTypes[i]) {
      case ArticulatedJointType::Free: {
        auto const posePos = Load<Vec4r>(&pose[jp.GetTransOffset()]);
        auto const deltaPos = Load<Vec4r>(&delta[jp.GetTransOffset()]);
        // Do not store Vec4r[3], to allow "outPose" to be the same container as one of the inputs.
        Store<RigidSize::kTrans>(&outPose[jp.GetTransOffset()], posePos + deltaPos);
        [[fallthrough]];
      }
      case ArticulatedJointType::Spherical: {
        auto const poseRot = Quaternion(Load<RigidSize::kRot, Vec4r>(&pose[jp.GetRotOffset()]));
        auto const deltaRot = Quaternion(Load<RigidSize::kRot, Vec4r>(&delta[jp.GetRotOffset()]));
        Store<RigidSize::kRot>(&outPose[jp.GetRotOffset()], (deltaRot * poseRot).data);
        break;
      }
      case ArticulatedJointType::Prismatic:
      case ArticulatedJointType::Revolute: {
        outPose(jp.offset) = pose(jp.offset) + delta(jp.offset);
        break;
      }
      default:
        AssertJointTypeCount<6>();
    }
  }
}

void AddLieDeltaToReducedPose(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> pose,
    ColumnVectorView<real const> delta,
    ColumnVectorView<real> outPose) {
  MOCHI_ASSERT_VERBOSE(pose.Rows() == outPose.Rows(), "Inconsistent sizes.");
  MOCHI_ASSERT_VERBOSE(delta.Rows() == GetReducedDofsSize(dofInfo), "Inconsistent sizes.");

  for (int i = 0; i < isize(jointTypes); ++i) {
    auto const& jp = poseInfo[i];
    auto const& jd = dofInfo[i];
    MOCHI_ASSERT_VERBOSE(jp.offset + jp.GetSize() <= pose.Rows(), "Inconsistent sizes.");
    MOCHI_ASSERT_VERBOSE(jd.offset + jd.GetSize() <= delta.Rows(), "Inconsistent sizes.");
    switch (jointTypes[i]) {
      case ArticulatedJointType::Free: {
        auto const pos = Load<Vec4r>(&pose[jp.GetTransOffset()]);
        auto const deltaPos = Load<Vec4r>(&delta[jd.GetTransOffset()]);
        // Do not store Vec4r[3], to allow "pose" and "outPose" to be the same container.
        Store<RigidSize::kTrans>(&outPose[jp.GetTransOffset()], pos + deltaPos);
        [[fallthrough]];
      }
      case ArticulatedJointType::Spherical: {
        auto const rot = Quaternion(Load<RigidSize::kRot, Vec4r>(&pose[jp.GetRotOffset()]));
        auto const deltaRot = Quaternion::FromRotationVector(
            Load<RigidSize::kDRot, Vec4r>(&delta[jd.GetRotOffset()]));
        Store<RigidSize::kRot>(&outPose[jp.GetRotOffset()], (deltaRot * rot).data);
        break;
      }
      case ArticulatedJointType::Prismatic:
      case ArticulatedJointType::Revolute: {
        outPose(jp.offset) = pose(jp.offset) + delta(jd.offset);
        break;
      }
      default:
        AssertJointTypeCount<6>();
    }
  }
}

static void ComputeTranslationDelta(
    int offsetIn,
    int offsetOut,
    ColumnVectorView<real const> start,
    ColumnVectorView<real const> end,
    ColumnVectorView<real> outDelta) {
  // Do not store Vec4r[3], to allow "outDelta" to be the same container as one of the inputs.
  auto xStart = Load<Vec4r>(&start[offsetIn]);
  auto xEnd = Load<Vec4r>(&end[offsetIn]);
  Store<RigidSize::kTrans>(&outDelta[offsetOut], xEnd - xStart);
}

static void ComputeRotationVectorDelta(
    int offsetIn,
    int offsetOut,
    ColumnVectorView<real const> start,
    ColumnVectorView<real const> end,
    ColumnVectorView<real> outDelta) {
  auto qStart = Quaternion(Load<RigidSize::kRot, Vec4r>(&start[offsetIn]));
  auto qEnd = Quaternion(Load<RigidSize::kRot, Vec4r>(&end[offsetIn]));
  Store<RigidSize::kDRot>(&outDelta[offsetOut], (qEnd * qStart.GetConjugate()).VToRotationVector());
}

static void ComputeQuaternionDelta(
    int offsetIn,
    int offsetOut,
    ColumnVectorView<real const> start,
    ColumnVectorView<real const> end,
    ColumnVectorView<real> outDelta) {
  auto qStart = Quaternion(Load<RigidSize::kRot, Vec4r>(&start[offsetIn]));
  auto qEnd = Quaternion(Load<RigidSize::kRot, Vec4r>(&end[offsetIn]));
  Store<RigidSize::kRot>(&outDelta[offsetOut], (qEnd * qStart.GetConjugate()).data);
}

void ComputeDeltaReducedPose(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> start,
    ColumnVectorView<real const> end,
    ColumnVectorView<real> outDelta) {
  MOCHI_ASSERT_VERBOSE(start.Rows() == GetReducedPoseSize(poseInfo), "Inconsistent sizes.");
  MOCHI_ASSERT_VERBOSE(end.Rows() == start.Rows(), "Inconsistent sizes.");
  MOCHI_ASSERT_VERBOSE(outDelta.Rows() == start.Rows(), "Inconsistent sizes.");

  for (int i = 0; i < isize(jointTypes); ++i) {
    auto const& jp = poseInfo[i];
    MOCHI_ASSERT_VERBOSE(jp.offset + jp.GetSize() <= start.Rows(), "Inconsistent sizes.");
    switch (jointTypes[i]) {
      case ArticulatedJointType::Free: {
        ComputeTranslationDelta(jp.GetTransOffset(), jp.GetTransOffset(), start, end, outDelta);
        [[fallthrough]];
      }
      case ArticulatedJointType::Spherical: {
        ComputeQuaternionDelta(jp.GetRotOffset(), jp.GetRotOffset(), start, end, outDelta);
        break;
      }
      case ArticulatedJointType::Prismatic:
      case ArticulatedJointType::Revolute: {
        outDelta(jp.offset) = end(jp.offset) - start(jp.offset);
        break;
      }
      default:
        AssertJointTypeCount<6>();
    }
  }
}

void ComputeLieDeltaReducedPose(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> start,
    ColumnVectorView<real const> end,
    ColumnVectorView<real> outDelta) {
  MOCHI_ASSERT_VERBOSE(start.Rows() == GetReducedPoseSize(poseInfo), "Inconsistent sizes.");
  MOCHI_ASSERT_VERBOSE(end.Rows() == start.Rows(), "Inconsistent sizes.");
  MOCHI_ASSERT_VERBOSE(outDelta.Rows() == GetReducedDofsSize(dofInfo), "Inconsistent sizes.");

  for (int i = 0; i < isize(jointTypes); ++i) {
    auto const& jp = poseInfo[i];
    auto const& jd = dofInfo[i];
    MOCHI_ASSERT_VERBOSE(jp.offset + jp.GetSize() <= start.Rows(), "Inconsistent sizes.");
    MOCHI_ASSERT_VERBOSE(jd.offset + jd.GetSize() <= outDelta.Rows(), "Inconsistent sizes.");
    switch (jointTypes[i]) {
      case ArticulatedJointType::Free: {
        ComputeTranslationDelta(jp.GetTransOffset(), jd.GetTransOffset(), start, end, outDelta);
        [[fallthrough]];
      }
      case ArticulatedJointType::Spherical: {
        ComputeRotationVectorDelta(jp.GetRotOffset(), jd.GetRotOffset(), start, end, outDelta);
        break;
      }
      case ArticulatedJointType::Prismatic:
      case ArticulatedJointType::Revolute: {
        outDelta(jd.offset) = end(jp.offset) - start(jp.offset);
        break;
      }
      default:
        AssertJointTypeCount<6>();
    }
  }
}

void ComputeLieDeltaFullPose(
    ColumnVectorView<real const> start,
    ColumnVectorView<real const> end,
    ColumnVectorView<real> outDelta) {
  MOCHI_ASSERT_VERBOSE(start.Rows() % RigidSize::kAll == 0, "Inconsistent sizes.");
  int numLinks = start.Rows() / RigidSize::kAll;
  MOCHI_ASSERT_VERBOSE(end.Rows() == start.Rows(), "Inconsistent sizes.");
  MOCHI_ASSERT_VERBOSE(outDelta.Rows() / RigidSize::kDAll == numLinks, "Inconsistent sizes.");

  // Use difference for translations, and rotation composition for rotations.
  for (int i = 0, offsetIn = 0, offsetOut = 0; i < numLinks;
       ++i, offsetIn += RigidSize::kAll, offsetOut += RigidSize::kDAll) {
    ComputeTranslationDelta(offsetIn, offsetOut, start, end, outDelta);
    ComputeRotationVectorDelta(
        offsetIn + RigidSize::kTrans, offsetOut + RigidSize::kDTrans, start, end, outDelta);
  }
}

void ComputeTransformsFromReducedPose(
    Span<ArticulatedJointType const> jointTypes,
    Span<Real3 const> jointAxes,
    Span<ArticulatedPoseInfo const> poseInfo,
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    ColumnVectorView<real const> reducedPose,
    Span<TransformRT> outJointTransforms,
    Span<TransformRT> outLinkTransforms) {
  internal::ComputeActiveJointTransforms(
      jointTypes, jointAxes, poseInfo, reducedPose, outJointTransforms);
  // ComputeWorldFromBone runs root-to-leaves, so it's safe to use the same container for
  // ComputeParentFromBone.
  internal::ComputeParentFromBone(restTransforms, outJointTransforms, outLinkTransforms);
  internal::ComputeWorldFromBone(parents, outLinkTransforms, worldFromRoot, outLinkTransforms);
}

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
    ColumnVectorView<real> outFullPose) {
  ComputeTransformsFromReducedPose(
      jointTypes,
      jointAxes,
      poseInfo,
      parents,
      restTransforms,
      worldFromRoot,
      reducedPose,
      outJointTransforms,
      outLinkTransforms);
  internal::ComputeFullPose(outLinkTransforms, outFullPose);
}

void ComputeJointTransformsFromLinkTransforms(
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT const> worldFromBoneTransforms,
    Span<TransformRT> outJointTransforms) {
  // ComputeActiveJointTransforms iterates over joint-link pairs, so it's safe to use the same
  // container for ComputeParentFromBone.
  internal::ComputeParentFromBone(
      parents, worldFromBoneTransforms, worldFromRoot, outJointTransforms);
  internal::ComputeActiveJointTransforms(restTransforms, outJointTransforms, outJointTransforms);
}

void ComputeReducedPoseFromTransforms(
    Span<ArticulatedJointType const> jointTypes,
    Span<Real3 const> jointAxes,
    Span<ArticulatedPoseInfo const> poseInfo,
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT const> worldFromBoneTransforms,
    Span<TransformRT> outJointTransforms,
    ColumnVectorView<real> outReducedPose) {
  ComputeJointTransformsFromLinkTransforms(
      parents, restTransforms, worldFromRoot, worldFromBoneTransforms, outJointTransforms);
  internal::ComputeReducedPose(jointTypes, jointAxes, poseInfo, outJointTransforms, outReducedPose);
}

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
    ColumnVectorView<real> outReducedPose) {
  internal::ComputeWorldFromBone(fullPose, outLinkTransforms);
  ComputeReducedPoseFromTransforms(
      jointTypes,
      jointAxes,
      poseInfo,
      parents,
      restTransforms,
      worldFromRoot,
      outLinkTransforms,
      outJointTransforms,
      outReducedPose);
}

RestTransformArray CreateRestTransforms(
    Span<Real3 const> comLocals,
    Span<int const> jointsChildLinks,
    Span<int const> jointsParentLinks,
    Span<TransformRT const> jointFromChildLink,
    Span<TransformRT const> parentLinkFromJoint) {
  int numJoints = isize(jointFromChildLink);
  RestTransformArray transforms;
  transforms.reserve(numJoints);
  for (int iJoint = 0; iJoint < numJoints; ++iJoint) {
    int const child = jointsChildLinks[iJoint];
    int const parent = jointsParentLinks[iJoint];

    // innerFromBone = jointFromChildLink * rootBoneFromComBone (pure translation)
    TransformRT const& jfc = jointFromChildLink[iJoint];
    TransformRT innerFromBone{jfc.GetRotation(), jfc.TransformPoint(comLocals[child])};

    // parentFromOuter = comParentFromRootParent (pure translation) * parentLinkFromJoint
    TransformRT const& plj = parentLinkFromJoint[iJoint];
    TransformRT parentFromOuter{
        plj.GetRotation(), plj.GetTranslation() - (parent >= 0 ? comLocals[parent] : Real3{})};

    transforms.emplace_back(innerFromBone, parentFromOuter);
  }
  return transforms;
}

TransformRT ComputeJointTransform(
    ColumnVectorView<real const> pose,
    ArticulatedJointType type,
    Real3 const& axis,
    ArticulatedPoseInfo const& poseInfo) {
  int offset = poseInfo.offset;
  MOCHI_ASSERT(poseInfo.GetSize() + offset <= pose.Rows(), "Invalid offset: out-of-bounds");
  switch (type) {
    case ArticulatedJointType::Free: {
      return TransformFromRawPose(pose.Slice<RigidSize::kAll>(offset, RigidSize::kAll));
    }
    case ArticulatedJointType::Revolute: {
      return TransformRT(Quaternion::FromAxisAngle(axis, pose[offset]));
    }
    case ArticulatedJointType::Prismatic: {
      return TransformRT(axis * pose[offset]);
    }
    case ArticulatedJointType::Hard: {
      return {};
    }
    case ArticulatedJointType::Spherical: {
      return TransformRT(Quaternion(Load<RigidSize::kRot, Vec4r>(&pose[offset])));
    }
    case ArticulatedJointType::Cycle: {
      MOCHI_ASSERT(false, "Cannot compute joint transforms for passive joints (cycles).");
      return {};
    }
    default:
      AssertJointTypeCount<6>();
  }
  return {};
}

void ComputeJointPose(
    ArticulatedJointType type,
    Real3 const& axis,
    ArticulatedPoseInfo const& poseInfo,
    TransformRT const& jointTransform,
    ColumnVectorView<real> outReducedPose) {
  int offset = poseInfo.offset;
  if (offset < 0) {
    MOCHI_ASSERT(
        type == ArticulatedJointType::Hard || type == ArticulatedJointType::Cycle,
        "Must be a hard or cycle joint");
    // Nothing to do, as this joint has no DoFs.
    return;
  }
  MOCHI_ASSERT(
      poseInfo.GetSize() + offset <= outReducedPose.Rows(), "Invalid offset: out-of-bounds");
  switch (type) {
    case ArticulatedJointType::Free: {
      TransformToRawPose(
          jointTransform, outReducedPose.Slice<RigidSize::kAll>(offset, RigidSize::kAll));
      return;
    }
    case ArticulatedJointType::Revolute: {
      Vec4r rotvec = jointTransform.GetRotation().VToRotationVector();
      outReducedPose[offset] = Dot(rotvec, ToSimd(axis));
      return;
    }
    case ArticulatedJointType::Prismatic: {
      Vec4r trans = ToSimd(jointTransform.GetTranslation());
      outReducedPose[offset] = Dot(trans, ToSimd(axis));
      return;
    }
    case ArticulatedJointType::Spherical: {
      Store<RigidSize::kRot>(&outReducedPose[offset], jointTransform.GetRotation().data);
      return;
    }
    default:
      AssertJointTypeCount<6>();
  }
}

void NormalizeQuaternions(
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real> outReducedPose) {
  MOCHI_ASSERT_VERBOSE(outReducedPose.Rows() == GetReducedPoseSize(poseInfo), "Invalid size");
  for (int i = 0; i < isize(jointTypes); ++i) {
    if (jointTypes[i] == ArticulatedJointType::Free ||
        jointTypes[i] == ArticulatedJointType::Spherical) {
      int const offset = poseInfo[i].GetRotOffset();
      Quaternion q(Load<RigidSize::kRot, Vec4r>(&outReducedPose[offset]));
      Store<RigidSize::kRot>(&outReducedPose[offset], Normalize(q).data);
    }
  }
}

/************************************************************************************************/
// Utility functions to convert between reduced pose, full pose and transforms
/************************************************************************************************/
namespace internal {

#define MOCHI_ASSERT_VERBOSE_SIZE_EQ(a, b) \
  MOCHI_ASSERT_VERBOSE((a).size() == (b).size(), "Size mismatch")
#define MOCHI_ASSERT_VERBOSE_SIZE_GE(a, b) \
  MOCHI_ASSERT_VERBOSE((a).size() >= (b).size(), "Size mismatch")

void ComputeActiveJointTransforms(
    Span<ArticulatedJointType const> jointTypes,
    Span<Real3 const> jointAxes,
    Span<ArticulatedPoseInfo const> poseInfo,
    ColumnVectorView<real const> reducedPose,
    Span<TransformRT> outJointTransforms) {
  MOCHI_ASSERT_VERBOSE_SIZE_GE(jointTypes, outJointTransforms);
  for (auto i = 0; i < isize(outJointTransforms); ++i) {
    outJointTransforms[i] =
        ComputeJointTransform(reducedPose, jointTypes[i], jointAxes[i], poseInfo[i]);
  }
}

void ComputeReducedPose(
    Span<ArticulatedJointType const> jointTypes,
    Span<Real3 const> jointAxes,
    Span<ArticulatedPoseInfo const> poseInfo,
    Span<TransformRT const> jointTransforms,
    ColumnVectorView<real> outReducedPose) {
  MOCHI_ASSERT_VERBOSE_SIZE_GE(jointTypes, jointTransforms);
  for (size_t i = 0; i < jointTransforms.size(); ++i) {
    ComputeJointPose(jointTypes[i], jointAxes[i], poseInfo[i], jointTransforms[i], outReducedPose);
  }
}

void ComputeParentFromBone(
    Span<ArticulatedRestTransform const> restTransforms,
    Span<TransformRT const> jointTransforms,
    Span<TransformRT> outParentFromBoneTransforms) {
  MOCHI_ASSERT_VERBOSE_SIZE_GE(restTransforms, jointTransforms);
  MOCHI_ASSERT_VERBOSE_SIZE_EQ(outParentFromBoneTransforms, jointTransforms);
  for (auto i = 0; i < jointTransforms.size(); ++i) {
    outParentFromBoneTransforms[i] =
        restTransforms[i].parentFromOuter * jointTransforms[i] * restTransforms[i].innerFromBone;
  }
}

void ComputeActiveJointTransforms(
    Span<ArticulatedRestTransform const> restTransforms,
    Span<TransformRT const> parentFromBoneTransforms,
    Span<TransformRT> outJointTransforms) {
  MOCHI_ASSERT_VERBOSE_SIZE_GE(restTransforms, parentFromBoneTransforms);
  MOCHI_ASSERT_VERBOSE_SIZE_EQ(outJointTransforms, parentFromBoneTransforms);
  for (auto i = 0; i < parentFromBoneTransforms.size(); ++i) {
    outJointTransforms[i] = restTransforms[i].outerFromParent * parentFromBoneTransforms[i] *
        restTransforms[i].boneFromInner;
  }
}

void ComputeWorldFromBone(
    Span<int const> parents,
    Span<TransformRT const> parentFromBoneTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT> outWorldFromBoneTransforms) {
  MOCHI_ASSERT_VERBOSE_SIZE_EQ(parentFromBoneTransforms, parents);
  MOCHI_ASSERT_VERBOSE_SIZE_EQ(outWorldFromBoneTransforms, parents);
  for (auto i = 0; i < parents.size(); ++i) {
    int parentIdx = parents[i];
    if (parentIdx == -1) { // No parent node
      outWorldFromBoneTransforms[i] =
          NormalizeRotation(worldFromRoot * parentFromBoneTransforms[i]);
    } else {
      outWorldFromBoneTransforms[i] =
          NormalizeRotation(outWorldFromBoneTransforms[parentIdx] * parentFromBoneTransforms[i]);
    }
  }
}

void ComputeParentFromBone(
    Span<int const> parents,
    Span<TransformRT const> worldFromBoneTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT> outParentFromBoneTransforms) {
  MOCHI_ASSERT_VERBOSE_SIZE_EQ(worldFromBoneTransforms, parents);
  MOCHI_ASSERT_VERBOSE_SIZE_EQ(outParentFromBoneTransforms, parents);
  auto rootFromWorld = Invert(worldFromRoot);
  for (auto i = 0; i < parents.size(); ++i) {
    int parentIdx = parents[i];
    if (parentIdx == -1) { // No parent node
      outParentFromBoneTransforms[i] = rootFromWorld * worldFromBoneTransforms[i];
    } else {
      outParentFromBoneTransforms[i] =
          Invert(worldFromBoneTransforms[parentIdx]) * worldFromBoneTransforms[i];
    }
  }
}

void ComputeFullPose(
    Span<TransformRT const> worldFromBoneTransforms,
    ColumnVectorView<real> outFullPose) {
  MOCHI_ASSERT_VERBOSE(
      RigidSize::kAll * isize(worldFromBoneTransforms) == outFullPose.Rows(),
      "Invalid size of full pose")
  int offset = 0;
  for (auto const& tx : worldFromBoneTransforms) {
    TransformToRawPose(tx, outFullPose.Slice<RigidSize::kAll>(offset, RigidSize::kAll));
    offset += RigidSize::kAll;
  }
}

void ComputeWorldFromBone(
    ColumnVectorView<real const> fullPose,
    Span<TransformRT> outWorldFromBoneTransforms) {
  MOCHI_ASSERT_VERBOSE(
      RigidSize::kAll * isize(outWorldFromBoneTransforms) == fullPose.Rows(),
      "Invalid size of full pose")
  int offset = 0;
  for (auto& tx : outWorldFromBoneTransforms) {
    tx = TransformFromRawPose(fullPose.Slice<RigidSize::kAll>(offset, RigidSize::kAll));
    offset += RigidSize::kAll;
  }
}

#undef MOCHI_ASSERT_VERBOSE_SIZE_EQ
#undef MOCHI_ASSERT_VERBOSE_SIZE_GE

} // namespace internal
} // namespace mochi::articulated
