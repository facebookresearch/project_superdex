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

#include <mochi_core/articulated_body/articulated_body_hessian.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/dtransform.h>
#include <mochi_core/utils/lie.h>

namespace mochi::articulated {

template <bool kTrans, bool kRot>
inline void ForEachDof(ArticulatedDofInfo const& dofInfo, std::function<void(int)> callback) {
  if constexpr (kTrans) {
    for (int i = 0; i < dofInfo.transSize; i++) {
      callback(dofInfo.GetTransOffset() + i);
    }
  }
  if constexpr (kRot) {
    for (int i = 0; i < dofInfo.rotSize; i++) {
      callback(dofInfo.GetRotOffset() + i);
    }
  }
}

template <bool kTrans, bool kRot>
inline void ForEachDof(ArticulatedDofInfo const& dofInfo, std::function<void(int, int)> callback) {
  if constexpr (kTrans) {
    for (int i = 0; i < dofInfo.transSize; i++) {
      callback(dofInfo.GetTransOffset() + i, i);
    }
  }
  if constexpr (kRot) {
    for (int i = 0; i < dofInfo.rotSize; i++) {
      callback(dofInfo.GetRotOffset() + i, i);
    }
  }
}

template <bool kTrans, bool kRot>
inline void ForEachDof(
    Span<ArticulatedDofInfo const> dofInfo,
    Span<int const> parents,
    int link,
    std::function<void(int)> callback) {
  for (; link >= 0; link = parents[link]) {
    ForEachDof<kTrans, kRot>(dofInfo[link], callback);
  }
}

static void HessianSelfBlock(
    Span<ArticulatedDofInfo const> dofInfo,
    Span<Real3 const> jointAxes,
    Span<int const> parents,
    int link,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT const> jointTransforms,
    Span<TransformRT const> linkTransforms,
    ArticulatedHessian& outHessian) {
  // Self-block: rotational part is always zero, and we only handle translational part
  int parent = parents[link];

  // Common link-joint Jacobian (dTLinkdTJoint = dRLinkdRJoint)
  Quaternion const& qOuter = restTransforms[link].parentFromOuter.GetRotation();
  Quaternion const& qParent = (parent != -1 ? linkTransforms[parent] : worldFromRoot).GetRotation();
  auto dXLinkdXJointT = ToVMatrix3x3Transpose(qParent * qOuter);

  int off = dofInfo[link].GetRotOffset();
  auto tInner =
      jointTransforms[link].GetRotation() * restTransforms[link].innerFromBone.VGetTranslation();
  if (dofInfo[link].rotSize == 1) {
    auto axisV = ToSimd(jointAxes[link]);
    // outHessian(link * RigidSize::kDAll + d, off, off)
    auto tmp = DotVecMat3x3(
        lie::MultVecbTD2MultRotVecaDRot2MultVecc(tInner, axisV, axisV), dXLinkdXJointT);
    Store<RigidSize::kDTrans>(&outHessian[off](link * RigidSize::kDAll, off), tmp);
  } else if (dofInfo[link].rotSize == 3) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j <= i; j++) {
        // Two symmetric blocks:
        // outHessian(link * RigidSize::kDAll + d, off + i, off + j)
        // outHessian(link * RigidSize::kDAll + d, off + j, off + i)
        auto tmp =
            DotVecMat3x3(lie::MultAxisaTD2MultRotVecDRot2MultAxisb(tInner, i, j), dXLinkdXJointT);
        Store<RigidSize::kDTrans>(&outHessian[off + j](link * RigidSize::kDAll, off + i), tmp);
        Store<RigidSize::kDTrans>(&outHessian[off + i](link * RigidSize::kDAll, off + j), tmp);
      }
    }
  }
}

// Compute the entire tensor of Hessian, see:
// https://overleaf.thefacebook.com/project/68c09b3562fc0ce28fa53543
void Hessian(
    Span<ArticulatedDofInfo const> dofInfo,
    Span<Real3 const> jointAxes,
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT const> jointTransforms,
    Span<TransformRT const> linkTransforms,
    RowMatrixView<real const> jacobian,
    ArticulatedHessian& outHessian) {
  for (auto& outHessianComp : outHessian) {
    outHessianComp.SetZero();
  }

  // For each link
  for (int link = 0; link < isize(parents); ++link) {
    int parent = parents[link];
    auto const& jointDofsLink = dofInfo[link];

    Vec4r dWdjV[3];
    Vec4r dtdjV[3];

    // Self-block: rotational part is always zero, and we only handle translational part
    HessianSelfBlock(
        dofInfo,
        jointAxes,
        parents,
        link,
        restTransforms,
        worldFromRoot,
        jointTransforms,
        linkTransforms,
        outHessian);

    // Recursive-block: t_{k-1}-part just copy. Write full Vec4r because rotation will overwrite.
    ForEachDof<true, true>(dofInfo, parents, parent, [&](int i) {
      ForEachDof<true, true>(dofInfo, parents, parent, [&](int j) {
        // outHessian(link * RigidSize::kDAll + d, i, j) =
        // outHessian(parent * RigidSize::kDAll + d, i, j)
        auto* linkPos = &outHessian[j](link * RigidSize::kDAll, i);
        auto const* parentPos = &outHessian[j](parent * RigidSize::kDAll, i);
        Store(linkPos, Load<Vec4r>(parentPos));
      });
    });

    // Recursive-block: omega_{k-1}-part just copy
    ForEachDof<false, true>(dofInfo, parents, parent, [&](int i) {
      ForEachDof<false, true>(dofInfo, parents, parent, [&](int j) {
        // outHessian(link * RigidSize::kDAll + RigidSize::kDTrans + d, i, j) =
        // outHessian(parent * RigidSize::kDAll + RigidSize::kDTrans + d, i, j)
        auto* linkPos = &outHessian[j](link * RigidSize::kDAll + RigidSize::kDTrans, i);
        auto const* parentPos = &outHessian[j](parent * RigidSize::kDAll + RigidSize::kDTrans, i);
        Store<RigidSize::kDRot>(linkPos, Load<RigidSize::kDRot, Vec4r>(parentPos));
      });
    });

    // Recursive-block: R_{k-1}-part
    if (parent >= 0) {
      auto tDelta =
          linkTransforms[link].VGetTranslation() - linkTransforms[parent].VGetTranslation();
      ForEachDof<false, true>(dofInfo, parents, parent, [&](int i) {
        // dw/di
        Vec4r dWdiV;
        AsColumnVectorView<3>(dWdiV) =
            jacobian.template Block<3, 1>(parent * RigidSize::kDAll + RigidSize::kDTrans, i, 3, 1);
        ForEachDof<false, true>(dofInfo, parents, parent, [&](int j) {
          // dw/dj
          Vec4r dWdjV;
          AsColumnVectorView<3>(dWdjV) = jacobian.template Block<3, 1>(
              parent * RigidSize::kDAll + RigidSize::kDTrans, j, 3, 1);
          // d2w/di/dj
          Vec4r ddWdidjV;
          AsColumnVectorView<3>(ddWdidjV) = outHessian[j].template Block<3, 1>(
              parent * RigidSize::kDAll + RigidSize::kDTrans, i, 3, 1);
          // ([d2w/di/dj]+0.5[dw/dj][dw/di]+0.5[dw/di][dw/dj])(t_k-t_{k-1})
          Vec4r tmp = Cross3(ddWdidjV, tDelta) +
              lie::MultVecbTD2MultRotVecaDRot2MultVecc(tDelta, dWdiV, dWdjV);
          auto* linkPos = &outHessian[j](link * RigidSize::kDAll, i);
          Store<RigidSize::kDRot>(linkPos, tmp + Load<3, Vec4r>(linkPos));
        });
      });
    }

    // Cross-block: translational part. Write full Vec4r because rotation will overwrite.
    {
      ForEachDof<false, true>(jointDofsLink, [&](int j, int jOff) {
        // we reuse memory here, since this is the last block
        AsColumnVectorView<3>(dWdjV[jOff]) =
            jacobian.template Block<3, 1>(link * RigidSize::kDAll, j, 3, 1);
      });
      ForEachDof<true, false>(jointDofsLink, [&](int j, int jOff) {
        AsColumnVectorView<3>(dtdjV[jOff]) =
            jacobian.template Block<3, 1>(link * RigidSize::kDAll, j, 3, 1);
      });
      ForEachDof<false, true>(dofInfo, parents, parent, [&](int i) {
        Vec4r dWdiV;
        AsColumnVectorView<3>(dWdiV) =
            jacobian.template Block<3, 1>(parent * RigidSize::kDAll + RigidSize::kDTrans, i, 3, 1);
        ForEachDof<false, true>(jointDofsLink, [&](int j, int jOff) {
          auto HEntry = Cross3(dWdiV, dWdjV[jOff]);
          // Two symmetric blocks:
          // outHessian(link * RigidSize::kDAll, i, j)
          // outHessian(link * RigidSize::kDAll, j, i)
          Store(&outHessian[i](link * RigidSize::kDAll, j), HEntry);
          Store(&outHessian[j](link * RigidSize::kDAll, i), HEntry);
        });
        ForEachDof<true, false>(jointDofsLink, [&](int j, int jOff) {
          auto HEntry = Cross3(dWdiV, dtdjV[jOff]);
          // Two symmetric blocks:
          // outHessian(link * RigidSize::kDAll, i, j)
          // outHessian(link * RigidSize::kDAll, j, i)
          Store<RigidSize::kDTrans>(&outHessian[i](link * RigidSize::kDAll, j), HEntry);
          Store<RigidSize::kDTrans>(&outHessian[j](link * RigidSize::kDAll, i), HEntry);
        });
      });
    }

    // Cross-block: rotational part
    {
      ForEachDof<false, true>(jointDofsLink, [&](int j, int jOff) {
        AsColumnVectorView<3>(dWdjV[jOff]) =
            jacobian.template Block<3, 1>(link * RigidSize::kDAll + RigidSize::kDTrans, j, 3, 1);
      });
      ForEachDof<false, true>(dofInfo, parents, parent, [&](int i) {
        Vec4r dWdiV;
        AsColumnVectorView<3>(dWdiV) =
            jacobian.template Block<3, 1>(parent * RigidSize::kDAll + RigidSize::kDTrans, i, 3, 1);
        ForEachDof<false, true>(jointDofsLink, [&](int j, int jOff) {
          auto HEntry = Cross3(dWdiV, dWdjV[jOff]) * .5_r;
          // Two symmetric blocks:
          // outHessian(link * RigidSize::kDAll + RigidSize::kDTrans, i, j)
          // outHessian(link * RigidSize::kDAll + RigidSize::kDTrans, j, i)
          auto* linkiPos = &outHessian[i](link * RigidSize::kDAll + RigidSize::kDTrans, j);
          auto* linkjPos = &outHessian[j](link * RigidSize::kDAll + RigidSize::kDTrans, i);
          Store<RigidSize::kDRot>(linkiPos, HEntry);
          Store<RigidSize::kDRot>(linkjPos, HEntry);
        });
      });
    }
  }
}

static void HessianContractSelfBlock(
    Span<ArticulatedDofInfo const> dofInfo,
    Span<Real3 const> jointAxes,
    Span<int const> parents,
    int link,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT const> jointTransforms,
    Span<TransformRT const> linkTransforms,
    Span<real const> contractedVector,
    RowMatrixView<real> outHessianContracted) {
  // Self-block: rotational part is always zero, and we only handle translational part
  int parent = parents[link];

  // Common link-joint Jacobian (dTLinkdTJoint = dRLinkdRJoint)
  Quaternion const& qOuter = restTransforms[link].parentFromOuter.GetRotation();
  Quaternion const& qParent = (parent != -1 ? linkTransforms[parent] : worldFromRoot).GetRotation();
  auto dXLinkdXJointT = ToVMatrix3x3Transpose(qParent * qOuter);

  int off = dofInfo[link].GetRotOffset();
  auto tInner =
      jointTransforms[link].GetRotation() * restTransforms[link].innerFromBone.VGetTranslation();
  Vec4r contractedVectorTransV = Load<3, Vec4r>(&contractedVector[link * RigidSize::kDAll]);
  if (dofInfo[link].rotSize == 1) {
    auto axisV = ToSimd(jointAxes[link]);
    auto tmp = DotVecMat3x3(
        lie::MultVecbTD2MultRotVecaDRot2MultVecc(tInner, axisV, axisV), dXLinkdXJointT);
    // outHessianContracted(off, off)
    outHessianContracted(off, off) += Dot<3>(tmp, contractedVectorTransV);
  } else if (dofInfo[link].rotSize == 3) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        auto tmp =
            DotVecMat3x3(lie::MultAxisaTD2MultRotVecDRot2MultAxisb(tInner, i, j), dXLinkdXJointT);
        // outHessianContracted(off + i, off + j)
        outHessianContracted(off + i, off + j) += Dot<3>(tmp, contractedVectorTransV);
      }
    }
  }
}

// Compute the Hessian contracted with a vector using adjoint method, see:
// https://overleaf.thefacebook.com/project/68c09b3562fc0ce28fa53543
void HessianContract(
    Span<ArticulatedDofInfo const> dofInfo,
    Span<Real3 const> jointAxes,
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    Span<TransformRT const> jointTransforms,
    Span<TransformRT const> linkTransforms,
    Span<real const> inContractedVector,
    RowMatrixView<real const> jacobian,
    RowMatrixView<real> outHessianContracted,
    ColumnVectorView<real> outReducedGradient) {
  MOCHI_ASSERT(
      (outHessianContracted.Rows() == jacobian.Cols() &&
       outHessianContracted.Cols() == jacobian.Cols()) ||
          outHessianContracted.size() == 0,
      "outHessianContracted can be empty then H:v is not computed. Otherwise, it must be of size |DOF| x |DOF|");
  MOCHI_ASSERT(
      outReducedGradient.Rows() == jacobian.Cols() || outReducedGradient.size() == 0,
      "outReducedGradient can be empty then J^Tv is not computed. Otherwise, it must be of size |DOF|");

  // Initialize all matrix Z to zero. Z is only used for H:v contraction
  MOCHI_FILO_STACK_ALLOCATOR(
      allocator, 100 * sizeof(VMatrix3x3r) + 100 * RigidSize::kDAll); // Stack mem for 100 links
  DynamicArray<VMatrix3x3r> Z(
      outHessianContracted.empty() ? 0 : isize(parents), VMatrix3x3r{}, &allocator);
  DynamicArray<real> contractedVector(inContractedVector.size(), &allocator);
  contractedVector = inContractedVector;

  Vec4r dWdiV;
  Vec4r tDelta;
  Vec4r dWdjV[3];
  Vec4r dtdjV[3];
  Vec4r tmpR, tmpT;
  Vec4r dXLinkdXJointV;
  Vec4r deltaZParentV;
  Vec4r rigidGradientRotAndTransV;
  VMatrix3x3r skewTDelta, ZDelta, ZLink;
  Vec4r contractedVectorTransV;
  Vec4r contractedVectorRotV;

  // For each link (cost=O(|q|^2))
  for (int link = isize(parents) - 1; link >= 0; --link) {
    int parent = parents[link];
    auto const& jointDofsLink = dofInfo[link];

    contractedVectorTransV = Load<3, Vec4r>(&contractedVector[link * RigidSize::kDAll]);
    contractedVectorRotV =
        Load<3, Vec4r>(&contractedVector[link * RigidSize::kDAll + RigidSize::kDTrans]);

    // Term I: (self-block) rotational part is always zero,
    // and we only handle translational part (cost=O(1))
    if (!outHessianContracted.empty()) {
      HessianContractSelfBlock(
          dofInfo,
          jointAxes,
          parents,
          link,
          restTransforms,
          worldFromRoot,
          jointTransforms,
          linkTransforms,
          contractedVector,
          outHessianContracted);
    }

    // Term II (cost=O(|q|))
    ForEachDof<false, true>(jointDofsLink, [&](int j, int jOff) {
      AsColumnVectorView<3>(dWdjV[jOff]) =
          jacobian.template Block<3, 1>(link * RigidSize::kDAll + RigidSize::kDTrans, j, 3, 1);
    });
    if (!outHessianContracted.empty()) {
      ForEachDof<false, true>(dofInfo, parents, parent, [&](int i) {
        AsColumnVectorView<3>(dWdiV) =
            jacobian.template Block<3, 1>(parent * RigidSize::kDAll + RigidSize::kDTrans, i, 3, 1);
        tmpR = Cross3(contractedVectorRotV, dWdiV);
        for (int j = 0; j < jointDofsLink.rotSize; j++) {
          auto HEntry = 0.5_r * Dot<3>(dWdjV[j], tmpR);
          // Two symmetric blocks: outHessianContracted(i, jointDofsLink.GetRotOffset() + j)
          outHessianContracted(i, jointDofsLink.GetRotOffset() + j) += HEntry;
          outHessianContracted(jointDofsLink.GetRotOffset() + j, i) += HEntry;
        }
      });
    }

    // Term III (cost=O(|1|))
    if (!outHessianContracted.empty()) {
      if (parent >= 0) {
        tDelta = linkTransforms[link].VGetTranslation() - linkTransforms[parent].VGetTranslation();
        skewTDelta = Skew3(tDelta);
        ZDelta = Dot3x3(Skew3(contractedVectorTransV), skewTDelta) * .5_r;
        ZDelta = Transpose3x3(ZDelta) + ZDelta;
        Z[parent] += Z[link] + ZDelta;
      }
      ForEachDof<false, true>(jointDofsLink, [&](int i, int iOff) {
        ForEachDof<false, true>(jointDofsLink, [&](int j, int jOff) {
          outHessianContracted(j, i) += Dot<3>(dWdjV[iOff], DotMatVec3x3(Z[link], dWdjV[jOff]));
        });
      });
    }

    // Common link-joint Jacobian (dTLinkdTJoint = dRLinkdRJoint)
    Quaternion const& qOuter = restTransforms[link].parentFromOuter.GetRotation();
    Quaternion const& qParent =
        (parent != -1 ? linkTransforms[parent] : worldFromRoot).GetRotation();
    dXLinkdXJointV = (qParent * qOuter * jointTransforms[link].GetRotation()) *
        restTransforms[link].innerFromBone.VGetTranslation();

    // Term IV (cost=O(|q|))
    // Prepare dtdj
    ForEachDof<true, false>(jointDofsLink, [&](int j, int jOff) {
      AsColumnVectorView<3>(dtdjV[jOff]) =
          jacobian.template Block<3, 1>(link * RigidSize::kDAll, j, 3, 1);
    });
    if (!outHessianContracted.empty()) {
      // Prepare for tensor to contract: dWdi : ZLink : dWdj
      ZLink = Z[link] + Dot3x3(Skew3(contractedVectorTransV), Skew3(dXLinkdXJointV));
      ForEachDof<false, true>(dofInfo, parents, parent, [&](int i) {
        AsColumnVectorView<3>(dWdiV) =
            jacobian.template Block<3, 1>(parent * RigidSize::kDAll + RigidSize::kDTrans, i, 3, 1);
        tmpR = DotVecMat3x3(dWdiV, ZLink);
        ForEachDof<false, true>(jointDofsLink, [&](int j, int jOff) {
          auto HEntry = Dot<3>(tmpR, dWdjV[jOff]);
          // Two symmetric blocks: outHessianContracted(i, j)
          outHessianContracted(i, j) += HEntry;
          outHessianContracted(j, i) += HEntry;
        });
        tmpT = Cross3(contractedVectorTransV, dWdiV);
        ForEachDof<true, false>(jointDofsLink, [&](int j, int jOff) {
          auto HEntry = Dot<3>(tmpT, dtdjV[jOff]);
          // Two symmetric blocks: outHessianContracted(i, j)
          outHessianContracted(i, j) += HEntry;
          outHessianContracted(j, i) += HEntry;
        });
      });
    }

    // Gradient term (cost=O(1)) upon user request
    if (!outReducedGradient.empty()) {
      rigidGradientRotAndTransV =
          contractedVectorRotV + Cross3(dXLinkdXJointV, contractedVectorTransV);
      ForEachDof<false, true>(jointDofsLink, [&](int j, int jOff) {
        outReducedGradient[j] += Dot<3>(rigidGradientRotAndTransV, dWdjV[jOff]);
      });
      ForEachDof<true, false>(jointDofsLink, [&](int j, int jOff) {
        outReducedGradient[j] += Dot<3>(contractedVectorTransV, dtdjV[jOff]);
      });
    }

    // Term V: Update z (cost=O(|1|))
    // This will modify the contractedVector[parent] in-place
    if (parent >= 0) {
      // z_{parent}^{rot} += [t_k-t_{k-1}] * Θ_k
      tDelta = linkTransforms[link].VGetTranslation() - linkTransforms[parent].VGetTranslation();
      deltaZParentV = Cross3(tDelta, contractedVectorTransV);
      AsView(contractedVector)
          .MiddleRows(parent * RigidSize::kDAll + RigidSize::kDTrans, RigidSize::kDRot) +=
          AsColumnVectorView<3>(deltaZParentV);
      // z_{parent} += z_{link}
      AsView(contractedVector).MiddleRows(parent * RigidSize::kDAll, RigidSize::kDAll) +=
          AsConstView(contractedVector).MiddleRows(link * RigidSize::kDAll, RigidSize::kDAll);
    }
  }
}

} // namespace mochi::articulated
