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

#include <mochi_core/element_operations/fem_rod.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/constraints.h>
#include <mochi_core/utils/lie.h>
#include <mochi_core/utils/rodrigues_utils.h>

namespace mochi {

void EvalDeformableNodeFixedConstraint(
    TransformRT const& worldFromLocal,
    Real3 const& posLocal,
    Real3 const& posTargetWorld,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 3>* outJac,
    RowMatrix<real, 3, 3>* outJacTarget) {
  if (outVal) {
    Vec4r posWorld = worldFromLocal.TransformPoint(ToSimd(posLocal));
    Vec4r vecDiff = posWorld - ToSimd(posTargetWorld);
    Store<3>(outVal->data(), vecDiff);
  }

  if (outJac) {
    *outJac = AsMatrixView(VGetRotationMatrix(worldFromLocal));
  }

  if (outJacTarget) {
    outJacTarget->SetIdentity();
    *outJacTarget = -(*outJacTarget);
  }
}

void EvalDeformableNodeToDeformableNodeConstraint(
    TransformRT const& worldFromLocalA,
    TransformRT const& worldFromLocalB,
    Real3 const& posLocalA,
    Real3 const& posLocalB,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 6>* outJac) {
  if (outVal) {
    Vec4r posAWorld = worldFromLocalA.TransformPoint(ToSimd(posLocalA));
    Vec4r posBWorld = worldFromLocalB.TransformPoint(ToSimd(posLocalB));
    Vec4r vecDiff = posAWorld - posBWorld;
    Store<3>(outVal->data(), vecDiff);
  }

  if (outJac) {
    outJac->LeftCols<3>(3) = AsMatrixView(VGetRotationMatrix(worldFromLocalA));
    outJac->RightCols<3>(3) = AsMatrixView(-VGetRotationMatrix(worldFromLocalB));
  }
}

void EvalDeformableNodeToRigidConstraint(
    TransformRT const& worldFromLocalRigid,
    TransformRT const& worldFromLocalDeformable,
    Real3 const& posLocalDeformable,
    Real3 const& posLocalRigid,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 9>* outJac) {
  static_assert(RigidSize::kDAll == 6, "Invalid rigid size");

  Vec4r radiusVec = worldFromLocalRigid.GetRotation() * ToSimd(posLocalRigid);

  if (outVal) {
    Vec4r posRigidWorld = worldFromLocalRigid.VGetTranslation() + radiusVec;
    Vec4r posNodeWorld = worldFromLocalDeformable.TransformPoint(ToSimd(posLocalDeformable));
    Vec4r vecDiff = posRigidWorld - posNodeWorld;
    Store<3>(outVal->data(), vecDiff);
  }

  if (outJac) {
    outJac->LeftCols<3>(3).SetIdentity();
    outJac->MiddleCols<3>(3, 3) = AsMatrixView(lie::DMultRotVecDRot(radiusVec));
    outJac->RightCols<3>(3) = AsMatrixView(-VGetRotationMatrix(worldFromLocalDeformable));
  }
}

void EvalRigidPositionFixedConstraint(
    TransformRT const& worldFromLocal,
    Real3 const& posLocal,
    Real3 const& posTargetWorld,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 6>* outJac,
    RowMatrix<real, 3, 3>* outJacTarget) {
  static_assert(RigidSize::kDAll == 6, "Invalid rigid size");

  Vec4r radiusVec = worldFromLocal.GetRotation() * ToSimd(posLocal);

  if (outVal) {
    Vec4r posWorld = radiusVec + worldFromLocal.VGetTranslation();
    Vec4r vecDiff = posWorld - ToSimd(posTargetWorld);
    Store<3>(outVal->data(), vecDiff);
  }

  if (outJac) {
    outJac->LeftCols<3>(3).SetIdentity();
    outJac->RightCols<3>(3) = AsMatrixView(lie::DMultRotVecDRot(radiusVec));
  }

  if (outJacTarget) {
    outJacTarget->SetIdentity();
    *outJacTarget = -(*outJacTarget);
  }
}

void EvalRigidPositionToRigidTargetConstraint(
    TransformRT const& worldFromLocal,
    TransformRT const& worldFromLocalTarget,
    Real3 const& posLocal,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 6>* outJac,
    RowMatrix<real, 3, 6>* outJacTarget) {
  static_assert(RigidSize::kDAll == 6, "Invalid rigid size");

  Vec4r radiusVec = worldFromLocal.GetRotation() * ToSimd(posLocal);
  Vec4r radiusVecTarget = worldFromLocalTarget.GetRotation() * ToSimd(posLocal);

  if (outVal) {
    Vec4r posWorld = worldFromLocal.VGetTranslation() + radiusVec;
    Vec4r posWorldTarget = worldFromLocalTarget.VGetTranslation() + radiusVecTarget;
    Vec4r vecDiff = posWorld - posWorldTarget;
    Store<3>(outVal->data(), vecDiff);
  }

  if (outJac) {
    outJac->LeftCols<3>(3).SetIdentity();
    outJac->RightCols<3>(3) = AsMatrixView(lie::DMultRotVecDRot(radiusVec));
  }

  if (outJacTarget) {
    outJacTarget->LeftCols<3>(3).SetIdentity();
    outJacTarget->LeftCols<3>(3) = -outJacTarget->LeftCols<3>(3);
    outJacTarget->RightCols<3>(3) = AsMatrixView(lie::DMultRotVecDRot(-radiusVecTarget));
  }
}

void EvalRotationFixedConstraint(
    Quaternion const& rotWorldFromLocal,
    Quaternion const& rotLocal,
    Quaternion const& rotTargetWorld,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 3>* outJac,
    RowMatrix<real, 3, 3>* outJacTarget) {
  static_assert(RigidSize::kDRot == 3, "Invalid rotation size");

  Quaternion rotWorld = rotWorldFromLocal * rotLocal;
  Quaternion qd = Normalize(rotWorld * rotTargetWorld.GetConjugate());
  Vec4r rd = qd.VToRotationVector();

  if (outVal) {
    Store<3>(outVal->data(), rd);
  }

  if (outJac || outJacTarget) {
    auto Rd = ToVMatrix3x3(qd);
    auto DrdDRd = DRotVectorDRotIncrement(rd, Rd);
    if (outJac) {
      // DRd/DR = eye
      *outJac = AsMatrixView(DrdDRd);
    }
    if (outJacTarget) {
      // DRd/DRt = -Rd
      *outJacTarget = AsMatrixView(Dot3x3(DrdDRd, lie::DMultRotaRotTRotbDRot(Rd)));
    }
  }
}

void EvalRigidSphericalJointConstraint(
    TransformRT const& worldFromLocalA,
    TransformRT const& worldFromLocalB,
    Real3 const& posLocalA,
    Real3 const& posLocalB,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 12>* outJac) {
  static_assert(RigidSize::kDAll == 6, "Invalid rigid size");

  Vec4r radiusVecA = worldFromLocalA.GetRotation() * ToSimd(posLocalA);
  Vec4r radiusVecB = worldFromLocalB.GetRotation() * ToSimd(posLocalB);

  if (outVal) {
    Vec4r posAWorld = worldFromLocalA.VGetTranslation() + radiusVecA;
    Vec4r posBWorld = worldFromLocalB.VGetTranslation() + radiusVecB;
    Vec4r vecDiff = posAWorld - posBWorld;
    Store<3>(outVal->data(), vecDiff);
  }

  if (outJac) {
    outJac->LeftCols<3>(3).SetIdentity();
    outJac->MiddleCols<3>(3, 3) = AsMatrixView(lie::DMultRotVecDRot(radiusVecA));
    outJac->MiddleCols<3>(6, 3) = -outJac->LeftCols<3>(3);
    outJac->RightCols<3>(3) = AsMatrixView(lie::DMultRotVecDRot(-radiusVecB));
  }
}

void EvalRigidPrismaticJointConstraint(
    TransformRT const& worlFromLocalA,
    Real3 const& posWorldFromLocalB,
    Quaternion const& localFrame,
    Real3 const& tRef,
    std::optional<real> max,
    std::optional<real> min,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 9>* outJac) {
  static_assert(RigidSize::kDAll == 6, "Invalid rigid size");

  // (Ra, ta): rotation and translation of actor A
  // (Rb, tb): rotation and translation of actor B
  // Rfree: rotation to free axis (expressed in local frame of A)
  // tref: reference translation (in the null-space of ufree)

  // trel = tb - ta: relative translation
  // trela = RaT trel: relative translation in local frame of A
  // tnull = Rfree trela: relative translation orthogonal to ufree

  // C = tnull - tref = Rfree RaT (tb - ta) - tref: constraint definition
  // dC/dta = - Rfree RaT
  // dC/dtb = Rfree RaT
  // dC/dra = Rfree RaT skew(trel)

  Vec4r trel = ToSimd(posWorldFromLocalB) - worlFromLocalA.VGetTranslation();
  VMatrix3x3r Ra = VGetRotationMatrix(worlFromLocalA);
  VMatrix3x3r Rfree = ToVMatrix3x3(localFrame);
  Vec4r c = DotMatVec3x3(Rfree, DotVecMat3x3(trel, Ra)) - ToSimd(tRef);

  bool limitActive = false;
  real cZ = c[2];
  if (max && cZ > max.value()) {
    cZ -= max.value();
    limitActive = true;
  } else if (min && cZ < min.value()) {
    cZ -= min.value();
    limitActive = true;
  } else {
    cZ = 0_r;
  }
  c = Set<2>(c, cZ);

  if (outVal) {
    Store<3>(outVal->data(), c);
  }

  if (outJac) {
    VMatrix3x3r dCdtrel = Dot3x3(Rfree, Transpose3x3(Ra));
    outJac->LeftCols<3>(3) = AsMatrixView(-dCdtrel);
    outJac->MiddleCols<3>(3, 3) = AsMatrixView(lie::DMultMatRotTVecDRot(dCdtrel, trel));
    outJac->RightCols<3>(3) = AsMatrixView(dCdtrel);
    if (!limitActive) {
      outJac->BottomRows<1>(1).SetZero();
    }
  }
}

void EvalJointRotationTargetConstraint(
    Quaternion const& rotA,
    Quaternion const& rotB,
    Quaternion const& refFrame,
    Quaternion const& targetRot,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 6>* outJac,
    RowMatrix<real, 3, 3>* outJacTarget) {
  static_assert(RigidSize::kDRot == 3, "Invalid rotation size");

  // Rab = R0^T Ra^T Rb R0
  // Rabr = R0^T Rr R0
  // Rd = Rabr^T Rab = R0^T Rr^T Ra^T Rb R0
  // define Ra0 = Ra Rr R0, Rb0 = Rb R0
  // Rd = Ra0^T Rb0

  Quaternion qA0Conj = (rotA * targetRot * refFrame).GetConjugate();
  Quaternion qd = qA0Conj * rotB * refFrame;
  Vec4r rd = qd.VToRotationVector();

  if (outVal) {
    Store<3>(outVal->data(), rd);
  }

  if (outJac || outJacTarget) {
    // C = rotVec(Rd),
    // dC/dRa = drd/dRd * dRd/dRa
    // dC/dRb = drd/dRd * dRd/dRb
    // dC/dRr = drd/dRd * dRd/dRr
    // dRd/dRa = - R0^T * Rr^T * Ra^T = - Ra0^T
    // dRd/dRb = R0^T * Rr^T * Ra^T = Ra0^T
    // dRd/dRr = - R0^T * Rr^T

    VMatrix3x3r DrdDRd = DRotVectorDRotIncrement(rd, qd);
    if (outJac) {
      VMatrix3x3r DRdDRb = lie::DMultRotaRotRotbDRot(ToVMatrix3x3(qA0Conj));
      VMatrix3x3r DrdDRb = Dot3x3(DrdDRd, DRdDRb);
      outJac->LeftCols<3>(3) = AsMatrixView(-DrdDRb);
      outJac->RightCols<3>(3) = AsMatrixView(DrdDRb);
    }
    if (outJacTarget) {
      VMatrix3x3r DRdDRr =
          lie::DMultRotaRotTRotbDRot(ToVMatrix3x3((targetRot * refFrame).GetConjugate()));
      VMatrix3x3r DrdDRr = Dot3x3(DrdDRd, DRdDRr);
      *outJacTarget = AsMatrixView(DrdDRr);
    }
  }
}

void EvalRodElementRotationToRigidConstraint(
    Quaternion const& rotRigid,
    TransformRT const& worldFromLocalRod,
    Vec4r const& X0Rod,
    Vec4r const& X1Rod,
    Vec4r const& frameAxisRod,
    Span<real const> rodElementDofs,
    Quaternion const& refFrame,
    Quaternion const& targetRot,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 11>* outJac,
    RowMatrix<real, 3, 3>* outJacTarget) {
  VMatrix3x3r rotRodMatLocal MOCHI_NO_INIT;
  NdArray<real, 3, 3, 8> drotRodMatLocal_dDofs MOCHI_NO_INIT;
  fem::ComputeRodElementRotationLocal(
      X0Rod,
      X1Rod,
      frameAxisRod,
      rodElementDofs,
      rotRodMatLocal,
      outJac ? &drotRodMatLocal_dDofs : nullptr);

  // Apply the world-from-local rotation to the rod element rotation
  Quaternion const rotRodLocal = Normalize(QuaternionFromMatrix(rotRodMatLocal));
  Quaternion const rotWflRod = worldFromLocalRod.GetRotation();
  Quaternion const rotRod = rotWflRod * rotRodLocal;

  // See EvalJointRotationTargetConstraint for comments on analogous computation.
  Quaternion const qRigid0Conj = (rotRigid * targetRot * refFrame).GetConjugate();
  Quaternion const qd = qRigid0Conj * rotRod * refFrame;
  Vec4r const rd = qd.VToRotationVector();

  if (outVal) {
    Store<3>(outVal->data(), rd);
  }
  if (outJac || outJacTarget) {
    VMatrix3x3r const DrdDRd = DRotVectorDRotIncrement(rd, qd);
    if (outJac) {
      // Derivative w.r.t. rigid rotation (cf. EvalJointRotationTargetConstraint)
      VMatrix3x3r const DrdDRRod =
          Dot3x3(DrdDRd, lie::DMultRotaRotRotbDRot(ToVMatrix3x3(qRigid0Conj)));
      outJac->LeftCols<3>(3) = AsMatrixView(-DrdDRRod);

      // Derivative w.r.t. rod element DoFs.
      VMatrix3x3r const preMat = ToVMatrix3x3(qRigid0Conj * rotWflRod);
      VMatrix3x3r const DrdDOmega = Dot3x3(DrdDRd, lie::DMultRotaRotRotbDRot(preMat));
      VMatrix3x3r const rotRodMatLocalT = Transpose3x3(rotRodMatLocal);

      for (int rodDofIndex = 0; rodDofIndex < 8; rodDofIndex++) {
        // dR/d(dof) for this DoF
        VMatrix3x3r DrDdof MOCHI_NO_INIT;
        for (int j = 0; j < 3; j++) {
          DrDdof[j] = Vec4r{
              drotRodMatLocal_dDofs[j][0][rodDofIndex],
              drotRodMatLocal_dDofs[j][1][rodDofIndex],
              drotRodMatLocal_dDofs[j][2][rodDofIndex]};
        }

        // Extract angular velocity: skew(ω) = dR/d(dof) · R^T
        Vec4r const omega = InvSkew3(Dot3x3(DrDdof, rotRodMatLocalT));

        // d(rd)/d(dof) = DrdDOmega · ω
        for (int i = 0; i < 3; i++) {
          (*outJac)(i, rodDofIndex + 3) = Dot<3>(DrdDOmega[i], omega);
        }
      }
    } // If outJac is requested

    if (outJacTarget) {
      VMatrix3x3r const DRdDRr =
          lie::DMultRotaRotTRotbDRot(ToVMatrix3x3((targetRot * refFrame).GetConjugate()));
      VMatrix3x3r const DrdDRr = Dot3x3(DrdDRd, DRdDRr);
      *outJacTarget = AsMatrixView(DrdDRr);
    }
  } // If either Jacobian is requested
}

void EvalJointRotationRangeConstraint(
    Quaternion const& rotA,
    Quaternion const& rotB,
    Quaternion const& refFrame,
    Quaternion const& refRot,
    Real3 const& minRotVec,
    Real3 const& maxRotVec,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 6>* outJac,
    bool& outActive) {
  static_assert(RigidSize::kDRot == 3, "Invalid rotation size");

  Quaternion qab =
      Normalize(RelativeRotation_Reference(rotA, rotB, refFrame) * refRot.GetConjugate());
  Vec4r rab = qab.VToRotationVector();

  Vec4r vec0 = Vec4r::Zero();
  Vec4r topValue = rab - ToSimd(maxRotVec);
  Vec4r botValue = ToSimd(minRotVec) - rab;
  Vec4r topMask = (topValue > vec0);
  Vec4r botMask = (botValue > vec0);
  Vec4r activeMask = topMask | botMask;

  outActive = AnyTrue<3>(activeMask);
  if (!outActive) {
    if (outVal) {
      outVal->SetZero();
    }
    if (outJac) {
      outJac->SetZero();
    }
    return;
  }

  if (outVal) {
    Vec4r value = vec0;
    value = Select(topMask, topValue, value);
    value = Select(botMask, botValue, value);
    Store<3>(outVal->data(), value);
  }

  if (outJac) {
    Vec4r sign = Select(botMask, Vec4r{-1_r}, Vec4r{1_r});

    // Compute Lie derivative of relative rotation vector rab w.r.t. each rigid rotation Ra, Rb.
    // We use the relative rotation Rab = R0^T Ra^T Rb R0 RrT.
    // drab/dRa = drab/dRab * dRab/dRa, with dRab/dRa = -R0^T Ra^T
    // drab/dRb = drab/dRab * dRab/dRb, with dRab/dRb = R0^T Ra^T
    // Then drab/dRa = - drab/dRb
    // drab/dRab is the derivative of rotation vector wrt incremental rotation.

    VMatrix3x3r DrabDRab = DRotVectorDRotIncrement(rab, qab);
    Quaternion qa0T = (rotA * refFrame).GetConjugate();
    VMatrix3x3r DrabDRb = Dot3x3(DrabDRab, lie::DMultRotaRotRotbDRot(ToVMatrix3x3(qa0T)));

    outJac->SetZero();
    if (AnyTrue<3>(activeMask & SimdMask<Vec4r>(true, false, false, false))) {
      Vec4r signi = Broadcast<0>(sign);
      Store<3>(&(*outJac)(0, 0), -DrabDRb[0] * signi);
      Store<3>(&(*outJac)(0, 3), DrabDRb[0] * signi);
    }
    if (AnyTrue<3>(activeMask & SimdMask<Vec4r>(false, true, false, false))) {
      Vec4r signi = Broadcast<1>(sign);
      Store<3>(&(*outJac)(1, 0), -DrabDRb[1] * signi);
      Store<3>(&(*outJac)(1, 3), DrabDRb[1] * signi);
    }
    if (AnyTrue<3>(activeMask & SimdMask<Vec4r>(false, false, true, false))) {
      Vec4r signi = Broadcast<2>(sign);
      Store<3>(&(*outJac)(2, 0), -DrabDRb[2] * signi);
      Store<3>(&(*outJac)(2, 3), DrabDRb[2] * signi);
    }
  }
}

void EvalRotationRangeConstraint(
    Real3 const& rotVec,
    Real3 const& minRotVec,
    Real3 const& maxRotVec,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 3>* outJac,
    bool& outActive) {
  if (outJac) {
    outJac->SetZero();
  }
  outActive = false;
  for (int i = 0; i < 3; ++i) {
    auto* valPtr = outVal ? &(*outVal)(i) : nullptr;
    auto* jacPtr = outJac ? &(*outJac)(i, i) : nullptr;
    bool thisActive = false;
    EvalSingleDofRangeConstraint(rotVec[i], minRotVec[i], maxRotVec[i], valPtr, jacPtr, thisActive);
    outActive |= thisActive;
  }
  if (outJac && outActive) {
    VMatrix3x3r drot = DRotVectorDRotIncrement(ToSimd(rotVec));
    // In-place product is not supported.
    *outJac = RowMatrix<real, 3, 3>((*outJac) * AsMatrixView(drot));
  }
}
} // namespace mochi
