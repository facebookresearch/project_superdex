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
#include <mochi_core/utils/activations.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/lie.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/quaternion_utils.h>
#include <mochi_core/utils/rodrigues_utils.h>
#include <mochi_core/utils/vmatrix.h>

/*
  Elasto-plastic dry friction of articulated joints (see ArticulatedJointFrictionParams).

  Each joint, with a single DoF or a 3D rotation, has a friction anchor a, which stays fixed during
  a step. With z the displacement of the joint from its anchor (q - a for a single DoF, the rotation
  vector of R a^T for a rotation), the dry friction energy is E = S f(|z|), with S = coulomb +
  stictionExtra the static peak, and f the Stribeck activation with the presliding displacement as
  threshold: the force S f'(|z|) grows from zero to S at the presliding displacement. Beyond it,
  |z| minus the presliding displacement is the slip since the anchor state, so the Stribeck
  displacement is stribeckVel times the time since the anchor state, and friction follows the
  Stribeck curve of the slip velocity.

  After a step, the anchor advances by radial return: it sticks while |z| is at most the presliding
  displacement, and follows the joint at that distance beyond it.

  Rotations use left increments, R <- exp(theta) R, like the DoFs of spherical joints.
*/

namespace mochi {

namespace details {

// Dry friction energy S f(x) of a displacement x from the anchor, and its derivatives (see
// StribeckActivation), with S the static peak.
struct JointFrictionPotential {
  real energy = 0_r;
  real stiffness = 0_r; // S f''(x)
  real secant = 0_r; // S f'(x) / x, robust at x = 0
};

[[nodiscard]] inline JointFrictionPotential EvalJointFrictionPotential(
    ArticulatedJointFrictionParams const& params,
    real dtFromAnchor,
    real displacement) {
  real const staticFriction = params.coulomb + params.stictionExtra;
  MOCHI_ASSERT_VERBOSE(staticFriction > 0_r, "Static friction must be strictly positive.");
  MOCHI_ASSERT_VERBOSE(
      params.preslidingDisplacement > 0_r, "The presliding displacement must be positive.");
  MOCHI_ASSERT_VERBOSE(
      dtFromAnchor >= 0_r, "The time since the anchor state must not be negative.");

  // With a Stribeck velocity, an infinite time since the anchor state (a quasistatic step) means
  // zero slip velocity, so friction keeps its static peak, and a zero time means infinite slip
  // velocity, so friction drops to the Coulomb force beyond the presliding displacement.
  real dfInfty = params.coulomb / staticFriction;
  real stribeckDisplacement = 0_r;
  if (params.stribeckVel > 0_r) {
    if (IsFinite(dtFromAnchor)) {
      stribeckDisplacement = params.stribeckVel * dtFromAnchor;
    } else {
      dfInfty = 1_r;
    }
  }

  real f MOCHI_NO_INIT;
  real df MOCHI_NO_INIT;
  real ddf MOCHI_NO_INIT;
  real df_x MOCHI_NO_INIT;
  StribeckActivation(
      displacement, params.preslidingDisplacement, dfInfty, stribeckDisplacement, f, df, ddf, df_x);
  return {
      .energy = staticFriction * f,
      .stiffness = staticFriction * ddf,
      .secant = staticFriction * df_x};
}

} // namespace details

/**
 * @brief Adds the energy, gradient and Hessian of the dry friction of a single DoF.
 *
 * @param useFittedHessian If true, the Hessian is the secant S f'(|z|) / |z|, which is more robust
 * than the exact Hessian S f''(|z|): the exact Hessian vanishes while the DoF slides, and turns
 * negative where the Stribeck term decreases the force.
 * @param psdDRes If true, clamps negative Hessian contributions.
 * @param params Friction parameters of the joint, with nonzero dry friction.
 * @param dtFromAnchor Time since the anchor state [s], the start of the step. May be zero or
 * infinite.
 * @param pos Position of the DoF.
 * @param anchor Friction anchor of the DoF.
 * @param[out] outEnergy Accumulates the energy, if not null.
 * @param[out] outGradient Accumulates the gradient with respect to pos, if not null. Its
 * derivative with respect to the anchor is the negated exact Hessian.
 * @param[out] outHessian Accumulates the Hessian with respect to pos, if not null.
 */
inline void AddSingleDofFriction(
    bool useFittedHessian,
    bool psdDRes,
    ArticulatedJointFrictionParams const& params,
    real dtFromAnchor,
    real pos,
    real anchor,
    double* outEnergy,
    real* outGradient,
    real* outHessian) {
  if (!outEnergy && !outGradient && !outHessian) {
    return;
  }
  real const z = pos - anchor;
  auto const p = details::EvalJointFrictionPotential(params, dtFromAnchor, Abs(z));
  if (outEnergy) {
    (*outEnergy) += static_cast<double>(p.energy);
  }
  if (outGradient) {
    (*outGradient) += p.secant * z;
  }
  if (outHessian) {
    (*outHessian) += useFittedHessian ? p.secant : (psdDRes ? Max(p.stiffness, 0_r) : p.stiffness);
  }
}

/**
 * @brief Advances the friction anchor of a single DoF by radial return: it sticks while the DoF is
 * within the presliding displacement of it, and follows the DoF at that distance beyond.
 */
[[nodiscard]] inline real
AdvanceSingleDofFrictionAnchor(real pos, real anchor, real preslidingDisplacement) {
  real const z = pos - anchor;
  return Abs(z) <= preslidingDisplacement ? anchor : pos - Sign(z) * preslidingDisplacement;
}

/**
 * @brief Vector-Jacobian product of @ref AdvanceSingleDofFrictionAnchor: adds the adjoint of the
 * advanced anchor to that of the anchor while the DoF sticks, and to that of the position while it
 * slides.
 */
inline void AdvanceSingleDofFrictionAnchorVjp(
    real pos,
    real anchor,
    real preslidingDisplacement,
    real newAnchorAdjoint,
    real& inOutPosAdjoint,
    real& inOutAnchorAdjoint) {
  if (Abs(pos - anchor) <= preslidingDisplacement) {
    inOutAnchorAdjoint += newAnchorAdjoint;
  } else {
    inOutPosAdjoint += newAnchorAdjoint;
  }
}

/**
 * @brief Adds the energy, gradient and Hessian of the dry friction of a 3D rotation, such as a
 * spherical joint. See @ref AddSingleDofFriction.
 *
 * @details With z the rotation vector of R a^T, C = dz/dtheta and H = d2E/dz2, the gradient with
 * respect to the left increments of R is C^T dE/dz = dE/dz, since C^T z = z. Its exact derivative,
 * H C, is not symmetric: the Hessian is its symmetric approximation C^T H C.
 *
 * @param rot Rotation R.
 * @param anchor Friction anchor a of the rotation.
 * @param[out] outGradient Accumulates the gradient with respect to the left increments of R, if
 * not null.
 * @param[out] outHessian Accumulates the Hessian with respect to the left increments of R, if not
 * null.
 */
inline void AddRotationFriction(
    bool useFittedHessian,
    bool psdDRes,
    ArticulatedJointFrictionParams const& params,
    real dtFromAnchor,
    Quaternion const& rot,
    Quaternion const& anchor,
    double* outEnergy,
    Vec4r* outGradient,
    VMatrix3x3r* outHessian) {
  if (!outEnergy && !outGradient && !outHessian) {
    return;
  }
  Vec4r const z = (rot * anchor.GetConjugate()).VToRotationVector();
  real const zNorm = Norm<3>(z);
  auto const p = details::EvalJointFrictionPotential(params, dtFromAnchor, zNorm);
  if (outEnergy) {
    (*outEnergy) += static_cast<double>(p.energy);
  }
  if (outGradient) {
    (*outGradient) += p.secant * z;
  }
  if (outHessian) {
    // H = S f'' u u^T + S (f' / |z|) (I - u u^T), with u = z / |z|. The fitted Hessian uses the
    // secant in all directions.
    VMatrix3x3r hessianZ = VDiagonalMatrix<3>(p.secant);
    if (zNorm > 0_r && !useFittedHessian) {
      Vec4r const u = z / zNorm;
      real const stiffness = psdDRes ? Max(p.stiffness, 0_r) : p.stiffness;
      hessianZ += Outer3((stiffness - p.secant) * u, u);
    }
    VMatrix3x3r const dzdtheta = DRotVectorDRotIncrement(z);
    (*outHessian) += Dot3x3(Transpose3x3(dzdtheta), Dot3x3(hessianZ, dzdtheta));
  }
}

/**
 * @brief Transpose product of the derivative of the dry friction gradient of a rotation (see
 * @ref AddRotationFriction) with respect to the left increments phi of its anchor:
 * (dg/dphi)^T v = (dz/dphi)^T H v, with the exact H, since g = dE/dz.
 */
[[nodiscard]] inline Vec4r RotationFrictionDGradientDAnchorTransposeProduct(
    ArticulatedJointFrictionParams const& params,
    real dtFromAnchor,
    Quaternion const& rot,
    Quaternion const& anchor,
    Vec4r v) {
  Quaternion const relRot = rot * anchor.GetConjugate();
  Vec4r const z = relRot.VToRotationVector();
  real const zNorm = Norm<3>(z);
  auto const p = details::EvalJointFrictionPotential(params, dtFromAnchor, zNorm);
  Vec4r hessianZV = p.secant * v;
  if (zNorm > 0_r) {
    Vec4r const u = z / zNorm;
    hessianZV += ((p.stiffness - p.secant) * Dot<3>(u, v)) * u;
  }
  // dz/dphi = dz/dtheta d(R a^T)/da.
  return DotVecMat3x3(
      DotVecMat3x3(hessianZV, DRotVectorDRotIncrement(z)),
      lie::DMultRotaRotTRotbDRot(ToVMatrix3x3(relRot)));
}

/**
 * @brief Advances the friction anchor of a rotation by radial return (see
 * @ref AdvanceSingleDofFrictionAnchor): while the rotation vector z of R a^T exceeds the presliding
 * displacement d, the anchor becomes exp(-d z / |z|) R.
 */
[[nodiscard]] inline Quaternion AdvanceRotationFrictionAnchor(
    Quaternion const& rot,
    Quaternion const& anchor,
    real preslidingDisplacement) {
  Vec4r const z = (rot * anchor.GetConjugate()).VToRotationVector();
  real const zNorm = Norm<3>(z);
  if (zNorm <= preslidingDisplacement) {
    return anchor;
  }
  Quaternion const newAnchor =
      Quaternion::FromRotationVector((-preslidingDisplacement / zNorm) * z) * rot;
  return Normalize(newAnchor);
}

/**
 * @brief Vector-Jacobian product of @ref AdvanceRotationFrictionAnchor, with the adjoints of the
 * rotations with respect to their left increments.
 *
 * @details While sliding, the new anchor is exp(w) R, with w = -d u and u = z / |z|. Then
 * psi = J(w) dw + exp(w) theta, with J the left Jacobian of SO(3), dw = P dz,
 * P = -(d / |z|) (I - u u^T), and dz = (dz/dtheta) (theta + d(R a^T)/da phi).
 */
inline void AdvanceRotationFrictionAnchorVjp(
    Quaternion const& rot,
    Quaternion const& anchor,
    real preslidingDisplacement,
    Vec4r newAnchorAdjoint,
    Vec4r& inOutRotAdjoint,
    Vec4r& inOutAnchorAdjoint) {
  Quaternion const relRot = rot * anchor.GetConjugate();
  Vec4r const z = relRot.VToRotationVector();
  real const zNorm = Norm<3>(z);
  if (zNorm <= preslidingDisplacement) {
    inOutAnchorAdjoint += newAnchorAdjoint;
    return;
  }
  Vec4r const u = z / zNorm;
  Vec4r const w = -preslidingDisplacement * u;

  // wBar = J(w)^T psiBar, zBar = P wBar (P is symmetric), and thetaBar from z = zBar dz/dtheta.
  Vec4r const wAdjoint = DotVecMat3x3(newAnchorAdjoint, DRotIncrementDRotVector(w));
  Vec4r const zAdjoint = (-preslidingDisplacement / zNorm) * (wAdjoint - Dot<3>(u, wAdjoint) * u);
  Vec4r const zThetaAdjoint = DotVecMat3x3(zAdjoint, DRotVectorDRotIncrement(z));
  inOutRotAdjoint += zThetaAdjoint +
      DotVecMat3x3(newAnchorAdjoint, ToVMatrix3x3(Quaternion::FromRotationVector(w)));
  inOutAnchorAdjoint +=
      DotVecMat3x3(zThetaAdjoint, lie::DMultRotaRotTRotbDRot(ToVMatrix3x3(relRot)));
}

namespace details {

// Trust margin from z0 · z and |z0| (see SingleDofFrictionTrustMargin).
[[nodiscard]] inline real JointFrictionTrustMargin(
    real referenceDotDisplacement,
    real referenceNorm,
    real preslidingDisplacement) {
  real constexpr kMinReferenceRatio = 0.1_r;
  if (referenceNorm < kMinReferenceRatio * preslidingDisplacement) {
    return kInf;
  }
  return 1_r + referenceDotDisplacement / (preslidingDisplacement * referenceNorm);
}

} // namespace details

/**
 * @brief Trust margin (see @ref SnleProblem::trustMargin) of the dry friction of a single DoF: the
 * signed distance from its displacement to the point where it yields in reverse, in presliding
 * displacements.
 *
 * @details As for contacts (see @ref ComputeBatchTrustMargin), with z0 the displacement from the
 * anchor at the stage start and z the current one, the trusted region is the half-space
 * z0 · z / |z0| >= -δ of displacements that have not yielded in reverse, past the anchor by the
 * presliding displacement δ, and the margin m = 1 + z0 · z / (δ |z0|) is the signed distance to its
 * boundary in units of δ, affine in the DoF. A reference shorter than 0.1 δ has no reliable
 * direction: such joints get +inf.
 */
[[nodiscard]] inline real SingleDofFrictionTrustMargin(
    real stageStartPos,
    real pos,
    real anchor,
    real preslidingDisplacement) {
  real const z0 = stageStartPos - anchor;
  return details::JointFrictionTrustMargin(z0 * (pos - anchor), Abs(z0), preslidingDisplacement);
}

/**
 * @brief Trust margin of the dry friction of a 3D rotation (see @ref SingleDofFrictionTrustMargin),
 * with z0 and z the rotation vectors of R0 a^T and R a^T, R0 the stage-start rotation. The margin
 * is affine in z, so nearly affine in the left increments of R at the small displacements where it
 * matters.
 */
[[nodiscard]] inline real RotationFrictionTrustMargin(
    Quaternion const& stageStartRot,
    Quaternion const& rot,
    Quaternion const& anchor,
    real preslidingDisplacement) {
  Quaternion const anchorInv = anchor.GetConjugate();
  Vec4r const z0 = (stageStartRot * anchorInv).VToRotationVector();
  Vec4r const z = (rot * anchorInv).VToRotationVector();
  return details::JointFrictionTrustMargin(Dot<3>(z0, z), Norm<3>(z0), preslidingDisplacement);
}

} // namespace mochi
