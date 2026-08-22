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

// PLEASE DO NOT ADD OTHER INCLUDES HERE. This header is included in the mochi_physics public API.
#include <mochi_core/contact/contact_params.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/reflection.h>

namespace mochi {

/**
 * @brief [Experimental] Controls whether force-saturation terms use a fitted quadratic Hessian
 * independently for each saturation pathway.
 *
 * @details For each flag, `true` selects the fitted Hessian, which is more stable. `false` selects
 * the exact analytical Hessian, which may converge faster but is less stable.
 *
 * @warning This is an experimental feature. It may be changed or removed in the future. Use at
 * your own risk.
 *
 * @note When a flag is `false`, the solver will try first with the true Hessian for that pathway.
 * If the Newton iteration fails to improve, all flags are set to `true` and the iteration is
 * retried with fitted Hessians for all pathways.
 */
struct SaturationHessianParams {
  /** @brief Use fitted Hessian for contact-friction saturation. */
  bool contactFriction = true;
  /** @brief Use fitted Hessian for joint-friction saturation. */
  bool jointFriction = false;
  /** @brief Use fitted Hessian for constraint saturation. */
  bool constraintSaturation = true;

  /** @brief Construct with all three flags set to `value`. */
  static constexpr SaturationHessianParams All(bool value);

  /** @brief Returns true iff all three flags are `true`. */
  [[nodiscard]] constexpr bool AllTrue() const;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(SaturationHessianParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::SaturationHessianParams)
  MOCHI_FIELD(contactFriction)
  MOCHI_FIELD(jointFriction)
  MOCHI_FIELD(constraintSaturation)
  MOCHI_STRUCT_END()
};

/**
 * @brief [Experimental] Evaluation settings common to the full scene.
 *
 * @details They tune the evaluation of internal models (contact, constraints, etc).
 *
 * @warning This is an experimental feature. It may be changed or removed in the future. Use at
 * your own risk.
 */
struct ExperimentalEvalParams {
  /**
   * @brief Use explicit normals (from stage-start kinematics) for the evaluation of friction.
   *
   * @details Whether to treat the colliding and collider normals explicitly (using stage-start
   * kinematics) or implicitly (using current kinematics) for the evaluation of alignment and the
   * friction plane.
   *
   * @note The SDF gradient for the normal collision force is always implicit.
   * @note explicitNormals = true and implicitNormalForceForDissipation = false produces contact
   * residuals that are the exact gradients of the contact merit. This improves convergence
   * guarantees of the Newton solve, but may be less stable due to the explicit treatment.
   * @note A differentiable scene requires explicitNormals = true.
   */
  bool explicitNormals = false;

  /**
   * @brief Fade friction coefficient based on normal alignment.
   *
   * @details When enabled, friction is scaled by (maxAlignmentNormals - alignment) /
   * (maxAlignmentNormals + 1), where "alignment" is defined as the dot product between colliding
   * and collider normals.
   *
   * @note For co-dimensional colliding actors with ambiguous normals, friction fading is disabled
   * regardless of fadeFriction (normal alignment cannot be computed).
   * @note fadeFriction = true adds a non-integrable term to the residual unless @ref
   * explicitNormals = true.
   *
   * @see ContactParams::maxAlignmentNormals
   */
  bool fadeFriction = true;

  /**
   * @brief Treat the normal contact force implicitly (if true) or explicitly (if false) for
   * dissipative contact terms.
   *
   * @note Implicit treatment improves stability (especially with high-order time integrators) but
   * the resulting dissipative force is not integrable, hence it cannot be derived from an
   * objective function. This (a) may hurt convergence with an objective-based line search, and (b)
   * makes the force dresidual non-symmetric (a symmetric approximation is used). With
   * implicitNormalForceForDissipation = false, the distance used for the explicit normal force is
   * approximate with @ref explicitNormals = false, but accurate with explicitNormals = true.
   * @note Dissipative contact terms include Coulomb friction, viscous friction, and normal viscous
   * damping. Using an implicit normal force for normal damping can improve the accuracy of the
   * effective coefficient of restitution at a given time step size.
   */
  bool implicitNormalForceForDissipation = false;

  /**
   * @brief Controls whether force-saturation terms use a fitted quadratic Hessian (more stable) or
   * the exact analytical Hessian (faster convergence but less stable), independently for each
   * saturation pathway.
   *
   * @note For any flag set to false, the solver will try first with the true Hessian. If it fails,
   * it will retry with the fitted Hessian.
   */
  SaturationHessianParams fittedSaturationHessian;

  /**
   * @brief Selects which Coulomb friction smoothing model to use.
   *
   * @details C1Regularized (default) preserves existing behavior exactly. CinfRegularized is an
   * additional model that can be selected at runtime without affecting C1Regularized users.
   */
  CoulombFrictionModel frictionModel = CoulombFrictionModel::Default;

  /**
   * @brief Validate that the residual is the gradient of the objective by a directional
   * finite-difference consistency check.
   *
   * @note This operation is expensive and should only be used for debugging.
   */
  bool consistencyResNorm = false;

  /**
   * @brief Step size for finite-difference consistency check.
   */
  real consistencyResNormStep = 1e-4_r;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ExperimentalEvalParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::ExperimentalEvalParams)
  MOCHI_FIELD(explicitNormals)
  MOCHI_FIELD(fadeFriction)
  MOCHI_FIELD(implicitNormalForceForDissipation)
  MOCHI_FIELD(fittedSaturationHessian)
  MOCHI_FIELD(frictionModel)
  MOCHI_FIELD(consistencyResNorm)
  MOCHI_FIELD(consistencyResNormStep)
  MOCHI_STRUCT_END()
};

} // namespace mochi

#include "eval_params_inl.h"
