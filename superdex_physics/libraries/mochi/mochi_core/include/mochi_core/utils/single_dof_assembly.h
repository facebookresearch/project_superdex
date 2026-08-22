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
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/differentiability.h>

namespace mochi {

// Function to add single-dof inertia terms.
template <GradTarget kGradTarget>
inline void AddSingleDofInertia(
    real currentPos,
    real inertia,
    real stageStartPos,
    real stageStartVel,
    real dtStage,
    double* outEnergy,
    real* outGradient,
    real* outHessian) {
  if (!outEnergy && !outGradient && !outHessian) {
    return;
  }

  real invDt = 1_r / dtStage;

  if (outEnergy || outGradient) {
    real deltaVel = invDt * (currentPos - stageStartPos) - stageStartVel;

    if (outEnergy) {
      (*outEnergy) += static_cast<double>(0.5_r * inertia * Sqr(deltaVel));
    }

    if (outGradient) {
      if constexpr (
          kGradTarget == GradTarget::Previous || kGradTarget == GradTarget::PreviousDelta) {
        (*outGradient) -= invDt * inertia * deltaVel;
      } else {
        static_assert(kGradTarget == GradTarget::Current, "Unknown target type");
        (*outGradient) += invDt * inertia * deltaVel;
      }
    }
  }

  if (outHessian) {
    (*outHessian) += inertia * Sqr(invDt);
  }
}

// Function to add single-dof damping terms.
template <GradTarget kGradTarget>
inline void AddSingleDofDamping(
    real currentPos,
    real damping,
    real stageStartPos,
    real dtStage,
    double* outEnergy,
    real* outGradient,
    real* outHessian) {
  static_assert(
      kGradTarget == GradTarget::Current || kGradTarget == GradTarget::Previous,
      "Unsupported target");
  if (!outEnergy && !outGradient && !outHessian) {
    return;
  }

  real dampingOverDt = damping / dtStage;

  if (outEnergy) {
    (*outEnergy) += static_cast<double>(0.5_r * dampingOverDt * Sqr(currentPos - stageStartPos));
  }

  if (outGradient) {
    if constexpr (kGradTarget == GradTarget::Previous) {
      (*outGradient) -= dampingOverDt * (currentPos - stageStartPos);
    } else {
      (*outGradient) += dampingOverDt * (currentPos - stageStartPos);
    }
  }

  if (outHessian) {
    (*outHessian) += dampingOverDt;
  }
}

/**
 * @brief Adds single-degree-of-freedom friction terms.
 *
 * This function computes and accumulates energy, gradient, and Hessian contributions related to
 * friction forces acting on a single degree of freedom. Static friction is regularized using the
 * merit function approximation of Coulomb friction from the Incremental Potential Contact paper.
 * Dynamic friction is attenuated with increasing velocity according to a Gaussian Stribeck model.
 * The computation depends on the gradient target type specified by the template parameter.
 *
 * @tparam kGradTarget The gradient target type. Determines whether derivatives are with respect to
 * the current state (Current) or the previous state (Previous). Possible values:
 * GradTarget::Previous, GradTarget::Current.
 *
 * @param useFittedHessian If true, approximates the Hessian by fitting a global quadratic function.
 * This is more robust than the local Hessian, which cancels out when friction saturates.
 * @param psdDRes If true, clamps negative Hessian contributions.
 * @param frictionParams Per-joint friction parameters including dynamic coefficient, falloff
 * velocity, extra stiction, and Stribeck velocity.
 * @param dtStage The time step duration of the current stage.
 * @param currentPos The current position of the degree of freedom.
 * @param stageStartPos The position at the start of the current stage.
 * @param[out] outEnergy Pointer to a double to accumulate energy contributions. Can be nullptr if
 * not needed.
 * @param[out] outGradient Pointer to a real to accumulate gradient contributions. Can be nullptr if
 * not needed.
 * @param[out] outHessian Pointer to a real to accumulate Hessian contributions. Can be nullptr if
 * not needed.
 */
template <GradTarget kGradTarget>
inline void AddSingleDofFriction(
    bool useFittedHessian,
    bool psdDRes,
    ArticulatedJointFrictionParams const& frictionParams,
    real dtStage,
    real currentPos,
    real stageStartPos,
    double* outEnergy,
    real* outGradient,
    real* outHessian) {
  static_assert(
      kGradTarget == GradTarget::Current || kGradTarget == GradTarget::Previous,
      "Unsupported target");
  MOCHI_ASSERT_VERBOSE(dtStage > 0_r, "Time step must be positive.");
  if (!outEnergy && !outGradient && !outHessian) {
    return;
  }

  real const coulombFriction = frictionParams.coulomb;
  real const falloffVel = frictionParams.falloffVel;
  real const stictionExtra = frictionParams.stictionExtra;
  real const stribeckVel = frictionParams.stribeckVel;

  real const staticFriction = coulombFriction + stictionExtra;
  MOCHI_ASSERT_VERBOSE(staticFriction > 0_r, "Static friction must be strictly positive.");
  real const dSmootherInfty = coulombFriction / staticFriction;

  real vel = (currentPos - stageStartPos) / dtStage;
  real velNorm = Abs(vel);

  real smoother{};
  real dSmoother{};
  real ddSmoother{};
  real dSmoother_velNorm{};
  StribeckActivation(
      velNorm,
      falloffVel,
      dSmootherInfty,
      stribeckVel,
      smoother,
      dSmoother,
      ddSmoother,
      dSmoother_velNorm);

  if (outEnergy) {
    (*outEnergy) += static_cast<double>(staticFriction * dtStage * smoother);
  }

  if (outGradient) {
    // grad = staticFriction * dSmoother * sign(vel). Flip the sign for GradTarget::Previous
    if constexpr (kGradTarget == GradTarget::Previous) {
      (*outGradient) -= staticFriction * dSmoother_velNorm * vel;
    } else {
      (*outGradient) += staticFriction * dSmoother_velNorm * vel;
    }
  }

  if (outHessian) {
    real hessian{};
    if (velNorm < 1e-9_r || useFittedHessian) {
      hessian = staticFriction * dSmoother_velNorm / dtStage;
    } else {
      real const ddSmootherClamped = psdDRes ? Max(ddSmoother, 0_r) : ddSmoother;
      hessian = staticFriction * ddSmootherClamped / dtStage;
    }
    (*outHessian) += hessian;
  }
}

// Function to add single-dof external-force terms.
inline void AddSingleDofExternalForce(real pos, real force, double* outEnergy, real* outGradient) {
  if (outEnergy) {
    (*outEnergy) -= pos * force;
  }

  if (outGradient) {
    (*outGradient) -= force;
  }
}

} // namespace mochi
