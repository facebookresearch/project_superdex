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

#include <mochi_core/utils/basic_utils.h>
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

// Function to add single-dof external-force terms. The energy is relative to the stage start.
inline void AddSingleDofExternalForce(
    real currentPos,
    real stageStartPos,
    real force,
    double* outEnergy,
    real* outGradient) {
  if (outEnergy) {
    (*outEnergy) -= (currentPos - stageStartPos) * force;
  }

  if (outGradient) {
    (*outGradient) -= force;
  }
}

} // namespace mochi
