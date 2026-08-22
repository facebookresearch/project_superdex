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

#include "mochi_attributes.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"

#include <mochi_core/linear_algebra/matrix.h>

#include <entt/entity/registry.hpp>

namespace mochi {

/**
 * @brief Per-DoF convergence weights for per-actor weighted L2 residual norm.
 *
 * @note Emplaced on all actors that have @ref CActorSnle.
 * @note Lifecycle
 * - Emplacement: @c reg.emplace<CActorConvergenceWeights>(e) at actor creation.
 * - Per-step invalidation: @ref InvalidateConfigDependentActorConvergenceWeights at the start of
 *   each step for actors whose weights are configuration-dependent.
 * - Weight computation: @ref UpdateActorConvergenceWeights during the first Newton assembly of the
 *   step, after all dependencies have been refreshed by the assembly pipeline. Only updated if @ref
 *   isValid is false.
 * - Adhoc invalidation: @ref InvalidateActorConvergenceWeights. Used when mass or material
 *   properties change or actor type changes (e.g. FOM/ROM switch).
 */
struct CActorConvergenceWeights : NoCopy {
  ColumnVector<real> values;
  bool isValid = false;

  MOCHI_STRUCT_BEGIN(mochi::CActorConvergenceWeights);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_FIELD(values);
  MOCHI_FIELD(isValid);
  MOCHI_STRUCT_END();
};

/**
 * @brief Recomputes convergence weights if the actor is participating in the solver (@ref
 * CActorSnle::useInSolver == true) and the weights are stale (@ref
 * CActorConvergenceWeights::isValid == false). No-op otherwise.
 */
void UpdateActorConvergenceWeights(
    entt::registry const& reg,
    entt::entity actor,
    CActorSnle const& actorSnle,
    CActorConvergenceWeights& outWeights);

/** @brief Invalidates convergence weights. Called e.g. when mass or material properties change. */
inline void InvalidateActorConvergenceWeights(entt::registry& reg, entt::entity actor) {
  if (auto* weights = reg.try_get<CActorConvergenceWeights>(actor)) {
    weights->isValid = false;
  }
}

/** @brief Invalidates weights for configuration-dependent actors. */
inline void InvalidateConfigDependentActorConvergenceWeights(
    entt::registry& reg,
    entt::entity actor) {
  if (reg.any_of<TagArticulatedActor, TagRigidActor, TagRomActor, TagRodActor>(actor)) {
    // - Articulated actors have configuration-dependent Jacobian and generalized masses.
    // - Rigid actors have rotation-dependent world-frame MOI.
    // - ROM actors may have configuration-dependent Jacobian and generalized masses.
    // - Rod actors have configuration-dependent curvature.
    //
    // Performance notes:
    // - Weights could be invalidated every N > 1 steps to improve performance. If so, please ensure
    //   deterministic capture/restore state is preserved.
    // - ROMs with constant Jacobian (e.g. linear, non-adaptive ROMs with fixed rigid transform)
    //   could skip weight update.
    InvalidateActorConvergenceWeights(reg, actor);
  }
}

} // namespace mochi
