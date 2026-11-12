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

#include "mochi_ecs.h"
#include "mochi_island.h"

#include <mochi_physics/utils/mochi_physics_macros.h>

namespace mochi {

/**************************************************************************
  ECS Simulation Step
*/

// Recompute the maximum world-space geometry speed for every dynamic geometry owner. Current rigid
// velocities must have clean vsym; PreStepEcs prepares it before calling this function.
void UpdateMaxGeometrySpeeds(entt::registry& reg);

/**
 * @brief Runs the scene-wide pre-step pipeline to prepare all scene entities for the next
 * simulation step.
 *
 * @note @ref CSceneTime must have a valid state at input (via @ref CSceneTime::Advance() or @ref
 * CSceneTime::Reset()) such that @ref CSceneTime::DeltaTimePrev() > 0 and @ref
 * CSceneTime::DeltaTime() > 0.
 *
 * @see CSceneTime
 */
void PreStepEcs(entt::registry& reg);

void StepEcs(entt::registry& reg);
void PostStepEcs(entt::registry& reg);

void PreStepIslandAsync(entt::registry& reg, CIslandDescendants const& descendants);

// Save the contact queries of an island's actors, once per step for each value of farSdfActors:
// - false: after the final stage solve, before step-end finalization (last assembly stage).
// - true: after the final far collision detection. Far-SDF actors get only SDF distances.
void UpdateIslandContactQueries(
    entt::registry& reg,
    CIslandDescendants const& descendants,
    bool farSdfActors);

/**
 * @brief Recompute scene queries, optionally including contact-derived queries.
 *
 * @remarks During a step, queries are updated by UpdateIslandContactQueries and, at step end,
 * PostStepIslandAsync. This function refreshes the same queries from Current. With
 * @p includeAssemblyQueries, the caller must provide freshly assembled, coherent Current contact
 * data.
 *
 * @param reg ECS registry
 * @param includeAssemblyQueries If false, queries that require contact detection or force assembly
 * (see QueryRequiresAssembly) are left unchanged.
 */
void UpdateAllActorQueries(entt::registry& reg, bool includeAssemblyQueries);

namespace step {
// Call once on startup, after all other modules' InitializeOnce.
void InitializeOnce(entt::registry& reg);
} // namespace step

} // namespace mochi
