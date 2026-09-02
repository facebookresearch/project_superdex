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

/**
 * @brief Call UpdateActorQueriesAsync for each actor in the scene.
 *
 * @remarks This work is normally done during PostStepIslandAsync. Thus you only need to call this
 * function if you want to forcibly recompute the queries without waiting for the next simulation
 * step.
 *
 * @param reg ECS registry
 */
void UpdateAllActorQueries(entt::registry& reg);

} // namespace mochi
