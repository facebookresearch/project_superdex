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

#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>

namespace mochi {

class TransmissionActuator;

namespace transmission {

// Add a linear transmission to an articulated actor
int AddLinearTransmission(
    entt::registry& reg,
    entt::entity e,
    experimental::LinearTransmissionParams const& params,
    Error& error);

// Add a spatial tendon (routed through link-local waypoints) to an articulated actor
int AddSpatialTendon(
    entt::registry& reg,
    entt::entity e,
    experimental::SpatialTendonParams const& params,
    Error& error);

// Attach different types of actuators to a given transmission
void AttachDisplacementControlActuator(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    experimental::DisplacementControlActuatorParams const& params,
    Error& error);

void AttachForceControlActuator(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    experimental::ForceControlActuatorParams const& params,
    Error& error);

void AttachMcKibbenActuator(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    experimental::McKibbenActuatorParams const& params,
    Error& error);

void SetTransmissionActuatorStateVariables(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    Span<real const> stateVariables,
    Error& error);

[[nodiscard]] real GetTransmissionDisplacement(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    Error& error);

void GetTransmissionDisplacementJacobian(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    Span<real> outJacobian,
    Error& error);

void GetTransmissionActuatorStateVariables(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    Span<real> outStateVariables,
    Error& error);

int GetNumTransmissionActuatorStateVariables(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    Error& error);
} // namespace transmission
} // namespace mochi
