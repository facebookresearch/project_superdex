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

#include "mochi_soft_skinned.h"

#include "mochi_physics/cpp_api/mochi_structs.h"

#include <memory>
#include <vector>

namespace mochi {

// Nested soft actors driven by an articulated actor, which owns them: destroying it destroys them.
// Present even when there is no blended surface.
struct CBlendedComposition : public NoCopy {
  std::vector<entt::entity> soft;
  std::vector<ActorHandle> softHandles;
};

// Blending data corresponding to all soft actors
struct CBlendingData : public std::vector<BlendingDataSourceMesh> {};

// Active nodes per soft actor
struct CBlendedActiveNodes : public std::vector<std::vector<int>> {};

/*
Systems for a blended body
*/
namespace blended {

// Attach soft actors to an articulated actor as its nested soft actors, to be initialized next by
// skinned::InitSkinnedActor.
void AttachNestedSoftActors(
    entt::registry& reg,
    entt::entity e,
    Span<ActorHandle const> softHandles);

// Initialize a blended actor after skinned::InitSkinnedActor on each of its nested soft actors.
void InitBlendedActor(
    entt::registry& reg,
    entt::entity e,
    ArticulatedActorParams const& params,
    std::shared_ptr<Shape const> blendedShapePtr,
    Error& error);

/*
 * Pipeline executed before each time integration stage.
 */
void PreStagePipeline(entt::registry& reg, Span<entt::entity const> entities);

/*
 * Pipeline executed after the last time integration stage of the time step. It must be called after
 * skinned::PostLastStagePipeline.
 */
void PostLastStagePipeline(entt::registry& reg, Span<entt::entity const> entities);

/*
 * Pipeline to resolve current blending displacements for all nodes, including inactive nodes when
 * subsampling is enabled.
 */
void ResolveAllNodeBlendingDisplacementsPipeline(
    entt::registry& reg,
    Span<entt::entity const> entities);

// Blend world-space nested soft velocities into world-space skinning velocity.
void UpdateBlendingVelocity(
    ecs::PartialRegistry<CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned> const>
        reg,
    CBlendedComposition const& composition,
    CBlendingData const& blendingData,
    CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned>& inOutVelocity);

/*
 * Pipeline to update quantities that are a function of the state (aka derived state) of the
 * blended actor and make them consistent with the state. Must be called after
 * skinned::UpdateDerivedStatePipeline()
 */
void UpdateDerivedStatePipeline(entt::registry& reg, Span<entt::entity const> entities);

void InitializeOnce(entt::registry& reg);

} // namespace blended

} // namespace mochi
