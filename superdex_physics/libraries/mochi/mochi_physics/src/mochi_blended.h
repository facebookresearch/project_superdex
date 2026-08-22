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

// Actor composition of a blended actor
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

// Method to initialize a blended actor.
void InitBlendedActor(
    entt::registry& reg,
    entt::entity e,
    ArticulatedActorParams const& params,
    Span<ActorHandle const> softHandles,
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
 * Pipeline to update quantities that are a function of the state (aka derived state) of the
 * blended actor and make them consistent with the state. Must be called after
 * skinned::UpdateDerivedStatePipeline()
 */
void UpdateDerivedStatePipeline(entt::registry& reg, Span<entt::entity const> entities);

/*
 * Pipeline to update Jacobians for contact. Must be called after
 * skinned::UpdateJacobiansPipeline()
 */
void UpdateJacobiansPipeline(entt::registry& reg, Span<entt::entity const> entities);

/*
 * System to compute contact Jacobians as colliding actor. It is called after the collision
 * detection pipeline, within each nonlinear solver iteration.
 */
template <typename DiscretizationT>
void SetupCollidingJacobians(
    ecs::Included<TagBlendedActor>,
    ecs::PartialRegistry<CDofOffset const, rom::CRomJacobian const> reg,
    CBlendedComposition const& composition,
    CDofOffset const& dofOffset,
    DiscretizationT const& discretization,
    CBlendingData const& blendingData,
    CContactPartitions const& contactPartitions,
    CArticulatedSkinningData const& skinningInfo,
    CArticulatedLinkTransforms<TimeStep::Current> const& linkTransforms,
    CCollJacs<CollRole::Colliding>& outJacobians);

void InitializeOnce(entt::registry& reg);

} // namespace blended

} // namespace mochi
