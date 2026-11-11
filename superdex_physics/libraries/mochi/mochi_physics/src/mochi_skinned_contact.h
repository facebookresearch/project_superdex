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

#include "mochi_blended.h"
#include "mochi_contact.h"
#include "mochi_soft_skinned.h"

/*
Contact Jacobians of skinned surfaces: articulated skins, soft-skinned actors and blended actors.
*/
namespace mochi {

namespace articulated::compound {

/*
 * System to compute contact Jacobians as colliding actor. It is called after the collision
 * detection pipeline, within each nonlinear solver iteration.
 */
template <typename DiscretizationT>
void SetupCollidingJacobians(
    ecs::Included<TagArticulatedActor>,
    ecs::Excluded<TagBlendedActor>,
    DiscretizationT const& discretization,
    CDofOffset const& dofOffset,
    CArticulatedJacobian const& jacobian,
    CArticulatedLinkTransforms<TimeStep::Current> const& linkTransforms,
    CArticulatedSkinningData const& skinningData,
    CContactPartitions const& contactPartitions,
    CCollJacs<CollRole::Colliding>& outJacobians);

} // namespace articulated::compound

namespace skinned {

/*
 * System to compute contact Jacobians as colliding actor. It is called after the collision
 * detection pipeline, within each nonlinear solver iteration.
 */
void SetupCollidingJacobians(
    ecs::Included<TagNestedSoftActor>,
    ecs::PartialRegistry<
        CDofOffset const,
        CArticulatedLinkTransforms<TimeStep::Current> const,
        CArticulatedJacobian const> reg,
    CSkinnedComposition const& composition,
    CDofOffset const& dofOffset,
    CContactPartitions const& contactPartitions,
    CFemBoundaryDiscretization const& discretization,
    CNodePositions const& positions,
    CArticulatedSkinningData const& skinningInfo,
    rom::CRomJacobian const* romJacobian,
    CCollJacs<CollRole::Colliding>& outJacobians);

} // namespace skinned

namespace blended {

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
    CArticulatedJacobian const& jacobian,
    DiscretizationT const& discretization,
    CBlendingData const& blendingData,
    CContactPartitions const& contactPartitions,
    CNodePositions const& positions,
    CArticulatedSkinningData const& skinningInfo,
    CArticulatedLinkTransforms<TimeStep::Current> const& linkTransforms,
    CCollJacs<CollRole::Colliding>& outJacobians);

} // namespace blended

} // namespace mochi
