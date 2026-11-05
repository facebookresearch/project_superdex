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

#include "mochi_common_components.h"
#include "mochi_contact.h"
#include "mochi_discretization_components.h"
#include "mochi_ecs.h"
#include "mochi_simulation.h"

#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/utils/mesh_embedding.h>
#include <mochi_core/utils/span.h>

// Systems for triangular contact skins whose nodes are linear combinations of physics-mesh nodes.
// Shared by soft and shell actors; rods use their own contact-skin embedding.
namespace mochi::linear_contact_skin {

// Emplaces the skinning Jacobian, deformed skin nodes, and skinned-contact components. The caller
// emplaces the contact skin's surface discretization first.
void EmplaceComponents(
    entt::registry& reg,
    entt::entity e,
    CFemSurfaceDiscretization const& surfaceDisc,
    TriangularMesh const& contactSkin,
    LinearMeshEmbedding const& embedding,
    Span<Real3 const> physicsNodeCoordinates);

// Builds the compact element-local map of a linear skinning Jacobian, which must be W tensor I_3.
void InitializeElementMap(
    CFemSurfaceDiscretization const& surfaceDisc,
    CContactSkinningData const& skinningData,
    CLinearContactSkinElementMap& outMap);

// Interpolates contact samples from the deformed skin nodes, which must already be up to date.
template <TimeStep kTimeStep>
void UpdateSamples(
    ecs::Excluded<TagRodActor>,
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    CFemSurfaceDiscretization const& surfaceDisc,
    CDeformedContactSkinNodes const& deformedNodes,
    CContactSamples<kTimeStep>& outSamples);

// Updates the deformed positions of the active contact-skin nodes, then runs @ref UpdateSamples.
template <TimeStep kTimeStep>
void UpdatePositionsAndSamples(
    ecs::Excluded<TagRodActor>,
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    CContactSkinMesh const& contactSkin,
    CFinalDisplacementRef<kTimeStep> const& displacements,
    CFemSurfaceDiscretization const& surfaceDisc,
    CDeformedContactSkinNodes& deformedNodes,
    CContactSamples<kTimeStep>& outSamples);

// Sets up colliding normals from the deformed contact-skin triangles. Shells are excluded because
// shell contact has no colliding normals. Current normals read the deformed skin node positions,
// which must be up to date (see @ref UpdateBounds).
template <bool kIsSync>
void SetupActiveCollisionNormals(
    ecs::Excluded<TagShellActor, TagRodActor>,
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CContactSkinMesh const& contactSkin,
    CDeformedContactSkinNodes const& deformedNodes,
    CFemSurfaceDiscretization const& surfaceDisc,
    CFinalDisplacementRef<TimeStep::StageStart> const& stageStartDispl,
    CRootTransform const& transform,
    CActiveCollisions<kIsSync, TimeStep::Current>& activeCollisions);

// Updates CBoundingVolume.localShape to cover the deformed contact skin, plus the displaced physics
// mesh when a collider is enabled (expanded by the point-cloud radius, if any). Also updates the
// deformed skin node positions, which later @ref UpdateSamples and @ref
// SetupActiveCollisionNormals calls rely on.
template <TimeStep kTimeStep>
void UpdateBounds(
    ecs::Excluded<TagRodActor>,
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    ecs::OptionalTag<TagSoftActor> isSoft,
    CSimplicialMesh const& physicsMesh,
    CContactSkinMesh const& contactSkin,
    CFinalDisplacementRef<kTimeStep> const& displacements,
    CColliderInfo const& collider,
    CPointCloudColliderParams const* pointCloudColliderParams,
    CDeformedContactSkinNodes& deformedNodes,
    CBoundingVolume& outBounds);

// Assembles the contact response of pointRange through the linear element map, into each output
// that results provides. Given the collider Jacobians of sync contact, it also assembles the
// skin-collider coupling blocks of the dresidual, but not the collider terms themselves.
void AssembleContactResponseRange(
    ContactDetectionResult const& query,
    CLinearContactSkinElementMap const& elementMap,
    CFemSurfaceDiscretization const& surfaceDisc,
    CRootTransform const& rootTransform,
    Span<real const> sampleWeights,
    int dofOffset,
    Interval<int> pointRange,
    CollisionResponseResult const& response,
    Span<ContactJac const*> colliderJacs,
    Allocator* allocator,
    AssemblyResults<real> results);

namespace details {
// Returns the contact indices of pointRange sorted by (sample index, contact index), so that the
// contacts of each element are contiguous.
DynamicArray<int> SortContactsBySample(
    Interval<int> pointRange,
    ContactDetectionResult const& query,
    Allocator* allocator);
} // namespace details

} // namespace mochi::linear_contact_skin
