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
#include "mochi_physics/mochi_physics_experimental.h"
#include "mochi_rod_pose.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"

#include <mochi_core/element_operations/element_assembler.h>
#include <mochi_core/element_operations/fem_rod.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/utils/graph.h>

#include <memory>
#include <utility>

namespace mochi {

// Precomputed data for embedding a triangular mesh into a rod's element frames. Each surface node
// is expressed as a weighted blend of affine transformations from nearby rod elements. The affine
// transformation for each (surface node, element) pair uses local coordinates ξ that encode the
// surface node's offset from the element midpoint in the element's *normalized* reference-frame
// basis {unit_reference_tangent, frame_axis, binormal}, where
// unit_reference_tangent = (X1 - X0) / referenceLength has unit length in the reference
// configuration. ξ[0] is therefore a signed arc-length offset and ξ[1], ξ[2] are length-units
// offsets along the frame axis and binormal directions. At runtime, ξ[0] is multiplied by
// invReferenceLengths[elemIdx] when applied to the deformed edge (x1 - x0), to give a
// stretch-aware skinning that stays well-conditioned under stretch and is independent of
// element edge length in the reference configuration.
struct RodSurfaceEmbeddingData {
  int weightsPerNode = 0;
  DynamicArray<int> elementIndices; // flat: numSurfaceNodes × weightsPerNode
  DynamicArray<real> weights; // flat: numSurfaceNodes × weightsPerNode
  DynamicArray<Real3> localCoordinates; // flat: numSurfaceNodes × weightsPerNode (precomputed ξ)
  // Per-element reciprocal reference lengths: invReferenceLengths[e] = 1 / |X1_e - X0_e|.
  // Size = numElements. Used to convert ξ[0] (in length units) into a stretched unit-tangent
  // multiplier at runtime.
  DynamicArray<real> invReferenceLengths;
};

// Forward declarations for ComputeRodSurfaceEmbedding.
class TriangularMesh;
struct SkinningData;

// Precompute a rod surface embedding from a polyline's reference-configuration nodes, element
// frame axes, triangular surface mesh, and skinning data. Returns a fully-constructed
// RodSurfaceEmbeddingData with local coordinates ξ for each (surface node, element) pair.
// The skinning indices/weights are moved into the returned embedding to avoid extra copies.
[[nodiscard]] RodSurfaceEmbeddingData ComputeRodSurfaceEmbedding(
    Span<Real3 const> nodes,
    Span<Real3 const> frameAxes,
    TriangularMesh const& surfaceMesh,
    SkinningData&& skinning,
    bool isClosedLoop);

// ECS component holding the rod-specific visual mesh embedding. This is separate from the
// linear MeshEmbedding used by soft/shell actors, because rod visual node positions depend
// nonlinearly on element frame axes via per-element affine transformations.
struct CRodVisualMeshEmbedding : public NoCopy {
  explicit CRodVisualMeshEmbedding(std::shared_ptr<RodSurfaceEmbeddingData const> dataIn)
      : data(std::move(dataIn)) {
    MOCHI_ASSERT(data != nullptr);
  }
  std::shared_ptr<RodSurfaceEmbeddingData const> data;
};

// Owns the triangular mesh and rod embedding selected for surface contact. These may alias the
// rod shape's visual data or describe a dedicated contact skin.
struct CRodContactSkin : public NoCopy {
  CRodContactSkin(
      std::shared_ptr<TriangularMesh const> meshIn,
      std::shared_ptr<RodSurfaceEmbeddingData const> embeddingIn)
      : mesh(std::move(meshIn)), embedding(std::move(embeddingIn)) {
    MOCHI_ASSERT(mesh != nullptr);
    MOCHI_ASSERT(embedding != nullptr);
  }

  std::shared_ptr<TriangularMesh const> mesh;
  std::shared_ptr<RodSurfaceEmbeddingData const> embedding;
};

// ECS component holding the contact-skin Jacobian ∂x_skin/∂(rod DoFs) as a sparse matrix.
// The matrix has 1 row per skin node and numRodDofs columns. Each non-zero entry is a Real3
// holding (x, y, z) Jacobian components for that (node, DoF) pair.
struct CRodContactSkinningData : public NoCopy {
  SparseMatrix<Real3> jacobian;
};

// Caches deformed contact-skin node positions. The flat array is pre-allocated at actor creation
// to avoid per-frame allocations during contact updates.
struct CRodDeformedContactSkinNodes : public NoCopy {
  DynamicArray<real> positions;
};

// This stores const spans of the rod actor's reference mesh. The underlying data is owned by the
// CShape component.
struct CPolylineMesh : public NoCopy {
  CPolylineMesh(Span<Real3 const> nodesIn, Span<int const> flatConnectivityIn, bool isClosedLoopIn)
      : nodes(nodesIn), flatConnectivity(flatConnectivityIn), isClosedLoop(isClosedLoopIn) {
    MOCHI_ASSERT(isize(nodes) >= 2);
    MOCHI_ASSERT(!isClosedLoop || isize(nodes) >= 3);
    MOCHI_ASSERT(isize(flatConnectivity) == 2 * NumElements(), "Invalid connectivity.");
  }

  int NumElements() const {
    return isClosedLoop ? isize(nodes) : isize(nodes) - 1;
  }

  Int2 ElementNodes(int elemIdx) const {
    return {elemIdx, (elemIdx + 1) % isize(nodes)};
  }

  Span<Real3 const> nodes;
  Span<int const> flatConnectivity;
  bool isClosedLoop = false;
};

// This stores lumped nodal masses for the rod actor, computed from the mesh and linear mass density
// at initialization.
struct CNodalMasses {
  DynamicArray<real> values;
};

// This stores lumped rotational inertias for the rod actor's elements.
struct CElementRotationalInertias {
  DynamicArray<real> values;
};

// This stores reference-configuration cross-section principal axes for the rod actor's elements.
// See the documentation for RodActorParams::elementFrameAxes for additional notes on the physical
// interpretation of these axes.
struct CReferenceElementFrameAxes : public NoCopy {
  // Reference-configuration axes owned by the CShape component.
  CReferenceElementFrameAxes(Span<Real3 const> axesIn) : axes(axesIn) {
    MOCHI_ASSERT(!axes.empty());
  }
  Span<Real3 const> axes;
};

// Per-node reference curvature binormal [1/m]. Computed at initialization from the rest
// configuration. For open polylines, endpoint entries are zero (no curvature defined at boundary
// nodes). For closed-loop polylines, every entry is computed via wraparound neighbors.
struct CReferenceNodeCurvatureBinormal : public NoCopy {
  explicit CReferenceNodeCurvatureBinormal(DynamicArray<Real3>&& valuesIn)
      : values(std::move(valuesIn)) {}
  DynamicArray<Real3> values;
};

struct CRodMaterialParams : public experimental::RodMaterialParams {};

// Generate variable-width rod connectivity and matching stencil positions. Each node gets an
// up-to-3-node stencil [i, i+1, i+2]. Open-rod boundary stencils are shorter, while closed-loop
// stencils wrap around using modular indexing.
[[nodiscard]] std::pair<Graph<int, int>, Graph<int, int>> GenerateRodConnectivityAndStencil(
    int numNodes,
    bool isClosedLoop);

// Generates a discrete Bishop frame for a polyline defined by the given nodes.
// Returns an array of element frame axes (one per edge) computed via parallel transport.
// The first axis is chosen to be well-conditioned (perpendicular to the first edge),
// and subsequent axes are transported using minimal rotations between consecutive tangents.
// For closed-loop polylines, any nonzero holonomy (i.e., angular deviation of the frame axis from
// its initial value caused by parallel transport around a non-planar closed curve) is redistributed
// evenly with respect to arc length, minimizing peak rate of twist.
[[nodiscard]] DynamicArray<Real3> GenerateDiscreteBishopFrame(
    Span<Real3 const> nodes,
    bool isClosedLoop);

// Initialize a rod actor in the ECS registry from its actor params and shape.
void InitRodActor(
    entt::registry& reg,
    entt::entity e,
    experimental::RodActorParams const& params,
    std::shared_ptr<PolylineShape const> shapePtr,
    Error& error);

/**************************************************************************
  ECS Components for Rod Actors
*/

// CRodPose<kTimeStep> holds the combined displacement+twist+axes state at each time level.
// CIntegrationRodPoses is the single integration bundle for multi-step/multi-stage methods.
// CIntegrationVelocitySlices handles velocity integration independently.

namespace rod {

/**************************************************************************
  ECS Systems for Rod Actors
*/

/// @brief Compose all batched rod element operations into a single functor for the FEM assembler.
/// Performs a single gather/scatter per element, combining gravity, inertia, and stress in one
/// pass.
[[nodiscard]] ElOpFnType<fem::RodStencilElement, fem::kNumRodFields> MakeBatchedBodyOp(
    Span<Real3 const> meshNodes,
    bool hasGravity,
    Span<real const> nodalMasses,
    Span<real const> elementRotationalInertias,
    Span<Real3 const> frameAxes,
    Span<Real3 const> referenceAxes,
    experimental::RodMaterialParams const& materialParams,
    Real3 gravity,
    Span<real const> stageStartDispl,
    Span<real const> stageStartVel,
    Span<Real3 const> stageStartFrameAxes,
    real dtStage);

void AssembleBody(
    AssemblyParams const& params,
    ecs::Included<TagRodActor>,
    ecs::CtxGlobal<CSceneGravity const> sceneGravity,
    ecs::OptionalTag<TagUseGravity> hasGravityTag,
    CLocal2GlobalMap const& l2g,
    CPolylineMesh const& polylineMesh,
    CRodPose<TimeStep::Current> const& currPose,
    CReferenceElementFrameAxes const& referenceAxes,
    CNodalMasses const& nodalMasses,
    CElementRotationalInertias const& elementRotationalInertias,
    CRootTransform const& rootTransform,
    CRodMaterialParams const& materialParams,
    CTimeIntegratorState const& intState,
    CRodPose<TimeStep::StageStart> const& stageStartPose,
    CVelocitySlice<real, TimeStep::StageStart> const& stageStartVel,
    CExternalForces const& externalForces,
    CActorSnle& outActorSnle,
    CNodalBasedStructure const& nbs);

/**************************************************************************
  ECS Systems for Rod Actors
*/

/*
 * System to serialize the rod actor's pose (displacement-twist DoFs + element frame axes)
 * into a column vector. The column vector represents the pose-sized portion of the
 * non-linear problem solution vector corresponding to the actor.
 */
void EntityGetSolution(
    ColumnVectorView<real> outSolution,
    ecs::Included<TagRodActor>,
    CRodPose<TimeStep::Current> const& currPose);

/*
 * System to deserialize a column vector into the rod actor's displacement-twist slice
 * and element frame axes. The column vector contains the pose-sized portion of the
 * non-linear problem solution vector corresponding to the actor.
 */
void EntitySetSolution(
    ColumnVectorView<real const> solution,
    ecs::Included<TagRodActor>,
    CRodPose<TimeStep::Current>& currPose);

/*
 * System executed after the solution of the non-linear problem is updated.
 * Extracts displacement-twist DoFs and element frame axes from the global solution.
 */
void EntityPostNewSolution(
    ColumnVectorView<real const> solution,
    ecs::Included<TagRodActor>,
    CDofOffset const& dofOffset,
    CActorDofInfo const& actorDofInfo,
    CRodPose<TimeStep::Current>& currPose);

/*
 * System to update the rod actor's displacement-twist slice from a reference solution
 * plus an increment. This is called after each Newton iteration to update the actor's
 * internal state with the new increment.
 */
void EntityPostNewIncrement(
    ColumnVectorView<real const> reference,
    ColumnVectorView<real const> increment,
    ecs::Included<TagRodActor>,
    CDofOffset const& dofOffset,
    CActorDofInfo const& actorDofInfo,
    CPolylineMesh const& mesh,
    CRodPose<TimeStep::Current>& currPose);

/*
 * System executed before the time step. Saves current state to previous and resets velocity.
 */
void EntityIncrementStep(
    ecs::Included<TagRodActor>,
    CRodPose<TimeStep::Current>& currPose,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CVelocitySlice<real, TimeStep::Previous>& prevVel,
    CRodPose<TimeStep::Previous>& prevPose,
    CIntegrationRodPoses& intPoses);

/*
 * System executed before the first time integration stage of the time step.
 * Computes the differential variables at the beginning of the step.
 */
void EntityPreFirstStage(
    ecs::Included<TagRodActor>,
    CTimeIntegratorState const& intState,
    CPolylineMesh const& mesh,
    CVelocitySlice<real, TimeStep::Previous> const& prevVel,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels,
    CRodPose<TimeStep::Previous> const& prevPose,
    CIntegrationRodPoses& intPoses);

/*
 * System executed before each time integration stage.
 * Computes the differential variables at the beginning of the stage.
 */
void EntityPreStage(
    ecs::Included<TagRodActor>,
    CTimeIntegratorState const& intState,
    CPolylineMesh const& mesh,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels,
    CVelocitySlice<real, TimeStep::StageStart>& stageStartVel,
    CIntegrationRodPoses& intPoses,
    CRodPose<TimeStep::StageStart>& stageStartPose);

/*
 * System executed after each time integration stage.
 * Updates velocity via finite differences and pushes to integration vectors.
 */
void EntityPostStage(
    ecs::Included<TagRodActor>,
    CConvergenceStatus const& convergence,
    CTimeIntegratorState const& intState,
    CRodPose<TimeStep::StageStart> const& stageStartPose,
    CRodPose<TimeStep::Current>& currPose,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels,
    CIntegrationRodPoses& intPoses,
    CReferenceElementFrameAxes const& referenceAxes);

/*
 * System executed after the last time integration stage.
 * Computes the differential variables at the end of the time step.
 */
void EntityPostLastStage(
    ecs::Included<TagRodActor>,
    CTimeIntegratorState const& intState,
    CPolylineMesh const& mesh,
    CRodPose<TimeStep::Current>& currPose,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels,
    CIntegrationRodPoses& intPoses);

void AssembleAsyncContact(
    AssemblyParams const& params,
    entt::entity e,
    ecs::Included<TagRodActor, TagUseContact>,
    ecs::OptionalTag<TagQueryActiveContacts> queryActiveContacts,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CTimeIntegratorState const& intState,
    ContactAssemblyReg reg,
    CFemSegmentDiscretization const& femDisc,
    CContactLocal2GlobalMap const& contactL2g,
    CContactNodalBasedStructure const& contactNbs,
    CContactSamples<TimeStep::Current> const& samples,
    CColliderInfo const& colliderInfo,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& collisions,
    CRootTransform const& rootTransform,
    CDeformablePointAsyncCollisionsResponse& outResponse,
    CActorSnle& outActorSnle);

void InitializeOnce(entt::registry& reg);

// Compute deformed visual surface node positions (and optionally normals) for a rod actor.
// Uses the rod-specific affine embedding rather than the linear MeshEmbedding used by soft/shell.
void UpdateQueryVisualNodePositionsAndNormals(
    CVisualMesh const& visualMesh,
    CRodVisualMeshEmbedding const& rodEmbedding,
    CPolylineMesh const& polylineMesh,
    CRodPose<TimeStep::Current> const& rodPose,
    CQueryVisualNodePositions& outVisPosQuery,
    CQueryVisualNodeNormals* outVisNormQuery);

// Builds the CSR sparsity pattern of the contact-skin Jacobian ∂x_skin/∂(rod DoFs). The sparsity
// depends only on topology-invariant embedding data, so this runs once during actor setup. The
// resulting matrix has the correct structure and zero values.
void InitializeContactSkinningJacobian(
    CRodContactSkin const& contactSkin,
    CPolylineMesh const& polylineMesh,
    CRodContactSkinningData& outSkinning);

// Computes contact-skin Jacobian values using the current rod frame axes. The sparsity pattern must
// already be initialized by InitializeContactSkinningJacobian.
void ResolveContactSkinningJacobian(
    CRodContactSkin const& contactSkin,
    CPolylineMesh const& polylineMesh,
    CRodPose<TimeStep::Current> const& rodPose,
    CRodContactSkinningData& outSkinning);

// Updates surface-contact samples from the selected triangular mesh. Deformed surface-node
// positions are computed into the pre-allocated buffer before evaluating the quadrature points.
template <TimeStep kTimeStep>
void UpdateSurfaceContactPositions(
    ecs::Included<TagRodActor>,
    ecs::RequiredTag<TagRodSurfaceContact>,
    CRodPose<kTimeStep> const& rodPose,
    CRodContactSkin const& contactSkin,
    CPolylineMesh const& polylineMesh,
    CFemSurfaceDiscretization const& surfaceDisc,
    CRodDeformedContactSkinNodes& deformedNodes,
    CContactSamples<kTimeStep>& outSamples);

// Sets up surface-contact colliding Jacobians through
// DMap<DQuad, DMapRTConst, DMapSparseSkinning>.
void SetupSurfaceCollidingJacobians(
    ecs::Included<TagRodActor>,
    ecs::RequiredTag<TagRodSurfaceContact>,
    CFemSurfaceDiscretization const& surfaceDisc,
    CRootTransform const& transform,
    CDofOffset const& dofOffset,
    CRodContactSkinningData const& skinningData,
    CCollJacs<CollRole::Colliding>& outJacobians);

// Updates the bounding volume from deformed surface-contact positions, reusing the pre-allocated
// deformed-node buffer to avoid per-frame allocations.
template <TimeStep kStep>
void UpdateSurfaceContactBounds(
    ecs::Included<TagRodActor>,
    ecs::RequiredTag<TagRodSurfaceContact>,
    CRodContactSkin const& contactSkin,
    CPolylineMesh const& polylineMesh,
    CRodPose<kStep> const& rodPose,
    CRodDeformedContactSkinNodes& deformedNodes,
    CBoundingVolume<TimeStep::Current>& outBounds);

// Get the mass of a rod actor.
[[nodiscard]] real GetActorMass(entt::registry const& reg, entt::entity actor);

// Compute per-node curvature binormal vectors [1/m] for a rod centerline polyline.
//
// For open polylines, the endpoints have no defined curvature and the corresponding entries are
// set to 0. For closed-loop polylines, every node has a well-defined curvature, computed via
// wraparound neighbors.
//
// @param[in]  nodes                  Node positions [m], size N (>=2 for open, >=3 for closed).
// @param[in]  isClosedLoop           Whether the polyline wraps around (last node connects to
// first).
// @param[out] outCurvatureBinormals  Output buffer for per-node curvature binormals, size N.
void ComputeRodNodeCurvatureBinormals(
    Span<Real3 const> nodes,
    bool isClosedLoop,
    Span<Real3> outCurvatureBinormals);

// Update CBoundingVolume<TimeStep::Current>.localShape based on the deformation of the rod. kStep
// defines the data to be used in the update, not the component storing the result. There's no
// CBoundingVolume<TimeStep::StageStart>, as it's not needed. We do bound checks in stage-start
// collision detection, but we can use CBoundingVolume<TimeStep::Current> for this.
// Note: Rods have 4 DoFs per node (3 displacement + 1 twist), so we extract just the displacement
// components (stride of 4) to compute the bounding volume.
// Excluded<CFemSurfaceDiscretization> ensures this only runs for centerline contact rods;
// contact-skin rods use UpdateSurfaceContactBounds instead.
template <TimeStep kStep>
void UpdateBounds(
    ecs::Excluded<CFemSurfaceDiscretization>,
    CPolylineMesh const& mesh,
    CFinalDisplacementRef<kStep> const& solComponent,
    CPointCloudColliderParams const* pointCloudColliderParams,
    CBoundingVolume<TimeStep::Current>& outBounds) {
  static_assert(kStep == TimeStep::Current || kStep == TimeStep::StageStart);
  MOCHI_PROFILE_SCOPE();
  auto const& sol = solComponent.value;
  int const numNodes = isize(mesh.nodes);

  // Compute AABB from deformed node positions
  // Rod DoFs are laid out as [dx0, dy0, dz0, twist0, dx1, dy1, dz1, twist1, ...]
  Vec4r min = ToSimd(mesh.nodes[0], 0_r) + Load<Vec4r>(&sol[0]);
  Vec4r max = min;
  for (int i = 1; i < numNodes; ++i) {
    int const offset = i * fem::kNumRodFields;
    Vec4r const pos = ToSimd(mesh.nodes[i], 0_r) + Load<Vec4r>(&sol[offset]);
    min = Min(min, pos);
    max = Max(max, pos);
  }
  Obb bounds = GetObb(Aabb{Set(min, 3, 0_r), Set(max, 3, 0_r)});
  if (pointCloudColliderParams) {
    bounds = ExpandShape(bounds, pointCloudColliderParams->radius);
  }
  outBounds.localShape = bounds;
}

} // namespace rod

/*
 * System to prepare rod actor for simulation step.
 * Updates boundary conditions from CDofPositionsBC to CDirichletBC.
 */
void PreStepRodActorAsync(entt::registry& reg, entt::entity e);

} // namespace mochi
