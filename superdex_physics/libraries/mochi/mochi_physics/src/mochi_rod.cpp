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

#include "mochi_rod.h"

#include "mochi_actor_convergence.h"
#include "mochi_contact_filter.h"
#include "mochi_deformable.h"
#include "mochi_discretization_functions.h"
#include "mochi_integration.h"
#include "mochi_island.h"
#include "mochi_point_cloud_contact.h"
#include "mochi_rod_pose.h"
#include "mochi_shape.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"

#include <mochi_core/contact/dmap.h>
#include <mochi_core/element_operations/element_assembler.h>
#include <mochi_core/element_operations/fem_rod.h>
#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/geometry/mesh_data_utils.h>
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/linear_algebra/any_matrix.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/assembly.h>
#include <mochi_core/utils/graph_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/vmatrix.h>

#include <algorithm>
#include <optional>

using namespace mochi;

// Initialize an (up-to-)3-node stencil starting at each node.
std::pair<Graph<int, int>, Graph<int, int>> mochi::GenerateRodConnectivityAndStencil(
    int numNodes,
    bool isClosedLoop) {
  int constexpr kNumStencilNodes = fem::kNumRodStencilNodes;
  int const maxConnectivitySize = kNumStencilNodes * numNodes;
  GraphBuilder<int, int> connectivityBuilder(numNodes, maxConnectivitySize);
  GraphBuilder<int, int> stencilBuilder(numNodes, maxConnectivitySize);
  for (int i = 0; i < numNodes; i++) {
    DynamicArray<int> connectivityRow;
    connectivityRow.reserve(kNumStencilNodes);
    DynamicArray<int> stencilRow;
    stencilRow.reserve(kNumStencilNodes);

    int const maxStep = isClosedLoop ? 2 : Min(2, numNodes - 1 - i);
    for (int s = 0; s <= maxStep; ++s) {
      connectivityRow.push_back((i + s) % numNodes);
      stencilRow.push_back(s);
    }

    connectivityBuilder.append(connectivityRow);
    stencilBuilder.append(stencilRow);
  }
  return {connectivityBuilder.Build(), stencilBuilder.Build()};
}

namespace mochi::rod {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CPolylineMesh>(reg);
  ecs::RegisterComponent<CNodalMasses>(reg);
  ecs::RegisterComponent<CElementRotationalInertias>(reg);
  ecs::RegisterComponent<CReferenceElementFrameAxes>(reg);
  ecs::RegisterComponent<CReferenceNodeCurvatureBinormal>(reg);
  ecs::RegisterComponent<CRodMaterialParams>(reg);
  ecs::RegisterComponent<CIntegrationRodPoses>(reg);
  ecs::RegisterComponent<CRodPose<TimeStep::Current>>(reg);
  ecs::RegisterComponent<CRodPose<TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CRodPose<TimeStep::Previous>>(reg);
  ecs::RegisterComponent<CRodVisualMeshEmbedding>(reg);
  ecs::RegisterComponent<CRodContactSkin>(reg);
  ecs::RegisterComponent<CRodContactSkinningData>(reg);
  ecs::RegisterComponent<CRodDeformedContactSkinNodes>(reg);
  ecs::RegisterComponent<TagRodSurfaceContact>(reg);
}

// Serializes frame axes to the packed pose vector layout [displacement_twist | axes].
static void SerializeFrameAxes(
    Span<Real3 const> frameAxes,
    int axesOffset,
    ColumnVectorView<real> outSolution) {
  int const numElements = isize(frameAxes);
  MOCHI_ASSERT_VERBOSE(
      isize(outSolution) >= axesOffset + 3 * numElements,
      "Size mismatch in frame axis serialization.");
  for (int i = 0; i < numElements; ++i) {
    int const base = axesOffset + 3 * i;
    outSolution[base + 0] = frameAxes[i][0];
    outSolution[base + 1] = frameAxes[i][1];
    outSolution[base + 2] = frameAxes[i][2];
  }
}

// Deserializes frame axes from the packed pose vector layout [displacement_twist | axes].
static void DeserializeFrameAxes(
    ColumnVectorView<real const> solution,
    int axesOffset,
    Span<Real3> outFrameAxes) {
  int const numElements = isize(outFrameAxes);
  MOCHI_ASSERT_VERBOSE(
      isize(solution) >= axesOffset + 3 * numElements,
      "Size mismatch in frame axis deserialization.");
  for (int i = 0; i < numElements; ++i) {
    int const base = axesOffset + 3 * i;
    outFrameAxes[i] = Real3{solution[base], solution[base + 1], solution[base + 2]};
  }
}

void EntityGetSolution(
    ColumnVectorView<real> outSolution,
    ecs::Included<TagRodActor>,
    CRodPose<TimeStep::Current> const& currPose) {
  MOCHI_PROFILE_SCOPE();
  int const dofsSize = isize(currPose.value.displacements);
  outSolution.MiddleRows(0, dofsSize) = currPose.value.displacements;
  SerializeFrameAxes(MakeConstSpan(currPose.value.frameAxes), dofsSize, outSolution);
}

void EntitySetSolution(
    ColumnVectorView<real const> solution,
    ecs::Included<TagRodActor>,
    CRodPose<TimeStep::Current>& currPose) {
  MOCHI_PROFILE_SCOPE();
  int const dofsSize = isize(currPose.value.displacements);
  currPose.value.displacements = solution.MiddleRows(0, dofsSize);
  DeserializeFrameAxes(solution, dofsSize, MakeSpan(currPose.value.frameAxes));
}

void EntityPostNewSolution(
    ColumnVectorView<real const> solution,
    ecs::Included<TagRodActor>,
    CDofOffset const& dofOffset,
    CActorDofInfo const& actorDofInfo,
    CRodPose<TimeStep::Current>& currPose) {
  MOCHI_PROFILE_SCOPE();
  currPose.value.displacements = solution.MiddleRows(dofOffset.poseOffset, actorDofInfo.dofsSize);
  int const axesOffset = dofOffset.poseOffset + actorDofInfo.dofsSize;
  DeserializeFrameAxes(solution, axesOffset, MakeSpan(currPose.value.frameAxes));
}

void EntityPostNewIncrement(
    ColumnVectorView<real const> reference,
    ColumnVectorView<real const> increment,
    ecs::Included<TagRodActor>,
    CDofOffset const& dofOffset,
    CActorDofInfo const& actorDofInfo,
    CPolylineMesh const& mesh,
    CRodPose<TimeStep::Current>& currPose) {
  MOCHI_PROFILE_SCOPE();
  auto const refPose = reference.MiddleRows(dofOffset.poseOffset, actorDofInfo.poseSize);
  auto const refDisplacement = refPose.MiddleRows(0, actorDofInfo.dofsSize);
  auto const dofDelta = increment.MiddleRows(dofOffset.dofsOffset, actorDofInfo.dofsSize);

  // Reference axes are stored contiguously after displacement DoFs in the packed pose vector,
  // with the same memory layout as Real3. Unflatten directly to avoid copying.
  int const numElements = mesh.NumElements();
  int const axesOffset = actorDofInfo.dofsSize;
  auto const refAxes = Unflatten<Real3 const>(
      Span<real const>{&refPose[axesOffset], static_cast<size_t>(3 * numElements)});

  rod::ApplyLieDeltaToPose(
      mesh.nodes,
      refDisplacement,
      refAxes,
      dofDelta,
      currPose.value.displacements,
      MakeSpan(currPose.value.frameAxes));
}

void EntityIncrementStep(
    ecs::Included<TagRodActor>,
    CRodPose<TimeStep::Current>& currPose,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CVelocitySlice<real, TimeStep::Previous>& prevVel,
    CRodPose<TimeStep::Previous>& prevPose,
    CIntegrationRodPoses& intPoses) {
  // Recenter twist DoFs: zero the current twist and shift all previous-step twist values by the
  // same amount, preserving differences used by multi-step integrators (analogous to translational
  // recentering for soft actors).
  int const numDisplacementDofs = isize(currPose.value.displacements);
  for (int i = fem::kRodThetaDofOffset; i < numDisplacementDofs; i += fem::kNumRodFields) {
    real const theta = currPose.value.displacements[i];
    currPose.value.displacements[i] = 0_r;
    for (auto& prev : intPoses.prevSteps) {
      prev.value.displacements[i] -= theta;
    }
  }
  prevPose.value = currPose.value;
  prevVel.CopyFrom(currVel);
  currVel.value.SetZero();
}

void EntityPreFirstStage(
    ecs::Included<TagRodActor>,
    CTimeIntegratorState const& intState,
    CPolylineMesh const& mesh,
    CVelocitySlice<real, TimeStep::Previous> const& prevVel,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels,
    CRodPose<TimeStep::Previous> const& prevPose,
    CIntegrationRodPoses& intPoses) {
  mochi::integration::ApplyTimeIntegrationStepStart(intState, intVels, prevVel, intVels.stepStart);
  mochi::integration::ApplyTimeIntegrationStepStart(
      MakeConstSpan(mesh.nodes), intState, intPoses, prevPose, intPoses.stepStart);
}

void EntityPreStage(
    ecs::Included<TagRodActor>,
    CTimeIntegratorState const& intState,
    CPolylineMesh const& mesh,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels,
    CVelocitySlice<real, TimeStep::StageStart>& stageStartVel,
    CIntegrationRodPoses& intPoses,
    CRodPose<TimeStep::StageStart>& stageStartPose) {
  MOCHI_PROFILE_SCOPE();
  mochi::integration::ApplyTimeIntegration<TimeTarget::StageStart>(
      intState, intVels, stageStartVel);
  mochi::integration::ApplyTimeIntegration<TimeTarget::StageStart>(
      MakeConstSpan(mesh.nodes), intState, intPoses, stageStartPose);
}

void EntityPostStage(
    ecs::Included<TagRodActor>,
    CConvergenceStatus const& convergence,
    CTimeIntegratorState const& intState,
    CRodPose<TimeStep::StageStart> const& stageStartPose,
    CRodPose<TimeStep::Current>& currPose,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels,
    CIntegrationRodPoses& intPoses,
    CReferenceElementFrameAxes const& referenceAxes) {
  MOCHI_PROFILE_SCOPE();
  currVel.value = (currPose.value.displacements - stageStartPose.value.displacements) *
      (1_r / intState.dtStage);
  if (convergence.stageStatus == ConvergenceStatus::Diverged) {
    currPose.value.displacements.SetZero();
    currVel.value.SetZero();
    currPose.value.frameAxes = referenceAxes.axes;
  }
  intPoses.stages[intState.currentStage].value = currPose.value;
  intVels.stages[intState.currentStage].value = currVel.value;
}

void EntityPostLastStage(
    ecs::Included<TagRodActor>,
    CTimeIntegratorState const& intState,
    CPolylineMesh const& mesh,
    CRodPose<TimeStep::Current>& currPose,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels,
    CIntegrationRodPoses& intPoses) {
  mochi::integration::ApplyTimeIntegration<TimeTarget::StepEnd>(
      MakeConstSpan(mesh.nodes), intState, intPoses, currPose);
  mochi::integration::ApplyTimeIntegration<TimeTarget::StepEnd>(intState, intVels, currVel);
}

// Update CDirichletBC based on CDofPositionsBC (if any)
static void UpdateDirichletBC(
    CRootTransform const& root,
    CFullSparsityPattern const& fullSparsity,
    CDofPositionsBC const* inWorldBC,
    CDirichletBC<real>& outLocalBC,
    CPolylineMesh const& mesh) {
  outLocalBC.Clear();
  if (inWorldBC && !inWorldBC->poseIndices.empty()) {
    TransformRT localFromWorld = Invert(root.worldFromLocal);
    VMatrix4x4r worldToLocalT = ToVMatrix4x4Transpose(localFromWorld);

    outLocalBC.dofIndices = inWorldBC->dofIndices;
    outLocalBC.poseIndices = inWorldBC->poseIndices;
    outLocalBC.colValueIndices = inWorldBC->colValueIndices;

    int const numBcDofs = isize(inWorldBC->dofIndices);
    outLocalBC.poseValues.resize(numBcDofs);
    for (int i = 0; i < numBcDofs; ++i) {
      int const dofIndex = inWorldBC->dofIndices[i];
      int const component = dofIndex % fem::kNumRodFields;
      if (component == 0) {
        // It is enforced through the public API that displacement BCs are specified in full ordered
        // blocks of three components for nodes.
        MOCHI_ASSERT_VERBOSE(
            (i + 2 < numBcDofs) && (inWorldBC->dofIndices[i + 1] == dofIndex + 1) &&
                (inWorldBC->dofIndices[i + 2] == dofIndex + 2),
            "Displacement BCs must be in ordered blocks of 3 for nodes of rod actors.");
        Vec4r const posWorld = ToSimdPoint(Load<3, Vec4r>(&(inWorldBC->poseValues[i])));
        Vec4r const posLocal = DotVecMat4x4(posWorld, worldToLocalT);
        Vec4r const posRef = ToSimd(mesh.nodes[dofIndex / fem::kNumRodFields], 1_r);
        Store<3>(&outLocalBC.poseValues[i], posLocal - posRef);
        // Next two DoF values have been handled; skip over them in the outer loop.
        i += 2;
      } else {
        // Twist DoF BCs (with values enforced by the public API to be zero) can be applied
        // directly.
        MOCHI_ASSERT_VERBOSE(
            component == fem::kRodThetaDofOffset, "Unexpected BC component for twist DoF");
        MOCHI_ASSERT_VERBOSE(inWorldBC->poseValues[i] == 0_r, "Twist BCs must be zero");
        outLocalBC.poseValues[i] = 0_r;
      }
    }
  } // Non-empty BC input

  // For open rods, always fix the last twist DoF, which is artificial padding for the block
  // structure. For closed-loop rods, all twist DOFs are physical — no padding needed.
  if (!mesh.isClosedLoop) {
    int const numDofs = fem::kNumRodFields * isize(mesh.nodes);
    outLocalBC.dofIndices.push_back(numDofs - 1);
    outLocalBC.poseIndices.push_back(numDofs - 1);
    outLocalBC.poseValues.push_back(0_r);
    AppendColValueIndexCache(
        fullSparsity.graph,
        MakeSingletonConstSpan(outLocalBC.dofIndices.back()),
        outLocalBC.colValueIndices);
  }
}

real GetActorMass(entt::registry const& reg, entt::entity actor) {
  MOCHI_ASSERT(reg.all_of<TagRodActor>(actor), "Expected rod actor.");
  auto const& nodalMasses = reg.get<CNodalMasses>(actor);
  real totalMass = 0_r;
  for (auto const& m : nodalMasses.values) {
    totalMass += m;
  }
  return totalMass;
}

void ComputeRodNodeCurvatureBinormals(
    Span<Real3 const> nodes,
    bool isClosedLoop,
    Span<Real3> outCurvatureBinormals) {
  int const numNodes = isize(nodes);
  MOCHI_ASSERT_VERBOSE(
      isize(outCurvatureBinormals) == numNodes,
      "outCurvatureBinormals must be sized to match nodes.");
  MOCHI_ASSERT_VERBOSE(
      numNodes >= (isClosedLoop ? 3 : 2),
      "Need at least 3 nodes for a closed-loop polyline, or 2 for an open polyline.");

  // Open: skip endpoints (no interior angle defined). Closed: visit every node with wraparound.
  int const firstNode = isClosedLoop ? 0 : 1;
  int const lastNode = isClosedLoop ? numNodes : numNodes - 1; // exclusive
  if (!isClosedLoop) {
    outCurvatureBinormals[0] = {};
    outCurvatureBinormals[numNodes - 1] = {};
  }
  for (int i = firstNode; i < lastNode; ++i) {
    int const iPrev = isClosedLoop ? ((i - 1 + numNodes) % numNodes) : (i - 1);
    int const iNext = isClosedLoop ? ((i + 1) % numNodes) : (i + 1);
    auto const e0 = Load<3, Vec4r>(nodes[i].data()) - Load<3, Vec4r>(nodes[iPrev].data());
    auto const e1 = Load<3, Vec4r>(nodes[iNext].data()) - Load<3, Vec4r>(nodes[i].data());
    real const L = 0.5_r * (Norm<3>(e0) + Norm<3>(e1));
    MOCHI_ASSERT_VERBOSE(L > 0_r, "Rod nodes must not be coincident (zero-length edges).");
    Vec4r const K = fem::IntegratedCurvatureBinormal(Normalize<3>(e0), Normalize<3>(e1)) / L;
    outCurvatureBinormals[i] = ToReal3(K);
  }
}

// Compute deformed triangular-surface node positions from a rod pose.
// Writes numSurfaceNodes × 3 values to outPositions (must be pre-sized).
//
// Implementation: two passes, with SIMD-friendly per-element transforms.
//   Pass 1 (per element, with periodic wrap): cache the per-element affine map
//     [scaledTDef | dDef | bDef | midDef] as the **transpose** of a 4x4 (VMatrix4x4r), with each
//     row holding one column of the affine. scaledTDef = (x1 - x0) / referenceLength is the
//     stretched unit tangent. Lane 3 of each stored row is intentionally unconstrained; it
//     would correspond to the affine's bottom row, which never reaches outPositions.
//   Pass 2 (per surface node × per weight): evaluate affine · [xi[0], xi[1], xi[2], 1] via
//     DotVecMat4x4 and accumulate the weighted contribution from each contributing element.
//     The meaningless homogeneous-output lane is discarded by Store<3>.
static void ComputeDeformedSurfaceNodePositions(
    TriangularMesh const& surfaceMesh,
    RodSurfaceEmbeddingData const& embedding,
    CPolylineMesh const& polylineMesh,
    RodPose const& rodPose,
    Span<real> outPositions) {
  int const numSurfaceNodes = surfaceMesh.GetNumNodes();
  int const K = embedding.weightsPerNode;
  int const numElements = polylineMesh.NumElements();
  auto const centerlineNodes = polylineMesh.nodes;
  auto const& displacements = rodPose.displacements;
  auto const& frameAxes = rodPose.frameAxes;

  MOCHI_ASSERT_VERBOSE(
      isize(outPositions) == kSpaceDim3 * numSurfaceNodes, "outPositions must be pre-sized");
  MOCHI_ASSERT_VERBOSE(
      isize(embedding.invReferenceLengths) == numElements,
      "invReferenceLengths size must match number of elements");

  // Pass 1: per-element affine transform stored as the transpose of a 4x4 (VMatrix4x4r).
  // Each VMatrix4x4r's rows are the four columns of the affine map
  // [scaledTDef | dDef | bDef | midDef]. The stack page is sized for ~64 elements; the
  // FiloAllocator transparently allocates additional heap pages on overflow.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 64 * sizeof(VMatrix4x4r));
  DynamicArray<VMatrix4x4r> elemTransforms(&allocator);
  elemTransforms.resize_noinit(numElements);
  for (int e = 0; e < numElements; ++e) {
    Int2 const en = polylineMesh.ElementNodes(e);
    Vec4r const x0 = ToSimd(centerlineNodes[en[0]], 0_r) +
        Load<Vec4r>(&displacements[fem::kNumRodFields * en[0]]);
    Vec4r const x1 = ToSimd(centerlineNodes[en[1]], 0_r) +
        Load<Vec4r>(&displacements[fem::kNumRodFields * en[1]]);
    Vec4r const tDef = x1 - x0;
    Vec4r const eHat = Normalize<3>(tDef);
    Vec4r const dDef = ToSimd(frameAxes[e], 0_r);
    Vec4r const bDef = Cross3(eHat, dDef);
    Vec4r const scaledTDef = embedding.invReferenceLengths[e] * tDef;
    Vec4r const midDef = 0.5_r * (x0 + x1);
    elemTransforms[e] = VMatrix4x4r{scaledTDef, dDef, bDef, midDef};
  }

  // Pass 2: per-surface-node skinning. Evaluate affine · [xi, 1] via DotVecMat4x4 on the
  // transposed transform; lane 3 of the result is meaningless and discarded by Store<3>.
  for (int i = 0; i < numSurfaceNodes; ++i) {
    Vec4r xSurface{};
    for (int m = 0; m < K; ++m) {
      int const idx = i * K + m;
      int const elemIdx = embedding.elementIndices[idx];
      real const w = embedding.weights[idx];
      Real3 const xi = embedding.localCoordinates[idx];

      Vec4r const localHom{xi[0], xi[1], xi[2], 1_r};
      Vec4r const xElem = DotVecMat4x4(localHom, elemTransforms[elemIdx]);
      xSurface += w * xElem;
    }
    Store<3>(&outPositions[kSpaceDim3 * i], xSurface);
  }
}

template <TimeStep kTimeStep>
void UpdateSurfaceContactPositions(
    ecs::Included<TagRodActor>,
    ecs::RequiredTag<TagRodSurfaceContact>,
    CRodPose<kTimeStep> const& rodPose,
    CRodContactSkin const& contactSkin,
    CPolylineMesh const& polylineMesh,
    CFemSurfaceDiscretization const& surfaceDisc,
    CRodDeformedContactSkinNodes& deformedNodes,
    CContactSamples<kTimeStep>& outSamples) {
  MOCHI_PROFILE_SCOPE();

  ComputeDeformedSurfaceNodePositions(
      *contactSkin.mesh,
      *contactSkin.embedding,
      polylineMesh,
      rodPose.value,
      MakeSpan(deformedNodes.positions));

  int constexpr kNumContactSkinNodesEstimate = 2048;
  MOCHI_FILO_STACK_ALLOCATOR(allocator, kSpaceDim3 * kNumContactSkinNodesEstimate * sizeof(real));
  ColumnVector<real> contactSkinNodeDisplacements(isize(deformedNodes.positions), &allocator);
  Span<real const> restPositions = Flatten(contactSkin.mesh->GetNodeCoordinates());
  contactSkinNodeDisplacements = AsConstView(deformedNodes.positions) - AsConstView(restPositions);

  UpdateCollisionSamplePositionsImpl</*kUpdateOnlyActiveFaces*/ false,
                                     CFemSurfaceDiscretization,
                                     /*kNumFields*/ 3>(
      contactSkinNodeDisplacements, surfaceDisc, nullptr, outSamples);
}

template void UpdateSurfaceContactPositions<TimeStep::Current>(
    ecs::Included<TagRodActor>,
    ecs::RequiredTag<TagRodSurfaceContact>,
    CRodPose<TimeStep::Current> const& rodPose,
    CRodContactSkin const& contactSkin,
    CPolylineMesh const& polylineMesh,
    CFemSurfaceDiscretization const& surfaceDisc,
    CRodDeformedContactSkinNodes& deformedNodes,
    CContactSamples<TimeStep::Current>& outSamples);

template void UpdateSurfaceContactPositions<TimeStep::StageStart>(
    ecs::Included<TagRodActor>,
    ecs::RequiredTag<TagRodSurfaceContact>,
    CRodPose<TimeStep::StageStart> const& rodPose,
    CRodContactSkin const& contactSkin,
    CPolylineMesh const& polylineMesh,
    CFemSurfaceDiscretization const& surfaceDisc,
    CRodDeformedContactSkinNodes& deformedNodes,
    CContactSamples<TimeStep::StageStart>& outSamples);

void SetupSurfaceCollidingJacobians(
    ecs::Included<TagRodActor>,
    ecs::RequiredTag<TagRodSurfaceContact>,
    CFemSurfaceDiscretization const& surfaceDisc,
    CRootTransform const& transform,
    CDofOffset const& dofOffset,
    CRodContactSkinningData const& skinningData,
    CCollJacs<CollRole::Colliding>& outJacobians) {
  MOCHI_PROFILE_SCOPE();

  auto skinJacView = AsConstView(skinningData.jacobian);
  dmap::DMapSparseSkinning sparseSkinning(0, dofOffset.dofsOffset, skinJacView);
  dmap::DMapRTConst dtransform(transform.worldFromLocal);

  surfaceDisc.Visit([&](auto const& discImpl) {
    using DiscT = std::decay_t<decltype(discImpl)>;
    using DQuad = dmap::DMapQuad<typename DiscT::ElementT>;

    MOCHI_FILO_STACK_ALLOCATOR(tempAlloc, 256 * sizeof(JacData*));
    auto allJacs = outJacobians.GetPtrsNonEmpty(&tempAlloc);
    if (allJacs.empty()) {
      return;
    }

    ParallelForEach("rod::SetupSurfaceCollidingJacobians Range", allJacs, 1, [&](JacData* jac) {
      DQuad dquad(discImpl.femElements, jac->query->jacColliderFromWorld);
      dmap::DMap<DQuad, dmap::DMapRTConst, dmap::DMapSparseSkinning> dmap(
          &dquad, &dtransform, &sparseSkinning);

      auto& jacs = *(jac->jacs);
      dmap.GetJac(jac->query->sampleIndices, jacs);
      jacs[0].CompressIndices();
    });
  });
}

template <TimeStep kStep>
void UpdateSurfaceContactBounds(
    ecs::Included<TagRodActor>,
    ecs::RequiredTag<TagRodSurfaceContact>,
    CRodContactSkin const& contactSkin,
    CPolylineMesh const& polylineMesh,
    CRodPose<kStep> const& rodPose,
    CRodDeformedContactSkinNodes& deformedNodes,
    CBoundingVolume<TimeStep::Current>& outBounds) {
  static_assert(kStep == TimeStep::Current || kStep == TimeStep::StageStart);
  MOCHI_PROFILE_SCOPE();

  if (contactSkin.mesh->GetNumNodes() == 0) {
    return;
  }

  ComputeDeformedSurfaceNodePositions(
      *contactSkin.mesh,
      *contactSkin.embedding,
      polylineMesh,
      rodPose.value,
      MakeSpan(deformedNodes.positions));
  auto const deformedNodesView = Unflatten<Real3 const>(MakeConstSpan(deformedNodes.positions));

  Vec4r min = ToSimd(deformedNodesView[0], 0_r);
  Vec4r max = min;
  for (int i = 1; i < isize(deformedNodesView); ++i) {
    Vec4r const pos = ToSimd(deformedNodesView[i], 0_r);
    min = Min(min, pos);
    max = Max(max, pos);
  }
  outBounds.localShape = GetObb(Aabb{Set(min, 3, 0_r), Set(max, 3, 0_r)});
}

template void UpdateSurfaceContactBounds<TimeStep::Current>(
    ecs::Included<TagRodActor>,
    ecs::RequiredTag<TagRodSurfaceContact>,
    CRodContactSkin const& contactSkin,
    CPolylineMesh const& polylineMesh,
    CRodPose<TimeStep::Current> const& rodPose,
    CRodDeformedContactSkinNodes& deformedNodes,
    CBoundingVolume<TimeStep::Current>& outBounds);

template void UpdateSurfaceContactBounds<TimeStep::StageStart>(
    ecs::Included<TagRodActor>,
    ecs::RequiredTag<TagRodSurfaceContact>,
    CRodContactSkin const& contactSkin,
    CPolylineMesh const& polylineMesh,
    CRodPose<TimeStep::StageStart> const& rodPose,
    CRodDeformedContactSkinNodes& deformedNodes,
    CBoundingVolume<TimeStep::Current>& outBounds);

} // namespace mochi::rod

RodSurfaceEmbeddingData mochi::ComputeRodSurfaceEmbedding(
    Span<Real3 const> nodes,
    Span<Real3 const> frameAxes,
    TriangularMesh const& surfaceMesh,
    SkinningData&& skinning,
    bool isClosedLoop) {
  int const weightsPerNode = skinning.weightsPerNode;
  int const numSurfaceNodes = surfaceMesh.GetNumNodes();
  int const numNodes = isize(nodes);
  int const numElements = isClosedLoop ? numNodes : numNodes - 1;
  MOCHI_ASSERT(
      isize(frameAxes) == numElements,
      "frameAxes must have one entry per polyline element (caller should generate if needed)");

  RodSurfaceEmbeddingData embedding;
  embedding.weightsPerNode = weightsPerNode;
  embedding.elementIndices = std::move(skinning.indices);
  embedding.weights = std::move(skinning.weights);
  embedding.localCoordinates.resize(numSurfaceNodes * weightsPerNode);

  // Precompute reciprocal reference lengths once per element, so the per-(surface node, element)
  // ξ solve below builds a unit-reference-tangent basis that is well conditioned independent of
  // edge length.
  embedding.invReferenceLengths.resize_noinit(numElements);
  for (int e = 0; e < numElements; ++e) {
    real const referenceLength = Norm(nodes[(e + 1) % numNodes] - nodes[e]);
    MOCHI_ASSERT_VERBOSE(referenceLength > 0_r, "Rod element has zero reference length");
    embedding.invReferenceLengths[e] = 1_r / referenceLength;
  }

  auto const surfaceCoordinates = surfaceMesh.GetNodeCoordinates();
  for (int i = 0; i < numSurfaceNodes; ++i) {
    for (int m = 0; m < weightsPerNode; ++m) {
      int const idx = i * weightsPerNode + m;
      int const elemIdx = embedding.elementIndices[idx];
      MOCHI_ASSERT_VERBOSE(elemIdx >= 0 && elemIdx < numElements);

      Int2 const en = {elemIdx, (elemIdx + 1) % numNodes};
      Real3 const x0 = nodes[en[0]];
      Real3 const x1 = nodes[en[1]];
      Real3 const mid = 0.5_r * (x0 + x1);
      // Use a *unit* reference tangent so xi[0] is in arc-length units. This makes the local solve
      // independent of the element edge length and keeps the runtime skinning frame well
      // conditioned under stretch (cf. xi[0] is rescaled by invReferenceLengths at runtime).
      Real3 const unitReferenceT = embedding.invReferenceLengths[elemIdx] * (x1 - x0);
      Real3 const d = frameAxes[elemIdx];
      Real3 const b = Cross(unitReferenceT, d);

      Real3 const rhs = surfaceCoordinates[i] - mid;
      // Solve [unitReferenceT | d | b] * xi = rhs. Since (unitReferenceT, d, b) form an
      // orthonormal basis, the matrix is a rotation and its inverse equals its transpose.
      Real3 const xi = DotMatVec(Matrix3x3r{unitReferenceT, d, b}, rhs);
      embedding.localCoordinates[idx] = xi;
    }
  }
  return embedding;
}

void mochi::rod::UpdateQueryVisualNodePositionsAndNormals(
    CVisualMesh const& visualMesh,
    CRodVisualMeshEmbedding const& rodEmbedding,
    CPolylineMesh const& polylineMesh,
    CRodPose<TimeStep::Current> const& rodPose,
    CQueryVisualNodePositions& outVisPosQuery,
    CQueryVisualNodeNormals* outVisNormQuery) {
  MOCHI_PROFILE_SCOPE();

  int const numVisualNodes = visualMesh.mesh->GetNumNodes();
  auto const numValues = static_cast<size_t>(kSpaceDim3) * numVisualNodes;
  outVisPosQuery.nodePositions.resize(numValues);

  ComputeDeformedSurfaceNodePositions(
      *visualMesh.mesh,
      *rodEmbedding.data,
      polylineMesh,
      rodPose.value,
      MakeSpan(outVisPosQuery.nodePositions));

  if (outVisNormQuery) {
    UpdateQueryVisualNodeNormals(false, visualMesh, outVisPosQuery, *outVisNormQuery);
  }
}

void mochi::rod::InitializeContactSkinningJacobian(
    CRodContactSkin const& contactSkin,
    CPolylineMesh const& polylineMesh,
    CRodContactSkinningData& outSkinning) {
  auto const& embData = *contactSkin.embedding;
  int const numContactSkinNodes = contactSkin.mesh->GetNumNodes();
  int const K = embData.weightsPerNode;
  int const numNodes = isize(polylineMesh.nodes);
  int const numDofs = numNodes * fem::kNumRodFields;

  // Collect unique, sorted DOF indices per contact-skin node, deduplicating overlapping elements.
  DynamicArray<DynamicArray<int>> dofIndicesPerNode(numContactSkinNodes);

  int const numRows = numContactSkinNodes;
  DynamicArray<int> rowPtrs;
  rowPtrs.resize_noinit(numRows + 1);
  rowPtrs[0] = 0;

  for (int i = 0; i < numContactSkinNodes; ++i) {
    auto& nodeDofs = dofIndicesPerNode[i];
    nodeDofs.reserve(K * 2 * fem::kNumRodFields);

    for (int m = 0; m < K; ++m) {
      int const idx = i * K + m;
      int const elemIdx = embData.elementIndices[idx];

      Int2 const en = polylineMesh.ElementNodes(elemIdx);
      int const node0Start = en[0] * fem::kNumRodFields;
      int const node1Start = en[1] * fem::kNumRodFields;

      // The 4 DoFs of a rod node are always pushed together, so checking only the first
      // suffices to dedup the whole block.
      for (int nodeStart : {node0Start, node1Start}) {
        if (!Contains(nodeDofs, nodeStart)) {
          for (int d = 0; d < fem::kNumRodFields; ++d) {
            nodeDofs.push_back(nodeStart + d);
          }
        }
      }
    }

    std::sort(nodeDofs.begin(), nodeDofs.end());
    rowPtrs[i + 1] = rowPtrs[i] + isize(nodeDofs);
  }

  int const totalNnz = rowPtrs[numRows];
  DynamicArray<int> colIndices(totalNnz);
  for (int i = 0; i < numContactSkinNodes; ++i) {
    auto const& nodeDofs = dofIndicesPerNode[i];
    std::copy_n(nodeDofs.data(), isize(nodeDofs), &colIndices[rowPtrs[i]]);
  }

  DynamicArray<Real3> values(totalNnz, Real3{});
  outSkinning.jacobian =
      SparseMatrix<Real3>(numDofs, std::move(rowPtrs), std::move(colIndices), std::move(values));
}

void mochi::rod::ResolveContactSkinningJacobian(
    CRodContactSkin const& contactSkin,
    CPolylineMesh const& polylineMesh,
    CRodPose<TimeStep::Current> const& rodPose,
    CRodContactSkinningData& outSkinning) {
  MOCHI_PROFILE_SCOPE();

  auto const& embData = *contactSkin.embedding;
  int const numContactSkinNodes = contactSkin.mesh->GetNumNodes();
  int const K = embData.weightsPerNode;
  auto const centerlineNodes = polylineMesh.nodes;
  auto const& displacements = rodPose.value.displacements;
  auto const& frameAxes = rodPose.value.frameAxes;

  auto& jac = outSkinning.jacobian;
  jac.SetZero();

  for (int i = 0; i < numContactSkinNodes; ++i) {
    auto const colIndices = jac.Indices(i);
    auto values = jac.Values(i);

    for (int m = 0; m < K; ++m) {
      int const idx = i * K + m;
      int const elemIdx = embData.elementIndices[idx];
      real const w = embData.weights[idx];
      Real3 const xi = embData.localCoordinates[idx];

      Int2 const en = polylineMesh.ElementNodes(elemIdx);
      NdArray<real, 8> elemDofs;
      int const node0Start = en[0] * fem::kNumRodFields;
      int const node1Start = en[1] * fem::kNumRodFields;
      Store(&elemDofs[0], Load<Vec4r>(&displacements[node0Start]));
      Store(&elemDofs[fem::kNumRodFields], Load<Vec4r>(&displacements[node1Start]));

      NdArray<real, 3, 8> elemJac;
      fem::ComputeEmbeddedPointElementJacobian(
          centerlineNodes[en[0]],
          centerlineNodes[en[1]],
          xi,
          embData.invReferenceLengths[elemIdx],
          frameAxes[elemIdx],
          MakeConstSpan(elemDofs),
          elemJac);

      // Find localCol0 via binary search in the pre-built sorted column indices.
      auto const* it0 = std::lower_bound(colIndices.begin(), colIndices.end(), node0Start);
      MOCHI_ASSERT_VERBOSE(
          it0 != colIndices.end() && *it0 == node0Start, "node0Start not in sparsity pattern");
      int const localCol0 = static_cast<int>(it0 - colIndices.begin());
      // Second node of rod element may wrap around to zero for closed-loop rod.
      int const localCol1 = (localCol0 + fem::kNumRodFields) % isize(colIndices);
      MOCHI_ASSERT_VERBOSE(
          colIndices[localCol1] == node1Start,
          "node1Start DOF must be present (possibly wrapped) in sparsity pattern");

      for (int d = 0; d < fem::kNumRodFields; ++d) {
        for (int r = 0; r < 3; ++r) {
          values[localCol0 + d][r] += w * elemJac[r][d];
          values[localCol1 + d][r] += w * elemJac[r][fem::kNumRodFields + d];
        }
      }
    }
  }
}

DynamicArray<Real3> mochi::GenerateDiscreteBishopFrame(Span<Real3 const> nodes, bool isClosedLoop) {
  int const numNodes = isize(nodes);
  MOCHI_ASSERT(numNodes >= 2, "At least two nodes are required to generate a discrete frame.");
  int const numElements = isClosedLoop ? numNodes : numNodes - 1;
  DynamicArray<Real3> elementFrameAxes(numElements);

  // Compute unit tangent for first element
  Real3 const edge0 = nodes[1] - nodes[0];
  real const length0 = Norm(edge0);
  Real3 const tangent0 = edge0 / length0;

  // Choose the element frame axis for the first element by finding the largest-norm
  // cross product of the tangent with one of the standard unit basis vectors.
  // This ensures we get a well-conditioned orthogonal axis even when the tangent
  // is nearly aligned with one of the basis vectors.
  Real3 axis0;
  real maxNormSq = 0_r;
  for (int i = 0; i < 3; ++i) {
    Real3 basisVec = {};
    basisVec[i] = 1_r;
    Real3 const cross = Cross(tangent0, basisVec);
    real const normSq = NormSqr(cross);
    if (normSq > maxNormSq) {
      maxNormSq = normSq;
      axis0 = Normalize(cross);
    }
  }
  elementFrameAxes[0] = axis0;

  // Transport the axis from each element to the next using minimal rotations
  Vec4r prevTangent = ToSimd(tangent0);
  Vec4r prevAxis = ToSimd(axis0);
  real totalLength = length0;
  for (int i = 1; i < numElements; ++i) {
    Real3 const edge = nodes[(i + 1) % numNodes] - nodes[i];
    real const length = Norm(edge);
    totalLength += length;
    Vec4r const currTangent = ToSimd(edge / length);

    // ParallelTransportOperator computes the minimal rotation from prevTangent to currTangent
    VMatrix3x3r const P = fem::ParallelTransportOperator(prevTangent, currTangent);
    Vec4r const currAxis = DotMatVec3x3(P, prevAxis);
    elementFrameAxes[i] = ToReal3(currAxis);

    prevTangent = currTangent;
    prevAxis = currAxis;
  }

  // Holonomy correction for closed-loop rods: Distribute holonomy evenly along rod for closed-loop
  // frame axes with minimal twisting energy.
  if (isClosedLoop && numElements >= 2) {
    Vec4r const t0 = ToSimd(tangent0);
    VMatrix3x3r const Pclosing = fem::ParallelTransportOperator(prevTangent, t0);
    Vec4r const transportedAxis = DotMatVec3x3(Pclosing, prevAxis);

    Vec4r const d0 = ToSimd(elementFrameAxes[0]);
    Vec4r const b0 = Cross3(t0, d0);
    real const sinAngle = Dot(transportedAxis, b0);
    real const cosAngle = Dot(transportedAxis, d0);
    real const holonomyAngle = ATan2(sinAngle, cosAngle);

    if (Abs(holonomyAngle) > kDefaultNearEqualEpsilon<real>) {
      real const invTotalLength = 1_r / totalLength;
      real cumulativeLength = length0;
      for (int i = 1; i < numElements; ++i) {
        Real3 const edge = nodes[(i + 1) % numNodes] - nodes[i];
        real const length = Norm(edge);
        cumulativeLength += length;
        real const alpha = -cumulativeLength * holonomyAngle * invTotalLength;
        Real3 const t = edge / length;
        Real3 const d = elementFrameAxes[i];
        elementFrameAxes[i] = Cos(alpha) * d + Sin(alpha) * Cross(t, d);
      }
    }
  }

  return elementFrameAxes;
}

MOCHI_API ModelData mochi::experimental::GenerateTubularRodModelData(
    Span<Real3 const> nodes,
    Span<Real3 const> elementFrameAxes,
    real radius,
    int numCrossSectionSegments,
    bool isClosedLoop,
    Error& error) {
  if (isClosedLoop) {
    MOCHI_ERROR_IF_NOT(isize(nodes) >= 3, error, "Must have at least 3 nodes for a closed loop.");
  } else {
    MOCHI_ERROR_IF_NOT(isize(nodes) >= 2, error, "Must have at least 2 nodes.");
  }
  MOCHI_ERROR_IF_NOT(
      radius > 0_r && IsFinite(radius), error, "Radius must be positive and finite.");
  MOCHI_ERROR_IF_NOT(
      numCrossSectionSegments >= 3, error, "numCrossSectionSegments must be at least 3.");
  MOCHI_ERROR_RETURN(error, {});

  // Validate that the polyline has well-defined element tangents (nonzero edge lengths and no
  // 180-degree consecutive tangent rotations) before invoking GenerateDiscreteBishopFrame, which
  // would otherwise fail with a division by zero or undefined parallel-transport rotation axis.
  mochi::model::ValidatePolylineGeometry(nodes, isClosedLoop, error);
  MOCHI_ERROR_RETURN(error, {});

  // Validate caller-supplied frame axes (if any). Empty means auto-generate via Bishop frame.
  if (!elementFrameAxes.empty()) {
    mochi::model::ValidatePolylineElementFrameAxes(elementFrameAxes, nodes, isClosedLoop, error);
    MOCHI_ERROR_RETURN(error, {});
  }

  int const numNodes = isize(nodes);
  int const numElements = isClosedLoop ? numNodes : numNodes - 1;

  // Compute a discrete Bishop frame for twist-free cross-section orientation. Note that these axes
  // are intentionally independent of the rod's material frame axes, to minimize distortion of the
  // tubular visual mesh in the reference configuration, even if the material frame axes have some
  // nontrivial twist.
  DynamicArray<Real3> const bishopAxes = GenerateDiscreteBishopFrame(nodes, isClosedLoop);

  // Use the provided element frame axes, or default to the Bishop frame if not provided
  Span<Real3 const> const frameAxes =
      elementFrameAxes.empty() ? MakeConstSpan(bishopAxes) : elementFrameAxes;

  if (isClosedLoop) {
    // Closed loop: one ring per element midpoint, no boundary rings, no end caps.
    int const numRings = numElements;
    int const numVisualNodes = numCrossSectionSegments * numRings;
    DynamicArray<real> visNodePositions;
    visNodePositions.reserve(3 * numVisualNodes);
    real const invSeg = 1_r / static_cast<real>(numCrossSectionSegments);
    for (int r = 0; r < numRings; ++r) {
      Int2 const en = {r, (r + 1) % numNodes};
      Real3 const center = 0.5_r * (nodes[en[0]] + nodes[en[1]]);
      Real3 const tangent = Normalize(nodes[en[1]] - nodes[en[0]]);
      Real3 const binormal = Cross(tangent, bishopAxes[r]);
      for (int s = 0; s < numCrossSectionSegments; ++s) {
        real const angle = 2_r * kPI * static_cast<real>(s) * invSeg;
        Real3 const pos = center + radius * (Cos(angle) * bishopAxes[r] + Sin(angle) * binormal);
        visNodePositions.append(MakeConstSpan(pos));
      }
    }
    DynamicArray<int> visTriangles;
    visTriangles.reserve(3 * 2 * numCrossSectionSegments * numRings);
    for (int i = 0; i < numRings; ++i) {
      int const nextRing = (i + 1) % numRings;
      for (int c = 0; c < numCrossSectionSegments; ++c) {
        int const c1 = (c + 1) % numCrossSectionSegments;
        int const v00 = i * numCrossSectionSegments + c;
        int const v01 = i * numCrossSectionSegments + c1;
        int const v10 = nextRing * numCrossSectionSegments + c;
        int const v11 = nextRing * numCrossSectionSegments + c1;
        Int3 const tri0{v00, v01, v11};
        visTriangles.append(MakeConstSpan(tri0));
        Int3 const tri1{v00, v11, v10};
        visTriangles.append(MakeConstSpan(tri1));
      }
    }
    int constexpr kWeightsPerNode = 1;
    DynamicArray<int> embeddingElementIndices;
    embeddingElementIndices.reserve(kWeightsPerNode * numVisualNodes);
    for (int r = 0; r < numRings; ++r) {
      for (int s = 0; s < numCrossSectionSegments; ++s) {
        embeddingElementIndices.push_back(r);
      }
    }
    DynamicArray<real> embeddingWeights(numVisualNodes, 1_r);
    ModelData model;
    model.mesh.emplace(MeshData{});
    model.mesh->nodesPerElement = 2;
    model.mesh->coordinates = Flatten(nodes);
    // Closed-loop polyline: connectivity must include the wrap-around segment.
    model.mesh->connectivity = MakeSequentialPolylineConnectivity(numNodes, /*isClosedLoop=*/true);
    model.elementFrameAxes = Flatten(frameAxes);
    model.visualMesh.emplace(MeshData{});
    model.visualMesh->nodesPerElement = 3;
    model.visualMesh->coordinates = std::move(visNodePositions);
    model.visualMesh->connectivity = std::move(visTriangles);
    model.visualMesh->skinning.emplace(SkinningData{});
    model.visualMesh->skinning->weightsPerNode = kWeightsPerNode;
    model.visualMesh->skinning->indices = std::move(embeddingElementIndices);
    model.visualMesh->skinning->weights = std::move(embeddingWeights);
    return model;
  }

  // --- Open polyline path ---

  // Cross-section rings are placed at element midpoints, with boundary rings at the first and last
  // nodes. Each ring uses the Bishop frame of its associated element directly (no averaging).
  int const numRings = numElements + 2;
  DynamicArray<Real3> ringD, ringB;
  ringD.resize_noinit(numRings);
  ringB.resize_noinit(numRings);
  for (int r = 0; r < numRings; ++r) {
    int const elem = Min(Max(r - 1, 0), numElements - 1);
    Real3 const tangent = Normalize(nodes[elem + 1] - nodes[elem]);
    Real3 const d = bishopAxes[elem];
    ringD[r] = d;
    ringB[r] = Cross(tangent, d);
  }

  // Visual surface nodes: numCrossSectionSegments vertices per ring, plus 2 cap-center vertices
  int const numVisualNodes = numCrossSectionSegments * numRings + 2;
  DynamicArray<real> visNodePositions;
  visNodePositions.reserve(3 * numVisualNodes);
  real const invNumCrossSectionSegments = 1_r / static_cast<real>(numCrossSectionSegments);
  for (int r = 0; r < numRings; ++r) {
    Real3 center;
    if (r == 0) {
      center = nodes[0];
    } else if (r == numRings - 1) {
      center = nodes[numNodes - 1];
    } else {
      center = 0.5_r * (nodes[r - 1] + nodes[r]);
    }
    for (int s = 0; s < numCrossSectionSegments; ++s) {
      real const angle = 2_r * kPI * static_cast<real>(s) * invNumCrossSectionSegments;
      Real3 const pos = center + radius * (Cos(angle) * ringD[r] + Sin(angle) * ringB[r]);
      visNodePositions.append(MakeConstSpan(pos));
    }
  }
  visNodePositions.append(MakeConstSpan(nodes[0]));
  visNodePositions.append(MakeConstSpan(nodes[numNodes - 1]));

  // Triangles: quad strips between consecutive rings + triangle-fan end caps
  int const numQuadStrips = numRings - 1;
  DynamicArray<int> visTriangles;
  visTriangles.reserve(
      3 * (2 * numCrossSectionSegments * numQuadStrips + 2 * numCrossSectionSegments));
  for (int i = 0; i < numQuadStrips; ++i) {
    for (int c = 0; c < numCrossSectionSegments; ++c) {
      int const c1 = (c + 1) % numCrossSectionSegments;
      int const v00 = i * numCrossSectionSegments + c;
      int const v01 = i * numCrossSectionSegments + c1;
      int const v10 = (i + 1) * numCrossSectionSegments + c;
      int const v11 = (i + 1) * numCrossSectionSegments + c1;
      Int3 const tri0{v00, v01, v11};
      visTriangles.append(MakeConstSpan(tri0));
      Int3 const tri1{v00, v11, v10};
      visTriangles.append(MakeConstSpan(tri1));
    }
  }
  // Start cap
  int const startCapCenter = numCrossSectionSegments * numRings;
  for (int s = 0; s < numCrossSectionSegments; ++s) {
    int const s1 = (s + 1) % numCrossSectionSegments;
    Int3 const tri{startCapCenter, s1, s};
    visTriangles.append(MakeConstSpan(tri));
  }
  // End cap
  int const endCapCenter = startCapCenter + 1;
  int const lastRingStart = (numRings - 1) * numCrossSectionSegments;
  for (int s = 0; s < numCrossSectionSegments; ++s) {
    int const s1 = (s + 1) % numCrossSectionSegments;
    Int3 const tri{endCapCenter, lastRingStart + s, lastRingStart + s1};
    visTriangles.append(MakeConstSpan(tri));
  }

  // Embedding: each visual vertex is assigned to exactly one element
  int constexpr kWeightsPerNode = 1;
  DynamicArray<int> embeddingElementIndices;
  DynamicArray<real> embeddingWeights;
  embeddingElementIndices.reserve(kWeightsPerNode * numVisualNodes);
  embeddingWeights.reserve(kWeightsPerNode * numVisualNodes);
  for (int r = 0; r < numRings; ++r) {
    int const elem = Min(Max(r - 1, 0), numElements - 1);
    for (int s = 0; s < numCrossSectionSegments; ++s) {
      embeddingElementIndices.push_back(elem);
      embeddingWeights.push_back(1_r);
    }
  }
  // Cap center vertices
  embeddingElementIndices.push_back(0);
  embeddingWeights.push_back(1_r);
  embeddingElementIndices.push_back(numElements - 1);
  embeddingWeights.push_back(1_r);

  // Construct ModelData
  ModelData model;

  // Polyline mesh (simulation mesh)
  model.mesh.emplace(MeshData{});
  model.mesh->nodesPerElement = 2;
  model.mesh->coordinates = Flatten(nodes);
  // Open polyline: emit explicit sequential connectivity to make the topology
  // self-describing through the public API.
  model.mesh->connectivity = MakeSequentialPolylineConnectivity(numNodes, /*isClosedLoop=*/false);
  model.elementFrameAxes = Flatten(frameAxes);

  // Visual mesh (triangular tube)
  model.visualMesh.emplace(MeshData{});
  model.visualMesh->nodesPerElement = 3;
  model.visualMesh->coordinates = std::move(visNodePositions);
  model.visualMesh->connectivity = std::move(visTriangles);

  // Skinning data
  model.visualMesh->skinning.emplace(SkinningData{});
  model.visualMesh->skinning->weightsPerNode = kWeightsPerNode;
  model.visualMesh->skinning->indices = std::move(embeddingElementIndices);
  model.visualMesh->skinning->weights = std::move(embeddingWeights);

  return model;
}

void mochi::PreStepRodActorAsync(entt::registry& reg, entt::entity e) {
  MOCHI_PROFILE_SCOPE();
  // Called to prepare a rod actor for simulation step
  ecs::InvokeOnEntity(&rod::UpdateDirichletBC, reg, e);
}

static void EmplaceRodActorContact(
    entt::registry& reg,
    entt::entity e,
    experimental::RodActorParams const& params,
    PolylineShape const& shape,
    CFemSegmentDiscretization const& segmentDisc,
    int numCollidingSamples,
    Error& error) {
  Obb const meshObb = CalcObb(MakeConstSpan(shape.GetNodes()));
  reg.emplace<CBoundingVolume<TimeStep::Current>>(e, meshObb);
  reg.emplace<CBoundingVolume<TimeStep::Previous>>(e, meshObb);

  ColliderType colliderType = params.colliderType;
  if (colliderType == ColliderType::Auto) {
    colliderType = ColliderType::PointCloud;
  }

  auto& collider = reg.emplace<CColliderInfo>(e);
  collider.type = colliderType;

  reg.emplace<CContactParams>(e, params.contact);

  // Components to detect and compute contact against other actors
  deformable::EmplaceContactComponents(reg, e, numCollidingSamples);
  reg.emplace<CDeformablePointAsyncCollisionsResponse>(e);

  reg.emplace<CActorAsyncContactSemaphore>(e);

  // Conservative bounds used for island formation.
  reg.emplace<CConservativeStepBounds>(e);

  if (colliderType == ColliderType::None) {
    return;
  }

  MOCHI_ERROR_IF_NOT(
      colliderType == ColliderType::PointCloud,
      error,
      "Collider type not supported for rod actors.");
  MOCHI_ERROR_RETURN(error);

  reg.emplace<TagUsePointCloudContact>(e);
  reg.emplace<CCollJacs<CollRole::Collider>>(e);

  ValidatePointCloudColliderParams(params.pointCloudCollider, error);
  MOCHI_ERROR_RETURN(error);
  auto& pcComponent = reg.emplace<CPointCloudColliderParams>(e, params.pointCloudCollider);
  pcComponent.integralDim = 1;

  // Create collider point cloud discretization from the segment element type.
  auto const& nodes = shape.GetNodes();
  CColliderPointCloudDiscretization colliderDisc = [&]() -> CColliderPointCloudDiscretization {
    if (!params.pointCloudCollider.colliderSegmentElementType.has_value()) {
      // Nodal collider: one point per node, weighted by lumped segment lengths assembled from the
      // contact discretization (mirroring the shell pattern with InitializeNodalWeights).
      DynamicArray<real> nodalWeights;
      segmentDisc.Visit([&](auto const& disc) { nodalWeights = InitializeNodalWeights(disc); });
      NodalColliderDiscretization nodalDisc;
      int const numNodes = isize(nodes);
      nodalDisc.femElements.reserve(numNodes);
      for (int i = 0; i < numNodes; ++i) {
        nodalDisc.femElements.emplace_back(i, MakeConstSpan(nodes), i, nodalWeights[i]);
      }
      return CColliderPointCloudDiscretization(std::move(nodalDisc), fem::kNumRodFields);
    }
    return CColliderPointCloudDiscretization(
        CFemSegmentDiscretization::Create(
            *params.pointCloudCollider.colliderSegmentElementType, nodes, shape.IsClosedLoop()),
        fem::kNumRodFields);
  }();
  auto& colliderDiscRef =
      reg.emplace<CColliderPointCloudDiscretization>(e, std::move(colliderDisc));

  reg.emplace<CSpatialHashTable>(
      e,
      CreateSpatialHashTable(
          params.pointCloudCollider,
          colliderDiscRef,
          params.contact.GetPenaltyThresholdDist(true)));
}

void mochi::InitRodActor(
    entt::registry& reg,
    entt::entity e,
    experimental::RodActorParams const& params,
    std::shared_ptr<PolylineShape const> shapePtr,
    Error& error) {
  // Validate material parameters for actor creation
  // (Geometry validation is done in model::Validate)
  MOCHI_ERROR_IF(params.material.axialStiffness <= 0_r, error, "Invalid axial stiffness");
  // Deliberately allow zero torsional or flexural stiffnesses, to turn off torsion and/or bending
  // in cable-like rods.
  MOCHI_ERROR_IF(params.material.torsionalStiffness < 0_r, error, "Invalid torsional stiffness");
  for (auto const& flexuralStiffnessComponent : params.material.flexuralStiffness) {
    MOCHI_ERROR_IF(flexuralStiffnessComponent < 0_r, error, "Invalid flexural stiffness");
  }
  MOCHI_ERROR_IF(params.material.linearDensity <= 0_r, error, "Invalid linear density");
  // Exact zero is well-posed if there is at least one BC on twisting DoFs.
  MOCHI_ERROR_IF(
      params.material.linearRotationalInertia < 0_r, error, "Invalid rotational inertia");
  MOCHI_ERROR_IF(
      params.material.linearRotationalInertia == 0_r && params.material.torsionalStiffness == 0_r,
      error,
      "Rotational inertia and torsional stiffness cannot both be zero.");
  MOCHI_ERROR_IF(
      !IsFinite(params.material.massDampingCoefficient) ||
          params.material.massDampingCoefficient < 0_r,
      error,
      "Invalid mass damping coefficient");
  MOCHI_ERROR_IF(
      !IsFinite(params.material.stiffnessDampingCoefficient) ||
          params.material.stiffnessDampingCoefficient < 0_r,
      error,
      "Invalid stiffness damping coefficient");
  ValidateContactParams(params.contact, error);
  MOCHI_ERROR_RETURN(error);

  // Get nodes and element frame axes from the shape
  DynamicArray<Real3> const& elementFrameAxes = shapePtr->GetElementFrameAxes();

  // Identification
  reg.emplace<TagRodActor>(e);
  reg.emplace<TagDeformableActor>(e);
  reg.emplace<CActorInfo>(e, std::string(params.name), ActorType::Rod);
  reg.emplace<CDofOffset>(e);

  EmplaceContactLayer(reg, e, params.layer);

  reg.emplace<CRootTransform>(e, params.worldFromLocal);
  reg.emplace<CTimeIntegratorState>(e);
  reg.emplace<CConvergenceStatus>(e);

  // Mesh geometry (const span of shape's nodes)
  bool const isClosedLoop = shapePtr->IsClosedLoop();
  auto const& mesh = reg.emplace<CPolylineMesh>(
      e, MakeConstSpan(shapePtr->GetNodes()), shapePtr->GetFlatConnectivity(), isClosedLoop);
  reg.emplace<CReferenceElementFrameAxes>(e, MakeConstSpan(shapePtr->GetElementFrameAxes()));

  // Store the shape
  reg.emplace<CShape>(e, shapePtr);

  // Set up DoF information
  // Each node has 4 DoFs: 3 displacement + 1 twist
  int const numNodes = isize(mesh.nodes);
  int const numElements = mesh.NumElements();
  int constexpr kNumDofsPerNode = fem::kNumRodFields;
  int const totalDofs = numNodes * kNumDofsPerNode;

  CActorDofInfo& dofInfo = reg.emplace<CActorDofInfo>(e);
  // Pose layout: [displacement_twist (numNodes*4) | axes (numElements*3)]
  dofInfo.poseSize = totalDofs + numElements * 3;
  dofInfo.dofsSize = totalDofs;

  // Rod pose components — single source of truth for displacements + frame axes
  auto& rodPoseCurr = reg.emplace<CRodPose<TimeStep::Current>>(e, numNodes, mesh.isClosedLoop);
  auto& rodPoseStageStart =
      reg.emplace<CRodPose<TimeStep::StageStart>>(e, numNodes, mesh.isClosedLoop);
  auto& rodPosePrev = reg.emplace<CRodPose<TimeStep::Previous>>(e, numNodes, mesh.isClosedLoop);

  // Rod pose integration (displacement-twist + frame axes, tracked across time levels)
  reg.emplace<CIntegrationRodPoses>(e, numNodes, mesh.isClosedLoop);

  // Initialize frame axes for all time steps from the shape
  rodPoseCurr.value.frameAxes = elementFrameAxes;
  rodPoseStageStart.value.frameAxes = elementFrameAxes;
  rodPosePrev.value.frameAxes = elementFrameAxes;

  // Velocity slices
  reg.emplace<CVelocitySlice<real, TimeStep::Current>>(e, totalDofs);
  reg.emplace<CVelocitySlice<real, TimeStep::Previous>>(e, totalDofs);
  reg.emplace<CVelocitySlice<real, TimeStep::StageStart>>(e, totalDofs);
  reg.emplace<CIntegrationVelocitySlices<DisplacementLayer::Default>>(e, totalDofs);

  // Set displacement references to point into rod pose data
  reg.emplace<CFinalDisplacementRef<TimeStep::Current>>(e, rodPoseCurr.value.displacements);
  reg.emplace<CFinalDisplacementRef<TimeStep::StageStart>>(
      e, rodPoseStageStart.value.displacements);

  // Boundary conditions
  reg.emplace<CDirichletBC<real>>(e);

  // External forces
  reg.emplace<CExternalForces>(e);

  // Rod body assembly uses an up-to-3-node stencil. Open-rod boundary stencils can be shorter, so
  // L2G stores raw connectivity plus stencil positions and then pads to fem::kNumRodStencilNodes *
  // kNumDofsPerNode DoFs for FEM assembly.
  CLocal2GlobalMap* l2g = &reg.emplace<CLocal2GlobalMap>(e);
  auto const connectivityAndStencil =
      GenerateRodConnectivityAndStencil(isize(mesh.nodes), mesh.isClosedLoop);
  auto const& connectivity = connectivityAndStencil.first;
  auto const& stencilGraph = connectivityAndStencil.second;
  l2g->InitializeFromElementNodeConnectivity(connectivity, kNumDofsPerNode);
  l2g->InitializeStencilIndices(stencilGraph);
  l2g->InitializePaddedIndices(fem::kNumRodStencilNodes * kNumDofsPerNode);

  // Build sparsity from the raw/unpadded L2G indices.
  auto& fullSparsity = reg.emplace<CFullSparsityPattern>(e, MakeSparsityGraph(*l2g, totalDofs));

  // Nodal based structure for FEM assembly. Uses padded stencil connectivity so every element
  // exposes fem::kNumRodStencilNodes nodes to the FEM assembler, while sparse indices are computed
  // from the original (unpadded) N-to-N graph to preserve the correct DResidual sparsity pattern.
  auto const& nbs = reg.emplace<CNodalBasedStructure>(
      e, BuildPaddedNodalBasedStructure<fem::kNumRodStencilNodes>(connectivity, stencilGraph));

  // DResidual matrix
  int const numRows = isize(fullSparsity.graph.GetPointers()) - 1;
  int const numCols = numRows;
  int const numNonZeros = fullSparsity.graph.NumTargets();
  DynamicArray<real> values(numNonZeros, 0_r);
  auto blockStructure = BlockedStructure<kNumDofsPerNode>(
      numCols,
      MakeConstSpan(fullSparsity.graph.GetPointers()),
      MakeConstSpan(fullSparsity.graph.GetTargets()));
  BlockSparseMatrix<real, kNumDofsPerNode> actorDRes(
      blockStructure.nBlockCols,
      std::move(blockStructure.ptr),
      std::move(blockStructure.ndIndices),
      std::move(values));

  // Actor SNLE data
  reg.emplace<CActorSnle>(
      e,
      std::move(actorDRes),
      // Use ILU0 actor preconditioner (O(n) and exact for rods).
      // NOTE: IC0 is also O(n) and exact, but (a) performance is slightly worse for pentadiagonal
      // systems and (b) it may break down with AssemblyParams::psdDRes = false.
      PreconditionerType::ILU0);

  // Non-linear solver convergence weights (lazily initialized).
  reg.emplace<CActorConvergenceWeights>(e);

  // FEM discretization for contact
  auto segmentDisc = CFemSegmentDiscretization::Create(
      params.contactElementType, shapePtr->GetNodes(), shapePtr->IsClosedLoop());
  int numCollidingSamples = segmentDisc.GetNumQuadPoints();

  std::shared_ptr<TriangularMesh const> contactSkinMesh;
  std::shared_ptr<RodSurfaceEmbeddingData const> contactSkinEmbedding;
  if (params.useVisualMeshContact) {
    MOCHI_ERROR_IF(
        !shapePtr->GetVisualMesh() || !shapePtr->GetRodVisualEmbedding(),
        error,
        "useVisualMeshContact requires a rod shape with visual mesh and embedding data.");
    MOCHI_ERROR_RETURN(error);
    contactSkinMesh = shapePtr->GetVisualMesh();
    contactSkinEmbedding = shapePtr->GetRodVisualEmbedding();
  }

  if (contactSkinMesh) {
    auto const& surfaceDisc = reg.emplace<CFemSurfaceDiscretization>(
        e,
        CFemSurfaceDiscretization::Create(params.visualMeshContactElementType, *contactSkinMesh));
    numCollidingSamples = surfaceDisc.GetNumQuadPoints();

    auto& contactSkin =
        reg.emplace<CRodContactSkin>(e, contactSkinMesh, std::move(contactSkinEmbedding));
    auto& skinningData = reg.emplace<CRodContactSkinningData>(e);
    rod::InitializeContactSkinningJacobian(contactSkin, mesh, skinningData);
    reg.emplace<TagRodSurfaceContact>(e);
    reg.emplace<CSkinnedContactSnle>(e);
    reg.emplace<TagSkinnedContact>(e);

    auto& deformedNodes = reg.emplace<CRodDeformedContactSkinNodes>(e);
    deformedNodes.positions.resize(
        static_cast<size_t>(kSpaceDim3) * contactSkinMesh->GetNumNodes());
  }

  // Contact. The point-cloud collider remains discretized on the rod centerline.
  EmplaceRodActorContact(reg, e, params, *shapePtr, segmentDisc, numCollidingSamples, error);
  MOCHI_ERROR_RETURN(error);

  // Centerline contact rods.
  if (!contactSkinMesh) {
    // Centerline contact assembly uses rod segments as the assembly elements. The contact-skin
    // path uses the skinned-contact subsystem instead.
    //
    // L2G has fixed 2-node x kNumDofsPerNode stride, matching CFemSegmentDiscretization. The NBS
    // uses the same segment element order, while sparse indices are computed against the actor's
    // full rod-stencil sparsity.
    DynamicArray<Int2> segmentConnectivity;
    segmentConnectivity.resize_noinit(numElements);
    for (int i = 0; i < numElements; ++i) {
      segmentConnectivity[i] = mesh.ElementNodes(i);
    }

    auto& contactL2g = reg.emplace<CContactLocal2GlobalMap>(e);
    contactL2g.InitializeFromElementNodeConnectivity(
        MakeConstSpan(segmentConnectivity), kNumDofsPerNode);
    reg.emplace<CContactNodalBasedStructure>(
        e,
        NodalBasedStructure(
            GraphFromRangeOfRanges<int, int>(MakeConstSpan(segmentConnectivity)), nbs.GetNToN()));

    // Emplace CFemSegmentDiscretization (after EmplaceRodActorContact has used it for collider
    // setup).
    reg.emplace<CFemSegmentDiscretization>(e, std::move(segmentDisc));
  }

  // Assemble nodal lumped masses; just add these directly to the DResidual diagonal during
  // problem assembly instead of precomputing a mass matrix with a larger sparsity pattern.
  DynamicArray<real> nodalMasses(numNodes);
  DynamicArray<real> elementRotationalInertias(numElements);
  for (int i = 0; i < numElements; i++) {
    Int2 const en = mesh.ElementNodes(i);
    real const elementLength = Norm(mesh.nodes[en[1]] - mesh.nodes[en[0]]);
    real const halfElementMass = 0.5_r * params.material.linearDensity * elementLength;
    nodalMasses[en[0]] += halfElementMass;
    nodalMasses[en[1]] += halfElementMass;
    elementRotationalInertias[i] = params.material.linearRotationalInertia * elementLength;
  }
  reg.emplace<CNodalMasses>(e, std::move(nodalMasses));
  reg.emplace<CElementRotationalInertias>(e, std::move(elementRotationalInertias));

  // Compute per-node reference curvature binormals from the rest configuration. For open rods,
  // endpoints have zero curvature (no interior angle defined at boundary nodes). For closed-loop
  // rods, every node has well-defined curvature via wraparound neighbors.
  DynamicArray<Real3> refCurvatureBinormals(numNodes, Real3{});
  rod::ComputeRodNodeCurvatureBinormals(
      mesh.nodes, mesh.isClosedLoop, MakeSpan(refCurvatureBinormals));
  reg.emplace<CReferenceNodeCurvatureBinormal>(e, std::move(refCurvatureBinormals));

  // Emplace the rod material parameters
  reg.emplace<CRodMaterialParams>(e, params.material);

  if (params.hasGravity) {
    reg.emplace<TagUseGravity>(e);
  }

  // Visual components are independent of the selected contact representation.
  if (shapePtr->GetVisualMesh() && shapePtr->GetRodVisualEmbedding()) {
    reg.emplace<CVisualMesh>(e, shapePtr->GetVisualMesh(), nullptr);
    reg.emplace<CRodVisualMeshEmbedding>(e, shapePtr->GetRodVisualEmbedding());
  }
}

namespace mochi::rod {

static void AddExternalForces(
    CExternalForces const& externalForces,
    TransformRT const& worldFromLocal,
    ColumnVectorView<real const> displacements,
    double* outObj,
    ColumnVectorView<real>* outRes) {
  if (!outObj && !outRes) {
    return;
  }
  int const numForces = isize(externalForces.dofs);
  VMatrix3x3r const worldFromLocalR = ToVMatrix3x3(worldFromLocal.GetRotation());
  for (int i = 0; i < numForces; i++) {
    int const dof = externalForces.dofs[i];
    int const component = dof % fem::kNumRodFields;
    if (component == fem::kRodThetaDofOffset) {
      // Treat external twisting forces as scalars in the local frame.
      real const force = externalForces.forces[i];
      if (outObj) {
        *outObj -= StaticCast<double>(displacements[dof] * force);
      }
      if (outRes) {
        (*outRes)[dof] -= force;
      }
    } else {
      int const nodeStartIndex = dof - component;
      i += deformable::details::AddTranslationalExternalForceEntry(
          externalForces,
          i,
          component,
          nodeStartIndex,
          worldFromLocalR,
          displacements,
          outObj,
          outRes);
    }
  }
}

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
    real dtStage) {
  constexpr int kBatchSize = kDefaultFemBatchSize;
  constexpr int kStencilDofs = fem::kNumRodStencilDofs;
  real const dtfi2 = 1_r / Sqr(dtStage);
  bool const hasMassDamping = materialParams.massDampingCoefficient > 0_r;
  bool const hasStiffnessDamping = materialParams.stiffnessDampingCoefficient > 0_r;
  real const massDampingScale =
      hasMassDamping ? materialParams.massDampingCoefficient / dtStage : 0_r;
  real const stiffnessDampingFactor =
      hasStiffnessDamping ? materialParams.stiffnessDampingCoefficient / dtStage : 0_r;

  return [= /*All copies are inexpensive*/](
             NdArray<int, kBatchSize> const& elemIndices,
             Span<int const> indicesFlat,
             fem::BatchRodVector<kBatchSize> const& displ,
             BatchDouble<kBatchSize>* outEnergy,
             fem::BatchRodVector<kBatchSize>* outRes,
             fem::BatchRodMatrix<kBatchSize>* outDRes,
             bool projectPsd) -> bool {
    using V = BatchReal<kBatchSize>;
    bool out = false;

    if (hasGravity && (outEnergy || outRes)) {
      out |= fem::RodGravity<kBatchSize>(
          nodalMasses, displ, elemIndices, indicesFlat, outEnergy, outRes, gravity);
    }

    // --- Gather stage-start displacement once for mass and/or stiffness damping ---
    //
    // Stiffness damping needs the full stencil; mass damping needs only stencil node 0, whose
    // kNumRodFields DoFs are the leading entries of that stencil. When both are active we gather
    // the superset once and mass damping reuses the leading sub-span instead of re-reading node 0.
    // The stiffness stage-start also feeds the stress dresidual, so it is gathered on every path;
    // mass damping reads its target only in the energy/residual branch. (The inertia predicted
    // target, stage-start + dt*velocity, is different data and is gathered separately below.)
    NdArray<V, fem::kNumRodStencilDofs> stageStartDispBatch MOCHI_NO_INIT;
    int const numStageStartNodes = hasStiffnessDamping
        ? fem::kNumRodStencilNodes
        : ((hasMassDamping && (outEnergy || outRes)) ? 1 : 0);
    if (numStageStartNodes > 0) {
      alignas(alignof(V)) real staging[fem::kNumRodStencilDofs][V::kSize]{};
      for (int b = 0; b < kBatchSize; ++b) {
        for (int n = 0; n < numStageStartNodes; ++n) {
          int const node = fem::details::RodStencilNodeIndex(indicesFlat, elemIndices[b], n);
          int const globalDof = node * fem::kNumRodFields;
          for (int f = 0; f < fem::kNumRodFields; ++f) {
            staging[n * fem::kNumRodFields + f][b] = stageStartDispl[globalDof + f];
          }
        }
      }
      for (int dof = 0; dof < numStageStartNodes * fem::kNumRodFields; ++dof) {
        stageStartDispBatch[dof] = Load<V>(staging[dof]);
      }
    }

    // --- Inertia on stencil node 0 ---
    // Left uninitialized: RodInertia reads the target only in its energy/residual branch, which
    // populates this below. The dresidual-only path never reads it, so the gather is skipped.
    NdArray<V, fem::kNumRodFields> batchPredTarget MOCHI_NO_INIT;
    if (outEnergy || outRes) {
      alignas(alignof(V)) real staging[fem::kNumRodFields][V::kSize]{};
      for (int b = 0; b < kBatchSize; ++b) {
        int const globalDof = indicesFlat[elemIndices[b] * kStencilDofs];
        for (int f = 0; f < fem::kNumRodFields; ++f) {
          staging[f][b] = stageStartDispl[globalDof + f] + dtStage * stageStartVel[globalDof + f];
        }
      }
      for (int f = 0; f < fem::kNumRodFields; ++f) {
        batchPredTarget[f] = Load<V>(staging[f]);
      }
    }

    out |= fem::RodInertia<kBatchSize>(
        nodalMasses,
        elementRotationalInertias,
        displ,
        batchPredTarget,
        elemIndices,
        indicesFlat,
        outEnergy,
        outRes,
        outDRes,
        dtfi2);

    if (hasMassDamping) {
      // Mass-proportional damping C = α·M, added as a second inertia term whose target is the
      // stage-start position with zero velocity extrapolation. This contributes residual
      // (α/dt)·M·(x − d_stageStart), the matching energy, and the diagonal (α/dt)·M. Unlike
      // shell (which folds inertia + mass damping into a precomputed global mass matrix), rods
      // have no such matrix, so this call also emits the DResidual diagonal.
      // Reuse the leading kNumRodFields DoFs (stencil node 0) of the shared stage-start gather
      // above instead of re-reading them from memory.
      NdArray<V, fem::kNumRodFields> massDampingTarget MOCHI_NO_INIT;
      if (outEnergy || outRes) {
        for (int f = 0; f < fem::kNumRodFields; ++f) {
          massDampingTarget[f] = stageStartDispBatch[f];
        }
      }

      out |= fem::RodInertia<kBatchSize>(
          nodalMasses,
          elementRotationalInertias,
          displ,
          massDampingTarget,
          elemIndices,
          indicesFlat,
          outEnergy,
          outRes,
          outDRes,
          massDampingScale);
    }

    // --- Stress (axial + bend-twist) on the full stencil ---
    // Stiffness damping consumes the shared stage-start gather above (full stencil). Collapsed
    // boundary stencils stay consistent via the same RodStencilNodeIndex mapping used by the
    // kernels; inactive lanes are masked inside the kernels.
    auto const* stageStartDispPtr = hasStiffnessDamping ? &stageStartDispBatch : nullptr;

    out |= fem::RodAxialStress<kBatchSize>(
        meshNodes,
        displ,
        elemIndices,
        indicesFlat,
        outEnergy,
        outRes,
        outDRes,
        materialParams.axialStiffness,
        projectPsd,
        stiffnessDampingFactor,
        stageStartDispPtr);

    if ((Max(materialParams.flexuralStiffness) > 0_r) ||
        (materialParams.torsionalStiffness > 0_r)) {
      out |= fem::RodBendTwistStress<kBatchSize>(
          meshNodes,
          frameAxes,
          referenceAxes,
          displ,
          elemIndices,
          indicesFlat,
          outEnergy,
          outRes,
          outDRes,
          materialParams.flexuralStiffness,
          materialParams.torsionalStiffness,
          stiffnessDampingFactor,
          stageStartDispPtr,
          hasStiffnessDamping ? stageStartFrameAxes : Span<Real3 const>{});
    }

    return out;
  };
}

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
    CNodalBasedStructure const& nbs) {
  MOCHI_PROFILE_SCOPE();
  bool const hasGravity = hasGravityTag;

  // Clear SNLE data
  if (params.assemObj) {
    outActorSnle.objective = 0.0;
  }
  if (params.assemRes) {
    outActorSnle.fullResidual.SetZero();
  }
  if (params.assemDRes) {
    MOCHI_PROFILE_SCOPE_N("InitRodDResidual");
    auto const numValues = GetNumValues(outActorSnle.fullDResidual);
    constexpr int kMinValuesPerTask = 150000;
    ParallelForRange(
        "InitRodDResidual", 0, numValues, kMinValuesPerTask, INT_MAX, [&](int rBegin, int rEnd) {
          MOCHI_ASSERT_VERBOSE(rBegin >= 0 && rBegin <= rEnd && rEnd <= numValues);
          ColumnVectorView<real> dresValues = AsView(GetValues(outActorSnle.fullDResidual));
          // For rods, we don't pre-scale with mass matrix, so just zero it out
          dresValues.MiddleRows(rBegin, rEnd - rBegin).SetZero();
        });
  }

  if (params.assemObj || params.assemRes || params.assemDRes) {
    // Assemble volume terms.
    auto const gravity =
        ToReal3(rootTransform.worldFromLocal.TransformDirectionInverse(sceneGravity->accel));

    auto bodyOp = MakeBatchedBodyOp(
        polylineMesh.nodes,
        hasGravity,
        MakeConstSpan(nodalMasses.values),
        MakeConstSpan(elementRotationalInertias.values),
        MakeConstSpan(currPose.value.frameAxes),
        referenceAxes.axes,
        materialParams,
        gravity,
        stageStartPose.value.displacements,
        stageStartVel.value,
        MakeConstSpan(stageStartPose.value.frameAxes),
        intState.dtStage);

    AssembleObjResDRes<fem::RodStencilElement, fem::kNumRodFields>(
        l2g,
        nbs,
        bodyOp,
        currPose.value.displacements,
        AssemblyResults<real>{
            .outObj = &outActorSnle.objective,
            .outRes = AsView(outActorSnle.fullResidual),
            .outDRes = AsView(outActorSnle.fullDResidual),
            .params = params});
  }

  if (!externalForces.Empty()) {
    ColumnVectorView<real> resView = AsView(outActorSnle.fullResidual);
    AddExternalForces(
        externalForces,
        rootTransform.worldFromLocal,
        AsConstView(currPose.value.displacements),
        params.assemObj ? &outActorSnle.objective : nullptr,
        params.assemRes ? &resView : nullptr);
  }
}

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
    CActorSnle& outActorSnle) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(params.assemObj || params.assemRes || params.assemDRes, "Must assemble something");

  if (collisions.empty()) {
    // No contacts.
    outResponse.Clear();
    return;
  }

  // Compute contact response per sample.
  deformable::ComputeAsyncContactResponse<CFemSegmentDiscretization, fem::kNumRodFields>(
      reg,
      e,
      simParams->experimentalEval,
      queryActiveContacts,
      femDisc,
      samples,
      colliderInfo,
      collisions,
      intState,
      rootTransform,
      params,
      outResponse);

  if (outResponse.Empty()) {
    // No contacts.
    return;
  }

  AssemblyResults<real> results{
      .outObj = &outActorSnle.objective,
      .outRes = AsView(outActorSnle.fullResidual),
      .outDRes = AsView(outActorSnle.fullDResidual),
      .params = params};

  // Assemble contact forces.
  femDisc.Visit([&](auto const& disc) {
    using DiscT = std::decay_t<decltype(disc)>;
    using ElementT = typename DiscT::ElementT;

    AssemblyActiveSubset const activeSubset = outResponse.ViewActiveContactElementSubset();

    auto boundaryOp = deformable::MakeBatchedBoundaryOp<ElementT, fem::kNumRodFields>(
        MakeConstSpan(disc.femElements), outResponse, Span<real const>{});

    AssembleObjResDRes<ElementT, fem::kNumRodFields>(
        contactL2g, contactNbs, boundaryOp, results, activeSubset);
  });
}

} // namespace mochi::rod
