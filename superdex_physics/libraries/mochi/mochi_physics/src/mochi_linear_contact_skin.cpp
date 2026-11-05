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

#include "mochi_linear_contact_skin.h"

#include "mochi_deformable.h"
#include "mochi_discretization_functions.h"

#include <mochi_core/element_operations/batched_element_utils.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/spmat_utils.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

using namespace mochi;

// Evaluates the active contact-skin node positions for the given physics-mesh displacements.
static void ComputeNodePositions(
    CContactSkinMesh const& contactSkin,
    CDeformedContactSkinNodes const& deformedNodes,
    ColumnVectorView<real const> displacements,
    Span<Real3> outPositions) {
  MOCHI_ASSERT_VERBOSE(contactSkin.embedding != nullptr, "Contact skin requires an embedding.");
  UpdateLinearEmbeddedNodePositionsFromDisplacements(
      *contactSkin.embedding,
      Unflatten<Real3 const>(MakeConstSpan(deformedNodes.referencePositions)),
      Unflatten<Real3 const>(displacements.GetConstSpan()),
      contactSkin.mesh->GetActiveNodes(),
      outPositions);
}

void mochi::linear_contact_skin::EmplaceComponents(
    entt::registry& reg,
    entt::entity e,
    CFemSurfaceDiscretization const& surfaceDisc,
    TriangularMesh const& contactSkin,
    LinearMeshEmbedding const& embedding,
    Span<Real3 const> physicsNodeCoordinates) {
  auto& skinningData = reg.emplace<CContactSkinningData>(e);
  InitializeLinearContactSkinningJacobian(
      embedding, isize(physicsNodeCoordinates), contactSkin.GetActiveNodes(), skinningData);
  InitializeElementMap(surfaceDisc, skinningData, reg.emplace<CLinearContactSkinElementMap>(e));
  auto& deformedNodes = reg.emplace<CDeformedContactSkinNodes>(e);
  deformedNodes.referencePositions.resize_noinit(kSpaceDim3 * contactSkin.GetNumNodes());
  embedding.Update(
      physicsNodeCoordinates, Unflatten<Real3>(MakeSpan(deformedNodes.referencePositions)));
  deformedNodes.positions.resize(kSpaceDim3 * contactSkin.GetNumActiveNodes());
  reg.emplace<TagUseDeformableContactSkin>(e);
  reg.emplace<CSkinnedContactSnle>(e);
  reg.emplace<TagSkinnedContact>(e);
}

void mochi::linear_contact_skin::InitializeElementMap(
    CFemSurfaceDiscretization const& surfaceDisc,
    CContactSkinningData const& skinningData,
    CLinearContactSkinElementMap& outMap) {
  MOCHI_PROFILE_SCOPE();

  DynamicArray<int> rowPointers;
  DynamicArray<int> physicsNodeIndices;
  DynamicArray<Real3> skinNodeWeights;
  surfaceDisc.Visit([&](auto const& discImpl) {
    using DiscT = std::decay_t<decltype(discImpl)>;
    using ElementT = typename DiscT::ElementT;
    static_assert(ElementT::kNumNodes == 3, "Linear contact skins require triangular surfaces.");

    int const numElements = isize(discImpl.femElements);
    outMap.samplesPerElement = ElementT::kNumQuadPoints;
    rowPointers.reserve(numElements + 1);
    rowPointers.push_back(0);

    DynamicArray<std::pair<int, Real3>> nodeWeights;
    for (int elementIndex = 0; elementIndex < numElements; ++elementIndex) {
      auto const surfaceNodes = discImpl.femElements[elementIndex].Nodes();
      nodeWeights.clear();
      for (int sourceNode = 0; sourceNode < ElementT::kNumNodes; ++sourceNode) {
        int const skinNode = surfaceNodes[sourceNode];
        MOCHI_ASSERT_VERBOSE(
            skinNode >= 0 && skinNode < skinningData.jacobian.Rows(),
            "Contact-skin node is outside the skinning Jacobian.");

        auto const columns = skinningData.jacobian.Indices(skinNode);
        auto const values = skinningData.jacobian.Values(skinNode);
        MOCHI_ASSERT_VERBOSE(
            columns.size() % kSpaceDim3 == 0, "Linear skinning Jacobian must be W tensor I_3.");
        for (int entry = 0; entry < isize(columns); entry += kSpaceDim3) {
          int const physicsNode = columns[entry] / kSpaceDim3;
          real const weight = values[entry][0];
          MOCHI_ASSERT_VERBOSE(IsFinite(weight), "Skin-node weight must be finite.");
          for (int component = 0; component < kSpaceDim3; ++component) {
            MOCHI_ASSERT_VERBOSE(
                columns[entry + component] == kSpaceDim3 * physicsNode + component,
                "Linear skinning Jacobian must be W tensor I_3.");
            for (int derivativeComponent = 0; derivativeComponent < kSpaceDim3;
                 ++derivativeComponent) {
              MOCHI_ASSERT_VERBOSE(
                  values[entry + component][derivativeComponent] ==
                      (derivativeComponent == component ? weight : 0_r),
                  "Linear skinning Jacobian must be W tensor I_3.");
            }
          }
          Real3 weights{};
          weights[sourceNode] = weight;
          nodeWeights.emplace_back(physicsNode, weights);
        }
      }

      // Merge the influences of each physics node on the element's skin nodes.
      std::ranges::sort(nodeWeights, {}, &std::pair<int, Real3>::first);
      int influence = 0;
      while (influence < isize(nodeWeights)) {
        int const physicsNode = nodeWeights[influence].first;
        Real3 weights{};
        do {
          weights += nodeWeights[influence].second;
          ++influence;
        } while (influence < isize(nodeWeights) && nodeWeights[influence].first == physicsNode);
        physicsNodeIndices.push_back(physicsNode);
        skinNodeWeights.push_back(weights);
      }
      MOCHI_ASSERT_VERBOSE(
          isize(physicsNodeIndices) > rowPointers.back(),
          "Contact-skin element must depend on at least one physics node.");
      rowPointers.push_back(isize(physicsNodeIndices));
    }
  });

  outMap.weights = SparseMatrix<Real3>(
      skinningData.jacobian.Cols() / kSpaceDim3,
      std::move(rowPointers),
      std::move(physicsNodeIndices),
      std::move(skinNodeWeights));
}

template <TimeStep kTimeStep>
void mochi::linear_contact_skin::UpdateSamples(
    ecs::Excluded<TagRodActor>,
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    CFemSurfaceDiscretization const& surfaceDisc,
    CDeformedContactSkinNodes const& deformedNodes,
    CContactSamples<kTimeStep>& outSamples) {
  MOCHI_PROFILE_SCOPE();
  UpdateCollisionSamplePositionsFromNodePositions(
      Unflatten<Real3 const>(MakeConstSpan(deformedNodes.positions)), surfaceDisc, outSamples);
}

template void mochi::linear_contact_skin::UpdateSamples<TimeStep::Current>(
    ecs::Excluded<TagRodActor>,
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    CFemSurfaceDiscretization const&,
    CDeformedContactSkinNodes const&,
    CContactSamples<TimeStep::Current>&);
template void mochi::linear_contact_skin::UpdateSamples<TimeStep::StageStart>(
    ecs::Excluded<TagRodActor>,
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    CFemSurfaceDiscretization const&,
    CDeformedContactSkinNodes const&,
    CContactSamples<TimeStep::StageStart>&);

template <TimeStep kTimeStep>
void mochi::linear_contact_skin::UpdatePositionsAndSamples(
    ecs::Excluded<TagRodActor>,
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    CContactSkinMesh const& contactSkin,
    CFinalDisplacementRef<kTimeStep> const& displacements,
    CFemSurfaceDiscretization const& surfaceDisc,
    CDeformedContactSkinNodes& deformedNodes,
    CContactSamples<kTimeStep>& outSamples) {
  MOCHI_PROFILE_SCOPE();
  ComputeNodePositions(
      contactSkin,
      deformedNodes,
      displacements.value,
      Unflatten<Real3>(MakeSpan(deformedNodes.positions)));
  UpdateSamples<kTimeStep>({}, {}, surfaceDisc, deformedNodes, outSamples);
}

template void mochi::linear_contact_skin::UpdatePositionsAndSamples<TimeStep::Current>(
    ecs::Excluded<TagRodActor>,
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    CContactSkinMesh const&,
    CFinalDisplacementRef<TimeStep::Current> const&,
    CFemSurfaceDiscretization const&,
    CDeformedContactSkinNodes&,
    CContactSamples<TimeStep::Current>&);
template void mochi::linear_contact_skin::UpdatePositionsAndSamples<TimeStep::StageStart>(
    ecs::Excluded<TagRodActor>,
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    CContactSkinMesh const&,
    CFinalDisplacementRef<TimeStep::StageStart> const&,
    CFemSurfaceDiscretization const&,
    CDeformedContactSkinNodes&,
    CContactSamples<TimeStep::StageStart>&);

template <bool kIsSync>
void mochi::linear_contact_skin::SetupActiveCollisionNormals(
    ecs::Excluded<TagShellActor, TagRodActor>,
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CContactSkinMesh const& contactSkin,
    CDeformedContactSkinNodes const& deformedNodes,
    CFemSurfaceDiscretization const& surfaceDisc,
    CFinalDisplacementRef<TimeStep::StageStart> const& stageStartDispl,
    CRootTransform const& transform,
    CActiveCollisions<kIsSync, TimeStep::Current>& activeCollisions) {
  MOCHI_PROFILE_SCOPE();
  if (activeCollisions.empty()) {
    return;
  }

  bool const explicitNormals = simParams->experimentalEval.explicitNormals;
  int constexpr kNumContactSkinNodesEstimate = 2048;
  MOCHI_FILO_STACK_ALLOCATOR(allocator, kNumContactSkinNodesEstimate * sizeof(Real3));
  DynamicArray<Real3> stageStartPositions(&allocator);
  Span<Real3 const> nodePositions = Unflatten<Real3 const>(MakeConstSpan(deformedNodes.positions));
  if (explicitNormals) {
    stageStartPositions.resize_noinit(contactSkin.mesh->GetNumActiveNodes());
    ComputeNodePositions(
        contactSkin, deformedNodes, stageStartDispl.value, MakeSpan(stageStartPositions));
    nodePositions = MakeConstSpan(stageStartPositions);
  }

  deformable::details::SetupActiveCollisionNormalsFromElements(
      explicitNormals,
      transform,
      surfaceDisc,
      activeCollisions,
      [&](auto const& disc, int elementIndex, int /*quadPointIndex*/) -> Vec4r {
        auto const nodes = disc.femElements[elementIndex].Nodes();
        Vec4r const p0 = ToSimd(nodePositions[nodes[0]]);
        Vec4r const p1 = ToSimd(nodePositions[nodes[1]]);
        Vec4r const p2 = ToSimd(nodePositions[nodes[2]]);
        return Normalize<3>(Cross3(p1 - p0, p2 - p0));
      });
}

template void mochi::linear_contact_skin::SetupActiveCollisionNormals</*kIsSync*/ false>(
    ecs::Excluded<TagShellActor, TagRodActor>,
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    ecs::CtxGlobal<CSimulationParams const>,
    CContactSkinMesh const&,
    CDeformedContactSkinNodes const&,
    CFemSurfaceDiscretization const&,
    CFinalDisplacementRef<TimeStep::StageStart> const&,
    CRootTransform const&,
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current>&);
template void mochi::linear_contact_skin::SetupActiveCollisionNormals</*kIsSync*/ true>(
    ecs::Excluded<TagShellActor, TagRodActor>,
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    ecs::CtxGlobal<CSimulationParams const>,
    CContactSkinMesh const&,
    CDeformedContactSkinNodes const&,
    CFemSurfaceDiscretization const&,
    CFinalDisplacementRef<TimeStep::StageStart> const&,
    CRootTransform const&,
    CActiveCollisions</*kIsSync*/ true, TimeStep::Current>&);

template <TimeStep kTimeStep>
void mochi::linear_contact_skin::UpdateBounds(
    ecs::Excluded<TagRodActor>,
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    ecs::OptionalTag<TagSoftActor> isSoft,
    CSimplicialMesh const& physicsMesh,
    CContactSkinMesh const& contactSkin,
    CFinalDisplacementRef<kTimeStep> const& displacements,
    CColliderInfo const& collider,
    CPointCloudColliderParams const* pointCloudColliderParams,
    CDeformedContactSkinNodes& deformedNodes,
    CBoundingVolume& outBounds) {
  MOCHI_PROFILE_SCOPE();
  auto const positions = Unflatten<Real3>(MakeSpan(deformedNodes.positions));
  ComputeNodePositions(contactSkin, deformedNodes, displacements.value, positions);
  Aabb bounds = CalcAabb(positions);
  if (collider.type != ColliderType::None) {
    auto const nodeCoordinates = physicsMesh.mesh->GetNodeCoordinates();
    auto const nodeDisplacements = Unflatten<Real3 const>(displacements.value.GetConstSpan());
    // Interior tetrahedral nodes cannot extend the bounds. Every shell node is on the surface.
    Aabb physicsBounds = isSoft
        ? CalcAabbWithDisplacementsAndSortedIndices(
              nodeCoordinates, nodeDisplacements, physicsMesh.mesh->GetBoundaryNodes())
        : CalcAabbWithDisplacements(nodeCoordinates, nodeDisplacements);
    if (pointCloudColliderParams) {
      physicsBounds = GetAabb(ExpandShape(GetObb(physicsBounds), pointCloudColliderParams->radius));
    }
    bounds = GetAabb(bounds, physicsBounds);
  }
  outBounds.localShape = GetObb(bounds);
}

template void mochi::linear_contact_skin::UpdateBounds<TimeStep::Current>(
    ecs::Excluded<TagRodActor>,
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    ecs::OptionalTag<TagSoftActor>,
    CSimplicialMesh const&,
    CContactSkinMesh const&,
    CFinalDisplacementRef<TimeStep::Current> const&,
    CColliderInfo const&,
    CPointCloudColliderParams const*,
    CDeformedContactSkinNodes&,
    CBoundingVolume&);
template void mochi::linear_contact_skin::UpdateBounds<TimeStep::StageStart>(
    ecs::Excluded<TagRodActor>,
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    ecs::OptionalTag<TagSoftActor>,
    CSimplicialMesh const&,
    CContactSkinMesh const&,
    CFinalDisplacementRef<TimeStep::StageStart> const&,
    CColliderInfo const&,
    CPointCloudColliderParams const*,
    CDeformedContactSkinNodes&,
    CBoundingVolume&);

namespace {

// Returns the rotation from the colliding actor's local frame to the current collider frame of a
// contact, as a function of the contact index. A collider frame shared by all contacts is composed
// only once.
auto MakeColliderFromCollidingRotation(
    ContactDetectionResult const& contactQuery,
    VMatrix3x3r const& worldFromCollidingRotation) {
  auto const& colliderFromWorld = contactQuery.jacColliderFromWorld;
  MOCHI_ASSERT_VERBOSE(
      colliderFromWorld.size() == 1 ||
          colliderFromWorld.size() == contactQuery.sampleIndices.size(),
      "Collider transform must be shared or specified per contact sample.");
  VMatrix3x3r const shared = Dot3x3(colliderFromWorld[0], worldFromCollidingRotation);
  return [&colliderFromWorld, worldFromCollidingRotation, shared](int contactIndex) {
    return colliderFromWorld.size() == 1
        ? shared
        : Dot3x3(colliderFromWorld[contactIndex], worldFromCollidingRotation);
  };
}

// Index groups that scatter the rows of an element's projected block, ordered as its physics
// nodes, to the actor DoFs at dofOffset.
void MakeOffsetElementIndexGroups(
    CLinearContactSkinElementMap const& elementMap,
    int elementIndex,
    int dofOffset,
    DynamicArray<IndexGroup>& outGroups) {
  auto const physicsNodes = elementMap.PhysicsNodeIndices(elementIndex);
  outGroups.clear();
  MOCHI_ASSERT_VERBOSE(outGroups.capacity() >= physicsNodes.size());
  for (int node = 0; node < isize(physicsNodes); ++node) {
    outGroups.push_back(
        {.src = kSpaceDim3 * node,
         .dst = dofOffset + kSpaceDim3 * physicsNodes[node],
         .count = kSpaceDim3});
  }
}

// Applies an element's skin-node weights, tensor I_3, to its skin DoFs: calls
// out(node, component, sum_k weights[node][k] * in(kSpaceDim3 * k + component)) for each physics
// node and spatial component.
template <int kNumSkinNodes, class InFn, class OutFn>
MOCHI_FORCE_INLINE void
MapSkinToPhysicsDofs(Span<Real3 const> weights, InFn const& in, OutFn const& out) {
  for (int node = 0; node < isize(weights); ++node) {
    for (int component = 0; component < kSpaceDim3; ++component) {
      real mapped = 0_r;
      for (int skinNode = 0; skinNode < kNumSkinNodes; ++skinNode) {
        mapped += weights[node][skinNode] * in(kSpaceDim3 * skinNode + component);
      }
      out(node, component, mapped);
    }
  }
}

// Assembles the active skin elements with batchedElementOp and projects each element's
// contribution onto the physics DoFs through the element map.
template <class ElementT, int kBatchSize = kDefaultFemBatchSize>
void AssembleAndProjectLinearContactSkin(
    [[maybe_unused]] Span<ElementT const> elements,
    CLinearContactSkinElementMap const& elementMap,
    NoDispElOpFnType<ElementT, kSpaceDim3, kBatchSize> const& batchedElementOp,
    AssemblyResults<real> results,
    Span<int const> activeElementIndices,
    int dofOffset,
    Allocator* allocator) {
  static_assert(ElementT::kNumNodes == 3, "Linear contact-skin assembly requires triangles.");
  static constexpr int kSkinDofs = ElementT::kNumNodes * kSpaceDim3;
  // Assemble only the outputs that the caller provides.
  bool const assemObj = results.outObj != nullptr;
  bool const assemRes = !results.outRes.empty();
  bool const assemDRes = GetNumValues(results.outDRes) > 0;
  MOCHI_ASSERT_VERBOSE(assemObj || assemRes || assemDRes, "Must assemble something.");
  MOCHI_ASSERT_VERBOSE(
      elementMap.NumElements() == isize(elements) &&
          elementMap.samplesPerElement == ElementT::kNumQuadPoints,
      "Linear contact-skin map does not match the surface discretization.");
  int const numActive = isize(activeElementIndices);

  DynamicArray<real> projectedDresValues(allocator);
  DynamicArray<IndexGroup> offsetGroups(allocator);
  DynamicArray<real> leftProjectedDresValues(allocator);
  if (assemDRes) {
    int maxPhysicsNodes = 0;
    for (int elementIndex : activeElementIndices) {
      maxPhysicsNodes = Max(maxPhysicsNodes, isize(elementMap.PhysicsNodeIndices(elementIndex)));
    }
    int const maxProjectedDofs = kSpaceDim3 * maxPhysicsNodes;
    projectedDresValues.reserve(maxProjectedDofs * maxProjectedDofs);
    offsetGroups.reserve(maxPhysicsNodes);
    leftProjectedDresValues.reserve(maxProjectedDofs * kSkinDofs);
  }

  for (int batchBegin = 0; batchBegin < numActive; batchBegin += kBatchSize) {
    int const actualBatchSize = Min(kBatchSize, numActive - batchBegin);
    NdArray<int, kBatchSize> elementIndices MOCHI_NO_INIT;
    for (int lane = 0; lane < kBatchSize; ++lane) {
      elementIndices[lane] = activeElementIndices[batchBegin + Min(lane, actualBatchSize - 1)];
    }

    BatchDouble<kBatchSize> elementEnergy MOCHI_NO_INIT;
    fem::BatchElementVector<kBatchSize, ElementT, kSpaceDim3> elementResidual MOCHI_NO_INIT;
    fem::BatchElementMatrix<kBatchSize, ElementT, kSpaceDim3> elementDResidual MOCHI_NO_INIT;
    if (assemObj) {
      elementEnergy = {};
    }
    if (assemRes) {
      elementResidual = {};
    }
    if (assemDRes) {
      elementDResidual = {};
    }
    bool const hasOutput = batchedElementOp(
        elementIndices,
        {},
        assemObj ? &elementEnergy : nullptr,
        assemRes ? &elementResidual : nullptr,
        assemDRes ? &elementDResidual : nullptr,
        results.params.psdDRes);
    if (!hasOutput) {
      continue;
    }

    for (int lane = 0; lane < actualBatchSize; ++lane) {
      int const elementIndex = elementIndices[lane];
      auto const physicsNodes = elementMap.PhysicsNodeIndices(elementIndex);
      auto const weights = elementMap.SkinNodeWeights(elementIndex);
      MOCHI_ASSERT_VERBOSE(physicsNodes.size() == weights.size());

      if (assemObj) {
        *results.outObj += elementEnergy[lane];
      }

      if (assemRes) {
        MapSkinToPhysicsDofs<ElementT::kNumNodes>(
            weights,
            [&](int skinDof) { return elementResidual[skinDof][lane]; },
            [&](int node, int component, real value) {
              results.outRes[dofOffset + kSpaceDim3 * physicsNodes[node] + component] += value;
            });
      }

      if (assemDRes) {
        // Project W^T K W as (W^T K) W, contracting skin rows first and skin columns second.
        int const projectedDofs = kSpaceDim3 * isize(physicsNodes);
        leftProjectedDresValues.resize_noinit(projectedDofs * kSkinDofs);
        RowMatrixView<real, krylov::kDynamic, kSkinDofs> leftProjectedDres(
            leftProjectedDresValues.data(), projectedDofs, kSkinDofs);
        for (int localCol = 0; localCol < kSkinDofs; ++localCol) {
          MapSkinToPhysicsDofs<ElementT::kNumNodes>(
              weights,
              [&](int skinDof) { return elementDResidual[skinDof * kSkinDofs + localCol][lane]; },
              [&](int node, int component, real value) {
                leftProjectedDres(kSpaceDim3 * node + component, localCol) = value;
              });
        }
        projectedDresValues.resize_noinit(projectedDofs * projectedDofs);
        RowMatrixView<real> projectedDres(projectedDresValues.data(), projectedDofs, projectedDofs);
        for (int row = 0; row < projectedDofs; ++row) {
          MapSkinToPhysicsDofs<ElementT::kNumNodes>(
              weights,
              [&](int skinDof) { return leftProjectedDres(row, skinDof); },
              [&](int node, int component, real value) {
                projectedDres(row, kSpaceDim3 * node + component) = value;
              });
        }
        MakeOffsetElementIndexGroups(elementMap, elementIndex, dofOffset, offsetGroups);
        MatAddSubBlocks(
            results.outDRes,
            MakeConstSpan(offsetGroups),
            MakeConstSpan(offsetGroups),
            RowMatrixView<real const>(projectedDres));
      }
    }
  }
}

// Per-sample contact responses for one contact range, accumulated by sample index and looked up
// through an open-addressing hash table sized by contact count rather than by sample count. Add()
// must be called with non-decreasing sample indices.
class LinearContactSkinResponseRange {
 public:
  static constexpr int kInvalidIndex = kSentinelIndex;
  // Keeps the lookup table load factor at most 1/2.
  static constexpr size_t kLookupCapacityFactor = 2;

  LinearContactSkinResponseRange(
      int capacity,
      int samplesPerElement,
      bool storeEnergy,
      bool storeGradient,
      bool storeHessian,
      Allocator* allocator)
      : _samplesPerElement(samplesPerElement),
        _sampleIndices(allocator),
        _lookupSampleIndices(allocator),
        _lookupResponseIndices(allocator),
        _energies(allocator),
        _gradients(allocator),
        _hessians(allocator),
        _activeElementIndices(allocator) {
    MOCHI_ASSERT_VERBOSE(capacity >= 0, "Response capacity must be non-negative.");
    MOCHI_ASSERT_VERBOSE(samplesPerElement > 0, "Samples per element must be positive.");
    _sampleIndices.reserve(capacity);
    size_t const lookupCapacity =
        NextPowerOfTwo(kLookupCapacityFactor * static_cast<size_t>(capacity));
    int const lookupBits = std::countr_zero(lookupCapacity);
    MOCHI_ASSERT_VERBOSE(lookupBits <= 32, "Lookup table exceeds the 32-bit hash range.");
    _lookupMask = lookupCapacity - 1;
    _lookupShift = 32 - lookupBits;
    _lookupSampleIndices.resize(lookupCapacity, kInvalidIndex);
    _lookupResponseIndices.resize_noinit(lookupCapacity);
    if (storeEnergy) {
      _energies.reserve(capacity);
    }
    if (storeGradient) {
      _gradients.reserve(capacity);
    }
    if (storeHessian) {
      _hessians.reserve(capacity);
    }
    _activeElementIndices.reserve(capacity);
  }

  void
  Add(int sampleIndex, double const* energy, Real3 const* gradient, VMatrix3x3r const* hessian) {
    MOCHI_ASSERT_VERBOSE(
        _sampleIndices.empty() || _sampleIndices.back() <= sampleIndex,
        "Responses must be added in non-decreasing sample order.");
    if (!_sampleIndices.empty() && _sampleIndices.back() == sampleIndex) {
      int const responseIndex = isize(_sampleIndices) - 1;
      if (energy) {
        _energies[responseIndex] += *energy;
      }
      if (gradient) {
        _gradients[responseIndex] += *gradient;
      }
      if (hessian) {
        _hessians[responseIndex] += ToNdArray3x3(*hessian);
      }
      return;
    }

    int const responseIndex = isize(_sampleIndices);
    size_t const lookupSlot = FindLookupSlot(sampleIndex);
    MOCHI_ASSERT_VERBOSE(
        _lookupSampleIndices[lookupSlot] == kInvalidIndex,
        "New response sample must have an empty lookup slot.");
    _lookupSampleIndices[lookupSlot] = sampleIndex;
    _lookupResponseIndices[lookupSlot] = responseIndex;
    _sampleIndices.push_back(sampleIndex);
    if (energy) {
      _energies.push_back(*energy);
    }
    if (gradient) {
      _gradients.push_back(*gradient);
    }
    if (hessian) {
      _hessians.push_back(ToNdArray3x3(*hessian));
    }
    int const elementIndex = sampleIndex / _samplesPerElement;
    if (_activeElementIndices.empty() || _activeElementIndices.back() != elementIndex) {
      _activeElementIndices.push_back(elementIndex);
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE int GetResponseIndexFromSampleIndex(int sampleIndex) const {
    size_t const lookupSlot = FindLookupSlot(sampleIndex);
    return _lookupSampleIndices[lookupSlot] == sampleIndex ? _lookupResponseIndices[lookupSlot]
                                                           : kInvalidIndex;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE double GetEnergy(int responseIndex) const {
    return _energies[responseIndex];
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Real3 const& GetGradient(int responseIndex) const {
    return _gradients[responseIndex];
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Matrix3x3r const& GetHessian(int responseIndex) const {
    return _hessians[responseIndex];
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Span<int const> ActiveElementIndices() const {
    return MakeConstSpan(_activeElementIndices);
  }

 private:
  [[nodiscard]] MOCHI_FORCE_INLINE size_t FindLookupSlot(int sampleIndex) const {
    MOCHI_ASSERT_VERBOSE(sampleIndex >= 0, "Sample index must be non-negative.");
    // Fibonacci hashing: the top bits of the product spread consecutive indices across slots.
    uint32_t const hash = static_cast<uint32_t>(sampleIndex) * kGoldenRatio32;
    auto lookupSlot = static_cast<size_t>(uint64_t{hash} >> _lookupShift);
    while (_lookupSampleIndices[lookupSlot] != kInvalidIndex &&
           _lookupSampleIndices[lookupSlot] != sampleIndex) {
      lookupSlot = (lookupSlot + 1) & _lookupMask;
    }
    return lookupSlot;
  }

  int _samplesPerElement = 0;
  size_t _lookupMask = 0;
  int _lookupShift = 32;
  DynamicArray<int> _sampleIndices;
  DynamicArray<int> _lookupSampleIndices;
  DynamicArray<int> _lookupResponseIndices;
  DynamicArray<double> _energies;
  DynamicArray<Real3> _gradients;
  DynamicArray<Matrix3x3r> _hessians;
  DynamicArray<int> _activeElementIndices;
};

// Returns the contact indices of pointRange sorted by sample index, so that the contacts of each
// element are contiguous.
DynamicArray<int> SortContactsBySample(
    Interval<int> pointRange,
    ContactDetectionResult const& query,
    Allocator* allocator) {
  DynamicArray<int> sortedContactIndices(allocator);
  sortedContactIndices.reserve(pointRange.Size());
  for (int contactIndex : pointRange) {
    sortedContactIndices.push_back(contactIndex);
  }
  std::ranges::sort(sortedContactIndices, {}, [&](int contactIndex) {
    return std::pair{query.sampleIndices[contactIndex], contactIndex};
  });
  return sortedContactIndices;
}

// Calls fn(elementIndex, contactIndices) for each element with contacts, given contact indices
// sorted by sample index.
template <class Fn>
void ForEachElementContacts(
    Span<int const> sortedContactIndices,
    ContactDetectionResult const& query,
    int samplesPerElement,
    Fn const& fn) {
  int const numContacts = isize(sortedContactIndices);
  int begin = 0;
  while (begin < numContacts) {
    int const elementIndex = query.sampleIndices[sortedContactIndices[begin]] / samplesPerElement;
    int end = begin + 1;
    while (end < numContacts &&
           query.sampleIndices[sortedContactIndices[end]] / samplesPerElement == elementIndex) {
      ++end;
    }
    fn(elementIndex, sortedContactIndices.subspan(begin, end - begin));
    begin = end;
  }
}

// Returns the largest number of physics nodes among the elements of contactIndices.
int MaxPhysicsNodes(
    CLinearContactSkinElementMap const& elementMap,
    ContactDetectionResult const& query,
    Span<int const> contactIndices) {
  int maxPhysicsNodes = 0;
  for (int contactIndex : contactIndices) {
    int const elementIndex = query.sampleIndices[contactIndex] / elementMap.samplesPerElement;
    maxPhysicsNodes = Max(maxPhysicsNodes, isize(elementMap.PhysicsNodeIndices(elementIndex)));
  }
  return maxPhysicsNodes;
}

// Accumulates the contact responses by sample, in the colliding actor's local frame.
LinearContactSkinResponseRange MakeLinearContactSkinResponseRange(
    Span<int const> sortedContactIndices,
    ContactDetectionResult const& query,
    CollisionResponseResult const& response,
    int samplesPerElement,
    bool assemObj,
    bool assemRes,
    bool assemDRes,
    VMatrix3x3r const& worldFromCollidingRotation,
    Allocator* allocator) {
  LinearContactSkinResponseRange out(
      isize(sortedContactIndices), samplesPerElement, assemObj, assemRes, assemDRes, allocator);
  auto const colliderFromCollidingRotation =
      MakeColliderFromCollidingRotation(query, worldFromCollidingRotation);
  for (int contactIndex : sortedContactIndices) {
    VMatrix3x3r const colliderFromColliding = colliderFromCollidingRotation(contactIndex);
    double const* energy = assemObj ? &response.energy[contactIndex] : nullptr;
    Real3 force MOCHI_NO_INIT;
    VMatrix3x3r dforce MOCHI_NO_INIT;
    if (assemRes) {
      force = TransformContactForceToCollidingLocal(
          colliderFromColliding, response.force[contactIndex]);
    }
    if (assemDRes) {
      dforce = TransformContactDForceToCollidingLocal(
          colliderFromColliding, response.dforce[contactIndex]);
    }
    out.Add(
        query.sampleIndices[contactIndex],
        energy,
        assemRes ? &force : nullptr,
        assemDRes ? &dforce : nullptr);
  }
  return out;
}

// Assembles the sync dresidual blocks that couple the mapped skin DoFs with the collider DoFs. For
// a collider whose contacts share DoFs, the blocks of each element's contacts share their
// destination, so they are summed and scattered once per element.
template <class ElementT>
void AssembleLinearContactSkinMixedBlocks(
    CLinearContactSkinElementMap const& elementMap,
    Span<ElementT const> elements,
    int dofOffset,
    Span<int const> sortedContactIndices,
    ContactDetectionResult const& query,
    CollisionResponseResult const& response,
    Span<real const> sampleWeights,
    Span<ContactJac const*> colliderJacs,
    VMatrix3x3r const& worldFromCollidingRotation,
    Allocator* allocator,
    AnyMatrixView<real>& outDRes) {
  static_assert(ElementT::kNumNodes == 3, "Linear contact-skin assembly requires triangles.");
  MOCHI_ASSERT_VERBOSE(elementMap.samplesPerElement == ElementT::kNumQuadPoints);
  int const maxPhysicsNodes = MaxPhysicsNodes(elementMap, query, sortedContactIndices);
  int maxStateDofs = 0;
  for (ContactJac const* colliderJac : colliderJacs) {
    maxStateDofs = Max(maxStateDofs, colliderJac->nDoFsState);
  }

  DynamicArray<IndexGroup> skinGroups(allocator);
  DynamicArray<real> mappedBasis(allocator);
  DynamicArray<real> contactToStateValues(allocator);
  DynamicArray<real> mixedValues(allocator);
  DynamicArray<real> transposeValues(allocator);
  skinGroups.reserve(maxPhysicsNodes);
  mappedBasis.reserve(maxPhysicsNodes);
  contactToStateValues.reserve(kSpaceDim3 * maxStateDofs);
  mixedValues.reserve(kSpaceDim3 * maxPhysicsNodes * maxStateDofs);
  transposeValues.reserve(kSpaceDim3 * maxPhysicsNodes * maxStateDofs);

  auto const colliderFromCollidingRotation =
      MakeColliderFromCollidingRotation(query, worldFromCollidingRotation);
  ForEachElementContacts(
      sortedContactIndices,
      query,
      elementMap.samplesPerElement,
      [&](int elementIndex, Span<int const> contactIndices) {
        auto const physicsNodes = elementMap.PhysicsNodeIndices(elementIndex);
        auto const skinWeights = elementMap.SkinNodeWeights(elementIndex);
        auto const& element = elements[elementIndex];
        MakeOffsetElementIndexGroups(elementMap, elementIndex, dofOffset, skinGroups);
        int const skinDofs = kSpaceDim3 * isize(physicsNodes);
        mappedBasis.resize_noinit(physicsNodes.size());

        for (ContactJac const* colliderJac : colliderJacs) {
          MOCHI_ASSERT_VERBOSE(colliderJac && colliderJac->groupsInitialized);
          int const colliderDofs = colliderJac->nDoFsState;
          contactToStateValues.resize_noinit(kSpaceDim3 * colliderDofs);
          mixedValues.resize_noinit(skinDofs * colliderDofs);
          transposeValues.resize_noinit(colliderDofs * skinDofs);
          RowMatrixView<real, kSpaceDim3> contactToState(
              contactToStateValues.data(), kSpaceDim3, colliderDofs);
          RowMatrixView<real> mixed(mixedValues.data(), skinDofs, colliderDofs);
          RowMatrixView<real> mixedTranspose(transposeValues.data(), colliderDofs, skinDofs);

          int const numContacts = isize(contactIndices);
          for (int contact = 0; contact < numContacts; ++contact) {
            int const contactIndex = contactIndices[contact];
            int const sampleIndex = query.sampleIndices[contactIndex];
            int const quadIndex = sampleIndex % elementMap.samplesPerElement;
            for (int physicsNode = 0; physicsNode < isize(physicsNodes); ++physicsNode) {
              mappedBasis[physicsNode] =
                  Dot(skinWeights[physicsNode], element.basisEvaluated[quadIndex]);
            }

            VMatrix3x3r const leftDResSimd = Dot3x3(
                Transpose3x3(colliderFromCollidingRotation(contactIndex)),
                -sampleWeights[sampleIndex] * response.dforce[contactIndex]);
            Matrix<real, kSpaceDim3, kSpaceDim3> const leftDRes(AsMatrixView(leftDResSimd));
            auto const pointJac = colliderJac->Jac(contactIndex);
            if (auto const auxiliaryJac = colliderJac->JacAux()) {
              MOCHI_ASSERT_VERBOSE(colliderJac->nDoFsInternal == RigidSize::kDAll);
              RowMatrix<real, kSpaceDim3, RigidSize::kDAll> const contactToInternal(
                  leftDRes * pointJac.LeftCols<RigidSize::kDAll>(RigidSize::kDAll));
              contactToState = contactToInternal * auxiliaryJac;
            } else {
              MOCHI_ASSERT_VERBOSE(colliderJac->nDoFsInternal == colliderDofs);
              contactToState = leftDRes * pointJac;
            }

            bool const startsBlock = !colliderJac->hasSharedDoFs || contact == 0;
            bool const endsBlock = !colliderJac->hasSharedDoFs || contact == numContacts - 1;
            for (int physicsNode = 0; physicsNode < isize(physicsNodes); ++physicsNode) {
              auto mixedBlock = mixed.MiddleRows<kSpaceDim3>(kSpaceDim3 * physicsNode, kSpaceDim3);
              if (startsBlock) {
                mixedBlock = mappedBasis[physicsNode] * contactToState;
              } else {
                mixedBlock += mappedBasis[physicsNode] * contactToState;
              }
            }
            if (endsBlock) {
              mixedTranspose = mixed.Transpose();
              auto const colliderGroups = colliderJac->IndGroups(contactIndex);
              MatAddSubBlocks(
                  outDRes,
                  MakeConstSpan(skinGroups),
                  colliderGroups,
                  RowMatrixView<real const>(mixed));
              MatAddSubBlocks(
                  outDRes,
                  colliderGroups,
                  MakeConstSpan(skinGroups),
                  RowMatrixView<real const>(mixedTranspose));
            }
          }
        }
      });
}

} // namespace

// Assembles the contact response of pointRange through the linear element map.
void mochi::linear_contact_skin::AssembleContactResponseRange(
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
    AssemblyResults<real> results) {
  if (pointRange.Size() == 0) {
    return;
  }
  bool const assemObj = results.outObj != nullptr;
  bool const assemRes = !results.outRes.empty();
  bool const assemDRes = GetNumValues(results.outDRes) > 0;
  auto const worldFromCollidingRotation = ToVMatrix3x3(rootTransform.worldFromLocal.GetRotation());

  auto const sortedContactIndices = SortContactsBySample(pointRange, query, allocator);
  auto const rangeResponse = MakeLinearContactSkinResponseRange(
      MakeConstSpan(sortedContactIndices),
      query,
      response,
      elementMap.samplesPerElement,
      assemObj,
      assemRes,
      assemDRes,
      worldFromCollidingRotation,
      allocator);

  auto const activeElementIndices = rangeResponse.ActiveElementIndices();
  surfaceDisc.Visit([&](auto const& disc) {
    using DiscT = std::decay_t<decltype(disc)>;
    using ElementT = typename DiscT::ElementT;
    auto const elements = MakeConstSpan(disc.femElements);
    auto const boundaryOp =
        deformable::MakeBatchedBoundaryOp(elements, rangeResponse, Span<real const>{});
    AssembleAndProjectLinearContactSkin<ElementT>(
        elements, elementMap, boundaryOp, results, activeElementIndices, dofOffset, allocator);
    if (assemDRes && !colliderJacs.empty()) {
      AssembleLinearContactSkinMixedBlocks(
          elementMap,
          elements,
          dofOffset,
          MakeConstSpan(sortedContactIndices),
          query,
          response,
          sampleWeights,
          colliderJacs,
          worldFromCollidingRotation,
          allocator,
          results.outDRes);
    }
  });
}
