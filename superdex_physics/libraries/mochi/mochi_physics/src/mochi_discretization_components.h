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
#include "mochi_ecs.h"
#include "mochi_query.h"

#include <mochi_core/element_operations/element_assembler.h>
#include <mochi_core/elements/segment/finite_element.h>
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/tetrahedral/finite_element_trace.h>
#include <mochi_core/elements/triangular/finite_element.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/utils/mesh_embedding.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace mochi {

/**************************************************************************
  Common ECS components related to discretization
*/

struct CMeshPivot {
  Real3 position = {};
};

/*
  Stores a constant pointer to a simplicial mesh
*/
struct CSimplicialMesh : public NoCopy {
  explicit CSimplicialMesh(std::shared_ptr<SimplicialMesh const> const& meshIn) : mesh(meshIn) {
    MOCHI_ASSERT(mesh);
  }
  std::shared_ptr<SimplicialMesh const> mesh;
};

/*
  Stores a constant pointer to a tetrahedral mesh
*/
struct CTetrahedralMesh : public NoCopy {
  explicit CTetrahedralMesh(std::shared_ptr<TetrahedralMesh const> const& meshIn) : mesh(meshIn) {
    MOCHI_ASSERT(mesh != nullptr);
  }
  std::shared_ptr<TetrahedralMesh const> mesh;
};

/*
  Stores a constant pointer to a surface mesh. In rigid actors, this mesh is used for the
  computation of the moment of inertia and contact. In the general case, it is used for the debug
  draw. This differs from the CVisualMesh as this is intended to be the surface portion of any
  computational mesh. In the case of actors with computational mesh being a volumetric mesh (eg.
  soft actors) this will be the boundary of the computational mesh (the full mesh, not the
  subsampled mesh) in the case of actors with computational mesh being a surface mesh (eg. rigid
  actors, shell etc) then this will be the pointer to the same mesh used for computations.
*/
struct CSurfaceMesh : public NoCopy {
  explicit CSurfaceMesh(std::shared_ptr<TriangularMesh const> const& meshIn) : mesh(meshIn) {
    MOCHI_ASSERT(mesh != nullptr);
  }
  std::shared_ptr<TriangularMesh const> mesh;
};

// NOTE: Functionally-equivalent to CSurfaceMesh, but with a different name to make it clear that
// it's not the surface of a volume mesh, but rather a mesh to be used for topologically-2D finite
// element formulations.  It could potentially diverge from CSurfaceMesh in the future.
struct CTriangularMesh : public NoCopy {
  explicit CTriangularMesh(std::shared_ptr<TriangularMesh const> const& meshIn) : mesh(meshIn) {
    MOCHI_ASSERT(mesh != nullptr);
  }
  std::shared_ptr<TriangularMesh const> mesh;
};

/*
  Stores a constant pointer to an embedded triangular mesh
*/
struct CVisualMesh : public NoCopy {
  explicit CVisualMesh(
      std::shared_ptr<TriangularMesh const> const& meshIn,
      std::shared_ptr<MeshEmbedding const> const& embeddingIn = {})
      : mesh(meshIn), embedding(embeddingIn) {
    MOCHI_ASSERT(mesh != nullptr);
  }
  std::shared_ptr<TriangularMesh const> mesh;
  std::shared_ptr<MeshEmbedding const> embedding;
};

/// @brief ECS component for the local-to-global map of the actor discretization.
struct CLocal2GlobalMap final : public Local2GlobalMap, public NoCopy {};

/// @brief Local-to-global DoF map for soft actor boundary contact.
/// @details Element order matches @ref CBoundaryNodalBasedStructure and @ref
/// CFemBoundaryDiscretization. Each boundary face maps to the 4 volume nodes of its parent tet, so
/// element DoFs scatter directly into the soft actor's volume DoFs.
struct CBoundaryLocal2GlobalMap final : public Local2GlobalMap, public NoCopy {
  explicit CBoundaryLocal2GlobalMap(Local2GlobalMap&& other) : Local2GlobalMap(std::move(other)) {}
};

/// @brief Local-to-global DoF map for codimensional deformable actor contact.
/// @details Element order matches @ref CContactNodalBasedStructure and the contact FEM
/// discretization. Each element maps from actor contact connectivity, such as shell triangles or
/// rod centerline segments, to actor global DoFs. Elements must have the DoF stride expected by the
/// contact FEM element. Variable-width connectivity must use matching stencil/padding.
struct CContactLocal2GlobalMap final : public Local2GlobalMap, public NoCopy {};

/*
  Stores the sparsity pattern for the full DOF system matrix of a simulated actor
*/
struct CFullSparsityPattern final : public NoCopy {
  MOCHI_DECLARE_MOVE(CFullSparsityPattern);

  CFullSparsityPattern() = default;
  explicit CFullSparsityPattern(Graph<int, int> graphIn) : graph(std::move(graphIn)) {}

  Graph<int, int> graph;
};

/*
  Stores the sparsity pattern for the reduced DOF system matrix of a simulated actor
*/
struct CReducedSparsityPattern final : public NoCopy {
  MOCHI_DECLARE_MOVE(CReducedSparsityPattern);

  CReducedSparsityPattern() = default;
  explicit CReducedSparsityPattern(Graph<int, int> graphIn) : graph(std::move(graphIn)) {}

  Graph<int, int> graph;
};

/*
  Stores the FEM discretization of a simulated actor.
*/
template <typename ElementT_>
struct FemDiscretizationBase : public NoCopy {
  using ElementT = ElementT_;
  static constexpr int kSpaceDim = ElementT::kSpaceDim;
  static constexpr int kNumEleNodes = ElementT::kNumDofs;
  static constexpr int kNumEleDofs = kSpaceDim * kNumEleNodes;
  static constexpr int kNumQuads = ElementT::kNumQuadPoints;
};

template <typename ElementT_>
struct FemDiscretization : public FemDiscretizationBase<ElementT_> {
  std::vector<ElementT_> femElements;

  int GetNumQuadPoints() const {
    return isize(femElements) * FemDiscretizationBase<ElementT_>::kNumQuads;
  }
};

/// @brief Light version of FEM discretization with minimal storage to reduce memory footprint.
/// @remark The memory for the coordinates and connectivity must remain valid for the lifetime of
/// the struct.
template <typename ElementT_>
struct FemDiscretizationLite : public FemDiscretizationBase<ElementT_> {
  using CoordT = NdArray<real, ElementT_::kSpaceDim>; // e.g. Real3
  using ConnecT = NdArray<int, ElementT_::kNumDofs>; // e.g. Int3
  Span<CoordT const> coordinates;
  Span<ConnecT const> connectivity;
};

/*
  Specializations of FemDiscretization for linear tetrahedra with different quadrature rules.
*/
using CFemVolumeDiscretizationP1Q1 = FemDiscretization<tetrahedral::Pk3DElement<1, 1>>;
using CFemVolumeDiscretizationP1Q4 = FemDiscretization<tetrahedral::Pk3DElement<1, 4>>;

/*
  Specializations of FemDiscretization for boundary trace elements with different quadrature rules.
*/
using CFemBoundaryDiscretizationP1Q1_1 =
    FemDiscretization<tetrahedral::Pk3DElementTrace<tetrahedral::Pk3DElement<1, 1>, 1>>;
using CFemBoundaryDiscretizationP1Q1_3 =
    FemDiscretization<tetrahedral::Pk3DElementTrace<tetrahedral::Pk3DElement<1, 1>, 3>>;
using CFemBoundaryDiscretizationP1Q1_6 =
    FemDiscretization<tetrahedral::Pk3DElementTrace<tetrahedral::Pk3DElement<1, 1>, 6>>;
using CFemBoundaryDiscretizationP1Q1_7 =
    FemDiscretization<tetrahedral::Pk3DElementTrace<tetrahedral::Pk3DElement<1, 1>, 7>>;
using CFemBoundaryDiscretizationP1Q1_12 =
    FemDiscretization<tetrahedral::Pk3DElementTrace<tetrahedral::Pk3DElement<1, 1>, 12>>;
using CFemBoundaryDiscretizationP1Q1_16 =
    FemDiscretization<tetrahedral::Pk3DElementTrace<tetrahedral::Pk3DElement<1, 1>, 16>>;

// TODO: Consolidate CFemBoundaryDiscretization and CFemSurfaceDiscretization into a single
// component that all actors use.
struct CFemBoundaryDiscretization : public CVariant<
                                        CFemBoundaryDiscretizationP1Q1_1,
                                        CFemBoundaryDiscretizationP1Q1_3,
                                        CFemBoundaryDiscretizationP1Q1_6,
                                        CFemBoundaryDiscretizationP1Q1_7,
                                        CFemBoundaryDiscretizationP1Q1_12,
                                        CFemBoundaryDiscretizationP1Q1_16> {
  using CVariant::CVariant;

  CFemBoundaryDiscretization() = delete;

  int GetNumQuadPoints() const {
    return Visit([&](auto const& disc) { return disc.GetNumQuadPoints(); });
  }

  template <typename FemVolumeDiscretizationT>
  static CFemBoundaryDiscretization Create(
      TetrahedralMesh const& mesh,
      FemVolumeDiscretizationT const& femVolDisc,
      ActorBoundaryElementType const& elementType) {
    auto disc = CreateNoInitialize(elementType);
    disc.Visit([&](auto& femBoundaryDiscImpl) {
      using BoundaryDiscretizationT = std::decay_t<decltype(femBoundaryDiscImpl)>;

      femBoundaryDiscImpl.femElements.reserve(mesh.GetNumBoundaryFaces());
      for (auto const& bdface : mesh.GetBoundaryFaces()) {
        femBoundaryDiscImpl.femElements.emplace_back(
            MakeBoundaryElement<BoundaryDiscretizationT>(
                femVolDisc.femElements[bdface.element], bdface.faceNum));
      }
    });

    return disc;
  }

 private:
  template <typename FemBoundaryDiscretizationT, typename FemVolumeElementT>
  static auto MakeBoundaryElement(FemVolumeElementT const& volElement, int faceNum) {
    using ElementT = typename FemBoundaryDiscretizationT::ElementT;
    if constexpr (std::is_same_v<FemBoundaryDiscretizationT, CFemBoundaryDiscretizationP1Q1_1>) {
      return ElementT{volElement, faceNum, tetrahedral::kTetrahedralTraceQuadrature1[faceNum]};
    } else if constexpr (std::is_same_v<
                             FemBoundaryDiscretizationT,
                             CFemBoundaryDiscretizationP1Q1_3>) {
      return ElementT{volElement, faceNum, tetrahedral::kTetrahedralTraceQuadrature3[faceNum]};
    } else if constexpr (std::is_same_v<
                             FemBoundaryDiscretizationT,
                             CFemBoundaryDiscretizationP1Q1_6>) {
      return ElementT{volElement, faceNum, tetrahedral::kTetrahedralTraceQuadrature6[faceNum]};
    } else if constexpr (std::is_same_v<
                             FemBoundaryDiscretizationT,
                             CFemBoundaryDiscretizationP1Q1_7>) {
      return ElementT{volElement, faceNum, tetrahedral::kTetrahedralTraceQuadrature7[faceNum]};
    } else if constexpr (std::is_same_v<
                             FemBoundaryDiscretizationT,
                             CFemBoundaryDiscretizationP1Q1_12>) {
      return ElementT{volElement, faceNum, tetrahedral::kTetrahedralTraceQuadrature12[faceNum]};
    } else if constexpr (std::is_same_v<
                             FemBoundaryDiscretizationT,
                             CFemBoundaryDiscretizationP1Q1_16>) {
      return ElementT{volElement, faceNum, tetrahedral::kTetrahedralTraceQuadrature16[faceNum]};
    } else {
      static_assert(
          std::is_void_v<FemBoundaryDiscretizationT>,
          "MakeBoundaryElement not implemented for this boundary discretization type.");
    }
  }

  static CFemBoundaryDiscretization CreateNoInitialize(
      ActorBoundaryElementType const& elementType) {
    switch (elementType) {
      case ActorBoundaryElementType::P1Q1:
        return CFemBoundaryDiscretizationP1Q1_1();
      case ActorBoundaryElementType::P1Q3:
        return CFemBoundaryDiscretizationP1Q1_3();
      case ActorBoundaryElementType::P1Q6:
        return CFemBoundaryDiscretizationP1Q1_6();
      case ActorBoundaryElementType::ExperimentalP1Q7:
        return CFemBoundaryDiscretizationP1Q1_7();
      case ActorBoundaryElementType::ExperimentalP1Q12:
        return CFemBoundaryDiscretizationP1Q1_12();
      case ActorBoundaryElementType::ExperimentalP1Q16:
        return CFemBoundaryDiscretizationP1Q1_16();
      default:
        MOCHI_ASSERT(false, "Invalid boundary element type.");
        return CFemBoundaryDiscretizationP1Q1_1();
    }
    static_assert(
        static_cast<int>(ActorBoundaryElementType::Count) == 6,
        "Please update the switch statement above if ActorBoundaryElementType enum changes");
  }
};

/*
  Specializations of FemDiscretization for triangular elements with different quadrature rules.
*/
using CFemSurfaceDiscretizationP1Q1 = FemDiscretization<triangular::Pk2DElement<1, 1>>;
using CFemSurfaceDiscretizationP1Q3 = FemDiscretization<triangular::Pk2DElement<1, 3>>;
using CFemSurfaceDiscretizationP1Q6 = FemDiscretization<triangular::Pk2DElement<1, 6>>;
using CFemSurfaceDiscretizationP1Q7 = FemDiscretization<triangular::Pk2DElement<1, 7>>;
using CFemSurfaceDiscretizationP1Q12 = FemDiscretization<triangular::Pk2DElement<1, 12>>;
using CFemSurfaceDiscretizationP1Q16 = FemDiscretization<triangular::Pk2DElement<1, 16>>;

struct CFemSurfaceDiscretization : public CVariant<
                                       CFemSurfaceDiscretizationP1Q1,
                                       CFemSurfaceDiscretizationP1Q3,
                                       CFemSurfaceDiscretizationP1Q6,
                                       CFemSurfaceDiscretizationP1Q7,
                                       CFemSurfaceDiscretizationP1Q12,
                                       CFemSurfaceDiscretizationP1Q16> {
  using CVariant::CVariant;
  static CFemSurfaceDiscretization Create(
      ActorBoundaryElementType const& elementType,
      TriangularMesh const& surfaceMesh) {
    CFemSurfaceDiscretization surfaceDiscr = CreateNoInitialize(elementType);
    surfaceDiscr.Visit([&](auto& discr) {
      discr.femElements.reserve(surfaceMesh.GetNumElements());
      for (int i = 0; i < surfaceMesh.GetNumElements(); ++i) {
        discr.femElements.emplace_back(
            i,
            surfaceMesh.GetActiveNodeCoordinates(),
            Unflatten<Int3 const>(surfaceMesh.GetActiveNodesFlatConnectivity()));
      }
    });
    return surfaceDiscr;
  }

  int GetNumQuadPoints() const {
    return Visit([&](auto const& disc) { return disc.GetNumQuadPoints(); });
  }

 private:
  static CFemSurfaceDiscretization CreateNoInitialize(ActorBoundaryElementType const& elementType) {
    switch (elementType) {
      case ActorBoundaryElementType::P1Q1:
        return CFemSurfaceDiscretizationP1Q1();
      case ActorBoundaryElementType::P1Q3:
        return CFemSurfaceDiscretizationP1Q3();
      case ActorBoundaryElementType::P1Q6:
        return CFemSurfaceDiscretizationP1Q6();
      case ActorBoundaryElementType::ExperimentalP1Q7:
        return CFemSurfaceDiscretizationP1Q7();
      case ActorBoundaryElementType::ExperimentalP1Q12:
        return CFemSurfaceDiscretizationP1Q12();
      case ActorBoundaryElementType::ExperimentalP1Q16:
        return CFemSurfaceDiscretizationP1Q16();
      default:
        MOCHI_ASSERT(false, "Invalid surface element type.");
        return CFemSurfaceDiscretizationP1Q1();
    }
    static_assert(
        static_cast<int>(ActorBoundaryElementType::Count) == 6,
        "Please update the switch statement above if ActorBoundaryElementType enum changes");
  };
};

/*
  Specializations of FemDiscretizationLite for triangular elements with different quadrature rules.
*/
using FemSurfaceDiscretizationLiteP1Q1 = FemDiscretizationLite<triangular::Pk2DElement<1, 1>>;
using FemSurfaceDiscretizationLiteP1Q3 = FemDiscretizationLite<triangular::Pk2DElement<1, 3>>;
using FemSurfaceDiscretizationLiteP1Q6 = FemDiscretizationLite<triangular::Pk2DElement<1, 6>>;
using FemSurfaceDiscretizationLiteP1Q7 = FemDiscretizationLite<triangular::Pk2DElement<1, 7>>;
using FemSurfaceDiscretizationLiteP1Q12 = FemDiscretizationLite<triangular::Pk2DElement<1, 12>>;
using FemSurfaceDiscretizationLiteP1Q16 = FemDiscretizationLite<triangular::Pk2DElement<1, 16>>;

struct CFemSurfaceDiscretizationLite : public CVariant<
                                           FemSurfaceDiscretizationLiteP1Q1,
                                           FemSurfaceDiscretizationLiteP1Q3,
                                           FemSurfaceDiscretizationLiteP1Q6,
                                           FemSurfaceDiscretizationLiteP1Q7,
                                           FemSurfaceDiscretizationLiteP1Q12,
                                           FemSurfaceDiscretizationLiteP1Q16> {
  using CVariant::CVariant;
  static CFemSurfaceDiscretizationLite Create(
      ActorBoundaryElementType const& elementType,
      TriangularMesh const& surfaceMesh) {
    auto connectivity = Unflatten<Int3 const>(surfaceMesh.GetActiveNodesFlatConnectivity());
    switch (elementType) {
      case ActorBoundaryElementType::P1Q1:
        return FemSurfaceDiscretizationLiteP1Q1{
            .coordinates = surfaceMesh.GetActiveNodeCoordinates(), .connectivity = connectivity};
      case ActorBoundaryElementType::P1Q3:
        return FemSurfaceDiscretizationLiteP1Q3{
            .coordinates = surfaceMesh.GetActiveNodeCoordinates(), .connectivity = connectivity};
      case ActorBoundaryElementType::P1Q6:
        return FemSurfaceDiscretizationLiteP1Q6{
            .coordinates = surfaceMesh.GetActiveNodeCoordinates(), .connectivity = connectivity};
      case ActorBoundaryElementType::ExperimentalP1Q7:
        return FemSurfaceDiscretizationLiteP1Q7{
            .coordinates = surfaceMesh.GetActiveNodeCoordinates(), .connectivity = connectivity};
      case ActorBoundaryElementType::ExperimentalP1Q12:
        return FemSurfaceDiscretizationLiteP1Q12{
            .coordinates = surfaceMesh.GetActiveNodeCoordinates(), .connectivity = connectivity};
      case ActorBoundaryElementType::ExperimentalP1Q16:
        return FemSurfaceDiscretizationLiteP1Q16{
            .coordinates = surfaceMesh.GetActiveNodeCoordinates(), .connectivity = connectivity};
      default:
        MOCHI_ASSERT(false, "Invalid surface element type.");
        return FemSurfaceDiscretizationLiteP1Q1();
    }
    static_assert(
        static_cast<int>(ActorBoundaryElementType::Count) == 6,
        "Please update the switch statement above if ActorBoundaryElementType enum changes");
  };
};

/*
  Specializations of FemDiscretization for segment elements with different quadrature rules.
*/
using CFemSegmentDiscretizationP1Q1 = FemDiscretization<segment::Pk1DElement<1, 1>>;
using CFemSegmentDiscretizationP1Q2 = FemDiscretization<segment::Pk1DElement<1, 2>>;
using CFemSegmentDiscretizationP1Q3 = FemDiscretization<segment::Pk1DElement<1, 3>>;

struct CFemSegmentDiscretization : public CVariant<
                                       CFemSegmentDiscretizationP1Q1,
                                       CFemSegmentDiscretizationP1Q2,
                                       CFemSegmentDiscretizationP1Q3> {
  using CVariant::CVariant;
  static CFemSegmentDiscretization Create(
      ActorSegmentElementType const& elementType,
      DynamicArray<Real3> const& nodes,
      bool isClosedLoop) {
    CFemSegmentDiscretization segmentDiscr = CreateNoInitialize(elementType);
    segmentDiscr.Visit([&](auto& discr) {
      int const numSegments = isClosedLoop ? isize(nodes) : isize(nodes) - 1;
      discr.femElements.reserve(numSegments);
      Span<Real3 const> coordinates = MakeConstSpan(nodes);
      for (int i = 0; i < numSegments; ++i) {
        discr.femElements.emplace_back(i, coordinates);
      }
    });
    return segmentDiscr;
  }

  int GetNumQuadPoints() const {
    return Visit([&](auto const& disc) { return disc.GetNumQuadPoints(); });
  }

 private:
  static CFemSegmentDiscretization CreateNoInitialize(ActorSegmentElementType const& elementType) {
    switch (elementType) {
      case ActorSegmentElementType::P1Q1:
        return CFemSegmentDiscretizationP1Q1();
      case ActorSegmentElementType::P1Q2:
        return CFemSegmentDiscretizationP1Q2();
      case ActorSegmentElementType::P1Q3:
        return CFemSegmentDiscretizationP1Q3();
      default:
        MOCHI_ASSERT(false, "Invalid segment element type.");
        return CFemSegmentDiscretizationP1Q2();
    }
    static_assert(
        static_cast<int>(ActorSegmentElementType::Count) == 3,
        "Please update the switch statement above if ActorSegmentElementType enum changes");
  };
};

/*
Component used to store the active subset of an actor's tet volume elements
as well as corresponding weights needed to perform FEM operations.

A few comments:

- this component is meant to be constructed once for a given underlying tet mesh.
  The Recompute method can be called to update the active elements and their weights,
  but it should only be called for the same underlying mesh.
  In other words, you cannot construct this component for a given mesh and then
  try to recompute it using a different mesh.
  Right now, if you try to do that, it results in some sort of UB.
  There might be a way to provide a better error for that scenario.

- this component also contains the weighting information needed to use
  the active elements for FEM operations. By default, upon construction,
  these weights are computed based on the tet element measures as follows:
  we calculate the ratio of the sum of the measures of the active elements
  over the sum of the measures of all elements.
*/
class CActiveVolumeElements : public NoCopy {
 private:
  std::shared_ptr<TetrahedralMesh const> _mesh;

  std::vector<int> _indices = {};

  std::vector<int> _uniqueNodeIds = {};

  // _isActive: vector used for the Contain method to efficiently check
  // if a vol element index is currently active or not.
  DynamicArray<bool> _isActive;

  // vector of weights that depends on the active elements
  std::vector<real> _weights = {};

 public:
  CActiveVolumeElements(
      std::shared_ptr<TetrahedralMesh const> const& tetMesh,
      Span<int const> activeIndices,
      Span<real const> activeElemWeights = {})
      : _mesh(tetMesh) {
    int const meshNumVolElems = _mesh->GetNumElements();

    _indices.reserve(meshNumVolElems);
    _uniqueNodeIds.reserve(meshNumVolElems * TetrahedralMesh::kNodesPerElement);

    // _isActive must be of the same size as the vol elements
    _isActive.resize(meshNumVolElems, false);

    // the weights must be of the same size as the vol elements
    _weights.resize(meshNumVolElems, 0_r);

    Recompute(activeIndices, activeElemWeights);
  }

  [[nodiscard]] bool empty() const {
    return _indices.empty();
  }

  [[nodiscard]] Span<real const> ViewWeights() const {
    return _indices.empty() ? Span<real const>{} : _weights;
  }

  [[nodiscard]] bool Contains(int testElemIndex) const {
    MOCHI_ASSERT_VERBOSE(testElemIndex >= 0 && testElemIndex < isize(_isActive));
    return static_cast<bool>(_isActive[testElemIndex]);
  }

  [[nodiscard]] Span<bool const> ViewIsActive() const {
    return _isActive;
  }

  [[nodiscard]] Span<int const> ViewIndices() const {
    return _indices;
  }

  [[nodiscard]] Span<int const> ViewUniqueNodes() const {
    MOCHI_ASSERT_VERBOSE(IsUnique(MakeConstSpan(_uniqueNodeIds)));
    return _uniqueNodeIds;
  }

  void Recompute(Span<int const> newActiveIndices, Span<real const> newActiveWeights = {}) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    MOCHI_ASSERT_VERBOSE(IsUnique(newActiveIndices), "Active elements must be unique");
    MOCHI_ASSERT_VERBOSE(
        newActiveWeights.empty() || (newActiveWeights.size() == newActiveIndices.size()),
        "The span of weights must be of the same size as the span of active volumes.");
    for (int activeIndex : newActiveIndices) {
      MOCHI_ASSERT_VERBOSE(
          activeIndex >= 0 && activeIndex < isize(_isActive),
          "Active volume element index is out of range.");
    }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

    // replace the new active indices
    _indices.assign(newActiveIndices.begin(), newActiveIndices.end());

    // recompute the "indicator" vector that for each element stores if it is active or not
    std::fill(_isActive.begin(), _isActive.end(), 0);
    for (auto i : _indices) {
      _isActive[i] = true;
    }

    // Store nodes from selected elements
    _mesh->UniqueNodesInElements(_indices, _uniqueNodeIds);

    // if provided a new span of weights to use, copy them and override the stored weights.
    // Otherwise, do the default reweighting.
#if MOCHI_DEBUG // Populate inactive weights with NaNs in DEBUG build to catch if used anywhere.
    for (auto& w : _weights) {
      w = std::numeric_limits<real>::signaling_NaN();
    }
#endif
    if (newActiveWeights.empty() && !_indices.empty()) {
      RecomputeWeightsBasedOnElementsMeasure();
    } else {
      // Only need to store the weights for the active elements.
      for (int i = 0; i < isize(newActiveIndices); ++i) {
        _weights[newActiveIndices[i]] = newActiveWeights[i];
      }
    }
  }

 private:
  void RecomputeWeightsBasedOnElementsMeasure() {
    MOCHI_ASSERT_VERBOSE(isize(_weights) == _mesh->GetNumElements());

    real activeSum = {};
    std::for_each(_indices.cbegin(), _indices.cend(), [&](int i) {
      activeSum += _mesh->GetElementMeasure(i);
    });
    MOCHI_ASSERT(activeSum > 0, "Active measure must be positive.");

    // Only need to store the weights for the active elements.
    real const totalMeasure = _mesh->GetTotalMeasure();
    std::for_each(_indices.cbegin(), _indices.cend(), [&](int elemIndex) {
      _weights[elemIndex] = totalMeasure / activeSum;
    });
  }
};

/*
Component used to store the subset of active surface faces of an actor, as well as the corresponding
weights to perform FEM operations. If an interior mesh is provided in the constructor, the
corresponding subset of active volume elements is also stored.

Note:

- This component is meant to be constructed once for a given underlying mesh. The Recompute method
  can be called to update the active boundary faces, but it must be called for the same underlying
  mesh. In other words, you must NOT construct this component with one mesh and then try to
  recompute it with a different mesh. This is currently enforced by caching some info upon
  construction, and then with a few asserts inside Recompute. There are better ways to enforce this
  constraint, for example, tying the component or even putting it inside the tet mesh component.

- If no explicit weights are provided, the weights are computed as the ratio of the sum of the
  measure of the active boundary faces over the sum of the measure of all boundary faces.

- If an interior mesh is provided in the constructor, it also stores the subset of volume elements
  that neighbor the active boundary faces.
*/
class CActiveBoundaryFaces : public NoCopy {
 private:
  // [Optional] Interior tetrahedral mesh.
  std::shared_ptr<TetrahedralMesh const> _interiorMesh;

  // Boundary triangular mesh.
  std::shared_ptr<TriangularMesh const> _boundaryMesh;

  // Number of quadrature points per boundary face.
  int _numQuadPerFace = 0;

  // Indices of the active boundary faces.
  std::vector<int> _activeBoundaryFaceInds = {};

  // Vector used for the Contain method to efficiently check if a boundary face index is active or
  // not. Size is equal to the number of ALL boundary faces.
  DynamicArray<bool> _isActive;

  // Weights for each boundary face. Size is equal to the number of ALL boundary faces. For
  // performance reasons, the entries corresponding to inactive faces are not reset to zero and
  // contain arbitrary values.
  std::vector<real> _weights = {};

  // [Optional] Indices of the nodes in the volume mesh (NOT in the boundary mesh) that belong to
  // the active boundary faces. Does not contain duplicate indices. Only populated if an interior
  // mesh is provided in the constructor.
  std::vector<int> _uniqueVolumeNodeInds = {};

  // [Optional] Indices of the volume elements that the active boundary faces belong to. Does not
  // contain duplicate indices. Size is equal to the number of volume elements neighboring the
  // active boundary faces (i.e. the size may be smaller than the number of active boundary faces).
  // Only populated if an interior mesh is provided in the constructor.
  std::vector<int> _uniqueVolumeElementInds = {};

 public:
  template <typename BoundaryTraceT>
  CActiveBoundaryFaces(
      std::shared_ptr<TetrahedralMesh const> const& interiorMesh,
      Span<int const> activeTraceIndices,
      Span<BoundaryTraceT const> boundaryTraces,
      Span<real const> activeTraceWeights = {}) {
    static_assert(
        BoundaryTraceT::kSpaceDimParam == 3,
        "Tetrahedral mesh constructor requires BoundaryTraceT to be the trace of a 3D element");
    MOCHI_ASSERT(interiorMesh, "Invalid tetrahedral mesh pointer.");
    MOCHI_ASSERT(interiorMesh->GetBoundaryMesh(), "Invalid triangular mesh pointer.");
    MOCHI_ASSERT(
        interiorMesh->GetBoundaryMesh()->GetNumElements() == isize(boundaryTraces),
        "Inconsistent number of boundary faces.");

    _interiorMesh = interiorMesh;
    _boundaryMesh = interiorMesh->GetBoundaryMesh();
    InitCommon(activeTraceIndices, boundaryTraces, activeTraceWeights);
  }

  template <typename BoundaryFaceT>
  CActiveBoundaryFaces(
      std::shared_ptr<TriangularMesh const> const& boundaryMesh,
      Span<int const> activeFaceIndices,
      Span<BoundaryFaceT const> boundaryFaces,
      Span<real const> activeFaceWeights = {})
      : _boundaryMesh(boundaryMesh) {
    static_assert(
        BoundaryFaceT::kSpaceDimParam == 2,
        "Triangular mesh constructor requires BoundaryFaceT to be a 2D face element");
    MOCHI_ASSERT(_boundaryMesh, "Invalid triangular mesh pointer.");
    MOCHI_ASSERT(
        _boundaryMesh->GetNumElements() == isize(boundaryFaces),
        "Inconsistent number of boundary faces.");
    InitCommon(activeFaceIndices, boundaryFaces, activeFaceWeights);
  }

  [[nodiscard]] bool empty() const {
    return _activeBoundaryFaceInds.empty();
  }

  [[nodiscard]] Span<real const> ViewWeights() const {
    return _activeBoundaryFaceInds.empty() ? Span<real const>{} : _weights;
  }

  [[nodiscard]] int NumQuadPerFace() const {
    return _numQuadPerFace;
  }

  [[nodiscard]] bool Contains(int faceIdx) const {
    MOCHI_ASSERT_VERBOSE(faceIdx >= 0 && faceIdx < isize(_isActive), "Face index is out of range.");
    return static_cast<bool>(_isActive[faceIdx]);
  }

  [[nodiscard]] Span<bool const> ViewIsActive() const {
    return _isActive;
  }

  [[nodiscard]] Span<int const> ViewIndices() const {
    return _activeBoundaryFaceInds;
  }

  [[nodiscard]] Span<int const> ViewUniqueVolumeNodes() const {
    MOCHI_ASSERT(_interiorMesh, "An interior mesh was not provided.");
    MOCHI_ASSERT_VERBOSE(IsUnique(MakeConstSpan(_uniqueVolumeNodeInds)));
    return _uniqueVolumeNodeInds;
  }

  template <typename BoundaryFaceT>
  void Recompute(
      Span<int const> newActiveFaceIndices,
      Span<BoundaryFaceT const> boundaryFaces,
      Span<real const> newActiveFaceWeights = {}) {
    MOCHI_ASSERT(_numQuadPerFace == BoundaryFaceT::kNumQuadPoints);
    MOCHI_ASSERT_VERBOSE(IsUnique(newActiveFaceIndices), "Active elements must be unique.");
    MOCHI_ASSERT_VERBOSE(
        newActiveFaceWeights.empty() ||
            (newActiveFaceWeights.size() == isize(newActiveFaceIndices)),
        "The span of weights must be of the same size as the span of active faces.");
    UpdateActiveSet(newActiveFaceIndices);
    UpdateInterior(boundaryFaces);
    UpdateWeights(newActiveFaceIndices, newActiveFaceWeights);
  }

 private:
  template <typename FaceT>
  void InitCommon(Span<int const> activeIdx, Span<FaceT const> faces, Span<real const> weights) {
    auto const numFaces = faces.size();
    _numQuadPerFace = FaceT::kNumQuadPoints;
    _activeBoundaryFaceInds.reserve(numFaces);
    _isActive.resize(numFaces, 0);
    _weights.resize(numFaces, 0_r);

    Recompute(activeIdx, faces, weights);
  }

  void UpdateActiveSet(Span<int const> newActiveFaceIndices) {
    _activeBoundaryFaceInds.assign(newActiveFaceIndices.begin(), newActiveFaceIndices.end());
    std::fill(_isActive.begin(), _isActive.end(), 0);
    for (auto i : _activeBoundaryFaceInds) {
      _isActive[i] = true;
    }
  }

  template <typename BoundaryFaceT>
  void UpdateInterior(Span<BoundaryFaceT const> boundaryFaces) {
    if (!_interiorMesh) {
      return;
    }

    if constexpr (BoundaryFaceT::kSpaceDimParam == 3) {
      // Find the volume elements owning the active traces.
      _uniqueVolumeElementInds.resize(_activeBoundaryFaceInds.size());
      for (int i = 0; i < isize(_activeBoundaryFaceInds); ++i) {
        int const elemIdx = boundaryFaces[_activeBoundaryFaceInds[i]].GetElementIndex();
        _uniqueVolumeElementInds[i] = elemIdx;
      }
      SortAndRemoveDuplicates(_uniqueVolumeElementInds);

      _interiorMesh->UniqueNodesInElements(_uniqueVolumeElementInds, _uniqueVolumeNodeInds);
    } else {
      MOCHI_ASSERT(false, "Element type must be a boundary trace element.")
    }
  }

  void UpdateWeights(Span<int const> newActiveFaceIndices, Span<real const> newActiveFaceWeights) {
    // if provided a new span of weights to use, copy them and override the stored weights.
    // Otherwise, do the default reweighting.
#if MOCHI_DEBUG // Populate inactive weights with NaNs in DEBUG build to catch if used anywhere.
    for (auto& w : _weights) {
      w = std::numeric_limits<real>::signaling_NaN();
    }
#endif
    if (newActiveFaceWeights.empty() && !_activeBoundaryFaceInds.empty()) {
      MOCHI_ASSERT(isize(_weights) == _boundaryMesh->GetNumElements());

      real activeMeasure = 0_r;
      std::for_each(_activeBoundaryFaceInds.cbegin(), _activeBoundaryFaceInds.cend(), [&](int i) {
        activeMeasure += _boundaryMesh->GetElementMeasure(i);
      });
      MOCHI_ASSERT(activeMeasure > 0, "Active measure must be positive.");

      // Only need to store the weights for the active boundary faces.
      real const totalMeasure = _boundaryMesh->GetTotalMeasure();
      std::for_each(
          _activeBoundaryFaceInds.cbegin(), _activeBoundaryFaceInds.cend(), [&](int bdFaceIdx) {
            _weights[bdFaceIdx] = totalMeasure / activeMeasure;
          });
    } else {
      // Only need to store the weights for the active boundary faces.
      for (int i = 0; i < isize(newActiveFaceIndices); ++i) {
        _weights[newActiveFaceIndices[i]] = newActiveFaceWeights[i];
      }
    }
  }
};

/*
  Component used to store the *unique* active volume node indices of an actor.
*/
class CActiveUniqueNodes : public NoCopy {
  std::shared_ptr<TetrahedralMesh const> _mesh;
  std::vector<int> _indices = {};

 public:
  CActiveUniqueNodes(
      std::shared_ptr<TetrahedralMesh const> const& tetMesh,
      CActiveVolumeElements const& activeVolElem,
      CActiveBoundaryFaces const& activeBdFaces)
      : _mesh(tetMesh) {
    _indices.reserve(_mesh->GetNumElements() * TetrahedralMesh::kNodesPerElement);
    Recompute(activeVolElem, activeBdFaces);
  }

  int Count() const {
    return isize(_indices);
  }

  Span<int const> ViewIds() const {
    return _indices;
  }

  void Recompute(
      CActiveVolumeElements const& activeVolElem,
      CActiveBoundaryFaces const& activeBdFaces) {
    // TODO: Dynamic hyper-reduction is currently only supported for the boundary. If this
    // limitation is to be maintained long term, 'seenNodes' could be precomputed.
    auto const& volNodes = activeVolElem.ViewUniqueNodes();
    auto const& bdNodes = activeBdFaces.ViewUniqueVolumeNodes();

    // NOTE: Performs dynamic memory allocation but cost seems negligible.
    std::vector<bool> seenNodes(_mesh->GetNumNodes(), false);

    _indices.clear();
    _indices.reserve(volNodes.size() + bdNodes.size());
    for (auto nodeIdx : volNodes) {
      // No need to check seenNodes. volNodes are unique.
      _indices.push_back(nodeIdx);
      seenNodes[nodeIdx] = true;
    }
    for (auto nodeIdx : bdNodes) {
      if (!seenNodes[nodeIdx]) {
        _indices.push_back(nodeIdx);
        // No need to update seenNodes. bdNodes are unique.
      }
    }
  }
};

} // namespace mochi
