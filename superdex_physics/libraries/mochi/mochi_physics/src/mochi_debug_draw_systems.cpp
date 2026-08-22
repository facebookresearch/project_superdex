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

#include "mochi_articulated_body.h"
#include "mochi_common_components.h"
#include "mochi_debug_draw.h"
#include "mochi_discretization_components.h"
#include "mochi_ecs_utils.h"
#include "mochi_group.h"
#include "mochi_hyper_reduction.h"
#include "mochi_island.h"
#include "mochi_point_cloud_contact.h"
#include "mochi_pose_controller.h"
#include "mochi_rigid.h"
#include "mochi_rod.h"
#include "mochi_rod_pose.h"
#include "mochi_snle.h"
#include "mochi_soft.h"

#include <mochi_core/articulated_body/articulated_body_params.h>
#include <mochi_core/materials/batched_smith_neo_hookean.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/string_utils.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace mochi {

using LineVertex = DebugDrawLineVertex;

static void RegisterDebugDrawSystem_ActorRootTransform(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Root Transform";
  system.description =
      "Draw an RGB tri-axis for the XYZ basis vectors of the actor's local coordinate space.";
  system.onDrawEntityLocalSpace = [](entt::registry const&, entt::entity, DebugDrawCollector& out) {
    constexpr real kScale = 0.05_r; // TODO: this should be configurable in some way
    out.AddTriAxis(TransformRT{}, kScale);
  };
  debugDraw.RegisterSystem<CRootTransform>(system);
}

static void RegisterDebugDrawSystem_ActorRigidPivotEvalPoint(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Rigid Pivot Eval Point";
  system.description =
      "Draw an RGB tri-axis for the point used to evaluate a soft actor's rigid pivot. Usually near the center-of-mass.";
  system.onDrawEntityLocalSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const& pivotEval = reg.get<CRigidTransformEval>(e);
        auto const& mesh = reg.get<CSimplicialMesh>(e);
        // The pivot is inside the mesh. Scale the tri-axis so it sticks out of the mesh.
        real scale = Max(mesh.mesh->GetAabb().GetSize());
        out.AddTriAxis(pivotEval.value, scale);
      };
  debugDraw.RegisterSystem<CRigidTransformEval, CSimplicialMesh>(system);
}

static void RegisterDebugDrawSystem_ActorRigidBodyInertia(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Rigid Body Inertia";
  system.description =
      "Draw a scaled tri-axis representing the actor's principal axes and moments of inertia at the COM.";
  // TODO: this would be better visualized as an ellipsoid
  system.onDrawEntityWorldSpace = [](entt::registry const& reg,
                                     entt::entity e,
                                     DebugDrawCollector& out) {
    auto const& inertia = reg.get<CRigidBodyInertia const>(e);
    auto const& root = reg.get<CRootTransform>(e);
    auto moiSym = SimdFullToSym(inertia.GetMomentOfInertiaLocal());
    if (moiSym == VSymMatrix3x3r{}) {
      return; // no inertia, no drawing for you
    }
    Real3 com = ToReal3(inertia.GetCenterOfMassLocal());
    com = root.worldFromLocal.TransformPoint(com);
    Vec4r vEigvals;
    VMatrix3x3r vEigvecs;
    AnalyticalEigendecompSym3x3(moiSym, vEigvals, &vEigvecs);
    if (!IsFinite(vEigvals) || !IsFinite(vEigvecs)) {
      return; // eigenvalues or eigenvectors are not finite, no drawing for you
    }
    // NB: SIMD eigvecs are transposed
    Matrix3x3r const eigvecsT = ToNdArray3x3(vEigvecs);
    Matrix3x3r const principalAxes = Transpose(eigvecsT);
    Real3 const principalMoments = ToReal3(vEigvals);
    real const maxMoment = Max(principalMoments) + std::numeric_limits<real>::min();
    constexpr real kScale = 0.1_r; // max axes length will be 10 cm
    for (int i = 0; i < 3; ++i) {
      auto dir = Real3{principalAxes[0][i], principalAxes[1][i], principalAxes[2][i]};
      dir = root.worldFromLocal.TransformDirection(dir);
      dir = Normalize(dir);
      auto const mag = principalMoments[i] / maxMoment * kScale;
      if (mag > 0) {
        auto v1 = com;
        auto v2 = com + dir * mag;
        out.AddLine(
            {.position = v1, .color = colors::kOrange}, {.position = v2, .color = colors::kOrange});
      }
    }
    out.AddSphere((DebugDrawSphere{com, 0.006, colors::kOrange}));
  };
  debugDraw.RegisterSystem<CRootTransform, CRigidBodyInertia>(system);
}

static void RegisterDebugDrawSystem_ActorAabbWorld(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor AABB World (tight fit)";
  system.description = "Draw a wireframe box for the actor's world-space Aabb.";
  system.sortingDepth -= 3_r; // Draw on top of default stuff, in case they overlap
  system.onDrawEntityWorldSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const& root = reg.get<CRootTransform>(e);
        auto const& bv = reg.get<CBoundingVolume<TimeStep::Current>>(e);
        if (std::holds_alternative<Plane>(bv.localShape)) {
          // This is an infinite plane. Don't draw anything.
        } else {
          out.AddWireframeAabb(
              GetAabb(TransformShape(root.worldFromLocal, bv.localShape)), MakeColor(0x80FF80FF));
        }
      };
  debugDraw.RegisterSystem<CRootTransform, CBoundingVolume<TimeStep::Current>>(system);
}

static void RegisterDebugDrawSystem_ActorAabbWorldConservative(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor AABB World (conservative)";
  system.description =
      "Draw a wireframe box for the actor's conservative step bounds (used for island culling).";
  system.sortingDepth -= 4_r; // Draw on top of default stuff, in case they overlap
  system.onDrawEntityWorldSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const& worldBounds = reg.get<CConservativeStepBounds const>(e).worldAabb;
        out.AddWireframeAabb(worldBounds, MakeColor(0xFF9933FF));
      };
  debugDraw.RegisterSystem<CConservativeStepBounds>(system);
}

static void RegisterDebugDrawSystem_ActorAabbLocal(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor AABB Local";
  system.description = "Draw a wireframe box for the actor's local-space AABB.";
  system.sortingDepth -= 2_r; // Draw after the world-space Aabb, but before most other stuff
  system.onDrawEntityLocalSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const& bv = reg.get<CBoundingVolume<TimeStep::Current>>(e);
        if (std::holds_alternative<Plane>(bv.localShape)) {
          // This is an infinite plane. Don't draw anything.
        } else {
          out.AddWireframeAabb(GetAabb(bv.localShape), colors::kGray);
        }
      };
  debugDraw.RegisterSystem<CBoundingVolume<TimeStep::Current>>(system);
}

// Get a world-space Aabb that contains CBoundingVolume for all members of a group (recursively).
static std::optional<Aabb> GetGroupWorldBounds(
    entt::registry const& reg,
    CGroupMembers const& members) {
  Vec4r min = {}, max = {};
  bool hasBounds = false;
  ForEachDescendant(reg, members, [&](auto e) {
    auto const* root = reg.try_get<CRootTransform const>(e);
    auto const* bounds = reg.try_get<CBoundingVolume<TimeStep::Current> const>(e);
    if (root && bounds) {
      Aabb worldAabb = GetAabb(TransformShape(root->worldFromLocal, bounds->localShape));
      if (hasBounds) {
        min = Min(min, worldAabb.VGetMin());
        max = Max(max, worldAabb.VGetMax());
      } else {
        min = worldAabb.VGetMin();
        max = worldAabb.VGetMax();
        hasBounds = true;
      }
    }
  });
  if (hasBounds) {
    return Aabb(min, max);
  } else {
    return std::nullopt;
  }
}

static void RegisterDebugDrawSystem_CompoundAabbWorld(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Compound AABB World";
  system.description = "Draw a wireframe box for the compound's world-space AABB.";
  system.sortingDepth -= 5_r; // Draw on top of default stuff, in case they overlap
  system.onDrawEntityWorldSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const& members = reg.get<CGroupMembers const>(e);
        if (auto worldAabb = GetGroupWorldBounds(reg, members)) {
          out.AddWireframeAabb(*worldAabb, MakeColor(0xFF8080FF));
        }
      };
  debugDraw.RegisterSystem<CGroupMembers>(system);
}

static void RegisterDebugDrawSystem_IslandAabbWorld(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Island AABB World (tight fit)";
  system.description = "Draw a wireframe box around the actors within an island.";
  system.sortingDepth -= 6_r; // Draw on top of default stuff, in case they overlap
  system.onDrawEntityWorldSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const& members = reg.get<CIslandMembers const>(e);
        if (auto worldAabb = GetGroupWorldBounds(reg, members)) {
          out.AddWireframeAabb(*worldAabb, MakeColor(0x8080FFFF));
        }
      };
  debugDraw.RegisterSystem<CIslandMembers>(system);
}

static void RegisterDebugDrawSystem_IslandAabbWorldConservative(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Island AABB World (conservative)";
  system.description =
      "Draw a wireframe box around the conservative step bounds of all actors within an island.";
  system.sortingDepth -= 7_r; // Draw on top of default stuff, in case they overlap
  system.onDrawEntityWorldSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const& members = reg.get<CIslandMembers const>(e);
        Vec4r min = {}, max = {};
        bool hasBounds = false;
        ForEachDescendant(reg, members, [&](auto e) {
          if (auto const* stepBounds = reg.try_get<CConservativeStepBounds const>(e)) {
            if (hasBounds) {
              min = Min(min, stepBounds->worldAabb.VGetMin());
              max = Max(max, stepBounds->worldAabb.VGetMax());
            } else {
              min = stepBounds->worldAabb.VGetMin();
              max = stepBounds->worldAabb.VGetMax();
              hasBounds = true;
            }
          }
        });
        if (hasBounds) {
          out.AddWireframeAabb(Aabb{min, max}, MakeColor(0xFF33FFFF));
        }
      };
  debugDraw.RegisterSystem<CIslandMembers>(system);
}

static void RegisterDebugDrawSystem_MeshSurface(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Mesh";
  system.description = "Draw wireframe triangles for the boundary of the actor's volume mesh.";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::SurfaceNodePositions, enabled, true);
  };
  system.onDrawEntityLocalSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const* query = reg.try_get<CQuerySurfaceNodePositions>(e);
        auto const& mesh = reg.get<CSurfaceMesh>(e);
        auto const* color = reg.try_get<CMeshColor>(e);
        if (query && !query->nodePositions.empty()) { // just in case
          out.AddWireframeMesh(
              Unflatten<Real3 const>(MakeSpan(query->nodePositions)),
              mesh.mesh->GetActiveNodesEdges(),
              color ? color->value : colors::kWhite);
        }
      };
  debugDraw.RegisterSystem<CSurfaceMesh>(system, ecs::Excluded<TagExcludedFromDebugDraw>{});
}

static void RegisterDebugDrawSystem_ActiveBoundaryFaces(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Mesh (Active Surface Elements)";
  system.description = "Draw wireframe triangles for the actor's active boundary faces.";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::NodePositions, enabled, true);
  };
  system.onDrawEntityLocalSpace = [](entt::registry const& reg,
                                     entt::entity e,
                                     DebugDrawCollector& out) {
    auto const* query = reg.try_get<CQueryNodePositions const>(e);
    auto const* tetMesh = reg.try_get<CTetrahedralMesh const>(e);
    auto const* activeBoundaryFaces = reg.try_get<CActiveBoundaryFaces const>(e);
    auto const chosenColor = colors::kPurple;

    if (query && tetMesh && tetMesh->mesh && activeBoundaryFaces && !activeBoundaryFaces->empty()) {
      auto const positions = Unflatten<Real3 const>(query->nodePositions);
      auto const connectivity = tetMesh->mesh->GetBoundaryFacesConnectivity();
      for (int i : activeBoundaryFaces->ViewIndices()) {
        auto const triangle = connectivity[i];
        DebugDrawLineVertex v1{.position = positions[triangle[0]], .color = chosenColor};
        DebugDrawLineVertex v2{.position = positions[triangle[1]], .color = chosenColor};
        DebugDrawLineVertex v3{.position = positions[triangle[2]], .color = chosenColor};
        out.AddLine(v1, v2);
        out.AddLine(v2, v3);
        out.AddLine(v3, v1);

        for (int k = 0; k < 3; ++k) {
          Real3 const coords = {
              positions[triangle[k]][0], positions[triangle[k]][1], positions[triangle[k]][2]};
          out.AddSphere(DebugDrawSphere{coords, 0.006, chosenColor});
        }
      }
    }
  };
  debugDraw.RegisterSystem<CSimplicialMesh>(system, ecs::Excluded<TagExcludedFromDebugDraw>{});
}

static void RegisterDebugDrawSystem_RodPolyline(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Rod Actor Polyline";
  system.description = "Draw deformed polyline centerline for rod actors";
  system.onDrawEntityLocalSpace = [](entt::registry const& reg,
                                     entt::entity e,
                                     DebugDrawCollector& out) {
    // Get the reference polyline mesh (undeformed configuration)
    auto const& polylineMesh = reg.get<CPolylineMesh>(e);

    // Get the current displacement + twist values from the rod pose
    auto const& currPose = reg.get<CRodPose<TimeStep::Current>>(e);
    auto const& dispTwist = currPose.value.displacements;

    int const numNodes = isize(polylineMesh.nodes);
    if (numNodes < 2) {
      return; // Need at least 2 nodes to draw a line
    }

    int const numElements = polylineMesh.NumElements();

    // Draw polyline segments
    Color const rodColor = colors::kOrange;
    for (int i = 0; i < numElements; ++i) {
      Int2 const en = polylineMesh.ElementNodes(i);
      // Get reference positions (undeformed)
      Real3 const& refPos0 = polylineMesh.nodes[en[0]];
      Real3 const& refPos1 = polylineMesh.nodes[en[1]];

      // Extract displacements from the solution vector
      int const dofBase0 = en[0] * fem::kNumRodFields;
      int const dofBase1 = en[1] * fem::kNumRodFields;
      Real3 const disp0{dispTwist[dofBase0 + 0], dispTwist[dofBase0 + 1], dispTwist[dofBase0 + 2]};
      Real3 const disp1{dispTwist[dofBase1 + 0], dispTwist[dofBase1 + 1], dispTwist[dofBase1 + 2]};

      // Compute deformed positions in local frame
      Real3 const localPos0 = refPos0 + disp0;
      Real3 const localPos1 = refPos1 + disp1;

      // Add line segment to debug draw
      out.AddLine(
          {.position = localPos0, .color = rodColor}, {.position = localPos1, .color = rodColor});
    }
  };
  debugDraw.RegisterSystem<CPolylineMesh, TagRodActor, CRodPose<TimeStep::Current>>(
      system, ecs::Excluded<TagExcludedFromDebugDraw>{});
}

static void RegisterDebugDrawSystem_RodElementFrameAxes(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Rod Element Frame Axes";
  system.description = "Draw local frame axes for rod elements to visualize twisting state";
  system.onDrawEntityLocalSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        // Get the reference polyline mesh (undeformed configuration)
        auto const& polylineMesh = reg.get<CPolylineMesh>(e);

        // Get the current rod pose (displacements + frame axes)
        auto const& currPose = reg.get<CRodPose<TimeStep::Current>>(e);
        auto const& dispTwist = currPose.value.displacements;
        auto const& axes = currPose.value.frameAxes;

        int const numNodes = isize(polylineMesh.nodes);
        if (numNodes < 2) {
          return; // Need at least 2 nodes to draw a line
        }

        int const numElements = polylineMesh.NumElements();

        // Draw local frame axes for each element
        for (int i = 0; i < numElements; ++i) {
          // Get reference position (undeformed)
          Real3 const& refPos0 = polylineMesh.nodes[i];

          // Each node has 4 DOFs: displacement (3) + twist (1)
          // Extract displacement from the solution vector
          Real3 const disp0{dispTwist[i * 4 + 0], dispTwist[i * 4 + 1], dispTwist[i * 4 + 2]};

          // Compute deformed position in local frame
          Real3 const localPos0 = refPos0 + disp0;

          // Add line segment for local frame axis to display twisting state
          Real3 const axis = axes[i];
          real constexpr kScale = 0.05_r;
          out.AddLine(
              {.position = localPos0, .color = colors::kRed},
              {.position = localPos0 + kScale * axis, .color = colors::kRed});
        }
      };
  debugDraw.RegisterSystem<CPolylineMesh, TagRodActor, CRodPose<TimeStep::Current>>(
      system, ecs::Excluded<TagExcludedFromDebugDraw>{});
}

static Color RedBlueColorMap(real val, real minVal, real maxVal) {
  if (val == -std::numeric_limits<real>::infinity()) {
    return colors::kBlue;
  }
  if (val == std::numeric_limits<real>::infinity()) {
    return colors::kRed;
  }
  MOCHI_ASSERT(minVal != -std::numeric_limits<real>::infinity());
  MOCHI_ASSERT(maxVal != std::numeric_limits<real>::infinity());
  MOCHI_ASSERT(minVal <= val);
  MOCHI_ASSERT(val <= maxVal);
  real scale = std::max(std::abs(maxVal), std::abs(minVal));
  if (scale == 0_r) {
    return colors::kWhite;
  } else if (val >= 0) {
    auto const alpha = (val / scale) * 255.0_r;
    return {
        static_cast<uint8_t>(255),
        static_cast<uint8_t>(255 - alpha),
        static_cast<uint8_t>(255 - alpha),
        static_cast<uint8_t>(255)};
  } else {
    auto const alpha = (std::abs(val) / scale) * 255.0_r;
    return {
        static_cast<uint8_t>(255 - alpha),
        static_cast<uint8_t>(255 - alpha),
        static_cast<uint8_t>(255),
        static_cast<uint8_t>(255)};
  }
};

static void RegisterDebugDrawSystem_BshDistanceSamples(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Bsh Distance Samples";
  system.description = "Draw wireframe points for distance.";
  system.onDrawEntityLocalSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const* bshManager = reg.try_get<rom::hyper::CDynamicSampleMeshBshManager const>(e);
        auto const* root = reg.try_get<CRootTransform const>(e);

        if (bshManager) {
          real maxVal = -std::numeric_limits<real>::infinity();
          real minVal = std::numeric_limits<real>::infinity();

          auto const& bsh = bshManager->value.GetBsh();
          for (int nodeIdx = 0; nodeIdx < isize(bsh); ++nodeIdx) {
            auto const& nodeData = bsh.GetNodeData(nodeIdx);
            if (nodeData.sampleIdx >= 0) {
              auto const& sample = bshManager->value.LastSampleData(nodeData.sampleIdx);
              if (sample.distance != -std::numeric_limits<real>::infinity()) {
                maxVal = std::max(maxVal, sample.distance);
                minVal = std::min(minVal, sample.distance);
              }
            }
          }
          for (int nodeIdx = 0; nodeIdx < isize(bsh); ++nodeIdx) {
            auto const& nodeData = bsh.GetNodeData(nodeIdx);
            if (nodeData.sampleIdx >= 0) {
              auto const& sample = bshManager->value.LastSampleData(nodeData.sampleIdx);
              out.AddSphere(
                  {.position = root ? root->worldFromLocal.TransformPointInverse(sample.position)
                                    : sample.position,
                   .radius = 0.01_r,
                   .color = RedBlueColorMap(sample.distance, minVal, maxVal)});
            }
          }
        }
      };
  debugDraw.RegisterSystem<CSimplicialMesh>(system, ecs::Excluded<TagExcludedFromDebugDraw>{});
}

static void RegisterDebugDrawSystem_BshProxyValues(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Bsh Proxy Values";
  system.description = "Draw wireframe points for distance.";
  system.onDrawEntityLocalSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const* bshManager = reg.try_get<rom::hyper::CDynamicSampleMeshBshManager const>(e);
        auto const* time = reg.try_ctx<CSceneTime const>();
        auto const* root = reg.try_get<CRootTransform const>(e);

        if (bshManager && time) {
          real maxVal = -std::numeric_limits<real>::infinity();
          real minVal = std::numeric_limits<real>::infinity();

          auto const& bsh = bshManager->value.GetBsh();
          for (int nodeIdx = 0; nodeIdx < isize(bsh); ++nodeIdx) {
            auto const& nodeData = bsh.GetNodeData(nodeIdx);
            if (nodeData.sampleIdx >= 0) {
              auto value = bshManager->value.EvaluateSdfLowerBound(nodeIdx, nodeData.position);
              if (value != -std::numeric_limits<real>::infinity()) {
                maxVal = std::max(maxVal, value);
                minVal = std::min(minVal, value);
              }
            }
          }

          for (int nodeIdx = 0; nodeIdx < isize(bsh); ++nodeIdx) {
            auto const& nodeData = bsh.GetNodeData(nodeIdx);
            if (nodeData.sampleIdx >= 0) {
              auto value = bshManager->value.EvaluateSdfLowerBound(nodeIdx, nodeData.position);
              out.AddSphere(
                  {root ? root->worldFromLocal.TransformPointInverse(nodeData.position)
                        : nodeData.position,
                   0.01_r,
                   RedBlueColorMap(value, minVal, maxVal)});
            }
          }
        }
      };
  debugDraw.RegisterSystem<CSimplicialMesh>(system, ecs::Excluded<TagExcludedFromDebugDraw>{});
}

static void RegisterDebugDrawSystem_BshStructure(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Bsh";
  system.description = "Draw Bsh Structure";
  system.onDrawEntityLocalSpace = [](entt::registry const& reg,
                                     entt::entity e,
                                     DebugDrawCollector& out) {
    auto const* bshManager = reg.try_get<rom::hyper::CDynamicSampleMeshBshManager const>(e);
    auto const* root = reg.try_get<CRootTransform const>(e);
    Color clr = colors::kWhite;

    if (bshManager) {
      auto const& bsh = bshManager->value.GetBsh();
      for (int nodeIdx = 0; nodeIdx < isize(bsh); ++nodeIdx) {
        auto const& nodeData = bsh.GetNodeData(nodeIdx);
        if (auto parentIdx = bsh.Parent(nodeIdx)) {
          auto const& parentData = bsh.GetNodeData(*parentIdx);
          if (nodeData.sampleIdx >= 0 && parentData.sampleIdx >= 0) {
            auto const& parentNode = bsh.GetNodeData(*parentIdx);

            DebugDrawLineVertex v1{
                .position = root ? root->worldFromLocal.TransformPointInverse(nodeData.position)
                                 : nodeData.position,
                .color = clr};
            DebugDrawLineVertex v2{
                .position = root ? root->worldFromLocal.TransformPointInverse(parentNode.position)
                                 : parentNode.position,
                .color = clr};
            out.AddLine(v1, v2);
          }
        }
      }
    }
  };
  debugDraw.RegisterSystem<CSimplicialMesh>(system, ecs::Excluded<TagExcludedFromDebugDraw>{});
}

// Component used by RegisterDebugDrawSystem_MeshSurfaceLocal to remember each actor's transform, as
// it was when the actor first spawned (or when this system was first enabled).
namespace mesh_surface_local {
struct CInitialRootTransform {
  TransformRT worldFromLocal;
};
} // namespace mesh_surface_local

static void RegisterDebugDrawSystem_MeshSurfaceLocal(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Mesh (freeze root)";
  system.description =
      "Draw a wireframe mesh for the surface of the actor, but freeze the actor's root\n"
      "transform wherever it is. This lets you see how the actor is moving in\n"
      "local-space (if at all).";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    // Remember the initial transform
    if (enabled) {
      auto& initialTransform = reg.get_or_emplace<mesh_surface_local::CInitialRootTransform>(e);
      initialTransform.worldFromLocal = reg.get<CRootTransform>(e).worldFromLocal;
    }

    // Ref count required components when enabled
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::SurfaceNodePositions, enabled, true);
  };
  system.onDrawEntityWorldSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const* query = reg.try_get<CQuerySurfaceNodePositions>(e);
        auto const& mesh = reg.get<CSurfaceMesh>(e);
        auto const& initialRoot = reg.get<mesh_surface_local::CInitialRootTransform>(e);
        // Freeze the pivot in world-space at the location where it spawned.
        TransformRT meshTransform = initialRoot.worldFromLocal;
        if (query && !query->nodePositions.empty()) {
          out.AddWireframeMesh(
              Unflatten<Real3 const>(MakeSpan(query->nodePositions)),
              mesh.mesh->GetBoundaryEdges(),
              colors::kWhite,
              meshTransform);
        }
      };
  debugDraw.RegisterSystem<CSurfaceMesh, CRootTransform, TagSoftActor>(system);
}

namespace mesh_deformation_local {
// Component used by RegisterDebugDrawSystem_MeshDeformationLocal to remember where to
// freeze the pivot for each actor.
struct CInitialPivotTransform {
  TransformRT worldFromPivot;
};
} // namespace mesh_deformation_local

static void RegisterDebugDrawSystem_MeshDeformationLocal(DebugDrawInternal& debugDraw) {
  // Component used by this system to remember where to freeze the pivot for each actor.

  DebugDrawSystem system;
  system.name = "Actor Mesh (freeze pivot)";
  system.description =
      "Draw a wireframe mesh for the surface of the actor, but freeze the actor's\n"
      "\"rigid pivot\" wherever it is. This lets you see how the actor is deforming\n"
      "independent of any global rotation or translation.";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    // Remember the initial transform
    if (enabled) {
      auto const& root = reg.get<CRootTransform>(e);
      auto& pivotEval = reg.get<CRigidTransformEval>(e);
      auto& initial = reg.get_or_emplace<mesh_deformation_local::CInitialPivotTransform>(e);
      initial.worldFromPivot = root.worldFromLocal * pivotEval.value;
    }

    // Ref count required components when enabled
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::NodePositions, enabled, true);
  };
  system.onDrawEntityWorldSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const* query = reg.try_get<CQueryNodePositions>(e);
        auto const& mesh = reg.get<CSimplicialMesh>(e);
        // auto const& root = reg.get<CRootTransform>(e);
        auto const& pivotEval = reg.get<CRigidTransformEval>(e);
        auto const& frozenWorldFromPivot =
            reg.get<mesh_deformation_local::CInitialPivotTransform>(e).worldFromPivot;
        // Freeze the pivot in world-space
        TransformRT meshTransform = frozenWorldFromPivot * Invert(pivotEval.value);
        if (query && !query->nodePositions.empty()) {
          out.AddWireframeMesh(
              Unflatten<Real3 const>(MakeSpan(query->nodePositions)),
              mesh.mesh->GetBoundaryEdges(),
              colors::kWhite,
              meshTransform);
        }
      };
  debugDraw.RegisterSystem<CSimplicialMesh, CRootTransform, CRigidTransformEval, TagSoftActor>(
      system);
}

static void RegisterDebugDrawSystem_MeshEnergy(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Mesh Elastic Energy";
  system.description =
      "Draw wireframe triangles for the boundary of the actor's volume mesh.\n"
      "Color mesh based on the elastic energy.";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    // Ref count required components when enabled
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::NodePositions, enabled, true);
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::ElasticEnergy, enabled, false);
  };
  system.onDrawEntityLocalSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const& mesh = reg.get<CSimplicialMesh>(e);
        auto const* posQuery = reg.try_get<CQueryNodePositions>(e);
        auto const* energyQuery = reg.try_get<CQueryElasticEnergy>(e);
        if (posQuery && !posQuery->nodePositions.empty() && energyQuery &&
            energyQuery->isEnergyAtRestInitialized) {
          // Compute mesh color based on elastic energy, as a percentage of an arbitrary "high
          // energy" threshold.
          constexpr real kHighEnergyMagnitude = 0.1_r;
          real mag = energyQuery->energy - energyQuery->energyAtRest;
          real t = mag / kHighEnergyMagnitude;

          // Ramp the color from no energy (green) to mid energy (yellow) to high energy (red)
          Color meshColor = colors::kBlack;
          meshColor[0] = static_cast<uint8_t>(RemapAndClamp(t, 0_r, 0.5_r, 0_r, 255_r));
          meshColor[1] = static_cast<uint8_t>(RemapAndClamp(t, 0.5_r, 1_r, 255_r, 0_r));
          out.AddWireframeMesh(
              Unflatten<Real3 const>(MakeSpan(posQuery->nodePositions)),
              mesh.mesh->GetBoundaryEdges(),
              meshColor);
        }
      };
  debugDraw.RegisterSystem<CSimplicialMesh>(system);
}

static void RegisterDebugDrawSystem_MeshInterior(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Mesh Interior";
  system.description =
      "Draw wireframe triangles for all the elements in the actor's volume mesh, including the interior.";
  system.sortingDepth +=
      1.0; // Draw behind default stuff in case they overlap (e.g. with the surface mesh)
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    // Ref count required components when enabled
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::NodePositions, enabled, true);
  };
  system.onDrawEntityLocalSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const* query = reg.try_get<CQueryNodePositions>(e);
        auto const& mesh = reg.get<CSimplicialMesh>(e);
        if (query && !query->nodePositions.empty()) { // just in case
          out.AddWireframeMesh(
              Unflatten<Real3 const>(MakeSpan(query->nodePositions)),
              mesh.mesh->GetEdges(),
              MakeColor(0x404040FF));
        }
      };
  debugDraw.RegisterSystem<CSimplicialMesh>(system);
}

static void RegisterDebugDrawSystem_NodeNormals(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Node Normals";
  system.description =
      "Draw a line for the normal of each boundary node on the actor's volume mesh.";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    // Ref count required components when enabled
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::SurfaceNodePositions, enabled, true);
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::SurfaceNodeNormals, enabled, true);
  };
  system.onDrawEntityLocalSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const* posQuery = reg.try_get<CQuerySurfaceNodePositions>(e);
        auto const* normQuery = reg.try_get<CQuerySurfaceNodeNormals>(e);
        if (posQuery && !posQuery->nodePositions.empty() && normQuery &&
            (posQuery->nodePositions.size() == normQuery->nodeNormals.size())) {
          Span<Real3 const> coords = Unflatten<Real3 const>(MakeSpan(posQuery->nodePositions));
          Span<Real3 const> normals = Unflatten<Real3 const>(MakeSpan(normQuery->nodeNormals));
          std::vector<LineVertex> verts(coords.size() * 2);
          Vec4r const scale = 0.01_r;
          int vi = 0;
          for (int i = 0; i < isize(coords); ++i) {
            verts[vi].position = coords[i];
            verts[vi].color = colors::kRed;
            ++vi;
            verts[vi].position = ToReal3(ToSimd(normals[i]) * scale + ToSimd(coords[i]));
            verts[vi].color = colors::kRed;
            ++vi;
          }
          out.AddLines(verts);
        }
      };
  debugDraw.RegisterSystem<CSurfaceMesh>(system);
}

static void RegisterDebugDrawSystem_ActorNodeBCs(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Node BCs";
  system.description = "Draw little boxes for each soft node with a boundary condition.";
  system.onDrawEntityWorldSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const& bcs = reg.get<CDofPositionsBC>(e);
        real constexpr kScale = 0.002_r;
        for (Real3 const& pt : Unflatten<Real3 const>(MakeSpan(bcs.poseValues))) {
          out.AddWireframeAabb(pt, kScale, colors::kGreen);
        }
      };
  debugDraw.RegisterSystem<CDofPositionsBC, TagSoftActor>(system);
}

static void RegisterDebugDrawSystem_SdfNormals(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor SDF Normals";
  system.description =
      "Draw lines for the surface normals of an actor's collision SDF (arbitrary sampling).";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    // Ref count required components when enabled
    AddRemoveOrRefComponent<CQuerySdfSurface>(reg, e, enabled); // Not a public QueryType
  };
  system.onDrawEntityLocalSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const* query = reg.try_get<CQuerySdfSurface>(e);
        if (query && (query->positions.size() == query->normals.size())) {
          std::vector<DebugDrawLineVertex> verts;
          verts.resize(2 * query->positions.size());
          constexpr real kScale = 0.005_r;
          int idx = 0;
          for (int i = 0; i < isize(query->positions); ++i) {
            Real3 const& pos = query->positions[i];
            Real3 const& norm = query->normals[i];
            verts[idx].position = pos;
            verts[idx].color = colors::kRed;
            ++idx;
            verts[idx].position = pos + norm * kScale;
            verts[idx].color = colors::kRed;
            ++idx;
          }
          out.AddLines(verts);
        }
      };
  debugDraw.RegisterSystem<CSdfCollider>(system);
}

static void RegisterDebugDrawSystem_VisualMesh(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Visual Mesh";
  system.description =
      "Draw wireframe triangles for the actor's visual mesh (high res mesh mapped to the simulation mesh).";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    // Ref count required components when enabled
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::VisualNodePositions, enabled, true);
  };
  system.onDrawEntityLocalSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const& vis = reg.get<CVisualMesh>(e);
        auto const* query = reg.try_get<CQueryVisualNodePositions>(e);
        size_t const numVisualNodes = vis.mesh->GetNumNodes();
        if (query && (query->nodePositions.size() == (numVisualNodes * kSpaceDim3))) {
          out.AddWireframeMesh(
              Unflatten<Real3 const>(MakeSpan(query->nodePositions)),
              vis.mesh->GetEdges(),
              colors::kWhite);
        }
      };
  debugDraw.RegisterSystem<CVisualMesh>(system);
}

static void RegisterDebugDrawSystem_VisualNormals(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Visual Normals";
  system.description = "Draw a line for the normal of each vertex in the actor's visual mesh.";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    // Ref count required components when enabled
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::VisualNodePositions, enabled, true);
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::VisualNodeNormals, enabled, true);
  };
  system.onDrawEntityLocalSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const* posQuery = reg.try_get<CQueryVisualNodePositions>(e);
        auto const* normQuery = reg.try_get<CQueryVisualNodeNormals>(e);
        auto const& vis = reg.get<CVisualMesh>(e);
        if (posQuery && !posQuery->nodePositions.empty() && normQuery &&
            (posQuery->nodePositions.size() == normQuery->nodeNormals.size())) {
          Span<Real3 const> coords = Unflatten<Real3 const>(MakeSpan(posQuery->nodePositions));
          Span<Real3 const> normals = Unflatten<Real3 const>(MakeSpan(normQuery->nodeNormals));
          int const numNodes = vis.mesh->GetNumNodes();
          std::vector<LineVertex> verts(numNodes * 2);
          Vec4r const scale = 0.01_r;
          size_t vi = 0;
          for (int i = 0; i < numNodes; ++i) {
            verts[vi].position = coords[i];
            verts[vi].color = colors::kRed;
            ++vi;
            verts[vi].position = ToReal3(ToSimd(normals[i]) * scale + ToSimd(coords[i]));
            verts[vi].color = colors::kRed;
            ++vi;
          }
          out.AddLines(verts);
        }
      };
  debugDraw.RegisterSystem<CVisualMesh>(system);
}

static void RegisterDebugDrawSystem_ContactSamples(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Contact Samples";
  system.description = "Draw a little grey box at each potential contact sample";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    // Ref count required components when enabled
    AddRemoveOrRefComponent<CQueryContactSamples>(reg, e, enabled); // not a public QueryType
  };
  system.onDrawEntityLocalSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const* samplesQuery = reg.try_get<CQueryContactSamples>(e);
        if (samplesQuery && !samplesQuery->contactSamples.empty()) {
          Span<Real3 const> points = Unflatten<Real3 const>(MakeSpan(samplesQuery->contactSamples));
          for (auto const& point : points) {
            out.AddSphere(DebugDrawSphere{point, 0.0005_r, colors::kGrey});
          }
        }
      };
  debugDraw.RegisterSystem<CContactSamples<TimeStep::Current>>(system);
}

// Traverses the contact sample bounding sphere hierarchy from the root and emits one colored
// sphere per node relevant to [targetDepth]. Nodes sitting exactly at [targetDepth] are drawn in
// light orange. Branches that bottom out before reaching [targetDepth] (leaf nodes at a shallower
// depth) are drawn in light blue, so the union of drawn spheres always covers the full sample set.
static void DrawBshLevel(BvhTree<Sphere> const& bvh, int targetDepth, DebugDrawCollector& out) {
  Color constexpr kAtDepthColor = MakeColor(0xFFB26680); // light orange
  Color constexpr kEarlyLeafColor = MakeColor(0x66B2FF80); // light blue

  if (bvh.GetNodeCount() == 0) {
    return;
  }

  // Explicit (nodeIndex, depth) stack to avoid recursion.
  DynamicArray<std::pair<int, int>> stack;
  stack.emplace_back(BvhTree<Sphere>::kRootNode, 0);
  while (!stack.empty()) {
    auto const [nodeIndex, depth] = stack.back();
    stack.pop_back();
    auto const& node = bvh.GetNode(nodeIndex);
    if (depth == targetDepth) {
      out.AddSphere(DebugDrawSphere{node.bv.GetCenter(), node.bv.GetRadius(), kAtDepthColor});
    } else if (node.isLeafNode) {
      // Cannot refine further; draw the early leaf to keep coverage complete.
      out.AddSphere(DebugDrawSphere{node.bv.GetCenter(), node.bv.GetRadius(), kEarlyLeafColor});
    } else {
      stack.emplace_back(node.leftChildIndex, depth + 1);
      stack.emplace_back(node.rightChildIndex, depth + 1);
    }
  }
}

static void RegisterDebugDrawSystem_ContactSamplesBsh(DebugDrawInternal& debugDraw) {
  constexpr int kMaxDepth = 9;
  for (int depth = 0; depth <= kMaxDepth; ++depth) {
    DebugDrawSystem system;
    system.name = Format("Actor Contact Samples BSH (level %.2d)", depth);
    system.description =
        Format("Draw the contact sample bounding sphere hierarchy at tree depth %d.", depth);
    system.onDrawEntityLocalSpace =
        [depth](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
          auto const& samples = reg.get<CContactSamples<TimeStep::Current>>(e);
          if (samples.bsh.has_value()) {
            DrawBshLevel(samples.bsh->GetBvh(), depth, out);
          }
        };
    debugDraw.RegisterSystem<CContactSamples<TimeStep::Current>>(system);
  }
}

static void RegisterDebugDrawSystem_QuadraturePoints(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Quadrature Points";
  system.description =
      "Draw a little red sphere at each quadrature point and deform according to def gradient";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    // Ref count required components when enabled
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::ElementsDeformationGradient, enabled);
    AddRemoveOrRefComponent<CQueryQuadraturePointsPosition>(
        reg, e, enabled); // not a public query type
  };
  system.onDrawEntityLocalSpace = [](entt::registry const& reg,
                                     entt::entity e,
                                     DebugDrawCollector& out) {
    auto const* quadratureQuery = reg.try_get<CQueryQuadraturePointsPosition>(e);
    if (quadratureQuery && !quadratureQuery->quadraturePointsWorldPosition.empty()) {
      Span<Real3 const> points =
          Unflatten<Real3 const>(MakeSpan(quadratureQuery->quadraturePointsWorldPosition));
      // Plots the quadrature points in the world space and scales
      // their size based on the Frobenius norm of the deformation gradient.
      auto const* deformationGradientQuery = reg.try_get<CQueryElementsDeformationGradient>(e);

      // If we did compute the deformation gradient, we can use it to scale the size of the
      // quadrature points. Otherwise, we just use the default size.
      if (deformationGradientQuery &&
          !deformationGradientQuery->elementsDeformationGradient.empty()) {
        // The deformation gradient is stored as a 3x3 matrix in a flattened array.
        constexpr int kDeformationGradSize = 9;
        Span<NdArray<real, kDeformationGradSize> const> deformationGradients =
            Unflatten<NdArray<real, kDeformationGradSize> const>(
                MakeSpan(deformationGradientQuery->elementsDeformationGradient));

        MOCHI_ASSERT(
            deformationGradients.size() == points.size(),
            "Mismatching number of quadrature points for deformation gradient and positions");

        for (int i = 0; i < points.size(); ++i) {
          auto const& point = points[i];
          auto const& defGradient = deformationGradients[i];
          // An arbitrary scaling factor to make the boxes more visible when deformed
          constexpr real kScaleFactor = 100_r;
          real scale = 1_r + kScaleFactor * std::sqrt(std::abs(Norm(defGradient) - std::sqrt(3_r)));
          out.AddSphere(
              DebugDrawSphere{
                  .position = point, .radius = scale * 0.0004_r, .color = colors::kRed});
        }
      } else {
        for (auto const& point : points) {
          out.AddSphere(
              DebugDrawSphere{.position = point, .radius = 0.0004_r, .color = colors::kRed});
        }
      }
    }
  };
  debugDraw.RegisterSystem<CContactSamples<TimeStep::Current>>(system);
}

static void RegisterDebugDrawSystem_ContactDistances(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor SDF Distances";
  system.description = "Draw a point representing distance to nearest collider at a contact point.";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::SdfDistances, enabled);
  };
  system.onDrawEntityWorldSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const& query = reg.get<CQuerySdfDistances>(e);

        MOCHI_ASSERT(isize(query.distances) == isize(query.worldPositions));

        if (query.distances.empty()) {
          return; // Avoid dereferencing if distances are empty
        }
        auto maxDist = *std::max_element(query.distances.begin(), query.distances.end());
        auto minDist = *std::min_element(query.distances.begin(), query.distances.end());

        for (int i = 0; i < isize(query.distances); ++i) {
          DebugDrawSphere s;
          s.position = query.worldPositions[i];
          s.color = RedBlueColorMap(query.distances[i], minDist, maxDist);
          s.radius = 0.004_r;
          out.AddSphere(s);
        }
      };
  debugDraw.RegisterSystem<CActiveCollisions<ContactType::Async, TimeStep::Current>>(system);
}

static void RegisterDebugDrawSystem_ActiveContactForces(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Active Contact Forces";
  system.description =
      "Draw a green line line representing the velocity of each active contact point.";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::ContactPoints, enabled);
  };
  system.onDrawEntityWorldSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        SceneHandle sceneHandle = reg.ctx<CSceneHandle const>().value;
        auto const& query = reg.get<CQueryContactPoints>(e);
        std::vector<LineVertex> verts(query.contactPoints.size() * 2);
        size_t vi = 0;
        for (auto const& contact : query.contactPoints) {
          Real3 force = contact.force;
          Real3 forceOrigin = contact.posA;
          Color c = colors::kBlue;
          if (contact.actorB == GetActorHandle(e, sceneHandle)) {
            // Show the direction of force as it was applied to entity e through sync coupling.
            // Change the color so we can see the difference visually.
            force = -force;
            forceOrigin = contact.posB;
            c = colors::kCyan;
          }
          verts[vi].position = forceOrigin;
          verts[vi].color = c;
          ++vi;
          static constexpr real kForceScalar = 20_r;
          verts[vi].position = forceOrigin + (force * kForceScalar);
          verts[vi].color = c;
          ++vi;
        }
        out.AddLines(verts);
      };
  // Actors with CRequiresFarSdfEvaluation do not support contact queries
  debugDraw.RegisterSystem<CActiveCollisions<ContactType::Async, TimeStep::Current>>(
      system, ecs::Excluded<CRequiresFarSdfEvaluation>{});
}

static void RegisterDebugDrawSystem_ActiveContactNormals(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Active Contact Normals";
  system.description = "Draw a red line for the normal direction of each active contact point.";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::ContactPoints, enabled);
  };
  system.onDrawEntityWorldSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const& query = reg.get<CQueryContactPoints>(e);
        std::vector<LineVertex> verts(query.contactPoints.size() * 2);
        size_t vi = 0;
        for (auto const& contact : query.contactPoints) {
          // Normal
          verts[vi].position = contact.posB;
          verts[vi].color = colors::kRed;
          ++vi;
          verts[vi].position = contact.posB + (contact.normal * 0.005_r);
          verts[vi].color = colors::kRed;
          ++vi;
        }
        out.AddLines(verts);
      };
  // Actors with CRequiresFarSdfEvaluation do not support contact queries
  debugDraw.RegisterSystem<CActiveCollisions<ContactType::Async, TimeStep::Current>>(
      system, ecs::Excluded<CRequiresFarSdfEvaluation>{});
}

static void RegisterDebugDrawSystem_ActiveContactPositions(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Active Contact Positions";
  system.description = "Draw a little sphere at each active contact point";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::ContactPoints, enabled);
    // The following is added to test the correctness of contact parametric info
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::SurfaceNodePositions, enabled);
  };
  system.onDrawEntityWorldSpace = [](entt::registry const& reg,
                                     entt::entity e,
                                     DebugDrawCollector& out) {
    auto const& query = reg.get<CQueryContactPoints const>(e);
    if (query.contactPoints.empty()) {
      return;
    }

#if MOCHI_ASSERT_VERBOSE_ENABLED
    // Get the actor's surface mesh for validation, unless it's a rod actor
    Span<Real3 const> coords = {};
    Span<Int3 const> faces = {};
    TransformRT const* transform = {};
    bool const isRodActor = reg.all_of<TagRodActor>(e);
    // Contact samples are computed from the DIRK stage variable during the Newton
    // solve, while surface node positions reflect the full-step displacement computed
    // in PostLastStageLocalPipeline. These differ when the stage variable is not the
    // full-step value (e.g., implicit midpoint with c=0.5).
    bool const stageEqualsStepEnd = [&] {
      auto const* intState = reg.try_get<CTimeIntegratorState const>(e);
      return intState && isize(intState->bTilde) == 1 && intState->bTilde[0] == 1_r;
    }();
    if (!isRodActor && stageEqualsStepEnd) {
      coords = Unflatten<Real3 const>(reg.get<CQuerySurfaceNodePositions const>(e).nodePositions);
      faces = Unflatten<Int3 const>(
          reg.get<CSurfaceMesh const>(e).mesh->GetActiveNodesFlatConnectivity());
      transform = &reg.get<CRootTransform const>(e).worldFromLocal;
    }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

    SceneHandle sceneHandle = reg.ctx<CSceneHandle const>().value;
    for (auto const& contact : query.contactPoints) {
      out.AddSphere(
          DebugDrawSphere{.position = contact.posB, .radius = 0.0004_r, .color = colors::kMagenta});

      if (GetActorHandle(e, sceneHandle) == contact.actorB) {
        continue;
      }

#if MOCHI_ASSERT_VERBOSE_ENABLED
      if (isRodActor || !stageEqualsStepEnd) {
        continue;
      }
      // Contact validation.
      Real3 pc = contact.parametricCoords;
      Int3 tri = faces[contact.elementIndex];
      Real3 v[3] = {coords[tri[0]], coords[tri[1]], coords[tri[2]]};
      Real3 ptOnTriangle = transform->TransformPoint(v[0] * pc[0] + v[1] * pc[1] + v[2] * pc[2]);
      MOCHI_ASSERT_VERBOSE(
          NearEqual(contact.posA, ptOnTriangle, 5e-4_r), "Error in contact parametric info");
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
    }
  };
  // Actors with CRequiresFarSdfEvaluation do not support contact queries
  debugDraw.RegisterSystem<CActiveCollisions<ContactType::Async, TimeStep::Current>>(
      system, ecs::Excluded<CRequiresFarSdfEvaluation>{});
}

static void RegisterDebugDrawSystem_ActiveContactVelocities(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Active Contact Velocities";
  system.description =
      "Draw a green line line representing the velocity of each active contact point.";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::ContactPoints, enabled);
  };
  system.onDrawEntityWorldSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const& query = reg.get<CQueryContactPoints>(e);
        std::vector<LineVertex> verts(query.contactPoints.size() * 4);
        size_t vi = 0;
        for (auto const& contact : query.contactPoints) {
          // Velocity
          verts[vi].position = contact.posA;
          verts[vi].color = colors::kCyan;
          ++vi;
          verts[vi].position = contact.posA + (contact.pointVelocityA * 0.1_r);
          verts[vi].color = colors::kCyan;
          ++vi;
          verts[vi].position = contact.posB;
          verts[vi].color = colors::kBlue;
          ++vi;
          verts[vi].position = contact.posB + (contact.pointVelocityB * 0.1_r);
          verts[vi].color = colors::kBlue;
          ++vi;
        }
        out.AddLines(verts);
      };
  // Actors with CRequiresFarSdfEvaluation do not support contact queries
  debugDraw.RegisterSystem<CActiveCollisions<ContactType::Async, TimeStep::Current>>(
      system, ecs::Excluded<CRequiresFarSdfEvaluation>{});
}

static void RegisterDebugDrawSystem_NodeContactForces(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Node Contact Forces";
  system.description = "Draw a magenta line representing the contact force on each surface node.";
  system.onEntityEnable = [](entt::registry& reg, entt::entity e, bool enabled) {
    if (reg.any_of<TagRodActor>(e)) {
      return;
    }
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::NodeContactForces, enabled);
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::NodePositions, enabled);
    AddRemoveOrRefComponentsForQuery(reg, e, QueryType::SurfaceNodePositions, enabled);
  };
  system.onDrawEntityWorldSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const& transform = reg.get<CRootTransform const>(e).worldFromLocal;
        auto const& contacts = reg.get<CQueryNodeContactForces const>(e).nodeContactForces;
        auto const& volumePositions = reg.get<CQueryNodePositions const>(e);
        auto const& surfacePositions = reg.get<CQuerySurfaceNodePositions const>(e);
        auto positions = Unflatten<Real3 const>(MakeSpan(
            volumePositions.nodePositions.empty() ? surfacePositions.nodePositions
                                                  : volumePositions.nodePositions));
        std::vector<LineVertex> verts(contacts.size() * 2);
        size_t vi = 0;
        for (auto const& contact : contacts) {
          auto pos = transform.TransformPoint(positions[contact.index]);
          verts[vi].position = pos;
          verts[vi].color = colors::kMagenta;
          ++vi;
          static constexpr real kForceScalar = 20_r;
          verts[vi].position = pos + (contact.force * kForceScalar);
          verts[vi].color = colors::kMagenta;
          ++vi;
        }
        out.AddLines(verts);
      };
  // Rod actors and actors with CRequiresFarSdfEvaluation do not support contact queries
  debugDraw.RegisterSystem<CActiveCollisions<ContactType::Async, TimeStep::Current>>(
      system, ecs::Excluded<CRequiresFarSdfEvaluation, TagRodActor>{});
}

static void RegisterDebugDrawSystem_Collider(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Actor Collider";
  system.description = "Draw a wireframe representing the current collider of the actor";
  system.onDrawEntityLocalSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        Color color = colors::kTeal;
        color[3] = 64;

        // Box collider
        auto const* boxColl = reg.try_get<CBoxCollider>(e);
        if (boxColl) {
          out.AddWireframeObb(boxColl->shape, color);
        }

        // Sphere collider
        auto const* sphereColl = reg.try_get<CSphereCollider>(e);
        if (sphereColl) {
          out.AddSphere(
              DebugDrawSphere{sphereColl->shape.GetCenter(), sphereColl->shape.GetRadius(), color});
        }

        // Mesh collider
        auto const* meshColl = reg.try_get<CMeshCollider>(e);
        if (meshColl) {
          out.AddWireframeMesh(meshColl->GetCoordinates(), meshColl->GetMesh()->GetEdges(), color);
        }
      };

  debugDraw.RegisterSystem<CColliderInfo>(system);
}

static void RegisterDebugDrawSystem_PointCloudCollider(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Point Cloud Collider";
  system.description =
      "Draw spheres at collider sample points of point-cloud colliders, sized by the collider "
      "radius";
  system.onDrawEntityLocalSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const& params = reg.get<CPointCloudColliderParams>(e);
        auto const& disc = reg.get<CColliderPointCloudDiscretization>(e);
        auto const& dispRef = reg.get<CFinalDisplacementRef<TimeStep::Current>>(e);
        real const radius = params.radius;
        int const dofsPerNode = disc.dofsPerNode;
        ColumnVectorView<real const> disp = dispRef.value;

        Color color = colors::kTeal;
        color[3] = 64;

        disc.VisitCollider([&](auto const& discretization) {
          using DiscretizationT = std::decay_t<decltype(discretization)>;
          int const numElements = isize(discretization.femElements);
          for (int elementIndex = 0; elementIndex < numElements; ++elementIndex) {
            auto const& element = discretization.femElements[elementIndex];
            for (int q = 0; q < DiscretizationT::kNumQuads; ++q) {
              Real3 const pos = details::InterpolateColliderPointPosition<DiscretizationT>(
                  element, elementIndex, q, disp, dofsPerNode);
              out.AddSphere({pos, radius, color});
            }
          }
        });
      };

  debugDraw.RegisterSystem<
      TagUsePointCloudContact,
      CPointCloudColliderParams,
      CColliderPointCloudDiscretization,
      CFinalDisplacementRef<TimeStep::Current>>(system);
}

static void RegisterDebugDrawSystem_PotentialColliders(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Potential Colliders";
  system.description = "Draw a purple line between pairs of actors in potential contact";
  system.onDrawEntityWorldSpace = [](entt::registry const& reg,
                                     entt::entity e,
                                     DebugDrawCollector& out) {
    auto const& actorRoot = reg.get<CRootTransform const>(e);
    auto const& actorBv = reg.get<CBoundingVolume<TimeStep::Current> const>(e);
    auto const actorCenter =
        actorRoot.worldFromLocal.TransformPoint(GetBoundingSphere(actorBv.localShape).GetCenter());

    // Async and sync colliders
    auto registerCollisionsFunc = [&](Span<PotentialColliderData const> colls) {
      for (auto const& coll : colls) {
        auto const& collRoot = reg.get<CRootTransform const>(coll.entity);
        auto const& collBv = reg.get<CBoundingVolume<TimeStep::Current> const>(coll.entity);
        auto const collCenter = collRoot.worldFromLocal.TransformPoint(
            GetBoundingSphere(collBv.localShape).GetCenter());
        LineVertex verts[2];
        verts[0].position = actorCenter;
        verts[0].color = colors::kPurple;
        verts[1].position = collCenter;
        verts[1].color = colors::kPurple;
        out.AddLines(verts);
      }
    };
    if (auto const* asyncColliders =
            reg.try_get<CPotentialColliders<ContactType::Async> const>(e)) {
      registerCollisionsFunc(*asyncColliders);
    }
    if (auto const* syncColliders = reg.try_get<CPotentialColliders<ContactType::Sync> const>(e)) {
      registerCollisionsFunc(*syncColliders);
    }
  };

  debugDraw.RegisterSystem<CRootTransform, CBoundingVolume<TimeStep::Current>, TagUseContact>(
      system);
}

static void RegisterDebugDrawSystem_RigidVelocity(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Rigid Velocity";
  system.description = "Draw a cyan/magenta lines representing actor's linear/angular velocity";
  system.onDrawEntityWorldSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        auto const& worldFromLocal = reg.get<CRootTransform const>(e).worldFromLocal;
        auto const& rigidVelocity = reg.get<CPrevRigidVelocity const>(e);
        auto comWorld = worldFromLocal.TransformPoint(rigidVelocity.centerOfMassLocal);
        LineVertex verts[2];
        verts[0].position = ToReal3(comWorld);
        verts[1].position = ToReal3(comWorld + rigidVelocity.linearVelocityWorld * 0.05_r);
        verts[0].color = verts[1].color = colors::kCyan;
        out.AddLines(verts);
        verts[1].position = ToReal3(comWorld + rigidVelocity.angularVelocityWorld * 0.05_r);
        verts[0].color = verts[1].color = colors::kMagenta;
        out.AddLines(verts);
      };

  debugDraw.RegisterSystem<CRootTransform, CPrevRigidVelocity>(system);
}

static void RegisterDebugDrawSystem_ArticulatedTargetPose(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Articulated Target Pose";
  system.description =
      "Draw wireframe meshes for the links of an articulated actor at the target pose of\n"
      "its pose controller. This visualizes where the controller is trying to move the actor.";
  system.sortingDepth -= 1_r; // Draw slightly on top of normal visualization

  system.onDrawEntityWorldSpace =
      [](entt::registry const& reg, entt::entity e, DebugDrawCollector& out) {
        // Get the target link transforms from the controller
        auto const& props = reg.get<CArticulatedProps const>(e);
        std::vector<TransformRT> targetTransforms(props.numLinks);
        Error error;
        articulated::compound::GetTargetLinkTransforms(reg, e, MakeSpan(targetTransforms), error);
        if (!error.IsOK()) {
          return;
        }

        // Draw each link's mesh at its target transform
        auto const& members = reg.get<CGroupMembers const>(e);
        Color constexpr kTargetColor = MakeColor(0x40FF40A0); // Semi-transparent green

        for (int i = 0; i < isize(members.actors); ++i) {
          auto link = members.actors[i];

          // Get the surface mesh for this link (contains reference/rest positions)
          auto const* surfaceMesh = reg.try_get<CSurfaceMesh const>(link);

          if (surfaceMesh && surfaceMesh->mesh) {
            // Get the reference positions directly from the mesh - no query needed for rigid bodies
            Span<Real3 const> refCoords = surfaceMesh->mesh->GetActiveNodeCoordinates();
            Span<Int2 const> edges = surfaceMesh->mesh->GetActiveNodesEdges();

            // Draw at the target transform (reference coords are in local space)
            TransformRT const& targetRoot = targetTransforms[i];
            out.AddWireframeMesh(refCoords, edges, kTargetColor, targetRoot);
          }
        }
      };

  // Register with required components - only for articulated actors with pose controllers
  debugDraw.RegisterSystem<CArticulatedProps, CGroupMembers, CControllerTarget<TimeStep::Current>>(
      system, ecs::Excluded<TagExcludedFromDebugDraw>{});
}

// Arrowhead is a small wireframe "X" of 4 line segments fanning back from the tip; it is sized as
// a fraction of the shaft length but clamped to an absolute maximum so short arrows still look
// proportionate. `kArrowheadAngleTan` is tan(half-angle) of the head.
static real constexpr kArrowheadFraction = 0.25_r;
static real constexpr kArrowheadMaxLen = 0.02_r;
static real constexpr kArrowheadAngleTan = 0.35_r; // ~19 degrees

// Draws an arrow from `from` to `to`: a shaft line plus a 4-segment wireframe "X" arrowhead.
// A (near-)zero-length shaft draws only the shaft (no arrowhead) to avoid a degenerate direction.
static void AddArrow(DebugDrawCollector& out, Real3 const& from, Real3 const& to, Color color) {
  out.AddLine({.position = from, .color = color}, {.position = to, .color = color});
  Real3 const shaft = to - from;
  real const shaftLen = Norm(shaft);
  if (shaftLen <= std::numeric_limits<real>::min()) {
    return;
  }
  Real3 const dir = shaft * (1_r / shaftLen);
  real const headLen = Min(kArrowheadMaxLen, kArrowheadFraction * shaftLen);
  real const halfWidth = headLen * kArrowheadAngleTan;
  Real3 const back = to - headLen * dir;
  Real3 const u = Normalize(OrthogonalVector(dir)); // unit-length
  Real3 const v = Cross(dir, u); // unit-length
  std::array<Real3, 4> const backPts = {
      back + halfWidth * u,
      back - halfWidth * u,
      back + halfWidth * v,
      back - halfWidth * v,
  };
  std::array<LineVertex, 8> headVerts;
  for (int i = 0; i < 4; ++i) {
    headVerts[2 * i] = {.position = to, .color = color};
    headVerts[2 * i + 1] = {.position = backPts[i], .color = color};
  }
  out.AddLines(MakeSpan(headVerts));
}

// Approximates a circle of the given `radius` centered at `center`, in the plane orthogonal to
// `axis`, as line segments.
static void AddCircle(
    DebugDrawCollector& out,
    Real3 const& center,
    Real3 const& axis,
    real radius,
    Color color) {
  int constexpr kNumSegments = 32;
  Real3 const u = Normalize(OrthogonalVector(axis)); // unit-length
  Real3 const v = Cross(axis, u); // unit-length (u, v, axis are orthonormal)
  real constexpr kDTheta = 2_r * kPI / static_cast<real>(kNumSegments);
  std::array<LineVertex, 2 * kNumSegments> circleVerts;
  Real3 prev = center + radius * u;
  for (int i = 0; i < kNumSegments; ++i) {
    real const theta = static_cast<real>(i + 1) * kDTheta;
    Real3 const curr = center + radius * (Cos(theta) * u + Sin(theta) * v);
    circleVerts[2 * i] = {.position = prev, .color = color};
    circleVerts[2 * i + 1] = {.position = curr, .color = color};
    prev = curr;
  }
  out.AddLines(MakeSpan(circleVerts));
}

// World-space frame of a single-DoF (fixed / linear) joint, plus its type and current DoF value.
// Shared by the SpatialTendon and LinearTransmission debug-draw systems.
namespace {
struct FixedJointFrame {
  Real3 originWorld;
  Real3 axisWorld;
  ArticulatedJointType type;
  real dofValue;
};
} // namespace

// Computes the world-space frame of the single-DoF joint at `jointIndex` for gizmo drawing.
// Active joints are index-aligned with their child links; only cycle joints break this invariant,
// and those are rejected by the type assertion before accessing link or pose data.
//
// `restTransforms[child].boneFromInner.GetTranslation()` is `t_joint - t_com` (the joint origin in
// the child link's local frame minus the child's CoM in local frame), which is already the joint
// origin expressed in the child's CoM frame (see `CreateRestTransforms()` in articulated_body.cpp).
// It therefore pairs directly with `linkTransforms[child]` (world-from-CoM). The axis is already
// normalized at joint construction; CoM and local frames share the same rotation, so
// `TransformDirection` is correct without re-normalization.
//
// The DoF value is read at the reduced-*pose* offset (`jointPoseInfo[...].offset`, NOT the
// reduced-DoF offset `dofInfo[...].offset`) because the reduced pose and reduced DoF vectors
// diverge in offsets whenever the chain contains a free, spherical, or hard joint (their per-joint
// pose and DoF sizes differ). This matches `SpatialTendon`'s own length computation.
static FixedJointFrame ComputeFixedJointFrame(
    int jointIndex,
    JointsData const& joints,
    CArticulatedLinkTransforms<TimeStep::Current> const& linkTransforms,
    CArticulatedRestTransforms const& restTransforms,
    CArticulatedJointPoseInfo const& jointPoseInfo,
    CArticulatedReducedPose<TimeStep::Current> const& reducedPose) {
  ArticulatedJointType const type = joints.jointTypes[jointIndex];
  bool const isSingleDofJoint =
      type == ArticulatedJointType::Revolute || type == ArticulatedJointType::Prismatic;
  MOCHI_ASSERT(
      isSingleDofJoint,
      "ComputeFixedJointFrame expects a single-DoF (revolute or prismatic) joint.");

  int const child = jointIndex;
  Real3 const originCom = restTransforms[child].boneFromInner.GetTranslation();
  return FixedJointFrame{
      .originWorld = linkTransforms[child].TransformPoint(originCom),
      .axisWorld = linkTransforms[child].TransformDirection(joints.jointAxes[jointIndex]),
      .type = type,
      .dofValue = reducedPose.value[jointPoseInfo[jointIndex].offset]};
}

// Draws the symbolic fixed-joint gizmo: an axis arrow whose length and sign equal the term's
// contribution (`coefficient * dofValue`, in [m]) and — for revolute joints — a circle of radius
// `|coefficient|` (the moment arm) in the plane orthogonal to the joint axis. Does not draw the
// joint-origin marker or connecting polyline; callers own those so per-system coloring is
// preserved.
static void DrawFixedJointGizmo(
    DebugDrawCollector& out,
    FixedJointFrame const& frame,
    real coefficient,
    Color color) {
  Real3 const tipWorld = frame.originWorld + coefficient * frame.dofValue * frame.axisWorld;
  AddArrow(out, frame.originWorld, tipWorld, color);
  if (frame.type == ArticulatedJointType::Revolute) {
    AddCircle(out, frame.originWorld, frame.axisWorld, Abs(coefficient), color);
  }
}

static void RegisterDebugDrawSystem_SpatialTendon(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Spatial Tendon";
  system.description =
      "Draw the routed path (line segments + waypoint markers) of spatial tendons on\n"
      "articulated actors. Linear-joint terms contribute a symbolic gizmo: a yellow\n"
      "marker at the joint origin, plus a cyan axis arrow whose length and sign equal\n"
      "the term's contribution to tendon length (`coefficient * dofValue`). Revolute\n"
      "joints additionally draw a cyan circle of radius `|coefficient|` (the moment\n"
      "arm) centered at the joint origin and orthogonal to the joint axis.";
  system.sortingDepth -= 1_r; // Draw slightly on top of normal visualization

  // The link transforms are already world-space, so emit world-space data directly.
  system.onDrawEntityWorldSpace = [](entt::registry const& reg,
                                     entt::entity e,
                                     DebugDrawCollector& out) {
    auto const& transmissions = reg.get<CTransmissions const>(e).transmissions;
    auto const& linkTransforms = reg.get<CArticulatedLinkTransforms<TimeStep::Current> const>(e);
    auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
    MOCHI_ASSERT_VERBOSE(
        joints != nullptr, "An articulated actor with transmissions must have joint data.");
    auto const& jointPoseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
    auto const& restTransforms = reg.get<CArticulatedRestTransforms const>(e);
    auto const& reducedPose = reg.get<CArticulatedReducedPose<TimeStep::Current> const>(e);

    Color constexpr kWaypointLineColor = colors::kPurple;
    Color constexpr kWaypointMarkerColor = colors::kOrange;
    Color constexpr kLinearJointLineColor = colors::kCyan;
    Color constexpr kLinearJointMarkerColor = colors::kYellow;
    Color constexpr kLinearJointGizmoColor = colors::kCyan;
    // Absolute world-space marker size; tune if tendons span very different scales.
    real constexpr kMarkerRadius = 0.005_r;

    for (auto const& transmission : transmissions) {
      // Only spatial tendons have a routed path; skip other transmission types.
      auto const* tendon = dynamic_cast<SpatialTendon const*>(transmission.get());
      if (!tendon) {
        continue;
      }

      // Single pass over the routing elements. Each element contributes exactly one polyline
      // vertex (flagged as either a true-waypoint vertex or a linear-joint visualization vertex);
      // a segment between consecutive vertices is colored cyan if either endpoint is a
      // linear-joint vertex, purple otherwise. Linear-joint elements additionally emit a symbolic
      // gizmo (axis arrow + circle for revolute joints); the linear-joint displacement is conveyed
      // by that axis arrow, not by a polyline segment.
      // `element.localPosition` (Waypoint) is in each link's CoM frame (the internal
      // representation returned by `SpatialTendon::GetRoutingElements()`), so it pairs with the
      // CoM-frame `CArticulatedLinkTransforms<Current>`. For LinearJoint elements we use
      // `restTransforms[child].boneFromInner.GetTranslation()`, which is the joint origin already
      // expressed in the child's CoM frame (= `t_joint - t_com` in the link's local frame; see
      // the implementation of `CreateRestTransforms()`).
      bool hasPrev = false;
      bool prevIsLinearJoint = false;
      Real3 prevWorld{};

      auto const emitVertex = [&](Real3 const& world, bool isLinearJoint) {
        Color const markerColor = isLinearJoint ? kLinearJointMarkerColor : kWaypointMarkerColor;
        out.AddSphere({.position = world, .radius = kMarkerRadius, .color = markerColor});
        if (hasPrev) {
          Color const segmentColor =
              (isLinearJoint || prevIsLinearJoint) ? kLinearJointLineColor : kWaypointLineColor;
          out.AddLine(
              {.position = prevWorld, .color = segmentColor},
              {.position = world, .color = segmentColor});
        }
        prevWorld = world;
        prevIsLinearJoint = isLinearJoint;
        hasPrev = true;
      };

      for (auto const& element : tendon->GetRoutingElements()) {
        if (element.type == RoutingElementType::Waypoint) {
          Real3 const world = linkTransforms[element.index].TransformPoint(element.localPosition);
          emitVertex(world, /*isLinearJoint=*/false);
          continue;
        }

        // LinearJoint: emit the joint-origin marker (+ polyline segment) then the symbolic gizmo.
        FixedJointFrame const frame = ComputeFixedJointFrame(
            element.index, *joints, linkTransforms, restTransforms, jointPoseInfo, reducedPose);
        emitVertex(frame.originWorld, /*isLinearJoint=*/true);
        DrawFixedJointGizmo(out, frame, element.coefficient, kLinearJointGizmoColor);
      }
    }
  };

  debugDraw.RegisterSystem<
      CTransmissions,
      CArticulatedLinkTransforms<TimeStep::Current>,
      CArticulatedBodyShape,
      CArticulatedJointPoseInfo,
      CArticulatedRestTransforms,
      CArticulatedReducedPose<TimeStep::Current>>(
      system, ecs::Excluded<TagExcludedFromDebugDraw>{});
}

static void RegisterDebugDrawSystem_LinearTransmission(DebugDrawInternal& debugDraw) {
  DebugDrawSystem system;
  system.name = "Linear Transmission Terms";
  system.description =
      "Draw a symbolic gizmo for each fixed-joint term of a linear transmission on an\n"
      "articulated actor: a yellow marker at the joint origin, plus a cyan axis arrow whose\n"
      "length and sign equal the term's contribution to the transmission displacement\n"
      "(`coefficient * dofValue`). Revolute joints additionally draw a cyan circle of radius\n"
      "`|coefficient|` (the moment arm) centered at the joint origin and orthogonal to the\n"
      "joint axis. Consecutive term origins are joined by a connecting polyline.";
  system.sortingDepth -= 1_r; // Draw slightly on top of normal visualization

  // The link transforms are already world-space, so emit world-space data directly.
  system.onDrawEntityWorldSpace = [](entt::registry const& reg,
                                     entt::entity e,
                                     DebugDrawCollector& out) {
    auto const& transmissions = reg.get<CTransmissions const>(e).transmissions;
    auto const& linkTransforms = reg.get<CArticulatedLinkTransforms<TimeStep::Current> const>(e);
    auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
    MOCHI_ASSERT_VERBOSE(
        joints != nullptr, "An articulated actor with transmissions must have joint data.");
    auto const& jointPoseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
    auto const& restTransforms = reg.get<CArticulatedRestTransforms const>(e);
    auto const& reducedPose = reg.get<CArticulatedReducedPose<TimeStep::Current> const>(e);

    Color constexpr kLineColor = colors::kCyan;
    Color constexpr kMarkerColor = colors::kYellow;
    Color constexpr kGizmoColor = colors::kCyan;
    // Absolute world-space marker size; tune if transmissions span very different scales.
    real constexpr kMarkerRadius = 0.005_r;

    for (auto const& transmission : transmissions) {
      // Only linear transmissions expose fixed-joint terms; skip other transmission types (a
      // SpatialTendon is not a LinearTransmission, so tendons are handled by their own system).
      auto const* linear = dynamic_cast<LinearTransmission const*>(transmission.get());
      if (!linear) {
        continue;
      }

      // Each term draws a marker + gizmo at its joint origin; consecutive origins are joined by a
      // connecting polyline.
      bool hasPrev = false;
      Real3 prevWorld{};
      for (auto const& term : linear->GetTerms()) {
        FixedJointFrame const frame = ComputeFixedJointFrame(
            term.jointIndex, *joints, linkTransforms, restTransforms, jointPoseInfo, reducedPose);
        out.AddSphere(
            {.position = frame.originWorld, .radius = kMarkerRadius, .color = kMarkerColor});
        if (hasPrev) {
          out.AddLine(
              {.position = prevWorld, .color = kLineColor},
              {.position = frame.originWorld, .color = kLineColor});
        }
        prevWorld = frame.originWorld;
        hasPrev = true;
        DrawFixedJointGizmo(out, frame, term.coefficient, kGizmoColor);
      }
    }
  };

  debugDraw.RegisterSystem<
      CTransmissions,
      CArticulatedLinkTransforms<TimeStep::Current>,
      CArticulatedBodyShape,
      CArticulatedJointPoseInfo,
      CArticulatedRestTransforms,
      CArticulatedReducedPose<TimeStep::Current>>(
      system, ecs::Excluded<TagExcludedFromDebugDraw>{});
}

// This suppresses a warning about no prior declaration of the function.
// There is no header for this cpp, but that's OK.
void RegisterDebugDrawSystems(DebugDrawInternal& debugDraw);

// Called once by mochi::Scene to register all the DebugDrawSystems.
void RegisterDebugDrawSystems(DebugDrawInternal& debugDraw) {
  // Transform
  RegisterDebugDrawSystem_ActorRootTransform(debugDraw);
  RegisterDebugDrawSystem_ActorRigidPivotEvalPoint(debugDraw);

  // Dynamics
  RegisterDebugDrawSystem_ActorRigidBodyInertia(debugDraw);

  // Bounding
  RegisterDebugDrawSystem_ActorAabbLocal(debugDraw);
  RegisterDebugDrawSystem_ActorAabbWorld(debugDraw);
  RegisterDebugDrawSystem_ActorAabbWorldConservative(debugDraw);
  RegisterDebugDrawSystem_CompoundAabbWorld(debugDraw);
  RegisterDebugDrawSystem_IslandAabbWorld(debugDraw);
  RegisterDebugDrawSystem_IslandAabbWorldConservative(debugDraw);

  // Simulation
  RegisterDebugDrawSystem_ActorNodeBCs(debugDraw);
  RegisterDebugDrawSystem_MeshEnergy(debugDraw);
  RegisterDebugDrawSystem_QuadraturePoints(debugDraw);

  // Geometry
  RegisterDebugDrawSystem_MeshSurface(debugDraw);
  RegisterDebugDrawSystem_MeshSurfaceLocal(debugDraw);
  RegisterDebugDrawSystem_VisualMesh(debugDraw);
  RegisterDebugDrawSystem_VisualNormals(debugDraw);
  RegisterDebugDrawSystem_MeshDeformationLocal(debugDraw);
  RegisterDebugDrawSystem_MeshInterior(debugDraw);
  RegisterDebugDrawSystem_NodeNormals(debugDraw);
  RegisterDebugDrawSystem_SdfNormals(debugDraw);
  RegisterDebugDrawSystem_ActiveBoundaryFaces(debugDraw);
  RegisterDebugDrawSystem_RodPolyline(debugDraw);
  RegisterDebugDrawSystem_RodElementFrameAxes(debugDraw);

  // Collisions
  RegisterDebugDrawSystem_Collider(debugDraw);
  RegisterDebugDrawSystem_PointCloudCollider(debugDraw);
  RegisterDebugDrawSystem_PotentialColliders(debugDraw);
  RegisterDebugDrawSystem_ContactSamples(debugDraw);
  RegisterDebugDrawSystem_ContactSamplesBsh(debugDraw);
  RegisterDebugDrawSystem_ActiveContactForces(debugDraw);
  RegisterDebugDrawSystem_ActiveContactNormals(debugDraw);
  RegisterDebugDrawSystem_ActiveContactPositions(debugDraw);
  RegisterDebugDrawSystem_ActiveContactVelocities(debugDraw);
  RegisterDebugDrawSystem_NodeContactForces(debugDraw);
  RegisterDebugDrawSystem_RigidVelocity(debugDraw);
  RegisterDebugDrawSystem_ContactDistances(debugDraw);

  // Hyper-Reduction
  RegisterDebugDrawSystem_BshDistanceSamples(debugDraw);
  RegisterDebugDrawSystem_BshProxyValues(debugDraw);
  RegisterDebugDrawSystem_BshStructure(debugDraw);

  // Pose controller
  RegisterDebugDrawSystem_ArticulatedTargetPose(debugDraw);

  // Transmissions
  RegisterDebugDrawSystem_SpatialTendon(debugDraw);
  RegisterDebugDrawSystem_LinearTransmission(debugDraw);

  // Do this last
  debugDraw.FinalizeSystems();
}

namespace debug_draw_systems {

// This suppresses a warning about no prior declaration of the function.
// There is no header for this cpp, but that's OK.
void InitializeOnce(entt::registry& reg);

void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<mesh_deformation_local::CInitialPivotTransform>(reg);
  ecs::RegisterComponent<mesh_surface_local::CInitialRootTransform>(reg);
}
} // namespace debug_draw_systems

} // namespace mochi
