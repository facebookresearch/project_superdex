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

#include <mochi_core/utils/container_utils.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/src/mochi_point_cloud_contact.h>
#include <mochi_physics/src/mochi_solve.h>
#include "mochi_core/test/mochi_test_helpers.h"
#include "mochi_physics_test_fixture.h"

#include <gtest/gtest.h>

#include <array>
#include <functional>
#include <utility>

using namespace mochi;
using namespace mochi::experimental;

namespace {

// Returns a new mesh consisting of the input mesh plus a second copy translated by `translation`.
// The second copy's connectivity indices are offset by the original vertex count so the two copies
// remain topologically separate within a single (disconnected) mesh.
std::pair<DynamicArray<Real3>, DynamicArray<Int3>> UnionWithTranslatedCopy(
    Span<Real3 const> coordinates,
    Span<Int3 const> connectivity,
    Real3 const& translation) {
  int const numVertices = isize(coordinates);

  DynamicArray<Real3> unionCoordinates;
  unionCoordinates.reserve(2 * coordinates.size());
  Append(unionCoordinates, coordinates);
  for (Real3 const& c : coordinates) {
    unionCoordinates.push_back(c + translation);
  }

  DynamicArray<Int3> unionConnectivity;
  unionConnectivity.reserve(2 * connectivity.size());
  Append(unionConnectivity, connectivity);
  for (Int3 const& t : connectivity) {
    unionConnectivity.push_back(Int3{t[0] + numVertices, t[1] + numVertices, t[2] + numVertices});
  }

  return {std::move(unionCoordinates), std::move(unionConnectivity)};
}

// Returns the island whose descendant actors contain all of `actors`, or entt::null if none does.
entt::entity FindIslandContaining(entt::registry& reg, Span<entt::entity const> actors) {
  entt::entity islandEntity = entt::null;
  reg.view<TagIsland, CIslandDescendants const>().each(
      [&](entt::entity island, CIslandDescendants const& descendants) {
        bool containsAll = true;
        for (entt::entity actor : actors) {
          containsAll = containsAll && Contains(descendants.actors, actor);
        }
        if (containsAll) {
          islandEntity = island;
        }
      });
  return islandEntity;
}

// Runs the collision-detection and contact-assembly pipeline for the given island and assembles the
// contact residuals into its CIslandContactSnle, which is returned for force extraction.
CIslandContactSnle& AssembleIslandContactResiduals(
    entt::registry& reg,
    entt::entity islandEntity,
    GradTarget gradTarget) {
  auto const& descendants = reg.get<CIslandDescendants const>(islandEntity);
  auto const& islandDofInfo = reg.get<CIslandDofInfo const>(islandEntity);
  auto& contactSnle = reg.get<CIslandContactSnle>(islandEntity);

  TaskSemaphore updateJacobianSem;
  solver::UpdateJacobiansSubpipeline(updateJacobianSem, reg, gradTarget, descendants);
  CollisionDetectionPipeline<TimeStep::Current>(reg, descendants);
  ContactJacobiansPipeline(reg, gradTarget, descendants, updateJacobianSem);

  AssemblyParams assemblyParams;
  assemblyParams.gradTarget = gradTarget;
  assemblyParams.assemObj = false;
  assemblyParams.assemRes = true;
  assemblyParams.assemDRes = false;
  AssembleIslandSyncContact(
      assemblyParams, false /*useBlockSparse3x3*/, reg, islandDofInfo, descendants, contactSnle);
  return contactSnle;
}

// Sums the local-frame contact force (i.e. -residual) over the DOF sub-range
// [dofOffset, dofOffset + numDofs), taking only the first three (spatial) components of each
// `dofsPerNode` node block, and only nodes for which `includeNode(localNodeIndex)` is true.
Real3 SumLocalContactForce(
    CIslandContactSnle const& snle,
    int dofOffset,
    int numDofs,
    int dofsPerNode,
    std::function<bool(int)> const& includeNode) {
  Real3 forceLocal{0_r, 0_r, 0_r};
  for (auto const& [offset, residual] : snle.residuals) {
    for (int i = 0; i < isize(residual); ++i) {
      int const globalDofIdx = offset + i;
      if (globalDofIdx < dofOffset || globalDofIdx >= dofOffset + numDofs) {
        continue;
      }
      int const localDofIdx = globalDofIdx - dofOffset;
      int const localNodeIndex = localDofIdx / dofsPerNode;
      int const component = localDofIdx % dofsPerNode;
      if (component < kSpaceDim3 && includeNode(localNodeIndex)) {
        forceLocal[component] -= residual[i];
      }
    }
  }
  return forceLocal;
}

} // namespace

class MochiShellContactTest : public ::testing::Test {
 protected:
  void SetUp() override {
    _params.radius = 0.02_r;
    _params.selfContactExclusionRatio = 1.5_r;
    _params.spatialHashLoadFactor = 0.0625_r;
    _params.selfContact = false;
    CreateSimpleTriangularDiscretization();
  }

  // Helper function to create a CFemSurfaceDiscretizationP1Q1 with multiple triangular elements.
  void CreateSimpleTriangularDiscretization() {
    // Create coordinates for a 2x2 grid of nodes forming a square (4 triangles)
    _coordinates = {
        Real3{0_r, 0_r, 0_r}, // Node 0: bottom-left
        Real3{1_r, 0_r, 0_r}, // Node 1: bottom-right
        Real3{0_r, 1_r, 0_r}, // Node 2: top-left
        Real3{1_r, 1_r, 0_r}, // Node 3: top-right
        Real3{0.5_r, 0.5_r, 0_r} // Node 4: center
    };

    // Create connectivity for 4 triangular elements forming a square
    _connectivity = {
        Int3{0, 1, 4}, // Bottom triangle
        Int3{1, 3, 4}, // Right triangle
        Int3{3, 2, 4}, // Top triangle
        Int3{2, 0, 4} // Left triangle
    };

    // Create multiple triangular elements
    int constexpr kNumElements = 4;
    _discretization.femElements.reserve(kNumElements);
    for (int i = 0; i < kNumElements; ++i) {
      _discretization.femElements.emplace_back(
          i, // element index
          MakeConstSpan(_coordinates),
          MakeConstSpan(_connectivity));
    }
  }

  PointCloudColliderParams _params;
  DynamicArray<Real3> _coordinates;
  DynamicArray<Int3> _connectivity;
  CFemSurfaceDiscretizationP1Q1 _discretization;
  std::unique_ptr<TriangularMesh> _colliderMesh;

  CColliderPointCloudDiscretization MakeColliderDiscretization(
      std::optional<ActorBoundaryElementType> colliderTriangleElementType) {
    if (!colliderTriangleElementType.has_value()) {
      NodalColliderDiscretization nodalDisc;
      DynamicArray<real> const weights = InitializeNodalWeights(_discretization);
      int const numNodes = isize(_coordinates);
      nodalDisc.femElements.reserve(numNodes);
      for (int i = 0; i < numNodes; ++i) {
        nodalDisc.femElements.emplace_back(i, MakeConstSpan(_coordinates), i, weights[i]);
      }
      return CColliderPointCloudDiscretization(std::move(nodalDisc), kSpaceDim3);
    }
    _colliderMesh =
        std::make_unique<TriangularMesh>(MakeConstSpan(_coordinates), MakeConstSpan(_connectivity));
    return CColliderPointCloudDiscretization(
        CFemSurfaceDiscretization::Create(colliderTriangleElementType.value(), *_colliderMesh),
        kSpaceDim3);
  }
};

// Parameterized fixture for tests that exercise both collider discretization types.
class MochiPointCloudColliderTest
    : public MochiShellContactTest,
      public ::testing::WithParamInterface<std::optional<ActorBoundaryElementType>> {};

// Test CreateSpatialHashTable function
TEST_P(MochiPointCloudColliderTest, CreateSpatialHashTable_BasicProperties) {
  auto colliderDisc = MakeColliderDiscretization(GetParam());
  int const numColliderPoints = colliderDisc.GetNumColliderPoints();

  SpatialHashTable hashTable =
      CreateSpatialHashTable(_params, colliderDisc, ContactParams{}.GetPenaltyThresholdDist(true));

  // Check that the hash table has the expected capacity
  EXPECT_EQ(hashTable.GetCapacity(), numColliderPoints);

  // Initially, the hash table should be empty
  EXPECT_EQ(hashTable.GetNumPoints(), 0);
}

// Test UpdateSpatialHashTable function
TEST_P(MochiPointCloudColliderTest, UpdateSpatialHashTable_BasicProperties) {
  auto colliderDisc = MakeColliderDiscretization(GetParam());
  int const numColliderPoints = colliderDisc.GetNumColliderPoints();
  int const numNodes = isize(_discretization.femElements[0].coordinates);
  int const numDofs = numNodes * kSpaceDim3;

  // Create displacement storage (initialized to zero) and a reference view for the API.
  CDisplacementSlice<real, TimeStep::Current> dispSlice(numDofs);
  CFinalDisplacementRef<TimeStep::Current> dispRef(dispSlice.value);

  // Create hash table with appropriate capacity
  SpatialHashTable hashTable{_params.radius, numColliderPoints, 10};

  // It should initially be empty
  EXPECT_EQ(hashTable.GetNumPoints(), 0);

  // Update the hash table; need to tag an entity as a shell actor, and add it to a potential
  // collider list to pass the built-in filter that skips unnecessary hash table updates.
  entt::registry reg;
  entt::entity e = reg.create();
  reg.emplace<TagShellActor>(e);
  CConservativePotentialColliders<ContactType::Sync> potentialColliders;
  potentialColliders.emplace_back(e);
  UpdateSpatialHashTable(
      ecs::Included<TagUsePointCloudContact>{}, colliderDisc, dispRef, hashTable);

  // Check that all collider points were added
  EXPECT_EQ(hashTable.GetNumPoints(), numColliderPoints);

  // Test reset functionality
  hashTable.Reset();
  EXPECT_EQ(hashTable.GetNumPoints(), 0);
}

INSTANTIATE_TEST_SUITE_P(
    ColliderPointTypes,
    MochiPointCloudColliderTest,
    ::testing::Values(
        std::nullopt,
        std::optional{ActorBoundaryElementType::P1Q1},
        std::optional{ActorBoundaryElementType::P1Q3}));

// Test InitializeNodalWeights function
TEST_F(MochiShellContactTest, InitializeNodalWeights_BasicProperties) {
  // Assemble nodal weights
  DynamicArray<real> weights = InitializeNodalWeights(_discretization);

  // Check that we have the right number of weights (5 nodes in our mesh: 4 corners + 1 center)
  int constexpr kNumNodes = 5;
  EXPECT_EQ(isize(weights), kNumNodes);

  // All weights should be positive
  for (int i = 0; i < kNumNodes; i++) {
    EXPECT_GT(weights[i], 0_r) << "Weight " << i << " should be positive";
  }

  // The sum of all nodal weights should be equal to 1, because the mesh is a unit square
  real totalWeight = 0_r;
  for (int i = 0; i < kNumNodes; i++) {
    totalWeight += weights[i];
  }
  EXPECT_NEAR(totalWeight, 1_r, 1e-5);
}

struct ShellContactTestParams {
  Quaternion rotation;
  int upAxis;
  bool explicitNormals;
  std::optional<ActorBoundaryElementType> colliderTriangleElementType;
};

// This sets up two initially-flat shell actors, with one hovering just slightly above the other,
// for integration testing of shell/shell contact.
class MochiShellActorContactScene : public test::MochiSceneTestBase,
                                    public ::testing::WithParamInterface<ShellContactTestParams> {
 protected:
  Actor* _bottomActor = nullptr;
  Actor* _topActor = nullptr;

 public:
  static real constexpr kScale = 1_r;
  static int constexpr kM = MOCHI_DEBUG ? 8 : 16;
  static int constexpr kN = MOCHI_DEBUG ? 8 : 16;
  static real constexpr kGravityMagnitude = 10_r;
  static real constexpr kBcEps = 1e-3_r;
  static real constexpr kContactRadius = kScale / (real)Min(kM, kN);
  static real constexpr kContactPenalty = 1e7_r;
  // Begin slightly out-of-range of the contact forces.
  static real constexpr kSeparationDistance = 1.5_r * kContactRadius;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();

    ShellContactTestParams const& params = GetParam();

    SolverParams solverParams = _scene->GetSolverParams();
    solverParams.experimentalEval.explicitNormals = params.explicitNormals;
    _scene->SetSolverParams(solverParams, ErrorAssert{});
    Quaternion const& rotation = params.rotation;
    int const upAxis = params.upAxis;

    auto& reg = GetRegistry();
    auto&& [coordinates, connectivity] =
        UniformSquareTriangularMeshData(Int2{kM, kN}, Real2{kScale, kScale});

    Real3 gravityDirection{0_r, 0_r, 0_r};
    gravityDirection[upAxis] = -kGravityMagnitude;
    _scene->SetGravity(gravityDirection);

    ShellActorParams bottomParams;
    bottomParams.shape = _scene->GetContext()->CreateTriMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), ErrorAssert{});
    bottomParams.pointCloudCollider.radius = kContactRadius;
    bottomParams.pointCloudCollider.selfContact = false;
    bottomParams.pointCloudCollider.colliderTriangleElementType =
        params.colliderTriangleElementType;
    bottomParams.contact.penaltyCoefficient = kContactPenalty;
    bottomParams.worldFromLocal.SetRotation(rotation);

    _bottomActor = CreateShellActor(_scene, bottomParams, ErrorAssert{});
    EXPECT_NE(entt::entity{}, mochi::GetEntity(reg, _bottomActor->GetHandle(), test::ExpectOK{}));

    ConstrainNodesByPosition(
        _bottomActor,
        [rotation](int, Real3 const& x) -> bool {
          Real3 const xLocal = Conjugate(rotation) * x;
          return (Abs(xLocal[0]) > 0.5_r * kScale - kBcEps) ||
              (Abs(xLocal[1]) > 0.5_r * kScale - kBcEps);
        },
        ErrorAssert{});

    ShellActorParams topParams;
    topParams.shape = _scene->GetContext()->CreateTriMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), ErrorAssert{});
    topParams.pointCloudCollider.radius = kContactRadius;
    topParams.pointCloudCollider.selfContact = false;
    topParams.pointCloudCollider.colliderTriangleElementType = params.colliderTriangleElementType;
    topParams.contact.penaltyCoefficient = kContactPenalty;

    Real3 separationVector{0_r, 0_r, 0_r};
    separationVector[upAxis] = kSeparationDistance;
    topParams.worldFromLocal.SetRotation(rotation);
    topParams.worldFromLocal.SetTranslation(separationVector);

    _topActor = CreateShellActor(_scene, topParams, ErrorAssert{});
    EXPECT_NE(entt::entity{}, mochi::GetEntity(reg, _topActor->GetHandle(), test::ExpectOK{}));
  }

  void TearDown() override {
    _scene->DestroyActor(_bottomActor);
    _scene->DestroyActor(_topActor);
    test::MochiSceneTestBase::TearDown();
  }
};

// This tests two parallel shell actors, with one constrained, and the other one freeling falling
// onto the constrained one. It then runs some basic sanity checks on the final configuration, to
// verify that contact is effective. Particular emphasis is placed on ensuring that the results are
// robust to rotations applied through actor-level world-from-local transforms, which have been a
// source of subtle bugs during implementation.
TEST_P(MochiShellActorContactScene, TwoShellActorsWithContact) {
  ShellContactTestParams const& params = GetParam();
  int const upAxis = params.upAxis;

  real constexpr kTimeInterval = 10_r;
  int constexpr kNumSteps = 8 * Max(kM, kN);
  real constexpr kTimeStep = kTimeInterval / (real)kNumSteps;

  for (int i = 0; i < kNumSteps; ++i) {
    _scene->Step(kTimeStep);
  }

  int const numDofsBottom = _bottomActor->GetNumDofs();
  int const numNodesBottom = numDofsBottom / 3;
  DynamicArray<real> dofValuesBottom(numDofsBottom);
  _bottomActor->GetDofValues({}, dofValuesBottom, ErrorAssert{});

  int const numDofsTop = _topActor->GetNumDofs();
  int const numNodesTop = numDofsTop / 3;
  DynamicArray<real> dofValuesTop(numDofsTop);
  _topActor->GetDofValues({}, dofValuesTop, ErrorAssert{});

  // Because the rotation of actor geometry is done via a world-from-local transform, the
  // z-displacement in the local coordinates is always aligned with gravity, even when gravity is
  // rotated to a different axis in world coordinates.
  real minZBottom = std::numeric_limits<real>::infinity();
  for (int nodeIndex = 0; nodeIndex < numNodesBottom; nodeIndex++) {
    minZBottom = Min(dofValuesBottom[3 * nodeIndex + upAxis], minZBottom);
  }

  real minZTop = std::numeric_limits<real>::infinity();
  real avgXTop = 0_r;
  real avgYTop = 0_r;
  for (int nodeIndex = 0; nodeIndex < numNodesTop; nodeIndex++) {
    minZTop = Min(dofValuesTop[3 * nodeIndex + upAxis], minZTop);
    avgXTop += dofValuesTop[3 * nodeIndex + 0];
    avgYTop += dofValuesTop[3 * nodeIndex + 1];
  }
  // The top actor's minimum local z coordinate is transformed here to the bottom actor's local
  // coordinate frame for comparison in the asserts.
  minZTop += kSeparationDistance;
  avgXTop /= (real)numNodesTop;
  avgYTop /= (real)numNodesTop;

  EXPECT_GT(minZTop, minZBottom) << "Top shell penetrated bottom shell: minZTop = " << minZTop
                                 << ", minZBottom = " << minZBottom;

  // This is not based on any tight calculation, but we generally don't expect the shell actors to
  // get significantly closer than the contact radius for a reasonable penalty value.
  real constexpr kMinSeparation = kContactRadius * 0.5_r;
  EXPECT_GT(minZTop - minZBottom, kMinSeparation)
      << "Shell separation too small: " << (minZTop - minZBottom);

  real constexpr kMaxDrift = kContactRadius * 0.1_r;
  EXPECT_LT(Abs(avgXTop), kMaxDrift) << "Top shell drifted too far in x: " << avgXTop;
  EXPECT_LT(Abs(avgYTop), kMaxDrift) << "Top shell drifted too far in y: " << avgYTop;

  // Some additional testing in the final configuration:

  // Check that shell contact forces between the two actors are equal and opposite by running the
  // full collision detection and contact assembly pipeline.
  auto& reg = GetRegistry();
  entt::entity bottomEntity = mochi::GetEntity(reg, _bottomActor->GetHandle(), test::ExpectOK{});
  entt::entity topEntity = mochi::GetEntity(reg, _topActor->GetHandle(), test::ExpectOK{});

  // Find the island that contains both actors.
  std::array<entt::entity, 2> const islandActors{bottomEntity, topEntity};
  entt::entity const islandEntity = FindIslandContaining(reg, MakeConstSpan(islandActors));
  ASSERT_TRUE(islandEntity != entt::null) << "No island contains both shell actors";

  // Get transforms for force conversion.
  TransformRT const& bottomTransform = reg.get<CRootTransform>(bottomEntity).worldFromLocal;
  TransformRT const& topTransform = reg.get<CRootTransform>(topEntity).worldFromLocal;

  // Get DOF offsets to extract per-actor residuals.
  int const bottomDofOffset = reg.get<CDofOffset>(bottomEntity).dofsOffset;
  int const topDofOffset = reg.get<CDofOffset>(topEntity).dofsOffset;

  // WARNING: The subsequent part of the test is a bit fragile in how it handles friction. It relies
  // on stage-start components being in the same state as in the last solve.

  // Run the collision detection + contact assembly pipeline for the island.
  CIslandContactSnle const& contactSnle =
      AssembleIslandContactResiduals(reg, islandEntity, GradTarget::Current);

  // Sum up contact forces for each actor in local frame from the contact SNLE residuals. Shell has
  // 3 spatial DOFs per node, and all nodes of each actor are included.
  auto const includeAllNodes = [](int) -> bool { return true; };
  Real3 const bottomForceLocal = SumLocalContactForce(
      contactSnle, bottomDofOffset, numDofsBottom, kSpaceDim3, includeAllNodes);
  Real3 const topForceLocal =
      SumLocalContactForce(contactSnle, topDofOffset, numDofsTop, kSpaceDim3, includeAllNodes);

  // Transform forces from local to world frame
  Real3 const bottomForceWorld = bottomTransform.GetRotation() * bottomForceLocal;
  Real3 const topForceWorld = topTransform.GetRotation() * topForceLocal;

  // Verify that the contact forces are nonzero, so the test is not trivial (i.e., the actors have
  // not somehow ended up in a non-contacting final configuration).
  real const forceScale =
      Max(Abs(bottomForceWorld[0]), Max(Abs(bottomForceWorld[1]), Abs(bottomForceWorld[2])));
  EXPECT_GT(forceScale, 1e-3_r) << "Contact forces are too small";

  // Verify that the net forces are equal and opposite in world frame
  real constexpr kForceTol = 3e-5_r;
  real const scaledTol = kForceTol * forceScale;
  EXPECT_NEAR_TOL(bottomForceWorld + topForceWorld, Real3{}, scaledTol);

  // Query world-space contact forces and compare with the manually-computed ones. The query uses
  // the previous-step configuration, so an additional step is needed to populate the query results.
  _bottomActor->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  _topActor->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  _scene->Step(kTimeStep);
  Real3 const bottomForceQueried = _bottomActor->GetContactForceWorld(test::ExpectOK{});
  Real3 const topForceQueried = _topActor->GetContactForceWorld(test::ExpectOK{});

  // Check that the queried forces match the manually-computed ones (from the previous step, which
  // is what the query uses). The comparison tolerance is larger because the configuration may have
  // changed slightly between the two steps.
  EXPECT_NEAR_TOL(bottomForceQueried, bottomForceWorld, scaledTol);
  EXPECT_NEAR_TOL(topForceQueried, topForceWorld, scaledTol);
}

INSTANTIATE_TEST_SUITE_P(
    UpAxisVariations,
    MochiShellActorContactScene,
    ::testing::Values(
        ShellContactTestParams{
            Quaternion::Identity(),
            2,
            false,
            std::optional{ActorBoundaryElementType::P1Q1}},
        // Include one case with explicit normals, to ensure that the code path is exercised.
        ShellContactTestParams{
            Quaternion::Identity(),
            2,
            true,
            std::optional{ActorBoundaryElementType::P1Q1}},
        ShellContactTestParams{
            Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, kPI / 2_r),
            0,
            false,
            std::optional{ActorBoundaryElementType::P1Q1}},
        ShellContactTestParams{
            Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, -kPI / 2_r),
            1,
            false,
            std::optional{ActorBoundaryElementType::P1Q1}},
        ShellContactTestParams{Quaternion::Identity(), 2, false, std::nullopt},
        // Include one case with explicit normals for Nodal, to match Quadrature coverage.
        ShellContactTestParams{Quaternion::Identity(), 2, true, std::nullopt},
        ShellContactTestParams{
            Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, kPI / 2_r),
            0,
            false,
            std::nullopt}));

// ---------------------------------------------------------------------------
// Shell self-contact integration test
// ---------------------------------------------------------------------------

// This sets up a single shell actor whose mesh is the union of two topologically-separated square
// regions (an upper region hovering above a lower one), for integration testing of shell
// self-contact. The upper region falls under gravity onto the lower region, exercising the
// self-collision detection and contact-assembly path. Because the contact force is internal to one
// actor, the net contact force on it is zero (the per-region forces form an action/reaction pair).
class MochiShellSelfContactScene : public test::MochiSceneTestBase,
                                   public ::testing::WithParamInterface<ShellContactTestParams> {
 protected:
  Actor* _actor = nullptr;

 public:
  static real constexpr kScale = 1_r;
  static int constexpr kM = MOCHI_DEBUG ? 8 : 16;
  static int constexpr kN = MOCHI_DEBUG ? 8 : 16;
  static real constexpr kGravityMagnitude = 10_r;
  static real constexpr kBcEps = 1e-3_r;
  static real constexpr kContactRadius = kScale / (real)Min(kM, kN);
  static real constexpr kContactPenalty = 1e7_r;
  static real constexpr kSelfContactExclusionRatio = 1.5_r;
  // Self-contact excludes a collider point from a colliding point when their rest distance is below
  // the exclusion range (radius * exclusionRatio + penaltyThreshold). The two regions must
  // therefore start farther apart in the rest configuration than this range, or self-contact
  // between them is suppressed entirely. We exceed it only by a small margin: the regions share the
  // same x/y footprint, so the minimum inter-region rest distance equals this vertical gap exactly,
  // and a small margin keeps the upper region's fall (hence impact velocity) close to the proven
  // two-actor scenario, avoiding penalty-contact tunneling that a much larger gap would cause.
  static real constexpr kSeparationDistance =
      (kSelfContactExclusionRatio + 0.05_r) * kContactRadius;
  // Nodes [0, kNumNodesPerRegion) form the lower region; the rest form the translated upper one.
  static int constexpr kNumNodesPerRegion = (kM + 1) * (kN + 1);

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();

    ShellContactTestParams const& params = GetParam();

    SolverParams solverParams = _scene->GetSolverParams();
    solverParams.experimentalEval.explicitNormals = params.explicitNormals;
    _scene->SetSolverParams(solverParams, ErrorAssert{});

    auto& reg = GetRegistry();

    // Gravity points along -z; the upper region falls onto the lower region. A single actor has
    // worldFromCollider == worldFromColliding, so the self-contact relative transform is always
    // identity regardless of actor rotation; an identity transform keeps the test minimal.
    _scene->SetGravity(Real3{0_r, 0_r, -kGravityMagnitude});

    // Build one disconnected mesh: a lower square in the z = 0 plane plus an upper copy translated
    // up by kSeparationDistance. Shell simulation does not require the mesh to be connected.
    auto&& [baseCoordinates, baseConnectivity] =
        UniformSquareTriangularMeshData(Int2{kM, kN}, Real2{kScale, kScale});
    auto&& [coordinates, connectivity] = UnionWithTranslatedCopy(
        MakeConstSpan(baseCoordinates),
        MakeConstSpan(baseConnectivity),
        Real3{0_r, 0_r, kSeparationDistance});

    ShellActorParams actorParams;
    actorParams.shape = _scene->GetContext()->CreateTriMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), ErrorAssert{});
    actorParams.pointCloudCollider.radius = kContactRadius;
    actorParams.pointCloudCollider.selfContact = true;
    actorParams.pointCloudCollider.selfContactExclusionRatio = kSelfContactExclusionRatio;
    actorParams.pointCloudCollider.colliderTriangleElementType = params.colliderTriangleElementType;
    actorParams.contact.penaltyCoefficient = kContactPenalty;

    _actor = CreateShellActor(_scene, actorParams, ErrorAssert{});
    EXPECT_NE(entt::entity{}, mochi::GetEntity(reg, _actor->GetHandle(), test::ExpectOK{}));

    // Pin only the lower region's boundary; the upper region is left entirely free to fall. With an
    // identity transform, the callback's world-space position equals the reference position.
    ConstrainNodesByPosition(
        _actor,
        [](int, Real3 const& x) -> bool {
          bool const isLowerRegion = x[2] < 0.5_r * kSeparationDistance;
          bool const onBoundary =
              (Abs(x[0]) > 0.5_r * kScale - kBcEps) || (Abs(x[1]) > 0.5_r * kScale - kBcEps);
          return isLowerRegion && onBoundary;
        },
        ErrorAssert{});
  }

  void TearDown() override {
    _scene->DestroyActor(_actor);
    test::MochiSceneTestBase::TearDown();
  }
};

// This tests a single shell actor whose upper region free-falls onto its pinned lower region,
// driving the self-contact code path. It runs basic sanity checks on the final configuration and
// verifies that the self-contact force is an internal, momentum-conserving pair (the per-region
// forces cancel, and the net contact force on the actor is zero).
TEST_P(MochiShellSelfContactScene, SingleShellActorWithSelfContact) {
  // TODO: Remove these custom solver settings after robustifying default convergence criteria.
  real constexpr kAbsTol = 0_r;
  real constexpr kRelTol = 1e-6_r;
  SolverParams solverParams = _scene->GetSolverParams();
  solverParams.nonLinearSolver.absTol = kAbsTol;
  solverParams.nonLinearSolver.relTol = kRelTol;
  // Explosion control is disabled because settling into static equilibrium can trigger false
  // positives (large relative residual with small absolute residual).
  solverParams.nonLinearSolver.explosionControl = false;
  solverParams.linearSolver.absTol = kAbsTol;
  solverParams.linearSolver.relTol = kRelTol;
  _scene->SetSolverParams(solverParams, ErrorAssert{});

  real constexpr kTimeInterval = 10_r;
  int constexpr kNumSteps = 8 * Max(kM, kN);
  real constexpr kTimeStep = kTimeInterval / (real)kNumSteps;

  for (int i = 0; i < kNumSteps; ++i) {
    _scene->Step(kTimeStep);
  }

  int const numDofs = _actor->GetNumDofs();
  int const numNodes = numDofs / 3;
  DynamicArray<real> dofValues(numDofs);
  _actor->GetDofValues({}, dofValues, ErrorAssert{});

  // Split nodes into the lower region [0, kNumNodesPerRegion) and the upper region [R, numNodes).
  // DOF values are displacements from the reference configuration. The lower region's reference z
  // is 0 (mesh lies in the z = 0 plane), so its DOF z equals its world z. The upper region's
  // reference z is kSeparationDistance, so its world z = DOF z + kSeparationDistance.
  real minZLower = std::numeric_limits<real>::infinity();
  for (int nodeIndex = 0; nodeIndex < kNumNodesPerRegion; ++nodeIndex) {
    minZLower = Min(dofValues[3 * nodeIndex + 2], minZLower);
  }

  int const numNodesUpper = numNodes - kNumNodesPerRegion;
  real minZUpper = std::numeric_limits<real>::infinity();
  real avgXUpper = 0_r;
  real avgYUpper = 0_r;
  for (int nodeIndex = kNumNodesPerRegion; nodeIndex < numNodes; ++nodeIndex) {
    real const worldZ = dofValues[3 * nodeIndex + 2] + kSeparationDistance;
    minZUpper = Min(worldZ, minZUpper);
    avgXUpper += dofValues[3 * nodeIndex + 0];
    avgYUpper += dofValues[3 * nodeIndex + 1];
  }
  avgXUpper /= (real)numNodesUpper;
  avgYUpper /= (real)numNodesUpper;

  // No interpenetration: the upper region stays above the lower region.
  EXPECT_GT(minZUpper, minZLower) << "Upper region penetrated lower region: minZUpper = "
                                  << minZUpper << ", minZLower = " << minZLower;

  // This is not based on any tight calculation, but we generally don't expect the two regions to
  // get significantly closer than the contact radius for a reasonable penalty value.
  real constexpr kMinSeparation = kContactRadius * 0.5_r;
  EXPECT_GT(minZUpper - minZLower, kMinSeparation)
      << "Self-contact separation too small: " << (minZUpper - minZLower);

  // No lateral drift: the regions share the same footprint, so the upper region lands squarely.
  real constexpr kMaxDrift = kContactRadius * 0.1_r;
  EXPECT_LT(Abs(avgXUpper), kMaxDrift) << "Upper region drifted too far in x: " << avgXUpper;
  EXPECT_LT(Abs(avgYUpper), kMaxDrift) << "Upper region drifted too far in y: " << avgYUpper;

  // --- Self-contact force verification ---

  auto& reg = GetRegistry();
  entt::entity const actorEntity = mochi::GetEntity(reg, _actor->GetHandle(), test::ExpectOK{});

  // A single self-contacting actor forms its own island with a populated contact SNLE.
  std::array<entt::entity, 1> const islandActors{actorEntity};
  entt::entity const islandEntity = FindIslandContaining(reg, MakeConstSpan(islandActors));
  ASSERT_TRUE(islandEntity != entt::null) << "No island contains the self-contacting actor";

  TransformRT const& actorTransform = reg.get<CRootTransform>(actorEntity).worldFromLocal;
  int const actorDofOffset = reg.get<CDofOffset>(actorEntity).dofsOffset;

  // Run the collision detection + contact assembly pipeline for the island.
  CIslandContactSnle const& contactSnle =
      AssembleIslandContactResiduals(reg, islandEntity, GradTarget::Current);

  // Split the single actor's DOFs into upper/lower regions by node index (3 spatial DOFs per node).
  Real3 const lowerForceLocal =
      SumLocalContactForce(contactSnle, actorDofOffset, numDofs, kSpaceDim3, [](int node) {
        return node < kNumNodesPerRegion;
      });
  Real3 const upperForceLocal =
      SumLocalContactForce(contactSnle, actorDofOffset, numDofs, kSpaceDim3, [](int node) {
        return node >= kNumNodesPerRegion;
      });

  // Forces are converted to world via the actor's root rotation (identity here, kept for parity
  // with the two-actor test).
  Real3 const lowerForceWorld = actorTransform.GetRotation() * lowerForceLocal;
  Real3 const upperForceWorld = actorTransform.GetRotation() * upperForceLocal;

  // Verify that the upper region's contact force is nonzero (it balances the upper region's
  // weight), so the test is not trivial and the self-contact path is genuinely exercised.
  real const forceScale = Max(Abs(upperForceWorld));
  EXPECT_GT(forceScale, 1e-3_r) << "Self-contact forces are too small";

  // The upper- and lower-region forces are an internal action/reaction pair: they cancel.
  real constexpr kForceTol = 3e-5_r;
  real const scaledTol = kForceTol * forceScale;
  EXPECT_NEAR_TOL(upperForceWorld + lowerForceWorld, Real3{}, scaledTol);

  // --- Query API cross-validation ---

  // The net contact force on the single self-contacting actor is zero. The query uses the
  // previous-step configuration, so an additional step is needed to populate the query results.
  _actor->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  _scene->Step(kTimeStep);
  Real3 const actorForceQueried = _actor->GetContactForceWorld(test::ExpectOK{});
  EXPECT_NEAR_TOL(actorForceQueried, Real3{}, scaledTol);
}

INSTANTIATE_TEST_SUITE_P(
    ColliderTypes,
    MochiShellSelfContactScene,
    ::testing::Values(
        ShellContactTestParams{
            Quaternion::Identity(),
            2,
            false,
            std::optional{ActorBoundaryElementType::P1Q1}},
        // Include one case with explicit normals, to ensure that the code path is exercised.
        ShellContactTestParams{
            Quaternion::Identity(),
            2,
            true,
            std::optional{ActorBoundaryElementType::P1Q1}},
        ShellContactTestParams{Quaternion::Identity(), 2, false, std::nullopt}));

// ---------------------------------------------------------------------------
// Rod/Rod contact integration test
// ---------------------------------------------------------------------------

struct RodContactTestParams {
  std::optional<ActorSegmentElementType> colliderSegmentElementType;
};

// This sets up two fixed bottom rods (parallel to the x-axis, separated in y) and one free-falling
// top rod (parallel to the y-axis, hovering above) for integration testing of rod/rod contact.
// Based on the RodDrapedOnRod sample geometry.
class MochiRodActorContactScene : public test::MochiSceneTestBase,
                                  public ::testing::WithParamInterface<RodContactTestParams> {
 protected:
  Actor* _bottomActor1 = nullptr;
  Actor* _bottomActor2 = nullptr;
  Actor* _topActor = nullptr;

 public:
  static real constexpr kLength = 1_r;
  static int constexpr kNumElements = MOCHI_DEBUG ? 16 : 32;
  static int constexpr kNumNodes = kNumElements + 1;
  static real constexpr kContactRadius = kLength / (real)kNumElements;
  static real constexpr kContactPenalty = 1e8_r;
  static real constexpr kSeparationDistance = kContactRadius;
  static real constexpr kGravityMagnitude = 10_r;
  static real constexpr kRadius = 1e-2_r;
  static real constexpr kDensity = 1e3_r;
  static real constexpr kYoungsModulus = 1e7_r;
  static real constexpr kShearModulus = 1e7_r;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();

    _scene->SetGravity(Real3{0_r, 0_r, -kGravityMagnitude});

    RodContactTestParams const& params = GetParam();

    // Material from cross-section geometry (matching the rod samples pattern).
    real const area = kPI * Sqr(kRadius);
    real const I = 0.25_r * kPI * Pow(kRadius, 4);
    real const J = 2_r * I;

    // Template params shared by all rods.
    RodActorParams rodTemplate;
    rodTemplate.material.linearDensity = kDensity * area;
    rodTemplate.material.linearRotationalInertia = kDensity * J;
    rodTemplate.material.axialStiffness = kYoungsModulus * area;
    rodTemplate.material.flexuralStiffness = kYoungsModulus * Real2{I, I};
    rodTemplate.material.torsionalStiffness = kShearModulus * J;
    rodTemplate.colliderType = ColliderType::PointCloud;
    rodTemplate.pointCloudCollider.radius = kContactRadius;
    rodTemplate.pointCloudCollider.selfContact = false;
    rodTemplate.pointCloudCollider.colliderSegmentElementType = params.colliderSegmentElementType;
    rodTemplate.contact.penaltyCoefficient = kContactPenalty;

    auto createRod = [&](Real3 const& start, Real3 const& end, Real3 const& frameAxis) -> Actor* {
      DynamicArray<Real3> nodes(kNumNodes);
      DynamicArray<Real3> axes(kNumElements, frameAxis);
      for (int i = 0; i < kNumNodes; ++i) {
        real const t = (real)i / (real)(kNumNodes - 1);
        nodes[i] = Lerp(start, end, t);
      }
      RodActorParams p = rodTemplate;
      p.shape = CreatePolylineShape(
          _scene->GetContext(), nodes, MakeConstSpan(axes), false, ErrorAssert{});
      return CreateRodActor(_scene, p, ErrorAssert{});
    };

    auto pinEndpoints = [&](Actor* actor, Real3 const& startPos, Real3 const& endPos) {
      DeformableNodePositionConstraintParams bc0;
      bc0.actor = actor->GetHandle();
      bc0.nodeIndex = 0;
      bc0.position = startPos;
      _scene->CreateDeformableNodePositionConstraint(bc0, ErrorAssert{});

      DeformableNodePositionConstraintParams bc1;
      bc1.actor = actor->GetHandle();
      bc1.nodeIndex = kNumNodes - 1;
      bc1.position = endPos;
      _scene->CreateDeformableNodePositionConstraint(bc1, ErrorAssert{});
    };

    // Bottom rod 1: along x-axis at y = -0.25
    Real3 const b1Start{-0.5_r * kLength, -0.25_r, 0_r};
    Real3 const b1End{0.5_r * kLength, -0.25_r, 0_r};
    _bottomActor1 = createRod(b1Start, b1End, Real3{0_r, 1_r, 0_r});
    pinEndpoints(_bottomActor1, b1Start, b1End);

    // Bottom rod 2: along x-axis at y = +0.25
    Real3 const b2Start{-0.5_r * kLength, 0.25_r, 0_r};
    Real3 const b2End{0.5_r * kLength, 0.25_r, 0_r};
    _bottomActor2 = createRod(b2Start, b2End, Real3{0_r, 1_r, 0_r});
    pinEndpoints(_bottomActor2, b2Start, b2End);

    // Top rod: along y-axis at z = kSeparationDistance
    _topActor = createRod(
        Real3{0_r, -0.5_r * kLength, kSeparationDistance},
        Real3{0_r, 0.5_r * kLength, kSeparationDistance},
        Real3{0_r, 0_r, 1_r});
  }

  void TearDown() override {
    _scene->DestroyActor(_bottomActor1);
    _scene->DestroyActor(_bottomActor2);
    _scene->DestroyActor(_topActor);
    test::MochiSceneTestBase::TearDown();
  }
};

TEST_P(MochiRodActorContactScene, ThreeRodActorsWithContact) {
  real constexpr kTimeInterval = 10_r;
  int constexpr kNumSteps = 8 * kNumNodes;
  real constexpr kTimeStep = kTimeInterval / (real)kNumSteps;

  for (int i = 0; i < kNumSteps; ++i) {
    _scene->Step(kTimeStep);
  }

  // Rod DoFs are laid out as [x0, y0, z0, twist0, x1, y1, z1, twist1, ...] with 4 DoFs per node.
  int constexpr kDofsPerNode = 4;

  int const numDofsBottom1 = _bottomActor1->GetNumDofs();
  DynamicArray<real> dofValuesBottom1(numDofsBottom1);
  _bottomActor1->GetDofValues({}, dofValuesBottom1, ErrorAssert{});

  int const numDofsBottom2 = _bottomActor2->GetNumDofs();
  DynamicArray<real> dofValuesBottom2(numDofsBottom2);
  _bottomActor2->GetDofValues({}, dofValuesBottom2, ErrorAssert{});

  int const numDofsTop = _topActor->GetNumDofs();
  int const numNodesTop = numDofsTop / kDofsPerNode;
  DynamicArray<real> dofValuesTop(numDofsTop);
  _topActor->GetDofValues({}, dofValuesTop, ErrorAssert{});

  // The top rod is along y. Its free ends
  // under gravity — this is expected. We check z-values near the two crossing points
  // (y ≈ ±0.25), where contact with the bottom rods should arrest the rod's fall. Penalty-based
  // contact permits some interpenetration; we only verify that the rod has not fallen freely.
  real minZTopNearContact = std::numeric_limits<real>::infinity();
  real avgXTop = 0_r;
  for (int i = 0; i < numNodesTop; ++i) {
    real const refY = -0.5_r * kLength + (real)i / (real)(numNodesTop - 1) * kLength;
    real const zDisp = dofValuesTop[kDofsPerNode * i + 2];
    avgXTop += dofValuesTop[kDofsPerNode * i + 0];
    if (Abs(Abs(refY) - 0.25_r) < 2_r * kContactRadius) {
      minZTopNearContact = Min(zDisp + kSeparationDistance, minZTopNearContact);
    }
  }
  avgXTop /= (real)numNodesTop;

  // Near the contact points, the top rod should be close to the bottom rods (not fallen freely).
  // Free-fall over the simulation time would give z ≈ -0.5 * g * T^2 = -500.
  EXPECT_GT(minZTopNearContact, -kLength)
      << "Top rod fell freely through bottom rods: z = " << minZTopNearContact;

  // Lateral drift should be small.
  real constexpr kMaxDrift = kContactRadius;
  EXPECT_LT(Abs(avgXTop), kMaxDrift) << "Top rod drifted too far in x: " << avgXTop;

  // --- Contact force verification ---

  auto& reg = GetRegistry();
  entt::entity bottom1Entity = mochi::GetEntity(reg, _bottomActor1->GetHandle(), test::ExpectOK{});
  entt::entity bottom2Entity = mochi::GetEntity(reg, _bottomActor2->GetHandle(), test::ExpectOK{});
  entt::entity topEntity = mochi::GetEntity(reg, _topActor->GetHandle(), test::ExpectOK{});

  std::array<entt::entity, 3> const islandActors{bottom1Entity, bottom2Entity, topEntity};
  entt::entity const islandEntity = FindIslandContaining(reg, MakeConstSpan(islandActors));
  ASSERT_TRUE(islandEntity != entt::null) << "No island contains all three rod actors";

  CIslandContactSnle const& contactSnle =
      AssembleIslandContactResiduals(reg, islandEntity, GradTarget::Current);

  TransformRT const& bottom1Transform = reg.get<CRootTransform>(bottom1Entity).worldFromLocal;
  TransformRT const& bottom2Transform = reg.get<CRootTransform>(bottom2Entity).worldFromLocal;
  TransformRT const& topTransform = reg.get<CRootTransform>(topEntity).worldFromLocal;

  int const bottom1DofOffset = reg.get<CDofOffset>(bottom1Entity).dofsOffset;
  int const bottom2DofOffset = reg.get<CDofOffset>(bottom2Entity).dofsOffset;
  int const topDofOffset = reg.get<CDofOffset>(topEntity).dofsOffset;

  auto const includeAllNodes = [](int) -> bool { return true; };
  Real3 const bottom1ForceLocal = SumLocalContactForce(
      contactSnle, bottom1DofOffset, numDofsBottom1, kDofsPerNode, includeAllNodes);
  Real3 const bottom2ForceLocal = SumLocalContactForce(
      contactSnle, bottom2DofOffset, numDofsBottom2, kDofsPerNode, includeAllNodes);
  Real3 const topForceLocal =
      SumLocalContactForce(contactSnle, topDofOffset, numDofsTop, kDofsPerNode, includeAllNodes);

  // Transform forces from local to world frame
  Real3 const bottom1ForceWorld = bottom1Transform.GetRotation() * bottom1ForceLocal;
  Real3 const bottom2ForceWorld = bottom2Transform.GetRotation() * bottom2ForceLocal;
  Real3 const topForceWorld = topTransform.GetRotation() * topForceLocal;

  // Verify contact forces are nonzero
  real const forceScale = Max(Max(Abs(bottom1ForceWorld), Abs(bottom2ForceWorld)));
  EXPECT_GT(forceScale, 1e-3_r) << "Contact forces are too small";

  // Verify net force balance (Newton's 3rd law)
  real constexpr kForceTol = 3e-5_r;
  real const scaledTol = kForceTol * forceScale;
  EXPECT_NEAR_TOL(bottom1ForceWorld + bottom2ForceWorld + topForceWorld, Real3{}, scaledTol);

  // --- Query API cross-validation ---

  _bottomActor1->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  _bottomActor2->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  _topActor->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  _scene->Step(kTimeStep);
  Real3 const bottom1ForceQueried = _bottomActor1->GetContactForceWorld(test::ExpectOK{});
  Real3 const bottom2ForceQueried = _bottomActor2->GetContactForceWorld(test::ExpectOK{});
  Real3 const topForceQueried = _topActor->GetContactForceWorld(test::ExpectOK{});

  // Check that the queried forces match the manually-computed ones (from the previous step, which
  // is what the query uses). The comparison tolerance is larger because the configuration may have
  // changed slightly between the two steps.
  // Rod-rod contact occurs at discrete crossing points, so forces are more sensitive to small
  // configuration changes between steps than shell-shell contact spread over a continuous surface.
  real constexpr kQueryForceTol = 1e-2_r;
  real const queryScaledTol = kQueryForceTol * forceScale;
  EXPECT_NEAR_TOL(bottom1ForceQueried, bottom1ForceWorld, queryScaledTol);
  EXPECT_NEAR_TOL(bottom2ForceQueried, bottom2ForceWorld, queryScaledTol);
  EXPECT_NEAR_TOL(topForceQueried, topForceWorld, queryScaledTol);
}

INSTANTIATE_TEST_SUITE_P(
    ColliderTypes,
    MochiRodActorContactScene,
    ::testing::Values(
        RodContactTestParams{std::optional{ActorSegmentElementType::Default}},
        RodContactTestParams{std::nullopt}));
