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

#include <mochi_core/contact/contact_utils.h>
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/solvers/snle_problem.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/span.h>
#include <mochi_physics/mochi_physics.h>

// mochi_physics/test is allowed to include private src headers
#include <mochi_physics/src/mochi_context.h>
#include <mochi_physics/src/mochi_ecs_utils.h>
#include <mochi_physics/src/mochi_scene.h>
#include <mochi_physics/src/mochi_shape.h>

#include <gtest/gtest.h>

#include <memory>
#include <optional>

namespace mochi::test {

// Get access to the registry owned by a Scene
inline entt::registry& GetRegistry(Scene* scene) {
  return assert_cast<SceneImpl*>(scene)->GetRegistry();
}
inline entt::registry const& GetRegistry(Scene const* scene) {
  return assert_cast<SceneImpl const*>(scene)->GetRegistry();
}

// Create a unit-cube tetrahedral mesh shape, scaled by `scale`, on the given context (no skinning).
inline ShapeHandle CreateUnitCubeTetMeshShape(
    Context* context,
    Real3 const& scale = Real3{1_r, 1_r, 1_r}) {
  auto cube = CreateMinimalTetMeshUnitCube(scale);
  return context->CreateTetMeshShape(
      Flatten(MakeSpan(cube.first)), Flatten(MakeSpan(cube.second)), ExpectOK{});
}

// Set the node velocities of a soft actor to those of a rigid motion with the given velocity at the
// local-space point `center`. Unlike Actor::SetVelocity, supports nonzero angular velocity.
inline void SetRigidNodeVelocitiesLocal(
    Actor* actor,
    Real3 const& linearVelocity,
    Real3 const& angularVelocity,
    Real3 const& center) {
  DynamicArray<Real3> velocities;
  for (Real3 const& node : Unflatten<Real3 const>(actor->GetMesh().coordinates)) {
    velocities.push_back(linearVelocity + Cross(angularVelocity, node - center));
  }
  actor->SetNodeVelocitiesLocal(Flatten(MakeConstSpan(velocities)), ExpectOK{});
}

// Build a minimal SkinningData with 1 weight per node (a "pinned to one bone" weight scheme),
// where every node is bound to bone `boneIndex` with weight 1.
inline SkinningData MakeSingleBoneSkinning(int numNodes, int boneIndex) {
  SkinningData data;
  data.weightsPerNode = 1;
  data.weights.resize(numNodes, 1_r);
  data.indices.resize(numNodes, boneIndex);
  return data;
}

// Build a soft-body TetrahedralMeshShape with single-bone skinning and one constrained node.
// Skinning data is required for the soft-skinned mesh code path; constrained nodes are required
// so the soft-skinned engine can apply Dirichlet boundary conditions on the soft body.
inline ShapeHandle CreateUnitCubeTetSoftShape(Context* context) {
  auto&& [coords, connectivity] = CreateMinimalTetMeshUnitCube();
  auto mesh = std::make_shared<TetrahedralMesh const>(coords, connectivity);
  auto skinning = std::make_shared<SkinningData const>(
      MakeSingleBoneSkinning(mesh->GetNumNodes(), /*boneIndex=*/0));
  auto constrained = std::make_shared<ConstrainedNodesData const>(DynamicArray<int>{0});
  auto shape = std::make_shared<TetrahedralMeshShape>(mesh, skinning, constrained);
  return assert_cast<ContextImpl*>(context)->RegisterShape(shape, ExpectOK{});
}

// Build a unit-cube tetrahedral model with a unit-cube triangular contact skin whose node i is
// embedded at tet node i with weight 1.
inline ModelData CreateUnitCubeContactSkinModel() {
  auto [tetCoordinates, tetConnectivity] = CreateMinimalTetMeshUnitCube();
  auto [skinCoordinates, skinConnectivity] = CreateMinimalTriMeshUnitCube();

  ModelData model;
  model.mesh.emplace();
  model.mesh->nodesPerElement = 4;
  model.mesh->coordinates = Flatten(MakeSpan(tetCoordinates));
  model.mesh->connectivity = Flatten(MakeSpan(tetConnectivity));
  model.contactSkinMesh.emplace();
  model.contactSkinMesh->nodesPerElement = 3;
  model.contactSkinMesh->coordinates = Flatten(MakeSpan(skinCoordinates));
  model.contactSkinMesh->connectivity = Flatten(MakeSpan(skinConnectivity));
  int const numSkinNodes = model.contactSkinMesh->GetNumNodes();
  auto& skinning = model.contactSkinMesh->skinning.emplace();
  skinning.weightsPerNode = 1;
  skinning.indices.resize_noinit(numSkinNodes);
  skinning.weights.resize(numSkinNodes, 1_r);
  for (int i = 0; i < numSkinNodes; ++i) {
    skinning.indices[i] = i;
  }
  return model;
}

// Pin a scene's integrator, making explicit any test that depends on a specific integrator
// (e.g. backward Euler's numerical damping) rather than on the default.
inline void SetSceneIntegrationMethod(Scene* scene, IntegrationMethod method) {
  auto solverParams = scene->GetSolverParams();
  solverParams.integrationMethod = method;
  scene->SetSolverParams(solverParams, ExpectOK{});
}

// Maps local positions relative to the actor's current, possibly recentered, root transform to
// local positions relative to `worldFromLocalInitial`.
inline TransformRT GetInitialFromCurrentLocal(
    Actor const& actor,
    TransformRT const& worldFromLocalInitial) {
  return Invert(worldFromLocalInitial) * actor.GetRootTransform();
}

// DoF values of a shell (3 per node) or rod (4 per node: xyz then twist) actor, with the xyz
// displacements expressed relative to `worldFromLocalInitial` rather than the actor's current,
// possibly recentered, root transform. Other DoFs (rod twist) are returned unchanged.
inline DynamicArray<real> GetDofValuesRelativeToInitialFrame(
    Actor const& actor,
    TransformRT const& worldFromLocalInitial,
    int numFieldsPerNode) {
  DynamicArray<real> dofs(actor.GetNumDofs());
  actor.GetDofValues({}, dofs, ExpectOK{});
  auto const reference = Unflatten<Real3 const>(actor.GetMesh().coordinates);
  TransformRT const initialFromCurrent = GetInitialFromCurrentLocal(actor, worldFromLocalInitial);
  for (int n = 0; n < isize(reference); ++n) {
    real* u = &dofs[n * numFieldsPerNode];
    Real3 const displacement =
        initialFromCurrent.TransformPoint(reference[n] + Real3{u[0], u[1], u[2]}) - reference[n];
    for (int j = 0; j < 3; ++j) {
      u[j] = displacement[j];
    }
  }
  return dofs;
}

// World-space node positions of a shell (3 DoFs per node) or rod (4 DoFs per node) actor, computed
// from its reference mesh, DoF values, and current root transform.
inline DynamicArray<Real3> GetWorldNodePositions(Actor const& actor, int numFieldsPerNode) {
  DynamicArray<real> dofs(actor.GetNumDofs());
  actor.GetDofValues({}, dofs, ExpectOK{});
  auto const reference = Unflatten<Real3 const>(actor.GetMesh().coordinates);
  TransformRT const worldFromLocal = actor.GetRootTransform();
  DynamicArray<Real3> positions;
  positions.reserve(reference.size());
  for (int i = 0; i < isize(reference); ++i) {
    real const* u = &dofs[i * numFieldsPerNode];
    positions.push_back(worldFromLocal.TransformPoint(reference[i] + Real3{u[0], u[1], u[2]}));
  }
  return positions;
}

// Expect that a shell or rod actor recentered with `recenteringParams` moves like an identical
// actor without recentering when both translate and spin under gravity with a multistep integrator,
// and that the recentered actor's root transform moved by at least `minRecenteringDistance`.
// `createActor()` creates the actor in `scene`.
template <typename CreateActorFn>
void ExpectRecenteringPreservesWorldMotion(
    Scene* scene,
    CreateActorFn const& createActor,
    RecenteringParams const& recenteringParams,
    int numFieldsPerNode,
    real minRecenteringDistance) {
  SetSceneIntegrationMethod(scene, IntegrationMethod::BDF3);
  scene->SetGravity({1_r, -2_r, 0.5_r});
  Actor* const recentered = createActor();
  recentered->SetRecenteringParams(recenteringParams, ExpectOK{});
  Actor* const reference = createActor();
  reference->SetRecenteringParams({.useRecentering = false}, ExpectOK{});
  TransformRT const worldFromLocalInitial = recentered->GetRootTransform();
  for (int i = 0; i < 40; ++i) {
    scene->Step(0.02_r);
  }

  real constexpr kTolerance = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-3_r;
  DynamicArray<Real3> const expectedNodes = GetWorldNodePositions(*reference, numFieldsPerNode);
  DynamicArray<Real3> const actualNodes = GetWorldNodePositions(*recentered, numFieldsPerNode);
  ASSERT_EQ(isize(expectedNodes), isize(actualNodes));
  for (int i = 0; i < isize(expectedNodes); ++i) {
    EXPECT_NEAR_TOL(expectedNodes[i], actualNodes[i], kTolerance) << "Node " << i;
  }
  EXPECT_EQ(reference->GetRootTransform(), worldFromLocalInitial);
  EXPECT_GT(
      Norm(
          recentered->GetRootTransform().GetTranslation() - worldFromLocalInitial.GetTranslation()),
      minRecenteringDistance);
}

// This generic test fixture simply creates a mochi::Context instance.
class MochiContextTestBase : public ::testing::Test {
 public:
  void SetUp() override; // Called just before each test case
  void TearDown() override; // Called just after each test case

 protected:
  ~MochiContextTestBase() override;

  static constexpr real kTolerance = 1e-5_r;
  Context* _mochiContext = nullptr;

  int _numWorkerThreads = 0; // Must be set before SetUp
};

struct MochiContextTestParams {
  int numWorkerThreads = 0; // Number of worker threads to create
  bool isSingleThreadedMode = false; // Can be true even if worker threads are available
};

// This test fixture is designed to be repeated multiple times with various combinations of
// the MochiContextTestParams.
class MochiContextTestWithParam : public MochiContextTestBase,
                                  public ::testing::WithParamInterface<MochiContextTestParams> {
 public:
  using Class = MochiContextTestBase;
  void SetUp() override;
};

// This generic test fixture creates an empty mochi::Scene (synchronous API).
class MochiSceneTestBase : public MochiContextTestBase {
 public:
  void SetUp() override; // Called just before each test case
  void TearDown() override; // Called just after each test case

  entt::registry& GetRegistry() {
    return test::GetRegistry(_scene);
  }

  entt::registry const& GetRegistry() const {
    return test::GetRegistry(_scene);
  }

  entt::entity GetEntity(ActorHandle actor) const {
    return mochi::GetEntity(GetRegistry(), actor, ExpectOK{});
  }

  entt::entity GetEntity(ConstraintHandle constraint) const {
    return mochi::GetEntity(GetRegistry(), constraint, ExpectOK{});
  }

  entt::entity GetEntity(Actor* actor) const {
    return GetEntity(actor->GetHandle());
  }

  // Return the number of entities with a given component or tag
  template <class Component>
  int CountEntitiesWith() const {
    int count = 0;
    GetRegistry().view<Component const>().each([&](auto const& /*component*/) { ++count; });
    return count;
  }

 protected:
  mochi::Scene* _scene = nullptr;
};

// This generic test fixture creates an empty mochi::AsyncScene.
class MochiAsyncSceneTestBase : public MochiContextTestBase {
 public:
  MochiAsyncSceneTestBase();
  void SetUp() override; // Called just before each test case
  void TearDown() override; // Called just after each test case

 protected:
  mochi::AsyncScene* _asyncScene = nullptr;
};

// Normally logging to the Error or Warning channel will automatically fail the test.
// If you want to explicitly test logging, you can create a temporary variable of this type. It
// expects at least one message on `channel`; messages on other channels behave as usual.
struct ExpectLoggingInScope {
  ExpectLoggingInScope(Context* mochiContext, LogChannel channel);
  ~ExpectLoggingInScope();
  Context* _mochiContext = nullptr;
  LogChannel _channel = LogChannel::Warning;
  LogFn _prevCallbackForMochiPhysics;
  int _messagesReceived = 0;
};

// Apparently buck uses a different version of gtest on macos vs windows. The macos version appears
// to be newer and complains the INSTATIATE_TEST_CASE_P has been deprecated in place
// INSTATIATE_TEST_SUITE_P, which does not exist in the current Windows version.
#define MOCHI_INSTANTIATE_TEST_SUITE_P(namePrefix, fixtureClass, values)              \
  MOCHI_WARNING_PUSH();                                                               \
  MOCHI_WARNING_IGNORE_GCC_CLANG(GCC diagnostic ignored "-Wdeprecated-declarations"); \
  INSTANTIATE_TEST_CASE_P(namePrefix, fixtureClass, values);                          \
  MOCHI_WARNING_POP();

// Expect two contact-skin column coalescing maps to be identical.
inline void ExpectEqualCoalescingMaps(
    ContactJacColumnCoalescingMap const& expected,
    ContactJacColumnCoalescingMap const& actual) {
  EXPECT_EQ(expected.sourceBlockCount, actual.sourceBlockCount);
  EXPECT_EQ(expected.samplesPerElement, actual.samplesPerElement);
  EXPECT_SPAN_EQ(
      MakeConstSpan(expected.elementDofOffsets), MakeConstSpan(actual.elementDofOffsets));
  EXPECT_SPAN_EQ(
      MakeConstSpan(expected.elementDofIndices), MakeConstSpan(actual.elementDofIndices));
  EXPECT_SPAN_EQ(
      MakeConstSpan(expected.elementSourceContributionOffsets),
      MakeConstSpan(actual.elementSourceContributionOffsets));
  EXPECT_SPAN_EQ(
      MakeConstSpan(expected.elementSourceContributionDstColumns),
      MakeConstSpan(actual.elementSourceContributionDstColumns));
}

// Flattened copy of a ContactJac's indices and values, for comparing results across setups that
// reuse the same ContactJac storage.
struct ContactJacSnapshot {
  DynamicArray<int> indices;
  DynamicArray<real> values;
};

inline ContactJacSnapshot SnapshotContactJac(ContactJac const& jac) {
  ContactJacSnapshot snapshot;
  for (int contact = 0; contact < jac.nContacts; ++contact) {
    snapshot.indices.append(jac.Inds(contact));
    auto const block = jac.Jac(contact);
    for (int column = 0; column < jac.nDoFsInternal; ++column) {
      for (int component = 0; component < ContactJac::kDofsPerNode; ++component) {
        snapshot.values.push_back(block(component, column));
      }
    }
  }
  return snapshot;
}

inline void ExpectEqualContactJacSnapshots(
    ContactJacSnapshot const& expected,
    ContactJacSnapshot const& actual) {
  EXPECT_SPAN_EQ(MakeConstSpan(expected.indices), MakeConstSpan(actual.indices));
  EXPECT_SPAN_EQ(MakeConstSpan(expected.values), MakeConstSpan(actual.values));
}

// Checks a soft or shell actor's body assembly at a uniform translation, where only inertia, mass
// damping and gravity act. Each body with inertia or gravity is checked: a nested soft actor's
// unposed and posed bodies separately.
void ExpectUniformTranslationAssembly(Scene* scene, Actor* actor, real relTol);

// The non-potential residuals of a problem (see SnleProblem::nonPotentialResiduals), summed into
// one vector over its DoFs.
ColumnVector<real> SumNonPotentialResiduals(SnleProblem<real> const& problem);

// Coefficients of friction and contact damping (see ContactParams).
struct DissipationCoefficients {
  real coulomb = 0_r;
  real viscous = 0_r;
  real damping = 0_r;
};

// Coulomb friction, viscous friction and contact damping, all acting.
inline constexpr DissipationCoefficients kDissipationCoefficients{
    .coulomb = 0.5_r,
    .viscous = 1_r,
    .damping = 1_r};

// Checks the energy N and residual g that friction and damping report as non-potential terms (see
// SnleProblem::nonPotentialEnergy) at the problem's state, under WolfeStrong, with every actor's
// contact coefficients set to coefficients, if given, then zeroed with the overrides disabled:
// - E and F change observably;
// - with implicit normals, E - N and F - g don't change;
// - with explicit normals, N and g don't change;
// - objective-only and combined assemblies, and residual requests that reuse the objective, agree;
// - ResidualNorm reports nothing, with the same E and F.
// Restores the line search, the actors' contact coefficients and the overrides.
void ExpectDissipationIsNonPotential(
    entt::registry& reg,
    SnleProblem<real>& problem,
    std::optional<DissipationCoefficients> coefficients);

// Steps a scene until some actor has contact at the end of a step, and once more, so that its
// contacts persist through a whole step. Then runs ExpectDissipationIsNonPotential with
// kDissipationCoefficients on every island, each of which must touch with friction or damping, at a
// random increment from the end of that step. With elasto-plastic friction, also expects some
// contact to have a nonzero anchor offset.
void ExpectDissipationIsNonPotentialAfterTouch(Scene* scene, real dt);

} // namespace mochi::test
