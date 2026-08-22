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

#include "mochi_physics_test_fixture.h"

#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/rigid_body_utils.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>

// Private src headers
#include <mochi_physics/src/mochi_articulated_body.h>
#include <mochi_physics/src/mochi_contact.h>
#include <mochi_physics/src/mochi_context.h>
#include <mochi_physics/src/mochi_group.h>
#include <mochi_physics/src/mochi_island.h>
#include <mochi_physics/src/mochi_rigid.h>
#include <mochi_physics/src/mochi_shape.h>
#include <mochi_physics/src/mochi_soft.h>
#include <mochi_physics/src/mochi_soft_rom_linear_systems.h>
#include <mochi_physics/src/mochi_soft_rom_systems.h>
#include <mochi_physics/src/mochi_solve.h>
#include <mochi_physics/src/mochi_step.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace mochi;
using namespace mochi::experimental;
using namespace mochi::test;

// Test collision response on multiple contacts by comparing analytical force and dforce to
// finite-difference approximations.
TEST(MochiContact, ComputeCollisionResponse) {
  auto runTest = [](bool explicitNormals, real coulombCoefficient, real viscousCoefficient) {
    // Contact eval config for derivative consistency tests
    ContactEvalConfig config{
        .explicitNormals = explicitNormals,
        .fadeFriction = explicitNormals,
        .useFittedHessian = false};

    ContactParams params;
    params.viscousFrictionCoefficient = viscousCoefficient;
    params.coulombFrictionCoefficient = coulombCoefficient;
    params.penaltyThresholdDefault = 0.005_r; // Large enough to cover the tested distances
    real dtStage = 0.01_r;

    // Pool of test data
    std::vector<real> distances = {0.003_r, -0.001_r, 0.002_r, -0.005_r};
    size_t nDistances = distances.size();
    std::vector<Real3> normals = {
        Real3{1_r, 2_r, 3_r}, Real3{-2_r, 3_r, -1_r}, Real3{3_r, -1_r, -2_r}};
    size_t nNormals = normals.size();
    for (size_t i = 0; i < nNormals; i++) {
      normals[i] = Normalize(normals[i]);
    }
    std::vector<Real3> positions = {Real3{4_r, 2_r, 3_r}, Real3{-2_r, 3_r, -4_r}};
    size_t nPositions = positions.size();
    std::vector<Real3> velocities = {
        Real3{-4_r, 2_r, -5_r}, Real3{-6_r, 3_r, 4_r}, Real3{5_r, 3_r, 4_r}};
    size_t nVelocities = velocities.size();

    // Fill the contact data
    ContactDetectionResult contactQuery;
    size_t nContacts = nDistances * nNormals * nPositions * nVelocities;
    contactQuery.sampleIndices.resize(nContacts);
    contactQuery.posColliding.resize(nContacts);
    contactQuery.posCollidingStageStart.resize(nContacts);
    contactQuery.sdfInfo.resize(nContacts);
    contactQuery.sdfInfoStageStart.resize(nContacts);
    contactQuery.normalColliding.resize(nContacts);
    size_t nContact = 0;
    for (auto const& distance : distances) {
      for (auto const& normal : normals) {
        for (auto const& position : positions) {
          for (auto const& velocity : velocities) {
            contactQuery.sampleIndices[nContact] = (int)nContact;
            contactQuery.sdfInfo.val[nContact] = distance;
            contactQuery.sdfInfo.grad[nContact] = normal;
            contactQuery.posColliding[nContact] = position;
            contactQuery.sdfInfoStageStart.val[nContact] =
                distance + Dot(normal, velocity * dtStage);
            contactQuery.sdfInfoStageStart.grad[nContact] = normal;
            contactQuery.posCollidingStageStart[nContact] = position + velocity * dtStage;
            contactQuery.normalColliding[nContact] = -normal;
            nContact++;
          }
        }
      }
    }

    CollisionResponseResult res;
    res.ResizeNoInit(isize(contactQuery.sampleIndices), true, true, true);
    mochi::ComputeCollisionResponse<GradTarget::Current>(
        contactQuery, params, config, dtStage, true, true, true, res);

    // Compute through finite differences
    CollisionResponseResult resTest;
    resTest.force.resize(nContacts, {});
    resTest.dforce.resize(nContacts, {});
    constexpr real kEps = 1e-4_r;
    for (int j = 0; j < 3; j++) {
      Real3 delta = kEps * BasisVector<real, 3>(j);
      for (size_t i = 0; i < nContacts; i++) {
        contactQuery.posColliding[i] = delta + contactQuery.posColliding[i];
        contactQuery.sdfInfo.val[i] += Dot(contactQuery.sdfInfo.grad[i], delta);
      }
      CollisionResponseResult resF;
      resF.ResizeNoInit(isize(contactQuery.sampleIndices), true, true, false);
      mochi::ComputeCollisionResponse<GradTarget::Current>(
          contactQuery, params, config, dtStage, true, true, false, resF);

      delta = -2_r * kEps * BasisVector<real, 3>(j);
      for (size_t i = 0; i < nContacts; i++) {
        contactQuery.posColliding[i] = delta + contactQuery.posColliding[i];
        contactQuery.sdfInfo.val[i] += Dot(contactQuery.sdfInfo.grad[i], delta);
      }
      CollisionResponseResult resB;
      resB.ResizeNoInit(isize(contactQuery.sampleIndices), true, true, false);
      mochi::ComputeCollisionResponse<GradTarget::Current>(
          contactQuery, params, config, dtStage, true, true, false, resB);

      delta = kEps * BasisVector<real, 3>(j);
      for (size_t i = 0; i < nContacts; i++) {
        contactQuery.posColliding[i] = delta + contactQuery.posColliding[i];
        contactQuery.sdfInfo.val[i] += Dot(contactQuery.sdfInfo.grad[i], delta);
      }

      for (size_t i = 0; i < nContacts; i++) {
        resTest.force[i][j] = static_cast<real>(resF.energy[i] - resB.energy[i]) / (-2_r * kEps);
        resTest.dforce[i][j] = ToSimd((resF.force[i] - resB.force[i]) / (2_r * kEps));
      }
    }

    // Test
    for (size_t i = 0; i < nContacts; i++) {
      EXPECT_NEAR(
          Norm(res.force[i] - resTest.force[i]) / Max(Norm(res.force[i]), Norm(resTest.force[i])),
          0_r,
          1e-2_r);
      EXPECT_NEAR(
          Norm3x3(res.dforce[i] - resTest.dforce[i]) /
              Max(Norm3x3(res.dforce[i]), Norm3x3(resTest.dforce[i])),
          0_r,
          1e-2_r);
    }
  };

  runTest(/* explicitNormals */ false, /*coulombCoefficient*/ 0_r, /*viscousCoefficient*/ 100.0_r);
  runTest(/* explicitNormals */ false, /*coulombCoefficient*/ 0.5_r, /*viscousCoefficient*/ 0_r);
  runTest(/* explicitNormals */ true, /*coulombCoefficient*/ 0_r, /*viscousCoefficient*/ 100.0_r);
  runTest(/* explicitNormals */ true, /*coulombCoefficient*/ 0.5_r, /*viscousCoefficient*/ 0_r);
}

namespace {
struct TestParams {
  std::string name = {};
  GradTarget gradTarget = GradTarget::Current;
  real coulombCoefficient = 0_r;
  real viscousCoefficient = 0_r;
  real dampingCoefficient = 0_r;
  bool explicitNormals = false;
  real resTol = 1e-3_r;
  real dresTol = 1e-3_r;
  bool ignoreColliderRotDres = false;
  real eps = 1e-4_r;
};
} // namespace

// Base class to implement contact tests between a colliding actor and a collider actor. The tests
// create a Mochi scene with both actors, compute objective, residual and dresidual at the initial
// state, and validate the residual and dresidual through finite differences.
class MochiContactTestBase : public test::MochiSceneTestBase,
                             public ::testing::WithParamInterface<TestParams> {
 protected:
  static constexpr real kDtStage = 1e-2_r;

  entt::entity _colliding = entt::null;
  entt::entity _collider = entt::null;

  int _numDofsA = 0;
  int _numDofsB = 0;
  int _numDofs = 0;

  double _objective = 0.0;
  ColumnVector<real> _residual;
  SparseMatrix<real> _dresidual;

 public:
  virtual void InitState() = 0;

  virtual void AddToState(int i, real eps) = 0;

  virtual void AddToOldState(int /*i*/, real /*eps*/) {
    MOCHI_ASSERT(false, "Not supported for this test configuration");
  }

  virtual void
  InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient) = 0;

  void SetUp() override { // Called just before each test case
    test::MochiSceneTestBase::SetUp();

    // Initialize result
    _objective = 0.0;
    _residual.Reset(_numDofs);
    _residual.SetZero();
    _dresidual.Reset(MakeDenseSparsityGraph(_numDofs, _numDofs));
  }

  void InitActors(Actor* collidingActor, Actor* colliderActor) {
    auto& reg = GetRegistry();

    // Store colliding & collider entities
    _colliding = GetEntity(collidingActor->GetHandle());
    _collider = GetEntity(colliderActor->GetHandle());
    EXPECT_EQ(_numDofsA, reg.get<CActorDofInfo>(_colliding).dofsSize);
    if (_numDofsB != 0) {
      EXPECT_EQ(_numDofsB, reg.get<CActorDofInfo>(_collider).dofsSize);
    }

    // Contact filtering
    _scene->EnableLayerContactSymmetric("Colliding", "Colliding", false, test::ExpectOK{});
    _scene->EnableLayerContactSymmetric("Collider", "Collider", false, test::ExpectOK{});
    _scene->EnableLayerContactSymmetric("Colliding", "Collider", true, test::ExpectOK{});

    // This test does not step the full simulation, but calling PreStep ensures that certain things
    // are initialized, including islands and DOF offsets.
    PreStep();
  }

  void SetTransformFromState(entt::entity entity, TransformRT const& rigidState) {
    auto& reg = GetRegistry();
    auto const& rigidInertia = reg.get<CRigidBodyInertia const>(entity);
    auto& rootTransform = reg.get<CRootTransform>(entity);
    mochi::rigid::RigidStateToRootTransform(
        rigidInertia.GetCenterOfMassLocal(), rigidState, rootTransform.worldFromLocal);
  }

  void SetStageStartTransformFromState(entt::entity entity, TransformRT const& rigidState) {
    auto& reg = GetRegistry();
    auto const& rigidInertia = reg.get<CRigidBodyInertia const>(entity);
    auto& rootTransform = reg.get<CRootTransform>(entity);
    mochi::rigid::RigidStateToRootTransform(
        rigidInertia.GetCenterOfMassLocal(), rigidState, rootTransform.worldFromLocalStageStart);
  }

  // Find out if a global DOF index is associated with the colliding or the collider entity.
  // Return the entity and the local DOF index for that entity.
  virtual std::pair<entt::entity, int> GetEntityAndLocalDofFromGlobalDof(int globalDof) {
    auto const& reg = GetRegistry();
    auto dofOffsetA = reg.get<CDofOffset>(_colliding).dofsOffset;
    auto dofOffsetB = (_numDofsB != 0) ? reg.get<CDofOffset>(_collider).dofsOffset : 0;
    if ((globalDof >= dofOffsetA) && (globalDof < dofOffsetA + _numDofsA)) {
      return std::make_pair(_colliding, globalDof - dofOffsetA);
    } else {
      EXPECT_TRUE(
          (globalDof >= dofOffsetB) &&
          (globalDof < dofOffsetB + _numDofsB)); // DOF index out of range
      return std::make_pair(_collider, globalDof - dofOffsetB);
    }
  }

  void PreStep() {
    // Prepare for contact by doing things that Scene::Step would normally take care of.
    // This includes: advancing time, which is required by UpdateConservativeStepBounds, which is
    // required by UpdateConservativePotentialColliders, which is required by
    // UpdatePotentialColliders, which happens during CollisionDetectionPipeline (see
    // ComputeResponse()).
    auto& reg = GetRegistry();
    double constexpr kTimeStep = 0.01; // Arbitrary
    reg.ctx<CSceneTime>().Advance(kTimeStep);
    PreStepEcs(reg);
  }

  void ComputeResponse(GradTarget gradTarget) {
    auto& reg = GetRegistry();

    // Clear result
    _objective = 0.0;
    _residual.SetZero();
    _dresidual.SetZero();

    // There should be exactly one island.
    entt::entity island = entt::null;
    for (auto [e] : reg.view<TagIsland>().each()) {
      EXPECT_FALSE(reg.valid(island));
      island = e;
    }
    EXPECT_TRUE(reg.valid(island));

    // TODO: Add this collision detection pass. Currently it fails because stage-start contacts of
    // the mapped collider produce no friction. Fix the mapped collider before adding this code.
    // T259662824
    // Run stage-start collision detection
    // auto const& descendants = reg.get<CIslandDescendants const>(island);
    // CollisionDetectionPipeline<TimeStep::StageStart>(reg, descendants);

    // Get island DOF info
    auto const& islandDofInfo = reg.get<CIslandDofInfo const>(island);
    int const solutionSize = islandDofInfo.poseSize;
    int const dofsSize = islandDofInfo.dofsSize;

    // Set up the SNLE problem with an assembly function
    SnleProblemFunctions<real> functions;
    functions.assemble = [&](SnleProblem<real>& problem, AssemblyParams const& params) {
      solver::AssembleIslandPipeline(reg, island, params, problem);
    };
    SnleProblem<real> problem(dofsSize, solutionSize, std::move(functions));

    // Initialize solution to zero
    problem.solution.SetZero();

    // Request full assembly (objective, residual, and dresidual) for GradTarget::Current, partial
    // assembly (objective and residual) for GradTarget::Previous.
    bool const assemDRes = gradTarget == GradTarget::Current;
    AssemblyParams params = {
        .assemObj = true,
        .assemRes = true,
        .assemDRes = assemDRes,
        .psdDRes = false,
        .fittedSaturationHessian = SaturationHessianParams::All(false),
        .gradTarget = gradTarget};
    problem.UpdateObjResDRes(params);

    // Extract results from the problem
    _objective = problem.GetObjective();
    _residual = problem.GetResidual();

    // If needed, copy dresidual to our sparse matrix
    if (assemDRes) {
      auto problemDRes = ToMatrix(problem.GetDResidual());
      for (int row = 0; row < _numDofs; ++row) {
        for (int col = 0; col < _numDofs; ++col) {
          _dresidual.SetValue(row, col, problemDRes(row, col));
        }
      }
    }
  }

  // Test with finite differences. For GradTarget::Current, test both residual and dresidual. For
  // GradTarget::Previous, test only the residual. The results for the dresidual are not perfect, as
  // the derivatives discard some nonlinearities (e.g., derivative of the SDF normal for some SDFs,
  // 2nd derivative of the mapping from contact sample to DoFs). In particular, error grows for the
  // rotation of the _collider, and its corresponding block in the Hessian is sometimes ignored
  void RunTest() {
    auto const& params = GetParam();

    auto solverParams = _scene->GetSolverParams();
    solverParams.experimentalEval.fittedSaturationHessian = SaturationHessianParams::All(false);
    solverParams.experimentalEval.explicitNormals = params.explicitNormals;
    // Disable fade friction if testing with GradTarget::Previous, or if testing with
    // GradTarget::Current and implicit normals.
    bool const isCurrentTarget = params.gradTarget == GradTarget::Current;
    solverParams.experimentalEval.fadeFriction = isCurrentTarget && params.explicitNormals;
    _scene->SetSolverParams(solverParams, test::ExpectOK{});

    InitializeScene(
        params.coulombCoefficient, params.viscousCoefficient, params.dampingCoefficient);

    InitState();
    PreStep();
    ComputeResponse(params.gradTarget);
    ColumnVector<real> res0 = _residual.Duplicate();
    SparseMatrix<real> dres = _dresidual.Duplicate();
    ColumnVector<real> resFD = _residual.Duplicate();
    SparseMatrix<real> dresFD = _dresidual.Duplicate();

    // Test through finite differences
    real const eps = params.eps;
    for (int i = 0; i < _numDofs; i++) {
      InitState();
      if (params.gradTarget == GradTarget::Current) {
        AddToState(i, eps);
      } else {
        AddToOldState(i, eps);
      }
      ComputeResponse(params.gradTarget);
      double objF = _objective;
      ColumnVector<real> resF = _residual.Duplicate();

      InitState();
      if (params.gradTarget == GradTarget::Current) {
        AddToState(i, -eps);
      } else {
        AddToOldState(i, -eps);
      }
      ComputeResponse(params.gradTarget);
      double objB = _objective;
      ColumnVector<real> resB = _residual.Duplicate();

      resFD[i] = (real)(objF - objB) / (2_r * eps);

      if (params.gradTarget == GradTarget::Current) {
        // Computing the dresidual by finite-differencing the residual is not accurate for Lie
        // rotation derivatives, but it seems sufficient in this case.
        ColumnVector<real> dresFDi = (resF - resB) * (1_r / (2_r * eps));
        for (int j = 0; j < _numDofs; ++j) {
          int row = j;
          int col = i;
          dresFD.SetValue(row, col, dresFDi[j]);
        }
      }
    }

    if (params.gradTarget == GradTarget::Current && params.ignoreColliderRotDres) {
      auto const colliderDofOffset = GetRegistry().get<CDofOffset const>(_collider).dofsOffset;
      std::vector<int> colliderRotDofs(RigidSize::kDRot);
      std::iota(
          colliderRotDofs.begin(), colliderRotDofs.end(), colliderDofOffset + RigidSize::kDTrans);
      SetZeroOnCols(AsView(dres), colliderRotDofs, 0, AsConstView(dres));
      SetZeroOnCols(AsView(dresFD), colliderRotDofs, 0, AsConstView(dresFD));
      SetZeroOnRows(AsView(dres), colliderRotDofs, 0, 1.0);
      SetZeroOnRows(AsView(dresFD), colliderRotDofs, 0, 1.0);
    }

    auto const resTol = params.resTol;
    ColumnVector<real> resDelta = res0 - resFD;
    auto const resNorm = resDelta.Norm();
    if (resNorm > 1e-6_r) {
      EXPECT_NEAR(resNorm / Max(res0.Norm(), resFD.Norm()), 0_r, resTol);
    }
    if (params.gradTarget == GradTarget::Current) {
      auto const dresTol = params.dresTol;
      EXPECT_EQ(dres.Pointers(), dresFD.Pointers()); // Expect same sparsity
      EXPECT_EQ(dres.Indices(), dresFD.Indices()); // Expect same sparsity
      ColumnVector<real> dresValueDelta = AsConstView(dres.Values()) - AsConstView(dresFD.Values());
      EXPECT_NEAR(dresValueDelta.Norm() / Max(dres.Norm(), dresFD.Norm()), 0_r, dresTol);
    }
  }

  Actor* InitStaticCube(
      real scale,
      TransformRT transform,
      real coulombCoefficient,
      real viscousCoefficient,
      real dampingCoefficient) {
    RigidActorParams params;
    params.isStatic = true;
    params.layer = "Collider";
    params.shape = _mochiContext->LoadShapeFromFile(
        test::GetAssetPath("cube/cube_mesh.mochi.json"),
        Real3{scale, scale, scale},
        TransformRT::Identity(),
        test::ExpectOK{});
    params.worldFromLocal = transform;

    params.contact.viscousFrictionCoefficient = viscousCoefficient;
    params.contact.coulombFrictionCoefficient = coulombCoefficient;
    params.contact.normalViscousDampingCoefficient = dampingCoefficient;
    params.contact.penaltyThresholdDefault = 5e-3_r;
    params.colliderType = ColliderType::Box;

    return _scene->CreateRigidActor(params, test::ExpectOK{});
  }

  Actor* InitRigidCube(
      real scale,
      TransformRT transform,
      bool isCollider,
      real coulombCoefficient,
      real viscousCoefficient,
      real dampingCoefficient) {
    auto& reg = GetRegistry();

    RigidActorParams params;
    params.density = 1e-10_r; // Negligible density to isolate contact forces from inertia.
    params.layer = isCollider ? "Collider" : "Colliding";
    params.shape = _mochiContext->LoadShapeFromFile(
        test::GetAssetPath("cube/cube_mesh.mochi.json"),
        Real3{scale, scale, scale},
        TransformRT::Identity(),
        test::ExpectOK{});
    params.worldFromLocal = transform;

    params.contact.viscousFrictionCoefficient = viscousCoefficient;
    params.contact.coulombFrictionCoefficient = coulombCoefficient;
    params.contact.normalViscousDampingCoefficient = dampingCoefficient;
    params.contact.penaltyThresholdDefault = 5e-3_r;
    params.colliderType = isCollider ? ColliderType::Box : ColliderType::None;

    auto* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
    auto entity = mochi::GetEntity(reg, actor->GetHandle(), test::ExpectOK{});
    reg.get<CTimeIntegratorState>(entity).dtStage = kDtStage;
    return actor;
  }

  Actor* InitSoftCube(
      TransformRT const& transform,
      bool isCollider,
      bool useDeepFlow,
      real coulombCoefficient,
      real viscousCoefficient,
      real dampingCoefficient) {
    auto& reg = GetRegistry();

    // Create soft actor. This is a cube of size 1, centered at 0.5
    SoftActorParams caparams;
    caparams.material.density =
        1e-10_r; // Negligible density to isolate contact forces from inertia.
    caparams.material.type = SoftMaterialType::NeoHookean;
    caparams.material.neoHookean.youngsModulus = 1e-6_r;
    caparams.layer = isCollider ? "Collider" : "Colliding";
    auto&& [unitCubeCoordinates, unitCubeConnectivity] = test::CreateMinimalTetMeshUnitCube();
    caparams.shape = _scene->GetContext()->CreateTetMeshShape(
        Flatten(MakeSpan(unitCubeCoordinates)),
        Flatten(MakeSpan(unitCubeConnectivity)),
        test::ExpectOK{});
    caparams.worldFromLocal = transform;

    caparams.contact.viscousFrictionCoefficient = viscousCoefficient;
    caparams.contact.coulombFrictionCoefficient = coulombCoefficient;
    caparams.contact.normalViscousDampingCoefficient = dampingCoefficient;

    ExperimentalSoftActorParams experimentalParams;
    if (isCollider) {
      experimentalParams.colliderType = ColliderType::Sdf;
      experimentalParams.sdf.resolutionDelta = {0.04_r, 0.04_r, 0.04_r};
      experimentalParams.sdf.resolutionMode = GridSdfResolutionMode::Explicit;
      caparams.contact.penaltyThresholdDefault = 0_r;

      if (useDeepFlow) {
        DeepModelParams flowParams;
        flowParams.shiftX = 0.49734753_r;
        flowParams.shiftY = 0.51221045_r;
        flowParams.shiftZ = 0.49657665_r;
        flowParams.scale = 1.3309090553162923_r;
        flowParams.deepModelPath = test::GetAssetPath("cube/cube_minimal_flow_revised.pt");
        flowParams.numDof = 24;
        mochi::ShapeHandle flow = CreateDeepFlowShape(
            _scene->GetContext(), flowParams, NeuralComputeType::MochiCpu, {}, test::ExpectOK());
        experimentalParams.flow = flow;
        caparams.contact.objScale = 1_r;
        caparams.contact.distanceErrorBound = flowParams.errorBound;
      }
    }
    auto* actor = CreateSoftActor(_scene, caparams, experimentalParams, test::ExpectOK{});
    auto entity = mochi::GetEntity(reg, actor->GetHandle(), test::ExpectOK{});
    reg.get<CTimeIntegratorState>(entity).dtStage = kDtStage;

    return actor;
  }

  void InitSoftCubeStageStart(entt::entity e) {
    auto& reg = GetRegistry();

    // Set up contact samples at stage start
    ecs::InvokeOnEntity(
        UpdateCollisionSamplePositions<
            CFemBoundaryDiscretization,
            TimeStep::StageStart,
            kSpaceDim3>,
        reg,
        e);

    // Set up the tetrahedral map at stage start
    ecs::TryInvokeOnEntity(soft::UpdateMap<TimeStep::StageStart>, reg, e);
  }
};

class MochiRigidStaticContact : public MochiContactTestBase {
  TransformRT _pos{};
  TransformRT _posOld{};

 public:
  MochiRigidStaticContact() {
    _numDofsA = 6; // Rigid
    _numDofsB = 0; // Static
    _numDofs = _numDofsA + _numDofsB;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    auto& reg = GetRegistry();

    auto transform =
        TransformRT(Quaternion::FromRotationVector(Real3(-1_r, -0.5_r, -0.4_r)), Real3{});
    auto* actorA =
        InitRigidCube(1_r, {}, false, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    transform = TransformRT(
        Quaternion::FromRotationVector(Real3(1_r, -0.1_r, 0.2_r)), Real3(0.8_r, 0.8_r, 0.8_r));
    auto* actorB =
        InitStaticCube(1_r, transform, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    InitActors(actorA, actorB);

    // Add velocity to the actor
    _pos = reg.get<CRigidState<TimeStep::Current>>(_colliding).value;
    auto& posOld = reg.get<CRigidState<TimeStep::StageStart>>(_colliding).value;
    posOld.SetTranslation(_pos.GetTranslation() + Real3(0.01_r, -0.02_r, 0.03_r));
    posOld.SetRotation(
        Quaternion::FromRotationVector(Vec4r(-0.2_r, 0.1_r, -0.1_r)) * _pos.GetRotation());
    SetStageStartTransformFromState(_colliding, posOld);
    _posOld = posOld;
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CRigidState<TimeStep::Current>>(_colliding).value = _pos;
    SetTransformFromState(_colliding, _pos);
    reg.get<CRigidState<TimeStep::StageStart>>(_colliding).value = _posOld;
    SetStageStartTransformFromState(_colliding, _posOld);
  }

  void AddToState(int globalDof, real eps) override {
    EXPECT_GT(_numDofs, globalDof);
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    Vec4r delta = SimdBasisVector(localDof % 3) * eps;
    auto& reg = GetRegistry();
    auto& rigidState = reg.get<CRigidState<TimeStep::Current>>(entity).value;
    if (localDof < 3) {
      // Change state of the rigid translation
      rigidState.SetTranslation(rigidState.GetTranslation() + ToReal3(delta));
    } else {
      // Change state of the rigid rotation
      rigidState.SetRotation(Quaternion::FromRotationVector(delta) * rigidState.GetRotation());
    }
    SetTransformFromState(entity, rigidState);
  }

  void AddToOldState(int globalDof, real eps) override {
    EXPECT_GT(_numDofs, globalDof);
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    Vec4r delta = SimdBasisVector(localDof % 3) * eps;
    auto& reg = GetRegistry();
    auto& rigidState = reg.get<CRigidState<TimeStep::StageStart>>(entity).value;
    if (localDof < 3) {
      // Change state of the rigid translation
      rigidState.SetTranslation(rigidState.GetTranslation() + ToReal3(delta));
    } else {
      // Change state of the rigid rotation
      rigidState.SetRotation(Quaternion::FromRotationVector(delta) * rigidState.GetRotation());
    }
    SetStageStartTransformFromState(entity, rigidState);
  }
};

TEST_P(MochiRigidStaticContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiRigidStaticContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 2e-3_r, 3e-2_r},
        TestParams{"Viscous", GradTarget::Current, 0_r, 1_r, 0_r, false, 2e-3_r, 3e-2_r},
        TestParams{"Coulomb", GradTarget::Current, 0.5_r, 0_r, 0_r, false, 2e-3_r, 3e-2_r},
        TestParams{"Damping", GradTarget::Current, 0_r, 0_r, 1_r, false, 2e-3_r, 4e-2_r},
        TestParams{"NoFrictionExplicit", GradTarget::Current, 0_r, 0_r, 0_r, true, 2e-3_r, 3e-2_r},
        TestParams{"ViscousExplicit", GradTarget::Current, 0_r, 1_r, 0_r, true, 1e-3_r, 3e-2_r},
        TestParams{"CoulombExplicit", GradTarget::Current, 0.5_r, 0_r, 0_r, true, 2e-3_r, 3e-2_r},
        TestParams{"DampingExplicit", GradTarget::Current, 0_r, 0_r, 1_r, true, 2e-3_r, 4e-2_r},
        TestParams{"NoFrictionPrevious", GradTarget::Previous, 0_r, 0_r, 0_r, true, {}, {}},
        TestParams{"ViscousPrevious", GradTarget::Previous, 0_r, 1_r, 0_r, true, 1e-3_r, {}},
        TestParams{"CoulombPrevious", GradTarget::Previous, 0.5_r, 0_r, 0_r, true, 1e-3_r, {}},
        TestParams{"DampingPrevious", GradTarget::Previous, 0_r, 0_r, 1_r, true, 3e-3_r, {}}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class MochiRigidRigidContact : public MochiContactTestBase {
  TransformRT _posA{};
  TransformRT _posB{};
  TransformRT _posAOld{};
  TransformRT _posBOld{};

 public:
  MochiRigidRigidContact() {
    _numDofsA = 6; // Rigid
    _numDofsB = 6; // Rigid
    _numDofs = _numDofsA + _numDofsB;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    auto& reg = GetRegistry();

    auto transform =
        TransformRT(Quaternion::FromRotationVector(Real3(-1_r, -0.5_r, -0.4_r)), Real3{});
    auto* actorA =
        InitRigidCube(1_r, {}, false, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    transform = TransformRT(
        Quaternion::FromRotationVector(Real3(1_r, -0.1_r, 0.2_r)), Real3(0.8_r, 0.8_r, 0.8_r));
    auto* actorB = InitRigidCube(
        1_r, transform, true, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    InitActors(actorA, actorB);

    // Add velocity to the actors
    _posA = reg.get<CRigidState<TimeStep::Current>>(_colliding).value;
    auto& posAold = reg.get<CRigidState<TimeStep::StageStart>>(_colliding).value;
    posAold.SetTranslation(_posA.GetTranslation() + Real3(0.01_r, -0.02_r, 0.03_r));
    posAold.SetRotation(
        Quaternion::FromRotationVector(Vec4r(-0.2_r, 0.1_r, -0.1_r)) * _posA.GetRotation());
    SetStageStartTransformFromState(_colliding, posAold);
    _posAOld = posAold;
    _posB = reg.get<CRigidState<TimeStep::Current>>(_collider).value;
    auto& posBold = reg.get<CRigidState<TimeStep::StageStart>>(_collider).value;
    posBold.SetTranslation(_posB.GetTranslation() + Real3(0.02_r, 0.01_r, -0.01_r));
    posBold.SetRotation(
        Quaternion::FromRotationVector(Vec4r(-0.1_r, -0.1_r, -0.2_r)) * _posB.GetRotation());
    SetStageStartTransformFromState(_collider, posBold);
    _posBOld = posBold;
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CRigidState<TimeStep::Current>>(_colliding).value = _posA;
    reg.get<CRigidState<TimeStep::Current>>(_collider).value = _posB;
    SetTransformFromState(_colliding, _posA);
    SetTransformFromState(_collider, _posB);
    reg.get<CRigidState<TimeStep::StageStart>>(_colliding).value = _posAOld;
    reg.get<CRigidState<TimeStep::StageStart>>(_collider).value = _posBOld;
    SetStageStartTransformFromState(_colliding, _posAOld);
    SetStageStartTransformFromState(_collider, _posBOld);
  }

  void AddToState(int globalDof, real eps) override {
    EXPECT_GT(_numDofs, globalDof);
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    Vec4r delta = SimdBasisVector(localDof % 3) * eps;
    auto& reg = GetRegistry();
    auto& rigidState = reg.get<CRigidState<TimeStep::Current>>(entity).value;
    if (localDof < 3) {
      // Change state of the rigid translation
      rigidState.SetTranslation(rigidState.GetTranslation() + ToReal3(delta));
    } else {
      // Change state of the rigid rotation
      rigidState.SetRotation(Quaternion::FromRotationVector(delta) * rigidState.GetRotation());
    }
    SetTransformFromState(entity, rigidState);
  }

  void AddToOldState(int globalDof, real eps) override {
    EXPECT_GT(_numDofs, globalDof);
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    Vec4r delta = SimdBasisVector(localDof % 3) * eps;
    auto& reg = GetRegistry();
    auto& rigidState = reg.get<CRigidState<TimeStep::StageStart>>(entity).value;
    if (localDof < 3) {
      // Change state of the rigid translation
      rigidState.SetTranslation(rigidState.GetTranslation() + ToReal3(delta));
    } else {
      // Change state of the rigid rotation
      rigidState.SetRotation(Quaternion::FromRotationVector(delta) * rigidState.GetRotation());
    }
    SetStageStartTransformFromState(entity, rigidState);
  }
};

TEST_P(MochiRigidRigidContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiRigidRigidContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 2e-3_r, 6e-2_r},
        TestParams{"Viscous", GradTarget::Current, 0_r, 1_r, 0_r, false, 2e-3_r, 4e-2_r},
        TestParams{"Coulomb", GradTarget::Current, 0.5_r, 0_r, 0_r, false, 4e-3_r, 6e-2_r},
        TestParams{"Damping", GradTarget::Current, 0_r, 0_r, 1_r, false, 2e-3_r, 3e-2_r},
        TestParams{"NoFrictionExplicit", GradTarget::Current, 0_r, 0_r, 0_r, true, 2e-3_r, 7e-2_r},
        TestParams{"ViscousExplicit", GradTarget::Current, 0_r, 1_r, 0_r, true, 2e-3_r, 4e-2_r},
        TestParams{"CoulombExplicit", GradTarget::Current, 0.5_r, 0_r, 0_r, true, 2e-3_r, 7e-2_r},
        TestParams{"DampingExplicit", GradTarget::Current, 0_r, 0_r, 1_r, true, 2e-3_r, 3e-2_r},
        TestParams{"NoFrictionPrevious", GradTarget::Previous, 0_r, 0_r, 0_r, true, {}, {}},
        TestParams{"ViscousPrevious", GradTarget::Previous, 0_r, 1_r, 0_r, true, 2e-3_r, {}},
        TestParams{"CoulombPrevious", GradTarget::Previous, 0.5_r, 0_r, 0_r, true, 1e-3_r, {}},
        TestParams{"DampingPrevious", GradTarget::Previous, 0_r, 0_r, 1_r, true, 4e-3_r, {}}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class MochiSoftStaticContact : public MochiContactTestBase {
  ColumnVector<real> _posA;

 public:
  MochiSoftStaticContact() {
    _numDofsA = 24; // a soft with 8 nodes
    _numDofsB = 0; // Static
    _numDofs = _numDofsA + _numDofsB;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    auto& reg = GetRegistry();

    auto* actorA =
        InitSoftCube({}, false, {}, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    auto transform = TransformRT(
        Quaternion::FromRotationVector(Real3(0.1_r, -0.1_r, 0.2_r)), Real3(0.6_r, 0.6_r, 0.6_r));
    auto* actorB =
        InitStaticCube(1_r, transform, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    InitActors(actorA, actorB);

    _posA =
        reg.get<CDisplacementSlice<real, TimeStep::Current> const>(_colliding).value.Duplicate();

    // Add velocity to the actor
    auto& posAold = reg.get<CDisplacementSlice<real, TimeStep::StageStart>>(_colliding);
    for (int i = 0; i < _numDofsA; i++) {
      posAold.value[i] = (i % 3 == 0) ? -0.01_r : 0.01_r;
    }

    InitSoftCubeStageStart(_colliding);
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CDisplacementSlice<real, TimeStep::Current>>(_colliding).value = _posA;
  }

  void AddToState(int globalDof, real eps) override {
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    auto& reg = GetRegistry();
    MOCHI_ASSERT_VERBOSE(entity == _colliding);

    // Change state of a node of object A
    auto& posA = reg.get<CDisplacementSlice<real, TimeStep::Current>>(_colliding);
    posA.value[localDof] += eps;
  }
};

TEST_P(MochiSoftStaticContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiSoftStaticContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 1e-3_r, 1e-3_r},
        TestParams{"Viscous", GradTarget::Current, 0_r, 1_r, 0_r, false, 1e-3_r, 1e-3_r},
        TestParams{"Coulomb", GradTarget::Current, 0.5_r, 0_r, 0_r, false, 1e-3_r, 1e-3_r},
        TestParams{"Damping", GradTarget::Current, 0_r, 0_r, 1_r, false, 1e-3_r, 1e-3_r},
        TestParams{"NoFrictionExplicit", GradTarget::Current, 0_r, 0_r, 0_r, true, 1e-3_r, 1e-3_r},
        TestParams{"ViscousExplicit", GradTarget::Current, 0_r, 1_r, 0_r, true, 1e-3_r, 1e-3_r},
        TestParams{"CoulombExplicit", GradTarget::Current, 0.5_r, 0_r, 0_r, true, 1e-3_r, 1e-3_r},
        TestParams{"DampingExplicit", GradTarget::Current, 0_r, 0_r, 1_r, true, 1e-3_r, 1e-3_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class MochiSoftRigidContact : public MochiContactTestBase {
  ColumnVector<real> _posA;
  TransformRT _posB;

 public:
  MochiSoftRigidContact() {
    _numDofsA = 24; // a soft with 8 nodes
    _numDofsB = 6; // Rigid
    _numDofs = _numDofsA + _numDofsB;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    auto& reg = GetRegistry();

    auto* actorA =
        InitSoftCube({}, false, {}, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    auto transform = TransformRT(
        Quaternion::FromRotationVector(Real3(0.1_r, -0.1_r, 0.2_r)), Real3(0.6_r, 0.6_r, 0.6_r));
    auto* actorB = InitRigidCube(
        1_r, transform, true, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    InitActors(actorA, actorB);

    _posA =
        reg.get<CDisplacementSlice<real, TimeStep::Current> const>(_colliding).value.Duplicate();

    // Add velocity to the actors
    auto& posAold = reg.get<CDisplacementSlice<real, TimeStep::StageStart>>(_colliding);
    for (int i = 0; i < _numDofsA; i++) {
      posAold.value[i] = (i % 3 == 0) ? -0.01_r : 0.01_r;
    }
    _posB = reg.get<CRigidState<TimeStep::Current>>(_collider).value;
    auto& posBold = reg.get<CRigidState<TimeStep::StageStart>>(_collider).value;
    posBold.SetTranslation(_posB.GetTranslation() + Real3(0.02_r, 0.01_r, -0.01_r));
    posBold.SetRotation(
        Quaternion::FromRotationVector(Vec4r(-0.1_r, -0.1_r, -0.2_r)) * _posB.GetRotation());

    InitSoftCubeStageStart(_colliding);
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CDisplacementSlice<real, TimeStep::Current>>(_colliding).value = _posA;
    reg.get<CRigidState<TimeStep::Current>>(_collider).value = _posB;
    SetTransformFromState(_collider, _posB);
  }

  void AddToState(int globalDof, real eps) override {
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    auto& reg = GetRegistry();
    if (entity == _colliding) {
      // Change state of a node of object A
      auto& posA = reg.get<CDisplacementSlice<real, TimeStep::Current>>(_colliding);
      posA.value[localDof] += eps;
    } else {
      Vec4r delta = SimdBasisVector(localDof % 3) * eps;
      auto& rigidState = reg.get<CRigidState<TimeStep::Current>>(_collider).value;
      if (localDof < 3) {
        // Change state of the rigid translation
        rigidState.SetTranslation(rigidState.GetTranslation() + ToReal3(delta));
      } else {
        // Change state of the rigid rotation
        rigidState.SetRotation(Quaternion::FromRotationVector(delta) * rigidState.GetRotation());
      }
      SetTransformFromState(_collider, rigidState);
    }
  }
};

TEST_P(MochiSoftRigidContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiSoftRigidContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 1e-3_r, 1e-3_r, true},
        TestParams{"Viscous", GradTarget::Current, 0_r, 1_r, 0_r, false, 1e-3_r, 1e-3_r, true},
        TestParams{"Coulomb", GradTarget::Current, 0.5_r, 0_r, 0_r, false, 3e-3_r, 1e-3_r, true},
        TestParams{
            "NoFrictionExplicit",
            GradTarget::Current,
            0_r,
            0_r,
            0_r,
            true,
            1e-3_r,
            1e-3_r,
            true},
        TestParams{
            "ViscousExplicit",
            GradTarget::Current,
            0_r,
            1_r,
            0_r,
            true,
            1e-3_r,
            1e-3_r,
            true},
        TestParams{
            "CoulombExplicit",
            GradTarget::Current,
            0.5_r,
            0_r,
            0_r,
            true,
            1e-3_r,
            1e-3_r,
            true},
        TestParams{"Damping", GradTarget::Current, 0_r, 0_r, 1_r, false, 1e-3_r, 1e-3_r, true},
        TestParams{
            "DampingExplicit",
            GradTarget::Current,
            0_r,
            0_r,
            1_r,
            true,
            1e-3_r,
            1e-3_r,
            true}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class MochiRigidSoftContact : public MochiContactTestBase {
  TransformRT _posA;
  ColumnVector<real> _posB;

 public:
  MochiRigidSoftContact() {
    _numDofsA = 6; // Rigid
    _numDofsB = 24; // a soft with 8 nodes
    _numDofs = _numDofsA + _numDofsB;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    auto& reg = GetRegistry();

    auto transform = TransformRT(
        Quaternion::FromRotationVector(Real3(0.1_r, -0.1_r, 0.2_r)), Real3(0.6_r, 0.6_r, 0.6_r));
    auto* actorA = InitRigidCube(
        1_r, transform, false, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    auto* actorB =
        InitSoftCube({}, true, {}, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    InitActors(actorA, actorB);

    // Add velocity to the actors
    _posA = reg.get<CRigidState<TimeStep::Current>>(_colliding).value;
    auto& posAold = reg.get<CRigidState<TimeStep::StageStart>>(_colliding).value;
    posAold.SetTranslation(_posA.GetTranslation() + Real3(0.02_r, 0.01_r, -0.01_r));
    posAold.SetRotation(
        Quaternion::FromRotationVector(Vec4r(-0.1_r, -0.1_r, -0.2_r)) * _posA.GetRotation());
    _posB = reg.get<CDisplacementSlice<real, TimeStep::Current> const>(_collider).value.Duplicate();
    auto& posBold = reg.get<CDisplacementSlice<real, TimeStep::StageStart>>(_collider);
    for (int i = 0; i < _numDofsB; i++) {
      posBold.value[i] = (i % 3 == 0) ? -0.01_r : 0.01_r;
    }

    InitSoftCubeStageStart(_collider);
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CRigidState<TimeStep::Current>>(_colliding).value = _posA;
    SetTransformFromState(_colliding, _posA);
    reg.get<CDisplacementSlice<real, TimeStep::Current>>(_collider).value = _posB;
  }

  void AddToState(int globalDof, real eps) override {
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    auto& reg = GetRegistry();
    if (entity == _colliding) {
      Vec4r delta = SimdBasisVector(localDof % 3) * eps;
      auto& rigidState = reg.get<CRigidState<TimeStep::Current>>(_colliding).value;
      if (localDof < 3) {
        // Change state of the rigid translation
        rigidState.SetTranslation(rigidState.GetTranslation() + ToReal3(delta));
      } else {
        // Change state of the rigid rotation
        rigidState.SetRotation(Quaternion::FromRotationVector(delta) * rigidState.GetRotation());
      }
      SetTransformFromState(_colliding, rigidState);
    } else {
      // Change state of a node of object B
      auto& posB = reg.get<CDisplacementSlice<real, TimeStep::Current>>(_collider);
      posB.value[localDof] += eps;
    }
  }
};

TEST_P(MochiRigidSoftContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiRigidSoftContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 1e-3_r, 3e-1_r},
        TestParams{"Viscous", GradTarget::Current, 0_r, 10_r, 0_r, false, 1e-3_r, 2e-2_r},
        TestParams{"Coulomb", GradTarget::Current, 0.5_r, 0_r, 0_r, false, 1e-3_r, 6e-2_r},
        TestParams{"Damping", GradTarget::Current, 0_r, 0_r, 10_r, false, 1e-3_r, 2e-2_r},
        TestParams{"NoFrictionExplicit", GradTarget::Current, 0_r, 0_r, 0_r, true, 1e-3_r, 3e-1_r},
        TestParams{"ViscousExplicit", GradTarget::Current, 0_r, 10_r, 0_r, true, 1e-3_r, 2e-2_r},
        TestParams{"CoulombExplicit", GradTarget::Current, 0.5_r, 0_r, 0_r, true, 1e-3_r, 3e-1_r},
        TestParams{"DampingExplicit", GradTarget::Current, 0_r, 0_r, 10_r, true, 2e-3_r, 1e-2_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// Scene with 2 soft actors. Actor A is a translated soft cube. Actor B is a soft cube at the
// origin, and it has a mapped SDF _collider. The result of collision detection doesn't need to be
// transformed, as B is at the origin.
class MochiSoftSoftContact : public MochiContactTestBase {
 protected:
  ColumnVector<real> _posA;
  ColumnVector<real> _posB;
  bool _useDeepFlow = false;

 public:
  MochiSoftSoftContact() {
    _numDofsA = 24; // a soft with 8 nodes
    _numDofsB = 24; // a soft with 8 nodes
    _numDofs = _numDofsA + _numDofsB;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    auto& reg = GetRegistry();

    // Apply a transformation to avoid contacts on edges
    TransformRT transformA(Vec4r(0.98_r, 0.24_r, 0.16_r));
    auto* actorA = InitSoftCube(
        transformA, false, {}, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    auto* actorB = InitSoftCube(
        {}, true, _useDeepFlow, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    InitActors(actorA, actorB);

    // Set initial positions
    auto& posA = reg.get<CDisplacementSlice<real, TimeStep::Current>>(_colliding);
    auto& posB = reg.get<CDisplacementSlice<real, TimeStep::Current>>(_collider);
    for (int i = 0; i < _numDofsA; i++) {
      posA.value[i] = (i % 3 == 0) ? 0_r : 0.01_r;
      posB.value[i] = (i % 3 == 0) ? 0_r : -0.01_r;
    }
    _posA = posA.value.Duplicate();
    _posB = posB.value.Duplicate();

    // Add velocity to the actors
    auto& posAold = reg.get<CDisplacementSlice<real, TimeStep::StageStart>>(_colliding);
    for (int i = 0; i < _numDofsA; i++) {
      posAold.value[i] = (i % 3 == 0) ? -0.01_r : 0.01_r;
    }
    auto& posBold = reg.get<CDisplacementSlice<real, TimeStep::StageStart>>(_collider);
    for (int i = 0; i < _numDofsB; i++) {
      posBold.value[i] = (i % 3 == 0) ? -0.01_r : 0.01_r;
    }

    InitSoftCubeStageStart(_colliding);
    InitSoftCubeStageStart(_collider);
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CDisplacementSlice<real, TimeStep::Current>>(_colliding).value = _posA;
    reg.get<CDisplacementSlice<real, TimeStep::Current>>(_collider).value = _posB;
  }

  void AddToState(int globalDof, real eps) override {
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    auto& reg = GetRegistry();
    auto& pos = reg.get<CDisplacementSlice<real, TimeStep::Current>>(entity);
    pos.value[localDof] += eps;
  }
};

TEST_P(MochiSoftSoftContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiSoftSoftContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 1e-3_r, 1e-1_r},
        TestParams{"Viscous", GradTarget::Current, 0_r, 10_r, 0_r, false, 1e-3_r, 1e-1_r},
        TestParams{"Coulomb", GradTarget::Current, 0.5_r, 0_r, 0_r, false, 1e-3_r, 1e-1_r},
        TestParams{"Damping", GradTarget::Current, 0_r, 0_r, 10_r, false, 1e-3_r, 5e-3_r},
        TestParams{"NoFrictionExplicit", GradTarget::Current, 0_r, 0_r, 0_r, true, 1e-3_r, 1e-1_r},
        TestParams{"ViscousExplicit", GradTarget::Current, 0_r, 10_r, 0_r, true, 1e-3_r, 1e-1_r},
        TestParams{"CoulombExplicit", GradTarget::Current, 0.5_r, 0_r, 0_r, true, 1e-3_r, 1e-1_r},
        TestParams{"DampingExplicit", GradTarget::Current, 0_r, 0_r, 10_r, true, 1e-3_r, 5e-3_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class MochiSoftSoftContactDeepFlow : public MochiSoftSoftContact {
 public:
  MochiSoftSoftContactDeepFlow() {
    _useDeepFlow = true;
  }
};

// Test of the integration of deep flow and biharmonic ROMs
// TODO[T152549129] DISABLED because it torch::jit::load crashes or never returns in debug builds.
// TODO Also disabled when real is type double, because the saved torch files only support floats.
#if MOCHI_USE_TORCH && !MOCHI_DEBUG && !MOCHI_USE_DOUBLE_PRECISION && MOCHI_ENABLE_DEEP_FLOW_ACTORS
#define MOCHI_CAN_TEST_DEEP_FLOW 0 // TODO(T228959906): Re-enable deep flow tests
#else
#define MOCHI_CAN_TEST_DEEP_FLOW 0
#endif

// The Deep Flow model data is not shipped externally.
#if MOCHI_CAN_TEST_DEEP_FLOW && MOCHI_INTERNAL
#define MOCHI_CAN_TEST_DEEP_FLOW_INTERNAL 1
#else
#define MOCHI_CAN_TEST_DEEP_FLOW_INTERNAL 0
#endif

TEST_IF_P(MOCHI_CAN_TEST_DEEP_FLOW_INTERNAL, MochiSoftSoftContactDeepFlow, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiSoftSoftContactDeepFlow,
    ::testing::Values(
        TestParams{"Viscous", GradTarget::Current, 0_r, 10_r, 0_r, false, 6e-2_r, 1e-1_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class DeepFlowBiharmonicRom : public MochiContactTestBase {
 protected:
  static constexpr real kCubeScale = 0.1_r;
  static constexpr real kDuckScale = 0.1_r;
  TransformRT _posA;
  ColumnVector<real> _posB;

 public:
  DeepFlowBiharmonicRom() {
    _numDofsA = 6; // Rigid
    _numDofsB = 30; // a biharmonic ROM with 30 DoFs
    _numDofs = _numDofsA + _numDofsB;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    auto* actorA = InitRigidCube(
        kCubeScale, {}, false, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    auto* actorB = InitRomDuck(coulombCoefficient, viscousCoefficient, dampingCoefficient);
    InitActors(actorA, actorB);
    auto& reg = GetRegistry();
    _posA = reg.get<CRigidState<TimeStep::Current>>(_colliding).value;
    _posB = reg.get<CRomModeAmplitudes const>(_collider).value.Duplicate();
  }

  Actor* InitRomDuck(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient) {
    ShapeHandle duck = _mochiContext->LoadShapeFromFile(
        test::GetAssetPath("duck/duck_coarse.mochi.h5"),
        Real3{kDuckScale, kDuckScale, kDuckScale},
        TransformRT::Identity(),
        test::ExpectOK());
    DeepModelParams dflowparams;
    dflowparams.deepModelPath =
        test::GetAssetPath("duck/duck_coarse_biharmonic_rom/duck_10handles_flow.pt");
    dflowparams.shiftX = 0.43273339_r;
    dflowparams.shiftY = 0.290418_r;
    dflowparams.shiftZ = 0.49103763_r;
    dflowparams.scale = 0.6447050910181614_r;
    dflowparams.numDof = 30;
    ShapeHandle flow = CreateDeepFlowShape(
        _mochiContext, dflowparams, NeuralComputeType::MochiCpu, {}, test::ExpectOK());
    SoftActorParams params;
    params.layer = "Collider";
    params.shape = duck;
    params.worldFromLocal.SetTranslation(Real3(0_r, 0.96_r * kDuckScale, 0_r));
    params.material.density = 1e-10_r; // Negligible density to isolate contact forces from inertia.
    params.material.type = SoftMaterialType::NeoHookean;
    params.material.neoHookean.youngsModulus = 1e-6_r;
    params.contact.objScale = kDuckScale;
    params.contact.penaltyThresholdDefault = 0_r;
    params.contact.penaltySmoothingHalfDistance = 0.001_r;
    params.contact.viscousFrictionCoefficient = viscousCoefficient;
    params.contact.coulombFrictionCoefficient = coulombCoefficient;
    params.contact.normalViscousDampingCoefficient = dampingCoefficient;
    params.contact.distanceErrorBound = kDuckScale * dflowparams.errorBound;

    ExperimentalSoftActorParams experimentalParams;
    experimentalParams.colliderType = ColliderType::Sdf;
    experimentalParams.flow = flow;
    experimentalParams.rom = RomParams{.source = "biharmonic_10"};

    return CreateSoftActor(_scene, params, experimentalParams, test::ExpectOK());
  }

  void PostResetStateRom(entt::entity entity) {
    auto& reg = GetRegistry();

    // Update FOM state
    // ResolveDisplacement requires setting the template kForceUseAllNodes.
    // This test does not use hyper-reduction on the ROM actor, so we could
    // set it true or false and should have the same result.
    ecs::TryInvokeOnEntity(
        &rom::linear::ResolveDisplacement</*kForceUseAllNodes = */ true>, reg, entity);

    // Update the current velocity of the ROM in reduced coordinates
    ecs::TryInvokeOnEntity(&rom::UpdateCurrentRomVelocity, reg, entity);
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CRigidState<TimeStep::Current>>(_colliding).value = _posA;
    SetTransformFromState(_colliding, _posA);
    reg.get<CRomModeAmplitudes>(_collider).value = _posB;
    PostResetStateRom(_collider);
  }

  void AddToState(int globalDof, real eps) override {
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    auto& reg = GetRegistry();
    if (entity == _colliding) {
      Vec4r delta = SimdBasisVector(localDof % 3) * eps;
      auto& rigidState = reg.get<CRigidState<TimeStep::Current>>(_colliding).value;
      if (localDof < 3) {
        // Change state of the rigid translation
        rigidState.SetTranslation(rigidState.GetTranslation() + ToReal3(delta));
      } else {
        // Change state of the rigid rotation
        rigidState.SetRotation(Quaternion::FromRotationVector(delta) * rigidState.GetRotation());
      }
      SetTransformFromState(_colliding, rigidState);
    } else {
      // Change ROM state
      auto& pos = reg.get<CRomModeAmplitudes>(_collider).value;
      pos(localDof) += eps;
      PostResetStateRom(_collider);
    }
  }
};

TEST_IF_P(MOCHI_CAN_TEST_DEEP_FLOW_INTERNAL, DeepFlowBiharmonicRom, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    DeepFlowBiharmonicRom,
    ::testing::Values(
        TestParams{"Viscous", GradTarget::Current, 0_r, 1_r, 0_r, false, 1e-2_r, 2e-1_r, false},
        TestParams{
            "Coulomb",
            GradTarget::Current,
            0.5_r,
            0_r,
            0_r,
            false,
            1e-2_r,
            2e-1_r,
            false,
            1e-5_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// Scene with 2 articulated actors. Each articulated actor consists of 2 rigid actors connected
// through a revolute joint.
using namespace mochi::articulated;
using namespace mochi::articulated::compound;
class MochiArticulatedArticulatedContact : public MochiContactTestBase {
 protected:
  // This class needs to overwrite the 'colliding' and 'collider' actors to use rigid bodies.
  // '_artA' and '_artB' store the articulated bodies for state updates.
  entt::entity _artA = entt::null;
  entt::entity _artB = entt::null;
  ColumnVector<real> _posA;
  ColumnVector<real> _posB;

 public:
  MochiArticulatedArticulatedContact() {
    _numDofsA = 9; // two bones with one spherical joint
    _numDofsB = 9; // two bones with one spherical joint
    _numDofs = _numDofsA + _numDofsB;
  }

  Actor* CreateArticulated(
      ShapeHandle const& boneShape,
      RigidActorParams const& rigidParams,
      std::vector<TransformRT> const& worldFromLocalTxs) {
    ArticulatedActorParams params;

    params.joints.resize(2);
    params.joints[0].type = ArticulatedJointType::Free;
    params.joints[0].parentLinkFromJoint = worldFromLocalTxs[0];
    params.joints[1].type = ArticulatedJointType::Spherical;
    params.joints[1].parentLinkFromJoint = Invert(worldFromLocalTxs[0]) * worldFromLocalTxs[1];

    params.links.resize(2);
    for (int i = 0; i < 2; ++i) {
      params.links[i].parentLink = i == 0 ? -1 : 0;
      params.links[i].shape = boneShape;
      params.links[i].layer = rigidParams.layer;
      params.links[i].colliderType = rigidParams.colliderType;
      params.links[i].contact = rigidParams.contact;
    }

    Actor* articulatedActor = _scene->CreateArticulatedActor(params, test::ExpectOK{});

    // Set time step size.
    auto actors = articulatedActor->GetNestedLinkActors(test::ExpectOK{});
    auto& reg = GetRegistry();
    for (auto actor : actors) {
      auto entity = mochi::GetEntity(reg, actor, test::ExpectOK{});
      reg.get<CTimeIntegratorState>(entity).dtStage = kDtStage;
    }

    return articulatedActor;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    auto& reg = GetRegistry();

    // Create bone shape
    ShapeHandle boneShape = _mochiContext->LoadShapeFromFile(
        test::GetAssetPath("cube/cube_mesh.mochi.json"), test::ExpectOK{});

    // Create colliding
    Actor* actorA = nullptr;
    {
      std::vector<TransformRT> worldFromLocalTxs = {{}, TransformRT(Real3(1_r, 0_r, 0_r))};

      RigidActorParams rigidParams;
      rigidParams.contact.viscousFrictionCoefficient = viscousCoefficient;
      rigidParams.contact.coulombFrictionCoefficient = coulombCoefficient;
      rigidParams.contact.normalViscousDampingCoefficient = dampingCoefficient;
      rigidParams.layer = "Colliding";

      actorA = CreateArticulated(boneShape, rigidParams, worldFromLocalTxs);
    }

    Actor* actorB = nullptr;
    {
      std::vector<TransformRT> worldFromLocalTxs = {
          TransformRT(Real3(2.5_r, 1_r, 0_r)), TransformRT(Real3(1.5_r, 1_r, 0_r))};

      RigidActorParams rigidParams;
      rigidParams.layer = "Collider";
      rigidParams.colliderType = ColliderType::Box;
      rigidParams.contact.penaltyThresholdDefault = 0_r;
      rigidParams.contact.viscousFrictionCoefficient = viscousCoefficient;
      rigidParams.contact.coulombFrictionCoefficient = coulombCoefficient;
      rigidParams.contact.normalViscousDampingCoefficient = dampingCoefficient;

      actorB = CreateArticulated(boneShape, rigidParams, worldFromLocalTxs);
    }

    // Create the compound and overwrite 'colliding' and 'collider'
    InitActors(actorA, actorB);
    _artA = _colliding;
    _artB = _collider;
    auto actorA2 = actorA->GetNestedLinkActors(test::ExpectOK{})[1];
    auto actorB2 = actorB->GetNestedLinkActors(test::ExpectOK{})[1];
    _colliding = mochi::GetEntity(reg, actorA2, test::ExpectOK{});
    _collider = mochi::GetEntity(reg, actorB2, test::ExpectOK{});

    // Initialize previous state
    CopyToPreviousState(_colliding);
    CopyToPreviousState(_collider);

    // Add some initial rotations
    auto tinyRot = Real3{0_r, 0_r, 0.01_r};
    auto& posA = reg.get<CArticulatedReducedPose<TimeStep::Current>>(_artA).value;
    posA.BottomRows<RigidSize::kRot>(RigidSize::kRot) =
        AsColumnVectorView(Quaternion::FromRotationVector(tinyRot).data);
    auto& posB = reg.get<CArticulatedReducedPose<TimeStep::Current>>(_artB).value;
    posB.BottomRows<RigidSize::kRot>(RigidSize::kRot) =
        AsColumnVectorView(Quaternion::FromRotationVector(-tinyRot).data);
    UpdateFullState(_artA, _colliding);
    UpdateFullState(_artB, _collider);
    _posA = posA.Duplicate();
    _posB = posB.Duplicate();

    // Update islands manually since this test does not step the scene.
    // This assigns DOF offsets, among other things.
    island::PreStep(reg);
  }

  std::pair<entt::entity, int> GetEntityAndLocalDofFromGlobalDof(int globalDof) override {
    // In this case the global DOFs belong to _artA or _artB, not the colliding or collider
    // entities, which are full DOF rigid bodies. See InitializeScene().
    auto const& reg = GetRegistry();
    auto dofOffsetA = reg.get<CDofOffset>(_artA).dofsOffset;
    auto dofOffsetB = reg.get<CDofOffset>(_artB).dofsOffset;
    if ((globalDof >= dofOffsetA) && (globalDof < dofOffsetA + _numDofsA)) {
      return std::make_pair(_artA, globalDof - dofOffsetA);
    } else {
      EXPECT_TRUE(
          (globalDof >= dofOffsetB) &&
          (globalDof < dofOffsetB + _numDofsB)); // DOF index out of range
      return std::make_pair(_artB, globalDof - dofOffsetB);
    }
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CArticulatedReducedPose<TimeStep::Current>>(_artA).value = _posA;
    reg.get<CArticulatedReducedPose<TimeStep::Current>>(_artB).value = _posB;
    UpdateFullState(_artA, _colliding);
    UpdateFullState(_artB, _collider);
  }

  void AddToState(int globalDof, real eps) override {
    auto [art, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    auto body = (art == _artA) ? _colliding : _collider;
    auto& posReduced = GetRegistry().get<CArticulatedReducedPose<TimeStep::Current>>(art).value;
    ColumnVector<real> delta(GetRegistry().get<CArticulatedProps const>(art).reducedDofsDim);
    delta.SetZero();
    delta[localDof] = eps;
    auto const* joints = GetRegistry().get<CArticulatedBodyShape const>(art).shape->GetJointsData();
    auto const& poseInfo = GetRegistry().get<CArticulatedJointPoseInfo const>(art);
    articulated::AddLieDeltaToReducedPose(
        joints->jointTypes, joints->dofInfo, poseInfo, posReduced, delta, posReduced);
    UpdateFullState(art, body);
  }

  void UpdateFullState(entt::entity art, entt::entity body) {
    auto& reg = GetRegistry();

    auto const* joints = reg.get<CArticulatedBodyShape const>(art).shape->GetJointsData();
    auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(art);
    auto const& parents = reg.get<CArticulatedParents const>(art);
    auto const& restTransforms = reg.get<CArticulatedRestTransforms const>(art);
    auto const& reducedPose = reg.get<CArticulatedReducedPose<TimeStep::Current>>(art).value;
    auto const& worldFromRoot = reg.get<CRootTransform const>(art).worldFromLocal;
    auto& jointTransforms = reg.get<CArticulatedJointTransforms<TimeStep::Current>>(art);
    auto& linkTransforms = reg.get<CArticulatedLinkTransforms<TimeStep::Current>>(art);
    auto& fullDofs = reg.get<CArticulatedFullPose>(art).value;
    ComputeFullPose(
        joints->jointTypes,
        joints->jointAxes,
        poseInfo,
        parents,
        restTransforms,
        worldFromRoot,
        reducedPose,
        jointTransforms,
        linkTransforms,
        fullDofs);

    auto const& posFull = reg.get<CArticulatedFullPose const>(art).value;
    auto const& offset = reg.get<CDofOffset const>(body).poseOffset;
    auto& rigidState = reg.get<CRigidState<TimeStep::Current>>(body);
    mochi::rigid::EntitySetSolution(
        posFull.MiddleRows(offset, RigidSize::kAll),
        ecs::Included<TagRigidActor>{},
        {},
        rigidState);
    auto const& rigidInertia = reg.get<CRigidBodyInertia const>(body);
    auto& rootTransform = reg.get<CRootTransform>(body);
    mochi::rigid::RigidStateToRootTransform(
        rigidInertia.GetCenterOfMassLocal(), rigidState.value, rootTransform.worldFromLocal);
  }

  void CopyToPreviousState(entt::entity body) {
    auto& reg = GetRegistry();

    auto const& rigidState = reg.get<CRigidState<TimeStep::Current> const>(body).value;
    auto& prevRigidState = reg.get<CRigidState<TimeStep::Previous>>(body).value;
    prevRigidState = rigidState;
  }
};

TEST_P(MochiArticulatedArticulatedContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiArticulatedArticulatedContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 2e-3_r, 1e-2_r},
        TestParams{"Viscous", GradTarget::Current, 0_r, 10_r, 0_r, false, 4e-3_r, 1e-2_r},
        TestParams{"Coulomb", GradTarget::Current, 0.5_r, 0_r, 0_r, false, 2e-3_r, 1e-2_r},
        TestParams{"Damping", GradTarget::Current, 0_r, 0_r, 10_r, false, 4e-3_r, 1e-2_r},
        TestParams{"NoFrictionExplicit", GradTarget::Current, 0_r, 0_r, 0_r, true, 2e-3_r, 1e-2_r},
        TestParams{"ViscousExplicit", GradTarget::Current, 0_r, 10_r, 0_r, true, 2e-3_r, 1e-2_r},
        TestParams{"CoulombExplicit", GradTarget::Current, 0.5_r, 0_r, 0_r, true, 2e-3_r, 1e-2_r},
        TestParams{"DampingExplicit", GradTarget::Current, 0_r, 0_r, 10_r, true, 2e-3_r, 1e-2_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// Verify consistency of contact torque queries on a rigid cube constrained at its center of mass,
// contacted by another orbiting rigid cube. Checks:
//   1) Body torque from GetContactTorqueWorld matches body torque summed from GetContactPointsWorld
//   2) dL/dt (finite difference of angular momentum) matches contact torque
class ContactTorqueScene : public test::MochiSceneTestBase {
 protected:
  static constexpr real kCubeSize = 0.1_r;
  static constexpr Real3 kTargetPosition = {0.3_r, 0.2_r, -0.1_r};
  static constexpr real kOrbitRadius = kCubeSize;

  Actor* _targetActor = nullptr;
  Constraint* _orbiterConstraint = nullptr;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();
    _scene->SetGravity(Real3{});

    auto&& [coords, connectivity] =
        test::CreateMinimalTetMeshUnitCube(Real3{kCubeSize, kCubeSize, kCubeSize});
    auto cubeShape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coords)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});

    // Target cube: constrained at center of mass
    {
      RigidActorParams params;
      params.name = "Target";
      params.shape = cubeShape;
      params.worldFromLocal.SetTranslation(kTargetPosition);
      params.colliderType = ColliderType::Box;
      // Disable normal damping (nonzero by default) to preserve the calibrated contact dynamics.
      params.contact.normalViscousDampingCoefficient = 0_r;
      _targetActor = _scene->CreateRigidActor(params, test::ExpectOK{});

      Real3 const com = _targetActor->GetCenterOfMassTransform(test::ExpectOK{}).GetTranslation();
      std::vector<int> bcDofIndices = {0, 1, 2};
      std::vector<real> bcDofValues = {com[0], com[1], com[2]};
      _targetActor->AddBoundaryConditionDofsWorld(bcDofIndices, bcDofValues, test::ExpectOK{});

      _targetActor->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
      _targetActor->RegisterQuery(QueryType::ContactPoints, test::ExpectOK{});
    }

    // Orbiter cube: soft-constrained via pivot constraint
    {
      Real3 const initialPos = kTargetPosition + Real3{kOrbitRadius, 0_r, 0_r};

      RigidActorParams params;
      params.name = "Orbiter";
      params.shape = cubeShape;
      params.worldFromLocal.SetTranslation(initialPos);
      params.colliderType = ColliderType::Box;
      // Disable normal damping (nonzero by default) to preserve the calibrated contact dynamics.
      params.contact.normalViscousDampingCoefficient = 0_r;
      auto* orbiter = _scene->CreateRigidActor(params, test::ExpectOK{});

      RigidPivotPositionConstraintParams conParams;
      conParams.localPosition = {0_r, 0_r, 0_r};
      conParams.targetPosition = initialPos;
      conParams.actor = orbiter->GetHandle();
      conParams.stiffness = 1e2_r;
      conParams.damping = 1e2_r;
      _orbiterConstraint = _scene->CreateRigidPivotPositionConstraint(conParams, test::ExpectOK{});
    }
  }
};

TEST_F(ContactTorqueScene, Consistency) {
  // NOTE: Time step size has a critical influence on the momentum difference error, because the
  // error is dominated by the integration method. Reducing time step by 10x reduces the error
  // roughly by 10x too. The inertia model (default merit-based vs. Newton-Euler) has no impact.
  real constexpr kDt = 1e-2_r;
  int constexpr kTotalSteps = 200;

  Real3 angularMomentumPrev = {};
  int numStepsWithContact = 0;
  real momentumDifferenceErrorRms = 0;
  real torqueRms = 0;

  // Constant body inertia
  Real6 const m = _targetActor->GetRigidMomentOfInertiaLocal(test::ExpectOK{});
  VMatrix3x3r const inertiaBody = {
      Vec4r{m[0], m[1], m[2]}, Vec4r{m[1], m[3], m[4]}, Vec4r{m[2], m[4], m[5]}};

  // Step the simulation and verify torque consistency whenever contact occurs
  for (int step = 0; step < kTotalSteps; ++step) {
    real const t = static_cast<real>(_scene->GetTotalSimulationTime());

    // Set new orbit position on a pseudo-random spherical trajectory
    real const theta = 0.5_r * (kPI + t) + 0.3_r * Sin(1.7_r * t);
    real const phi = 0.7_r * t + 0.4_r * Sin(2.3_r * t) + 0.2_r * Sin(3.7_r * t);
    Real3 const orbitPosition = kTargetPosition +
        kOrbitRadius * Real3{Sin(theta) * Cos(phi), Sin(theta) * Sin(phi), Cos(theta)};
    _orbiterConstraint->SetTargetPosition(orbitPosition, test::ExpectOK{});

    // Step the scene
    _scene->Step(kDt);

    // Get queries. Skip if there's no contact
    auto const points = _targetActor->GetContactPointsWorld(test::ExpectOK{});
    if (points.empty()) {
      continue;
    }
    Real3 const torque = _targetActor->GetContactTorqueWorld(test::ExpectOK{});

    numStepsWithContact++;

    // Compute torque from contact points
    Real3 const com = _targetActor->GetCenterOfMassTransform(test::ExpectOK{}).GetTranslation();
    Real3 torqueFromContacts = {};
    for (auto const& cp : points) {
      Real3 const force = (cp.actorA == _targetActor->GetHandle()) ? cp.force : -cp.force;
      Real3 const r = cp.posA - com;
      torqueFromContacts += Cross(r, force);
    }

    // Test relative error
    real const torqueNormSqr = NormSqr(torqueFromContacts);
    real const torqueError = Norm(torque - torqueFromContacts) / Sqrt(torqueNormSqr);
    EXPECT_LE(torqueError, 1e-5_r); // Torque query error should be tiny

    // Compute angular momentum
    Real3 const omega = _targetActor->GetAngularVelocity(test::ExpectOK{});
    Quaternion const q = _targetActor->GetRootTransform().GetRotation();
    VMatrix3x3r const inertia = RotateInertia(inertiaBody, q);
    Real3 const angularMomentum = ToReal3(DotMatVec3x3(inertia, ToSimd(omega)));

    // Estimate torque from angular momentum difference
    Real3 const torqueFromAngularMomentum = (angularMomentum - angularMomentumPrev) / kDt;

    // Accumulate torque and error for RMS evaluation
    torqueRms += torqueNormSqr;
    momentumDifferenceErrorRms += NormSqr(torqueFromAngularMomentum - torqueFromContacts);

    // Shift values
    angularMomentumPrev = angularMomentum;
  }

  // Test aggregate values
  EXPECT_GE(numStepsWithContact, kTotalSteps / 10); // Expect at least 10% of the steps in contact
  real const momentumDifferenceError = Sqrt(momentumDifferenceErrorRms / torqueRms);
  EXPECT_LE(
      momentumDifferenceError, 0.1_r); // Momentum difference should be same order of magnitude
}

// Test the MeshCollider. Compare collision-detection result by refitting the MeshCollider vs.
// creating a new one from scratch.
// The collider mesh is not shipped externally.
TEST_IF(MOCHI_INTERNAL, MeshCollider, MeshCollider_Refit) {
  auto* context = CreateContext(0 /*numWorkerThreads*/);
  MOCHI_DEFER(DestroyContext(context));

  auto generator = mochi::RandomGenerator(20);

  // Load a mesh at rest
  ShapeHandle shape = context->LoadShapeFromFile(
      test::GetAssetPath("armadillo/armadillo_coarse_mesh.mochi.json"), ExpectOK{});

  // Extract the loaded shape data
  auto const* contextImpl = assert_cast<ContextImpl const*>(context);
  auto const* shapeArmadilloRest =
      assert_cast<TetrahedralMeshShape const*>(contextImpl->GetShapeSharedPtr(shape).get());
  auto const& meshArmadilloRest = shapeArmadilloRest->GetMesh()->GetBoundaryMesh();

  // Create a MeshCollider at rest
  MeshCollider colliderRest(meshArmadilloRest);
  colliderRest.Initialize(/* allowRefitting */ true);

  // Create another mesh by adding random displacements to the coordinates
  std::vector<Real3> coords = {
      meshArmadilloRest->GetNodeCoordinates().begin(),
      meshArmadilloRest->GetNodeCoordinates().end()};
  std::vector<real> displacements(3 * coords.size());
  SetRandom(generator, -1_r, 1_r, MakeSpan(displacements));
  auto displacements3 = Unflatten<Real3 const>(displacements);
  for (int i = 0; i < coords.size(); i++) {
    coords[i] += displacements3[i];
  }
  auto meshArmadilloDeformed =
      std::make_shared<TriangularMesh>(coords, meshArmadilloRest->GetElementConnectivity());

  // Refit the original MeshCollider
  colliderRest.Refit(AsConstView(displacements), ErrorAssert{});

  // Create another MeshCollider
  MeshCollider colliderDeformed(meshArmadilloDeformed);
  colliderDeformed.Initialize();

  // Create test points
  std::vector<Real3> points(1000);
  SetRandom(generator, -1_r, 1_r, MakeSpan(points));

  // Query and compare
  ContactDetectionResult resultA;
  ContactDetectionResult resultB;
  FindPointContactsT(
      points,
      &colliderRest,
      {},
      {},
      resultA.sampleIndices,
      resultA.posColliding,
      resultA.sdfInfo,
      resultA.isSdfGradUnitary);
  FindPointContactsT(
      points,
      &colliderDeformed,
      {},
      {},
      resultB.sampleIndices,
      resultB.posColliding,
      resultB.sdfInfo,
      resultB.isSdfGradUnitary);
  EXPECT_EQ(resultA.sampleIndices.size(), resultB.sampleIndices.size());
  if (resultA.sampleIndices.size() == resultB.sampleIndices.size()) {
    for (int i = 0; i < resultA.sampleIndices.size(); ++i) {
      EXPECT_EQ(resultA.sampleIndices[i], resultB.sampleIndices[i]);
      EXPECT_NEAR_EQ(resultA.sdfInfo.val[i], resultB.sdfInfo.val[i]);
    }
  }
}
