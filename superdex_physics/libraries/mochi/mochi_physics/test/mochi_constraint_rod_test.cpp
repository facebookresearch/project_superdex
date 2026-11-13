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

#include "mochi_constraint_test.h"

#include <mochi_core/element_operations/fem_rod.h>
#include <mochi_physics/src/mochi_rod_pose.h>

#include <array>
#include <limits>
#include <utility>

using namespace mochi;
using namespace mochi::test;
using namespace mochi::constraint_test;

/********************************************************************************
  Helpers shared by the rod element rotation constraint tests
********************************************************************************/
namespace {

// Resets a rod to its natural configuration: zero displacements and reference frame axes
template <TimeStep kTimeStep>
void ResetRodPose(entt::registry& reg, entt::entity rod) {
  auto& rodPose = reg.get<CRodPose<kTimeStep>>(rod);
  rodPose.value.displacements.SetZero();
  auto const& refAxes = reg.get<CReferenceElementFrameAxes const>(rod);
  for (int i = 0; i < isize(rodPose.value.frameAxes); ++i) {
    rodPose.value.frameAxes[i] = refAxes.axes[i];
  }
}

// Perturbs up to two DoFs of a rod element, given as element-local DoF indices in [0, 8), keeping
// displacements and frame axes consistent. Negative indices are ignored.
template <TimeStep kTimeStep>
void AddToRodElementState(
    entt::registry& reg,
    entt::entity rod,
    int element,
    int idxi,
    real di,
    int idxj,
    real dj) {
  auto& rodPose = reg.get<CRodPose<kTimeStep>>(rod);
  auto const& mesh = reg.get<CPolylineMesh const>(rod);
  int const numDofs = isize(rodPose.value.displacements);
  Int2 const elementNodes = mesh.ElementNodes(element);

  ColumnVector<real> dofDelta = ColumnVector<real>::Zero(numDofs);
  bool hasDelta = false;
  for (auto const& [idx, delta] : {std::pair{idxi, di}, std::pair{idxj, dj}}) {
    if (idx >= 0 && delta != 0_r) {
      int const node = elementNodes[idx / fem::kNumRodFields];
      dofDelta[fem::kNumRodFields * node + idx % fem::kNumRodFields] += delta;
      hasDelta = true;
    }
  }
  if (!hasDelta) {
    return;
  }

  ColumnVector<real> outDisp(numDofs);
  DynamicArray<Real3> outAxes(isize(rodPose.value.frameAxes));
  rod::ApplyLieDeltaToPose(
      mesh.nodes,
      rodPose.value.displacements,
      MakeConstSpan(rodPose.value.frameAxes),
      dofDelta,
      outDisp,
      MakeSpan(outAxes));
  rodPose.value.displacements = outDisp;
  rodPose.value.frameAxes = outAxes;
}

// Computes the current world rotation of a rod element
Quaternion ComputeRodElementRotation(entt::registry const& reg, entt::entity rod, int element) {
  auto const& mesh = reg.get<CPolylineMesh const>(rod);
  auto const& pose = reg.get<CRodPose<TimeStep::Current> const>(rod).value;
  Int2 const en = mesh.ElementNodes(element);
  NdArray<real, 2 * fem::kNumRodFields> dofs;
  for (int n = 0; n < 2; ++n) {
    for (int f = 0; f < fem::kNumRodFields; ++f) {
      dofs[fem::kNumRodFields * n + f] = pose.displacements[fem::kNumRodFields * en[n] + f];
    }
  }
  VMatrix3x3r rotLocalMat MOCHI_NO_INIT;
  fem::ComputeRodElementRotationLocal(
      ToSimd(mesh.nodes[en[0]]),
      ToSimd(mesh.nodes[en[1]]),
      Normalize<3>(ToSimd(pose.frameAxes[element])),
      MakeConstSpan(dofs),
      rotLocalMat,
      nullptr);
  return reg.get<CRootTransform const>(rod).worldFromLocal.GetRotation() *
      Normalize(QuaternionFromMatrix(rotLocalMat));
}

/********************************************************************************
  RodElementRotationToRigidConstraint
********************************************************************************/
struct RodElementRotationToRigidData {
  static bool constexpr kHasDifferentiableTarget = true;
  static bool constexpr kHasMixedLinks = false;
  static bool constexpr kHasRigidActor = true;
  static int constexpr kNumDofs = 11; // 3 rigid rotation DoFs + 8 rod element DoFs
  TimeStepPair<Quaternion> r = {};
  TimeStepPair<Quaternion> target = {};
  int constrainedElement = -1;

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    SetRigidRotation<kTimeStep>(reg, entities[0], r[kTimeStep]);
    ResetRodPose<kTimeStep>(reg, entities[1]);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    // Indices 0-2 are rigid rotation DoFs, and 3-10 are rod element DoFs. Returns the rod element
    // DoF index, or -1 for rigid DoFs.
    auto addToRigid = [&](int idx, real delta) {
      if (idx < 3) {
        AddToRigidState<kTimeStep>(reg, entities[0], idx + 3, delta);
        return -1;
      }
      return idx - 3;
    };
    int const rodIdxI = addToRigid(idxi, di);
    int const rodIdxJ = addToRigid(idxj, dj);
    AddToRodElementState<kTimeStep>(reg, entities[1], constrainedElement, rodIdxI, di, rodIdxJ, dj);
  }

  template <TimeStep kTimeStep>
  void InitTarget(entt::registry& reg, entt::entity e) {
    SetRotationTarget<kTimeStep>(reg, e, target[kTimeStep]);
  }

  template <TimeStep kTimeStep>
  void AddToTarget(entt::registry& reg, entt::entity e, int idx, real d) {
    AddToRotationTarget<kTimeStep>(reg, e, idx, d);
  }

  ConstraintHandle InitConstraint(
      entt::registry& reg,
      Scene* scene,
      Span<entt::entity const> entities,
      real stiffness,
      real damping,
      real saturation,
      real dtStage,
      Real3& cVal,
      Real3& dCVal) {
    entt::entity const rigidEntity = entities[0];
    entt::entity const rodEntity = entities[1];
    ResetRodPose<TimeStep::Current>(reg, rodEntity);

    // Create the constraint with an identity rigid rotation, so the stored target rotation is the
    // rod element rotation
    r[TimeStep::Current] = Quaternion::Identity();
    SetRigidRotation<TimeStep::Current>(reg, rigidEntity, r[TimeStep::Current]);
    RodElementRotationToRigidConstraintParams conParams;
    conParams.rigidActor = GetActorHandle(rigidEntity, scene->GetHandle());
    conParams.rodActor = GetActorHandle(rodEntity, scene->GetHandle());
    conParams.elementIndex = constrainedElement;
    conParams.refFrameRotVec = Real3{};
    conParams.stiffness = stiffness;
    conParams.damping = damping;
    conParams.saturation = saturation;
    auto const* constraint =
        scene->CreateRodElementRotationToRigidConstraint(conParams, ExpectOK{});
    auto const constraintEntity = GetEntity(reg, constraint->GetHandle(), ExpectOK{});
    reg.get<CConstraintInfo>(constraintEntity).actorTargets = {{0, 1, 2}, {}};

    Quaternion const qRod = ComputeRodElementRotation(reg, rodEntity, constrainedElement);
    target[TimeStep::Current] = qRod;
    target[TimeStep::StageStart] = qRod;

    // With an identity reference frame, the constraint value is Log(qd) = Log((qRigid * qr)^-1 *
    // qRod), with qr = qRod. Choose qRigid = qRod * qd^-1 * qr^-1 to achieve cVal at the current
    // time and cVal - dtStage * dCVal at stage start.
    auto rigidRotation = [&](Real3 const& c) {
      return Normalize(
          qRod * Quaternion::FromRotationVector(c).GetConjugate() * qRod.GetConjugate());
    };
    r[TimeStep::Current] = rigidRotation(cVal);
    r[TimeStep::StageStart] = rigidRotation(cVal - dtStage * dCVal);
    SetRigidRotation<TimeStep::Current>(reg, rigidEntity, r[TimeStep::Current]);
    SetRigidRotation<TimeStep::StageStart>(reg, rigidEntity, r[TimeStep::StageStart]);

    return constraint->GetHandle();
  }
};

/********************************************************************************
  RodElementRotationConstraint
********************************************************************************/
struct RodElementRotationData {
  static bool constexpr kHasDifferentiableTarget = true;
  static bool constexpr kHasMixedLinks = false;
  static bool constexpr kHasRigidActor = false;
  static int constexpr kNumDofs = 8; // Rod element DoFs
  TimeStepPair<Quaternion> target = {};
  int constrainedElement = -1;

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    ResetRodPose<kTimeStep>(reg, entities[0]);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    AddToRodElementState<kTimeStep>(reg, entities[0], constrainedElement, idxi, di, idxj, dj);
  }

  template <TimeStep kTimeStep>
  void InitTarget(entt::registry& reg, entt::entity e) {
    SetRotationTarget<kTimeStep>(reg, e, target[kTimeStep]);
  }

  template <TimeStep kTimeStep>
  void AddToTarget(entt::registry& reg, entt::entity e, int idx, real d) {
    AddToRotationTarget<kTimeStep>(reg, e, idx, d);
  }

  ConstraintHandle InitConstraint(
      entt::registry& reg,
      Scene* scene,
      Span<entt::entity const> entities,
      real stiffness,
      real damping,
      real saturation,
      real dtStage,
      Real3& cVal,
      Real3& dCVal) {
    entt::entity const rodEntity = entities[0];
    ResetRodPose<TimeStep::Current>(reg, rodEntity);
    ResetRodPose<TimeStep::StageStart>(reg, rodEntity);

    RodElementRotationConstraintParams conParams;
    conParams.actor = GetActorHandle(rodEntity, scene->GetHandle());
    conParams.elementIndex = constrainedElement;
    conParams.refFrameRotVec = Real3{};
    conParams.stiffness = stiffness;
    conParams.damping = damping;
    conParams.saturation = saturation;
    auto const* constraint = scene->CreateRodElementRotationConstraint(conParams, ExpectOK{});
    auto const constraintEntity = GetEntity(reg, constraint->GetHandle(), ExpectOK{});
    reg.get<CConstraintInfo>(constraintEntity).actorTargets = {{0, 1, 2}};

    // With an identity reference frame, the constraint value is Log(qt^-1 * qRod). Choose the
    // target qt = qRod * qd^-1 to achieve cVal at the current time and cVal - dtStage * dCVal at
    // stage start.
    Quaternion const qRod = ComputeRodElementRotation(reg, rodEntity, constrainedElement);
    auto targetRotation = [&](Real3 const& c) {
      return Normalize(qRod * Quaternion::FromRotationVector(c).GetConjugate());
    };
    target[TimeStep::Current] = targetRotation(cVal);
    target[TimeStep::StageStart] = targetRotation(cVal - dtStage * dCVal);
    SetRotationTarget<TimeStep::Current>(reg, constraintEntity, target[TimeStep::Current]);
    SetRotationTarget<TimeStep::StageStart>(reg, constraintEntity, target[TimeStep::StageStart]);

    return constraint->GetHandle();
  }
};

// Fixture for rod element rotation constraints, on the last element of an open rod with 2 elements
// or on the closing element of a closed-loop rod with 4 elements
template <typename TData, bool kIsClosedLoop>
class ConstraintRodElementRotationTestT : public ConstraintTestBaseT<TData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    this->_conSize = 3;
    this->_supSize = TData::kNumDofs;
    this->_targetSize = 3;

    if constexpr (TData::kHasRigidActor) {
      RigidActorParams rigidParams;
      rigidParams.shape = GetUnitCubeShape(this->_scene->GetContext());
      rigidParams.colliderType = ColliderType::None;
      auto* rigidActor = this->_scene->CreateRigidActor(rigidParams, ExpectOK{});
      this->_entitiesActors.push_back(this->GetEntity(rigidActor));
    }

    DynamicArray<Real3> nodes;
    if constexpr (kIsClosedLoop) {
      real constexpr kSideLength = 0.5_r;
      nodes = {
          Real3{0_r, 0_r, 0_r},
          Real3{kSideLength, 0_r, 0_r},
          Real3{kSideLength, 0_r, kSideLength},
          Real3{0_r, 0_r, kSideLength},
      };
    } else {
      nodes = {Real3{0_r, 0_r, 0_r}, Real3{0.5_r, 0_r, 0_r}, Real3{1_r, 0_r, 0_r}};
    }
    int const numElements = kIsClosedLoop ? isize(nodes) : isize(nodes) - 1;
    DynamicArray<Real3> const frameAxes(numElements, Real3{0_r, 1_r, 0_r});

    RodActorParams rodParams;
    rodParams.shape = experimental::CreatePolylineShape(
        this->_scene->GetContext(), nodes, frameAxes, kIsClosedLoop, ExpectOK{});
    auto* rodActor = this->_scene->CreateRodActor(rodParams, ExpectOK{});
    auto const rodEntity = this->GetEntity(rodActor->GetHandle());
    this->_entitiesActors.push_back(rodEntity);
    auto& reg = this->GetRegistry();
    reg.template emplace<CDiffInputOffset>(rodEntity);
    reg.template emplace<CActorDiffInputInfo>(rodEntity);

    this->_constraintData.constrainedElement = numElements - 1;

    this->InitBase();
  }
};

using ConstraintRodElementRotationToRigid =
    ConstraintRodElementRotationTestT<RodElementRotationToRigidData, false>;
using ConstraintRodElementRotationToRigidClosedLoop =
    ConstraintRodElementRotationTestT<RodElementRotationToRigidData, true>;
using ConstraintRodElementRotation =
    ConstraintRodElementRotationTestT<RodElementRotationData, false>;
using ConstraintRodElementRotationClosedLoop =
    ConstraintRodElementRotationTestT<RodElementRotationData, true>;

} // namespace

TEST_F(ConstraintRodElementRotationToRigid, Test) {
  RunAllTests();
}

TEST_F(ConstraintRodElementRotationToRigidClosedLoop, Test) {
  RunAllTests();
}

TEST_F(ConstraintRodElementRotation, Test) {
  RunAllTests();
}

TEST_F(ConstraintRodElementRotationClosedLoop, Test) {
  RunAllTests();
}

/********************************************************************************
  RodElementRotationConstraint through the public API
********************************************************************************/
class RodElementRotationConstraintTest : public test::MochiSceneTestBase {
 protected:
  static int constexpr kNumElements = 4;

  // Creates a horizontal rod along x, starting at origin, with its first node pinned
  Actor* CreatePinnedRod(Real3 const& origin) {
    DynamicArray<Real3> nodes;
    for (int i = 0; i <= kNumElements; ++i) {
      nodes.push_back(origin + Real3{0.25_r * StaticCast<real>(i), 0_r, 0_r});
    }
    DynamicArray<Real3> const frameAxes(kNumElements, Real3{0_r, 1_r, 0_r});
    RodActorParams rodParams;
    rodParams.shape = experimental::CreatePolylineShape(
        _scene->GetContext(), nodes, frameAxes, /*isClosedLoop=*/false, ExpectOK{});
    Actor* const rod = _scene->CreateRodActor(rodParams, ExpectOK{});

    DeformableNodePositionConstraintParams pinParams;
    pinParams.actor = rod->GetHandle();
    pinParams.nodeIndex = 0;
    pinParams.position = origin;
    pinParams.stiffness = kPinStiffness;
    _scene->CreateDeformableNodePositionConstraint(pinParams, ExpectOK{});
    return rod;
  }

  void StepAndCompareDisplacements(Actor const* rodA, Actor const* rodB) {
    for (int i = 0; i < kNumSteps; ++i) {
      _scene->Step(kDt);
    }
    auto const dispA = test::GetDofValues(*rodA);
    auto const dispB = test::GetDofValues(*rodB);
    ASSERT_EQ(isize(dispA), isize(dispB));
    // Make sure the rods deformed under gravity
    EXPECT_GT(Abs(dispA[fem::kNumRodFields * kNumElements + 1]), 1e-3_r);
    for (int i = 0; i < isize(dispA); ++i) {
      EXPECT_NEAR(dispA[i], dispB[i], 1e-4_r) << "DoF " << i;
    }
  }

  // Compares a rod with a rod element rotation constraint against a rod with a rod-to-rigid
  // rotation constraint to a rigid actor fixed at the given rotation
  void RunMatchesRodToFixedRigid(Quaternion const& rigidRotation) {
    Actor* const rod = CreatePinnedRod(Real3{0_r, 0_r, 0_r});
    RodElementRotationConstraintParams params;
    params.actor = rod->GetHandle();
    params.elementIndex = 0;
    params.refFrameRotVec = Real3{0.1_r, 0.2_r, 0.3_r};
    params.stiffness = kRotStiffness;
    Constraint* const constraint = _scene->CreateRodElementRotationConstraint(params, ExpectOK{});

    Actor* const rodRigid = CreatePinnedRod(Real3{0_r, 0_r, 5_r});
    RigidActorParams rigidParams;
    rigidParams.shape = GetUnitCubeShape(_scene->GetContext());
    rigidParams.colliderType = ColliderType::None;
    rigidParams.worldFromLocal = TransformRT{rigidRotation, Real3{0_r, 0_r, -5_r}};
    Actor* const rigid = _scene->CreateRigidActor(rigidParams, ExpectOK{});
    // Static rigid actors are not supported by RodElementRotationToRigid, so fix all DoFs instead
    std::array<int, 6> const rigidDofIndices = {0, 1, 2, 3, 4, 5};
    std::array<real, 6> rigidDofValues = {};
    rigid->GetDofValues(rigidDofIndices, rigidDofValues, ExpectOK{});
    rigid->AddBoundaryConditionDofsWorld(rigidDofIndices, rigidDofValues, ExpectOK{});
    RodElementRotationToRigidConstraintParams rigidConstraintParams;
    rigidConstraintParams.rigidActor = rigid->GetHandle();
    rigidConstraintParams.rodActor = rodRigid->GetHandle();
    rigidConstraintParams.elementIndex = 0;
    rigidConstraintParams.refFrameRotVec = params.refFrameRotVec;
    rigidConstraintParams.stiffness = kRotStiffness;
    Constraint* const rigidConstraint =
        _scene->CreateRodElementRotationToRigidConstraint(rigidConstraintParams, ExpectOK{});

    StepAndCompareDisplacements(rod, rodRigid);

    // Deviations are only expressed in the same frame if the rigid rotation is the identity
    if (rigidRotation == Quaternion::Identity()) {
      auto const deviation = constraint->GetDeviation();
      auto const deviationRigid = rigidConstraint->GetDeviation();
      EXPECT_GT(Norm(Real3{deviation[0], deviation[1], deviation[2]}), 1e-4_r);
      for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(deviation[i], deviationRigid[i], 1e-4_r);
      }
    }
  }

  static real constexpr kPinStiffness = 1e4_r;
  static real constexpr kRotStiffness = 1e2_r;
  static real constexpr kDt = 0.01_r;
  static int constexpr kNumSteps = 20;
};

// A rod element rotation constraint should behave like a rod-to-rigid rotation constraint with a
// fixed rigid actor
TEST_F(RodElementRotationConstraintTest, MatchesRodToFixedRigid) {
  RunMatchesRodToFixedRigid(Quaternion::Identity());
}

TEST_F(RodElementRotationConstraintTest, MatchesRodToRotatedFixedRigid) {
  RunMatchesRodToFixedRigid(Quaternion::FromAxisAngle(Normalize(Real3{1_r, 2_r, 3_r}), 0.7_r));
}

TEST_F(RodElementRotationConstraintTest, InvalidParams) {
  Actor* const rod = CreatePinnedRod(Real3{0_r, 0_r, 0_r});
  RigidActorParams rigidParams;
  rigidParams.shape = GetUnitCubeShape(_scene->GetContext());
  rigidParams.colliderType = ColliderType::None;
  Actor* const rigid = _scene->CreateRigidActor(rigidParams, ExpectOK{});

  auto expectInvalid = [&](auto&& modify) {
    RodElementRotationConstraintParams params;
    params.actor = rod->GetHandle();
    modify(params);
    EXPECT_EQ(nullptr, _scene->CreateRodElementRotationConstraint(params, ExpectNotOK{}));
  };
  expectInvalid([&](auto& params) { params.actor = rigid->GetHandle(); });
  expectInvalid([](auto& params) { params.elementIndex = -1; });
  expectInvalid([](auto& params) { params.elementIndex = kNumElements; });
  expectInvalid([](auto& params) {
    params.refFrameRotVec = Real3{std::numeric_limits<real>::quiet_NaN(), 0_r, 0_r};
  });
  expectInvalid([](auto& params) { params.stiffness = -1_r; });
}

TEST_F(RodElementRotationConstraintTest, TargetRotation) {
  Actor* const rod = CreatePinnedRod(Real3{0_r, 0_r, 0_r});
  RodElementRotationConstraintParams params;
  params.actor = rod->GetHandle();
  params.elementIndex = 1;
  Constraint* const constraint = _scene->CreateRodElementRotationConstraint(params, ExpectOK{});

  // The target is initialized to the current rotation
  for (real const d : constraint->GetDeviation()) {
    EXPECT_NEAR(0_r, d, 1e-6_r);
  }

  // The element's material frame is aligned with the world axes, so a target rotation by an angle
  // about a world axis yields a deviation of that magnitude
  real constexpr kAngle = 0.3_r;
  constraint->SetTargetRotation(
      Quaternion::FromAxisAngle(Real3{0_r, 0_r, 1_r}, kAngle), ExpectOK{});
  auto const deviation = constraint->GetDeviation();
  EXPECT_NEAR(kAngle, Norm(Real3{deviation[0], deviation[1], deviation[2]}), 1e-5_r);

  // Relative-rotation targets are not supported
  constraint->SetRefRelativeRotation(Quaternion::Identity(), Quaternion::Identity(), ExpectNotOK{});
}

class ContactSkinnedRodConstraintTest : public test::MochiSceneTestBase {};

TEST_F(ContactSkinnedRodConstraintTest, ConstraintUsesCenterlineNodeIndices) {
  using namespace mochi::experimental;

  ModelData model;
  model.mesh.emplace();
  model.mesh->nodesPerElement = 2;
  model.mesh->coordinates = {0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 2_r, 0_r, 0_r, 3_r, 0_r, 0_r};
  model.mesh->connectivity = {0, 1, 1, 2, 2, 3};
  model.elementFrameAxes = DynamicArray<real>{0_r, 1_r, 0_r, 0_r, 1_r, 0_r, 0_r, 1_r, 0_r};
  model.contactSkinMesh.emplace();
  model.contactSkinMesh->nodesPerElement = 3;
  model.contactSkinMesh->coordinates = {0.5_r, 0_r, 0_r, 0.5_r, 0.1_r, 0_r, 0.5_r, 0_r, 0.1_r};
  model.contactSkinMesh->connectivity = {0, 1, 2};
  model.contactSkinMesh->skinning.emplace();
  model.contactSkinMesh->skinning->weightsPerNode = 1;
  model.contactSkinMesh->skinning->indices = {0, 0, 0};
  model.contactSkinMesh->skinning->weights = {1_r, 1_r, 1_r};

  RodActorParams rodParams;
  rodParams.shape = _scene->GetContext()->CreateModelShape(model, ExpectOK{});
  Actor* const rod = _scene->CreateRodActor(rodParams, ExpectOK{});
  ASSERT_EQ(4, rod->GetMesh().GetNumNodes());
  ASSERT_EQ(3, rod->GetContactSkinMesh().GetNumNodes());

  RigidActorParams rigidParams;
  rigidParams.shape = GetUnitCubeShape(_scene->GetContext());
  rigidParams.colliderType = ColliderType::None;
  Actor* const rigid = _scene->CreateRigidActor(rigidParams, ExpectOK{});

  DeformableNodeToRigidConstraintParams params;
  params.rigidActor = rigid->GetHandle();
  params.deformableActor = rod->GetHandle();
  params.deformableNodeIndex = 3;
  params.findClosest = false;
  Constraint* const constraint = _scene->CreateDeformableNodeToRigidConstraint(params, ExpectOK{});

  auto const& info = GetRegistry().get<CConstraintInfo const>(GetEntity(constraint->GetHandle()));
  ASSERT_EQ(2, isize(info.actorDofs));
  EXPECT_EQ((std::vector<int>{12, 13, 14}), info.actorDofs[1]);
}
