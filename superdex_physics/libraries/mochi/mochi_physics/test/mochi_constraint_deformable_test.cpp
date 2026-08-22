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

using namespace mochi;
using namespace mochi::test;
using namespace mochi::constraint_test;

/********************************************************************************
  DeformableNodeToDeformableNodeConstraint
********************************************************************************/
namespace {

struct DeformableNodeToDeformableNodeData {
  static bool constexpr kHasDifferentiableTarget = false;
  static bool constexpr kHasMixedLinks = false;
  TimeStepPair<Real3> dispA = {};
  TimeStepPair<Real3> dispB = {};

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    SetSoftState<kTimeStep>(reg, entities[0], dispA[kTimeStep]);
    SetSoftState<kTimeStep>(reg, entities[1], dispB[kTimeStep]);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    AddToSoftState<kTimeStep>(reg, entities[idxi / 3], idxi % 3, di);
    AddToSoftState<kTimeStep>(reg, entities[idxj / 3], idxj % 3, dj);
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
    DeformableNodeToDeformableNodeConstraintParams conParams;
    conParams.nodeIndexA = 0;
    conParams.nodeIndexB = 0;
    conParams.actorA = GetActorHandle(entities[0], scene->GetHandle());
    conParams.actorB = GetActorHandle(entities[1], scene->GetHandle());
    conParams.findClosest = false;
    conParams.stiffness = stiffness;
    conParams.damping = damping;
    conParams.saturation = saturation;
    auto const* constraint =
        scene->CreateDeformableNodeToDeformableNodeConstraint(conParams, ExpectOK{});
    Real3 restA = reg.get<CTetrahedralMesh const>(entities[0]).mesh->GetNodeCoordinates()[0];
    Real3 restB = reg.get<CTetrahedralMesh const>(entities[1]).mesh->GetNodeCoordinates()[0];

    auto& txA = reg.get<CRootTransform>(entities[0]).worldFromLocal;
    Quaternion qA = Quaternion::FromAxisAngle(Normalize(Real3{-1_r, 3_r, -1_r}), kPI * 0.3);
    Real3 tA = {0.5_r, -0.5_r, 0.3_r};
    txA = TransformRT(qA, tA);
    auto& txB = reg.get<CRootTransform>(entities[1]).worldFromLocal;
    Quaternion qB = Quaternion::FromAxisAngle(Normalize(Real3{-2_r, -1_r, 1_r}), kPI * 0.3);
    Real3 tB = {0.1_r, -0.2_r, -0.3_r};
    txB = TransformRT(qB, tB);

    dispA[TimeStep::Current] = {0.2_r, 0.4_r, 0.6_r};
    Real3 wA = txA.TransformPoint(restA + dispA[TimeStep::Current]);
    dispB[TimeStep::Current] = txB.TransformPointInverse(wA - cVal) - restB;
    InitState<TimeStep::Current>(reg, entities);

    dispA[TimeStep::StageStart] = dispA[TimeStep::Current];
    dispB[TimeStep::StageStart] = txB.TransformPointInverse(wA - cVal + dtStage * dCVal) - restB;
    InitState<TimeStep::StageStart>(reg, entities);

    return constraint->GetHandle();
  }
};

class ConstraintDeformableNodeToDeformableNode
    : public ConstraintTestBaseT<DeformableNodeToDeformableNodeData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 6;
    _targetSize = 0;

    SoftActorParams params;
    params.shape = GetUnitCubeShape(_scene->GetContext());
    auto* actorA = _scene->CreateSoftActor(params, ExpectOK{});
    auto* actorB = _scene->CreateSoftActor(params, ExpectOK{});
    _entitiesActors.push_back(GetEntity(actorA));
    _entitiesActors.push_back(GetEntity(actorB));

    _isLinear = true;

    InitBase();
  }
};

} // namespace

TEST_F(ConstraintDeformableNodeToDeformableNode, Test) {
  RunAllTests();
}

/********************************************************************************
  DeformableNodeToRigidConstraint
********************************************************************************/
namespace {

struct DeformableNodeToRigidCouplingData {
  static bool constexpr kHasDifferentiableTarget = false;
  static bool constexpr kHasMixedLinks = false;
  TimeStepPair<Real3> disp = {};
  TimeStepPair<Quaternion> r = {};
  TimeStepPair<Real3> t = {};

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    SetRigidState<kTimeStep>(reg, entities[0], t[kTimeStep], r[kTimeStep]);
    SetSoftState<kTimeStep>(reg, entities[1], disp[kTimeStep]);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    auto addToState = [&](int i, real eps) {
      if (i < 6) {
        AddToRigidState<kTimeStep>(reg, entities[0], i, eps);
      } else {
        AddToSoftState<kTimeStep>(reg, entities[1], i - 6, eps);
      }
    };
    addToState(idxi, di);
    addToState(idxj, dj);
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
    DeformableNodeToRigidConstraintParams conParams;
    conParams.rigidLocalPos = {1_r, 2_r, 3_r};
    conParams.rigidActor = GetActorHandle(entities[0], scene->GetHandle());
    conParams.deformableActor = GetActorHandle(entities[1], scene->GetHandle());
    conParams.deformableNodeIndex = 0;
    conParams.findClosest = false;
    conParams.fixToDeformablePos = false;
    conParams.stiffness = stiffness;
    conParams.damping = damping;
    conParams.saturation = saturation;
    auto const* constraint = scene->CreateDeformableNodeToRigidConstraint(conParams, ExpectOK{});

    r[TimeStep::Current] = Quaternion::FromAxisAngle(Normalize(Real3{1_r, -2_r, -5_r}), kPI * 0.2);
    t[TimeStep::Current] = Real3{0.3_r, -0.2_r, 0.7_r};
    auto& softTx = reg.get<CRootTransform>(entities[1]).worldFromLocal;
    Quaternion qs = Quaternion::FromAxisAngle(Normalize(Real3{-2_r, -1_r, 1_r}), kPI * 0.3);
    Real3 ts = {0.1_r, -0.5_r, 0.3_r};
    softTx = TransformRT(qs, ts);

    Real3 rbLocalCom =
        ToReal3(reg.get<CRigidBodyInertia const>(entities[0]).GetCenterOfMassLocal());
    Real3 r_local = conParams.rigidLocalPos - rbLocalCom;
    Real3 rworld = TransformRT(r[TimeStep::Current], t[TimeStep::Current]).TransformPoint(r_local);
    Real3 pS = rworld - cVal;
    Real3 softRest = reg.get<CTetrahedralMesh const>(entities[1]).mesh->GetNodeCoordinates()[0];
    disp[TimeStep::Current] = softTx.TransformPointInverse(pS) - softRest;

    InitState<TimeStep::Current>(reg, entities);

    Real3 pR0 = rworld - 0.5_r * dCVal * dtStage;
    Real3 pS0 = pS + 0.5_r * dCVal * dtStage;
    r[TimeStep::StageStart] = Quaternion::FromAxisAngle(Real3{3_r, -2_r, -1_r}, kPI * 0.1);
    t[TimeStep::StageStart] =
        pR0 - TransformRT(r[TimeStep::StageStart], Real3{}).TransformPoint(r_local);

    disp[TimeStep::StageStart] = softTx.TransformPointInverse(pS0) - softRest;
    InitState<TimeStep::StageStart>(reg, entities);

    return constraint->GetHandle();
  }
};

class ConstraintDeformableNodeToRigidCoupling
    : public ConstraintTestBaseT<DeformableNodeToRigidCouplingData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 9;
    _targetSize = 0;

    RigidActorParams paramsRigid;
    paramsRigid.shape = GetUnitCubeShape(_scene->GetContext());
    paramsRigid.colliderType = ColliderType::None;
    auto* actorRigid = _scene->CreateRigidActor(paramsRigid, ExpectOK{});
    _entitiesActors.push_back(GetEntity(actorRigid));
    SoftActorParams paramsSoft;
    paramsSoft.shape = paramsRigid.shape;
    auto* actorSoft = _scene->CreateSoftActor(paramsSoft, ExpectOK{});
    _entitiesActors.push_back(GetEntity(actorSoft));

    InitBase();
  }
};

} // namespace

TEST_F(ConstraintDeformableNodeToRigidCoupling, Test) {
  RunAllTests();
}

/********************************************************************************
  DeformableNodePositionConstraint
********************************************************************************/
namespace {

struct DeformableNodePositionData {
  static bool constexpr kHasDifferentiableTarget = true;
  static bool constexpr kHasMixedLinks = false;
  TimeStepPair<Real3> disp = {};
  TimeStepPair<Real3> target = {};

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    SetSoftState<kTimeStep>(reg, entities[0], disp[kTimeStep]);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    AddToSoftState<kTimeStep>(reg, entities[0], idxi, di);
    AddToSoftState<kTimeStep>(reg, entities[0], idxj, dj);
  }

  template <TimeStep kTimeStep>
  void InitTarget(entt::registry& reg, entt::entity e) {
    SetPositionTarget<kTimeStep>(reg, e, target[kTimeStep]);
  }

  template <TimeStep kTimeStep>
  void AddToTarget(entt::registry& reg, entt::entity e, int idx, real d) {
    AddToPositionTarget<kTimeStep>(reg, e, idx, d);
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
    target[TimeStep::Current] = Real3{1_r, 2_r, 3_r};
    target[TimeStep::StageStart] = target[TimeStep::Current];

    DeformableNodePositionConstraintParams conParams;
    conParams.nodeIndex = 0;
    conParams.actor = GetActorHandle(entities[0], scene->GetHandle());
    conParams.position = target[TimeStep::Current];
    conParams.stiffness = stiffness;
    conParams.damping = damping;
    conParams.saturation = saturation;
    auto const* constraint = scene->CreateDeformableNodePositionConstraint(conParams, ExpectOK{});

    // Set actor targets
    auto& info = reg.get<CConstraintInfo>(GetEntity(reg, constraint->GetHandle(), ExpectOK{}));
    info.actorTargets = {{0, 1, 2}};

    auto& tx = reg.get<CRootTransform>(entities[0]).worldFromLocal;
    Quaternion q = Quaternion::FromAxisAngle(Normalize(Real3{-1_r, 3_r, -1_r}), kPI * 0.3);
    Real3 t = {0.5_r, -0.5_r, 0.3_r};
    tx = TransformRT(q, t);

    Real3 rest = reg.get<CTetrahedralMesh const>(entities[0]).mesh->GetNodeCoordinates()[0];
    disp[TimeStep::Current] = tx.TransformPointInverse(cVal + conParams.position) - rest;
    InitState<TimeStep::Current>(reg, entities);

    disp[TimeStep::StageStart] =
        tx.TransformPointInverse(cVal - dtStage * dCVal + conParams.position) - rest;
    InitState<TimeStep::StageStart>(reg, entities);

    return constraint->GetHandle();
  }
};

class ConstraintDeformableNodePosition : public ConstraintTestBaseT<DeformableNodePositionData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 3;
    _targetSize = 3;

    SoftActorParams params;
    params.shape = GetUnitCubeShape(_scene->GetContext());
    auto* actor = _scene->CreateSoftActor(params, ExpectOK{});
    auto actorEntity = GetEntity(actor);
    _entitiesActors.push_back(actorEntity);
    auto& reg = GetRegistry();
    reg.emplace<CDiffInputOffset>(actorEntity);
    reg.emplace<CActorDiffInputInfo>(actorEntity);

    _isLinear = true;

    InitBase();
  }
};

} // namespace

TEST_F(ConstraintDeformableNodePosition, Test) {
  RunAllTests();
}
