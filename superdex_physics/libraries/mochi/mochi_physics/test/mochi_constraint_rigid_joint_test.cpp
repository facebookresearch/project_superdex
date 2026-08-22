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
  RigidSphericalJointConstraint
********************************************************************************/
namespace {

struct RigidSphericalJointData {
  static bool constexpr kHasDifferentiableTarget = false;
  static bool constexpr kHasMixedLinks = false;
  TimeStepPair<Real3> tA = {};
  TimeStepPair<Quaternion> rA = {};
  TimeStepPair<Real3> tB = {};
  TimeStepPair<Quaternion> rB = {};

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    SetRigidState<kTimeStep>(reg, entities[0], tA[kTimeStep], rA[kTimeStep]);
    SetRigidState<kTimeStep>(reg, entities[1], tB[kTimeStep], rB[kTimeStep]);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    AddToRigidState<kTimeStep>(reg, entities[idxi / 6], idxi % 6, di);
    AddToRigidState<kTimeStep>(reg, entities[idxj / 6], idxj % 6, dj);
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
    RigidSphericalJointConstraintParams params;
    params.actorA = GetActorHandle(entities[0], scene->GetHandle());
    params.actorB = GetActorHandle(entities[1], scene->GetHandle());
    params.localPosA = {0.2_r, 0.5_r, 0.7_r};
    params.localPosB = {-0.3_r, -0.1_r, -0.4_r};
    params.stiffness = stiffness;
    params.damping = damping;
    params.saturation = saturation;
    auto const* constraint = scene->CreateRigidSphericalJointConstraint(params, ExpectOK{});
    Real3 localComA = ToReal3(reg.get<CRigidBodyInertia const>(entities[0]).GetCenterOfMassLocal());
    Real3 localComB = ToReal3(reg.get<CRigidBodyInertia const>(entities[1]).GetCenterOfMassLocal());

    Real3 rA_local = params.localPosA - localComA;
    Real3 rB_local = params.localPosB - localComB;

    rA[TimeStep::Current] = Quaternion::FromAxisAngle(Normalize(Real3{1_r, 2_r, 3_r}), kPI * 0.3);
    tA[TimeStep::Current] = Real3{1_r, 2_r, 3_r};
    Real3 pA = TransformRT(rA[TimeStep::Current], tA[TimeStep::Current]).TransformPoint(rA_local);
    Real3 pB = pA - cVal;
    rB[TimeStep::Current] = Quaternion::FromAxisAngle(Real3{-2_r, -1_r, -1_r}, kPI * 0.4);
    tB[TimeStep::Current] =
        pB - TransformRT(rB[TimeStep::Current], Real3{}).TransformPoint(rB_local);
    InitState<TimeStep::Current>(reg, entities);

    Real3 pA0 = pA - 0.5_r * dCVal * dtStage;
    Real3 pB0 = pB + 0.5_r * dCVal * dtStage;
    rA[TimeStep::StageStart] = Quaternion::FromAxisAngle(Real3{3_r, -2_r, -1_r}, kPI * 0.1);
    tA[TimeStep::StageStart] =
        pA0 - TransformRT(rA[TimeStep::StageStart], Real3{}).TransformPoint(rA_local);
    rB[TimeStep::StageStart] = Quaternion::FromAxisAngle(Real3{-1_r, 3_r, 2_r}, kPI * 0.2);
    tB[TimeStep::StageStart] =
        pB0 - TransformRT(rB[TimeStep::StageStart], Real3{}).TransformPoint(rB_local);
    InitState<TimeStep::StageStart>(reg, entities);

    return constraint->GetHandle();
  }
};

class ConstraintRigidSphericalJoint : public ConstraintTestBaseT<RigidSphericalJointData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 12;
    _targetSize = 0;

    RigidActorParams params;
    params.shape = GetUnitCubeShape(_scene->GetContext());
    params.colliderType = ColliderType::None;
    auto* actorA = _scene->CreateRigidActor(params, ExpectOK{});
    auto* actorB = _scene->CreateRigidActor(params, ExpectOK{});
    _entitiesActors.push_back(GetEntity(actorA));
    _entitiesActors.push_back(GetEntity(actorB));

    InitBase();
  }
};

} // namespace

TEST_F(ConstraintRigidSphericalJoint, Test) {
  RunAllTests();
}

/********************************************************************************
  RigidPrismaticJointConstraint
********************************************************************************/
namespace {

struct RigidPrismaticJointData {
  static bool constexpr kHasDifferentiableTarget = false;
  static bool constexpr kHasMixedLinks = false;
  TimeStepPair<Real3> tA = {};
  TimeStepPair<Quaternion> rA = {};
  TimeStepPair<Real3> tB = {};

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    SetRigidState<kTimeStep>(reg, entities[0], tA[kTimeStep], rA[kTimeStep]);
    SetRigidTranslation<kTimeStep>(reg, entities[1], tB[kTimeStep]);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    AddToRigidState<kTimeStep>(reg, entities[idxi / 6], idxi % 6, di);
    AddToRigidState<kTimeStep>(reg, entities[idxj / 6], idxj % 6, dj);
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
    TuneValuesForRangeConstraint(cVal[2], dCVal[2]);

    real max = 1_r;
    real min = -1_r;

    Real3 localAxis{3_r, -2_r, 4_r};
    localAxis = Normalize(localAxis);
    Real3 axis = Cross(localAxis, Real3{0_r, 0_r, 1_r});
    real normSqr = NormSqr(axis);
    if (normSqr > 1e-6) {
      real angle = std::acos(Dot(localAxis, Real3{0_r, 0_r, 1_r}));
      axis = axis / std::sqrt(normSqr) * angle;
    }
    Quaternion localFrame = Quaternion::FromRotationVector(axis);
    Real3 tref{0.1_r, -0.2_r, 0.3_r};

    tA[TimeStep::Current] = Real3{2_r, 1_r, 2_r};
    rA[TimeStep::Current] = Quaternion::FromRotationVector(Real3{0.1_r, -0.2_r, 0.1_r});
    tB[TimeStep::Current] =
        rA[TimeStep::Current] * localFrame.GetConjugate() * tref + tA[TimeStep::Current];
    InitState<TimeStep::Current>(reg, entities);
    Quaternion rB = Quaternion::FromRotationVector(Real3{-0.2_r, 0.1_r, -0.1_r});
    SetRigidRotation<TimeStep::Current>(reg, entities[1], rB);

    RigidPrismaticJointConstraintParams params;
    params.freeAxis = rA[TimeStep::Current] * localAxis;
    params.actorA = GetActorHandle(entities[0], scene->GetHandle());
    params.actorB = GetActorHandle(entities[1], scene->GetHandle());
    params.max = max;
    params.min = min;
    params.stiffness = stiffness;
    params.damping = damping;
    params.saturation = saturation;
    auto const* constraint = scene->CreateRigidPrismaticJointConstraint(params, ExpectOK{});

    Real3 disp = cVal;
    disp[2] = disp[2] == 0 ? disp[2] : (max + disp[2]);

    auto relRotation = Quaternion::FromRotationVector(Real3{0.2_r, -0.1_r, 0.15_r});
    rA[TimeStep::Current] = relRotation * rA[TimeStep::Current];
    tB[TimeStep::Current] =
        rA[TimeStep::Current] * localFrame.GetConjugate() * (tref + disp) + tA[TimeStep::Current];
    InitState<TimeStep::Current>(reg, entities);
    rB = relRotation * rB;
    SetRigidRotation<TimeStep::Current>(reg, entities[1], rB);

    tA[TimeStep::StageStart] = tA[TimeStep::Current];
    rA[TimeStep::StageStart] = rA[TimeStep::Current];
    tB[TimeStep::StageStart] =
        rA[TimeStep::Current] * localFrame.GetConjugate() * (tref + disp - dtStage * dCVal) +
        tA[TimeStep::Current];
    InitState<TimeStep::StageStart>(reg, entities);
    SetRigidRotation<TimeStep::StageStart>(reg, entities[1], rB);

    return constraint->GetHandle();
  }
};

class ConstraintRigidPrismaticJoint : public ConstraintTestBaseT<RigidPrismaticJointData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 9;
    _targetSize = 0;

    RigidActorParams params;
    params.shape = GetUnitCubeShape(_scene->GetContext());
    params.colliderType = ColliderType::None;
    auto* actorA = _scene->CreateRigidActor(params, ExpectOK{});
    auto* actorB = _scene->CreateRigidActor(params, ExpectOK{});
    _entitiesActors.push_back(GetEntity(actorA));
    _entitiesActors.push_back(GetEntity(actorB));

    InitBase();
  }
};

} // namespace

TEST_F(ConstraintRigidPrismaticJoint, Test) {
  RunAllTests();
}

TEST_F(ConstraintRigidPrismaticJoint, RejectsZeroFreeAxis) {
  RigidPrismaticJointConstraintParams params;
  params.actorA = GetActorHandle(_entitiesActors[0], _scene->GetHandle());
  params.actorB = GetActorHandle(_entitiesActors[1], _scene->GetHandle());

  params.freeAxis = {};
  _scene->CreateRigidPrismaticJointConstraint(params, ExpectNotOK{});

  params.freeAxis = {0.5_r * kDefaultNearEqualEpsilon<real>, 0_r, 0_r};
  _scene->CreateRigidPrismaticJointConstraint(params, ExpectNotOK{});
}
