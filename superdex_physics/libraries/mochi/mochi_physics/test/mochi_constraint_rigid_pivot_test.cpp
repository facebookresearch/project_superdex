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
  RigidPivotPositionConstraint
********************************************************************************/
namespace {

struct RigidPivotPositionData {
  static bool constexpr kHasDifferentiableTarget = true;
  static bool constexpr kHasMixedLinks = false;
  TimeStepPair<Real3> t = {};
  TimeStepPair<Quaternion> r = {};
  TimeStepPair<Real3> target = {};

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    SetRigidState<kTimeStep>(reg, entities[0], t[kTimeStep], r[kTimeStep]);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    AddToRigidState<kTimeStep>(reg, entities[0], idxi, di);
    AddToRigidState<kTimeStep>(reg, entities[0], idxj, dj);
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
    target[TimeStep::Current] = {1_r, 2_r, 3_r};
    target[TimeStep::StageStart] = target[TimeStep::Current];
    Real3 localPosition = {0.5_r, 0.7_r, -0.1_r};

    RigidPivotPositionConstraintParams params;
    params.targetPosition = target[TimeStep::Current];
    params.localPosition = localPosition;
    params.actor = GetActorHandle(entities[0], scene->GetHandle());
    params.stiffness = stiffness;
    params.damping = damping;
    params.saturation = saturation;
    auto const* constraint = scene->CreateRigidPivotPositionConstraint(params, ExpectOK{});

    // Set actor targets
    auto& info = reg.get<CConstraintInfo>(GetEntity(reg, constraint->GetHandle(), ExpectOK{}));
    info.actorTargets = {{0, 1, 2}};

    Real3 rlocal = localPosition -
        ToReal3(reg.get<CRigidBodyInertia const>(entities[0]).GetCenterOfMassLocal());
    r[TimeStep::Current] = Quaternion::FromAxisAngle(Normalize(Real3{1_r, 2_r, 3_r}), kPI * 0.3);
    t[TimeStep::Current] = cVal -
        TransformRT(r[TimeStep::Current], Real3{}).TransformPoint(rlocal) +
        target[TimeStep::Current];
    InitState<TimeStep::Current>(reg, entities);

    Real3 p = TransformRT(r[TimeStep::Current], t[TimeStep::Current]).TransformPoint(rlocal);
    Real3 p0 = p - dCVal * dtStage;
    r[TimeStep::StageStart] =
        Quaternion::FromAxisAngle(Normalize(Real3{3_r, -2_r, -1_r}), kPI * 0.1);
    t[TimeStep::StageStart] =
        p0 - TransformRT(r[TimeStep::StageStart], Real3{}).TransformPoint(rlocal);
    InitState<TimeStep::StageStart>(reg, entities);

    return constraint->GetHandle();
  }
};

class ConstraintRigidPivotPosition : public ConstraintTestBaseT<RigidPivotPositionData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 6;
    _targetSize = 3;

    RigidActorParams params;
    params.shape = GetUnitCubeShape(_scene->GetContext());
    params.colliderType = ColliderType::None;
    auto* actor = _scene->CreateRigidActor(params, ExpectOK{});
    _entitiesActors.push_back(GetEntity(actor));

    InitBase();
  }
};

} // namespace

TEST_F(ConstraintRigidPivotPosition, Test) {
  RunAllTests();
}

/********************************************************************************
  RigidPivotToRigidTargetConstraint
********************************************************************************/
namespace {

struct RigidPivotToRigidTargetData {
  static bool constexpr kHasDifferentiableTarget = true;
  static bool constexpr kHasMixedLinks = false;
  TimeStepPair<Real3> t = {};
  TimeStepPair<Quaternion> r = {};
  TimeStepPair<Real3> tTarget = {};
  TimeStepPair<Quaternion> rTarget = {};

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    SetRigidState<kTimeStep>(reg, entities[0], t[kTimeStep], r[kTimeStep]);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    AddToRigidState<kTimeStep>(reg, entities[0], idxi, di);
    AddToRigidState<kTimeStep>(reg, entities[0], idxj, dj);
  }

  template <TimeStep kTimeStep>
  void InitTarget(entt::registry& reg, entt::entity e) {
    SetTransformTarget<kTimeStep>(reg, e, tTarget[kTimeStep], rTarget[kTimeStep]);
  }

  template <TimeStep kTimeStep>
  void AddToTarget(entt::registry& reg, entt::entity e, int idx, real d) {
    AddToTransformTarget<kTimeStep>(reg, e, idx, d);
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
    tTarget[TimeStep::Current] = {0.1_r, -0.2_r, -0.3_r};
    rTarget[TimeStep::Current] = Quaternion::FromRotationVector(Real3{0.3_r, 0.1_r, -0.2_r});
    tTarget[TimeStep::StageStart] = tTarget[TimeStep::Current];
    rTarget[TimeStep::StageStart] = rTarget[TimeStep::Current];

    Real3 localPosition = {0.5_r, 0.7_r, -0.1_r};

    RigidPivotToRigidTargetConstraintParams params;
    params.targetTransform = TransformRT(rTarget[TimeStep::Current], tTarget[TimeStep::Current]);
    params.localPosition = localPosition;
    params.actor = GetActorHandle(entities[0], scene->GetHandle());
    params.stiffness = stiffness;
    params.damping = damping;
    params.saturation = saturation;
    auto const* constraint = scene->CreateRigidPivotToRigidTargetConstraint(params, ExpectOK{});

    Real3 rlocal = localPosition -
        ToReal3(reg.get<CRigidBodyInertia const>(entities[0]).GetCenterOfMassLocal());
    r[TimeStep::Current] = Quaternion::FromAxisAngle(Normalize(Real3{1_r, 2_r, 3_r}), kPI * 0.3);
    t[TimeStep::Current] = cVal -
        TransformRT(r[TimeStep::Current], Real3{}).TransformPoint(rlocal) +
        TransformRT(rTarget[TimeStep::Current], tTarget[TimeStep::Current]).TransformPoint(rlocal);
    InitState<TimeStep::Current>(reg, entities);

    Real3 p = TransformRT(r[TimeStep::Current], t[TimeStep::Current]).TransformPoint(rlocal);
    Real3 p0 = p - dCVal * dtStage;
    r[TimeStep::StageStart] =
        Quaternion::FromAxisAngle(Normalize(Real3{3_r, -2_r, -1_r}), kPI * 0.1);
    t[TimeStep::StageStart] =
        p0 - TransformRT(r[TimeStep::StageStart], Real3{}).TransformPoint(rlocal);
    InitState<TimeStep::StageStart>(reg, entities);

    return constraint->GetHandle();
  }
};

class ConstraintRigidPivotToRigidTarget : public ConstraintTestBaseT<RigidPivotToRigidTargetData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 6;
    _targetSize = 6;

    RigidActorParams params;
    params.shape = GetUnitCubeShape(_scene->GetContext());
    params.colliderType = ColliderType::None;
    auto* actor = _scene->CreateRigidActor(params, ExpectOK{});
    _entitiesActors.push_back(GetEntity(actor));

    InitBase();
  }
};

} // namespace

TEST_F(ConstraintRigidPivotToRigidTarget, Test) {
  RunAllTests();
}

/********************************************************************************
  RigidPivotRotationConstraint
********************************************************************************/
namespace {

struct RigidPivotRotationData {
  static bool constexpr kHasDifferentiableTarget = true;
  static bool constexpr kHasMixedLinks = false;
  TimeStepPair<Quaternion> r = {};
  TimeStepPair<Quaternion> target = {};

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    SetRigidRotation<kTimeStep>(reg, entities[0], r[kTimeStep]);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    AddToRigidState<kTimeStep>(reg, entities[0], idxi + 3, di);
    AddToRigidState<kTimeStep>(reg, entities[0], idxj + 3, dj);
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
    Quaternion qlocal = Quaternion::FromAxisAngle(Normalize(Real3{1_r, 2_r, 3_r}), kPI * 0.3);
    target[TimeStep::Current] =
        Quaternion::FromAxisAngle(Normalize(Real3{2_r, 1_r, 5_r}), kPI * 0.2);
    target[TimeStep::StageStart] = target[TimeStep::Current];

    RigidPivotRotationConstraintParams params;
    params.targetRotation = target[TimeStep::Current].ToRotationVector();
    params.localRotation = qlocal.ToRotationVector();
    params.actor = GetActorHandle(entities[0], scene->GetHandle());
    params.stiffness = stiffness;
    params.damping = damping;
    params.saturation = saturation;
    auto const* constraint = scene->CreateRigidPivotRotationConstraint(params, ExpectOK{});

    Real3 x_com{0.2_r, -0.1_r, 0.1_r};
    Quaternion qd = Quaternion::FromRotationVector(cVal);
    r[TimeStep::Current] = Normalize(qd * target[TimeStep::Current] * qlocal.GetConjugate());
    SetRigidState<TimeStep::Current>(reg, entities[0], x_com, r[TimeStep::Current]);

    Real3 C_old = cVal - dtStage * dCVal;
    Quaternion qd_old = Quaternion::FromRotationVector(C_old);
    r[TimeStep::StageStart] = Normalize(qd_old * target[TimeStep::Current] * qlocal.GetConjugate());
    SetRigidState<TimeStep::StageStart>(reg, entities[0], x_com, r[TimeStep::StageStart]);

    return constraint->GetHandle();
  }
};

class ConstraintRigidPivotRotation : public ConstraintTestBaseT<RigidPivotRotationData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 3;
    _targetSize = 3;

    RigidActorParams params;
    params.shape = GetUnitCubeShape(_scene->GetContext());
    params.colliderType = ColliderType::None;
    auto* actor = _scene->CreateRigidActor(params, ExpectOK{});
    _entitiesActors.push_back(GetEntity(actor));

    InitBase();
  }
};

} // namespace

TEST_F(ConstraintRigidPivotRotation, Test) {
  RunAllTests();
}
