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
  JointRotationRangeConstraint
********************************************************************************/
namespace {

struct JointRotationRangeData {
  static bool constexpr kHasDifferentiableTarget = false;
  static bool constexpr kHasMixedLinks = false;
  TimeStepPair<Quaternion> rA = {};
  TimeStepPair<Quaternion> rB = {};

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    SetRigidRotation<kTimeStep>(reg, entities[0], rA[kTimeStep]);
    SetRigidRotation<kTimeStep>(reg, entities[1], rB[kTimeStep]);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    AddToRigidState<kTimeStep>(reg, entities[idxi / 3], (idxi % 3) + 3, di);
    AddToRigidState<kTimeStep>(reg, entities[idxj / 3], (idxj % 3) + 3, dj);
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
    TuneValuesForRangeConstraint(cVal[0], dCVal[0]);
    TuneValuesForRangeConstraint(cVal[1], dCVal[1]);
    TuneValuesForRangeConstraint(cVal[2], dCVal[2]);

    Real3 vmin = {-0.2_r, -0.1_r, -1.2_r};
    Real3 vmax = {0.2_r, 1.4_r, 0.5_r};

    Quaternion q0 = Quaternion::FromAxisAngle(Normalize(Real3{1_r, 1_r, 1_r}), kPI * 0.4);
    Quaternion qr = Quaternion::FromAxisAngle(Normalize(Real3{0.4_r, 0.5_r, 0.1_r}), kPI * 0.1_r);

    rA[TimeStep::Current] = Quaternion::FromAxisAngle(Normalize(Real3{1_r, 2_r, 3_r}), kPI * 0.3_r);
    rB[TimeStep::Current] = Normalize(rA[TimeStep::Current] * q0 * qr * q0.GetConjugate());
    InitState<TimeStep::Current>(reg, entities);

    JointRotationRangeConstraintParams params;
    params.actorA = GetActorHandle(entities[0], scene->GetHandle());
    params.actorB = GetActorHandle(entities[1], scene->GetHandle());
    params.refFrameRotVec = (rA[TimeStep::Current] * q0).ToRotationVector();
    params.angleRangeX = {vmin[0], vmax[0]};
    params.angleRangeY = {vmin[1], vmax[1]};
    params.angleRangeZ = {vmin[2], vmax[2]};
    params.stiffness = stiffness;
    params.damping = damping;
    params.saturation = saturation;
    auto const* constraint = scene->CreateJointRotationRangeConstraint(params, ExpectOK{});

    Real3 rAB = {};
    for (int i = 0; i < 3; ++i) {
      if (std::abs(cVal[i]) < 1e-4_r) {
        rAB[i] = (vmin[i] + vmax[i]) * 0.5_r;
      } else if (std::abs(vmin[i]) <= std::abs(vmax[i])) {
        rAB[i] = vmin[i] - cVal[i];
      } else {
        rAB[i] = cVal[i] + vmax[i];
      }
    }
    Quaternion qAB = Quaternion::FromRotationVector(rAB);
    rB[TimeStep::Current] = Normalize(rA[TimeStep::Current] * q0 * qAB * qr * q0.GetConjugate());
    SetRigidRotation<TimeStep::Current>(reg, entities[1], rB[TimeStep::Current]);

    Real3 C_old = cVal - dtStage * dCVal;
    Real3 rAB_old = {};
    for (int i = 0; i < 3; ++i) {
      if (std::abs(C_old[i]) < 1e-4_r) {
        rAB_old[i] = (vmin[i] + vmax[i]) * 0.5_r;
      } else if (std::abs(vmin[i]) <= std::abs(vmax[i])) {
        rAB_old[i] = vmin[i] - C_old[i];
      } else {
        rAB_old[i] = C_old[i] + vmax[i];
      }
    }
    Quaternion qAB_old = Quaternion::FromRotationVector(rAB_old);
    rA[TimeStep::StageStart] = rA[TimeStep::Current];
    rB[TimeStep::StageStart] =
        Normalize(rA[TimeStep::Current] * q0 * qAB_old * qr * q0.GetConjugate());
    InitState<TimeStep::StageStart>(reg, entities);

    return constraint->GetHandle();
  }
};

class ConstraintJointRotationRange : public ConstraintTestBaseT<JointRotationRangeData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 6;
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

TEST_F(ConstraintJointRotationRange, Test) {
  RunAllTests();
}

/********************************************************************************
  JointRotationTrackingConstraint
********************************************************************************/
namespace {

struct JointRotationTrackingData {
  static bool constexpr kHasDifferentiableTarget = true;
  static bool constexpr kHasMixedLinks = false;
  TimeStepPair<Quaternion> rA = {};
  TimeStepPair<Quaternion> rB = {};
  TimeStepPair<Quaternion> target = {};

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    SetRigidRotation<kTimeStep>(reg, entities[0], rA[kTimeStep]);
    SetRigidRotation<kTimeStep>(reg, entities[1], rB[kTimeStep]);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    AddToRigidState<kTimeStep>(reg, entities[idxi / 3], (idxi % 3) + 3, di);
    AddToRigidState<kTimeStep>(reg, entities[idxj / 3], (idxj % 3) + 3, dj);
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
    Quaternion q0 = Quaternion::FromAxisAngle(Normalize(Real3{1_r, 1_r, 1_r}), kPI * 0.4);
    Quaternion qr = Quaternion::FromAxisAngle(Normalize(Real3{2_r, 3_r, 4_r}), kPI * 0.2);

    rA[TimeStep::Current] =
        Quaternion::FromAxisAngle(Normalize(Real3{1_r, -2_r, 3_r}), kPI * 0.3_r);
    rB[TimeStep::Current] = Normalize(rA[TimeStep::Current] * qr * q0 * q0.GetConjugate());
    InitState<TimeStep::Current>(reg, entities);

    target[TimeStep::Current] =
        Normalize(rA[TimeStep::Current].GetConjugate() * rB[TimeStep::Current]);
    target[TimeStep::StageStart] = target[TimeStep::Current];

    JointRotationTrackingConstraintParams params;
    params.actorA = GetActorHandle(entities[0], scene->GetHandle());
    params.actorB = GetActorHandle(entities[1], scene->GetHandle());
    params.refFrameRotVec = (rA[TimeStep::Current] * q0).ToRotationVector();
    params.stiffness = stiffness;
    params.damping = damping;
    params.saturation = saturation;
    auto const* constraint = scene->CreateJointRotationTrackingConstraint(params, ExpectOK{});

    // Set actor targets
    auto& info = reg.get<CConstraintInfo>(GetEntity(reg, constraint->GetHandle(), ExpectOK{}));
    info.actorTargets = {{0, 1, 2}, {}};

    Quaternion qd = Quaternion::FromRotationVector(cVal);
    rB[TimeStep::Current] = Normalize(rA[TimeStep::Current] * qr * q0 * qd * q0.GetConjugate());
    SetRigidRotation<TimeStep::Current>(reg, entities[1], rB[TimeStep::Current]);

    Real3 C_old = cVal - dtStage * dCVal;
    Quaternion qd_old = Quaternion::FromRotationVector(C_old);

    rA[TimeStep::StageStart] = rA[TimeStep::Current];
    rB[TimeStep::StageStart] =
        Normalize(rA[TimeStep::Current] * qr * q0 * qd_old * q0.GetConjugate());
    InitState<TimeStep::StageStart>(reg, entities);

    return constraint->GetHandle();
  }
};

class ConstraintJointRotationTracking : public ConstraintTestBaseT<JointRotationTrackingData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 6;
    _targetSize = 3;

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

TEST_F(ConstraintJointRotationTracking, Test) {
  RunAllTests();
}

/********************************************************************************
  JointRotationTrackingConstraint - Articulated Link to Isolated Rigid
********************************************************************************/
namespace {

struct JointRotationTrackingLinkToRigidData {
  static bool constexpr kHasDifferentiableTarget = false;
  static bool constexpr kHasMixedLinks = true;
  TimeStepPair<Quaternion> rRigid = {};
  TimeStepPair<ColumnVector<real, 8>> articulatedDofs = {};
  entt::entity link = {};

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    // entities[0] is the articulated actor, entities[1] is the isolated rigid actor
    SetArticulatedStateFromDofs<kTimeStep>(reg, entities[0], articulatedDofs[kTimeStep]);
    SetRigidRotation<kTimeStep>(reg, entities[1], rRigid[kTimeStep]);

    // Initialize the articulated Jacobian
    ecs::InvokeForEachGlobal(&articulated::compound::UpdateJacobianState<kTimeStep>, reg);
    ecs::InvokeForEachGlobal(&articulated::rigid::EntityUpdateJacobian, reg);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    auto addToDof = [&](int idx, real d) {
      if (idx < 3) {
        // First 3 DOFs are articulated link rotation
        AddToRigidState<kTimeStep>(reg, link, idx + 3, d);
      } else {
        // Next 3 DOFs are rigid actor rotation
        AddToRigidState<kTimeStep>(reg, entities[1], idx, d);
      }
    };
    addToDof(idxi, di);
    addToDof(idxj, dj);
  }

  template <TimeStep kTimeStep>
  void AddToStateFull(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    auto addToDof = [&](int idx, real d) {
      if (idx < 6) {
        // First 6 DOFs are articulated state
        AddToArticulatedState<kTimeStep>(reg, entities[0], idx, d);
      } else {
        // Next 3 DOFs are rigid actor rotation
        AddToRigidState<kTimeStep>(reg, entities[1], idx - 3, d);
      }
    };
    addToDof(idxi, di);
    addToDof(idxj, dj);
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
    // entities[0] is the articulated actor, entities[1] is the isolated rigid
    // Get link 1 of the articulated actor
    auto const& articulatedMembers = reg.get<CGroupMembers const>(entities[0]);
    link = articulatedMembers.actors[1];
    ActorHandle link1Handle = GetActorHandle(link, scene->GetHandle());
    ActorHandle rigidHandle = GetActorHandle(entities[1], scene->GetHandle());

    Quaternion q0 = Quaternion::FromAxisAngle(Normalize(Real3{1_r, 1_r, 1_r}), kPI * 0.4);
    Quaternion qr = Quaternion::FromAxisAngle(Normalize(Real3{2_r, 3_r, 4_r}), kPI * 0.2);

    // Initialize state
    articulatedDofs[TimeStep::Current] =
        reg.get<CArticulatedReducedPose<TimeStep::Current> const>(entities[0]).value;
    auto const& linkRot = reg.get<CRigidState<TimeStep::Current> const>(link).value.GetRotation();

    rRigid[TimeStep::Current] = Normalize(linkRot * qr * q0 * q0.GetConjugate());
    SetRigidRotation<TimeStep::Current>(reg, entities[1], rRigid[TimeStep::Current]);

    JointRotationTrackingConstraintParams params;
    params.actorA = link1Handle;
    params.actorB = rigidHandle;
    params.refFrameRotVec = (linkRot * q0).ToRotationVector();
    params.stiffness = stiffness;
    params.damping = damping;
    params.saturation = saturation;
    auto const* constraint = scene->CreateJointRotationTrackingConstraint(params, ErrorAssert{});

    Quaternion qd = Quaternion::FromRotationVector(cVal);
    rRigid[TimeStep::Current] = Normalize(linkRot * qr * q0 * qd * q0.GetConjugate());
    SetRigidRotation<TimeStep::Current>(reg, entities[1], rRigid[TimeStep::Current]);
    InitState<TimeStep::Current>(reg, entities);

    Real3 C_old = cVal - dtStage * dCVal;
    Quaternion qd_old = Quaternion::FromRotationVector(C_old);

    articulatedDofs[TimeStep::StageStart] = articulatedDofs[TimeStep::Current].Duplicate();
    rRigid[TimeStep::StageStart] = Normalize(linkRot * qr * q0 * qd_old * q0.GetConjugate());
    InitState<TimeStep::StageStart>(reg, entities);

    // Initialize dtStage on the link
    reg.get<CTimeIntegratorState>(link).dtStage = dtStage;

    return constraint->GetHandle();
  }
};

class ConstraintJointRotationTrackingLinkToRigid
    : public ConstraintTestBaseT<JointRotationTrackingLinkToRigidData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 6; // 3 articulated DOFs + 3 rigid rotation DOFs

    // Create articulated actor with 2 links
    auto linkShape = GetUnitCubeShape(_scene->GetContext());
    ArticulatedActorParams articulatedParams;
    articulatedParams.joints = {
        {
            .type = ArticulatedJointType::Spherical //
        },
        {
            .type = ArticulatedJointType::Spherical,
            .parentLinkFromJoint = TransformRT{Real3{1_r, 0_r, 0_r}} //
        }};
    articulatedParams.links = {
        {
            .parentLink = -1, //
            .shape = linkShape, //
            .colliderType = ColliderType::None //
        },
        {
            .parentLink = 0,
            .parentJointFromLink = TransformRT{Real3{1_r, 0_r, 0_r}},
            .shape = linkShape,
            .colliderType = ColliderType::None //
        }};
    auto const* articulatedActor = _scene->CreateArticulatedActor(articulatedParams, ExpectOK{});
    _entitiesActors.push_back(GetEntity(articulatedActor->GetHandle()));

    // Create isolated rigid actor
    RigidActorParams rigidParams;
    rigidParams.shape = linkShape;
    rigidParams.colliderType = ColliderType::None;
    rigidParams.worldFromLocal = TransformRT{Quaternion::Identity(), Real3{4_r, 0_r, 0_r}};
    auto* rigidActor = _scene->CreateRigidActor(rigidParams, ExpectOK{});
    _entitiesActors.push_back(GetEntity(rigidActor));

    InitBase();
  }
};

} // namespace

TEST_F(ConstraintJointRotationTrackingLinkToRigid, Test) {
  RunAllTests();
}

/********************************************************************************
  JointRotationTrackingConstraint - Two Links of Same Articulated Actor
  It can reuse JointRotationTrackingData.
********************************************************************************/
namespace {

class ConstraintJointRotationTrackingTwoLinks
    : public ConstraintTestBaseT<JointRotationTrackingData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 6; // 3 DOFs for link1's joint + 3 DOFs for link2's joint
    _targetSize = 3;

    // Create articulated actor with 3 links
    auto linkShape = GetUnitCubeShape(_scene->GetContext());
    TransformRT const offset{Real3{1_r, 0_r, 0_r}};
    ArticulatedActorParams articulatedParams;
    articulatedParams.joints = {
        {.type = ArticulatedJointType::Spherical},
        {.type = ArticulatedJointType::Spherical, .parentLinkFromJoint = offset},
        {.type = ArticulatedJointType::Spherical, .parentLinkFromJoint = offset}};
    articulatedParams.links = {
        {
            .parentLink = -1, //
            .shape = linkShape, //
            .colliderType = ColliderType::None //
        },
        {
            .parentLink = 0,
            .parentJointFromLink = offset,
            .shape = linkShape,
            .colliderType = ColliderType::None //
        },
        {
            .parentLink = 1,
            .parentJointFromLink = offset,
            .shape = linkShape,
            .colliderType = ColliderType::None //
        }};
    auto const* articulatedActor = _scene->CreateArticulatedActor(articulatedParams, ExpectOK{});
    auto& reg = GetRegistry();
    auto const& links =
        reg.get<CGroupMembers const>(GetEntity(articulatedActor->GetHandle())).actors;
    _entitiesActors.push_back(links[1]);
    _entitiesActors.push_back(links[2]);

    reg.emplace<CDiffInputOffset>(links[1]);
    reg.emplace<CActorDiffInputInfo>(links[1]);
    reg.emplace<CDiffInputOffset>(links[2]);
    reg.emplace<CActorDiffInputInfo>(links[2]);

    InitBase();
  }
};

} // namespace

TEST_F(ConstraintJointRotationTrackingTwoLinks, Test) {
  RunAllTests();
}

/********************************************************************************
  JointRotationTrackingConstraint - Links of Different Articulated Actors
  Uses ConstraintTestBaseT framework.
********************************************************************************/
namespace {

struct JointRotationTrackingDifferentActorsData {
  static bool constexpr kHasDifferentiableTarget = false;
  static bool constexpr kHasMixedLinks = true;
  TimeStepPair<ColumnVector<real, 8>> actor1Dofs = {};
  TimeStepPair<ColumnVector<real, 8>> actor2Dofs = {};
  entt::entity link1 = {};
  entt::entity link2 = {};

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    // entities[0] is first articulated actor, entities[1] is second
    SetArticulatedStateFromDofs<kTimeStep>(reg, entities[0], actor1Dofs[kTimeStep]);
    SetArticulatedStateFromDofs<kTimeStep>(reg, entities[1], actor2Dofs[kTimeStep]);

    // Initialize the articulated Jacobians
    ecs::InvokeForEachGlobal(&articulated::compound::UpdateJacobianState<kTimeStep>, reg);
    ecs::InvokeForEachGlobal(&articulated::rigid::EntityUpdateJacobian, reg);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> /*entities*/,
      int idxi,
      real di,
      int idxj,
      real dj) {
    auto addToDof = [&](int idx, real d) {
      if (idx < 3) {
        // First 3 DOFs are link1 rotation
        AddToRigidState<kTimeStep>(reg, link1, idx + 3, d);
      } else {
        // Next 3 DOFs are link2 rotation
        AddToRigidState<kTimeStep>(reg, link2, idx, d);
      }
    };
    addToDof(idxi, di);
    addToDof(idxj, dj);
  }

  template <TimeStep kTimeStep>
  void AddToStateFull(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    AddToArticulatedState<kTimeStep>(reg, entities[idxi / 6], idxi % 6, di);
    AddToArticulatedState<kTimeStep>(reg, entities[idxj / 6], idxj % 6, dj);
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
    // entities[0] is first articulated actor, entities[1] is second
    auto const& members1 = reg.get<CGroupMembers const>(entities[0]);
    auto const& members2 = reg.get<CGroupMembers const>(entities[1]);
    link1 = members1.actors[1]; // Link 1 of actor 1
    link2 = members2.actors[1]; // Link 1 of actor 2

    Quaternion q0 = Quaternion::FromAxisAngle(Normalize(Real3{1_r, 1_r, 1_r}), kPI * 0.4);
    Quaternion qr = Quaternion::FromAxisAngle(Normalize(Real3{2_r, 3_r, 4_r}), kPI * 0.2);

    // Initialize state
    actor1Dofs[TimeStep::Current] =
        reg.get<CArticulatedReducedPose<TimeStep::Current> const>(entities[0]).value;
    actor2Dofs[TimeStep::Current] =
        reg.get<CArticulatedReducedPose<TimeStep::Current> const>(entities[1]).value;
    auto const& link1Rot = reg.get<CRigidState<TimeStep::Current> const>(link1).value.GetRotation();

    // Set link2's rotation to achieve offset qr relative to link1
    auto& pose2 = reg.get<CArticulatedReducedPose<TimeStep::Current>>(entities[1]).value;
    Quaternion link2JointRot = Normalize(qr * q0 * q0.GetConjugate());
    auto link2JointReal4 = link2JointRot.ToReal4();
    for (int i = 0; i < 4; ++i) {
      pose2[4 + i] = link2JointReal4[i];
    }
    actor2Dofs[TimeStep::Current] = pose2.Duplicate();
    articulated::compound::SetArticulatedBodyPose(
        reg, entities[1], pose2.GetConstSpan(), ErrorAssert{});

    JointRotationTrackingConstraintParams params;
    params.actorA = GetActorHandle(link1, scene->GetHandle());
    params.actorB = GetActorHandle(link2, scene->GetHandle());
    params.refFrameRotVec = (link1Rot * q0).ToRotationVector();
    params.stiffness = stiffness;
    params.damping = damping;
    params.saturation = saturation;
    auto const* constraint = scene->CreateJointRotationTrackingConstraint(params, ErrorAssert{});

    // Set link2's rotation to achieve cVal
    Quaternion qd = Quaternion::FromRotationVector(cVal);
    link2JointRot = Normalize(qr * q0 * qd * q0.GetConjugate());
    link2JointReal4 = link2JointRot.ToReal4();
    for (int i = 0; i < 4; ++i) {
      pose2[4 + i] = link2JointReal4[i];
    }
    actor2Dofs[TimeStep::Current] = pose2.Duplicate();
    articulated::compound::SetArticulatedBodyPose(
        reg, entities[1], pose2.GetConstSpan(), ErrorAssert{});
    InitState<TimeStep::Current>(reg, entities);

    Real3 C_old = cVal - dtStage * dCVal;
    Quaternion qd_old = Quaternion::FromRotationVector(C_old);

    actor1Dofs[TimeStep::StageStart] = actor1Dofs[TimeStep::Current].Duplicate();
    Quaternion link2JointRotOld = Normalize(qr * q0 * qd_old * q0.GetConjugate());
    auto link2JointReal4Old = link2JointRotOld.ToReal4();
    actor2Dofs[TimeStep::StageStart] = actor2Dofs[TimeStep::Current].Duplicate();
    for (int i = 0; i < 4; ++i) {
      actor2Dofs[TimeStep::StageStart][4 + i] = link2JointReal4Old[i];
    }
    InitState<TimeStep::StageStart>(reg, entities);

    // Initialize dtStage on the links
    reg.get<CTimeIntegratorState>(link1).dtStage = dtStage;
    reg.get<CTimeIntegratorState>(link2).dtStage = dtStage;

    return constraint->GetHandle();
  }
};

class ConstraintJointRotationTrackingDifferentActors
    : public ConstraintTestBaseT<JointRotationTrackingDifferentActorsData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 6; // 3 DOFs for actor1's link + 3 DOFs for actor2's link

    auto linkShape = GetUnitCubeShape(_scene->GetContext());
    TransformRT const offset{Real3{1_r, 0_r, 0_r}};

    // Create first articulated actor with 2 links
    ArticulatedActorParams articulatedParams1;
    articulatedParams1.joints = {
        {
            .type = ArticulatedJointType::Spherical //
        },
        {
            .type = ArticulatedJointType::Spherical, //
            .parentLinkFromJoint = offset //
        }};
    articulatedParams1.links = {
        {
            .parentLink = -1, //
            .shape = linkShape,
            .colliderType = ColliderType::None //
        },
        {
            .parentLink = 0,
            .parentJointFromLink = offset,
            .shape = linkShape,
            .colliderType = ColliderType::None //
        }};
    auto const* articulatedActor1 = _scene->CreateArticulatedActor(articulatedParams1, ExpectOK{});
    _entitiesActors.push_back(GetEntity(articulatedActor1->GetHandle()));

    // Create second articulated actor with 2 links, positioned away from the first
    ArticulatedActorParams articulatedParams2;
    articulatedParams2.worldFromRoot = TransformRT{Real3{6_r, 0_r, 0_r}};
    articulatedParams2.joints = {
        {
            .type = ArticulatedJointType::Spherical //
        },
        {
            .type = ArticulatedJointType::Spherical, //
            .parentLinkFromJoint = offset //
        }};
    articulatedParams2.links = {
        {
            .parentLink = -1, //
            .shape = linkShape, //
            .colliderType = ColliderType::None //
        },
        {
            .parentLink = 0,
            .parentJointFromLink = offset,
            .shape = linkShape,
            .colliderType = ColliderType::None //
        }};
    auto const* articulatedActor2 = _scene->CreateArticulatedActor(articulatedParams2, ExpectOK{});
    _entitiesActors.push_back(GetEntity(articulatedActor2->GetHandle()));

    InitBase();
  }
};

} // namespace

TEST_F(ConstraintJointRotationTrackingDifferentActors, Test) {
  RunAllTests();
}
