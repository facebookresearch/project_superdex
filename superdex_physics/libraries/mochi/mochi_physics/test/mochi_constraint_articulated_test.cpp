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
  Articulated3dRotationTargetConstraint
********************************************************************************/

namespace {

struct Articulated3dRotationTargetData {
  static bool constexpr kHasDifferentiableTarget = true;
  static bool constexpr kHasMixedLinks = false;
  TimeStepPair<TransformRT> jointTx = {};
  TimeStepPair<Quaternion> target = {};

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    SetArticulatedStateFromJointTransform<kTimeStep>(reg, entities[0], jointTx[kTimeStep]);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    AddToArticulatedState<kTimeStep>(reg, entities[0], idxi, di);
    AddToArticulatedState<kTimeStep>(reg, entities[0], idxj, dj);
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
    Real3 rotAxis = {1.0_r, 0.0_r, 0.0_r};
    target[TimeStep::Current] = Quaternion::FromAxisAngle(rotAxis, 0.5_r);
    target[TimeStep::StageStart] = target[TimeStep::Current];

    Articulated3dRotationTargetConstraintParams conParams;
    conParams.actor = GetActorHandle(entities[0], scene->GetHandle());
    conParams.jointIndex = 0;
    conParams.target = target[TimeStep::Current];
    conParams.stiffness = stiffness;
    conParams.damping = damping;
    conParams.saturation = saturation;
    auto const* constraint =
        scene->CreateArticulated3dRotationTargetConstraint(conParams, ErrorAssert{});

    auto quat = Quaternion::FromRotationVector(cVal) * conParams.target;
    jointTx[TimeStep::Current] = TransformRT{quat, Real3{}};
    InitState<TimeStep::Current>(reg, entities);

    quat = Quaternion::FromRotationVector(cVal - dtStage * dCVal) * conParams.target;
    jointTx[TimeStep::StageStart] = TransformRT{quat, Real3{}};
    InitState<TimeStep::StageStart>(reg, entities);

    return constraint->GetHandle();
  }
};

class ConstraintArticulated3dRotationTarget
    : public ConstraintTestBaseT<Articulated3dRotationTargetData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 3;
    _targetSize = 3;

    ArticulatedActorParams actorParams;
    actorParams.joints = {{.type = ArticulatedJointType::Spherical, .axis = Real3{1_r, 0_r, 0_r}}};
    actorParams.links = {
        {.parentLink = -1,
         .shape = GetUnitCubeShape(_scene->GetContext()),
         .colliderType = ColliderType::None}};
    auto const* actor = _scene->CreateArticulatedActor(actorParams, ExpectOK{});
    _entitiesActors.push_back(GetEntity(actor->GetHandle()));

    InitBase();
  }
};

} // namespace

TEST_F(ConstraintArticulated3dRotationTarget, Test) {
  RunAllTests();
}

TEST_F(ConstraintArticulated3dRotationTarget, JointRotVectorWrapAround) {
  real relAngle = 179_r;
  _saturation = -1_r;
  Real3 dCVal{};

  // Evaluations for 179 degrees
  Real3 eValP = Quaternion::FromAxisAngle(Normalize(Real3{1_r, 2_r, 3_r}), relAngle * kPI / 180_r)
                    .ToRotationVector();
  InitConstraint(eValP, dCVal);
  ComputeConstraintValuesCurrAndStageStart(); // Sets internal _cVal value
  ComputeRes<GradTarget::Current>();
  auto const* constraint =
      _scene->GetConstraint(GetConstraintHandle(_entityConstraint, _scene->GetHandle()));
  double eP = 0.5 * constraint->GetStiffness() * _cVal.Dot(_cVal); // Energy
  auto const& resultP = GetRegistry().get<CCompoundConstraintSnle>(_entityCompound);
  auto resSize = resultP.residuals[0].second.size();
  ColumnVector<real> resP = resultP.residuals[0].second.Duplicate();
  auto resPSpan = resP.GetSpan(); // Explicity span, so that we can debug easily
  auto const* cResData = resultP.residuals[0].second.GetConstSpan().data();
  memcpy(resPSpan.data(), cResData, sizeof(real) * resSize);

  // Evaluations for -179 degrees
  Real3 eValM = Quaternion::FromAxisAngle(Normalize(Real3{1_r, 2_r, 3_r}), -relAngle * kPI / 180_r)
                    .ToRotationVector();
  InitConstraint(eValM, dCVal);
  ComputeConstraintValuesCurrAndStageStart(); // Sets internal _cVal value
  ComputeRes<GradTarget::Current>();
  double eM = 0.5 * constraint->GetStiffness() * _cVal.Dot(_cVal); // Energy
  auto const& resultM = GetRegistry().get<CCompoundConstraintSnle>(_entityCompound);
  ColumnVector<real> resM = resultM.residuals[0].second.Duplicate();
  auto resMSpan = resM.GetSpan(); // Explicit span, so that we can debug easily
  cResData = resultM.residuals[0].second.GetConstSpan().data();
  memcpy(resMSpan.data(), cResData, sizeof(real) * resSize);

  EXPECT_NEAR(eP, eM, 1e-3_r); // Energies should be very similar

  ColumnVector<real> delta = resP - resM;
  real err = delta.Norm();
  real resP_norm = resP.Norm();
  real resM_norm = resM.Norm();
  real norm = (resP_norm < 1e-5_r ? 1_r : std::max(resP_norm, resM_norm));
  real rerr = err / norm;

  EXPECT_GE(rerr, 1e-1_r); // Note that the error is large
}

/********************************************************************************
  ArticulatedSingleDofTargetConstraint
********************************************************************************/
namespace {

struct ArticulatedSingleDofTargetData {
  static bool constexpr kHasDifferentiableTarget = true;
  static bool constexpr kHasMixedLinks = false;
  TimeStepPair<ColumnVector<real, 3>> dofs = {};
  TimeStepPair<real> target = {};

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    SetArticulatedStateFromDofs<kTimeStep>(reg, entities[0], dofs[kTimeStep]);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    AddToArticulatedState<kTimeStep>(reg, entities[0], idxi, di);
    AddToArticulatedState<kTimeStep>(reg, entities[0], idxj, dj);
  }

  template <TimeStep kTimeStep>
  void InitTarget(entt::registry& reg, entt::entity e) {
    reg.get<CConstraintTarget<real, kTimeStep>>(e).value = target[kTimeStep];
  }

  template <TimeStep kTimeStep>
  void AddToTarget(entt::registry& reg, entt::entity e, int /* idx */, real d) {
    reg.get<CConstraintTarget<real, kTimeStep>>(e).value += d;
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
    dofs[TimeStep::Current] = AsConstView(Real3{0.2_r, -0.8_r, 0.7_r});
    InitState<TimeStep::Current>(reg, entities);

    target[TimeStep::Current] = dofs[TimeStep::Current][0] - cVal[0];
    target[TimeStep::StageStart] = target[TimeStep::Current];

    ArticulatedSingleDofTargetConstraintParams conParams;
    conParams.actor = GetActorHandle(entities[0], scene->GetHandle());
    conParams.dofIndex = 0;
    conParams.targetValue = target[TimeStep::Current];
    conParams.stiffness = stiffness;
    conParams.damping = damping;
    conParams.saturation = saturation;
    auto const* constraint =
        scene->CreateArticulatedSingleDofTargetConstraint(conParams, ErrorAssert{});

    reg.get<CArticulatedReducedPose<TimeStep::StageStart>>(entities[0]).value =
        reg.get<CArticulatedReducedPose<TimeStep::Current> const>(entities[0]).value;
    dofs[TimeStep::StageStart] = dofs[TimeStep::Current].Duplicate();
    dofs[TimeStep::StageStart][0] = conParams.targetValue + cVal[0] - dtStage * dCVal[0];
    InitState<TimeStep::StageStart>(reg, entities);

    return constraint->GetHandle();
  }
};

class ConstraintArticulatedSingleDofTarget
    : public ConstraintTestBaseT<ArticulatedSingleDofTargetData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 1;
    _supSize = 1;
    _targetSize = 1;

    ArticulatedActorParams actorParams;
    actorParams.joints = {{.type = ArticulatedJointType::Free}};
    actorParams.links = {
        {.parentLink = -1,
         .shape = GetUnitCubeShape(_scene->GetContext()),
         .colliderType = ColliderType::None}};
    auto const* actor = _scene->CreateArticulatedActor(actorParams, ExpectOK{});
    _entitiesActors.push_back(GetEntity(actor->GetHandle()));

    _isLinear = true;

    InitBase();
  }
};

} // namespace

TEST_F(ConstraintArticulatedSingleDofTarget, Test) {
  RunAllTests();
}

/********************************************************************************
  Articulated3dRotationRangeConstraint
********************************************************************************/
namespace {

struct Articulated3dRotationRangeData {
  static bool constexpr kHasDifferentiableTarget = false;
  static bool constexpr kHasMixedLinks = false;
  TimeStepPair<Quaternion> dofs = {};

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    SetArticulatedStateFromDofs<kTimeStep>(
        reg, entities[0], AsConstView(dofs[kTimeStep].ToReal4()));
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    AddToArticulatedState<kTimeStep>(reg, entities[0], idxi, di);
    AddToArticulatedState<kTimeStep>(reg, entities[0], idxj, dj);
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

    Real3 minValues = {-1_r, -1_r, -1_r};
    Real3 maxValues = {1_r, 1_r, 1_r};

    Articulated3dRotationRangeConstraintParams conParams;
    conParams.actor = GetActorHandle(entities[0], scene->GetHandle());
    conParams.jointIndex = 0;
    conParams.minValues = minValues;
    conParams.maxValues = maxValues;
    conParams.stiffness = stiffness;
    conParams.damping = damping;
    conParams.saturation = saturation;
    auto const* constraint =
        scene->CreateArticulated3dRotationRangeConstraint(conParams, ErrorAssert{});
    Real3 dofsRotVector;
    for (auto i = 0; i < 3; ++i) {
      if (cVal[i] == 0) {
        dofsRotVector[i] = 0.5_r * (minValues[i] + maxValues[i]);
      } else {
        dofsRotVector[i] = maxValues[i] + cVal[i];
      }
    }
    dofs[TimeStep::Current] = Quaternion::FromRotationVector(dofsRotVector);
    InitState<TimeStep::Current>(reg, entities);

    auto cVal0 = cVal - dtStage * dCVal;
    for (auto i = 0; i < 3; ++i) {
      if (cVal0[i] == 0) {
        dofsRotVector[i] = 0.5_r * (minValues[i] + maxValues[i]);
      } else {
        dofsRotVector[i] = maxValues[i] + cVal0[i];
      }
    }
    dofs[TimeStep::StageStart] = Quaternion::FromRotationVector(dofsRotVector);
    InitState<TimeStep::StageStart>(reg, entities);

    return constraint->GetHandle();
  }
};

class ConstraintArticulated3dRotationRange
    : public ConstraintTestBaseT<Articulated3dRotationRangeData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 3;
    _targetSize = 0;

    ArticulatedActorParams actorParams;
    actorParams.joints = {{.type = ArticulatedJointType::Spherical}};
    actorParams.links = {
        {.parentLink = -1,
         .shape = GetUnitCubeShape(_scene->GetContext()),
         .colliderType = ColliderType::None}};
    auto const* actor = _scene->CreateArticulatedActor(actorParams, ExpectOK{});
    _entitiesActors.push_back(GetEntity(actor->GetHandle()));

    InitBase();
  }
};

} // namespace

TEST_F(ConstraintArticulated3dRotationRange, Test) {
  RunAllTests();
}

/********************************************************************************
  ArticulatedSingleDofRangeConstraint
********************************************************************************/
namespace {

struct ArticulatedSingleDofRangeData {
  static bool constexpr kHasDifferentiableTarget = false;
  static bool constexpr kHasMixedLinks = false;
  TimeStepPair<real> dof = {};

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    SetArticulatedStateFromDofs<kTimeStep>(
        reg, entities[0], ColumnVectorView<real const, 1>(&dof[kTimeStep], 1), RigidSize::kRot);
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    AddToArticulatedState<kTimeStep>(reg, entities[0], idxi, di, RigidSize::kDRot);
    AddToArticulatedState<kTimeStep>(reg, entities[0], idxj, dj, RigidSize::kDRot);
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

    real minValue = -1_r;
    real maxValue = 1_r;

    ArticulatedSingleDofRangeConstraintParams conParams;
    conParams.actor = GetActorHandle(entities[0], scene->GetHandle());
    conParams.jointIndex = 1;
    conParams.dofIndex = 0;
    conParams.minValue = minValue;
    conParams.maxValue = maxValue;
    conParams.stiffness = stiffness;
    conParams.damping = damping;
    conParams.saturation = saturation;
    auto const* constraint =
        scene->CreateArticulatedSingleDofRangeConstraint(conParams, ErrorAssert{});

    dof[TimeStep::Current] =
        (cVal[0] == 0) ? (0.5_r * (minValue + maxValue)) : (maxValue + cVal[0]);
    InitState<TimeStep::Current>(reg, entities);

    reg.get<CArticulatedReducedPose<TimeStep::StageStart>>(entities[0]).value =
        reg.get<CArticulatedReducedPose<TimeStep::Current> const>(entities[0]).value;
    auto cVal0 = cVal[0] - dtStage * dCVal[0];
    dof[TimeStep::StageStart] = (cVal0 == 0) ? (0.5_r * (minValue + maxValue)) : (maxValue + cVal0);
    InitState<TimeStep::StageStart>(reg, entities);

    return constraint->GetHandle();
  }
};

class ConstraintArticulatedSingleDofRange
    : public ConstraintTestBaseT<ArticulatedSingleDofRangeData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 1;
    _supSize = 1;
    _targetSize = 0;

    auto linkShape = GetUnitCubeShape(_scene->GetContext());
    ArticulatedActorParams actorParams;
    actorParams.joints = {
        {.type = ArticulatedJointType::Spherical},
        {.type = ArticulatedJointType::Revolute, .axis = Real3{0_r, 1_r, 0_r}}};
    actorParams.links = {
        {.parentLink = -1, .shape = linkShape, .colliderType = ColliderType::None},
        {.parentLink = 0, .shape = linkShape, .colliderType = ColliderType::None}};
    auto const* actor = _scene->CreateArticulatedActor(actorParams, ExpectOK{});
    _entitiesActors.push_back(GetEntity(actor->GetHandle()));

    _isLinear = true;

    InitBase();
  }
};

} // namespace

TEST_F(ConstraintArticulatedSingleDofRange, Test) {
  RunAllTests();
}
