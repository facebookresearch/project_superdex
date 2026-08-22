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

#include "mochi_physics_test_fixture.h"

#include <mochi_core/articulated_body/articulated_body_params.h>
#include <mochi_core/element_operations/fem_rod.h>
#include <mochi_core/solvers/snle_problem.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/constraints.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/rodrigues_utils.h>
#include <mochi_physics/src/mochi_articulated_body.h>
#include <mochi_physics/src/mochi_compound.h>
#include <mochi_physics/src/mochi_constraint.h>
#include <mochi_physics/src/mochi_group.h>
#include <mochi_physics/src/mochi_rod.h>
#include <mochi_physics/src/mochi_skinning.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace mochi::constraint_test {

template <typename T>
struct TimeStepPair : public std::array<T, 2> {
  constexpr T const& operator[](TimeStep timeStep) const {
    return std::array<T, 2>::operator[](static_cast<size_t>(timeStep));
  }
  constexpr T& operator[](TimeStep timeStep) {
    return std::array<T, 2>::operator[](static_cast<size_t>(timeStep));
  }
};

ShapeHandle GetUnitCubeShape(Context* context);

void TuneValuesForRangeConstraint(real& c, real& dc);

template <TimeStep kTimeStep>
void SetRigidTranslation(entt::registry& reg, entt::entity e, Real3 const& t) {
  auto& state = reg.get<CRigidState<kTimeStep>>(e).value;
  state.SetTranslation(t);
}

template <TimeStep kTimeStep>
void SetRigidRotation(entt::registry& reg, entt::entity e, Quaternion const& r) {
  auto& state = reg.get<CRigidState<kTimeStep>>(e).value;
  state.SetRotation(r);
}

template <TimeStep kTimeStep>
void SetRigidState(entt::registry& reg, entt::entity e, Real3 const& t, Quaternion const& r) {
  auto& state = reg.get<CRigidState<kTimeStep>>(e).value;
  state.SetTranslation(t);
  state.SetRotation(r);
}

void AddToTransformRT(int i, real eps, TransformRT& outTransform);

template <TimeStep kTimeStep>
void AddToRigidState(entt::registry& reg, entt::entity e, int i, real eps) {
  auto& state = reg.get<CRigidState<kTimeStep>>(e).value;
  AddToTransformRT(i, eps, state);
}

template <TimeStep kTimeStep>
void SetSoftState(entt::registry& reg, entt::entity e, Real3 const& disp) {
  auto& sol = reg.get<CDisplacementSlice<real, kTimeStep>>(e).value;
  for (int i = 0; i < 3; ++i) {
    sol[i] = disp[i];
  }
}

template <TimeStep kTimeStep>
void AddToSoftState(entt::registry& reg, entt::entity e, int i, real eps) {
  auto& sol = reg.get<CDisplacementSlice<real, kTimeStep>>(e).value;
  sol[i] += eps;
}

template <TimeStep kTimeStep>
void UpdateLinkStates(entt::registry& reg, entt::entity e) {
  auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
  auto const& pose = reg.get<CArticulatedReducedPose<kTimeStep> const>(e).value;
  auto const& parents = reg.get<CArticulatedParents const>(e);
  auto const& restTransforms = reg.get<CArticulatedRestTransforms const>(e);
  auto const& worldFromRoot = reg.get<CRootTransform const>(e).worldFromLocal;
  DynamicArray<TransformRT> jointTransforms(parents.size());
  DynamicArray<TransformRT> linkTransforms(parents.size());
  articulated::ComputeTransformsFromReducedPose(
      joints->jointTypes,
      joints->jointAxes,
      poseInfo,
      parents,
      restTransforms,
      worldFromRoot,
      pose,
      jointTransforms,
      linkTransforms);
  auto const& links = reg.get<CGroupMembers const>(e).actors;
  for (int i = 0; i < isize(links); ++i) {
    auto& state = reg.get<CRigidState<kTimeStep>>(links[i]).value;
    state = linkTransforms[i];
  }
}

template <TimeStep kTimeStep>
void SetArticulatedStateFromJointTransform(
    entt::registry& reg,
    entt::entity e,
    TransformRT const& transform) {
  auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
  auto& pose = reg.get<CArticulatedReducedPose<kTimeStep>>(e).value;
  mochi::articulated::ComputeJointPose(
      joints->jointTypes[0], joints->jointAxes[0], poseInfo[0], transform, pose);
  UpdateLinkStates<kTimeStep>(reg, e);
}

template <TimeStep kTimeStep>
void SetArticulatedStateFromDofs(
    entt::registry& reg,
    entt::entity e,
    ColumnVectorView<real const> dofs,
    int offset = 0) {
  auto& pose = reg.get<CArticulatedReducedPose<kTimeStep>>(e).value;
  pose.MiddleRows(offset, dofs.Rows()) = dofs;
  UpdateLinkStates<kTimeStep>(reg, e);
}

template <TimeStep kTimeStep>
void AddToArticulatedState(entt::registry& reg, entt::entity e, int i, real eps, int offset = 0) {
  auto& pose = reg.get<CArticulatedReducedPose<kTimeStep>>(e).value;
  auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
  auto delta = ColumnVector<real>::Zero(articulated::GetReducedDofsSize(joints->dofInfo));
  delta(offset + i) += eps;
  articulated::AddLieDeltaToReducedPose(
      joints->jointTypes, joints->dofInfo, poseInfo, pose, delta, pose);
  UpdateLinkStates<kTimeStep>(reg, e);
}

template <TimeStep kTimeStep>
void SetPositionTarget(entt::registry& reg, entt::entity e, Real3 const& p) {
  reg.get<CConstraintTarget<Real3, kTimeStep>>(e).value = p;
}

template <TimeStep kTimeStep>
void AddToPositionTarget(entt::registry& reg, entt::entity e, int i, real eps) {
  reg.get<CConstraintTarget<Real3, kTimeStep>>(e).value[i] += eps;
}

template <TimeStep kTimeStep>
void SetRotationTarget(entt::registry& reg, entt::entity e, Quaternion const& r) {
  reg.get<CConstraintTarget<Quaternion, kTimeStep>>(e).value = r;
}

template <TimeStep kTimeStep>
void AddToRotationTarget(entt::registry& reg, entt::entity e, int i, real eps) {
  auto& rot = reg.get<CConstraintTarget<Quaternion, kTimeStep>>(e).value;
  Real3 delta{};
  delta[i] = eps;
  rot = Quaternion::FromRotationVector(delta) * rot;
}

template <TimeStep kTimeStep>
void SetTransformTarget(entt::registry& reg, entt::entity e, Real3 const& p, Quaternion const& r) {
  auto& transform = reg.get<CConstraintTarget<TransformRT, kTimeStep>>(e).value;
  transform.SetTranslation(p);
  transform.SetRotation(r);
}

template <TimeStep kTimeStep>
void AddToTransformTarget(entt::registry& reg, entt::entity e, int i, real eps) {
  auto& transform = reg.get<CConstraintTarget<TransformRT, kTimeStep>>(e).value;
  AddToTransformRT(i, eps, transform);
}

/********************************************************************************
  Base for test fixtures. Derived classes test various types of constraints using
  the internal ECS code (not the public mochi_physics API).
********************************************************************************/
class ConstraintTestBase : public test::MochiSceneTestBase {
 protected:
  entt::entity _entityConstraint = {};
  std::vector<entt::entity> _entitiesActors;
  entt::entity _entityCompound = {};
  int _conSize = -1;
  int _supSize = -1;
  int _targetSize = -1;
  ColumnVector<real> _cVal; // for stiffness term
  ColumnVector<real> _dCVal; // for damping term
  real _dtStage = 0.2_r; // for damping term
  real _stiffness = 1_r;
  real _damping = 1_r;
  real _saturation = -1_r;
  bool _isLinear = false;

 public:
  void SetUp() override;
  void InitBase();
  void DeleteConstraint();

  // To be implemented by the type-specific derived class.
  virtual bool constexpr HasDifferentiableTarget() const = 0;
  virtual bool constexpr HasMixedLinks() const = 0;
  virtual void InitConstraint(Real3& cVal, Real3& dCVal) = 0;
  virtual void InitState(TimeStep timeStep) = 0;
  virtual void
  AddToState(TimeStep timeStep, int idxi, real di, int idxj, real dj, bool useFull) = 0;
  virtual void InitTarget(TimeStep timeStep) = 0;
  virtual void AddToTarget(TimeStep timeStep, int idx, real d) = 0;

  void ComputeConstraintValuesCurrAndStageStart();

  template <GradTarget kGradTarget>
  void ComputeObjResDRes(bool assemObj, bool assemRes, bool assemDRes) {
    AssemblyParams params{
        .assemObj = assemObj,
        .assemRes = assemRes,
        .assemDRes = assemDRes,
        .fittedSaturationHessian = SaturationHessianParams::All(false),
        .gradTarget = kGradTarget};
    entt::registry& reg = GetRegistry();
    auto& result = reg.get<CCompoundConstraintSnle>(_entityCompound);
    result.SetZero(params);
    AssembleConstraint<kGradTarget>(reg, _entityConstraint, params, result);
  }

  void ComputeObj();

  template <GradTarget kGradTarget>
  void ComputeRes() {
    ComputeObjResDRes<kGradTarget>(
        /*assemObj*/ false, /*assemRes*/ true, /*assemDRes*/ false);
  }

  void ComputeDRes();
  void RunTest_Obj();
  void SanityCheck_CVal(ColumnVectorView<real const> cVal);
  void SanityCheck_DCVal(ColumnVectorView<real const> dCVal);

  template <GradTarget kGradTarget>
  void RunTest_Res() {
    static TimeStep constexpr kTimeStep = GetTimeStep<kGradTarget>();
    static bool constexpr kIsInputTarget =
        kGradTarget == GradTarget::CurrentInput || kGradTarget == GradTarget::PreviousInput;

    // Define functions and values depending on regular or target gradient
    auto init = [&]() {
      if constexpr (kIsInputTarget) {
        InitTarget(kTimeStep);
      } else {
        InitState(kTimeStep);
      }
    };

    auto add = [&](int i, real eps) {
      if constexpr (kIsInputTarget) {
        AddToTarget(kTimeStep, i, eps);
      } else {
        AddToState(kTimeStep, i, eps, i, 0_r, true);
      }
    };

    init();
    ComputeRes<kGradTarget>();
    auto const& reg = GetRegistry();
    auto const& result = reg.get<CCompoundConstraintSnle const>(_entityCompound);
    auto const& globalResIndices = kIsInputTarget
        ? reg.get<CConstraintGlobalInputSparsityCache const>(_entityConstraint).resIndices
        : reg.get<CConstraintGlobalSparsityCache const>(_entityConstraint).resIndices;
    auto resSize = isize(globalResIndices);
    ColumnVector<real> res0(resSize);
    for (int i = 0; i < resSize; ++i) {
      res0[i] = result.residuals[0].second[globalResIndices[i]];
    }
    ColumnVector<real> resFD = res0.Duplicate();

    real dx = 2.0e-3_r;
    for (int i = 0; i < resSize; ++i) {
      init();
      add(i, dx);
      ComputeObj();
      double cplus = result.objective;

      init();
      add(i, -dx);
      ComputeObj();
      double cminus = result.objective;

      resFD[i] = real(cplus - cminus) / (2_r * dx);
    }

    ColumnVector<real> delta = (res0 - resFD);
    real err = delta.Norm();
    real res0_norm = res0.Norm();
    real resFD_norm = resFD.Norm();
    real norm = (res0_norm < 1e-5_r ? 1_r : std::max(res0_norm, resFD_norm));
    real rerr = err / norm;

    real tol = 2e-2_r;
    EXPECT_LE(rerr, tol);

    init();
  }

  template <TimeStep kTimeStep>
  void RunTest_CJac() {
    InitState(kTimeStep);

    entt::registry& reg = GetRegistry();
    RowMatrix<real> cjac(_conSize, _supSize);
    bool isActive{};
    EvalConstraint<kTimeStep>(reg, _entityConstraint, {}, cjac, {}, isActive);
    auto cjacFD = RowMatrix<real>::Zero(_conSize, _supSize);

    real dx = 2.0e-3_r;
    for (int i = 0; i < _supSize; ++i) {
      InitState(kTimeStep);
      AddToState(kTimeStep, i, dx, i, 0_r, false);
      EvalConstraint<kTimeStep>(reg, _entityConstraint, _cVal, {}, {}, isActive);
      ColumnVector<real> cvalFwd = _cVal.Duplicate();
      InitState(kTimeStep);
      AddToState(kTimeStep, i, -dx, i, 0_r, false);
      EvalConstraint<kTimeStep>(reg, _entityConstraint, _cVal, {}, {}, isActive);
      ColumnVector<real> cvalBwd = _cVal.Duplicate();
      cjacFD.Col(i) = (cvalFwd - cvalBwd) * (1_r / (2_r * dx));
    }

    Matrix<real> errMatrix = cjac - cjacFD;
    real errNorm = errMatrix.Norm();
    real cjacNorm = cjac.Norm();
    real cjacFDNorm = cjacFD.Norm();
    real norm = (cjacNorm < 1e-5_r ? 1_r : std::max(cjacNorm, cjacFDNorm));
    real rerr = errNorm / norm;

    EXPECT_LE(rerr, 2e-2_r);

    InitState(kTimeStep);
  }

  template <TimeStep kTimeStep>
  void RunTest_CJacTarget() {
    InitTarget(kTimeStep);

    entt::registry& reg = GetRegistry();
    RowMatrix<real> cjac(_conSize, _targetSize);
    bool isActive{};
    EvalConstraint<kTimeStep>(reg, _entityConstraint, {}, {}, cjac, isActive);
    auto cjacFD = RowMatrix<real>::Zero(_conSize, _targetSize);

    real dx = 1.0e-3_r;
    for (int i = 0; i < _targetSize; ++i) {
      InitTarget(kTimeStep);
      AddToTarget(kTimeStep, i, dx);
      EvalConstraint<kTimeStep>(reg, _entityConstraint, _cVal, {}, {}, isActive);
      ColumnVector<real> cvalFwd = _cVal.Duplicate();
      InitTarget(kTimeStep);
      AddToTarget(kTimeStep, i, -dx);
      EvalConstraint<kTimeStep>(reg, _entityConstraint, _cVal, {}, {}, isActive);
      ColumnVector<real> cvalBwd = _cVal.Duplicate();
      cjacFD.Col(i) = (cvalFwd - cvalBwd) * (1_r / (2_r * dx));
    }

    Matrix<real> errMatrix = cjac - cjacFD;
    real errNorm = errMatrix.Norm();
    real cjacNorm = cjac.Norm();
    real cjacFDNorm = cjacFD.Norm();
    real norm = (cjacNorm < 1e-5_r ? 1_r : std::max(cjacNorm, cjacFDNorm));
    real rerr = errNorm / norm;

    EXPECT_LE(rerr, 2e-2_r);

    InitTarget(kTimeStep);
  }

  void RunTest_DRes();
  void RunTest_GetStiffness(real target);
  void RunTest_SetStiffness(real stiffness);
  void RunTest_GetDamping(real target);
  void RunTest_SetDamping(real damping);
  void RunTest(Real3 cVal, Real3 dCVal, bool testObj, bool testHessian);
  void RunAllTests();
};

// Implements the parts of ConstraintTestBase that depend on the constraint data type
template <class TConstraintData>
class ConstraintTestBaseT : public ConstraintTestBase {
 protected:
  TConstraintData _constraintData;

 public:
  // Return if the constraint has a differentiable target
  bool constexpr HasDifferentiableTarget() const override {
    return TConstraintData::kHasDifferentiableTarget;
  }

  // Return if the constraint has mixed links
  bool constexpr HasMixedLinks() const override {
    return TConstraintData::kHasMixedLinks;
  }

  // Create the constraint and initialize the actors
  void InitConstraint(Real3& cVal, Real3& dCVal) override {
    auto& reg = GetRegistry();
    auto constraintHandle = _constraintData.InitConstraint(
        reg, _scene, _entitiesActors, _stiffness, _damping, _saturation, _dtStage, cVal, dCVal);
    _entityConstraint = GetEntity(constraintHandle);

    // Add the constraint to the compound
    AddConstraintToCompound(reg, _entityCompound, _entityConstraint, test::ExpectOK{});

    // Initialize global residual and dresidual sparsity indices for the constraint
    auto& members = reg.template get<CGroupMembers const>(_entityCompound);
    compound::UpdateConstraintGlobalSparsity(reg, members, _entityCompound);
    if (HasDifferentiableTarget()) {
      reg.template emplace<TagConstraintWithDifferentiableInput>(_entityConstraint);
      reg.template emplace<CConstraintGlobalInputSparsityCache>(_entityConstraint);
      compound::UpdateConstraintGlobalInputSparsity(reg, members, _entityCompound);
    }
  }

  void InitState(TimeStep timeStep) override {
    auto& reg = GetRegistry();
    switch (timeStep) {
      case TimeStep::Current:
        _constraintData.template InitState<TimeStep::Current>(reg, _entitiesActors);
        break;
      case TimeStep::StageStart:
        _constraintData.template InitState<TimeStep::StageStart>(reg, _entitiesActors);
        break;
      default:
        MOCHI_ASSERT_VERBOSE(false, "Unsupported time step type");
        break;
    }
  }

  void AddToState(TimeStep timeStep, int idxi, real di, int idxj, real dj, bool useFull) override {
    auto& reg = GetRegistry();
    auto addToState = [&]() {
      switch (timeStep) {
        case TimeStep::Current:
          _constraintData.template AddToState<TimeStep::Current>(
              reg, _entitiesActors, idxi, di, idxj, dj);
          break;
        case TimeStep::StageStart:
          _constraintData.template AddToState<TimeStep::StageStart>(
              reg, _entitiesActors, idxi, di, idxj, dj);
          break;
        default:
          MOCHI_ASSERT_VERBOSE(false, "Unsupported time step type");
          break;
      }
    };

    if constexpr (TConstraintData::kHasMixedLinks) {
      if (useFull) {
        switch (timeStep) {
          case TimeStep::Current:
            _constraintData.template AddToStateFull<TimeStep::Current>(
                reg, _entitiesActors, idxi, di, idxj, dj);
            break;
          case TimeStep::StageStart:
            _constraintData.template AddToStateFull<TimeStep::StageStart>(
                reg, _entitiesActors, idxi, di, idxj, dj);
            break;
          default:
            MOCHI_ASSERT_VERBOSE(false, "Unsupported time step type");
            break;
        }
      } else {
        addToState();
      }
    } else {
      addToState();
    }
  }

  void InitTarget(TimeStep timeStep) override {
    if constexpr (TConstraintData::kHasDifferentiableTarget) {
      auto& reg = GetRegistry();
      switch (timeStep) {
        case TimeStep::Current:
          _constraintData.template InitTarget<TimeStep::Current>(reg, _entityConstraint);
          break;
        case TimeStep::StageStart:
          _constraintData.template InitTarget<TimeStep::StageStart>(reg, _entityConstraint);
          break;
        default:
          MOCHI_ASSERT_VERBOSE(false, "Unsupported time step type");
          break;
      }
    } else {
      MOCHI_ASSERT_VERBOSE(false, "Constraint does not have target");
    }
  }

  void AddToTarget(TimeStep timeStep, int idx, real d) override {
    if constexpr (TConstraintData::kHasDifferentiableTarget) {
      auto& reg = GetRegistry();
      switch (timeStep) {
        case TimeStep::Current:
          _constraintData.template AddToTarget<TimeStep::Current>(reg, _entityConstraint, idx, d);
          break;
        case TimeStep::StageStart:
          _constraintData.template AddToTarget<TimeStep::StageStart>(
              reg, _entityConstraint, idx, d);
          break;
        default:
          MOCHI_ASSERT_VERBOSE(false, "Unsupported time step type");
          break;
      }
    } else {
      MOCHI_ASSERT_VERBOSE(false, "Constraint does not have target");
    }
  }
};

} // namespace mochi::constraint_test
