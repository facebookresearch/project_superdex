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

ShapeHandle constraint_test::GetUnitCubeShape(Context* context) {
  auto&& [coordinates, connectivity] = CreateMinimalTetMeshUnitCube();
  return context->CreateTetMeshShape(
      Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), ExpectOK{});
}

void constraint_test::TuneValuesForRangeConstraint(real& c, real& dc) {
  if (c == 0_r) {
    // If c = 0, then dc = 0
    dc = 0_r;
  } else if (Abs(c) < 1e-2_r) {
    // Non-zero c must be larger than the FD delta
    c = 1e-2_r;
  }

  // c - dtStage * dc must be > 0. Make sure c is > 0 and dc < 0.
  c = c < 0_r ? -c : c;
  dc = dc > 0_r ? -dc : dc;
}

void constraint_test::AddToTransformRT(int i, real eps, TransformRT& outTransform) {
  Real3 delta{};
  delta[i % 3] = eps;
  if (i < 3) {
    outTransform.SetTranslation(outTransform.GetTranslation() + delta);
  } else {
    outTransform.SetRotation(Quaternion::FromRotationVector(delta) * outTransform.GetRotation());
  }
}

void ConstraintTestBase::SetUp() {
  MochiSceneTestBase::SetUp();
  // Back-propagation requires backward Euler.
  SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
  MakeSceneDifferentiableInternal(_scene, ExpectOK{});
}

void ConstraintTestBase::InitBase() {
  // Resize constraint values
  _cVal.Resize(_conSize);
  _cVal.SetZero();
  _dCVal.Resize(_conSize);
  _dCVal.SetZero();

  // Create a compound if needed, and add actors to it
  auto& reg = GetRegistry();
  _entityCompound = TryGetParentArticulatedActor(reg, _entitiesActors[0]);
  for (int i = 1; i < isize(_entitiesActors); ++i) {
    if (TryGetParentArticulatedActor(reg, _entitiesActors[i]) != _entityCompound) {
      _entityCompound = entt::null;
    }
  }
  if (_entityCompound == entt::null) {
    _entityCompound = reg.create();
    InitCompoundActor(reg, _entityCompound, test::ExpectOK{});
    for (auto actor : _entitiesActors) {
      AddActorToCompound(reg, _entityCompound, actor, test::ExpectOK{});
    }
  }
  auto const& members = reg.template get<CGroupMembers const>(_entityCompound);
  compound::UpdateDofInfo(reg, _entityCompound, members, 0, 0, 0, 0);

  // Set dtStage on all actors
  for (auto const& e : _entitiesActors) {
    reg.template get<CTimeIntegratorState>(e).dtStage = _dtStage;
  }

  // Do not use the fitted Hessian for constraint saturation, to improve finite-difference
  // consistency tests.
  auto solverParams = _scene->GetSolverParams();
  solverParams.experimentalEval.fittedSaturationHessian = SaturationHessianParams::All(false);
  _scene->SetSolverParams(solverParams, test::ExpectOK{});
}

void ConstraintTestBase::DeleteConstraint() {
  _scene->DestroyConstraint(GetConstraintHandle(_entityConstraint, _scene->GetHandle()));
}

void ConstraintTestBase::ComputeConstraintValuesCurrAndStageStart() {
  entt::registry& reg = GetRegistry();
  bool isActive{};
  EvalConstraint<TimeStep::Current>(reg, _entityConstraint, _cVal, {}, {}, isActive);
  EvalConstraint<TimeStep::StageStart>(reg, _entityConstraint, _dCVal, {}, {}, isActive);
  _dCVal = (_cVal - _dCVal) * (1_r / _dtStage);
}

void ConstraintTestBase::ComputeObj() {
  ComputeObjResDRes<GradTarget::Current>(
      /*assemObj*/ true, /*assemRes*/ false, /*assemDRes*/ false);
}

void ConstraintTestBase::ComputeDRes() {
  ComputeObjResDRes<GradTarget::Current>(
      /*assemObj*/ false, /*assemRes*/ false, /*assemDRes*/ true);
}

void ConstraintTestBase::RunTest_Obj() {
  ComputeObj();
  double cobj = GetRegistry().get<CCompoundConstraintSnle>(_entityCompound).objective;

  ComputeConstraintValuesCurrAndStageStart(); // Sets internal _cVal and _dCVal values

  auto const* constraint =
      _scene->GetConstraint(GetConstraintHandle(_entityConstraint, _scene->GetHandle()));
  real expected = 0.5 * constraint->GetStiffness() * _cVal.Dot(_cVal) +
      0.5 * _dtStage * constraint->GetDamping() * _dCVal.Dot(_dCVal);

  EXPECT_NEAR(expected, cobj, 1e-6_r);
}

void ConstraintTestBase::SanityCheck_CVal(ColumnVectorView<real const> cVal) {
  ComputeConstraintValuesCurrAndStageStart(); // Sets internal _cVal and _dCVal values
  ColumnVector<real> err = _cVal - cVal;
  EXPECT_LE(err.Norm(), 1e-4_r);
}

void ConstraintTestBase::SanityCheck_DCVal(ColumnVectorView<real const> dCVal) {
  ComputeConstraintValuesCurrAndStageStart(); // Sets internal _cVal and _dCVal values
  ColumnVector<real> err = (_dCVal - dCVal);
  EXPECT_LE(err.Norm(), 1e-4_r);
}

void ConstraintTestBase::RunTest_DRes() {
  InitState(TimeStep::Current);
  ComputeObj();
  auto const& reg = GetRegistry();
  auto const& result = reg.get<CCompoundConstraintSnle>(_entityCompound);
  double cbase = result.objective;

  ComputeDRes();
  auto const& dres = std::get<SparseMatrix<real>>(result.dresiduals[0].matrix);
  auto const& sparsity = reg.get<CConstraintGlobalSparsityCache const>(_entityConstraint);
  int const resSize = isize(sparsity.resIndices);
  Matrix<real> dres0(resSize, resSize);
  for (int i = 0; i < resSize; ++i) {
    for (int j = 0; j < resSize; ++j) {
      dres0(i, j) = dres.Values()[sparsity.dresIndices[i * resSize + j]];
    }
  }
  Matrix<real> dresFD = dres0.Duplicate();

  real dx = 3e-3_r;
  for (int i = 0; i < resSize; ++i) {
    for (int j = 0; j < resSize; ++j) {
      if (i == j) {
        InitState(TimeStep::Current);
        AddToState(TimeStep::Current, i, dx, i, 0_r, true);
        ComputeObj();
        double cplus = result.objective;

        InitState(TimeStep::Current);
        AddToState(TimeStep::Current, i, -dx, i, 0_r, true);
        ComputeObj();
        double cminus = result.objective;

        auto val = (cplus - 2_r * cbase + cminus) / (dx * dx);
        dresFD(i, i) = val;
      } else {
        InitState(TimeStep::Current);
        AddToState(TimeStep::Current, i, dx, j, dx, true);
        ComputeObj();
        double cplusplus = result.objective;

        InitState(TimeStep::Current);
        AddToState(TimeStep::Current, i, -dx, j, dx, true);
        ComputeObj();
        double cminusplus = result.objective;

        InitState(TimeStep::Current);
        AddToState(TimeStep::Current, i, dx, j, -dx, true);
        ComputeObj();
        double cplusminus = result.objective;

        InitState(TimeStep::Current);
        AddToState(TimeStep::Current, i, -dx, j, -dx, true);
        ComputeObj();
        double cminusminus = result.objective;

        auto val = (cplusplus + cminusminus - cplusminus - cminusplus) * (1_r / (4_r * dx * dx));
        dresFD(i, j) = val;
      }
    }
  }

  // For debugging purposes
  std::vector<real> dresSpan;
  dresSpan.resize(resSize * resSize);
  std::vector<real> dresFDSpan;
  dresFDSpan.resize(resSize * resSize);
  for (int i = 0; i < resSize; i++) {
    for (int j = 0; j < resSize; j++) {
      dresSpan[resSize * i + j] = dres0(i, j);
      dresFDSpan[resSize * i + j] = dresFD(i, j);
    }
  }

  Matrix<real> delta = dres0 - dresFD;
  real tol = 1e-2_r;
  EXPECT_NEAR(delta.Norm() / std::max(std::max(dres0.Norm(), dresFD.Norm()), 1.0e-6_r), 0_r, tol);

  InitState(TimeStep::Current);
}

void ConstraintTestBase::RunTest_GetStiffness(real target) {
  auto const* constraint =
      _scene->GetConstraint(GetConstraintHandle(_entityConstraint, _scene->GetHandle()));
  EXPECT_EQ(target, constraint->GetStiffness());
}

void ConstraintTestBase::RunTest_SetStiffness(real stiffness) {
  // Get previous settings.
  auto* constraint =
      _scene->GetConstraint(GetConstraintHandle(_entityConstraint, _scene->GetHandle()));
  real stiffnessOld = constraint->GetStiffness();
  real dampingOld = constraint->GetDamping();
  real saturationOld = constraint->GetSaturation();

  // Check negative stiffness is illegal.
  constraint->SetStiffness(-1_r, test::ExpectNotOK{});
  // Check zero saturation is illegal.
  constraint->SetSaturation(0_r, test::ExpectNotOK{});

  // Eliminate damping and saturation for the test.
  constraint->SetDamping(0_r, test::ExpectOK{});
  constraint->SetSaturation(-1_r, test::ExpectOK{});

  auto const& result = GetRegistry().get<CCompoundConstraintSnle>(_entityCompound);

  // Evaluate objective with old stiffness
  ComputeObj();
  double cobjOld = result.objective;

  // Evaluate objective with new stiffness
  constraint->SetStiffness(stiffness, test::ExpectOK{});
  ComputeObj();
  double cobjNew = result.objective;

  // Test if the change in objective is the same as the change in stiffness
  EXPECT_EQ(cobjNew / cobjOld, stiffness / stiffnessOld);

  // Restore old settings
  constraint->SetStiffness(stiffnessOld, test::ExpectOK{});
  constraint->SetDamping(dampingOld, test::ExpectOK{});
  constraint->SetSaturation(saturationOld, test::ExpectOK{});
}

void ConstraintTestBase::RunTest_GetDamping(real target) {
  auto const* constraint =
      _scene->GetConstraint(GetConstraintHandle(_entityConstraint, _scene->GetHandle()));
  EXPECT_EQ(target, constraint->GetDamping());
}

void ConstraintTestBase::RunTest_SetDamping(real damping) {
  // Get previous settings.
  auto* constraint =
      _scene->GetConstraint(GetConstraintHandle(_entityConstraint, _scene->GetHandle()));
  real stiffnessOld = constraint->GetStiffness();
  real dampingOld = constraint->GetDamping();
  real saturationOld = constraint->GetSaturation();

  // Check negative damping is illegal.
  constraint->SetDamping(-1_r, test::ExpectNotOK{});
  // Check zero saturation is illegal.
  constraint->SetSaturation(0_r, test::ExpectNotOK{});

  // Eliminate stiffness and saturation for the test.
  constraint->SetStiffness(0_r, test::ExpectOK{});
  constraint->SetSaturation(-1_r, test::ExpectOK{});

  auto const& result = GetRegistry().get<CCompoundConstraintSnle>(_entityCompound);

  // Evaluate objective with old damping
  ComputeObj();
  double cobjOld = result.objective;

  // Evaluate objective with new damping
  constraint->SetDamping(damping, test::ExpectOK{});
  ComputeObj();
  double cobjNew = result.objective;

  // Test if the change in objective is the same as the change in damping
  EXPECT_EQ(cobjNew / cobjOld, damping / dampingOld);

  // Restore old settings
  constraint->SetStiffness(stiffnessOld, test::ExpectOK{});
  constraint->SetDamping(dampingOld, test::ExpectOK{});
  constraint->SetSaturation(saturationOld, test::ExpectOK{});
}

void ConstraintTestBase::RunTest(Real3 cVal, Real3 dCVal, bool testObj, bool testHessian) {
  InitConstraint(cVal, dCVal);
  SanityCheck_CVal(AsConstView(cVal).TopRows(_conSize));
  SanityCheck_DCVal(AsConstView(dCVal).TopRows(_conSize));
  RunTest_CJac<TimeStep::Current>();
  RunTest_CJac<TimeStep::StageStart>();
  if (HasDifferentiableTarget()) {
    RunTest_CJacTarget<TimeStep::Current>();
    RunTest_CJacTarget<TimeStep::StageStart>();
  }
  if (testObj) {
    RunTest_Obj();
  }
  RunTest_Res<GradTarget::Current>();
  RunTest_Res<GradTarget::Previous>();
  if (HasDifferentiableTarget()) {
    RunTest_Res<GradTarget::CurrentInput>();
    RunTest_Res<GradTarget::PreviousInput>();
  }
  if (testHessian) {
    RunTest_DRes();
  }
  DeleteConstraint();
}

void ConstraintTestBase::RunAllTests() {
  // Values of c and dc for the tests
  std::array<Real3, 3> cVals = {
      Real3{}, Real3{0.001_r, 0.002_r, 0.003_r}, Real3{0.1_r, 0.2_r, 0.3_r}};
  std::array<Real3, 3> dcVals = {
      Real3{}, Real3{0.002_r, 0.003_r, -0.001_r}, Real3{0.2_r, 0.3_r, -0.1_r}};

  // Tests for parameter get/set
  Real3 cVal = cVals[2];
  Real3 dcVal = dcVals[2];
  InitConstraint(cVal, dcVal);
  RunTest_GetStiffness(1_r);
  RunTest_SetStiffness(2_r);
  RunTest_GetDamping(1_r);
  RunTest_SetDamping(2_r);
  DeleteConstraint();

  // Consistency tests with and without saturation.
  auto runTests = [&]() {
    bool testObj = _saturation < 0_r;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        bool testHessian = _isLinear || (i < 2 && j < 2);
        RunTest(cVals[i], dcVals[j], testObj, testHessian);
      }
    }
  };
  _saturation = -1_r;
  runTests();
  _saturation = 0.1_r;
  runTests();
}
