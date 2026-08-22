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

#include "mochi_autodiff.h"
#include "mochi_physics_test_fixture.h"

#if MOCHI_USE_EIGEN

#define TRANS(M) (M).template block<3, 1>(0, 3)
#define ROT(M) (M).template block<3, 3>(0, 0)

using namespace mochi;
using namespace mochi::experimental;
using namespace mochi::test;

namespace mochi::autodiff {

static VecAD2 InitADVector(ColumnVectorView<real const> x0) {
  int const n = isize(x0);
  VecAD2 x(n);
  for (int i = 0; i < n; ++i) {
    // Set primal value
    x(i).value().value() = x0(i);

    // Inner (1st-order) derivatives for value() -> gradient
    x(i).value().derivatives() = Vec::Zero(n);
    x(i).value().derivatives()(i) = 1_r;

    // Outer derivatives (d x_i / d x_j) as AD1 constants
    x(i).derivatives() = VecAD1(n);
    for (int j = 0; j < n; ++j) {
      x(i).derivatives()(j).value() = (i == j) ? 1_r : 0_r;
      // derivative of a constant is zero
      // (this is what enables 2nd derivatives to accumulate correctly)
      x(i).derivatives()(j).derivatives() = Vec::Zero(n);
    }
  }
  return x;
}

static AD2 Const(real value) {
  return static_cast<AD2>(value);
}

static Vec3AD2 Const(Real3 const& value) {
  Vec3AD2 result;
  for (int r = 0; r < 3; ++r) {
    result[r] = Const(value[r]);
  }
  return result;
}

static Vec3AD2 Const(Vec4r const& value) {
  return Const(ToReal3(value));
}

static Mat3AD2 Const(VMatrix3x3r const& value) {
  Mat3AD2 result;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      result(r, c) = Const(value[r][c]);
    }
  }
  return result;
}

static Mat3AD2 Eye3() {
  return Const(VEye<3>());
}

static Mat3x4AD2 ToEigen(TransformRT const& trans) {
  Mat3x4AD2 result;
  ROT(result) = Const(ToVMatrix3x3(trans.GetRotation()));
  TRANS(result) = Const(trans.GetTranslation());
  return result;
}

/**
 * @brief Iterates over all dynamic rigid actors in the scene, invoking a callback with per-actor
 * DOF and index information.
 *
 * @details Iterates all actors in the scene via @ref Scene::ForEachActor. For each actor that is
 * non-static and of type @ref ActorType::Rigid, invokes the callback with:
 *   - The actor pointer.
 *   - The cumulative DOF offset in the combined state vector (increments by 6 per actor,
 *     corresponding to @ref RigidSize::kDAll = 3 translation + 3 rotation DOFs).
 *   - The zero-based index among dynamic rigid actors only.
 *   - The zero-based index among all dynamic actors (including articulated, soft, etc.).
 *
 * @param[in] scene The scene to iterate actors from.
 * @param[in] callback Function called for each dynamic rigid actor with the following arguments:
 *   - @c rigidActor: Pointer to the dynamic rigid actor.
 *   - @c dofOffset: Starting index of this actor's 6 DOFs in the combined state vector.
 *   - @c rigidIndex: Zero-based index among dynamic rigid actors (0, 1, 2, ...).
 *   - @c actorIndex: Zero-based index among all dynamic actors in the scene.
 */
static void ForEachDynamicRigidActor(
    Scene* scene,
    std::function<void(Actor*, int, int, int)> const& callback) {
  int dofOffset = 0;
  int rigidIndex = 0;
  int actorIndex = 0;
  scene->ForEachActor([&](Actor* actor) {
    if (!actor->IsStatic() && actor->GetType() == ActorType::Rigid) {
      callback(actor, dofOffset, rigidIndex, actorIndex);
      dofOffset += 6;
      ++rigidIndex;
    }
    if (!actor->IsStatic()) {
      ++actorIndex;
    }
  });
}

/// @brief Returns the number of dynamic rigid actors in the scene.
static int NumDynamicRigidActors(Scene* scene) {
  int numActors = 0;
  ForEachDynamicRigidActor(scene, [&](Actor*, int, int, int) { ++numActors; });
  return numActors;
}

/// @brief Returns the total number of DOFs across all dynamic rigid actors (6 per actor).
static int NumDofs(Scene* scene) {
  return NumDynamicRigidActors(scene) * 6;
}

static void
BuildRigidTransforms(Scene* scene, DynamicArray<Mat3x4AD2>& transforms, VecAD2 deltaState) {
  transforms.clear();
  ForEachDynamicRigidActor(scene, [&](Actor* actor, int dofOffset, int, int) {
    Mat3x4AD2 deltaTrans;
    Mat3x4AD2 trans = ToEigen(actor->GetCenterOfMassTransform(ExpectOK{}));
    // translation
    TRANS(deltaTrans) = TRANS(trans);
    TRANS(deltaTrans) += deltaState.template segment<RigidSize::kDTrans>(dofOffset);
    // rotation: use second-order Taylor expansion: (I + [w] + 0.5 * [w] * [w]) * rot0
    Mat3AD2 w =
        Skew3<AD2>(deltaState.template segment<RigidSize::kDRot>(dofOffset + RigidSize::kDTrans));
    Mat3AD2 deltaR = Eye3() + w + w * w * Const(0.5_r);
    ROT(deltaTrans) = deltaR * ROT(trans);
    // assign to the transforms array
    transforms.emplace_back(deltaTrans);
  });
}

static AD2
BuildRigidMerits(Scene* scene, NdArray<DynamicArray<Mat3x4AD2>, 3> const& transforms, real dt) {
  AD2 merit = Const(0_r);
  auto const& reg = assert_cast<SceneImpl*>(scene)->GetRegistry();
  CSceneGravity const& gravity = reg.ctx<CSceneGravity const>();
  ForEachDynamicRigidActor(scene, [&](Actor* actor, int, int rigidIndex, int) {
    auto const& newPose = transforms[0][rigidIndex];
    auto const& currPose = transforms[1][rigidIndex];
    auto const& oldPose = transforms[2][rigidIndex];
    auto entity = GetEntity(reg, actor->GetHandle(), ExpectOK{});
    auto const& inertia = reg.get<CRigidBodyInertia const>(entity);
    // translational part of inertia energy
    Vec3AD2 accel = TRANS(newPose) - TRANS(currPose) * Const(2_r) + TRANS(oldPose);
    merit += accel.squaredNorm() * Const(inertia.GetMass() / (2_r * dt * dt));
    // rotational part of inertia energy
    Mat3AD2 M = Const(inertia.GetSecondMomentLocal() / (dt * dt));
    Mat3AD2 dRot = ROT(newPose) - ROT(oldPose);
    merit -= (dRot * M * dRot.transpose()).trace() * Const(0.5_r);
    dRot = ROT(newPose) - ROT(currPose);
    merit += (dRot * M * dRot.transpose()).trace();
    // gravity
    merit -= inertia.GetMass() * TRANS(newPose).dot(Const(gravity.accel));
  });
  return merit;
}

// AutoDiffAssembly implementation
AutoDiffAssembly::AutoDiffAssembly(Scene* scene, std::array<StateHandle, 3> states)
    : _scene(scene), _states(states), _n(NumDofs(scene) * 3) {}

void AutoDiffAssembly::Assemble(real dt) {
  // Collect delta states
  int numDofs = NumDofs(_scene);
  ColumnVector<real> deltaState(numDofs * 3);
  deltaState.SetZero();
  _deltaState = InitADVector(deltaState);
  // Collect delta transforms
  for (int stateIndex = 0; stateIndex < 3; ++stateIndex) {
    _scene->RestoreState(_states[stateIndex], false, ExpectOK{});
    BuildRigidTransforms(
        _scene, _transforms[stateIndex], _deltaState.segment(stateIndex * numDofs, numDofs));
  }
  // Collect merits
  _merits = Const(0_r);
  _merits += BuildRigidMerits(_scene, _transforms, dt);
  // Extract gradient (either of these is fine; they should match)
  _grad = _merits.value().derivatives(); // size n
  // Extract Hessian
  _hess.resize(_n, _n);
  for (int j = 0; j < _n; ++j) {
    // y.derivatives()(j) is dy/dx_j as AD1
    // its derivatives()(i) is d/dx_i (dy/dx_j) = H(j,i)
    _hess.row(j) = _merits.derivatives()(j).derivatives().transpose();
  }
}

real AutoDiffAssembly::GetMerit() const {
  return _merits.value().value();
}

ColumnVector<real> AutoDiffAssembly::GetRes() const {
  Vec result = _grad.segment(0, _n / 3);
  return ColumnVectorView<real>(result.data(), isize(result));
}

Matrix<real> AutoDiffAssembly::GetDRes(int d) const {
  Mat result = _hess.block(0, d * _n / 3, _n / 3, _n / 3);
  return MatrixView<real>(
      result.data(), static_cast<int>(result.rows()), static_cast<int>(result.cols()));
}

} // namespace mochi::autodiff

#undef ROT
#undef TRANS
#endif
