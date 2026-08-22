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

#include <mochi_physics/src/mochi_rod_pose.h>

using namespace mochi;
using namespace mochi::test;
using namespace mochi::constraint_test;

/********************************************************************************
  RodElementRotationToRigidConstraint
********************************************************************************/
namespace {

struct RodElementRotationToRigidData {
  static bool constexpr kHasDifferentiableTarget = true;
  static bool constexpr kHasMixedLinks = false;
  TimeStepPair<Quaternion> r = {};
  TimeStepPair<Quaternion> target = {};

  // Rod topology configuration (set by fixture SetUp before InitBase)
  int constrainedElement = 1;
  bool isClosedLoop = false;

  template <TimeStep kTimeStep>
  void InitState(entt::registry& reg, Span<entt::entity const> entities) {
    SetRigidRotation<kTimeStep>(reg, entities[0], r[kTimeStep]);

    // Reset rod to natural configuration: zero displacements and reference frame axes
    auto& rodPose = reg.get<CRodPose<kTimeStep>>(entities[1]);
    rodPose.value.displacements.SetZero();
    auto const& refAxes = reg.get<CReferenceElementFrameAxes const>(entities[1]);
    for (int i = 0; i < isize(rodPose.value.frameAxes); ++i) {
      rodPose.value.frameAxes[i] = refAxes.axes[i];
    }
  }

  template <TimeStep kTimeStep>
  void AddToState(
      entt::registry& reg,
      Span<entt::entity const> entities,
      int idxi,
      real di,
      int idxj,
      real dj) {
    // First 3 indices (0, 1, 2) are rigid rotation DOFs
    // Remaining 8 indices (3-10) are rod element DOFs (Lie algebra perturbation)

    auto handleRigidDof = [&](int constraintDofIdx, real delta) -> bool {
      if (constraintDofIdx < 3) {
        AddToRigidState<kTimeStep>(reg, entities[0], constraintDofIdx + 3, delta);
        return true;
      }
      return false;
    };

    bool const rigidI = handleRigidDof(idxi, di);
    bool const rigidJ = handleRigidDof(idxj, dj);

    if (rigidI && rigidJ) {
      return;
    }

    // Build a DOF delta vector and apply via ApplyLieDeltaToPose to keep
    // displacements and frame axes consistent.
    auto& rodPose = reg.get<CRodPose<kTimeStep>>(entities[1]);
    auto const& mesh = reg.get<CPolylineMesh const>(entities[1]);
    int const numDofs = isize(rodPose.value.displacements);

    ColumnVector<real> dofDelta = ColumnVector<real>::Zero(numDofs);

    auto addRodDelta = [&](int constraintDofIdx, real delta) {
      if (constraintDofIdx >= 3 && delta != 0_r) {
        int const elementLocalDofIdx = constraintDofIdx - 3; // 0-7
        int const localNode = elementLocalDofIdx / fem::kNumRodFields; // 0 or 1
        int const fieldIdx = elementLocalDofIdx % fem::kNumRodFields; // 0-3
        Int2 const elementNodes = mesh.ElementNodes(constrainedElement);
        int const globalNode = elementNodes[localNode];
        int const globalRodDofIdx = fem::kNumRodFields * globalNode + fieldIdx;
        dofDelta[globalRodDofIdx] = delta;
      }
    };

    if (!rigidI) {
      addRodDelta(idxi, di);
    }
    if (!rigidJ) {
      addRodDelta(idxj, dj);
    }

    int const numElements = isize(rodPose.value.frameAxes);
    ColumnVector<real> outDisp(numDofs);
    DynamicArray<Real3> outAxes(numElements);

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
    // Use the actors that were already created in SetUp()
    entt::entity rigidEntity = entities[0];
    entt::entity rodEntity = entities[1];

    auto rigidActor = GetActorHandle(rigidEntity, scene->GetHandle());
    auto rodActor = GetActorHandle(rodEntity, scene->GetHandle());

    // Keep rod at reference configuration (zero displacements, reference frame axes)
    auto& currPose = reg.get<CRodPose<TimeStep::Current>>(rodEntity);
    currPose.value.displacements.SetZero();
    auto const& refAxes = reg.get<CReferenceElementFrameAxes const>(rodEntity);
    for (int i = 0; i < isize(currPose.value.frameAxes); ++i) {
      currPose.value.frameAxes[i] = refAxes.axes[i];
    }

    // Set initial rigid rotation to identity and create constraint
    // The constraint will compute qr = qRigid^-1 * qRod from this initial state
    r[TimeStep::Current] = Quaternion::Identity();
    SetRigidRotation<TimeStep::Current>(reg, rigidEntity, r[TimeStep::Current]);

    // Use identity reference frame for simplicity
    Quaternion q0 = Quaternion::Identity();

    // Create the constraint - this computes and stores qr (target rotation)
    RodElementRotationToRigidConstraintParams conParams;
    conParams.rigidActor = rigidActor;
    conParams.rodActor = rodActor;
    conParams.elementIndex = constrainedElement;
    conParams.refFrameRotVec = q0.ToRotationVector();
    conParams.stiffness = stiffness;
    conParams.damping = damping;
    conParams.saturation = saturation;

    auto const* constraint =
        scene->CreateRodElementRotationToRigidConstraint(conParams, ExpectOK{});

    // Set actor targets
    auto& info = reg.get<CConstraintInfo>(GetEntity(reg, constraint->GetHandle(), ExpectOK{}));
    info.actorTargets = {{0, 1, 2}, {}};

    // Now adjust rigid rotation to achieve desired cVal
    // Constraint equation: qd = (qRigid * qr * q0)^-1 * qRod * q0
    // With q0 = identity:
    // qd = (qRigid * qr)^-1 * qRod = qr^-1 * qRigid^-1 * qRod
    // Solving for qRigid: qRigid^-1 = qr * qd * qRod^-1
    // Therefore: qRigid = qRod * qd^-1 * qr^-1

    // Get qr (target rotation) from the constraint. In the constraint equation
    // qd = (qRigid * qr * q0)^-1 * qRod * q0, "qr" is the stored target rotation.
    auto const constraintEntity = GetEntity(reg, constraint->GetHandle(), ExpectOK{});
    auto const& constraintTarget =
        reg.get<CConstraintTarget<Quaternion, TimeStep::Current> const>(constraintEntity);
    Quaternion qr = constraintTarget.value;

    // Compute rod element rotation for straight rod (all DOFs zero)
    // Need to get the rod's world-from-local transform and element geometry
    auto const& rodWfl = reg.get<CRootTransform const>(rodEntity).worldFromLocal;
    auto const& polylineMesh = reg.get<CPolylineMesh const>(rodEntity);
    Int2 const en = polylineMesh.ElementNodes(constrainedElement);

    // Get element node positions and element frame axes
    Vec4r X0 = ToSimd(polylineMesh.nodes[en[0]]);
    Vec4r X1 = ToSimd(polylineMesh.nodes[en[1]]);
    auto const& rodPose = reg.get<CRodPose<TimeStep::Current> const>(rodEntity);
    Vec4r frameAxis = Normalize<3>(ToSimd(rodPose.value.frameAxes[constrainedElement]));

    // Rod element DOFs are all zero (straight configuration)
    std::array<real, 8> rodDofs = {};

    // Compute rod element rotation using the same method as the constraint
    VMatrix3x3r rotRodMat;
    fem::ComputeRodElementRotationLocal(X0, X1, frameAxis, rodDofs, rotRodMat, nullptr);
    Quaternion rotRodLocal = Normalize(QuaternionFromMatrix(rotRodMat));
    Quaternion qRod = rodWfl.GetRotation() * rotRodLocal;

    // Store the target values for InitTarget to restore (matches what
    // InitConstraint_RodElementRotationToRigid stored: rotRigid^-1 * rotRod at creation time,
    // with rotRigid = identity).
    target[TimeStep::Current] = qRod;
    target[TimeStep::StageStart] = qRod;

    // Set rigid rotation to achieve cVal at current time
    Quaternion qd = Quaternion::FromRotationVector(cVal);
    r[TimeStep::Current] = Normalize(qRod * qd.GetConjugate() * qr.GetConjugate());
    SetRigidRotation<TimeStep::Current>(reg, rigidEntity, r[TimeStep::Current]);

    // Set rigid rotation to achieve cVal - dtStage * dCVal at stage start
    Real3 cVal0 = cVal - dtStage * dCVal;
    Quaternion qd0 = Quaternion::FromRotationVector(cVal0);
    r[TimeStep::StageStart] = Normalize(qRod * qd0.GetConjugate() * qr.GetConjugate());
    SetRigidRotation<TimeStep::StageStart>(reg, rigidEntity, r[TimeStep::StageStart]);

    // Store entity handles in the entities array
    const_cast<entt::entity&>(entities[0]) = rigidEntity;
    const_cast<entt::entity&>(entities[1]) = rodEntity;

    return constraint->GetHandle();
  }
};

class ConstraintRodElementRotationToRigid
    : public ConstraintTestBaseT<RodElementRotationToRigidData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 11; // 3 rigid rotation DOFs + 8 rod element DOFs
    _targetSize = 3; // 3 target rotation DOFs

    // Create rigid actor
    RigidActorParams rigidParams;
    rigidParams.shape = GetUnitCubeShape(_scene->GetContext());
    rigidParams.colliderType = ColliderType::None;
    auto* rigidActor = _scene->CreateRigidActor(rigidParams, ExpectOK{});
    _entitiesActors.push_back(GetEntity(rigidActor));

    // Create rod actor with 2 elements (3 nodes)
    using namespace mochi::experimental;
    DynamicArray<Real3> nodes;
    DynamicArray<Real3> frameAxes;
    int const numElements = 2;
    int const numNodes = numElements + 1;
    real const length = 1_r;

    for (int i = 0; i < numNodes; ++i) {
      real t = static_cast<real>(i) / static_cast<real>(numElements);
      nodes.push_back(Real3{t * length, 0_r, 0_r});
      if (i < numElements) {
        frameAxes.push_back(Real3{0_r, 1_r, 0_r});
      }
    }

    RodActorParams rodParams;
    rodParams.shape = CreatePolylineShape(
        _scene->GetContext(), nodes, frameAxes, /*isClosedLoop=*/false, ExpectOK{});

    auto* rodActor = CreateRodActor(_scene, rodParams, ExpectOK{});
    auto actorEntity = GetEntity(rodActor->GetHandle());
    _entitiesActors.push_back(actorEntity);
    auto& reg = GetRegistry();
    reg.emplace<CDiffInputOffset>(actorEntity);
    reg.emplace<CActorDiffInputInfo>(actorEntity);

    // Configure constraint data for open rod: constrain last element (element 1)
    _constraintData.constrainedElement = numElements - 1;
    _constraintData.isClosedLoop = false;

    InitBase();
  }
};

class ConstraintRodElementRotationToRigidClosedLoop
    : public ConstraintTestBaseT<RodElementRotationToRigidData> {
 public:
  void SetUp() override {
    ConstraintTestBase::SetUp();

    _conSize = 3;
    _supSize = 11; // 3 rigid rotation DOFs + 8 rod element DOFs
    _targetSize = 3; // 3 target rotation DOFs

    // Create rigid actor
    RigidActorParams rigidParams;
    rigidParams.shape = GetUnitCubeShape(_scene->GetContext());
    rigidParams.colliderType = ColliderType::None;
    auto* rigidActor = _scene->CreateRigidActor(rigidParams, ExpectOK{});
    _entitiesActors.push_back(GetEntity(rigidActor));

    // Create closed-loop rod actor with 4 nodes (4 elements) forming a square
    using namespace mochi::experimental;
    int constexpr kNumNodes = 4;
    real constexpr kSideLength = 0.5_r;

    DynamicArray<Real3> nodes = {
        Real3{0_r, 0_r, 0_r},
        Real3{kSideLength, 0_r, 0_r},
        Real3{kSideLength, 0_r, kSideLength},
        Real3{0_r, 0_r, kSideLength},
    };
    // NOTE: DynamicArray fill constructor for Real3 is miscompiled by MSVC in double precision,
    // leading to a zero vector for the last element. Filling with an explicit for loop to work
    // around it.
    DynamicArray<Real3> frameAxes(kNumNodes);
    for (int i = 0; i < kNumNodes; ++i) {
      frameAxes[i] = Real3{0_r, 1_r, 0_r};
    }

    RodActorParams rodParams;
    rodParams.shape = CreatePolylineShape(
        _scene->GetContext(), nodes, frameAxes, /*isClosedLoop=*/true, ExpectOK{});

    auto* rodActor = CreateRodActor(_scene, rodParams, ExpectOK{});
    auto actorEntity = GetEntity(rodActor->GetHandle());
    _entitiesActors.push_back(actorEntity);
    auto& reg = GetRegistry();
    reg.emplace<CDiffInputOffset>(actorEntity);
    reg.emplace<CActorDiffInputInfo>(actorEntity);

    // Configure constraint data for closed-loop rod: constrain closing element (element
    // kNumNodes-1)
    _constraintData.constrainedElement = kNumNodes - 1;
    _constraintData.isClosedLoop = true;

    InitBase();
  }
};

} // namespace

TEST_F(ConstraintRodElementRotationToRigid, Test) {
  RunAllTests();
}

TEST_F(ConstraintRodElementRotationToRigidClosedLoop, Test) {
  RunAllTests();
}
