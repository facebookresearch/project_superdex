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

#include "mochi_physics_test_fixture.h"

#include <mochi_physics/src/mochi_soft_rom_components.h>
#include <mochi_physics/src/mochi_soft_rom_systems.h>

#include <mochi_core/contact/dmap.h>
#include <mochi_core/utils/rigid_body_size.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <utility>

using namespace mochi;
using namespace mochi::test;
using namespace mochi::dmap;

static Real3 WorldPoint(TransformRT const& rootXf, TransformRT const& localXf, Real3 const& x) {
  return rootXf.TransformPoint(localXf.TransformPoint(x));
}

static void AddEpsRigid(int dof, real eps, TransformRT& xf) {
  if (dof < RigidSize::kDTrans) {
    Real3 t = xf.GetTranslation();
    t[dof] += eps;
    xf.SetTranslation(t);
  } else {
    Real3 inc{};
    inc[dof - RigidSize::kDTrans] = eps;
    xf.SetRotation(Quaternion::FromRotationVector(inc) * xf.GetRotation());
  }
}

class SoftRomJacobianMatch : public test::MochiSceneTestBase {
 public:
  void SetUp() override { // Called just before each test case
    test::MochiSceneTestBase::SetUp();
  }

  void TearDown() override { // Called just after each test case
    test::MochiSceneTestBase::TearDown();
  }
};

// Test to validate that the computation of contact Jacobians of soft and ROM actors don't diverge.
// For an identity map between ROM and soft-actor DoFs, the Jacobians should match.
TEST_F(SoftRomJacobianMatch, SoftRomJacobianMatch) {
  auto& reg = GetRegistry();

  // Create soft actor. This is a cube of size 1, centered at 0.5
  auto&& [unitCubeCoordinates, unitCubeConnectivity] = test::CreateMinimalTetMeshUnitCube();
  TransformRT transform(Quaternion::FromRotationVector(Real3(0.2_r, -0.3_r, 0.5_r)));
  SoftActorParams caparams;
  caparams.shape = _scene->GetContext()->CreateTetMeshShape(
      Flatten(MakeSpan(unitCubeCoordinates)),
      Flatten(MakeSpan(unitCubeConnectivity)),
      ErrorAssert{});
  caparams.worldFromLocal = transform;
  Actor* actor = _scene->CreateSoftActor(caparams, ErrorAssert{});
  entt::entity ent = mochi::GetEntity(reg, actor->GetHandle(), test::ExpectOK{});
  int const numDofs = reg.get<CActorDofInfo const>(ent).dofsSize;

  // Prepare collision points
  ecs::TryInvokeOnEntity(
      &UpdateCollisionSamplePositions<CFemBoundaryDiscretization, TimeStep::Current, kSpaceDim3>,
      reg,
      ent);
  auto const& sample = reg.get<CContactSamples<TimeStep::Current> const>(ent);
  int const numSamples = isize(sample.positions);
  ContactDetectionResult collResult;
  collResult.sampleIndices.resize(numSamples);
  std::iota(collResult.sampleIndices.begin(), collResult.sampleIndices.end(), 0);
  collResult.jacColliderFromWorld.resize(1);
  collResult.jacColliderFromWorld[0] = VEye<3>();

  // Set an identity Jacobian for the ROM
  auto romMatrix = Matrix<real>::Zero(numDofs, numDofs);
  for (int i = 0; i < numDofs; i++) {
    romMatrix(i, i) = 1_r;
  }
  DMapRom::VariantJacobian romJacobian(std::move(romMatrix));

  auto const& discretization = reg.get<CFemBoundaryDiscretization const>(ent);
  discretization.Visit([&](auto const& discretizationImpl) {
    using DiscretizationT = std::decay_t<decltype(discretizationImpl)>;
    using ElementT = typename DiscretizationT::ElementT;

    // Create differentiable maps
    using DQuad = DMapQuad<ElementT>;
    DMapRTConst dtransform(transform);
    DQuad dquad(discretizationImpl.femElements, collResult.jacColliderFromWorld);
    DMapSoft dsoft(0, 0);
    DMapRom drom(0, romJacobian, 0);
    DMap<DQuad, DMapRTConst, DMapSoft> dmapSoft(&dquad, &dtransform, &dsoft);
    DMap<DQuad, DMapRTConst, DMapRom> dmapRom(&dquad, &dtransform, &drom);

    // Get contact Jacobians
    ContactJac jacSoft;
    ContactJac jacRom;
    dmapSoft.GetJac(collResult.sampleIndices, Span(&jacSoft, 1));
    dmapRom.GetJac(collResult.sampleIndices, Span(&jacRom, 1));

    // Jacobians of the ROM actor should be of size numDofs.
    EXPECT_EQ(jacRom.nDoFsState, numDofs);
    EXPECT_EQ(jacSoft.Inds(0).size(), jacSoft.nDoFsState);
    EXPECT_EQ(jacRom.Inds(0).size(), numDofs);
    EXPECT_EQ(jacSoft.nContacts, jacRom.nContacts);
    for (int i = 0; i < numSamples; i++) {
      for (int j = 0; j < numDofs; j++) {
        EXPECT_EQ(jacRom.Inds(i)[j], j);

        // Try to find this DOF in the soft Jacobian
        auto* dofIndexSoft = std::find(jacSoft.Inds(i).begin(), jacSoft.Inds(i).end(), j);

        Real3 jacRomij(jacRom.Jac(i)(0, j), jacRom.Jac(i)(1, j), jacRom.Jac(i)(2, j));
        if (dofIndexSoft == jacSoft.Inds(i).end()) {
          // If not found, ROM Jacobian should be zero
          EXPECT_EQ(jacRomij, Real3(0_r, 0_r, 0_r));
        } else {
          // If found, ROM and soft Jacobians should be equal
          int col = dofIndexSoft - jacSoft.Inds(i).begin();
          Real3 jacSoftij(jacSoft.Jac(i)(0, col), jacSoft.Jac(i)(1, col), jacSoft.Jac(i)(2, col));
          EXPECT_EQ(jacRomij, jacSoftij);
        }
      }
    }
  });
}

TEST(AddRigidContactJacobians, MatchesFiniteDifferenceWithLocalRotation) {
  TransformRT const rootXf(
      Quaternion::FromRotationVector(Real3{0.2_r, -0.3_r, 0.5_r}), Real3{1_r, 2_r, 3_r});
  TransformRT const localXf(
      Quaternion::FromRotationVector(Real3{0.4_r, 0.1_r, -0.2_r}), Real3{0.1_r, 0.2_r, -0.1_r});
  Real3 const x{0.5_r, -0.3_r, 0.7_r};

  entt::registry reg;
  entt::entity const e = reg.create();
  reg.emplace<TagRomActor>(e);
  reg.emplace<CRootTransform>(e, rootXf);
  reg.emplace<CRigidState<TimeStep::Current>>(e, localXf);

  ContactDetectionResult contact;
  contact.ndofs = 1;
  contact.sampleIndices.resize(1);
  contact.sampleIndices[0] = 0;
  contact.posColliding.resize(1);
  contact.posColliding[0] = WorldPoint(rootXf, localXf, x);
  contact.jacWorldFromDofs.resize(1);

  Real3 const softJacSentinel{9_r, 8_r, 7_r};
  contact.jacWorldFromDofs[0].jac[0] = ToSimd(softJacSentinel);

  rom::AddRigidContactJacobians(reg, e, contact);

  ASSERT_EQ(contact.ndofs, RigidSize::kDAll + 1);
  auto const& jacDofs = contact.jacWorldFromDofs[0];

  for (int i = 0; i < contact.ndofs; ++i) {
    EXPECT_EQ(jacDofs.inds[i], i);
  }
  EXPECT_NEAR_TOL(ToReal3(jacDofs.jac[RigidSize::kDAll]), softJacSentinel, 1e-6_r);

  real constexpr kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-5_r : 1e-3_r;
  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-2_r;
  real constexpr kInvTwoEps = 1_r / (2_r * kEps);
  for (int dof = 0; dof < RigidSize::kDAll; ++dof) {
    TransformRT localP = localXf;
    TransformRT localM = localXf;
    AddEpsRigid(dof, kEps, localP);
    AddEpsRigid(dof, -kEps, localM);

    Real3 const fd = (WorldPoint(rootXf, localP, x) - WorldPoint(rootXf, localM, x)) * kInvTwoEps;
    EXPECT_NEAR_TOL(ToReal3(jacDofs.jac[dof]), fd, kTol) << "rigid DoF " << dof;
  }
}
