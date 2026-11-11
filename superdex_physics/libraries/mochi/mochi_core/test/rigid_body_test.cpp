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

#include <mochi_core/test/mochi_test_helpers.h>

#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/rigid_body_utils.h>

#include <gtest/gtest.h>

#include <functional>
#include <limits>
#include <memory>
#include <string>

using namespace mochi;

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

static std::unique_ptr<TetrahedralMesh> CreateMesh_File(std::string const& path) {
  return LoadTetrahedralMesh(path, test::ExpectOK{});
}

static std::unique_ptr<TetrahedralMesh> CreateMesh_Grid(mochi::Int3 dims, mochi::Real3 size) {
  auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitGrid(size, dims);
  return std::make_unique<TetrahedralMesh>(coordinates, connectivity);
}

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

static constexpr real kRoundingRelTol = 1e3_r * std::numeric_limits<real>::epsilon();

static void TestMassProperties(
    std::unique_ptr<TetrahedralMesh>& mesh,
    VMatrix3x3r const* momentOfInertiaExact = nullptr,
    real const density = 1_r,
    real const exactRelTol = kRoundingRelTol) {
  real const volume = mesh->GetTotalMeasure();
  Vec4r const centerOfMass = ToSimd(mesh->GetBarycenter());
  VMatrix3x3r I = ComputeSecondMomentOfInertia(
      mesh->GetNodeCoordinates(), mesh->GetElementConnectivity(), density, centerOfMass);

  // Error
  if (momentOfInertiaExact) {
    real const normExact = Norm3x3(*momentOfInertiaExact);
    EXPECT_NEAR(Norm3x3(I - *momentOfInertiaExact), 0_r, exactRelTol * normExact);
  }

  TriangularMesh boundaryMesh = CreateBoundaryMesh(*mesh);
  Vec4r boundaryCentroid = {};
  real boundaryVolume = {};
  ComputeVolumeAndCentroid(
      boundaryMesh.GetNodeCoordinates(),
      boundaryMesh.GetElementConnectivity(),
      boundaryCentroid,
      boundaryVolume);
  VMatrix3x3r boundaryI = ComputeSecondMomentOfInertia(
      boundaryMesh.GetNodeCoordinates(),
      boundaryMesh.GetElementConnectivity(),
      density,
      boundaryCentroid);

  // The surface integrals agree with the mesh's volume and barycenter, and with the volume
  // integral, to rounding.
  real const mass = density * volume;
  real const radiusOfGyration = Sqrt(Trace3x3(I) / mass);
  EXPECT_NEAR(density * boundaryVolume, mass, kRoundingRelTol * mass);
  EXPECT_NEAR(Norm<3>(boundaryCentroid - centerOfMass), 0_r, kRoundingRelTol * radiusOfGyration);
  EXPECT_NEAR(Norm3x3(boundaryI - I), 0_r, kRoundingRelTol * Norm3x3(I));
}

TEST(RigidBody, MomentOfInertia_Cube) {
  // Analytic
  VMatrix3x3r momentOfInertiaExact;
  real density = 2500_r;
  Vec4r sizes(0.1_r, 0.1_r, 0.1_r);
  ComputeSecondMomentOfInertiaCuboid(density, sizes, momentOfInertiaExact);

  std::unique_ptr<TetrahedralMesh> mesh = CreateMesh_Grid(Int3{5, 5, 5}, ToReal3(sizes));
  TestMassProperties(mesh, &momentOfInertiaExact, density);
}

TEST(RigidBody, MomentOfInertia_Cuboid) {
  // Analytic
  VMatrix3x3r momentOfInertiaExact;
  real density = 2500_r;
  Vec4r sizes(0.05_r, 0.1_r, 0.15_r);
  ComputeSecondMomentOfInertiaCuboid(density, sizes, momentOfInertiaExact);

  std::unique_ptr<TetrahedralMesh> mesh = CreateMesh_Grid(Int3{5, 6, 7}, ToReal3(sizes));
  TestMassProperties(mesh, &momentOfInertiaExact, density);
}

TEST(RigidBody, MomentOfInertia_Sphere) {
  // Analytic
  VMatrix3x3r momentOfInertiaExact;
  real density = 2500_r;
  real radius = 1_r;
  ComputeSecondMomentOfInertiaSphere(density, radius, momentOfInertiaExact);

  std::string filePath = test::GetAssetPath("sphere/icosphere_4subdiv.1.mochi.json");
  std::unique_ptr<TetrahedralMesh> mesh = CreateMesh_File(filePath);

  // The icosphere's inertia differs from the sphere's by the tessellation error.
  TestMassProperties(mesh, &momentOfInertiaExact, density, 5e-2_r);
}

// The Duck and Armadillo meshes used below are not shipped externally.
TEST_IF(MOCHI_INTERNAL, RigidBody, MomentOfInertia_Duck) {
  real density = 2500_r;

  std::string filePath = test::GetAssetPath("duck/duck_fine_mesh.mochi.json");
  std::unique_ptr<TetrahedralMesh> mesh = CreateMesh_File(filePath);

  TestMassProperties(mesh, nullptr, density);
}

TEST_IF(MOCHI_INTERNAL, RigidBody, MomentOfInertia_Armadillo) {
  real density = 2500_r;

  std::string filePath = test::GetAssetPath("armadillo/armadillo_coarse_mesh.mochi.json");
  std::unique_ptr<TetrahedralMesh> mesh = CreateMesh_File(filePath);

  TestMassProperties(mesh, nullptr, density);
}

// The volume, center of mass and second moment of a small part authored far from its mesh origin,
// from its tet mesh and from its boundary, and the area and barycenter of its boundary mesh, are
// accurate to rounding them, up to the error of their double-precision arithmetic.
TEST(RigidBody, MassProperties_FarFromOrigin) {
  auto [coordinates, connectivity] =
      test::CreateMinimalTetMeshUnitGrid(Real3{0.05_r, 0.02_r, 0.01_r}, Int3{16, 16, 16});
  for (Real3& x : coordinates) {
    x += Real3{8_r, 5_r, 3_r};
  }
  TetrahedralMesh const mesh(coordinates, connectivity);
  TriangularMesh const boundaryMesh = CreateBoundaryMesh(mesh);
  Vec4r boundaryCentroid = {};
  real boundaryVolume = {};
  ComputeVolumeAndCentroid(
      boundaryMesh.GetNodeCoordinates(),
      boundaryMesh.GetElementConnectivity(),
      boundaryCentroid,
      boundaryVolume);

  // The rounded coordinates still span a box exactly. The tolerance is the rounding of the results,
  // plus the worst-case error of summing their terms in double: an epsilon per tet, and per
  // boundary face times the cancellation in the boundary integrals, whose terms are measured from
  // the origin.
  Aabb const box = mesh.GetAabb();
  Real3 const size = box.GetSize();
  real const volume = Prod(size);
  real const area = 2_r * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0]);
  Real3 const center = box.GetCenter();
  real const cancellation = Norm(center) * area / volume;
  real const tol = 4_r * std::numeric_limits<real>::epsilon() +
      (StaticCast<real>(mesh.GetNumElements()) +
       StaticCast<real>(boundaryMesh.GetNumElements()) * cancellation) *
          StaticCast<real>(std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(mesh.GetTotalMeasure(), volume, tol * volume);
  EXPECT_NEAR(boundaryVolume, volume, tol * volume);
  EXPECT_NEAR(boundaryMesh.GetTotalMeasure(), area, tol * area);
  EXPECT_NEAR(Norm(mesh.GetBarycenter() - center), 0_r, tol * Norm(center));
  EXPECT_NEAR(Norm(ToReal3(boundaryCentroid) - center), 0_r, tol * Norm(center));
  EXPECT_NEAR(Norm(boundaryMesh.GetBarycenter() - center), 0_r, tol * Norm(center));

  // The box's second moment about its center, per unit density
  Vec4r const sizes = ToSimd(size);
  VMatrix3x3r const secondMoment = VDiagonalMatrix<3>((volume / 12_r) * sizes * sizes);
  for (VMatrix3x3r const& computed :
       {ComputeSecondMomentOfInertia(
            mesh.GetNodeCoordinates(), mesh.GetElementConnectivity(), 1_r, ToSimd(center)),
        ComputeSecondMomentOfInertia(
            boundaryMesh.GetNodeCoordinates(),
            boundaryMesh.GetElementConnectivity(),
            1_r,
            ToSimd(center))}) {
    EXPECT_NEAR(Norm3x3(computed - secondMoment), 0_r, tol * Norm3x3(secondMoment));
  }
}

static void TestReal3(Real3 const& a, Real3 const& b, real tol) {
  real error = Norm(a - b) / Max(Norm(a), Norm(b));
  EXPECT_NEAR(0_r, error, tol);
}

// IntegrateRigidTransformLie and FiniteDifferenceRigidTransformLie are exact inverses, even for
// large rotations, and integration steps compose exactly.
TEST(RigidBody, IntegrateAndFiniteDifferenceRigidTransformLie) {
  TransformRT const transformPrev{
      Quaternion::FromRotationVector(Real3{0.5_r, -1_r, 1_r}), Real3{2_r, -0.5_r, -1_r}};
  Vec4r const pivotLocal = ToSimd(Real3{1_r, 0.5_r, -0.5_r});
  Vec4r const linVel = ToSimd(Real3{-1_r, 2_r, 1_r});
  Vec4r const angVel = ToSimd(Real3{5_r, -5_r, 10_r});
  real constexpr kDt = 0.2_r; // Rotation angle of about 2.4 rad

  TransformRT const transform =
      IntegrateRigidTransformLie(kDt, transformPrev, pivotLocal, linVel, angVel);
  TestReal3(
      ToReal3(transformPrev.TransformPoint(pivotLocal) + linVel * kDt),
      ToReal3(transform.TransformPoint(pivotLocal)),
      kRoundingRelTol);

  Vec4r linVelTest;
  Vec4r angVelTest;
  FiniteDifferenceRigidTransformLie(
      kDt, transform, transformPrev, pivotLocal, linVelTest, angVelTest);
  TestReal3(ToReal3(linVel), ToReal3(linVelTest), kRoundingRelTol);
  TestReal3(ToReal3(angVel), ToReal3(angVelTest), kRoundingRelTol);

  // Two half steps land on the full step.
  TransformRT const halfSteps = IntegrateRigidTransformLie(
      0.5_r * kDt,
      IntegrateRigidTransformLie(0.5_r * kDt, transformPrev, pivotLocal, linVel, angVel),
      pivotLocal,
      linVel,
      angVel);
  TestReal3(transform.GetTranslation(), halfSteps.GetTranslation(), kRoundingRelTol);
  EXPECT_TRUE(NearEqual(transform.GetRotation(), halfSteps.GetRotation(), kRoundingRelTol));
}

// Wrapper of RigidBodyVel::EvalTimeSteppedRotation()
static VMatrix3x3r
EvalTimeSteppedRotation(VMatrix3x3r const& R, RigidBodyVel const& vel, real dtStage) {
  return vel.EvalTimeSteppedRotation(R, dtStage);
}

// Alternative implementation of EvalTimeSteppedRotation(), by integrating velocity in Lie algebra
// and performing group composition.
static VMatrix3x3r
EvalTimeSteppedRotationAccurate(VMatrix3x3r const& R, RigidBodyVel const& vel, real dtStage) {
  return Dot3x3(Rodrigues(dtStage * vel.GetOmegaAndVSym().first), R);
}

// Wrapper of RigidBodyVel::SetFromFiniteDifferencePose().
static void EvalFiniteDifferenceRotationVelocity(
    Quaternion const& qOld,
    Quaternion const& qNew,
    real dtStage,
    RigidBodyVel& outVel) {
  outVel.SetFromFiniteDifferencePose(TransformRT{qOld}, TransformRT{qNew}, dtStage);
}

// Alternative implementation of EvalFiniteDifferenceRotationVelocity(), finite-differencing the
// relative rotation vector.
static void EvalFiniteDifferenceRotationVelocityAccurate(
    Quaternion const& qOld,
    Quaternion const& qNew,
    real dtStage,
    RigidBodyVel& outVel) {
  Vec4r linVel, angVel;
  FiniteDifferenceRigidTransformLie(
      dtStage, TransformRT{qNew}, TransformRT{qOld}, SimdZero(), linVel, angVel);
  outVel.SetOmega(angVel);
  outVel.UpdateVSymIfDirty(dtStage);
}

TEST(RigidBodyUtils, EvalTimeSteppedRotation) {
  // Test data
  VMatrix3x3r R = Rodrigues(Vec4r{0.8_r, -1.2_r, 0.5_r});
  RigidBodyVel vel;

  auto test = [&](real dt,
                  std::function<VMatrix3x3r(VMatrix3x3r const&, RigidBodyVel const&, real)> func) {
    // Evaluate the time-stepped rotation
    VMatrix3x3r Rnew = func(R, vel, dt);

    // Check the determinant
    real det = Determinant3x3(AsMatrixView(Rnew));
    EXPECT_TRUE(NearEqual(det, 1_r, kRoundingRelTol));
  };

  real dt = 1e-3_r;
  vel.SetOmega({1.1_r, -0.7_r, 0.8_r});
  vel.UpdateVSymIfDirty(dt);

  // Small time-step approximate function
  test(dt, EvalTimeSteppedRotation);

  // Small time-step accurate function
  test(dt, EvalTimeSteppedRotationAccurate);

  dt = 3e-1_r;
  vel.SetOmega({1.1_r, -0.7_r, 0.8_r});
  vel.UpdateVSymIfDirty(dt);

  // Large time-step approximate function
  test(dt, EvalTimeSteppedRotation);

  // Large time-step accurate function
  test(dt, EvalTimeSteppedRotationAccurate);
}

TEST(RigidBodyUtils, EvalFiniteDifferenceRotationVelocity) {
  // Test data
  VMatrix3x3r R = Rodrigues(Vec4r{0.8_r, -1.2_r, 0.5_r});
  RigidBodyVel vel;
  VMatrix3x3r Rother = Rodrigues(Vec4r{0.7_r, -1.3_r, 0.6_r});

  auto test =
      [&](std::function<VMatrix3x3r(VMatrix3x3r const&, RigidBodyVel const&, real)> evalRotation,
          std::function<void(Quaternion const&, Quaternion const&, real, RigidBodyVel&)>
              evalVelocity,
          real dt,
          real tol) {
        // Perform round-trip conversion, first time-step rotation, then finite-difference velocity.
        auto Rnew = evalRotation(R, vel, dt);
        RigidBodyVel velTest;
        evalVelocity(QuaternionFromMatrix(R), QuaternionFromMatrix(Rnew), dt, velTest);

        // Validate
        TestReal3(
            ToReal3(vel.GetOmegaAndVSym().first),
            ToReal3(velTest.GetOmegaAndVSym().first),
            kRoundingRelTol);

        // Perform round-trip conversion, first finite-difference velocity, then time-step rotation.
        // This operation has a round-trip error for approximate functions, because evalVelocity
        // uses the new rotation and evalRotation uses the old rotation.
        RigidBodyVel velFD;
        evalVelocity(QuaternionFromMatrix(R), QuaternionFromMatrix(Rother), dt, velFD);
        auto Rtest = evalRotation(R, velFD, dt);

        // Validate
        EXPECT_TRUE(NearEqual(ToNdArray3x3(Rother), ToNdArray3x3(Rtest), tol));
      };

  real dt = 1e-2_r;
  vel.SetOmega({1.1_r, -0.7_r, 0.8_r});
  vel.UpdateVSymIfDirty(dt);

  // Small time-step approximate functions
  test(EvalTimeSteppedRotation, EvalFiniteDifferenceRotationVelocity, dt, 3e-2_r);

  // Small time-step accurate functions
  test(
      EvalTimeSteppedRotationAccurate,
      EvalFiniteDifferenceRotationVelocityAccurate,
      dt,
      kRoundingRelTol);

  dt = 3e-1_r;
  vel.SetOmega({1.1_r, -0.7_r, 0.8_r});
  vel.UpdateVSymIfDirty(dt);

  // Large time-step approximate functions, whose round-trip error is O(|Rother - R|^2).
  test(EvalTimeSteppedRotation, EvalFiniteDifferenceRotationVelocity, dt, 3e-2_r);

  // Large time-step accurate functions
  test(
      EvalTimeSteppedRotationAccurate,
      EvalFiniteDifferenceRotationVelocityAccurate,
      dt,
      kRoundingRelTol);
}

TEST(RigidBodyUtils, GetFiniteStepRotationVector) {
  real constexpr kDt = 1e-2_r;
  Quaternion const qOld = Quaternion::FromRotationVector(Vec4r{0.8_r, -1.2_r, 0.5_r});

  // Large step of 0.8 rad: finite-differencing the rotation and recovering the step round-trips,
  // forward and backward.
  Vec4r const step{0.48_r, 0_r, -0.64_r};
  RigidBodyVel vel;
  vel.SetFromFiniteDifferencePose(
      TransformRT{qOld}, TransformRT{Quaternion::FromRotationVector(step) * qOld}, kDt);
  TestReal3(ToReal3(vel.GetFiniteStepRotationVector(kDt)), ToReal3(step), kRoundingRelTol);
  TestReal3(
      ToReal3(vel.GetFiniteStepRotationVector(-kDt)), Real3{-0.48_r, 0_r, 0.64_r}, kRoundingRelTol);

  // Small steps approach omega * dt, and zero velocity gives a zero step.
  Vec4r const omega{1e-3_r, -2e-3_r, 1.5e-3_r};
  vel.SetOmega(omega);
  EXPECT_TRUE(
      NearEqual(ToReal3(vel.GetFiniteStepRotationVector(kDt)), ToReal3(kDt * omega), 1e-12_r));
  vel.SetZero();
  EXPECT_TRUE(NearEqual(ToReal3(vel.GetFiniteStepRotationVector(kDt)), Real3{}, 0_r));
}

TEST(RigidBodyUtils, SetOmegaWithZeroVSym) {
  real constexpr kDt = 3e-1_r;
  RigidBodyVel vel;
  vel.SetOmega({1.1_r, -0.7_r, 0.8_r});
  vel.UpdateVSymIfDirty(kDt); // Nonzero vsym, to be cleared below

  Vec4r const omega{0.3_r, 0.2_r, -0.5_r};
  vel.SetOmegaWithZeroVSym(omega);
  EXPECT_FALSE(vel.IsVSymDirty());

  // A clean velocity is not modified by the vsym update.
  vel.UpdateVSymIfDirty(kDt);
  auto const [outOmega, outVSym] = vel.GetOmegaAndVSym();
  EXPECT_TRUE(NearEqual(ToReal3(omega), ToReal3(outOmega), 0_r));
  EXPECT_TRUE(NearEqual(ToNdArray3x3(SimdSymToFull(outVSym)), Matrix3x3r{}, 0_r));
}

TEST(RigidBodyUtils, RotateInertia) {
  auto q = Quaternion::FromRotationVector(Vec4r{0.3_r, -1.4_r, 0.5_r});
  real m = 2.1_r;
  auto matReal = RotateInertia(m, q);
  auto matVec4 = RotateInertia(Vec4r(m), q);
  auto matVMatrix3x3 = RotateInertia(VDiagonalMatrix<3>(m), q);
  EXPECT_NEAR_TOL(matVec4, matReal, kRoundingRelTol * m);
  EXPECT_NEAR_TOL(matVMatrix3x3, matReal, kRoundingRelTol * m);
  Vec4r vecM{1.8_r, 2.1_r, 1.1_r};
  matVec4 = RotateInertia(vecM, q);
  matVMatrix3x3 = RotateInertia(VDiagonalMatrix<3>(vecM), q);
  EXPECT_NEAR_TOL(matVMatrix3x3, matVec4, kRoundingRelTol * HMax<3>(vecM));
}

TEST(RigidBodyUtils, IsMomentOfInertiaValid) {
  // Diagonal MOI tensors: valid, singular-but-valid, and triangle-inequality failures.
  EXPECT_TRUE(IsMomentOfInertiaValid(Real6{1_r, 0_r, 0_r, 2_r, 0_r, 3_r}));
  EXPECT_TRUE(IsMomentOfInertiaValid(Real6{1_r, 0_r, 0_r, 1_r, 0_r, 0_r}));
  EXPECT_FALSE(IsMomentOfInertiaValid(Real6{1_r, 0_r, 0_r, 1_r, 0_r, 2.001_r}));
  EXPECT_TRUE(IsMomentOfInertiaValid(Real6{1_r, 0_r, 0_r, 1_r, 0_r, 2.001_r}, 1e-2_r));
  EXPECT_FALSE(IsMomentOfInertiaValid(Real6{1_r, 0_r, 0_r, 1_r, 0_r, 3_r}));

  // Non-diagonal tensors exercise eigendecomposition and upper-triangle mapping.
  EXPECT_TRUE(IsMomentOfInertiaValid(Real6{1.5_r, -0.5_r, 0_r, 1.5_r, 0_r, 2.5_r}));
  EXPECT_FALSE(IsMomentOfInertiaValid(Real6{1_r, 5_r, 0_r, 1_r, 0_r, 1_r}));

  // Rotated slender bodies: a repeated principal moment and a tight triangle inequality.
  auto rng = RandomGenerator(42);
  for (real const axialMoment : {0_r, 2_r * kMomentOfInertiaValidationRelTol}) {
    for (int i = 0; i < 1000; ++i) {
      Real3 rotVector;
      SetRandom(rng, -kPI, kPI, rotVector);
      VMatrix3x3r const I =
          RotateInertia(Vec4r{1_r, 1_r, axialMoment}, Quaternion::FromRotationVector(rotVector));
      EXPECT_TRUE(
          IsMomentOfInertiaValid(Real6{I[0][0], I[0][1], I[0][2], I[1][1], I[1][2], I[2][2]}));
    }
  }

  // Non-finite tensor entries are rejected.
  EXPECT_FALSE(IsMomentOfInertiaValid(Real6{1_r, 0_r, 0_r, 1_r, 0_r, kInf}));
}

TEST(RigidBodyInertia, SetDensityIdempotentAndLossless) {
  real const density = 1732.050808_r;
  Vec4r const com = {0.141421_r, -0.273861_r, 0.618034_r};
  real const volume = 0.00314159_r;
  VMatrix3x3r const moi = {
      Vec4r{7.07107e-6_r, -1.41421e-9_r, 2.23607e-7_r},
      Vec4r{-1.41421e-9_r, 3.14159e-6_r, -1.73205e-9_r},
      Vec4r{2.23607e-7_r, -1.73205e-9_r, 2.71828e-6_r}};

  RigidBodyInertia inertia(com, volume, moi, density);

  // Constructor preserves all properties.
  EXPECT_EQ(density, inertia.GetDensity());
  EXPECT_EQ(moi, inertia.GetMomentOfInertiaLocal());

  // Snapshot all properties after construction.
  auto const mass0 = inertia.GetMass();
  auto const mtwo0 = inertia.GetSecondMomentLocal();

  // Cycle through several densities.
  for (real d : {1414.21356_r, 271.828183_r, 3141.59265_r, 1618.03399_r, 577.21567_r}) {
    inertia.SetDensity(d);
  }

  // Restore construction density. All properties must be equal.
  inertia.SetDensity(density);
  EXPECT_EQ(density, inertia.GetDensity());
  EXPECT_EQ(mass0, inertia.GetMass());
  EXPECT_EQ(moi, inertia.GetMomentOfInertiaLocal());
  EXPECT_EQ(mtwo0, inertia.GetSecondMomentLocal());

  // Shuffled path to a non-construction density equals direct path.
  real const target = 1414.21356_r;
  inertia.SetDensity(3141.59265_r);
  inertia.SetDensity(577.21567_r);
  inertia.SetDensity(target);

  RigidBodyInertia reference(com, volume, moi, density);
  reference.SetDensity(target);

  EXPECT_EQ(reference.GetDensity(), inertia.GetDensity());
  EXPECT_EQ(reference.GetMass(), inertia.GetMass());
  EXPECT_EQ(reference.GetMomentOfInertiaLocal(), inertia.GetMomentOfInertiaLocal());
  EXPECT_EQ(reference.GetSecondMomentLocal(), inertia.GetSecondMomentLocal());
}

// A rigid body without a volume has mass and inertia, but no density.
TEST(RigidBodyInertia, WithoutVolume) {
  Vec4r const com = {0.2_r, 0_r, 0_r};
  VMatrix3x3r const moi = {
      Vec4r{0.01_r, 0_r, 0_r}, Vec4r{0_r, 0.02_r, 0_r}, Vec4r{0_r, 0_r, 0.03_r}};

  auto const expectInertia = [](RigidBodyInertia const& inertia,
                                real expectedMass,
                                Vec4r const& expectedCom,
                                VMatrix3x3r const& expectedMoi) {
    EXPECT_EQ(expectedMass, inertia.GetMass());
    EXPECT_EQ(ToReal4(expectedCom), ToReal4(inertia.GetCenterOfMassLocal()));
    EXPECT_EQ(expectedMoi, inertia.GetMomentOfInertiaLocal());
    EXPECT_EQ(MomentOfInertiaToSecondMoment(expectedMoi), inertia.GetSecondMomentLocal());
    EXPECT_EQ(0_r, inertia.GetVolume());
    EXPECT_EQ(0_r, inertia.GetDensity());
  };

  RigidBodyInertia inertia(com, moi, 2_r);
  expectInertia(inertia, 2_r, com, moi);

  Vec4r const newCom = {0_r, 0.1_r, 0_r};
  VMatrix3x3r const newMoi = moi * 3_r;
  inertia.SetInertiaProperties(3_r, newCom, newMoi);
  expectInertia(inertia, 3_r, newCom, newMoi);

  RigidBodyInertia const massless(Vec4r{}, VMatrix3x3r{}, 0_r);
  expectInertia(massless, 0_r, Vec4r{}, VMatrix3x3r{});
}
