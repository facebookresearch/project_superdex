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

#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/rigid_body_utils.h>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

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

static void TestMomentOfInertia(
    std::unique_ptr<TetrahedralMesh>& mesh,
    VMatrix3x3r const* momentOfInertiaExact = nullptr,
    real const density = 1_r) {
  using ElementT = tetrahedral::Pk3DElement<1, 4>;

  int numEle = mesh->GetNumElements();
  auto coordinates = mesh->GetNodeCoordinates();
  auto connectivity = mesh->GetElementConnectivity();
  std::vector<ElementT> elements;
  elements.reserve(numEle);
  for (int i = 0; i < numEle; ++i) {
    elements.emplace_back(int(i), coordinates, connectivity, tetrahedral::kTetrahedralQuadrature4);
  }
  real mass = {};
  Vec4r comLocal = {};
  ComputeCenterOfMassFem(Span<ElementT const>(elements), density, comLocal, mass);
  VMatrix3x3r I =
      ComputeSecondMomentOfInertiaFem(Span<ElementT const>(elements), density, comLocal);

  // Error
  real normExact = {};
  if (momentOfInertiaExact) {
    normExact = Norm3x3(*momentOfInertiaExact);
    EXPECT_NEAR(Norm3x3(I - *momentOfInertiaExact), 0_r, 5e-2 * normExact);
  }

  TriangularMesh boundaryMesh = CreateBoundaryMesh(*mesh);

  using BoundaryElementT = triangular::Pk2DElement<1>;
  std::vector<BoundaryElementT> boundaryElements;
  int numBoundaryElements = boundaryMesh.GetNumElements();
  boundaryElements.reserve(numBoundaryElements);
  for (int i = 0; i < numBoundaryElements; ++i) {
    boundaryElements.emplace_back(
        i, boundaryMesh.GetNodeCoordinates(), boundaryMesh.GetElementConnectivity());
  }
  Vec4r boundaryComLocal = {};
  real boundaryMass = {};
  ComputeCenterOfMassFem(
      Span<BoundaryElementT const>(boundaryElements), density, boundaryComLocal, boundaryMass);
  VMatrix3x3r boundaryI = ComputeSecondMomentOfInertiaFem(
      Span<BoundaryElementT const>(boundaryElements), density, boundaryComLocal);
  if (momentOfInertiaExact) {
    EXPECT_NEAR(Norm3x3(boundaryI - *momentOfInertiaExact), 0_r, 5e-2 * normExact);
  }

  if (!momentOfInertiaExact) {
    normExact = Norm3x3(I);
    EXPECT_NEAR(Norm3x3(boundaryI - I), 0_r, 1e-2 * normExact);
  }
}

TEST(RigidBody, MomentOfInertia_Cube) {
  // Analytic
  VMatrix3x3r momentOfInertiaExact;
  real density = 2500_r;
  Vec4r sizes(0.1_r, 0.1_r, 0.1_r);
  ComputeSecondMomentOfInertiaCuboid(density, sizes, momentOfInertiaExact);

  std::unique_ptr<TetrahedralMesh> mesh = CreateMesh_Grid(Int3{5, 5, 5}, ToReal3(sizes));
  TestMomentOfInertia(mesh, &momentOfInertiaExact, density);
}

TEST(RigidBody, MomentOfInertia_Cuboid) {
  // Analytic
  VMatrix3x3r momentOfInertiaExact;
  real density = 2500_r;
  Vec4r sizes(0.05_r, 0.1_r, 0.15_r);
  ComputeSecondMomentOfInertiaCuboid(density, sizes, momentOfInertiaExact);

  std::unique_ptr<TetrahedralMesh> mesh = CreateMesh_Grid(Int3{5, 6, 7}, ToReal3(sizes));
  TestMomentOfInertia(mesh, &momentOfInertiaExact, density);
}

TEST(RigidBody, MomentOfInertia_Sphere) {
  // Analytic
  VMatrix3x3r momentOfInertiaExact;
  real density = 2500_r;
  real radius = 1_r;
  ComputeSecondMomentOfInertiaSphere(density, radius, momentOfInertiaExact);

  // FEM approximation
  std::string filePath = test::GetAssetPath("sphere/icosphere_4subdiv.1.mochi.json");
  std::unique_ptr<TetrahedralMesh> mesh = CreateMesh_File(filePath);

  TestMomentOfInertia(mesh, &momentOfInertiaExact, density);
}

// The Duck and Armadillo meshes used below are not shipped externally.
TEST_IF(MOCHI_INTERNAL, RigidBody, MomentOfInertia_Duck) {
  real density = 2500_r;

  std::string filePath = test::GetAssetPath("duck/duck_fine_mesh.mochi.json");
  std::unique_ptr<TetrahedralMesh> mesh = CreateMesh_File(filePath);

  TestMomentOfInertia(mesh, nullptr, density);
}

TEST_IF(MOCHI_INTERNAL, RigidBody, MomentOfInertia_Armadillo) {
  real density = 2500_r;

  std::string filePath = test::GetAssetPath("armadillo/armadillo_coarse_mesh.mochi.json");
  std::unique_ptr<TetrahedralMesh> mesh = CreateMesh_File(filePath);

  TestMomentOfInertia(mesh, nullptr, density);
}

static void TestReal3(Real3 const& a, Real3 const& b, real tol) {
  real error = Norm(a - b) / Max(Norm(a), Norm(b));
  EXPECT_NEAR(0_r, error, tol);
}

TEST(RigidBody, ComputeRigidVelocityWorldSpace) {
  // Define an initial transform
  TransformRT transformPrev{
      Quaternion::FromRotationVector(Real3{0.5_r, -1_r, 1_r}), Real3{2_r, -0.5_r, -1_r}};

  // Define the position of the pivot in local coordinates
  Real3 pivotLocal{1_r, 0.5_r, -0.5_r};

  // Define linear and angular velocity in global coordinates
  Real3 linVel{-1_r, 2_r, 1_r};
  Real3 angVel{0.5_r, -0.5_r, 1_r};

  // Compute the old and new positions of the pivot
  real dt = 1e-3_r;
  Real3 pivotGlobalPrev = transformPrev.TransformPoint(pivotLocal);
  Real3 pivotGlobal = pivotGlobalPrev + linVel * dt;

  // Compute the new rotation by integrating the angular velocity
  Quaternion rotation = Quaternion::FromRotationVector(angVel * dt) * transformPrev.GetRotation();

  // Compute the new translation based on the new pivot position
  Real3 translation = pivotGlobal - rotation * pivotLocal;

  // Define the new transform
  TransformRT transform{rotation, translation};

  // Get the velocities of the pivot in local coordinates
  Vec4r linVelTest;
  Vec4r angVelTest;
  ComputeRigidVelocityWorldSpace(
      dt, transform, transformPrev, ToSimd(pivotLocal), linVelTest, angVelTest);

  real tolerance = 100_r * kDefaultNearEqualEpsilon<real>;
  TestReal3(linVel, ToReal3(linVelTest), tolerance);
  TestReal3(angVel, ToReal3(angVelTest), tolerance);
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
  outVel.SetOmega((qNew * qOld.GetConjugate()).VToRotationVector() / dtStage);
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
    EXPECT_TRUE(NearEqual(det, 1_r, 1e-3_r));
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
        EXPECT_TRUE(NearEqual(
            ToReal3(vel.GetOmegaAndVSym().first),
            ToReal3(velTest.GetOmegaAndVSym().first),
            1e-5_r));

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
  test(EvalTimeSteppedRotationAccurate, EvalFiniteDifferenceRotationVelocityAccurate, dt, 1e-5_r);

  dt = 3e-1_r;
  vel.SetOmega({1.1_r, -0.7_r, 0.8_r});
  vel.UpdateVSymIfDirty(dt);

  // Large time-step approximate functions. The test fails.
  test(EvalTimeSteppedRotation, EvalFiniteDifferenceRotationVelocity, dt, 3e-2_r);

  // Large time-step accurate functions
  test(EvalTimeSteppedRotationAccurate, EvalFiniteDifferenceRotationVelocityAccurate, dt, 1e-5_r);
}

TEST(RigidBodyUtils, RotateInertia) {
  auto q = Quaternion::FromRotationVector(Vec4r{0.3_r, -1.4_r, 0.5_r});
  real m = 2.1_r;
  auto matReal = RotateInertia(m, q);
  auto matVec4 = RotateInertia(Vec4r(m), q);
  auto matVMatrix3x3 = RotateInertia(VDiagonalMatrix<3>(m), q);
  EXPECT_NEAR_EQ(matVec4, matReal);
  EXPECT_NEAR_EQ(matVMatrix3x3, matReal);
  Vec4r vecM{1.8_r, 2.1_r, 1.1_r};
  matVec4 = RotateInertia(vecM, q);
  matVMatrix3x3 = RotateInertia(VDiagonalMatrix<3>(vecM), q);
  EXPECT_NEAR_EQ(matVMatrix3x3, matVec4);
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
