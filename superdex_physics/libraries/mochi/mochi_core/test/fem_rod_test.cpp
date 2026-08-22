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

#include <gtest/gtest.h>
#include <mochi_core/element_operations/fem_rod.h>
#include <mochi_core/materials/material_params_utils.h>
#include <mochi_core/test/batched_work_test_utils.h>
#include <mochi_core/test/fem_rod_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <limits>
#include <string>

using namespace mochi;
using namespace mochi::fem;
using namespace mochi::test;

class FemRodTest : public ::testing::Test {
 protected:
  // Finite difference parameters
  static constexpr real kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-7_r : 1e-3_r;
  static constexpr real kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-2_r;

  void SetUp() override {
    // Set up common test data
    SetupTestRodStencil();
  }

  void SetupTestRodStencil() {
    // Create a simple rod stencil with 3 nodes
    // Reference positions
    _testRefNodes[0] = Real3{0.1_r, 0_r, 0_r}; // Node 0
    _testRefNodes[1] = Real3{1_r, 0_r, 0_r}; // Node 1
    _testRefNodes[2] = Real3{2.3_r, 0_r, 0_r}; // Node 2

    // Convert to SIMD Vec4r for computation
    _testX[0] = ToSimd(_testRefNodes[0]);
    _testX[1] = ToSimd(_testRefNodes[1]);
    _testX[2] = ToSimd(_testRefNodes[2]);

    // Test displacements (includes both position and twist angle for each node)
    _testDisplacements[0] = 0.1_r; // u_x node 0
    _testDisplacements[1] = 0.2_r; // u_y node 0
    _testDisplacements[2] = 0.3_r; // u_z node 0
    _testDisplacements[3] = 0.1_r; // theta node 0
    _testDisplacements[4] = 0.15_r; // u_x node 1
    _testDisplacements[5] = 0.25_r; // u_y node 1
    _testDisplacements[6] = 0.35_r; // u_z node 1
    _testDisplacements[7] = 0.15_r; // theta node 1
    _testDisplacements[8] = 0.2_r; // u_x node 2
    _testDisplacements[9] = 0.3_r; // u_y node 2
    _testDisplacements[10] = 0.4_r; // u_z node 2
    _testDisplacements[11] = 0.2_r; // theta node 2
  }

  // Test data
  NdArray<Real3, kNumRodStencilNodes> _testRefNodes{};
  NdArray<Vec4r, kNumRodStencilNodes> _testX{};
  NdArray<real, kNumRodStencilDofs> _testDisplacements{};
};

// ===== IntegratedCurvatureBinormal Tests =====

TEST_F(FemRodTest, IntegratedCurvatureBinormal_RightAngle) {
  Vec4r const e0Hat = Vec4r{1_r, 0_r, 0_r};
  Vec4r const e1Hat = Vec4r{0_r, 1_r, 0_r};

  Vec4r const result = IntegratedCurvatureBinormal(e0Hat, e1Hat);

  Vec4r const expected = Vec4r{0_r, 0_r, 2_r};
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR_EQ(result[i], expected[i]);
  }
}

TEST_F(FemRodTest, IntegratedCurvatureBinormal_Straight) {
  Vec4r const e0Hat = Normalize<3>(Vec4r{1_r, 2_r, 3_r});
  Vec4r const e1Hat = e0Hat;

  Vec4r const result = IntegratedCurvatureBinormal(e0Hat, e1Hat);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR_EQ(result[i], 0_r);
  }
}

// ===== ComputeRodElementRotationLocal Tests =====

// Test that outDRotation is the derivative of outRotation w.r.t. elementDofs
TEST_F(FemRodTest, ComputeRodElementRotationLocal_DRotationConsistency) {
  // Set up element DoFs (8 values: displacement + twist for 2 nodes)
  NdArray<real, 2 * kNumRodFields> baseElementDofs;
  for (int i = 0; i < 2 * kNumRodFields; i++) {
    baseElementDofs[i] = _testDisplacements[i];
  }

  // Set up frame axis (orthogonal to deformed edge direction)
  Vec4r const deformedEdge = Normalize<3>(
      _testX[1] + Load<Vec4r>(&baseElementDofs[kNumRodFields]) - _testX[0] -
      Load<Vec4r>(&baseElementDofs[0]));
  Vec4r frameAxis = Vec4r{0_r, 1_r, 0_r};
  // Make sure axis is orthogonal to the deformed edge
  frameAxis = Normalize<3>(frameAxis - Dot<3>(frameAxis, deformedEdge) * deformedEdge);

  // Compute rotation and its derivative
  VMatrix3x3r rotation;
  NdArray<real, 3, 3, 8> dRotation;
  ComputeRodElementRotationLocal(
      _testX[0], _testX[1], frameAxis, MakeConstSpan(baseElementDofs), rotation, &dRotation);

  // Compute finite differences of rotation to verify dRotation
  NdArray<real, 3, 3, 2 * kNumRodFields> dRotationFD{};
  // Perturbation to apply to DoFs
  NdArray<real, 2 * kNumRodFields> perturbation{};
  for (int dofIndex = 0; dofIndex < 2 * kNumRodFields; dofIndex++) {
    // Perturb each DoF
    perturbation[dofIndex] = kEps;

    // Apply the perturbation to the DoFs and frame axis consistently
    NdArray<real, 2 * kNumRodFields> perturbedElementDofs = baseElementDofs + perturbation;
    Vec4r const perturbedFrameAxis = test::ComputePerturbedFrameAxis(
        _testX[0],
        _testX[1],
        MakeConstSpan(baseElementDofs),
        MakeConstSpan(perturbation),
        frameAxis);

    VMatrix3x3r rotationPerturbed;
    ComputeRodElementRotationLocal(
        _testX[0],
        _testX[1],
        perturbedFrameAxis,
        MakeConstSpan(perturbedElementDofs),
        rotationPerturbed,
        nullptr);

    // Compute finite difference derivative for this DoF
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        dRotationFD[i][j][dofIndex] = (rotationPerturbed[i][j] - rotation[i][j]) / kEps;
      }
    }

    // Restore perturbation to zero
    perturbation[dofIndex] = 0_r;
  }

  // Compare finite differences to computed dRotation
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int dofIndex = 0; dofIndex < 2 * kNumRodFields; dofIndex++) {
        real const dRotAnalytic = dRotation[i][j][dofIndex];
        real const dRotFD = dRotationFD[i][j][dofIndex];
        EXPECT_NEAR_TOL(dRotFD, dRotAnalytic, std::max(kTol, kTol * Abs(dRotAnalytic)));
      }
    }
  }
}

// ===== TransportFrameAxis Tests =====

// Transporting along the same tangent with zero twist should be the identity.
TEST_F(FemRodTest, TransportFrameAxis_Identity) {
  Vec4r const tangent = Normalize<3>(Vec4r{1_r, 2_r, 3_r});
  Vec4r const frameAxis = Vec4r{0_r, 0_r, 1_r};

  Vec4r const result = TransportFrameAxis(tangent, tangent, 0_r, frameAxis);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR_EQ(result[i], frameAxis[i]);
  }
}

// When frameAxis is orthogonal to baseTangent, the result should be orthogonal to newTangent.
TEST_F(FemRodTest, TransportFrameAxis_OrthogonalityPreservation) {
  Vec4r const baseTangent = Vec4r{1_r, 0_r, 0_r};
  Vec4r const newTangent = Normalize<3>(Vec4r{1_r, 1_r, 0_r});
  Vec4r const frameAxis = Vec4r{0_r, 1_r, 0_r}; // orthogonal to baseTangent
  real const twist = 0.3_r;

  Vec4r const result = TransportFrameAxis(baseTangent, newTangent, twist, frameAxis);

  EXPECT_NEAR_EQ(Dot<3>(result, newTangent), 0_r);
}

// A twist of pi should negate the frame axis when it is entirely perpendicular to the tangent.
TEST_F(FemRodTest, TransportFrameAxis_TwistPi) {
  Vec4r const tangent = Vec4r{1_r, 0_r, 0_r};
  Vec4r const frameAxis = Vec4r{0_r, 1_r, 0_r}; // orthogonal to tangent

  Vec4r const result = TransportFrameAxis(tangent, tangent, kPI, frameAxis);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR_EQ(result[i], -frameAxis[i]);
  }
}

// General case: verify against composing ParallelTransportOperator and RotationAboutAxis directly.
TEST_F(FemRodTest, TransportFrameAxis_GeneralCase) {
  Vec4r const baseTangent = Vec4r{1_r, 0_r, 0_r};
  Vec4r const newTangent = Normalize<3>(Vec4r{1_r, 1_r, 1_r});
  Vec4r const frameAxis = Vec4r{0_r, 0_r, 1_r}; // orthogonal to baseTangent
  real const twist = 0.7_r;

  // Reference: compose PT(baseTangent->newTangent) and R(newTangent, twist) as matrices
  VMatrix3x3r const PT = ParallelTransportOperator(baseTangent, newTangent);
  VMatrix3x3r const R = RotationAboutAxis(newTangent, twist);
  VMatrix3x3r const combined = Dot3x3(R, PT);
  // combined * frameAxis = DotVecMat3x3(frameAxis, combined^T)
  Vec4r const expected = DotVecMat3x3(frameAxis, Transpose3x3(combined));

  Vec4r const result = TransportFrameAxis(baseTangent, newTangent, twist, frameAxis);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR_EQ(result[i], expected[i]);
  }
}

TEST(Rod, DParallelTransportedVecFiniteDifference) {
  constexpr real kFdEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-7_r : 1e-3_r;
  constexpr real kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-2_r;

  RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
    using V = BatchReal<kBS>;
    using V3 = BatchReal3<kBS>;

    auto const pack = [](NdArray<Real3, kBS> const& lanes) {
      V3 out MOCHI_NO_INIT;
      alignas(alignof(V)) real staging[V::kSize]{};
      for (int i = 0; i < 3; ++i) {
        for (int b = 0; b < kBS; ++b) {
          staging[b] = lanes[b][i];
        }
        out[i] = Load<V>(staging);
      }
      return out;
    };

    NdArray<Real3, kBS> n0Lanes;
    NdArray<Real3, kBS> nLanes;
    NdArray<Real3, kBS> vLanes;
    for (int b = 0; b < kBS; ++b) {
      real const rb = StaticCast<real>(b);
      n0Lanes[b] = Normalize(Real3{1_r + 0.1_r * rb, 2_r - 0.03_r * rb, 0.5_r + 0.07_r * rb});
      nLanes[b] = Normalize(Real3{-0.5_r + 0.04_r * rb, 1_r + 0.08_r * rb, 2_r - 0.05_r * rb});
      vLanes[b] = Real3{0.3_r + 0.02_r * rb, -0.7_r + 0.03_r * rb, 1.1_r - 0.04_r * rb};
    }

    V3 const n0 = pack(n0Lanes);
    V3 const n = pack(nLanes);
    V3 const v = pack(vLanes);
    auto const [dPT_dn0, dPT_dn] = mochi::fem::details::DParallelTransportedVec(n0, n, v);
    V3 const base = DotVecMat(v, mochi::fem::details::ParallelTransportOperator(n, n0));

    for (int j = 0; j < 3; ++j) {
      V3 n0Pert = n0;
      n0Pert[j] += V{kFdEps};
      V3 const n0Fd =
          (DotVecMat(v, mochi::fem::details::ParallelTransportOperator(n, n0Pert)) - base) /
          V{kFdEps};

      V3 nPert = n;
      nPert[j] += V{kFdEps};
      V3 const nFd =
          (DotVecMat(v, mochi::fem::details::ParallelTransportOperator(nPert, n0)) - base) /
          V{kFdEps};

      for (int b = 0; b < kBS; ++b) {
        for (int i = 0; i < 3; ++i) {
          EXPECT_NEAR_TOL(n0Fd[i][b], dPT_dn0[i][j][b], Max(kTol, kTol * Abs(n0Fd[i][b])));
          EXPECT_NEAR_TOL(nFd[i][b], dPT_dn[i][j][b], Max(kTol, kTol * Abs(nFd[i][b])));
        }
      }
    }
  });
}

// ============================================================================
// Embedded Point Jacobian Tests
// ============================================================================

class FemRodEmbeddingTest : public ::testing::Test {
 protected:
  static constexpr real kFdEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-4_r;
  static constexpr real kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-4_r : 1e-2_r;
  static constexpr int kElementDofs = 2 * fem::kNumRodFields;

  // Test configuration: 2-node element along x-axis with non-unit reference length, so that
  // invReferenceLength != 1 and the test exercises the unit-reference-tangent rescaling.
  Real3 _X0 = {0_r, 0_r, 0_r};
  Real3 _X1 = {2.5_r, 0_r, 0_r};
  Real3 _refFrameAxis = {0_r, 1_r, 0_r}; // y-axis
};

TEST_F(FemRodEmbeddingTest, ComputeEmbeddedPointElementJacobian_FiniteDifferenceConsistency) {
  // Compute the embedded point position independently of the Jacobian function under test.
  // Mirrors the production forward map in unit-reference-tangent parametrization:
  //   x_vis = mid + xi[0] · (e · invReferenceLength) + xi[1] · d + xi[2] · b.
  auto const embeddedPosition = [](Vec4r const& X0,
                                   Vec4r const& X1,
                                   Real3 const& xi,
                                   real invReferenceLength,
                                   Vec4r const& frameAxis,
                                   Span<real const> dofs) -> Vec4r {
    Vec4r const x0 = X0 + Load<Vec4r>(&dofs[0]);
    Vec4r const x1 = X1 + Load<Vec4r>(&dofs[fem::kNumRodFields]);
    Vec4r const e = x1 - x0;
    Vec4r const eHat = Normalize<3>(e);
    Vec4r const binormal = Cross3(eHat, frameAxis);
    return 0.5_r * (x0 + x1) + (xi[0] * invReferenceLength) * e + xi[1] * frameAxis +
        xi[2] * binormal;
  };

  // Test with various local coordinates and element DoFs
  std::array<Real3, 4> testXi = {{
      {0_r, 0_r, 0_r}, // At midpoint
      {0.25_r, 0_r, 0_r}, // Offset along tangent
      {0_r, 0.1_r, 0_r}, // Offset along frame axis
      {0_r, 0_r, 0.05_r}, // Offset along binormal
  }};

  // Nonzero first-node twist angles deform currentFrameAxis but do not otherwise affect the
  // embedding. The second node's twist does not influence this element's frame axis.
  std::array<NdArray<real, kElementDofs>, 3> testDofs = {{
      {0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r}, // Zero deformation
      {0.1_r, 0.05_r, -0.02_r, 0.15_r, -0.05_r, 0.1_r, 0.03_r, 0_r}, // Small deformation
      {0.5_r, -0.3_r, 0.4_r, 0.5_r, -0.4_r, 0.6_r, -0.2_r, 0_r}, // Larger deformation
  }};

  NdArray<real, kElementDofs> const zeroDofs = {0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r};

  real const invReferenceLength = 1_r / Norm(_X1 - _X0);

  for (Real3 const& xi : testXi) {
    for (auto& baseDofs : testDofs) {
      // Compute the current frame axis by transporting from reference
      Vec4r const currentFrameAxis = test::ComputePerturbedFrameAxis(
          ToSimd(_X0),
          ToSimd(_X1),
          MakeConstSpan(zeroDofs),
          MakeConstSpan(baseDofs),
          ToSimd(_refFrameAxis));

      // Compute analytic Jacobian. The test fixture uses a non-unit reference length, so the
      // analytic Jacobian and the reference embeddedPosition above must both apply the
      // invReferenceLength rescaling of xi[0] for the comparison to be meaningful.
      NdArray<real, 3, kElementDofs> analyticJac;
      ComputeEmbeddedPointElementJacobian(
          _X0,
          _X1,
          xi,
          invReferenceLength,
          ToReal3(currentFrameAxis),
          MakeConstSpan(baseDofs),
          analyticJac);

      // Compute finite difference derivatives for each DoF
      for (int j = 0; j < kElementDofs; ++j) {
        NdArray<real, kElementDofs> perturbation = {};
        perturbation[j] = kFdEps;

        // Perturbed DoFs and frame axis (plus)
        NdArray<real, kElementDofs> perturbedDofsPlus = baseDofs + perturbation;
        Vec4r const perturbedFrameAxisPlus = test::ComputePerturbedFrameAxis(
            ToSimd(_X0),
            ToSimd(_X1),
            MakeConstSpan(baseDofs),
            MakeConstSpan(perturbation),
            currentFrameAxis);

        Vec4r const positionPlus = embeddedPosition(
            ToSimd(_X0),
            ToSimd(_X1),
            xi,
            invReferenceLength,
            perturbedFrameAxisPlus,
            MakeConstSpan(perturbedDofsPlus));

        // Perturbed DoFs and frame axis (minus)
        perturbation[j] = -kFdEps;
        NdArray<real, kElementDofs> perturbedDofsMinus = baseDofs + perturbation;
        Vec4r const perturbedFrameAxisMinus = test::ComputePerturbedFrameAxis(
            ToSimd(_X0),
            ToSimd(_X1),
            MakeConstSpan(baseDofs),
            MakeConstSpan(perturbation),
            currentFrameAxis);

        Vec4r const positionMinus = embeddedPosition(
            ToSimd(_X0),
            ToSimd(_X1),
            xi,
            invReferenceLength,
            perturbedFrameAxisMinus,
            MakeConstSpan(perturbedDofsMinus));

        Vec4r const fdDeriv = (positionPlus - positionMinus) / (2_r * kFdEps);

        for (int i = 0; i < 3; ++i) {
          EXPECT_NEAR(analyticJac[i][j], fdDeriv[i], kTol)
              << "Mismatch at dPosition[" << i << "][" << j << "] for xi=(" << xi[0] << "," << xi[1]
              << "," << xi[2] << ")";
        }
      }
    }
  }
}

// The four rod kernels (gravity, inertia, axial stress, bend/twist stress) are covered by
// finite-difference consistency checks. For each kernel, the residual is checked as dE/du. For
// kernels that produce dresidual, dresidual is checked as d(residual)/du over the 12 stencil DoFs.
//   - gravity / inertia / axial are frame-independent, so the FD perturbs displacement only.
//   - bend/twist is linearized around parallel-transported material frames, so the FD perturbs the
//     displacement AND re-transports the frame axes via test::ComputePerturbedFrameAxis. A
//     fixed-frame FD would NOT match the bend/twist tangent.
//   - bend/twist dresidual is checked at rest, where the quadratic material-stiffness tangent is
//     exact.
// Boundary stencils (1 or 2 nodes) must produce zero from ops needing a complete element (masking),
// the rest state has zero elastic energy, axial PSD projection is SPD, and each lane carries
// distinct data with shuffled global indices (mixed-lane + connectivity-agnostic gather).

static constexpr real kAxialStiffness = 12.0_r;
static constexpr Real2 kFlexuralStiffness = {1.3_r, 0.7_r};
static constexpr real kTorsionalStiffness = 0.5_r;
static constexpr Real3 kGravity = {1.0_r, -2.0_r, 3.0_r};
static constexpr real kDtfi2 = 1.0_r / (0.01_r * 0.01_r);

static constexpr int kStencilStride = kNumRodStencilNodes * kNumRodFields; // 12

static constexpr real kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-7_r : 1e-3_r;
static constexpr real kResRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-2_r;
static constexpr real kResAbsTol = MOCHI_USE_DOUBLE_PRECISION ? 2e-7_r : 4e-4_r;
static constexpr real kDResRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-2_r;
static constexpr real kDResAbsTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-2_r;

namespace {
struct RodStencil {
  NdArray<Real3, kNumRodStencilNodes> X; // reference node positions
  NdArray<real, kNumRodStencilDofs> disp; // displacement DoFs (incl. twist at 3, 7, 11)
  NdArray<Real3, 2> a; // current per-edge frame axes (parallel-transported)
  NdArray<Real3, 2> A; // reference per-edge frame axes
  NdArray<real, kNumRodStencilNodes> nodalMass;
  NdArray<real, 2> rotInertia;
  NdArray<real, kNumRodFields> predTarget; // node-0 predicted target (xexp + dt*vexp)
  int nodeCount = kNumRodStencilNodes; // 1, 2, or 3 distinct nodes
};

struct RodResult {
  double energy = 0.0;
  NdArray<real, kNumRodStencilDofs> residual = {};
  NdArray<real, kNumRodStencilDofs * kNumRodStencilDofs> dresidual = {};
};

template <int kBS>
struct BatchLayout {
  DynamicArray<Real3> meshNodes;
  DynamicArray<Real3> frameAxes;
  DynamicArray<Real3> referenceAxes;
  DynamicArray<real> nodalMasses;
  DynamicArray<real> rotInertias;
  DynamicArray<int> l2gFlat;
  NdArray<int, kBS> elementIndices;
  BatchRodVector<kBS> disp;
  NdArray<BatchReal<kBS>, kNumRodFields> predTarget;
  // Stiffness-damping inputs (zero factor => disabled, matching the kernel defaults).
  real stiffnessDampingFactor = 0_r;
  BatchRodVector<kBS> stageStartDisp = {};
  DynamicArray<Real3> stageStartFrameAxes;
};
} // namespace

static Real3 NodeDisp(RodStencil const& s, int n) {
  return Real3{
      s.disp[n * kNumRodFields], s.disp[n * kNumRodFields + 1], s.disp[n * kNumRodFields + 2]};
}

// Builds reference axes (orthogonal to the reference tangent) and current axes (the reference axis
// parallel-transported to the deformed tangent and rotated by the node twist), matching the frame
// contract used by the bend/twist kernel.
static void ComputeFrames(RodStencil& s) {
  for (int e = 0; e < 2; ++e) {
    Real3 const seed = (e == 0) ? Real3{0_r, 1_r, 0_r} : Real3{0_r, 0_r, 1_r};
    Real3 const TrefHat = Normalize(s.X[e + 1] - s.X[e]);
    Real3 const tDefHat = Normalize((s.X[e + 1] + NodeDisp(s, e + 1)) - (s.X[e] + NodeDisp(s, e)));
    Real3 const refAxis = Normalize(Cross(TrefHat, seed));
    real const twistE = s.disp[e * kNumRodFields + kRodThetaDofOffset];
    Vec4r const curAxis =
        TransportFrameAxis(ToSimd(TrefHat), ToSimd(tDefHat), twistE, ToSimd(refAxis));
    s.A[e] = refAxis;
    s.a[e] = ToReal3(curAxis);
  }
}

static RodStencil MakeRandomStencil(unsigned int seed) {
  RodStencil s;
  NdArray<real, 9> const rp = MakeRandomArray<9>(seed, -0.2_r, 0.2_r);
  s.X[0] = Real3{0.0_r, 0.0_r, 0.0_r} + Real3{rp[0], rp[1], rp[2]};
  s.X[1] = Real3{1.0_r, 0.0_r, 0.0_r} + Real3{rp[3], rp[4], rp[5]};
  s.X[2] = Real3{2.0_r, 0.0_r, 0.0_r} + Real3{rp[6], rp[7], rp[8]};
  s.disp = MakeRandomArray<kNumRodStencilDofs>(seed + 1u, -0.03_r, 0.03_r);
  NdArray<real, 5> const mp = MakeRandomArray<5>(seed + 3u, 0.5_r, 2.0_r);
  s.nodalMass = {mp[0], mp[1], mp[2]};
  s.rotInertia = {mp[3], mp[4]};
  s.predTarget = MakeRandomArray<kNumRodFields>(seed + 4u, -0.1_r, 0.1_r);
  ComputeFrames(s);
  return s;
}

enum class Deformation { Random, Zero, Bent, Twisted, Combined, AxialExtension, AxialCompression };

static RodStencil MakeStencil(unsigned int seed, Deformation kind, int nodeCount) {
  RodStencil s = MakeRandomStencil(seed);
  s.nodeCount = nodeCount;
  switch (kind) {
    case Deformation::Random:
      break;
    case Deformation::Zero:
      s.disp = {};
      s.predTarget = {};
      break;
    case Deformation::Bent:
      s.disp = {};
      s.disp[kNumRodFields + 1] = 0.4_r;
      break;
    case Deformation::Twisted:
      s.disp = {};
      s.disp[kRodThetaDofOffset] = 0.5_r;
      s.disp[kNumRodFields + kRodThetaDofOffset] = -0.3_r;
      break;
    case Deformation::Combined:
      s.disp = {};
      s.disp[0] = 0.1_r;
      s.disp[1] = 0.2_r;
      s.disp[2] = 0.1_r;
      s.disp[kRodThetaDofOffset] = 0.2_r;
      s.disp[kNumRodFields] = 0.15_r;
      s.disp[kNumRodFields + 1] = 0.4_r;
      s.disp[kNumRodFields + 2] = 0.15_r;
      s.disp[kNumRodFields + kRodThetaDofOffset] = 0.4_r;
      s.disp[2 * kNumRodFields] = 0.2_r;
      s.disp[2 * kNumRodFields + 1] = 0.3_r;
      s.disp[2 * kNumRodFields + 2] = 0.2_r;
      break;
    case Deformation::AxialExtension:
      s.disp = {};
      s.disp[0] = -0.05_r;
      s.disp[kNumRodFields] = 0.1_r;
      break;
    case Deformation::AxialCompression: {
      s.disp = {};
      Real3 const referenceEdge = s.X[1] - s.X[0];
      for (int d = 0; d < 3; ++d) {
        s.disp[kNumRodFields + d] = -0.1_r * referenceEdge[d];
      }
      break;
    }
  }
  ComputeFrames(s);
  return s;
}

static constexpr std::array kDeformations = {
    Deformation::Random,
    Deformation::Zero,
    Deformation::Bent,
    Deformation::Twisted,
    Deformation::Combined,
    Deformation::AxialExtension,
    Deformation::AxialCompression};
static constexpr int kNumDeformations = static_cast<int>(kDeformations.size());
static constexpr int kNodeCounts[] = {kNumRodStencilNodes, 2, 1};

template <int kBS>
static void PackDisp(NdArray<RodStencil, kBS> const& stencils, BatchRodVector<kBS>& disp) {
  using V = BatchReal<kBS>;
  alignas(alignof(V)) real staging[V::kSize]{};
  for (int dof = 0; dof < kNumRodStencilDofs; ++dof) {
    for (int b = 0; b < kBS; ++b) {
      staging[b] = stencils[b].disp[dof];
    }
    disp[dof] = Load<V>(staging);
  }
}

template <int kBS>
static void PackPredTarget(
    NdArray<RodStencil, kBS> const& stencils,
    NdArray<BatchReal<kBS>, kNumRodFields>& predTarget) {
  using V = BatchReal<kBS>;
  alignas(alignof(V)) real staging[V::kSize]{};
  for (int d = 0; d < kNumRodFields; ++d) {
    for (int b = 0; b < kBS; ++b) {
      staging[b] = stencils[b].predTarget[d];
    }
    predTarget[d] = Load<V>(staging);
  }
}

template <int kBS>
static void ResizeLayout(BatchLayout<kBS>& layout, int numNodes, int numEdges) {
  layout.meshNodes.resize(numNodes);
  layout.nodalMasses.resize(numNodes);
  layout.frameAxes.resize(numEdges);
  layout.referenceAxes.resize(numEdges);
  layout.rotInertias.resize(numEdges);
  layout.l2gFlat.resize(kBS * kStencilStride);
}

// kBS independent stencils, each at a distinct global node block; nodes beyond nodeCount-1 collapse
// onto the last valid node so boundary stencils are masked by the kernels.
template <int kBS>
static BatchLayout<kBS> BuildIndependent(NdArray<RodStencil, kBS> const& stencils) {
  BatchLayout<kBS> layout;
  ResizeLayout(layout, kNumRodStencilNodes * kBS, kNumRodStencilNodes * kBS);
  for (int b = 0; b < kBS; ++b) {
    RodStencil const& s = stencils[b];
    int const base = b * kNumRodStencilNodes;
    for (int n = 0; n < kNumRodStencilNodes; ++n) {
      layout.meshNodes[base + n] = s.X[n];
      layout.nodalMasses[base + n] = s.nodalMass[n];
      layout.frameAxes[base + n] = (n < 2) ? s.a[n] : s.a[1];
      layout.referenceAxes[base + n] = (n < 2) ? s.A[n] : s.A[1];
      layout.rotInertias[base + n] = (n < 2) ? s.rotInertia[n] : 0.0_r;
    }
    layout.elementIndices[b] = b;
    for (int n = 0; n < kNumRodStencilNodes; ++n) {
      int const globalNode = base + Min(n, s.nodeCount - 1);
      for (int d = 0; d < kNumRodFields; ++d) {
        layout.l2gFlat[b * kStencilStride + n * kNumRodFields + d] = globalNode * kNumRodFields + d;
      }
    }
  }
  PackDisp<kBS>(stencils, layout.disp);
  PackPredTarget<kBS>(stencils, layout.predTarget);
  return layout;
}

// Full stencils whose global node indices are deliberately non-consecutive within each lane.
template <int kBS>
static BatchLayout<kBS> BuildShuffledIndependent(NdArray<RodStencil, kBS> const& stencils) {
  BatchLayout<kBS> layout;
  constexpr int kNodesPerLane = 5;
  ResizeLayout(layout, kNodesPerLane * kBS, kNodesPerLane * kBS);
  for (int b = 0; b < kBS; ++b) {
    RodStencil const& s = stencils[b];
    int const base = b * kNodesPerLane;
    NdArray<int, kNumRodStencilNodes> const nodes{base + 3, base, base + 4};
    for (int n = 0; n < kNumRodStencilNodes; ++n) {
      int const node = nodes[n];
      layout.meshNodes[node] = s.X[n];
      layout.nodalMasses[node] = s.nodalMass[n];
    }
    for (int e = 0; e < 2; ++e) {
      int const edgeNode = nodes[e];
      layout.frameAxes[edgeNode] = s.a[e];
      layout.referenceAxes[edgeNode] = s.A[e];
      layout.rotInertias[edgeNode] = s.rotInertia[e];
    }
    layout.elementIndices[b] = b;
    for (int n = 0; n < kNumRodStencilNodes; ++n) {
      for (int d = 0; d < kNumRodFields; ++d) {
        layout.l2gFlat[b * kStencilStride + n * kNumRodFields + d] = nodes[n] * kNumRodFields + d;
      }
    }
  }
  PackDisp<kBS>(stencils, layout.disp);
  PackPredTarget<kBS>(stencils, layout.predTarget);
  return layout;
}

// Private per-lane cyclic layouts exercise periodic-style wrapped gathers without sharing frame
// axes across lanes, keeping finite differences lane-local.
template <int kBS>
static BatchLayout<kBS> BuildCyclicIndependent(NdArray<RodStencil, kBS> const& stencils) {
  BatchLayout<kBS> layout;
  ResizeLayout(layout, kNumRodStencilNodes * kBS, kNumRodStencilNodes * kBS);
  for (int b = 0; b < kBS; ++b) {
    RodStencil const& s = stencils[b];
    int const base = b * kNumRodStencilNodes;
    int const shift = b % kNumRodStencilNodes;
    NdArray<int, kNumRodStencilNodes> nodes;
    for (int n = 0; n < kNumRodStencilNodes; ++n) {
      nodes[n] = base + ((shift + n) % kNumRodStencilNodes);
    }
    for (int n = 0; n < kNumRodStencilNodes; ++n) {
      int const node = nodes[n];
      layout.meshNodes[node] = s.X[n];
      layout.nodalMasses[node] = s.nodalMass[n];
      layout.rotInertias[node] = 0_r;
    }
    for (int e = 0; e < 2; ++e) {
      int const edgeNode = nodes[e];
      layout.frameAxes[edgeNode] = s.a[e];
      layout.referenceAxes[edgeNode] = s.A[e];
      layout.rotInertias[edgeNode] = s.rotInertia[e];
    }
    layout.elementIndices[b] = b;
    for (int n = 0; n < kNumRodStencilNodes; ++n) {
      for (int d = 0; d < kNumRodFields; ++d) {
        layout.l2gFlat[b * kStencilStride + n * kNumRodFields + d] = nodes[n] * kNumRodFields + d;
      }
    }
  }
  PackDisp<kBS>(stencils, layout.disp);
  PackPredTarget<kBS>(stencils, layout.predTarget);
  return layout;
}

// Open rod layout: kBS stencils over kBS nodes, with boundary stencils collapsed and per-edge
// arrays sized as production stores them (numNodes - 1).
template <int kBS>
static BatchLayout<kBS> BuildOpenRod(NdArray<RodStencil, kBS> const& stencils) {
  BatchLayout<kBS> layout;
  ResizeLayout(layout, kBS, kBS - 1);
  for (int i = 0; i < kBS; ++i) {
    layout.meshNodes[i] = Real3{StaticCast<real>(i), 0_r, 0_r};
    layout.nodalMasses[i] = stencils[i].nodalMass[0];
    layout.elementIndices[i] = i;
    for (int n = 0; n < kNumRodStencilNodes; ++n) {
      int const node = Min(i + n, kBS - 1);
      for (int d = 0; d < kNumRodFields; ++d) {
        layout.l2gFlat[i * kStencilStride + n * kNumRodFields + d] = node * kNumRodFields + d;
      }
    }
  }
  for (int e = 0; e < kBS - 1; ++e) {
    layout.frameAxes[e] = Real3{0_r, 1_r, 0_r};
    layout.referenceAxes[e] = Real3{0_r, 1_r, 0_r};
    layout.rotInertias[e] = stencils[e].rotInertia[0];
  }
  PackDisp<kBS>(stencils, layout.disp);
  PackPredTarget<kBS>(stencils, layout.predTarget);
  return layout;
}

template <int kBS>
static void ExtractLane(
    int b,
    OutputConfig cfg,
    BatchDouble<kBS> const& energy,
    BatchRodVector<kBS> const& res,
    BatchRodMatrix<kBS> const& dres,
    RodResult& out) {
  if (cfg.energy) {
    out.energy = energy[b];
  }
  if (cfg.residual) {
    for (int i = 0; i < kNumRodStencilDofs; ++i) {
      out.residual[i] = res[i][b];
    }
  }
  if (cfg.dresidual) {
    for (int i = 0; i < kNumRodStencilDofs * kNumRodStencilDofs; ++i) {
      out.dresidual[i] = dres[i][b];
    }
  }
}

template <int kBS, class Kernel>
static NdArray<RodResult, kBS> RunBatched(OutputConfig cfg, Kernel const& kernel) {
  BatchDouble<kBS> energy{0.0};
  BatchRodVector<kBS> res = {};
  BatchRodMatrix<kBS> dres = {};
  kernel(
      cfg.energy ? &energy : nullptr,
      cfg.residual ? &res : nullptr,
      cfg.dresidual ? &dres : nullptr);
  NdArray<RodResult, kBS> out{};
  for (int b = 0; b < kBS; ++b) {
    ExtractLane<kBS>(b, cfg, energy, res, dres, out[b]);
  }
  return out;
}

template <int kBS>
static bool HasAnyActiveAxialLane(BatchLayout<kBS> const& layout) {
  Span<int const> const l2g = MakeConstSpan(layout.l2gFlat);
  for (int b = 0; b < kBS; ++b) {
    int const elementIndex = layout.elementIndices[b];
    int const node0 = mochi::fem::details::RodStencilNodeIndex(l2g, elementIndex, 0);
    int const node1 = mochi::fem::details::RodStencilNodeIndex(l2g, elementIndex, 1);
    if (node0 != node1) {
      return true;
    }
  }
  return false;
}

template <int kBS>
static bool HasAnyActiveBendTwistLane(BatchLayout<kBS> const& layout) {
  Span<int const> const l2g = MakeConstSpan(layout.l2gFlat);
  for (int b = 0; b < kBS; ++b) {
    int const elementIndex = layout.elementIndices[b];
    int const node0 = mochi::fem::details::RodStencilNodeIndex(l2g, elementIndex, 0);
    int const node1 = mochi::fem::details::RodStencilNodeIndex(l2g, elementIndex, 1);
    int const node2 = mochi::fem::details::RodStencilNodeIndex(l2g, elementIndex, 2);
    if (node0 != node1 && node1 != node2 && node0 != node2) {
      return true;
    }
  }
  return false;
}

template <int kBS>
static NdArray<RodResult, kBS> RunRodGravity(BatchLayout<kBS> const& layout, OutputConfig cfg) {
  return RunBatched<kBS>(
      OutputConfig{.energy = cfg.energy, .residual = cfg.residual, .dresidual = false},
      [&](BatchDouble<kBS>* e, BatchRodVector<kBS>* r, BatchRodMatrix<kBS>* /*d*/) {
        bool const wrote = RodGravity<kBS>(
            MakeConstSpan(layout.nodalMasses),
            layout.disp,
            layout.elementIndices,
            MakeConstSpan(layout.l2gFlat),
            e,
            r,
            kGravity);
        EXPECT_EQ((e != nullptr) || (r != nullptr), wrote);
      });
}

template <int kBS>
static NdArray<RodResult, kBS> RunRodInertia(BatchLayout<kBS> const& layout, OutputConfig cfg) {
  return RunBatched<kBS>(
      cfg, [&](BatchDouble<kBS>* e, BatchRodVector<kBS>* r, BatchRodMatrix<kBS>* d) {
        bool const wrote = RodInertia<kBS>(
            MakeConstSpan(layout.nodalMasses),
            MakeConstSpan(layout.rotInertias),
            layout.disp,
            layout.predTarget,
            layout.elementIndices,
            MakeConstSpan(layout.l2gFlat),
            e,
            r,
            d,
            kDtfi2);
        EXPECT_EQ((e != nullptr) || (r != nullptr) || (d != nullptr), wrote);
      });
}

template <int kBS>
static NdArray<RodResult, kBS>
RunRodAxialStress(BatchLayout<kBS> const& layout, OutputConfig cfg, bool projectPsd) {
  return RunBatched<kBS>(
      cfg, [&](BatchDouble<kBS>* e, BatchRodVector<kBS>* r, BatchRodMatrix<kBS>* d) {
        bool const wrote = RodAxialStress<kBS>(
            MakeConstSpan(layout.meshNodes),
            layout.disp,
            layout.elementIndices,
            MakeConstSpan(layout.l2gFlat),
            e,
            r,
            d,
            kAxialStiffness,
            projectPsd,
            layout.stiffnessDampingFactor,
            layout.stiffnessDampingFactor > 0_r ? &layout.stageStartDisp : nullptr);
        bool const hasOutput = (e != nullptr) || (r != nullptr) || (d != nullptr);
        EXPECT_EQ(hasOutput && HasAnyActiveAxialLane(layout), wrote);
      });
}

template <int kBS>
static NdArray<RodResult, kBS> RunRodBendTwistStress(
    BatchLayout<kBS> const& layout,
    OutputConfig cfg) {
  return RunBatched<kBS>(
      cfg, [&](BatchDouble<kBS>* e, BatchRodVector<kBS>* r, BatchRodMatrix<kBS>* d) {
        bool const wrote = RodBendTwistStress<kBS>(
            MakeConstSpan(layout.meshNodes),
            MakeConstSpan(layout.frameAxes),
            MakeConstSpan(layout.referenceAxes),
            layout.disp,
            layout.elementIndices,
            MakeConstSpan(layout.l2gFlat),
            e,
            r,
            d,
            kFlexuralStiffness,
            kTorsionalStiffness,
            layout.stiffnessDampingFactor,
            layout.stiffnessDampingFactor > 0_r ? &layout.stageStartDisp : nullptr,
            layout.stiffnessDampingFactor > 0_r ? MakeConstSpan(layout.stageStartFrameAxes)
                                                : Span<Real3 const>{});
        bool const hasOutput = (e != nullptr) || (r != nullptr) || (d != nullptr);
        EXPECT_EQ(hasOutput && HasAnyActiveBendTwistLane(layout), wrote);
      });
}

template <size_t N>
static void ExpectNearL2(
    NdArray<real, N> const& ref,
    NdArray<real, N> const& actual,
    real relTol,
    real absTol) {
  EXPECT_LE(Norm(ref - actual), Max(absTol, relTol * Norm(ref)));
}

static void ExpectNearResult(RodResult const& ref, RodResult const& actual, OutputConfig cfg) {
  if (cfg.energy) {
    ExpectNearEnergy(ref.energy, actual.energy);
  }
  if (cfg.residual) {
    ExpectNearL2(ref.residual, actual.residual, kResRelTol, kResAbsTol);
  }
  if (cfg.dresidual) {
    ExpectNearL2(ref.dresidual, actual.dresidual, kDResRelTol, kDResAbsTol);
  }
}

static void ExpectZeroEnergyResDRes(RodResult const& actual) {
  EXPECT_EQ(0.0, actual.energy);
  EXPECT_EQ(0_r, Norm(actual.residual));
  EXPECT_EQ(0_r, Norm(actual.dresidual));
}

template <int kBS, class Run>
static void VerifyOutputModes(BatchLayout<kBS> const& layout, Run const& run) {
  auto const all = run(layout, OutputConfig{.energy = true, .residual = true, .dresidual = true});
  for (OutputConfig const cfg : kAllOutputConfigs) {
    auto const actual = run(layout, cfg);
    for (int b = 0; b < kBS; ++b) {
      ExpectNearResult(all[b], actual[b], cfg);
    }
  }
}

// FD for frame-independent ops (gravity / inertia / axial): perturb displacement only.
template <int kBS, bool kHasDRes, class Run>
static void VerifyPlainFd(BatchLayout<kBS> const& base, Run const& run) {
  using V = BatchReal<kBS>;
  constexpr int kN = kNumRodStencilDofs;
  auto const baseR =
      run(base, OutputConfig{.energy = false, .residual = true, .dresidual = kHasDRes});
  NdArray<NdArray<real, kN>, kBS> resFd{};
  NdArray<NdArray<real, kN * kN>, kBS> dresFd{};
  for (int j = 0; j < kN; ++j) {
    BatchLayout<kBS> lp = base;
    BatchLayout<kBS> lm = base;
    lp.disp[j] = lp.disp[j] + V{kEps};
    lm.disp[j] = lm.disp[j] - V{kEps};
    auto const ep = run(lp, OutputConfig{.energy = true, .residual = false, .dresidual = false});
    auto const em = run(lm, OutputConfig{.energy = true, .residual = false, .dresidual = false});
    NdArray<RodResult, kBS> rp{}, rm{};
    if constexpr (kHasDRes) {
      rp = run(lp, OutputConfig{.energy = false, .residual = true, .dresidual = false});
      rm = run(lm, OutputConfig{.energy = false, .residual = true, .dresidual = false});
    }
    for (int b = 0; b < kBS; ++b) {
      resFd[b][j] =
          static_cast<real>((ep[b].energy - em[b].energy) / (2.0 * static_cast<double>(kEps)));
      if constexpr (kHasDRes) {
        for (int i = 0; i < kN; ++i) {
          dresFd[b][i * kN + j] = (rp[b].residual[i] - rm[b].residual[i]) / (2_r * kEps);
        }
      }
    }
  }
  for (int b = 0; b < kBS; ++b) {
    ExpectNearL2(resFd[b], baseR[b].residual, kResRelTol, kResAbsTol);
    if constexpr (kHasDRes) {
      ExpectNearL2(dresFd[b], baseR[b].dresidual, kDResRelTol, kDResAbsTol);
    }
  }
}

// Re-transports a stencil's per-edge frame axes for a single-DoF displacement perturbation, writing
// the perturbed axes into the lane's layout slots.
template <int kBS>
static void PerturbFrames(
    NdArray<RodStencil, kBS> const& stencils,
    BatchLayout<kBS>& layout,
    int dof,
    real signedEps) {
  for (int b = 0; b < kBS; ++b) {
    RodStencil const& s = stencils[b];
    int const row = layout.elementIndices[b] * kStencilStride;
    for (int e = 0; e < 2; ++e) {
      NdArray<real, 2 * kNumRodFields> baseDofs{};
      for (int k = 0; k < 2 * kNumRodFields; ++k) {
        baseDofs[k] = s.disp[e * kNumRodFields + k];
      }
      NdArray<real, 2 * kNumRodFields> pert{};
      int const local = dof - e * kNumRodFields;
      if (local >= 0 && local < 2 * kNumRodFields) {
        pert[local] = signedEps;
      }
      Vec4r const aPert = test::ComputePerturbedFrameAxis(
          ToSimd(s.X[e]),
          ToSimd(s.X[e + 1]),
          MakeConstSpan(baseDofs),
          MakeConstSpan(pert),
          ToSimd(s.a[e]));
      int const globalNode = layout.l2gFlat[row + e * kNumRodFields] / kNumRodFields;
      layout.frameAxes[Min(globalNode, isize(layout.frameAxes) - 1)] = ToReal3(aPert);
    }
  }
}

template <int kBS>
static void VerifyBendTwistFd(
    NdArray<RodStencil, kBS> const& stencils,
    BatchLayout<kBS> const& base) {
  using V = BatchReal<kBS>;
  constexpr int kN = kNumRodStencilDofs;

  // Residual == d(energy)/d(disp) at the general (base) deformation.
  auto const baseR = RunRodBendTwistStress(
      base, OutputConfig{.energy = false, .residual = true, .dresidual = false});
  NdArray<NdArray<real, kN>, kBS> resFd{};
  for (int j = 0; j < kN; ++j) {
    BatchLayout<kBS> lp = base;
    BatchLayout<kBS> lm = base;
    lp.disp[j] = lp.disp[j] + V{kEps};
    lm.disp[j] = lm.disp[j] - V{kEps};
    PerturbFrames<kBS>(stencils, lp, j, kEps);
    PerturbFrames<kBS>(stencils, lm, j, -kEps);
    auto const ep = RunRodBendTwistStress(
        lp, OutputConfig{.energy = true, .residual = false, .dresidual = false});
    auto const em = RunRodBendTwistStress(
        lm, OutputConfig{.energy = true, .residual = false, .dresidual = false});
    for (int b = 0; b < kBS; ++b) {
      resFd[b][j] =
          static_cast<real>((ep[b].energy - em[b].energy) / (2.0 * static_cast<double>(kEps)));
    }
  }
  for (int b = 0; b < kBS; ++b) {
    ExpectNearL2(resFd[b], baseR[b].residual, kResRelTol, kResAbsTol);
  }

  // Dresidual == d(residual)/d(disp) at the rest state, where the bend/twist tangent is exact.
  NdArray<RodStencil, kBS> rest = stencils;
  for (int b = 0; b < kBS; ++b) {
    rest[b].disp = {};
    ComputeFrames(rest[b]);
  }
  BatchLayout<kBS> restLayout = base;
  // Preserve the layout topology under test. At rest, current material axes match reference axes.
  PackDisp<kBS>(rest, restLayout.disp);
  restLayout.frameAxes = restLayout.referenceAxes;
  // Reset the stage-start to rest so the damped tangent (scaled by 1+factor) is exact at rest.
  // No-op when damping is disabled, since these fields are unused in that case.
  restLayout.stageStartDisp = restLayout.disp;
  restLayout.stageStartFrameAxes = restLayout.referenceAxes;
  auto const restR = RunRodBendTwistStress(
      restLayout, OutputConfig{.energy = false, .residual = true, .dresidual = true});
  NdArray<NdArray<real, kN * kN>, kBS> dresFd{};
  for (int j = 0; j < kN; ++j) {
    BatchLayout<kBS> lp = restLayout;
    BatchLayout<kBS> lm = restLayout;
    lp.disp[j] = lp.disp[j] + V{kEps};
    lm.disp[j] = lm.disp[j] - V{kEps};
    PerturbFrames<kBS>(rest, lp, j, kEps);
    PerturbFrames<kBS>(rest, lm, j, -kEps);
    auto const rp = RunRodBendTwistStress(
        lp, OutputConfig{.energy = false, .residual = true, .dresidual = false});
    auto const rm = RunRodBendTwistStress(
        lm, OutputConfig{.energy = false, .residual = true, .dresidual = false});
    for (int b = 0; b < kBS; ++b) {
      for (int i = 0; i < kN; ++i) {
        dresFd[b][i * kN + j] = (rp[b].residual[i] - rm[b].residual[i]) / (2_r * kEps);
      }
    }
  }
  for (int b = 0; b < kBS; ++b) {
    ExpectNearL2(dresFd[b], restR[b].dresidual, kDResRelTol, kDResAbsTol);
  }
}

template <int kBS>
static void VerifyAllOpsFd(
    NdArray<RodStencil, kBS> const& stencils,
    BatchLayout<kBS> const& layout) {
  VerifyPlainFd<kBS, /*kHasDRes*/ false>(
      layout, [](BatchLayout<kBS> const& l, OutputConfig c) { return RunRodGravity(l, c); });
  VerifyPlainFd<kBS, true>(
      layout, [](BatchLayout<kBS> const& l, OutputConfig c) { return RunRodInertia(l, c); });
  VerifyPlainFd<kBS, true>(layout, [](BatchLayout<kBS> const& l, OutputConfig c) {
    return RunRodAxialStress(l, c, /*projectPsd*/ false);
  });
  VerifyBendTwistFd<kBS>(stencils, layout);
}

// Stiffness-proportional damping is unified into the elastic response: each material stiffness is
// scaled to (1+factor)·k and the modified strain ε̃ = ε − (factor/(1+factor))·ε_stageStart is fed
// through the same formulas as the undamped path. When factor == 0 the kernels run the existing
// undamped path. The stage-start state is held fixed during a stage, so the FD scaffolding (which
// perturbs `disp` only) carries damping through transparently.
static constexpr real kStiffnessDampingFactor = 0.5_r;

// Build a stage-start stencil that shares the reference geometry of @p current but carries a
// distinct fixed deformation, so the damping delta-strains are nonzero.
static RodStencil
MakeStageStartFor(RodStencil const& current, unsigned int seed, Deformation kind) {
  RodStencil ss = MakeStencil(seed, kind, current.nodeCount);
  ss.X = current.X;
  ComputeFrames(ss);
  return ss;
}

// Attach a stage-start state to a layout for stiffness damping. The stage-start stencils share the
// reference geometry/topology of the base layout, so a parallel BuildShuffledIndependent yields
// aligned packed displacements and (global) frame axes.
template <int kBS>
static void AttachStageStart(
    BatchLayout<kBS>& layout,
    NdArray<RodStencil, kBS> const& stageStartStencils,
    real factor) {
  BatchLayout<kBS> const ss = BuildShuffledIndependent<kBS>(stageStartStencils);
  layout.stiffnessDampingFactor = factor;
  layout.stageStartDisp = ss.disp;
  layout.stageStartFrameAxes = ss.frameAxes;
}

// Independent full stencils, distinct random deformation per lane (mixed-lane), all batch sizes.
// Displacements are modest so the second-difference (dResidual) finite differences stay
// well-conditioned; larger deformations are covered by the residual FD and the rest/masking checks.
TEST(Rod, AllOpsFd) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
    for (int trial = 0; trial < kNumDeformations; ++trial) {
      NdArray<RodStencil, kBS> stencils;
      for (int b = 0; b < kBS; ++b) {
        Deformation const kind = kDeformations[(b + trial) % kNumDeformations];
        stencils[b] = MakeStencil(static_cast<unsigned int>(1000 * trial + 31 * b + 1), kind, 3);
      }
      for (real const dampingFactor : {0_r, kStiffnessDampingFactor}) {
        SCOPED_TRACE("dampingFactor=" + std::to_string(dampingFactor));
        auto layout = BuildShuffledIndependent<kBS>(stencils);
        if (dampingFactor > 0_r) {
          NdArray<RodStencil, kBS> stageStart;
          for (int b = 0; b < kBS; ++b) {
            Deformation const ssKind = kDeformations[(b + trial + 2) % kNumDeformations];
            stageStart[b] = MakeStageStartFor(
                stencils[b], static_cast<unsigned int>(9100 + 1000 * trial + 31 * b), ssKind);
          }
          AttachStageStart<kBS>(layout, stageStart, dampingFactor);
        }
        VerifyAllOpsFd<kBS>(stencils, layout);
      }
    }
  });
}

TEST(Rod, PeriodicWrappedGatherFd) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
    NdArray<RodStencil, kBS> stencils;
    for (int b = 0; b < kBS; ++b) {
      Deformation const kind = kDeformations[b % kNumDeformations];
      stencils[b] = MakeStencil(static_cast<unsigned int>(601 + 17 * b), kind, 3);
    }
    VerifyAllOpsFd<kBS>(stencils, BuildCyclicIndependent<kBS>(stencils));
  });
}

// Boundary stencils: ops needing a complete element produce zero on incomplete lanes (masking).
TEST(Rod, BoundaryMasking) {
  // Collapsed lanes clamp L to the smallest normal real, so invL ~ 1/min. This displacement makes
  // invL * dx overflow to inf, verifying the isActive masks scrub it before it reaches the outputs.
  real constexpr kOverflowDisp = 10_r;
  static_assert(
      kOverflowDisp > std::numeric_limits<real>::min() * std::numeric_limits<real>::max(),
      "Displacement must be large enough that invL * dx overflows in collapsed lanes.");
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
    NdArray<RodStencil, kBS> stencils;
    for (int b = 0; b < kBS; ++b) {
      int const nodeCount = (kBS == 1) ? 1 : kNodeCounts[b % 3];
      stencils[b] = MakeStencil(static_cast<unsigned int>(77 + b), Deformation::Random, nodeCount);
      if (nodeCount < 2) {
        stencils[b].disp[kNumRodFields] = kOverflowDisp;
      }
    }
    auto layout = BuildIndependent<kBS>(stencils);
    OutputConfig const all{.energy = true, .residual = true, .dresidual = true};
    auto const axial = RunRodAxialStress(layout, all, false);
    auto const axialPsd = RunRodAxialStress(layout, all, true);
    auto const bend = RunRodBendTwistStress(layout, all);

    // Damping normalizes the stage-start edge by the same overflowing invL, so mirror the
    // displacement to overflow that path too. Only the axial overflow path is exercised here. The
    // bend/twist damping path additionally needs stageStartFrameAxes, which this layout leaves
    // empty, so any bend/twist run must stay above this point.
    NdArray<RodStencil, kBS> stageStart = stencils;
    for (int b = 0; b < kBS; ++b) {
      stageStart[b].disp[kNumRodFields] = -stencils[b].disp[kNumRodFields];
    }
    layout.stiffnessDampingFactor = kStiffnessDampingFactor;
    PackDisp<kBS>(stageStart, layout.stageStartDisp);
    auto const axialDamped = RunRodAxialStress(layout, all, false);
    auto const axialDampedPsd = RunRodAxialStress(layout, all, true);

    for (int b = 0; b < kBS; ++b) {
      if (stencils[b].nodeCount < 2) {
        ExpectZeroEnergyResDRes(axial[b]);
        ExpectZeroEnergyResDRes(axialPsd[b]);
        ExpectZeroEnergyResDRes(axialDamped[b]);
        ExpectZeroEnergyResDRes(axialDampedPsd[b]);
      }
      if (stencils[b].nodeCount < kNumRodStencilNodes) {
        ExpectZeroEnergyResDRes(bend[b]);
      }
    }
  });
}

TEST(Rod, OpenRodBoundarySentinels) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
    NdArray<RodStencil, kBS> stencils;
    for (int b = 0; b < kBS; ++b) {
      stencils[b] = MakeStencil(static_cast<unsigned int>(211 + b), Deformation::Random, 3);
    }
    auto const layout = BuildOpenRod<kBS>(stencils);
    OutputConfig const all{.energy = true, .residual = true, .dresidual = true};
    auto const inertia = RunRodInertia(layout, all);
    auto const axial = RunRodAxialStress(layout, all, false);
    auto const bend = RunRodBendTwistStress(layout, all);

    int constexpr kTheta0 = kRodThetaDofOffset;
    int constexpr kDResTheta0 = kTheta0 * kNumRodStencilDofs + kTheta0;
    EXPECT_EQ(0_r, inertia[kBS - 1].residual[kTheta0]);
    EXPECT_EQ(0_r, inertia[kBS - 1].dresidual[kDResTheta0]);
    ExpectZeroEnergyResDRes(axial[kBS - 1]);
    if constexpr (kBS >= 2) {
      ExpectZeroEnergyResDRes(bend[kBS - 2]);
    }
    ExpectZeroEnergyResDRes(bend[kBS - 1]);

    VerifyPlainFd<kBS, /*kHasDRes*/ false>(
        layout, [](BatchLayout<kBS> const& l, OutputConfig c) { return RunRodGravity(l, c); });
    VerifyPlainFd<kBS, true>(
        layout, [](BatchLayout<kBS> const& l, OutputConfig c) { return RunRodInertia(l, c); });
  });
}

TEST(Rod, OutputModes) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
    NdArray<RodStencil, kBS> stencils;
    for (int b = 0; b < kBS; ++b) {
      stencils[b] = MakeStencil(static_cast<unsigned int>(401 + b), Deformation::Random, 3);
    }
    auto const layout = BuildShuffledIndependent<kBS>(stencils);
    VerifyOutputModes<kBS>(
        layout, [](BatchLayout<kBS> const& l, OutputConfig c) { return RunRodGravity(l, c); });
    VerifyOutputModes<kBS>(
        layout, [](BatchLayout<kBS> const& l, OutputConfig c) { return RunRodInertia(l, c); });
    VerifyOutputModes<kBS>(layout, [](BatchLayout<kBS> const& l, OutputConfig c) {
      return RunRodAxialStress(l, c, false);
    });
    VerifyOutputModes<kBS>(layout, [](BatchLayout<kBS> const& l, OutputConfig c) {
      return RunRodBendTwistStress(l, c);
    });
  });
}

// Rest state: zero displacement => zero elastic (axial + bend/twist) energy.
TEST(Rod, RestState) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
    NdArray<RodStencil, kBS> stencils;
    for (int b = 0; b < kBS; ++b) {
      stencils[b] = MakeStencil(static_cast<unsigned int>(5 + b), Deformation::Zero, 3);
    }
    auto const layout = BuildIndependent<kBS>(stencils);
    OutputConfig const e{.energy = true, .residual = false, .dresidual = false};
    auto const axial = RunRodAxialStress(layout, e, false);
    auto const bend = RunRodBendTwistStress(layout, e);
    for (int b = 0; b < kBS; ++b) {
      EXPECT_NEAR(0.0, axial[b].energy, 1e-9);
      EXPECT_NEAR(0.0, bend[b].energy, 1e-9);
    }
  });
}

// Tangent symmetry plus v^T K v >= 0 for a handful of random vectors.
static void ExpectSymmetricPsd(RodResult const& actual, unsigned int seed) {
  int constexpr kN = kNumRodStencilDofs;
  // The projected tangent is PSD by construction, so v^T K v can only go negative through
  // rounding, which scales with the summed term magnitudes rather than the result. An absolute
  // floor would be vacuous for large tangents; a lost projection is a relative-order-one
  // violation, so it is still caught easily.
  double constexpr kPsdRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-12 : 1e-5;
  for (int i = 0; i < kN; ++i) {
    for (int j = i + 1; j < kN; ++j) {
      EXPECT_EQ(actual.dresidual[i * kN + j], actual.dresidual[j * kN + i]);
    }
  }
  for (int t = 0; t < 5; ++t) {
    auto const v = MakeRandomArray<kN>(seed + static_cast<unsigned int>(11 * t), -1_r, 1_r);
    double quad = 0.0;
    double quadScale = 0.0;
    for (int i = 0; i < kN; ++i) {
      for (int j = 0; j < kN; ++j) {
        auto const term = static_cast<double>(v[i]) *
            static_cast<double>(actual.dresidual[i * kN + j]) * static_cast<double>(v[j]);
        quad += term;
        quadScale += Abs(term);
      }
    }
    EXPECT_GE(quad, -kPsdRelTol * quadScale);
  }
}

// PSD projection => symmetric positive-semi-definite axial dresidual.
TEST(Rod, AxialPsd) {
  // Verifies the closed-form results are scale invariant: the reference length spans ~13 (float) /
  // ~103 (double) decades below unity, so the tangent reaches ~1e15 / ~1e105 while the rescaled
  // assertions below stay exact. Powers of two keep the rescaling itself free of rounding error.
  real constexpr kTinyScale = MOCHI_USE_DOUBLE_PRECISION ? 0x1p-342_r : 0x1p-43_r;
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
    int constexpr kN = kNumRodStencilDofs;
    real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-12_r : 1e-5_r;
    // Closed form for a straight rod with EA = kAxialStiffness = 12 and a stage-start displacement
    // mirrored about the rest state: effEA = (1 + f)·EA = 18, ssWeight = f/(1 + f) = 1/3.
    //   tension:     q = 1.5, qSs = 0.5 -> effStrain =  0.75,  effStress =  13.5
    //   compression: q = 0.5, qSs = 1.5 -> effStrain = -7/12,  effStress = -10.5
    // residual = effStress·q, material = effEA·q²/L, geometric = effStress/L (floored at eps).
    for (bool const tension : {false, true}) {
      real const expectedResidual = tension ? 20.25_r : -5.25_r;
      real const expectedMaterial = tension ? 40.5_r : 4.5_r;
      real const expectedGeometric = 13.5_r;
      for (real const scale : {1_r, kTinyScale}) {
        NdArray<RodStencil, kBS> stencils{};
        NdArray<RodStencil, kBS> stageStart{};
        for (int b = 0; b < kBS; ++b) {
          stencils[b].X = {Real3{}, Real3{scale, 0_r, 0_r}, Real3{2_r * scale, 0_r, 0_r}};
          stencils[b].disp[kNumRodFields] = (tension ? 0.5_r : -0.5_r) * scale;
          stageStart[b] = stencils[b];
          stageStart[b].disp[kNumRodFields] = -stencils[b].disp[kNumRodFields];
        }
        auto layout = BuildShuffledIndependent<kBS>(stencils);
        AttachStageStart<kBS>(layout, stageStart, kStiffnessDampingFactor);
        auto const psd = RunRodAxialStress(
            layout,
            OutputConfig{.energy = false, .residual = true, .dresidual = true},
            /*projectPsd*/ true);
        for (int b = 0; b < kBS; ++b) {
          EXPECT_NEAR_TOL(psd[b].residual[kNumRodFields], expectedResidual, kTol);
          real const Kxx = psd[b].dresidual[0];
          real const Kyy = psd[b].dresidual[kN + 1];
          if (tension) {
            EXPECT_NEAR_TOL(scale * Kyy, expectedGeometric, kTol);
          } else {
            // Compression gives effStress < 0, so the projection floors the geometric term.
            EXPECT_EQ(materials::kMinProjectedEigenvalue, Kyy);
          }
          EXPECT_NEAR_TOL(scale * (Kxx - Kyy), expectedMaterial, kTol);
          // The rod is x-aligned, so the 3x3 block is diagonal and transversely isotropic.
          EXPECT_EQ(Kyy, psd[b].dresidual[2 * kN + 2]);
          EXPECT_EQ(0_r, psd[b].dresidual[1]);
          EXPECT_EQ(0_r, psd[b].dresidual[2]);
          EXPECT_EQ(0_r, psd[b].dresidual[kN + 2]);
          ExpectSymmetricPsd(psd[b], static_cast<unsigned int>(b));
        }
      }
    }

    // Generic orientation: the aligned cases above have identically zero off-diagonal components,
    // so they cannot exercise the projection on a fully populated tangent.
    NdArray<RodStencil, kBS> generic;
    for (int b = 0; b < kBS; ++b) {
      generic[b] = MakeStencil(static_cast<unsigned int>(303 + b), Deformation::Random, 3);
    }
    auto const genericPsd = RunRodAxialStress(
        BuildShuffledIndependent<kBS>(generic),
        OutputConfig{.energy = false, .residual = false, .dresidual = true},
        /*projectPsd*/ true);
    for (int b = 0; b < kBS; ++b) {
      ExpectSymmetricPsd(genericPsd[b], static_cast<unsigned int>(303 + b));
    }
  });
}
