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

#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/utils/lie.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/rigid_body_utils.h>
#include <mochi_core/utils/rodrigues_utils.h>

#include <array>
#include <functional>
#include "mochi_core/test/mochi_test_helpers.h"

using namespace mochi;
using namespace lie;

/**************************************************************************************************
 * Unit tests for the Lie derivative functions in lie.h.
 * Each unit test is designed by programming a function and its derivative, and both receive a
 * rotation as argument. These lambda functions are used in a general test code.
 */

// Data used in the tests
namespace {
real constexpr kEps = 1e-2_r;
real constexpr kTol = 1e-2_r;
Vec4r veca{1.2_r, -0.9_r, 0.3_r};
Vec4r vecb{-0.2_r, -0.6_r, 0.8_r};
Vec4r vecc{0.7_r, 1.1_r, -0.5_r};
VMatrix3x3r mat{
    Vec4r{1.3_r, 0.8_r, -1.1_r},
    Vec4r{-0.7_r, -1.2_r, -1.4_r},
    Vec4r{-0.6_r, 0.6_r, -1.1_r}};
Quaternion rota = Quaternion::FromRotationVector(Real3{-1.2_r, 0.9_r, -0.3_r});
Quaternion rotb = Quaternion::FromRotationVector(Real3{-1.1_r, -0.7_r, 0.4_r});
std::array<Quaternion, 4> Rall = {
    Quaternion::Identity(),
    Quaternion::RotationX(0.5_r * kPI),
    Quaternion::RotationY(kPI),
    Quaternion::FromRotationVector(Real3{1.1_r, -0.5_r, -0.9_r})};
} // namespace

// Overloaded function to compute the norm of the first 3 components of a SIMD vector or the
// upper-left 3x3 portion of a SIMD matrix.
MOCHI_FORCE_INLINE real Norm3(VMatrix3x3r const& a) {
  return Norm3x3(a);
}
MOCHI_FORCE_INLINE real Norm3(Vec4r const& a) {
  return Norm<3>(a);
}

// Implementations of first-order finite differences for various types
static void FiniteDiff(Vec4r const& valp, Vec4r const& valm, int col, VMatrix3x3r& outDerivative) {
  AsMatrixView(outDerivative).Col(col) = AsColumnVectorView<3>((valp - valm) / (2_r * kEps));
}

static void FiniteDiff(Vec4r const& valp, Vec4r const& valm, int col, VMatrix4x3r& outDerivative) {
  AsMatrixView(outDerivative).Col(col) = AsColumnVectorView((valp - valm) / (2_r * kEps));
}

static void FiniteDiff(real valp, real valm, int col, Vec4r& outDerivative) {
  outDerivative = Set(outDerivative, col, (valp - valm) / (2_r * kEps));
}

static void FiniteDiff(
    VMatrix3x3r const& valp,
    VMatrix3x3r const& valm,
    int col,
    VTensor3x3x3r& outDerivative) {
  VMatrix3x3r derivative = (valp - valm) / (2_r * kEps);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      outDerivative[i][j] = Set(outDerivative[i][j], col, derivative[i][j]);
    }
  }
}

// This is a first-order finite difference for a rotation, which itself must be implemented using
// Lie algebra. The difference rotation is obtained by composition, and then it is mapped to the Lie
// algebra using the log map.
static void
FiniteDiff(VMatrix3x3r const& valp, VMatrix3x3r const& valm, int col, VMatrix3x3r& outDerivative) {
  AsMatrixView(outDerivative).Col(col) =
      AsColumnVectorView<3>(InvRodrigues(Dot3x3(valp, Transpose3x3(valm))) / (2_r * kEps));
}

// Implementations of second-order finite differences for various types
static void FiniteDiff2(real val, real valp, real valm, int col, VMatrix3x3r& outDerivative) {
  AsMatrixView(outDerivative)(col, col) = (valp + valm - 2_r * val) / (kEps * kEps);
}

static void FiniteDiff2(
    real valpp,
    real valpm,
    real valmp,
    real valmm,
    int row,
    int col,
    VMatrix3x3r& outDerivative) {
  AsMatrixView(outDerivative)(row, col) = (valpp + valmm - valpm - valmp) / (4_r * kEps * kEps);
}

// Implementations of norms
static real Norm(Vec4r const& v) {
  return Norm<3>(v);
}

static real Norm(VMatrix3x3r const& m) {
  return Norm3x3(m);
}

static real Norm(VMatrix4x3r const& m) {
  return Sqrt(HSum<3>(m[0] * m[0] + m[1] * m[1] + m[2] * m[2] + m[3] * m[3]));
}

// General test code for first derivative using finite differences
template <typename FuncVal, typename DerivativeVal>
static void TestDerivative(
    std::function<FuncVal(Quaternion const&)> func,
    std::function<DerivativeVal(Quaternion const&)> derivative) {
  for (auto const& R : Rall) {
    DerivativeVal result = derivative(R);
    DerivativeVal test = result;
    for (int i = 0; i < 3; ++i) {
      Vec4r delta = SimdBasisVector(i) * kEps;
      FuncVal valp = func(Quaternion::FromRotationVector(delta) * R);
      FuncVal valm = func(Quaternion::FromRotationVector(-delta) * R);
      FiniteDiff(valp, valm, i, test);
    }
    auto diffNorm = Norm(test - result);
    auto maxNorm = Max(Norm(test), Norm(result));
    EXPECT_NEAR(diffNorm / maxNorm, 0_r, kTol);
  }
}

// General test code for second derivative using second-order finite differences
template <typename FuncVal, typename DerivativeVal>
static void TestSecondDerivative(
    std::function<FuncVal(Quaternion const&)> func,
    std::function<DerivativeVal(Quaternion const&)> derivative) {
  for (auto const& R : Rall) {
    FuncVal val = func(R);
    DerivativeVal result = derivative(R);
    DerivativeVal test = result;
    for (int i = 0; i < 3; ++i) {
      Vec4r deltai = SimdBasisVector(i) * kEps;
      for (int j = 0; j < 3; ++j) {
        if (i == j) {
          FuncVal valp = func(Quaternion::FromRotationVector(deltai) * R);
          FuncVal valm = func(Quaternion::FromRotationVector(-deltai) * R);
          FiniteDiff2(val, valp, valm, i, test);
        } else {
          Vec4r deltaj = SimdBasisVector(j) * kEps;
          FuncVal valpp = func(Quaternion::FromRotationVector(deltai + deltaj) * R);
          FuncVal valpm = func(Quaternion::FromRotationVector(deltai - deltaj) * R);
          FuncVal valmp = func(Quaternion::FromRotationVector(-deltai + deltaj) * R);
          FuncVal valmm = func(Quaternion::FromRotationVector(-deltai - deltaj) * R);
          FiniteDiff2(valpp, valpm, valmp, valmm, i, j, test);
        }
      }
    }
    auto diffNorm = Norm(test - result);
    auto maxNorm = Max(Norm(test), Norm(result));
    EXPECT_NEAR(diffNorm / maxNorm, 0_r, kTol);
  }
}

template <typename DerivativeVal, typename SecondDerivativeVal>
static void TestSecondDerivativeMixed(
    std::function<DerivativeVal(Quaternion const&, Quaternion const&)> derivative,
    std::function<SecondDerivativeVal(Quaternion const&, Quaternion const&)>
        secondDerivativeMixed) {
  for (auto const& Ra : Rall) {
    for (auto const& Rb : Rall) {
      SecondDerivativeVal test;
      SecondDerivativeVal result = secondDerivativeMixed(Ra, Rb);
      for (int j = 0; j < 3; ++j) {
        Vec4r deltaj = SimdBasisVector(j) * kEps;
        DerivativeVal valp = derivative(Ra, Quaternion::FromRotationVector(deltaj) * Rb);
        DerivativeVal valm = derivative(Ra, Quaternion::FromRotationVector(-deltaj) * Rb);
        FiniteDiff(valp, valm, j, test);
      }
      auto diffNorm = Norm(test - result);
      auto maxNorm = Max(Norm(test), Norm(result));
      EXPECT_NEAR(diffNorm / maxNorm, 0_r, kTol);
    }
  }
}

// General test code for second derivative using finite differences of the first derivative. It
// can use an optional post-process operation on the finite-differenced result.
template <
    typename Derivative1Val,
    typename Derivative2Val,
    typename IntermediateVal = Derivative2Val>
static void TestSecondDerivative(
    std::function<Derivative1Val(Quaternion const&, Vec4r const&)> derivative1,
    std::function<Derivative2Val(Quaternion const&)> derivative2,
    std::function<Derivative2Val(IntermediateVal const&)> post = nullptr) {
  for (auto const& R : Rall) {
    Derivative2Val result = derivative2(R);
    IntermediateVal testIntermediate{};
    for (int i = 0; i < 3; ++i) {
      Vec4r delta = SimdBasisVector(i) * kEps;
      Derivative1Val valp = derivative1(R, delta);
      Derivative1Val valm = derivative1(R, -delta);
      FiniteDiff(valp, valm, i, testIntermediate);
    }
    Derivative2Val test{};
    if constexpr (std::is_same_v<IntermediateVal, Derivative2Val>) {
      test = testIntermediate;
    } else {
      test = post(testIntermediate);
    }
    auto diffNorm = Norm3(test - result);
    auto maxNorm = Max(Norm3(test), Norm3(result));
    EXPECT_NEAR(diffNorm / maxNorm, 0_r, kTol);
  }
}

TEST(Lie, DMultRotVecDRot) {
  auto func = [&](Quaternion const& R) { return R * veca; };
  auto derivative = [&](Quaternion const& R) { return DMultRotVecDRot(R * veca); };
  TestDerivative<Vec4r, VMatrix3x3r>(func, derivative);
}

TEST(Lie, DMultRotTVecDRot) {
  auto func = [&](Quaternion const& R) { return R.GetConjugate() * veca; };
  auto derivative = [&](Quaternion const& R) {
    auto rotT = ToVMatrix3x3Transpose(R);
    return DMultRotTVecDRot(rotT, DotMatVec3x3(rotT, veca));
  };
  TestDerivative<Vec4r, VMatrix3x3r>(func, derivative);
}

TEST(Lie, MultVecaTDMultRotVecbDRot) {
  auto func = [&](Quaternion const& R) { return Dot<3>(veca, R * vecb); };
  auto derivative = [&](Quaternion const& R) { return MultVecaTDMultRotVecbDRot(R * vecb, veca); };
  TestDerivative<real, Vec4r>(func, derivative);
}

TEST(Lie, DMultMatRotVecDRot) {
  auto func = [&](Quaternion const& R) {
    return DotMatVec3x3(mat, DotMatVec3x3(ToVMatrix3x3(R), veca));
  };
  auto derivative = [&](Quaternion const& R) { return DMultMatRotVecDRot(mat, R * veca); };
  TestDerivative<Vec4r, VMatrix3x3r>(func, derivative);
}

TEST(Lie, DMultMatRotTVecDRot) {
  auto func = [&](Quaternion const& R) {
    return DotMatVec3x3(mat, DotVecMat3x3(veca, ToVMatrix3x3(R)));
  };
  auto derivative = [&](Quaternion const& R) {
    return DMultMatRotTVecDRot(Dot3x3(mat, ToVMatrix3x3(R.GetConjugate())), veca);
  };
  TestDerivative<Vec4r, VMatrix3x3r>(func, derivative);
}

TEST(Lie, DMultRotaRotRotbDRot) {
  auto func = [&](Quaternion const& R) { return ToVMatrix3x3(rota * R * rotb); };
  auto derivative = [&](Quaternion const& /* R */) {
    return DMultRotaRotRotbDRot(ToVMatrix3x3(rota));
  };
  TestDerivative<VMatrix3x3r, VMatrix3x3r>(func, derivative);
}

TEST(Lie, DMultRotaRotTRotbDRot) {
  auto func = [&](Quaternion const& R) { return ToVMatrix3x3(rota * R.GetConjugate() * rotb); };
  auto derivative = [&](Quaternion const& R) {
    return DMultRotaRotTRotbDRot(ToVMatrix3x3(rota * R.GetConjugate()));
  };
  TestDerivative<VMatrix3x3r, VMatrix3x3r>(func, derivative);
}

TEST(Lie, DTrMultRotMatDRot) {
  auto func = [&](Quaternion const& R) { return Trace3x3(Dot3x3(ToVMatrix3x3(R), mat)); };
  auto derivative = [&](Quaternion const& R) {
    return DTrMultRotMatDRot(Dot3x3(ToVMatrix3x3(R), mat));
  };
  TestDerivative<real, Vec4r>(func, derivative);
}

// Test second derivative of trace by second-order finite differences
TEST(Lie, D2TrMultRotMatDRot2) {
  auto func = [&](Quaternion const& R) { return Trace3x3(Dot3x3(ToVMatrix3x3(R), mat)); };
  auto derivative = [&](Quaternion const& R) {
    return D2TrMultRotMatDRot2(Dot3x3(ToVMatrix3x3(R), mat));
  };
  TestSecondDerivative<real, VMatrix3x3r>(func, derivative);
}

// Test second derivative of trace by finite differences of the first derivative with transport
TEST(Lie, D2TrMultRotMatDRot2Transport) {
  auto derivative1 = [&](Quaternion const& R, Vec4r const& delta) {
    Quaternion deltaR = Quaternion::FromRotationVector(delta);
    auto grad = DTrMultRotMatDRot(Dot3x3(ToVMatrix3x3(deltaR * R), mat));
    // Confirm that the transport uses the fast version of DRotIncrementDRotVector
    EXPECT_LT(NormSqr<3>(delta), drotvector::kThresholdDRotIncrementDRotVector<real>);
    TransportInputOfLieJacobian(delta, AsColumnVectorView<RigidSize::kDRot>(grad).Transpose());
    return grad;
  };
  auto derivative2 = [&](Quaternion const& R) {
    return D2TrMultRotMatDRot2(Dot3x3(ToVMatrix3x3(R), mat));
  };
  TestSecondDerivative<Vec4r, VMatrix3x3r>(derivative1, derivative2);
}

TEST(Lie, D2TrMultRotaMatRotbTDRotbDRota) {
  {
    auto derivative = [&](Quaternion const& Ra, Quaternion const& Rb) {
      return DTrMultRotMatDRot(
          Dot3x3(ToVMatrix3x3(Ra), Dot3x3(mat, Transpose3x3(ToVMatrix3x3(Rb)))));
    };
    auto secondDerivativeMixed = [&](Quaternion const& Ra, Quaternion const& Rb) {
      return Transpose3x3(D2TrMultRotaMatRotbTDRotbDRota(
          Dot3x3(ToVMatrix3x3(Ra), Dot3x3(mat, Transpose3x3(ToVMatrix3x3(Rb))))));
    };
    TestSecondDerivativeMixed<Vec4r, VMatrix3x3r>(derivative, secondDerivativeMixed);
  }
}

static Vec4r ContractDim2Dim3(VTensor3x3x3r const& tensor, Vec4r const& vec2, Vec4r const& vec3) {
  // Contract first on the 3rd dimension, then on the 2nd dimension
  VMatrix3x3r temp;
  for (int i = 0; i < 3; ++i) {
    temp[i] = DotMatVec3x3(tensor[i], vec3);
  }
  return DotMatVec3x3(temp, vec2);
}

// Test second derivative of rotated vector contracted on another vector
TEST(Lie, MultVecbTD2MultRotVecaDRot2MultVecc) {
  auto derivative1 = [&](Quaternion const& R, Vec4r const& delta) {
    Quaternion deltaR = Quaternion::FromRotationVector(delta);
    auto grad = DMultRotVecDRot((deltaR * R) * veca);
    TransportInputOfLieJacobian(delta, AsMatrixView(grad));
    return grad;
  };
  auto derivative2 = [&](Quaternion const& R) {
    return MultVecbTD2MultRotVecaDRot2MultVecc(R * veca, vecb, vecc);
  };
  auto post = [&](VTensor3x3x3r const& intermediate) {
    return ContractDim2Dim3(intermediate, vecb, vecc);
  };
  TestSecondDerivative<VMatrix3x3r, Vec4r, VTensor3x3x3r>(derivative1, derivative2, post);
}

// Test second derivative of rotated vector contracted on axes
TEST(Lie, MultAxisaTD2MultRotVecDRot2MultAxisb) {
  int i = {};
  int j = {};
  auto derivative1 = [&](Quaternion const& R, Vec4r const& delta) {
    Quaternion deltaR = Quaternion::FromRotationVector(delta);
    auto grad = DMultRotVecDRot((deltaR * R) * veca);
    TransportInputOfLieJacobian(delta, AsMatrixView(grad));
    return grad;
  };
  auto derivative2 = [&](Quaternion const& R) {
    return MultAxisaTD2MultRotVecDRot2MultAxisb(R * veca, i, j);
  };
  auto post = [&](VTensor3x3x3r const& intermediate) {
    return ContractDim2Dim3(intermediate, Set(Vec4r::Zero(), i, 1_r), Set(Vec4r::Zero(), j, 1_r));
  };
  for (i = 0; i < 3; ++i) {
    for (j = 0; j < 3; ++j) {
      TestSecondDerivative<VMatrix3x3r, Vec4r, VTensor3x3x3r>(derivative1, derivative2, post);
    }
  }
}

TEST(Lie, DQuatDRot) {
  auto func = [&](Quaternion const& q) { return q.data; };
  auto derivative = [&](Quaternion const& q) { return DQuatDRot(q); };
  TestDerivative<Vec4r, VMatrix4x3r>(func, derivative);
}

// Define the test function as Rot(Quat(Rot)), not just Rot(Quat), to reuse the test infrastructure.
TEST(Lie, DRotDQuat) {
  auto func = [&](Quaternion const& q) { return ToVMatrix3x3(q); };
  auto derivative = [&](Quaternion const& q) {
    VMatrix3x4r dRotDQuat = DRotDQuat(q);
    VMatrix4x3r dQuatDRot = DQuatDRot(q);
    // Force the 4th column of dQuatDRot to be zero
    for (int i = 0; i < 4; ++i) {
      dQuatDRot[i] = Blend<0, 0, 0, 1>(dQuatDRot[i], Vec4r{});
    }
    return VMatrix3x3r{
        DotVecMat4x4(dRotDQuat[0], dQuatDRot),
        DotVecMat4x4(dRotDQuat[1], dQuatDRot),
        DotVecMat4x4(dRotDQuat[2], dQuatDRot)};
  };
  TestDerivative<VMatrix3x3r, VMatrix3x3r>(func, derivative);
}

// Wrapper of WeightedRotationDifferenceMerit
static float WeightedRotationDifferenceMeritRobust(
    VMatrix3x3f const& ra,
    VMatrix3x3f const& rb,
    VMatrix3x3f const& w) {
  return WeightedRotationDifferenceMerit(ra, rb, w);
}

// Alternative implementation to WeightedRotationDifferenceMerit that lacks numerical
// robustness. Implemented as Psi = sum(w) - tr(Ra * diag(w) * RbT).
static float WeightedRotationDifferenceMeritNonRobust(
    VMatrix3x3f const& ra,
    VMatrix3x3f const& rb,
    VMatrix3x3f const& w) {
  auto matrix = Dot3x3(Dot3x3(ra, w), Transpose3x3(rb));
  return Trace3x3(w) - Trace3x3(matrix);
}

static VMatrix3x3f ToVMatrix3x3f(VMatrix3x3d const& in) {
  VMatrix3x3f out;
  for (int i = 0; i < 3; ++i) {
    out[i] = Vec4f(Get0(in[i]), Get<1>(in[i]), Get<2>(in[i]));
  }
  return out;
}

template <typename TestFunc>
static double EvalError(TestFunc testFunc, double val) {
  auto Ra = Rodrigues(Vec4d{0.3, -1.4, 0.5});
  auto Rb = Dot3x3(Rodrigues(val * Vec4d{1.0, 0.0, 0.0}), Ra);
  auto w = VDiagonalMatrix<3>(Vec4d{1.2, 0.7, 0.3});
  auto resultGold = WeightedRotationDifferenceMerit<double>(Ra, Rb, w);
  auto resultTest =
      static_cast<double>(testFunc(ToVMatrix3x3f(Ra), ToVMatrix3x3f(Rb), ToVMatrix3x3f(w)));
  auto error = Abs(resultTest - resultGold) / Abs(resultGold);
  return error;
}

TEST(Lie, WeightedRotationDifferenceMerit) {
  // Using the double-precision implementation as gold standard, evaluate relative error of the
  // single-precision implementation for some relative rotation angles.
  // Compare with the less robust implementation of the merit function.
  EXPECT_GT(EvalError(WeightedRotationDifferenceMeritNonRobust, 1e-5), 2000.0);
  EXPECT_LT(EvalError(WeightedRotationDifferenceMeritRobust, 1e-5), 1e-3);
  EXPECT_GT(EvalError(WeightedRotationDifferenceMeritNonRobust, 1e-4), 25.0);
  EXPECT_LT(EvalError(WeightedRotationDifferenceMeritRobust, 1e-4), 1e-4);
  EXPECT_GT(EvalError(WeightedRotationDifferenceMeritNonRobust, 1e-3), 0.2);
  EXPECT_LT(EvalError(WeightedRotationDifferenceMeritRobust, 1e-3), 1e-5);
  EXPECT_GT(EvalError(WeightedRotationDifferenceMeritNonRobust, 1e-2), 2e-3);
  EXPECT_LT(EvalError(WeightedRotationDifferenceMeritRobust, 1e-2), 1e-5);
  EXPECT_GT(EvalError(WeightedRotationDifferenceMeritNonRobust, 1e-1), 5e-6);
  EXPECT_LT(EvalError(WeightedRotationDifferenceMeritRobust, 1e-1), 1e-7);
  // Close to 1rad the accuracy of both implementations becomes very similar.
  EXPECT_GT(EvalError(WeightedRotationDifferenceMeritNonRobust, 1.0), 1e-8);
  EXPECT_LT(EvalError(WeightedRotationDifferenceMeritNonRobust, 1.0), 2e-7);
  EXPECT_GT(EvalError(WeightedRotationDifferenceMeritRobust, 1.0), 1e-8);
  EXPECT_LT(EvalError(WeightedRotationDifferenceMeritRobust, 1.0), 1e-7);
}

// Test gradient of the weighted rotation difference merit function
TEST(Lie, WeightedRotationDifferenceGradient) {
  VMatrix3x3r w = VDiagonalMatrix<3>(Vec4r{1.5_r, 1.2_r, 0.8_r});
  auto func = [&](Quaternion const& R) {
    return WeightedRotationDifferenceMerit(ToVMatrix3x3(R), mat, w);
  };
  auto derivative = [&](Quaternion const& R) {
    auto meritMatrix = WeightedRotationDifferenceMatrix(ToVMatrix3x3(R), mat, w);
    return WeightedRotationDifferenceGradient(meritMatrix);
  };
  TestDerivative<real, Vec4r>(func, derivative);
}

// Test Hessian of the weighted rotation difference merit function
TEST(Lie, WeightedRotationDifferenceHessian) {
  VMatrix3x3r w = VDiagonalMatrix<3>(Vec4r{1.5_r, 1.2_r, 0.8_r});
  auto gradient = [&](Quaternion const& R, Vec4r const& delta) {
    Quaternion deltaR = Quaternion::FromRotationVector(delta);
    auto meritMatrix = WeightedRotationDifferenceMatrix(ToVMatrix3x3(deltaR * R), mat, w);
    auto grad = WeightedRotationDifferenceGradient(meritMatrix);
    TransportInputOfLieJacobian(delta, AsColumnVectorView<RigidSize::kDRot>(grad).Transpose());
    return grad;
  };
  auto hessian = [&](Quaternion const& R) {
    auto meritMatrix = WeightedRotationDifferenceMatrix(ToVMatrix3x3(R), mat, w);
    return WeightedRotationDifferenceHessian(meritMatrix);
  };
  TestSecondDerivative<Vec4r, VMatrix3x3r>(gradient, hessian);
}

// Test mixed Hessian of the weighted rotation difference merit function
TEST(Lie, WeightedRotationDifferenceHessianMixed) {
  VMatrix3x3r w = VDiagonalMatrix<3>(Vec4r{1.5_r, 1.2_r, 0.8_r});
  auto derivative = [&](Quaternion const& Ra, Quaternion const& Rb) {
    auto meritMatrix = WeightedRotationDifferenceMatrix(ToVMatrix3x3(Ra), ToVMatrix3x3(Rb), w);
    return WeightedRotationDifferenceGradient(meritMatrix);
  };
  auto secondDerivativeMixed = [&](Quaternion const& Ra, Quaternion const& Rb) {
    auto meritMatrix = WeightedRotationDifferenceMatrix(ToVMatrix3x3(Ra), ToVMatrix3x3(Rb), w);
    return Transpose3x3(WeightedRotationDifferenceHessianMixed(meritMatrix));
  };
  TestSecondDerivativeMixed<Vec4r, VMatrix3x3r>(derivative, secondDerivativeMixed);
}

// Test that the different overloads produce the same results
TEST(Lie, WeightedRotationDifferenceOverloads) {
  auto Ra = Rodrigues(Vec4r{0.3_r, -1.4_r, 0.5_r});
  auto Rb = Rodrigues(Vec4r{1.1_r, 0.6_r, -0.8_r});
  real w = 2.1_r;
  auto matReal = WeightedRotationDifferenceMatrix(Ra, Rb, w);
  auto meritReal = WeightedRotationDifferenceMerit(Ra, Rb, w);
  auto matVec4 = WeightedRotationDifferenceMatrix(Ra, Rb, Vec4r(w));
  auto meritVec4 = WeightedRotationDifferenceMerit(Ra, Rb, Vec4r(w));
  auto matVMatrix3x3 = WeightedRotationDifferenceMatrix(Ra, Rb, VDiagonalMatrix<3>(w));
  auto meritVMatrix3x3 = WeightedRotationDifferenceMerit(Ra, Rb, VDiagonalMatrix<3>(w));
  EXPECT_NEAR_EQ(matVec4, matReal);
  EXPECT_NEAR_EQ(meritVec4, meritReal);
  EXPECT_NEAR_EQ(matVMatrix3x3, matReal);
  EXPECT_NEAR_EQ(meritVMatrix3x3, meritReal);
  Vec4r vecW{1.8_r, 2.1_r, 1.1_r};
  matVec4 = WeightedRotationDifferenceMatrix(Ra, Rb, vecW);
  meritVec4 = WeightedRotationDifferenceMerit(Ra, Rb, vecW);
  matVMatrix3x3 = WeightedRotationDifferenceMatrix(Ra, Rb, VDiagonalMatrix<3>(vecW));
  meritVMatrix3x3 = WeightedRotationDifferenceMerit(Ra, Rb, VDiagonalMatrix<3>(vecW));
  EXPECT_NEAR_EQ(matVMatrix3x3, matVec4);
  EXPECT_NEAR_EQ(meritVMatrix3x3, meritVec4);
}
